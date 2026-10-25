"""Minimal daily data pipeline for the cash / gold / QQQ sub-strategy.

Fetches from Yahoo Finance (yfinance) and FRED, normalises every series to a
tz-naive *local trading date* index, validates it, caches raw CSVs under
``data/raw/`` (git-ignored) and records a committed ``data/manifest.json`` so a
later run can tell whether the vendor silently rewrote history.

Lessons behind each step are written up in ``docs/data_pipeline_lessons.md``.

Usage (from the repository root)::

    python strategies/gold_qqq_cash/data_pipeline.py            # fetch missing / stale
    python strategies/gold_qqq_cash/data_pipeline.py --refresh  # re-download everything
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"
MANIFEST = HERE / "data" / "manifest.json"
NY = ZoneInfo("America/New_York")
# US cash equities close 16:00 ET; Yahoo usually finalises the daily bar a bit later.
SESSION_FINAL_NY = (16, 30)
MAX_AGE = timedelta(hours=20)
# Yahoo's adj_close differs by ~1e-6 (relative) between identical calls, so
# revisions are detected with a tolerance, never with an exact hash. Prices are
# compared in log-return space (blind to the uniform rescale every dividend
# causes); rates in absolute percentage points (they can be 0 or negative).
REVISION_TOL = 1e-4
RATE_REVISION_TOL = 1e-3


@dataclass(frozen=True)
class Series:
    name: str
    source: str  # "yahoo" or "fred"
    symbol: str
    role: str
    kind: str = "price"  # "price" (tradeable), "level" (index/vol level) or "rate" (percent)


UNIVERSE: dict[str, Series] = {s.name: s for s in (
    Series("QQQ", "yahoo", "QQQ", "risk asset (Nasdaq-100 ETF), since 1999-03"),
    Series("GLD", "yahoo", "GLD", "gold leg (SPDR Gold Shares), since 2004-11; closes 16:00 ET like QQQ"),
    Series("IAU", "yahoo", "IAU", "gold cross-check (iShares Gold Trust)"),
    Series("GC_F", "yahoo", "GC=F", "COMEX front-month gold; diagnostic only (settles 13:30 ET, roll jumps)"),
    Series("VIX", "yahoo", "^VIX", "equity implied vol, conditioning variable", "level"),
    Series("DXY", "yahoo", "DX-Y.NYB", "US dollar index, conditioning variable", "level"),
    Series("BIL", "yahoo", "BIL", "1-3M T-bill ETF, cash-leg proxy"),
    Series("DFII10", "fred", "DFII10", "10Y TIPS real yield (%), conditioning variable", "rate"),
    Series("DTB3", "fred", "DTB3", "3M T-bill discount rate (%), cash-leg yield", "rate"),
)}


# --------------------------------------------------------------------------- utils

def _permanent(exc: Exception) -> bool:
    """HTTP 4xx (bad id, forbidden) will not heal on retry; 429 is rate limiting."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status is not None and 400 <= status < 500 and status != 429


def _retry(fn, *, tries: int = 4, base: float = 2.0):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # yfinance raises a zoo of types (rate limit, JSON, curl)
            if attempt == tries - 1 or _permanent(exc):
                raise
            time.sleep(base * 2 ** attempt)


def _atomic_write(path: Path, write) -> None:
    """Write via a temp file + rename so an interrupted run never leaves a
    truncated file that the mtime-based freshness check would then trust."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _cache_path(series: Series) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", series.name)
    return RAW_DIR / series.source / f"{safe}.csv"


def _anchors(values: pd.Series, n: int = 6) -> dict[str, float]:
    """A few evenly spaced (date, value) fingerprints of a series."""
    picks = values.iloc[np.linspace(0, len(values) - 1, n).astype(int)]
    return {d.date().isoformat(): float(f"{v:.8g}") for d, v in picks.items()}


def _revisions(new: pd.Series, old: pd.Series, kind: str) -> pd.Series:
    """Per-date revision of ``new`` against ``old`` on their common dates.

    Prices/levels: change in log return between consecutive common dates, so a
    uniform rescale (new dividend or split) reads as zero while a patched print
    does not. Rates: absolute change in level.
    """
    common = new.index.intersection(old.index)
    a, b = new.loc[common], old.loc[common]
    if kind == "rate":
        return (a - b).abs()
    return (np.log(a).diff() - np.log(b).diff()).abs().dropna()


def revision_check(new: pd.Series, kind: str, *, old: pd.Series | None = None,
                   anchors: dict[str, float] | None = None) -> dict:
    """Compare a fresh download with the previous cache (every date) or, on a
    fresh clone, with the committed manifest anchors (a few dates)."""
    basis = "previous_cache" if old is not None else "anchors" if anchors else None
    if basis == "anchors":
        old = pd.Series(anchors).set_axis(pd.DatetimeIndex(list(anchors)))
    if basis is None:
        return {"revision_basis": None, "history_revised": False}
    rev = _revisions(new, old, kind)
    tol = RATE_REVISION_TOL if kind == "rate" else REVISION_TOL
    bad = rev[rev > tol]
    common = new.index.intersection(old.index)
    rescale = None
    if kind != "rate" and len(common):
        rescale = float(np.median(new.loc[common] / old.loc[common]) - 1)
    return {
        "revision_basis": basis,
        "compared_dates": len(common),
        "max_revision": float(rev.max()) if len(rev) else None,
        "revised_dates": [d.date().isoformat() for d in bad.index[:10]],
        "rescale_vs_previous": rescale,
        "history_revised": bool(len(bad)),
    }


# ----------------------------------------------------------------------- fetchers

def fetch_yahoo_raw(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    def call():
        # auto_adjust=False keeps both Close (split-adjusted only) and Adj Close
        # (split + dividend adjusted); the yfinance default flipped to True in 0.2.51.
        df = yf.Ticker(symbol).history(period="max", auto_adjust=False, actions=True)
        if df is None or df.empty:
            raise RuntimeError(f"yahoo returned no rows for {symbol}")
        return df

    return _retry(call)


def normalize_yahoo(raw: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Return a tz-naive, de-duplicated, sorted daily frame with snake_case columns.

    The index keeps the exchange-local calendar date. Converting to UTC first
    would shift Asian listings by a day; ``tz_localize(None)`` keeps wall time.
    A bar for *today* is dropped until the session is final.
    """
    df = raw.copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize().rename("date")
    df.columns = [re.sub(r"\W+", "_", str(c)).strip("_").lower() for c in df.columns]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["close"])
    if "adj_close" not in df:
        df["adj_close"] = df["close"]

    # A bar dated after today (futures evening session) is never final; today's
    # bar is final only after the cash close.
    now_ny = (now or datetime.now(NY)).astimezone(NY)
    today = pd.Timestamp(now_ny.date())
    final = (now_ny.hour, now_ny.minute) >= SESSION_FINAL_NY
    return df[df.index <= today] if final else df[df.index < today]


def fetch_fred_raw(series_id: str) -> str:
    import requests

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    def call():
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        if not resp.text.strip() or "<html" in resp.text[:200].lower():
            raise RuntimeError(f"FRED returned no CSV for {series_id}")
        return resp.text

    return _retry(call)


def normalize_fred(text: str, series_id: str) -> pd.DataFrame:
    """Parse fredgraph CSV. The date column was ``DATE`` before 2024 and is
    ``observation_date`` now; missing values were historically ``.``."""
    df = pd.read_csv(io.StringIO(text))
    date_col = next(c for c in df.columns if c.lower() in ("observation_date", "date"))
    out = pd.DataFrame({"close": pd.to_numeric(df[series_id], errors="coerce")})
    out.index = pd.DatetimeIndex(pd.to_datetime(df[date_col]), name="date")
    out = out[~out.index.duplicated(keep="last")].sort_index().dropna()
    out["adj_close"] = out["close"]
    return out


# --------------------------------------------------------------------- validation

def validate(df: pd.DataFrame, name: str, *, kind: str = "price") -> dict:
    """Hard errors raise; soft issues are returned for the manifest and a human."""
    if not df.index.is_monotonic_increasing or df.index.has_duplicates:
        raise ValueError(f"{name}: index not strictly increasing")
    if kind != "rate" and (df["close"] <= 0).any():
        raise ValueError(f"{name}: non-positive prices")

    gaps = df.index.to_series().diff().dt.days
    big_gaps = gaps[gaps > 7]
    close = df["close"]
    same = close.diff().eq(0)
    run_id = (~same).cumsum()
    stale_runs = same.groupby(run_id).sum()
    issues: dict = {
        "gaps_over_7d": [d.date().isoformat() for d in big_gaps.index[:10]],
        "longest_stale_run": int(stale_runs.max()) if len(stale_runs) else 0,
    }
    if kind == "price":
        ret = np.log(df["adj_close"]).diff()
        jumps = ret[ret.abs() > 0.15]
        issues["abs_log_return_over_15pct"] = {d.date().isoformat(): round(float(v), 4) for d, v in jumps.items()}
    if kind == "price" and "volume" in df:
        issues["zero_volume_days"] = int((df["volume"] == 0).sum())
    return issues


# --------------------------------------------------------------------- cache/update

def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}


def _write_manifest(manifest: dict) -> None:
    text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    _atomic_write(MANIFEST, lambda p: p.write_text(text, encoding="utf-8"))


def update(names: list[str] | None = None, *, refresh: bool = False, now: datetime | None = None) -> dict:
    manifest = _load_manifest()
    now = now or datetime.now(NY)
    for name in names or list(UNIVERSE):
        s = UNIVERSE[name]
        path = _cache_path(s)
        if path.exists() and not refresh:
            age = now.timestamp() - path.stat().st_mtime
            if age < MAX_AGE.total_seconds():
                continue
        if s.source == "yahoo":
            df = normalize_yahoo(fetch_yahoo_raw(s.symbol), now=now)
        else:
            df = normalize_fred(fetch_fred_raw(s.symbol), s.symbol)
        issues = validate(df, name, kind=s.kind)

        prev = manifest.get(name, {})
        # Vendors patch bad prints; every new dividend also rescales all earlier
        # adj_close values, which revision_check deliberately ignores.
        old = pd.read_csv(path, index_col="date", parse_dates=["date"])["adj_close"] if path.exists() else None
        revision = revision_check(df["adj_close"], s.kind, old=old, anchors=prev.get("anchors"))
        _atomic_write(path, lambda p: df.to_csv(p, float_format="%.8g"))
        manifest[name] = {
            "source": s.source, "symbol": s.symbol, "role": s.role,
            "rows": len(df), "first": df.index[0].date().isoformat(), "last": df.index[-1].date().isoformat(),
            "anchors": _anchors(df["adj_close"]),
            "revision": revision,
            "fetched_at_ny": now.astimezone(NY).isoformat(timespec="seconds"),
            "issues": issues,
        }
        # Persist per series so a later failure cannot leave cache and manifest out of step.
        _write_manifest(manifest)
        flag = "  REVISED " + ",".join(revision["revised_dates"]) if revision["history_revised"] else ""
        print(f"{name:7s} {s.symbol:9s} rows={len(df):6d} {manifest[name]['first']} -> {manifest[name]['last']}{flag}")
    return manifest


def load(name: str, *, auto_fetch: bool = True) -> pd.DataFrame:
    path = _cache_path(UNIVERSE[name])
    if not path.exists():
        if not auto_fetch:
            raise FileNotFoundError(path)
        update([name])
    return pd.read_csv(path, index_col="date", parse_dates=["date"])


def aligned_prices(names: list[str], field: str = "adj_close") -> pd.DataFrame:
    """Inner-join tradeable prices on common dates. Join *before* differencing so
    every asset's return spans the same interval."""
    frame = pd.concat({n: load(n)[field] for n in names}, axis=1, join="inner")
    return frame.dropna()


def asof_align(s: pd.Series, index: pd.DatetimeIndex, max_stale_days: int = 5) -> pd.Series:
    """Last value of ``s`` dated on or before each date of ``index``, if it is at
    most ``max_stale_days`` calendar days old (row counts depend on the calendar)."""
    s = s.dropna()
    left = pd.DataFrame({"date": pd.DatetimeIndex(index).astype("datetime64[ns]")})
    right = pd.DataFrame({"date": s.index.astype("datetime64[ns]"), "value": s.to_numpy()})
    out = pd.merge_asof(left, right, on="date", direction="backward", tolerance=pd.Timedelta(days=max_stale_days))
    return pd.Series(out["value"].to_numpy(), index=index, name=s.name)


def asof_on(index: pd.DatetimeIndex, name: str, field: str = "close", max_stale_days: int = 5) -> pd.Series:
    """Align a conditioning series (different calendar, e.g. FRED) onto ``index``
    using only values known on or before each date."""
    return asof_align(load(name)[field].rename(name), index, max_stale_days)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help=f"subset of {list(UNIVERSE)}")
    parser.add_argument("--refresh", action="store_true", help="ignore cache age and re-download")
    args = parser.parse_args()
    update(args.names or None, refresh=args.refresh)
