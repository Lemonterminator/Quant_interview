"""Study 02 - momentum and mean reversion in the GLD/QQQ price ratio.

The ratio x_t = ln(GLD_t / QQQ_t) is the relative price of the two risky legs;
its daily change is r_gld - r_qqq. Questions: does the ratio trend (time-series
momentum) or revert, at which horizons, is any of it stable across the sample,
and is it large enough to steer how new DCA money is split between gold and QQQ?

Run from the repository root::

    python strategies/gold_qqq_cash/study_02_ratio_momentum_reversion.py

Outputs go to ``artifacts/02_ratio_momentum_reversion/``; the write-up is
``reports/02_ratio_momentum_reversion.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_pipeline as dp  # noqa: E402
import stats_utils as su  # noqa: E402
import study_01_rolling_correlation as s01  # noqa: E402

OUT = HERE / "artifacts" / "02_ratio_momentum_reversion"
SEED = 20261027
N_SIM = 2000
TRADING_DAYS = 252
LOOKBACKS = (20, 60, 120, 250)          # ~1, 3, 6, 12 months
HORIZONS = (20, 60, 120, 250)
VR_LAGS = (2, 5, 10, 20, 60, 120, 250)
Z_WINDOW = 250
MONTHS = (1, 3, 6, 12)
SPLIT = "2015-12-31"


# ------------------------------------------------------------------------ data

def build_frame() -> pd.DataFrame:
    px = dp.aligned_prices(["QQQ", "GLD"])
    df = pd.DataFrame({"x": np.log(px.GLD / px.QQQ)})
    df["d"] = df.x.diff()
    roll = df.x.rolling(Z_WINDOW)
    df["z"] = (df.x - roll.mean()) / roll.std()
    df = df.iloc[1:]
    r = np.log(px).diff().iloc[1:]  # rolling_corr uses cumulative sums: no leading NaN allowed
    df["corr60"] = s01.rolling_corr(r.QQQ.values, r.GLD.values, 60)
    df["state60"] = s01.classify(df.corr60, s01.critical_r(60))
    return df


# -------------------------------------------------------------------- analyses

def variance_ratios(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, sub in (("full", df), ("2004-2015", df.loc[:SPLIT]), ("2016-2026", df.loc[SPLIT:])):
        for q in VR_LAGS:
            rows.append({"period": period, **su.variance_ratio(sub.x.values, q)})
    return pd.DataFrame(rows)


def predictive_grid(df: pd.DataFrame) -> pd.DataFrame:
    """fwd h-day ratio change on past L-day change, daily overlapping data with
    Newey-West(h) errors, plus a non-overlapping (every h days) OLS cross-check."""
    rows = []
    x = df.x
    for L in LOOKBACKS:
        past = x - x.shift(L)
        for h in HORIZONS:
            fwd = x.shift(-h) - x
            # Overlap makes fwd an MA(h-1); NW with h lags covers it (and still
            # over-rejects at long h, hence the non-overlapping check below).
            fit = su.ols_nw(fwd.values, past.values, lags=h)
            pair = pd.concat([fwd, past], axis=1).dropna().iloc[::h]
            fit_no = su.ols_nw(pair.iloc[:, 0].values, pair.iloc[:, 1].values, lags=0)
            up = fwd[past > 0].mean()
            dn = fwd[past < 0].mean()
            rows.append({"lookback": L, "horizon": h, "beta": fit["beta"][1], "t_nw": fit["t"][1], "r2": fit["r2"],
                         "t_nonoverlap": fit_no["t"][1], "n_nonoverlap": fit_no["n"],
                         "fwd_after_up_pct": 100 * up, "fwd_after_down_pct": 100 * dn,
                         "up_minus_down_pct": 100 * (up - dn)})
    return pd.DataFrame(rows)


def zscore_buckets(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, -2, -1, 1, 2, np.inf]
    labels = ["z ≤ −2", "−2 < z ≤ −1", "|z| < 1", "1 ≤ z < 2", "z ≥ 2"]
    b = pd.cut(df.z, bins=bins, labels=labels, right=False)
    rows = []
    for h in (20, 60, 250):
        fwd = df.x.shift(-h) - df.x
        drift = fwd.mean()
        for lab in labels:
            sel = fwd[b == lab].dropna()
            inside = (b == lab) & fwd.notna()
            # Overlapping windows make "days" overstate the evidence; count distinct visits.
            episodes = int((inside & ~inside.shift(fill_value=False)).sum())
            rows.append({"horizon": h, "bucket": lab, "days": len(sel), "episodes": episodes,
                         "fwd_mean_pct": 100 * sel.mean() if len(sel) else np.nan,
                         "fwd_excess_pct": 100 * (sel.mean() - drift) if len(sel) else np.nan,
                         "share_positive": float((sel > 0).mean()) if len(sel) else np.nan})
    return pd.DataFrame(rows)


def monthly_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Month-end signals and the next month's ratio return (what a DCA tilt
    decided at month end would earn). Positive = favour gold over QQQ."""
    m = df[["x", "z", "state60"]].resample("ME").last()
    m = m[m.index <= df.index[-1]]  # the running month is not a month-end yet
    m["d_next"] = m.x.shift(-1) - m.x
    for k in MONTHS:
        m[f"mom{k}"] = np.sign(m.x - m.x.shift(k))
    m["rev_z"] = np.where(m.z >= 1, -1.0, np.where(m.z <= -1, 1.0, 0.0))
    m.loc[m.z.isna(), "rev_z"] = np.nan
    return m.dropna(subset=["d_next"])


def evaluate(m: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The ratio fell ~97 log-% over the sample, so a signal that is mostly
    "favour QQQ" earns that drift without any timing skill. P&L is split into
    drift (mean position x mean return) and timing (the rest); the circular
    shift keeps the position mix, so its p-value speaks to timing only."""
    rows = []
    mu = m.d_next.mean()
    for sig in [f"mom{k}" for k in MONTHS] + ["rev_z"]:
        for period, sub in (("full", m), ("2004-2015", m.loc[:SPLIT]), ("2016-2026", m.loc[SPLIT:])):
            s = sub[[sig, "d_next"]].dropna()
            pnl = s[sig] * s.d_next
            active = s[sig] != 0
            fit = su.ols_nw(pnl.values, np.empty((len(pnl), 0)), lags=3)
            row = {"signal": sig, "period": period, "months": len(pnl), "active_share": float(active.mean()),
                   "ann_mean_pct": 1200 * pnl.mean(), "ann_vol_pct": 100 * np.sqrt(12) * pnl.std(),
                   "sharpe": float(np.sqrt(12) * pnl.mean() / pnl.std()), "t_nw": float(fit["t"][0]),
                   "hit_rate": float((pnl[active] > 0).mean()),
                   "ann_drift_pct": 1200 * s[sig].mean() * mu,
                   "ann_timing_pct": 1200 * (s[sig] * (s.d_next - mu)).mean()}
            if period == "full":
                p, _ = su.circular_shift_pvalue(s[sig].values, s.d_next.values,
                                                lambda a, b: float((a * b).mean() / (a * b).std()), N_SIM, rng, 12)
                row["p_circular_shift"] = p
            rows.append(row)
    return pd.DataFrame(rows)


def daily_reversal_sources(df: pd.DataFrame) -> dict:
    """Split the lag-1 autocovariance of d = g - q into its four pieces:
    acov(g) + acov(q) - cov(g_t, q_t+1) - cov(q_t, g_t+1)."""
    px = dp.aligned_prices(["QQQ", "GLD"])
    r = np.log(px).diff().dropna()
    g, q = r.GLD, r.QQQ

    def lagcov(a, b):
        return float((a * b.shift(-1)).mean() - a.mean() * b.mean())

    parts = {"acov_gld": lagcov(g, g), "acov_qqq": lagcov(q, q),
             "minus_cov_gld_t_qqq_t1": -lagcov(g, q), "minus_cov_qqq_t_gld_t1": -lagcov(q, g)}
    total = sum(parts.values())
    out = {"ratio_acf1": float((g - q).autocorr()), "share": {k: v / total for k, v in parts.items()}}
    for period, sl in (("2004-2015", slice(None, SPLIT)), ("2016-2026", slice(SPLIT, None))):
        out[period] = {"ratio_acf1": float((g - q).loc[sl].autocorr()), "qqq_acf1": float(q.loc[sl].autocorr()),
                       "gld_acf1": float(g.loc[sl].autocorr())}
    return out


def ratio_vol_by_state(df: pd.DataFrame) -> dict:
    """Mechanical link to study 01: var(d) = var(g) + var(q) - 2 cov(g, q)."""
    out = {}
    for s, label in s01.STATE_LABEL.items():
        d = df.d[df.state60 == s]
        out[label] = {"days": len(d), "ann_vol_pct": float(100 * d.std() * np.sqrt(TRADING_DAYS))}
    return out


def current_reading(df: pd.DataFrame, m: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    out = {"date": df.index[-1].date().isoformat(), "z250": float(last.z),
           "ratio_level": float(np.exp(last.x))}
    for L in LOOKBACKS:
        out[f"past_{L}d_pct"] = float(100 * (df.x.iloc[-1] - df.x.iloc[-1 - L]))
    out["last_scored_month"] = m.index[-1].date().isoformat()  # last month-end whose next month is complete
    return out


# ----------------------------------------------------------------------- plots

def plot_ratio(df: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]},
                             facecolor=s01.SURFACE)
    ax = axes[0]
    ratio = np.exp(df.x)
    ax.plot(df.index, ratio, color=s01.C_GLD, lw=1.3, label="GLD / QQQ (adjusted)")
    ax.plot(df.index, np.exp(df.x.rolling(Z_WINDOW).mean()), color=s01.INK_2, lw=1, ls="--", label="250d mean")
    ax.set_yscale("log")
    ticks = [0.4, 0.6, 1, 1.5, 2, 3, 4]
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylabel("price ratio (log scale)", color=s01.INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax = axes[1]
    ax.plot(df.index, df.z, color=s01.INK, lw=0.9)
    for v, ls in ((0, "-"), (1, ":"), (-1, ":"), (2, "--"), (-2, "--")):
        ax.axhline(v, color=s01.INK_2, lw=0.7, ls=ls)
    ax.set_ylabel("z-score vs 250d mean/std", color=s01.INK_2, fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for a in axes:
        s01._style(a)
    fig.suptitle("Gold / QQQ price ratio", color=s01.INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=s01.SURFACE)
    plt.close(fig)


def plot_vr(vr: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=s01.SURFACE)
    styles = {"full": (s01.INK, "-", "o"), "2004-2015": (s01.C_QQQ, "--", "s"), "2016-2026": (s01.C_GLD, "--", "^")}
    for period, (c, ls, mk) in styles.items():
        sub = vr[vr.period == period]
        ax.plot(sub.q, sub.vr, color=c, ls=ls, marker=mk, ms=6, lw=1.4, label=period)
        if period == "full":
            se = (sub.vr - 1) / sub.z_robust
            ax.fill_between(sub.q, 1 - 1.96 * se, 1 + 1.96 * se, color=s01.C_NEUTRAL, alpha=0.25, lw=0,
                            label="95% band around 1 (full sample)")
    ax.axhline(1, color=s01.INK_2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(VR_LAGS)
    ax.set_xticklabels([str(q) for q in VR_LAGS])
    ax.set_xlabel("holding period q (trading days)", color=s01.INK_2, fontsize=9)
    ax.set_ylabel("variance ratio VR(q)", color=s01.INK_2, fontsize=9)
    ax.set_title(">1 trending, <1 mean-reverting", color=s01.INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    s01._style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=s01.SURFACE)
    plt.close(fig)


def plot_grid(grid: pd.DataFrame, path: Path):
    t = grid.pivot(index="lookback", columns="horizon", values="t_nw")
    fig, ax = plt.subplots(figsize=(6.5, 4.6), facecolor=s01.SURFACE)
    lim = max(3.0, float(np.nanmax(np.abs(t.values))))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [s01.C_NEG, "#f0efec", s01.C_POS])
    im = ax.imshow(t.values, cmap=cmap, vmin=-lim, vmax=lim)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            ax.text(j, i, f"{t.values[i, j]:+.2f}", ha="center", va="center", fontsize=9, color=s01.INK)
    ax.set_xticks(range(t.shape[1]), [f"{h}d" for h in t.columns])
    ax.set_yticks(range(t.shape[0]), [f"{L}d" for L in t.index])
    ax.set_xlabel("forward horizon h", color=s01.INK_2, fontsize=9)
    ax.set_ylabel("lookback L", color=s01.INK_2, fontsize=9)
    ax.set_title("Newey-West t of β: fwd ratio change on past change\n(red = momentum, blue = reversal)",
                 color=s01.INK, fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.tick_params(colors=s01.INK_2, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=s01.SURFACE)
    plt.close(fig)


def plot_buckets(zb: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False, facecolor=s01.SURFACE)
    for ax, h in zip(axes, (20, 60, 250)):
        sub = zb[zb.horizon == h]
        vals = sub.fwd_excess_pct
        colors = [s01.C_POS if v > 0 else s01.C_NEG for v in vals.fillna(0)]
        ax.bar(range(len(sub)), vals, color=colors, edgecolor=s01.SURFACE, linewidth=2)
        span = float(np.nanmax(np.abs(vals)))
        ax.set_ylim(-1.45 * span, 1.45 * span)
        for i, (v, n) in enumerate(zip(vals, sub.episodes)):
            if np.isfinite(v):
                ax.annotate(f"{v:+.1f}%\n{n} ep.", (i, v), xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                            ha="center", va="bottom" if v >= 0 else "top", fontsize=7, color=s01.INK_2)
        ax.axhline(0, color=s01.INK_2, lw=0.8)
        ax.set_xticks(range(len(sub)), sub.bucket, rotation=30, fontsize=7)
        ax.set_title(f"next {h}d, excess over drift", color=s01.INK, fontsize=10, loc="left")
        s01._style(ax)
    fig.suptitle("Forward GLD/QQQ ratio change by current z-score (minus the unconditional mean)",
                 color=s01.INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=s01.SURFACE)
    plt.close(fig)


def plot_tsmom(m: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=s01.SURFACE)
    series = (("mom12", "12-month momentum", "#2a78d6"), ("mom3", "3-month momentum", "#eb6834"),
              ("rev_z", "z-score reversal (|z| ≥ 1)", "#1baf7a"))
    for sig, label, c in series:
        pnl = (m[sig] * m.d_next).dropna()
        ax.plot(pnl.index, 100 * pnl.cumsum(), color=c, lw=1.4, label=label)
        ax.annotate(label.split(" ")[0], (pnl.index[-1], 100 * pnl.cumsum().iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=s01.INK, va="center")
    ax.axhline(0, color=s01.INK_2, lw=0.8)
    ax.set_ylabel("cumulative gold-minus-QQQ log return (%)", color=s01.INK_2, fontsize=9)
    ax.set_title("Monthly long/short tilt on the ratio, signal at month end", color=s01.INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    s01._style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=s01.SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------ main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = build_frame()

    vr = variance_ratios(df)
    grid = predictive_grid(df)
    zb = zscore_buckets(df)
    m = monthly_signals(df)
    ev = evaluate(m, rng)
    adf_full = su.adf(df.x.values)
    adf_split = {p: su.adf(sub.x.values) for p, sub in (("2004-2015", df.loc[:SPLIT]), ("2016-2026", df.loc[SPLIT:]))}

    manifest = dp._load_manifest()
    results = {
        "sample": {"start": df.index[0].date().isoformat(), "end": df.index[-1].date().isoformat(), "days": len(df)},
        "data": {n: {k: manifest.get(n, {}).get(k) for k in ("last", "fetched_at_ny")} for n in ("QQQ", "GLD")},
        "ratio": {"total_change_pct": float(100 * (df.x.iloc[-1] - df.x.iloc[0])),
                  "ann_vol_pct": float(100 * df.d.std() * np.sqrt(TRADING_DAYS)),
                  "daily_autocorr_lag1": float(df.d.autocorr(1))},
        "stationarity": {"adf_full": adf_full, "adf_split": adf_split, "ar1": su.ar1_half_life(df.x.values)},
        "variance_ratio": vr.to_dict(orient="records"),
        "predictive_grid": grid.to_dict(orient="records"),
        "zscore_buckets": zb.to_dict(orient="records"),
        "monthly_signals": ev.to_dict(orient="records"),
        "daily_reversal_sources": daily_reversal_sources(df),
        "ratio_vol_by_state60": ratio_vol_by_state(df),
        "current": current_reading(df, m),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float) + "\n",
                                      encoding="utf-8")
    vr.to_csv(OUT / "variance_ratio.csv", index=False, float_format="%.4f")
    grid.to_csv(OUT / "predictive_grid.csv", index=False, float_format="%.4f")
    zb.to_csv(OUT / "zscore_buckets.csv", index=False, float_format="%.4f")
    ev.to_csv(OUT / "monthly_signals.csv", index=False, float_format="%.4f")

    plot_ratio(df, OUT / "ratio_overview.png")
    plot_vr(vr, OUT / "variance_ratio.png")
    plot_grid(grid, OUT / "predictive_tstats.png")
    plot_buckets(zb, OUT / "zscore_buckets.png")
    plot_tsmom(m, OUT / "monthly_tilt.png")

    pd.set_option("display.width", 200)
    print(json.dumps({k: results[k] for k in ("sample", "ratio", "stationarity", "daily_reversal_sources",
                                              "ratio_vol_by_state60", "current")},
                     indent=1, default=float))
    print(vr.pivot(index="q", columns="period", values=["vr", "z_robust"]).round(3).to_string())
    print(grid.round(3).to_string())
    print(zb.round(3).to_string())
    print(ev.round(3).to_string())


if __name__ == "__main__":
    main()
