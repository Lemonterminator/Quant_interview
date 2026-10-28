"""Study 03 - who leads whom: VIX changes and next-day gold (GLD) returns.

Question: does today's change in equity implied volatility (Δln VIX_t) predict
tomorrow's gold return, and does gold say anything about tomorrow's VIX?

Timing matters more than usual here. GLD and QQQ close at 16:00 ET, but the
daily VIX close is the 16:15 ET print, so Δln VIX_t already contains 15 minutes
GLD can only price on t+1. The study therefore (a) splits GLD's next-day return
into overnight (close -> open) and intraday (open -> close) legs - only the
intraday leg is tradeable on a signal known at 16:15 - and (b) re-runs the key
test with a 16:00 VIX built from hourly bars (2024-01 onwards).

Run from the repository root::

    python strategies/gold_qqq_cash/data_pipeline.py VIX_1600   # hourly VIX, accumulates
    python strategies/gold_qqq_cash/study_03_vix_gold_lead_lag.py

Outputs go to ``artifacts/03_vix_gold_lead_lag/``; the write-up is
``reports/03_vix_gold_lead_lag.md``.
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

OUT = HERE / "artifacts" / "03_vix_gold_lead_lag"
LAGS = range(-5, 6)
NW_LAGS = 5
GRANGER_P = 5
ROLL = 750
SPIKE_Q = 0.95
SUBPERIODS = (("2004-2012", "2004", "2012"), ("2013-2019", "2013", "2019"), ("2020-2026", "2020", "2026"))
C_OVERNIGHT, C_INTRADAY = "#2a78d6", "#eb6834"


# ------------------------------------------------------------------------ data

def build_frame() -> pd.DataFrame:
    gld = dp.load("GLD")
    # GLD pays no dividends, so raw open/close and adj_close share one basis.
    assert np.allclose(gld.close, gld.adj_close, rtol=1e-6), "GLD open/close split needs a dividend-free basis"
    px = pd.concat({"gld_open": gld.open, "gld_close": gld.close, "qqq": dp.load("QQQ").adj_close,
                    "vix": dp.load("VIX").close}, axis=1, join="inner").dropna()
    df = pd.DataFrame(index=px.index)
    df["g"] = np.log(px.gld_close).diff()
    df["g_on"] = np.log(px.gld_open / px.gld_close.shift())
    df["g_id"] = np.log(px.gld_close / px.gld_open)
    df["q"] = np.log(px.qqq).diff()
    df["v"] = np.log(px.vix).diff()
    df["vix"] = px.vix
    df = df.iloc[1:]
    df["corr60"] = s01.rolling_corr(df.q.values, df.g.values, 60)
    df["state60"] = s01.classify(df.corr60, s01.critical_r(60))
    for col in ("g", "g_on", "g_id", "v", "q"):
        df[f"{col}_next"] = df[col].shift(-1)
    return df


# -------------------------------------------------------------------- analyses

def cross_correlations(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({name: su.lagged_corr(df.v, df[col], LAGS)
                        for name, col in (("GLD", "g"), ("GLD overnight", "g_on"), ("GLD intraday", "g_id"),
                                          ("QQQ", "q"))})
    out["GLD leads VIX"] = su.lagged_corr(df.g, df.v, LAGS)
    out.index.name = "k (x_t vs y_t+k)"
    return out


def predictive(df: pd.DataFrame) -> dict:
    res = {}
    specs = {
        "gld_next_on_v": ("g_next", ["v"]),
        "gld_next_on_v_controls": ("g_next", ["v", "g", "q"]),
        "gld_overnight_next": ("g_on_next", ["v", "g", "q"]),
        "gld_intraday_next": ("g_id_next", ["v", "g", "q"]),
        "vix_next_on_gld": ("v_next", ["g", "v", "q"]),
        "qqq_next_on_v": ("q_next", ["v", "q"]),
    }
    for name, (y, xs) in specs.items():
        fit = su.ols_nw(df[y].values, df[xs].values, lags=NW_LAGS)
        res[name] = {"y": y, "x": xs, "n": fit["n"], "r2": fit["r2"],
                     "beta": dict(zip(["const"] + xs, map(float, fit["beta"]))),
                     "t_nw": dict(zip(["const"] + xs, map(float, fit["t"])))}
    return res


def granger(df: pd.DataFrame) -> dict:
    """Wald tests with Newey-West covariance on 5 lags of each variable."""
    cols = {}
    for var in ("v", "g", "q"):
        for lag in range(GRANGER_P):
            cols[f"{var}_l{lag}"] = df[var].shift(lag)
    X = pd.DataFrame(cols)
    names = list(X.columns)
    out = {}
    for label, y, tested in (("VIX -> GLD", "g_next", "v"), ("GLD -> VIX", "v_next", "g"),
                             ("VIX -> GLD intraday", "g_id_next", "v"), ("VIX -> GLD overnight", "g_on_next", "v")):
        fit = su.ols_nw(df[y].values, X.values, lags=NW_LAGS)
        idx = [1 + i for i, n in enumerate(names) if n.startswith(tested + "_")]
        out[label] = su.wald(fit, idx)
    return out


def quintiles(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["v", "g_next", "g_on_next", "g_id_next"])
    bucket = pd.qcut(d.v, 5, labels=[f"Q{i}" for i in range(1, 6)])
    rows = []
    for b, sub in d.groupby(bucket, observed=True):
        row = {"quintile": b, "n": len(sub), "v_mean_pct": 100 * sub.v.mean()}
        for col, name in (("g_next", "gld"), ("g_on_next", "overnight"), ("g_id_next", "intraday")):
            x = sub[col]
            row[f"{name}_bp"] = 1e4 * x.mean()
            row[f"{name}_t"] = x.mean() / (x.std() / np.sqrt(len(x)))
        rows.append(row)
    return pd.DataFrame(rows)


def spikes(df: pd.DataFrame) -> dict:
    """Top-5% VIX up-days, split by the study-01 regime known at that close."""
    d = df.dropna(subset=["v", "g_next"])
    cut = d.v.quantile(SPIKE_Q)
    sp = d[d.v >= cut]
    g5 = df.g.rolling(5).sum().shift(-5)
    out = {"v_cutoff_pct": float(100 * cut), "by_state": {}}
    groups = {"all": sp}
    groups.update({s01.STATE_LABEL[s]: sp[sp.state60 == s] for s in (1, 0, -1)})
    for name, sub in groups.items():
        rec = {"n": len(sub), "same_day_gld_bp": float(1e4 * sub.g.mean())}
        for col, label in (("g_next", "next_gld"), ("g_on_next", "next_overnight"), ("g_id_next", "next_intraday")):
            x = sub[col].dropna()
            rec[f"{label}_bp"] = float(1e4 * x.mean())
            rec[f"{label}_t"] = float(x.mean() / (x.std() / np.sqrt(len(x)))) if len(x) > 2 else None
        rec["next_gld_hit"] = float((sub.g_next > 0).mean())
        rec["next5_gld_bp"] = float(1e4 * g5.loc[sub.index].mean())
        out["by_state"][name] = rec
    return out


def conditional_checks(df: pd.DataFrame) -> dict:
    """Is the 'big VIX drop -> gold up tomorrow' pattern just QQQ's big up-days,
    is it stable, and is the spike-day reversal more than gold's own reversal?"""
    d = df.dropna(subset=["v", "g_next", "g_on_next"])
    vq1 = (pd.qcut(d.v, 5, labels=False) == 0).astype(float)
    qq5 = (pd.qcut(d.q, 5, labels=False) == 4).astype(float)
    out = {"vix_q1_share_also_qqq_q5": float((vq1 * qq5).sum() / vq1.sum())}
    by_qqq = d.g_next.groupby(pd.qcut(d.q, 5, labels=[f"Q{i}" for i in range(1, 6)]), observed=True).mean()
    out["next_gld_bp_by_qqq_quintile"] = {k: float(1e4 * v) for k, v in by_qqq.items()}
    for name, X in (("vs_linear_qqq", pd.DataFrame({"vq1": vq1, "q": d.q, "g": d.g})),
                    ("vs_qqq_top_quintile", pd.DataFrame({"vq1": vq1, "qq5": qq5, "g": d.g}))):
        fit = su.ols_nw(d.g_next.values, X.values, lags=NW_LAGS)
        out[f"vix_q1_{name}"] = {"bp": float(1e4 * fit["beta"][1]), "t_nw": float(fit["t"][1])}
    out["vix_q1_by_subperiod"] = {}
    for name, a, b in SUBPERIODS:
        sub = d.loc[a:b]
        x = sub.g_next[pd.qcut(sub.v, 5, labels=False) == 0]
        out["vix_q1_by_subperiod"][name] = {"n": len(x), "bp": float(1e4 * x.mean()),
                                            "t": float(x.mean() / (x.std() / np.sqrt(len(x))))}
    spike = d.v >= d.v.quantile(SPIKE_Q)
    big = (d.g.abs() >= d.g.abs().quantile(0.90)) & ~spike
    for name, mask in (("spike_days", spike), ("big_gold_move_non_spike_days", big), ("all_days", d.v.notna())):
        fit = su.ols_nw(d.g_next[mask].values, d.g[mask].values, lags=0)
        out[f"reversal_slope_{name}"] = {"n": int(mask.sum()), "slope": float(fit["beta"][1]), "t": float(fit["t"][1])}
    return out


def subperiods(df: pd.DataFrame) -> dict:
    out = {}
    for name, a, b in SUBPERIODS:
        sub = df.loc[a:b]
        fit = su.ols_nw(sub.g_next.values, sub[["v", "g", "q"]].values, lags=NW_LAGS)
        out[name] = {"n": fit["n"], "corr_v_g_next": float(sub.v.corr(sub.g_next)),
                     "corr_v_g_same_day": float(sub.v.corr(sub.g)),
                     "beta_v": float(fit["beta"][1]), "t_nw_v": float(fit["t"][1])}
    return out


def vix_timing_check(df: pd.DataFrame) -> dict:
    """Split Δln VIX_t (16:15 to 16:15) into a 16:00-to-16:00 change and the
    post-close piece ln(VIX_16:15 / VIX_16:00) on day t."""
    v1600 = dp.load("VIX_1600").close.reindex(df.index)
    d = df.copy()
    d["post"] = np.log(d.vix / v1600)
    d["v1600"] = np.log(v1600).diff()
    d = d.dropna(subset=["post", "v1600", "g_next", "g_on_next", "g_id_next"])
    fit = su.ols_nw(d.g_next.values, d[["v1600", "post"]].values, lags=NW_LAGS)
    fit_on = su.ols_nw(d.g_on_next.values, d[["v1600", "post"]].values, lags=NW_LAGS)
    fit_full = su.ols_nw(d.g_next.values, d[["v"]].values, lags=NW_LAGS)
    return {
        "sample": {"start": d.index[0].date().isoformat(), "end": d.index[-1].date().isoformat(), "days": len(d)},
        "post_close_share_of_dlnvix_variance": float(d.post.var() / d.v.var()),
        "post_close_sd_pct": float(100 * d.post.std()),
        "corr_post_with_same_day_gld": float(d.post.corr(d.g)),
        "corr_post_with_same_day_qqq": float(d.post.corr(d.q)),
        "corr_v1615_g_next": float(d.v.corr(d.g_next)),
        "corr_v1600_g_next": float(d.v1600.corr(d.g_next)),
        "corr_post_g_next": float(d.post.corr(d.g_next)),
        "corr_post_g_overnight_next": float(d.post.corr(d.g_on_next)),
        "corr_post_g_intraday_next": float(d.post.corr(d.g_id_next)),
        "regression_g_next": {"beta_v1615_alone": float(fit_full["beta"][1]), "t_v1615_alone": float(fit_full["t"][1]),
                              "beta_v1600": float(fit["beta"][1]), "t_v1600": float(fit["t"][1]),
                              "beta_post": float(fit["beta"][2]), "t_post": float(fit["t"][2])},
        "regression_g_overnight_next": {"beta_post": float(fit_on["beta"][2]), "t_post": float(fit_on["t"][2])},
    }


# ----------------------------------------------------------------------- plots

def plot_ccf(ccf: pd.DataFrame, n: int, path: Path):
    band = 1.96 / np.sqrt(n)
    panels = ["GLD", "GLD overnight", "GLD intraday", "GLD leads VIX"]
    titles = {"GLD": "corr(ΔlnVIX_t, GLD_t+k)", "GLD overnight": "corr(ΔlnVIX_t, GLD overnight_t+k)",
              "GLD intraday": "corr(ΔlnVIX_t, GLD intraday_t+k)", "GLD leads VIX": "corr(GLD_t, ΔlnVIX_t+k)"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True, facecolor=s01.SURFACE)
    for ax, col in zip(axes.ravel(), panels):
        vals = ccf[col]
        colors = [s01.C_GLD if k == 1 else s01.C_NEUTRAL for k in vals.index]
        ax.axhspan(-band, band, color=s01.GRID, alpha=0.6, lw=0, zorder=0)
        ax.bar(vals.index, vals.values, color=colors, width=0.7, edgecolor=s01.SURFACE, linewidth=2, zorder=2)
        ax.axhline(0, color=s01.INK_2, lw=0.8)
        for k in (0, 1):
            v = vals.loc[k]
            ax.annotate(f"{v:+.3f}", (k, v), xytext=(0, 3 if v >= 0 else -3), textcoords="offset points",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=7, color=s01.INK)
        lim = max(0.08, 1.25 * float(np.abs(vals.drop(0)).max()), 1.2 * abs(float(vals.loc[0])))
        ax.set_ylim(-lim, lim)
        ax.set_xticks(list(LAGS))
        ax.set_title(titles[col], color=s01.INK, fontsize=10, loc="left")
        s01._style(ax)
    for ax in axes[1]:
        ax.set_xlabel("k (days); orange bar = next day", color=s01.INK_2, fontsize=9)
    fig.suptitle(f"Lead-lag correlations, grey band = ±1.96/√n (n={n})", color=s01.INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=s01.SURFACE)
    plt.close(fig)


def plot_quintiles(q: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=s01.SURFACE)
    x = np.arange(len(q))
    w = 0.36
    ax.bar(x - w / 2, q.overnight_bp, width=w, color=C_OVERNIGHT, edgecolor=s01.SURFACE, linewidth=2,
           label="next-day overnight (close → open)")
    ax.bar(x + w / 2, q.intraday_bp, width=w, color=C_INTRADAY, edgecolor=s01.SURFACE, linewidth=2,
           label="next-day intraday (open → close)")
    ax.plot(x, q.gld_bp, color=s01.INK, marker="D", ms=7, lw=0, label="next-day total")
    for xi, v in zip(x, q.gld_bp):
        ax.annotate(f"{v:+.1f}", (xi, v), xytext=(8, 0), textcoords="offset points", fontsize=8, color=s01.INK,
                    va="center")
    ax.axhline(0, color=s01.INK_2, lw=0.8)
    ax.set_xticks(x, [f"{b}\n(ΔlnVIX {m:+.1f}%)" for b, m in zip(q.quintile, q.v_mean_pct)], fontsize=8)
    ax.set_ylabel("mean GLD return (bp)", color=s01.INK_2, fontsize=9)
    ax.set_title("Next-day GLD return by today's VIX change quintile", color=s01.INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    s01._style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=s01.SURFACE)
    plt.close(fig)


def plot_rolling(df: pd.DataFrame, path: Path):
    lead = df.v.rolling(ROLL).corr(df.g_next)
    same = df.v.rolling(ROLL).corr(df.g)
    band = 1.96 / np.sqrt(ROLL)
    fig, ax = plt.subplots(figsize=(12, 3.8), facecolor=s01.SURFACE)
    ax.axhspan(-band, band, color=s01.GRID, alpha=0.6, lw=0)
    ax.plot(same.index, same, color=s01.C_NEUTRAL, lw=1.2, label="same day: corr(ΔlnVIX_t, GLD_t)")
    ax.plot(lead.index, lead, color=s01.C_GLD, lw=1.5, label="next day: corr(ΔlnVIX_t, GLD_t+1)")
    ax.axhline(0, color=s01.INK_2, lw=0.8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title(f"Rolling {ROLL}-day correlations (grey band = ±1.96/√{ROLL})", color=s01.INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    s01._style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=s01.SURFACE)
    plt.close(fig)


def plot_spikes(sp: dict, path: Path):
    order = ["all", "liquidity (+)", "neutral", "safe-haven (-)"]
    recs = [sp["by_state"][k] for k in order]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=s01.SURFACE)
    x = np.arange(len(order))
    w = 0.36
    same = [r["same_day_gld_bp"] for r in recs]
    nxt = [r["next_gld_bp"] for r in recs]
    ax.bar(x - w / 2, same, width=w, color=s01.C_NEUTRAL, edgecolor=s01.SURFACE, linewidth=2, label="spike day")
    ax.bar(x + w / 2, nxt, width=w, color=s01.C_GLD, edgecolor=s01.SURFACE, linewidth=2, label="next day")
    for xi, r in zip(x, recs):
        t = r["next_gld_t"]
        ax.annotate(f"n={r['n']}\nt={t:+.1f}" if t is not None else f"n={r['n']}", (xi + w / 2, r["next_gld_bp"]),
                    xytext=(0, 3 if r["next_gld_bp"] >= 0 else -3), textcoords="offset points", ha="center",
                    va="bottom" if r["next_gld_bp"] >= 0 else "top", fontsize=7, color=s01.INK_2)
    ax.axhline(0, color=s01.INK_2, lw=0.8)
    ax.set_xticks(x, ["all spikes", "liquidity (+)", "neutral", "safe-haven (−)"], fontsize=8)
    ax.set_ylabel("mean GLD return (bp)", color=s01.INK_2, fontsize=9)
    ax.set_title(f"GLD around VIX spikes (ΔlnVIX ≥ {sp['v_cutoff_pct']:.1f}%), by study-01 regime at the spike close",
                 color=s01.INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    s01._style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=s01.SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------ main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_frame()
    ccf = cross_correlations(df)
    pred = predictive(df)
    gr = granger(df)
    qt = quintiles(df)
    sp = spikes(df)
    sub = subperiods(df)
    cond = conditional_checks(df)
    timing = vix_timing_check(df)
    n = int(df[["v", "g_next"]].dropna().shape[0])

    manifest = dp._load_manifest()
    results = {
        "sample": {"start": df.index[0].date().isoformat(), "end": df.index[-1].date().isoformat(), "days": len(df)},
        "data": {k: {f: manifest.get(k, {}).get(f) for f in ("last", "fetched_at_ny")}
                 for k in ("GLD", "QQQ", "VIX", "VIX_1600")},
        "iid_band_95": float(1.96 / np.sqrt(n)),
        "cross_correlations": {col: {str(k): float(v) for k, v in ccf[col].items()} for col in ccf.columns},
        "predictive_regressions": pred,
        "granger_wald": gr,
        "quintiles": qt.to_dict(orient="records"),
        "spikes": sp,
        "subperiods": sub,
        "conditional_checks": cond,
        "vix_timing_check": timing,
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float) + "\n",
                                      encoding="utf-8")
    ccf.to_csv(OUT / "cross_correlations.csv", float_format="%.4f")
    qt.to_csv(OUT / "quintiles.csv", index=False, float_format="%.4f")

    plot_ccf(ccf, n, OUT / "lead_lag.png")
    plot_quintiles(qt, OUT / "quintiles.png")
    plot_rolling(df, OUT / "rolling_lead.png")
    plot_spikes(sp, OUT / "spikes_by_regime.png")

    pd.set_option("display.width", 200)
    print(ccf.round(4).to_string())
    print(json.dumps({k: results[k] for k in ("sample", "iid_band_95", "predictive_regressions", "granger_wald",
                                              "spikes", "subperiods", "conditional_checks",
                                              "vix_timing_check")}, indent=1, default=float))
    print(qt.round(2).to_string())


if __name__ == "__main__":
    main()
