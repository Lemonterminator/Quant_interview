"""Study 01 - GLD/QQQ rolling correlation (20d / 60d) and its sign flips.

Hypothesis under test: positive gold-equity correlation = liquidity-driven
regime, negative = safe-haven regime. The script measures the correlation,
separates real regime changes from sampling noise, checks the label against
macro context (VIX, real yields, dollar), and asks whether the current regime
says anything about the next quarter (hedge value, diversification).

Run from the repository root::

    python strategies/gold_qqq_cash/data_pipeline.py      # once, or to refresh
    python strategies/gold_qqq_cash/study_01_rolling_correlation.py

Outputs go to ``artifacts/01_rolling_correlation/``; the write-up is
``reports/01_rolling_correlation.md``.
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

OUT = HERE / "artifacts" / "01_rolling_correlation"
WINDOWS = (20, 60)
N_SIM = 300
SEED = 20261025
TRADING_DAYS = 252

# Palette: reference dataviz instance (categorical slots 1-2, diverging blue<->red).
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
C_QQQ, C_GLD = "#2a78d6", "#eb6834"
C_ROBUST = "#6250d6"
C_POS, C_NEG, C_NEUTRAL = "#e34948", "#2a78d6", "#9e9d98"
STATE_LABEL = {1: "liquidity (+)", 0: "neutral", -1: "safe-haven (-)"}


# ------------------------------------------------------------------- statistics

def critical_r(n: int, z: float = 1.96) -> float:
    """Two-sided 95% bound for a sample correlation under rho=0 (Fisher z)."""
    return float(np.tanh(z / np.sqrt(n - 3)))


def rolling_corr(x: np.ndarray, y: np.ndarray, w: int) -> np.ndarray:
    """Rolling Pearson correlation along axis 0; works on (T,) or (T, k) arrays."""
    def roll_sum(a):
        c = np.cumsum(a, axis=0)
        out = np.full_like(c, np.nan)
        out[w - 1] = c[w - 1]
        out[w:] = c[w:] - c[:-w]
        return out

    sx, sy = roll_sum(x), roll_sum(y)
    sxx, syy, sxy = roll_sum(x * x), roll_sum(y * y), roll_sum(x * y)
    cov = sxy - sx * sy / w
    var_x, var_y = sxx - sx * sx / w, syy - sy * sy / w
    return cov / np.sqrt(var_x * var_y)


def robust_returns(r: pd.Series, lam: float = 0.94, clip: float = 3.0) -> pd.Series:
    """Standardise by trailing EWMA vol (known at t-1) and clip, so one tariff-pause
    style day cannot own a 60-day Pearson correlation and create a ghost flip
    when it leaves the window."""
    vol = np.sqrt((r ** 2).ewm(alpha=1 - lam, min_periods=20).mean().shift(1))
    return (r / vol).clip(-clip, clip)


def sign_flips_per_year(c: np.ndarray) -> np.ndarray:
    s = np.sign(c)
    valid = ~np.isnan(c)
    flips = ((s[1:] != s[:-1]) & valid[1:] & valid[:-1]).sum(axis=0)
    return flips / (valid.sum(axis=0) / TRADING_DAYS)


def classify(c: pd.Series, crit: float) -> pd.Series:
    state = pd.Series(0, index=c.index, dtype=float)
    state[c >= crit] = 1
    state[c <= -crit] = -1
    return state.where(c.notna())


def significant_flips(state: np.ndarray) -> np.ndarray:
    """Count +1 <-> -1 transitions, ignoring neutral spells in between."""
    state = np.atleast_2d(state.T).T
    counts = []
    for col in state.T:
        nz = col[(col != 0) & ~np.isnan(col)]
        counts.append(int((nz[1:] != nz[:-1]).sum()))
    return np.array(counts)


def episodes(state: pd.Series) -> pd.DataFrame:
    s = state.dropna()
    run = (s != s.shift()).cumsum()
    rows = []
    for _, block in s.groupby(run):
        if block.iloc[0] == 0:
            continue
        rows.append({"state": int(block.iloc[0]), "start": block.index[0], "end": block.index[-1], "days": len(block)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ study

def build_frame() -> pd.DataFrame:
    px = dp.aligned_prices(["QQQ", "GLD"])
    r = np.log(px).diff().dropna()
    df = pd.DataFrame({"r_qqq": r["QQQ"], "r_gld": r["GLD"]})
    for w in WINDOWS:
        df[f"corr{w}"] = rolling_corr(df.r_qqq.values, df.r_gld.values, w)
        df[f"state{w}"] = classify(df[f"corr{w}"], critical_r(w))
    zq, zg = robust_returns(df.r_qqq), robust_returns(df.r_gld)
    df["corr60_robust"] = zq.rolling(60).corr(zg)
    df["state60_robust"] = classify(df.corr60_robust, critical_r(60))
    df["vix"] = dp.asof_on(df.index, "VIX")
    df["real_yield"] = dp.asof_on(df.index, "DFII10")
    df["dxy"] = dp.asof_on(df.index, "DXY")
    return df


def null_models(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Two nulls: (A) Gaussian with constant full-sample rho; (B) circular shift of
    GLD against QQQ, which keeps fat tails and vol clustering but has rho ~ 0."""
    x, y = df.r_qqq.values, df.r_gld.values
    t = len(x)
    rho = float(np.corrcoef(x, y)[0, 1])
    z1 = rng.standard_normal((t, N_SIM))
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal((t, N_SIM))
    shifts = rng.integers(TRADING_DAYS, t - TRADING_DAYS, N_SIM)
    yb = np.column_stack([np.roll(y, k) for k in shifts])
    xb = np.repeat(x[:, None], N_SIM, axis=1)

    out = {"full_sample_rho": rho}
    for w in WINDOWS:
        crit = critical_r(w)
        obs = df[f"corr{w}"].values
        res = {"critical_r": crit, "observed": {
            "sign_flips_per_year": float(sign_flips_per_year(obs[:, None])[0]),
            "share_abs_above_crit": float(np.nanmean(np.abs(obs) >= crit)),
            "significant_flips": int(significant_flips(df[f"state{w}"].values[:, None])[0]),
            "std": float(np.nanstd(obs)),
        }}
        for label, (a, b) in {"gaussian_const_rho": (z1, z2), "circular_shift": (xb, yb)}.items():
            c = rolling_corr(a, b, w)
            st = np.where(np.isnan(c), np.nan, np.where(c >= crit, 1, np.where(c <= -crit, -1, 0)))
            flips = sign_flips_per_year(c)
            share = np.nanmean(np.abs(c) >= crit, axis=0)
            sig = significant_flips(st)
            res[label] = {
                "sign_flips_per_year": [float(np.mean(flips)), float(np.percentile(flips, 5)), float(np.percentile(flips, 95))],
                "share_abs_above_crit": [float(np.mean(share)), float(np.percentile(share, 5)), float(np.percentile(share, 95))],
                "significant_flips": [float(np.mean(sig)), float(np.percentile(sig, 5)), float(np.percentile(sig, 95))],
                "std": float(np.nanstd(c)),
            }
            if label == "gaussian_const_rho" and w == 60:
                out["_null_sample_corr60"] = c[:, :20].ravel()
            if label == "gaussian_const_rho" and w == 20:
                out["_null_sample_corr20"] = c[:, :20].ravel()
        out[f"w{w}"] = res
    return out


def regime_context(df: pd.DataFrame, w: int = 60) -> pd.DataFrame:
    """Contemporaneous macro context of each window, grouped by regime."""
    ctx = pd.DataFrame(index=df.index)
    ctx["state"] = df[f"state{w}"]
    ctx["qqq_ret"] = df.r_qqq.rolling(w).sum()
    ctx["gld_ret"] = df.r_gld.rolling(w).sum()
    ctx["vix_mean"] = df.vix.rolling(w).mean()
    ctx["d_real_yield_bp"] = df.real_yield.diff(w) * 100
    ctx["d_dxy"] = np.log(df.dxy).diff(w)
    ctx["both_up"] = (ctx.qqq_ret > 0) & (ctx.gld_ret > 0)
    ctx["both_down"] = (ctx.qqq_ret < 0) & (ctx.gld_ret < 0)
    ctx["qqq_down_gld_up"] = (ctx.qqq_ret < 0) & (ctx.gld_ret > 0)
    ctx = ctx.dropna(subset=["state", "vix_mean", "d_real_yield_bp", "d_dxy"])
    agg = ctx.groupby("state").agg(
        windows=("qqq_ret", "size"),
        qqq_ret_med=("qqq_ret", "median"), gld_ret_med=("gld_ret", "median"),
        vix_mean_med=("vix_mean", "median"),
        d_real_yield_bp_med=("d_real_yield_bp", "median"), abs_d_real_yield_bp_med=("d_real_yield_bp", lambda s: s.abs().median()),
        d_dxy_med=("d_dxy", "median"),
        share_both_up=("both_up", "mean"), share_both_down=("both_down", "mean"),
        share_qqq_down_gld_up=("qqq_down_gld_up", "mean"),
    )
    return agg.rename(index=STATE_LABEL.get)


def factor_exposure(df: pd.DataFrame, w: int = 60) -> pd.DataFrame:
    """Rolling correlation of each leg with daily real-yield and dollar changes.
    If positive GLD/QQQ correlation is a common macro factor, both legs should
    load on it with the same sign in the + regime."""
    d_ry = df.real_yield.diff()
    d_dxy = np.log(df.dxy).diff()
    f = pd.DataFrame({"state": df[f"state{w}"]})
    for leg in ("qqq", "gld"):
        r = df[f"r_{leg}"]
        f[f"{leg}_vs_real_yield"] = r.rolling(w).corr(d_ry)
        f[f"{leg}_vs_dxy"] = r.rolling(w).corr(d_dxy)
    f["same_sign_real_yield"] = np.sign(f.qqq_vs_real_yield) == np.sign(f.gld_vs_real_yield)
    f["same_sign_dxy"] = np.sign(f.qqq_vs_dxy) == np.sign(f.gld_vs_dxy)
    f = f.dropna()
    return f.groupby("state").mean().rename(index=STATE_LABEL.get)


def ex_ante(df: pd.DataFrame) -> dict:
    """Use only information up to t; evaluate on t+1 onward."""
    res: dict = {}
    for w in WINDOWS:
        c = df[f"corr{w}"]
        fwd = c.shift(-w)  # correlation of the next, non-overlapping w-day window
        pair = pd.concat([c, fwd], axis=1).dropna()
        stepped = pair.iloc[::w]
        st_now = df[f"state{w}"]
        st_fwd = st_now.shift(-w)
        trans = pd.crosstab(st_now.iloc[::w].rename("now"), st_fwd.iloc[::w].rename("next"), normalize="index")
        res[f"w{w}"] = {
            "corr_now_vs_next_all": float(pair.corr().iloc[0, 1]),
            "corr_now_vs_next_nonoverlap": float(stepped.corr().iloc[0, 1]),
            "n_nonoverlap": len(stepped),
            "transition_nonoverlap": {STATE_LABEL[int(i)]: {STATE_LABEL[int(j)]: round(float(v), 3) for j, v in row.items()}
                                      for i, row in trans.iterrows()},
        }
        res[f"_pairs{w}"] = stepped

    # Tail days: QQQ in its worst 5%; regime known the day before.
    prior = df.state60.shift(1)
    cut = df.r_qqq.quantile(0.05)
    tail = df[df.r_qqq <= cut].assign(prior=prior)
    rows = {"all": tail}
    rows.update({STATE_LABEL[s]: tail[tail.prior == s] for s in (1, 0, -1)})
    res["tail_days"] = {
        "qqq_cutoff": float(cut),
        "by_prior_state": {k: {"n": len(v), "gld_mean_bp": float(v.r_gld.mean() * 1e4),
                               "gld_t_stat": float(v.r_gld.mean() / (v.r_gld.std() / np.sqrt(len(v)))),
                               "gld_hit_rate": float((v.r_gld > 0).mean()),
                               "qqq_mean_bp": float(v.r_qqq.mean() * 1e4)} for k, v in rows.items()},
    }

    # Next 60 days: 50/50 daily-rebalanced diversification and hedge value.
    w = 60
    port = 0.5 * df.r_qqq + 0.5 * df.r_gld
    fwd = pd.DataFrame({
        "state": df.state60,
        "corr_next": df.corr60.shift(-w),
        "qqq_next": df.r_qqq.rolling(w).sum().shift(-w),
        "gld_next": df.r_gld.rolling(w).sum().shift(-w),
        "div_ratio_next": (port.rolling(w).std() / (0.5 * df.r_qqq.rolling(w).std() + 0.5 * df.r_gld.rolling(w).std())).shift(-w),
    }).dropna()
    fwd = fwd.iloc[::w]
    by = {}
    for s in (1, 0, -1):
        f = fwd[fwd.state == s]
        down = f[f.qqq_next < 0]
        by[STATE_LABEL[s]] = {
            "n": len(f), "corr_next_mean": float(f.corr_next.mean()),
            "div_ratio_next_mean": float(f.div_ratio_next.mean()),
            "n_qqq_down": len(down),
            "gld_next_when_qqq_down_mean": float(down.gld_next.mean()) if len(down) else None,
        }
    res["next_60d_nonoverlap"] = by
    return res


def robustness(df: pd.DataFrame) -> dict:
    out: dict = {}
    # Same gold exposure through another ETF.
    px = dp.aligned_prices(["QQQ", "GLD", "IAU"])
    r = np.log(px).diff().dropna()
    c_gld = r.QQQ.rolling(60).corr(r.GLD)
    c_iau = r.QQQ.rolling(60).corr(r.IAU)
    out["iau_vs_gld_corr60_max_abs_diff"] = float((c_gld - c_iau).abs().max())
    out["iau_vs_gld_corr60_mean_abs_diff"] = float((c_gld - c_iau).abs().mean())

    # Futures settle at 13:30 ET, 2.5h before QQQ's close -> asynchronous returns.
    px = dp.aligned_prices(["QQQ", "GLD", "GC_F"])
    r = np.log(px).diff().dropna().loc["2005":]
    out["lead_lag_corr_qqq_t_vs_gold_t_plus_k"] = {
        g: {str(k): round(float(r.QQQ.corr(r[g].shift(-k))), 4) for k in (-1, 0, 1)} for g in ("GLD", "GC_F")
    }
    c_fut = r.QQQ.rolling(60).corr(r.GC_F)
    c_etf = r.QQQ.rolling(60).corr(r.GLD)
    out["gcf_vs_gld_corr60_mean_abs_diff"] = float((c_fut - c_etf).abs().mean())
    out["gcf_vs_gld_corr60_mean_diff"] = float((c_fut - c_etf).mean())

    # Weekly (Friday) returns largely neutralise any close-time mismatch.
    wk = np.log(dp.aligned_prices(["QQQ", "GLD"]).resample("W-FRI").last()).diff().dropna()
    c13 = wk.QQQ.rolling(13).corr(wk.GLD)
    daily_on_fri = df.corr60.resample("W-FRI").last().reindex(c13.index)
    both = pd.concat([c13, daily_on_fri], axis=1).dropna()
    out["weekly13_vs_daily60"] = {
        "corr": float(both.corr().iloc[0, 1]),
        "sign_agreement": float((np.sign(both.iloc[:, 0]) == np.sign(both.iloc[:, 1])).mean()),
        "full_sample_weekly_corr": float(wk.QQQ.corr(wk.GLD)),
        "full_sample_daily_corr": float(df.r_qqq.corr(df.r_gld)),
    }
    return out


def outlier_sensitivity(df: pd.DataFrame) -> dict:
    both = df[["corr60", "corr60_robust", "state60", "state60_robust"]].dropna()
    jumps = both.corr60.diff().abs()
    return {
        "corr_raw_vs_robust": float(both.corr60.corr(both.corr60_robust)),
        "mean_abs_diff": float((both.corr60 - both.corr60_robust).abs().mean()),
        "state_agreement": float((both.state60 == both.state60_robust).mean()),
        "opposite_state_share": float((both.state60 * both.state60_robust == -1).mean()),
        "max_1d_jump_raw": float(jumps.max()), "max_1d_jump_raw_date": jumps.idxmax().date().isoformat(),
        "max_1d_jump_robust": float(both.corr60_robust.diff().abs().max()),
        "p99_1d_jump_raw": float(jumps.quantile(0.99)),
        "p99_1d_jump_robust": float(both.corr60_robust.diff().abs().quantile(0.99)),
        "significant_flips_robust": int(significant_flips(both.state60_robust.values[:, None])[0]),
    }


def by_year(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(df.index.year)
    tab = pd.DataFrame({
        "corr_daily": g.apply(lambda d: d.r_qqq.corr(d.r_gld)),
        "corr60_mean": g.corr60.mean(),
        "share_liquidity": g.state60.apply(lambda s: (s == 1).mean()),
        "share_haven": g.state60.apply(lambda s: (s == -1).mean()),
        "qqq_ret": g.r_qqq.sum().apply(np.expm1),
        "gld_ret": g.r_gld.sum().apply(np.expm1),
    })
    tab.index.name = "year"
    return tab


# ------------------------------------------------------------------------ plots

def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _shade_states(ax, state: pd.Series):
    s = state.dropna()
    run = (s != s.shift()).cumsum()
    for _, block in s.groupby(run):
        v = block.iloc[0]
        if v != 0:
            ax.axvspan(block.index[0], block.index[-1], color=C_POS if v > 0 else C_NEG, alpha=0.13, linewidth=0)


def plot_overview(df: pd.DataFrame, path: Path):
    px = np.exp(df[["r_qqq", "r_gld"]].cumsum()) * 100
    c60 = critical_r(60)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.4, 0.7]},
                             facecolor=SURFACE)
    ax = axes[0]
    ax.plot(px.index, px.r_qqq, color=C_QQQ, lw=1.4, label="QQQ")
    ax.plot(px.index, px.r_gld, color=C_GLD, lw=1.4, label="GLD")
    ax.set_yscale("log")
    ax.set_ylabel("total return index (log, start=100)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for col, name in ((px.r_qqq, "QQQ"), (px.r_gld, "GLD")):
        ax.annotate(name, (col.index[-1], col.iloc[-1]), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=8, va="center")

    ax = axes[1]
    _shade_states(ax, df.state60)
    ax.plot(df.index, df.corr20, color=C_NEUTRAL, lw=0.6, alpha=0.8, label="20d corr")
    ax.plot(df.index, df.corr60, color=INK, lw=1.2, label="60d corr")
    for v in (c60, -c60):
        ax.axhline(v, color=INK_2, lw=0.7, ls="--")
    ax.axhline(0, color=INK_2, lw=0.7)
    ax.set_ylim(-1, 1)
    ax.set_ylabel("rolling corr of daily log returns", color=INK_2, fontsize=9)
    handles = ax.get_legend_handles_labels()[0] + [
        plt.Rectangle((0, 0), 1, 1, color=C_POS, alpha=0.25), plt.Rectangle((0, 0), 1, 1, color=C_NEG, alpha=0.25)]
    ax.legend(handles, ["20d corr", "60d corr", f"60d ≥ +{c60:.2f} (liquidity)", f"60d ≤ −{c60:.2f} (safe-haven)"],
              frameon=False, fontsize=8, loc="lower left", ncol=4)

    ax = axes[2]
    ax.plot(df.index, df.vix, color=INK_2, lw=0.8)
    ax.set_ylabel("VIX", color=INK_2, fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for a in axes:
        _style(a)
    fig.suptitle("GLD vs QQQ: rolling correlation regimes", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_recent(df: pd.DataFrame, path: Path, years: int = 3):
    d = df.loc[df.index[-1] - pd.DateOffset(years=years):]
    c20, c60 = critical_r(20), critical_r(60)
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=SURFACE)
    _shade_states(ax, d.state60)
    ax.plot(d.index, d.corr20, color=C_NEUTRAL, lw=0.9, label="20d corr")
    ax.plot(d.index, d.corr60, color=INK, lw=1.4, label="60d corr")
    ax.plot(d.index, d.corr60_robust, color=C_ROBUST, lw=1.2, label="60d corr, vol-scaled & clipped ±3σ")
    ax.axhline(0, color=INK_2, lw=0.7)
    for v, ls in ((c60, "--"), (-c60, "--"), (c20, ":"), (-c20, ":")):
        ax.axhline(v, color=INK_2, lw=0.7, ls=ls)
    ax.set_ylim(-1, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=3)
    ax.set_title(f"Last {years} years (dashed ±{c60:.2f}: 60d 95% band, dotted ±{c20:.2f}: 20d band)",
                 color=INK, fontsize=10, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_null(df: pd.DataFrame, nulls: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=SURFACE)
    bins = np.linspace(-1, 1, 61)
    for ax, w in zip(axes, WINDOWS):
        ax.hist(nulls[f"_null_sample_corr{w}"][~np.isnan(nulls[f"_null_sample_corr{w}"])], bins=bins, density=True,
                color=C_NEUTRAL, alpha=0.55, label=f"null: constant ρ={nulls['full_sample_rho']:.2f}")
        ax.hist(df[f"corr{w}"].dropna(), bins=bins, density=True, histtype="step", color=INK, lw=1.5, label="observed")
        c = critical_r(w)
        for v in (c, -c):
            ax.axvline(v, color=INK_2, ls="--", lw=0.7)
        o, g = nulls[f"w{w}"]["observed"], nulls[f"w{w}"]["gaussian_const_rho"]
        ax.set_title(f"{w}d window: sd {o['std']:.2f} vs null {g['std']:.2f}; "
                     f"sign flips/yr {o['sign_flips_per_year']:.1f} vs {g['sign_flips_per_year'][0]:.1f}",
                     color=INK, fontsize=9, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_by_year(tab: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(12, 3.8), facecolor=SURFACE)
    colors = [C_POS if v > 0 else C_NEG for v in tab.corr_daily]
    ax.bar(tab.index, tab.corr_daily, color=colors, width=0.7, edgecolor=SURFACE, linewidth=2)
    ax.axhline(0, color=INK_2, lw=0.8)
    for x, v in zip(tab.index, tab.corr_daily):
        ax.annotate(f"{v:+.2f}", (x, v), xytext=(0, 3 if v >= 0 else -10), textcoords="offset points",
                    ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(tab.index)
    ax.tick_params(axis="x", rotation=90)
    ax.set_title("Calendar-year correlation of daily GLD and QQQ returns", color=INK, fontsize=10, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_persistence(pairs: pd.DataFrame, path: Path, w: int):
    fig, ax = plt.subplots(figsize=(5.5, 5), facecolor=SURFACE)
    x, y = pairs.iloc[:, 0], pairs.iloc[:, 1]
    ax.scatter(x, y, s=22, color=C_QQQ, edgecolor=SURFACE, linewidth=1, alpha=0.85)
    beta = np.polyfit(x, y, 1)
    xs = np.linspace(-1, 1, 10)
    ax.plot(xs, np.polyval(beta, xs), color=INK, lw=1.2)
    ax.axhline(0, color=INK_2, lw=0.6)
    ax.axvline(0, color=INK_2, lw=0.6)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(f"{w}d corr at t", color=INK_2, fontsize=9)
    ax.set_ylabel(f"{w}d corr over next {w} days", color=INK_2, fontsize=9)
    ax.set_title(f"Persistence (non-overlapping): r = {x.corr(y):.2f}, slope = {beta[0]:.2f}", color=INK, fontsize=10, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = build_frame()

    nulls = null_models(df, rng)
    ctx = regime_context(df)
    fx = factor_exposure(df)
    ea = ex_ante(df)
    rob = robustness(df)
    tab = by_year(df)
    eps = episodes(df.state60)

    def annotate_episode(e):
        win = df.loc[e.start:e.end]
        return pd.Series({
            "mean_corr60": win.corr60.mean(),
            "qqq_ret": np.expm1(win.r_qqq.sum()), "gld_ret": np.expm1(win.r_gld.sum()),
            "vix_mean": win.vix.mean(),
            "d_real_yield_bp": (win.real_yield.iloc[-1] - win.real_yield.iloc[0]) * 100,
        })

    eps = pd.concat([eps, eps.apply(annotate_episode, axis=1)], axis=1)

    last = df.dropna(subset=["corr60"]).iloc[-1]
    # load() may have fetched some series mid-run; record which snapshot each one came from.
    manifest = dp._load_manifest()
    provenance = {n: {k: manifest.get(n, {}).get(k) for k in ("last", "fetched_at_ny")}
                  for n in ("QQQ", "GLD", "IAU", "GC_F", "VIX", "DXY", "DFII10")}
    results = {
        "sample": {"start": df.index[0].date().isoformat(), "end": df.index[-1].date().isoformat(), "days": len(df)},
        "data": provenance,
        "latest": {"date": df.index[-1].date().isoformat(), "corr20": float(last.corr20), "corr60": float(last.corr60),
                   "corr60_robust": float(last.corr60_robust),
                   "state20": STATE_LABEL[int(last.state20)], "state60": STATE_LABEL[int(last.state60)]},
        "distribution": {f"corr{w}": {
            "mean": float(df[f"corr{w}"].mean()), "median": float(df[f"corr{w}"].median()),
            "share_positive": float((df[f"corr{w}"] > 0).mean()),
            "share_liquidity": float((df[f"state{w}"] == 1).mean()),
            "share_haven": float((df[f"state{w}"] == -1).mean()),
        } for w in WINDOWS},
        "null_models": {k: v for k, v in nulls.items() if not k.startswith("_")},
        "episodes60": {
            "count": len(eps), "liquidity": int((eps.state == 1).sum()), "haven": int((eps.state == -1).sum()),
            "median_days": float(eps.days.median()), "max_days": int(eps.days.max()),
            "at_least_20d": int((eps.days >= 20).sum()),
            "at_least_20d_liquidity": int(((eps.days >= 20) & (eps.state == 1)).sum()),
            "at_least_20d_haven": int(((eps.days >= 20) & (eps.state == -1)).sum()),
        },
        "ex_ante": {k: v for k, v in ea.items() if not k.startswith("_")},
        "robustness": rob,
        "outlier_sensitivity": outlier_sensitivity(df),
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    df[["corr20", "corr60", "corr60_robust", "state20", "state60", "state60_robust"]].to_csv(OUT / "rolling_corr.csv", float_format="%.4f")
    ctx.to_csv(OUT / "regime_context.csv", float_format="%.4f")
    fx.to_csv(OUT / "factor_exposure.csv", float_format="%.4f")
    tab.to_csv(OUT / "by_year.csv", float_format="%.4f")
    eps.assign(start=eps.start.dt.date, end=eps.end.dt.date).to_csv(OUT / "episodes60.csv", index=False, float_format="%.4f")

    plot_overview(df, OUT / "overview.png")
    plot_recent(df, OUT / "recent_3y.png")
    plot_null(df, nulls, OUT / "null_vs_observed.png")
    plot_by_year(tab, OUT / "by_year.png")
    plot_persistence(ea["_pairs60"], OUT / "persistence60.png", 60)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(ctx.round(3).T.to_string())
    print(fx.round(3).T.to_string())
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
