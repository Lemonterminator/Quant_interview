"""Small, dependency-free statistics helpers shared by studies 02 and 03.

Only numpy/pandas are required (no scipy/statsmodels): OLS with Newey-West
standard errors, Wald tests with exact chi-square tails for integer degrees of
freedom, the Lo-MacKinlay variance ratio, an ADF unit-root test and an AR(1)
half-life. Each has an offline unit test in ``tests/``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# MacKinnon (2010) asymptotic ADF critical values, regression with a constant.
ADF_CRITICAL = {"1%": -3.43, "5%": -2.86, "10%": -2.57}


def norm_sf(z: float) -> float:
    """Upper tail of the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def two_sided_p(t: float) -> float:
    return 2 * norm_sf(abs(t))


def chi2_sf(x: float, df: int) -> float:
    """Exact upper tail of chi-square with integer ``df`` (closed form)."""
    if x <= 0:
        return 1.0
    half = x / 2
    if df % 2 == 0:
        term, total = 1.0, 1.0
        for k in range(1, df // 2):
            term *= half / k
            total += term
        return math.exp(-half) * total
    total = math.erfc(math.sqrt(half))
    if df > 1:
        term = math.sqrt(2 * x / math.pi) * math.exp(-half)
        acc = term
        for k in range(1, (df - 1) // 2):
            term *= x / (2 * k + 1)
            acc += term
        total += acc
    return total


def ols_nw(y, X, lags: int = 0, add_const: bool = True) -> dict:
    """OLS with Newey-West (Bartlett) covariance; ``lags=0`` gives White (HC0).

    NaN rows are dropped jointly. Returns betas, standard errors, t-stats and
    the covariance so callers can run Wald tests.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if add_const:
        X = np.column_stack([np.ones(len(X)), X])
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[keep], X[keep]
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    xu = X * u[:, None]
    S = xu.T @ xu
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1 - lag / (lags + 1)
        g = xu[lag:].T @ xu[:-lag]
        S += w * (g + g.T)
    cov = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.diag(cov))
    tss = ((y - y.mean()) ** 2).sum()
    return {"beta": beta, "se": se, "t": beta / se, "cov": cov, "n": n,
            "r2": float(1 - (u ** 2).sum() / tss) if tss > 0 else float("nan")}


def wald(fit: dict, idx: list[int]) -> dict:
    """Joint test that ``beta[idx] == 0`` using the fit's (robust) covariance."""
    b = fit["beta"][idx]
    V = fit["cov"][np.ix_(idx, idx)]
    stat = float(b @ np.linalg.solve(V, b))
    return {"chi2": stat, "df": len(idx), "p": chi2_sf(stat, len(idx))}


def variance_ratio(log_price, q: int) -> dict:
    """Lo-MacKinlay (1988) variance ratio with overlapping q-period returns and
    the heteroskedasticity-robust z*. VR > 1: trending; VR < 1: mean-reverting."""
    p = np.asarray(log_price, dtype=float)
    p = p[np.isfinite(p)]
    r = np.diff(p)
    n = len(r)
    mu = r.mean()
    dev = r - mu
    var_1 = (dev ** 2).sum() / (n - 1)
    rq = p[q:] - p[:-q] - q * mu
    m = q * (n - q + 1) * (1 - q / n)
    var_q = (rq ** 2).sum() / m
    vr = var_q / var_1
    d2 = dev ** 2
    denom = d2.sum() ** 2
    theta = 0.0
    for j in range(1, q):
        delta = n * (d2[j:] * d2[:-j]).sum() / denom
        theta += (2 * (q - j) / q) ** 2 * delta
    z = math.sqrt(n) * (vr - 1) / math.sqrt(theta)
    return {"q": q, "vr": float(vr), "z_robust": float(z), "p": two_sided_p(z)}


def adf(x, max_lag: int = 12) -> dict:
    """Augmented Dickey-Fuller with a constant; lag order chosen by AIC on a
    common sample. Compare ``stat`` with ``ADF_CRITICAL``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    dx = np.diff(x)
    best = None
    start = max_lag + 1
    y = dx[start - 1:]
    for p in range(max_lag + 1):
        cols = [x[start - 1:-1]] + [dx[start - 1 - i:len(dx) - i] for i in range(1, p + 1)]
        X = np.column_stack([np.ones(len(y))] + cols)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        u = y - X @ beta
        n, k = X.shape
        aic = n * math.log((u ** 2).sum() / n) + 2 * k
        if best is None or aic < best[0]:
            s2 = (u ** 2).sum() / (n - k)
            se = math.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
            best = (aic, p, beta[1] / se, n)
    _, lag, stat, n = best
    return {"stat": float(stat), "lags": lag, "n": n,
            "reject_5pct": bool(stat < ADF_CRITICAL["5%"]), "critical": ADF_CRITICAL}


def ar1_half_life(x) -> dict:
    """Half-life implied by x_t = a + phi * x_{t-1}; infinite when phi >= 1."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    fit = ols_nw(x[1:], x[:-1])
    phi = float(fit["beta"][1])
    hl = math.log(0.5) / math.log(phi) if 0 < phi < 1 else float("inf")
    return {"phi": phi, "half_life_days": hl}


def lagged_corr(x: pd.Series, y: pd.Series, lags) -> pd.Series:
    """corr(x_t, y_{t+k}) for each k (k > 0: x leads y)."""
    return pd.Series({k: x.corr(y.shift(-k)) for k in lags}, name="corr")


def circular_shift_pvalue(signal: np.ndarray, returns: np.ndarray, stat, n_sim: int,
                          rng: np.random.Generator, min_shift: int) -> tuple[float, np.ndarray]:
    """One-sided p-value of ``stat(signal, returns)`` against the same statistic
    with the signal circularly shifted: both series keep their own persistence,
    only the alignment between them is destroyed."""
    observed = stat(signal, returns)
    t = len(signal)
    shifts = rng.integers(min_shift, t - min_shift, n_sim)
    null = np.array([stat(np.roll(signal, k), returns) for k in shifts])
    return float((null >= observed).mean()), null
