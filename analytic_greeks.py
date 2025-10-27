"""
Core analytics for Black–Scholes helpers (Python).

This module intentionally contains only the core, reusable functions:
  - Phi_real: standard normal CDF using erfc
  - phi: standard normal PDF
  - _log_F_over_K: numerically stable log(F/K)
  - delta_call: Black–Scholes call delta (with continuous dividend yield q)
  - Gamma: Black–Scholes call gamma
  - gamma_ref: scalar reference implementation for gamma

All plotting and test scaffolding has been moved to analytic_greek_test.py.
Comments are ASCII-only to avoid encoding issues in some terminals.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

# Constants
INV_SQRT_2 = 0.70710678118654752440
INV_SQRT_2PI = 0.39894228040143267794
# log(sqrt(2*pi)) = 0.5 * log(2*pi)
LOG_SQRT_2PI = 0.91893853320467274178


def Phi_real(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF Phi(z) using erfc for good tail behavior."""
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * erfc(-z * INV_SQRT_2)


def phi(z: np.ndarray) -> np.ndarray:
    """Standard normal PDF phi(z)."""
    z = np.asarray(z, dtype=np.float64)
    return INV_SQRT_2PI * np.exp(-0.5 * z**2)


def _log_F_over_K(F: np.ndarray, K: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Vectorized, numerically-stable log(F/K) with log1p trick when K>0 and |(F-K)/K| is small.
    Matches the C++ branching:
        if (K>0) { x = (F-K)/K; ln = (|x|<=eps) ? log1p(x) : log(F/K); }
        else     { ln = log(F/K); }
    """
    # Accept scalars or arrays
    F = np.asarray(F, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    assert F.shape == K.shape, "F and K must have the same shape"

    out = np.empty_like(F)
    out.fill(np.nan)

    # masks
    posK = K > 0.0
    nzK = K != 0.0

    # ratio = F/K computed only where K != 0 to avoid warnings
    ratio = np.empty_like(F)
    np.divide(F, K, out=ratio, where=nzK)

    # (If K==0 anywhere, ratio will be garbage; we won't read it except where nzK)
    np.log(ratio, out=out, where=nzK)

    # For K>0 and |x| small, use log1p(x)
    x = np.empty_like(F)
    np.divide(F - K, K, out=x, where=posK)
    small = posK & (np.abs(x) <= eps)
    out[small] = np.log1p(x[small])

    return out


def delta_call(
    S: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Black–Scholes delta for a European call with continuous yield q."""
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    F = S * np.exp((r - q) * T)
    ln = _log_F_over_K(F, K, eps=1e-12)

    num = ln + 0.5 * (sigma * sigma) * T
    den = sigma * np.sqrt(T)

    d1 = np.empty_like(num)
    d1.fill(np.nan)
    # regular division where den>0
    mask = den > 0.0
    np.divide(num, den, out=d1, where=mask)
    # limit as den -> 0: sign(num)*inf
    zmask = ~mask
    d1[zmask] = np.sign(num[zmask]) * np.inf

    # Delta = exp(-q*T) * Phi(d1)
    return np.exp(-q * T) * Phi_real(d1)


def Gamma(
    S: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Black–Scholes gamma for a European call with continuous yield q."""
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    # Clamp T >= 0
    Tpos = np.maximum(T, 0.0)
    sT = sigma * np.sqrt(Tpos)

    out = np.zeros_like(S, dtype=np.float64)

    # Only when S>0, sigma>0, T>0, Gamma is non-zero
    mask = (S > 0.0) & (sigma > 0.0) & (Tpos > 0.0)
    if not np.any(mask):
        return out

    Sm, Km, rm, qm, sigm, Tm, sTm = S[mask], K[mask], r[mask], q[mask], sigma[mask], Tpos[mask], sT[mask]

    F = Sm * np.exp((rm - qm) * Tm)
    ln = _log_F_over_K(F, Km, eps=1e-12)

    d1 = (ln + 0.5 * sigm * sigm * Tm) / sTm

    # log phi(d1) = -0.5*d1^2 - log(sqrt(2*pi))
    log_phi_d1 = -0.5 * d1 * d1 - LOG_SQRT_2PI
    phi_d1 = np.exp(log_phi_d1)

    out[mask] = np.exp(-qm * Tm) * phi_d1 / (Sm * sTm)
    return out


def gamma_ref(S, K, r, q, sigma, T, **kwargs):
    """Scalar reference for gamma; useful for testing."""
    S = float(S)
    K = float(K)
    r = float(r)
    q = float(q)
    sigma = float(sigma)
    T = float(T)

    Tpos = max(T, 0.0)
    sT = sigma * np.sqrt(Tpos)
    if S <= 0.0 or sigma <= 0.0 or Tpos <= 0.0:
        return 0.0

    F = S * np.exp((r - q) * Tpos)
    ln = float(_log_F_over_K(F, K, 1e-12))
    d1 = (ln + 0.5 * sigma * sigma * Tpos) / sT
    phi_val = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * d1 * d1)
    return np.exp(-q * Tpos) * phi_val / (S * sT)


__all__ = [
    "INV_SQRT_2",
    "INV_SQRT_2PI",
    "LOG_SQRT_2PI",
    "Phi_real",
    "phi",
    "_log_F_over_K",
    "delta_call",
    "Gamma",
    "gamma_ref",
]

