"""Unified Black-Scholes analytics, finite differences, and complex-step Greeks."""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

# ---------------------------------------------------------------------------
# Standard normal helpers and analytic Greeks (from analytic_greeks.py)
# ---------------------------------------------------------------------------

INV_SQRT_2 = 0.70710678118654752440
INV_SQRT_2PI = 0.39894228040143267794
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
    """Vectorized, numerically-stable log(F/K) with log1p trick near F ≈ K."""
    F = np.asarray(F, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    assert F.shape == K.shape, "F and K must have the same shape"

    out = np.empty_like(F)
    out.fill(np.nan)

    posK = K > 0.0
    nzK = K != 0.0

    ratio = np.empty_like(F)
    np.divide(F, K, out=ratio, where=nzK)
    np.log(ratio, out=out, where=nzK)

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
    """Black-Scholes delta for a European call with continuous yield q."""

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
    mask = den > 0.0
    np.divide(num, den, out=d1, where=mask)
    d1[~mask] = np.sign(num[~mask]) * np.inf

    return np.exp(-q * T) * Phi_real(d1)


def Gamma(
    S: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Black-Scholes gamma for a European call with continuous yield q."""

    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    Tpos = np.maximum(T, 0.0)
    sT = sigma * np.sqrt(Tpos)

    out = np.zeros_like(S, dtype=np.float64)

    mask = (S > 0.0) & (sigma > 0.0) & (Tpos > 0.0)
    if not np.any(mask):
        return out

    Sm, Km, rm, qm, sigm, Tm, sTm = (
        S[mask],
        K[mask],
        r[mask],
        q[mask],
        sigma[mask],
        Tpos[mask],
        sT[mask],
    )

    F = Sm * np.exp((rm - qm) * Tm)
    ln = _log_F_over_K(F, Km, eps=1e-12)
    d1 = (ln + 0.5 * sigm * sigm * Tm) / sTm
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


# ---------------------------------------------------------------------------
# Classical forward differences (from classical_forward_differences.py)
# ---------------------------------------------------------------------------

def bs_price_call(
    S: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Black-Scholes price for a European call with continuous yield q."""

    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    DF = np.exp(-r * T)
    F = S * np.exp((r - q) * T)
    sT = sigma * np.sqrt(np.maximum(T, 0.0))

    out = np.empty_like(DF)

    zero_mask = sT == 0.0
    if np.any(zero_mask):
        out[zero_mask] = DF[zero_mask] * np.maximum(F[zero_mask] - K[zero_mask], 0.0)

    pos_mask = ~zero_mask
    if np.any(pos_mask):
        ln_F_over_K = _log_F_over_K(F[pos_mask], K[pos_mask], eps=1e-12)
        d1 = (ln_F_over_K + 0.5 * sigma[pos_mask] * sigma[pos_mask] * T[pos_mask]) / sT[pos_mask]
        d2 = d1 - sT[pos_mask]
        out[pos_mask] = DF[pos_mask] * (F[pos_mask] * Phi_real(d1) - K[pos_mask] * Phi_real(d2))

    return out


def delta_forward(
    S: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    T: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """Delta via classical forward difference: [C(S+h)-C(S)]/h."""

    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    price_plus = bs_price_call(S + h, K, r, q, sigma, T)
    price = bs_price_call(S, K, r, q, sigma, T)
    return (price_plus - price) / h


def gamma_forward(
    S: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    T: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """Gamma via classical forward difference: [C(S+2h)-2C(S+h)+C(S)]/h^2."""

    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    C_plus_2 = bs_price_call(S + 2 * h, K, r, q, sigma, T)
    C_plus = bs_price_call(S + h, K, r, q, sigma, T)
    C = bs_price_call(S, K, r, q, sigma, T)
    return (C_plus_2 - 2 * C_plus + C) / (h * h)


# Backward-compatible aliases
detla_forward = delta_forward
Gamma_forward = gamma_forward


# ---------------------------------------------------------------------------
# Complex-step helpers (from complex_step_differentiation.py)
# ---------------------------------------------------------------------------


def Phi_t(z):
    """
    Type-dispatched standard normal CDF that preserves complex-step accuracy.

    Real inputs defer to Phi_real. For complex inputs we use the complex-step
    expansion Phi(a + i b) ≈ Phi(a) + i b phi(a) + 0.5 a b^2 phi(a) so that the
    imaginary component retains O(h) accuracy and the real component is
    accurate enough for second-derivative estimators.
    """

    arr = np.asarray(z)
    if not np.iscomplexobj(arr):
        return Phi_real(arr)

    zr = np.real(arr)
    zi = np.imag(arr)
    pdf = phi(zr)
    real_part = Phi_real(zr) + 0.5 * zr * (zi * zi) * pdf
    imag_part = pdf * zi
    return real_part + 1j * imag_part


Phi_dispatch = Phi_t


def _promote_and_broadcast(*args):
    """Convert inputs to a common dtype and broadcast to a shared shape."""

    raw = [np.asarray(arg) for arg in args]
    dtype = np.result_type(*raw, np.float64)
    promoted = [np.asarray(arg, dtype=dtype) for arg in raw]
    broadcasted = np.broadcast_arrays(*promoted) if promoted else []
    scalar_output = all(arr.ndim == 0 for arr in broadcasted)
    return dtype, broadcasted, scalar_output


def _log_F_over_K_t(F, K, eps=1e-12):
    """Type-dispatched log(F/K) that falls back to the complex variant."""

    if np.iscomplexobj(F) or np.iscomplexobj(K):
        F = np.asarray(F, dtype=np.complex128)
        K = np.asarray(K, dtype=np.complex128)
        assert F.shape == K.shape, "F and K must have the same shape"

        out = np.empty_like(F, dtype=np.complex128)
        out.fill(np.nan)

        nzK = K != 0
        zK = ~nzK

        if np.any(nzK):
            ratio = np.empty_like(out)
            np.divide(F, K, out=ratio, where=nzK)
            np.log(ratio, out=out, where=nzK)

            x = np.empty_like(out)
            np.divide(F - K, K, out=x, where=nzK)
            small = nzK & (np.abs(x) <= eps)
            if np.any(small):
                out[small] = np.log1p(x[small])

        if np.any(zK):
            out[zK] = np.log(F[zK]) - np.log(K[zK])

        return out

    return _log_F_over_K(F, K, eps=eps)


def bs_price_call_t(S, K, r, q, sigma, T, eps=1e-12):
    """
    Black-Scholes call price that supports real or complex scalars/arrays.

    Real inputs reproduce the analytic pricing function, while complex inputs
    (complex-step perturbations) evaluate the formula analytically.
    """

    dtype, (S, K, r, q, sigma, T), scalar = _promote_and_broadcast(S, K, r, q, sigma, T)
    is_complex = np.issubdtype(dtype, np.complexfloating)

    DF = np.exp(-r * T)
    F = S * np.exp((r - q) * T)

    if is_complex:
        sqrt_T = np.sqrt(T)
    else:
        sqrt_T = np.sqrt(np.maximum(T, 0.0))
    sigma_sqrt_T = sigma * sqrt_T

    zero_mask = np.isclose(sigma_sqrt_T, 0.0)
    out = np.empty_like(F, dtype=dtype)

    if np.any(zero_mask):
        if is_complex:
            raise ValueError("Complex inputs require non-zero sigma*sqrt(T) for stable pricing.")
        intrinsic = np.maximum(F[zero_mask] - K[zero_mask], 0.0)
        out[zero_mask] = DF[zero_mask] * intrinsic

    if np.any(~zero_mask):
        mask = ~zero_mask
        Fm = F[mask]
        Km = K[mask]
        sigTm = sigma_sqrt_T[mask]
        ln = _log_F_over_K_t(Fm, Km, eps=eps)
        num = ln + 0.5 * sigma[mask] * sigma[mask] * T[mask]
        d1 = num / sigTm
        d2 = d1 - sigTm
        price = DF[mask] * (Fm * Phi_t(d1) - Km * Phi_t(d2))
        out[mask] = price

    if scalar:
        return out.item()
    return out


def delta_cs(S, K, r, q, sigma, T, h=1e-20):
    """Black-Scholes delta obtained via complex-step differentiation."""

    S_c = np.asarray(S, dtype=np.complex128) + 1j * h
    price_c = bs_price_call_t(S_c, K, r, q, sigma, T)
    return np.imag(price_c) / h


def gamma_cs_real(S, K, r, q, sigma, T, h=1e-5):
    """Gamma estimate from the real part of the complex-step perturbation."""

    S_c_ih = np.asarray(S, dtype=np.complex128) + 1j * h
    R_C_S_ih = np.real(bs_price_call_t(S_c_ih, K, r, q, sigma, T))
    C_s = bs_price_call_t(S, K, r, q, sigma, T)
    return -2.0 / (h * h) * (R_C_S_ih - C_s)


def gamma_45_deg(S, K, r, q, sigma, T, h=1e-5):
    """Gamma estimate using the 45-degree complex-step direction."""

    omega = np.complex128((1.0 + 1.0j) / np.sqrt(2.0))
    S0 = np.asarray(S, dtype=np.complex128)
    S_fwd = S0 + omega * h
    S_bwd = S0 - omega * h
    C_fwd = bs_price_call_t(S_fwd, K, r, q, sigma, T)
    C_bwd = bs_price_call_t(S_bwd, K, r, q, sigma, T)
    return np.imag(C_fwd + C_bwd) / (h * h)


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
    "bs_price_call",
    "delta_forward",
    "gamma_forward",
    "detla_forward",
    "Gamma_forward",
    "Phi_t",
    "Phi_dispatch",
    "_log_F_over_K_t",
    "bs_price_call_t",
    "delta_cs",
    "gamma_cs_real",
    "gamma_45_deg",
]
