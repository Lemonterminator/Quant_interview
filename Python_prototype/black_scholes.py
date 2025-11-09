"""Unified Black-Scholes analytics, finite differences, and complex-step Greeks.

This module provides:
  - Analytic Black–Scholes pricing/Greeks with numerically robust building blocks.
  - Classical finite-difference estimators (forward delta, 3-point gamma).
  - Complex-step estimators for first/second derivatives, including a 45° scheme.

Design + numerical notes (high level):
  - We compute log(F/K) via a log1p branch when F≈K to avoid catastrophic
    cancellation; this is used everywhere d1 or the intrinsic value depends on ln(F/K).
  - Guard divisions (e.g., σ√T) and branch on degenerate cases (σ√T→0) to fall
    back to intrinsic value or ±∞ limits for d1; this keeps vectorized code safe.
  - Complex-step paths keep the imaginary component intact at O(h) accuracy and
    avoid real-axis subtractive cancellation entirely, which stabilizes ∆ and Γ.
"""

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
    """Standard normal CDF Phi(z) using erfc for good tail behavior.

    Using erfc avoids exp(-z^2/2) underflow in the far tails and gives good
    relative accuracy for |z|≫1 compared to a naive integral/series.
    """
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * erfc(-z * INV_SQRT_2)


def phi(z: np.ndarray) -> np.ndarray:
    """Standard normal PDF phi(z). Simple exponential with 1/√(2π) factor."""
    z = np.asarray(z, dtype=np.float64)
    return INV_SQRT_2PI * np.exp(-0.5 * z**2)


def _log_F_over_K(F: np.ndarray, K: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Vectorized, numerically-stable log(F/K) with log1p trick near F ≈ K.

    When F≈K we evaluate log(1 + (F-K)/K) to reduce cancellation error. The
    implementation is branchless over arrays using boolean masks.
    """
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

    # Compute forward price F = S * exp((r-q)T) ensuring we propagate arrays
    F = S * np.exp((r - q) * T)
    ln = _log_F_over_K(F, K, eps=1e-12)

    # d1 = (ln(F/K) + 0.5 σ²T) / (σ√T); guard division when T=0 or σ=0.
    # If σ√T==0, we assign d1 the appropriate ±∞ sign so Phi(d1) returns 0/1.
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

    # Default Γ=0 for degenerate inputs; we fill in the valid subset below.
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

    # analytically evaluate phi(d1) in log-space for numerical stability
    F = Sm * np.exp((rm - qm) * Tm)
    ln = _log_F_over_K(F, Km, eps=1e-12)
    d1 = (ln + 0.5 * sigm * sigm * Tm) / sTm
    # Evaluate φ(d1) in log-space to avoid under/overflow in the tails.
    log_phi_d1 = -0.5 * d1 * d1 - LOG_SQRT_2PI
    phi_d1 = np.exp(log_phi_d1)

    out[mask] = np.exp(-qm * Tm) * phi_d1 / (Sm * sTm)
    return out


def gamma_ref(S, K, r, q, sigma, T, **kwargs):
    """Scalar reference for gamma; useful for testing and debugging.

    Mirrors Gamma() logic in scalar form with explicit branching; handy for
    unit tests and to sanity-check vectorized edge cases.
    """

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

    DF = np.exp(-r * T)  # discount factor
    F = S * np.exp((r - q) * T)
    sT = sigma * np.sqrt(np.maximum(T, 0.0))

    out = np.empty_like(DF)

    # Degenerate branch: σ√T=0 → price collapses to discounted intrinsic value.
    zero_mask = sT == 0.0
    if np.any(zero_mask):
        out[zero_mask] = DF[zero_mask] * np.maximum(F[zero_mask] - K[zero_mask], 0.0)

    pos_mask = ~zero_mask
    if np.any(pos_mask):
        # Stable log(F/K) and usual d1/d2 definitions.
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
    """Delta via classical forward difference: [C(S+h)-C(S)]/h.

    Sensitive to the usual truncation vs. roundoff trade-off. In practice,
    small h suffers from subtractive cancellation; large h increases truncation.
    """

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
    """Gamma via classical forward difference: [C(S+2h)-2C(S+h)+C(S)]/h^2.

    This 3-point forward stencil is particularly sensitive to cancellation,
    since C(S+2h), C(S+h), C(S) become nearly equal as h→0.
    """

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
    expansion Φ(a + i b) ≈ Φ(a) + i b φ(a) + 0.5 a b^2 φ(a) so that:
      - Im[Φ] captures the first derivative at O(h) without real cancellation.
      - Re[Φ] is accurate enough for second-derivative estimators (Γ via CS).
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
    """Convert inputs to a common dtype and broadcast to a shared shape.

    Ensures downstream vectorized code can assume consistent dtype/shape for
    real or complex scalars/arrays.
    """

    raw = [np.asarray(arg) for arg in args]
    dtype = np.result_type(*raw, np.float64)
    promoted = [np.asarray(arg, dtype=dtype) for arg in raw]
    broadcasted = np.broadcast_arrays(*promoted) if promoted else []
    scalar_output = all(arr.ndim == 0 for arr in broadcasted)
    return dtype, broadcasted, scalar_output


def _log_F_over_K_t(F, K, eps=1e-12):
    """Type-dispatched log(F/K) that falls back to the complex variant.

    For complex inputs, we must operate in complex space; we still use the
    log1p trick when |(F-K)/K| is small to reduce cancellation.
    """

    # When inputs are complex we cannot rely on real-only shortcuts; use complex arrays.
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

    Real inputs reproduce the analytic pricing function. Complex inputs are
    used for complex-step perturbations to estimate derivatives analytically
    without finite differencing.
    """

    dtype, (S, K, r, q, sigma, T), scalar = _promote_and_broadcast(S, K, r, q, sigma, T)
    is_complex = np.issubdtype(dtype, np.complexfloating)

    # discount factor and forward price appear in both real+complex math
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
    """Black-Scholes delta obtained via complex-step differentiation.

    Perturb S → S + i h. Then ∂C/∂S ≈ Im[C(S+ih)]/h with machine-precision
    accuracy in the imaginary component and no subtraction in the real axis.
    """

    S_c = np.asarray(S, dtype=np.complex128) + 1j * h
    price_c = bs_price_call_t(S_c, K, r, q, sigma, T)
    # Only the imaginary part carries derivative information for complex-step.
    return np.imag(price_c) / h


def gamma_cs_real(S, K, r, q, sigma, T, h=1e-5):
    """Gamma estimate from the real part of the complex-step perturbation.

    Using Re[C(S+ih)] removes the Im[·] term but preserves curvature effects;
    a simple second-order relation yields Γ from the real-part difference.
    """

    S_c_ih = np.asarray(S, dtype=np.complex128) + 1j * h
    R_C_S_ih = np.real(bs_price_call_t(S_c_ih, K, r, q, sigma, T))
    C_s = bs_price_call_t(S, K, r, q, sigma, T)
    # Using the real part removes the imaginary component while preserving curvature.
    return -2.0 / (h * h) * (R_C_S_ih - C_s)


def gamma_45_deg(S, K, r, q, sigma, T, h=1e-5):
    """Gamma estimate using the 45-degree complex-step direction.

    Displacing S by ±(1+i)h/√2 symmetrizes the perturbation in real/imag axes,
    cancelling lower-order terms; this improves truncation (≈O(h^4)).
    """

    omega = np.complex128((1.0 + 1.0j) / np.sqrt(2.0))
    S0 = np.asarray(S, dtype=np.complex128)
    S_fwd = S0 + omega * h
    S_bwd = S0 - omega * h
    C_fwd = bs_price_call_t(S_fwd, K, r, q, sigma, T)
    C_bwd = bs_price_call_t(S_bwd, K, r, q, sigma, T)
    # Symmetric displacement along (1+i)/√2 isolates the second derivative more cleanly.
    return np.imag(C_fwd + C_bwd) / (h * h)

Gamma_cs_real = gamma_cs_real
Gamma_45_deg = gamma_45_deg

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
