import numpy as np
from analytic_greeks import _log_F_over_K, Phi_real


def bs_price_call(
    S: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Black-Scholes price for a European call with continuous yield q (vectorized)."""
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

    # Where sT == 0, price collapses to discounted intrinsic on forward
    zero_mask = (sT == 0.0)
    if np.any(zero_mask):
        out[zero_mask] = DF[zero_mask] * np.maximum(F[zero_mask] - K[zero_mask], 0.0)

    # Where sT > 0, use standard BS formula
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
    """Gamma via classical second forward difference: [C(S+2h)-2C(S+h)+C(S)]/h^2."""
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


# Backward-compatible aliases for any existing imports
detla_forward = delta_forward
Gamma_forward = gamma_forward

