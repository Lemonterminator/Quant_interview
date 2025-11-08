"""
Tests for complex_step_differentiation.py

Run:
  python complex_step_differentiation_test.py
"""

from __future__ import annotations

import numpy as np

from black_scholes import (
    Gamma,
    Phi_real,
    Phi_t,
    bs_price_call,
    bs_price_call_t,
    delta_call,
    delta_cs,
    gamma_45_deg,
    gamma_cs_real,
    phi,
)


def test_bs_price_matches_real_pricer(seed: int = 1001):
    rng = np.random.default_rng(seed)
    n = 4000

    S = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=n)
    K = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=n)
    r = rng.uniform(-0.01, 0.05, size=n)
    q = rng.uniform(0.00, 0.03, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    T = rng.uniform(1 / 365, 2.0, size=n)

    price_t = bs_price_call_t(S, K, r, q, sigma, T)
    price_ref = bs_price_call(S, K, r, q, sigma, T)

    abs_err = np.max(np.abs(price_t - price_ref))
    rel_err = np.max(np.abs((price_t - price_ref) / np.maximum(1e-12, np.abs(price_ref))))
    print(f"[price] max abs = {abs_err:.3e}, max rel = {rel_err:.3e}")

    assert np.isfinite(price_t).all()
    assert abs_err < 1e-12 and rel_err < 1e-12


def test_price_edge_cases():
    # Vector inputs with sigma=0 or T=0 should reduce to discounted intrinsic on forward
    S = np.array([100.0, 100.0, 80.0, 120.0])
    K = np.array([100.0, 90.0, 100.0, 110.0])
    r = np.array([0.02, 0.00, 0.01, -0.01])
    q = np.array([0.00, 0.01, 0.00, 0.02])
    sigma = np.array([0.0, 0.2, 0.0, 0.0])
    T = np.array([1.0, 0.0, 0.5, 2.0])

    DF = np.exp(-r * T)
    F = S * np.exp((r - q) * T)
    expected = DF * np.maximum(F - K, 0.0)

    price = bs_price_call_t(S, K, r, q, sigma, T)
    err = np.max(np.abs(price - expected))
    print(f"[price edge] max abs err = {err:.3e}")
    assert err <= 1e-12


def test_phi_t_complex_step(seed: int = 2025, h: float = 1e-20):
    rng = np.random.default_rng(seed)
    z = rng.normal(loc=0.0, scale=2.0, size=1000)
    zc = z.astype(np.complex128) + 1j * h
    val = Phi_t(zc)
    # Real part matches Phi_real(z), imag/h matches phi(z)
    abs_r = np.max(np.abs(np.real(val) - Phi_real(z)))
    abs_i = np.max(np.abs(np.imag(val) / h - phi(z)))
    print(f"[Phi_t] max |Re-Ref| = {abs_r:.3e}, max |Im/h - phi| = {abs_i:.3e}")
    assert abs_r < 1e-15 and abs_i < 1e-12


def test_delta_cs_matches_analytic(seed: int = 1234):
    rng = np.random.default_rng(seed)
    n = 4000

    S = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=n)
    K = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=n)
    r = rng.uniform(-0.01, 0.05, size=n)
    q = rng.uniform(0.00, 0.03, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    T = rng.uniform(1 / 365, 2.0, size=n)

    delta_csd = delta_cs(S, K, r, q, sigma, T)
    delta_an = delta_call(S, K, r, q, sigma, T)

    abs_err = np.max(np.abs(delta_csd - delta_an))
    rel_err = np.max(np.abs((delta_csd - delta_an) / np.maximum(1e-12, np.abs(delta_an))))
    print(f"[delta cs] max abs = {abs_err:.3e}, max rel = {rel_err:.3e}")

    assert np.isfinite(delta_csd).all()
    # Complex-step first derivative is extremely accurate
    assert abs_err < 1e-12 and rel_err < 1e-12


def test_gamma_cs_variants_match_analytic(seed: int = 4321):
    rng = np.random.default_rng(seed)
    n = 4000

    S = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=n)
    K = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=n)
    r = rng.uniform(-0.01, 0.05, size=n)
    q = rng.uniform(0.00, 0.03, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    T = rng.uniform(1 / 365, 2.0, size=n)

    gamma_r = gamma_cs_real(S, K, r, q, sigma, T)
    gamma_o = gamma_45_deg(S, K, r, q, sigma, T)
    gamma_an = Gamma(S, K, r, q, sigma, T)

    # Absolute error
    abs_err_r = np.max(np.abs(gamma_r - gamma_an))
    abs_err_o = np.max(np.abs(gamma_o - gamma_an))

    print(
        f"[gamma cs] real: max abs={abs_err_r:.3e}; 45deg: max abs={abs_err_o:.3e}"
    )

    assert np.isfinite(gamma_r).all() and np.isfinite(gamma_o).all()
    # 45-degree complex-step is highly accurate
    assert abs_err_o < 1e-8
    # Real-part complex-step is sensitive; accept looser bound
    assert abs_err_r < 5e-3


def test_complex_zero_sigmaT_rejected(h: float = 1e-20):
    # Complex-step pricing should reject sigma*sqrt(T) ~ 0 to avoid division by zero
    S = 100.0
    K = 100.0
    r = 0.01
    q = 0.00
    T = 1.0
    sigma_zero = 0.0

    try:
        _ = bs_price_call_t(np.complex128(S) + 1j * h, K, r, q, sigma_zero, T)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for complex-step with sigma*sqrt(T)==0")

    # Also reject T == 0
    T_zero = 0.0
    sigma_pos = 0.2
    try:
        _ = bs_price_call_t(np.complex128(S) + 1j * h, K, r, q, sigma_pos, T_zero)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for complex-step with sigma*sqrt(T)==0 (T==0)")


def main():
    test_bs_price_matches_real_pricer()
    test_price_edge_cases()
    test_phi_t_complex_step()
    test_delta_cs_matches_analytic()
    test_gamma_cs_variants_match_analytic()
    test_complex_zero_sigmaT_rejected()
    print("All complex-step differentiation tests passed.")


if __name__ == "__main__":
    main()
