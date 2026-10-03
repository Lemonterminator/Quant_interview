"""
Tests for classical_forward_differences.py

Run:
  python classical_forward_differences_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Python_prototype"))

import numpy as np

from black_scholes import (
    Gamma,
    bs_price_call,
    delta_call,
    delta_forward,
    gamma_forward,
)


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

    price = bs_price_call(S, K, r, q, sigma, T)
    err = np.max(np.abs(price - expected))
    print(f"[price edge] max abs err = {err:.3e}")
    assert err <= 1e-12


def test_delta_forward_matches_analytic(seed: int = 123):
    rng = np.random.default_rng(seed)
    n = 5000

    S = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=n)
    K = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=n)
    r = rng.uniform(-0.01, 0.05, size=n)
    q = rng.uniform(0.00, 0.03, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    T = rng.uniform(1 / 365, 2.0, size=n)

    h = 1e-6
    delta_fd = delta_forward(S, K, r, q, sigma, T, h=h)
    delta_an = delta_call(S, K, r, q, sigma, T)

    abs_err = np.max(np.abs(delta_fd - delta_an))
    rel_err = np.max(np.abs((delta_fd - delta_an) / np.maximum(1e-12, np.abs(delta_an))))
    print(f"[delta] max abs = {abs_err:.3e}, max rel = {rel_err:.3e}")

    assert np.isfinite(delta_fd).all()
    assert abs_err < 2e-4 and rel_err < 2e-3


def test_gamma_forward_matches_analytic(seed: int = 456):
    rng = np.random.default_rng(seed)
    n = 5000

    S = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=n)
    K = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=n)
    r = rng.uniform(-0.01, 0.05, size=n)
    q = rng.uniform(0.00, 0.03, size=n)
    sigma = rng.uniform(0.05, 0.60, size=n)
    T = rng.uniform(1 / 365, 2.0, size=n)

    h = 1e-5
    gamma_fd = gamma_forward(S, K, r, q, sigma, T, h=h)
    gamma_an = Gamma(S, K, r, q, sigma, T)

    diff = gamma_fd - gamma_an
    abs_err = np.max(np.abs(diff))
    # Relative error only where analytic gamma is not tiny
    mask = np.abs(gamma_an) > 1e-6
    rel_err = (np.max(np.abs(diff[mask] / gamma_an[mask])) if np.any(mask) else 0.0)
    print(f"[gamma] max abs = {abs_err:.3e}, max rel (|g|>1e-6) = {rel_err:.3e}")

    assert np.isfinite(gamma_fd).all()
    # Use absolute tolerance; forward difference for 2nd derivative is O(h)
    assert abs_err < 2e-3


def main():
    test_price_edge_cases()
    test_delta_forward_matches_analytic()
    test_gamma_forward_matches_analytic()
    print("All classical forward differences tests passed.")


if __name__ == "__main__":
    main()
