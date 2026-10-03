"""
Tests and plots for analytic_greeks.py

Run:
  python analytic_greek_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Python_prototype"))

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtr

from black_scholes import (
    Phi_real,
    phi,
    _log_F_over_K,
    delta_call,
    Gamma,
    gamma_ref,
)


def Phi_real_test():
    fig, ax = plt.subplots(1, 2)

    # Left: Full CDF
    z = np.linspace(-5, 5, 200)
    Phi = Phi_real(z)
    ax[0].plot(z, Phi, color="navy", lw=2)
    ax[0].set_title("Standard Normal CDF Phi(z)")
    ax[0].set_xlabel("z")
    ax[0].set_ylabel("Phi(z)")
    ax[0].grid(True, ls="--", alpha=0.6)

    # Right: Right-tail stability
    z_right_tail = np.linspace(4, 9, 400)
    Phi_right_tail = Phi_real(z_right_tail)
    ax[1].plot(z_right_tail, 1 - Phi_right_tail, color="crimson", lw=2, label="1 - Phi(z)")

    # First z where Phi == 1.0 (numerical saturation)
    saturated = np.where(Phi_right_tail == 1.0)[0]
    if len(saturated) > 0:
        z_sat = z_right_tail[saturated[0]]
        ax[1].axvline(z_sat, color="k", ls="--", label=f"Saturation at z ~ {z_sat:.3f}")

    ax[1].set_yscale("log")
    ax[1].set_title("Right-tail (float64 stability)")
    ax[1].set_xlabel("z")
    ax[1].set_ylabel("1 - Phi(z) (log scale)")
    ax[1].grid(True, which="both", ls="--", alpha=0.6)
    ax[1].legend()

    plt.tight_layout()


def phi_test():
    z = np.linspace(-5, 5, 200)
    phi_values = phi(z)

    fig, ax = plt.subplots(1, 2)
    # Left: Full PDF
    ax[0].plot(z, phi_values, color="green", lw=2)
    ax[0].set_title("Standard Normal PDF phi(z)")
    ax[0].set_xlabel("z")
    ax[0].set_ylabel("phi(z)")
    ax[0].grid(True, ls="--", alpha=0.6)

    # Right: Right-tail stability
    z_right_tail = np.linspace(4, 50, 500)
    phi_right_tail = phi(z_right_tail)
    ax[1].plot(z_right_tail, phi_right_tail, color="crimson", lw=2, label="phi(z)")
    # First z where phi == 0.0 (underflow)
    underflow = np.where(phi_right_tail == 0.0)[0]
    if len(underflow) > 0:
        z_underflow = z_right_tail[underflow[0]]
        ax[1].axvline(z_underflow, color="k", ls="--", label=f"Underflow at z ~ {z_underflow:.3f}")

    ax[1].set_yscale("log")
    ax[1].set_title("Right-tail (float64 stability)")
    ax[1].set_xlabel("z")
    ax[1].set_ylabel("phi(z) (log scale)")
    ax[1].grid(True, which="both", ls="--", alpha=0.6)
    ax[1].legend()

    plt.tight_layout()


def _log_F_over_K_test():
    # Generate vectors around F ~= K to test accuracy of log1p
    n = 1000
    K = np.linspace(1e5, 2e5, n)
    F = K * (1.0 + np.linspace(-1e-10, 1e-10, n))  # nearly equal

    ln_stable = _log_F_over_K(F, K)
    ln_direct = np.log(F / K)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Left: values
    ax[0].plot(K, ln_stable, color="blue", lw=2, label="Stable log(F/K)")
    ax[0].plot(K, ln_direct, color="orange", lw=1.2, alpha=0.7, label="Direct log(F/K)")
    ax[0].set_title("Numerically stable log(F/K)")
    ax[0].set_xlabel("K  (F ~= K)")
    ax[0].set_ylabel("ln(F/K)")
    ax[0].grid(True, ls="--", alpha=0.6)
    ax[0].legend()

    # Right: error
    diff = np.abs(ln_stable - ln_direct)
    ax[1].plot(K, diff, color="crimson", lw=1.8, label="|stable - direct|")
    ax[1].set_yscale("log")
    ax[1].set_title("Difference magnitude (float64 precision)")
    ax[1].set_xlabel("K")
    ax[1].set_ylabel("abs diff")
    ax[1].grid(True, which="both", ls="--", alpha=0.6)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def delta_call_test():
    rng = np.random.default_rng(42)

    n = 2000

    # S, K positive and comparable
    S = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=n)
    K = rng.lognormal(mean=np.log(100.0), sigma=0.25, size=n)

    # Rates / div yields
    r = rng.uniform(-0.01, 0.05, size=n)      # can be negative
    q = rng.uniform(0.00, 0.03, size=n)

    # Vols and maturities
    sigma = rng.uniform(0.05, 0.60, size=n)
    T = rng.uniform(1/365, 5.0, size=n)

    delta_vec = delta_call(S, K, r, q, sigma, T)

    # Reference with the same robust handling
    F = S * np.exp((r - q) * T)
    ln = _log_F_over_K(F, K)
    num = ln + 0.5 * (sigma * sigma) * T
    den = sigma * np.sqrt(T)
    d1_ref = np.empty_like(num)
    mask = den > 0.0
    np.divide(num, den, out=d1_ref, where=mask)
    d1_ref[~mask] = np.sign(num[~mask]) * np.inf
    delta_ref = np.exp(-q * T) * ndtr(d1_ref)

    err_abs = np.nanmax(np.abs(delta_vec - delta_ref))
    err_rel = np.nanmax(np.abs((delta_vec - delta_ref) / np.maximum(1e-16, np.abs(delta_ref))))
    print(f"[Random] max abs = {err_abs:.3e}, max rel = {err_rel:.3e}")

    # Sanity: finite and in [0,1]
    assert np.isfinite(delta_vec).all(), "NaNs/Infs in delta"
    assert (delta_vec >= -1e-15).all() and (delta_vec <= 1 + 1e-15).all(), "Delta out of [0,1]"

    # ATM stability
    n2 = 500
    K2 = np.linspace(1e5, 2e5, n2)
    F2 = K2 * (1.0 + np.linspace(-1e-10, 1e-10, n2))  # nearly equal
    r2, q2, T2, sigma2 = 0.01, 0.00, 0.5, 0.2
    S2 = F2 * np.exp(-(r2 - q2) * T2)

    delta_vec2 = delta_call(S2, K2, r2*np.ones_like(K2), q2*np.ones_like(K2), sigma2*np.ones_like(K2), T2*np.ones_like(K2))
    F2c = S2 * np.exp((r2 - q2) * T2)
    ln2 = _log_F_over_K(F2c, K2)
    num2 = ln2 + 0.5 * sigma2*sigma2 * T2
    den2 = sigma2 * np.sqrt(T2)
    d1_2 = np.sign(num2) * np.inf if den2 == 0 else num2/den2
    delta_ref2 = np.exp(-q2 * T2) * ndtr(d1_2)
    err_abs2 = np.max(np.abs(delta_vec2 - delta_ref2))
    print(f"[ATM] max abs = {err_abs2:.3e}")

    # Small T (including 0)
    T3 = np.zeros((n,))
    sigma3 = rng.uniform(0.1, 0.5, size=(n,))
    delta_vec3 = delta_call(S, K, r, q, sigma3, T3)
    F3 = S * np.exp((r - q) * T3)
    lim3 = np.exp(-q * T3) * (((F3 > K).astype(np.float64)) + 0.5*(F3 == K))
    print(f"[Small T incl. 0] max abs to limit = {np.max(np.abs(delta_vec3 - lim3)):.3e}")

    # Small sigma (including 0)
    sigma4 = np.zeros((n,))
    T4 = rng.uniform(0.1, 1.0, size=(n,))
    delta_vec4 = delta_call(S, K, r, q, sigma4, T4)
    F4 = S * np.exp((r - q) * T4)
    lim4 = np.exp(-q * T4) * (((F4 > K).astype(np.float64)) + 0.5*(F4 == K))
    print(f"[Small sigma incl. 0] max abs to limit = {np.max(np.abs(delta_vec4 - lim4)):.3e}")

    print("Delta tests completed.")


def self_check_gamma():
    cases = [
        dict(S=100, K=100, r=0,    q=0,    sigma=0.2,  T=1.0,      name="ATM S1"),
        dict(S=100, K=100, r=0,    q=0,    sigma=0.01, T=1/365.0,  name="Near-expiry low-vol S2"),
        dict(S=100, K=90,  r=0.01, q=0.02, sigma=0.35, T=0.5,      name="Generic"),
    ]

    RTOL = 1e-12
    ATOL = 1e-15

    all_ok = True
    for c in cases:
        try:
            g = Gamma(np.array([c['S']]),
                      np.array([c['K']]),
                      np.array([c['r']]),
                      np.array([c['q']]),
                      np.array([c['sigma']]),
                      np.array([c['T']]))[0]

            gr = gamma_ref(**c)
            rel = abs(g - gr) / max(abs(gr), 1e-16)

            status = "PASS" if (abs(g - gr) <= (ATOL + RTOL * abs(gr))) else "FAIL"
            if status == "PASS":
                print(f"[PASS] {c['name']}: gamma={g:.12e}, ref={gr:.12e}, rel.err={rel:.3e}")
            else:
                all_ok = False
                print(f"[FAIL] {c['name']}: gamma={g:.12e}, ref={gr:.12e}, rel.err={rel:.3e}  "
                      f"(tol: rtol={RTOL}, atol={ATOL})")

        except Exception:
            all_ok = False
            print(f"[ERROR] {c.get('name','<no-name>')} case crashed with exception:")
            print("  Params:", {k: v for k, v in c.items()})
            import traceback as _tb
            _tb.print_exc()

    if all_ok:
        print("All self-check gamma cases PASSED within tolerance "
              f"(rtol={RTOL}, atol={ATOL}).")
    else:
        print("Some gamma self-check cases FAILED. See details above.")


def main():
    # Uncomment any of the following to visualize
    # Phi_real_test()
    # phi_test()
    # _log_F_over_K_test()
    # plt.show()

    # Quick gamma self-checks
    self_check_gamma()


if __name__ == "__main__":
    main()
