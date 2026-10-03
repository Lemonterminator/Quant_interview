"""Numerical-method checks with regime-aware expectations.

Explicit order=2 retains the historical second-order CDF/stencil diagnostic.
P3-P6 convergence and propagation are checked in test_complex_cdf_order.py.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Python_prototype"))
from black_scholes import Gamma, Phi_t, delta_call, delta_cs, delta_forward, gamma_forward, phi  # noqa: E402


class NumericalAccuracyTests(unittest.TestCase):
    def test_delta_methods_at_resolvable_step(self):
        args = (100.0, 100.0, 0.0, 0.0, 0.2, 1.0)
        exact = float(delta_call(*args))
        self.assertLess(abs(float(delta_forward(*args, h=0.01)) - exact), 1e-4)
        self.assertLess(abs(float(delta_cs(*args, h=1e-8)) - exact), 1e-8)

    def test_forward_gamma_at_resolvable_step(self):
        args = (100.0, 100.0, 0.0, 0.0, 0.2, 1.0)
        exact = float(Gamma(*args))
        self.assertLess(abs(float(gamma_forward(*args, h=0.1)) - exact), 5e-4)

    def test_tiny_fd_step_is_unresolvable(self):
        self.assertEqual(100.0 + 1e-16, 100.0)

    def test_truncated_complex_cdf_changes_45_degree_order(self):
        # Apply the 45-degree formula directly to f=Phi, where f''=phi'.
        x = 0.3
        omega = (1 + 1j) / np.sqrt(2)
        truth = -x * float(phi(x))

        def estimate(h):
            return float(np.imag(Phi_t(x + omega * h, order=2) + Phi_t(x - omega * h, order=2)) / h**2)

        error_big = abs(estimate(0.04) - truth)
        error_small = abs(estimate(0.02) - truth)
        self.assertGreater(error_big / error_small, 3.5)
        self.assertLess(error_big / error_small, 4.5)


if __name__ == "__main__":
    unittest.main()
