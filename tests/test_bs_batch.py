"""Contract tests for the Python facade and the cross-language golden CSV."""

from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Python_prototype"))
from bs_batch import evaluate_call  # noqa: E402


class BatchContractTests(unittest.TestCase):
    def test_golden_cases(self):
        with (ROOT / "testdata" / "bs_call_golden.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertGreaterEqual(len(rows), 8)
        args = [np.array([float(row[name]) for row in rows]) for name in ("S", "K", "r", "q", "sigma", "T")]
        result = evaluate_call(*args)
        for name in ("price", "delta", "gamma"):
            expected = np.array([float(row[name]) for row in rows])
            tolerance = np.where(
                np.abs(expected) < 1e-4,
                1e-15 + 1e-8 * np.abs(expected),
                5e-13 + 5e-11 * np.abs(expected),
            )
            np.testing.assert_array_less(
                np.abs(getattr(result, name) - expected), tolerance,
                err_msg=f"{name}: golden case order is {[row['case'] for row in rows]}",
            )

    def test_scalar_and_broadcast_shapes(self):
        scalar = evaluate_call(100, 100, 0, 0, 0.2, 1)
        self.assertEqual(scalar.price.shape, ())
        grid = evaluate_call(np.array([[90], [110]]), np.array([[80, 100, 120]]), 0.01, 0, 0.2, 1)
        self.assertEqual(grid.price.shape, (2, 3))
        for i, s in enumerate((90, 110)):
            for j, k in enumerate((80, 100, 120)):
                one = evaluate_call(s, k, 0.01, 0, 0.2, 1)
                for name in ("price", "delta", "gamma"):
                    self.assertAlmostEqual(float(getattr(grid, name)[i, j]), float(getattr(one, name)), places=12)

    def test_empty_and_input_immutability(self):
        spots = np.array([80.0, 100.0, 120.0])
        before = spots.copy()
        evaluate_call(spots, 100, 0, 0, 0.2, 1)
        np.testing.assert_array_equal(spots, before)
        self.assertEqual(evaluate_call(np.array([]), 100, 0, 0, 0.2, 1).price.size, 0)

    def test_invalid_inputs(self):
        for bad in (-1.0, 0.0, np.nan, np.inf):
            with self.subTest(S=bad), self.assertRaises(ValueError):
                evaluate_call(bad, 100, 0, 0, 0.2, 1)
        for sigma, expiry in ((-0.1, 1), (0.2, -1)):
            with self.subTest(sigma=sigma, T=expiry), self.assertRaises(ValueError):
                evaluate_call(100, 100, 0, 0, sigma, expiry)
        with self.assertRaises(ValueError):
            evaluate_call(np.ones(2), np.ones(3), 0, 0, 0.2, 1)

    def test_degenerate_midpoint_convention(self):
        result = evaluate_call(np.array([90, 100, 110]), 100, 0, 0, 0, 1)
        np.testing.assert_array_equal(result.price, [0, 0, 10])
        np.testing.assert_array_equal(result.delta, [0, 0.5, 1])
        np.testing.assert_array_equal(result.gamma, [0, 0, 0])

    def test_metamorphic_properties(self):
        rng = np.random.default_rng(20261003)
        s = rng.uniform(70, 130, 256)
        k = rng.uniform(70, 130, 256)
        r = rng.uniform(-0.02, 0.05, 256)
        q = rng.uniform(0, 0.04, 256)
        sigma = rng.uniform(0.08, 0.7, 256)
        t = rng.uniform(0.05, 3, 256)
        base = evaluate_call(s, k, r, q, sigma, t)
        scaled = evaluate_call(7 * s, 7 * k, r, q, sigma, t)
        np.testing.assert_allclose(scaled.price, 7 * base.price, rtol=2e-11, atol=2e-11)
        np.testing.assert_allclose(scaled.delta, base.delta, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(scaled.gamma, base.gamma / 7, rtol=2e-11, atol=2e-11)
        a = s * np.exp(-q * t)
        np.testing.assert_array_less(np.maximum(a - k * np.exp(-r * t), 0) - 1e-10, base.price)
        np.testing.assert_array_less(base.price, a + 1e-10)
        higher_vol = evaluate_call(s, k, r, q, sigma + 0.1, t)
        np.testing.assert_array_less(base.price - 1e-10, higher_vol.price)


if __name__ == "__main__":
    unittest.main()
