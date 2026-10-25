"""Offline tests for the data pipeline normalisers and study statistics.

Run from the repository root::

    python -m unittest discover -s strategies/gold_qqq_cash/tests
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import data_pipeline as dp  # noqa: E402
import study_01_rolling_correlation as study  # noqa: E402


def _yahoo_frame(dates, tz="America/New_York"):
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(tz)
    n = len(idx)
    return pd.DataFrame({
        "Open": np.arange(1, n + 1, dtype=float), "High": 1.0, "Low": 1.0,
        "Close": np.arange(1, n + 1, dtype=float), "Adj Close": np.arange(1, n + 1, dtype=float) * 0.9,
        "Volume": 100, "Dividends": 0.0, "Stock Splits": 0.0,
    }, index=idx)


class NormalizeYahooTests(unittest.TestCase):
    def test_keeps_local_date_for_non_ny_timezone(self):
        raw = _yahoo_frame(["2024-03-01", "2024-03-04"], tz="America/Chicago")
        out = dp.normalize_yahoo(raw, now=datetime(2024, 3, 10, 12, tzinfo=dp.NY))
        self.assertIsNone(out.index.tz)
        self.assertEqual(list(out.index.strftime("%Y-%m-%d")), ["2024-03-01", "2024-03-04"])
        self.assertIn("adj_close", out.columns)
        self.assertIn("stock_splits", out.columns)

    def test_drops_duplicates_and_sorts(self):
        raw = _yahoo_frame(["2024-03-04", "2024-03-01", "2024-03-04"])
        out = dp.normalize_yahoo(raw, now=datetime(2024, 3, 10, 12, tzinfo=dp.NY))
        self.assertTrue(out.index.is_monotonic_increasing)
        self.assertFalse(out.index.has_duplicates)
        self.assertEqual(out.loc["2024-03-04", "close"], 3.0)  # keep="last"

    def test_drops_unfinished_session_only_before_close(self):
        raw = _yahoo_frame(["2024-03-07", "2024-03-08"])
        intraday = dp.normalize_yahoo(raw, now=datetime(2024, 3, 8, 11, tzinfo=dp.NY))
        after = dp.normalize_yahoo(raw, now=datetime(2024, 3, 8, 17, tzinfo=dp.NY))
        self.assertEqual(intraday.index[-1], pd.Timestamp("2024-03-07"))
        self.assertEqual(after.index[-1], pd.Timestamp("2024-03-08"))


class NormalizeFredTests(unittest.TestCase):
    def test_both_header_styles_and_missing_dots(self):
        for header in ("DATE", "observation_date"):
            text = f"{header},DFII10\n2024-01-02,1.5\n2024-01-03,.\n2024-01-04,\n2024-01-05,1.6\n"
            out = dp.normalize_fred(text, "DFII10")
            self.assertEqual(len(out), 2, header)
            self.assertAlmostEqual(out["close"].iloc[-1], 1.6)


class ValidateTests(unittest.TestCase):
    def test_rejects_non_positive_prices_but_not_rates(self):
        df = pd.DataFrame({"close": [1.0, -0.5], "adj_close": [1.0, -0.5]},
                          index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]))
        with self.assertRaises(ValueError):
            dp.validate(df, "x")
        dp.validate(df, "x", kind="rate")

    def test_flags_jumps_and_gaps(self):
        idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-20"])
        df = pd.DataFrame({"close": [100.0, 130.0, 131.0], "adj_close": [100.0, 130.0, 131.0], "volume": [1, 0, 1]},
                          index=idx)
        issues = dp.validate(df, "x")
        self.assertEqual(issues["gaps_over_7d"], ["2024-01-20"])
        self.assertIn("2024-01-03", issues["abs_log_return_over_15pct"])
        self.assertEqual(issues["zero_volume_days"], 1)

    def test_revision_detection_tolerates_vendor_noise(self):
        s = pd.Series([10.0, 11.0, 12.0], index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"]))
        anchors = dp._anchors(s, n=3)
        self.assertLess(dp._max_revision(s * (1 + 1e-6), anchors), dp.REVISION_TOL)
        self.assertGreater(dp._max_revision(s * 0.995, anchors), dp.REVISION_TOL)


class StatsTests(unittest.TestCase):
    def test_rolling_corr_matches_pandas(self):
        rng = np.random.default_rng(0)
        x, y = rng.standard_normal(300), rng.standard_normal(300)
        ours = study.rolling_corr(x, y, 20)
        ref = pd.Series(x).rolling(20).corr(pd.Series(y)).values
        np.testing.assert_allclose(ours[19:], ref[19:], atol=1e-10)
        self.assertTrue(np.isnan(ours[:19]).all())

    def test_critical_r(self):
        self.assertAlmostEqual(study.critical_r(60), np.tanh(1.96 / np.sqrt(57)))
        self.assertGreater(study.critical_r(20), study.critical_r(60))

    def test_significant_flips_ignore_neutral(self):
        state = np.array([1, 1, 0, 0, 1, 0, -1, -1, 0, -1, 1, np.nan])
        self.assertEqual(study.significant_flips(state[:, None])[0], 2)

    def test_robust_returns_have_no_lookahead(self):
        rng = np.random.default_rng(1)
        r = pd.Series(rng.standard_normal(200) * 0.01)
        base = study.robust_returns(r)
        bumped = r.copy()
        bumped.iloc[150] = 0.5
        changed = study.robust_returns(bumped)
        pd.testing.assert_series_equal(base.iloc[:150], changed.iloc[:150])
        self.assertLessEqual(changed.abs().max(), 3.0)


if __name__ == "__main__":
    unittest.main()
