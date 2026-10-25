"""Offline tests for the data pipeline normalisers and study statistics.

Run from the repository root::

    python -m unittest discover -s strategies/gold_qqq_cash/tests
"""

from __future__ import annotations

import sys
import tempfile
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

    def test_drops_future_dated_futures_bar_in_the_evening(self):
        # Sunday 20:00 ET: the futures session already prints a Monday bar.
        raw = _yahoo_frame(["2024-03-08", "2024-03-11"])
        out = dp.normalize_yahoo(raw, now=datetime(2024, 3, 10, 20, tzinfo=dp.NY))
        self.assertEqual(out.index[-1], pd.Timestamp("2024-03-08"))


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



class RevisionTests(unittest.TestCase):
    def setUp(self):
        idx = pd.bdate_range("2024-01-02", periods=50)
        self.old = pd.Series(100 * np.exp(np.cumsum(np.full(50, 0.001))), index=idx)

    def test_vendor_noise_is_not_a_revision(self):
        noisy = self.old * (1 + 1e-6 * np.sin(np.arange(50)))
        self.assertFalse(dp.revision_check(noisy, "price", old=self.old)["history_revised"])

    def test_dividend_rescale_is_not_a_revision(self):
        # A new dividend rescales every earlier adj_close by ~1e-3 (QQQ) to 3e-3 (BIL).
        rescaled = self.old * (1 - 0.003)
        for basis in ({"old": self.old}, {"anchors": dp._anchors(self.old)}):
            res = dp.revision_check(rescaled, "price", **basis)
            self.assertFalse(res["history_revised"], basis.keys())
            self.assertAlmostEqual(res["rescale_vs_previous"], -0.003, places=7)  # anchors keep 8 sig. digits

    def test_patched_print_is_found_on_every_date_not_only_anchors(self):
        patched = self.old.copy()
        patched.iloc[17] *= 1.01  # between anchors
        res = dp.revision_check(patched, "price", old=self.old)
        self.assertTrue(res["history_revised"])
        self.assertEqual(res["revised_dates"][0], self.old.index[17].date().isoformat())
        self.assertEqual(res["revision_basis"], "previous_cache")

    def test_rates_use_absolute_tolerance_and_survive_zero(self):
        idx = pd.bdate_range("2024-01-02", periods=4)
        old = pd.Series([0.0, 0.01, -0.05, 4.04], index=idx)
        same = dp.revision_check(old + 1e-9, "rate", old=old)
        self.assertFalse(same["history_revised"])
        moved = dp.revision_check(old.where(old != 0.0, 0.02), "rate", old=old)
        self.assertTrue(moved["history_revised"])
        self.assertTrue(np.isfinite(moved["max_revision"]))

    def test_no_reference_means_no_claim(self):
        self.assertEqual(dp.revision_check(self.old, "price"), {"revision_basis": None, "history_revised": False})


class RobustnessTests(unittest.TestCase):
    def test_atomic_write_keeps_old_file_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.csv"
            path.write_text("old\n")

            def broken(p):
                p.write_text("partial")
                raise OSError("disk full")

            with self.assertRaises(OSError):
                dp._atomic_write(path, broken)
            self.assertEqual(path.read_text(), "old\n")
            self.assertEqual(sorted(q.name for q in Path(tmp).iterdir()), ["x.csv"])

    def test_retry_gives_up_at_once_on_permanent_http_errors(self):
        class Resp:
            def __init__(self, status):
                self.status_code = status

        class HTTPError(Exception):
            def __init__(self, status):
                self.response = Resp(status)

        for status, expected_calls in ((404, 1), (429, 3), (503, 3)):
            calls = []

            def fn():
                calls.append(1)
                raise HTTPError(status)

            with self.subTest(status=status), self.assertRaises(HTTPError):
                dp._retry(fn, tries=3, base=0)
            self.assertEqual(len(calls), expected_calls)

    def test_asof_staleness_is_in_calendar_days(self):
        s = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2024-03-01", "2024-03-04"]))  # Fri, Mon
        idx = pd.bdate_range("2024-03-01", "2024-03-15")
        out = dp.asof_align(s, idx, max_stale_days=5)
        self.assertEqual(out.loc["2024-03-01"], 1.0)
        self.assertEqual(out.loc["2024-03-08"], 2.0)  # 4 calendar days old
        self.assertTrue(out.loc["2024-03-11":].isna().all())  # 7+ days old, only 5 rows later


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
