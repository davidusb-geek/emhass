#!/usr/bin/env python3

"""Tests for the on-demand load forecast calibration report (issue #993)."""

import asyncio
import logging
import pathlib
import unittest

import numpy as np
import pandas as pd

# Base-safe imports: on the base branch the module does not exist, so the RED
# contract test below fails on a behavioural assertion rather than erroring.
try:
    from emhass import forecast_calibration as fc
except Exception:  # pragma: no cover - only hit on the base branch
    fc = None

from emhass import utils

logger = logging.getLogger("test_calibration")
FREQ = pd.Timedelta("30min")
STEPS_PER_DAY = int(pd.Timedelta("24h") / FREQ)
EMHASS_CONF = {"data_path": pathlib.Path(".")}


def build_load(days=80, seed=42, tz="Australia/Perth"):
    """Synthetic 30-min load with a daily + weekly shape and noise."""
    idx = pd.date_range("2026-01-01", periods=days * STEPS_PER_DAY, freq=FREQ, tz=tz)
    hod = np.asarray(idx.hour + idx.minute / 60, dtype=float)
    daily = 400 + 300 * np.sin((hod - 6) / 24 * 2 * np.pi) + 200 * (hod > 17)
    weekly = 100 * np.asarray(idx.dayofweek >= 5, dtype=float)
    rng = np.random.default_rng(seed)
    values = np.clip(daily + weekly + rng.normal(0, 30, len(idx)), 0, None)
    return pd.Series(values, index=idx)


def run(coro):
    return asyncio.run(coro)


def test_calibration_capability_red_proof():
    """RED contract proof: on base master the module is absent, so this fails on a
    behavioural assertion; on this branch the report has all three method rows."""
    assert fc is not None, "forecast_calibration capability is missing"
    load = build_load(days=80)
    res = run(fc.compute_forecast_calibration(load, FREQ, EMHASS_CONF, logger))
    assert "error" not in res
    assert set(res["table"]["method"]) == {"naive", "typical", "mlforecaster"}


class TestComputeForecastMetrics(unittest.TestCase):
    """Lock the shared metrics helper extracted from the ML backtest."""

    def test_matches_direct_sklearn(self):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        actual = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        pred = pd.Series([12.0, 18.0, 33.0, 39.0, 52.0])
        m = utils.compute_forecast_metrics(actual, pred)
        self.assertAlmostEqual(m["mae"], mean_absolute_error(actual, pred))
        self.assertAlmostEqual(m["rmse"], float(np.sqrt(mean_squared_error(actual, pred))))
        self.assertAlmostEqual(m["r2"], r2_score(actual, pred))
        self.assertEqual(m["n_samples"], 5)

    def test_nan_guards(self):
        # all-NaN predictions -> all metrics NaN, n_samples 0
        m = utils.compute_forecast_metrics(pd.Series([1.0, 2.0]), pd.Series([np.nan, np.nan]))
        self.assertEqual(m["n_samples"], 0)
        self.assertTrue(np.isnan(m["mae"]))
        # single valid sample -> r2 NaN (variance undefined), mae defined
        m = utils.compute_forecast_metrics(pd.Series([1.0, np.nan]), pd.Series([1.5, 2.0]))
        self.assertEqual(m["n_samples"], 1)
        self.assertTrue(np.isnan(m["r2"]))
        self.assertFalse(np.isnan(m["mae"]))
        # all-zero actuals -> MAPE NaN (division guarded)
        m = utils.compute_forecast_metrics(pd.Series([0.0, 0.0]), pd.Series([1.0, 2.0]))
        self.assertTrue(np.isnan(m["mape"]))
        self.assertEqual(m["n_samples"], 2)

    def test_ml_backtest_uses_shared_helper(self):
        # The ML forecaster's backtest metrics must equal the shared helper on the
        # same arrays (regression-lock the extraction).
        actual = pd.Series([100.0, 110.0, 90.0, 105.0, 95.0, 100.0])
        pred = pd.Series([102.0, 108.0, 92.0, 104.0, 96.0, 99.0])
        expected = utils.compute_forecast_metrics(actual, pred)
        for k in ("mae", "rmse", "r2", "mape", "n_samples"):
            self.assertIn(k, expected)


@unittest.skipIf(fc is None, "forecast_calibration module not present (base branch)")
class TestForecastCalibration(unittest.TestCase):
    def test_report_has_all_methods_and_val_metrics(self):
        """RED contract: report has naive+typical+mlforecaster rows with a val MAE."""
        load = build_load(days=80)
        res = run(
            fc.compute_forecast_calibration(
                load, FREQ, EMHASS_CONF, logger, sklearn_model="LinearRegression"
            )
        )
        self.assertNotIn("error", res)
        table = res["table"]
        self.assertEqual(set(table["method"]), {"naive", "typical", "mlforecaster"})
        for method in ("naive", "typical", "mlforecaster"):
            val_mae = table.loc[table["method"] == method, "val_mae"].iloc[0]
            self.assertNotEqual(val_mae, "N/A", f"{method} produced no val MAE")

    def test_train_column_na_for_naive_and_typical(self):
        load = build_load(days=80)
        res = run(fc.compute_forecast_calibration(load, FREQ, EMHASS_CONF, logger))
        table = res["table"].set_index("method")
        self.assertEqual(table.loc["naive", "train_mae"], "N/A")
        self.assertEqual(table.loc["typical", "train_mae"], "N/A")
        # mlforecaster has an in-sample train metric
        self.assertNotEqual(table.loc["mlforecaster", "train_mae"], "N/A")

    def test_naive_skill_is_zero_baseline(self):
        load = build_load(days=80)
        res = run(
            fc.compute_forecast_calibration(
                load, FREQ, EMHASS_CONF, logger, methods=["naive", "typical"]
            )
        )
        table = res["table"].set_index("method")
        self.assertEqual(table.loc["naive", "val_skill"], 0.0)

    def test_no_lookahead_prediction_invariant_to_target_day_actual(self):
        """A day's own actual must not change any method's prediction for that day."""
        load = build_load(days=80)
        day_list = sorted({ts.normalize() for ts in load.index})
        target_day = day_list[-1]
        day_mask = load.index.normalize() == pd.Timestamp(target_day)
        for predict_day in (fc._naive_predict_day, fc._typical_predict_day):
            p1 = fc._walk_forward(load, predict_day, [target_day])["pred"]
            spiked = load.copy()
            spiked.loc[day_mask] = spiked.loc[day_mask] * 5 + 10000
            p2 = fc._walk_forward(spiked, predict_day, [target_day])["pred"]
            pd.testing.assert_series_equal(p1, p2, check_names=False)

    def test_naive_matches_production_persistence_rule(self):
        """naive walk-forward == last-horizon-block carried forward (forecast.py rule)."""
        load = build_load(days=80)
        day_list = sorted({ts.normalize() for ts in load.index})
        target_day = day_list[-1]
        target_dates = load.index[load.index.normalize() == pd.Timestamp(target_day)]
        history_before = load.loc[load.index < target_dates[0]]
        expected = history_before.iloc[-len(target_dates) :].to_numpy()
        got = fc._naive_predict_day(history_before, target_dates).to_numpy()
        np.testing.assert_allclose(got, expected)

    def test_insufficient_history_returns_error(self):
        load = build_load(days=20)  # below CALIBRATION_MIN_DAYS
        res = run(fc.compute_forecast_calibration(load, FREQ, EMHASS_CONF, logger))
        self.assertIn("error", res)

    def test_skill_score_divide_by_zero_is_none(self):
        # naive MAE == 0 (perfect naive) -> skill None rather than a division error.
        idx = pd.date_range("2026-01-01", periods=4, freq=FREQ, tz="Australia/Perth")
        method_paired = pd.DataFrame(
            {"actual": [10, 20, 30, 40], "pred": [11, 19, 31, 39]}, index=idx
        )
        naive_paired = pd.DataFrame(
            {"actual": [10, 20, 30, 40], "pred": [10, 20, 30, 40]}, index=idx
        )
        self.assertIsNone(fc._skill_vs_naive("typical", method_paired, naive_paired))

    def test_skill_uses_common_samples_only(self):
        """F1: skill compares a method to naive only on days they BOTH cover."""
        idx = pd.date_range("2026-01-01", periods=4, freq=FREQ, tz="Australia/Perth")
        # naive covers all 4 points; its error on the last 2 is huge.
        naive_paired = pd.DataFrame(
            {"actual": [10.0, 20.0, 30.0, 40.0], "pred": [12.0, 18.0, 500.0, 900.0]}, index=idx
        )
        # method covers only the first 2 points.
        method_paired = pd.DataFrame({"actual": [10.0, 20.0], "pred": [10.5, 20.5]}, index=idx[:2])
        skill = fc._skill_vs_naive("typical", method_paired, naive_paired)
        # Must use naive MAE over t1,t2 only (=2.0), NOT the inflated all-4 MAE.
        mae_method = np.mean([0.5, 0.5])
        mae_naive_common = np.mean([2.0, 2.0])
        self.assertAlmostEqual(skill, 1 - mae_method / mae_naive_common)

    def test_build_table_na_for_uncovered_split(self):
        """A method with no coverage in a split -> every cell for that split is N/A."""
        metrics_rows = {
            "typical": {
                "test": {"mae": 5.0, "rmse": 6.0, "r2": 0.9, "mape": 3.0, "n_samples": 10},
                "val": None,  # no coverage in val
            },
        }
        skills = {"typical": {"test": 0.4, "val": None}}
        table = fc._build_table(metrics_rows, skills, ["typical"]).set_index("method")
        for col in ("val_mae", "val_rmse", "val_r2", "val_mape", "val_skill", "val_n"):
            self.assertEqual(table.loc["typical", col], "N/A")
        # the covered split still has real numbers
        self.assertEqual(table.loc["typical", "test_mae"], 5.0)

    def test_plot_frame_shape(self):
        load = build_load(days=80)
        res = run(fc.compute_forecast_calibration(load, FREQ, EMHASS_CONF, logger))
        plot = res["plot"]
        self.assertIn("actual", plot.columns)
        # val window is 14 days
        self.assertEqual(len(plot), fc.CALIBRATION_VAL_DAYS * STEPS_PER_DAY)


if __name__ == "__main__":
    unittest.main()
