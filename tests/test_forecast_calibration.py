#!/usr/bin/env python3

"""Tests for the on-demand load forecast calibration report (issue #993)."""

import asyncio
import logging
import pathlib
import unittest
from unittest import mock

import numpy as np
import pandas as pd

# Base-safe imports: on the base branch the module does not exist, so the RED
# contract test below fails on a behavioural assertion rather than erroring.
try:
    from emhass import forecast_calibration as fc
except Exception:  # pragma: no cover - only hit on the base branch
    fc = None

from emhass import command_line, utils
from emhass.retrieve_hass import RetrieveHass

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
        """naive walk-forward on a full day == the previous day carried forward.

        The production rule (same time of day, ``Forecast.get_naive_load_forecast``)
        equals the last-horizon block when a one-day target starts right after the history.
        """
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

    def test_custom_val_days_change_report_window(self):
        """A non-default val_days must resize the report's val window, proving the
        runtime knob reaches the report rather than being ignored."""
        load = build_load(days=80)
        custom_val_days = 21
        self.assertNotEqual(custom_val_days, fc.CALIBRATION_VAL_DAYS)
        res = run(
            fc.compute_forecast_calibration(
                load, FREQ, EMHASS_CONF, logger, test_days=10, val_days=custom_val_days
            )
        )
        self.assertNotIn("error", res)
        self.assertEqual(len(res["plot"]), custom_val_days * STEPS_PER_DAY)
        start, end = res["val_window"]
        self.assertEqual((pd.Timestamp(end) - pd.Timestamp(start)).days, custom_val_days - 1)


VAR_LOAD = "sensor.power_load_no_var_loads"


class _StubHistoryRetrieveHass(RetrieveHass):
    """A real RetrieveHass whose backend retrieval returns a fixed raw frame, so the
    genuine prepare_data() runs without any HA / InfluxDB / VictoriaMetrics access."""

    def __init__(self, raw: pd.DataFrame):
        super().__init__("http://stub", "token", FREQ, "Australia/Perth", None, EMHASS_CONF, logger)
        self._raw = raw

    async def get_data(self, days_list, var_list, *args, **kwargs):
        self.df_final = self._raw.copy()
        self.var_list = var_list
        return True

    def prepare_data(self, *args, **kwargs):
        # Record exactly what generic preparation is handed (pre-trim or trimmed).
        self.prepare_input = self.df_final.copy()
        return super().prepare_data(*args, **kwargs)


def build_raw_history(days, leading_missing_days=0, first_value=None, gap=None):
    """Raw backend-shaped history (UTC, one column) as a time series DB returns it:
    NaN for every bucket before the sensor's first sample, optional internal gap."""
    load = build_load(days=days, tz="UTC")
    load.iloc[: leading_missing_days * STEPS_PER_DAY] = np.nan
    if first_value is not None:
        load.iloc[leading_missing_days * STEPS_PER_DAY] = first_value
    if gap is not None:
        load.iloc[gap] = np.nan
    return load.to_frame(VAR_LOAD)


def calibration_conf(**overrides):
    conf = {
        "sensor_power_load_no_var_loads": VAR_LOAD,
        "load_negative": False,
        "set_zero_min": True,
        "sensor_replace_zero": [],
        "sensor_linear_interp": [VAR_LOAD],
    }
    conf.update(overrides)
    return conf


def run_calibration_action(raw: pd.DataFrame, rh=None, **conf_overrides):
    """Run the /action/forecast-calibration entry point on ``raw`` and capture the
    load series actually handed to the calibration report."""
    input_data_dict = {
        "params": {"passed_data": {}},
        "retrieve_hass_conf": calibration_conf(**conf_overrides),
        "rh": rh or _StubHistoryRetrieveHass(raw),
        "emhass_conf": EMHASS_CONF,
    }
    with mock.patch.object(
        command_line,
        "compute_forecast_calibration",
        wraps=command_line.compute_forecast_calibration,
    ) as spy:
        result = run(command_line.forecast_calibration(input_data_dict, logger))
    return result, spy.call_args.args[0]


class TestCalibrationObservationBoundary(unittest.TestCase):
    """#1109: history before the first genuine observation is not realised 0 W load."""

    LEADING_DAYS = 15

    def test_leading_prehistory_excluded_zero_first_observation_sets_boundary(self):
        raw = build_raw_history(90, self.LEADING_DAYS, first_value=0.0)
        first_obs = raw.index[self.LEADING_DAYS * STEPS_PER_DAY]
        result, load = run_calibration_action(raw)
        self.assertIsNotNone(result)
        # A recorded 0 W is an observation: calibration history starts exactly there.
        self.assertEqual(load.index[0], first_obs)
        # Its value then follows the configured repair (set_zero_min NaNs it, the
        # sensor_linear_interp fill restores 0.0), not a calibration-specific rule.
        self.assertEqual(load.iloc[0], 0.0)
        self.assertEqual(str(load.index.tz), "Australia/Perth")
        self.assertEqual(load.index.freq, FREQ)
        n_days = len({ts.normalize() for ts in load.index})
        self.assertLessEqual(n_days, 90 - self.LEADING_DAYS + 1)

    def test_report_matches_backend_starting_at_first_observation(self):
        """Leading no-observation days must not change any train/test/val count or
        metric: the report equals the one from a backend that begins at the first
        sample (e.g. VictoriaMetrics, which omits leading buckets)."""
        raw = build_raw_history(90, self.LEADING_DAYS, first_value=0.0)
        trimmed = raw.iloc[self.LEADING_DAYS * STEPS_PER_DAY :]
        with_prehistory, _ = run_calibration_action(raw)
        without_prehistory, _ = run_calibration_action(trimmed)
        pd.testing.assert_frame_equal(with_prehistory["table"], without_prehistory["table"])
        self.assertEqual(with_prehistory["val_window"], without_prehistory["val_window"])

    def test_insufficient_eligible_history_after_trim(self):
        # 70 requested days >= CALIBRATION_MIN_DAYS, but only 55 observed.
        raw = build_raw_history(70, self.LEADING_DAYS)
        self.assertGreaterEqual(70, fc.CALIBRATION_MIN_DAYS)
        self.assertLess(70 - self.LEADING_DAYS, fc.CALIBRATION_MIN_DAYS)
        result, _ = run_calibration_action(raw)
        self.assertIsNone(result)

    def test_no_observation_at_all_fails_cleanly(self):
        raw = build_raw_history(90, leading_missing_days=90)
        input_data_dict = {
            "params": {"passed_data": {}},
            "retrieve_hass_conf": {
                "sensor_power_load_no_var_loads": VAR_LOAD,
                "sensor_linear_interp": [VAR_LOAD],
            },
            "rh": _StubHistoryRetrieveHass(raw),
            "emhass_conf": EMHASS_CONF,
        }
        with mock.patch.object(command_line, "compute_forecast_calibration") as spy:
            result = run(command_line.forecast_calibration(input_data_dict, logger))
        self.assertIsNone(result)
        # The all-missing window must never reach the report as synthetic 0 W history.
        spy.assert_not_called()

    def test_zero_boundary_then_configured_set_zero_min_without_repair(self):
        """A recorded 0 W sets the boundary, and afterwards set_zero_min keeps its
        meaning when the load is in neither repair list. Guards against both finding
        the boundary with ``value != 0`` and overriding set_zero_min for calibration."""
        raw = build_raw_history(90, self.LEADING_DAYS, first_value=0.0)
        boundary = self.LEADING_DAYS * STEPS_PER_DAY
        conf = {"set_zero_min": True, "sensor_replace_zero": [], "sensor_linear_interp": []}
        rh = _StubHistoryRetrieveHass(raw)
        _, load = run_calibration_action(raw, rh=rh, **conf)
        # A. The raw 0.0 is the first row generic preparation ever sees.
        self.assertEqual(rh.prepare_input.index[0], raw.index[boundary])
        self.assertEqual(rh.prepare_input[VAR_LOAD].iloc[0], 0.0)
        self.assertEqual(len(rh.prepare_input), len(raw) - boundary)
        self.assertEqual(load.index[0], raw.index[boundary])
        # B. Afterwards the load is exactly what prepare_data() yields for that
        # configuration on the eligible history, so the zero becomes missing as usual.
        reference = _StubHistoryRetrieveHass(raw)
        reference.df_final = raw.iloc[boundary:].copy()
        reference.var_list = [VAR_LOAD]
        reference.prepare_data(
            VAR_LOAD,
            load_negative=False,
            var_replace_zero=[],
            var_interp=[],
            set_zero_min=True,
            skip_renaming=True,
        )
        pd.testing.assert_series_equal(load, reference.df_final[VAR_LOAD])
        self.assertTrue(np.isnan(load.iloc[0]))

    def test_internal_gap_still_uses_configured_interpolation(self):
        start = self.LEADING_DAYS * STEPS_PER_DAY
        gap = slice(start + 40 * STEPS_PER_DAY, start + 40 * STEPS_PER_DAY + 4)
        raw = build_raw_history(90, self.LEADING_DAYS, gap=gap)
        _, load = run_calibration_action(raw)
        before, after = raw.index[gap.start - 1], raw.index[gap.stop]
        expected = np.linspace(raw.loc[before, VAR_LOAD], raw.loc[after, VAR_LOAD], 6)[1:-1]
        np.testing.assert_allclose(load.loc[raw.index[gap]].to_numpy(), expected)

    def test_complete_history_unchanged(self):
        raw = build_raw_history(90)
        _, load = run_calibration_action(raw)
        self.assertEqual(len(load), len(raw))
        self.assertEqual(load.index[0], raw.index[0])

    def test_helper_boundary_uses_missingness_not_value(self):
        raw = build_raw_history(5, leading_missing_days=2, first_value=0.0)
        trimmed = fc.trim_to_first_observation(raw, VAR_LOAD)
        self.assertEqual(trimmed.index[0], raw.index[2 * STEPS_PER_DAY])
        self.assertEqual(trimmed[VAR_LOAD].iloc[0], 0.0)
        self.assertEqual(trimmed.index.freq, raw.index.freq)
        self.assertEqual(trimmed.index.tz, raw.index.tz)
        # An owned frame, since prepare_data() then modifies df_final in place.
        self.assertFalse(np.shares_memory(trimmed[VAR_LOAD].to_numpy(), raw[VAR_LOAD].to_numpy()))
        complete = build_raw_history(5)
        pd.testing.assert_frame_equal(fc.trim_to_first_observation(complete, VAR_LOAD), complete)
        self.assertIsNone(fc.trim_to_first_observation(raw.iloc[:10], VAR_LOAD))
        self.assertIsNone(fc.trim_to_first_observation(raw, "sensor.absent"))


if __name__ == "__main__":
    unittest.main()
