"""Regression tests for ML load forecast DST sizing and UTC alignment (#1248).

The test scope deliberately avoids any Home Assistant live writes. It exercises
the optimization-facing ML path with controlled model predictions and history.
"""

import logging
import pathlib
import pickle
import tempfile
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import orjson
import pandas as pd
import pytz
import time_machine

from emhass import forecast as forecast_module
from emhass import utils
from emhass.command_line import set_input_data_dict
from emhass.forecast import Forecast
from emhass.machine_learning_forecaster import MLForecaster
from emhass.retrieve_hass import RetrieveHass
from tests import test_command_line_utils as command_line_tests

emhass_conf = command_line_tests.emhass_conf

logger, _ = utils.get_logger(__name__, emhass_conf, save_to_file=False)
TZ_NAME = "Europe/Berlin"


class _Predictor:
    """Minimal fitted ML model exposing the real prediction-index contract."""

    def __init__(self, *, shift=pd.Timedelta(0), ignore_steps=False, weather=False):
        self.num_lags = 72
        self.is_tuned = False
        self.forecaster = SimpleNamespace(window_size=72)
        self.weather_features = ["temp_air"] if weather else []
        self.shift = shift
        self.ignore_steps = ignore_steps
        self.received_steps = None
        self.received_history_end = None
        self.received_weather = None

    async def predict(self, data_last_window, weather_future=None, steps=None):
        self.received_steps = steps
        self.received_history_end = data_last_window.index[-1]
        self.received_weather = weather_future
        n = self.num_lags if self.ignore_steps else steps
        index = pd.date_range(
            start=data_last_window.index[-1] + data_last_window.index.freq + self.shift,
            periods=n,
            freq=data_last_window.index.freq,
        )
        return pd.Series(np.arange(n, dtype=float), index=index)


class TestMlDstAndTimestampAlignment(unittest.IsolatedAsyncioTestCase):
    """Forecast horizon is a calendar-day grid; ML lags are not the horizon."""

    TZ = pytz.timezone("Europe/Berlin")
    FREQ = pd.Timedelta(hours=1)

    @classmethod
    def _case(cls, start):
        start = pd.Timestamp(start, tz=cls.TZ)
        end = start + pd.DateOffset(days=3)
        grid = pd.date_range(start=start, end=end, freq=cls.FREQ, inclusive="left")
        fcst = object.__new__(Forecast)
        fcst.forecast_dates = grid
        fcst.time_zone = cls.TZ
        fcst.freq = cls.FREQ
        fcst.var_load = "sensor.test_load"
        fcst.var_load_new = "sensor.test_load_positive"
        fcst.retrieve_hass_conf = {"sensor_power_load_no_var_loads": fcst.var_load}
        fcst.logger = logging.getLogger(__name__)
        history_index = pd.date_range(end=start - cls.FREQ, periods=100, freq=cls.FREQ)
        history = pd.DataFrame(
            {fcst.var_load_new: np.arange(len(history_index), dtype=float) + 500},
            index=history_index,
        )
        return fcst, history

    async def test_calendar_day_horizons_not_model_lag_counts(self):
        for date, count in (
            ("2026-02-20 12:00", 72),
            ("2026-03-28 12:00", 71),
            ("2026-10-24 12:00", 73),
        ):
            with self.subTest(date=date):
                fcst, history = self._case(date)
                self.assertEqual(len(fcst.forecast_dates), count)
                model = _Predictor()
                result = await fcst._get_load_forecast_ml(history, True, model, True)
                self.assertIsInstance(result, pd.DataFrame)
                self.assertEqual(model.received_steps, count)
                self.assertEqual(len(result), count)
                self.assertTrue(result.index.equals(fcst.forecast_dates))
                self.assertTrue(result["yhat"].notna().all())

    async def test_current_forecast_bucket_excluded_from_training_window(self):
        fcst, history = self._case("2026-10-24 12:00")
        history.loc[fcst.forecast_dates[0], fcst.var_load_new] = -12345
        history = history.sort_index()
        model = _Predictor()
        result = await fcst._get_load_forecast_ml(history, True, model, True)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(model.received_history_end, fcst.forecast_dates[0] - self.FREQ)
        self.assertEqual(model.received_steps, 73)

    async def test_missing_last_completed_bucket_rejected(self):
        fcst, history = self._case("2026-02-20 12:00")
        model = _Predictor()
        result = await fcst._get_load_forecast_ml(history.iloc[:-1], True, model, True)
        self.assertIs(result, False)
        self.assertIsNone(model.received_steps)

    async def test_irregular_recent_history_rejected(self):
        fcst, history = self._case("2026-02-20 12:00")
        model = _Predictor()
        result = await fcst._get_load_forecast_ml(
            history.drop(history.index[-5]), True, model, True
        )
        self.assertIs(result, False)
        self.assertIsNone(model.received_steps)

    async def test_shifted_predictions_rejected_even_with_matching_length(self):
        fcst, history = self._case("2026-02-20 12:00")
        model = _Predictor(shift=self.FREQ)
        result = await fcst._get_load_forecast_ml(history, True, model, True)
        self.assertIs(result, False)

    async def test_predictor_ignoring_autumn_extra_step_rejected(self):
        fcst, history = self._case("2026-10-24 12:00")
        model = _Predictor(ignore_steps=True)
        result = await fcst._get_load_forecast_ml(history, True, model, True)
        self.assertIs(result, False)

    async def test_weather_covariates_follow_extended_autumn_grid(self):
        fcst, history = self._case("2026-10-24 12:00")
        observed_index = []

        async def weather_for(index, features):
            observed_index.append(index)
            self.assertEqual(features, ["temp_air"])
            return pd.DataFrame({"temp_air": np.full(len(index), 18.0)}, index=index)

        fcst.get_weather_covariates = weather_for
        model = _Predictor(weather=True)
        result = await fcst._get_load_forecast_ml(history, True, model, True)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(model.received_weather), 73)
        self.assertEqual(len(observed_index), 1)
        self.assertTrue(
            observed_index[0].tz_convert("UTC").equals(fcst.forecast_dates.tz_convert("UTC"))
        )

    async def test_standalone_ml_predict_uses_lag_count_without_override(self):
        fcst, history = self._case("2026-10-24 12:00")
        observations = history.rename(columns={fcst.var_load_new: fcst.var_load})
        mlf = object.__new__(MLForecaster)
        mlf.is_tuned = False
        mlf.num_lags = 72
        mlf.var_model = fcst.var_load
        mlf.weather_features = []
        mlf.logger = logging.getLogger(__name__)

        def fake_skforecast_predict(*, steps, last_window, exog):
            self.assertEqual(len(last_window), len(observations))
            index = pd.date_range(
                start=last_window.index[-1] + last_window.index.freq,
                periods=steps,
                freq=last_window.index.freq,
            )
            return pd.Series(np.full(steps, 25.0), index=index)

        mlf.forecaster = SimpleNamespace(predict=fake_skforecast_predict)
        old = await mlf.predict(observations)
        self.assertEqual(len(old), 72)
        adjusted = await mlf.predict(observations, steps=73)
        self.assertEqual(len(adjusted), 73)


class TestHistoryBucketPhase(unittest.IsolatedAsyncioTestCase):
    """Zones with a non-whole-hour UTC offset put history buckets on another phase."""

    @staticmethod
    def _case(tz, freq, start_utc, history_offset):
        tz = pytz.timezone(tz)
        freq = pd.Timedelta(freq)
        start = pd.Timestamp(start_utc, tz="UTC").tz_convert(tz)
        fcst = object.__new__(Forecast)
        fcst.forecast_dates = pd.date_range(start=start, periods=6, freq=freq)
        fcst.time_zone = tz
        fcst.freq = freq
        fcst.var_load = "sensor.test_load"
        fcst.var_load_new = "sensor.test_load_positive"
        fcst.retrieve_hass_conf = {"sensor_power_load_no_var_loads": fcst.var_load}
        fcst.logger = logging.getLogger(__name__)
        index = pd.date_range(end=start - freq - history_offset, periods=100, freq=freq)
        history = pd.DataFrame({fcst.var_load_new: np.arange(100, dtype=float)}, index=index)
        return fcst, history

    async def test_sub_step_phase_is_aligned_to_the_grid(self):
        for tz, freq, offset in (
            ("Asia/Kolkata", "60min", "30min"),
            ("Asia/Kathmandu", "30min", "15min"),
            ("Asia/Kathmandu", "60min", "45min"),
        ):
            with self.subTest(tz=tz, freq=freq, offset=offset):
                fcst, history = self._case(tz, freq, "2026-02-20 10:00", pd.Timedelta(offset))
                model = _Predictor()
                result = await fcst._get_load_forecast_ml(history, True, model, True)
                self.assertIsInstance(result, pd.DataFrame)
                self.assertTrue(result.index.equals(fcst.forecast_dates))
                self.assertEqual(model.received_steps, 6)

    async def test_full_step_lag_is_stale_whatever_the_phase(self):
        fcst, history = self._case("Asia/Kolkata", "60min", "2026-02-20 10:00", pd.Timedelta("1h"))
        model = _Predictor()
        with self.assertLogs(logger, "ERROR") as logs:
            self.assertIs(await fcst._get_load_forecast_ml(history, True, model, True), False)
        self.assertIn("missing, stale or irregular", "\n".join(logs.output))

    async def test_offset_off_every_bucket_lattice_is_stale_not_phase(self):
        # 20 minutes late is on neither the UTC nor the local hourly lattice, so it
        # cannot be a backend bucket phase: it must not be hidden as one.
        fcst, history = self._case(
            "Asia/Kolkata", "60min", "2026-02-20 10:00", pd.Timedelta("20min")
        )
        with self.assertLogs(logger, "ERROR") as logs:
            self.assertIs(
                await fcst._get_load_forecast_ml(history, True, _Predictor(), True), False
            )
        self.assertIn("missing, stale or irregular", "\n".join(logs.output))

    async def test_completed_overlapping_bucket_is_not_a_negative_phase(self):
        # REST/InfluxDB/VictoriaMetrics buckets sit on UTC multiples and run up to the
        # bucket containing now; the grid start is wall-aligned ("first") or rounded.
        for tz, freq, now, start in (
            ("Asia/Kolkata", "60min", "2026-02-20 05:10", "2026-02-20 04:30"),
            ("Asia/Kolkata", "60min", "2026-02-20 05:25", "2026-02-20 05:30"),
            ("Asia/Kathmandu", "30min", "2026-02-20 10:35", "2026-02-20 10:15"),
        ):
            with self.subTest(tz=tz, now=now):
                fcst, _ = self._case(tz, freq, start, pd.Timedelta(0))
                utc_now = pd.Timestamp(now, tz="UTC")
                index = pd.date_range(end=utc_now.floor(freq), periods=100, freq=freq)
                history = pd.DataFrame(
                    {fcst.var_load_new: np.arange(100, dtype=float)}, index=index.tz_convert(tz)
                )
                model = _Predictor()
                with time_machine.travel(utc_now.to_pydatetime(), tick=False):
                    result = await fcst._get_load_forecast_ml(history, True, model, True)
                self.assertIsInstance(result, pd.DataFrame)
                self.assertTrue(result.index.equals(fcst.forecast_dates))
                # Only completed buckets that end by the first needed step are observations.
                self.assertLessEqual(
                    model.received_history_end + pd.Timedelta(freq), fcst.forecast_dates[0]
                )

    async def test_history_ahead_of_the_grid_is_rejected(self):
        fcst, history = self._case(
            "Asia/Kolkata", "60min", "2026-02-20 10:00", -pd.Timedelta("20min")
        )
        # Buckets labelled later than the last completed grid step cannot be placed.
        self.assertIs(await fcst._get_load_forecast_ml(history, True, _Predictor(), True), False)


class TestRestUnchangedSensorState(unittest.TestCase):
    """REST history has one row per state change; its last row is the current state."""

    NOW = datetime(2026, 2, 20, 10, 20, tzinfo=UTC)

    def _process(self, rows, is_last_day=True, day="2026-02-20", now=None):
        rh = object.__new__(RetrieveHass)
        rh.freq = pd.Timedelta("60min")
        rh.logger = logging.getLogger(__name__)
        data = [{"state": state, "last_changed": at} for state, at in rows]
        with time_machine.travel(now or self.NOW, tick=False):
            return rh._process_history_dataframe(
                data, "sensor.load", pd.Timestamp(day, tz="UTC"), False, is_last_day
            )

    def test_unchanged_numeric_state_is_held_to_the_last_bucket_before_now(self):
        frame = self._process(
            [("500", "2026-02-20T00:00:00+00:00"), ("520", "2026-02-20T06:10:00+00:00")]
        )
        self.assertEqual(frame.index[-1], pd.Timestamp("2026-02-20 10:00", tz="UTC"))
        self.assertTrue((frame["sensor.load"].iloc[6:] == 520.0).all())

    def test_unavailable_last_state_is_not_held(self):
        frame = self._process(
            [("500", "2026-02-20T00:00:00+00:00"), ("unavailable", "2026-02-20T06:10:00+00:00")]
        )
        self.assertEqual(frame.index[-1], pd.Timestamp("2026-02-20 06:00", tz="UTC"))

    def test_unavailable_after_a_number_in_the_same_bucket_is_not_held(self):
        frame = self._process(
            [("5000", "2026-02-20T06:05:00+00:00"), ("unavailable", "2026-02-20T06:20:00+00:00")]
        )
        self.assertEqual(frame.index[-1], pd.Timestamp("2026-02-20 06:00", tz="UTC"))

    def test_held_value_is_the_last_state_not_the_bucket_mean(self):
        frame = self._process(
            [("1000", "2026-02-20T06:00:00+00:00"), ("0", "2026-02-20T06:25:00+00:00")]
        )
        self.assertEqual(frame.index[-1], pd.Timestamp("2026-02-20 10:00", tz="UTC"))
        self.assertTrue((frame["sensor.load"].iloc[-2:] == 0.0).all())

    def test_no_hold_just_after_utc_midnight(self):
        frame = self._process(
            [("500", "2026-02-20T00:00:00+00:00")], now=datetime(2026, 2, 20, 0, 20, tzinfo=UTC)
        )
        self.assertEqual(frame.index[-1], pd.Timestamp("2026-02-20 00:00", tz="UTC"))

    def test_past_days_and_old_data_are_never_extended(self):
        rows = [("500", "2026-02-19T00:00:00+00:00"), ("520", "2026-02-19T06:10:00+00:00")]
        self.assertEqual(
            self._process(rows, is_last_day=False, day="2026-02-19").index[-1],
            pd.Timestamp("2026-02-19 06:00", tz="UTC"),
        )
        self.assertEqual(
            self._process(rows, day="2026-02-20").index[-1],
            pd.Timestamp("2026-02-19 06:00", tz="UTC"),
        )


def _utc_sine(index):
    """A load profile that is a pure function of the absolute (UTC) instant."""
    hours = np.asarray(index.tz_convert("UTC").asi8, dtype=float) / 3.6e12
    return 500.0 + 200.0 * np.sin(2 * np.pi * hours / 24.0)


class _RecorderRetrieveHass:
    """RetrieveHass stand-in replaying a recorder around the frozen clock.

    The history ends at the (partial) bucket containing "now", like a live Home
    Assistant retrieval. ``drop_last`` simulates a stale recorder and ``gap_back``
    a missing interval inside the model's last window.
    """

    days = 6
    drop_last = 0
    gap_back = None

    def __init__(self, hass_url, long_lived_token, freq, time_zone, params, emhass_conf, logger):
        self.freq = freq
        self.time_zone = time_zone
        self.df_final = None

    async def get_data(self, days_list, var_list, **kwargs):
        self._var = var_list[0]
        end = pd.Timestamp.now(tz="UTC").floor(self.freq)
        index = pd.date_range(end=end, periods=self.days * 24, freq=self.freq).tz_convert(
            self.time_zone
        )
        values = _utc_sine(index)
        values[-1] = -9999.0  # the still-running bucket holds a partial value
        frame = pd.DataFrame({self._var: values}, index=index)
        if self.drop_last:
            frame = frame.iloc[: -self.drop_last]
        if self.gap_back is not None:
            frame = frame.drop(frame.index[-self.gap_back])
        self.df_final = frame
        return True

    def prepare_data(self, var_load, **kwargs):
        self.df_final = self.df_final.rename(columns={self._var: var_load + "_positive"})
        return True


class TestDayaheadOptimMlForecasterPath(unittest.IsolatedAsyncioTestCase):
    """The production path: set_input_data_dict -> Forecast.get_load_forecast -> pickled model."""

    CASES = (
        ("ordinary", datetime(2026, 2, 20, 11, 20, tzinfo=UTC), 72),
        ("spring", datetime(2026, 3, 28, 11, 20, tzinfo=UTC), 71),
        ("autumn", datetime(2026, 10, 24, 10, 20, tzinfo=UTC), 73),
    )
    MODEL_TYPE = "dst_alignment_e2e"

    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.conf = dict(emhass_conf)
        self.conf["data_path"] = pathlib.Path(self._tmp.name)
        _RecorderRetrieveHass.drop_last = 0
        _RecorderRetrieveHass.gap_back = None
        _RecorderRetrieveHass.days = 6

    async def _save_model(self, *, num_lags=72, tuned_lags=None, weather=False):
        index = pd.date_range("2026-01-01", periods=24 * 20, freq="1h", tz=TZ_NAME)
        data = pd.DataFrame({"sensor.test_load": _utc_sine(index)}, index=index)
        kwargs = {}
        if weather:
            data["temp_air"] = 10.0 + np.arange(len(index)) % 5
            kwargs["weather_features"] = ["temp_air"]
        mlf = MLForecaster(
            data,
            self.MODEL_TYPE,
            "sensor.test_load",
            "LinearRegression",
            num_lags,
            self.conf,
            logger,
            **kwargs,
        )
        await mlf.fit()
        if tuned_lags is not None:
            mlf.is_tuned = True
            mlf.lags_opt = tuned_lags
        with open(self.conf["data_path"] / f"{self.MODEL_TYPE}_mlf.pkl", "wb") as f:
            pickle.dump(mlf, f)

    async def _setup(self, now, method_ts_round="nearest"):
        params = await command_line_tests.TestCommandLineAsyncUtils.get_test_params()
        params["retrieve_hass_conf"]["time_zone"] = TZ_NAME
        params["retrieve_hass_conf"]["optimization_time_step"] = 60
        params["retrieve_hass_conf"]["sensor_power_load_no_var_loads"] = "sensor.test_load"
        params["retrieve_hass_conf"]["method_ts_round"] = method_ts_round
        params["optim_conf"]["delta_forecast_daily"] = 3
        params["optim_conf"]["set_use_pv"] = False
        params["optim_conf"]["load_forecast_method"] = "mlforecaster"
        params["optim_conf"]["set_use_battery"] = False
        runtime = {"model_type": self.MODEL_TYPE}

        async def no_ha_config(self_):
            self_.ha_config = {}
            return True

        with (
            time_machine.travel(now, tick=False),
            patch.object(forecast_module, "RetrieveHass", _RecorderRetrieveHass),
            patch.object(RetrieveHass, "get_ha_config", no_ha_config),
        ):
            return await set_input_data_dict(
                self.conf,
                "profit",
                orjson.dumps(params).decode(),
                orjson.dumps(runtime).decode(),
                "dayahead-optim",
                logger,
                get_data_from_file=False,
            )

    async def test_calendar_day_horizon_and_utc_alignment(self):
        await self._save_model()
        for name, now, steps in self.CASES:
            with self.subTest(case=name):
                idd = await self._setup(now)
                self.assertTrue(idd, "dayahead-optim input preparation failed")
                frame = idd["df_input_data_dayahead"]
                self.assertEqual(len(frame), steps)
                grid = idd["fcst"].forecast_dates
                self.assertTrue(frame.index.tz_convert("UTC").equals(grid.tz_convert("UTC")))
                # One hour of shift moves this profile by up to ~52 W.
                np.testing.assert_allclose(frame["p_load_forecast"], _utc_sine(grid), atol=15.0)

    async def test_rounding_modes_align_to_the_grid(self):
        await self._save_model()
        for mode in ("nearest", "first", "last"):
            for minute in (20, 40):
                with self.subTest(mode=mode, minute=minute):
                    now = datetime(2026, 10, 24, 10, minute, tzinfo=UTC)
                    idd = await self._setup(now, method_ts_round=mode)
                    self.assertTrue(idd)
                    grid = idd["fcst"].forecast_dates
                    frame = idd["df_input_data_dayahead"]
                    self.assertEqual(len(frame), 73)
                    np.testing.assert_allclose(frame["p_load_forecast"], _utc_sine(grid), atol=15.0)

    async def test_tuned_model_horizon_independent_of_lags_opt(self):
        await self._save_model(tuned_lags=24)
        idd = await self._setup(self.CASES[2][1])
        self.assertTrue(idd)
        self.assertEqual(len(idd["df_input_data_dayahead"]), 73)

    async def test_weather_covariate_model_on_autumn_grid(self):
        await self._save_model(weather=True)
        seen = []

        async def weather_for(self_, index, features):
            seen.append(index)
            return pd.DataFrame({"temp_air": np.full(len(index), 12.0)}, index=index)

        with patch.object(Forecast, "get_weather_covariates", weather_for):
            idd = await self._setup(self.CASES[2][1])
        self.assertTrue(idd)
        self.assertEqual(len(idd["df_input_data_dayahead"]), 73)
        self.assertEqual({len(i) for i in seen}, {73})

    async def test_recorder_without_partial_bucket_is_accepted(self):
        await self._save_model()
        _RecorderRetrieveHass.drop_last = 1  # only the still-running bucket is absent
        self.assertTrue(await self._setup(self.CASES[0][1]))

    async def test_stale_history_fails_closed(self):
        await self._save_model()
        for drop in (2, 3):  # last completed bucket(s) missing
            with self.subTest(dropped_buckets=drop):
                _RecorderRetrieveHass.drop_last = drop
                with self.assertLogs(logger, "ERROR") as logs:
                    self.assertFalse(await self._setup(self.CASES[0][1]))
                self.assertIn("missing, stale or irregular", "\n".join(logs.output))

    async def test_missing_interval_in_last_window_fails_closed(self):
        await self._save_model()
        _RecorderRetrieveHass.gap_back = 10
        with self.assertLogs(logger, "ERROR") as logs:
            self.assertFalse(await self._setup(self.CASES[0][1]))
        self.assertIn("missing, stale or irregular", "\n".join(logs.output))

    async def test_insufficient_history_fails_closed(self):
        await self._save_model()
        _RecorderRetrieveHass.days = 2
        with self.assertLogs(logger, "ERROR") as logs:
            self.assertFalse(await self._setup(self.CASES[0][1]))
        self.assertIn("Not enough load history", "\n".join(logs.output))
