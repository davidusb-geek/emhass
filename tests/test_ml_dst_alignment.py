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
from unittest.mock import AsyncMock, patch

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

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "config_path": root / "config.json",
    "defaults_path": root / "src/emhass/data/config_defaults.json",
    "associations_path": root / "src/emhass/data/associations.csv",
}

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
                # One more bucket is predicted; each grid step is the time-weighted mean of
                # the two lattice buckets it overlaps (the predictor returns 0, 1, 2, ...).
                self.assertEqual(model.received_steps, 7)
                weight = pd.Timedelta(offset) / pd.Timedelta(freq)
                np.testing.assert_allclose(result["yhat"], np.arange(6) + weight)

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
        # The last case is (warm-up steps, weight of the later bucket) of the blend.
        for tz, freq, now, start, warm_up, weight in (
            ("Asia/Kolkata", "60min", "2026-02-20 05:10", "2026-02-20 04:30", 0, 0.5),
            ("Asia/Kolkata", "60min", "2026-02-20 05:25", "2026-02-20 05:30", 1, 0.5),
            ("Asia/Kathmandu", "30min", "2026-02-20 10:35", "2026-02-20 10:15", 0, 0.5),
            ("Asia/Kathmandu", "60min", "2026-02-20 11:20", "2026-02-20 10:45", 0, 0.75),
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
                # The predictor returns 0, 1, 2, ...: the warm-up steps are dropped after blending.
                np.testing.assert_allclose(result["yhat"], np.arange(warm_up, warm_up + 6) + weight)
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

    def _process(self, rows, is_last_day=True, day="2026-02-20", now=None, hold=True):
        rh = object.__new__(RetrieveHass)
        rh.freq = pd.Timedelta("60min")
        rh.logger = logging.getLogger(__name__)
        data = [{"state": state, "last_changed": at} for state, at in rows]
        with time_machine.travel(now or self.NOW, tick=False):
            return rh._process_history_dataframe(
                data,
                "sensor.load",
                pd.Timestamp(day, tz="UTC"),
                False,
                is_last_day,
                hold_unchanged_tail=hold,
            )

    def _unavailable(self, rows):
        found = []
        rh = object.__new__(RetrieveHass)
        rh.freq = pd.Timedelta("60min")
        rh.logger = logging.getLogger(__name__)
        data = [{"state": state, "last_changed": at} for state, at in rows]
        with time_machine.travel(self.NOW, tick=False):
            rh._process_history_dataframe(
                data,
                "sensor.load",
                pd.Timestamp("2026-02-20", tz="UTC"),
                False,
                True,
                unavailable=found,
            )
        return [at.strftime("%H:%M") for at in found[0]]

    def test_unavailable_state_marks_the_buckets_it_covers(self):
        rows = [
            ("500", "2026-02-20T00:00:00+00:00"),
            ("unavailable", "2026-02-20T02:10:00+00:00"),
            ("510", "2026-02-20T05:30:00+00:00"),
        ]
        # Only buckets without a numeric reading of their own while unavailable count; the
        # empty bucket before it (01:00) just follows the earlier numeric state.
        self.assertEqual(self._unavailable(rows), ["02:00", "03:00", "04:00"])

    def test_unavailable_blip_inside_an_observed_bucket_is_not_flagged(self):
        rows = [
            ("500", "2026-02-20T00:00:00+00:00"),
            ("unavailable", "2026-02-20T02:10:00+00:00"),
            ("530", "2026-02-20T02:40:00+00:00"),
            ("540", "2026-02-20T05:30:00+00:00"),
        ]
        self.assertEqual(self._unavailable(rows), [])

    def test_bucket_ending_unavailable_after_a_reading_is_observed(self):
        rows = [
            ("500", "2026-02-20T00:00:00+00:00"),
            ("510", "2026-02-20T01:05:00+00:00"),
            ("unavailable", "2026-02-20T01:20:00+00:00"),
            ("520", "2026-02-20T03:30:00+00:00"),
        ]
        self.assertEqual(self._unavailable(rows), ["02:00"])

    def test_outage_at_the_end_of_a_past_day_covers_the_rest_of_that_day(self):
        rh = object.__new__(RetrieveHass)
        rh.freq = pd.Timedelta("60min")
        rh.logger = logging.getLogger(__name__)
        found = []
        rows = [
            {"state": "500", "last_changed": "2026-02-19T00:00:00+00:00"},
            {"state": "unavailable", "last_changed": "2026-02-19T22:30:00+00:00"},
        ]
        with time_machine.travel(self.NOW, tick=False):
            rh._process_history_dataframe(
                rows,
                "sensor.load",
                pd.Timestamp("2026-02-19", tz="UTC"),
                False,
                False,
                unavailable=found,
            )
        self.assertEqual([at.strftime("%H:%M") for at in found[0]], ["22:00", "23:00"])

    def _naive_day(self, is_last_day):
        """Days built without a time zone were accepted before the unavailable bookkeeping."""
        rows = [("500", "2026-02-20T00:00:00+00:00"), ("unavailable", "2026-02-20T06:10:00+00:00")]
        found = []
        data = [{"state": state, "last_changed": at} for state, at in rows]
        rh = object.__new__(RetrieveHass)
        rh.freq = pd.Timedelta("60min")
        rh.logger = logging.getLogger(__name__)
        with time_machine.travel(self.NOW, tick=False):
            frame = rh._process_history_dataframe(
                data,
                "sensor.load",
                pd.Timestamp("2026-02-20"),
                False,
                is_last_day,
                hold_unchanged_tail=True,
                unavailable=found,
            )
        self.assertEqual(frame.index[0], pd.Timestamp("2026-02-20 00:00", tz="UTC"))
        self.assertEqual(found[0][0], pd.Timestamp("2026-02-20 06:00", tz="UTC"))

    def test_a_naive_requested_day_is_read_as_utc_on_a_past_day(self):
        self._naive_day(False)

    def test_a_naive_requested_day_is_read_as_utc_on_the_last_day(self):
        self._naive_day(True)

    def test_other_consumers_keep_the_unextended_history(self):
        """Without the opt-in flag the retrieved history is exactly the recorded rows."""
        rows = [("500", "2026-02-20T00:00:00+00:00"), ("520", "2026-02-20T06:10:00+00:00")]
        frame = self._process(rows, hold=False)
        self.assertEqual(frame.index[-1], pd.Timestamp("2026-02-20 06:00", tz="UTC"))
        held = self._process(rows)
        pd.testing.assert_frame_equal(held.loc[: frame.index[-1]], frame, check_names=False)

    def test_stalled_recorder_is_indistinguishable_from_an_unchanged_sensor(self):
        """The history API carries no freshness: the tail is a documented assumption."""
        rows = [("500", "2026-02-20T00:00:00+00:00")]
        frame = self._process(rows)
        self.assertEqual(frame.index[-1], pd.Timestamp("2026-02-20 10:00", tz="UTC"))
        self.assertTrue((frame["sensor.load"] == 500.0).all())

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


async def _default_params():
    config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
    _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
    return await utils.build_params(emhass_conf, secrets, config, logger)


def _utc_hours(index):
    """Hours since the epoch, whatever the index resolution (pandas 3 defaults to microseconds)."""
    return np.asarray((index.tz_convert("UTC") - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(hours=1))


def _utc_sine(index):
    """A load profile that is a pure function of the absolute (UTC) instant."""
    hours = _utc_hours(index)
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
    traveller = None  # time_machine coordinates of the running test
    advance_to = None  # clock moved to once the history has been read

    def __init__(self, hass_url, long_lived_token, freq, time_zone, params, emhass_conf, logger):
        self.freq = freq
        self.time_zone = time_zone
        self.df_final = None
        self.unavailable_buckets = {}

    async def get_data(self, days_list, var_list, **kwargs):
        type(self).get_data_kwargs = kwargs
        self._var = var_list[0]
        end = pd.Timestamp.now(tz="UTC").floor(self.freq)
        index = pd.date_range(
            end=end, periods=pd.Timedelta(days=self.days) // self.freq, freq=self.freq
        ).tz_convert(self.time_zone)
        values = _utc_sine(index)
        values[-1] = -9999.0  # the still-running bucket holds a partial value
        frame = pd.DataFrame({self._var: values}, index=index)
        if self.drop_last:
            frame = frame.iloc[: -self.drop_last]
        if self.gap_back is not None:
            frame = frame.drop(frame.index[-self.gap_back])
        self.df_final = frame
        if self.advance_to is not None:
            self.traveller.move_to(self.advance_to)
        return True

    def prepare_data(self, var_load, **kwargs):
        self.df_final = self.df_final.rename(columns={self._var: var_load + "_positive"})
        return True


class TestHoldFlagThreading(unittest.IsolatedAsyncioTestCase):
    """Only the mlforecaster load retrieval opts in; get_data passes the flag to REST only."""

    async def _rest_kwargs(self, **kwargs):
        rh = object.__new__(RetrieveHass)
        rh.use_influxdb = rh.use_victoriametrics = rh.use_websocket = False
        rh.logger = logging.getLogger(__name__)
        with patch.object(
            RetrieveHass, "_get_data_rest_api", new=AsyncMock(return_value=True)
        ) as rest:
            await rh.get_data(pd.date_range("2026-02-20", periods=1, tz="UTC"), ["s"], **kwargs)
        return rest.call_args.kwargs

    async def test_default_does_not_hold(self):
        self.assertFalse((await self._rest_kwargs())["hold_unchanged_tail"])

    async def test_flag_reaches_the_rest_path(self):
        self.assertTrue((await self._rest_kwargs(hold_unchanged_tail=True))["hold_unchanged_tail"])

    async def test_websocket_fallback_keeps_the_flag(self):
        rh = object.__new__(RetrieveHass)
        rh.use_influxdb = rh.use_victoriametrics = False
        rh.use_websocket = True
        rh.logger = logging.getLogger(__name__)
        with (
            patch.object(RetrieveHass, "get_data_websocket", new=AsyncMock(return_value=False)),
            patch.object(
                RetrieveHass, "_get_data_rest_api", new=AsyncMock(return_value=True)
            ) as rest,
        ):
            await rh.get_data(
                pd.date_range("2026-02-20", periods=1, tz="UTC"), ["s"], hold_unchanged_tail=True
            )
        self.assertTrue(rest.call_args.kwargs["hold_unchanged_tail"])

    async def test_rest_path_extends_the_final_day_only_when_asked(self):
        rows = [
            {"state": "500", "last_changed": "2026-02-20T00:00:00+00:00"},
            {"state": "520", "last_changed": "2026-02-20T06:10:00+00:00"},
        ]
        for hold, last in ((True, "2026-02-20 10:00"), (False, "2026-02-20 06:00")):
            with self.subTest(hold=hold):
                rh = object.__new__(RetrieveHass)
                rh.freq = pd.Timedelta("60min")
                rh.long_lived_token = "token"
                rh.hass_url = "http://ha.invalid/"
                rh.ssl_verify = True
                rh.logger = logging.getLogger(__name__)
                with (
                    time_machine.travel(datetime(2026, 2, 20, 10, 20, tzinfo=UTC), tick=False),
                    patch.object(
                        RetrieveHass, "_fetch_history_data", new=AsyncMock(return_value=rows)
                    ),
                ):
                    ok = await rh._get_data_rest_api(
                        pd.date_range("2026-02-20", periods=1, tz="UTC"),
                        ["sensor.load"],
                        hold_unchanged_tail=hold,
                    )
                self.assertTrue(ok)
                self.assertEqual(rh.df_final.index[-1], pd.Timestamp(last, tz="UTC"))

    async def _hass_load_flag(self, method):
        recorded = {}

        class _Rh:
            unavailable_buckets = {}

            def __init__(self, *args):
                pass

            async def get_data(self, days_list, var_list, **kwargs):
                recorded["hold_unchanged_tail"] = kwargs["hold_unchanged_tail"]
                return False

        fcst = object.__new__(Forecast)
        fcst.logger = logging.getLogger(__name__)
        fcst.var_load = "sensor.load"
        fcst.time_zone = pytz.timezone(TZ_NAME)
        fcst.freq = pd.Timedelta("60min")
        fcst.params = {}
        fcst.emhass_conf = {}
        fcst.retrieve_hass_conf = {"hass_url": "u", "long_lived_token": "t"}
        fcst.get_data_from_file = False
        with patch.object(forecast_module, "RetrieveHass", _Rh):
            self.assertFalse(await fcst._prepare_hass_load_data(3, method))
        return recorded["hold_unchanged_tail"]

    async def test_only_the_ml_load_retrieval_holds(self):
        self.assertTrue(await self._hass_load_flag("mlforecaster"))
        self.assertFalse(await self._hass_load_flag("naive"))


def _ha_history(rows):
    """Mock of the Home Assistant /history/period response for one sensor.

    Like the real API it returns the recorded changes inside the requested day, led by
    a row stamped at the day's start that carries the state in force before it.
    """

    async def fetch(self, session, url, headers, var, day, is_first_day):
        start = pd.Timestamp(day)
        start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
        before = [state for at, state in rows if at < start]
        out = [(start, before[-1])] if before else []
        out += [(at, state) for at, state in rows if start <= at < start + pd.Timedelta(days=1)]
        return [{"state": state, "last_changed": at.isoformat()} for at, state in out] or None

    return fetch


def _step_weather(index):
    """Discontinuous covariate: a function of the absolute instant, changing every 3 UTC hours."""
    return 10.0 + 10.0 * ((_utc_hours(index) // 3) % 2)


def _step_load(index):
    return 300.0 + 4.0 * _step_weather(index)


def _ha_rows(now, freq_min, *, last_change=None, unavailable=None, profile=_utc_sine):
    """State changes of a load following ``_utc_sine``: one per step up to ``last_change``.

    ``unavailable`` is ``(start, end)``; one unavailable state is recorded at ``start`` and
    the numeric changes resume at ``end`` (``None``: still unavailable).
    """
    step = pd.Timedelta(minutes=freq_min)
    now = pd.Timestamp(now)
    index = pd.date_range(
        (now - pd.Timedelta(days=8)).floor("D"), last_change or now.floor(step), freq=step
    )
    rows = [(at, f"{value:.6f}") for at, value in zip(index, profile(index))]
    if unavailable:
        begin, end = unavailable
        rows = [(at, state) for at, state in rows if begin > at or (end is not None and at >= end)]
        rows.append((begin, "unavailable"))
        rows.sort(key=lambda row: row[0])
    return rows


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
        _RecorderRetrieveHass.advance_to = None

    async def _save_model(
        self, *, num_lags=72, tuned_lags=None, weather=False, freq_min=60, time_zone=TZ_NAME
    ):
        index = pd.date_range(
            "2026-01-01", periods=20 * 24 * 60 // freq_min, freq=f"{freq_min}min", tz=time_zone
        )
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

    async def _setup(
        self,
        now,
        method_ts_round="nearest",
        *,
        time_zone=TZ_NAME,
        freq_min=60,
        recorder=_RecorderRetrieveHass,
        rest_rows=None,
    ):
        """Run dayahead-optim input preparation; ``rest_rows`` replays a mocked HA REST history."""
        params = await _default_params()
        params["retrieve_hass_conf"]["time_zone"] = time_zone
        params["retrieve_hass_conf"]["optimization_time_step"] = freq_min
        params["retrieve_hass_conf"]["hass_url"] = "http://ha.invalid/"
        params["retrieve_hass_conf"]["long_lived_token"] = "token"
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
            time_machine.travel(now, tick=False) as traveller,
            patch.object(forecast_module, "RetrieveHass", recorder or RetrieveHass),
            patch.object(RetrieveHass, "get_ha_config", no_ha_config),
            patch.object(RetrieveHass, "_fetch_history_data", _ha_history(rest_rows or [])),
        ):
            _RecorderRetrieveHass.traveller = traveller
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
                self.assertTrue(_RecorderRetrieveHass.get_data_kwargs["hold_unchanged_tail"])
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

    # -- mocked Home Assistant REST history through RetrieveHass, preparation and ML ------

    async def _rest(self, now, freq_min=60, rows=None, **kwargs):
        rows = _ha_rows(now, freq_min) if rows is None else rows
        return await self._setup(now, recorder=None, rest_rows=rows, freq_min=freq_min, **kwargs)

    @staticmethod
    def _atol(freq_min):
        """Well below the change a one-step shift makes to the sine profile."""
        return 0.3 * 200.0 * 2 * np.pi * freq_min / 1440.0

    async def test_rest_history_aligns_at_every_resolution_across_dst(self):
        for freq_min in (5, 15, 30, 60):
            await self._save_model(freq_min=freq_min, num_lags=1440 // freq_min)
            for name, now, steps in self.CASES[1:]:
                with self.subTest(freq_min=freq_min, case=name):
                    idd = await self._rest(now, freq_min)
                    self.assertTrue(idd, "dayahead-optim input preparation failed")
                    frame = idd["df_input_data_dayahead"]
                    grid = idd["fcst"].forecast_dates
                    self.assertEqual(len(frame), steps * 60 // freq_min)
                    np.testing.assert_allclose(
                        frame["p_load_forecast"], _utc_sine(grid), atol=self._atol(freq_min)
                    )

    async def test_rest_history_rounding_modes_and_weather(self):
        await self._save_model(weather=True, freq_min=15, num_lags=96)

        async def weather_for(self_, index, features):
            return pd.DataFrame({"temp_air": np.full(len(index), 12.0)}, index=index)

        for mode in ("nearest", "first", "last"):
            for minute in (20, 40):
                with self.subTest(mode=mode, minute=minute):
                    now = datetime(2026, 10, 24, 10, minute, tzinfo=UTC)
                    with patch.object(Forecast, "get_weather_covariates", weather_for):
                        idd = await self._rest(now, 15, method_ts_round=mode)
                    self.assertTrue(idd)
                    grid = idd["fcst"].forecast_dates
                    np.testing.assert_allclose(
                        idd["df_input_data_dayahead"]["p_load_forecast"],
                        _utc_sine(grid),
                        atol=self._atol(15),
                    )

    async def test_unchanged_sensor_is_held_and_logged(self):
        await self._save_model()
        now = self.CASES[0][1]
        rows = _ha_rows(now, 60, last_change=pd.Timestamp("2026-02-20 03:00", tz="UTC"))
        with self.assertLogs(logger, "INFO") as logs:
            idd = await self._rest(now, rows=rows)
        self.assertTrue(idd)
        self.assertEqual(len(idd["df_input_data_dayahead"]), 72)
        self.assertIn("cannot show whether the sensor is still reporting", "\n".join(logs.output))

    async def test_stopped_recorder_is_accepted_like_an_unchanged_sensor(self):
        """Documented limitation: the history API carries no freshness information.

        A recorder that stopped yesterday returns the same rows as a sensor that has not
        changed since, so the held history is accepted.
        """
        await self._save_model()
        now = self.CASES[0][1]
        stopped = _ha_rows(now, 60, last_change=pd.Timestamp("2026-02-19 08:00", tz="UTC"))
        self.assertTrue(await self._rest(now, rows=stopped))

    async def test_unavailable_sensor_is_not_interpolated_into_the_model_input(self):
        await self._save_model()
        now = self.CASES[0][1]
        at = lambda hours: pd.Timestamp(now).floor("h") - pd.Timedelta(hours=hours)  # noqa: E731
        refused = {
            # The history then ends before the optimization start: stale.
            "unavailable_until_now": ((at(5), None), "must be consecutive"),
            "recovered_inside_the_window": ((at(10), at(6)), "unavailable or unknown"),
        }
        for name, (window, message) in refused.items():
            with self.subTest(case=name):
                with self.assertLogs(logger, "ERROR") as logs:
                    idd = await self._rest(now, rows=_ha_rows(now, 60, unavailable=window))
                self.assertFalse(idd)
                self.assertIn(message, "\n".join(logs.output))
        with self.subTest(case="older_than_the_window"):
            await self._save_model(num_lags=24)
            idd = await self._rest(now, rows=_ha_rows(now, 60, unavailable=(at(60), at(40))))
            self.assertTrue(idd)

    async def test_outage_across_utc_midnight_is_not_interpolated(self):
        await self._save_model(num_lags=48)
        now = self.CASES[0][1]
        begin = pd.Timestamp("2026-02-19 22:30", tz="UTC")
        end = pd.Timestamp("2026-02-20 00:00", tz="UTC")
        with self.assertLogs(logger, "ERROR") as logs:
            idd = await self._rest(now, rows=_ha_rows(now, 60, unavailable=(begin, end)))
        self.assertFalse(idd)
        self.assertIn("unavailable or unknown", "\n".join(logs.output))

    async def test_phase_adjustment_is_logged_and_cannot_hide_missing_history(self):
        zone = "Asia/Kathmandu"
        await self._save_model(time_zone=zone)
        now = datetime(2026, 2, 20, 11, 20, tzinfo=UTC)
        with self.assertLogs(logger, "INFO") as logs:
            idd = await self._rest(now, method_ts_round="first", time_zone=zone)
        self.assertTrue(idd)
        self.assertIn("time-weighted mean", "\n".join(logs.output))
        grid = idd["fcst"].forecast_dates
        np.testing.assert_allclose(
            idd["df_input_data_dayahead"]["p_load_forecast"], _utc_sine(grid), atol=self._atol(60)
        )
        # A recorder that lost its last completed steps is still refused in this zone.
        _RecorderRetrieveHass.drop_last = 3
        with self.assertLogs(logger, "ERROR") as logs:
            self.assertFalse(await self._setup(now, "first", time_zone=zone))
        self.assertIn("missing, stale or irregular", "\n".join(logs.output))

    async def test_step_boundary_crossed_after_the_history_is_read(self):
        """A bucket still running when the history was read is never an observation.

        The retrieval takes place just before a step boundary and ML preparation just
        after it; the forecast grid was captured before the boundary.
        """
        cases = (
            ("autumn", datetime(2026, 10, 25, 0, 59, 50, tzinfo=UTC)),
            ("spring", datetime(2026, 3, 29, 0, 59, 50, tzinfo=UTC)),
            ("ordinary", datetime(2026, 2, 20, 10, 59, 50, tzinfo=UTC)),
        )
        for freq_min in (5, 60):
            await self._save_model(freq_min=freq_min, num_lags=1440 // freq_min)
            for name, now in cases:
                with self.subTest(freq_min=freq_min, case=name):
                    boundary = pd.Timestamp(now).ceil(f"{freq_min}min")
                    _RecorderRetrieveHass.advance_to = (
                        boundary + pd.Timedelta(seconds=5)
                    ).to_pydatetime()
                    idd = await self._setup(now, freq_min=freq_min)
                    self.assertTrue(idd)
                    grid = idd["fcst"].forecast_dates
                    self.assertEqual(grid[0], boundary.tz_convert(TZ_NAME))
                    # The still-running bucket holds -9999 W: taking it for a completed one
                    # would put the error in the first forecast steps.
                    np.testing.assert_allclose(
                        idd["df_input_data_dayahead"]["p_load_forecast"],
                        _utc_sine(grid),
                        atol=self._atol(freq_min),
                    )

    async def _check_blend_against_truth(self, zone, now, rounding):
        """Discontinuous weather and a load that follows it, on a grid offset from the history.

        The truth for a grid step is the load averaged over that step. Moving the model's
        output by the phase without blending the neighbouring lattice step would put the
        weather change away from where it happened.
        """
        index = pd.date_range("2026-01-01", periods=20 * 24, freq="60min", tz=zone)
        data = pd.DataFrame(
            {"sensor.test_load": _step_load(index), "temp_air": _step_weather(index)}, index=index
        )
        mlf = MLForecaster(
            data,
            self.MODEL_TYPE,
            "sensor.test_load",
            "LinearRegression",
            2,
            self.conf,
            logger,
            weather_features=["temp_air"],
        )
        await mlf.fit()
        with open(self.conf["data_path"] / f"{self.MODEL_TYPE}_mlf.pkl", "wb") as f:
            pickle.dump(mlf, f)

        async def weather_for(self_, index, features):
            return pd.DataFrame({"temp_air": _step_weather(index)}, index=index)

        rows = _ha_rows(now, 60, profile=_step_load)
        with patch.object(Forecast, "get_weather_covariates", weather_for):
            idd = await self._setup(
                now, rounding, time_zone=zone, recorder=None, rest_rows=rows, freq_min=60
            )
        self.assertTrue(idd)
        grid = idd["fcst"].forecast_dates
        minutes = pd.date_range(grid[0], grid[-1] + pd.Timedelta("59min"), freq="1min")
        truth = pd.Series(_step_load(minutes), index=minutes).resample("60min").mean()
        np.testing.assert_allclose(
            idd["df_input_data_dayahead"]["p_load_forecast"].to_numpy(),
            truth.reindex(grid).to_numpy(),
            atol=1.0,
        )
        return grid

    async def test_phase_correction_keeps_covariates_on_the_physical_interval(self):
        grid = await self._check_blend_against_truth(
            "Asia/Kathmandu", datetime(2026, 2, 20, 11, 20, tzinfo=UTC), "first"
        )
        self.assertEqual(len(grid), 72)

    async def test_phase_with_a_warm_up_step_keeps_the_physical_interval(self):
        # "last" rounds the start up to 12:15 UTC, past now (11:20): one warm-up step is
        # predicted from the last completed bucket and dropped after the blend.
        now = datetime(2026, 2, 20, 11, 20, tzinfo=UTC)
        grid = await self._check_blend_against_truth("Asia/Kathmandu", now, "last")
        self.assertGreater(grid[0].tz_convert("UTC"), pd.Timestamp(now))

    async def test_phase_across_a_dst_change_keeps_the_physical_interval(self):
        # Adelaide (UTC+10:30 -> +09:30 on 2026-04-05) puts the hourly grid 30 min off the
        # UTC history lattice both before and after the change; the horizon spans it.
        grid = await self._check_blend_against_truth(
            "Australia/Adelaide", datetime(2026, 4, 4, 7, 20, tzinfo=UTC), "first"
        )
        self.assertEqual(len(grid), 73)
