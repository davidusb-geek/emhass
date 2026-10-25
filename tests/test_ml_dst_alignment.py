"""Regression tests for ML load forecast DST sizing and UTC alignment (#1248).

The test scope deliberately avoids any Home Assistant live writes. It exercises
the optimization-facing ML path with controlled model predictions and history.
"""

import logging
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytz

from emhass.forecast import Forecast
from emhass.machine_learning_forecaster import MLForecaster


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
        history_index = pd.date_range(
            end=start - cls.FREQ, periods=100, freq=cls.FREQ
        )
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
        result = await fcst._get_load_forecast_ml(history.drop(history.index[-5]), True, model, True)
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
