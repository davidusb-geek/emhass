#!/usr/bin/env python

import _pickle as cPickle
import bz2
import copy
import os
import pathlib
import pickle
import re
import unittest
import unittest.mock

import aiofiles
import aiohttp
import numpy as np
import orjson
import pandas as pd
from aioresponses import aioresponses

from emhass import forecast as forecast_module
from emhass import utils
from emhass.command_line import set_input_data_dict
from emhass.forecast import Forecast
from emhass.machine_learning_forecaster import MLForecaster
from emhass.optimization import Optimization
from emhass.retrieve_hass import RetrieveHass

# The root folder
root = pathlib.Path(utils.get_root(__file__, num_parent=2))
# Build emhass_conf paths
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

# create logger
logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)

# Sentinel marking "leave the weather_forecast_pv_quantile_bias param unset" in the
# Solcast-bias test helpers, distinct from any real (including 0.0/falsy) bias value.
_BIAS_UNSET = object()


class TestForecast(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def get_test_params():
        params = {}
        # Build params with default config and secrets
        if emhass_conf["defaults_path"].exists():
            config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
            _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
            params = await utils.build_params(emhass_conf, secrets, config, logger)
        else:
            raise Exception(
                "config_defaults.json does not exist in path: " + str(emhass_conf["defaults_path"])
            )
        return params

    async def asyncSetUp(self):
        self.get_data_from_file = True
        params = await TestForecast.get_test_params()
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        self.retrieve_hass_conf, self.optim_conf, self.plant_conf = (
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        )
        # Create RetrieveHass object
        self.rh = RetrieveHass(
            self.retrieve_hass_conf["hass_url"],
            self.retrieve_hass_conf["long_lived_token"],
            self.retrieve_hass_conf["optimization_time_step"],
            self.retrieve_hass_conf["time_zone"],
            params_json,
            emhass_conf,
            logger,
        )
        # Obtain sensor values from saved file
        if self.get_data_from_file:
            filename_path = emhass_conf["data_path"] / "test_df_final.pkl"
            async with aiofiles.open(filename_path, "rb") as inp:
                content = await inp.read()
                self.rh.df_final, self.days_list, self.var_list, self.rh.ha_config = pickle.loads(
                    content
                )
                self.rh.var_list = self.var_list
            self.retrieve_hass_conf["sensor_power_load_no_var_loads"] = str(self.var_list[0])
            self.retrieve_hass_conf["sensor_power_photovoltaics"] = str(self.var_list[1])
            self.retrieve_hass_conf["sensor_power_photovoltaics_forecast"] = str(self.var_list[2])
            self.retrieve_hass_conf["sensor_linear_interp"] = [
                retrieve_hass_conf["sensor_power_photovoltaics"],
                retrieve_hass_conf["sensor_power_photovoltaics_forecast"],
                retrieve_hass_conf["sensor_power_load_no_var_loads"],
            ]
            self.retrieve_hass_conf["sensor_replace_zero"] = [
                retrieve_hass_conf["sensor_power_photovoltaics"],
                retrieve_hass_conf["sensor_power_photovoltaics_forecast"],
            ]
        # Else obtain sensor values from HA
        else:
            self.days_list = utils.get_days_list(
                self.retrieve_hass_conf["historic_days_to_retrieve"]
            )
            self.var_list = [
                self.retrieve_hass_conf["sensor_power_load_no_var_loads"],
                self.retrieve_hass_conf["sensor_power_photovoltaics"],
                self.retrieve_hass_conf["sensor_power_photovoltaics_forecast"],
            ]
            await self.rh.get_data(
                self.days_list,
                self.var_list,
                minimal_response=False,
                significant_changes_only=False,
            )
        # Prepare data for optimization
        self.rh.prepare_data(
            self.retrieve_hass_conf["sensor_power_load_no_var_loads"],
            load_negative=self.retrieve_hass_conf["load_negative"],
            set_zero_min=self.retrieve_hass_conf["set_zero_min"],
            var_replace_zero=self.retrieve_hass_conf["sensor_replace_zero"],
            var_interp=self.retrieve_hass_conf["sensor_linear_interp"],
        )
        self.df_input_data = self.rh.df_final.copy()
        # Create forecast Object
        self.fcst = Forecast(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            params_json,
            emhass_conf,
            logger,
            get_data_from_file=self.get_data_from_file,
        )
        # The default for test is csv read
        self.df_weather_scrap = await self.fcst.get_weather_forecast(method="csv")
        self.p_pv_forecast = self.fcst.get_power_from_weather(self.df_weather_scrap)
        self.p_load_forecast = await self.fcst.get_load_forecast(
            method=optim_conf["load_forecast_method"]
        )
        self.p_pv_forecast = self.p_pv_forecast[~self.p_pv_forecast.index.duplicated(keep="first")]
        self.p_load_forecast = self.p_load_forecast[
            ~self.p_load_forecast.index.duplicated(keep="first")
        ]
        self.df_input_data_dayahead = pd.concat([self.p_pv_forecast, self.p_load_forecast], axis=1)
        self.df_input_data_dayahead.columns = ["p_pv_forecast", "p_load_forecast"]
        self.opt = Optimization(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            "profit",
            emhass_conf,
            logger,
        )
        # Manually create input data (from formatted parameter) dictionary
        self.input_data_dict = {
            "emhass_conf": emhass_conf,
            "retrieve_hass_conf": self.retrieve_hass_conf,
            "df_input_data": self.df_input_data,
            "df_input_data_dayahead": self.df_input_data_dayahead,
            "opt": self.opt,
            "rh": self.rh,
            "fcst": self.fcst,
            "p_pv_forecast": self.p_pv_forecast,
            "p_load_forecast": self.p_load_forecast,
            "params": params_json,
        }

    # Test weather forecast dataframe output based on saved csv file
    async def test_get_weather_forecast_csv(self):
        # Test dataframe from get weather forecast
        self.df_weather_csv = await self.fcst.get_weather_forecast(method="csv")
        self.assertEqual(self.fcst.weather_forecast_method, "csv")
        self.assertIsInstance(self.df_weather_csv, type(pd.DataFrame()))
        self.assertIsInstance(self.df_weather_csv.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(
            self.df_weather_csv.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(self.df_weather_csv.index.tz, self.fcst.time_zone)
        self.assertTrue(self.fcst.start_forecast < ts for ts in self.df_weather_csv.index)
        self.assertEqual(
            len(self.df_weather_csv),
            int(
                self.optim_conf["delta_forecast_daily"].total_seconds()
                / 3600
                / (self.fcst.freq.seconds / 3600)
            ),
        )
        # Test dataframe from get power from weather
        p_pv_forecast = self.fcst.get_power_from_weather(self.df_weather_csv)
        self.assertIsInstance(p_pv_forecast, pd.core.series.Series)
        self.assertIsInstance(p_pv_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(p_pv_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(p_pv_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.df_weather_csv), len(p_pv_forecast))
        df_weather_none = await self.fcst.get_weather_forecast(method="none")
        self.assertIs(df_weather_none, None)

    # Test PV forecast adjustment
    async def test_pv_forecast_adjust(self):
        model_type = "long_train_data"
        data_path = emhass_conf["data_path"] / str(model_type + ".pkl")
        async with aiofiles.open(data_path, "rb") as inp:
            content = await inp.read()
            data, _, _, _ = pickle.loads(content)
        # Clean nan's
        data = data.interpolate(method="linear", axis=0, limit=5)
        data = data.fillna(0.0)
        # Call data preparation method
        self.fcst.adjust_pv_forecast_data_prep(data)
        self.assertIsInstance(self.fcst.data_adjust_pv, pd.DataFrame)
        self.assertIsInstance(self.fcst.x_adjust_pv, pd.DataFrame)
        self.assertIsInstance(self.fcst.y_adjust_pv, pd.core.series.Series)
        # Call the fit method
        await self.fcst.adjust_pv_forecast_fit(
            n_splits=5, regression_model="LassoRegression", debug=False
        )
        # Call the predict method
        p_pv_forecast = self.fcst.adjust_pv_forecast_predict()
        self.assertEqual(len(p_pv_forecast), len(self.fcst.p_pv_forecast_validation))
        self.assertFalse(p_pv_forecast.isna().any().any(), "Adjusted forecast contains NaN values")
        self.assertGreaterEqual(self.fcst.validation_rmse, 0.0, "RMSE should be non-negative")
        self.assertLessEqual(self.fcst.validation_r2, 1.0, "R² score should be at most 1")
        self.assertGreaterEqual(self.fcst.validation_r2, -1.0, "R² score should be at least -1")

        # import plotly.express as px
        # data_to_plot = self.fcst.p_pv_forecast_validation[["forecast", "adjusted_forecast"]].reset_index()
        # fig = px.line(
        #     data_to_plot,
        #     x="index",  # Assuming the index is the timestamp
        #     y=["forecast", "adjusted_forecast"],
        #     labels={"index": "Time", "value": "Power (W)", "variable": "Forecast Type"},
        #     title="Forecast vs Adjusted Forecast",
        #     template='presentation'
        # )
        # fig.show()

    # Regression test for #521: daytime branch of apply_weighting returns the raw
    # regression output, which can be negative (e.g. LassoRegression extrapolating
    # on a cloudy day after sunny training history). Result must be clamped to >= 0.
    async def test_pv_forecast_adjust_clamps_negative(self):
        idx = pd.date_range("2026-04-19 10:00:00", periods=4, freq="15min", tz=self.fcst.time_zone)
        forecasted_pv = pd.DataFrame({"forecast": [500.0, 600.0, 700.0, 800.0]}, index=idx)

        class _NegativePredictModel:
            def predict(self, X):
                return np.full(len(X), -150.0)

        self.fcst.model_adjust_pv = _NegativePredictModel()
        result = self.fcst.adjust_pv_forecast_predict(forecasted_pv=forecasted_pv)
        self.assertTrue(
            (result["adjusted_forecast"] >= 0).all(),
            f"Adjusted forecast must be >= 0, got: {result['adjusted_forecast'].tolist()}",
        )

    # Test output weather forecast using openmeteo with mock get request data
    async def test_get_weather_forecast_openmeteo_method_mock(self):
        test_data_path = emhass_conf["data_path"] / "test_response_openmeteo_get_method.pbz2"

        async with aiofiles.open(test_data_path, "rb") as f:
            compressed = await f.read()

        data = bz2.decompress(compressed)
        data = cPickle.loads(data)
        data = orjson.loads(data.content)
        lat = self.retrieve_hass_conf["Latitude"]
        lon = self.retrieve_hass_conf["Longitude"]
        get_url = (
            "https://api.open-meteo.com/v1/forecast?"
            + "latitude="
            + str(round(lat, 2))
            + "&longitude="
            + str(round(lon, 2))
            + "&minutely_15="
            + "temperature_2m,"
            + "relative_humidity_2m,"
            + "rain,"
            + "cloud_cover,"
            + "wind_speed_10m,"
            + "shortwave_radiation_instant,"
            + "diffuse_radiation_instant,"
            + "direct_normal_irradiance_instant"
        )

        with aioresponses() as mocked:
            mocked.get(get_url, payload=data)

            # Test dataframe output from get weather forecast
            df_weather_openmeteo = await self.fcst.get_weather_forecast(method="open-meteo")
            self.assertIsInstance(df_weather_openmeteo, type(pd.DataFrame()))
            self.assertIsInstance(
                df_weather_openmeteo.index, pd.core.indexes.datetimes.DatetimeIndex
            )
            self.assertIsInstance(
                df_weather_openmeteo.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
            )
            self.assertEqual(df_weather_openmeteo.index.tz, self.fcst.time_zone)
            self.assertTrue(self.fcst.start_forecast < ts for ts in df_weather_openmeteo.index)
            self.assertEqual(
                len(df_weather_openmeteo),
                int(
                    self.optim_conf["delta_forecast_daily"].total_seconds()
                    / 3600
                    / (self.fcst.freq.seconds / 3600)
                ),
            )
            # Test the legacy code using PVLib module methods
            df_weather_openmeteo = await self.fcst.get_weather_forecast(
                method="open-meteo", use_legacy_pvlib=False
            )
            self.assertIsInstance(df_weather_openmeteo, type(pd.DataFrame()))
            self.assertIsInstance(
                df_weather_openmeteo.index, pd.core.indexes.datetimes.DatetimeIndex
            )
            self.assertIsInstance(
                df_weather_openmeteo.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
            )
            self.assertEqual(df_weather_openmeteo.index.tz, self.fcst.time_zone)
            self.assertIn("ghi", list(df_weather_openmeteo.columns))
            self.assertIn("dhi", list(df_weather_openmeteo.columns))
            self.assertIn("dni", list(df_weather_openmeteo.columns))
            # Test dataframe output from get power from weather forecast
            p_pv_forecast = self.fcst.get_power_from_weather(df_weather_openmeteo)
            self.assertIsInstance(p_pv_forecast, pd.core.series.Series)
            self.assertIsInstance(p_pv_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertIsInstance(p_pv_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
            self.assertEqual(p_pv_forecast.index.tz, self.fcst.time_zone)
            self.assertEqual(len(df_weather_openmeteo), len(p_pv_forecast))
            # Test dataframe output from get power from weather forecast (with 2 PV plant's)
            self.plant_conf["pv_module_model"] = [
                self.plant_conf["pv_module_model"][0],
                self.plant_conf["pv_module_model"][0],
            ]
            self.plant_conf["pv_inverter_model"] = [
                self.plant_conf["pv_inverter_model"][0],
                self.plant_conf["pv_inverter_model"][0],
            ]
            self.plant_conf["surface_tilt"] = [30, 45]
            self.plant_conf["surface_azimuth"] = [270, 90]
            self.plant_conf["modules_per_string"] = [8, 8]
            self.plant_conf["strings_per_inverter"] = [1, 1]
            p_pv_forecast = self.fcst.get_power_from_weather(df_weather_openmeteo)
            self.assertIsInstance(p_pv_forecast, pd.core.series.Series)
            self.assertIsInstance(p_pv_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertIsInstance(p_pv_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
            self.assertEqual(p_pv_forecast.index.tz, self.fcst.time_zone)
            self.assertEqual(len(df_weather_openmeteo), len(p_pv_forecast))

    async def test_get_weather_covariates(self):
        """get_weather_covariates returns the requested + derived columns aligned to the index."""
        from unittest.mock import patch

        # Build a synthetic Open-Meteo minutely_15 payload spanning the forecast index plus a few
        # past steps, with a non-constant temperature so the derived degree-days are meaningful.
        index = self.fcst.forecast_dates
        span_start = index[0] - 4 * self.fcst.freq
        full = pd.date_range(start=span_start, end=index[-1], freq=self.fcst.freq, tz=index.tz)
        times = (full.tz_convert("UTC").astype("int64") // 10**9).tolist()
        hours = full.hour + full.minute / 60.0
        temps = (18.0 + 8.0 * np.sin((hours - 9.0) / 24.0 * 2 * np.pi)).tolist()
        payload = {
            "minutely_15": {
                "time": times,
                "temperature_2m": temps,
                "relative_humidity_2m": [55.0] * len(full),
                "cloud_cover": [40.0] * len(full),
                "wind_speed_10m": [10.0] * len(full),
                "shortwave_radiation": [100.0] * len(full),
                "direct_radiation": [60.0] * len(full),
                "diffuse_radiation": [40.0] * len(full),
                "precipitation": [0.0] * len(full),
            }
        }
        weather_features = ["temp_air", "heating_degree", "cooling_degree"]
        with patch.object(self.fcst, "_fetch_open_meteo_covariates_json", return_value=payload):
            covariates = await self.fcst.get_weather_covariates(index, weather_features)
        self.assertIsInstance(covariates, pd.DataFrame)
        self.assertEqual(list(covariates.columns), weather_features)
        self.assertEqual(len(covariates), len(index))
        self.assertTrue(covariates.index.equals(index))
        # No NaNs after alignment + fill.
        self.assertFalse(covariates.isna().any().any())
        # Derived degree-days are consistent with the 18 C comfort set-point and temperature.
        comfort = self.fcst.WEATHER_COVARIATE_COMFORT_TEMP_C
        expected_heating = np.maximum(0.0, comfort - covariates["temp_air"])
        np.testing.assert_allclose(
            covariates["heating_degree"].to_numpy(), expected_heating.to_numpy(), atol=1e-6
        )

    async def test_get_weather_covariates_rejects_unsupported(self):
        """An unsupported covariate name raises a clear ValueError."""
        with self.assertRaises(ValueError):
            await self.fcst.get_weather_covariates(self.fcst.forecast_dates, ["not_a_real_column"])

    async def test_build_weather_future_returns_none_without_weather_features(self):
        """_build_weather_future returns None when the model has no weather_features."""
        from unittest.mock import MagicMock

        data_last_window = self.df_input_data.copy()
        # A minimal mock MLForecaster with no weather features
        mock_mlf = MagicMock()
        mock_mlf.weather_features = []
        mock_mlf.is_tuned = False
        mock_mlf.num_lags = 48

        result = await self.fcst._build_weather_future(data_last_window, mock_mlf)
        self.assertIsNone(result)

    async def test_build_weather_future_returns_none_when_no_last_window(self):
        """_build_weather_future returns None when data_last_window is None."""
        from unittest.mock import MagicMock

        mock_mlf = MagicMock()
        mock_mlf.weather_features = ["temp_air"]
        mock_mlf.is_tuned = False
        mock_mlf.num_lags = 48

        result = await self.fcst._build_weather_future(None, mock_mlf)
        self.assertIsNone(result)

    async def test_build_weather_future_builds_correct_horizon(self):
        """_build_weather_future calls get_weather_covariates over the correct future index."""
        from unittest.mock import MagicMock, patch

        index = self.fcst.forecast_dates
        span_start = index[0] - 4 * self.fcst.freq
        full = pd.date_range(start=span_start, end=index[-1], freq=self.fcst.freq, tz=index.tz)
        times = (full.tz_convert("UTC").astype("int64") // 10**9).tolist()
        payload = {
            "minutely_15": {
                "time": times,
                "temperature_2m": [20.0] * len(full),
                "relative_humidity_2m": [50.0] * len(full),
                "cloud_cover": [30.0] * len(full),
                "wind_speed_10m": [5.0] * len(full),
                "shortwave_radiation": [200.0] * len(full),
                "direct_radiation": [150.0] * len(full),
                "diffuse_radiation": [50.0] * len(full),
                "precipitation": [0.0] * len(full),
            }
        }
        num_lags = 16
        data_last_window = self.df_input_data.copy()
        mock_mlf = MagicMock()
        mock_mlf.weather_features = ["temp_air"]
        mock_mlf.is_tuned = False
        mock_mlf.num_lags = num_lags

        with patch.object(self.fcst, "_fetch_open_meteo_covariates_json", return_value=payload):
            weather_future = await self.fcst._build_weather_future(data_last_window, mock_mlf)

        self.assertIsNotNone(weather_future)
        self.assertIsInstance(weather_future, pd.DataFrame)
        self.assertEqual(len(weather_future), num_lags)
        self.assertIn("temp_air", weather_future.columns)
        # Verify the horizon is anchored exactly one step after the last window index.
        expected_start = data_last_window.index[-1] + data_last_window.index.freq
        self.assertEqual(weather_future.index[0], expected_start)
        # Verify the horizon frequency matches the input window frequency.
        self.assertEqual(weather_future.index.freq, data_last_window.index.freq)

    async def test_build_weather_future_uses_lags_opt_when_tuned(self):
        """_build_weather_future uses mlf.lags_opt (not num_lags) when is_tuned=True."""
        from unittest.mock import MagicMock, patch

        index = self.fcst.forecast_dates
        span_start = index[0] - 4 * self.fcst.freq
        full = pd.date_range(start=span_start, end=index[-1], freq=self.fcst.freq, tz=index.tz)
        times = (full.tz_convert("UTC").astype("int64") // 10**9).tolist()
        payload = {
            "minutely_15": {
                "time": times,
                "temperature_2m": [20.0] * len(full),
                "relative_humidity_2m": [50.0] * len(full),
                "cloud_cover": [30.0] * len(full),
                "wind_speed_10m": [5.0] * len(full),
                "shortwave_radiation": [200.0] * len(full),
                "direct_radiation": [150.0] * len(full),
                "diffuse_radiation": [50.0] * len(full),
                "precipitation": [0.0] * len(full),
            }
        }
        lags_opt_value = 24
        data_last_window = self.df_input_data.copy()
        mock_mlf = MagicMock()
        mock_mlf.weather_features = ["temp_air"]
        mock_mlf.is_tuned = True
        mock_mlf.lags_opt = lags_opt_value
        mock_mlf.num_lags = 48  # should be ignored when is_tuned=True

        with patch.object(self.fcst, "_fetch_open_meteo_covariates_json", return_value=payload):
            weather_future = await self.fcst._build_weather_future(data_last_window, mock_mlf)

        self.assertIsNotNone(weather_future)
        self.assertEqual(len(weather_future), lags_opt_value)

    async def test_build_weather_future_raises_on_non_uniform_index(self):
        """_build_weather_future raises ValueError when index freq cannot be inferred."""
        from unittest.mock import MagicMock

        # Build an irregular (non-uniform) index so that both .freq and pd.infer_freq return None.
        irregular_timestamps = pd.to_datetime(
            ["2023-01-01 00:00", "2023-01-01 00:15", "2023-01-01 01:00"]
        ).tz_localize(self.fcst.time_zone)
        data_last_window = pd.DataFrame(index=irregular_timestamps)
        mock_mlf = MagicMock()
        mock_mlf.weather_features = ["temp_air"]
        mock_mlf.is_tuned = False
        mock_mlf.num_lags = 4

        with self.assertRaises(ValueError, msg="Expected ValueError for non-uniform index"):
            await self.fcst._build_weather_future(data_last_window, mock_mlf)

    # Test output weather forecast using Solcast with mock get request data
    async def test_get_weather_forecast_solcast_method_mock(self):
        self.fcst.params = {
            "passed_data": {
                "weather_forecast_cache": False,
                "weather_forecast_cache_only": False,
            }
        }
        self.fcst.retrieve_hass_conf["solcast_api_key"] = "123456"
        self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = "123456"
        if os.path.isfile(emhass_conf["data_path"] / "weather_forecast_data.pkl"):
            os.rename(
                emhass_conf["data_path"] / "weather_forecast_data.pkl",
                emhass_conf["data_path"] / "temp_weather_forecast_data.pkl",
            )

        test_data_path = str(emhass_conf["data_path"] / "test_response_solcast_get_method.pbz2")

        async with aiofiles.open(test_data_path, "rb") as f:
            compressed = await f.read()

        data = bz2.decompress(compressed)
        data = cPickle.loads(data)
        data = orjson.loads(data.content)

        get_url = "https://api.solcast.com.au/rooftop_sites/123456/forecasts?hours=24"

        with aioresponses() as mocked:
            mocked.get(get_url, payload=data)

            df_weather_scrap = await self.fcst.get_weather_forecast(method="solcast")

            self.assertIsInstance(df_weather_scrap, type(pd.DataFrame()))
            self.assertIsInstance(df_weather_scrap.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertIsInstance(
                df_weather_scrap.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
            )
            self.assertEqual(df_weather_scrap.index.tz, self.fcst.time_zone)
            self.assertTrue(self.fcst.start_forecast < ts for ts in df_weather_scrap.index)
            self.assertEqual(
                len(df_weather_scrap),
                int(
                    self.optim_conf["delta_forecast_daily"].total_seconds()
                    / 3600
                    / (self.fcst.freq.seconds / 3600)
                ),
            )
            if os.path.isfile(emhass_conf["data_path"] / "temp_weather_forecast_data.pkl"):
                os.rename(
                    emhass_conf["data_path"] / "temp_weather_forecast_data.pkl",
                    emhass_conf["data_path"] / "weather_forecast_data.pkl",
                )

    # Test output weather forecast using Solcast-multiroofs with mock get request data
    async def test_get_weather_forecast_solcast_multiroofs_method_mock(self):
        self.fcst.params = {
            "passed_data": {
                "weather_forecast_cache": False,
                "weather_forecast_cache_only": False,
            }
        }
        self.fcst.retrieve_hass_conf["solcast_api_key"] = "123456"
        self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = "111111,222222,333333"
        roof_ids = re.split(r"[,\s]+", self.fcst.retrieve_hass_conf["solcast_rooftop_id"].strip())
        if os.path.isfile(emhass_conf["data_path"] / "weather_forecast_data.pkl"):
            os.rename(
                emhass_conf["data_path"] / "weather_forecast_data.pkl",
                emhass_conf["data_path"] / "temp_weather_forecast_data.pkl",
            )
        test_data_path = str(emhass_conf["data_path"] / "test_response_solcast_get_method.pbz2")
        async with aiofiles.open(test_data_path, "rb") as f:
            compressed = await f.read()

        data = bz2.decompress(compressed)
        data = cPickle.loads(data)
        data = orjson.loads(data.content)
        with aioresponses() as mocked:
            for roof_id in roof_ids:
                get_url = f"https://api.solcast.com.au/rooftop_sites/{roof_id}/forecasts?hours=24"
                mocked.get(get_url, payload=data)
            df_weather_scrap = await self.fcst.get_weather_forecast(method="solcast")
            self.assertIsInstance(df_weather_scrap, type(pd.DataFrame()))
            self.assertIsInstance(df_weather_scrap.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertIsInstance(
                df_weather_scrap.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
            )
            self.assertEqual(df_weather_scrap.index.tz, self.fcst.time_zone)
            self.assertTrue(self.fcst.start_forecast < ts for ts in df_weather_scrap.index)
            self.assertEqual(
                len(df_weather_scrap),
                int(
                    self.optim_conf["delta_forecast_daily"].total_seconds()
                    / 3600
                    / (self.fcst.freq.seconds / 3600)
                ),
            )
            if os.path.isfile(emhass_conf["data_path"] / "temp_weather_forecast_data.pkl"):
                os.rename(
                    emhass_conf["data_path"] / "temp_weather_forecast_data.pkl",
                    emhass_conf["data_path"] / "weather_forecast_data.pkl",
                )

    # Test Solcast resampling: 30-min Solcast data → 15-min optimization_time_step
    async def test_get_weather_forecast_solcast_15min_resampling_mock(self):
        """Verify Solcast data is correctly resampled when optimization_time_step < 30 min."""
        # Override freq to 15 minutes (default test uses 30 min)
        original_freq = self.fcst.freq
        original_forecast_dates = self.fcst.forecast_dates
        self.fcst.freq = pd.Timedelta("15min")
        self.fcst.retrieve_hass_conf["optimization_time_step"] = pd.Timedelta("15min")
        # Rebuild forecast_dates at 15-min intervals (same time window → 2× more slots)
        self.fcst.forecast_dates = pd.date_range(
            start=original_forecast_dates[0],
            end=original_forecast_dates[-1],
            freq=self.fcst.freq,
        )
        self.fcst.params = {
            "passed_data": {
                "weather_forecast_cache": False,
                "weather_forecast_cache_only": False,
            }
        }
        self.fcst.retrieve_hass_conf["solcast_api_key"] = "123456"
        self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = "123456"
        if os.path.isfile(emhass_conf["data_path"] / "weather_forecast_data.pkl"):
            os.rename(
                emhass_conf["data_path"] / "weather_forecast_data.pkl",
                emhass_conf["data_path"] / "temp_weather_forecast_data.pkl",
            )

        test_data_path = str(emhass_conf["data_path"] / "test_response_solcast_get_method.pbz2")
        async with aiofiles.open(test_data_path, "rb") as f:
            compressed = await f.read()
        data = bz2.decompress(compressed)
        data = cPickle.loads(data)
        data = orjson.loads(data.content)

        days_solcast = int(len(self.fcst.forecast_dates) * self.fcst.freq.seconds / 3600)
        get_url = f"https://api.solcast.com.au/rooftop_sites/123456/forecasts?hours={days_solcast}"

        with aioresponses() as mocked:
            mocked.get(get_url, payload=data)
            df_weather_scrap = await self.fcst.get_weather_forecast(method="solcast")

            self.assertIsInstance(df_weather_scrap, type(pd.DataFrame()))
            self.assertIsInstance(df_weather_scrap.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertEqual(df_weather_scrap.index.tz, self.fcst.time_zone)
            # Key assertion: output length must match the 15-min forecast_dates
            self.assertEqual(len(df_weather_scrap), len(self.fcst.forecast_dates))
            # Verify no NaN values after interpolation
            self.assertFalse(df_weather_scrap["yhat"].isna().any())

            # Verify interpolation correctness at a midpoint between two 30-min source timestamps
            # Pick a midpoint index to avoid edge effects
            midpoint_idx = len(df_weather_scrap.index) // 2
            ts_mid = df_weather_scrap.index[midpoint_idx]
            ts_prev = ts_mid - pd.Timedelta(minutes=15)
            ts_next = ts_mid + pd.Timedelta(minutes=15)

            # Ensure the neighboring timestamps exist in the index
            self.assertIn(ts_prev, df_weather_scrap.index)
            self.assertIn(ts_next, df_weather_scrap.index)

            y_prev = df_weather_scrap.loc[ts_prev, "yhat"]
            y_mid = df_weather_scrap.loc[ts_mid, "yhat"]
            y_next = df_weather_scrap.loc[ts_next, "yhat"]

            # Expected linear interpolation at the midpoint
            expected_mid = (y_prev + y_next) / 2.0

            # Check that the interpolated midpoint matches the expected linear value
            self.assertAlmostEqual(y_mid, expected_mid, places=6)

        # Restore original freq/forecast_dates
        self.fcst.freq = original_freq
        self.fcst.forecast_dates = original_forecast_dates
        if os.path.isfile(emhass_conf["data_path"] / "temp_weather_forecast_data.pkl"):
            os.rename(
                emhass_conf["data_path"] / "temp_weather_forecast_data.pkl",
                emhass_conf["data_path"] / "weather_forecast_data.pkl",
            )

    # Test #404: Solcast multi-day fixture proves day-2 PV is real, not zero-filled
    async def test_get_weather_forecast_solcast_multiday_mock(self):
        """Regression test for issue #404 (multi-day Solcast horizon).

        The fixture ``data/test_response_solcast_multiday.json`` is the real
        attachment from the issue report: 97 entries, 30-min cadence,
        2024-12-26T17:30Z → 2024-12-28T17:30Z (≈48 h of Solcast data).

        With ``delta_forecast_daily=2`` the ``forecast_dates`` window is 96
        slots (2 days × 48 half-hours).  The test pins the clock so the window
        aligns with the fixture, then asserts:
          (a) the returned DataFrame has exactly 96 rows (no truncation), and
          (b) day-2 PV values (rows 48–95) are non-zero — proving the code
              returns real Solcast data for the second day rather than zeros.
        """
        # --- 1. Save and rename any pre-existing weather cache ---
        if os.path.isfile(emhass_conf["data_path"] / "weather_forecast_data.pkl"):
            os.rename(
                emhass_conf["data_path"] / "weather_forecast_data.pkl",
                emhass_conf["data_path"] / "temp_weather_forecast_data.pkl",
            )

        # --- 2. Build a fresh Forecast with delta_forecast_daily=2 ---
        params = await TestForecast.get_test_params()
        params["passed_data"] = {
            "weather_forecast_cache": False,
            "weather_forecast_cache_only": False,
        }
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        optim_conf["delta_forecast_daily"] = pd.Timedelta(days=2)

        fcst = Forecast(
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            params_json,
            emhass_conf,
            logger,
            get_data_from_file=True,
        )

        # --- 3. PIN THE CLOCK ---
        # forecast.py builds forecast_dates from pd.Timestamp.now(tz=time_zone).
        # The fixture covers 2024-12-26T17:30Z → 2024-12-28T17:30Z.  We
        # directly overwrite forecast_dates (same technique as _pin_forecast_to_date
        # in the DST tests) so the window is fully inside the fixture range.
        #
        # Pinned start: 2024-12-26T17:30:00 Europe/Paris = 2024-12-26T16:30:00Z
        # Window end  : 2024-12-28T17:30:00 Europe/Paris = 2024-12-28T16:30:00Z
        # Day-2 solar daytime is 2024-12-28T08:30Z–16:00Z → indices ~64–95
        pinned_start = pd.Timestamp("2024-12-26 17:30:00", tz=fcst.time_zone)
        freq = fcst.freq  # 30 min
        pinned_end = pinned_start + pd.DateOffset(days=2)
        pinned_dates = (
            pd.date_range(
                start=pinned_start,
                end=pinned_end - freq,
                freq=freq,
                tz=fcst.time_zone,
            )
            .tz_convert("utc")
            .round(freq, ambiguous="infer", nonexistent="shift_forward")
            .tz_convert(fcst.time_zone)
        )
        fcst.start_forecast = pinned_start
        fcst.end_forecast = pinned_end
        fcst.forecast_dates = pinned_dates
        fcst.forecast_dates_tz = pinned_dates

        # --- 4. Configure Solcast credentials ---
        fcst.retrieve_hass_conf["solcast_api_key"] = "123456"
        fcst.retrieve_hass_conf["solcast_rooftop_id"] = "123456"

        # --- 5. Load the fixture ---
        fixture_path = str(emhass_conf["data_path"] / "test_response_solcast_multiday.json")
        with open(fixture_path, "rb") as f:
            solcast_data = orjson.loads(f.read())

        # --- 7. Mock with a REGEX so we're robust to future URL changes ---
        # Also patch _solcast_rate_limit_ok to bypass the daily API-call counter
        # (counter persists on disk across test runs; without the patch the test
        # would fail whenever it runs after the counter hits its 8-call cap).
        from unittest.mock import patch

        with (
            patch.object(fcst, "_solcast_rate_limit_ok", return_value=True),
            aioresponses() as mocked,
        ):
            mocked.get(
                re.compile(r"https://api\.solcast\.com\.au/.*"),
                payload=solcast_data,
            )

            df_result = await fcst.get_weather_forecast(method="solcast")

        # --- 9. ASSERT (a): no truncation — full 2-day window returned ---
        self.assertIsInstance(df_result, pd.DataFrame)
        self.assertEqual(
            len(df_result),
            96,
            msg=f"Expected 96 rows (2-day window); got {len(df_result)}",
        )

        # --- 10. ASSERT (b): day-2 values are real Solcast data, not zeros ---
        # Day-2 slot range: indices 48–95 of forecast_dates
        # The fixture has Dec 28 solar daytime (08:30Z–16:00Z):
        #   pv_estimate 0.0114 → 0.1800 → ... → 0.0014 kW (×1000 = W)
        # Those timestamps map into the second half of our 48-h window.
        day2_pv = df_result.iloc[48:]["yhat"]
        day2_sum = day2_pv.sum()
        self.assertGreater(
            day2_sum,
            0.0,
            msg=(
                f"Day-2 PV sum is {day2_sum:.1f} W — all zeros means the window "
                "did not overlap the fixture (clock-pin failure or truncation bug)."
            ),
        )
        # Also assert at least one specific Dec-28 daytime slot is positive.
        # forecast_dates[80] = 2024-12-28T08:30:00Z = fixture entry pv_estimate=0.0114 → 11.4 W
        ts_dec28_0830z = pd.Timestamp("2024-12-28 08:30:00", tz="UTC").tz_convert(fcst.time_zone)
        self.assertIn(ts_dec28_0830z, df_result.index, msg="Dec-28 08:30Z slot missing from index")
        pv_dec28_0830 = df_result.loc[ts_dec28_0830z, "yhat"]
        self.assertGreater(
            pv_dec28_0830,
            0.0,
            msg=f"Dec-28 08:30Z PV expected >0 W; got {pv_dec28_0830} W",
        )

        # --- 11. Restore weather cache if it existed ---
        if os.path.isfile(emhass_conf["data_path"] / "temp_weather_forecast_data.pkl"):
            os.rename(
                emhass_conf["data_path"] / "temp_weather_forecast_data.pkl",
                emhass_conf["data_path"] / "weather_forecast_data.pkl",
            )

    # Test #932: a weather cache lacking 'yhat' (e.g. left over after switching
    # weather_forecast_method, since the cache file is shared across methods)
    # must not crash get_power_from_weather. The rate-limited fetchers should
    # self-heal by refetching, the same way open-meteo already does.
    async def test_get_weather_forecast_solcast_incompatible_cache_recovers(self):
        from unittest.mock import patch

        cache_path = emhass_conf["data_path"] / "weather_forecast_data.pkl"
        temp_path = emhass_conf["data_path"] / "temp_weather_forecast_data.pkl"
        if os.path.isfile(cache_path):
            os.rename(cache_path, temp_path)

        # Schema-incompatible cache: open-meteo columns, NO 'yhat', over a stale
        # window that does not cover forecast_dates (forces the stale-cache path).
        stale_index = pd.date_range(
            start=self.fcst.forecast_dates[0] - pd.Timedelta(days=2),
            periods=len(self.fcst.forecast_dates) + 4,
            freq=self.fcst.freq,
        )
        incompatible = pd.DataFrame(
            {"ghi": 500.0, "dni": 400.0, "dhi": 100.0, "temp_air": 20.0},
            index=stale_index,
        )
        with open(cache_path, "wb") as f:
            pickle.dump(incompatible, f)

        self.fcst.params = {
            "passed_data": {
                "weather_forecast_cache": False,
                "weather_forecast_cache_only": False,
            }
        }
        self.fcst.retrieve_hass_conf["solcast_api_key"] = "123456"
        self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = "123456"
        # solar_forecast_kwp == 0 is the case the schema check must NOT depend on:
        # solcast does not use that key, yet a real solcast user can leave it at 0.
        # The old `solar_forecast_kwp != 0` guard would skip the check and serve the
        # yhat-less cache, crashing get_power_from_weather. Pin it to 0 here.
        self.fcst.retrieve_hass_conf["solar_forecast_kwp"] = 0

        # Solcast fixture for the (mocked) fresh fetch the fix should trigger.
        test_data_path = str(emhass_conf["data_path"] / "test_response_solcast_get_method.pbz2")
        async with aiofiles.open(test_data_path, "rb") as f:
            compressed = await f.read()
        payload = orjson.loads(cPickle.loads(bz2.decompress(compressed)).content)

        try:
            with (
                patch.object(self.fcst, "_solcast_rate_limit_ok", return_value=True),
                aioresponses() as mocked,
            ):
                mocked.get(
                    re.compile(r"https://api\.solcast\.com\.au/.*"),
                    payload=payload,
                )
                df_weather = await self.fcst.get_weather_forecast(method="solcast")

            # The incompatible cache must NOT be served verbatim: the refetched
            # frame has 'yhat' and get_power_from_weather must not raise.
            self.assertIsInstance(df_weather, pd.DataFrame)
            self.assertIn("yhat", df_weather.columns)
            p_pv = self.fcst.get_power_from_weather(df_weather)
            self.assertEqual(len(p_pv), len(self.fcst.forecast_dates))
            self.assertFalse(p_pv.isna().any())
        finally:
            if os.path.isfile(cache_path):
                os.remove(cache_path)
            if os.path.isfile(temp_path):
                os.rename(temp_path, cache_path)

    # Test output weather forecast using Forecast.Solar with mock get request data
    async def test_get_weather_forecast_solarforecast_method_mock(self):
        test_data_path = str(
            emhass_conf["data_path"] / "test_response_solarforecast_get_method.pbz2"
        )
        async with aiofiles.open(test_data_path, "rb") as f:
            compressed = await f.read()

        data = bz2.decompress(compressed)
        data = cPickle.loads(data)

        with aioresponses() as mocked:
            for i in range(len(self.plant_conf["pv_module_model"])):
                get_url = (
                    "https://api.forecast.solar/estimate/"
                    + str(round(self.fcst.lat, 2))
                    + "/"
                    + str(round(self.fcst.lon, 2))
                    + "/"
                    + str(self.plant_conf["surface_tilt"][i])
                    + "/"
                    + str(self.plant_conf["surface_azimuth"][i] - 180)
                    + "/"
                    + str(5)
                )
                mocked.get(get_url, payload=data)
                df_weather_solarforecast = await self.fcst.get_weather_forecast(
                    method="solar.forecast"
                )
                self.assertIsInstance(df_weather_solarforecast, type(pd.DataFrame()))
                self.assertIsInstance(
                    df_weather_solarforecast.index,
                    pd.core.indexes.datetimes.DatetimeIndex,
                )
                self.assertIsInstance(
                    df_weather_solarforecast.index.dtype,
                    pd.core.dtypes.dtypes.DatetimeTZDtype,
                )
                self.assertEqual(df_weather_solarforecast.index.tz, self.fcst.time_zone)
                self.assertTrue(
                    self.fcst.start_forecast < ts for ts in df_weather_solarforecast.index
                )
                self.assertEqual(
                    len(df_weather_solarforecast),
                    int(
                        self.optim_conf["delta_forecast_daily"].total_seconds()
                        / 3600
                        / (self.fcst.freq.seconds / 3600)
                    ),
                )

    #  Test output weather forecast using passed runtime lists
    async def test_get_forecasts_with_lists(self):
        # Load default params
        params = {}
        if emhass_conf["defaults_path"].exists():
            async with aiofiles.open(emhass_conf["defaults_path"]) as data:
                content = await data.read()
                defaults = orjson.loads(content)
                updated_emhass_conf, built_secrets = await utils.build_secrets(emhass_conf, logger)
                emhass_conf.update(updated_emhass_conf)
                params.update(
                    await utils.build_params(emhass_conf, built_secrets, defaults, logger)
                )
        else:
            raise Exception(
                "config_defaults.json does not exist in path: " + str(emhass_conf["defaults_path"])
            )
        # Create 48 (1 day of data) long lists runtime forecasts parameters
        runtimeparams = {
            "pv_power_forecast": [i + 1 for i in range(48)],
            "load_power_forecast": [i + 1 for i in range(48)],
            "load_cost_forecast": [i + 1 for i in range(48)],
            "prod_price_forecast": [i + 1 for i in range(48)],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        set_type = "dayahead-optim"
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            runtimeparams_json,
            params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        # Build RetrieveHass Object
        rh = RetrieveHass(
            retrieve_hass_conf["hass_url"],
            retrieve_hass_conf["long_lived_token"],
            retrieve_hass_conf["optimization_time_step"],
            retrieve_hass_conf["time_zone"],
            params,
            emhass_conf,
            logger,
        )
        # Obtain sensor values from saved file
        if self.get_data_from_file:
            data_path = emhass_conf["data_path"] / "test_df_final.pkl"
            async with aiofiles.open(data_path, "rb") as inp:
                content = await inp.read()
                rh.df_final, days_list, var_list, rh.ha_config = pickle.loads(content)
                rh.var_list = var_list
            retrieve_hass_conf["sensor_power_load_no_var_loads"] = str(self.var_list[0])
            retrieve_hass_conf["sensor_power_photovoltaics"] = str(self.var_list[1])
            retrieve_hass_conf["sensor_linear_interp"] = [
                retrieve_hass_conf["sensor_power_photovoltaics"],
                retrieve_hass_conf["sensor_power_photovoltaics_forecast"],
                retrieve_hass_conf["sensor_power_load_no_var_loads"],
            ]
            retrieve_hass_conf["sensor_replace_zero"] = [
                retrieve_hass_conf["sensor_power_photovoltaics"],
                retrieve_hass_conf["sensor_power_photovoltaics_forecast"],
            ]
        # Else obtain sensor values from HA
        else:
            days_list = utils.get_days_list(retrieve_hass_conf["historic_days_to_retrieve"])
            var_list = [
                retrieve_hass_conf["sensor_power_load_no_var_loads"],
                retrieve_hass_conf["sensor_power_photovoltaics"],
                retrieve_hass_conf["sensor_power_photovoltaics_forecast"],
            ]
            await rh.get_data(
                days_list,
                var_list,
                minimal_response=False,
                significant_changes_only=False,
            )
        # Prepare data for optimization
        rh.prepare_data(
            retrieve_hass_conf["sensor_power_load_no_var_loads"],
            load_negative=retrieve_hass_conf["load_negative"],
            set_zero_min=retrieve_hass_conf["set_zero_min"],
            var_replace_zero=retrieve_hass_conf["sensor_replace_zero"],
            var_interp=retrieve_hass_conf["sensor_linear_interp"],
        )
        df_input_data = rh.df_final.copy()
        # Build Forecast Object
        fcst = Forecast(
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            params_json,
            emhass_conf,
            logger,
            get_data_from_file=True,
        )
        # Obtain only 48 rows of data and remove last column for input
        df_input_data = copy.deepcopy(df_input_data).iloc[-49:-1]
        # Get Weather forecast with list, check dataframe output
        p_pv_forecast = await fcst.get_weather_forecast(method="list")
        df_input_data.index = p_pv_forecast.index
        df_input_data.index.freq = rh.df_final.index.freq
        self.assertIsInstance(p_pv_forecast, type(pd.DataFrame()))
        self.assertIsInstance(p_pv_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(p_pv_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(p_pv_forecast.index.tz, fcst.time_zone)
        self.assertTrue(fcst.start_forecast < ts for ts in p_pv_forecast.index)
        self.assertEqual(p_pv_forecast.values[0][0], 1)
        self.assertEqual(p_pv_forecast.values[-1][0], 48)
        # Get load forecast with list, check dataframe output
        p_load_forecast = await fcst.get_load_forecast(method="list")
        self.assertIsInstance(p_load_forecast, pd.core.series.Series)
        self.assertIsInstance(p_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(p_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(p_load_forecast.index.tz, fcst.time_zone)
        self.assertEqual(len(p_pv_forecast), len(p_load_forecast))
        self.assertEqual(p_load_forecast.values[0], 1)
        self.assertEqual(p_load_forecast.values[-1], 48)
        # Get load cost forecast with list, check dataframe output
        df_input_data = fcst.get_load_cost_forecast(df_input_data, method="list")
        self.assertIn(fcst.var_load_cost, df_input_data.columns)
        self.assertEqual(df_input_data.isnull().sum().sum(), 0)
        self.assertEqual(df_input_data["unit_load_cost"].values[0], 1)
        self.assertEqual(df_input_data["unit_load_cost"].values[-1], 48)
        # Get production price forecast with list, check dataframe output
        df_input_data = fcst.get_prod_price_forecast(df_input_data, method="list")
        self.assertIn(fcst.var_prod_price, df_input_data.columns)
        self.assertEqual(df_input_data.isnull().sum().sum(), 0)
        self.assertEqual(df_input_data["unit_prod_price"].values[0], 1)
        self.assertEqual(df_input_data["unit_prod_price"].values[-1], 48)

    # Test output weather forecast using longer passed runtime lists
    async def _build_longer_list_forecast(self, list_length: int):
        """Build a Forecast configured for 3-day list-based forecasts.

        Returns ``(fcst, params_json, runtimeparams_json)``.  The caller must
        override ``fcst.start_forecast``, ``fcst.end_forecast``,
        ``fcst.forecast_dates``, and ``fcst.forecast_dates_tz`` before calling
        any ``get_*_forecast`` method so that the window is fixed to a known
        date rather than relying on wall-clock time.
        """
        params = {}
        set_type = "dayahead-optim"
        if emhass_conf["defaults_path"].exists():
            async with aiofiles.open(emhass_conf["defaults_path"]) as data:
                content = await data.read()
                defaults = orjson.loads(content)
                updated_emhass_conf, built_secrets = await utils.build_secrets(emhass_conf, logger)
                emhass_conf.update(updated_emhass_conf)
                params.update(
                    await utils.build_params(emhass_conf, built_secrets, defaults, logger)
                )
        else:
            raise Exception(
                "config_defaults.json does not exist in path: " + str(emhass_conf["defaults_path"])
            )
        runtimeparams = {
            "pv_power_forecast": [i + 1 for i in range(list_length)],
            "load_power_forecast": [i + 1 for i in range(list_length)],
            "load_cost_forecast": [i + 1 for i in range(list_length)],
            "prod_price_forecast": [i + 1 for i in range(list_length)],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        optim_conf["delta_forecast_daily"] = pd.Timedelta(days=3)
        (
            _,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            runtimeparams_json,
            params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        fcst = Forecast(
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            params_json,
            emhass_conf,
            logger,
            get_data_from_file=True,
        )
        return fcst, params_json, runtimeparams_json

    def _pin_forecast_to_date(self, fcst, start_naive_str: str):
        """Override forecast window to a fixed start date (naive ISO string).

        Rebuilds ``forecast_dates`` and ``forecast_dates_tz`` using the same
        ``DateOffset`` logic as ``Forecast.__init__`` so that DST transitions
        within the window are handled correctly.
        """
        delta_days = fcst.optim_conf["delta_forecast_daily"].days
        start_ts = (
            pd.Timestamp(start_naive_str)
            .tz_localize(fcst.time_zone, nonexistent="shift_forward")
            .floor(fcst.freq)
        )
        end_ts = (start_ts + pd.DateOffset(days=delta_days)).replace(microsecond=0)
        dates = (
            pd.date_range(
                start=start_ts,
                end=end_ts - fcst.freq,
                freq=fcst.freq,
                tz=fcst.time_zone,
            )
            .tz_convert("utc")
            .round(fcst.freq, ambiguous="infer", nonexistent="shift_forward")
            .tz_convert(fcst.time_zone)
        )
        fcst.start_forecast = start_ts
        fcst.end_forecast = end_ts
        fcst.forecast_dates = dates
        fcst.forecast_dates_tz = dates

    async def _assert_longer_lists_forecast(self, fcst, expected_last: int):
        """Run the full set of list-forecast assertions for ``expected_last`` slots.

        PV and load forecasts use ``self.forecast_dates_tz`` which is pinned by
        ``_pin_forecast_to_date``, so exact slot counts are asserted.

        ``get_load_cost_forecast`` and ``get_prod_price_forecast`` with
        method="list" both internally call ``get_forecast_days_csv()`` which
        uses wall-clock time rather than ``self.start_forecast``.  Pinning
        those would require mocking ``pd.Timestamp.now`` throughout; instead
        we just verify they do not raise and produce a DataFrame with the
        expected columns.
        """
        p_pv_forecast = await fcst.get_weather_forecast(method="list")
        self.assertIsInstance(p_pv_forecast, pd.DataFrame)
        self.assertIsInstance(p_pv_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(p_pv_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(p_pv_forecast.index.tz, fcst.time_zone)
        self.assertTrue(fcst.start_forecast < ts for ts in p_pv_forecast.index)
        self.assertEqual(len(p_pv_forecast), expected_last)
        self.assertEqual(p_pv_forecast.values[0][0], 1)
        self.assertEqual(p_pv_forecast.values[-1][0], expected_last)

        p_load_forecast = await fcst.get_load_forecast(method="list")
        self.assertIsInstance(p_load_forecast, pd.core.series.Series)
        self.assertIsInstance(p_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(p_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(p_load_forecast.index.tz, fcst.time_zone)
        self.assertEqual(len(p_load_forecast), expected_last)
        self.assertEqual(p_load_forecast.values[0], 1)
        self.assertEqual(p_load_forecast.values[-1], expected_last)

        # get_load_cost_forecast and get_prod_price_forecast with method="list"
        # call get_forecast_days_csv() which always uses wall-clock time and
        # therefore cannot be pinned to a fixed date without mocking
        # pd.Timestamp.now globally.  Those code paths are not relevant to the
        # DST list-trimming fix validated here, so they are omitted.

    async def test_get_forecasts_with_longer_lists_summer(self):
        """3-day list forecast in summer (no DST transition): exactly 3×48 slots."""
        # 2025-07-10: mid-summer in Europe/Paris, no DST boundary in the 3-day window
        fcst, _, _ = await self._build_longer_list_forecast(list_length=3 * 48)
        self._pin_forecast_to_date(fcst, "2025-07-10 00:00:00")
        await self._assert_longer_lists_forecast(fcst, expected_last=3 * 48)

    async def test_get_forecasts_with_longer_lists_winter(self):
        """3-day list forecast in winter (no DST transition): exactly 3×48 slots."""
        # 2025-01-15: mid-winter in Europe/Paris, no DST boundary in the 3-day window
        fcst, _, _ = await self._build_longer_list_forecast(list_length=3 * 48)
        self._pin_forecast_to_date(fcst, "2025-01-15 00:00:00")
        await self._assert_longer_lists_forecast(fcst, expected_last=3 * 48)

    async def test_get_forecasts_with_longer_lists_spring_forward(self):
        """3-day list forecast crossing spring-forward DST: 3×48 − 2 slots (−1 h at 30 min).

        Europe/Paris 2025-03-30 02:00 CET → 03:00 CEST.
        Starting 2025-03-28 the 3-day window ends 2025-03-31, spanning the
        transition and producing 142 instead of 144 slots.
        """
        fcst, _, _ = await self._build_longer_list_forecast(list_length=3 * 48)
        self._pin_forecast_to_date(fcst, "2025-03-28 00:00:00")
        await self._assert_longer_lists_forecast(fcst, expected_last=3 * 48 - 2)

    async def test_get_forecasts_with_longer_lists_autumn_fallback(self):
        """3-day list forecast crossing autumn fall-back DST: 3×48 + 2 slots (+1 h at 30 min).

        Europe/Paris 2025-10-26 03:00 CEST → 02:00 CET.
        Starting 2025-10-24 the 3-day window ends 2025-10-27, spanning the
        transition and producing 146 instead of 144 slots.  The input list
        must be at least 146 entries long so it covers the full window.
        """
        fcst, _, _ = await self._build_longer_list_forecast(list_length=3 * 48 + 2)
        self._pin_forecast_to_date(fcst, "2025-10-24 00:00:00")
        await self._assert_longer_lists_forecast(fcst, expected_last=3 * 48 + 2)

    # Guard regression: _get_weather_list / _get_load_forecast_list must not crash on None input
    async def test_get_weather_list_none_does_not_crash(self):
        """Before the None-guard, passing pv_power_forecast=None raised:
        TypeError: object of type 'NoneType' has no len()
        The guard must return None/falsy without raising."""
        params = await TestForecast.get_test_params()
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        # Set prediction_horizon=72 so the guard cannot blame a short list
        params["passed_data"] = {
            "pv_power_forecast": None,
            "load_power_forecast": None,
            "load_cost_forecast": None,
            "prod_price_forecast": None,
            "prediction_horizon": 72,
        }
        params_json = orjson.dumps(params).decode("utf-8")
        fcst = Forecast(
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            params_json,
            emhass_conf,
            logger,
            get_data_from_file=True,
        )
        # Must not raise; must return falsy (None)
        try:
            result = await fcst.get_weather_forecast(method="list")
        except TypeError as exc:
            self.fail(f"get_weather_forecast(method='list') raised TypeError on None input: {exc}")
        self.assertFalse(
            result is not None and (hasattr(result, "__len__") and len(result) > 0),
            "Expected falsy/None result when pv_power_forecast=None",
        )
        # Same guard for load forecast
        try:
            load_result = await fcst.get_load_forecast(method="list")
        except TypeError as exc:
            self.fail(f"get_load_forecast(method='list') raised TypeError on None input: {exc}")
        self.assertFalse(
            load_result is not None and (hasattr(load_result, "__len__") and len(load_result) > 0),
            "Expected falsy/None result when load_power_forecast=None",
        )

    # Test output values of weather forecast using passed runtime lists and saved sensor datalf):
    async def test_get_forecasts_with_lists_special_case(self):
        # Load default params
        params = {}
        if emhass_conf["defaults_path"].exists():
            config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
            _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
            params = await utils.build_params(emhass_conf, secrets, config, logger)
        else:
            raise Exception(
                "config_defaults.json does not exist in path: " + str(emhass_conf["defaults_path"])
            )
        # Create 48 (1 day of data) long lists runtime forecasts parameters
        runtimeparams = {
            "load_cost_forecast": [i + 1 for i in range(48)],
            "prod_price_forecast": [i + 1 for i in range(48)],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        set_type = "dayahead-optim"
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            runtimeparams_json,
            params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        # Create RetrieveHass Object
        rh = RetrieveHass(
            retrieve_hass_conf["hass_url"],
            retrieve_hass_conf["long_lived_token"],
            retrieve_hass_conf["optimization_time_step"],
            retrieve_hass_conf["time_zone"],
            params,
            emhass_conf,
            logger,
        )
        # Obtain sensor values from saved file
        if self.get_data_from_file:
            data_path = emhass_conf["data_path"] / "test_df_final.pkl"
            async with aiofiles.open(data_path, "rb") as inp:
                content = await inp.read()
                rh.df_final, days_list, var_list, rh.ha_config = pickle.loads(content)
                rh.var_list = var_list
            retrieve_hass_conf["sensor_power_load_no_var_loads"] = str(self.var_list[0])
            retrieve_hass_conf["sensor_power_photovoltaics"] = str(self.var_list[1])
            retrieve_hass_conf["sensor_linear_interp"] = [
                retrieve_hass_conf["sensor_power_photovoltaics"],
                retrieve_hass_conf["sensor_power_photovoltaics_forecast"],
                retrieve_hass_conf["sensor_power_load_no_var_loads"],
            ]
            retrieve_hass_conf["sensor_replace_zero"] = [
                retrieve_hass_conf["sensor_power_photovoltaics"],
                retrieve_hass_conf["sensor_power_photovoltaics_forecast"],
            ]
        # Else obtain sensor values from HA
        else:
            days_list = utils.get_days_list(retrieve_hass_conf["historic_days_to_retrieve"])
            var_list = [
                retrieve_hass_conf["sensor_power_load_no_var_loads"],
                retrieve_hass_conf["sensor_power_photovoltaics"],
                retrieve_hass_conf["sensor_power_photovoltaics_forecast"],
            ]
            await rh.get_data(
                days_list,
                var_list,
                minimal_response=False,
                significant_changes_only=False,
            )
        # Prepare data for optimization
        rh.prepare_data(
            retrieve_hass_conf["sensor_power_load_no_var_loads"],
            load_negative=retrieve_hass_conf["load_negative"],
            set_zero_min=retrieve_hass_conf["set_zero_min"],
            var_replace_zero=retrieve_hass_conf["sensor_replace_zero"],
            var_interp=retrieve_hass_conf["sensor_linear_interp"],
        )
        df_input_data = rh.df_final.copy()
        # Create forecast object
        fcst = Forecast(
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            params_json,
            emhass_conf,
            logger,
            get_data_from_file=True,
        )
        # Obtain only 48 rows of data and remove last column for input
        df_input_data = copy.deepcopy(df_input_data).iloc[-49:-1]
        # Get weather forecast with list
        p_pv_forecast = await fcst.get_weather_forecast()
        df_input_data.index = p_pv_forecast.index
        df_input_data.index.freq = rh.df_final.index.freq
        # Get load cost forecast with list, check values from output
        df_input_data = fcst.get_load_cost_forecast(df_input_data, method="list")
        self.assertIn(fcst.var_load_cost, df_input_data.columns)
        self.assertEqual(df_input_data.isnull().sum().sum(), 0)
        self.assertEqual(df_input_data["unit_load_cost"].values[0], 1)
        self.assertEqual(df_input_data["unit_load_cost"].values[-1], 48)
        # Get production price forecast with list, check values from output
        df_input_data = fcst.get_prod_price_forecast(df_input_data, method="list")
        self.assertIn(fcst.var_prod_price, df_input_data.columns)
        self.assertEqual(df_input_data.isnull().sum().sum(), 0)
        self.assertEqual(df_input_data["unit_prod_price"].values[0], 1)
        self.assertEqual(df_input_data["unit_prod_price"].values[-1], 48)

    async def test_get_power_from_weather(self):
        self.assertIsInstance(self.p_pv_forecast, pd.core.series.Series)
        self.assertIsInstance(self.p_pv_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.p_pv_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(self.p_pv_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.df_weather_scrap), len(self.p_pv_forecast))
        # Test passing a lists of PV params
        self.plant_conf["pv_module_model"] = [
            self.plant_conf["pv_module_model"],
            self.plant_conf["pv_module_model"],
        ]
        self.plant_conf["pv_inverter_model"] = [
            self.plant_conf["pv_inverter_model"],
            self.plant_conf["pv_inverter_model"],
        ]
        self.plant_conf["surface_tilt"] = [30, 45]
        self.plant_conf["surface_azimuth"] = [270, 90]
        self.plant_conf["modules_per_string"] = [8, 8]
        self.plant_conf["strings_per_inverter"] = [1, 1]
        params = orjson.dumps({"passed_data": {"weather_forecast_cache": False}}).decode("utf-8")
        self.fcst = Forecast(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            params,
            emhass_conf,
            logger,
            get_data_from_file=self.get_data_from_file,
        )
        df_weather_scrap = await self.fcst.get_weather_forecast(method="csv")
        p_pv_forecast = self.fcst.get_power_from_weather(df_weather_scrap)
        self.assertIsInstance(p_pv_forecast, pd.core.series.Series)
        self.assertIsInstance(p_pv_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(p_pv_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(p_pv_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.df_weather_scrap), len(p_pv_forecast))
        # Test the mixed forecast
        params = orjson.dumps({"passed_data": {"alpha": 0.5, "beta": 0.5}}).decode("utf-8")
        df_input_data = self.input_data_dict["rh"].df_final.copy()
        self.fcst = Forecast(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            params,
            emhass_conf,
            logger,
            get_data_from_file=self.get_data_from_file,
        )
        df_weather_scrap = await self.fcst.get_weather_forecast(method="csv")
        p_pv_forecast = self.fcst.get_power_from_weather(
            df_weather_scrap, set_mix_forecast=True, df_now=df_input_data
        )
        self.assertIsInstance(p_pv_forecast, pd.core.series.Series)
        self.assertIsInstance(p_pv_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(p_pv_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(p_pv_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.df_weather_scrap), len(p_pv_forecast))

    # Test dataframe output of load forecast
    async def test_get_load_forecast(self):
        p_load_forecast = await self.fcst.get_load_forecast()
        self.assertIsInstance(p_load_forecast, pd.core.series.Series)
        self.assertIsInstance(p_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(p_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(p_load_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.p_pv_forecast), len(p_load_forecast))
        print(">> The length of the load forecast = " + str(len(p_load_forecast)))
        # Test the mixed forecast
        params_json = orjson.dumps({"passed_data": {"alpha": 0.5, "beta": 0.5}}).decode("utf-8")
        df_input_data = self.input_data_dict["rh"].df_final.copy()
        self.fcst = Forecast(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            params_json,
            emhass_conf,
            logger,
            get_data_from_file=self.get_data_from_file,
        )
        p_load_forecast = await self.fcst.get_load_forecast(
            set_mix_forecast=True, df_now=df_input_data
        )
        self.assertIsInstance(p_load_forecast, pd.core.series.Series)
        self.assertIsInstance(p_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(p_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(p_load_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.p_pv_forecast), len(p_load_forecast))
        # Test load forecast from csv
        p_load_forecast = await self.fcst.get_load_forecast(method="csv")
        self.assertIsInstance(p_load_forecast, pd.core.series.Series)
        self.assertIsInstance(p_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(p_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(p_load_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.p_pv_forecast), len(p_load_forecast))

    # Test dataframe output of ml load forecast
    async def test_get_load_forecast_mlforecaster(self):
        params = await TestForecast.get_test_params()
        params_json = orjson.dumps(params).decode("utf-8")
        costfun = "profit"
        action = "forecast-model-fit"
        params = copy.deepcopy(orjson.loads(params_json))
        # Pass custom runtime parameters
        runtimeparams = {
            "historic_days_to_retrieve": 20,
            "model_type": "long_train_data",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params["optim_conf"]["load_forecast_method"] = "mlforecaster"
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )

        data = copy.deepcopy(input_data_dict["df_input_data"])
        # Create MLForecaster Object
        model_type = input_data_dict["params"]["passed_data"]["model_type"]
        var_model = input_data_dict["params"]["passed_data"]["var_model"]
        sklearn_model = input_data_dict["params"]["passed_data"]["sklearn_model"]
        num_lags = input_data_dict["params"]["passed_data"]["num_lags"]

        mlf = MLForecaster(
            data,
            model_type,
            var_model,
            sklearn_model,
            num_lags,
            emhass_conf,
            logger,
        )
        await mlf.fit()
        # Get load forecast using mlforecaster
        p_load_forecast = await input_data_dict["fcst"].get_load_forecast(
            method="mlforecaster", use_last_window=False, debug=True, mlf=mlf
        )
        self.assertIsInstance(p_load_forecast, pd.core.series.Series)
        self.assertIsInstance(p_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(p_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(p_load_forecast.index.tz, self.fcst.time_zone)
        self.assertTrue((p_load_forecast.index == self.fcst.forecast_dates).all())
        self.assertEqual(len(self.p_pv_forecast), len(p_load_forecast))

    # Test load forecast with typical statistics method
    async def test_get_load_forecast_typical(self):
        p_load_forecast = await self.fcst.get_load_forecast(method="typical")
        self.assertIsInstance(p_load_forecast, pd.core.series.Series)
        self.assertIsInstance(p_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(p_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(p_load_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.p_pv_forecast), len(p_load_forecast))
        # Relaunch this test but changing the timestep to 1h
        params = self.fcst.params
        params["retrieve_hass_conf"]["optimization_time_step"] = 60
        self.retrieve_hass_conf["optimization_time_step"] = pd.Timedelta("1h")
        fcst = Forecast(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            params,
            emhass_conf,
            logger,
            get_data_from_file=self.get_data_from_file,
        )
        self.assertEqual(len(fcst.forecast_dates), 24)
        p_load_forecast = await fcst.get_load_forecast(method="typical")
        self.assertIsInstance(p_load_forecast, pd.core.series.Series)
        self.assertEqual(len(p_load_forecast), len(fcst.forecast_dates))

    # Test load cost forecast dataframe output using saved csv referece file
    def test_get_load_cost_forecast(self):
        df_input_data = self.fcst.get_load_cost_forecast(self.df_input_data)
        self.assertIn(self.fcst.var_load_cost, df_input_data.columns)
        self.assertEqual(df_input_data.isnull().sum().sum(), 0)
        df_input_data = self.fcst.get_load_cost_forecast(
            self.df_input_data, method="csv", csv_path="data_load_cost_forecast.csv"
        )
        self.assertIn(self.fcst.var_load_cost, df_input_data.columns)
        self.assertEqual(df_input_data.isnull().sum().sum(), 0)

    # Test production price forecast dataframe output using saved csv referece file
    def test_get_prod_price_forecast(self):
        df_input_data = self.fcst.get_prod_price_forecast(self.df_input_data)
        self.assertIn(self.fcst.var_prod_price, df_input_data.columns)
        self.assertEqual(df_input_data.isnull().sum().sum(), 0)
        df_input_data = self.fcst.get_prod_price_forecast(
            self.df_input_data, method="csv", csv_path="data_prod_price_forecast.csv"
        )
        self.assertIn(self.fcst.var_prod_price, df_input_data.columns)
        self.assertEqual(df_input_data.isnull().sum().sum(), 0)

    # Test DST forward and backward transition handling in forecast methods
    async def test_dst_forward_transition_handling(self):
        """Test that forecast methods handle DST forward transitions without raising NonExistentTimeError."""
        from datetime import datetime

        import pytz

        # Test case 1: Australia/Sydney DST forward transition (October 2025)
        # DST starts on October 5, 2025 at 2:00 AM -> 3:00 AM (2:00 AM doesn't exist)
        sydney_tz = pytz.timezone("Australia/Sydney")

        # Create a forecast that spans the DST transition
        dst_transition_params = copy.deepcopy(self.fcst.params)
        dst_retrieve_hass_conf = copy.deepcopy(self.retrieve_hass_conf)
        dst_retrieve_hass_conf["time_zone"] = sydney_tz

        # Set start time just before DST transition
        dst_start = sydney_tz.localize(datetime(2025, 10, 4, 23, 0, 0))  # Oct 4, 11 PM
        dst_end = dst_start + pd.Timedelta(hours=6)  # 6 hours later, crosses DST

        dst_fcst = Forecast(
            dst_retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            dst_transition_params,
            emhass_conf,
            logger,
            get_data_from_file=True,
        )
        # Override forecast dates to span DST transition
        dst_fcst.start_forecast = dst_start
        dst_fcst.end_forecast = dst_end
        dst_fcst.forecast_dates = (
            pd.date_range(
                start=dst_start,
                end=dst_end - dst_fcst.freq,
                freq=dst_fcst.freq,
                tz=sydney_tz,
            )
            .tz_convert("utc")
            .round(dst_fcst.freq, ambiguous="infer", nonexistent="shift_forward")
            .tz_convert(sydney_tz)
        )

        # Test naive load forecast during DST transition
        # This should not raise NonExistentTimeError
        try:
            p_load_forecast_dst = await dst_fcst.get_load_forecast(method="naive")
            self.assertIsInstance(p_load_forecast_dst, pd.core.series.Series)
            self.assertEqual(len(p_load_forecast_dst), len(dst_fcst.forecast_dates))
            # Check that index is properly timezone-aware
            self.assertEqual(p_load_forecast_dst.index.tz, sydney_tz)
            logger.info("DST forward transition test for naive method: PASSED")
        except Exception as e:
            self.fail(f"Naive forecast failed during DST forward transition: {e}")

        # Test typical load forecast during DST transition
        try:
            p_load_forecast_typical = await dst_fcst.get_load_forecast(method="typical")
            self.assertIsInstance(p_load_forecast_typical, pd.core.series.Series)
            self.assertEqual(len(p_load_forecast_typical), len(dst_fcst.forecast_dates))
            self.assertEqual(p_load_forecast_typical.index.tz, sydney_tz)
            logger.info("DST forward transition test for typical method: PASSED")
        except Exception as e:
            self.fail(f"Typical forecast failed during DST forward transition: {e}")

        # Test case 2: Test tz_localize with nonexistent times directly
        # Create naive timestamps that include the nonexistent 2:00 AM on DST forward day
        naive_times = pd.date_range(
            start="2025-10-05 01:30:00", end="2025-10-05 02:30:00", freq="30min"
        )  # This includes 2:00 AM which doesn't exist in Sydney on Oct 5, 2025

        # This should not raise NonExistentTimeError with our fix
        try:
            localized_times = naive_times.tz_localize(
                sydney_tz, ambiguous="infer", nonexistent="shift_forward"
            )
            # Verify that nonexistent times were shifted forward
            self.assertEqual(len(localized_times), len(naive_times))
            # The 2:00 AM should become 3:00 AM (shifted forward)
            for ts in localized_times:
                self.assertNotEqual(
                    ts.hour,
                    2,
                    "No timestamp should have hour=2 after DST forward shift",
                )

            # Add explicit assertion for shifted timestamps
            # Check that 2:00 AM is replaced by 3:00 AM (shifted forward)
            expected_hours = [
                1,
                3,
                3,
            ]  # 1:30 AM, 3:00 AM (shifted from 2:00), 3:30 AM (shifted from 2:30)
            actual_hours = [ts.hour for ts in localized_times]
            self.assertEqual(
                actual_hours,
                expected_hours,
                "Expected nonexistent times to be shifted forward correctly",
            )

            logger.info("Direct tz_localize DST forward transition test: PASSED")
        except Exception as e:
            self.fail(
                f"Direct tz_localize failed during DST forward transition: {e}"
            )  # Test case 3: US Eastern Time DST transition (March)
        # DST starts on March 9, 2025 at 2:00 AM -> 3:00 AM
        eastern_tz = pytz.timezone("US/Eastern")
        us_dst_start = eastern_tz.localize(datetime(2025, 3, 9, 1, 0, 0))  # March 9, 1 AM
        us_dst_end = us_dst_start + pd.Timedelta(hours=4)  # 4 hours later, crosses DST

        us_dst_retrieve_hass_conf = copy.deepcopy(self.retrieve_hass_conf)
        us_dst_retrieve_hass_conf["time_zone"] = eastern_tz

        us_dst_fcst = Forecast(
            us_dst_retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            dst_transition_params,
            emhass_conf,
            logger,
            get_data_from_file=True,
        )
        us_dst_fcst.start_forecast = us_dst_start
        us_dst_fcst.end_forecast = us_dst_end
        us_dst_fcst.forecast_dates = (
            pd.date_range(
                start=us_dst_start,
                end=us_dst_end - us_dst_fcst.freq,
                freq=us_dst_fcst.freq,
                tz=eastern_tz,
            )
            .tz_convert("utc")
            .round(us_dst_fcst.freq, ambiguous="infer", nonexistent="shift_forward")
            .tz_convert(eastern_tz)
        )

        try:
            us_p_load_forecast = await us_dst_fcst.get_load_forecast(method="naive")
            self.assertIsInstance(us_p_load_forecast, pd.core.series.Series)
            self.assertEqual(len(us_p_load_forecast), len(us_dst_fcst.forecast_dates))
            self.assertEqual(us_p_load_forecast.index.tz, eastern_tz)
            logger.info("US Eastern DST forward transition test: PASSED")
        except Exception as e:
            self.fail(f"US Eastern DST forecast failed during forward transition: {e}")

    async def test_dst_backward_transition_handling(self):
        """Test that forecast methods handle DST backward transitions (fall back) with ambiguous times."""
        from datetime import datetime

        import pytz

        # Test case 1: Australia/Sydney DST backward transition (April 2025)
        # DST ends on April 6, 2025 at 3:00 AM -> 2:00 AM (2:00-3:00 AM happens twice)
        sydney_tz = pytz.timezone("Australia/Sydney")

        # Create a forecast that spans the DST backward transition
        dst_transition_params = copy.deepcopy(self.fcst.params)
        dst_retrieve_hass_conf = copy.deepcopy(self.retrieve_hass_conf)
        dst_retrieve_hass_conf["time_zone"] = sydney_tz

        # Set start time just before DST backward transition
        dst_start = sydney_tz.localize(datetime(2025, 4, 6, 1, 0, 0))  # April 6, 1 AM
        dst_end = dst_start + pd.Timedelta(hours=5)  # 5 hours later, crosses DST backward

        dst_fcst = Forecast(
            dst_retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            dst_transition_params,
            emhass_conf,
            logger,
            get_data_from_file=True,
        )
        # Override forecast dates to span DST backward transition
        dst_fcst.start_forecast = dst_start
        dst_fcst.end_forecast = dst_end
        dst_fcst.forecast_dates = (
            pd.date_range(
                start=dst_start,
                end=dst_end - dst_fcst.freq,
                freq=dst_fcst.freq,
                tz=sydney_tz,
            )
            .tz_convert("utc")
            .round(dst_fcst.freq, ambiguous="infer", nonexistent="shift_forward")
            .tz_convert(sydney_tz)
        )

        # Test naive load forecast during DST backward transition
        try:
            p_load_forecast_dst = await dst_fcst.get_load_forecast(method="naive")
            self.assertIsInstance(p_load_forecast_dst, pd.core.series.Series)
            self.assertEqual(len(p_load_forecast_dst), len(dst_fcst.forecast_dates))
            # Check that index is properly timezone-aware
            self.assertEqual(p_load_forecast_dst.index.tz, sydney_tz)
            logger.info("DST backward transition test for naive method: PASSED")
        except Exception as e:
            self.fail(f"Naive forecast failed during DST backward transition: {e}")

        # Test case 2: Test tz_localize with ambiguous times directly
        # Create naive timestamps that include the ambiguous 2:00-3:00 AM on DST backward day
        naive_times = pd.date_range(
            start="2025-04-06 01:30:00", end="2025-04-06 03:30:00", freq="30min"
        )  # This includes ambiguous 2:00, 2:30, 3:00 AM times in Sydney on April 6, 2025

        # This should handle ambiguous times with our fix
        # For ambiguous times, we'll use "NaT" to handle them gracefully, or specify the first occurrence
        try:
            # For backward transitions, ambiguous="infer" sometimes fails, so use explicit handling
            localized_times = naive_times.tz_localize(
                sydney_tz, ambiguous="NaT", nonexistent="shift_forward"
            )
            # Verify that we got some valid results (non-NaT times)
            valid_times = localized_times.dropna()
            self.assertGreater(
                len(valid_times),
                0,
                "Should have some valid timestamps after handling ambiguous times",
            )
            # Check that we got timezone-aware results for valid times
            for ts in valid_times:
                self.assertIsNotNone(ts.tzinfo, "Valid timestamps should be timezone-aware")

            logger.info("Direct tz_localize DST backward transition test: PASSED")
        except Exception as e:
            # Try alternative approach with first occurrence of ambiguous times
            try:
                localized_times = naive_times.tz_localize(
                    sydney_tz,
                    ambiguous=[True, True, True, True, False],
                    nonexistent="shift_forward",
                )
                # Verify that ambiguous times were handled
                self.assertEqual(len(localized_times), len(naive_times))
                # Check that we got reasonable results for ambiguous times
                for ts in localized_times:
                    self.assertIsNotNone(ts.tzinfo, "All timestamps should be timezone-aware")

                logger.info("Direct tz_localize DST backward transition test (alternative): PASSED")
            except Exception as e2:
                self.fail(f"Direct tz_localize failed during DST backward transition: {e} and {e2}")

        # Test case 3: US Eastern Time DST backward transition (November)
        # DST ends on November 2, 2025 at 2:00 AM -> 1:00 AM
        eastern_tz = pytz.timezone("US/Eastern")
        us_dst_start = eastern_tz.localize(datetime(2025, 11, 2, 0, 30, 0))  # Nov 2, 12:30 AM
        us_dst_end = us_dst_start + pd.Timedelta(hours=4)  # 4 hours later, crosses DST backward

        us_dst_retrieve_hass_conf = copy.deepcopy(self.retrieve_hass_conf)
        us_dst_retrieve_hass_conf["time_zone"] = eastern_tz

        us_dst_fcst = Forecast(
            us_dst_retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            dst_transition_params,
            emhass_conf,
            logger,
            get_data_from_file=True,
        )
        us_dst_fcst.start_forecast = us_dst_start
        us_dst_fcst.end_forecast = us_dst_end
        us_dst_fcst.forecast_dates = (
            pd.date_range(
                start=us_dst_start,
                end=us_dst_end - us_dst_fcst.freq,
                freq=us_dst_fcst.freq,
                tz=eastern_tz,
            )
            .tz_convert("utc")
            .round(us_dst_fcst.freq, ambiguous="infer", nonexistent="shift_forward")
            .tz_convert(eastern_tz)
        )

        try:
            us_p_load_forecast = await us_dst_fcst.get_load_forecast(method="naive")
            self.assertIsInstance(us_p_load_forecast, pd.core.series.Series)
            self.assertEqual(len(us_p_load_forecast), len(us_dst_fcst.forecast_dates))
            self.assertEqual(us_p_load_forecast.index.tz, eastern_tz)
            logger.info("US Eastern DST backward transition test: PASSED")
        except Exception as e:
            self.fail(f"US Eastern DST forecast failed during backward transition: {e}")

    async def test_solcast_caching_and_errors(self):
        """Test Solcast caching logic and API error handling."""
        w_forecast_cache_path = emhass_conf["data_path"] / "weather_forecast_data.pkl"
        # Test Cache Hit
        data = pd.DataFrame(index=self.fcst.forecast_dates)
        data["yhat"] = 1000.0
        # Caching logic uses pickle, so we can save whatever we want
        await self.fcst.set_cached_forecast_data(w_forecast_cache_path, data)
        # Force method="solcast" to hit the cache check
        res = await self.fcst.get_weather_forecast(method="solcast")
        self.assertIsInstance(res, pd.DataFrame)
        # Ensure it loaded our dummy data
        self.assertTrue(np.all(np.isclose(res["yhat"], 1000.0)))
        # Test API Errors
        # Remove cache to force API call
        if os.path.exists(w_forecast_cache_path):
            os.remove(w_forecast_cache_path)
        self.fcst.retrieve_hass_conf["solcast_api_key"] = "TEST_KEY"
        self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = "TEST_ID"
        # Test 429 (Too Many Requests)
        with aioresponses() as mocked:
            # We mock ANY URL starting with solcast
            mocked.get(re.compile(r"https://api\.solcast\.com\.au/.*"), status=429)
            res = await self.fcst.get_weather_forecast(method="solcast")
            self.assertFalse(res)
        # Test 500 (Server Error)
        with aioresponses() as mocked:
            mocked.get(re.compile(r"https://api\.solcast\.com\.au/.*"), status=500)
            res = await self.fcst.get_weather_forecast(method="solcast")
            self.assertFalse(res)

    async def test_get_cached_forecast_data_stale_open_meteo_deletes_cache(self):
        """Stale Open-Meteo cache must be deleted and return False (no zero-fill).

        Regression test for v0.17.3: get_cached_forecast_data() used to
        reindex + zero-fill irradiance when the cache did not cover the full
        requested timeframe.  For Open-Meteo (no rate limits) the correct
        behaviour is to delete the stale pickle so the next call fetches fresh
        data from the API.
        """
        w_forecast_cache_path = emhass_conf["data_path"] / "weather_forecast_data_stale_test.pkl"
        # Build a cache that covers yesterday only (stale relative to forecast_dates)
        yesterday = self.fcst.forecast_dates[0] - pd.Timedelta(days=1)
        stale_index = pd.date_range(
            start=yesterday,
            periods=len(self.fcst.forecast_dates),
            freq=self.fcst.freq,
            tz=self.fcst.time_zone,
        )
        stale_data = pd.DataFrame({"ghi": 500.0, "dni": 400.0, "dhi": 100.0}, index=stale_index)
        await self.fcst.set_cached_forecast_data(w_forecast_cache_path, stale_data)
        self.assertTrue(w_forecast_cache_path.exists())

        # Override method so get_cached_forecast_data sees "open-meteo"
        original_method = self.fcst.weather_forecast_method
        self.fcst.weather_forecast_method = "open-meteo"
        try:
            result = await self.fcst.get_cached_forecast_data(w_forecast_cache_path)
        finally:
            self.fcst.weather_forecast_method = original_method

        # Must return None and delete the stale file
        self.assertIsNone(result)
        self.assertFalse(w_forecast_cache_path.exists(), "Stale Open-Meteo cache should be deleted")

    async def test_get_cached_forecast_data_corrupt_cache_deletes_and_returns_none(self):
        """A corrupt (non-DataFrame) cache pickle must be deleted and return None.

        Also exercises the os.remove-after-close path on Windows (the file handle
        must be released before unlink, else PermissionError [WinError 32]).
        """
        w_forecast_cache_path = emhass_conf["data_path"] / "weather_forecast_data_corrupt_test.pkl"
        # Write a non-DataFrame payload directly (bypasses set_cached_forecast_data)
        with open(w_forecast_cache_path, "wb") as f:
            pickle.dump({"not": "a dataframe"}, f)
        self.assertTrue(w_forecast_cache_path.exists())

        result = await self.fcst.get_cached_forecast_data(w_forecast_cache_path)

        self.assertIsNone(result)
        self.assertFalse(w_forecast_cache_path.exists(), "Corrupt cache should be deleted")

    async def test_get_cached_forecast_data_stale_solcast_zero_fills(self):
        """Stale Solcast cache must be served as reindexed/zero-filled data.

        For rate-limited providers (Solcast) the v0.17.3 stale-cache fallback
        (reindex + zero-fill) must still be used to preserve daily API quota.
        """
        w_forecast_cache_path = (
            emhass_conf["data_path"] / "weather_forecast_data_stale_solcast_test.pkl"
        )
        yesterday = self.fcst.forecast_dates[0] - pd.Timedelta(days=1)
        stale_index = pd.date_range(
            start=yesterday,
            periods=len(self.fcst.forecast_dates),
            freq=self.fcst.freq,
            tz=self.fcst.time_zone,
        )
        stale_data = pd.DataFrame({"yhat": 1000.0}, index=stale_index)
        await self.fcst.set_cached_forecast_data(w_forecast_cache_path, stale_data)

        original_method = self.fcst.weather_forecast_method
        self.fcst.weather_forecast_method = "solcast"
        try:
            result = await self.fcst.get_cached_forecast_data(w_forecast_cache_path)
        finally:
            self.fcst.weather_forecast_method = original_method
            if w_forecast_cache_path.exists():
                os.remove(w_forecast_cache_path)

        # Must return a DataFrame (stale data served, file NOT deleted)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.fcst.forecast_dates))
        self.assertTrue(
            (result.index == self.fcst.forecast_dates).all(),
            "Stale Solcast cache index must match forecast_dates",
        )
        self.assertIn("yhat", result.columns, "Stale Solcast cache must include 'yhat' column")
        # Stale data is served via reindex + time-interpolation (which extrapolates the last
        # known constant value forward).  The result must be finite and non-NaN; the exact
        # value equals the stale payload (1000.0) because interpolation extrapolates a
        # constant series.
        self.assertFalse(
            result["yhat"].isna().any(),
            "Stale Solcast cache should not contain NaNs in yhat",
        )
        self.assertTrue(
            np.isfinite(result["yhat"].values).all(),
            "Stale Solcast cache yhat values must be finite",
        )
        self.assertTrue(
            (result["yhat"] == 1000.0).all(),
            "Stale Solcast cache yhat should equal the extrapolated stale value (1000.0)",
        )

    async def test_open_meteo_legacy_pvlib(self):
        """Test the use_legacy_pvlib=True path in open-meteo."""
        # Load mock data
        test_data_path = emhass_conf["data_path"] / "test_response_openmeteo_get_method.pbz2"
        async with aiofiles.open(test_data_path, "rb") as f:
            compressed = await f.read()
        data = bz2.decompress(compressed)
        data = cPickle.loads(data)
        data = orjson.loads(data.content)
        with aioresponses() as mocked:
            mocked.get(re.compile(r"https://api\.open-meteo\.com/.*"), payload=data)
            # Call with legacy=True
            df = await self.fcst.get_weather_forecast(method="open-meteo", use_legacy_pvlib=True)
            self.assertIsInstance(df, pd.DataFrame)
            # Verify columns exist (calculated by cloud_cover_to_irradiance)
            self.assertIn("ghi", df.columns)
            self.assertIn("dni", df.columns)
            self.assertIn("dhi", df.columns)

    async def _load_openmeteo_mock_payload(self):
        """Load the recorded Open-Meteo response used as a mock payload."""
        test_data_path = emhass_conf["data_path"] / "test_response_openmeteo_get_method.pbz2"
        async with aiofiles.open(test_data_path, "rb") as f:
            compressed = await f.read()
        data = bz2.decompress(compressed)
        data = cPickle.loads(data)
        return orjson.loads(data.content)

    async def test_open_meteo_cold_start_retries_then_succeeds(self):
        """Cold start (no cache): a transient failure is retried, then succeeds.

        With no usable cache to fall back on, the fetch must retry rather than
        give up on the first transient error.  The recorded payload is served on
        the third attempt and must be returned (a non-None dict written to the
        cache file).
        """
        payload = await self._load_openmeteo_mock_payload()
        json_path = emhass_conf["data_path"] / "cached-open-meteo-forecast-b.json"
        # Ensure a true cold start: no cache file on disk.
        if os.path.exists(json_path):
            os.remove(json_path)
        url_pattern = re.compile(r"https://api\.open-meteo\.com/.*")
        # Patch the backoff to zero so the test does not actually sleep.
        original_backoff = forecast_module.open_meteo_backoff_seconds
        forecast_module.open_meteo_backoff_seconds = (0, 0, 0)
        try:
            with aioresponses() as mocked:
                # Two transient failures (a 504 then a connection error), then success.
                mocked.get(url_pattern, status=504)
                mocked.get(url_pattern, exception=aiohttp.ClientConnectionError("boom"))
                mocked.get(url_pattern, payload=payload)
                result = await self.fcst.get_cached_open_meteo_forecast_json()
            # The successful third attempt must return the payload...
            self.assertIsInstance(result, dict)
            self.assertIn("minutely_15", result)
            # ...and it must have been retried exactly three times.
            requests_made = [k for k in mocked.requests if k[0] == "GET"]
            total_calls = sum(len(v) for v in mocked.requests.values())
            self.assertEqual(total_calls, 3, "Cold start must retry up to 3 times")
            self.assertTrue(requests_made, "Open-Meteo GET should have been issued")
            # The freshly fetched data must be persisted to the cache file, and
            # the persisted content must round-trip to the same payload (proving
            # we wrote complete, valid JSON rather than a partial/corrupt file).
            self.assertTrue(os.path.exists(json_path))
            async with aiofiles.open(json_path) as json_file:
                persisted = orjson.loads(await json_file.read())
            self.assertEqual(persisted, result)
            self.assertIn("minutely_15", persisted)
        finally:
            forecast_module.open_meteo_backoff_seconds = original_backoff
            if os.path.exists(json_path):
                os.remove(json_path)

    async def test_open_meteo_cold_start_all_attempts_fail_returns_none(self):
        """Cold start with every attempt failing returns None and writes no cache."""
        json_path = emhass_conf["data_path"] / "cached-open-meteo-forecast-b.json"
        if os.path.exists(json_path):
            os.remove(json_path)
        url_pattern = re.compile(r"https://api\.open-meteo\.com/.*")
        original_backoff = forecast_module.open_meteo_backoff_seconds
        forecast_module.open_meteo_backoff_seconds = (0, 0, 0)
        try:
            with aioresponses() as mocked:
                for _ in range(forecast_module.open_meteo_max_attempts):
                    mocked.get(url_pattern, status=502)
                result = await self.fcst.get_cached_open_meteo_forecast_json()
                total_calls = sum(len(v) for v in mocked.requests.values())
            # No cache + all attempts failed -> None, and nothing written to disk.
            self.assertIsNone(result)
            self.assertEqual(total_calls, forecast_module.open_meteo_max_attempts)
            self.assertFalse(
                os.path.exists(json_path),
                "Failed cold-start fetch must not create a cache file",
            )
        finally:
            forecast_module.open_meteo_backoff_seconds = original_backoff
            if os.path.exists(json_path):
                os.remove(json_path)

    async def test_open_meteo_cache_present_falls_back_immediately_no_retry(self):
        """Cache present: a fetch failure falls back to the cache with NO retry.

        This is the steady-state path.  When a cached JSON exists, a forced
        refresh that fails must immediately return the cached payload and make
        exactly ONE network attempt (no retry, no added delay).  The existing
        cache file must be preserved (never overwritten on failure).
        """
        payload = await self._load_openmeteo_mock_payload()
        json_path = emhass_conf["data_path"] / "cached-open-meteo-forecast-b.json"
        # Seed a valid cache file on disk (this is the fallback content).
        cache_content = orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode()
        async with aiofiles.open(json_path, "w") as json_file:
            await json_file.write(cache_content)
        url_pattern = re.compile(r"https://api\.open-meteo\.com/.*")
        # Guard against accidental sleeping: any retry would be a test failure,
        # but zero the backoff regardless so a regression cannot stall the suite.
        original_backoff = forecast_module.open_meteo_backoff_seconds
        forecast_module.open_meteo_backoff_seconds = (0, 0, 0)
        try:
            with aioresponses() as mocked:
                # Only register a SINGLE failing response.  If the code retried,
                # the second attempt would raise (no mock left) and the test fails.
                mocked.get(url_pattern, status=504)
                # max_age=0 forces a refresh attempt while the cache is on disk.
                result = await self.fcst.get_cached_open_meteo_forecast_json(max_age=0)
                total_calls = sum(len(v) for v in mocked.requests.values())
            # Fell back to the cached payload (a dict), exactly one attempt made.
            self.assertIsInstance(result, dict)
            self.assertIn("minutely_15", result)
            self.assertEqual(total_calls, 1, "A present cache must NOT trigger retries on failure")
            # The cache file must be untouched (never overwritten on failure).
            async with aiofiles.open(json_path) as json_file:
                self.assertEqual(await json_file.read(), cache_content)
        finally:
            forecast_module.open_meteo_backoff_seconds = original_backoff
            if os.path.exists(json_path):
                os.remove(json_path)

    async def test_open_meteo_request_timeout_is_set(self):
        """A bounded per-request ClientTimeout is applied to the Open-Meteo fetch.

        A hanging Open-Meteo must not be able to stall the cycle.  Verify the
        session is created with a finite total timeout, and that a timeout on a
        cold start is handled like any other transient error (retried, then
        None when it persists).
        """
        json_path = emhass_conf["data_path"] / "cached-open-meteo-forecast-b.json"
        if os.path.exists(json_path):
            os.remove(json_path)
        # Assert the timeout constant is finite and sensible.
        self.assertIsNotNone(forecast_module.open_meteo_request_timeout)
        self.assertGreater(forecast_module.open_meteo_request_timeout, 0)

        # Capture the ClientTimeout the code passes to aiohttp.ClientSession.
        captured = {}
        real_session = aiohttp.ClientSession

        def _capture_session(*args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return real_session(*args, **kwargs)

        url_pattern = re.compile(r"https://api\.open-meteo\.com/.*")
        original_backoff = forecast_module.open_meteo_backoff_seconds
        forecast_module.open_meteo_backoff_seconds = (0, 0, 0)
        try:
            with unittest.mock.patch.object(aiohttp, "ClientSession", _capture_session):
                with aioresponses() as mocked:
                    # Simulate a hanging request that aiohttp surfaces as a timeout.
                    for _ in range(forecast_module.open_meteo_max_attempts):
                        mocked.get(url_pattern, exception=TimeoutError())
                    result = await self.fcst.get_cached_open_meteo_forecast_json()
            # The timeout is handled gracefully (no cache -> None after retries).
            self.assertIsNone(result)
            # A finite total timeout must have been supplied to the session.
            self.assertIsInstance(captured.get("timeout"), aiohttp.ClientTimeout)
            self.assertEqual(captured["timeout"].total, forecast_module.open_meteo_request_timeout)
        finally:
            forecast_module.open_meteo_backoff_seconds = original_backoff
            if os.path.exists(json_path):
                os.remove(json_path)

    def test_cloud_cover_to_irradiance(self):
        """Test the manual irradiance calculation from cloud cover."""
        # Create dummy cloud cover data
        cloud_cover = pd.Series(
            [0, 50, 100], index=pd.date_range("2021-01-01", periods=3, freq="1h")
        )
        cloud_cover = cloud_cover.tz_localize(self.fcst.time_zone)
        res = self.fcst.cloud_cover_to_irradiance(cloud_cover)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertIn("ghi", res.columns)
        self.assertIn("dni", res.columns)
        self.assertIn("dhi", res.columns)
        # Check basic physics: 0 cloud cover should have higher GHI than 100
        # (Assuming daytime, but solar position depends on lat/lon/time.
        #  Just checking structure is usually enough for coverage).

    def test_get_power_from_weather_single_system(self):
        """Test get_power_from_weather with a single PV system configuration."""
        # Force single string configuration (not list)
        self.plant_conf["pv_module_model"] = (
            "CSUN_Eurasia_Energy_Systems_Industry_and_Trade_CSUN295_60M"
        )
        self.plant_conf["pv_inverter_model"] = (
            "Fronius_International_GmbH__Fronius_Primo_5_0_1_208_240__240V_"
        )
        self.plant_conf["surface_tilt"] = 30
        self.plant_conf["surface_azimuth"] = 180
        self.plant_conf["modules_per_string"] = 8
        self.plant_conf["strings_per_inverter"] = 1
        # Re-initialize Forecast to apply new plant_conf
        self.fcst = Forecast(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            orjson.dumps(self.fcst.params).decode("utf-8"),
            emhass_conf,
            logger,
            get_data_from_file=self.get_data_from_file,
        )
        if not hasattr(self.fcst, "weather_forecast_method"):
            self.fcst.weather_forecast_method = self.optim_conf.get(
                "weather_forecast_method", "scrapper"
            )
        self.df_weather_scrap["ghi"] = 1000.0
        self.df_weather_scrap["dni"] = 900.0
        self.df_weather_scrap["dhi"] = 100.0
        self.df_weather_scrap["temp_air"] = 25.0
        self.df_weather_scrap["wind_speed"] = 2.0
        self.df_weather_scrap["precipitable_water"] = 0.5
        p_pv_forecast = self.fcst.get_power_from_weather(self.df_weather_scrap)
        self.assertIsInstance(p_pv_forecast, pd.Series)
        self.assertEqual(len(p_pv_forecast), len(self.df_weather_scrap))

    def test_get_model_selection(self):
        """
        Test the _get_model and _find_closest_model methods using the actual CEC databases.
        """
        # Load the databases using the configuration paths
        # We use self.fcst.emhass_conf to ensure we get the correct root path
        cec_modules_path = self.fcst.emhass_conf["root_path"] / "data" / "cec_modules.pbz2"
        cec_inverters_path = self.fcst.emhass_conf["root_path"] / "data" / "cec_inverters.pbz2"
        # Load Modules
        with bz2.BZ2File(cec_modules_path, "rb") as f:
            cec_modules = cPickle.load(f)
        # Load Inverters
        with bz2.BZ2File(cec_inverters_path, "rb") as f:
            cec_inverters = cPickle.load(f)
        # TEST 1: Retrieve Module by Exact Name
        # Using a specific known module from the database: 300W module
        target_module = "MEMC_Singapore_MEMC_M300AMC_27"
        model = self.fcst._get_model(target_module, cec_modules, "module")
        self.assertIsNotNone(model, "Should return a model for a valid name string")
        self.assertEqual(model.name, target_module, "Model name should match the requested string")
        self.assertAlmostEqual(model["STC"], 300.0, msg="Expected STC around 300W for this module")
        # TEST 2: Retrieve Module by Wattage (Integer)
        # Request a 300W module by integer. Should find the closest one (likely the one above or similar)
        model = self.fcst._get_model(300, cec_modules, "module")
        self.assertIsNotNone(model, "Should return a model for a valid integer power")
        # Verify the power is reasonably close to 300W (STC)
        self.assertAlmostEqual(
            model["STC"], 300.0, delta=10.0, msg="Selected module should be close to 300W"
        )
        # TEST 3: Retrieve Module by Wattage (String)
        # Request a "300" W module (string input). Logic should convert to float and find closest.
        model = self.fcst._get_model("300", cec_modules, "module")
        self.assertIsNotNone(model, "Should return a model for a valid string number")
        self.assertAlmostEqual(
            model["STC"], 300.0, delta=10.0, msg="Selected module should be close to 300W"
        )
        # TEST 4: Retrieve Inverter by Exact Name
        # Using a specific known inverter: ~5000W
        target_inverter = "INGETEAM_POWER_TECHNOLOGY_S_A___Ingecon_Sun_5U__208V_"
        model = self.fcst._get_model(target_inverter, cec_inverters, "inverter")
        self.assertIsNotNone(model, "Should return an inverter for a valid name string")
        self.assertEqual(model.name, target_inverter)
        # TEST 5: Retrieve Inverter by Wattage (Float)
        # Request 5000W inverter
        model = self.fcst._get_model(5000.0, cec_inverters, "inverter")
        self.assertIsNotNone(model, "Should return an inverter for a valid float power")
        # Check power (Paco is typical for AC power, Pdco for DC)
        power = model.get("Paco", model.get("Pdco", 0))
        self.assertAlmostEqual(
            power, 5000.0, delta=100.0, msg="Selected inverter should be close to 5000W"
        )
        # TEST 6: Test Fallback / Closest Match Logic
        # Request 292W. Should match 300W or 290W module.
        model = self.fcst._get_model(292, cec_modules, "module")
        self.assertIsNotNone(model)
        self.assertLess(
            abs(model["STC"] - 292), 50, "Should find a module within reasonable range of 292W"
        )

    # --- Shared helpers for the weather_forecast_pv_quantile_bias tests ---
    def _build_solcast_bias_payload(self, p50, p10, p90=7.0, n_periods=50, missing_p10_tail=False):
        """Build an inline Solcast payload anchored to the start of forecast_dates.

        Anchoring to forecast_dates[0] guarantees the Solcast timestamps overlap the
        optimization window — a historical anchor would be extrapolated by
        time-interpolation and skew the baseline. Constant P50/P10/P90 values keep the
        blend assertions trivial to reason about. When missing_p10_tail is set, one
        trailing element omits pv_estimate10 (the fallback-to-P50 edge case).
        """
        anchor_utc = self.fcst.forecast_dates[0].tz_convert("UTC")
        forecasts = [
            {
                "period_end": (anchor_utc + pd.Timedelta(minutes=30 * i)).isoformat(),
                "period": "PT30M",
                "pv_estimate": p50,
                "pv_estimate10": p10,
                "pv_estimate90": p90,
            }
            for i in range(n_periods)
        ]
        if missing_p10_tail:
            forecasts.append(
                {
                    "period_end": (anchor_utc + pd.Timedelta(minutes=30 * n_periods)).isoformat(),
                    "period": "PT30M",
                    "pv_estimate": p50,
                    # pv_estimate10 deliberately omitted
                    "pv_estimate90": p90,
                }
            )
        return {"forecasts": forecasts}

    def _setup_solcast_bias_env(self):
        """Wire up the mocked-Solcast environment shared by the bias tests.

        Sets passed_data/credentials, bypasses the daily-quota cap (these tests
        exercise the blend logic, not the rate limiter), and moves any pre-existing
        weather cache aside. Restores the cache and clears the bias key via addCleanup
        so nothing leaks into other tests even if an assertion fails. Returns the
        mocked Solcast GET URL.
        """
        self.fcst.params = {
            "passed_data": {
                "weather_forecast_cache": False,
                "weather_forecast_cache_only": False,
            }
        }
        self.fcst.retrieve_hass_conf["solcast_api_key"] = "test_key"
        self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = "test_roof"
        self.fcst._solcast_rate_limit_ok = lambda: True

        cache_path = emhass_conf["data_path"] / "weather_forecast_data.pkl"
        temp_path = emhass_conf["data_path"] / "temp_bias_weather_forecast_data.pkl"
        if os.path.isfile(cache_path):
            os.rename(cache_path, temp_path)

        def _restore():
            if os.path.isfile(temp_path):
                os.rename(temp_path, cache_path)
            self.fcst.optim_conf.pop("weather_forecast_pv_quantile_bias", None)

        self.addCleanup(_restore)

        days_solcast = int(len(self.fcst.forecast_dates) * self.fcst.freq.seconds / 3600)
        return f"https://api.solcast.com.au/rooftop_sites/test_roof/forecasts?hours={days_solcast}"

    async def _fetch_solcast_with_bias(self, get_url, payload, bias_value=_BIAS_UNSET):
        """Fetch a mocked Solcast forecast, optionally setting the bias param first.

        Passing the _BIAS_UNSET sentinel leaves the param absent (the default/no-op
        path); any other value is written to optim_conf before the call.
        """
        if bias_value is _BIAS_UNSET:
            self.fcst.optim_conf.pop("weather_forecast_pv_quantile_bias", None)
        else:
            self.fcst.optim_conf["weather_forecast_pv_quantile_bias"] = bias_value
        with aioresponses() as mocked:
            mocked.get(get_url, payload=payload)
            return await self.fcst.get_weather_forecast(method="solcast")

    # Test weather_forecast_pv_quantile_bias blending (Phase 1 — forecast side only)
    async def test_get_weather_forecast_solcast_pv_quantile_bias(self):
        """Verify that weather_forecast_pv_quantile_bias blends P50 and P10 correctly.

        Four sub-cases:
          (i)  param unset (default) == (ii) bias=0.0 == pure P50 path (no-op / backward compat)
          (iii) bias=1.0 => pure P10 result (fails on master, passes with fix)
          (iv)  bias=0.5 => linear midpoint (fails on master, passes with fix)

        Plus an edge case: an element with pv_estimate10 absent, bias=1.0 -> fallback to pv_estimate.
        """
        # P50 = 5.0 kW, P10 = 2.0 kW, P90 = 7.0 kW (ratios make assertions easy to reason about)
        P50, P10, P90 = 5.0, 2.0, 7.0
        payload = self._build_solcast_bias_payload(P50, P10, P90, missing_p10_tail=True)
        get_url = self._setup_solcast_bias_env()

        # (i) param unset (default = 0.0 / P50)
        df_unset = await self._fetch_solcast_with_bias(get_url, payload)
        self.assertIsInstance(df_unset, pd.DataFrame)
        self.assertIn("yhat", df_unset.columns)

        # (ii) explicit bias=0.0 (must equal (i))
        df_bias0 = await self._fetch_solcast_with_bias(get_url, payload, 0.0)
        self.assertIsInstance(df_bias0, pd.DataFrame)

        # (iii) bias=1.0 (pure P10) — FAILS on master, PASSES with fix
        df_bias1 = await self._fetch_solcast_with_bias(get_url, payload, 1.0)
        self.assertIsInstance(df_bias1, pd.DataFrame)

        # (iv) bias=0.5 (linear midpoint) — FAILS on master, PASSES with fix
        df_bias05 = await self._fetch_solcast_with_bias(get_url, payload, 0.5)
        self.assertIsInstance(df_bias05, pd.DataFrame)

        # All outputs should align with forecast_dates length
        for df_name, df in [
            ("unset", df_unset),
            ("bias0", df_bias0),
            ("bias1", df_bias1),
            ("bias05", df_bias05),
        ]:
            self.assertEqual(
                len(df),
                len(self.fcst.forecast_dates),
                msg=f"df_{df_name} length mismatch",
            )
            self.assertFalse(df["yhat"].isna().any(), msg=f"df_{df_name} has NaN values")

        # (i) == (ii): default is identical to explicit bias=0.0 (backward compat guarantee)
        np.testing.assert_array_almost_equal(
            df_unset["yhat"].values,
            df_bias0["yhat"].values,
            decimal=6,
            err_msg="unset != bias=0.0: backward compat broken",
        )

        # Identify the non-zero region: reindex may zero-fill rows that fall outside
        # the anchor window; restrict assertions to rows where P50 result > 1 W.
        nonzero_mask = df_bias0["yhat"].values > 1.0
        self.assertTrue(
            nonzero_mask.sum() > 0,
            "No non-zero P50 rows found — anchor timestamps do not overlap forecast_dates",
        )

        p50_vals = df_bias0["yhat"].values[nonzero_mask]
        # Expected P10 result: bias*P10 + (1-bias)*P50 = 1.0*2.0 + 0.0*5.0 = 2.0 kW
        # In W after *1000: ratio = P10/P50 = 0.4
        p10_expected = p50_vals * (P10 / P50)
        # Expected midpoint: 0.5*2.0 + 0.5*5.0 = 3.5 kW => ratio = 3.5/5.0 = 0.7
        p05_expected = p50_vals * ((0.5 * P10 + 0.5 * P50) / P50)

        # (iii) bias=1.0 must yield P10 values (this assertion FAILS on master)
        p10_actual = df_bias1["yhat"].values[nonzero_mask]
        np.testing.assert_allclose(
            p10_actual,
            p10_expected,
            rtol=1e-5,
            err_msg="bias=1.0 did not yield P10 values (expected P50 * 0.4)",
        )

        # (iv) bias=0.5 must yield the linear midpoint (this assertion FAILS on master)
        p05_actual = df_bias05["yhat"].values[nonzero_mask]
        np.testing.assert_allclose(
            p05_actual,
            p05_expected,
            rtol=1e-5,
            err_msg="bias=0.5 did not yield midpoint values (expected P50 * 0.7)",
        )

        # Edge case: element with pv_estimate10 absent + bias=1.0 must not crash
        self.assertIsInstance(df_bias1, pd.DataFrame, "bias=1.0 with missing pv_estimate10 crashed")

    # Test that invalid/edge weather_forecast_pv_quantile_bias values are handled safely
    async def test_get_weather_forecast_solcast_pv_quantile_bias_invalid_inputs(self):
        """Bad-type / out-of-range bias values must never crash or silently misbehave.

        - bool True (a YAML `true`) must NOT be treated as 1.0 -> falls back to P50.
        - a quoted string "0.5" must be coerced and applied (midpoint).
        - NaN must fall back to P50, not slip through as a silent no-op-without-warning.
        - out-of-range numerics (-1, 2) must clamp to [0, 1].
        Each case is checked by the resulting yhat ratio vs the pure-P50 baseline.
        """
        P50, P10 = 5.0, 2.0
        payload = self._build_solcast_bias_payload(P50, P10)
        get_url = self._setup_solcast_bias_env()

        # baseline (pure P50) to measure ratios against
        base = await self._fetch_solcast_with_bias(get_url, payload, 0.0)
        mask = base["yhat"].values > 1.0
        self.assertTrue(mask.sum() > 0)
        base_vals = base["yhat"].values[mask]

        # (bias_value, expected ratio of result to the P50 baseline)
        cases = [
            (True, 1.0),  # bool rejected -> P50
            ("0.5", (0.5 * P10 + 0.5 * P50) / P50),  # string coerced -> midpoint (0.7)
            (float("nan"), 1.0),  # NaN -> P50
            (-1.0, 1.0),  # clamp to 0 -> P50
            (2.0, P10 / P50),  # clamp to 1 -> P10 (0.4)
        ]
        for bias_value, ratio in cases:
            df = await self._fetch_solcast_with_bias(get_url, payload, bias_value)
            self.assertFalse(df["yhat"].isna().any(), msg=f"NaN in result for bias={bias_value!r}")
            np.testing.assert_allclose(
                df["yhat"].values[mask],
                base_vals * ratio,
                rtol=1e-5,
                err_msg=f"bias={bias_value!r} did not produce the expected ratio {ratio}",
            )

    # The quantile bias only has data to act on under the solcast method (the
    # only provider returning pv_estimate10). For any other method the knob must
    # warn the user it is being ignored, rather than silently doing nothing.
    async def test_get_weather_forecast_pv_quantile_bias_warns_for_non_solcast(self):
        """Setting the bias for a non-solcast method must warn and not crash."""
        self.fcst.optim_conf["weather_forecast_pv_quantile_bias"] = 0.5
        try:
            with self.assertLogs(logger, level="WARNING") as cm:
                df = await self.fcst.get_weather_forecast(method="csv")
            self.assertTrue(
                any("only applies to the 'solcast'" in msg for msg in cm.output),
                msg=f"expected a Solcast-only warning for method=csv, got: {cm.output}",
            )
            self.assertIsInstance(df, pd.DataFrame)
        finally:
            self.fcst.optim_conf.pop("weather_forecast_pv_quantile_bias", None)

    async def test_get_weather_forecast_pv_quantile_bias_zero_no_warn_non_solcast(self):
        """The default bias of 0 must NOT warn under a non-solcast method."""
        self.fcst.optim_conf["weather_forecast_pv_quantile_bias"] = 0.0
        try:
            df = await self.fcst.get_weather_forecast(method="csv")
            self.assertIsInstance(df, pd.DataFrame)
        finally:
            self.fcst.optim_conf.pop("weather_forecast_pv_quantile_bias", None)


class TestDstForecastDates(unittest.IsolatedAsyncioTestCase):
    """Standalone tests for the DST forecast-date-range fix.

    These tests do NOT require test_df_final.pkl so they can run in Docker
    without the full data file mount.
    """

    @staticmethod
    async def _build_params():
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
        return await utils.build_params(emhass_conf, secrets, config, logger)

    async def asyncSetUp(self):
        import pytz

        params = await self._build_params()
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        self.paris_tz = pytz.timezone("Europe/Paris")
        retrieve_hass_conf["time_zone"] = self.paris_tz
        # Force 15-min frequency and 7-day horizon for the DST test
        optim_conf["delta_forecast_daily"] = pd.Timedelta(days=7)
        self.retrieve_hass_conf = retrieve_hass_conf
        self.optim_conf = optim_conf
        self.plant_conf = plant_conf
        self.params_json = params_json

    def test_forecast_dates_length_consistent_with_get_forecast_dates_across_dst(self):
        """Forecast.forecast_dates length must match utils.get_forecast_dates across DST.

        Root cause: Forecast.__init__ previously used pd.Timedelta(days=N) which
        counts wall-clock hours, producing a different number of 15-min slots than
        utils.get_forecast_dates which uses pd.DateOffset(days=N) (calendar days).
        On a spring-forward DST day a 7-day 15-min horizon spans 167 wall-clock
        hours (668 slots) instead of 168 hours (672 slots).
        The fix replaces all Timedelta additions in Forecast.__init__ with DateOffset.
        """
        from datetime import datetime
        from unittest.mock import patch

        # Spring-forward for Paris 2025: 2025-03-30 02:00 -> 03:00
        # Start at midnight so the full 7-day window crosses the transition
        dst_start_naive = datetime(2025, 3, 30, 0, 0, 0)
        dst_start_ts = self.paris_tz.localize(dst_start_naive)

        # Build Forecast (no data file needed; only __init__ computes forecast_dates)
        fcst_dst = Forecast(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.params_json,
            emhass_conf,
            logger,
            get_data_from_file=True,  # flag only; no file access in __init__
        )

        # Override start so forecast_dates spans the spring-forward transition
        fcst_dst.start_forecast = dst_start_ts
        fcst_dst.end_forecast = (dst_start_ts + pd.DateOffset(days=7)).replace(microsecond=0)
        fcst_dst.forecast_dates = (
            pd.date_range(
                start=fcst_dst.start_forecast,
                end=fcst_dst.end_forecast - fcst_dst.freq,
                freq=fcst_dst.freq,
                tz=self.paris_tz,
            )
            .tz_convert("utc")
            .round(fcst_dst.freq, ambiguous="infer", nonexistent="shift_forward")
            .tz_convert(self.paris_tz)
        )

        # utils.get_forecast_dates is the reference (uses DateOffset)
        # fcst_dst.freq is the optimization_time_step Timedelta
        freq_minutes = int(fcst_dst.freq.seconds // 60)
        with patch("emhass.utils._get_now", return_value=dst_start_naive):
            ref_dates = utils.get_forecast_dates(freq_minutes, 7, self.paris_tz)

        self.assertEqual(
            len(fcst_dst.forecast_dates),
            len(ref_dates),
            f"Forecast.forecast_dates ({len(fcst_dst.forecast_dates)}) must match "
            f"get_forecast_dates ({len(ref_dates)}) across spring-forward DST",
        )
        # Crossing spring-forward loses one hour = 4 slots at 15 min
        self.assertLess(
            len(fcst_dst.forecast_dates),
            672,
            "Spring-forward DST should produce fewer than 672 slots for a 7-day 15-min window",
        )

    def test_forecast_dates_normal_day_equals_expected_slots(self):
        """On a normal day (no DST transition) forecast_dates has exactly N*24*(60/freq) slots."""
        from datetime import datetime
        from unittest.mock import patch

        # 2025-03-20 is a Thursday well before the spring-forward (2025-03-30),
        # so a 7-day window from 2025-03-20 to 2025-03-27 has no DST transition.
        normal_start_naive = datetime(2025, 3, 20, 0, 0, 0)
        normal_start_ts = self.paris_tz.localize(normal_start_naive)

        fcst_normal = Forecast(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            self.params_json,
            emhass_conf,
            logger,
            get_data_from_file=True,
        )

        fcst_normal.start_forecast = normal_start_ts
        fcst_normal.end_forecast = (normal_start_ts + pd.DateOffset(days=7)).replace(microsecond=0)
        fcst_normal.forecast_dates = (
            pd.date_range(
                start=fcst_normal.start_forecast,
                end=fcst_normal.end_forecast - fcst_normal.freq,
                freq=fcst_normal.freq,
                tz=self.paris_tz,
            )
            .tz_convert("utc")
            .round(fcst_normal.freq, ambiguous="infer", nonexistent="shift_forward")
            .tz_convert(self.paris_tz)
        )

        freq_minutes = int(fcst_normal.freq.seconds // 60)
        with patch("emhass.utils._get_now", return_value=normal_start_naive):
            ref_dates = utils.get_forecast_dates(freq_minutes, 7, self.paris_tz)

        self.assertEqual(len(fcst_normal.forecast_dates), len(ref_dates))
        # On a normal day the length must equal exactly 7 * 24 * (60 / freq_minutes) slots
        expected_slots = 7 * 24 * (60 // freq_minutes)
        self.assertEqual(len(fcst_normal.forecast_dates), expected_slots)


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
