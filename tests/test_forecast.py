#!/usr/bin/env python

import _pickle as cPickle
import bz2
import copy
import glob
import os
import pathlib
import pickle
import re
import tempfile
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
        # Time of day must be encoded continuously (no raw integer hour): the
        # raw hour feature caused hour-boundary sawtooth in the adjusted forecast
        self.assertNotIn("hour", self.fcst.x_adjust_pv.columns)
        self.assertIn("hour_sin", self.fcst.x_adjust_pv.columns)
        self.assertIn("hour_cos", self.fcst.x_adjust_pv.columns)
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

    # add_cyclic_hour_features requires a DatetimeIndex, like add_date_features
    async def test_add_cyclic_hour_features_requires_datetime_index(self):
        df = pd.DataFrame({"forecast": [100.0, 200.0]}, index=[0, 1])
        with self.assertRaises(ValueError):
            Forecast.add_cyclic_hour_features(df)

    # Regression test for the hour-boundary sawtooth: the time-of-day features
    # fed to the regressor must be continuous at sub-hourly resolution, so a
    # model with weight on them cannot introduce jumps at :00 that are absent
    # from its input. With the raw integer hour feature a constant input curve
    # produced steps of the full hour-coefficient at every hour boundary.
    async def test_pv_forecast_adjust_no_hour_boundary_jump(self):
        idx = pd.date_range("2026-04-19 10:00:00", periods=17, freq="15min", tz=self.fcst.time_zone)
        forecasted_pv = pd.DataFrame({"forecast": np.full(len(idx), 1000.0)}, index=idx)

        class _TimeFeatureModel:
            # Mimics a fitted linear model with weight on the time-of-day features
            def predict(self, X):
                return 1000.0 + 500.0 * X["hour_sin"].to_numpy() + 500.0 * X["hour_cos"].to_numpy()

        self.fcst.model_adjust_pv = _TimeFeatureModel()
        result = self.fcst.adjust_pv_forecast_predict(forecasted_pv=forecasted_pv)
        steps = result["adjusted_forecast"].diff().dropna()
        at_hour = steps[steps.index.minute == 0].abs()
        within_hour = steps[steps.index.minute != 0].abs()
        self.assertLessEqual(
            at_hour.max(),
            within_hour.max() * 1.5,
            "Adjusted forecast jumps at hour boundaries (sawtooth regression)",
        )

    # Issue #1026: curtailed timesteps (plus a one-step margin) must be excluded
    # from the PV adjustment training set; without a curtailment series the
    # training set is unchanged.
    async def test_pv_forecast_adjust_drops_curtailed_timesteps(self):
        idx = pd.date_range("2026-04-19 10:00:00", periods=16, freq="30min", tz=self.fcst.time_zone)
        data = pd.DataFrame(
            {
                self.fcst.var_pv: np.full(len(idx), 1000.0),
                self.fcst.var_pv_forecast: np.full(len(idx), 1100.0),
            },
            index=idx,
        )
        curtailment = pd.Series(0.0, index=idx)
        curtailment.iloc[5] = 500.0
        self.fcst.adjust_pv_forecast_data_prep(data, curtailment_series=curtailment)
        self.assertEqual(len(self.fcst.data_adjust_pv), len(idx) - 3)
        for pos in (4, 5, 6):
            self.assertNotIn(idx[pos], self.fcst.data_adjust_pv.index)
        # a curtailment series on a different index is aligned (missing -> 0)
        self.fcst.adjust_pv_forecast_data_prep(data, curtailment_series=curtailment.iloc[:2])
        self.assertEqual(len(self.fcst.data_adjust_pv), len(idx))
        # and without a curtailment series nothing is dropped
        self.fcst.adjust_pv_forecast_data_prep(data)
        self.assertEqual(len(self.fcst.data_adjust_pv), len(idx))

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

    @staticmethod
    def _open_meteo_covariate_payload(full: pd.DatetimeIndex) -> dict:
        """Synthetic Open-Meteo minutely_15 payload covering the ``full`` (tz-aware) index."""
        times = (full.tz_convert("UTC").astype("int64") // 10**9).tolist()
        n = len(full)
        return {
            "minutely_15": {
                "time": times,
                "temperature_2m": [20.0] * n,
                "relative_humidity_2m": [50.0] * n,
                "cloud_cover": [30.0] * n,
                "wind_speed_10m": [5.0] * n,
                "shortwave_radiation": [200.0] * n,
                "direct_radiation": [150.0] * n,
                "diffuse_radiation": [50.0] * n,
                "precipitation": [0.0] * n,
            }
        }

    async def test_build_weather_future_localizes_naive_window_index(self):
        """A tz-naive last-window index no longer crashes the weather horizon build (#1036).

        The optim path's websocket statistics retrieval used to hand this method a tz-naive
        index, and the naive future horizon crashed get_weather_covariates with
        'Cannot subtract tz-naive and tz-aware datetime-like objects'. The horizon must now
        reach get_weather_covariates as tz-aware, in the configured time zone.
        """
        from unittest.mock import MagicMock, patch

        num_lags = 16
        naive_window_index = self.fcst.forecast_dates[:8].tz_localize(None)
        data_last_window = pd.DataFrame(index=naive_window_index)
        # Naive wall times are interpreted as the configured local time zone.
        expected_index = pd.date_range(
            start=naive_window_index[-1] + self.fcst.freq,
            periods=num_lags,
            freq=self.fcst.freq,
        ).tz_localize(self.fcst.time_zone)
        full = pd.date_range(
            start=expected_index[0] - 4 * self.fcst.freq,
            end=expected_index[-1] + 4 * self.fcst.freq,
            freq=self.fcst.freq,
        )
        payload = self._open_meteo_covariate_payload(full)
        mock_mlf = MagicMock()
        mock_mlf.weather_features = ["temp_air"]
        mock_mlf.is_tuned = False
        mock_mlf.num_lags = num_lags

        with patch.object(self.fcst, "_fetch_open_meteo_covariates_json", return_value=payload):
            weather_future = await self.fcst._build_weather_future(data_last_window, mock_mlf)

        self.assertIsNotNone(weather_future)
        self.assertEqual(len(weather_future), num_lags)
        self.assertIsNotNone(weather_future.index.tz)
        self.assertEqual(weather_future.index.tz, self.fcst.time_zone)
        self.assertTrue(weather_future.index.equals(expected_index))

    async def test_build_weather_future_converts_aware_window_index(self):
        """A tz-aware last-window index in another zone keeps its instants.

        Cross-zone tz-aware input never crashed; the horizon is now normalized to the
        configured time zone, which must not move the instants.
        """
        from unittest.mock import MagicMock, patch

        num_lags = 16
        utc_window_index = self.fcst.forecast_dates[:8].tz_convert("UTC")
        data_last_window = pd.DataFrame(index=utc_window_index)
        expected_index = pd.date_range(
            start=utc_window_index[-1] + self.fcst.freq,
            periods=num_lags,
            freq=self.fcst.freq,
            tz="UTC",
        ).tz_convert(self.fcst.time_zone)
        full = pd.date_range(
            start=expected_index[0] - 4 * self.fcst.freq,
            end=expected_index[-1] + 4 * self.fcst.freq,
            freq=self.fcst.freq,
        )
        payload = self._open_meteo_covariate_payload(full)
        mock_mlf = MagicMock()
        mock_mlf.weather_features = ["temp_air"]
        mock_mlf.is_tuned = False
        mock_mlf.num_lags = num_lags

        with patch.object(self.fcst, "_fetch_open_meteo_covariates_json", return_value=payload):
            weather_future = await self.fcst._build_weather_future(data_last_window, mock_mlf)

        self.assertIsNotNone(weather_future)
        self.assertEqual(len(weather_future), num_lags)
        self.assertEqual(weather_future.index.tz, self.fcst.time_zone)
        self.assertTrue(weather_future.index.equals(expected_index))

    async def test_prepare_hass_load_data_uses_configured_time_zone(self):
        """_prepare_hass_load_data builds its internal RetrieveHass with the configured tz (#1036).

        With time_zone=None the websocket statistics retrieval path strips the index tz
        (tz_convert(None)), producing the naive index behind the #1036 crash.
        """
        from unittest.mock import patch

        # self.fcst.get_data_from_file is True by default, so after construction the code
        # takes the aiofiles-pickle branch, not rh.get_data() -- there's no natural await
        # failure to stop on. Raise a sentinel from the constructor itself to capture its
        # args without also mocking the unrelated file-read/prepare_data pipeline.
        class _CtorCapture(Exception):
            pass

        captured = {}

        def _capture_ctor(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            raise _CtorCapture()

        with patch.object(forecast_module, "RetrieveHass", side_effect=_capture_ctor):
            with self.assertRaises(_CtorCapture):
                await self.fcst._prepare_hass_load_data(3, "mlforecaster")

        # Signature: (hass_url, long_lived_token, freq, time_zone, params, emhass_conf, logger)
        passed_time_zone = captured["args"][3]
        self.assertIsNotNone(passed_time_zone)
        self.assertEqual(passed_time_zone, self.fcst.time_zone)

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

    # --- issue #997: list weather method keeps GHI/temp_air for thermal loads ---
    async def _build_list_fcst_pinned(self, with_thermal: bool):
        """Build a pinned 1-day list-method Forecast, optionally with a thermal load."""
        fcst, _, _ = await self._build_longer_list_forecast(48)
        fcst.optim_conf["delta_forecast_daily"] = pd.Timedelta(days=1)
        self._pin_forecast_to_date(fcst, "2024-06-15 00:00:00")
        fcst.params["passed_data"]["prediction_horizon"] = None
        fcst.optim_conf["def_load_config"] = (
            [{"thermal_config": {"window_area": 2.0, "u_value": 1.0}}] if with_thermal else []
        )
        return fcst

    def _fake_open_meteo_frame(self, fcst):
        """An open-meteo-shaped weather frame on the forecast index (ghi + temp_air)."""
        idx = fcst.forecast_dates_tz
        return pd.DataFrame(
            {
                "ghi": np.linspace(0.0, 800.0, len(idx)),
                "temp_air": np.linspace(10.0, 25.0, len(idx)),
            },
            index=idx,
        )

    async def test_get_weather_forecast_list_keeps_ghi_for_thermal(self):
        """#997: a thermal load + passed PV list still gets GHI/temp from open-meteo."""
        fcst = await self._build_list_fcst_pinned(with_thermal=True)
        fake = self._fake_open_meteo_frame(fcst)
        with unittest.mock.patch.object(
            fcst,
            "_get_weather_open_meteo",
            new=unittest.mock.AsyncMock(return_value=fake),
        ) as m:
            data = await fcst.get_weather_forecast(method="list")
        m.assert_awaited_once()
        self.assertIn("ghi", data.columns)
        self.assertIn("temp_air", data.columns)
        self.assertFalse(data["ghi"].isnull().any())
        # PV power (yhat) is exactly the passed list, untouched by the augmentation
        self.assertEqual(data["yhat"].iloc[0], 1)
        self.assertEqual(data["yhat"].iloc[-1], 48)
        # weather method stays "list" so get_power_from_weather still returns yhat
        self.assertEqual(fcst.weather_forecast_method, "list")

    async def test_get_weather_forecast_list_no_thermal_is_noop(self):
        """#997 no-op: with no thermal load, no open-meteo fetch fires, only yhat."""
        fcst = await self._build_list_fcst_pinned(with_thermal=False)
        with unittest.mock.patch.object(
            fcst,
            "_get_weather_open_meteo",
            new=unittest.mock.AsyncMock(return_value=self._fake_open_meteo_frame(fcst)),
        ) as m:
            data = await fcst.get_weather_forecast(method="list")
        m.assert_not_awaited()
        self.assertEqual(list(data.columns), ["yhat"])
        self.assertEqual(data["yhat"].iloc[0], 1)
        self.assertEqual(data["yhat"].iloc[-1], 48)

    async def test_get_weather_forecast_list_open_meteo_failure_is_soft(self):
        """#997 fail-soft: an open-meteo error leaves the plain list frame and warns."""
        fcst = await self._build_list_fcst_pinned(with_thermal=True)
        with unittest.mock.patch.object(
            fcst,
            "_get_weather_open_meteo",
            new=unittest.mock.AsyncMock(side_effect=aiohttp.ClientError("boom")),
        ):
            with self.assertLogs(logger, level="WARNING") as cm:
                data = await fcst.get_weather_forecast(method="list")
        self.assertEqual(list(data.columns), ["yhat"])
        self.assertIn("issue #997", "\n".join(cm.output))

    async def test_get_weather_forecast_list_open_meteo_none_is_soft(self):
        """#997 fail-soft: open-meteo returning None leaves the plain list frame."""
        fcst = await self._build_list_fcst_pinned(with_thermal=True)
        with unittest.mock.patch.object(
            fcst,
            "_get_weather_open_meteo",
            new=unittest.mock.AsyncMock(return_value=None),
        ):
            with self.assertLogs(logger, level="WARNING") as cm:
                data = await fcst.get_weather_forecast(method="list")
        self.assertEqual(list(data.columns), ["yhat"])
        self.assertIn("no data", "\n".join(cm.output))

    async def test_get_cached_forecast_data_list_method_refetches_stale(self):
        """#997: under the open-meteo weather augmentation (method='list'), a cache
        that does not cover the window is treated like open-meteo (deleted for a
        fresh refetch) rather than served best-effort stale with zero-filled GHI."""
        cache_path = emhass_conf["data_path"] / "weather_forecast_data.pkl"
        temp_path = emhass_conf["data_path"] / "temp_weather_forecast_data.pkl"
        if os.path.isfile(cache_path):
            os.rename(cache_path, temp_path)
        stale_index = pd.date_range(
            start=self.fcst.forecast_dates[0] - pd.Timedelta(days=5),
            periods=8,
            freq=self.fcst.freq,
        )
        stale = pd.DataFrame({"ghi": 100.0, "temp_air": 20.0}, index=stale_index)
        with open(cache_path, "wb") as f:
            pickle.dump(stale, f)
        self.fcst.weather_forecast_method = "list"
        try:
            result = await self.fcst.get_cached_forecast_data(str(cache_path))
            removed = not os.path.isfile(cache_path)
        finally:
            if os.path.isfile(cache_path):
                os.remove(cache_path)
            if os.path.isfile(temp_path):
                os.rename(temp_path, cache_path)
        self.assertIsNone(result)
        self.assertTrue(removed)

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
        method="list" internally call ``get_forecast_days_csv()``.  That used to
        read the wall clock, so neither could be pinned here and both were left
        out; since #1076 it builds from ``self.start_forecast``, which
        ``_pin_forecast_to_date`` sets, so their slot counts are asserted too.
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

        # The two list-priced paths, now that the pinned start reaches them.
        df_input = pd.DataFrame(
            {
                "p_pv_forecast": p_pv_forecast.values[:, 0],
                "p_load_forecast": p_load_forecast.values,
            },
            index=p_pv_forecast.index,
        )
        df_cost = fcst.get_load_cost_forecast(df_input, method="list")
        self.assertEqual(len(df_cost), expected_last)
        self.assertEqual(df_cost[fcst.var_load_cost].isna().sum(), 0)
        self.assertEqual(df_cost[fcst.var_load_cost].values[0], 1)
        self.assertEqual(df_cost[fcst.var_load_cost].values[-1], expected_last)

        df_price = fcst.get_prod_price_forecast(df_cost, method="list")
        self.assertEqual(len(df_price), expected_last)
        self.assertEqual(df_price[fcst.var_prod_price].isna().sum(), 0)
        self.assertEqual(df_price[fcst.var_prod_price].values[0], 1)
        self.assertEqual(df_price[fcst.var_prod_price].values[-1], expected_last)

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

    # --- Multi-rooftop Solcast quota accounting + UTC quota day ---
    #
    # The daily counter reserves Solcast provider-call budget atomically for
    # ALL configured rooftop IDs before any HTTP request is made, so a
    # reservation that doesn't fit under the cap can never yield a partial
    # multi-rooftop aggregate. Keyed by UTC calendar date, since Solcast's
    # quota resets on the UTC day, not the site's local day.

    def _isolate_solcast_counter_dir(self):
        """Route the quota counter file into a private per-test temp dir so
        these tests never touch a real developer/runtime quota counter."""
        counter_dir = tempfile.TemporaryDirectory()
        self.addCleanup(counter_dir.cleanup)
        patcher = unittest.mock.patch(
            "emhass.forecast.tempfile.gettempdir", return_value=counter_dir.name
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self._counter_dir = counter_dir.name

    def _read_solcast_counter(self):
        paths = glob.glob(os.path.join(self._counter_dir, "emhass_solcast_calls_*.count"))
        assert len(paths) == 1, f"expected exactly one counter file, found {paths}"
        with open(paths[0]) as f:
            return int(f.read().strip())

    async def _fetch_solcast_quota_mock(self, roof_ids_csv):
        """Configure the given comma-separated rooftop IDs and fetch, with a
        mock registered for every possible roof (111111/222222) regardless of
        how many are actually configured -- so a rejected reservation proves
        zero requests even though a mock exists that would have succeeded.
        Caller must isolate the counter dir first (needed before any
        pre-loading reservation call, so it isn't done here)."""
        self.fcst.params = {
            "passed_data": {"weather_forecast_cache": False, "weather_forecast_cache_only": False}
        }
        self.fcst.retrieve_hass_conf["solcast_api_key"] = "k"
        self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = roof_ids_csv
        if os.path.isfile(emhass_conf["data_path"] / "weather_forecast_data.pkl"):
            os.remove(emhass_conf["data_path"] / "weather_forecast_data.pkl")
        payload = {
            "forecasts": [
                {
                    "period_end": "2026-08-22T10:00:00.0000000Z",
                    "period": "PT30M",
                    "pv_estimate": 0.5,
                }
            ]
        }
        with aioresponses() as mocked:
            for rid in ("111111", "222222"):
                mocked.get(
                    f"https://api.solcast.com.au/rooftop_sites/{rid}/forecasts?hours=24",
                    payload=payload,
                )
            res = await self.fcst.get_weather_forecast(method="solcast")
            return res, len(mocked.requests)

    # (initial reserved count, roof count) -> allowed?, final count, actual
    # HTTP requests. Covers 1/2-roof reservation, the 6+2->8 boundary, 7+2
    # rejection (no partial aggregate), and the 8-call cap for both roof
    # counts.
    async def test_solcast_quota_reservation_table(self):
        cases = [
            (0, 1, True, 1, 1),
            (0, 2, True, 2, 2),
            (6, 2, True, 8, 2),
            (7, 2, False, 7, 0),
            (8, 1, False, 8, 0),
            (8, 2, False, 8, 0),
        ]
        for initial, n_roofs, allowed, final_count, n_requests in cases:
            with self.subTest(initial=initial, n_roofs=n_roofs):
                roof_ids_csv = ",".join(["111111", "222222"][:n_roofs])
                self._isolate_solcast_counter_dir()
                if initial:
                    self.assertTrue(self.fcst._solcast_rate_limit_ok(required_calls=initial))
                res, n_http = await self._fetch_solcast_quota_mock(roof_ids_csv)
                self.assertEqual(res is not False, allowed)
                if allowed:
                    self.assertIsInstance(res, pd.DataFrame)
                else:
                    self.assertNotIsInstance(res, pd.DataFrame)
                self.assertEqual(n_http, n_requests)
                self.assertEqual(self._read_solcast_counter(), final_count)

    async def test_solcast_quota_lock_error_path_preserved(self):
        """The OSError/ValueError-safe fallback (return False, log an error)
        must still work with the required_calls-aware signature."""
        with unittest.mock.patch("builtins.open", side_effect=OSError("disk full")):
            self.assertFalse(self.fcst._solcast_rate_limit_ok(required_calls=2))

    async def test_solcast_quota_single_rooftop_default(self):
        """Calling with no explicit required_calls (the pre-existing call
        pattern used elsewhere) still reserves exactly 1."""
        self._isolate_solcast_counter_dir()
        self.assertTrue(self.fcst._solcast_rate_limit_ok())
        self.assertEqual(self._read_solcast_counter(), 1)

    async def test_solcast_quota_uses_utc_day(self):
        """UTC quota-day semantics in one deterministic test (no wall clock):
        local midnight does not reset the counter, but real UTC midnight
        does start a fresh one."""
        fixed = {"now": pd.Timestamp("2026-01-01 23:30:00", tz="UTC")}

        def fake_now(tz=None):
            return fixed["now"].tz_convert(tz) if tz is not None else fixed["now"]

        patcher = unittest.mock.patch.object(pd.Timestamp, "now", side_effect=fake_now)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._isolate_solcast_counter_dir()

        # 23:30 UTC on Jan 1 == 00:30 CET Jan 2 (Europe/Paris in winter): local
        # calendar date has already rolled over, UTC has not -- must still
        # land on the Jan-1 UTC-day file.
        self.assertTrue(self.fcst._solcast_rate_limit_ok(required_calls=5))
        self.assertEqual(self._read_solcast_counter(), 5)
        paths = glob.glob(os.path.join(self._counter_dir, "emhass_solcast_calls_*.count"))
        self.assertIn("emhass_solcast_calls_2026-01-01.count", paths[0])

        # Real UTC midnight rollover creates a fresh counter, not reused.
        fixed["now"] = pd.Timestamp("2026-01-02 00:05:00", tz="UTC")
        self.assertTrue(self.fcst._solcast_rate_limit_ok())
        paths = sorted(glob.glob(os.path.join(self._counter_dir, "emhass_solcast_calls_*.count")))
        self.assertEqual(len(paths), 2, "a new UTC-day counter file must be created, not reused")
        with open(paths[1]) as f:
            self.assertEqual(int(f.read().strip()), 1, "the new UTC day's counter starts fresh")

    async def test_solcast_quota_reservation_fails_closed_without_lock(self):
        """If the lock cannot be acquired the reservation fails closed
        (returns False, reserves nothing) rather than proceeding without
        atomicity -- this is also the observable behaviour of the "neither
        fcntl nor msvcrt" fallback branch."""
        self._isolate_solcast_counter_dir()
        with unittest.mock.patch(
            "emhass.forecast._flock_acquire", side_effect=OSError("no locking primitive")
        ):
            self.assertFalse(self.fcst._solcast_rate_limit_ok(required_calls=2))
        # The failed attempt reserved nothing: the next real call starts at 1.
        self.assertTrue(self.fcst._solcast_rate_limit_ok(required_calls=1))
        self.assertEqual(self._read_solcast_counter(), 1)

    @unittest.skipUnless(os.name == "nt", "Windows msvcrt locking branch")
    def test_flock_windows_mutual_exclusion(self):
        """On Windows, _flock_acquire takes an exclusive lock a second file
        handle cannot take until _flock_release runs."""
        import msvcrt

        from emhass.forecast import _flock_acquire, _flock_release

        counter_dir = tempfile.TemporaryDirectory()
        self.addCleanup(counter_dir.cleanup)
        path = os.path.join(counter_dir.name, "lock.count")
        with open(path, "a+") as f1, open(path, "a+") as f2:
            _flock_acquire(f1)
            try:
                f2.seek(0)
                with self.assertRaises(OSError):
                    msvcrt.locking(f2.fileno(), msvcrt.LK_NBLCK, 1)
            finally:
                _flock_release(f1)
            # The region is free again once released.
            f2.seek(0)
            msvcrt.locking(f2.fileno(), msvcrt.LK_NBLCK, 1)
            msvcrt.locking(f2.fileno(), msvcrt.LK_UNLCK, 1)

    async def test_solcast_empty_rooftop_id_rejected_before_reservation(self):
        """An empty / whitespace / separator-only solcast_rooftop_id is
        rejected before any quota reservation and before any HTTP request."""
        for bad in ("", "   ", ",", " , "):
            with self.subTest(value=repr(bad)):
                self._isolate_solcast_counter_dir()
                self.fcst.params = {
                    "passed_data": {
                        "weather_forecast_cache": False,
                        "weather_forecast_cache_only": False,
                    }
                }
                self.fcst.retrieve_hass_conf["solcast_api_key"] = "k"
                self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = bad
                if os.path.isfile(emhass_conf["data_path"] / "weather_forecast_data.pkl"):
                    os.remove(emhass_conf["data_path"] / "weather_forecast_data.pkl")
                with aioresponses() as mocked:
                    res = await self.fcst.get_weather_forecast(method="solcast")
                self.assertFalse(res)
                self.assertEqual(len(mocked.requests), 0)
                self.assertFalse(
                    glob.glob(os.path.join(self._counter_dir, "emhass_solcast_calls_*.count")),
                    "no quota counter file must be created",
                )

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
        self.fcst._solcast_rate_limit_ok = lambda *a, **kw: True

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
        with patch("emhass.utils._get_now", return_value=dst_start_ts):
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
        with patch("emhass.utils._get_now", return_value=normal_start_ts):
            ref_dates = utils.get_forecast_dates(freq_minutes, 7, self.paris_tz)

        self.assertEqual(len(fcst_normal.forecast_dates), len(ref_dates))
        # On a normal day the length must equal exactly 7 * 24 * (60 / freq_minutes) slots
        expected_slots = 7 * 24 * (60 // freq_minutes)
        self.assertEqual(len(fcst_normal.forecast_dates), expected_slots)


class TestGetMixForecast(unittest.TestCase):
    """Unit tests for the static Forecast.get_mix_forecast mix-correction helper.

    The callers pass the forecast as a pandas Series and the current real
    values as a DataFrame keyed by sensor column (see get_power_from_weather /
    get_load_forecast).
    """

    def test_missing_sensor_column_returns_forecast_unchanged(self):
        # Issue #764: when the forecast is supplied as a runtime list, df_now has
        # no column for the configured sensor, so there is no live value to blend.
        # get_mix_forecast must skip the correction, not raise KeyError.
        col = "sensor.pv_production_watts"
        forecast = pd.Series([1000.0, 900.0, 800.0])
        df_now = pd.DataFrame({"sensor.other": [42]})  # lacks `col`
        out = Forecast.get_mix_forecast(df_now, forecast.copy(), 0.5, 0.5, col)
        pd.testing.assert_series_equal(out, forecast)

    def test_empty_df_now_returns_forecast_unchanged(self):
        col = "sensor.pv_production_watts"
        forecast = pd.Series([1000.0, 900.0, 800.0])
        out = Forecast.get_mix_forecast(pd.DataFrame(), forecast.copy(), 0.5, 0.5, col)
        pd.testing.assert_series_equal(out, forecast)

    def test_present_sensor_column_still_blends_first_step(self):
        # Counterfactual: when df_now HAS the column, the first step is still
        # blended, so the guard does not disable the normal correction path.
        col = "sensor.pv_production_watts"
        forecast = pd.Series([1000.0, 900.0, 800.0])
        df_now = pd.DataFrame({col: [600, 500]})  # latest real value = 500
        out = Forecast.get_mix_forecast(df_now, forecast.copy(), 0.5, 0.5, col)
        # first step = round(0.5*1000 + 0.5*500) = 750; the rest are unchanged
        self.assertEqual(int(out.iloc[0]), 750)
        self.assertEqual(int(out.iloc[1]), 900)
        self.assertEqual(int(out.iloc[2]), 800)


class TestForecastDatesTieAlignment(unittest.IsolatedAsyncioTestCase):
    """Forecast date-range construction when now() lands exactly on a half-interval tie.

    With ``method_ts_round: "nearest"`` a constructor running at exactly HH:15:00
    or HH:45:00 (with a 30-min step) makes every stamp of the built range an
    exact round-to-freq tie; rounding the whole index stamp-by-stamp then
    collapses neighbouring stamps into duplicates via round-half-to-even, and the
    first downstream index assignment raises "ValueError: Length mismatch:
    Expected axis has 24/25 elements, new values have 48". These tests pin
    ``pd.Timestamp.now`` to tie seconds to prove the index stays unique, and pin
    non-tie seconds to prove the aligned-start construction is a no-op for all
    three ``method_ts_round`` modes.
    """

    async def asyncSetUp(self):
        import pytz

        params = await TestForecast.get_test_params()
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        self.retrieve_hass_conf = retrieve_hass_conf
        self.optim_conf = optim_conf
        self.plant_conf = plant_conf
        self.params = params
        self.time_zone = pytz.timezone("Europe/Paris")

    def _build_forecast(self, now_utc, method_ts_round="nearest", tz_name=None, step=None):
        """Construct a Forecast with ``pd.Timestamp.now`` pinned to ``now_utc``.

        Only ``now`` is patched (works on pandas 2.2 and 3.x, which both allow
        class attribute assignment on ``Timestamp``), so every other use of the
        class keeps its real behaviour. The patch scope is confined to the
        constructor call, the only place ``Forecast.__init__`` reads the clock.
        """
        import pytz

        conf = copy.deepcopy(self.retrieve_hass_conf)
        conf["method_ts_round"] = method_ts_round
        conf["time_zone"] = pytz.timezone(tz_name) if tz_name else self.time_zone
        if step is not None:
            conf["optimization_time_step"] = step
        real_timestamp = pd.Timestamp

        def _pinned_now(tz=None):
            ts = real_timestamp(now_utc, tz="UTC")
            return ts.tz_convert(tz) if tz is not None else ts.tz_localize(None)

        with unittest.mock.patch.object(pd.Timestamp, "now", staticmethod(_pinned_now)):
            fcst = Forecast(
                conf,
                self.optim_conf,
                self.plant_conf,
                copy.deepcopy(self.params),
                emhass_conf,
                logger,
                get_data_from_file=True,
            )
        return fcst

    def test_nearest_tie_second_keeps_forecast_dates_unique(self):
        # 03:45:00 UTC collapses to 24 unique stamps on the unfixed builder,
        # 03:15:00 UTC to 25 - the two lengths seen in the CI failures.
        for now_utc in ("2026-08-05 03:45:00", "2026-08-05 03:15:00"):
            with self.subTest(now_utc=now_utc):
                fcst = self._build_forecast(now_utc)
                fd = fcst.forecast_dates
                self.assertEqual(len(fd), 48)
                self.assertTrue(
                    fd.is_unique,
                    f"forecast_dates has duplicates: {list(fd[fd.duplicated()])}",
                )
                self.assertTrue((fd[1:] - fd[:-1] == fcst.freq).all())

    async def test_typical_load_forecast_survives_tie_second(self):
        # End-to-end repro of the CI crash: on the unfixed builder this raises
        # ValueError("Length mismatch: Expected axis has 24 elements, new
        # values have 48") when _get_load_forecast_typical assigns
        # forecast_dates as the output index.
        fcst = self._build_forecast("2026-08-05 03:45:00")
        p_load_forecast = await fcst.get_load_forecast(method="typical")
        self.assertEqual(len(p_load_forecast), len(fcst.forecast_dates))
        self.assertTrue(p_load_forecast.index.equals(fcst.forecast_dates))

    def test_tie_second_across_dst_transitions(self):
        # A tie-second start whose one-day range crosses a DST transition:
        # spring-forward day has 46 half-hour slots, fall-back day 50, and the
        # index must stay unique through both.
        for now_utc, expected_len in (
            ("2026-03-28 22:45:00", 46),  # crosses Paris spring-forward 2026-03-29
            ("2026-10-24 22:45:00", 50),  # crosses Paris fall-back 2026-10-25
        ):
            with self.subTest(now_utc=now_utc):
                fcst = self._build_forecast(now_utc)
                fd = fcst.forecast_dates
                self.assertEqual(len(fd), expected_len)
                self.assertTrue(
                    fd.is_unique,
                    f"forecast_dates has duplicates: {list(fd[fd.duplicated()])}",
                )

    def test_non_tie_starts_match_rounded_index_for_all_modes(self):
        # Counterfactual no-op guard: away from tie seconds the aligned-start
        # construction must reproduce, stamp for stamp, what the previous
        # build-then-round pipeline produced, for all three method_ts_round
        # modes. This recomputes that pipeline inline as the expectation, so it
        # passes on the old builder too - it pins byte-identity, not the fix.
        # One instant rounds down and one rounds up, so a floor posing as a
        # round cannot slip through.
        for now_utc in ("2026-08-05 03:37:23", "2026-08-05 03:52:23"):
            for mode in ("nearest", "first", "last"):
                with self.subTest(now_utc=now_utc, mode=mode):
                    fcst = self._build_forecast(now_utc, method_ts_round=mode)
                    tz = fcst.time_zone
                    start_raw = pd.Timestamp(now_utc, tz="UTC").tz_convert(tz)
                    if mode == "first":
                        base_start = start_raw.floor(freq=fcst.freq)
                        self.assertEqual(fcst.start_forecast, base_start)
                    elif mode == "last":
                        base_start = start_raw.ceil(freq=fcst.freq)
                        self.assertEqual(fcst.start_forecast, base_start)
                    else:
                        base_start = start_raw
                    base_end = (base_start + pd.DateOffset(days=1)).replace(microsecond=0)
                    expected = (
                        pd.date_range(
                            start=base_start,
                            end=base_end - fcst.freq,
                            freq=fcst.freq,
                            tz=tz,
                        )
                        .tz_convert("utc")
                        .round(fcst.freq, ambiguous="infer", nonexistent="shift_forward")
                        .tz_convert(tz)
                    )
                    self.assertTrue(fcst.forecast_dates.equals(expected))

    def test_nearest_alignment_rounds_in_utc_not_local_wall_time(self):
        # Australia/Adelaide sits at UTC+9:30, so with a 60-min step the local
        # wall-time grid and the UTC grid disagree by 30 minutes. The old
        # pipeline rounded the built index in UTC; the aligned start must take
        # the same route. A local wall-time round would land on :00 local,
        # 30 minutes away from every stamp the old builder produced.
        fcst = self._build_forecast(
            "2026-08-05 03:37:23",
            tz_name="Australia/Adelaide",
            step=pd.Timedelta("1h"),
        )
        expected_start = (
            pd.Timestamp("2026-08-05 03:37:23", tz="UTC").round("1h").tz_convert(fcst.time_zone)
        )
        self.assertEqual(fcst.forecast_dates[0], expected_start)
        self.assertEqual(len(fcst.forecast_dates), 24)
        self.assertTrue(fcst.forecast_dates.is_unique)


class TestForecastDaysCsvStartAlignment(unittest.IsolatedAsyncioTestCase):
    """The csv/list forecast grid must match the one the input data is indexed on.

    ``Forecast.__init__`` freezes ``self.start_forecast`` (and ``forecast_dates``)
    from the clock at set-input-data time. ``get_forecast_days_csv`` used to read
    the clock a second time and round it again, so a run whose data prep straddled
    a rounding boundary built a grid one step off from ``df_final``. The strict
    lookup in ``_extract_daily_forecast`` then raised
    ``KeyError: [Timestamp(...)] not in index`` naming the stamp one step behind
    (issue #1076, reported at 15 min/"first" and again at 30 min/"nearest").
    """

    TZ = "Europe/Brussels"
    HORIZON = 10

    async def asyncSetUp(self):
        import pytz

        self.params = await TestForecast.get_test_params()
        self.params["passed_data"]["prediction_horizon"] = self.HORIZON
        self.params["passed_data"]["load_cost_forecast"] = [0.25] * 200
        self.params["passed_data"]["prod_price_forecast"] = [0.05] * 200
        params_json = orjson.dumps(self.params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        retrieve_hass_conf["optimization_time_step"] = pd.to_timedelta(15, "minutes")
        retrieve_hass_conf["time_zone"] = pytz.timezone(self.TZ)
        self.retrieve_hass_conf = retrieve_hass_conf
        self.optim_conf = optim_conf
        self.plant_conf = plant_conf
        self.params_json = params_json

    @staticmethod
    def _pinned_now(now_local, tz_name):
        """Patch only ``pd.Timestamp.now``, fixed to one instant.

        ``Forecast`` has no injected clock seam to patch instead, so ``now``
        itself is the narrowest available target; every other use of
        ``Timestamp`` keeps its real behaviour. Unlike
        ``TestForecastDatesTieAlignment._build_forecast``, which patches only
        for the duration of construction, this returns the context manager so
        a single test can pin two instants in turn: one for ``__init__``, one
        for the later call that exposes the skew.
        """
        real_timestamp = pd.Timestamp
        fixed_utc = real_timestamp(now_local, tz=tz_name).tz_convert("UTC")

        def _now(tz=None):
            return fixed_utc.tz_convert(tz) if tz is not None else fixed_utc.tz_localize(None)

        return unittest.mock.patch.object(pd.Timestamp, "now", staticmethod(_now))

    def _build(self, init_now, method_ts_round="first"):
        conf = copy.deepcopy(self.retrieve_hass_conf)
        conf["method_ts_round"] = method_ts_round
        with self._pinned_now(init_now, self.TZ):
            return Forecast(
                conf,
                self.optim_conf,
                self.plant_conf,
                self.params_json,
                emhass_conf,
                logger,
                get_data_from_file=True,
            )

    def _dayahead_input(self, fcst):
        """Mirrors df_input_data_dayahead: indexed on the frozen forecast dates."""
        return pd.DataFrame(
            {"p_pv_forecast": 0.0, "p_load_forecast": 500.0},
            index=fcst.forecast_dates[: self.HORIZON],
        )

    def test_grid_start_is_the_frozen_start_not_the_clock(self):
        """The returned grid starts at ``self.start_forecast`` however late it is called."""
        fcst = self._build("2026-08-16 17:44:55")
        # Call it a full step later than construction: the old builder rounded
        # this second read to 17:45 and started the grid there instead.
        with self._pinned_now("2026-08-16 17:45:09", self.TZ):
            dates = fcst.get_forecast_days_csv(timedelta_days=0)
        self.assertEqual(dates[0], fcst.start_forecast)
        self.assertTrue(dates.equals(fcst.forecast_dates[: self.HORIZON]))

    def test_boundary_straddle_does_not_raise(self):
        """Init at 17:44:55 (floors to 17:30), load-cost call at 17:45:09 (floors to 17:45)."""
        fcst = self._build("2026-08-16 17:44:55")
        df = self._dayahead_input(fcst)
        self.assertEqual(str(df.index[0]), "2026-08-16 17:30:00+02:00")
        with self._pinned_now("2026-08-16 17:45:09", self.TZ):
            out = fcst.get_load_cost_forecast(df, method="list")
        self.assertEqual(len(out), self.HORIZON)
        self.assertTrue(out.index.equals(df.index))
        self.assertEqual(out[fcst.var_load_cost].isna().sum(), 0)
        self.assertTrue((out[fcst.var_load_cost] == 0.25).all())

    def test_boundary_straddle_does_not_raise_prod_price(self):
        """The production-price list path shares the builder and the strict lookup."""
        fcst = self._build("2026-08-16 17:44:55")
        df = self._dayahead_input(fcst)
        with self._pinned_now("2026-08-16 17:45:09", self.TZ):
            out = fcst.get_prod_price_forecast(df, method="list")
        self.assertEqual(len(out), self.HORIZON)
        self.assertEqual(out[fcst.var_prod_price].isna().sum(), 0)

    def test_nearest_half_hour_tie_boundary(self):
        """The reporter's second config: 30 min steps, "nearest", crash only on the :15 run.

        Round-half-to-even sends :15:00 down to the hour and :45:00 up, so only a
        run firing at :15 has its two clock reads disagree.
        """
        self.retrieve_hass_conf["optimization_time_step"] = pd.to_timedelta(30, "minutes")
        fcst = self._build("2026-08-17 08:15:00", method_ts_round="nearest")
        df = self._dayahead_input(fcst)
        with self._pinned_now("2026-08-17 08:15:44", self.TZ):
            out = fcst.get_load_cost_forecast(df, method="list")
        self.assertEqual(len(out), self.HORIZON)
        self.assertTrue(out.index.equals(df.index))

    def test_perfect_optim_list_path_prices_are_not_shifted(self):
        """The silent case: ``list_and_perfect`` never raises, it just mis-prices.

        The perfect-optim list path slices with ``between_time`` instead of the
        strict ``.loc``, so a straddling run came back the right length with the
        wrong values and nothing to notice: on master the tail repeats
        (``[0.17, 0.18, 0.18]``) where it should read ``[0.17, 0.18, 0.19]``.
        """
        prices = [round(0.10 + 0.01 * i, 2) for i in range(200)]
        self.params["passed_data"]["load_cost_forecast"] = prices
        self.params_json = orjson.dumps(self.params).decode("utf-8")
        fcst = self._build("2026-08-16 17:44:55")
        df = self._dayahead_input(fcst)
        with self._pinned_now("2026-08-16 17:45:09", self.TZ):
            out = fcst.get_load_cost_forecast(df, method="list", list_and_perfect=True)
        self.assertEqual(len(out), self.HORIZON)
        # The frozen grid starts at slot 0 of the list, so the horizon is the
        # first HORIZON prices in order, with no repeated tail value.
        self.assertEqual(list(out[fcst.var_load_cost]), prices[: self.HORIZON])

    def test_same_step_call_is_unchanged(self):
        """Counterfactual: both reads inside one step behaved correctly before and after."""
        fcst = self._build("2026-08-16 17:44:45")
        df = self._dayahead_input(fcst)
        with self._pinned_now("2026-08-16 17:44:58", self.TZ):
            out = fcst.get_load_cost_forecast(df, method="list")
        self.assertEqual(len(out), self.HORIZON)
        self.assertTrue(out.index.equals(df.index))
        self.assertTrue((out[fcst.var_load_cost] == 0.25).all())


class _FakeRecorderRetrieveHass:
    """Stand-in for RetrieveHass simulating a recorder with a bounded history.

    ``get_data`` synthesizes ``min(requested, available_days)`` full days of
    30-min load samples (no partial "today", the conservative case), so the
    number of samples the forecaster receives is a deterministic function of the
    days the caller asked for -- which is exactly what the #1067 fix must size
    correctly. The class records the last constructed instance and the requested
    day count for assertions.
    """

    last_instance = None
    available_days = None  # None = recorder has whatever is asked for

    def __init__(self, hass_url, long_lived_token, freq, time_zone, params, emhass_conf, logger):
        self.freq = freq
        self.time_zone = time_zone
        self.requested_days = None
        self.df_final = None
        type(self).last_instance = self

    async def get_data(self, days_list, var_list, **kwargs):
        # get_days_list(N) yields N prior days + today, i.e. N+1 stamps.
        self.requested_days = len(days_list) - 1
        self._var = var_list[0]
        available = type(self).available_days
        n_days = self.requested_days if available is None else min(self.requested_days, available)
        samples_per_day = int(pd.Timedelta(days=1) / self.freq)
        periods = max(1, n_days * samples_per_day)
        end = pd.Timestamp.now(tz=self.time_zone).floor(self.freq)
        index = pd.date_range(end=end, periods=periods, freq=self.freq)
        values = 500.0 + 200.0 * np.sin(np.arange(periods) * 2 * np.pi / samples_per_day)
        self.df_final = pd.DataFrame({self._var: values}, index=index)
        return True

    def prepare_data(self, var_load, **kwargs):
        self.df_final = self.df_final.rename(columns={self._var: var_load + "_positive"})
        return True


class TestMlforecasterLastWindowSizing(unittest.IsolatedAsyncioTestCase):
    """The mlforecaster load-forecast history retrieval is sized from the model (#1067).

    The optimization callers (naive-mpc-optim, dayahead-optim) pass
    ``days_min_load_forecast = delta_forecast_daily.days`` (default 1), which
    says nothing about how many past samples the auto-regressive model needs as
    its skforecast ``last_window``. A model with 144 lags at a 30-min step needs
    3 days of history, so every optimization run used to die inside skforecast
    with "`last_window` must have as many values as needed to generate the
    predictors". The retrieval window must be sized from the loaded model, and a
    still-too-short history must abort with a clear error instead of a raw
    skforecast ValueError.
    """

    async def asyncSetUp(self):
        params = await TestForecast.get_test_params()
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        self.retrieve_hass_conf = retrieve_hass_conf
        self.optim_conf = optim_conf
        self.plant_conf = plant_conf
        self.params = params
        self.var_load = retrieve_hass_conf["sensor_power_load_no_var_loads"]
        self.time_zone = retrieve_hass_conf["time_zone"]
        _FakeRecorderRetrieveHass.last_instance = None
        _FakeRecorderRetrieveHass.available_days = None

    def _build_forecast(self):
        """A Forecast wired for a live (non-file) retrieval path."""
        fcst = Forecast(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            copy.deepcopy(self.params),
            emhass_conf,
            logger,
            get_data_from_file=False,
        )
        # The ml path resolves the pickle filename from passed_data["model_type"]
        # (and pre-fix code read it even in debug mode); the default test params
        # do not carry one.
        fcst.params["passed_data"]["model_type"] = "test_1067"
        return fcst

    async def _fit_mlf(self, num_lags, train_days=14):
        """Fit a real KNN MLForecaster with ``num_lags`` on synthetic 30-min data."""
        samples = train_days * 48
        index = pd.date_range(
            start=pd.Timestamp("2024-06-01", tz=self.time_zone),
            periods=samples,
            freq=pd.Timedelta("30min"),
        )
        rng = np.random.default_rng(1067)
        values = (
            500.0 + 200.0 * np.sin(np.arange(samples) * 2 * np.pi / 48) + rng.normal(0, 20, samples)
        )
        data = pd.DataFrame({self.var_load: values}, index=index)
        mlf = MLForecaster(
            data,
            "test_1067",
            self.var_load,
            "KNeighborsRegressor",
            num_lags,
            emhass_conf,
            logger,
        )
        await mlf.fit()
        return mlf

    async def _run_ml_forecast(self, fcst, mlf, days_min):
        """Run the mlforecaster load forecast against the fake recorder.

        RetrieveHass is patched at the module-class level because
        _prepare_hass_load_data constructs its own instance internally --
        there is no narrower seam to inject the fake through. Returns the
        forecast result, or the raised ValueError (the base failure mode) so
        assertions fail on behaviour rather than error out.
        """
        with unittest.mock.patch.object(forecast_module, "RetrieveHass", _FakeRecorderRetrieveHass):
            try:
                return await fcst.get_load_forecast(
                    days_min_load_forecast=days_min,
                    method="mlforecaster",
                    use_last_window=True,
                    debug=True,
                    mlf=mlf,
                )
            except ValueError as e:
                return e

    async def test_history_retrieval_sized_from_model_lags(self):
        """A 144-lag model with delta_forecast_daily=1 must still forecast.

        On the unsized retrieval the recorder returns 1 day (48 samples), the
        model needs 144, and skforecast raises -- the #1067 crash. The fix must
        enlarge the fetch to ceil(144/48)+1 = 4 days and succeed.
        """
        fcst = self._build_forecast()
        mlf = await self._fit_mlf(num_lags=144)

        result = await self._run_ml_forecast(fcst, mlf, days_min=1)

        self.assertIsInstance(
            result,
            pd.Series,
            f"mlforecaster load forecast failed instead of succeeding: {result!r}",
        )
        self.assertEqual(len(result), len(fcst.forecast_dates))
        self.assertFalse(result.isna().any())
        # The retrieval was sized from the model, not from delta_forecast_daily.
        self.assertEqual(_FakeRecorderRetrieveHass.last_instance.requested_days, 4)

    async def test_short_recorder_aborts_cleanly_not_skforecast_error(self):
        """Recorder retaining less than the model needs -> False, not a raw ValueError."""
        _FakeRecorderRetrieveHass.available_days = 1  # recorder only has 1 day
        fcst = self._build_forecast()
        mlf = await self._fit_mlf(num_lags=144)

        result = await self._run_ml_forecast(fcst, mlf, days_min=1)

        self.assertIs(
            result,
            False,
            f"expected a clean False abort on short history, got: {result!r}",
        )

    async def test_small_model_keeps_days_min_load_forecast(self):
        """A model needing less than days_min_load_forecast must not shrink or grow the fetch."""
        fcst = self._build_forecast()
        # 52 lags (not 48) so the model can still fill the 50-step horizon of the
        # 25-hour day before a fall-back DST transition; the sizing under test is
        # unchanged: ceil(52/48)+1 = 3 days needed, below the days_min of 4.
        mlf = await self._fit_mlf(num_lags=52)

        result = await self._run_ml_forecast(fcst, mlf, days_min=4)

        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(fcst.forecast_dates))
        # 3 days needed; days_min=4 is larger and must win.
        self.assertEqual(_FakeRecorderRetrieveHass.last_instance.requested_days, 4)

    async def test_missing_model_pickle_fails_before_any_retrieval(self):
        """With no saved model the run must fail before the history fetch happens."""
        fcst = self._build_forecast()
        fcst.params["passed_data"]["model_type"] = "no_such_model_1067"

        with unittest.mock.patch.object(forecast_module, "RetrieveHass", _FakeRecorderRetrieveHass):
            result = await fcst.get_load_forecast(
                days_min_load_forecast=1,
                method="mlforecaster",
                use_last_window=True,
                debug=False,
            )

        self.assertIs(result, False)
        self.assertIsNone(
            _FakeRecorderRetrieveHass.last_instance,
            "history retrieval ran despite the model pickle being missing",
        )

    def test_mlf_window_size_prefers_fitted_forecaster(self):
        """forecaster.window_size wins over lags_opt/num_lags (non-contiguous lags)."""
        mlf = unittest.mock.MagicMock()
        mlf.forecaster.window_size = 144
        mlf.is_tuned = True
        mlf.lags_opt = 4  # len(lags) of a non-contiguous [1, 2, 3, 144]
        mlf.num_lags = 48
        self.assertEqual(Forecast._mlf_window_size(mlf), 144)

    def test_mlf_window_size_fallbacks(self):
        """Without a fitted forecaster: lags_opt when tuned, num_lags otherwise."""
        tuned = unittest.mock.MagicMock()
        tuned.forecaster = None
        tuned.is_tuned = True
        tuned.lags_opt = 96
        tuned.num_lags = 48
        self.assertEqual(Forecast._mlf_window_size(tuned), 96)

        untuned = unittest.mock.MagicMock()
        untuned.forecaster = None
        untuned.is_tuned = False
        untuned.num_lags = 48
        self.assertEqual(Forecast._mlf_window_size(untuned), 48)

    def test_mlf_required_history_days_math(self):
        """needed_days = ceil(window_size / samples_per_day) + 1, floored at days_min."""
        fcst = self._build_forecast()
        mlf = unittest.mock.MagicMock()
        mlf.forecaster.window_size = 144
        # 30-min steps: 48/day -> ceil(144/48)+1 = 4
        fcst.freq = pd.Timedelta("30min")
        self.assertEqual(fcst._mlf_required_history_days(mlf, 1), 4)
        # 1-h steps: 24/day -> ceil(144/24)+1 = 7
        fcst.freq = pd.Timedelta("1h")
        self.assertEqual(fcst._mlf_required_history_days(mlf, 1), 7)
        # Non-divisible: 50 lags at 30-min -> ceil(50/48)+1 = 3
        mlf.forecaster.window_size = 50
        fcst.freq = pd.Timedelta("30min")
        self.assertEqual(fcst._mlf_required_history_days(mlf, 1), 3)
        # days_min larger than the model's need wins.
        mlf.forecaster.window_size = 24
        self.assertEqual(fcst._mlf_required_history_days(mlf, 5), 5)


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
