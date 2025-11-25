#!/usr/bin/env python

import _pickle as cPickle
import bz2
import copy
import os
import pathlib
import pickle
import re
import unittest

import aiofiles
import orjson
import pandas as pd
from aioresponses import aioresponses

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


class TestForecast(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def get_test_params():
        params = {}
        # Build params with default config and secrets
        if emhass_conf["defaults_path"].exists():
            config = await utils.build_config(
                emhass_conf, logger, emhass_conf["defaults_path"]
            )
            _, secrets = await utils.build_secrets(
                emhass_conf, logger, no_response=True
            )
            params = await utils.build_params(emhass_conf, secrets, config, logger)
        else:
            raise Exception(
                "config_defaults.json does not exist in path: "
                + str(emhass_conf["defaults_path"])
            )
        return params

    async def asyncSetUp(self):
        self.get_data_from_file = True
        params = await TestForecast.get_test_params()
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(
            params_json, logger
        )
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
                self.rh.df_final, self.days_list, self.var_list, self.rh.ha_config = (
                    pickle.loads(content)
                )
                self.rh.var_list = self.var_list
            self.retrieve_hass_conf["sensor_power_load_no_var_loads"] = str(
                self.var_list[0]
            )
            self.retrieve_hass_conf["sensor_power_photovoltaics"] = str(
                self.var_list[1]
            )
            self.retrieve_hass_conf["sensor_power_photovoltaics_forecast"] = str(
                self.var_list[2]
            )
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
        self.P_PV_forecast = self.fcst.get_power_from_weather(self.df_weather_scrap)
        self.P_load_forecast = await self.fcst.get_load_forecast(
            method=optim_conf["load_forecast_method"]
        )
        self.df_input_data_dayahead = pd.concat(
            [self.P_PV_forecast, self.P_load_forecast], axis=1
        )
        self.df_input_data_dayahead.columns = ["P_PV_forecast", "P_load_forecast"]
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
            "P_PV_forecast": self.P_PV_forecast,
            "P_load_forecast": self.P_load_forecast,
            "params": params_json,
        }

    # Test weather forecast dataframe output based on saved csv file
    async def test_get_weather_forecast_csv(self):
        # Test dataframe from get weather forecast
        self.df_weather_csv = await self.fcst.get_weather_forecast(method="csv")
        self.assertEqual(self.fcst.weather_forecast_method, "csv")
        self.assertIsInstance(self.df_weather_csv, type(pd.DataFrame()))
        self.assertIsInstance(
            self.df_weather_csv.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        self.assertIsInstance(
            self.df_weather_csv.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(self.df_weather_csv.index.tz, self.fcst.time_zone)
        self.assertTrue(
            self.fcst.start_forecast < ts for ts in self.df_weather_csv.index
        )
        self.assertEqual(
            len(self.df_weather_csv),
            int(
                self.optim_conf["delta_forecast_daily"].total_seconds()
                / 3600
                / self.fcst.timeStep
            ),
        )
        # Test dataframe from get power from weather
        P_PV_forecast = self.fcst.get_power_from_weather(self.df_weather_csv)
        self.assertIsInstance(P_PV_forecast, pd.core.series.Series)
        self.assertIsInstance(
            P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        self.assertIsInstance(
            P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(P_PV_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.df_weather_csv), len(P_PV_forecast))
        df_weather_none = await self.fcst.get_weather_forecast(method="none")
        self.assertTrue(df_weather_none is None)

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
        self.assertIsInstance(self.fcst.X_adjust_pv, pd.DataFrame)
        self.assertIsInstance(self.fcst.y_adjust_pv, pd.core.series.Series)
        # Call the fit method
        await self.fcst.adjust_pv_forecast_fit(
            n_splits=5, regression_model="LassoRegression", debug=False
        )
        # Call the predict method
        P_PV_forecast = self.fcst.adjust_pv_forecast_predict()
        self.assertEqual(len(P_PV_forecast), len(self.fcst.P_PV_forecast_validation))
        self.assertFalse(
            P_PV_forecast.isna().any().any(), "Adjusted forecast contains NaN values"
        )
        self.assertGreaterEqual(
            self.fcst.validation_rmse, 0.0, "RMSE should be non-negative"
        )
        self.assertLessEqual(
            self.fcst.validation_r2, 1.0, "R² score should be at most 1"
        )
        self.assertGreaterEqual(
            self.fcst.validation_r2, -1.0, "R² score should be at least -1"
        )

        # import plotly.express as px
        # data_to_plot = self.fcst.P_PV_forecast_validation[["forecast", "adjusted_forecast"]].reset_index()
        # fig = px.line(
        #     data_to_plot,
        #     x="index",  # Assuming the index is the timestamp
        #     y=["forecast", "adjusted_forecast"],
        #     labels={"index": "Time", "value": "Power (W)", "variable": "Forecast Type"},
        #     title="Forecast vs Adjusted Forecast",
        #     template='presentation'
        # )
        # fig.show()

    # Test output weather forecast using openmeteo with mock get request data
    async def test_get_weather_forecast_openmeteo_method_mock(self):
        test_data_path = (
            emhass_conf["data_path"] / "test_response_openmeteo_get_method.pbz2"
        )

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
        get_url = "https://api.open-meteo.com/v1/forecast"

        with aioresponses() as mocked:
            mocked.get(get_url, payload=data)

            # Test dataframe output from get weather forecast
            df_weather_openmeteo = await self.fcst.get_weather_forecast(
                method="open-meteo"
            )
            self.assertIsInstance(df_weather_openmeteo, type(pd.DataFrame()))
            self.assertIsInstance(
                df_weather_openmeteo.index, pd.core.indexes.datetimes.DatetimeIndex
            )
            self.assertIsInstance(
                df_weather_openmeteo.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
            )
            self.assertEqual(df_weather_openmeteo.index.tz, self.fcst.time_zone)
            self.assertTrue(
                self.fcst.start_forecast < ts for ts in df_weather_openmeteo.index
            )
            self.assertEqual(
                len(df_weather_openmeteo),
                int(
                    self.optim_conf["delta_forecast_daily"].total_seconds()
                    / 3600
                    / self.fcst.timeStep
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
            self.assertTrue("ghi" in list(df_weather_openmeteo.columns))
            self.assertTrue("dhi" in list(df_weather_openmeteo.columns))
            self.assertTrue("dni" in list(df_weather_openmeteo.columns))
            # Test dataframe output from get power from weather forecast
            P_PV_forecast = self.fcst.get_power_from_weather(df_weather_openmeteo)
            self.assertIsInstance(P_PV_forecast, pd.core.series.Series)
            self.assertIsInstance(
                P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex
            )
            self.assertIsInstance(
                P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
            )
            self.assertEqual(P_PV_forecast.index.tz, self.fcst.time_zone)
            self.assertEqual(len(df_weather_openmeteo), len(P_PV_forecast))
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
            P_PV_forecast = self.fcst.get_power_from_weather(df_weather_openmeteo)
            self.assertIsInstance(P_PV_forecast, pd.core.series.Series)
            self.assertIsInstance(
                P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex
            )
            self.assertIsInstance(
                P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
            )
            self.assertEqual(P_PV_forecast.index.tz, self.fcst.time_zone)
            self.assertEqual(len(df_weather_openmeteo), len(P_PV_forecast))

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

        test_data_path = str(
            emhass_conf["data_path"] / "test_response_solcast_get_method.pbz2"
        )

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
            self.assertIsInstance(
                df_weather_scrap.index, pd.core.indexes.datetimes.DatetimeIndex
            )
            self.assertIsInstance(
                df_weather_scrap.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
            )
            self.assertEqual(df_weather_scrap.index.tz, self.fcst.time_zone)
            self.assertTrue(
                self.fcst.start_forecast < ts for ts in df_weather_scrap.index
            )
            self.assertEqual(
                len(df_weather_scrap),
                int(
                    self.optim_conf["delta_forecast_daily"].total_seconds()
                    / 3600
                    / self.fcst.timeStep
                ),
            )
            if os.path.isfile(
                emhass_conf["data_path"] / "temp_weather_forecast_data.pkl"
            ):
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
        roof_ids = re.split(
            r"[,\s]+", self.fcst.retrieve_hass_conf["solcast_rooftop_id"].strip()
        )
        if os.path.isfile(emhass_conf["data_path"] / "weather_forecast_data.pkl"):
            os.rename(
                emhass_conf["data_path"] / "weather_forecast_data.pkl",
                emhass_conf["data_path"] / "temp_weather_forecast_data.pkl",
            )
        test_data_path = str(
            emhass_conf["data_path"] / "test_response_solcast_get_method.pbz2"
        )
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
            self.assertIsInstance(
                df_weather_scrap.index, pd.core.indexes.datetimes.DatetimeIndex
            )
            self.assertIsInstance(
                df_weather_scrap.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
            )
            self.assertEqual(df_weather_scrap.index.tz, self.fcst.time_zone)
            self.assertTrue(
                self.fcst.start_forecast < ts for ts in df_weather_scrap.index
            )
            self.assertEqual(
                len(df_weather_scrap),
                int(
                    self.optim_conf["delta_forecast_daily"].total_seconds()
                    / 3600
                    / self.fcst.timeStep
                ),
            )
            if os.path.isfile(
                emhass_conf["data_path"] / "temp_weather_forecast_data.pkl"
            ):
                os.rename(
                    emhass_conf["data_path"] / "temp_weather_forecast_data.pkl",
                    emhass_conf["data_path"] / "weather_forecast_data.pkl",
                )

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
                    self.fcst.start_forecast < ts
                    for ts in df_weather_solarforecast.index
                )
                self.assertEqual(
                    len(df_weather_solarforecast),
                    int(
                        self.optim_conf["delta_forecast_daily"].total_seconds()
                        / 3600
                        / self.fcst.timeStep
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
                updated_emhass_conf, built_secrets = await utils.build_secrets(
                    emhass_conf, logger
                )
                emhass_conf.update(updated_emhass_conf)
                params.update(
                    await utils.build_params(
                        emhass_conf, built_secrets, defaults, logger
                    )
                )
        else:
            raise Exception(
                "config_defaults.json does not exist in path: "
                + str(emhass_conf["defaults_path"])
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
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(
            params_json, logger
        )
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
            days_list = utils.get_days_list(
                retrieve_hass_conf["historic_days_to_retrieve"]
            )
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
        P_PV_forecast = await fcst.get_weather_forecast(method="list")
        df_input_data.index = P_PV_forecast.index
        df_input_data.index.freq = rh.df_final.index.freq
        self.assertIsInstance(P_PV_forecast, type(pd.DataFrame()))
        self.assertIsInstance(
            P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        self.assertIsInstance(
            P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(P_PV_forecast.index.tz, fcst.time_zone)
        self.assertTrue(fcst.start_forecast < ts for ts in P_PV_forecast.index)
        self.assertTrue(P_PV_forecast.values[0][0] == 1)
        self.assertTrue(P_PV_forecast.values[-1][0] == 48)
        # Get load forecast with list, check dataframe output
        P_load_forecast = await fcst.get_load_forecast(method="list")
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(
            P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        self.assertIsInstance(
            P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(P_load_forecast.index.tz, fcst.time_zone)
        self.assertEqual(len(P_PV_forecast), len(P_load_forecast))
        self.assertTrue(P_load_forecast.values[0] == 1)
        self.assertTrue(P_load_forecast.values[-1] == 48)
        # Get load cost forecast with list, check dataframe output
        df_input_data = fcst.get_load_cost_forecast(df_input_data, method="list")
        self.assertTrue(fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum() == 0)
        self.assertTrue(df_input_data["unit_load_cost"].values[0] == 1)
        self.assertTrue(df_input_data["unit_load_cost"].values[-1] == 48)
        # Get production price forecast with list, check dataframe output
        df_input_data = fcst.get_prod_price_forecast(df_input_data, method="list")
        self.assertTrue(fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum() == 0)
        self.assertTrue(df_input_data["unit_prod_price"].values[0] == 1)
        self.assertTrue(df_input_data["unit_prod_price"].values[-1] == 48)

    # Test output weather forecast using longer passed runtime lists
    async def test_get_forecasts_with_longer_lists(self):
        # Load default params
        params = {}
        set_type = "dayahead-optim"
        if emhass_conf["defaults_path"].exists():
            async with aiofiles.open(emhass_conf["defaults_path"]) as data:
                content = await data.read()
                defaults = orjson.loads(content)
                updated_emhass_conf, built_secrets = await utils.build_secrets(
                    emhass_conf, logger
                )
                emhass_conf.update(updated_emhass_conf)
                params.update(
                    await utils.build_params(
                        emhass_conf, built_secrets, defaults, logger
                    )
                )
        else:
            raise Exception(
                "config_defaults.json does not exist in path: "
                + str(emhass_conf["defaults_path"])
            )

        # Create 3*48 (3 days of data) long lists runtime forecasts parameters
        list_length = 3 * 48  # 3 days
        runtimeparams = {
            "pv_power_forecast": [i + 1 for i in range(list_length)],
            "load_power_forecast": [i + 1 for i in range(list_length)],
            "load_cost_forecast": [i + 1 for i in range(list_length)],
            "prod_price_forecast": [i + 1 for i in range(list_length)],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(
            params_json, logger
        )
        optim_conf["delta_forecast_daily"] = pd.Timedelta(days=3)
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
        # Create Forecast Object
        fcst = Forecast(
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            params_json,
            emhass_conf,
            logger,
            get_data_from_file=True,
        )
        # Get weather forecast with list, check dataframe output
        P_PV_forecast = await fcst.get_weather_forecast(method="list")
        self.assertIsInstance(P_PV_forecast, type(pd.DataFrame()))
        self.assertIsInstance(
            P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        self.assertIsInstance(
            P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(P_PV_forecast.index.tz, fcst.time_zone)
        self.assertTrue(fcst.start_forecast < ts for ts in P_PV_forecast.index)
        self.assertTrue(P_PV_forecast.values[0][0] == 1)
        self.assertTrue(P_PV_forecast.values[-1][0] == 3 * 48)
        # Get load forecast with list, check dataframe output
        P_load_forecast = await fcst.get_load_forecast(method="list")
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(
            P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        self.assertIsInstance(
            P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(P_load_forecast.index.tz, fcst.time_zone)
        self.assertEqual(len(P_PV_forecast), len(P_load_forecast))
        self.assertTrue(P_load_forecast.values[0] == 1)
        self.assertTrue(P_load_forecast.values[-1] == 3 * 48)
        df_input_data_dayahead = pd.concat([P_PV_forecast, P_load_forecast], axis=1)
        df_input_data_dayahead = utils.set_df_index_freq(df_input_data_dayahead)
        df_input_data_dayahead.columns = ["P_PV_forecast", "P_load_forecast"]
        # Get load cost forecast with list, check dataframe output
        df_input_data_dayahead = fcst.get_load_cost_forecast(
            df_input_data_dayahead, method="list"
        )
        self.assertTrue(fcst.var_load_cost in df_input_data_dayahead.columns)
        self.assertTrue(df_input_data_dayahead.isnull().sum().sum() == 0)
        self.assertTrue(df_input_data_dayahead[fcst.var_load_cost].iloc[0] == 1)
        self.assertTrue(df_input_data_dayahead[fcst.var_load_cost].iloc[-1] == 3 * 48)
        # Get production price forecast with list, check dataframe output
        df_input_data_dayahead = fcst.get_prod_price_forecast(
            df_input_data_dayahead, method="list"
        )
        self.assertTrue(fcst.var_prod_price in df_input_data_dayahead.columns)
        self.assertTrue(df_input_data_dayahead.isnull().sum().sum() == 0)
        self.assertTrue(df_input_data_dayahead[fcst.var_prod_price].iloc[0] == 1)
        self.assertTrue(df_input_data_dayahead[fcst.var_prod_price].iloc[-1] == 3 * 48)

    # Test output values of weather forecast using passed runtime lists and saved sensor datalf):
    async def test_get_forecasts_with_lists_special_case(self):
        # Load default params
        params = {}
        if emhass_conf["defaults_path"].exists():
            config = await utils.build_config(
                emhass_conf, logger, emhass_conf["defaults_path"]
            )
            _, secrets = await utils.build_secrets(
                emhass_conf, logger, no_response=True
            )
            params = await utils.build_params(emhass_conf, secrets, config, logger)
        else:
            raise Exception(
                "config_defaults.json does not exist in path: "
                + str(emhass_conf["defaults_path"])
            )
        # Create 48 (1 day of data) long lists runtime forecasts parameters
        runtimeparams = {
            "load_cost_forecast": [i + 1 for i in range(48)],
            "prod_price_forecast": [i + 1 for i in range(48)],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(
            params_json, logger
        )
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
            days_list = utils.get_days_list(
                retrieve_hass_conf["historic_days_to_retrieve"]
            )
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
        P_PV_forecast = await fcst.get_weather_forecast()
        df_input_data.index = P_PV_forecast.index
        df_input_data.index.freq = rh.df_final.index.freq
        # Get load cost forecast with list, check values from output
        df_input_data = fcst.get_load_cost_forecast(df_input_data, method="list")
        self.assertTrue(fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum() == 0)
        self.assertTrue(df_input_data["unit_load_cost"].values[0] == 1)
        self.assertTrue(df_input_data["unit_load_cost"].values[-1] == 48)
        # Get production price forecast with list, check values from output
        df_input_data = fcst.get_prod_price_forecast(df_input_data, method="list")
        self.assertTrue(fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum() == 0)
        self.assertTrue(df_input_data["unit_prod_price"].values[0] == 1)
        self.assertTrue(df_input_data["unit_prod_price"].values[-1] == 48)

    async def test_get_power_from_weather(self):
        self.assertIsInstance(self.P_PV_forecast, pd.core.series.Series)
        self.assertIsInstance(
            self.P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        self.assertIsInstance(
            self.P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(self.P_PV_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.df_weather_scrap), len(self.P_PV_forecast))
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
        params = orjson.dumps(
            {"passed_data": {"weather_forecast_cache": False}}
        ).decode("utf-8")
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
        P_PV_forecast = self.fcst.get_power_from_weather(df_weather_scrap)
        self.assertIsInstance(P_PV_forecast, pd.core.series.Series)
        self.assertIsInstance(
            P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        self.assertIsInstance(
            P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(P_PV_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.df_weather_scrap), len(P_PV_forecast))
        # Test the mixed forecast
        params = orjson.dumps({"passed_data": {"alpha": 0.5, "beta": 0.5}}).decode(
            "utf-8"
        )
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
        P_PV_forecast = self.fcst.get_power_from_weather(
            df_weather_scrap, set_mix_forecast=True, df_now=df_input_data
        )
        self.assertIsInstance(P_PV_forecast, pd.core.series.Series)
        self.assertIsInstance(
            P_PV_forecast.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        self.assertIsInstance(
            P_PV_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(P_PV_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.df_weather_scrap), len(P_PV_forecast))

    # Test dataframe output of load forecast
    async def test_get_load_forecast(self):
        P_load_forecast = await self.fcst.get_load_forecast()
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(
            P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        self.assertIsInstance(
            P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(P_load_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.P_PV_forecast), len(P_load_forecast))
        print(">> The length of the load forecast = " + str(len(P_load_forecast)))
        # Test the mixed forecast
        params_json = orjson.dumps({"passed_data": {"alpha": 0.5, "beta": 0.5}}).decode(
            "utf-8"
        )
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
        P_load_forecast = await self.fcst.get_load_forecast(
            set_mix_forecast=True, df_now=df_input_data
        )
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(
            P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        self.assertIsInstance(
            P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(P_load_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.P_PV_forecast), len(P_load_forecast))
        # Test load forecast from csv
        P_load_forecast = await self.fcst.get_load_forecast(method="csv")
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(
            P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        self.assertIsInstance(
            P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(P_load_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.P_PV_forecast), len(P_load_forecast))

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
        P_load_forecast = await input_data_dict["fcst"].get_load_forecast(
            method="mlforecaster", use_last_window=False, debug=True, mlf=mlf
        )
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(
            P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        self.assertIsInstance(
            P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(P_load_forecast.index.tz, self.fcst.time_zone)
        self.assertTrue((P_load_forecast.index == self.fcst.forecast_dates).all())
        self.assertEqual(len(self.P_PV_forecast), len(P_load_forecast))

    # Test load forecast with typical statistics method
    async def test_get_load_forecast_typical(self):
        P_load_forecast = await self.fcst.get_load_forecast(method="typical")
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertIsInstance(
            P_load_forecast.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        self.assertIsInstance(
            P_load_forecast.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(P_load_forecast.index.tz, self.fcst.time_zone)
        self.assertEqual(len(self.P_PV_forecast), len(P_load_forecast))
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
        self.assertTrue(len(fcst.forecast_dates) == 24)
        P_load_forecast = await fcst.get_load_forecast(method="typical")
        self.assertIsInstance(P_load_forecast, pd.core.series.Series)
        self.assertTrue(len(P_load_forecast) == len(fcst.forecast_dates))

    # Test load cost forecast dataframe output using saved csv referece file
    def test_get_load_cost_forecast(self):
        df_input_data = self.fcst.get_load_cost_forecast(self.df_input_data)
        self.assertTrue(self.fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum() == 0)
        df_input_data = self.fcst.get_load_cost_forecast(
            self.df_input_data, method="csv", csv_path="data_load_cost_forecast.csv"
        )
        self.assertTrue(self.fcst.var_load_cost in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum() == 0)

    # Test production price forecast dataframe output using saved csv referece file
    def test_get_prod_price_forecast(self):
        df_input_data = self.fcst.get_prod_price_forecast(self.df_input_data)
        self.assertTrue(self.fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum() == 0)
        df_input_data = self.fcst.get_prod_price_forecast(
            self.df_input_data, method="csv", csv_path="data_prod_price_forecast.csv"
        )
        self.assertTrue(self.fcst.var_prod_price in df_input_data.columns)
        self.assertTrue(df_input_data.isnull().sum().sum() == 0)

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
            P_load_forecast_dst = await dst_fcst.get_load_forecast(method="naive")
            self.assertIsInstance(P_load_forecast_dst, pd.core.series.Series)
            self.assertEqual(len(P_load_forecast_dst), len(dst_fcst.forecast_dates))
            # Check that index is properly timezone-aware
            self.assertEqual(P_load_forecast_dst.index.tz, sydney_tz)
            logger.info("DST forward transition test for naive method: PASSED")
        except Exception as e:
            self.fail(f"Naive forecast failed during DST forward transition: {e}")

        # Test typical load forecast during DST transition
        try:
            P_load_forecast_typical = await dst_fcst.get_load_forecast(method="typical")
            self.assertIsInstance(P_load_forecast_typical, pd.core.series.Series)
            self.assertEqual(len(P_load_forecast_typical), len(dst_fcst.forecast_dates))
            self.assertEqual(P_load_forecast_typical.index.tz, sydney_tz)
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
            self.assertTrue(len(localized_times) == len(naive_times))
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
        us_dst_start = eastern_tz.localize(
            datetime(2025, 3, 9, 1, 0, 0)
        )  # March 9, 1 AM
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
            us_P_load_forecast = await us_dst_fcst.get_load_forecast(method="naive")
            self.assertIsInstance(us_P_load_forecast, pd.core.series.Series)
            self.assertEqual(len(us_P_load_forecast), len(us_dst_fcst.forecast_dates))
            self.assertEqual(us_P_load_forecast.index.tz, eastern_tz)
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
        dst_end = dst_start + pd.Timedelta(
            hours=5
        )  # 5 hours later, crosses DST backward

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
            P_load_forecast_dst = await dst_fcst.get_load_forecast(method="naive")
            self.assertIsInstance(P_load_forecast_dst, pd.core.series.Series)
            self.assertEqual(len(P_load_forecast_dst), len(dst_fcst.forecast_dates))
            # Check that index is properly timezone-aware
            self.assertEqual(P_load_forecast_dst.index.tz, sydney_tz)
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
                self.assertIsNotNone(
                    ts.tzinfo, "Valid timestamps should be timezone-aware"
                )

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
                self.assertTrue(len(localized_times) == len(naive_times))
                # Check that we got reasonable results for ambiguous times
                for ts in localized_times:
                    self.assertIsNotNone(
                        ts.tzinfo, "All timestamps should be timezone-aware"
                    )

                logger.info(
                    "Direct tz_localize DST backward transition test (alternative): PASSED"
                )
            except Exception as e2:
                self.fail(
                    f"Direct tz_localize failed during DST backward transition: {e} and {e2}"
                )

        # Test case 3: US Eastern Time DST backward transition (November)
        # DST ends on November 2, 2025 at 2:00 AM -> 1:00 AM
        eastern_tz = pytz.timezone("US/Eastern")
        us_dst_start = eastern_tz.localize(
            datetime(2025, 11, 2, 0, 30, 0)
        )  # Nov 2, 12:30 AM
        us_dst_end = us_dst_start + pd.Timedelta(
            hours=4
        )  # 4 hours later, crosses DST backward

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
            us_P_load_forecast = await us_dst_fcst.get_load_forecast(method="naive")
            self.assertIsInstance(us_P_load_forecast, pd.core.series.Series)
            self.assertEqual(len(us_P_load_forecast), len(us_dst_fcst.forecast_dates))
            self.assertEqual(us_P_load_forecast.index.tz, eastern_tz)
            logger.info("US Eastern DST backward transition test: PASSED")
        except Exception as e:
            self.fail(f"US Eastern DST forecast failed during backward transition: {e}")


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
