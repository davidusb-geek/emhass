import _pickle as cPickle
import bz2
import copy
import datetime
import pathlib
import pickle
import unittest
from unittest.mock import MagicMock, patch

import aiofiles
import numpy as np
import orjson
import pandas as pd
from aioresponses import aioresponses

from emhass import utils_async as utils
from emhass.retrieve_hass_async import RetrieveHass
from emhass.utils_async import get_days_list, get_logger, get_yaml_parse

# The root folder
root = pathlib.Path(utils.get_root(__file__, num_parent=2))
# Build emhass_conf paths
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["options_path"] = root / "options.json"
emhass_conf["secrets_path"] = root / "secrets_emhass(example).yaml"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

# create logger
logger, ch = get_logger(__name__, emhass_conf, save_to_file=False)


class TestRetrieveHass(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.get_data_from_file = True
        save_data_to_file = False
        model_type = "test_df_final"  # Options: "test_df_final" or "long_train_data"

        # Build params with default secrets (no config)
        if emhass_conf["defaults_path"].exists():
            if self.get_data_from_file:
                _, secrets = await utils.build_secrets(
                    emhass_conf, logger, no_response=True
                )
                params = await utils.build_params(emhass_conf, secrets, {}, logger)
                retrieve_hass_conf, _, _ = get_yaml_parse(params, logger)
            else:
                emhass_conf["secrets_path"] = root / "secrets_emhass.yaml"
                config = await utils.build_config(
                    emhass_conf, logger, emhass_conf["defaults_path"]
                )
                _, secrets = await utils.build_secrets(
                    emhass_conf,
                    logger,
                    secrets_path=emhass_conf["secrets_path"],
                    no_response=True,
                )
                params = await utils.build_params(emhass_conf, secrets, config, logger)
                retrieve_hass_conf, _, _ = get_yaml_parse(params, logger)
                params = None
        else:
            raise Exception(
                "config_defaults. does not exist in path: "
                + str(emhass_conf["defaults_path"])
            )

        # Force config params for testing
        retrieve_hass_conf["optimization_time_step"] = pd.to_timedelta(30, "minutes")
        retrieve_hass_conf["sensor_power_photovoltaics"] = "sensor.power_photovoltaics"
        retrieve_hass_conf["sensor_power_photovoltaics_forecast"] = (
            "sensor.p_pv_forecast"
        )
        retrieve_hass_conf["sensor_power_load_no_var_loads"] = (
            "sensor.power_load_no_var_loads"
        )
        retrieve_hass_conf["sensor_replace_zero"] = [
            "sensor.power_photovoltaics",
            "sensor.p_pv_forecast",
        ]
        retrieve_hass_conf["sensor_linear_interp"] = [
            "sensor.power_photovoltaics",
            "sensor.p_pv_forecast",
            "sensor.power_load_no_var_loads",
        ]
        retrieve_hass_conf["set_zero_min"] = True
        retrieve_hass_conf["load_negative"] = True

        self.retrieve_hass_conf = retrieve_hass_conf
        self.rh = RetrieveHass(
            self.retrieve_hass_conf["hass_url"],
            self.retrieve_hass_conf["long_lived_token"],
            self.retrieve_hass_conf["optimization_time_step"],
            self.retrieve_hass_conf["time_zone"],
            params,
            emhass_conf,
            logger,
            get_data_from_file=self.get_data_from_file,
        )
        # Obtain sensor values from saved file
        if self.get_data_from_file:
            async with aiofiles.open(
                emhass_conf["data_path"] / str(model_type + ".pkl"), "rb"
            ) as f:
                content = await f.read()
                self.rh.df_final, self.days_list, self.var_list, self.rh.ha_config = (
                    pickle.loads(content)
                )
                self.rh.var_list = self.var_list
        # Else obtain sensor values from HA
        else:
            if model_type == "long_train_data":
                days_to_retrieve = 365
            else:
                days_to_retrieve = self.retrieve_hass_conf["historic_days_to_retrieve"]
            self.days_list = get_days_list(days_to_retrieve)
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
            # Mocking retrieve of ha_config using: self.rh.get_ha_config()
            self.rh.ha_config = {
                "country": "FR",
                "currency": "EUR",
                "elevation": 4807,
                "latitude": 48.83,
                "longitude": 6.86,
                "time_zone": "Europe/Paris",
                "unit_system": {
                    "length": "km",
                    "accumulated_precipitation": "mm",
                    "area": "m²",
                    "mass": "g",
                    "pressure": "Pa",
                    "temperature": "°C",
                    "volume": "L",
                    "wind_speed": "m/s",
                },
            }
            # Check to save updated data to file
            if save_data_to_file:
                with open(
                    emhass_conf["data_path"] / str(model_type + ".pkl"), "wb"
                ) as outp:
                    pickle.dump(
                        (
                            self.rh.df_final,
                            self.days_list,
                            self.var_list,
                            self.rh.ha_config,
                        ),
                        outp,
                        pickle.HIGHEST_PROTOCOL,
                    )
        self.df_raw = self.rh.df_final.copy()

    # Check yaml parse in setUp worked
    def test_get_yaml_parse(self):
        self.assertIsInstance(self.retrieve_hass_conf, dict)
        self.assertTrue("hass_url" in self.retrieve_hass_conf.keys())
        if self.get_data_from_file:
            self.assertTrue(
                self.retrieve_hass_conf["hass_url"] == "https://myhass.duckdns.org/"
            )

    # Check yaml parse worked
    async def test_yaml_parse_web_server(self):
        params = {}
        if emhass_conf["defaults_path"].exists():
            async with aiofiles.open(emhass_conf["defaults_path"]) as file:
                data = await file.read()
                defaults = orjson.loads(data)
                params.update(
                    await utils.build_params(emhass_conf, {}, defaults, logger)
                )
        _, optim_conf, _ = get_yaml_parse(params, logger)
        # Just check forecast methods
        self.assertFalse(optim_conf.get("weather_forecast_method") is None)
        self.assertFalse(optim_conf.get("load_forecast_method") is None)
        self.assertFalse(optim_conf.get("load_cost_forecast_method") is None)
        self.assertFalse(optim_conf.get("production_price_forecast_method") is None)

    # Assume get_data to HA fails
    async def test_get_data_failed(self):
        days_list = get_days_list(1)
        var_list = [self.retrieve_hass_conf["sensor_power_load_no_var_loads"]]
        response = await self.rh.get_data(days_list, var_list)
        if self.get_data_from_file:
            self.assertFalse(response)
        else:
            self.assertTrue(response)

    # Test with html mock response
    async def test_get_data_mock(self):
        with aioresponses() as mocked:
            test_data_path = (
                emhass_conf["data_path"] / "test_response_get_data_get_method.pbz2"
            )

            async with aiofiles.open(test_data_path, "rb") as f:
                compressed = await f.read()

            data = bz2.decompress(compressed)
            data = cPickle.loads(data)
            data = orjson.loads(data.content)
            days_list = get_days_list(1)
            var_list = [self.retrieve_hass_conf["sensor_power_load_no_var_loads"]]
            get_url = self.retrieve_hass_conf["hass_url"]
            # with aioresponses() as mocked:
            get_url = self.retrieve_hass_conf["hass_url"]
            mocked.get(get_url, payload=data, repeat=True)
            await self.rh.get_data(
                days_list,
                var_list,
                minimal_response=False,
                significant_changes_only=False,
                test_url=self.retrieve_hass_conf["hass_url"],
            )
            self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
            self.assertIsInstance(
                self.rh.df_final.index, pd.core.indexes.datetimes.DatetimeIndex
            )
            self.assertIsInstance(
                self.rh.df_final.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
            )
            self.assertEqual(len(self.rh.df_final.columns), len(var_list))
            self.assertEqual(
                self.rh.df_final.index.freq,
                self.retrieve_hass_conf["optimization_time_step"],
            )
            self.assertEqual(self.rh.df_final.index.tz, datetime.UTC)

    # Check the dataframe was formatted correctly
    async def test_prepare_data(self):
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertIsInstance(
            self.rh.df_final.index, pd.core.indexes.datetimes.DatetimeIndex
        )
        self.assertIsInstance(
            self.rh.df_final.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
        )
        self.assertEqual(len(self.rh.df_final.columns), len(self.var_list))
        self.assertEqual(
            self.rh.df_final.index.isin(self.days_list).sum(), len(self.days_list)
        )
        self.assertEqual(
            self.rh.df_final.index.freq,
            self.retrieve_hass_conf["optimization_time_step"],
        )
        self.assertEqual(self.rh.df_final.index.tz, datetime.UTC)
        self.rh.prepare_data(
            self.retrieve_hass_conf["sensor_power_load_no_var_loads"],
            load_negative=self.retrieve_hass_conf["load_negative"],
            set_zero_min=self.retrieve_hass_conf["set_zero_min"],
            var_replace_zero=self.retrieve_hass_conf["sensor_replace_zero"],
            var_interp=self.retrieve_hass_conf["sensor_linear_interp"],
        )
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertEqual(
            self.rh.df_final.index.isin(self.days_list).sum(),
            self.df_raw.index.isin(self.days_list).sum(),
        )
        self.assertEqual(len(self.rh.df_final.columns), len(self.df_raw.columns))
        self.assertEqual(
            self.rh.df_final.index.freq,
            self.retrieve_hass_conf["optimization_time_step"],
        )
        self.assertEqual(
            self.rh.df_final.index.tz, self.retrieve_hass_conf["time_zone"]
        )

    # Test negative load
    async def test_prepare_data_negative_load(self):
        self.rh.df_final[
            self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        ] = -self.rh.df_final[self.retrieve_hass_conf["sensor_power_load_no_var_loads"]]
        self.rh.prepare_data(
            self.retrieve_hass_conf["sensor_power_load_no_var_loads"],
            load_negative=True,
            set_zero_min=self.retrieve_hass_conf["set_zero_min"],
            var_replace_zero=self.retrieve_hass_conf["sensor_replace_zero"],
            var_interp=None,
        )
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertEqual(
            self.rh.df_final.index.isin(self.days_list).sum(),
            self.df_raw.index.isin(self.days_list).sum(),
        )
        self.assertEqual(len(self.rh.df_final.columns), len(self.df_raw.columns))
        self.assertEqual(
            self.rh.df_final.index.freq,
            self.retrieve_hass_conf["optimization_time_step"],
        )
        self.assertEqual(
            self.rh.df_final.index.tz, self.retrieve_hass_conf["time_zone"]
        )

    # Tests that the prepare_data method does convert missing PV values to zero
    # and also ignores any missing sensor columns.
    async def test_prepare_data_missing_pv(self):
        load_sensor = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        actual_pv_sensor = self.retrieve_hass_conf["sensor_power_photovoltaics"]
        forecast_pv_sensor = self.retrieve_hass_conf[
            "sensor_power_photovoltaics_forecast"
        ]
        var_replace_zero = [actual_pv_sensor, forecast_pv_sensor, "sensor.missing1"]
        var_interp = [actual_pv_sensor, load_sensor, "sensor.missing2"]
        # Replace actual and forecast PV zero values with NaN's (to test they get replaced back)
        self.rh.df_final[actual_pv_sensor] = self.rh.df_final[actual_pv_sensor].replace(
            0, np.nan
        )
        self.rh.df_final[forecast_pv_sensor] = self.rh.df_final[
            forecast_pv_sensor
        ].replace(0, np.nan)
        # Verify a non-zero number of missing values in the actual and forecast PV columns before prepare_data
        self.assertTrue(self.rh.df_final[actual_pv_sensor].isna().sum() > 0)
        self.assertTrue(self.rh.df_final[forecast_pv_sensor].isna().sum() > 0)
        self.rh.prepare_data(
            load_sensor,
            load_negative=False,
            set_zero_min=True,
            var_replace_zero=var_replace_zero,
            var_interp=var_interp,
        )
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertEqual(
            self.rh.df_final.index.isin(self.days_list).sum(),
            self.df_raw.index.isin(self.days_list).sum(),
        )
        # Check the before and after actual and forecast PV columns have the same number of values
        self.assertEqual(
            len(self.df_raw[actual_pv_sensor]), len(self.rh.df_final[actual_pv_sensor])
        )
        self.assertEqual(
            len(self.df_raw[forecast_pv_sensor]),
            len(self.rh.df_final[forecast_pv_sensor]),
        )
        # Verify no missing values in the actual and forecast PV columns after prepare_data
        self.assertTrue(self.rh.df_final[actual_pv_sensor].isna().sum() == 0)
        self.assertTrue(self.rh.df_final[forecast_pv_sensor].isna().sum() == 0)

    # Test publish data
    async def test_publish_data(self):
        response, data = await self.rh.post_data(
            self.df_raw[self.df_raw.columns[0]],
            10,
            "sensor.p_pv_forecast",
            "power",
            "Unit",
            "Variable",
            type_var="power",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            data["state"]
            == "{:.2f}".format(
                np.round(
                    self.df_raw.loc[self.df_raw.index[10], self.df_raw.columns[0]], 2
                )
            )
        )
        self.assertTrue(data["attributes"]["unit_of_measurement"] == "Unit")
        self.assertTrue(data["attributes"]["friendly_name"] == "Variable")
        # Lets test publishing a forecast with more added attributes
        df = copy.deepcopy(self.df_raw.iloc[0:30])
        df.columns = ["P_Load", "P_PV", "P_PV_forecast"]
        df["P_batt"] = 1000.0
        df["SOC_opt"] = 0.5
        response, data = await self.rh.post_data(
            df["P_PV_forecast"],
            10,
            "sensor.p_pv_forecast",
            "power",
            "W",
            "PV Forecast",
            type_var="power",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            data["state"] == f"{np.round(df.loc[df.index[10], df.columns[2]], 2):.2f}"
        )
        self.assertTrue(data["attributes"]["unit_of_measurement"] == "W")
        self.assertTrue(data["attributes"]["friendly_name"] == "PV Forecast")
        self.assertIsInstance(data["attributes"]["forecasts"], list)
        response, data = await self.rh.post_data(
            df["P_batt"],
            25,
            "sensor.p_batt_forecast",
            "power",
            "W",
            "Battery Power Forecast",
            type_var="batt",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["attributes"]["unit_of_measurement"] == "W")
        self.assertTrue(data["attributes"]["friendly_name"] == "Battery Power Forecast")
        response, data = await self.rh.post_data(
            df["SOC_opt"],
            25,
            "sensor.SOC_forecast",
            "battery",
            "%",
            "Battery SOC Forecast",
            type_var="SOC",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["attributes"]["unit_of_measurement"] == "%")
        self.assertTrue(data["attributes"]["friendly_name"] == "Battery SOC Forecast")


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
