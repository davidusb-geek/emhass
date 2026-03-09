import _pickle as cPickle
import asyncio
import bz2
import copy
import datetime
import pathlib
import pickle
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiofiles
import numpy as np
import orjson
import pandas as pd
from aioresponses import aioresponses

from emhass import utils
from emhass.retrieve_hass import RetrieveHass
from emhass.utils import get_days_list, get_logger, get_yaml_parse

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
        emhass_conf["data_path"] = root / "data/"

        self.get_data_from_file = True
        save_data_to_file = False
        model_type = "test_df_final"  # Options: "test_df_final" or "long_train_data"

        # Build params with default secrets (no config)
        if emhass_conf["defaults_path"].exists():
            if self.get_data_from_file:
                _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
                params = await utils.build_params(emhass_conf, secrets, {}, logger)
                retrieve_hass_conf, _, _ = get_yaml_parse(params, logger)
            else:
                emhass_conf["secrets_path"] = root / "secrets_emhass.yaml"
                config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
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
                "config_defaults. does not exist in path: " + str(emhass_conf["defaults_path"])
            )

        # Force config params for testing
        retrieve_hass_conf["optimization_time_step"] = pd.to_timedelta(30, "minutes")
        retrieve_hass_conf["sensor_power_photovoltaics"] = "sensor.power_photovoltaics"
        retrieve_hass_conf["sensor_power_photovoltaics_forecast"] = "sensor.p_pv_forecast"
        retrieve_hass_conf["sensor_power_load_no_var_loads"] = "sensor.power_load_no_var_loads"
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
                self.rh.df_final, self.days_list, self.var_list, self.rh.ha_config = pickle.loads(
                    content
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
                async with aiofiles.open(
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

    async def asyncTearDown(self):
        """Clean up after each test - close any open HTTP sessions."""
        if hasattr(self, "rh") and self.rh is not None:
            await self.rh.close()

    # Check yaml parse in setUp worked
    def test_get_yaml_parse(self):
        self.assertIsInstance(self.retrieve_hass_conf, dict)
        self.assertIn("hass_url", self.retrieve_hass_conf.keys())
        if self.get_data_from_file:
            self.assertEqual(self.retrieve_hass_conf["hass_url"], "https://myhass.duckdns.org/")

    # Check yaml parse worked
    async def test_yaml_parse_web_server(self):
        params = {}
        if emhass_conf["defaults_path"].exists():
            async with aiofiles.open(emhass_conf["defaults_path"]) as file:
                data = await file.read()
                defaults = orjson.loads(data)
                params.update(await utils.build_params(emhass_conf, {}, defaults, logger))
        _, optim_conf, _ = get_yaml_parse(params, logger)
        # Just check forecast methods
        self.assertIsNot(optim_conf.get("weather_forecast_method"), None)
        self.assertIsNot(optim_conf.get("load_forecast_method"), None)
        self.assertIsNot(optim_conf.get("load_cost_forecast_method"), None)
        self.assertIsNot(optim_conf.get("production_price_forecast_method"), None)

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
            test_data_path = emhass_conf["data_path"] / "test_response_get_data_get_method.pbz2"

            async with aiofiles.open(test_data_path, "rb") as f:
                compressed = await f.read()

            data = bz2.decompress(compressed)
            data = cPickle.loads(data)
            data = orjson.loads(data.content)
            days_list = get_days_list(1)
            var_list = [self.retrieve_hass_conf["sensor_power_load_no_var_loads"]]
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
            self.assertIsInstance(self.rh.df_final.index, pd.core.indexes.datetimes.DatetimeIndex)
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
    def test_prepare_data(self):
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertIsInstance(self.rh.df_final.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.rh.df_final.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(len(self.rh.df_final.columns), len(self.var_list))
        self.assertEqual(self.rh.df_final.index.isin(self.days_list).sum(), len(self.days_list))
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
        self.assertEqual(self.rh.df_final.index.tz, self.retrieve_hass_conf["time_zone"])

    # Test negative load
    def test_prepare_data_negative_load(self):
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
        self.assertEqual(self.rh.df_final.index.tz, self.retrieve_hass_conf["time_zone"])

    # Tests that the prepare_data method does convert missing PV values to zero
    # and also ignores any missing sensor columns.
    def test_prepare_data_missing_pv(self):
        load_sensor = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        actual_pv_sensor = self.retrieve_hass_conf["sensor_power_photovoltaics"]
        forecast_pv_sensor = self.retrieve_hass_conf["sensor_power_photovoltaics_forecast"]
        var_replace_zero = [actual_pv_sensor, forecast_pv_sensor, "sensor.missing1"]
        var_interp = [actual_pv_sensor, load_sensor, "sensor.missing2"]
        # Replace actual and forecast PV zero values with NaN's (to test they get replaced back)
        self.rh.df_final[actual_pv_sensor] = self.rh.df_final[actual_pv_sensor].replace(0, np.nan)
        self.rh.df_final[forecast_pv_sensor] = self.rh.df_final[forecast_pv_sensor].replace(
            0, np.nan
        )
        # Verify a non-zero number of missing values in the actual and forecast PV columns before prepare_data
        self.assertGreater(self.rh.df_final[actual_pv_sensor].isna().sum(), 0)
        self.assertGreater(self.rh.df_final[forecast_pv_sensor].isna().sum(), 0)
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
        self.assertEqual(self.rh.df_final[actual_pv_sensor].isna().sum(), 0)
        self.assertEqual(self.rh.df_final[forecast_pv_sensor].isna().sum(), 0)

    # Proposed new test method for InfluxDB
    @patch("influxdb.InfluxDBClient", autospec=True)
    async def test_get_data_influxdb_mock(self, mock_influx_client_class):
        """
        Test the get_data_influxdb method by mocking the InfluxDB client.
        """
        # Build a correctly structured params dictionary for the test
        params_influx = {
            "retrieve_hass_conf": {
                "use_influxdb": True,
                "influxdb_host": "fake-host",
                "influxdb_port": 8086,
                "influxdb_username": "fake-user",
                "influxdb_password": "fake-pass",  # pragma: allowlist secret
                "influxdb_database": "fake-db",
                "influxdb_measurement": "W",
                # Add other necessary keys from the original conf
                "sensor_power_photovoltaics": self.retrieve_hass_conf["sensor_power_photovoltaics"],
                "sensor_power_load_no_var_loads": self.retrieve_hass_conf[
                    "sensor_power_load_no_var_loads"
                ],
            }
        }

        # Instantiate RetrieveHass with the correctly nested configuration
        rh_influx = RetrieveHass(
            self.retrieve_hass_conf["hass_url"],
            self.retrieve_hass_conf["long_lived_token"],
            self.retrieve_hass_conf["optimization_time_step"],
            self.retrieve_hass_conf["time_zone"],
            params_influx,
            emhass_conf,
            logger,
            get_data_from_file=False,
        )

        # Mock the client instance that will be created inside the method
        mock_client_instance = mock_influx_client_class.return_value

        # Define mock data points to be returned by the client
        mock_pv_data = [
            {"time": "2023-04-01T10:00:00Z", "mean_value": 1500.0},
            {"time": "2023-04-01T10:30:00Z", "mean_value": 1800.0},
        ]
        mock_load_data = [
            {"time": "2023-04-01T10:00:00Z", "mean_value": 500.0},
            {"time": "2023-04-01T10:30:00Z", "mean_value": 450.0},
        ]

        # Define a side_effect function to handle different queries
        def query_side_effect(query):
            mock_result = MagicMock()
            if "SHOW MEASUREMENTS" in query:
                mock_result.get_points.return_value = [{"name": "W"}]
            elif "SHOW TAG VALUES" in query and '"W"' in query:
                mock_result.get_points.return_value = [
                    {"value": "power_photovoltaics"},
                    {"value": "power_load_no_var_loads"},
                ]
            elif "entity_id" in query and "'power_photovoltaics'" in query:
                mock_result.get_points.return_value = mock_pv_data
            elif "entity_id" in query and "'power_load_no_var_loads'" in query:
                mock_result.get_points.return_value = mock_load_data
            else:
                mock_result.get_points.return_value = []
            return mock_result

        # Assign the handler to the mock instance's query method
        mock_client_instance.query.side_effect = query_side_effect

        # Define the inputs for the get_data method
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        var_list = [
            params_influx["retrieve_hass_conf"]["sensor_power_photovoltaics"],
            params_influx["retrieve_hass_conf"]["sensor_power_load_no_var_loads"],
        ]

        # Call the method to be tested
        success = await rh_influx.get_data(days_list, var_list)

        # Verify the outcomes
        self.assertTrue(success)  # Check if the method reports success

        # Verify that the InfluxDB client was initialized correctly
        mock_influx_client_class.assert_called_with(
            host="fake-host",
            port=8086,
            username="fake-user",
            password="fake-pass",  # pragma: allowlist secret
            database="fake-db",
            ssl=False,
            verify_ssl=False,
        )
        mock_client_instance.ping.assert_called_once()
        mock_client_instance.close.assert_called_once()

        # Verify the resulting DataFrame
        df = rh_influx.df_final
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.index), 2)
        self.assertEqual(list(df.columns), var_list)
        self.assertEqual(
            df.loc["2023-04-01 10:00:00+00:00"]["sensor.power_photovoltaics"],
            1500.0,
        )
        self.assertEqual(
            df.loc["2023-04-01 10:30:00+00:00"]["sensor.power_load_no_var_loads"],
            450.0,
        )

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
        self.assertEqual(
            data["state"],
            f"{np.round(self.df_raw.loc[self.df_raw.index[10], self.df_raw.columns[0]], 2):.2f}",
        )
        self.assertEqual(data["attributes"]["unit_of_measurement"], "Unit")
        self.assertEqual(data["attributes"]["friendly_name"], "Variable")
        # Lets test publishing a forecast with more added attributes
        df = copy.deepcopy(self.df_raw.iloc[0:30])
        df.columns = ["P_Load", "P_PV", "p_pv_forecast"]
        df["P_batt"] = 1000.0
        df["SOC_opt"] = 0.5
        response, data = await self.rh.post_data(
            df["p_pv_forecast"],
            10,
            "sensor.p_pv_forecast",
            "power",
            "W",
            "PV Forecast",
            type_var="power",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["state"], f"{np.round(df.loc[df.index[10], df.columns[2]], 2):.2f}")
        self.assertEqual(data["attributes"]["unit_of_measurement"], "W")
        self.assertEqual(data["attributes"]["friendly_name"], "PV Forecast")
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
        self.assertEqual(data["attributes"]["unit_of_measurement"], "W")
        self.assertEqual(data["attributes"]["friendly_name"], "Battery Power Forecast")
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
        self.assertEqual(data["attributes"]["unit_of_measurement"], "%")
        self.assertEqual(data["attributes"]["friendly_name"], "Battery SOC Forecast")

    @patch("emhass.retrieve_hass.get_websocket_client")
    async def test_get_ha_config(self, mock_get_ws):
        # Test REST API success
        with aioresponses() as mocked:
            mocked.get(
                self.retrieve_hass_conf["hass_url"] + "api/config",
                payload={"time_zone": "Europe/Paris", "currency": "EUR"},
                status=200,
            )
            self.rh.use_websocket = False
            result = await self.rh.get_ha_config()
            self.assertTrue(result)
            self.assertEqual(self.rh.ha_config["time_zone"], "Europe/Paris")

        # Test REST API failure (401)
        with aioresponses() as mocked:
            mocked.get(
                self.retrieve_hass_conf["hass_url"] + "api/config",
                status=401,
            )
            result = await self.rh.get_ha_config()
            self.assertFalse(result)

        # Test WebSocket success
        self.rh.use_websocket = True
        mock_client = MagicMock()
        mock_client.get_config = AsyncMock(return_value={"time_zone": "Asia/Tokyo"})
        mock_get_ws.return_value = mock_client

        result = await self.rh.get_ha_config()
        self.assertEqual(result, {"time_zone": "Asia/Tokyo"})

        # Reset for other tests
        self.rh.use_websocket = False

    @patch("emhass.retrieve_hass.get_websocket_client", new_callable=AsyncMock)
    @patch("emhass.retrieve_hass.RetrieveHass._get_data_rest_api")
    async def test_get_data_websocket(self, mock_rest_fallback, mock_get_ws):
        # Setup common vars
        days_list = pd.date_range(start="2024-01-01", periods=2, freq="D", tz="UTC")
        var_list = ["sensor.power_load"]

        # Test Successful WebSocket Retrieval
        self.rh.use_websocket = True

        # Configure the mock client
        mock_client = MagicMock()
        mock_get_ws.return_value = mock_client

        # Mock statistics return data with ISO timestamp to ensure robust parsing
        start_iso = days_list[0].isoformat()
        mock_stats = {
            "sensor.power_load": [
                {"start": start_iso, "mean": 1000.0},
                # Add more data points to ensure valid resampling
                {"start": (days_list[0] + pd.Timedelta("30min")).isoformat(), "mean": 1500.0},
            ]
        }
        mock_client.get_statistics = AsyncMock(return_value=mock_stats)

        success = await self.rh.get_data_websocket(days_list, var_list)

        self.assertTrue(success, "get_data_websocket returned False")
        self.assertFalse(self.rh.df_final.empty, "Resulting DataFrame is empty")
        self.assertIn("sensor.power_load", self.rh.df_final.columns)

        # Test Connection Failure -> Fallback to REST
        mock_get_ws.side_effect = Exception("Connection refused")
        mock_rest_fallback.return_value = True  # Mock REST success

        success = await self.rh.get_data(days_list, var_list)

        self.assertTrue(success)
        mock_rest_fallback.assert_called_once()

        # Reset side effect
        mock_get_ws.side_effect = None
        self.rh.use_websocket = False

    async def test_get_data_rest_api_errors(self):
        days_list = pd.date_range(start="2024-01-01", periods=1, freq="D", tz="UTC")
        var_list = ["sensor.test"]
        url = (
            self.retrieve_hass_conf["hass_url"]
            + "api/history/period/"
            + days_list[0].isoformat()
            + "?filter_entity_id=sensor.test"
        )

        # Test Connection Error (Exception)
        with aioresponses() as mocked:
            mocked.get(url, exception=Exception("Network down"))
            result = await self.rh._get_data_rest_api(days_list, var_list)
            self.assertFalse(result)

        # Test 401 Unauthorized
        with aioresponses() as mocked:
            mocked.get(url, status=401)
            result = await self.rh._get_data_rest_api(days_list, var_list)
            self.assertFalse(result)

        # Test Empty JSON Response (IndexError)
        with aioresponses() as mocked:
            mocked.get(url, payload=[], status=200)
            result = await self.rh._get_data_rest_api(days_list, var_list)
            self.assertFalse(result)

    @patch("aiofiles.open")
    async def test_post_data_extended(self, mock_aio_open):
        self.rh.get_data_from_file = False

        # Setup mock file context for save_entities=True
        mock_f = AsyncMock()
        mock_aio_open.return_value.__aenter__.return_value = mock_f

        # Create dummy data
        idx = 0
        entity_id = "sensor.p_pv_forecast"
        data_df = pd.Series(
            [100.55, 200.00], index=pd.date_range("2024-01-01", periods=2, freq="30min")
        )
        data_df.name = "test_data"

        # Test "cost_fun" type
        response, data = await self.rh.post_data(
            data_df, idx, entity_id, "monetary", "EUR", "Cost Function", "cost_fun"
        )
        self.assertEqual(data["state"], f"{data_df.sum():.2f}")

        # Test "optim_status" type
        status_df = pd.Series(["Optimal"], index=[0])
        response, data = await self.rh.post_data(
            status_df, 0, "sensor.optim_status", "none", "", "Status", "optim_status"
        )
        self.assertEqual(data["state"], "Optimal")

        # Test "deferrable" type (complex attributes)
        response, data = await self.rh.post_data(
            data_df, idx, entity_id, "power", "W", "Deferrable", "deferrable"
        )
        self.assertIn("deferrables_schedule", data["attributes"])

        # Test "unit_load_cost" (4 decimals)
        response, data = await self.rh.post_data(
            data_df, idx, entity_id, "monetary", "EUR/kWh", "Load Cost", "unit_load_cost"
        )
        self.assertEqual(data["state"], f"{data_df.iloc[0]:.4f}")
        self.assertIn("unit_load_cost_forecasts", data["attributes"])

        # Test save_entities=True
        # Save old path to restore later
        original_path = self.rh.emhass_conf["data_path"]
        try:
            self.rh.emhass_conf["data_path"] = pathlib.Path("/tmp")

            # Mock os.path.isfile to return False (triggers new metadata file creation)
            with patch("os.path.isfile", return_value=False):
                # Mock pathlib.Path.mkdir to avoid file system errors
                with patch("pathlib.Path.mkdir"):
                    # FIX: Pass dont_post=True to bypass network failure and force response_ok=True
                    # This ensures the save_entities logic block is actually reached
                    response, data = await self.rh.post_data(
                        data_df,
                        idx,
                        entity_id,
                        "power",
                        "W",
                        "PV",
                        "power",
                        save_entities=True,
                        dont_post=True,
                    )
        finally:
            # Restore path to prevent polluting other tests
            self.rh.emhass_conf["data_path"] = original_path

        # Verify file write called (once for data, once for metadata)
        self.assertTrue(mock_f.write.called)
        self.assertGreaterEqual(mock_f.write.call_count, 2)

        # Test Error Handling (response_ok = False)
        # We need to un-patch the aioresponses or create a new specific patch for client session
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.ok = False
            mock_resp.status = 500
            mock_resp.__aenter__.return_value = mock_resp
            mock_post.return_value = mock_resp

            # Use dont_post=False to force network attempt
            # Ensure get_data_from_file is False (set at start of test)
            response, _ = await self.rh.post_data(
                data_df, idx, entity_id, "power", "W", "Fail", "power", dont_post=False
            )

            self.assertFalse(response.ok)
            self.assertEqual(response.status_code, 500)

    async def test_session_lazy_initialization(self):
        """Test that session is lazily initialized on first use."""
        # Session should be None initially
        self.assertIsNone(self.rh._session)

        # Get session should create one
        session = await self.rh._get_session()
        self.assertIsNotNone(session)
        self.assertFalse(session.closed)

        # Getting session again should return the same instance
        session2 = await self.rh._get_session()
        self.assertIs(session, session2)

        # Clean up
        await self.rh.close()

    async def test_session_reuse_across_post_data_calls(self):
        """Test that the same session is reused across multiple post_data calls."""
        self.rh.get_data_from_file = False

        # Create test data
        data_df = pd.Series(
            [100.0, 200.0], index=pd.date_range("2024-01-01", periods=2, freq="30min")
        )

        # Mock aiohttp session.post to track calls
        with patch.object(self.rh, "_get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.ok = True
            mock_response.status = 200
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_get_session.return_value = mock_session

            # Make multiple post_data calls
            await self.rh.post_data(data_df, 0, "sensor.test1", "power", "W", "Test 1", "power")
            await self.rh.post_data(data_df, 0, "sensor.test2", "power", "W", "Test 2", "power")
            await self.rh.post_data(data_df, 0, "sensor.test3", "power", "W", "Test 3", "power")

            # _get_session should have been called 3 times (once per post_data)
            # but it returns the same session each time
            self.assertEqual(mock_get_session.call_count, 3)

    async def test_session_close(self):
        """Test that close() properly closes the session."""
        # Create a session first
        session = await self.rh._get_session()
        self.assertIsNotNone(session)
        self.assertFalse(session.closed)

        # Close should work
        await self.rh.close()
        self.assertIsNone(self.rh._session)

        # Closing again should be a no-op (no error)
        await self.rh.close()

    async def test_session_recreated_after_close(self):
        """Test that a new session is created after closing the old one."""
        # Create initial session
        session1 = await self.rh._get_session()

        # Close it
        await self.rh.close()

        # Get a new session
        session2 = await self.rh._get_session()

        # Should be a different session
        self.assertIsNot(session1, session2)
        self.assertFalse(session2.closed)

        # Clean up
        await self.rh.close()

    async def test_async_context_manager(self):
        """Test that RetrieveHass works as an async context manager."""
        async with self.rh as rh:
            # Should return self
            self.assertIs(rh, self.rh)
            # Create a session inside the context
            session = await rh._get_session()
            self.assertIsNotNone(session)
            self.assertFalse(session.closed)

        # After exiting context, session should be closed
        self.assertIsNone(self.rh._session)

    async def test_concurrent_get_session(self):
        """Test that concurrent _get_session calls only create one session."""
        # Launch multiple concurrent _get_session calls
        sessions = await asyncio.gather(
            self.rh._get_session(),
            self.rh._get_session(),
            self.rh._get_session(),
        )
        # All should return the same session instance
        self.assertIs(sessions[0], sessions[1])
        self.assertIs(sessions[1], sessions[2])

        # Clean up
        await self.rh.close()


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
