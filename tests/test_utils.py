#!/usr/bin/env python

import pathlib
import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np
import orjson
import pandas as pd
import pytz

from emhass import utils

# The root folder
root = pathlib.Path(utils.get_root(__file__, num_parent=2))
# Build emhass_conf paths
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["options_path"] = root / "options.json"
emhass_conf["config_path"] = root / "config.json"
emhass_conf["secrets_path"] = root / "secrets_emhass(example).yaml"
emhass_conf["legacy_config_path"] = (
    pathlib.Path(utils.get_root(__file__, num_parent=1)) / "config_emhass.yaml"
)
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

# Create logger
logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)


class TestCommandLineUtils(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def get_test_params():
        print(emhass_conf["legacy_config_path"])
        # Build params with default config and secrets
        if emhass_conf["defaults_path"].exists():
            config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
            _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
            # Add Altitude secret manually for testing get_yaml_parse
            secrets["Altitude"] = 8000.0
            params = await utils.build_params(emhass_conf, secrets, config, logger)
        else:
            raise Exception(
                "config_defaults. does not exist in path: " + str(emhass_conf["defaults_path"])
            )

        return params

    async def asyncSetUp(self):
        params = await TestCommandLineUtils.get_test_params()
        # Add runtime parameters for forecast lists
        runtimeparams = {
            "pv_power_forecast": [i + 1 for i in range(48)],
            "load_power_forecast": [i + 1 for i in range(48)],
            "load_cost_forecast": [i + 1 for i in range(48)],
            "prod_price_forecast": [i + 1 for i in range(48)],
        }
        self.runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params["optim_conf"]["weather_forecast_method"] = "list"
        params["optim_conf"]["load_forecast_method"] = "list"
        params["optim_conf"]["load_cost_forecast_method"] = "list"
        params["optim_conf"]["production_price_forecast_method"] = "list"
        self.params_json = orjson.dumps(params).decode("utf-8")
        # Create dummy data resembling optimization output
        dates = pd.date_range(start="2024-01-01", periods=24, freq="1h")
        self.df = pd.DataFrame(index=dates)
        self.df["P_PV"] = np.random.rand(24) * 1000
        self.df["P_Load"] = np.random.rand(24) * 500
        self.df["optim_status"] = "Optimal"
        self.df["cost_fun_profit"] = 0.5

    async def test_build_config(self):
        # Test building with the different config methods
        config = {}
        params = {}
        # Test with defaults
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        params = await utils.build_params(emhass_conf, {}, config, logger)
        self.assertEqual(params["optim_conf"]["lp_solver"], "default")
        self.assertEqual(params["optim_conf"]["lp_solver_path"], "empty")
        self.assertEqual(
            config["load_peak_hour_periods"],
            {
                "period_hp_1": [{"start": "02:54"}, {"end": "15:24"}],
                "period_hp_2": [{"start": "17:24"}, {"end": "20:24"}],
            },
        )
        self.assertEqual(
            params["retrieve_hass_conf"]["sensor_replace_zero"],
            ["sensor.power_photovoltaics", "sensor.p_pv_forecast"],
        )
        # Test with config.json
        config = await utils.build_config(
            emhass_conf,
            logger,
            emhass_conf["defaults_path"],
            emhass_conf["config_path"],
        )
        params = await utils.build_params(emhass_conf, {}, config, logger)
        self.assertEqual(params["optim_conf"]["lp_solver"], "default")
        self.assertEqual(params["optim_conf"]["lp_solver_path"], "empty")
        # Test with legacy config_emhass yaml
        config = await utils.build_config(
            emhass_conf,
            logger,
            emhass_conf["defaults_path"],
            legacy_config_path=emhass_conf["legacy_config_path"],
        )
        params = await utils.build_params(emhass_conf, {}, config, logger)
        self.assertEqual(
            params["retrieve_hass_conf"]["sensor_replace_zero"], ["sensor.power_photovoltaics"]
        )
        self.assertEqual(
            config["load_peak_hour_periods"],
            {
                "period_hp_1": [{"start": "02:54"}, {"end": "15:24"}],
                "period_hp_2": [{"start": "17:24"}, {"end": "20:24"}],
            },
        )
        self.assertEqual(params["plant_conf"]["battery_charge_efficiency"], 0.95)

    async def test_get_yaml_parse(self):
        # Test get_yaml_parse with only secrets
        params = {}
        updated_emhass_conf, secrets = await utils.build_secrets(emhass_conf, logger)
        emhass_conf.update(updated_emhass_conf)
        params.update(await utils.build_params(emhass_conf, secrets, {}, logger))
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(
            orjson.dumps(params).decode("utf-8"), logger
        )
        self.assertIsInstance(retrieve_hass_conf, dict)
        self.assertIsInstance(optim_conf, dict)
        self.assertIsInstance(plant_conf, dict)
        self.assertEqual(retrieve_hass_conf["Altitude"], 4807.8)
        # Test get_yaml_parse with built params in get_test_params
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json, logger)
        self.assertEqual(retrieve_hass_conf["Altitude"], 8000.0)

    @patch("emhass.utils._get_now")
    def test_get_forecast_dates_standard_day(self, mock_ts_now):
        """
        Tests the forecast date generation on a standard 24-hour day.
        """
        # 1. Define parameters for this specific test
        time_zone = pytz.timezone("Australia/Sydney")
        freq = 60  # in minutes
        delta_forecast = 1  # in days

        # 2. Define the mock 'now' and the expected results
        mock_now = datetime(2025, 10, 11, 7, 0, 0)
        expected_start = "2025-10-11T07:00:00"
        expected_end = "2025-10-12T06:00:00"
        expected_range = pd.date_range(
            start=expected_start, end=expected_end, freq=f"{freq}min", tz=time_zone
        )
        expected_dates = [ts.isoformat() for ts in expected_range]

        # 3. Set the return value for the mock (which is now passed in as an argument)
        mock_ts_now.return_value = mock_now

        actual_dates = utils.get_forecast_dates(freq, delta_forecast, time_zone)

        # 4. Perform assertions
        self.assertIsInstance(actual_dates, list)
        self.assertEqual(len(actual_dates), 24)
        self.assertListEqual(actual_dates, expected_dates)

    @patch("emhass.utils._get_now")
    def test_get_forecast_dates_dst_crossing(self, mock_ts_now):
        """
        Tests the forecast date generation on a day with a DST transition (23 hours).
        """
        # 1. Define parameters for this specific test
        time_zone = pytz.timezone("Australia/Sydney")
        freq = 60  # in minutes
        delta_forecast = 1  # in days

        # 2. Define mock 'now' and expected results
        mock_now = datetime(2025, 10, 4, 23, 0, 0)
        expected_start = "2025-10-04T23:00:00"
        expected_end = "2025-10-05T22:00:00"
        expected_range = pd.date_range(
            start=expected_start, end=expected_end, freq=f"{freq}min", tz=time_zone
        )
        expected_dates = [ts.isoformat() for ts in expected_range]

        # 3. Set the return value for the mock
        mock_ts_now.return_value = mock_now

        actual_dates = utils.get_forecast_dates(freq, delta_forecast, time_zone)
        # 4. Perform assertions
        self.assertIsInstance(actual_dates, list)
        self.assertEqual(len(actual_dates), 23)  # This day correctly has 23 hours
        self.assertListEqual(actual_dates, expected_dates)
        self.assertIn("+10:00", actual_dates[2])
        self.assertIn("+11:00", actual_dates[3])

    async def test_treat_runtimeparams(self):
        # Test dayahead runtime params
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json, logger)
        set_type = "dayahead-optim"
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            self.runtimeparams_json,
            self.params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        self.assertIsInstance(params, str)
        params = orjson.loads(params)
        self.assertIsInstance(params["passed_data"]["pv_power_forecast"], list)
        self.assertIsInstance(params["passed_data"]["load_power_forecast"], list)
        self.assertIsInstance(params["passed_data"]["load_cost_forecast"], list)
        self.assertIsInstance(params["passed_data"]["prod_price_forecast"], list)
        self.assertEqual(optim_conf["weather_forecast_method"], "list")
        self.assertEqual(optim_conf["load_forecast_method"], "list")
        self.assertEqual(optim_conf["load_cost_forecast_method"], "list")
        self.assertEqual(optim_conf["production_price_forecast_method"], "list")
        # Test naive MPC runtime params
        set_type = "naive-mpc-optim"
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            self.runtimeparams_json,
            self.params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        self.assertIsInstance(params, str)
        params = orjson.loads(params)
        self.assertEqual(params["passed_data"]["prediction_horizon"], 10)
        self.assertEqual(
            params["passed_data"]["soc_init"], plant_conf["battery_target_state_of_charge"]
        )
        self.assertEqual(
            params["passed_data"]["soc_final"], plant_conf["battery_target_state_of_charge"]
        )
        self.assertEqual(
            params["optim_conf"]["operating_hours_of_each_deferrable_load"],
            optim_conf["operating_hours_of_each_deferrable_load"],
        )
        # Test passing optimization and plant configuration parameters at runtime
        runtimeparams = orjson.loads(self.runtimeparams_json)
        runtimeparams.update({"number_of_deferrable_loads": 3})
        runtimeparams.update({"nominal_power_of_deferrable_loads": [3000.0, 750.0, 2500.0]})
        runtimeparams.update({"operating_hours_of_each_deferrable_load": [5, 8, 10]})
        runtimeparams.update({"treat_deferrable_load_as_semi_cont": [True, True, True]})
        runtimeparams.update({"set_deferrable_load_single_constant": [False, False, False]})
        runtimeparams.update({"weight_battery_discharge": 2.0})
        runtimeparams.update({"weight_battery_charge": 2.0})
        runtimeparams.update({"solcast_api_key": "yoursecretsolcastapikey"})
        runtimeparams.update({"solcast_rooftop_id": "yourrooftopid"})
        runtimeparams.update({"solar_forecast_kwp": 5.0})
        runtimeparams.update({"battery_target_state_of_charge": 0.4})
        runtimeparams.update({"publish_prefix": "emhass_"})
        runtimeparams.update({"custom_pv_forecast_id": "my_custom_pv_forecast_id"})
        runtimeparams.update({"custom_load_forecast_id": "my_custom_load_forecast_id"})
        runtimeparams.update({"custom_batt_forecast_id": "my_custom_batt_forecast_id"})
        runtimeparams.update({"custom_batt_soc_forecast_id": "my_custom_batt_soc_forecast_id"})
        runtimeparams.update({"custom_grid_forecast_id": "my_custom_grid_forecast_id"})
        runtimeparams.update({"custom_cost_fun_id": "my_custom_cost_fun_id"})
        runtimeparams.update({"custom_optim_status_id": "my_custom_optim_status_id"})
        runtimeparams.update({"custom_unit_load_cost_id": "my_custom_unit_load_cost_id"})
        runtimeparams.update({"custom_unit_prod_price_id": "my_custom_unit_prod_price_id"})
        runtimeparams.update({"custom_deferrable_forecast_id": "my_custom_deferrable_forecast_id"})
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json, logger)
        set_type = "dayahead-optim"
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            runtimeparams,
            self.params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        self.assertIsInstance(params, str)
        params = orjson.loads(params)
        self.assertIsInstance(params["passed_data"]["pv_power_forecast"], list)
        self.assertIsInstance(params["passed_data"]["load_power_forecast"], list)
        self.assertIsInstance(params["passed_data"]["load_cost_forecast"], list)
        self.assertIsInstance(params["passed_data"]["prod_price_forecast"], list)
        self.assertEqual(optim_conf["number_of_deferrable_loads"], 3)
        self.assertEqual(optim_conf["nominal_power_of_deferrable_loads"], [3000.0, 750.0, 2500.0])
        self.assertEqual(optim_conf["operating_hours_of_each_deferrable_load"], [5, 8, 10])
        self.assertEqual(optim_conf["treat_deferrable_load_as_semi_cont"], [True, True, True])
        self.assertEqual(optim_conf["set_deferrable_load_single_constant"], [False, False, False])
        self.assertEqual(optim_conf["weight_battery_discharge"], 2.0)
        self.assertEqual(optim_conf["weight_battery_charge"], 2.0)
        self.assertEqual(retrieve_hass_conf["solcast_api_key"], "yoursecretsolcastapikey")
        self.assertEqual(retrieve_hass_conf["solcast_rooftop_id"], "yourrooftopid")
        self.assertEqual(retrieve_hass_conf["solar_forecast_kwp"], 5.0)
        self.assertEqual(plant_conf["battery_target_state_of_charge"], 0.4)
        self.assertEqual(params["passed_data"]["publish_prefix"], "emhass_")
        self.assertEqual(params["passed_data"]["custom_pv_forecast_id"], "my_custom_pv_forecast_id")
        self.assertEqual(
            params["passed_data"]["custom_load_forecast_id"], "my_custom_load_forecast_id"
        )
        self.assertEqual(
            params["passed_data"]["custom_batt_forecast_id"], "my_custom_batt_forecast_id"
        )
        self.assertEqual(
            params["passed_data"]["custom_batt_soc_forecast_id"], "my_custom_batt_soc_forecast_id"
        )
        self.assertEqual(
            params["passed_data"]["custom_grid_forecast_id"], "my_custom_grid_forecast_id"
        )
        self.assertEqual(params["passed_data"]["custom_cost_fun_id"], "my_custom_cost_fun_id")
        self.assertEqual(
            params["passed_data"]["custom_optim_status_id"], "my_custom_optim_status_id"
        )
        self.assertEqual(
            params["passed_data"]["custom_unit_load_cost_id"], "my_custom_unit_load_cost_id"
        )
        self.assertEqual(
            params["passed_data"]["custom_unit_prod_price_id"], "my_custom_unit_prod_price_id"
        )
        self.assertEqual(
            params["passed_data"]["custom_deferrable_forecast_id"],
            "my_custom_deferrable_forecast_id",
        )

    async def test_treat_runtimeparams_failed(self):
        # Test treatment of nan values
        params = await TestCommandLineUtils.get_test_params()
        runtimeparams = {
            "pv_power_forecast": [1, 2, 3, 4, 5, "nan", 7, 8, 9, 10],
            "load_power_forecast": [1, 2, "nan", 4, 5, 6, 7, 8, 9, 10],
            "load_cost_forecast": [1, 2, 3, 4, 5, 6, 7, 8, "nan", 10],
            "prod_price_forecast": [1, 2, 3, 4, "nan", 6, 7, 8, 9, 10],
        }
        params["passed_data"] = runtimeparams
        params["optim_conf"]["weather_forecast_method"] = "list"
        params["optim_conf"]["load_forecast_method"] = "list"
        params["optim_conf"]["load_cost_forecast_method"] = "list"
        params["optim_conf"]["production_price_forecast_method"] = "list"
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
        set_type = "dayahead-optim"
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            runtimeparams,
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )

        self.assertGreater(
            len([x for x in runtimeparams["pv_power_forecast"] if not str(x).isdigit()]), 0
        )
        self.assertGreater(
            len([x for x in runtimeparams["load_power_forecast"] if not str(x).isdigit()]), 0
        )
        self.assertGreater(
            len([x for x in runtimeparams["load_cost_forecast"] if not str(x).isdigit()]), 0
        )
        self.assertGreater(
            len([x for x in runtimeparams["prod_price_forecast"] if not str(x).isdigit()]), 0
        )
        # Test list embedded into a string
        params = await TestCommandLineUtils.get_test_params()
        runtimeparams = {
            "pv_power_forecast": "[1,2,3,4,5,6,7,8,9,10]",
            "load_power_forecast": "[1,2,3,4,5,6,7,8,9,10]",
            "load_cost_forecast": "[1,2,3,4,5,6,7,8,9,10]",
            "prod_price_forecast": "[1,2,3,4,5,6,7,8,9,10]",
        }
        params["passed_data"] = runtimeparams
        params["optim_conf"]["weather_forecast_method"] = "list"
        params["optim_conf"]["load_forecast_method"] = "list"
        params["optim_conf"]["load_cost_forecast_method"] = "list"
        params["optim_conf"]["production_price_forecast_method"] = "list"
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
        set_type = "dayahead-optim"
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            runtimeparams,
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        self.assertIsInstance(runtimeparams["pv_power_forecast"], list)
        self.assertIsInstance(runtimeparams["load_power_forecast"], list)
        self.assertIsInstance(runtimeparams["load_cost_forecast"], list)
        self.assertIsInstance(runtimeparams["prod_price_forecast"], list)
        # Test string of numbers
        params = await TestCommandLineUtils.get_test_params()
        runtimeparams = {
            "pv_power_forecast": "1,2,3,4,5,6,7,8,9,10",
            "load_power_forecast": "1,2,3,4,5,6,7,8,9,10",
            "load_cost_forecast": "1,2,3,4,5,6,7,8,9,10",
            "prod_price_forecast": "1,2,3,4,5,6,7,8,9,10",
        }
        params["passed_data"] = runtimeparams
        params["optim_conf"]["weather_forecast_method"] = "list"
        params["optim_conf"]["load_forecast_method"] = "list"
        params["optim_conf"]["load_cost_forecast_method"] = "list"
        params["optim_conf"]["production_price_forecast_method"] = "list"
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params, logger)
        set_type = "dayahead-optim"
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            runtimeparams,
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        self.assertIsInstance(runtimeparams["pv_power_forecast"], str)
        self.assertIsInstance(runtimeparams["load_power_forecast"], str)
        self.assertIsInstance(runtimeparams["load_cost_forecast"], str)
        self.assertIsInstance(runtimeparams["prod_price_forecast"], str)

    async def test_update_params_with_ha_config(self):
        # Test dayahead runtime params
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json, logger)
        set_type = "dayahead-optim"
        (
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
        ) = await utils.treat_runtimeparams(
            self.runtimeparams_json,
            self.params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )
        ha_config = {"currency": "USD", "unit_system": {"temperature": "°F"}}
        params_with_ha_config_json = utils.update_params_with_ha_config(
            params,
            ha_config,
        )
        params_with_ha_config = orjson.loads(params_with_ha_config_json)
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_cost_fun_id"]["unit_of_measurement"], "$"
        )
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_unit_load_cost_id"]["unit_of_measurement"],
            "$/kWh",
        )
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_unit_prod_price_id"][
                "unit_of_measurement"
            ],
            "$/kWh",
        )

    async def test_update_params_with_ha_config_special_case(self):
        # Test special passed runtime params
        runtimeparams = {
            "prediction_horizon": 28,
            "pv_power_forecast": [
                523,
                873,
                1059,
                1195,
                1291,
                1352,
                1366,
                1327,
                1254,
                1150,
                1004,
                813,
                589,
                372,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                153,
                228,
                301,
                363,
                407,
                438,
                456,
                458,
                443,
                417,
                381,
                332,
                269,
                195,
                123,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "num_def_loads": 2,
            "P_deferrable_nom": [0, 0],
            "def_total_hours": [0, 0],
            "treat_def_as_semi_cont": [1, 1],
            "set_def_constant": [1, 1],
            "def_start_timestep": [0, 0],
            "def_end_timestep": [0, 0],
            "soc_init": 0.64,
            "soc_final": 0.9,
            "load_cost_forecast": [
                0.2751,
                0.2751,
                0.2729,
                0.2729,
                0.2748,
                0.2748,
                0.2746,
                0.2746,
                0.2815,
                0.2815,
                0.2841,
                0.2841,
                0.282,
                0.282,
                0.288,
                0.288,
                0.29,
                0.29,
                0.2841,
                0.2841,
                0.2747,
                0.2747,
                0.2677,
                0.2677,
                0.2628,
                0.2628,
                0.2532,
                0.2532,
            ],
            "prod_price_forecast": [
                0.1213,
                0.1213,
                0.1192,
                0.1192,
                0.121,
                0.121,
                0.1208,
                0.1208,
                0.1274,
                0.1274,
                0.1298,
                0.1298,
                0.1278,
                0.1278,
                0.1335,
                0.1335,
                0.1353,
                0.1353,
                0.1298,
                0.1298,
                0.1209,
                0.1209,
                0.1143,
                0.1143,
                0.1097,
                0.1097,
                0.1007,
                0.1007,
            ],
            "alpha": 1,
            "beta": 0,
            "load_power_forecast": [
                399,
                300,
                400,
                600,
                300,
                200,
                200,
                200,
                200,
                300,
                300,
                200,
                400,
                200,
                200,
                400,
                400,
                400,
                300,
                300,
                300,
                600,
                800,
                500,
                400,
                400,
                500,
                500,
                2400,
                2300,
                2400,
                2400,
                2300,
                2400,
                2400,
                2400,
                2300,
                2400,
                2400,
                200,
                200,
                300,
                300,
                300,
                300,
                300,
                300,
                300,
            ],
        }
        params_ = orjson.loads(self.params_json)
        params_["passed_data"].update(runtimeparams)

        runtimeparams_json = orjson.dumps(runtimeparams).decode()
        params_json = orjson.dumps(params_).decode()

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
        ha_config = {"currency": "USD"}
        params_with_ha_config_json = utils.update_params_with_ha_config(
            params,
            ha_config,
        )
        params_with_ha_config = orjson.loads(params_with_ha_config_json)
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_cost_fun_id"]["unit_of_measurement"], "$"
        )
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_unit_load_cost_id"]["unit_of_measurement"],
            "$/kWh",
        )
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_unit_prod_price_id"][
                "unit_of_measurement"
            ],
            "$/kWh",
        )
        # Test with 0 deferrable loads
        runtimeparams = {
            "prediction_horizon": 28,
            "pv_power_forecast": [
                523,
                873,
                1059,
                1195,
                1291,
                1352,
                1366,
                1327,
                1254,
                1150,
                1004,
                813,
                589,
                372,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                153,
                228,
                301,
                363,
                407,
                438,
                456,
                458,
                443,
                417,
                381,
                332,
                269,
                195,
                123,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "num_def_loads": 0,
            "def_start_timestep": [0, 0],
            "def_end_timestep": [0, 0],
            "soc_init": 0.64,
            "soc_final": 0.9,
            "load_cost_forecast": [
                0.2751,
                0.2751,
                0.2729,
                0.2729,
                0.2748,
                0.2748,
                0.2746,
                0.2746,
                0.2815,
                0.2815,
                0.2841,
                0.2841,
                0.282,
                0.282,
                0.288,
                0.288,
                0.29,
                0.29,
                0.2841,
                0.2841,
                0.2747,
                0.2747,
                0.2677,
                0.2677,
                0.2628,
                0.2628,
                0.2532,
                0.2532,
            ],
            "prod_price_forecast": [
                0.1213,
                0.1213,
                0.1192,
                0.1192,
                0.121,
                0.121,
                0.1208,
                0.1208,
                0.1274,
                0.1274,
                0.1298,
                0.1298,
                0.1278,
                0.1278,
                0.1335,
                0.1335,
                0.1353,
                0.1353,
                0.1298,
                0.1298,
                0.1209,
                0.1209,
                0.1143,
                0.1143,
                0.1097,
                0.1097,
                0.1007,
                0.1007,
            ],
            "alpha": 1,
            "beta": 0,
            "load_power_forecast": [
                399,
                300,
                400,
                600,
                300,
                200,
                200,
                200,
                200,
                300,
                300,
                200,
                400,
                200,
                200,
                400,
                400,
                400,
                300,
                300,
                300,
                600,
                800,
                500,
                400,
                400,
                500,
                500,
                2400,
                2300,
                2400,
                2400,
                2300,
                2400,
                2400,
                2400,
                2300,
                2400,
                2400,
                200,
                200,
                300,
                300,
                300,
                300,
                300,
                300,
                300,
            ],
        }
        params_ = orjson.loads(self.params_json)
        params_["passed_data"].update(runtimeparams)
        runtimeparams_json = orjson.dumps(runtimeparams).decode()
        params_json = orjson.dumps(params_).decode()
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
        ha_config = {"currency": "USD"}
        params_with_ha_config_json = utils.update_params_with_ha_config(
            params,
            ha_config,
        )
        params_with_ha_config = orjson.loads(params_with_ha_config_json)
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_cost_fun_id"]["unit_of_measurement"], "$"
        )
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_unit_load_cost_id"]["unit_of_measurement"],
            "$/kWh",
        )
        self.assertEqual(
            params_with_ha_config["passed_data"]["custom_unit_prod_price_id"][
                "unit_of_measurement"
            ],
            "$/kWh",
        )

    async def test_build_secrets(self):
        # Test the build_secrets defaults from get_test_params()
        params = await TestCommandLineUtils.get_test_params()
        expected_keys = [
            "retrieve_hass_conf",
            "params_secrets",
            "optim_conf",
            "plant_conf",
            "passed_data",
        ]
        for key in expected_keys:
            self.assertIn(key, params.keys())
        self.assertEqual(params["retrieve_hass_conf"]["time_zone"], "Europe/Paris")
        self.assertEqual(params["retrieve_hass_conf"]["hass_url"], "https://myhass.duckdns.org/")
        self.assertEqual(params["retrieve_hass_conf"]["long_lived_token"], "thatverylongtokenhere")
        # Test Secrets from options.json
        params = {}
        secrets = {}
        _, secrets = await utils.build_secrets(
            emhass_conf,
            logger,
            options_path=emhass_conf["options_path"],
            secrets_path="",
            no_response=True,
        )
        params = await utils.build_params(emhass_conf, secrets, {}, logger)
        for key in expected_keys:
            self.assertIn(key, params.keys())
        self.assertEqual(params["retrieve_hass_conf"]["time_zone"], "Europe/Paris")
        self.assertEqual(params["retrieve_hass_conf"]["hass_url"], "https://myhass.duckdns.org/")
        self.assertEqual(params["retrieve_hass_conf"]["long_lived_token"], "thatverylongtokenhere")
        # Test Secrets from secrets_emhass(example).yaml
        params = {}
        secrets = {}
        _, secrets = await utils.build_secrets(
            emhass_conf, logger, secrets_path=emhass_conf["secrets_path"]
        )
        params = await utils.build_params(emhass_conf, secrets, {}, logger)
        for key in expected_keys:
            self.assertIn(key, params.keys())
        self.assertEqual(params["retrieve_hass_conf"]["time_zone"], "Europe/Paris")
        self.assertEqual(params["retrieve_hass_conf"]["hass_url"], "https://myhass.duckdns.org/")
        self.assertEqual(params["retrieve_hass_conf"]["long_lived_token"], "thatverylongtokenhere")
        # Test Secrets from arguments (command_line cli)
        params = {}
        secrets = {}
        _, secrets = await utils.build_secrets(
            emhass_conf, logger, {"url": "test.url", "key": "test.key"}, secrets_path=""
        )
        logger.debug("Obtaining long_lived_token from passed argument")
        params = await utils.build_params(emhass_conf, secrets, {}, logger)
        for key in expected_keys:
            self.assertIn(key, params.keys())
        self.assertEqual(params["retrieve_hass_conf"]["time_zone"], "Europe/Paris")
        self.assertEqual(params["retrieve_hass_conf"]["hass_url"], "test.url")
        self.assertEqual(params["retrieve_hass_conf"]["long_lived_token"], "test.key")

    async def test_get_injection_dict_with_thermal(self):
        # Add thermal columns to dummy df
        self.df["predicted_temp_heater1"] = 21.0
        self.df["target_temp_heater1"] = 22.0
        # Run function
        injection_dict = utils.get_injection_dict(self.df.copy())
        # Verify Keys
        self.assertIn("figure_0", injection_dict, "Powers plot missing")
        self.assertIn("figure_thermal", injection_dict, "Thermal plot missing")
        self.assertIn("figure_2", injection_dict, "Cost plot missing")
        # Verify Content
        self.assertIn("Thermal loads temperature schedule", injection_dict["figure_thermal"])
        self.assertIn("Temperature (°C)", injection_dict["figure_thermal"])

    async def test_get_injection_dict_without_thermal(self):
        # Ensure no thermal columns
        cols = [c for c in self.df.columns if "heater" not in c]
        df_clean = self.df[cols].copy()
        # Run function
        injection_dict = utils.get_injection_dict(df_clean)
        # Verify Thermal is NOT present
        self.assertNotIn("figure_thermal", injection_dict)
        self.assertIn("figure_0", injection_dict)


class TestHeatingDemand(unittest.TestCase):
    def test_calculate_heating_demand_basic(self):
        """Test heating demand calculation with basic parameters."""
        specific_heating_demand = 100.0  # kWh/m²/year
        floor_area = 150.0  # m²
        # Outdoor temps: cold weather requiring heating
        outdoor_temps = np.array([5.0, 10.0, 15.0, 8.0, 12.0, 6.0, 9.0, 11.0, 7.0, 13.0])
        base_temperature = 18.0
        annual_reference_hdd = 3000.0
        optimization_time_step = 30  # minutes

        heating_demand = utils.calculate_heating_demand(
            specific_heating_demand,
            floor_area,
            outdoor_temps,
            base_temperature,
            annual_reference_hdd,
            optimization_time_step,
        )

        # Verify output is numpy array
        self.assertIsInstance(heating_demand, np.ndarray)
        # Verify output length matches input length
        self.assertEqual(len(heating_demand), len(outdoor_temps))
        # Verify all values are non-negative
        self.assertTrue(np.all(heating_demand >= 0.0))

        # Manual verification for first timestep: outdoor_temp = 5°C
        # HDD = max(18 - 5, 0) = 13 degree-days
        # HDD scaled to 30 min = 13 * (0.5 / 24) = 0.270833
        # heating_demand = 100 * 150 * (0.270833 / 3000) = 1.354 kWh
        hdd_first = max(base_temperature - outdoor_temps[0], 0.0)
        hours_per_timestep = optimization_time_step / 60.0
        hdd_scaled = hdd_first * (hours_per_timestep / 24.0)
        expected_demand = specific_heating_demand * floor_area * (hdd_scaled / annual_reference_hdd)
        self.assertAlmostEqual(heating_demand[0], expected_demand, places=6)

    def test_calculate_heating_demand_no_heating_needed(self):
        """Test heating demand when outdoor temp exceeds base temperature."""
        specific_heating_demand = 100.0
        floor_area = 150.0
        # Summer temperatures - all above base temperature
        outdoor_temps = np.array([20.0, 25.0, 22.0, 24.0, 28.0])
        base_temperature = 18.0

        heating_demand = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps, base_temperature
        )

        # All heating demand should be zero when outdoor temp >= base temp
        self.assertTrue(np.allclose(heating_demand, 0.0))

    def test_calculate_heating_demand_pandas_series(self):
        """Test heating demand with pandas Series input."""
        specific_heating_demand = 100.0
        floor_area = 150.0
        outdoor_temps_array = np.array([5.0, 10.0, 15.0, 8.0, 12.0])
        outdoor_temps_series = pd.Series(outdoor_temps_array)

        heating_demand_array = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_array
        )
        heating_demand_series = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_series
        )

        # Results should be identical regardless of input type
        np.testing.assert_array_almost_equal(heating_demand_array, heating_demand_series)

    def test_calculate_heating_demand_different_timestep(self):
        """Test heating demand with different optimization time steps."""
        specific_heating_demand = 100.0
        floor_area = 150.0
        outdoor_temps = np.array([10.0, 12.0, 8.0])
        base_temperature = 18.0

        # Compare 30-minute vs 60-minute timesteps
        demand_30min = utils.calculate_heating_demand(
            specific_heating_demand,
            floor_area,
            outdoor_temps,
            base_temperature,
            optimization_time_step=30,
        )
        demand_60min = utils.calculate_heating_demand(
            specific_heating_demand,
            floor_area,
            outdoor_temps,
            base_temperature,
            optimization_time_step=60,
        )

        # 60-minute timestep should have exactly double the demand of 30-minute
        np.testing.assert_array_almost_equal(demand_60min, demand_30min * 2.0)

    def test_calculate_heating_demand_different_reference_hdd(self):
        """Test heating demand with different annual reference HDD values."""
        specific_heating_demand = 100.0
        floor_area = 150.0
        outdoor_temps = np.array([5.0, 10.0, 15.0])

        # Compare different reference HDD values
        demand_hdd_3000 = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps, annual_reference_hdd=3000.0
        )
        demand_hdd_1500 = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps, annual_reference_hdd=1500.0
        )

        # Half the reference HDD should double the heating demand
        np.testing.assert_array_almost_equal(demand_hdd_1500, demand_hdd_3000 * 2.0)

    def test_calculate_heating_demand_at_base_temperature(self):
        """Test heating demand exactly at base temperature (boundary condition)."""
        specific_heating_demand = 100.0
        floor_area = 150.0
        # Outdoor temp exactly at base temperature
        outdoor_temps = np.array([18.0, 18.0, 18.0])
        base_temperature = 18.0

        heating_demand = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps, base_temperature
        )

        # At base temperature, HDD should be zero, so heating demand should be zero
        self.assertTrue(np.allclose(heating_demand, 0.0))

    def test_calculate_heating_demand_realistic_scenario(self):
        """Test heating demand with realistic winter scenario."""
        # Realistic parameters for Central European home
        specific_heating_demand = 80.0  # kWh/m²/year (modern insulated home)
        floor_area = 120.0  # m² (typical family home)
        # Typical winter week hourly temperatures (°C)
        outdoor_temps = np.array([2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0, 8.0])
        base_temperature = 18.0
        annual_reference_hdd = 2800.0  # Typical for Central Europe
        optimization_time_step = 60  # 1-hour timestep

        heating_demand = utils.calculate_heating_demand(
            specific_heating_demand,
            floor_area,
            outdoor_temps,
            base_temperature,
            annual_reference_hdd,
            optimization_time_step,
        )

        # Verify all values are positive (cold weather)
        self.assertTrue(np.all(heating_demand > 0.0))

        # Verify coldest temperature has highest demand
        coldest_idx = np.argmin(outdoor_temps)
        self.assertEqual(coldest_idx, np.argmax(heating_demand))

        # Verify warmer temperature has lower demand
        warmest_idx = np.argmax(outdoor_temps)
        self.assertEqual(warmest_idx, np.argmin(heating_demand))

    def test_calculate_heating_demand_auto_infer_timestep(self):
        """Test automatic inference of optimization_time_step from pandas Series index."""
        specific_heating_demand = 100.0
        floor_area = 150.0
        outdoor_temps_values = np.array([5.0, 10.0, 15.0, 8.0, 12.0])

        # Create pandas Series with 30-minute DatetimeIndex
        start_date = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        date_range_30min = pd.date_range(
            start=start_date, periods=len(outdoor_temps_values), freq="30min"
        )
        outdoor_temps_30min = pd.Series(outdoor_temps_values, index=date_range_30min)

        # Create pandas Series with 60-minute DatetimeIndex
        date_range_60min = pd.date_range(
            start=start_date, periods=len(outdoor_temps_values), freq="60min"
        )
        outdoor_temps_60min = pd.Series(outdoor_temps_values, index=date_range_60min)

        # Test auto-inference (should infer 30 min from Series)
        demand_auto_30 = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_30min
        )

        # Test explicit 30 min parameter (should match auto-inference)
        demand_explicit_30 = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_30min, optimization_time_step=30
        )

        # Results should be identical
        np.testing.assert_array_almost_equal(demand_auto_30, demand_explicit_30)

        # Test auto-inference with 60-minute frequency
        demand_auto_60 = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_60min
        )

        # Test explicit 60 min parameter
        demand_explicit_60 = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_60min, optimization_time_step=60
        )

        # Results should be identical
        np.testing.assert_array_almost_equal(demand_auto_60, demand_explicit_60)

        # Verify 60-min is double the demand of 30-min (when auto-inferred)
        np.testing.assert_array_almost_equal(demand_auto_60, demand_auto_30 * 2.0)

    def test_calculate_heating_demand_fallback_to_default(self):
        """Test fallback to default 30-minute timestep when not inferrable."""
        specific_heating_demand = 100.0
        floor_area = 150.0
        outdoor_temps_array = np.array([5.0, 10.0, 15.0])

        # Test with numpy array (should fall back to 30 min)
        demand_array = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_array
        )

        # Test explicit 30 min parameter
        demand_explicit = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_array, optimization_time_step=30
        )

        # Results should be identical (both use 30 min)
        np.testing.assert_array_almost_equal(demand_array, demand_explicit)

        # Test with pandas Series without DatetimeIndex (should fall back to 30 min)
        outdoor_temps_series_no_dt = pd.Series(outdoor_temps_array)
        demand_series_no_dt = utils.calculate_heating_demand(
            specific_heating_demand, floor_area, outdoor_temps_series_no_dt
        )

        # Should also match explicit 30 min
        np.testing.assert_array_almost_equal(demand_series_no_dt, demand_explicit)

    def test_calculate_heating_demand_physics_no_solar_basic_monotonic(self):
        """No solar gains: zero demand when outdoor >= indoor, higher demand for colder steps."""
        indoor_temp = 21.0
        # Outdoor temps: some above, some below indoor
        outdoor_temps = np.array([22.0, 21.0, 20.0, 15.0, 10.0, 5.0])
        optimization_time_step = 60  # minutes
        u_value = 0.35  # W/m²K
        envelope_area = 380.0  # m²
        ventilation_rate = 0.4  # ACH
        heated_volume = 240.0  # m³

        demand = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
            solar_irradiance_forecast=None,
            window_area=None,
        )

        # 1) Non-negative demand
        self.assertTrue(np.all(demand >= 0.0), "All heating demands should be non-negative")

        # 2) Zero demand when outdoor >= indoor (first two steps)
        self.assertEqual(demand[0], 0.0, "No heating needed when outdoor (22°C) > indoor (21°C)")
        self.assertEqual(demand[1], 0.0, "No heating needed when outdoor (21°C) = indoor (21°C)")

        # 3) Positive demand when outdoor < indoor
        self.assertGreater(demand[2], 0.0, "Heating needed when outdoor (20°C) < indoor (21°C)")
        self.assertGreater(demand[3], 0.0, "Heating needed when outdoor (15°C) < indoor (21°C)")

        # 4) Colder timesteps yield higher demand (monotonic relationship)
        colder_indices = [2, 3, 4, 5]
        for i in range(len(colder_indices) - 1):
            idx_warmer = colder_indices[i]
            idx_colder = colder_indices[i + 1]
            self.assertGreaterEqual(
                demand[idx_colder],
                demand[idx_warmer],
                msg=f"Demand at colder step {idx_colder} ({outdoor_temps[idx_colder]}°C) "
                f"should be >= step {idx_warmer} ({outdoor_temps[idx_warmer]}°C)",
            )

    def test_calculate_heating_demand_physics_with_solar_gains_reduces_demand(self):
        """Solar gains reduce demand vs. no-solar case, and demand never becomes negative."""
        indoor_temp = 21.0
        outdoor_temps = np.array([0.0, 0.0, 0.0, 0.0])
        optimization_time_step = 60  # minutes
        u_value = 0.35  # W/m²K
        envelope_area = 380.0  # m²
        ventilation_rate = 0.4  # ACH
        heated_volume = 240.0  # m³
        window_area = 28.0  # m²
        shgc = 0.6  # Solar Heat Gain Coefficient

        # Simple GHI profile with some non-zero irradiance
        solar_irradiance = np.array([0.0, 200.0, 400.0, 0.0])  # W/m²

        demand_no_solar = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
            solar_irradiance_forecast=None,
            window_area=None,
        )

        demand_with_solar = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=optimization_time_step,
            solar_irradiance_forecast=solar_irradiance,
            window_area=window_area,
            shgc=shgc,
        )

        # Demand must never be negative
        self.assertTrue(
            np.all(demand_with_solar >= 0.0), "Demand with solar gains should never be negative"
        )

        # With solar gains, demand should not increase at any timestep
        self.assertTrue(
            np.all(demand_with_solar <= demand_no_solar),
            msg=f"Demand with solar gains should be <= no-solar demand at all timesteps.\n"
            f"no_solar={demand_no_solar}, with_solar={demand_with_solar}",
        )

        # For timesteps with non-zero irradiance, some reduction is expected
        self.assertLess(
            np.sum(demand_with_solar[solar_irradiance > 0.0]),
            np.sum(demand_no_solar[solar_irradiance > 0.0]),
            "Solar irradiance should reduce total heating demand during sunny periods",
        )

    def test_calculate_heating_demand_physics_scaling_with_timestep(self):
        """Sanity check: total demand scales appropriately with optimization_time_step."""
        indoor_temp = 21.0
        outdoor_temps = np.array([5.0, 5.0, 5.0, 5.0])  # constant cold
        u_value = 0.35  # W/m²K
        envelope_area = 380.0  # m²
        ventilation_rate = 0.4  # ACH
        heated_volume = 240.0  # m³

        # Case 1: 30-minute timestep
        demand_30min = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=30,
            solar_irradiance_forecast=None,
            window_area=None,
        )

        # Case 2: 60-minute timestep with same temperatures
        demand_60min = utils.calculate_heating_demand_physics(
            u_value=u_value,
            envelope_area=envelope_area,
            ventilation_rate=ventilation_rate,
            heated_volume=heated_volume,
            indoor_target_temperature=indoor_temp,
            outdoor_temperature_forecast=outdoor_temps,
            optimization_time_step=60,
            solar_irradiance_forecast=None,
            window_area=None,
        )

        total_30 = np.sum(demand_30min)
        total_60 = np.sum(demand_60min)

        # For a purely linear time scaling, 60-minute steps should yield about 2× 30-minute steps
        # (depending on implementation details, allow a small numerical tolerance).
        self.assertAlmostEqual(
            total_60,
            2.0 * total_30,
            delta=0.01 * total_60,
            msg=f"60-minute timestep total ({total_60:.3f}) should be ~2x 30-minute total ({total_30:.3f})",
        )

    def test_calculate_cop_heatpump(self):
        """Test heat pump COP calculation utility function with Carnot-based formula."""
        # Test basic calculation with example outdoor temperatures
        supply_temp = 35.0  # °C
        carnot_efficiency = 0.4  # Typical value for real heat pumps (40% of Carnot)
        outdoor_temps = np.array([0.0, 5.0, 10.0, 15.0, 20.0])

        cops = utils.calculate_cop_heatpump(supply_temp, carnot_efficiency, outdoor_temps)

        # Verify output is numpy array
        self.assertIsInstance(cops, np.ndarray)
        # Verify output length matches input length
        self.assertEqual(len(cops), len(outdoor_temps))

        # Manually verify first value using Carnot formula:
        # COP = carnot_efficiency * T_supply_kelvin / (T_supply_kelvin - T_outdoor_kelvin)
        # COP = 0.4 * (35 + 273.15) / |(35 + 273.15) - (0 + 273.15)|
        # COP = 0.4 * 308.15 / 35 = 3.521...
        supply_kelvin = supply_temp + 273.15
        outdoor_kelvin = outdoor_temps[0] + 273.15
        expected_first_cop = carnot_efficiency * supply_kelvin / abs(supply_kelvin - outdoor_kelvin)
        self.assertAlmostEqual(cops[0], expected_first_cop, places=6)

        # Verify all COPs are non-negative
        self.assertTrue(np.all(cops >= 0.0))

        # Test with pandas Series input
        outdoor_temps_series = pd.Series(outdoor_temps)
        cops_from_series = utils.calculate_cop_heatpump(
            supply_temp, carnot_efficiency, outdoor_temps_series
        )
        np.testing.assert_array_almost_equal(cops, cops_from_series)

        # Test that COP decreases as temperature difference increases
        # When outdoor temp gets further from supply temp, COP should decrease
        outdoor_increasing = np.array([30.0, 25.0, 20.0, 15.0, 10.0])  # Getting colder
        cops_decreasing = utils.calculate_cop_heatpump(
            supply_temp, carnot_efficiency, outdoor_increasing
        )
        # Each successive COP should be lower as temp difference increases
        for i in range(len(cops_decreasing) - 1):
            self.assertGreaterEqual(cops_decreasing[i], cops_decreasing[i + 1])

        # Test with different carnot_efficiency values
        carnot_eff_high = 0.5
        cops_high_eff = utils.calculate_cop_heatpump(supply_temp, carnot_eff_high, outdoor_temps)
        # Higher Carnot efficiency should give proportionally higher COPs (subject to 8.0 cap)
        expected_ratio = carnot_eff_high / carnot_efficiency
        expected_cops_uncapped = cops * expected_ratio
        expected_cops_capped = np.minimum(expected_cops_uncapped, 8.0)
        np.testing.assert_array_almost_equal(cops_high_eff, expected_cops_capped)

        # Test realistic scenario: heat pump at 35°C supply, 5°C outdoor
        # COP = 0.4 * 308.15 / |308.15 - 278.15| = 0.4 * 308.15 / 30 = 4.108
        cop_realistic = utils.calculate_cop_heatpump(35.0, 0.4, np.array([5.0]))
        expected_realistic = 0.4 * (35 + 273.15) / abs((35 + 273.15) - (5 + 273.15))
        self.assertAlmostEqual(cop_realistic[0], expected_realistic, places=6)
        # Typical heat pump COP should be in range 2-6 for normal conditions
        self.assertGreater(cop_realistic[0], 2.0)
        self.assertLess(cop_realistic[0], 6.0)

    def test_calculate_cop_heatpump_edge_case_warning(self):
        """Test COP calculation logs warning when outdoor temp >= supply temp."""

        # Test case where outdoor temps exceed or equal supply temp
        supply_temp = 30.0
        carnot_eff = 0.4
        # Mix of normal and problematic outdoor temps
        outdoor_temps = np.array([5.0, 10.0, 30.0, 35.0, 40.0])  # Last 3 >= supply

        # Capture log messages
        with self.assertLogs("emhass.utils", level="WARNING") as log_context:
            cops = utils.calculate_cop_heatpump(supply_temp, carnot_eff, outdoor_temps)

            # Verify warning was logged
            self.assertTrue(
                any(
                    "outdoor temperature >= supply temperature" in msg for msg in log_context.output
                ),
                "Should log warning about non-physical temperature scenario",
            )

        # Verify result is still valid (uses COP=1.0 for non-physical scenarios)
        self.assertIsInstance(cops, np.ndarray)
        self.assertEqual(len(cops), len(outdoor_temps))
        self.assertTrue(np.all(cops >= 1.0), "All COPs should be >= 1.0 (lower bound)")
        self.assertTrue(np.all(cops <= 8.0), "All COPs should be <= 8.0 (upper bound)")
        self.assertTrue(np.all(np.isfinite(cops)), "All COPs should be finite (no inf/nan)")
        # Non-physical scenarios (outdoor >= supply) should get COP=1.0 (direct electric heating)
        # outdoor_temps = [5, 10, 30, 35, 40], supply = 30
        # Valid: cops[0], cops[1]  (5 < 30, 10 < 30)
        # Invalid: cops[2], cops[3], cops[4]  (30 >= 30, 35 > 30, 40 > 30)
        self.assertEqual(cops[2], 1.0, "Boundary case (equal temps) should have COP=1.0")
        self.assertEqual(
            cops[3], 1.0, "Non-physical scenario (outdoor > supply) should have COP=1.0"
        )
        self.assertEqual(
            cops[4], 1.0, "Non-physical scenario (outdoor > supply) should have COP=1.0"
        )
        # Valid scenarios should have reasonable COP > 1.0
        self.assertGreater(cops[0], 1.0, "Valid scenario should have COP > 1.0")
        self.assertGreater(cops[1], 1.0, "Valid scenario should have COP > 1.0")

    def test_calculate_thermal_loss_signed(self):
        """Test thermal loss sign-switching utility function based on Langer & Volling (2020)."""
        # Test basic calculation with temperatures crossing the indoor threshold
        indoor_temp = 20.0
        base_loss = 0.045
        # Outdoor temps: some below indoor (loss), some above indoor (gain)
        outdoor_temps = np.array([10.0, 15.0, 20.0, 25.0, 30.0])

        losses = utils.calculate_thermal_loss_signed(outdoor_temps, indoor_temp, base_loss)

        # Verify output is numpy array
        self.assertIsInstance(losses, np.ndarray)
        # Verify output length matches input length
        self.assertEqual(len(losses), len(outdoor_temps))

        # Verify sign switching based on temperature threshold
        # When outdoor < indoor: Hot(h) = 0, Loss = base_loss * (1 - 2*0) = +base_loss (positive loss)
        # When outdoor >= indoor: Hot(h) = 1, Loss = base_loss * (1 - 2*1) = -base_loss (negative loss/gain)
        self.assertAlmostEqual(losses[0], base_loss, places=6)  # 10°C < 20°C: +loss
        self.assertAlmostEqual(losses[1], base_loss, places=6)  # 15°C < 20°C: +loss
        self.assertAlmostEqual(losses[2], -base_loss, places=6)  # 20°C >= 20°C: -loss (gain)
        self.assertAlmostEqual(losses[3], -base_loss, places=6)  # 25°C >= 20°C: -loss (gain)
        self.assertAlmostEqual(losses[4], -base_loss, places=6)  # 30°C >= 20°C: -loss (gain)

        # Test with pandas Series input
        outdoor_temps_series = pd.Series(outdoor_temps)
        losses_from_series = utils.calculate_thermal_loss_signed(
            outdoor_temps_series, indoor_temp, base_loss
        )
        np.testing.assert_array_almost_equal(losses, losses_from_series)

        # Test winter scenario: all outdoor temps below indoor (all positive losses)
        outdoor_winter = np.array([0.0, 5.0, 10.0, 15.0])
        losses_winter = utils.calculate_thermal_loss_signed(outdoor_winter, indoor_temp, base_loss)
        self.assertTrue(np.all(losses_winter > 0))
        self.assertTrue(np.allclose(losses_winter, base_loss))

        # Test summer scenario: all outdoor temps above indoor (all negative losses)
        outdoor_summer = np.array([25.0, 30.0, 35.0, 40.0])
        losses_summer = utils.calculate_thermal_loss_signed(outdoor_summer, indoor_temp, base_loss)
        self.assertTrue(np.all(losses_summer < 0))
        self.assertTrue(np.allclose(losses_summer, -base_loss))

        # Test with different base_loss value
        base_loss_2 = 0.1
        losses_2 = utils.calculate_thermal_loss_signed(outdoor_temps, indoor_temp, base_loss_2)
        # Verify magnitude is scaled by base_loss
        expected_ratio = base_loss_2 / base_loss
        np.testing.assert_array_almost_equal(losses_2, losses * expected_ratio)

        # Test formula correctness per Langer & Volling (2020) Equation B.13
        # Loss+/- = base_loss * (1 - 2 * Hot(h))
        # Manual verification for outdoor_temp = 18°C (< 20°C indoor)
        loss_manual_cold = base_loss * (1 - 2 * 0)
        self.assertAlmostEqual(loss_manual_cold, base_loss, places=6)

        # Manual verification for outdoor_temp = 22°C (>= 20°C indoor)
        loss_manual_warm = base_loss * (1 - 2 * 1)
        self.assertAlmostEqual(loss_manual_warm, -base_loss, places=6)


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
