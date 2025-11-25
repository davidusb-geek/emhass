#!/usr/bin/env python

import copy
import pathlib
import unittest
from unittest.mock import Mock, patch

import numpy as np
import orjson
import pandas as pd

from emhass import utils
from emhass.command_line import (
    dayahead_forecast_optim,
    export_influxdb_to_csv,
    forecast_model_fit,
    forecast_model_predict,
    forecast_model_tune,
    main,
    naive_mpc_optim,
    perfect_forecast_optim,
    publish_data,
    regressor_model_fit,
    regressor_model_predict,
    set_input_data_dict,
)

# The root folder
root = pathlib.Path(utils.get_root(__file__, num_parent=2))
# Build emhass_conf paths
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["config_path"] = root / "config.json"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

# create logger
logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)


class TestCommandLineAsyncUtils(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def get_test_params(set_use_pv=False):
        # Build params with default config and secrets
        if emhass_conf["defaults_path"].exists():
            config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
            _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
            params = await utils.build_params(emhass_conf, secrets, config, logger)
            if set_use_pv:
                params["optim_conf"]["set_use_pv"] = True
        else:
            raise Exception(
                "config_defaults. does not exist in path: " + str(emhass_conf["defaults_path"])
            )
        return params

    async def asyncSetUp(self):
        params = await TestCommandLineAsyncUtils.get_test_params(set_use_pv=True)
        # Add runtime parameters for forecast lists
        runtimeparams = {
            "pv_power_forecast": [i + 1 for i in range(48)],
            "load_power_forecast": [i + 1 for i in range(48)],
            "load_cost_forecast": [i + 1 for i in range(48)],
            "prod_price_forecast": [i + 1 for i in range(48)],
        }
        self.runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        self.params_json = orjson.dumps(params).decode("utf-8")

    # Test input data for actions (using data from file)
    async def test_set_input_data_dict(self):
        costfun = "profit"
        # Test dayahead
        action = "dayahead-optim"
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            self.params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertIsInstance(input_data_dict, dict)
        self.assertTrue(input_data_dict["df_input_data"] is None)
        self.assertIsInstance(input_data_dict["df_input_data_dayahead"], pd.DataFrame)
        self.assertTrue(input_data_dict["df_input_data_dayahead"].index.freq is not None)
        self.assertTrue(input_data_dict["df_input_data_dayahead"].isnull().sum().sum() == 0)
        self.assertTrue(input_data_dict["fcst"].optim_conf["weather_forecast_method"] == "list")
        self.assertTrue(input_data_dict["fcst"].optim_conf["load_forecast_method"] == "list")
        self.assertTrue(input_data_dict["fcst"].optim_conf["load_cost_forecast_method"] == "list")
        self.assertTrue(
            input_data_dict["fcst"].optim_conf["production_price_forecast_method"] == "list"
        )
        # Test publish data
        action = "publish-data"
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            self.params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertTrue(input_data_dict["df_input_data"] is None)
        self.assertTrue(input_data_dict["df_input_data_dayahead"] is None)
        self.assertTrue(input_data_dict["P_PV_forecast"] is None)
        self.assertTrue(input_data_dict["P_load_forecast"] is None)
        # Test naive mpc
        action = "naive-mpc-optim"
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            self.params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertIsInstance(input_data_dict, dict)
        self.assertIsInstance(input_data_dict["df_input_data_dayahead"], pd.DataFrame)
        self.assertTrue(input_data_dict["df_input_data_dayahead"].index.freq is not None)
        self.assertTrue(input_data_dict["df_input_data_dayahead"].isnull().sum().sum() == 0)
        self.assertTrue(
            len(input_data_dict["df_input_data_dayahead"]) == 10
        )  # The default value for prediction_horizon
        # Test Naive mpc with a shorter forecast =
        runtimeparams = {
            "pv_power_forecast": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "load_power_forecast": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "load_cost_forecast": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "prod_price_forecast": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "prediction_horizon": 10,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params = copy.deepcopy(orjson.loads(self.params_json))
        params["passed_data"] = runtimeparams
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
        self.assertIsInstance(input_data_dict, dict)
        self.assertIsInstance(input_data_dict["df_input_data_dayahead"], pd.DataFrame)
        self.assertTrue(input_data_dict["df_input_data_dayahead"].index.freq is not None)
        self.assertTrue(input_data_dict["df_input_data_dayahead"].isnull().sum().sum() == 0)
        self.assertTrue(
            len(input_data_dict["df_input_data_dayahead"]) == 10
        )  # The default value for prediction_horizon
        # Test naive mpc with a shorter forecast and prediction horizon = 10
        action = "naive-mpc-optim"
        runtimeparams["prediction_horizon"] = 10
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params = copy.deepcopy(orjson.loads(self.params_json))
        params["passed_data"] = runtimeparams
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
        self.assertIsInstance(input_data_dict, dict)
        self.assertIsInstance(input_data_dict["df_input_data_dayahead"], pd.DataFrame)
        self.assertTrue(input_data_dict["df_input_data_dayahead"].index.freq is not None)
        self.assertTrue(input_data_dict["df_input_data_dayahead"].isnull().sum().sum() == 0)
        self.assertTrue(
            len(input_data_dict["df_input_data_dayahead"]) == 10
        )  # The fixed value for prediction_horizon
        # Test passing just load cost and prod price as lists
        action = "dayahead-optim"
        params = await TestCommandLineAsyncUtils.get_test_params()
        runtimeparams = {
            "load_cost_forecast": [i + 1 for i in range(48)],
            "prod_price_forecast": [i + 1 for i in range(48)],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
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
        self.assertTrue(input_data_dict["fcst"].optim_conf["load_cost_forecast_method"] == "list")
        self.assertTrue(
            input_data_dict["fcst"].optim_conf["production_price_forecast_method"] == "list"
        )

    # Test day-ahead optimization
    async def test_webserver_get_injection_dict(self):
        costfun = "profit"
        action = "dayahead-optim"
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            self.params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res = await dayahead_forecast_optim(input_data_dict, logger, debug=True)
        injection_dict = utils.get_injection_dict(opt_res)
        self.assertIsInstance(injection_dict, dict)
        self.assertIsInstance(injection_dict["table1"], str)
        self.assertIsInstance(injection_dict["table2"], str)

    # Test data formatting of dayahead optimization with load cost and prod price as lists
    async def test_dayahead_forecast_optim(self):
        # Test dataframe output of profit dayahead optimization
        costfun = "profit"
        action = "dayahead-optim"
        params = copy.deepcopy(orjson.loads(self.params_json))
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            self.params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res = await dayahead_forecast_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum() == 0)
        self.assertTrue(len(opt_res) == len(params["passed_data"]["pv_power_forecast"]))
        # Test dayahead output, passing just load cost and prod price as runtime lists (costfun=profit)
        action = "dayahead-optim"
        params = await TestCommandLineAsyncUtils.get_test_params()
        runtimeparams = {
            "load_cost_forecast": [i + 1 for i in range(48)],
            "prod_price_forecast": [i + 1 for i in range(48)],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
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
        opt_res = await dayahead_forecast_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum() == 0)
        self.assertTrue(input_data_dict["fcst"].optim_conf["load_cost_forecast_method"] == "list")
        self.assertTrue(
            input_data_dict["fcst"].optim_conf["production_price_forecast_method"] == "list"
        )
        self.assertEqual(
            opt_res["unit_load_cost"].values.tolist(),
            runtimeparams["load_cost_forecast"],
        )
        self.assertEqual(
            opt_res["unit_prod_price"].values.tolist(),
            runtimeparams["prod_price_forecast"],
        )
        # Test dayahead output, using set_use_adjusted_pv = True
        params = await TestCommandLineAsyncUtils.get_test_params()
        params["optim_conf"]["set_use_adjusted_pv"] = True
        params["optim_conf"]["set_use_pv"] = True
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res = await dayahead_forecast_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum() == 0)

    # Test dataframe output of perfect forecast optimization
    async def test_perfect_forecast_optim(self):
        costfun = "profit"
        action = "perfect-optim"
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            self.params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res = await perfect_forecast_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum() == 0)
        self.assertIsInstance(opt_res.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(opt_res.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertTrue("cost_fun_" + input_data_dict["costfun"] in opt_res.columns)

    # Test naive mpc optimization
    async def test_naive_mpc_optim(self):
        # Test mpc optimization
        costfun = "profit"
        action = "naive-mpc-optim"
        params = copy.deepcopy(orjson.loads(self.params_json))
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            self.params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res = await naive_mpc_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum() == 0)
        self.assertTrue(len(opt_res) == 10)
        # Test mpc optimization with runtime parameters similar to the documentation
        runtimeparams = {
            "pv_power_forecast": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "prediction_horizon": 10,
            "soc_init": 0.5,
            "soc_final": 0.6,
            "operating_hours_of_each_deferrable_load": [1, 3],
            "start_timesteps_of_each_deferrable_load": [-3, 0],
            "end_timesteps_of_each_deferrable_load": [8, 0],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params["optim_conf"]["weather_forecast_method"] = "list"
        params["optim_conf"]["load_forecast_method"] = "naive"
        params["optim_conf"]["load_cost_forecast_method"] = "hp_hc_periods"
        params["optim_conf"]["production_price_forecast_method"] = "constant"
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
        opt_res = await naive_mpc_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum() == 0)
        self.assertTrue(len(opt_res) == 10)
        # Test publish after passing the forecast as list
        # with method_ts_round=first
        costfun = "profit"
        action = "naive-mpc-optim"
        params = copy.deepcopy(orjson.loads(self.params_json))
        params["retrieve_hass_conf"]["method_ts_round"] = "first"
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res = await naive_mpc_optim(input_data_dict, logger, debug=True)
        action = "publish-data"
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            None,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res_first = await publish_data(input_data_dict, logger, opt_res_latest=opt_res)
        self.assertTrue(len(opt_res_first) == 1)
        # test mpc and publish with method_ts_round=last and set_use_battery=true
        action = "naive-mpc-optim"
        params = copy.deepcopy(orjson.loads(self.params_json))
        params["retrieve_hass_conf"]["method_ts_round"] = "last"
        params["optim_conf"]["set_use_battery"] = True
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res = await naive_mpc_optim(input_data_dict, logger, debug=True)
        action = "publish-data"
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            None,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res_last = await publish_data(input_data_dict, logger, opt_res_latest=opt_res)
        self.assertTrue(len(opt_res_last) == 1)
        # Reproduce when trying to publish data params=None and runtimeparams=None
        # action = 'publish-data'
        # input_data_dict = await set_input_data_dict(emhass_conf, costfun, None, None,
        #                                       action, logger, get_data_from_file=True)
        # opt_res_last = await publish_data(input_data_dict, logger, opt_res_latest=opt_res)
        # self.assertTrue(len(opt_res_last)==1)
        # Check if status is published
        from datetime import datetime

        now_precise = datetime.now(input_data_dict["retrieve_hass_conf"]["time_zone"]).replace(
            second=0, microsecond=0
        )
        idx_closest = opt_res.index.get_indexer([now_precise], method="nearest")[0]
        custom_cost_fun_id = {
            "entity_id": "sensor.optim_status",
            "unit_of_measurement": "",
            "friendly_name": "EMHASS optimization status",
        }
        publish_prefix = ""
        response, data = await input_data_dict["rh"].post_data(
            opt_res["optim_status"],
            idx_closest,
            custom_cost_fun_id["entity_id"],
            "",
            custom_cost_fun_id["unit_of_measurement"],
            custom_cost_fun_id["friendly_name"],
            type_var="optim_status",
            publish_prefix=publish_prefix,
        )
        self.assertTrue(hasattr(response, "__class__"))
        self.assertTrue(data["attributes"]["friendly_name"] == "EMHASS optimization status")
        # When using set_use_adjusted_pv = True
        action = "naive-mpc-optim"
        params = copy.deepcopy(orjson.loads(self.params_json))
        params["optim_conf"]["set_use_adjusted_pv"] = True
        params["optim_conf"]["set_use_pv"] = True
        params_json = orjson.dumps(params).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        opt_res = await naive_mpc_optim(input_data_dict, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum() == 0)
        self.assertTrue(len(opt_res) == 10)

    # Test outputs of fit, predict and tune
    async def test_forecast_model_fit_predict_tune(self):
        costfun = "profit"
        action = "forecast-model-fit"  # fit, predict and tune methods
        params = await TestCommandLineAsyncUtils.get_test_params()
        runtimeparams = {
            "historic_days_to_retrieve": 20,
            "model_type": "long_train_data",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48,
            "split_date_delta": "48h",
            "perform_backtest": False,
            "model_predict_publish": True,
            "model_predict_entity_id": "sensor.p_load_forecast_knn",
            "model_predict_unit_of_measurement": "W",
            "model_predict_friendly_name": "Load Power Forecast KNN regressor",
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["optim_conf"]["load_forecast_method"] = "skforecast"
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
        self.assertTrue(input_data_dict["params"]["passed_data"]["model_type"] == "long_train_data")
        self.assertTrue(
            input_data_dict["params"]["passed_data"]["sklearn_model"] == "KNeighborsRegressor"
        )
        self.assertTrue(input_data_dict["params"]["passed_data"]["perform_backtest"] is False)
        # Check that the default params are loaded
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            self.params_json,
            self.runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertTrue(input_data_dict["params"]["passed_data"]["model_type"] == "long_train_data")
        self.assertTrue(
            input_data_dict["params"]["passed_data"]["sklearn_model"] == "KNeighborsRegressor"
        )
        self.assertIsInstance(input_data_dict["df_input_data"], pd.DataFrame)
        # Test the fit method
        df_fit_pred, df_fit_pred_backtest, mlf = await forecast_model_fit(
            input_data_dict, logger, debug=True
        )
        self.assertIsInstance(df_fit_pred, pd.DataFrame)
        self.assertTrue(df_fit_pred_backtest is None)
        # Test ijection_dict for fit method on webui
        injection_dict = utils.get_injection_dict_forecast_model_fit(df_fit_pred, mlf)
        self.assertIsInstance(injection_dict, dict)
        self.assertIsInstance(injection_dict["figure_0"], str)
        # Test the predict method on observations following the train period
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        df_pred = await forecast_model_predict(
            input_data_dict, logger, use_last_window=False, debug=True, mlf=mlf
        )
        self.assertIsInstance(df_pred, pd.Series)
        self.assertTrue(df_pred.isnull().sum().sum() == 0)
        # Now a predict using last_window
        df_pred = await forecast_model_predict(input_data_dict, logger, debug=True, mlf=mlf)
        self.assertIsInstance(df_pred, pd.Series)
        self.assertTrue(df_pred.isnull().sum().sum() == 0)
        # Test the tune method
        df_pred_optim, mlf = await forecast_model_tune(input_data_dict, logger, debug=True, mlf=mlf)
        self.assertIsInstance(df_pred_optim, pd.DataFrame)
        self.assertTrue(mlf.is_tuned is True)
        # Test injection_dict for tune method on webui
        injection_dict = utils.get_injection_dict_forecast_model_tune(df_fit_pred, mlf)
        self.assertIsInstance(injection_dict, dict)
        self.assertIsInstance(injection_dict["figure_0"], str)

    # Test data formatting of regressor model fit amd predict
    async def test_regressor_model_fit_predict(self):
        costfun = "profit"
        action = "regressor-model-fit"  # fit and predict methods
        params = await TestCommandLineAsyncUtils.get_test_params()
        runtimeparams = {
            "csv_file": "heating_prediction.csv",
            "features": ["degreeday", "solar"],
            "target": "hour",
            "regression_model": "LassoRegression",
            "model_type": "heating_hours_degreeday",
            "timestamp": "timestamp",
            "date_features": ["month", "day_of_week"],
            "mlr_predict_entity_id": "sensor.predicted_hours_test",
            "mlr_predict_unit_of_measurement": "h",
            "mlr_predict_friendly_name": "Predicted hours",
            "new_values": [12.79, 4.766, 1, 2],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
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
        self.assertTrue(
            input_data_dict["params"]["passed_data"]["model_type"] == "heating_hours_degreeday",
        )
        self.assertTrue(
            input_data_dict["params"]["passed_data"]["regression_model"] == "LassoRegression",
        )
        self.assertTrue(
            input_data_dict["params"]["passed_data"]["csv_file"] == "heating_prediction.csv",
        )
        mlr = await regressor_model_fit(input_data_dict, logger, debug=True)

        # def test_regressor_model_predict(self):
        costfun = "profit"
        action = "regressor-model-predict"  # predict methods
        params = await TestCommandLineAsyncUtils.get_test_params()
        runtimeparams = {
            "csv_file": "heating_prediction.csv",
            "features": ["degreeday", "solar"],
            "target": "hour",
            "regression_model": "LassoRegression",
            "model_type": "heating_hours_degreeday",
            "timestamp": "timestamp",
            "date_features": ["month", "day_of_week"],
            "mlr_predict_entity_id": "sensor.predicted_hours_test",
            "mlr_predict_unit_of_measurement": "h",
            "mlr_predict_friendly_name": "Predicted hours",
            "new_values": [12.79, 4.766, 1, 2],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
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
        self.assertTrue(
            input_data_dict["params"]["passed_data"]["model_type"] == "heating_hours_degreeday",
        )
        self.assertTrue(
            input_data_dict["params"]["passed_data"]["mlr_predict_friendly_name"]
            == "Predicted hours",
        )

        await regressor_model_predict(input_data_dict, logger, debug=True, mlr=mlr)

    # CLI test action that does not exist
    async def test_main_wrong_action(self):
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "test",
                "--config",
                str(emhass_conf["config_path"]),
                "--debug",
                "True",
            ],
        ):
            opt_res = await main()
            self.assertEqual(opt_res, None)

    # CLI test action perfect-optim action
    async def test_main_perfect_forecast_optim(self):
        test_params = await TestCommandLineAsyncUtils.get_test_params(set_use_pv=True)
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "perfect-optim",
                "--config",
                str(emhass_conf["config_path"]),
                "--debug",
                "True",
                "--params",
                orjson.dumps(test_params).decode("utf-8"),
            ],
        ):
            opt_res = await main()
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum() == 0)
        self.assertIsInstance(opt_res.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(
            opt_res.index.dtype,
            pd.core.dtypes.dtypes.DatetimeTZDtype,
        )

    # CLI test dayahead forecast optimzation action
    async def test_main_dayahead_forecast_optim(self):
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "dayahead-optim",
                "--config",
                str(emhass_conf["config_path"]),
                "--params",
                self.params_json,
                "--runtimeparams",
                self.runtimeparams_json,
                "--debug",
                "True",
            ],
        ):
            opt_res = await main()
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum() == 0)

    # CLI test naive mpc optimzation action
    async def test_main_naive_mpc_optim(self):
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "naive-mpc-optim",
                "--config",
                str(emhass_conf["config_path"]),
                "--params",
                self.params_json,
                "--runtimeparams",
                self.runtimeparams_json,
                "--debug",
                "True",
            ],
        ):
            opt_res = await main()
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertTrue(opt_res.isnull().sum().sum() == 0)
        self.assertTrue(len(opt_res) == 10)

    # CLI test forecast model fit action
    async def test_main_forecast_model_fit(self):
        params = copy.deepcopy(orjson.loads(self.params_json))
        runtimeparams = {
            "historic_days_to_retrieve": 20,
            "model_type": "long_train_data",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48,
            "split_date_delta": "48h",
            "perform_backtest": False,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params["optim_conf"]["load_forecast_method"] = "skforecast"
        params_json = orjson.dumps(params).decode("utf-8")
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "forecast-model-fit",
                "--config",
                str(emhass_conf["config_path"]),
                "--params",
                params_json,
                "--runtimeparams",
                runtimeparams_json,
                "--debug",
                "True",
            ],
        ):
            df_fit_pred, df_fit_pred_backtest, mlf = await main()
        self.assertIsInstance(df_fit_pred, pd.DataFrame)
        self.assertTrue(df_fit_pred_backtest is None)

    # CLI test forecast model predict action
    async def test_main_forecast_model_predict(self):
        params = copy.deepcopy(orjson.loads(self.params_json))
        runtimeparams = {
            "historic_days_to_retrieve": 20,
            "model_type": "long_train_data",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48,
            "split_date_delta": "48h",
            "perform_backtest": False,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params["optim_conf"]["load_forecast_method"] = "skforecast"
        params_json = orjson.dumps(params).decode("utf-8")
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "forecast-model-predict",
                "--config",
                str(emhass_conf["config_path"]),
                "--params",
                params_json,
                "--runtimeparams",
                runtimeparams_json,
                "--debug",
                "True",
            ],
        ):
            df_pred = await main()
        self.assertIsInstance(df_pred, pd.Series)
        self.assertTrue(df_pred.isnull().sum().sum() == 0)

    # CLI test forecast model tune action
    async def test_main_forecast_model_tune(self):
        params = copy.deepcopy(orjson.loads(self.params_json))
        runtimeparams = {
            "historic_days_to_retrieve": 20,
            "model_type": "long_train_data",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48,
            "split_date_delta": "48h",
            "perform_backtest": False,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        # params = await TestCommandLineAsyncUtils.get_test_params()
        params["passed_data"] = runtimeparams
        params["optim_conf"]["load_forecast_method"] = "skforecast"
        params_json = orjson.dumps(params).decode("utf-8")
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "forecast-model-tune",
                "--config",
                str(emhass_conf["config_path"]),
                "--params",
                params_json,
                "--runtimeparams",
                runtimeparams_json,
                "--debug",
                "True",
            ],
        ):
            df_pred_optim, mlf = await main()
        self.assertIsInstance(df_pred_optim, pd.DataFrame)
        self.assertTrue(mlf.is_tuned is True)

    # CLI test regressor model fit action
    async def test_main_regressor_model_fit(self):
        params = copy.deepcopy(orjson.loads(self.params_json))
        runtimeparams = {
            "csv_file": "heating_prediction.csv",
            "features": ["degreeday", "solar"],
            "target": "hour",
            "regression_model": "LassoRegression",
            "model_type": "heating_hours_degreeday",
            "timestamp": "timestamp",
            "date_features": ["month", "day_of_week"],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "regressor-model-fit",
                "--config",
                str(emhass_conf["config_path"]),
                "--params",
                params_json,
                "--runtimeparams",
                runtimeparams_json,
                "--debug",
                "True",
            ],
        ):
            await main()

    # CLI test regressor model predict action
    async def test_main_regressor_model_predict(self):
        params = copy.deepcopy(orjson.loads(self.params_json))
        runtimeparams = {
            "csv_file": "heating_prediction.csv",
            "features": ["degreeday", "solar"],
            "target": "hour",
            "regression_model": "LassoRegression",
            "model_type": "heating_hours_degreeday",
            "timestamp": "timestamp",
            "date_features": ["month", "day_of_week"],
            "new_values": [12.79, 4.766, 1, 2],
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params["optim_conf"]["load_forecast_method"] = "skforecast"
        params_json = orjson.dumps(params).decode("utf-8")
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "regressor-model-predict",
                "--config",
                str(emhass_conf["config_path"]),
                "--params",
                params_json,
                "--runtimeparams",
                runtimeparams_json,
                "--debug",
                "True",
            ],
        ):
            prediction = await main()
        self.assertIsInstance(prediction, np.ndarray)

    # CLI test publish data action
    async def test_main_publish_data(self):
        with patch(
            "sys.argv",
            [
                "main",
                "--action",
                "publish-data",
                "--config",
                str(emhass_conf["config_path"]),
                "--debug",
                "True",
            ],
        ):
            opt_res = await main()
            self.assertFalse(opt_res.empty)

    # Test export_influxdb_to_csv
    async def test_export_influxdb_to_csv(self):
        costfun = "profit"
        action = "export-influxdb-to-csv"

        # Test Success Case
        params = copy.deepcopy(orjson.loads(self.params_json))
        runtimeparams = {
            "sensor_list": [
                "sensor.power_load_no_var_loads",
                "sensor.power_photovoltaics",
            ],
            "csv_filename": "test_export.csv",
            "start_time": "2025-11-10",
            "end_time": "2025-11-11",
            "resample_freq": "30min",
            "handle_nan": "interpolate",
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")

        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,  # Use True to avoid HA calls
        )

        # Mock rh.use_influxdb
        input_data_dict["rh"].use_influxdb = True

        # Create mock data
        index = pd.date_range(
            start="2025-11-10",
            end="2025-11-12",
            freq="10min",
            tz=input_data_dict["rh"].time_zone,
        )
        data = {
            "sensor.power_load_no_var_loads": np.random.rand(len(index)) * 1000,
            "sensor.power_photovoltaics": np.random.rand(len(index)) * 5000,
        }
        df_final_mock = pd.DataFrame(data, index=index)
        # Add some NaNs to test handle_nan
        df_final_mock.iloc[5:10, 0] = np.nan

        # Mock rh.get_data
        input_data_dict["rh"].get_data = Mock(return_value=True)
        input_data_dict["rh"].df_final = df_final_mock

        # Mock the final to_csv call to avoid writing a file
        with patch("pandas.DataFrame.to_csv") as mock_to_csv:
            success = await export_influxdb_to_csv(input_data_dict, logger)
            self.assertTrue(success)
            # Check if to_csv was called
            mock_to_csv.assert_called_once()
            # Check call args
            args, kwargs = mock_to_csv.call_args
            self.assertEqual(kwargs["index"], False)
            self.assertIsInstance(args[0], pathlib.Path)
            self.assertEqual(args[0].name, "test_export.csv")

        # Test InfluxDB Disabled
        input_data_dict["rh"].use_influxdb = False
        success = await export_influxdb_to_csv(input_data_dict, logger)
        self.assertFalse(success)

        # Test Missing Params (e.g., sensor_list)
        params_no_sensors = copy.deepcopy(orjson.loads(self.params_json))
        runtimeparams_no_sensors = {
            "csv_filename": "test_export.csv",
            "start_time": "2025-11-10",
        }
        runtimeparams_no_sensors_json = orjson.dumps(runtimeparams_no_sensors).decode("utf-8")
        params_no_sensors["passed_data"] = runtimeparams_no_sensors
        params_no_sensors_json = orjson.dumps(params_no_sensors).decode("utf-8")

        input_data_dict_no_sensors = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_no_sensors_json,
            runtimeparams_no_sensors_json,
            action,
            logger,
            get_data_from_file=True,
        )
        input_data_dict_no_sensors["rh"].use_influxdb = True
        # This should fail inside export_influxdb_to_csv due to missing 'sensor_list'
        success = await export_influxdb_to_csv(input_data_dict_no_sensors, logger)
        self.assertFalse(success)

        # Test rh.get_data fails
        input_data_dict["rh"].use_influxdb = True  # Reset from test 2
        input_data_dict["rh"].get_data = Mock(return_value=False)  # Mock get_data to fail
        input_data_dict["rh"].df_final = None

        success = await export_influxdb_to_csv(input_data_dict, logger)
        self.assertFalse(success)


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
