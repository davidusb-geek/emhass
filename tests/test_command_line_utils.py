#!/usr/bin/env python

import copy
import pathlib
import pickle
import unittest
from unittest.mock import Mock, patch

import numpy as np
import orjson
import pandas as pd

from emhass import utils
from emhass.command_line import (
    adjust_pv_forecast,
    dayahead_forecast_optim,
    export_influxdb_to_csv,
    forecast_model_fit,
    forecast_model_predict,
    forecast_model_tune,
    is_model_outdated,
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
        self.assertIs(input_data_dict["df_input_data"], None)
        self.assertIsInstance(input_data_dict["df_input_data_dayahead"], pd.DataFrame)
        self.assertIsNot(input_data_dict["df_input_data_dayahead"].index.freq, None)
        self.assertEqual(input_data_dict["df_input_data_dayahead"].isnull().sum().sum(), 0)
        self.assertEqual(input_data_dict["fcst"].optim_conf["weather_forecast_method"], "list")
        self.assertEqual(input_data_dict["fcst"].optim_conf["load_forecast_method"], "list")
        self.assertEqual(input_data_dict["fcst"].optim_conf["load_cost_forecast_method"], "list")
        self.assertEqual(
            input_data_dict["fcst"].optim_conf["production_price_forecast_method"], "list"
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
        self.assertIs(input_data_dict["df_input_data"], None)
        self.assertIs(input_data_dict["df_input_data_dayahead"], None)
        self.assertIs(input_data_dict["p_pv_forecast"], None)
        self.assertIs(input_data_dict["p_load_forecast"], None)
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
        self.assertIsNot(input_data_dict["df_input_data_dayahead"].index.freq, None)
        self.assertEqual(input_data_dict["df_input_data_dayahead"].isnull().sum().sum(), 0)
        self.assertEqual(
            len(input_data_dict["df_input_data_dayahead"]), 10
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
        self.assertIsNot(input_data_dict["df_input_data_dayahead"].index.freq, None)
        self.assertEqual(input_data_dict["df_input_data_dayahead"].isnull().sum().sum(), 0)
        self.assertEqual(
            len(input_data_dict["df_input_data_dayahead"]), 10
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
        self.assertIsNot(input_data_dict["df_input_data_dayahead"].index.freq, None)
        self.assertEqual(input_data_dict["df_input_data_dayahead"].isnull().sum().sum(), 0)
        self.assertEqual(
            len(input_data_dict["df_input_data_dayahead"]), 10
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
        self.assertEqual(input_data_dict["fcst"].optim_conf["load_cost_forecast_method"], "list")
        self.assertEqual(
            input_data_dict["fcst"].optim_conf["production_price_forecast_method"], "list"
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
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
        self.assertEqual(len(opt_res), len(params["passed_data"]["pv_power_forecast"]))
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
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
        self.assertEqual(input_data_dict["fcst"].optim_conf["load_cost_forecast_method"], "list")
        self.assertEqual(
            input_data_dict["fcst"].optim_conf["production_price_forecast_method"], "list"
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
        self.assertEqual(opt_res.isnull().sum().sum(), 0)

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
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
        self.assertIsInstance(opt_res.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(opt_res.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertIn("cost_fun_" + input_data_dict["costfun"], opt_res.columns)

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
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
        self.assertEqual(len(opt_res), 10)
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
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
        self.assertEqual(len(opt_res), 10)
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
        self.assertEqual(len(opt_res_first), 1)
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
        self.assertEqual(len(opt_res_last), 1)

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
        self.assertEqual(data["attributes"]["friendly_name"], "EMHASS optimization status")
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
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
        self.assertEqual(len(opt_res), 10)

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
        # Test with explicit runtime parameters
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        self.assertEqual(input_data_dict["params"]["passed_data"]["model_type"], "long_train_data")
        self.assertEqual(
            input_data_dict["params"]["passed_data"]["sklearn_model"], "KNeighborsRegressor"
        )
        self.assertIs(input_data_dict["params"]["passed_data"]["perform_backtest"], False)
        # Check that the default params are loaded
        default_file_path = emhass_conf["data_path"] / "load_forecast.pkl"
        created_dummy = False
        if not default_file_path.exists():
            # Create a dummy file to satisfy the initial load check in set_input_data_dict
            dummy_data = (pd.DataFrame(), None, None, None)
            with default_file_path.open("wb") as f:
                pickle.dump(dummy_data, f)
            created_dummy = True
        try:
            input_data_dict = await set_input_data_dict(
                emhass_conf,
                costfun,
                self.params_json,
                self.runtimeparams_json,
                action,
                logger,
                get_data_from_file=True,
            )
        finally:
            if created_dummy and default_file_path.exists():
                default_file_path.unlink()
        self.assertEqual(input_data_dict["params"]["passed_data"]["model_type"], "load_forecast")
        self.assertEqual(
            input_data_dict["params"]["passed_data"]["sklearn_model"], "KNeighborsRegressor"
        )
        self.assertIsInstance(input_data_dict["df_input_data"], pd.DataFrame)
        # Inject Fresh Data for Training
        idx = pd.date_range(end=pd.Timestamp.now(), periods=48 * 10, freq="30min")
        data = np.random.rand(len(idx)) * 100  # Random load data
        df_fresh = pd.DataFrame({"sensor.power_load_no_var_loads": data}, index=idx)
        df_fresh = utils.set_df_index_freq(df_fresh)  # Ensure freq is set
        input_data_dict["df_input_data"] = df_fresh
        # Test the fit method
        df_fit_pred, df_fit_pred_backtest, mlf = await forecast_model_fit(
            input_data_dict, logger, debug=True
        )
        self.assertIsInstance(df_fit_pred, pd.DataFrame)
        self.assertIs(df_fit_pred_backtest, None)
        # Test injection_dict for fit method on webui
        injection_dict = utils.get_injection_dict_forecast_model_fit(df_fit_pred, mlf)
        self.assertIsInstance(injection_dict, dict)
        self.assertIsInstance(injection_dict["figure_0"], str)
        # Test the predict method on observations following the train period
        # We need to re-inject the fresh data because set_input_data_dict is called again
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        input_data_dict["df_input_data"] = df_fresh  # Inject again for consistency
        df_pred = await forecast_model_predict(
            input_data_dict, logger, use_last_window=False, debug=True, mlf=mlf
        )
        self.assertIsInstance(df_pred, pd.Series)
        self.assertEqual(df_pred.isnull().sum().sum(), 0)
        # Now a predict using last_window
        df_pred = await forecast_model_predict(input_data_dict, logger, debug=True, mlf=mlf)
        self.assertIsInstance(df_pred, pd.Series)
        self.assertEqual(df_pred.isnull().sum().sum(), 0)
        # Test the tune method
        df_pred_optim, mlf = await forecast_model_tune(input_data_dict, logger, debug=True, mlf=mlf)
        self.assertIsInstance(df_pred_optim, pd.DataFrame)
        self.assertIs(mlf.is_tuned, True)
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
        self.assertEqual(
            input_data_dict["params"]["passed_data"]["model_type"],
            "heating_hours_degreeday",
        )
        self.assertEqual(
            input_data_dict["params"]["passed_data"]["regression_model"],
            "LassoRegression",
        )
        self.assertEqual(
            input_data_dict["params"]["passed_data"]["csv_file"],
            "heating_prediction.csv",
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
        self.assertEqual(
            input_data_dict["params"]["passed_data"]["model_type"],
            "heating_hours_degreeday",
        )
        self.assertEqual(
            input_data_dict["params"]["passed_data"]["mlr_predict_friendly_name"],
            "Predicted hours",
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
            self.assertIsNone(opt_res)

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
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
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
        self.assertEqual(opt_res.isnull().sum().sum(), 0)

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
        self.assertEqual(opt_res.isnull().sum().sum(), 0)
        self.assertEqual(len(opt_res), 10)

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
            df_fit_pred, df_fit_pred_backtest, _ = await main()
        self.assertIsInstance(df_fit_pred, pd.DataFrame)
        self.assertIs(df_fit_pred_backtest, None)

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
        self.assertEqual(df_pred.isnull().sum().sum(), 0)

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
        self.assertIs(mlf.is_tuned, True)

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
            self.assertFalse(kwargs["index"], False)
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

    # Test that runtime costfun parameter overrides config costfun parameter
    async def test_costfun_runtime_override(self):
        """Test that runtime costfun parameter correctly overrides config costfun parameter."""
        # Build params with default config
        params = await TestCommandLineAsyncUtils.get_test_params(set_use_pv=True)

        # Set costfun in config to 'profit'
        params["optim_conf"]["costfun"] = "profit"

        # Add runtime parameters with costfun override
        runtimeparams = {
            "pv_power_forecast": [i + 1 for i in range(48)],
            "load_power_forecast": [i + 1 for i in range(48)],
            "load_cost_forecast": [i + 1 for i in range(48)],
            "prod_price_forecast": [i + 1 for i in range(48)],
            "costfun": "cost",  # Override to 'cost'
        }

        params_json = orjson.dumps(params).decode("utf-8")
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")

        # The costfun passed to set_input_data_dict is from the config (before runtime params)
        costfun_from_config = "profit"
        action = "dayahead-optim"

        # Call set_input_data_dict
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun_from_config,  # This is 'profit' from config
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )

        # Check that the costfun in input_data_dict is the runtime parameter value ('cost')
        self.assertEqual(
            input_data_dict["costfun"],
            "cost",
            "Runtime parameter 'costfun' should override config parameter",
        )

        # Also test with 'self-consumption' as another option
        runtimeparams["costfun"] = "self-consumption"
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")

        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun_from_config,  # Still 'profit' from config
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )

        self.assertEqual(
            input_data_dict["costfun"],
            "self-consumption",
            "Runtime parameter 'costfun' should override config parameter for self-consumption",
        )

        # Also test when costfun is NOT provided as runtime parameter
        runtimeparams_no_costfun = {
            "pv_power_forecast": [i + 1 for i in range(48)],
            "load_power_forecast": [i + 1 for i in range(48)],
            "load_cost_forecast": [i + 1 for i in range(48)],
            "prod_price_forecast": [i + 1 for i in range(48)],
            # No costfun parameter
        }
        runtimeparams_no_costfun_json = orjson.dumps(runtimeparams_no_costfun).decode("utf-8")

        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun_from_config,  # 'profit' from config
            params_json,
            runtimeparams_no_costfun_json,
            action,
            logger,
            get_data_from_file=True,
        )

        # When no runtime costfun is provided, should use config value
        self.assertEqual(
            input_data_dict["costfun"],
            "profit",
            "Should use config parameter when runtime parameter is not provided",
        )

    def test_is_model_outdated(self):
        """Test the is_model_outdated function for various scenarios."""
        import os
        import tempfile
        from datetime import datetime, timedelta

        # Test 1: Non-existent file should return True
        with tempfile.TemporaryDirectory() as tmpdir:
            non_existent_path = pathlib.Path(tmpdir) / "nonexistent_model.pkl"
            result = is_model_outdated(non_existent_path, 24, logger)
            self.assertTrue(result, "Should return True for non-existent file")

        # Test 2: max_age_hours = 0 should force refit (return True)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            result = is_model_outdated(tmp_path, 0, logger)
            self.assertTrue(result, "Should return True when max_age_hours = 0")
        finally:
            tmp_path.unlink()

        # Test 3: Fresh model (just created) should return False
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            result = is_model_outdated(tmp_path, 24, logger)
            self.assertFalse(result, "Should return False for fresh model")
        finally:
            tmp_path.unlink()

        # Test 4: Old model (simulated old modification time) should return True
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
            # Set modification time to 48 hours ago
            old_time = (datetime.now() - timedelta(hours=48)).timestamp()
            os.utime(tmp_path, (old_time, old_time))
        try:
            result = is_model_outdated(tmp_path, 24, logger)
            self.assertTrue(result, "Should return True for model older than max_age")
        finally:
            tmp_path.unlink()

        # Test 5: Model just under the threshold should return False
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
            # Set modification time to 23 hours ago (just under 24h threshold)
            recent_time = (datetime.now() - timedelta(hours=23)).timestamp()
            os.utime(tmp_path, (recent_time, recent_time))
        try:
            result = is_model_outdated(tmp_path, 24, logger)
            self.assertFalse(result, "Should return False for model just under max_age threshold")
        finally:
            tmp_path.unlink()

    async def test_adjusted_pv_model_max_age_runtime_override(self):
        """Test that runtime adjusted_pv_model_max_age parameter overrides config parameter."""
        # Build params with default config
        params = await TestCommandLineAsyncUtils.get_test_params(set_use_pv=True)

        # Set adjusted_pv_model_max_age in config to 24
        params["optim_conf"]["adjusted_pv_model_max_age"] = 24

        # Add runtime parameters with adjusted_pv_model_max_age override
        runtimeparams = {
            "pv_power_forecast": [i + 1 for i in range(48)],
            "load_power_forecast": [i + 1 for i in range(48)],
            "adjusted_pv_model_max_age": 6,  # Override to 6 hours
        }

        params_json = orjson.dumps(params).decode("utf-8")
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")

        costfun = "profit"
        action = "dayahead-optim"

        # Call set_input_data_dict
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )

        # Check that adjusted_pv_model_max_age was overridden in the forecast object
        self.assertEqual(
            input_data_dict["fcst"].optim_conf["adjusted_pv_model_max_age"],
            6,
            "Runtime parameter 'adjusted_pv_model_max_age' should override config parameter",
        )

        # Test with different value
        runtimeparams["adjusted_pv_model_max_age"] = 0  # Force refit
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")

        input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )

        self.assertEqual(
            input_data_dict["fcst"].optim_conf["adjusted_pv_model_max_age"],
            0,
            "Runtime parameter should override with value 0 (force refit)",
        )

    async def test_adjust_pv_forecast_corrupted_model_recovery(self):
        """Test that adjust_pv_forecast gracefully handles corrupted model files."""
        import tempfile
        from unittest.mock import AsyncMock, MagicMock, patch

        import aiofiles

        from emhass.command_line import adjust_pv_forecast
        from emhass.forecast import Forecast

        # Create a corrupted pickle file using tempfile for the path, then write async
        tmp_fd, tmp_name = tempfile.mkstemp(suffix=".pkl")
        import os

        os.close(tmp_fd)  # Close the file descriptor immediately
        tmp_path = pathlib.Path(tmp_name)

        # Write corrupted data asynchronously
        async with aiofiles.open(tmp_path, "wb") as tmp:
            await tmp.write(b"This is not a valid pickle file!")

        try:
            # Setup mock objects
            fcst = MagicMock(spec=Forecast)
            p_pv_forecast = pd.Series([100, 200, 300], name="P_PV")

            test_emhass_conf = {
                "data_path": tmp_path.parent,
            }

            test_optim_conf = {
                "adjusted_pv_model_max_age": 24,
                "adjusted_pv_regression_model": "LassoRegression",
            }

            test_retrieve_hass_conf = {}
            rh = MagicMock()

            # Rename temp file to expected model name
            model_path = tmp_path.parent / "adjust_pv_regressor.pkl"
            tmp_path.rename(model_path)

            # Mock the data retrieval and fit methods
            with patch("emhass.command_line.retrieve_home_assistant_data") as mock_retrieve:
                mock_retrieve.return_value = (True, pd.DataFrame(), None)
                fcst.adjust_pv_forecast_data_prep = MagicMock()
                fcst.adjust_pv_forecast_fit = AsyncMock()
                fcst.adjust_pv_forecast_predict = MagicMock(
                    return_value=pd.DataFrame({"adjusted_forecast": [100, 200, 300]})
                )

                # Call adjust_pv_forecast - should handle corruption and re-fit
                result = await adjust_pv_forecast(
                    logger,
                    fcst,
                    p_pv_forecast,
                    True,
                    test_retrieve_hass_conf,
                    test_optim_conf,
                    rh,
                    test_emhass_conf,
                    pd.DataFrame(),
                )

                # Verify that it called re-fit after detecting corruption
                fcst.adjust_pv_forecast_fit.assert_called_once()
                self.assertIsNotNone(result, "Should return valid result after recovery")

        finally:
            # Cleanup - unlink_missing_ok handles non-existent files safely
            model_path.unlink(missing_ok=True)

    async def test_adjusted_pv_model_max_age_affects_model_refit_behavior(self):
        """
        Test that adjusted_pv_model_max_age controls whether a cached model is reused
        vs. refit within adjust_pv_forecast.
        """
        import os
        import tempfile
        from datetime import datetime, timedelta
        from unittest.mock import AsyncMock, MagicMock

        from emhass.forecast import Forecast

        # Create a temporary data_path with a synthetic PV model file
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = pathlib.Path(tmpdir)
            model_path = data_path / "adjust_pv_regressor.pkl"

            # Create a simple picklable object to represent a valid model
            # (Using a dict instead of MagicMock since MagicMock isn't picklable)
            import aiofiles

            mock_model = {"model_type": "test", "params": [1, 2, 3]}
            async with aiofiles.open(model_path, "wb") as f:
                await f.write(pickle.dumps(mock_model))

            # Setup test objects
            fcst = MagicMock(spec=Forecast)
            p_pv_forecast = pd.Series([100, 200, 300], name="P_PV")

            test_emhass_conf = {
                "data_path": data_path,
            }

            test_retrieve_hass_conf = {}
            rh = MagicMock()

            # Mock the data retrieval to avoid real I/O
            with patch("emhass.command_line.retrieve_home_assistant_data") as mock_retrieve:
                mock_retrieve.return_value = (True, pd.DataFrame(), None)
                fcst.adjust_pv_forecast_data_prep = MagicMock()
                fcst.adjust_pv_forecast_fit = AsyncMock()
                fcst.adjust_pv_forecast_predict = MagicMock(
                    return_value=pd.DataFrame({"adjusted_forecast": [100, 200, 300]})
                )

                # Test Case 1: Fresh model with large max_age -> should load existing, no refit
                test_optim_conf_fresh = {
                    "adjusted_pv_model_max_age": 24,
                    "adjusted_pv_regression_model": "LassoRegression",
                }

                fcst.adjust_pv_forecast_fit.reset_mock()
                mock_retrieve.reset_mock()

                result = await adjust_pv_forecast(
                    logger,
                    fcst,
                    p_pv_forecast.copy(),
                    True,
                    test_retrieve_hass_conf,
                    test_optim_conf_fresh,
                    rh,
                    test_emhass_conf,
                    pd.DataFrame(),
                )

                # Should NOT call fit when model is fresh
                fcst.adjust_pv_forecast_fit.assert_not_called()
                # Should NOT retrieve data when model is fresh
                mock_retrieve.assert_not_called()
                self.assertIsNotNone(result, "Should return valid result using cached model")

                # Test Case 2: max_age = 0 -> should force refit
                test_optim_conf_force = {
                    "adjusted_pv_model_max_age": 0,
                    "adjusted_pv_regression_model": "LassoRegression",
                }

                fcst.adjust_pv_forecast_fit.reset_mock()
                mock_retrieve.reset_mock()

                result = await adjust_pv_forecast(
                    logger,
                    fcst,
                    p_pv_forecast.copy(),
                    True,
                    test_retrieve_hass_conf,
                    test_optim_conf_force,
                    rh,
                    test_emhass_conf,
                    pd.DataFrame(),
                )

                # Should call fit when max_age = 0
                fcst.adjust_pv_forecast_fit.assert_called_once()
                # Should retrieve data when refitting
                mock_retrieve.assert_called_once()
                self.assertIsNotNone(result, "Should return valid result after forced refit")

                # Test Case 3: Old model (48h old) with max_age=24 -> should refit
                # Set model file modification time to 48 hours ago
                old_time = (datetime.now() - timedelta(hours=48)).timestamp()
                os.utime(model_path, (old_time, old_time))

                test_optim_conf_stale = {
                    "adjusted_pv_model_max_age": 24,
                    "adjusted_pv_regression_model": "LassoRegression",
                }

                fcst.adjust_pv_forecast_fit.reset_mock()
                mock_retrieve.reset_mock()

                result = await adjust_pv_forecast(
                    logger,
                    fcst,
                    p_pv_forecast.copy(),
                    True,
                    test_retrieve_hass_conf,
                    test_optim_conf_stale,
                    rh,
                    test_emhass_conf,
                    pd.DataFrame(),
                )

                # Should call fit when model is stale
                fcst.adjust_pv_forecast_fit.assert_called_once()
                # Should retrieve data when refitting
                mock_retrieve.assert_called_once()
                self.assertIsNotNone(
                    result, "Should return valid result after refitting stale model"
                )

                # Test Case 4: Model just under threshold (23h old, max_age=24) -> should reuse
                # Set model file modification time to 23 hours ago
                recent_time = (datetime.now() - timedelta(hours=23)).timestamp()
                os.utime(model_path, (recent_time, recent_time))

                test_optim_conf_under = {
                    "adjusted_pv_model_max_age": 24,
                    "adjusted_pv_regression_model": "LassoRegression",
                }

                fcst.adjust_pv_forecast_fit.reset_mock()
                mock_retrieve.reset_mock()

                result = await adjust_pv_forecast(
                    logger,
                    fcst,
                    p_pv_forecast.copy(),
                    True,
                    test_retrieve_hass_conf,
                    test_optim_conf_under,
                    rh,
                    test_emhass_conf,
                    pd.DataFrame(),
                )

                # Should NOT call fit when model is just under threshold
                fcst.adjust_pv_forecast_fit.assert_not_called()
                # Should NOT retrieve data when model is still fresh enough
                mock_retrieve.assert_not_called()
                self.assertIsNotNone(
                    result,
                    "Should return valid result using cached model just under threshold",
                )


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
