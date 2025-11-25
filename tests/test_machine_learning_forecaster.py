import copy
import pathlib
import pickle
import unittest

import aiofiles
import numpy as np
import orjson
import pandas as pd
from skforecast.recursive import ForecasterRecursive

from emhass import utils
from emhass.command_line import set_input_data_dict
from emhass.machine_learning_forecaster import MLForecaster
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


class TestMLForecasterAsync(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def get_test_params():
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
                "config_defaults. does not exist in path: "
                + str(emhass_conf["defaults_path"])
            )
        return params

    async def asyncSetUp(self):
        params = await TestMLForecasterAsync.get_test_params()
        costfun = "profit"
        action = "forecast-model-fit"  # fit, predict and tune methods
        # Create runtime parameters
        runtimeparams = {
            "historic_days_to_retrieve": 20,
            "model_type": "long_train_data",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48,
        }
        runtimeparams_json = orjson.dumps(runtimeparams).decode()
        params["passed_data"] = runtimeparams
        params["optim_conf"]["load_forecast_method"] = "skforecast"
        # Create input dictionary
        params_json = orjson.dumps(params).decode()
        self.input_data_dict = await set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        # Create MLForcaster object
        data = copy.deepcopy(self.input_data_dict["df_input_data"])
        model_type = self.input_data_dict["params"]["passed_data"]["model_type"]
        var_model = self.input_data_dict["params"]["passed_data"]["var_model"]
        sklearn_model = self.input_data_dict["params"]["passed_data"]["sklearn_model"]
        num_lags = self.input_data_dict["params"]["passed_data"]["num_lags"]
        self.mlf = MLForecaster(
            data, model_type, var_model, sklearn_model, num_lags, emhass_conf, logger
        )
        # Create RetrieveHass Object
        get_data_from_file = True
        params = None
        self.retrieve_hass_conf, self.optim_conf, _ = utils.get_yaml_parse(
            params_json, logger
        )
        self.rh = RetrieveHass(
            self.retrieve_hass_conf["hass_url"],
            self.retrieve_hass_conf["long_lived_token"],
            self.retrieve_hass_conf["optimization_time_step"],
            self.retrieve_hass_conf["time_zone"],
            params_json,
            emhass_conf,
            logger,
            get_data_from_file=get_data_from_file,
        )
        # Open and extract saved sensor data to test against
        async with aiofiles.open(
            emhass_conf["data_path"] / "test_df_final.pkl", "rb"
        ) as inp:
            content = await inp.read()
            self.rh.df_final, self.days_list, self.var_list, self.rh.ha_config = (
                pickle.loads(content)
            )

    async def test_fit(self):
        df_pred, df_pred_backtest = await self.mlf.fit()
        self.assertIsInstance(self.mlf.forecaster, ForecasterRecursive)
        self.assertIsInstance(df_pred, pd.DataFrame)
        self.assertTrue(df_pred_backtest is None)
        # Refit with backtest evaluation
        df_pred, df_pred_backtest = await self.mlf.fit(perform_backtest=True)
        self.assertIsInstance(self.mlf.forecaster, ForecasterRecursive)
        self.assertIsInstance(df_pred, pd.DataFrame)
        self.assertIsInstance(df_pred_backtest, pd.DataFrame)

    async def test_predict(self):
        await self.mlf.fit()
        predictions = await self.mlf.predict()
        self.assertIsInstance(predictions, pd.Series)
        self.assertTrue(predictions.isnull().sum().sum() == 0)
        # Test predict in production env using last_window
        data_tmp = copy.deepcopy(self.rh.df_final)[[self.mlf.var_model]]
        data_last_window = data_tmp[data_tmp.index[-1] - pd.offsets.Day(2) :]
        predictions = await self.mlf.predict(data_last_window)
        self.assertIsInstance(predictions, pd.Series)
        self.assertTrue(predictions.isnull().sum().sum() == 0)
        # Test again with last_window data but with NaNs
        data_last_window.at[data_last_window.index[10], self.mlf.var_model] = np.nan
        data_last_window.at[data_last_window.index[11], self.mlf.var_model] = np.nan
        data_last_window.at[data_last_window.index[12], self.mlf.var_model] = np.nan
        predictions = await self.mlf.predict(data_last_window)
        self.assertIsInstance(predictions, pd.Series)
        self.assertTrue(predictions.isnull().sum().sum() == 0)
        # Emulate predict on optimized forecaster
        self.mlf.is_tuned = True
        self.mlf.lags_opt = 48
        await self.mlf.fit()
        predictions = await self.mlf.predict()
        self.assertIsInstance(predictions, pd.Series)
        self.assertTrue(predictions.isnull().sum().sum() == 0)

    async def test_tune(self):
        await self.mlf.fit()
        df_pred_optim = await self.mlf.tune(debug=True)
        self.assertIsInstance(df_pred_optim, pd.DataFrame)
        self.assertTrue(self.mlf.is_tuned is True)
        # Test LinearRegression
        data = copy.deepcopy(self.input_data_dict["df_input_data"])
        model_type = self.input_data_dict["params"]["passed_data"]["model_type"]
        var_model = self.input_data_dict["params"]["passed_data"]["var_model"]
        sklearn_model = "LinearRegression"
        num_lags = self.input_data_dict["params"]["passed_data"]["num_lags"]
        self.mlf = MLForecaster(
            data, model_type, var_model, sklearn_model, num_lags, emhass_conf, logger
        )
        await self.mlf.fit()
        df_pred_optim = await self.mlf.tune(debug=True)
        self.assertIsInstance(df_pred_optim, pd.DataFrame)
        self.assertTrue(self.mlf.is_tuned is True)
        # Test ElasticNet
        data = copy.deepcopy(self.input_data_dict["df_input_data"])
        model_type = self.input_data_dict["params"]["passed_data"]["model_type"]
        var_model = self.input_data_dict["params"]["passed_data"]["var_model"]
        sklearn_model = "ElasticNet"
        num_lags = self.input_data_dict["params"]["passed_data"]["num_lags"]
        self.mlf = MLForecaster(
            data, model_type, var_model, sklearn_model, num_lags, emhass_conf, logger
        )
        await self.mlf.fit()
        df_pred_optim = await self.mlf.tune(debug=True)
        self.assertIsInstance(df_pred_optim, pd.DataFrame)
        self.assertTrue(self.mlf.is_tuned is True)


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
