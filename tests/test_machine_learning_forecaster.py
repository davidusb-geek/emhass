import pathlib
import unittest

import numpy as np
import orjson
import pandas as pd

from emhass import utils
from emhass.machine_learning_forecaster import MLForecaster

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


class TestMLForecasterAsync(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Load Raw Parameters
        self.params = await TestMLForecasterAsync.get_test_params()
        # Serialize to JSON (Raw types)
        self.params_json = orjson.dumps(self.params).decode("utf-8")
        # Convert ALL time parameters to Timedelta objects for Python code
        if "optimization_time_step" in self.params["retrieve_hass_conf"]:
            self.params["retrieve_hass_conf"]["optimization_time_step"] = pd.to_timedelta(
                self.params["retrieve_hass_conf"]["optimization_time_step"], "minutes"
            )
        if "delta_forecast_daily" in self.params["optim_conf"]:
            self.params["optim_conf"]["delta_forecast_daily"] = pd.to_timedelta(
                self.params["optim_conf"]["delta_forecast_daily"], "days"
            )
        # Populate passed_data defaults
        self.params["passed_data"].update(
            {
                "model_type": "load_forecast",
                "var_model": "sensor.power_load_no_var_loads",
                "sklearn_model": "KNeighborsRegressor",
                "num_lags": 48,
                "split_date_delta": "48h",
                "perform_backtest": False,
            }
        )
        self.runtimeparams = {
            "historic_days_to_retrieve": 20,
            "model_type": "load_forecast",
            "var_model": "sensor.power_load_no_var_loads",
            "sklearn_model": "KNeighborsRegressor",
            "num_lags": 48,
            "split_date_delta": "48h",
            "perform_backtest": False,
        }
        self.runtimeparams_json = orjson.dumps(self.runtimeparams).decode("utf-8")
        self.action = "forecast-model-fit"
        self.costfun = "profit"
        # Generate test data
        idx = pd.date_range(end=pd.Timestamp.now(), periods=48 * 20, freq="30min")
        np.random.seed(42)
        data_values = np.random.rand(len(idx)) * 1000 + 500
        self.data = pd.DataFrame({"sensor.power_load_no_var_loads": data_values}, index=idx)
        self.data.index.name = "timestamp"
        self.data = utils.set_df_index_freq(self.data)
        self.input_data_dict = {
            "root": emhass_conf["root_path"],
            "retrieve_hass_conf": self.params["retrieve_hass_conf"],
            "df_input_data": self.data,
            "df_input_data_dayahead": self.data,
            "opt_pars": self.params["optim_conf"],
            "params": self.params,
        }

    @staticmethod
    async def get_test_params():
        if emhass_conf["defaults_path"].exists():
            config_path = (
                emhass_conf["config_path"] if emhass_conf["config_path"].exists() else None
            )
            config = await utils.build_config(
                emhass_conf, logger, emhass_conf["defaults_path"], config_path
            )
            params = await utils.build_params(emhass_conf, {}, config, logger)
            return params
        else:
            raise Exception(
                "config_defaults.json not found in " + str(emhass_conf["defaults_path"])
            )

    def _get_standard_mlf(self):
        """Helper to initialize MLForecaster with standard parameters."""
        self.input_data_dict["params"]["passed_data"]["model_type"] = "load_forecast"
        self.input_data_dict["params"]["passed_data"]["sklearn_model"] = "KNeighborsRegressor"
        return MLForecaster(
            self.data,
            self.input_data_dict["params"]["passed_data"]["model_type"],
            self.input_data_dict["params"]["passed_data"]["var_model"],
            self.input_data_dict["params"]["passed_data"]["sklearn_model"],
            self.input_data_dict["params"]["passed_data"]["num_lags"],
            emhass_conf,
            logger,
        )

    async def _fit_standard_mlf(self, mlf):
        """Helper to fit the model with standard parameters."""
        return await mlf.fit(
            split_date_delta=self.input_data_dict["params"]["passed_data"]["split_date_delta"],
            perform_backtest=self.input_data_dict["params"]["passed_data"]["perform_backtest"],
        )

    async def test_mlforecaster_init(self):
        mlf = self._get_standard_mlf()
        self.assertIsInstance(mlf, MLForecaster)

    async def test_mlforecaster_fit(self):
        mlf = self._get_standard_mlf()
        df_pred, df_pred_backtest = await self._fit_standard_mlf(mlf)
        self.assertIsInstance(df_pred, pd.DataFrame)
        self.assertTrue(df_pred_backtest is None)

    async def test_mlforecaster_predict(self):
        mlf = self._get_standard_mlf()
        await self._fit_standard_mlf(mlf)
        df_pred = await mlf.predict(
            data_last_window=self.data,
        )
        self.assertIsInstance(df_pred, pd.Series)

    async def test_mlforecaster_tune(self):
        mlf = self._get_standard_mlf()
        await self._fit_standard_mlf(mlf)
        df_pred_optim = await mlf.tune(
            debug=True,
            split_date_delta=self.input_data_dict["params"]["passed_data"]["split_date_delta"],
        )
        self.assertIsInstance(df_pred_optim, pd.DataFrame)

    async def test_error_handling_and_fallbacks(self):
        """Test error handling for invalid models or data."""
        self.input_data_dict["params"]["passed_data"]["sklearn_model"] = "InvalidModel"
        mlf = MLForecaster(
            self.data,
            self.input_data_dict["params"]["passed_data"]["model_type"],
            self.input_data_dict["params"]["passed_data"]["var_model"],
            self.input_data_dict["params"]["passed_data"]["sklearn_model"],
            self.input_data_dict["params"]["passed_data"]["num_lags"],
            emhass_conf,
            logger,
        )
        try:
            await mlf.fit(
                split_date_delta=self.input_data_dict["params"]["passed_data"]["split_date_delta"],
                perform_backtest=self.input_data_dict["params"]["passed_data"]["perform_backtest"],
            )
        except Exception as e:
            self.assertIsInstance(e, (ValueError, AttributeError))

    async def test_tune_edge_case_short_data(self):
        """Test tuning with very short data (edge case)."""
        short_data = self.data.iloc[:50]  # minimal data
        mlf = MLForecaster(
            short_data,
            self.input_data_dict["params"]["passed_data"]["model_type"],
            self.input_data_dict["params"]["passed_data"]["var_model"],
            self.input_data_dict["params"]["passed_data"]["sklearn_model"],
            self.input_data_dict["params"]["passed_data"]["num_lags"],
            emhass_conf,
            logger,
        )
        try:
            await mlf.fit(split_date_delta="1h", perform_backtest=False)
            await mlf.tune(
                debug=True,
                split_date_delta="1h",
            )
        except Exception as e:
            self.assertIsInstance(e, (ValueError, IndexError, RuntimeError))

    async def test_treat_runtimeparams_ml_lags_real_associations(self):
        """
        Test that num_lags passed at runtime is correctly mapped using the REAL
        associations.csv file shipped with the package.
        """
        runtime_input = {
            "num_lags": 96,
            "model_type": "load_forecast",
            "perform_backtest": True,
            "sklearn_model": "RandomForestRegressor",
            "n_trials": 50,
        }
        runtimeparams_json = orjson.dumps(runtime_input).decode("utf-8")
        params = await TestMLForecasterAsync.get_test_params()
        # Modify the base config to test precedence
        params["optim_conf"]["num_lags"] = 48
        params["optim_conf"]["sklearn_model"] = "XGBRegressor"
        params["optim_conf"]["regression_model"] = "LinearRegression"
        if "split_date_delta" in params["optim_conf"]:
            del params["optim_conf"]["split_date_delta"]
        if "optimization_time_step" in params["retrieve_hass_conf"]:
            params["retrieve_hass_conf"]["optimization_time_step"] = pd.to_timedelta(
                params["retrieve_hass_conf"]["optimization_time_step"], "minutes"
            )
        if "delta_forecast_daily" in params["optim_conf"]:
            params["optim_conf"]["delta_forecast_daily"] = pd.to_timedelta(
                params["optim_conf"]["delta_forecast_daily"], "days"
            )
        params_json_res, _, _, _ = await utils.treat_runtimeparams(
            runtimeparams_json,
            params,
            {},  # retrieve_hass_conf
            {},  # optim_conf
            {},  # plant_conf
            "forecast-model-fit",
            logger,
            emhass_conf,
        )
        params_result = orjson.loads(params_json_res)
        passed_data = params_result["passed_data"]
        optim_conf = params_result["optim_conf"]
        # Runtime Overrides
        self.assertEqual(optim_conf.get("num_lags"), 96)
        self.assertEqual(passed_data.get("num_lags"), 96)
        self.assertEqual(passed_data.get("sklearn_model"), "RandomForestRegressor")
        self.assertTrue(passed_data.get("perform_backtest"))
        self.assertEqual(passed_data.get("n_trials"), 50)
        # Config Preservation
        self.assertEqual(passed_data.get("regression_model"), "LinearRegression")
        # Default Fallback
        self.assertEqual(passed_data.get("split_date_delta"), "48h")


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
