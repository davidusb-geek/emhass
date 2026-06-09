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

rng = np.random.default_rng()


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
        data_values = rng.random(len(idx)) * 1000 + 500
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

    def _data_with_weather(self):
        """Return a copy of the test data with two synthetic weather covariate columns."""
        data = self.data.copy()
        # A smooth daily temperature cycle + a derived heating-degree column, so the columns are
        # meaningful (not constant) and aligned to the load index.
        hours = data.index.hour + data.index.minute / 60.0
        temp_air = 18.0 + 8.0 * np.sin((hours - 9.0) / 24.0 * 2 * np.pi)
        data["temp_air"] = temp_air
        data["heating_degree"] = np.maximum(0.0, 18.0 - temp_air)
        return data

    def _weather_future_frame(self, periods=48, freq="30min"):
        """Build a future weather frame for the predict horizon following the train window."""
        start = self.data.index[-1] + self.data.index.freq
        future_index = pd.date_range(start=start, periods=periods, freq=freq)
        hours = future_index.hour + future_index.minute / 60.0
        temp_air = 18.0 + 8.0 * np.sin((hours - 9.0) / 24.0 * 2 * np.pi)
        return pd.DataFrame(
            {"temp_air": temp_air, "heating_degree": np.maximum(0.0, 18.0 - temp_air)},
            index=future_index,
        )

    def test_mlforecaster_init(self):
        mlf = self._get_standard_mlf()
        self.assertIsInstance(mlf, MLForecaster)

    async def test_mlforecaster_fit(self):
        mlf = self._get_standard_mlf()
        df_pred, df_pred_backtest = await self._fit_standard_mlf(mlf)
        self.assertIsInstance(df_pred, pd.DataFrame)
        self.assertIs(df_pred_backtest, None)

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

    async def _treat_runtimeparams_passed_data(self, runtime_input):
        """Run treat_runtimeparams with the real associations and return passed_data."""
        params = await TestMLForecasterAsync.get_test_params()
        if "optimization_time_step" in params["retrieve_hass_conf"]:
            params["retrieve_hass_conf"]["optimization_time_step"] = pd.to_timedelta(
                params["retrieve_hass_conf"]["optimization_time_step"], "minutes"
            )
        if "delta_forecast_daily" in params["optim_conf"]:
            params["optim_conf"]["delta_forecast_daily"] = pd.to_timedelta(
                params["optim_conf"]["delta_forecast_daily"], "days"
            )
        params_json_res, _, _, _ = await utils.treat_runtimeparams(
            orjson.dumps(runtime_input).decode("utf-8"),
            params,
            {},  # retrieve_hass_conf
            {},  # optim_conf
            {},  # plant_conf
            "forecast-model-fit",
            logger,
            emhass_conf,
        )
        return orjson.loads(params_json_res)["passed_data"]

    async def test_treat_runtimeparams_weather_features_default_empty(self):
        """Without a runtime override the weather features default to an empty list."""
        passed_data = await self._treat_runtimeparams_passed_data({"model_type": "load_forecast"})
        self.assertEqual(passed_data.get("mlforecaster_weather_features"), [])

    async def test_treat_runtimeparams_weather_features_override(self):
        """A runtime mlforecaster_weather_features list is routed into passed_data."""
        passed_data = await self._treat_runtimeparams_passed_data(
            {
                "model_type": "load_forecast",
                "mlforecaster_weather_features": ["temp_air", "heating_degree"],
            }
        )
        self.assertEqual(
            passed_data["mlforecaster_weather_features"], ["temp_air", "heating_degree"]
        )

    async def test_mlforecaster_fit_backtest(self):
        """Test the backtesting block using a fast linear model and reduced data to save time."""
        # Use a fast model and a small dataset slice (e.g., last 200 rows)
        fast_data = self.data.iloc[-200:]
        mlf = MLForecaster(
            fast_data,
            self.input_data_dict["params"]["passed_data"]["model_type"],
            self.input_data_dict["params"]["passed_data"]["var_model"],
            "LinearRegression",  # Fast model to skip heavy computation
            self.input_data_dict["params"]["passed_data"]["num_lags"],
            emhass_conf,
            logger,
        )
        # Call fit with perform_backtest=True
        df_pred, df_pred_backtest = await mlf.fit(split_date_delta="48h", perform_backtest=True)
        # Assertions to ensure backtest was actually performed and formatted correctly
        self.assertIsInstance(df_pred, pd.DataFrame)
        self.assertIsInstance(df_pred_backtest, pd.DataFrame)
        self.assertIn("pred", df_pred_backtest.columns)
        self.assertIn("train", df_pred_backtest.columns)
        self.assertEqual(len(df_pred_backtest), len(fast_data))

    async def test_backtest_metrics_populated_when_perform_backtest_true(self):
        """backtest_metrics_ must be a dict with the expected keys after perform_backtest=True."""
        fast_data = self.data.iloc[-200:]
        mlf = MLForecaster(
            fast_data,
            self.input_data_dict["params"]["passed_data"]["model_type"],
            self.input_data_dict["params"]["passed_data"]["var_model"],
            "LinearRegression",
            self.input_data_dict["params"]["passed_data"]["num_lags"],
            emhass_conf,
            logger,
        )
        self.assertIsNone(mlf.backtest_metrics_)
        await mlf.fit(split_date_delta="48h", perform_backtest=True)
        self.assertIsNotNone(mlf.backtest_metrics_)
        self.assertIsInstance(mlf.backtest_metrics_, dict)
        for expected_key in ("mae", "rmse", "r2", "mape", "n_samples"):
            self.assertIn(expected_key, mlf.backtest_metrics_)
        # Sanity: MAE and RMSE must be non-negative finite numbers
        self.assertGreaterEqual(mlf.backtest_metrics_["mae"], 0.0)
        self.assertGreaterEqual(mlf.backtest_metrics_["rmse"], 0.0)
        self.assertGreater(mlf.backtest_metrics_["n_samples"], 0)

    async def test_backtest_metrics_none_when_perform_backtest_false(self):
        """backtest_metrics_ must remain None when perform_backtest=False (the default)."""
        mlf = self._get_standard_mlf()
        await mlf.fit(split_date_delta="48h", perform_backtest=False)
        self.assertIsNone(mlf.backtest_metrics_)

    async def test_backtest_metrics_reset_on_refit_without_backtest(self):
        """backtest_metrics_ must be reset to None when fit() is called without perform_backtest.

        Ensures a reused instance does not carry stale metrics from a previous fit.
        """
        fast_data = self.data.iloc[-200:]
        mlf = MLForecaster(
            fast_data,
            self.input_data_dict["params"]["passed_data"]["model_type"],
            self.input_data_dict["params"]["passed_data"]["var_model"],
            "LinearRegression",
            self.input_data_dict["params"]["passed_data"]["num_lags"],
            emhass_conf,
            logger,
        )
        # First fit WITH backtest — populates backtest_metrics_
        await mlf.fit(split_date_delta="48h", perform_backtest=True)
        self.assertIsNotNone(mlf.backtest_metrics_)
        # Second fit WITHOUT backtest — metrics must be cleared, not carry over from the first fit
        await mlf.fit(split_date_delta="48h", perform_backtest=False)
        self.assertIsNone(mlf.backtest_metrics_)

    async def test_mlforecaster_tune_short_train_size_recovery(self):
        """Test the fallback logic when initial_train_size <= window_size during tuning."""
        # Use 50 rows (25 hours). This is enough data for fit() to succeed.
        short_data = self.data.iloc[:50]
        mlf = MLForecaster(
            short_data,
            self.input_data_dict["params"]["passed_data"]["model_type"],
            self.input_data_dict["params"]["passed_data"]["var_model"],
            "LinearRegression",
            3,  # num_lags (skforecast needs len(y) > num_lags)
            emhass_conf,
            logger,
        )
        # Fit with a small test split ("5h" = 10 rows).
        # This leaves 40 rows for `self.data_train`. `fit` succeeds because 40 > 3.
        await mlf.fit(split_date_delta="5h", perform_backtest=False)
        # Tune with a huge validation split relative to the remaining data.
        # "19h" = 38 rows. It subtracts this from the 40 training rows.
        # initial_train_size becomes 40 - 38 = 2.
        # debug=True sets window_size=3.
        # Because 2 <= 3, the recovery block is triggered!
        with self.assertLogs(logger, level="WARNING") as cm:
            df_pred_optim = await mlf.tune(debug=True, split_date_delta="19h", n_trials=2)
        # Verify it survived and returned the dataframe
        self.assertIsInstance(df_pred_optim, pd.DataFrame)
        # Verify the specific fallback warnings were logged correctly
        self.assertTrue(any("Calculated initial_train_size" in log for log in cm.output))
        self.assertTrue(any("Adjusting initial_train_size" in log for log in cm.output))

    # --- Weather covariate (exog) support -------------------------------------------------

    def test_weather_features_default_is_empty(self):
        """By default no weather features are configured (backward compatible)."""
        mlf = self._get_standard_mlf()
        self.assertEqual(mlf.weather_features, [])

    async def test_fit_exog_columns_unchanged_without_weather(self):
        """Without weather features, the fit exog columns must be exactly the date features."""
        mlf = self._get_standard_mlf()
        await self._fit_standard_mlf(mlf)
        date_only = utils.add_date_features(pd.DataFrame(index=self.data.index))
        # data_exo also carries the target column; drop it before comparing the exog set.
        exog_cols = [c for c in mlf.data_exo.columns if c != mlf.var_model]
        self.assertEqual(sorted(exog_cols), sorted(date_only.columns))

    async def test_fit_includes_weather_exog_columns(self):
        """With weather features the fit exog must include the configured weather columns."""
        data = self._data_with_weather()
        weather_features = ["temp_air", "heating_degree"]
        mlf = MLForecaster(
            data,
            "load_forecast",
            "sensor.power_load_no_var_loads",
            "LinearRegression",
            48,
            emhass_conf,
            logger,
            weather_features=weather_features,
        )
        df_pred, _ = await mlf.fit(split_date_delta="48h", perform_backtest=False)
        self.assertIsInstance(df_pred, pd.DataFrame)
        for column in weather_features:
            self.assertIn(column, mlf.data_exo.columns)
        # The fitted skforecast forecaster must have registered the weather columns as exog.
        self.assertTrue(set(weather_features).issubset(set(mlf.forecaster.exog_names_in_)))

    async def test_predict_with_weather_future(self):
        """A weather-trained model predicts when supplied future weather over the horizon."""
        data = self._data_with_weather()
        weather_features = ["temp_air", "heating_degree"]
        mlf = MLForecaster(
            data,
            "load_forecast",
            "sensor.power_load_no_var_loads",
            "LinearRegression",
            48,
            emhass_conf,
            logger,
            weather_features=weather_features,
        )
        await mlf.fit(split_date_delta="48h", perform_backtest=False)
        weather_future = self._weather_future_frame(periods=48)
        df_pred = await mlf.predict(data_last_window=data, weather_future=weather_future)
        self.assertIsInstance(df_pred, pd.Series)
        self.assertEqual(len(df_pred), 48)

    async def test_predict_missing_weather_future_raises(self):
        """A weather-trained model must error clearly if no future weather is supplied."""
        data = self._data_with_weather()
        mlf = MLForecaster(
            data,
            "load_forecast",
            "sensor.power_load_no_var_loads",
            "LinearRegression",
            48,
            emhass_conf,
            logger,
            weather_features=["temp_air", "heating_degree"],
        )
        await mlf.fit(split_date_delta="48h", perform_backtest=False)
        with self.assertRaises(ValueError):
            await mlf.predict(data_last_window=data, weather_future=None)

    async def test_fit_missing_weather_column_raises(self):
        """Configured weather features absent from the training data must raise KeyError."""
        mlf = MLForecaster(
            self.data,  # no weather columns
            "load_forecast",
            "sensor.power_load_no_var_loads",
            "LinearRegression",
            48,
            emhass_conf,
            logger,
            weather_features=["temp_air"],
        )
        with self.assertRaises(KeyError):
            await mlf.fit(split_date_delta="48h", perform_backtest=False)

    async def test_generate_exog_merges_weather(self):
        """generate_exog attaches the requested weather columns from the future frame."""
        weather_future = self._weather_future_frame(periods=10)
        exog = await MLForecaster.generate_exog(
            self.data,
            10,
            "sensor.power_load_no_var_loads",
            weather_features=["temp_air", "heating_degree"],
            weather_future=weather_future,
        )
        self.assertIn("temp_air", exog.columns)
        self.assertIn("heating_degree", exog.columns)
        self.assertEqual(len(exog), 10)
        self.assertFalse(exog["temp_air"].isna().any())

    async def test_tune_with_weather_features(self):
        """Tuning a weather-trained model works (weather exog carried through from fit)."""
        data = self._data_with_weather()
        mlf = MLForecaster(
            data,
            "load_forecast",
            "sensor.power_load_no_var_loads",
            "LinearRegression",
            48,
            emhass_conf,
            logger,
            weather_features=["temp_air", "heating_degree"],
        )
        await mlf.fit(split_date_delta="48h", perform_backtest=False)
        df_pred_optim = await mlf.tune(debug=True, split_date_delta="48h")
        self.assertIsInstance(df_pred_optim, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
