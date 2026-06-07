#!/usr/bin/env python
"""Tests for the native Darts (LightGBM time-series) load forecaster.

The functional tests are skipped automatically when the optional ``darts`` /
``lightgbm`` extra is not installed, so the suite still passes on a stock EMHASS
environment. The dependency-error path and the config-wiring are always tested.
"""

import copy
import pathlib
import pickle
import unittest

import aiofiles
import numpy as np
import orjson
import pandas as pd

from emhass import utils
from emhass.command_line import set_input_data_dict
from emhass.forecast import Forecast
from emhass.retrieve_hass import RetrieveHass

# Detect whether the optional Darts extra is importable.
try:
    import darts  # noqa: F401
    import lightgbm  # noqa: F401

    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False

# The root folder
root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)


class TestDartsForecasterDependency(unittest.TestCase):
    """Tests that do not require the Darts extra to be installed."""

    def test_dependency_error_is_clear(self):
        """A clear, actionable error is raised when Darts is missing."""
        from emhass.darts_forecaster import DartsDependencyError, _require_darts

        if DARTS_AVAILABLE:
            # When installed, _require_darts must not raise.
            self.assertIsNone(_require_darts())
        else:
            with self.assertRaises(DartsDependencyError) as ctx:
                _require_darts()
            self.assertIn("pip install emhass[darts]", str(ctx.exception))

    def test_darts_is_a_valid_load_forecast_method(self):
        """The config schema accepts 'darts' as a load_forecast_method option."""
        import json

        defs = json.load(
            open(
                emhass_conf["root_path"] / "static/data/param_definitions.json",
                encoding="utf-8",
            )
        )
        # Walk all categories to find the load_forecast_method definition.
        found = None
        for _cat, params in defs.items():
            if "load_forecast_method" in params:
                found = params["load_forecast_method"]
                break
        self.assertIsNotNone(found, "load_forecast_method not found in param_definitions.json")
        self.assertIn("darts", found["select_options"])

    def test_new_params_have_defaults(self):
        """The two new Darts params have config defaults and are list-typed."""
        import json

        cfg = json.load(open(emhass_conf["defaults_path"], encoding="utf-8"))
        self.assertIn("darts_covariate_columns", cfg)
        self.assertIn("darts_quantiles", cfg)
        self.assertEqual(cfg["darts_covariate_columns"], [])
        self.assertEqual(cfg["darts_quantiles"], [])


@unittest.skipUnless(DARTS_AVAILABLE, "optional 'darts' extra not installed")
class TestDartsForecaster(unittest.IsolatedAsyncioTestCase):
    """Functional tests for the DartsForecaster, run only when Darts is present."""

    @staticmethod
    async def get_test_params():
        params = {}
        if emhass_conf["defaults_path"].exists():
            config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
            _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
            params = await utils.build_params(emhass_conf, secrets, config, logger)
        else:
            raise Exception("config_defaults.json missing: " + str(emhass_conf["defaults_path"]))
        return params

    async def asyncSetUp(self):
        params = await TestDartsForecaster.get_test_params()
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        self.retrieve_hass_conf = retrieve_hass_conf
        self.optim_conf = optim_conf
        self.plant_conf = plant_conf
        self.rh = RetrieveHass(
            self.retrieve_hass_conf["hass_url"],
            self.retrieve_hass_conf["long_lived_token"],
            self.retrieve_hass_conf["optimization_time_step"],
            self.retrieve_hass_conf["time_zone"],
            params_json,
            emhass_conf,
            logger,
        )
        filename_path = emhass_conf["data_path"] / "test_df_final.pkl"
        async with aiofiles.open(filename_path, "rb") as inp:
            content = await inp.read()
            self.rh.df_final, self.days_list, self.var_list, self.rh.ha_config = pickle.loads(
                content
            )
            self.rh.var_list = self.var_list
        self.retrieve_hass_conf["sensor_power_load_no_var_loads"] = str(self.var_list[0])
        self.var_load = str(self.var_list[0])

    def _build_training_frame(self):
        """A simple training frame: the load column on a regular index."""
        df = self.rh.df_final.copy()
        col = self.var_load if self.var_load in df.columns else df.columns[0]
        frame = df[[col]].rename(columns={col: self.var_load})
        frame = frame.dropna()
        return frame

    async def test_fit_and_predict_deterministic(self):
        """A deterministic Darts model fits and predicts a non-negative forecast."""
        from emhass.darts_forecaster import DartsForecaster

        frame = self._build_training_frame()
        dartsf = DartsForecaster(frame, "test_darts", self.var_load, 48, emhass_conf, logger)
        df_pred, df_pred_backtest = await dartsf.fit(split_date_delta="24h")
        self.assertIsInstance(df_pred, pd.DataFrame)
        self.assertIsNone(df_pred_backtest)
        self.assertIsNotNone(dartsf.model)

        forecast = await dartsf.predict(frame)
        self.assertIsInstance(forecast, pd.Series)
        self.assertGreater(len(forecast), 0)
        self.assertTrue((forecast.to_numpy() >= 0).all())
        self.assertTrue(np.isfinite(forecast.to_numpy()).all())

    async def test_fit_and_predict_quantiles(self):
        """A probabilistic Darts model exposes quantile bands."""
        from emhass.darts_forecaster import DartsForecaster

        frame = self._build_training_frame()
        dartsf = DartsForecaster(
            frame,
            "test_darts_q",
            self.var_load,
            48,
            emhass_conf,
            logger,
            quantiles=[0.1, 0.5, 0.9],
        )
        await dartsf.fit(split_date_delta="24h")
        forecast = await dartsf.predict(frame)
        self.assertIsInstance(forecast, pd.Series)
        self.assertIsNotNone(dartsf.last_quantiles)
        # P50 (returned) should sit within [P10, P90] on average.
        p10 = dartsf._quantile_column(dartsf.last_quantiles, 0.1).clip(lower=0)
        p90 = dartsf._quantile_column(dartsf.last_quantiles, 0.9)
        self.assertLessEqual(float(p10.mean()), float(forecast.mean()) + 1e-6)
        self.assertGreaterEqual(float(p90.mean()) + 1e-6, float(forecast.mean()))

    async def test_tune_runs(self):
        """The light hyper-parameter search runs and refits."""
        from emhass.darts_forecaster import DartsForecaster

        frame = self._build_training_frame()
        dartsf = DartsForecaster(frame, "test_darts_tune", self.var_load, 48, emhass_conf, logger)
        await dartsf.fit(split_date_delta="24h")
        df_opt = await dartsf.tune(split_date_delta="24h", n_trials=2)
        self.assertIsInstance(df_opt, pd.DataFrame)
        self.assertIn("pred_optim", df_opt.columns)

    def test_sanity_check_flags_divergence(self):
        """The sanity check flags a grossly diverging forecast, passes a sane one."""
        from emhass.darts_forecaster import DartsForecaster

        frame = self._build_training_frame()
        dartsf = DartsForecaster(frame, "test_darts_sanity", self.var_load, 48, emhass_conf, logger)
        history = pd.Series([500.0] * 300)
        sane = pd.Series([520.0] * 100)
        insane = pd.Series([50000.0] * 100)
        self.assertTrue(dartsf.sanity_check(sane, history)["ok"])
        self.assertFalse(dartsf.sanity_check(insane, history)["ok"])

    async def test_get_load_forecast_darts_dispatch(self):
        """Forecast.get_load_forecast(method='darts') returns an aligned Series."""
        from emhass.darts_forecaster import DartsForecaster

        params = await TestDartsForecaster.get_test_params()
        params = copy.deepcopy(params)
        params["optim_conf"]["load_forecast_method"] = "darts"
        # model_type must match a training-data fixture pkl present in data/.
        runtimeparams = {
            "model_type": "long_train_data",
            "var_model": self.var_load,
            "num_lags": 48,
        }
        params["passed_data"] = runtimeparams
        params_json = orjson.dumps(params).decode("utf-8")
        runtimeparams_json = orjson.dumps(runtimeparams).decode("utf-8")
        input_data_dict = await set_input_data_dict(
            emhass_conf,
            "profit",
            params_json,
            runtimeparams_json,
            "forecast-model-fit",
            logger,
            get_data_from_file=True,
        )
        data = copy.deepcopy(input_data_dict["df_input_data"])
        var_model = input_data_dict["params"]["passed_data"]["var_model"]
        dartsf = DartsForecaster(data, "long_train_data", var_model, 48, emhass_conf, logger)
        await dartsf.fit(split_date_delta="24h")
        fcst: Forecast = input_data_dict["fcst"]
        p_load = await fcst.get_load_forecast(
            method="darts", use_last_window=False, debug=True, dartsf=dartsf
        )
        self.assertIsInstance(p_load, pd.Series)
        self.assertTrue((p_load.index == fcst.forecast_dates).all())


if __name__ == "__main__":
    unittest.main()
