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
        """The new Darts params have config defaults of the expected type."""
        import json

        cfg = json.load(open(emhass_conf["defaults_path"], encoding="utf-8"))
        self.assertIn("darts_covariate_columns", cfg)
        self.assertIn("darts_quantiles", cfg)
        self.assertEqual(cfg["darts_covariate_columns"], [])
        self.assertEqual(cfg["darts_quantiles"], [])
        # The price/cost models need a target that can go negative.
        self.assertIn("darts_non_negative", cfg)
        self.assertEqual(cfg["darts_non_negative"], True)

    def test_darts_is_a_valid_cost_and_price_method(self):
        """The schema accepts 'darts' for the load-cost and prod-price methods."""
        import json

        defs = json.load(
            open(
                emhass_conf["root_path"] / "static/data/param_definitions.json",
                encoding="utf-8",
            )
        )
        for key in ("load_cost_forecast_method", "production_price_forecast_method"):
            found = None
            for _cat, params in defs.items():
                if key in params:
                    found = params[key]
                    break
            self.assertIsNotNone(found, f"{key} not found in param_definitions.json")
            self.assertIn("darts", found["select_options"])


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

    def test_predict_allows_negative_when_not_non_negative(self):
        """A non_negative=False target (a price) is not floored at zero."""
        from emhass.darts_forecaster import DartsForecaster

        async def _run():
            frame = self._build_training_frame()
            # Shift the series strongly negative so the forecast must be < 0
            # unless the (load-only) zero floor is wrongly applied.
            price = frame - float(frame.iloc[:, 0].mean()) - 1000.0
            dartsf = DartsForecaster(
                price,
                "test_darts_price",
                self.var_load,
                48,
                emhass_conf,
                logger,
                non_negative=False,
            )
            await dartsf.fit(split_date_delta="24h")
            forecast = await dartsf.predict(price)
            return forecast

        import asyncio as _asyncio

        forecast = _asyncio.run(_run())
        self.assertIsInstance(forecast, pd.Series)
        # The defining behaviour: at least one forecast value is negative,
        # which is only possible because the zero floor was not applied.
        self.assertTrue((forecast.to_numpy() < 0).any())

    async def _fit_and_pickle_price_model(self, fcst, suffix):
        """Fit a Darts model on a price-like series and pickle it for dispatch."""
        from emhass.darts_forecaster import DartsForecaster

        frame = self._build_training_frame()
        # A price-like series: small magnitude, mean-shifted to include negatives.
        scale = float(frame.iloc[:, 0].std()) or 1.0
        price = (frame - frame.iloc[:, 0].mean()) / scale * 0.1
        dartsf = DartsForecaster(
            price,
            "long_train_data",
            self.var_load,
            48,
            emhass_conf,
            logger,
            non_negative=False,
        )
        await dartsf.fit(split_date_delta="24h")
        filename_path = emhass_conf["data_path"] / f"long_train_data_{suffix}_darts.pkl"
        async with aiofiles.open(filename_path, "wb") as outp:
            await outp.write(pickle.dumps(dartsf, pickle.HIGHEST_PROTOCOL))
        return filename_path

    async def _make_price_dispatch_fcst(self):
        """Build a get_data_from_file Forecast object for a price dispatch test."""
        params = copy.deepcopy(await TestDartsForecaster.get_test_params())
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
        return input_data_dict["fcst"]

    async def test_get_load_cost_forecast_darts_dispatch(self):
        """Forecast.get_load_cost_forecast(method='darts') appends the cost column."""
        fcst: Forecast = await self._make_price_dispatch_fcst()
        await self._fit_and_pickle_price_model(fcst, "load_cost")
        # get_data_from_file=True makes the helper forecast from the model tail
        # (no live HA fetch), so the dispatch is exercised offline.
        fcst.get_data_from_file = True
        df_final = pd.DataFrame(index=fcst.forecast_dates)
        out = fcst.get_load_cost_forecast(df_final, method="darts")
        self.assertIsInstance(out, pd.DataFrame)
        self.assertIn(fcst.var_load_cost, out.columns)
        self.assertEqual(len(out[fcst.var_load_cost].dropna()), len(fcst.forecast_dates))
        self.assertTrue(np.isfinite(out[fcst.var_load_cost].to_numpy()).all())

    async def test_get_prod_price_forecast_darts_dispatch(self):
        """Forecast.get_prod_price_forecast(method='darts') appends the price column."""
        fcst: Forecast = await self._make_price_dispatch_fcst()
        await self._fit_and_pickle_price_model(fcst, "prod_price")
        fcst.get_data_from_file = True
        df_final = pd.DataFrame(index=fcst.forecast_dates)
        out = fcst.get_prod_price_forecast(df_final, method="darts")
        self.assertIsInstance(out, pd.DataFrame)
        self.assertIn(fcst.var_prod_price, out.columns)
        self.assertEqual(len(out[fcst.var_prod_price].dropna()), len(fcst.forecast_dates))

    async def test_get_load_cost_forecast_darts_missing_model(self):
        """A missing price-model pickle surfaces as False, not an exception."""
        fcst: Forecast = await self._make_price_dispatch_fcst()
        # Ensure no pickle exists for this suffix.
        stale = emhass_conf["data_path"] / "long_train_data_load_cost_darts.pkl"
        if stale.exists():
            stale.unlink()
        fcst.get_data_from_file = True
        df_final = pd.DataFrame(index=fcst.forecast_dates)
        out = fcst.get_load_cost_forecast(df_final, method="darts")
        self.assertFalse(out)


if __name__ == "__main__":
    unittest.main()
