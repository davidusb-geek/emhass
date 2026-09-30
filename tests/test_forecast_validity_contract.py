#!/usr/bin/env python
"""Regression tests for the forecast validity contract (issue #1135).

Two separate rules are covered here:

* numerical validity - every externally supplied forecast value must be a
  finite real number (NaN, +/-Inf, booleans and non-numeric values are
  rejected and the optimization cycle fails before the solver);
* physical-domain validity - the optimizer-facing household load returned by
  ``Forecast.get_load_forecast()`` is finite and ``>= 0 W``. A finite negative
  load is clipped to 0 W with one summarized warning; a non-finite load fails
  the cycle. Signed prices and outdoor temperatures stay signed.
"""

import copy
import logging
import pathlib
import pickle
import tempfile
import unittest
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import orjson
import pandas as pd

from emhass import command_line, utils, web_server
from emhass.command_line import (
    OptimizationCache,
    dayahead_forecast_optim,
    forecast_model_predict,
    naive_mpc_optim,
    set_input_data_dict,
)
from emhass.forecast import Forecast
from emhass.machine_learning_forecaster import MLForecaster
from emhass.optimization import Optimization
from emhass.retrieve_hass import RetrieveHass

ROOT = pathlib.Path(utils.get_root(__file__, num_parent=2))
EMHASS_CONF = {
    "data_path": ROOT / "data/",
    "root_path": ROOT / "src/emhass/",
    "config_path": ROOT / "config.json",
    "defaults_path": ROOT / "src/emhass/data/config_defaults.json",
    "associations_path": ROOT / "src/emhass/data/associations.csv",
}

logger = logging.getLogger("test_forecast_validity_contract")

FORECAST_KEYS = {
    "pv_power_forecast": "weather_forecast_method",
    "load_power_forecast": "load_forecast_method",
    "load_cost_forecast": "load_cost_forecast_method",
    "prod_price_forecast": "production_price_forecast_method",
    "outdoor_temperature_forecast": "outdoor_temperature_forecast_method",
}

INVALID_VALUES = {
    "NaN": float("nan"),
    "+Inf": float("inf"),
    "-Inf": float("-inf"),
    "bool": True,
    "numpy bool": np.bool_(False),
    "string": "abc",
    "None": None,
}

# Long enough for any 30 min / 1 day horizon, including DST days (50 steps).
LONG = 96


async def _build_params(**optim_overrides) -> dict:
    config = await utils.build_config(EMHASS_CONF, logger, EMHASS_CONF["defaults_path"])
    _, secrets = await utils.build_secrets(EMHASS_CONF, logger, no_response=True)
    params = await utils.build_params(EMHASS_CONF, secrets, config, logger)
    params["optim_conf"].update(optim_overrides)
    return params


async def _treat(runtimeparams: dict, set_type: str = "dayahead-optim"):
    """Run the public runtime boundary with a dict payload (keeps NaN/bool types)."""
    params_json = orjson.dumps(await _build_params()).decode("utf-8")
    retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
    treated, _, optim_conf, _ = await utils.treat_runtimeparams(
        copy.deepcopy(runtimeparams),
        params_json,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        set_type,
        logger,
        EMHASS_CONF,
    )
    return orjson.loads(treated), optim_conf


def _errors_mentioning(captured, text):
    return [
        record
        for record in captured.records
        if record.levelno >= logging.ERROR and text in record.getMessage()
    ]


class _PinnedNow(unittest.IsolatedAsyncioTestCase):
    """Pin the runtime forecast grid to a fixed, non-DST instant."""

    async def asyncSetUp(self):
        now_patch = patch(
            "emhass.utils._get_now",
            return_value=pd.Timestamp("2026-06-26T07:00:00", tz=UTC).to_pydatetime(),
        )
        now_patch.start()
        self.addCleanup(now_patch.stop)
        params = await _build_params()
        retrieve_hass_conf, optim_conf, _ = utils.get_yaml_parse(
            orjson.dumps(params).decode("utf-8"), logger
        )
        self.time_zone = retrieve_hass_conf["time_zone"]
        step = int(retrieve_hass_conf["optimization_time_step"].total_seconds() / 60)
        self.forecast_dates = utils.get_forecast_dates(
            step, optim_conf["delta_forecast_daily"].days, self.time_zone
        )
        self.horizon = len(self.forecast_dates)


class TestRuntimeNumericalValidity(_PinnedNow):
    """External runtime forecasts must be finite real numbers."""

    async def test_valid_plain_lists_are_passed_unchanged(self):
        n = self.horizon
        runtimeparams = {
            "pv_power_forecast": [0.0] * 4 + [1500.5] * (n - 4),
            # Zero load, positive load and a finite negative load are all
            # numerically valid; the physical load domain is enforced later at
            # Forecast.get_load_forecast().
            "load_power_forecast": [0, 250.0, -12.5] + [400] * (n - 3),
            # Signed prices and temperatures are legitimate values.
            "load_cost_forecast": [-0.05, 0.0] + [0.25] * (n - 2),
            "prod_price_forecast": [-0.12] * n,
            "outdoor_temperature_forecast": [-15.5, -0.5] + [3] * (n - 2),
        }
        params, optim_conf = await _treat(runtimeparams)
        for key, method in FORECAST_KEYS.items():
            with self.subTest(key=key):
                self.assertEqual(params["passed_data"][key], runtimeparams[key])
                self.assertEqual(optim_conf[method], "list")

    async def test_long_list_is_accepted_and_short_list_keeps_existing_invalid_behavior(self):
        n = self.horizon
        params, optim_conf = await _treat({"load_power_forecast": [100.0] * (n + 10)})
        self.assertEqual(params["passed_data"]["load_power_forecast"], [100.0] * (n + 10))
        self.assertEqual(optim_conf["load_forecast_method"], "list")

        # Existing contract: a list shorter than the horizon is rejected with an
        # error; it is not padded and is not passed on to the forecast.
        with self.assertLogs(logger, level="ERROR") as captured:
            params, optim_conf = await _treat({"load_power_forecast": [100.0] * (n - 1)})
        self.assertIsNone(params["passed_data"]["load_power_forecast"])
        self.assertTrue(_errors_mentioning(captured, "the length is not correct"))

    async def test_invalid_plain_list_values_fail_closed(self):
        n = self.horizon
        for key, method in FORECAST_KEYS.items():
            for label, bad in INVALID_VALUES.items():
                with self.subTest(key=key, invalid=label):
                    values = [10.0] * n
                    values[5] = bad
                    values[9] = bad  # a second bad point must not add a second error
                    with self.assertLogs(logger, level="ERROR") as captured:
                        params, optim_conf = await _treat({key: values})
                    # Invalid data is not forwarded, and the method is forced to
                    # "list" so the cycle fails instead of falling back to the
                    # configured forecast method.
                    self.assertIsNone(params["passed_data"][key])
                    self.assertEqual(optim_conf[method], "list")
                    errors = _errors_mentioning(captured, key)
                    self.assertEqual(len(errors), 1, [r.getMessage() for r in errors])
                    message = errors[0].getMessage()
                    self.assertIn("position 5", message)
                    self.assertIn(repr(bad), message)
                    self.assertIn("finite real numbers", message)

    async def test_invalid_mapping_source_values_fail_closed_with_timestamp(self):
        stamps = [self.forecast_dates[i] for i in (0, 2, 4)]
        for key, method in FORECAST_KEYS.items():
            for label, bad in INVALID_VALUES.items():
                with self.subTest(key=key, invalid=label):
                    mapping = dict(zip(stamps, [10.0, bad, 30.0], strict=True))
                    with self.assertLogs(logger, level="ERROR") as captured:
                        params, optim_conf = await _treat({key: mapping})
                    self.assertIsNone(params["passed_data"][key])
                    self.assertEqual(optim_conf[method], "list")
                    errors = _errors_mentioning(captured, key)
                    self.assertEqual(len(errors), 1, [r.getMessage() for r in errors])
                    self.assertIn(stamps[1], errors[0].getMessage())
                    self.assertIn(repr(bad), errors[0].getMessage())


class TestRuntimeMappingAlignmentUnchanged(_PinnedNow):
    """Valid timestamp mappings keep the existing aggregation/alignment semantics."""

    async def test_mapping_alignment_semantics(self):
        dates = [pd.Timestamp(d) for d in self.forecast_dates]
        step = dates[1] - dates[0]
        quarter = step / 2
        utc = [d.tz_convert("UTC").isoformat() for d in dates]
        mapping = {
            # Leading target slots 0..1 precede the first supplied point and
            # are back-filled from it.
            dates[2].isoformat(): -4.0,
            # Two sub-step points inside slot 2 are averaged in local time.
            (dates[2] + quarter).isoformat(): -2.0,
            # Keys supplied in UTC align by instant.
            utc[5]: 7.0,
            # Hold-last: slots 6.. keep 7.0 until the next point at slot 8.
            utc[8]: 1.5,
        }
        expected = [-3.0, -3.0, -3.0, -3.0, -3.0, 7.0, 7.0, 7.0, 1.5, 1.5]
        for key in ("load_cost_forecast", "prod_price_forecast", "outdoor_temperature_forecast"):
            with self.subTest(key=key):
                params, _ = await _treat({key: mapping})
                aligned = params["passed_data"][key]
                self.assertEqual(len(aligned), self.horizon)
                self.assertEqual(aligned[:10], expected)
                self.assertTrue(all(v == 1.5 for v in aligned[10:]))
                # Byte/value equivalence with the shared alignment helper.
                reference = utils._align_runtime_forecast_mapping(
                    mapping, self.forecast_dates, int(step.total_seconds() / 60), self.time_zone
                )
                self.assertEqual(aligned, reference)

        # A negative finite load mapping is numerically valid at this boundary;
        # the final load boundary clips it (covered below).
        params, optim_conf = await _treat({"load_power_forecast": mapping})
        self.assertEqual(params["passed_data"]["load_power_forecast"][:10], expected)
        self.assertEqual(optim_conf["load_forecast_method"], "list")


class _ForecastFixture(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        params = await _build_params()
        params["passed_data"]["alpha"] = 0.5
        params["passed_data"]["beta"] = 0.5
        self.params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json, logger)
        self.fcst = Forecast(
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            self.params_json,
            EMHASS_CONF,
            logger,
            get_data_from_file=True,
        )
        self.n = len(self.fcst.forecast_dates)

    async def _list_load(self, values, **kwargs):
        self.fcst.params["passed_data"]["load_power_forecast"] = values
        return await self.fcst.get_load_forecast(method="list", **kwargs)


class TestFinalLoadBoundary(_ForecastFixture):
    async def test_valid_non_negative_load_is_unchanged(self):
        values = [0.0, 0.0] + [float(100 + i) for i in range(self.n - 2)]
        with patch.object(logger, "warning") as warning:
            result = await self._list_load(values)
        self.assertEqual(result.tolist(), values)
        self.assertEqual(result.name, "yhat")
        self.assertTrue(result.index.equals(self.fcst.forecast_dates))
        warning.assert_not_called()

    async def test_finite_negative_load_is_clipped_with_one_summary_warning(self):
        values = [300.0] * self.n
        values[3], values[7], values[8] = -0.881, -37.386, -5.0
        with self.assertLogs(logger, level="WARNING") as captured:
            result = await self._list_load(values)
        expected = [max(v, 0.0) for v in values]
        self.assertEqual(result.tolist(), expected)
        self.assertEqual(result.name, "yhat")
        self.assertTrue(result.index.equals(self.fcst.forecast_dates))
        warnings = [r.getMessage() for r in captured.records if "clipped to 0 W" in r.getMessage()]
        self.assertEqual(len(warnings), 1, captured.output)
        self.assertIn("3 negative", warnings[0])
        self.assertIn("-37.386", warnings[0])
        self.assertIn(str(self.fcst.forecast_dates[3]), warnings[0])

    async def test_non_finite_or_non_numeric_load_fails(self):
        for label, bad in INVALID_VALUES.items():
            if label in ("bool", "numpy bool"):
                continue  # covered through the CSV/ML paths and the runtime boundary
            with self.subTest(invalid=label):
                values = [300.0] * self.n
                values[4] = bad
                with self.assertLogs(logger, level="ERROR") as captured:
                    result = await self._list_load(values)
                self.assertIs(result, False)
                self.assertEqual(
                    len([r for r in captured.records if r.levelno >= logging.ERROR]), 1
                )
                self.assertIn(str(self.fcst.forecast_dates[4]), captured.output[0])

    async def test_every_load_method_is_routed_through_the_boundary(self):
        def frame(values):
            return pd.DataFrame({"yhat": values}, index=self.fcst.forecast_dates)

        values = [-10.0] + [50.0] * (self.n - 1)
        cases = {
            "typical": ("_get_load_forecast_typical", AsyncMock(return_value=frame(values))),
            "naive": ("_get_load_forecast_naive", MagicMock(return_value=frame(values))),
            "csv": ("_get_load_forecast_csv", MagicMock(return_value=frame(values))),
            "list": ("_get_load_forecast_list", MagicMock(return_value=frame(values))),
        }
        for method, (helper, mock) in cases.items():
            with self.subTest(method=method):
                with (
                    patch.object(self.fcst, helper, mock),
                    patch.object(
                        self.fcst, "_prepare_hass_load_data", AsyncMock(return_value=frame(values))
                    ),
                ):
                    result = await self.fcst.get_load_forecast(method=method)
                self.assertEqual(result.iloc[0], 0.0)
                self.assertEqual(result.iloc[1:].tolist(), values[1:])


class TestCsvLoad(_ForecastFixture):
    async def _csv_load(self, cells):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "load.csv"
            rows = [
                f"{ts.isoformat()},{cell}"
                for ts, cell in zip(self.fcst.forecast_dates, cells, strict=True)
            ]
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            return await self.fcst.get_load_forecast(method="csv", csv_path=path)

    async def test_valid_csv_load_is_unchanged(self):
        cells = [0] + [120.5] * (self.n - 1)
        result = await self._csv_load(cells)
        self.assertEqual(result.tolist(), [float(c) for c in cells])

    async def test_negative_csv_load_is_clipped_by_common_boundary(self):
        cells = [120.5] * self.n
        cells[2] = -8.25
        with self.assertLogs(logger, level="WARNING") as captured:
            result = await self._csv_load(cells)
        self.assertEqual(result.iloc[2], 0.0)
        self.assertEqual(result.drop(result.index[2]).tolist(), [120.5] * (self.n - 1))
        self.assertTrue(any("clipped to 0 W" in line for line in captured.output))

    async def test_invalid_csv_load_fails_cleanly(self):
        for bad in ("nan", "inf", "-inf", "abc", "True"):
            with self.subTest(cell=bad):
                cells = [120.5] * self.n
                cells[6] = bad
                with self.assertLogs(logger, level="ERROR"):
                    result = await self._csv_load(cells)
                self.assertIs(result, False)


class _ControlledEstimator:
    """Deterministic stand-in for the fitted skforecast estimator (no model fit)."""

    def __init__(self, values, index):
        self._predictions = pd.Series(values, index=index, name="pred")

    def predict(self, steps, exog=None, last_window=None):
        return self._predictions.iloc[:steps].copy()


class TestNativeMlLoad(_ForecastFixture):
    def _model(self, values):
        # A real MLForecaster, so MLForecaster.predict() itself is exercised;
        # only the fitted estimator underneath is replaced.
        mlf = MLForecaster.__new__(MLForecaster)
        mlf.forecaster = _ControlledEstimator(values, self.fcst.forecast_dates)
        mlf.num_lags = len(values)
        mlf.var_model = "sensor.power_load_no_var_loads"
        mlf.data_test = pd.DataFrame({mlf.var_model: values, "hour": 0})
        mlf.is_tuned = False
        mlf.weather_features = []
        mlf.logger = logger
        return mlf

    async def _ml_load(self, model):
        return await self.fcst.get_load_forecast(
            method="mlforecaster", use_last_window=False, debug=True, mlf=model
        )

    async def test_positive_ml_load_is_unchanged(self):
        values = [float(200 + i) for i in range(self.n)]
        result = await self._ml_load(self._model(values))
        self.assertEqual(result.tolist(), values)

    async def test_negative_ml_load_is_clipped_but_raw_prediction_stays_raw(self):
        values = [200.0] * self.n
        values[1] = -12.0
        model = self._model(values)

        # Raw model output (forecast-model-predict) is not post-processed.
        input_data_dict = {
            "emhass_conf": EMHASS_CONF,
            "retrieve_hass_conf": self.fcst.retrieve_hass_conf,
            "fcst": self.fcst,
            "params": {
                "passed_data": {
                    "model_type": "controlled",
                    "model_predict_publish": False,
                    "model_predict_entity_id": "sensor.p_load_forecast_custom_model",
                    "model_predict_device_class": "power",
                    "model_predict_unit_of_measurement": "W",
                    "model_predict_friendly_name": "Load Power Forecast custom ML model",
                    "publish_prefix": "",
                }
            },
        }
        raw = await forecast_model_predict(
            input_data_dict, logger, use_last_window=False, debug=True, mlf=model
        )
        self.assertEqual(raw.iloc[1], -12.0)
        self.assertEqual(raw.tolist(), values)

        # The optimizer-facing load is clipped at the physical boundary.
        with self.assertLogs(logger, level="WARNING") as captured:
            result = await self._ml_load(model)
        self.assertEqual(result.iloc[1], 0.0)
        self.assertTrue(any("clipped to 0 W" in line for line in captured.output))
        # ...and the model's own prediction is left untouched.
        self.assertEqual((await model.predict()).iloc[1], -12.0)

    async def test_non_finite_ml_load_fails(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad):
                values = [200.0] * self.n
                values[10] = bad
                with self.assertLogs(logger, level="ERROR"):
                    result = await self._ml_load(self._model(values))
                self.assertIs(result, False)


class TestPostMixBoundary(_ForecastFixture):
    def _df_now(self, live_value):
        index = pd.date_range(
            end=self.fcst.forecast_dates[0], periods=4, freq=self.fcst.freq, tz=self.fcst.time_zone
        )
        return pd.DataFrame(
            {self.fcst.var_load_new: [500.0, 400.0, 300.0, live_value]}, index=index
        )

    async def test_negative_mix_result_is_clipped_after_mixing(self):
        # Pre-mix forecast is valid; the blend with a negative live reading
        # (0.5 * 100 + 0.5 * -1000 = -450) makes the first step negative.
        values = [100.0] * self.n
        with self.assertLogs(logger, level="WARNING") as captured:
            result = await self._list_load(
                values, set_mix_forecast=True, df_now=self._df_now(-1000.0)
            )
        self.assertEqual(result.iloc[0], 0.0)
        self.assertEqual(result.iloc[1:].tolist(), values[1:])
        warnings = [line for line in captured.output if "clipped to 0 W" in line]
        self.assertEqual(len(warnings), 1)
        self.assertIn("-450", warnings[0])

    async def test_non_finite_mix_result_cannot_reach_optimization(self):
        def mix_producing_nan(df_now, df_forecast, *args, **kwargs):
            mixed = df_forecast.copy()
            mixed.iloc[0] = float("nan")
            return mixed

        values = [100.0] * self.n
        with (
            patch.object(Forecast, "get_mix_forecast", side_effect=mix_producing_nan) as mix,
            self.assertLogs(logger, level="ERROR"),
        ):
            result = await self._list_load(
                values, set_mix_forecast=True, df_now=self._df_now(100.0)
            )
        mix.assert_called_once()
        self.assertIs(result, False)

    async def test_non_finite_forecast_does_not_crash_the_mix(self):
        # +/-Inf in the first step used to reach round() inside the blend.
        for bad in (float("inf"), float("-inf")):
            with self.subTest(value=bad):
                values = [100.0] * self.n
                values[0] = bad
                with self.assertLogs(logger, level="ERROR"):
                    result = await self._list_load(
                        values, set_mix_forecast=True, df_now=self._df_now(100.0)
                    )
                self.assertIs(result, False)


class TestSignConventions(_ForecastFixture):
    async def test_external_positive_load_is_not_inverted_by_load_negative(self):
        self.fcst.retrieve_hass_conf["load_negative"] = True
        values = [float(100 + i) for i in range(self.n)]
        result = await self._list_load(values)
        self.assertEqual(result.tolist(), values)

    async def test_load_negative_still_normalizes_retrieved_history(self):
        async def prepared(load_negative, negate):
            params_json = self.params_json
            retrieve_hass_conf, _, _ = utils.get_yaml_parse(params_json, logger)
            rh = RetrieveHass(
                retrieve_hass_conf["hass_url"],
                retrieve_hass_conf["long_lived_token"],
                retrieve_hass_conf["optimization_time_step"],
                retrieve_hass_conf["time_zone"],
                params_json,
                EMHASS_CONF,
                logger,
            )
            with open(EMHASS_CONF["data_path"] / "test_df_final.pkl", "rb") as inp:
                rh.df_final, _, var_list, rh.ha_config = pickle.load(inp)
            rh.var_list = var_list
            if negate:
                rh.df_final[var_list[0]] = -rh.df_final[var_list[0]]
            rh.prepare_data(
                var_list[0],
                load_negative=load_negative,
                set_zero_min=False,
                var_replace_zero=None,
                var_interp=None,
            )
            return rh.df_final[var_list[0] + "_positive"]

        reference = await prepared(load_negative=False, negate=False)
        inverted = await prepared(load_negative=True, negate=True)
        pd.testing.assert_series_equal(inverted, reference)


class TestPvDomainUnchanged(_ForecastFixture):
    async def test_negative_pv_is_still_clipped_by_existing_pv_correction(self):
        self.fcst.params["passed_data"]["pv_power_forecast"] = [-3.0, 0.0] + [800.0] * (self.n - 2)
        self.fcst.params["passed_data"]["pv_power_forecast_p10"] = None
        df_weather = await self.fcst.get_weather_forecast(method="list")
        p_pv = self.fcst.get_power_from_weather(df_weather)
        self.assertEqual(p_pv.tolist(), [0.0, 0.0] + [800.0] * (self.n - 2))


class TestSolverBoundary(unittest.IsolatedAsyncioTestCase):
    """Invalid external forecasts must stop the action before the optimizer."""

    async def asyncSetUp(self):
        params = await _build_params(set_use_pv=True)
        self.params_json = orjson.dumps(params).decode("utf-8")
        # Always build a fresh Optimization so an instance-level mock left on a
        # cached object by another test cannot hide a solver call.
        for target, kwargs in (
            ("get", {"return_value": None}),
            ("put", {}),
        ):
            p = patch.object(OptimizationCache, target, **kwargs)
            p.start()
            self.addCleanup(p.stop)
        for name in ("_record_optim_snapshot", "_log_optimization_summary"):
            p = patch(f"emhass.command_line.{name}")
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def _valid_runtime():
        return {
            "pv_power_forecast": [1000.0] * LONG,
            "load_power_forecast": [500.0] * LONG,
            "load_cost_forecast": [0.2] * LONG,
            "prod_price_forecast": [0.1] * LONG,
            "outdoor_temperature_forecast": [10.0] * LONG,
        }

    async def _run(self, action, runtimeparams):
        solved = pd.DataFrame({"p_grid": [0.0]})
        with (
            patch.object(
                Optimization, "perform_dayahead_forecast_optim", return_value=solved
            ) as dayahead,
            patch.object(Optimization, "perform_naive_mpc_optim", return_value=solved) as mpc,
            patch.object(Optimization, "perform_optimization") as solve,
        ):
            input_data_dict = await set_input_data_dict(
                EMHASS_CONF,
                "profit",
                self.params_json,
                runtimeparams,
                action,
                logger,
                get_data_from_file=True,
            )
            result = input_data_dict
            if input_data_dict:
                action_fn = (
                    dayahead_forecast_optim if action == "dayahead-optim" else naive_mpc_optim
                )
                result = await action_fn(input_data_dict, logger, debug=True)
        solver_calls = dayahead.call_count + mpc.call_count + solve.call_count
        entry = dayahead if action == "dayahead-optim" else mpc
        return result, solver_calls, entry

    async def test_valid_signed_inputs_reach_the_solver(self):
        runtimeparams = self._valid_runtime()
        runtimeparams["load_cost_forecast"] = [-0.05] * LONG
        runtimeparams["prod_price_forecast"] = [-0.12] * LONG
        runtimeparams["outdoor_temperature_forecast"] = [-7.5] * LONG
        for action in ("dayahead-optim", "naive-mpc-optim"):
            with self.subTest(action=action):
                result, calls, entry = await self._run(action, runtimeparams)
                self.assertIsInstance(result, pd.DataFrame)
                self.assertEqual(calls, 1)
                df = entry.call_args[0][0]
                self.assertTrue((df["outdoor_temperature_forecast"] == -7.5).all())
                self.assertTrue((df.filter(like="load_cost").to_numpy() == -0.05).all())
                self.assertTrue((df.filter(like="prod_price").to_numpy() == -0.12).all())

    async def test_invalid_external_forecast_never_reaches_the_solver(self):
        for action in ("dayahead-optim", "naive-mpc-optim"):
            for key in FORECAST_KEYS:
                for label in ("NaN", "+Inf", "bool", "string"):
                    with self.subTest(action=action, key=key, invalid=label):
                        runtimeparams = self._valid_runtime()
                        runtimeparams[key][3] = INVALID_VALUES[label]
                        result, calls, _ = await self._run(action, runtimeparams)
                        self.assertEqual(calls, 0)
                        self.assertFalse(isinstance(result, pd.DataFrame))
                        self.assertFalse(result)

    async def test_invalid_external_pv_pair_never_reaches_the_solver(self):
        runtimeparams = self._valid_runtime()
        runtimeparams["pv_power_forecast_p10"] = [500.0] * LONG
        runtimeparams["pv_power_forecast_p10"][2] = True
        result, calls, _ = await self._run("dayahead-optim", runtimeparams)
        self.assertEqual(calls, 0)
        self.assertFalse(result)

    async def test_negative_external_load_reaches_the_solver_clipped(self):
        runtimeparams = self._valid_runtime()
        runtimeparams["load_power_forecast"][0] = -25.0
        with self.assertLogs(logger, level="WARNING") as captured:
            result, calls, entry = await self._run("dayahead-optim", runtimeparams)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(calls, 1)
        p_load = entry.call_args[0][2]
        self.assertEqual(p_load.iloc[0], 0.0)
        self.assertGreaterEqual(p_load.min(), 0.0)
        self.assertTrue(any("clipped to 0 W" in line for line in captured.output))


class TestWebActionBoundary(unittest.IsolatedAsyncioTestCase):
    """The /action route answers 400, not a crash, when a forecast is rejected."""

    async def asyncSetUp(self):
        params = await _build_params(set_use_pv=True)
        self.params_json = orjson.dumps(params).decode("utf-8")
        real_set_input = command_line.set_input_data_dict

        async def from_file(*args, **kwargs):
            return await real_set_input(*args, get_data_from_file=True, **kwargs)

        patches = [
            patch.object(web_server, "emhass_conf", dict(EMHASS_CONF)),
            patch.object(web_server, "set_input_data_dict", side_effect=from_file),
            patch.object(web_server, "_save_injection_dict", AsyncMock()),
            patch.object(OptimizationCache, "get", return_value=None),
            patch.object(OptimizationCache, "put"),
            patch("emhass.command_line._record_optim_snapshot"),
            patch("emhass.command_line._log_optimization_summary"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    async def _post(self, action, runtimeparams):
        with (
            patch.object(
                web_server,
                "_load_params_and_runtime",
                AsyncMock(
                    return_value=(
                        self.params_json,
                        "profit",
                        orjson.dumps(runtimeparams).decode("utf-8"),
                    )
                ),
            ),
            patch.object(Optimization, "perform_dayahead_forecast_optim") as dayahead,
            patch.object(Optimization, "perform_naive_mpc_optim") as mpc,
        ):
            response = await web_server.app.test_client().post(f"/action/{action}", json={})
        return response.status_code, dayahead.call_count + mpc.call_count

    async def test_rejected_forecast_returns_400_without_solving(self):
        # load: rejected while the input data is set up; price/temperature:
        # rejected inside the optimization action itself.
        for action in ("dayahead-optim", "naive-mpc-optim"):
            for key in (
                "load_power_forecast",
                "load_cost_forecast",
                "outdoor_temperature_forecast",
            ):
                with self.subTest(action=action, key=key):
                    runtimeparams = TestSolverBoundary._valid_runtime()
                    runtimeparams[key][3] = True
                    status, solver_calls = await self._post(action, runtimeparams)
                    self.assertEqual(solver_calls, 0)
                    self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
