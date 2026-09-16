#!/usr/bin/env python

import json
import logging
import pathlib
import unittest
from unittest.mock import patch

import numpy as np
import orjson
import pandas as pd

from emhass import utils, web_server
from emhass.command_line import pv_bias_calibration
from emhass.forecast import Forecast

logger = logging.getLogger("test_external_pv_p10")

ROOT = pathlib.Path(utils.get_root(__file__, num_parent=2))
EMHASS_CONF = {
    "data_path": ROOT / "data/",
    "root_path": ROOT / "src/emhass/",
    "defaults_path": ROOT / "src/emhass/data/config_defaults.json",
    "associations_path": ROOT / "src/emhass/data/associations.csv",
}


async def _runtime_context():
    config = json.loads(EMHASS_CONF["defaults_path"].read_text(encoding="utf-8"))
    _, secrets = await utils.build_secrets(EMHASS_CONF, logger, no_response=True)
    params = await utils.build_params(EMHASS_CONF, secrets, config, logger)
    if params is False:
        raise AssertionError("build_params failed")
    params_json = orjson.dumps(params).decode("utf-8")
    retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
    return params_json, retrieve_hass_conf, optim_conf, plant_conf


async def _treat_runtime(runtimeparams):
    params_json, retrieve_hass_conf, optim_conf, plant_conf = await _runtime_context()
    treated, retrieve_hass_conf, optim_conf, plant_conf = await utils.treat_runtimeparams(
        orjson.dumps(runtimeparams).decode("utf-8"),
        params_json,
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        "dayahead-optim",
        logger,
        EMHASS_CONF,
    )
    return orjson.loads(treated), retrieve_hass_conf, optim_conf, plant_conf


class TestExternalPvP10(unittest.TestCase):
    @staticmethod
    def _forecast(p50, p10, bias):
        fcst = Forecast.__new__(Forecast)
        fcst.params = {
            "passed_data": {
                "pv_power_forecast": p50,
                "pv_power_forecast_p10": p10,
                "prediction_horizon": None,
            }
        }
        fcst.optim_conf = {"weather_forecast_pv_quantile_bias": bias}
        fcst.forecast_dates_tz = list(
            pd.date_range("2026-09-16T00:00:00+00:00", periods=len(p50), freq="30min")
        )
        fcst.logger = logger
        return fcst

    def test_list_external_p10_uses_existing_bias_formula(self):
        fcst = self._forecast([100.0, 200.0, 300.0], [40.0, 80.0, 120.0], 0.5)
        result = fcst._get_weather_list()
        self.assertEqual(result["yhat"].tolist(), [70.0, 140.0, 210.0])

    def test_bias_zero_keeps_external_p50_exactly(self):
        p50 = [100, 200, 300]
        fcst = self._forecast(p50, [40.0, 80.0, 120.0], 0.0)
        result = fcst._get_weather_list()
        self.assertEqual(result["yhat"].tolist(), p50)

    def test_omitting_external_p10_keeps_existing_p50_path(self):
        p50 = [100.0, 200.0, 300.0]
        fcst = self._forecast(p50, None, 1.0)
        result = fcst._get_weather_list()
        self.assertEqual(result["yhat"].tolist(), p50)

    def test_invalid_external_p10_fails_even_when_bias_zero(self):
        fcst = self._forecast([100.0, 200.0, 300.0], [40.0, 80.0], 0.0)
        self.assertIsNone(fcst._get_weather_list())

    def test_non_finite_external_p10_fails(self):
        fcst = self._forecast([100.0, 200.0, 300.0], [40.0, float("nan"), 120.0], 0.5)
        self.assertIsNone(fcst._get_weather_list())


class TestExternalPvP10Runtime(unittest.IsolatedAsyncioTestCase):
    async def test_list_pair_flows_through_public_runtime_boundary(self):
        p50 = [100.0 + i for i in range(96)]
        p10 = [40.0 + i for i in range(96)]
        params, _, optim_conf, _ = await _treat_runtime(
            {
                "pv_power_forecast": p50,
                "pv_power_forecast_p10": p10,
                "weather_forecast_pv_quantile_bias": 0.5,
            }
        )

        aligned_p50 = params["passed_data"]["pv_power_forecast"]
        aligned_p10 = params["passed_data"]["pv_power_forecast_p10"]
        self.assertEqual(optim_conf["weather_forecast_method"], "list")
        self.assertEqual(optim_conf["weather_forecast_pv_quantile_bias"], 0.5)
        self.assertEqual(aligned_p50, p50[: len(aligned_p50)])
        self.assertEqual(aligned_p10, p10[: len(aligned_p10)])

        fcst = Forecast.__new__(Forecast)
        fcst.params = params
        fcst.optim_conf = optim_conf
        fcst.forecast_dates_tz = list(
            pd.date_range(
                "2026-09-16T00:00:00+00:00",
                periods=len(aligned_p50),
                freq="30min",
            )
        )
        fcst.logger = logger
        result = fcst._get_weather_list()["yhat"].to_numpy()
        expected = 0.5 * np.asarray(aligned_p10) + 0.5 * np.asarray(aligned_p50)
        np.testing.assert_allclose(result, expected)

    async def test_timestamped_pair_reuses_mapping_alignment_and_leading_backfill(self):
        params_json, retrieve_hass_conf, optim_conf, plant_conf = await _runtime_context()
        step_minutes = int(retrieve_hass_conf["optimization_time_step"].total_seconds() / 60)
        forecast_dates = utils.get_forecast_dates(
            step_minutes,
            optim_conf["delta_forecast_daily"].days,
            retrieve_hass_conf["time_zone"],
        )
        keys = [str(forecast_dates[i]) for i in (2, 4, 6)]
        payload = {
            "pv_power_forecast": dict(zip(keys, [100.0, 200.0, 300.0])),
            "pv_power_forecast_p10": dict(zip(keys, [50.0, 100.0, 150.0])),
        }

        treated, _, optim_conf, _ = await utils.treat_runtimeparams(
            orjson.dumps(payload).decode("utf-8"),
            params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "dayahead-optim",
            logger,
            EMHASS_CONF,
        )
        params = orjson.loads(treated)
        p50 = params["passed_data"]["pv_power_forecast"]
        p10 = params["passed_data"]["pv_power_forecast_p10"]

        self.assertEqual(optim_conf["weather_forecast_method"], "list")
        self.assertEqual(p50[:7], [100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 300.0])
        self.assertEqual(p10[:7], [50.0, 50.0, 50.0, 50.0, 100.0, 100.0, 150.0])

    async def test_timestamp_mismatch_is_rejected_at_public_boundary(self):
        params_json, retrieve_hass_conf, optim_conf, plant_conf = await _runtime_context()
        step_minutes = int(retrieve_hass_conf["optimization_time_step"].total_seconds() / 60)
        forecast_dates = utils.get_forecast_dates(
            step_minutes,
            optim_conf["delta_forecast_daily"].days,
            retrieve_hass_conf["time_zone"],
        )
        p50_keys = [str(forecast_dates[i]) for i in (0, 2)]
        p10_keys = [str(forecast_dates[i]) for i in (0, 3)]
        payload = {
            "pv_power_forecast": dict(zip(p50_keys, [100.0, 200.0])),
            "pv_power_forecast_p10": dict(zip(p10_keys, [50.0, 100.0])),
        }

        with self.assertLogs(logger, level="ERROR") as captured:
            treated, _, optim_conf, _ = await utils.treat_runtimeparams(
                orjson.dumps(payload).decode("utf-8"),
                params_json,
                retrieve_hass_conf,
                optim_conf,
                plant_conf,
                "dayahead-optim",
                logger,
                EMHASS_CONF,
            )

        params = orjson.loads(treated)
        self.assertEqual(optim_conf["weather_forecast_method"], "list")
        self.assertIsNone(params["passed_data"]["pv_power_forecast"])
        self.assertEqual(params["passed_data"]["pv_power_forecast_p10"], [])
        self.assertTrue(any("timestamp mismatch" in line for line in captured.output))

    async def test_list_length_and_invalid_numeric_faults_are_explicit(self):
        cases = [
            (
                {
                    "pv_power_forecast": [1.0] * 96,
                    "pv_power_forecast_p10": [0.5] * 95,
                },
                "length mismatch",
            ),
            (
                {
                    "pv_power_forecast": [1.0] * 96,
                    "pv_power_forecast_p10": [0.5] * 95 + [None],
                },
                "non-finite",
            ),
        ]
        for payload, expected in cases:
            with self.subTest(expected=expected):
                with self.assertLogs(logger, level="ERROR") as captured:
                    params, _, _, _ = await _treat_runtime(payload)
                self.assertIsNone(params["passed_data"]["pv_power_forecast"])
                self.assertEqual(params["passed_data"]["pv_power_forecast_p10"], [])
                self.assertTrue(any(expected in line for line in captured.output))


class TestPvBiasCalibrationAction(unittest.IsolatedAsyncioTestCase):
    async def test_issue_1128_acceptance_c_counts_are_reported_honestly(self):
        # Synthetic contract fixture reproducing the published #1128 counts.
        # This is not the retained private HA/BJReplay dataset.
        n_observations = 7720
        n_below_p10 = 4296
        payload = {
            "p10": [80.0] * n_observations,
            "p50": [100.0] * n_observations,
            "actual": [70.0] * n_below_p10 + [90.0] * (n_observations - n_below_p10),
            "target_shortfall_rate": 0.10,
        }

        result = await pv_bias_calibration(orjson.dumps(payload).decode("utf-8"), logger)

        self.assertEqual(result["n_observations"], n_observations)
        self.assertEqual(result["n_curtailed_excluded"], 0)
        self.assertEqual(result["feasible_shortfall_range"][0], 0.5565)
        self.assertFalse(result["target_feasible"])
        self.assertFalse(result["converged"])
        self.assertIn("recommended_bias", result)
        self.assertIn("achieved_shortfall_rate", result)

    async def test_action_requires_independent_curtailment_signal(self):
        base = {
            "p10": [80.0, 80.0, 80.0],
            "p50": [100.0, 100.0, 100.0],
            "actual": [0.0, 90.0, 90.0],
        }
        unmasked = await pv_bias_calibration(base, logger)
        masked = await pv_bias_calibration({**base, "curtailed": [True, False, False]}, logger)

        self.assertEqual(unmasked["n_curtailed_excluded"], 0)
        self.assertEqual(unmasked["n_observations"], 3)
        self.assertEqual(masked["n_curtailed_excluded"], 1)
        self.assertEqual(masked["n_observations"], 2)


class TestPvBiasCalibrationWebRoute(unittest.IsolatedAsyncioTestCase):
    @patch("emhass.web_server.check_file_log")
    @patch("emhass.web_server.set_input_data_dict")
    @patch("emhass.web_server._load_params_and_runtime")
    async def test_route_returns_engine_result_without_input_data_setup(
        self, mock_load, mock_set_input, mock_check_log
    ):
        payload = {
            "p10": [80.0, 80.0, 80.0],
            "p50": [100.0, 100.0, 100.0],
            "actual": [70.0, 90.0, 90.0],
        }
        mock_load.return_value = ({}, "profit", orjson.dumps(payload).decode("utf-8"))
        mock_check_log.return_value = False

        response = await web_server.app.test_client().post("/action/pv-bias-calibration", json={})
        body = await response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertIn("recommended_bias", body)
        self.assertIn("target_feasible", body)
        mock_set_input.assert_not_called()

    @patch("emhass.web_server.check_file_log")
    @patch("emhass.web_server.set_input_data_dict")
    @patch("emhass.web_server._load_params_and_runtime")
    async def test_route_returns_400_for_missing_history_without_input_data_setup(
        self, mock_load, mock_set_input, mock_check_log
    ):
        mock_load.return_value = (
            {},
            "profit",
            orjson.dumps({"p10": [1.0]}).decode("utf-8"),
        )

        response = await web_server.app.test_client().post("/action/pv-bias-calibration", json={})
        body = await response.get_json()

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", body)
        mock_set_input.assert_not_called()
        mock_check_log.assert_not_awaited()
