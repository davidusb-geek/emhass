#!/usr/bin/env python
"""End-to-end tests for two inverters through the real actions (#1150, PR 3b part 2).

``set_input_data_dict`` plus ``dayahead_forecast_optim`` / ``naive_mpc_optim`` /
``perfect_forecast_optim``, with one runtime PV list per PV inverter and no network: the
per-inverter result columns, the site totals, the PV-only clip, the hybrid spill, a
per-inverter ``compute_curtailment`` and the clear perfect-optim refusal.
"""

import copy
import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import orjson

from emhass import last_run, plan_store, utils
from emhass.command_line import (
    dayahead_forecast_optim,
    naive_mpc_optim,
    perfect_forecast_optim,
    set_input_data_dict,
)
from tests import test_command_line_utils as cl_tests
from tests.fixture_clock import fixture_clock

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
logger, ch = utils.get_logger(__name__, cl_tests.emhass_conf, save_to_file=False)

COSTFUN = "profit"
HORIZON = 10
EFF = 0.95  # DC to AC of both inverters
CAP_HYBRID = 5000
CAP_STRING = 3000
# inverter 0: hybrid with the battery; inverter 1: PV-only string inverter
TWO_INVERTERS = {
    "number_of_inverters": 2,
    "inverter_ac_output_max": [CAP_HYBRID, CAP_STRING],
    "inverter_ac_input_max": [CAP_HYBRID, CAP_STRING],
    "inverter_efficiency_dc_ac": [EFF, EFF],
    "inverter_efficiency_ac_dc": [EFF, EFF],
    "inverter_power_curve_dc_ac": [],
    "inverter_power_curve_ac_dc": [],
    "inverter_stress_cost": [0.0, 0.0],
    "compute_curtailment": [False, False],
    "battery_inverter_index": [0],
    "pv_plant_inverter_index": [0, 1],
}


def _inverter_columns(frame):
    return sorted(
        c
        for c in frame.columns
        if str(c).startswith(("P_PV_", "P_hybrid_inverter_", "P_inverter_"))
        and not str(c).startswith("P_PV_plant")
    )


class MultiInverterActionBase(unittest.IsolatedAsyncioTestCase):
    """A two-inverter site run through the real actions on a throwaway data directory."""

    async def asyncSetUp(self):
        data_path = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        shutil.copytree(root / "data", data_path, dirs_exist_ok=True)
        self.emhass_conf = {**cl_tests.emhass_conf, "data_path": data_path}
        # The plan and last-run caches are module globals: do not leak ours into other tests.
        self.enterContext(patch.object(plan_store, "_cache", None))
        self.enterContext(patch.object(last_run, "_cache", None))

    async def params(self, plant=None, optim=None, inverters=2):
        params = copy.deepcopy(await cl_tests.TestCommandLineAsyncUtils.get_test_params(True))
        params["optim_conf"]["set_use_battery"] = True
        params["optim_conf"].update(optim or {})
        if inverters == 2:
            params["plant_conf"].update(copy.deepcopy(TWO_INVERTERS))
        params["plant_conf"].update(plant or {})
        return params

    async def build(self, action, runtime, params=None):
        params = params if params is not None else await self.params()
        params["passed_data"] = runtime
        return await set_input_data_dict(
            self.emhass_conf,
            COSTFUN,
            orjson.dumps(params).decode("utf-8"),
            orjson.dumps(runtime).decode("utf-8"),
            action,
            logger,
            get_data_from_file=True,
        )

    @staticmethod
    def runtime(n, pv, mpc=False, **extra):
        """Runtime lists of n steps; ``pv`` is one list (or constant) per PV inverter."""
        runtime = {
            "pv_power_forecast": [[p] * n if np.isscalar(p) else list(p) for p in pv],
            "load_power_forecast": [400.0 + 10 * (i % 5) for i in range(n)],
            "load_cost_forecast": [0.25] * n,
            "prod_price_forecast": [0.05] * n,
            **extra,
        }
        if mpc:
            runtime.update({"prediction_horizon": n, "soc_init": 0.5, "soc_final": 0.5})
        return runtime

    async def run_action(self, action, runtime, params=None):
        idd = await self.build(action, runtime, params)
        if action == "dayahead-optim":
            return await dayahead_forecast_optim(idd, logger, debug=True)
        return await naive_mpc_optim(idd, logger, debug=True)

    def assert_two_inverter_result(self, res):
        self.assertTrue((res["optim_status"] == "Optimal").all(), res["optim_status"].unique())
        for name in ("P_PV_0", "P_PV_1", "P_hybrid_inverter_0", "P_hybrid_inverter_1"):
            self.assertIn(name, res.columns)
        np.testing.assert_allclose(res["P_PV"], res["P_PV_0"] + res["P_PV_1"], atol=1e-4)
        np.testing.assert_allclose(
            res["P_hybrid_inverter"],
            res["P_hybrid_inverter_0"] + res["P_hybrid_inverter_1"],
            atol=1e-4,
        )
        self.assertIn("P_PV_spill_0", res.columns)
        self.assertNotIn("P_PV_spill_1", res.columns)
        self.assertNotIn("P_PV_curtailment_0", res.columns)


class TestTwoInvertersThroughTheActions(MultiInverterActionBase):
    @fixture_clock()
    async def test_dayahead_optim(self):
        # string PV 4 kW is above its 3 kW cap / 0.95 efficiency: it is clipped before the solve
        runtime = self.runtime(48, [2500.0, 4000.0])
        res = await self.run_action("dayahead-optim", runtime)
        self.assertEqual(len(res), 48)
        self.assert_two_inverter_result(res)
        limit = CAP_STRING / EFF
        np.testing.assert_allclose(res["P_PV_1"], limit, atol=1e-3)
        np.testing.assert_allclose(res["P_PV_0"], 2500.0, atol=1e-3)

    @fixture_clock()
    async def test_naive_mpc_optim(self):
        runtime = self.runtime(HORIZON, [2500.0, 4000.0], mpc=True)
        res = await self.run_action("naive-mpc-optim", runtime)
        self.assertEqual(len(res), HORIZON)
        self.assert_two_inverter_result(res)
        self.assertLessEqual(res["P_PV_1"].max(), CAP_STRING / EFF + 1e-3)
        self.assertGreater(res["P_PV_1"].max(), CAP_STRING / EFF - 1e-3)

    @fixture_clock()
    async def test_the_hybrid_spills_when_nothing_else_can_take_its_pv(self):
        # 9 kW PV on a 5 kW hybrid, battery already full, no export, 0.4 kW load
        params = await self.params(plant={"maximum_power_to_grid": 0})
        runtime = self.runtime(HORIZON, [9000.0, 0.0], mpc=True, soc_init=0.9, soc_final=0.9)
        res = await self.run_action("naive-mpc-optim", runtime, params)
        self.assert_two_inverter_result(res)
        self.assertTrue((res["P_PV_spill_0"] > 1.0).all(), res["P_PV_spill_0"].tolist())
        np.testing.assert_array_less(res["P_grid_neg"].abs(), 1e-3)

    @fixture_clock()
    async def test_compute_curtailment_per_inverter(self):
        params = await self.params(plant={"compute_curtailment": [True, False]})
        res = await self.run_action(
            "dayahead-optim", self.runtime(48, [2500.0, 2000.0]), params=params
        )
        self.assertTrue((res["optim_status"] == "Optimal").all())
        self.assertIn("P_PV_curtailment_0", res.columns)
        self.assertNotIn("P_PV_curtailment_1", res.columns)
        # a curtailing hybrid does not need the spill variable
        self.assertNotIn("P_PV_spill_0", res.columns)

    @fixture_clock()
    async def test_one_inverter_has_no_per_inverter_columns(self):
        params = await self.params(inverters=1)
        runtime = self.runtime(48, [3000.0])
        runtime["pv_power_forecast"] = runtime["pv_power_forecast"][0]
        res = await self.run_action("dayahead-optim", runtime, params)
        self.assertTrue((res["optim_status"] == "Optimal").all())
        self.assertEqual(_inverter_columns(res), [])
        self.assertIn("P_PV", res.columns)


class TestPerfectOptimAboveOneInverter(MultiInverterActionBase):
    @fixture_clock()
    async def test_the_user_gets_the_clear_refusal(self):
        params = await self.params()
        runtime = {
            "load_cost_forecast": [0.25] * 48,
            "prod_price_forecast": [0.05] * 48,
        }
        idd = await self.build("perfect-optim", runtime, params)
        self.assertIsInstance(idd, dict)
        with self.assertRaisesRegex(
            utils.InverterConfigError, "perfect-optim has no PV history per inverter"
        ):
            await perfect_forecast_optim(idd, logger, debug=True)
