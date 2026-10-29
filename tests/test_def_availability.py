#!/usr/bin/env python
"""Tests for per-deferrable-load availability (``def_availability``): a fraction
in [0, 1] per timestep that scales a deferrable load's power ceiling, with ``0``
as a hard "must not run". Same shape and validation as ``battery_availability``.

Synthetic and self-contained, built on the multi-battery test helper.
"""

import unittest

import numpy as np
import pandas as pd

from emhass import utils
from tests.test_multi_battery_optimization import (
    VALID_OPTIMAL_STATUSES,
    build_optimization,
)

N = 12  # 6 h at 30 min

TWO_LOADS = {
    "set_use_battery": False,
    "number_of_deferrable_loads": 2,
    "nominal_power_of_deferrable_loads": [2000.0, 1000.0],
    "minimum_power_of_deferrable_loads": [0.0, 0.0],
    "treat_deferrable_load_as_semi_cont": [False, False],
    "set_deferrable_load_single_constant": [False, False],
    "set_deferrable_startup_penalty": [0.0, 0.0],
    "operating_hours_of_each_deferrable_load": [2, 1],
    "start_timesteps_of_each_deferrable_load": [0, 0],
    "end_timesteps_of_each_deferrable_load": [0, 0],
}


def scenario(n=N):
    """Cheapest in the first four steps, so an unrestricted load runs there."""
    index = pd.date_range("2026-11-01", periods=n, freq="30min", tz="Europe/Brussels")
    p_pv = pd.Series([0.0] * n, index=index)
    p_load = pd.Series([500.0] * n, index=index)
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = [0.05] * 4 + [0.30] * (n - 4)
    df_input["unit_prod_price"] = [0.0] * n
    df_input["outdoor_temperature_forecast"] = [5.0] * n
    return p_pv, p_load, df_input


def run(opt, n=N, steps=(4, 2), **kwargs):
    """One naive-MPC run; ``steps`` = operating timesteps per load."""
    p_pv, p_load, df_input = scenario(n)
    loads = opt.optim_conf["number_of_deferrable_loads"]
    return opt.perform_naive_mpc_optim(
        df_input,
        p_pv,
        p_load,
        n,
        def_total_hours=[0] * loads,
        def_total_timestep=list(steps)[:loads],
        def_start_timestep=[0] * loads,
        def_end_timestep=[0] * loads,
        **kwargs,
    )


def loads(**overrides):
    conf = dict(TWO_LOADS)
    conf.update(overrides)
    return build_optimization(optim_overrides=conf)


class TestDefAvailability(unittest.TestCase):
    def test_default_is_unchanged(self):
        """No availability input = identical objective and schedule."""
        a = loads()
        res_a = run(a)
        b = loads()
        res_b = run(b, def_availability=[1, [1.0] * N])
        self.assertIn(a.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertAlmostEqual(a.prob.value, b.prob.value, places=6)
        np.testing.assert_allclose(res_a["P_deferrable0"], res_b["P_deferrable0"], atol=1e-6)

    def test_blocked_steps_carry_no_power(self):
        """The load moves out of the cheap steps it may not use; the other load
        is not affected."""
        free = loads()
        p_free = run(free)["P_deferrable0"].to_numpy()
        self.assertGreater(p_free[:4].sum(), 1.0)
        opt = loads()
        res = run(opt, def_availability=[[0] * 4 + [1] * (N - 4), None])
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        p0 = res["P_deferrable0"].to_numpy()
        np.testing.assert_allclose(p0[:4], 0.0, atol=1e-6)
        self.assertAlmostEqual(p0.sum() * 0.5, 2000.0 * 2, delta=1.0)
        self.assertGreater(res["P_deferrable1"].to_numpy()[:4].sum(), 1.0)

    def test_fraction_scales_the_ceiling(self):
        opt = loads()
        res = run(opt, def_availability=[0.5, None])
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        p0 = res["P_deferrable0"].to_numpy()
        self.assertTrue((p0 <= 0.5 * 2000.0 + 1e-3).all())
        self.assertAlmostEqual(p0.sum() * 0.5, 2000.0 * 2, delta=1.0)

    def test_semi_continuous_load_is_off_below_full_availability(self):
        """A semi-continuous load runs at its ON level or not at all, so a
        fraction below 1 blocks the step."""
        opt = loads(treat_deferrable_load_as_semi_cont=[True, False])
        res = run(opt, def_availability=[[0.5] * 4 + [1] * (N - 4), None])
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        p0 = res["P_deferrable0"].to_numpy()
        np.testing.assert_allclose(p0[:4], 0.0, atol=1e-6)
        self.assertAlmostEqual(p0.sum() * 0.5, 2000.0 * 2, delta=1.0)

    def test_running_load_with_min_on_remainder_is_released(self):
        """A load that is on and still owes minimum on-time is forced on at the
        start of the horizon; a blocked step ends that run instead of making the
        MILP infeasible."""
        opt = loads(
            treat_deferrable_load_as_semi_cont=[True, False],
            def_current_state=[True, False],
            def_minimum_on_time=[6, 0],
            def_current_on_timesteps=[1, 0],
        )
        res = run(opt, def_availability=[[1, 1, 0] + [1] * (N - 3), None])
        self.assertEqual(opt.optim_status, "Optimal")
        p0 = res["P_deferrable0"].to_numpy()
        self.assertEqual(p0[2], 0.0)
        np.testing.assert_allclose(opt.param_running_lb[0].value[2:], 0.0)

    def test_running_single_constant_load_is_released(self):
        opt = loads(
            set_deferrable_load_single_constant=[True, False],
            treat_deferrable_load_as_semi_cont=[True, False],
            def_current_state=[True, False],
        )
        res = run(opt, steps=(6, 2), def_availability=[[0] + [1] * (N - 1), None])
        self.assertEqual(opt.optim_status, "Optimal")
        p0 = res["P_deferrable0"].to_numpy()
        self.assertEqual(p0[0], 0.0)
        self.assertAlmostEqual(p0.sum() * 0.5, 2000.0 * 3, delta=1.0)
        self.assertEqual(opt.param_already_running_sc[0].value, 0.0)

    def test_current_power_pin_above_the_ceiling_is_dropped(self):
        opt = loads(def_current_power=[1500.0, 0.0])
        res = run(opt, def_availability=[[0] + [1] * (N - 1), None])
        self.assertEqual(opt.optim_status, "Optimal")
        self.assertEqual(res["P_deferrable0"].to_numpy()[0], 0.0)
        self.assertEqual(opt.param_def_current_power_active[0].value, 0.0)

    def test_mask_change_on_the_same_problem_is_honoured(self):
        """The mask is a Parameter: a new value on a reused (cached) problem
        applies without a rebuild, and no input restores full availability."""
        opt = loads()
        run(opt)
        prob = opt.prob
        res = run(opt, def_availability=[[0] * 4 + [1] * (N - 4), None])
        self.assertIs(opt.prob, prob)
        np.testing.assert_allclose(res["P_deferrable0"].to_numpy()[:4], 0.0, atol=1e-6)
        res = run(opt)
        self.assertIs(opt.prob, prob)
        self.assertGreater(res["P_deferrable0"].to_numpy()[:4].sum(), 1.0)
        np.testing.assert_allclose(opt.param_window_masks[0].value, 1.0)

    def test_wrong_length_raises(self):
        opt = loads()
        with self.assertRaises(ValueError):
            run(opt, def_availability=[1, 1, 1])

    def test_invalid_values_raise(self):
        """A percentage, a negative value or text are rejected instead of
        silently clipped."""
        cases = ([None, 50], [None, [1] * 6 + [-1] * 6], [None, "on"])
        for value in cases:
            with self.subTest(def_availability=value):
                opt = loads()
                with self.assertRaises(ValueError):
                    run(opt, def_availability=value)

    def test_single_load_flat_series(self):
        opt = loads(
            number_of_deferrable_loads=1,
            nominal_power_of_deferrable_loads=[2000.0],
            minimum_power_of_deferrable_loads=[0.0],
            treat_deferrable_load_as_semi_cont=[False],
            set_deferrable_load_single_constant=[False],
            set_deferrable_startup_penalty=[0.0],
            operating_hours_of_each_deferrable_load=[2],
            start_timesteps_of_each_deferrable_load=[0],
            end_timesteps_of_each_deferrable_load=[0],
        )
        res = run(opt, def_availability=[0] * 4 + [1] * (N - 4))
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        p0 = res["P_deferrable0"].to_numpy()
        # read as a time series, not as one entry per load
        np.testing.assert_allclose(p0[:4], 0.0, atol=1e-6)
        self.assertGreater(p0[4:].sum(), 1.0)

    def test_dayahead_after_masked_mpc_is_unrestricted(self):
        """Same object (the cached production case): a dayahead run after an
        MPC run with a blocked load must not inherit the mask."""
        opt = loads()
        run(opt, def_availability=[0, None])
        p_pv, p_load, df_input = scenario()
        res = opt.perform_dayahead_forecast_optim(df_input, p_pv, p_load)
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertGreater(res["P_deferrable0"].sum(), 1.0)


class TestHeatSourceAvailability(unittest.TestCase):
    """A heat_topology source is a deferrable load, so the same key blocks it.
    This is the case that motivated the key (#539): a source that must not run
    (manual mode, outside its operating range, fuel not released) used to be
    blocked by a prohibitive price on its cost_track, and a zone that starts
    below its floor gets its floors priced at SHARED_TANK_START_RECOVERY_PENALTY,
    which outbids any such price."""

    LOCK = 9.99  # currency/kWh, roughly 170x the real price

    def topology(self):
        def source(sid, power, track):
            return {
                "id": sid,
                "type": "constant_efficiency",
                "efficiency": 1.0,
                "electric": False,
                "nominal_power": power,
                "min_power": power,
                "treat_as_semi_cont": True,
                "operating_hours": 0,
                "cost_track": track,
            }

        return {
            "sources": [source("locked", 2000, "t_locked"), source("free", 700, "t_free")],
            "storage": [
                {
                    "id": "zone",
                    "thermal_mass": 2.0,
                    "loss_coefficient": 0.03,
                    "start_temperature": 19.0,  # 1 K below its floor: recovery
                    "min_temperature": [20.0] * N,
                    "max_temperature": [24.0] * N,
                    "comfort_sense": "heat",
                }
            ],
            "flows": [{"from": "locked", "to": "zone"}, {"from": "free", "to": "zone"}],
            "cost_tracks": {"t_locked": [self.LOCK] * N, "t_free": [0.157] * N},
        }

    def solve(self, **kwargs):
        opt = build_optimization(
            optim_overrides={
                "set_use_battery": False,
                **utils.compile_heat_topology(self.topology()),
            }
        )
        p_pv, p_load, df_input = scenario()
        res = opt.perform_naive_mpc_optim(
            df_input,
            p_pv,
            p_load,
            N,
            def_total_hours=[0, 0],
            def_total_timestep=[0, 0],
            def_start_timestep=[0, 0],
            def_end_timestep=[0, 0],
            **kwargs,
        )
        return opt, res

    def test_blocked_source_is_not_bought_during_start_recovery(self):
        opt, res = self.solve()
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        # the price alone does not hold the lock while the zone recovers
        self.assertGreater(res["P_deferrable0"].sum(), 1.0)
        n_bool = sum(v.size for v in opt.prob.variables() if v.attributes.get("boolean"))

        opt, res = self.solve(def_availability=[0, None])
        self.assertEqual(opt.optim_status, "Optimal")
        np.testing.assert_allclose(res["P_deferrable0"].to_numpy(), 0.0, atol=1e-6)
        # the free source does the recovery instead
        self.assertGreater(res["P_deferrable1"].sum(), 1.0)
        # a parameter, not a structural change: no extra binaries
        self.assertEqual(
            sum(v.size for v in opt.prob.variables() if v.attributes.get("boolean")), n_bool
        )


if __name__ == "__main__":
    unittest.main()
