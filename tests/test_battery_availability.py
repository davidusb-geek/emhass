#!/usr/bin/env python
"""Tests for per-battery availability windows (EV / V2G battery that is not
always connected): ``battery_availability``, per-battery ``soc_target`` at
departure and ``battery_discharge_energy_max``.

Synthetic and self-contained, built on the multi-battery test helper.
"""

import unittest

import numpy as np
import pandas as pd

from tests.test_multi_battery_optimization import (
    VALID_OPTIMAL_STATUSES,
    build_optimization,
)

N = 12  # 6 h at 30 min

HOUSE_AND_EV = {
    "number_of_batteries": 2,
    # battery 0 = stationary house battery, battery 1 = EV on a DC charger
    "battery_nominal_energy_capacity": [10000, 60000],
    "battery_discharge_power_max": [5000, 7000],
    "battery_charge_power_max": [5000, 11000],
    "battery_discharge_efficiency": [0.95, 0.95],
    "battery_charge_efficiency": [0.95, 0.95],
    "battery_minimum_state_of_charge": [0.1, 0.2],
    "battery_maximum_state_of_charge": [1.0, 0.9],
    "battery_target_state_of_charge": [0.5, 0.5],
}


def scenario():
    """Cheap first half, expensive second half: an available EV is worth
    charging early and discharging late."""
    index = pd.date_range("2026-11-01", periods=N, freq="30min", tz="Europe/Brussels")
    p_pv = pd.Series([0.0] * N, index=index)
    p_load = pd.Series([1500.0] * N, index=index)
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = [0.05] * 6 + [0.45] * 6
    df_input["unit_prod_price"] = [0.01] * 6 + [0.40] * 6
    return p_pv, p_load, df_input


def run(opt, **kwargs):
    p_pv, p_load, df_input = scenario()
    kwargs.setdefault("soc_init", [0.5, 0.5])
    kwargs.setdefault("soc_final", [0.5, 0.5])
    res = opt.perform_naive_mpc_optim(
        df_input,
        p_pv,
        p_load,
        N,
        def_total_hours=[],
        def_total_timestep=[],
        def_start_timestep=[],
        def_end_timestep=[],
        **kwargs,
    )
    return res


class TestBatteryAvailability(unittest.TestCase):
    def test_default_is_unchanged(self):
        """No availability input = identical objective and schedule."""
        a = build_optimization(plant_overrides=HOUSE_AND_EV)
        res_a = run(a)
        b = build_optimization(plant_overrides=HOUSE_AND_EV)
        res_b = run(b, battery_availability=[1, [1.0] * N], battery_discharge_energy_max=None)
        self.assertIn(a.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertAlmostEqual(a.prob.value, b.prob.value, places=6)
        np.testing.assert_allclose(res_a["P_batt_1"], res_b["P_batt_1"], atol=1e-6)

    def test_unavailable_steps_carry_no_power_and_freeze_soc(self):
        opt = build_optimization(plant_overrides=HOUSE_AND_EV)
        mask = [1] * 4 + [0] * (N - 4)
        res = run(opt, battery_availability=[None, mask])
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        np.testing.assert_allclose(res["P_batt_1"].to_numpy()[4:], 0.0, atol=1e-6)
        soc = res["SOC_opt_1"].to_numpy()
        np.testing.assert_allclose(soc[4:], soc[3], atol=1e-6)
        # the house battery is not affected by the EV mask
        self.assertGreater(np.abs(res["P_batt_0"].to_numpy()).max(), 1.0)

    def test_partial_availability_scales_power(self):
        opt = build_optimization(plant_overrides=HOUSE_AND_EV)
        res = run(opt, battery_availability=[1, 0.5], soc_final=[0.5, 0.4])
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        p1 = res["P_batt_1"].to_numpy()
        self.assertTrue((p1 <= 0.95 * 0.5 * 7000 + 1e-3).all())
        self.assertTrue((p1 >= -(1 / 0.95) * 0.5 * 11000 - 1e-3).all())

    def test_departure_target_defaults_to_last_connected_step(self):
        """soc_target[1] without a timestep is required at the last connected
        step; the house battery keeps its own (absent) target."""
        opt = build_optimization(plant_overrides=HOUSE_AND_EV)
        mask = [1] * 8 + [0] * (N - 8)
        res = run(
            opt,
            soc_init=[0.5, 0.3],
            soc_final=[0.5, 0.8],
            battery_availability=[None, mask],
            soc_target=[None, 0.8],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        soc1 = res["SOC_opt_1"].to_numpy()
        self.assertGreaterEqual(soc1[7], 0.8 - 1e-4)
        np.testing.assert_allclose(soc1[8:], soc1[7], atol=1e-6)
        np.testing.assert_allclose(opt.param_soc_target_floor[0].value, 0.0)

    def test_discharge_energy_budget_is_respected(self):
        """Expensive second half makes V2G attractive; the budget caps it."""
        free = build_optimization(plant_overrides=HOUSE_AND_EV)
        res_free = run(free, soc_init=[0.5, 0.8], soc_final=[0.5, 0.3])
        dis_free = np.clip(res_free["P_batt_1"].to_numpy(), 0, None).sum() * 0.5 / 0.95
        budget = 4000.0
        self.assertGreater(dis_free, budget)
        capped = build_optimization(plant_overrides=HOUSE_AND_EV)
        res = run(
            capped,
            soc_init=[0.5, 0.8],
            soc_final=[0.5, 0.75],
            battery_discharge_energy_max=[None, budget],
        )
        self.assertIn(capped.optim_status, VALID_OPTIMAL_STATUSES)
        dis = np.clip(res["P_batt_1"].to_numpy(), 0, None).sum() * 0.5 / 0.95
        self.assertLessEqual(dis, budget + 1e-3)

    def test_absent_ev_below_min_soc_stays_feasible(self):
        """An EV that is not connected at all, with soc_init below soc_min,
        must not make the LP infeasible (no recovery is possible)."""
        opt = build_optimization(plant_overrides=HOUSE_AND_EV)
        res = run(
            opt,
            soc_init=[0.5, 0.1],
            soc_final=[0.5, 0.1],
            battery_availability=[1, 0],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        np.testing.assert_allclose(res["P_batt_1"].to_numpy(), 0.0, atol=1e-6)
        np.testing.assert_allclose(res["SOC_opt_1"].to_numpy(), 0.1, atol=1e-6)

    def test_mask_change_on_the_same_problem_is_honoured(self):
        """The ceilings are Parameters: a new mask on a reused (cached)
        problem must apply without a rebuild."""
        opt = build_optimization(plant_overrides=HOUSE_AND_EV)
        run(opt, battery_availability=[1, 1], soc_init=[0.5, 0.8], soc_final=[0.5, 0.3])
        res = run(opt, battery_availability=[1, 0], soc_init=[0.5, 0.8], soc_final=[0.5, 0.8])
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        np.testing.assert_allclose(res["P_batt_1"].to_numpy(), 0.0, atol=1e-6)

    def test_wrong_length_raises(self):
        opt = build_optimization(plant_overrides=HOUSE_AND_EV)
        with self.assertRaises(ValueError):
            run(opt, battery_availability=[1, 1, 1])

    def test_single_battery_flat_series(self):
        opt = build_optimization()
        p_pv, p_load, df_input = scenario()
        res = opt.perform_naive_mpc_optim(
            df_input,
            p_pv,
            p_load,
            N,
            soc_init=0.5,
            soc_final=0.5,
            def_total_hours=[],
            def_total_timestep=[],
            def_start_timestep=[],
            def_end_timestep=[],
            battery_availability=[0] * 6 + [1] * 6,
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        np.testing.assert_allclose(res["P_batt"].to_numpy()[:6], 0.0, atol=1e-6)


class TestBackwardCompatibility(unittest.TestCase):
    """Callers that never pass the new runtime keys must see no change."""

    def test_n1_without_new_keys_matches_explicit_defaults(self):
        a = build_optimization()
        p_pv, p_load, df_input = scenario()
        common = {
            "soc_init": 0.5,
            "soc_final": 0.5,
            "def_total_hours": [],
            "def_total_timestep": [],
            "def_start_timestep": [],
            "def_end_timestep": [],
        }
        res_a = a.perform_naive_mpc_optim(df_input, p_pv, p_load, N, **common)
        b = build_optimization()
        res_b = b.perform_naive_mpc_optim(
            df_input,
            p_pv,
            p_load,
            N,
            battery_availability=None,
            battery_discharge_energy_max=None,
            **common,
        )
        self.assertAlmostEqual(a.prob.value, b.prob.value, places=6)
        np.testing.assert_allclose(res_a["P_batt"], res_b["P_batt"], atol=1e-6)

    def test_scalar_soc_target_still_applies_to_every_battery_at_horizon_end(self):
        """#553 semantics: a bare soc_target without timestep is required at the
        last timestep of every battery."""
        opt = build_optimization(plant_overrides=HOUSE_AND_EV)
        run(opt, soc_init=[0.3, 0.3], soc_final=[0.3, 0.3], soc_target=0.6)
        for k, cap in enumerate([10000, 60000]):
            floor = opt.param_soc_target_floor[k].value
            self.assertAlmostEqual(floor[-1], 0.6 * cap)
            self.assertEqual(int(np.count_nonzero(floor)), 1)

    def test_dayahead_ignores_the_new_keys(self):
        opt = build_optimization(plant_overrides=HOUSE_AND_EV)
        p_pv, p_load, df_input = scenario()
        opt.perform_dayahead_forecast_optim(
            df_input, p_pv, p_load, soc_init=[0.5, 0.5], soc_final=[0.5, 0.5]
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        for k in range(2):
            self.assertTrue(np.all(opt.param_batt_charge_ceiling[k].value > 0))


if __name__ == "__main__":
    unittest.main()
