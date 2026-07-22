#!/usr/bin/env python
"""Tests for the multi-battery optimization model (issue #610).

Scope: src/emhass/optimization.py battery formulation (variables, constraints,
objective, power balance, results dataframe) generalised from a single battery
to N batteries via ``number_of_batteries`` (plant_conf).

All scenarios are synthetic and self-contained (mirrors
tests/test_soc_recovery_prototype.py's build_optimization() helper) - no
dependency on the real forecast data files used by TestOptimization in
test_optimization.py, so this file solves quickly and deterministically.
"""

import logging
import math
import pathlib
import unittest

import numpy as np
import pandas as pd

from emhass.command_line import OptimizationCache
from emhass.optimization import Optimization

TEST_ROOT = pathlib.Path(__file__).resolve().parents[1]

VALID_OPTIMAL_STATUSES = ["Optimal", "Optimal (Relaxed)"]


def build_optimization(
    optim_overrides=None, plant_overrides=None, opt_time_delta=5
) -> Optimization:
    """Self-contained Optimization builder, mirroring
    test_soc_recovery_prototype.py's build_optimization(). Single battery by
    default; pass plant_overrides={"number_of_batteries": N, ...} for N>1.
    """
    logger = logging.getLogger("multi_battery_test")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())

    retrieve_hass_conf = {
        "optimization_time_step": pd.to_timedelta(30, "minutes"),
        "time_zone": "Europe/Tallinn",
        "sensor_power_photovoltaics": "pv",
        "sensor_power_load_no_var_loads": "load",
    }
    optim_conf = {
        "delta_forecast_daily": pd.Timedelta(hours=5),
        "num_threads": 0,
        "set_use_battery": True,
        "set_use_pv": True,
        "set_total_pv_sell": False,
        "set_nocharge_from_grid": False,
        "set_nodischarge_to_grid": False,
        "set_battery_dynamic": False,
        "set_battery_first_priority": False,
        "battery_dynamic_max": 0.9,
        "battery_dynamic_min": -0.9,
        "weight_battery_discharge": 0.0,
        "weight_battery_charge": 0.0,
        "battery_soc_deficit_threshold": 0.2,
        "battery_soc_deficit_cost": 0.0,
        "battery_soc_surplus_threshold": 0.9,
        "battery_soc_surplus_cost": 0.0,
        "number_of_deferrable_loads": 0,
        "nominal_power_of_deferrable_loads": [],
        "treat_deferrable_load_as_semi_cont": [],
        "set_deferrable_load_single_constant": [],
        "set_deferrable_startup_penalty": [],
        "operating_hours_of_each_deferrable_load": [],
        "start_timesteps_of_each_deferrable_load": [],
        "end_timesteps_of_each_deferrable_load": [],
        "lp_solver_timeout": 45,
        "lp_solver_mip_rel_gap": 0,
    }
    if optim_overrides:
        optim_conf.update(optim_overrides)

    plant_conf = {
        "inverter_is_hybrid": False,
        "compute_curtailment": False,
        "maximum_power_from_grid": 50000,
        "maximum_power_to_grid": 50000,
        "battery_discharge_power_max": 5000,
        "battery_charge_power_max": 5000,
        "battery_minimum_state_of_charge": 0.3,
        "battery_maximum_state_of_charge": 0.8,
        "battery_target_state_of_charge": 0.6,
        "battery_nominal_energy_capacity": 10000,
        "battery_discharge_efficiency": 1.0,
        "battery_charge_efficiency": 1.0,
        "battery_stress_cost": 0.0,
        "battery_stress_segments": 10,
    }
    if plant_overrides:
        plant_conf.update(plant_overrides)

    emhass_conf = {
        "root_path": TEST_ROOT / "src" / "emhass",
        "data_path": TEST_ROOT / "data",
    }
    return Optimization(
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        "unit_load_cost",
        "unit_prod_price",
        "profit",
        emhass_conf,
        logger,
        opt_time_delta=opt_time_delta,
    )


def make_pin_scenario():
    """The exact scenario used to capture the N=1 regression pin values
    below. Deterministic, 10 timesteps / 30 min."""
    index = pd.date_range("2026-02-01", periods=10, freq="30min", tz="Europe/Tallinn")
    p_pv = pd.Series([0, 0, 0, 500, 3000, 4500, 3200, 800, 0, 0], index=index)
    p_load = pd.Series([600, 550, 600, 700, 900, 1100, 1000, 800, 700, 650], index=index)
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = [0.30, 0.28, 0.28, 0.32, 0.10, 0.10, 0.12, 0.35, 0.40, 0.38]
    df_input["unit_prod_price"] = [0.05, 0.05, 0.05, 0.06, 0.04, 0.04, 0.04, 0.06, 0.07, 0.07]
    return index, p_pv, p_load, df_input


class TestMultiBatteryN1RegressionPin(unittest.TestCase):
    """N=1 pin: N=1 (the default) must be MATHEMATICALLY UNCHANGED vs current
    master - same objective value, same battery schedule on identical inputs.
    Values below were captured by running the identical scenario
    (make_pin_scenario + the optim/plant_conf in build_optimization) against
    the pre-change optimization.py, i.e. commit 79da54e5, before this diff's
    per-battery rewrite ever touched the file.
    """

    PIN_TOLERANCE = 1e-6

    PINNED_DAYAHEAD = {
        "optim_status": "Optimal",
        "objective_value": -0.19514378860672618,
        "P_batt": [
            600.0,
            550.0,
            600.0,
            200.0,
            -2100.0,
            -3400.0,
            -425.4175245939154,
            0.0,
            700.0,
            650.0,
        ],
        "SOC_opt": [
            0.36808510638297876,
            0.33882978723404256,
            0.3069148936170213,
            0.29627659574468085,
            0.39392659574468086,
            0.5520265957446808,
            0.571808510638298,
            0.571808510638298,
            0.5345744680851064,
            0.5000000000000001,
        ],
    }
    PINNED_NAIVE_MPC = PINNED_DAYAHEAD  # identical scenario, same numbers

    # Exact optim_conf/plant_conf overrides used when these values were
    # captured - must match byte-for-byte or the pin is meaningless.
    # Deliberately distinct from build_optimization()'s own
    # defaults (nondefault weights/efficiencies/limits) so the pin is not
    # accidentally satisfied by a coincidental default match.
    PIN_OPTIM_OVERRIDES = {
        "set_nodischarge_to_grid": True,
        "weight_battery_discharge": 0.05,
        "weight_battery_charge": 0.05,
    }
    PIN_PLANT_OVERRIDES = {
        "battery_discharge_power_max": 4000,
        "battery_charge_power_max": 3500,
        "battery_minimum_state_of_charge": 0.2,
        "battery_maximum_state_of_charge": 0.9,
        "battery_target_state_of_charge": 0.5,
        "battery_nominal_energy_capacity": 10000,
        "battery_discharge_efficiency": 0.94,
        "battery_charge_efficiency": 0.93,
    }

    def _build_pin_optimization(self):
        return build_optimization(
            optim_overrides=self.PIN_OPTIM_OVERRIDES, plant_overrides=self.PIN_PLANT_OVERRIDES
        )

    def test_n1_dayahead_objective_and_schedule_pinned(self):
        opt = self._build_pin_optimization()
        index, p_pv, p_load, df_input = make_pin_scenario()
        opt_res = opt.perform_dayahead_forecast_optim(
            df_input, p_pv, p_load, soc_init=0.4, soc_final=0.5
        )
        self.assertEqual(opt.optim_status, self.PINNED_DAYAHEAD["optim_status"])
        self.assertAlmostEqual(
            opt.prob.value, self.PINNED_DAYAHEAD["objective_value"], delta=self.PIN_TOLERANCE
        )
        np.testing.assert_allclose(
            opt_res["P_batt"].to_numpy(),
            self.PINNED_DAYAHEAD["P_batt"],
            atol=self.PIN_TOLERANCE,
        )
        np.testing.assert_allclose(
            opt_res["SOC_opt"].to_numpy(),
            self.PINNED_DAYAHEAD["SOC_opt"],
            atol=self.PIN_TOLERANCE,
        )
        # N=1 must not emit any per-battery suffixed column (no-op pin).
        self.assertNotIn("P_batt_0", opt_res.columns)
        self.assertNotIn("SOC_opt_0", opt_res.columns)

    def test_n1_naive_mpc_objective_and_schedule_pinned(self):
        opt = self._build_pin_optimization()
        index, p_pv, p_load, df_input = make_pin_scenario()
        opt_res = opt.perform_naive_mpc_optim(
            df_input,
            p_pv,
            p_load,
            10,
            soc_init=0.4,
            soc_final=0.5,
            def_total_hours=[],
            def_total_timestep=[],
            def_start_timestep=[],
            def_end_timestep=[],
        )
        self.assertEqual(opt.optim_status, self.PINNED_NAIVE_MPC["optim_status"])
        self.assertAlmostEqual(
            opt.prob.value, self.PINNED_NAIVE_MPC["objective_value"], delta=self.PIN_TOLERANCE
        )
        np.testing.assert_allclose(
            opt_res["P_batt"].to_numpy(),
            self.PINNED_NAIVE_MPC["P_batt"],
            atol=self.PIN_TOLERANCE,
        )
        np.testing.assert_allclose(
            opt_res["SOC_opt"].to_numpy(),
            self.PINNED_NAIVE_MPC["SOC_opt"],
            atol=self.PIN_TOLERANCE,
        )


class TestMultiBatteryModelShape(unittest.TestCase):
    """N=2 basic model shape: distinct capacities/efficiencies -> solves
    optimal; P_batt_0 + P_batt_1 == fleet P_batt column; per-battery SOC
    columns present and each within its own min/max band."""

    def _scenario(self, n=8):
        index = pd.date_range("2026-03-01", periods=n, freq="30min", tz="Europe/Tallinn")
        rng = np.random.default_rng(42)
        p_pv = pd.Series(rng.uniform(0, 3000, n), index=index)
        p_load = pd.Series(rng.uniform(400, 1500, n), index=index)
        df_input = pd.DataFrame(index=index)
        df_input["unit_load_cost"] = rng.uniform(0.15, 0.35, n)
        df_input["unit_prod_price"] = rng.uniform(0.03, 0.08, n)
        return index, p_pv, p_load, df_input

    def test_n2_distinct_batteries_solves_optimal_with_expected_columns(self):
        opt = build_optimization(
            plant_overrides={
                "number_of_batteries": 2,
                "battery_nominal_energy_capacity": [8000, 12000],
                "battery_discharge_power_max": [3000, 4000],
                "battery_charge_power_max": [2500, 3500],
                "battery_discharge_efficiency": [0.92, 0.96],
                "battery_charge_efficiency": [0.90, 0.95],
                "battery_minimum_state_of_charge": [0.15, 0.25],
                "battery_maximum_state_of_charge": [0.85, 0.95],
                "battery_target_state_of_charge": [0.5, 0.6],
            }
        )
        index, p_pv, p_load, df_input = self._scenario()
        opt_res = opt.perform_dayahead_forecast_optim(
            df_input, p_pv, p_load, soc_init=[0.5, 0.6], soc_final=[0.5, 0.6]
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)

        for col in ("P_batt", "P_batt_0", "P_batt_1", "SOC_opt_0", "SOC_opt_1"):
            self.assertIn(col, opt_res.columns)
        self.assertNotIn("SOC_opt", opt_res.columns)  # no aggregate SOC at N>1

        np.testing.assert_allclose(
            opt_res["P_batt"].to_numpy(),
            (opt_res["P_batt_0"] + opt_res["P_batt_1"]).to_numpy(),
            atol=1e-6,
        )
        self.assertTrue((opt_res["SOC_opt_0"] >= 0.15 - 1e-6).all())
        self.assertTrue((opt_res["SOC_opt_0"] <= 0.85 + 1e-6).all())
        self.assertTrue((opt_res["SOC_opt_1"] >= 0.25 - 1e-6).all())
        self.assertTrue((opt_res["SOC_opt_1"] <= 0.95 + 1e-6).all())


class TestMultiBatteryPowerLimits(unittest.TestCase):
    """N=2 respects per-battery power limits: set them asymmetric, assert
    neither is exceeded."""

    def test_n2_asymmetric_power_limits_never_exceeded(self):
        n = 6
        index = pd.date_range("2026-03-02", periods=n, freq="30min", tz="Europe/Tallinn")
        # 3h horizon. Heavy load with no PV makes discharging attractive; the
        # terminal SoC deltas below are sized so each battery's REQUIRED
        # total drain is a large fraction of what its OWN discharge_power_max
        # can deliver over the horizon (tight-but-feasible), so the cap
        # genuinely constrains the schedule instead of being a slack no-op.
        p_pv = pd.Series([0] * n, index=index)
        p_load = pd.Series([1000] * n, index=index)
        df_input = pd.DataFrame(index=index)
        df_input["unit_load_cost"] = [0.30] * n
        df_input["unit_prod_price"] = [0.05] * n

        discharge_max = [1000, 3000]
        charge_max = [900, 2800]
        cap = [6000, 6000]
        # battery 0: needed = 0.45*6000 = 2700 Wh <= 1000W*3h = 3000 Wh (90%).
        # battery 1: needed = 0.85*6000 = 5100 Wh <= 3000W*3h = 9000 Wh (57%).
        soc_final = [0.45, 0.05]
        opt = build_optimization(
            plant_overrides={
                "number_of_batteries": 2,
                "battery_nominal_energy_capacity": cap,
                "battery_discharge_power_max": discharge_max,
                "battery_charge_power_max": charge_max,
                "battery_discharge_efficiency": [1.0, 1.0],
                "battery_charge_efficiency": [1.0, 1.0],
                "battery_minimum_state_of_charge": [0.05, 0.05],
                "battery_maximum_state_of_charge": [0.95, 0.95],
                "battery_target_state_of_charge": [0.9, 0.9],
            }
        )
        opt_res = opt.perform_dayahead_forecast_optim(
            df_input, p_pv, p_load, soc_init=[0.9, 0.9], soc_final=soc_final
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)

        tol = 1.0  # W
        self.assertTrue((opt_res["P_batt_0"] <= discharge_max[0] + tol).all())
        self.assertTrue((opt_res["P_batt_0"] >= -charge_max[0] - tol).all())
        self.assertTrue((opt_res["P_batt_1"] <= discharge_max[1] + tol).all())
        self.assertTrue((opt_res["P_batt_1"] >= -charge_max[1] - tol).all())
        # The scenario should actually be power-limited (else the test proves
        # nothing): battery 1 (larger limit) should be doing more of the work.
        self.assertGreater(opt_res["P_batt_1"].sum(), opt_res["P_batt_0"].sum())


def _free_allocation_scenario():
    """A scenario where the solver has genuine cross-battery allocation freedom.

    Net-zero terminal SOC (0.5 -> 0.5) with one expensive price spike: covering
    the spike from battery and recharging in the cheap steps is profitable, one
    battery alone can do the whole job, and the terminal-SOC equality does NOT
    force any particular split. Which battery does the work is therefore decided
    by real cost differences if any exist, and by the index tilt if none do.
    (An earlier version of this scenario pinned the terminal SOC in a way that
    left zero allocation freedom, so it could not exercise the epsilon at all.)
    """
    n = 6
    index = pd.date_range("2026-03-03", periods=n, freq="30min", tz="Europe/Tallinn")
    p_pv = pd.Series([0] * n, index=index)
    p_load = pd.Series([1200] * n, index=index)
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = [0.05, 0.05, 0.05, 0.05, 0.90, 0.05]
    df_input["unit_prod_price"] = [0.01] * n
    plant_overrides = {
        "number_of_batteries": 2,
        "battery_nominal_energy_capacity": [5000, 5000],
        "battery_discharge_power_max": [2000, 2000],
        "battery_charge_power_max": [2000, 2000],
        "battery_discharge_efficiency": [0.95, 0.95],
        "battery_charge_efficiency": [0.95, 0.95],
        "battery_minimum_state_of_charge": [0.1, 0.1],
        "battery_maximum_state_of_charge": [0.9, 0.9],
        "battery_target_state_of_charge": [0.5, 0.5],
    }
    return index, p_pv, p_load, df_input, plant_overrides


class TestMultiBatteryTiebreakDeterminism(unittest.TestCase):
    """Tie-break determinism: with two IDENTICAL batteries, battery 0 does
    (weakly) more of the work and re-solving gives the identical plan.

    Uses the free-allocation scenario so the tilt genuinely decides the split:
    with identical batteries at equal real cost, only BATTERY_TIEBREAK_EPS
    breaks the degeneracy, and it must steer the work to battery 0.

    The tilt penalizes total throughput (charge + discharge) scaled by
    battery index, so ties in BOTH directions resolve to the lowest-index
    battery, never the highest - see docs/config.md for the precise
    statement. This test exercises a discharge tie (the spike here is
    covered by discharge); either direction is dominated by any real
    cost/efficiency difference (see TestMultiBatteryDominanceGuard below).
    """

    def test_battery_0_carries_the_work_and_resolve_is_identical(self):
        _, p_pv, p_load, df_input, plant_overrides = _free_allocation_scenario()

        def solve_once():
            opt = build_optimization(plant_overrides=plant_overrides)
            opt_res = opt.perform_dayahead_forecast_optim(
                df_input, p_pv, p_load, soc_init=[0.5, 0.5], soc_final=[0.5, 0.5]
            )
            self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
            return opt_res

        res1 = solve_once()
        res2 = solve_once()

        usage0 = res1["P_batt_0"].abs().sum()
        usage1 = res1["P_batt_1"].abs().sum()
        # The spike is worth covering at all (the scenario is not vacuous)...
        self.assertGreater(usage0 + usage1, 100.0)
        # ...battery 0 does STRICTLY more of it (the tilt decided a real,
        # otherwise-degenerate choice), and battery 1 sits essentially idle.
        self.assertGreater(usage0, usage1)
        self.assertLess(usage1, 1.0)
        # Re-solving gives the identical plan (determinism, not solver noise).
        for col in ("P_batt_0", "P_batt_1", "SOC_opt_0", "SOC_opt_1"):
            np.testing.assert_allclose(res1[col].to_numpy(), res2[col].to_numpy(), atol=1e-6)


class TestMultiBatteryDominanceGuard(unittest.TestCase):
    """Dominance guard: two batteries where battery 1 is SLIGHTLY more
    efficient (0.1% better round-trip): battery 1 must carry (strictly) more
    discharge than battery 0 despite the index tilt - proves the epsilon
    never overrides a real difference."""

    def test_more_efficient_battery1_wins_a_genuinely_free_choice(self):
        # Free-allocation scenario (net-zero terminal SOC, one price spike,
        # either battery could cover it alone): with battery 1 given a 0.1%
        # better round-trip efficiency, the REAL cost difference must beat the
        # index-0 tilt (BATTERY_TIEBREAK_EPS) and battery 1 must carry the
        # work. An earlier, pinned-terminal version of this test was forced
        # by the terminal-SOC equality and would have passed at any epsilon;
        # this version fails if the epsilon is ever raised enough to
        # override a real efficiency difference.
        _, p_pv, p_load, df_input, plant_overrides = _free_allocation_scenario()

        eff = 0.95
        # 0.1% better round trip for battery 1, split across both directions.
        eff_better = eff * math.sqrt(1.001)
        plant_overrides["battery_discharge_efficiency"] = [eff, eff_better]
        plant_overrides["battery_charge_efficiency"] = [eff, eff_better]

        opt = build_optimization(plant_overrides=plant_overrides)
        opt_res = opt.perform_dayahead_forecast_optim(
            df_input, p_pv, p_load, soc_init=[0.5, 0.5], soc_final=[0.5, 0.5]
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)

        usage0 = opt_res["P_batt_0"].abs().sum()
        usage1 = opt_res["P_batt_1"].abs().sum()
        self.assertGreater(usage0 + usage1, 100.0)  # spike actually covered
        self.assertGreater(
            usage1,
            usage0,
            msg=(
                f"battery 1 (0.1% more efficient) should carry the work "
                f"despite the index-0 tie-break tilt: "
                f"usage0={usage0}, usage1={usage1}"
            ),
        )
        self.assertLess(usage0, 1.0)


class TestMultiBatteryModelSize(unittest.TestCase):
    """Variable/constraint counts scale as expected with N: no new binaries
    beyond the replicated per-battery direction binary and the two
    SOC-recovery flags. Going N=1 -> N=2 on an identical scenario must add
    EXACTLY 3*n boolean scalars (E, soc_low_recovered, soc_high_recovered for
    the extra battery) and nothing else boolean - the shared grid binary D
    and the (disabled here) battery-first gate must not multiply."""

    @staticmethod
    def _boolean_scalar_count(opt) -> int:
        return sum(
            int(np.prod(v.shape)) if v.shape else 1
            for v in opt.prob.variables()
            if v.attributes.get("boolean")
        )

    def test_n2_adds_exactly_the_replicated_battery_binaries(self):
        _, p_pv, p_load, df_input, plant_overrides = _free_allocation_scenario()
        n = len(df_input)

        counts = {}
        for n_batt in (1, 2):
            overrides = dict(plant_overrides)
            overrides["number_of_batteries"] = n_batt
            if n_batt == 1:
                overrides = {k: (v[0] if isinstance(v, list) else v) for k, v in overrides.items()}
                overrides["number_of_batteries"] = 1
            opt = build_optimization(plant_overrides=overrides)
            opt.perform_dayahead_forecast_optim(
                df_input,
                p_pv,
                p_load,
                soc_init=[0.5] * n_batt if n_batt > 1 else 0.5,
                soc_final=[0.5] * n_batt if n_batt > 1 else 0.5,
            )
            self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
            counts[n_batt] = self._boolean_scalar_count(opt)

        self.assertEqual(
            counts[2] - counts[1],
            3 * n,
            msg=(
                f"N=1->N=2 must add exactly 3*n={3 * n} boolean scalars "
                f"(per-battery E + 2 SOC-recovery flags), got "
                f"{counts[2] - counts[1]} (N1={counts[1]}, N2={counts[2]})"
            ),
        )


class TestMultiBatteryFirstAggregate(unittest.TestCase):
    """Battery-first aggregate N=2: no grid import while AGGREGATE stored
    energy is above the aggregate minimum (re-derived from
    test_battery_first_priority_drains_before_import at
    test_optimization.py:1208)."""

    def test_no_import_while_aggregate_soc_above_aggregate_min(self):
        n = 8
        index = pd.date_range("2026-03-05", periods=n, freq="30min", tz="Europe/Tallinn")
        p_pv = pd.Series([0] * n, index=index)
        load_w = 1500.0
        p_load = pd.Series([load_w] * n, index=index)
        df_input = pd.DataFrame(index=index)
        # Increasing price makes "import early while charged" the baseline
        # optimum absent the feature, same trick as the single-battery test.
        df_input["unit_load_cost"] = 0.20 + 0.001 * np.arange(n)
        df_input["unit_prod_price"] = 0.05

        # Mirrors test_battery_first_priority_drains_before_import's sizing
        # trick (test_optimization.py:1208): cap sized so the FLEET's usable
        # energy (soc_init -> soc_min, summed over both batteries) covers
        # exactly half the horizon's load, so draining fully to the aggregate
        # minimum is achievable via load consumption alone (no need to export
        # the difference, which set_nodischarge_to_grid would forbid).
        # total load energy = 1500W * 4h = 6000 Wh; half = 3000 Wh, split
        # evenly: (0.9-0.1)*cap_k*2 == 3000 -> cap_k = 1875.
        cap = [1875, 1875]
        soc_min = [0.1, 0.1]
        plant_overrides = {
            "number_of_batteries": 2,
            "battery_nominal_energy_capacity": cap,
            "battery_discharge_power_max": [10000, 10000],
            "battery_charge_power_max": [10000, 10000],
            "battery_discharge_efficiency": [1.0, 1.0],
            "battery_charge_efficiency": [1.0, 1.0],
            "battery_minimum_state_of_charge": soc_min,
            "battery_maximum_state_of_charge": [1.0, 1.0],
            "battery_target_state_of_charge": [0.5, 0.5],
        }
        optim_overrides = {
            "set_battery_first_priority": True,
            "set_nodischarge_to_grid": True,
        }
        opt = build_optimization(optim_overrides=optim_overrides, plant_overrides=plant_overrides)
        opt_res = opt.perform_dayahead_forecast_optim(
            df_input,
            p_pv,
            p_load,
            soc_init=[0.9, 0.9],
            soc_final=[soc_min[0], soc_min[1]],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)

        aggregate_min_energy = sum(soc_min[k] * cap[k] for k in range(2))
        aggregate_stored = opt_res["SOC_opt_0"] * cap[0] + opt_res["SOC_opt_1"] * cap[1]
        above_min = aggregate_stored > aggregate_min_energy + 0.02 * sum(cap)
        self.assertGreater(above_min.sum(), 0, msg="no timesteps to test")
        self.assertLess(
            opt_res.loc[above_min, "P_grid_pos"].abs().max(),
            1.0,
            msg="grid import happened while AGGREGATE stored energy was above aggregate min",
        )


class TestMultiBatteryDeficitSurplusCost(unittest.TestCase):
    """Per-battery deficit/surplus: battery with deficit_cost>0 holds SOC
    higher than its sibling with 0, all else equal."""

    def test_battery_with_deficit_cost_holds_higher_soc(self):
        n = 6
        index = pd.date_range("2026-03-06", periods=n, freq="30min", tz="Europe/Tallinn")
        p_pv = pd.Series([0] * n, index=index)
        p_load = pd.Series([1600] * n, index=index)
        df_input = pd.DataFrame(index=index)
        df_input["unit_load_cost"] = [0.30] * n
        df_input["unit_prod_price"] = [0.05] * n

        opt = build_optimization(
            optim_overrides={
                # Per-battery optim_conf array: battery 0 penalized for
                # dipping below 0.5 SoC, battery 1 has no such penalty.
                "battery_soc_deficit_threshold": [0.5, 0.5],
                "battery_soc_deficit_cost": [5.0, 0.0],
            },
            plant_overrides={
                "number_of_batteries": 2,
                "battery_nominal_energy_capacity": [5000, 5000],
                "battery_discharge_power_max": [1000, 1000],
                "battery_charge_power_max": [1000, 1000],
                "battery_discharge_efficiency": [1.0, 1.0],
                "battery_charge_efficiency": [1.0, 1.0],
                "battery_minimum_state_of_charge": [0.05, 0.05],
                "battery_maximum_state_of_charge": [0.95, 0.95],
                "battery_target_state_of_charge": [0.6, 0.6],
            },
        )
        opt_res = opt.perform_dayahead_forecast_optim(
            df_input, p_pv, p_load, soc_init=[0.6, 0.6], soc_final=[0.1, 0.1]
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)

        self.assertGreater(
            opt_res["SOC_opt_0"].mean(),
            opt_res["SOC_opt_1"].mean(),
            msg="battery 0 (deficit_cost=5.0) should hold higher average SoC "
            "than battery 1 (deficit_cost=0.0)",
        )


class TestMultiBatteryNodischargeToGridCurtailment(unittest.TestCase):
    """set_nodischarge_to_grid + curtailment with N=2 (the #936 three-site
    trap): no battery energy leaks to grid through any of the three sites.
    Adapted from test_nodischarge_to_grid_curtailment_no_battery_export_issue936
    (test_optimization.py:7243) to N=2 - the export-bound constraint is
    aggregate (shared grid connection), so the leak check is on the fleet
    total, exactly as the original single-battery test checked it."""

    def test_n2_no_battery_export_via_curtailment_escape_valve(self):
        n = 6
        index = pd.date_range("2026-03-07", periods=n, freq="30min", tz="Europe/Tallinn")
        p_pv_w = 3000.0
        p_load_w = 500.0
        p_pv = pd.Series([p_pv_w] * n, index=index)
        p_load = pd.Series([p_load_w] * n, index=index)
        df_input = pd.DataFrame(index=index)
        df_input["unit_load_cost"] = 0.20
        df_input["unit_prod_price"] = 1.00  # high feed-in incentivises export

        cap = [5000.0, 5000.0]  # loss-free 10 kWh fleet total
        discharge_power = [4000.0, 4000.0]

        opt = build_optimization(
            optim_overrides={
                "set_nodischarge_to_grid": True,
                "set_nocharge_from_grid": True,
                "set_battery_dynamic": False,
            },
            plant_overrides={
                "number_of_batteries": 2,
                "inverter_is_hybrid": False,
                "compute_curtailment": True,
                "battery_nominal_energy_capacity": cap,
                "battery_discharge_power_max": discharge_power,
                "battery_charge_power_max": discharge_power,
                "battery_discharge_efficiency": [1.0, 1.0],
                "battery_charge_efficiency": [1.0, 1.0],
                "battery_minimum_state_of_charge": [0.0, 0.0],
                "battery_maximum_state_of_charge": [1.0, 1.0],
            },
        )
        opt_res = opt.perform_dayahead_forecast_optim(
            df_input, p_pv, p_load, soc_init=[0.9, 0.9], soc_final=[0.1, 0.1]
        )

        if opt.optim_status not in VALID_OPTIMAL_STATUSES:
            # Same as the single-battery precedent: infeasible is an
            # acceptable, correct outcome here (the forced drain has no legal
            # non-exporting path); nothing more to assert.
            return

        epsilon = 1e-3
        p_grid_neg = opt_res["P_grid_neg"].to_numpy()
        pv_available = opt_res["P_PV"].to_numpy() - opt_res["P_PV_curtailment"].to_numpy()
        battery_export = -(p_grid_neg) - pv_available
        self.assertTrue(
            (battery_export <= epsilon).all(),
            msg=f"battery energy exported to grid: {battery_export.tolist()}",
        )


class TestMultiBatteryHybridInverter(unittest.TestCase):
    """Hybrid inverter N=2: DC-bus balance holds (P_batt sum folds into
    p_hybrid_inverter)."""

    def test_n2_hybrid_dc_bus_balance_holds(self):
        n = 5
        index = pd.date_range("2026-03-08", periods=n, freq="30min", tz="Europe/Tallinn")
        p_pv = pd.Series([0, 200, 0, 0, 0], index=index)
        p_load = pd.Series([1500] * n, index=index)
        df_input = pd.DataFrame(index=index)
        df_input["unit_load_cost"] = 0.25
        df_input["unit_prod_price"] = 0.05

        opt = build_optimization(
            plant_overrides={
                "number_of_batteries": 2,
                "inverter_is_hybrid": True,
                "inverter_ac_output_max": 8000,
                "inverter_ac_input_max": 8000,
                "inverter_efficiency_dc_ac": 1.0,
                "inverter_efficiency_ac_dc": 1.0,
                "battery_nominal_energy_capacity": [5000, 5000],
                "battery_discharge_power_max": [2000, 2000],
                "battery_charge_power_max": [2000, 2000],
                "battery_discharge_efficiency": [1.0, 1.0],
                "battery_charge_efficiency": [1.0, 1.0],
                "battery_minimum_state_of_charge": [0.05, 0.05],
                "battery_maximum_state_of_charge": [0.95, 0.95],
                "battery_target_state_of_charge": [0.5, 0.5],
            }
        )
        opt_res = opt.perform_dayahead_forecast_optim(
            df_input, p_pv, p_load, soc_init=[0.9, 0.9], soc_final=[0.3, 0.3]
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)

        # Fleet total is the structural sum of the two per-battery columns.
        np.testing.assert_allclose(
            opt_res["P_batt"].to_numpy(),
            (opt_res["P_batt_0"] + opt_res["P_batt_1"]).to_numpy(),
            atol=1e-6,
        )
        # DC-bus balance (lossless inverter here): P_hybrid_inverter must
        # equal P_PV + fleet battery power (no curtailment configured).
        np.testing.assert_allclose(
            opt_res["P_hybrid_inverter"].to_numpy(),
            (opt_res["P_PV"] + opt_res["P_batt"]).to_numpy(),
            atol=1e-6,
        )


class TestMultiBatteryDynamic(unittest.TestCase):
    """set_battery_dynamic with N=2: ramp limits applied per battery against
    ITS OWN power max."""

    def test_n2_ramp_limits_are_per_battery(self):
        n = 6
        index = pd.date_range("2026-03-09", periods=n, freq="30min", tz="Europe/Tallinn")
        # Alternate heavy/no load to try to force a big swing in battery
        # power, so the ramp constraint actually binds.
        p_pv = pd.Series([0] * n, index=index)
        p_load = pd.Series([0, 3000, 0, 3000, 0, 3000], index=index)
        df_input = pd.DataFrame(index=index)
        df_input["unit_load_cost"] = 0.25
        df_input["unit_prod_price"] = 0.05

        discharge_max = [1000, 3000]
        dynamic_max = 0.5  # fraction of own power max, per timestep
        opt = build_optimization(
            optim_overrides={
                "set_battery_dynamic": True,
                "battery_dynamic_max": dynamic_max,
                "battery_dynamic_min": -dynamic_max,
            },
            plant_overrides={
                "number_of_batteries": 2,
                "battery_nominal_energy_capacity": [5000, 5000],
                "battery_discharge_power_max": discharge_max,
                "battery_charge_power_max": discharge_max,
                "battery_discharge_efficiency": [1.0, 1.0],
                "battery_charge_efficiency": [1.0, 1.0],
                "battery_minimum_state_of_charge": [0.05, 0.05],
                "battery_maximum_state_of_charge": [0.95, 0.95],
                "battery_target_state_of_charge": [0.6, 0.6],
            },
        )
        # delta=0.2 * cap=5000 = 1000 Wh needed for EACH battery (eff=1.0),
        # well under battery 0's own 1000W*3h=3000 Wh ceiling (and its ramp-
        # limited deliverable, since ramping from 0 to 1000W in 250 W/step
        # increments still reaches full power within the 6-step horizon).
        opt.perform_dayahead_forecast_optim(
            df_input, p_pv, p_load, soc_init=[0.9, 0.9], soc_final=[0.7, 0.7]
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)

        # Check the ramp limit against the underlying p_sto_pos/p_sto_neg
        # variables directly (not the net P_batt column): the ramp
        # constraint bounds each of those separately, so a timestep where a
        # battery flips from charging to discharging (or vice versa) can
        # legitimately show a bigger jump in the NET P_batt = p_sto_pos +
        # p_sto_neg than either individual ramp limit, without violating
        # anything - that is a modelling fact carried over unchanged from the
        # single-battery formulation, not a #610 regression.
        time_step_h = 0.5
        tol = 1.0  # W
        limit0 = dynamic_max * discharge_max[0] * time_step_h
        limit1 = dynamic_max * discharge_max[1] * time_step_h
        p_sto_pos_0 = np.array(opt.vars["p_sto_pos"][0].value)
        p_sto_pos_1 = np.array(opt.vars["p_sto_pos"][1].value)
        p_sto_neg_0 = np.array(opt.vars["p_sto_neg"][0].value)
        p_sto_neg_1 = np.array(opt.vars["p_sto_neg"][1].value)
        diff_pos_0 = np.abs(np.diff(p_sto_pos_0))
        diff_pos_1 = np.abs(np.diff(p_sto_pos_1))
        diff_neg_0 = np.abs(np.diff(p_sto_neg_0))
        diff_neg_1 = np.abs(np.diff(p_sto_neg_1))
        self.assertTrue((diff_pos_0 <= limit0 + tol).all(), msg=diff_pos_0.tolist())
        self.assertTrue((diff_pos_1 <= limit1 + tol).all(), msg=diff_pos_1.tolist())
        self.assertTrue((diff_neg_0 <= limit0 + tol).all(), msg=diff_neg_0.tolist())
        self.assertTrue((diff_neg_1 <= limit1 + tol).all(), msg=diff_neg_1.tolist())
        # Sanity: the scenario is genuinely ramp-limited for battery 0 (else
        # the test proves nothing) - some step must be at (near) its cap.
        self.assertGreater(diff_pos_0.max(), limit0 - tol)


class TestMultiBatteryStructuralCache(unittest.TestCase):
    """number_of_batteries must land in the OptimizationCache structural
    catch-all (command_line.py ~279-284's ``plant_runtime_keys`` explicit
    allowlist does NOT include it, so
    ``config_hash(plant_conf, plant_runtime_keys)`` at command_line.py:402
    changes whenever number_of_batteries changes -> cache MISS). This test
    exercises the existing (unmodified by this diff) OptimizationCache
    directly, mirroring test_cache_miss_battery_config_changed in
    test_command_line_utils.py:2314."""

    def setUp(self):
        OptimizationCache.clear()
        self.logger = logging.getLogger("multi_battery_cache_test")
        self.optim_conf = {"number_of_deferrable_loads": 0, "set_use_battery": True}
        self.plant_conf = {
            "number_of_batteries": 1,
            "battery_nominal_energy_capacity": 10000,
            "inverter_is_hybrid": False,
            "compute_curtailment": False,
        }
        self.retrieve_hass_conf = {"optimization_time_step": pd.Timedelta(minutes=30)}
        self.costfun = "profit"

    def tearDown(self):
        OptimizationCache.clear()

    def test_number_of_batteries_change_is_a_cache_miss(self):
        from unittest.mock import MagicMock

        mock_opt = MagicMock()
        OptimizationCache.put(
            mock_opt,
            self.optim_conf,
            self.plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )
        modified_plant_conf = self.plant_conf.copy()
        modified_plant_conf["number_of_batteries"] = 2

        result = OptimizationCache.get(
            self.optim_conf,
            modified_plant_conf,
            self.costfun,
            self.retrieve_hass_conf,
            self.logger,
        )
        self.assertIsNone(result, "number_of_batteries change should invalidate the cache")


class TestMultiBatteryUpdatePowerLimits(unittest.TestCase):
    """Per-k Parameter update path for update_battery_power_limits (#610):
    power limits stay a per-battery cp.Parameter list, updated via a loop
    over k, so a runtime power-limit change updates in place without a
    problem rebuild."""

    def test_update_battery_power_limits_updates_each_battery_independently(self):
        opt = build_optimization(
            plant_overrides={
                "number_of_batteries": 2,
                "battery_discharge_power_max": [1000, 2000],
                "battery_charge_power_max": [900, 1900],
            }
        )
        self.assertEqual(opt.param_battery_discharge_power_max[0].value, 1000)
        self.assertEqual(opt.param_battery_discharge_power_max[1].value, 2000)
        self.assertEqual(opt.param_battery_charge_power_max[0].value, 900)
        self.assertEqual(opt.param_battery_charge_power_max[1].value, 1900)

        opt.update_battery_power_limits(
            {
                "battery_discharge_power_max": [1234, 5678],
                "battery_charge_power_max": [1111, 2222],
            }
        )
        self.assertEqual(opt.param_battery_discharge_power_max[0].value, 1234)
        self.assertEqual(opt.param_battery_discharge_power_max[1].value, 5678)
        self.assertEqual(opt.param_battery_charge_power_max[0].value, 1111)
        self.assertEqual(opt.param_battery_charge_power_max[1].value, 2222)


if __name__ == "__main__":
    unittest.main()
