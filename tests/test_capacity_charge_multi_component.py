#!/usr/bin/env python
"""Behaviour tests for generic multi-component capacity/demand charges (issue #540 Part B).

``capacity_cost_per_kw`` given as a list of K rates selects K independent
capacity/demand-charge components priced in ONE optimisation (one solver call,
one physical dispatch). Each component has its own rate, tariff measurement
interval, eligibility window, MPC consideration, realised open-interval history
and already-incurred incumbent peak - none shared with any other component. A
bare scalar runs the released K=1 machinery, whose semantics are preserved and
regression-pinned here.

All scenarios are synthetic and tariff-provider agnostic: EMHASS never sees a
tariff name, calendar or clock time - the caller owns those and passes plain
per-timestep numeric masks.
"""

import io
import logging
import pathlib
import unittest

import numpy as np
import pandas as pd

from emhass.optimization import Optimization

TEST_ROOT = pathlib.Path(__file__).resolve().parents[1]
VALID_OPTIMAL = ["Optimal", "Optimal (Relaxed)"]


def build_optimization(optim_overrides=None, plant_overrides=None) -> Optimization:
    logger = logging.getLogger("capacity_multi_component_test")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())

    retrieve_hass_conf = {
        "optimization_time_step": pd.to_timedelta(5, "minutes"),
        "time_zone": "Europe/Tallinn",
        "sensor_power_photovoltaics": "pv",
        "sensor_power_load_no_var_loads": "load",
    }
    optim_conf = {
        "delta_forecast_daily": pd.Timedelta(hours=2),
        "num_threads": 0,
        "set_use_battery": False,
        "set_use_pv": True,
        "set_total_pv_sell": False,
        "set_nocharge_from_grid": False,
        "set_nodischarge_to_grid": True,
        "set_battery_dynamic": False,
        "set_battery_first_priority": False,
        "battery_dynamic_max": 0.9,
        "battery_dynamic_min": -0.9,
        "weight_battery_discharge": 0.1,
        "weight_battery_charge": 0.1,
        "battery_soc_deficit_threshold": 0.0,
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
        "capacity_cost_per_kw": 0.0,
        "capacity_charge_interval_timesteps": 1,
    }
    if optim_overrides:
        optim_conf.update(optim_overrides)

    plant_conf = {
        "inverter_is_hybrid": False,
        "compute_curtailment": False,
        "maximum_power_from_grid": 50000,
        "maximum_power_to_grid": 50000,
        "battery_discharge_power_max": 20000,
        "battery_charge_power_max": 20000,
        "battery_minimum_state_of_charge": 0.0,
        "battery_maximum_state_of_charge": 1.0,
        "battery_target_state_of_charge": 0.5,
        "battery_nominal_energy_capacity": 10000,
        "battery_discharge_efficiency": 1.0,
        "battery_charge_efficiency": 1.0,
        "battery_stress_cost": 0.0,
        "battery_stress_segments": 10,
    }
    if plant_overrides:
        plant_conf.update(plant_overrides)

    return Optimization(
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        "unit_load_cost",
        "unit_prod_price",
        "profit",
        {"root_path": TEST_ROOT / "src" / "emhass", "data_path": TEST_ROOT / "data"},
        logger,
        opt_time_delta=5,
    )


def make_scenario(load, unit_load_cost=0.20, unit_prod_price=0.20):
    n = len(load)
    index = pd.date_range("2026-02-01", periods=n, freq="5min", tz="Europe/Tallinn")
    p_pv = pd.Series(np.zeros(n), index=index)
    p_load = pd.Series(load, index=index, dtype=float)
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = unit_load_cost
    df_input["unit_prod_price"] = unit_prod_price
    return p_pv, p_load, df_input


def _capture_debug(opt):
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    opt.logger.addHandler(handler)
    opt.logger.setLevel(logging.DEBUG)
    return buf


class TestMultiComponentCapacityCharge(unittest.TestCase):
    # ---- K=1 compatibility -------------------------------------------------

    def test_scalar_keeps_legacy_k1_structure(self):
        opt = build_optimization(optim_overrides={"capacity_cost_per_kw": 2.0})
        self.assertFalse(opt._capacity_multi)
        self.assertIn("peak_import", opt.vars)
        self.assertNotIn("peak_import_k", opt.vars)
        self.assertFalse(hasattr(opt, "param_capacity_window_k"))

    def test_released_k1_debug_diagnostics_are_preserved(self):
        """The shared-helper refactor must keep the three released v0.18.2 K=1
        DEBUG lines; K>N emits component-indexed equivalents."""
        load = [1000.0, 1000.0, 5000.0, 1000.0, 1000.0, 1000.0]
        pv, load_s, df = make_scenario(load)

        opt = build_optimization(optim_overrides={"capacity_cost_per_kw": 2.0})
        buf = _capture_debug(opt)
        opt.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            6,
            current_period_peak=1200.0,
            capacity_charge_window=[1, 1, 1, 0, 0, 0],
            capacity_charge_consideration=[1, 1, 0, 1, 1, 1],
        )
        out = buf.getvalue()
        self.assertIn("Capacity charge: demand-window mask active on 3/6 timesteps.", out)
        self.assertIn("Capacity charge: MPC consideration active on", out)
        self.assertIn(
            "Capacity charge: flooring peak_import at already-incurred "
            "current_period_peak = 1200.0 W.",
            out,
        )

        opt2 = build_optimization(optim_overrides={"capacity_cost_per_kw": [2.0, 2.0]})
        buf2 = _capture_debug(opt2)
        opt2.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            6,
            current_period_peak=[1200.0, 900.0],
            capacity_charge_window=[[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]],
        )
        out2 = buf2.getvalue()
        self.assertIn(
            "capacity_cost_per_kw[0]: Capacity charge: demand-window mask active on", out2
        )
        self.assertIn(
            "capacity_cost_per_kw[1]: Capacity charge: flooring peak_import at "
            "already-incurred current_period_peak = 900.0 W.",
            out2,
        )

    def test_singleton_list_canonicalises_to_legacy_k1(self):
        """A config-UI-serialised singleton list must canonicalise to the legacy
        scalar K=1 form BEFORE any structural decision: the same K=1 model
        (no K>N objects), the same cache key, the same dispatch."""
        load = [1000.0, 2000.0, 1000.0, 5000.0, 1000.0, 1000.0]
        pv, load_s, df = make_scenario(load)

        opt_scalar = build_optimization(optim_overrides={"capacity_cost_per_kw": 3.0})
        opt_list = build_optimization(
            optim_overrides={
                "capacity_cost_per_kw": [3.0],
                "capacity_charge_interval_timesteps": [1],
            }
        )
        # Structural identity: [3.0] built the legacy K=1 machinery, not K>N.
        for opt in (opt_scalar, opt_list):
            self.assertFalse(opt._capacity_multi)
            self.assertEqual(opt.optim_conf["capacity_cost_per_kw"], 3.0)
            self.assertEqual(opt.optim_conf["capacity_charge_interval_timesteps"], 1)
            self.assertFalse(hasattr(opt, "param_capacity_window_k"))
            self.assertFalse(hasattr(opt, "param_current_period_peak_k"))

        res_scalar = opt_scalar.perform_naive_mpc_optim(
            df, pv, load_s, len(load), current_period_peak=800.0
        )
        res_list = opt_list.perform_naive_mpc_optim(
            df, pv, load_s, len(load), current_period_peak=800.0
        )
        self.assertIn(opt_scalar.optim_status, VALID_OPTIMAL)
        self.assertIn(opt_list.optim_status, VALID_OPTIMAL)
        np.testing.assert_allclose(
            res_list["P_grid_pos"].to_numpy(), res_scalar["P_grid_pos"].to_numpy(), atol=1e-3
        )

    def test_empty_rate_list_canonicalises_to_disabled(self):
        """capacity_cost_per_kw = [] -> 0.0 -> the standard disabled path, not a
        synthetic K=1 multi model."""
        opt = build_optimization(optim_overrides={"capacity_cost_per_kw": []})
        self.assertFalse(opt._capacity_multi)
        self.assertEqual(opt.optim_conf["capacity_cost_per_kw"], 0.0)
        self.assertNotIn("peak_import", opt.vars)
        self.assertNotIn("peak_import_k", opt.vars)

    # ---- independence of the K components --------------------------------

    def test_independent_windows_rates_and_incumbents(self):
        load = [1000.0, 2000.0, 1000.0, 1000.0, 3000.0, 1000.0]
        pv, load_s, df = make_scenario(load)
        opt = build_optimization(optim_overrides={"capacity_cost_per_kw": [3.0, 7.0]})
        opt.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            len(load),
            current_period_peak=[500.0, 900.0],
            capacity_charge_window=[[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        # Component 0 only "sees" the first half: its priced peak is the 2 kW
        # event, not the later 3 kW one. Component 1 is the mirror image, but
        # its incumbent floor (900 W) is below its 3 kW event so does not bind.
        self.assertAlmostEqual(opt.vars["peak_import_k"][0].value, 2000.0, places=2)
        self.assertAlmostEqual(opt.vars["peak_import_k"][1].value, 3000.0, places=2)
        np.testing.assert_array_equal(opt.param_capacity_window_k[0].value, [1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(opt.param_capacity_window_k[1].value, [0, 0, 0, 1, 1, 1])
        self.assertAlmostEqual(opt.param_current_period_peak_k[0].value, 500.0)
        self.assertAlmostEqual(opt.param_current_period_peak_k[1].value, 900.0)

    def test_independent_incumbent_floors_bind_per_component(self):
        load = [0.0] * 6
        pv, load_s, df = make_scenario(load)
        opt = build_optimization(optim_overrides={"capacity_cost_per_kw": [1.0, 1.0]})
        opt.perform_naive_mpc_optim(df, pv, load_s, 6, current_period_peak=[1500.0, 4200.0])
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        self.assertAlmostEqual(opt.vars["peak_import_k"][0].value, 1500.0, places=2)
        self.assertAlmostEqual(opt.vars["peak_import_k"][1].value, 4200.0, places=2)

    def test_independent_consideration(self):
        load = [0.0, 2000.0, 0.0, 0.0, 5000.0, 0.0]
        pv, load_s, df = make_scenario(load)
        opt = build_optimization(optim_overrides={"capacity_cost_per_kw": [2.0, 2.0]})
        # Component 0 excludes the 5 kW timestep from consideration; component 1
        # considers everything. Same physical dispatch, different priced peaks.
        opt.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            len(load),
            capacity_charge_consideration=[[1, 1, 1, 1, 0, 1], [1] * 6],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        self.assertAlmostEqual(opt.vars["peak_import_k"][0].value, 2000.0, places=2)
        self.assertAlmostEqual(opt.vars["peak_import_k"][1].value, 5000.0, places=2)

    def test_consideration_never_erases_a_components_incumbent(self):
        load = [0.0] * 6
        pv, load_s, df = make_scenario(load)
        opt = build_optimization(optim_overrides={"capacity_cost_per_kw": [1.0, 1.0]})
        opt.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            6,
            current_period_peak=[0.0, 3200.0],
            capacity_charge_consideration=[[1] * 6, [0] * 6],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        # Component 1 considers no prospective timestep, but its realised
        # incumbent (3200 W) is untouched by consideration.
        self.assertAlmostEqual(opt.vars["peak_import_k"][1].value, 3200.0, places=2)

    def test_effective_eligibility_is_window_times_consideration_per_component(self):
        load = [4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0]
        pv, load_s, df = make_scenario(load)
        opt = build_optimization(optim_overrides={"capacity_cost_per_kw": [1.0, 1.0]})
        opt.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            6,
            capacity_charge_window=[[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]],
            capacity_charge_consideration=[[0, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        np.testing.assert_array_equal(opt.param_capacity_window_k[0].value, [0, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(opt.param_capacity_window_k[1].value, [1, 1, 0, 0, 0, 0])

    def test_k3_is_generic_not_hardcoded_k2(self):
        load = [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 1000.0]
        pv, load_s, df = make_scenario(load)
        opt = build_optimization(optim_overrides={"capacity_cost_per_kw": [1.0, 2.0, 3.0]})
        opt.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            len(load),
            capacity_charge_window=[
                [1, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1],
            ],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        self.assertEqual(len(opt.vars["peak_import_k"]), 3)
        self.assertAlmostEqual(opt.vars["peak_import_k"][0].value, 2000.0, places=2)
        self.assertAlmostEqual(opt.vars["peak_import_k"][1].value, 4000.0, places=2)
        self.assertAlmostEqual(opt.vars["peak_import_k"][2].value, 5000.0, places=2)

    def test_zero_rate_component_is_economically_inactive(self):
        opt = build_optimization(optim_overrides={"capacity_cost_per_kw": [2.0, 0.0]})
        # No decision variable, no epigraph, no interval aggregation for the
        # zero-rate component; its inert indexed Parameter containers may exist.
        self.assertIsNotNone(opt.vars["peak_import_k"][0])
        self.assertIsNone(opt.vars["peak_import_k"][1])
        self.assertIsNone(opt.param_capacity_interval_matrix_k[1])
        self.assertFalse(opt._capacity_interval_aggregation_active_list[1])

    def test_all_zero_multi_component_is_valid_but_inactive(self):
        """[0.0, 0.0] is structurally K=2 but economically inert: the dispatch
        must match the equivalent no-capacity-charge plan."""
        load = [1000.0, 6000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        pv, load_s, df = make_scenario(load)

        opt_zero = build_optimization(
            optim_overrides={"capacity_cost_per_kw": [0.0, 0.0], "set_use_battery": True}
        )
        res_zero = opt_zero.perform_naive_mpc_optim(
            df, pv, load_s, len(load), soc_init=0.5, soc_final=0.5
        )
        opt_off = build_optimization(
            optim_overrides={"capacity_cost_per_kw": 0.0, "set_use_battery": True}
        )
        res_off = opt_off.perform_naive_mpc_optim(
            df, pv, load_s, len(load), soc_init=0.5, soc_final=0.5
        )
        self.assertIn(opt_zero.optim_status, VALID_OPTIMAL)
        self.assertTrue(opt_zero._capacity_multi)
        self.assertEqual(opt_zero.n_capacity_components, 2)
        self.assertIsNone(opt_zero.vars["peak_import_k"][0])
        self.assertIsNone(opt_zero.vars["peak_import_k"][1])
        np.testing.assert_allclose(
            res_zero["P_grid_pos"].to_numpy(), res_off["P_grid_pos"].to_numpy(), atol=1e-3
        )

    def test_both_component_costs_are_summed_into_the_one_objective(self):
        """Both components' peak terms sit in the SAME single objective. With
        the dispatch pinned (flat zero load, both peaks set only by their
        incumbent floors) the objective drop caused by component 1's rate is
        exactly rate1 * peak1 / 1000 - i.e. the sum sum_k rate_k*peak_k/1000."""
        load = [0.0] * 6
        pv, load_s, df = make_scenario(load)

        opt_a = build_optimization(optim_overrides={"capacity_cost_per_kw": [3.0, 0.0]})
        opt_a.perform_naive_mpc_optim(df, pv, load_s, 6, current_period_peak=[2000.0, 4000.0])
        opt_b = build_optimization(optim_overrides={"capacity_cost_per_kw": [3.0, 6.0]})
        opt_b.perform_naive_mpc_optim(df, pv, load_s, 6, current_period_peak=[2000.0, 4000.0])
        self.assertIn(opt_a.optim_status, VALID_OPTIMAL)
        self.assertIn(opt_b.optim_status, VALID_OPTIMAL)
        # objective is maximised profit; adding a 6 ¤/kW charge on a pinned
        # 4 kW peak costs exactly 6 * 4 = 24.
        self.assertAlmostEqual(opt_a.prob.value - opt_b.prob.value, 6.0 * 4000.0 / 1000.0, places=2)

    # ---- N>1 interval aggregation, per component -------------------------

    def test_per_component_measurement_interval_basis(self):
        load = [3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0, 3000.0]
        pv, load_s, df = make_scenario(load)
        opt = build_optimization(
            optim_overrides={
                "capacity_cost_per_kw": [1.0, 1.0],
                "capacity_charge_interval_timesteps": [1, 4],
            }
        )
        self.assertEqual(opt._capacity_interval_aggregation_active_list, [False, True])
        opt.perform_naive_mpc_optim(df, pv, load_s, len(load))
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        # Same flat 3 kW dispatch: N=1 component prices 3 kW; the N=4 component
        # also averages to 3 kW here. Distinct machinery, same number.
        self.assertAlmostEqual(opt.vars["peak_import_k"][0].value, 3000.0, places=1)
        self.assertAlmostEqual(opt.vars["peak_import_k"][1].value, 3000.0, places=1)

    def test_shared_interval_basis_broadcasts_from_scalar_and_singleton(self):
        # Both a bare scalar N and a singleton list [N] must canonicalise to the
        # same shared measurement basis broadcast to every component.
        for iv in (6, [6]):
            opt = build_optimization(
                optim_overrides={
                    "capacity_cost_per_kw": [1.0, 2.0],
                    "capacity_charge_interval_timesteps": iv,
                }
            )
            self.assertEqual(opt.optim_conf["capacity_charge_interval_timesteps"], 6)
            self.assertEqual(opt._capacity_charge_interval_timesteps_list, [6, 6])
            self.assertEqual(opt._capacity_interval_aggregation_active_list, [True, True])

    def test_per_component_partial_open_interval_history(self):
        # N=4, component 1 has 2 realised samples [4000, 4000]; its first
        # completed interval ends at decision index 1 and averages
        # (4000+4000+2000+2000)/4 = 3000 W.
        load = [2000.0, 2000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pv, load_s, df = make_scenario(load)
        opt = build_optimization(
            optim_overrides={
                "capacity_cost_per_kw": [1.0, 1.0],
                "capacity_charge_interval_timesteps": [1, 4],
            }
        )
        opt.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            len(load),
            capacity_charge_current_interval_history=[[], [4000.0, 4000.0]],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        self.assertAlmostEqual(opt.vars["peak_import_k"][0].value, 2000.0, places=1)
        self.assertAlmostEqual(opt.vars["peak_import_k"][1].value, 3000.0, places=1)

    def test_incomplete_terminal_interval_is_unpriced(self):
        # horizon 7, N=4: only ONE full interval completes (indices 0..3); the
        # trailing 3 steps never complete and cannot raise either component's
        # peak (#1079 semantics, per component).
        load = [1000.0, 1000.0, 1000.0, 1000.0, 9000.0, 9000.0, 9000.0]
        pv, load_s, df = make_scenario(load)
        opt = build_optimization(
            optim_overrides={
                "capacity_cost_per_kw": [1.0, 1.0],
                "capacity_charge_interval_timesteps": [4, 4],
            }
        )
        opt.perform_naive_mpc_optim(df, pv, load_s, len(load))
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        # Only the completed 1 kW interval is priced; the 9 kW tail is ignored.
        self.assertAlmostEqual(opt.vars["peak_import_k"][0].value, 1000.0, places=1)
        self.assertAlmostEqual(opt.vars["peak_import_k"][1].value, 1000.0, places=1)

    def test_misalignment_warnings_are_attributed_per_component_and_per_param(self):
        load = [2000.0] * 12
        pv, load_s, df = make_scenario(load)
        opt = build_optimization(
            optim_overrides={
                "capacity_cost_per_kw": [1.0, 1.0],
                "capacity_charge_interval_timesteps": [6, 6],
            }
        )
        with self.assertLogs(level="WARNING") as logs:
            opt.perform_naive_mpc_optim(
                df,
                pv,
                load_s,
                12,
                # component 0: window misaligned; component 1: consideration misaligned
                capacity_charge_window=[[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1], [1] * 12],
                capacity_charge_consideration=[[1] * 12, [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]],
            )
        blob = "\n".join(logs.output)
        self.assertIn(
            "capacity_cost_per_kw[0]: capacity_charge_window changes within a completed",
            blob,
        )
        self.assertIn(
            "capacity_cost_per_kw[1]: capacity_charge_consideration changes within a completed",
            blob,
        )
        self.assertNotIn(
            "capacity_cost_per_kw[0]: capacity_charge_consideration changes within a completed",
            blob,
        )

    # ---- runtime input validation --------------------------------------

    def test_wrong_length_runtime_list_is_ignored_wholesale_with_a_warning(self):
        load = [1000.0, 8000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        pv, load_s, df = make_scenario(load)
        opt = build_optimization(optim_overrides={"capacity_cost_per_kw": [2.0, 2.0]})
        with self.assertLogs(level="WARNING") as logs:
            opt.perform_naive_mpc_optim(
                df,
                pv,
                load_s,
                len(load),
                # 3 entries for a K=2 charge - must be dropped for BOTH, not
                # partially mapped.
                current_period_peak=[100.0, 200.0, 300.0],
            )
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        self.assertTrue(any("current_period_peak has 3 entries" in line for line in logs.output))
        self.assertAlmostEqual(opt.param_current_period_peak_k[0].value, 0.0)
        self.assertAlmostEqual(opt.param_current_period_peak_k[1].value, 0.0)

    def test_bare_scalar_runtime_value_is_rejected_when_multi(self):
        load = [1000.0, 8000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        pv, load_s, df = make_scenario(load)
        opt = build_optimization(optim_overrides={"capacity_cost_per_kw": [2.0, 2.0]})
        with self.assertLogs(level="WARNING") as logs:
            opt.perform_naive_mpc_optim(df, pv, load_s, len(load), current_period_peak=500.0)
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        self.assertTrue(
            any("current_period_peak must be a list of 2 entries" in line for line in logs.output)
        )

    def test_one_bad_rate_entry_disables_only_that_component(self):
        with self.assertLogs(level="WARNING") as logs:
            opt = build_optimization(optim_overrides={"capacity_cost_per_kw": [3.0, "oops", 5.0]})
        self.assertEqual(opt._capacity_cost_per_kw_list, [3.0, 0.0, 5.0])
        self.assertIsNone(opt.vars["peak_import_k"][1])
        self.assertIsNotNone(opt.vars["peak_import_k"][0])
        self.assertIsNotNone(opt.vars["peak_import_k"][2])
        self.assertTrue(any("capacity_cost_per_kw[1]" in line for line in logs.output))

    def test_wrong_length_interval_list_raises_not_reinterprets_tariff_metric(self):
        # A K=3 rate list with a length-2 interval list must be an explicit
        # error - never a silent drop to N=1 (which would turn a 30-minute
        # average-demand tariff into a native 5-minute peak tariff).
        with self.assertRaises(ValueError) as ctx:
            build_optimization(
                optim_overrides={
                    "capacity_cost_per_kw": [1.0, 1.0, 1.0],
                    "capacity_charge_interval_timesteps": [6, 6],
                }
            )
        self.assertIn("capacity_charge_interval_timesteps has 2 entries", str(ctx.exception))
        # A K=1 rate with a 2-entry interval list is equally an error.
        with self.assertRaises(ValueError):
            build_optimization(
                optim_overrides={
                    "capacity_cost_per_kw": 3.0,
                    "capacity_charge_interval_timesteps": [6, 3],
                }
            )

    def test_runtime_masks_do_not_leak_between_ticks_or_components(self):
        load = [0.0, 2000.0, 0.0, 0.0, 5000.0, 0.0]
        pv, load_s, df = make_scenario(load)
        opt = build_optimization(optim_overrides={"capacity_cost_per_kw": [2.0, 2.0]})

        opt.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            len(load),
            capacity_charge_window=[[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]],
        )
        self.assertAlmostEqual(opt.vars["peak_import_k"][0].value, 2000.0, places=2)
        self.assertAlmostEqual(opt.vars["peak_import_k"][1].value, 5000.0, places=2)

        # Next tick omits every mask: both components must fall back to the
        # full-horizon peak, not reuse the previous windows.
        opt.perform_naive_mpc_optim(df, pv, load_s, len(load))
        self.assertAlmostEqual(opt.vars["peak_import_k"][0].value, 5000.0, places=2)
        self.assertAlmostEqual(opt.vars["peak_import_k"][1].value, 5000.0, places=2)

    # ---- DPP / warm-start / problem identity ---------------------------

    def test_runtime_updates_preserve_dpp_and_problem_identity(self):
        load = [1000.0, 2000.0, 1000.0, 1000.0, 3000.0, 1000.0]
        pv, load_s, df = make_scenario(load)
        opt = build_optimization(
            optim_overrides={
                "capacity_cost_per_kw": [3.0, 7.0],
                "capacity_charge_interval_timesteps": [1, 2],
            }
        )
        opt.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            len(load),
            current_period_peak=[200.0, 400.0],
            capacity_charge_window=[[1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1]],
            capacity_charge_consideration=[[1] * 6, [1, 1, 1, 1, 0, 1]],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        self.assertTrue(opt.prob.is_dcp(dpp=True))
        self.assertTrue(opt.prob.is_dpp())
        prob_id = id(opt.prob)

        opt.perform_naive_mpc_optim(
            df,
            pv,
            load_s,
            len(load),
            current_period_peak=[900.0, 100.0],
            capacity_charge_window=[[0, 0, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]],
            capacity_charge_consideration=[[1, 1, 0, 1, 1, 1], [1] * 6],
            capacity_charge_current_interval_history=[[], [1500.0]],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        self.assertTrue(opt.prob.is_dpp())
        self.assertEqual(id(opt.prob), prob_id, msg="a runtime-only update rebuilt the problem")

    def test_prediction_horizon_resize_rebuilds_per_component_params(self):
        load6 = [1000.0, 2000.0, 1000.0, 1000.0, 3000.0, 1000.0]
        pv6, load6_s, df6 = make_scenario(load6)
        opt = build_optimization(
            optim_overrides={
                "capacity_cost_per_kw": [2.0, 2.0],
                "capacity_charge_interval_timesteps": [1, 3],
            }
        )
        opt.perform_naive_mpc_optim(df6, pv6, load6_s, 6)
        self.assertEqual(opt.param_capacity_window_k[0].value.shape[0], 6)

        load9 = load6 + [1000.0, 1000.0, 1000.0]
        pv9, load9_s, df9 = make_scenario(load9)
        opt.perform_naive_mpc_optim(df9, pv9, load9_s, 9)
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        self.assertEqual(opt.param_capacity_window_k[0].value.shape[0], 9)
        self.assertEqual(opt.param_capacity_interval_matrix_k[1].shape[1], 9)

    # ---- maximal interaction ------------------------------------------

    def test_maximal_interaction_battery_k2_ngt1_windows_consideration_incumbents(self):
        """One realistic flexible-dispatch solve exercising every Part B
        dimension at once: battery + K=2 + both rates > 0 + per-component
        N > 1 + independent windows + independent incumbents + independent
        consideration + open-interval history + time-varying energy price.
        Assert economic invariants, not the exact battery profile."""
        n = 12
        idx = pd.date_range("2026-02-01", periods=n, freq="5min", tz="Europe/Tallinn")
        # A big import spike in each component's window; cheap energy early so the
        # battery has a real reason to pre-charge and shave.
        load = np.full(n, 1000.0)
        load[3] = 9000.0  # inside component 0's window
        load[9] = 9000.0  # inside component 1's window
        p_pv = pd.Series(np.zeros(n), index=idx)
        p_load = pd.Series(load, index=idx, dtype=float)
        df = pd.DataFrame(index=idx)
        df["unit_load_cost"] = [0.05] * 6 + [0.40] * 6
        df["unit_prod_price"] = 0.05

        opt = build_optimization(
            optim_overrides={
                "capacity_cost_per_kw": [6.0, 9.0],
                "capacity_charge_interval_timesteps": [3, 3],
                "set_use_battery": True,
                "weight_battery_discharge": 0.0,
                "weight_battery_charge": 0.0,
            },
            plant_overrides={
                "battery_minimum_state_of_charge": 0.0,
                "battery_maximum_state_of_charge": 1.0,
                "battery_discharge_efficiency": 1.0,
                "battery_charge_efficiency": 1.0,
            },
        )
        res = opt.perform_naive_mpc_optim(
            df,
            p_pv,
            p_load,
            n,
            soc_init=0.5,
            soc_final=0.5,
            current_period_peak=[800.0, 1200.0],
            capacity_charge_window=[
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            ],
            capacity_charge_consideration=[[1] * 12, [1] * 12],
            capacity_charge_current_interval_history=[[500.0, 500.0], [700.0, 700.0]],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        self.assertTrue(opt.prob.is_dcp(dpp=True))
        prob_id = id(opt.prob)

        p0 = opt.vars["peak_import_k"][0].value
        p1 = opt.vars["peak_import_k"][1].value
        # Both epigraphs participate and each stays between its own incumbent
        # floor and the un-shaved 3-step spike average (~(9000+1000+1000)/3
        # ~= 3667): the battery shaves each spike but cannot go below the
        # already-realised incumbent.
        self.assertGreaterEqual(p0, 800.0 - 1e-6)
        self.assertGreaterEqual(p1, 1200.0 - 1e-6)
        self.assertLess(p0, 3600.0)
        self.assertLess(p1, 3600.0)
        # No cross-leak: component 0's window is the first half, so its priced
        # average cannot be driven by the load[9] spike in the second half, and
        # vice versa - the battery physically shaves both spikes.
        self.assertLess(res["P_grid_pos"].iloc[3], 9000.0 - 500.0)
        self.assertLess(res["P_grid_pos"].iloc[9], 9000.0 - 500.0)

        # Consideration only narrows: dropping component 1's second-window
        # consideration must not raise component 0's peak and must not lift
        # either incumbent.
        opt.perform_naive_mpc_optim(
            df,
            p_pv,
            p_load,
            n,
            soc_init=0.5,
            soc_final=0.5,
            current_period_peak=[800.0, 1200.0],
            capacity_charge_window=[
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            ],
            capacity_charge_consideration=[[1] * 12, [0] * 12],
            capacity_charge_current_interval_history=[[500.0, 500.0], [700.0, 700.0]],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        self.assertEqual(
            id(opt.prob), prob_id, msg="a runtime-only consideration change rebuilt the problem"
        )
        self.assertAlmostEqual(opt.vars["peak_import_k"][0].value, p0, delta=50.0)
        # Component 1: no prospective consideration, so its peak drops back to
        # its realised incumbent (700 W history average <= 1200 W incumbent).
        self.assertAlmostEqual(opt.vars["peak_import_k"][1].value, 1200.0, delta=1.0)

    # ---- dayahead / perfect-forecast smoke ---------------------------

    def _dayahead_scenario(self, n=24):
        idx = pd.date_range("2026-02-01", periods=n, freq="5min", tz="Europe/Tallinn")
        load = np.full(n, 1000.0)
        load[5] = 7000.0
        p_pv = pd.Series(np.zeros(n), index=idx)
        p_load = pd.Series(load, index=idx, dtype=float)
        df = pd.DataFrame(index=idx)
        df["unit_load_cost"] = 0.20
        df["unit_prod_price"] = 0.20
        return p_pv, p_load, df

    def test_dayahead_optim_k2_smoke(self):
        p_pv, p_load, df = self._dayahead_scenario()
        opt = build_optimization(optim_overrides={"capacity_cost_per_kw": [3.0, 5.0]})
        res = opt.perform_dayahead_forecast_optim(df, p_pv, p_load)
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertEqual(len(opt.vars["peak_import_k"]), 2)
        # dayahead takes no runtime masks: both components price the full-horizon
        # peak of the same single dispatch.
        self.assertAlmostEqual(
            opt.vars["peak_import_k"][0].value, opt.vars["peak_import_k"][1].value, places=1
        )

    def test_perfect_and_dayahead_core_k2_ngt1_smoke(self):
        p_pv, p_load, df = self._dayahead_scenario()
        opt = build_optimization(
            optim_overrides={
                "capacity_cost_per_kw": [3.0, 5.0],
                "capacity_charge_interval_timesteps": 6,
            }
        )
        # perform_optimization is the shared core both dayahead-optim and
        # perfect-optim call after their own input prep.
        res = opt.perform_optimization(
            df,
            p_pv.to_numpy().ravel(),
            p_load.to_numpy().ravel(),
            df["unit_load_cost"].to_numpy(),
            df["unit_prod_price"].to_numpy(),
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertTrue(opt.prob.is_dcp(dpp=True))
        self.assertEqual(opt._capacity_charge_interval_timesteps_list, [6, 6])

    def test_perfect_forecast_optim_wrapper_k2_smoke(self):
        """The actual perform_perfect_forecast_optim entry path (not just the
        shared core) must run under a K=2 config."""
        tz = "Europe/Tallinn"
        # build_optimization uses delta_forecast_daily = 2h -> a 24-step day.
        idx = pd.date_range("2026-02-01", periods=48, freq="5min", tz=tz)
        load = np.full(48, 1000.0)
        load[10] = 6000.0
        df = pd.DataFrame(index=idx)
        df["pv"] = 0.0
        df["load_positive"] = load
        df["unit_load_cost"] = 0.20
        df["unit_prod_price"] = 0.20
        # perform_perfect_forecast_optim drops the last day, so pass 2.
        days_list = pd.date_range("2026-02-01", periods=2, freq="D", tz=tz)

        opt = build_optimization(
            optim_overrides={
                "capacity_cost_per_kw": [3.0, 5.0],
                "capacity_charge_interval_timesteps": 6,
            }
        )
        res = opt.perform_perfect_forecast_optim(df, days_list)
        self.assertIn(opt.optim_status, VALID_OPTIMAL)
        self.assertIsInstance(res, pd.DataFrame)
        self.assertGreater(len(res), 0)
        self.assertIn("P_grid", res.columns)
        self.assertEqual(len(opt.vars["peak_import_k"]), 2)
        self.assertIsNotNone(opt.vars["peak_import_k"][0])
        self.assertIsNotNone(opt.vars["peak_import_k"][1])

    # ---- alternate cost function ------------------------------------

    def test_k2_capacity_charge_under_cost_costfun(self):
        """The capacity objective term must not be gated to costfun='profit'."""
        n = 12
        idx = pd.date_range("2026-02-01", periods=n, freq="5min", tz="Europe/Tallinn")
        load = np.full(n, 1000.0)
        load[4] = 8000.0
        p_pv = pd.Series(np.zeros(n), index=idx)
        p_load = pd.Series(load, index=idx, dtype=float)
        df = pd.DataFrame(index=idx)
        df["unit_load_cost"] = 0.20
        df["unit_prod_price"] = 0.20

        def _run(cap):
            opt = build_optimization(
                optim_overrides={
                    "capacity_cost_per_kw": cap,
                    "set_use_battery": True,
                    "set_nodischarge_to_grid": True,
                    "weight_battery_discharge": 0.1,
                    "weight_battery_charge": 0.1,
                },
                plant_overrides={
                    "battery_minimum_state_of_charge": 0.0,
                    "battery_maximum_state_of_charge": 1.0,
                    "battery_discharge_efficiency": 1.0,
                    "battery_charge_efficiency": 1.0,
                },
            )
            opt.costfun = "cost"
            # Balanced SOC: the battery only moves for an economic reason, so any
            # spike shaving in the "on" run is caused by the capacity term.
            res = opt.perform_naive_mpc_optim(df, p_pv, p_load, n, soc_init=0.5, soc_final=0.5)
            self.assertIn(opt.optim_status, VALID_OPTIMAL)
            return opt, res

        opt_off, res_off = _run([0.0, 0.0])
        opt_on, res_on = _run([50.0, 50.0])
        self.assertTrue(opt_on.prob.is_dcp(dpp=True))
        self.assertEqual(len(opt_on.vars["peak_import_k"]), 2)
        self.assertGreater(res_off["P_grid_pos"].iloc[4], 6000.0)
        # A real K=2 capacity charge under costfun='cost' still shaves the spike.
        self.assertLess(res_on["P_grid_pos"].iloc[4], res_off["P_grid_pos"].iloc[4] - 1000.0)

    # ---- cache identity ---------------------------------------------

    def test_scalar_and_singleton_list_have_identical_cache_key(self):
        from emhass.command_line import OptimizationCache

        rh = {"optimization_time_step": pd.to_timedelta(5, "minutes")}
        plant = {"inverter_is_hybrid": False, "compute_curtailment": False}
        base = {"number_of_deferrable_loads": 0}

        key_scalar = OptimizationCache._compute_cache_key(
            {**base, "capacity_cost_per_kw": 3.0, "capacity_charge_interval_timesteps": 6},
            plant,
            "profit",
            rh,
        )
        key_singleton = OptimizationCache._compute_cache_key(
            {
                **base,
                "capacity_cost_per_kw": [3.0],
                "capacity_charge_interval_timesteps": [6],
            },
            plant,
            "profit",
            rh,
        )
        key_empty = OptimizationCache._compute_cache_key(
            {**base, "capacity_cost_per_kw": []}, plant, "profit", rh
        )
        key_disabled = OptimizationCache._compute_cache_key(
            {**base, "capacity_cost_per_kw": 0.0}, plant, "profit", rh
        )
        key_k2 = OptimizationCache._compute_cache_key(
            {**base, "capacity_cost_per_kw": [3.0, 7.0]}, plant, "profit", rh
        )
        self.assertEqual(key_scalar, key_singleton)
        self.assertEqual(key_empty, key_disabled)
        self.assertNotEqual(key_scalar, key_k2)


if __name__ == "__main__":
    unittest.main()
