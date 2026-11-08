#!/usr/bin/env python
"""Frozen-number regression pins for today's one-inverter optimizer (#1150).

Every scenario below runs one inverter (``number_of_inverters`` absent) through the
optimizer and compares the status, the objective and the key result columns against
numbers captured on upstream/master commit 60555f73, before the per-inverter optimizer
work for #1150 touches optimization.py. They are the proof that one inverter stays
mathematically unchanged.

Do not re-capture these values to make a multi-inverter change pass. A failure here means
the one-inverter model changed; fix the change, not the pin. A pin should only move in a
PR whose stated purpose is to change the one-inverter model, with the reason given in
that PR.

Determinism: ``lp_solver_mip_rel_gap`` is 0, all inputs are fixed and every efficiency is
below 1 with distinct prices, so there are no ties. A per-column pin is only kept where
the optimum is unique (re-solved under tiny random price perturbations and a different
``num_threads`` at capture time, columns identical). Where it is not unique (the split
between two batteries) only the fleet totals are pinned.
"""

import logging
import pathlib
from unittest import mock

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from emhass.optimization import Optimization

TEST_ROOT = pathlib.Path(__file__).resolve().parents[1]

OBJECTIVE_TOL = 1e-6  # currency
POWER_TOL = 5e-3  # W
SOC_TOL = 1e-5  # fraction

# 10 half-hour steps. The shape is chosen so that, with the plant below, the AC output cap
# (3000 W) and the AC input cap (2500 W) both bind, the battery charges from the grid in
# the two cheap steps, PV charges it midday and the evening export is battery-fed.
INDEX = pd.date_range("2026-02-01", periods=10, freq="30min", tz="Europe/Tallinn")
PV = [0, 0, 300, 1500, 4200, 5800, 4600, 1800, 200, 0]
PV_OVER_CAP = [0, 0, 300, 1500, 4200, 7400, 4600, 1800, 200, 0]  # step 5 exceeds what fits
LOAD = [650, 600, 700, 900, 1200, 1000, 1400, 1600, 1900, 1700]
LOAD_COST = [0.02, 0.03, 0.29, 0.18, 0.11, 0.09, 0.14, 0.37, 0.44, 0.41]
PROD_PRICE = [0.052, 0.047, 0.058, 0.061, 0.043, 0.039, 0.046, 0.071, 0.083, 0.077]

OUTPUT_CAP, INPUT_CAP = 3000, 2500
DC_AC_CURVE = [[500, 0.90], [1500, 0.96], [3000, 0.94]]
AC_DC_CURVE = [[500, 0.88], [1200, 0.93], [2500, 0.95]]
TWO_BATTERIES = {
    "number_of_batteries": 2,
    "battery_nominal_energy_capacity": [8000, 6000],
    "battery_discharge_power_max": [2000, 1500],
    "battery_charge_power_max": [1800, 1200],
    "battery_discharge_efficiency": [0.95, 0.92],
    "battery_charge_efficiency": [0.93, 0.91],
    "battery_minimum_state_of_charge": [0.2, 0.25],
    "battery_maximum_state_of_charge": [0.9, 0.85],
    "battery_target_state_of_charge": [0.5, 0.5],
}
BOTH_CURVES = {
    "inverter_power_curve_dc_ac": DC_AC_CURVE,
    "inverter_power_curve_ac_dc": AC_DC_CURVE,
}
STRESS = {"inverter_stress_cost": 0.3, "inverter_stress_segments": 6}
AC_COUPLED = {"inverter_is_hybrid": False, "compute_curtailment": True}

# name -> inputs. "paths" are the entry points the same numbers are pinned through:
# dayahead, naive MPC (same horizon, so identical numbers), a second solve on the cached
# problem, or a relaxed-LP rescue.
CASES = {
    "hybrid_scalar_caps": {"paths": ("dayahead", "mpc", "cached")},
    "hybrid_dc_ac_curve": {"plant": {"inverter_power_curve_dc_ac": DC_AC_CURVE}},
    "hybrid_ac_dc_curve": {"plant": {"inverter_power_curve_ac_dc": AC_DC_CURVE}},
    "hybrid_both_curves_two_batteries": {
        "plant": {**BOTH_CURVES, **TWO_BATTERIES},
        "soc": ([0.3, 0.4], [0.5, 0.5]),
        "paths": ("dayahead", "mpc", "cached"),
    },
    "hybrid_curtailment": {
        "plant": {"compute_curtailment": True},
        "pv": PV_OVER_CAP,
        "paths": ("dayahead", "mpc"),
    },
    "hybrid_nodischarge_to_grid": {"optim": {"set_nodischarge_to_grid": True}},
    "acoupled_nodischarge_curtailment": {
        "optim": {"set_nodischarge_to_grid": True},
        "plant": {**AC_COUPLED, "maximum_power_to_grid": 1500},
        "paths": ("dayahead", "mpc"),
    },
    "hybrid_inverter_stress": {"plant": STRESS},
    "hybrid_nocharge_from_grid": {"optim": {"set_nocharge_from_grid": True}},
    "hybrid_both_curves_relaxed_retry": {"plant": BOTH_CURVES, "paths": ("relaxed",)},
}

# Captured on 60555f73. Do not edit, do not re-capture (see the module docstring).
# fmt: off
PINS = {
    "hybrid_scalar_caps": {
        "status": "Optimal",
        "objective": 0.08410622700000008,
        "columns": {
            "P_batt": [-2350.0, -2350.0, 429.1666666666667, -562.5, -2950.0,
                       -3000.0, -3000.000000000002, 757.5770833333358, 2925.0, 3125.0],
            "SOC_opt": [0.37805357142857143, 0.4561071428571428, 0.43997305764411027, 0.45865609335839597, 0.5566382362155389,
                        0.656281093358396, 0.7559239505012532, 0.7274436090225563, 0.6174812030075187, 0.5],
            "P_grid_pos": [3150.0, 3100.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0],
            "P_grid_neg": [0.0, 0.0, 0.0, 0.0, 0.0,
                           -1688.000000000001, -135.999999999999, -855.2740000000022, -1100.0, -1300.0],
            "P_hybrid_inverter": [-2500.0, -2500.0, 700.0, 900.0, 1200.0,
                                  2688.0, 1535.9999999999982, 2455.274000000002, 3000.0, 3000.0],
        },
    },
    "hybrid_dc_ac_curve": {
        "status": "Optimal",
        "objective": 0.07634839600410498,
        "columns": {
            "P_batt": [-2350.0, -2349.9999999999964, 456.9444444444445, 0.0, -2939.583333333333,
                       -3000.0, -3000.0, 90.64870715129612, 2991.4893617021285, 3191.489361702128],
            "SOC_opt": [0.37805357142857143, 0.4561071428571427, 0.4389287802840433, 0.4389287802840433, 0.5365649409983291,
                        0.6362077981411862, 0.7358506552840434, 0.732442809150536, 0.6199808030715086, 0.5],
            "P_grid_pos": [3149.9999999999995, 3099.9999999999964, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0],
            "P_grid_neg": [0.0, 0.0, 0.0, -537.9310344827586, 0.0,
                           -1639.5102040816323, -134.53061224489747, -202.16468707482608, -1100.0, -1299.9999999999998],
            "P_hybrid_inverter": [-2499.9999999999995, -2499.9999999999964, 700.0, 1437.9310344827586, 1200.0,
                                  2639.5102040816323, 1534.5306122448974, 1802.1646870748261, 3000.0, 3000.0],
        },
    },
    "hybrid_ac_dc_curve": {
        "status": "Optimal",
        "objective": 0.08561171100000003,
        "columns": {
            "P_batt": [-2375.0, -2375.0, 429.16666666666663, -562.5, -2949.999999999999,
                       -3000.0, -3000.0, 801.7520833333298, 2925.0000000000014, 3125.000000000003],
            "SOC_opt": [0.37888392857142855, 0.45776785714285717, 0.44163377192982456, 0.46031680764411026, 0.5582989505012531,
                        0.6579418076441103, 0.7575846647869674, 0.7274436090225566, 0.617481203007519, 0.5],
            "P_grid_pos": [3150.0, 3100.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0],
            "P_grid_neg": [0.0, 0.0, 0.0, 0.0, 0.0,
                           -1688.0, -135.99999999999974, -897.6819999999966, -1100.0000000000011, -1300.000000000003],
            "P_hybrid_inverter": [-2500.0, -2500.0, 700.0, 900.0, 1200.0,
                                  2688.0, 1535.9999999999998, 2497.6819999999966, 3000.000000000001, 3000.0000000000027],
        },
    },
    "hybrid_both_curves_two_batteries": {
        "status": "Optimal",
        "objective": 0.10912947674854383,
        "columns": {
            "P_batt": [-2374.9999999999995, -2375.0000000000005, 456.9444444447039, 0.0, -2939.583333333333,
                       -2999.999999999999, -3000.000000000002, 1093.6959988179663, 2991.489361702128, 3191.489361702127],
            "P_grid_pos": [3149.9999999999995, 3100.0000000000005, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0],
            "P_grid_neg": [0.0, 0.0, 0.0, -537.9310344827588, 0.0,
                           -1639.5102040816328, -134.53061224489608, -1125.7870095238102, -1099.9999999999995, -1299.9999999999998],
            "P_hybrid_inverter": [-2499.9999999999995, -2500.0000000000005, 700.0, 1437.9310344827588, 1200.0,
                                  2639.5102040816328, 1534.530612244896, 2725.7870095238104, 2999.9999999999995, 3000.0],
        },
    },
    "hybrid_curtailment": {
        "status": "Optimal",
        "objective": 0.09012647700000001,
        "columns": {
            "P_batt": [-2350.0, -2350.0, 429.16666666666674, -562.4999999999999, -2950.0,
                       -3000.0, -3000.0, 757.5770833333337, 2925.0, 3125.0],
            "SOC_opt": [0.37805357142857143, 0.4561071428571428, 0.43997305764411027, 0.45865609335839597, 0.5566382362155389,
                        0.656281093358396, 0.7559239505012532, 0.7274436090225563, 0.6174812030075187, 0.5],
            "P_grid_pos": [3150.0, 3100.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0],
            "P_grid_neg": [0.0, 0.0, 0.0, 0.0, 0.0,
                           -2000.0, -136.0, -855.2740000000003, -1100.0, -1300.0],
            "P_hybrid_inverter": [-2500.0, -2500.0, 700.0, 900.0, 1200.0,
                                  3000.0, 1536.0, 2455.2740000000003, 3000.0, 3000.0],
            "P_PV_curtailment": [0.0, 0.0, 0.0, 0.0, 0.0,
                                 1275.0, 0.0, 0.0, 0.0, 0.0],
        },
    },
    "hybrid_nodischarge_to_grid": {
        "status": "Optimal",
        "objective": 0.04574631013016412,
        "columns": {
            "P_batt": [-2350.0, -2350.0, 429.16666666666674, 0.0, -1350.372571212978,
                       -3000.000000000001, -1475.0, 0.0, 1779.1666666666667, 1770.8333333333335],
            "SOC_opt": [0.37805357142857143, 0.4561071428571428, 0.43997305764411027, 0.43997305764411027, 0.48482471804511273,
                        0.58446757518797, 0.6334586466165413, 0.6334586466165413, 0.5665726817042607, 0.5],
            "P_grid_pos": [3150.0, 3100.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0],
            "P_grid_neg": [0.0, 0.0, 0.0, -540.0, -1535.6423316355413,
                           -1687.999999999999, -1599.9999999999998, -128.0, 0.0, 0.0],
            "P_hybrid_inverter": [-2500.0, -2500.0, 700.0, 1440.0, 2735.642331635541,
                                  2687.999999999999, 3000.0, 1728.0, 1900.0, 1700.0],
        },
    },
    "acoupled_nodischarge_curtailment": {
        "status": "Optimal",
        "objective": 0.059046290322580644,
        "columns": {
            "P_batt": [-3000.0, -1122.5806451612902, 400.0, 0.0, -1500.0,
                       -3000.0, -1700.0, 0.0, 1700.0, 1700.0],
            "SOC_opt": [0.39964285714285713, 0.43692857142857144, 0.42189097744360904, 0.42189097744360904, 0.4717124060150376,
                        0.5713552631578946, 0.6278195488721805, 0.6278195488721805, 0.5639097744360902, 0.5],
            "P_grid_pos": [3650.0, 1722.5806451612902, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0],
            "P_grid_neg": [0.0, 0.0, 0.0, -600.0, -1500.0,
                           -1500.0, -1500.0, -200.0, 0.0, 0.0],
            "P_PV_curtailment": [0.0, 0.0, 0.0, 0.0, 0.0,
                                 300.0, 0.0, 0.0, 0.0, 0.0],
        },
    },
    "hybrid_inverter_stress": {
        "status": "Optimal",
        "objective": -1.0798104115722844,
        "columns": {
            "P_batt": [-72.90133936992714, 0.0, 429.16666666666674, -979.1666666666665, -3000.0,
                       -3000.0, -3000.0, -237.50000000000014, 1779.1666666666667, 1562.5],
            "SOC_opt": [0.30242136591478685, 0.30242136591478685, 0.28628728070175424, 0.3188096021303257, 0.41845245927318286,
                        0.51809531641604, 0.6177381735588972, 0.62562656641604, 0.5587406015037593, 0.4999999999999999],
            "P_grid_pos": [727.5546163509897, 600.0, 0.0, 400.0, 48.0,
                           0.0, 0.0, 100.0000000000002, 0.0, 200.0],
            "P_grid_neg": [0.0, 0.0, 0.0, 0.0, 0.0,
                           -1688.0, -136.0, 0.0, 0.0, 0.0],
            "P_hybrid_inverter": [-77.55461635098632, 0.0, 700.0, 500.0, 1152.0,
                                  2688.0, 1536.0, 1499.9999999999998, 1900.0, 1500.0],
            "inv_stress_cost": [0.0019388654087746575, 0.0, 0.02749999999999999, 0.012499999999999997, 0.06899999999999998,
                                0.36419999999999997, 0.11880000000000002, 0.11249999999999998, 0.1825, 0.11249999999999999],
        },
    },
    "hybrid_nocharge_from_grid": {
        "status": "Optimal",
        "objective": -0.053768189907284256,
        "columns": {
            "P_batt": [0.0, 0.0, 429.1666666666667, -1392.0392378796428, -3000.0,
                       -3000.0, -3000.000000000001, -133.33333333333326, 1779.1666666666667, 1770.8333333333335],
            "SOC_opt": [0.3, 0.3, 0.28386591478696743, 0.3301015037593984, 0.42974436090225554,
                        0.5293872180451127, 0.6290300751879698, 0.6334586466165413, 0.5665726817042607, 0.4999999999999999],
            "P_grid_pos": [650.0000153448048, 600.0, 0.0, 796.3576683644569, 48.0,
                           0.0, 0.0, 0.0, 0.0, 0.0],
            "P_grid_neg": [-1.534480481495848e-05, 0.0, 0.0, 0.0, 0.0,
                           -1688.000000000001, -135.999999999999, 0.0, 0.0, 0.0],
            "P_hybrid_inverter": [0.0, 0.0, 700.0, 103.64233163554293, 1152.0,
                                  2688.0, 1535.999999999999, 1600.0, 1900.0, 1700.0],
        },
    },
    "hybrid_both_curves_relaxed_retry": {
        "status": "Optimal (Relaxed)",
        "objective": 0.07779243167757442,
        "columns": {
            "P_batt": [-2375.0, -2374.9999999999964, 456.9444444444443, 0.0, -2939.5833333333353,
                       -3000.000000000001, -2999.9999999999977, 134.8237071512985, 2991.48936170213, 3191.489361702126],
            "SOC_opt": [0.37888392857142855, 0.457767857142857, 0.4405894945697576, 0.4405894945697576, 0.5382256552840434,
                        0.6378685124269006, 0.7375113695697577, 0.732442809150536, 0.6199808030715085, 0.5],
            "P_grid_pos": [3150.0, 3099.9999999999955, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0],
            "P_grid_neg": [0.0, 0.0, 0.0, -537.9310344827587, 0.0,
                           -1639.5102040816312, -134.53061224489932, -242.841748299318, -1100.0, -1299.999999999998],
            "P_hybrid_inverter": [-2500.0, -2499.9999999999955, 700.0, 1437.9310344827586, 1200.0,
                                  2639.510204081631, 1534.5306122448992, 1842.841748299318, 3000.0, 2999.999999999998],
        },
    },
}
# fmt: on


def build_optimization(optim_overrides=None, plant_overrides=None) -> Optimization:
    """Self-contained one-inverter hybrid builder (no forecast data files)."""
    logger = logging.getLogger("inverter_n1_pins")
    logger.handlers = [logging.NullHandler()]
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
    optim_conf.update(optim_overrides or {})
    plant_conf = {
        "inverter_is_hybrid": True,
        "compute_curtailment": False,
        "maximum_power_from_grid": 9000,
        "maximum_power_to_grid": 9000,
        "inverter_ac_output_max": OUTPUT_CAP,
        "inverter_ac_input_max": INPUT_CAP,
        "inverter_efficiency_dc_ac": 0.96,
        "inverter_efficiency_ac_dc": 0.94,
        "battery_discharge_power_max": 3500,
        "battery_charge_power_max": 3000,
        "battery_minimum_state_of_charge": 0.2,
        "battery_maximum_state_of_charge": 0.9,
        "battery_target_state_of_charge": 0.5,
        "battery_nominal_energy_capacity": 14000,
        "battery_discharge_efficiency": 0.95,
        "battery_charge_efficiency": 0.93,
        "battery_stress_cost": 0.0,
        "battery_stress_segments": 10,
    }
    plant_conf.update(plant_overrides or {})
    emhass_conf = {"root_path": TEST_ROOT / "src" / "emhass", "data_path": TEST_ROOT / "data"}
    return Optimization(
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        "unit_load_cost",
        "unit_prod_price",
        "profit",
        emhass_conf,
        logger,
        opt_time_delta=5,
    )


def solve(case, path):
    """Run one case through one entry point; return (status, objective, results)."""
    opt = build_optimization(case.get("optim"), case.get("plant"))
    p_pv = pd.Series(case.get("pv", PV), index=INDEX, dtype=float)
    p_load = pd.Series(LOAD, index=INDEX, dtype=float)
    df_input = pd.DataFrame(
        {"unit_load_cost": LOAD_COST, "unit_prod_price": PROD_PRICE}, index=INDEX
    )
    soc_init, soc_final = case.get("soc", (0.3, 0.5))
    if path == "mpc":
        res = opt.perform_naive_mpc_optim(
            df_input,
            p_pv,
            p_load,
            len(INDEX),
            soc_init=soc_init,
            soc_final=soc_final,
            def_total_hours=[],
            def_total_timestep=[],
            def_start_timestep=[],
            def_end_timestep=[],
        )
        return opt.optim_status, float(opt.prob.value), res
    if path == "cached":
        # Cache-hit path: the first solve builds and caches the problem on different inputs,
        # the pinned solve then re-values the cp.Parameters on that cached problem instead
        # of rebuilding, so a Parameter left stale on the cache hit fails the pin.
        shifted = [s + 0.1 for s in soc_init] if isinstance(soc_init, list) else soc_init + 0.1
        opt.perform_dayahead_forecast_optim(
            df_input * 1.5, p_pv * 0.5, p_load * 0.8, soc_init=shifted, soc_final=soc_final
        )
        cached = opt.prob
        res = opt.perform_dayahead_forecast_optim(
            df_input, p_pv, p_load, soc_init=soc_init, soc_final=soc_final
        )
        assert opt.prob is cached, "the second solve rebuilt the problem instead of reusing it"
        return opt.optim_status, float(opt.prob.value), res
    if path == "dayahead":
        res = opt.perform_dayahead_forecast_optim(
            df_input, p_pv, p_load, soc_init=soc_init, soc_final=soc_final
        )
        return opt.optim_status, float(opt.prob.value), res

    # "relaxed": make the first solver call report a time-out so the one-shot relaxed-LP
    # rescue runs for real. self.prob keeps the failed MILP, so the objective is read from
    # the problem the rescue solved. cvxpy has no public way to report user_limit without
    # a real time-out, so cp.Problem.solve is patched at class level, the same seam as
    # _rescued_solve_patch in tests/test_optimization.py (#1048).
    real_solve = cp.Problem.solve
    calls = []

    def fail_first_solve(prob, *args, **kwargs):
        calls.append(prob)
        if len(calls) == 1:
            prob._status = "user_limit"
            prob._value = None
            return None
        return real_solve(prob, *args, **kwargs)

    with mock.patch.object(cp.Problem, "solve", new=fail_first_solve):
        res = opt.perform_dayahead_forecast_optim(
            df_input, p_pv, p_load, soc_init=soc_init, soc_final=soc_final
        )
    rescued = [prob for prob in calls[1:] if prob.value is not None]
    assert rescued, "the relaxed-LP rescue did not solve"
    return opt.optim_status, float(rescued[-1].value), res


def check_pin(name, status, objective, res):
    """Compare a solve against the frozen numbers of case ``name``."""
    pin = PINS[name]
    np.testing.assert_allclose(
        res["P_PV"].to_numpy(), CASES[name].get("pv", PV), atol=0, rtol=0, err_msg=f"{name}: P_PV"
    )
    assert status == pin["status"], f"{name}: status"
    assert objective == pytest.approx(pin["objective"], abs=OBJECTIVE_TOL), f"{name}: objective"
    for col, expected in pin["columns"].items():
        tol = SOC_TOL if col.startswith("SOC_opt") else POWER_TOL
        np.testing.assert_allclose(
            res[col].to_numpy(), expected, atol=tol, rtol=0, err_msg=f"{name}: {col}"
        )
    # One inverter keeps its legacy column set exactly, so no per-inverter column (P_PV_0,
    # P_hybrid_inverter_0, ...) can appear next to the pinned ones.
    assert set(res.columns) == expected_columns(name), f"{name}: result columns"


# The result columns on 60555f73. The optional ones exist exactly where they are pinned.
BASE_COLUMNS = {
    "P_Load", "P_PV", "P_batt", "P_grid", "P_grid_neg", "P_grid_pos", "cost_fun_profit",
    "cost_profit", "maximum_power_from_grid", "maximum_power_to_grid", "optim_status",
    "unit_load_cost", "unit_prod_price",
}  # fmt: skip
OPTIONAL_COLUMNS = ("P_hybrid_inverter", "P_PV_curtailment", "inv_stress_cost")
PER_BATTERY_COLUMNS = ("SOC_opt", "soc_deficit_cost", "soc_surplus_cost")


def expected_columns(name):
    n_batt = CASES[name].get("plant", {}).get("number_of_batteries", 1)
    if n_batt == 1:
        battery = set(PER_BATTERY_COLUMNS)
    else:
        battery = {f"{c}_{k}" for c in ("P_batt", *PER_BATTERY_COLUMNS) for k in range(n_batt)}
    optional = {c for c in OPTIONAL_COLUMNS if c in PINS[name]["columns"]}
    return BASE_COLUMNS | battery | optional


RUNS = [(name, path) for name, case in CASES.items() for path in case.get("paths", ("dayahead",))]


@pytest.mark.parametrize(("name", "path"), RUNS, ids=[f"{n}-{p}" for n, p in RUNS])
def test_one_inverter_numbers_are_frozen(name, path):
    check_pin(name, *solve(CASES[name], path))


def _col(name, column):
    return np.asarray(PINS[name]["columns"][column], dtype=float)


# The tests below guard against a pin going vacuous: each scenario really engages its
# feature. Most read the pinned numbers, to document what each scenario reaches. The live
# check is test_switching_the_feature_off_changes_the_result further down: it re-solves
# each scenario with its feature off, so a feature that stops changing the result fails
# there even while the pinned numbers still match.


def test_scalar_scenario_binds_both_caps():
    hybrid = _col("hybrid_scalar_caps", "P_hybrid_inverter")
    assert hybrid.max() == pytest.approx(OUTPUT_CAP)
    assert hybrid.min() == pytest.approx(-INPUT_CAP)
    assert (_col("hybrid_scalar_caps", "P_grid_pos") > 0).any()


@pytest.mark.parametrize(
    "name",
    [
        "hybrid_dc_ac_curve",
        "hybrid_ac_dc_curve",
        "hybrid_both_curves_two_batteries",
        "hybrid_both_curves_relaxed_retry",
    ],
)
def test_curve_changes_the_economics(name):
    assert abs(PINS[name]["objective"] - PINS["hybrid_scalar_caps"]["objective"]) > 1e-3


def test_relaxed_scenario_reaches_the_rescue():
    assert PINS["hybrid_both_curves_relaxed_retry"]["status"] == "Optimal (Relaxed)"


def test_curtailment_scenario_curtails_at_the_cap():
    curtailed = _col("hybrid_curtailment", "P_PV_curtailment")
    assert curtailed.max() > 1000
    assert _col("hybrid_curtailment", "P_hybrid_inverter")[curtailed.argmax()] == OUTPUT_CAP


def test_nodischarge_rule_removes_battery_export():
    # The E[k] <= D rule removes every step that discharges the battery while exporting,
    # which the unconstrained scalar scenario does.
    def exports_from_battery(name):
        return (_col(name, "P_batt") > 1e-6) & (_col(name, "P_grid_neg") < -1e-6)

    assert exports_from_battery("hybrid_scalar_caps").any()
    assert not exports_from_battery("hybrid_nodischarge_to_grid").any()


def test_acoupled_export_ceiling_binds():
    # Live solves, rule on and off: with the rule the export stays within the PV surplus
    # (the #936 ceiling) at every step. Without it the battery exports above the surplus
    # in the evening, so the ceiling is what removes that export.
    name = "acoupled_nodischarge_curtailment"
    ceiling = np.maximum(0, np.asarray(PV, dtype=float) - np.asarray(LOAD, dtype=float))
    _, _, on = solve(CASES[name], "dayahead")
    _, _, off = solve(_feature_off(name, optim={"set_nodischarge_to_grid": False}), "dayahead")
    assert (-on["P_grid_neg"].to_numpy() <= ceiling + POWER_TOL).all()
    export_off = -off["P_grid_neg"].to_numpy()
    assert ((export_off > ceiling + 100) & (off["P_batt"].to_numpy() > 100)).any()
    assert on["P_PV_curtailment"].max() > 0


def test_stress_segments_are_active():
    assert _col("hybrid_inverter_stress", "inv_stress_cost").sum() > 0.1


def test_nocharge_rule_removes_night_grid_charge():
    def night_grid_charge(name):
        pv = np.asarray(PV, dtype=float)
        return (pv == 0) & (_col(name, "P_batt") < -1e-6)

    assert night_grid_charge("hybrid_scalar_caps").any()
    assert not night_grid_charge("hybrid_nocharge_from_grid").any()


def _feature_off(name, optim=None, plant=None, drop=()):
    """CASES[name] with ``optim``/``plant`` overrides applied and ``drop`` keys removed."""
    case = CASES[name]
    merged_plant = {**case.get("plant", {}), **(plant or {})}
    return {
        **case,
        "optim": {**case.get("optim", {}), **(optim or {})},
        "plant": {k: v for k, v in merged_plant.items() if k not in drop},
    }


# Each scenario with the feature it pins switched off. The AC-coupled export ceiling has its
# own rule-on/rule-off test above, and the relaxed rescue is checked live through the
# pinned "Optimal (Relaxed)" status.
FEATURE_OFF = {
    "output_cap": ("hybrid_scalar_caps", {"plant": {"inverter_ac_output_max": 9000}}),
    "input_cap": ("hybrid_scalar_caps", {"plant": {"inverter_ac_input_max": 9000}}),
    "dc_ac_curve": ("hybrid_dc_ac_curve", {"drop": ("inverter_power_curve_dc_ac",)}),
    "ac_dc_curve": ("hybrid_ac_dc_curve", {"drop": ("inverter_power_curve_ac_dc",)}),
    "both_curves": ("hybrid_both_curves_two_batteries", {"drop": tuple(BOTH_CURVES)}),
    "curtailment": ("hybrid_curtailment", {"plant": {"compute_curtailment": False}}),
    "nodischarge": ("hybrid_nodischarge_to_grid", {"optim": {"set_nodischarge_to_grid": False}}),
    "stress": ("hybrid_inverter_stress", {"plant": {"inverter_stress_cost": 0.0}}),
    "nocharge": ("hybrid_nocharge_from_grid", {"optim": {"set_nocharge_from_grid": False}}),
}


@pytest.mark.parametrize("feature", FEATURE_OFF)
def test_switching_the_feature_off_changes_the_result(feature):
    # Both solves are live, so this fails if the model stops applying the feature, whatever
    # the pins say. Without curtailment the PV above the cap has nowhere to go, so that
    # counterfactual is infeasible rather than cheaper.
    name, off = FEATURE_OFF[feature]
    on_status, on_objective, _ = solve(CASES[name], "dayahead")
    off_status, off_objective, _ = solve(_feature_off(name, **off), "dayahead")
    assert on_status == "Optimal"
    if feature == "curtailment":
        assert off_status == "Infeasible"
    else:
        assert off_status == "Optimal"
        assert abs(off_objective - on_objective) > 1e-3
