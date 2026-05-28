#!/usr/bin/env python

import logging
import pathlib

import pandas as pd

from emhass.optimization import Optimization


TEST_ROOT = pathlib.Path(__file__).resolve().parents[1]


def build_optimization(optim_overrides=None, plant_overrides=None) -> Optimization:
    logger = logging.getLogger("soc_recovery_test")
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
        "battery_dynamic_max": 0.9,
        "battery_dynamic_min": -0.9,
        "weight_battery_discharge": 0.0,
        "weight_battery_charge": 0.0,
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
        opt_time_delta=5,
    )


def test_low_soc_recovery_waits_for_pv_when_grid_is_expensive():
    opt = build_optimization()
    index = pd.date_range("2026-01-01", periods=10, freq="30min", tz="Europe/Tallinn")

    p_pv = pd.Series([0, 0, 0, 0, 0, 6000, 6000, 0, 0, 0], index=index)
    p_load = pd.Series([0] * 10, index=index)
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = [0.40] * 10
    df_input["unit_prod_price"] = [0.0] * 10

    opt_res = opt.perform_naive_mpc_optim(
        df_input,
        p_pv,
        p_load,
        10,
        soc_init=0.1,
        soc_final=0.4,
        def_total_hours=[],
        def_total_timestep=[],
        def_start_timestep=[],
        def_end_timestep=[],
    )

    assert opt.optim_status == "Optimal"
    assert (opt_res["P_batt"].iloc[:5] > -100).all(), opt_res["P_batt"].tolist()
    assert opt_res["P_batt"].iloc[5:7].min() < -1000
    assert abs(opt_res["SOC_opt"].iloc[-1] - 0.4) < 1e-3


def test_high_soc_recovery_waits_for_load_instead_of_immediate_export():
    opt = build_optimization()
    index = pd.date_range("2026-01-01", periods=10, freq="30min", tz="Europe/Tallinn")

    p_pv = pd.Series([0] * 10, index=index)
    p_load = pd.Series([0, 0, 0, 0, 5000, 5000, 0, 0, 0, 0], index=index)
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = [0.40] * 10
    df_input["unit_prod_price"] = [0.0] * 10

    opt_res = opt.perform_naive_mpc_optim(
        df_input,
        p_pv,
        p_load,
        10,
        soc_init=0.9,
        soc_final=0.6,
        def_total_hours=[],
        def_total_timestep=[],
        def_start_timestep=[],
        def_end_timestep=[],
    )

    assert opt.optim_status == "Optimal"
    assert (opt_res["P_batt"].iloc[:4] < 100).all(), opt_res["P_batt"].tolist()
    assert opt_res["P_batt"].iloc[4:6].max() > 1000
    assert abs(opt_res["SOC_opt"].iloc[-1] - 0.6) < 1e-3


def test_low_soc_stays_above_min_once_back_inside_band():
    opt = build_optimization()
    index = pd.date_range("2026-01-01", periods=14, freq="30min", tz="Europe/Tallinn")

    p_pv = pd.Series([0, 0, 0, 0, 0, 6000, 6000, 0, 0, 0, 0, 0, 6000, 6000], index=index)
    p_load = pd.Series([0, 0, 0, 0, 0, 0, 0, 5000, 5000, 5000, 0, 0, 0, 0], index=index)
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = [
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        5.0,
        5.0,
        5.0,
        0.2,
        0.2,
        0.2,
        0.2,
    ]
    df_input["unit_prod_price"] = [0.0] * 14

    opt_res = opt.perform_naive_mpc_optim(
        df_input,
        p_pv,
        p_load,
        14,
        soc_init=0.1,
        soc_final=0.3,
        def_total_hours=[],
        def_total_timestep=[],
        def_start_timestep=[],
        def_end_timestep=[],
    )

    first_inside = opt_res.index[opt_res["SOC_opt"] > 0.30001][0]
    recovery_prefix = opt_res.loc[:first_inside, "SOC_opt"]
    assert opt.optim_status == "Optimal"
    assert (recovery_prefix.diff().fillna(0) >= -1e-4).all(), recovery_prefix.tolist()
    assert opt_res.loc[first_inside:, "SOC_opt"].min() >= 0.3 - 1e-4


def test_high_soc_stays_below_max_once_back_inside_band():
    opt = build_optimization()
    index = pd.date_range("2026-01-01", periods=14, freq="30min", tz="Europe/Tallinn")

    p_pv = pd.Series([0, 0, 0, 0, 0, 6000, 6000, 0, 0, 0, 0, 0, 0, 0], index=index)
    p_load = pd.Series([2000, 0, 0, 0, 0, 0, 0, 5000, 5000, 0, 0, 0, 0, 0], index=index)
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = [0.2] * 14
    df_input["unit_prod_price"] = [0.0] * 14

    opt_res = opt.perform_naive_mpc_optim(
        df_input,
        p_pv,
        p_load,
        14,
        soc_init=0.9,
        soc_final=0.6,
        def_total_hours=[],
        def_total_timestep=[],
        def_start_timestep=[],
        def_end_timestep=[],
    )

    first_inside = opt_res.index[opt_res["SOC_opt"] < 0.79999][0]
    recovery_prefix = opt_res.loc[:first_inside, "SOC_opt"]
    assert opt.optim_status == "Optimal"
    assert (recovery_prefix.diff().fillna(0) <= 1e-4).all(), recovery_prefix.tolist()
    assert opt_res.loc[first_inside:, "SOC_opt"].max() <= 0.8 + 1e-4


def test_low_soc_recovery_with_non_ideal_efficiency_remains_monotonic():
    opt = build_optimization(
        plant_overrides={
            "battery_discharge_efficiency": 0.92,
            "battery_charge_efficiency": 0.88,
            "battery_charge_power_max": 4000,
            "battery_discharge_power_max": 4000,
        }
    )
    index = pd.date_range("2026-01-01", periods=16, freq="30min", tz="Europe/Tallinn")

    p_pv = pd.Series([0, 0, 0, 0, 0, 0, 4500, 4500, 4500, 4500, 0, 0, 0, 0, 0, 0], index=index)
    p_load = pd.Series([0] * 16, index=index)
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = [0.35] * 16
    df_input["unit_prod_price"] = [0.0] * 16

    opt_res = opt.perform_naive_mpc_optim(
        df_input,
        p_pv,
        p_load,
        16,
        soc_init=0.1,
        soc_final=0.4,
        def_total_hours=[],
        def_total_timestep=[],
        def_start_timestep=[],
        def_end_timestep=[],
    )

    first_inside = opt_res.index[opt_res["SOC_opt"] > 0.30001][0]
    recovery_prefix = opt_res.loc[:first_inside, "SOC_opt"]
    assert opt.optim_status == "Optimal"
    assert (recovery_prefix.diff().fillna(0) >= -1e-4).all(), recovery_prefix.tolist()
    assert opt_res.loc[first_inside:, "SOC_opt"].min() >= 0.3 - 1e-4
    assert abs(opt_res["SOC_opt"].iloc[-1] - 0.4) < 1e-3
