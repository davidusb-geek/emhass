#!/usr/bin/env python
"""Tests for battery_charge_power_derating (issue #807).

Two layers, both covered here: the config-layer normalisation in
utils.check_batt_charge_derating (shape disambiguation, per-battery broadcast,
what happens to an unusable value) and the constraint itself in
optimization.py (does the ceiling bind, at which SOC, and is it inert without
the table).

The Optimization builder mirrors the one in test_multi_battery_optimization.py
(which in turn mirrors test_soc_recovery_prototype.py): synthetic and
self-contained, so this file solves quickly and depends on no data files.
"""

import asyncio
import json
import logging
import pathlib

import numpy as np
import orjson
import pandas as pd
import pytest

from emhass import utils
from emhass.optimization import Optimization

TEST_ROOT = pathlib.Path(__file__).resolve().parents[1]
VALID_OPTIMAL_STATUSES = ["Optimal", "Optimal (Relaxed)"]

EMHASS_CONF = {
    "data_path": TEST_ROOT / "data/",
    "root_path": TEST_ROOT / "src/emhass/",
    "defaults_path": TEST_ROOT / "src/emhass/data/config_defaults.json",
    "associations_path": TEST_ROOT / "src/emhass/data/associations.csv",
}

DERATING_PARAM = "battery_charge_power_derating"
DERATING = [[0.5, 0.84], [0.7, 0.42], [0.9, 0.23]]

logger = logging.getLogger("charge_derating_test")


# --------------------------------------------------------------------------- #
# Config layer: utils.check_batt_charge_derating
# --------------------------------------------------------------------------- #

def _errors(caplog) -> list[str]:
    return [rec.message for rec in caplog.records if rec.levelname == "ERROR"]


@pytest.mark.parametrize("value", [None, [], "absent"])
def test_absent_null_or_empty_is_left_alone(value, caplog):
    """All three ways of saying "no table" are accepted and change nothing."""
    conf = {} if value == "absent" else {DERATING_PARAM: value}
    with caplog.at_level(logging.ERROR):
        utils.check_batt_charge_derating(2, conf, DERATING_PARAM, logger)
    assert conf == ({} if value == "absent" else {DERATING_PARAM: value})
    assert _errors(caplog) == []


def test_shared_table_at_n1_is_noop(caplog):
    """One battery: the table is used exactly as written, no nesting."""
    conf = {DERATING_PARAM: [row[:] for row in DERATING]}
    with caplog.at_level(logging.ERROR):
        utils.check_batt_charge_derating(1, conf, DERATING_PARAM, logger)
    assert conf[DERATING_PARAM] == DERATING
    assert _errors(caplog) == []


def test_shared_table_broadcasts_to_every_battery(caplog):
    """A list of pairs is one shared table, whatever the battery count."""
    conf = {DERATING_PARAM: [row[:] for row in DERATING]}
    with caplog.at_level(logging.ERROR):
        utils.check_batt_charge_derating(3, conf, DERATING_PARAM, logger)
    assert conf[DERATING_PARAM] == [DERATING, DERATING, DERATING]
    assert _errors(caplog) == []


def test_shared_table_entries_are_independent_copies():
    """Broadcasting must not alias one table object across batteries."""
    conf = {DERATING_PARAM: [row[:] for row in DERATING]}
    utils.check_batt_charge_derating(2, conf, DERATING_PARAM, logger)
    tables = conf[DERATING_PARAM]
    assert tables[0] is not tables[1]
    tables[0][0][1] = 0.11
    assert tables[1][0][1] == 0.84, "editing one battery's table leaked into its sibling"


def test_per_battery_tables_pass_through(caplog):
    """A list of tables is already per-battery and is left as-is."""
    other = [[0.5, 0.5], [0.9, 0.2]]
    conf = {DERATING_PARAM: [DERATING, other]}
    with caplog.at_level(logging.ERROR):
        utils.check_batt_charge_derating(2, conf, DERATING_PARAM, logger)
    assert conf[DERATING_PARAM] == [DERATING, other]
    assert _errors(caplog) == []


def test_shared_two_row_table_is_not_split_across_two_batteries(caplog):
    """Two rows and two batteries: one shared table, not one row per battery."""
    shared = [[0.5, 0.84], [0.9, 0.23]]
    conf = {DERATING_PARAM: [row[:] for row in shared]}
    with caplog.at_level(logging.ERROR):
        utils.check_batt_charge_derating(2, conf, DERATING_PARAM, logger)
    assert conf[DERATING_PARAM] == [shared, shared]
    assert _errors(caplog) == []


def test_wrong_table_count_is_dropped_not_raised(caplog):
    """A per-battery list of the wrong length is reported and ignored."""
    conf = {DERATING_PARAM: [DERATING, DERATING]}
    with caplog.at_level(logging.ERROR):
        utils.check_batt_charge_derating(3, conf, DERATING_PARAM, logger)
    assert conf[DERATING_PARAM] == []
    errors = _errors(caplog)
    assert len(errors) == 1
    assert DERATING_PARAM in errors[0]
    assert "3" in errors[0]


@pytest.mark.parametrize("bad", [0.84, "0.84", [0.5, 1.0], [[]], [None]])
def test_unusable_value_is_dropped_not_raised(bad, caplog):
    """A malformed value costs the refinement, never the optimization."""
    conf = {DERATING_PARAM: bad}
    with caplog.at_level(logging.ERROR):
        utils.check_batt_charge_derating(1, conf, DERATING_PARAM, logger)
    assert conf[DERATING_PARAM] == []
    errors = _errors(caplog)
    assert len(errors) == 1
    assert DERATING_PARAM in errors[0]


@pytest.mark.parametrize(
    "table,expected",
    [
        ([[0.9, 0.23], [0.5, 0.84]], "ascend"),  # thresholds out of order
        ([[0.5, 0.42], [0.9, 0.84]], "cannot rise"),  # limit rises as it fills
        ([[0.5, 84], [0.9, 23]], "between 0 and 1"),  # percent instead of /100
        ([[0.5, 0.84, 0.1]], "expected a [soc_threshold, power_max] pair"),
        ([[0.5, "0.84"]], "expected a number"),
    ],
)
def test_content_faults_are_named_and_dropped(table, expected, caplog):
    """An unusable table is dropped and the message names the offending row."""
    conf = {DERATING_PARAM: table}
    with caplog.at_level(logging.ERROR):
        utils.check_batt_charge_derating(1, conf, DERATING_PARAM, logger)
    assert conf[DERATING_PARAM] == []
    errors = _errors(caplog)
    assert len(errors) == 1
    assert DERATING_PARAM in errors[0]
    assert "row 1" in errors[0] or "row 2" in errors[0], errors[0]
    assert expected in errors[0], errors[0]


def test_content_faults_are_caught_per_battery_table(caplog):
    """The check runs on every table, not only on a shared one."""
    conf = {DERATING_PARAM: [DERATING, [[0.9, 0.23], [0.5, 0.84]]]}
    with caplog.at_level(logging.ERROR):
        utils.check_batt_charge_derating(2, conf, DERATING_PARAM, logger)
    assert conf[DERATING_PARAM] == []
    assert "ascend" in _errors(caplog)[0]


def _build_params(overrides: dict) -> dict:
    config = json.loads(EMHASS_CONF["defaults_path"].read_text(encoding="utf-8"))
    config.update(overrides)

    async def _build():
        _, secrets = await utils.build_secrets(EMHASS_CONF, logger, no_response=True)
        params = await utils.build_params(EMHASS_CONF, secrets, config, logger)
        assert params is not False, "build_params failed (see logged error)"
        return params

    return asyncio.run(_build())


def _treat_runtime(runtimeparams: dict, base_params: dict) -> dict:
    """Run a runtime payload through the real entry point and return plant_conf."""
    params_json = orjson.dumps(base_params).decode("utf-8")
    rh_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)

    async def _treat():
        return await utils.treat_runtimeparams(
            orjson.dumps(runtimeparams).decode("utf-8"),
            params_json,
            rh_conf,
            optim_conf,
            plant_conf,
            "dayahead-optim",
            logger,
            EMHASS_CONF,
        )

    _, _, _, plant_conf = asyncio.run(_treat())
    return plant_conf


def test_runtime_table_overrides_the_configured_one():
    """A runtime table replaces the configured one.

    Rides the generic associations.csv route, so this holds only while that row
    exists and the normaliser runs on the runtime path.
    """
    base = _build_params({DERATING_PARAM: DERATING})
    runtime = [[0.6, 0.5], [0.9, 0.2]]
    plant_conf = _treat_runtime({DERATING_PARAM: runtime}, base)
    assert plant_conf[DERATING_PARAM] == runtime


def test_runtime_table_broadcasts_per_battery():
    """A shared runtime table reaches every battery, same as a configured one."""
    base = _build_params({"number_of_batteries": 2, DERATING_PARAM: DERATING})
    runtime = [[0.6, 0.5], [0.9, 0.2]]
    plant_conf = _treat_runtime({DERATING_PARAM: runtime}, base)
    assert plant_conf[DERATING_PARAM] == [runtime, runtime]


# --------------------------------------------------------------------------- #
# Optimization layer: does the ceiling bind
# --------------------------------------------------------------------------- #

CHARGE_MAX = 5000
CAP = 10000
SOLVER_DERATING = [[0.5, 0.4], [0.9, 0.2]]


def build_optimization(plant_overrides=None) -> Optimization:
    """Self-contained Optimization builder, mirroring the one in
    test_multi_battery_optimization.py. Single battery, no deferrable loads."""
    build_logger = logging.getLogger("charge_derating_build")
    build_logger.handlers = []
    build_logger.addHandler(logging.NullHandler())

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
    plant_conf = {
        "inverter_is_hybrid": False,
        "compute_curtailment": False,
        "maximum_power_from_grid": 50000,
        "maximum_power_to_grid": 50000,
        "battery_discharge_power_max": CHARGE_MAX,
        "battery_charge_power_max": CHARGE_MAX,
        "battery_minimum_state_of_charge": 0.05,
        "battery_maximum_state_of_charge": 1.0,
        "battery_target_state_of_charge": 1.0,
        "battery_nominal_energy_capacity": CAP,
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
        build_logger,
        opt_time_delta=4,
    )


def _scenario():
    """Cheap power, no PV, small load: filling the battery is the whole job."""
    n = 8
    index = pd.date_range("2026-03-04", periods=n, freq="30min", tz="Europe/Tallinn")
    p_pv = pd.Series([0] * n, index=index)
    p_load = pd.Series([200] * n, index=index)
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = [0.05] * n
    df_input["unit_prod_price"] = [0.01] * n
    return df_input, p_pv, p_load


def _solve(with_derating: bool):
    overrides = {DERATING_PARAM: SOLVER_DERATING} if with_derating else {}
    opt = build_optimization(plant_overrides=overrides)
    df_input, p_pv, p_load = _scenario()
    opt_res = opt.perform_dayahead_forecast_optim(
        df_input, p_pv, p_load, soc_init=0.1, soc_final=1.0
    )
    assert opt.optim_status in VALID_OPTIMAL_STATUSES
    return opt_res


def _soc_at_step_start(opt_res, soc_init):
    """SOC_opt is the state at the END of a step; the ceiling follows the state
    the step STARTS at."""
    return np.concatenate(([soc_init], opt_res["SOC_opt"].to_numpy()[:-1]))


def test_charge_power_respects_the_limit_for_the_soc_it_starts_at():
    opt_res = _solve(with_derating=True)
    soc_start = _soc_at_step_start(opt_res, 0.1)
    charge = -opt_res["P_batt"].to_numpy()  # charging is negative in P_batt
    tol = 1.0  # W

    checked_high = 0
    for soc, power in zip(soc_start, charge, strict=True):
        if power <= tol:
            continue  # not charging in this step
        if soc >= 0.9:
            assert power <= 0.2 * CHARGE_MAX + tol
            checked_high += 1
        elif soc >= 0.5:
            assert power <= 0.4 * CHARGE_MAX + tol
        else:
            assert power <= CHARGE_MAX + tol
    assert checked_high > 0, "scenario never charged above 90% SOC, so it proves nothing"


def test_derating_spreads_the_fill_over_more_timesteps():
    flat = _solve(with_derating=False)
    tapered = _solve(with_derating=True)
    flat_steps = int((-flat["P_batt"] > 1.0).sum())
    tapered_steps = int((-tapered["P_batt"] > 1.0).sum())
    assert tapered_steps > flat_steps
    assert -tapered["P_batt"].min() <= -flat["P_batt"].min() + 1.0


def test_empty_table_leaves_the_plan_untouched():
    """The feature is inert unless the table is configured."""
    without_key = _solve(with_derating=False)
    opt = build_optimization(plant_overrides={DERATING_PARAM: []})
    df_input, p_pv, p_load = _scenario()
    empty_table = opt.perform_dayahead_forecast_optim(
        df_input, p_pv, p_load, soc_init=0.1, soc_final=1.0
    )
    np.testing.assert_allclose(
        empty_table["P_batt"].to_numpy(), without_key["P_batt"].to_numpy(), atol=1e-6
    )
