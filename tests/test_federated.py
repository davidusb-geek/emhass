"""The opt-in coordinator backend (optimization_backend), served by the optional
home-energy-optimizer package (pip install emhass[federated]).

Tests that need the package skip without it; the default path and the
fallbacks are tested either way.
"""

import asyncio
import copy
import importlib.util
import pathlib
import sys
from unittest import mock

import numpy as np
import orjson
import pandas as pd
import pytest

from emhass import optimization
from emhass.optimization import Optimization
from emhass.utils import (
    build_config,
    build_params,
    build_secrets,
    get_injection_dict,
    get_logger,
    get_root,
    get_yaml_parse,
)

root = pathlib.Path(get_root(__file__, num_parent=2))
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
}
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"
logger, _ = get_logger(__name__, emhass_conf, save_to_file=False)

needs_package = pytest.mark.skipif(
    importlib.util.find_spec("home_energy_optimizer") is None,
    reason="needs the federated extra (home-energy-optimizer)",
)


def _confs():
    """EMHASS's default configuration, built as at startup.

    Returns:
        tuple: (retrieve_hass_conf, optim_conf, plant_conf) dicts.
    """

    async def build():
        """Build config, secrets and params from the defaults; returns the parsed confs."""
        config = await build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        _, secrets = await build_secrets(emhass_conf, logger, no_response=True)
        params = await build_params(emhass_conf, secrets, config, logger)
        return get_yaml_parse(orjson.dumps(params).decode("utf-8"), logger)

    return asyncio.run(build())


def _site():
    """A day at 30-minute steps: a 10 kWh battery, a 3 kW load for 4 h and a
    0.75 kW load for 2 h (both on/off), 5 kW of PV, a time-of-use tariff.

    Returns:
        tuple: (retrieve_hass_conf, optim_conf, plant_conf, data, pv, load, buy,
        sell), where data is the input DataFrame, pv and load are W per step,
        and buy and sell are currency/kWh per step.
    """
    rh, oc, pc = _confs()
    oc.update(set_use_battery=True, set_use_pv=True, operating_hours_of_each_deferrable_load=[4, 2])
    pc.update(
        battery_nominal_energy_capacity=10000,
        battery_charge_power_max=5000,
        battery_discharge_power_max=5000,
    )
    n = int(pd.Timedelta(days=1) / rh["optimization_time_step"])
    index = pd.date_range(
        "2026-10-01", periods=n, freq=rh["optimization_time_step"], tz=rh["time_zone"]
    )
    h = np.arange(n) * rh["optimization_time_step"].seconds / 3600
    buy = 0.20 + 0.10 * np.sin((h - 13) / 24 * 2 * np.pi) + 0.08 * (np.abs(h - 19) < 2)
    sell = np.full(n, 0.06)
    pv = np.clip(5000 * np.sin((h - 6) / 12 * np.pi), 0, None)
    load = 400 + 1200 * np.exp(-0.5 * ((h - 19.5) / 1.8) ** 2)
    data = pd.DataFrame({"unit_load_cost": buy, "unit_prod_price": sell}, index=index)
    return rh, oc, pc, data, pv, load, buy, sell


def _plan(rh, oc, pc, data, pv, load, buy, sell, **options):
    """Run perform_optimization on a copy of the site, the battery from 50% to 50%.

    Args:
        rh, oc, pc, data, pv, load, buy, sell: The site, as _site returns it.
        **options: optim_conf overrides (e.g. optimization_backend).

    Returns:
        tuple: (Optimization, the opt_res DataFrame).
    """
    oc = copy.deepcopy(oc)
    oc.update(options)
    opt = Optimization(
        rh,
        oc,
        copy.deepcopy(pc),
        "unit_load_cost",
        "unit_prod_price",
        "profit",
        emhass_conf,
        logger,
    )
    return opt, opt.perform_optimization(data, pv, load, buy, sell, soc_init=0.5, soc_final=0.5)


def _bill(res):
    """The plan's energy bill (currency; negative is a net profit)."""
    return -float(res["cost_profit"].sum())


def _balance_holds(res, loads):
    """Whether supply (PV + battery + grid) equals use (house + the first
    `loads` deferrable loads) in every step, within 1 mW."""
    supplied = res["P_PV"] + res["P_batt"] + res["P_grid"]
    used = res["P_Load"] + sum(res[f"P_deferrable{k}"] for k in range(loads))
    return np.allclose(supplied, used, atol=1e-3)


PROVENANCE = ["backend_requested", "backend_used", "backend_fallback_reason"]


def _fell_back(res, milp, reason: str) -> None:
    """`res` is exactly the default plan `milp`, marked as made by cvxpy
    instead of dantzig_wolfe, for `reason`."""
    pd.testing.assert_frame_equal(res.drop(columns=PROVENANCE), milp)
    assert (res["backend_requested"] == "dantzig_wolfe").all()
    assert (res["backend_used"] == "cvxpy").all()
    assert res["backend_fallback_reason"].str.contains(reason, regex=False).all()


def test_default_backend_is_cvxpy_and_never_dispatches():
    """By default the backend is cvxpy and the coordinator is never called,
    and the plan has exactly its usual columns: no provenance columns."""
    rh, oc, pc = _confs()
    assert oc["optimization_backend"] == "cvxpy"
    assert not oc.get("participants")
    with mock.patch.object(optimization, "_coordinated_plan") as coordinated:
        _, res = _plan(*_site())
    coordinated.assert_not_called()
    assert not set(PROVENANCE) & set(res.columns)


def test_missing_package_falls_back_to_cvxpy():
    """dantzig_wolfe without the optional package returns exactly the default
    plan, and says that cvxpy made it, and why."""
    site = _site()
    _, milp = _plan(*site)
    with mock.patch.dict(sys.modules, {"home_energy_optimizer.integrations.emhass": None}):
        opt, res = _plan(*site, optimization_backend="dantzig_wolfe")
    _fell_back(res, milp, "the optional extra is not installed")


def test_the_web_page_shows_which_backend_made_the_plan():
    """The web page's tables and plots take a plan with text columns: the
    backend that made it, and why, go in the summary table, not the plots."""
    with mock.patch.dict(sys.modules, {"home_energy_optimizer.integrations.emhass": None}):
        _, res = _plan(*_site(), optimization_backend="dantzig_wolfe")
    page = get_injection_dict(res.copy())
    assert "backend_used" in page["table2"] and "cvxpy" in page["table2"]
    assert "the optional extra is not installed" in page["table2"]
    assert "backend_used" not in page["table1"]


@needs_package
def test_dantzig_wolfe_matches_the_milp_on_a_battery_and_two_loads():
    """dantzig_wolfe returns every default column plus fed_*, a balanced plan
    that runs each load its hours and ends the battery at 50%, with a bill
    within a cent of the default MILP's."""
    site = _site()
    _, milp = _plan(*site)
    opt, res = _plan(*site, optimization_backend="dantzig_wolfe")
    assert opt.optim_status == "Optimal"
    assert set(milp.columns) <= set(res.columns)
    assert {"fed_meter_price", "fed_lower_bound", "fed_gap"} <= set(res.columns)
    assert (res["backend_used"] == "dantzig_wolfe").all()
    assert (res["backend_fallback_reason"] == "").all()
    assert _balance_holds(res, loads=2)
    for k, (watts, hours) in enumerate(((3000.0, 4), (750.0, 2))):
        assert res[f"P_deferrable{k}"].sum() * 0.5 == pytest.approx(watts * hours, rel=1e-6)
    assert res["SOC_opt"].iloc[-1] == pytest.approx(0.5, abs=1e-3)
    assert _bill(res) <= _bill(milp) + 0.01  # the same plan value, within a cent


@needs_package
def test_grouped_participants_and_a_battery_planned_by_the_package():
    """With both loads grouped as one EMHASS participant and the battery
    planned by home-energy-optimizer, the plan balances and runs each load
    its hours."""
    site = _site()
    opt, res = _plan(
        *site,
        optimization_backend="dantzig_wolfe",
        participants=[
            {"devices": ["battery"], "solver": "home_energy_optimizer"},
            {"devices": ["deferrable0", "deferrable1"], "solver": "emhass"},
        ],
    )
    assert opt.optim_status == "Optimal"
    assert _balance_holds(res, loads=2)
    for k, (watts, hours) in enumerate(((3000.0, 4), (750.0, 2))):
        assert res[f"P_deferrable{k}"].sum() * 0.5 == pytest.approx(watts * hours, rel=1e-6)
    assert np.isfinite(_bill(res))


@needs_package
def test_unsupported_option_falls_back_to_cvxpy():
    """An option that cannot be split per device (set_nocharge_from_grid)
    returns exactly the default plan and logs the option's name."""
    rh, oc, pc, *rest = _site()
    oc["set_nocharge_from_grid"] = True  # ties the battery to PV: not split per device
    site = (rh, oc, pc, *rest)
    _, milp = _plan(*site)
    with mock.patch.object(logger, "warning") as warning:
        _, res = _plan(*site, optimization_backend="dantzig_wolfe")
    _fell_back(res, milp, "set_nocharge_from_grid")
    assert any("set_nocharge_from_grid" in str(c) for c in warning.call_args_list)


def _hep_at_least(*version: int) -> bool:
    """Whether home-energy-optimizer is installed, at `version` or later."""
    try:
        from home_energy_optimizer import __version__
    except ImportError:
        return False
    return tuple(int(x) for x in __version__.split(".")[:3]) >= version


def _hybrid_ready() -> bool:
    """home-energy-optimizer plans a hybrid inverter from 0.2.5."""
    return _hep_at_least(0, 2, 5)


@pytest.mark.skipif(not _hybrid_ready(), reason="needs home-energy-optimizer >= 0.2.5")
@pytest.mark.parametrize("rating_w", [5000, 3000, 2000])
def test_a_hybrid_inverter_plans_as_the_milp_does(rating_w):
    """The PV and the battery on a hybrid inverter's DC bus, 8 kW of PV into
    a smaller inverter: the coordinator holds the inverter as a sub-meter,
    and its plan matches the default MILP's - the bill, the inverter's AC
    power within its rating, and the first step an MPC loop applies
    (P_grid, P_batt, P_hybrid_inverter, SOC_opt)."""
    rh, oc, pc, data, pv, load, buy, sell = _site()
    oc = dict(
        oc, set_nodischarge_to_grid=False
    )  # with a hybrid inverter that ties the battery to the meter
    pc = dict(
        pc,
        inverter_is_hybrid=True,
        inverter_ac_output_max=rating_w,
        inverter_ac_input_max=rating_w,
        inverter_efficiency_dc_ac=0.97,
        inverter_efficiency_ac_dc=0.97,
        compute_curtailment=True,
    )
    site = (rh, oc, pc, data, pv * 1.6, load, buy, sell)
    _, milp = _plan(*site)
    opt, res = _plan(
        *site,
        optimization_backend="dantzig_wolfe",
        participants=[
            {"devices": ["battery"], "solver": "emhass"},
            {"devices": ["deferrable0", "deferrable1"], "solver": "emhass"},
        ],
    )
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    assert _bill(res) == pytest.approx(_bill(milp), abs=0.01)
    assert res["P_hybrid_inverter"].max() <= rating_w + 1e-3
    assert res["P_hybrid_inverter"].min() >= -rating_w - 1e-3
    for col in ("P_grid", "P_batt", "P_hybrid_inverter", "SOC_opt"):
        assert res[col].iloc[0] == pytest.approx(milp[col].iloc[0], abs=1.0), col
    # the AC bus balances: the inverter and the grid supply the load and the loads
    supplied = res["P_hybrid_inverter"] + res["P_grid"]
    used = res["P_Load"] + res["P_deferrable0"] + res["P_deferrable1"]
    assert np.allclose(supplied, used, atol=1e-3)


LOAD_GROUP = [{"names": ["deferrable0", "deferrable1"], "max_power": 3000}]


@pytest.mark.skipif(not _hep_at_least(0, 2, 6), reason="needs home-energy-optimizer >= 0.2.6")
def test_a_load_group_inside_one_participant_plans_as_the_milp_does():
    """deferrable_load_groups with both loads in one participant group: that
    participant's own EMHASS model holds the 3 kW budget, and the plan is the
    default MILP's."""
    rh, oc, pc, data, pv, load, buy, sell = _site()
    site = (rh, dict(oc, deferrable_load_groups=LOAD_GROUP), pc, data, pv, load, buy, sell)
    _, milp = _plan(*site)
    opt, res = _plan(
        *site,
        optimization_backend="dantzig_wolfe",
        participants=[{"devices": ["deferrable0", "deferrable1"], "solver": "emhass"}],
    )
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    assert (res["P_deferrable0"] + res["P_deferrable1"]).max() <= 3000 + 1e-3
    assert _bill(res) == pytest.approx(_bill(milp), abs=0.01)


@pytest.mark.skipif(not _hep_at_least(0, 2, 6), reason="needs home-energy-optimizer >= 0.2.6")
def test_a_load_group_across_participants_is_held_and_priced():
    """The same budget with each load its own participant: the coordinator
    holds it as a group limit, the loads still run their hours within 3 kW,
    the plan is within a few cents of the MILP's, and the group's own price
    is a column."""
    rh, oc, pc, data, pv, load, buy, sell = _site()
    site = (rh, dict(oc, deferrable_load_groups=LOAD_GROUP), pc, data, pv, load, buy, sell)
    _, milp = _plan(*site)
    opt, res = _plan(*site, optimization_backend="dantzig_wolfe")
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    assert (res["P_deferrable0"] + res["P_deferrable1"]).max() <= 3000 + 1e-3
    for k, (watts, hours) in enumerate(((3000.0, 4), (750.0, 2))):
        assert res[f"P_deferrable{k}"].sum() * 0.5 == pytest.approx(watts * hours, rel=1e-6)
    assert _bill(res) <= _bill(milp) + 0.05
    assert "fed_limit_price_deferrable0+deferrable1" in res.columns


TOPOLOGY_READY = pytest.mark.skipif(
    not _hep_at_least(0, 2, 7), reason="needs home-energy-optimizer >= 0.2.7"
)


@TOPOLOGY_READY
def test_a_topology_constraint_holds_and_is_priced():
    """A constraint over the two loads (each its own participant): their total
    stays within 3.5 kW, its premium is fed_limit_price_<name>, and a node
    that splits a participant group falls back, saying so."""
    site = _site()
    topology = {
        "constraints": [
            {"name": "loads", "devices": ["deferrable0", "deferrable1"], "max_import": 3500}
        ]
    }
    opt, res = _plan(*site, optimization_backend="dantzig_wolfe", electrical_topology=topology)
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    assert (res["P_deferrable0"] + res["P_deferrable1"]).max() <= 3500 + 1e-3
    assert "fed_limit_price_loads" in res.columns
    with mock.patch.object(logger, "warning") as warning:
        _, res = _plan(
            *site,
            optimization_backend="dantzig_wolfe",
            participants=[{"devices": ["deferrable0", "deferrable1"], "solver": "emhass"}],
            electrical_topology={
                "nodes": [{"id": "half", "max_import": 1000}],
                "devices": {"deferrable0": "half"},
            },
        )
    assert (res["backend_used"] == "cvxpy").all()
    assert res["backend_fallback_reason"].str.contains("splits the participant").all()
    # ...and says the default solver does not hold the topology's limits
    assert any("does not hold electrical_topology" in str(c) for c in warning.call_args_list)


FOUR_DER_TOPOLOGY = {
    "nodes": [
        {
            "id": "inverter",
            "type": "hybrid_inverter",
            "max_import": 4000,
            "max_export": 4000,
            "efficiency_from_parent": 0.97,
            "efficiency_to_parent": 0.97,
        },
        {"id": "garage", "type": "panel", "max_import": 7400, "max_export": 0},
        {"id": "heat", "type": "breaker", "parent": "garage", "max_import": 3500},
    ],
    "devices": {
        "pv": "inverter",
        "battery": "inverter",
        "water_heater": "heat",
        "hvac": "heat",
        "deferrable0": "garage",
        "deferrable1": "garage",
    },
}


@TOPOLOGY_READY
def test_four_ders_on_an_electrical_topology():
    """The README's four-DER house, from tools/emhass-coordination/
    config_four_der.json: the battery (EMHASS's model) and the PV behind a
    4 kW hybrid inverter; a garage panel (7.4 kW, no backfeed) with the two
    loads (one EMHASS participant, a 3 kW budget its own model holds) and,
    under it, a 3.5 kW breaker with the tank and the heat pump
    (home-energy-optimizer's models). Every limit holds, every node is priced
    and reported, and every player has a share."""
    rh, oc, pc, data, pv, load, buy, sell = _site()
    n = len(data)
    h = np.arange(n) * rh["optimization_time_step"].seconds / 3600
    data = data.copy()
    data["outdoor_temperature_forecast"] = 10 + 5 * np.sin((h - 9) / 24 * 2 * np.pi)
    oc = dict(
        oc,
        set_nodischarge_to_grid=False,
        deferrable_load_groups=[{"names": ["deferrable0", "deferrable1"], "max_power": 3000}],
    )
    pc = dict(pc, compute_curtailment=True)
    site = (rh, oc, pc, data, pv * 1.6, load, buy, sell)
    opt, res = _plan(
        *site,
        optimization_backend="dantzig_wolfe",
        participants=[
            {"devices": ["battery"], "solver": "emhass"},
            {"devices": ["deferrable0", "deferrable1"], "solver": "emhass"},
            {
                "devices": ["water_heater"],
                "solver": "home_energy_optimizer",
                "config": {"power_kw": 3.0, "liters": 180.0, "t_comfort": 55.0, "n_duty_levels": 2},
            },
            {
                "devices": ["hvac"],
                "solver": "home_energy_optimizer",
                "config": {
                    "power_kw": 1.5,
                    "cop": 3.5,
                    "t_comfort_low": 21.0,
                    "t_comfort_high": 25.0,
                },
            },
        ],
        electrical_topology=FOUR_DER_TOPOLOGY,
    )
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    assert res["P_hybrid_inverter"].abs().max() <= 4000 + 1e-3
    heat = res["P_water_heater"] + res["P_hvac"]
    garage = heat + res["P_deferrable0"] + res["P_deferrable1"]
    assert heat.max() <= 3500 + 1e-3
    assert garage.max() <= 7400 + 1e-3 and garage.min() >= -1e-3  # no backfeed
    assert (res["P_deferrable0"] + res["P_deferrable1"]).max() <= 3000 + 1e-3
    for node in ("inverter", "garage", "heat"):
        assert f"fed_local_price_{node}" in res.columns and f"fed_node_power_{node}" in res.columns
    assert np.allclose(-res["fed_node_power_garage"], garage, atol=1e-3)  # + = up the tree
    assert {
        f"fed_share_{p}"
        for p in ("solar", "battery", "water_heater", "hvac", "deferrable0+deferrable1")
    } <= set(res.columns)
    # the main meter's bus balances: the inverter and the grid supply the load and the garage
    assert np.allclose(res["P_hybrid_inverter"] + res["P_grid"], res["P_Load"] + garage, atol=1e-3)


def test_a_hybrid_inverter_tied_to_the_meter_falls_back():
    """set_nodischarge_to_grid with a hybrid inverter (EMHASS: no discharge
    while exporting) is not split per device: the default MILP plans, and the
    plan says why."""
    rh, oc, pc, data, pv, load, buy, sell = _site()
    pc = dict(pc, inverter_is_hybrid=True, inverter_ac_output_max=5000)
    site = (rh, dict(oc, set_nodischarge_to_grid=True), pc, data, pv, load, buy, sell)
    _, res = _plan(*site, optimization_backend="dantzig_wolfe")
    assert (res["backend_used"] == "cvxpy").all()
    if _hybrid_ready():
        assert res["backend_fallback_reason"].str.contains("set_nodischarge_to_grid").all()


@needs_package
def test_a_participant_answers_the_same_query_the_same_way():
    """A heat pump with thermal inertia answers the same prices with the same
    plan, whatever it was asked in between: each query starts from the same
    state, not from the heat input of the previous solve."""
    from home_energy_optimizer.integrations.emhass import EmhassParticipant
    from home_energy_optimizer.interface import Query

    rh, oc, pc, data, pv, load, buy, sell = _site()
    n = len(data)
    h = np.arange(n) * rh["optimization_time_step"].seconds / 3600
    data["outdoor_temperature_forecast"] = 8 + 5 * np.sin((h - 9) / 24 * 2 * np.pi)
    heat_pump = {
        "supply_temperature": 35.0,
        "volume": 20.0,
        "start_temperature": 20.2,  # just above the minimum: it must heat at once
        "min_temperatures": [20.0] * n,
        "max_temperatures": [26.0] * n,
        "carnot_efficiency": 0.45,
        "u_value": 0.35,
        "envelope_area": 280.0,
        "ventilation_rate": 0.4,
        "heated_volume": 320.0,
        "thermal_inertia_time_constant": 2.0,
    }
    oc.update(
        nominal_power_of_deferrable_loads=[3000.0, 3000.0],
        operating_hours_of_each_deferrable_load=[4, 0],
        treat_deferrable_load_as_semi_cont=[True, False],
        def_load_config=[{}, {"thermal_battery": heat_pump}],
    )
    opt = Optimization(
        rh, oc, pc, "unit_load_cost", "unit_prod_price", "profit", emhass_conf, logger
    )
    hp = EmhassParticipant(opt, "deferrable1", False, [1], data, None, None, {}, buy)
    first = hp.respond(Query("price_response", buy, buy))
    cheap_first = np.where(np.arange(n) < 8, 0.01, 0.5)  # heats hard at the start
    hp.respond(Query("price_response", cheap_first, cheap_first))
    again = hp.respond(Query("price_response", buy, buy))
    np.testing.assert_allclose(again.plan_kw, first.plan_kw, atol=1e-6)
