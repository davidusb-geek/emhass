"""The opt-in coordinated backend (optimization_backend: dantzig_wolfe),
emhass.coordinator: the default path never reaches it, and what it cannot
plan falls back to the default solver.
"""

import asyncio
import copy
import pathlib
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
    assert not oc.get("site")
    with mock.patch.object(optimization, "_coordinated_plan") as coordinated:
        _, res = _plan(*_site())
    coordinated.assert_not_called()
    assert not set(PROVENANCE) & set(res.columns)


def _raises(*_args, **_kwargs):
    """A coordinator that fails."""
    raise RuntimeError("coordinator exploded")


def test_a_failing_coordinator_falls_back_to_cvxpy():
    """dantzig_wolfe whose coordinator raises returns exactly the default
    plan, and says that cvxpy made it, and why."""
    site = _site()
    _, milp = _plan(*site)
    with mock.patch("emhass.coordinator.backend.optimize", _raises):
        opt, res = _plan(*site, optimization_backend="dantzig_wolfe")
    _fell_back(res, milp, "coordinator exploded")


@pytest.mark.parametrize(
    "name, value",
    [
        ("battery_availability", 0.5),
        ("battery_discharge_energy_max", 2000.0),
        ("def_availability", 0.5),
    ],
)
def test_inputs_the_coordinator_does_not_model_mean_the_default_solver(name, value):
    """Runtime inputs the coordinator does not model yet (battery and load
    availability, a discharge energy cap): cvxpy plans, and says which input."""
    rh, oc, pc, data, pv, load, buy, sell = _site()
    oc = {**copy.deepcopy(oc), "optimization_backend": "dantzig_wolfe"}
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
    res = opt.perform_optimization(
        data, pv, load, buy, sell, soc_init=0.5, soc_final=0.5, **{name: value}
    )
    assert (res["backend_used"] == "cvxpy").all()
    assert res["backend_fallback_reason"].str.contains(name, regex=False).all()


def test_the_web_page_shows_which_backend_made_the_plan():
    """The web page's tables and plots take a plan with text columns: the
    backend that made it, and why, go in the summary table, not the plots."""
    with mock.patch("emhass.coordinator.backend.optimize", _raises):
        _, res = _plan(*_site(), optimization_backend="dantzig_wolfe")
    page = get_injection_dict(res.copy())
    assert "backend_used" in page["table2"] and "cvxpy" in page["table2"]
    assert "coordinator exploded" in page["table2"]
    assert "backend_used" not in page["table1"]


LOADS = [
    {"id": "deferrable0", "parent": "grid", "group": "loads"},
    {"id": "deferrable1", "parent": "grid", "group": "loads"},
]


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


def test_grouped_loads_and_the_battery_by_its_indexed_name():
    """With both loads in one group, planned together by EMHASS's model, and
    the battery named battery0 (#1032's indexed name; with one battery its
    columns stay P_batt and fed_share_battery), the plan balances and runs
    each load its hours; the group has one share of the saving."""
    site = _site()
    opt, res = _plan(
        *site,
        optimization_backend="dantzig_wolfe",
        site=[{"id": "battery0", "parent": "grid"}, *LOADS],
    )
    assert "fed_share_loads" in res.columns
    assert opt.optim_status == "Optimal"
    assert _balance_holds(res, loads=2)
    for k, (watts, hours) in enumerate(((3000.0, 4), (750.0, 2))):
        assert res[f"P_deferrable{k}"].sum() * 0.5 == pytest.approx(watts * hours, rel=1e-6)
    assert np.isfinite(_bill(res))


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
        site=LOADS,
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


def test_a_load_group_inside_one_group_plans_as_the_milp_does():
    """deferrable_load_groups with both loads in one group: that group's own
    EMHASS model holds the 3 kW budget, and the plan is the default MILP's."""
    rh, oc, pc, data, pv, load, buy, sell = _site()
    site = (rh, dict(oc, deferrable_load_groups=LOAD_GROUP), pc, data, pv, load, buy, sell)
    _, milp = _plan(*site)
    opt, res = _plan(
        *site,
        optimization_backend="dantzig_wolfe",
        site=LOADS,
    )
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    assert (res["P_deferrable0"] + res["P_deferrable1"]).max() <= 3000 + 1e-3
    assert _bill(res) == pytest.approx(_bill(milp), abs=0.01)


def test_a_load_group_across_groups_is_held_and_priced():
    """The same budget with each load planned on its own: the coordinator
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


def test_a_site_limit_holds_and_is_priced():
    """A limit over the two loads (each planned on its own): their total stays
    within 3.5 kW and its premium is fed_limit_price_<id>; a group that sits
    on two parents falls back, saying so."""
    site = _site()
    limit = [
        {"id": "loads_cap", "type": "limit", "max_import": 3500},
        {"id": "deferrable0", "parent": "grid", "limits": ["loads_cap"]},
        {"id": "deferrable1", "parent": "grid", "limits": ["loads_cap"]},
    ]
    opt, res = _plan(*site, optimization_backend="dantzig_wolfe", site=limit)
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    assert (res["P_deferrable0"] + res["P_deferrable1"]).max() <= 3500 + 1e-3
    assert "fed_limit_price_loads_cap" in res.columns
    with mock.patch.object(logger, "warning") as warning:
        _, res = _plan(
            *site,
            optimization_backend="dantzig_wolfe",
            site=[
                {"id": "half", "parent": "grid", "max_import": 1000},
                {"id": "deferrable0", "parent": "half", "group": "loads"},
                {"id": "deferrable1", "parent": "grid", "group": "loads"},
            ],
        )
    assert (res["backend_used"] == "cvxpy").all()
    assert res["backend_fallback_reason"].str.contains("sits on 'half' and 'grid'").all()
    # ...and says the default solver does not hold the site's limits
    assert any("does not hold site's nodes and limits" in str(c) for c in warning.call_args_list)


FOUR_DER_SITE = [
    {
        "id": "inverter",
        "parent": "grid",
        "type": "hybrid_inverter",
        "max_import": 4000,
        "max_export": 4000,
        "efficiency_import": 0.97,
        "efficiency_export": 0.97,
    },
    {"id": "garage", "parent": "grid", "type": "panel", "max_import": 7400, "max_export": 0},
    {"id": "heat", "type": "breaker", "parent": "garage", "max_import": 500},
    {"id": "pv", "parent": "inverter"},
    {"id": "battery", "parent": "inverter"},
    {"id": "deferrable0", "parent": "garage"},
    {"id": "deferrable1", "parent": "heat"},
]


def test_four_ders_on_one_site():
    """A hybrid inverter (4 kW) with the PV and the battery, and a garage
    panel (7.4 kW, no backfeed) holding one load and, under a 500 W breaker,
    the other, with a 3 kW budget over both loads (deferrable_load_groups,
    held by the coordinator as a limit). Every limit holds, every node is
    priced and reported, and every player has a share."""
    rh, oc, pc, data, pv, load, buy, sell = _site()
    oc = dict(
        oc,
        set_nodischarge_to_grid=False,
        treat_deferrable_load_as_semi_cont=[False, False],
        deferrable_load_groups=[{"names": ["deferrable0", "deferrable1"], "max_power": 3000}],
    )
    pc = dict(pc, compute_curtailment=True)
    site = (rh, oc, pc, data, pv * 1.6, load, buy, sell)
    opt, res = _plan(*site, optimization_backend="dantzig_wolfe", site=FOUR_DER_SITE)
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    assert res["P_hybrid_inverter"].abs().max() <= 4000 + 1e-3
    garage = res["P_deferrable0"] + res["P_deferrable1"]
    assert res["P_deferrable1"].max() <= 500 + 1e-3
    assert garage.max() <= 3000 + 1e-3 and garage.min() >= -1e-3  # the budget; no backfeed
    for node in ("inverter", "garage", "heat"):
        assert f"fed_local_price_{node}" in res.columns and f"fed_node_power_{node}" in res.columns
    assert np.allclose(-res["fed_node_power_garage"], garage, atol=1e-3)  # + = up the tree
    shares = {f"fed_share_{p}" for p in ("solar", "battery", "deferrable0", "deferrable1")}
    assert shares <= set(res.columns)
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
    assert res["backend_fallback_reason"].str.contains("set_nodischarge_to_grid").all()


def test_a_participant_answers_the_same_query_the_same_way():
    """A heat pump with thermal inertia answers the same prices with the same
    plan, whatever it was asked in between: each query starts from the same
    state, not from the heat input of the previous solve."""
    from emhass.coordinator.backend import EmhassParticipant
    from emhass.coordinator.interface import Query

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


# --------------------------------------------------------------------------
# Review findings (#1362): what the coordinator plans, and what it declines
# --------------------------------------------------------------------------


def _coordinated(site=None, pc_update=None, **options):
    """The test site, coordinated, with optim_conf `options` and plant_conf
    `pc_update`; returns (Optimization, opt_res)."""
    rh, oc, pc, data, pv, load, buy, sell = _site()
    pc = {**pc, **(pc_update or {})}
    if site is not None:
        options["site"] = site
    return _plan(
        rh, oc, pc, data, pv, load, buy, sell, optimization_backend="dantzig_wolfe", **options
    )


@pytest.mark.parametrize(
    "load",
    [
        {"nominal_power_of_deferrable_loads": [3000, [750, 750, 500, 500]]},  # a sequence
        {"minimum_power_of_deferrable_loads": [0, 400]},  # a minimum power
    ],
)
def test_a_load_with_a_binary_is_never_blended(load):
    """A sequence load or one with a minimum power has a binary in EMHASS's
    model: a mix of its plans is no plan it can run, so its participant is
    on/off and never modulating, even when it is not semi-continuous."""
    from emhass.coordinator.backend import EmhassParticipant

    rh, oc, pc, data, pv, load_w, buy, sell = _site()
    oc = {**oc, "set_use_battery": False, "treat_deferrable_load_as_semi_cont": [False, False]}
    oc.update(load)
    opt = Optimization(
        rh, oc, pc, "unit_load_cost", "unit_prod_price", "profit", emhass_conf, logger
    )
    p = EmhassParticipant(opt, "deferrable1", False, [1], data, None, None, {}, buy)
    assert not p.modulating and p.onoff
    plain = EmhassParticipant(
        Optimization(
            rh,
            {
                **oc,
                "nominal_power_of_deferrable_loads": [3000, 750],
                "minimum_power_of_deferrable_loads": [0, 0],
            },
            pc,
            "unit_load_cost",
            "unit_prod_price",
            "profit",
            emhass_conf,
            logger,
        ),
        "deferrable1",
        False,
        [1],
        data,
        None,
        None,
        {},
        buy,
    )
    assert plain.modulating and not plain.onoff


def test_a_sequence_load_runs_its_sequence():
    """The reviewer's case: a sequence load under a shared limit is planned as
    its sequence, at its own powers, not a blend of shifted copies."""
    sequence = [750, 750, 500, 500]
    _, res = _coordinated(
        site=[
            {"id": "l1", "type": "limit", "max_import": 3400},
            {"id": "deferrable0", "parent": "grid", "limits": ["l1"]},
            {"id": "deferrable1", "parent": "grid", "limits": ["l1"]},
        ],
        set_use_battery=False,
        nominal_power_of_deferrable_loads=[3000, sequence],
        treat_deferrable_load_as_semi_cont=[True, False],
    )
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    # the whole sequence, in order, once, and nothing else
    p = np.round(res["P_deferrable1"].values, 3)
    on = np.flatnonzero(p)
    assert len(on) == len(sequence), p
    assert np.array_equal(on, np.arange(on[0], on[0] + len(sequence))), p
    assert p[on].tolist() == sequence, p


def test_a_plan_that_breaks_a_hard_limit_is_declined():
    """maximum_power_from_grid is hard in EMHASS: when the on/off loads cannot
    fit under it, the default solver plans (relaxing a load), never a
    coordinated plan past the fuse."""
    rh, oc, pc, data, pv, load, buy, sell = _site()
    pc = {**pc, "maximum_power_from_grid": 2500}
    site = (rh, {**oc, "set_use_battery": False}, pc, data, pv * 0, load, buy, sell)
    _, res = _plan(*site, optimization_backend="dantzig_wolfe")
    assert (res["backend_used"] == "cvxpy").all()
    assert res["backend_fallback_reason"].str.contains("exceeds a limit").all()
    assert res["P_grid_pos"].max() <= 2500 + 1e-3


def test_a_relaxed_participant_is_read_from_the_problem_it_published():
    """1.25 h at 30-minute steps cannot be met by a semi-continuous load, so
    EMHASS relaxes its model; the participant's cost is that solve's, finite,
    not the failed solve's."""
    opt, res = _coordinated(
        set_use_battery=False, operating_hours_of_each_deferrable_load=[1.25, 2]
    )
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    assert np.isfinite(res["fed_lower_bound"].iloc[0] + res["fed_gap"].iloc[0])


def test_a_further_battery_in_site_is_declined_by_name():
    """site may name battery1 (#1032's convention), but EMHASS plans one
    battery here: the default solver plans, saying so, rather than the limit
    on it being dropped."""
    _, res = _coordinated(
        site=[
            {"id": "l1", "type": "limit", "max_import": 500},
            {"id": "battery1", "parent": "grid", "limits": ["l1"]},
        ]
    )
    assert (res["backend_used"] == "cvxpy").all()
    assert res["backend_fallback_reason"].str.contains("battery1").all()


def test_battery0_is_the_battery():
    """battery0 and battery are one battery: a limit on battery0 holds."""
    _, res = _coordinated(
        site=[
            {"id": "l1", "type": "limit", "max_import": 500},
            {"id": "battery0", "parent": "grid", "limits": ["l1"]},
        ],
        set_nodischarge_to_grid=False,
    )
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    assert (-res["P_batt"]).max() <= 500 + 1e-3


def test_the_master_holds_the_battery_only_where_its_model_is_linear():
    """The master holds EMHASS's battery as LP variables only when that is its
    whole model; otherwise EMHASS's own model plans it, as a participant."""
    from emhass.coordinator.backend import _linear_battery

    rh, oc, pc = _confs()
    oc = {**oc, "set_use_battery": True}
    lo, hi = pc["battery_minimum_state_of_charge"], pc["battery_maximum_state_of_charge"]
    buy = np.full(4, 0.2)
    assert _linear_battery(oc, pc, (lo + hi) / 2, buy)
    # a reward on the final state of charge, which the master does not price
    assert not _linear_battery({**oc, "battery_soc_final_reward_factor": 2.0}, pc, lo, buy)
    # starting outside the window: EMHASS's model recovers from there
    assert not _linear_battery(oc, pc, min(hi + 0.05, 1.0), buy)
    # a negative import price: without EMHASS's binary the LP would charge and
    # discharge at once, to be paid for the loss
    assert not _linear_battery(oc, pc, lo, np.array([0.2, -0.1, 0.2, 0.2]))


def test_a_negative_price_keeps_the_battery_consistent():
    """With a negative import price the battery is EMHASS's own model: its
    published power and state of charge agree in every slot."""
    rh, oc, pc, data, pv, load, buy, sell = _site()
    buy = buy.copy()
    buy[10:14] = -0.05
    sell = np.minimum(sell, buy)
    data = data.assign(unit_load_cost=buy, unit_prod_price=sell)
    # Paid to draw, everything runs at once; room at the meter, so this tests
    # the battery, not the grid limit (a plan past that is declined)
    pc = {**pc, "maximum_power_from_grid": 30000}
    opt, res = _plan(rh, oc, pc, data, pv, load, buy, sell, optimization_backend="dantzig_wolfe")
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    cap = pc["battery_nominal_energy_capacity"]
    eta_c, eta_d = pc["battery_charge_efficiency"], pc["battery_discharge_efficiency"]
    soc = np.concatenate([[0.5], res["SOC_opt"].values])
    p = res["P_batt"].values  # + discharging
    expected = np.where(p < 0, -p * eta_c, -p / eta_d) * 0.5 / cap
    assert np.allclose(np.diff(soc), expected, atol=1e-4)


def test_a_non_electric_load_is_declined():
    """EMHASS prices a non-electric load's commodity at the import tariff it
    is given, so a query would change that load's own cost, not only the
    electricity's: the default solver plans, saying why."""
    _, res = _coordinated(is_electric_load=[True, False])
    assert (res["backend_used"] == "cvxpy").all()
    assert res["backend_fallback_reason"].str.contains("non-electric").all()


def test_what_the_coordinator_cannot_split_yet_is_declined_by_name():
    """Per-slot grid limits, and a thermal load whose gains follow the house's
    load (a price response sets that load to zero)."""
    from emhass.coordinator.backend import unsupported

    rh, oc, pc = _confs()
    assert "as a list" in unsupported(
        oc, {**pc, "maximum_power_from_grid": [9000] * 24 + [2000] * 24}, "profit", {}
    )
    gains = [{"thermal_battery": {"internal_gains_factor": 0.5}}, {}]
    assert "internal_gains_factor" in unsupported(
        {**oc, "def_load_config": gains}, pc, "profit", {}
    )
    assert unsupported(oc, pc, "profit", {}) is None


def test_a_zero_export_cap_is_planned_and_held():
    """No export at all is a cap the coordinator holds, not a reason to fall
    back."""
    rh, oc, pc, data, pv, load, buy, sell = _site()
    pc = {**pc, "maximum_power_to_grid": 0}
    site = (rh, oc, pc, data, pv * 0, load, buy, sell)
    _, res = _plan(*site, optimization_backend="dantzig_wolfe", set_nodischarge_to_grid=False)
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    assert res["P_grid_neg"].min() >= -1e-3


def test_a_lossy_node_is_planned_within_its_draw():
    """A node that draws 1 / efficiency from its parent: the master's bounds
    are sized for what reaches the meter through it."""
    rh, oc, pc, data, pv, load, buy, sell = _site()
    site_list = [
        {"id": "lossy", "parent": "grid", "type": "meter", "efficiency_import": 0.25},
        {"id": "deferrable0", "parent": "lossy"},
    ]
    oc = {**oc, "set_use_battery": False}
    pc = {**pc, "maximum_power_from_grid": 30000}
    site = (rh, oc, pc, data, pv * 0, load * 0, buy, sell)
    _, res = _plan(*site, optimization_backend="dantzig_wolfe", site=site_list)
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    assert res["P_deferrable0"].sum() * 0.5 == pytest.approx(3000.0 * 4, rel=1e-6)
    assert np.allclose(res["P_grid"], res["P_deferrable0"] / 0.25 + res["P_deferrable1"], atol=1e-2)


def test_the_coordinator_stops_at_its_time_limit():
    """Past its time budget (EMHASS's lp_solver_timeout) the coordinator stops
    iterating, and the best plan from what it has stands, said so."""
    from emhass.coordinator.master import DWCoordinator

    run = DWCoordinator.run

    def hurried(self, **kwargs):
        return run(self, **{**kwargs, "max_seconds": 0.0})

    with mock.patch.object(DWCoordinator, "run", hurried):
        _, res = _coordinated()
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    assert (res["fed_stop_reason"] == "time limit").all()


def test_a_cheaper_plan_with_the_same_power_replaces_the_dearer():
    """Two plans with the same power: the pool keeps the cheaper, so recovery
    never picks a dearer one for no reason."""
    from emhass.coordinator.master import Column, Device

    dev = Device("p", "participant", cfg=None)
    power = np.array([1.0, 0.0])
    assert dev.add(Column(power, np.zeros(3), 10.0, "baseline"))
    assert not dev.add(Column(power.copy(), np.zeros(3), 12.0, "price"))
    assert dev.add(Column(power.copy(), np.ones(3), 0.0, "price"))
    assert len(dev.columns) == 1 and dev.columns[0].cost == 0.0
    assert np.all(dev.columns[0].trajectory == 1.0)


def test_delivered_export_is_priced_under_the_export_ceiling():
    """The saving split prices what is delivered: export past the ceiling is
    curtailed PV, as the plan's own score and results take it."""
    from emhass.coordinator.model import Forecasts
    from emhass.coordinator.settlement import _under_ceiling

    fc = Forecasts(
        buy=np.ones(2),
        sell=np.ones(2),
        load=np.zeros(2),
        solar=np.ones(2),
        export_ceiling=np.array([1.0, 0.0]),
    )
    assert np.allclose(_under_ceiling(fc, np.array([-2.0, -1.0])), [-1.0, 0.0])
    assert np.allclose(_under_ceiling(fc, np.array([3.0, 0.5])), [3.0, 0.5])


def _battery_steps_agree(res, pc, soc0, step_h=0.5):
    """Whether the published P_batt (+ discharging) steps SOC_opt from `soc0`
    as one battery would, in every slot."""
    cap = pc["battery_nominal_energy_capacity"]
    eta_c, eta_d = pc["battery_charge_efficiency"], pc["battery_discharge_efficiency"]
    soc = np.concatenate([[soc0], res["SOC_opt"].values])
    p = res["P_batt"].values
    expected = np.where(p < 0, -p * eta_c, -p / eta_d) * step_h / cap
    return np.allclose(np.diff(soc), expected, atol=1e-4)


def test_a_battery_starting_above_its_window_keeps_its_charge():
    """Starting at 95% with a 90% maximum: EMHASS's own model plans the
    battery (the master's would clamp it to 90%), so the plan starts at 95%
    and every step follows from P_batt - no energy lost before any power."""
    rh, oc, pc, data, pv, load, buy, sell = _site()
    oc = {**oc, "optimization_backend": "dantzig_wolfe"}
    pc = {**pc, "battery_maximum_state_of_charge": 0.9}
    opt = Optimization(
        rh, oc, pc, "unit_load_cost", "unit_prod_price", "profit", emhass_conf, logger
    )
    res = opt.perform_optimization(data, pv, load, buy, sell, soc_init=0.95, soc_final=0.5)
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    assert _battery_steps_agree(res, pc, 0.95)


def test_a_battery_plan_that_cycles_is_never_published():
    """CodeRabbit's case at positive prices: a full battery, no export and no
    curtailment, with PV to spare. The master's battery (no charge-or-
    discharge binary) charges and discharges at once to burn the surplus;
    that plan is declined by name, never published."""
    from emhass.coordinator import backend

    rh, oc, pc, data, pv, load, buy, sell = _site()
    oc = {**oc, "optimization_backend": "dantzig_wolfe", "set_nodischarge_to_grid": False}
    pc = {
        **pc,
        "maximum_power_to_grid": 0,
        "compute_curtailment": False,
        "battery_maximum_state_of_charge": 0.9,
        "battery_charge_efficiency": 0.9,
        "battery_discharge_efficiency": 0.9,
    }
    opt = Optimization(
        rh, oc, pc, "unit_load_cost", "unit_prod_price", "profit", emhass_conf, logger
    )
    assert backend.optimize(opt, data, pv, load, buy, sell, soc_init=0.9, soc_final=0.9) is None
    assert "charges and discharges" in opt.fed_fallback_reason


def test_the_cycle_check_declines_a_battery_plan_that_does_not_step():
    """The check itself: a battery plan whose net power does not step its
    state (charge and discharge at once) is declined, and the default
    solver plans."""
    import dataclasses

    from emhass.coordinator.master import DWCoordinator

    run = DWCoordinator.run

    def cycling(self, **kwargs):
        r = run(self, **kwargs)
        b = r.plan["battery"]
        flat = dataclasses.replace(b, trajectory=np.full_like(b.trajectory, b.trajectory[0]))
        flat.power = np.where(np.arange(len(b.power)) == 3, 0.5, b.power)
        return dataclasses.replace(r, plan={**r.plan, "battery": flat})

    with mock.patch.object(DWCoordinator, "run", cycling):
        _, res = _coordinated()
    assert (res["backend_used"] == "cvxpy").all()
    assert res["backend_fallback_reason"].str.contains("charges and discharges").all()


def test_a_plan_that_cannot_run_under_the_export_ceiling_is_declined():
    """A recovered plan that scores infinite - past the export ceiling with no
    PV left to curtail - is never published: the default solver plans and
    says why."""
    import dataclasses

    from emhass.coordinator.master import DWCoordinator

    run = DWCoordinator.run
    calls = []

    def unrunnable(self, **kwargs):
        r = run(self, **kwargs)
        calls.append(r)
        return dataclasses.replace(r, upper=np.inf) if len(calls) == 1 else r

    with mock.patch.object(DWCoordinator, "run", unrunnable):
        _, res = _coordinated()
    assert (res["backend_used"] == "cvxpy").all()
    assert res["backend_fallback_reason"].str.contains("within the export ceiling").all()


def test_settlement_prices_the_export_delivered_under_the_ceiling():
    """The saving split's slot cost prices what is delivered: a battery that
    would export past the ceiling is priced at the ceiling (the rest is
    curtailed PV), and a subset of slots is priced as those slots of the
    whole."""
    from emhass.coordinator import settlement
    from emhass.coordinator.master import DWCoordinator
    from emhass.coordinator.submeter import site_meter

    seen = []
    run = DWCoordinator.run

    def keep(self, **kwargs):
        r = run(self, **kwargs)
        seen.append((self, r.plan))
        return r

    with mock.patch.object(DWCoordinator, "run", keep):
        _, res = _coordinated(pc_update={"compute_curtailment": True})
    (co, plan), (co_dark, plan_dark) = seen[0], seen[1]
    assert co.fc.export_ceiling is not None
    n = len(co.fc.buy)
    surplus = np.flatnonzero(co.fc.export_ceiling > 0.5)
    assert len(surplus), "the test day needs PV to spare"
    discharge = np.zeros(n)
    discharge[surplus] = -2.0  # kW, + = drawn: the battery exporting on top of the PV
    x = settlement._draws(co, co.fc, {"battery": discharge})
    priced = settlement._slot_cost(co, co.fc, x)
    t = surplus[0]
    raw = settlement.meter_from_draws(co.cfg, co.fc, *settlement._split(x)).net[t]
    assert -raw > co.fc.export_ceiling[t] + 1e-6  # past the ceiling before it applies
    delivered = -co.fc.export_ceiling[t] * co.fc.sell[t] * co.dt
    assert priced[t] == pytest.approx(delivered)
    idx = np.array([t, 0, t])
    sub = settlement._slot_cost(
        co, settlement._slots(co.fc, idx), {k: v[idx] for k, v in x.items()}
    )
    assert np.allclose(sub, priced[idx])
    # The ledger's bill after, for a plan with that export past the ceiling,
    # is the bill on the export delivered
    import dataclasses

    b = plan["battery"]
    # half a kW more export where there is PV to curtail for it
    extra = np.where(co.fc.solar > 1.0, -0.5, 0.0)
    pushed = {**plan, "battery": dataclasses.replace(b, power=b.power + extra)}
    flows = site_meter(co.cfg, co.fc, {k: c.power for k, c in pushed.items()})
    assert np.any(-flows.net > co.fc.export_ceiling + 1e-6)
    z = co._delivered(flows)
    assert z is not None
    delivered_bill = float(
        np.sum((np.maximum(z, 0.0) * co.fc.buy - np.maximum(-z, 0.0) * co.fc.sell) * co.dt)
    )
    totals = settlement.ledger(co, pushed, co_dark, plan_dark)["totals"]
    assert totals["house_bill_after"] == pytest.approx(delivered_bill)


def test_the_objective_breakdown_totals_the_score():
    """`parts()` reports the plan on the flow delivered under the export
    ceiling, as `score()` does, so its total is the plan's score."""
    from emhass.coordinator.master import DWCoordinator

    seen = []
    run = DWCoordinator.run

    def keep(self, **kwargs):
        r = run(self, **kwargs)
        seen.append((self, r))
        return r

    with mock.patch.object(DWCoordinator, "run", keep):
        _coordinated(pc_update={"compute_curtailment": True})
    import dataclasses

    co, r = seen[0]
    assert co.export_ceiling is not None
    assert co.parts(r.plan)["total"] == pytest.approx(co.score(r.plan))
    # a plan exporting past the ceiling where PV can be curtailed: the
    # breakdown prices the delivered export, as the score does
    b = r.plan["battery"]
    surplus = co.export_ceiling > 0.5
    pushed = dataclasses.replace(b, power=np.where(surplus, -1.0, b.power))
    plan = {**r.plan, "battery": pushed}
    assert np.isfinite(co.score(plan))
    assert co.parts(plan)["total"] == pytest.approx(co.score(plan))


def test_a_plan_past_the_ceiling_with_nothing_to_curtail_scores_infinite_and_is_declined():
    """The scoring path itself: the battery exporting at night, past the
    export ceiling (the PV surplus, zero then) with no PV to curtail, scores
    infinite; the coordinator's plan replaced by it is declined and the
    default solver plans."""
    import dataclasses

    from emhass.coordinator.master import DWCoordinator

    run = DWCoordinator.run
    scored = []

    def night_export(self, **kwargs):
        r = run(self, **kwargs)
        if scored:  # the saving split's run: as it was
            return r
        b = r.plan["battery"]
        night = self.export_ceiling <= 1e-9
        plan = {**r.plan, "battery": dataclasses.replace(b, power=np.where(night, -1.0, b.power))}
        scored.append(self.score(plan))
        return dataclasses.replace(r, plan=plan, upper=scored[0])

    with mock.patch.object(DWCoordinator, "run", night_export):
        _, res = _coordinated()
    assert scored and scored[0] == np.inf
    assert (res["backend_used"] == "cvxpy").all()
    assert res["backend_fallback_reason"].str.contains("within the export ceiling").all()


# --------------------------------------------------------------------------
# CodeRabbit's second round (#1362)
# --------------------------------------------------------------------------


def test_what_moves_a_participants_own_cost_with_the_query_is_declined():
    """Configurations where EMHASS's model prices something of the
    participant's own at the queried price, or carries state between runs
    the coordinator does not: the default solver plans, saying which."""
    from emhass.coordinator.backend import unsupported

    rh, oc, pc = _confs()
    battery = {**oc, "set_use_battery": True}
    assert "battery_soc_final_reward_factor" in unsupported(
        {**battery, "battery_soc_final_reward_factor": 2.0}, pc, "profit", {}
    )
    inertia = [{"thermal_battery": {"thermal_inertia_time_constant": 2.0}}, {}]
    assert "thermal_inertia_time_constant" in unsupported(
        {**oc, "def_load_config": inertia}, pc, "profit", {}
    )
    # with q_input_initial, both solvers start every run from it
    pinned = [{"thermal_battery": {"thermal_inertia_time_constant": 2.0, "q_input_initial": 1.0}}]
    assert unsupported({**oc, "def_load_config": pinned + [{}]}, pc, "profit", {}) is None
    assert "non-electric" in unsupported(
        {**oc, "is_electric_load": [True, False]}, pc, "profit", {}
    )


def test_a_thermal_battery_with_an_overshoot_gate_is_never_blended():
    """A thermal battery's overshoot temperature gates its power with a
    binary, as a thermal_config's does: its participant is on/off."""
    from emhass.coordinator.backend import EmhassParticipant

    rh, oc, pc, data, pv, load_w, buy, sell = _site()
    oc = {
        **oc,
        "set_use_battery": False,
        "treat_deferrable_load_as_semi_cont": [False, False],
        "def_load_config": [{}, {"thermal_battery": {"overshoot_temperature": 21.0}}],
    }
    opt = Optimization(
        rh, oc, pc, "unit_load_cost", "unit_prod_price", "profit", emhass_conf, logger
    )
    p = EmhassParticipant(opt, "deferrable1", False, [1], data, None, None, {}, buy)
    assert not p.modulating and p.onoff


def test_an_lp_with_every_variable_fixed_is_solved_or_refused():
    """Nothing left to solve: x is its bounds, optimal when it meets A x = b,
    infeasible otherwise - never an error from an empty reduction."""
    from emhass.coordinator.lpsolver import Triplets, linprog

    A = Triplets((1, 1))
    A[0, 0] = 1.0
    ok = linprog(np.array([2.0]), A, np.array([5.0]), np.array([5.0]), np.array([5.0]))
    assert ok.status == 0 and np.allclose(ok.x, [5.0]) and ok.fun == pytest.approx(10.0)
    bad = linprog(np.array([2.0]), A, np.array([4.0]), np.array([5.0]), np.array([5.0]))
    assert bad.status != 0


def test_only_main_bus_pv_is_curtailed_for_the_export_ceiling():
    """PV behind a node is that node's: curtailing it would change the node's
    flow. A plan exporting past the ceiling with PV only behind a node cannot
    run, and the breakdown and score say so."""
    import dataclasses

    from emhass.coordinator.master import DWCoordinator
    from emhass.coordinator.submeter import site_meter

    seen = []
    run = DWCoordinator.run

    def keep(self, **kwargs):
        r = run(self, **kwargs)
        seen.append((self, r.plan))
        return r

    site = [
        {"id": "roof", "parent": "grid", "type": "inverter"},
        {"id": "pv", "parent": "roof"},
        {"id": "battery0", "parent": "grid"},
    ]
    with mock.patch.object(DWCoordinator, "run", keep):
        _coordinated(site=site, pc_update={"compute_curtailment": True})
    co, plan = seen[0]
    b = plan["battery"]
    surplus = co.export_ceiling > 0.5
    pushed = {**plan, "battery": dataclasses.replace(b, power=np.where(surplus, -1.0, b.power))}
    flows = site_meter(co.cfg, co.fc, {k: c.power for k, c in pushed.items()})
    assert np.any(-flows.net > co.export_ceiling + 1e-6)
    assert co._delivered(flows) is None
    assert co.score(pushed) == np.inf and co.parts(pushed)["total"] == np.inf


def test_the_coordinators_inputs_are_checked():
    """A ceiling of the wrong length or not finite, a node with a name the
    saving split keeps for itself, and PV counted in a set limit (its row sums
    devices) are refused when the coordinator is built."""
    from emhass.coordinator import site as site_mod
    from emhass.coordinator.model import Forecasts, Horizon, SetLimit, SiteConfig, SubMeter

    h = Horizon(dt=0.5, hours=1.0)
    base = {"buy": np.ones(2), "sell": np.zeros(2), "load": np.zeros(2), "solar": np.zeros(2)}
    for ceiling in (np.array([np.nan, 0.0]), np.array([1.0])):
        with pytest.raises(ValueError, match="export_ceiling"):
            Forecasts(**base, export_ceiling=ceiling).validate(h)
    with pytest.raises(ValueError, match="reserved"):
        SiteConfig(horizon=h, battery=None, submeters=(SubMeter("ac", ("x",)),)).validate()
    assert "reserved" in site_mod.site_problem(
        [{"id": "ac", "parent": "grid", "type": "panel"}, {"id": "deferrable0", "parent": "ac"}]
    )
    # "solar" is the ledger's player for the PV: no node or group takes it
    assert "reserved" in site_mod.site_problem(
        [{"id": "solar", "parent": "grid"}, {"id": "deferrable0", "parent": "solar"}]
    )
    assert "reserved" in site_mod.site_problem(
        [{"id": "deferrable0", "parent": "grid", "group": "solar"}]
    )
    from emhass.coordinator.master import DWCoordinator

    cfg = SiteConfig(horizon=h, set_limits=(SetLimit("l", ("pv",), max_export_kw=0.0),))
    with pytest.raises(ValueError, match="no device"):
        DWCoordinator(cfg, Forecasts(**base))


def test_an_omitted_efficiency_takes_the_plants_written_as_text():
    """A plant efficiency given as text in a one-element list (as the plant's
    own keys accept) is the site's default, not 1.0."""
    from emhass.coordinator import site as site_mod

    site = [
        {"id": "grid"},
        {"id": "inverter", "parent": "grid", "type": "hybrid_inverter"},
        {"id": "pv", "parent": "inverter"},
        {"id": "battery0", "parent": "inverter"},
    ]
    plant = {"inverter_efficiency_dc_ac": ["0.96"], "inverter_efficiency_ac_dc": "0.95"}
    assert site_mod.compile_site(site, plant, logger, battery=True)
    assert plant["inverter_efficiency_dc_ac"] == pytest.approx(0.96)
    assert plant["inverter_efficiency_ac_dc"] == pytest.approx(0.95)


def test_a_nodes_own_clipping_leaves_the_main_bus_pv_to_curtail():
    """CodeRabbit's case: 1 kW of PV on the main bus, and a node with a zero
    export cap clipping its own 2 kW. With an export ceiling of zero, the main
    bus's 1 kW is curtailed; the node's clipping does not use it up."""
    from emhass.coordinator.master import DWCoordinator
    from emhass.coordinator.model import Forecasts, GridLimits, Horizon, SiteConfig, SubMeter
    from emhass.coordinator.submeter import site_meter

    h = Horizon(dt=1.0, hours=2.0)
    cfg = SiteConfig(
        horizon=h,
        battery=None,
        grid=GridLimits(allow_curtailment=True),
        submeters=(SubMeter("node", ("pv2",), max_export_kw=0.0),),
    )
    fc = Forecasts(
        buy=np.ones(2),
        sell=np.full(2, 0.5),
        load=np.zeros(2),
        solar=np.ones(2),
        pv_arrays={"pv2": np.full(2, 2.0)},
    )
    co = DWCoordinator(cfg, fc, export_ceiling=np.zeros(2))
    flows = site_meter(cfg, co.fc, {})
    delivered = co._delivered(flows)
    assert delivered is not None and np.allclose(delivered, 0.0)


def test_a_grid_connection_may_not_take_a_reserved_name():
    """The saving split keeps the main bus's total as "ac" and set totals
    under "set:": a further grid connection may take neither, as a node may
    not."""
    from emhass.coordinator.model import GridConnection, Horizon, SiteConfig

    h = Horizon(dt=0.5, hours=1.0)
    for name in ("ac", "set:x"):
        with pytest.raises(ValueError, match="reserved"):
            SiteConfig(horizon=h, battery=None, connections=(GridConnection(name),)).validate()


def test_the_soc_miss_is_priced_at_the_participants_own_tariff():
    """EMHASS scales its terminal-SoC penalty with the highest import price it
    is given. A participant pins it to its own tariff, so a query changes the
    battery's bill, not its objective, and each answer minimises the
    objective it reports (CodeRabbit's case: an own tariff of 0.001 and a
    query at 1, where discharging the target's kWh pays)."""
    from emhass.coordinator.backend import EmhassParticipant
    from emhass.coordinator.interface import Query

    rh, oc, pc, data, pv, load, buy, sell = _site()
    oc = {
        **oc,
        "set_use_battery": True,
        "number_of_deferrable_loads": 0,
        "set_nodischarge_to_grid": False,
        "weight_battery_charge": 0.001,  # the battery's own model, not the master's
    }
    pc = {**pc, "battery_charge_efficiency": 1.0, "battery_discharge_efficiency": 1.0}
    opt = Optimization(
        rh, oc, pc, "unit_load_cost", "unit_prod_price", "profit", emhass_conf, logger
    )
    n = len(data)
    own = np.full(n, 0.001)
    p = EmhassParticipant(opt, "battery", True, [], data, 0.5, 0.5, {}, own)
    answers = {}
    for price in (0.001, 1.0):
        q = np.full(n, price)
        answers[price] = p.respond(Query("price_response", q, q))
        # the coefficient is the own tariff's, whatever the query
        assert p.opt.param_soc_final_penalty.value == pytest.approx(0.001 * 100 * 0.001)

    def objective(a, price):
        """What a price response minimises: private cost plus the energy at
        the query's price."""
        return a.private_cost + price * float(np.sum(a.plan_kw)) * 0.5

    # each answer is the best of the two at its own query's price
    for price, a in answers.items():
        for other in answers.values():
            assert objective(a, price) <= objective(other, price) + 1e-6
    # at 1.0 the battery's energy is worth more than the miss costs: it sells
    assert np.sum(answers[1.0].plan_kw) < -0.5


def test_an_answer_starts_from_the_soc_the_solve_used():
    """With no initial SoC, EMHASS starts the battery at the final one; the
    answer's state trajectory starts there too, not at the target."""
    from emhass.coordinator.backend import EmhassParticipant
    from emhass.coordinator.interface import Query

    rh, oc, pc, data, pv, load, buy, sell = _site()
    oc = {
        **oc,
        "set_use_battery": True,
        "number_of_deferrable_loads": 0,
        "weight_battery_charge": 0.001,
    }
    pc = {**pc, "battery_target_state_of_charge": 0.5}
    opt = Optimization(
        rh, oc, pc, "unit_load_cost", "unit_prod_price", "profit", emhass_conf, logger
    )
    p = EmhassParticipant(opt, "battery", True, [], data, None, 0.8, {}, buy)
    a = p.respond(Query("price_response", buy, buy))
    cap = pc["battery_nominal_energy_capacity"] / 1000.0
    assert a.trajectory[0] == pytest.approx(0.8 * cap)


def test_the_ledger_counts_the_pv_the_export_ceiling_curtails():
    """The ledger's curtailed energy includes what the export ceiling
    curtails at the main meter, as the results do."""
    from emhass.coordinator import settlement
    from emhass.coordinator.master import DWCoordinator

    seen = []
    run = DWCoordinator.run

    def keep(self, **kwargs):
        r = run(self, **kwargs)
        seen.append((self, r.plan))
        return r

    with mock.patch.object(DWCoordinator, "run", keep):
        _, res = _coordinated(pc_update={"compute_curtailment": True})
    (co, plan), (co_dark, plan_dark) = seen[0], seen[1]
    import dataclasses

    b = plan["battery"]
    # half a kW more export where there is PV to curtail for it: past the
    # ceiling, that much PV is curtailed
    extra = np.where(co.fc.solar > 1.0, -0.5, 0.0)
    pushed = {**plan, "battery": dataclasses.replace(b, power=b.power + extra)}
    totals = settlement.ledger(co, pushed, co_dark, plan_dark)["totals"]
    from emhass.coordinator.submeter import site_meter

    flows = site_meter(co.cfg, co.fc, {k: c.power for k, c in pushed.items()})
    delivered = co._delivered(flows)
    assert delivered is not None and np.any(delivered - flows.net > 1e-6)
    expected = float((flows.curtail + (delivered - flows.net)).sum() * co.dt)
    assert totals["curtailed_kwh"] == pytest.approx(expected)


def test_a_cell_with_two_kinks_inside_is_cut(monkeypatch):
    """The same gradient just inside both ends of a cell does not make it one
    linear piece: the bill can dip and level off between them. Such a cell is
    halved until its gradient explains its change, so the change is priced
    where it happens and not handed to whichever total moved most."""
    import types

    from emhass.coordinator import settlement

    def bill(_co, _fc, x):
        # flat, down by 0.1 between 0.05 and 0.15 kW of "a", flat again; "b"
        # (an independent connection at no tariff) costs nothing
        return -np.clip((x["a"] - 0.05) / 0.1, 0.0, 1.0) * 0.1

    monkeypatch.setattr(settlement, "_slot_cost", bill)
    monkeypatch.setattr(settlement, "_slots", lambda fc, idx: idx)
    co = types.SimpleNamespace(n=1)
    x0 = {"a": np.zeros(1), "b": np.zeros(1)}
    x1 = {"a": np.array([3.2]), "b": np.array([4.0])}
    q = settlement._bus_prices(co, None, x0, x1)
    assert q["b"][0] == pytest.approx(0.0, abs=1e-9)  # it changed nothing
    assert q["a"][0] * 3.2 == pytest.approx(-0.1)  # the whole change


def test_the_baseline_bills_add_up_under_an_export_ceiling(monkeypatch):
    """With no PV the export ceiling is zero, so a baseline that exports
    earns nothing in the house's bill before: the players' bills before are
    split the same way, and add up to it."""
    from emhass.coordinator import settlement
    from emhass.coordinator.master import DWCoordinator

    seen = []
    run = DWCoordinator.run

    def keep(self, **kwargs):
        r = run(self, **kwargs)
        seen.append((self, r.plan))
        return r

    with mock.patch.object(DWCoordinator, "run", keep):
        _, res = _coordinated(set_nodischarge_to_grid=True)
    (co, plan), (co_dark, plan_dark) = seen[0], seen[1]
    assert co.fc.export_ceiling is not None and not co.cfg.has_tree
    baseline = settlement._baseline

    def exporting(c, key):
        p, traj, cost = baseline(c, key)
        if key == "battery":  # its baseline sells 5 kW in the first slots
            p = p.copy()
            p[:4] = -5.0
        return p, traj, cost

    monkeypatch.setattr(settlement, "_baseline", exporting)
    out = settlement.ledger(co, plan, co_dark, plan_dark)
    before = sum(r["bill_before"] for r in out["rows"])
    assert before == pytest.approx(out["totals"]["house_bill_before"])


def _schema_accepts(site) -> bool:
    """Whether the site schema in param_definitions.json (and so
    openapi.json) accepts `site`: the subset of JSON Schema it uses -
    required, properties, additionalProperties, anyOf of required, and per
    value type, const, pattern, not, enum, minimum, maximum,
    exclusiveMinimum, maxItems, items and nested objects."""
    import json
    import pathlib
    import re

    defs = json.loads(
        (
            pathlib.Path(__file__).resolve().parents[1]
            / "src/emhass/static/data/param_definitions.json"
        ).read_text(encoding="utf-8")
    )
    schema = defs["Local"]["site"]["schema"]
    alternatives = schema["items"]["anyOf"]

    kinds = {
        "string": lambda v: isinstance(v, str),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "null": lambda v: v is None,
        "boolean": lambda v: isinstance(v, bool),
        "object": lambda v: isinstance(v, dict),
    }

    def value_ok(schema, v):
        if "const" in schema and v != schema["const"]:
            return False
        types = schema.get("type")
        if types is not None:
            types = [types] if isinstance(types, str) else types
            if not any(kinds[t](v) for t in types):
                return False
        if "enum" in schema and v not in schema["enum"]:
            return False
        if isinstance(v, str) and "pattern" in schema and not re.search(schema["pattern"], v):
            return False
        if kinds["number"](v):
            if "minimum" in schema and v < schema["minimum"]:
                return False
            if "maximum" in schema and v > schema["maximum"]:
                return False
            if "exclusiveMinimum" in schema and v <= schema["exclusiveMinimum"]:
                return False
        if isinstance(v, dict) and "properties" in schema and not accepts(schema, v):
            return False  # an object (a remote's solver): checked as an element is
        if isinstance(v, list):
            if len(v) > schema.get("maxItems", len(v)):
                return False
            if "items" in schema and not all(value_ok(schema["items"], x) for x in v):
                return False
        return not ("not" in schema and value_ok(schema["not"], v))

    def accepts(alt, e):
        props = alt["properties"]
        if any(k not in e for k in alt.get("required", [])) or set(e) - set(props):
            return False
        if "anyOf" in alt and not any(all(k in e for k in a["required"]) for a in alt["anyOf"]):
            return False
        return all(value_ok(props[k], v) for k, v in e.items())

    if not value_ok({k: v for k, v in schema.items() if k != "items"}, site):
        return False  # the list itself: its type and maxItems
    return all(any(accepts(alt, e) for alt in alternatives) for e in site)


@pytest.mark.parametrize(
    "site",
    [
        [{"id": "grid", "max_import": 9000}, {"id": "pv", "parent": "grid"}],
        [
            {"id": "roof", "parent": "grid", "type": "inverter", "max_export": 5000},
            {"id": "pv", "parent": "roof"},
            {"id": "feed", "type": "limit", "max_export": 3000},
            {"id": "deferrable0", "parent": "grid", "group": "pool", "limits": ["feed"]},
            {"id": "battery0", "parent": "roof", "solver": "emhass"},
        ],
        [{"id": "garage", "parent": "grid", "solver": {"url": "https://garage.lan/p"}}],
        [
            {
                "id": "garage",
                "parent": "grid",
                "solver": {"url": "http://garage.lan:5000/p", "trusted_network": True},
            }
        ],
        [
            {"id": "grid", "max_import": 9000.5, "max_export": 0},
            {"id": "panel", "parent": "grid", "efficiency_import": None, "efficiency_export": 0.97},
            {"id": "deferrable0", "parent": "panel"},
        ],
    ],
)
def test_the_site_schema_accepts_what_the_site_check_accepts(site):
    from emhass.coordinator import site as site_mod

    assert site_mod.site_problem(site) is None and _schema_accepts(site)


@pytest.mark.parametrize(
    "site",
    [
        [{"id": "grid", "parent": "grid"}],  # the main meter has no parent
        [{"id": "ac", "parent": "grid"}],  # reserved for the main bus's total
        [{"id": "solar", "parent": "grid"}],  # reserved for the ledger's PV
        [{"id": "pv", "type": "limit", "max_import": 1000}],  # a device's name
        [{"id": "grid", "type": "limit", "max_import": 1000}],
        [{"id": "feed", "type": "limit"}],  # no cap
        [{"id": "pv", "parent": "grid", "solver": "emhass"}],  # the PV is a forecast
        [{"id": "pv", "parent": "grid", "group": "g"}],
        [{"id": "deferrable0", "parent": "grid", "group": "solar"}],
        [{"id": "grid", "max_export": "5000"}],  # a number, not text
        [{"id": "grid", "max_import": -1}],
        [{"id": "panel", "parent": "grid", "efficiency_export": 1.5}],
        [{"id": "deferrable0", "parent": "grid", "limits": "feed"}],  # a list
        [{"id": "grid"}] + [{"id": f"n{k}", "parent": "grid"} for k in range(64)],  # over 64
        [{"id": "solar", "parent": "grid", "solver": {"url": "https://g.lan/p"}}],  # a remote
        [{"id": "pv", "parent": "grid", "solver": {"url": "https://g.lan/p"}}],
        [{"id": "g", "parent": "grid", "solver": {"url": "https://g.lan/p", "trusted_network": 1}}],
    ],
)
def test_the_site_schema_refuses_what_the_site_check_refuses(site):
    """The schema does not accept an element the site check refuses: a
    client that keeps to it does not have its site ignored at run time."""
    from emhass.coordinator import site as site_mod

    assert site_mod.site_problem(site) is not None
    assert not _schema_accepts(site)
