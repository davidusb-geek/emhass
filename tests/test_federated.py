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
