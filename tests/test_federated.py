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


def test_default_backend_is_cvxpy_and_never_dispatches():
    """By default the backend is cvxpy and the coordinator is never called."""
    rh, oc, pc = _confs()
    assert oc["optimization_backend"] == "cvxpy"
    assert not oc.get("participants")
    with mock.patch.object(optimization, "_coordinated_plan") as coordinated:
        _plan(*_site())
    coordinated.assert_not_called()


def test_missing_package_falls_back_to_cvxpy():
    """dantzig_wolfe without the optional package returns exactly the default plan."""
    site = _site()
    _, milp = _plan(*site)
    with mock.patch.dict(sys.modules, {"home_energy_optimizer.integrations.emhass": None}):
        opt, res = _plan(*site, optimization_backend="dantzig_wolfe")
    pd.testing.assert_frame_equal(res, milp)


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
    pd.testing.assert_frame_equal(res, milp)
    assert any("set_nocharge_from_grid" in str(c) for c in warning.call_args_list)
