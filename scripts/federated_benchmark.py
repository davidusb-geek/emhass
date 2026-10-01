#!/usr/bin/env python3
"""Compare the coordinated backend (optimization_backend: dantzig_wolfe) with
EMHASS's default MILP on the same inputs.

Two sets of runs, each solved by both backends:

* EMHASS's study cases (docs/study_cases), through EMHASS's own day-ahead and
  naive-MPC actions with the bundled test data, as dry runs (nothing written).
* Synthetic days: one PV and load profile, three tariffs (dynamic, day/night,
  flat), 15- and 30-minute steps; a battery with two on/off loads (with and
  without set_nodischarge_to_grid), and a battery, an on/off load and a heat
  pump.

Cost is EMHASS's own objective, the bill plus its penalty terms: minus the
MILP's objective value, and the coordinator's plan value (which prices the
same terms through each participant's EMHASS model). The MILP is solved
exactly (lp_solver_mip_rel_gap: 0) unless --mip-gap says otherwise.

Needs the optional extra: pip install emhass[federated]

Usage:
    python scripts/federated_benchmark.py [--mip-gap 0.01]
"""

import argparse
import asyncio
import copy
import pathlib
import statistics
import time
from unittest import mock

import numpy as np
import orjson
import pandas as pd

from emhass.command_line import dayahead_forecast_optim, naive_mpc_optim, set_input_data_dict
from emhass.optimization import Optimization
from emhass.utils import build_config, build_params, build_secrets, get_logger, get_yaml_parse

try:
    from home_energy_optimizer.dw.coordinator import DWCoordinator
except ImportError as exc:  # pragma: no cover - a script, not a test
    raise SystemExit("needs the optional extra: pip install emhass[federated]") from exc

ROOT = pathlib.Path(__file__).resolve().parents[1]
EMHASS_CONF = {
    "data_path": ROOT / "data/",
    "root_path": ROOT / "src/emhass/",
    "defaults_path": ROOT / "src/emhass/data/config_defaults.json",
    "associations_path": ROOT / "src/emhass/data/associations.csv",
}
LOGGER, _ = get_logger("federated_benchmark", EMHASS_CONF, save_to_file=False)
LOGGER.setLevel("ERROR")
H = 48  # study cases: 48 half-hour steps


# ----------------------------------------------------------------- helpers
async def _params() -> dict:
    """EMHASS's default parameters, built as at startup.

    Returns:
        dict: The params dict (retrieve_hass_conf, optim_conf, plant_conf, ...).
    """
    config = await build_config(EMHASS_CONF, LOGGER, EMHASS_CONF["defaults_path"])
    _, secrets = await build_secrets(EMHASS_CONF, LOGGER, no_response=True)
    return await build_params(EMHASS_CONF, secrets, config, LOGGER)


class _Capture:
    """Records the coordinator's first result in a solve (the plan; a second
    run, without PV, only splits the saving)."""

    def __init__(self):
        self.result = None
        self._run = DWCoordinator.run

    def __enter__(self):
        capture = self

        def run(coordinator, **kwargs):
            r = capture._run(coordinator, **kwargs)
            if capture.result is None:
                capture.result = r
            return r

        self._patch = mock.patch.object(DWCoordinator, "run", run)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()


def _objective(opt, capture: _Capture, backend: str) -> float:
    """EMHASS's objective (bill plus penalties, lower is better) of a solve.

    Args:
        opt: The Optimization that solved.
        capture: The coordinator result recorded during the solve, if any.
        backend: "cvxpy" or "dantzig_wolfe".

    Returns:
        float: The cost; NaN if the coordinated backend fell back to the MILP.
    """
    if backend == "cvxpy":
        return -float(opt.prob.value)
    return float(capture.result.upper) if capture.result is not None else float("nan")


# ------------------------------------------------------------ study cases
BATTERY = {"set_use_battery": True}
MPC = {"prediction_horizon": H, "soc_init": 0.5, "soc_final": 0.6}
HEAT_PUMP = {
    "thermal_battery": {
        "supply_temperature": 35.0,
        "volume": 20.0,
        "start_temperature": 22.0,
        "min_temperatures": [20.0] * H,
        "max_temperatures": [26.0] * H,
        "carnot_efficiency": 0.45,
        "u_value": 0.35,
        "envelope_area": 280.0,
        "ventilation_rate": 0.4,
        "heated_volume": 320.0,
        "thermal_inertia_time_constant": 2.0,
    }
}
HOT_WATER = {
    "thermal_config": {
        "heating_rate": 5.0,
        "cooling_constant": 0.02,
        "start_temperature": 45.0,
        "sense": "heat",
        "overshoot_temperature": 59.0,
        "desired_temperatures": [50.0] * 12 + [55.0] * 12 + [50.0] * 24,
    }
}
OUTDOOR = [float(8 + 5 * np.sin((i / 2 - 9) / 24 * 2 * np.pi)) for i in range(H)]

STUDY_CASES = [  # (name, action, optim_conf overrides, runtime parameters)
    (
        "Basic, no PV",
        "dayahead-optim",
        {
            "set_use_pv": False,
            "nominal_power_of_deferrable_loads": [3000, 750],
            "operating_hours_of_each_deferrable_load": [5, 8],
        },
        {},
    ),
    ("Basic, PV", "dayahead-optim", {"set_use_pv": True}, {}),
    ("PV and battery", "dayahead-optim", {"set_use_pv": True, **BATTERY}, {}),
    (
        "MPC, PV and battery",
        "naive-mpc-optim",
        {"set_use_pv": True, **BATTERY},
        {**MPC, "def_total_hours": [3, 5]},
    ),
    (
        "EV as a third on/off load (7.4 kW)",
        "naive-mpc-optim",
        {
            "set_use_pv": True,
            **BATTERY,
            "number_of_deferrable_loads": 3,
            "nominal_power_of_deferrable_loads": [3000, 750, 7400],
            "operating_hours_of_each_deferrable_load": [5, 8, 0],
            "start_timesteps_of_each_deferrable_load": [0, 0, 0],
            "end_timesteps_of_each_deferrable_load": [48, 48, 0],
        },
        {
            **MPC,
            "def_total_hours": [5, 8, 2.5],
            "end_timesteps_of_each_deferrable_load": [H, H, 14],
        },
    ),
    (
        "Heat pump (thermal_battery)",
        "naive-mpc-optim",
        {
            "set_use_pv": True,
            **BATTERY,
            "nominal_power_of_deferrable_loads": [2000, 3000],
            "operating_hours_of_each_deferrable_load": [2, 0],
            "treat_deferrable_load_as_semi_cont": [True, False],
        },
        {
            **MPC,
            "def_total_hours": [2, 0],
            "def_load_config": [{}, HEAT_PUMP],
            "outdoor_temperature_forecast": OUTDOOR,
        },
    ),
    (
        "Hot water (thermal_config)",
        "naive-mpc-optim",
        {
            "set_use_pv": True,
            **BATTERY,
            "nominal_power_of_deferrable_loads": [0, 2500],
            "operating_hours_of_each_deferrable_load": [0, 0],
            "treat_deferrable_load_as_semi_cont": [False, False],
        },
        {**MPC, "def_load_config": [{}, HOT_WATER]},
    ),
]


async def study_case(optim: dict, action: str, runtime: dict, backend: str, gap: float):
    """Run one study case through EMHASS's own action, as a dry run.

    Args:
        optim: optim_conf overrides for the case.
        action: "dayahead-optim" or "naive-mpc-optim".
        runtime: Runtime parameters for the action.
        backend: "cvxpy" or "dantzig_wolfe".
        gap: The MILP's lp_solver_mip_rel_gap.

    Returns:
        tuple: (cost, seconds).
    """
    params = await _params()
    params["optim_conf"].update(optim, optimization_backend=backend, lp_solver_mip_rel_gap=gap)
    runtime = {**runtime, "dry_run": True}
    params["passed_data"] = runtime
    with _Capture() as capture:
        t0 = time.perf_counter()
        idd = await set_input_data_dict(
            EMHASS_CONF,
            "profit",
            orjson.dumps(params).decode(),
            orjson.dumps(runtime).decode(),
            action,
            LOGGER,
            get_data_from_file=True,
        )
        run = dayahead_forecast_optim if action == "dayahead-optim" else naive_mpc_optim
        await run(idd, LOGGER)
        seconds = time.perf_counter() - t0
    return _objective(idd["opt"], capture, backend), seconds


# ---------------------------------------------------------- synthetic days
def synthetic_day(step_min: int, tariff: str, nodischarge: bool, heat_pump: bool, gap: float):
    """One synthetic day's inputs.

    Args:
        step_min: Time step, minutes.
        tariff: "dynamic", "day_night" or "flat".
        nodischarge: set_nodischarge_to_grid.
        heat_pump: Replace the second load with a heat pump (thermal_config).
        gap: The MILP's lp_solver_mip_rel_gap.

    Returns:
        tuple: (retrieve_hass_conf, optim_conf, plant_conf, data, pv W, load W,
        buy, sell) with buy and sell in currency/kWh.
    """
    rh, oc, pc = get_yaml_parse(orjson.dumps(asyncio.run(_params())).decode(), LOGGER)
    rh["optimization_time_step"] = pd.Timedelta(minutes=step_min)
    oc.update(
        set_use_battery=True,
        set_use_pv=True,
        set_nodischarge_to_grid=nodischarge,
        operating_hours_of_each_deferrable_load=[4, 2],
        lp_solver_mip_rel_gap=gap,
    )
    pc.update(
        battery_nominal_energy_capacity=10000,
        battery_charge_power_max=5000,
        battery_discharge_power_max=5000,
    )
    n = int(pd.Timedelta(days=1) / rh["optimization_time_step"])
    index = pd.date_range(
        "2026-10-01", periods=n, freq=rh["optimization_time_step"], tz=rh["time_zone"]
    )
    h = np.arange(n) * step_min / 60
    if tariff == "dynamic":
        buy = 0.20 + 0.10 * np.sin((h - 13) / 24 * 2 * np.pi) + 0.08 * (np.abs(h - 19) < 2)
    elif tariff == "day_night":
        buy = np.where((h >= 7) & (h < 23), 0.30, 0.15)
    else:
        buy = np.full(n, 0.25)
    sell = np.full(n, 0.06)
    pv = np.clip(5000 * np.sin((h - 6) / 12 * np.pi), 0, None)
    load = (
        400
        + 1200 * np.exp(-0.5 * ((h - 19.5) / 1.8) ** 2)
        + 600 * np.exp(-0.5 * ((h - 7.5) / 1.2) ** 2)
    )
    data = pd.DataFrame({"unit_load_cost": buy, "unit_prod_price": sell}, index=index)
    if heat_pump:
        data["outdoor_temperature_forecast"] = 8 + 5 * np.sin((h - 9) / 24 * 2 * np.pi)
        oc.update(
            nominal_power_of_deferrable_loads=[3000.0, 2000.0],
            operating_hours_of_each_deferrable_load=[4, 0],
            treat_deferrable_load_as_semi_cont=[True, False],
            def_load_config=[
                {},
                {
                    "thermal_config": {
                        "heating_rate": 2.0,
                        "cooling_constant": 0.1,
                        "start_temperature": 20.5,
                        "min_temperatures": [20.0] * n,
                        "max_temperatures": [23.0] * n,
                    }
                },
            ],
        )
    return rh, oc, pc, data, pv, load, buy, sell


def solve_day(inputs, backend: str):
    """Solve one synthetic day with `backend`; returns (cost, seconds)."""
    rh, oc, pc, data, pv, load, buy, sell = inputs
    oc = {**copy.deepcopy(oc), "optimization_backend": backend}
    opt = Optimization(
        rh,
        oc,
        copy.deepcopy(pc),
        "unit_load_cost",
        "unit_prod_price",
        "profit",
        EMHASS_CONF,
        LOGGER,
    )
    with _Capture() as capture:
        t0 = time.perf_counter()
        opt.perform_optimization(data, pv, load, buy, sell, soc_init=0.5, soc_final=0.5)
        seconds = time.perf_counter() - t0
    return _objective(opt, capture, backend), seconds


# ------------------------------------------------------------------ main
def _row(name, milp, coord):
    diff = abs(coord[0] - milp[0]) / max(abs(milp[0]), 1e-9) * 100
    print(
        f"  {name:42s} {milp[0]:11.6f} {coord[0]:11.6f} {diff:9.4f}%   {coord[1]:5.2f} / {milp[1]:5.2f} s"
    )
    return diff, coord[1], milp[1]


def _summary(label, rows):
    diffs, tc, tm = zip(*rows, strict=True)
    print(
        f"{label}: {len(rows)} runs, cost difference mean {statistics.mean(diffs):.4f}% / largest "
        f"{max(diffs):.4f}%, time coordinated {min(tc):.2f}-{max(tc):.2f} s / MILP {min(tm):.2f}-{max(tm):.2f} s\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mip-gap", type=float, default=0.0, help="the MILP's lp_solver_mip_rel_gap (default 0)"
    )
    gap = parser.parse_args().mip_gap
    header = (
        f"  {'run':42s} {'MILP':>11s} {'coordinated':>11s} {'|diff|':>10s}   time coord. / MILP"
    )

    print(f"EMHASS study cases (lp_solver_mip_rel_gap {gap})\n{header}")
    rows = []
    for name, action, optim, runtime in STUDY_CASES:
        milp = asyncio.run(study_case(optim, action, runtime, "cvxpy", gap))
        coord = asyncio.run(study_case(optim, action, runtime, "dantzig_wolfe", gap))
        rows.append(_row(name, milp, coord))
    _summary("Study cases", rows)

    for label, heat_pump, nodischarge_options in (
        ("battery and two on/off loads", False, (False, True)),
        ("battery, an on/off load and a heat pump", True, (True,)),
    ):
        print(f"Synthetic days: {label}\n{header}")
        rows = []
        for step in (30, 15):
            for tariff in ("dynamic", "day_night", "flat"):
                for nodischarge in nodischarge_options:
                    inputs = synthetic_day(step, tariff, nodischarge, heat_pump, gap)
                    name = f"{step} min, {tariff}" + (
                        ", no discharge to grid" if nodischarge else ""
                    )
                    rows.append(
                        _row(name, solve_day(inputs, "cvxpy"), solve_day(inputs, "dantzig_wolfe"))
                    )
        _summary(f"Synthetic days, {label}", rows)


if __name__ == "__main__":
    main()
