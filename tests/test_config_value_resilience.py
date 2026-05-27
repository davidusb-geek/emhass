"""
Parametrised resilience tests: stringly-typed and null config values.

Extends the #882 prevention suite with runtime-path coverage: for every
per-load-array parameter derived from param_definitions.json, inject bad
runtime values (scalar "null", None element, string element) via
treat_runtimeparams and assert the optimizer survives without raising.

Design:
- Param inventory loaded from param_definitions.json at collection time, so
  future array-typed params are covered automatically without editing this file.
- Only optim_conf params are exercised (associations.csv used to determine
  category); retrieve_hass_conf / plant_conf params aren't accessed during
  perform_optimization with defaults.
- Guarded params (cost_forecast_per_deferrable_load) MUST pass — these lock
  in the shipped fix as a class-level regression guard.
- Unguarded bugs are xfail(strict=True): CI stays green, the gap is visible,
  and the test flips to a live regression guard when the fix lands.

AGENTS.md: test-only PR, no prod changes. Follows Section 7 commit convention.
"""

import asyncio
import csv
import json
import pathlib
import unittest

import numpy as np
import orjson
import pandas as pd
import pytest

from emhass import utils
from emhass.optimization import Optimization

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "defaults_path": root / "src/emhass/data/config_defaults.json",
    "associations_path": root / "src/emhass/data/associations.csv",
}
logger, _ = utils.get_logger(__name__, emhass_conf, save_to_file=False)

_VAR_LOAD_COST = "unit_load_cost"
_VAR_PROD_PRICE = "unit_prod_price"

# ─────────────────────────── param inventory ────────────────────────────────


def _optim_array_param_names() -> list[str]:
    """Return all optim_conf params whose 'input' starts with 'array.'.

    Reads param_definitions.json (schema source of truth) and associations.csv
    (config-category mapping) so the inventory is derived, not hardcoded.
    """
    pd_path = root / "src/emhass/static/data/param_definitions.json"
    pd_data = json.loads(pd_path.read_text(encoding="utf-8"))

    optim_conf_params: set[str] = set()
    with open(emhass_conf["associations_path"], newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 3 and row[0] == "optim_conf":
                optim_conf_params.add(row[2])

    result: list[str] = []
    for section_entries in pd_data.values():
        for name, defn in section_entries.items():
            if defn.get("input", "").startswith("array.") and name in optim_conf_params:
                result.append(name)
    return result


_OPTIM_ARRAY_PARAMS: list[str] = _optim_array_param_names()

# Params whose guard is already merged — tests MUST pass and act as regression
# guards.  Add to this set when a new guard lands.
_GUARDED: frozenset[str] = frozenset({"cost_forecast_per_deferrable_load"})

# Params known to crash because no stringly-typed guard exists yet.
# Key = param name, value = reason string for xfail (include issue ref).
# When a guard lands, remove the entry; the test becomes a live regression guard.
_XFAIL_REASON: dict[str, str] = {
    "nominal_power_of_deferrable_loads": (
        "no stringly-typed guard; crashes at CVXPY constraint build (see #900)"
    ),
    "minimum_power_of_deferrable_loads": (
        "no stringly-typed guard; pad_list fails on non-list value (see #900)"
    ),
    "operating_hours_of_each_deferrable_load": (
        "no stringly-typed guard; pad_list fails on non-list value (see #900)"
    ),
    "set_deferrable_startup_penalty": (
        "no stringly-typed guard; string char used as float in penalty (see #900)"
    ),
    "set_deferrable_max_startups": (
        "no stringly-typed guard; string char used as int for max_starts (see #900)"
    ),
}

# Per-param, per-bad-value overrides for the xfail marker.
# Maps (param_name, value_id) — if present, this specific (param, value) combo
# is NOT marked xfail even if param is in _XFAIL_REASON; a partial guard exists
# that handles this specific bad-value shape.
_XFAIL_EXCLUDE: frozenset[tuple[str, str]] = frozenset(
    {
        # start timesteps: validate_def_timewindow(0.5, 0) produces start>end →
        # slice [def_start:def_end] is never reached.  None element handled by
        # the [s if s is not None else 0] guard (opt.py:2868).
        ("start_timesteps_of_each_deferrable_load", "none_element"),
        # set_deferrable_max_startups: `if max_starts and max_starts > 0:` guard
        # short-circuits on None (falsy) and 0.5 only appears as a CVXPY float
        # bound which is valid.  No crash for the [None, 0.5] shape.
        ("set_deferrable_max_startups", "none_element"),
    }
)

# Bad values to inject; (id_suffix, value) pairs.
_BAD_VALUES: list[tuple[str, object]] = [
    ("scalar_null", "null"),
    ("none_element", [None, 0.5]),
    ("str_element", ["0.0", "0.0"]),
]

# start/end timesteps crash with scalar_null and str_element (no guard there)
_XFAIL_REASON["start_timesteps_of_each_deferrable_load"] = (
    "no stringly-typed guard for scalar/string values; "
    "None-element path is guarded but scalar-null/string-element crash (see #900)"
)
_XFAIL_REASON["end_timesteps_of_each_deferrable_load"] = (
    "no stringly-typed guard for scalar/string values; "
    "None-element path is guarded but scalar-null/string-element crash (see #900)"
)


def _make_cases() -> list:
    cases = []
    for param in _OPTIM_ARRAY_PARAMS:
        reason = _XFAIL_REASON.get(param)
        for val_id, bad_val in _BAD_VALUES:
            marks: list = []
            if reason and (param, val_id) not in _XFAIL_EXCLUDE:
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            cases.append(
                pytest.param(param, bad_val, id=f"{param}-{val_id}", marks=marks)
            )
    return cases


_CASES: list = _make_cases()

# ──────────────────────── module-level default params ───────────────────────


async def _build_default_params() -> str:
    config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
    _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
    params = await utils.build_params(emhass_conf, secrets, config, logger)
    return orjson.dumps(params).decode("utf-8")


# Built once at import; reused across all test cases for speed.
_PARAMS_JSON: str = asyncio.run(_build_default_params())
_RHCONF, _OPTCONF, _PCONF = utils.get_yaml_parse(_PARAMS_JSON, logger)

# ─────────────────────── parametrised resilience tests ──────────────────────


def _make_forecast_inputs(rh_conf: dict, n: int = 48):
    freq = rh_conf["optimization_time_step"]
    tz = rh_conf["time_zone"]
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz=tz)
    ulc = np.full(n, 0.2)
    upp = np.full(n, 0.1)
    df = pd.DataFrame({_VAR_LOAD_COST: ulc, _VAR_PROD_PRICE: upp}, index=idx)
    return df, np.zeros(n), np.full(n, 1000.0), ulc, upp


@pytest.mark.parametrize("param_name,bad_value", _CASES)
def test_stringly_typed_resilience(param_name: str, bad_value: object):
    """Injecting bad runtime values for param_name must not raise.

    pytest.mark.parametrize is not supported on unittest.TestCase methods, so
    this test lives as a module-level function.  The async treat_runtimeparams
    call is wrapped in asyncio.run() which is fine outside an active event loop.
    """
    runtimeparams_json = orjson.dumps({param_name: bad_value}).decode("utf-8")

    async def _run():
        return await utils.treat_runtimeparams(
            runtimeparams_json,
            _PARAMS_JSON,
            _RHCONF,
            _OPTCONF,
            _PCONF,
            "dayahead-optim",
            logger,
            emhass_conf,
        )

    _, rh_conf, opt_conf, pl_conf = asyncio.run(_run())

    df, p_pv, p_load, ulc, upp = _make_forecast_inputs(rh_conf)
    opt = Optimization(
        rh_conf,
        opt_conf,
        pl_conf,
        _VAR_LOAD_COST,
        _VAR_PROD_PRICE,
        "profit",
        emhass_conf,
        logger,
    )
    opt.perform_optimization(df, p_pv, p_load, ulc, upp)


# ─────────────────────── thermal sense fixture ──────────────────────────────


class TestThermalSenseNullSmoke(unittest.IsolatedAsyncioTestCase):
    """sense: null in def_load_config must not crash (#898 unfixed)."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "sense: null crashes normalize_heat_cool_mode via str(None) = 'none' "
            "which isn't 'heat'/'cool'; fix pending in #898"
        ),
    )
    async def test_sense_null_in_thermal_config_does_not_crash(self):
        """Thermal load with sense=null must not raise ValueError."""
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
        params = await utils.build_params(emhass_conf, secrets, config, logger)

        # Inject a thermal_battery load with sense explicitly null — the exact
        # shape that the HA add-on UI persists for an unset optional field.
        params["optim_conf"]["number_of_deferrable_loads"] = 1
        params["optim_conf"]["nominal_power_of_deferrable_loads"] = [1000.0]
        params["optim_conf"]["minimum_power_of_deferrable_loads"] = [0.0]
        params["optim_conf"]["treat_deferrable_load_as_semi_cont"] = [True]
        params["optim_conf"]["operating_hours_of_each_deferrable_load"] = [4]
        params["optim_conf"]["set_deferrable_load_single_constant"] = [False]
        params["optim_conf"]["set_deferrable_startup_penalty"] = [0.0]
        params["optim_conf"]["set_deferrable_max_startups"] = [0]
        params["optim_conf"]["start_timesteps_of_each_deferrable_load"] = [0]
        params["optim_conf"]["end_timesteps_of_each_deferrable_load"] = [0]
        params["optim_conf"]["def_load_config"] = [
            {
                "thermal_battery": {
                    "sense": None,  # null — the crash trigger for #898
                    "nominal_thermal_power": 3000.0,
                    "thermal_inertia": 50.0,
                    "u_value": 0.05,
                    "initial_temperature": 20.0,
                    "desired_temperatures": [21.0] * 48,
                    "outdoor_temperature_forecast": [10.0] * 48,
                }
            }
        ]

        params_json = orjson.dumps(params).decode("utf-8")
        rh_conf, opt_conf, pl_conf = utils.get_yaml_parse(params_json, logger)  # type: ignore[misc]

        n = 48
        freq = rh_conf["optimization_time_step"]
        tz = rh_conf["time_zone"]
        idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz=tz)
        ulc = np.full(n, 0.2)
        upp = np.full(n, 0.1)
        df = pd.DataFrame({_VAR_LOAD_COST: ulc, _VAR_PROD_PRICE: upp}, index=idx)

        opt = Optimization(
            rh_conf,
            opt_conf,
            pl_conf,
            _VAR_LOAD_COST,
            _VAR_PROD_PRICE,
            "profit",
            emhass_conf,
            logger,
        )
        opt.perform_optimization(df, np.zeros(n), np.full(n, 1000.0), ulc, upp)
