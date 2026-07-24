"""
Multi-battery config plumbing tests (#610).

Covers only the config-layer normalisation: number_of_batteries, the per-battery
array broadcast/validate helper (check_batt_params), the weight_battery_* nested
disambiguation (check_batt_weight_params), and the soc_init/soc_final runtime
fallback chain. The optimization.py per-battery model and the publish path are
out of scope here.

Base-safety: the normaliser functions do not exist on master yet, so every test that
exercises them resolves the callable via getattr() and asserts it is present before
calling it. This keeps a RED run before implementation a clean, descriptive
AssertionError ("check_batt_params not implemented yet") rather than a raw
AttributeError - the behavioural gap IS that the function is missing.
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

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "defaults_path": root / "src/emhass/data/config_defaults.json",
    "associations_path": root / "src/emhass/data/associations.csv",
}
logger, _ = utils.get_logger(__name__, emhass_conf, save_to_file=False)


def _get_func(name: str):
    func = getattr(utils, name, None)
    assert func is not None, f"utils.{name} not implemented yet (#610)"
    return func


def _default_config() -> dict:
    return json.loads(emhass_conf["defaults_path"].read_text(encoding="utf-8"))


async def _build_params(overrides: dict | None = None) -> dict:
    config = _default_config()
    if overrides:
        config.update(overrides)
    _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
    params = await utils.build_params(emhass_conf, secrets, config, logger)
    assert params is not False, "build_params failed (see logged error)"
    return params


def build_params(overrides: dict | None = None) -> dict:
    return asyncio.run(_build_params(overrides))


async def _treat_runtime(runtimeparams: dict, base_params: dict, set_type="dayahead-optim"):
    params_json = orjson.dumps(base_params).decode("utf-8")
    rh_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
    rp_json = orjson.dumps(runtimeparams).decode("utf-8")
    return await utils.treat_runtimeparams(
        rp_json,
        params_json,
        rh_conf,
        optim_conf,
        plant_conf,
        set_type,
        logger,
        emhass_conf,
    )


def treat_runtime(runtimeparams: dict, base_params: dict, set_type="dayahead-optim"):
    _, rh_conf, optim_conf, plant_conf = asyncio.run(
        _treat_runtime(runtimeparams, base_params, set_type)
    )
    return rh_conf, optim_conf, plant_conf


# ─────────────────────── number_of_batteries plumbing ───────────────────────


def test_number_of_batteries_defaults_to_one():
    params = build_params()
    assert params["plant_conf"].get("number_of_batteries") == 1


def test_number_of_batteries_overridable_via_config():
    params = build_params({"number_of_batteries": 3})
    assert params["plant_conf"]["number_of_batteries"] == 3


# ──────────── per-battery array broadcast / validate (check_batt_params) ────────


# (parameter, default, home dict key)
_PLANT_CONF_ARRAY_PARAMS = [
    ("battery_discharge_power_max", 1000),
    ("battery_charge_power_max", 1000),
    ("battery_discharge_efficiency", 0.95),
    ("battery_charge_efficiency", 0.95),
    ("battery_nominal_energy_capacity", 5000),
    ("battery_minimum_state_of_charge", 0.3),
    ("battery_maximum_state_of_charge", 0.9),
    ("battery_target_state_of_charge", 0.6),
    ("battery_stress_cost", 0.0),
]

_OPTIM_CONF_ARRAY_PARAMS = [
    ("battery_soc_deficit_threshold", 0.4),
    ("battery_soc_deficit_cost", 0.0),
    ("battery_soc_surplus_threshold", 0.9),
    ("battery_soc_surplus_cost", 0.0),
]

_ALL_ARRAY_PARAMS = _PLANT_CONF_ARRAY_PARAMS + _OPTIM_CONF_ARRAY_PARAMS


@pytest.mark.parametrize("param_name,default", _ALL_ARRAY_PARAMS)
def test_check_batt_params_scalar_broadcasts_to_n(param_name, default):
    """A scalar value for a per-battery array param broadcasts to a length-N list for N>1."""
    check_batt_params = _get_func("check_batt_params")
    parameter = {param_name: 42.0}
    result = check_batt_params(3, parameter, default, param_name, logger)
    assert result == [42.0, 42.0, 42.0]
    assert parameter[param_name] == [42.0, 42.0, 42.0]


@pytest.mark.parametrize("param_name,default", _ALL_ARRAY_PARAMS)
def test_check_batt_params_exact_length_list_passthrough(param_name, default):
    """A list of exactly N entries passes through unchanged (values preserved)."""
    check_batt_params = _get_func("check_batt_params")
    values = [1.0, 2.0, 3.0]
    parameter = {param_name: list(values)}
    result = check_batt_params(3, parameter, default, param_name, logger)
    assert result == values


@pytest.mark.parametrize("param_name,default", _ALL_ARRAY_PARAMS)
def test_check_batt_params_wrong_length_raises(param_name, default):
    """A list whose length != number_of_batteries is a hard error (no silent pad)."""
    check_batt_params = _get_func("check_batt_params")
    parameter = {param_name: [1.0, 2.0]}
    with pytest.raises(ValueError) as excinfo:
        check_batt_params(3, parameter, default, param_name, logger)
    message = str(excinfo.value)
    assert param_name in message
    assert "3" in message


@pytest.mark.parametrize("param_name,default", _ALL_ARRAY_PARAMS)
def test_check_batt_params_missing_key_defaults_and_broadcasts(param_name, default):
    """A missing key gets filled with [default] * N when N > 1."""
    check_batt_params = _get_func("check_batt_params")
    parameter = {}
    result = check_batt_params(3, parameter, default, param_name, logger)
    assert result == [default, default, default]


@pytest.mark.parametrize("param_name,default", _ALL_ARRAY_PARAMS)
def test_check_batt_params_n1_is_true_noop(param_name, default):
    """N=1 (default/off) must be byte-identical to master: a plain scalar stays a
    plain scalar, never wrapped into a 1-element list (optimization.py's
    per-battery model reduces to reading that same scalar at index 0)."""
    check_batt_params = _get_func("check_batt_params")
    parameter = {param_name: 123.0}
    result = check_batt_params(1, parameter, default, param_name, logger)
    assert result == 123.0
    assert not isinstance(result, list)


@pytest.mark.parametrize("param_name,default", _ALL_ARRAY_PARAMS)
def test_check_batt_params_n1_wrong_length_list_still_raises(param_name, default):
    """Declaring number_of_batteries=1 but supplying a 2-entry array is still a
    hard error - this is exactly the shape the #901 resilience sweep injects, and
    is why these 6 new optim_conf arrays carry an _XFAIL_REASON entry there."""
    check_batt_params = _get_func("check_batt_params")
    parameter = {param_name: [1.0, 2.0]}
    with pytest.raises(ValueError):
        check_batt_params(1, parameter, default, param_name, logger)


# ───────── #901 bad shapes (scalar null, none-element, string-element) ─────────


@pytest.mark.parametrize("param_name,default", _OPTIM_CONF_ARRAY_PARAMS)
def test_check_batt_params_survives_stringly_null_scalar(param_name, default):
    """Bad shape 1 (#901): a stringly-typed 'null' scalar coerces to the default,
    then broadcasts."""
    check_batt_params = _get_func("check_batt_params")
    parameter = {param_name: "null"}
    result = check_batt_params(2, parameter, default, param_name, logger)
    assert result == [default, default]


@pytest.mark.parametrize("param_name,default", _OPTIM_CONF_ARRAY_PARAMS)
def test_check_batt_params_survives_none_element_when_length_matches(param_name, default):
    """Bad shape 2 (#901): a None element inside a correctly-sized list resolves to
    the per-slot default."""
    check_batt_params = _get_func("check_batt_params")
    parameter = {param_name: [None, 0.5]}
    result = check_batt_params(2, parameter, default, param_name, logger)
    assert result == [default, 0.5]


@pytest.mark.parametrize("param_name,default", _OPTIM_CONF_ARRAY_PARAMS)
def test_check_batt_params_survives_string_element_when_length_matches(param_name, default):
    """Bad shape 3 (#901): numeric-string elements inside a correctly-sized list
    coerce to float, or raise a clear ValueError if non-numeric."""
    check_batt_params = _get_func("check_batt_params")
    parameter = {param_name: ["0.0", "0.0"]}
    result = check_batt_params(2, parameter, default, param_name, logger)
    assert result == [0.0, 0.0]


def test_check_batt_params_non_numeric_string_element_raises_clear_error():
    """A truly non-numeric string element raises a clear ValueError, not a
    silent coercion to 0."""
    check_batt_params = _get_func("check_batt_params")
    parameter = {"battery_soc_deficit_cost": ["not_a_number", 0.1]}
    with pytest.raises(ValueError):
        check_batt_params(2, parameter, 0.0, "battery_soc_deficit_cost", logger)


# ────────────── weight_battery_* nested disambiguation rules ──────────────


@pytest.mark.parametrize("param_name", ["weight_battery_charge", "weight_battery_discharge"])
def test_weight_n1_is_byte_identical_noop(param_name):
    """N==1 -> exactly today's behaviour, no nesting, scalar or series as-is."""
    check_batt_weight_params = _get_func("check_batt_weight_params")
    parameter = {param_name: 2.0}
    check_batt_weight_params(1, parameter, param_name, logger)
    assert parameter[param_name] == 2.0

    series = [0.1, 0.2, 0.3]
    parameter2 = {param_name: list(series)}
    check_batt_weight_params(1, parameter2, param_name, logger)
    assert parameter2[param_name] == series


@pytest.mark.parametrize("param_name", ["weight_battery_charge", "weight_battery_discharge"])
def test_weight_n_gt_1_scalar_broadcasts_to_every_battery(param_name):
    """N>1, scalar value -> broadcast, each battery gets the same scalar."""
    check_batt_weight_params = _get_func("check_batt_weight_params")
    parameter = {param_name: 2.0}
    check_batt_weight_params(3, parameter, param_name, logger)
    assert parameter[param_name] == [2.0, 2.0, 2.0]


@pytest.mark.parametrize("param_name", ["weight_battery_charge", "weight_battery_discharge"])
def test_weight_n_gt_1_list_of_n_scalars_or_series_passes_through(param_name):
    """N>1, list of length N whose elements are scalars/lists -> already
    per-battery, entry k applies to battery k, passed through unchanged."""
    check_batt_weight_params = _get_func("check_batt_weight_params")
    nested = [1.0, [0.1, 0.2, 0.3]]
    parameter = {param_name: [list(x) if isinstance(x, list) else x for x in nested]}
    check_batt_weight_params(2, parameter, param_name, logger)
    assert parameter[param_name] == nested


@pytest.mark.parametrize("param_name", ["weight_battery_charge", "weight_battery_discharge"])
def test_weight_n_gt_1_flat_list_not_length_n_is_shared_series(param_name):
    """N>1, flat list whose length != N -> shared time series, same series
    applied to every battery (today's semantics preserved, just replicated per k)."""
    check_batt_weight_params = _get_func("check_batt_weight_params")
    series = [0.1, 0.2, 0.3, 0.4, 0.5]  # length 5, num_batteries=2 -> not length-N
    parameter = {param_name: list(series)}
    check_batt_weight_params(2, parameter, param_name, logger)
    assert parameter[param_name] == [series, series]


@pytest.mark.parametrize("param_name", ["weight_battery_charge", "weight_battery_discharge"])
def test_weight_ambiguous_flat_numeric_list_of_length_n_is_per_battery(param_name):
    """The documented ambiguous corner - a flat NUMERIC list whose length
    happens to equal number_of_batteries is treated as per-battery, NOT as a
    shared series of that length. Users who want a shared series of exactly N
    points must nest explicitly ([series] * N)."""
    check_batt_weight_params = _get_func("check_batt_weight_params")
    flat = [0.1, 0.2]  # length == num_batteries(2); could be misread as a series
    parameter = {param_name: list(flat)}
    check_batt_weight_params(2, parameter, param_name, logger)
    # Per-battery interpretation: battery 0 gets scalar 0.1, battery 1 gets 0.2 -
    # NOT [[0.1, 0.2], [0.1, 0.2]] (which would be the shared-series reading).
    assert parameter[param_name] == flat


# ────── weight_battery_* stringly-typed guard at ALL N ──────
#
# check_batt_weight_params previously opened with "if num_batteries == 1: return",
# a deliberate no-op for VALID shapes at N=1 - but it meant these two params got
# ZERO validation of any kind at N=1 (the default, and the only value every
# non-adopting user has). Confirmed live crash: set_use_battery=True + runtime
# {"weight_battery_charge": "null"} left the literal string 'null' in
# optim_conf, then perform_optimization crashed with "ValueError: could not
# convert string to float: np.str_('null')" inside
# _batt_weight_list -> np.array(weight_dis_k) (optimization.py). The fix adds
# a leaf-wise coercion pass (_coerce_batt_weight_value) that runs BEFORE the
# N==1 no-op check, so a bad leaf value is fixed at any N while the SHAPE
# (scalar stays scalar, series stays series) is untouched for already-valid
# values - the N=1 no-op for VALID inputs still holds.


@pytest.mark.parametrize("param_name", ["weight_battery_charge", "weight_battery_discharge"])
@pytest.mark.parametrize("num_batteries", [1, 2])
def test_weight_scalar_null_coerces_to_default(param_name, num_batteries):
    check_batt_weight_params = _get_func("check_batt_weight_params")
    parameter = {param_name: "null"}
    check_batt_weight_params(num_batteries, parameter, param_name, logger)
    expected = 0.0 if num_batteries == 1 else [0.0] * num_batteries
    assert parameter[param_name] == expected


@pytest.mark.parametrize("param_name", ["weight_battery_charge", "weight_battery_discharge"])
@pytest.mark.parametrize("num_batteries", [1, 2])
def test_weight_series_string_element_coerces_to_float(param_name, num_batteries):
    """A flat time series with numeric-string elements (e.g. from HA JSON)
    coerces every element to float; the series SHAPE is preserved at N=1 (no
    nesting), and shared at N>1 (flat series not of length N -> per-battery
    copy of the coerced series)."""
    check_batt_weight_params = _get_func("check_batt_weight_params")
    parameter = {param_name: ["0.1", "0.2", "0.3", "0.4", "0.5"]}  # length 5
    check_batt_weight_params(num_batteries, parameter, param_name, logger)
    coerced_series = [0.1, 0.2, 0.3, 0.4, 0.5]
    if num_batteries == 1:
        assert parameter[param_name] == coerced_series
    else:
        assert parameter[param_name] == [coerced_series] * num_batteries


@pytest.mark.parametrize("param_name", ["weight_battery_charge", "weight_battery_discharge"])
def test_weight_nested_bad_element_coerces_at_n1(param_name):
    """Even though N=1 never NESTS per-battery, the coercion recurses through
    whatever shape is present (defensive - a hand-built config could still
    contain a nested list) so a bad leaf anywhere is fixed, not just at the
    top level. Shape is fully preserved (list-of-lists stays list-of-lists)."""
    check_batt_weight_params = _get_func("check_batt_weight_params")
    parameter = {param_name: [["null", 0.2], ["0.3", 0.4]]}
    check_batt_weight_params(1, parameter, param_name, logger)
    assert parameter[param_name] == [[0.0, 0.2], [0.3, 0.4]]


@pytest.mark.parametrize("param_name", ["weight_battery_charge", "weight_battery_discharge"])
def test_weight_nested_bad_element_coerces_at_n2(param_name):
    """Nested per-battery form (list of length num_batteries, each entry a
    scalar or its own time series) with a bad element inside battery 1's
    series - only that leaf coerces, the nesting/shape is untouched."""
    check_batt_weight_params = _get_func("check_batt_weight_params")
    parameter = {param_name: [1.0, ["null", "0.2", 0.3]]}
    check_batt_weight_params(2, parameter, param_name, logger)
    assert parameter[param_name] == [1.0, [0.0, 0.2, 0.3]]


@pytest.mark.parametrize("param_name", ["weight_battery_charge", "weight_battery_discharge"])
@pytest.mark.parametrize("num_batteries", [1, 2])
def test_weight_non_numeric_string_raises_clear_error(param_name, num_batteries):
    """A genuinely non-numeric string element is the 'clear ValueError' branch,
    never a silent coercion to 0 - matches check_batt_params' contract."""
    check_batt_weight_params = _get_func("check_batt_weight_params")
    parameter = {param_name: "not_a_number"}
    with pytest.raises(ValueError, match=param_name):
        check_batt_weight_params(num_batteries, parameter, param_name, logger)


# ── End-to-end: set_use_battery=True is REQUIRED to exercise this ──
#
# The #901 resilience sweep's module-level fixture builds from
# config_defaults.json, whose set_use_battery default is False - so the
# objective-building code that ever touches weight_dis/weight_chg never runs,
# and the sweep's 6 parametrized weight_battery_* cases pass FOR THE WRONG
# REASON (feature off, not because a guard exists). These tests force
# set_use_battery=True and drive perform_optimization for real, closing that
# blind spot.


def test_weight_stringly_null_survives_real_optimization_n1():
    from emhass.optimization import Optimization

    base = build_params({"set_use_battery": True})
    rh_conf, optim_conf, plant_conf = treat_runtime(
        {"weight_battery_charge": "null", "weight_battery_discharge": "null"},
        base,
    )
    n = 8
    idx = pd.date_range("2024-01-01", periods=n, freq="30min", tz=rh_conf["time_zone"])
    df = pd.DataFrame({"unit_load_cost": [0.2] * n, "unit_prod_price": [0.1] * n}, index=idx)
    opt = Optimization(
        rh_conf,
        optim_conf,
        plant_conf,
        "unit_load_cost",
        "unit_prod_price",
        "profit",
        emhass_conf,
        logger,
    )
    # Must not raise (pre-fix: ValueError: could not convert string to float).
    opt.perform_optimization(
        df,
        np.zeros(n),
        np.full(n, 1000.0),
        df["unit_load_cost"].values,
        df["unit_prod_price"].values,
    )


def test_weight_stringly_null_survives_real_optimization_n2():
    from emhass.optimization import Optimization

    base = build_params({"set_use_battery": True, "number_of_batteries": 2})
    rh_conf, optim_conf, plant_conf = treat_runtime(
        {"weight_battery_charge": "null", "weight_battery_discharge": ["null", "0.05"]},
        base,
    )
    n = 8
    idx = pd.date_range("2024-01-01", periods=n, freq="30min", tz=rh_conf["time_zone"])
    df = pd.DataFrame({"unit_load_cost": [0.2] * n, "unit_prod_price": [0.1] * n}, index=idx)
    opt = Optimization(
        rh_conf,
        optim_conf,
        plant_conf,
        "unit_load_cost",
        "unit_prod_price",
        "profit",
        emhass_conf,
        logger,
    )
    opt.perform_optimization(
        df,
        np.zeros(n),
        np.full(n, 1000.0),
        df["unit_load_cost"].values,
        df["unit_prod_price"].values,
    )


# ─────────── N=1 no-op pin: built params identical to master's scalars ──────────


def test_n1_default_config_battery_values_identical_to_master():
    """No-op pin: with number_of_batteries at its default (1) and no per-battery
    arrays supplied, build_params must produce the exact same scalar values
    master does today - not lists, not broadcast copies."""
    params = build_params()
    plant_conf = params["plant_conf"]
    optim_conf = params["optim_conf"]

    expected_plant = {
        "battery_discharge_power_max": 1000,
        "battery_charge_power_max": 1000,
        "battery_discharge_efficiency": 0.95,
        "battery_charge_efficiency": 0.95,
        "battery_nominal_energy_capacity": 5000,
        "battery_minimum_state_of_charge": 0.3,
        "battery_maximum_state_of_charge": 0.9,
        "battery_target_state_of_charge": 0.6,
        "battery_stress_cost": 0.0,
    }
    for key, expected in expected_plant.items():
        assert plant_conf[key] == expected, f"{key}: {plant_conf[key]!r} != {expected!r}"
        assert not isinstance(plant_conf[key], list), f"{key} must stay a scalar at N=1"

    expected_optim = {
        "weight_battery_discharge": 0.0,
        "weight_battery_charge": 0.0,
        "battery_soc_deficit_threshold": 0.4,
        "battery_soc_deficit_cost": 0.0,
        "battery_soc_surplus_threshold": 0.9,
        "battery_soc_surplus_cost": 0.0,
    }
    for key, expected in expected_optim.items():
        assert optim_conf[key] == expected, f"{key}: {optim_conf[key]!r} != {expected!r}"
        assert not isinstance(optim_conf[key], list), f"{key} must stay a scalar at N=1"


def test_n_gt_1_default_config_battery_values_are_broadcast_lists():
    """Companion to the no-op pin: with number_of_batteries=3 and no per-battery
    arrays supplied, build_params broadcasts every default to a 3-element list."""
    params = build_params({"number_of_batteries": 3})
    plant_conf = params["plant_conf"]
    assert plant_conf["battery_nominal_energy_capacity"] == [5000, 5000, 5000]
    assert plant_conf["battery_minimum_state_of_charge"] == [0.3, 0.3, 0.3]
    assert params["optim_conf"]["battery_soc_deficit_threshold"] == [0.4, 0.4, 0.4]


# ─────────────────────────── build_params end-to-end ───────────────────────────


def test_build_params_broadcasts_explicit_scalar_for_n_gt_1():
    params = build_params({"number_of_batteries": 2, "battery_nominal_energy_capacity": 7500})
    assert params["plant_conf"]["battery_nominal_energy_capacity"] == [7500, 7500]


def test_build_params_accepts_exact_length_list_for_n_gt_1():
    params = build_params(
        {
            "number_of_batteries": 2,
            "battery_nominal_energy_capacity": [5000, 10000],
        }
    )
    assert params["plant_conf"]["battery_nominal_energy_capacity"] == [5000, 10000]


def test_build_params_wrong_length_list_raises():
    with pytest.raises(ValueError):
        build_params(
            {
                "number_of_batteries": 2,
                "battery_nominal_energy_capacity": [5000, 10000, 15000],
            }
        )


# ───────────────────── treat_runtimeparams array overrides ─────────────────


def test_treat_runtimeparams_scalar_override_broadcasts_for_n_gt_1():
    base = build_params({"number_of_batteries": 3})
    _, optim_conf, plant_conf = treat_runtime({"battery_charge_power_max": 2500}, base)
    assert plant_conf["battery_charge_power_max"] == [2500, 2500, 2500]


def test_treat_runtimeparams_exact_length_list_override_for_n_gt_1():
    base = build_params({"number_of_batteries": 3})
    _, optim_conf, plant_conf = treat_runtime(
        {"battery_charge_power_max": [1000, 2000, 3000]}, base
    )
    assert plant_conf["battery_charge_power_max"] == [1000, 2000, 3000]


def test_treat_runtimeparams_wrong_length_override_raises_for_n_gt_1():
    base = build_params({"number_of_batteries": 3})
    with pytest.raises(ValueError):
        treat_runtime({"battery_charge_power_max": [1000, 2000]}, base)


def test_treat_runtimeparams_n1_scalar_override_stays_scalar():
    """Runtime override at N=1 (default) must stay a scalar - no wrapping."""
    base = build_params()
    _, optim_conf, plant_conf = treat_runtime({"battery_charge_power_max": 4321}, base)
    assert plant_conf["battery_charge_power_max"] == 4321
    assert not isinstance(plant_conf["battery_charge_power_max"], list)


# ──────────── runtime scalar masking a configured per-battery list ──────────
# A runtime scalar overrides a configured value and broadcasts to every
# battery. When the configured value was a list with distinct per-battery
# entries, the broadcast silently flattens them (a real two-battery deployment
# lost its per-unit power limits this way, see PR #1032). The values stay
# exactly the documented override semantics; the only new behaviour is a
# warning making the mask visible.


def _mask_warnings(caplog, parameter_name: str) -> list:
    return [
        rec
        for rec in caplog.records
        if rec.levelname == "WARNING"
        and parameter_name in rec.message
        and "overrides the configured per-battery list" in rec.message
    ]


def test_runtime_scalar_over_distinct_config_list_warns(caplog):
    base = build_params({"number_of_batteries": 2, "battery_charge_power_max": [2500, 2400]})
    with caplog.at_level(logging.WARNING):
        _, optim_conf, plant_conf = treat_runtime({"battery_charge_power_max": 4200}, base)
    warnings = _mask_warnings(caplog, "battery_charge_power_max")
    assert len(warnings) == 1, "expected exactly one mask warning"
    assert "[2500, 2400]" in warnings[0].message
    assert "4200" in warnings[0].message
    # The override semantics themselves are unchanged: scalar still broadcasts.
    assert plant_conf["battery_charge_power_max"] == [4200, 4200]


def test_runtime_scalar_over_distinct_optim_conf_list_warns(caplog):
    base = build_params({"number_of_batteries": 2, "battery_soc_deficit_cost": [0.1, 0.3]})
    with caplog.at_level(logging.WARNING):
        _, optim_conf, plant_conf = treat_runtime({"battery_soc_deficit_cost": 0.2}, base)
    assert len(_mask_warnings(caplog, "battery_soc_deficit_cost")) == 1
    assert optim_conf["battery_soc_deficit_cost"] == [0.2, 0.2]


def test_runtime_list_over_distinct_config_list_no_warning(caplog):
    base = build_params({"number_of_batteries": 2, "battery_charge_power_max": [2500, 2400]})
    with caplog.at_level(logging.WARNING):
        _, optim_conf, plant_conf = treat_runtime({"battery_charge_power_max": [3000, 2900]}, base)
    assert _mask_warnings(caplog, "battery_charge_power_max") == []
    assert plant_conf["battery_charge_power_max"] == [3000, 2900]


def test_runtime_scalar_over_uniform_config_no_warning(caplog):
    """A scalar config broadcasts to a uniform list before treat_runtimeparams
    runs; a runtime scalar over it is exactly what the caller meant and MPC
    re-sends it every cycle - it must stay silent."""
    base = build_params({"number_of_batteries": 2, "battery_charge_power_max": 2500})
    with caplog.at_level(logging.WARNING):
        _, optim_conf, plant_conf = treat_runtime({"battery_charge_power_max": 4200}, base)
    assert _mask_warnings(caplog, "battery_charge_power_max") == []
    assert plant_conf["battery_charge_power_max"] == [4200, 4200]


def test_runtime_scalar_via_legacy_name_over_distinct_config_list_warns(caplog):
    """The association loop also applies overrides passed under the legacy
    parameter name (Pc_max -> battery_charge_power_max); detection reads the
    post-loop state, so the mask warning must fire for those too."""
    base = build_params({"number_of_batteries": 2, "battery_charge_power_max": [2500, 2400]})
    with caplog.at_level(logging.WARNING):
        _, optim_conf, plant_conf = treat_runtime({"Pc_max": 4200}, base)
    assert len(_mask_warnings(caplog, "battery_charge_power_max")) == 1
    assert plant_conf["battery_charge_power_max"] == [4200, 4200]


def test_runtime_scalar_n1_never_warns(caplog):
    base = build_params()
    with caplog.at_level(logging.WARNING):
        _, optim_conf, plant_conf = treat_runtime({"battery_charge_power_max": 4200}, base)
    assert _mask_warnings(caplog, "battery_charge_power_max") == []
    assert plant_conf["battery_charge_power_max"] == 4200


# ─────────────────────── soc_init / soc_final runtime ──────────────────────


def test_soc_init_soc_final_n1_stays_plain_float_backward_compat():
    """Existing test_utils.py pins (:250, :1397, :1425) require soc_init/soc_final
    to stay a plain float at N=1 - this is the same invariant, isolated here."""
    base = build_params()
    _, optim_conf, plant_conf = treat_runtime(
        {"soc_init": 0.5, "soc_final": 0.6, "prediction_horizon": 10},
        base,
        set_type="naive-mpc-optim",
    )
    # passed_data isn't returned by treat_runtime() above (it strips to
    # rh/optim/plant); re-run raw to inspect passed_data directly.
    params_json = orjson.dumps(base).decode("utf-8")
    rh_conf, oc, pc = utils.get_yaml_parse(params_json, logger)
    rp_json = orjson.dumps({"soc_init": 0.5, "soc_final": 0.6, "prediction_horizon": 10}).decode(
        "utf-8"
    )
    params_out, _, _, _ = asyncio.run(
        utils.treat_runtimeparams(
            rp_json, params_json, rh_conf, oc, pc, "naive-mpc-optim", logger, emhass_conf
        )
    )
    params_out = orjson.loads(params_out)
    assert params_out["passed_data"]["soc_init"] == 0.5
    assert params_out["passed_data"]["soc_final"] == 0.6
    assert not isinstance(params_out["passed_data"]["soc_init"], list)
    assert not isinstance(params_out["passed_data"]["soc_final"], list)


def _naive_mpc_passed_data(runtimeparams: dict, base_params: dict) -> dict:
    params_json = orjson.dumps(base_params).decode("utf-8")
    rh_conf, oc, pc = utils.get_yaml_parse(params_json, logger)
    rp_json = orjson.dumps(runtimeparams).decode("utf-8")
    params_out, _, _, _ = asyncio.run(
        utils.treat_runtimeparams(
            rp_json, params_json, rh_conf, oc, pc, "naive-mpc-optim", logger, emhass_conf
        )
    )
    return orjson.loads(params_out)["passed_data"]


# ── naive-mpc N=1 soc_init/soc_final list handling ──────────


def test_soc_init_n1_length_one_list_unwraps_to_scalar():
    """naive-mpc at N=1 must tolerate a length-1 runtime list the same way the
    dayahead branch already does (_passthrough_soc_runtime) - unwrap to the
    scalar, not crash with an opaque TypeError comparing a list to a float
    (the pre-fix repro: soc_init=[0.4] raised
    'TypeError: '<' not supported between instances of 'list' and 'float''
    at the battery_minimum_state_of_charge clamp check)."""
    base = build_params()  # number_of_batteries defaults to 1
    passed_data = _naive_mpc_passed_data(
        {"soc_init": [0.4], "soc_final": [0.5], "prediction_horizon": 10}, base
    )
    assert passed_data["soc_init"] == 0.4
    assert passed_data["soc_final"] == 0.5
    assert not isinstance(passed_data["soc_init"], list)
    assert not isinstance(passed_data["soc_final"], list)


def test_soc_init_n1_wrong_length_list_raises_clear_value_error_not_type_error():
    """A genuinely mis-shaped list at N=1 (e.g. 2 entries) must raise the same
    clear, parameter-naming ValueError the N>1 path raises via
    _resolve_soc_runtime_list - not the opaque TypeError from comparing a list
    to a float."""
    base = build_params()
    with pytest.raises(ValueError, match="soc_init"):
        _naive_mpc_passed_data({"soc_init": [0.4, 0.5], "prediction_horizon": 10}, base)


def test_soc_final_n1_wrong_length_list_raises_clear_value_error():
    base = build_params()
    with pytest.raises(ValueError, match="soc_final"):
        _naive_mpc_passed_data(
            {"soc_init": 0.4, "soc_final": [0.4, 0.5], "prediction_horizon": 10}, base
        )


def test_soc_init_n1_naive_mpc_dayahead_parity_on_length_one_list():
    """naive-mpc and dayahead must AGREE at N=1 given a length-1 runtime
    list - neither crashes, and both resolve to the same per-battery value
    once normalised by Optimization._normalize_soc_arg (dayahead's
    _passthrough_soc_runtime keeps the [0.4] list shape; naive-mpc unwraps to
    the bare 0.4 scalar per the N=1 no-op pin - both denote 'battery 0 =
    0.4')."""
    base = build_params()
    naive_passed = _naive_mpc_passed_data({"soc_init": [0.4]}, base)

    params_json = orjson.dumps(base).decode("utf-8")
    rh_conf, oc, pc = utils.get_yaml_parse(params_json, logger)
    rp_json = orjson.dumps({"soc_init": [0.4]}).decode("utf-8")
    params_out, _, _, _ = asyncio.run(
        utils.treat_runtimeparams(
            rp_json, params_json, rh_conf, oc, pc, "dayahead-optim", logger, emhass_conf
        )
    )
    dayahead_passed = orjson.loads(params_out)["passed_data"]
    assert dayahead_passed["soc_init"] == [0.4]

    from emhass.optimization import Optimization

    opt = Optimization(
        rh_conf, oc, pc, "unit_load_cost", "unit_prod_price", "profit", emhass_conf, logger
    )
    assert opt._normalize_soc_arg(naive_passed["soc_init"]) == [0.4]
    assert opt._normalize_soc_arg(dayahead_passed["soc_init"]) == [0.4]


def test_soc_init_scalar_broadcasts_for_n_gt_1():
    base = build_params({"number_of_batteries": 2})
    passed_data = _naive_mpc_passed_data(
        {"soc_init": 0.5, "soc_final": 0.6, "prediction_horizon": 10}, base
    )
    assert passed_data["soc_init"] == [0.5, 0.5]
    assert passed_data["soc_final"] == [0.6, 0.6]


def test_soc_init_exact_length_list_for_n_gt_1():
    base = build_params({"number_of_batteries": 2})
    passed_data = _naive_mpc_passed_data(
        {"soc_init": [0.4, 0.5], "soc_final": [0.5, 0.6], "prediction_horizon": 10}, base
    )
    assert passed_data["soc_init"] == [0.4, 0.5]
    assert passed_data["soc_final"] == [0.5, 0.6]


def test_soc_init_wrong_length_list_raises_for_n_gt_1():
    base = build_params({"number_of_batteries": 2})
    with pytest.raises(ValueError):
        _naive_mpc_passed_data({"soc_init": [0.4, 0.5, 0.6], "prediction_horizon": 10}, base)


def test_soc_final_clamps_per_battery_against_own_min_max():
    """soc_final clamps per battery k to [min[k], max[k]] (soc_init does not)."""
    base = build_params(
        {
            "number_of_batteries": 2,
            "battery_minimum_state_of_charge": [0.2, 0.3],
            "battery_maximum_state_of_charge": [0.8, 0.9],
        }
    )
    passed_data = _naive_mpc_passed_data(
        {"soc_final": [0.05, 0.95], "prediction_horizon": 10}, base
    )
    assert passed_data["soc_final"] == [0.2, 0.9]


def test_soc_init_unclamped_per_battery_but_warns():
    """soc_init stays deliberately unclamped (existing recovery semantics),
    per battery, even though it logs a warning for the out-of-band entry."""
    base = build_params(
        {
            "number_of_batteries": 2,
            "battery_minimum_state_of_charge": [0.2, 0.3],
            "battery_maximum_state_of_charge": [0.8, 0.9],
        }
    )
    passed_data = _naive_mpc_passed_data({"soc_init": [0.05, 0.05], "prediction_horizon": 10}, base)
    assert passed_data["soc_init"] == [0.05, 0.05]


# --- Config-layer edge-case fix regression tests ---


@pytest.mark.parametrize("bad_count", [0, -2, 1.5, True, "abc", None, [2]])
def test_validate_num_batteries_rejects_invalid_counts(bad_count):
    func = _get_func("validate_num_batteries")
    with pytest.raises(ValueError, match="number_of_batteries"):
        func({"number_of_batteries": bad_count})


@pytest.mark.parametrize(("raw", "expected"), [(1, 1), (3, 3), (2.0, 2), ("2", 2)])
def test_validate_num_batteries_accepts_and_coerces(raw, expected):
    func = _get_func("validate_num_batteries")
    assert func({"number_of_batteries": raw}) == expected


def test_validate_num_batteries_missing_key_defaults_to_one():
    func = _get_func("validate_num_batteries")
    assert func({}) == 1


def test_build_params_rejects_zero_batteries():
    with pytest.raises(ValueError, match="number_of_batteries"):
        build_params({"number_of_batteries": 0})


def test_weight_shared_series_entries_are_independent_copies():
    func = _get_func("check_batt_weight_params")
    conf = {"weight_battery_charge": [0.1, 0.2, 0.3]}  # flat series, N=2 -> shared
    func(2, conf, "weight_battery_charge", logger)
    entries = conf["weight_battery_charge"]
    assert len(entries) == 2
    assert entries[0] == entries[1]
    assert entries[0] is not entries[1], "per-battery series must not alias one object"
    entries[0][0] = 99.0
    assert entries[1][0] == 0.1, "mutating one battery's series leaked into its sibling"
