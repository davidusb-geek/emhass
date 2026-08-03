"""
Deferrable-load per-load array re-normalisation after runtime params (#1040).

The association loop inside treat_runtimeparams (utils.py ~1430) copies a raw
runtime value into optim_conf for every associations.csv row, modern or
legacy name, with no length validation. Battery arrays got a post-loop
re-normalisation for this (#610, ~1469); deferrable arrays never did, so a
stale short array in a runtime payload survived treat_runtimeparams and blew
up as an IndexError deep inside optimization.py's
_add_deferrable_load_constraints (whichever per-load array is read first).

This file proves the fix: after treat_runtimeparams returns, every per-load
array in utils.DEF_LOAD_ARRAY_PARAMS is padded to the final
number_of_deferrable_loads, a runtime-provided short array gets a visible
warning (config-sourced padding stays silent/debug per #929), and correctly
sized arrays pass through unchanged with no warning.

Base-safety: DEF_LOAD_ARRAY_PARAMS does not exist on master yet, so
DEF_ARRAY_NAMES below is a hardcoded literal (not read off
utils.DEF_LOAD_ARRAY_PARAMS) and tests assert on LENGTH/VALUE behaviour, not
on the new symbol existing - so a RED run on base fails at the behavioural
assertion, not an ImportError/AttributeError.
"""

import asyncio
import json
import logging
import pathlib

import orjson

from emhass import utils

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "defaults_path": root / "src/emhass/data/config_defaults.json",
    "associations_path": root / "src/emhass/data/associations.csv",
}
logger, _ = utils.get_logger(__name__, emhass_conf, save_to_file=False)

# The 9 arrays build_params normalises via check_def_loads (utils.py, #929).
# Hardcoded (not read from utils.DEF_LOAD_ARRAY_PARAMS) so the RED-on-base
# tests still know what to check for when that table doesn't exist yet.
DEF_ARRAY_NAMES = [
    "start_timesteps_of_each_deferrable_load",
    "end_timesteps_of_each_deferrable_load",
    "set_deferrable_load_single_constant",
    "treat_deferrable_load_as_semi_cont",
    "set_deferrable_startup_penalty",
    "deferrable_load_max_cost",
    "set_deferrable_max_startups",
    "operating_hours_of_each_deferrable_load",
    "nominal_power_of_deferrable_loads",
]


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


def _short_warnings(caplog, parameter_name: str) -> list:
    return [
        rec
        for rec in caplog.records
        if rec.levelname == "WARNING"
        and parameter_name in rec.message
        and "padded from" in rec.message
    ]


# ─────────────────────── RED-on-base: length after renormalise ─────────────


def test_short_deferrable_load_max_cost_gets_padded_to_final_count():
    """Reporter's exact repro shape: runtime bumps the load count to 3 and
    supplies a stale 2-element deferrable_load_max_cost. Every table array
    must come out length 3 after treat_runtimeparams, not just the one the
    caller happened to also touch (the crash moves to whichever key
    optimization.py reads first)."""
    base = build_params()
    _, optim_conf, _ = treat_runtime(
        {
            "number_of_deferrable_loads": 3,
            "deferrable_load_max_cost": [0, 0],
        },
        base,
    )
    assert optim_conf.get("number_of_deferrable_loads") == 3
    for name in DEF_ARRAY_NAMES:
        value = optim_conf.get(name)
        assert isinstance(value, list), f"{name} should be a list, got {value!r}"
        assert len(value) == 3, f"{name} has {len(value)} entries, expected 3"


def test_short_set_deferrable_load_single_constant_gets_padded_to_final_count():
    """Second variant from the reporter: the short array is a different key
    (set_deferrable_load_single_constant), proving the fix isn't keyed to one
    specific param name."""
    base = build_params()
    _, optim_conf, _ = treat_runtime(
        {
            "number_of_deferrable_loads": 3,
            "set_deferrable_load_single_constant": [False],
        },
        base,
    )
    assert optim_conf.get("number_of_deferrable_loads") == 3
    for name in DEF_ARRAY_NAMES:
        value = optim_conf.get(name)
        assert isinstance(value, list), f"{name} should be a list, got {value!r}"
        assert len(value) == 3, f"{name} has {len(value)} entries, expected 3"


# ─────────────────────────── counterfactual passthrough ────────────────────


def test_correctly_sized_runtime_arrays_pass_through_unchanged(caplog):
    """All 9 arrays supplied already sized to the final count: values must be
    unchanged and no padding/no warning should fire."""
    base = build_params()
    runtimeparams = {
        "number_of_deferrable_loads": 3,
        "start_timesteps_of_each_deferrable_load": [0, 1, 2],
        "end_timesteps_of_each_deferrable_load": [10, 11, 12],
        "set_deferrable_load_single_constant": [False, True, False],
        "treat_deferrable_load_as_semi_cont": [True, False, True],
        "set_deferrable_startup_penalty": [0.0, 0.1, 0.2],
        "deferrable_load_max_cost": [0.0, 1.5, 3.0],
        "set_deferrable_max_startups": [0, 2, 4],
        "operating_hours_of_each_deferrable_load": [1, 2, 3],
        "nominal_power_of_deferrable_loads": [1000, 2000, 3000],
    }
    with caplog.at_level(logging.WARNING):
        _, optim_conf, _ = treat_runtime(runtimeparams, base)
    for name in DEF_ARRAY_NAMES:
        assert optim_conf.get(name) == runtimeparams[name], (
            f"{name}: expected unchanged passthrough of {runtimeparams[name]!r}, "
            f"got {optim_conf.get(name)!r}"
        )
        assert _short_warnings(caplog, name) == [], f"{name} should not have warned"


def test_absent_array_pads_silently_config_sourced(caplog):
    """An array absent from BOTH config and runtime is config-sourced padding
    (#929) - debug only, never the new runtime-short warning."""
    base = build_params()
    with caplog.at_level(logging.WARNING):
        _, optim_conf, _ = treat_runtime({"number_of_deferrable_loads": 3}, base)
    assert len(optim_conf.get("deferrable_load_max_cost", [])) == 3
    assert _short_warnings(caplog, "deferrable_load_max_cost") == []


def test_explicit_null_array_pads_silently_from_config(caplog):
    """The association loop skips a JSON null runtime value (it only copies
    when the runtime value is not None), so an explicit null never reaches
    optim_conf - the array that gets padded is the CONFIG one, not a
    runtime-provided one. Zero re-normalisation warnings, same as if the key
    were absent (the deferrable_load_max_cost-specific check mirrors the
    absent-array test above; a bare-message scan guards against a
    differently-worded regression escaping that filter)."""
    base = build_params()
    with caplog.at_level(logging.WARNING):
        _, optim_conf, _ = treat_runtime(
            {"number_of_deferrable_loads": 3, "deferrable_load_max_cost": None},
            base,
        )
    assert len(optim_conf.get("deferrable_load_max_cost", [])) == 3
    assert _short_warnings(caplog, "deferrable_load_max_cost") == []
    renorm_warnings = [
        rec.message
        for rec in caplog.records
        if rec.levelname == "WARNING" and "deferrable_load_max_cost" in rec.message
    ]
    assert renorm_warnings == [], f"expected zero warnings, got {renorm_warnings}"


# ─────────────────────────── legacy-name variant ────────────────────────────


def test_short_array_via_legacy_name_warns_and_pads(caplog):
    """The association loop also applies overrides passed under the legacy
    parameter name (P_deferrable_nom -> nominal_power_of_deferrable_loads,
    src/emhass/data/associations.csv row 48); the short-array warning and
    padding must fire for those too."""
    base = build_params()
    with caplog.at_level(logging.WARNING):
        _, optim_conf, _ = treat_runtime(
            {
                "number_of_deferrable_loads": 3,
                "P_deferrable_nom": [1000.0, 2000.0],
            },
            base,
        )
    assert optim_conf.get("nominal_power_of_deferrable_loads") == [1000.0, 2000.0, 0]
    warnings = _short_warnings(caplog, "nominal_power_of_deferrable_loads")
    assert len(warnings) == 1, "expected exactly one runtime-short warning"


# ────────────────────── def_load_config interaction ────────────────────────


def test_def_load_config_reset_then_short_array_still_normalised():
    """The def_load_config runtime handling resets number_of_deferrable_loads
    AFTER the association loop has already applied a short
    deferrable_load_max_cost; the final normalisation must use the
    POST-reset count, pinning the fix's placement after that reset."""
    base = build_params()
    _, optim_conf, _ = treat_runtime(
        {
            "deferrable_load_max_cost": [0, 0],
            "def_load_config": [{}, {}, {}],
        },
        base,
    )
    assert optim_conf.get("number_of_deferrable_loads") == 3
    assert len(optim_conf.get("deferrable_load_max_cost", [])) == 3


# ────────────────────────────── string count ────────────────────────────────


def test_string_number_of_deferrable_loads_does_not_crash():
    """number_of_deferrable_loads can arrive at runtime as a string (the
    association loop copies the raw runtime value); the re-normalisation
    block must cast defensively instead of crashing on list * str.

    The cast int must also be written back into optim_conf, not just used
    locally to size the padded arrays - otherwise every downstream reader of
    optim_conf["number_of_deferrable_loads"] (a bare range()/arithmetic use
    in optimization.py) still crashes on the string, just relocated. This is
    the end-to-end claim the test's own name makes."""
    base = build_params()
    _, optim_conf, _ = treat_runtime(
        {
            "number_of_deferrable_loads": "3",
            "deferrable_load_max_cost": [0, 0],
        },
        base,
    )
    assert len(optim_conf.get("deferrable_load_max_cost", [])) == 3
    assert optim_conf.get("number_of_deferrable_loads") == 3
    assert isinstance(optim_conf.get("number_of_deferrable_loads"), int), (
        "number_of_deferrable_loads must be cast back to int, not left as the "
        f"raw runtime string; got {optim_conf.get('number_of_deferrable_loads')!r}"
    )
    # range() over the returned count must not raise - the exact downstream
    # symptom this pins (optimization.py's bare range() reads).
    list(range(optim_conf["number_of_deferrable_loads"]))


# ─────────────── caller-dict mutation / passed_data aliasing ───────────────


def test_treat_runtimeparams_does_not_mutate_callers_dict_and_warns_each_call(caplog):
    """The association loop assigns runtime list values BY REFERENCE, so
    optim_conf[name] could end up being the exact same list object as
    runtimeparams[name]. check_def_loads pads in place, so without a
    defensive copy, padding optim_conf[name] would also silently pad the
    caller's own dict - and, on a second call reusing that now-already-padded
    dict, the warning would stop firing (nothing looks short anymore).

    Calls treat_runtimeparams twice with the SAME dict object (not
    re-serialised - a JSON round-trip would mask this, since orjson.loads
    always returns a fresh object)."""
    base = build_params()
    params_json = orjson.dumps(base).decode("utf-8")
    rp = {"number_of_deferrable_loads": 3, "deferrable_load_max_cost": [0, 0]}

    async def _call():
        rh_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        return await utils.treat_runtimeparams(
            rp, params_json, rh_conf, optim_conf, plant_conf, "dayahead-optim", logger, emhass_conf
        )

    warn_counts = []
    for _ in range(2):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _, _, optim_conf_out, _ = asyncio.run(_call())
        warn_counts.append(len(_short_warnings(caplog, "deferrable_load_max_cost")))
        # The caller's dict must be untouched after every call, not just the first.
        assert rp["deferrable_load_max_cost"] == [0, 0], (
            f"caller's runtimeparams dict was mutated: {rp['deferrable_load_max_cost']!r}"
        )
        assert optim_conf_out["deferrable_load_max_cost"] is not rp["deferrable_load_max_cost"], (
            "optim_conf must not alias the caller's list object"
        )
        assert optim_conf_out["deferrable_load_max_cost"] == [0, 0, 0.0]
    assert warn_counts == [1, 1], (
        f"warning must fire on every call reusing the same dict, got counts {warn_counts}"
    )


def test_naive_mpc_passed_data_alias_stays_short(caplog):
    """Same root cause as the mutation test, different victim: the naive-mpc
    branch aliases params["passed_data"][name] to params["optim_conf"][name]
    before the re-normalisation block runs. Because that block copies before
    padding, optim_conf's array becomes a distinct (padded) object while
    passed_data keeps pointing at the original (short, pre-pad) one - so the
    padding is confined to optim_conf, exactly where it belongs, and the
    serialized passed_data isn't silently lengthened to a count nothing
    asked it to report."""
    base = build_params()
    with caplog.at_level(logging.WARNING):
        params_str, _, optim_conf, _ = asyncio.run(
            _treat_runtime(
                {
                    "number_of_deferrable_loads": 3,
                    "operating_hours_of_each_deferrable_load": [4, 0],
                },
                base,
                set_type="naive-mpc-optim",
            )
        )
    params_out = orjson.loads(params_str)
    assert optim_conf["operating_hours_of_each_deferrable_load"] == [4, 0, 0]
    assert params_out["passed_data"]["operating_hours_of_each_deferrable_load"] == [4, 0], (
        "passed_data alias must stay at its pre-padding length/values, not "
        f"follow optim_conf's padding; got "
        f"{params_out['passed_data']['operating_hours_of_each_deferrable_load']!r}"
    )


# ──────────────── None-heal on a correctly-sized array ──────────────────────


def test_correctly_sized_none_element_healed_and_warns(caplog):
    """A None element in a correctly-sized RUNTIME array is deliberately
    healed to the param default - matching what build_params has always done
    for config arrays - and now fires a visible warning (previously this
    healed silently with no diagnostic). This test pins that behaviour so it
    can't be re-litigated as an accidental regression."""
    base = build_params()
    with caplog.at_level(logging.WARNING):
        _, optim_conf, _ = treat_runtime(
            {
                "number_of_deferrable_loads": 2,
                "treat_deferrable_load_as_semi_cont": [None, True],
            },
            base,
        )
    assert optim_conf.get("treat_deferrable_load_as_semi_cont") == [True, True]
    warnings = [
        rec
        for rec in caplog.records
        if rec.levelname == "WARNING" and "treat_deferrable_load_as_semi_cont" in rec.message
    ]
    assert len(warnings) == 1, "expected exactly one warning for the healed None element"
    assert "None" in warnings[0].message or "null" in warnings[0].message, (
        f"warning should describe a None-heal, not overclaim padding: {warnings[0].message!r}"
    )
    assert "padded from" not in warnings[0].message, (
        "same-length change is a None-heal, not padding - the message must not overclaim"
    )


# ──────────────── heat_topology overwrite ───────────────────────────────────


_TOPO = {
    "sources": [
        {
            "id": "boiler",
            "type": "gas",
            "efficiency": 0.9,
            "nominal_power": 10000,
            "min_power": 2000,
        }
    ],
    "storage": [
        {
            "id": "tank",
            "volume": 0.1,
            "start_temperature": 35,
            "min_temperature": [25] * 48,
            "max_temperature": [60] * 48,
            "thermal_loss": 0.05,
        }
    ],
    "flows": [{"from": "boiler", "to": "tank"}],
}


def test_heat_topology_overwrite_short_runtime_array_matches_compiled_sizing(caplog):
    """Pins the placement decision: heat_topology compiles to its own
    number_of_deferrable_loads and force-overwrites every one of the 9 table
    arrays, which runs AFTER the association loop, so the fix must land
    after that compile step, not immediately after the association loop. A
    short runtime array for one of the 9 must not survive - the final arrays
    must match the heat_topology-compiled sizing, with no crash and no
    false-positive warning (nothing was actually padded FROM the runtime
    value; heat_topology's own value won)."""
    base = build_params({"heat_topology": _TOPO})
    with caplog.at_level(logging.WARNING):
        _, optim_conf, _ = treat_runtime(
            {"deferrable_load_max_cost": []},
            base,
        )
    compiled_count = optim_conf.get("number_of_deferrable_loads")
    assert compiled_count == 1, f"expected the topology to compile to 1 load, got {compiled_count}"
    for name in DEF_ARRAY_NAMES:
        value = optim_conf.get(name)
        assert isinstance(value, list) and len(value) == compiled_count, (
            f"{name} must match the compiled count {compiled_count}, got {value!r}"
        )
    assert optim_conf.get("nominal_power_of_deferrable_loads") == [10000.0], (
        "heat_topology's own compiled value must win, not the runtime override's absence"
    )
    assert _short_warnings(caplog, "deferrable_load_max_cost") == [], (
        "no warning should fire: heat_topology already replaced the runtime value "
        "before this block runs, so nothing was actually padded from it"
    )


# ─────────────────────── runtimeparams=None passthrough ────────────────────


def test_runtimeparams_none_passes_through_unchanged():
    """The block runs unconditionally (even with runtimeparams=None, since
    heat_topology can mutate these arrays from static config alone). For a
    WELL-FORMED config (already correctly padded by build_params), that must
    still be a true no-op: every table array unchanged from what build_params
    produced."""
    base = build_params()

    async def _call():
        params_json = orjson.dumps(base).decode("utf-8")
        rh_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        return await utils.treat_runtimeparams(
            None,
            params_json,
            rh_conf,
            optim_conf,
            plant_conf,
            "dayahead-optim",
            logger,
            emhass_conf,
        )

    _, _, optim_conf, _ = asyncio.run(_call())
    for name in DEF_ARRAY_NAMES:
        assert optim_conf.get(name) == base["optim_conf"].get(name), (
            f"{name}: expected unchanged passthrough with runtimeparams=None, "
            f"got {optim_conf.get(name)!r} vs base {base['optim_conf'].get(name)!r}"
        )
    assert optim_conf.get("number_of_deferrable_loads") == base["optim_conf"].get(
        "number_of_deferrable_loads"
    )


# ───────────────────── oversize array is not truncated ─────────────────────


def test_oversize_runtime_array_is_not_truncated():
    """Non-goal, explicitly: no length-check/truncation of OVERSIZED arrays
    (the config path doesn't truncate them either, #929's check_def_loads
    only ever enlarges)."""
    base = build_params()
    oversized = [1000.0, 2000.0, 3000.0, 4000.0]
    _, optim_conf, _ = treat_runtime(
        {
            "number_of_deferrable_loads": 2,
            "nominal_power_of_deferrable_loads": oversized,
        },
        base,
    )
    assert optim_conf.get("nominal_power_of_deferrable_loads") == oversized, (
        "an oversized runtime array must pass through untouched, not be truncated "
        f"to number_of_deferrable_loads; got {optim_conf.get('nominal_power_of_deferrable_loads')!r}"
    )


# ───────────────────────── runtime scalar broadcast ─────────────────────────


def test_runtime_scalar_broadcasts_to_final_count(caplog):
    """A runtime scalar means "every load", not "the first N loads, padded
    with the table default for the rest". Base crashes on this exact shape
    (set_deferrable_load_single_constant: true bumping the count to 3 dies
    with an IndexError deep in the optimizer, since nothing before this fix
    ever re-sizes the array the earlier scalar-broadcast handling produces),
    so this pins the head behaviour rather than guarding a regression.
    Mirrors check_batt_params, which broadcasts a scalar the same way."""
    base = build_params()
    with caplog.at_level(logging.WARNING):
        _, optim_conf, _ = treat_runtime(
            {
                "number_of_deferrable_loads": 3,
                "set_deferrable_load_single_constant": True,
            },
            base,
        )
    assert optim_conf.get("set_deferrable_load_single_constant") == [True, True, True], (
        "a runtime scalar True must broadcast to every load, not leave the "
        f"added load at the table default; got "
        f"{optim_conf.get('set_deferrable_load_single_constant')!r}"
    )
    # A pure scalar broadcast is the user's stated intent, not a stale/short
    # payload - no re-normalisation warning should fire for it (see the
    # implementer's notes on this choice).
    assert _short_warnings(caplog, "set_deferrable_load_single_constant") == []


def test_runtime_scalar_broadcast_via_legacy_name():
    """The same broadcast must apply when the scalar arrives under the
    legacy name (set_def_constant -> set_deferrable_load_single_constant,
    src/emhass/data/associations.csv row 55)."""
    base = build_params()
    _, optim_conf, _ = treat_runtime(
        {"number_of_deferrable_loads": 3, "set_def_constant": False},
        base,
    )
    assert optim_conf.get("set_deferrable_load_single_constant") == [False, False, False]
