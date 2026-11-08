"""
Multi-inverter config plumbing tests (#1150, PR 1 of the per-inverter phasing).

Covers only the config layer: number_of_inverters, the per-inverter broadcast and cap
parameters (check_inverter_params), the battery_inverter_index / pv_plant_inverter_index
lists. More than one inverter is accepted once every specific fault is cleared. The
optimizer is out of scope here. Curve-list tests live in test_inverter_power_curves.py,
the PV plant length check in test_forecast.py, and the /set-config, /get-config and
/action behaviour in the "Web: per-inverter configuration" section of
test_inverter_power_curves.py, which holds the ``web`` fixture (there is no shared
conftest).

Base-safety: the new functions do not exist on master, so every test that exercises
them resolves the callable via getattr() and asserts it is present before calling it
(same approach as test_multi_battery_config.py). A run before the implementation fails
on a descriptive assertion rather than on import.
"""

import asyncio
import copy
import json
import logging
import pathlib
import tracemalloc

import orjson
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
    assert func is not None, f"utils.{name} not implemented yet (#1150)"
    return func


def _error_class():
    return _get_func("InverterConfigError")


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


def treat_no_runtime(base_params: dict, set_type="dayahead-optim"):
    """The CLI path: no runtime parameters at all (the runtime block is skipped)."""

    async def go():
        params_json = orjson.dumps(base_params).decode("utf-8")
        rh_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        return await utils.treat_runtimeparams(
            None,
            params_json,
            rh_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )

    _, rh_conf, optim_conf, plant_conf = asyncio.run(go())
    return rh_conf, optim_conf, plant_conf


# Values master carries for the inverter keys with the shipped defaults.
MASTER_INVERTER_DEFAULTS = {
    "inverter_is_hybrid": False,
    "inverter_ac_output_max": 5000,
    "inverter_ac_input_max": 5000,
    "inverter_efficiency_dc_ac": 1.0,
    "inverter_efficiency_ac_dc": 1.0,
    "inverter_power_curve_dc_ac": [],
    "inverter_power_curve_ac_dc": [],
    "inverter_stress_cost": 0.0,
    "inverter_stress_segments": 10,
    "compute_curtailment": False,
}

# A valid two-inverter shape: inverter 0 carries the battery, inverter 1 the PV plant.
TWO_INVERTERS = {
    "number_of_inverters": 2,
    "inverter_ac_output_max": [5000, 3000],
    "inverter_ac_input_max": [5000, 3000],
    "battery_inverter_index": [0],
    "pv_plant_inverter_index": [1],
}


def _inverter_records(caplog) -> list:
    """Log records about the new inverter keys (none are expected for a default config)."""
    return [
        rec
        for rec in caplog.records
        if "number_of_inverters" in rec.getMessage() or "_inverter_index" in rec.getMessage()
    ]


# ───────────────────────── number_of_inverters count ─────────────────────────


def test_number_of_inverters_defaults_to_one():
    params = build_params()
    assert params["plant_conf"].get("number_of_inverters") == 1


def test_inverter_config_error_is_a_value_error():
    assert issubclass(_error_class(), ValueError)


@pytest.mark.parametrize("bad", ["abc", 0, -1, 1.5, True])
def test_validate_num_inverters_rejects_invalid_counts(bad):
    validate = _get_func("validate_num_inverters")
    with pytest.raises(_error_class(), match="number_of_inverters must be a positive integer"):
        validate({"number_of_inverters": bad})


@pytest.mark.parametrize("null_like", [None, "null", " NULL ", "", "  "])
def test_validate_num_inverters_reads_a_null_count_as_one(null_like):
    """A null count is the default, as number_of_batteries: null is: the UI saves an
    emptied number input as null and build_params skips null values."""
    assert _get_func("validate_num_inverters")({"number_of_inverters": null_like}) == 1
    assert "plant_conf" in build_params({"number_of_inverters": null_like})


@pytest.mark.parametrize("raw,expected", [(1, 1), (3, 3), (2.0, 2), ("2", 2), (" 4 ", 4)])
def test_validate_num_inverters_accepts_and_coerces(raw, expected):
    validate = _get_func("validate_num_inverters")
    result = validate({"number_of_inverters": raw})
    assert result == expected
    assert type(result) is int


def test_validate_num_inverters_missing_key_defaults_to_one():
    assert _get_func("validate_num_inverters")({}) == 1


# ───────────────── N = 1 is a true no-op for every existing config ───────────


def test_n1_default_config_inverter_values_identical_to_master(caplog):
    """The shipped defaults reach plant_conf unchanged through build_params and
    treat_runtimeparams (with and without runtime parameters), with no new log lines."""
    with caplog.at_level(logging.INFO):
        base = build_params()
        _, _, plant_runtime = treat_runtime({}, base)
        _, _, plant_cli = treat_no_runtime(base)
    for plant_conf in (base["plant_conf"], plant_runtime, plant_cli):
        for key, expected in MASTER_INVERTER_DEFAULTS.items():
            assert plant_conf[key] == expected, key
            assert type(plant_conf[key]) is type(expected), key
        assert "battery_inverter_index" not in plant_conf
        assert "pv_plant_inverter_index" not in plant_conf
    assert _inverter_records(caplog) == []


def test_n1_scalar_overrides_stay_scalar_through_both_paths():
    overrides = {
        "inverter_is_hybrid": True,
        "inverter_ac_output_max": 6000,
        "inverter_ac_input_max": 4000,
        "inverter_efficiency_dc_ac": 0.97,
        "inverter_efficiency_ac_dc": 0.96,
        "inverter_stress_cost": 0.02,
        "compute_curtailment": True,
    }
    base = build_params(overrides)
    _, _, plant_conf = treat_runtime({"inverter_efficiency_dc_ac": 0.95}, base)
    for key, value in overrides.items():
        expected = 0.95 if key == "inverter_efficiency_dc_ac" else value
        assert base["plant_conf"][key] == overrides[key]
        assert plant_conf[key] == expected
        assert not isinstance(plant_conf[key], list)


def test_compute_curtailment_one_element_list_is_falsy_after_build_params():
    """The web UI saves array inputs as one-element lists, and [False] is truthy at every
    compute_curtailment read site, so the list must collapse to False."""
    params = build_params({"compute_curtailment": [False]})
    assert params["plant_conf"]["compute_curtailment"] is False
    assert not params["plant_conf"]["compute_curtailment"]


# (override key, UI-shaped value, scalar expected in plant_conf)
_UI_SHAPES = [
    ("compute_curtailment", [False], False),
    ("compute_curtailment", [True], True),
    ("inverter_efficiency_dc_ac", [0.97], 0.97),
    ("inverter_efficiency_ac_dc", [0.96], 0.96),
    ("inverter_stress_cost", [0.05], 0.05),
    ("inverter_ac_output_max", [6000], 6000),
    ("inverter_ac_input_max", [4000], 4000),
    ("inverter_efficiency_dc_ac", [None], 1.0),
    ("inverter_efficiency_dc_ac", ["0.98"], 0.98),
    ("compute_curtailment", [None], False),
    ("compute_curtailment", ["false"], False),
    ("compute_curtailment", ["TRUE"], True),
]


@pytest.mark.parametrize("key,value,expected", _UI_SHAPES)
def test_n1_one_element_lists_collapse_to_the_master_scalar(key, value, expected, caplog):
    with caplog.at_level(logging.WARNING):
        base = build_params({key: copy.deepcopy(value)})
    assert base["plant_conf"][key] == expected
    assert type(base["plant_conf"][key]) is type(expected)
    # And again when the same list arrives as a runtime override.
    default = build_params()
    _, _, plant_conf = treat_runtime({key: copy.deepcopy(value)}, default)
    assert plant_conf[key] == expected
    assert type(plant_conf[key]) is type(expected)
    assert _inverter_records(caplog) == []


def test_n1_one_element_cap_list_with_null_element_removes_the_cap():
    """The UI saves an emptied number input as [null]; the key then reads as absent, like
    a null scalar does for build_params."""
    conf = {"number_of_inverters": 1, "inverter_ac_output_max": [None]}
    _get_func("check_inverter_params")(conf, logger)
    assert "inverter_ac_output_max" not in conf


@pytest.mark.parametrize(
    "key,value",
    [
        ("inverter_efficiency_dc_ac", [0.9, 0.9]),
        ("inverter_efficiency_ac_dc", []),
        ("inverter_stress_cost", [0.0, 0.0, 0.0]),
        ("compute_curtailment", [False, False]),
        ("inverter_ac_output_max", [5000, 5000]),
        ("inverter_ac_input_max", []),
    ],
)
def test_n1_wrong_length_list_is_an_error_naming_the_key(key, value):
    conf = {key: list(value)}
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    message = str(excinfo.value)
    assert key in message
    assert f"has {len(value)} entries" in message
    assert "number_of_inverters=1" in message


def test_n1_wrong_length_list_through_build_params_logs_and_does_not_raise(caplog):
    with caplog.at_level(logging.ERROR):
        params = build_params({"inverter_efficiency_dc_ac": [0.9, 0.9]})
    assert params["plant_conf"]["inverter_efficiency_dc_ac"] == [0.9, 0.9]
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("inverter_efficiency_dc_ac" in m and "number_of_inverters=1" in m for m in errors)


def test_n1_wrong_length_list_through_treat_runtimeparams_raises_inverter_error():
    base = build_params()
    with pytest.raises(_error_class(), match="inverter_stress_cost"):
        treat_runtime({"inverter_stress_cost": [0.1, 0.2]}, base)


@pytest.mark.parametrize("unset", [None, [], [None], ["", None], [""], ["null"]])
def test_n1_unset_index_lists_are_silent_and_removed(unset, caplog):
    """What the web UI saves for an empty array input points at nothing: it is dropped, as
    a null cap is, so a one-inverter save gains no keys."""
    conf = {
        "battery_inverter_index": copy.deepcopy(unset),
        "pv_plant_inverter_index": copy.deepcopy(unset),
    }
    with caplog.at_level(logging.WARNING):
        _get_func("check_inverter_params")(conf, logger)
    assert _inverter_records(caplog) == []
    assert conf == {}


def test_n1_ui_shaped_save_leaves_neither_index_key_in_the_normalised_conf():
    params = build_params({"battery_inverter_index": [None], "pv_plant_inverter_index": [None]})
    assert "battery_inverter_index" not in params["plant_conf"]
    assert "pv_plant_inverter_index" not in params["plant_conf"]


def test_n1_a_set_index_list_stays_and_an_unset_one_beside_it_goes():
    conf = {"battery_inverter_index": [0], "pv_plant_inverter_index": [None]}
    _get_func("check_inverter_params")(conf, logger)
    assert conf == {"battery_inverter_index": [0]}


def test_n1_absent_index_lists_are_silent(caplog):
    with caplog.at_level(logging.WARNING):
        _get_func("check_inverter_params")({}, logger)
    assert caplog.records == []


def test_n1_set_index_list_warns_once_per_key_and_is_left_alone(caplog):
    conf = {"battery_inverter_index": [0], "pv_plant_inverter_index": [0]}
    with caplog.at_level(logging.WARNING):
        _get_func("check_inverter_params")(conf, logger)
    for key in ("battery_inverter_index", "pv_plant_inverter_index"):
        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING and r.getMessage().startswith(f"{key} is ignored")
        ]
        assert warnings == [
            f"{key} is ignored: number_of_inverters is 1, so there is only one inverter "
            "to point at."
        ]
        assert conf[key] == [0]


def test_n1_inverter_is_hybrid_is_untouched():
    conf = {"inverter_is_hybrid": True}
    _get_func("check_inverter_params")(conf, logger)
    assert conf == {"inverter_is_hybrid": True}


# ───────────────── N > 1: broadcast parameters (scalar or exact list) ─────────


def _two(**overrides) -> dict:
    conf = copy.deepcopy(TWO_INVERTERS)
    conf.update(overrides)
    return conf


_BROADCAST = [
    ("inverter_efficiency_dc_ac", 1.0),
    ("inverter_efficiency_ac_dc", 1.0),
    ("inverter_stress_cost", 0.0),
    ("compute_curtailment", False),
]


@pytest.mark.parametrize("name,default", _BROADCAST)
def test_n_gt_1_missing_key_broadcasts_the_default(name, default):
    conf = _two()
    _get_func("check_inverter_params")(conf, logger)
    assert conf[name] == [default, default]


@pytest.mark.parametrize("name,default", _BROADCAST)
def test_n_gt_1_none_value_broadcasts_the_default(name, default):
    conf = _two(**{name: None})
    _get_func("check_inverter_params")(conf, logger)
    assert conf[name] == [default, default]


def test_n_gt_1_scalar_broadcasts_to_every_inverter():
    conf = _two(
        inverter_efficiency_dc_ac=0.97,
        inverter_efficiency_ac_dc="0.95",
        inverter_stress_cost=0.1,
        compute_curtailment=True,
    )
    _get_func("check_inverter_params")(conf, logger)
    assert conf["inverter_efficiency_dc_ac"] == [0.97, 0.97]
    assert conf["inverter_efficiency_ac_dc"] == [0.95, 0.95]
    assert conf["inverter_stress_cost"] == [0.1, 0.1]
    assert conf["compute_curtailment"] == [True, True]


def test_n_gt_1_exact_length_list_passes_through_with_coercion():
    conf = _two(
        inverter_efficiency_dc_ac=[0.97, "0.95"],
        inverter_efficiency_ac_dc=[None, 0.9],
        inverter_stress_cost=["null", 0.2],
        compute_curtailment=[True, None],
    )
    _get_func("check_inverter_params")(conf, logger)
    assert conf["inverter_efficiency_dc_ac"] == [0.97, 0.95]
    assert conf["inverter_efficiency_ac_dc"] == [1.0, 0.9]
    assert conf["inverter_stress_cost"] == [0.0, 0.2]
    assert conf["compute_curtailment"] == [True, False]


@pytest.mark.parametrize("name,default", _BROADCAST)
@pytest.mark.parametrize("length", [1, 3])
def test_n_gt_1_wrong_length_list_names_param_length_and_count(name, default, length):
    conf = _two(**{name: [default] * length})
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    message = str(excinfo.value)
    assert name in message
    assert f"has {length} {'entry' if length == 1 else 'entries'}" in message
    assert "number_of_inverters=2" in message


@pytest.mark.parametrize(
    "name,element",
    [
        ("inverter_efficiency_dc_ac", "high"),
        ("inverter_stress_cost", True),
        ("inverter_efficiency_ac_dc", [0.9]),
        ("inverter_efficiency_dc_ac", float("nan")),
        ("compute_curtailment", 1),
        ("compute_curtailment", "maybe"),
    ],
)
def test_n_gt_1_bad_list_element_is_an_error_naming_the_slot(name, element):
    conf = _two(**{name: [1.0 if name != "compute_curtailment" else False, element]})
    with pytest.raises(_error_class(), match=rf"{name}\[1\]"):
        _get_func("check_inverter_params")(conf, logger)


@pytest.mark.parametrize("scalar", ["maybe", 3, [1]])
def test_n_gt_1_compute_curtailment_bad_scalar_is_an_error(scalar):
    conf = _two(compute_curtailment=scalar)
    with pytest.raises(_error_class(), match="compute_curtailment"):
        _get_func("check_inverter_params")(conf, logger)


def test_compute_curtailment_string_elements_are_case_insensitive_booleans():
    conf = _two(compute_curtailment=["True", " false "])
    _get_func("check_inverter_params")(conf, logger)
    assert conf["compute_curtailment"] == [True, False]


# ───────────────── N > 1: caps are never broadcast ───────────────────────────


@pytest.mark.parametrize("name", ["inverter_ac_output_max", "inverter_ac_input_max"])
@pytest.mark.parametrize("scalar", [5000, 12000.0, "4000", None])
def test_n_gt_1_cap_scalar_is_refused_with_an_explanation(name, scalar):
    conf = _two(**{name: scalar})
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    message = str(excinfo.value)
    assert name in message
    assert "number_of_inverters=2" in message
    assert "5000" in message  # the shipped default is named as the likely cause


@pytest.mark.parametrize("name", ["inverter_ac_output_max", "inverter_ac_input_max"])
def test_n_gt_1_cap_missing_key_is_refused(name):
    conf = _two()
    del conf[name]
    with pytest.raises(_error_class(), match=name):
        _get_func("check_inverter_params")(conf, logger)


@pytest.mark.parametrize("name", ["inverter_ac_output_max", "inverter_ac_input_max"])
@pytest.mark.parametrize("length", [1, 3])
def test_n_gt_1_cap_wrong_length_list_names_param_and_count(name, length):
    conf = _two(**{name: [5000] * length})
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    message = str(excinfo.value)
    assert name in message
    assert f"has {length} {'entry' if length == 1 else 'entries'}" in message
    assert "number_of_inverters=2" in message


@pytest.mark.parametrize("bad", [-1, float("inf"), float("nan"), True, "lots", None, [5000]])
def test_n_gt_1_cap_elements_must_be_finite_and_not_negative(bad):
    conf = _two(inverter_ac_output_max=[5000, bad])
    with pytest.raises(_error_class(), match=r"inverter_ac_output_max\[1\]"):
        _get_func("check_inverter_params")(conf, logger)


def test_n_gt_1_cap_list_is_coerced_and_zero_is_allowed():
    conf = _two(inverter_ac_output_max=["5000", 0], inverter_ac_input_max=[4000.5, 0.0])
    _get_func("check_inverter_params")(conf, logger)
    assert conf["inverter_ac_output_max"] == [5000.0, 0]
    assert conf["inverter_ac_input_max"] == [4000.5, 0.0]


# ───────────────── N > 1: index lists ─────────────────────────────────────────


@pytest.mark.parametrize("key", ["battery_inverter_index", "pv_plant_inverter_index"])
@pytest.mark.parametrize("unset", ["absent", None, [], [None], [""]])
def test_n_gt_1_unset_index_list_is_an_error(key, unset):
    conf = _two()
    if unset == "absent":
        del conf[key]
    else:
        conf[key] = unset
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    assert key in str(excinfo.value)
    assert "number_of_inverters=2" in str(excinfo.value)


@pytest.mark.parametrize("key", ["battery_inverter_index", "pv_plant_inverter_index"])
@pytest.mark.parametrize("bad", [True, False, 1.5, "x", "1.5", [0], None, 2, -1])
def test_n_gt_1_index_element_must_be_an_in_range_integer(key, bad):
    conf = _two(battery_inverter_index=[0], pv_plant_inverter_index=[1, 0])
    conf[key] = [0, bad] if key == "pv_plant_inverter_index" else [bad]
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    assert key in str(excinfo.value)


def test_n_gt_1_index_scalar_is_an_error():
    conf = _two(battery_inverter_index=0)
    with pytest.raises(_error_class(), match="battery_inverter_index"):
        _get_func("check_inverter_params")(conf, logger)


def test_n_gt_1_index_elements_are_normalised_to_plain_ints():
    conf = _two(
        number_of_batteries=2,
        battery_inverter_index=["0", 1.0],
        pv_plant_inverter_index=[" 1 ", 0.0],
    )
    _get_func("check_inverter_params")(conf, logger)
    assert conf["battery_inverter_index"] == [0, 1]
    assert conf["pv_plant_inverter_index"] == [1, 0]
    for key in ("battery_inverter_index", "pv_plant_inverter_index"):
        assert all(type(i) is int for i in conf[key])


def test_n_gt_1_battery_index_length_must_match_number_of_batteries():
    conf = _two(number_of_batteries=2, battery_inverter_index=[0])
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    message = str(excinfo.value)
    assert "battery_inverter_index" in message
    assert "number_of_batteries=2" in message
    # Two batteries may share one inverter.
    ok = _two(number_of_batteries=2, battery_inverter_index=[0, 0])
    _get_func("check_inverter_params")(ok, logger)
    assert ok["battery_inverter_index"] == [0, 0]


def test_n_gt_1_pv_plant_index_length_is_not_checked_here():
    """It depends on the forecast method, so it is checked at forecast time."""
    conf = _two(pv_plant_inverter_index=[1, 1, 0, 1])
    _get_func("check_inverter_params")(conf, logger)
    assert conf["pv_plant_inverter_index"] == [1, 1, 0, 1]


def test_n_gt_1_every_inverter_needs_a_battery_or_a_pv_plant():
    conf = _two(
        number_of_inverters=3,
        inverter_ac_output_max=[5000, 3000, 2000],
        inverter_ac_input_max=[5000, 3000, 2000],
        battery_inverter_index=[0],
        pv_plant_inverter_index=[0, 1],
    )
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    assert "inverter 2 has no battery and no PV plant" in str(excinfo.value)


def test_n_gt_1_inverter_is_hybrid_is_ignored_with_a_warning(caplog):
    conf = _two(inverter_is_hybrid=True)
    with caplog.at_level(logging.WARNING):
        _get_func("check_inverter_params")(conf, logger)
    assert (
        "inverter_is_hybrid is ignored above one inverter: each inverter's shape comes from "
        "what battery_inverter_index and pv_plant_inverter_index attach to it."
    ) in [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert conf["inverter_is_hybrid"] is True  # left as written


def test_n_gt_1_inverter_is_hybrid_false_is_silent(caplog):
    conf = _two(inverter_is_hybrid=False)
    with caplog.at_level(logging.WARNING):
        _get_func("check_inverter_params")(conf, logger)
    assert [r for r in caplog.records if "inverter_is_hybrid" in r.getMessage()] == []


def test_n_gt_1_bad_count_is_an_error():
    with pytest.raises(_error_class(), match="number_of_inverters must be a positive integer"):
        _get_func("check_inverter_params")({"number_of_inverters": 0}, logger)


# ───────────────── idempotence and in-place semantics ─────────────────────────


@pytest.mark.parametrize(
    "conf",
    [
        {},
        {"inverter_efficiency_dc_ac": [0.97], "compute_curtailment": [False]},
        {"inverter_ac_output_max": [6000], "inverter_stress_cost": ["0.1"]},
        {"battery_inverter_index": [None], "pv_plant_inverter_index": [0]},
        {"inverter_ac_output_max": [None]},
        _two(inverter_efficiency_dc_ac=[0.97, "0.9"], compute_curtailment=[True, "false"]),
        _two(
            number_of_batteries=2,
            battery_inverter_index=["0", 1.0],
            pv_plant_inverter_index=["1"],
            inverter_stress_cost="0.5",
        ),
    ],
)
def test_check_inverter_params_is_idempotent(conf):
    check = _get_func("check_inverter_params")
    once = copy.deepcopy(conf)
    check(once, logger)
    twice = copy.deepcopy(once)
    check(twice, logger)
    assert twice == once


# --------------- more than one inverter is accepted --------------------------


def test_reject_multiple_inverters_no_longer_exists():
    assert getattr(utils, "reject_multiple_inverters", None) is None


def test_a_usable_count_above_one_is_not_a_fault():
    faults = _get_func("inverter_config_faults")
    assert faults(copy.deepcopy(TWO_INVERTERS)) == []
    # a count of 3 alone is faulted only for its missing per-inverter settings
    found = faults({"number_of_inverters": 3})
    assert any("inverter_ac_output_max" in fault for fault in found)
    assert not any("not supported" in fault for fault in found)


def test_inverter_config_faults_returns_messages_and_leaves_the_input_alone():
    faults = _get_func("inverter_config_faults")
    assert faults({}) == []
    assert faults({"inverter_efficiency_dc_ac": [0.97]}) == []
    config = {"number_of_inverters": 2}
    snapshot = copy.deepcopy(config)
    messages = faults(config)
    assert config == snapshot
    # Above one inverter: every specific fault (here both caps and both index lists),
    # and nothing else is appended.
    assert len(messages) == 4
    assert "inverter_ac_output_max" in messages[0]
    assert not any("not supported" in m for m in messages)
    assert faults(_two()) == []
    # One inverter: only the specific fault.
    one = faults({"inverter_efficiency_dc_ac": [0.9, 0.9]})
    assert len(one) == 1 and "inverter_efficiency_dc_ac" in one[0]
    assert faults({"number_of_inverters": "abc"}) == [
        "number_of_inverters must be a positive integer, got 'abc'"
    ]


def test_inverter_config_faults_adds_no_handler_per_call():
    faults = _get_func("inverter_config_faults")
    quiet = logging.getLogger(f"{utils.__name__}.inverter_config_faults")

    # pytest adds its own capture handlers to loggers; only ours is counted.
    def ours():
        return [h for h in quiet.handlers if type(h) is logging.NullHandler]

    faults(_two())
    assert len(ours()) == 1
    for _ in range(3):
        faults(_two())
    assert len(ours()) == 1
    assert quiet.propagate is False


def test_faults_are_reported_in_every_entry_point_at_more_than_one_inverter():
    """A config with both a specific fault and number_of_inverters > 1 reports the fault."""
    config = _two(inverter_efficiency_dc_ac=[0.9, 0.9, 0.9])
    assert "inverter_efficiency_dc_ac" in _get_func("inverter_config_faults")(config)[0]
    base = build_params()
    with pytest.raises(_error_class(), match="inverter_efficiency_dc_ac"):
        treat_runtime(copy.deepcopy(config), base)


def test_treat_runtimeparams_accepts_more_than_one_inverter():
    base = build_params()
    _, _, plant_conf = treat_runtime(copy.deepcopy(TWO_INVERTERS), base)
    assert plant_conf["number_of_inverters"] == 2


def test_treat_runtimeparams_accepts_saved_more_than_one_inverter_without_runtime_params():
    """The CLI passes no runtime parameters: a saved usable configuration goes through."""
    base = build_params(copy.deepcopy(TWO_INVERTERS))
    treat_no_runtime(base)
    saved_fault = build_params(_two(inverter_ac_output_max=5000))
    with pytest.raises(_error_class(), match="inverter_ac_output_max"):
        treat_no_runtime(saved_fault)


@pytest.mark.parametrize("bad", ["abc", 0, -1, 1.5, True])
def test_treat_runtimeparams_logs_a_bad_runtime_count_before_raising(bad, caplog):
    """Every InverterConfigError on the action path is logged at error level first, so the
    action log (and the CLI) carry the reason."""
    base = build_params()
    with caplog.at_level(logging.ERROR):
        with pytest.raises(_error_class(), match="number_of_inverters must be a positive"):
            treat_runtime({"number_of_inverters": bad}, base)
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("number_of_inverters must be a positive integer" in m for m in errors), errors


@pytest.mark.parametrize("bad", ["abc", 0, -1, 1.5])
def test_treat_runtimeparams_logs_a_bad_saved_count_without_runtime_params(bad, caplog):
    base = build_params({"number_of_inverters": bad})
    with caplog.at_level(logging.ERROR):
        caplog.clear()
        with pytest.raises(_error_class(), match="number_of_inverters must be a positive"):
            treat_no_runtime(base)
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("number_of_inverters must be a positive integer" in m for m in errors), errors


def test_require_usable_inverter_config_logs_every_fault_and_nothing_else(caplog):
    require = _get_func("require_usable_inverter_config")
    conf = {"number_of_inverters": 2}
    with caplog.at_level(logging.ERROR):
        with pytest.raises(_error_class()) as excinfo:
            require(conf, logger)
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    # Both caps and both index lists: each logged on its own, nothing after them.
    assert len(errors) == 4
    assert "inverter_ac_output_max" in errors[0]
    assert not any("not supported" in m for m in errors)
    # The raised error carries all of them, so a body built from it says so too.
    message = str(excinfo.value)
    assert "inverter_ac_output_max" in message and "not supported" not in message
    assert ".." not in message


def test_require_usable_inverter_config_accepts_a_usable_more_than_one_inverter_config(caplog):
    with caplog.at_level(logging.ERROR):
        _get_func("require_usable_inverter_config")(_two(), logger)
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


def test_require_usable_inverter_config_is_silent_for_a_usable_one_inverter_config(caplog):
    with caplog.at_level(logging.WARNING):
        _get_func("require_usable_inverter_config")({"inverter_efficiency_dc_ac": [0.9]}, logger)
    assert caplog.records == []


def test_one_inverter_wording_is_singular_and_has_no_double_periods():
    check = _get_func("check_inverter_params")
    with pytest.raises(_error_class()) as excinfo:
        check({"inverter_efficiency_dc_ac": [0.9, 0.9]}, logger)
    message = str(excinfo.value)
    assert "exactly 1 entry" in message
    assert "1 entries" not in message
    with pytest.raises(_error_class(), match="exactly 2 entries"):
        check(_two(inverter_efficiency_dc_ac=[0.9]), logger)
    with pytest.raises(_error_class()) as two_faults:
        check({"number_of_inverters": 2}, logger)
    assert ".." not in str(two_faults.value)


# ───────────────── build_params never raises on an inverter fault ─────────────


@pytest.mark.parametrize(
    "overrides,needle",
    [
        ({"number_of_inverters": 2}, "inverter_ac_output_max"),
        ({"number_of_inverters": 0}, "number_of_inverters must be a positive integer"),
        ({"number_of_inverters": "two"}, "number_of_inverters must be a positive integer"),
        ({"inverter_efficiency_dc_ac": [0.9, 0.9]}, "inverter_efficiency_dc_ac"),
        ({"inverter_power_curve_dc_ac": 5}, "inverter_power_curve_dc_ac"),
        (
            {
                "number_of_inverters": 2,
                "inverter_power_curve_dc_ac": [[[1000, 0.9], [2000, 0.95]]],
            },
            "inverter_power_curve_dc_ac",
        ),
    ],
)
def test_build_params_never_raises_on_an_inverter_fault(overrides, needle, caplog):
    with caplog.at_level(logging.ERROR):
        params = build_params(overrides)
    assert "plant_conf" in params
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(needle in m for m in errors), errors


def test_build_params_n_gt_1_valid_config_is_normalised_and_accepted(caplog):
    with caplog.at_level(logging.ERROR):
        params = build_params(copy.deepcopy(TWO_INVERTERS))
    plant_conf = params["plant_conf"]
    assert plant_conf["inverter_efficiency_dc_ac"] == [1.0, 1.0]
    assert plant_conf["compute_curtailment"] == [False, False]
    assert plant_conf["battery_inverter_index"] == [0]
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert not any("number_of_inverters" in m for m in errors), errors


def test_build_params_n_gt_1_logs_the_specific_faults_and_nothing_else(caplog):
    with caplog.at_level(logging.ERROR):
        build_params({"number_of_inverters": 2})
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("inverter_ac_output_max" in m for m in errors)
    assert not any("not supported" in m for m in errors)
    assert not any(".." in m for m in errors)


# ───────────────── round 1 review: size-proportional validation (F2) ──────────

_BIG = 10**6  # large enough that a per-inverter list shows up as MBs, small enough to be base-safe


def _peak_kib(func) -> float:
    tracemalloc.start()
    try:
        func()
        return tracemalloc.get_traced_memory()[1] / 1024
    finally:
        tracemalloc.stop()


def test_huge_inverter_count_does_not_allocate_per_inverter_lists():
    """Validation work is proportional to the size of the input, not to number_of_inverters."""
    conf = {
        "number_of_inverters": _BIG,
        "inverter_efficiency_dc_ac": 0.97,
        "inverter_efficiency_ac_dc": 0.96,
        "inverter_stress_cost": 0.1,
        "compute_curtailment": True,
        "inverter_ac_output_max": [5000],
        "inverter_ac_input_max": [5000],
        "battery_inverter_index": [0],
        "pv_plant_inverter_index": [1],
    }
    check = _get_func("check_inverter_params")
    raised: list = []

    def run():
        with pytest.raises(_error_class()) as excinfo:
            check(conf, logger)
        raised.append(str(excinfo.value))

    peak = _peak_kib(run)
    # A broadcast list of 10**6 entries is 8 MB each; the bound is far below that.
    assert peak < 2048, f"peak {peak:.0f} KiB: validation allocated per inverter"
    assert "inverter_ac_output_max has 1 entry" in raised[0]
    assert "inverter_ac_input_max has 1 entry" in raised[0]


def test_huge_inverter_count_coverage_message_is_bounded():
    conf = _two(number_of_inverters=_BIG)
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    message = str(excinfo.value)
    # inverters 0 and 1 are attached; 2..11 are listed, the rest are counted.
    assert "inverter 2 has no battery and no PV plant" in message
    assert "inverter 11 has no battery and no PV plant" in message
    assert "inverter 12 has no battery" not in message
    assert f"and {_BIG - 2 - 10} more inverters have no battery and no PV plant" in message
    assert len(message) < 4096


def test_coverage_message_with_one_more_than_ten_missing_uses_the_singular():
    conf = _two(number_of_inverters=13)  # 0 and 1 attached, 2..12 missing
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    message = str(excinfo.value)
    assert "inverter 11 has no battery and no PV plant" in message
    assert "inverter 12 has no battery" not in message
    assert "and 1 more inverter has no battery and no PV plant" in message


def test_coverage_message_up_to_ten_missing_is_unchanged():
    conf = _two(
        number_of_inverters=4,
        inverter_ac_output_max=[1, 2, 3, 4],
        inverter_ac_input_max=[1, 2, 3, 4],
        pv_plant_inverter_index=[0],
    )
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    assert str(excinfo.value) == "; ".join(
        f"inverter {k} has no battery and no PV plant: add {k} to "
        "battery_inverter_index or pv_plant_inverter_index"
        for k in (1, 2, 3)
    )


def test_coverage_message_with_exactly_ten_missing_lists_all_ten_and_no_tail():
    conf = _two(
        number_of_inverters=11,
        inverter_ac_output_max=[1] * 11,
        inverter_ac_input_max=[1] * 11,
        pv_plant_inverter_index=[0],
    )
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    message = str(excinfo.value)
    assert message.count("has no battery and no PV plant") == 10
    assert "more inverter" not in message


# ───────────────── round 1 review: OverflowError (F3) ─────────────────────────


def test_validate_num_inverters_rejects_an_int_too_large_for_a_float():
    with pytest.raises(_error_class(), match="number_of_inverters must be a positive integer"):
        _get_func("validate_num_inverters")({"number_of_inverters": 10**400})


def test_huge_battery_count_is_an_inverter_error_not_an_overflow():
    conf = _two(number_of_batteries=10**400)
    with pytest.raises(_error_class(), match="number_of_batteries must be at most 100"):
        _get_func("check_inverter_params")(conf, logger)


# ───────────────── round 1 review: every fault at once (F4) ───────────────────

_FOUR_FAULTS = {
    "inverter_efficiency_dc_ac": [0.9, 0.9, 0.9],
    "inverter_stress_cost": "abc",
    "inverter_ac_output_max": 5000,
    "pv_plant_inverter_index": [7],
}


def test_check_inverter_params_reports_every_fault_in_check_order():
    conf = _two(**_FOUR_FAULTS)
    snapshot = copy.deepcopy(conf)
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    message = str(excinfo.value)
    problems = getattr(excinfo.value, "problems", None)
    assert problems is not None and len(problems) == 4, message
    for problem, name in zip(problems, _FOUR_FAULTS, strict=True):
        assert name in str(problem), problem
    assert message == ". ".join(str(p) for p in problems)
    assert conf == snapshot  # nothing applied when anything is wrong


def test_inverter_config_faults_lists_every_fault_and_nothing_else():
    messages = _get_func("inverter_config_faults")(_two(**_FOUR_FAULTS))
    assert len(messages) == 4
    for message, name in zip(messages, _FOUR_FAULTS, strict=True):
        assert name in message


def test_only_the_first_bad_element_of_a_parameter_is_reported():
    conf = _two(inverter_efficiency_dc_ac=["abc", "def"])
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    assert "inverter_efficiency_dc_ac[0]" in str(excinfo.value)
    assert "inverter_efficiency_dc_ac[1]" not in str(excinfo.value)


def test_a_single_fault_keeps_its_exact_message():
    conf = _two(inverter_efficiency_dc_ac=[0.9, 0.9, 0.9])
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    assert str(excinfo.value) == (
        "inverter_efficiency_dc_ac has 3 entries but number_of_inverters=2; provide a scalar "
        "(broadcast to every inverter) or a list of exactly 2 entries"
    )
    with pytest.raises(_error_class()) as one:
        _get_func("check_inverter_params")({"inverter_efficiency_dc_ac": [0.9, 0.9]}, logger)
    assert str(one.value) == (
        "inverter_efficiency_dc_ac has 2 entries but number_of_inverters=1; provide a single "
        "value or a list of exactly 1 entry"
    )


def test_one_inverter_faults_are_collected_too():
    conf = {"inverter_efficiency_dc_ac": [0.9, 0.9], "inverter_ac_output_max": [1, 2]}
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    message = str(excinfo.value)
    assert "inverter_efficiency_dc_ac" in message and "inverter_ac_output_max" in message
    assert conf == {"inverter_efficiency_dc_ac": [0.9, 0.9], "inverter_ac_output_max": [1, 2]}


def test_a_bad_battery_count_is_reported_once_and_skips_the_dependent_checks():
    conf = _two(number_of_batteries="abc")
    faults = _get_func("inverter_config_faults")(conf)
    assert faults[0] == "number_of_batteries must be a positive integer, got 'abc'"
    assert [f for f in faults if "number_of_batteries" in f] == faults[:1]
    assert not any("has no battery and no PV plant" in f for f in faults)


def test_the_coverage_check_waits_for_both_index_lists_to_be_usable():
    conf = _two(pv_plant_inverter_index=[9], battery_inverter_index=[0])
    with pytest.raises(_error_class()) as excinfo:
        _get_func("check_inverter_params")(conf, logger)
    assert "pv_plant_inverter_index[0]=9" in str(excinfo.value)
    assert "has no battery and no PV plant" not in str(excinfo.value)


def test_hybrid_warning_is_logged_even_when_there_are_faults(caplog):
    conf = _two(inverter_is_hybrid=True, inverter_ac_output_max=5000)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(_error_class()):
            _get_func("check_inverter_params")(conf, logger)
    assert any("inverter_is_hybrid is ignored" in r.getMessage() for r in caplog.records)


def test_require_usable_inverter_config_logs_and_raises_every_problem(caplog):
    conf = _two(**_FOUR_FAULTS)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(_error_class()) as excinfo:
            _get_func("require_usable_inverter_config")(conf, logger)
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 4
    message = str(excinfo.value)
    for name in _FOUR_FAULTS:
        assert name in message
    assert "not supported" not in message
    assert ".." not in message
