"""
Guards around more than one inverter (#1150, PR 3b part 1).

Covers the pieces that do not depend on the optimizer: the warning when a runtime scalar
masks a configured per-inverter list, the efficiency range above one inverter, the
curtailment helpers, the curtailment-history filter used by the PV forecast adjustment
and the site totals that publish_data posts above one inverter. Every behavioural
assertion fails on master on the assertion itself (new names are looked up with getattr).
"""

import asyncio
import copy
import json
import logging
import pathlib
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import orjson
import pandas as pd
import pytest

from emhass import command_line, utils

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "defaults_path": root / "src/emhass/data/config_defaults.json",
    "associations_path": root / "src/emhass/data/associations.csv",
}
logger, _ = utils.get_logger(__name__, emhass_conf, save_to_file=False)

TWO = {
    "number_of_inverters": 2,
    "inverter_ac_output_max": [5000, 3000],
    "inverter_ac_input_max": [5000, 3000],
    "battery_inverter_index": [0],
    "pv_plant_inverter_index": [1],
}


def _two(**overrides) -> dict:
    conf = copy.deepcopy(TWO)
    conf.update(overrides)
    return conf


def _build_params(overrides: dict) -> dict:
    async def go():
        config = json.loads(emhass_conf["defaults_path"].read_text(encoding="utf-8"))
        config.update(overrides)
        _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
        params = await utils.build_params(emhass_conf, secrets, config, logger)
        assert params is not False
        return params

    return asyncio.run(go())


def _treat_runtime(runtimeparams: dict, base_params: dict):
    async def go():
        params_json = orjson.dumps(base_params).decode("utf-8")
        rh_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        return await utils.treat_runtimeparams(
            orjson.dumps(runtimeparams).decode("utf-8"),
            params_json,
            rh_conf,
            optim_conf,
            plant_conf,
            "dayahead-optim",
            logger,
            emhass_conf,
        )

    return asyncio.run(go())


def _masking_warnings(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "per-inverter list" in r.getMessage()
    ]


# --------------------- runtime scalar masks a per-inverter list ---------------------


def test_runtime_scalar_over_distinct_list_warns_and_proceeds(caplog):
    base = _build_params(_two(inverter_efficiency_dc_ac=[0.95, 0.97]))
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        _treat_runtime({"inverter_efficiency_dc_ac": 0.96}, base)
    warnings = _masking_warnings(caplog)
    assert len(warnings) == 1, caplog.messages
    assert "inverter_efficiency_dc_ac" in warnings[0]
    assert "0.96" in warnings[0]
    assert "[0.95, 0.97]" in warnings[0]
    assert "list of 2 entries" in warnings[0]


def test_runtime_scalar_warns_for_each_broadcast_key(caplog):
    base = _build_params(
        _two(
            inverter_efficiency_ac_dc=[0.9, 0.95],
            inverter_stress_cost=[0.1, 0.2],
            compute_curtailment=[True, False],
        )
    )
    runtime = {
        "inverter_efficiency_ac_dc": 0.93,
        "inverter_stress_cost": 0.15,
        "compute_curtailment": True,
    }
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        _treat_runtime(runtime, base)
    text = " ".join(_masking_warnings(caplog))
    for key in runtime:
        assert key in text, caplog.messages


@pytest.mark.parametrize(
    ("configured", "runtime"),
    [
        ([0.95, 0.95], {"inverter_efficiency_dc_ac": 0.96}),  # uniform list
        ([0.95, 0.97], {"inverter_efficiency_dc_ac": [0.96, 0.98]}),  # runtime list
        ([0.95, 0.97], {}),  # nothing at runtime
    ],
)
def test_runtime_scalar_warning_is_silent_when_nothing_is_masked(configured, runtime, caplog):
    base = _build_params(_two(inverter_efficiency_dc_ac=configured))
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        _treat_runtime(runtime, base)
    assert _masking_warnings(caplog) == []


def test_runtime_scalar_warning_is_silent_with_one_inverter(caplog):
    base = _build_params({"inverter_efficiency_dc_ac": [0.95]})
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        _treat_runtime({"inverter_efficiency_dc_ac": 0.96}, base)
    assert _masking_warnings(caplog) == []


def test_runtime_scalar_warning_truncates_a_long_configured_list():
    warn = getattr(utils, "_warn_if_runtime_scalar_masks_inverter_list", None)
    assert warn is not None, "warning helper not implemented yet"
    messages: list[str] = []
    log = MagicMock()
    log.warning.side_effect = messages.append
    configured = [round(0.5 + i / 1000, 3) for i in range(500)]
    name = "inverter_stress_cost"
    warn(500, {name: 0.1}, name, {name: configured}, log)
    assert len(messages) == 1
    assert len(messages[0]) < 400
    assert "490 more entries" in messages[0]


# ------------------------- efficiency range above one inverter -------------------------


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("inverter_efficiency_dc_ac", [0.95, 0], "inverter_efficiency_dc_ac[1]=0 must be above 0"),
        ("inverter_efficiency_ac_dc", [1.2, 0.9], "inverter_efficiency_ac_dc[0]=1.2"),
        ("inverter_efficiency_dc_ac", 0, "inverter_efficiency_dc_ac=0 must be above 0"),
        ("inverter_efficiency_ac_dc", 1.5, "inverter_efficiency_ac_dc=1.5"),
        ("inverter_efficiency_dc_ac", [0.9, -0.1], "inverter_efficiency_dc_ac[1]=-0.1"),
    ],
)
def test_efficiency_out_of_range_is_a_fault_naming_the_slot(name, value, expected):
    faults = utils.inverter_config_faults(_two(**{name: value}))
    assert any(expected in fault and "at most 1" in fault for fault in faults), faults


def test_efficiency_fault_names_only_the_first_bad_slot_per_parameter():
    faults = utils.inverter_config_faults(_two(inverter_efficiency_dc_ac=[0, 0]))
    range_faults = [f for f in faults if "at most 1" in f]
    assert len(range_faults) == 1
    assert "[0]" in range_faults[0]
    assert "[1]" not in range_faults[0]


def test_efficiency_of_exactly_one_is_not_a_range_fault():
    faults = utils.inverter_config_faults(_two(inverter_efficiency_dc_ac=[0.95, 1.0]))
    assert not any("at most 1" in f for f in faults), faults


@pytest.mark.parametrize("value", [0, [0], [1.5], "0"])
def test_efficiency_range_is_not_checked_for_one_inverter(value):
    # One inverter keeps today's rules: the web UI's one-element list collapses as before.
    assert utils.inverter_config_faults({"inverter_efficiency_dc_ac": value}) == []


def test_efficiency_fault_is_reported_with_the_other_inverter_faults():
    conf = _two(inverter_efficiency_dc_ac=[0.95, 0], inverter_stress_cost=[0.1])
    faults = utils.inverter_config_faults(conf)
    assert any("at most 1" in f for f in faults), faults
    assert any("inverter_stress_cost" in f for f in faults), faults


# ----------------------------- curtailment helpers -----------------------------


@pytest.mark.parametrize(
    ("conf", "flags", "any_curtails"),
    [
        ({"compute_curtailment": True}, [True], True),
        ({"compute_curtailment": False}, [False], False),
        ({}, [False], False),
        ({"compute_curtailment": [False, False]}, [False, False], False),
        ({"compute_curtailment": [False, True]}, [False, True], True),
        ({"compute_curtailment": ["false", "TRUE"]}, [False, True], True),
        ({"compute_curtailment": [None, False]}, [False, False], False),
        ({"compute_curtailment": ["maybe", True]}, [False, True], True),
    ],
)
def test_curtailment_helpers(conf, flags, any_curtails):
    flags_func = getattr(utils, "inverter_curtailment_flags", None)
    any_func = getattr(utils, "any_inverter_curtails", None)
    assert flags_func is not None, "inverter_curtailment_flags not implemented yet"
    assert any_func is not None, "any_inverter_curtails not implemented yet"
    assert flags_func(conf) == flags
    assert any_func(conf) is any_curtails


# ----------------------- curtailment filter of the PV adjustment -----------------------


def _fit_pv_model_fetches(monkeypatch, compute_curtailment) -> int:
    """Run the real _retrieve_and_fit_pv_model; count the curtailment history fetches."""
    index = pd.date_range("2026-01-01", periods=48, freq="30min", tz="UTC")
    monkeypatch.setattr(
        command_line,
        "retrieve_home_assistant_data",
        AsyncMock(return_value=(True, pd.DataFrame({"x": np.zeros(48)}, index=index), index[:1])),
    )
    rh = MagicMock()
    rh.get_data = AsyncMock(return_value=False)
    fcst = MagicMock()
    fcst.plant_conf = {"compute_curtailment": compute_curtailment}
    fcst.params = {"passed_data": {}}
    fcst.x_adjust_pv = None
    fcst.adjust_pv_forecast_fit = AsyncMock()
    asyncio.run(
        command_line._retrieve_and_fit_pv_model(
            fcst, False, {}, {"adjusted_pv_regression_model": "LassoRegression"}, rh, {}, None
        )
    )
    return sum(
        1 for call in rh.get_data.call_args_list if "sensor.p_pv_curtailment" in call.args[1]
    )


def test_pv_adjustment_all_false_list_does_not_fetch_the_curtailment_history(monkeypatch):
    assert _fit_pv_model_fetches(monkeypatch, [False, False]) == 0


def test_pv_adjustment_any_true_in_a_list_fetches_the_curtailment_history(monkeypatch):
    assert _fit_pv_model_fetches(monkeypatch, [False, True]) == 1


def test_pv_adjustment_scalar_true_still_fetches_the_curtailment_history(monkeypatch):
    assert _fit_pv_model_fetches(monkeypatch, True) == 1


# --------------------------- publish_data above one inverter ---------------------------


def _publish(plant_overrides: dict, columns: list[str]):
    """Run the real publish_data on a fabricated frame; return (posted entities, log mock)."""
    params = _treat_runtime({}, _build_params({}))[0]
    if isinstance(params, str):
        params = orjson.loads(params)
    plant_conf = {"number_of_inverters": 1, "inverter_is_hybrid": False}
    plant_conf.update(plant_overrides)
    plant_conf.setdefault("compute_curtailment", False)
    index = pd.date_range(pd.Timestamp.now(tz="UTC").floor("h"), periods=6, freq="h")
    frame = pd.DataFrame(1.0, index=index, columns=["P_Load", "P_PV", "P_grid", *columns])
    frame["unit_load_cost"] = 0.3
    frame["unit_prod_price"] = 0.1
    frame["cost_fun_profit"] = 0.0
    frame["optim_status"] = "Optimal"
    posted: list[str] = []

    async def post_data(data, idx, entity_id, *args, **kwargs):
        posted.append(entity_id)

    rh = MagicMock()
    rh.post_data = AsyncMock(side_effect=post_data)
    input_data_dict = {
        "rh": rh,
        "params": params,
        "optim_conf": {"number_of_deferrable_loads": 0, "set_use_battery": False},
        "plant_conf": plant_conf,
        "retrieve_hass_conf": {"time_zone": ZoneInfo("UTC")},
    }
    log = MagicMock(spec=logging.Logger)
    asyncio.run(command_line.publish_data(input_data_dict, log, opt_res_latest=frame))
    return posted, log


_N2 = {
    "number_of_inverters": 2,
    "inverter_is_hybrid": False,
    "battery_inverter_index": [0],
    "pv_plant_inverter_index": [1],
}
_CURT = "sensor.p_pv_curtailment"
_HYB = "sensor.p_hybrid_inverter"
_BOTH = ["P_PV_curtailment", "P_hybrid_inverter"]


def test_publish_n2_posts_hybrid_total_and_no_curtailment_when_none_curtails():
    posted, _ = _publish({**_N2, "compute_curtailment": [False, False]}, _BOTH)
    assert _HYB in posted, posted
    assert _CURT not in posted, posted


def test_publish_n2_posts_curtailment_total_when_a_pv_inverter_curtails():
    posted, _ = _publish({**_N2, "compute_curtailment": [False, True]}, _BOTH)
    assert _HYB in posted, posted
    assert _CURT in posted, posted


def test_publish_n2_skips_a_missing_column_with_a_warning():
    posted, log = _publish({**_N2, "compute_curtailment": [False, True]}, ["P_PV_curtailment"])
    assert _HYB not in posted, posted
    assert _CURT in posted, posted
    warnings = " ".join(str(call.args[0]) for call in log.warning.call_args_list)
    assert "P_hybrid_inverter" in warnings


def test_publish_n2_curtailment_on_an_inverter_without_pv_expects_no_total():
    # Inverter 0 has the flag but no PV (its PV plant is on inverter 1), so the optimizer
    # makes no curtailment column and the saved plan has none: nothing posts, nothing warns.
    posted, log = _publish({**_N2, "compute_curtailment": [True, False]}, ["P_hybrid_inverter"])
    assert _HYB in posted, posted
    assert _CURT not in posted, posted
    warnings = " ".join(str(call.args[0]) for call in log.warning.call_args_list)
    assert "P_PV_curtailment" not in warnings, warnings


def test_publish_n2_curtailment_flag_on_the_other_inverter_without_pv_expects_no_total():
    # The mirror case: PV on inverter 0, the flag on inverter 1 only.
    conf = {**_N2, "pv_plant_inverter_index": [0], "compute_curtailment": [False, True]}
    posted, log = _publish(conf, ["P_hybrid_inverter"])
    assert _CURT not in posted, posted
    warnings = " ".join(str(call.args[0]) for call in log.warning.call_args_list)
    assert "P_PV_curtailment" not in warnings, warnings


def test_publish_n2_posts_curtailment_when_one_of_two_pv_inverters_curtails():
    conf = {**_N2, "pv_plant_inverter_index": [0, 1], "compute_curtailment": [False, True]}
    posted, _ = _publish(conf, _BOTH)
    assert _CURT in posted, posted


def test_publish_n2_single_curtailment_flag_applies_to_the_pv_inverter():
    posted, _ = _publish({**_N2, "compute_curtailment": True}, _BOTH)
    assert _CURT in posted, posted


def test_publish_n1_is_unchanged():
    posted, _ = _publish({"compute_curtailment": False, "inverter_is_hybrid": False}, _BOTH)
    assert _HYB not in posted
    assert _CURT not in posted
    posted, _ = _publish({"compute_curtailment": True, "inverter_is_hybrid": True}, _BOTH)
    assert _HYB in posted
    assert _CURT in posted


def test_publish_n1_missing_column_still_fails_as_before():
    # The stale-plan skip is only above one inverter; one inverter keeps today's KeyError.
    with pytest.raises(KeyError):
        _publish({"compute_curtailment": False, "inverter_is_hybrid": True}, ["P_PV_curtailment"])
