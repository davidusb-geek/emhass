"""battery_stress_segments and inverter_stress_segments validation.

Both keys used to be read with an unchecked plant_conf.get(..., 10). 0 or -1
gave an empty curve, so the stress penalty quietly dropped out. -2 or lower, a
float such as 10.0 or a numeric string (runtime params arrive verbatim) raised
while the problem was built. Now a value that is not a whole number from 1 to
STRESS_SEGMENTS_MAX warns and falls back to 10, the same for both keys.
"""

import json
import logging
import pathlib
from unittest import mock

import pandas as pd
import pytest

from emhass.optimization import STRESS_SEGMENTS_MAX
from tests.test_multi_battery_optimization import build_optimization

LOGGER_NAME = "multi_battery_test"


def scenario():
    index = pd.date_range("2026-02-01", periods=6, freq="30min", tz="Europe/Tallinn")
    p_pv = pd.Series([0, 2000, 4000, 4000, 1000, 0], index=index)
    p_load = pd.Series([800, 700, 900, 1000, 1200, 900], index=index)
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = [0.30, 0.10, 0.10, 0.12, 0.35, 0.40]
    df_input["unit_prod_price"] = [0.05, 0.04, 0.04, 0.04, 0.06, 0.07]
    return p_pv, p_load, df_input


HYBRID_PLANT = {
    "inverter_is_hybrid": True,
    "inverter_ac_output_max": 5000,
    "inverter_ac_input_max": 5000,
    "inverter_efficiency_dc_ac": 1.0,
    "inverter_efficiency_ac_dc": 1.0,
}


def segment_counts(plant_overrides):
    """Solve once and return the segment count of every stress curve built,
    in build order (inverter first, then each battery)."""
    opt = build_optimization(plant_overrides=plant_overrides)
    p_pv, p_load, df_input = scenario()
    with mock.patch.object(
        opt, "_add_stress_constraints", wraps=opt._add_stress_constraints
    ) as spy:
        opt.perform_dayahead_forecast_optim(df_input, p_pv, p_load, soc_init=0.5, soc_final=0.5)
    assert opt.optim_status in ("Optimal", "Optimal (Relaxed)")
    # args are (constraints, power_expression, stress_var, seg_params)
    return [len(call.args[3]) for call in spy.call_args_list]


def warnings_for(caplog, key):
    return [
        r
        for r in caplog.records
        if r.name == LOGGER_NAME and r.levelno == logging.WARNING and key in r.getMessage()
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (1, 1),
        (10, 10),
        (STRESS_SEGMENTS_MAX, STRESS_SEGMENTS_MAX),
        (10.0, 10),
        ("7", 7),
        ("10.0", 10),
    ],
)
def test_valid_values_are_used(caplog, raw, expected):
    opt = build_optimization(plant_overrides={"battery_stress_segments": raw})
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert opt._get_stress_segments("battery_stress_segments") == expected
    assert warnings_for(caplog, "battery_stress_segments") == []


@pytest.mark.parametrize(
    "raw",
    [
        0,
        -1,
        -2,
        STRESS_SEGMENTS_MAX + 1,
        2.5,
        "abc",
        "",
        None,
        True,
        False,
        float("nan"),
        float("inf"),
        10**400,
        [10],
    ],
)
def test_invalid_values_warn_and_fall_back_to_10(caplog, raw):
    opt = build_optimization(plant_overrides={"inverter_stress_segments": raw})
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert opt._get_stress_segments("inverter_stress_segments") == 10
    records = warnings_for(caplog, "inverter_stress_segments")
    assert len(records) == 1
    assert repr(raw) in records[0].getMessage()


def test_missing_key_defaults_to_10_without_warning(caplog):
    opt = build_optimization()
    opt.plant_conf.pop("battery_stress_segments")
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert opt._get_stress_segments("battery_stress_segments") == 10
    assert warnings_for(caplog, "battery_stress_segments") == []


def test_battery_zero_segments_no_longer_drops_the_penalty():
    # Before: 0 built an empty curve, so the battery stress cost silently did nothing.
    assert segment_counts({"battery_stress_cost": 0.2, "battery_stress_segments": 0}) == [10]


def test_battery_numeric_string_from_runtime_is_accepted():
    # Before: a string raised TypeError at segments + 1 while the problem was built.
    assert segment_counts({"battery_stress_cost": 0.2, "battery_stress_segments": "5"}) == [5]


def test_inverter_negative_segments_no_longer_raise():
    # Before: -2 raised ValueError in np.linspace while the problem was built.
    plant = {**HYBRID_PLANT, "inverter_stress_cost": 0.2, "inverter_stress_segments": -2}
    assert segment_counts(plant) == [10]


def test_inverter_float_segments_are_accepted():
    # Before: a float such as 4.0 raised TypeError while the problem was built.
    plant = {**HYBRID_PLANT, "inverter_stress_cost": 0.2, "inverter_stress_segments": 4.0}
    assert segment_counts(plant) == [4]


def test_valid_values_still_reach_both_curves():
    plant = {
        **HYBRID_PLANT,
        "inverter_stress_cost": 0.2,
        "inverter_stress_segments": 3,
        "battery_stress_cost": 0.2,
        "battery_stress_segments": 6,
    }
    assert segment_counts(plant) == [3, 6]


def test_shared_battery_key_warns_once_for_several_batteries(caplog):
    plant = {
        "number_of_batteries": 2,
        "battery_stress_cost": 0.2,
        "battery_stress_segments": 0,
    }
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert segment_counts(plant) == [10, 10]
    assert len(warnings_for(caplog, "battery_stress_segments")) == 1


def test_bad_value_is_ignored_quietly_when_stress_cost_is_off(caplog):
    # The segments only matter when a stress cost is set; no warning otherwise.
    plant = {**HYBRID_PLANT, "inverter_stress_cost": 0.0, "inverter_stress_segments": 0}
    plant["battery_stress_segments"] = "abc"
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert segment_counts(plant) == []
    assert warnings_for(caplog, "stress_segments") == []


def test_published_metadata_matches_the_accepted_range():
    # The web-config definitions and the generated OpenAPI spec advertise the same
    # 1..STRESS_SEGMENTS_MAX range that _get_stress_segments enforces.
    static = pathlib.Path(__file__).resolve().parents[1] / "src" / "emhass" / "static"
    definitions = json.loads(
        (static / "data" / "param_definitions.json").read_text(encoding="utf-8")
    )
    spec = json.loads((static / "openapi.json").read_text(encoding="utf-8"))
    props = spec["components"]["schemas"]["Config"]["properties"]
    for key in ("battery_stress_segments", "inverter_stress_segments"):
        found = [d[key] for d in definitions.values() if isinstance(d, dict) and key in d]
        assert len(found) == 1
        assert (found[0]["minimum"], found[0]["maximum"]) == (1, STRESS_SEGMENTS_MAX)
        assert (props[key]["minimum"], props[key]["maximum"]) == (1, STRESS_SEGMENTS_MAX)
