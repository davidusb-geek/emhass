#!/usr/bin/env python
"""Tests for the per-inverter optimizer model above one inverter (#1150, PR 3a).

Topology and PV path: the topology each inverter resolves to, the per-inverter PV path
(Parameters, assembly, the column-to-inverter map), the variables a multi-inverter
``Optimization`` creates, and the structural cache key. An ``Optimization`` is built
directly here with hand-normalised per-inverter lists, the form utils hands it.
``inverter_is_hybrid`` is left OUT of the plant_conf and ``compute_curtailment`` is a
list, so any bare read of either key fails loudly.

``build_multi`` is the shared builder; the per-inverter model tests further down reuse it.
"""

import asyncio
import logging
import pathlib
from unittest import mock

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from emhass import command_line
from emhass.command_line import OptimizationCache
from emhass.forecast import Forecast
from emhass.optimization import BATTERY_TIEBREAK_EPS, Optimization
from emhass.utils import InverterConfigError

TEST_ROOT = pathlib.Path(__file__).resolve().parents[1]
CHARGE_MAX = 5000
CAP = 10000
HORIZON = 8
DC_AC = "inverter_power_curve_dc_ac"
AC_DC = "inverter_power_curve_ac_dc"
CURVE = [[1000, 0.8], [4000, 0.95]]


def build_multi(
    n_inv=2,
    battery_inverter_index=(0,),
    pv_plant_inverter_index=(0, 1),
    cap_out=None,
    cap_in=None,
    eff_dc_ac=None,
    eff_ac_dc=None,
    curves_dc_ac=None,
    curves_ac_dc=None,
    curtail=None,
    stress=None,
    use_battery=True,
    plant_overrides=None,
    optim_overrides=None,
    logger=None,
) -> Optimization:
    """An N>1 Optimization from hand-normalised lists (no deferrable loads).

    One battery per ``battery_inverter_index`` entry. Caps default to 5 kW per inverter.
    """
    n_batt = len(battery_inverter_index)
    build_logger = logger or logging.getLogger("multi_inverter_build")
    if logger is None:
        build_logger.handlers = [logging.NullHandler()]
    retrieve_hass_conf = {
        "optimization_time_step": pd.to_timedelta(30, "minutes"),
        "time_zone": "Europe/Tallinn",
        "sensor_power_photovoltaics": "pv",
        "sensor_power_load_no_var_loads": "load",
    }
    optim_conf = {
        "delta_forecast_daily": pd.Timedelta(hours=4),
        "num_threads": 0,
        "set_use_battery": use_battery,
        "set_use_pv": True,
        "set_total_pv_sell": False,
        "set_nocharge_from_grid": False,
        "set_nodischarge_to_grid": False,
        "set_battery_dynamic": False,
        "set_battery_first_priority": False,
        "battery_dynamic_max": 0.9,
        "battery_dynamic_min": -0.9,
        "weight_battery_discharge": 0.0,
        "weight_battery_charge": 0.0,
        "battery_soc_deficit_threshold": 0.2,
        "battery_soc_deficit_cost": 0.0,
        "battery_soc_surplus_threshold": 0.9,
        "battery_soc_surplus_cost": 0.0,
        "number_of_deferrable_loads": 0,
        "nominal_power_of_deferrable_loads": [],
        "treat_deferrable_load_as_semi_cont": [],
        "set_deferrable_load_single_constant": [],
        "set_deferrable_startup_penalty": [],
        "operating_hours_of_each_deferrable_load": [],
        "start_timesteps_of_each_deferrable_load": [],
        "end_timesteps_of_each_deferrable_load": [],
        "lp_solver_timeout": 45,
        "lp_solver_mip_rel_gap": 0,
    }
    optim_conf.update(optim_overrides or {})
    plant_conf = {
        "number_of_inverters": n_inv,
        "number_of_batteries": n_batt,
        "battery_inverter_index": list(battery_inverter_index),
        "pv_plant_inverter_index": list(pv_plant_inverter_index),
        "inverter_ac_output_max": cap_out if cap_out is not None else [5000] * n_inv,
        "inverter_ac_input_max": cap_in if cap_in is not None else [5000] * n_inv,
        "inverter_efficiency_dc_ac": eff_dc_ac if eff_dc_ac is not None else [1.0] * n_inv,
        "inverter_efficiency_ac_dc": eff_ac_dc if eff_ac_dc is not None else [1.0] * n_inv,
        DC_AC: curves_dc_ac if curves_dc_ac is not None else [],
        AC_DC: curves_ac_dc if curves_ac_dc is not None else [],
        "compute_curtailment": curtail if curtail is not None else [False] * n_inv,
        "inverter_stress_cost": stress if stress is not None else [0.0] * n_inv,
        "maximum_power_from_grid": 50000,
        "maximum_power_to_grid": 50000,
        "battery_discharge_power_max": [CHARGE_MAX] * n_batt,
        "battery_charge_power_max": [CHARGE_MAX] * n_batt,
        "battery_minimum_state_of_charge": [0.05] * n_batt,
        "battery_maximum_state_of_charge": [0.95] * n_batt,
        "battery_target_state_of_charge": [0.5] * n_batt,
        "battery_nominal_energy_capacity": [CAP] * n_batt,
        "battery_discharge_efficiency": [1.0] * n_batt,
        "battery_charge_efficiency": [1.0] * n_batt,
        "battery_stress_cost": [0.0] * n_batt,
        "battery_stress_segments": 10,
    }
    plant_conf.update(plant_overrides or {})
    return Optimization(
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        "unit_load_cost",
        "unit_prod_price",
        "profit",
        {"root_path": TEST_ROOT / "src" / "emhass", "data_path": TEST_ROOT / "data"},
        build_logger,
        opt_time_delta=4,
    )


def _frame(n=HORIZON, pv=1000.0, load=300.0):
    index = pd.date_range("2026-03-04", periods=n, freq="30min", tz="Europe/Tallinn")
    df_input = pd.DataFrame(
        {"unit_load_cost": [0.2] * n, "unit_prod_price": [0.05] * n}, index=index
    )
    return df_input, pd.Series([pv] * n, index=index), pd.Series([load] * n, index=index)


def _plants(*columns):
    return pd.DataFrame({f"P_PV_plant{j}": col for j, col in enumerate(columns)})


# --------------------------------------------------------------------------- #
# Spec resolver
# --------------------------------------------------------------------------- #


def _summary(opt):
    return [(s["shape"], s["batteries"], s["binary"], s["curtail"]) for s in opt.inverter_specs]


def test_n1_has_no_specs_or_per_inverter_state():
    opt = build_multi(
        n_inv=1,
        pv_plant_inverter_index=(0,),
        plant_overrides={
            "inverter_is_hybrid": True,
            "inverter_ac_output_max": 5000,
            "inverter_ac_input_max": 5000,
            "inverter_efficiency_dc_ac": 1.0,
            "inverter_efficiency_ac_dc": 1.0,
            "compute_curtailment": False,
            "inverter_stress_cost": 0.0,
        },
    )
    assert opt.n_inv == 1
    assert opt.inverter_specs == [] and opt.param_pv_inv == []
    assert "p_hybrid_inverter" in opt.vars and "p_inverter" not in opt.vars


def test_hybrid_and_pv_only_shapes():
    opt = build_multi(
        battery_inverter_index=(0,), pv_plant_inverter_index=(0, 1), curtail=[True, True]
    )
    assert _summary(opt) == [("hybrid", [0], "own", True), ("pv_only", [], None, True)]
    assert opt.pv_inverters == [0, 1]
    assert [p is not None for p in opt.param_pv_inv] == [True, True]


def test_battery_only_one_battery_reuses_its_binary_and_two_get_their_own():
    one = build_multi(n_inv=2, battery_inverter_index=(1,), pv_plant_inverter_index=(0,))
    assert _summary(one) == [("pv_only", [], None, False), ("battery_only", [0], "battery", False)]
    two = build_multi(n_inv=2, battery_inverter_index=(1, 1), pv_plant_inverter_index=(0,))
    assert two.inverter_specs[1]["shape"] == "battery_only"
    assert two.inverter_specs[1]["batteries"] == [0, 1]
    assert two.inverter_specs[1]["binary"] == "own"
    assert two.param_pv_inv[1] is None  # no PV on a battery-only inverter


def test_curtailment_flag_only_counts_on_an_inverter_with_pv():
    opt = build_multi(
        n_inv=2, battery_inverter_index=(1,), pv_plant_inverter_index=(0,), curtail=[True, True]
    )
    assert [s["curtail"] for s in opt.inverter_specs] == [True, False]
    curt = opt.vars["p_pv_curtailment_inv"]
    assert curt[0] is not None and curt[1] is None
    assert curt[0].shape == (HORIZON,) and len(opt.vars["p_inverter"]) == 2


def test_inactive_battery_leaves_an_inert_inverter_and_one_warning(caplog):
    log = logging.getLogger("multi_inverter_inert")
    with caplog.at_level(logging.WARNING, logger=log.name):
        opt = build_multi(
            battery_inverter_index=(1,), pv_plant_inverter_index=(0,), use_battery=False, logger=log
        )
    assert opt.inverter_specs[1]["shape"] == "inert"
    assert opt.inverter_specs[1]["batteries"] == []
    warned = [r for r in caplog.records if "Inverter 1" in r.getMessage()]
    assert len(warned) == 1 and "0 W" in warned[0].getMessage()
    # A hybrid whose battery is off is a PV-only inverter.
    off = build_multi(
        battery_inverter_index=(0,), pv_plant_inverter_index=(0, 1), use_battery=False
    )
    assert [s["shape"] for s in off.inverter_specs] == ["pv_only", "pv_only"]


def test_specs_carry_caps_efficiencies_and_stress_per_inverter():
    opt = build_multi(
        cap_out=[3000, 4000],
        cap_in=[2000, 1000],
        eff_dc_ac=[0.95, 0.9],
        eff_ac_dc=[0.93, 0.92],
        stress=[0.0, 0.01],
    )
    got = [
        (s["cap_out"], s["cap_in"], s["eff_dc_ac"], s["eff_ac_dc"], s["stress_cost"])
        for s in opt.inverter_specs
    ]
    assert got == [(3000, 2000, 0.95, 0.93, 0.0), (4000, 1000, 0.9, 0.92, 0.01)]


@pytest.mark.parametrize("key", ["inverter_ac_output_max", "inverter_ac_input_max"])
@pytest.mark.parametrize("bad", [5000, [5000], [5000, 5000, 5000], None])
def test_cap_lists_must_have_one_entry_per_inverter(key, bad):
    with pytest.raises(ValueError, match=rf"{key} (must be a list of 2 entries|has \d+ entr)"):
        build_multi(plant_overrides={key: bad})


def test_cap_length_is_checked_before_anything_sized_by_the_inverter_count():
    # A count of a billion would hang or run out of memory if anything were sized by it
    # before this check, so the error being raised is the proof.
    with pytest.raises(ValueError, match="inverter_ac_output_max has 2 entries"):
        build_multi(plant_overrides={"number_of_inverters": 10**9})


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"battery_inverter_index": [5]}, r"battery_inverter_index\[0\]=5 is not an inverter"),
        ({"battery_inverter_index": [0, 1]}, "battery_inverter_index has 2 entries"),
        ({"inverter_efficiency_dc_ac": [1.0]}, "inverter_efficiency_dc_ac has 1 entry"),
        ({"inverter_ac_input_max": [5000, -1]}, r"inverter_ac_input_max\[1\]=-1 must not be neg"),
    ],
)
def test_malformed_per_inverter_lists_name_the_key(overrides, match):
    with pytest.raises(ValueError, match=match):
        build_multi(plant_overrides=overrides)


@pytest.mark.parametrize(
    "key",
    [
        "inverter_ac_output_max",
        "inverter_ac_input_max",
        "inverter_efficiency_dc_ac",
        "inverter_efficiency_ac_dc",
        "inverter_stress_cost",
    ],
)
@pytest.mark.parametrize("bad", [True, "abc", float("nan"), 10**400])
def test_a_bad_per_inverter_number_names_its_key_and_slot(key, bad):
    # The same acceptance as the config path: no bool, no text, nothing non-finite, and an
    # integer too large for a float is refused rather than escaping as an OverflowError.
    with pytest.raises(ValueError, match=rf"{key}\[1\]"):
        build_multi(plant_overrides={key: [1.0, bad]})


def test_curve_entries_are_read_per_inverter_and_ac_dc_is_dropped_without_a_battery():
    opt = build_multi(
        battery_inverter_index=(0,),
        pv_plant_inverter_index=(0, 1),
        curves_dc_ac=[CURVE, []],
        curves_ac_dc=[CURVE, CURVE],
    )
    s0, s1 = opt.inverter_specs
    ac_pts, dc_pts = s0["curve_dc_ac"]  # the origin is added as the first point
    assert ac_pts == [0, 1000, 4000]
    np.testing.assert_allclose(dc_pts, [0, 1000 / 0.8, 4000 / 0.95])
    assert s0["curve_ac_dc"] is not None
    assert s1["curve_dc_ac"] is None  # empty entry: the scalar efficiency applies
    assert s1["curve_ac_dc"] is None  # PV-only inverter has no battery, entry dropped


@pytest.mark.parametrize(
    "curves,match",
    [
        ([[], [[1000, "x"]]], r"inverter_power_curve_dc_ac\[1\]"),
        # a list that is not one entry per inverter is refused, not read up to its end
        ([CURVE, [], CURVE], "inverter_power_curve_dc_ac must be a list.*number_of_inverters=2"),
        ([CURVE], "inverter_power_curve_dc_ac must be a list.*number_of_inverters=2"),
    ],
)
def test_a_malformed_curve_list_names_its_parameter(curves, match):
    with pytest.raises(ValueError, match=match):
        build_multi(curves_dc_ac=curves)


# --------------------------------------------------------------------------- #
# PV assembly
# --------------------------------------------------------------------------- #


def test_one_pv_inverter_takes_the_whole_aggregate():
    opt = build_multi(n_inv=2, battery_inverter_index=(0,), pv_plant_inverter_index=(1,))
    pv_inv, total = opt._assemble_inverter_pv(np.arange(HORIZON, dtype=float), None, None)
    assert pv_inv[0] is None
    np.testing.assert_array_equal(pv_inv[1], np.arange(HORIZON))
    np.testing.assert_array_equal(total, np.arange(HORIZON))


def test_two_pv_inverters_sum_their_mapped_columns():
    opt = build_multi(n_inv=2, battery_inverter_index=(0,), pv_plant_inverter_index=(0, 0, 1))
    plants = _plants(*[np.full(HORIZON, v) for v in (100.0, 200.0, 400.0)])
    # pvlib style: one column per plant, plants 0 and 1 on inverter 0
    pv_inv, total = opt._assemble_inverter_pv(np.zeros(HORIZON), plants, [0, 0, 1])
    np.testing.assert_allclose(pv_inv[0], 300.0)
    np.testing.assert_allclose(pv_inv[1], 400.0)
    np.testing.assert_allclose(total, 700.0)
    # csv/list style: one column per PV inverter
    pv_inv, total = opt._assemble_inverter_pv(np.zeros(HORIZON), plants.iloc[:, :2], [0, 1])
    np.testing.assert_allclose(pv_inv[0], 100.0)
    np.testing.assert_allclose(pv_inv[1], 200.0)
    np.testing.assert_allclose(total, 300.0)


@pytest.mark.parametrize(
    "plants,inverters,message",
    [
        (None, None, "this forecast has none"),
        (None, [0, 1], "this forecast has none"),
        ("frame", None, "do not match the inverters"),
        ("frame", [0, 0], "do not match the inverters"),
        ("frame", [0], "do not match the inverters"),
        ("short", [0, 1], "8-step horizon"),
    ],
)
def test_two_pv_inverters_without_a_usable_split_fail_with_the_refresh_hint(
    plants, inverters, message
):
    opt = build_multi(n_inv=2, battery_inverter_index=(0,), pv_plant_inverter_index=(0, 1))
    rows = HORIZON - 2 if plants == "short" else HORIZON
    frame = _plants(np.ones(rows), np.ones(rows)) if plants else None
    with pytest.raises(ValueError, match=f"{message}.*weather-forecast-cache"):
        opt._assemble_inverter_pv(np.ones(HORIZON), frame, inverters)


def test_a_non_finite_pv_value_names_the_inverter():
    opt = build_multi(n_inv=2, battery_inverter_index=(0,), pv_plant_inverter_index=(0, 1))
    bad = np.ones(HORIZON)
    bad[3] = np.nan
    with pytest.raises(InverterConfigError, match="inverter 1 has non-finite"):
        opt._assemble_inverter_pv(np.ones(HORIZON), _plants(np.ones(HORIZON), bad), [0, 1])
    single = build_multi(n_inv=2, battery_inverter_index=(0,), pv_plant_inverter_index=(1,))
    with pytest.raises(InverterConfigError, match="inverter 1 has non-finite"):
        single._assemble_inverter_pv(bad, None, None)


@pytest.mark.parametrize(
    "index,pv,message",
    [
        ((), 999.0, "nowhere to go"),
        ((), 0.0, None),
        ((0, 1), 0.0, None),
    ],
)
def test_pv_without_a_destination_fails_and_zero_pv_needs_no_split(index, pv, message):
    opt = build_multi(n_inv=2, battery_inverter_index=(0,), pv_plant_inverter_index=index)
    if message:
        with pytest.raises(ValueError, match=message):
            opt._assemble_inverter_pv(np.full(HORIZON, pv), None, None)
        return
    pv_inv, total = opt._assemble_inverter_pv(np.full(HORIZON, pv), None, None)
    assert total.tolist() == [0.0] * HORIZON
    assert [series is not None for series in pv_inv] == [bool(index)] * 2
    assert all(series.tolist() == [0.0] * HORIZON for series in pv_inv if series is not None)


@pytest.mark.parametrize("mapping", [[0.0, 1.0], [True, False], [0, 1.5]])
def test_a_map_that_is_not_whole_numbers_gets_the_mismatch_error(mapping):
    opt = build_multi(n_inv=2, battery_inverter_index=(0,), pv_plant_inverter_index=(0, 1))
    plants = _plants(np.ones(HORIZON), np.ones(HORIZON))
    with pytest.raises(ValueError, match="do not match the inverters"):
        opt._assemble_inverter_pv(np.ones(HORIZON), plants, mapping)


def test_pv_parameters_are_set_and_recreated_when_the_horizon_changes():
    opt = build_multi(n_inv=2, battery_inverter_index=(0,), pv_plant_inverter_index=(0, 1))
    first = list(opt.param_pv_inv)
    assert all(p.shape == (HORIZON,) for p in first)
    n = HORIZON + 4
    # the frame's PV (999) differs from the split sum (1000), so the assertion on the
    # aggregate proves it was rebuilt from the split
    df_input, p_pv, p_load = _frame(n=n, pv=999.0)
    plants = _plants(np.full(n, 600.0), np.full(n, 400.0))
    opt.perform_dayahead_forecast_optim(
        df_input, p_pv, p_load, p_pv_plants=plants, pv_plant_inverters=[0, 1]
    )
    assert opt.optim_status == "Optimal"
    assert all(p.shape == (n,) for p in opt.param_pv_inv)
    assert all(new is not old for new, old in zip(opt.param_pv_inv, first, strict=True))
    np.testing.assert_allclose(opt.param_pv_inv[0].value, 600.0)
    np.testing.assert_allclose(opt.param_pv_inv[1].value, 400.0)
    np.testing.assert_allclose(opt.param_pv_forecast.value, 1000.0)  # the rebuilt aggregate
    assert len(opt.vars["p_inverter"][0].value) == n


def test_perfect_optim_is_refused_above_one_inverter():
    opt = build_multi()
    with pytest.raises(ValueError, match="perfect-optim.*one inverter only"):
        opt.perform_perfect_forecast_optim(pd.DataFrame(), pd.date_range("2026-03-04", periods=1))


# --------------------------------------------------------------------------- #
# Forecast.get_pv_plant_inverters
# --------------------------------------------------------------------------- #


def _forecast(method, index, n_cols, n_inv=2):
    fcst = Forecast.__new__(Forecast)
    fcst.logger = logging.getLogger("multi_inverter_forecast")
    fcst.plant_conf = {"number_of_inverters": n_inv, "pv_plant_inverter_index": index}
    fcst.weather_forecast_method = method
    fcst.pv_plant_raw = (
        None if n_cols is None else pd.DataFrame(np.ones((3, n_cols)), columns=range(n_cols))
    )
    return fcst


@pytest.mark.parametrize(
    "method,index,n_cols,expected",
    [
        ("pvlib", [0, 0, 1], 3, [0, 0, 1]),
        ("solcast", [1, 0], 2, [1, 0]),
        ("solar.forecast", [0, 1], 2, [0, 1]),
        ("csv", [0, 0, 1], 2, [0, 1]),  # one column per PV inverter, not per plant
        ("list", [1, 1, 0], 2, [0, 1]),
        ("pvlib", [0, 0, 1], 2, None),  # length mismatch
        ("csv", [0, 0, 1], 3, None),
        ("list", [0, 1], None, None),  # no per-plant series
        ("pvlib", "bad", 2, None),  # unusable index
        ("pvlib", [0, 5], 2, None),  # out of range, as for csv
        ("pvlib", [0, 1.5], 2, None),  # not a whole number
    ],
)
def test_forecast_maps_split_columns_to_inverters_per_method(method, index, n_cols, expected):
    assert _forecast(method, index, n_cols).get_pv_plant_inverters() == expected


def test_forecast_gives_no_map_at_one_inverter():
    assert _forecast("pvlib", [0, 0], 2, n_inv=1).get_pv_plant_inverters() is None


# --------------------------------------------------------------------------- #
# Cache key
# --------------------------------------------------------------------------- #


def _key(opt_plant_overrides=None, n_inv=2):
    opt = build_multi(n_inv=n_inv, plant_overrides=opt_plant_overrides, battery_inverter_index=(0,))
    return OptimizationCache._compute_cache_key(
        opt.optim_conf, opt.plant_conf, "profit", opt.retrieve_hass_conf
    )


def test_n_inv_cache_key_is_hashable_and_ignores_the_hybrid_flag():
    base = _key({"compute_curtailment": [True, False]})
    assert isinstance(base.compute_curtailment, tuple)
    assert hash(base) == hash(_key({"compute_curtailment": [True, False]}))
    # the flag is not read above one inverter, so toggling it must not rebuild the problem
    assert _key({"compute_curtailment": [True, False], "inverter_is_hybrid": True}) == base
    assert _key({"compute_curtailment": [True, False], "inverter_is_hybrid": False}) == base
    # a per-inverter change still does
    assert _key({"compute_curtailment": [False, False]}) != base
    assert _key({"compute_curtailment": [True, False], "inverter_ac_output_max": [1, 2]}) != base


def test_one_inverter_cache_key_is_unchanged():
    single = {
        "inverter_is_hybrid": True,
        "compute_curtailment": True,
        "inverter_ac_output_max": 5000,
        "inverter_ac_input_max": 5000,
        "inverter_efficiency_dc_ac": 1.0,
        "inverter_efficiency_ac_dc": 1.0,
        "inverter_stress_cost": 0.0,
    }
    key = _key(single, n_inv=1)
    assert key.inverter_is_hybrid is True and key.compute_curtailment is True
    assert key != _key({**single, "inverter_is_hybrid": False}, n_inv=1)


def test_command_line_hands_the_inverter_map_to_the_optimizer(monkeypatch):
    opt = mock.MagicMock(_solve_lock=None)
    data = {
        "params": {"passed_data": {}},
        "stage_times": {},
        "opt": opt,
        "p_pv_forecast": "pv",
        "p_load_forecast": "load",
        "p_pv_plants": "plants",
        "pv_plant_inverters": [0, 1],
    }
    monkeypatch.setattr(command_line, "prepare_forecast_and_weather_data", lambda *a, **k: "df")
    for name in (  # the side steps around the call under test, all returning nothing
        "_saves_entities",
        "_log_optimization_summary",
        "_record_optim_snapshot",
        "_apply_cached_opt_runtime_updates",
    ):
        monkeypatch.setattr(command_line, name, lambda *a, **k: None)
    asyncio.run(command_line.dayahead_forecast_optim(data, logging.getLogger("x"), debug=True))
    kwargs = opt.perform_dayahead_forecast_optim.call_args.kwargs
    assert kwargs["pv_plant_inverters"] == [0, 1] and kwargs["p_pv_plants"] == "plants"


# --------------------------------------------------------------------------- #
# The per-inverter model, solved
# --------------------------------------------------------------------------- #


def solve(
    opt, pv=(2000.0, 1000.0), load=300.0, cost=0.2, prod=0.05, soc=0.5, final=None, n=HORIZON
):
    """Day-ahead solve. ``pv``: one value or series per PV inverter, in inverter order."""

    def series(v):
        return np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy()

    index = pd.date_range("2026-03-04", periods=n, freq="30min", tz="Europe/Tallinn")
    df_input = pd.DataFrame(
        {"unit_load_cost": series(cost), "unit_prod_price": series(prod)}, index=index
    )
    columns = [series(v) for v in pv]
    kwargs = {}
    if len(opt.pv_inverters) > 1:
        kwargs = {"p_pv_plants": _plants(*columns), "pv_plant_inverters": list(opt.pv_inverters)}

    def per_battery(v):
        return v if isinstance(v, list) or opt.n_batt < 2 else [v] * opt.n_batt

    final = soc if final is None else final
    res = opt.perform_dayahead_forecast_optim(
        df_input,
        pd.Series(sum(columns), index=index),
        pd.Series(series(load), index=index),
        soc_init=per_battery(soc),
        soc_final=per_battery(final),
        **kwargs,
    )
    assert opt.optim_status == "Optimal"
    return res


def col(res, name):
    return res[name].to_numpy()


def test_caps_bind_per_inverter_even_when_the_sum_would_allow_more():
    # 5 kW of PV on inverter 0, whose own AC cap is 3 kW; inverter 1 (cap 3 kW) has none.
    opt = build_multi(
        battery_inverter_index=(0, 1),
        cap_out=[3000, 3000],
        eff_dc_ac=[0.95] * 2,
        eff_ac_dc=[0.95] * 2,
    )
    surplus = 5000 - 3000 / 0.95
    # battery 0's target takes exactly the overflow (8 steps of 0.5 h), so nothing is spilled
    final = [0.1 + surplus * 4 / CAP, 0.1]
    res = solve(opt, pv=(5000.0, 0.0), soc=[0.1, 0.1], final=final)
    assert col(res, "P_hybrid_inverter_0").max() == pytest.approx(3000, abs=1e-3)
    assert col(res, "P_hybrid_inverter_1").max() <= 3000 + 1e-3
    # what the cap cannot pass goes into battery 0 through the DC side of the inverter
    assert col(res, "P_batt_0") == pytest.approx(-surplus, abs=1e-2)
    assert col(res, "P_PV_spill_0") == pytest.approx(0, abs=1e-3)
    assert col(res, "P_batt_1") == pytest.approx(0, abs=1e-3)
    # counterfactual: one aggregate 6 kW hybrid on the same site passes more than 3 kW
    agg = build_multi(
        n_inv=1,
        battery_inverter_index=(0, 0),
        pv_plant_inverter_index=(0,),
        plant_overrides={
            "inverter_is_hybrid": True,
            "inverter_ac_output_max": 6000,
            "inverter_ac_input_max": 6000,
            "inverter_efficiency_dc_ac": 0.95,
            "inverter_efficiency_ac_dc": 0.95,
            "compute_curtailment": False,
            "inverter_stress_cost": 0.0,
        },
    )
    assert col(solve(agg, pv=(5000.0,), soc=0.1), "P_hybrid_inverter").max() > 3000 + 100


def test_pv_of_one_inverter_reaches_a_battery_on_another_through_both_efficiencies():
    # PV-only inverter 0, battery-only inverter 1; paying to export, so the PV is stored.
    opt = build_multi(
        battery_inverter_index=(1,),
        pv_plant_inverter_index=(0,),
        eff_dc_ac=[0.9, 0.9],
        eff_ac_dc=[0.9, 0.9],
    )
    res = solve(opt, pv=(2000.0,), load=0.0, prod=-0.1, soc=0.1, final=0.748)
    assert col(res, "P_hybrid_inverter_0") == pytest.approx(1800, abs=1e-3)
    assert col(res, "P_hybrid_inverter_1") == pytest.approx(-1800, abs=1e-3)
    assert col(res, "P_batt") == pytest.approx(-2000 * 0.9 * 0.9, abs=1e-3)
    assert col(res, "P_grid") == pytest.approx(0, abs=1e-3)


def test_a_pv_only_inverter_never_takes_ac_and_an_inert_one_stays_at_zero():
    # inverter 0: PV only; inverter 1: battery only; inverter 2: nothing on it (inert)
    opt = build_multi(
        n_inv=3,
        battery_inverter_index=(1,),
        pv_plant_inverter_index=(0,),
        eff_dc_ac=[0.95] * 3,
        eff_ac_dc=[0.95] * 3,
    )
    # Paid-to-import night: an AC input on the PV-only inverter would let it burn grid
    # power on top of what the battery inverter takes, so it must have none.
    cost = [-0.1] * 4 + [0.5] * 4
    res = solve(opt, pv=(0.0,), load=1500.0, cost=cost, soc=0.1)
    internals = opt.vars["inverter_internals"]
    assert "p_ac_dc" not in internals[0] and "is_dc_sourcing" not in internals[0]
    assert internals[2] == {}
    assert col(res, "P_hybrid_inverter_0").min() >= -1e-6
    assert col(res, "P_hybrid_inverter_2") == pytest.approx(0, abs=1e-6)
    assert col(res, "P_hybrid_inverter_1").min() < -500  # the battery inverter does import AC
    assert "P_PV_2" not in res.columns


@pytest.mark.parametrize("batteries", [(1,), (1, 1)], ids=["one battery", "two batteries"])
def test_a_battery_only_inverter_cannot_run_both_ways_to_burn_surplus(batteries):
    # Full battery(ies) that cannot discharge, surplus PV and a negative export price:
    # running the inverter both ways at once would burn the surplus and pay. One battery
    # reuses its own direction binary, two get one of their own; either way the inverter
    # stays at 0 W.
    opt = build_multi(
        battery_inverter_index=batteries,
        pv_plant_inverter_index=(0,),
        eff_dc_ac=[0.9, 0.9],
        eff_ac_dc=[0.9, 0.9],
        plant_overrides={"battery_discharge_power_max": [0] * len(batteries)},
    )
    res = solve(opt, pv=(1000.0,), load=300.0, prod=-0.1, soc=0.95)
    assert col(res, "P_hybrid_inverter_1") == pytest.approx(0, abs=1e-6)
    assert col(res, "P_grid") == pytest.approx(-(900 - 300), abs=1e-3)
    own = "is_dc_sourcing" in opt.vars["inverter_internals"][1]
    assert own == (len(batteries) == 2)


def test_each_inverter_obeys_its_own_dc_to_ac_curve():
    curve_a = [[1000, 0.8], [4000, 0.95]]
    curve_b = [[1000, 0.9], [4000, 0.99]]
    idle = {"battery_discharge_power_max": [0, 0], "battery_charge_power_max": [0, 0]}
    opt = build_multi(
        battery_inverter_index=(0, 1), curves_dc_ac=[curve_a, curve_b], plant_overrides=idle
    )
    res = solve(opt, pv=(3000.0, 3000.0), load=0.0, soc=0.5)
    for i, curve in enumerate((curve_a, curve_b)):
        ac = [0.0] + [p[0] for p in curve]
        dc = [0.0] + [p[0] / p[1] for p in curve]
        # PV is used in full (no curtailment), so its 3 kW of DC is what the curve converts
        assert col(res, f"P_hybrid_inverter_{i}") == pytest.approx(
            np.interp(3000, dc, ac), abs=1e-3
        )
        assert f"inv_curve_dc_ac_{i}_u" in opt.vars["inverter_internals"][i]
    assert abs(col(res, "P_hybrid_inverter_0")[0] - col(res, "P_hybrid_inverter_1")[0]) > 50


def test_an_ac_to_dc_curve_converts_a_battery_only_inverter_charge():
    curve = [[1000, 0.9], [3000, 0.95]]
    opt = build_multi(
        battery_inverter_index=(1,), pv_plant_inverter_index=(0,), curves_ac_dc=[[], curve]
    )
    res = solve(opt, pv=(0.0,), load=2000.0, cost=[0.02] * 4 + [0.5] * 4, soc=0.1)
    ac_in = -col(res, "P_hybrid_inverter_1")
    charging = ac_in > 1
    assert charging.any()
    dc = opt.vars["inverter_internals"][1]["p_ac_dc"].value
    expected = np.interp(ac_in[charging], [0, 1000, 3000], [0, 900, 2850])
    assert dc[charging] == pytest.approx(expected, abs=1e-3)
    assert -col(res, "P_batt")[charging] == pytest.approx(expected, abs=1e-3)


def test_nodischarge_to_grid_bounds_export_by_the_pv_surplus_across_inverters():
    # Inverter 0: hybrid with 2 kW PV. Inverter 1: battery only. Export pays well in the
    # first half and the batteries start full, so without the rule they would export.
    def build(flag):
        return build_multi(
            battery_inverter_index=(0, 1),
            pv_plant_inverter_index=(0,),
            eff_dc_ac=[0.95] * 2,
            eff_ac_dc=[0.95] * 2,
            optim_overrides={"set_nodischarge_to_grid": flag},
        )

    prod = [0.5] * 4 + [0.05] * 4
    free = solve(build(False), pv=(2000.0,), prod=prod, soc=0.9)
    held_opt = build(True)
    held = solve(held_opt, pv=(2000.0,), prod=prod, soc=0.9)
    ceiling = 2000 * 0.95 - 300
    assert (-col(free, "P_grid_neg")).max() > ceiling + 500  # the rule is what limits it
    assert (-col(held, "P_grid_neg") <= ceiling + 1e-3).all()
    assert (-col(held, "P_grid_neg")).max() == pytest.approx(ceiling, abs=1e-3)
    np.testing.assert_allclose(held_opt.param_export_ceiling.value, ceiling)


def test_nodischarge_lets_a_hybrid_export_while_another_inverters_battery_discharges():
    # Hybrid 0 (2 kW PV, battery 0 wants to fill) and battery-only 1 (battery 1 wants to
    # empty). Export pays well, so the plan exports up to the PV surplus while battery 1
    # discharges. One-inverter E <= D would forbid that: battery 1 may not discharge in any
    # step that exports. Here only the export is bounded, by the PV surplus on the AC side
    # (the #936 AC-coupled rule), so battery energy cannot add to the export.
    opt = build_multi(
        battery_inverter_index=(0, 1),
        pv_plant_inverter_index=(0,),
        eff_dc_ac=[0.95] * 2,
        eff_ac_dc=[0.95] * 2,
        optim_overrides={"set_nodischarge_to_grid": True},
    )
    res = solve(opt, pv=(2000.0,), load=1000.0, prod=0.5, soc=[0.1, 0.9], final=[0.9, 0.1])
    export = -col(res, "P_grid_neg")
    both = (export > 1) & (col(res, "P_batt_1") > 1)
    assert both.any()  # exporting and discharging the other inverter's battery together
    ceiling = 2000 * 0.95 - 1000
    assert (export <= ceiling + 1e-3).all()


def test_nodischarge_with_curtailment_also_bounds_export_by_the_uncurtailed_pv(monkeypatch):
    # Dear export, full batteries that want to empty and two curtailing hybrids. The
    # ceiling alone would let the plan curtail all the PV and export battery energy in
    # its place; the second bound ties export to the PV that was kept.
    def run():
        opt = build_multi(
            battery_inverter_index=(0, 1),
            eff_dc_ac=[0.95] * 2,
            eff_ac_dc=[0.95] * 2,
            curtail=[True, True],
            optim_overrides={"set_nodischarge_to_grid": True},
        )
        return solve(opt, pv=(3000.0, 1000.0), prod=0.5, cost=0.6, soc=0.95, final=0.1)

    res = run()
    kept = col(res, "P_PV") - col(res, "P_PV_curtailment")
    assert (-col(res, "P_grid_neg") <= 0.95 * kept + 1e-3).all()
    assert col(res, "P_PV_curtailment").max() > 1000  # the bound is what limits export
    # With the bound removed the same site curtails everything and exports the batteries
    monkeypatch.setattr(Optimization, "_inverter_max_efficiency", staticmethod(lambda spec: 1e6))
    loose = run()
    kept = col(loose, "P_PV") - col(loose, "P_PV_curtailment")
    assert (-col(loose, "P_grid_neg") > 0.95 * kept + 100).any()


def test_export_ceiling_value_uses_the_ac_the_pv_can_deliver():
    scalar = {"cap_out": 1000.0, "eff_dc_ac": 0.9, "curve_dc_ac": None}
    curve = ([0, 1000, 4000], [0, 1250, 4210.526])  # the validator adds the origin
    curved = {"cap_out": 3000.0, "eff_dc_ac": 1.0, "curve_dc_ac": curve}
    opt = build_multi()
    opt.inverter_specs = [scalar, curved]
    opt.num_timesteps = 3
    out = opt._inverter_pv_ac_output
    np.testing.assert_allclose(out([np.array([500.0, 2000.0, 0.0]), None]), [450, 1000, 0])
    # curve: DC 1250 is AC 1000; DC beyond the last point is held at its AC; the cap clips
    np.testing.assert_allclose(
        out([None, np.array([1250.0, 9000.0, 0.0])]), [1000, 3000, 0], atol=1e-2
    )
    # a flat DC stretch (allowed by the validator): AC 1000 and 2000 both need DC 2000,
    # so 2000 W of PV delivers the highest of them
    flat = {"cap_out": 3000.0, "eff_dc_ac": 1.0, "curve_dc_ac": ([0, 1000, 2000], [0, 2000, 2000])}
    opt.inverter_specs = [scalar, flat]
    np.testing.assert_allclose(out([None, np.array([1000.0, 2000.0, 5000.0])]), [500, 2000, 2000])
    opt.inverter_specs = [scalar, curved]
    assert opt._inverter_max_efficiency(curved) == pytest.approx(0.95, abs=1e-4)
    assert opt._inverter_max_efficiency(scalar) == 0.9


def test_curtailment_is_per_inverter_bounded_by_its_pv_and_splits_by_index():
    # Identical inverters, surplus PV, paying to export and full batteries: the PV must be
    # curtailed and the split between the two identical inverters is an exact tie. The
    # index tilt makes inverter 0 curtail first.
    opt = build_multi(
        battery_inverter_index=(0, 1),
        eff_dc_ac=[0.95] * 2,
        eff_ac_dc=[0.95] * 2,
        curtail=[True, True],
    )
    res = solve(opt, pv=(2000.0, 2000.0), load=300.0, prod=-0.1, soc=0.95)
    curt0, curt1 = col(res, "P_PV_curtailment_0"), col(res, "P_PV_curtailment_1")
    assert curt0 == pytest.approx(2000, abs=1e-2)  # all of inverter 0's PV, up to its bound
    assert curt1 == pytest.approx(4000 - 2000 - 300 / 0.95, abs=1e-2)
    assert (curt1 <= col(res, "P_PV_1") + 1e-6).all()
    np.testing.assert_allclose(col(res, "P_PV_curtailment"), curt0 + curt1)
    assert col(res, "P_grid") == pytest.approx(0, abs=1e-3)


def test_curtailment_time_weights_do_not_recruit_battery_round_trips():
    # Export is paid to be avoided and the batteries must end where they started, so the
    # surplus PV can only be curtailed. The late-curtailment time weights alone would make
    # a round trip through the AC bus (one inverter's output into another's battery, lossy
    # at 0.95) a free way to curtail later; the cost on AC input stops that loop.
    cost = [0.01] * 4 + [0.5] * 4
    hybrid_plus_battery = build_multi(
        battery_inverter_index=(0, 1),
        pv_plant_inverter_index=(0,),
        eff_dc_ac=[0.95] * 2,
        eff_ac_dc=[0.95] * 2,
        curtail=[True, False],
        optim_overrides={"set_nocharge_from_grid": True},
    )
    res = solve(hybrid_plus_battery, pv=(2000.0,), load=300.0, cost=cost, prod=-0.2, soc=0.1)
    assert col(res, "P_batt_1") == pytest.approx(0, abs=1e-3)
    two_hybrids = build_multi(
        battery_inverter_index=(0, 1),
        eff_dc_ac=[0.95] * 2,
        eff_ac_dc=[0.95] * 2,
        curtail=[True, True],
        optim_overrides={"set_nocharge_from_grid": True},
    )
    solve(two_hybrids, pv=(2000.0, 2000.0), load=300.0, cost=cost, prod=-0.2, soc=0.1)
    for internals in two_hybrids.vars["inverter_internals"]:
        assert internals["p_ac_dc"].value == pytest.approx(0, abs=1e-3)


def test_stress_cost_is_per_inverter_and_only_on_the_inverters_that_have_it():
    kw = {"battery_inverter_index": (0, 1), "pv_plant_inverter_index": (0,)}
    # Export pays more in steps 2-3 and 6-7. Without stress, battery 0 shifts PV into those
    # steps and inverter 0 peaks at its 5 kW cap. With stress, a flat 4 kW costs less.
    # Battery 1 is slightly lossy so the shift has to go through battery 0: with two
    # lossless batteries routing it through battery 1 instead is an equal optimum, and
    # only the 1e-9 battery tie-break would pick between them.
    prod = [0.05, 0.05, 0.12, 0.12, 0.05, 0.05, 0.12, 0.12]
    lossy_battery_1 = {"battery_discharge_efficiency": [1.0, 0.99]}

    def run(stress):
        opt = build_multi(stress=stress, plant_overrides=lossy_battery_1, **kw)
        return solve(opt, pv=(4000.0,), load=0.0, prod=prod, soc=0.05, final=0.05)

    plain, stressed = run([0.0, 0.0]), run([0.3, 0.0])
    assert "inv_stress_cost_0" in stressed.columns and "inv_stress_cost_1" not in stressed.columns
    assert not any(c.startswith("inv_stress_cost") for c in plain.columns)
    p0 = col(stressed, "P_hybrid_inverter_0")
    assert p0.max() < col(plain, "P_hybrid_inverter_0").max() - 500  # power is shed off it
    # the piecewise cost is never below the quadratic it approximates (0.3 /kWh at 5 kW)
    quadratic = 5 * 0.3 * 0.5 * (np.abs(p0) / 5000) ** 2
    assert (col(stressed, "inv_stress_cost_0") >= quadratic - 1e-6).all()
    assert (col(stressed, "inv_stress_cost_0") > 0).any()


def test_result_columns_and_site_totals_with_one_curtailing_inverter():
    opt = build_multi(
        battery_inverter_index=(0, 1),
        eff_dc_ac=[0.95] * 2,
        eff_ac_dc=[0.95] * 2,
        curtail=[True, False],
        stress=[0.2, 0.0],
    )
    res = solve(opt, pv=(2000.0, 1000.0), prod=-0.1, soc=0.95)
    per_inverter = {
        "P_PV_0", "P_PV_1", "P_PV_curtailment_0", "P_hybrid_inverter_0", "P_hybrid_inverter_1",
        "inv_stress_cost_0",
    }  # fmt: skip
    totals = {"P_PV", "P_PV_curtailment", "P_hybrid_inverter", "P_PV_plant0", "P_PV_plant1"}
    assert (per_inverter | totals) <= set(res.columns)
    assert not {"P_PV_curtailment_1", "inv_stress_cost_1", "inv_stress_cost"} & set(res.columns)
    np.testing.assert_allclose(col(res, "P_PV"), col(res, "P_PV_0") + col(res, "P_PV_1"))
    np.testing.assert_allclose(col(res, "P_PV_curtailment"), col(res, "P_PV_curtailment_0"))
    np.testing.assert_allclose(
        col(res, "P_hybrid_inverter"),
        col(res, "P_hybrid_inverter_0") + col(res, "P_hybrid_inverter_1"),
    )
    # the site balance closes on the inverter totals
    np.testing.assert_allclose(
        col(res, "P_grid"), col(res, "P_Load") - col(res, "P_hybrid_inverter"), atol=1e-6
    )
    assert col(res, "P_PV_curtailment").max() > 100  # the scenario really curtails


def test_no_curtailment_flags_means_no_curtailment_columns_and_no_hybrid_flag_is_read():
    opt = build_multi(battery_inverter_index=(0, 1), curtail=[False, False])
    assert "inverter_is_hybrid" not in opt.plant_conf
    res = solve(opt)
    assert not [c for c in res.columns if c.startswith("P_PV_curtailment")]
    assert "P_hybrid_inverter" in res.columns
    # production configs keep the flag next to the lists: it must not change the answer
    flagged = build_multi(
        battery_inverter_index=(0, 1),
        curtail=[False, False],
        plant_overrides={"inverter_is_hybrid": False},
    )
    pd.testing.assert_frame_equal(solve(flagged), res)


def economic_objective(opt):
    """The solved objective without the #610 battery tie-break term.

    HiGHS does not always drive that 1e-9 per W term to zero: with some random seeds it
    returns a plan that cycles the second battery at no economic cost, and the objective
    then sits a few 1e-6 below the tie-broken one (macOS CI hit 6.2e-6).
    """
    pairs = zip(opt.vars["p_sto_pos"], opt.vars["p_sto_neg"], strict=True)
    tilt = sum(k * np.sum(pos.value - neg.value) for k, (pos, neg) in enumerate(pairs))
    return opt.prob.value + BATTERY_TIEBREAK_EPS * tilt


def test_two_lossless_hybrids_match_one_aggregate_hybrid_on_objective_and_grid_energy():
    # At efficiency 1.0 the flow between the two inverters is a real tie, so only the
    # objective and the site grid totals are compared, not the split.
    cost = [0.1, 0.1, 0.4, 0.4, 0.1, 0.1, 0.4, 0.4]
    multi = build_multi(battery_inverter_index=(0, 1))
    res = solve(multi, pv=(2500.0, 1500.0), load=900.0, cost=cost, soc=0.3)
    agg = build_multi(
        n_inv=1,
        battery_inverter_index=(0, 0),
        pv_plant_inverter_index=(0,),
        plant_overrides={
            "inverter_is_hybrid": True,
            "inverter_ac_output_max": 10000,
            "inverter_ac_input_max": 10000,
            "inverter_efficiency_dc_ac": 1.0,
            "inverter_efficiency_ac_dc": 1.0,
            "compute_curtailment": False,
            "inverter_stress_cost": 0.0,
        },
    )
    ref = solve(agg, pv=(4000.0,), load=900.0, cost=cost, soc=0.3)
    assert economic_objective(multi) == pytest.approx(economic_objective(agg), abs=1e-6)
    assert col(res, "P_grid").sum() == pytest.approx(col(ref, "P_grid").sum(), abs=1e-3)
    assert col(res, "P_hybrid_inverter").sum() == pytest.approx(
        col(ref, "P_hybrid_inverter").sum(), abs=1e-3
    )


def test_mixed_site_is_dpp_and_reuses_the_problem_on_the_next_run():
    curve = [[1000, 0.9], [3000, 0.95]]
    opt = build_multi(
        n_inv=3,
        battery_inverter_index=(0, 2),
        pv_plant_inverter_index=(0, 1),
        curves_dc_ac=[curve, [], []],
        curtail=[True, True, False],
        stress=[0.1, 0.1, 0.1],
        eff_dc_ac=[0.95] * 3,
        eff_ac_dc=[0.95] * 3,
        optim_overrides={"set_nodischarge_to_grid": True, "set_nocharge_from_grid": True},
    )
    solve(opt, pv=(1500.0, 800.0))
    prob = opt.prob
    assert prob.is_dcp(dpp=True)
    second = solve(opt, pv=(500.0, 2400.0))
    assert opt.prob is prob  # new PV values reach the cached problem through its Parameters
    np.testing.assert_allclose(col(second, "P_PV_1"), 2400.0)


def _fail_first_solve(calls):
    real_solve = cp.Problem.solve

    def fail_first(prob, *args, **kwargs):
        calls.append(prob)
        if len(calls) == 1:
            prob._status = "user_limit"
            prob._value = None
            return None
        return real_solve(prob, *args, **kwargs)

    return mock.patch.object(cp.Problem, "solve", new=fail_first)


def test_relaxed_retry_rebuilds_the_inverters_and_restores_the_cached_internals():
    curve = [[1000, 0.9], [3000, 0.95]]
    opt = build_multi(
        battery_inverter_index=(0, 1),
        curves_dc_ac=[curve, curve],
        eff_dc_ac=[0.95] * 2,
        eff_ac_dc=[0.95] * 2,
        curtail=[True, False],
        stress=[0.1, 0.1],
    )
    first = solve(opt, soc=0.9)
    cached = opt.vars["inverter_internals"]
    cached_ids = {v.id for d in cached for k, v in d.items() if not isinstance(v, list)}
    calls = []
    with _fail_first_solve(calls):
        df_input, p_pv, p_load = _frame(pv=3000.0)
        retried = opt.perform_dayahead_forecast_optim(
            df_input,
            p_pv,
            p_load,
            soc_init=[0.9, 0.9],
            soc_final=[0.9, 0.9],
            p_pv_plants=_plants(np.full(HORIZON, 2000.0), np.full(HORIZON, 1000.0)),
            pv_plant_inverters=[0, 1],
        )
    relaxed = [p for p in calls[1:] if p.status == "Optimal (Relaxed)"]
    assert relaxed and opt.optim_status == "Optimal (Relaxed)"
    # the rescue built its own internal variables ...
    assert not cached_ids & {v.id for v in relaxed[-1].variables()}
    # ... and the instance points at the cached problem's again afterwards
    assert opt.vars["inverter_internals"] is cached
    # published results come from the rescue, which solved the same problem
    for name in ("P_hybrid_inverter_0", "P_hybrid_inverter_1", "P_PV_curtailment_0"):
        assert col(retried, name) == pytest.approx(col(first, name), abs=1e-3)
    assert col(retried, "P_hybrid_inverter_0").any()


def test_the_horizon_can_change_between_runs():
    opt = build_multi(battery_inverter_index=(0, 1), curtail=[True, True])
    short = solve(opt)
    longer = solve(opt, n=HORIZON + 4)
    assert len(short) == HORIZON and len(longer) == HORIZON + 4
    assert all(len(v.value) == HORIZON + 4 for v in opt.vars["p_inverter"])
