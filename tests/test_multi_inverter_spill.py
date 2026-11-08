#!/usr/bin/env python
"""Tests for the PV-only clip, the hybrid spill and the lossless warning (#1150, PR 3b).

Above one inverter an inverter that may not curtail still has to place all its PV. A
PV-only one is clipped to what it can convert before the solve; a hybrid one gets a
priced ``P_PV_spill_{i}`` variable. Both build on the helpers of
``test_multi_inverter_optimization`` (``build_multi``, ``solve``).

New names are read with ``getattr`` so the tests fail on their behavioural assertion
against the code before the change, not on an ``AttributeError``.
"""

import contextlib
import logging
import re
from unittest import mock

import cvxpy as cp
import numpy as np
import pytest

from emhass.optimization import Optimization
from tests.test_multi_inverter_optimization import (
    CURVE,
    HORIZON,
    _fail_first_solve,
    _frame,
    _plants,
    build_multi,
    col,
    solve,
)

CLIP_MSG = "cannot convert all of its PV forecast"
LOSSLESS_MSG = "no conversion loss"


def _log(name, caplog, level=logging.INFO):
    log = logging.getLogger(name)
    log.handlers = [logging.NullHandler()]
    caplog.set_level(level, logger=name)
    return log


def _messages(caplog, needle):
    return [r.getMessage() for r in caplog.records if needle in r.getMessage()]


def _spill_site(**kwargs):
    """One hybrid (inverter 0, 5 kW cap, lossless, battery) and an inert second inverter."""
    kwargs.setdefault("battery_inverter_index", (0,))
    kwargs.setdefault("pv_plant_inverter_index", (0,))
    # a small weight on both directions keeps the battery from shuffling energy for free
    weights = {"weight_battery_charge": 0.05, "weight_battery_discharge": 0.05}
    kwargs["optim_overrides"] = {**weights, **kwargs.get("optim_overrides", {})}
    return build_multi(**kwargs)


def _spill(res, i=0):
    return col(res, f"P_PV_spill_{i}")


# --------------------------------------------------------------------------- #
# PV-only clip
# --------------------------------------------------------------------------- #


def _limit(spec):
    helper = getattr(Optimization, "_pv_only_clip_limit", None)
    assert helper is not None, "no clip-limit helper"
    return helper(spec)


def test_clip_limit_scalar_and_curve():
    assert _limit({"cap_out": 3000.0, "eff_dc_ac": 0.9, "curve_dc_ac": None}) == pytest.approx(
        3000 / 0.9
    )
    ac, dc = [0, 1000, 4000], [0, 1000 / 0.8, 4000 / 0.95]
    curve = {"cap_out": 3000.0, "eff_dc_ac": 1.0, "curve_dc_ac": (ac, dc)}
    # 3000 W AC lies two thirds along the second segment
    assert _limit(curve) == pytest.approx(dc[1] + (2 / 3) * (dc[2] - dc[1]))
    # a cap above the last AC point is held at the far end of the curve
    assert _limit({**curve, "cap_out": 9000.0}) == pytest.approx(dc[2])
    # a segment where DC stops rising while AC still does: the limit is that DC value
    flat_dc = ([0, 1000, 2000], [0, 2000, 2000])
    assert _limit({"cap_out": 1500.0, "eff_dc_ac": 1.0, "curve_dc_ac": flat_dc}) == 2000
    assert _limit({"cap_out": 0.0, "eff_dc_ac": 1.0, "curve_dc_ac": (ac, dc)}) == 0


def test_scalar_pv_only_inverter_is_clipped_and_the_totals_follow(caplog):
    log = _log("clip_scalar", caplog)
    opt = build_multi(cap_out=[5000, 3000], eff_dc_ac=[1.0, 0.9], logger=log)
    res = solve(opt, pv=(2000.0, 5000.0))  # inverter 1: PV-only, converts 3000 / 0.9 W of DC
    np.testing.assert_allclose(col(res, "P_PV_1"), 3000 / 0.9)
    np.testing.assert_allclose(col(res, "P_PV_0"), 2000.0)
    np.testing.assert_allclose(col(res, "P_PV"), col(res, "P_PV_0") + col(res, "P_PV_1"))
    np.testing.assert_allclose(opt.param_pv_forecast.value, col(res, "P_PV"))
    # the model sees the clipped series too, so the PV-only inverter runs at its cap
    assert col(res, "P_hybrid_inverter_1") == pytest.approx(3000, abs=1e-3)
    lines = _messages(caplog, CLIP_MSG)
    assert len(lines) == 1
    assert "Inverter 1" in lines[0] and "8 steps" in lines[0]
    assert "3333 W" in lines[0] and "peak 5000 W" in lines[0]


def test_curved_pv_only_inverter_is_clipped_to_the_curve_limit():
    opt = build_multi(cap_out=[5000, 3000], curves_dc_ac=[[], CURVE])
    res = solve(opt, pv=(2000.0, 5000.0))
    dc_1000, dc_4000 = 1000 / 0.8, 4000 / 0.95
    limit = dc_1000 + (2000 / 3000) * (dc_4000 - dc_1000)
    np.testing.assert_allclose(col(res, "P_PV_1"), limit)
    assert col(res, "P_hybrid_inverter_1") == pytest.approx(3000, abs=1e-2)
    np.testing.assert_allclose(col(res, "P_PV"), col(res, "P_PV_0") + col(res, "P_PV_1"))


def test_a_partly_over_limit_series_is_clipped_only_in_those_steps(caplog):
    log = _log("clip_partial", caplog)
    opt = build_multi(cap_out=[5000, 3000], logger=log)
    pv_1 = np.array([1000, 1000, 4000, 4000, 1000, 1000, 1000, 1000], dtype=float)
    res = solve(opt, pv=(2000.0, pv_1))
    np.testing.assert_allclose(col(res, "P_PV_1"), np.minimum(pv_1, 3000.0))
    assert "clipped in 2 steps to 3000 W (peak 4000 W)" in _messages(caplog, CLIP_MSG)[0]


def test_the_clip_line_is_logged_once_and_repeats_go_to_debug(caplog):
    log = _log("clip_repeat", caplog, logging.DEBUG)
    opt = build_multi(cap_out=[5000, 3000], logger=log)
    for _ in range(3):  # an MPC loop calls the optimizer again every few minutes
        res = solve(opt, pv=(2000.0, 5000.0))
        np.testing.assert_allclose(col(res, "P_PV_1"), 3000 / 1.0)
    records = [r for r in caplog.records if CLIP_MSG in r.getMessage()]
    assert [r.levelname for r in records] == ["INFO", "DEBUG", "DEBUG"]


def test_nothing_is_clipped_or_logged_when_the_pv_fits(caplog):
    log = _log("clip_none", caplog)
    opt = build_multi(cap_out=[5000, 3000], logger=log)
    res = solve(opt, pv=(2000.0, 2500.0))
    np.testing.assert_allclose(col(res, "P_PV_1"), 2500.0)
    assert not _messages(caplog, CLIP_MSG)


def test_a_curtailing_pv_only_inverter_is_not_clipped(caplog):
    log = _log("clip_curtail", caplog)
    opt = build_multi(cap_out=[5000, 3000], curtail=[False, True], logger=log)
    res = solve(opt, pv=(2000.0, 5000.0))
    np.testing.assert_allclose(col(res, "P_PV_1"), 5000.0)  # the full forecast
    assert col(res, "P_PV_curtailment_1").min() >= 2000 - 1e-3  # curtailment takes the rest
    assert not _messages(caplog, CLIP_MSG)


def test_a_hybrid_with_the_battery_off_is_a_pv_only_inverter_and_gets_the_clip(caplog):
    log = _log("clip_battery_off", caplog)
    opt = build_multi(cap_out=[3000, 3000], use_battery=False, logger=log)
    assert [s["shape"] for s in opt.inverter_specs] == ["pv_only", "pv_only"]
    res = solve(opt, pv=(4000.0, 1000.0))
    np.testing.assert_allclose(col(res, "P_PV_0"), 3000.0)
    np.testing.assert_allclose(col(res, "P_PV_1"), 1000.0)
    assert len(_messages(caplog, CLIP_MSG)) == 1


# --------------------------------------------------------------------------- #
# Hybrid spill
# --------------------------------------------------------------------------- #


def test_a_hybrid_above_its_cap_with_a_full_battery_is_feasible_and_spills():
    # 3a: PV 7 kW into a 5 kW hybrid with a full battery and no curtailment was infeasible.
    opt = _spill_site()
    res = solve(opt, pv=(7000.0,), load=1000.0, soc=0.95)
    assert "P_PV_spill_0" in res.columns
    np.testing.assert_allclose(_spill(res), 2000.0, atol=1e-3)
    np.testing.assert_allclose(col(res, "P_hybrid_inverter_0"), 5000.0, atol=1e-3)
    np.testing.assert_allclose(col(res, "P_PV_0"), 7000.0)
    np.testing.assert_allclose(col(res, "P_PV"), 7000.0)  # the forecast, spill included


@pytest.mark.parametrize(
    "curtail,battery_inverters,pv_inverters,expected",
    [
        ([False, False], (0, 1), (0, 1), {"P_PV_spill_0", "P_PV_spill_1"}),
        ([True, False], (0, 1), (0, 1), {"P_PV_spill_1"}),
        ([False, False], (0,), (0, 1), {"P_PV_spill_0"}),  # inverter 1 is PV-only
        ([False, False], (1,), (0,), set()),  # inverter 0 PV-only, inverter 1 battery-only
        ([False, True], (0, 1), (0, 1), {"P_PV_spill_0"}),
    ],
)
def test_spill_columns_exist_only_for_a_hybrid_that_does_not_curtail(
    curtail, battery_inverters, pv_inverters, expected
):
    opt = build_multi(
        battery_inverter_index=battery_inverters,
        pv_plant_inverter_index=pv_inverters,
        curtail=curtail,
    )
    res = solve(opt, pv=(1000.0,) * len(pv_inverters))
    assert {c for c in res.columns if c.startswith("P_PV_spill")} == expected
    spill_vars = getattr(opt, "vars", {}).get("p_pv_spill_inv")
    assert [v is not None for v in spill_vars] == [f"P_PV_spill_{i}" in expected for i in range(2)]


def test_the_battery_fills_before_any_pv_is_spilled():
    # 2 kW of DC surplus over 4 h is 8 kWh; the battery has 4.5 kWh of room (0.5 to 0.95).
    full = solve(_spill_site(), pv=(7000.0,), load=300.0, soc=0.95, final=0.95)
    room = solve(_spill_site(), pv=(7000.0,), load=300.0, soc=0.5, final=0.95)
    assert _spill(full).sum() * 0.5 == pytest.approx(8000, abs=1)
    assert _spill(room).sum() * 0.5 == pytest.approx(8000 - 4500, abs=1)
    assert col(room, "SOC_opt").max() == pytest.approx(0.95, abs=1e-6)


def test_spill_stays_zero_when_the_pv_fits():
    res = solve(_spill_site(), pv=(4000.0,), load=300.0, soc=0.95)
    assert _spill(res) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize(
    "costfun,cost,prod,expected",
    [
        (
            "profit",
            0.2,
            0.05,
            0.002,
        ),  # positive prices never compete with spill: the floor, doubled
        ("cost", 0.3, 0.05, 0.002),
        ("profit", 5.0, 0.05, 0.002),  # a price spike does not raise it
        ("profit", 0.2, -0.5, 1.0),  # a negative export price counts by its size, doubled
        ("profit", -0.7, 0.05, 1.4),  # so does a negative import price
        ("profit", -0.7, -0.9, 1.8),  # the more negative of the two
        # the floor is a tie-break and is not scaled by SELF_CONSUMPTION_BIGM
        ("self-consumption", 0.2, 0.05, 0.002),
        ("self-consumption", -0.7, 0.05, 1400.0),
        # self-consumption weighs export at its plain price, so a negative export price
        # counts unscaled (not times the big-M)
        ("self-consumption", 0.2, -0.5, 1.0),
        ("self-consumption", 0.2, -5.0, 10.0),
        # the cost objective has no export term, so a negative export price does not count
        ("cost", 0.3, -0.5, 0.002),
        ("cost", -0.7, -0.9, 1.4),
        ("profit", 0.0, 0.0, 0.002),
    ],
)
def test_spill_price_follows_the_most_negative_price(costfun, cost, prod, expected):
    opt = _spill_site()
    opt.costfun = costfun
    solve(opt, pv=(7000.0,), load=1000.0, cost=cost, prod=prod, soc=0.95)
    price = getattr(opt, "param_spill_price", None)
    assert price is not None and price.shape == (HORIZON,)
    np.testing.assert_allclose(price.value, expected)  # the same price in every step here


@pytest.mark.parametrize(
    "cost,prod",
    [
        (np.r_[np.full(3, 0.2), -0.7, np.full(HORIZON - 4, 0.2)], 0.05),  # one negative import step
        (
            0.2,
            np.r_[np.full(5, 0.05), -0.3, np.full(HORIZON - 6, 0.05)],
        ),  # one negative export step
    ],
    ids=["import", "export"],
)
def test_spill_price_is_set_per_step_and_only_a_negative_step_raises_it(cost, prod):
    opt = _spill_site()
    solve(opt, pv=(7000.0,), load=1000.0, cost=cost, prod=prod, soc=0.95)
    expected = 2 * np.maximum.reduce(
        [-np.asarray(cost) * np.ones(HORIZON), -prod * np.ones(HORIZON)]
    )
    expected = np.maximum(expected, 0.002)
    price = opt.param_spill_price.value
    np.testing.assert_allclose(price, expected)
    assert (price > 0.002 + 1e-9).sum() == 1  # exactly the one negative step
    assert price.max() > 0.5


def test_spill_price_follows_new_prices_on_the_cached_problem():
    opt = _spill_site()
    solve(opt, pv=(7000.0,), prod=-0.2, soc=0.95)
    prob = opt.prob
    np.testing.assert_allclose(opt.param_spill_price.value, 0.4)
    solve(opt, pv=(7000.0,), prod=-0.5, soc=0.95)
    assert opt.prob is prob and opt.prob.is_dcp(dpp=True)
    np.testing.assert_allclose(opt.param_spill_price.value, 1.0)
    solve(opt, pv=(7000.0,), prod=0.05, soc=0.95)  # back to the floor
    np.testing.assert_allclose(opt.param_spill_price.value, 0.002)


@pytest.mark.parametrize(
    "stress,segments,expected",
    [
        ([0.01, 0.0], 10, 0.002 + 0.01 * 1.9),  # steepest of 10 segments: 2 - 1/10
        ([0.01, 0.0], 2, 0.002 + 0.01 * 1.5),
        ([0.01, 0.03], 10, 0.002 + 0.03 * 1.9),  # two spilling hybrids: the larger rate
        ([0.0, 0.0], 10, 0.002),
    ],
)
def test_the_spilling_hybrids_stress_cost_is_added_to_the_spill_price(stress, segments, expected):
    opt = build_multi(
        battery_inverter_index=(0, 1),
        stress=stress,
        plant_overrides={"inverter_stress_segments": segments},
    )
    solve(opt, pv=(7000.0, 7000.0), load=1000.0)
    np.testing.assert_allclose(opt.param_spill_price.value, expected)


def test_a_stress_cost_on_an_inverter_that_cannot_spill_adds_nothing():
    # inverter 0 curtails (no spill variable), inverter 1 is PV only
    opt = build_multi(curtail=[True, False], battery_inverter_index=(0,), stress=[0.05, 0.05])
    solve(opt, pv=(1000.0, 1000.0))
    np.testing.assert_allclose(opt.param_spill_price.value, 0.002)


def test_a_stress_cost_does_not_make_the_hybrid_spill_pv_that_fits():
    # 4 kW of PV into a 5 kW hybrid with a full battery: it all fits. A stress cost on the
    # inverter makes passing less power cheaper, so a spill priced only at the floor would
    # idle the inverter and throw the PV away. The stress rate in the price stops that.
    def spilled(patch_rate):
        opt = _spill_site(stress=[0.5, 0.0])
        rate = mock.patch.object(Optimization, "_spill_stress_rate", return_value=0.0)
        with rate if patch_rate else contextlib.nullcontext():
            res = solve(opt, pv=(4000.0,), load=300.0, soc=0.95, final=0.95)
        return _spill(res).sum() * 0.5, col(res, "P_hybrid_inverter_0")

    kept, ac = spilled(False)
    assert kept == pytest.approx(0, abs=1e-3)
    np.testing.assert_allclose(ac, 4000.0, atol=1e-3)
    assert spilled(True)[0] > 1000  # counterfactual: without the stress rate it dodges


def _constraints_with(opt, *variables):
    ids = {v.id for v in variables}
    return [c for c in opt.prob.constraints if ids <= {v.id for v in c.variables()}]


def test_the_export_bound_for_no_discharge_to_grid_includes_spill():
    opt = _spill_site(optim_overrides={"set_nodischarge_to_grid": True})
    res = solve(opt, pv=(7000.0,), load=1000.0, soc=0.95)
    spill = getattr(opt, "vars", {}).get("p_pv_spill_inv", [None])[0]
    assert spill is not None
    # export is tied to the PV that was neither curtailed nor spilled
    assert _constraints_with(opt, opt.vars["p_grid_neg"], spill)
    assert (-col(res, "P_grid_neg") <= 7000 - _spill(res) + 1e-3).all()
    # without the option no such bound exists
    plain = _spill_site()
    solve(plain, pv=(7000.0,), load=1000.0, soc=0.95)
    assert not _constraints_with(plain, plain.vars["p_grid_neg"], plain.vars["p_pv_spill_inv"][0])


def test_the_relaxed_retry_and_the_next_run_keep_the_spill_variable():
    opt = _spill_site()
    first = solve(opt, pv=(7000.0,), load=1000.0, soc=0.95)
    spill_var = opt.vars["p_pv_spill_inv"][0]
    calls = []
    with _fail_first_solve(calls):
        df_input, p_pv, p_load = _frame(pv=7000.0, load=1000.0)
        retried = opt.perform_dayahead_forecast_optim(df_input, p_pv, p_load, 0.95, 0.95)
    assert opt.optim_status == "Optimal (Relaxed)"
    assert any(p.status == "Optimal (Relaxed)" for p in calls[1:])
    assert opt.vars["p_pv_spill_inv"][0] is spill_var
    np.testing.assert_allclose(_spill(retried), _spill(first), atol=1e-3)
    assert _spill(retried).min() > 1000
    # the cached problem is intact: the next run reuses it and still spills
    again = solve(opt, pv=(7000.0,), load=1000.0, soc=0.95)
    assert _spill(again).min() > 1000


def test_the_horizon_can_change_with_a_spill_variable():
    opt = _spill_site()
    price = getattr(opt, "param_spill_price", None)
    short = solve(opt, pv=(7000.0,), load=1000.0, soc=0.95)
    longer = solve(opt, pv=(7000.0,), load=1000.0, soc=0.95, n=HORIZON + 4)
    assert len(short) == HORIZON and len(longer) == HORIZON + 4
    assert opt.vars["p_pv_spill_inv"][0].shape == (HORIZON + 4,)
    np.testing.assert_allclose(_spill(longer), 2000.0, atol=1e-3)
    # the per-step price follows the horizon: a new vector of the new length, set for the run
    assert price is not None and opt.param_spill_price.shape == (HORIZON + 4,)
    assert (opt.param_spill_price.value > 0).all()


def test_the_spill_log_is_bounded_to_ten_steps_plus_a_count(caplog):
    log = _log("spill_log_long", caplog)
    opt = _spill_site(logger=log)
    solve(opt, pv=(7000.0,), load=1000.0, soc=0.95, n=16)
    lines = _messages(caplog, "spilled PV")
    assert len(lines) == 1
    assert "Inverter 0" in lines[0] and "16 steps" in lines[0]
    assert len(re.findall(r"\d\+0\d:\d\d: 2000 W", lines[0])) == 10  # timestamp: N W
    assert lines[0].endswith("and 6 more")
    assert "could not or would not take" in lines[0]


def test_the_spill_log_lists_every_step_when_there_are_few_and_is_silent_without_spill(caplog):
    log = _log("spill_log_short", caplog)
    opt = _spill_site(logger=log)
    pv = np.array([7000, 7000, 7000, 1000, 1000, 1000, 1000, 1000], dtype=float)
    solve(opt, pv=(pv,), load=1000.0, soc=0.95)
    lines = _messages(caplog, "spilled PV")
    assert len(lines) == 1 and "3 steps" in lines[0] and "more" not in lines[0]
    assert len(re.findall(r": 2000 W", lines[0])) == 3
    caplog.clear()
    solve(opt, pv=(1000.0,), load=1000.0, soc=0.95)
    assert not _messages(caplog, "spilled PV")


# --------------------------------------------------------------------------- #
# Spill is the last resort: no battery cycling to shrink it
# --------------------------------------------------------------------------- #

# A 20 kWh battery at 95 % efficiency under a 5 kW hybrid: the site where moving energy
# through the battery shrinks the spill, and every cycle costs 10 % in losses.
BELL_BATTERY = {
    "battery_nominal_energy_capacity": [20000],
    "battery_charge_efficiency": [0.95],
    "battery_discharge_efficiency": [0.95],
    "battery_minimum_state_of_charge": [0.1],
    "battery_maximum_state_of_charge": [1.0],
    "battery_discharge_power_max": [5000],
    "battery_charge_power_max": [5000],
}
STEP_H = 0.5


def _throughput_wh(res):
    return float(np.abs(col(res, "P_batt")).sum() * STEP_H)


def _discharged_wh(res):
    return float(np.clip(col(res, "P_batt"), 0, None).sum() * STEP_H)


def _real_cost(res, weight):
    """What the user pays: grid import less export income, plus the battery weights."""
    grid = (
        (
            col(res, "unit_load_cost") * col(res, "P_grid_pos")
            + col(res, "unit_prod_price") * col(res, "P_grid_neg")
        )
        * STEP_H
        / 1000
    )
    batt = col(res, "P_batt")
    weights = weight * (np.clip(batt, 0, None) - np.clip(batt, None, 0)) * STEP_H / 1000
    return float((grid + weights).sum())


def _bell_day(curtail, weight, cost=0.30, n=24, prod=0.05, costfun=None, total_pv_sell=False):
    """Bell-shaped PV peaking at 9 kW over a day of half hours on a 5 kW hybrid."""
    t = np.arange(n)
    pv = np.clip(9000 * np.exp(-(((t - 12) / 5.0) ** 2)), 0, None)
    opt = build_multi(
        battery_inverter_index=(0,),
        pv_plant_inverter_index=(0,),
        curtail=[curtail, False],
        eff_dc_ac=[0.95, 0.95],
        eff_ac_dc=[0.95, 0.95],
        plant_overrides=BELL_BATTERY,
        optim_overrides={"weight_battery_charge": weight, "weight_battery_discharge": weight},
    )
    if costfun is not None:
        opt.costfun = costfun
    if total_pv_sell:
        opt.optim_conf["set_total_pv_sell"] = True
    load = 800 + 600 * (t % 4 == 0)
    res = solve(opt, pv=(pv,), load=load, cost=cost, prod=prod, soc=0.9, final=0.9, n=n)
    return opt, res


@pytest.mark.parametrize(
    "costfun,cost,prod,expected",
    [
        ("profit", -0.7, 0.05, 0.002),  # grid import is not priced, so it does not count
        ("profit", -0.7, -0.9, 1.8),  # a negative export price still does
        ("cost", -0.7, 0.05, 0.002),
        ("self-consumption", -0.7, 0.05, 1400.0),  # self-consumption still prices import
    ],
)
def test_with_total_pv_sell_a_negative_import_price_does_not_raise_the_spill_price(
    costfun, cost, prod, expected
):
    opt = _spill_site()
    opt.costfun = costfun
    opt.optim_conf["set_total_pv_sell"] = True
    solve(opt, pv=(7000.0,), load=1000.0, cost=cost, prod=prod, soc=0.95)
    np.testing.assert_allclose(opt.param_spill_price.value, expected)


def test_with_total_pv_sell_a_negative_import_day_plans_the_battery_like_the_curtailing_twin():
    # Grid import is not priced with set_total_pv_sell, so a negative tariff all day gives
    # spill nothing to compete with and the battery plan is the curtailing twin's. Counting
    # the import price there cycled 7829 Wh against the twin's 0 at weight 0.05.
    def day(curtail):
        return _bell_day(curtail=curtail, weight=0.05, cost=-0.2, total_pv_sell=True)

    _, spill_res = day(False)
    _, twin_res = day(True)
    assert _discharged_wh(spill_res) == pytest.approx(_discharged_wh(twin_res), abs=1)
    assert _throughput_wh(spill_res) == pytest.approx(_throughput_wh(twin_res), abs=1)


@pytest.mark.parametrize("weight", [0.0, 0.05])
def test_a_self_consumption_clipping_day_plans_the_battery_like_the_curtailing_twin(weight):
    # Under self-consumption only import gets the big-M, so the floor stays a tie-break and
    # the battery plan is the curtailing twin's. A floor scaled by the big-M (2.0 per kWh)
    # paid the battery to make room for PV: 7829 Wh out against the twin's 3438 at 0.05.
    opt, spill_res = _bell_day(curtail=False, weight=weight, costfun="self-consumption")
    _, twin_res = _bell_day(curtail=True, weight=weight, costfun="self-consumption")
    assert _discharged_wh(spill_res) == pytest.approx(_discharged_wh(twin_res), abs=1)
    assert _throughput_wh(spill_res) == pytest.approx(_throughput_wh(twin_res), abs=1)
    assert _real_cost(spill_res, weight) == pytest.approx(_real_cost(twin_res, weight), abs=1e-3)
    np.testing.assert_allclose(opt.param_spill_price.value, 0.002)


@pytest.mark.parametrize("weight", [0.0, 0.05])
def test_a_clipping_day_plans_the_battery_like_the_curtailing_twin(weight):
    # Spill is priced as a tie-break on a positive-price day, so the battery does what the
    # user's weights say, exactly as it does when the same inverter may curtail.
    opt, spill_res = _bell_day(curtail=False, weight=weight)
    _, twin_res = _bell_day(curtail=True, weight=weight)
    np.testing.assert_allclose(opt.param_spill_price.value, 0.002)
    assert _discharged_wh(spill_res) == pytest.approx(_discharged_wh(twin_res), abs=1)
    assert _throughput_wh(spill_res) == pytest.approx(_throughput_wh(twin_res), abs=1)
    assert _real_cost(spill_res, weight) == pytest.approx(_real_cost(twin_res, weight), abs=1e-3)
    # counterfactual: the day does cycle the battery (7.8 kWh out at weight 0, 3.4 at 0.05)
    # and, at 0.05, it does clip: the twin curtails and the spill plan spills the same energy
    assert _discharged_wh(twin_res) > 3000
    if weight > 0:
        twin_curtailed = col(twin_res, "P_PV_curtailment_0").sum() * STEP_H
        assert twin_curtailed > 4000
        assert _spill(spill_res).sum() * STEP_H == pytest.approx(twin_curtailed, abs=1)


def test_one_negative_export_step_hours_after_the_clipping_does_not_cause_cycling():
    # The only negative price is an evening export step, long after the last clipping step.
    # It prices spill at 2 x 0.05 in that step alone; the clipping steps stay at the floor,
    # so the battery plan is the curtailing twin's. A price set from the horizon-wide worst
    # step would put 0.1 on every clipping step and empty the battery in the morning to
    # refill it from PV that would otherwise be spilled.
    prod = np.full(24, 0.05)
    prod[22] = -0.05
    opt, spill_res = _bell_day(curtail=False, weight=0.05, prod=prod)
    _, twin_res = _bell_day(curtail=True, weight=0.05, prod=prod)
    assert _discharged_wh(spill_res) == pytest.approx(_discharged_wh(twin_res), abs=1)
    assert _throughput_wh(spill_res) == pytest.approx(_throughput_wh(twin_res), abs=1)
    assert _real_cost(spill_res, 0.05) == pytest.approx(_real_cost(twin_res, 0.05), abs=1e-3)
    price = opt.param_spill_price.value
    assert price[22] == pytest.approx(0.1)
    np.testing.assert_allclose(np.delete(price, 22), 0.002)
    # counterfactual: the same plan priced at 0.1 in every step (what one horizon-wide
    # price gave) cycles the battery far beyond the twin
    flat = mock.patch.object(Optimization, "_spill_stress_rate", return_value=0.098)
    with flat:
        scalar_opt, scalar_res = _bell_day(curtail=False, weight=0.05, prod=prod)
    assert scalar_opt.param_spill_price.value[0] == pytest.approx(0.1)
    assert _discharged_wh(scalar_res) > _discharged_wh(twin_res) + 1000


@pytest.mark.parametrize("spike", [0.30, 1.0, 5.0], ids=["flat", "spike_1", "spike_5"])
def test_a_price_spike_does_not_make_the_battery_burn_energy_at_weight_zero(spike):
    # A full 20 kWh battery that must end full under 7 kW of PV on a 5 kW hybrid: 2 kW is
    # spilled every step whatever happens. Discharging and refilling from PV that would
    # have been spilled must not pay, at any price level and with no battery weight.
    cost = np.r_[np.full(HORIZON - 1, 0.30), spike]
    opt = _spill_site(
        cap_in=[0, 0],
        plant_overrides=BELL_BATTERY,
        optim_overrides={"weight_battery_charge": 0.0, "weight_battery_discharge": 0.0},
    )
    res = solve(opt, pv=(7000.0,), load=1000.0, cost=cost, prod=0.05, soc=1.0, final=1.0)
    assert _discharged_wh(res) < 1
    assert _spill(res).sum() * STEP_H == pytest.approx(8000, abs=1)  # still all forced spill
    np.testing.assert_allclose(opt.param_spill_price.value, 0.002)  # the spike is not in the price


def test_negative_export_price_spills_only_what_is_forced_and_burns_nothing():
    # Spill is priced at twice the loss, so it is never a way to avoid exporting at a loss
    # (the curtailing twin curtails all of it), but a forced spill still happens and still
    # does not pay to cycle the battery.
    kwargs = {
        "cap_in": [0, 0],
        "plant_overrides": BELL_BATTERY,
        "optim_overrides": {"weight_battery_charge": 0.0, "weight_battery_discharge": 0.0},
    }
    opt = _spill_site(**kwargs)
    res = solve(opt, pv=(7000.0,), load=1000.0, cost=0.30, prod=-0.2, soc=1.0, final=1.0)
    np.testing.assert_allclose(opt.param_spill_price.value, 0.4)
    np.testing.assert_allclose(_spill(res), 2000.0, atol=1e-3)  # the forced overflow only
    assert _discharged_wh(res) < 1
    assert col(res, "P_grid_neg") == pytest.approx(-4000, abs=1e-3)  # the loss is still taken
    twin = build_multi(
        battery_inverter_index=(0,),
        pv_plant_inverter_index=(0,),
        curtail=[True, False],
        cap_in=[0, 0],
        plant_overrides=BELL_BATTERY,
    )
    twin_res = solve(twin, pv=(7000.0,), load=1000.0, cost=0.30, prod=-0.2, soc=1.0, final=1.0)
    assert col(twin_res, "P_PV_curtailment_0").sum() * STEP_H > 2 * _spill(res).sum() * STEP_H


def _overlap(res):
    return (_spill(res) > 1e-3) & (col(res, "P_batt") > 1e-3)


def test_no_step_spills_and_discharges_the_battery_on_the_same_inverter():
    # 7 kW of PV into a 5 kW hybrid with a 10 kWh battery, default efficiency, no weights:
    # discharging 4.75 kW into the clipping window and refilling later used to be free.
    opt = build_multi(
        battery_inverter_index=(0,),
        pv_plant_inverter_index=(0,),
        plant_overrides={**BELL_BATTERY, "battery_nominal_energy_capacity": [10000]},
    )
    res = solve(opt, pv=(7000.0,), load=1000.0, soc=0.5, final=0.5)
    assert (_spill(res) > 1000).all()  # it does spill, every step
    assert not _overlap(res).any()
    assert _throughput_wh(res) < 1  # and the battery stays still
    # on a bell day with a weight, discharge and spill both happen, but never together
    _, day = _bell_day(curtail=False, weight=0.05)
    assert (_spill(day) > 1000).any() and (col(day, "P_batt") > 1000).any()
    assert not _overlap(day).any()


def test_the_relaxed_retry_keeps_the_no_discharge_while_spilling_rule():
    opt = _spill_site(
        optim_overrides={"weight_battery_charge": 0.0, "weight_battery_discharge": 0.0}
    )
    calls = []
    with _fail_first_solve(calls):
        df_input, p_pv, p_load = _frame(pv=7000.0, load=1000.0)
        res = opt.perform_dayahead_forecast_optim(df_input, p_pv, p_load, 0.95, 0.95)
    assert opt.optim_status == "Optimal (Relaxed)"
    assert _spill(res).min() > 1000 and not _overlap(res).any()


def test_a_battery_above_its_soc_target_cannot_shed_energy_through_a_clipping_inverter():
    # Side effect of the rule above, kept on purpose: a battery that has to come down to a
    # lower soc_final cannot do it in a step where its inverter spills PV. It stays where
    # it is and the target is missed (the soft penalty), instead of throwing PV away to
    # burn stored energy.
    kwargs = {
        "cap_in": [0, 0],
        "plant_overrides": BELL_BATTERY,
        "optim_overrides": {"weight_battery_charge": 0.0, "weight_battery_discharge": 0.0},
    }
    clipped = solve(_spill_site(**kwargs), pv=(7000.0,), load=1000.0, cost=0.30, soc=1.0, final=0.7)
    assert _discharged_wh(clipped) < 1
    assert col(clipped, "SOC_opt")[-1] == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(_spill(clipped), 2000.0, atol=1e-3)
    # counterfactual: with PV that fits the inverter the same battery does reach its target
    fits = solve(_spill_site(**kwargs), pv=(3000.0,), load=1000.0, cost=0.30, soc=1.0, final=0.7)
    assert col(fits, "SOC_opt")[-1] == pytest.approx(0.7, abs=1e-3)
    assert _discharged_wh(fits) > 5000


def test_the_pv_that_must_be_spilled_goes_at_the_end_of_the_horizon():
    # 8 kWh of overflow and 4.5 kWh of battery room: the battery fills first in time and
    # the spill lands in the later steps, as curtailment does (the #342 tie-break). Without
    # the tie-break the solver spills in the early steps and charges later.
    opt = _spill_site(
        optim_overrides={"weight_battery_charge": 0.0, "weight_battery_discharge": 0.0}
    )
    res = solve(opt, pv=(7000.0,), load=300.0, soc=0.5, final=0.95)
    np.testing.assert_allclose(_spill(res), [0, 0, 0, 0, 1000, 2000, 2000, 2000], atol=1e-3)
    np.testing.assert_allclose(col(res, "P_batt")[:5], [-2000] * 4 + [-1000], atol=1e-3)


# --------------------------------------------------------------------------- #
# Lossless warning
# --------------------------------------------------------------------------- #

LOSSY_CURVE = [[1000, 1.0], [4000, 0.99]]
LOSSLESS_CURVE = [[1000, 1.0], [4000, 1.0]]


def _warnings(caplog):
    return _messages(caplog, LOSSLESS_MSG)


def test_two_lossless_inverters_with_batteries_warn_once_per_optimization(caplog):
    log = _log("lossless_pair", caplog, logging.WARNING)
    opt = build_multi(battery_inverter_index=(0, 1), logger=log)
    solve(opt)
    solve(opt)
    lines = _warnings(caplog)
    assert len(lines) == 1
    assert "[0, 1]" in lines[0] and "arbitrary" in lines[0] and "efficiency" in lines[0]


def test_a_string_inverter_with_a_battery_only_inverter_does_not_warn(caplog):
    log = _log("lossless_string", caplog, logging.WARNING)
    build_multi(battery_inverter_index=(1,), pv_plant_inverter_index=(0,), logger=log)
    assert not _warnings(caplog)


def test_one_lossless_inverter_with_the_only_lossy_neighbour_does_not_warn(caplog):
    # inverter 0 is lossless both ways but it is the only one: nothing to trade with
    log = _log("lossless_single", caplog, logging.WARNING)
    build_multi(
        battery_inverter_index=(0, 1), eff_dc_ac=[1.0, 0.9], eff_ac_dc=[1.0, 0.9], logger=log
    )
    assert not _warnings(caplog)


@pytest.mark.parametrize(
    "overrides,warns",
    [
        ({"cap_in": [0, 0]}, False),  # no AC can reach either battery
        ({"cap_out": [0, 0]}, False),  # no AC can leave either unit
        ({"cap_in": [5000, 0], "cap_out": [5000, 0]}, False),  # inverter 1 moves no AC at all
        ({"cap_in": [5000, 0]}, True),  # inverter 1 out, inverter 0 in: a real pair
    ],
)
def test_an_inverter_with_a_zero_cap_is_no_way_for_energy_to_travel(caplog, overrides, warns):
    log = _log(f"lossless_cap_{warns}_{len(overrides)}", caplog, logging.WARNING)
    build_multi(battery_inverter_index=(0, 1), logger=log, **overrides)
    assert bool(_warnings(caplog)) is warns


def test_a_third_inverter_joins_the_pair_in_the_warning(caplog):
    # lossless out: 0 and 1; lossless in: 0 only; the pair (1, 0) exists, inverter 2 is lossy
    log = _log("lossless_three", caplog, logging.WARNING)
    build_multi(
        n_inv=3,
        battery_inverter_index=(0, 1, 2),
        pv_plant_inverter_index=(0, 1),
        eff_dc_ac=[1.0, 1.0, 0.9],
        eff_ac_dc=[1.0, 0.9, 0.9],
        logger=log,
    )
    lines = _warnings(caplog)
    assert len(lines) == 1 and "[0, 1]" in lines[0]


def test_an_inactive_battery_does_not_count(caplog):
    log = _log("lossless_inactive", caplog, logging.WARNING)
    build_multi(battery_inverter_index=(0, 1), use_battery=False, logger=log)
    assert not _warnings(caplog)


@pytest.mark.parametrize("curve,warns", [(LOSSLESS_CURVE, True), (LOSSY_CURVE, False)])
def test_a_curve_counts_as_lossless_only_when_every_point_is(caplog, curve, warns):
    log = _log(f"lossless_curve_{warns}", caplog, logging.WARNING)
    build_multi(
        battery_inverter_index=(0, 1),
        eff_dc_ac=[0.9, 0.9],
        eff_ac_dc=[0.9, 0.9],
        curves_dc_ac=[curve, []],
        curves_ac_dc=[[], curve],
        logger=log,
    )
    assert bool(_warnings(caplog)) is warns


def test_the_warning_is_not_raised_by_realistic_efficiencies(caplog):
    log = _log("lossless_realistic", caplog, logging.WARNING)
    build_multi(
        battery_inverter_index=(0, 1), eff_dc_ac=[0.97] * 2, eff_ac_dc=[0.97] * 2, logger=log
    )
    assert not _warnings(caplog)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_the_spill_problem_is_dpp_and_has_no_spill_term_without_a_hybrid():
    opt = _spill_site()
    solve(opt, pv=(7000.0,), load=1000.0, soc=0.95)
    assert opt.prob.is_dcp(dpp=True)
    none = build_multi(battery_inverter_index=(1,), pv_plant_inverter_index=(0,))
    solve(none, pv=(1000.0,))
    assert all(v is None for v in none.vars["p_pv_spill_inv"])
    assert not [v for v in none.prob.variables() if v.name().startswith("p_pv_spill")]
    assert isinstance(getattr(opt, "param_spill_price", None), cp.Parameter)


def test_a_per_call_input_is_not_mutated_by_the_clip():
    opt = build_multi(cap_out=[5000, 3000])
    index_pv = np.full(HORIZON, 5000.0)
    df_input, _, p_load = _frame(load=300.0)
    p_pv = _plants(np.full(HORIZON, 2000.0), index_pv)
    p_pv.index = df_input.index
    opt.perform_dayahead_forecast_optim(
        df_input,
        p_pv.sum(axis=1),
        p_load,
        soc_init=0.5,
        soc_final=0.5,
        p_pv_plants=p_pv,
        pv_plant_inverters=[0, 1],
    )
    assert p_pv["P_PV_plant1"].max() == 5000.0  # the caller's frame keeps the raw forecast
    assert opt.param_pv_inv[1].value.max() < 5000.0


def test_the_clip_and_spill_are_not_used_at_one_inverter():
    with mock.patch.object(Optimization, "_warn_lossless_battery_inverters", create=True) as warn:
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
    warn.assert_not_called()
    assert not hasattr(opt, "param_spill_price") and "p_pv_spill_inv" not in opt.vars
