"""P1, P2, P3, P6: one time grid per run, end to end through set_input_data_dict.

For every zone x step x method_ts_round x run instant a naive-mpc-optim input is built from
plain list forecasts (no history needed), with prediction_horizon = one nominal day of steps.

P1  runtime grid (utils.get_forecast_dates, used by treat_runtimeparams to size and align
    runtime forecasts) == Forecast grid (Forecast.forecast_dates) == optimizer input index.
P2  optimizer index is unique, strictly increasing, constant real step.
P3  the requested horizon is delivered in full, or the run fails explicitly - never a
    silently shorter plan.
P6  the first step obeys method_ts_round on the real timeline: "first" may contain the
    running step, "nearest" is at most half a step away, "last" is not in the past.
P7  the first step is exactly the one an independent oracle gives (cases.expected_start: the
    previous/next wall-clock aligned instant found by scanning real time, "nearest" rounded on
    the UTC clock), not merely within the P6 bounds.
"""

from __future__ import annotations

import functools

import pandas as pd
import pytest
from cases import MODES, STEPS, TRANSITIONS, expected_start, instants_for
from harness import Env, list_runtimeparams, utc_list


def _matrix():
    for zone in TRANSITIONS:
        for step in STEPS:
            for inst in instants_for(zone, step):
                for mode in MODES:
                    yield pytest.param(zone, step, mode, inst, id=f"{inst.id()}-{step}min-{mode}")


_ENVS: dict = {}


def _env(zone, step, mode) -> Env:
    key = (zone, step, mode)
    if key not in _ENVS:
        _ENVS[key] = Env(zone, step, mode)
    return _ENVS[key]


@functools.cache
def _run(zone, step, mode, utc):
    env = _env(zone, step, mode)
    horizon = 1440 // step
    idd, err = env.set_input(utc, list_runtimeparams(step, horizon), "naive-mpc-optim")
    out = {"error": err, "horizon": horizon}
    if err:
        return out
    delta = idd["fcst"].optim_conf["delta_forecast_daily"].days
    out["runtime"] = env.runtime_grid(utc, delta, horizon)
    out["forecast"] = utc_list(idd["fcst"].forecast_dates)
    out["optim"] = utc_list(idd["df_input_data_dayahead"].index)
    return out


@pytest.mark.parametrize(("zone", "step", "mode", "inst"), list(_matrix()))
def test_p1_one_grid(request, zone, step, mode, inst):
    r = _run(zone, step, mode, inst.utc)
    assert r["error"] is None, r["error"]
    assert r["runtime"] == r["forecast"], (
        f"runtime starts {r['runtime'][0]}, Forecast starts {r['forecast'][0]}"
    )
    assert r["forecast"] == r["optim"]


@pytest.mark.parametrize(("zone", "step", "mode", "inst"), list(_matrix()))
def test_p2_grid_well_formed(request, zone, step, mode, inst):
    r = _run(zone, step, mode, inst.utc)
    assert r["error"] is None, r["error"]
    idx = pd.DatetimeIndex(r["optim"])
    assert idx.is_unique
    assert idx.is_monotonic_increasing
    assert set(idx.to_series().diff().dropna()) <= {pd.Timedelta(minutes=step)}


@pytest.mark.parametrize(("zone", "step", "mode", "inst"), list(_matrix()))
def test_p3_horizon_complete_or_explicit_error(request, zone, step, mode, inst):
    r = _run(zone, step, mode, inst.utc)
    if r["error"] is not None:
        return  # an explicit failure satisfies P3 (P1/P2/P6 report it)
    assert len(r["optim"]) == r["horizon"]


@pytest.mark.parametrize(("zone", "step", "mode", "inst"), list(_matrix()))
def test_p6_first_step_follows_rounding_rule(request, zone, step, mode, inst):
    r = _run(zone, step, mode, inst.utc)
    assert r["error"] is None, r["error"]
    start, now, s = r["optim"][0], inst.utc, pd.Timedelta(minutes=step)
    if mode == "first":
        assert start <= now < start + s
    elif mode == "last":
        assert start - s < now <= start
    else:
        assert abs(start - now) <= s / 2


@pytest.mark.parametrize(("zone", "step", "mode", "inst"), list(_matrix()))
def test_p7_first_step_is_the_independent_expectation(zone, step, mode, inst):
    if zone == "Australia/Lord_Howe" and step == 60:
        pytest.skip(
            "K3 (design question): around its 30-minute DST change the wall-clock aligned hours "
            "are not one real hour apart, so first/last have two defensible neighbours; the "
            "P6 bounds still apply"
        )
    r = _run(zone, step, mode, inst.utc)
    assert r["error"] is None, r["error"]
    assert r["optim"][0] == expected_start(inst.utc, zone, step, mode)
