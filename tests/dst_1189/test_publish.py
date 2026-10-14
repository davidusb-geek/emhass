"""P5 (and P6 for publishing): publish-data uses the instants of the computed plan.

A plan is computed with naive-mpc-optim shortly before the transition (horizon one day,
so it spans the repeated hour) and written to opt_res_latest.csv exactly like the add-on
does. publish-data is then called on its own, as Home Assistant triggers it, at instants
inside and after the repeated hour. Nothing is posted (dont_post=True).

Expected: the published row is the plan row chosen by method_ts_round on the real
timeline (first: last row <= now, last: first row >= now, nearest: closest row), and its
timestamp is one of the plan's timestamps. publish-data only reads the saved plan, so it
needs no new time grid.
"""

from __future__ import annotations

import functools

import pandas as pd
import pytest
from cases import MODES
from harness import Env, list_runtimeparams, utc_list
from known import neg_dst_fold_in_horizon, start_floor_fails, xfail

STEP = 30

# zone -> (instant of the plan, publish instants)
SCENARIOS = {
    "Europe/Berlin": (
        "2026-10-24T23:50Z",  # 01:50 CEST
        [
            "2026-10-25T00:10Z",
            "2026-10-25T00:40Z",
            "2026-10-25T01:10Z",
            "2026-10-25T01:40Z",
            "2026-10-25T02:10Z",
        ],
    ),
    "Australia/Sydney": (
        "2026-04-04T14:50Z",  # 01:50 AEDT
        [
            "2026-04-04T15:10Z",
            "2026-04-04T15:40Z",
            "2026-04-04T16:10Z",
            "2026-04-04T16:40Z",
            "2026-04-04T17:10Z",
        ],
    ),
    "Europe/Dublin": (
        "2026-10-24T23:50Z",  # 00:50 IST
        [
            "2026-10-25T00:10Z",
            "2026-10-25T00:40Z",
            "2026-10-25T01:10Z",
            "2026-10-25T01:40Z",
            "2026-10-25T02:10Z",
        ],
    ),
    "Europe/Berlin normal day": (
        "2026-10-14T08:05Z",
        ["2026-10-14T08:05Z", "2026-10-14T08:20Z", "2026-10-14T08:40Z"],
    ),
}


def _cases():
    for key, (plan_at, pubs) in SCENARIOS.items():
        zone = key.split()[0]
        for mode in MODES:
            for pub in pubs:
                local = pd.Timestamp(pub).tz_convert(zone)
                yield pytest.param(
                    zone,
                    mode,
                    pd.Timestamp(plan_at),
                    pd.Timestamp(pub),
                    id=f"{zone.split('/')[-1]}-{local:%m%d_%H%M%z}-{mode}",
                )


@functools.cache
def _plan(zone, mode, plan_at):
    env = Env(zone, STEP, mode)
    idd, err = env.set_input(plan_at, list_runtimeparams(STEP, 1440 // STEP), "naive-mpc-optim")
    if err:
        return env, None, err
    res = env.naive_mpc(idd, plan_at)
    if res is None or res is False or "P_Load" not in res:
        return env, None, f"optimization failed: {res}"
    return env, utc_list(res.index), None


def _expected_row(plan_index, now, mode):
    if mode == "first":
        return max(t for t in plan_index if t <= now)
    if mode == "last":
        return min(t for t in plan_index if t >= now)
    return min(plan_index, key=lambda t: abs(t - now))


@pytest.mark.parametrize(("zone", "mode", "plan_at", "pub_at"), list(_cases()))
def test_p5_publish_uses_the_plan_instants(request, zone, mode, plan_at, pub_at):
    reason = start_floor_fails(zone, STEP, mode, pub_at)
    if reason:
        reason = (
            "K1 in publish-data: set_input_data_dict runs treat_runtimeparams -> "
            "get_forecast_dates although publish only reads the saved plan; " + reason
        )
    plan_reason = start_floor_fails(zone, STEP, mode, plan_at)
    if plan_reason:
        plan_reason = "plan itself: " + plan_reason
    xfail(
        request,
        reason,
        plan_reason,
        neg_dst_fold_in_horizon(zone, STEP, mode, plan_at, 1440 // STEP),
    )
    env, plan_index, err = _plan(zone, mode, plan_at)
    assert err is None, f"plan at {plan_at}: {err}"
    res, err = env.publish(pub_at)
    assert err is None, err
    published = pd.Timestamp(res.index[0]).tz_convert("UTC")
    assert published in plan_index
    assert published == _expected_row(plan_index, pub_at, mode), (
        f"published {published}, expected {_expected_row(plan_index, pub_at, mode)} "
        f"for now={pub_at} ({mode})"
    )
