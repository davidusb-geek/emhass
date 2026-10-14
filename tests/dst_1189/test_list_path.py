"""P4 at method level: Forecast.get_forecast_out_from_csv_or_list maps a list onto any valid
grid, including grids that start inside a repeated wall-clock interval.

The start computation is taken out of the picture: the grid is handed in directly. This
isolates the per-day split in _extract_daily_forecast, which selects rows with
between_time(first_hour, last_hour) on wall-clock times. A day fragment that starts in the
first pass of the repeated hour has later rows (the second pass) whose wall time is
*earlier* than first_hour; between_time drops them and the frame is one or more rows short
of its index ("Shape of passed values is (44, 1), indices imply (45, 1)" in #1189).
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest
import pytz
from cases import TRANSITIONS
from harness import Env, list_runtimeparams, marker

STEPS = (15, 30, 60)
ZONES = ("Europe/Berlin", "Australia/Sydney", "Europe/Dublin", "Australia/Lord_Howe")


def _fold_starts(zone, step):
    for kind, t_iso, length in TRANSITIONS[zone]:
        if kind != "fall":
            continue
        t = pd.Timestamp(t_iso)
        span = pd.Timedelta(minutes=length)
        yield "fold1-start", t - span
        yield "fold2-start", t
        if length > step:
            yield "fold1-mid", t - span + pd.Timedelta(minutes=step)
            yield "fold2-mid", t + pd.Timedelta(minutes=step)
        yield "before", t - span - pd.Timedelta(minutes=step)


def _cases():
    for zone in ZONES:
        for step in STEPS:
            for label, start in _fold_starts(zone, step):
                loc = start.tz_convert(pytz.timezone(zone))
                yield pytest.param(
                    zone,
                    step,
                    label,
                    start,
                    id=f"{zone.split('/')[-1]}-{label}-{loc:%H%M%z}-{step}min",
                )


@functools.cache
def _forecast(zone, step):
    """A Forecast object built by the regular pipeline on a quiet day (only its methods are
    used; the grid under test is passed explicitly)."""
    env = Env(zone, step, "first")
    now = pd.Timestamp("2026-10-14T00:00Z")
    idd, err = env.set_input(now, list_runtimeparams(step, 1440 // step), "naive-mpc-optim")
    assert err is None, err
    return env, idd["fcst"]


def _wall_time_dropped(grid: pd.DatetimeIndex) -> bool:
    """Does a later row of the first local day carry a wall time before the first row's?"""
    day = grid[grid.day == grid[0].day]
    first = day[0].hour * 60 + day[0].minute
    return any((t.hour * 60 + t.minute) < first for t in day[1:])


@pytest.mark.parametrize(("zone", "step", "label", "start"), list(_cases()))
def test_list_onto_grid_starting_in_fold(request, zone, step, label, start):
    tz = pytz.timezone(zone)
    grid = pd.date_range(start.tz_convert(tz), periods=1440 // step, freq=f"{step}min")
    if _wall_time_dropped(grid):
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    "K9: _extract_daily_forecast selects the day with between_time() on wall "
                    "time; rows of the second pass that are earlier than first_hour are dropped"
                ),
            )
        )
    env, fcst = _forecast(zone, step)
    df_final = pd.DataFrame(index=grid)
    values = [marker(t) for t in grid]
    out = fcst.get_forecast_out_from_csv_or_list(df_final, grid, None, data_list=values)
    got = np.ravel(out.to_numpy(dtype=float))
    assert len(got) == len(grid)
    exp = np.array(values)
    bad = np.flatnonzero(~np.isclose(got, exp, rtol=0, atol=1e-6))  # NaN counts as wrong
    assert bad.size == 0, f"{bad.size} steps carry another instant's value, first at {grid[bad[0]]}"
