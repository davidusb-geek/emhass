"""Known failure classes on v0.18.5 (= master 2f3dc75d), used to mark expected failures.

The predicates only describe *when* a failure is expected; they never feed into the
assertions. All markers are strict: a fixed class turns into XPASS(strict) = FAIL, which is
the signal to update this table.

K1   #1189: get_forecast_dates floors the local wall time onto an ambiguous (fold) or
     nonexistent (gap) wall time; pandas re-localizes with ambiguous/nonexistent="raise".
K1b  the same for the ceil() of method_ts_round="last" in Forecast.__init__.
K2   two grids: get_forecast_dates always floors the wall time, Forecast.__init__ rounds per
     method_ts_round (nearest in UTC). Runtime forecasts sized/aligned on the first grid are
     used positionally on the second. K2/P3: the #1166 extension measures the runtime grid,
     Forecast's own grid can be one step shorter (silently short plan).
K3   design question: 30-min DST with a 60-min step; the wall-clock grid is not one real step
     apart, so no wall-aligned start satisfies the real-time rounding rule.
K4   Timestamp.replace(microsecond=0) re-localizes the wall time and can switch the fold
     occurrence (pytz "now" carries fold=0; negative-DST zones swap even after tz_convert).
K5   merge_asof(direction="nearest") without tolerance in get_forecast_out_from_csv_or_list.
K6   price list path: astype("datetime64[ns, <tz name>]") turns the pytz index into zoneinfo
     (pandas 3); .loc with zoneinfo keys on the pytz frame fails in negative-DST folds.
K7   CSV forecasts with ISO timestamps in local time spanning a DST change (two UTC offsets)
     crash in the datetime sniffing of _load_forecast_data (pd.to_datetime without utc=True).
K8   Open-Meteo path re-rounds the Forecast index in UTC; grids that are not UTC-aligned
     collapse.
K9   _extract_daily_forecast splits days with between_time() on wall-clock time; rows of the
     second pass that are earlier than the fragment's first wall time are dropped
     (test_list_path.py).
"""

from __future__ import annotations

import pandas as pd
import pytest
from cases import (
    TRANSITIONS,
    forecast_init_start,
    replace_moves_instant,
    runtime_start,
    violates_rounding_rule,
    wall_clock_ceil_problem,
    wall_clock_floor_problem,
    wall_grid_conflict,
)

REF_1189 = "davidusb-geek/emhass#1189"


# ---------------------------------------------------------------------------------------


def start_floor_fails(zone, step, mode, utc) -> str | None:
    p = wall_clock_floor_problem(utc, zone, step)
    if p:
        return (
            f"K1 {REF_1189}: get_forecast_dates floors the local wall time to an {p} "
            "wall time and pandas re-localizes with ambiguous/nonexistent='raise'"
        )
    if mode == "last" and wall_clock_ceil_problem(utc, zone, step):
        return "K1b: Forecast.__init__ ceil()s the local wall time into a fold/gap"
    return None


def two_grids(zone, step, mode, utc) -> str | None:
    try:
        rt = runtime_start(utc, zone, step)
        fc = forecast_init_start(utc, zone, step, mode)
    except Exception:  # noqa: BLE001 - covered by K1
        return None
    if rt != fc:
        why = (
            "K2: runtime grid (get_forecast_dates, always floor of the local wall time) and "
            f"Forecast grid (method_ts_round={mode}) start on different instants"
        )
        if replace_moves_instant(utc, zone, "now") or replace_moves_instant(utc, zone, "convert"):
            why += "; K4: Timestamp.replace(microsecond=0) switched fold occurrence"
        return why
    return None


def start_breaks_rule(zone, step, mode, utc) -> str | None:
    try:
        start = forecast_init_start(utc, zone, step, mode)
    except Exception:  # noqa: BLE001 - covered by K1
        return None
    if not violates_rounding_rule(start, utc, step, mode):
        return None
    if wall_grid_conflict(utc, zone, step, mode):
        return (
            "K3 (design question): the wall-clock-aligned grid is not one real step apart "
            "here (30-min DST, 60-min step); a wall-aligned start cannot also satisfy the "
            "real-time rounding rule"
        )
    if replace_moves_instant(utc, zone, "now"):
        return "K4: Timestamp.replace(microsecond=0) in Forecast.__init__ switched fold occurrence"
    return "K3/K4: start breaks the rounding rule"


def xfail(request, *reasons):
    reasons = [r for r in reasons if r]
    if reasons:
        request.applymarker(pytest.mark.xfail(reason=" | ".join(reasons), strict=True))


NEGATIVE_DST_ZONES = ("Europe/Dublin", "Africa/Casablanca")


def neg_dst_fold_in_horizon(zone, step, mode, utc, horizon_steps) -> str | None:
    """K6: the list/runtime price path converts the Forecast index with
    astype("datetime64[ns, <tz name>]") (pandas 3 then uses zoneinfo, the dtype check never
    matches because pandas 3 stores "us") and looks the zoneinfo keys up in the pytz-indexed
    frame with .loc; in a negative-DST fold pandas cannot find the second occurrence."""
    if zone not in NEGATIVE_DST_ZONES:
        return None
    try:
        start = forecast_init_start(utc, zone, step, mode)
    except Exception:  # noqa: BLE001 - covered by K1
        return None
    end = start + pd.Timedelta(minutes=step) * horizon_steps
    for kind, t_iso, length in TRANSITIONS[zone]:
        if kind != "fall":
            continue
        t = pd.Timestamp(t_iso)
        if start < t + pd.Timedelta(minutes=length) and end > t:
            return (
                "K6: price list path - astype('datetime64[ns, <tz>]') switches pytz to zoneinfo "
                "(pandas 3), then df.loc[zoneinfo keys] on the pytz index raises KeyError for "
                "the second occurrence of a negative-DST fold"
            )
    return None


def grid_not_utc_aligned(zone, step, mode, utc) -> str | None:
    """K8: _get_weather_open_meteo re-rounds the Forecast index in UTC
    (index_utc.round(freq)); a wall-clock-aligned grid whose UTC offset is not a multiple
    of the step (Kathmandu +05:45 with 30/60 min, Lord_Howe +10:30 with 60 min) collapses
    into duplicates (round half to even) and the frame shrinks."""
    try:
        start = forecast_init_start(utc, zone, step, mode)
    except Exception:  # noqa: BLE001
        return None
    if start != start.round(f"{step}min"):
        return (
            "K8: Open-Meteo path rounds the Forecast index in UTC; the local grid is not "
            "UTC-aligned here, stamps collapse and the weather frame gets shorter"
        )
    return None


def extension_counts_other_grid(zone, step, mode, utc, horizon, delta=1) -> str | None:
    """K2 -> P3: the #1166 auto-extension compares prediction_horizon with the length of the
    runtime grid; Forecast builds its own grid from its own start, which can be one step
    shorter (start rounded up, window end shifted out of a spring-forward gap)."""
    from cases import local_calendar_end, steps_between

    try:
        rt = runtime_start(utc, zone, step)
        fc = forecast_init_start(utc, zone, step, mode)
    except Exception:  # noqa: BLE001 - covered by K1
        return None
    d = delta
    while steps_between(rt, local_calendar_end(rt, d, zone), step) < horizon:
        d += 1
    if steps_between(fc, local_calendar_end(fc, d, zone), step) < horizon:
        return (
            "K2/P3: the #1166 extension counts the runtime grid, but Forecast's own grid "
            "(other start) is shorter -> the plan is silently one step short"
        )
    return None
