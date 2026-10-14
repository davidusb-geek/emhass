"""Test matrix and reference rules for the EMHASS time-grid contract.

Everything here is independent of EMHASS: the expected start of a run is derived from
the real (UTC) timeline and the tz database, not from pandas' wall-clock rounding.

A grid point ("aligned instant") is a real instant whose local wall-clock time is a whole
multiple of the optimization step since local midnight. In the repeated hour of a fall-back
transition both occurrences of e.g. 02:30 are aligned instants; in a spring-forward gap the
missing wall times simply do not exist.
"""

from __future__ import annotations

import dataclasses
import datetime as dt

import pandas as pd
import pytz

UTC = dt.UTC

STEPS = (5, 15, 30, 60)
MODES = ("first", "nearest", "last")


@dataclasses.dataclass(frozen=True)
class Instant:
    zone: str
    utc: pd.Timestamp  # tz-aware UTC
    label: str  # e.g. "fall:fold1-mid"

    @property
    def local(self) -> pd.Timestamp:
        return self.utc.tz_convert(pytz.timezone(self.zone))

    def id(self) -> str:
        return f"{self.zone.split('/')[-1]}-{self.label}-{self.local:%m%d_%H%M%z}"


# Transitions in 2026 as encoded in the tz database (pytz 2026.5 / tzdata 2026c agree):
# (utc instant of the transition, length of the fold or gap)
TRANSITIONS = {
    "Europe/Berlin": [
        ("spring", "2026-03-29T01:00Z", 60),
        ("fall", "2026-10-25T01:00Z", 60),
    ],
    "Australia/Sydney": [
        ("fall", "2026-04-04T16:00Z", 60),
        ("spring", "2026-10-03T16:00Z", 60),
    ],
    # Negative DST in the tz database: winter time (GMT) carries dst() == -1 h.
    "Europe/Dublin": [
        ("spring", "2026-03-29T01:00Z", 60),
        ("fall", "2026-10-25T01:00Z", 60),
    ],
    # Ramadan 2026: +01 -> +00 on 15 Feb, back to +01 on 22 Mar (negative DST in pytz).
    "Africa/Casablanca": [
        ("fall", "2026-02-15T02:00Z", 60),
        ("spring", "2026-03-22T02:00Z", 60),
    ],
    # 30-minute DST.
    "Australia/Lord_Howe": [
        ("fall", "2026-04-04T15:00Z", 30),
        ("spring", "2026-10-03T15:30Z", 30),
    ],
    # +05:45, no DST: only grid alignment matters.
    "Asia/Kathmandu": [],
}

NORMAL_DAY_LOCAL = "2026-10-14 10:00"  # a Wednesday with no transition in any zone above


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s).tz_convert("UTC")


def instants_for(zone: str, step: int) -> list[Instant]:
    """Run instants for one zone and step: transition cases plus a normal day."""
    out: list[Instant] = []
    tz = pytz.timezone(zone)
    for kind, t_iso, length in TRANSITIONS[zone]:
        t = _ts(t_iso)
        span = pd.Timedelta(minutes=length)
        if kind == "fall":
            # The repeated wall-clock interval runs first in [t-span, t), then in [t, t+span).
            pts = {
                "fall:before": t - span - pd.Timedelta(minutes=7),
                "fall:fold1-start": t - span,
                "fall:fold1-mid": t - span / 2,
                "fall:fold2-start": t,
                "fall:fold2-mid": t + span / 2,
                "fall:after": t + span + pd.Timedelta(minutes=7),
                # the day before, same wall time as fold1-mid: the one-day window ends in
                # the repeated interval (#1163/#1164)
                "fall:eve": t - span / 2 - pd.Timedelta(days=1),
            }
        else:
            pts = {
                "spring:before": t - pd.Timedelta(minutes=7),
                "spring:after-gap": t,
                "spring:after-gap+7": t + pd.Timedelta(minutes=7),
                # the day before, at a wall time that will not exist tomorrow (#1163/#1164)
                "spring:eve": t + span / 2 - pd.Timedelta(days=1),
            }
        out += [Instant(zone, v, k) for k, v in pts.items()]
    base = tz.localize(pd.Timestamp(NORMAL_DAY_LOCAL).to_pydatetime())
    base = pd.Timestamp(base).tz_convert("UTC")
    lo = max(1, round(step * 0.4))  # rounds down under "nearest"
    hi = max(lo + 1, round(step * 0.6))  # rounds up under "nearest"
    out += [
        Instant(zone, base, "normal:on-boundary"),
        Instant(zone, base + pd.Timedelta(minutes=lo), "normal:early-in-step"),
        Instant(zone, base + pd.Timedelta(minutes=hi), "normal:late-in-step"),
    ]
    return out


def is_aligned(utc: pd.Timestamp, zone: str, step: int) -> bool:
    loc = utc.tz_convert(pytz.timezone(zone))
    return loc.second == 0 and (loc.hour * 60 + loc.minute) % step == 0


def aligned_neighbours(now_utc: pd.Timestamp, zone: str, step: int):
    """(previous-or-equal, next-or-equal) aligned instants on the real timeline."""
    minute = pd.Timedelta(minutes=1)
    t0 = now_utc.floor("min")
    prev = t0
    while not is_aligned(prev, zone, step):
        prev -= minute
    nxt = t0 if t0 == now_utc else t0 + minute
    while not is_aligned(nxt, zone, step):
        nxt += minute
    return prev, nxt


def expected_start(now_utc: pd.Timestamp, zone: str, step: int, mode: str) -> pd.Timestamp:
    """Start of the plan per method_ts_round, on the local wall-clock grid, by real time."""
    prev, nxt = aligned_neighbours(now_utc, zone, step)
    if mode == "first":
        return prev
    if mode == "last":
        return nxt
    # nearest; the matrix avoids exact ties
    return prev if (now_utc - prev) <= (nxt - now_utc) else nxt


# ---------------------------------------------------------------------------------------
# Known failure classes on v0.18.5 (= master 2f3dc75d). Predicates only describe the
# condition under which the failure is expected; they never feed into the assertions.
# ---------------------------------------------------------------------------------------


def _wall_time_problem(local_wall: pd.Timestamp, zone: str) -> str | None:
    tz = pytz.timezone(zone)
    try:
        tz.localize(local_wall.to_pydatetime(), is_dst=None)
    except pytz.AmbiguousTimeError:
        return "ambiguous"
    except pytz.NonExistentTimeError:
        return "nonexistent"
    return None


def wall_clock_floor_problem(now_utc: pd.Timestamp, zone: str, step: int) -> str | None:
    """Does flooring the *local wall time* (what get_forecast_dates does) land on an
    ambiguous or nonexistent wall time? Then pandas re-localizes with ambiguous='raise'."""
    loc = _emhass_local_now(now_utc, zone, "convert").tz_localize(None)
    return _wall_time_problem(loc.floor(f"{step}min"), zone)


def wall_clock_ceil_problem(now_utc: pd.Timestamp, zone: str, step: int) -> str | None:
    loc = _emhass_local_now(now_utc, zone, "now").tz_localize(None)
    return _wall_time_problem(loc.ceil(f"{step}min"), zone)


def _emhass_local_now(now_utc: pd.Timestamp, zone: str, source: str = "convert") -> pd.Timestamp:
    """Local "now" with microseconds stripped, the way EMHASS builds it.

    source="convert": pd.Timestamp(utc).tz_convert(tz).replace(microsecond=0)
                      (utils.get_forecast_dates)
    source="now":     pd.Timestamp.now(tz=tz).replace(microsecond=0)  (Forecast.__init__);
                      datetime.now(pytz_tz) always carries fold=0.
    Timestamp.replace() re-localizes the wall time using the fold attribute; inside a
    repeated interval it can therefore land on the other occurrence (pandas 3.0.6,
    pytz 2026.5): "now" loses the second occurrence in positive-DST zones, "convert" swaps
    occurrences in negative-DST zones (Europe/Dublin, Africa/Casablanca)."""
    tz = pytz.timezone(zone)
    if source == "now":
        d = tz.fromutc(now_utc.tz_localize(None).to_pydatetime().replace(tzinfo=tz))
        return pd.Timestamp(d).replace(microsecond=0)
    return now_utc.tz_convert(tz).replace(microsecond=0)


def replace_moves_instant(now_utc: pd.Timestamp, zone: str, source: str) -> bool:
    try:
        moved = _emhass_local_now(now_utc, zone, source).tz_convert("UTC")
    except Exception:  # noqa: BLE001
        return True
    return moved != now_utc.floor("s")


def forecast_init_start(now_utc: pd.Timestamp, zone: str, step: int, mode: str) -> pd.Timestamp:
    """Start that Forecast.__init__ uses today: nearest = round in UTC, first/last =
    floor/ceil of the local wall time."""
    loc = _emhass_local_now(now_utc, zone, "now")
    if mode == "nearest":
        return loc.tz_convert("UTC").round(f"{step}min")
    return (loc.floor(f"{step}min") if mode == "first" else loc.ceil(f"{step}min")).tz_convert(
        "UTC"
    )


def runtime_start(now_utc: pd.Timestamp, zone: str, step: int) -> pd.Timestamp:
    """Start of utils.get_forecast_dates: floor of the local wall time."""
    return _emhass_local_now(now_utc, zone, "convert").floor(f"{step}min").tz_convert("UTC")


def violates_rounding_rule(
    start: pd.Timestamp, now_utc: pd.Timestamp, step: int, mode: str
) -> bool:
    s = pd.Timedelta(minutes=step)
    if mode == "first":
        return not (start <= now_utc < start + s)
    if mode == "last":
        return not (start - s < now_utc <= start)
    return abs(start - now_utc) > s / 2


def wall_grid_conflict(now_utc: pd.Timestamp, zone: str, step: int, mode: str) -> bool:
    """No wall-clock-aligned start satisfies the real-time rounding rule here, because the
    aligned points around now are not one real step apart (e.g. Lord_Howe's 30-minute
    transition with a 60-minute step)."""
    return violates_rounding_rule(expected_start(now_utc, zone, step, mode), now_utc, step, mode)


def local_calendar_end(start_utc: pd.Timestamp, days: int, zone: str) -> pd.Timestamp:
    """Same wall time `days` local calendar days later; a nonexistent wall time moves
    forward by the gap, an ambiguous one takes the later instant (the #1164 contract)."""
    tz = pytz.timezone(zone)
    wall = start_utc.tz_convert(tz).tz_localize(None) + pd.Timedelta(days=days)
    try:
        return pd.Timestamp(tz.localize(wall.to_pydatetime(), is_dst=None)).tz_convert("UTC")
    except pytz.NonExistentTimeError:
        return pd.Timestamp(
            tz.normalize(tz.localize(wall.to_pydatetime(), is_dst=False))
        ).tz_convert("UTC")
    except pytz.AmbiguousTimeError:
        a = tz.localize(wall.to_pydatetime(), is_dst=True)
        b = tz.localize(wall.to_pydatetime(), is_dst=False)
        return pd.Timestamp(max(a, b)).tz_convert("UTC")


def steps_between(a: pd.Timestamp, b: pd.Timestamp, step: int) -> int:
    return int((b - a) / pd.Timedelta(minutes=step))


def forecast_grid_points(now_utc: pd.Timestamp, zone: str, step: int, mode: str, horizon: int):
    """The grid Forecast builds: its own start, window end after d local calendar days, where
    d is the #1166 extension measured on the runtime grid; cut to the horizon."""
    rt = runtime_start(now_utc, zone, step)
    d = 1
    while steps_between(rt, local_calendar_end(rt, d, zone), step) < horizon:
        d += 1
    start = forecast_init_start(now_utc, zone, step, mode)
    end = local_calendar_end(start, d, zone)
    s = pd.Timedelta(minutes=step)
    pts, t = [], start
    while t <= end - s and len(pts) < horizon:  # date_range(start, end - freq)
        pts.append(t)
        t += s
    return pts
