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

UTC = dt.UTC  # dt.UTC needs Python 3.11; the project supports 3.10

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


# ---------------------------------------------------------------------------------------
# Known failure classes on v0.18.5 (= master 2f3dc75d). Predicates only describe the
# condition under which the failure is expected; they never feed into the assertions.
# ---------------------------------------------------------------------------------------


def aligned_neighbours(now_utc: pd.Timestamp, zone: str, step: int):
    """(previous-or-equal, next-or-equal) wall-clock aligned instants on the real timeline,
    found by scanning minute by minute (independent of EMHASS' rounding code)."""
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
    """Independent oracle of the first step per method_ts_round: ``first``/``last`` are the
    previous/next wall-clock aligned instant by real time, ``nearest`` rounds on the UTC clock
    (the established contract; the matrix avoids exact ties)."""
    if mode == "nearest":
        return now_utc.floor("s").round(f"{step}min")
    prev, nxt = aligned_neighbours(now_utc, zone, step)
    return prev if mode == "first" else nxt


def forecast_init_start(now_utc: pd.Timestamp, zone: str, step: int, mode: str) -> pd.Timestamp:
    """Start of the grid of a run (independent oracle, see expected_start)."""
    return expected_start(now_utc, zone, step, mode)


def runtime_start(
    now_utc: pd.Timestamp, zone: str, step: int, mode: str = "nearest"
) -> pd.Timestamp:
    """Start of the runtime grid: the same start as Forecast."""
    return expected_start(now_utc, zone, step, mode)


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
    rt = runtime_start(now_utc, zone, step, mode)
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
