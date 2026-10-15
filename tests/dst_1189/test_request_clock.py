"""One request, one grid: the request clock is read once (#1189).

``treat_runtimeparams`` reads ``utils._get_now()`` and ``Forecast.__init__`` used to read
``pd.Timestamp.now()`` again. A request that crossed a time-step boundary between the two
reads (the Home Assistant call in between takes time) aligned its timestamped runtime
forecasts to one grid and used them on the other, one step off.

Here the first read happens 1 s before a boundary and the second 1 s after it. Every value
of the timestamped runtime forecasts encodes its own UTC instant, so a value on the wrong
step names its true source. The boundaries and the expected values are computed without
``get_forecast_start``: grid points are hour-aligned wall-clock instants given in UTC, and a
``nearest`` boundary is a half step after a multiple of the step on the UTC clock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from harness import Env, bucket_mean_marker, marker_dict

STEPS = (5, 15, 30, 60)
MODES = ("first", "nearest", "last")
# (zone, id, UTC instant of an hour-aligned local wall-clock time)
BASES = (
    ("Europe/Berlin", "normal", "2026-10-14T08:00Z"),
    ("Europe/Berlin", "fold-first-pass", "2026-10-25T00:00Z"),
    ("Europe/Berlin", "fold-second-pass", "2026-10-25T01:00Z"),
    ("Europe/Berlin", "spring", "2026-03-29T00:00Z"),
    ("Australia/Sydney", "normal", "2026-10-13T23:00Z"),
    ("Australia/Sydney", "fold-first-pass", "2027-04-03T15:00Z"),
    ("Australia/Sydney", "fold-second-pass", "2027-04-03T16:00Z"),
    ("Europe/Dublin", "fold-first-pass", "2026-10-25T00:00Z"),
    ("Europe/Dublin", "fold-second-pass", "2026-10-25T01:00Z"),
    ("Asia/Kathmandu", "normal", "2026-10-14T04:15Z"),
)


def _boundary(base: pd.Timestamp, k: int, step: int, mode: str) -> pd.Timestamp:
    t = base + pd.Timedelta(minutes=k * step)
    if mode == "nearest":  # rounds on the UTC clock: the change is half a step after a multiple
        return t.floor(f"{step}min") + pd.Timedelta(minutes=step) / 2
    return t  # first/last change on the wall-clock grid


def _cases():
    for zone, label, base in BASES:
        for step in STEPS:
            for mode in MODES:
                for k in (0, 1):
                    yield pytest.param(
                        zone,
                        step,
                        mode,
                        pd.Timestamp(base),
                        k,
                        id=f"{zone.split('/')[-1]}-{label}-{step}min-{mode}-k{k}",
                    )


@pytest.mark.parametrize(("zone", "step", "mode", "base", "k"), list(_cases()))
def test_request_crossing_a_step_boundary_keeps_one_grid(zone, step, mode, base, k):
    edge = _boundary(base, k, step, mode)
    before, after = edge - pd.Timedelta(seconds=1), edge + pd.Timedelta(seconds=1)
    env = Env(zone, step, mode)
    try:
        rp = {"prediction_horizon": 1440 // step, "optimization_time_step": step}
        series = marker_dict(before)
        rp.update(
            dict.fromkeys(
                (
                    "pv_power_forecast",
                    "load_power_forecast",
                    "load_cost_forecast",
                    "prod_price_forecast",
                ),
                series,
            )
        )
        idd, err = env.set_input(before, rp, "naive-mpc-optim", forecast_now=after)
        assert err is None, err
        df = env.prepare_optim_frame(idd, after)
        assert df is not False
        expected = np.array([bucket_mean_marker(t, step) for t in df.index])
        for col in ("p_pv_forecast", "p_load_forecast", "unit_load_cost", "unit_prod_price"):
            got = df[col].to_numpy(dtype=float)
            bad = np.flatnonzero(~np.isclose(got, expected, rtol=0, atol=1e-6))
            assert bad.size == 0, (
                f"{col}: {bad.size} of {len(df)} steps carry another instant's value, "
                f"first at {df.index[bad[0]]}"
            )
    finally:
        env.close()
