"""P4: every forecast value lands on the UTC instant it belongs to.

Inputs carry markers: value = minutes since 2026-01-01T00:00Z of the instant the value
belongs to. A value that arrives at the wrong grid point therefore names its true source.

* dict    timestamped runtime forecasts (one point every 5 min, the documented resample-mean
          rule turns them into marker(t) + (step - 5) / 2 at grid point t)
* csv     data_load_cost_forecast.csv with a timestamp column, one row per step
* csv-gap the same CSV ending two hours into the horizon: the uncovered steps must fail or
          stay empty, not silently repeat the last row (merge_asof without tolerance)
* open-meteo  a pre-seeded Open-Meteo cache file (unix timestamps, 15-min data) read through
          dayahead-optim; temp_air carries the marker
"""

from __future__ import annotations

import functools
import json
import os

import numpy as np
import pandas as pd
import pytest
import pytz
from cases import (
    MODES,
    STEPS,
    TRANSITIONS,
    forecast_grid_points,
    forecast_init_start,
    instants_for,
    is_aligned,
    runtime_start,
)
from harness import Env, bucket_mean_marker, list_runtimeparams, marker, marker_dict
from known import grid_not_utc_aligned, neg_dst_fold_in_horizon, start_floor_fails, xfail

COLS = ("p_pv_forecast", "p_load_forecast", "unit_load_cost", "unit_prod_price")


def _matrix(zones=None, steps=STEPS):
    for zone in zones or TRANSITIONS:
        for step in steps:
            for inst in instants_for(zone, step):
                for mode in MODES:
                    yield pytest.param(zone, step, mode, inst, id=f"{inst.id()}-{step}min-{mode}")


def _k2_shift(zone, step, mode, utc) -> str | None:
    try:
        if runtime_start(utc, zone, step) != forecast_init_start(utc, zone, step, mode):
            return (
                "K2: timestamped runtime forecasts are aligned to the get_forecast_dates grid "
                "and then used positionally on the Forecast grid -> every value one step off"
            )
    except Exception:  # noqa: BLE001
        return None
    return None


def _resample_bins_off_grid(zone, step, mode, utc) -> str | None:
    """_align_runtime_forecast_mapping resamples in local time from local midnight of the
    first input point with real-time bins; after a 30-min transition 60-min bins sit on
    :30 while the grid is wall-clock aligned on :00."""
    try:
        start = runtime_start(utc, zone, step)
    except Exception:  # noqa: BLE001
        return None
    tz = pytz.timezone(zone)
    first_point = (utc.floor("h") - pd.Timedelta(hours=3)).tz_convert(tz)
    bins = pd.date_range(first_point.normalize(), periods=6 * 1440 // step, freq=f"{step}min")
    if start not in bins:
        return (
            "K3 (design question): resample bins of timestamped forecasts run on real time "
            "from local midnight, the grid on wall-clock time; across a 30-min transition "
            "with a 60-min step they disagree by 30 min"
        )
    return None


# --------------------------------------------------------------------------- dict -------


@functools.cache
def _dict_run(zone, step, mode, utc):
    env = Env(zone, step, mode)
    try:
        series = marker_dict(utc)
        rp = {"prediction_horizon": 1440 // step, "optimization_time_step": step}
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
        idd, err = env.set_input(utc, rp, "naive-mpc-optim")
        if err:
            return {"error": err}
        df = env.prepare_optim_frame(idd, utc)
        if df is False or df is None:
            return {"error": "prepare_forecast_and_weather_data failed"}
        return {"error": None, "df": df}
    finally:
        env.close()


@pytest.mark.parametrize(("zone", "step", "mode", "inst"), list(_matrix()))
def test_p4_dict_values_land_on_their_instant(request, zone, step, mode, inst):
    xfail(
        request,
        start_floor_fails(zone, step, mode, inst.utc),
        _k2_shift(zone, step, mode, inst.utc),
        _resample_bins_off_grid(zone, step, mode, inst.utc),
        neg_dst_fold_in_horizon(zone, step, mode, inst.utc, 1440 // step),
    )
    r = _dict_run(zone, step, mode, inst.utc)
    assert r["error"] is None, r["error"]
    df = r["df"]
    expected = np.array([bucket_mean_marker(t, step) for t in df.index])
    for col in COLS:
        off = df[col].to_numpy(dtype=float) - expected
        bad = np.flatnonzero(~np.isclose(off, 0.0, rtol=0, atol=1e-6))  # NaN counts as wrong
        assert bad.size == 0, (
            f"{col}: {bad.size} of {len(df)} steps carry another instant's value; first at "
            f"{df.index[bad[0]]}, off by {off[bad[0]]:+.1f} min"
        )


# --------------------------------------------------------------------------- csv --------


def _write_csv(env: Env, utc: pd.Timestamp, step: int, end_after=None, fmt="utc"):
    tz = env.params_tz()
    start = utc.tz_convert(tz).normalize() - pd.Timedelta(days=1)
    end = (utc + (end_after if end_after is not None else pd.Timedelta(days=4))).tz_convert(tz)
    idx = pd.date_range(start, end, freq="5min")
    idx = idx[[is_aligned(t, env.zone, step) for t in idx]]  # wall-clock hours/half-hours
    stamps = [(t.tz_convert("UTC") if fmt == "utc" else t).isoformat() for t in idx]
    pd.DataFrame({"ts": stamps, "v": [marker(t) for t in idx]}).to_csv(
        env.data_path / "data_load_cost_forecast.csv", header=False, index=False
    )
    return idx


@functools.cache
def _csv_run(zone, step, mode, utc, gap: bool, fmt: str = "utc"):
    env = Env(zone, step, mode, load_cost_forecast_method="csv")
    try:
        horizon = 1440 // step
        idx = _write_csv(env, utc, step, pd.Timedelta(hours=2) if gap else None, fmt)
        rp = list_runtimeparams(step, horizon)
        del rp["load_cost_forecast"]
        idd, err = env.set_input(utc, rp, "naive-mpc-optim")
        if err:
            return {"error": err}
        df = env.prepare_optim_frame(idd, utc)
        if df is False or df is None:
            return {"error": "prepare_forecast_and_weather_data failed"}
        return {"error": None, "df": df, "csv_index": idx}
    except Exception as exc:  # noqa: BLE001 - an explicit failure is an observation
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        env.close()


def _offgrid(zone, step, mode, utc) -> str | None:
    """Some Forecast grid point is not a row of the (wall-clock aligned) CSV: nearest rounds
    in UTC (Kathmandu, Lord_Howe), or a real-time grid crosses a 30-min transition."""
    try:
        grid = forecast_grid_points(utc, zone, step, mode, 1440 // step)
    except Exception:  # noqa: BLE001
        return None
    if not all(is_aligned(t, zone, step) for t in grid):
        return (
            "K5: a Forecast grid point is not on the wall-clock grid of the CSV; the value "
            "becomes NaN or merge_asof(direction='nearest') without tolerance silently takes "
            "a neighbouring row"
        )
    return None


def _csv_regridded(zone, step, mode, utc) -> str | None:
    """set_df_index_freq() re-grids the timestamped CSV with asfreq(median step) from its
    first row on real time; wall-clock rows after a 30-min transition fall off that grid."""
    try:
        grid = forecast_grid_points(utc, zone, step, mode, 1440 // step)
    except Exception:  # noqa: BLE001
        return None
    tz = pytz.timezone(zone)
    first = utc.tz_convert(tz).normalize() - pd.Timedelta(days=1)
    real = {t.value for t in pd.date_range(first, periods=6 * 1440 // step, freq=f"{step}min")}
    if any(t.value not in real for t in grid):  # compare instants, not tz objects
        return (
            "K3 (design question): set_df_index_freq re-grids a timestamped CSV with "
            "asfreq() on real time from its first row; wall-clock rows after a 30-min "
            "transition fall off that grid and the values become NaN"
        )
    return None


def _mixed_offsets(zone, utc) -> str | None:
    tz = pytz.timezone(zone)
    start = utc.tz_convert(tz).normalize() - pd.Timedelta(days=1)
    idx = pd.date_range(start, (utc + pd.Timedelta(days=4)).tz_convert(tz), freq="15min")
    if len({t.utcoffset() for t in idx}) > 1:
        return (
            "K7: a CSV in local ISO time that spans a DST change has two UTC offsets; "
            "_load_forecast_data sniffs it with pd.to_datetime(..., errors='coerce') "
            "without utc=True, which raises 'Mixed timezones detected' in pandas 3"
        )
    return None


@pytest.mark.parametrize("fmt", ["utc", "local"])
@pytest.mark.parametrize(("zone", "step", "mode", "inst"), list(_matrix(steps=(15, 30, 60))))
def test_p4_csv_values_land_on_their_instant(request, zone, step, mode, inst, fmt):
    xfail(
        request,
        start_floor_fails(zone, step, mode, inst.utc),
        _offgrid(zone, step, mode, inst.utc),
        _csv_regridded(zone, step, mode, inst.utc),
        neg_dst_fold_in_horizon(zone, step, mode, inst.utc, 1440 // step),
        _mixed_offsets(zone, inst.utc) if fmt == "local" else None,
    )
    r = _csv_run(zone, step, mode, inst.utc, False, fmt)
    assert r["error"] is None, r["error"]
    df = r["df"]
    got = df["unit_load_cost"].to_numpy(dtype=float)
    expected = np.array([marker(t) for t in df.index])
    bad = np.flatnonzero(~np.isclose(got, expected, rtol=0, atol=1e-6))  # NaN counts as wrong
    assert bad.size == 0, (
        f"{bad.size} of {len(df)} steps carry another instant's value; first at "
        f"{df.index[bad[0]]}, off by {got[bad[0]] - expected[bad[0]]:+.1f} min"
    )


@pytest.mark.parametrize(
    ("zone", "step", "mode", "inst"),
    [p for p in _matrix(zones=["Europe/Berlin"], steps=(30,)) if "normal" in p.id],
)
def test_p4_csv_not_covering_the_horizon_is_not_filled_silently(request, zone, step, mode, inst):
    xfail(
        request,
        "K5: merge_asof(direction='nearest') without tolerance repeats the last CSV row "
        "over the uncovered part of the horizon, without an error or warning",
    )
    r = _csv_run(zone, step, mode, inst.utc, True)
    if r["error"] is not None:
        return  # explicit failure is acceptable
    df = r["df"]
    last = r["csv_index"][-1]
    uncovered = df.index[df.index > last]
    assert len(uncovered) > 0
    vals = df.loc[uncovered, "unit_load_cost"]
    assert vals.isna().all(), (
        f"{len(uncovered)} uncovered steps were filled with the value of {last} "
        f"(e.g. {uncovered[-1]} = {vals.iloc[-1]:.1f})"
    )


# ------------------------------------------------------------------------ open-meteo ----


def _seed_open_meteo(env: Env, utc: pd.Timestamp):
    idx = pd.date_range(
        utc.floor("D") - pd.Timedelta(days=1), utc + pd.Timedelta(days=5), freq="15min"
    )
    m = [marker(t) for t in idx]
    data = {
        "minutely_15": {
            "time": [int(t.timestamp()) for t in idx],
            "temperature_2m": m,
            "relative_humidity_2m": [50.0] * len(idx),
            "rain": [0.0] * len(idx),
            "cloud_cover": [0.0] * len(idx),
            "wind_speed_10m": [1.0] * len(idx),
            "shortwave_radiation_instant": [0.0] * len(idx),
            "diffuse_radiation_instant": [0.0] * len(idx),
            "direct_normal_irradiance_instant": [0.0] * len(idx),
        }
    }
    p = env.data_path / "cached-open-meteo-forecast-b.json"
    p.write_text(json.dumps(data))
    os.utime(p, (utc.timestamp(), utc.timestamp()))  # "fresh" at the frozen instant


@functools.cache
def _om_run(zone, step, mode, utc):
    env = Env(zone, step, mode, weather_forecast_method="open-meteo", set_use_pv=True)
    try:
        _seed_open_meteo(env, utc)
        rp = list_runtimeparams(step, None)
        del rp["pv_power_forecast"], rp["prediction_horizon"]
        idd, err = env.set_input(utc, rp, "dayahead-optim")
        if err:
            return {"error": err}
        return {"error": None, "weather": idd["df_weather"], "fcst": idd["fcst"].forecast_dates}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        env.close()


@pytest.mark.parametrize(("zone", "step", "mode", "inst"), list(_matrix(steps=(15, 30, 60))))
def test_p4_open_meteo_values_land_on_their_instant(request, zone, step, mode, inst):
    xfail(
        request,
        start_floor_fails(zone, step, mode, inst.utc),
        grid_not_utc_aligned(zone, step, mode, inst.utc),
    )
    r = _om_run(zone, step, mode, inst.utc)
    assert r["error"] is None, r["error"]
    w = r["weather"]
    assert list(w.index) == list(r["fcst"]), "weather index differs from the Forecast grid"
    got = w["temp_air"].to_numpy(dtype=float)
    expected = np.array([marker(t) for t in w.index])
    bad = np.flatnonzero(~np.isclose(got, expected, rtol=0, atol=1e-6))  # NaN counts as wrong
    assert bad.size == 0, (
        f"{bad.size} of {len(w)} steps carry another instant's value; first at "
        f"{w.index[bad[0]]}, off by {got[bad[0]] - expected[bad[0]]:+.1f} min"
    )
