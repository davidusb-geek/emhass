"""Open-Meteo irradiance -> pvlib -> P_PV stays on its UTC instant (investigation of #1287/#1330).

``test_alignment`` proves that a ``temp_air`` marker lands on its instant. This module follows
the irradiance columns through the rest of the PV chain: an Open-Meteo cache (unix timestamps,
15-minute clear-sky GHI/DNI/DHI computed at the coordinates of #1287) is read by
``dayahead-optim`` and the resulting ``p_pv_forecast`` is compared, step by step, with the same
pvlib model evaluated on a pure-UTC 15-minute frame. A displacement of the irradiance, of the
pvlib index or of the grid by one hour changes the result by hundreds of watts (asserted below
as a precondition), so a one-hour shift cannot pass unnoticed.

The optimiser reads each PV value as the mean power of its step. With a step longer than 15
minutes the expected value is therefore the trapezoidal mean of the 15-minute pvlib power over
the step, not the power at the step start (``_step_average``); a 15-minute step is unchanged.
"""

from __future__ import annotations

import functools
import json
import os

import numpy as np
import pandas as pd
import pytest
from harness import Env, list_runtimeparams
from pvlib.irradiance import get_total_irradiance
from pvlib.location import Location

LAT, LON = 55.65, 11.67
MODES = ("first", "nearest", "last")

# (zone, run instant, label): fall-back day, the day after, an ordinary winter day, spring-forward
# control, UTC control. Run instants are a few minutes past a local hour.
RUNS = (
    ("Europe/Copenhagen", "2026-10-24T22:05Z", "fallback-day"),
    ("Europe/Copenhagen", "2026-10-25T23:05Z", "day-after"),
    ("Europe/Copenhagen", "2026-12-10T23:05Z", "winter"),
    ("Europe/Copenhagen", "2026-03-28T22:05Z", "spring-forward"),
    ("UTC", "2026-10-25T23:05Z", "utc"),
)


@functools.cache
def _clear_sky(utc: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(
        utc.floor("D") - pd.Timedelta(days=1), utc + pd.Timedelta(days=5), freq="15min", tz="UTC"
    )
    return Location(LAT, LON).get_clearsky(idx, model="ineichen")[["ghi", "dni", "dhi"]]


def _seed(env: Env, utc: pd.Timestamp) -> pd.DataFrame:
    cs = _clear_sky(utc)
    n = len(cs)
    doc = {
        "minutely_15": {
            "time": [int(t.timestamp()) for t in cs.index],
            "temperature_2m": [10.0] * n,
            "relative_humidity_2m": [50.0] * n,
            "rain": [0.0] * n,
            "cloud_cover": [0.0] * n,
            "wind_speed_10m": [2.0] * n,
            "shortwave_radiation_instant": cs["ghi"].round(3).tolist(),
            "diffuse_radiation_instant": cs["dhi"].round(3).tolist(),
            "direct_normal_irradiance_instant": cs["dni"].round(3).tolist(),
        }
    }
    p = env.data_path / "cached-open-meteo-forecast-b.json"
    p.write_text(json.dumps(doc))
    os.utime(p, (utc.timestamp(), utc.timestamp()))  # fresh at the frozen instant
    return cs


@functools.cache
def _run(zone: str, step: int, mode: str, utc: pd.Timestamp):
    env = Env(zone, step, mode, weather_forecast_method="open-meteo", set_use_pv=True)
    try:
        env.params["retrieve_hass_conf"]["Latitude"] = LAT
        env.params["retrieve_hass_conf"]["Longitude"] = LON
        cs = _seed(env, utc)
        rp = list_runtimeparams(step, None)
        del rp["pv_power_forecast"], rp["prediction_horizon"]
        idd, err = env.set_input(utc, rp, "dayahead-optim")
        assert err is None, err
        fcst = idd["fcst"]
        frame = pd.DataFrame(
            {
                "ghi": cs["ghi"],
                "dni": cs["dni"],
                "dhi": cs["dhi"],
                "temp_air": 10.0,
                "wind_speed": 2.0,
                "relative_humidity": 50.0,
                "precipitable_water": 1.0,
                "cloud_cover": 0.0,
            }
        )
        truth = fcst._calculate_pvlib_power(frame).clip(lower=0)  # pure UTC index, 15 min
        # Second, independent reference: plane-of-array irradiance from pvlib's solar position
        # at the UTC instants and the seeded GHI/DNI/DHI. It shares no code with the ModelChain
        # or with EMHASS' index handling, so it also catches a displaced pvlib time index.
        sol = Location(LAT, LON).get_solarposition(cs.index)
        poa = get_total_irradiance(
            fcst.plant_conf["surface_tilt"][0],
            fcst.plant_conf["surface_azimuth"][0],
            sol["apparent_zenith"],
            sol["azimuth"],
            cs["dni"],
            cs["ghi"],
            cs["dhi"],
        )["poa_global"]
        return idd["df_input_data_dayahead"]["p_pv_forecast"].copy(), truth, poa
    finally:
        env.close()


def _step_average(truth: pd.Series, inst: pd.DatetimeIndex, step: int) -> np.ndarray:
    """Independent reference: trapezoidal mean of the 15-minute power over each step."""
    if step == 15:
        return truth.reindex(inst).to_numpy()
    n = step // 15
    out = []
    for t in inst:
        nodes = truth.reindex(pd.date_range(t, periods=n + 1, freq="15min")).to_numpy()
        out.append((nodes[0] / 2 + nodes[1:-1].sum() + nodes[-1] / 2) / n)
    return np.array(out)


def _cases():
    for zone, utc, label in RUNS:
        for step in (15, 60):
            for mode in MODES:
                yield pytest.param(
                    zone, step, mode, utc, id=f"{zone.split('/')[-1]}-{label}-{step}min-{mode}"
                )


@pytest.mark.parametrize(("zone", "step", "mode", "utc"), list(_cases()))
def test_open_meteo_pv_power_is_the_utc_instants_power(zone, step, mode, utc):
    pv, truth, poa = _run(zone, step, mode, pd.Timestamp(utc))
    inst = pd.DatetimeIndex(pv.index).tz_convert("UTC")
    assert inst.is_unique, "the forecast grid repeats an instant"
    expected = _step_average(truth, inst, step)
    assert not np.isnan(expected).any(), "grid leaves the seeded Open-Meteo window"
    # Precondition: the day must have daylight and a one-hour shift must be clearly visible.
    for hours in (-1, 1):
        shifted = _step_average(truth, inst + pd.Timedelta(hours=hours), step)
        assert np.nanmax(np.abs(shifted - expected)) > 100.0, "test cannot see a 1 h shift"
    got = pv.to_numpy(dtype=float)
    bad = np.flatnonzero(~np.isclose(got, expected, rtol=0, atol=0.5))
    assert bad.size == 0, (
        f"{bad.size} of {len(pv)} steps carry another instant's PV power; first at "
        f"{pv.index[bad[0]]}: got {got[bad[0]]:.1f} W, expected {expected[bad[0]]:.1f} W"
    )


@pytest.mark.parametrize(("zone", "step", "mode", "utc"), list(_cases()))
def test_open_meteo_pv_power_follows_the_sun_at_its_instant(zone, step, mode, utc):
    """P_PV tracks an independent plane-of-array reference at the middle of its step, and tracks
    it better than the same reference displaced by one hour in either direction."""
    pv, _, poa = _run(zone, step, mode, pd.Timestamp(utc))
    # A step value is the mean over the step: it is centred on the middle of the step.
    inst = pd.DatetimeIndex(pv.index).tz_convert("UTC") + pd.Timedelta(minutes=step) / 2
    inst = inst.floor("15min")
    got = pv.to_numpy(dtype=float)

    def corr(shift_hours: int) -> float:
        ref = poa.reindex(inst + pd.Timedelta(hours=shift_hours)).to_numpy()
        return float(np.corrcoef(got, ref)[0, 1])

    assert corr(0) > 0.99, (
        f"P_PV does not follow the solar geometry at its instants: r={corr(0):.3f}"
    )
    assert corr(0) > corr(-1) + 0.01
    assert corr(0) > corr(1) + 0.01


@pytest.mark.parametrize("zone", ["Europe/Copenhagen", "UTC"])
def test_hourly_step_is_the_step_mean_not_the_step_start_sample(zone):
    """#1287: at a 60-minute step the forecast of the last daylight hour is the mean over that hour,
    not the (much higher) power 35 minutes before sunset; the first one is not zero either."""
    utc = pd.Timestamp("2026-10-25T23:05Z")
    pv, truth, _ = _run(zone, 60, "nearest", utc)
    inst = pd.DatetimeIndex(pv.index).tz_convert("UTC")
    sampled = truth.reindex(inst).to_numpy()
    mean = _step_average(truth, inst, 60)
    lit = np.flatnonzero(mean > 1.0)
    first, last = lit[0], lit[-1]
    got = pv.to_numpy(dtype=float)
    assert sampled[first] < 0.5 * mean[first] or sampled[last] > 1.5 * mean[last], (
        "no edge effect to see"
    )
    assert got[first] == pytest.approx(mean[first], abs=0.5)
    assert got[last] == pytest.approx(mean[last], abs=0.5)
    # the energy of the day is the same either way: the change is in the timing, not the total
    assert got.sum() == pytest.approx(sampled.sum(), rel=0.03)


def test_pv_plants_of_a_step_average_add_up_to_the_aggregate():
    utc = pd.Timestamp("2026-10-25T23:05Z")
    env = Env(
        "Europe/Copenhagen", 60, "nearest", weather_forecast_method="open-meteo", set_use_pv=True
    )
    try:
        env.params["retrieve_hass_conf"]["Latitude"] = LAT
        env.params["retrieve_hass_conf"]["Longitude"] = LON
        _seed(env, utc)
        rp = list_runtimeparams(60, None)
        del rp["pv_power_forecast"], rp["prediction_horizon"]
        idd, err = env.set_input(utc, rp, "dayahead-optim")
        assert err is None, err
        fcst = idd["fcst"]
        plant = fcst.plant_conf
        for key in (
            "pv_module_model",
            "pv_inverter_model",
            "surface_tilt",
            "surface_azimuth",
            "modules_per_string",
            "strings_per_inverter",
        ):
            plant[key] = [plant[key][0]] * 2 if isinstance(plant[key], list) else [plant[key]] * 2
        plant["surface_azimuth"] = [150, 250]
        df_weather = idd["df_weather"]
        total = fcst.get_power_from_weather(df_weather)
        raw = fcst.pv_plant_raw
        assert raw is not None and raw.shape[1] == 2
        assert np.allclose(raw.sum(axis=1).to_numpy(), total.to_numpy(), atol=1.0)
        assert raw.index.equals(total.index) and len(total) == len(df_weather)
    finally:
        env.close()


def test_a_weather_frame_off_the_native_grid_falls_back_to_the_step_start_value():
    """A cached or foreign frame (not the one just fetched) is evaluated as it always was."""
    utc = pd.Timestamp("2026-10-25T23:05Z")
    pv, truth, _ = _run("Europe/Copenhagen", 60, "nearest", utc)
    env = Env(
        "Europe/Copenhagen", 60, "nearest", weather_forecast_method="open-meteo", set_use_pv=True
    )
    try:
        env.params["retrieve_hass_conf"]["Latitude"] = LAT
        env.params["retrieve_hass_conf"]["Longitude"] = LON
        _seed(env, utc)
        rp = list_runtimeparams(60, None)
        del rp["pv_power_forecast"], rp["prediction_horizon"]
        idd, err = env.set_input(utc, rp, "dayahead-optim")
        assert err is None, err
        fcst = idd["fcst"]
        frame = idd["df_weather"].iloc[3:].copy()  # not the grid the native frame was built for
        got = fcst.get_power_from_weather(frame)
        expected = truth.reindex(pd.DatetimeIndex(frame.index).tz_convert("UTC")).to_numpy()
        assert np.allclose(got.to_numpy(), expected, atol=0.5)
    finally:
        env.close()
