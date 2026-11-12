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
import orjson
import pandas as pd
import pytest
import time_machine
from harness import LOG, Env, list_runtimeparams, run_async
from pvlib.irradiance import get_total_irradiance
from pvlib.location import Location

from emhass import command_line as cl

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


def _set_up(step: int, utc: pd.Timestamp, **optim) -> Env:
    env = Env(
        "Europe/Copenhagen",
        step,
        "nearest",
        weather_forecast_method="open-meteo",
        set_use_pv=True,
        **optim,
    )
    env.params["retrieve_hass_conf"]["Latitude"] = LAT
    env.params["retrieve_hass_conf"]["Longitude"] = LON
    _seed(env, utc)
    return env


def _dayahead_pv(env: Env, step: int, utc: pd.Timestamp) -> pd.Series:
    rp = list_runtimeparams(step, None)
    del rp["pv_power_forecast"], rp["prediction_horizon"]
    idd, err = env.set_input(utc, rp, "dayahead-optim")
    assert err is None, err
    return idd["df_input_data_dayahead"]["p_pv_forecast"].copy()


def test_a_weather_frame_without_the_native_weather_keeps_the_step_start_value():
    """A frame from elsewhere (an old pickle, a caller) is evaluated as it always was."""
    utc = pd.Timestamp("2026-10-25T23:05Z")
    _, truth, _ = _run("Europe/Copenhagen", 60, "nearest", utc)
    env = _set_up(60, utc)
    try:
        rp = list_runtimeparams(60, None)
        del rp["pv_power_forecast"], rp["prediction_horizon"]
        idd, err = env.set_input(utc, rp, "dayahead-optim")
        assert err is None, err
        frame = idd["df_weather"].copy()
        assert "pv_native_weather" in frame.attrs
        frame.attrs.clear()
        got = idd["fcst"].get_power_from_weather(frame)
        expected = truth.reindex(pd.DatetimeIndex(frame.index).tz_convert("UTC")).to_numpy()
        assert np.allclose(got.to_numpy(), expected, atol=0.5)
    finally:
        env.close()


@pytest.mark.parametrize("step", [30, 60, 120])
def test_a_cached_weather_frame_gives_the_pv_power_of_a_fetched_one(step):
    """``weather-forecast-cache`` pickles the weather frame; a later run reads that pickle instead of
    the Open-Meteo JSON. Same data in, same PV forecast out: the native weather rides in the frame."""
    t_cache = pd.Timestamp("2026-10-25T23:05Z")
    t_run = t_cache + pd.Timedelta(hours=2)  # still inside the cached (doubled) horizon
    fetched_env = _set_up(step, t_cache)
    cached_env = _set_up(step, t_cache)
    try:
        os.utime(
            fetched_env.data_path / "cached-open-meteo-forecast-b.json", (t_run.timestamp(),) * 2
        )
        fetched = _dayahead_pv(fetched_env, step, t_run)
        params = orjson.dumps(cached_env.params).decode()
        with time_machine.travel(t_cache.to_pydatetime(), tick=False):
            ok = run_async(cl.weather_forecast_cache(cached_env.conf, params, "{}", LOG))
        assert ok
        pkl = cached_env.data_path / "weather_forecast_data.pkl"
        assert pkl.is_file()
        os.utime(
            cached_env.data_path / "cached-open-meteo-forecast-b.json", (t_run.timestamp(),) * 2
        )
        cached = _dayahead_pv(cached_env, step, t_run)
        assert pkl.is_file(), "the pickle was dropped, so the cached path was not exercised"
        assert fetched.index.equals(cached.index)
        np.testing.assert_allclose(cached.to_numpy(float), fetched.to_numpy(float), atol=1e-6)
        # and it is the step mean, not the step-start sample, on both paths
        _, truth, _ = _run("Europe/Copenhagen", 60, "nearest", t_cache)
        expected = _step_average(truth, pd.DatetimeIndex(cached.index).tz_convert("UTC"), step)
        np.testing.assert_allclose(cached.to_numpy(float), expected, atol=0.5)
    finally:
        fetched_env.close()
        cached_env.close()


def test_the_last_step_without_its_right_hand_endpoint_holds_the_last_weather():
    """Open-Meteo normally reaches past the horizon. When the data ends at the last step start,
    that step is finite, every other step is untouched and the missing node holds the last weather
    (the tail policy of the step-start path)."""
    utc = pd.Timestamp("2026-10-25T23:05Z")
    full_env = _set_up(60, utc)
    cut_env = _set_up(60, utc)
    try:
        full = _dayahead_pv(full_env, 60, utc)
        last = full.index[-1].tz_convert("UTC")
        path = cut_env.data_path / "cached-open-meteo-forecast-b.json"
        doc = json.loads(path.read_text())
        keep = [i for i, t in enumerate(doc["minutely_15"]["time"]) if t <= last.timestamp()]
        doc["minutely_15"] = {k: [v[i] for i in keep] for k, v in doc["minutely_15"].items()}
        path.write_text(json.dumps(doc))
        os.utime(path, (utc.timestamp(), utc.timestamp()))
        cut = _dayahead_pv(cut_env, 60, utc)
        assert np.isfinite(cut.to_numpy(float)).all()
        np.testing.assert_allclose(cut.to_numpy(float)[:-1], full.to_numpy(float)[:-1], atol=1e-6)
    finally:
        full_env.close()
        cut_env.close()


@pytest.mark.parametrize("utc", ["2026-10-24T22:05Z", "2026-10-25T23:05Z"])
@pytest.mark.parametrize("zone", ["Europe/Copenhagen", "Europe/Amsterdam"])
@pytest.mark.parametrize("step", [30, 60, 120])
def test_step_mean_holds_for_each_step_zone_and_side_of_the_dst_change(step, zone, utc):
    """Every multiple-of-15 step longer than 15 minutes is the trapezoidal mean of the 15-minute
    power, before and after the 25 October 2026 change, in both zones, and the optimiser's energy
    (value x step) is the energy of the 15-minute forecast over the same span (conservation)."""
    pv, truth, _ = _run(zone, step, "nearest", pd.Timestamp(utc))
    inst = pd.DatetimeIndex(pv.index).tz_convert("UTC")
    expected = _step_average(truth, inst, step)
    np.testing.assert_allclose(pv.to_numpy(float), expected, atol=0.5)
    span = truth.reindex(
        pd.date_range(inst[0], inst[-1] + pd.Timedelta(minutes=step), freq="15min")
    )
    trapezoid_wh = (span.iloc[0] / 2 + span.iloc[1:-1].sum() + span.iloc[-1] / 2) * 0.25
    assert pv.sum() * step / 60 == pytest.approx(trapezoid_wh, abs=2.0)


def test_legacy_pvlib_and_caller_supplied_forecasts_are_not_averaged():
    """The cloud-cover (legacy) weather has no native 15-minute data and keeps the frame's own
    values; a forecast passed by the caller reaches the optimiser untouched."""
    utc = pd.Timestamp("2026-10-25T23:05Z")
    env = _set_up(60, utc)
    try:
        rp = list_runtimeparams(60, None)
        del rp["pv_power_forecast"], rp["prediction_horizon"]
        idd, err = env.set_input(utc, rp, "dayahead-optim")
        assert err is None, err
        legacy = run_async(idd["fcst"].get_weather_forecast("open-meteo", use_legacy_pvlib=True))
        assert "pv_native_weather" not in legacy.attrs
        rp = list_runtimeparams(60, None)  # keeps pv_power_forecast: 1000 W everywhere
        del rp["prediction_horizon"]
        idd, err = env.set_input(utc, rp, "dayahead-optim")
        assert err is None, err
        assert (idd["df_input_data_dayahead"]["p_pv_forecast"] == 1000.0).all()
    finally:
        env.close()
