"""Open-Meteo irradiance -> pvlib -> P_PV stays on its UTC instant (#1287).

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
import logging
import os
import pickle

import aiohttp
import numpy as np
import orjson
import pandas as pd
import pytest
import time_machine
from cases import expected_start
from harness import LOG, Env, list_runtimeparams, run_async
from pvlib.irradiance import get_total_irradiance
from pvlib.location import Location
from pvlib.solarposition import get_solarposition

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


@pytest.fixture(autouse=True)
def _no_live_open_meteo(monkeypatch):
    """Fail, not fall back to the seeded JSON, when a test would call the live Open-Meteo API.

    EMHASS treats a transport error as 'keep the old cache', so a test whose cache looks stale
    (an unfrozen clock) would pass offline and break where the network answers. The error raised
    here is not a ``ClientError``, so it cannot be swallowed by that fallback."""

    async def refuse(self, method, str_or_url, *args, **kwargs):
        raise AssertionError(f"live HTTP request in an offline test: {method} {str_or_url}")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", refuse)


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


def _assert_grid_is_the_runs_own(pv, step, mode, utc):
    """The forecast grid is the one the run instant defines, not one derived from the result: it
    starts at the independently computed first step and advances by ``step`` of real time. A grid
    and its values shifted together by an hour would otherwise still agree with each other."""
    inst = pd.DatetimeIndex(pv.index).tz_convert("UTC")
    zone = str(pv.index.tz)
    assert inst[0] == expected_start(utc, zone, step, mode), "the grid starts at another instant"
    assert (inst[1:] - inst[:-1] == pd.Timedelta(minutes=step)).all(), (
        "the grid is not evenly spaced"
    )


@pytest.mark.parametrize(("zone", "step", "mode", "utc"), list(_cases()))
def test_open_meteo_pv_power_is_the_utc_instants_power(zone, step, mode, utc):
    pv, truth, poa = _run(zone, step, mode, pd.Timestamp(utc))
    _assert_grid_is_the_runs_own(pv, step, mode, pd.Timestamp(utc))
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
    _assert_grid_is_the_runs_own(pv, step, mode, pd.Timestamp(utc))
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


def _assert_step_start_power(fcst, legacy: pd.DataFrame, modern: pd.DataFrame) -> None:
    """The PV power of a legacy frame is pvlib on that frame itself (the step-start values),
    not the step mean that the same request would get from the native weather."""
    got = fcst.get_power_from_weather(legacy).to_numpy(float)
    expected = fcst._calculate_pvlib_power(legacy).clip(lower=0).to_numpy(float)
    assert expected.max() > 100.0
    np.testing.assert_allclose(got, expected, atol=1e-6)
    averaged = fcst.get_power_from_weather(modern).to_numpy(float)
    assert np.abs(averaged - got).max() > 10.0, "averaged equals legacy: proves nothing"


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
        # Make the still-fresh JSON unusable as a source: a run that reread it (and ignored the
        # pickle) would see no irradiance and could not match the fetched forecast below.
        json_path = cached_env.data_path / "cached-open-meteo-forecast-b.json"
        doc = json.loads(json_path.read_text())
        for key in (
            "shortwave_radiation_instant",
            "diffuse_radiation_instant",
            "direct_normal_irradiance_instant",
        ):
            doc["minutely_15"][key] = [0.0] * len(doc["minutely_15"][key])
        json_path.write_text(json.dumps(doc))
        os.utime(json_path, (t_run.timestamp(),) * 2)
        cached = _dayahead_pv(cached_env, step, t_run)
        assert pkl.is_file(), "the pickle was dropped, so the cached path was not exercised"
        assert fetched.max() > 100.0, "the fetched reference carries no production"
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
    utc = pd.Timestamp("2026-10-25T10:05Z")  # the last step is in daylight
    full_env = _set_up(60, utc)
    cut_env = _set_up(60, utc)
    try:
        rp = list_runtimeparams(60, None)
        del rp["pv_power_forecast"], rp["prediction_horizon"]
        idd, err = full_env.set_input(utc, rp, "dayahead-optim")
        assert err is None, err
        fcst = idd["fcst"]
        full = idd["df_input_data_dayahead"]["p_pv_forecast"].copy()
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
        # Independent value of the last step: the weather of the last step start held on the
        # five nodes of the step, with the sun moving on, trapezoid-averaged.
        nodes = pd.date_range(last, periods=5, freq="15min")
        held = _clear_sky(utc).loc[last]
        frame = pd.DataFrame(
            {
                "ghi": held["ghi"],
                "dni": held["dni"],
                "dhi": held["dhi"],
                "temp_air": 10.0,
                "wind_speed": 2.0,
                "relative_humidity": 50.0,
                "precipitable_water": 1.0,
                "cloud_cover": 0.0,
            },
            index=nodes,
        )
        power = fcst._calculate_pvlib_power(frame).clip(lower=0).to_numpy(float)
        expected_last = (power[0] / 2 + power[1:-1].sum() + power[-1] / 2) / 4
        assert expected_last > 1.0, "the last step carries no production, so it proves nothing"
        assert cut.to_numpy(float)[-1] == pytest.approx(expected_last, abs=0.5)
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
        with time_machine.travel(utc.to_pydatetime(), tick=False):  # the seeded JSON is fresh
            legacy = run_async(
                idd["fcst"].get_weather_forecast("open-meteo", use_legacy_pvlib=True)
            )
        assert "pv_native_weather" not in legacy.attrs
        _assert_step_start_power(idd["fcst"], legacy, idd["df_weather"])
        rp = list_runtimeparams(60, None)  # keeps pv_power_forecast: 1000 W everywhere
        del rp["prediction_horizon"]
        idd, err = env.set_input(utc, rp, "dayahead-optim")
        assert err is None, err
        assert (idd["df_input_data_dayahead"]["p_pv_forecast"] == 1000.0).all()
    finally:
        env.close()


# -- the step-mean routine on synthetic frames: clipping and rejection of the native weather ------

T0 = pd.Timestamp("2026-10-25T00:00Z")
REJECTED_LOG = "does not match the weather frame"


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture(scope="module")
def fcst():
    env = _set_up(60, pd.Timestamp("2026-10-25T23:05Z"))
    try:
        rp = list_runtimeparams(60, None)
        del rp["pv_power_forecast"], rp["prediction_horizon"]
        idd, err = env.set_input(pd.Timestamp("2026-10-25T23:05Z"), rp, "dayahead-optim")
        assert err is None, err
        yield idd["fcst"]
    finally:
        env.close()


def _frame(n_steps: int, step: int = 60, starts=None, native=None) -> pd.DataFrame:
    """A weather frame of ``n_steps`` step starts that carries 15-minute native weather."""
    starts = (
        starts
        if starts is not None
        else [T0 + pd.Timedelta(minutes=step * i) for i in range(n_steps)]
    )
    frame = pd.DataFrame(
        {"ghi": 0.0}, index=pd.DatetimeIndex(starts).tz_convert("Europe/Copenhagen")
    )
    n_nodes = n_steps * (step // 15) + 1
    if native is None:
        native = {
            "time": [int((T0 + pd.Timedelta(minutes=15 * i)).timestamp()) for i in range(n_nodes)],
            "ghi": [0.0] * n_nodes,
        }
    frame.attrs["pv_native_weather"] = native
    return frame


def _stub_power(fcst, monkeypatch, power=None):
    """Replace the pvlib call: the power is the node number (or ``power``), the frame is recorded."""
    calls: list[pd.DataFrame] = []

    def fake(df):
        calls.append(df)
        values = power if power is not None else np.arange(len(df), dtype=float)
        return pd.Series(np.asarray(values, dtype=float), index=df.index)

    monkeypatch.setattr(fcst, "_calculate_pvlib_power", fake)
    fcst.pv_plant_raw = None
    return calls


def test_negative_node_power_is_clipped_before_it_is_averaged(fcst, monkeypatch):
    """A night-time inverter draw is not production: it must not lower the step that ends or
    starts at it. Two 60-minute steps over nodes -100 -100 50 100 200 100 50 -100 -100."""
    monkeypatch.setattr(fcst, "freq", pd.Timedelta(minutes=60))
    calls = _stub_power(fcst, monkeypatch, power=[-100, -100, 50, 100, 200, 100, 50, -100, -100])
    got = fcst._pv_power_from_weather_frame(_frame(2))
    assert len(calls[0]) == 9, "the power was not evaluated on the native nodes"
    np.testing.assert_allclose(got.to_numpy(), [62.5, 62.5])  # (0 + 0 + 50 + 100 + 100) / 4


def test_valid_native_weather_is_averaged_and_not_logged_as_rejected(fcst, monkeypatch):
    monkeypatch.setattr(fcst, "freq", pd.Timedelta(minutes=60))
    calls = _stub_power(fcst, monkeypatch)  # node power 0, 1, ..., 8
    handler = _Records()
    fcst.logger.addHandler(handler)
    old_level = fcst.logger.level
    fcst.logger.setLevel(logging.DEBUG)
    try:
        got = fcst._pv_power_from_weather_frame(_frame(2))
    finally:
        fcst.logger.removeHandler(handler)
        fcst.logger.setLevel(old_level)
    assert len(calls[0]) == 9
    np.testing.assert_allclose(got.to_numpy(), [2.0, 6.0])  # trapezoid of a ramp = its middle
    assert not any(REJECTED_LOG in m for m in handler.messages)


def _uneven():
    starts = [T0 + pd.Timedelta(minutes=m) for m in (0, 60, 120, 240)]
    return 60, _frame(4, starts=starts)


def _off_grid():
    n = 2 * 4 + 1
    native = {
        "time": [int((T0 + pd.Timedelta(minutes=15 * i + 7)).timestamp()) for i in range(n)],
        "ghi": [0.0] * n,
    }
    return 60, _frame(2, native=native)


def _short_native():
    native = _frame(2).attrs["pv_native_weather"]
    return 60, _frame(2, native={k: v[:-1] for k, v in native.items()})  # no right-hand endpoint


def _native_starts_late():
    native = _frame(2).attrs["pv_native_weather"]
    return 60, _frame(2, native={k: v[1:] for k, v in native.items()})  # misses the first node


def _step_of_the_frame_is_not_the_run_step():
    return 30, _frame(2, step=60)  # a 60-minute frame read by a 30-minute run


def _step_not_a_multiple_of_15():
    return 20, _frame(2, step=20, native=_frame(2).attrs["pv_native_weather"])


@pytest.mark.parametrize(
    "case",
    [
        _uneven,
        _off_grid,
        _short_native,
        _native_starts_late,
        _step_of_the_frame_is_not_the_run_step,
        _step_not_a_multiple_of_15,
    ],
)
def test_native_weather_that_does_not_fit_the_frame_is_rejected_and_logged(fcst, monkeypatch, case):
    """Irregular, off-grid, short or step-incompatible native weather is never used: the power is
    evaluated on the frame itself, as before this change, and the debug log says so."""
    step, frame = case()
    monkeypatch.setattr(fcst, "freq", pd.Timedelta(minutes=step))
    calls = _stub_power(fcst, monkeypatch)
    handler = _Records()
    fcst.logger.addHandler(handler)
    old_level = fcst.logger.level
    fcst.logger.setLevel(logging.DEBUG)
    try:
        got = fcst._pv_power_from_weather_frame(frame)
    finally:
        fcst.logger.removeHandler(handler)
        fcst.logger.setLevel(old_level)
    assert len(calls) == 1 and calls[0] is frame, "the native weather was used"
    np.testing.assert_allclose(got.to_numpy(), np.arange(len(frame), dtype=float))
    assert any(REJECTED_LOG in m for m in handler.messages)


def test_native_weather_on_a_timezone_naive_frame_falls_back_to_the_step_start(fcst, monkeypatch):
    """A caller that strips the timezone but keeps the attrs gets the step-start power, not a
    TypeError from the UTC conversion."""
    calls = _stub_power(fcst, monkeypatch)
    frame = _frame(3)
    frame.index = frame.index.tz_convert("UTC").tz_localize(None)
    got = fcst._pv_power_from_weather_frame(frame)
    assert len(calls) == 1 and calls[0] is frame
    assert got.to_numpy().tolist() == [0.0, 1.0, 2.0]


def test_a_frame_without_native_weather_is_evaluated_silently(fcst, monkeypatch):
    monkeypatch.setattr(fcst, "freq", pd.Timedelta(minutes=60))
    calls = _stub_power(fcst, monkeypatch)
    frame = _frame(2)
    frame.attrs.clear()
    handler = _Records()
    fcst.logger.addHandler(handler)
    old_level = fcst.logger.level
    fcst.logger.setLevel(logging.DEBUG)
    try:
        fcst._pv_power_from_weather_frame(frame)
    finally:
        fcst.logger.removeHandler(handler)
        fcst.logger.setLevel(old_level)
    assert calls[0] is frame
    assert not any(REJECTED_LOG in m for m in handler.messages)


def _raw_15min(n_nodes: int, drop=()) -> pd.DataFrame:
    """Open-Meteo-like 15-minute rows (UTC clock, shown in the run zone) without the ``drop`` nodes."""
    idx = pd.date_range(T0, periods=n_nodes, freq="15min")
    idx = idx[[i for i in range(n_nodes) if i not in set(drop)]].tz_convert("Europe/Copenhagen")
    frame = pd.DataFrame(
        {
            "temp_air": 10.0,
            "relative_humidity": 80.0,
            "precipitable_water": 10.0,
            "cloud_cover": 0.0,
            "wind_speed": 1.0,
            "ghi": np.arange(len(idx), dtype=float),
            "dhi": 1.0,
            "dni": 1.0,
        },
        index=idx,
    )
    return frame


def _build_native(fcst, monkeypatch, raw):
    monkeypatch.setattr(fcst, "freq", pd.Timedelta(minutes=60))
    monkeypatch.setattr(
        fcst,
        "forecast_dates",
        pd.date_range(T0, periods=2, freq="60min").tz_convert("Europe/Copenhagen"),
    )
    return fcst._build_pv_native_weather(raw)


def test_complete_native_weather_is_built(fcst, monkeypatch):
    assert _build_native(fcst, monkeypatch, _raw_15min(9)) is not None


@pytest.mark.parametrize(
    "drop",
    [(1, 2, 3, 5, 6, 7), (4,), (1,), (0,)],
    ids=["hourly-only", "mid-node", "node-1", "first-node"],
)
def test_missing_internal_native_intervals_are_not_interpolated_into_the_step_mean(
    fcst, monkeypatch, drop
):
    """Hourly rows without the intervening 15-minute rows must not become 'native' weather by
    linear interpolation: the PV power then keeps the step-start values (documented fallback)."""
    assert _build_native(fcst, monkeypatch, _raw_15min(9, drop)) is None


@pytest.mark.parametrize(
    "node",
    [0, 2, 4, 6],
    ids=["first-node", "off-step-node", "step-start-node", "hole-in-the-last-step"],
)
def test_a_null_cell_in_a_present_row_is_not_interpolated_into_the_step_mean(
    fcst, monkeypatch, node
):
    """A null in a row that exists is a missing 15-minute sample, like a missing row: the native
    weather is not built (the PV power keeps the step-start values) rather than invented."""
    raw = _raw_15min(9)
    raw.iloc[node, raw.columns.get_loc("ghi")] = np.nan
    assert _build_native(fcst, monkeypatch, raw) is None


@pytest.mark.parametrize("nulls", [(8,), (7, 8)], ids=["endpoint", "trailing-run"])
def test_a_trailing_null_run_in_the_last_step_is_held_like_a_missing_endpoint(
    fcst, monkeypatch, nulls
):
    """The documented tail policy: nulls that only trail the last step hold the last value."""
    raw = _raw_15min(9)
    raw.iloc[list(nulls), raw.columns.get_loc("ghi")] = np.nan
    got = _build_native(fcst, monkeypatch, raw)
    assert got is not None
    assert got["ghi"][nulls[0] :] == [got["ghi"][nulls[0] - 1]] * len(nulls)


def test_only_the_missing_final_endpoint_is_held(fcst, monkeypatch):
    """The documented policy: data that stops at the last step start holds its last weather at the
    one missing right-hand endpoint."""
    got = _build_native(fcst, monkeypatch, _raw_15min(9, drop=(8,)))
    assert got is not None and len(got["time"]) == 9
    assert got["ghi"][8] == got["ghi"][7]


def test_data_stopping_at_the_last_step_start_still_builds_and_a_hole_inside_it_does_not(
    fcst, monkeypatch
):
    held = _build_native(fcst, monkeypatch, _raw_15min(9, drop=(5, 6, 7, 8)))
    assert held is not None and held["ghi"][5:] == [held["ghi"][4]] * 4
    assert _build_native(fcst, monkeypatch, _raw_15min(9, drop=(6,))) is None


def test_a_legacy_request_reading_a_cache_saved_by_a_modern_run_gets_no_native_weather():
    """The cache is shared: a legacy request keeps the step-start evaluation even when the cached
    frame was saved with the native weather, and the modern frame in the cache is not changed."""
    utc = pd.Timestamp("2026-10-25T23:05Z")
    env = _set_up(60, utc)
    try:
        rp = list_runtimeparams(60, None)
        del rp["pv_power_forecast"], rp["prediction_horizon"]
        idd, err = env.set_input(utc, rp, "dayahead-optim")
        assert err is None, err
        fcst = idd["fcst"]
        with time_machine.travel(utc.to_pydatetime(), tick=False):
            modern = run_async(fcst.get_weather_forecast("open-meteo"))
            assert "pv_native_weather" in modern.attrs
            cache = os.path.abspath(env.data_path / "weather_forecast_data.pkl")
            fcst.params["passed_data"]["weather_forecast_cache"] = True
            run_async(fcst.set_cached_forecast_data(cache, modern))
            assert os.path.isfile(cache)
            legacy = run_async(fcst.get_weather_forecast("open-meteo", use_legacy_pvlib=True))
            assert "pv_native_weather" not in legacy.attrs
            _assert_step_start_power(fcst, legacy, modern)
            again = run_async(fcst.get_weather_forecast("open-meteo"))
            assert "pv_native_weather" in again.attrs, "the shared cache lost the native weather"
    finally:
        env.close()


def test_the_shared_cache_holds_the_open_meteo_weather_whatever_the_requesting_pv_mode():
    """The weather cache is shared by modern and legacy requests, so it always stores the Open-Meteo
    weather (irradiance and native 15-minute samples). A legacy request gets the cloud-cover
    irradiance on the frame it returns, never in the cache: a later modern request (PV power and
    the ``ghi`` of the thermal solar gain) matches one that fetched the weather itself, and a
    legacy request reading the cache gets what a legacy request that fetched would."""
    utc = pd.Timestamp("2026-10-25T23:05Z")
    env = _set_up(60, utc)
    try:
        rp = list_runtimeparams(60, None)
        del rp["pv_power_forecast"], rp["prediction_horizon"]
        idd, err = env.set_input(utc, rp, "dayahead-optim")
        assert err is None, err
        fcst = idd["fcst"]
        # An overcast sky: the cloud-cover irradiance of a legacy request then differs from the
        # Open-Meteo irradiance (under the clear-sky seed the two are the same).
        json_path = env.data_path / "cached-open-meteo-forecast-b.json"
        doc = json.loads(json_path.read_text())
        doc["minutely_15"]["cloud_cover"] = [90.0] * len(doc["minutely_15"]["cloud_cover"])
        json_path.write_text(json.dumps(doc))
        os.utime(json_path, (utc.timestamp(),) * 2)
        cache = os.path.abspath(env.data_path / "weather_forecast_data.pkl")
        with time_machine.travel(utc.to_pydatetime(), tick=False):
            reference = run_async(fcst.get_weather_forecast("open-meteo"))
            legacy_reference = run_async(
                fcst.get_weather_forecast("open-meteo", use_legacy_pvlib=True)
            )
            assert not os.path.isfile(cache), "the reference runs must not have saved a cache"
            fcst.params["passed_data"]["weather_forecast_cache"] = True
            legacy = run_async(fcst.get_weather_forecast("open-meteo", use_legacy_pvlib=True))
            assert os.path.isfile(cache), "the legacy request saved no cache: nothing was tested"
            with open(cache, "rb") as fh:
                saved = pickle.loads(fh.read())
            modern = run_async(fcst.get_weather_forecast("open-meteo"))
            legacy_again = run_async(fcst.get_weather_forecast("open-meteo", use_legacy_pvlib=True))
        assert "pv_native_weather" not in legacy.attrs
        # the legacy request itself is unchanged: cloud-cover irradiance, step-start evaluation
        assert np.abs(legacy["ghi"].to_numpy() - reference["ghi"].to_numpy()).max() > 10.0
        pd.testing.assert_frame_equal(legacy, legacy_reference)
        # the cache is canonical
        pd.testing.assert_frame_equal(saved, reference, check_freq=False)
        assert saved.attrs == reference.attrs
        # a modern request reading it is a fresh modern request: ghi (thermal) and PV power
        pd.testing.assert_frame_equal(modern, reference)
        assert modern.attrs == reference.attrs
        got = fcst.get_power_from_weather(modern).to_numpy(float)
        expected = fcst.get_power_from_weather(reference).to_numpy(float)
        assert expected.max() > 100.0
        np.testing.assert_allclose(got, expected, atol=1e-6)
        # a legacy request reading the cache is a legacy request that fetched
        pd.testing.assert_frame_equal(legacy_again, legacy_reference)
        assert "pv_native_weather" not in legacy_again.attrs
    finally:
        env.close()


@pytest.mark.parametrize(
    "damage",
    [
        lambda a: a.pop("time"),
        lambda a: a.update(ghi=a["ghi"][:-1]),
        lambda a: a.update(time=[a["time"][0]] + a["time"][:-1]),
        lambda a: a.update(ghi=["bright"] * len(a["ghi"])),
        lambda a: a.update(time=["noon"] * len(a["time"])),
        lambda a: a.pop("ghi"),
    ],
    ids=["no-time", "unequal-lengths", "duplicate-times", "text-values", "text-times", "no-ghi"],
)
def test_malformed_native_weather_metadata_falls_back_to_the_step_start(fcst, monkeypatch, damage):
    """The attrs travel through a pickle a caller or an older version may have written: a schema
    that cannot be read is treated like absent native weather, not raised to the forecast."""
    monkeypatch.setattr(fcst, "freq", pd.Timedelta(minutes=60))
    calls = _stub_power(fcst, monkeypatch)
    frame = _frame(2)
    damage(frame.attrs["pv_native_weather"])
    got = fcst._pv_power_from_weather_frame(frame)
    assert len(calls) == 1 and calls[0] is frame, "the damaged native weather was used"
    np.testing.assert_allclose(got.to_numpy(), [0.0, 1.0])


def test_a_programming_error_in_the_pv_model_is_not_taken_for_malformed_metadata(fcst, monkeypatch):
    """Only the reading of the stored schema falls back; a failure of the power calculation on
    valid native weather still surfaces."""
    monkeypatch.setattr(fcst, "freq", pd.Timedelta(minutes=60))

    def broken(df):
        raise RuntimeError("pvlib exploded")

    monkeypatch.setattr(fcst, "_calculate_pvlib_power", broken)
    with pytest.raises(RuntimeError, match="pvlib exploded"):
        fcst._pv_power_from_weather_frame(_frame(2))


def _elevation(fcst, starts) -> np.ndarray:
    return get_solarposition(starts, fcst.lat, fcst.lon)["elevation"].to_numpy()


def _sunrise_steps(fcst, step: int) -> pd.DatetimeIndex:
    """Step starts around the late-October sunrise: exactly one step starts below the horizon and
    ends above it."""
    starts = pd.date_range("2026-10-26T04:00Z", periods=8, freq=f"{step}min")
    start_el = _elevation(fcst, starts)
    end_el = _elevation(fcst, starts + pd.Timedelta(minutes=step))
    assert ((start_el <= 0) & (end_el > 0)).sum() == 1, "the test day has no single sunrise step"
    return starts.tz_convert("Europe/Copenhagen")


class _Identity:
    """A fitted model that returns the forecast: the weighting is the only thing under test."""

    def predict(self, x):
        return x["forecast"].to_numpy(float)


def _adjusted(fcst, monkeypatch, starts, values, step_mean: bool, step: int = 60):
    monkeypatch.setattr(fcst, "freq", pd.Timedelta(minutes=step))
    monkeypatch.setattr(fcst, "model_adjust_pv", _Identity(), raising=False)
    monkeypatch.setattr(fcst, "_pv_forecast_is_step_mean", step_mean)
    frame = pd.DataFrame({"forecast": np.asarray(values, dtype=float)}, index=starts)
    return fcst.adjust_pv_forecast_predict(frame)["adjusted_forecast"]


def test_the_adjusted_pv_forecast_keeps_the_mean_of_a_step_that_contains_the_sunrise(
    fcst, monkeypatch
):
    """A step mean is labelled at the step start. The step that starts below the horizon and ends
    above it has a positive mean, which the night mask must not zero."""
    starts = _sunrise_steps(fcst, 60)
    sunrise_step = int(np.argmax(_elevation(fcst, starts) > 0)) - 1
    values = np.zeros(len(starts))
    values[sunrise_step:] = 300.0
    adjusted = _adjusted(fcst, monkeypatch, starts, values, step_mean=True)
    assert adjusted.iloc[sunrise_step - 1] == 0.0, "a step wholly before the sunrise stays zero"
    assert adjusted.iloc[sunrise_step] > 0.0, "the sunrise-step mean was discarded"
    assert adjusted.iloc[sunrise_step + 1] > 0.0


def test_the_low_sun_weighting_of_a_step_mean_uses_the_higher_elevation_of_the_step(
    fcst, monkeypatch
):
    starts = _sunrise_steps(fcst, 60)
    sunrise_step = int(np.argmax(_elevation(fcst, starts) > 0)) - 1
    end_el = _elevation(fcst, starts[sunrise_step : sunrise_step + 1] + pd.Timedelta(minutes=60))[0]
    threshold = fcst.optim_conf["adjusted_pv_solar_elevation_threshold"]
    assert 0 < end_el < threshold, "the sunrise step must end in the low-sun band"
    adjusted = _adjusted(fcst, monkeypatch, starts, np.full(len(starts), 300.0), step_mean=True)
    np.testing.assert_allclose(adjusted.iloc[sunrise_step], 300.0 * end_el / threshold)


def test_a_pv_calculation_that_does_not_use_native_weather_clears_the_step_mean_mark(
    fcst, monkeypatch
):
    monkeypatch.setattr(fcst, "_pv_forecast_is_step_mean", True)
    monkeypatch.setitem(fcst.retrieve_hass_conf, "solar_forecast_kwp", 0)
    fcst.get_power_from_weather(_frame(2))
    assert fcst._pv_forecast_is_step_mean is False


def test_the_adjusted_pv_forecast_of_a_step_start_forecast_is_unchanged(fcst, monkeypatch):
    """15-minute steps, legacy, Solcast and the other methods do not hand over step means: the
    night mask stays on the step start."""
    starts = _sunrise_steps(fcst, 60)
    adjusted = _adjusted(fcst, monkeypatch, starts, np.full(len(starts), 300.0), step_mean=False)
    assert (adjusted.to_numpy()[_elevation(fcst, starts) <= 0] == 0.0).all()


def test_the_adjusted_pv_forecast_at_sunset_keeps_the_step_that_contains_it(fcst, monkeypatch):
    starts = pd.date_range("2026-10-26T13:00Z", periods=7, freq="60min").tz_convert(
        "Europe/Copenhagen"
    )
    last_up = int(np.where(_elevation(fcst, starts) > 0)[0].max())  # the step with the sunset
    assert 0 < last_up < len(starts) - 1
    adjusted = _adjusted(fcst, monkeypatch, starts, np.full(len(starts), 100.0), step_mean=True)
    assert (adjusted.to_numpy()[: last_up + 1] > 0.0).all()
    assert (adjusted.to_numpy()[last_up + 1 :] == 0.0).all(), "a step wholly after the sunset"


def test_the_pv_power_of_a_native_weather_frame_marks_the_forecast_as_a_step_mean(
    fcst, monkeypatch
):
    monkeypatch.setattr(fcst, "freq", pd.Timedelta(minutes=60))
    monkeypatch.setattr(fcst, "_pv_forecast_is_step_mean", False)
    _stub_power(fcst, monkeypatch)
    fcst._pv_power_from_weather_frame(_frame(2))
    assert fcst._pv_forecast_is_step_mean is True
    frame = _frame(2)
    frame.attrs.clear()
    fcst._pv_power_from_weather_frame(frame)
    assert fcst._pv_forecast_is_step_mean is False


def test_plant_shares_add_up_to_the_aggregate_when_one_plant_is_negative(fcst, monkeypatch):
    """Per-plant node power is clipped for the shares, the aggregate after summing, and the split
    renormalises to the aggregate: mixed-sign plants still add up to the step mean."""
    monkeypatch.setattr(fcst, "freq", pd.Timedelta(minutes=60))
    plant_a = np.array([-5.0, 100, 200, 300, 100, 50, 0, -5, -5])
    plant_b = np.array([-20.0, -30, -40, 50, 60, 70, -10, -20, -20])

    def fake(df):
        idx = df.index
        fcst.pv_plant_raw = fcst._pv_plant_frame(
            [pd.Series(plant_a, index=idx), pd.Series(plant_b, index=idx)]
        )
        return pd.Series(plant_a + plant_b, index=idx)

    monkeypatch.setattr(fcst, "_calculate_pvlib_power", fake)
    aggregate = fcst._pv_power_from_weather_frame(_frame(2))
    weights = np.array([0.5, 1, 1, 1, 0.5])
    expected = [
        (np.clip((plant_a + plant_b)[i : i + 5], 0, None) * weights).sum() / 4 for i in (0, 4)
    ]
    np.testing.assert_allclose(aggregate.to_numpy(), expected)
    plants = fcst.split_pv_plants(aggregate, fcst.pv_plant_raw)
    np.testing.assert_allclose(plants.sum(axis=1).to_numpy(), aggregate.to_numpy())
    assert (plants.to_numpy() >= 0).all()
