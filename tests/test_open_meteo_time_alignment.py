"""Offline regressions for Open-Meteo timestamp alignment (#1353)."""

import logging
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import numpy as np
import pandas as pd
import pytz

from emhass.forecast import OPEN_METEO_TIME_ALIGNED_ATTR, PV_NATIVE_WEATHER_ATTR, Forecast


class TestOpenMeteoTimeAlignment(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _native_payload(source_index):
        values = [0.0, 90.0, 0.0, 90.0, 0.0]
        n = len(source_index)
        return {
            "minutely_15": {
                "time": [int(t.timestamp()) for t in source_index],
                "temperature_2m": [20.0] * n,
                "relative_humidity_2m": [50.0] * n,
                "rain": [0.0] * n,
                "cloud_cover": [20.0] * n,
                "wind_speed_10m": [1.0] * n,
                "shortwave_radiation_instant": values,
                "diffuse_radiation_instant": values,
                "direct_normal_irradiance_instant": values,
            }
        }

    async def _native_weather(self, source_index, target_index):
        with tempfile.TemporaryDirectory() as temp_dir:
            fcst = object.__new__(Forecast)
            fcst.time_zone = target_index.tz
            fcst.forecast_dates = target_index
            fcst.freq = target_index[1] - target_index[0]
            fcst.logger = logging.getLogger(__name__)
            fcst.emhass_conf = {"data_path": Path(temp_dir)}
            fcst.optim_conf = {
                "open_meteo_cache_max_age": 60,
                "delta_forecast_daily": pd.Timedelta(days=1),
            }
            fcst.params = {"passed_data": {}}
            fcst.get_cached_open_meteo_forecast_json = AsyncMock(
                return_value=self._native_payload(source_index)
            )
            result = await fcst._get_weather_open_meteo(
                str(Path(temp_dir) / "missing-cache.pkl"), use_legacy_pvlib=False
            )
            fcst.get_cached_open_meteo_forecast_json.assert_awaited_once()
            return result

    async def test_native_nonlinear_15_to_20_minute_grid(self):
        source = pd.date_range("2026-08-22 10:00", periods=5, freq="15min", tz="UTC")
        target = pd.date_range(source[0], periods=4, freq="20min")
        actual = await self._native_weather(source, target)
        self.assertTrue(actual.index.equals(target))
        np.testing.assert_allclose(actual["ghi"].tolist(), [0.0, 60.0, 60.0, 0.0], rtol=0, atol=1e-10)
        np.testing.assert_allclose(actual["dhi"].tolist(), [0.0, 60.0, 60.0, 0.0], rtol=0, atol=1e-10)
        self.assertTrue(actual.notna().all().all())
        np.testing.assert_allclose(actual["precipitable_water"].tolist(), [0.1] * 4, rtol=0, atol=1e-10)

    async def test_native_offset_and_aligned_grids(self):
        source = pd.date_range("2026-08-22 10:00", periods=5, freq="15min", tz="UTC")
        cases = (
            (
                pd.date_range(source[0] + pd.Timedelta(minutes=5), periods=3, freq="20min"),
                [30.0, 30.0, 90.0],
            ),
            (
                pd.date_range(source[0], periods=13, freq="5min"),
                [0.0, 30.0, 60.0, 90.0, 60.0, 30.0, 0.0, 30.0, 60.0, 90.0, 60.0, 30.0, 0.0],
            ),
            (
                pd.date_range(source[0], periods=5, freq="15min"),
                [0.0, 90.0, 0.0, 90.0, 0.0],
            ),
            (
                pd.date_range(source[0], periods=3, freq="30min"),
                [0.0, 0.0, 0.0],
            ),
        )
        for target, expected in cases:
            with self.subTest(step=target[1] - target[0], start=target[0]):
                actual = await self._native_weather(source, target)
                self.assertTrue(actual.index.equals(target))
                np.testing.assert_allclose(actual["ghi"].tolist(), expected, rtol=0, atol=1e-10)
                if target[1] - target[0] == pd.Timedelta(minutes=30):
                    self.assertIn(PV_NATIVE_WEATHER_ATTR, actual.attrs)

    async def test_native_dst_fallback_uses_absolute_instants(self):
        source = pd.date_range("2026-10-25 00:30", periods=5, freq="15min", tz="UTC")
        source = source.tz_convert("Europe/Berlin")
        target = pd.date_range(source[0], periods=4, freq="20min")
        actual = await self._native_weather(source, target)
        self.assertTrue(actual.index.equals(target))
        np.testing.assert_allclose(actual["ghi"].tolist(), [0.0, 60.0, 60.0, 0.0], rtol=0, atol=1e-10)
        self.assertNotEqual(target[0].utcoffset(), target[-1].utcoffset())

    async def _covariates(self, source_index, target_index):
        fcst = object.__new__(Forecast)
        fcst.time_zone = target_index.tz
        fcst.logger = logging.getLogger(__name__)
        raw = {
            "minutely_15": {
                "time": [int(t.timestamp()) for t in source_index],
                "temperature_2m": [0.0, 90.0, 0.0],
                "relative_humidity_2m": [30.0, 60.0, 30.0],
            }
        }
        fcst._fetch_open_meteo_covariates_json = AsyncMock(return_value=raw)
        fcst._fetch_open_meteo_historical_covariates_json = AsyncMock(
            side_effect=AssertionError("Unexpected historical weather fetch")
        )
        columns = ["relative_humidity", "temp_air", "heating_degree"]
        result = await fcst.get_weather_covariates(target_index, columns)
        fcst._fetch_open_meteo_covariates_json.assert_awaited_once()
        fcst._fetch_open_meteo_historical_covariates_json.assert_not_awaited()
        self.assertEqual(list(result.columns), columns)
        self.assertTrue(result.index.equals(target_index))
        return result

    async def test_covariates_nonlinear_15_to_10_minute_grid(self):
        source = pd.date_range("2026-08-22 10:00", periods=3, freq="15min", tz="UTC")
        target = pd.date_range(source[0], periods=4, freq="10min")
        actual = await self._covariates(source, target)
        np.testing.assert_allclose(actual["temp_air"].tolist(), [0.0, 60.0, 60.0, 0.0], rtol=0, atol=1e-10)
        np.testing.assert_allclose(actual["relative_humidity"].tolist(), [30.0, 50.0, 50.0, 30.0], rtol=0, atol=1e-10)
        # Derived degree features retain the existing native-row transformation.
        np.testing.assert_allclose(actual["heating_degree"].tolist(), [18.0, 6.0, 6.0, 18.0], rtol=0, atol=1e-10)

    async def test_covariates_offset_irregular_and_boundary(self):
        source = pd.date_range("2026-08-22 10:00", periods=3, freq="15min", tz="UTC")
        target = pd.DatetimeIndex([source[0] + pd.Timedelta(minutes=n) for n in (4, 11, 18, 40)])
        actual = await self._covariates(source, target)
        np.testing.assert_allclose(actual["temp_air"].tolist(), [24.0, 66.0, 72.0, 0.0], rtol=0, atol=1e-10)

    async def test_covariates_dst_fallback_uses_absolute_instants(self):
        source = pd.date_range("2026-10-25 00:45", periods=3, freq="15min", tz="UTC")
        source = source.tz_convert("Europe/Berlin")
        target = pd.date_range(source[0], periods=4, freq="10min")
        actual = await self._covariates(source, target)
        np.testing.assert_allclose(actual["temp_air"].tolist(), [0.0, 60.0, 60.0, 0.0], rtol=0, atol=1e-10)
        self.assertNotEqual(target[0].utcoffset(), target[-1].utcoffset())

    async def test_covariates_overlap_precedence_and_null_row_are_time_weighted(self):
        source = pd.date_range("2026-08-22 10:00", periods=4, freq="15min", tz="UTC")

        def payload(times, temps):
            return {
                "minutely_15": {
                    "time": [int(t.timestamp()) for t in times],
                    "temperature_2m": temps,
                    "relative_humidity_2m": temps,
                }
            }

        # Forecast rows: 10:15 (overlap), a fully null 10:30 (dropped), 10:45.
        forecast = payload(source[1:], [90.0, None, 0.0])
        # History: 10:00, a conflicting 10:15 that must lose, and 10:30 that fills the null row.
        history = payload(source[:3], [0.0, 999.0, 90.0])
        fcst = object.__new__(Forecast)
        fcst.time_zone = pytz.UTC
        fcst.logger = logging.getLogger(__name__)
        fcst._fetch_open_meteo_covariates_json = AsyncMock(return_value=forecast)
        fcst._fetch_open_meteo_historical_covariates_json = AsyncMock(return_value=history)
        target = pd.date_range(source[0], periods=5, freq="10min")
        result = await fcst.get_weather_covariates(target, ["temp_air"])
        fcst._fetch_open_meteo_historical_covariates_json.assert_awaited_once()
        np.testing.assert_allclose(result["temp_air"].tolist(), [0.0, 60.0, 90.0, 90.0, 30.0], rtol=0, atol=1e-10)


class TestOpenMeteoWeatherCache(unittest.IsolatedAsyncioTestCase):
    """The weather pickle outlives an upgrade: a frame built by the positional interpolation of
    an earlier version must not be served once the interpolation is by elapsed time (#1353)."""

    SOURCE = pd.date_range("2026-08-22 10:00", periods=5, freq="15min", tz="UTC")
    TARGET = pd.date_range(SOURCE[0], periods=4, freq="20min")
    GOOD = [0.0, 60.0, 60.0, 0.0]
    DEFECTIVE = [0.0, 0.0, 0.0, 0.0]

    def _forecast(self, temp_dir, dry_run=False, cache=True):
        fcst = object.__new__(Forecast)
        fcst.time_zone = pytz.UTC
        fcst.forecast_dates = self.TARGET
        fcst.start_forecast = self.TARGET[0]
        fcst.freq = pd.Timedelta(minutes=20)
        fcst.weather_forecast_method = "open-meteo"
        fcst.lat, fcst.lon = 45.0, 6.0
        fcst.logger = logging.getLogger(__name__)
        fcst.emhass_conf = {"data_path": Path(temp_dir)}
        fcst.optim_conf = {
            "open_meteo_cache_max_age": 60,
            "delta_forecast_daily": pd.Timedelta(days=1),
        }
        fcst.params = {"passed_data": {"weather_forecast_cache": cache, "dry_run": dry_run}}
        payload = TestOpenMeteoTimeAlignment._native_payload(self.SOURCE)
        fcst.get_cached_open_meteo_forecast_json = AsyncMock(return_value=payload)
        return fcst

    def _write_old_cache(self, path, marked=False):
        columns = ["temp_air", "relative_humidity", "precipitable_water", "cloud_cover"]
        old = pd.DataFrame(1.0, index=self.TARGET, columns=columns)
        for name in ("ghi", "dhi", "dni"):
            old[name] = self.DEFECTIVE
        if marked:
            old.attrs[OPEN_METEO_TIME_ALIGNED_ATTR] = True
        path.write_bytes(pickle.dumps(old))
        return old

    async def test_cache_from_positional_interpolation_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weather.pkl"
            self._write_old_cache(path)
            fcst = self._forecast(temp_dir)
            data = await fcst._get_weather_open_meteo(str(path), use_legacy_pvlib=False)
            fcst.get_cached_open_meteo_forecast_json.assert_awaited_once()
            np.testing.assert_allclose(data["ghi"].tolist(), self.GOOD, rtol=0, atol=1e-10)
            self.assertTrue(data.attrs[OPEN_METEO_TIME_ALIGNED_ATTR])
            self.assertTrue(pickle.loads(path.read_bytes()).attrs[OPEN_METEO_TIME_ALIGNED_ATTR])

    async def test_current_cache_is_reused_and_matches_fresh_frame(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weather.pkl"
            fcst = self._forecast(temp_dir)
            fresh = await fcst._get_weather_open_meteo(str(path), use_legacy_pvlib=False)
            cached = await fcst._get_weather_open_meteo(str(path), use_legacy_pvlib=False)
            fcst.get_cached_open_meteo_forecast_json.assert_awaited_once()
            pd.testing.assert_frame_equal(fresh, cached)
            self.assertEqual(fresh.attrs, cached.attrs)
            np.testing.assert_allclose(cached["ghi"].tolist(), self.GOOD, rtol=0, atol=1e-10)

    async def test_legacy_request_reads_current_cache_without_native_weather(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weather.pkl"
            fcst = self._forecast(temp_dir)
            await fcst._get_weather_open_meteo(str(path), use_legacy_pvlib=False)
            legacy = await fcst._get_weather_open_meteo(str(path), use_legacy_pvlib=True)
            fcst.get_cached_open_meteo_forecast_json.assert_awaited_once()
            self.assertTrue(legacy.attrs[OPEN_METEO_TIME_ALIGNED_ATTR])
            self.assertNotIn(PV_NATIVE_WEATHER_ATTR, legacy.attrs)

    async def test_dry_run_leaves_old_cache_untouched(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weather.pkl"
            self._write_old_cache(path)
            before = path.read_bytes()
            fcst = self._forecast(temp_dir, dry_run=True)
            data = await fcst._get_weather_open_meteo(str(path), use_legacy_pvlib=False)
            np.testing.assert_allclose(data["ghi"].tolist(), self.GOOD, rtol=0, atol=1e-10)
            self.assertEqual(path.read_bytes(), before)
