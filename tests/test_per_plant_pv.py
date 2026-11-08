#!/usr/bin/env python
"""Per-plant PV forecast (#1150): capture, split, cache rule and the solar_forecast_kwp list."""

import math
import pathlib
import pickle
import re
import tempfile
import unittest
import unittest.mock

import numpy as np
import orjson
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from emhass import forecast as forecast_module
from emhass import utils
from emhass.forecast import Forecast
from tests import test_forecast as test_forecast_module
from tests.fixture_clock import fixture_clock
from tests.http_test_server import HttpTestServer

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)

SOLAR_FORECAST_URL = re.compile(r"https://api\.forecast\.solar/estimate/.*")

# The plant_conf lists that hold one entry per PV plant when pv_module_model is a list.
PLANT_LIST_KEYS = (
    "pv_module_model",
    "pv_inverter_model",
    "surface_tilt",
    "surface_azimuth",
    "modules_per_string",
    "strings_per_inverter",
)


def make_two_plant_pvlib_conf(plant_conf):
    """Turn a one-plant plant_conf into two pvlib plants, every sibling list at length 2."""
    plant_conf["pv_module_model"] = [plant_conf["pv_module_model"][0]] * 2
    plant_conf["pv_inverter_model"] = [plant_conf["pv_inverter_model"][0]] * 2
    plant_conf["surface_tilt"] = [30, 45]
    plant_conf["surface_azimuth"] = [270, 90]
    plant_conf["modules_per_string"] = [8, 8]
    plant_conf["strings_per_inverter"] = [1, 1]


# solar.forecast answers (watts per timestamp) for two planes, a and b.
SOLAR_FORECAST_WATTS = {
    "a": {
        "2026-08-22 10:00:00": 100,
        "2026-08-22 11:00:00": 200,
        "2026-08-22 12:00:00": 300,
    },
    "b": {
        "2026-08-22 10:00:00": 10,
        "2026-08-22 11:00:00": 40,
        "2026-08-22 12:00:00": 70,
    },
}

# No error line may carry a payload, so every line stays short.
MAX_LOG_LINE = 500


def _agg_series(values, tz="Europe/Paris"):
    index = pd.date_range("2026-08-22 10:00", periods=len(values), freq="30min", tz=tz)
    return pd.Series(values, index=index, dtype=float)


class PerPlantBase(unittest.IsolatedAsyncioTestCase):
    """A Forecast built from the default config, with its own empty data directory."""

    async def asyncSetUp(self):
        params = await test_forecast_module.TestForecast.get_test_params()
        params_json = orjson.dumps(params).decode("utf-8")
        self.retrieve_hass_conf, self.optim_conf, self.plant_conf = utils.get_yaml_parse(
            params_json, logger
        )
        with fixture_clock():
            self.fcst = Forecast(
                self.retrieve_hass_conf,
                self.optim_conf,
                self.plant_conf,
                params_json,
                emhass_conf,
                logger,
                get_data_from_file=True,
            )
        self.fcst.params = {
            "passed_data": {"weather_forecast_cache": False, "weather_forecast_cache_only": False}
        }
        self.data_dir = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(
            unittest.mock.patch.object(
                self.fcst, "emhass_conf", {**self.fcst.emhass_conf, "data_path": self.data_dir}
            )
        )
        self.cache_path = self.data_dir / "weather_forecast_data.pkl"
        # The real HTTPS hosts the fetchers call, served from the loopback interface.
        self.server = await self.enterAsyncContext(
            HttpTestServer(https=("api.solcast.com.au", "api.forecast.solar"))
        )
        self.enterContext(self.server.intercept(forecast_module))
        # Never touch the real Solcast daily quota counter.
        self.fcst._solcast_rate_limit_ok = lambda *a, **kw: True
        # A retried transport error must not sleep.
        self.enterContext(
            unittest.mock.patch.object(forecast_module, "forecast_solar_backoff_seconds", (0, 0, 0))
        )
        self.enterContext(
            unittest.mock.patch.object(forecast_module, "solcast_backoff_seconds", (0, 0, 0))
        )

    def set_grid(self, freq="15min", start="09:30:00", end="11:00:00"):
        """Point the forecast at a fixed local window [start, end) at the given freq."""
        tz = self.fcst.time_zone
        step = pd.Timedelta(freq)
        self.fcst.freq = step
        self.retrieve_hass_conf["optimization_time_step"] = step
        base = pd.Timestamp(f"2026-08-22 {start}", tz=tz)
        end_ts = pd.Timestamp(f"2026-08-22 {end}", tz=tz)
        self.fcst.forecast_dates = pd.date_range(start=base, end=end_ts - step, freq=step)

    def win(self, hhmm):
        return pd.Timestamp(f"2026-08-22 {hhmm}", tz=self.fcst.time_zone)

    def write_cache(self, frame):
        self.cache_path.write_bytes(pickle.dumps(frame))

    def cache_frame(self, columns):
        """A cache frame on the current grid; ``columns`` maps name -> value per step."""
        index = self.fcst.forecast_dates
        return pd.DataFrame(
            {name: np.full(len(index), float(value)) for name, value in columns.items()},
            index=index,
        )

    def two_plant_pvlib_conf(self):
        """Two PV plants with every sibling list at the matching length."""
        make_two_plant_pvlib_conf(self.plant_conf)

    def single_plant_conf(self, i):
        """The plant_conf with only plant ``i`` of the per-plant lists, as a plain value."""
        return {**self.plant_conf, **{key: self.plant_conf[key][i] for key in PLANT_LIST_KEYS}}

    def day_weather(self):
        """One local day at 30 minutes, with sun from 06:00 to 18:00 and none at night."""
        index = pd.date_range("2026-08-22 00:00", periods=48, freq="30min", tz=self.fcst.time_zone)
        hours = index.hour + index.minute / 60
        ghi = np.clip(800.0 * np.sin(np.pi * (hours - 6) / 12), 0.0, None)
        return pd.DataFrame(
            {
                "ghi": ghi,
                "dni": 0.7 * ghi,
                "dhi": 0.3 * ghi,
                "temp_air": 20.0,
                "wind_speed": 1.0,
            },
            index=index,
        )

    @staticmethod
    def solcast_payload(p1, p2, p3):
        """Three consecutive PT30M periods ending 10:00, 10:30 and 11:00 local (kW)."""
        elements = [
            {"period_end": f"2026-08-22T{t}.0000000Z", "period": "PT30M", "pv_estimate": est}
            for t, est in zip(("08:00:00", "08:30:00", "09:00:00"), (p1, p2, p3), strict=True)
        ]
        return {"forecasts": elements}

    async def fetch_solcast(self, payload_by_roof):
        """Fetch Solcast for the given rooftops {id: payload}; ids keep their order."""
        self.fcst.retrieve_hass_conf["solcast_api_key"] = "k"
        self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = ",".join(payload_by_roof)
        hours = int(len(self.fcst.forecast_dates) * self.fcst.freq.seconds / 3600)
        for roof_id, payload in payload_by_roof.items():
            url = f"https://api.solcast.com.au/rooftop_sites/{roof_id}/forecasts?hours={hours}"
            self.server.get(url, payload=payload)
        return await self.fcst.get_weather_forecast(method="solcast")

    def configure_planes(self, n_planes, kwp=5):
        self.plant_conf["pv_module_model"] = ["module"] * n_planes
        self.plant_conf["surface_tilt"] = [30 + 5 * i for i in range(n_planes)]
        self.plant_conf["surface_azimuth"] = [180 + 10 * i for i in range(n_planes)]
        self.retrieve_hass_conf["solar_forecast_kwp"] = kwp

    async def fetch_solar_forecast(self, watts_per_plane):
        """Fetch solar.forecast with one mocked response per plane, in plane order.

        Returns the frame and the URLs that were requested, in request order.
        """
        first_request = len(self.server.requests)
        for watts in watts_per_plane:
            self.server.get(SOLAR_FORECAST_URL, payload={"result": {"watts": watts}})
        frame = await self.fcst.get_weather_forecast(method="solar.forecast")
        return frame, [request.url for request in self.server.requests[first_request:]]


class TestSplitHelper(unittest.TestCase):
    """The pure split: shares against the clipped sum, fallbacks, invariants."""

    def split(self, aggregate, raw):
        return Forecast.split_pv_plants(aggregate, raw)

    def test_normal_shares(self):
        aggregate = _agg_series([100.0, 40.0, 0.0])
        raw = pd.DataFrame({"a": [60.0, 10.0, 0.0], "b": [20.0, 30.0, 0.0]}, index=aggregate.index)
        plants = self.split(aggregate, raw)
        self.assertEqual(list(plants.columns), ["P_PV_plant0", "P_PV_plant1"])
        self.assertTrue(plants.index.equals(aggregate.index))
        np.testing.assert_allclose(plants["P_PV_plant0"], [75.0, 10.0, 0.0])
        np.testing.assert_allclose(plants["P_PV_plant1"], [25.0, 30.0, 0.0])

    def test_adjusted_aggregate_keeps_the_raw_proportions(self):
        raw = pd.DataFrame({"a": [60.0, 10.0], "b": [20.0, 30.0]}, index=_agg_series([0, 0]).index)
        adjusted = _agg_series([160.0, 20.0])  # twice the raw sums 80 and 40
        plants = self.split(adjusted, raw)
        np.testing.assert_allclose(plants.sum(axis=1), adjusted)
        np.testing.assert_allclose(plants["P_PV_plant0"], [120.0, 5.0])

    def test_tare_plant_is_clipped_before_the_shares(self):
        """A plant at a small negative raw value (pvlib night tare) gets nothing, and the
        producing plant gets the whole aggregate; no share is negative or above one."""
        aggregate = _agg_series([50.0, 30.0])
        raw = pd.DataFrame({"a": [-0.55, -0.55], "b": [50.0, 30.0]}, index=aggregate.index)
        plants = self.split(aggregate, raw)
        self.assertTrue((plants >= 0).all().all())
        np.testing.assert_allclose(plants["P_PV_plant0"], [0.0, 0.0])
        np.testing.assert_allclose(plants["P_PV_plant1"], aggregate)
        np.testing.assert_allclose(plants.sum(axis=1), aggregate)

    def test_zero_raw_with_positive_aggregate_uses_the_horizon_share(self):
        aggregate = _agg_series([8.0, 40.0, 4.0])
        raw = pd.DataFrame({"a": [0.0, 10.0, -1.0], "b": [0.0, 30.0, 0.0]}, index=aggregate.index)
        plants = self.split(aggregate, raw)
        # Horizon totals after clipping are 10 and 30, so the shares are 1/4 and 3/4.
        np.testing.assert_allclose(plants["P_PV_plant0"], [2.0, 10.0, 1.0])
        np.testing.assert_allclose(plants["P_PV_plant1"], [6.0, 30.0, 3.0])

    def test_all_zero_horizon_splits_equally(self):
        aggregate = _agg_series([4.0, 0.0, 6.0])
        raw = pd.DataFrame(
            {"a": [0.0, -0.5, 0.0], "b": [0.0, 0.0, -0.2], "c": [0.0, 0.0, 0.0]},
            index=aggregate.index,
        )
        plants = self.split(aggregate, raw)
        np.testing.assert_allclose(plants.iloc[0], [4 / 3] * 3)
        np.testing.assert_allclose(plants.iloc[1], [0.0] * 3)
        np.testing.assert_allclose(plants.iloc[2], [2.0] * 3)

    def test_zero_aggregate_gives_zero_plants(self):
        aggregate = _agg_series([0.0, 0.0])
        raw = pd.DataFrame({"a": [100.0, 5.0], "b": [100.0, 0.0]}, index=aggregate.index)
        plants = self.split(aggregate, raw)
        self.assertTrue((plants == 0.0).all().all())

    def test_nan_aggregate_step_is_passed_on_to_every_plant(self):
        aggregate = _agg_series([10.0, np.nan, 20.0])
        raw = pd.DataFrame({"a": [1.0, 1.0, 3.0], "b": [1.0, 1.0, 1.0]}, index=aggregate.index)
        plants = self.split(aggregate, raw)
        self.assertTrue(plants.iloc[1].isna().all())
        self.assertFalse(plants.iloc[[0, 2]].isna().any().any())
        np.testing.assert_allclose(plants.iloc[[0, 2]].sum(axis=1), [10.0, 20.0])

    def test_non_finite_values_never_make_a_nan_share(self):
        aggregate = _agg_series([10.0, 10.0, np.inf, 5.0])
        raw = pd.DataFrame(
            {"a": [np.inf, np.nan, 1.0, -np.inf], "b": [5.0, 5.0, 1.0, 0.0]}, index=aggregate.index
        )
        plants = self.split(aggregate, raw)
        # A non-finite raw value counts as zero for the proportions.
        np.testing.assert_allclose(plants.iloc[0], [0.0, 10.0])
        np.testing.assert_allclose(plants.iloc[1], [0.0, 10.0])
        # An infinite aggregate is passed on, not turned into NaN by 0 * inf.
        self.assertTrue(np.isinf(plants.iloc[2]).all())
        # Nothing from any plant at this step, so the aggregate is shared by the horizon
        # totals (a 1, b 11); the plants still add up to it.
        np.testing.assert_allclose(plants.iloc[3].sum(), 5.0)
        self.assertFalse(plants.iloc[[0, 1, 3]].isna().any().any())

    def test_fewer_than_two_plants_gives_none(self):
        aggregate = _agg_series([1.0, 2.0])
        one_plant = pd.DataFrame({"a": [1.0, 2.0]}, index=aggregate.index)
        self.assertIsNone(self.split(aggregate, one_plant))
        self.assertIsNone(self.split(aggregate, pd.DataFrame(index=aggregate.index)))
        self.assertIsNone(self.split(aggregate, None))

    def test_inputs_are_not_modified(self):
        aggregate = _agg_series([10.0, 0.0, 5.0])
        raw = pd.DataFrame({"a": [-1.0, 3.0, 0.0], "b": [4.0, 0.0, 0.0]}, index=aggregate.index)
        aggregate_before, raw_before = aggregate.copy(), raw.copy()
        self.split(aggregate, raw)
        assert_series_equal(aggregate, aggregate_before)
        assert_frame_equal(raw, raw_before)

    def test_invariant_holds_for_random_frames(self):
        """Every finite step: all plants >= 0 and the plants add up to the aggregate."""
        rng = np.random.default_rng(1150)
        for n_plants in (2, 3, 5):
            for _ in range(20):
                steps = 48
                raw_values = rng.normal(0.0, 400.0, size=(steps, n_plants))
                raw_values[rng.random((steps, n_plants)) < 0.3] = 0.0
                aggregate_values = np.abs(rng.normal(0.0, 800.0, size=steps))
                aggregate_values[rng.random(steps) < 0.2] = 0.0
                aggregate_values[rng.random(steps) < 0.05] = np.nan
                aggregate = _agg_series(aggregate_values)
                raw = pd.DataFrame(raw_values, index=aggregate.index)
                plants = self.split(aggregate, raw)
                finite = aggregate.notna().to_numpy()
                self.assertTrue((plants[finite] >= 0).all().all())
                np.testing.assert_allclose(
                    plants[finite].sum(axis=1), aggregate[finite], rtol=1e-12, atol=1e-9
                )
                self.assertTrue(plants[~finite].isna().all().all())


class TestPvlibPlants(PerPlantBase):
    """pvlib: the per-plant AC power that is summed today is kept and split."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.two_plant_pvlib_conf()
        self.fcst.weather_forecast_method = "open-meteo"

    def test_two_plants_give_two_columns_that_add_up_to_the_aggregate(self):
        weather = self.day_weather()
        p_pv = self.fcst.get_power_from_weather(weather)
        plants = self.fcst.get_pv_plant_split(p_pv)
        self.assertIsNotNone(plants, "two pvlib plants must give plant columns")
        self.assertEqual(list(plants.columns), ["P_PV_plant0", "P_PV_plant1"])
        self.assertTrue(plants.index.equals(p_pv.index))
        self.assertGreater(p_pv.max(), 0)
        self.assertTrue((plants >= 0).all().all())
        np.testing.assert_allclose(plants.sum(axis=1), p_pv, rtol=1e-12, atol=1e-9)
        # The two orientations differ, so the plants are not just a copy of each other.
        self.assertFalse(np.allclose(plants["P_PV_plant0"], plants["P_PV_plant1"]))

    def test_aggregate_equals_the_sum_of_the_plants_computed_one_by_one(self):
        """No-op pin: the multi-plant aggregate is what summing individually computed plants
        gives (0 + plant0 + plant1, then the negative clip), as before per-plant capture."""
        weather = self.day_weather()
        expected = pd.Series(0, index=weather.index)
        for i in range(2):
            with unittest.mock.patch.object(self.fcst, "plant_conf", self.single_plant_conf(i)):
                expected = expected + self.fcst._calculate_pvlib_power(weather)
        expected[expected < 0] = 0
        actual = self.fcst.get_power_from_weather(weather)
        self.assertIsInstance(actual, pd.Series)
        assert_series_equal(actual, expected, check_names=False, check_exact=True)

    def test_night_tare_keeps_every_plant_non_negative(self):
        weather = self.day_weather()
        p_pv = self.fcst.get_power_from_weather(weather)
        plants = self.fcst.get_pv_plant_split(p_pv)
        self.assertIsNotNone(plants)
        night = plants.between_time("00:00", "04:30")
        self.assertGreater(len(night), 0)
        self.assertTrue((night == 0.0).all().all())
        self.assertTrue((plants >= 0).all().all())

    def test_each_plant_column_is_that_plant_alone(self):
        """Each plant's raw series is that plant simulated on its own, and where every plant
        is producing the split gives back exactly those values (not a running sum)."""
        weather = self.day_weather()
        singles = []
        for i in range(2):
            with unittest.mock.patch.object(self.fcst, "plant_conf", self.single_plant_conf(i)):
                singles.append(self.fcst._calculate_pvlib_power(weather))
        p_pv = self.fcst.get_power_from_weather(weather)
        raw = self.fcst.pv_plant_raw
        for j, single in enumerate(singles):
            np.testing.assert_array_equal(raw.iloc[:, j].to_numpy(), single.to_numpy())
        plants = self.fcst.get_pv_plant_split(p_pv)
        producing = (singles[0] > 0) & (singles[1] > 0)
        self.assertTrue(producing.any())
        for j, single in enumerate(singles):
            np.testing.assert_allclose(
                plants[f"P_PV_plant{j}"][producing], single[producing], rtol=1e-12
            )

    def test_the_carrier_is_reset_on_every_call(self):
        self.fcst.get_power_from_weather(self.day_weather())
        self.assertIsNotNone(
            self.fcst.get_pv_plant_split(pd.Series(0.0, index=self.day_weather().index))
        )
        # A later call that has no plant series must not serve the earlier ones. The zero
        # kWp branch with a single plant sets no plant data itself, so only the reset at
        # the start of the call clears the two plants of the first call.
        for key in PLANT_LIST_KEYS:
            self.fcst.plant_conf[key] = self.fcst.plant_conf[key][:1]
        self.fcst.retrieve_hass_conf["solar_forecast_kwp"] = 0
        p_pv = self.fcst.get_power_from_weather(self.day_weather())
        self.assertIsNone(self.fcst.pv_plant_raw)
        self.assertIsNone(self.fcst.get_pv_plant_split(p_pv))

    def test_a_single_plant_has_no_plant_data(self):
        for key in PLANT_LIST_KEYS:
            self.plant_conf[key] = self.plant_conf[key][0]
        p_pv = self.fcst.get_power_from_weather(self.day_weather())
        self.assertIsNone(self.fcst.get_pv_plant_split(p_pv))
        # A list with one plant is one plant as well.
        for key in PLANT_LIST_KEYS:
            self.plant_conf[key] = [self.plant_conf[key]]
        p_pv = self.fcst.get_power_from_weather(self.day_weather())
        self.assertIsNone(self.fcst.get_pv_plant_split(p_pv))

    def test_zero_kwp_branch_still_gives_zero_plants(self):
        """solar_forecast_kwp == 0 zeroes the PV for every method; the plants follow."""
        self.retrieve_hass_conf["solar_forecast_kwp"] = 0
        p_pv = self.fcst.get_power_from_weather(self.day_weather())
        self.assertTrue((p_pv == 0).all())
        plants = self.fcst.get_pv_plant_split(p_pv)
        self.assertIsNotNone(plants)
        self.assertEqual(plants.shape, (len(p_pv), 2))
        self.assertTrue((plants == 0.0).all().all())

    def test_a_length_mismatch_leaves_the_plants_out_with_a_warning(self):
        p_pv = self.fcst.get_power_from_weather(self.day_weather())
        with self.assertLogs(self.fcst.logger, level="WARNING") as logs:
            self.assertIsNone(self.fcst.get_pv_plant_split(p_pv.iloc[:-3]))
        self.assertIn("P_PV_plant columns are left out", chr(10).join(logs.output))

    def test_zero_kwp_works_before_get_weather_forecast_was_called(self):
        """Scripts call get_power_from_weather on its own, so the method is not set yet."""
        del self.fcst.weather_forecast_method
        self.retrieve_hass_conf["solar_forecast_kwp"] = 0
        p_pv = self.fcst.get_power_from_weather(self.day_weather())
        self.assertTrue((p_pv == 0).all())
        self.assertEqual(self.fcst.get_pv_plant_split(p_pv).shape, (len(p_pv), 2))

    def test_split_follows_the_mixed_and_adjusted_aggregate(self):
        """The plants add up to the aggregate after the step 0 mix and after an adjustment
        that lifts a step the raw plants had at zero."""
        weather = self.day_weather()
        noon = weather.index.get_loc(pd.Timestamp("2026-08-22 12:00", tz=self.fcst.time_zone))
        weather = weather.iloc[noon - 1 :]  # step 0 is 11:30, with sun
        df_now = pd.DataFrame({self.fcst.var_pv: [123.0]})
        self.fcst.params["passed_data"]["alpha"] = 0.5
        self.fcst.params["passed_data"]["beta"] = 0.5
        p_pv = self.fcst.get_power_from_weather(weather, set_mix_forecast=True, df_now=df_now)
        plants = self.fcst.get_pv_plant_split(p_pv)
        self.assertIsNotNone(plants)
        np.testing.assert_allclose(plants.sum(axis=1), p_pv, rtol=1e-12, atol=1e-9)
        # An adjustment that doubles the aggregate and lifts the (zero) night step.
        adjusted = p_pv * 2.0
        adjusted.iloc[-1] = 77.0
        self.assertEqual(float(p_pv.iloc[-1]), 0.0)
        plants = self.fcst.get_pv_plant_split(adjusted)
        np.testing.assert_allclose(plants.sum(axis=1), adjusted, rtol=1e-12, atol=1e-9)
        self.assertTrue((plants >= 0).all().all())


class TestSolcastPlants(PerPlantBase):
    """Solcast: plant j is rooftop j in solcast_rooftop_id order; the plants ride the cache."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.set_grid("15min")
        self.roofs = {
            "111111": self.solcast_payload(0.2, 0.8, 0.5),
            "222222": self.solcast_payload(0.1, 0.3, 0.2),
        }

    async def test_two_rooftops_give_two_columns_that_add_up_to_the_aggregate(self):
        frame = await self.fetch_solcast(self.roofs)
        self.assertIn("yhat_plant0", frame.columns)
        self.assertIn("yhat_plant1", frame.columns)
        p_pv = self.fcst.get_power_from_weather(frame)
        plants = self.fcst.get_pv_plant_split(p_pv)
        self.assertIsNotNone(plants, "two Solcast rooftops must give plant columns")
        self.assertEqual(list(plants.columns), ["P_PV_plant0", "P_PV_plant1"])
        # Rooftop order: the first ID is plant 0.
        for hhmm, first, second in (("09:30", 200.0, 100.0), ("10:00", 800.0, 300.0)):
            self.assertAlmostEqual(plants.loc[self.win(hhmm), "P_PV_plant0"], first, places=6)
            self.assertAlmostEqual(plants.loc[self.win(hhmm), "P_PV_plant1"], second, places=6)
        np.testing.assert_allclose(plants.sum(axis=1), p_pv, rtol=1e-12, atol=1e-9)

    async def test_the_rooftop_order_is_the_id_order(self):
        reversed_roofs = dict(reversed(list(self.roofs.items())))
        frame = await self.fetch_solcast(reversed_roofs)
        p_pv = self.fcst.get_power_from_weather(frame)
        plants = self.fcst.get_pv_plant_split(p_pv)
        self.assertIsNotNone(plants)
        self.assertAlmostEqual(plants.loc[self.win("10:00"), "P_PV_plant0"], 300.0, places=6)
        self.assertAlmostEqual(plants.loc[self.win("10:00"), "P_PV_plant1"], 800.0, places=6)

    async def test_yhat_is_unchanged_by_the_plant_columns(self):
        """No-op pin: the aggregate is the rooftop sum, per step, as it always was."""
        frame = await self.fetch_solcast(self.roofs)
        for hhmm, expected in (("09:30", 300.0), ("10:00", 1100.0), ("10:30", 700.0)):
            self.assertAlmostEqual(frame.loc[self.win(hhmm), "yhat"], expected, places=6)
        p_pv = self.fcst.get_power_from_weather(frame)
        self.assertIsInstance(p_pv, pd.Series)
        assert_series_equal(p_pv, frame["yhat"], check_names=False)

    async def test_a_single_rooftop_writes_no_plant_columns(self):
        frame = await self.fetch_solcast({"111111": self.roofs["111111"]})
        self.assertEqual(list(frame.columns), ["yhat"])
        p_pv = self.fcst.get_power_from_weather(frame)
        self.assertIsNone(self.fcst.pv_plant_raw)
        self.assertAlmostEqual(p_pv.loc[self.win("10:00")], 800.0, places=6)

    async def test_the_plants_are_written_to_the_cache_next_to_yhat(self):
        self.fcst.params["passed_data"]["weather_forecast_cache"] = True
        await self.fetch_solcast(self.roofs)
        cached = pickle.loads(self.cache_path.read_bytes())
        self.assertEqual(list(cached.columns), ["yhat", "yhat_plant0", "yhat_plant1"])
        np.testing.assert_allclose(
            cached["yhat_plant0"] + cached["yhat_plant1"], cached["yhat"], atol=1e-9
        )

    async def test_zero_kwp_zeroes_the_plants_too(self):
        frame = await self.fetch_solcast(self.roofs)
        self.retrieve_hass_conf["solar_forecast_kwp"] = 0
        p_pv = self.fcst.get_power_from_weather(frame)
        self.assertTrue((p_pv == 0).all())
        plants = self.fcst.get_pv_plant_split(p_pv)
        self.assertIsNotNone(plants)
        self.assertTrue((plants == 0.0).all().all())

    async def test_mix_and_clip_do_not_reach_the_plant_data(self):
        """The aggregate is a view of the frame's yhat column that the mix writes into; the
        plant data must be copies, so it still holds the fetched values afterwards."""
        frame = await self.fetch_solcast(self.roofs)
        before = frame[["yhat_plant0", "yhat_plant1"]].copy()
        self.fcst.params["passed_data"]["alpha"] = 0.5
        self.fcst.params["passed_data"]["beta"] = 0.5
        df_now = pd.DataFrame({self.fcst.var_pv: [1000.0]})
        p_pv = self.fcst.get_power_from_weather(frame, set_mix_forecast=True, df_now=df_now)
        self.assertEqual(float(p_pv.iloc[0]), 650.0)  # 0.5 * 300 + 0.5 * 1000
        raw = self.fcst.pv_plant_raw
        assert_frame_equal(raw, before.set_axis(raw.columns, axis=1), check_names=False)
        # Owned memory, not a view of the frame: a value check alone cannot tell under
        # pandas copy-on-write, where a view is protected from the in-place writes.
        for col in raw.columns:
            for source in ("yhat", "yhat_plant0", "yhat_plant1"):
                self.assertFalse(
                    np.shares_memory(raw[col].to_numpy(), frame[source].to_numpy()),
                    f"{col} shares memory with {source}",
                )
        plants = self.fcst.get_pv_plant_split(p_pv)
        np.testing.assert_allclose(plants.sum(axis=1), p_pv, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(plants.iloc[0], [650.0 * 2 / 3, 650.0 / 3])


class TestSolarForecastPlants(PerPlantBase):
    """solar.forecast: plant j is plane j, the order of pv_module_model."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.set_grid("30min", start="10:00:00", end="12:30:00")
        self.configure_planes(2)

    async def test_two_planes_give_two_columns_that_add_up_to_the_aggregate(self):
        frame, _ = await self.fetch_solar_forecast(
            [SOLAR_FORECAST_WATTS["a"], SOLAR_FORECAST_WATTS["b"]]
        )
        self.assertIn("yhat_plant0", frame.columns)
        self.assertIn("yhat_plant1", frame.columns)
        p_pv = self.fcst.get_power_from_weather(frame)
        plants = self.fcst.get_pv_plant_split(p_pv)
        self.assertIsNotNone(plants, "two solar.forecast planes must give plant columns")
        self.assertAlmostEqual(plants.loc[self.win("10:30"), "P_PV_plant0"], 150.0, places=6)
        self.assertAlmostEqual(plants.loc[self.win("10:30"), "P_PV_plant1"], 25.0, places=6)
        np.testing.assert_allclose(plants.sum(axis=1), p_pv, rtol=1e-12, atol=1e-9)

    async def test_yhat_is_unchanged_by_the_plant_columns(self):
        """No-op pin: the aggregate is the sum of the planes, as it always was."""
        frame, _ = await self.fetch_solar_forecast(
            [SOLAR_FORECAST_WATTS["a"], SOLAR_FORECAST_WATTS["b"]]
        )
        for hhmm, expected in (("10:00", 110.0), ("10:30", 175.0), ("11:00", 240.0)):
            self.assertAlmostEqual(frame.loc[self.win(hhmm), "yhat"], expected, places=6)
        p_pv = self.fcst.get_power_from_weather(frame)
        assert_series_equal(p_pv, frame["yhat"], check_names=False)

    async def test_a_single_plane_writes_no_plant_columns(self):
        self.configure_planes(1)
        frame, _ = await self.fetch_solar_forecast([SOLAR_FORECAST_WATTS["a"]])
        self.assertEqual(list(frame.columns), ["yhat"])

    async def test_an_error_status_ends_the_fetch_without_a_retry(self):
        self.configure_planes(1)
        for status in (429, 503):
            with self.subTest(status=status):
                first_request = len(self.server.requests)
                # A JSON body, so only the status tells this apart from a good reply.
                self.server.get(SOLAR_FORECAST_URL, status=status, payload={"message": "refused"})
                with self.assertLogs(self.fcst.logger, level="ERROR") as logs:
                    result = await self.fcst.get_weather_forecast(method="solar.forecast")
                self.assertIs(result, False)
                self.assertEqual(len(self.server.requests) - first_request, 1)
                self.assertTrue(any(f"HTTP {status}" in line for line in logs.output))

    async def test_the_plants_are_written_to_the_cache_next_to_yhat(self):
        self.fcst.params["passed_data"]["weather_forecast_cache"] = True
        await self.fetch_solar_forecast([SOLAR_FORECAST_WATTS["a"], SOLAR_FORECAST_WATTS["b"]])
        cached = pickle.loads(self.cache_path.read_bytes())
        self.assertEqual(list(cached.columns), ["yhat", "yhat_plant0", "yhat_plant1"])


class TestCacheReader(PerPlantBase):
    """A cache with the plant columns gives plant data; an old or mismatched one is served
    for yhat exactly as before, with one info log and nothing dropped or fetched."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.set_grid("30min", start="10:00:00", end="12:30:00")
        self.fcst.retrieve_hass_conf["solcast_api_key"] = "k"
        self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = "111111,222222"
        self.configure_planes(2)

    async def read_cache(self, method):
        """Serve the cache through the real path; no request may leave."""
        with self.assertLogs(self.fcst.logger, level="INFO") as logs:
            frame = await self.fcst.get_weather_forecast(method=method)
            p_pv = self.fcst.get_power_from_weather(frame)
        self.assertEqual(self.server.requests, [], "a cache read must not fetch")
        self.assertTrue(self.cache_path.exists(), "a cache read must not drop the cache")
        notices = [line for line in logs.output if "without the P_PV_plant columns" in line]
        return frame, p_pv, notices

    async def test_new_cache_gives_plant_data_for_both_methods(self):
        for method in ("solcast", "solar.forecast"):
            with self.subTest(method=method):
                self.write_cache(
                    self.cache_frame({"yhat": 30.0, "yhat_plant0": 10.0, "yhat_plant1": 20.0})
                )
                _, p_pv, notices = await self.read_cache(method)
                plants = self.fcst.get_pv_plant_split(p_pv)
                self.assertIsNotNone(plants)
                np.testing.assert_allclose(plants["P_PV_plant0"], 10.0)
                np.testing.assert_allclose(plants["P_PV_plant1"], 20.0)
                self.assertEqual(notices, [])

    async def test_old_cache_is_served_as_before_with_one_info_log(self):
        for method in ("solcast", "solar.forecast"):
            with self.subTest(method=method):
                self.write_cache(self.cache_frame({"yhat": 30.0}))
                frame, p_pv, notices = await self.read_cache(method)
                self.assertEqual(list(frame.columns), ["yhat"])
                np.testing.assert_allclose(p_pv, 30.0)
                self.assertIsNone(self.fcst.get_pv_plant_split(p_pv))
                self.assertEqual(len(notices), 1, notices)
                self.assertIn("INFO", notices[0])
                # The refresh that actually works for each method: solar.forecast serves an
                # existing cache file even on the weather-forecast-cache action.
                if method == "solcast":
                    self.assertIn("weather-forecast-cache action", notices[0])
                else:
                    self.assertIn("Delete weather_forecast_data.pkl", notices[0])
                    self.assertNotIn("weather-forecast-cache", notices[0])

    async def test_plant_count_mismatch_is_served_as_before_with_one_info_log(self):
        mismatched = (
            {"yhat": 30.0, "yhat_plant0": 30.0},  # one plant column for two plants
            {"yhat": 30.0, "yhat_plant0": 10.0, "yhat_plant1": 10.0, "yhat_plant2": 10.0},
        )
        for method in ("solcast", "solar.forecast"):
            for columns in mismatched:
                with self.subTest(method=method, columns=list(columns)):
                    self.write_cache(self.cache_frame(columns))
                    _, p_pv, notices = await self.read_cache(method)
                    np.testing.assert_allclose(p_pv, 30.0)
                    self.assertIsNone(self.fcst.get_pv_plant_split(p_pv))
                    self.assertEqual(len(notices), 1, notices)

    async def test_a_single_plant_site_logs_nothing_for_an_old_cache(self):
        self.configure_planes(1)
        self.fcst.retrieve_hass_conf["solcast_rooftop_id"] = "111111"
        for method in ("solcast", "solar.forecast"):
            with self.subTest(method=method):
                self.write_cache(self.cache_frame({"yhat": 30.0}))
                _, p_pv, notices = await self.read_cache(method)
                np.testing.assert_allclose(p_pv, 30.0)
                self.assertIsNone(self.fcst.get_pv_plant_split(p_pv))
                self.assertEqual(notices, [])

    async def test_a_stale_cache_keeps_the_plants_adding_up(self):
        """The cache is served stale (reindexed and filled) when it does not cover the grid;
        the plant columns go through the same fill, so the split still adds up."""
        index = self.fcst.forecast_dates[:3]
        frame = pd.DataFrame({"yhat": 30.0, "yhat_plant0": 10.0, "yhat_plant1": 20.0}, index=index)
        self.write_cache(frame)
        _, p_pv, _ = await self.read_cache("solcast")
        plants = self.fcst.get_pv_plant_split(p_pv)
        self.assertIsNotNone(plants)
        np.testing.assert_allclose(plants.sum(axis=1), p_pv, rtol=1e-12, atol=1e-9)

    async def test_csv_and_list_never_discover_plants_from_columns(self):
        frame = self.cache_frame({"yhat": 30.0, "yhat_plant0": 10.0, "yhat_plant1": 20.0})
        for method in ("csv", "list"):
            with self.subTest(method=method):
                self.fcst.weather_forecast_method = method
                p_pv = self.fcst.get_power_from_weather(frame.copy())
                self.assertIsNone(self.fcst.get_pv_plant_split(p_pv))

    async def test_augment_skips_the_plant_columns(self):
        weather = pd.DataFrame(
            {
                "ghi": 400.0,
                "temp_air": 18.0,
                "yhat": 5.0,
                "yhat_plant0": 2.0,
                "yhat_plant1": 3.0,
            },
            index=self.fcst.forecast_dates,
        )
        passed = pd.DataFrame({"yhat": 77.0}, index=self.fcst.forecast_dates)
        with unittest.mock.patch.object(self.fcst, "_get_weather_open_meteo", return_value=weather):
            result = await self.fcst._augment_list_with_open_meteo(
                passed, str(self.cache_path), False
            )
        self.assertEqual(list(result.columns), ["yhat", "ghi", "temp_air"])
        np.testing.assert_allclose(result["yhat"], 77.0)


class TestSolarForecastKwp(PerPlantBase):
    """solar_forecast_kwp: a scalar keeps its meaning, a list is one value per plane."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.set_grid("30min", start="10:00:00", end="12:30:00")
        self.configure_planes(2)
        self.watts = SOLAR_FORECAST_WATTS

    def kwp_segments(self, urls):
        """The last URL segment (the kWp) of each request, in request order."""
        return [url.rstrip("/").split("/")[-1] for url in urls]

    async def test_a_list_sends_each_plane_its_own_kwp(self):
        self.retrieve_hass_conf["solar_forecast_kwp"] = [3.5, 4]
        frame, urls = await self.fetch_solar_forecast([self.watts["a"], self.watts["b"]])
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(self.kwp_segments(urls), ["3.5", "4"])
        # The planes keep their own tilt and azimuth.
        self.assertIn("/30/0/3.5", urls[0])
        self.assertIn("/35/10/4", urls[1])

    async def test_a_scalar_sends_the_same_kwp_to_every_plane(self):
        self.retrieve_hass_conf["solar_forecast_kwp"] = 7
        _, urls = await self.fetch_solar_forecast([self.watts["a"], self.watts["b"]])
        self.assertEqual(self.kwp_segments(urls), ["7", "7"])

    async def test_a_stringified_list_is_read_as_a_list(self):
        for text in ("[3, 4]", "  [3, 4]  ", "[3.0, 4.0]"):
            with self.subTest(text=text):
                self.retrieve_hass_conf["solar_forecast_kwp"] = text
                frame, urls = await self.fetch_solar_forecast([self.watts["a"], self.watts["b"]])
                self.assertIsInstance(frame, pd.DataFrame)
                self.assertEqual(
                    self.kwp_segments(urls), [s.strip() for s in text.strip("[] ").split(",")]
                )

    async def test_a_stringified_list_that_cannot_be_used_fails_before_any_request(self):
        for text in ("[3,", "[3, 4", "[3] + [4]", "[__import__('os')]", "[[3], 4]", "[]", "[3]"):
            with self.subTest(text=text):
                self.retrieve_hass_conf["solar_forecast_kwp"] = text
                with self.assertLogs(self.fcst.logger, level="ERROR") as logs:
                    result = await self.fcst.get_weather_forecast(method="solar.forecast")
                self.assertEqual(self.server.requests, [], "no request may be made")
                self.assertIs(result, False)
                self.assertTrue(any("solar_forecast_kwp" in line for line in logs.output))
                self.assertTrue(all(len(line) < MAX_LOG_LINE for line in logs.output), logs.output)

    async def test_a_string_that_is_not_a_list_keeps_its_old_meaning(self):
        for text in ("5", "abc"):
            with self.subTest(text=text):
                self.retrieve_hass_conf["solar_forecast_kwp"] = text
                _, urls = await self.fetch_solar_forecast([self.watts["a"], self.watts["b"]])
                self.assertEqual(self.kwp_segments(urls), [text, text])

    async def conf_from_yaml_file(self, yaml_text):
        """retrieve_hass_conf built from a secrets-style yaml file with the solar.forecast method."""
        yaml_file = self.data_dir / "kwp_test.yaml"
        yaml_file.write_text(yaml_text)
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        config["weather_forecast_method"] = "solar.forecast"
        _, secrets = await utils.build_secrets(
            emhass_conf, logger, secrets_path=str(yaml_file), no_response=True
        )
        params = await utils.build_params(emhass_conf, secrets, config, logger)
        retrieve_hass_conf, _, _ = utils.get_yaml_parse(
            orjson.dumps(params).decode("utf-8"), logger
        )
        return retrieve_hass_conf

    async def test_a_runtime_list_reaches_the_forecaster_through_treat_runtimeparams(self):
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
        params = await utils.build_params(emhass_conf, secrets, config, logger)
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        with fixture_clock():
            _, retrieve_hass_conf, optim_conf, _ = await utils.treat_runtimeparams(
                {"weather_forecast_method": "solar.forecast", "solar_forecast_kwp": [3, 4]},
                params_json,
                retrieve_hass_conf,
                optim_conf,
                plant_conf,
                "dayahead-optim",
                logger,
                emhass_conf,
            )
        self.assertEqual(optim_conf["weather_forecast_method"], "solar.forecast")
        self.assertEqual(retrieve_hass_conf["solar_forecast_kwp"], [3, 4])
        self.fcst.retrieve_hass_conf = retrieve_hass_conf
        _, urls = await self.fetch_solar_forecast([self.watts["a"], self.watts["b"]])
        self.assertEqual(self.kwp_segments(urls), ["3", "4"])

    async def test_a_list_in_the_secrets_file_reaches_the_forecaster(self):
        for label, line in (
            ("yaml list", "solar_forecast_kwp: [3, 4]"),
            ("quoted list", "solar_forecast_kwp: '[3, 4]'"),
        ):
            with self.subTest(label):
                self.fcst.retrieve_hass_conf = await self.conf_from_yaml_file(line + "\n")
                _, urls = await self.fetch_solar_forecast([self.watts["a"], self.watts["b"]])
                self.assertEqual(self.kwp_segments(urls), ["3", "4"])

    async def test_a_missing_or_zero_scalar_still_becomes_five(self):
        del self.retrieve_hass_conf["solar_forecast_kwp"]
        _, urls = await self.fetch_solar_forecast([self.watts["a"], self.watts["b"]])
        self.assertEqual(self.kwp_segments(urls), ["5", "5"])
        self.assertEqual(self.retrieve_hass_conf["solar_forecast_kwp"], 5)
        self.retrieve_hass_conf["solar_forecast_kwp"] = 0
        _, urls = await self.fetch_solar_forecast([self.watts["a"], self.watts["b"]])
        self.assertEqual(self.kwp_segments(urls), ["5", "5"])
        self.assertEqual(self.retrieve_hass_conf["solar_forecast_kwp"], 5)

    def test_a_zero_scalar_still_gives_zero_pv(self):
        """The zero-PV branch of get_power_from_weather is the scalar's, unchanged."""
        self.fcst.weather_forecast_method = "solar.forecast"
        self.retrieve_hass_conf["solar_forecast_kwp"] = 0
        frame = pd.DataFrame({"yhat": 100.0}, index=self.fcst.forecast_dates)
        p_pv = self.fcst.get_power_from_weather(frame)
        self.assertTrue((p_pv == 0).all())
        self.assertEqual(len(p_pv), len(frame))

    async def test_a_bad_list_fails_the_forecast_before_any_request(self):
        bad_lists = {
            "too short": [3.5],
            "too long": [3.5, 4, 5],
            "empty": [],
            "zero entry": [3.5, 0],
            "negative entry": [-1, 4],
            "bool entry": [True, 4],
            "nan entry": [3.5, float("nan")],
            "inf entry": [float("inf"), 4],
            "string entry": ["3.5", 4],
            "none entry": [None, 4],
            "nested entry": [[3.5], 4],
            "huge int entry": [10**400, 4],
        }
        for label, bad in bad_lists.items():
            with self.subTest(label):
                self.retrieve_hass_conf["solar_forecast_kwp"] = bad
                with self.assertLogs(self.fcst.logger, level="ERROR") as logs:
                    result = await self.fcst.get_weather_forecast(method="solar.forecast")
                self.assertEqual(self.server.requests, [], "no request may be made")
                self.assertIs(result, False)
                self.assertTrue(any("solar_forecast_kwp" in line for line in logs.output))

    async def test_the_error_does_not_echo_a_long_list(self):
        self.retrieve_hass_conf["solar_forecast_kwp"] = [1.0] * 100_000
        with self.assertLogs(self.fcst.logger, level="ERROR") as logs:
            result = await self.fcst.get_weather_forecast(method="solar.forecast")
        self.assertIs(result, False)
        self.assertTrue(all(len(line) < MAX_LOG_LINE for line in logs.output), logs.output)

    async def test_a_list_cannot_make_a_non_positive_scalar_slip_through(self):
        """A list is never treated as the scalar 0: no warning about 'setting to default 5'."""
        self.retrieve_hass_conf["solar_forecast_kwp"] = [3.5, 4]
        with self.assertNoLogs(self.fcst.logger, level="WARNING"):
            await self.fetch_solar_forecast([self.watts["a"], self.watts["b"]])
        self.assertEqual(self.retrieve_hass_conf["solar_forecast_kwp"], [3.5, 4])

    async def test_a_bad_list_is_refused_even_when_a_cache_exists(self):
        self.write_cache(self.cache_frame({"yhat": 30.0, "yhat_plant0": 10.0, "yhat_plant1": 20.0}))
        cache_before = self.cache_path.read_bytes()
        for label, bad in {"wrong length": [3.5], "zero entry": [3.5, 0], "text": "[3"}.items():
            with self.subTest(label):
                self.retrieve_hass_conf["solar_forecast_kwp"] = bad
                with self.assertLogs(self.fcst.logger, level="ERROR") as logs:
                    result = await self.fcst.get_weather_forecast(method="solar.forecast")
                self.assertEqual(self.server.requests, [], "no request may be made")
                self.assertIs(result, False)
                self.assertTrue(any("solar_forecast_kwp" in line for line in logs.output))
                self.assertEqual(self.cache_path.read_bytes(), cache_before)

    async def test_a_good_list_is_served_from_the_cache(self):
        self.write_cache(self.cache_frame({"yhat": 30.0, "yhat_plant0": 10.0, "yhat_plant1": 20.0}))
        self.retrieve_hass_conf["solar_forecast_kwp"] = [3.5, 4]
        frame = await self.fcst.get_weather_forecast(method="solar.forecast")
        self.assertEqual(self.server.requests, [], "a cache read must not fetch")
        self.assertIsInstance(frame, pd.DataFrame)
        np.testing.assert_allclose(frame["yhat"], 30.0)

    def test_describe_invalid_solar_forecast_kwp_directly(self):
        check = forecast_module._describe_invalid_solar_forecast_kwp
        self.assertIsNone(check([1, 2.5], 2))
        self.assertIn("1 entries but pv_module_model has 2", check([1], 2))
        self.assertIn("entry 0 is zero, negative or not finite", check([0, 4], 2))
        self.assertIn("entry 1 is zero, negative or not finite", check([3, -3], 2))
        self.assertIn("entry 0 is zero, negative or not finite", check([math.nan], 1))
        self.assertIn("entry 0 is a bool", check([False], 1))
        self.assertIn("entry 0 is a str", check(["3"], 1))
        # Every bad position is named, up to five, then a count.
        message = check([0] * 12, 12)
        self.assertIn("entry 4", message)
        self.assertNotIn("entry 5", message)
        self.assertIn("(and 7 more)", message)
        # A huge integer is a bad entry, not an exception, and is never echoed.
        message = check([10**5000, 4], 2)
        self.assertIn("entry 0 is zero, negative or not finite", message)
        self.assertLess(len(message), MAX_LOG_LINE)
