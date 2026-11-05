#!/usr/bin/env python
"""Per-plant PV forecast (#1150): the plant columns in the optimization results.

End to end through ``set_input_data_dict`` and the optim actions, with a two-plant pvlib site:
the ``P_PV_plant{j}`` columns next to ``P_PV``, the CSV and plan, publish_data unchanged, the
adjustment, a single plant and perfect-optim, and the plan schema version.
"""

import copy
import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
import orjson
import pandas as pd
from pandas.testing import assert_frame_equal

from emhass import last_run, plan_store, utils
from emhass.command_line import (
    EMHASS_SCHEMA_VERSION,
    dayahead_forecast_optim,
    naive_mpc_optim,
    perfect_forecast_optim,
    publish_data,
    set_input_data_dict,
)
from emhass.forecast import Forecast
from tests import test_command_line_utils as cl_tests
from tests.fixture_clock import fixture_clock
from tests.test_per_plant_pv import make_two_plant_pvlib_conf

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
logger, ch = utils.get_logger(__name__, cl_tests.emhass_conf, save_to_file=False)

PLANT_COLUMNS = ["P_PV_plant0", "P_PV_plant1"]
COSTFUN = "profit"

# The columns of a default single-plant day-ahead run (no battery, no thermal load), in order.
# A single plant must keep exactly this set.
SINGLE_PLANT_DAYAHEAD_COLUMNS = [
    "P_PV",
    "P_Load",
    "P_grid_pos",
    "P_grid_neg",
    "P_grid",
    "P_deferrable0",
    "P_deferrable1",
    "unit_load_cost",
    "unit_prod_price",
    "maximum_power_from_grid",
    "maximum_power_to_grid",
    "cost_profit",
    "cost_fun_profit",
    "optim_status",
]


def _plant_columns(frame):
    return [col for col in frame.columns if str(col).startswith("P_PV_plant")]


class PerPlantResultsBase(unittest.IsolatedAsyncioTestCase):
    """A two-plant pvlib site run through the real actions, on a throwaway data directory."""

    async def asyncSetUp(self):
        data_path = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        shutil.copytree(root / "data", data_path, dirs_exist_ok=True)
        self.emhass_conf = {**cl_tests.emhass_conf, "data_path": data_path}
        self.data_path = data_path
        # The plan and last-run caches are module globals: do not leak ours into other tests.
        self.enterContext(patch.object(plan_store, "_cache", None))
        self.enterContext(patch.object(last_run, "_cache", None))
        self.enterContext(patch.object(Forecast, "get_weather_forecast", self._fake_weather))

    @staticmethod
    async def _fake_weather(fcst, method="open-meteo", csv_path=None, use_legacy_pvlib=False):
        """A clear-sky day on the forecast grid, no network."""
        fcst.weather_forecast_method = method
        index = fcst.forecast_dates
        local = index.tz_convert(fcst.time_zone) if index.tz is not None else index
        hours = local.hour + local.minute / 60
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

    async def params(self, plants=2, adjusted=False):
        params = copy.deepcopy(await cl_tests.TestCommandLineAsyncUtils.get_test_params(True))
        params["optim_conf"]["weather_forecast_method"] = "open-meteo"
        params["optim_conf"]["set_use_adjusted_pv"] = adjusted
        if plants == 2:
            make_two_plant_pvlib_conf(params["plant_conf"])
        return params

    async def build(self, action, runtime, params=None, plants=2, adjusted=False):
        params = params if params is not None else await self.params(plants, adjusted)
        params["passed_data"] = runtime
        return await set_input_data_dict(
            self.emhass_conf,
            COSTFUN,
            orjson.dumps(params).decode("utf-8"),
            orjson.dumps(runtime).decode("utf-8"),
            action,
            logger,
            get_data_from_file=True,
        )

    @staticmethod
    def dayahead_runtime(**extra):
        return {
            "load_power_forecast": [300.0 + 10 * (i % 6) for i in range(48)],
            "load_cost_forecast": [0.2] * 48,
            "prod_price_forecast": [0.05] * 48,
            **extra,
        }

    @staticmethod
    def mpc_runtime(horizon, **extra):
        return {
            "prediction_horizon": horizon,
            "soc_init": 0.5,
            "soc_final": 0.5,
            "load_power_forecast": [300.0 + 10 * (i % 6) for i in range(horizon)],
            "load_cost_forecast": [0.2] * horizon,
            "prod_price_forecast": [0.05] * horizon,
            **extra,
        }

    def assert_has_plant_columns(self, frame):
        self.assertEqual(_plant_columns(frame), PLANT_COLUMNS, "the plant columns are missing")

    def assert_plants_add_up(self, opt_res):
        self.assert_has_plant_columns(opt_res)
        plants = opt_res[PLANT_COLUMNS]
        self.assertFalse(plants.isnull().any().any())
        self.assertTrue((plants >= 0).all().all())
        np.testing.assert_allclose(plants.sum(axis=1), opt_res["P_PV"], rtol=1e-9, atol=1e-6)
        self.assertGreater(opt_res["P_PV"].max(), 0, "the test needs a day with sun")
        # They really split the total: two differently oriented plants are not a copy.
        self.assertFalse(np.allclose(plants["P_PV_plant0"], plants["P_PV_plant1"]))


class TestDayaheadPlants(PerPlantResultsBase):
    @fixture_clock()
    async def test_two_plants_give_columns_that_add_up_to_p_pv(self):
        idd = await self.build("dayahead-optim", self.dayahead_runtime())
        opt_res = await dayahead_forecast_optim(idd, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assert_plants_add_up(opt_res)
        # Right after P_PV, as the other main power columns follow it.
        columns = list(opt_res.columns)
        self.assertEqual(
            columns[columns.index("P_PV") + 1 : columns.index("P_PV") + 3], PLANT_COLUMNS
        )
        self.assertEqual(len(opt_res), 48)
        # The frame the run carried is the one that was written to the results.
        carried = idd.get("p_pv_plants")
        self.assertIsInstance(carried, pd.DataFrame)
        np.testing.assert_allclose(carried.to_numpy(), opt_res[PLANT_COLUMNS].to_numpy())

    @fixture_clock()
    async def test_the_plants_do_not_change_anything_else(self):
        """Every other column is what the same run gives without the plant frame."""
        idd = await self.build("dayahead-optim", self.dayahead_runtime())
        with_plants = await dayahead_forecast_optim(idd, logger, debug=True)
        self.assert_has_plant_columns(with_plants)
        idd["p_pv_plants"] = None
        without = await dayahead_forecast_optim(idd, logger, debug=True)
        self.assertEqual(_plant_columns(without), [])
        assert_frame_equal(
            with_plants.drop(columns=PLANT_COLUMNS),
            without,
            check_exact=False,
            rtol=1e-9,
            atol=1e-6,
        )
        self.assertEqual(
            list(with_plants.drop(columns=PLANT_COLUMNS).columns), list(without.columns)
        )

    @fixture_clock()
    async def test_csv_and_plan_carry_the_plant_columns(self):
        idd = await self.build("dayahead-optim", self.dayahead_runtime())
        opt_res = await dayahead_forecast_optim(idd, logger)  # not debug: writes CSV and plan
        self.assert_has_plant_columns(opt_res)
        csv = pd.read_csv(self.data_path / "opt_res_latest.csv", index_col="timestamp")
        self.assertEqual(_plant_columns(csv), PLANT_COLUMNS)
        np.testing.assert_allclose(
            csv[PLANT_COLUMNS].sum(axis=1), csv["P_PV"], rtol=1e-9, atol=1e-6
        )
        snapshot = plan_store.read(self.data_path)
        self.assertIsNotNone(snapshot, "no plan was recorded")
        self.assertEqual(snapshot["emhass_schema_version"], EMHASS_SCHEMA_VERSION)
        first = snapshot["plan"][0]
        for col in PLANT_COLUMNS:
            self.assertIn(col, first)
        # serialize() on the result frame gives the same keys (what /api/v1/plan serves).
        serialized = plan_store.serialize(opt_res)
        self.assertTrue(set(PLANT_COLUMNS) <= set(serialized[0]))
        self.assertEqual(len(serialized), 48)

    @fixture_clock()
    async def test_publish_data_is_unchanged(self):
        """No new HA entity, and the published frame has no plant column."""
        idd = await self.build("dayahead-optim", self.dayahead_runtime())
        opt_res = await dayahead_forecast_optim(idd, logger, debug=True)
        self.assert_has_plant_columns(opt_res)
        publish_idd = await self.build("publish-data", {}, params=await self.params())
        posted = []

        async def record_post(data, idx, entity_id, *args, **kwargs):
            posted.append(entity_id)

        publish_idd["rh"].post_data = AsyncMock(side_effect=record_post)
        published = await publish_data(publish_idd, logger, opt_res_latest=opt_res)
        self.assertEqual(_plant_columns(published), [])
        self.assertTrue(posted, "publish_data posted nothing")
        self.assertFalse([entity for entity in posted if "plant" in str(entity).lower()])
        # The same published frame as for the result without the plant columns.
        expected = await publish_data(
            publish_idd, logger, opt_res_latest=opt_res.drop(columns=PLANT_COLUMNS)
        )
        assert_frame_equal(published, expected)

    @fixture_clock()
    async def test_adjustment_on_keeps_the_plants_adding_up(self):
        scaled = AsyncMock(side_effect=lambda log, fcst, p_pv, *a, **kw: p_pv * 0.6)
        with patch("emhass.command_line.adjust_pv_forecast", scaled):
            idd = await self.build("dayahead-optim", self.dayahead_runtime(), adjusted=True)
        self.assertGreaterEqual(scaled.await_count, 1, "the adjustment was not applied")
        opt_res = await dayahead_forecast_optim(idd, logger, debug=True)
        self.assert_plants_add_up(opt_res)
        # The adjusted total, with each plant keeping its share of the raw forecast.
        raw = idd["fcst"].pv_plant_raw.clip(lower=0)
        share0 = (raw.iloc[:, 0] / raw.sum(axis=1)).fillna(0).to_numpy()
        producing = opt_res["P_PV"].to_numpy() > 0
        np.testing.assert_allclose(
            opt_res["P_PV_plant0"].to_numpy()[producing],
            (opt_res["P_PV"].to_numpy() * share0)[producing],
            rtol=1e-9,
            atol=1e-6,
        )

    @fixture_clock()
    async def test_adjustment_failure_returns_raw_and_plants_still_add_up(self):
        raw_back = AsyncMock(side_effect=lambda log, fcst, p_pv, *a, **kw: p_pv)
        with patch("emhass.command_line.adjust_pv_forecast", raw_back):
            idd = await self.build("dayahead-optim", self.dayahead_runtime(), adjusted=True)
        self.assertGreaterEqual(raw_back.await_count, 1, "the adjustment was not attempted")
        opt_res = await dayahead_forecast_optim(idd, logger, debug=True)
        self.assert_plants_add_up(opt_res)

    @fixture_clock()
    async def test_a_single_plant_keeps_the_column_set(self):
        """With one plant the result has exactly the usual columns."""
        idd = await self.build("dayahead-optim", self.dayahead_runtime(), plants=1)
        self.assertIsNone(idd.get("p_pv_plants"))
        opt_res = await dayahead_forecast_optim(idd, logger, debug=True)
        self.assertEqual(_plant_columns(opt_res), [])
        self.assertEqual(list(opt_res.columns), SINGLE_PLANT_DAYAHEAD_COLUMNS)
        # And it is what a two-plant run gives apart from the plant columns.
        idd2 = await self.build("dayahead-optim", self.dayahead_runtime(), plants=2)
        opt_res2 = await dayahead_forecast_optim(idd2, logger, debug=True)
        self.assert_has_plant_columns(opt_res2)
        self.assertEqual(list(opt_res2.drop(columns=PLANT_COLUMNS).columns), list(opt_res.columns))

    @fixture_clock()
    async def test_set_use_pv_off_gives_no_plants(self):
        params = await self.params()
        params["optim_conf"]["set_use_pv"] = False
        idd = await self.build("dayahead-optim", self.dayahead_runtime(), params=params)
        self.assertIsNone(idd.get("p_pv_plants"))
        opt_res = await dayahead_forecast_optim(idd, logger, debug=True)
        self.assertEqual(_plant_columns(opt_res), [])
        self.assertTrue((opt_res["P_PV"] == 0).all())


class TestNaiveMpcPlants(PerPlantResultsBase):
    @fixture_clock()
    async def test_columns_line_up_with_p_pv_when_the_horizon_is_shorter_than_the_forecast(self):
        horizon = 10
        idd = await self.build("naive-mpc-optim", self.mpc_runtime(20))
        forecast_steps = len(idd["p_pv_forecast"])
        # The run asks for fewer steps than the forecast holds: the optimizer slices to it.
        # A real run does not get here (Forecast.__init__ already cuts the forecast to
        # prediction_horizon); the narrowing is deliberate, to pin that the plant columns
        # are sliced by position exactly as P_PV is if the two lengths ever differ.
        idd["params"]["passed_data"]["prediction_horizon"] = horizon
        self.assertGreater(forecast_steps, horizon, "the forecast must be longer than the horizon")
        carried = idd.get("p_pv_plants")
        self.assertIsInstance(carried, pd.DataFrame, "the plant frame was not carried")
        self.assertEqual(len(carried), forecast_steps)
        opt_res = await naive_mpc_optim(idd, logger, debug=True)
        self.assertEqual(len(opt_res), horizon)
        self.assert_plants_add_up(opt_res)
        # Row for row the first `horizon` steps of the carried frame, as P_PV is.
        np.testing.assert_allclose(
            opt_res[PLANT_COLUMNS].to_numpy(), carried.iloc[:horizon].to_numpy()
        )
        np.testing.assert_allclose(
            opt_res["P_PV"].to_numpy(), idd["p_pv_forecast"].iloc[:horizon].to_numpy()
        )

    @fixture_clock()
    async def test_the_plants_do_not_change_anything_else(self):
        idd = await self.build("naive-mpc-optim", self.mpc_runtime(10))
        with_plants = await naive_mpc_optim(idd, logger, debug=True)
        self.assert_has_plant_columns(with_plants)
        idd["p_pv_plants"] = None
        without = await naive_mpc_optim(idd, logger, debug=True)
        assert_frame_equal(
            with_plants.drop(columns=PLANT_COLUMNS),
            without,
            check_exact=False,
            rtol=1e-9,
            atol=1e-6,
        )

    @fixture_clock()
    async def test_adjustment_on_and_failed_keep_the_plants_adding_up(self):
        for label, factor in (("scaled", 0.5), ("failed, raw returned", 1.0)):
            with self.subTest(label):
                adjust = AsyncMock(side_effect=lambda log, fcst, p_pv, *a, f=factor, **kw: p_pv * f)
                with patch("emhass.command_line.adjust_pv_forecast", adjust):
                    idd = await self.build("naive-mpc-optim", self.mpc_runtime(10), adjusted=True)
                self.assertGreaterEqual(adjust.await_count, 1, "the adjustment was not applied")
                opt_res = await naive_mpc_optim(idd, logger, debug=True)
                self.assert_plants_add_up(opt_res)

    @fixture_clock()
    async def test_csv_carries_the_plant_columns(self):
        idd = await self.build("naive-mpc-optim", self.mpc_runtime(10))
        opt_res = await naive_mpc_optim(idd, logger)
        self.assert_has_plant_columns(opt_res)
        csv = pd.read_csv(self.data_path / "opt_res_latest.csv", index_col="timestamp")
        self.assertEqual(_plant_columns(csv), PLANT_COLUMNS)
        self.assertEqual(len(csv), 10)


class TestOtherActions(PerPlantResultsBase):
    @fixture_clock()
    async def test_perfect_optim_has_no_plant_columns(self):
        params = await self.params()
        idd = await self.build("perfect-optim", self.dayahead_runtime(), params=params)
        self.assertIsNone(idd.get("p_pv_plants"))
        opt_res = await perfect_forecast_optim(idd, logger, debug=True)
        self.assertIsInstance(opt_res, pd.DataFrame)
        self.assertGreater(len(opt_res), 0)
        self.assertEqual(_plant_columns(opt_res), [])
        self.assertIn("P_PV", opt_res.columns)

    @fixture_clock()
    async def test_actions_without_a_pv_forecast_default_the_key_to_none(self):
        idd = await self.build("publish-data", {})
        self.assertIn("p_pv_plants", idd, "the default key is missing")
        self.assertIsNone(idd["p_pv_plants"])


class TestOptimizerKeyword(PerPlantResultsBase):
    """The optimizer entry points, called directly."""

    @fixture_clock()
    async def test_keyword_none_is_identical_and_the_frame_is_aligned_by_position(self):
        idd = await self.build("dayahead-optim", self.dayahead_runtime())
        carried = idd.get("p_pv_plants")
        self.assertIsInstance(carried, pd.DataFrame, "the plant frame was not carried")
        opt = idd["opt"]
        df = idd["df_input_data_dayahead"]
        df = df.assign(unit_load_cost=0.2, unit_prod_price=0.05)
        # A frame on another index must still line up by position, as the P_PV array does.
        shifted = carried.copy()
        shifted.index = pd.date_range("2000-01-01", periods=len(shifted), freq="30min")
        res = opt.perform_dayahead_forecast_optim(
            df, idd["p_pv_forecast"], idd["p_load_forecast"], p_pv_plants=shifted
        )
        np.testing.assert_allclose(res[PLANT_COLUMNS].to_numpy(), carried.to_numpy())
        plain = opt.perform_dayahead_forecast_optim(
            df, idd["p_pv_forecast"], idd["p_load_forecast"], p_pv_plants=None
        )
        self.assertEqual(_plant_columns(plain), [])
        self.assertEqual(list(res.drop(columns=PLANT_COLUMNS).columns), list(plain.columns))


if __name__ == "__main__":
    unittest.main()
