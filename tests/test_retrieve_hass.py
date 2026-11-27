import asyncio
import copy
import datetime
import inspect
import itertools
import logging
import os
import pathlib
import pickle
import re
import select
import socket
import struct
import tempfile
import threading
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import aiofiles
import aiohttp
import numpy as np
import orjson
import pandas as pd
import time_machine

from emhass import retrieve_hass, utils
from emhass.command_line import _retrieve_from_hass
from emhass.retrieve_hass import RetrieveHass
from emhass.utils import get_days_list, get_logger, get_yaml_parse
from tests.aioresponses_compat import aioresponses

# The root folder
root = pathlib.Path(utils.get_root(__file__, num_parent=2))
# Build emhass_conf paths
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["options_path"] = root / "options.json"
emhass_conf["secrets_path"] = root / "secrets_emhass(example).yaml"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

# create logger
logger, ch = get_logger(__name__, emhass_conf, save_to_file=False)


class TestRetrieveHass(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # build_secrets reads EMHASS_URL, SUPERVISOR_TOKEN and TIME_ZONE from the environment; a
        # developer's own values must not reach these tests. The environment is restored after
        # each test.
        environ = self.enterContext(patch.dict(os.environ))
        for name in ("EMHASS_URL", "SUPERVISOR_TOKEN", "TIME_ZONE"):
            environ.pop(name, None)
        emhass_conf["data_path"] = root / "data/"

        self.get_data_from_file = True
        save_data_to_file = False
        model_type = "test_df_final"  # Options: "test_df_final" or "long_train_data"

        # Build params with default secrets (no config)
        if emhass_conf["defaults_path"].exists():
            if self.get_data_from_file:
                _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
                params = await utils.build_params(emhass_conf, secrets, {}, logger)
                retrieve_hass_conf, _, _ = get_yaml_parse(params, logger)
            else:
                emhass_conf["secrets_path"] = root / "secrets_emhass.yaml"
                config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
                _, secrets = await utils.build_secrets(
                    emhass_conf,
                    logger,
                    secrets_path=emhass_conf["secrets_path"],
                    no_response=True,
                )
                params = await utils.build_params(emhass_conf, secrets, config, logger)
                retrieve_hass_conf, _, _ = get_yaml_parse(params, logger)
                params = None
        else:
            raise Exception(
                "config_defaults. does not exist in path: " + str(emhass_conf["defaults_path"])
            )

        # Force config params for testing
        retrieve_hass_conf["optimization_time_step"] = pd.to_timedelta(30, "minutes")
        retrieve_hass_conf["sensor_power_photovoltaics"] = "sensor.power_photovoltaics"
        retrieve_hass_conf["sensor_power_photovoltaics_forecast"] = "sensor.p_pv_forecast"
        retrieve_hass_conf["sensor_power_load_no_var_loads"] = "sensor.power_load_no_var_loads"
        retrieve_hass_conf["sensor_replace_zero"] = [
            "sensor.power_photovoltaics",
            "sensor.p_pv_forecast",
        ]
        retrieve_hass_conf["sensor_linear_interp"] = [
            "sensor.power_photovoltaics",
            "sensor.p_pv_forecast",
            "sensor.power_load_no_var_loads",
        ]
        retrieve_hass_conf["set_zero_min"] = True
        retrieve_hass_conf["load_negative"] = True

        self.retrieve_hass_conf = retrieve_hass_conf
        self.rh = RetrieveHass(
            self.retrieve_hass_conf["hass_url"],
            self.retrieve_hass_conf["long_lived_token"],
            self.retrieve_hass_conf["optimization_time_step"],
            self.retrieve_hass_conf["time_zone"],
            params,
            emhass_conf,
            logger,
            get_data_from_file=self.get_data_from_file,
        )
        # Obtain sensor values from saved file
        if self.get_data_from_file:
            async with aiofiles.open(
                emhass_conf["data_path"] / str(model_type + ".pkl"), "rb"
            ) as f:
                content = await f.read()
                self.rh.df_final, self.days_list, self.var_list, self.rh.ha_config = pickle.loads(
                    content
                )
                self.rh.var_list = self.var_list
        # Else obtain sensor values from HA
        else:
            if model_type == "long_train_data":
                days_to_retrieve = 365
            else:
                days_to_retrieve = self.retrieve_hass_conf["historic_days_to_retrieve"]
            self.days_list = get_days_list(days_to_retrieve)
            self.var_list = [
                self.retrieve_hass_conf["sensor_power_load_no_var_loads"],
                self.retrieve_hass_conf["sensor_power_photovoltaics"],
                self.retrieve_hass_conf["sensor_power_photovoltaics_forecast"],
            ]
            await self.rh.get_data(
                self.days_list,
                self.var_list,
                minimal_response=False,
                significant_changes_only=False,
            )
            # Mocking retrieve of ha_config using: self.rh.get_ha_config()
            self.rh.ha_config = {
                "country": "FR",
                "currency": "EUR",
                "elevation": 4807,
                "latitude": 48.83,
                "longitude": 6.86,
                "time_zone": "Europe/Paris",
                "unit_system": {
                    "length": "km",
                    "accumulated_precipitation": "mm",
                    "area": "m²",
                    "mass": "g",
                    "pressure": "Pa",
                    "temperature": "°C",
                    "volume": "L",
                    "wind_speed": "m/s",
                },
            }
            # Check to save updated data to file
            if save_data_to_file:
                async with aiofiles.open(
                    emhass_conf["data_path"] / str(model_type + ".pkl"), "wb"
                ) as outp:
                    pickle.dump(
                        (
                            self.rh.df_final,
                            self.days_list,
                            self.var_list,
                            self.rh.ha_config,
                        ),
                        outp,
                        pickle.HIGHEST_PROTOCOL,
                    )
        self.df_raw = self.rh.df_final.copy()

    async def asyncTearDown(self):
        """Clean up after each test - close any open HTTP sessions."""
        if hasattr(self, "rh") and self.rh is not None:
            await self.rh.close()

    # Check yaml parse in setUp worked
    def test_get_yaml_parse(self):
        self.assertIsInstance(self.retrieve_hass_conf, dict)
        self.assertIn("hass_url", self.retrieve_hass_conf.keys())
        if self.get_data_from_file:
            self.assertEqual(
                self.retrieve_hass_conf["hass_url"], "https://homeassistant.invalid:8123/"
            )
            self.assertEqual(str(self.retrieve_hass_conf["time_zone"]), "Europe/Paris")

    # Check yaml parse worked
    async def test_yaml_parse_web_server(self):
        params = {}
        if emhass_conf["defaults_path"].exists():
            async with aiofiles.open(emhass_conf["defaults_path"]) as file:
                data = await file.read()
                defaults = orjson.loads(data)
                params.update(await utils.build_params(emhass_conf, {}, defaults, logger))
        _, optim_conf, _ = get_yaml_parse(params, logger)
        # Just check forecast methods
        self.assertIsNot(optim_conf.get("weather_forecast_method"), None)
        self.assertIsNot(optim_conf.get("load_forecast_method"), None)
        self.assertIsNot(optim_conf.get("load_cost_forecast_method"), None)
        self.assertIsNot(optim_conf.get("production_price_forecast_method"), None)

    # Assume get_data to HA fails
    async def test_get_data_failed(self):
        days_list = get_days_list(1)
        var_list = [self.retrieve_hass_conf["sensor_power_load_no_var_loads"]]
        test_url = "http://mock.invalid/history"
        with aioresponses() as mocked, self.assertLogs(logger, level="ERROR") as logs:
            mocked.get(test_url, status=401, repeat=True)
            response = await self.rh.get_data(days_list, var_list, test_url=test_url)
        self.assertFalse(response)
        # Only the HTTP 401 branch logs this; a connection error logs "Network error" instead
        self.assertTrue(any("TOKEN/KEY is invalid or missing" in line for line in logs.output))
        self.assertIn(("GET", test_url), [(m, str(u)) for m, u in mocked.requests])

    # Test with html mock response
    async def test_get_data_mock(self):
        with aioresponses() as mocked:
            test_data_path = emhass_conf["data_path"] / "test_response_get_data_get_method.json"

            async with aiofiles.open(test_data_path, "rb") as f:
                data = orjson.loads(await f.read())
            days_list = get_days_list(1)
            var_list = [self.retrieve_hass_conf["sensor_power_load_no_var_loads"]]
            # with aioresponses() as mocked:
            get_url = self.retrieve_hass_conf["hass_url"]
            mocked.get(get_url, payload=data, repeat=True)
            await self.rh.get_data(
                days_list,
                var_list,
                minimal_response=False,
                significant_changes_only=False,
                test_url=self.retrieve_hass_conf["hass_url"],
            )
            self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
            self.assertIsInstance(self.rh.df_final.index, pd.core.indexes.datetimes.DatetimeIndex)
            self.assertIsInstance(
                self.rh.df_final.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype
            )
            self.assertEqual(len(self.rh.df_final.columns), len(var_list))
            self.assertEqual(
                self.rh.df_final.index.freq,
                self.retrieve_hass_conf["optimization_time_step"],
            )
            self.assertEqual(self.rh.df_final.index.tz, datetime.UTC)

    # Check the dataframe was formatted correctly
    def test_prepare_data(self):
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertIsInstance(self.rh.df_final.index, pd.core.indexes.datetimes.DatetimeIndex)
        self.assertIsInstance(self.rh.df_final.index.dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        self.assertEqual(len(self.rh.df_final.columns), len(self.var_list))
        self.assertEqual(self.rh.df_final.index.isin(self.days_list).sum(), len(self.days_list))
        self.assertEqual(
            self.rh.df_final.index.freq,
            self.retrieve_hass_conf["optimization_time_step"],
        )
        self.assertEqual(self.rh.df_final.index.tz, datetime.UTC)
        self.rh.prepare_data(
            self.retrieve_hass_conf["sensor_power_load_no_var_loads"],
            load_negative=self.retrieve_hass_conf["load_negative"],
            set_zero_min=self.retrieve_hass_conf["set_zero_min"],
            var_replace_zero=self.retrieve_hass_conf["sensor_replace_zero"],
            var_interp=self.retrieve_hass_conf["sensor_linear_interp"],
        )
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertEqual(
            self.rh.df_final.index.isin(self.days_list).sum(),
            self.df_raw.index.isin(self.days_list).sum(),
        )
        self.assertEqual(len(self.rh.df_final.columns), len(self.df_raw.columns))
        self.assertEqual(
            self.rh.df_final.index.freq,
            self.retrieve_hass_conf["optimization_time_step"],
        )
        self.assertEqual(self.rh.df_final.index.tz, self.retrieve_hass_conf["time_zone"])

    # Test negative load
    def test_prepare_data_negative_load(self):
        self.rh.df_final[
            self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        ] = -self.rh.df_final[self.retrieve_hass_conf["sensor_power_load_no_var_loads"]]
        self.rh.prepare_data(
            self.retrieve_hass_conf["sensor_power_load_no_var_loads"],
            load_negative=True,
            set_zero_min=self.retrieve_hass_conf["set_zero_min"],
            var_replace_zero=self.retrieve_hass_conf["sensor_replace_zero"],
            var_interp=None,
        )
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertEqual(
            self.rh.df_final.index.isin(self.days_list).sum(),
            self.df_raw.index.isin(self.days_list).sum(),
        )
        self.assertEqual(len(self.rh.df_final.columns), len(self.df_raw.columns))
        self.assertEqual(
            self.rh.df_final.index.freq,
            self.retrieve_hass_conf["optimization_time_step"],
        )
        self.assertEqual(self.rh.df_final.index.tz, self.retrieve_hass_conf["time_zone"])

    # Tests that the prepare_data method does convert missing PV values to zero
    # and also ignores any missing sensor columns.
    def test_prepare_data_missing_pv(self):
        load_sensor = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        actual_pv_sensor = self.retrieve_hass_conf["sensor_power_photovoltaics"]
        forecast_pv_sensor = self.retrieve_hass_conf["sensor_power_photovoltaics_forecast"]
        var_replace_zero = [actual_pv_sensor, forecast_pv_sensor, "sensor.missing1"]
        var_interp = [actual_pv_sensor, load_sensor, "sensor.missing2"]
        # Replace actual and forecast PV zero values with NaN's (to test they get replaced back)
        self.rh.df_final[actual_pv_sensor] = self.rh.df_final[actual_pv_sensor].replace(0, np.nan)
        self.rh.df_final[forecast_pv_sensor] = self.rh.df_final[forecast_pv_sensor].replace(
            0, np.nan
        )
        # Verify a non-zero number of missing values in the actual and forecast PV columns before prepare_data
        self.assertGreater(self.rh.df_final[actual_pv_sensor].isna().sum(), 0)
        self.assertGreater(self.rh.df_final[forecast_pv_sensor].isna().sum(), 0)
        self.rh.prepare_data(
            load_sensor,
            load_negative=False,
            set_zero_min=True,
            var_replace_zero=var_replace_zero,
            var_interp=var_interp,
        )
        self.assertIsInstance(self.rh.df_final, type(pd.DataFrame()))
        self.assertEqual(
            self.rh.df_final.index.isin(self.days_list).sum(),
            self.df_raw.index.isin(self.days_list).sum(),
        )
        # Check the before and after actual and forecast PV columns have the same number of values
        self.assertEqual(
            len(self.df_raw[actual_pv_sensor]), len(self.rh.df_final[actual_pv_sensor])
        )
        self.assertEqual(
            len(self.df_raw[forecast_pv_sensor]),
            len(self.rh.df_final[forecast_pv_sensor]),
        )
        # Verify no missing values in the actual and forecast PV columns after prepare_data
        self.assertEqual(self.rh.df_final[actual_pv_sensor].isna().sum(), 0)
        self.assertEqual(self.rh.df_final[forecast_pv_sensor].isna().sum(), 0)

    # Issue #1084: a var_replace_zero/var_interp entry that doesn't match any
    # retrieved sensor is silently dropped. If that silence coincides with
    # NaNs surviving cleaning elsewhere, it is a strong signal of a
    # sensor-name typo in the config, so prepare_data must warn once, naming
    # both the dropped entry and the still-affected column.
    def test_prepare_data_dropped_sensor_warns_when_nan_remains(self):
        load_sensor = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        actual_pv_sensor = self.retrieve_hass_conf["sensor_power_photovoltaics"]
        forecast_pv_sensor = self.retrieve_hass_conf["sensor_power_photovoltaics_forecast"]
        # Zero out a value in a column NOT covered by either repair list below,
        # so set_zero_min turns it into an unrepaired NaN.
        self.rh.df_final.loc[self.rh.df_final.index[0], forecast_pv_sensor] = 0.0
        with self.assertLogs(logger, level="WARNING") as cm:
            self.rh.prepare_data(
                load_sensor,
                load_negative=False,
                set_zero_min=True,
                var_replace_zero=[actual_pv_sensor],
                var_interp=[actual_pv_sensor, load_sensor, "sensor.missing_typo"],
            )
        self.assertEqual(
            sum("sensor.missing_typo" in line for line in cm.output),
            1,
            f"expected exactly one warning naming the dropped sensor, got: {cm.output}",
        )
        self.assertTrue(
            any(forecast_pv_sensor in line for line in cm.output),
            f"expected the warning to name the still-NaN column, got: {cm.output}",
        )

    # Mirror of the test above for the sensor_replace_zero half: a dropped
    # replace-zero entry plus a surviving NaN must fire the warning naming
    # the parameter and the dropped entry.
    def test_prepare_data_dropped_replace_zero_warns_when_nan_remains(self):
        load_sensor = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        actual_pv_sensor = self.retrieve_hass_conf["sensor_power_photovoltaics"]
        forecast_pv_sensor = self.retrieve_hass_conf["sensor_power_photovoltaics_forecast"]
        self.rh.df_final.loc[self.rh.df_final.index[0], forecast_pv_sensor] = 0.0
        with self.assertLogs(logger, level="WARNING") as cm:
            self.rh.prepare_data(
                load_sensor,
                load_negative=False,
                set_zero_min=True,
                var_replace_zero=[actual_pv_sensor, "sensor.missing_typo"],
                var_interp=[actual_pv_sensor, load_sensor],
            )
        warning_lines = [line for line in cm.output if "sensor.missing_typo" in line]
        self.assertEqual(len(warning_lines), 1, f"expected one dropped-sensor warning: {cm.output}")
        self.assertIn("sensor_replace_zero", warning_lines[0])
        self.assertIn(forecast_pv_sensor, warning_lines[0])

    # Correctly-configured lists (nothing dropped) must never trigger the
    # dropped-sensor warning, even when an unrelated NaN legitimately survives.
    def test_prepare_data_no_warning_when_no_sensor_dropped(self):
        load_sensor = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        actual_pv_sensor = self.retrieve_hass_conf["sensor_power_photovoltaics"]
        forecast_pv_sensor = self.retrieve_hass_conf["sensor_power_photovoltaics_forecast"]
        # Same unrepaired-NaN setup as above, but every configured sensor name
        # actually matches a retrieved column this time: nothing is dropped.
        self.rh.df_final.loc[self.rh.df_final.index[0], forecast_pv_sensor] = 0.0
        with self.assertNoLogs(logger, level="WARNING"):
            self.rh.prepare_data(
                load_sensor,
                load_negative=False,
                set_zero_min=True,
                var_replace_zero=[actual_pv_sensor],
                var_interp=[actual_pv_sensor, load_sensor],
            )

    # A dropped entry with nothing left to warn about (every column is fully
    # repaired) must not trigger a warning either.
    def test_prepare_data_no_warning_when_dropped_but_no_nan_remains(self):
        load_sensor = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        actual_pv_sensor = self.retrieve_hass_conf["sensor_power_photovoltaics"]
        forecast_pv_sensor = self.retrieve_hass_conf["sensor_power_photovoltaics_forecast"]
        with self.assertNoLogs(logger, level="WARNING"):
            self.rh.prepare_data(
                load_sensor,
                load_negative=False,
                set_zero_min=True,
                var_replace_zero=[actual_pv_sensor, forecast_pv_sensor, "sensor.missing_typo"],
                var_interp=[
                    actual_pv_sensor,
                    forecast_pv_sensor,
                    load_sensor,
                    "sensor.missing_typo",
                ],
            )

    # The exact #1084 trap: the load sensor is missing from the repair lists
    # (stale entry dropped), its zero readings become unrepaired NaN, and the
    # warning must name it under its CONFIGURED name, not the internal
    # var_load + "_positive" rename.
    def test_prepare_data_warning_names_configured_load_name_not_rename(self):
        load_sensor = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        actual_pv_sensor = self.retrieve_hass_conf["sensor_power_photovoltaics"]
        # A zero load reading becomes NaN via set_zero_min; the load sensor is
        # absent from var_interp, so nothing repairs it.
        self.rh.df_final.loc[self.rh.df_final.index[0], load_sensor] = 0.0
        with self.assertLogs(logger, level="WARNING") as cm:
            self.rh.prepare_data(
                load_sensor,
                load_negative=False,
                set_zero_min=True,
                var_replace_zero=[actual_pv_sensor],
                var_interp=[actual_pv_sensor, "sensor.missing_typo"],
            )
        warning_lines = [line for line in cm.output if "sensor.missing_typo" in line]
        self.assertEqual(len(warning_lines), 1, f"expected one dropped-sensor warning: {cm.output}")
        self.assertIn(load_sensor, warning_lines[0])
        self.assertNotIn(load_sensor + "_positive", warning_lines[0])

    # The battery-identification path deliberately retrieves a subset of the
    # configured sensors (no PV) and keeps by-design NaN in its protected
    # battery columns, so structurally dropped entries plus protected-column
    # NaN must never trigger the dropped-sensor warning.
    def test_prepare_data_no_warning_for_protected_nan_only(self):
        load_sensor = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        battery_sensor = "sensor.power_battery"
        idx = self.rh.df_final.index[:8]
        self.rh.df_final = pd.DataFrame(
            {
                load_sensor: [100.0] * 8,
                battery_sensor: [50.0, -50.0, np.nan, 30.0, -30.0, 0.0, 20.0, 10.0],
            },
            index=idx,
        )
        self.rh.var_list = [load_sensor, battery_sensor]
        with self.assertNoLogs(logger, level="WARNING"):
            self.rh.prepare_data(
                load_sensor,
                load_negative=False,
                set_zero_min=True,
                var_replace_zero=["sensor.power_photovoltaics_not_retrieved"],
                var_interp=["sensor.power_photovoltaics_not_retrieved", load_sensor],
                protected_columns=[battery_sensor],
            )
        # The by-design NaN in the protected column must survive untouched.
        self.assertTrue(self.rh.df_final[battery_sensor].isna().any())

    # protected_columns holds CONFIGURED sensor names, but the load column is
    # renamed to var_load + "_positive" before the dropped-sensor NaN check
    # runs, so a protected load must keep its exclusion across the rename. A
    # zero load reading turned NaN by set_zero_min plus a structurally dropped
    # entry must not fire the warning for the protected load.
    def test_prepare_data_protected_load_keeps_exclusion_across_rename(self):
        load_sensor = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        idx = self.rh.df_final.index[:4]
        self.rh.df_final = pd.DataFrame(
            {load_sensor: [100.0, 0.0, 200.0, 150.0]},
            index=idx,
        )
        self.rh.var_list = [load_sensor]
        with self.assertNoLogs(logger, level="WARNING"):
            self.rh.prepare_data(
                load_sensor,
                load_negative=False,
                set_zero_min=True,
                var_replace_zero=["sensor.power_photovoltaics_not_retrieved"],
                var_interp=["sensor.power_photovoltaics_not_retrieved"],
                protected_columns=[load_sensor],
            )
        # The unrepaired NaN really is present under the renamed column, so
        # the no-warning assertion above exercised the exclusion, not an
        # accidentally clean frame.
        self.assertTrue(self.rh.df_final[load_sensor + "_positive"].isna().any())

    # The single-sensor ML paths (model fit/tune/predict, forecast calibration)
    # retrieve only their target sensor while forwarding the full configured
    # lists, and mark themselves with skip_renaming=True. Dropped entries are
    # structural there and must never trigger the dropped-sensor warning, even
    # when the target column keeps an unrepaired NaN (e.g. a legitimate zero
    # reading turned NaN by set_zero_min).
    def test_prepare_data_no_warning_when_skip_renaming(self):
        var_model = "sensor.my_custom_model_input"
        idx = self.rh.df_final.index[:4]
        self.rh.df_final = pd.DataFrame(
            {var_model: [100.0, 0.0, 200.0, 150.0]},
            index=idx,
        )
        self.rh.var_list = [var_model]
        with self.assertNoLogs(logger, level="WARNING"):
            self.rh.prepare_data(
                var_model,
                load_negative=False,
                set_zero_min=True,
                var_replace_zero=["sensor.power_photovoltaics"],
                var_interp=["sensor.power_photovoltaics", "sensor.power_load_no_var_loads"],
                skip_renaming=True,
            )

    # Battery identification needs the signed battery power and a possible
    # measured 0% SoC to survive prepare_data's set_zero_min treatment (#1041).
    # Base-safe: the protected_columns kwarg is only passed when the running
    # source accepts it, so on a source without the fix this test fails on the
    # behavioural assertions below, not on a TypeError.
    def test_prepare_data_protected_columns(self):
        import inspect

        load_sensor = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        power_col = "sensor.power_battery_test"
        soc_col = "sensor.battery_soc_test"
        probe_col = "sensor.unprotected_probe"
        n = len(self.rh.df_final)
        # Signed battery power: both flow directions present
        power = np.full(n, 300.0)
        power[: n // 2] = -400.0
        self.rh.df_final[power_col] = power
        # SoC sweep including a legitimately measured 0% sample
        soc = np.linspace(0.0, 100.0, n)
        soc[0] = 0.0
        self.rh.df_final[soc_col] = soc
        # Counterfactual: an unprotected negative column must still be clipped
        self.rh.df_final[probe_col] = np.full(n, -1.0)
        self.rh.var_list = list(self.var_list) + [power_col, soc_col, probe_col]
        neg_before = int((self.rh.df_final[power_col] < 0).sum())
        self.assertGreater(neg_before, 0)
        kwargs = {}
        if "protected_columns" in inspect.signature(self.rh.prepare_data).parameters:
            kwargs["protected_columns"] = [power_col, soc_col]
        self.rh.prepare_data(
            load_sensor,
            load_negative=self.retrieve_hass_conf["load_negative"],
            set_zero_min=True,
            var_replace_zero=self.retrieve_hass_conf["sensor_replace_zero"],
            var_interp=self.retrieve_hass_conf["sensor_linear_interp"],
            **kwargs,
        )
        # Protected columns: discharge samples and the 0% SoC sample survive
        self.assertEqual(int((self.rh.df_final[power_col] < 0).sum()), neg_before)
        self.assertEqual(self.rh.df_final[power_col].isna().sum(), 0)
        self.assertEqual(int((self.rh.df_final[soc_col] == 0.0).sum()), 1)
        self.assertEqual(self.rh.df_final[soc_col].isna().sum(), 0)
        # Counterfactual: the unprotected probe column was clipped then NaN'd
        self.assertEqual(int((self.rh.df_final[probe_col] < 0).sum()), 0)
        self.assertTrue(self.rh.df_final[probe_col].isna().all())
        # Load handling is unchanged: renamed column present, no negatives
        self.assertIn(load_sensor + "_positive", self.rh.df_final.columns)
        self.assertFalse((self.rh.df_final[load_sensor + "_positive"] < 0).any())

    # protected_columns=None, an omitted kwarg, and a list naming no column in
    # the frame must all reproduce today's clipping behaviour exactly.
    def test_prepare_data_protected_columns_default_noop(self):
        import inspect

        if "protected_columns" not in inspect.signature(self.rh.prepare_data).parameters:
            self.skipTest("running source has no protected_columns support")
        load_sensor = self.retrieve_hass_conf["sensor_power_load_no_var_loads"]
        signed_col = "sensor.signed_probe"
        df_raw = self.rh.df_final.copy()
        df_raw[signed_col] = np.linspace(-100.0, 100.0, len(df_raw))
        var_list_raw = list(self.var_list) + [signed_col]
        runs = {}
        for label, protected in (
            ("omitted", "OMIT"),
            ("none", None),
            ("absent", ["sensor.not_in_frame"]),
        ):
            self.rh.df_final = df_raw.copy()
            self.rh.var_list = list(var_list_raw)
            kwargs = {} if protected == "OMIT" else {"protected_columns": protected}
            self.rh.prepare_data(
                load_sensor,
                load_negative=self.retrieve_hass_conf["load_negative"],
                set_zero_min=True,
                var_replace_zero=self.retrieve_hass_conf["sensor_replace_zero"],
                var_interp=self.retrieve_hass_conf["sensor_linear_interp"],
                **kwargs,
            )
            runs[label] = self.rh.df_final.copy()
        pd.testing.assert_frame_equal(runs["omitted"], runs["none"])
        pd.testing.assert_frame_equal(runs["omitted"], runs["absent"])
        # And the clip really ran: no negatives survive anywhere
        self.assertEqual(int((runs["omitted"][signed_col] < 0).sum()), 0)

    # Proposed new test method for InfluxDB
    @patch("influxdb.InfluxDBClient", autospec=True)
    async def test_get_data_influxdb_mock(self, mock_influx_client_class):
        """
        Test the get_data_influxdb method by mocking the InfluxDB client.
        """
        # Build a correctly structured params dictionary for the test
        params_influx = {
            "retrieve_hass_conf": {
                "use_influxdb": True,
                "influxdb_host": "fake-host",
                "influxdb_port": 8086,
                "influxdb_username": "fake-user",
                "influxdb_password": "fake-pass",  # pragma: allowlist secret
                "influxdb_database": "fake-db",
                "influxdb_measurement": "W",
                # Add other necessary keys from the original conf
                "sensor_power_photovoltaics": self.retrieve_hass_conf["sensor_power_photovoltaics"],
                "sensor_power_load_no_var_loads": self.retrieve_hass_conf[
                    "sensor_power_load_no_var_loads"
                ],
            }
        }

        # Instantiate RetrieveHass with the correctly nested configuration
        rh_influx = RetrieveHass(
            self.retrieve_hass_conf["hass_url"],
            self.retrieve_hass_conf["long_lived_token"],
            self.retrieve_hass_conf["optimization_time_step"],
            self.retrieve_hass_conf["time_zone"],
            params_influx,
            emhass_conf,
            logger,
            get_data_from_file=False,
        )

        # Mock the client instance that will be created inside the method
        mock_client_instance = mock_influx_client_class.return_value

        # Define mock data points to be returned by the client
        mock_pv_data = [
            {"time": "2023-04-01T10:00:00Z", "mean_value": 1500.0},
            {"time": "2023-04-01T10:30:00Z", "mean_value": 1800.0},
        ]
        mock_load_data = [
            {"time": "2023-04-01T10:00:00Z", "mean_value": 500.0},
            {"time": "2023-04-01T10:30:00Z", "mean_value": 450.0},
        ]

        # Define a side_effect function to handle different queries
        def query_side_effect(query):
            mock_result = MagicMock()
            if "SHOW MEASUREMENTS" in query:
                mock_result.get_points.return_value = [{"name": "W"}]
            elif "SHOW TAG VALUES" in query and '"W"' in query:
                mock_result.get_points.return_value = [
                    {"value": "power_photovoltaics"},
                    {"value": "power_load_no_var_loads"},
                ]
            elif "entity_id" in query and "'power_photovoltaics'" in query:
                mock_result.get_points.return_value = mock_pv_data
            elif "entity_id" in query and "'power_load_no_var_loads'" in query:
                mock_result.get_points.return_value = mock_load_data
            else:
                mock_result.get_points.return_value = []
            return mock_result

        # Assign the handler to the mock instance's query method
        mock_client_instance.query.side_effect = query_side_effect

        # Define the inputs for the get_data method
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        var_list = [
            params_influx["retrieve_hass_conf"]["sensor_power_photovoltaics"],
            params_influx["retrieve_hass_conf"]["sensor_power_load_no_var_loads"],
        ]

        # Call the method to be tested
        success = await rh_influx.get_data(days_list, var_list)

        # Verify the outcomes
        self.assertTrue(success)  # Check if the method reports success

        # Verify that the InfluxDB client was initialized correctly
        mock_influx_client_class.assert_called_with(
            host="fake-host",
            port=8086,
            username="fake-user",
            password="fake-pass",  # pragma: allowlist secret
            database="fake-db",
            ssl=False,
            verify_ssl=False,
            timeout=(5, 60),
            retries=1,
            session=ANY,  # a session that does not follow redirects
        )
        self.assertEqual(mock_influx_client_class.call_args.kwargs["session"].max_redirects, 0)
        mock_client_instance.ping.assert_called_once()
        mock_client_instance.close.assert_called_once()

        # Verify the resulting DataFrame
        df = rh_influx.df_final
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.index), 2)
        self.assertEqual(list(df.columns), var_list)
        self.assertEqual(
            df.loc["2023-04-01 10:00:00+00:00"]["sensor.power_photovoltaics"],
            1500.0,
        )
        self.assertEqual(
            df.loc["2023-04-01 10:30:00+00:00"]["sensor.power_load_no_var_loads"],
            450.0,
        )

    # ------------------------------------------------------------------
    # InfluxDB arithmetic expression support in var_list
    # ------------------------------------------------------------------
    # A var_list entry may use the "{{ ... }}" syntax to combine several timeseries
    # with simple arithmetic, e.g. "{{'sensor.power_a' - 'sensor.power_b' * 1000}}".
    # Each referenced entity is queried separately and the operation is applied
    # element-wise on the values for the matching timestamps. (Idea by @dewi-ny-je, #803.)

    def _make_influxdb_rh(self):
        """Build a RetrieveHass instance configured to use InfluxDB."""
        params_influx = {
            "retrieve_hass_conf": {
                "use_influxdb": True,
                "influxdb_host": "fake-host",
                "influxdb_port": 8086,
                "influxdb_username": "fake-user",
                "influxdb_password": "fake-pass",  # pragma: allowlist secret
                "influxdb_database": "fake-db",
                "influxdb_measurement": "W",
            }
        }
        return RetrieveHass(
            self.retrieve_hass_conf["hass_url"],
            self.retrieve_hass_conf["long_lived_token"],
            self.retrieve_hass_conf["optimization_time_step"],
            self.retrieve_hass_conf["time_zone"],
            params_influx,
            emhass_conf,
            logger,
            get_data_from_file=False,
        )

    @staticmethod
    def _influx_query_side_effect(entity_data, tag_values=None):
        """Build a client.query side_effect serving discovery and data queries.

        :param entity_data: mapping of InfluxDB entity_id -> list of points (each a dict
            with "time" and "mean_value"). An empty list emulates a sensor that exists
            but returns no data.
        :param tag_values: entity_ids advertised by "SHOW TAG VALUES" (defaults to the
            keys of entity_data).
        """
        if tag_values is None:
            tag_values = list(entity_data)

        def query_side_effect(query):
            mock_result = MagicMock()
            if "SHOW MEASUREMENTS" in query:
                mock_result.get_points.return_value = [{"name": "W"}]
            elif "SHOW TAG VALUES" in query:
                mock_result.get_points.return_value = [{"value": value} for value in tag_values]
            else:
                points = []
                for entity_id, data in entity_data.items():
                    if f"'{entity_id}'" in query:
                        points = data
                        break
                mock_result.get_points.return_value = points
            return mock_result

        return query_side_effect

    def test_influx_is_expression(self):
        """_is_influx_expression detects the {{ ... }} arithmetic syntax."""
        self.assertTrue(self.rh._is_influx_expression("{{'sensor.a' - 'sensor.b'}}"))
        self.assertTrue(self.rh._is_influx_expression("  {{ 'sensor.a' * 2 }}  "))
        self.assertFalse(self.rh._is_influx_expression("sensor.power_a"))
        self.assertFalse(self.rh._is_influx_expression("{{ not closed"))
        self.assertFalse(self.rh._is_influx_expression("not opened }}"))
        self.assertFalse(self.rh._is_influx_expression(""))

    def test_influx_extract_expression_entities(self):
        """Quoted entity IDs are replaced by safe tokens and de-duplicated."""
        parsed, entities, token_to_entity = self.rh._extract_influx_expression_entities(
            "{{'sensor.power_a' - 'sensor.power_b' * 1000}}"
        )
        self.assertEqual(parsed, "_v0 - _v1 * 1000")
        self.assertEqual(entities, ["sensor.power_a", "sensor.power_b"])
        self.assertEqual(token_to_entity, {"_v0": "sensor.power_a", "_v1": "sensor.power_b"})
        # Double quotes are accepted as well
        parsed2, entities2, _ = self.rh._extract_influx_expression_entities(
            '{{"sensor.a" + "sensor.b"}}'
        )
        self.assertEqual(parsed2, "_v0 + _v1")
        self.assertEqual(entities2, ["sensor.a", "sensor.b"])
        # A repeated entity reuses the same token (a single query is enough)
        parsed3, entities3, _ = self.rh._extract_influx_expression_entities(
            "{{'sensor.a' + 'sensor.a' / 2}}"
        )
        self.assertEqual(entities3, ["sensor.a"])
        self.assertEqual(parsed3, "_v0 + _v0 / 2")
        # An expression without any entity is rejected
        with self.assertRaises(ValueError):
            self.rh._extract_influx_expression_entities("{{1 + 2}}")

    def test_influx_evaluate_expression(self):
        """Arithmetic is applied element-wise with standard operator precedence."""
        idx = pd.date_range("2023-04-01 10:00", periods=3, freq="30min", tz="UTC")
        series_a = pd.Series([1500.0, 1800.0, 2000.0], index=idx)
        series_b = pd.Series([0.5, 0.3, 1.0], index=idx)
        mapping = {"_v0": series_a, "_v1": series_b}
        # Multiplication binds tighter than subtraction: a - (b * 1000)
        result = self.rh._evaluate_influx_expression("_v0 - _v1 * 1000", mapping)
        pd.testing.assert_series_equal(result, series_a - series_b * 1000, check_names=False)
        # Division combined with a unary minus
        result_div = self.rh._evaluate_influx_expression("-_v0 / _v1", mapping)
        pd.testing.assert_series_equal(result_div, -series_a / series_b, check_names=False)
        # An unknown token raises (no silent zero-fill)
        with self.assertRaises(ValueError):
            self.rh._evaluate_influx_expression("_v0 + _v9", mapping)
        # A malformed expression surfaces a SyntaxError
        with self.assertRaises(SyntaxError):
            self.rh._evaluate_influx_expression("_v0 +", mapping)
        # An expression that does not yield a Series is rejected
        with self.assertRaises(ValueError):
            self.rh._evaluate_influx_expression("1 + 2", {})

    def test_influx_evaluate_expression_rejects_unsafe(self):
        """The evaluator allows only arithmetic over Series and numeric constants."""
        idx = pd.date_range("2023-04-01 10:00", periods=2, freq="30min", tz="UTC")
        mapping = {
            "_v0": pd.Series([1.0, 2.0], index=idx),
            "_v1": pd.Series([3.0, 4.0], index=idx),
        }
        rejected = [
            "_v0.__class__",  # attribute access
            "_v0[0]",  # subscript
            "_v0 > _v1",  # comparison
            "_v0 and _v1",  # boolean operator
            "_v0 * True",  # bool constant (must not be treated as 1)
            "__import__('os')",  # function call
            "os",  # bare name not in the mapping
            "_v0 + 'x'",  # string constant
            "_v0 ** 100000",  # exponent magnitude over the DoS cap
            "(10 ** 1000) ** 100",  # chained power building a huge base is also rejected
        ]
        for expr in rejected:
            with self.assertRaises(ValueError, msg=f"should reject: {expr}"):
                self.rh._evaluate_influx_expression(expr, mapping)
        # A reasonable exponent is still allowed
        result = self.rh._evaluate_influx_expression("_v0 ** 2", mapping)
        pd.testing.assert_series_equal(result, mapping["_v0"] ** 2, check_names=False)

    async def test_influx_expression_pads_and_slices_entity_fetch(self):
        """Expression entities are queried over a padded window then sliced to [start, end)."""
        from emhass.retrieve_hass import INFLUX_EXPRESSION_LOOKBACK

        rh = self._make_influxdb_rh()
        start = pd.Timestamp("2026-06-01 00:00:00")
        end = pd.Timestamp("2026-06-02 00:00:00")
        full_idx = pd.date_range("2026-05-31 12:00", "2026-06-01 12:00", freq="30min", tz="UTC")
        captured = {}

        def fake_fetch(client, entity, fetch_start, fetch_end):
            captured["start"] = fetch_start
            return pd.DataFrame({entity: range(len(full_idx))}, index=full_idx)

        with patch.object(rh, "_fetch_sensor_data", side_effect=fake_fetch):
            df = rh._build_influx_expression_df(MagicMock(), "{{'sensor.a' * 2}}", start, end, {})

        # The entity was fetched over a window padded earlier than the requested start
        self.assertEqual(captured["start"], start - INFLUX_EXPRESSION_LOOKBACK)
        # The result is sliced back to [start, end): no pre-window rows leak through
        start_utc = pd.Timestamp("2026-06-01 00:00:00", tz="UTC")
        self.assertEqual(int((df.index < start_utc).sum()), 0)
        self.assertGreaterEqual(df.index.min(), start_utc)

    async def test_influx_expression_pathological_fails_soft(self):
        """A pathological expression (deep nesting, overflow, /0) fails soft, not crash.

        These raise RecursionError / OverflowError / ZeroDivisionError rather than ValueError,
        so they must be caught and turned into a clean retrieval failure instead of aborting the
        run.
        """
        rh = self._make_influxdb_rh()
        idx = pd.date_range("2026-06-01 00:00", periods=4, freq="30min", tz="UTC")
        series_df = pd.DataFrame({"sensor.a": [1.0, 2.0, 3.0, 4.0]}, index=idx)
        start = pd.Timestamp("2026-06-01 00:00:00")
        end = pd.Timestamp("2026-06-02 00:00:00")

        with patch.object(rh, "_fetch_sensor_data", return_value=series_df):
            # Deep nesting overflows the parser/evaluator recursion limit
            deep = "{{" + "(" * 3000 + "'sensor.a'" + ")" * 3000 + "}}"
            self.assertIsNone(rh._build_influx_expression_df(MagicMock(), deep, start, end, {}))
            # A large constant power overflows when converted to float during the multiply
            overflow = "{{'sensor.a' * (10 ** 1000)}}"
            self.assertIsNone(rh._build_influx_expression_df(MagicMock(), overflow, start, end, {}))
            # A constant division-by-zero raises ZeroDivisionError (an ArithmeticError). Note a
            # Series-by-zero divide does NOT raise (pandas yields inf), so the /0 must be between
            # scalar constants to exercise this branch.
            div_zero = "{{'sensor.a' + 1 / 0}}"
            self.assertIsNone(rh._build_influx_expression_df(MagicMock(), div_zero, start, end, {}))

    async def test_influx_expression_dtype_mismatch_fails_soft(self):
        """A non-numeric (object-dtype) series raises TypeError under arithmetic and fails soft."""
        rh = self._make_influxdb_rh()
        idx = pd.date_range("2026-06-01 00:00", periods=3, freq="30min", tz="UTC")
        # A sensor that returned text values: subtracting a number raises TypeError in pandas
        string_df = pd.DataFrame({"sensor.a": ["on", "off", "on"]}, index=idx)
        start = pd.Timestamp("2026-06-01 00:00:00")
        end = pd.Timestamp("2026-06-02 00:00:00")

        with patch.object(rh, "_fetch_sensor_data", return_value=string_df):
            self.assertIsNone(
                rh._build_influx_expression_df(MagicMock(), "{{'sensor.a' - 1000}}", start, end, {})
            )

    @patch("influxdb.InfluxDBClient", autospec=True)
    async def test_get_data_influxdb_expression(self, mock_influx_client_class):
        """get_data_influxdb evaluates an arithmetic expression across sensors."""
        rh_influx = self._make_influxdb_rh()
        mock_client_instance = mock_influx_client_class.return_value
        mock_client_instance.query.side_effect = self._influx_query_side_effect(
            {
                "power_a": [
                    {"time": "2023-04-01T10:00:00Z", "mean_value": 1500.0},
                    {"time": "2023-04-01T10:30:00Z", "mean_value": 1800.0},
                ],
                "power_b": [
                    {"time": "2023-04-01T10:00:00Z", "mean_value": 0.5},
                    {"time": "2023-04-01T10:30:00Z", "mean_value": 0.3},
                ],
            }
        )
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        expression = "{{'sensor.power_a' - 'sensor.power_b' * 1000}}"
        success = await rh_influx.get_data(days_list, [expression])
        self.assertTrue(success)

        df = rh_influx.df_final
        self.assertEqual(list(df.columns), [expression])
        self.assertAlmostEqual(df.loc["2023-04-01 10:00:00+00:00"][expression], 1500.0 - 0.5 * 1000)
        self.assertAlmostEqual(df.loc["2023-04-01 10:30:00+00:00"][expression], 1800.0 - 0.3 * 1000)
        mock_client_instance.close.assert_called_once()

    @patch("influxdb.InfluxDBClient", autospec=True)
    async def test_get_data_influxdb_expression_entity_cached(self, mock_influx_client_class):
        """An entity used in multiple expressions is queried only once."""
        rh_influx = self._make_influxdb_rh()
        mock_client_instance = mock_influx_client_class.return_value
        mock_client_instance.query.side_effect = self._influx_query_side_effect(
            {
                "power_a": [
                    {"time": "2023-04-01T10:00:00Z", "mean_value": 1000.0},
                    {"time": "2023-04-01T10:30:00Z", "mean_value": 1200.0},
                ],
                "power_b": [
                    {"time": "2023-04-01T10:00:00Z", "mean_value": 100.0},
                    {"time": "2023-04-01T10:30:00Z", "mean_value": 200.0},
                ],
            }
        )
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        e1 = "{{'sensor.power_a' + 'sensor.power_b'}}"
        e2 = "{{'sensor.power_a' - 'sensor.power_b'}}"
        success = await rh_influx.get_data(days_list, [e1, e2])
        self.assertTrue(success)

        df = rh_influx.df_final
        self.assertAlmostEqual(df.loc["2023-04-01 10:00:00+00:00"][e1], 1100.0)
        self.assertAlmostEqual(df.loc["2023-04-01 10:00:00+00:00"][e2], 900.0)
        power_a_data_queries = [
            c.args[0]
            for c in mock_client_instance.query.call_args_list
            if 'AND "entity_id"=' in c.args[0] and "'power_a'" in c.args[0]
        ]
        self.assertEqual(len(power_a_data_queries), 1)

    @patch("influxdb.InfluxDBClient", autospec=True)
    async def test_get_data_influxdb_expression_mixed(self, mock_influx_client_class):
        """var_list may mix a plain sensor and an expression."""
        rh_influx = self._make_influxdb_rh()
        mock_client_instance = mock_influx_client_class.return_value
        mock_client_instance.query.side_effect = self._influx_query_side_effect(
            {
                "power_a": [
                    {"time": "2023-04-01T10:00:00Z", "mean_value": 1500.0},
                    {"time": "2023-04-01T10:30:00Z", "mean_value": 1800.0},
                ],
                "power_b": [
                    {"time": "2023-04-01T10:00:00Z", "mean_value": 500.0},
                    {"time": "2023-04-01T10:30:00Z", "mean_value": 450.0},
                ],
            }
        )
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        expression = "{{'sensor.power_a' + 'sensor.power_b'}}"
        var_list = ["sensor.power_a", expression]
        success = await rh_influx.get_data(days_list, var_list)
        self.assertTrue(success)

        df = rh_influx.df_final
        self.assertEqual(list(df.columns), var_list)
        self.assertAlmostEqual(df.loc["2023-04-01 10:00:00+00:00"]["sensor.power_a"], 1500.0)
        self.assertAlmostEqual(df.loc["2023-04-01 10:00:00+00:00"][expression], 2000.0)
        self.assertAlmostEqual(df.loc["2023-04-01 10:30:00+00:00"][expression], 2250.0)

    @patch("influxdb.InfluxDBClient", autospec=True)
    async def test_get_data_influxdb_expression_missing_entity(self, mock_influx_client_class):
        """An expression referencing a sensor with no data fails cleanly."""
        rh_influx = self._make_influxdb_rh()
        mock_client_instance = mock_influx_client_class.return_value
        mock_client_instance.query.side_effect = self._influx_query_side_effect(
            {
                "power_a": [
                    {"time": "2023-04-01T10:00:00Z", "mean_value": 1500.0},
                    {"time": "2023-04-01T10:30:00Z", "mean_value": 1800.0},
                ],
                "power_missing": [],
            },
            tag_values=["power_a"],
        )
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        expression = "{{'sensor.power_a' + 'sensor.power_missing'}}"
        success = await rh_influx.get_data(days_list, [expression])
        self.assertFalse(success)

    @patch("influxdb.InfluxDBClient", autospec=True)
    async def test_get_data_influxdb_plain_missing_sensor_skipped(self, mock_influx_client_class):
        """A missing PLAIN sensor is skipped, not a hard failure (unchanged behavior).

        Only expressions abort the retrieval; a plain var_list keeps its prior behavior
        of dropping a sensor that returned no data and proceeding with the rest.
        """
        rh_influx = self._make_influxdb_rh()
        mock_client_instance = mock_influx_client_class.return_value
        mock_client_instance.query.side_effect = self._influx_query_side_effect(
            {
                "power_a": [
                    {"time": "2023-04-01T10:00:00Z", "mean_value": 1500.0},
                    {"time": "2023-04-01T10:30:00Z", "mean_value": 1800.0},
                ],
            },
            tag_values=["power_a"],
        )
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        success = await rh_influx.get_data(days_list, ["sensor.power_a", "sensor.power_missing"])
        self.assertTrue(success)
        self.assertEqual(list(rh_influx.df_final.columns), ["sensor.power_a"])

    # ------------------------------------------------------------------
    # VictoriaMetrics data source: PromQL query_range over HTTP, mocked at the
    # single request helper (_vm_query_range) so no network is involved.

    def _make_vm_rh(self, **overrides):
        """Build a RetrieveHass instance configured to use VictoriaMetrics."""
        conf = {
            "use_victoriametrics": True,
            "victoriametrics_host": "fake-vm",
            "victoriametrics_port": 8428,
            "victoriametrics_username": "fake-user",
            "victoriametrics_password": "fake-pass",  # pragma: allowlist secret
            "victoriametrics_database": "homeassistant",
        }
        conf.update(overrides)
        return RetrieveHass(
            self.retrieve_hass_conf["hass_url"],
            self.retrieve_hass_conf["long_lived_token"],
            self.retrieve_hass_conf["optimization_time_step"],
            self.retrieve_hass_conf["time_zone"],
            {"retrieve_hass_conf": conf},
            emhass_conf,
            logger,
            get_data_from_file=False,
        )

    @staticmethod
    def _vm_query_side_effect(
        entity_data, calls=None, keep_metric_name=True, entity_label="entity_id"
    ):
        """Build a ``_vm_query_range`` side_effect emulating VictoriaMetrics.

        :param entity_data: mapping of entity label value -> list of series, each a tuple
            ``(metric_name, [(bucket_start_iso, value), ...])``. Timestamps are given as
            the START of the bucket (the label EMHASS must produce); the fake server
            returns them the way VictoriaMetrics does, at the END of the
            ``avg_over_time`` window (bucket start + step). An empty list emulates an
            entity with no data.
        :param calls: optional list collecting ``(query, start, end, step)`` per call.
        :param keep_metric_name: emulate MetricsQL (``__name__`` kept through
            ``avg_over_time``, the default) or a plain PromQL backend that drops it.
        :param entity_label: label the selector locates the entity with (``entity_id``
            for InfluxDB-integration data, ``entity`` for Prometheus-integration data).
        """

        async def query_side_effect(session, query, start_s, end_s, step_s):
            if calls is not None:
                calls.append((query, start_s, end_s, step_s))
            entity_id = query.split(f'{{{entity_label}="')[1].split('"')[0]
            result = []
            for metric_name, points in entity_data.get(entity_id, []):
                values = []
                for ts_iso, value in points:
                    ts = int(pd.Timestamp(ts_iso).timestamp()) + step_s
                    if start_s <= ts <= end_s:
                        values.append([ts, str(value)])
                if values:
                    metric = {entity_label: entity_id, "domain": "sensor"}
                    if keep_metric_name:
                        metric["__name__"] = metric_name
                    result.append({"metric": metric, "values": values})
            return result

        return query_side_effect

    def test_vm_selector(self):
        """The selector matches on entity_id/domain labels and a metric name regex."""
        rh = self._make_vm_rh()
        self.assertEqual(
            rh._vm_selector("sensor.power_a"),
            '{entity_id="power_a",__name__=~".+_value",domain="sensor",db="homeassistant"}',
        )
        # No domain in the entry and no database filter configured
        rh = self._make_vm_rh(victoriametrics_database="", victoriametrics_metric_regex="W_value")
        self.assertEqual(rh._vm_selector("power_a"), '{entity_id="power_a",__name__=~"W_value"}')
        # Quotes and backslashes in a value cannot break out of the matcher
        self.assertIn('entity_id="a\\"b\\\\c"', rh._vm_selector('sensor.a"b\\c'))

    def test_vm_selector_entity_label(self):
        """With an entity label the full entity id is matched on that label alone."""
        rh = self._make_vm_rh(
            victoriametrics_database="",
            victoriametrics_metric_regex="homeassistant_sensor_.+",
            victoriametrics_entity_label=" entity ",
        )
        self.assertEqual(
            rh._vm_selector("sensor.power_a"),
            '{entity="sensor.power_a",__name__=~"homeassistant_sensor_.+"}',
        )
        self.assertIn('entity="sensor.a\\"b\\\\c"', rh._vm_selector('sensor.a"b\\c'))
        # The database filter still applies when configured
        rh = self._make_vm_rh(victoriametrics_entity_label="entity")
        self.assertTrue(rh._vm_selector("sensor.power_a").endswith(',db="homeassistant"}'))
        # A label name cannot be quoted, so an invalid one is refused with an error and
        # the entity_id/domain matching is used instead
        with self.assertLogs(logger, level="ERROR") as logs:
            rh = self._make_vm_rh(victoriametrics_entity_label='entity="x"} or vector(1) #')
        self.assertIn("Invalid victoriametrics_entity_label", logs.output[0])
        self.assertTrue(rh._vm_selector("sensor.power_a").startswith('{entity_id="power_a",'))

    def test_vm_connection_settings(self):
        """Host/port/ssl/auth settings map onto the HTTP client arguments."""
        rh = self._make_vm_rh()
        self.assertEqual(rh._vm_base_url(), "http://fake-vm:8428")
        self.assertEqual(
            rh._vm_auth_headers(), {"Authorization": "Basic ZmFrZS11c2VyOmZha2UtcGFzcw=="}
        )
        self.assertIsNone(rh._vm_ssl())
        rh = self._make_vm_rh(
            victoriametrics_username="",
            victoriametrics_use_ssl=True,
            victoriametrics_verify_ssl=False,
        )
        self.assertEqual(rh._vm_base_url(), "https://fake-vm:8428")
        self.assertEqual(rh._vm_auth_headers(), {})
        self.assertIs(rh._vm_ssl(), False)

    @staticmethod
    def _vm_mock_session(
        status=200, body=b'{"status": "success", "data": {"result": []}}', exc=None
    ):
        """Mock aiohttp session whose ``get`` records its call and yields a canned response."""
        response = MagicMock()
        response.status = status
        response.read = AsyncMock(return_value=body)
        context = MagicMock()
        if exc is None:
            context.__aenter__ = AsyncMock(return_value=response)
        else:
            context.__aenter__ = AsyncMock(side_effect=exc)
        context.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.get = MagicMock(return_value=context)
        return session

    async def _vm_request(self, rh, session):
        return await rh._vm_query_range(session, "up", 0, 60, 60)

    async def test_vm_query_range_authorization_header(self):
        """Credentials are sent as an explicit Authorization header (latin-1, as BasicAuth did)."""
        cases = [
            ({}, "Basic ZmFrZS11c2VyOmZha2UtcGFzcw=="),
            (
                {"victoriametrics_username": "usér", "victoriametrics_password": "pässwörd"},
                "Basic dXPpcjpw5HNzd/ZyZA==",
            ),
            ({"victoriametrics_password": ""}, "Basic ZmFrZS11c2VyOg=="),
            ({"victoriametrics_password": None}, "Basic ZmFrZS11c2VyOg=="),
            (
                {"victoriametrics_username": "u", "victoriametrics_password": "p:w"},
                "Basic dTpwOnc=",
            ),
        ]
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                rh = self._make_vm_rh(**overrides)
                session = self._vm_mock_session()
                self.assertEqual(await self._vm_request(rh, session), [])
                session.get.assert_called_once()
                kwargs = session.get.call_args.kwargs
                self.assertEqual(kwargs["headers"], {"Authorization": expected})
                self.assertNotIn("auth", kwargs)
                self.assertEqual(
                    kwargs["params"], {"query": "up", "start": 0, "end": 60, "step": 60}
                )
                self.assertEqual(
                    session.get.call_args.args, ("http://fake-vm:8428/api/v1/query_range",)
                )
                self.assertIsNone(kwargs["ssl"])

    async def test_vm_query_range_no_username_sends_no_authorization(self):
        """Without a username no Authorization header is sent, even if a password is set."""
        for username in ("", None):
            with self.subTest(username=username):
                rh = self._make_vm_rh(victoriametrics_username=username)
                session = self._vm_mock_session()
                self.assertEqual(await self._vm_request(rh, session), [])
                kwargs = session.get.call_args.kwargs
                self.assertEqual(kwargs["headers"], {})
                self.assertNotIn("auth", kwargs)

    async def test_vm_query_range_no_deprecation_warning(self):
        """Neither authenticated nor unauthenticated requests warn (warnings as errors)."""
        for overrides in ({}, {"victoriametrics_username": ""}):
            with self.subTest(overrides=overrides):
                rh = self._make_vm_rh(**overrides)
                session = self._vm_mock_session()
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    self.assertEqual(await self._vm_request(rh, session), [])

    async def test_vm_query_range_errors_are_non_fatal_and_logged(self):
        """401, HTTP errors, network failures and invalid JSON return None and are logged."""
        cases = [
            ({"status": 401}, "401 Unauthorized"),
            ({"status": 500, "body": b"boom"}, "query failed: unexpected reply"),
            ({"exc": aiohttp.ClientConnectionError("down")}, "Network error connecting"),
            ({"body": b"not json"}, "query failed: unexpected reply"),
        ]
        for kwargs, expected in cases:
            with self.subTest(expected=expected):
                rh = self._make_vm_rh()
                session = self._vm_mock_session(**kwargs)
                with self.assertLogs(logger, level="ERROR") as logs:
                    self.assertIsNone(await self._vm_request(rh, session))
                self.assertEqual(len(logs.output), 1)
                self.assertIn(expected, logs.output[0])
                self.assertEqual(
                    session.get.call_args.kwargs["headers"],
                    {"Authorization": "Basic ZmFrZS11c2VyOmZha2UtcGFzcw=="},
                )
                self.assertNotIn("ZmFrZS11c2VyOmZha2UtcGFzcw", logs.output[0])
                self.assertNotIn("fake-pass", logs.output[0])

    async def test_vm_query_range_invalid_credentials_are_non_fatal_and_not_leaked(self):
        """A colon in the username or an unencodable character fails softly, without leaks."""
        secret_user = "sec:retuser"  # pragma: allowlist secret
        cases = [
            {"victoriametrics_username": secret_user, "victoriametrics_password": "hunter2"},
            {"victoriametrics_username": "sec€user", "victoriametrics_password": "hunter2"},
            {"victoriametrics_username": "secuser", "victoriametrics_password": "hunt€r2"},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                rh = self._make_vm_rh(**overrides)
                session = self._vm_mock_session()
                with self.assertLogs(logger, level="ERROR") as logs:
                    self.assertIsNone(await self._vm_request(rh, session))
                session.get.assert_not_called()
                self.assertEqual(len(logs.output), 1)
                self.assertIn("Invalid VictoriaMetrics credentials", logs.output[0])
                with self.assertRaises(ValueError) as ctx:
                    rh._vm_auth_headers()
                self.assertTrue(
                    ctx.exception.__context__ is None or ctx.exception.__suppress_context__
                )
                for text in (logs.output[0], str(ctx.exception)):
                    for leaked in ("sec", "hunt", "retuser", "u20ac", "€"):
                        self.assertNotIn(leaked, text)

    async def test_get_data_victoriametrics_mock(self):
        """get_data routes to VictoriaMetrics and yields the InfluxDB-shaped frame."""
        rh = self._make_vm_rh()
        calls = []
        side_effect = self._vm_query_side_effect(
            {
                "power_photovoltaics": [
                    (
                        "W_value",
                        [("2023-04-01T10:00:00Z", 1500.0), ("2023-04-01T10:30:00Z", 1800.0)],
                    )
                ],
                "power_load_no_var_loads": [
                    ("W_value", [("2023-04-01T10:00:00Z", 500.0), ("2023-04-01T10:30:00Z", 450.0)])
                ],
            },
            calls,
        )
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        var_list = ["sensor.power_photovoltaics", "sensor.power_load_no_var_loads"]
        with patch.object(rh, "_vm_query_range", side_effect=side_effect):
            success = await rh.get_data(days_list, var_list)
        self.assertTrue(success)

        df = rh.df_final
        self.assertEqual(list(df.columns), var_list)
        # The grid is held (FILL previous) from the first sample through the last bucket
        # strictly before end_time (2023-04-02T00:00:00Z), i.e. 10:00 .. 23:30 at 30min.
        self.assertEqual(len(df), 28)
        self.assertEqual(df.index.freq, pd.Timedelta("30min"))
        self.assertEqual(str(df.index.tz), "UTC")
        # Buckets are labelled by their start, like InfluxDB GROUP BY time()
        self.assertAlmostEqual(df.loc["2023-04-01 10:00:00+00:00"][var_list[0]], 1500.0)
        self.assertAlmostEqual(df.loc["2023-04-01 10:30:00+00:00"][var_list[1]], 450.0)
        # The terminal state (last known sample) is held through the last eligible bucket
        self.assertAlmostEqual(df.loc["2023-04-01 23:30:00+00:00"][var_list[0]], 1800.0)
        self.assertAlmostEqual(df.loc["2023-04-01 23:30:00+00:00"][var_list[1]], 450.0)
        self.assertEqual(rh.var_list, var_list)
        # One query_range call per sensor, on a step-aligned grid at the optimization step
        self.assertEqual(len(calls), 2)
        query, start_s, end_s, step_s = calls[0]
        self.assertEqual(step_s, 1800)
        self.assertEqual(start_s % 1800, 0)
        self.assertEqual(start_s, int(pd.Timestamp("2023-04-01T00:30:00Z").timestamp()))
        self.assertEqual(end_s, int(pd.Timestamp("2023-04-02T00:00:00Z").timestamp()))
        self.assertTrue(query.startswith("avg_over_time({entity_id="))
        self.assertTrue(query.endswith("}[1800s])"))

    async def test_get_data_victoriametrics_entity_label(self):
        """Prometheus-integration series are read through the entity label, and series
        split by a label change (same metric name) are joined into one history."""
        rh = self._make_vm_rh(
            victoriametrics_database="",
            victoriametrics_metric_regex="homeassistant_sensor_.+",
            victoriametrics_entity_label="entity",
        )
        calls = []
        side_effect = self._vm_query_side_effect(
            {
                "sensor.power_photovoltaics": [
                    (
                        "homeassistant_sensor_power_w",
                        [("2023-04-01T10:00:00Z", 1500.0), ("2023-04-01T10:30:00Z", 1800.0)],
                    )
                ],
                "sensor.power_load_no_var_loads": [
                    ("homeassistant_sensor_power_w", [("2023-04-01T10:00:00Z", 500.0)]),
                    ("homeassistant_sensor_power_w", [("2023-04-01T10:30:00Z", 450.0)]),
                ],
            },
            calls,
            entity_label="entity",
        )
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        var_list = ["sensor.power_photovoltaics", "sensor.power_load_no_var_loads"]
        with patch.object(rh, "_vm_query_range", side_effect=side_effect):
            self.assertTrue(await rh.get_data(days_list, var_list))
        df = rh.df_final
        self.assertEqual(list(df.columns), var_list)
        self.assertEqual(len(df), 28)
        self.assertAlmostEqual(df.loc["2023-04-01 10:00:00+00:00"][var_list[0]], 1500.0)
        self.assertAlmostEqual(df.loc["2023-04-01 10:00:00+00:00"][var_list[1]], 500.0)
        self.assertAlmostEqual(df.loc["2023-04-01 10:30:00+00:00"][var_list[1]], 450.0)
        self.assertAlmostEqual(df.loc["2023-04-01 23:30:00+00:00"][var_list[1]], 450.0)
        self.assertEqual(len(calls), 2)
        self.assertTrue(
            calls[0][0].startswith('avg_over_time({entity="sensor.power_photovoltaics",')
        )

    async def test_get_data_victoriametrics_fill_previous(self):
        """Empty buckets are forward-filled like InfluxDB FILL(previous)."""
        rh = self._make_vm_rh()
        side_effect = self._vm_query_side_effect(
            {
                "price": [
                    (
                        "EUR/kWh_value",
                        [("2023-04-01T10:00:00Z", 0.2), ("2023-04-01T11:30:00Z", 0.3)],
                    )
                ],
            }
        )
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        with patch.object(rh, "_vm_query_range", side_effect=side_effect):
            self.assertTrue(await rh.get_data(days_list, ["sensor.price"]))
        df = rh.df_final
        # Internal gap (10:00 -> 11:30) uses the previous value; the terminal state (0.3)
        # is then held through the last bucket strictly before end_time (23:30).
        self.assertEqual(len(df), 28)
        self.assertEqual(df["sensor.price"].tolist()[:4], [0.2, 0.2, 0.2, 0.3])
        self.assertEqual(df["sensor.price"].tolist()[4:], [0.3] * 24)
        self.assertEqual(df.index.max(), pd.Timestamp("2023-04-01 23:30:00+00:00"))

    async def test_get_data_victoriametrics_multiple_metrics(self):
        """When a sensor changed unit the metric with the most samples is kept."""
        rh = self._make_vm_rh()
        side_effect = self._vm_query_side_effect(
            {
                "power_a": [
                    ("kW_value", [("2023-04-01T09:30:00Z", 1.0)]),
                    (
                        "W_value",
                        [("2023-04-01T10:00:00Z", 1000.0), ("2023-04-01T10:30:00Z", 1200.0)],
                    ),
                ],
            }
        )
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        with (
            patch.object(rh, "_vm_query_range", side_effect=side_effect),
            self.assertLogs(logger, level="WARNING") as logs,
        ):
            self.assertTrue(await rh.get_data(days_list, ["sensor.power_a"]))
        values = rh.df_final["sensor.power_a"].tolist()
        self.assertEqual(values[:2], [1000.0, 1200.0])
        self.assertEqual(values[2:], [1200.0] * 26)
        self.assertTrue(any("several VictoriaMetrics metrics" in line for line in logs.output))

    async def test_get_data_victoriametrics_metric_name_dropped(self):
        """A backend that drops __name__ still yields one series per label set.

        Two unit metrics of one sensor then collapse into one series with overlapping
        timestamps: the first value is kept and a warning is logged, instead of failing.
        """
        rh = self._make_vm_rh()
        side_effect = self._vm_query_side_effect(
            {
                "power_a": [
                    ("kW_value", [("2023-04-01T10:00:00Z", 1.0)]),
                    (
                        "W_value",
                        [("2023-04-01T10:00:00Z", 1000.0), ("2023-04-01T10:30:00Z", 1200.0)],
                    ),
                ],
            },
            keep_metric_name=False,
        )
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        with (
            patch.object(rh, "_vm_query_range", side_effect=side_effect),
            self.assertLogs(logger, level="WARNING") as logs,
        ):
            self.assertTrue(await rh.get_data(days_list, ["sensor.power_a"]))
        values = rh.df_final["sensor.power_a"].tolist()
        self.assertEqual(len(rh.df_final), 28)
        self.assertEqual(values[:2], [1.0, 1200.0])
        self.assertEqual(values[2:], [1200.0] * 26)
        self.assertTrue(any("duplicate timestamps" in line for line in logs.output))
        self.assertFalse(any("several VictoriaMetrics metrics" in line for line in logs.output))

    async def test_get_data_victoriametrics_expression_mixed(self):
        """var_list may mix a plain sensor and a {{ ... }} expression, as with InfluxDB."""
        rh = self._make_vm_rh()
        calls = []
        side_effect = self._vm_query_side_effect(
            {
                "power_a": [
                    (
                        "W_value",
                        [("2023-04-01T10:00:00Z", 1500.0), ("2023-04-01T10:30:00Z", 1800.0)],
                    )
                ],
                "power_b": [
                    ("kW_value", [("2023-04-01T10:00:00Z", 0.5), ("2023-04-01T10:30:00Z", 0.3)])
                ],
            },
            calls,
        )
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        expression = "{{'sensor.power_a' - 'sensor.power_b' * 1000}}"
        var_list = ["sensor.power_a", expression]
        with patch.object(rh, "_vm_query_range", side_effect=side_effect):
            self.assertTrue(await rh.get_data(days_list, var_list))
        df = rh.df_final
        self.assertEqual(list(df.columns), var_list)
        self.assertAlmostEqual(df.loc["2023-04-01 10:00:00+00:00"]["sensor.power_a"], 1500.0)
        self.assertAlmostEqual(df.loc["2023-04-01 10:00:00+00:00"][expression], 1000.0)
        self.assertAlmostEqual(df.loc["2023-04-01 10:30:00+00:00"][expression], 1500.0)
        # Plain entry over the requested window, expression entities over a padded one:
        # power_a is fetched twice (plain + expression), power_b once (expression only)
        from emhass.retrieve_hass import INFLUX_EXPRESSION_LOOKBACK

        window_starts: dict[str, list[int]] = {}
        for q, s, _, _ in calls:
            window_starts.setdefault(q.split('entity_id="')[1].split('"')[0], []).append(s)
        lookback = int(INFLUX_EXPRESSION_LOOKBACK.total_seconds())
        self.assertEqual(len(window_starts["power_a"]), 2)
        self.assertEqual(max(window_starts["power_a"]) - min(window_starts["power_a"]), lookback)
        self.assertEqual(window_starts["power_b"], [min(window_starts["power_a"])])

    async def test_get_data_victoriametrics_missing_sensor_and_entity(self):
        """A missing plain sensor is skipped; a missing expression entity fails cleanly."""
        rh = self._make_vm_rh()
        side_effect = self._vm_query_side_effect(
            {
                "power_a": [
                    (
                        "W_value",
                        [("2023-04-01T10:00:00Z", 1500.0), ("2023-04-01T10:30:00Z", 1800.0)],
                    )
                ],
                "power_missing": [],
            }
        )
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        with patch.object(rh, "_vm_query_range", side_effect=side_effect):
            self.assertTrue(
                await rh.get_data(days_list, ["sensor.power_a", "sensor.power_missing"])
            )
            self.assertEqual(list(rh.df_final.columns), ["sensor.power_a"])
            self.assertFalse(
                await rh.get_data(days_list, ["{{'sensor.power_a' + 'sensor.power_missing'}}"])
            )

    async def test_get_data_victoriametrics_query_error(self):
        """A failed query_range (None from the helper) aborts the retrieval."""
        rh = self._make_vm_rh()
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        with patch.object(rh, "_vm_query_range", return_value=None):
            self.assertFalse(await rh.get_data(days_list, ["sensor.power_a"]))

    async def test_vm_fetch_chunks_long_windows(self):
        """Long windows are split into contiguous chunks under the points-per-request cap."""
        from emhass.retrieve_hass import VM_MAX_POINTS_PER_REQUEST

        rh = self._make_vm_rh()
        rh.freq = pd.Timedelta("15min")
        calls = []
        side_effect = self._vm_query_side_effect({}, calls)
        start = pd.Timestamp("2025-01-01 00:00:00")
        end = pd.Timestamp("2026-01-01 00:00:00")  # 35040 steps > 30000 default cap
        with patch.object(rh, "_vm_query_range", side_effect=side_effect):
            async with aiohttp.ClientSession() as session:
                result = await rh._fetch_sensor_data_vm(session, "sensor.power_a", start, end)
        self.assertIsNone(result)  # no data in the fake server
        self.assertEqual(len(calls), 4)
        step = 900
        for _, s, e, st in calls:
            self.assertEqual(st, step)
            self.assertLessEqual((e - s) // step + 1, VM_MAX_POINTS_PER_REQUEST)
        # Chunks are contiguous, one step apart, and cover the whole window
        for (_, _, e_prev, _), (_, s_next, _, _) in zip(calls, calls[1:], strict=False):
            self.assertEqual(s_next, e_prev + step)
        self.assertEqual(calls[0][1], int(start.tz_localize("UTC").timestamp()) + step)
        self.assertEqual(calls[-1][2], int(end.tz_localize("UTC").timestamp()))

    # Regression coverage for the VictoriaMetrics terminal FILL(previous) tail defect:
    # the reindex used to stop at the last SAMPLE timestamp instead of the last ELIGIBLE
    # bucket before end_time, silently dropping trailing held-state buckets as NaN.

    async def test_vm_fill_previous_no_backward_fill_before_first_sample(self):
        """Leading buckets before the first known state stay NaN (no backward fill)."""
        rh = self._make_vm_rh()
        side_effect = self._vm_query_side_effect(
            {"sensor_a": [("W_value", [("2023-04-01T02:00:00Z", 10.0)])]}
        )
        start = pd.Timestamp("2023-04-01T00:00:00Z")
        end = pd.Timestamp("2023-04-01T04:00:00Z")
        with patch.object(rh, "_vm_query_range", side_effect=side_effect):
            async with aiohttp.ClientSession() as session:
                df = await rh._fetch_sensor_data_vm(session, "sensor.sensor_a", start, end)
        # No buckets at all before the first sample (00:00, 00:30, 01:00, 01:30 absent)
        self.assertEqual(df.index.min(), pd.Timestamp("2023-04-01 02:00:00+00:00"))
        self.assertNotIn(pd.Timestamp("2023-04-01 00:00:00+00:00"), df.index)
        self.assertNotIn(pd.Timestamp("2023-04-01 01:30:00+00:00"), df.index)

    async def test_vm_fill_previous_terminal_state_extends_to_query_end(self):
        """A sparse state last written before end_time is held through every eligible
        bucket strictly before end_time (half-open interval [start_time, end_time))."""
        rh = self._make_vm_rh()
        side_effect = self._vm_query_side_effect(
            {"sensor_a": [("W_value", [("2023-04-01T02:00:00Z", 10.0)])]}
        )
        start = pd.Timestamp("2023-04-01T02:00:00Z")
        end = pd.Timestamp("2023-04-01T04:00:00Z")
        with patch.object(rh, "_vm_query_range", side_effect=side_effect):
            async with aiohttp.ClientSession() as session:
                df = await rh._fetch_sensor_data_vm(session, "sensor.sensor_a", start, end)
        # Last eligible bucket is end_time - step = 03:30; no bucket at or beyond end_time
        self.assertEqual(df.index.max(), pd.Timestamp("2023-04-01 03:30:00+00:00"))
        self.assertNotIn(pd.Timestamp("2023-04-01 04:00:00+00:00"), df.index)
        self.assertEqual(df["sensor.sensor_a"].tolist(), [10.0] * len(df))

    async def test_vm_fill_previous_non_aligned_end_time_uses_last_complete_bucket(self):
        """An off-grid end_time includes only complete buckets wholly inside [start, end)."""
        rh = self._make_vm_rh()
        side_effect = self._vm_query_side_effect(
            {"sensor_a": [("W_value", [("2023-04-01T02:00:00Z", 10.0)])]}
        )
        start = pd.Timestamp("2023-04-01T02:00:00Z")
        end = pd.Timestamp("2023-04-01T04:10:00Z")
        with patch.object(rh, "_vm_query_range", side_effect=side_effect):
            async with aiohttp.ClientSession() as session:
                df = await rh._fetch_sensor_data_vm(session, "sensor.sensor_a", start, end)

        # With 30-minute buckets, 03:30 labels the final complete bucket (03:30, 04:00].
        # The next bucket would end at 04:30 and therefore lies outside [start, 04:10).
        self.assertEqual(df.index.max(), pd.Timestamp("2023-04-01 03:30:00+00:00"))
        self.assertNotIn(pd.Timestamp("2023-04-01 04:00:00+00:00"), df.index)
        self.assertTrue((df.index < end).all())
        self.assertEqual(df["sensor.sensor_a"].tolist(), [10.0] * len(df))

    async def test_vm_fill_previous_terminal_constrained_positive_state_held(self):
        """A terminal bounded/limited positive state (e.g. a ceiling register) remains at
        its held value through end_time, not just at the sample timestamp."""
        rh = self._make_vm_rh()
        side_effect = self._vm_query_side_effect(
            {
                "limit_register": [
                    (
                        "W_value",
                        [
                            ("2023-04-01T02:00:00Z", 0.0),
                            ("2023-04-01T02:30:00Z", 4000.0),  # becomes constrained/binding
                        ],
                    )
                ]
            }
        )
        start = pd.Timestamp("2023-04-01T02:00:00Z")
        end = pd.Timestamp("2023-04-01T04:00:00Z")
        with patch.object(rh, "_vm_query_range", side_effect=side_effect):
            async with aiohttp.ClientSession() as session:
                df = await rh._fetch_sensor_data_vm(session, "sensor.limit_register", start, end)
        self.assertEqual(df.index.max(), pd.Timestamp("2023-04-01 03:30:00+00:00"))
        # The binding positive value is held through every trailing bucket, not lost to NaN
        self.assertEqual(df["sensor.limit_register"].tolist(), [0.0, 4000.0, 4000.0, 4000.0])
        self.assertFalse(df["sensor.limit_register"].isna().any())

    async def test_vm_fill_previous_terminal_zero_state_held(self):
        """A terminal non-binding zero state is correctly held at zero through end_time
        (not merely coincidentally zero because of a NaN->fillna(0.0) downstream bug)."""
        rh = self._make_vm_rh()
        side_effect = self._vm_query_side_effect(
            {
                "limit_register": [
                    (
                        "W_value",
                        [
                            ("2023-04-01T02:00:00Z", 4000.0),
                            ("2023-04-01T02:30:00Z", 0.0),  # constraint released
                        ],
                    )
                ]
            }
        )
        start = pd.Timestamp("2023-04-01T02:00:00Z")
        end = pd.Timestamp("2023-04-01T04:00:00Z")
        with patch.object(rh, "_vm_query_range", side_effect=side_effect):
            async with aiohttp.ClientSession() as session:
                df = await rh._fetch_sensor_data_vm(session, "sensor.limit_register", start, end)
        self.assertEqual(df.index.max(), pd.Timestamp("2023-04-01 03:30:00+00:00"))
        self.assertEqual(df["sensor.limit_register"].tolist(), [4000.0, 0.0, 0.0, 0.0])
        self.assertFalse(df["sensor.limit_register"].isna().any())

    async def test_vm_fill_previous_transition_at_end_time_excluded(self):
        """A state transition occurring exactly at end_time must not create a bucket at or
        beyond end_time, and must not leak backward into earlier buckets."""
        rh = self._make_vm_rh()
        side_effect = self._vm_query_side_effect(
            {
                "sensor_a": [
                    (
                        "W_value",
                        [
                            ("2023-04-01T02:00:00Z", 10.0),
                            ("2023-04-01T04:00:00Z", 99.0),  # exactly at end_time
                        ],
                    )
                ]
            }
        )
        start = pd.Timestamp("2023-04-01T02:00:00Z")
        end = pd.Timestamp("2023-04-01T04:00:00Z")
        with patch.object(rh, "_vm_query_range", side_effect=side_effect):
            async with aiohttp.ClientSession() as session:
                df = await rh._fetch_sensor_data_vm(session, "sensor.sensor_a", start, end)
        self.assertNotIn(pd.Timestamp("2023-04-01 04:00:00+00:00"), df.index)
        self.assertEqual(df.index.max(), pd.Timestamp("2023-04-01 03:30:00+00:00"))
        # The later (end_time) value never leaks backward into earlier held buckets
        self.assertTrue((df["sensor.sensor_a"] == 10.0).all())

    async def test_vm_fill_previous_terminal_fill_across_chunk_boundary(self):
        """The terminal fill is correct even when the window spans several VM query
        chunks (the last sample lands in an earlier chunk than the final bucket)."""
        rh = self._make_vm_rh()
        rh.freq = pd.Timedelta("15min")
        from emhass.retrieve_hass import VM_MAX_POINTS_PER_REQUEST

        step = 900
        chunk_span_steps = VM_MAX_POINTS_PER_REQUEST
        # Last sample lands in the first chunk; the window extends ~1.5 chunks further so
        # the terminal fill must be carried through a chunk boundary with no new data.
        start = pd.Timestamp("2025-01-01T00:00:00Z")
        last_sample = start + pd.Timedelta(minutes=15)
        end = start + pd.Timedelta(seconds=int(chunk_span_steps * step * 1.5))
        calls = []
        side_effect = self._vm_query_side_effect(
            {"sensor_a": [("W_value", [(last_sample.isoformat(), 7.0)])]}, calls
        )
        with patch.object(rh, "_vm_query_range", side_effect=side_effect):
            async with aiohttp.ClientSession() as session:
                df = await rh._fetch_sensor_data_vm(session, "sensor.sensor_a", start, end)
        self.assertGreaterEqual(len(calls), 2)  # confirms chunking actually occurred
        expected_last_bucket = end - rh.freq
        self.assertEqual(df.index.max(), expected_last_bucket)
        self.assertTrue((df["sensor.sensor_a"] == 7.0).all())
        self.assertFalse(df["sensor.sensor_a"].isna().any())

    async def test_vm_fill_previous_dense_sensor_unchanged(self):
        """A dense sensor (a sample at every bucket already) is value-for-value unchanged
        by the terminal-fill fix: the reindex extension is a no-op when there is no gap."""
        rh = self._make_vm_rh()
        points = [
            (f"2023-04-01T{hour:02d}:{minute:02d}:00Z", float(hour * 2 + minute / 30))
            for hour in range(2, 4)
            for minute in (0, 30)
        ]
        side_effect = self._vm_query_side_effect({"dense_sensor": [("W_value", points)]})
        start = pd.Timestamp("2023-04-01T02:00:00Z")
        end = pd.Timestamp("2023-04-01T04:00:00Z")
        with patch.object(rh, "_vm_query_range", side_effect=side_effect):
            async with aiohttp.ClientSession() as session:
                df = await rh._fetch_sensor_data_vm(session, "sensor.dense_sensor", start, end)
        expected = [float(h * 2 + m / 30) for h, m in [(2, 0), (2, 30), (3, 0), (3, 30)]]
        self.assertEqual(df["sensor.dense_sensor"].tolist(), expected)
        self.assertFalse(df["sensor.dense_sensor"].isna().any())

    async def test_get_data_victoriametrics_expression_terminal_fill(self):
        """End-to-end: an arithmetic expression built over VM-retrieved entities receives
        the corrected terminal-filled series, not a truncated-then-NaN tail."""
        rh = self._make_vm_rh()
        side_effect = self._vm_query_side_effect(
            {
                "power_a": [("W_value", [("2023-04-01T10:00:00Z", 1500.0)])],
                "power_b": [("kW_value", [("2023-04-01T10:00:00Z", 0.5)])],
            }
        )
        days_list = pd.date_range(start="2023-04-01", periods=1, freq="D", tz="UTC")
        expression = "{{'sensor.power_a' - 'sensor.power_b' * 1000}}"
        with patch.object(rh, "_vm_query_range", side_effect=side_effect):
            self.assertTrue(await rh.get_data(days_list, [expression]))
        df = rh.df_final
        # Held through the last bucket of the day (23:30), not just at the 10:00 sample
        self.assertFalse(df[expression].isna().any())
        self.assertAlmostEqual(df.loc["2023-04-01 23:30:00+00:00"][expression], 1000.0)

    # Test publish data
    async def test_publish_data(self):
        response, data = await self.rh.post_data(
            self.df_raw[self.df_raw.columns[0]],
            10,
            "sensor.p_pv_forecast",
            "power",
            "Unit",
            "Variable",
            type_var="power",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            data["state"],
            f"{np.round(self.df_raw.loc[self.df_raw.index[10], self.df_raw.columns[0]], 2):.2f}",
        )
        self.assertEqual(data["attributes"]["unit_of_measurement"], "Unit")
        self.assertEqual(data["attributes"]["friendly_name"], "Variable")
        # Lets test publishing a forecast with more added attributes
        df = copy.deepcopy(self.df_raw.iloc[0:30])
        df.columns = ["P_Load", "P_PV", "p_pv_forecast"]
        df["P_batt"] = 1000.0
        df["SOC_opt"] = 0.5
        response, data = await self.rh.post_data(
            df["p_pv_forecast"],
            10,
            "sensor.p_pv_forecast",
            "power",
            "W",
            "PV Forecast",
            type_var="power",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["state"], f"{np.round(df.loc[df.index[10], df.columns[2]], 2):.2f}")
        self.assertEqual(data["attributes"]["unit_of_measurement"], "W")
        self.assertEqual(data["attributes"]["friendly_name"], "PV Forecast")
        self.assertIsInstance(data["attributes"]["forecasts"], list)
        response, data = await self.rh.post_data(
            df["P_batt"],
            25,
            "sensor.p_batt_forecast",
            "power",
            "W",
            "Battery Power Forecast",
            type_var="batt",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["attributes"]["unit_of_measurement"], "W")
        self.assertEqual(data["attributes"]["friendly_name"], "Battery Power Forecast")
        response, data = await self.rh.post_data(
            df["SOC_opt"],
            25,
            "sensor.SOC_forecast",
            "battery",
            "%",
            "Battery SOC Forecast",
            type_var="SOC",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["attributes"]["unit_of_measurement"], "%")
        self.assertEqual(data["attributes"]["friendly_name"], "Battery SOC Forecast")

    @patch("emhass.retrieve_hass.get_websocket_client")
    async def test_get_ha_config(self, mock_get_ws):
        # Test REST API success
        with aioresponses() as mocked:
            mocked.get(
                self.retrieve_hass_conf["hass_url"] + "api/config",
                payload={"time_zone": "Europe/Paris", "currency": "EUR"},
                status=200,
            )
            self.rh.use_websocket = False
            result = await self.rh.get_ha_config()
            self.assertTrue(result)
            self.assertEqual(self.rh.ha_config["time_zone"], "Europe/Paris")

        # Test REST API failure (401)
        with aioresponses() as mocked:
            mocked.get(
                self.retrieve_hass_conf["hass_url"] + "api/config",
                status=401,
            )
            result = await self.rh.get_ha_config()
            self.assertFalse(result)

        # Test WebSocket success
        self.rh.use_websocket = True
        mock_client = MagicMock()
        mock_client.get_config = AsyncMock(return_value={"time_zone": "Asia/Tokyo"})
        mock_get_ws.return_value = mock_client

        result = await self.rh.get_ha_config()
        self.assertEqual(result, {"time_zone": "Asia/Tokyo"})

        # Reset for other tests
        self.rh.use_websocket = False

    @patch("emhass.retrieve_hass.get_websocket_client", new_callable=AsyncMock)
    @patch("emhass.retrieve_hass.RetrieveHass._get_data_rest_api")
    async def test_get_data_websocket(self, mock_rest_fallback, mock_get_ws):
        # Setup common vars
        days_list = pd.date_range(start="2024-01-01", periods=2, freq="D", tz="UTC")
        var_list = ["sensor.power_load"]

        # Test Successful WebSocket Retrieval
        self.rh.use_websocket = True

        # Configure the mock client
        mock_client = MagicMock()
        mock_get_ws.return_value = mock_client

        # Mock statistics return data with ISO timestamp to ensure robust parsing
        start_iso = days_list[0].isoformat()
        mock_stats = {
            "sensor.power_load": [
                {"start": start_iso, "mean": 1000.0},
                # Add more data points to ensure valid resampling
                {"start": (days_list[0] + pd.Timedelta("30min")).isoformat(), "mean": 1500.0},
            ]
        }
        mock_client.get_statistics = AsyncMock(return_value=mock_stats)

        success = await self.rh.get_data_websocket(days_list, var_list)

        self.assertTrue(success, "get_data_websocket returned False")
        self.assertFalse(self.rh.df_final.empty, "Resulting DataFrame is empty")
        self.assertIn("sensor.power_load", self.rh.df_final.columns)

        # Test Connection Failure -> Fallback to REST
        mock_get_ws.side_effect = Exception("Connection refused")
        mock_rest_fallback.return_value = True  # Mock REST success

        success = await self.rh.get_data(days_list, var_list)

        self.assertTrue(success)
        mock_rest_fallback.assert_called_once()

        # Reset side effect
        mock_get_ws.side_effect = None
        self.rh.use_websocket = False

    # -- #1360: websocket statistics gaps for battery identification ----------------

    _WS_A, _WS_B = "sensor.ws_power_a", "sensor.ws_power_b"

    def _ws_gap_stats(self):
        """
        One day of 5-minute statistics for two sensors, as the HA websocket returns
        them. Both have no rows 02:00-04:00 UTC and sensor B alone has none
        10:00-11:00 UTC. Returns the statistics and, per sensor, the 30-minute
        buckets (UTC) that hold no statistics row at all.
        """
        idx = pd.date_range("2024-01-01", periods=288, freq="5min", tz="UTC")
        values = np.arange(288, dtype=float)
        stats = {}
        for entity_id, offset, holes in (
            (self._WS_A, 0.0, [("02:00", "04:00")]),
            (self._WS_B, 1000.0, [("02:00", "04:00"), ("10:00", "11:00")]),
        ):
            keep = np.ones(len(idx), dtype=bool)
            for start, end in holes:
                keep &= ~(
                    (idx >= pd.Timestamp(f"2024-01-01 {start}", tz="UTC"))
                    & (idx < pd.Timestamp(f"2024-01-01 {end}", tz="UTC"))
                )
            starts_ms = (idx[keep].as_unit("ns").asi8 // 1_000_000).tolist()
            stats[entity_id] = [
                {"start": ms, "mean": v + offset}
                for ms, v in zip(starts_ms, values[keep], strict=True)
            ]
        common = ["2024-01-01 02:00", "2024-01-01 02:30", "2024-01-01 03:00", "2024-01-01 03:30"]
        expected_nan = {
            self._WS_A: common,
            self._WS_B: common + ["2024-01-01 10:00", "2024-01-01 10:30"],
        }
        return stats, [self._WS_A, self._WS_B], expected_nan

    def _legacy_ws_frame(self, stats, var_list):
        """The websocket frame as built before #1360: resample, forward-fill, set freq."""
        df = None
        for entity_id in var_list:
            rows = stats[entity_id]
            entity_df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime([r["start"] for r in rows], unit="ms", utc=True),
                    entity_id: [float(r["mean"]) for r in rows],
                }
            ).set_index("timestamp")
            df = entity_df if df is None else df.join(entity_df, how="outer")
        df.index = df.index.tz_convert(self.rh.time_zone)
        df = df.sort_index().resample(self.rh.freq).mean().ffill()
        return utils.set_df_index_freq(df)

    def _assert_gaps_only_where_filled(self, gaps, filled, expected_nan):
        """gaps differs from the filled frame only by NaN in the buckets with no rows."""
        pd.testing.assert_index_equal(gaps.index, filled.index)
        pd.testing.assert_index_equal(gaps.columns, filled.columns)
        self.assertFalse(filled.isna().any().any(), "the filled frame should have no NaN")
        utc = gaps.index.tz_convert("UTC")
        for entity_id, buckets in expected_nan.items():
            nan_at = [ts.strftime("%Y-%m-%d %H:%M") for ts in utc[gaps[entity_id].isna()]]
            self.assertEqual(nan_at, buckets, entity_id)
        pd.testing.assert_frame_equal(gaps.ffill(), filled)

    async def _ws_get_data(self, stats, var_list, **kwargs):
        """get_data on the websocket path with a fake client; REST must not be reached."""
        days_list = pd.date_range(start="2024-01-01", periods=1, freq="D", tz="UTC")
        client = MagicMock()
        client.get_statistics = AsyncMock(return_value=stats)
        self.rh.use_websocket = True
        try:
            with (
                patch(
                    "emhass.retrieve_hass.get_websocket_client",
                    new=AsyncMock(return_value=client),
                ),
                patch.object(
                    RetrieveHass, "_get_data_rest_api", new=AsyncMock(return_value=False)
                ) as rest,
            ):
                success = await self.rh.get_data(days_list, var_list, **kwargs)
        finally:
            self.rh.use_websocket = False
        rest.assert_not_called()
        self.assertTrue(success)
        return self.rh.df_final.copy()

    def _no_forward_fill_kwargs(self):
        """forward_fill=False, only when the converter has the parameter (so the tests also run, red, on base)."""
        params = inspect.signature(self.rh._convert_statistics_to_dataframe).parameters
        return {"forward_fill": False} if "forward_fill" in params else {}

    def test_convert_statistics_without_forward_fill_keeps_gaps(self):
        """C1 (#1360): forward_fill=False leaves every bucket with no rows as NaN."""
        stats, var_list, expected_nan = self._ws_gap_stats()
        kwargs = self._no_forward_fill_kwargs()
        gaps = self.rh._convert_statistics_to_dataframe(stats, var_list, **kwargs)
        filled = self.rh._convert_statistics_to_dataframe(stats, var_list)
        self._assert_gaps_only_where_filled(gaps, filled, expected_nan)

    def test_convert_statistics_default_is_unchanged(self):
        """C2 (#1360): called as before, the converter still forward-fills, byte for byte."""
        stats, var_list, _ = self._ws_gap_stats()
        pd.testing.assert_frame_equal(
            self.rh._convert_statistics_to_dataframe(stats, var_list),
            self._legacy_ws_frame(stats, var_list),
        )

    def test_convert_statistics_without_forward_fill_short_time_step(self):
        """
        C3 (#1360): with a time step under 5 minutes, each statistics row still
        covers its own 5-minute period, so only the real holes stay NaN. At 5
        and 7 minutes nothing is held (the hold == 0 branch).
        """
        stats, var_list, _ = self._ws_gap_stats()
        holes = {
            self._WS_A: [("02:00", "04:00")],
            self._WS_B: [("02:00", "04:00"), ("10:00", "11:00")],
        }
        kwargs = self._no_forward_fill_kwargs()
        freq = self.rh.freq
        try:
            for minutes in (1, 2, 3, 4, 5, 7):
                with self.subTest(minutes=minutes):
                    self.rh.freq = pd.Timedelta(minutes=minutes)
                    gaps = self.rh._convert_statistics_to_dataframe(stats, var_list, **kwargs)
                    filled = self.rh._convert_statistics_to_dataframe(stats, var_list)
                    utc = gaps.index.tz_convert("UTC")
                    for entity_id, windows in holes.items():
                        in_hole = np.zeros(len(utc), dtype=bool)
                        for start, end in windows:
                            in_hole |= (utc >= pd.Timestamp(f"2024-01-01 {start}", tz="UTC")) & (
                                utc < pd.Timestamp(f"2024-01-01 {end}", tz="UTC")
                            )
                        is_nan = gaps[entity_id].isna().to_numpy()
                        if minutes == 7:
                            # A 7-minute bin can straddle the edge of a hole, so check
                            # against the rows: a bin is NaN exactly when no row starts in it.
                            rows = pd.to_datetime(
                                [r["start"] for r in stats[entity_id]], unit="ms", utc=True
                            )
                            nxt = np.minimum(rows.searchsorted(utc), len(rows) - 1)
                            has_row = (rows[nxt] >= utc) & (rows[nxt] < utc + self.rh.freq)
                            np.testing.assert_array_equal(is_nan, ~has_row, err_msg=entity_id)
                        else:
                            np.testing.assert_array_equal(is_nan, in_hole, err_msg=entity_id)
                        keep = ~in_hole & ~is_nan
                        np.testing.assert_array_equal(
                            gaps[entity_id].to_numpy()[keep], filled[entity_id].to_numpy()[keep]
                        )
        finally:
            self.rh.freq = freq

    # battery_id passes both flags (command_line._retrieve_from_hass).
    _BATTERY_ID_FLAGS = {"keep_partial_days": True, "keep_websocket_gaps": True}

    async def test_get_data_websocket_keep_websocket_gaps_keeps_gaps(self):
        """G1 (#1360): battery_id's keep_websocket_gaps=True reaches the websocket path."""
        stats, var_list, expected_nan = self._ws_gap_stats()
        gaps = await self._ws_get_data(stats, var_list, **self._BATTERY_ID_FLAGS)
        self._assert_gaps_only_where_filled(
            gaps, self._legacy_ws_frame(stats, var_list), expected_nan
        )

    async def test_get_data_websocket_default_is_unchanged(self):
        """G2 (#1360): every other websocket retrieval gets today's frame, byte for byte."""
        stats, var_list, _ = self._ws_gap_stats()
        df = await self._ws_get_data(stats, var_list)
        pd.testing.assert_frame_equal(df, self._legacy_ws_frame(stats, var_list))

    async def test_get_data_websocket_keep_websocket_gaps_does_not_leak(self):
        """G3 (#1360): the flag applies to its own call only, not to later calls."""
        stats, var_list, expected_nan = self._ws_gap_stats()
        legacy = self._legacy_ws_frame(stats, var_list)
        gaps = await self._ws_get_data(stats, var_list, **self._BATTERY_ID_FLAGS)
        filled = await self._ws_get_data(stats, var_list)
        pd.testing.assert_frame_equal(filled, legacy)
        self._assert_gaps_only_where_filled(gaps, legacy, expected_nan)

    async def test_get_data_websocket_keep_partial_days_alone_still_fills(self):
        """
        G4 (#1360): keep_partial_days=True on its own (thermal calibration sets it
        for REST reasons, #1380) leaves the websocket frame forward-filled, byte
        for byte as before.
        """
        stats, var_list, _ = self._ws_gap_stats()
        df = await self._ws_get_data(stats, var_list, keep_partial_days=True)
        pd.testing.assert_frame_equal(df, self._legacy_ws_frame(stats, var_list))

    async def test_thermal_calibration_websocket_history_still_forward_fills(self):
        """
        T1 (#1360): _load_thermal_from_ha (#1380) also calls get_data with
        keep_partial_days=True. Over the websocket it must still get the filled
        frame: a temperature gap keeps its rows at the last value and a heater
        power gap holds the last value, not 0 W.
        """
        stats2, _, _ = self._ws_gap_stats()
        t_in, t_out, power = "sensor.ws_t_in", "sensor.ws_t_out", "sensor.ws_heater_power"
        # Indoor and outdoor temperature miss 02:00-04:00, heater power misses
        # 02:00-04:00 and 10:00-11:00.
        stats = {t_in: stats2[self._WS_A], t_out: stats2[self._WS_A], power: stats2[self._WS_B]}
        var_list = [t_in, t_out, power]
        start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        end = pd.Timestamp("2024-01-01 23:59", tz="UTC")
        cfg = {
            "dt_minutes": 30,
            "dt_hours": 0.5,
            "has_explicit_window": True,
            "start_utc": start,
            "end_utc": end,
            "lookback_hours": 24,
            "entity_t_in": t_in,
            "entity_t_out": t_out,
            "entity_power": power,
        }
        rh_conf = {
            "hass_url": "http://ha.invalid:8123/",
            "long_lived_token": "token",
            "time_zone": self.rh.time_zone,
            "use_websocket": True,
        }
        client = MagicMock()
        client.get_statistics = AsyncMock(return_value=stats)
        with (
            patch("emhass.retrieve_hass.get_websocket_client", new=AsyncMock(return_value=client)),
            patch.object(
                RetrieveHass, "_get_data_rest_api", new=AsyncMock(return_value=False)
            ) as rest,
        ):
            df = await utils._load_thermal_from_ha(emhass_conf, {}, rh_conf, cfg, logger)
        rest.assert_not_called()

        legacy = self._legacy_ws_frame(stats, var_list)
        expected = pd.DataFrame(
            {
                "T_in": legacy[t_in],
                "T_out": legacy[t_out],
                "P_def": legacy[power].clip(lower=0.0),
            },
            index=legacy.index,
        )
        utc = expected.index.tz_convert("UTC")
        expected = expected.loc[(utc >= start) & (utc <= end)]
        self.assertEqual(len(df), 48, "rows in a websocket gap were dropped")
        pd.testing.assert_frame_equal(df, expected, check_freq=False)
        # The heater power gap is held at the last value, not read as 0 W.
        held = df.loc[df.index.tz_convert("UTC") == pd.Timestamp("2024-01-01 10:00", tz="UTC")]
        self.assertGreater(held["P_def"].iloc[0], 0.0)

    async def test_retrieve_from_hass_keep_websocket_gaps_plumbing(self):
        """G5 (#1360): _retrieve_from_hass sets keep_websocket_gaps for battery_id only."""
        retrieve_hass_conf = dict(self.retrieve_hass_conf)
        retrieve_hass_conf["historic_days_to_retrieve"] = 1
        retrieve_hass_conf["sensor_power_battery"] = "sensor.power_battery"
        retrieve_hass_conf["sensor_battery_state_of_charge"] = "sensor.battery_soc"
        mock_rh = MagicMock()
        mock_rh.get_data = AsyncMock(return_value=True)
        for set_type, expected in (
            ("battery_id", True),
            ("perfect-optim", False),
            ("naive-mpc-optim", False),
        ):
            with self.subTest(set_type=set_type):
                mock_rh.get_data.reset_mock()
                await _retrieve_from_hass(
                    set_type, retrieve_hass_conf, {"set_use_pv": False}, mock_rh, logger
                )
                _, call_kwargs = mock_rh.get_data.call_args
                self.assertIs(call_kwargs.get("keep_websocket_gaps", False), expected)

    async def test_get_data_rest_api_errors(self):
        days_list = pd.date_range(start="2024-01-01", periods=1, freq="D", tz="UTC")
        var_list = ["sensor.test"]
        url = (
            self.retrieve_hass_conf["hass_url"]
            + "api/history/period/"
            + days_list[0].isoformat()
            + "?filter_entity_id=sensor.test"
        )

        # Test Connection Error (Exception)
        with aioresponses() as mocked:
            mocked.get(url, exception=Exception("Network down"))
            result = await self.rh._get_data_rest_api(days_list, var_list)
            self.assertFalse(result)

        # Test 401 Unauthorized
        with aioresponses() as mocked:
            mocked.get(url, status=401)
            result = await self.rh._get_data_rest_api(days_list, var_list)
            self.assertFalse(result)

        # Test Empty JSON Response — all days empty should still fail
        with aioresponses() as mocked:
            mocked.get(url, payload=[], status=200)
            result = await self.rh._get_data_rest_api(days_list, var_list)
            self.assertFalse(result)

    async def test_get_data_rest_api_skips_empty_days(self):
        """Test that days with no data are skipped gracefully instead of aborting."""
        # Create a 3-day range where day 1 returns empty, days 2–3 return data
        days_list = pd.date_range(start="2024-01-01", periods=3, freq="D", tz="UTC")
        var_list = ["sensor.test_power"]

        # Build URLs for each day
        base_url = self.retrieve_hass_conf["hass_url"] + "api/history/period/"
        urls = [
            base_url + day.strftime("%Y-%m-%dT%H:%M:%SZ") + "?filter_entity_id=sensor.test_power"
            for day in days_list
        ]

        # Build mock data: 96 records per day at 15-min intervals
        def make_day_data(date_str, entity_id="sensor.test_power"):
            records = []
            base = pd.Timestamp(date_str, tz="UTC")
            for i in range(96):
                ts = base + pd.Timedelta(minutes=15 * i)
                records.append(
                    {
                        "entity_id": entity_id,
                        "state": str(100.0 + i),
                        "attributes": {},
                        "last_changed": ts.isoformat(),
                        "last_updated": ts.isoformat(),
                    }
                )
            return [records]  # HA wraps per-entity in a list

        with aioresponses() as mocked:
            # Day 1: empty (sensor didn't exist yet)
            mocked.get(urls[0], payload=[], status=200)
            # Day 2: has data
            mocked.get(urls[1], payload=make_day_data("2024-01-02"), status=200)
            # Day 3: has data
            mocked.get(urls[2], payload=make_day_data("2024-01-03"), status=200)

            result = await self.rh._get_data_rest_api(days_list, var_list)
            # Should succeed with partial data
            self.assertTrue(result)
            # Should have data from 2 days, not 3
            self.assertIsInstance(self.rh.df_final, pd.DataFrame)
            self.assertFalse(self.rh.df_final.empty)
            # First data point should be from day 2, not day 1
            self.assertGreaterEqual(self.rh.df_final.index[0], pd.Timestamp("2024-01-02", tz="UTC"))

    async def test_get_data_rest_api_all_days_empty_fails(self):
        """Test that if ALL days return empty, it still fails with an error."""
        days_list = pd.date_range(start="2024-01-01", periods=3, freq="D", tz="UTC")
        var_list = ["sensor.test_power"]

        base_url = self.retrieve_hass_conf["hass_url"] + "api/history/period/"
        urls = [
            base_url + day.strftime("%Y-%m-%dT%H:%M:%SZ") + "?filter_entity_id=sensor.test_power"
            for day in days_list
        ]

        with aioresponses() as mocked:
            for url in urls:
                mocked.get(url, payload=[], status=200)

            result = await self.rh._get_data_rest_api(days_list, var_list)
            # All days empty → should fail
            self.assertFalse(result)

    async def test_get_data_rest_api_error_still_aborts(self):
        """Test that real errors (HTTP 401, network) still abort immediately."""
        days_list = pd.date_range(start="2024-01-01", periods=3, freq="D", tz="UTC")
        var_list = ["sensor.test_power"]

        base_url = self.retrieve_hass_conf["hass_url"] + "api/history/period/"
        urls = [
            base_url + day.strftime("%Y-%m-%dT%H:%M:%SZ") + "?filter_entity_id=sensor.test_power"
            for day in days_list
        ]

        # Day 1: empty (skip), Day 2: auth error (abort)
        with aioresponses() as mocked:
            mocked.get(urls[0], payload=[], status=200)
            mocked.get(urls[1], status=401)

            result = await self.rh._get_data_rest_api(days_list, var_list)
            self.assertFalse(result)

    @staticmethod
    def _make_partial_day_data(date_str, entity_id, num_records=96, minutes_step=15):
        """Build HA history payload records for one (day, sensor) pair, #1061 tests."""
        records = []
        base = pd.Timestamp(date_str, tz="UTC")
        for i in range(num_records):
            ts = base + pd.Timedelta(minutes=minutes_step * i)
            records.append(
                {
                    "entity_id": entity_id,
                    "state": str(100.0 + i),
                    "attributes": {},
                    "last_changed": ts.isoformat(),
                    "last_updated": ts.isoformat(),
                }
            )
        return [records]  # HA wraps per-entity in a list

    def _partial_day_url(self, day, var):
        base_url = self.retrieve_hass_conf["hass_url"] + "api/history/period/"
        return base_url + day.strftime("%Y-%m-%dT%H:%M:%SZ") + f"?filter_entity_id={var}"

    async def test_get_data_rest_api_keep_partial_days_keeps_day(self):
        """T1 (#1061): with keep_partial_days on, a day with SOME variables present
        is kept instead of dropped; the missing variable shows up as NaN for that
        day rather than removing every variable's data for it."""
        days_list = pd.date_range(start="2024-01-01", periods=2, freq="D", tz="UTC")
        var_a = "sensor.var_a"
        var_b = "sensor.var_b"
        var_list = [var_a, var_b]

        kwargs = {}
        if "keep_partial_days" in inspect.signature(self.rh._get_data_rest_api).parameters:
            kwargs["keep_partial_days"] = True

        with aioresponses() as mocked:
            # Day 1: var_a has data, var_b has none for this day
            mocked.get(
                self._partial_day_url(days_list[0], var_a),
                payload=self._make_partial_day_data("2024-01-01", var_a),
                status=200,
            )
            mocked.get(self._partial_day_url(days_list[0], var_b), payload=[], status=200)
            # Day 2: both variables have data
            mocked.get(
                self._partial_day_url(days_list[1], var_a),
                payload=self._make_partial_day_data("2024-01-02", var_a),
                status=200,
            )
            mocked.get(
                self._partial_day_url(days_list[1], var_b),
                payload=self._make_partial_day_data("2024-01-02", var_b),
                status=200,
            )

            result = await self.rh._get_data_rest_api(days_list, var_list, **kwargs)

        self.assertTrue(result)
        df = self.rh.df_final
        # Boolean masks rather than .loc partial-string lookups: on base (kwarg
        # stripped) day 1 is dropped entirely and .loc would raise KeyError,
        # masking the intended behavioural RED failure on the assertion below.
        day1 = df[df.index.normalize() == pd.Timestamp("2024-01-01", tz="UTC")]
        day2 = df[df.index.normalize() == pd.Timestamp("2024-01-02", tz="UTC")]
        self.assertFalse(day1.empty, "day 1 rows should be present when keep_partial_days is on")
        self.assertFalse(day1[var_a].isna().all(), "var_a should have real values on day 1")
        self.assertTrue(day1[var_b].isna().all(), "var_b should be all-NaN on day 1 (missing)")
        self.assertFalse(day2[var_b].isna().all(), "var_b should have real values on day 2")

    async def test_get_data_rest_api_default_still_skips_partial_days(self):
        """T2 (#1061): no-op proof. With the flag not passed at all, default
        behaviour is unchanged: a day missing any variable is dropped whole, and
        there is no NaN stretch left behind. Must pass on base and patched."""
        days_list = pd.date_range(start="2024-01-01", periods=2, freq="D", tz="UTC")
        var_a = "sensor.var_a"
        var_b = "sensor.var_b"
        var_list = [var_a, var_b]

        with aioresponses() as mocked:
            mocked.get(
                self._partial_day_url(days_list[0], var_a),
                payload=self._make_partial_day_data("2024-01-01", var_a),
                status=200,
            )
            mocked.get(self._partial_day_url(days_list[0], var_b), payload=[], status=200)
            mocked.get(
                self._partial_day_url(days_list[1], var_a),
                payload=self._make_partial_day_data("2024-01-02", var_a),
                status=200,
            )
            mocked.get(
                self._partial_day_url(days_list[1], var_b),
                payload=self._make_partial_day_data("2024-01-02", var_b),
                status=200,
            )

            result = await self.rh._get_data_rest_api(days_list, var_list)

        self.assertTrue(result)
        df = self.rh.df_final
        # Day 1 fully dropped: index starts on day 2, no all-NaN stretch anywhere
        self.assertGreaterEqual(df.index[0], pd.Timestamp("2024-01-02", tz="UTC"))
        self.assertFalse(df[var_a].isna().all())
        self.assertFalse(df[var_b].isna().all())
        self.assertEqual(df[var_a].isna().sum(), 0)
        self.assertEqual(df[var_b].isna().sum(), 0)

    async def test_get_data_rest_api_keep_partial_days_all_missing_still_skipped(self):
        """T3 (#1061): with the flag on, a day is only kept if at least one
        variable has data; a day where every variable is missing is still
        skipped, same as default mode."""
        days_list = pd.date_range(start="2024-01-01", periods=2, freq="D", tz="UTC")
        var_a = "sensor.var_a"
        var_b = "sensor.var_b"
        var_list = [var_a, var_b]

        kwargs = {}
        if "keep_partial_days" in inspect.signature(self.rh._get_data_rest_api).parameters:
            kwargs["keep_partial_days"] = True

        with aioresponses() as mocked:
            # Day 1: both variables missing
            mocked.get(self._partial_day_url(days_list[0], var_a), payload=[], status=200)
            mocked.get(self._partial_day_url(days_list[0], var_b), payload=[], status=200)
            # Day 2: both variables have data
            mocked.get(
                self._partial_day_url(days_list[1], var_a),
                payload=self._make_partial_day_data("2024-01-02", var_a),
                status=200,
            )
            mocked.get(
                self._partial_day_url(days_list[1], var_b),
                payload=self._make_partial_day_data("2024-01-02", var_b),
                status=200,
            )

            result = await self.rh._get_data_rest_api(days_list, var_list, **kwargs)

        self.assertTrue(result)
        df = self.rh.df_final
        self.assertGreaterEqual(df.index[0], pd.Timestamp("2024-01-02", tz="UTC"))

    async def test_get_data_rest_api_keep_partial_days_partial_last_day_no_fabricated_tail(self):
        """T4 (#1061): partial last day (the reporter's-sketch failure mode). One
        variable has data only for the morning of the last day, the other is
        missing entirely for that day. With the flag on the day is kept, but the
        index must end at the last real timestamp, never a fabricated full-day
        tail."""
        days_list = pd.date_range(start="2024-01-01", periods=2, freq="D", tz="UTC")
        var_a = "sensor.var_a"
        var_b = "sensor.var_b"
        var_list = [var_a, var_b]

        kwargs = {}
        if "keep_partial_days" in inspect.signature(self.rh._get_data_rest_api).parameters:
            kwargs["keep_partial_days"] = True

        # Day 2 (last day): var_a only reports for the first 12 hours (48 records
        # at 15-minute steps), var_b has no data at all for that day.
        last_day_records = 48
        last_ts = pd.Timestamp("2024-01-02", tz="UTC") + pd.Timedelta(
            minutes=15 * (last_day_records - 1)
        )

        with aioresponses() as mocked:
            mocked.get(
                self._partial_day_url(days_list[0], var_a),
                payload=self._make_partial_day_data("2024-01-01", var_a),
                status=200,
            )
            mocked.get(
                self._partial_day_url(days_list[0], var_b),
                payload=self._make_partial_day_data("2024-01-01", var_b),
                status=200,
            )
            mocked.get(
                self._partial_day_url(days_list[1], var_a),
                payload=self._make_partial_day_data(
                    "2024-01-02", var_a, num_records=last_day_records
                ),
                status=200,
            )
            mocked.get(self._partial_day_url(days_list[1], var_b), payload=[], status=200)

            result = await self.rh._get_data_rest_api(days_list, var_list, **kwargs)

        self.assertTrue(result)
        df = self.rh.df_final
        # No fabricated rows past the last real sample on the last day
        self.assertLessEqual(df.index[-1], last_ts)
        self.assertEqual(df.index[-1].date(), last_ts.date())

    async def test_retrieve_from_hass_keep_partial_days_plumbing(self):
        """T5 (#1061): _retrieve_from_hass threads keep_partial_days=True only
        for battery_id, leaving the other set_types at the historical default."""
        from emhass.command_line import _retrieve_from_hass

        retrieve_hass_conf = dict(self.retrieve_hass_conf)
        retrieve_hass_conf["historic_days_to_retrieve"] = 1
        retrieve_hass_conf["sensor_power_battery"] = "sensor.power_battery"
        retrieve_hass_conf["sensor_battery_state_of_charge"] = "sensor.battery_soc"

        optim_conf = {"set_use_pv": False}

        mock_rh = MagicMock()
        mock_rh.get_data = AsyncMock(return_value=True)

        await _retrieve_from_hass("battery_id", retrieve_hass_conf, optim_conf, mock_rh, logger)
        _, call_kwargs = mock_rh.get_data.call_args
        self.assertTrue(call_kwargs.get("keep_partial_days"))

        mock_rh.get_data.reset_mock()
        await _retrieve_from_hass("perfect-optim", retrieve_hass_conf, optim_conf, mock_rh, logger)
        _, call_kwargs = mock_rh.get_data.call_args
        self.assertFalse(call_kwargs.get("keep_partial_days", False))

    @patch("emhass.retrieve_hass.get_websocket_client", new_callable=AsyncMock)
    @patch("emhass.retrieve_hass.RetrieveHass._get_data_rest_api")
    async def test_get_data_websocket_fallback_threads_keep_partial_days(
        self, mock_rest_fallback, mock_get_ws
    ):
        """T6 (#1061): when the websocket path fails and falls back to the REST
        API, keep_partial_days reaches that fallback call too."""
        days_list = pd.date_range(start="2024-01-01", periods=2, freq="D", tz="UTC")
        var_list = ["sensor.power_load"]

        self.rh.use_websocket = True
        mock_get_ws.side_effect = Exception("Connection refused")
        mock_rest_fallback.return_value = True

        if "keep_partial_days" not in inspect.signature(self.rh.get_data).parameters:
            self.skipTest("keep_partial_days not present on this source")

        success = await self.rh.get_data(days_list, var_list, keep_partial_days=True)

        self.assertTrue(success)
        mock_rest_fallback.assert_called_once()
        self.assertTrue(mock_rest_fallback.call_args.kwargs.get("keep_partial_days"))

        # Reset for other tests
        mock_get_ws.side_effect = None
        self.rh.use_websocket = False

    # --- #1073: staleness aux series for battery identification ----------------
    @staticmethod
    def _ha_rows(entity_id, rows):
        """HA history payload for one (day, sensor) from (iso timestamp, state) pairs."""
        return [
            [
                {
                    "entity_id": entity_id,
                    "state": state,
                    "attributes": {},
                    "last_changed": ts,
                    "last_updated": ts,
                }
                for ts, state in rows
            ]
        ]

    def _staleness_payloads(self):
        """Two days of a battery power sensor plus a SoC sensor created mid-day.

        Day 1 power: the start-of-day row HA carries into the period (00:00),
        a real reading at 00:40, a reading then an ``unavailable`` row in the
        01:00 bucket, nothing in 01:30, readings at 02:10 and 23:40.
        Day 2 power: the carried row is ``unavailable``, a real reading at 00:50.
        SoC first reports at 05:10 on day 1 (no carried row: it did not exist),
        then once on day 2 at 00:00 exactly (a real change stamped at the start,
        which HA cannot tell apart from the carried row) and at 00:40.
        """
        p, s = "sensor.batt_power", "sensor.batt_soc"
        return (
            p,
            s,
            {
                ("2024-01-01", p): self._ha_rows(
                    p,
                    [
                        ("2024-01-01T00:00:00+00:00", "500"),
                        ("2024-01-01T00:40:00+00:00", "600"),
                        ("2024-01-01T01:05:00+00:00", "700"),
                        ("2024-01-01T01:20:00+00:00", "unavailable"),
                        ("2024-01-01T02:10:00+00:00", "800"),
                        ("2024-01-01T23:40:00+00:00", "1000"),
                    ],
                ),
                ("2024-01-02", p): self._ha_rows(
                    p,
                    [
                        ("2024-01-02T00:00:00+00:00", "unavailable"),
                        ("2024-01-02T00:50:00+00:00", "1100"),
                    ],
                ),
                ("2024-01-01", s): self._ha_rows(
                    s,
                    [("2024-01-01T05:10:00+00:00", "40"), ("2024-01-01T23:50:00+00:00", "45")],
                ),
                ("2024-01-02", s): self._ha_rows(
                    s,
                    [("2024-01-02T00:00:00+00:00", "46"), ("2024-01-02T00:40:00+00:00", "47")],
                ),
            },
        )

    async def _rest_with_payloads(self, payloads, **kwargs):
        days_list = pd.date_range(start="2024-01-01", periods=2, freq="D", tz="UTC")
        var_list = sorted({var for _, var in payloads})
        with aioresponses() as mocked:
            for (date, var), payload in payloads.items():
                mocked.get(
                    self._partial_day_url(pd.Timestamp(date, tz="UTC"), var),
                    payload=payload,
                    status=200,
                    repeat=True,
                )
            ok = await self.rh.get_data(
                days_list,
                var_list,
                minimal_response=False,
                significant_changes_only=False,
                **kwargs,
            )
        self.assertTrue(ok)
        return self.rh.df_final.copy()

    @staticmethod
    def _staleness_kwargs(rh):
        """The get_data keywords for a battery_id style retrieval."""
        return {"keep_partial_days": True, "record_staleness": True}

    async def test_record_staleness_records_last_real_state_and_marker_buckets(self):
        """Only the flagged retrieval records anything."""
        p, s, payloads = self._staleness_payloads()
        await self._rest_with_payloads(payloads, **self._staleness_kwargs(self.rh))
        last = getattr(self.rh, "bucket_last_state", {})
        markers = getattr(self.rh, "marker_buckets", {})
        self.assertIn(p, last)

        def t(hhmm, day="2024-01-01"):
            return pd.Timestamp(f"{day} {hhmm}", tz="UTC")

        lp = last[p]
        # The carried start-of-day row is no real reading: its bucket has none.
        self.assertTrue(np.isnan(lp[t("00:00")]))
        self.assertEqual(lp[t("00:30")], 600.0)
        # Last REAL state of a bucket whose last row is a marker.
        self.assertEqual(lp[t("01:00")], 700.0)
        self.assertTrue(np.isnan(lp[t("01:30")]))
        self.assertEqual(lp[t("02:00")], 800.0)
        self.assertEqual(lp[t("23:30")], 1000.0)
        self.assertTrue(np.isnan(lp[t("00:00", "2024-01-02")]))
        self.assertEqual(lp[t("00:30", "2024-01-02")], 1100.0)
        # The marker rows' buckets, the carried unavailable row included, and the
        # observed 01:00 bucket that also holds one.
        self.assertEqual(list(markers[p]), [t("01:00"), t("00:00", "2024-01-02")])
        # SoC: a first row not stamped at the day start is a real reading; the
        # first row stamped exactly at the start is treated as carried.
        ls = last[s]
        self.assertEqual(ls[t("05:00")], 40.0)
        self.assertTrue(np.isnan(ls[t("00:00", "2024-01-02")]))
        self.assertEqual(len(markers[s]), 0)
        # Real readings per bucket: the carried row and marker rows do not count.
        counts = getattr(self.rh, "bucket_row_count", {})
        self.assertIn(p, counts)
        cp = counts[p]
        self.assertTrue(cp.index.equals(lp.index))
        self.assertEqual(cp[t("00:00")], 0)
        self.assertEqual(cp[t("00:30")], 1)
        self.assertEqual(cp[t("01:00")], 1)
        self.assertEqual(cp[t("01:30")], 0)
        self.assertEqual(cp[t("00:00", "2024-01-02")], 0)
        self.assertEqual(cp[t("00:30", "2024-01-02")], 1)
        self.assertEqual(int(counts[s].sum()), 3)
        # The aux sits on df_final's grid (UTC), like the frame it describes.
        for series in last.values():
            self.assertTrue(series.index.isin(self.rh.df_final.index).all())

    async def test_record_staleness_counts_each_reading_once(self):
        """Several readings in a
        bucket are counted, rows sharing a timestamp (an attribute-only update)
        once, and neither the carried start-of-day row nor a marker row counts."""
        p = "sensor.batt_power"
        payloads = {
            ("2024-01-01", p): self._ha_rows(
                p,
                [
                    ("2024-01-01T00:00:00+00:00", "500"),
                    ("2024-01-01T00:05:00+00:00", "600"),
                    ("2024-01-01T00:10:00+00:00", "650"),
                    ("2024-01-01T00:10:00+00:00", "650"),
                    ("2024-01-01T00:20:00+00:00", "unavailable"),
                    ("2024-01-01T00:40:00+00:00", "700"),
                    ("2024-01-01T23:40:00+00:00", "800"),
                ],
            ),
            ("2024-01-02", p): self._ha_rows(
                p,
                [
                    ("2024-01-02T00:00:00+00:00", "800"),
                    ("2024-01-02T00:35:00+00:00", "900"),
                    ("2024-01-02T23:40:00+00:00", "950"),
                ],
            ),
        }
        await self._rest_with_payloads(payloads, **self._staleness_kwargs(self.rh))
        counts = getattr(self.rh, "bucket_row_count", {}).get(p)
        self.assertIsNotNone(counts)
        self.assertEqual(counts[pd.Timestamp("2024-01-01 00:00", tz="UTC")], 2)
        self.assertEqual(counts[pd.Timestamp("2024-01-01 00:30", tz="UTC")], 1)
        self.assertEqual(counts[pd.Timestamp("2024-01-02 00:00", tz="UTC")], 0)
        self.assertEqual(counts[pd.Timestamp("2024-01-02 00:30", tz="UTC")], 1)
        self.assertEqual(int(counts.sum()), 6)

    async def test_record_staleness_keeps_aux_on_df_final_grid_for_non_divisor_step(self):
        """Regression: with a step that does not divide a day (7 min), the aux must
        keep to the labels df_final holds; when day 2 sat off day 1's grid (before
        #1381 put every day on the epoch grid), the fill rejected the whole series,
        day 1's eligible silence included."""
        from emhass.battery_identification import apply_staleness_fill

        p, s, load = "sensor.batt_power", "sensor.batt_soc", "sensor.load"

        def rows(day, entries):
            return [(f"{day}T{hhmm}:00+00:00", v) for hhmm, v in entries]

        # Day 1 on the epoch 7-min grid (00:01, 00:08, ...): power reads twice in
        # buckets 00:01, 00:08, 00:29 and 00:36 and is silent in 00:15 and 00:22;
        # SoC and the load read in every bucket, so the silence is witnessed.
        d1_power = [("00:01", "100"), ("00:03", "110"), ("00:08", "120"), ("00:10", "130")]
        d1_power += [("00:29", "200"), ("00:31", "210"), ("00:36", "220"), ("00:38", "230")]
        d1_soc = [(f"00:{m:02d}", str(50 + m)) for m in (1, 3, 8, 10, 15, 17, 22, 24, 29, 31)]
        d1_soc += [("00:36", "90"), ("00:38", "91")]
        d1_load = [(f"00:{m:02d}", "500") for m in (2, 9, 16, 23, 30, 37)]
        payloads = {
            ("2024-01-01", p): self._ha_rows(p, rows("2024-01-01", d1_power)),
            ("2024-01-01", s): self._ha_rows(s, rows("2024-01-01", d1_soc)),
            ("2024-01-01", load): self._ha_rows(load, rows("2024-01-01", d1_load)),
        }
        for var, value in ((p, "300"), (s, "60"), (load, "400")):
            payloads[("2024-01-02", var)] = self._ha_rows(
                var, rows("2024-01-02", [("00:20", value), ("00:40", value)])
            )
        self.rh.freq = pd.Timedelta(minutes=7)
        await self._rest_with_payloads(payloads, **self._staleness_kwargs(self.rh))
        final = self.rh.df_final
        self.assertEqual(final.index.freq, pd.Timedelta(minutes=7))
        for aux in (self.rh.bucket_last_state, self.rh.bucket_row_count):
            for var in (p, s, load):
                self.assertTrue(aux[var].index.isin(final.index).all(), var)
        for markers in self.rh.marker_buckets.values():
            self.assertTrue(markers.isin(final.index).all())
        # Day 1's silence (buckets 00:15 and 00:22) is held at 130 W.
        out = apply_staleness_fill(
            final,
            p,
            s,
            self.rh.bucket_last_state,
            self.rh.marker_buckets,
            self.rh.freq,
            row_counts=self.rh.bucket_row_count,
        )
        t0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        held = out.loc[[t0 + pd.Timedelta(minutes=15), t0 + pd.Timedelta(minutes=22)], p]
        self.assertEqual(list(held), [130.0, 130.0])

    async def test_record_staleness_does_not_change_df_final(self):
        """The flag only records aux: df_final is identical with and without it,
        and a call without it leaves both attributes empty."""
        _, _, payloads = self._staleness_payloads()
        with_flag = await self._rest_with_payloads(payloads, **self._staleness_kwargs(self.rh))
        without = await self._rest_with_payloads(payloads, keep_partial_days=True)
        pd.testing.assert_frame_equal(with_flag, without)
        self.assertEqual(getattr(self.rh, "bucket_last_state", {}), {})
        self.assertEqual(getattr(self.rh, "bucket_row_count", {}), {})
        self.assertEqual(getattr(self.rh, "marker_buckets", {}), {})

    async def test_default_retrieval_byte_identical_with_and_without_record_staleness(self):
        """Default-is-a-true-no-op: a non-battery_id retrieval (no keep_partial_days,
        no record_staleness) gets the same df_final as the same retrieval with
        record_staleness on."""
        _, _, payloads = self._staleness_payloads()
        default = await self._rest_with_payloads(payloads)
        flagged = await self._rest_with_payloads(payloads, record_staleness=True)
        self.assertTrue(default.equals(flagged))

    async def test_staleness_aux_reset_on_every_get_data(self):
        """An earlier REST call's aux never leaks into a later retrieval through
        another backend (websocket here), so the consumer is a no-op there."""
        _, _, payloads = self._staleness_payloads()
        await self._rest_with_payloads(payloads, **self._staleness_kwargs(self.rh))
        self.rh.use_websocket = True
        try:
            with patch.object(self.rh, "get_data_websocket", AsyncMock(return_value=True)):
                ok = await self.rh.get_data(
                    pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"),
                    ["sensor.batt_power"],
                    **self._staleness_kwargs(self.rh),
                )
        finally:
            self.rh.use_websocket = False
        self.assertTrue(ok)
        self.assertEqual(getattr(self.rh, "bucket_last_state", {}), {})
        self.assertEqual(getattr(self.rh, "bucket_row_count", {}), {})
        self.assertEqual(getattr(self.rh, "marker_buckets", {}), {})

    async def test_retrieve_from_hass_record_staleness_plumbing(self):
        """_retrieve_from_hass asks for the aux only for battery_id."""
        from emhass.command_line import _retrieve_from_hass

        retrieve_hass_conf = dict(self.retrieve_hass_conf)
        retrieve_hass_conf["historic_days_to_retrieve"] = 1
        retrieve_hass_conf["sensor_power_battery"] = "sensor.power_battery"
        retrieve_hass_conf["sensor_battery_state_of_charge"] = "sensor.battery_soc"
        mock_rh = MagicMock()
        mock_rh.get_data = AsyncMock(return_value=True)

        await _retrieve_from_hass("battery_id", retrieve_hass_conf, {}, mock_rh, logger)
        self.assertTrue(mock_rh.get_data.call_args.kwargs.get("record_staleness"))
        for set_type in ("perfect-optim", "naive-mpc-optim", "adjust_pv", "dayahead-optim"):
            mock_rh.get_data.reset_mock()
            await _retrieve_from_hass(set_type, retrieve_hass_conf, {}, mock_rh, logger)
            self.assertFalse(mock_rh.get_data.call_args.kwargs.get("record_staleness", False))

    @patch("emhass.retrieve_hass.get_websocket_client", new_callable=AsyncMock)
    @patch("emhass.retrieve_hass.RetrieveHass._get_data_rest_api")
    async def test_get_data_websocket_fallback_threads_record_staleness(
        self, mock_rest_fallback, mock_get_ws
    ):
        """When the websocket path fails and falls back to REST, record_staleness
        reaches that fallback call too (like keep_partial_days)."""
        days_list = pd.date_range(start="2024-01-01", periods=2, freq="D", tz="UTC")
        self.rh.use_websocket = True
        mock_get_ws.side_effect = Exception("Connection refused")
        mock_rest_fallback.return_value = True
        try:
            success = await self.rh.get_data(
                days_list, ["sensor.batt_power"], **self._staleness_kwargs(self.rh)
            )
        finally:
            self.rh.use_websocket = False
        self.assertTrue(success)
        mock_rest_fallback.assert_called_once()
        self.assertTrue(mock_rest_fallback.call_args.kwargs.get("record_staleness"))

    async def test_websocket_retrieval_records_no_staleness_aux(self):
        """#1360 boundary: the websocket statistics path has fixed-cadence rows and
        no raw states, so it records no aux and the battery_id fill stays a no-op
        there (its forward fill is #1360's subject, unchanged here)."""
        self.rh.use_websocket = True
        try:
            with patch.object(self.rh, "get_data_websocket", AsyncMock(return_value=True)):
                ok = await self.rh.get_data(
                    pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"),
                    ["sensor.batt_power", "sensor.batt_soc"],
                    **self._staleness_kwargs(self.rh),
                )
        finally:
            self.rh.use_websocket = False
        self.assertTrue(ok)
        self.assertEqual(getattr(self.rh, "bucket_last_state", None), {})
        self.assertEqual(getattr(self.rh, "bucket_row_count", None), {})
        self.assertEqual(getattr(self.rh, "marker_buckets", None), {})

    @patch("aiofiles.open")
    async def test_post_data_extended(self, mock_aio_open):
        self.rh.get_data_from_file = False
        # Every POST goes to a controlled response: four successes, then one 503 below
        mocked = self.enterContext(aioresponses())
        states_url = re.compile(r".*/states/.*")
        for _ in range(4):
            mocked.post(states_url, status=200)

        # Setup mock file context for save_entities=True
        mock_f = AsyncMock()
        mock_aio_open.return_value.__aenter__.return_value = mock_f

        # Create dummy data
        idx = 0
        entity_id = "sensor.p_pv_forecast"
        data_df = pd.Series(
            [100.55, 200.00], index=pd.date_range("2024-01-01", periods=2, freq="30min")
        )
        data_df.name = "test_data"

        # Test "cost_fun" type
        response, data = await self.rh.post_data(
            data_df, idx, entity_id, "monetary", "EUR", "Cost Function", "cost_fun"
        )
        self.assertEqual(data["state"], f"{data_df.sum():.2f}")
        self.assertTrue(response.ok)

        # Test "optim_status" type
        status_df = pd.Series(["Optimal"], index=[0])
        response, data = await self.rh.post_data(
            status_df, 0, "sensor.optim_status", "none", "", "Status", "optim_status"
        )
        self.assertEqual(data["state"], "Optimal")
        self.assertTrue(response.ok)

        # Test "deferrable" type (complex attributes)
        response, data = await self.rh.post_data(
            data_df, idx, entity_id, "power", "W", "Deferrable", "deferrable"
        )
        self.assertIn("deferrables_schedule", data["attributes"])
        self.assertTrue(response.ok)

        # Test "unit_load_cost" (4 decimals)
        response, data = await self.rh.post_data(
            data_df, idx, entity_id, "monetary", "EUR/kWh", "Load Cost", "unit_load_cost"
        )
        self.assertEqual(data["state"], f"{data_df.iloc[0]:.4f}")
        self.assertIn("unit_load_cost_forecasts", data["attributes"])
        self.assertTrue(response.ok)

        # Test save_entities=True
        # Save old path to restore later
        original_path = self.rh.emhass_conf["data_path"]
        try:
            self.rh.emhass_conf["data_path"] = pathlib.Path("/tmp")

            # Mock os.path.isfile to return False (triggers new metadata file creation)
            with patch("os.path.isfile", return_value=False):
                # Mock pathlib.Path.mkdir to avoid file system errors, and
                # os.replace since aiofiles.open is mocked (no real temp file is
                # written for the atomic metadata commit to move into place).
                with patch("pathlib.Path.mkdir"), patch("os.replace") as mock_replace:
                    # FIX: Pass dont_post=True to bypass network failure and force response_ok=True
                    # This ensures the save_entities logic block is actually reached
                    response, data = await self.rh.post_data(
                        data_df,
                        idx,
                        entity_id,
                        "power",
                        "W",
                        "PV",
                        "power",
                        save_entities=True,
                        dont_post=True,
                    )
                    # Both the entity data file and metadata.json are committed
                    # atomically: each is os.replace'd from a temp file into
                    # place (never an in-place write). Exactly two replaces, one
                    # per file; the relative order is not significant.
                    self.assertEqual(mock_replace.call_count, 2)
                    replaced = {}
                    for call in mock_replace.call_args_list:
                        tmp_src, dest = call.args
                        self.assertTrue(str(tmp_src).endswith(".tmp"))
                        replaced[pathlib.Path(dest).name] = str(tmp_src)
                    self.assertIn("metadata.json", replaced)
                    self.assertIn(entity_id + ".json", replaced)
        finally:
            # Restore path to prevent polluting other tests
            self.rh.emhass_conf["data_path"] = original_path

        # Verify file write called (once for data, once for metadata)
        self.assertTrue(mock_f.write.called)
        self.assertGreaterEqual(mock_f.write.call_count, 2)

        # Test Error Handling (response_ok = False): the next POST gets a 503. post_data reports
        # 500 for any transport exception, so a 503 can only come from the mocked response.
        mocked.post(states_url, status=503)
        # Use dont_post=False to force the POST (get_data_from_file is False, set at start)
        response, _ = await self.rh.post_data(
            data_df, idx, entity_id, "power", "W", "Fail", "power", dont_post=False
        )

        self.assertFalse(response.ok)
        self.assertEqual(response.status_code, 503)
        # The four successful posts and the failing one all reached the mocked endpoint
        posts = sum(
            len(calls) for (method, _), calls in mocked.requests.items() if method == "POST"
        )
        self.assertEqual(posts, 5)

    async def test_concurrent_save_entities_metadata_integrity(self):
        """Concurrent publishes that share entities/metadata.json must not
        corrupt it. The read-modify-write is serialized by a process-wide lock
        and committed via an atomic os.replace, so the final file is always
        valid JSON containing every published entity (regression for the
        shared-state race between the dh/mpc/hwc pipelines)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.rh.emhass_conf["data_path"] = pathlib.Path(tmpdir)
            # get_data_from_file=True skips the network POST but still reaches
            # the save_entities block (response_ok is forced True).
            self.rh.get_data_from_file = True

            data_df = pd.Series(
                [100.0, 200.0],
                index=pd.date_range("2024-01-01", periods=2, freq="30min"),
            )
            data_df.name = "test_data"

            entity_ids = [f"sensor.race_test_{i}" for i in range(25)]

            async def publish(entity_id):
                await self.rh.post_data(
                    data_df,
                    0,
                    entity_id,
                    "power",
                    "W",
                    entity_id,
                    "power",
                    save_entities=True,
                    dont_post=True,
                )

            # Fire them all concurrently so their await points interleave.
            await asyncio.gather(*(publish(e) for e in entity_ids))

            entities_path = pathlib.Path(tmpdir) / "entities"
            metadata_path = entities_path / "metadata.json"
            self.assertTrue(metadata_path.is_file())

            # Must parse cleanly: no concatenated or truncated documents.
            with open(metadata_path, "rb") as f:
                metadata = orjson.loads(f.read())

            # Every concurrently-published entity must survive (no lost writes).
            for entity_id in entity_ids:
                self.assertIn(entity_id, metadata)

            # The atomic commit must not leave temp files behind.
            leftover = list(entities_path.glob("metadata.json.*.tmp"))
            self.assertEqual(leftover, [])

    async def test_save_entities_recovers_from_corrupt_metadata(self):
        """A pre-existing, unparseable metadata.json must be quarantined to
        metadata_corrupt.json and replaced by a fresh, valid index containing
        the entity being published (recovery branch of the shared-state race
        handling)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.rh.emhass_conf["data_path"] = pathlib.Path(tmpdir)
            self.rh.get_data_from_file = True

            entities_path = pathlib.Path(tmpdir) / "entities"
            entities_path.mkdir(parents=True)
            metadata_path = entities_path / "metadata.json"
            # Two concatenated documents: exactly the corruption the fix targets.
            garbage = b'{"sensor.old": {}}\n{"sensor.old": {}}\n'
            metadata_path.write_bytes(garbage)

            data_df = pd.Series(
                [100.0, 200.0],
                index=pd.date_range("2024-01-01", periods=2, freq="30min"),
            )
            data_df.name = "test_data"

            await self.rh.post_data(
                data_df,
                0,
                "sensor.recovered",
                "power",
                "W",
                "Recovered",
                "power",
                save_entities=True,
                dont_post=True,
            )

            # The corrupt file is quarantined verbatim ...
            corrupt_path = entities_path / "metadata_corrupt.json"
            self.assertTrue(corrupt_path.is_file())
            self.assertEqual(corrupt_path.read_bytes(), garbage)

            # ... and metadata.json is rebuilt as a valid index with the entity.
            with open(metadata_path, "rb") as f:
                metadata = orjson.loads(f.read())
            self.assertIn("sensor.recovered", metadata)
            self.assertNotIn("sensor.old", metadata)

    async def test_corrupt_quarantine_tolerates_lost_race(self):
        """If a concurrent process already moved the corrupt metadata file, the
        FileNotFoundError from the quarantine os.replace is swallowed and the
        publish still rebuilds a valid index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.rh.emhass_conf["data_path"] = pathlib.Path(tmpdir)
            self.rh.get_data_from_file = True

            entities_path = pathlib.Path(tmpdir) / "entities"
            entities_path.mkdir(parents=True)
            metadata_path = entities_path / "metadata.json"
            metadata_path.write_bytes(b"not json{{{")

            data_df = pd.Series(
                [100.0, 200.0],
                index=pd.date_range("2024-01-01", periods=2, freq="30min"),
            )
            data_df.name = "test_data"

            real_replace = os.replace

            def replace_side_effect(src, dst):
                # Simulate the corrupt file having vanished out from under us;
                # let the final atomic commit proceed for real.
                if pathlib.Path(dst).name == "metadata_corrupt.json":
                    raise FileNotFoundError(src)
                return real_replace(src, dst)

            with patch("os.replace", side_effect=replace_side_effect):
                response, _ = await self.rh.post_data(
                    data_df,
                    0,
                    "sensor.x",
                    "power",
                    "W",
                    "X",
                    "power",
                    save_entities=True,
                    dont_post=True,
                )

            self.assertTrue(response.ok)
            with open(metadata_path, "rb") as f:
                metadata = orjson.loads(f.read())
            self.assertIn("sensor.x", metadata)

    async def test_entity_data_file_written_atomically(self):
        """The per-entity data file must be committed via a temp file + os.replace,
        not an in-place truncating write, so the continual_publish reader never
        observes the zero-byte window that raises ``ValueError: Expected object or
        value`` in pd.read_json. Regression for issue #1000."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.rh.emhass_conf["data_path"] = pathlib.Path(tmpdir)
            # get_data_from_file=True skips the network POST but still reaches
            # the save_entities block (response_ok is forced True).
            self.rh.get_data_from_file = True

            data_df = pd.Series(
                [100.0, 200.0],
                index=pd.date_range("2024-01-01", periods=2, freq="30min"),
            )
            data_df.name = "test_data"
            entity_id = "sensor.atomic_test"

            replaced_names = []
            real_replace = os.replace

            def spy_replace(src, dst):
                replaced_names.append(pathlib.Path(dst).name)
                return real_replace(src, dst)

            with patch("os.replace", side_effect=spy_replace):
                response, _ = await self.rh.post_data(
                    data_df,
                    0,
                    entity_id,
                    "power",
                    "W",
                    entity_id,
                    "power",
                    save_entities=True,
                    dont_post=True,
                )

            self.assertTrue(response.ok)
            entities_path = pathlib.Path(tmpdir) / "entities"
            entity_file = entities_path / (entity_id + ".json")
            # The entity data file is committed through os.replace, exactly like
            # metadata.json (RED on base: base only os.replace's metadata.json).
            # Exactly two atomic commits: the entity file and metadata.json.
            self.assertEqual(len(replaced_names), 2)
            self.assertIn(entity_id + ".json", replaced_names)
            # The final file exists and parses cleanly.
            self.assertTrue(entity_file.is_file())
            with open(entity_file, "rb") as f:
                orjson.loads(f.read())
            # The atomic commit must not leave temp files behind.
            self.assertEqual(list(entities_path.glob("*.tmp")), [])

    async def test_session_lazy_initialization(self):
        """Test that session is lazily initialized on first use."""
        # Session should be None initially
        self.assertIsNone(self.rh._session)

        # Get session should create one
        session = await self.rh._get_session()
        self.assertIsNotNone(session)
        self.assertFalse(session.closed)

        # Getting session again should return the same instance
        session2 = await self.rh._get_session()
        self.assertIs(session, session2)

        # Clean up
        await self.rh.close()

    async def test_session_reuse_across_post_data_calls(self):
        """Verify session reuse and entity-specific HTTP publications."""
        self.rh.get_data_from_file = False
        self.rh.hass_url = "https://example.invalid/"
        self.rh.long_lived_token = "stage3c-test-token"
        self.rh.ssl_verify = True
        self.rh.publish_horizon_attributes = True

        data_df = pd.Series(
            [100.0, 200.0],
            index=pd.date_range("2024-01-01", periods=2, freq="30min"),
        )

        mock_response = MagicMock(ok=True, status=200)
        mock_response.read = AsyncMock(return_value=b"")

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        expected_payloads = []

        try:
            # Use the real session lookup; intercept only HTTP transport.
            with patch.object(
                aiohttp.ClientSession, "post", return_value=mock_context
            ) as mock_post:
                first_session = None

                for number in range(1, 4):
                    entity_id = f"sensor.test{number}"

                    response, returned_data = await self.rh.post_data(
                        data_df * number,
                        0,
                        entity_id,
                        "power",
                        "W",
                        f"Test {number}",
                        "power",
                    )

                    self.assertTrue(response.ok)
                    self.assertEqual(response.status_code, 200)

                    expected_payload = {
                        "state": f"{100 * number:.2f}",
                        "attributes": {
                            "device_class": "power",
                            "unit_of_measurement": "W",
                            "friendly_name": f"Test {number}",
                            "forecasts": [
                                {
                                    "date": "2024-01-01T00:00:00",
                                    f"test{number}": f"{100 * number:.1f}",
                                },
                                {
                                    "date": "2024-01-01T00:30:00",
                                    f"test{number}": f"{200 * number:.1f}",
                                },
                            ],
                            "state_class": "measurement",
                        },
                    }

                    expected_payloads.append(expected_payload)
                    self.assertEqual(returned_data, expected_payload)

                    session = self.rh._session
                    self.assertIsInstance(session, aiohttp.ClientSession)
                    self.assertFalse(session.closed)

                    if first_session is None:
                        first_session = session
                    else:
                        self.assertIs(session, first_session)

                # Verify all requests were made, in order.
                self.assertEqual(mock_post.call_count, 3)

                for number, request in enumerate(mock_post.call_args_list, start=1):
                    # Three distinct Home Assistant entity URLs.
                    self.assertEqual(
                        request.args,
                        (f"https://example.invalid/api/states/sensor.test{number}",),
                    )

                    self.assertEqual(
                        set(request.kwargs),
                        {"headers", "data", "ssl"},
                    )

                    # Exact HTTP authentication and content type.
                    self.assertEqual(
                        request.kwargs["headers"],
                        {
                            "Authorization": "Bearer stage3c-test-token",
                            "content-type": "application/json",
                        },
                    )

                    self.assertIs(request.kwargs["ssl"], True)

                    # Decode the actual HTTP payload and verify its
                    # entity-specific state, metadata and forecast.
                    self.assertEqual(
                        orjson.loads(request.kwargs["data"]),
                        expected_payloads[number - 1],
                    )

                self.assertEqual(mock_response.read.await_count, 3)
                self.assertEqual(mock_context.__aenter__.await_count, 3)
                self.assertEqual(mock_context.__aexit__.await_count, 3)

        finally:
            await self.rh.close()

        self.assertIsNone(self.rh._session)

    async def test_post_scalar_sensor_numpy_values(self):
        """Test that post_scalar_sensor serialises numpy scalars (#1131)."""
        self.rh.get_data_from_file = False

        with patch.object(self.rh, "_get_session") as mock_get_session:
            mock_session = MagicMock()
            mock_response = AsyncMock()
            mock_response.ok = True
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_get_session.return_value = mock_session

            posted = await self.rh.post_scalar_sensor(
                "sensor.battery_identified_capacity",
                np.float64(9.8765),
                {"ci_low": np.float64(9.5), "ci_high": np.float64(10.25)},
            )

        self.assertTrue(posted)
        body = orjson.loads(mock_session.post.call_args.kwargs["data"])
        self.assertEqual(body["state"], 9.8765)
        self.assertEqual(body["attributes"], {"ci_low": 9.5, "ci_high": 10.25})

    async def test_session_close(self):
        """Test that close() properly closes the session."""
        # Create a session first
        session = await self.rh._get_session()
        self.assertIsNotNone(session)
        self.assertFalse(session.closed)

        # Close should work
        await self.rh.close()
        self.assertIsNone(self.rh._session)

        # Closing again should be a no-op (no error)
        await self.rh.close()

    async def test_session_recreated_after_close(self):
        """Test that a new session is created after closing the old one."""
        # Create initial session
        session1 = await self.rh._get_session()

        # Close it
        await self.rh.close()

        # Get a new session
        session2 = await self.rh._get_session()

        # Should be a different session
        self.assertIsNot(session1, session2)
        self.assertFalse(session2.closed)

        # Clean up
        await self.rh.close()

    async def test_async_context_manager(self):
        """Test that RetrieveHass works as an async context manager."""
        async with self.rh as rh:
            # Should return self
            self.assertIs(rh, self.rh)
            # Create a session inside the context
            session = await rh._get_session()
            self.assertIsNotNone(session)
            self.assertFalse(session.closed)

        # After exiting context, session should be closed
        self.assertIsNone(self.rh._session)

    async def test_concurrent_get_session(self):
        """Test that concurrent _get_session calls only create one session."""
        # Launch multiple concurrent _get_session calls
        sessions = await asyncio.gather(
            self.rh._get_session(),
            self.rh._get_session(),
            self.rh._get_session(),
        )
        # All should return the same session instance
        self.assertIs(sessions[0], sessions[1])
        self.assertIs(sessions[1], sessions[2])

        # Clean up
        await self.rh.close()

    @patch.dict(os.environ, {"SUPERVISOR_TOKEN": "mock_supervisor_token"})
    async def test_supervisor_token_fallback_success(self):
        """SUPERVISOR_TOKEN stands in for an empty long_lived_token, for the Supervisor URL only."""
        supervisor_url = "http://supervisor/core/api"

        def make_rh(url):
            return RetrieveHass(
                url,
                "empty",  # Trigger the token == "empty" fallback
                self.retrieve_hass_conf["optimization_time_step"],
                self.retrieve_hass_conf["time_zone"],
                {},
                emhass_conf,
                logger,
                get_data_from_file=False,
            )

        def sent_tokens(mocked):
            return [
                call.kwargs["headers"]["Authorization"]
                for calls in mocked.requests.values()
                for call in calls
            ]

        days_list = pd.date_range(start="2024-01-01", periods=1, freq="D", tz="UTC")
        var_list = ["sensor.test"]

        # Supervisor URL: get_ha_config uses the environment token
        rh_supervisor = make_rh(supervisor_url)
        with self.assertLogs(logger, level="DEBUG") as cm:
            with aioresponses() as mocked:
                mocked.get(supervisor_url + "/config", status=200, payload={"time_zone": "UTC"})
                self.assertTrue(await rh_supervisor.get_ha_config())
                self.assertEqual(sent_tokens(mocked), ["Bearer mock_supervisor_token"])
        self.assertTrue(any("Using SUPERVISOR_TOKEN" in log for log in cm.output))
        self.assertEqual(rh_supervisor.ha_config, {"time_zone": "UTC"})

        # Supervisor URL: _get_data_rest_api uses it too
        with self.assertLogs(logger, level="DEBUG") as cm:
            with aioresponses() as mocked:
                mocked.get(
                    re.compile(r"^http://supervisor/core/api/history/period/.*"),
                    status=200,
                    payload=[],
                )
                await rh_supervisor._get_data_rest_api(days_list, var_list)
                self.assertEqual(sent_tokens(mocked), ["Bearer mock_supervisor_token"])
        self.assertTrue(any("Using SUPERVISOR_TOKEN" in log for log in cm.output))

        # Any other URL never receives the Supervisor token: nothing is sent
        rh_custom = make_rh("https://myhass.duckdns.org/")
        with aioresponses() as mocked:
            mocked.get(re.compile(r".*"), status=200, payload={"time_zone": "UTC"}, repeat=True)
            self.assertTrue(await rh_custom.get_ha_config())
            self.assertEqual(rh_custom.ha_config, {})
            with self.assertLogs(logger, level="ERROR") as cm:
                self.assertFalse(await rh_custom._get_data_rest_api(days_list, var_list))
            self.assertEqual(sent_tokens(mocked), [])
        self.assertTrue(any("No valid authentication token found" in log for log in cm.output))

        await rh_supervisor.close()
        await rh_custom.close()

    @patch.dict(os.environ, {}, clear=True)
    async def test_supervisor_token_fallback_failure(self):
        """Test that an error is logged and returns False when no token is available."""
        # Ensure SUPERVISOR_TOKEN is completely missing from environment
        if "SUPERVISOR_TOKEN" in os.environ:
            del os.environ["SUPERVISOR_TOKEN"]

        rh_no_token = RetrieveHass(
            self.retrieve_hass_conf["hass_url"],
            "empty",  # Trigger the token == "empty" fallback
            self.retrieve_hass_conf["optimization_time_step"],
            self.retrieve_hass_conf["time_zone"],
            {},
            emhass_conf,
            logger,
            get_data_from_file=False,
        )

        days_list = pd.date_range(start="2024-01-01", periods=1, freq="D", tz="UTC")
        var_list = ["sensor.test"]

        # Test the final coverage gap (lines 454-457): No token anywhere
        with self.assertLogs(logger, level="ERROR") as cm:
            result = await rh_no_token._get_data_rest_api(days_list, var_list)

        self.assertFalse(result)
        self.assertTrue(any("No valid authentication token found" in log for log in cm.output))


class TestSupervisorUrlForms(unittest.IsolatedAsyncioTestCase):
    """#1403: every REST path uses the Supervisor's own paths for any way of writing its
    URL, and https://supervisor/core/api is not the Supervisor (its proxy is HTTP only)."""

    SUPERVISOR_FORMS = (
        "http://supervisor/core/api",
        "http://supervisor/core/api/",
        " HTTP://Supervisor/core/api/ ",
    )
    SUPERVISOR_API = "http://supervisor/core/api"
    # Handled like any other Home Assistant URL
    HTTPS_FORM = "https://supervisor/core/api/"

    @staticmethod
    def _rh(url):
        return RetrieveHass(
            url,
            "llat",
            pd.Timedelta(minutes=30),
            datetime.UTC,
            {},
            emhass_conf,
            logger,
        )

    @staticmethod
    def _sent(mocked):
        return [str(url) for (_, url), calls in mocked.requests.items() for _ in calls]

    def _cases(self, path):
        """(hass_url, expected URL) for a REST path written without its leading ``api/``."""
        cases = [(url, f"{self.SUPERVISOR_API}/{path}") for url in self.SUPERVISOR_FORMS]
        return [*cases, (self.HTTPS_FORM, f"{self.HTTPS_FORM}api/{path}")]

    def test_only_the_http_form_uses_the_supervisor_api(self):
        from emhass.retrieve_hass import _uses_supervisor_api

        for url in self.SUPERVISOR_FORMS:
            with self.subTest(url=url):
                self.assertTrue(_uses_supervisor_api(url))
        # An unset URL is defaulted by the caller first, never turned into "empty/states/..."
        for url in (None, "", "empty", "https://supervisor/core/api", self.HTTPS_FORM):
            with self.subTest(url=url):
                self.assertFalse(_uses_supervisor_api(url))

    async def test_get_ha_config(self):
        for url, expected in self._cases("config"):
            with self.subTest(url=url), aioresponses() as mocked:
                rh = self._rh(url)
                self.addAsyncCleanup(rh.close)
                mocked.get(re.compile(r".*"), status=200, payload={}, repeat=True)
                self.assertTrue(await rh.get_ha_config())
                self.assertEqual(self._sent(mocked), [expected])

    def test_history_url(self):
        day = pd.Timestamp("2026-11-01", tz="UTC")
        for url, expected in self._cases("history/period/2026-11-01T00:00:00Z"):
            with self.subTest(url=url):
                built = self._rh(url)._build_history_url(day, "sensor.x", "empty", False, False)
                self.assertEqual(built, expected + "?filter_entity_id=sensor.x")

    async def test_live_state(self):
        for url, expected in self._cases("states/sensor.x"):
            with self.subTest(url=url), aioresponses() as mocked:
                rh = self._rh(url)
                self.addAsyncCleanup(rh.close)
                session = await rh._get_session()
                mocked.get(re.compile(r".*"), status=200, payload={"state": "1"}, repeat=True)
                live = await rh._fetch_live_state(session, "sensor.x", {})
                self.assertEqual(live, {"state": "1"})
                self.assertEqual(self._sent(mocked), [expected])

    # Core URLs with and without the trailing slash: publishing gets one "/api/states/".
    CORE_STATES = (
        ("http://ha:8123", "http://ha:8123/api/states/sensor.x"),
        ("http://ha:8123/", "http://ha:8123/api/states/sensor.x"),
        ("https://ha.example.invalid/ha/", "https://ha.example.invalid/ha/api/states/sensor.x"),
        ("https://supervisor/core/api", "https://supervisor/core/api/api/states/sensor.x"),
    )

    def _publish_cases(self):
        return [*self._cases("states/sensor.x"), *self.CORE_STATES]

    async def test_post_data(self):
        data = pd.Series([1.0, 2.0], index=pd.date_range("2026-11-01", periods=2, freq="30min"))
        data.name = "x"
        for url, expected in self._publish_cases():
            with self.subTest(url=url), aioresponses() as mocked:
                rh = self._rh(url)
                self.addAsyncCleanup(rh.close)
                mocked.post(re.compile(r".*"), status=200, repeat=True)
                await rh.post_data(data, 0, "sensor.x", "power", "W", "X", "power")
                self.assertEqual(self._sent(mocked), [expected])

    async def test_post_scalar_sensor(self):
        for url, expected in self._publish_cases():
            with self.subTest(url=url), aioresponses() as mocked:
                rh = self._rh(url)
                self.addAsyncCleanup(rh.close)
                mocked.post(re.compile(r".*"), status=200, repeat=True)
                self.assertTrue(await rh.post_scalar_sensor("sensor.x", 1, {}))
                self.assertEqual(self._sent(mocked), [expected])


class TestRestHistoryEpochGrid(unittest.IsolatedAsyncioTestCase):
    """#1381: REST history keeps one grid when the time step does not divide a day."""

    DAYS = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    LOAD, PV = "sensor.power_load", "sensor.power_pv"
    NOW = datetime.datetime(2026, 2, 20, 10, 20, tzinfo=datetime.UTC)

    @staticmethod
    def _rh(freq):
        return RetrieveHass(
            "http://ha.invalid/",
            "token",
            pd.Timedelta(freq),
            datetime.UTC,
            {},
            emhass_conf,
            logger,
        )

    @staticmethod
    def _rows(entries):
        """HA history rows from (iso timestamp, state) pairs."""
        return [{"state": state, "last_changed": ts} for ts, state in entries]

    def _minute_payloads(self):
        """One reading a minute for each day and sensor, on a curve that never repeats a day."""
        payloads = {}
        for day in self.DAYS:
            stamps = pd.date_range(day, periods=1440, freq="min")
            t = stamps.asi8 / 6e10
            for var, base in ((self.LOAD, 1000.0), (self.PV, 3000.0)):
                values = base + 500.0 * np.sin(2 * np.pi * t / 97.0) + (t % 13)
                payloads[(day.date(), var)] = self._rows(
                    (at.isoformat(), f"{value:.3f}") for at, value in zip(stamps, values)
                )
        return payloads

    async def _get(self, rh, payloads, days=None, **kwargs):
        async def fetch(_self, session, url, headers, var, day, is_first_day=False):
            return payloads.get((pd.Timestamp(day).date(), var))

        days = self.DAYS if days is None else days
        var_list = sorted({var for _, var in payloads})
        with patch.object(RetrieveHass, "_fetch_history_data", new=fetch):
            ok = await rh.get_data(days, var_list, **kwargs)
        self.assertTrue(ok)
        return rh.df_final

    @staticmethod
    def _on_epoch_grid(index, freq):
        return bool(((index - pd.Timestamp(0, tz="UTC")) % freq == pd.Timedelta(0)).all())

    @staticmethod
    def _series(rows):
        index = pd.to_datetime([row["last_changed"] for row in rows], format="ISO8601")
        return pd.Series([float(row["state"]) for row in rows], index=index).sort_index()

    async def test_every_day_keeps_its_data_when_the_step_does_not_divide_a_day(self):
        """At 7, 25 and 50 minutes every day comes back, and each bucket is the mean of
        all the rows in it, the ones either side of midnight included."""
        payloads = self._minute_payloads()
        for minutes in (7, 25, 50):
            with self.subTest(minutes=minutes):
                rh = self._rh(f"{minutes}min")
                df = await self._get(rh, payloads)
                for day in self.DAYS:
                    in_day = df[(df.index >= day) & (df.index < day + pd.Timedelta(days=1))]
                    self.assertFalse(in_day.isna().any().any(), f"{day.date()} lost data")
                self.assertEqual(df.index.freq, rh.freq)
                self.assertTrue(self._on_epoch_grid(df.index, rh.freq))
                expected = pd.DataFrame(
                    {
                        var: self._series(
                            [row for (_, v), rows in payloads.items() if v == var for row in rows]
                        )
                        .resample(rh.freq, origin="epoch")
                        .mean()
                        for var in (self.LOAD, self.PV)
                    }
                )
                pd.testing.assert_index_equal(df.index, expected.index, check_names=False)
                np.testing.assert_allclose(
                    df[[self.LOAD, self.PV]].to_numpy(), expected.to_numpy(), rtol=1e-12
                )

    async def test_a_step_that_divides_a_day_keeps_the_same_frame(self):
        """Steps that divide a day put every UTC midnight on the epoch grid, so the frame
        is exactly what resampling each day from its own midnight gives."""
        payloads = self._minute_payloads()
        for minutes in (5, 15, 30, 60, 120):
            with self.subTest(minutes=minutes):
                rh = self._rh(f"{minutes}min")
                with patch.object(RetrieveHass, "_merge_straddling_buckets", create=True) as merge:
                    df = await self._get(rh, payloads, record_staleness=True)
                # Nothing to merge: each bucket comes from a single day.
                merge.assert_not_called()
                for var in (self.LOAD, self.PV):
                    # Every row but the one stamped at each day start is a real reading.
                    self.assertTrue(rh.bucket_row_count[var].index.is_unique)
                    self.assertEqual(int(rh.bucket_row_count[var].sum()), 4 * 1439)
                days = []
                for day in self.DAYS:
                    frames = []
                    for var in sorted((self.LOAD, self.PV)):
                        series = self._series(payloads[(day.date(), var)])
                        series.index.name = "last_changed"
                        frames.append(series.to_frame(var).resample(rh.freq).mean())
                    days.append(pd.concat(frames, axis=1))
                expected = utils.set_df_index_freq(pd.concat(days, axis=0))
                pd.testing.assert_frame_equal(df, expected)

    async def test_the_bucket_across_midnight_is_the_mean_of_all_its_rows(self):
        """Each day's half of the straddling bucket is weighted by its rows; a sensor with
        rows on one side only keeps that side's mean."""
        rh = self._rh("7min")
        days = self.DAYS[:2]
        midnight = days[1]
        straddle = midnight.floor(rh.freq)
        self.assertEqual(midnight - straddle, pd.Timedelta(minutes=4))

        def at(offset):
            return (straddle + pd.Timedelta(offset)).isoformat()

        payloads = {
            (days[0].date(), self.LOAD): self._rows(
                [
                    (days[0].isoformat(), "50"),
                    (at("30s"), "100"),
                    (at("2min"), "200"),
                    (at("210s"), "300"),
                ]
            ),
            (days[1].date(), self.LOAD): self._rows(
                [(midnight.isoformat(), "600"), (at("30min"), "700")]
            ),
            (days[0].date(), self.PV): self._rows(
                [(days[0].isoformat(), "1"), ((days[0] + pd.Timedelta(hours=12)).isoformat(), "2")]
            ),
            (days[1].date(), self.PV): self._rows(
                [(midnight.isoformat(), "10"), (at("5min"), "20")]
            ),
        }
        df = await self._get(rh, payloads, days=days)
        self.assertTrue(df.index.is_unique)
        # (100 + 200 + 300 + 600) / 4, not day 1's half alone (200) or the halves' mean (400)
        self.assertEqual(df[self.LOAD].get(straddle), 300.0)
        self.assertEqual(df[self.PV].get(straddle), 15.0)

    async def test_staleness_at_the_bucket_across_midnight(self):
        """The later real reading is the bucket's last state and the readings add up; the
        row HA carries in at the day start is no reading."""
        days = self.DAYS[:2]
        midnight = days[1]
        for day2, last, count in (
            ([("0s", "600"), ("90s", "650"), ("30min", "700")], 650.0, 3),
            ([("0s", "600"), ("30min", "700")], 200.0, 2),
        ):
            with self.subTest(last=last):
                rh = self._rh("7min")
                straddle = midnight.floor(rh.freq)
                payloads = {
                    (days[0].date(), self.LOAD): self._rows(
                        [
                            (days[0].isoformat(), "50"),
                            ((straddle + pd.Timedelta("30s")).isoformat(), "100"),
                            ((straddle + pd.Timedelta("2min")).isoformat(), "200"),
                        ]
                    ),
                    (days[1].date(), self.LOAD): self._rows(
                        [((midnight + pd.Timedelta(o)).isoformat(), v) for o, v in day2]
                    ),
                }
                await self._get(
                    rh, payloads, days=days, keep_partial_days=True, record_staleness=True
                )
                states = rh.bucket_last_state[self.LOAD]
                counts = rh.bucket_row_count[self.LOAD]
                self.assertTrue(states.index.is_unique)
                self.assertTrue(counts.index.is_unique)
                self.assertEqual(states.get(straddle), last)
                self.assertEqual(counts.get(straddle), count)

    async def test_unavailable_half_of_the_bucket_across_midnight_with_a_reading(self):
        """A straddling bucket with a reading on one side is observed, not unavailable,
        whether day 1 has an unavailable row in it or its frame stops before it."""
        days = pd.date_range("2026-02-19", periods=2, freq="D", tz="UTC")
        midnight = days[1]
        for day1_in_bucket in (True, False):
            with self.subTest(day1_in_bucket=day1_in_bucket):
                rh = self._rh("7min")
                straddle = midnight.floor(rh.freq)
                day1 = [
                    (days[0].isoformat(), "500"),
                    ((straddle - pd.Timedelta("10min")).isoformat(), "unavailable"),
                ]
                if day1_in_bucket:
                    day1.append(((straddle + pd.Timedelta("1min")).isoformat(), "unavailable"))
                payloads = {
                    (days[0].date(), self.LOAD): self._rows(day1),
                    (days[1].date(), self.LOAD): self._rows(
                        [
                            (midnight.isoformat(), "unavailable"),
                            ((midnight + pd.Timedelta("1min")).isoformat(), "510"),
                        ]
                    ),
                }
                with (
                    time_machine.travel(self.NOW, tick=False),
                    patch.object(RetrieveHass, "_fetch_live_state", new=AsyncMock(return_value={})),
                    patch.object(RetrieveHass, "_live_state_is_fresh", return_value=True),
                ):
                    df = await self._get(rh, payloads, days=days, hold_unchanged_tail=True)
                self.assertEqual(df[self.LOAD].get(straddle), 510.0)
                listed = rh.unavailable_buckets[self.LOAD]
                self.assertIn(straddle - rh.freq, listed)
                self.assertNotIn(straddle, listed)

    def _process(self, rows, day, is_last_day, freq="7min", **kwargs):
        rh = self._rh(freq)
        with time_machine.travel(self.NOW, tick=False):
            frame = rh._process_history_dataframe(
                self._rows(rows),
                self.LOAD,
                pd.Timestamp(day, tz="UTC"),
                False,
                is_last_day,
                **kwargs,
            )
        return rh, frame

    def test_tail_hold_when_the_first_bucket_starts_before_midnight(self):
        """The last day's only row is the state HA carries in at midnight, which sits in
        a bucket that starts the day before; it is still held up to the running bucket."""
        rh, frame = self._process(
            [("2026-02-20T00:00:00+00:00", "520")],
            "2026-02-20",
            True,
            hold_unchanged_tail=True,
        )
        running = pd.Timestamp(self.NOW).floor(rh.freq)
        self.assertTrue(self._on_epoch_grid(frame.index, rh.freq))
        self.assertEqual(frame.index[0], pd.Timestamp("2026-02-19 23:55", tz="UTC"))
        self.assertEqual(frame.index[-1], running)
        self.assertTrue((frame[self.LOAD] == 520.0).all())

    def test_outage_at_the_end_of_a_past_day_covers_its_last_bucket(self):
        """The day's last bucket is the one holding its last instant, even when that
        bucket ends after midnight."""
        found = []
        rh, _ = self._process(
            [("2026-02-19T00:00:00+00:00", "500"), ("2026-02-19T22:30:00+00:00", "unavailable")],
            "2026-02-19",
            False,
            unavailable=found,
        )
        self.assertEqual(found[0][-1], pd.Timestamp("2026-02-19 23:55", tz="UTC"))
        self.assertTrue(self._on_epoch_grid(found[0], rh.freq))

    def test_outage_at_the_end_of_a_past_day_at_a_step_that_divides_a_day(self):
        """At a step that divides a day the covered buckets are the same as before, in
        the frame's own time unit."""
        found = []
        _, frame = self._process(
            [("2026-02-19T00:00:00+00:00", "500"), ("2026-02-19T22:30:00+00:00", "unavailable")],
            "2026-02-19",
            False,
            freq="60min",
            unavailable=found,
        )
        expected = pd.date_range("2026-02-19 22:00", "2026-02-19 23:00", freq="60min", tz="UTC")
        self.assertEqual(list(found[0]), list(expected))
        self.assertEqual(found[0].dtype, frame.index.dtype)


class TestSslNoVerify(unittest.IsolatedAsyncioTestCase):
    """ssl_no_verify, saved in the configuration, reaches the Home Assistant calls."""

    async def build(self, ssl_no_verify=None):
        environ = self.enterContext(patch.dict(os.environ))
        for name in ("EMHASS_URL", "SUPERVISOR_TOKEN", "TIME_ZONE"):
            environ.pop(name, None)
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        if ssl_no_verify is not None:
            config["ssl_no_verify"] = ssl_no_verify
        _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
        params = await utils.build_params(emhass_conf, secrets, config, logger)
        return RetrieveHass(
            "https://ha.invalid:8123/", "token", 30, "UTC", params, emhass_conf, logger
        )

    async def test_default_verifies(self):
        self.assertIsNone((await self.build()).ssl_verify)
        self.assertIsNone((await self.build(False)).ssl_verify)
        # only a real true turns it off, not text such as "false"
        self.assertIsNone((await self.build("false")).ssl_verify)

    async def test_set_in_the_configuration_turns_verification_off(self):
        with self.assertLogs(logger, level="WARNING") as cm:
            rh = await self.build(True)
        self.assertIs(rh.ssl_verify, False)
        self.assertTrue(any("SSL verification is disabled" in m for m in cm.output))

    async def test_requests_are_sent_with_it(self):
        rh = await self.build(True)
        self.addAsyncCleanup(rh.close)
        with aioresponses() as mocked:
            mocked.get("https://ha.invalid:8123/api/config", payload={"time_zone": "UTC"})
            await rh.get_ha_config()
            (call,) = mocked.requests.values()
        self.assertIs(call[0].kwargs.get("ssl"), False)


class _SilentServer:
    """Accepts TCP connections on 127.0.0.1 and never answers. Each connection is closed
    after ``hold_s``, so a client with no timeout of its own still finishes."""

    def __init__(self, hold_s: float):
        self.hold_s = hold_s
        self.accepted = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self._sock.settimeout(0.05)
        self.port = self._sock.getsockname()[1]
        self._open = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
                self.accepted += 1
                self._open.append((time.monotonic() + self.hold_s, conn))
            except TimeoutError:
                pass
            now = time.monotonic()
            for item in [item for item in self._open if item[0] <= now]:
                item[1].close()
                self._open.remove(item)

    def close(self):
        self._stop.set()
        self._thread.join()
        for _, conn in self._open:
            conn.close()
        self._sock.close()


class _TrickleServer:
    """Answers the first connection on 127.0.0.1 with a reply that never ends: one header
    byte every 0.5 s, inside any read timeout, for up to ``seconds``. Over TLS when given
    a server ``tls`` context. Records the request and whether the client shut the
    connection."""

    # The reply's start: by default its headers never end. A complete head that ends the
    # connection (Connection: close) is closed by http.client before its body is read.
    HEADER_TRICKLE = b"HTTP/1.1 204 No Content\r\nX-Influxdb-Version: 1.8\r\nX-Pad: "
    BODY_TRICKLE = (
        b"HTTP/1.1 200 OK\r\nX-Influxdb-Version: 1.8\r\nContent-Type: text/plain\r\n"
        b"Content-Length: 100000\r\nConnection: close\r\n\r\n"
    )

    def __init__(self, seconds: float, tls=None, head=HEADER_TRICKLE):
        self.seconds = seconds
        self.tls = tls
        self.head = head
        self.request = b""
        self.answering = threading.Event()
        self.client_closed = threading.Event()
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(4)
        self._sock.settimeout(0.05)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
                break
            except TimeoutError:
                continue
        else:
            return
        conn.settimeout(2)
        if self.tls is not None:
            try:
                conn = self.tls.wrap_socket(conn, server_side=True)
            except OSError:
                conn.close()
                return
        with conn:
            self.request = conn.recv(4096)
            conn.sendall(self.head)
            self.answering.set()
            end = time.monotonic() + self.seconds
            while time.monotonic() < end and not self._stop.is_set():
                readable, _, _ = select.select([conn], [], [], 0.5)
                try:
                    if readable and not conn.recv(1):
                        self.client_closed.set()
                        return
                    conn.sendall(b"a")
                except OSError:
                    self.client_closed.set()
                    return

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self._sock.close()


def _selected_measurement(query: str) -> str:
    """The measurement a SELECT built by _build_influx_query_for_measurement reads."""
    source = query.split("FROM", 1)[1].split("WHERE", 1)[0].strip()
    return source.rsplit('"."', 1)[-1].strip('"')


class _InfluxStandin:
    """An InfluxDB 1.x stand-in on 127.0.0.1: SHOW MEASUREMENTS lists ``measurements``,
    SHOW TAG VALUES FROM a measurement lists the entities ``entity_in`` gives it, and a
    SELECT for an entity returns its ``points``, from any measurement, or with
    ``by_measurement`` only from one ``entity_in`` lists it in. A query holding a key of
    ``slow`` waits that many seconds first; one holding a key of ``faults`` gets that
    reply instead (see FAULTS). With ``close`` every reply ends its connection."""

    # How a loaded server, or a proxy in front of it, can answer a query
    FAULTS = {
        "http400": (400, b'{"error":"error parsing query"}', "application/json"),
        "http403": (403, b'{"error":"error authorizing query"}', "application/json"),
        # Not now, rather than never: a proxy's request timeout, a rate limit
        "http408": (408, b"<html>408 Request Time-out</html>", "text/html"),
        "http429": (
            429,
            b'{"code":"too many requests","message":"rate limited"}',
            "application/json",
        ),
        "http500": (500, b'{"error":"timeout"}', "application/json"),
        "http504": (504, b"<html>504 Gateway Time-out</html>", "text/html"),
        "query timeout": (
            200,
            b'{"results":[{"statement_id":0,"error":"query-timeout limit exceeded"}]}',
            "application/json",
        ),
        "max queries": (
            200,
            b'{"results":[{"statement_id":0,'
            b'"error":"max-concurrent-queries limit exceeded(20, 20)"}]}',
            "application/json",
        ),
        # A 200 cut off part way with no Content-Length: it ends with the connection
        "cut, no length": (200, None, "application/json"),
    }

    def __init__(
        self,
        measurements=(),
        entity_in=None,
        slow=None,
        points=None,
        close=False,
        faults=None,
        by_measurement=False,
    ):
        self.queries = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _send(self, code, body=b""):
                self.send_response(code)
                self.send_header("X-Influxdb-Version", "1.8.10")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                if close:
                    self.send_header("Connection", "close")
                    self.close_connection = True
                self.end_headers()
                self.wfile.write(body)

            def _fault(self, query, code, body, content_type):
                if body is None:
                    whole = orjson.dumps({"results": [{"statement_id": 0, **self._result(query)}]})
                    self.send_response(code)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(whole[: len(whole) // 2])
                    self.close_connection = True
                    return
                self.send_response(code)
                self.send_header("X-Influxdb-Version", "1.8.10")
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _result(self, query):
                if query.startswith("SHOW MEASUREMENTS"):
                    values = [[m] for m in measurements]
                    return {
                        "series": [{"name": "measurements", "columns": ["name"], "values": values}]
                    }
                if query.startswith("SHOW TAG VALUES"):
                    measurement = query.split('FROM "', 1)[1].split('"', 1)[0]
                    values = [["entity_id", e] for e in (entity_in or {}).get(measurement, [])]
                    if values:
                        return {
                            "series": [
                                {"name": measurement, "columns": ["key", "value"], "values": values}
                            ]
                        }
                if "SELECT" in query:
                    entity = query.split('"entity_id"=\'', 1)[1].split("'", 1)[0]
                    held = (entity_in or {}).get(_selected_measurement(query), [])
                    if (points or {}).get(entity) and (not by_measurement or entity in held):
                        return {
                            "series": [
                                {
                                    "name": "x",
                                    "columns": ["time", "mean_value"],
                                    "values": points[entity],
                                }
                            ]
                        }
                return {}

            def do_GET(self):
                try:
                    if self.path.startswith("/ping"):
                        return self._send(204)
                    query = parse_qs(urlparse(self.path).query).get("q", [""])[0]
                    outer.queries.append(query)
                    for key, seconds in (slow or {}).items():
                        if key in query:
                            time.sleep(seconds)
                    for key, fault in (faults or {}).items():
                        if key in query:
                            return self._fault(query, *outer.FAULTS[fault])
                    body = {"results": [{"statement_id": 0, **self._result(query)}]}
                    self._send(200, orjson.dumps(body))
                except OSError:
                    # The client gave up on a slow reply
                    self.close_connection = True

            do_POST = do_GET

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()


class TestDatabaseReadLimits(unittest.IsolatedAsyncioTestCase):
    """A database server that is slow or never answers cannot hold the event loop or a
    read open (finding 6). Only 127.0.0.1 is contacted."""

    DAYS = pd.date_range(start="2026-11-01", periods=1, freq="D", tz="UTC")

    def setUp(self):
        # Loaded before any timed window, so the first import does not count in it
        import influxdb  # noqa: F401
        import requests  # noqa: F401

    @staticmethod
    def _rh(retrieve_hass_conf):
        return RetrieveHass(
            "http://127.0.0.1:1/",
            "token",
            pd.Timedelta(minutes=30),
            datetime.UTC,
            {"retrieve_hass_conf": retrieve_hass_conf},
            emhass_conf,
            logger,
        )

    def _influx_rh(self, port):
        return self._rh({"use_influxdb": True, "influxdb_host": "127.0.0.1", "influxdb_port": port})

    def _short_limits(self, timeout_s=(0.3, 0.3), deadline_s=30):
        # create=True: the same test runs against a build without these constants
        for name, value in (
            ("INFLUXDB_TIMEOUT_S", timeout_s),
            ("INFLUXDB_READ_DEADLINE_S", deadline_s),
            ("INFLUXDB_DEADLINE_MARGIN_S", 0.3),
            ("INFLUXDB_CUT_GRACE_S", 1),
            ("VM_CLIENT_TIMEOUT", aiohttp.ClientTimeout(total=0.5)),
        ):
            self.enterContext(patch.object(retrieve_hass, name, value, create=True))

    async def test_a_silent_influxdb_server_leaves_the_event_loop_free(self):
        self._short_limits(timeout_s=(0.5, 0.5))
        server = _SilentServer(hold_s=1.5)
        self.addCleanup(server.close)
        ticks = []

        async def ticker():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.02)

        task = asyncio.create_task(ticker())
        try:
            self.assertFalse(await self._influx_rh(server.port).get_data(self.DAYS, ["sensor.x"]))
        finally:
            task.cancel()
        # The loop kept running other work while the read waited, with no long stall
        self.assertGreaterEqual(len(ticks), 5)
        self.assertLess(max(b - a for a, b in itertools.pairwise(ticks)), 0.25)

    async def test_the_request_timeout_ends_an_influxdb_read(self):
        self._short_limits()
        server = _SilentServer(hold_s=2)
        self.addCleanup(server.close)
        start = time.monotonic()
        with self.assertLogs(logger, level="ERROR") as cm:
            self.assertFalse(await self._influx_rh(server.port).get_data(self.DAYS, ["sensor.x"]))
        self.assertLess(time.monotonic() - start, 1.5)
        self.assertEqual(server.accepted, 1)  # one attempt, no retries
        self.assertTrue(any("could not reach the configured InfluxDB" in x for x in cm.output))

    async def test_the_overall_deadline_ends_an_influxdb_read(self):
        # Every query answers, slowly, inside the request timeout; the read as a whole
        # must still stop at the deadline
        requests_seen = []

        class SlowInflux(BaseHTTPRequestHandler):
            def _reply(self):
                requests_seen.append(self.path)
                if self.path.startswith("/ping"):
                    self.send_response(204)
                    self.send_header("X-Influxdb-Version", "1.8")
                    self.end_headers()
                    return
                time.sleep(0.25)
                query = parse_qs(urlparse(self.path).query).get("q", [""])[0]
                result = {"statement_id": 0}
                if query == "SHOW MEASUREMENTS":
                    result["series"] = [
                        {"name": "measurements", "columns": ["name"], "values": [["W"]]}
                    ]
                body = orjson.dumps({"results": [result]})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = _reply

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), SlowInflux)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self._short_limits(timeout_s=(1, 1), deadline_s=0.6)
        sensors = [f"sensor.s{i}" for i in range(4)]
        start = time.monotonic()
        with self.assertLogs(logger, level="ERROR") as cm:
            ok = await self._influx_rh(server.server_port).get_data(self.DAYS, sensors)
        self.assertFalse(ok)
        self.assertLess(time.monotonic() - start, 1.5)
        self.assertTrue(any("InfluxDB read stopped" in x for x in cm.output))
        # 4 sensors take 12 slow queries; the deadline stopped it after a few
        self.assertLessEqual(len(requests_seen), 5)

    async def test_a_trickling_influxdb_reply_is_cut_at_the_deadline(self):
        # A byte every 0.5 s never trips the 1 s read timeout, so only the deadline can
        # end this read: the action returns and the connection is shut
        self._short_limits(timeout_s=(1, 1), deadline_s=1)
        server = _TrickleServer(seconds=8)
        self.addCleanup(server.close)
        start = time.monotonic()
        with self.assertLogs(logger, level="ERROR") as cm:
            self.assertFalse(await self._influx_rh(server.port).get_data(self.DAYS, ["sensor.x"]))
        took = time.monotonic() - start
        self.assertTrue(server.answering.is_set())
        self.assertLess(took, 3.5)
        self.assertTrue(any("InfluxDB read stopped" in x for x in cm.output))
        # The client shut the connection, long before the reply would have ended
        self.assertTrue(server.client_closed.wait(2))

    async def test_a_stuck_influxdb_read_does_not_hold_up_to_thread(self):
        # One thread for asyncio.to_thread, as on a small machine with every thread busy:
        # an InfluxDB read stuck on a slow server must not be using it
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        self._short_limits(timeout_s=(1, 1), deadline_s=30)
        server = _TrickleServer(seconds=8)
        self.addCleanup(server.close)
        read = asyncio.create_task(self._influx_rh(server.port).get_data(self.DAYS, ["sensor.x"]))
        for _ in range(100):
            if server.answering.is_set():
                break
            await asyncio.sleep(0.05)
        self.assertTrue(server.answering.is_set())
        start = time.monotonic()
        try:
            await asyncio.wait_for(asyncio.to_thread(time.sleep, 0), 2)
            waited = time.monotonic() - start
        except TimeoutError:
            waited = None
        finally:
            # Ends the read: the server closes the connection
            server.close()
            with self.assertLogs(logger, level="ERROR"):
                self.assertFalse(await read)
        self.assertIsNotNone(waited, "asyncio.to_thread waited behind the InfluxDB read")
        self.assertLess(waited, 0.5)

    def _tls_context(self):
        import ssl

        import trustme

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        trustme.CA().issue_cert("127.0.0.1").configure_cert(context)
        return context

    async def _assert_cut(self, server, rh):
        """The read returns at its deadline and its connection is shut at once."""
        start = time.monotonic()
        with self.assertLogs(logger, level="ERROR") as cm:
            self.assertFalse(await rh.get_data(self.DAYS, ["sensor.x"]))
        self.assertTrue(server.answering.is_set())
        self.assertLess(time.monotonic() - start, 3.5)
        self.assertTrue(any("InfluxDB read stopped" in x for x in cm.output))
        self.assertTrue(server.client_closed.wait(2), "the connection was not shut")

    async def test_a_trickling_tls_reply_is_cut_at_the_deadline(self):
        # The TLS wrap takes over the socket urllib3 opened: shutting that one does nothing
        self._short_limits(timeout_s=(1, 1), deadline_s=1)
        server = _TrickleServer(seconds=8, tls=self._tls_context())
        self.addCleanup(server.close)
        rh = self._rh(
            {
                "use_influxdb": True,
                "influxdb_host": "127.0.0.1",
                "influxdb_port": server.port,
                "influxdb_use_ssl": True,
                "influxdb_verify_ssl": False,
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # InsecureRequestWarning
            await self._assert_cut(server, rh)

    async def test_a_read_through_a_trickling_proxy_is_cut_at_the_deadline(self):
        # requests builds a proxy's connection pools apart from the others, on first use
        for use_ssl, variable in ((False, "HTTP_PROXY"), (True, "HTTPS_PROXY")):
            with self.subTest(proxy=variable):
                self._short_limits(timeout_s=(1, 1), deadline_s=1)
                server = _TrickleServer(seconds=8)
                self.addCleanup(server.close)
                proxy = {variable: f"http://127.0.0.1:{server.port}"}
                with patch.dict(os.environ, proxy):
                    for name in ("NO_PROXY", "no_proxy", "ALL_PROXY", "all_proxy"):
                        os.environ.pop(name, None)
                    rh = self._rh(
                        {
                            "use_influxdb": True,
                            "influxdb_host": "127.0.0.1",
                            "influxdb_port": 9,  # never reached: the proxy stands in
                            "influxdb_use_ssl": use_ssl,
                            "influxdb_verify_ssl": False,
                        }
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        await self._assert_cut(server, rh)
                # It went through the proxy: a proxy request or a CONNECT
                first = server.request.split(b" ", 2)[:2]
                self.assertIn(
                    first, ([b"GET", b"http://127.0.0.1:9/ping"], [b"CONNECT", b"127.0.0.1:9"])
                )

    async def test_a_queued_read_is_dropped_at_its_deadline(self):
        # One InfluxDB thread, held by another read: this one waits in the queue, and
        # once its action has given up on it, it must never run
        self._short_limits(timeout_s=(1, 1), deadline_s=0.3)
        self.enterContext(patch.object(retrieve_hass, "INFLUXDB_CUT_GRACE_S", 0.1))
        executor = ThreadPoolExecutor(max_workers=1)
        self.enterContext(patch.object(retrieve_hass, "_influx_executor", executor))
        release = threading.Event()
        blocker = executor.submit(release.wait, 10)
        self.addCleanup(release.set)
        rh = self._influx_rh(9)
        with patch.object(rh, "get_data_influxdb", return_value=True) as read:
            with self.assertLogs(logger, level="ERROR"):
                self.assertFalse(await rh.get_data(self.DAYS, ["sensor.x"]))
            release.set()
            blocker.result(timeout=5)
            await asyncio.to_thread(executor.shutdown, wait=True)
        read.assert_not_called()

    async def test_what_an_abandoned_read_logs_later_stays_out_of_the_action_log(self):
        self._short_limits(timeout_s=(1, 1), deadline_s=0.3)
        self.enterContext(patch.object(retrieve_hass, "INFLUXDB_CUT_GRACE_S", 0.1))
        release, done = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        rh = self._influx_rh(9)

        def stuck_read(*_args):
            # A read that does not notice the cut, and logs once it gets going again
            release.wait(10)
            rh.logger.error("LATE LINE from the abandoned read")
            done.set()
            return False

        action_lines, detail_lines = [], []
        for name, lines in ((logger.name, action_lines), ("emhass.database", detail_lines)):
            handler = logging.Handler(logging.DEBUG)
            handler.emit = lambda record, lines=lines: lines.append(record.getMessage())
            target = logging.getLogger(name)
            target.addHandler(handler)
            self.addCleanup(target.removeHandler, handler)
            self.addCleanup(target.setLevel, target.level)
            target.setLevel(logging.DEBUG)
        with patch.object(rh, "get_data_influxdb", side_effect=stuck_read):
            self.assertFalse(await rh.get_data(self.DAYS, ["sensor.x"]))
            release.set()
            self.assertTrue(await asyncio.to_thread(done.wait, 5))
        self.assertFalse(any("LATE LINE" in line for line in action_lines), action_lines)
        self.assertTrue(any("LATE LINE" in line for line in detail_lines), detail_lines)
        # The thread is free again, and what it logs next reaches its action as before
        self.assertFalse(retrieve_hass._abandoned_reads)

    def test_a_server_that_cannot_be_reached_stops_the_read(self):
        # No query on the default measurement, and no next sensor, once the server has
        # failed to answer
        import requests

        rh = self._influx_rh(9)
        rh._measurement_cache = {}  # as _init_influx_client leaves it
        client = MagicMock()
        client.query.side_effect = requests.exceptions.ConnectionError("down")
        with (
            patch.object(rh, "_init_influx_client", return_value=client),
            self.assertLogs(logger, level="ERROR") as cm,
        ):
            self.assertFalse(rh.get_data_influxdb(self.DAYS, ["sensor.a", "sensor.b"]))
        self.assertEqual(client.query.call_count, 1)
        self.assertTrue(any("could not reach" in x for x in cm.output))
        client.close.assert_called_once()

    def test_the_deadline_starts_at_the_first_request(self):
        # Loading the client and building it do not count against the read
        class Ping(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Ping)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        session = retrieve_hass._influx_session(0.3)
        self.addCleanup(session.close)
        time.sleep(0.4)
        url = f"http://127.0.0.1:{server.server_port}/ping"
        self.assertEqual(session.request("GET", url, timeout=(1, 1)).status_code, 204)
        time.sleep(0.4)
        with self.assertRaises(TimeoutError):
            session.request("GET", url, timeout=(1, 1))

    def test_a_deadline_during_the_ping_reads_as_could_not_reach(self):
        rh = self._influx_rh(9)
        with self.assertLogs(logger, level="ERROR") as cm:
            self.assertIsNone(rh._init_influx_client(session=retrieve_hass._influx_session(0)))
        self.assertTrue(cm.output[-1].endswith("could not reach the configured InfluxDB server"))
        self.assertNotIn("Deadline", cm.output[-1])

    def test_a_pool_manager_that_cannot_be_tracked_is_logged_once(self):
        import requests

        self.enterContext(patch.object(retrieve_hass, "_logged_once", set()))
        with self.assertLogs("emhass.database", level="ERROR") as cm:
            for _ in range(2):
                adapter = requests.adapters.HTTPAdapter()
                adapter.poolmanager = object()
                retrieve_hass._influx_session(1).mount("http://", adapter)
        self.assertEqual(len(cm.output), 1)
        self.assertIn("cannot be cut", cm.output[0])

    def _points(self):
        start = self.DAYS[0]
        return [
            [(start + pd.Timedelta(minutes=30 * i)).strftime("%Y-%m-%dT%H:%M:%SZ"), 100.0 + i]
            for i in range(48)
        ]

    def _standin(self, **kwargs):
        server = _InfluxStandin(**kwargs)
        self.addCleanup(server.stop)
        return server

    def test_a_server_that_closes_every_connection_keeps_one_duplicate_open(self):
        # Connection: close, as some reverse proxies answer: a new connection for each
        # query. Each one's duplicate is closed once urllib3 is done with that
        # connection, not when the whole read ends
        server = self._standin(measurements=[f"m{i}" for i in range(60)], close=True)
        made, peak = [], [0]
        real_dup = socket.socket.dup

        def counting_dup(sock):
            duplicate = real_dup(sock)
            made.append(duplicate)
            peak[0] = max(peak[0], sum(d.fileno() != -1 for d in made))
            return duplicate

        rh = self._influx_rh(server.port)
        with patch.object(socket.socket, "dup", counting_dup), self.assertLogs(logger, "DEBUG"):
            rh.get_data_influxdb(self.DAYS, ["sensor.x"])
        self.assertGreater(len(server.queries), 60)
        self.assertGreater(len(made), 60)
        # No more than the connections the pool can hold (10)
        self.assertLessEqual(peak[0], 10)
        self.assertEqual(sum(d.fileno() != -1 for d in made), 0)

    def test_a_connection_urllib3_drops_is_closed_at_once(self):
        # The server sees the client close a connection when urllib3 drops it, not when
        # the read ends
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        self.addCleanup(listener.close)
        saw_close = []

        def serve():
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(2)
                conn.recv(65536)
                conn.sendall(b"HTTP/1.1 204 No Content\r\nX-Influxdb-Version: 1.8\r\n\r\n")
                try:
                    saw_close.append(conn.recv(1) == b"")
                except OSError:
                    saw_close.append(False)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        session = retrieve_hass._influx_session(60)
        self.addCleanup(session.close)
        url = f"http://127.0.0.1:{listener.getsockname()[1]}/ping"
        self.assertEqual(session.get(url, timeout=(1, 1)).status_code, 204)
        for adapter in session.adapters.values():
            adapter.poolmanager.clear()
        thread.join(3)
        self.assertEqual(saw_close, [True])

    async def test_a_body_trickling_after_connection_close_is_cut_at_the_deadline(self):
        # http.client closes such a connection before reading the body: its duplicate
        # must still be there for the cut
        self._short_limits(timeout_s=(1, 1), deadline_s=1)
        server = _TrickleServer(seconds=8, head=_TrickleServer.BODY_TRICKLE)
        self.addCleanup(server.close)
        await self._assert_cut(server, self._influx_rh(server.port))

    def test_a_connection_opened_after_the_cut_is_shut_at_once(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        self.addCleanup(listener.close)
        session = retrieve_hass._influx_session(60)
        self.addCleanup(session.close)
        session.cut()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}/"
        conn = session.get_adapter(url).poolmanager.connection_from_url(url)._new_conn()
        conn.connect()
        self.addCleanup(conn.close)
        # The server sees the client end it at once
        accepted, _ = listener.accept()
        self.addCleanup(accepted.close)
        accepted.settimeout(1)
        try:
            got = accepted.recv(1)
        except TimeoutError:
            got = "not shut"
        except OSError:
            got = b""
        self.assertEqual(got, b"")

    def test_a_socket_that_cannot_be_duplicated_is_still_read(self):
        # A SOCKS socket may raise anything from dup(): the read goes on, uncuttable
        server = self._standin()
        self.enterContext(patch.object(retrieve_hass, "_logged_once", set()))

        def no_dup(_sock):
            raise ValueError("cannot dup")

        session = retrieve_hass._influx_session(60)
        self.addCleanup(session.close)
        url = f"http://127.0.0.1:{server.port}/ping"
        with (
            patch.object(socket.socket, "dup", no_dup),
            self.assertLogs("emhass.database", level="ERROR") as cm,
        ):
            for _ in range(2):
                self.assertEqual(session.get(url, timeout=(1, 1)).status_code, 204)
                for adapter in session.adapters.values():
                    adapter.poolmanager.clear()
        self.assertEqual(len(cm.output), 1)
        self.assertIn("cannot be duplicated", cm.output[0])

    def _tracked_connection(self):
        """(session, a connection of its own pool to a listener on 127.0.0.1, the
        duplicates made from here on)."""
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        self.addCleanup(listener.close)
        session = retrieve_hass._influx_session(60)
        self.addCleanup(session.close)
        url = f"http://127.0.0.1:{listener.getsockname()[1]}/"
        conn = session.get_adapter(url).poolmanager.connection_from_url(url)._new_conn()
        made = []
        real_dup = socket.socket.dup

        def recording_dup(sock):
            duplicate = real_dup(sock)
            made.append(duplicate)
            return duplicate

        self.enterContext(patch.object(socket.socket, "dup", recording_dup))
        return session, conn, made

    def test_a_connection_that_reconnects_in_place_closes_its_old_duplicate(self):
        # urllib3 can open a new socket on the same connection object: the duplicate of
        # the socket it had is closed once that socket is, not kept for the whole read
        session, conn, made = self._tracked_connection()
        conn.connect()
        first = conn.sock
        first.close()
        conn.connect()
        self.addCleanup(conn.close)
        self.assertEqual(len(made), 2)
        self.assertEqual(made[0].fileno(), -1)
        self.assertNotEqual(made[1].fileno(), -1)

    def test_the_session_closes_a_duplicate_still_waiting_for_its_socket(self):
        # A connection closed while its reply is still being read (Connection: close)
        # keeps its duplicate until that socket closes; the session's close ends it
        session, conn, made = self._tracked_connection()
        conn.connect()
        # As http.client leaves it: the reply's file object keeps the socket open
        reply = conn.sock.makefile("rb")
        self.addCleanup(reply.close)
        conn.close()
        self.assertNotEqual(made[0].fileno(), -1)
        session.close()
        self.assertEqual(made[0].fileno(), -1)

    def test_a_socket_that_cannot_be_weakly_referenced_keeps_its_duplicate(self):
        # It cannot tell when that socket closes: the duplicate is kept until the
        # session closes, so a reply still on that socket can still be cut
        session, conn, made = self._tracked_connection()

        def no_weak_reference(_obj):
            raise TypeError("cannot create weak reference")

        self.enterContext(
            patch.object(retrieve_hass, "weakref", SimpleNamespace(ref=no_weak_reference))
        )
        conn.connect()
        first = conn.sock
        self.addCleanup(first.close)
        conn.connect()
        self.addCleanup(conn.close)
        self.assertNotEqual(made[0].fileno(), -1)
        session.cut()
        session.close()
        self.assertEqual(made[0].fileno(), -1)

    def test_a_socket_that_cannot_say_whether_it_is_open_keeps_its_duplicate(self):
        # Its fileno() raises: "cannot tell" is not taken as closed, so the duplicate is
        # kept, through the next connection's check, until the session closes
        session, conn, made = self._tracked_connection()
        conn.connect()
        first = conn.sock
        self.addCleanup(first.close)

        class Unsure:
            def fileno(self):
                raise OSError("cannot tell")

        unsure = Unsure()
        conn.sock = unsure
        conn.connect()
        self.addCleanup(conn.close)
        other = session.get_adapter(f"http://127.0.0.1:{conn.port}/")
        another = other.poolmanager.connection_from_url(f"http://127.0.0.1:{conn.port}/")
        extra = another._new_conn()
        extra.connect()
        self.addCleanup(extra.close)
        self.assertEqual(len(made), 3)
        self.assertNotEqual(made[0].fileno(), -1)
        session.close()
        self.assertEqual(made[0].fileno(), -1)
        self.assertIsNotNone(unsure)

    def test_a_request_ends_the_read_when_the_cut_or_the_deadline_failed_it(self):
        # How a failed request is told apart from one query the server could not answer.
        # The clock is the session's own, held still unless a case moves it
        import requests

        clock = [1000.0]
        self.enterContext(
            patch.object(retrieve_hass, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        )
        url = "http://127.0.0.1:9/query"

        def failing(session, exc, effect=None):
            def request(*_args, **_kwargs):
                if effect is not None:
                    effect(session)
                raise exc

            return request

        def advance(_session):
            clock[0] += 100

        cases = (
            # its read timeout was the time left: the deadline
            ("read timeout at the deadline", 0.5, requests.exceptions.ReadTimeout(), None, True),
            (
                "connect timeout at the deadline",
                0.5,
                requests.exceptions.ConnectTimeout(),
                None,
                True,
            ),
            # a read timeout shorter than the time left: one slow query
            (
                "read timeout before the deadline",
                100,
                requests.exceptions.ReadTimeout(),
                None,
                False,
            ),
            (
                "reset after the deadline passed",
                100,
                requests.exceptions.ConnectionError(),
                advance,
                True,
            ),
            ("reset because of the cut", 100, requests.exceptions.ConnectionError(), "cut", True),
            ("reset", 100, requests.exceptions.ConnectionError(), None, False),
        )
        for name, seconds, exc, effect, ends in cases:
            with self.subTest(case=name):
                session = retrieve_hass._influx_session(seconds)
                self.addCleanup(session.close)
                if effect == "cut":
                    effect = type(session).cut
                with patch.object(requests.Session, "request", failing(session, exc, effect)):
                    expected = retrieve_hass._InfluxDeadlineExceeded if ends else type(exc)
                    with self.assertRaises(expected) as caught:
                        session.request("GET", url, timeout=(5, 5))
                    if not ends:
                        self.assertNotIsInstance(caught.exception, TimeoutError)

    def test_only_a_failure_to_connect_is_taken_as_the_server_down(self):
        # The read ends on these; on anything else the query is skipped
        import requests

        closed = socket.socket()
        closed.bind(("127.0.0.1", 0))
        refused_port = closed.getsockname()[1]
        closed.close()
        silent = _SilentServer(hold_s=2)
        self.addCleanup(silent.close)

        def one_shot(reply: bytes, reset: bool):
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            self.addCleanup(listener.close)

            def serve():
                conn, _ = listener.accept()
                conn.recv(65536)
                conn.sendall(reply)
                if reset:
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                conn.close()

            threading.Thread(target=serve, daemon=True).start()
            return listener.getsockname()[1]

        session = retrieve_hass._influx_session(60)
        self.addCleanup(session.close)
        cases = (
            ("refused", refused_port, True),
            ("read timeout", silent.port, False),
            ("reset", one_shot(b"", reset=True), False),
            (
                "cut off",
                one_shot(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial", reset=False),
                False,
            ),
        )
        for name, port, ends in cases:
            with self.subTest(case=name):
                with self.assertRaises(requests.exceptions.RequestException) as caught:
                    _ = session.get(f"http://127.0.0.1:{port}/query", timeout=(1, 0.3)).content
                self.assertEqual(retrieve_hass._could_not_connect(caught.exception), ends)
                # Each still reads as "could not reach" in the action log
                self.assertTrue(retrieve_hass._unreachable(caught.exception))

    def test_one_slow_query_skips_that_query_not_the_read(self):
        # A read timeout on one tag query or one SELECT: the server answers the rest. A
        # skipped tag query sends the sensor it was looking for to that measurement's
        # SELECT (see the next test), here where it is
        self._short_limits(timeout_s=(1, 0.3), deadline_s=30)
        points = self._points()
        server = self._standin(
            measurements=["W", "kWh"],
            entity_in={"W": ["w"], "kWh": ["e"]},
            slow={'FROM "kWh"': 0.8},
            points={"e": points, "w": points},
            by_measurement=True,
        )
        rh = self._influx_rh(server.port)
        with self.assertLogs(logger, "DEBUG"):
            self.assertTrue(rh.get_data_influxdb(self.DAYS, ["sensor.w", "sensor.e"]))
        self.assertEqual(list(rh.df_final.columns), ["sensor.w", "sensor.e"])
        self.assertEqual(rh._measurement_cache, {"w": "W"})
        server = self._standin(
            measurements=["W"],
            entity_in={"W": ["a", "b"]},
            slow={"'a'": 0.8},
            points={"a": points, "b": points},
        )
        rh = self._influx_rh(server.port)
        with self.assertLogs(logger, "DEBUG"):
            self.assertTrue(rh.get_data_influxdb(self.DAYS, ["sensor.a", "sensor.b"]))
        self.assertEqual(list(rh.df_final.columns), ["sensor.b"])

    @staticmethod
    def _selected_from(server):
        """The measurement of each SELECT the stand-in was sent, in order."""
        return [_selected_measurement(q) for q in server.queries if "SELECT" in q]

    def _assert_read_from_w_only(self, server, rh, sensors, read, cm):
        """``read`` are the sensors read; the rest are skipped with the fixed line. Every
        SELECT is from W, kW is never asked, and nothing is cached."""
        self.assertEqual(list(rh.df_final.columns) if read else [], read)
        self.assertEqual(self._selected_from(server), ["W"] * len(sensors))
        self.assertFalse([q for q in server.queries if 'FROM "kW"' in q])
        self.assertEqual(rh._measurement_cache, {})
        skipped = [
            f"InfluxDB sensor {s.removeprefix('sensor.')!r} skipped: a query looking for its "
            "measurement failed"
            for s in sensors
            if s not in read
        ]
        lines = [r.getMessage() for r in cm.records if "skipped" in r.getMessage()]
        self.assertEqual(lines, skipped)

    def test_a_skipped_tag_query_takes_no_other_measurement(self):
        # A sensor whose unit changed from W to kW is in both. With the tag query on W
        # skipped, kW would be the first match, read with values 1000 times off; the
        # default measurement would be a guess. Neither is taken: W's own SELECT decides,
        # as a tag query that found the sensor there would have. Data there is read;
        # none skips the sensor, with a fixed line. Nothing is cached either way
        self._short_limits(timeout_s=(1, 0.3), deadline_s=30)
        points = self._points()
        for entity_in, read in (
            ({"W": ["e"], "kW": ["e"]}, ["sensor.e"]),
            ({"W": ["e"]}, ["sensor.e"]),
            ({"kW": ["e"]}, []),
        ):
            with self.subTest(entity_in=entity_in):
                server = self._standin(
                    measurements=["W", "kW"],
                    entity_in=entity_in,
                    slow={'SHOW TAG VALUES FROM "W"': 0.8},
                    points={"e": points},
                    by_measurement=True,
                )
                rh = self._influx_rh(server.port)
                with self.assertLogs(logger, "DEBUG") as cm:
                    self.assertEqual(rh.get_data_influxdb(self.DAYS, ["sensor.e"]), bool(read))
                self._assert_read_from_w_only(server, rh, ["sensor.e"], read, cm)

    def test_a_failed_tag_query_takes_no_other_measurement(self):
        # A 5xx, a 408 or 429 (not now), an error inside a 200 (InfluxDB's own query
        # timeout) or a reply cut off with no length (it arrives as JSON that does not
        # parse) may hide the entity as well as no reply does: W's SELECT decides, never
        # kW, which would read it 1000 times off
        self._short_limits(timeout_s=(1, 1), deadline_s=30)
        points = self._points()
        faults = (
            "http500",
            "http504",
            "http408",
            "http429",
            "query timeout",
            "max queries",
            "cut, no length",
        )
        for fault in faults:
            for entity_in, read in (
                ({"W": ["e"], "kW": ["e"]}, ["sensor.e"]),
                ({"kW": ["e"]}, []),
            ):
                with self.subTest(fault=fault, entity_in=entity_in):
                    server = self._standin(
                        measurements=["W", "kW"],
                        entity_in=entity_in,
                        faults={'SHOW TAG VALUES FROM "W"': fault},
                        points={"e": points},
                        by_measurement=True,
                    )
                    rh = self._influx_rh(server.port)
                    with self.assertLogs(logger, "DEBUG") as cm:
                        self.assertEqual(rh.get_data_influxdb(self.DAYS, ["sensor.e"]), bool(read))
                    self._assert_read_from_w_only(server, rh, ["sensor.e"], read, cm)
        # Every tag query failing so: a sensor in W is still read from it; one only in
        # kW, a later measurement, is skipped, not read from W or the default
        for fault in faults:
            with self.subTest(fault=fault, every_tag_query=True):
                server = self._standin(
                    measurements=["W", "kW", "%"],
                    entity_in={"W": ["a"], "kW": ["b"]},
                    faults={"SHOW TAG VALUES": fault},
                    points={"a": points, "b": points},
                    by_measurement=True,
                )
                rh = self._influx_rh(server.port)
                sensors = ["sensor.a", "sensor.b"]
                with self.assertLogs(logger, "DEBUG") as cm:
                    self.assertTrue(rh.get_data_influxdb(self.DAYS, sensors))
                self._assert_read_from_w_only(server, rh, sensors, ["sensor.a"], cm)
                asked = [q for q in server.queries if "SHOW TAG VALUES" in q]
                self.assertEqual(len(asked), 1)
        # A 4xx refuses the query: the next measurement is searched as before; an empty
        # reply is how a measurement without the entity answers
        for faults, entity_in in (
            ({'SHOW TAG VALUES FROM "W"': "http400"}, {"W": ["e"], "kW": ["e"]}),
            ({'SHOW TAG VALUES FROM "W"': "http403"}, {"W": ["e"], "kW": ["e"]}),
            ({}, {"kW": ["e"]}),
        ):
            with self.subTest(faults=faults):
                server = self._standin(
                    measurements=["W", "kW"],
                    entity_in=entity_in,
                    faults=faults,
                    points={"e": points},
                )
                rh = self._influx_rh(server.port)
                with self.assertLogs(logger, "DEBUG"):
                    self.assertTrue(rh.get_data_influxdb(self.DAYS, ["sensor.e"]))
                self.assertEqual(rh._measurement_cache, {"e": "kW"})

    def test_a_failed_tag_query_is_asked_once_per_read(self):
        # Each sensor behind it would wait out the same timeout again: the measurement
        # is remembered for the rest of the read, and asked again by the next read. The
        # slow reply comes well after the read timeout, so a loaded machine still times
        # it out
        self._short_limits(timeout_s=(1, 0.6), deadline_s=30)
        points = self._points()
        for name, how in (
            ("slow", {"slow": {'SHOW TAG VALUES FROM "W"': 1.5}}),
            ("http500", {"faults": {'SHOW TAG VALUES FROM "W"': "http500"}}),
        ):
            with self.subTest(failure=name):
                server = self._standin(
                    measurements=["W", "kW"],
                    entity_in={"kW": ["a", "b", "c"]},
                    points={"a": points, "b": points, "c": points},
                    by_measurement=True,
                    **how,
                )
                rh = self._influx_rh(server.port)
                sensors = ["sensor.a", "sensor.b", "sensor.c"]
                for reads in (1, 2):
                    with self.assertLogs(logger, "DEBUG") as cm:
                        self.assertFalse(rh.get_data_influxdb(self.DAYS, sensors))
                    asked = [q for q in server.queries if 'SHOW TAG VALUES FROM "W"' in q]
                    self.assertEqual(len(asked), reads)
                    skipped = [r.getMessage() for r in cm.records if "skipped" in r.getMessage()]
                    self.assertEqual(len(skipped), 3, skipped)
                    self.assertEqual(self._selected_from(server), ["W"] * 3 * reads)
                self.assertFalse([q for q in server.queries if 'FROM "kW"' in q])

    def test_a_tag_query_the_server_refuses_is_passed_over(self):
        # A 4xx says the server refuses the query: the next measurement is searched as
        # before, and the default is still the fallback for a server that refuses all.
        # On any other failure the failed measurement is given for its SELECT to decide,
        # never a later one, and nothing is cached
        import requests
        from influxdb.exceptions import InfluxDBClientError, InfluxDBServerError

        def shown(*names):
            result = MagicMock()
            result.get_points.return_value = [{"name": n} for n in names]
            return result

        def tags(*entities):
            result = MagicMock()
            result.get_points.return_value = [{"key": "entity_id", "value": e} for e in entities]
            return result

        refused = InfluxDBClientError("error parsing query", 400)
        forbidden = InfluxDBClientError("error authorizing query", 403)
        for answers, found in (
            ([shown("W", "kW"), refused, tags("e")], "kW"),
            ([shown("W", "kW"), forbidden, tags("e")], "kW"),
            ([shown("W", "kW"), refused, refused], "W"),
            ([shown("W", "kW"), requests.exceptions.ReadTimeout("slow")], "W"),
            ([shown("W", "kW"), requests.exceptions.ChunkedEncodingError("cut")], "W"),
            ([shown("W", "kW"), InfluxDBServerError("timeout")], "W"),
            ([shown("W", "kW"), InfluxDBClientError("request time-out", 408)], "W"),
            ([shown("W", "kW"), InfluxDBClientError("too early", 425)], "W"),
            ([shown("W", "kW"), InfluxDBClientError("too many requests", 429)], "W"),
            ([shown("W", "kW"), InfluxDBClientError("query-timeout limit exceeded")], "W"),
            ([shown("W", "kW"), requests.exceptions.JSONDecodeError("cut", "{", 1)], "W"),
            ([shown("W", "kW"), KeyError("value")], "W"),
        ):
            with self.subTest(answers=answers[1:], found=found):
                rh = self._influx_rh(9)
                rh._measurement_cache = {}  # as _init_influx_client leaves it
                client = MagicMock()
                client.query.side_effect = answers
                self.assertEqual(rh._discover_entity_measurement(client, "e"), found)
                self.assertEqual(client.query.call_count, len(answers))
                self.assertEqual(rh._measurement_cache, {"e": found} if found == "kW" else {})

    def test_a_failure_to_connect_on_any_query_ends_the_read(self):
        import requests
        from urllib3.exceptions import MaxRetryError, NewConnectionError

        refused = requests.exceptions.ConnectionError(
            MaxRetryError(None, "/query", NewConnectionError(None, "refused"))
        )
        shown = MagicMock()
        shown.get_points.return_value = [{"name": "W"}]
        tags = MagicMock()
        tags.get_points.return_value = [{"key": "entity_id", "value": "a"}]
        for site, answers in (("tag query", [shown, refused]), ("select", [shown, tags, refused])):
            with self.subTest(site=site):
                rh = self._influx_rh(9)
                rh._measurement_cache = {}  # as _init_influx_client leaves it
                client = MagicMock()
                client.query.side_effect = answers
                with (
                    patch.object(rh, "_init_influx_client", return_value=client),
                    self.assertLogs(logger, level="ERROR") as cm,
                ):
                    self.assertFalse(rh.get_data_influxdb(self.DAYS, ["sensor.a", "sensor.b"]))
                # Nothing after the failure: no default measurement, no next sensor
                self.assertEqual(client.query.call_count, len(answers))
                self.assertTrue(any("could not reach" in x for x in cm.output))

    def test_a_query_that_runs_into_the_deadline_ends_the_read(self):
        # Its read timeout was the time left: the deadline, not one slow query
        self._short_limits(timeout_s=(2, 2), deadline_s=0.6)
        points = self._points()
        server = self._standin(
            measurements=["W"],
            entity_in={"W": ["a", "b"]},
            slow={"'b'": 3},
            points={"a": points, "b": points},
        )
        rh = self._influx_rh(server.port)
        with self.assertLogs(logger, level="ERROR") as cm:
            self.assertFalse(rh.get_data_influxdb(self.DAYS, ["sensor.a", "sensor.b"]))
        self.assertTrue(any("took longer than" in x for x in cm.output), cm.output)

    async def test_a_cancel_during_the_grace_wait_still_lets_the_read_go(self):
        self._short_limits(timeout_s=(1, 1), deadline_s=0.2)
        self.enterContext(patch.object(retrieve_hass, "INFLUXDB_CUT_GRACE_S", 5))
        release, done = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        rh = self._influx_rh(9)

        def stuck_read(*_args):
            release.wait(10)
            rh.logger.error("LATE LINE from the cancelled read")
            done.set()
            return False

        action_lines, detail_lines = self._capture_lines(logger.name, "emhass.database")
        with patch.object(rh, "get_data_influxdb", side_effect=stuck_read):
            task = asyncio.create_task(rh.get_data(self.DAYS, ["sensor.x"]))
            # Past the deadline and its margin, inside the grace wait
            await asyncio.sleep(1.2)
            self.assertFalse(task.done())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            release.set()
            self.assertTrue(await asyncio.to_thread(done.wait, 5))
        self.assertTrue(any("the action was cancelled" in x for x in action_lines), action_lines)
        self.assertFalse(any("LATE LINE" in x for x in action_lines), action_lines)
        self.assertTrue(any("LATE LINE" in x for x in detail_lines), detail_lines)

    async def test_an_abandoned_read_on_the_database_logger_itself_does_not_loop(self):
        self._short_limits(timeout_s=(1, 1), deadline_s=0.2)
        self.enterContext(patch.object(retrieve_hass, "INFLUXDB_CUT_GRACE_S", 0.1))
        db_logger = logging.getLogger("emhass.database")
        self.addCleanup(db_logger.removeFilter, retrieve_hass._abandoned_read_filter)
        release, done = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        rh = self._influx_rh(9)
        rh.logger = db_logger

        def stuck_read(*_args):
            release.wait(10)
            rh.logger.error("LATE LINE on the database logger")
            done.set()
            return False

        (detail_lines,) = self._capture_lines("emhass.database")
        with patch.object(rh, "get_data_influxdb", side_effect=stuck_read):
            self.assertFalse(await rh.get_data(self.DAYS, ["sensor.x"]))
            release.set()
            self.assertTrue(await asyncio.to_thread(done.wait, 5))
        self.assertEqual(sum("LATE LINE" in x for x in detail_lines), 1, detail_lines)

    def _capture_lines(self, *names):
        """The messages each named logger handles from here on, one list per name."""
        captured = []
        for name in names:
            lines = []
            handler = logging.Handler(logging.DEBUG)
            handler.emit = lambda record, lines=lines: lines.append(record.getMessage())
            target = logging.getLogger(name)
            target.addHandler(handler)
            self.addCleanup(target.removeHandler, handler)
            self.addCleanup(target.setLevel, target.level)
            target.setLevel(logging.DEBUG)
            captured.append(lines)
        return captured

    async def test_the_victoriametrics_session_has_a_timeout(self):
        rh = self._rh({"use_victoriametrics": True, "victoriametrics_host": "127.0.0.1"})
        with (
            patch.object(rh, "_fetch_sensor_data_vm", AsyncMock(return_value=None)),
            patch(
                "emhass.retrieve_hass.aiohttp.ClientSession", wraps=aiohttp.ClientSession
            ) as session_class,
        ):
            await rh.get_data_victoriametrics(self.DAYS, ["sensor.x"])
        timeout = session_class.call_args.kwargs["timeout"]
        self.assertEqual((timeout.total, timeout.connect, timeout.sock_read), (60, 10, 30))

    async def test_a_silent_victoriametrics_server_times_out_cleanly(self):
        self._short_limits()
        server = _SilentServer(hold_s=2)
        self.addCleanup(server.close)
        rh = self._rh(
            {
                "use_victoriametrics": True,
                "victoriametrics_host": "127.0.0.1",
                "victoriametrics_port": server.port,
            }
        )
        start = time.monotonic()
        with self.assertLogs(logger, level="ERROR") as cm:
            self.assertFalse(await rh.get_data(self.DAYS, ["sensor.x"]))
        self.assertLess(time.monotonic() - start, 1.5)
        line = next(x for x in cm.output if "Network error" in x)
        self.assertIn("could not reach the configured VictoriaMetrics server", line)
        self.assertNotIn(str(server.port), line)


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
