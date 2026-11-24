"""Security hardening: runtime parameter validation, config-only keys, request checks,
and the Supervisor token guard."""

import asyncio
import datetime
import html
import itertools
import logging
import os
import pathlib
import pickle
import re
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pandas as pd
import pytz

from emhass import api_auth, command_line, retrieve_hass, utils, web_server
from emhass.retrieve_hass import RetrieveHass, _is_supervisor_url
from emhass.utils import RUNTIME_DENIED_KEYS, RuntimeParamError
from tests.aioresponses_compat import aioresponses
from tests.api_auth_support import ADMIN, BEARER, SUPERVISOR, AuthSupportBase

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["options_path"] = root / "options.json"
emhass_conf["config_path"] = root / "config.json"
emhass_conf["secrets_path"] = root / "secrets_emhass(example).yaml"
emhass_conf["legacy_config_path"] = (
    pathlib.Path(utils.get_root(__file__, num_parent=1)) / "config_emhass.yaml"
)
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)


class TestFileNameValidation(unittest.TestCase):
    def test_plain_names_are_accepted(self):
        for value in ("my export 2024.csv", "Load_Forecast.v2", "a.csv", "x" * 128):
            with self.subTest(value=value):
                self.assertEqual(utils.validate_runtime_file_name("model_type", value), value)

    def test_unsafe_names_are_rejected(self):
        for value in (
            "a/b.csv",
            "..\\x.csv",
            "/etc/passwd",
            "C:\\x.csv",
            "C:/x.csv",
            ".hidden.csv",
            "a..b.csv",
            "bad\nname.csv",
            "bad\x00name.csv",
            "bad\x7fname.csv",
            "",
            "x" * 129,
            None,
            5,
            ["a.csv"],
        ):
            with self.subTest(value=value), self.assertRaises(RuntimeParamError):
                utils.validate_runtime_file_name("csv_file", value)

    def test_csv_suffix(self):
        self.assertEqual(
            utils.validate_runtime_file_name("csv_filename", "OUT.CSV", ".csv"), "OUT.CSV"
        )
        with self.assertRaises(RuntimeParamError) as ctx:
            utils.validate_runtime_file_name("csv_filename", "poc_mlr.pkl", ".csv")
        self.assertIn("csv_filename", str(ctx.exception))

    def test_entity_ids(self):
        for value in ("sensor.p_pv_forecast", "sensor.P_PV", "number.x_1", "sensor.a" + "b" * 189):
            with self.subTest(value=value):
                self.assertTrue(utils.is_valid_entity_id(value))
        for value in (
            "../events/emhass_poc",
            "sensor.x/../../events/y",
            "sensor.x?y=1",
            "sensor.x#y",
            "sensor.x y",
            "sensor",
            ".x",
            "sensor.",
            "sensor.a.b",
            "sensor.a-b",
            "sensor.x\n",
            "sensor." + "b" * 191,
            "",
            None,
            5,
            ["sensor.x"],
        ):
            with self.subTest(value=value):
                self.assertFalse(utils.is_valid_entity_id(value))

    def test_data_file_path_stays_inside_the_data_folder(self):
        base = pathlib.Path(emhass_conf["data_path"])
        for name in ("opt_res_latest.csv", "my export 2026.csv", "entities/sensor.x.json"):
            with self.subTest(name=name):
                self.assertEqual(
                    utils.data_file_path(base, name), pathlib.Path(os.path.realpath(base)) / name
                )
        outside = [
            "opt_res_perfect_optim_/../../tmp/x.csv",
            "../x.csv",
            "a/../../x.csv",
            "/etc/passwd",
            "",
            ".",
            "x\x00.csv",
            # Names Windows reads as "." or "..": refused everywhere
            ". ",
            "...",
            ".. ",
            "entities/.. ",
        ]
        if os.name == "nt":
            # On POSIX this is a legal file name inside the folder
            outside += ["C:\\x.csv"]
        for name in outside:
            with self.subTest(name=name), self.assertRaises(RuntimeParamError):
                utils.data_file_path(base, name)

    def test_data_file_path_follows_a_file_link_but_not_a_folder_link(self):
        # A results file may be a symlink to elsewhere on purpose (atomic_write replaces
        # its target); a folder that links out of the data folder is refused.
        with tempfile.TemporaryDirectory() as tmp:
            data = pathlib.Path(tmp) / "data"
            outside = pathlib.Path(tmp) / "outside"
            data.mkdir()
            outside.mkdir()
            (outside / "target.csv").write_text("x")
            try:
                os.symlink(outside / "target.csv", data / "opt_res_latest.csv")
                os.symlink(outside, data / "sub", target_is_directory=True)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"cannot create symlinks here: {exc}")
            path = utils.data_file_path(data, "opt_res_latest.csv")
            self.assertEqual(path, pathlib.Path(os.path.realpath(data)) / "opt_res_latest.csv")
            self.assertTrue(path.is_symlink())
            with self.assertRaises(RuntimeParamError):
                utils.data_file_path(data, "sub/x.csv")

    def test_data_file_path_refuses_a_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            (pathlib.Path(tmp) / "entities").mkdir()
            with self.assertRaises(RuntimeParamError):
                utils.data_file_path(tmp, "entities")
            # Positive control: a file in that folder is fine
            self.assertEqual(
                utils.data_file_path(tmp, "entities/sensor.x.json"),
                pathlib.Path(os.path.realpath(tmp)) / "entities" / "sensor.x.json",
            )

    @unittest.skipUnless(os.name == "nt", "Windows junctions")
    def test_data_file_path_refuses_a_junction_out_of_the_data_folder(self):
        # Junctions need no privilege, so this runs on any Windows machine
        import _winapi

        with tempfile.TemporaryDirectory() as tmp:
            data = pathlib.Path(tmp) / "data"
            outside = pathlib.Path(tmp) / "outside"
            data.mkdir()
            outside.mkdir()
            _winapi.CreateJunction(str(outside), str(data / "jout"))
            for name in ("jout/x.csv", "jout"):
                with self.subTest(name=name), self.assertRaises(RuntimeParamError):
                    utils.data_file_path(data, name)
            # Through the junction and back again stays inside
            self.assertEqual(
                utils.data_file_path(data, "jout/../x.csv"),
                pathlib.Path(os.path.realpath(data)) / "x.csv",
            )

    def test_db_host(self):
        for value in ("localhost", "192.0.2.10", "a0d7b954-victoriametrics", "[::1]", "my_db"):
            with self.subTest(value=value):
                self.assertTrue(utils.is_valid_db_host(value))
        for value in (
            "127.0.0.1:9/s/txt?x=",
            "http://db",
            "user@db",
            "db#x",
            "db x",
            "db\n",
            "",
            None,
            8086,
        ):
            with self.subTest(value=value):
                self.assertFalse(utils.is_valid_db_host(value))

    def test_config_value_faults(self):
        self.assertEqual(
            utils.config_value_faults(
                {"costfun": "self-consumption", "influxdb_host": "db", "model_type": "load"}
            ),
            [],
        )
        faults = utils.config_value_faults(
            {
                "costfun": "/../../tmp/x",
                "influxdb_host": "127.0.0.1:9/s?x=",
                "victoriametrics_host": "http://vm",
                "model_type": "../../x",
            }
        )
        self.assertEqual(len(faults), 4)
        for key in ("costfun", "influxdb_host", "victoriametrics_host", "model_type"):
            self.assertTrue(any(key in fault for fault in faults), key)

    def test_db_ports(self):
        for value in (8086, "8428", 1, 65535):
            with self.subTest(value=value):
                self.assertTrue(utils.is_valid_db_port(value))
        for value in ("9/s/txt?x=", "80 ", "", 0, 65536, -1, True, None, 8086.5, "８０"):
            with self.subTest(value=value):
                self.assertFalse(utils.is_valid_db_port(value))
        faults = utils.config_value_faults(
            {"use_victoriametrics": True, "victoriametrics_port": "9/s/txt?x="}
        )
        self.assertEqual(faults, ["victoriametrics_port must be a port number from 1 to 65535"])

    def test_cleared_db_fields_are_fine_while_the_integration_is_off(self):
        # The web UI sends "" for a cleared text field
        cleared = {"influxdb_host": "", "victoriametrics_host": "", "influxdb_port": ""}
        self.assertEqual(utils.config_value_faults(cleared), [])
        self.assertEqual(len(utils.config_value_faults({**cleared, "use_influxdb": True})), 2)

    def test_timestamp_col_name(self):
        for value in ("timestamp", "Time stamp-1.x", "a" * 64):
            with self.subTest(value=value):
                self.assertEqual(utils.validate_timestamp_col_name(value), value)
        for value in ("a\x02b", "timestamp\n", "", "a" * 65, "a;b", "a/b", None, 3):
            with self.subTest(value=value), self.assertRaises(RuntimeParamError):
                utils.validate_timestamp_col_name(value)


class TestTreatRuntimeparams(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
        secrets["Altitude"] = 8000.0
        params = await utils.build_params(emhass_conf, secrets, config, logger)
        params["passed_data"] = {}
        self.params_json = orjson.dumps(params).decode("utf-8")

    async def _treat(self, runtimeparams: dict, set_type: str):
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(self.params_json, logger)
        return await utils.treat_runtimeparams(
            orjson.dumps(runtimeparams).decode("utf-8"),
            self.params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            set_type,
            logger,
            emhass_conf,
        )

    async def test_denied_keys_are_ignored_with_one_warning_each(self):
        base = orjson.loads(self.params_json)
        runtime = {
            "influxdb_host": "attacker.example",
            "influxdb_password": "stolen",
            "use_victoriametrics": True,
            "victoriametrics_host": "attacker.example",
            "use_websocket": True,
            "hass_url": "http://attacker.example:8123/",
            "long_lived_token": "attacker-token",
        }
        with self.assertLogs(logger, level="WARNING") as cm:
            params, retrieve_hass_conf, *_ = await self._treat(runtime, "dayahead-optim")
        params = orjson.loads(params)
        conf = params["retrieve_hass_conf"]
        for key in ("hass_url", "long_lived_token"):
            with self.subTest(key=key):
                self.assertEqual(conf.get(key), base["retrieve_hass_conf"].get(key))
                self.assertEqual(retrieve_hass_conf.get(key), base["retrieve_hass_conf"].get(key))
                self.assertEqual(
                    params.get("params_secrets", {}).get(key),
                    base.get("params_secrets", {}).get(key),
                )
        self.assertEqual(conf.get("influxdb_host"), base["retrieve_hass_conf"].get("influxdb_host"))
        self.assertEqual(
            conf.get("victoriametrics_host"), base["retrieve_hass_conf"].get("victoriametrics_host")
        )
        self.assertEqual(conf.get("use_websocket"), base["retrieve_hass_conf"].get("use_websocket"))
        self.assertNotEqual(conf.get("influxdb_password"), "stolen")
        self.assertNotEqual(params.get("params_secrets", {}).get("influxdb_password"), "stolen")
        warned = [m for m in cm.output if "only be set in the configuration" in m]
        self.assertEqual(len(warned), len(runtime))

    async def test_use_influxdb_is_allowed_at_runtime(self):
        # The web UI's export passes it. It picks the configured server, which stays fixed.
        base = orjson.loads(self.params_json)
        flipped = not base["retrieve_hass_conf"].get("use_influxdb", False)
        params, *_ = await self._treat(
            {"use_influxdb": flipped, "influxdb_host": "attacker.example"}, "dayahead-optim"
        )
        conf = orjson.loads(params)["retrieve_hass_conf"]
        self.assertEqual(conf.get("use_influxdb"), flipped)
        self.assertEqual(conf.get("influxdb_host"), base["retrieve_hass_conf"].get("influxdb_host"))

    async def test_a_runtime_use_influxdb_that_is_not_true_or_false_is_ignored(self):
        # It would reach the reader, which logs such a value once, for a hand-written
        # config.json: a request must not be the one that logs it
        base = orjson.loads(self.params_json)
        utils._db_settings_warned.clear()
        for value in ("true", 1, "x"):
            with self.subTest(value=value):
                with self.assertLogs(logger, level="WARNING") as cm:
                    params, *_ = await self._treat({"use_influxdb": value}, "dayahead-optim")
                conf = orjson.loads(params)["retrieve_hass_conf"]
                self.assertEqual(
                    conf.get("use_influxdb"), base["retrieve_hass_conf"].get("use_influxdb")
                )
                self.assertTrue(
                    any("use_influxdb ignored: it must be true or false" in m for m in cm.output)
                )
        self.assertNotIn("use_influxdb", utils._db_settings_warned)

    async def test_denied_keys_cover_every_connection_key_in_associations(self):
        lines = (emhass_conf["associations_path"]).read_text(encoding="utf-8").splitlines()
        names = {line.split(",")[2] for line in lines[1:] if line}
        expected = {n for n in names if "influx" in n or "victoria" in n}
        expected |= {"use_websocket", "hass_url", "long_lived_token", "ssl_no_verify"}
        expected.discard("use_influxdb")
        self.assertLessEqual(expected, RUNTIME_DENIED_KEYS, expected - RUNTIME_DENIED_KEYS)
        self.assertNotIn("use_influxdb", RUNTIME_DENIED_KEYS)
        self.assertGreaterEqual(len(RUNTIME_DENIED_KEYS), 23)

    async def test_runtime_entity_ids_and_publish_prefix(self):
        good = {
            "custom_pv_forecast_id": {
                "entity_id": "sensor.P_PV_forecast",
                "unit_of_measurement": "W",
                "friendly_name": "PV <forecast>",
            },
            "custom_deferrable_forecast_id": [
                {"entity_id": "sensor.p_deferrable0"},
                {"entity_id": "sensor.p_deferrable1"},
            ],
            "model_predict_entity_id": "sensor.p_load_forecast_custom_model",
            "mlr_predict_entity_id": "sensor.mlr_predict",
            "publish_prefix": "my_",
        }
        params, *_ = await self._treat(good, "dayahead-optim")
        passed = orjson.loads(params)["passed_data"]
        self.assertEqual(passed["custom_pv_forecast_id"]["entity_id"], "sensor.P_PV_forecast")
        self.assertEqual(passed["publish_prefix"], "my_")
        bad_cases = (
            ("custom_pv_forecast_id", {"custom_pv_forecast_id": {"entity_id": "../events/x"}}),
            (
                "custom_deferrable_forecast_id",
                {
                    "custom_deferrable_forecast_id": [
                        {"entity_id": "sensor.ok"},
                        {"entity_id": "sensor.a/../b"},
                    ]
                },
            ),
            ("custom_cost_fun_id", {"custom_cost_fun_id": {"entity_id": None}}),
            ("model_predict_entity_id", {"model_predict_entity_id": "sensor.x?y"}),
            ("mlr_predict_entity_id", {"mlr_predict_entity_id": "../../config"}),
            ("publish_prefix", {"publish_prefix": "../"}),
            ("publish_prefix", {"publish_prefix": 5}),
            ("publish_prefix", {"publish_prefix": None}),
            # Only sensor entities may be published to at runtime
            ("custom_pv_forecast_id", {"custom_pv_forecast_id": {"entity_id": "switch.garage"}}),
            (
                "custom_optim_status_id",
                {"custom_optim_status_id": {"entity_id": "binary_sensor.front_door"}},
            ),
            ("mlr_predict_entity_id", {"mlr_predict_entity_id": "input_number.setpoint"}),
            (
                "custom_deferrable_state_id",
                {"custom_deferrable_state_id": [{"entity_id": "alarm_control_panel.home"}]},
            ),
            # Each is fine alone, but together they make an id that post_data would refuse
            (
                "custom_pv_forecast_id",
                {
                    "publish_prefix": "p" * 64,
                    "custom_pv_forecast_id": {"entity_id": "sensor." + "x" * 130},
                },
            ),
        )
        for key, runtime in bad_cases:
            with self.subTest(runtime=runtime):
                with self.assertRaises(RuntimeParamError) as ctx:
                    await self._treat(runtime, "dayahead-optim")
                self.assertIn(key, str(ctx.exception))

    async def test_costfun_is_one_of_the_known_values(self):
        for value in ("profit", "cost", "self-consumption"):
            with self.subTest(value=value):
                _, _, optim_conf, _ = await self._treat({"costfun": value}, "perfect-optim")
                self.assertEqual(optim_conf["costfun"], value)
        for value in ("/../../tmp/review_costfun_poc", "Profit", "x", 5):
            with self.subTest(value=value), self.assertRaises(RuntimeParamError) as ctx:
                await self._treat({"costfun": value}, "perfect-optim")
            self.assertIn("costfun", str(ctx.exception))

    async def test_model_predict_publish_is_true_or_false(self):
        for value, expected in ((True, True), ("true", True), ("False", False), (False, False)):
            with self.subTest(value=value):
                params, *_ = await self._treat({"model_predict_publish": value}, "dayahead-optim")
                params = params if isinstance(params, dict) else orjson.loads(params)
                self.assertIs(params["passed_data"]["model_predict_publish"], expected)
        # The parser's error would quote the value capitalized: a mixed-case
        # secret sent here by mistake, changed so the log mask cannot match it.
        secret = "9Bcdefghij0123456789KLMNOPQRSTUVWXYZ_-abcde"
        for value in (secret, "yes", "1 +"):
            with self.subTest(value=value):
                with self.assertRaises(RuntimeParamError) as ctx:
                    await self._treat({"model_predict_publish": value}, "dayahead-optim")
                self.assertEqual(str(ctx.exception), "model_predict_publish must be true or false")
                self.assertIsNone(ctx.exception.__cause__)
                self.assertTrue(ctx.exception.__suppress_context__)

    async def test_bad_def_current_state_is_not_quoted(self):
        with self.assertLogs(logger, level="WARNING") as logs:
            await self._treat({"def_current_state": "[Hidden_Value_0123456789"}, "dayahead-optim")
        text = "\n".join(logs.output)
        self.assertIn("def_current_state", text)
        self.assertIn("24 characters", text)
        self.assertNotIn("hidden", text.lower())

    async def test_bad_per_load_numbers_are_not_quoted(self):
        for name in (
            "def_current_on_timesteps",
            "def_current_off_timesteps",
            "def_current_power",
            "def_current_operating_timesteps",
        ):
            for value in ("[Hidden_Value_0123456789", "Hidden_Value_0123456789", ["Hidden_Value"]):
                with self.subTest(name=name, value=value):
                    with self.assertLogs(logger, level="DEBUG") as logs:
                        logger.debug("start")
                        with self.assertRaises(RuntimeParamError) as ctx:
                            await self._treat({name: value}, "dayahead-optim")
                    self.assertEqual(
                        str(ctx.exception), f"{name} must be a number or a list of numbers"
                    )
                    self.assertIsNone(ctx.exception.__cause__)
                    self.assertNotIn("hidden", "\n".join(logs.output).lower())

    async def test_per_load_numbers_still_parse(self):
        for name, cast in (("def_current_on_timesteps", int), ("def_current_power", float)):
            for value, expected in (("[1, 2]", [1, 2]), ([1, 2], [1, 2]), (3, [3]), ("3", [3])):
                with self.subTest(name=name, value=value):
                    params, *_ = await self._treat({name: value}, "dayahead-optim")
                    params = params if isinstance(params, dict) else orjson.loads(params)
                    got = params["optim_conf"][name]
                    self.assertEqual(got[: len(expected)], [cast(v) for v in expected])
                    self.assertTrue(all(isinstance(v, cast) for v in got))

    async def test_per_load_list_does_not_need_the_load_count(self):
        # A runtime nominal_power_of_deferrable_loads can arrive as one number.
        params, *_ = await self._treat(
            {"nominal_power_of_deferrable_loads": 3000, "def_current_power": [1, 2]},
            "dayahead-optim",
        )
        params = params if isinstance(params, dict) else orjson.loads(params)
        self.assertEqual(params["optim_conf"]["def_current_power"], [1.0, 2.0])

    async def test_saved_costfun_and_model_type_are_checked_too(self):
        # A value stored in the configuration gets the same check as a runtime one
        for key, value, conf in (
            ("costfun", "/../../tmp/x", "optim_conf"),
            ("model_type", "../../ESCAPED_model", "optim_conf"),
        ):
            saved = orjson.loads(self.params_json)
            saved[conf][key] = value
            saved_json = orjson.dumps(saved).decode()
            retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(saved_json, logger)
            with self.subTest(key=key):
                with self.assertRaises(RuntimeParamError) as ctx:
                    await utils.treat_runtimeparams(
                        "{}",
                        saved_json,
                        retrieve_hass_conf,
                        optim_conf,
                        plant_conf,
                        "regressor-model-fit",
                        logger,
                        emhass_conf,
                    )
                self.assertIn(key, str(ctx.exception))

    async def test_results_file_cannot_leave_the_data_folder(self):
        data_path = pathlib.Path(emhass_conf["data_path"])
        input_data_dict = {"emhass_conf": {"data_path": data_path}}
        with self.assertRaises(RuntimeParamError):
            command_line._write_results(
                input_data_dict,
                pd.DataFrame({"x": [1.0]}),
                "opt_res_perfect_optim_/../../review_costfun_poc.csv",
            )
        # data/opt_res_perfect_optim_/../../x resolves to the data folder's parent
        self.assertFalse((data_path.parent / "review_costfun_poc.csv").exists())

    # The checks must run before anything connects out or writes a file
    @patch("emhass.command_line.RetrieveHass")
    async def test_export_direct_mode_rejects_before_connecting(self, mock_rh):
        data_path = pathlib.Path(emhass_conf["data_path"])
        for runtime in (
            {"csv_filename": "poc_mlr.pkl"},
            {"csv_filename": "poc.csv", "timestamp_col_name": "a\x02b"},
        ):
            runtime = {
                "sensor_list": ["sensor.a"],
                "start_time": "2024-01-01",
                "use_influxdb": True,
                **runtime,
            }
            with self.subTest(runtime=runtime), self.assertRaises(RuntimeParamError):
                await command_line.export_influxdb_to_csv(
                    None, logger, emhass_conf, self.params_json, runtime
                )
        mock_rh.assert_not_called()
        self.assertFalse((data_path / "poc_mlr.pkl").exists())
        self.assertFalse((data_path / "poc.csv").exists())

    @patch("emhass.command_line.RetrieveHass")
    async def test_set_input_data_dict_rejects_before_connecting(self, mock_rh):
        for set_type, runtime in (
            ("regressor-model-predict", {"model_type": "../x", "new_values": [1]}),
            ("regressor-model-fit", {"csv_file": "../data.csv"}),
            ("forecast-model-fit", {"model_type": "a/b"}),
            ("dayahead-optim", {"model_type": ".x"}),
            ("publish-data", {"custom_pv_forecast_id": {"entity_id": "../events/sensor.x"}}),
            ("publish-data", {"publish_prefix": "/../../events/"}),
        ):
            with self.subTest(set_type=set_type):
                runtime_json = orjson.dumps(runtime).decode("utf-8")
                with self.assertRaises(RuntimeParamError):
                    await command_line.set_input_data_dict(
                        emhass_conf,
                        "profit",
                        self.params_json,
                        runtime_json,
                        set_type,
                        logger,
                    )
        mock_rh.assert_not_called()

    @patch("emhass.command_line.Forecast")
    async def test_weather_forecast_cache_rejects_before_fetching(self, mock_fcst):
        runtime_json = orjson.dumps({"model_type": "../x"}).decode()
        with self.assertRaises(RuntimeParamError):
            await command_line.weather_forecast_cache(
                emhass_conf, self.params_json, runtime_json, logger
            )
        mock_fcst.assert_not_called()

    async def test_export_csv_filename_and_timestamp_col_name(self):
        sensors = ["sensor.a"]
        ok, *_ = await self._treat(
            {
                "sensor_list": sensors,
                "csv_filename": "my export 2024.csv",
                "timestamp_col_name": "time stamp",
            },
            "export-influxdb-to-csv",
        )
        passed = orjson.loads(ok)["passed_data"]
        self.assertEqual(passed["csv_filename"], "my export 2024.csv")
        self.assertEqual(passed["timestamp_col_name"], "time stamp")
        for bad in ("poc_mlr.pkl", "a/b.csv", "..\\x.csv", ".hidden.csv", "x\n.csv"):
            with self.subTest(bad=bad), self.assertRaises(RuntimeParamError):
                await self._treat(
                    {"sensor_list": sensors, "csv_filename": bad}, "export-influxdb-to-csv"
                )
        for bad in ("a\x02b", "timestamp\n"):
            with self.subTest(bad=bad), self.assertRaises(RuntimeParamError):
                await self._treat(
                    {"sensor_list": sensors, "csv_filename": "x.csv", "timestamp_col_name": bad},
                    "export-influxdb-to-csv",
                )

    async def test_regressor_csv_file_and_model_type(self):
        for set_type in ("regressor-model-fit", "regressor-model-predict"):
            with self.subTest(set_type=set_type):
                params, *_ = await self._treat({"csv_file": "data.csv"}, set_type)
                self.assertEqual(orjson.loads(params)["passed_data"]["csv_file"], "data.csv")
                with self.assertRaises(RuntimeParamError):
                    await self._treat({"csv_file": "../data.csv"}, set_type)
        params, *_ = await self._treat({"model_type": "Load_Forecast.v2"}, "forecast-model-fit")
        self.assertEqual(orjson.loads(params)["passed_data"]["model_type"], "Load_Forecast.v2")
        for bad in ("../evil", "a/b", ".x", "a\nb"):
            with self.subTest(bad=bad), self.assertRaises(RuntimeParamError):
                await self._treat({"model_type": bad}, "forecast-model-fit")


class TestPlotTitlesAreEscaped(unittest.TestCase):
    MODEL_TYPE = '<img src=x onerror="alert(1)">'
    VAR_MODEL = "sensor.x<script>alert(2)</script>"

    def _mlf(self):
        return SimpleNamespace(model_type=self.MODEL_TYPE, var_model=self.VAR_MODEL)

    def _df(self):
        index = pd.date_range("2024-01-01", periods=3, freq="30min", tz="UTC")
        return pd.DataFrame({"train": [1.0, 2.0, 3.0], "test": [1.5, 2.5, 3.5]}, index=index)

    def test_fit_and_tune_titles(self):
        for builder in (
            utils.get_injection_dict_forecast_model_fit,
            utils.get_injection_dict_forecast_model_tune,
        ):
            with self.subTest(builder=builder.__name__):
                title = builder(self._df(), self._mlf())["subsubtitle0"]
                self.assertNotIn("<img", title)
                self.assertNotIn("<script", title)
                self.assertIn(html.escape(self.MODEL_TYPE), title)
                self.assertIn(html.escape(self.VAR_MODEL), title)
                # The builder's own markup is kept
                self.assertTrue(title.startswith("<h4>"))
                self.assertIn("<br>", title)


class TestRequestChecks(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = web_server.app.test_client()
        self.mock_conf = {
            "data_path": pathlib.Path("/tmp/emhass/data"),
            "config_path": pathlib.Path("/tmp/emhass/config.json"),
            "defaults_path": pathlib.Path("/tmp/emhass/defaults.json"),
            "associations_path": pathlib.Path("/tmp/emhass/assoc.csv"),
            "legacy_config_path": pathlib.Path("/tmp/emhass/legacy.yaml"),
            "root_path": pathlib.Path("/tmp/emhass/root"),
        }
        self.original_conf = web_server.emhass_conf.copy()
        web_server.emhass_conf = self.mock_conf
        self.original_handlers = web_server.app.logger.handlers[:]

    async def asyncTearDown(self):
        web_server.emhass_conf = self.original_conf
        web_server.app.logger.handlers = self.original_handlers

    @patch("emhass.web_server.export_influxdb_to_csv", new_callable=AsyncMock)
    @patch("emhass.web_server._load_params_and_runtime", new_callable=AsyncMock)
    async def test_action_content_type(self, mock_load, mock_export):
        mock_load.return_value = ({}, "profit", "{}")
        mock_export.return_value = True
        url = "/action/export-influxdb-to-csv"
        # A body that is not declared as JSON is refused before anything runs
        for content_type in ("text/plain", "application/x-www-form-urlencoded"):
            with self.subTest(content_type=content_type):
                response = await self.client.post(
                    url, data="{}", headers={"Content-Type": content_type}
                )
                self.assertEqual(response.status_code, 415)
        mock_load.assert_not_called()
        mock_export.assert_not_called()
        # No body: accepted as before, with or without a content type
        for headers in ({}, {"Content-Type": "text/plain"}):
            with self.subTest(empty=headers):
                response = await self.client.post(url, headers=headers)
                self.assertEqual(response.status_code, 200)
        # JSON body
        response = await self.client.post(
            url, data="{}", headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 200)
        response = await self.client.post(url, json={})
        self.assertEqual(response.status_code, 200)

    async def test_set_config_content_type(self):
        text = await self.client.post(
            "/set-config", data='{"a": 1}', headers={"Content-Type": "text/plain"}
        )
        self.assertEqual(text.status_code, 415)
        empty = await self.client.post("/set-config")
        self.assertEqual(empty.status_code, 400)

    @patch("emhass.web_server.export_influxdb_to_csv", new_callable=AsyncMock)
    @patch("emhass.web_server._load_params_and_runtime", new_callable=AsyncMock)
    async def test_cross_site_post_is_refused(self, mock_load, mock_export):
        mock_load.return_value = ({}, "profit", "{}")
        mock_export.return_value = True
        url = "/action/export-influxdb-to-csv"
        response = await self.client.post(url, json={}, headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(response.status_code, 403)
        mock_export.assert_not_called()
        for value in ("same-origin", "same-site", "none"):
            with self.subTest(value=value):
                response = await self.client.post(url, json={}, headers={"Sec-Fetch-Site": value})
                self.assertEqual(response.status_code, 200)
        # No header at all (curl, Home Assistant, scripts)
        response = await self.client.post(url, json={})
        self.assertEqual(response.status_code, 200)
        # Reads are not affected
        response = await self.client.get("/healthz", headers={"Sec-Fetch-Site": "cross-site"})
        self.assertNotEqual(response.status_code, 403)

    async def test_set_config_refuses_unsafe_values(self):
        for key, value in (
            ("costfun", "/../../tmp/review_costfun_poc"),
            ("influxdb_host", "127.0.0.1:9/s/txt?x="),
            ("victoriametrics_host", "http://vm.example"),
            ("model_type", "../../ESCAPED_model"),
        ):
            with self.subTest(key=key):
                response = await self.client.post("/set-config", json={key: value})
                self.assertEqual(response.status_code, 400)
                self.assertIn(key, await response.get_data(as_text=True))

    async def test_get_json_answers_as_json(self):
        body = "costfun: '<script>alert(1)</script>'\n"
        with patch.object(web_server, "emhass_conf", emhass_conf):
            response = await self.client.post(
                "/get-json", data=body, headers={"Content-Type": "text/plain"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(
            orjson.loads(await response.get_data())["costfun"], "<script>alert(1)</script>"
        )

    @patch("emhass.web_server.set_input_data_dict", new_callable=AsyncMock)
    @patch("emhass.web_server._load_params_and_runtime", new_callable=AsyncMock)
    async def test_api_error_replies_are_plain_text(self, mock_load, mock_set):
        mock_load.return_value = ({}, "profit", "{}")
        mock_set.side_effect = RuntimeParamError("custom_<img src=x>_id entity_id must be")
        response = await self.client.post("/action/publish-data", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.mimetype, "text/plain")
        # Pages are still pages
        page = await self.client.get("/configuration")
        self.assertNotEqual(page.mimetype, "text/plain")

    async def test_every_response_has_nosniff(self):
        for method, url in (("GET", "/healthz"), ("POST", "/configuration"), ("GET", "/nope")):
            with self.subTest(url=url):
                response = await self.client.open(url, method=method)
                self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")

    @patch("emhass.web_server._handle_action_dispatch", new_callable=AsyncMock)
    @patch("emhass.web_server.set_input_data_dict", new_callable=AsyncMock)
    @patch("emhass.web_server.grab_log", new_callable=AsyncMock)
    @patch("emhass.web_server._load_params_and_runtime", new_callable=AsyncMock)
    async def test_runtime_param_error_during_an_action_is_a_short_400(
        self, mock_load, mock_log, mock_set, mock_dispatch
    ):
        mock_load.return_value = ({}, "profit", "{}")
        mock_log.return_value = ["SHARED LOG TAIL"]
        mock_set.return_value = {"retrieve_hass_conf": {}}
        mock_dispatch.side_effect = RuntimeParamError("'../x.csv' is not a file name inside")
        response = await self.client.post("/action/perfect-optim", json={})
        mock_dispatch.assert_awaited_once()
        self.assertEqual(response.status_code, 400)
        text = await response.get_data(as_text=True)
        self.assertIn("not a file name inside", text)
        self.assertNotIn("SHARED LOG TAIL", text)

    async def test_configuration_is_get_only(self):
        response = await self.client.post("/configuration", data={"long_lived_token": "x"})
        self.assertEqual(response.status_code, 405)

    @patch("emhass.web_server.set_input_data_dict", new_callable=AsyncMock)
    @patch("emhass.web_server.grab_log", new_callable=AsyncMock)
    @patch("emhass.web_server._load_params_and_runtime", new_callable=AsyncMock)
    async def test_runtime_param_error_is_a_short_400(self, mock_load, mock_log, mock_set):
        mock_load.return_value = ({}, "profit", "{}")
        mock_log.return_value = ["SHARED LOG TAIL"]
        mock_set.side_effect = RuntimeParamError("model_type must be a plain file name")
        response = await self.client.post("/action/forecast-model-fit", json={})
        self.assertEqual(response.status_code, 400)
        text = await response.get_data(as_text=True)
        self.assertIn("model_type", text)
        self.assertNotIn("SHARED LOG TAIL", text)

    @patch("emhass.web_server.export_influxdb_to_csv", new_callable=AsyncMock)
    @patch("emhass.web_server.grab_log", new_callable=AsyncMock)
    @patch("emhass.web_server._load_params_and_runtime", new_callable=AsyncMock)
    async def test_runtime_param_error_on_direct_export_dispatch(
        self, mock_load, mock_log, mock_export
    ):
        mock_load.return_value = ({}, "profit", "{}")
        mock_log.return_value = ["SHARED LOG TAIL"]
        mock_export.side_effect = RuntimeParamError("csv_filename must end with .csv")
        response = await self.client.post("/action/export-influxdb-to-csv", json={})
        self.assertEqual(response.status_code, 400)
        text = await response.get_data(as_text=True)
        self.assertIn("csv_filename", text)
        self.assertNotIn("SHARED LOG TAIL", text)

    @patch("emhass.web_server.set_input_data_dict", new_callable=AsyncMock)
    @patch("emhass.web_server._load_params", new_callable=AsyncMock)
    async def test_runtime_param_error_on_dry_run_is_a_400(self, mock_load, mock_set):
        mock_load.return_value = ("{}", "profit")
        mock_set.side_effect = RuntimeParamError("model_type must be a plain file name")
        response = await self.client.post("/api/v1/dry-run/dayahead-optim", json={})
        # The error reached the new handler (without it the route answers 500). The
        # body's "errors" list is not checked: it comes from the app logger, whose
        # state other tests in the suite change.
        mock_set.assert_awaited_once()
        self.assertEqual(response.status_code, 400)
        body = await response.get_json()
        self.assertEqual(body["status"], "not-run")


class TestSupervisorTokenGuard(unittest.IsolatedAsyncioTestCase):
    SUPERVISOR = "http://supervisor/core/api"

    @staticmethod
    def _rh(url, token):
        return RetrieveHass(url, token, pd.Timedelta(minutes=30), pytz.UTC, {}, emhass_conf, logger)

    @patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token"})
    def test_supervisor_token_is_not_kept_for_another_host(self):
        with self.assertLogs(logger, level="ERROR"):
            rh = self._rh("https://example.org/", "sv-token")
        self.assertEqual(rh.long_lived_token, "")

    @patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token"})
    def test_supervisor_token_is_kept_for_the_supervisor(self):
        for url in (self.SUPERVISOR, self.SUPERVISOR + "/", "empty", ""):
            with self.subTest(url=url):
                self.assertEqual(self._rh(url, "sv-token").long_lived_token, "sv-token")

    @patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token"})
    def test_the_https_form_is_not_the_supervisor(self):
        # The Supervisor proxy is HTTP only (#1403)
        for url in ("https://supervisor/core/api", "https://supervisor/core/api/"):
            with self.subTest(url=url), self.assertLogs(logger, level="ERROR"):
                self.assertEqual(self._rh(url, "sv-token").long_lived_token, "")

    @patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token"})
    def test_own_long_lived_token_for_a_custom_url_is_kept(self):
        rh = self._rh("http://homeassistant.local:8123/", "my-own-llat")
        self.assertEqual(rh.long_lived_token, "my-own-llat")

    @patch.dict(os.environ, {}, clear=True)
    def test_without_supervisor_token_nothing_changes(self):
        rh = self._rh("https://example.org/", "whatever")
        self.assertEqual(rh.long_lived_token, "whatever")

    @patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token"})
    async def test_env_fallback_only_for_the_supervisor(self):
        # Custom URL, no token of its own: the env token is not used, so nothing is sent
        rh = self._rh("https://example.org/", "empty")
        days = pd.date_range(start="2024-01-01", periods=1, freq="D", tz="UTC")
        with aioresponses() as mocked:
            mocked.get(re.compile(r".*"), status=200, payload={}, repeat=True)
            self.assertTrue(await rh.get_ha_config())
            self.assertEqual(rh.ha_config, {})
            # REST history retrieval refuses to run without a token
            self.assertFalse(await rh._get_data_rest_api(days, ["sensor.x"]))
            self.assertEqual(sum(len(calls) for calls in mocked.requests.values()), 0)
        await rh.close()

    LONG_SV = "sv-token-0123456789abcdefghij"

    def test_a_supervisor_token_inside_a_longer_value_is_found(self):
        from emhass.connection_manager import (
            holds_supervisor_token,
            is_supervisor_token_for_another_host,
        )

        with patch.dict(os.environ, {"SUPERVISOR_TOKEN": self.LONG_SV}):
            for value in (
                self.LONG_SV,
                f" {self.LONG_SV}\t",
                f"Bearer {self.LONG_SV}",
                f"{self.LONG_SV}\x00",
                f"{self.LONG_SV}\u200b",
                f"x{self.LONG_SV}x",
            ):
                with self.subTest(value=value):
                    self.assertTrue(holds_supervisor_token(value))
                    self.assertTrue(
                        is_supervisor_token_for_another_host("http://other-ha.test:8123/", value)
                    )
                    self.assertFalse(is_supervisor_token_for_another_host(self.SUPERVISOR, value))
            for value in ("own-llat", self.LONG_SV[:-1], "", None):
                with self.subTest(value=value):
                    self.assertFalse(holds_supervisor_token(value))
        # A short Supervisor token is only compared whole, so it cannot match by accident
        with patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token"}):
            self.assertTrue(holds_supervisor_token(" sv-token "))
            self.assertFalse(holds_supervisor_token("Bearer sv-token"))
        # The line is at 16 characters
        for token, found in (("x" * 15, False), ("x" * 16, True)):
            with (
                self.subTest(length=len(token)),
                patch.dict(os.environ, {"SUPERVISOR_TOKEN": token}),
            ):
                self.assertTrue(holds_supervisor_token(token))
                self.assertEqual(holds_supervisor_token(f"Bearer {token}"), found)

    def test_a_decorated_supervisor_token_is_not_kept_for_another_host(self):
        with patch.dict(os.environ, {"SUPERVISOR_TOKEN": self.LONG_SV}):
            with self.assertLogs(logger, level="ERROR"):
                rh = self._rh("http://other-ha.test:8123/", f"Bearer {self.LONG_SV}")
            self.assertEqual(rh.long_lived_token, "")
            self.assertEqual(
                self._rh(self.SUPERVISOR, f"Bearer {self.LONG_SV}").long_lived_token,
                f"Bearer {self.LONG_SV}",
            )

    def test_request_urls_on_the_supervisor(self):
        from emhass.connection_manager import is_supervisor_request_url

        for url in (
            "http://supervisor/core/api",
            "http://supervisor/core/api/history/period/2024-01-01T00:00:00Z",
            " HTTP://Supervisor/core/api/states/sensor.x",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_supervisor_request_url(url))
        for url in (
            None,
            "",
            "empty",
            # The Supervisor proxy is HTTP only (#1403)
            "https://supervisor/core/api/states/sensor.x",
            "http://supervisor/core/apix",
            "http://supervisor/core/api@other-ha.test/x",
            "http://other-ha.test/supervisor/core/api/x",
            "http://supervisor.other-ha.test/core/api/x",
            "http://supervisor:8123/core/api/x",
        ):
            with self.subTest(url=url):
                self.assertFalse(is_supervisor_request_url(url))

    @patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token"})
    async def test_history_requests_are_judged_by_the_test_url(self):
        """A test_url replaces the URL the history is asked of: the Supervisor token only
        goes there when it is on the Supervisor proxy, its own or as the fallback."""
        days = pd.date_range(start="2024-01-01", periods=1, freq="D", tz="UTC")
        other = "http://other-ha.test:8123/api/history/period/2024-01-01T00:00:00Z"
        on_supervisor = "http://supervisor/core/api/history/period/2024-01-01T00:00:00Z"
        for token in ("sv-token", "empty"):
            with self.subTest(token=token):
                rh = self._rh(self.SUPERVISOR, token)
                with aioresponses() as mocked, self.assertLogs(logger, level="ERROR"):
                    mocked.get(re.compile(r".*"), status=500, repeat=True)
                    self.assertFalse(
                        await rh._get_data_rest_api(days, ["sensor.x"], test_url=other)
                    )
                    self.assertEqual(sum(len(calls) for calls in mocked.requests.values()), 0)
                await rh.close()
        for token in ("sv-token", "empty"):
            with self.subTest(token=token, url="on the Supervisor"):
                rh = self._rh(self.SUPERVISOR, token)
                with aioresponses() as mocked, self.assertLogs(logger, level="ERROR"):
                    mocked.get(re.compile(r".*"), status=500, repeat=True)
                    await rh._get_data_rest_api(days, ["sensor.x"], test_url=on_supervisor)
                    sent = [
                        c.kwargs["headers"]["Authorization"]
                        for calls in mocked.requests.values()
                        for c in calls
                    ]
                await rh.close()
                self.assertEqual(set(sent), {"Bearer sv-token"})

    async def test_history_with_a_decorated_supervisor_token_is_refused_for_a_test_url(self):
        days = pd.date_range(start="2024-01-01", periods=1, freq="D", tz="UTC")
        other = "http://other-ha.test:8123/api/history/period/2024-01-01T00:00:00Z"
        with patch.dict(os.environ, {"SUPERVISOR_TOKEN": self.LONG_SV}):
            rh = self._rh(self.SUPERVISOR, f"Bearer {self.LONG_SV}")
            with aioresponses() as mocked, self.assertLogs(logger, level="ERROR"):
                mocked.get(re.compile(r".*"), status=500, repeat=True)
                self.assertFalse(await rh._get_data_rest_api(days, ["sensor.x"], test_url=other))
                self.assertEqual(sum(len(calls) for calls in mocked.requests.values()), 0)
            await rh.close()

    @patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token"})
    async def test_history_on_the_supervisor_with_another_hass_url_sends_nothing(self):
        """The live-state check goes to hass_url, not test_url: with hass_url on another
        host the Supervisor token is not used, even for a test_url on the Supervisor."""
        days = pd.date_range(start="2024-01-01", periods=1, freq="D", tz="UTC")
        on_supervisor = "http://supervisor/core/api/history/period/2024-01-01T00:00:00Z"
        for token in ("empty", "sv-token"):
            with self.subTest(token=token):
                with self.assertLogs(logger, level="ERROR"):
                    rh = self._rh("http://other-ha.test:8123/", token)
                    with aioresponses() as mocked:
                        mocked.get(
                            re.compile(r".*"),
                            status=200,
                            payload=[[{"state": "1"}]],
                            repeat=True,
                        )
                        self.assertFalse(
                            await rh._get_data_rest_api(
                                days,
                                ["sensor.x"],
                                test_url=on_supervisor,
                                hold_unchanged_tail=True,
                            )
                        )
                        self.assertEqual(sum(len(calls) for calls in mocked.requests.values()), 0)
                await rh.close()

    async def test_post_data_refuses_a_non_entity_id(self):
        # Whatever the source (runtime, config, a saved file), it is not sent or saved
        rh = self._rh("http://ha.test:8123/", "my-own-llat")
        entities = pathlib.Path(emhass_conf["data_path"]) / "entities"
        before = set(entities.rglob("*")) if entities.exists() else set()
        data = pd.Series([100.0, 200.0], index=pd.date_range("2024-01-01", periods=2, freq="30min"))
        data.name = "x"

        async def post(entity_id, **kwargs):
            await rh.post_data(data, 0, entity_id, "power", "W", "X", "power", **kwargs)

        with aioresponses() as mocked:
            mocked.post(re.compile(r".*"), status=200, repeat=True)
            for entity_id in ("../events/sensor.emhass_poc", "sensor.x/../../events/y"):
                with self.subTest(entity_id=entity_id), self.assertLogs(logger, level="ERROR"):
                    await post(entity_id, save_entities=True)
            # A prefix cannot smuggle one in either
            with self.assertLogs(logger, level="ERROR"):
                await post("sensor.x", publish_prefix="/../../events/")
            self.assertEqual(sum(len(calls) for calls in mocked.requests.values()), 0)
            # Positive control: in the same setup a plain id is posted, once, where expected
            await post("sensor.x")
            sent = [str(url) for (_, url), calls in mocked.requests.items() for _ in calls]
            self.assertEqual(sent, ["http://ha.test:8123/api/states/sensor.x"])
        after = set(entities.rglob("*")) if entities.exists() else set()
        self.assertEqual(after, before)
        self.assertFalse((pathlib.Path(emhass_conf["data_path"]) / "events").exists())
        await rh.close()

    async def test_post_scalar_sensor_refuses_a_non_entity_id(self):
        rh = self._rh("http://ha.test:8123/", "my-own-llat")
        with aioresponses() as mocked:
            mocked.post(re.compile(r".*"), status=200, repeat=True)
            with self.assertLogs(logger, level="ERROR"):
                self.assertFalse(await rh.post_scalar_sensor("../events/sensor.x", 1, {}))
            self.assertEqual(sum(len(calls) for calls in mocked.requests.values()), 0)
            # Positive control
            self.assertTrue(await rh.post_scalar_sensor("sensor.x", 1, {}))
            self.assertEqual(sum(len(calls) for calls in mocked.requests.values()), 1)
        await rh.close()

    async def test_live_state_is_not_fetched_for_a_non_entity_id(self):
        rh = self._rh("http://ha.test:8123/", "my-own-llat")
        session = await rh._get_session()
        with aioresponses() as mocked:
            mocked.get(re.compile(r".*"), status=200, payload={"state": "1"}, repeat=True)
            with self.assertLogs(logger, level="ERROR"):
                self.assertIsNone(await rh._fetch_live_state(session, "../config", {}))
            self.assertEqual(sum(len(calls) for calls in mocked.requests.values()), 0)
            # Positive control: a plain id is fetched from the expected path
            self.assertEqual(await rh._fetch_live_state(session, "sensor.x", {}), {"state": "1"})
            sent = [str(url) for (_, url), calls in mocked.requests.items() for _ in calls]
            self.assertEqual(sent, ["http://ha.test:8123/api/states/sensor.x"])
        await rh.close()

    def test_history_url_encodes_the_sensor_name(self):
        rh = self._rh("http://ha.test:8123/", "my-own-llat")
        day = pd.Timestamp("2026-11-01", tz="UTC")
        url = rh._build_history_url(day, "sensor.p_pv", "empty", False, False)
        self.assertTrue(url.endswith("?filter_entity_id=sensor.p_pv"))
        url = rh._build_history_url(day, "sensor.x&end_time=2030-01-01", "empty", False, False)
        self.assertTrue(url.endswith("?filter_entity_id=sensor.x%26end_time%3D2030-01-01"))
        self.assertNotIn("&end_time", url)

    def test_database_host_with_a_path_disables_the_integration(self):
        for key_use, key_host in (
            ("use_influxdb", "influxdb_host"),
            ("use_victoriametrics", "victoriametrics_host"),
        ):
            params = {"retrieve_hass_conf": {key_use: True, key_host: "127.0.0.1:9/s/txt?x="}}
            with self.subTest(key=key_host), self.assertLogs(logger, level="ERROR"):
                rh = RetrieveHass(
                    "http://ha.test:8123/",
                    "llat",
                    pd.Timedelta(minutes=30),
                    pytz.UTC,
                    params,
                    emhass_conf,
                    logger,
                )
            self.assertFalse(getattr(rh, key_use))
            # Positive control: a plain host keeps it enabled
            params = {"retrieve_hass_conf": {key_use: True, key_host: "192.0.2.10"}}
            rh = RetrieveHass(
                "http://ha.test:8123/",
                "llat",
                pd.Timedelta(minutes=30),
                pytz.UTC,
                params,
                emhass_conf,
                logger,
            )
            self.assertTrue(getattr(rh, key_use))

    def test_only_the_supervisor_url_counts_as_the_supervisor(self):
        for url in (
            "http://supervisor.evil.test/core/api",
            "http://supervisor/core/api@evil.test",
            "http://evil.test/?http://supervisor/core/api",
            "http://supervisor:80/core/api",
            "http://supervisor/core/api/../../evil",
            "http://supervisor/core/apix",
            "https://supervisor/core/api",
            "https://example.org/",
        ):
            with self.subTest(url=url):
                self.assertFalse(_is_supervisor_url(url))
        for url in (self.SUPERVISOR, "HTTP://SUPERVISOR/CORE/API/", None, "", "empty"):
            with self.subTest(url=url):
                self.assertTrue(_is_supervisor_url(url))


class TestDatabaseQueriesAndErrors(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _rh(retrieve_hass_conf):
        return RetrieveHass(
            "http://ha.test:8123/",
            "llat",
            pd.Timedelta(minutes=30),
            pytz.UTC,
            {"retrieve_hass_conf": retrieve_hass_conf},
            emhass_conf,
            logger,
        )

    def test_influxql_values_are_escaped(self):
        rh = self._rh(
            {
                "use_influxdb": True,
                "influxdb_host": "db",
                "influxdb_database": 'home"; DROP',
                "influxdb_retention_policy": "auto\\gen",
            }
        )
        start = pd.Timestamp("2026-11-01", tz="UTC")
        query = rh._build_influx_query_for_measurement(
            "x'; SHOW DATABASES; --", 'W"x', start, start + pd.Timedelta(days=1)
        )
        self.assertIn("\"entity_id\"='x\\'; SHOW DATABASES; --'", query)
        self.assertIn('FROM "home\\"; DROP"."auto\\\\gen"."W\\"x"', query)
        # Positive control: a normal name is unchanged
        query = rh._build_influx_query_for_measurement(
            "p_pv", "W", start, start + pd.Timedelta(days=1)
        )
        self.assertIn("\"entity_id\"='p_pv'", query)

    def test_upstream_error_leaves_out_the_reply(self):
        class ClientError(Exception):
            code = 500

        text = retrieve_hass._upstream_error(ClientError("SECRET-REPLY-BODY"))
        self.assertEqual(text, "ClientError (HTTP 500)")
        self.assertNotIn("SECRET", retrieve_hass._upstream_error(ValueError("SECRET")))

    async def test_victoriametrics_errors_do_not_log_the_reply(self):
        rh = self._rh(
            {"use_victoriametrics": True, "victoriametrics_host": "vm", "victoriametrics_port": 1}
        )
        for status, body in (
            (500, b"SECRET-REPLY-BODY"),
            (302, b"SECRET-REPLY-BODY"),
            (200, b'{"status": "error", "error": "SECRET-REPLY-BODY"}'),
            (200, b'{"status": "SECRET-REPLY-STATUS"}'),
            (200, b'{"status": {"SECRET": "nested"}}'),
        ):
            response = MagicMock()
            response.status = status
            response.read = AsyncMock(return_value=body)
            context = MagicMock()
            context.__aenter__ = AsyncMock(return_value=response)
            context.__aexit__ = AsyncMock(return_value=False)
            session = MagicMock()
            session.get = MagicMock(return_value=context)
            with self.subTest(body=body):
                with self.assertLogs(logger, level="DEBUG") as cm:
                    self.assertIsNone(await rh._vm_query_range(session, "up", 0, 60, 60))
                self.assertFalse(any("SECRET" in line for line in cm.output))
                # The configured server cannot redirect EMHASS elsewhere
                self.assertIs(session.get.call_args.kwargs["allow_redirects"], False)

    def test_influxdb_client_does_not_follow_redirects(self):
        rh = self._rh({"use_influxdb": True, "influxdb_host": "db"})
        with patch("influxdb.InfluxDBClient") as client_class:
            self.assertIs(rh._init_influx_client(), client_class.return_value)
        session = client_class.call_args.kwargs["session"]
        self.assertEqual(session.max_redirects, 0)

    def test_influxdb_errors_do_not_log_the_reply(self):
        from influxdb.exceptions import InfluxDBClientError

        rh = self._rh({"use_influxdb": True, "influxdb_host": "db"})
        reply = '{"error":"SECRET-REPLY-BODY database not found: homeassistant"}'
        # The connection check
        with patch("influxdb.InfluxDBClient") as client_class:
            client_class.return_value.ping.side_effect = InfluxDBClientError(reply, 200)
            with self.assertLogs(logger, level="DEBUG") as cm:
                self.assertIsNone(rh._init_influx_client())
        self.assertFalse(any("SECRET" in line for line in cm.output))
        # Any reply but a 401 or 403 reads the same, whatever the server said
        self.assertTrue(any("unexpected reply" in line for line in cm.output))
        self.assertFalse(any("database not found" in line for line in cm.output))

        # Measurement discovery and the query itself, at debug level too
        def query(text):
            if text == "SHOW MEASUREMENTS":
                result = MagicMock()
                result.get_points.return_value = [{"name": "W"}]
                return result
            raise InfluxDBClientError("SECRET-REPLY-BODY", 400)

        client = MagicMock()
        client.query.side_effect = query
        rh._measurement_cache = {}
        start = pd.Timestamp("2026-11-01", tz="UTC")
        with self.assertLogs(logger, level="DEBUG") as cm:
            self.assertIsNone(
                rh._fetch_sensor_data(client, "sensor.x", start, start + pd.Timedelta(days=1))
            )
        self.assertTrue(any("unexpected reply" in line for line in cm.output))
        self.assertFalse(any("HTTP 400" in line or "SECRET" in line for line in cm.output))

    def test_influxdb_client_refuses_a_real_redirect(self):
        # Loopback: the real client and requests, not mocks
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        reached = []

        class Target(BaseHTTPRequestHandler):
            def do_GET(self):
                reached.append(self.path)
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args):
                pass

        target = HTTPServer(("127.0.0.1", 0), Target)

        class Redirector(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{target.server_port}/x?y=1")
                self.end_headers()

            def log_message(self, *args):
                pass

        redirector = HTTPServer(("127.0.0.1", 0), Redirector)
        for server in (target, redirector):
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            rh = self._rh(
                {
                    "use_influxdb": True,
                    "influxdb_host": "127.0.0.1",
                    "influxdb_port": redirector.server_port,
                }
            )
            with self.assertLogs(logger, level="ERROR") as cm:
                self.assertIsNone(rh._init_influx_client())
            self.assertEqual(reached, [])
            # Read like any other reply, so it does not confirm an HTTP service either
            self.assertTrue(any("unexpected reply" in line for line in cm.output))
        finally:
            for server in (target, redirector):
                server.shutdown()
                server.server_close()

    def test_server_supplied_names_stay_out_of_the_log(self):
        # InfluxDB: a measurement name the server lists
        rh = self._rh({"use_influxdb": True, "influxdb_host": "db"})

        def query(text):
            result = MagicMock()
            if text == "SHOW MEASUREMENTS":
                result.get_points.return_value = [{"name": "LEAKED_MEASUREMENT"}]
            elif text.startswith("SHOW TAG VALUES"):
                result.get_points.return_value = [{"value": "x"}]
            else:
                result.get_points.return_value = []
            return result

        client = MagicMock()
        client.query.side_effect = query
        rh._measurement_cache = {}
        start = pd.Timestamp("2026-11-01", tz="UTC")
        with self.assertLogs(logger, level="DEBUG") as cm:
            rh._fetch_sensor_data(client, "sensor.x", start, start + pd.Timedelta(days=1))
        self.assertTrue(any("found on the server" in line for line in cm.output))
        self.assertFalse(any("LEAKED" in line for line in cm.output))

    async def test_server_supplied_metric_names_stay_out_of_the_log(self):
        # VictoriaMetrics: several series with the server's names and labels
        rh = self._rh(
            {"use_victoriametrics": True, "victoriametrics_host": "vm", "victoriametrics_port": 1}
        )
        series = [
            {"metric": {"__name__": "LEAKED_METRIC_A"}, "values": [[1, "1"]]},
            {
                "metric": {"job": "LEAKED_JOB", "instance": "198.51.100.8:9100"},
                "values": [[1, "2"]],
            },
        ]
        start = pd.Timestamp("2026-11-01", tz="UTC")
        with patch.object(rh, "_vm_query_range", AsyncMock(return_value=series)):
            with self.assertLogs(logger, level="DEBUG") as cm:
                await rh._fetch_sensor_data_vm(
                    MagicMock(), "sensor.x", start, start + pd.Timedelta(hours=1)
                )
        self.assertTrue(any("several VictoriaMetrics metrics (2)" in line for line in cm.output))
        self.assertFalse(any("LEAKED" in line or "198.51.100.8" in line for line in cm.output))

    def test_upstream_error_prefers_status_and_names_known_reasons(self):
        class AiohttpLike(Exception):
            status = 503
            code = 999  # deprecated alias on aiohttp errors

        self.assertEqual(retrieve_hass._upstream_error(AiohttpLike("x")), "AiohttpLike (HTTP 503)")
        self.assertEqual(
            retrieve_hass._upstream_error(ValueError("Authorization failed: SECRET")),
            "ValueError: authorization failed",
        )


class TestSupervisorTokenNotSentElsewhere(unittest.IsolatedAsyncioTestCase):
    """The Supervisor token goes to the Supervisor proxy only: no WebSocket, no config call."""

    OTHER = "https://other-ha.test:8123/api"

    def setUp(self):
        import asyncio

        import emhass.connection_manager as cm

        cm._global_client = None
        cm._lock = asyncio.Lock()
        # A developer's own EMHASS_KEY (docs/develop.md) would skip the Supervisor path.
        env = self.enterContext(patch.dict(os.environ))
        env.pop("EMHASS_KEY", None)

    async def asyncTearDown(self):
        from emhass.connection_manager import close_global_connection

        await close_global_connection()

    @patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token"})
    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_websocket_refuses_the_supervisor_token_for_another_host(self, client_class):
        from emhass.connection_manager import get_websocket_client
        from emhass.websocket_client import ConnectionError as WsConnectionError

        for token in ("sv-token", " sv-token\t"):
            with self.subTest(token=token), self.assertLogs(logger, level="ERROR"):
                with self.assertRaises(WsConnectionError):
                    await get_websocket_client(self.OTHER, token, logger)
        client_class.assert_not_called()

    @patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token"})
    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_websocket_still_connects_where_it_should(self, client_class):
        import emhass.connection_manager as cm
        from emhass.connection_manager import get_websocket_client

        client_class.return_value = AsyncMock(connected=True)
        for url, token in (
            ("http://supervisor/core/api", "sv-token"),
            ("http://supervisor/core/api/", "sv-token"),
            (self.OTHER, "own-llat"),
        ):
            with self.subTest(url=url, token=token):
                cm._global_client = None
                await get_websocket_client(url, token, logger)
                client_class.assert_called_with(hass_url=url, long_lived_token=token, logger=logger)

    def _options(self, tmp, options):
        path = pathlib.Path(tmp) / "options.json"
        path.write_bytes(orjson.dumps(options))
        conf = dict(emhass_conf)
        conf["data_path"] = pathlib.Path(tmp)
        return conf, str(path)

    @patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token"})
    async def test_build_secrets_does_not_send_the_supervisor_token_to_the_options_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            conf, options_path = self._options(
                tmp, {"hass_url": self.OTHER, "long_lived_token": "own-llat"}
            )
            with aioresponses() as mocked:
                # Another Home Assistant answers the Supervisor token with 401
                mocked.get(re.compile(r".*"), status=401, repeat=True)
                _, secrets = await utils.build_secrets(
                    conf, logger, options_path=options_path, secrets_path=""
                )
                sent = [
                    c.kwargs["headers"]["Authorization"]
                    for cs in mocked.requests.values()
                    for c in cs
                ]
                self.assertEqual(sent, [])
        self.assertEqual(secrets["hass_url"], self.OTHER)
        self.assertEqual(secrets["long_lived_token"], "own-llat")

    @patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token"})
    async def test_build_secrets_still_asks_the_supervisor(self):
        payload = {
            "time_zone": "Europe/Paris",
            "latitude": 1.5,
            "longitude": 2.5,
            "elevation": 3,
        }
        for options_url in ("", "http://supervisor/core/api"):
            with self.subTest(options_url=options_url), tempfile.TemporaryDirectory() as tmp:
                conf, options_path = self._options(
                    tmp, {"hass_url": options_url, "long_lived_token": "own-llat"}
                )
                with aioresponses() as mocked:
                    mocked.get(
                        "http://supervisor/core/api/config",
                        status=200,
                        payload=payload,
                        repeat=True,
                    )
                    _, secrets = await utils.build_secrets(
                        conf, logger, options_path=options_path, secrets_path=""
                    )
                    calls = [c for (_, url), cs in mocked.requests.items() for c in cs]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].kwargs["headers"]["Authorization"], "Bearer sv-token")
                self.assertEqual(secrets["hass_url"], "http://supervisor/core/api")
                self.assertEqual(secrets["time_zone"], "Europe/Paris")

    def _sent_tokens(self, mocked):
        return [
            c.kwargs.get("headers", {}).get("Authorization")
            for cs in mocked.requests.values()
            for c in cs
        ]

    async def test_build_secrets_never_returns_the_supervisor_token_with_another_url(self):
        """Every way of ending up with another URL and no token of its own drops the token."""
        with tempfile.TemporaryDirectory() as tmp:
            secrets_file = pathlib.Path(tmp) / "secrets.yaml"
            secrets_file.write_text(f"hass_url: {self.OTHER}\n", encoding="utf-8")
            other_only = {"hass_url": self.OTHER}
            cases = [
                ("options token empty", {**other_only, "long_lived_token": ""}, {}, {}),
                ("options token 'empty'", {**other_only, "long_lived_token": "empty"}, {}, {}),
                ("options token absent", other_only, {}, {}),
                ("EMHASS_URL, no options", None, {"EMHASS_URL": self.OTHER}, {}),
                (
                    "EMHASS_KEY is the Supervisor token",
                    None,
                    {"EMHASS_URL": self.OTHER, "EMHASS_KEY": "sv-token"},
                    {},
                ),
                # No URL leaves the example URL, so the example-URL check drops the token
                # first, with its own message: tests/test_placeholder_hass_url.py.
                ("secrets file URL only", None, {}, {"secrets_path": str(secrets_file)}),
                ("argument URL only", None, {}, {"argument": {"url": self.OTHER}}),
                # no_response skips the options, so the URL comes from EMHASS_URL here.
                ("no_response", other_only, {"EMHASS_URL": self.OTHER}, {"no_response": True}),
            ]
            for name, options, env, kwargs in cases:
                with (
                    self.subTest(name),
                    patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token", **env}),
                ):
                    if options is None:
                        conf, options_path = dict(emhass_conf), ""
                    else:
                        conf, options_path = self._options(tmp, options)
                    kwargs = {"secrets_path": "", **kwargs}
                    with aioresponses() as mocked, self.assertLogs(logger, level="ERROR") as logs:
                        mocked.get(re.compile(r".*"), status=401, repeat=True)
                        _, secrets = await utils.build_secrets(
                            conf, logger, options_path=options_path, **kwargs
                        )
                        self.assertEqual(self._sent_tokens(mocked), [])
                    self.assertNotEqual(secrets["hass_url"], "http://supervisor/core/api")
                    self.assertEqual(secrets["long_lived_token"], "")
                    self.assertTrue(any("not be sent" in line for line in logs.output))

    @patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token"})
    async def test_build_secrets_keeps_the_supervisor_token_when_the_supervisor_fails(self):
        """A Supervisor error falls back to the options and keeps the Supervisor URL and token."""
        for options_url in ("", "empty", " HTTP://Supervisor/CORE/API/ "):
            with self.subTest(options_url=options_url), tempfile.TemporaryDirectory() as tmp:
                conf, options_path = self._options(
                    tmp, {"hass_url": options_url, "long_lived_token": "", "time_zone": "UTC"}
                )
                with aioresponses() as mocked:
                    mocked.get("http://supervisor/core/api/config", status=500, repeat=True)
                    _, secrets = await utils.build_secrets(
                        conf, logger, options_path=options_path, secrets_path=""
                    )
                    self.assertEqual(self._sent_tokens(mocked), ["Bearer sv-token"])
                self.assertEqual(secrets["hass_url"], "http://supervisor/core/api")
                self.assertEqual(secrets["long_lived_token"], "sv-token")
                self.assertEqual(secrets["time_zone"], "UTC")

    def test_supervisor_url_forms(self):
        from emhass.connection_manager import canonical_supervisor_url, is_supervisor_url

        for url in (
            "http://supervisor/core/api",
            "http://supervisor/core/api/",
            " HTTP://Supervisor/CORE/API\t",
        ):
            with self.subTest(url=url):
                self.assertTrue(is_supervisor_url(url))
                self.assertEqual(canonical_supervisor_url(url), "http://supervisor/core/api")
        for url in (None, "", "empty"):
            with self.subTest(url=url):
                self.assertTrue(is_supervisor_url(url))
                self.assertEqual(canonical_supervisor_url(url), url)
        for url in (
            self.OTHER,
            "http://supervisor.other-ha.test/core/api",
            "http://other-ha.test/supervisor/core/api",
            "http://user@supervisor/core/api",
            "http://supervisor@other-ha.test/core/api",
            "http://supervisor:8123/core/api",
            "http://supervisor/core/api?x=1",
            "http://supervisor/core/api#x",
            "http://supervisor/core/apix",
            "http://supervisor/core/api/states",
            "ws://supervisor/core/api",
            # The Supervisor proxy is HTTP only (#1403)
            "https://supervisor/core/api",
            "https://supervisor/core/api/",
        ):
            with self.subTest(url=url):
                self.assertFalse(is_supervisor_url(url))
                self.assertEqual(canonical_supervisor_url(url), url)

    @patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token"})
    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_websocket_refusal_caches_nothing_and_the_cache_is_not_redirected(
        self, client_class
    ):
        import emhass.connection_manager as cm
        from emhass.connection_manager import get_websocket_client
        from emhass.websocket_client import ConnectionError as WsConnectionError

        with self.assertLogs(logger, level="ERROR"), self.assertRaises(WsConnectionError):
            await get_websocket_client(self.OTHER, "sv-token", logger)
        self.assertIsNone(cm._global_client)

        client_class.return_value = AsyncMock(connected=True)
        first = await get_websocket_client("http://supervisor/core/api", "sv-token", logger)
        # A later call with another URL reuses the checked client and dials nothing new.
        again = await get_websocket_client(self.OTHER, "sv-token", logger)
        self.assertIs(again, first)
        client_class.assert_called_once_with(
            hass_url="http://supervisor/core/api", long_lived_token="sv-token", logger=logger
        )

        # A reconnect re-dials the same checked client, never a new one for another URL.
        first.connected = False
        with patch("emhass.connection_manager._reconnect_client", AsyncMock()) as reconnect:
            await get_websocket_client(self.OTHER, "sv-token", logger)
        reconnect.assert_awaited_once()
        self.assertIs(reconnect.await_args.args[0], first)
        self.assertEqual(client_class.call_count, 1)

    @patch.dict(os.environ, {"SUPERVISOR_TOKEN": "sv-token"})
    @patch("emhass.connection_manager.AsyncWebSocketClient")
    async def test_websocket_is_not_opened_without_a_token(self, client_class):
        import emhass.connection_manager as cm
        from emhass.connection_manager import get_websocket_client
        from emhass.websocket_client import ConnectionError as WsConnectionError

        for url in (self.OTHER, "http://supervisor/core/api"):
            for token in ("", " ", "empty", None):
                with (
                    self.subTest(url=url, token=token),
                    self.assertLogs(logger, level="ERROR"),
                    self.assertRaises(WsConnectionError),
                ):
                    await get_websocket_client(url, token, logger)
        client_class.assert_not_called()
        self.assertIsNone(cm._global_client)

    async def test_websocket_connect_ignores_environment_proxies(self):
        import inspect

        import websockets

        from emhass.websocket_client import AsyncWebSocketClient

        # Decided by the library, not by the module under test: emptying the module's
        # own proxy setting must fail here, not skip.
        if "proxy" not in inspect.signature(websockets.connect).parameters:
            self.skipTest("this websockets version never uses an environment proxy")
        client = AsyncWebSocketClient("http://supervisor/core/api", "sv-token", logger)
        connect = AsyncMock(side_effect=OSError("stop here"))
        with (
            patch.dict(os.environ, {"HTTP_PROXY": "http://proxy.other-ha.test:3128"}),
            patch("emhass.websocket_client._ws_connect", connect),
            self.assertRaises(OSError),
        ):
            await client._connect()
        self.assertIsNone(connect.call_args.kwargs["proxy"])

    async def _redirecting_host(self, target_port, status):
        """A host that answers the WebSocket upgrade with a redirect to the target."""
        import asyncio

        asked = []

        async def answer(reader, writer):
            try:
                asked.append(await reader.readuntil(b"\r\n\r\n"))
                writer.write(
                    f"HTTP/1.1 {status} Redirect\r\n"
                    f"Location: ws://127.0.0.1:{target_port}/api/websocket\r\n"
                    "Content-Length: 0\r\nConnection: close\r\n\r\n".encode()
                )
                await writer.drain()
            except (asyncio.IncompleteReadError, OSError):
                pass
            finally:
                writer.close()

        server = await asyncio.start_server(answer, "127.0.0.1", 0)
        return server, asked

    async def _home_assistant_websocket(self):
        """A stand-in for Home Assistant's WebSocket API: records every token it is sent."""
        from websockets.asyncio.server import serve

        tokens = []

        async def handler(ws):
            await ws.send(orjson.dumps({"type": "auth_required"}).decode())
            async for message in ws:
                data = orjson.loads(message)
                if data.get("type") == "auth":
                    tokens.append(data.get("access_token"))
                    await ws.send(orjson.dumps({"type": "auth_ok"}).decode())

        server = await serve(handler, "127.0.0.1", 0)
        return server, tokens

    async def test_websocket_never_follows_a_redirect(self):
        """A redirect answer to the upgrade is not followed, on the first connect (300, 301,
        302, 303, 307, 308) and on a reconnect (302): the host the redirect names is sent no
        token. A client with the same token reaching that host directly does send it, so the
        stand-in would have recorded one."""
        from websockets.exceptions import InvalidStatus

        from emhass.websocket_client import REDIRECT_NOT_FOLLOWED, AsyncWebSocketClient

        target, tokens = await self._home_assistant_websocket()
        target_port = target.sockets[0].getsockname()[1]
        try:
            for status in (300, 301, 302, 303, 307, 308):
                redirect, asked = await self._redirecting_host(target_port, status)
                port = redirect.sockets[0].getsockname()[1]
                client = AsyncWebSocketClient(f"http://127.0.0.1:{port}", "dummy-ha-token", logger)
                try:
                    with self.subTest(status=status, path="startup"):
                        with self.assertLogs(logger, level="ERROR") as logs:
                            with self.assertRaises(InvalidStatus):
                                await client.startup()
                        self.assertIn(REDIRECT_NOT_FOLLOWED, logs.output[0])
                    if status == 302:
                        with self.subTest(status=status, path="reconnect"):
                            with self.assertLogs(logger, level="ERROR"):
                                with self.assertRaises(InvalidStatus):
                                    await client.reconnect()
                        with self.subTest(status=status, path="send"):
                            with self.assertLogs(logger, level="ERROR"):
                                with self.assertRaises(InvalidStatus):
                                    await client.send("get_config")
                finally:
                    await client.shutdown()
                    redirect.close()
                    await redirect.wait_closed()
                self.assertEqual(len(asked), 3 if status == 302 else 1)
                self.assertEqual(tokens, [])
            client = AsyncWebSocketClient(
                f"http://127.0.0.1:{target_port}", "dummy-ha-token", logger
            )
            await client.startup()
            await client.shutdown()
            self.assertEqual(tokens, ["dummy-ha-token"])
        finally:
            target.close()
            await target.wait_closed()

    def test_a_websockets_client_that_cannot_refuse_a_redirect_is_not_used(self):
        import warnings

        import websockets

        from emhass import websocket_client

        class FollowsRedirects:  # as websockets' legacy client: it never asks
            pass

        self.assertIsNone(websocket_client._without_redirects(FollowsRedirects))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                from websockets.legacy.client import connect as legacy_connect
            except ImportError:
                legacy_connect = None
        if legacy_connect is not None:
            self.assertIsNone(websocket_client._without_redirects(legacy_connect))
        refusing = websocket_client._without_redirects(websockets.connect)
        self.assertTrue(issubclass(refusing, websockets.connect))
        self.assertIs(websocket_client._ws_connect.__bases__[0], websockets.connect)

    async def test_websocket_is_not_opened_when_redirects_cannot_be_refused(self):
        from emhass.websocket_client import REDIRECTS_NOT_REFUSABLE, AsyncWebSocketClient
        from emhass.websocket_client import ConnectionError as WsConnectionError

        client = AsyncWebSocketClient("http://supervisor/core/api", "sv-token", logger)
        with (
            patch("emhass.websocket_client._ws_connect", None),
            self.assertLogs(logger, level="ERROR") as logs,
            self.assertRaises(WsConnectionError),
        ):
            await client._connect()
        self.assertIn(REDIRECTS_NOT_REFUSABLE, logs.output[0])


class TestDatabaseServerChange(AuthSupportBase):
    """/set-config cannot move a database whose login is stored (finding 5)."""

    SAVED = {
        "use_influxdb": False,
        "influxdb_host": "127.0.0.1",
        "influxdb_port": 8086,
        "influxdb_use_ssl": True,
        "influxdb_verify_ssl": True,
        "use_victoriametrics": False,
        "victoriametrics_host": "127.0.0.1",
        "victoriametrics_port": 8428,
        "victoriametrics_use_ssl": False,
        "victoriametrics_verify_ssl": False,
    }
    LOGIN = {"influxdb_username": "admin", "influxdb_password": "real-password"}

    async def asyncSetUp(self):
        await super().asyncSetUp()
        os.environ.pop("INFLUXDB_USERNAME", None)
        os.environ.pop("INFLUXDB_PASSWORD", None)
        (self.dir / "defaults.json").write_text("{}")
        web_server.emhass_conf.update(
            defaults_path=self.dir / "defaults.json", config_path=self.dir / "config.json"
        )
        (self.data / web_server.params_file).write_bytes(
            pickle.dumps((self.dir / "config.json", {"retrieve_hass_conf": dict(self.SAVED)}))
        )
        # build_params and param_to_config reduced to a pass-through, so what is saved
        # is the defaults (none) plus the body
        self.enterContext(
            patch(
                "emhass.web_server.build_params",
                new=AsyncMock(
                    side_effect=lambda _c, _s, cfg, _l: {"retrieve_hass_conf": dict(cfg)}
                ),
            )
        )
        self.enterContext(
            patch(
                "emhass.web_server.param_to_config",
                side_effect=lambda params, _l: dict(params["retrieve_hass_conf"]),
            )
        )
        self.replace = self.enterContext(
            patch("emhass.web_server.replace_file", new_callable=AsyncMock)
        )
        self.secrets = self.enterContext(patch.object(web_server, "params_secrets", {}))
        os.environ[api_auth.INGRESS_PORT_ENV] = "8099"
        await self.enable()

    async def save(self, changes, *, ingress=None):
        body = {**self.SAVED, **changes}
        if ingress is None:
            return await self.send("POST", "/set-config", headers=BEARER, json=body)
        return await self.send(
            "POST", "/set-config", port=8099, peer=SUPERVISOR, headers=ingress, json=body
        )

    async def assert_refused(self, response):
        self.assertEqual(response.status_code, 400)
        self.assertEqual(await response.get_json(), [web_server.DB_SERVER_CHANGE_REFUSED])
        self.replace.assert_not_awaited()

    async def test_a_server_change_is_refused_while_a_login_is_stored(self):
        self.secrets.update(self.LOGIN)
        for changes in (
            {"influxdb_host": "127.0.0.9"},
            {"influxdb_port": 8087},
            {"influxdb_use_ssl": False},
            {"influxdb_verify_ssl": False},
            # the integration off makes no difference: it can be switched on at runtime
            {"influxdb_host": "127.0.0.9", "use_influxdb": False},
        ):
            with self.subTest(changes=changes):
                await self.assert_refused(await self.save(changes))

    async def test_victoriametrics_and_an_environment_login_count_too(self):
        self.secrets.update({"victoriametrics_username": "vm", "victoriametrics_password": ""})
        await self.assert_refused(await self.save({"victoriametrics_host": "127.0.0.9"}))
        self.secrets.clear()
        os.environ["INFLUXDB_PASSWORD"] = "from-env"
        await self.assert_refused(await self.save({"influxdb_port": "8087"}))

    async def test_a_placeholder_or_no_login_does_not_block_a_change(self):
        for secrets in (
            {},
            {"influxdb_username": "", "influxdb_password": ""},
            {"influxdb_username": "empty", "influxdb_password": "empty"},
            {
                "influxdb_username": "yourinfluxdbusername",
                "influxdb_password": "yourinfluxdbpassword",
            },
        ):
            with self.subTest(secrets=secrets):
                self.secrets.clear()
                self.secrets.update(secrets)
                response = await self.save(
                    {"influxdb_host": "127.0.0.9", "victoriametrics_port": 1}
                )
                self.assertEqual(response.status_code, 200)

    async def test_resending_the_saved_server_is_allowed(self):
        # The UI posts every field on each save
        self.secrets.update(self.LOGIN)
        self.secrets.update({"victoriametrics_username": "vm", "victoriametrics_password": "pw"})
        for changes in (
            {},
            {"influxdb_port": "8086"},
            {"use_influxdb": True, "influxdb_database": "other"},
            {"victoriametrics_verify_ssl": True},  # turning checks on is fine
        ):
            with self.subTest(changes=changes):
                response = await self.save(changes)
                self.assertEqual(response.status_code, 200)
        # A host name is compared without regard to case
        self.assertEqual(
            utils.db_endpoint_changes(
                {"influxdb_host": "LocalHost"},
                {"influxdb_host": "localhost"},
                utils.stored_db_logins(self.LOGIN),
            ),
            [],
        )

    async def test_a_save_is_refused_when_the_saved_params_cannot_be_read(self):
        # Nothing to compare with: compared with the defaults instead, a save naming the
        # default server would pass whatever server config.json then names
        self.secrets.update(self.LOGIN)
        defaults = {
            "influxdb_host": "localhost",
            "influxdb_port": 8086,
            "influxdb_use_ssl": False,
            "influxdb_verify_ssl": False,
        }
        path = self.data / web_server.params_file
        config_path = self.dir / "config.json"
        for state, content in (
            ("missing", None),
            ("not a pickle", b"not a pickle"),
            # unpickling these raises ModuleNotFoundError and AttributeError
            ("no such module", b"cno_such_module_p3r3\nThing\n."),
            ("no such name", b"cos\nno_such_name_p3r3\n."),
            ("not a pair", pickle.dumps({"retrieve_hass_conf": {}})),
            ("params not a dict", pickle.dumps((config_path, ["x"]))),
            ("conf a list", pickle.dumps((config_path, {"retrieve_hass_conf": ["x"]}))),
            ("conf a text", pickle.dumps((config_path, {"retrieve_hass_conf": "x"}))),
        ):
            with self.subTest(state=state):
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(content)
                for changes in (defaults, {}):
                    response = await self.save(changes)
                    # Its own text, not the one about a database server: no save at all
                    # can go ahead until a restart rebuilds the file
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(
                        await response.get_json(), [web_server.SAVED_PARAMS_UNREADABLE]
                    )
                    self.replace.assert_not_awaited()
        # With no login stored there is nothing to keep, and the save goes through
        self.secrets.clear()
        self.assertEqual((await self.save(defaults)).status_code, 200)

    async def test_a_home_assistant_administrator_on_ingress_may_change_it(self):
        self.secrets.update(self.LOGIN)
        os.environ["SUPERVISOR_TOKEN"] = "supervisor-token"
        response = await self.save({"influxdb_host": "127.0.0.9"}, ingress=ADMIN)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.replace.await_count, 2)

    async def test_the_administrator_exception_needs_a_real_add_on(self):
        # Without SUPERVISOR_TOKEN this is not an add-on, and with EMHASS_INGRESS_PEERS
        # set the peer is not known to be a Supervisor that strips a client's user headers
        self.secrets.update(self.LOGIN)
        for env in ({}, {"SUPERVISOR_TOKEN": "t", api_auth.INGRESS_PEERS_ENV: SUPERVISOR}):
            with self.subTest(env=env), patch.dict(os.environ, env):
                await self.assert_refused(
                    await self.save({"influxdb_host": "127.0.0.9"}, ingress=ADMIN)
                )

    async def test_any_other_ingress_user_may_not(self):
        self.secrets.update(self.LOGIN)
        os.environ["SUPERVISOR_TOKEN"] = "supervisor-token"
        for headers in (
            {"X-Remote-User-Id": "ha-user-2", "X-Remote-User-Groups": "system-users"},
            {},
        ):
            with self.subTest(headers=headers):
                await self.assert_refused(
                    await self.save({"influxdb_host": "127.0.0.9"}, ingress=headers)
                )

    def test_the_refusal_names_the_web_ui_from_the_sidebar(self):
        text = web_server.DB_SERVER_CHANGE_REFUSED
        self.assertIn("EMHASS web UI opened from the Home Assistant sidebar", text)
        self.assertNotIn("configuration page", text)
        # What counts as a change of server, and that turning a check on is one way
        self.assertIn(
            "its host or port, or turning SSL or certificate checks off, or a different CA file",
            text,
        )
        self.assertIn("turning them off again needs one of those two", text)
        self.assertNotIn("SSL setting", text)


class TestDatabaseErrorTexts(unittest.IsolatedAsyncioTestCase):
    """A caller reads the action log, so a database failure there is in fixed words."""

    def test_every_failure_to_reach_the_server_reads_the_same(self):
        import aiohttp
        import requests

        fixed = "could not reach the configured InfluxDB server"
        for exc in (
            requests.exceptions.ConnectionError("host='127.0.0.9' port=1 SECRET"),
            requests.exceptions.ConnectTimeout("SECRET"),
            requests.exceptions.ReadTimeout("SECRET"),
            requests.exceptions.SSLError("SECRET"),
            # a reply cut off part way
            requests.exceptions.ChunkedEncodingError("SECRET"),
            requests.exceptions.ContentDecodingError("SECRET"),
            aiohttp.ClientConnectionError("SECRET"),
            aiohttp.ServerDisconnectedError("SECRET"),
            TimeoutError("SECRET"),
            ConnectionRefusedError("SECRET"),
        ):
            with self.subTest(exc=type(exc).__name__):
                self.assertEqual(retrieve_hass._db_error(exc, "InfluxDB"), fixed)
        # A 401 or 403 is still named by its status, so a wrong login can be told apart
        from influxdb.exceptions import InfluxDBClientError

        for code in (401, 403):
            self.assertEqual(
                retrieve_hass._db_error(InfluxDBClientError("SECRET", code), "InfluxDB"),
                f"InfluxDBClientError (HTTP {code})",
            )

    def test_every_other_reply_reads_the_same(self):
        import aiohttp
        import requests
        from influxdb.exceptions import InfluxDBClientError

        redirect = aiohttp.ClientResponseError(
            MagicMock(), (), status=307, message="redirect not followed"
        )
        for exc in (
            InfluxDBClientError("SECRET", 404),
            InfluxDBClientError("SECRET", 500),
            InfluxDBClientError('{"error":"database not found: SECRET"}', 200),
            InfluxDBClientError("error parsing query: SECRET"),
            requests.exceptions.TooManyRedirects("SECRET"),
            redirect,
            RuntimeError("InfluxDB query returned a statement error"),
            ValueError("SECRET"),
        ):
            with self.subTest(exc=repr(exc)[:40]):
                self.assertEqual(
                    retrieve_hass._db_error(exc, "InfluxDB"),
                    "unexpected reply from the configured InfluxDB server",
                )

    def test_the_detail_line_cannot_forge_a_log_line(self):
        # The server's reply is in the error's text: control characters are escaped and
        # the text is cut short, so it stays one bounded line on the console
        forged = "x\r\n\x1b[2K[2026-11-24 00:00:00] INFO in web_server: forged\u2028" + "y" * 5000
        with self.assertLogs("emhass.database", level="WARNING") as cm:
            retrieve_hass._db_error(ValueError(forged), "InfluxDB")
        message = cm.records[-1].getMessage()
        for raw in ("\r", "\n", "\x1b", "\u2028"):
            self.assertNotIn(raw, message)
        self.assertIn("\\x1b", message)
        self.assertLess(len(message), 400)

    def test_the_influxdb_connection_check_does_not_name_the_host(self):
        import requests

        rh = TestDatabaseQueriesAndErrors._rh({"use_influxdb": True, "influxdb_host": "127.0.0.9"})
        with patch("influxdb.InfluxDBClient") as client_class:
            client_class.return_value.ping.side_effect = requests.exceptions.ConnectionError(
                "HTTPConnectionPool(host='127.0.0.9', port=8086): Max retries exceeded"
            )
            with self.assertLogs(logger, level="ERROR") as cm:
                self.assertIsNone(rh._init_influx_client())
        self.assertEqual(
            cm.output[-1].split(":", 2)[-1],
            "Failed to connect to InfluxDB: could not reach the configured InfluxDB server",
        )

    async def test_victoriametrics_errors_do_not_name_the_url_or_the_size(self):
        import aiohttp

        rh = TestDatabaseQueriesAndErrors._rh(
            {"use_victoriametrics": True, "victoriametrics_host": "127.0.0.9"}
        )
        for kwargs, expected in (
            ({"exc": aiohttp.ClientConnectionError("SECRET")}, "could not reach the configured"),
            ({"status": 500, "body": b"x" * 1234}, "VictoriaMetrics query failed: unexpected"),
            ({"status": 403, "body": b""}, "403 Forbidden"),
        ):
            session = MagicMock()
            if "exc" in kwargs:
                session.get = MagicMock(side_effect=kwargs["exc"])
            else:
                response = MagicMock()
                response.status = kwargs["status"]
                response.read = AsyncMock(return_value=kwargs["body"])
                context = MagicMock()
                context.__aenter__ = AsyncMock(return_value=response)
                context.__aexit__ = AsyncMock(return_value=False)
                session.get = MagicMock(return_value=context)
            with self.subTest(expected=expected):
                with self.assertLogs(logger, level="ERROR") as cm:
                    self.assertIsNone(await rh._vm_query_range(session, "up", 0, 60, 60))
                line = cm.output[-1]
                self.assertIn(expected, line)
                for leaked in ("SECRET", "127.0.0.9", "1234", "bytes", "ClientConnectionError"):
                    self.assertNotIn(leaked, line)

    async def test_the_thermal_reader_warning_is_fixed_text(self):
        import aiohttp
        import requests

        idx = pd.date_range("2026-11-19", periods=4, freq="30min", tz="UTC")
        cfg = {
            "entity_t_in": "sensor.t_in",
            "entity_t_out": "sensor.t_out",
            "entity_power": "",
            "entity_t_pred": "",
            "entity_switch": "",
            "start_utc": idx[0],
            "end_utc": idx[-1],
            "dt_minutes": 30,
            "dt_hours": 0.5,
            "nominal_power": 1000.0,
            "lookback_hours": 2,
            "sense": "heat",
        }
        for exc, expected in (
            (
                requests.exceptions.ConnectionError("http://127.0.0.9:1/query SECRET"),
                "could not reach the configured InfluxDB server",
            ),
            (RuntimeError("http://127.0.0.9:1/query SECRET"), "unexpected reply"),
            (
                aiohttp.ClientResponseError(
                    MagicMock(), (), status=302, message="redirect not followed"
                ),
                "unexpected reply",
            ),
        ):
            with self.subTest(exc=type(exc).__name__):
                with (
                    patch("emhass.utils._fetch_influx_thermal_window", side_effect=exc),
                    self.assertLogs(logger, level="WARNING") as cm,
                ):
                    _, source = await utils._load_thermal_calibration_frame(
                        emhass_conf, {}, {"use_influxdb": True}, {}, cfg, False, logger
                    )
                self.assertEqual(source, "offline_demo")
                line = next(x for x in cm.output if "thermal history" in x)
                self.assertIn(expected, line)
                self.assertNotIn("SECRET", line)
                self.assertNotIn("127.0.0.9", line)


class TestDatabaseFlags(AuthSupportBase):
    """/set-config through the real build_params and param_to_config: a database on/off
    setting must be true or false. "true", 1 or "yes" would be read as off by the InfluxDB
    client, but looked unchanged to the check that keeps a stored login with its server,
    so a saved use_ssl true could be turned into plain HTTP."""

    LOGIN = {"influxdb_username": "admin", "influxdb_password": "real-password"}
    FLAGS = (
        "use_influxdb",
        "influxdb_use_ssl",
        "influxdb_verify_ssl",
        "use_victoriametrics",
        "victoriametrics_use_ssl",
        "victoriametrics_verify_ssl",
    )
    SERVER = {
        "use_influxdb": True,
        "influxdb_host": "127.0.0.1",
        "influxdb_port": 8086,
        "influxdb_use_ssl": True,
        "influxdb_verify_ssl": True,
    }

    async def asyncSetUp(self):
        await super().asyncSetUp()
        os.environ.pop("INFLUXDB_USERNAME", None)
        os.environ.pop("INFLUXDB_PASSWORD", None)
        self.src = pathlib.Path(web_server.__file__).parent
        web_server.emhass_conf.update(
            defaults_path=self.src / "data/config_defaults.json",
            associations_path=self.src / "data/associations.csv",
            config_path=self.dir / "config.json",
            legacy_config_path=self.dir / "legacy_missing.yaml",
            root_path=self.src,
        )
        self.enterContext(patch.object(web_server, "params_secrets", dict(self.LOGIN)))
        await self.enable()

    async def _startup(self, overrides):
        """What a worker does as it starts: config.json, build_config, build_params, params.pkl."""
        config = orjson.loads((self.src / "data/config_defaults.json").read_bytes())
        config.update(overrides)
        (self.dir / "config.json").write_bytes(orjson.dumps(config))
        built = await utils.build_config(
            web_server.emhass_conf,
            logger,
            str(self.src / "data/config_defaults.json"),
            str(self.dir / "config.json"),
            None,
        )
        params = await utils.build_params(
            web_server.emhass_conf, web_server.params_secrets, built, logger
        )
        (self.data / web_server.params_file).write_bytes(
            pickle.dumps((self.dir / "config.json", params))
        )
        return built

    async def _save(self, body):
        return await self.send("POST", "/set-config", headers=BEARER, json=body)

    async def test_a_flag_that_is_not_true_or_false_is_refused(self):
        built = await self._startup(self.SERVER)
        saved = (self.data / web_server.params_file).read_bytes()
        for key in self.FLAGS:
            for value in ("true", 1, "yes"):
                with self.subTest(key=key, value=value):
                    response = await self._save({**built, key: value})
                    self.assertEqual(response.status_code, 400)
                    # Fixed text, without the value
                    self.assertEqual(await response.get_json(), [f"{key} must be true or false"])
        self.assertEqual((self.data / web_server.params_file).read_bytes(), saved)

    async def test_true_and_false_still_save(self):
        # The web UI sends every field on each save, as JSON booleans
        built = await self._startup(self.SERVER)
        self.assertEqual((await self._save(built)).status_code, 200)
        response = await self._save({**built, "use_influxdb": False, "influxdb_verify_ssl": True})
        self.assertEqual(response.status_code, 200)
        # Turning SSL off is still a change of server
        response = await self._save({**built, "influxdb_use_ssl": False})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(await response.get_json(), [web_server.DB_SERVER_CHANGE_REFUSED])

    def _reader(self):
        """A RetrieveHass built from the saved params.pkl, as an action builds it."""
        params = pickle.loads((self.data / web_server.params_file).read_bytes())[1]
        return RetrieveHass(
            "http://127.0.0.1:1/",
            "token",
            pd.Timedelta(minutes=30),
            datetime.UTC,
            {"retrieve_hass_conf": params["retrieve_hass_conf"]},
            web_server.emhass_conf,
            logger,
        )

    async def test_hand_written_values_are_read_on_their_safe_side(self):
        # config.json edited by hand, read as at start-up
        await self._startup({"use_influxdb": "true", "influxdb_host": "127.0.0.1"})
        self.assertFalse(self._reader().use_influxdb)
        for value in ("true", "false"):
            with self.subTest(value=value):
                await self._startup(
                    {
                        "use_victoriametrics": True,
                        "victoriametrics_host": "127.0.0.1",
                        "victoriametrics_use_ssl": value,
                    }
                )
                self.assertTrue(self._reader()._vm_base_url().startswith("https://"))

    async def test_a_hand_written_ca_bundle_path_is_kept_for_influxdb(self):
        trusted, other = _TLSInflux(), _TLSInflux()
        for server in (trusted, other):
            self.addCleanup(server.close)
        ca_path = trusted.ca_file(self.dir)
        for server, connects in ((trusted, True), (other, False)):
            with self.subTest(connects=connects):
                await self._startup(
                    {**self.SERVER, "influxdb_port": server.port, "influxdb_verify_ssl": ca_path}
                )
                rh = self._reader()
                self.assertEqual(rh.influxdb_verify_ssl, ca_path)
                with self.assertLogs(logger, level="DEBUG"):
                    client = rh._init_influx_client()
                if client is not None:
                    client.close()
                self.assertEqual(client is not None, connects)
                # The login reached the server the bundle vouches for, and no other
                self.assertEqual(server.logins, [True] if connects else [])

    async def test_turning_ssl_or_the_check_on_saves_and_off_does_not(self):
        built = await self._startup(
            {**self.SERVER, "influxdb_use_ssl": False, "influxdb_verify_ssl": False}
        )
        for key in ("influxdb_use_ssl", "influxdb_verify_ssl"):
            with self.subTest(on=key):
                built[key] = True
                self.assertEqual((await self._save(built)).status_code, 200)
        for changes in (
            {"influxdb_use_ssl": False},
            {"influxdb_verify_ssl": False},
            {"influxdb_port": 8087},
        ):
            with self.subTest(changes=changes):
                response = await self._save({**built, **changes})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(await response.get_json(), [web_server.DB_SERVER_CHANGE_REFUSED])
        # On at once with another port is a change of server too
        built = await self._startup(
            {**self.SERVER, "influxdb_use_ssl": False, "influxdb_verify_ssl": False}
        )
        response = await self._save({**built, "influxdb_use_ssl": True, "influxdb_port": 443})
        self.assertEqual(response.status_code, 400)

    async def _ingress_admin_save(self, body):
        with patch.dict(
            os.environ, {"SUPERVISOR_TOKEN": "supervisor-token", api_auth.INGRESS_PORT_ENV: "8099"}
        ):
            return await self.send(
                "POST", "/set-config", port=8099, peer=SUPERVISOR, headers=ADMIN, json=body
            )

    def _saved_conf(self):
        params = pickle.loads((self.data / web_server.params_file).read_bytes())[1]
        return params["retrieve_hass_conf"], orjson.loads((self.dir / "config.json").read_bytes())

    async def test_a_ui_resave_keeps_a_hand_written_ca_path(self):
        # The web UI shows a CA path as a ticked checkbox and sends true for it. The path
        # stays, in params.pkl and config.json, for a token caller and for an ingress
        # administrator, and the rest of the save goes through
        influx_ca, vm_ca = str(self.dir / "influx-ca.pem"), str(self.dir / "vm-ca.pem")
        hand_written = {
            **self.SERVER,
            "influxdb_verify_ssl": influx_ca,
            "victoriametrics_use_ssl": True,
            "victoriametrics_verify_ssl": vm_ca,
        }
        for caller, save in (("token", self._save), ("ingress admin", self._ingress_admin_save)):
            with self.subTest(caller=caller):
                built = await self._startup(hand_written)
                body = {
                    **built,
                    "influxdb_verify_ssl": True,
                    "victoriametrics_verify_ssl": True,
                    "influxdb_database": "resaved",
                }
                response = await save(body)
                self.assertEqual(response.status_code, 200, await response.get_json())
                params, config = self._saved_conf()
                for key, path in (
                    ("influxdb_verify_ssl", influx_ca),
                    ("victoriametrics_verify_ssl", vm_ca),
                ):
                    self.assertEqual(params[key], path)
                    self.assertEqual(config[key], path)
                self.assertEqual(params["influxdb_database"], "resaved")
        # A "false" written as text is on: the ticked checkbox resaves it as true
        built = await self._startup({**self.SERVER, "influxdb_use_ssl": "false"})
        self.assertEqual((await self._save({**built, "influxdb_use_ssl": True})).status_code, 200)
        self.assertIs(self._saved_conf()[0]["influxdb_use_ssl"], True)

    async def test_false_still_turns_a_ca_path_off(self):
        # Refused for a token caller; an ingress administrator may, and false is saved
        built = await self._startup({**self.SERVER, "influxdb_verify_ssl": "/ca/bundle.pem"})
        response = await self._save({**built, "influxdb_verify_ssl": False})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(await response.get_json(), [web_server.DB_SERVER_CHANGE_REFUSED])
        self.assertEqual(self._saved_conf()[0]["influxdb_verify_ssl"], "/ca/bundle.pem")
        response = await self._ingress_admin_save({**built, "influxdb_verify_ssl": False})
        self.assertEqual(response.status_code, 200)
        self.assertIs(self._saved_conf()[0]["influxdb_verify_ssl"], False)

    async def test_the_saved_ca_path_posted_back_saves_unchanged(self):
        # GET /get-config returns the path; the web UI's JSON view, or an API client,
        # posts that reply back as it is. It counts as unchanged: saved, path kept
        ca = str(self.dir / "private-ca.pem")
        hand_written = {
            **self.SERVER,
            "influxdb_verify_ssl": ca,
            "victoriametrics_use_ssl": True,
            "victoriametrics_verify_ssl": "false",
        }
        for caller, save in (("token", self._save), ("ingress admin", self._ingress_admin_save)):
            with self.subTest(caller=caller):
                await self._startup(hand_written)
                reply = await self.send("GET", "/get-config", headers=BEARER)
                self.assertEqual(reply.status_code, 200)
                shown = await reply.get_json()
                self.assertEqual(shown["influxdb_verify_ssl"], ca)
                response = await save({**shown, "influxdb_database": "resaved"})
                self.assertEqual(response.status_code, 200, await response.get_json())
                params, config = self._saved_conf()
                self.assertEqual(params["influxdb_database"], "resaved")
                for key, value in (
                    ("influxdb_verify_ssl", ca),
                    ("victoriametrics_verify_ssl", "false"),
                ):
                    self.assertEqual(params[key], value)
                    self.assertEqual(config[key], value)
        # Any other text is still refused, for every caller
        built = await self._startup(hand_written)
        saved = (self.data / web_server.params_file).read_bytes()
        for value in (str(self.dir / "other-ca.pem"), "true", ca.upper()):
            for caller, save in (
                ("token", self._save),
                ("ingress admin", self._ingress_admin_save),
            ):
                with self.subTest(value=value, caller=caller):
                    response = await save({**built, "influxdb_verify_ssl": value})
                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(
                        await response.get_json(), ["influxdb_verify_ssl must be true or false"]
                    )
        self.assertEqual((self.data / web_server.params_file).read_bytes(), saved)

    def _edit_config_json(self, **changes):
        """Edit config.json by hand, after the last start: newer than params.pkl."""
        path = self.dir / "config.json"
        written = orjson.loads(path.read_bytes())
        written.update(changes)
        path.write_bytes(orjson.dumps(written))
        built = (self.data / web_server.params_file).stat().st_mtime_ns
        os.utime(path, ns=(built + 10**9, built + 10**9))

    async def test_a_ca_path_edited_into_config_json_since_the_start_is_kept(self):
        # params.pkl still holds the old path. A save keeps the one in config.json, which
        # the web UI showed: an ingress administrator saves it; for a token caller it is
        # a different CA file than the readers use, so refused, and nothing is written
        old_ca, new_ca = str(self.dir / "old-ca.pem"), str(self.dir / "new-ca.pem")
        built = await self._startup({**self.SERVER, "influxdb_verify_ssl": old_ca})
        self._edit_config_json(influxdb_verify_ssl=new_ca)
        body = {**built, "influxdb_verify_ssl": True}
        response = await self._save(body)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(await response.get_json(), [web_server.DB_SERVER_CHANGE_REFUSED])
        self.assertEqual(self._saved_conf()[0]["influxdb_verify_ssl"], old_ca)
        response = await self._ingress_admin_save(body)
        self.assertEqual(response.status_code, 200)
        params, config = self._saved_conf()
        self.assertEqual(params["influxdb_verify_ssl"], new_ca)
        self.assertEqual(config["influxdb_verify_ssl"], new_ca)

    async def test_a_ca_path_is_kept_from_config_json_when_params_pkl_cannot_be_read(self):
        ca = str(self.dir / "private-ca.pem")
        built = await self._startup({**self.SERVER, "influxdb_verify_ssl": ca})
        (self.data / web_server.params_file).write_bytes(b"not a pickle")
        response = await self._ingress_admin_save({**built, "influxdb_verify_ssl": True})
        self.assertEqual(response.status_code, 200)
        params, config = self._saved_conf()
        self.assertEqual(params["influxdb_verify_ssl"], ca)
        self.assertEqual(config["influxdb_verify_ssl"], ca)

    def _age_config_json(self):
        """config.json older than params.pkl, as after restoring a backup of it."""
        built = (self.data / web_server.params_file).stat().st_mtime_ns
        os.utime(self.dir / "config.json", ns=(built - 10**9, built - 10**9))

    async def test_a_config_json_older_than_params_pkl_gives_no_ca_path(self):
        # Only a config.json written after params.pkl was edited since: an older one
        # (a restored backup) leaves the params.pkl path in place, for every caller
        saved_ca, older_ca = str(self.dir / "saved-ca.pem"), str(self.dir / "older-ca.pem")
        for caller, save in (("token", self._save), ("ingress admin", self._ingress_admin_save)):
            with self.subTest(caller=caller):
                built = await self._startup({**self.SERVER, "influxdb_verify_ssl": saved_ca})
                self._edit_config_json(influxdb_verify_ssl=older_ca)
                self._age_config_json()
                response = await save({**built, "influxdb_verify_ssl": True})
                self.assertEqual(response.status_code, 200, await response.get_json())
                params, config = self._saved_conf()
                self.assertEqual(params["influxdb_verify_ssl"], saved_ca)
                self.assertEqual(config["influxdb_verify_ssl"], saved_ca)

    async def test_a_posted_path_counts_as_unchanged_only_if_it_is_the_one_kept(self):
        # The text is matched against the path the save will keep, not against either
        # file: posting the other file's path is refused, as any other text, and saves
        # nothing; posting the kept one saves it
        pkl_ca, json_ca = str(self.dir / "pkl-ca.pem"), str(self.dir / "json-ca.pem")
        refused = ["influxdb_verify_ssl must be true or false"]
        for newer, kept in (("params.pkl", pkl_ca), ("config.json", json_ca)):
            other = json_ca if kept == pkl_ca else pkl_ca
            with self.subTest(newer=newer):
                built = await self._startup({**self.SERVER, "influxdb_verify_ssl": pkl_ca})
                self._edit_config_json(influxdb_verify_ssl=json_ca)
                if newer == "params.pkl":
                    self._age_config_json()
                before = (self.data / web_server.params_file).read_bytes()
                for caller, save in (
                    ("token", self._save),
                    ("ingress admin", self._ingress_admin_save),
                ):
                    response = await save({**built, "influxdb_verify_ssl": other})
                    self.assertEqual(response.status_code, 400, caller)
                    self.assertEqual(await response.get_json(), refused)
                self.assertEqual((self.data / web_server.params_file).read_bytes(), before)
                response = await self._ingress_admin_save({**built, "influxdb_verify_ssl": kept})
                self.assertEqual(response.status_code, 200, await response.get_json())
                params, config = self._saved_conf()
                self.assertEqual(params["influxdb_verify_ssl"], kept)
                self.assertEqual(config["influxdb_verify_ssl"], kept)
        # With params.pkl unreadable the config.json path is the one kept, and matches
        ca = str(self.dir / "private-ca.pem")
        built = await self._startup({**self.SERVER, "influxdb_verify_ssl": ca})
        (self.data / web_server.params_file).write_bytes(b"not a pickle")
        response = await self._ingress_admin_save({**built, "influxdb_verify_ssl": ca})
        self.assertEqual(response.status_code, 200, await response.get_json())
        self.assertEqual(self._saved_conf()[0]["influxdb_verify_ssl"], ca)

    async def test_a_config_json_that_cannot_be_parsed_keeps_the_saved_ca_path(self):
        # Newer than params.pkl, as after a hand edit with a typo: it says nothing about
        # the CA path, so the one in params.pkl is kept, never true in its place. For the
        # ingress administrator, and for any caller while no login is stored
        ca = str(self.dir / "pinned-ca.pem")
        cases = (
            ("ingress admin", self._ingress_admin_save, self.LOGIN),
            ("token, no login", self._save, {}),
            ("token, login", self._save, self.LOGIN),
        )
        for caller, save, secrets in cases:
            for text in (b'{"influxdb_verify_ssl": "/other/ca.pem",', b"[1, 2]"):
                for posted in (True, ca):
                    with (
                        self.subTest(caller=caller, config_json=text, posted=posted),
                        patch.object(web_server, "params_secrets", dict(secrets)),
                    ):
                        built = await self._startup({**self.SERVER, "influxdb_verify_ssl": ca})
                        (self.dir / "config.json").write_bytes(text)
                        built_ns = (self.data / web_server.params_file).stat().st_mtime_ns
                        os.utime(self.dir / "config.json", ns=(built_ns + 10**9, built_ns + 10**9))
                        response = await save({**built, "influxdb_verify_ssl": posted})
                        self.assertEqual(response.status_code, 200, await response.get_json())
                        params, config = self._saved_conf()
                        self.assertEqual(params["influxdb_verify_ssl"], ca)
                        self.assertEqual(config["influxdb_verify_ssl"], ca)

    async def test_with_neither_file_readable_a_save_keeps_no_ca_path(self):
        # params.pkl cannot be read and config.json cannot be parsed: neither names a CA
        # path, so none is kept and the save is decided as one with no path in it. True
        # saves for the ingress administrator and for a token caller while no login is
        # stored; with one stored there is nothing to compare its server with. A posted
        # path matches no kept one, so it is refused as any other text
        ca = str(self.dir / "pinned-ca.pem")
        unreadable = [web_server.SAVED_PARAMS_UNREADABLE]
        refused = ["influxdb_verify_ssl must be true or false"]
        cases = (
            ("ingress admin", self._ingress_admin_save, self.LOGIN, 200),
            ("token, no login", self._save, {}, 200),
            ("token, login", self._save, self.LOGIN, 400),
        )
        for caller, save, secrets, status in cases:
            for posted in (True, ca):
                with (
                    self.subTest(caller=caller, posted=posted),
                    patch.object(web_server, "params_secrets", dict(secrets)),
                ):
                    built = await self._startup({**self.SERVER, "influxdb_verify_ssl": ca})
                    (self.data / web_server.params_file).write_bytes(b"not a pickle")
                    (self.dir / "config.json").write_bytes(b'{"influxdb_verify_ssl": ')
                    response = await save({**built, "influxdb_verify_ssl": posted})
                    if posted is not True:
                        self.assertEqual(response.status_code, 400)
                        self.assertEqual(await response.get_json(), refused)
                        self.assertEqual(
                            (self.data / web_server.params_file).read_bytes(), b"not a pickle"
                        )
                        continue
                    self.assertEqual(response.status_code, status, await response.get_json())
                    if status == 200:
                        params, config = self._saved_conf()
                        self.assertIs(params["influxdb_verify_ssl"], True)
                        self.assertIs(config["influxdb_verify_ssl"], True)
                    else:
                        self.assertEqual(await response.get_json(), unreadable)

    async def test_a_config_json_that_cannot_be_replaced_gets_a_fixed_answer(self):
        # A folder where config.json should be stands in for a file that cannot be
        # replaced. It cannot be read either, so the params.pkl path is kept and the save
        # gets as far as the write: a fixed 500, and neither file changes
        ca = str(self.dir / "pinned-ca.pem")
        config_json = self.dir / "config.json"
        for caller, save in (("token", self._save), ("ingress admin", self._ingress_admin_save)):
            for posted in (True, ca):
                with self.subTest(caller=caller, posted=posted):
                    built = await self._startup({**self.SERVER, "influxdb_verify_ssl": ca})
                    before = (self.data / web_server.params_file).read_bytes()
                    config_json.unlink()
                    config_json.mkdir()
                    try:
                        response = await save({**built, "influxdb_verify_ssl": posted})
                        self.assertEqual(response.status_code, 500)
                        self.assertEqual(await response.get_json(), ["Unable to save config file"])
                        self.assertEqual((self.data / web_server.params_file).read_bytes(), before)
                        self.assertEqual(list(config_json.iterdir()), [])
                        self.assertEqual(list(self.dir.glob("*.tmp")), [])
                    finally:
                        config_json.rmdir()

    async def test_the_saved_params_are_read_with_the_save_lock_held(self):
        built = await self._startup(self.SERVER)
        held = []
        real_read = web_server._read_saved_params

        async def read_checking_the_lock():
            with open(web_server._config_save_slot(), "a+") as other:
                taken = utils.try_lock_file(other)
                if taken:
                    utils.unlock_file(other)
            held.append(not taken)
            return await real_read()

        with patch.object(web_server, "_read_saved_params", read_checking_the_lock):
            self.assertEqual((await self._save(built)).status_code, 200)
        self.assertEqual(held, [True])

    async def test_a_refused_save_does_not_use_up_the_error_for_a_hand_written_value(self):
        # The error for a use_* that is not true or false is logged once per process; a
        # request that /set-config refuses must not be the one that logs it
        utils._db_settings_warned.clear()
        captured = []
        handler = logging.Handler(logging.DEBUG)
        handler.emit = captured.append
        db_logger = logging.getLogger("emhass.database")
        self.addCleanup(db_logger.setLevel, db_logger.level)
        db_logger.setLevel(logging.DEBUG)
        db_logger.addHandler(handler)
        self.addCleanup(db_logger.removeHandler, handler)
        built = await self._startup(self.SERVER)
        response = await self._save({**built, "use_influxdb": "x"})
        self.assertEqual(response.status_code, 400)
        self.assertFalse([r for r in captured if "use_influxdb" in r.getMessage()])
        TestDatabaseFlagReaders._rh({"use_influxdb": "x"})
        self.assertEqual(
            [r.getMessage() for r in captured if "use_influxdb" in r.getMessage()],
            ["use_influxdb is not true or false, so it is taken as off"],
        )

    async def test_a_hand_written_null_is_taken_as_not_set(self):
        # build_params drops a null, so it reads as the default, off, as before this
        # series; the web UI shows it unticked and its save changes nothing
        built = await self._startup(
            {**self.SERVER, "influxdb_use_ssl": None, "influxdb_verify_ssl": None}
        )
        rh = self._reader()
        self.assertIs(rh.influxdb_use_ssl, False)
        self.assertIs(rh.influxdb_verify_ssl, False)
        response = await self._save(
            {**built, "influxdb_use_ssl": False, "influxdb_verify_ssl": False}
        )
        self.assertEqual(response.status_code, 200)


def _written(path: pathlib.Path, text: str) -> str:
    path.write_text(text)
    return str(path)


class _TLSInflux:
    """A TLS stand-in for InfluxDB on 127.0.0.1 with a certificate from its own throwaway
    CA. Counts the connections it accepts and records, per connection, whether the
    client sent a login."""

    def __init__(self):
        import socket
        import ssl
        import threading

        import trustme

        self.ca = trustme.CA()
        self.accepted = 0
        self.logins = []
        self._ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self.ca.issue_cert("127.0.0.1").configure_cert(self._ctx)
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self._sock.settimeout(0.05)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def ca_file(self, folder: pathlib.Path) -> str:
        path = folder / f"ca-{id(self)}.pem"
        self.ca.cert_pem.write_to_path(str(path))
        return str(path)

    def _run(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            self.accepted += 1
            try:
                conn.settimeout(2)
                conn = self._ctx.wrap_socket(conn, server_side=True)
                request = conn.recv(65536).decode("latin1").lower()
                self.logins.append("authorization:" in request)
                conn.sendall(
                    b"HTTP/1.1 204 No Content\r\nX-Influxdb-Version: 1.8\r\n"
                    b"Content-Length: 0\r\nConnection: close\r\n\r\n"
                )
            except OSError:
                pass
            finally:
                conn.close()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self._sock.close()


class TestDatabaseFlagReaders(unittest.IsolatedAsyncioTestCase):
    """Every reader, and the /set-config check, take a database setting the same way, and
    each fails in its safe direction: use_influxdb and use_victoriametrics are on only
    when they are the boolean True, the SSL settings are off only when they are the
    boolean False. A hand-written config.json can still hold "true", "false" or the path
    of a CA bundle."""

    NOT_BOOL = ("true", "false", 1, 0, "yes", None)

    def setUp(self):
        utils._db_settings_warned.clear()

    @staticmethod
    def _rh(conf):
        return RetrieveHass(
            "http://127.0.0.1:1/",
            "token",
            pd.Timedelta(minutes=30),
            datetime.UTC,
            {"retrieve_hass_conf": conf},
            emhass_conf,
            logger,
        )

    def _db_log(self, db_logger=None):
        """What the "emhass.database" logger (or ``db_logger``) logs from here on."""
        captured = []
        handler = logging.Handler(logging.DEBUG)
        handler.emit = lambda record: captured.append(record)
        db_logger = db_logger or logging.getLogger("emhass.database")
        self.addCleanup(db_logger.setLevel, db_logger.level)
        db_logger.setLevel(logging.DEBUG)
        db_logger.addHandler(handler)
        self.addCleanup(db_logger.removeHandler, handler)
        return captured

    def test_a_use_flag_that_is_not_true_is_off(self):
        for value in self.NOT_BOOL:
            with self.subTest(value=value):
                self.assertFalse(self._rh({"use_influxdb": value}).use_influxdb)
                self.assertFalse(self._rh({"use_victoriametrics": value}).use_victoriametrics)

    def test_an_ssl_setting_that_is_not_false_is_on(self):
        for value in self.NOT_BOOL:
            with self.subTest(value=value):
                rh = self._rh(
                    {
                        "use_influxdb": True,
                        "influxdb_use_ssl": value,
                        "influxdb_verify_ssl": value,
                        "use_victoriametrics": True,
                        "victoriametrics_use_ssl": value,
                        "victoriametrics_verify_ssl": value,
                    }
                )
                self.assertIs(rh.influxdb_use_ssl, True)
                self.assertTrue(rh._vm_base_url().startswith("https://"))
                if isinstance(value, str):
                    # A text is taken as a CA path; none of these names one, so the
                    # database is not contacted at all (see the tests below)
                    self.assertEqual(rh.influxdb_verify_ssl, value)
                    with self.assertRaises(retrieve_hass._DbCaUnusable):
                        rh._vm_ssl()
                else:
                    # The default CAs
                    self.assertIs(rh.influxdb_verify_ssl, True)
                    self.assertIsNone(rh._vm_ssl())
        # Not set: the default, off
        rh = self._rh({"use_influxdb": True, "use_victoriametrics": True})
        self.assertIs(rh.influxdb_use_ssl, False)
        self.assertIs(rh.influxdb_verify_ssl, False)
        self.assertTrue(rh._vm_base_url().startswith("http://"))
        # The boolean False is off
        rh = self._rh(
            {
                "use_victoriametrics": True,
                "victoriametrics_use_ssl": True,
                "victoriametrics_verify_ssl": False,
            }
        )
        self.assertIs(rh._vm_ssl(), False)

    async def test_a_ca_bundle_path_is_loaded_once_off_the_event_loop(self):
        import ssl

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        server = _TLSInflux()
        self.addCleanup(server.close)
        ca_path = server.ca_file(pathlib.Path(tmp.name))
        rh = self._rh(
            {
                "use_victoriametrics": True,
                "victoriametrics_use_ssl": True,
                "victoriametrics_verify_ssl": ca_path,
            }
        )
        self.assertTrue(await rh._vm_load_ca())
        context = rh._vm_ssl()
        self.assertIsInstance(context, ssl.SSLContext)
        self.assertTrue(await rh._vm_load_ca())
        self.assertIs(rh._vm_ssl(), context)  # built once

    # (name, how to make it) of CA paths that cannot be used
    UNUSABLE_CA = (
        ("missing", lambda folder: str(folder / "missing.pem")),
        ("empty", lambda folder: _written(folder / "empty.pem", "")),
        ("junk", lambda folder: _written(folder / "junk.pem", "not a certificate")),
        ("the text false", lambda folder: "false"),
        ("a device", lambda folder: os.devnull),
    )
    UNUSABLE_CA_LINE = "The CA file named by {} cannot be loaded, so that database is not contacted"

    def _readers(self, port, path):
        """(name, the line it logs, a coroutine reading through it) for the three readers,
        each pointed at 127.0.0.1:``port`` over SSL with ``path`` as its CA."""
        conf = {
            "influxdb_host": "127.0.0.1",
            "influxdb_port": port,
            "influxdb_username": "admin",
            "influxdb_password": "real-password",
            "influxdb_use_ssl": True,
            "influxdb_verify_ssl": path,
        }
        days = pd.date_range("2026-11-19", periods=1, freq="D", tz="UTC")
        influx = self._rh({"use_influxdb": True, **conf})
        vm = self._rh(
            {
                "use_victoriametrics": True,
                "victoriametrics_host": "127.0.0.1",
                "victoriametrics_port": port,
                "victoriametrics_username": "admin",
                "victoriametrics_password": "real-password",
                "victoriametrics_use_ssl": True,
                "victoriametrics_verify_ssl": path,
            }
        )
        idx = pd.date_range("2026-11-19", periods=4, freq="30min", tz="UTC")
        thermal_args = ("sensor.t_in", "sensor.t_out", "", "", "", idx[0], idx[-1], 30, 1.0)

        async def influx_read():
            return await influx._get_data_influxdb_off_loop(days, ["sensor.x"])

        async def vm_read():
            return await vm.get_data_victoriametrics(days, ["sensor.x"])

        async def thermal_read():
            try:
                frame = await utils._fetch_influx_thermal_window(conf, *thermal_args, logger)
            except Exception:  # noqa: BLE001 - this stand-in sends no query results
                return False
            return not frame.empty

        return (
            ("InfluxDB", "influxdb_verify_ssl", influx_read),
            ("VictoriaMetrics", "victoriametrics_verify_ssl", vm_read),
            ("InfluxDB", "influxdb_verify_ssl", thermal_read),
        )

    async def test_an_unusable_ca_path_contacts_no_database(self):
        # A path naming no CA file or folder that can be loaded: no reader connects, so
        # no login leaves, where checking against the default CAs would send it to any
        # server they vouch for. The action log has a fixed line on every read;
        # "emhass.database" an error naming the setting, once per process
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = pathlib.Path(tmp.name)
        server = _TLSInflux()
        self.addCleanup(server.close)
        for name, make in self.UNUSABLE_CA:
            path = make(folder)
            utils._db_settings_warned.clear()
            captured = self._db_log()
            for reader, _key, read in self._readers(server.port, path):
                with self.subTest(path=name, reader=read.__name__):
                    for _ in range(2):
                        with self.assertLogs(logger, level="ERROR") as cm:
                            self.assertFalse(await read())
                        self.assertIn(
                            f"the configured CA file for {reader} cannot be used; "
                            "the database was not contacted",
                            [r.getMessage() for r in cm.records],
                        )
                    self.assertEqual(server.accepted, 0)
                    self.assertEqual(server.logins, [])
            lines = [r.getMessage() for r in captured if r.levelno == logging.ERROR]
            for key in ("influxdb_verify_ssl", "victoriametrics_verify_ssl"):
                self.assertEqual(lines.count(self.UNUSABLE_CA_LINE.format(key)), 1, lines)
            for line in lines:
                self.assertNotIn(str(folder), line)

    async def test_a_usable_ca_path_is_contacted_by_every_reader(self):
        # A bundle file, or a folder (here not prepared with openssl rehash, which is
        # hinted at once): the reader connects and checks against it
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = pathlib.Path(tmp.name)
        server = _TLSInflux()
        self.addCleanup(server.close)
        bundle = server.ca_file(folder)
        for path in (bundle, str(folder)):
            utils._db_settings_warned.clear()
            captured = self._db_log()
            action_log = self._db_log(logger)
            for _reader, _key, read in self._readers(server.port, path):
                with self.subTest(path=pathlib.Path(path).name, reader=read.__name__):
                    before = server.accepted
                    await read()
                    self.assertGreater(server.accepted, before)
            self.assertFalse([r for r in action_log if "CA file" in r.getMessage()])
            lines = [r.getMessage() for r in captured]
            self.assertFalse([line for line in lines if "cannot be loaded, so" in line])
            hint = [line for line in lines if "openssl rehash gives" in line]
            self.assertEqual(len(hint), 0 if path == bundle else 2, lines)

    async def test_a_ca_path_is_not_opened_on_the_event_loop(self):
        # A path that blocks (a FIFO, CON, a share that does not answer) must not hold
        # the server: it is loaded on a worker thread, never while the loop runs
        import ssl

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        server = _TLSInflux()
        self.addCleanup(server.close)
        path = server.ca_file(pathlib.Path(tmp.name))
        on_loop = []
        real = ssl.create_default_context

        def recording(*args, **kwargs):
            if kwargs.get("cafile") == path:
                on_loop.append(asyncio._get_running_loop() is not None)
            return real(*args, **kwargs)

        with patch.object(ssl, "create_default_context", recording):
            for _reader, _key, read in self._readers(server.port, path):
                await read()
        # Loaded by each of the three readers, never on the loop
        self.assertEqual(on_loop, [False, False, False])

    async def test_a_ca_load_that_blocks_is_given_up(self):
        # One that does not return in DB_CA_LOAD_TIMEOUT_S counts as unusable: the
        # action goes on, and the loop keeps serving meanwhile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        server = _TLSInflux()
        self.addCleanup(server.close)
        path = server.ca_file(pathlib.Path(tmp.name))
        release = threading.Event()
        self.addCleanup(release.set)

        def blocking(*_args):
            release.wait(5)

        self.enterContext(patch.object(utils, "DB_CA_LOAD_TIMEOUT_S", 0.3, create=True))
        self.enterContext(patch.object(utils, "db_ca_context", blocking, create=True))
        ticks = []

        async def ticker():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.02)

        task = asyncio.create_task(ticker())
        try:
            # VictoriaMetrics and the thermal reader; InfluxDB loads on its own thread
            for reader, _key, read in self._readers(server.port, path)[1:]:
                with self.subTest(reader=read.__name__):
                    start = time.monotonic()
                    with self.assertLogs(logger, level="ERROR") as cm:
                        self.assertFalse(await read())
                    self.assertLess(time.monotonic() - start, 2)
                    self.assertIn(
                        f"the configured CA file for {reader} cannot be used; "
                        "the database was not contacted",
                        [r.getMessage() for r in cm.records],
                    )
        finally:
            task.cancel()
        self.assertLess(max(b - a for a, b in itertools.pairwise(ticks)), 0.25)
        self.assertEqual(server.accepted, 0)

    async def test_a_ca_load_that_never_returns_holds_no_shared_thread(self):
        # A stat on a stale network mount can block for ever. Its load holds one thread
        # of the CA loader's own, however many reads ask for that path: asyncio.to_thread
        # and aiofiles, which every action uses to read params.pkl, keep working. A read
        # after the first wait is refused at once instead of starting another load
        import aiofiles

        release = threading.Event()
        self.addCleanup(release.set)
        loads, started = [], threading.Event()

        def never_returns(path, _key):
            loads.append(path)
            started.set()
            release.wait(30)

        self.enterContext(patch.object(utils, "DB_CA_LOAD_TIMEOUT_S", 1, create=True))
        self.enterContext(patch.object(utils, "db_ca_context", never_returns, create=True))
        shared = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(shared.shutdown, wait=False)
        asyncio.get_running_loop().set_default_executor(shared)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = str(pathlib.Path(tmp.name) / "stale-mount" / "ca.pem")
        took = []
        for n in range(6):
            start = time.monotonic()
            with self.assertLogs("emhass.database", "ERROR") as cm:
                read = asyncio.create_task(utils.db_aiohttp_ssl(path, "victoriametrics_verify_ssl"))
                if n == 0:
                    # Its load runs on its thread before the later reads are timed
                    # against it; one still queued at its due time would be dropped
                    self.assertTrue(await asyncio.to_thread(started.wait, 5))
                self.assertIsNone(await read)
                # The once-per-process line, logged here for the test's own capture
                utils._db_settings_warned.clear()
            took.append(time.monotonic() - start)
            self.assertIn(self.UNUSABLE_CA_LINE.format("victoriametrics_verify_ssl"), cm.output[0])
            self.assertEqual(await asyncio.wait_for(asyncio.to_thread(lambda: 1), 2), 1)

            async def read_this_file():
                async with aiofiles.open(__file__, "rb") as handle:
                    return len(await handle.read())

            self.assertGreater(await asyncio.wait_for(read_this_file(), 2), 0)
        self.assertEqual(loads, [path])
        self.assertGreater(took[0], 0.75)
        self.assertLess(max(took[1:]), 0.1)
        # Once it returns, the next read loads the path again
        release.set()
        for _ in range(100):
            if path not in utils._db_ca_loads:
                break
            await asyncio.sleep(0.02)
        self.assertIsNone(await utils.db_aiohttp_ssl(path, "victoriametrics_verify_ssl"))
        self.assertEqual(loads, [path, path])

    async def test_reads_asking_for_one_ca_path_at_once_share_its_load(self):
        import ssl

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        server = _TLSInflux()
        self.addCleanup(server.close)
        path = server.ca_file(pathlib.Path(tmp.name))
        real, loads = utils.db_ca_context, []

        def slow_load(load_path, key):
            loads.append(load_path)
            time.sleep(0.2)
            return real(load_path, key)

        self.enterContext(patch.object(utils, "db_ca_context", slow_load, create=True))
        first, second = await asyncio.gather(
            utils.db_aiohttp_ssl(path, "victoriametrics_verify_ssl"),
            utils.db_aiohttp_ssl(path, "influxdb_verify_ssl"),
        )
        self.assertIsInstance(first, ssl.SSLContext)
        self.assertIs(second, first)
        self.assertEqual(loads, [path])

    def _ca_loader_stand_in(self, timeout_s):
        """db_ca_context stood in for, with DB_CA_LOAD_TIMEOUT_S at ``timeout_s``: a path
        whose name starts with "stuck" blocks until ``release`` is set (by the cleanup
        at the latest, which then waits until every load has ended), any other returns a
        text. Returns (loads, release): the paths loaded, in order, and the event."""
        loads, release = [], threading.Event()

        def stand_in(path, _key):
            loads.append(path)
            if path.startswith("stuck"):
                release.wait(30)
                return None
            return f"context for {path}"

        def drain():
            release.set()
            deadline = time.monotonic() + 5
            while utils._db_ca_loads and time.monotonic() < deadline:
                time.sleep(0.01)

        self.enterContext(patch.object(utils, "DB_CA_LOAD_TIMEOUT_S", timeout_s, create=True))
        self.enterContext(patch.object(utils, "db_ca_context", stand_in, create=True))
        self.addCleanup(drain)
        return loads, release

    async def _until(self, condition, timeout_s=5):
        deadline = time.monotonic() + timeout_s
        while not condition():
            self.assertLess(time.monotonic(), deadline, "the condition did not come true")
            await asyncio.sleep(0.01)

    async def _hold_both_ca_threads(self, loads):
        """Two loads that never return (until released), running on both threads."""
        holders = [
            asyncio.create_task(utils.db_aiohttp_ssl(f"stuck-{n}", "victoriametrics_verify_ssl"))
            for n in (1, 2)
        ]
        await self._until(lambda: len(loads) >= 2)
        return holders

    async def test_a_queued_ca_load_is_dropped_only_when_its_last_caller_goes(self):
        # Both loader threads busy: reads of a good path wait for its load, queued. One
        # whose action is cancelled leaves it to the others, which get it once a thread
        # frees, well within their wait. Only when the last one goes is it dropped, and
        # then never run
        loads, release = self._ca_loader_stand_in(5)
        holders = await self._hold_both_ca_threads(loads)
        first = asyncio.create_task(utils.db_aiohttp_ssl("good-ca.pem", "influxdb_verify_ssl"))
        second = asyncio.create_task(
            utils.db_aiohttp_ssl("good-ca.pem", "victoriametrics_verify_ssl")
        )
        alone = asyncio.create_task(utils.db_aiohttp_ssl("other-ca.pem", "influxdb_verify_ssl"))
        await asyncio.sleep(0.1)
        for task in (first, alone):
            task.cancel()
        await asyncio.gather(first, alone, return_exceptions=True)
        self.assertNotIn("other-ca.pem", utils._db_ca_loads)
        await asyncio.sleep(0.05)
        release.set()
        self.assertEqual(await asyncio.wait_for(second, 4), "context for good-ca.pem")
        self.assertEqual(await asyncio.gather(*holders), [None, None])
        await self._until(lambda: utils._db_ca_executor._work_queue.empty())
        await self._until(lambda: not utils._db_ca_loads)
        self.assertEqual(sorted(loads), ["good-ca.pem", "stuck-1", "stuck-2"])

    async def test_a_new_ca_path_is_refused_at_once_while_every_loader_thread_is_held(self):
        # Two paths whose loads never return hold both threads past their due time. A
        # third path could only queue behind them, every read adding an item that is
        # never run: it is refused at once, with the usual line, and nothing is queued.
        # A load queued before then is dropped at its due time. Once a thread frees, the
        # path loads again
        loads, release = self._ca_loader_stand_in(0.3)
        holders = await self._hold_both_ca_threads(loads)
        queued = asyncio.create_task(
            utils.db_aiohttp_ssl("good-ca.pem", "victoriametrics_verify_ssl")
        )
        self.assertIsNone(await queued)
        self.assertEqual(await asyncio.gather(*holders), [None, None])
        self.assertNotIn("good-ca.pem", utils._db_ca_loads)
        # A wait can end up to one tick of a coarse clock (16 ms on Windows) before the
        # due time it waits for: be well past it
        await asyncio.sleep(0.05)
        waiting = utils._db_ca_executor._work_queue.qsize()
        for _ in range(20):
            utils._db_settings_warned.clear()
            start = time.monotonic()
            with self.assertLogs("emhass.database", "ERROR") as cm:
                self.assertIsNone(
                    await utils.db_aiohttp_ssl("good-ca.pem", "victoriametrics_verify_ssl")
                )
            self.assertLess(time.monotonic() - start, 0.1)
            self.assertIn(self.UNUSABLE_CA_LINE.format("victoriametrics_verify_ssl"), cm.output[0])
        self.assertEqual(utils._db_ca_executor._work_queue.qsize(), waiting)
        self.assertNotIn("good-ca.pem", utils._db_ca_loads)
        self.assertEqual(sorted(loads), ["stuck-1", "stuck-2"])
        release.set()
        await self._until(lambda: not utils._db_ca_loads)
        self.assertEqual(
            await utils.db_aiohttp_ssl("good-ca.pem", "victoriametrics_verify_ssl"),
            "context for good-ca.pem",
        )

    def test_only_a_regular_file_or_a_folder_is_loaded(self):
        # A FIFO, a device or a socket could block the load for ever: refused before it is
        # opened. Stood in for by a valid bundle reported as one, so it would load were it
        # not refused, on Windows too
        import ssl
        import stat

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        folder = pathlib.Path(tmp.name)
        server = _TLSInflux()
        self.addCleanup(server.close)
        bundle = server.ca_file(folder)
        key = "influxdb_verify_ssl"
        # A file is loaded as a bundle (its CA in the store now), a folder as a folder
        # (looked up by hash name during a handshake, so nothing in the store yet)
        as_file = utils.db_ca_context(bundle, key)
        self.assertIsInstance(as_file, ssl.SSLContext)
        self.assertEqual(as_file.cert_store_stats()["x509_ca"], 1)
        as_folder = utils.db_ca_context(str(folder), key)
        self.assertIsInstance(as_folder, ssl.SSLContext)
        self.assertEqual(as_folder.cert_store_stats()["x509_ca"], 0)
        real_stat = os.stat
        for kind in (stat.S_IFIFO, stat.S_IFCHR, stat.S_IFBLK, stat.S_IFSOCK):
            with self.subTest(kind=stat.filemode(kind)):

                def reported_as(path, *args, kind=kind, **kwargs):
                    if os.fspath(path) == bundle:
                        return os.stat_result((kind | 0o644, 0, 0, 1, 0, 0, 0, 0, 0, 0))
                    return real_stat(path, *args, **kwargs)

                with (
                    patch.object(utils.os, "stat", reported_as),
                    self.assertLogs("emhass.database", "ERROR"),
                ):
                    utils._db_settings_warned.clear()
                    self.assertIsNone(utils.db_ca_context(bundle, key))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "needs a FIFO")
    def test_a_fifo_is_refused_without_being_opened(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fifo = os.path.join(tmp.name, "ca.pem")
        os.mkfifo(fifo)
        result = []
        loader = threading.Thread(
            target=lambda: result.append(utils.db_ca_context(fifo, "influxdb_verify_ssl")),
            daemon=True,
        )
        with self.assertLogs("emhass.database", "ERROR"):
            loader.start()
            loader.join(2)
        if loader.is_alive():
            # Opened for reading, it waits for a writer: give it one, and an end
            os.close(os.open(fifo, os.O_WRONLY | os.O_NONBLOCK))
            loader.join(2)
            self.fail("the FIFO was opened")
        self.assertEqual(result, [None])

    async def test_over_plain_http_no_reader_checks_the_ca_path(self):
        # No certificate is checked, so a path that cannot be used stops nothing: the
        # aiohttp readers neither load it nor log that it cannot be used
        missing = str(pathlib.Path(tempfile.gettempdir()) / "p3-missing-dir" / "missing.pem")
        seen = []

        async def fetch(_session, conn, *_args):
            seen.append((conn["base_url"].split(":", 1)[0], conn["verify_ssl"]))
            return pd.DataFrame()

        idx = pd.date_range("2026-11-19", periods=4, freq="30min", tz="UTC")
        args = ("sensor.t_in", "sensor.t_out", "", "", "", idx[0], idx[-1], 30, 1000.0, logger)
        conf = {"influxdb_host": "127.0.0.1", "influxdb_use_ssl": False}
        conf["influxdb_verify_ssl"] = missing
        loaded = MagicMock(side_effect=AssertionError("loaded"))
        with (
            patch.object(utils, "db_ca_context", loaded, create=True),
            patch("emhass.utils._fetch_influx_numeric_series", side_effect=fetch),
            self.assertNoLogs(logger, "ERROR"),
        ):
            await utils._fetch_influx_thermal_window(conf, *args)
            vm = self._rh(
                {
                    "use_victoriametrics": True,
                    "victoriametrics_host": "127.0.0.1",
                    "victoriametrics_use_ssl": False,
                    "victoriametrics_verify_ssl": missing,
                }
            )
            self.assertTrue(await vm._vm_load_ca())
            self.assertIsNone(vm._vm_ssl())
        self.assertEqual(seen, [("http", True)])
        loaded.assert_not_called()

    def test_an_unusable_influxdb_ca_path_is_not_contacted(self):
        # Before, the certificate was checked against the default CAs instead (and before
        # that requests refused the path, logged as "unexpected reply")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        server = _TLSInflux()
        self.addCleanup(server.close)
        rh = self._rh(
            {
                "use_influxdb": True,
                "influxdb_host": "127.0.0.1",
                "influxdb_port": server.port,
                "influxdb_username": "admin",
                "influxdb_password": "real-password",
                "influxdb_use_ssl": True,
                "influxdb_verify_ssl": str(pathlib.Path(tmp.name) / "missing.pem"),
            }
        )
        with self.assertLogs(logger, level="ERROR") as cm:
            self.assertIsNone(rh._init_influx_client())
        self.assertEqual(
            cm.records[-1].getMessage(),
            "the configured CA file for InfluxDB cannot be used; the database was not contacted",
        )
        self.assertEqual(server.accepted, 0)
        # Over plain HTTP no certificate is checked, so the path is not needed
        rh = self._rh(
            {
                "use_influxdb": True,
                "influxdb_host": "127.0.0.1",
                "influxdb_port": 9,
                "influxdb_use_ssl": False,
                "influxdb_verify_ssl": str(pathlib.Path(tmp.name) / "missing.pem"),
            }
        )
        with self.assertLogs(logger, level="ERROR") as cm:
            self.assertIsNone(rh._init_influx_client())
        self.assertTrue(cm.output[-1].endswith("could not reach the configured InfluxDB server"))

    def test_a_non_boolean_is_warned_about_by_key_once(self):
        captured = self._db_log()
        secret = str(pathlib.Path(tempfile.gettempdir()) / "SECRET-ca.pem")
        for _ in range(2):
            self._rh(
                {
                    "use_influxdb": True,
                    "influxdb_use_ssl": "SECRET-VALUE",
                    "influxdb_verify_ssl": secret,
                }
            )
        lines = [record.getMessage() for record in captured]
        for key in ("influxdb_use_ssl", "influxdb_verify_ssl"):
            self.assertEqual(sum(line.startswith(key + " ") for line in lines), 1, lines)
        for line in lines:
            self.assertNotIn("SECRET", line)
        # An error, so that an integration taken as off is never silent
        self.assertTrue(all(record.levelno == logging.ERROR for record in captured))
        captured.clear()
        self._rh({"use_influxdb": "true"})
        self.assertEqual(
            [(r.levelno, r.getMessage()) for r in captured],
            [(logging.ERROR, "use_influxdb is not true or false, so it is taken as off")],
        )

    async def test_the_thermal_reader_takes_them_the_same_way(self):
        import ssl

        seen = []

        async def fetch(_session, conn, *_args):
            seen.append((conn["base_url"].split(":", 1)[0], conn["verify_ssl"]))
            return pd.DataFrame()

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        server = _TLSInflux()
        self.addCleanup(server.close)
        ca_path = server.ca_file(pathlib.Path(tmp.name))
        idx = pd.date_range("2026-11-19", periods=4, freq="30min", tz="UTC")
        args = ("sensor.t_in", "sensor.t_out", "", "", "", idx[0], idx[-1], 30, 1000.0, logger)
        with patch("emhass.utils._fetch_influx_numeric_series", side_effect=fetch):
            for value in self.NOT_BOOL:
                conf = {"influxdb_host": "127.0.0.1", "influxdb_use_ssl": value}
                conf["influxdb_verify_ssl"] = value
                await utils._fetch_influx_thermal_window(conf, *args)
            for conf in (
                {"influxdb_use_ssl": False, "influxdb_verify_ssl": False},
                {},
                {"influxdb_use_ssl": True, "influxdb_verify_ssl": ca_path},
            ):
                await utils._fetch_influx_thermal_window(
                    {"influxdb_host": "127.0.0.1", **conf}, *args
                )
        # A text is a CA path: one that names no file is not contacted at all
        others = [v for v in self.NOT_BOOL if not isinstance(v, str)]
        self.assertEqual(seen[: len(others)], [("https", True)] * len(others))
        self.assertEqual(seen[len(others) : -1], [("http", False)] * 2)
        self.assertEqual(seen[-1][0], "https")
        self.assertIsInstance(seen[-1][1], ssl.SSLContext)
        # use_influxdb "true" does not send the thermal read to InfluxDB
        cfg = {
            "entity_t_in": "sensor.t_in",
            "entity_t_out": "sensor.t_out",
            "entity_power": "",
            "entity_t_pred": "",
            "entity_switch": "",
            "start_utc": idx[0],
            "end_utc": idx[-1],
            "dt_minutes": 30,
            "dt_hours": 0.5,
            "nominal_power": 1000.0,
            "lookback_hours": 2,
            "sense": "heat",
        }
        with patch("emhass.utils._fetch_influx_thermal_window", new=AsyncMock()) as window:
            await utils._load_thermal_calibration_frame(
                emhass_conf, {}, {"use_influxdb": "true"}, {}, cfg, False, logger
            )
        window.assert_not_awaited()

    def test_the_server_change_check_reads_them_the_same_way(self):
        logins = utils.stored_db_logins(
            {"influxdb_password": "real-password", "victoriametrics_password": "pw"}
        )

        def changes(saved, new):
            return utils.db_endpoint_changes(saved, new, logins)

        for value in self.NOT_BOOL:
            with self.subTest(value=value):
                # A non-boolean is on, as the readers take it
                self.assertEqual(
                    changes({"influxdb_use_ssl": True}, {"influxdb_use_ssl": value}), []
                )
                self.assertEqual(
                    changes({"influxdb_use_ssl": value}, {"influxdb_use_ssl": False}),
                    ["InfluxDB"],
                )
                self.assertEqual(
                    changes({"influxdb_verify_ssl": value}, {"influxdb_verify_ssl": False}),
                    ["InfluxDB"],
                )
        for prefix, name in (("influxdb", "InfluxDB"), ("victoriametrics", "VictoriaMetrics")):
            for key in (f"{prefix}_use_ssl", f"{prefix}_verify_ssl"):
                with self.subTest(key=key):
                    # On, at the same host and port, sends the login nowhere new
                    self.assertEqual(changes({key: False}, {key: True}), [])
                    self.assertEqual(changes({}, {key: True}), [])
                    # Off is a change, and so is on together with another host
                    self.assertEqual(changes({key: True}, {key: False}), [name])
                    self.assertEqual(
                        changes({key: False}, {key: True, f"{prefix}_host": "127.0.0.9"}),
                        [name],
                    )
        # Another CA bundle, or none, is a change; the same one is not
        saved = {"influxdb_verify_ssl": "/ca/a.pem"}
        self.assertEqual(changes(saved, {"influxdb_verify_ssl": "/ca/b.pem"}), ["InfluxDB"])
        self.assertEqual(changes(saved, {"influxdb_verify_ssl": True}), ["InfluxDB"])
        self.assertEqual(changes(saved, {"influxdb_verify_ssl": False}), ["InfluxDB"])
        self.assertEqual(changes(saved, dict(saved)), [])

    def test_ssl_without_certificate_checks_is_warned_about_once(self):
        login = {"influxdb_username": "SECRET-USER", "influxdb_password": "SECRET-PASS"}
        for conf, warned in (
            ({"influxdb_use_ssl": True, "influxdb_verify_ssl": False}, True),
            ({"influxdb_use_ssl": True}, True),
            ({"influxdb_use_ssl": True, "influxdb_verify_ssl": True}, False),
            ({"influxdb_use_ssl": False}, False),
        ):
            with self.subTest(conf=conf):
                utils._db_settings_warned.clear()
                captured = self._db_log()
                with self.assertLogs(logger, level="DEBUG") as cm:
                    for _ in range(2):
                        self._rh(
                            {"use_influxdb": True, "influxdb_host": "198.51.100.7"} | login | conf
                        )
                lines = [r.getMessage() for r in captured if "certificate" in r.getMessage()]
                self.assertEqual(len(lines), 1 if warned else 0)
                for line in lines:
                    for leaked in ("198.51.100.7", "SECRET"):
                        self.assertNotIn(leaked, line)
                # Not in the action's log, where it would repeat on every action
                self.assertFalse(any("certificate" in line for line in cm.output))
        captured = self._db_log()
        self._rh(
            {
                "use_victoriametrics": True,
                "victoriametrics_host": "198.51.100.7",
                "victoriametrics_use_ssl": True,
            }
        )
        self.assertTrue(any("VictoriaMetrics is read over SSL" in r.getMessage() for r in captured))


class TestDatabaseFailureDetail(unittest.IsolatedAsyncioTestCase):
    """The cause of a database failure reaches the operator on the "emhass.database"
    logger, and never the action log a caller can read."""

    async def test_the_cause_is_logged_apart_from_the_action_log(self):
        import requests

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        data = pathlib.Path(tmp.name)
        app_logger = web_server.app.logger
        db_logger = logging.getLogger("emhass.database")
        before = (
            list(app_logger.handlers),
            app_logger.level,
            app_logger.propagate,
            list(db_logger.handlers),
            db_logger.propagate,
            db_logger.level,
        )
        captured = []
        catcher = logging.Handler(logging.DEBUG)
        catcher.emit = lambda record: captured.append(record.getMessage())
        self.enterContext(patch.object(web_server, "emhass_conf", {"data_path": data}))
        try:
            await web_server._configure_logging("DEBUG")
            # The configured level, not one inherited from an ancestor
            self.assertEqual(db_logger.level, logging.DEBUG)
            db_logger.addHandler(catcher)
            # Where the action log is written from: app.logger only, which does not
            # propagate, and neither the database logger nor an ancestor writes it
            action_log_path = os.path.abspath(data / web_server.action_log_str)
            writers = [
                name
                for name in ("emhass.web_server", "emhass.database", "emhass", "")
                for h in logging.getLogger(name).handlers
                if os.path.abspath(getattr(h, "baseFilename", "")) == action_log_path
            ]
            self.assertEqual(writers, [app_logger.name])
            self.assertFalse(app_logger.propagate)
            rh = RetrieveHass(
                "http://127.0.0.1:1/",
                "token",
                pd.Timedelta(minutes=30),
                datetime.UTC,
                {"retrieve_hass_conf": {"use_influxdb": True, "influxdb_host": "127.0.0.1"}},
                {"data_path": data},
                app_logger,
            )
            with patch("influxdb.InfluxDBClient") as client_class:
                client_class.return_value.ping.side_effect = requests.exceptions.SSLError(
                    "certificate verify failed for 127.0.0.9 SECRET-DETAIL"
                )
                self.assertIsNone(rh._init_influx_client())
            for handler in app_logger.handlers:
                handler.flush()
            action_log = (data / web_server.action_log_str).read_text()
        finally:
            for handler in app_logger.handlers:
                if handler not in before[0]:
                    app_logger.removeHandler(handler)
                    handler.close()
            app_logger.setLevel(before[1])
            app_logger.propagate = before[2]
            for handler in list(db_logger.handlers):
                if handler not in before[3]:
                    db_logger.removeHandler(handler)
            db_logger.propagate = before[4]
            db_logger.setLevel(before[5])
        self.assertIn("could not reach the configured InfluxDB server", action_log)
        for leaked in ("SECRET-DETAIL", "SSLError", "certificate verify", "127.0.0.9"):
            self.assertNotIn(leaked, action_log)
        self.assertTrue(
            any("SSLError" in line and "SECRET-DETAIL" in line for line in captured), captured
        )


if __name__ == "__main__":
    unittest.main()
