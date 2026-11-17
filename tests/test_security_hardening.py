"""Security hardening: runtime parameter validation, config-only keys, request checks,
and the Supervisor token guard."""

import html
import os
import pathlib
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pandas as pd
import pytz

from emhass import command_line, retrieve_hass, utils, web_server
from emhass.retrieve_hass import RetrieveHass, _is_supervisor_url
from emhass.utils import RUNTIME_DENIED_KEYS, RuntimeParamError
from tests.aioresponses_compat import aioresponses

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

    async def test_denied_keys_cover_every_connection_key_in_associations(self):
        lines = (emhass_conf["associations_path"]).read_text(encoding="utf-8").splitlines()
        names = {line.split(",")[2] for line in lines[1:] if line}
        expected = {n for n in names if "influx" in n or "victoria" in n}
        expected |= {"use_websocket", "hass_url", "long_lived_token", "ssl_no_verify"}
        expected.discard("use_influxdb")
        self.assertTrue(expected <= RUNTIME_DENIED_KEYS, expected - RUNTIME_DENIED_KEYS)
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
            with self.subTest(set_type=set_type), self.assertRaises(RuntimeParamError):
                await command_line.set_input_data_dict(
                    emhass_conf,
                    "profit",
                    self.params_json,
                    orjson.dumps(runtime).decode("utf-8"),
                    set_type,
                    logger,
                )
        mock_rh.assert_not_called()

    @patch("emhass.command_line.Forecast")
    async def test_weather_forecast_cache_rejects_before_fetching(self, mock_fcst):
        with self.assertRaises(RuntimeParamError):
            await command_line.weather_forecast_cache(
                emhass_conf, self.params_json, orjson.dumps({"model_type": "../x"}).decode(), logger
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
        for url in (
            self.SUPERVISOR,
            self.SUPERVISOR + "/",
            "https://supervisor/core/api/",
            "empty",
            "",
        ):
            with self.subTest(url=url):
                self.assertEqual(self._rh(url, "sv-token").long_lived_token, "sv-token")

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
        # A known reason is still named, in fixed words
        self.assertTrue(any("database not found" in line for line in cm.output))

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
        self.assertTrue(any("HTTP 400" in line for line in cm.output))
        self.assertFalse(any("SECRET" in line for line in cm.output))

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
            self.assertTrue(any("does not follow" in line for line in cm.output))
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


if __name__ == "__main__":
    unittest.main()
