"""Security hardening: runtime parameter validation, config-only keys, request checks,
and the Supervisor token guard."""

import html
import os
import pathlib
import re
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import orjson
import pandas as pd
import pytz

from emhass import command_line, utils, web_server
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


if __name__ == "__main__":
    unittest.main()
