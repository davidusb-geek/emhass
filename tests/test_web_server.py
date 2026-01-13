import logging
import pathlib
import pickle
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pandas as pd

from emhass import web_server

# Disable logging propagation to avoid spamming console during tests
logging.getLogger("quart.app").setLevel(logging.CRITICAL)
logging.getLogger("emhass").setLevel(logging.CRITICAL)


class TestWebServer(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Create a test client
        self.client = web_server.app.test_client()
        # Mock the global emhass_conf in web_server module
        self.mock_conf = {
            "data_path": pathlib.Path("/tmp/emhass/data"),
            "config_path": pathlib.Path("/tmp/emhass/config.json"),
            "defaults_path": pathlib.Path("/tmp/emhass/defaults.json"),
            "associations_path": pathlib.Path("/tmp/emhass/assoc.csv"),
            "legacy_config_path": pathlib.Path("/tmp/emhass/legacy.yaml"),
            "root_path": pathlib.Path("/tmp/emhass/root"),
        }
        # Save original config to prevent leaks
        self.original_conf = web_server.emhass_conf.copy()
        web_server.emhass_conf = self.mock_conf
        # Mock params_secrets
        web_server.params_secrets = {"hass_url": "http://localhost", "long_lived_token": "token"}
        # Save original handlers to restore later
        self.original_handlers = web_server.app.logger.handlers[:]

    async def asyncTearDown(self):
        # Restore original config
        web_server.emhass_conf = self.original_conf
        # Restore original log handlers to prevent 'MagicMock' level errors in subsequent tests
        web_server.app.logger.handlers = self.original_handlers

    def test_mark_safe(self):
        self.assertEqual(web_server.mark_safe(None), "")
        self.assertEqual(web_server.mark_safe("<b>test</b>"), "<b>test</b>")

    @patch("emhass.web_server.aiofiles.open")
    @patch("os.path.exists")
    async def test_index(self, mock_exists, mock_file):
        mock_exists.return_value = True
        mock_data = pickle.dumps({"table1": "<html>table</html>"})
        f = AsyncMock()
        f.read.return_value = mock_data
        mock_file.return_value.__aenter__.return_value = f
        response = await self.client.get("/")
        self.assertEqual(response.status_code, 200)
        result = await response.get_data(as_text=True)
        self.assertIn("EMHASS", result)

    @patch("emhass.web_server.aiofiles.open")
    @patch("os.path.exists")
    async def test_configuration(self, mock_exists, mock_file):
        mock_exists.return_value = True
        mock_data = pickle.dumps((pathlib.Path("/config"), {"some": "param"}))
        f = AsyncMock()
        f.read.return_value = mock_data
        mock_file.return_value.__aenter__.return_value = f
        response = await self.client.get("/configuration")
        self.assertEqual(response.status_code, 200)

    @patch("emhass.web_server.aiofiles.open")
    @patch("os.path.exists")
    async def test_template_action(self, mock_exists, mock_file):
        mock_exists.return_value = True
        mock_data = pickle.dumps({"table1": "data"})
        f = AsyncMock()
        f.read.return_value = mock_data
        mock_file.return_value.__aenter__.return_value = f
        response = await self.client.get("/template")
        self.assertEqual(response.status_code, 200)

    @patch("emhass.web_server.build_config")
    @patch("emhass.web_server.build_params")
    @patch("emhass.web_server.param_to_config")
    async def test_get_config(self, mock_p2c, mock_build_params, mock_build_config):
        mock_build_config.return_value = {"some": "config"}
        mock_build_params.return_value = {"some": "params"}
        mock_p2c.return_value = {"final": "config"}
        response = await self.client.get("/get-config")
        self.assertEqual(response.status_code, 201)
        data = await response.get_json()
        self.assertEqual(data, {"final": "config"})

    @patch("emhass.web_server.build_config")
    @patch("emhass.web_server.build_params")
    @patch("emhass.web_server.param_to_config")
    async def test_config_get_defaults(self, mock_p2c, mock_build_params, mock_build_config):
        mock_build_config.return_value = {"default": "config"}
        mock_build_params.return_value = {"default": "params"}
        mock_p2c.return_value = {"final": "default"}
        response = await self.client.get("/get-config/defaults")
        self.assertEqual(response.status_code, 201)
        data = await response.get_json()
        self.assertEqual(data, {"final": "default"})

    @patch("emhass.web_server.build_legacy_config_params")
    @patch("emhass.web_server.build_params")
    @patch("emhass.web_server.param_to_config")
    async def test_json_convert(self, mock_p2c, mock_build_params, mock_legacy):
        mock_legacy.return_value = None
        mock_build_params.return_value = {"converted": "params"}
        mock_p2c.return_value = {"converted": "config"}
        # Test successful conversion
        yaml_data = b"foo: bar"
        response = await self.client.post("/get-json", data=yaml_data)
        self.assertEqual(response.status_code, 201)
        data = await response.get_data()
        self.assertEqual(orjson.loads(data), {"converted": "config"})
        # Test invalid YAML
        # Expect 500 because the app currently lets YAMLError propagate
        response = await self.client.post("/get-json", data=b": - invalid")
        self.assertEqual(response.status_code, 500)

    @patch("emhass.web_server.aiofiles.open")
    @patch("os.path.exists")
    @patch("emhass.web_server.build_params")
    @patch("emhass.web_server.param_to_config")
    async def test_parameter_set(self, mock_p2c, mock_build_params, mock_exists, mock_file):
        mock_exists.return_value = True
        f_defaults = AsyncMock()
        f_defaults.read.return_value = orjson.dumps({"default": 1})
        f_write = AsyncMock()
        mock_file.return_value.__aenter__.side_effect = [f_defaults, f_write, f_write]
        mock_build_params.return_value = {"new": "params"}
        mock_p2c.return_value = {"new": "config"}
        response = await self.client.post("/set-config", json={"some": "data"})
        self.assertEqual(response.status_code, 201)
        self.assertTrue(f_write.write.called)

    @patch("emhass.web_server.set_input_data_dict")
    @patch("emhass.web_server.perfect_forecast_optim")
    @patch("emhass.web_server._save_injection_dict")
    @patch("emhass.web_server.get_injection_dict")
    @patch("emhass.web_server._load_params_and_runtime")
    async def test_action_perfect_optim(
        self, mock_load, mock_get_inject, mock_save, mock_optim, mock_set_input
    ):
        mock_load.return_value = ({"optim_conf": {}}, "profit", "{}")
        mock_set_input.return_value = {"retrieve_hass_conf": {"continual_publish": False}}
        mock_optim.return_value = pd.DataFrame()
        mock_get_inject.return_value = {}
        response = await self.client.post("/action/perfect-optim", json={})
        self.assertEqual(response.status_code, 201)
        mock_optim.assert_called_once()

    @patch("emhass.web_server.export_influxdb_to_csv")
    @patch("emhass.web_server._load_params_and_runtime")
    async def test_action_export_csv(self, mock_load, mock_export):
        mock_load.return_value = ({}, "profit", "{}")
        mock_export.return_value = True
        response = await self.client.post("/action/export-influxdb-to-csv", json={})
        self.assertEqual(response.status_code, 201)

    @patch("emhass.web_server.grab_log")
    @patch("emhass.web_server.export_influxdb_to_csv")
    @patch("emhass.web_server._load_params_and_runtime")
    async def test_action_export_csv_failure(self, mock_load, mock_export, mock_grab_log):
        mock_load.return_value = ({}, "profit", "{}")
        mock_export.return_value = False
        mock_grab_log.return_value = ["Error log"]
        response = await self.client.post("/action/export-influxdb-to-csv", json={})
        # App returns 400 for failures in this specific action
        self.assertEqual(response.status_code, 400)

    @patch("emhass.web_server.set_input_data_dict")
    @patch("emhass.web_server.forecast_model_fit")
    @patch("emhass.web_server.get_injection_dict_forecast_model_fit")
    @patch("emhass.web_server._save_injection_dict")
    @patch("emhass.web_server._load_params_and_runtime")
    async def test_action_forecast_model_fit(
        self, mock_load, mock_save, mock_get_inject, mock_fit, mock_set_input
    ):
        mock_load.return_value = ({}, "profit", "{}")
        mock_set_input.return_value = {"retrieve_hass_conf": {"continual_publish": False}}
        mock_fit.return_value = (pd.DataFrame(), None, MagicMock())
        mock_get_inject.return_value = {}
        response = await self.client.post("/action/forecast-model-fit", json={})
        self.assertEqual(response.status_code, 201)
        mock_fit.assert_called_once()

    @patch("emhass.web_server.set_input_data_dict")
    @patch("emhass.web_server.forecast_model_predict")
    @patch("emhass.web_server._save_injection_dict")
    @patch("emhass.web_server._load_params_and_runtime")
    async def test_action_forecast_model_predict(
        self, mock_load, mock_save, mock_predict, mock_set_input
    ):
        mock_load.return_value = ({}, "profit", "{}")
        mock_set_input.return_value = {"retrieve_hass_conf": {"continual_publish": False}}
        # Success
        mock_predict.return_value = pd.DataFrame({"col": [1, 2]})
        response = await self.client.post("/action/forecast-model-predict", json={})
        self.assertEqual(response.status_code, 201)
        # Fail
        mock_predict.return_value = None
        # Mock check_file_log to return False, simulating no error in log, but code returns 400
        with patch("emhass.web_server.check_file_log", new=AsyncMock(return_value=False)):
            response = await self.client.post("/action/forecast-model-predict", json={})
            self.assertEqual(response.status_code, 400)

    @patch("emhass.web_server.set_input_data_dict")
    @patch("emhass.web_server.forecast_model_tune")
    @patch("emhass.web_server.get_injection_dict_forecast_model_tune")
    @patch("emhass.web_server._save_injection_dict")
    @patch("emhass.web_server._load_params_and_runtime")
    async def test_action_forecast_model_tune(
        self, mock_load, mock_save, mock_get_inject, mock_tune, mock_set_input
    ):
        mock_load.return_value = ({}, "profit", "{}")
        mock_set_input.return_value = {"retrieve_hass_conf": {"continual_publish": False}}
        mock_get_inject.return_value = {}
        # Success case
        mock_tune.return_value = (pd.DataFrame(), MagicMock())
        response = await self.client.post("/action/forecast-model-tune", json={})
        self.assertEqual(response.status_code, 201)
        # Fail case
        mock_tune.return_value = (None, None)
        with patch("emhass.web_server.check_file_log", new=AsyncMock(return_value=False)):
            response = await self.client.post("/action/forecast-model-tune", json={})
            self.assertEqual(response.status_code, 400)

    @patch("emhass.web_server.set_input_data_dict")
    @patch("emhass.web_server.regressor_model_fit")
    @patch("emhass.web_server._load_params_and_runtime")
    async def test_action_regressor_model_fit(self, mock_load, mock_fit, mock_set_input):
        mock_load.return_value = ({}, "profit", "{}")
        mock_set_input.return_value = {"retrieve_hass_conf": {"continual_publish": False}}
        mock_fit.return_value = True
        response = await self.client.post("/action/regressor-model-fit", json={})
        self.assertEqual(response.status_code, 201)

    @patch("emhass.web_server.set_input_data_dict")
    @patch("emhass.web_server.regressor_model_predict")
    @patch("emhass.web_server._load_params_and_runtime")
    async def test_action_regressor_model_predict(self, mock_load, mock_predict, mock_set_input):
        mock_load.return_value = ({}, "profit", "{}")
        mock_set_input.return_value = {"retrieve_hass_conf": {"continual_publish": False}}
        mock_predict.return_value = True
        response = await self.client.post("/action/regressor-model-predict", json={})
        self.assertEqual(response.status_code, 201)

    @patch("emhass.web_server.aiofiles.open")
    async def test_check_file_log(self, mock_file):
        # Mock the path object in global config
        mock_path = MagicMock()
        web_server.emhass_conf["data_path"] = mock_path
        # Configure file path behavior
        mock_file_path = MagicMock()
        mock_path.__truediv__.return_value = mock_file_path
        # Case 1: File exists and has error
        mock_file_path.exists.return_value = True
        f = AsyncMock()
        # FIX: The split logic in check_file_log splits on " ".
        # "ERROR: problem" -> ["ERROR:", "problem"]. "ERROR:" != "ERROR"
        # "ERROR - problem" -> ["ERROR", "-", "problem"]. "ERROR" == "ERROR"
        f.read.return_value = "INFO: normal\nERROR - bad things\n"
        mock_file.return_value.__aenter__.return_value = f
        self.assertTrue(await web_server.check_file_log())
        # Case 2: File exists, no error
        f.read.return_value = "INFO: normal\n"
        self.assertFalse(await web_server.check_file_log())
        # Case 3: File missing
        mock_file_path.exists.return_value = False
        result = await web_server.check_file_log()
        self.assertFalse(result)

    @patch("emhass.web_server.aiofiles.open")
    async def test_grab_log(self, mock_file):
        mock_path = MagicMock()
        web_server.emhass_conf["data_path"] = mock_path
        mock_file_path = MagicMock()
        mock_path.__truediv__.return_value = mock_file_path
        mock_file_path.exists.return_value = True
        f = AsyncMock()
        f.read.return_value = "INFO: step 1\nINFO: step 2\n"
        mock_file.return_value.__aenter__.return_value = f
        lines = await web_server.grab_log("step 1")
        self.assertIn("INFO: step 2", lines)

    @patch("emhass.web_server.aiofiles.open")
    async def test_clear_file_log(self, mock_file):
        mock_path = MagicMock()
        web_server.emhass_conf["data_path"] = mock_path
        mock_file_path = MagicMock()
        mock_path.__truediv__.return_value = mock_file_path
        mock_file_path.exists.return_value = True
        f = AsyncMock()
        mock_file.return_value.__aenter__.return_value = f
        await web_server.clear_file_log()
        f.write.assert_called_with("")

    @patch("emhass.web_server.initialize")
    async def test_before_serving(self, mock_init):
        # Happy path
        await web_server.before_serving()
        mock_init.assert_called_once()
        # Error path
        mock_init.side_effect = Exception("init failed")
        # Should not raise
        await web_server.before_serving()

    @patch("emhass.web_server.is_connected")
    @patch("emhass.web_server.close_global_connection")
    async def test_after_serving(self, mock_close, mock_is_conn):
        # Connected
        mock_is_conn.return_value = True
        await web_server.after_serving()
        mock_close.assert_called_once()
        # Not connected
        mock_is_conn.return_value = False
        mock_close.reset_mock()
        await web_server.after_serving()
        mock_close.assert_not_called()

    @patch("pathlib.Path.exists")
    @patch("emhass.web_server.aiofiles.open")
    @patch("emhass.web_server.set_input_data_dict", new_callable=AsyncMock)
    @patch("emhass.web_server.dayahead_forecast_optim", new_callable=AsyncMock)
    @patch("emhass.web_server.get_injection_dict")
    @patch("emhass.web_server.app.logger")
    async def test_action_malformed_json(
        self,
        mock_logger,
        mock_get_inject,
        mock_optim,
        mock_set_input,
        mock_file,
        mock_path_exists,
    ):
        """
        Test that sending malformed JSON (e.g. Python syntax 'False' instead of 'false')
        triggers a logged error and proceeds with empty runtime parameters.
        """
        mock_path_exists.return_value = True

        # Mock .read() for everything
        def aiofiles_side_effect(file, mode="r", *args, **kwargs):
            context_mock = AsyncMock()
            file_handle = AsyncMock()
            context_mock.__aenter__.return_value = file_handle
            filename = str(file)
            if "params.pkl" in filename:
                content = pickle.dumps((pathlib.Path("/config"), {"optim_conf": {}}))
                file_handle.read.return_value = content
            else:
                log_content = (
                    "2024-01-01 12:00:00,000 INFO >> Obtaining params\n"
                    "2024-01-01 12:00:01,000 INFO >> Setting input data dict\n"
                    "2024-01-01 12:00:02,000 INFO >> Performing optimization..."
                )
                file_handle.read.return_value = log_content
            return context_mock

        mock_file.side_effect = aiofiles_side_effect
        mock_set_input.return_value = {
            "params": {},
            "retrieve_hass_conf": {"continual_publish": False},
        }
        mock_optim.return_value = MagicMock()
        mock_get_inject.return_value = {
            "optim_status": "Optimal",
            "table1": "<div>Table</div>",
            "div1": "<div>Plot1</div>",
            "div2": "<div>Plot2</div>",
        }
        # Execute Test
        response = await self.client.post(
            "/action/dayahead-optim",
            data='{"perform_backtest": False}',
            headers={"Content-Type": "application/json"},
        )
        # Assertions
        self.assertEqual(response.status_code, 201)
        log_found = False
        for call in mock_logger.error.call_args_list:
            args, _ = call
            if args and "Error parsing runtime params JSON" in str(args[0]):
                log_found = True
                break
        self.assertTrue(log_found, "Expected error log message not found")
        # Verify downstream call received empty string/dict for runtime params
        call_args = mock_set_input.call_args
        # runtimeparams is the 4th argument (index 3)
        self.assertEqual(call_args[0][3], "{}")

    @patch("pathlib.Path.exists")
    @patch("emhass.web_server.aiofiles.open")
    @patch("emhass.web_server.set_input_data_dict", new_callable=AsyncMock)
    @patch("emhass.web_server.dayahead_forecast_optim", new_callable=AsyncMock)
    @patch("emhass.web_server.get_injection_dict")
    @patch("emhass.web_server.app.logger")
    async def test_action_valid_json(
        self,
        mock_logger,
        mock_get_inject,
        mock_optim,
        mock_set_input,
        mock_file,
        mock_path_exists,
    ):
        """Test that sending valid JSON works correctly."""
        mock_path_exists.return_value = True

        # Reuse side effect
        def aiofiles_side_effect(file, mode="r", *args, **kwargs):
            context_mock = AsyncMock()
            file_handle = AsyncMock()
            context_mock.__aenter__.return_value = file_handle
            if "params.pkl" in str(file):
                content = pickle.dumps((pathlib.Path("/config"), {"optim_conf": {}}))
                file_handle.read.return_value = content
            else:
                file_handle.read.return_value = "INFO >> Success"
            return context_mock

        mock_file.side_effect = aiofiles_side_effect
        mock_set_input.return_value = {
            "params": {},
            "retrieve_hass_conf": {"continual_publish": False},
        }
        mock_optim.return_value = MagicMock()
        mock_get_inject.return_value = {"optim_status": "Optimal"}
        # Execute Test with valid JSON
        valid_json_str = '{"perform_backtest": false}'
        response = await self.client.post(
            "/action/dayahead-optim",
            data=valid_json_str,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 201)
        # Verify no error was logged
        self.assertFalse(mock_logger.error.called)
        # Verify runtime params were passed correctly
        call_args = mock_set_input.call_args
        actual_arg = call_args[0][3]
        # Compare as dictionaries to ignore whitespace differences (e.g. " : " vs ":")
        self.assertEqual(orjson.loads(actual_arg), orjson.loads(valid_json_str))

    @patch("os.getenv")
    async def test_setup_paths(self, mock_getenv):
        # Configure env vars
        env_vars = {
            "DATA_PATH": "/env/data",
            "CONFIG_PATH": "/env/config.json",
            "OPTIONS_PATH": "/env/options.json",
            "ROOT_PATH": "/env/root",
            "DEFAULTS_PATH": "/env/defaults.json",
            "ASSOCIATIONS_PATH": "/env/assoc.csv",
            "LEGACY_CONFIG_PATH": "/env/legacy.yaml",
        }
        mock_getenv.side_effect = lambda key, default=None: env_vars.get(key, default)
        # Run the helper
        paths = await web_server._setup_paths()
        # Unpack results
        (config_path, options_path, defaults_path, assoc_path, legacy_path, root_path) = paths
        # Assertions: Compare against pathlib objects to handle OS separators automatically
        self.assertEqual(config_path, pathlib.Path("/env/config.json"))
        self.assertEqual(web_server.emhass_conf["data_path"], pathlib.Path("/env/data"))
        self.assertEqual(web_server.emhass_conf["root_path"], pathlib.Path("/env/root"))

    @patch("pathlib.Path.mkdir")
    @patch("os.path.isdir")
    def test_validate_data_path(self, mock_isdir, mock_mkdir):
        root = pathlib.Path("/root")
        # Case 1: Path exists
        web_server.emhass_conf["data_path"] = pathlib.Path("/existing/data")
        mock_isdir.return_value = True
        web_server._validate_data_path(root)
        self.assertEqual(web_server.emhass_conf["data_path"], pathlib.Path("/existing/data"))
        # Case 2: Path missing, fallback to /data/ exists
        web_server.emhass_conf["data_path"] = pathlib.Path("/missing/data")
        # isdir side effect: First call (check emhass_conf) -> False, Second call (check /data/) -> True
        mock_isdir.side_effect = [False, True]
        web_server._validate_data_path(root)
        self.assertEqual(web_server.emhass_conf["data_path"], pathlib.Path("/data"))
        # Case 3: Path missing, fallback missing -> create root/data
        web_server.emhass_conf["data_path"] = pathlib.Path("/missing/data")
        mock_isdir.side_effect = [False, False]
        web_server._validate_data_path(root)
        self.assertEqual(web_server.emhass_conf["data_path"], root / "data/")
        mock_mkdir.assert_called()

    @patch("emhass.web_server.get_websocket_client")
    async def test_initialize_connections(self, mock_ws_client):
        # Re-enable logging for this test so assertLogs can capture output
        logger = logging.getLogger("emhass")
        original_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            # Case 1: Websocket Enabled - Success
            params = {"retrieve_hass_conf": {"use_websocket": True}}
            await web_server._initialize_connections(params)
            mock_ws_client.assert_called()
            # Case 2: Websocket Enabled - Failure
            mock_ws_client.side_effect = ConnectionError("Connection fail")
            with self.assertRaises(ConnectionError):
                await web_server._initialize_connections(params)
            # Case 3: InfluxDB Enabled
            mock_ws_client.reset_mock()
            params = {"retrieve_hass_conf": {"use_websocket": False, "use_influxdb": True}}
            with self.assertLogs("emhass", level="INFO") as cm:
                await web_server._initialize_connections(params)
            self.assertTrue(any("InfluxDB mode enabled" in log for log in cm.output))
            mock_ws_client.assert_not_called()
            # Case 4: Neither (REST API)
            params = {"retrieve_hass_conf": {"use_websocket": False, "use_influxdb": False}}
            with self.assertLogs("emhass", level="INFO") as cm:
                await web_server._initialize_connections(params)
            self.assertTrue(any("using REST API" in log for log in cm.output))
        finally:
            # Restore original logging level
            logger.setLevel(original_level)


if __name__ == "__main__":
    unittest.main()
