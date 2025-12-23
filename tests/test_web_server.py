import logging
import pathlib
import pickle
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd

from emhass import web_server

# Disable logging
logging.basicConfig(level=logging.CRITICAL)


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
        web_server.emhass_conf = self.mock_conf

        # Mock params_secrets
        web_server.params_secrets = {"hass_url": "http://localhost", "long_lived_token": "token"}

    @patch("emhass.web_server.aiofiles.open")
    @patch("os.path.exists")
    async def test_index(self, mock_exists, mock_file):
        # Mock file existence for injection_dict.pkl
        mock_exists.return_value = True

        # Mock content of injection_dict.pkl
        mock_data = pickle.dumps({"table1": "<html>table</html>"})

        # Setup async file read mock
        f = AsyncMock()
        f.read.return_value = mock_data
        mock_file.return_value.__aenter__.return_value = f

        response = await self.client.get("/")
        self.assertEqual(response.status_code, 200)
        result = await response.get_data(as_text=True)
        self.assertIn("EMHASS", result)

    @patch("emhass.web_server.build_config")
    @patch("emhass.web_server.build_params")
    @patch("emhass.web_server.param_to_config")
    async def test_get_config(self, mock_p2c, mock_build_params, mock_build_config):
        # Setup mocks
        mock_build_config.return_value = {"some": "config"}
        mock_build_params.return_value = {"some": "params"}
        mock_p2c.return_value = {"final": "config"}

        response = await self.client.get("/get-config")
        self.assertEqual(response.status_code, 201)
        data = await response.get_json()
        self.assertEqual(data, {"final": "config"})

    @patch("emhass.web_server.set_input_data_dict")
    @patch("emhass.web_server.perfect_forecast_optim")
    @patch("emhass.web_server._save_injection_dict")
    @patch("emhass.web_server.get_injection_dict")
    @patch("emhass.web_server._load_params_and_runtime")
    async def test_action_perfect_optim(
        self, mock_load, mock_get_inject, mock_save, mock_optim, mock_set_input
    ):
        # Mock parameter loading
        mock_load.return_value = ({"optim_conf": {}}, "profit", "{}")

        # Mock input data set
        mock_set_input.return_value = {
            "retrieve_hass_conf": {"continual_publish": False},
            "some": "data",
        }

        # Mock optimization result
        mock_df = pd.DataFrame()
        mock_optim.return_value = mock_df

        # Mock injection dict
        mock_get_inject.return_value = {}

        response = await self.client.post("/action/perfect-optim", json={})

        self.assertEqual(response.status_code, 201)
        self.assertIn("Action perfect-optim executed", await response.get_data(as_text=True))

        mock_optim.assert_called_once()

    @patch("emhass.web_server.export_influxdb_to_csv")
    @patch("emhass.web_server._load_params_and_runtime")
    async def test_action_export_csv(self, mock_load, mock_export):
        mock_load.return_value = ({}, "profit", "{}")
        mock_export.return_value = True  # Success

        response = await self.client.post("/action/export-influxdb-to-csv", json={})

        self.assertEqual(response.status_code, 201)
        mock_export.assert_called_once()

    @patch("emhass.web_server.set_input_data_dict")
    @patch("emhass.web_server.forecast_model_fit")
    @patch("emhass.web_server.get_injection_dict_forecast_model_fit")
    @patch("emhass.web_server._save_injection_dict")
    @patch("emhass.web_server._load_params_and_runtime")
    async def test_action_forecast_model_fit(
        self, mock_load, mock_save, mock_get_inject_ml, mock_fit, mock_set_input
    ):
        # Mock parameter loading
        mock_load.return_value = ({}, "profit", "{}")

        # Mock input data set
        mock_set_input.return_value = {
            "retrieve_hass_conf": {"continual_publish": False},
            "data": "test",
        }

        # Mock fit return (df, something, mlf)
        mock_fit.return_value = (pd.DataFrame(), None, MagicMock())

        # Mock injection dict for ML (prevent util function crash)
        mock_get_inject_ml.return_value = {}

        response = await self.client.post("/action/forecast-model-fit", json={})

        self.assertEqual(response.status_code, 201)
        mock_fit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
