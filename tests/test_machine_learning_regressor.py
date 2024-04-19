"""Machine learning regressor test module."""

import copy
import json
import pathlib
import unittest

import numpy as np
import pandas as pd
import yaml
from emhass import utils
from emhass.command_line import set_input_data_dict
from emhass.machine_learning_regressor import MLRegressor
from sklearn.pipeline import Pipeline

# the root folder
root = str(utils.get_root(__file__, num_parent=2))
emhass_conf = {}
emhass_conf["config_path"] = pathlib.Path(root) / "config_emhass.yaml"
emhass_conf["data_path"] = pathlib.Path(root) / "data/"
emhass_conf["root_path"] = pathlib.Path(root)
# create logger
logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)


class TestMLRegressor(unittest.TestCase):
    @staticmethod
    def get_test_params():
        with open(emhass_conf["config_path"]) as file:
            params = yaml.safe_load(file)
        params.update(
            {
                "params_secrets": {
                    "hass_url": "http://supervisor/core/api",
                    "long_lived_token": "${SUPERVISOR_TOKEN}",
                    "time_zone": "Europe/Paris",
                    "lat": 45.83,
                    "lon": 6.86,
                    "alt": 8000.0,
                },
            },
        )
        return params

    def setUp(self):
        params = TestMLRegressor.get_test_params()
        params_json = json.dumps(params)
        costfun = "profit"
        action = "regressor-model-fit"  # fit and predict methods
        params = copy.deepcopy(json.loads(params_json))
        runtimeparams = {
            "csv_file": "heating_prediction.csv",
            "features": ["degreeday", "solar"],
            "target": "hour",
            "regression_model": "AdaBoostRegression",
            "model_type": "heating_hours_degreeday",
            "timestamp": "timestamp",
            "date_features": ["month", "day_of_week"],
            "new_values": [12.79, 4.766, 1, 2],
        }
        runtimeparams_json = json.dumps(runtimeparams)
        params["passed_data"] = runtimeparams
        params["optim_conf"]["load_forecast_method"] = "skforecast"
        params_json = json.dumps(params)
        self.input_data_dict = set_input_data_dict(
            emhass_conf,
            costfun,
            params_json,
            runtimeparams_json,
            action,
            logger,
            get_data_from_file=True,
        )
        data = copy.deepcopy(self.input_data_dict["df_input_data"])
        self.assertIsInstance(data, pd.DataFrame)
        self.csv_file = self.input_data_dict["params"]["passed_data"]["csv_file"]
        features = self.input_data_dict["params"]["passed_data"]["features"]
        target = self.input_data_dict["params"]["passed_data"]["target"]
        regression_model = self.input_data_dict["params"]["passed_data"][
            "regression_model"
        ]
        model_type = self.input_data_dict["params"]["passed_data"]["model_type"]
        timestamp = self.input_data_dict["params"]["passed_data"]["timestamp"]
        self.date_features = self.input_data_dict["params"]["passed_data"][
            "date_features"
        ]
        self.new_values = self.input_data_dict["params"]["passed_data"]["new_values"]
        self.mlr = MLRegressor(
            data,
            model_type,
            regression_model,
            features,
            target,
            timestamp,
            logger,
        )

    def test_fit(self):
        self.mlr.fit(self.date_features)
        self.assertIsInstance(self.mlr.model, Pipeline)

    def test_predict(self):
        self.mlr.fit(self.date_features)
        predictions = self.mlr.predict(self.new_values)
        self.assertIsInstance(predictions, np.ndarray)


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
