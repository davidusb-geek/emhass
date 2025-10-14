"""Machine learning regressor test module."""

import copy
import pathlib
import unittest

import numpy as np
import orjson
import pandas as pd
from sklearn.pipeline import Pipeline

from emhass import utils_async as utils
from emhass.command_line_async import set_input_data_dict
from emhass.machine_learning_regressor_async import MLRegressor

# The root folder
root = pathlib.Path(utils.get_root(__file__, num_parent=2))
# Build emhass_conf paths
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

# create logger
logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)

class TestMLRegressorAsync(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def get_test_params():
        # Build params with default config and secrets
        if emhass_conf["defaults_path"].exists():
            config = await utils.build_config(
                emhass_conf, logger, emhass_conf["defaults_path"]
            )
            _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
            params = await utils.build_params(emhass_conf, secrets, config, logger)
        else:
            raise Exception(
                "config_defaults. does not exist in path: "
                + str(emhass_conf["defaults_path"])
            )
        return params

    async def asyncSetUp(self):
        # parameters
        params = await TestMLRegressorAsync.get_test_params()
        costfun = "profit"
        action = "regressor-model-fit"  # fit and predict methods
        params["optim_conf"]["load_forecast_method"] = "skforecast"
        # runtime parameters
        runtimeparams = {
            "csv_file": "heating_prediction.csv",
            "features": ["degreeday", "solar"],
            "target": "hour",
            "regression_model": "LassoRegression",
            "model_type": "heating_hours_degreeday",
            "timestamp": "timestamp",
            "date_features": ["month", "day_of_week"],
            "new_values": [12.79, 4.766, 1, 2],
        }
        params["passed_data"] = runtimeparams
        runtimeparams_json = orjson.dumps(runtimeparams).decode()
        params_json = orjson.dumps(params).decode()
        # build data dictionary
        self.input_data_dict = await set_input_data_dict(
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
        # create MLRegressor object
        self.csv_file = self.input_data_dict["params"]["passed_data"]["csv_file"]
        features = self.input_data_dict["params"]["passed_data"]["features"]
        target = self.input_data_dict["params"]["passed_data"]["target"]
        regression_model = self.input_data_dict["params"]["passed_data"]["regression_model"]
        model_type = self.input_data_dict["params"]["passed_data"]["model_type"]
        timestamp = self.input_data_dict["params"]["passed_data"]["timestamp"]
        self.date_features = self.input_data_dict["params"]["passed_data"]["date_features"]
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

    # Test Regressor fit
    async def test_fit(self):
        await self.mlr.fit(self.date_features)
        self.assertIsInstance(self.mlr.model, Pipeline)

    # Test Regressor tune
    async def test_predict(self):
        await self.mlr.fit(self.date_features)
        predictions = await self.mlr.predict(self.new_values)
        self.assertIsInstance(predictions, np.ndarray)

if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
