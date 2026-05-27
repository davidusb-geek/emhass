import pathlib
import unittest

import numpy as np
import orjson
import pandas as pd

from emhass import utils
from emhass.optimization import Optimization

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "defaults_path": root / "src/emhass/data/config_defaults.json",
    "associations_path": root / "src/emhass/data/associations.csv",
}
logger, _ = utils.get_logger(__name__, emhass_conf, save_to_file=False)

_VAR_LOAD_COST = "unit_load_cost"
_VAR_PROD_PRICE = "unit_prod_price"


class TestDefaultsSmoke(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
        params = await utils.build_params(emhass_conf, secrets, config, logger)
        self.params_json = orjson.dumps(params).decode("utf-8")
        self.retrieve_hass_conf, self.optim_conf, self.plant_conf = utils.get_yaml_parse(
            self.params_json, logger
        )

    def _make_forecast_inputs(self, n: int = 48):
        """Build minimal constant-value forecast inputs."""
        freq = self.retrieve_hass_conf["optimization_time_step"]
        tz = self.retrieve_hass_conf["time_zone"]
        idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz=tz)
        p_pv = np.zeros(n)
        p_load = np.full(n, 1000.0)
        ulc = np.full(n, 0.2)
        upp = np.full(n, 0.1)
        df = pd.DataFrame({_VAR_LOAD_COST: ulc, _VAR_PROD_PRICE: upp}, index=idx)
        return df, p_pv, p_load, ulc, upp

    def _build_and_run(self, retrieve_hass_conf, optim_conf, plant_conf):
        df, p_pv, p_load, ulc, upp = self._make_forecast_inputs()
        opt = Optimization(
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            _VAR_LOAD_COST,
            _VAR_PROD_PRICE,
            "profit",
            emhass_conf,
            logger,
        )
        opt.perform_optimization(df, p_pv, p_load, ulc, upp)

    def test_perform_optimization_with_in_code_defaults_runs(self):
        """Optimization built from the default-config pipeline must not raise."""
        self._build_and_run(self.retrieve_hass_conf, self.optim_conf, self.plant_conf)

    async def test_perform_optimization_with_config_defaults_json_runs(self):
        """Optimization after treat_runtimeparams with config_defaults must not raise.

        Exercises the treat_runtimeparams -> compile_heat_topology guard path.
        Pre-#878 with heat_topology as the string "null" + old truthy guard raised
        AttributeError before reaching perform_optimization.
        """
        runtimeparams_json = orjson.dumps({}).decode("utf-8")
        _, retrieve_hass_conf, optim_conf, plant_conf = await utils.treat_runtimeparams(
            runtimeparams_json,
            self.params_json,
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            "dayahead-optim",
            logger,
            emhass_conf,
        )
        self._build_and_run(retrieve_hass_conf, optim_conf, plant_conf)
