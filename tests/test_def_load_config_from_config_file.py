"""Regression test for #1130: a def_load_config block in config.json must reach optim_conf."""

import asyncio
import json
import pathlib

import orjson

from emhass import utils

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
logger, _ = utils.get_logger(__name__, {}, save_to_file=False)

THERMAL_BATTERY = {
    "thermal_battery": {
        "supply_temperature": 35.0,
        "volume": 14.91,
        "start_temperature": 22.0,
        "min_temperatures": [20.0],
        "max_temperatures": [28.0],
        "carnot_efficiency": 0.45,
        "u_value": 0.326,
        "envelope_area": 649.4,
        "ventilation_rate": 0.12,
        "heated_volume": 573.5,
    }
}
DEF_LOAD_CONFIG = [THERMAL_BATTERY, {}, {}]


def _build_optim_conf_from_config_file(config_path: pathlib.Path) -> dict:
    emhass_conf = {
        "data_path": root / "data/",
        "root_path": root / "src/emhass/",
        "defaults_path": root / "src/emhass/data/config_defaults.json",
        "associations_path": root / "src/emhass/data/associations.csv",
        "config_path": config_path,
    }

    async def _run() -> dict:
        config = await utils.build_config(
            emhass_conf, logger, emhass_conf["defaults_path"], config_path
        )
        _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
        params = await utils.build_params(emhass_conf, secrets, config, logger)
        params_json = orjson.dumps(params, default=str).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        _, _, optim_conf, _ = await utils.treat_runtimeparams(
            None,
            params,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "naive-mpc-optim",
            logger,
            emhass_conf,
        )
        return optim_conf

    return asyncio.run(_run())


def test_def_load_config_from_config_file_reaches_optim_conf(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "number_of_deferrable_loads": 3,
                "operating_hours_of_each_deferrable_load": [0, 0, 6],
                "def_load_config": DEF_LOAD_CONFIG,
            }
        ),
        encoding="utf-8",
    )

    optim_conf = _build_optim_conf_from_config_file(config_path)

    assert optim_conf["def_load_config"] == DEF_LOAD_CONFIG
