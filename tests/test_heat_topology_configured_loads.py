"""Regression tests for #1127: deferrable loads and shared tanks configured in
config.json alongside, or instead of, a heat_topology."""

import asyncio
import json
import logging
import pathlib

import orjson

from emhass import utils

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
logger, _ = utils.get_logger(__name__, {}, save_to_file=False)

TOPOLOGY = {
    "sources": [
        {
            "id": "hp",
            "type": "heatpump",
            "nominal_power": 3000,
            "supply_temperature": 45,
            "treat_as_semi_cont": False,
        }
    ],
    "storage": [
        {
            "id": "dhw",
            "volume": 0.2,
            "start_temperature": 50.0,
            "min_temperature": [45.0],
            "max_temperature": [60.0] * 48,
        }
    ],
    "flows": [{"from": "hp", "to": "dhw"}],
}


def _optim_conf_from(config_path: pathlib.Path) -> dict:
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


def _write(tmp_path, config: dict) -> pathlib.Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _replace_warnings(caplog):
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "extend_deferrable_loads" in r.getMessage()
    ]


def test_shared_thermal_tanks_from_config_file_reach_optim_conf(tmp_path):
    """A flat shared_thermal_tanks block in config.json is kept, like
    def_load_config since #1203 (it used to be dropped without a word)."""
    tanks = [
        {
            "id": "dhw",
            "load_ids": [0],
            "volume": 0.2,
            "start_temperature": 50.0,
            "min_temperatures": [45.0] * 48,
            "max_temperatures": [60.0] * 48,
        }
    ]
    optim_conf = _optim_conf_from(
        _write(
            tmp_path,
            {
                "number_of_deferrable_loads": 1,
                "def_load_config": [{"thermal_source": {"supply_temperature": 45}}],
                "shared_thermal_tanks": tanks,
            },
        )
    )
    assert optim_conf["shared_thermal_tanks"] == tanks


def test_replaced_configured_loads_are_named_in_a_warning(tmp_path, caplog):
    """Without extend_deferrable_loads the topology replaces the configured
    loads. Loads the user set up beyond the topology's flows are then not
    optimized, so that must be said, with the way to keep them."""
    caplog.set_level(logging.WARNING)
    optim_conf = _optim_conf_from(
        _write(
            tmp_path,
            {
                "number_of_deferrable_loads": 3,
                "nominal_power_of_deferrable_loads": [1550, 520, 2250],
                "operating_hours_of_each_deferrable_load": [0, 2, 1],
                "heat_topology": TOPOLOGY,
            },
        )
    )
    assert optim_conf["number_of_deferrable_loads"] == 1
    warnings = _replace_warnings(caplog)
    assert len(warnings) == 1, warnings
    assert "1..2" in warnings[0]


def test_the_shipped_example_loads_are_replaced_without_a_warning(tmp_path, caplog):
    """The defaults ship two example loads; a user who never touched them gets
    no warning when the topology replaces them."""
    caplog.set_level(logging.WARNING)
    optim_conf = _optim_conf_from(_write(tmp_path, {"heat_topology": TOPOLOGY}))
    assert optim_conf["number_of_deferrable_loads"] == 1
    assert _replace_warnings(caplog) == []


def test_a_customised_example_load_is_named_in_the_warning(tmp_path, caplog):
    """The example loads count as untouched only if every setting the topology
    overwrites is still the shipped one: a changed minimum power is a
    configured load."""
    caplog.set_level(logging.WARNING)
    _optim_conf_from(
        _write(
            tmp_path,
            {"minimum_power_of_deferrable_loads": [0.0, 300.0], "heat_topology": TOPOLOGY},
        )
    )
    warnings = _replace_warnings(caplog)
    assert len(warnings) == 1, warnings
    assert "1..1" in warnings[0]


def test_invalid_shared_thermal_tanks_from_config_file_are_ignored(tmp_path, caplog):
    """Tanks from config.json get the same check as runtime ones: a tank whose
    load_ids point past the configured loads is reported and ignored instead
    of failing inside the optimizer."""
    caplog.set_level(logging.WARNING)
    optim_conf = _optim_conf_from(
        _write(
            tmp_path,
            {
                "number_of_deferrable_loads": 1,
                "shared_thermal_tanks": [{"id": "dhw", "load_ids": [0, 5], "volume": 0.2}],
            },
        )
    )
    assert not optim_conf.get("shared_thermal_tanks")
    assert any(
        "shared_thermal_tanks in the configuration" in r.getMessage() for r in caplog.records
    )


def test_extend_mode_keeps_the_loads_without_a_warning(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    optim_conf = _optim_conf_from(
        _write(
            tmp_path,
            {
                "number_of_deferrable_loads": 3,
                "nominal_power_of_deferrable_loads": [1550, 520, 2250],
                "operating_hours_of_each_deferrable_load": [0, 2, 1],
                "heat_topology": {**TOPOLOGY, "extend_deferrable_loads": True},
            },
        )
    )
    assert optim_conf["number_of_deferrable_loads"] == 4
    assert _replace_warnings(caplog) == []
