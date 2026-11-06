"""Request bodies that the standard json module accepts but orjson cannot write back.

Quart parses a request body with the standard json module, which takes an integer of any
size and any nesting depth. EMHASS then writes the /action runtime parameters and the
saved config.json with orjson, which refuses an integer outside the 64-bit range and
nesting deeper than 254 levels. Both used to escape as a 500 (and /set-config left
config.json empty). These tests go through the real app on scratch paths.
"""

import asyncio
import logging
import pathlib
from unittest.mock import AsyncMock, patch

import orjson
import pytest

from emhass import utils

TEST_ROOT = pathlib.Path(__file__).resolve().parents[1]
logger = logging.getLogger("big_numbers_test")

TOO_BIG = 2**64  # the smallest positive integer orjson refuses
LARGEST_OK = 2**64 - 1  # the largest it writes
TOO_DEEP = "[" * 300 + "]" * 300  # orjson stops at 254 levels


@pytest.fixture
def web(tmp_path, monkeypatch):
    """The real web_server app on scratch paths, seeded with one good save."""
    from emhass import web_server

    conf = {
        "data_path": tmp_path,
        "config_path": tmp_path / "config.json",
        "defaults_path": TEST_ROOT / "src/emhass/data/config_defaults.json",
        "associations_path": TEST_ROOT / "src/emhass/data/associations.csv",
        "legacy_config_path": tmp_path / "legacy_missing.yaml",
        "root_path": TEST_ROOT / "src/emhass/",
    }
    _, secrets = asyncio.run(utils.build_secrets(conf, logger, no_response=True))
    monkeypatch.setattr(web_server, "emhass_conf", conf)
    monkeypatch.setattr(web_server, "params_secrets", secrets)
    client = web_server.app.test_client()

    def post(path, body: str):
        async def go():
            r = await client.post(
                path, data=body.encode(), headers={"Content-Type": "application/json"}
            )
            return r.status_code, await r.get_data(as_text=True)

        return asyncio.run(go())

    assert post("/set-config", '{"battery_minimum_state_of_charge": 0.3}')[0] == 200
    return type("Web", (), {"conf": conf, "post": staticmethod(post)})


@pytest.mark.parametrize(
    "body",
    [
        f'{{"soc_init": {TOO_BIG}}}',
        f'{{"not_a_parameter": {TOO_BIG}}}',
        f'{{"pv_power_forecast": [1, 2, {-(2**63) - 1}]}}',
        f'{{"load_power_forecast": {TOO_DEEP}}}',
    ],
)
def test_action_body_orjson_cannot_write_is_a_bad_request(web, body):
    with patch("emhass.web_server.set_input_data_dict", new_callable=AsyncMock) as set_input:
        status, _ = web.post("/action/dayahead-optim", body)
    assert status == 400
    set_input.assert_not_called()


def test_action_body_with_the_largest_64_bit_integer_still_reaches_the_action(web):
    with patch("emhass.web_server.set_input_data_dict", new_callable=AsyncMock) as set_input:
        set_input.return_value = None
        web.post("/action/dayahead-optim", f'{{"soc_init": {LARGEST_OK}}}')
    set_input.assert_called_once()
    runtimeparams = set_input.call_args[0][3]
    assert orjson.loads(runtimeparams) == {"soc_init": LARGEST_OK}


@pytest.mark.parametrize(
    "body",
    [
        f'{{"battery_nominal_energy_capacity": {TOO_BIG}}}',
        f'{{"not_a_parameter": {TOO_BIG}}}',
        f'{{"not_a_parameter": {TOO_DEEP}}}',
        # Deep enough that copying the body for the inverter checks hits the recursion limit.
        '{"not_a_parameter": ' + "[" * 600 + "]" * 600 + "}",
    ],
)
def test_set_config_body_orjson_cannot_write_is_refused_and_writes_nothing(web, body):
    config_before = web.conf["config_path"].read_bytes()
    params_before = (web.conf["data_path"] / "params.pkl").read_bytes()
    status, text = web.post("/set-config", body)
    assert status == 400
    assert "64 bits" in text
    assert web.conf["config_path"].read_bytes() == config_before
    assert (web.conf["data_path"] / "params.pkl").read_bytes() == params_before


def test_set_config_with_the_largest_64_bit_integer_still_saves(web):
    status, _ = web.post("/set-config", f'{{"battery_nominal_energy_capacity": {LARGEST_OK}}}')
    assert status == 200
    saved = orjson.loads(web.conf["config_path"].read_bytes())
    assert saved["battery_nominal_energy_capacity"] == LARGEST_OK


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_action_logs_why_the_body_was_refused(web):
    from emhass import web_server

    # Listen on the app logger itself: other tests turn its level up and its
    # propagation off.
    records = _Records()
    app_logger = web_server.app.logger
    level = app_logger.level
    app_logger.setLevel(logging.ERROR)
    app_logger.addHandler(records)
    try:
        with patch("emhass.web_server.set_input_data_dict", new_callable=AsyncMock):
            status, _ = web.post("/action/dayahead-optim", f'{{"soc_init": {TOO_BIG}}}')
    finally:
        app_logger.removeHandler(records)
        app_logger.setLevel(level)
    assert status == 400
    assert any("64 bits" in m for m in records.messages)


def test_set_config_body_nested_past_the_recursion_limit_is_refused(web):
    # The json module raises RecursionError here, which get_json lets through.
    config_before = web.conf["config_path"].read_bytes()
    deep = "[" * 5000 + "]" * 5000
    status, text = web.post("/set-config", f'{{"not_a_parameter": {deep}}}')
    assert status == 400
    assert "nested" in text
    assert web.conf["config_path"].read_bytes() == config_before


def test_dry_run_body_nested_deeper_than_orjson_writes_is_refused(web):
    # orjson reads up to 1024 levels deep but writes only 254.
    with patch("emhass.web_server.set_input_data_dict", new_callable=AsyncMock) as set_input:
        status, text = web.post("/api/v1/dry-run/dayahead-optim", f'{{"x": {TOO_DEEP}}}')
    assert status == 400
    assert orjson.loads(text)["status"] == "not-run"
    set_input.assert_not_called()


def test_set_config_body_that_is_not_utf8_is_refused(web):
    config_before = web.conf["config_path"].read_bytes()

    async def go():
        from emhass import web_server

        r = await web_server.app.test_client().post(
            "/set-config",
            data=b'{"battery_minimum_state_of_charge": "\xff\xfe"}',
            headers={"Content-Type": "application/json"},
        )
        return r.status_code

    assert asyncio.run(go()) == 400
    assert web.conf["config_path"].read_bytes() == config_before
