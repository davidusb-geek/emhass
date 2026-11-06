"""number_of_batteries is a whole number from 1 to MAX_BATTERIES, refused with a 400.

Every per-battery setting is expanded to one entry per battery, so an unbounded count
let one request allocate memory in proportion to the number it sent (about 110 GB for
10**9). A bad count also escaped /set-config as a 500, and /action as a 500 or a
MemoryError. The web tests go through the real app on scratch paths.
"""

import asyncio
import logging
import pathlib
from unittest.mock import patch

import orjson
import pytest

from emhass import utils

TEST_ROOT = pathlib.Path(__file__).resolve().parents[1]
logger = logging.getLogger("battery_count_test")
MAX = getattr(utils, "MAX_BATTERIES", 100)


@pytest.mark.parametrize("count", [1, 2, MAX, float(MAX), str(MAX)])
def test_counts_up_to_the_limit_are_accepted(count):
    assert utils.validate_num_batteries({"number_of_batteries": count}) == int(float(count))


@pytest.mark.parametrize("count", [MAX + 1, 10**9, 1e300, 10**400, "1e300", "1e309"])
def test_counts_above_the_limit_are_refused(count):
    with pytest.raises(ValueError, match=f"number_of_batteries must be at most {MAX}"):
        utils.validate_num_batteries({"number_of_batteries": count})


@pytest.mark.parametrize("count", [0, -1, 1.5, "abc", True, None])
def test_invalid_counts_keep_their_message(count):
    with pytest.raises(ValueError, match="number_of_batteries must be a positive integer"):
        utils.validate_num_batteries({"number_of_batteries": count})


def test_a_count_too_long_to_print_is_still_the_battery_error():
    # repr() of an int past sys.get_int_max_str_digits() raises a plain ValueError.
    with pytest.raises(
        ValueError, match=f"number_of_batteries must be at most {MAX}, got an integer"
    ):
        utils.validate_num_batteries({"number_of_batteries": 10**5000})


def test_the_cli_path_without_runtime_params_checks_the_count_too():
    # The CLI merges --params into plant_conf; with no --runtimeparams the runtime
    # block that checks the count is skipped.
    conf = {
        "data_path": TEST_ROOT / "data/",
        "root_path": TEST_ROOT / "src/emhass/",
        "defaults_path": TEST_ROOT / "src/emhass/data/config_defaults.json",
        "associations_path": TEST_ROOT / "src/emhass/data/associations.csv",
    }
    defaults = orjson.loads((TEST_ROOT / "src/emhass/data/config_defaults.json").read_bytes())
    params = asyncio.run(utils.build_params(conf, {}, defaults, logger))
    params["plant_conf"]["number_of_batteries"] = MAX + 20
    params_json = orjson.dumps(params, default=str).decode()
    rh_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
    with pytest.raises(ValueError, match=f"number_of_batteries must be at most {MAX}"):
        asyncio.run(
            utils.treat_runtimeparams(
                None, params_json, rh_conf, optim_conf, plant_conf, "dayahead-optim", logger, conf
            )
        )


def test_a_huge_count_is_refused_before_any_per_battery_list_is_built():
    defaults = orjson.loads((TEST_ROOT / "src/emhass/data/config_defaults.json").read_bytes())
    conf = {
        "data_path": TEST_ROOT / "data/",
        "root_path": TEST_ROOT / "src/emhass/",
        "defaults_path": TEST_ROOT / "src/emhass/data/config_defaults.json",
        "associations_path": TEST_ROOT / "src/emhass/data/associations.csv",
    }
    # check_batt_params is what expands each setting to one entry per battery
    # (10**6 batteries used to take about 115 MiB here).
    with (
        patch.object(utils, "check_batt_params", wraps=utils.check_batt_params) as expand,
        pytest.raises(ValueError),
    ):
        asyncio.run(
            utils.build_params(
                conf, {}, dict(defaults, number_of_batteries=10**6, set_use_battery=True), logger
            )
        )
    expand.assert_not_called()


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

    def files():
        return conf["config_path"].read_bytes(), (tmp_path / "params.pkl").read_bytes()

    assert post("/set-config", '{"battery_minimum_state_of_charge": 0.3}')[0] == 200
    return type("Web", (), {"post": staticmethod(post), "files": staticmethod(files)})


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ('"abc"', "must be a positive integer"),
        ('"null"', "must be a positive integer"),
        ("0", "must be a positive integer"),
        ("1e300", f"must be at most {MAX}"),
        ("1000000000000", f"must be at most {MAX}"),
        (str(MAX + 1), f"must be at most {MAX}"),
    ],
)
def test_set_config_refuses_a_bad_count_and_writes_nothing(web, value, message):
    before = web.files()
    status, text = web.post("/set-config", f'{{"number_of_batteries": {value}}}')
    assert status == 400
    assert f"number_of_batteries {message}" in text
    assert text.count("number_of_batteries") == 1
    assert web.files() == before


@pytest.mark.parametrize("value", [str(MAX), "null"])
def test_set_config_still_saves_the_limit_and_a_null_count(web, value):
    status, _ = web.post("/set-config", f'{{"number_of_batteries": {value}}}')
    assert status == 200


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.mark.parametrize("value", [str(MAX + 1), "1000000000000", '"abc"'])
def test_action_refuses_a_bad_runtime_count_before_anything_runs(web, value):
    from emhass import web_server

    # Listen on the app logger itself: other tests turn its level up and its
    # propagation off.
    records = _Records()
    app_logger = web_server.app.logger
    level = app_logger.level
    app_logger.setLevel(logging.ERROR)
    app_logger.addHandler(records)
    try:
        with patch.object(utils, "check_batt_params", wraps=utils.check_batt_params) as expand:
            status, _ = web.post("/action/dayahead-optim", f'{{"number_of_batteries": {value}}}')
    finally:
        app_logger.removeHandler(records)
        app_logger.setLevel(level)
    assert status == 400
    assert any("number_of_batteries must be" in m for m in records.messages)
    expand.assert_not_called()
