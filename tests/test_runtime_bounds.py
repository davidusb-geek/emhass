"""The settings that size one run are held to bounds (utils.PARAM_BOUNDS).

Without them one small request could hold or kill the single worker: a long
prediction_horizon or forecast at a short step, a large number_of_deferrable_loads or
historic_days_to_retrieve, a curtailment_margin that spins a loop, many
deferrable_load_groups for the coordinator, a long optimization_backend copied into
every row of the plan, and a thermal calibration over centuries of history. Each
refusal is a fixed text that names the key and the range, never the value. Web tests
go through the real app on scratch paths and reach no network.
"""

import asyncio
import itertools
import json
import logging
import os
import pathlib
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np
import orjson
import pytest

from emhass import command_line, utils
from emhass.pv_bias_calibration import _widen_mask

TEST_ROOT = pathlib.Path(__file__).resolve().parents[1]
logger = logging.getLogger("runtime_bounds_test")
CONF = {
    "data_path": TEST_ROOT / "data/",
    "root_path": TEST_ROOT / "src/emhass/",
    "defaults_path": TEST_ROOT / "src/emhass/data/config_defaults.json",
    "associations_path": TEST_ROOT / "src/emhass/data/associations.csv",
}
# A value no fault text may quote
ODD = 987654321
# The most solver threads a setting may ask for on this machine
THREADS = min(64, os.cpu_count() or 1)
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


def _bounds():
    return getattr(utils, "PARAM_BOUNDS", {})


def _faults(values):
    check = getattr(utils, "bounded_value_faults", None)
    assert check is not None, "utils.bounded_value_faults is missing"
    return check(values)


def _params_json(**config) -> str:
    defaults = orjson.loads(CONF["defaults_path"].read_bytes())
    defaults.update(config)
    params = asyncio.run(utils.build_params(CONF, {}, defaults, logger))
    params["passed_data"] = {}
    return orjson.dumps(params, default=str).decode()


def _params_json_with_secrets(**config) -> str:
    """_params_json with the secrets a run reads (the time zone among them)."""
    defaults = orjson.loads(CONF["defaults_path"].read_bytes())
    defaults.update(config)
    _, secrets = asyncio.run(utils.build_secrets(CONF, logger, no_response=True))
    params = asyncio.run(utils.build_params(CONF, secrets, defaults, logger))
    params["passed_data"] = {}
    return orjson.dumps(params, default=str).decode()


def _treat(runtime, set_type="dayahead-optim", params_json=None):
    params_json = params_json or _params_json()
    rh, oc, pc = utils.get_yaml_parse(params_json, logger)
    return asyncio.run(
        utils.treat_runtimeparams(
            orjson.dumps(runtime).decode(), params_json, rh, oc, pc, set_type, logger, CONF
        )
    )


@pytest.fixture(autouse=True)
def _local_network_only(monkeypatch):
    """Nothing here may reach past this machine, even on a tree without the bounds,
    where a request would go on to fetch history and forecasts."""
    local = ("127.0.0.1", "localhost", "::1")
    getaddrinfo = socket.getaddrinfo
    connect = socket.socket.connect

    def local_getaddrinfo(host, *args, **kwargs):
        if host not in local:
            raise OSError("network access is disabled in this test")
        return getaddrinfo(host, *args, **kwargs)

    def local_connect(self, address):
        if isinstance(address, tuple) and address[0] not in local:
            raise OSError("network access is disabled in this test")
        return connect(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", local_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", local_connect)


# The table itself


def test_the_table_names_the_bounded_settings():
    expected = {
        "prediction_horizon": ("int", 2, 5760),
        "optimization_time_step": ("int", 1, 1440),
        "freq": ("int", 1, 1440),
        "delta_forecast_daily": ("int", 1, 5760),
        "delta_forecast": ("int", 1, 5760),
        "number_of_deferrable_loads": ("int", 0, 100),
        "num_def_loads": ("int", 0, 100),
        "def_load_config": ("list", 0, 100),
        "historic_days_to_retrieve": ("int", 0, 730),
        "days_to_retrieve": ("int", 0, 730),
        "calibration_days_to_retrieve": ("int", 0, 730),
        "curtailment_margin": ("int", 0, 48),
        "lookback_hours": ("int", 0, 17520),
        "lp_solver_timeout": ("positive", 0, 300),
        "num_threads": ("int", 0, THREADS),
        "lp_solver_mip_rel_gap": ("number", 0, 1),
        "def_minimum_on_time": ("steps", 0, 5760),
        "def_minimum_off_time": ("steps", 0, 5760),
        "n_trials": ("int", 1, 500),
        "num_lags": ("int", 1, 10080),
    }
    assert _bounds() == expected


def _int_keys():
    return [k for k, (kind, *_) in _bounds().items() if kind == "int"] or ["prediction_horizon"]


@pytest.mark.parametrize("key", _int_keys())
def test_whole_numbers_within_the_bounds_pass(key):
    _, low, high = _bounds()[key]
    for value in (high, float(high), str(high), low, None):
        assert _faults({key: value}) == [], (key, value)


@pytest.mark.parametrize("key", _int_keys())
def test_values_outside_the_bounds_are_refused_by_key_and_range(key):
    _, low, high = _bounds()[key]
    text = f"{key} must be a whole number from {low} to {high}"
    bad = [high + 1, ODD, 10**400, "1e309", float("nan"), 1.5, True, "abc", "", [1], {}]
    bad += [low - 1, -1e300, -(10**18)]
    for value in bad:
        assert _faults({key: value}) == [text], (key, value)
    assert str(ODD) not in text


def test_def_load_config_is_a_list_of_at_most_100_entries():
    assert _faults({"def_load_config": [{}] * 100}) == []
    for value in ([{}] * 101, {"a": 1}, "[]"):
        assert _faults({"def_load_config": value}) == [
            "def_load_config must be a list of at most 100 entries"
        ]


def test_optimization_backend_is_one_of_the_known_ones():
    for value in ("cvxpy", "dantzig_wolfe", None):
        assert _faults({"optimization_backend": value}) == []
    for value in ("B" * 5000, "admm", "", 5, {"x": ["y"]}, ["cvxpy"]):
        assert _faults({"optimization_backend": value}) == [
            "optimization_backend must be one of cvxpy, dantzig_wolfe"
        ]


def test_load_groups_are_bounded_in_number_and_names():
    group = {"names": ["deferrable0"], "max_power": 1000}
    text = "deferrable_load_groups must have at most 256 groups of at most 200 names each"
    assert _faults({"deferrable_load_groups": [group] * 256}) == []
    assert _faults({"deferrable_load_groups": [group] * 257}) == [text]
    assert _faults({"deferrable_load_groups": [{"names": ["deferrable0"] * 201}]}) == [text]
    # Other shapes are check_deferrable_load_groups' to judge
    assert _faults({"deferrable_load_groups": "x"}) == []


def test_every_fault_is_reported_once():
    faults = _faults({"prediction_horizon": 0, "number_of_deferrable_loads": -1, "x": 10**9})
    assert faults == [
        "prediction_horizon must be a whole number from 2 to 5760",
        "number_of_deferrable_loads must be a whole number from 0 to 100",
    ]


def test_published_metadata_matches_the_bounds():
    static = TEST_ROOT / "src" / "emhass" / "static"
    definitions = json.loads((static / "data" / "param_definitions.json").read_text("utf-8"))
    runtime = json.loads((static / "data" / "runtime_params.json").read_text("utf-8"))
    spec = json.loads((static / "openapi.json").read_text("utf-8"))
    props = spec["components"]["schemas"]["Config"]["properties"]
    flat = {
        k: v
        for section in definitions.values()
        if isinstance(section, dict)
        for k, v in section.items()
    }
    for key in (
        "optimization_time_step",
        "delta_forecast_daily",
        "number_of_deferrable_loads",
        "historic_days_to_retrieve",
    ):
        _, low, high = _bounds()[key]
        assert (flat[key].get("minimum"), flat[key]["maximum"]) == (low, high), key
        assert (props[key].get("minimum"), props[key]["maximum"]) == (low, high), key
    horizon = runtime["MPC"]["prediction_horizon"]
    assert (horizon["minimum"], horizon["maximum"]) == (2, 5760)


# A request (treat_runtimeparams)


@pytest.mark.parametrize(
    ("runtime", "set_type", "text"),
    [
        (
            {"prediction_horizon": 5761},
            "naive-mpc-optim",
            "prediction_horizon must be a whole number from 2 to 5760",
        ),
        (
            {"number_of_deferrable_loads": 101},
            "dayahead-optim",
            "number_of_deferrable_loads must be a whole number from 0 to 100",
        ),
        (
            {"num_def_loads": 4321},
            "dayahead-optim",
            "num_def_loads must be a whole number from 0 to 100",
        ),
        (
            {"historic_days_to_retrieve": 731},
            "forecast-model-fit",
            "historic_days_to_retrieve must be a whole number from 0 to 730",
        ),
        (
            {"optimization_time_step": 0},
            "dayahead-optim",
            "optimization_time_step must be a whole number from 1 to 1440",
        ),
        (
            {"delta_forecast_daily": 5761},
            "dayahead-optim",
            "delta_forecast_daily must be a whole number from 1 to 5760",
        ),
        (
            {"def_load_config": [{}] * 150},
            "dayahead-optim",
            "def_load_config must be a list of at most 100 entries",
        ),
        (
            {"optimization_backend": "B" * 5000},
            "dayahead-optim",
            "optimization_backend must be one of cvxpy, dantzig_wolfe",
        ),
        (
            {
                "deferrable_load_groups": [{"names": ["deferrable0"], "max_power": 1}] * 17,
                "optimization_backend": "dantzig_wolfe",
            },
            "dayahead-optim",
            "deferrable_load_groups must have at most 16 groups of at most 200 names each "
            "with optimization_backend dantzig_wolfe",
        ),
        (
            {"deferrable_load_groups": [{"names": ["deferrable0"], "max_power": 1}] * 257},
            "dayahead-optim",
            "deferrable_load_groups must have at most 256 groups of at most 200 names each",
        ),
    ],
)
def test_a_request_over_a_bound_is_refused_before_anything_is_built(runtime, set_type, text):
    params_json = _params_json()
    with (
        patch.object(utils, "get_forecast_dates", wraps=utils.get_forecast_dates) as grid,
        patch.object(utils, "check_def_loads", wraps=utils.check_def_loads) as per_load,
        pytest.raises(utils.RuntimeParamError) as caught,
    ):
        _treat(runtime, set_type, params_json)
    assert str(caught.value) == text
    assert "BBBB" not in str(caught.value) and str(ODD) not in str(caught.value)
    grid.assert_not_called()
    per_load.assert_not_called()


def test_requests_at_the_bounds_still_run():
    params, *_ = _treat({"historic_days_to_retrieve": 730}, "forecast-model-fit")
    assert orjson.loads(params)["passed_data"]["historic_days_to_retrieve"] == 730
    params, *_ = _treat({"prediction_horizon": 96}, "naive-mpc-optim")
    assert orjson.loads(params)["passed_data"]["prediction_horizon"] == 96
    _, _, optim_conf, _ = _treat({"number_of_deferrable_loads": 4})
    assert optim_conf["number_of_deferrable_loads"] == 4


def test_the_grid_a_request_asks_for_is_bounded_too():
    # Each value is within its own bound; together they are 7200 steps
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({"optimization_time_step": 1, "delta_forecast_daily": 5})
    assert str(caught.value) == (
        "delta_forecast_daily and optimization_time_step give more than 5760 time steps"
    )
    # 5760 steps exactly, and a short step with the saved one-day forecast
    _treat({"optimization_time_step": 1, "delta_forecast_daily": 4})
    _treat({"freq": 1})


def test_loads_a_request_adds_through_heat_topology_are_counted():
    runtime = {
        "number_of_deferrable_loads": 100,
        "heat_topology": {**TOPOLOGY, "extend_deferrable_loads": True},
    }
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat(runtime)
    assert str(caught.value) == "The request gives more than 100 deferrable loads in all"
    _, _, optim_conf, _ = _treat({**runtime, "number_of_deferrable_loads": 99})
    assert optim_conf["number_of_deferrable_loads"] == 100


def test_a_saved_value_over_a_bound_is_reported_not_refused(caplog):
    params_json = _params_json(number_of_deferrable_loads=150, historic_days_to_retrieve=1000)
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _, rh_conf, optim_conf, _ = _treat({}, "dayahead-optim", params_json)
    assert optim_conf["number_of_deferrable_loads"] == 150
    warned = [r.getMessage() for r in caplog.records if "saved configuration" in r.getMessage()]
    assert warned == [
        "number_of_deferrable_loads in the saved configuration is above 100, the most a "
        "request or a new save may set",
        "historic_days_to_retrieve in the saved configuration is above 730, the most a "
        "request or a new save may set",
        "The saved configuration gives more than 100 deferrable loads in all, the most a "
        "request or a new save may set",
    ]
    # A request that sets the key itself is held to the bound, and is not warned about
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _treat({"number_of_deferrable_loads": 3}, "dayahead-optim", params_json)
    assert not any(
        "number_of_deferrable_loads in the saved" in r.getMessage() for r in caplog.records
    )


def test_runtime_parameters_must_be_an_object():
    params_json = _params_json()
    rh, oc, pc = utils.get_yaml_parse(params_json, logger)
    for body in ("[1]", "5", '"x"'):
        with pytest.raises(utils.RuntimeParamError, match="must be a JSON object"):
            asyncio.run(
                utils.treat_runtimeparams(
                    body, params_json, rh, oc, pc, "dayahead-optim", logger, CONF
                )
            )


def test_get_days_list_never_spans_more_than_730_days():
    assert len(utils.get_days_list(10**6)) == 731
    assert len(utils.get_days_list(2)) == 3


# deferrable_load_groups and the coordinator


def _distinct_groups(count):
    names = [f"deferrable{i}" for i in range(6)]
    subsets = [list(c) for r in range(2, 7) for c in itertools.combinations(names, r)]
    return [{"names": s, "max_power": 9000 + k} for k, s in enumerate(subsets[:count])]


@pytest.mark.parametrize("config_file", [True, False])
def test_saved_groups_over_the_cap_are_reported_not_refused(config_file, caplog):
    optim = {
        "number_of_deferrable_loads": 6,
        "set_use_battery": False,
        "optimization_backend": "dantzig_wolfe",
    }
    error_cls = ValueError if config_file else utils.RuntimeParamError
    with caplog.at_level(logging.WARNING, logger=logger.name):
        for count in (16, 17):
            utils.check_deferrable_load_groups(
                {**optim, "deferrable_load_groups": _distinct_groups(count)},
                {},
                config_file=config_file,
                error_cls=error_cls,
                logger=logger,
            )
    assert [r.getMessage() for r in caplog.records] == [
        "In the saved configuration, deferrable_load_groups must have at most 16 groups "
        "of at most 200 names each with optimization_backend dantzig_wolfe; a request or "
        "a new save is held to that"
    ]


# pv-bias-calibration


def _pv_body(**extra):
    return {
        "p10": [1.0] * 10,
        "p50": [2.0] * 10,
        "actual": [1.5] * 10,
        "curtailed": [0] * 9 + [1],
        **extra,
    }


def test_pv_bias_payload_is_bounded():
    for extra, text in (
        ({"curtailment_margin": 49}, "curtailment_margin must be a whole number from 0 to 48"),
        ({"curtailment_margin": -1}, "curtailment_margin must be a whole number from 0 to 48"),
        ({"p10": [1.0] * 105121}, "p10 must have at most 105120 entries"),
        ({"curtailed": [0] * 105121}, "curtailed must have at most 105120 entries"),
    ):
        started = time.perf_counter()
        with pytest.raises(utils.RuntimeParamError) as caught:
            asyncio.run(command_line.pv_bias_calibration(_pv_body(**extra), logger))
        assert str(caught.value) == text
        assert time.perf_counter() - started < 5
    result = asyncio.run(command_line.pv_bias_calibration(_pv_body(curtailment_margin=1), logger))
    assert isinstance(result, dict)
    # 48 is accepted; on 10 rows it then drops every row, which the calibration says
    with pytest.raises(ValueError, match="no usable observations left") as caught:
        asyncio.run(command_line.pv_bias_calibration(_pv_body(curtailment_margin=48), logger))
    assert not isinstance(caught.value, utils.RuntimeParamError)


def test_widen_mask_is_bounded_by_the_history():
    mask = np.zeros(10, dtype=bool)
    mask[4] = True
    # Two million steps took about a second and a half; the loop now stops at nine
    started = time.perf_counter()
    widened = _widen_mask(mask, 2 * 10**6)
    assert time.perf_counter() - started < 0.5
    assert widened.all()
    assert (widened == _widen_mask(mask, 9)).all()
    assert _widen_mask(np.ones(1, dtype=bool), 5).tolist() == [True]


# thermal-model-calibration


@pytest.mark.parametrize(
    ("payload", "text"),
    [
        ({"lookback_hours": 17521}, "lookback_hours must be a whole number from 0 to 17520"),
        (
            {"indoor_temperature": [20.0] * 20161, "outdoor_temperature": [10.0] * 20161},
            "indoor_temperature must have at most 20160 entries",
        ),
        ({"heater_power": [0.0] * 20161}, "heater_power must have at most 20160 entries"),
        ({"date_ranges": [["1900-01-01", "2250-01-01"]]}, None),
        ({"date_ranges": [["2026-01-02", "2026-01-01"]]}, None),
        ({"date_ranges": [["not a date", "2026-01-01"]]}, None),
        ({"date_ranges": [["2026-01-01", "2026-01-02"]] * 17}, None),
        ({"date_ranges": [["2026-01-01"]]}, None),
        ({"date_ranges": "2026-01-01"}, None),
        ({"date_ranges": [["x" * 65, "2026-01-01"]]}, None),
    ],
)
def test_thermal_payload_is_bounded_before_any_history_is_read(payload, text):
    text = text or (
        "date_ranges must be a list of at most 16 [start, end] pairs, each start "
        "before its end and at most 730 days apart"
    )
    params_json = _params_json()
    with (
        patch.object(utils, "_load_thermal_calibration_frame") as load,
        pytest.raises(utils.RuntimeParamError) as caught,
    ):
        asyncio.run(utils.compute_thermal_model_calibration(CONF, params_json, payload, logger))
    assert str(caught.value) == text
    assert "1900" not in text and "not a date" not in text
    load.assert_not_called()


def test_thermal_payload_within_the_bounds_runs():
    params_json = _params_json()
    for payload in (
        {"lookback_hours": 48},
        {"date_ranges": [["2026-01-01", "2026-01-03"]], "lookback_hours": 48},
    ):
        result = asyncio.run(
            utils.compute_thermal_model_calibration(
                CONF, params_json, payload, logger, get_data_from_file=True
            )
        )
        assert result["data_source"] == "offline_demo"


# The web routes


@pytest.fixture
def web(tmp_path, monkeypatch):
    """The real web_server app on scratch paths, seeded with one good save."""
    from emhass import web_server

    conf = {
        "data_path": tmp_path,
        "config_path": tmp_path / "config.json",
        "defaults_path": CONF["defaults_path"],
        "associations_path": CONF["associations_path"],
        "legacy_config_path": tmp_path / "legacy_missing.yaml",
        "root_path": CONF["root_path"],
    }
    _, secrets = asyncio.run(utils.build_secrets(conf, logger, no_response=True))
    monkeypatch.setattr(web_server, "emhass_conf", conf)
    monkeypatch.setattr(web_server, "params_secrets", secrets)
    client = web_server.app.test_client()

    def post(path, body: bytes, content_type="application/json"):
        async def go():
            r = await client.post(path, data=body, headers={"Content-Type": content_type})
            return r.status_code, await r.get_data(as_text=True)

        return asyncio.run(go())

    def files():
        return conf["config_path"].read_bytes(), (tmp_path / "params.pkl").read_bytes()

    assert post("/set-config", b'{"battery_minimum_state_of_charge": 0.3}')[0] == 200
    return type("Web", (), {"post": staticmethod(post), "files": staticmethod(files)})


@pytest.mark.parametrize(
    ("body", "text"),
    [
        ({"number_of_deferrable_loads": 10**6}, "number_of_deferrable_loads must be"),
        ({"historic_days_to_retrieve": ODD}, "historic_days_to_retrieve must be"),
        ({"optimization_time_step": 0}, "optimization_time_step must be"),
        ({"delta_forecast_daily": 10**4}, "delta_forecast_daily must be"),
        ({"optimization_backend": "B" * 5000}, "optimization_backend must be one of"),
        (
            {
                "deferrable_load_groups": [{"names": ["deferrable0"], "max_power": 1}] * 17,
                "optimization_backend": "dantzig_wolfe",
            },
            "deferrable_load_groups must have at most 16 groups",
        ),
        (
            {"deferrable_load_groups": [{"names": ["deferrable0"], "max_power": 1}] * 257},
            "deferrable_load_groups must have at most 256 groups",
        ),
    ],
)
def test_set_config_refuses_a_value_over_a_bound_and_writes_nothing(web, body, text):
    from emhass import web_server

    before = web.files()
    with patch.object(web_server, "build_params", wraps=web_server.build_params) as built:
        status, reply = web.post("/set-config", orjson.dumps(body))
    assert status == 400
    assert text in reply
    assert "BBBB" not in reply and str(ODD) not in reply and "1000000" not in reply
    assert web.files() == before
    # Only the configuration saved now is built (to compare with), never the body
    for call in built.call_args_list:
        assert not any(same in call.args[2].items() for same in body.items())


def test_set_config_still_saves_values_at_the_bounds(web):
    body = {
        "number_of_deferrable_loads": 2,
        "historic_days_to_retrieve": 730,
        "optimization_time_step": 1440,
        "delta_forecast_daily": 5760,
        "optimization_backend": "dantzig_wolfe",
    }
    assert web.post("/set-config", orjson.dumps(body))[0] == 200


@pytest.mark.parametrize(
    ("path", "body", "text"),
    [
        (
            "/action/dayahead-optim",
            {"number_of_deferrable_loads": 101},
            "number_of_deferrable_loads must be a whole number from 0 to 100",
        ),
        (
            "/action/naive-mpc-optim",
            {"prediction_horizon": 5761},
            "prediction_horizon must be a whole number from 2 to 5760",
        ),
        (
            "/action/dayahead-optim",
            {"optimization_backend": "B" * 5000},
            "optimization_backend must be one of cvxpy, dantzig_wolfe",
        ),
        (
            "/action/forecast-model-fit",
            {"historic_days_to_retrieve": 731},
            "historic_days_to_retrieve must be a whole number from 0 to 730",
        ),
        (
            "/action/pv-bias-calibration",
            _pv_body(curtailment_margin=49),
            "curtailment_margin must be a whole number from 0 to 48",
        ),
        (
            "/action/thermal-model-calibration",
            {"lookback_hours": 20000},
            "lookback_hours must be a whole number from 0 to 17520",
        ),
        (
            "/action/thermal-model-calibration",
            {"date_ranges": [["1900-01-01", "2250-01-01"]]},
            "date_ranges must be a list of at most 16 [start, end] pairs",
        ),
    ],
)
def test_actions_refuse_a_request_over_a_bound(web, path, body, text):
    # Nothing past the parameters may be reached: no Home Assistant session is made
    # for an action, and the thermal history is not read
    with (
        patch.object(command_line, "RetrieveHass", side_effect=AssertionError("reached")),
        patch.object(utils, "_load_thermal_from_ha", side_effect=AssertionError("reached")),
    ):
        status, reply = web.post(path, orjson.dumps(body))
    assert status == 400
    assert text in reply
    for raw in ("BBBB", "5761", "731", "49", "20000", "1900", "2250"):
        assert raw not in reply


def test_a_dry_run_refuses_a_request_over_a_bound(web):
    with patch.object(command_line, "RetrieveHass", side_effect=AssertionError("reached")):
        status, reply = web.post("/api/v1/dry-run/naive-mpc-optim", b'{"prediction_horizon": 5761}')
    assert status == 400
    assert json.loads(reply)["status"] == "not-run"


def test_get_json_checks_the_bounds_before_building(web):
    with patch("emhass.web_server.build_params") as build:
        status, reply = web.post(
            "/get-json", b"number_of_deferrable_loads: 1000000\n", "text/plain"
        )
    assert status == 400
    assert "number_of_deferrable_loads must be a whole number from 0 to 100" in reply
    build.assert_not_called()


# The solver, the forecaster and the minimum on/off times


@pytest.mark.parametrize(
    ("key", "good", "bad", "text"),
    [
        (
            "lp_solver_timeout",
            [300, 0.5, "45", 1e-6],
            [0, -1, 301, 1e9, float("nan"), float("inf"), "abc", True, [1]],
            "lp_solver_timeout must be a number above 0 and at most 300",
        ),
        (
            "lp_solver_mip_rel_gap",
            [0, 1, 0.01, "0.05"],
            [-0.1, 1.5, float("nan"), "x", False],
            "lp_solver_mip_rel_gap must be a number from 0 to 1",
        ),
        (
            "num_threads",
            [0, THREADS, "1"],
            [-1, THREADS + 1, 4096, 1.5],
            f"num_threads must be a whole number from 0 to {THREADS}",
        ),
        ("n_trials", [1, 500], [0, 501, 10**6], "n_trials must be a whole number from 1 to 500"),
        (
            "num_lags",
            [1, 10080],
            [0, 10081, 10**6],
            "num_lags must be a whole number from 1 to 10080",
        ),
        (
            "def_minimum_on_time",
            [5760, [0, 5760], [None, 3], "3", ["3", 2.0], [0] * 100, [1.5]],
            [5761, [5761, 0], [-1], [0] * 101, "x", [float("nan")], {"a": 1}, [True]],
            "def_minimum_on_time must be a number of time steps from 0 to 5760, or a list "
            "of at most 100 of them",
        ),
        (
            "def_minimum_off_time",
            [0, [12, 0]],
            [[10**9]],
            "def_minimum_off_time must be a number of time steps from 0 to 5760, or a list "
            "of at most 100 of them",
        ),
    ],
)
def test_solver_forecaster_and_minimum_time_settings_are_bounded(key, good, bad, text):
    for value in good:
        assert _faults({key: value}) == [], (key, value)
    for value in bad:
        assert _faults({key: value}) == [text], (key, value)


def test_a_request_cannot_set_solver_or_forecaster_sizes_over_their_bounds():
    for runtime, set_type, text in (
        (
            {"lp_solver_timeout": 1e9, "num_threads": 4096, "lp_solver_mip_rel_gap": 0},
            "dayahead-optim",
            "lp_solver_timeout must be a number above 0 and at most 300",
        ),
        ({"num_threads": 4096}, "dayahead-optim", "num_threads must be a whole number"),
        ({"num_lags": 10**6}, "forecast-model-fit", "num_lags must be a whole number"),
        ({"n_trials": 10**6}, "forecast-model-tune", "n_trials must be a whole number"),
        ({"def_minimum_on_time": [5761, 0]}, "dayahead-optim", "def_minimum_on_time must be"),
    ):
        with pytest.raises(utils.RuntimeParamError) as caught:
            _treat(runtime, set_type)
        assert str(caught.value).startswith(text), runtime
        assert "1000000" not in str(caught.value) and "4096" not in str(caught.value)


def test_saved_solver_options_over_a_bound_are_reported(caplog):
    params_json = _params_json(lp_solver_timeout=1e9, num_threads=10**5, lp_solver_mip_rel_gap=2)
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _treat({}, "dayahead-optim", params_json)
    warned = [r.getMessage() for r in caplog.records if "saved configuration" in r.getMessage()]
    assert warned == [
        "lp_solver_timeout in the saved configuration is above 300, the most a request or "
        "a new save may set",
        f"num_threads in the saved configuration is above {THREADS}, the "
        "most a request or a new save may set",
        "lp_solver_mip_rel_gap in the saved configuration is above 1, the most a request "
        "or a new save may set",
    ]


class _SolveStopped(BaseException):
    """Raised in place of a solve, past every handler in the optimizer."""


def test_saved_solver_options_reach_highs_within_their_bounds(caplog):
    import cvxpy as cp

    from tests.test_multi_battery_optimization import build_optimization, make_pin_scenario

    getattr(utils, "_SOLVER_OPTIONS_WARNED", set()).clear()
    opt = build_optimization(optim_overrides={"lp_solver_timeout": 1e9, "num_threads": 10**5})
    opt.logger = logger
    _, p_pv, p_load, df_input = make_pin_scenario()
    seen = {}

    def solve(self, *args, **kwargs):
        seen.update(kwargs)
        raise _SolveStopped

    with (
        patch.object(cp.Problem, "solve", solve),
        caplog.at_level(logging.WARNING, logger=logger.name),
        pytest.raises(_SolveStopped),
    ):
        opt.perform_dayahead_forecast_optim(df_input, p_pv, p_load, soc_init=0.4, soc_final=0.5)
    assert seen["time_limit"] == 300.0
    assert seen["threads"] == THREADS
    warned = [r.getMessage() for r in caplog.records if "the solver uses" in r.getMessage()]
    assert warned == [
        "In the configuration, num_threads must be a whole number from 0 to "
        f"{THREADS}; the solver uses {THREADS}",
        "In the configuration, lp_solver_timeout must be a number above 0 and at most 300; "
        "the solver uses 300.0",
    ]


def test_the_coordinator_time_budget_is_bounded_too():
    from emhass.coordinator.master import DWCoordinator
    from tests.test_federated import _coordinated

    run = DWCoordinator.run
    budgets = []

    def hurried(self, **kwargs):
        budgets.append(kwargs.get("max_seconds"))
        return run(self, **{**kwargs, "max_seconds": 0.0})

    with patch.object(DWCoordinator, "run", hurried):
        _coordinated(lp_solver_timeout=1e9)
    assert budgets and set(budgets) == {300.0}


def test_bounded_solver_option_clamps_and_falls_back():
    quiet = logging.getLogger("runtime_bounds_quiet")
    check = utils.bounded_solver_option
    assert check("lp_solver_timeout", 1e9, 45, quiet) == 300.0
    for value in (0, -5, float("nan"), "abc", None):
        assert check("lp_solver_timeout", value, 45, quiet) == 45.0
    assert check("lp_solver_timeout", "30", 45, quiet) == 30.0
    # The thread bound of an 8-core machine, whatever this one has
    with patch.dict(utils.PARAM_BOUNDS, {"num_threads": ("int", 0, 8)}):
        assert check("num_threads", -3, 0, quiet) == 0
        assert check("num_threads", "2", 0, quiet) == 2
        assert check("num_threads", 9, 0, quiet) == 8


def test_the_thread_bound_follows_the_cores():
    assert [utils.max_solver_threads(n) for n in (None, 1, 16, 64, 65, 256)] == [
        1,
        1,
        16,
        64,
        64,
        64,
    ]
    assert utils.MAX_SOLVER_THREADS == utils.max_solver_threads(os.cpu_count())


@pytest.mark.parametrize(
    ("saved", "threads"), [(0, None), (1, 1), (10**5, THREADS), ("auto", None)]
)
def test_num_threads_0_passes_no_thread_count_to_highs(saved, threads):
    import cvxpy as cp

    from tests.test_multi_battery_optimization import build_optimization, make_pin_scenario

    opt = build_optimization(optim_overrides={"num_threads": saved})
    _, p_pv, p_load, df_input = make_pin_scenario()
    seen = {}

    def solve(self, *args, **kwargs):
        seen.update(kwargs)
        raise _SolveStopped

    with patch.object(cp.Problem, "solve", solve), pytest.raises(_SolveStopped):
        opt.perform_dayahead_forecast_optim(df_input, p_pv, p_load, soc_init=0.4, soc_final=0.5)
    assert seen.get("threads") == threads
    assert opt.num_threads == (threads or 0)


# Two solves in one process, as the web server runs them. HiGHS fixes its thread pool
# at the first solve of a process, so this runs in a process of its own.
_TWO_SOLVES = """
from tests.test_multi_battery_optimization import build_optimization, make_pin_scenario

_, p_pv, p_load, df_input = make_pin_scenario()
for threads in (1, 0):
    opt = build_optimization(optim_overrides={"num_threads": threads})
    opt.perform_dayahead_forecast_optim(df_input, p_pv, p_load, soc_init=0.4, soc_final=0.5)
    print("solve", threads, opt.optim_status, flush=True)
"""


def test_a_solve_with_the_default_threads_runs_after_one_with_a_count():
    root = pathlib.Path(__file__).resolve().parents[1]
    src = pathlib.Path(utils.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(src), str(root)])}
    done = subprocess.run(
        [sys.executable, "-c", _TWO_SOLVES],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    solves = [line.split() for line in done.stdout.splitlines() if line.startswith("solve ")]
    assert solves == [["solve", "1", "Optimal"], ["solve", "0", "Optimal"]], done.stdout


# The time grid


def test_a_save_holds_the_time_grid_with_the_defaults_for_a_missing_key():
    defaults = {"optimization_time_step": 30, "delta_forecast_daily": 1}
    text = "delta_forecast_daily and optimization_time_step give more than 5760 time steps"
    faults = utils.config_value_faults
    assert faults({"optimization_time_step": 1}, defaults) == []
    assert faults({"delta_forecast_daily": 200}, defaults) == [text]
    assert faults({"optimization_time_step": 1}, {"delta_forecast_daily": 5}) == [text]
    assert faults({"freq": 1, "delta_forecast": 5}, {}) == [text]
    assert faults({"freq": 1440, "delta_forecast": 5760}, {}) == []


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"optimization_time_step": 1, "delta_forecast_daily": 30}, 400),
        ({"delta_forecast_daily": 200}, 400),
        ({"optimization_time_step": 1, "delta_forecast_daily": 4}, 200),
        ({"optimization_time_step": 1}, 200),
    ],
)
def test_set_config_holds_the_time_grid(web, body, status):
    before = web.files()
    code, reply = web.post("/set-config", orjson.dumps(body))
    assert code == status, reply
    if status == 400:
        assert "give more than 5760 time steps" in reply
        assert web.files() == before


def test_a_grid_from_saved_settings_is_warned_about_then_refused(caplog):
    longer = _params_json(optimization_time_step=1, delta_forecast_daily=5)  # 7200 steps
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _treat({}, "dayahead-optim", longer)
    assert (
        "The saved delta_forecast_daily and optimization_time_step give more than 5760 time "
        "steps, the most a request or a new save may set"
    ) in [r.getMessage() for r in caplog.records]
    # A request that asks for a longer grid is held to the request's bound
    with pytest.raises(utils.RuntimeParamError, match="give more than 5760 time steps"):
        _treat({"optimization_time_step": 1, "delta_forecast_daily": 6}, "dayahead-optim", longer)
    # Past four times the bound, from saved settings alone: refused before the grid
    too_long = _params_json(optimization_time_step=1, delta_forecast_daily=17)  # 24480
    with (
        patch.object(utils, "get_forecast_dates", wraps=utils.get_forecast_dates) as grid,
        pytest.raises(utils.RuntimeParamError) as caught,
    ):
        _treat({}, "dayahead-optim", too_long)
    assert str(caught.value) == (
        "The saved delta_forecast_daily and optimization_time_step give more than 23040 time steps"
    )
    grid.assert_not_called()


def test_a_run_without_runtime_parameters_holds_the_saved_grid_too():
    for config, refused in (
        ({"optimization_time_step": 1, "delta_forecast_daily": 17}, True),
        ({"optimization_time_step": 30, "delta_forecast_daily": 1}, False),
    ):
        params_json = _params_json(**config)
        rh, oc, pc = utils.get_yaml_parse(params_json, logger)
        call = utils.treat_runtimeparams(
            None, params_json, rh, oc, pc, "dayahead-optim", logger, CONF
        )
        if refused:
            with pytest.raises(utils.RuntimeParamError, match="more than 23040 time steps"):
                asyncio.run(call)
        else:
            asyncio.run(call)


def test_a_dst_day_does_not_tip_a_grid_at_the_bound_over():
    # The four days from here cross the Europe/Paris fall-back: 5820 one-minute steps
    now = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    zone = utils.pytz.timezone("Europe/Paris")
    assert len(utils.get_forecast_dates(1, 4, zone, now=now)) == 5820
    runtime = {"optimization_time_step": 1, "delta_forecast_daily": 4, "time_zone": "Europe/Paris"}
    with patch.object(utils, "_get_now", return_value=now):
        _treat(runtime)


# The naive-mpc forecast extension


def _fewest_days(horizon, step, delta, zone, now):
    days = delta
    while len(utils.get_forecast_dates(step, days, zone, now=now)) < horizon:
        days += 1
    return days


@pytest.mark.parametrize(
    ("horizon", "step", "start"),
    [
        (300, 1440, (2026, 3, 1)),
        (97, 30, (2026, 3, 28)),  # over the spring-forward
        (50, 30, (2026, 10, 25)),  # on the fall-back day
        (1500, 1, (2026, 10, 24)),
        (5760, 1, (2026, 3, 27)),
        (7, 1440, (2026, 6, 1)),
    ],
)
def test_the_horizon_extension_finds_the_fewest_days_in_a_few_grids(horizon, step, start):
    now = datetime(*start, 10, 0, tzinfo=UTC)
    zone = utils.pytz.timezone("Europe/Paris")
    expected = _fewest_days(horizon, step, 1, zone, now)
    runtime = {
        "prediction_horizon": horizon,
        "optimization_time_step": step,
        "delta_forecast_daily": 1,
        "time_zone": "Europe/Paris",
    }
    with (
        patch.object(utils, "_get_now", return_value=now),
        patch.object(utils, "get_forecast_dates", wraps=utils.get_forecast_dates) as grid,
    ):
        _, _, optim_conf, _ = _treat(runtime, "naive-mpc-optim")
    assert optim_conf["delta_forecast_daily"].days == expected
    assert grid.call_count <= 8


# heat_topology and the deferrable load count


def _topology(sources, extend=True):
    return {
        "sources": [
            {
                "id": f"hp{i}",
                "type": "heatpump",
                "nominal_power": 3000,
                "supply_temperature": 45,
                "treat_as_semi_cont": False,
            }
            for i in range(sources)
        ],
        "storage": [
            {
                "id": f"t{i}",
                "volume": 0.2,
                "start_temperature": 50.0,
                "min_temperature": [45.0],
                "max_temperature": [60.0] * 48,
            }
            for i in range(sources)
        ],
        "flows": [{"from": f"hp{i}", "to": f"t{i}"} for i in range(sources)],
        "extend_deferrable_loads": extend,
    }


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"number_of_deferrable_loads": 2, "heat_topology": _topology(150)}, 400),
        ({"number_of_deferrable_loads": 2, "heat_topology": _topology(99)}, 400),
        ({"heat_topology": _topology(101, extend=False)}, 400),
        ({"number_of_deferrable_loads": 2, "heat_topology": _topology(98)}, 200),
        ({"heat_topology": _topology(100, extend=False)}, 200),
    ],
)
def test_set_config_holds_the_loads_a_heat_topology_gives(web, body, status):
    before = web.files()
    code, reply = web.post("/set-config", orjson.dumps(body))
    assert code == status, reply
    if status == 400:
        assert json.loads(reply) == [
            "number_of_deferrable_loads and heat_topology give more than 100 deferrable "
            "loads in all"
        ]
        assert web.files() == before


def test_only_the_loads_a_request_gives_count_against_its_bound(caplog):
    # A configuration saved before the bound: 2 loads and a 150-source topology
    saved = _params_json(number_of_deferrable_loads=2, heat_topology=_topology(150))
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _, _, optim_conf, _ = _treat({}, "dayahead-optim", saved)
    assert optim_conf["number_of_deferrable_loads"] == 152
    assert (
        "The saved configuration gives more than 100 deferrable loads in all, the most a "
        "request or a new save may set"
    ) in [r.getMessage() for r in caplog.records]
    # A request that adds no loads runs on it
    _, _, optim_conf, _ = _treat({"number_of_deferrable_loads": 2}, "dayahead-optim", saved)
    assert optim_conf["number_of_deferrable_loads"] == 152
    # A topology the request sends counts in full
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({"heat_topology": _topology(101)}, "dayahead-optim", _params_json())
    assert str(caught.value) == "The request gives more than 100 deferrable loads in all"


def test_a_saved_load_count_over_the_saved_ceiling_is_refused():
    text = "The saved configuration gives more than 400 deferrable loads in all"
    # Publish ids for at most the ceiling are built before the final count is known:
    # as many for 1600 saved loads as for 401
    built_for = {}
    for loads in (401, 1600):
        ids = patch.object(
            utils, "_default_load_publish_ids", wraps=utils._default_load_publish_ids
        )
        with ids as built, pytest.raises(utils.RuntimeParamError) as caught:
            _treat({}, "dayahead-optim", _params_json(number_of_deferrable_loads=loads))
        assert str(caught.value) == text
        built_for[loads] = built.call_count
    assert built_for[1600] == built_for[401] <= 400
    saved = _params_json(number_of_deferrable_loads=100, heat_topology=_topology(301))
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({}, "dayahead-optim", saved)
    assert str(caught.value) == text


# deferrable_load_groups and the backend


def test_the_16_group_cap_is_the_coordinators():
    groups = _distinct_groups(17)
    six = {"number_of_deferrable_loads": 6}
    # The default solver takes them, on a request and on a save
    _treat({**six, "deferrable_load_groups": groups})
    assert _faults({"deferrable_load_groups": groups}) == []
    # A request that turns the coordinator on over 17 saved groups is refused
    saved = _params_json(**six, deferrable_load_groups=groups)
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({"optimization_backend": "dantzig_wolfe"}, "dayahead-optim", saved)
    assert str(caught.value) == (
        "deferrable_load_groups must have at most 16 groups of at most 200 names each "
        "with optimization_backend dantzig_wolfe"
    )


def test_groups_saved_over_the_cap_do_not_fail_build_params_or_a_run(caplog):
    saved = _params_json(
        number_of_deferrable_loads=6,
        deferrable_load_groups=_distinct_groups(17),
        optimization_backend="dantzig_wolfe",
    )
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _, _, optim_conf, _ = _treat({}, "dayahead-optim", saved)
    assert len(optim_conf["deferrable_load_groups"]) == 17
    messages = [r.getMessage() for r in caplog.records]
    assert any("In the saved configuration, deferrable_load_groups" in m for m in messages)


def test_set_config_takes_17_groups_for_the_default_solver(web):
    body = {"number_of_deferrable_loads": 6, "deferrable_load_groups": _distinct_groups(17)}
    assert web.post("/set-config", orjson.dumps(body))[0] == 200


# The participant window


def test_a_participant_window_longer_than_the_step_bound_is_refused():
    from emhass import participant_api
    from emhass.coordinator.remote import PlanWindow, WindowRefused

    params = orjson.loads(_params_json())
    params["retrieve_hass_conf"]["optimization_time_step"] = 1
    reached = AssertionError("the inputs were built")
    with patch.object(participant_api, "set_input_data_dict", side_effect=reached):
        with pytest.raises(WindowRefused, match="plans 1 to 5760 slots"):
            asyncio.run(participant_api.build_session(CONF, params, PlanWindow(1, 6000), logger))
        with pytest.raises(AssertionError, match="the inputs were built"):
            asyncio.run(participant_api.build_session(CONF, params, PlanWindow(1, 5760), logger))


# Numbers sent as text or as floats


@pytest.mark.parametrize(
    ("runtime", "set_type", "check"),
    [
        ({"prediction_horizon": "96"}, "naive-mpc-optim", ("passed", "prediction_horizon", 96)),
        ({"prediction_horizon": 96.0}, "naive-mpc-optim", ("passed", "prediction_horizon", 96)),
        ({"prediction_horizon": "9.6e1"}, "naive-mpc-optim", ("passed", "prediction_horizon", 96)),
        ({"optimization_time_step": "30.0"}, "dayahead-optim", None),
        (
            {"number_of_deferrable_loads": "3.0"},
            "dayahead-optim",
            ("optim", "number_of_deferrable_loads", 3),
        ),
        (
            {"historic_days_to_retrieve": "10.0"},
            "forecast-model-fit",
            ("passed", "historic_days_to_retrieve", 10),
        ),
    ],
)
def test_a_request_number_sent_as_text_or_float_is_used_as_a_number(runtime, set_type, check):
    params, _, optim_conf, _ = _treat(runtime, set_type)
    if check is not None:
        where, key, value = check
        got = orjson.loads(params)["passed_data"][key] if where == "passed" else optim_conf[key]
        assert got == value and type(got) is int


@pytest.mark.parametrize(
    ("body", "key", "saved"),
    [
        ({"optimization_time_step": "30.0"}, "optimization_time_step", 30),
        ({"number_of_deferrable_loads": "3.0"}, "number_of_deferrable_loads", 3),
        ({"delta_forecast_daily": "1e0"}, "delta_forecast_daily", 1),
        ({"lp_solver_timeout": "60"}, "lp_solver_timeout", 60.0),
    ],
)
def test_set_config_saves_a_number_sent_as_text_as_a_number(web, tmp_path, body, key, saved):
    code, reply = web.post("/set-config", orjson.dumps(body))
    assert code == 200, reply
    value = orjson.loads((tmp_path / "config.json").read_bytes())[key]
    assert value == saved and type(value) is type(saved)
    # And a run from it works
    _treat({}, "dayahead-optim", _params_json(**{key: value}))


@pytest.mark.parametrize("text", [b"'3.0'", b"3.0", b"'3'"])
def test_get_json_converts_a_number_sent_as_text_or_float(web, text):
    body = b"number_of_deferrable_loads: " + text + b"\n"
    status, reply = web.post("/get-json", body, "text/plain")
    assert status == 200, reply
    assert json.loads(reply)["number_of_deferrable_loads"] == 3


def test_get_json_names_an_alias_refusal(web):
    for yaml_text in (b"a: &x 1\nb: *x\n", b"base: &b {x: 1}\nother:\n  <<: *b\n  y: 2\n"):
        status, reply = web.post("/get-json", yaml_text, "text/plain")
        assert status == 400
        assert json.loads(reply) == ["YAML aliases and merge keys are not accepted"]


# The OptimizationCache


def test_a_cache_hit_takes_the_runs_time_zone():
    from types import SimpleNamespace

    zone = utils.pytz.timezone("Asia/Tokyo")
    opt = SimpleNamespace(
        time_zone=utils.pytz.timezone("Europe/Paris"),
        optim_conf={},
        update_thermal_start_temps=lambda conf: None,
        update_battery_power_limits=lambda conf: None,
    )
    command_line._apply_cached_opt_runtime_updates(
        opt,
        {"optim_conf": {}, "plant_conf": {}, "retrieve_hass_conf": {"time_zone": zone}},
        logger,
    )
    assert opt.time_zone is zone


@pytest.fixture(autouse=True)
def _fresh_saved_warnings():
    """Saved-value warnings are logged once per process: each test starts afresh."""
    for name in ("_SAVED_WARNED", "_SOLVER_OPTIONS_WARNED"):
        getattr(utils, name, set()).clear()


# Deferrable loads x time steps


LOAD_STEPS_TEXT = (
    "number_of_deferrable_loads plus the batteries in use (number_of_batteries, with "
    "set_use_battery) x the time steps of one solve (prediction_horizon, or "
    "delta_forecast_daily x 1440 / optimization_time_step) must be at most 100000"
)


def test_a_request_over_the_load_step_budget_is_refused():
    text = LOAD_STEPS_TEXT
    # 100 loads x 4 days at 1-minute steps: each within its own bound
    runtime = {
        "number_of_deferrable_loads": 100,
        "optimization_time_step": 1,
        "delta_forecast_daily": 4,
    }
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat(runtime)
    assert str(caught.value) == text
    # A horizon over the saved loads
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat(
            {"prediction_horizon": 5760},
            "naive-mpc-optim",
            _params_json(number_of_deferrable_loads=20),
        )
    assert str(caught.value) == text
    # Within the budget: 69 loads x 1440 steps
    _treat(
        {"number_of_deferrable_loads": 69, "optimization_time_step": 1, "delta_forecast_daily": 1}
    )
    # An action that does not optimize is not held to it
    _treat(
        runtime | {"delta_forecast_daily": 1, "number_of_deferrable_loads": 100},
        "forecast-model-fit",
    )


def test_saved_settings_over_the_load_step_budget_warn_then_refuse(caplog):
    over = _params_json(
        number_of_deferrable_loads=80, optimization_time_step=1, delta_forecast_daily=2
    )
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _treat({}, "dayahead-optim", over)
        _treat({}, "dayahead-optim", over)
    warned = [m for m in (r.getMessage() for r in caplog.records) if "x the time steps" in m]
    # Once per process, not at every run
    assert warned == [
        "The saved number_of_deferrable_loads plus the batteries in use x the time steps is "
        "more than 100000, the most a request or a new save may set"
    ]
    far_over = _params_json(
        number_of_deferrable_loads=100, optimization_time_step=1, delta_forecast_daily=3
    )
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({}, "dayahead-optim", far_over)
    assert str(caught.value) == (
        "The saved number_of_deferrable_loads plus the batteries in use x the time steps is "
        "more than 400000"
    )


@pytest.mark.parametrize(
    ("body", "status"),
    [
        (
            {
                "number_of_deferrable_loads": 100,
                "optimization_time_step": 1,
                "delta_forecast_daily": 4,
            },
            400,
        ),
        ({"number_of_deferrable_loads": 69, "optimization_time_step": 1}, 200),
        (
            {
                "number_of_deferrable_loads": 2,
                "heat_topology": _topology(70),
                "optimization_time_step": 1,
            },
            400,
        ),
    ],
)
def test_set_config_holds_the_load_step_budget(web, body, status):
    before = web.files()
    code, reply = web.post("/set-config", orjson.dumps(body))
    assert code == status, reply
    if status == 400:
        assert "x the time steps" in reply
        assert web.files() == before


# A configuration saved before the bounds can be saved again


def _write_saved(web, tmp_path, **config):
    saved = orjson.loads((tmp_path / "config.json").read_bytes())
    saved.update(config)
    (tmp_path / "config.json").write_bytes(orjson.dumps(saved))
    return saved


def test_an_unchanged_resave_of_a_value_over_a_bound_is_taken(web, tmp_path):
    saved = _write_saved(
        web, tmp_path, lp_solver_timeout=1000, optimization_time_step=1, delta_forecast_daily=5
    )
    # The page sends the whole configuration back, with one other setting changed
    body = {**saved, "battery_minimum_state_of_charge": 0.25}
    code, reply = web.post("/set-config", orjson.dumps(body))
    assert code == 200, reply
    # A change that is still over the bound is held to it
    for change in ({"lp_solver_timeout": 2000}, {"delta_forecast_daily": 6}):
        code, reply = web.post("/set-config", orjson.dumps({**body, **change}))
        assert code == 400, change
    # A change that shortens a grid still over the bound is taken
    assert web.post("/set-config", orjson.dumps({**body, "delta_forecast_daily": 4}))[0] == 200


def test_a_request_that_repeats_the_saved_step_is_judged_as_the_saved_settings(caplog):
    saved = _params_json(optimization_time_step=5, delta_forecast_daily=30)  # 8640 steps
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _treat({"optimization_time_step": 5}, "dayahead-optim", saved)
    assert any("give more than 5760 time steps" in r.getMessage() for r in caplog.records)
    with pytest.raises(utils.RuntimeParamError, match="give more than 5760 time steps"):
        _treat({"optimization_time_step": 4}, "dayahead-optim", saved)


# deferrable_load_groups: the coordinator falls back, a topology's groups count


def test_the_coordinator_falls_back_above_16_groups():
    from emhass.optimization import _coordinated_config_problem

    oc = {"optimization_backend": "dantzig_wolfe", "deferrable_load_groups": _distinct_groups(17)}
    assert _coordinated_config_problem(oc) == "more than 16 deferrable_load_groups"
    assert (
        _coordinated_config_problem({**oc, "deferrable_load_groups": _distinct_groups(16)}) is None
    )


def _actuated_topology(groups):
    topology = _topology(groups, extend=False)
    topology["actuator_groups"] = [
        {"flows": [[f"hp{i}", f"t{i}"]], "mutual_exclusion": True} for i in range(groups)
    ]
    return topology


def test_set_config_counts_a_heat_topologys_groups(web):
    topology = _actuated_topology(17)
    compiled = utils.compile_heat_topology(topology)
    assert len(compiled["deferrable_load_groups"]) == 17
    body = {"optimization_backend": "dantzig_wolfe", "heat_topology": topology}
    code, reply = web.post("/set-config", orjson.dumps(body))
    assert code == 400
    assert "at most 16 groups" in reply
    assert web.post("/set-config", orjson.dumps({"heat_topology": topology}))[0] == 200


# The documented maximum step, 1440 minutes


def test_a_one_day_step_runs_end_to_end(tmp_path):
    import shutil

    data = tmp_path / "data"
    shutil.copytree(TEST_ROOT / "data", data)
    conf = {**CONF, "data_path": data}
    params = orjson.loads(
        _params_json_with_secrets(optimization_time_step=1440, delta_forecast_daily=7)
    )
    runtime = {
        "prediction_horizon": 7,
        "pv_power_forecast": [1000.0] * 7,
        "load_power_forecast": [500.0] * 7,
        "load_cost_forecast": [0.2] * 7,
        "prod_price_forecast": [0.1] * 7,
    }
    for rt in (runtime, {**runtime, "optimization_time_step": 1440}):
        idd = asyncio.run(
            command_line.set_input_data_dict(
                conf,
                "profit",
                orjson.dumps(params).decode(),
                orjson.dumps(rt).decode(),
                "naive-mpc-optim",
                logger,
                get_data_from_file=True,
            )
        )
        assert idd
        res = asyncio.run(command_line.naive_mpc_optim(idd, logger))
        assert len(res) == 7
        assert idd["opt"].time_step == 24.0


# Other values of the wrong type or shape


@pytest.mark.parametrize(
    "runtime",
    [
        {"def_load_config": [5]},
        {"heat_topology": {"sources": 5, "storage": "x", "flows": [1]}},
        {"pv_power_forecast": 5},
        {"pv_power_forecast": {"2026-01-01T00:00:00+00:00": 1, "x": 2}},
        {"soc_init": "x"},
        {"weight_battery_discharge": "x"},
    ],
)
def test_a_value_of_the_wrong_shape_is_a_fixed_400(web, runtime):
    with patch.object(command_line, "RetrieveHass", side_effect=AssertionError("reached")):
        status, reply = web.post("/action/dayahead-optim", orjson.dumps(runtime))
    assert status == 400, reply
    assert "a value has the wrong type or shape" in reply
    assert "'x'" not in reply


@pytest.mark.parametrize(
    "body",
    [
        {"deferrable_load_groups": "x"},
        {"deferrable_load_groups": {"names": ["deferrable0"]}},
        {"deferrable_load_groups": [5]},
    ],
)
def test_set_config_answers_an_unreadable_setting_with_a_400(web, body):
    before = web.files()
    status, reply = web.post("/set-config", orjson.dumps(body))
    assert status == 400, reply
    assert web.files() == before


def test_get_json_answers_an_unreadable_configuration_with_a_400(web):
    for text in (
        b"deferrable_load_groups: [{names: [deferrable0]}]\nnumber_of_deferrable_loads: 1\n",
        b"number_of_batteries: 1000000\n",
    ):
        status, reply = web.post("/get-json", text, "text/plain")
        assert status == 400, (text, reply)


def test_solar_forecast_kwp_text_is_read_with_a_small_bound():
    from types import SimpleNamespace

    from emhass.forecast import Forecast

    long_text = "[" + "1," * 5000 + "1]"
    planes = SimpleNamespace(plant_conf={"pv_module_model": ["a", "b"]})
    with patch.object(utils.ast, "literal_eval", wraps=utils.ast.literal_eval) as parse:
        planes.retrieve_hass_conf = {"solar_forecast_kwp": long_text}
        values, fault = Forecast._solar_forecast_kwp_per_plane(planes)
        assert values is None and fault is not None
        parse.assert_not_called()
        planes.retrieve_hass_conf = {"solar_forecast_kwp": "[3.5, 4]"}
        assert Forecast._solar_forecast_kwp_per_plane(planes) == ([3.5, 4], None)


def test_thermal_lookback_is_bounded_in_rows():
    params_json = _params_json(optimization_time_step=30)
    with (
        patch.object(utils, "_load_thermal_calibration_frame") as load,
        pytest.raises(utils.RuntimeParamError) as caught,
    ):
        # 17520 hours at 30 minutes: 35040 rows
        asyncio.run(
            utils.compute_thermal_model_calibration(
                CONF, params_json, {"lookback_hours": 17520}, logger
            )
        )
    assert str(caught.value) == (
        "lookback_hours must give at most 20160 rows at the optimization_time_step"
    )
    load.assert_not_called()
    with pytest.raises(utils.RuntimeParamError) as caught:
        asyncio.run(
            utils.compute_thermal_model_calibration(
                CONF, params_json, {"sense": 5, "load_index": "x"}, logger
            )
        )
    assert "the wrong type" in str(caught.value)


def test_pv_bias_arrays_may_not_nest():
    body = _pv_body(p10=[[1.0] * 200])
    with pytest.raises(utils.RuntimeParamError) as caught:
        asyncio.run(command_line.pv_bias_calibration(body, logger))
    assert str(caught.value) == "p10 must be a list of numbers, not of lists"


# The request's time zone reaches the grid and the forecast


def test_a_request_time_zone_reaches_the_forecast(tmp_path):
    import shutil

    data = tmp_path / "data"
    shutil.copytree(TEST_ROOT / "data", data)
    conf = {**CONF, "data_path": data}
    params_json = _params_json_with_secrets()  # Europe/Paris
    runtime = {
        "time_zone": "Asia/Tokyo",
        "prediction_horizon": 10,
        "pv_power_forecast": [0.0] * 10,
        "load_power_forecast": [500.0] * 10,
        "load_cost_forecast": [0.2] * 10,
        "prod_price_forecast": [0.1] * 10,
    }
    idd = asyncio.run(
        command_line.set_input_data_dict(
            conf,
            "profit",
            params_json,
            orjson.dumps(runtime).decode(),
            "naive-mpc-optim",
            logger,
            get_data_from_file=True,
        )
    )
    assert str(idd["fcst"].time_zone) == "Asia/Tokyo"
    assert idd["fcst"].forecast_dates[0].utcoffset().total_seconds() == 9 * 3600
    assert str(idd["retrieve_hass_conf"]["time_zone"]) == "Asia/Tokyo"


def test_a_request_time_zone_across_a_dst_change_gives_its_grid():
    # 2026-03-08 is the New York spring-forward: that local day has 23 hours
    now = datetime(2026, 3, 8, 4, 0, tzinfo=UTC)
    runtime = {"time_zone": "America/New_York", "optimization_time_step": 60}
    real, grids = utils.get_forecast_dates, []

    def grid(*args, **kwargs):
        grids.append(real(*args, **kwargs))
        return grids[-1]

    with (
        patch.object(utils, "_get_now", return_value=now),
        patch.object(utils, "get_forecast_dates", side_effect=grid),
    ):
        params, rh, *_ = _treat(runtime)
    assert str(rh["time_zone"]) == "America/New_York"
    assert len(grids[0]) == 23
    assert orjson.loads(params)["passed_data"]["forecast_start"].endswith("-05:00")


# Batteries in use count with the loads against the load-step budget


BATTERIES = {"set_use_battery": True, "number_of_batteries": 100, "number_of_deferrable_loads": 0}


@pytest.mark.parametrize("flag", [True, 1, "true", "false", "no", False, 0, None])
def test_batteries_in_use_reads_set_use_battery_as_the_optimizer_does(tmp_path, flag):
    idd = _setup(tmp_path, {"set_use_battery": flag})
    res = asyncio.run(command_line.dayahead_forecast_optim(idd, logger))
    modelled = "P_batt" in res.columns
    assert modelled == bool(flag)
    assert utils.batteries_in_use(idd["optim_conf"], idd["plant_conf"]) == int(modelled)


def test_batteries_in_use_count_in_the_load_step_budget():
    # 100 batteries over 5760 one-minute steps: about 35 to 40 KB a battery step
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat(
            {**BATTERIES, "prediction_horizon": 5760, "optimization_time_step": 1},
            "naive-mpc-optim",
        )
    assert str(caught.value) == LOAD_STEPS_TEXT
    # 100 x 1000 is the budget itself
    _treat(
        {**BATTERIES, "prediction_horizon": 1000, "optimization_time_step": 1}, "naive-mpc-optim"
    )
    # Loads and batteries add up: 100 x 1001
    with pytest.raises(utils.RuntimeParamError, match="plus the batteries in use"):
        _treat(
            {
                **BATTERIES,
                "number_of_batteries": 50,
                "number_of_deferrable_loads": 50,
                "prediction_horizon": 1001,
                "optimization_time_step": 1,
            },
            "naive-mpc-optim",
        )
    # Batteries the run does not use do not count
    _treat(
        {
            **BATTERIES,
            "set_use_battery": False,
            "prediction_horizon": 5760,
            "optimization_time_step": 1,
        },
        "naive-mpc-optim",
    )


def test_saved_batteries_count_in_the_load_step_budget(caplog):
    # (2 loads + 100 batteries) x 1440 steps
    over = _params_json(set_use_battery=True, number_of_batteries=100, optimization_time_step=1)
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _treat({}, "dayahead-optim", over)
    assert any(
        "plus the batteries in use x the time steps is more than 100000" in r.getMessage()
        for r in caplog.records
    )
    # x 5760 steps: over four times the budget
    far_over = _params_json(
        set_use_battery=True,
        number_of_batteries=100,
        optimization_time_step=1,
        delta_forecast_daily=4,
    )
    with pytest.raises(utils.RuntimeParamError, match="more than 400000"):
        _treat({}, "dayahead-optim", far_over)


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"set_use_battery": True, "number_of_batteries": 100, "optimization_time_step": 1}, 400),
        ({"set_use_battery": True, "number_of_batteries": 60, "optimization_time_step": 1}, 200),
        ({"set_use_battery": False, "number_of_batteries": 100, "optimization_time_step": 1}, 200),
    ],
)
def test_set_config_counts_the_batteries_in_use(web, body, status):
    before = web.files()
    code, reply = web.post("/set-config", orjson.dumps(body))
    assert code == status, reply
    if status == 400:
        assert LOAD_STEPS_TEXT in reply
        assert web.files() == before


# Minimum on and off times: steps x min(minimum time, steps) terms a load


MIN_TIME_TEXT = (
    "def_minimum_on_time and def_minimum_off_time may give at most 5000000 terms in all "
    "(for each load, the time steps of one solve x the smaller of its minimum time and "
    "those time steps)"
)


def test_a_request_over_the_minimum_time_budget_is_refused():
    one = {"number_of_deferrable_loads": 1, "optimization_time_step": 1, "prediction_horizon": 2880}
    # 2880 x 2880 terms
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({**one, "def_minimum_on_time": [2880]}, "naive-mpc-optim")
    assert str(caught.value) == MIN_TIME_TEXT
    # A minimum longer than the steps counts as the steps
    assert utils.min_time_terms({"def_minimum_on_time": [5760]}, 1, 2880) == 2880 * 2880
    # 2880 x 1736 = 4999680, within it; one more off-time step tips it over
    _treat({**one, "def_minimum_on_time": [1736]}, "naive-mpc-optim")
    with pytest.raises(utils.RuntimeParamError, match="at most 5000000 terms"):
        _treat(
            {**one, "def_minimum_on_time": [1736], "def_minimum_off_time": [1]}, "naive-mpc-optim"
        )
    # A single value stands for every load: 3 x 1440 x 1440
    with pytest.raises(utils.RuntimeParamError, match="at most 5000000 terms"):
        _treat(
            {
                "number_of_deferrable_loads": 3,
                "optimization_time_step": 1,
                "def_minimum_on_time": 1440,
            }
        )
    # A sequence load gets no window constraints; a single-constant load does in the
    # relaxed retry, so it counts
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat(
            {**one, "def_minimum_on_time": [2880], "set_deferrable_load_single_constant": [True]},
            "naive-mpc-optim",
        )
    assert str(caught.value) == MIN_TIME_TEXT
    _treat(
        {**one, "def_minimum_on_time": [2880], "nominal_power_of_deferrable_loads": [[1000, 500]]},
        "naive-mpc-optim",
    )
    # An action that does not optimize is not held to it
    _treat({**one, "def_minimum_on_time": [2880]}, "forecast-model-fit")


def test_saved_minimum_times_over_the_budget_warn_then_refuse(caplog):
    # 3 x 1440 x 1440 terms
    over = _params_json(
        optimization_time_step=1, def_minimum_on_time=[1440, 1440], def_minimum_off_time=[1440, 0]
    )
    with caplog.at_level(logging.WARNING, logger=logger.name):
        _treat({}, "dayahead-optim", over)
        _treat({}, "dayahead-optim", over)
    warned = [m for m in (r.getMessage() for r in caplog.records) if "terms is more" in m]
    assert warned == [
        "The saved count of def_minimum_on_time and def_minimum_off_time terms is more than "
        "5000000, the most a request or a new save may set"
    ]
    # 4 x 4320 x 4320 terms: over four times the budget
    far_over = _params_json(
        optimization_time_step=1,
        delta_forecast_daily=3,
        def_minimum_on_time=[4320, 4320],
        def_minimum_off_time=[4320, 4320],
    )
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({}, "dayahead-optim", far_over)
    assert str(caught.value) == (
        "The saved count of def_minimum_on_time and def_minimum_off_time terms is more than "
        "20000000"
    )
    # A request that does not change them is judged as the saved settings are
    _treat({"optimization_time_step": 1}, "dayahead-optim", over)


def test_a_single_constant_load_counts_in_the_minimum_time_terms():
    single = {"set_deferrable_load_single_constant": [True, False], "def_minimum_on_time": [100, 0]}
    assert utils.min_time_terms(single, 2, 1000) == 100 * 1000
    # A request with one: 5760 x 1000 terms, over the budget
    saved = _params_json(optimization_time_step=1, delta_forecast_daily=4)
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat(
            {
                "set_deferrable_load_single_constant": [True, False],
                "def_minimum_on_time": [1000, 0],
            },
            "dayahead-optim",
            saved,
        )
    assert str(caught.value) == MIN_TIME_TEXT


def test_the_relaxed_retry_builds_the_minimum_times_of_a_single_constant_load(tmp_path):
    import cvxpy as cp

    real = cp.Problem.__init__

    def built(minimum):
        sizes = []

        def init(self, objective, constraints=None, *args, **kwargs):
            sizes.append(len(constraints or []))
            return real(self, objective, constraints, *args, **kwargs)

        runtime = {
            "set_deferrable_load_single_constant": [True, False],
            "set_deferrable_startup_penalty": [1, 0],
            "operating_hours_of_each_deferrable_load": [2, 2],
            "def_minimum_on_time": [minimum, 0],
            # Too short for the main solve, so the relaxed retry runs
            "lp_solver_timeout": 0.001,
        }
        idd = _setup(tmp_path / f"m{minimum}", runtime)
        with patch.object(cp.Problem, "__init__", init):
            asyncio.run(command_line.dayahead_forecast_optim(idd, logger))
        return sizes[0], sizes[-1]

    (main_0, relaxed_0), (main_30, relaxed_30) = built(0), built(30)
    # The main build leaves them out for the single-constant load; the retry has one
    # minimum on time constraint per step of the 48
    assert main_30 == main_0
    assert relaxed_30 - relaxed_0 == 48


@pytest.mark.parametrize(
    ("body", "status"),
    [
        (
            {
                "optimization_time_step": 1,
                "def_minimum_on_time": [1440, 1440],
                "def_minimum_off_time": [1440, 0],
            },
            400,
        ),
        ({"optimization_time_step": 1, "def_minimum_on_time": [1000, 0]}, 200),
    ],
)
def test_set_config_holds_the_minimum_time_budget(web, body, status):
    before = web.files()
    code, reply = web.post("/set-config", orjson.dumps(body))
    assert code == status, reply
    if status == 400:
        assert MIN_TIME_TEXT in reply
        assert web.files() == before


# perfect-optim solves one forecast window per history day


def test_perfect_optim_is_sized_by_one_forecast_window():
    # 365 history days at 5 minutes and 4 loads: each window is 288 steps
    saved = _params_json(
        optimization_time_step=5, historic_days_to_retrieve=365, number_of_deferrable_loads=4
    )
    _treat({}, "perfect-optim", saved)
    rh, oc, pc = utils.get_yaml_parse(saved, logger)
    asyncio.run(utils.treat_runtimeparams(None, saved, rh, oc, pc, "perfect-optim", logger, CONF))
    _treat({"historic_days_to_retrieve": 365, "number_of_deferrable_loads": 6}, "perfect-optim")
    # A window over the budget is refused with the settings that size it
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({"number_of_deferrable_loads": 70, "optimization_time_step": 1}, "perfect-optim")
    assert str(caught.value) == LOAD_STEPS_TEXT


# The ceilings of saved settings refuse only the actions that solve


def test_saved_ceilings_refuse_only_the_actions_that_solve(caplog):
    too_long = _params_json(optimization_time_step=1, delta_forecast_daily=17)  # 24480 steps
    too_many = _params_json(number_of_deferrable_loads=401)
    with caplog.at_level(logging.WARNING, logger=logger.name):
        for set_type in ("publish-data", "forecast-model-fit", "forecast-model-predict"):
            for saved in (too_long, too_many):
                _treat({}, set_type, saved)
                rh, oc, pc = utils.get_yaml_parse(saved, logger)
                asyncio.run(
                    utils.treat_runtimeparams(None, saved, rh, oc, pc, set_type, logger, CONF)
                )
    messages = [r.getMessage() for r in caplog.records]
    assert any("give more than 5760 time steps, the most" in m for m in messages)
    assert any("more than 100 deferrable loads in all, the most" in m for m in messages)
    for set_type in ("dayahead-optim", "naive-mpc-optim", "perfect-optim"):
        for saved, text in ((too_long, "23040 time steps"), (too_many, "400 deferrable loads")):
            with pytest.raises(utils.RuntimeParamError, match=text):
                _treat({}, set_type, saved)


# thermal-model-calibration date_ranges


def test_thermal_date_ranges_are_bounded_in_rows():
    params_json = _params_json(optimization_time_step=30)
    # 729 days at 30 minutes: 34992 rows
    year = {"date_ranges": [["2024-01-01T00:00:00Z", "2025-12-30T00:00:00Z"]]}
    with (
        patch.object(utils, "_load_thermal_calibration_frame") as load,
        pytest.raises(utils.RuntimeParamError) as caught,
    ):
        asyncio.run(utils.compute_thermal_model_calibration(CONF, params_json, year, logger))
    assert str(caught.value) == (
        "date_ranges must give at most 20160 rows at the optimization_time_step"
    )
    load.assert_not_called()

    class Read(Exception):
        pass

    # 200 days: 9600 rows, and 420 days: 20160 rows, the most, are read
    for end in ("2025-07-20T00:00:00Z", "2026-02-25T00:00:00Z"):
        within = {"date_ranges": [["2025-01-01T00:00:00Z", end]]}
        with (
            patch.object(utils, "_load_thermal_calibration_frame", side_effect=Read),
            pytest.raises(Read),
        ):
            asyncio.run(utils.compute_thermal_model_calibration(CONF, params_json, within, logger))


# A saved value that is not a number, sent back unchanged


def test_bounded_value_forms_leave_a_value_outside_its_bound():
    forms = utils.bounded_value_forms({"lp_solver_timeout": "abc", "num_threads": "1", "x": 1})
    assert forms == {"num_threads": 1}


@pytest.mark.parametrize(
    ("key", "value"), [("lp_solver_timeout", "abc"), ("num_threads", True), ("num_lags", "auto")]
)
def test_an_unchanged_resave_of_a_saved_non_number_is_taken(web, tmp_path, key, value):
    saved = _write_saved(web, tmp_path, **{key: value})
    body = {**saved, "battery_minimum_state_of_charge": 0.25}
    code, reply = web.post("/set-config", orjson.dumps(body))
    assert code == 200, reply
    assert orjson.loads((tmp_path / "config.json").read_bytes())[key] == value
    # A change is held to the bound
    code, reply = web.post("/set-config", orjson.dumps({**body, key: "abd"}))
    assert code == 400, reply


# prediction_horizon and the per-load numbers of a run


def test_prediction_horizon_is_two_steps_at_least():
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({"prediction_horizon": 1}, "naive-mpc-optim")
    assert str(caught.value) == "prediction_horizon must be a whole number from 2 to 5760"
    _treat({"prediction_horizon": 2}, "naive-mpc-optim")


@pytest.mark.parametrize(
    "runtime",
    [
        {"nominal_power_of_deferrable_loads": ["x", "y"]},
        {"nominal_power_of_deferrable_loads": [[1000, "x"], 500]},
        {"operating_hours_of_each_deferrable_load": ["x", 1]},
        {"start_timesteps_of_each_deferrable_load": ["x", 0]},
        {"start_timesteps_of_each_deferrable_load": ["2.5", 0]},
        {"def_start_timestep": [{"a": 1}, 0]},
        # The whole-number rule holds for a float too, and the size for every entry
        {"end_timesteps_of_each_deferrable_load": [10.5, 0]},
        {"set_deferrable_max_startups": [1.5, 0]},
        {"nominal_power_of_deferrable_loads": [1e308, 1000]},
        {"nominal_power_of_deferrable_loads": [[1000, -1e16], 500]},
        {"operating_hours_of_each_deferrable_load": ["1e300", 1]},
        {"start_timesteps_of_each_deferrable_load": [0.5, 1.5]},
        {"nominal_power_of_deferrable_loads": [-5, -5]},
        {"nominal_power_of_deferrable_loads": [[], 500]},
        {"minimum_power_of_deferrable_loads": ["x", 1]},
        {"minimum_power_of_deferrable_loads": [None, 1]},
    ],
)
@pytest.mark.parametrize("set_type", ["dayahead-optim", "naive-mpc-optim"])
def test_a_per_load_number_that_is_not_one_is_refused(runtime, set_type):
    name = next(iter(runtime))
    name = {"def_start_timestep": "start_timesteps_of_each_deferrable_load"}.get(name, name)
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat(runtime, set_type)
    assert str(caught.value) == utils.load_numbers_text(name)
    low = "0" if "power" in name else "-1e+15"
    kind = "whole number" if "timesteps" in name or "startups" in name else "number"
    assert str(caught.value) == (
        f"{name} must be a {kind} from {low} to 1e+15, or a list of {kind}s with one per "
        "deferrable load"
    )


def test_an_integral_float_per_load_step_is_read_as_an_int():
    _, _, optim_conf, _ = _treat(
        {
            "start_timesteps_of_each_deferrable_load": [3.0, 0],
            "set_deferrable_max_startups": [2.0, 1],
            "nominal_power_of_deferrable_loads": [1e15, 500],
        }
    )
    starts = optim_conf["start_timesteps_of_each_deferrable_load"]
    assert starts == [3, 0] and type(starts[0]) is int
    assert type(optim_conf["set_deferrable_max_startups"][0]) is int
    assert optim_conf["nominal_power_of_deferrable_loads"] == [1e15, 500]


def test_a_single_minimum_power_stands_for_every_load(tmp_path):
    _, _, optim_conf, _ = _treat({"minimum_power_of_deferrable_loads": 5})
    assert optim_conf["minimum_power_of_deferrable_loads"] == [5, 5]
    idd = _setup(tmp_path, {"minimum_power_of_deferrable_loads": "100"})
    assert len(asyncio.run(command_line.dayahead_forecast_optim(idd, logger))) == 48


def test_a_per_load_number_that_is_not_one_is_a_400_on_the_action(web):
    runtime = {"operating_hours_of_each_deferrable_load": ["x", 1]}
    with patch.object(command_line, "RetrieveHass", side_effect=AssertionError("reached")):
        status, reply = web.post("/action/dayahead-optim", orjson.dumps(runtime))
    assert status == 400, reply
    assert "operating_hours_of_each_deferrable_load must be a number" in reply
    assert "'x'" not in reply


def test_a_per_load_number_sent_as_text_is_read_as_a_number():
    _, _, optim_conf, _ = _treat(
        {
            "operating_hours_of_each_deferrable_load": ["2", 1.5],
            "start_timesteps_of_each_deferrable_load": ["3", 0],
            "nominal_power_of_deferrable_loads": ["1000", [500, "250"]],
            "set_deferrable_max_startups": [None, "2"],
        }
    )
    assert optim_conf["operating_hours_of_each_deferrable_load"] == [2.0, 1.5]
    assert optim_conf["start_timesteps_of_each_deferrable_load"] == [3, 0]
    assert optim_conf["nominal_power_of_deferrable_loads"] == [1000.0, [500, 250.0]]
    # A null entry takes the default, as before
    assert optim_conf["set_deferrable_max_startups"] == [0, 2]


# A save is judged by what it stores, against the configuration as it is served


@pytest.mark.parametrize(
    ("body", "text"),
    [
        # freq is not stored: the save keeps 30 minutes, 5760 x 48 steps
        ({"freq": 1440, "delta_forecast_daily": 5760}, "give more than 5760 time steps"),
        ({"freq": 1440, "delta_forecast_daily": 480}, "give more than 5760 time steps"),
        # 83 days at 30 minutes is 3984 steps, x 100 loads
        (
            {"freq": 1440, "delta_forecast_daily": 83, "number_of_deferrable_loads": 100},
            LOAD_STEPS_TEXT,
        ),
    ],
)
def test_set_config_judges_the_grid_it_stores_not_a_legacy_name(web, body, text):
    before = web.files()
    code, reply = web.post("/set-config", orjson.dumps(body))
    assert code == 400, reply
    assert text in reply
    assert web.files() == before


def test_stored_config_faults_read_the_stored_names_only():
    stored = {"optimization_time_step": 30, "delta_forecast_daily": 5760, "freq": 1440}
    assert utils.stored_config_faults(stored, None, None, None) == [
        "delta_forecast_daily and optimization_time_step give more than 5760 time steps"
    ]
    # No longer than the grid served now: taken
    assert utils.stored_config_faults(stored, None, dict(stored), None) == []


@pytest.mark.parametrize("key", ["number_of_deferrable_loads", "lp_solver_timeout", "num_threads"])
def test_a_boolean_is_not_the_same_as_a_saved_one(web, tmp_path, key):
    saved = _write_saved(web, tmp_path, **{key: 1})
    before = web.files()
    code, reply = web.post("/set-config", orjson.dumps({**saved, key: True}))
    assert code == 400, reply
    assert f"{key} must be" in reply
    assert web.files() == before


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("num_threads", "auto"),
        ("lp_solver_timeout", "abc"),
        ("num_lags", [1, 2, 3]),
        ("def_minimum_on_time", "[0, 0]"),
        ("historic_days_to_retrieve", 9.5),
        ("optimization_time_step", 7.5),
    ],
)
def test_a_kept_value_is_saved_as_it_was(web, tmp_path, key, value):
    saved = _write_saved(web, tmp_path, **{key: value})
    # The page sends it back with an unrelated change, twice
    for _ in range(2):
        body = {**orjson.loads((tmp_path / "config.json").read_bytes())}
        body["battery_minimum_state_of_charge"] = 0.25
        code, reply = web.post("/set-config", orjson.dumps(body))
        assert code == 200, reply
        stored = orjson.loads((tmp_path / "config.json").read_bytes())[key]
        assert utils.same_value(stored, value) and type(stored) is type(value), stored
    assert saved[key] == value


def test_same_value_tells_a_boolean_and_a_text_apart():
    same = utils.same_value
    assert same(600, 600.0) and same([1, 2.0], [1.0, 2]) and same({"a": [1]}, {"a": [1.0]})
    assert not same(True, 1) and not same(1, True) and not same("1", 1)
    assert not same([True], [1]) and not same({"a": 1}, {"b": 1})
    assert not same({"a": True}, {"a": 1}) and same({"a": [1, {"b": 2}]}, {"a": [1.0, {"b": 2}]})


def test_bounded_value_forms_write_a_value_within_its_bound_as_its_number():
    values = {"historic_days_to_retrieve": "3", "num_threads": 1.0, "num_lags": "auto"}
    forms = utils.bounded_value_forms(values)
    assert forms == {"historic_days_to_retrieve": 3, "num_threads": 1}
    assert all(type(value) is int for value in forms.values())


@pytest.mark.parametrize(
    ("key", "saved", "sent"),
    [("number_of_deferrable_loads", 2, 2.0), ("optimization_time_step", 30, 30.0)],
)
def test_an_integral_float_sent_over_a_saved_int_is_saved_as_the_int(
    web, tmp_path, key, saved, sent
):
    body = _write_saved(web, tmp_path, **{key: saved})
    code, reply = web.post("/set-config", orjson.dumps({**body, key: sent}))
    assert code == 200, reply
    stored = orjson.loads((tmp_path / "config.json").read_bytes())
    assert stored[key] == saved and type(stored[key]) is int, stored[key]
    # And every later run can use it
    params_json = _params_json_with_secrets(**stored)
    for set_type in ("dayahead-optim", "naive-mpc-optim", "publish-data"):
        _treat({}, set_type, params_json)


@pytest.mark.parametrize("error", [RuntimeError, OverflowError, RecursionError])
def test_a_saved_configuration_that_cannot_be_built_is_compared_as_it_is(web, tmp_path, error):
    from emhass import web_server

    real = web_server.build_params
    served = []

    async def build(*args, **kwargs):
        if sys._getframe(1).f_code.co_name == "_served_config":
            served.append(error.__name__)
            raise error("ZZMARK")
        return await real(*args, **kwargs)

    body = {**orjson.loads((tmp_path / "config.json").read_bytes())}
    body["battery_minimum_state_of_charge"] = 0.25
    with patch.object(web_server, "build_params", build):
        code, reply = web.post("/set-config", orjson.dumps(body))
    assert served == [error.__name__]
    assert code == 200, reply
    stored = orjson.loads((tmp_path / "config.json").read_bytes())
    assert stored["battery_minimum_state_of_charge"] == 0.25


@pytest.mark.parametrize(
    ("key", "value", "text"),
    [
        ("operating_hours_of_each_deferrable_load", "x", None),
        ("operating_hours_of_each_deferrable_load", [{"a": 1}, 2], None),
        ("start_timesteps_of_each_deferrable_load", [0.5, 1.5], None),
        ("set_deferrable_max_startups", [[1], [1]], None),
        ("deferrable_load_max_cost", {"a": 1}, None),
        ("nominal_power_of_deferrable_loads", [-1, -1], None),
        ("minimum_power_of_deferrable_loads", [None, None], None),
        ("minimum_power_of_deferrable_loads", "x", None),
        (
            "treat_deferrable_load_as_semi_cont",
            "3",
            "treat_deferrable_load_as_semi_cont must be true or false, or a list of them "
            "with one per deferrable load",
        ),
        (
            "set_deferrable_load_single_constant",
            {"a": 1},
            "set_deferrable_load_single_constant must be true or false, or a list of them "
            "with one per deferrable load",
        ),
    ],
)
def test_a_save_of_a_per_load_value_a_run_cannot_read_is_refused(web, key, value, text):
    before = web.files()
    code, reply = web.post("/set-config", orjson.dumps({key: value}))
    assert code == 400, reply
    assert (text or utils.load_numbers_text(key)) in reply
    assert "ZZ" not in reply
    assert web.files() == before


def test_a_saved_per_load_text_is_stored_as_the_number_a_run_reads(web, tmp_path):
    body = {
        "operating_hours_of_each_deferrable_load": ["3", 4],
        "start_timesteps_of_each_deferrable_load": [2.0, None],
        "minimum_power_of_deferrable_loads": 5,
    }
    code, reply = web.post("/set-config", orjson.dumps(body))
    assert code == 200, reply
    stored = orjson.loads((tmp_path / "config.json").read_bytes())
    assert stored["operating_hours_of_each_deferrable_load"] == [3.0, 4]
    assert stored["start_timesteps_of_each_deferrable_load"][0] == 2
    assert type(stored["start_timesteps_of_each_deferrable_load"][0]) is int
    assert stored["minimum_power_of_deferrable_loads"] == [5, 5]
    # And a run on it solves
    keys = (*body, "number_of_deferrable_loads")
    idd = _setup(tmp_path / "run", {}, **{key: stored[key] for key in keys})
    assert len(asyncio.run(command_line.dayahead_forecast_optim(idd, logger))) == 48


def test_a_saved_per_load_text_sent_back_unchanged_is_saved_as_it_was(web, tmp_path):
    _write_saved(web, tmp_path, operating_hours_of_each_deferrable_load="x")
    body = {**orjson.loads((tmp_path / "config.json").read_bytes())}
    body["battery_minimum_state_of_charge"] = 0.25
    code, reply = web.post("/set-config", orjson.dumps(body))
    assert code == 200, reply
    stored = orjson.loads((tmp_path / "config.json").read_bytes())
    assert stored["operating_hours_of_each_deferrable_load"] == "x"
    # A change is held to the reader
    code, reply = web.post(
        "/set-config", orjson.dumps({**body, "operating_hours_of_each_deferrable_load": "y"})
    )
    assert code == 400, reply


def test_an_unchanged_resave_is_judged_against_a_legacy_yaml(web, tmp_path):
    from emhass import web_server

    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(
        "retrieve_hass_conf:\n  freq: 5\n  days_to_retrieve: 9\n"
        "optim_conf:\n  lp_solver_timeout: 600\n  delta_forecast: 30\n  num_def_loads: 2\n"
        "plant_conf:\n  P_from_grid_max: 9000\n"
    )
    (tmp_path / "config.json").unlink()
    with patch.dict(web_server.emhass_conf, {"legacy_config_path": legacy}):
        client = web_server.app.test_client()

        async def served():
            r = await client.get("/get-config")
            return r.status_code, await r.get_json()

        code, config = asyncio.run(served())
        assert code == 200
        assert (config["lp_solver_timeout"], config["optimization_time_step"]) == (600, 5)
        # 30 days at 5 minutes, 8640 steps, and a 600 s limit: as served, so taken
        code, reply = web.post("/set-config", orjson.dumps(config))
        assert code == 200, reply
        code, reply = web.post("/set-config", orjson.dumps({**config, "lp_solver_timeout": 601}))
        assert code == 400, reply


def test_an_unchanged_resave_of_padded_lists_is_taken(web, tmp_path):
    from emhass import web_server

    # Written by hand: 120 loads, per-load lists of two
    _write_saved(web, tmp_path, number_of_deferrable_loads=120)
    client = web_server.app.test_client()

    async def served():
        r = await client.get("/get-config")
        return await r.get_json()

    config = asyncio.run(served())
    assert len(config["def_minimum_on_time"]) == 120
    code, reply = web.post("/set-config", orjson.dumps(config))
    assert code == 200, reply


def test_the_load_count_ceiling_is_judged_after_the_request():
    saved = _params_json(number_of_deferrable_loads=450)
    with pytest.raises(utils.RuntimeParamError, match="more than 400 deferrable loads"):
        _treat({}, "dayahead-optim", saved)
    # A request that lowers the count runs
    _, _, optim_conf, _ = _treat({"number_of_deferrable_loads": 2}, "dayahead-optim", saved)
    assert optim_conf["number_of_deferrable_loads"] == 2
    # An action that does not solve gets every load's publish ids
    params, *_ = _treat({}, "publish-data", saved)
    passed = orjson.loads(params)["passed_data"]
    assert len(passed["custom_deferrable_forecast_id"]) == 450


def test_a_request_that_repeats_the_saved_groups_is_judged_as_the_saved_settings(caplog):
    groups = _distinct_groups(17)
    saved = _params_json(
        optimization_backend="dantzig_wolfe",
        number_of_deferrable_loads=17,
        deferrable_load_groups=groups,
    )
    _treat({}, "dayahead-optim", saved)
    _treat({"deferrable_load_groups": groups}, "dayahead-optim", saved)
    _treat(
        {"deferrable_load_groups": groups, "optimization_backend": "dantzig_wolfe"},
        "dayahead-optim",
        saved,
    )
    # A change to them is held to the cap
    with pytest.raises(utils.RuntimeParamError, match="at most 16 groups"):
        _treat({"deferrable_load_groups": groups[:-1] + groups[:2]}, "dayahead-optim", saved)


# The settings a run cannot use: a fixed 400, the log names only the error's type


def _setup(tmp_path, runtime, set_type="dayahead-optim", **saved):
    import shutil

    data = tmp_path / "data"
    if not data.exists():
        shutil.copytree(TEST_ROOT / "data", data)
    conf = {**CONF, "data_path": data}
    n = 48
    body = {
        "pv_power_forecast": [1000.0] * n,
        "load_power_forecast": [500.0] * n,
        "load_cost_forecast": [0.2] * n,
        "prod_price_forecast": [0.1] * n,
        **({"prediction_horizon": n} if set_type == "naive-mpc-optim" else {}),
        **runtime,
    }
    return asyncio.run(
        command_line.set_input_data_dict(
            conf,
            "profit",
            _params_json_with_secrets(**saved),
            orjson.dumps(body).decode(),
            set_type,
            logger,
            get_data_from_file=True,
        )
    )


@pytest.mark.parametrize(
    "runtime",
    [
        {"def_load_config": [{"thermal_config": "x"}]},
        {"method_ts_round": "x"},
        {"weight_battery_discharge": "ZZMARK-<b>caller text</b>"},
    ],
)
def test_settings_the_forecast_or_optimizer_cannot_use_are_a_fixed_400(tmp_path, runtime, caplog):
    with (
        caplog.at_level(logging.ERROR, logger=logger.name),
        pytest.raises(utils.RuntimeParamError) as caught,
    ):
        _setup(tmp_path, runtime)
    assert str(caught.value) == command_line.SETTINGS_REFUSED
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "The run's settings could not be used (" in logged
    assert "ZZMARK" not in logged and "'x'" not in logged


@pytest.mark.parametrize(
    ("runtime", "text"),
    [
        (
            {"set_use_battery": True, "battery_charge_power_max": [1000, 2000]},
            "battery_charge_power_max has 2 entries but number_of_batteries=1",
        ),
        (
            {"weight_battery_discharge": ["ZZMARK", 1]},
            "weight_battery_discharge: an entry is a text that is not a number",
        ),
    ],
)
def test_a_settings_refusal_logs_the_text_of_emhass_own_check(tmp_path, runtime, text, caplog):
    with (
        caplog.at_level(logging.ERROR, logger=logger.name),
        pytest.raises(utils.RuntimeParamError) as caught,
    ):
        _setup(tmp_path, runtime)
    assert str(caught.value) == command_line.SETTINGS_REFUSED
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert text in logged
    # The place is the last line of EMHASS the error passed, and no value is quoted
    assert "(SettingsError at utils.py:" in logged
    assert "ZZMARK" not in logged


def test_a_settings_refusal_logs_the_whole_error_at_debug_level(tmp_path, caplog):
    runtime = {"weight_battery_discharge": "ZZMARK-<b>caller text</b>"}
    with (
        caplog.at_level(logging.DEBUG, logger=logger.name),
        pytest.raises(utils.RuntimeParamError),
    ):
        _setup(tmp_path, runtime)
    full = [r for r in caplog.records if "the error in full" in r.getMessage()]
    assert len(full) == 1
    assert full[0].levelno == logging.DEBUG and full[0].exc_info[1] is not None


@pytest.mark.parametrize("level", [logging.INFO, logging.DEBUG])
def test_a_dry_run_does_not_return_the_callers_text(web, level):
    from emhass import web_server

    body = {"weight_battery_discharge": "ZZMARK-<b>caller text</b>"}
    debug = []

    class Debug(logging.Handler):
        def emit(self, record):
            if record.levelno == logging.DEBUG:
                debug.append(record.getMessage())

    app_logger, handler = web_server.app.logger, Debug(logging.DEBUG)
    before = app_logger.level
    app_logger.setLevel(level)
    app_logger.addHandler(handler)
    try:
        status, reply = web.post("/api/v1/dry-run/naive-mpc-optim", orjson.dumps(body))
    finally:
        app_logger.removeHandler(handler)
        app_logger.setLevel(before)
    assert status == 400, reply
    assert "ZZMARK" not in reply and "<b>" not in reply
    # At DEBUG level the whole error was logged, and still not returned
    assert any("the error in full" in m for m in debug) == (level == logging.DEBUG)


def test_a_dry_run_that_fails_in_the_solve_does_not_return_the_callers_text(
    web, tmp_path, monkeypatch
):
    import functools
    import shutil

    from emhass import web_server

    # The run reads its data from the test files, not from Home Assistant
    shutil.copytree(TEST_ROOT / "data", tmp_path, dirs_exist_ok=True)
    monkeypatch.setattr(
        web_server,
        "set_input_data_dict",
        functools.partial(command_line.set_input_data_dict, get_data_from_file=True),
    )
    # A per-load cost forecast of nested texts passes the setup and fails in the run
    n = 48
    body = {
        "pv_power_forecast": [1000.0] * n,
        "load_power_forecast": [500.0] * n,
        "load_cost_forecast": [0.2] * n,
        "prod_price_forecast": [0.1] * n,
        "prediction_horizon": n,
        "cost_forecast_per_deferrable_load": [[["ZZMARK-<b>caller text</b>"]]],
    }
    # Another test module lowers the "emhass" logger to CRITICAL, which would drop the
    # error line the response collects
    before = web_server.app.logger.level
    web_server.app.logger.setLevel(logging.INFO)
    try:
        status, reply = web.post("/api/v1/dry-run/naive-mpc-optim", orjson.dumps(body))
    finally:
        web_server.app.logger.setLevel(before)
    assert status == 500, reply
    assert "ZZMARK" not in reply and "<b>" not in reply
    errors = orjson.loads(reply)["errors"]
    assert any(e.startswith("Dry run of naive-mpc-optim failed (") for e in errors), errors


@pytest.mark.parametrize("key", ["def_minimum_on_time", "def_minimum_off_time"])
def test_a_saved_single_minimum_time_stands_for_every_load(tmp_path, key):
    idd = _setup(tmp_path, {}, **{key: 3})
    assert idd["optim_conf"][key] == [3, 3]
    res = asyncio.run(command_line.dayahead_forecast_optim(idd, logger))
    assert len(res) == 48


def _setup_typical(tmp_path, set_type="dayahead-optim", **saved):
    """_setup with the default typical load forecast: no load_power_forecast list."""
    import shutil

    data = tmp_path / "data"
    shutil.copytree(TEST_ROOT / "data", data)
    body = {"pv_power_forecast": [1000.0] * 96, "load_cost_forecast": [0.2] * 96}
    body["prod_price_forecast"] = [0.1] * 96
    if set_type == "naive-mpc-optim":
        body["prediction_horizon"] = 10
    return asyncio.run(
        command_line.set_input_data_dict(
            {**CONF, "data_path": data},
            "profit",
            _params_json_with_secrets(**saved),
            orjson.dumps(body).decode(),
            set_type,
            logger,
            get_data_from_file=True,
        )
    )


@pytest.mark.parametrize("step", [7, 45, 90, 1440])
def test_a_step_the_typical_load_forecast_cannot_be_made_for_is_a_400(tmp_path, step):
    from emhass import forecast

    with pytest.raises(utils.RuntimeParamError) as caught:
        _setup_typical(tmp_path, optimization_time_step=step, delta_forecast_daily=2)
    assert str(caught.value) == forecast.TYPICAL_FORECAST_REFUSED


@pytest.mark.parametrize("step", [20, 30, 60])
def test_a_step_the_typical_load_forecast_can_be_made_for_runs(tmp_path, step):
    idd = _setup_typical(tmp_path, optimization_time_step=step)
    assert len(idd["df_input_data_dayahead"]) == 1440 // step


def test_a_cost_list_over_a_one_step_grid_is_a_400(tmp_path):
    from emhass import forecast

    idd = _setup(tmp_path, {}, optimization_time_step=1440, delta_forecast_daily=1)
    # The cost forecast is read in the action, which answers RuntimeParamError with a 400
    with pytest.raises(utils.RuntimeParamError) as caught:
        asyncio.run(command_line.dayahead_forecast_optim(idd, logger))
    assert str(caught.value) == forecast.ONE_STEP_FORECAST_REFUSED


# Rules the budget and the write-back hang on, one test each


def test_naive_mpc_is_sized_by_its_horizon_not_the_grid():
    # 70 loads over a saved 4-day one-minute grid: 403200 load steps a day-ahead solve
    saved = _params_json(
        number_of_deferrable_loads=70, optimization_time_step=1, delta_forecast_daily=4
    )
    with pytest.raises(utils.RuntimeParamError, match="more than 400000"):
        _treat({}, "dayahead-optim", saved)
    # naive-mpc solves the horizon: 70 x 1000
    _treat({"prediction_horizon": 1000}, "naive-mpc-optim", saved)
    # 70 x 1500 is over the budget, but no more than the saved settings give at their
    # own grid, and within the saved ceiling: judged as saved, so it runs
    _treat({"prediction_horizon": 1500}, "naive-mpc-optim", saved)


def test_a_naive_mpc_horizon_is_held_to_the_load_steps_of_the_saved_grid():
    # 18 loads over the saved 48-step grid: 864 load steps
    saved = _params_json(number_of_deferrable_loads=18)
    _treat({"prediction_horizon": 5555}, "naive-mpc-optim", saved)  # 99990
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({"prediction_horizon": 5556}, "naive-mpc-optim", saved)  # 100008
    assert str(caught.value) == LOAD_STEPS_TEXT


def test_a_naive_mpc_horizon_alone_is_judged_as_the_saved_minimum_times():
    # 10 loads at on 180 and off 180, saved at a 1-minute step over one day:
    # 10 x 1440 x 360 = 5184000 terms at the saved grid
    saved = _params_json(
        optimization_time_step=1,
        number_of_deferrable_loads=10,
        def_minimum_on_time=[180] * 10,
        def_minimum_off_time=[180] * 10,
    )
    # The same count at a horizon of 1440: over the budget, but no more than the
    # saved settings give, so it runs (with the saved-value warning)
    _treat({"prediction_horizon": 1440}, "naive-mpc-optim", saved)
    # So does a request that sends the saved minimum times back
    _treat(
        {"prediction_horizon": 1440, "def_minimum_on_time": [180] * 10},
        "naive-mpc-optim",
        saved,
    )
    # One that gives more terms than the saved grid does is held to the budget
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat(
            {"prediction_horizon": 1440, "def_minimum_on_time": [181] * 10},
            "naive-mpc-optim",
            saved,
        )
    assert str(caught.value) == MIN_TIME_TEXT
    # ... as is a longer horizon (10 x 5760 x 360), with the request's text
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({"prediction_horizon": 5760}, "naive-mpc-optim", saved)
    assert str(caught.value) == MIN_TIME_TEXT


def test_a_naive_mpc_horizon_cannot_grow_small_saved_minimum_times_past_the_budget():
    # Minimum times of 5760 saved over the default 48-step grid: 2 x 2 x 48 x 48 terms
    saved = _params_json(
        number_of_deferrable_loads=2,
        def_minimum_on_time=[5760] * 2,
        def_minimum_off_time=[5760] * 2,
    )
    assert utils.saved_run_size(orjson.loads(saved)).min_time_terms == 9216
    _treat({"prediction_horizon": 1118}, "naive-mpc-optim", saved)  # 4999696 terms
    # 2 x 2 x 1119 x 1119 = 5008644 terms: over the budget at one time it, not four
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({"prediction_horizon": 1119}, "naive-mpc-optim", saved)
    assert str(caught.value) == MIN_TIME_TEXT


def test_a_request_no_larger_than_the_saved_budget_is_judged_as_saved(caplog):
    # 80 loads x 2880 steps saved: over the budget, under the saved ceiling
    saved = _params_json(
        number_of_deferrable_loads=80, optimization_time_step=1, delta_forecast_daily=2
    )
    _treat({"number_of_deferrable_loads": 80}, "dayahead-optim", saved)
    _treat({"number_of_deferrable_loads": 79}, "dayahead-optim", saved)
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({"number_of_deferrable_loads": 81}, "dayahead-optim", saved)
    assert str(caught.value) == LOAD_STEPS_TEXT


def test_a_request_topology_sizes_the_budget():
    saved = _params_json(number_of_deferrable_loads=69, optimization_time_step=1)  # 99360
    _treat({}, "dayahead-optim", saved)
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({"heat_topology": _topology(2)}, "dayahead-optim", saved)
    assert str(caught.value) == LOAD_STEPS_TEXT


def test_legacy_names_size_the_budget_at_run_time():
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({"num_def_loads": 100, "freq": 1, "delta_forecast": 4})
    assert str(caught.value) == LOAD_STEPS_TEXT


@pytest.mark.parametrize("error", [KeyError, AttributeError, IndexError, TypeError, ValueError])
@pytest.mark.parametrize("where", ["treat_runtimeparams", "Forecast"])
def test_each_shape_error_of_the_setup_is_a_fixed_400(tmp_path, error, where):
    target = (
        (utils, "treat_runtimeparams")
        if where == "treat_runtimeparams"
        else (command_line, "Forecast")
    )
    with (
        patch.object(*target, side_effect=error("ZZMARK")),
        pytest.raises(utils.RuntimeParamError) as caught,
    ):
        _setup(tmp_path, {})
    assert str(caught.value) == command_line.SETTINGS_REFUSED


def test_the_write_back_reaches_the_calibrations_and_the_steps_kind():
    payload = {"lookback_hours": "48.0", "curtailment_margin": "2"}
    utils.check_thermal_calibration_payload(payload)
    assert payload["lookback_hours"] == 48 and type(payload["lookback_hours"]) is int
    # 48 as text is used as 48: on 10 rows it drops every row, as 48 does
    with pytest.raises(ValueError, match="no usable observations left"):
        asyncio.run(command_line.pv_bias_calibration(_pv_body(curtailment_margin="48.0"), logger))
    _, _, optim_conf, _ = _treat({"def_minimum_on_time": ["3", 2.0]})
    assert optim_conf["def_minimum_on_time"] == [3, 2]
    assert [type(v) for v in optim_conf["def_minimum_on_time"]] == [int, int]


# The batteries, the minimum times and the per-load numbers in the run size, one test each


def test_the_saved_run_size_counts_the_batteries_in_use():
    saved = orjson.loads(_params_json(set_use_battery=True, number_of_batteries=3))
    assert utils.saved_run_size(saved).batteries == 3
    assert utils.batteries_in_use({"set_use_battery": True}, {}) == 1


def test_battery_keys_in_a_request_size_the_load_step_budget():
    saved = _params_json(
        number_of_deferrable_loads=0, optimization_time_step=1, delta_forecast_daily=4
    )
    # 20 batteries x 5760 steps: over the budget, within the saved ceiling
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({"set_use_battery": True, "number_of_batteries": 20}, "dayahead-optim", saved)
    assert str(caught.value) == LOAD_STEPS_TEXT


def test_a_single_minimum_time_counts_for_every_load():
    assert utils.min_time_terms({"def_minimum_on_time": 10}, 3, 100) == 3 * 100 * 10


@pytest.mark.parametrize(
    "runtime", [{"def_minimum_on_time": 1000}, {"number_of_deferrable_loads": 2}]
)
def test_minimum_time_and_load_keys_in_a_request_size_the_terms(runtime):
    # Saved: one load with a minimum on time of 900 at 5760 steps, 5184000 terms (over
    # the budget, within the saved ceiling); the request asks for more
    saved = _params_json(
        optimization_time_step=1,
        delta_forecast_daily=4,
        number_of_deferrable_loads=1,
        def_minimum_on_time=900,
        def_minimum_off_time=0,
    )
    _treat({}, "dayahead-optim", saved)
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat(runtime, "dayahead-optim", saved)
    assert str(caught.value) == MIN_TIME_TEXT


def test_naive_mpc_terms_are_counted_by_the_horizon_not_the_grid():
    # 4 loads at on 1000 over a 1-minute grid of 4 days: over the saved ceiling there
    saved = _params_json(
        optimization_time_step=1,
        delta_forecast_daily=4,
        number_of_deferrable_loads=4,
        def_minimum_on_time=[1000] * 4,
        def_minimum_off_time=[0] * 4,
    )
    with pytest.raises(utils.RuntimeParamError, match="terms is more than 20000000"):
        _treat({}, "dayahead-optim", saved)
    _treat({"prediction_horizon": 100}, "naive-mpc-optim", saved)


def test_the_per_load_numbers_are_read_only_for_an_optimizing_action():
    _treat({"nominal_power_of_deferrable_loads": ["x", 1]}, "publish-data")


def test_saved_per_load_lists_are_not_read_at_run_time():
    saved = _params_json(operating_hours_of_each_deferrable_load=["x", 1])
    _treat({}, "dayahead-optim", saved)


def test_the_per_load_number_log_names_the_entry_not_its_value(caplog):
    with (
        caplog.at_level(logging.ERROR, logger=logger.name),
        pytest.raises(utils.RuntimeParamError),
    ):
        _treat({"nominal_power_of_deferrable_loads": [1000, "ZZMARK"]})
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "nominal_power_of_deferrable_loads[1]" in logged and "ZZMARK" not in logged


@pytest.mark.parametrize(
    "error", [utils.SocFractionError, utils.InverterConfigError, utils.BatteryConfigError]
)
def test_an_error_with_its_own_answer_is_not_the_fixed_400(tmp_path, error):
    with (
        patch.object(command_line, "Forecast", side_effect=error("its own text")),
        pytest.raises(error),
    ):
        _setup(tmp_path, {})


def test_a_save_over_the_term_budget_is_taken_when_the_saved_one_gives_as_much(web, tmp_path):
    saved = _write_saved(
        web,
        tmp_path,
        optimization_time_step=1,
        delta_forecast_daily=4,
        def_minimum_on_time=[1000, 1000],
        def_minimum_off_time=[0, 0],
    )
    body = {**saved, "battery_minimum_state_of_charge": 0.25}
    code, reply = web.post("/set-config", orjson.dumps(body))
    assert code == 200, reply
    code, reply = web.post(
        "/set-config", orjson.dumps({**body, "def_minimum_on_time": [1001, 1000]})
    )
    assert code == 400 and MIN_TIME_TEXT in reply, reply


# Legacy names, flag texts, short grids and the thermal lookback, one test each


@pytest.mark.parametrize("key", ["number_of_batteries", "num_batteries"])
def test_a_request_battery_count_sizes_the_budget_under_either_name(key):
    # 2 loads and one battery over a saved 4-day 1-minute grid: 17280 load steps
    saved = _params_json(
        number_of_deferrable_loads=2,
        optimization_time_step=1,
        delta_forecast_daily=4,
        set_use_battery=True,
    )
    _treat({key: 15}, "dayahead-optim", saved)  # 17 x 5760 = 97920
    # 18 x 5760 = 103680; under the legacy name it ran up to 67 batteries
    for count in (16, 67):
        with pytest.raises(utils.RuntimeParamError) as caught:
            _treat({key: count}, "dayahead-optim", saved)
        assert str(caught.value) == LOAD_STEPS_TEXT


def test_every_legacy_name_of_a_sizing_setting_sizes_a_request():
    import csv

    legacy = {}
    with open(CONF["associations_path"], encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 3 and row[1] != row[2]:
                legacy.setdefault(row[2], set()).add(row[1])
    tuples = {
        "load count": utils._REQUEST_LOAD_COUNT_KEYS,
        "load steps": utils._REQUEST_LOAD_STEP_KEYS,
        "minimum time terms": utils._REQUEST_MIN_TIME_KEYS,
        "time grid": tuple(k for pair in utils._GRID_KEYS for k in pair),
    }
    for name, keys in tuples.items():
        for key in keys:
            missing = legacy.get(key, set()) - set(keys)
            assert not missing, (name, key, missing)


def test_an_unchanged_resave_of_a_saved_text_flag_is_taken(web, tmp_path):
    _write_saved(web, tmp_path, treat_deferrable_load_as_semi_cont="true")
    body = {**orjson.loads((tmp_path / "config.json").read_bytes())}
    body["battery_minimum_state_of_charge"] = 0.25
    code, reply = web.post("/set-config", orjson.dumps(body))
    assert code == 200, reply
    stored = orjson.loads((tmp_path / "config.json").read_bytes())
    assert stored["treat_deferrable_load_as_semi_cont"] == "true"
    # A new text is held to the reader
    code, reply = web.post(
        "/set-config", orjson.dumps({**body, "treat_deferrable_load_as_semi_cont": "false"})
    )
    assert code == 400, reply


def test_a_two_step_grid_runs_with_cost_lists(tmp_path):
    n = 2
    lists = {
        "pv_power_forecast": [1000.0] * n,
        "load_power_forecast": [500.0] * n,
        "load_cost_forecast": [0.2] * n,
        "prod_price_forecast": [0.1] * n,
    }
    idd = _setup(tmp_path, lists, optimization_time_step=720, delta_forecast_daily=1)
    assert len(asyncio.run(command_line.dayahead_forecast_optim(idd, logger))) == n


@pytest.mark.parametrize("step, hours", [(30, 10080), (1, 336)])
def test_a_thermal_lookback_of_exactly_the_row_bound_is_read(step, hours):
    params_json = _params_json(optimization_time_step=step)

    class Read(Exception):
        pass

    with (
        patch.object(utils, "_load_thermal_calibration_frame", side_effect=Read) as load,
        pytest.raises(Read),
    ):
        # hours / step: exactly 20160 rows
        asyncio.run(
            utils.compute_thermal_model_calibration(
                CONF, params_json, {"lookback_hours": hours}, logger
            )
        )
    load.assert_called_once()


def test_a_capacity_interval_list_of_the_wrong_length_is_a_settings_error():
    optim_conf = {
        "capacity_cost_per_kw": [1.0, 2.0],
        "capacity_charge_interval_timesteps": [1, 2, 3],
    }
    with pytest.raises(utils.SettingsError) as caught:
        utils.canonicalize_capacity_charge_config(optim_conf, logger)
    assert str(caught.value).startswith(
        "capacity_charge_interval_timesteps has 3 entries but capacity_cost_per_kw defines 2"
    )


# A save and a run count the size of a configuration with one function (config_shape)


def _replaced_loads(loads, minimum, **config):
    """Settings of ``loads`` loads with minimum times of ``minimum`` steps at 1-minute
    steps, which a saved heat_topology in replace mode compiles to one load."""
    return {
        "optimization_time_step": 1,
        "number_of_deferrable_loads": loads,
        "nominal_power_of_deferrable_loads": [1000.0] * loads,
        "def_minimum_on_time": [minimum] * loads,
        "def_minimum_off_time": [minimum] * loads,
        "heat_topology": _topology(1, extend=False),
        **config,
    }


def test_a_horizon_is_measured_against_the_saved_topology_loads():
    # Five loads the topology replaces with one: 2 x 1440 x 1440 saved terms, as a
    # save counts them, not five times that
    saved = _params_json(**_replaced_loads(5, 5760))
    size = utils.saved_run_size(orjson.loads(saved))
    assert (size.loads, size.min_time_terms) == (1, 2 * 1440 * 1440)
    _treat({"prediction_horizon": 1440}, "naive-mpc-optim", saved)
    # 2 x 1600 x 1600 = 5120000 terms for the one load: more than saved, over the budget
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({"prediction_horizon": 1600}, "naive-mpc-optim", saved)
    assert str(caught.value) == MIN_TIME_TEXT


def test_a_request_topology_is_measured_against_the_saved_topology_loads():
    # 20 loads a saved topology replaces with one, at 5760 steps: 5760 load steps
    # saved, not 20 x 5760, whether or not the request sends its own topology
    saved = _params_json(**_replaced_loads(20, 0, delta_forecast_daily=4))
    _treat({"heat_topology": _topology(17, extend=False)}, "dayahead-optim", saved)  # 97920
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({"heat_topology": _topology(18, extend=False)}, "dayahead-optim", saved)
    assert str(caught.value) == LOAD_STEPS_TEXT


def test_a_save_counts_a_replaced_sequence_load_with_the_power_the_run_uses(web):
    # A sequence load, which no minimum time is counted for, that the topology replaces
    # with its heat pump: the run counts 2 x 2880 x 2880 terms on two 1-minute days
    body = _replaced_loads(1, 5760, delta_forecast_daily=2)
    body["nominal_power_of_deferrable_loads"] = [[1000.0, 1000.0]]
    before = web.files()
    code, reply = web.post("/set-config", orjson.dumps(body))
    assert code == 400, reply
    assert MIN_TIME_TEXT in reply
    assert web.files() == before
    # The run of such a configuration, saved before, counts the same terms
    saved = orjson.loads(_params_json(**body))
    assert utils.saved_run_size(saved).min_time_terms == 2 * 2880 * 2880


def _flag_text(name):
    return f"{name} must be true or false, or a list of them with one per deferrable load"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("treat_deferrable_load_as_semi_cont", ["false", "x"]),
        ("treat_deferrable_load_as_semi_cont", [2, True]),
        ("set_deferrable_load_single_constant", {}),
        ("set_deferrable_load_single_constant", [{}, []]),
        ("set_def_constant", [True, "maybe"]),
    ],
)
def test_a_request_flag_a_run_cannot_read_is_refused(key, value):
    name = {"set_def_constant": "set_deferrable_load_single_constant"}.get(key, key)
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({key: value})
    assert str(caught.value) == _flag_text(name)


def test_a_request_flag_is_read_as_the_boolean_it_names():
    _, _, optim_conf, _ = _treat(
        {
            "treat_deferrable_load_as_semi_cont": "false",
            "set_deferrable_load_single_constant": ["TRUE", 0],
        }
    )
    assert optim_conf["treat_deferrable_load_as_semi_cont"] == [False, False]
    assert optim_conf["set_deferrable_load_single_constant"] == [True, False]
    for key in ("treat_deferrable_load_as_semi_cont", "set_deferrable_load_single_constant"):
        assert all(type(flag) is bool for flag in optim_conf[key])


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("treat_deferrable_load_as_semi_cont", [True, "false"]),
        ("treat_deferrable_load_as_semi_cont", [2, 0]),
        ("set_deferrable_load_single_constant", [{}, []]),
    ],
)
def test_a_save_reads_each_flag_entry(web, key, value):
    before = web.files()
    code, reply = web.post("/set-config", orjson.dumps({key: value}))
    assert code == 400, reply
    assert _flag_text(key) in reply
    assert web.files() == before
    # 0 and 1 are read as false and true by every reader of a flag
    assert web.post("/set-config", orjson.dumps({key: [True, 0]}))[0] == 200


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("nominal_power_of_deferrable_loads", [True, False]),
        ("start_timesteps_of_each_deferrable_load", [True, 0]),
        ("set_deferrable_max_startups", True),
    ],
)
def test_a_boolean_is_not_a_per_load_number(web, key, value):
    with pytest.raises(utils.RuntimeParamError) as caught:
        _treat({key: value})
    assert str(caught.value) == utils.load_numbers_text(key)
    before = web.files()
    code, reply = web.post("/set-config", orjson.dumps({key: value}))
    assert code == 400, reply
    assert utils.load_numbers_text(key) in reply
    assert web.files() == before
