"""Dry runs: POST /api/v1/dry-run/<action> solves as POST /action/<action>
does, returns the plan, and writes nothing.

Nothing means: no results CSV, entity file, plan store or last-run record, no
weather cache or adjusted-PV model saved, refreshed or deleted, no battery
identification, no publish, and the OptimizationCache left as the last live
run left it. A dry run's log records stay out of the shared action log, and
its response carries only its own. A dry_run parameter anywhere else is
refused, so no caller gets a live run it did not ask for.
"""

import asyncio
import hashlib
import json
import logging
import os
import pathlib
import pickle
import shutil
import tempfile
import time
from contextlib import contextmanager
from functools import partial
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pandas as pd
import pytest

from emhass import command_line, last_run, plan_store, utils, web_server
from emhass.command_line import (
    OptimizationCache,
    dayahead_forecast_optim,
    is_dry_run,
    main,
    naive_mpc_optim,
    perfect_forecast_optim,
    set_input_data_dict,
)
from emhass.forecast import Forecast
from emhass.utils import (
    build_config,
    build_params,
    build_secrets,
    dry_run_request_problem,
    get_logger,
    get_root,
    read_flag,
)

root = pathlib.Path(get_root(__file__, num_parent=2))
logger, _ = get_logger(__name__, {"data_path": root / "data/"}, save_to_file=False)
SCHEMA = json.loads((root / "docs/api/v1/dry-run.schema.json").read_text(encoding="utf-8"))

FORECASTS = {
    "pv_power_forecast": [i + 1 for i in range(48)],
    "load_power_forecast": [i + 1 for i in range(48)],
    "load_cost_forecast": [0.1 + (i % 6) / 20 for i in range(48)],
    "prod_price_forecast": [0.05] * 48,
}
ACTIONS = {
    "perfect-optim": perfect_forecast_optim,
    "dayahead-optim": dayahead_forecast_optim,
    "naive-mpc-optim": naive_mpc_optim,
}


def _conf(data_path: pathlib.Path) -> dict:
    """EMHASS paths with `data_path` as the data folder."""
    return {
        "data_path": data_path,
        "root_path": root / "src/emhass/",
        "defaults_path": root / "src/emhass/data/config_defaults.json",
        "associations_path": root / "src/emhass/data/associations.csv",
    }


@contextmanager
def _data_copy():
    """A copy of the test data folder, removed afterwards."""
    with tempfile.TemporaryDirectory() as tmp:
        data_path = pathlib.Path(tmp) / "data"
        shutil.copytree(root / "data", data_path)
        yield data_path


def _files(data_path: pathlib.Path) -> dict:
    """Every file under `data_path`, with its size, modification time and hash."""
    return {
        p.relative_to(data_path): (
            p.stat().st_size,
            p.stat().st_mtime_ns,
            hashlib.sha256(p.read_bytes()).hexdigest(),
        )
        for p in data_path.rglob("*")
        if p.is_file()
    }


def _changed(before: dict, after: dict) -> list:
    """The files created, deleted or modified between two _files() calls."""
    return sorted(str(p) for p in set(before) | set(after) if before.get(p) != after.get(p))


async def _params(data_path: pathlib.Path) -> dict:
    """The default parameters, with PV on (as tests/test_command_line_utils.py)."""
    ec = _conf(data_path)
    _, secrets = await build_secrets(ec, logger, no_response=True)
    params = await build_params(
        ec, secrets, await build_config(ec, logger, ec["defaults_path"]), logger
    )
    params["optim_conf"]["set_use_pv"] = True
    return params


def _run(
    data_path: pathlib.Path,
    action: str,
    *,
    dry: bool,
    runtime: dict | None = None,
    save: bool = False,
    configure=None,
    costfun: str = "profit",
):
    """Build the inputs for `action` and run its wrapper, live or dry, not in
    debug mode (debug also skips the CSV, which would make the checks below
    vacuous). Returns (input_data_dict, the wrapper's result)."""

    async def run():
        params = await _params(data_path)
        if configure is not None:
            configure(params)
        rt = {**FORECASTS, **(runtime or {})}
        if action == "naive-mpc-optim":
            rt["prediction_horizon"] = 48
        idd = await set_input_data_dict(
            _conf(data_path),
            costfun,
            orjson.dumps(params).decode(),
            orjson.dumps(rt).decode(),
            action,
            logger,
            get_data_from_file=True,
            dry_run=dry,
        )
        assert idd, f"{action}: the inputs could not be built"
        return idd, await ACTIONS[action](idd, logger, save_data_to_file=save, debug=False)

    return asyncio.run(run())


@pytest.fixture(autouse=True)
def _cold_cache():
    """Each test starts and ends with an empty OptimizationCache, and leaves
    the in-memory last-run and plan snapshots as it found them."""
    saved = last_run._cache, plan_store._cache
    OptimizationCache.clear()
    yield
    OptimizationCache.clear()
    last_run._cache, plan_store._cache = saved


# --------------------------------------------------------------------------
# The request: refused before anything runs, wherever it cannot be honoured
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        (None, False),
        ("true", True),
        ("True", True),
        (" yes ", True),
        ("on", True),
        ("1", True),
        ("false", False),
        ("no", False),
        ("off", False),
        ("0", False),
        ("", False),
    ],
)
def test_a_flag_is_read_strictly(value, expected):
    """In particular "false" is False: bool("false") is True."""
    assert read_flag(value, "x") is expected


@pytest.mark.parametrize("value", ["maybe", "2", 2, -1, 1.0, [], {}, [True]])
def test_a_flag_that_cannot_be_read_raises(value):
    with pytest.raises(ValueError, match="x must be true or false"):
        read_flag(value, "x")


@pytest.mark.parametrize(
    "runtime, dry, refused",
    [
        ({}, False, False),
        ({"dry_run": False}, False, False),
        ({"dry_run": "false"}, False, False),
        ({"entity_save": True}, False, False),
        ({"dry_run": True}, False, True),
        ({"dry_run": "yes"}, False, True),
        ({"dry_run": "maybe"}, False, True),
        ({}, True, False),
        ({"dry_run": True}, True, False),
        ({"weather_forecast_cache_only": True}, True, False),
        ({"entity_save": False}, True, False),
        ({"dry_run": False}, True, True),
        ({"dry_run": "maybe"}, True, True),
        ({"entity_save": True}, True, True),
        ({"entity_save": "true"}, True, True),
        ({"weather_forecast_cache": True}, True, True),
    ],
)
def test_which_requests_are_refused(runtime, dry, refused):
    """An action URL refuses a dry_run that is not false (or cannot be read).
    The dry-run URL refuses a dry_run that is not true, and the parameters
    that ask for a write; a read-only one (weather_forecast_cache_only) is
    fine."""
    problem = dry_run_request_problem(runtime, dry_run=dry)
    assert (problem is not None) is refused, problem
    if refused:
        assert problem.endswith("Nothing was run")


def test_a_live_refusal_names_the_dry_run_url():
    problem = dry_run_request_problem({"dry_run": True}, dry_run=False)
    assert "/api/v1/dry-run/" in problem


# --------------------------------------------------------------------------
# The optimisation: a live run writes, a dry run writes nothing
# --------------------------------------------------------------------------


def _continual_publish(params: dict) -> None:
    params["retrieve_hass_conf"]["continual_publish"] = True


CONFIGS = {"plain": None, "continual_publish": _continual_publish}


@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("save", [False, True], ids=["latest", "dated"])
@pytest.mark.parametrize("action", list(ACTIONS))
def test_a_live_run_writes_its_results(action, save, config):
    """The control for the test below: the same run, live, writes its results
    CSV, the last-run record and the plan store (and, under continual publish,
    the entity files), so a dry run writing none is the dry run's doing."""
    with _data_copy() as data_path:
        before = _files(data_path)
        idd, res = _run(data_path, action, dry=False, save=save, configure=CONFIGS[config])
        changed = _changed(before, _files(data_path))
    assert isinstance(res, pd.DataFrame)
    assert any(p.endswith(".csv") for p in changed), changed
    assert "last_run.json" in changed and "plan_latest.json" in changed, changed
    if config == "continual_publish":
        assert any(p.startswith("entities") for p in changed), changed
    assert "run_record" not in idd


@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize("save", [False, True], ids=["latest", "dated"])
@pytest.mark.parametrize("action", list(ACTIONS))
def test_a_dry_run_writes_nothing(action, save, config):
    """Every optimisation action, saving the latest results or a dated file,
    with continual publish on or off, returns its plan on a dry run and leaves
    the data folder byte for byte as it was. The record it would have stored
    for /api/v1/last-run comes back with it instead."""
    with _data_copy() as data_path:
        before = _files(data_path)
        idd, res = _run(data_path, action, dry=True, save=save, configure=CONFIGS[config])
        assert _changed(before, _files(data_path)) == []
    assert isinstance(res, pd.DataFrame) and len(res) > 0
    assert is_dry_run(idd)
    assert idd["run_record"]["action"] == action
    assert idd["run_record"]["status"] == "ok"


def _cache_state() -> tuple:
    """What the next live solve inherits: the cached object, its key, its
    configuration, and the solution it warm-starts from."""
    opt = OptimizationCache._instance
    values = tuple(None if v.value is None else v.value.copy() for v in opt.prob.variables())
    return opt, OptimizationCache._cache_key, opt.optim_conf, opt.plant_conf, values


def _same(a: tuple, b: tuple) -> bool:
    """Whether two cache states are identical, objects and values alike."""
    opt_a, key_a, oc_a, pc_a, vals_a = a
    opt_b, key_b, oc_b, pc_b, vals_b = b
    return (
        opt_a is opt_b
        and key_a == key_b
        and oc_a is oc_b
        and pc_a is pc_b
        and len(vals_a) == len(vals_b)
        and all(
            (x is None and y is None) or (x is not None and y is not None and (x == y).all())
            for x, y in zip(vals_a, vals_b, strict=True)
        )
    )


def test_a_dry_run_leaves_the_live_optimisation_untouched():
    """A dry run at other prices (a cache hit, were it read) and one with
    another cost function (a miss, which would evict the live problem) leave
    the cached problem as the live solve left it: same object, key,
    configuration and solution to warm-start from."""
    with _data_copy() as data_path:
        _, live = _run(data_path, "dayahead-optim", dry=False)
        assert isinstance(live, pd.DataFrame) and OptimizationCache._instance is not None
        before = _cache_state()
        prices = [0.5 if i % 2 else 0.01 for i in range(48)]
        _, trial = _run(
            data_path, "dayahead-optim", dry=True, runtime={"load_cost_forecast": prices}
        )
        assert isinstance(trial, pd.DataFrame)
        assert _same(before, _cache_state())
        _, other = _run(data_path, "dayahead-optim", dry=True, costfun="cost")
        assert isinstance(other, pd.DataFrame)
        assert _same(before, _cache_state())


@pytest.mark.parametrize("action", list(ACTIONS))
def test_a_dry_run_plans_exactly_as_a_live_run(action):
    """The same inputs, each from a cold cache, give the same plan live and
    dry: a dry run changes what is written, never what is solved."""
    with _data_copy() as data_path:
        _, live = _run(data_path, action, dry=False)
        OptimizationCache.clear()
        _, dry = _run(data_path, action, dry=True)
    pd.testing.assert_frame_equal(dry, live)


def test_a_dry_run_shows_what_a_per_step_import_cap_does():
    """The shadow evaluation in the docs: the live request with
    maximum_power_from_grid as a per-step list. The dry plan holds the cap at
    every step; the live plan, written and kept, is the uncapped one."""
    cap = [9000.0] * 48
    cap[16:24] = [500.0] * 8
    with _data_copy() as data_path:
        _, live = _run(data_path, "naive-mpc-optim", dry=False)
        before = _files(data_path)
        _, capped = _run(
            data_path, "naive-mpc-optim", dry=True, runtime={"maximum_power_from_grid": cap}
        )
        assert _changed(before, _files(data_path)) == []
    assert (capped["P_grid"].to_numpy() <= pd.Series(cap).to_numpy() + 1e-6).all()
    assert live["P_grid"].iloc[16:24].max() > 500.0, "the cap should bind for the test to mean much"


def _battery_identification(params: dict) -> None:
    params["optim_conf"]["set_use_battery"] = True
    params["optim_conf"]["set_use_battery_identification"] = True


@pytest.mark.parametrize("dry", [False, True], ids=["live", "dry"])
def test_a_dry_run_does_not_identify_the_battery(dry):
    """Battery identification saves its estimate, restarts its cadence and
    publishes to Home Assistant, so a dry run skips it. A live run runs it."""
    with (
        _data_copy() as data_path,
        patch.object(command_line, "identify_battery", AsyncMock()) as identify,
    ):
        _run(data_path, "dayahead-optim", dry=dry, configure=_battery_identification)
    assert identify.called is not dry


def _adjusted_pv(params: dict) -> None:
    params["optim_conf"]["set_use_adjusted_pv"] = True


@pytest.mark.parametrize("dry", [False, True], ids=["live", "dry"])
def test_a_stale_adjusted_pv_model_is_refitted_but_saved_only_live(dry):
    """A model older than adjusted_pv_model_max_age is refitted. A live run
    saves the new one; a dry run fits it in memory, as the live run would,
    and leaves the saved one for the next live run."""
    with _data_copy() as data_path:
        model = data_path / "adjust_pv_regressor.pkl"
        model.unlink(missing_ok=True)
        _run(data_path, "dayahead-optim", dry=False, configure=_adjusted_pv)
        assert model.exists(), "the live run should have fitted and saved a model"
        old = time.time() - 48 * 3600
        os.utime(model, (old, old))
        before = _files(data_path)
        idd, res = _run(data_path, "dayahead-optim", dry=dry, configure=_adjusted_pv)
        changed = _changed(before, _files(data_path))
    assert isinstance(res, pd.DataFrame)
    assert getattr(idd["fcst"], "model_adjust_pv", None) is not None
    if dry:
        assert changed == []
    else:
        assert "adjust_pv_regressor.pkl" in changed


# --------------------------------------------------------------------------
# The weather caches: read on a dry run, never saved, refreshed or deleted
# --------------------------------------------------------------------------


class _Response:
    """An aiohttp response that answers with `payload`."""

    status = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    """An aiohttp session whose GETs answer with `payload`; it counts them."""

    def __init__(self, payload, calls):
        self._payload = payload
        self._calls = calls

    def get(self, url, headers=None):
        self._calls.append(url)
        if len(self._calls) > 3:
            raise AssertionError("fetched again and again: a loop")
        return _Response(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _open_meteo_payload() -> dict:
    """An Open-Meteo answer in the unixtime format EMHASS asks for, covering
    a day before now to four days after."""
    now = int(time.time()) // 900 * 900
    times = list(range(now - 86400, now + 4 * 86400, 900))
    fields = {
        "temperature_2m": 15.0,
        "relative_humidity_2m": 60.0,
        "rain": 0.0,
        "cloud_cover": 20.0,
        "wind_speed_10m": 3.0,
        "shortwave_radiation_instant": 300.0,
        "diffuse_radiation_instant": 100.0,
        "direct_normal_irradiance_instant": 400.0,
    }
    return {"minutely_15": {"time": times, **{k: [v] * len(times) for k, v in fields.items()}}}


async def _forecast(data_path: pathlib.Path, dry: bool, log: logging.Logger) -> Forecast:
    """A Forecast on `data_path`, for a dry run or a live one."""
    params = await _params(data_path)
    if dry:
        params["passed_data"]["dry_run"] = True
    params_json = orjson.dumps(params).decode()
    retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, log)
    return Forecast(retrieve_hass_conf, optim_conf, plant_conf, params_json, _conf(data_path), log)


@pytest.mark.parametrize("dry", [False, True], ids=["live", "dry"])
def test_the_open_meteo_caches_are_refreshed_only_live(dry):
    """With a stale Open-Meteo JSON cache, a stale weather cache and DEBUG
    logging, a live run deletes the stale cache, fetches, saves the new JSON
    and writes its debug CSV. A dry run fetches and plans with the fresh
    forecast too, and leaves every file as it was."""
    debug_log = logging.getLogger(f"{__name__}.debug")
    debug_log.setLevel(logging.DEBUG)
    debug_log.propagate = False
    calls: list = []
    with (
        _data_copy() as data_path,
        patch(
            "emhass.forecast.aiohttp.ClientSession",
            lambda *a, **k: _Session(_open_meteo_payload(), calls),
        ),
    ):
        json_cache = data_path / "cached-open-meteo-forecast-b.json"
        json_cache.write_bytes(orjson.dumps(_open_meteo_payload()))
        stale = pd.DataFrame(
            {"temp_air": [10.0, 11.0]},
            index=pd.date_range("2020-01-01", periods=2, freq="30min", tz="UTC"),
        )
        (data_path / "weather_forecast_data.pkl").write_bytes(pickle.dumps(stale))
        old = time.time() - 2 * 3600
        os.utime(json_cache, (old, old))
        for debug_csv in data_path.glob("debug-*.csv"):
            debug_csv.unlink()
        before = _files(data_path)

        async def fetch():
            fcst = await _forecast(data_path, dry, debug_log)
            return await fcst.get_weather_forecast(method="open-meteo")

        weather = asyncio.run(fetch())
        changed = _changed(before, _files(data_path))
    assert isinstance(weather, pd.DataFrame) and len(weather) > 0
    assert len(calls) == 1, calls
    if dry:
        assert changed == []
    else:
        assert {
            "cached-open-meteo-forecast-b.json",
            "weather_forecast_data.pkl",
            "debug-weather-forecast-open-meteo.csv",
        } <= set(changed), changed


@pytest.mark.parametrize("dry", [False, True], ids=["live", "dry"])
def test_the_covariates_cache_is_saved_only_live(dry):
    """The ML forecaster's Open-Meteo covariates are fetched either way, and
    cached only by a live run."""
    payload = {"minutely_15": {"time": [], "temperature_2m": []}}
    with (
        _data_copy() as data_path,
        patch(
            "emhass.forecast.aiohttp.ClientSession",
            lambda *a, **k: _Session(payload, []),
        ),
    ):
        cache = data_path / "cached-open-meteo-covariates.json"
        cache.unlink(missing_ok=True)

        async def fetch():
            fcst = await _forecast(data_path, dry, logger)
            return await fcst._fetch_open_meteo_covariates_json(1, 1)

        asyncio.run(fetch())
        assert cache.exists() is not dry


def test_a_dry_run_does_not_save_the_weather_cache():
    """set_cached_forecast_data, which weather_forecast_cache_only reaches on
    a missing Solcast cache, saves nothing on a dry run."""
    with _data_copy() as data_path:
        cache = data_path / "weather_forecast_data.pkl"
        cache.unlink(missing_ok=True)

        async def save():
            fcst = await _forecast(data_path, True, logger)
            data = pd.DataFrame({"yhat": 1.0}, index=fcst.forecast_dates)
            return await fcst.set_cached_forecast_data(str(cache), data)

        data = asyncio.run(save())
        assert not cache.exists()
    assert isinstance(data, pd.DataFrame)


# --------------------------------------------------------------------------
# The URLs: /action/* refuses dry_run; /api/v1/dry-run/* refuses what it can't do
# --------------------------------------------------------------------------


def _conforms(body: dict) -> None:
    """Assert that a dry-run response follows docs/api/v1/dry-run.schema.json."""
    props = SCHEMA["properties"]
    assert set(SCHEMA["required"]) <= set(body), set(SCHEMA["required"]) - set(body)
    assert set(body) <= set(props), set(body) - set(props)
    assert body["dry_run"] is True
    assert body["status"] in props["status"]["enum"]
    assert all(isinstance(m, str) for m in body["warnings"] + body["errors"])
    if body["status"] == "not-run":
        assert body["plan"] is None
    else:
        assert set(SCHEMA["allOf"][0]["else"]["required"]) <= set(body)
        assert isinstance(body["plan"], list)


@contextmanager
def _web(data_path: pathlib.Path, level: int = logging.INFO):
    """The web app on `data_path`, logging at `level` to the action log there
    as _configure_logging sets it up. Restores all of it afterwards."""
    saved_conf = web_server.emhass_conf
    saved_level = web_server.app.logger.level
    saved_threads = list(web_server.continual_publish_thread)
    web_server.emhass_conf = {**_conf(data_path), "config_path": data_path / "config.json"}
    handler = logging.FileHandler(str(data_path / web_server.action_log_str))
    handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
    handler.addFilter(web_server._not_from_a_dry_run)
    web_server.app.logger.addHandler(handler)
    web_server.app.logger.setLevel(level)
    try:
        yield web_server.app.test_client()
    finally:
        web_server.app.logger.removeHandler(handler)
        handler.close()
        web_server.app.logger.setLevel(saved_level)
        web_server.emhass_conf = saved_conf
        web_server.continual_publish_thread[:] = saved_threads


def _loaded(runtime: dict):
    """A stand-in for _load_params_and_runtime that loads `runtime`."""
    return AsyncMock(return_value=({"optim_conf": {}}, "profit", orjson.dumps(runtime).decode()))


ALL_ACTIONS = [
    "perfect-optim",
    "dayahead-optim",
    "naive-mpc-optim",
    "publish-data",
    "weather-forecast-cache",
    "export-influxdb-to-csv",
    "pv-bias-calibration",
    "forecast-model-fit",
]


@pytest.mark.parametrize("action", ALL_ACTIONS)
def test_an_action_url_refuses_a_dry_run(action):
    """POST /action/<any action> with dry_run true is refused before anything
    runs, publish-data included: it would otherwise publish."""
    with (
        tempfile.TemporaryDirectory() as tmp,
        _web(pathlib.Path(tmp)) as client,
        patch.object(web_server, "_load_params_and_runtime", _loaded({"dry_run": True})),
        patch.object(web_server, "set_input_data_dict", AsyncMock()) as build,
        patch.object(web_server, "_handle_action_dispatch", AsyncMock()) as dispatch,
        patch.object(web_server, "publish_data", AsyncMock()) as publish,
    ):

        async def post():
            response = await client.post(f"/action/{action}", json={"dry_run": True})
            return response.status_code, await response.get_data(as_text=True)

        status, text = asyncio.run(post())
    assert status == 400
    assert "/api/v1/dry-run/" in text
    build.assert_not_called()
    dispatch.assert_not_called()
    publish.assert_not_called()


def test_an_action_url_runs_with_dry_run_false():
    """The control: dry_run false is a live run, as it always was."""
    with (
        tempfile.TemporaryDirectory() as tmp,
        _web(pathlib.Path(tmp)) as client,
        patch.object(web_server, "_load_params_and_runtime", _loaded({"dry_run": False})),
        patch.object(
            web_server,
            "set_input_data_dict",
            AsyncMock(return_value={"retrieve_hass_conf": {}, "params": {}}),
        ),
        patch.object(web_server, "_handle_action_dispatch", AsyncMock(return_value=("ok", 200))),
        patch.object(web_server, "check_file_log", AsyncMock(return_value=False)),
    ):
        status = asyncio.run(client.post("/action/dayahead-optim", json={})).status_code
    assert status == 200


@pytest.mark.parametrize(
    "action, body",
    [
        ("publish-data", {}),
        ("forecast-model-fit", {}),
        ("naive-mpc-optim", {"entity_save": True}),
        ("naive-mpc-optim", {"weather_forecast_cache": True}),
        ("naive-mpc-optim", {"dry_run": False}),
        ("naive-mpc-optim", {"dry_run": "maybe"}),
        ("naive-mpc-optim", "not json"),
        ("naive-mpc-optim", "[1, 2]"),
    ],
)
def test_the_dry_run_url_refuses_what_it_cannot_do(action, body):
    """An action a dry run cannot do, a parameter that asks for a write, a
    contradicting dry_run, and a body that is not a JSON object are refused
    (400, status not-run, the reason in errors) before anything runs."""
    with (
        tempfile.TemporaryDirectory() as tmp,
        _web(pathlib.Path(tmp)) as client,
        patch.object(web_server, "set_input_data_dict", AsyncMock()) as build,
    ):

        async def post():
            if isinstance(body, str):
                response = await client.post(f"/api/v1/dry-run/{action}", data=body)
            else:
                response = await client.post(f"/api/v1/dry-run/{action}", json=body)
            return response.status_code, await response.get_json()

        status, answer = asyncio.run(post())
    assert status == 400
    _conforms(answer)
    assert answer["status"] == "not-run" and answer["action"] == action
    assert answer["errors"] and answer["errors"][-1].endswith("Nothing was run")
    build.assert_not_called()


def _write_params(data_path: pathlib.Path, configure=None) -> None:
    """Save params.pkl as the web server does, for _load_params_and_runtime."""
    params = asyncio.run(_params(data_path))
    if configure is not None:
        configure(params)
    (data_path / web_server.params_file).write_bytes(
        pickle.dumps((data_path / "config.json", params))
    )


def _from_file():
    """set_input_data_dict on the saved test data rather than Home Assistant."""
    return patch.object(
        web_server, "set_input_data_dict", partial(set_input_data_dict, get_data_from_file=True)
    )


@pytest.mark.parametrize("action", list(ACTIONS))
def test_the_dry_run_url_returns_the_plan_and_writes_nothing(action):
    """End to end through the web app, with continual publish configured:
    200, the run record and the plan, and the data folder unchanged. No page
    injection file is saved and the continual-publish loop is not started (it
    keeps the inputs it starts with, which must be a live run's)."""
    with _data_copy() as data_path:
        _write_params(data_path, _continual_publish)
        body = {**FORECASTS, "prediction_horizon": 48} if action == "naive-mpc-optim" else FORECASTS
        with (
            _web(data_path) as client,
            _from_file(),
            patch.object(web_server, "_save_injection_dict", AsyncMock()) as inject,
            patch.object(web_server.app, "add_background_task", MagicMock()) as start,
        ):
            before = _files(data_path)

            async def post():
                response = await client.post(f"/api/v1/dry-run/{action}", json=body)
                return response.status_code, response.headers, await response.get_json()

            status, headers, answer = asyncio.run(post())
            changed = _changed(before, _files(data_path))
    assert status == 200, answer["errors"]
    _conforms(answer)
    assert answer["status"] == "ok" and answer["action"] == action
    assert len(answer["plan"]) > 0 and "optim_status" in answer["plan"][0]
    assert headers["Cache-Control"] == "no-store"
    assert changed == []
    inject.assert_not_called()
    start.assert_not_called()


def test_the_plan_and_last_run_endpoints_do_not_see_a_dry_run():
    """After a live run, GET /api/v1/plan and /api/v1/last-run serve it. A dry
    run at other prices, through the dry-run URL, changes neither response,
    in memory or on disk, though its own plan differs."""
    with _data_copy() as data_path:
        _write_params(data_path)
        _, live = _run(data_path, "dayahead-optim", dry=False)
        prices = [0.5 if i % 2 else 0.01 for i in range(48)]
        with _web(data_path) as client, _from_file():

            async def endpoints():
                plan = await client.get("/api/v1/plan")
                run = await client.get("/api/v1/last-run")
                return await plan.get_json(), await run.get_json()

            async def dry():
                body = {**FORECASTS, "load_cost_forecast": prices}
                response = await client.post("/api/v1/dry-run/dayahead-optim", json=body)
                return await response.get_json()

            before = asyncio.run(endpoints())
            answer = asyncio.run(dry())
            last_run._cache = None  # read from disk too, not only the in-memory copy
            plan_store._cache = None
            after = asyncio.run(endpoints())
    assert before[0]["status"] == "ok" and before[1]["action"] == "dayahead-optim"
    assert after == before
    assert [r["P_grid"] for r in answer["plan"]] != live["P_grid"].tolist()


# --------------------------------------------------------------------------
# A request's outcome is its own: concurrent requests cannot mix them up
# --------------------------------------------------------------------------


def _plan() -> pd.DataFrame:
    index = pd.date_range("2026-10-01", periods=2, freq="30min", tz="UTC")
    return pd.DataFrame({"P_grid": [100.0, -50.0], "optim_status": "Optimal"}, index=index)


def _overlapping(dry_wrapper, live_wrapper):
    """POST a dry run of naive-mpc-optim and, while it is in flight, a live
    dayahead-optim. Returns ((status, body) of the dry run, status of the
    live run, the action log)."""
    dry_idd = {"params": {"passed_data": {"dry_run": True}}, "retrieve_hass_conf": {}}
    live_idd = {"params": {"passed_data": {}}, "retrieve_hass_conf": {}}

    async def build(*args, dry_run=False, **kwargs):
        return dict(dry_idd) if dry_run else dict(live_idd)

    with (
        tempfile.TemporaryDirectory() as tmp,
        _web(pathlib.Path(tmp)) as client,
        patch.object(web_server, "_load_params_and_runtime", _loaded({})),
        patch.object(web_server, "set_input_data_dict", build),
        patch.object(web_server, "naive_mpc_optim", dry_wrapper),
        patch.object(web_server, "dayahead_forecast_optim", live_wrapper),
        patch.object(web_server, "get_injection_dict", MagicMock(return_value={})),
        patch.object(web_server, "_save_injection_dict", AsyncMock()),
    ):

        async def both():
            dry = asyncio.create_task(client.post("/api/v1/dry-run/naive-mpc-optim", json={}))
            live = asyncio.create_task(client.post("/action/dayahead-optim", json={}))
            dry_response, live_response = await asyncio.gather(dry, live)
            return (
                (dry_response.status_code, await dry_response.get_json()),
                live_response.status_code,
            )

        dry_result, live_status = asyncio.run(both())
        action_log = (pathlib.Path(tmp) / web_server.action_log_str).read_text()
    return dry_result, live_status, action_log


def test_a_failing_live_request_does_not_fail_a_dry_run():
    """A dry run that succeeds while a live request fails beside it gets 200
    and no errors: the live request's error is in the action log, and not in
    the dry run's response."""
    started = asyncio.Event()

    async def dry_wrapper(idd, log):
        started.set()
        await asyncio.sleep(0.05)  # the live request fails meanwhile
        idd["run_record"] = last_run.snapshot("naive-mpc-optim", {}, "Optimal", False, 0.0, "1.0")
        return _plan()

    async def live_wrapper(idd, log):
        await started.wait()
        log.error("the live solve failed")
        return False

    (dry_status, dry_body), live_status, action_log = _overlapping(dry_wrapper, live_wrapper)
    assert dry_status == 200 and dry_body["errors"] == [] and dry_body["status"] == "ok"
    _conforms(dry_body)
    assert live_status == 400
    assert "the live solve failed" in action_log


def test_a_failing_dry_run_does_not_fail_a_live_request():
    """A dry run that fails while a live request succeeds beside it: the live
    request gets 200, because the dry run's error never reaches the action
    log that the live request's status is read from; the dry run gets 400
    and its own error."""
    logged = asyncio.Event()

    async def dry_wrapper(idd, log):
        log.error("the trial inputs are infeasible")
        logged.set()
        return False

    async def live_wrapper(idd, log):
        await logged.wait()
        return _plan()

    (dry_status, dry_body), live_status, action_log = _overlapping(dry_wrapper, live_wrapper)
    assert live_status == 200
    assert dry_status == 400 and dry_body["errors"] == ["the trial inputs are infeasible"]
    _conforms(dry_body)
    assert "infeasible" not in action_log


def test_a_dry_run_that_raises_answers_500_with_its_error():
    async def dry_wrapper(idd, log):
        raise RuntimeError("solver crashed")

    async def live_wrapper(idd, log):
        return _plan()

    (dry_status, dry_body), live_status, _ = _overlapping(dry_wrapper, live_wrapper)
    assert dry_status == 500 and live_status == 200
    _conforms(dry_body)
    assert any("solver crashed" in e for e in dry_body["errors"])


# --------------------------------------------------------------------------
# The command line, the record, the schema
# --------------------------------------------------------------------------


def test_the_command_line_refuses_a_dry_run():
    """The command line has no dry run: asked for one, it refuses rather than
    run live."""
    argv = ["main", "--action", "dayahead-optim", "--runtimeparams", '{"dry_run": true}']
    with (
        patch("sys.argv", argv),
        patch.object(command_line, "set_input_data_dict", AsyncMock()) as build,
        pytest.raises(Exception, match="/api/v1/dry-run/"),
    ):
        asyncio.run(main())
    build.assert_not_called()


def test_a_snapshot_is_not_stored():
    """last_run.snapshot builds the record that record() stores, and stores
    nothing itself."""
    saved = last_run._cache
    try:
        last_run._cache = None
        snap = last_run.snapshot(
            "dayahead-optim", {"optim_solve": 0.1}, "Infeasible", True, 1.0, "1.0"
        )
        assert last_run._cache is None
    finally:
        last_run._cache = saved
    assert snap["status"] == "infeasible" and snap["action"] == "dayahead-optim"
    assert set(snap) == set(SCHEMA["allOf"][0]["else"]["required"]) | {"status", "action"}


def test_only_a_real_true_is_a_dry_run():
    """is_dry_run trusts nothing but True: a stray truthy value stays live."""
    for value in ("false", "yes", 1, [1]):
        assert not is_dry_run({"params": {"passed_data": {"dry_run": value}}})
    assert is_dry_run({"params": orjson.dumps({"passed_data": {"dry_run": True}}).decode()})
    assert not is_dry_run({"params": {}})
