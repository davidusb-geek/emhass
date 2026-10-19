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
import sys
import tempfile
import time
import types
import uuid
from contextlib import contextmanager
from functools import partial
from unittest.mock import AsyncMock, MagicMock, patch

import aiofiles
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
from emhass.optimization import _BACKENDS, _coordinated_config_problem
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
    """A model older than adjusted_pv_model_max_age is refitted, live and dry
    alike: the fit runs once either way. A live run saves the new model; a dry
    run keeps it in memory and leaves the saved one for the next live run."""
    with _data_copy() as data_path:
        model = data_path / "adjust_pv_regressor.pkl"
        model.unlink(missing_ok=True)
        _run(data_path, "dayahead-optim", dry=False, configure=_adjusted_pv)
        assert model.exists(), "the live run should have fitted and saved a model"
        old = time.time() - 48 * 3600
        os.utime(model, (old, old))
        before = _files(data_path)
        fit = Forecast.adjust_pv_forecast_fit
        with patch.object(
            Forecast, "adjust_pv_forecast_fit", autospec=True, side_effect=fit
        ) as spy:
            idd, res = _run(data_path, "dayahead-optim", dry=dry, configure=_adjusted_pv)
        changed = _changed(before, _files(data_path))
    assert isinstance(res, pd.DataFrame)
    assert spy.await_count == 1, "the stale model must be refitted, not reused"
    if dry:
        assert changed == []
    else:
        assert "adjust_pv_regressor.pkl" in changed


# --------------------------------------------------------------------------
# Forecast services: a dry run calls none, and plans with what live runs saved
# --------------------------------------------------------------------------


class _Fetched(BaseException):
    """A forecast service was called. A BaseException, so that no fail-soft
    `except Exception` on the way can hide the call."""


class _NoService:
    """An aiohttp session that raises _Fetched the moment it is used."""

    def __init__(self, *args, **kwargs):
        pass

    def get(self, url, headers=None):
        raise _Fetched(url)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


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


def _open_meteo_payload(past_days: int = 1) -> dict:
    """An Open-Meteo answer in the unixtime format EMHASS asks for, covering
    `past_days` before now to four days after."""
    now = int(time.time()) // 900 * 900
    times = list(range(now - past_days * 86400, now + 4 * 86400, 900))
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


class _Records(logging.Handler):
    """Keeps the messages logged to it."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextmanager
def _log(level: int = logging.INFO):
    """A logger of its own at `level`, and the messages it gets."""
    log = logging.getLogger(f"{__name__}.{uuid.uuid4().hex}")
    log.setLevel(level)
    log.propagate = False
    records = _Records()
    log.addHandler(records)
    yield log, records.messages


async def _forecast(data_path: pathlib.Path, dry: bool, log: logging.Logger) -> Forecast:
    """A Forecast on `data_path`, for a dry run or a live one."""
    params = await _params(data_path)
    if dry:
        params["passed_data"]["dry_run"] = True
    params_json = orjson.dumps(params).decode()
    retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, log)
    return Forecast(retrieve_hass_conf, optim_conf, plant_conf, params_json, _conf(data_path), log)


def _no_saved_forecast(data_path: pathlib.Path) -> None:
    """Remove every saved weather forecast from `data_path`."""
    for name in (
        "weather_forecast_data.pkl",
        "cached-open-meteo-forecast-b.json",
        "cached-open-meteo-covariates.json",
    ):
        (data_path / name).unlink(missing_ok=True)


@contextmanager
def _writes_spied():
    """Every path forecast.py opens for writing."""
    real_open = aiofiles.open
    opened: list[str] = []

    def spy(path, mode="r", *args, **kwargs):
        if any(c in mode for c in "wax"):
            opened.append(str(path))
        return real_open(path, mode, *args, **kwargs)

    with patch("emhass.forecast.aiofiles.open", spy):
        yield opened


def test_a_live_run_refreshes_a_stale_open_meteo_forecast():
    """The control for the test below. With a stale Open-Meteo JSON cache, a
    stale weather cache and DEBUG logging, a live run deletes the stale
    cache, fetches once, saves the new JSON (through a temporary file, never
    by rewriting the cache in place) and writes its debug CSV."""
    calls: list = []
    with (
        _data_copy() as data_path,
        _log(logging.DEBUG) as (log, _),
        patch(
            "emhass.forecast.aiohttp.ClientSession",
            lambda *a, **k: _Session(_open_meteo_payload(), calls),
        ),
        _writes_spied() as opened,
    ):
        json_cache = _stale_open_meteo_caches(data_path)
        before = _files(data_path)

        async def fetch():
            return await (await _forecast(data_path, False, log)).get_weather_forecast(
                method="open-meteo"
            )

        weather = asyncio.run(fetch())
        changed = _changed(before, _files(data_path))
        leftovers = list(data_path.glob("*.tmp"))
    assert isinstance(weather, pd.DataFrame) and len(weather) > 0
    assert len(calls) == 1, calls
    assert {
        "cached-open-meteo-forecast-b.json",
        "weather_forecast_data.pkl",
        "debug-weather-forecast-open-meteo.csv",
    } <= set(changed), changed
    assert str(json_cache) not in opened and leftovers == []


def _stale_open_meteo_caches(data_path: pathlib.Path) -> pathlib.Path:
    """A two-hour-old Open-Meteo JSON cache, a weather cache that does not
    cover the horizon, and no debug CSVs. Returns the JSON cache's path."""
    _no_saved_forecast(data_path)
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
    return json_cache


def test_a_dry_run_plans_with_the_saved_open_meteo_forecast_however_old():
    """The same stale caches, on a dry run: no call to Open-Meteo, a plan
    from the saved forecast with a warning that says how old it is, and every
    file left as it was."""
    with (
        _data_copy() as data_path,
        _log(logging.DEBUG) as (log, messages),
        patch("emhass.forecast.aiohttp.ClientSession", _NoService),
    ):
        _stale_open_meteo_caches(data_path)
        before = _files(data_path)

        async def fetch():
            return await (await _forecast(data_path, True, log)).get_weather_forecast(
                method="open-meteo"
            )

        weather = asyncio.run(fetch())
        assert _changed(before, _files(data_path)) == []
    assert isinstance(weather, pd.DataFrame) and len(weather) > 0
    assert any("saved Open-Meteo forecast" in m and "minutes old" in m for m in messages)


def _solcast(fcst: Forecast) -> None:
    fcst.retrieve_hass_conf["solcast_api_key"] = "test-key"
    fcst.retrieve_hass_conf["solcast_rooftop_id"] = "test-roof"


@pytest.mark.parametrize("dry", [False, True], ids=["live", "dry"])
@pytest.mark.parametrize("method", ["open-meteo", "solcast", "solar.forecast"])
def test_with_no_saved_forecast_a_dry_run_calls_no_service(method, dry):
    """With nothing saved, a live run calls the forecast service (the
    control); a dry run calls none, spends no Solcast call budget, and plans as
    if the service were down: no forecast, and a warning that says what to do."""
    budget = MagicMock(return_value=True)
    with (
        _data_copy() as data_path,
        _log() as (log, messages),
        patch("emhass.forecast.aiohttp.ClientSession", _NoService),
        patch.object(Forecast, "_solcast_rate_limit_ok", budget),
    ):
        _no_saved_forecast(data_path)

        async def fetch():
            fcst = await _forecast(data_path, dry, log)
            _solcast(fcst)
            return await fcst.get_weather_forecast(method=method)

        if dry:
            weather = asyncio.run(fetch())
        else:
            with pytest.raises(_Fetched):
                asyncio.run(fetch())
    if dry:
        assert weather is None or weather is False
        assert not budget.called
        provider = {"open-meteo": "Open-Meteo", "solcast": "Solcast"}.get(method, method)
        assert any(f"no usable {provider} forecast" in m for m in messages), messages


@pytest.mark.parametrize("saved", [False, True], ids=["nothing-saved", "stale-saved"])
@pytest.mark.parametrize("dry", [False, True], ids=["live", "dry"])
def test_the_covariates_are_fetched_only_live(dry, saved):
    """The ML forecaster's Open-Meteo covariates: a live run fetches them when
    the saved ones are stale or missing, and saves them; a dry run uses the
    saved ones however old, or has none, and calls nothing either way."""
    calls: list = []
    session = _NoService if dry else (lambda *a, **k: _Session({"minutely_15": {}}, calls))
    with (
        _data_copy() as data_path,
        _log() as (log, _),
        patch("emhass.forecast.aiohttp.ClientSession", session),
    ):
        _no_saved_forecast(data_path)
        cache = data_path / "cached-open-meteo-covariates.json"
        if saved:
            cache.write_bytes(orjson.dumps(_open_meteo_payload(past_days=3)))
            old = time.time() - 2 * 3600
            os.utime(cache, (old, old))
        before = _files(data_path)

        async def fetch():
            return await (await _forecast(data_path, dry, log))._fetch_open_meteo_covariates_json(
                1, 1
            )

        data = asyncio.run(fetch())
        changed = _changed(before, _files(data_path))
    if dry:
        assert (data is not None) is saved
        assert changed == []
    else:
        assert len(calls) == 1 and changed == ["cached-open-meteo-covariates.json"]


def test_a_dry_run_does_not_save_the_weather_cache():
    """set_cached_forecast_data saves nothing on a dry run."""
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


def test_a_live_run_replaces_the_weather_cache_whole():
    """A live run writes the weather cache to a temporary file and moves it
    into place, so a dry run reading it at the same moment sees the old cache
    or the new one, never a partly written one."""
    with _data_copy() as data_path, _writes_spied() as opened:
        cache = data_path / "weather_forecast_data.pkl"
        cache.write_bytes(pickle.dumps(pd.DataFrame({"yhat": [0.0]})))

        async def save():
            fcst = await _forecast(data_path, False, logger)
            data = pd.DataFrame({"yhat": 1.0}, index=fcst.forecast_dates)
            await fcst.set_cached_forecast_data(str(cache), data)
            return data

        data = asyncio.run(save())
        saved = pickle.loads(cache.read_bytes())
        leftovers = list(data_path.glob("*.tmp"))
    assert opened and str(cache) not in opened
    pd.testing.assert_frame_equal(saved, data)
    assert leftovers == []


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
        ("naive-mpc-optim", "null"),
        ("naive-mpc-optim", '"text"'),
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


def test_an_empty_body_is_an_empty_request():
    """No body at all is a request with no runtime parameters, as on
    /action/<action>; it plans, and logs no error about the body."""
    built = AsyncMock(return_value={"params": {"passed_data": {"dry_run": True}}})

    async def wrapper(idd, log):
        idd["run_record"] = last_run.snapshot("naive-mpc-optim", {}, "Optimal", False, 0.0, "1.0")
        return _plan()

    with (
        tempfile.TemporaryDirectory() as tmp,
        _web(pathlib.Path(tmp)) as client,
        patch.object(web_server, "_load_params", AsyncMock(return_value=("{}", "profit"))),
        patch.object(web_server, "set_input_data_dict", built),
        patch.object(web_server, "naive_mpc_optim", wrapper),
    ):

        async def post():
            response = await client.post("/api/v1/dry-run/naive-mpc-optim")
            return response.status_code, await response.get_json()

        status, answer = asyncio.run(post())
    assert status == 200 and answer["errors"] == []
    assert built.call_args.args[3] == "{}"


def test_a_dry_run_with_no_saved_forecast_is_refused():
    """End to end: a body without pv_power_forecast, and no forecast saved by a
    live run. The dry run calls no service, and is refused (400): the warning
    says what to pass, the error that the inputs could not be built."""
    with _data_copy() as data_path:
        _no_saved_forecast(data_path)
        _write_params(data_path)
        body = {k: v for k, v in FORECASTS.items() if k != "pv_power_forecast"}
        with (
            _web(data_path) as client,
            _from_file(),
            patch("emhass.forecast.aiohttp.ClientSession", _NoService),
        ):

            async def post():
                response = await client.post("/api/v1/dry-run/dayahead-optim", json=body)
                return response.status_code, await response.get_json()

            status, answer = asyncio.run(post())
    assert status == 400
    _conforms(answer)
    assert any("no usable Open-Meteo forecast" in w for w in answer["warnings"])
    assert any("inputs could not be built" in e for e in answer["errors"])


def test_a_dry_run_flag_left_in_saved_params_does_not_make_a_dry_run():
    """Only the dry-run URL makes a dry run. A dry_run left in the saved
    parameters (or the command line's --params) is dropped, so the live run
    still writes its results."""

    def stray(params: dict) -> None:
        params["passed_data"]["dry_run"] = True

    with _data_copy() as data_path:
        before = _files(data_path)
        idd, res = _run(data_path, "dayahead-optim", dry=False, configure=stray)
        changed = _changed(before, _files(data_path))
    assert isinstance(res, pd.DataFrame) and not is_dry_run(idd)
    assert "last_run.json" in changed and "opt_res_latest.csv" in changed, changed


@pytest.mark.parametrize("failure", ["home-assistant", "inputs"])
def test_a_failed_setup_closes_the_home_assistant_session(failure):
    """When the inputs cannot be built - Home Assistant's configuration is
    unreadable, or a forecast is missing - set_input_data_dict closes the
    session it opened before answering False."""
    with (
        _data_copy() as data_path,
        patch.object(command_line.RetrieveHass, "close", AsyncMock()) as close,
        patch.object(command_line.RetrieveHass, "get_ha_config", AsyncMock(return_value=False)),
        patch.object(command_line, "_prepare_dayahead_optim", AsyncMock(return_value=None)),
    ):

        async def build():
            params = await _params(data_path)
            return await set_input_data_dict(
                _conf(data_path),
                "profit",
                orjson.dumps(params).decode(),
                orjson.dumps(FORECASTS).decode(),
                "dayahead-optim",
                logger,
                get_data_from_file=failure == "inputs",
                dry_run=True,
            )

        assert asyncio.run(build()) is False
    close.assert_awaited_once()


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
        patch.object(web_server, "_load_params", AsyncMock(return_value=("{}", "profit"))),
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


# --------------------------------------------------------------------------
# Whatever the coordinator does, a plan is made: live and dry alike
# --------------------------------------------------------------------------


def _coordinated(data_path: pathlib.Path, runtime: dict, site=None, dry: bool = False):
    """Day-ahead with the coordinated backend switched on (and `site` if
    given), live or dry; returns the plan."""

    async def run():
        """Build the inputs with the backend on, and solve."""
        ec = _conf(data_path)
        _, secrets = await build_secrets(ec, logger, no_response=True)
        params = await build_params(
            ec, secrets, await build_config(ec, logger, ec["defaults_path"]), logger
        )
        params["optim_conf"]["optimization_backend"] = "dantzig_wolfe"
        params["optim_conf"]["site"] = site
        idd = await set_input_data_dict(
            ec,
            "profit",
            orjson.dumps(params).decode(),
            orjson.dumps({**FORECASTS, **runtime}).decode(),
            "dayahead-optim",
            logger,
            get_data_from_file=True,
            dry_run=dry,
        )
        return await dayahead_forecast_optim(idd, logger, debug=True)

    return asyncio.run(run())


@pytest.mark.parametrize("dry", [False, True], ids=["live", "dry"])
def test_a_failing_coordinator_falls_back_to_the_default_solver(dry, caplog):
    """The coordinator raising - for any reason - costs the coordination, not
    the plan: the default solver plans instead, and says why."""
    boom = types.ModuleType("home_energy_optimizer.integrations.emhass")

    def optimize(*_a, **_k):
        """A coordinator that fails."""
        raise RuntimeError("coordinator exploded")

    boom.optimize = optimize
    OptimizationCache.clear()
    with (
        tempfile.TemporaryDirectory() as tmp,
        patch.dict(sys.modules, {"home_energy_optimizer.integrations.emhass": boom}),
        caplog.at_level("ERROR"),
    ):
        data_path = pathlib.Path(tmp) / "data"
        shutil.copytree(root / "data", data_path)
        res = _coordinated(data_path, {}, dry=dry)
    assert isinstance(res, pd.DataFrame) and len(res) > 0
    assert (res["optim_status"] == "Optimal").all()
    assert "coordinator raised RuntimeError" in caplog.text
    # ...and the plan says which solver made it, and why
    assert (res["backend_used"] == "cvxpy").all()
    assert res["backend_fallback_reason"].str.contains("coordinator exploded").all()
    OptimizationCache.clear()


SITE = [
    {"id": "grid", "max_import": 9000, "max_export": 5000},
    {
        "id": "inverter",
        "parent": "grid",
        "type": "hybrid_inverter",
        "max_import": 4000,
        "max_export": 4000,
        "efficiency_import": 0.97,
        "efficiency_export": 0.97,
    },
    {"id": "garage", "parent": "grid", "type": "panel", "max_import": 7400, "max_export": 0},
    {"id": "heat", "type": "breaker", "parent": "garage", "max_import": 3500},
    {"id": "l1", "type": "limit", "max_import": 5000},
    {"id": "pv", "parent": "inverter"},
    {"id": "battery", "parent": "inverter", "solver": "home_energy_optimizer", "limits": ["l1"]},
    {"id": "deferrable0", "parent": "garage", "group": "loads"},
    {"id": "deferrable1", "parent": "garage", "group": "loads"},
    {
        "id": "water_heater",
        "parent": "heat",
        "solver": "home_energy_optimizer",
        "config": {"power_kw": 3.0, "n_duty_levels": 2, "comfort_mode": "linear"},
    },
    {"id": "hvac", "parent": "heat", "solver": "home_energy_optimizer", "limits": ["l1"]},
]
REMOTE = {
    "id": "garage_emhass",
    "parent": "garage",
    "solver": {"url": "http://emhass-garage:5000/participant", "token_secret": "garage_token"},
}


def _with(index: int, **change):
    """SITE with element `index` changed (a value of None removes the key)."""
    element = {k: v for k, v in {**SITE[index], **change}.items() if v is not None}
    return SITE[:index] + [element] + SITE[index + 1 :]


@pytest.mark.parametrize(
    "site, problem",
    [
        ({"nodes": []}, "must be a list"),
        ([{"id": f"n{i}"} for i in range(129)], "at most 128 elements"),
        (["inverter"], "must be an object with an id"),
        ([{"type": "panel"}], "must be an object with an id"),
        (_with(1, id="Inverter!"), "short lower-case name"),
        (_with(1, id="__import__('os')"), "short lower-case name"),
        (_with(1, rogue=1), "may have only"),
        (_with(0, parent="inverter"), "may have only"),
        (_with(4, parent="grid"), "may have only"),
        (_with(5, solver="emhass"), "planned by no solver"),
        (_with(6, solver="os.system"), "solver must be one of"),
        (_with(6, group="Loads!"), "group must be a short lower-case name"),
        (_with(6, limits="l1"), "limits must be a list of limit ids"),
        (_with(6, limits=["nope"]), "which is no limit"),
        (
            _with(9, config={"power_kw": {"__class__": 1}}),
            "finite number, a boolean or a short string",
        ),
        (_with(9, config={"t_comfort": "x" * 65}), "finite number, a boolean or a short string"),
        (_with(10, config={"cop": float("nan")}), "finite number, a boolean or a short string"),
        (_with(1, type="reactor"), "type must be one of"),
        (_with(1, max_import=-1), "max_import must be a finite number of W, >= 0"),
        (_with(1, max_export=float("inf")), "max_export must be a finite number of W, >= 0"),
        (_with(1, max_export=True), "max_export must be a finite number of W, >= 0"),
        (_with(1, efficiency_export=1.2), "efficiency_export must be in (0, 1]"),
        (_with(1, efficiency_import=0), "efficiency_import must be in (0, 1]"),
        (_with(4, max_import=None), "needs max_import, max_export or both"),
        (_with(3, parent="nowhere"), "parent must be 'grid' or a node's id"),
        (_with(1, parent=None), "'inverter' needs a parent"),
        (_with(7, parent=None), "'deferrable0' needs a parent"),
        (_with(7, parent="l1"), "parent must be 'grid' or a node's id"),
        (SITE + [{"id": "garage", "parent": "grid", "type": "breaker"}], "appears twice"),
        ([{"id": "a", "parent": "b"}, {"id": "b", "parent": "a"}], "form a loop"),
        (
            [{"id": "x", "parent": "grid", "solver": {"url": "file:///etc/passwd"}}],
            "solver must be {url",
        ),
        (
            [{"id": "x", "parent": "grid", "solver": {"url": "http://h/", "rogue": 1}}],
            "solver must be {url",
        ),
        ([{**REMOTE, "solver": {**REMOTE["solver"], "token_secret": "A!"}}], "must name a secret"),
    ],
)
def test_the_site_is_checked_before_anything_reads_it(site, problem):
    """`site` can arrive with a request, so its shape and types are checked
    first; anything unexpected means the default solver, never an exception or
    an unknown name reaching the coordinator."""
    reason = _coordinated_config_problem({"optimization_backend": "dantzig_wolfe", "site": site})
    assert reason and problem in reason


def test_a_known_backend_and_a_valid_site_pass():
    assert _coordinated_config_problem({"optimization_backend": "dantzig_wolfe"}) is None
    for site in (SITE, SITE + [REMOTE], []):
        assert (
            _coordinated_config_problem({"optimization_backend": "dantzig_wolfe", "site": site})
            is None
        )
    assert "unknown backend" in _coordinated_config_problem({"optimization_backend": "rogue"})
    # documented as future work, not offered: it says so and the default solver plans
    assert "later release" in _coordinated_config_problem({"optimization_backend": "admm"})


def test_a_remote_solver_comes_from_the_configuration_only():
    """A site sent with a request may not name a remote solver: no request
    makes EMHASS call a host of its choosing."""
    assert utils.site_problem(SITE + [REMOTE]) is None
    reason = utils.site_problem(SITE + [REMOTE], from_request=True)
    assert reason and "from the configuration only" in reason


def test_the_published_schema_is_the_check():
    """openapi.json describes site as the check reads it - one allowed key
    set per kind of element, these device names and solvers - and offers only
    the backends EMHASS runs."""
    spec = json.loads((root / "src/emhass/static/openapi.json").read_text())
    props = spec["components"]["schemas"]["Config"]["properties"]
    assert props["optimization_backend"]["enum"] == list(_BACKENDS)
    assert props["site"]["type"] == ["array", "null"]
    kinds = {k["title"].split(" (")[0]: k for k in props["site"]["items"]["anyOf"]}
    names = {
        "the main meter": "grid",
        "a node": "node",
        "a limit": "limit",
        "a device": "device",
        "a remote solver": "remote",
    }
    assert set(kinds) == set(names)
    for title, kind in names.items():
        assert kinds[title]["additionalProperties"] is False
        assert set(kinds[title]["properties"]) == utils._SITE_KEYS[kind]
    device = kinds["a device"]["properties"]
    assert device["id"]["pattern"] == utils._SITE_DEVICE.pattern
    assert device["solver"]["enum"] == list(utils._SITE_SOLVERS)
    assert kinds["a node"]["properties"]["type"]["enum"] == list(utils._SITE_NODE_TYPES)
    assert kinds["a node"]["properties"]["id"]["pattern"] == utils._SITE_ID.pattern
    for title in ("a node", "a device", "a remote solver"):
        assert "parent" in kinds[title]["required"]


def test_the_sites_meter_and_inverter_set_emhass_keys(caplog):
    """The main meter's limits, and a hybrid_inverter node on it holding the PV
    and the battery, fill EMHASS's own keys - so the default solver plans the
    same meter and inverter; where the keys said otherwise, site wins, with a
    warning."""
    plant = {
        "inverter_is_hybrid": False,
        "inverter_ac_output_max": 5000,
        "maximum_power_from_grid": 9000,
    }
    with caplog.at_level("WARNING"):
        assert utils.compile_site(SITE, plant, logger)
    assert plant == {
        "inverter_is_hybrid": True,
        "inverter_ac_output_max": 4000,
        "inverter_ac_input_max": 4000,
        "inverter_efficiency_dc_ac": 0.97,
        "inverter_efficiency_ac_dc": 0.97,
        "maximum_power_from_grid": 9000,
        "maximum_power_to_grid": 5000,
    }
    assert "replaces inverter_ac_output_max=5000" in caplog.text


@pytest.mark.parametrize(
    "change",
    [
        {"parent": "garage"},  # behind a panel
        {"type": "inverter"},  # not a hybrid one
    ],
)
def test_an_inverter_emhass_cannot_hold_leaves_its_keys(change, caplog):
    plant = {"inverter_is_hybrid": False}
    with caplog.at_level("WARNING"):
        assert not utils.compile_site(_with(1, **change)[1:], plant, logger)
    assert plant == {"inverter_is_hybrid": False}


def test_a_second_device_on_the_inverter_leaves_its_keys():
    site = SITE + [{"id": "deferrable2", "parent": "inverter"}]
    assert utils.site_inverter(site) is None


def test_the_input_pipeline_compiles_the_site():
    """Set in the configuration (or a request), the site's inverter and meter
    reach the default solver's plant configuration; a request's site that
    names a remote solver is not used."""

    async def run(runtime: dict):
        with tempfile.TemporaryDirectory() as tmp:
            data_path = pathlib.Path(tmp) / "data"
            shutil.copytree(root / "data", data_path)
            ec = _conf(data_path)
            _, secrets = await build_secrets(ec, logger, no_response=True)
            params = await build_params(
                ec, secrets, await build_config(ec, logger, ec["defaults_path"]), logger
            )
            params["optim_conf"]["site"] = SITE
            idd = await set_input_data_dict(
                ec,
                "profit",
                orjson.dumps(params).decode(),
                orjson.dumps({**FORECASTS, **runtime}).decode(),
                "dayahead-optim",
                logger,
                get_data_from_file=True,
            )
            return idd["opt"].plant_conf, idd["opt"].optim_conf

    plant, _ = asyncio.run(run({}))
    assert plant["inverter_is_hybrid"] is True and plant["inverter_ac_output_max"] == 4000
    assert plant["maximum_power_to_grid"] == 5000
    _, optim = asyncio.run(run({"site": [REMOTE]}))
    assert optim["site"] == SITE  # the configured one, kept


@pytest.mark.parametrize("dry", [False, True], ids=["live", "dry"])
def test_a_malformed_site_in_a_request_still_plans(dry, caplog):
    """The default solver plans, live or dry, and the log says why."""
    OptimizationCache.clear()
    with tempfile.TemporaryDirectory() as tmp, caplog.at_level("WARNING"):
        data_path = pathlib.Path(tmp) / "data"
        shutil.copytree(root / "data", data_path)
        res = _coordinated(data_path, {}, site=[{"nodevices": 1}], dry=dry)
    assert isinstance(res, pd.DataFrame) and (res["optim_status"] == "Optimal").all()
    assert (res["backend_used"] == "cvxpy").all()
    # refused by the check, by name - the coordinator never saw it
    assert "site[0] must be an object with an id" in caplog.text
    assert "coordinator raised" not in caplog.text
    OptimizationCache.clear()
