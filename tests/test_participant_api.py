"""This EMHASS's devices served to a coordinator elsewhere (participant_api),
and the coordinator's side: a remote solver named in `site`.

The remote side is served by home-energy-optimizer's ParticipantService on a
local port (remote.serve), building its model with EMHASS's own input path;
the coordinator is another EMHASS's optimize, as a live run calls it."""

import asyncio
import contextlib
import copy
import functools
import importlib.util
import io
import json
import pathlib
import pickle
import shutil
import sys
import tempfile
import types
from unittest import mock

import numpy as np
import orjson
import pandas as pd
import pytest

from emhass import last_run, participant_api, plan_store, utils, web_server
from emhass.optimization import Optimization
from emhass.utils import (
    build_config,
    build_params,
    build_secrets,
    get_logger,
    get_root,
    get_yaml_parse,
)

root = pathlib.Path(get_root(__file__, num_parent=2))
logger, _ = get_logger(__name__, {"data_path": root / "data/"}, save_to_file=False)
EMHASS_CONF = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "defaults_path": root / "src/emhass/data/config_defaults.json",
    "associations_path": root / "src/emhass/data/associations.csv",
}


def _confs():
    """EMHASS's default configuration, as at startup: (retrieve_hass_conf,
    optim_conf, plant_conf)."""
    params = _params(EMHASS_CONF)
    return get_yaml_parse(orjson.dumps(params).decode("utf-8"), logger)


def _hep_at_least(*version: int) -> bool:
    """Whether home-energy-optimizer is installed, at `version` or later."""
    if importlib.util.find_spec("home_energy_optimizer") is None:
        return False
    from home_energy_optimizer import __version__

    return tuple(int(x) for x in __version__.split(".")[:3]) >= version


REMOTE_READY = pytest.mark.skipif(
    not _hep_at_least(0, 2, 9), reason="needs home-energy-optimizer >= 0.2.9 (remote solvers)"
)


def _conf(data_path: pathlib.Path) -> dict:
    return {
        "data_path": data_path,
        "root_path": root / "src/emhass/",
        "defaults_path": root / "src/emhass/data/config_defaults.json",
        "associations_path": root / "src/emhass/data/associations.csv",
    }


def _params(ec: dict, **optim) -> dict:
    """EMHASS's default params, with `optim` in optim_conf."""

    async def build():
        _, secrets = await build_secrets(ec, logger, no_response=True)
        return await build_params(
            ec, secrets, await build_config(ec, logger, ec["defaults_path"]), logger
        )

    params = asyncio.run(build())
    params["optim_conf"].update(optim)
    return params


GARAGE = {
    # the remote: an EV charger, 7 kW for 2 h, nothing else
    "set_use_battery": False,
    "set_use_pv": False,
    "number_of_deferrable_loads": 1,
    "nominal_power_of_deferrable_loads": [7000],
    "operating_hours_of_each_deferrable_load": [2],
    "treat_deferrable_load_as_semi_cont": [True],
    "set_deferrable_load_single_constant": [False],
    "start_timesteps_of_each_deferrable_load": [0],
    "end_timesteps_of_each_deferrable_load": [0],
    "minimum_power_of_deferrable_loads": [0],
    "set_deferrable_startup_penalty": [0],
    "set_deferrable_max_startups": [0],
    "deferrable_load_max_cost": [0],
    "def_minimum_on_time": [0],
    "def_minimum_off_time": [0],
    "participant_api": {"publish_prefix": "garage_", "key": "garage_emhass"},
}


@pytest.fixture
def data_path():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "data"
        shutil.copytree(root / "data", path)
        yield path


def _client_get(data_path: pathlib.Path, params: dict | None, path: str, token: str | None = None):
    """GET `path` from the web app, with `params` as its params.pkl."""

    async def get():
        saved_conf, saved_secret = (
            web_server.emhass_conf,
            web_server.params_secrets.get("participant_token"),
        )
        web_server.emhass_conf = {**saved_conf, "data_path": data_path}
        if params is not None:
            (data_path / "params.pkl").write_bytes(pickle.dumps((None, params)))
        if token is None:
            web_server.params_secrets.pop("participant_token", None)
        else:
            web_server.params_secrets["participant_token"] = token
        try:
            client = web_server.app.test_client()
            r = await client.get(path, headers={"Authorization": "Bearer wrong"})
            return r.status_code, await r.get_json()
        finally:
            web_server.emhass_conf = saved_conf
            if saved_secret is None:
                web_server.params_secrets.pop("participant_token", None)
            else:
                web_server.params_secrets["participant_token"] = saved_secret

    return asyncio.run(get())


def test_the_participant_api_is_off_unless_set(data_path):
    params = _params(_conf(data_path))
    status, body = _client_get(data_path, params, "/participant/v1/describe", token="t")
    assert status == 404 and "participant_api" in body["detail"]


def test_the_participant_api_needs_a_token(data_path):
    params = _params(_conf(data_path), **GARAGE)
    status, body = _client_get(data_path, params, "/participant/v1/describe")
    assert status == 503 and "participant_token" in body["detail"]


def test_the_participant_api_needs_one_web_worker(data_path, monkeypatch):
    """A coordinator's answers live in one worker's memory: with more, its
    commit could reach a worker that never gave the answer it names."""
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    params = _params(_conf(data_path), **GARAGE)
    status, body = _client_get(data_path, params, "/participant/v1/describe", token="t")
    assert status == 503 and "one web worker" in body["detail"]


def test_a_cancelled_commit_keeps_the_slot_until_it_ends(data_path):
    """The slot is held in the commit's own thread: a request cancelled while
    its commit runs does not free it for a live optimization under it."""
    import threading

    saved_conf = web_server.emhass_conf
    web_server.emhass_conf = {**saved_conf, "data_path": data_path}
    started, release = threading.Event(), threading.Event()

    def commit():
        started.set()
        assert release.wait(10)
        return 200, {"accepted": True}

    def slot_free() -> bool:
        with open(web_server._live_optim_slot(), "a+") as f:
            if utils.try_lock_file(f):
                utils.unlock_file(f)
                return True
            return False

    async def cancel_mid_commit():
        task = asyncio.ensure_future(asyncio.to_thread(web_server._in_live_optim_slot, commit))
        assert await asyncio.to_thread(started.wait, 10)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        held = not slot_free()
        release.set()
        for _ in range(100):
            if slot_free():
                return held, True
            await asyncio.sleep(0.05)
        return held, False

    try:
        held, freed = asyncio.run(cancel_mid_commit())
    finally:
        release.set()
        web_server.emhass_conf = saved_conf
    assert held, "the slot was freed while the commit still ran"
    assert freed, "the slot was not freed once the commit ended"


@pytest.mark.parametrize("busy", [False, True], ids=["free", "live-run"])
def test_a_commit_waits_for_no_live_optimization(data_path, busy):
    """A commit ends a live run - it writes the latest results and publishes -
    so while this EMHASS runs a live optimization of its own (it holds the
    live-optimization slot) a commit is refused with 429 and does nothing."""
    params = _params(_conf(data_path), **GARAGE)
    service = mock.MagicMock()
    service.handle.return_value = (200, {"accepted": True})

    async def post():
        saved_conf = web_server.emhass_conf
        saved_secret = web_server.params_secrets.get("participant_token")
        web_server.emhass_conf = {**saved_conf, "data_path": data_path}
        web_server.params_secrets["participant_token"] = "t"
        (data_path / "params.pkl").write_bytes(pickle.dumps((None, params)))
        try:
            with open(web_server._live_optim_slot(), "a+") as live:
                if busy:
                    assert utils.try_lock_file(live)
                try:
                    r = await web_server.app.test_client().post(
                        "/participant/v1/commit", data=b"{}", headers={"Authorization": "Bearer t"}
                    )
                finally:
                    if busy:
                        utils.unlock_file(live)
            return r.status_code, r.headers.get("Retry-After")
        finally:
            web_server.emhass_conf = saved_conf
            if saved_secret is None:
                web_server.params_secrets.pop("participant_token", None)
            else:
                web_server.params_secrets["participant_token"] = saved_secret

    # The route needs the federated extra only to build a service, mocked here,
    # so the slot is tested with or without the package installed.
    package = types.ModuleType("home_energy_optimizer")
    package.remote = types.ModuleType("home_energy_optimizer.remote")
    with (
        mock.patch.object(participant_api, "make_service", return_value=service),
        mock.patch.object(web_server, "_participant_service", None),
        mock.patch.dict(
            sys.modules,
            {"home_energy_optimizer": package, "home_energy_optimizer.remote": package.remote},
        ),
    ):
        status, retry = asyncio.run(post())
    if busy:
        assert status == 429 and retry == "2"
        service.handle.assert_not_called()
    else:
        assert status == 200
        service.handle.assert_called_once()


@REMOTE_READY
def test_a_wrong_token_is_refused(data_path):
    params = _params(_conf(data_path), **GARAGE)
    status, _ = _client_get(data_path, params, "/participant/v1/describe", token="right")
    assert status == 401


@REMOTE_READY
def test_a_session_answers_for_a_horizon_and_refuses_another_slot_length(data_path):
    """Built through EMHASS's own input path, for the horizon asked about: its
    EV at a price runs its 2 h, the same query gives the same answer, and
    another slot length is refused."""
    from home_energy_optimizer.interface import Query
    from home_energy_optimizer.remote import PlanWindow, WindowRefused

    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    p = asyncio.run(
        participant_api.build_session(
            ec, params, PlanWindow(30, 48), logger, get_data_from_file=True
        )
    )
    price = np.linspace(0.1, 0.3, 48)
    a = p.respond(Query("price_response", price, price))
    assert a.status == "ok" and a.plan_kw.sum() * 0.5 == pytest.approx(14.0)
    assert np.argmax(a.plan_kw) < 4  # where it is cheapest
    p._last = None
    assert np.array_equal(p.respond(Query("price_response", price, price)).plan_kw, a.plan_kw)
    with pytest.raises(WindowRefused, match="30-minute slots"):
        asyncio.run(
            participant_api.build_session(
                ec, params, PlanWindow(15, 96), logger, get_data_from_file=True
            )
        )


def _house(start: pd.Timestamp):
    """The coordinator's house: a battery, two loads, 5 kW of PV, a tariff,
    on a day from `start` (the remote plans the same slots)."""
    rh, oc, pc = _confs()
    oc.update(set_use_battery=True, set_use_pv=True, operating_hours_of_each_deferrable_load=[4, 2])
    pc.update(
        battery_nominal_energy_capacity=10000,
        battery_charge_power_max=5000,
        battery_discharge_power_max=5000,
    )
    n = 48
    index = pd.date_range(start, periods=n, freq=rh["optimization_time_step"])
    h = np.arange(n) / 2
    buy = 0.20 + 0.10 * np.sin((h - 13) / 24 * 2 * np.pi)
    sell = np.full(n, 0.06)
    pv = np.clip(5000 * np.sin((h - 6) / 12 * np.pi), 0, None)
    load = 400 + 1200 * np.exp(-0.5 * ((h - 19.5) / 1.8) ** 2)
    data = pd.DataFrame({"unit_load_cost": buy, "unit_prod_price": sell}, index=index)
    return rh, oc, pc, data, pv, load, buy, sell


def _now() -> pd.Timestamp:
    """The first slot EMHASS plans now, as EMHASS rounds it (to the nearest
    slot, by default): the remote plans from its own clock, and refuses a
    horizon that starts on another slot."""
    rh = _confs()[0]
    step = rh["optimization_time_step"]
    return utils.get_forecast_start(
        pd.Timestamp.now(tz="UTC"),
        step if isinstance(step, pd.Timedelta) else pd.Timedelta(minutes=float(step)),
        rh["time_zone"],
        rh.get("method_ts_round", "nearest"),
    )


def _coordinate(house, site, dry_run=False):
    rh, oc, pc, data, pv, load, buy, sell = house
    oc = copy.deepcopy(oc)
    oc.update(optimization_backend="dantzig_wolfe", site=site)
    opt = Optimization(
        rh,
        oc,
        copy.deepcopy(pc),
        "unit_load_cost",
        "unit_prod_price",
        "profit",
        EMHASS_CONF,
        logger,
    )
    opt.fed_dry_run = dry_run
    return opt.perform_optimization(data, pv, load, buy, sell, soc_init=0.5, soc_final=0.5)


@REMOTE_READY
def test_a_remote_emhass_is_planned_with_the_house_and_runs_its_part(data_path):
    """A second EMHASS (an EV charger), served over HTTP, is one participant of
    this EMHASS's plan: its 2 h run, its power and share in the plan; on a
    live run it is asked to run the plan chosen, on a dry run it is not."""
    from home_energy_optimizer.integrations.emhass import emhass_description
    from home_energy_optimizer.remote import ParticipantService, serve

    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    committed = []

    def build(window):
        return asyncio.run(
            participant_api.build_session(ec, params, window, logger, get_data_from_file=True)
        )

    def describe():
        from home_energy_optimizer.remote import PlanWindow

        return emhass_description(build(PlanWindow(30, 48)))

    server = serve(
        ParticipantService(describe, build, lambda p, a: (committed.append(a), (True, ""))[1])
    )
    try:
        house = _house(_now())
        site = [{"id": "garage_emhass", "parent": "grid", "solver": {"url": server.url}}]
        res = _coordinate(house, site)
        assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[
            0
        ]
        ev = res["fed_remote_power_garage_emhass"]
        assert ev.sum() * 0.5 == pytest.approx(14000.0)  # 7 kW for 2 h, in W
        assert (res["fed_remote_status_garage_emhass"] == "committed").all()
        assert len(committed) == 1 and np.allclose(committed[0].plan_kw * 1000.0, ev.values)
        assert "fed_share_garage_emhass" in res.columns
        # the meter carries the EV too
        balance = res["P_PV"] + res["P_grid"] + res["P_batt"]
        used = res["P_Load"] + res["P_deferrable0"] + res["P_deferrable1"] + ev
        assert np.allclose(balance, used, atol=1e-3)
        res = _coordinate(house, site, dry_run=True)
        assert (res["fed_remote_status_garage_emhass"] == "planned (dry run)").all()
        assert len(committed) == 1  # a dry run asks nothing
    finally:
        server.shutdown()


@REMOTE_READY
def test_a_remote_that_does_not_answer_is_planned_without(data_path):
    """A remote that is down costs its part of the coordination, not the plan:
    the house is planned without it, and the plan says so."""
    from home_energy_optimizer.remote import ParticipantService, serve

    server = serve(ParticipantService(lambda: {}, lambda w: None))
    url = server.url
    server.shutdown()
    server.server_close()
    res = _coordinate(
        _house(_now()), [{"id": "garage_emhass", "parent": "grid", "solver": {"url": url}}]
    )
    assert (res["backend_used"] == "dantzig_wolfe").all()
    assert (res["fed_remote_status_garage_emhass"] == "unreachable").all()
    assert "fed_remote_power_garage_emhass" not in res.columns


@REMOTE_READY
@pytest.mark.parametrize("soc", [0.15, 0.6, 0.95])
def test_a_remote_battery_behind_a_panel_holds_every_limit(data_path, monkeypatch, soc):
    """The remote's EV and a battery behind a garage panel (7.4 kW in, no
    backfeed), sharing a 7 kW limit with the house's water heater, from a
    nearly empty, a half and a full battery. The remote's own model holds its
    connection's limits (its maximum_power_from_grid / _to_grid), however big
    its EV; the coordinator holds what it shares with the water heater. No
    limit is breached."""
    from home_energy_optimizer.integrations.emhass import emhass_description
    from home_energy_optimizer.remote import ParticipantService, PlanWindow, serve

    ec = _conf(data_path)
    params = _params(ec, **{**GARAGE, "set_use_battery": True})
    params["plant_conf"].update(
        battery_nominal_energy_capacity=5000,
        battery_charge_power_max=3000,
        battery_discharge_power_max=3000,
        maximum_power_from_grid=7000,  # its connection: its share of the garage
        maximum_power_to_grid=0,  # and no backfeed
    )
    # its battery's state now; it ends the day at 50%
    monkeypatch.setattr(participant_api, "soc_now", lambda *_: soc)

    def build(window):
        return asyncio.run(
            participant_api.build_session(ec, params, window, logger, get_data_from_file=True)
        )

    server = serve(
        ParticipantService(
            lambda: emhass_description(build(PlanWindow(30, 48))), build, lambda p, a: (True, "")
        )
    )
    try:
        site = [
            {
                "id": "garage",
                "parent": "grid",
                "type": "panel",
                "max_import": 7400,
                "max_export": 0,
            },
            {"id": "l1", "type": "limit", "max_import": 7000},
            {
                "id": "water_heater",
                "parent": "garage",
                "solver": "home_energy_optimizer",
                "limits": ["l1"],
                "config": {"power_kw": 3.0, "n_duty_levels": 2},
            },
            {
                "id": "garage_emhass",
                "parent": "garage",
                "limits": ["l1"],
                "solver": {"url": server.url},
            },
        ]
        res = _coordinate(_house(_now()), site)
    finally:
        server.shutdown()
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    garage = res["fed_node_power_garage"]  # W, + = up the tree
    assert garage.max() <= 1.0 and garage.min() >= -7400 - 1.0, (garage.min(), garage.max())
    shared = res["P_water_heater"] + res["fed_remote_power_garage_emhass"]
    assert shared.max() <= 7000 + 1.0, shared.max()


SHED = {
    **GARAGE,
    "nominal_power_of_deferrable_loads": [1100],
    "operating_hours_of_each_deferrable_load": [6],
    "participant_api": {"publish_prefix": "shed_", "key": "shed"},
}


@REMOTE_READY
@pytest.mark.parametrize("soc", [0.15, 0.6, 0.95])
def test_two_remotes_and_a_tank_share_a_panel_within_its_rating(tmp_path, monkeypatch, soc):
    """Two remote EMHASS behind one garage panel (7.4 kW in, no backfeed) with
    the house's water heater: the garage's EV and battery (holding its own 7 kW
    connection), the shed's pool pump (1.1 kW for 6 h). Each is on/off, so the
    plan picks one of each one's plans; the panel's rating holds."""
    from home_energy_optimizer.integrations.emhass import emhass_description
    from home_energy_optimizer.remote import ParticipantService, PlanWindow, serve

    monkeypatch.setattr(participant_api, "soc_now", lambda *_: soc)
    servers = []
    for name, extra, plant in (
        (
            "garage",
            {**GARAGE, "set_use_battery": True},
            {
                "battery_nominal_energy_capacity": 5000,
                "battery_charge_power_max": 3000,
                "battery_discharge_power_max": 3000,
                "maximum_power_from_grid": 7000,
                "maximum_power_to_grid": 0,
            },
        ),
        ("shed", SHED, {}),
    ):
        path = tmp_path / name
        shutil.copytree(root / "data", path)
        ec = _conf(path)
        params = _params(ec, **extra)
        params["plant_conf"].update(plant)

        def build(window, ec=ec, params=params):
            return asyncio.run(
                participant_api.build_session(ec, params, window, logger, get_data_from_file=True)
            )

        servers.append(
            serve(
                ParticipantService(
                    lambda build=build: emhass_description(build(PlanWindow(30, 48))),
                    build,
                    lambda p, a: (True, ""),
                )
            )
        )
    try:
        site = [
            {
                "id": "garage",
                "parent": "grid",
                "type": "panel",
                "max_import": 7400,
                "max_export": 0,
            },
            {"id": "l1", "type": "limit", "max_import": 7000},
            {
                "id": "water_heater",
                "parent": "garage",
                "solver": "home_energy_optimizer",
                "limits": ["l1"],
                "config": {"power_kw": 3.0, "n_duty_levels": 2},
            },
            {
                "id": "garage_emhass",
                "parent": "garage",
                "limits": ["l1"],
                "solver": {"url": servers[0].url},
            },
            {"id": "shed_emhass", "parent": "garage", "solver": {"url": servers[1].url}},
        ]
        res = _coordinate(_house(_now()), site)
    finally:
        for server in servers:
            server.shutdown()
    assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[0]
    garage = res["fed_node_power_garage"]
    assert garage.max() <= 1.0 and garage.min() >= -7400 - 1.0, (garage.min(), garage.max())
    shared = res["P_water_heater"] + res["fed_remote_power_garage_emhass"]
    assert shared.max() <= 7000 + 1.0, shared.max()
    assert res["fed_remote_power_shed_emhass"].sum() * 0.5 == pytest.approx(6600.0)


# --------------------------------------------------------------------------
# A commit: published as the plan it is, then saved and recorded
# --------------------------------------------------------------------------

DAY = {
    "pv_power_forecast": [0.0] * 48,
    "load_power_forecast": [500.0] * 48,
    "load_cost_forecast": [0.1 + (i % 6) / 20 for i in range(48)],
    "prod_price_forecast": [0.05] * 48,
    "prediction_horizon": 48,
}


def _a_plan(ec: dict, params: dict) -> pd.DataFrame:
    """A plan for the garage, as the default solver makes it (a dry run)."""

    async def run():
        from emhass.command_line import naive_mpc_optim, set_input_data_dict

        idd = await set_input_data_dict(
            ec,
            "profit",
            orjson.dumps(params).decode(),
            orjson.dumps(DAY).decode(),
            "naive-mpc-optim",
            logger,
            get_data_from_file=True,
            dry_run=True,
        )
        return await naive_mpc_optim(idd, logger, debug=True)

    return asyncio.run(run())


def _commit(
    ec: dict,
    params: dict,
    res: pd.DataFrame,
    published: list,
    ok: bool = True,
    refuse: str | None = None,
):
    """publish_commit on the saved test data; `published` gets each
    (entity_id, publish_prefix) posted to Home Assistant, which refuses (HTTP
    500) the entities whose id contains `refuse`."""

    async def post(self, data_df, idx, entity_id, *args, **kwargs):
        published.append((entity_id, kwargs.get("publish_prefix")))
        taken = refuse is None or refuse not in entity_id
        return mock.MagicMock(ok=taken, status_code=200 if taken else 500), {}

    saved = mock.AsyncMock(return_value=pd.DataFrame({"stale": [1.0]}))
    with (
        mock.patch("emhass.retrieve_hass.RetrieveHass.post_data", post),
        mock.patch("emhass.command_line._publish_from_saved_entities", saved),
        mock.patch.object(participant_api, "publish_data", mock.AsyncMock(return_value=None))
        if not ok
        else contextlib.nullcontext(),
    ):
        out = asyncio.run(
            participant_api.publish_commit(ec, params, res, logger, get_data_from_file=True)
        )
    return out, saved


@REMOTE_READY
def test_a_commit_through_the_route_publishes_and_records_the_plan_it_names(data_path):
    """End to end through /participant/v1: a baseline answered by this
    EMHASS's own service (make_service), then a commit of that answer, which
    publishes it under the prefix and records it as the latest plan. Only
    Home Assistant itself (post_data) is stood in for."""
    from home_energy_optimizer.remote import SCHEMA_VERSION

    params = _params(_conf(data_path), **GARAGE)
    published: list = []

    async def post(self, data_df, idx, entity_id, *args, **kwargs):
        published.append((entity_id, kwargs.get("publish_prefix")))
        return mock.MagicMock(ok=True, status_code=200), {}

    horizon = {"step_minutes": 30.0, "slots": 48, "start": _now().isoformat()}

    async def run():
        saved_conf = web_server.emhass_conf
        saved_secret = web_server.params_secrets.get("participant_token")
        web_server.emhass_conf = _conf(data_path)
        web_server.params_secrets["participant_token"] = "t"
        (data_path / "params.pkl").write_bytes(pickle.dumps((None, params)))
        auth = {"Authorization": "Bearer t", "Content-Type": "application/json"}
        try:
            client = web_server.app.test_client()
            r = await client.post(
                "/participant/v1/baseline",
                data=json.dumps(horizon).encode(),
                headers={**auth, "Query-Id": "q1"},
            )
            assert r.status_code == 200, await r.get_data()
            answer = await r.get_json()
            commit = {
                "version": SCHEMA_VERSION,
                "horizon": horizon,
                "plan_kw": answer["plan_kw"],
                "query_id": "q1",
            }
            r = await client.post(
                "/participant/v1/commit", data=json.dumps(commit).encode(), headers=auth
            )
            return r.status_code, await r.get_json(), answer
        finally:
            web_server.emhass_conf = saved_conf
            if saved_secret is None:
                web_server.params_secrets.pop("participant_token", None)
            else:
                web_server.params_secrets["participant_token"] = saved_secret

    # the saved test data, not a live Home Assistant: the real functions, from files
    from_files = {
        name: functools.partial(getattr(participant_api, name), get_data_from_file=True)
        for name in ("build_session", "publish_commit")
    }
    with (
        mock.patch("emhass.retrieve_hass.RetrieveHass.post_data", post),
        mock.patch.object(web_server, "_participant_service", None),
        mock.patch.object(participant_api, "build_session", from_files["build_session"]),
        mock.patch.object(participant_api, "publish_commit", from_files["publish_commit"]),
    ):
        status, out, answer = asyncio.run(run())
    assert status == 200 and out.get("accepted") is True, out
    assert published and {prefix for _, prefix in published} == {"garage_"}
    plan = plan_store.read(data_path)
    assert plan and len(plan["plan"]) == len(answer["plan_kw"])
    assert pd.Timestamp(plan["plan"][0]["timestamp"]) == pd.Timestamp(horizon["start"])
    assert last_run.read(data_path)["action"] == "naive-mpc-optim"


def test_a_commit_home_assistant_did_not_take_is_refused_and_not_recorded(data_path):
    """publish_data logs an entity Home Assistant refused and goes on; the
    commit is refused, names it, and saves and records nothing."""
    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    res = _a_plan(ec, params)
    csv = data_path / "opt_res_latest.csv"
    before = (
        plan_store.read(data_path),
        last_run.read(data_path),
        csv.exists() and csv.read_bytes(),
    )
    published: list = []
    (ok, why), _ = _commit(ec, params, res, published, refuse="p_deferrable0")
    assert not ok and "did not take" in why and "p_deferrable0" in why and "part way" in why
    assert published  # it was published as far as it went
    after = (
        plan_store.read(data_path),
        last_run.read(data_path),
        csv.exists() and csv.read_bytes(),
    )
    assert after == before


def test_a_commit_publishes_its_plan_under_the_prefix_and_records_it(data_path):
    """The committed plan itself is published - under participant_api's
    publish_prefix, never the entity files a prefix would otherwise republish -
    then saved as the latest results and recorded for /api/v1/plan and
    /api/v1/last-run."""
    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    res = _a_plan(ec, params)
    published: list = []
    (ok, why), saved = _commit(ec, params, res, published)
    assert ok, why
    saved.assert_not_called()
    assert published and {prefix for _, prefix in published} == {"garage_"}
    latest = pd.read_csv(data_path / "opt_res_latest.csv", index_col="timestamp")
    assert len(latest) == len(res)
    plan = plan_store.read(data_path)
    assert plan and len(plan["plan"]) == len(res)
    assert pd.Timestamp(plan["plan"][0]["timestamp"]) == res.index[0]
    assert last_run.read(data_path)["action"] == "naive-mpc-optim"


def test_a_commit_that_fails_to_publish_keeps_the_latest_plan(data_path):
    """Nothing is saved or recorded until the plan is published; the answer
    says it may have been published part way."""
    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    res = _a_plan(ec, params)
    before = (data_path / "opt_res_latest.csv").read_bytes()
    (ok, why), _ = _commit(ec, params, res, [], ok=False)
    assert not ok and "part way" in why
    assert (data_path / "opt_res_latest.csv").read_bytes() == before


def test_a_commit_is_refused_while_continual_publish_runs(data_path):
    """Its loop republishes the entity files of an earlier run, which would
    put that plan back over the committed one."""
    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    params["retrieve_hass_conf"]["continual_publish"] = True
    published: list = []
    (ok, why), _ = _commit(ec, params, _a_plan(ec, params), published)
    assert not ok and "continual_publish" in why and not published


def test_an_older_commit_does_not_replace_a_newer_plan(data_path):
    """A late or repeated commit for an earlier horizon is refused once a
    plan that starts later is in place."""
    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    res = _a_plan(ec, params)
    later = res.copy()
    later.index = later.index + pd.Timedelta(hours=1)
    plan_store.record(data_path, plan_store.serialize(later), "2026-11-15T00:00:00Z", "1.0")
    published: list = []
    (ok, why), _ = _commit(ec, params, res, published)
    assert not ok and "before the latest plan" in why and not published


@pytest.mark.parametrize(
    "state, unit, expected",
    [
        ("0.5", "%", 0.005),
        ("55", "%", 0.55),
        ("1", "%", 0.01),
        ("55", None, 0.55),
        ("0.5", None, 0.005),
        ("150", "%", None),
        ("-3", "%", None),
        ("0.55", "kWh", None),
    ],
)
def test_the_battery_charge_is_read_in_percent(state, unit, expected):
    """The sensor is in percent, as EMHASS documents it: never guessed from
    the value; another unit, or a charge outside 0-100 %, is None."""
    body = {"state": state, "attributes": {"unit_of_measurement": unit} if unit else {}}
    response = mock.MagicMock()
    response.__enter__.return_value = io.BytesIO(json.dumps(body).encode())
    conf = {
        "sensor_battery_state_of_charge": "sensor.soc",
        "hass_url": "http://ha",
        "long_lived_token": "t",
    }
    with mock.patch("urllib.request.urlopen", return_value=response):
        got = participant_api.soc_now(conf, logger)
    assert got is None if expected is None else got == pytest.approx(expected)


@pytest.mark.parametrize(
    "base, no_verify, expected",
    [
        ("http://ha:8123/", False, "http://ha:8123/api/states/sensor.soc"),
        ("http://supervisor/core/api", False, "http://supervisor/core/api/states/sensor.soc"),
        ("https://ha.local", True, "https://ha.local/api/states/sensor.soc"),
    ],
)
def test_the_battery_charge_is_asked_where_home_assistant_answers(base, no_verify, expected):
    """Under the add-on hass_url is the Supervisor's, which already ends in
    /api; ssl_no_verify holds for this call as for EMHASS's others."""
    response = mock.MagicMock()
    response.__enter__.return_value = io.BytesIO(b'{"state": "55", "attributes": {}}')
    conf = {
        "sensor_battery_state_of_charge": "sensor.soc",
        "hass_url": base,
        "long_lived_token": "t",
        "ssl_no_verify": no_verify,
    }
    with mock.patch("urllib.request.urlopen", return_value=response) as urlopen:
        assert participant_api.soc_now(conf, logger) == pytest.approx(0.55)
    assert urlopen.call_args.args[0].full_url == expected
    assert (urlopen.call_args.kwargs["context"] is not None) == no_verify


@pytest.mark.parametrize("name", ["long_lived_token", "participant_token", "solcast_api_key"])
def test_a_remote_may_not_be_sent_one_of_emhass_own_secrets(name):
    """A saved site naming EMHASS's own secret as a remote's token would send
    it to that remote's URL: the site is refused, and the secret is never
    among the remotes' tokens."""
    site = [
        {"id": "grid"},
        {
            "id": "elsewhere",
            "parent": "grid",
            "solver": {"url": "http://elsewhere.example/x", "token_secret": name},
        },
    ]
    assert "not one of EMHASS's secrets" in utils.site_problem(site)
    secrets = {"long_lived_token": "HA", "participant_token": "P", "solcast_api_key": "S"}

    async def build():
        config = orjson.loads((root / "src/emhass/data/config_defaults.json").read_bytes())
        config["site"] = site
        return await build_params(EMHASS_CONF, secrets, config, logger)

    assert "remote_tokens" not in asyncio.run(build())["retrieve_hass_conf"]
    site[1]["solver"]["token_secret"] = "garage_token"
    assert utils.site_problem(site) is None


def test_a_corrupt_configuration_is_reported_not_called_off(data_path):
    """An unreadable params.pkl is a server error, not "the API is off"."""
    (data_path / "params.pkl").write_bytes(b"not a pickle")
    status, body = _client_get(data_path, None, "/participant/v1/describe", token="t")
    assert status == 503 and "could not be read" in body["detail"]


def test_a_large_body_is_refused_before_it_is_read(data_path):
    """Queries and commits are a few kB: a body over 1 MiB is refused (413)."""

    async def post():
        client = web_server.app.test_client()
        body = b"x" * (web_server.PARTICIPANT_BODY_MAX + 1)
        r = await client.post(
            "/participant/v1/query",
            data=body,
            headers={"Authorization": "Bearer t", "Content-Length": str(len(body))},
        )
        return r.status_code

    with mock.patch.object(participant_api, "read_params", mock.AsyncMock()) as read:
        assert asyncio.run(post()) == 413
    read.assert_not_called()


def _stream(data_path, chunks: list[bytes], token: str = "t"):
    """POST /participant/v1/query as a chunked upload that sends `chunks` and
    then stops, never ending its body: the status answered before the end
    (None if none), and the service the route would have called."""
    params = _params(_conf(data_path), **GARAGE)
    service = mock.MagicMock()
    service.handle.return_value = (200, {})

    async def upload():
        saved_conf = web_server.emhass_conf
        saved_secret = web_server.params_secrets.get("participant_token")
        web_server.emhass_conf = {**saved_conf, "data_path": data_path}
        web_server.params_secrets["participant_token"] = "t"
        (data_path / "params.pkl").write_bytes(pickle.dumps((None, params)))
        try:
            client = web_server.app.test_client()
            async with client.request(
                "/participant/v1/query",
                method="POST",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            ) as connection:
                for chunk in chunks:
                    await connection.send(chunk)
                for _ in range(200):  # up to 4 s for an answer, the body not ended
                    if connection.status_code is not None:
                        break
                    await asyncio.sleep(0.02)
                status = connection.status_code
                await connection.send_complete()  # so a route that waits for it ends
            return status
        finally:
            web_server.emhass_conf = saved_conf
            if saved_secret is None:
                web_server.params_secrets.pop("participant_token", None)
            else:
                web_server.params_secrets["participant_token"] = saved_secret

    package = types.ModuleType("home_energy_optimizer")
    package.remote = types.ModuleType("home_energy_optimizer.remote")
    with (
        mock.patch.object(participant_api, "make_service", return_value=service),
        mock.patch.object(web_server, "_participant_service", None),
        mock.patch.dict(
            sys.modules,
            {"home_energy_optimizer": package, "home_energy_optimizer.remote": package.remote},
        ),
    ):
        return asyncio.run(upload()), service


def test_a_chunked_large_body_is_refused_as_it_arrives(data_path):
    """Chunks each well below 1 MiB, more than 1 MiB together, and no
    Content-Length: refused (413) before the upload ends."""
    chunk = b" " * (64 * 1024)
    status, service = _stream(
        data_path, [chunk] * (web_server.PARTICIPANT_BODY_MAX // len(chunk) + 1)
    )
    assert status == 413
    service.handle.assert_not_called()


def test_a_body_that_stalls_is_refused(data_path, monkeypatch):
    """A body that stops arriving is refused (408), as Quart's own
    BODY_TIMEOUT refuses one read whole."""
    monkeypatch.setattr(web_server, "PARTICIPANT_BODY_TIMEOUT_S", 0.2)
    status, service = _stream(data_path, [b"{"])
    assert status == 408
    service.handle.assert_not_called()


def test_a_caller_without_the_token_is_refused_before_its_body(data_path):
    """A wrong token is refused (401) before any body is read: the upload has
    not ended when the answer comes."""
    status, service = _stream(data_path, [b"{"], token="wrong")
    assert status == 401
    service.handle.assert_not_called()


@REMOTE_READY
@pytest.mark.parametrize(
    "window, problem",
    [
        ({"step_minutes": 30, "slots": 7 * 48 + 1}, "1 to 336 slots"),
        ({"step_minutes": 30, "slots": 0}, "1 to 336 slots"),
    ],
)
def test_a_session_refuses_a_horizon_it_cannot_plan(data_path, window, problem):
    from home_energy_optimizer.remote import PlanWindow, WindowRefused

    ec = _conf(data_path)
    with pytest.raises(WindowRefused, match=problem):
        asyncio.run(
            participant_api.build_session(
                ec, _params(ec, **GARAGE), PlanWindow(**window), logger, get_data_from_file=True
            )
        )


@REMOTE_READY
def test_a_session_refuses_slots_it_does_not_forecast(data_path):
    """Asked about slots one step off its own, a remote refuses rather than
    plan them with the neighbouring slots' forecasts."""
    from home_energy_optimizer.remote import PlanWindow, WindowRefused

    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    p = asyncio.run(
        participant_api.build_session(
            ec, params, PlanWindow(30, 48), logger, get_data_from_file=True
        )
    )
    off = (p.data.index[0] + pd.Timedelta(minutes=30)).isoformat()  # one step after its own
    with pytest.raises(WindowRefused, match="must agree on the slot"):
        asyncio.run(
            participant_api.build_session(
                ec, params, PlanWindow(30, 48, off), logger, get_data_from_file=True
            )
        )


def test_the_published_api_says_how_to_authenticate():
    """A client generated from openapi.json sends the participant token."""
    spec = json.loads((root / "src/emhass/static/openapi.json").read_text(encoding="utf-8"))
    path = spec["paths"]["/participant/v1/{op}"]
    scheme = spec["components"]["securitySchemes"]["participantToken"]
    assert scheme == {**scheme, "type": "http", "scheme": "bearer"}
    for method in ("get", "post"):
        assert path[method]["security"] == [{"participantToken": []}]
        assert "501" in path[method]["responses"]
    assert "application/json" in path["post"]["requestBody"]["content"]
    # a commit's 429 and lock-unavailable 503 are retryable: the client is told when
    for code in ("429", "503"):
        assert "Retry-After" in path["post"]["responses"][code]["headers"]
    assert "lock" in path["post"]["responses"]["503"]["description"]
