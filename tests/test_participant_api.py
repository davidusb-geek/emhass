"""This EMHASS's devices served to a coordinator elsewhere (participant_api),
and the coordinator's side: a remote solver named in `site`.

The remote side is emhass.coordinator.remote's ParticipantService, served on
a local port (`serve`, below), building its model with EMHASS's own input path;
the coordinator is another EMHASS's optimize, as a live run calls it."""

import asyncio
import contextlib
import copy
import functools
import io
import json
import pathlib
import pickle
import shutil
import ssl
import tempfile
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Self
from unittest import mock

import numpy as np
import orjson
import pandas as pd
import pytest
import time_machine

from emhass import last_run, participant_api, plan_store, utils, web_server
from emhass.coordinator.backend import emhass_description
from emhass.coordinator.interface import SCHEMA_VERSION, Answer, Query
from emhass.coordinator.remote import ParticipantService, PlanWindow, WindowRefused
from emhass.optimization import Optimization
from emhass.utils import (
    build_config,
    build_params,
    build_secrets,
    get_logger,
    get_root,
    get_yaml_parse,
)

from .api_auth_support import BEARER as API_BEARER
from .api_auth_support import AuthSupportBase

root = pathlib.Path(get_root(__file__, num_parent=2))
logger, _ = get_logger(__name__, {"data_path": root / "data/"}, save_to_file=False)
EMHASS_CONF = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "defaults_path": root / "src/emhass/data/config_defaults.json",
    "associations_path": root / "src/emhass/data/associations.csv",
}

# participant_token as the route requires it: at least api_auth.MIN_LENGTH characters
TOKEN = "participant-token-0123456789"


def _confs():
    """EMHASS's default configuration, as at startup: (retrieve_hass_conf,
    optim_conf, plant_conf)."""
    params = _params(EMHASS_CONF)
    return get_yaml_parse(orjson.dumps(params).decode("utf-8"), logger)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    service: ParticipantService
    url: str

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
        self.server_close()


_servers: list[_Server] = []  # every server a test started, closed after it


@pytest.fixture(autouse=True)
def _servers_closed():
    """Every server a test started is shut down and closed after it, however
    the test ended - one started before a later one failed to start too."""
    yield
    while _servers:
        server = _servers.pop()
        with contextlib.suppress(Exception):
            server.shutdown()
        with contextlib.suppress(Exception):
            server.server_close()


class _Handler(BaseHTTPRequestHandler):
    """The participant API on the standard library's HTTP server: another
    EMHASS, as far as the coordinator can tell."""

    server: _Server

    def _reply(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else None
        status, out = self.server.service.handle(
            self.command, self.path.split("?", 1)[0], dict(self.headers), body
        )
        raw = json.dumps(out).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    do_GET = do_POST = _reply

    def log_message(self, format, *args):  # noqa: A002 - the stdlib's name
        pass


def serve(service: ParticipantService) -> _Server:
    """Serve `service` on a free local port from a daemon thread; `url` is
    the API's base. Usable as a context manager; whatever a test leaves
    running is shut down and closed after it (_servers_closed)."""
    server = _Server(("127.0.0.1", 0), _Handler)
    server.service = service
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    except BaseException:
        server.server_close()  # never served: shutdown() would wait for ever
        raise
    _servers.append(server)
    return server


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


def _client_get(
    data_path: pathlib.Path,
    params: dict | None,
    path: str,
    token: str | None = None,
    presented: str | None = None,
):
    """GET `path` from the web app, with `params` as its params.pkl, the
    server holding `token` and the caller presenting `presented` (the
    server's token by default)."""

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
            r = await client.get(
                path, headers={"Authorization": f"Bearer {presented or token or 'wrong'}"}
            )
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
    status, body = _client_get(data_path, params, "/participant/v1/describe", token=TOKEN)
    assert status == 404 and "participant_api" in body["detail"]


@pytest.mark.parametrize("configured", [False, True])
def test_a_wrong_token_learns_nothing_of_the_configuration(data_path, configured):
    """Without the token a caller is refused (401) whether the API is on or
    off, and nothing is read for it: not params.pkl, not its body."""
    params = _params(_conf(data_path), **(GARAGE if configured else {}))
    with mock.patch.object(
        participant_api, "read_params", side_effect=AssertionError("read")
    ) as read:
        status, _ = _client_get(
            data_path, params, "/participant/v1/describe", token=TOKEN, presented="wrong"
        )
    assert status == 401
    read.assert_not_called()


def test_the_participant_api_needs_a_token(data_path):
    params = _params(_conf(data_path), **GARAGE)
    status, body = _client_get(data_path, params, "/participant/v1/describe")
    assert status == 503 and "participant_token" in body["detail"]


def test_a_short_participant_token_is_not_served(data_path):
    """As the API token: a participant_token shorter than api_auth.MIN_LENGTH
    is guessable, so the API answers 503 and says so, even to its holder."""
    from emhass import api_auth

    params = _params(_conf(data_path), **GARAGE)
    short = "x" * (api_auth.MIN_LENGTH - 1)
    status, body = _client_get(data_path, params, "/participant/v1/describe", token=short)
    assert status == 503 and str(api_auth.MIN_LENGTH) in body["detail"]
    status, body = _client_get(data_path, params, "/participant/v1/describe", token=short + "x")
    assert not (status == 503 and str(api_auth.MIN_LENGTH) in body["detail"]), body


def test_the_participant_api_needs_one_web_worker(data_path, monkeypatch):
    """A coordinator's answers live in one worker's memory: with more, its
    commit could reach a worker that never gave the answer it names."""
    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    params = _params(_conf(data_path), **GARAGE)
    status, body = _client_get(data_path, params, "/participant/v1/describe", token=TOKEN)
    assert status == 503 and "one web worker" in body["detail"]


class _Fixed:
    """A participant with one plan, for the commit tests: no model built."""

    key, max_power_kw, modulating, onoff = "fixed", 1.0, False, True

    def baseline(self):
        return Answer(np.array([1.0, 0.0]), 0.0, np.zeros(3))

    def respond(self, query):
        return self.baseline()


_HORIZON = {"step_minutes": 30.0, "slots": 2, "start": "2026-11-21T10:00:00+00:00"}
_PLAN = [1.0, 0.0]


def _service(commit, begin=None, log=None):
    """A ParticipantService over _Fixed with one answer, q1, to commit."""
    from emhass.coordinator.remote import CommitLog

    service = ParticipantService(
        lambda: {"key": "fixed", "max_power_kw": 1.0},
        lambda window: _Fixed(),
        commit,
        begin=begin,
        log=log if log is not None else CommitLog(),
    )
    status, _ = service.handle(
        "POST", "/v1/baseline", {"Query-Id": "q1"}, json.dumps(_HORIZON).encode()
    )
    assert status == 200
    return service


def _commit_body(plan=_PLAN, qid="q1"):
    body = {"version": SCHEMA_VERSION, "horizon": _HORIZON, "plan_kw": plan, "query_id": qid}
    return json.dumps(body).encode()


def _until_final(service, cid="q1"):
    for _ in range(200):
        status, record = service.handle("GET", f"/v1/commit/{cid}", {}, None)
        if status != 202:
            return status, record
        threading.Event().wait(0.02)
    raise AssertionError("the commit did not end")


def test_a_commit_is_answered_at_once_and_holds_the_slot_until_it_has_run(data_path):
    """The participant answers a commit at once (202, running) and runs it
    after, holding the live-optimization slot until it has run: a live
    optimization cannot start under it. Its outcome is then asked for."""
    saved_conf = web_server.emhass_conf
    web_server.emhass_conf = {**saved_conf, "data_path": data_path}
    started, go = threading.Event(), threading.Event()

    def commit(participant, answer):
        started.set()
        assert go.wait(10)
        return "accepted", ""

    def slot_free() -> bool:
        with open(web_server._live_optim_slot(), "a+") as f:
            if utils.try_lock_file(f):
                utils.unlock_file(f)
                return True
            return False

    try:
        service = _service(commit, begin=web_server._take_live_optim_slot)
        status, record = service.handle("POST", "/v1/commit", {}, _commit_body())
        assert status == 202 and record["state"] == "running" and record["accepted"] is None
        assert started.wait(10)
        assert not slot_free(), "the slot was free while the commit ran"
        assert service.handle("GET", "/v1/commit/q1", {}, None)[0] == 202
        go.set()
        status, record = _until_final(service)
        assert status == 200 and record["state"] == "accepted" and record["accepted"] is True
        for _ in range(100):
            if slot_free():
                break
            threading.Event().wait(0.02)
        else:
            raise AssertionError("the slot was not freed once the commit ended")
    finally:
        go.set()
        web_server.emhass_conf = saved_conf


def test_a_commit_while_a_live_optimization_runs_is_refused_and_runs_nothing(data_path):
    """While this EMHASS runs its own optimization (it holds the slot), a
    commit is refused (429, Retry-After): nothing runs and nothing is
    recorded, so the same commit may be sent again."""
    saved_conf = web_server.emhass_conf
    web_server.emhass_conf = {**saved_conf, "data_path": data_path}
    ran = []
    try:
        service = _service(
            lambda p, a: (ran.append(1), ("accepted", ""))[1],
            begin=web_server._take_live_optim_slot,
        )
        with open(web_server._live_optim_slot(), "a+") as live:
            assert utils.try_lock_file(live)
            try:
                status, problem = service.handle("POST", "/v1/commit", {}, _commit_body())
            finally:
                utils.unlock_file(live)
        assert status == 429, problem
        assert service.handle("GET", "/v1/commit/q1", {}, None)[0] == 404
        assert service.handle("POST", "/v1/commit", {}, _commit_body())[0] == 202
        assert _until_final(service)[1]["state"] == "accepted" and ran == [1]
    finally:
        web_server.emhass_conf = saved_conf


def test_the_route_tells_a_busy_coordinator_when_to_retry(data_path):
    """The service's 429 reaches the coordinator with Retry-After."""
    params = _params(_conf(data_path), **GARAGE)
    service = mock.MagicMock()
    service.handle.return_value = (429, {"title": "Too Many Requests", "status": 429})

    async def post():
        saved_conf = web_server.emhass_conf
        web_server.emhass_conf = {**saved_conf, "data_path": data_path}
        web_server.params_secrets["participant_token"] = TOKEN
        (data_path / "params.pkl").write_bytes(pickle.dumps((None, params)))
        try:
            r = await web_server.app.test_client().post(
                "/participant/v1/commit",
                data=b"{}",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
            return r.status_code, r.headers.get("Retry-After")
        finally:
            web_server.emhass_conf = saved_conf
            web_server.params_secrets.pop("participant_token", None)

    with (
        mock.patch.object(participant_api, "make_service", return_value=service),
        mock.patch.object(web_server, "_participant_service", None),
    ):
        assert asyncio.run(post()) == (429, "2")


def test_a_commit_sent_again_is_never_run_twice():
    """A coordinator whose commit got no answer may send it again: the
    participant answers that commit's record, running or ended, and never
    runs it twice; the same id with another plan is refused (409)."""
    go, ran = threading.Event(), []

    def commit(participant, answer):
        ran.append(1)
        assert go.wait(10)
        return "accepted", ""

    service = _service(commit)
    assert service.handle("POST", "/v1/commit", {}, _commit_body())[0] == 202
    again = service.handle("POST", "/v1/commit", {}, _commit_body())
    assert again[0] == 202 and again[1]["state"] == "running"
    go.set()
    _until_final(service)
    again = service.handle("POST", "/v1/commit", {}, _commit_body())
    assert again[0] == 200 and again[1]["state"] == "accepted"
    assert service.handle("POST", "/v1/commit", {}, _commit_body([0.0, 1.0]))[0] == 409
    assert ran == [1]


@pytest.mark.parametrize(
    "outcome, state, accepted",
    [
        (("accepted", ""), "accepted", True),
        (("refused", "no"), "refused", False),
        (("partial", "cut off"), "partial", None),
        ((True, ""), "accepted", True),  # a host's (bool, reason)
        ((False, "no"), "refused", False),
        (RuntimeError("boom"), "partial", None),  # it may have run part of it
    ],
)
def test_a_commits_outcome_is_recorded_as_it_ended(outcome, state, accepted):
    def commit(participant, answer):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    service = _service(commit)
    assert service.handle("POST", "/v1/commit", {}, _commit_body())[0] == 202
    status, record = _until_final(service)
    assert status == 200 and record["state"] == state and record["accepted"] is accepted
    assert record["terms"] is None and record["signatures"] == []  # reserved
    _, described = service.handle("GET", "/v1/describe", {}, None)
    if state in ("accepted", "partial"):  # the plan it runs from now on (in part)
        assert described["active_commit"]["commit_id"] == "q1"
        assert described["active_commit"]["plan_kw"] == _PLAN
    else:  # refused: the plan before it stays (none here)
        assert "active_commit" not in described
    assert service.handle("GET", "/v1/commit/unknown", {}, None)[0] == 404


def test_the_commit_log_outlives_a_restart(tmp_path):
    """Records are kept in the data folder: a restarted participant still
    answers for its commits, and one that was running when it stopped is
    read back as partial (what it published is not known)."""
    from emhass.coordinator.remote import CommitLog

    path = tmp_path / "participant_commits.json"
    log = CommitLog(path)
    log.begin("done", [1.0])
    log.end("done", "accepted", "")
    log.begin("cut", [1.0])  # never ended: the process stopped
    again = CommitLog(path)
    assert again.get("done")["state"] == "accepted"
    assert again.get("cut")["state"] == "partial" and "stopped" in again.get("cut")["reason"]
    assert again.get("cut").get("interrupted") is True
    old = CommitLog(path, keep_s=0.0)  # past its keeping time
    assert old.get("cut") is not None  # kept: the latest that ran (in part)
    # kept too while the one after it was cut off by the stop: its host may
    # find nothing of that one ran, and then this plan is the one that runs
    assert old.get("done") is not None
    old.end("cut", "refused", "EMHASS stopped before it published this plan")
    old.records()  # pruned as it is read
    assert old.get("done") is not None  # the latest that ran now
    old.begin("next", [1.0])
    old.end("next", "accepted", "")
    old.records()
    assert old.get("done") is None and old.get("cut") is None  # pruned: a later one ran whole


def test_a_wrong_token_is_refused(data_path):
    params = _params(_conf(data_path), **GARAGE)
    status, _ = _client_get(
        data_path, params, "/participant/v1/describe", token=TOKEN, presented="wrong"
    )
    assert status == 401


def test_a_hybrid_inverter_is_not_served(data_path):
    """A participant's own solve has no inverter, so its answers would not
    hold a hybrid inverter's AC limit, however it is set: refused."""
    ec = _conf(data_path)
    params = _params(ec, **{**GARAGE, "set_use_battery": True})
    params["optim_conf"]["set_nodischarge_to_grid"] = False  # declined on its own
    params["plant_conf"].update(inverter_is_hybrid=True, inverter_ac_output_max=3000)
    with pytest.raises(WindowRefused, match="its AC limits"):
        asyncio.run(
            participant_api.build_session(
                ec, params, PlanWindow(30, 48, _now().isoformat()), logger, get_data_from_file=True
            )
        )


@pytest.mark.parametrize("path", ["/v1/baseline", "/v1/query", "/v1/commit"])
def test_a_horizon_without_its_start_is_refused(path):
    """A session is a model built for given slots: a horizon that does not
    name its first slot could be answered from slots that have passed."""
    built = []
    service = ParticipantService(lambda: {}, lambda w: built.append(w))
    horizon = {"step_minutes": 30, "slots": 2}
    body = (
        horizon
        if path == "/v1/baseline"
        else {"version": 1, "kind": "price_response", "horizon": horizon, "plan_kw": [0, 0]}
    )
    status, problem = service.handle("POST", path, {}, json.dumps(body).encode())
    assert status == 400 and "horizon.start" in problem["detail"], problem
    assert built == []


def test_a_session_answers_for_a_horizon_and_refuses_another_slot_length(data_path):
    """Built through EMHASS's own input path, for the horizon asked about: its
    EV at a price runs its 2 h, the same query gives the same answer, and
    another slot length is refused."""

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


@pytest.fixture(autouse=True)
def _mid_slot():
    """Each test runs from the middle of a slot, as EMHASS rounds it (to the
    nearest): _now() and a remote's own clock, read a moment later, then
    place the first slot alike, however long the test takes up to a quarter
    hour. The clock still runs."""
    step = _confs()[0]["optimization_time_step"]
    step = step if isinstance(step, pd.Timedelta) else pd.Timedelta(minutes=float(step))
    mid = pd.Timestamp.now(tz="UTC").round(step)
    with time_machine.travel(mid.to_pydatetime(), tick=True):
        yield


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


def test_a_remote_emhass_is_planned_with_the_house_and_runs_its_part(data_path):
    """A second EMHASS (an EV charger), served over HTTP, is one participant of
    this EMHASS's plan: its 2 h run, its power and share in the plan; on a
    live run it is asked to run the plan chosen, on a dry run it is not."""

    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    committed = []

    def build(window):
        return asyncio.run(
            participant_api.build_session(ec, params, window, logger, get_data_from_file=True)
        )

    def describe():
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


def test_a_remote_that_does_not_answer_is_planned_without(data_path):
    """A remote that is down costs its part of the coordination, not the plan:
    the house is planned without it, and the plan says so."""

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


def test_a_remote_that_refuses_its_plan_is_planned_without(data_path):
    """The garage takes its plan; the shed refuses its own. The house is
    planned again without the shed, the garage held to the plan it took (not
    asked again): the published plan holds neither the shed's power nor a
    plan the garage does not run."""

    ec = _conf(data_path)
    asked = {"garage": [], "shed": []}

    def remote(optim, key, ok):
        params = _params(ec, **optim)

        def build(window):
            return asyncio.run(
                participant_api.build_session(ec, params, window, logger, get_data_from_file=True)
            )

        def describe():
            return emhass_description(build(PlanWindow(30, 48)))

        def commit(_p, a):
            asked[key].append(a)
            return ok, "" if ok else "its owner is away"

        return serve(ParticipantService(describe, build, commit))

    garage, shed = remote(GARAGE, "garage", True), remote(SHED, "shed", False)
    try:
        site = [
            {"id": "garage", "parent": "grid", "solver": {"url": garage.url}},
            {"id": "shed", "parent": "grid", "solver": {"url": shed.url}},
        ]
        res = _coordinate(_house(_now()), site)
        assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[
            0
        ]
        assert (res["fed_remote_status_shed"] == "refused: its owner is away").all()
        assert (res["fed_remote_status_garage"] == "committed").all()
        assert len(asked["shed"]) == 1 and len(asked["garage"]) == 1
        assert "fed_remote_power_shed" not in res.columns
        ev = res["fed_remote_power_garage"]
        assert np.allclose(asked["garage"][0].plan_kw * 1000.0, ev.values)
        balance = res["P_PV"] + res["P_grid"] + res["P_batt"]
        used = res["P_Load"] + res["P_deferrable0"] + res["P_deferrable1"] + ev
        assert np.allclose(balance, used, atol=1e-3)
    finally:
        garage.shutdown()
        shed.shutdown()


def test_a_commit_that_gets_no_answer_keeps_the_remotes_part(data_path, monkeypatch):
    """A commit that times out may have reached the remote and be running:
    the plan keeps its part (not planned around it as absent), and says the
    outcome is uncertain."""
    from emhass.coordinator.remote import RemoteError, RemoteParticipant

    ec = _conf(data_path)
    params = _params(ec, **GARAGE)

    def build(window):
        return asyncio.run(
            participant_api.build_session(ec, params, window, logger, get_data_from_file=True)
        )

    def describe():
        return emhass_description(build(PlanWindow(30, 48)))

    from emhass.coordinator import backend

    call = RemoteParticipant._call
    sent, asked = [], []

    def no_answer_to_commits(self, method, path, body=None, query_id=None, deadline=None):
        if path == "/v1/commit":
            sent.append(body)
            raise RemoteError(f"{self.key}: timed out")  # status 0: nothing came back
        if path.startswith("/v1/commit/"):
            asked.append(path)
            raise RemoteError(f"{self.key}: timed out")
        return call(self, method, path, body, query_id, deadline)

    monkeypatch.setattr(RemoteParticipant, "_call", no_answer_to_commits)
    monkeypatch.setattr(backend, "REMOTE_COMMIT_WAIT_S", 0.5)
    server = serve(ParticipantService(describe, build, lambda p, a: (True, "")))
    try:
        site = [{"id": "garage", "parent": "grid", "solver": {"url": server.url}}]
        res = _coordinate(_house(_now()), site)
        assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[
            0
        ]
        assert res["fed_remote_status_garage"].iloc[0].startswith("uncertain")
        # its outcome was asked for, and the commit was never sent blind again
        assert len(sent) == 1 and asked
        ev = res["fed_remote_power_garage"]
        assert np.allclose(np.asarray(sent[0]["plan_kw"]) * 1000.0, ev.values)
        balance = res["P_PV"] + res["P_grid"] + res["P_batt"]
        used = res["P_Load"] + res["P_deferrable0"] + res["P_deferrable1"] + ev
        assert np.allclose(balance, used, atol=1e-3)
    finally:
        server.shutdown()


def test_a_failed_plan_after_a_commit_plans_around_the_remotes_running(data_path, monkeypatch):
    """The garage takes its plan; the shed refuses; planning the house again
    without the shed fails outright. The plan made with the shed cannot be
    published (it counts the shed's power), and the garage runs already: so
    EMHASS's default solver plans the house around the garage's power, as
    load it cannot move, without the shed. Both statuses are kept."""
    from emhass.coordinator import backend

    ec = _conf(data_path)

    def remote(optim, ok):
        params = _params(ec, **optim)

        def build(window):
            return asyncio.run(
                participant_api.build_session(ec, params, window, logger, get_data_from_file=True)
            )

        def describe():
            return emhass_description(build(PlanWindow(30, 48)))

        return serve(ParticipantService(describe, build, lambda p, a: (ok, "" if ok else "no")))

    built = []
    coordinator = backend.DWCoordinator

    def fails_the_third(*args, **kwargs):
        built.append(1)
        if len(built) == 3:  # the plan, its split, then the plan made again
            raise RuntimeError("the re-plan failed")
        return coordinator(*args, **kwargs)

    monkeypatch.setattr(backend, "DWCoordinator", fails_the_third)
    garage, shed = remote(GARAGE, True), remote(SHED, False)
    try:
        site = [
            {"id": "garage", "parent": "grid", "solver": {"url": garage.url}},
            {"id": "shed", "parent": "grid", "solver": {"url": shed.url}},
        ]
        house = _house(_now())
        res = _coordinate(house, site)
        assert (res["backend_used"] == "cvxpy").all()
        assert "running a plan" in res["backend_fallback_reason"].iloc[0]
        assert (res["fed_remote_status_garage"] == "committed").all()
        assert res["fed_remote_status_shed"].iloc[0].startswith("refused")
        assert "fed_remote_power_shed" not in res.columns
        ev = res["fed_remote_power_garage"]
        assert ev.sum() * 0.5 == pytest.approx(14000.0)  # its 7 kW for 2 h, in W
        # the default solver counted it as load
        assert np.allclose(res["P_Load"].values, np.asarray(house[5]) + ev.values)
        assert len(built) == 3
    finally:
        garage.shutdown()
        shed.shutdown()


def test_the_plan_says_which_commit_each_remote_runs(data_path):
    """A remote reports the commit whose plan it runs (describe's
    active_commit): the plan shows it, so a run starts from what the remote
    actually does - here, the commit of the run before, published part way
    (it runs that, in part) and then accepted."""
    ec = _conf(data_path)
    params = _params(ec, **GARAGE)

    def build(window):
        return asyncio.run(
            participant_api.build_session(ec, params, window, logger, get_data_from_file=True)
        )

    def describe():
        return emhass_description(build(PlanWindow(30, 48)))

    outcome = {"next": ("partial", "cut off")}
    server = serve(ParticipantService(describe, build, lambda p, a: outcome["next"]))
    try:
        site = [{"id": "garage", "parent": "grid", "solver": {"url": server.url}}]
        first = _coordinate(_house(_now()), site)
        assert "fed_remote_last_commit_garage" not in first.columns  # no commit before
        assert first["fed_remote_status_garage"].iloc[0].startswith("uncertain")
        outcome["next"] = ("accepted", "")
        second = _coordinate(_house(_now()), site)
        assert second["fed_remote_last_commit_garage"].iloc[0] == "partial: cut off"
        third = _coordinate(_house(_now()), site)
        assert third["fed_remote_last_commit_garage"].iloc[0] == "accepted"
    finally:
        server.shutdown()


def _declined_for_a_limit(res) -> bool:
    """Whether the plan is the default solver's because the coordinator's own
    breached a limit by a little (its on/off plans could not quite fit, as on
    some platforms' solvers): the one fallback these tests accept. Any other
    fallback fails the test; a coordinated plan is checked by the caller."""
    if (res["backend_used"] == "dantzig_wolfe").all():
        return False
    reason = res["backend_fallback_reason"].iloc[0]
    assert "exceeds a limit" in reason, reason
    return True


@pytest.mark.parametrize("soc", [0.15, 0.6, 0.95])
def test_a_remote_battery_behind_a_panel_holds_every_limit(data_path, monkeypatch, soc):
    """The remote's EV and a battery behind a garage panel (7.4 kW in, no
    backfeed), sharing a 7 kW limit with the house's first load, from a
    nearly empty, a half and a full battery. The remote's own model holds its
    connection's limits (its maximum_power_from_grid / _to_grid), however big
    its EV; the coordinator holds what it shares with that load. No
    limit is breached."""

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
            {"id": "deferrable0", "parent": "garage", "limits": ["l1"]},
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
    if _declined_for_a_limit(res):
        return
    garage = res["fed_node_power_garage"]  # W, + = up the tree
    assert garage.max() <= 1.0 and garage.min() >= -7400 - 1.0, (garage.min(), garage.max())
    shared = res["P_deferrable0"] + res["fed_remote_power_garage_emhass"]
    assert shared.max() <= 7000 + 1.0, shared.max()


SHED = {
    **GARAGE,
    "nominal_power_of_deferrable_loads": [1100],
    "operating_hours_of_each_deferrable_load": [6],
    "participant_api": {"publish_prefix": "shed_", "key": "shed"},
}


@pytest.mark.parametrize("soc", [0.15, 0.6, 0.95])
def test_two_remotes_and_a_load_share_a_panel_within_its_rating(tmp_path, monkeypatch, soc):
    """Two remote EMHASS behind one garage panel (7.4 kW in, no backfeed) with
    the house's first load: the garage's EV and battery (holding its own 7 kW
    connection), the shed's pool pump (1.1 kW for 6 h). Each is on/off, so the
    plan picks one of each one's plans; the panel's rating holds."""

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
            {"id": "deferrable0", "parent": "garage", "limits": ["l1"]},
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
    if _declined_for_a_limit(res):
        return
    garage = res["fed_node_power_garage"]
    assert garage.max() <= 1.0 and garage.min() >= -7400 - 1.0, (garage.min(), garage.max())
    shared = res["P_deferrable0"] + res["fed_remote_power_garage_emhass"]
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
    **kw,
):
    """publish_commit on the saved test data (`kw`: its own keywords); `published` gets each
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
            participant_api.publish_commit(ec, params, res, logger, get_data_from_file=True, **kw)
        )
    return out, saved


def test_a_commit_through_the_route_publishes_and_records_the_plan_it_names(data_path):
    """End to end through /participant/v1: a baseline answered by this
    EMHASS's own service (make_service), then a commit of that answer, which
    publishes it under the prefix and records it as the latest plan. Only
    Home Assistant itself (post_data) is stood in for."""

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
        web_server.params_secrets["participant_token"] = TOKEN
        (data_path / "params.pkl").write_bytes(pickle.dumps((None, params)))
        auth = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
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
            taken = (r.status_code, await r.get_json())
            # it runs after it is answered: its outcome, once known
            for _ in range(200):
                r = await client.get("/participant/v1/commit/q1", headers=auth)
                if r.status_code == 200:
                    break
                await asyncio.sleep(0.05)
            return taken, (r.status_code, await r.get_json()), answer
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
        taken, (status, out), answer = asyncio.run(run())
    assert taken[0] == 202 and taken[1]["state"] == "running", taken
    assert status == 200 and out["state"] == "accepted" and out["accepted"] is True, out
    assert published and {prefix for _, prefix in published} == {"garage_"}
    plan = plan_store.read(data_path)
    assert plan and len(plan["plan"]) == len(answer["plan_kw"])
    assert pd.Timestamp(plan["plan"][0]["timestamp"]) == pd.Timestamp(horizon["start"])
    assert last_run.read(data_path)["action"] == "naive-mpc-optim"
    # saved as made by that commit: known for its own whatever its record says
    assert plan.get("origin") == "commit:q1"


def test_a_commit_home_assistant_did_not_take_is_partial_and_not_recorded(data_path):
    """publish_data logs an entity Home Assistant refused and goes on; the
    commit is partial (Home Assistant keeps what it took), names the entity,
    and saves and records nothing."""
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
    (state, why), _ = _commit(ec, params, res, published, refuse="p_deferrable0")
    assert state == "partial" and "did not take" in why and "p_deferrable0" in why, why
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
    (state, why), saved = _commit(ec, params, res, published)
    assert state == "accepted", why
    saved.assert_not_called()
    assert published and {prefix for _, prefix in published} == {"garage_"}
    latest = pd.read_csv(data_path / "opt_res_latest.csv", index_col="timestamp")
    assert len(latest) == len(res)
    plan = plan_store.read(data_path)
    assert plan and len(plan["plan"]) == len(res)
    assert pd.Timestamp(plan["plan"][0]["timestamp"]) == res.index[0]
    assert last_run.read(data_path)["action"] == "naive-mpc-optim"


def test_a_commit_that_fails_to_publish_keeps_the_latest_plan(data_path):
    """Nothing is saved or recorded until the plan is published. Failing
    before anything was sent to Home Assistant, it is refused; failing after,
    partial (some of the plan may run)."""
    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    res = _a_plan(ec, params)
    before = (data_path / "opt_res_latest.csv").read_bytes()
    (state, why), _ = _commit(ec, params, res, [], ok=False)
    assert state == "refused" and "nothing was published" in why, why
    assert (data_path / "opt_res_latest.csv").read_bytes() == before
    # publishing raises after Home Assistant took an entity: partial
    published: list = []

    async def fails_part_way(idd, logger, **kwargs):
        await idd["rh"].post_data(None, 0, "sensor.p_deferrable0", publish_prefix="garage_")
        raise RuntimeError("cut off")

    with mock.patch.object(participant_api, "publish_data", fails_part_way):
        (state, why), _ = _commit(ec, params, res, published)
    assert state == "partial" and "part way" in why and published, why
    assert (data_path / "opt_res_latest.csv").read_bytes() == before


def test_a_commit_is_refused_while_continual_publish_runs(data_path):
    """Its loop republishes the entity files of an earlier run, which would
    put that plan back over the committed one."""
    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    params["retrieve_hass_conf"]["continual_publish"] = True
    published: list = []
    (state, why), _ = _commit(ec, params, _a_plan(ec, params), published)
    assert state == "refused" and "continual_publish" in why and not published


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
    (state, why), _ = _commit(ec, params, res, published)
    assert state == "refused" and "before the latest plan" in why and not published


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
    with mock.patch("urllib.request.OpenerDirector.open", return_value=response):
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
    contexts = []
    build = urllib.request.build_opener

    def opener(*handlers):
        contexts.extend(h._context for h in handlers if isinstance(h, urllib.request.HTTPSHandler))
        return build(*handlers)

    with (
        mock.patch("urllib.request.build_opener", side_effect=opener),
        mock.patch("urllib.request.OpenerDirector.open", return_value=response) as opened,
    ):
        assert participant_api.soc_now(conf, logger) == pytest.approx(0.55)
    assert opened.call_args.args[0].full_url == expected
    # the handler's own default context verifies; ssl_no_verify's does not
    assert len(contexts) == 1 and (contexts[0].verify_mode == ssl.CERT_NONE) == no_verify


@pytest.mark.parametrize(
    "base", ["http://ha:8123", "https://ha.example", "http://supervisor.evil/core/api"]
)
def test_the_battery_charge_never_sends_the_supervisor_token_to_another_host(base, monkeypatch):
    """The add-on's Supervisor token as the token for a URL that is not the
    Supervisor: nothing is asked (the token would go to that host), and the
    target stands in."""
    monkeypatch.setenv("SUPERVISOR_TOKEN", "the-supervisor-token")
    conf = {
        "sensor_battery_state_of_charge": "sensor.soc",
        "hass_url": base,
        "long_lived_token": " the-supervisor-token ",
    }
    with mock.patch("urllib.request.OpenerDirector.open") as opened:
        assert participant_api.soc_now(conf, logger) is None
    opened.assert_not_called()


@pytest.mark.parametrize(
    "base, expected",
    [
        ("http://supervisor/core/api", "http://supervisor/core/api/states/sensor.soc"),
        ("HTTP://Supervisor/core/api/", "http://supervisor/core/api/states/sensor.soc"),
        ("https://supervisor/core/api", "https://supervisor/core/api/states/sensor.soc"),
    ],
)
def test_the_battery_charge_asks_the_supervisor_with_its_token(base, expected, monkeypatch):
    """The Supervisor token still reaches the Supervisor, in any form of its
    URL, on the Supervisor's own path."""
    monkeypatch.setenv("SUPERVISOR_TOKEN", "the-supervisor-token")
    response = mock.MagicMock()
    response.__enter__.return_value = io.BytesIO(b'{"state": "55", "attributes": {}}')
    conf = {
        "sensor_battery_state_of_charge": "sensor.soc",
        "hass_url": base,
        "long_lived_token": "the-supervisor-token",
    }
    with mock.patch("urllib.request.OpenerDirector.open", return_value=response) as opened:
        assert participant_api.soc_now(conf, logger) == pytest.approx(0.55)
    request = opened.call_args.args[0]
    assert request.full_url == expected
    assert request.get_header("Authorization") == "Bearer the-supervisor-token"


def test_the_battery_charge_is_not_asked_of_a_redirect():
    """Home Assistant's URL redirecting elsewhere: the redirect is not
    followed (it would carry the token there), and the target stands in."""
    seen = []

    class Other(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"state": "55", "attributes": {}}')

        def log_message(self, *args):
            pass

    other = _raw_server(Other)

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            port = other.server_address[1]
            self.send_header("Location", f"http://127.0.0.1:{port}/api/states/sensor.soc")
            self.end_headers()

        def log_message(self, *args):
            pass

    redirect = _raw_server(Redirect)
    try:
        conf = {
            "sensor_battery_state_of_charge": "sensor.soc",
            "hass_url": f"http://127.0.0.1:{redirect.server_address[1]}",
            "long_lived_token": "t",
        }
        assert participant_api.soc_now(conf, logger) is None
    finally:
        redirect.shutdown()
        other.shutdown()
    assert seen == []


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
    status, body = _client_get(data_path, None, "/participant/v1/describe", token=TOKEN)
    assert status == 503 and "could not be read" in body["detail"]


def test_a_large_body_is_refused_before_it_is_read(data_path):
    """Queries and commits are a few kB: a body over 1 MiB is refused (413)."""

    async def post():
        client = web_server.app.test_client()
        body = b"x" * (web_server.PARTICIPANT_BODY_MAX + 1)
        r = await client.post(
            "/participant/v1/query",
            data=body,
            headers={"Authorization": f"Bearer {TOKEN}", "Content-Length": str(len(body))},
        )
        return r.status_code

    with mock.patch.object(participant_api, "read_params", mock.AsyncMock()) as read:
        assert asyncio.run(post()) == 413
    read.assert_not_called()


def _stream(data_path, chunks: list[bytes], token: str = TOKEN):
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
        web_server.params_secrets["participant_token"] = TOKEN
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

    with (
        mock.patch.object(participant_api, "make_service", return_value=service),
        mock.patch.object(web_server, "_participant_service", None),
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


@pytest.mark.parametrize(
    "window, problem",
    [
        ({"step_minutes": 30, "slots": 7 * 48 + 1}, "1 to 336 slots"),
        ({"step_minutes": 30, "slots": 0}, "1 to 336 slots"),
    ],
)
def test_a_session_refuses_a_horizon_it_cannot_plan(data_path, window, problem):

    ec = _conf(data_path)
    with pytest.raises(WindowRefused, match=problem):
        asyncio.run(
            participant_api.build_session(
                ec, _params(ec, **GARAGE), PlanWindow(**window), logger, get_data_from_file=True
            )
        )


def test_a_session_refuses_slots_it_does_not_forecast(data_path):
    """Asked about slots one step off its own, a remote refuses rather than
    plan them with the neighbouring slots' forecasts."""

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
    assert "application/json" in path["post"]["requestBody"]["content"]
    # a commit's 429 and lock-unavailable 503 are retryable: the client is told when
    for code in ("429", "503"):
        assert "Retry-After" in path["post"]["responses"][code]["headers"]
    # the lock unavailable refuses a commit (409): nothing ran
    assert "lock" in path["post"]["responses"]["409"]["description"]
    assert "lock" not in path["post"]["responses"]["503"]["description"]


# --------------------------------------------------------------------------
# The wire format: EMHASS's own schemas, and fixed bodies another solver sends
# --------------------------------------------------------------------------

SCHEMAS = root / "src/emhass/static/participant-api"


def _fits(body: dict, schema: dict) -> None:
    """`body` has every field `schema` requires, and no other (the schemas
    allow no additional properties)."""
    assert set(schema["required"]) <= set(body), set(schema["required"]) - set(body)
    assert set(body) <= set(schema["properties"]), set(body) - set(schema["properties"])


def test_queries_and_answers_are_the_published_schemas():
    """What EMHASS sends and answers is what device-query.v1.json and
    device-answer.v1.json describe, at their version."""
    from emhass.coordinator.interface import Answer, answer_to_dict, query_to_dict

    query_schema = json.loads((SCHEMAS / "device-query.v1.json").read_text(encoding="utf-8"))
    answer_schema = json.loads((SCHEMAS / "device-answer.v1.json").read_text(encoding="utf-8"))
    prices = np.full(4, 0.2)
    for q in (
        Query("price_response", prices, prices),
        Query("best_response", prices, prices, residual_kw=np.ones(4)),
    ):
        body = query_to_dict(q, 30.0, "2026-11-15T00:00:00+00:00")
        _fits(body, query_schema)
        assert body["version"] == SCHEMA_VERSION
    answer = answer_to_dict(Answer(np.zeros(4), 0.0, np.zeros(5)), solver="emhass")
    _fits(answer, answer_schema)
    spec = json.loads((SCHEMAS / "participant-api.v1.json").read_text(encoding="utf-8"))
    assert "home_energy_optimizer" not in json.dumps(spec)


def test_a_fixed_body_from_another_solver_is_read_as_written():
    """A solver that is not EMHASS sends these bodies, written by hand from
    the schemas: they are read field for field, and a body of another
    version is refused rather than misread."""
    from emhass.coordinator.interface import answer_from_dict, query_from_dict

    query = {
        "version": SCHEMA_VERSION,
        "kind": "price_response",
        "horizon": {"step_minutes": 30.0, "slots": 3, "start": "2026-11-15T00:00:00+00:00"},
        "price_draw": [0.1, 0.2, 0.3],
        "price_supply": [0.05, 0.05, 0.05],
    }
    q = query_from_dict(query)
    assert q.kind == "price_response"
    assert list(q.price_draw) == [0.1, 0.2, 0.3] and list(q.price_supply) == [0.05] * 3
    answer = {
        "version": SCHEMA_VERSION,
        "plan_kw": [1.0, 0.0, -1.0],
        "private_cost": 0.25,
        "trajectory": [5.0, 5.5, 5.5, 4.5],
        "status": "ok",
        "solver": "someone-elses-optimizer",
    }
    a = answer_from_dict(answer)
    assert list(a.plan_kw) == [1.0, 0.0, -1.0] and a.private_cost == 0.25
    assert list(a.trajectory) == [5.0, 5.5, 5.5, 4.5] and a.status == "ok"
    # The service refuses a query of another version (400), never misreads it
    service = ParticipantService(lambda: {}, lambda window: None, token=TOKEN)
    status, _ = service.handle(
        "POST",
        "/v1/query",
        {"Authorization": f"Bearer {TOKEN}", "Query-Id": "q1"},
        json.dumps({**query, "version": SCHEMA_VERSION + 1}).encode(),
    )
    assert status == 400


class TestTheParticipantApiHasItsOwnToken(AuthSupportBase):
    """With the server's API token on (GHSA-c39q-j4r5-9xq5), the participant
    route answers to participant_token alone - one Authorization header
    carries one token - while the other routes still need the API token."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.enable()
        web_server.params_secrets["participant_token"] = TOKEN

    async def asyncTearDown(self):
        web_server.params_secrets.pop("participant_token", None)
        await super().asyncTearDown()

    async def test_its_own_token_reaches_the_route_and_the_api_token_does_not(self):
        path = "/participant/v1/describe"
        # participant_token, without the API token: past the hook, to the route
        # (no params.pkl here, so the API is off: 404)
        own = await self.send("GET", path, headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(own.status_code, 404)
        # the API token is not participant_token: the route's own refusal
        api = await self.send("GET", path, headers=API_BEARER)
        self.assertEqual(api.status_code, 401)
        self.assertIn("participant_token", (await api.get_json())["detail"])
        # and participant_token opens nothing else
        other = await self.send("GET", "/api/v1/plan", headers={"Authorization": f"Bearer {TOKEN}"})
        self.assertEqual(other.status_code, 401)
        # while the API token opens it, as before
        plan = await self.send("GET", "/api/v1/plan", headers=API_BEARER)
        self.assertNotEqual(plan.status_code, 401)


# --------------------------------------------------------------------------
# The coordinator's client against a remote that misbehaves (#1372 review)
# --------------------------------------------------------------------------


def _raw_server(handler_cls):
    """A local HTTP server for `handler_cls`, from a daemon thread."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_a_redirect_is_not_followed_with_the_token():
    """A remote that redirects is answered as it answered (an HTTP error):
    the bearer token never reaches a host the configuration did not name."""
    from emhass.coordinator.remote import RemoteError, RemoteParticipant

    seen = []

    class Other(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    other = _raw_server(Other)

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(307)
            self.send_header("Location", f"http://127.0.0.1:{other.server_address[1]}/steal")
            self.end_headers()

        def log_message(self, *args):
            pass

    redirect = _raw_server(Redirect)
    try:
        with pytest.raises(RemoteError, match="HTTP 307"):
            RemoteParticipant(
                "r",
                f"http://127.0.0.1:{redirect.server_address[1]}/participant",
                PlanWindow(30, 48),
                token="garage-secret",
                timeout=5,
            )
    finally:
        redirect.shutdown()
        other.shutdown()
    assert seen == []


@pytest.mark.parametrize("answer", ["slow", "slow headers", "large"])
def test_a_remote_answer_is_bounded_in_time_and_size(answer):
    """The timeout bounds the whole call, not each read: a remote dripping
    its headers or its answer is cut off, as is an answer over the size cap."""
    import time as clock

    from emhass.coordinator import remote

    class Bad(BaseHTTPRequestHandler):
        def do_GET(self):
            if answer == "slow headers":
                self.wfile.write(b"HTTP/1.0 200 OK\r\n")
                for byte in b"X-Pad: " + b"x" * 40 + b"\r\n\r\n{}":
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    clock.sleep(0.2)
                return
            self.send_response(200)
            self.end_headers()
            if answer == "slow":
                for byte in b'{"version": 1, "kinds": []}':
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    clock.sleep(0.2)
            else:  # a valid description, padded past the cap
                pad = "x" * ((1 << 20) + 10)  # past a 1 MiB cap
                self.wfile.write(
                    (
                        '{"version": 1, "kinds": ["price_response"], "max_power_kw": 1.0, '
                        f'"pad": "{pad}"}}'
                    ).encode()
                )

        def log_message(self, *args):
            pass

    server = _raw_server(Bad)
    try:
        t0 = clock.perf_counter()
        with pytest.raises(remote.RemoteError):
            remote.RemoteParticipant(
                "r", f"http://127.0.0.1:{server.server_address[1]}", PlanWindow(30, 2), timeout=1.0
            )
        assert clock.perf_counter() - t0 < 3.0
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    "url, token, trusted, refused",
    [
        ("http://garage.lan:5000/participant", "t", False, True),
        ("http://garage.lan:5000/participant", "t", True, False),  # opted in
        ("https://garage.lan:5000/participant", "t", False, False),
        ("http://127.0.0.1:5000/participant", "t", False, False),
        ("http://localhost:5000/participant", "t", False, False),
        ("http://garage.lan:5000/participant", None, False, False),  # no token to send
    ],
)
def test_a_token_goes_in_the_clear_only_on_a_trusted_network(url, token, trusted, refused):
    """A bearer token over plain http:// to another machine can be read on the
    network: refused (the remote is not called at all) unless its solver is
    marked trusted_network. https://, this machine, or no token need no mark."""
    from emhass.coordinator import remote

    called = []

    def transport(method, url, headers, data):
        called.append(url)
        return 200, b'{"version": 1, "kinds": ["price_response"], "max_power_kw": 1.0}'

    def build():
        return remote.RemoteParticipant(
            "r", url, PlanWindow(30, 2), token=token, transport=transport, trusted_network=trusted
        )

    if refused:
        with pytest.raises(remote.PlainHttpRefused, match="trusted_network"):
            build()
        assert called == []
    else:
        build()
        assert len(called) == 1


def test_a_remote_over_plain_http_is_declined_unless_trusted(data_path, monkeypatch):
    """The coordinator plans the house without a remote it would send its
    token to in the clear, and says why; marked trusted_network, it is
    planned with it."""
    from emhass.coordinator import remote

    ec = _conf(data_path)
    params = _params(ec, **GARAGE)

    def build(window):
        return asyncio.run(
            participant_api.build_session(ec, params, window, logger, get_data_from_file=True)
        )

    def describe():
        return emhass_description(build(PlanWindow(30, 48)))

    # the test server is on this machine: taken as another for the check
    monkeypatch.setattr(remote, "_loopback", lambda url: False)
    server = serve(ParticipantService(describe, build, lambda p, a: (True, ""), token=TOKEN))
    try:
        house = _house(_now())
        rh = house[0]
        house = (
            {**rh, "remote_tokens": {"garage_token": TOKEN}},
            *house[1:],
        )
        solver = {"url": server.url, "token_secret": "garage_token"}
        site = [{"id": "garage", "parent": "grid", "solver": solver}]
        res = _coordinate(house, site)
        assert (res["backend_used"] == "dantzig_wolfe").all()
        status = res["fed_remote_status_garage"].iloc[0]
        assert status.startswith("declined") and "trusted_network" in status, status
        assert "fed_remote_power_garage" not in res.columns
        site[0]["solver"] = {**solver, "trusted_network": True}
        res = _coordinate(house, site)
        assert (res["fed_remote_status_garage"] == "committed").all()
    finally:
        server.shutdown()


def test_trusted_network_is_a_flag_of_a_remote_solver():
    """trusted_network is true or false, on a remote solver only."""
    from emhass.coordinator import site as site_mod

    def site(solver):
        return [{"id": "garage", "parent": "grid", "solver": solver}]

    url = "http://garage.lan:5000/participant"
    assert site_mod.site_problem(site({"url": url, "trusted_network": True})) is None
    assert site_mod.site_problem(site({"url": url, "trusted_network": False})) is None
    assert site_mod.site_problem(site({"url": url, "trusted_network": "yes"})) is not None


_RUNNING = {"commit_id": "q", "state": "running", "accepted": None}


def _ended(state):
    return {"commit_id": "q", "state": state, "accepted": None, "reason": state}


@pytest.mark.parametrize(
    "script, expected",
    [
        # (method, status, body): what the remote answers, call by call
        ([("POST", 202, _RUNNING), ("GET", 200, _ended("accepted"))], True),
        ([("POST", 202, _RUNNING), ("GET", 202, _RUNNING), ("GET", 200, _ended("refused"))], False),
        ([("POST", 202, _RUNNING), ("GET", 200, _ended("partial"))], None),  # some may run
        # no answer to the commit: asked about, not sent again
        ([("POST", 0, None), ("GET", 200, _ended("accepted"))], True),
        # no answer, and the remote never had it: sent again, once
        (
            [("POST", 0, None), ("GET", 404, None), ("POST", 202, _RUNNING)]
            + [("GET", 200, _ended("accepted"))],
            True,
        ),
        ([("POST", 429, None)], False),  # busy: refused, nothing ran
        ([("POST", 409, None)], False),
        # sent again, and that refused (busy with the first, perhaps): asked
        # about again, never a refusal - here the first had run
        (
            [("POST", 0, None), ("GET", 404, None), ("POST", 429, None)]
            + [("GET", 200, _ended("accepted"))],
            True,
        ),
        ([("POST", 0, None), ("GET", 404, None), ("POST", 409, None), ("GET", 404, None)], None),
        # a gateway's error, its own failure, or a 2xx that is no record: it
        # may have taken the commit - asked about, never a refusal
        ([("POST", 504, None), ("GET", 200, _ended("accepted"))], True),
        (
            [("POST", 502, None), ("GET", 404, None), ("POST", 202, _RUNNING)]
            + [("GET", 200, _ended("refused"))],
            False,
        ),
        ([("POST", 500, None), ("GET", 0, None)], None),
        ([("POST", 202, {"what": "?"}), ("GET", 200, _ended("accepted"))], True),
        ([("POST", 200, {"accepted": "yes"}), ("GET", 0, None)], None),
        ([("POST", 200, {"accepted": True})], True),  # a participant of the first draft
        ([("POST", 0, None), ("GET", 0, None)], None),  # no outcome in time
        ([("POST", 202, _RUNNING), ("GET", 202, _RUNNING)], None),  # still running
    ],
)
def test_a_commits_outcome_is_asked_for_until_it_is_known(script, expected):
    """A commit is answered at once and run after: the coordinator asks for
    its outcome until it is known, or the timeout passes (None: not known, so
    its part stays in the plan). A commit that got no answer is asked about,
    and sent again only if the remote never had it."""
    from emhass.coordinator.remote import RemoteParticipant

    steps = list(script)
    seen = []

    def transport(method, url, headers, data):
        if url.endswith("/v1/describe"):
            return 200, (
                b'{"version": 1, "kinds": ["price_response"], "max_power_kw": 1.0, '
                b'"features": ["commit_status"]}'
            )
        want, status, body = steps.pop(0) if len(steps) > 1 else steps[0]
        assert method == want, (method, url, want)
        seen.append((method, url.rsplit("/v1/", 1)[1]))
        if status == 0:
            raise TimeoutError("no answer")
        return status, json.dumps(body or {"title": "x", "status": status}).encode()

    rp = RemoteParticipant(
        "r", "http://127.0.0.1:1", PlanWindow(30, 2, "x"), transport=transport, timeout=0.5
    )
    ok, why = rp.commit({"query_id": "q"}, np.zeros(2))
    assert ok is expected, (ok, why, seen)
    assert seen[0] == ("POST", "commit")
    assert [m for m, _ in seen].count("POST") <= 2  # never sent more than twice
    assert all(path in ("commit", "commit/q") for _, path in seen)


def test_a_participant_without_commit_status_is_never_asked_about_a_commit():
    """A participant of the first draft has no commit record: a commit it
    did not answer is not known (it may run), and is neither asked about nor
    sent again."""
    from emhass.coordinator.remote import RemoteParticipant

    seen = []

    def transport(method, url, headers, data):
        if url.endswith("/v1/describe"):
            return 200, b'{"version": 1, "kinds": ["price_response"], "max_power_kw": 1.0}'
        seen.append((method, url))
        raise TimeoutError("no answer")

    rp = RemoteParticipant("r", "http://127.0.0.1:1", PlanWindow(30, 2, "x"), transport=transport)
    ok, _ = rp.commit({"query_id": "q"}, np.zeros(2))
    assert ok is None and len(seen) == 1


def test_a_commit_is_recorded_before_it_takes_the_slot():
    """From the moment it may run, a commit is asked about as running, never
    as unknown (404): the record is made before the slot is taken, and
    dropped again if the slot is busy (nothing ran)."""
    service = None
    seen = []

    def begin():
        seen.append(service.handle("GET", "/v1/commit/q1", {}, None))
        return "busy", None

    service = _service(lambda p, a: ("accepted", ""), begin=begin)
    assert service.handle("POST", "/v1/commit", {}, _commit_body())[0] == 429
    assert seen[0][0] == 202 and seen[0][1]["state"] == "running"
    assert service.handle("GET", "/v1/commit/q1", {}, None)[0] == 404  # dropped


def test_a_call_past_its_deadline_never_sends(monkeypatch):
    """A call still connecting when its deadline passes (a slow name lookup,
    a slow connection) sends nothing when it connects later: a commit cannot
    arrive after the coordinator has given up on it."""
    import http.client
    import socket as sockets
    import time as clock

    from emhass.coordinator.remote import urllib_transport

    received = []

    class Count(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.path)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = _raw_server(Count)
    connect = sockets.create_connection

    def slow(*args, **kwargs):
        clock.sleep(0.6)  # past the 0.2 s deadline
        return connect(*args, **kwargs)

    monkeypatch.setattr(http.client.socket, "create_connection", slow)
    try:
        send = urllib_transport(0.2)
        with pytest.raises(TimeoutError):
            send("POST", f"http://127.0.0.1:{server.server_address[1]}/v1/commit", {}, b"{}")
        clock.sleep(1.0)  # the call has connected by now
        assert received == []
    finally:
        server.shutdown()


def test_a_horizon_is_answered_only_while_its_first_slot_is_current():
    """A session built for a horizon is not answered from once its first
    slot has passed: baseline, query and a new commit are refused (409),
    while a commit already made is still reported."""
    from emhass.coordinator.remote import WindowRefused

    now = {"start": _HORIZON["start"]}

    def current(window):
        if window.start != now["start"]:
            raise WindowRefused("the coordinator and this EMHASS must agree on the slot")

    service = ParticipantService(
        lambda: {"key": "fixed", "max_power_kw": 1.0},
        lambda window: _Fixed(),
        lambda p, a: ("accepted", ""),
        current=current,
    )
    body = json.dumps(_HORIZON).encode()
    query = json.dumps(
        {
            "version": SCHEMA_VERSION,
            "kind": "price_response",
            "horizon": _HORIZON,
            "price_draw": [0.1, 0.2],
            "rho": 0.0,
        }
    ).encode()
    assert service.handle("POST", "/v1/baseline", {"Query-Id": "q1"}, body)[0] == 200
    assert service.handle("POST", "/v1/baseline", {"Query-Id": "q2"}, body)[0] == 200
    assert service.handle("POST", "/v1/query", {"Query-Id": "q2"}, query)[0] == 200
    assert service.handle("POST", "/v1/commit", {}, _commit_body(qid="q1"))[0] == 202
    now["start"] = "2026-11-21T10:30:00+00:00"  # the next slot
    assert service.handle("POST", "/v1/baseline", {"Query-Id": "q3"}, body)[0] == 409
    assert service.handle("POST", "/v1/query", {"Query-Id": "q2"}, query)[0] == 409
    assert service.handle("POST", "/v1/commit", {}, _commit_body(qid="q2"))[0] == 409
    assert service.handle("POST", "/v1/commit", {}, _commit_body(qid="q1"))[0] in (200, 202)


def test_emhass_checks_a_horizon_against_its_own_current_slot(data_path):
    """EMHASS's check rounds its own clock as a run does: a horizon starting
    on the current slot passes, the one before is refused."""
    import time_machine as travel

    from emhass.coordinator.remote import WindowRefused

    params = _params(_conf(data_path), **GARAGE)
    check = participant_api.current_slot_check(params)
    with travel.travel(pd.Timestamp("2026-11-21T10:07:00Z").to_pydatetime(), tick=False):
        now = _now()
        check(PlanWindow(30, 48, now.isoformat()))
        with pytest.raises(WindowRefused, match="agree on the slot"):
            check(PlanWindow(30, 48, (now - pd.Timedelta(minutes=30)).isoformat()))


@pytest.mark.parametrize(
    "power",
    [
        "",
        '"max_power_kw": null, ',
        '"max_power_kw": Infinity, ',
        '"max_power_kw": true, ',
        '"max_power_kw": 18446744073709551616, ',  # numpy cannot test it as an integer
        '"max_power_kw": 1' + "0" * 400 + ", ",  # too large for a float
        '"max_power_kw": 1e9, ',  # absurd for a site
    ],
)
def test_a_description_without_a_finite_power_is_refused(power):
    """max_power_kw bounds the coordinator's meter: missing, null, infinite
    or not a number, the remote cannot be planned with (RemoteError, so the
    house is planned without it)."""
    from emhass.coordinator.remote import RemoteError, RemoteParticipant

    def transport(method, url, headers, data):
        return 200, ('{"version": 1, ' + power + '"kinds": ["price_response"]}').encode()

    with pytest.raises(RemoteError, match="max_power_kw"):
        RemoteParticipant("r", "http://r", PlanWindow(30, 2), transport=transport)


def test_the_description_is_built_once_per_service(data_path):
    """Every coordinator run asks for the description: it is built once for
    the service (its params do not change while it lives), not every time."""
    params = _params(_conf(data_path), **GARAGE)
    built = []

    async def build(*args, **kwargs):
        built.append(args[2])
        return mock.MagicMock(battery=False, loads=[0], key="garage", max_power_kw=7.0, onoff=True)

    with mock.patch.object(participant_api, "build_session", build):
        service = participant_api.make_service(_conf(data_path), params, "t", logger)
        first = service.handle("GET", "/v1/describe", {"Authorization": "Bearer t"}, None)
        again = service.handle("GET", "/v1/describe", {"Authorization": "Bearer t"}, None)
    assert first == again and first[0] == 200 and len(built) == 1


@pytest.mark.parametrize(
    "field, value", [("private_cost", "NaN"), ("trajectory", "[0, Infinity, 0]")]
)
def test_an_answer_that_is_not_finite_makes_the_remote_unavailable(field, value):
    """A NaN cost or an infinite state would break the coordinator's LP for
    the whole house: such an answer is no plan, and the remote is planned
    without."""
    from emhass.coordinator.remote import RemoteParticipant, RemoteUnavailable

    body = {"plan_kw": "[0, 0]", "private_cost": "0", "trajectory": "[0, 0, 0]"}
    body[field] = value
    answer = (
        '{"version": 1, "status": "ok", ' + ", ".join(f'"{k}": {v}' for k, v in body.items()) + "}"
    ).encode()

    def transport(method, url, headers, data):
        if url.endswith("/v1/describe"):
            return 200, b'{"version": 1, "kinds": ["price_response"], "max_power_kw": 1.0}'
        return 200, answer

    rp = RemoteParticipant("r", "http://x", PlanWindow(30, 2), transport=transport)
    with pytest.raises(RemoteUnavailable, match="not a finite plan"):
        rp.respond(Query("price_response", np.ones(2), np.ones(2)))


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"is_electric_load": [False]}, "non-electric"),
        ({"set_deferrable_startup_penalty": [1.0]}, "startup_penalty"),
    ],
)
def test_an_emhass_declines_to_serve_what_the_coordinator_declines(data_path, change, reason):
    """An EMHASS serving its devices refuses (409, "declined" at the
    coordinator) what the coordinator would decline in its own house: its
    answers would not be what the coordinator can coordinate."""
    ec = _conf(data_path)
    params = _params(ec, **{**GARAGE, **change})
    with pytest.raises(WindowRefused, match=reason):
        asyncio.run(
            participant_api.build_session(
                ec, params, PlanWindow(30, 48), logger, get_data_from_file=True
            )
        )


# --------------------------------------------------------------------------
# Versioning: the handshake can grow without breaking either side
# --------------------------------------------------------------------------


def test_the_participant_says_its_version_and_features():
    """describe and every commit record carry the major version; describe
    lists what the participant serves beyond the first draft (features)."""
    from emhass.coordinator.interface import FEATURES

    service = _service(lambda p, a: ("accepted", ""))
    _, described = service.handle("GET", "/v1/describe", {}, None)
    assert described["version"] == SCHEMA_VERSION
    assert set(FEATURES) <= set(described["features"]) and "commit_status" in FEATURES
    assert service.handle("POST", "/v1/commit", {}, _commit_body())[1]["version"] == SCHEMA_VERSION
    assert _until_final(service)[1]["version"] == SCHEMA_VERSION


def _scripted(commit_body=None, answer=None, describe=None):
    """A RemoteParticipant whose remote answers describe, a query and a commit
    as given (dicts, sent as JSON)."""
    from emhass.coordinator.remote import RemoteParticipant

    d = describe or {"version": 1, "kinds": ["price_response"], "max_power_kw": 1.0}

    def transport(method, url, headers, data):
        if url.endswith("/v1/describe"):
            return 200, json.dumps(d).encode()
        if "/v1/commit" in url:
            return 200, json.dumps(commit_body).encode()
        return 200, json.dumps(answer).encode()

    return RemoteParticipant(
        "r", "http://127.0.0.1:1", PlanWindow(30, 2, "x"), transport=transport, timeout=5.0
    )


def test_a_newer_participant_is_read_as_far_as_it_is_understood():
    """Within a major version the API only grows: a coordinator ignores fields
    and features it does not know, and treats a commit state it does not know
    as not known - at once, not after waiting for one it does know."""
    import time as clock

    newer = {
        "version": 1,
        "kinds": ["price_response", "some_future_kind"],
        "max_power_kw": 1.0,
        "features": ["commit_status", "some_future_feature"],
        "some_future_field": {"x": 1},
    }
    answer = {
        "version": 1,
        "plan_kw": [1.0, 0.0],
        "private_cost": 0.5,
        "trajectory": [0.0, 0.5, 0.5],
        "status": "ok",
        "some_future_field": [1, 2],
    }
    rp = _scripted(answer=answer, describe=newer)
    assert rp.features == {"commit_status", "some_future_feature"}
    assert rp.kinds == ("price_response",)
    assert rp.baseline().plan_kw.tolist() == [1.0, 0.0]
    record = {"version": 1, "commit_id": "q", "state": "superseded", "some_future_field": 1}
    t0 = clock.monotonic()
    ok, why = _scripted(commit_body=record).commit({"query_id": "q"}, np.zeros(2))
    assert ok is None and "superseded" in why and clock.monotonic() - t0 < 1.0


def test_another_major_version_is_never_misread():
    """A description, an answer or a commit record of another major version
    is not read as this one: the remote is left out, its answer is no plan,
    and a commit's outcome is not known."""
    from emhass.coordinator.remote import RemoteError, RemoteUnavailable

    with pytest.raises(RemoteError, match="version"):
        _scripted(describe={"version": 2, "kinds": ["price_response"], "max_power_kw": 1.0})
    answer = {"version": 2, "plan_kw": [1.0, 0.0], "private_cost": 0.0, "trajectory": [0, 0, 0]}
    with pytest.raises(RemoteUnavailable):
        _scripted(answer=answer).baseline()
    record = {"version": 2, "commit_id": "q", "state": "accepted", "accepted": True}
    ok, why = _scripted(commit_body=record).commit({"query_id": "q"}, np.zeros(2))
    assert ok is None and "version" in why


def test_requests_are_closed_and_answers_open_to_new_fields():
    """The schemas' versioning rule: every path is below /v1/, info.version
    is the minor version of major 1; what a participant answers may gain
    fields (open), what it is asked may not without a feature (closed)."""
    base = root / "src/emhass/static/participant-api"
    api = json.loads((base / "participant-api.v1.json").read_text(encoding="utf-8"))
    assert all(path.startswith("/v1/") for path in api["paths"])
    major, minor, _ = api["info"]["version"].split("-")[0].split(".")
    assert major == str(SCHEMA_VERSION) and int(minor) >= 1
    schemas = api["components"]["schemas"]
    for name in ("Description", "CommitResult"):
        assert schemas[name].get("additionalProperties", True) is not False, name
    for name in ("Commit", "Horizon"):
        assert schemas[name]["additionalProperties"] is False, name
    answer = json.loads((base / "device-answer.v1.json").read_text(encoding="utf-8"))
    query = json.loads((base / "device-query.v1.json").read_text(encoding="utf-8"))
    assert answer.get("additionalProperties", True) is not False
    assert query["additionalProperties"] is False
    assert schemas["CommitResult"]["properties"]["version"] == {"const": SCHEMA_VERSION}


# --------------------------------------------------------------------------
# A remote runs what it last took until a new plan replaces it (blind review)
# --------------------------------------------------------------------------


def _garage_service(ec, begin=None):
    params = _params(ec, **GARAGE)

    def build(window):
        return asyncio.run(
            participant_api.build_session(ec, params, window, logger, get_data_from_file=True)
        )

    def describe():
        return emhass_description(build(PlanWindow(30, 48)))

    return ParticipantService(describe, build, lambda p, a: ("accepted", ""), begin=begin)


def _balanced(res, ev):
    balance = res["P_PV"] + res["P_grid"] + res["P_batt"]
    used = res["P_Load"] + res["P_deferrable0"] + res["P_deferrable1"] + ev
    return np.allclose(balance, used, atol=1e-3)


def test_a_remote_that_refuses_this_run_is_held_to_the_plan_it_runs(data_path):
    """Run one commits the garage's EV. Run two, in the same slot, the
    garage is busy (429): it did not take the new plan, but Home Assistant
    still has the first, so the house is planned around what is left of it,
    never as if the EV were off."""
    ec = _conf(data_path)
    busy = {"on": False}
    service = _garage_service(ec, begin=lambda: ("busy", None) if busy["on"] else ("ok", None))
    server = serve(service)
    try:
        site = [{"id": "garage", "parent": "grid", "solver": {"url": server.url}}]
        house = _house(_now())
        first = _coordinate(house, site)
        assert (first["fed_remote_status_garage"] == "committed").all()
        took = first["fed_remote_power_garage"].values
        busy["on"] = True
        second = _coordinate(house, site)
        status = second["fed_remote_status_garage"].iloc[0]
        assert status.startswith("refused") and "held" in status, status
        assert np.allclose(second["fed_remote_power_garage"].values, took)
        assert _balanced(second, second["fed_remote_power_garage"])
    finally:
        server.shutdown()


def test_an_unreachable_remote_is_held_to_the_plan_it_was_last_committed_to(data_path):
    """The coordinator keeps what it committed each remote to: one it cannot
    reach next run is held to what is left of that plan."""
    ec = _conf(data_path)
    server = serve(_garage_service(ec))
    site = [{"id": "garage", "parent": "grid", "solver": {"url": server.url}}]
    house = _house(_now())
    try:
        first = _coordinate(house, site)
        assert (first["fed_remote_status_garage"] == "committed").all()
        took = first["fed_remote_power_garage"].values
    finally:
        server.shutdown()
        server.server_close()
    second = _coordinate(house, site)
    status = second["fed_remote_status_garage"].iloc[0]
    assert status.startswith("unreachable") and "held" in status, status
    assert np.allclose(second["fed_remote_power_garage"].values, took)
    assert _balanced(second, second["fed_remote_power_garage"])


def test_only_a_plan_that_starts_now_is_live():
    """The coordinator commits remotes only for a plan that starts at its
    current slot, as EMHASS rounds its clock: a backtest or another day's
    plan is not run."""
    import time_machine as travel

    from emhass.coordinator.backend import _starts_now

    rh = _confs()[0]
    opt = mock.MagicMock(retrieve_hass_conf=rh, time_step=0.5)
    with travel.travel(pd.Timestamp("2026-11-21T10:07:00Z").to_pydatetime(), tick=False):
        now = _now()
        assert _starts_now(opt, pd.DataFrame(index=pd.date_range(now, periods=4, freq="30min")))
        for other in (now + pd.Timedelta(days=1), now - pd.Timedelta(days=30)):
            frame = pd.DataFrame(index=pd.date_range(other, periods=4, freq="30min"))
            assert not _starts_now(opt, frame)


def test_a_plan_that_is_not_live_commits_no_remote(data_path, monkeypatch):
    """Planned with the remotes, but none is asked to run it."""
    from emhass.coordinator import backend

    monkeypatch.setattr(backend, "_starts_now", lambda opt, data_opt: False)
    ran = []
    service = _garage_service(_conf(data_path))
    service._commit = lambda p, a: (ran.append(1), ("accepted", ""))[1]
    server = serve(service)
    try:
        site = [{"id": "garage", "parent": "grid", "solver": {"url": server.url}}]
        res = _coordinate(_house(_now()), site)
        assert res["fed_remote_status_garage"].iloc[0].startswith("planned without (not live")
        assert ran == []
        # not asked at all, and no power it was never asked to run is published
        assert "fed_remote_power_garage" not in res.columns
    finally:
        server.shutdown()


def test_a_commit_that_raises_is_not_known(data_path, monkeypatch):
    """Whatever fails while committing, the commit may have been sent: its
    remote is uncertain, its part kept, never planned around as refused."""
    from emhass.coordinator.remote import RemoteParticipant

    def raises(self, detail, plan_kw):
        raise RecursionError("deeply nested")

    monkeypatch.setattr(RemoteParticipant, "commit", raises)
    server = serve(_garage_service(_conf(data_path)))
    try:
        site = [{"id": "garage", "parent": "grid", "solver": {"url": server.url}}]
        res = _coordinate(_house(_now()), site)
        assert res["fed_remote_status_garage"].iloc[0].startswith("uncertain")
        assert "fed_remote_power_garage" in res.columns
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    "answer",
    [
        {"private_cost": 10**400},  # too large for a float
        {"plan_kw": [5.0, 0.0]},  # past its max_power_kw of 1 kW
        {"trajectory": [0.0, 1e12, 0.0]},
    ],
)
def test_an_answer_out_of_bounds_leaves_out_its_remote_not_the_house(answer):
    """An answer it cannot use makes that remote unavailable (or falls back to
    its last good answer); it never escapes to fail the whole plan."""
    from emhass.coordinator.remote import RemoteUnavailable

    body = {
        "version": 1,
        "plan_kw": [1.0, 0.0],
        "private_cost": 0.0,
        "trajectory": [0.0, 0.5, 0.5],
        "status": "ok",
        **answer,
    }
    rp = _scripted(answer=json.loads(json.dumps(body)))
    with pytest.raises(RemoteUnavailable):
        rp.baseline()


def test_a_description_it_cannot_read_is_a_remote_error():
    from emhass.coordinator.remote import RemoteError

    describe = {"version": 1, "kinds": ["price_response"], "max_power_kw": 1.0, "devices": 5}
    with pytest.raises(RemoteError, match="cannot read"):
        _scripted(describe=describe)


def test_no_proxy_from_the_environment_sees_a_token(monkeypatch):
    """HTTP_PROXY is not used by the participant API's client or by soc_now:
    a proxy would receive the bearer token, even for this machine."""
    from emhass.coordinator.remote import urllib_transport

    got = []

    class Echo(BaseHTTPRequestHandler):
        def do_GET(self):
            got.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = _raw_server(Echo)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")  # nothing listens there
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    try:
        send = urllib_transport(5.0)
        url = f"http://127.0.0.1:{server.server_address[1]}/v1/describe"
        assert send("GET", url, {"Authorization": "Bearer t"}, None)[0] == 200
        assert got == ["Bearer t"]  # straight to the participant
    finally:
        server.shutdown()
    built = []
    build = urllib.request.build_opener

    def spy(*handlers):
        built.extend(handlers)
        return build(*handlers)

    conf = {
        "sensor_battery_state_of_charge": "sensor.soc",
        "hass_url": "http://ha",
        "long_lived_token": "t",
    }
    with (
        mock.patch("urllib.request.build_opener", side_effect=spy),
        mock.patch("urllib.request.OpenerDirector.open", side_effect=OSError("no")),
    ):
        participant_api.soc_now(conf, logger)
    proxies = [h for h in built if isinstance(h, urllib.request.ProxyHandler)]
    assert proxies and all(h.proxies == {} for h in proxies)


def test_a_commit_that_cannot_start_leaves_no_record_and_frees_the_slot(monkeypatch):
    """The host fails to take its slot, or no thread can be started: the
    commit never ran, so it is not recorded (it may be sent again) and the
    slot is not held."""

    def begin_fails():
        raise OSError("disk")

    service = _service(lambda p, a: ("accepted", ""), begin=begin_fails)
    assert service.handle("POST", "/v1/commit", {}, _commit_body())[0] == 409  # nothing ran
    assert service.handle("GET", "/v1/commit/q1", {}, None)[0] == 404
    released = []
    service = _service(
        lambda p, a: ("accepted", ""), begin=lambda: ("ok", lambda: released.append(1))
    )

    def no_thread(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", no_thread)
    assert service.handle("POST", "/v1/commit", {}, _commit_body())[0] == 409
    assert service.handle("GET", "/v1/commit/q1", {}, None)[0] == 404 and released == [1]


def test_a_commit_runs_in_a_thread_the_process_waits_for():
    """A worker that is stopped (a recycled web worker) finishes publishing
    before it exits: the commit's thread is not a daemon."""
    seen = []
    service = _service(
        lambda p, a: (seen.append(threading.current_thread().daemon), ("accepted", ""))[1]
    )
    service.handle("POST", "/v1/commit", {}, _commit_body())
    _until_final(service)
    assert seen == [False]


@pytest.mark.parametrize(
    "change",
    [{"kind": None}, {"price_draw": None}, {"rho": float("nan")}, {"rho": -1.0}],
)
def test_a_query_it_cannot_answer_is_a_bad_request(change):
    """A query with no kind, a price response with no price, a rho that is
    not a finite number of at least 0: 400, never the model's failure (503)."""
    service = _service(lambda p, a: ("accepted", ""))
    query = {
        "version": SCHEMA_VERSION,
        "kind": "price_response",
        "horizon": _HORIZON,
        "price_draw": [0.1, 0.2],
        "price_supply": [0.0, 0.0],
    }
    for k, v in change.items():
        if v is None:
            query.pop(k)
        else:
            query[k] = v
    body = json.dumps(query).encode()
    status, problem = service.handle("POST", "/v1/query", {"Query-Id": "x"}, body)
    assert status == 400, problem


# --------------------------------------------------------------------------
# What a remote runs between commits (second blind review)
# --------------------------------------------------------------------------


def _record(plan, start="2026-11-21T10:00:00+00:00", step=30.0, state="accepted"):
    return {
        "state": state,
        "horizon": {"step_minutes": step, "slots": len(plan), "start": start},
        "plan_kw": plan,
    }


def _at(start, slots=4):
    return PlanWindow(30.0, slots, start)


def test_a_stepping_remote_runs_what_is_left_of_its_plan_then_its_last_value():
    """A remote that steps through its committed plan (plan_steps) runs, a
    slot later, the rest of it; once it has ended, its last value, which Home
    Assistant keeps - never nothing."""
    from emhass.coordinator.backend import _carried

    rec = _record([7.0, 7.0, 0.0])
    assert _carried(rec, _at("2026-11-21T10:00:00+00:00"), True).tolist() == [7, 7, 0, 0]
    assert _carried(rec, _at("2026-11-21T10:30:00+00:00"), True).tolist() == [7, 0, 0, 0]
    rec = _record([0.0, 7.0])
    assert _carried(rec, _at("2026-11-21T12:00:00+00:00"), True).tolist() == [7, 7, 7, 7]
    # in another time zone, the same instant
    assert _carried(rec, _at("2026-11-21T11:30:00+01:00"), True).tolist() == [7, 7, 7, 7]


def test_a_remote_that_does_not_step_runs_what_its_commit_published():
    """Without plan_steps, Home Assistant keeps the value the commit published
    (its first slot), with no end."""
    from emhass.coordinator.backend import _carried

    rec = _record([7.0, 0.0, 0.0])
    for start in ("2026-11-21T10:30:00+00:00", "2026-11-22T10:00:00+00:00"):
        assert _carried(rec, _at(start), False).tolist() == [7, 7, 7, 7]


@pytest.mark.parametrize(
    "record",
    [
        _record([1.0], start=None),
        _record([1.0], step=float("nan")),
        {**_record([1.0]), "plan_kw": [float("nan")]},
        _record([1.0], state="refused"),  # nothing of it runs
        {"state": "accepted"},
        "garbage",
    ],
)
def test_a_record_it_cannot_use_holds_nothing_and_never_raises(record):
    from emhass.coordinator.backend import _carried

    assert _carried(record, _at("2026-11-21T10:30:00+00:00"), True, 10.0) is None


def test_a_record_of_other_slots_is_read_by_time():
    """A plan of other slots is read by time: finer slots averaged within
    each of this window's (the same energy), coarser ones looked up; before
    the plan, nothing of it; past it, its last value. A value past what the
    remote can draw or supply is clipped to it, not dropped."""
    from emhass.coordinator.backend import _carried

    rec = _record([2.0, -9.0, 1.0], step=15.0)
    assert _carried(rec, _at("2026-11-21T10:00:00+00:00"), True).tolist() == [-3.5, 1, 1, 1]
    rec = _record([4.0, 1.0], step=60.0)
    assert _carried(rec, _at("2026-11-21T10:30:00+00:00"), True).tolist() == [4, 1, 1, 1]
    rec = _record([7.0, 0.0])
    assert _carried(rec, _at("2026-11-21T09:30:00+00:00"), True).tolist() == [0, 7, 0, 0]
    rec = _record([1e300, 0.0])
    assert _carried(rec, _at("2026-11-21T10:00:00+00:00"), True, 10.0).tolist() == [10, 0, 0, 0]


def test_the_remotes_own_record_wins_over_the_coordinators():
    """A remote that answered says itself what it runs: with no plan running,
    nothing is held, whatever this coordinator last committed it to; one that
    did not answer is held to that, only if it is at the same url."""
    from emhass.coordinator.backend import _held_plan

    window = _at("2026-11-21T10:00:00+00:00")
    saved = {"garage": {**_record([7.0]), "url": "http://g", "steps": True, "max_power_kw": 8.0}}
    answered = mock.MagicMock(active_commit=None, features=frozenset({"plan_steps"}))
    answered.max_power_kw = 8.0
    assert _held_plan(answered, "garage", "http://g", window, saved) is None
    assert _held_plan(None, "garage", "http://g", window, saved).tolist() == [7.0] * 4
    assert _held_plan(None, "garage", "http://elsewhere", window, saved) is None


def test_a_remote_held_to_its_plan_is_counted_when_the_coordinator_declines(data_path, monkeypatch):
    """The garage is held to its last plan (unreachable), and the coordinator
    then cannot plan: EMHASS's default solver plans the house around the
    garage's power, never as if it were off."""
    from emhass.coordinator import backend

    ec = _conf(data_path)
    server = serve(_garage_service(ec))
    site = [{"id": "garage", "parent": "grid", "solver": {"url": server.url}}]
    house = _house(_now())
    try:
        took = _coordinate(house, site)["fed_remote_power_garage"].values
    finally:
        server.shutdown()
        server.server_close()

    def fails(self, **kwargs):
        raise RuntimeError("the coordinator failed")

    monkeypatch.setattr(backend.DWCoordinator, "run", fails)
    res = _coordinate(house, site)
    assert (res["backend_used"] == "cvxpy").all()
    assert "held" in res["fed_remote_status_garage"].iloc[0]
    assert np.allclose(res["fed_remote_power_garage"].values, took)
    assert np.allclose(res["P_Load"].values, np.asarray(house[5]) + took)


def test_a_run_that_crosses_a_slot_boundary_is_still_live():
    """A run started just before a boundary (or a cron job on it, or two
    clocks a little apart) plans the slot next to the clock's: still live."""
    import time_machine as travel

    from emhass import utils
    from emhass.coordinator.backend import _starts_now

    rh = _confs()[0]
    opt = mock.MagicMock(retrieve_hass_conf=rh, time_step=0.5)
    step = pd.Timedelta(minutes=30)
    for method in ("nearest", "first", "last"):
        rh_m = {**rh, "method_ts_round": method}
        opt.retrieve_hass_conf = rh_m
        with travel.travel(pd.Timestamp("2026-11-21T10:14:20Z").to_pydatetime(), tick=False):
            planned = utils.get_forecast_start(
                pd.Timestamp.now(tz="UTC"), step, rh["time_zone"], method
            )
        with travel.travel(pd.Timestamp("2026-11-21T10:15:40Z").to_pydatetime(), tick=False):
            frame = pd.DataFrame(index=pd.date_range(planned, periods=4, freq="30min"))
            assert _starts_now(opt, frame), method
            stale = pd.DataFrame(index=pd.date_range(planned - 4 * step, periods=4, freq="30min"))
            assert not _starts_now(opt, stale), method


def test_a_null_rho_is_a_bad_request():
    service = _service(lambda p, a: ("accepted", ""))
    query = {
        "version": SCHEMA_VERSION,
        "kind": "price_response",
        "horizon": _HORIZON,
        "price_draw": [0.1, 0.2],
        "rho": None,
    }
    status, problem = service.handle("POST", "/v1/query", {}, json.dumps(query).encode())
    assert status == 400 and "rho" in problem["detail"]


class _StepperCase:
    """A committed plan of four 30-minute slots from 10:00, saved as the
    latest plan, and a stepper whose publishes are counted."""

    def __init__(self, data_path):
        from emhass.participant_api import PlanStepper

        self.ec = _conf(data_path)
        self.params = _params(self.ec, **GARAGE)
        self.res = pd.DataFrame(
            {"P_deferrable0": [7000.0, 7000.0, 0.0, 0.0]},
            index=pd.date_range("2026-11-21T10:00:00Z", periods=4, freq="30min"),
        )
        plan_store.record(data_path, plan_store.serialize(self.res), "2026-11-21T10:00:00Z", "1.0")
        self.published = []
        self.stepper = PlanStepper()
        self.gen = 1
        self.stepper._gen = 1
        self.stamp = plan_store.stamp(plan_store.read(data_path))

    def tick(self, at, begin=None):
        return self.stepper.tick(
            self.gen,
            self.stamp,
            self.ec,
            self.params,
            self.res,
            logger,
            begin,
            lambda *a: (self.published.append(at), ("accepted", ""))[1],
            pd.Timestamp(at),
        )


def test_a_committed_plan_is_published_again_at_each_slot(data_path):
    """After an accepted commit, its plan is published again at each slot
    (as publish-data does from the latest results), skipping a slot while a
    live optimization runs, until the plan has ended."""
    case = _StepperCase(data_path)
    assert case.tick("2026-11-21T10:30:01Z") == "published"
    assert case.published == ["2026-11-21T10:30:01Z"]
    assert case.tick("2026-11-21T11:00:01Z", begin=lambda: ("busy", None)) == "busy"  # again
    assert case.published == ["2026-11-21T10:30:01Z"]
    released = []
    out = case.tick("2026-11-21T11:30:01Z", begin=lambda: ("ok", lambda: released.append(1)))
    assert out == "published" and released == [1] and len(case.published) == 2
    assert case.tick("2026-11-21T12:00:01Z") == "stop"  # ended: Home Assistant keeps the last
    assert len(case.published) == 2 and not case.stepper.superseded


def test_stepping_stops_for_a_newer_commit_or_plan(data_path):
    """A newer commit replaces the plan stepped through; a newer plan saved
    (this EMHASS ran its own) stops it too."""
    case = _StepperCase(data_path)
    case.stepper._gen = 2  # a newer commit is followed
    assert case.tick("2026-11-21T10:30:01Z") == "stop" and case.published == []
    # this EMHASS saved a plan of its own - starting in the same slot, too
    case = _StepperCase(data_path)
    plan_store.record(data_path, plan_store.serialize(case.res), "2026-11-21T10:10:00Z", "1.0")
    assert case.tick("2026-11-21T10:30:01Z") == "stop" and case.published == []
    assert case.stepper.superseded


def test_emhass_says_it_steps_its_plans():
    from emhass.coordinator.interface import FEATURES

    service = _service(lambda p, a: ("accepted", ""))
    _, described = service.handle("GET", "/v1/describe", {}, None)
    assert "plan_steps" in FEATURES and "plan_steps" in described["features"]


# --------------------------------------------------------------------------
# One record of what each remote runs, read by every fallback (third review)
# --------------------------------------------------------------------------


def test_a_reachable_remote_running_a_plan_is_counted_when_the_first_plan_fails(
    data_path, monkeypatch
):
    """The garage answers, and runs the plan of the run before; this run's
    first plan fails outright. EMHASS's default solver plans the house around
    what the garage runs, as it reports it."""
    from emhass.coordinator import backend

    server = serve(_garage_service(_conf(data_path)))
    site = [{"id": "garage", "parent": "grid", "solver": {"url": server.url}}]
    house = _house(_now())
    try:
        took = _coordinate(house, site)["fed_remote_power_garage"].values

        def fails(self, **kwargs):
            raise RuntimeError("the coordinator failed")

        monkeypatch.setattr(backend.DWCoordinator, "run", fails)
        res = _coordinate(house, site)
    finally:
        server.shutdown()
    assert (res["backend_used"] == "cvxpy").all()
    assert np.allclose(res["fed_remote_power_garage"].values, took)
    assert np.allclose(res["P_Load"].values, np.asarray(house[5]) + took)


def test_an_early_decline_still_counts_what_the_remotes_run(data_path):
    """The coordinator declines before it reaches any remote (an export price
    above the import price): the default solver still plans around what this
    coordinator last committed them to."""
    server = serve(_garage_service(_conf(data_path)))
    site = [{"id": "garage", "parent": "grid", "solver": {"url": server.url}}]
    house = _house(_now())
    try:
        took = _coordinate(house, site)["fed_remote_power_garage"].values
    finally:
        server.shutdown()
        server.server_close()
    rh, oc, pc, data, pv, load, buy, sell = house
    above = (rh, oc, pc, data, pv, load, buy, np.asarray(buy) + 0.5)  # sells above buying
    res = _coordinate(above, site)
    assert (res["backend_used"] == "cvxpy").all()
    assert "export price" in res["backend_fallback_reason"].iloc[0]
    assert np.allclose(res["P_Load"].values, np.asarray(load) + took)
    assert "held" in res["fed_remote_status_garage"].iloc[0]


def test_a_remote_without_plan_steps_is_held_to_what_was_committed(data_path, monkeypatch):
    """A remote of an earlier draft (no plan_steps) answers, but cannot say
    what it runs: it is held to what this coordinator committed it to, the
    value its commit published, never to nothing."""
    from emhass.coordinator import interface, remote

    monkeypatch.setattr(remote, "FEATURES", ("commit_status",))
    monkeypatch.setattr(interface, "FEATURES", ("commit_status",))
    busy = {"on": False}
    service = _garage_service(
        _conf(data_path), begin=lambda: ("busy", None) if busy["on"] else ("ok", None)
    )
    server = serve(service)
    site = [{"id": "garage", "parent": "grid", "solver": {"url": server.url}}]
    house = _house(_now())
    try:
        took = _coordinate(house, site)["fed_remote_power_garage"].values
        busy["on"] = True
        res = _coordinate(house, site)
    finally:
        server.shutdown()
    assert "held" in res["fed_remote_status_garage"].iloc[0]
    assert np.allclose(res["fed_remote_power_garage"].values, took[0])  # its first slot, held


def test_a_partial_commit_is_followed_and_reported(data_path):
    """Published part way, a commit's plan is stepped through all the same
    (Home Assistant converges on it) and reported as the one it runs."""
    from emhass.coordinator.remote import CommitLog
    from emhass.participant_api import PlanStepper, _active_commit

    log = CommitLog()
    stepper = PlanStepper()
    log.begin("a", [1.0], _HORIZON)
    log.end("a", "accepted", "", {"plan_stamp": "s1"})
    stepper.following, stepper.stamp = "a", "s1"
    assert _active_commit(log, stepper)["commit_id"] == "a"
    log.begin("b", [2.0], _HORIZON)
    assert _active_commit(log, stepper)["state"] == "running"  # it may run
    log.end("b", "partial", "cut off", {"plan_stamp": "s1"})  # made over the same plan
    stepper.following = "b"  # it steps that one now
    assert _active_commit(log, stepper)["commit_id"] == "b"
    log.begin("c", [3.0], _HORIZON)
    log.end("c", "refused", "no")
    assert _active_commit(log, stepper)["commit_id"] == "b"  # a refusal changes nothing
    stepper.superseded = True  # a plan of its own since
    assert _active_commit(log, stepper) is None


def test_stepping_resumes_at_once_after_a_restart(data_path, monkeypatch):
    """After a restart the committed plan's current slot is published at once
    (not a slot late), from its saved copy - read back across a change of
    daylight saving time too."""
    import emhass.participant_api as pa
    from emhass.coordinator.remote import CommitLog

    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    index = pd.date_range("2026-10-24T22:00:00Z", periods=96, freq="30min").tz_convert(
        "Europe/Paris"
    )  # across the end of summer time
    res = pd.DataFrame({"P_deferrable0": 1000.0}, index=index)
    path = pa.committed_plan_path(ec, "a")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(res.to_csv(index_label="timestamp"))
    plan_store.record(data_path, plan_store.serialize(res), "2026-10-24T22:00:01Z", "1.0")
    log = CommitLog(data_path / pa.COMMIT_LOG_FILE)
    log.begin("a", [1.0], _HORIZON)
    log.end("a", "accepted", "", {"plan_stamp": plan_store.stamp(plan_store.read(data_path))})
    monkeypatch.setattr(pa, "_commit_logs", {data_path / pa.COMMIT_LOG_FILE: log})
    published = threading.Event()
    stepper = pa.PlanStepper()
    monkeypatch.setattr(pa, "_publish_now", lambda *a: (published.set(), ("accepted", ""))[1])
    import time_machine as travel

    with travel.travel(pd.Timestamp("2026-10-25T09:05:00Z").to_pydatetime(), tick=True):
        pa._resume(ec, params, logger, stepper, None)
        assert published.wait(5)
    stepper.stop()


def test_a_slot_that_is_busy_for_a_moment_is_published_when_it_is_free(data_path):
    """The stepper retries a busy slot within the slot, and a failure in one
    slot never ends stepping."""
    from emhass.participant_api import PlanStepper

    case = _StepperCase(data_path)
    stepper = PlanStepper()
    stepper.RETRY_S = 0.05
    tries = {"n": 0}
    done = threading.Event()

    def begin():
        tries["n"] += 1
        return ("busy", None) if tries["n"] < 3 else ("ok", None)

    def publish(*a):
        done.set()
        return "accepted", ""

    case.res.index = pd.date_range(
        pd.Timestamp.now(tz="UTC").floor("30min"), periods=4, freq="30min"
    )
    plan_store.record(data_path, plan_store.serialize(case.res), "2026-11-21T10:00:00Z", "1.0")
    stamp = plan_store.stamp(plan_store.read(data_path))
    stepper.follow(
        "a",
        stamp,
        case.ec,
        case.params,
        case.res,
        logger,
        begin=begin,
        publish=publish,
        now_first=True,
    )
    assert done.wait(5) and tries["n"] == 3
    stepper.stop()
    failing = PlanStepper()
    calls = []

    def boom(*a):
        calls.append(1)
        raise RuntimeError("Home Assistant is down")

    # its next slot begins in a moment: a publish that raised loses that
    # slot, never the plan, and the next slot is published
    soon = pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=0.3)
    case.res.index = pd.date_range(soon - pd.Timedelta(minutes=30), periods=4, freq="30min")
    try:
        failing.follow(
            "a", stamp, case.ec, case.params, case.res, logger, publish=boom, now_first=True
        )
        for _ in range(250):
            if len(calls) >= 2:
                break
            threading.Event().wait(0.02)
        assert calls == [1, 1]  # the slot that raised, then the next
        assert failing._gen == 1  # still stepping
    finally:
        failing.stop()
        assert failing.join(10.0)


def test_a_commit_waits_briefly_for_a_slot_the_stepper_holds(data_path):
    """The stepper holds the live-optimization slot a moment at each slot
    boundary: a commit then waits for it, rather than being refused."""
    saved = web_server.emhass_conf
    web_server.emhass_conf = {**saved, "data_path": data_path}
    try:
        holder = open(web_server._live_optim_slot(), "a+")
        assert utils.try_lock_file(holder)
        threading.Timer(0.5, lambda: (utils.unlock_file(holder), holder.close())).start()
        taken, release = web_server._take_live_optim_slot()
        assert taken == "ok"
        release()
    finally:
        web_server.emhass_conf = saved


def test_a_stale_fallback_is_never_planned_around(data_path):
    """The coordinator's record of what its remotes run is reset on every
    run, even one it declines at once: an Optimization used again never plans
    around remotes of an earlier run."""
    from emhass import optimization

    opt = mock.MagicMock()
    opt.optim_conf = {"optimization_backend": "dantzig_wolfe", "number_of_inverters": 2}
    opt.plant_conf = {"number_of_inverters": 2, "inverter_ac_output_max": [1, 1]}
    opt.fed_running = {"garage": np.ones(2)}
    opt.fed_site_remotes = ["garage"]
    res, reason = optimization._coordinated_plan(opt)
    assert res is None and reason
    assert opt.fed_running == {} and optimization._remotes_running_w(opt) is None


# --------------------------------------------------------------------------
# Fourth blind review
# --------------------------------------------------------------------------


def test_a_config_the_coordinator_cannot_plan_still_counts_what_remotes_run(tmp_path):
    """Declined before the coordinator runs at all (here, two inverters): the
    default solver still plans around what the remotes it last committed run."""
    from emhass import optimization
    from emhass.coordinator import backend

    index = pd.date_range("2026-11-21T10:00:00Z", periods=4, freq="30min")
    record = {
        "state": "committed",
        "url": "http://garage:5000/participant",
        "steps": True,
        "max_power_kw": 8.0,
        "horizon": {"step_minutes": 30.0, "slots": 4, "start": index[0].isoformat()},
        "plan_kw": [7.0, 7.0, 0.0, 0.0],
    }
    (tmp_path / "coordinator_remote_commits.json").write_text(json.dumps({"garage": record}))
    opt = mock.MagicMock(time_step=0.5, emhass_conf={"data_path": tmp_path})
    opt.optim_conf = {
        "optimization_backend": "dantzig_wolfe",
        "site": [{"id": "garage", "parent": "grid", "solver": {"url": record["url"]}}],
    }
    opt.plant_conf = {"number_of_inverters": 2, "inverter_ac_output_max": [1, 1]}
    res, reason = optimization._coordinated_plan(opt, data_opt=pd.DataFrame(index=index))
    assert res is None and reason
    assert optimization._remotes_running_w(opt).tolist() == [7000.0, 7000.0, 0.0, 0.0]
    assert backend.fallback_columns(opt)["fed_remote_status_garage"].endswith("held)")


def test_a_plan_of_its_own_ends_the_commit_it_reports_at_once(data_path):
    """Once this EMHASS saves a plan of its own, it reports no commit as the
    one it runs - even after the committed plan has ended and nothing steps."""
    from emhass.coordinator.remote import CommitLog
    from emhass.participant_api import PlanStepper, _active_commit

    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    stamp = plan_store.stamp(plan_store.read(data_path))
    log, stepper = CommitLog(), PlanStepper()
    log.begin("a", [1.0], _HORIZON)
    log.end("a", "accepted", "", {"plan_stamp": stamp})
    stepper.following, stepper.stamp = "a", stamp
    assert _active_commit(log, stepper, data_path)["commit_id"] == "a"
    plan_store.record(data_path, [], "2026-11-21T10:20:00Z", "1.0")  # its own run
    assert _active_commit(log, stepper, data_path) is None and stepper.superseded


def test_a_step_checks_again_once_it_has_the_slot(data_path):
    """A live optimization it waited for saved a plan of this EMHASS's own:
    the step does not publish the committed plan over it."""
    case = _StepperCase(data_path)

    def begin():  # the slot comes free once the live run has saved its plan
        plan_store.record(data_path, [], "2026-11-21T10:31:00Z", "1.0")
        return "ok", None

    assert case.tick("2026-11-21T10:30:01Z", begin=begin) == "stop"
    assert case.published == [] and case.stepper.superseded


def test_a_step_not_published_is_tried_again_in_the_slot(data_path):
    case = _StepperCase(data_path)
    out = case.stepper.tick(
        case.gen,
        case.stamp,
        case.ec,
        case.params,
        case.res,
        logger,
        None,
        lambda *a: ("refused", "Home Assistant is not answering"),
        pd.Timestamp("2026-11-21T10:30:01Z"),
    )
    assert out == "busy"


def test_a_restart_resumes_only_the_plan_of_the_commit_it_names(data_path, monkeypatch):
    """The plan kept for resuming names its commit: one of another commit (a
    restart part way through keeping it) is never stepped."""
    import emhass.participant_api as pa
    from emhass.coordinator.remote import CommitLog

    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    res = pd.DataFrame({"P_deferrable0": 1.0}, index=pd.date_range(_now(), periods=4, freq="30min"))
    path = pa.committed_plan_path(ec, "older")  # kept for another commit than the last
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(res.to_csv(index_label="timestamp"))
    log = CommitLog()
    log.begin("newer", [1.0], _HORIZON)
    log.end("newer", "accepted", "", {})
    monkeypatch.setattr(pa, "_commit_logs", {data_path / pa.COMMIT_LOG_FILE: log})
    stepper = pa.PlanStepper()
    followed = []
    monkeypatch.setattr(stepper, "follow", lambda *a, **k: followed.append(1))
    pa._resume(ec, params, logger, stepper, None)
    assert followed == []


def test_a_commit_that_raises_while_publishing_is_followed_as_partial(data_path, monkeypatch):
    """Publishing raised part way: Home Assistant may hold part of the plan,
    so it is recorded partial and stepped through all the same."""
    import emhass.participant_api as pa
    from emhass.coordinator.interface import Answer

    ec = _conf(data_path)
    params = _params(ec, **GARAGE)

    async def raises(*a, **k):
        raise RuntimeError("Home Assistant went away")

    monkeypatch.setattr(pa, "publish_commit", raises)
    followed = []
    stepper = pa.plan_stepper(ec)
    monkeypatch.setattr(stepper, "follow", lambda *a, **k: followed.append(a[0]))
    monkeypatch.setattr(pa, "_resume", lambda *a, **k: None)
    service = pa.make_service(ec, params, TOKEN, logger)
    res = pd.DataFrame({"P_deferrable0": 1.0}, index=pd.date_range(_now(), periods=2, freq="30min"))
    out = service._commit(None, Answer(np.zeros(2), 0.0, np.zeros(3), detail=res), "c1")
    assert out[0] == "partial" and followed == ["c1"]
    assert pa.committed_plan_path(ec, "c1").exists()


def test_the_participant_service_starts_with_the_web_server(data_path, monkeypatch):
    """At start-up the participant API's service is built (so a committed plan
    is stepped again at once after a restart), not only on the first request."""
    params = _params(_conf(data_path), **GARAGE)
    built = []

    async def read(conf):
        return params

    monkeypatch.setattr(participant_api, "read_params", read)
    monkeypatch.setattr(participant_api, "make_service", lambda *a, **k: built.append(1) or "svc")
    resumed = []
    monkeypatch.setattr(participant_api, "resume_steps", lambda *a, **k: resumed.append(1))
    monkeypatch.setattr(web_server, "_participant_service", None)
    monkeypatch.setitem(web_server.params_secrets, "participant_token", TOKEN)
    assert asyncio.run(web_server._participant_service_for_params()) == "svc"
    assert asyncio.run(web_server._participant_service_for_params()) == "svc"
    assert built == [1]  # once per params and token
    assert resumed == [1]  # and then: stepping resumed if it had stopped


def test_what_a_stepping_remote_reports_corrects_the_saved_record(tmp_path):
    """A remote that steps its plans and reports none running is no longer
    held to the plan this coordinator last committed it to, the next time it
    cannot be reached."""
    from emhass.coordinator import backend

    opt = mock.MagicMock(fed_dry_run=False)
    saved = {"garage": {**_record([7.0]), "url": "http://g", "steps": True, "max_power_kw": 8.0}}
    running = {"garage": np.full(4, 7.0)}
    rp = mock.MagicMock(key="garage", url="http://g", active_commit=None, max_power_kw=8.0)
    rp.features = frozenset({"plan_steps"})
    backend._heard(opt, running, saved, rp, _at("2026-11-21T10:00:00+00:00"))
    assert running == {} and saved == {}
    stored = json.loads((tmp_path / "coordinator_remote_commits.json").read_text())
    assert stored == {}


# --------------------------------------------------------------------------
# Fifth blind review: which commit the stepper follows, explicitly
# --------------------------------------------------------------------------


def _resume_case(data_path, monkeypatch, records):
    """A participant restarting with these commit records (id, state, fields),
    each with its plan kept; returns (stepper, followed, log)."""
    import emhass.participant_api as pa
    from emhass.coordinator.remote import CommitLog

    ec = _conf(data_path)
    log = CommitLog(data_path / pa.COMMIT_LOG_FILE)
    res = pd.DataFrame({"P_deferrable0": 1.0}, index=pd.date_range(_now(), periods=4, freq="30min"))
    for cid, state, fields in records:
        log.begin(cid, [1.0], _HORIZON)
        if state != "running":
            log.end(cid, state, "", fields)
        path = pa.committed_plan_path(ec, cid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(res.to_csv(index_label="timestamp"))
    monkeypatch.setattr(pa, "_commit_logs", {data_path / pa.COMMIT_LOG_FILE: log})
    stepper = pa.PlanStepper()
    followed = []
    monkeypatch.setattr(stepper, "follow", lambda *a, **k: followed.append(a[:2]))
    return ec, stepper, followed, log


def test_a_commit_cut_off_by_a_restart_keeps_the_stamp_it_was_resumed_with(data_path, monkeypatch):
    """Resumed once (no stamp: it was cut off while publishing), it keeps the
    stamp it resumed with: after this EMHASS saves a plan of its own, a later
    resume never steps it over that plan."""
    import emhass.participant_api as pa

    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    s2 = plan_store.stamp(plan_store.read(data_path))
    ec, stepper, followed, log = _resume_case(data_path, monkeypatch, [("b", "running", {})])
    log._records["b"]["state"] = "partial"  # as a restart reads a running one back
    params = _params(ec, **GARAGE)
    pa._resume(ec, params, logger, stepper, None)
    assert followed == [("b", s2)] and log.get("b")["plan_stamp"] == s2
    plan_store.record(data_path, [], "2026-11-21T10:20:00Z", "1.0")  # its own run
    stepper.resumed = None  # a second restart (or new params)
    pa._resume(ec, params, logger, stepper, None)
    assert followed == [("b", s2)] and stepper.superseded
    assert pa._active_commit(log, stepper, data_path) is None


def test_a_refused_commit_leaves_the_running_plan_to_resume(data_path, monkeypatch):
    """A refused commit keeps its plan under its own name: after a restart,
    the plan that runs (the accepted one before it) is the one resumed."""
    import emhass.participant_api as pa

    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    stamp = plan_store.stamp(plan_store.read(data_path))
    ec, stepper, followed, log = _resume_case(
        data_path,
        monkeypatch,
        [("a", "accepted", {"plan_stamp": stamp}), ("b", "refused", {})],
    )
    pa._resume(ec, _params(ec, **GARAGE), logger, stepper, None)
    assert followed == [("a", stamp)]


def test_a_first_commit_on_a_new_participant_is_reported_while_stepped(data_path, monkeypatch):
    """No plan saved before (a stamp of None) is a stamp all the same: the
    partial first commit it steps is reported as the one it runs."""
    from emhass.coordinator.remote import CommitLog
    from emhass.participant_api import PlanStepper, _active_commit

    monkeypatch.setattr(plan_store, "_cache", None)  # a new participant: nothing saved

    log, stepper = CommitLog(), PlanStepper()
    log.begin("first", [1.0], _HORIZON)
    log.end("first", "partial", "cut off", {"plan_stamp": None})
    stepper.following, stepper.stamp = "first", None
    assert plan_store.read(data_path) is None
    assert _active_commit(log, stepper, data_path)["commit_id"] == "first"


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p["optim_conf"].pop("participant_api"),
        lambda p: p["retrieve_hass_conf"].__setitem__("continual_publish", True),
    ],
    ids=["switched-off", "continual-publish"],
)
def test_stepping_follows_its_params(data_path, monkeypatch, change):
    """The participant API switched off changes nothing: a plan committed
    runs on whether or not it answers. continual_publish turned on (its loop
    publishes this EMHASS's own plan) ends it, for good."""
    import emhass.participant_api as pa

    case = _StepperCase(data_path)
    now = copy.deepcopy(case.params)
    change(now)
    monkeypatch.setattr(pa, "_current_params", lambda conf, fallback: now)
    out = case.tick("2026-11-21T10:30:01Z")
    if "participant_api" in now["optim_conf"]:  # continual_publish
        assert out == "stop" and case.stepper.superseded and case.published == []
    else:  # switched off: stepped all the same, under the prefix it was committed with
        assert out == "published" and not case.stepper.superseded


def test_a_new_prefix_is_used_at_the_next_step(data_path, monkeypatch):
    import emhass.participant_api as pa

    case = _StepperCase(data_path)
    now = copy.deepcopy(case.params)
    now["optim_conf"]["participant_api"]["publish_prefix"] = "garage2_"
    monkeypatch.setattr(pa, "_current_params", lambda conf, fallback: now)
    used = []
    out = case.stepper.tick(
        case.gen,
        case.stamp,
        case.ec,
        case.params,
        case.res,
        logger,
        None,
        lambda conf, params, *a: (used.append(params), ("accepted", ""))[1],
        pd.Timestamp("2026-11-21T10:30:01Z"),
    )
    assert out == "published"
    assert used[0]["optim_conf"]["participant_api"]["publish_prefix"] == "garage2_"


def test_a_commit_id_never_names_a_path(data_path):
    from emhass.participant_api import COMMITTED_PLANS, committed_plan_path

    ec = _conf(data_path)
    for cid in ("../../etc/passwd", "a/b", "q1"):
        path = committed_plan_path(ec, cid)
        assert path.parent == data_path / COMMITTED_PLANS and "/" not in path.name


def test_a_dry_run_writes_no_record_of_what_remotes_run(tmp_path):
    from emhass.coordinator import backend

    opt = mock.MagicMock(fed_dry_run=True)
    saved = {"garage": {**_record([7.0]), "url": "http://g", "steps": True, "max_power_kw": 8.0}}
    rp = mock.MagicMock(key="garage", url="http://g", active_commit=None, max_power_kw=8.0)
    rp.features = frozenset({"plan_steps"})
    backend._heard(opt, {}, saved, rp, _at("2026-11-21T10:00:00+00:00"))
    assert not (tmp_path / "coordinator_remote_commits.json").exists()


# --------------------------------------------------------------------------
# Sixth blind review
# --------------------------------------------------------------------------


def test_the_commit_that_may_run_is_never_pruned(tmp_path):
    """The latest commit that may run (accepted, partial or publishing)
    outlives the log's keeping time and size - and a restart: its plan may
    run however long the coordinator is away."""
    from emhass.coordinator.remote import CommitLog

    path = tmp_path / "participant_commits.json"
    log = CommitLog(path, keep_s=3600.0, max_entries=2)
    log.begin("old", [1.0], _HORIZON)
    log.end("old", "accepted", "")
    log._records["old"]["received_ts"] -= 7200  # past its keeping time
    for cid in ("b", "c", "d"):
        log.begin(cid, [1.0], _HORIZON)
        log.end(cid, "refused", "no")
    assert log.get("old") is not None and len(log.records()) == 2
    again = CommitLog(path, keep_s=3600.0, max_entries=2)  # a restart
    assert again.get("old") is not None and again.latest(running=True)["commit_id"] == "old"


def test_a_commit_carries_its_stamp_before_it_publishes(data_path, monkeypatch):
    """Cut off while publishing, its record already names the plan it was
    made over."""
    import emhass.participant_api as pa
    from emhass.coordinator.interface import Answer

    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    stamp = plan_store.stamp(plan_store.read(data_path))
    seen = []

    async def publishing(*a, **k):
        seen.append(log.get("c1").get("plan_stamp"))
        return "refused", "no"

    monkeypatch.setattr(pa, "publish_commit", publishing)
    monkeypatch.setattr(pa, "_resume", lambda *a, **k: None)
    service = pa.make_service(ec, params, TOKEN, logger)
    log = service.log
    log.begin("c1", [1.0], _HORIZON)
    res = pd.DataFrame({"P_deferrable0": 1.0}, index=pd.date_range(_now(), periods=2, freq="30min"))
    service._commit(None, Answer(np.zeros(2), 0.0, np.zeros(3), detail=res), "c1")
    assert seen == [stamp]


def test_a_partial_commit_is_stepped_at_once(data_path, monkeypatch):
    import emhass.participant_api as pa
    from emhass.coordinator.interface import Answer

    ec = _conf(data_path)

    async def partial(*a, **k):
        return "partial", "cut off"

    monkeypatch.setattr(pa, "publish_commit", partial)
    monkeypatch.setattr(pa, "_resume", lambda *a, **k: None)
    stepper = pa.plan_stepper(ec)
    calls = []
    monkeypatch.setattr(stepper, "follow", lambda *a, **k: calls.append(k.get("now_first")))
    service = pa.make_service(ec, _params(ec, **GARAGE), TOKEN, logger)
    res = pd.DataFrame({"P_deferrable0": 1.0}, index=pd.date_range(_now(), periods=2, freq="30min"))
    service._commit(None, Answer(np.zeros(2), 0.0, np.zeros(3), detail=res), "c1")
    assert calls == [True]


def test_a_commit_this_emhass_cannot_take_is_refused_not_unavailable(data_path):
    """Without its token, a commit is refused (409: nothing ran), never 503,
    which a coordinator reads as "it may have run"; other calls stay 503."""
    params = _params(_conf(data_path), **GARAGE)

    async def post(op):
        saved = web_server.emhass_conf
        web_server.emhass_conf = {**saved, "data_path": data_path}
        web_server.params_secrets.pop("participant_token", None)
        (data_path / "params.pkl").write_bytes(pickle.dumps((None, params)))
        try:
            client = web_server.app.test_client()
            if op == "commit":
                r = await client.post("/participant/v1/commit", data=b"{}")
            else:
                r = await client.get("/participant/v1/describe")
            return r.status_code
        finally:
            web_server.emhass_conf = saved

    assert asyncio.run(post("commit")) == 409
    assert asyncio.run(post("describe")) == 503


def test_a_dry_run_of_another_day_asks_no_remote(data_path, monkeypatch):
    """A dry run counts as live only when its plan starts now: one of another
    day plans the remotes as held, and asks them nothing."""
    from emhass.coordinator import backend

    monkeypatch.setattr(backend, "_starts_now", lambda opt, data_opt: False)
    asked = []
    service = _garage_service(_conf(data_path))
    describe = service._describe
    service._describe = lambda: (asked.append(1), describe())[1]
    server = serve(service)
    try:
        site = [{"id": "garage", "parent": "grid", "solver": {"url": server.url}}]
        res = _coordinate(_house(_now()), site, dry_run=True)
        assert res["fed_remote_status_garage"].iloc[0].startswith("planned without (not live")
        assert asked == []
    finally:
        server.shutdown()


def test_a_resume_at_start_up_that_fails_never_stops_serving(monkeypatch):
    async def fails():
        raise RuntimeError("a thread could not be started")

    monkeypatch.setattr(web_server, "_participant_service_for_params", fails)
    monkeypatch.setattr(web_server, "initialize", mock.AsyncMock())
    monkeypatch.setattr(web_server, "_configure_api_auth", mock.AsyncMock())
    asyncio.run(web_server.before_serving())  # does not raise


# --------------------------------------------------------------------------
# Seventh blind review
# --------------------------------------------------------------------------


def test_continual_publish_ends_a_plan_for_good(data_path, monkeypatch):
    """Turned on, continual_publish ends the plan stepped; turned off again,
    or after a restart, that old plan is never run again."""
    import emhass.participant_api as pa

    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    stamp = plan_store.stamp(plan_store.read(data_path))
    ec, stepper, followed, log = _resume_case(
        data_path, monkeypatch, [("a", "accepted", {"plan_stamp": stamp})]
    )
    params = _params(ec, **GARAGE)
    on = copy.deepcopy(params)
    on["retrieve_hass_conf"]["continual_publish"] = True
    pa._resume(ec, on, logger, stepper, None)
    assert followed == [] and log.get("a")["superseded"] is True
    assert pa._active_commit(log, stepper, data_path) is None
    stepper2 = pa.PlanStepper()  # a restart, with it off again
    monkeypatch.setattr(stepper2, "follow", lambda *a, **k: followed.append(a[:2]))
    pa._resume(ec, params, logger, stepper2, None)
    assert followed == []


def test_a_plan_that_ended_while_down_is_published_once_at_its_last_value(data_path):
    """Down over the plan's end: back up, its last value is published once
    (Home Assistant was left on a value from its middle), then it stops."""
    case = _StepperCase(data_path)
    out = case.stepper.tick(
        case.gen,
        case.stamp,
        case.ec,
        case.params,
        case.res,
        logger,
        None,
        lambda *a: (case.published.append("last"), ("accepted", ""))[1],
        pd.Timestamp("2026-11-21T14:00:01Z"),
        True,
    )
    assert out == "stop" and case.published == ["last"]
    assert case.tick("2026-11-21T14:30:01Z") == "stop" and case.published == ["last"]


def test_a_remote_that_does_not_step_is_held_from_its_commit_on():
    """Without plan_steps, held to the value its commit published - from the
    commit's slot on, never before it."""
    from emhass.coordinator.backend import _carried

    rec = _record([5.0, 0.0])
    assert _carried(rec, _at("2026-11-21T09:30:00+00:00"), False).tolist() == [0, 5, 5, 5]
    assert _carried(rec, _at("2026-10-01T10:00:00+00:00"), False) is None
    assert _carried(rec, _at("2026-11-22T10:00:00+00:00"), False).tolist() == [5, 5, 5, 5]


def test_an_accepted_commit_names_its_own_plan_as_soon_as_it_is_saved(data_path, monkeypatch):
    """Its plan saved, its record names that plan at once: a crash before the
    commit is recorded as accepted resumes it rather than taking it for a
    plan of this EMHASS's own."""
    import emhass.participant_api as pa
    from emhass.coordinator.interface import Answer

    ec = _conf(data_path)
    seen = {}

    async def publishing(conf, params, res, logger, on_recorded=None, **k):
        plan_store.record(data_path, [], "2026-11-21T10:05:00Z", "1.0")
        on_recorded(plan_store.stamp(plan_store.read(data_path)))
        seen["stamp"] = log.get("c1").get("plan_stamp")
        return "accepted", ""

    monkeypatch.setattr(pa, "publish_commit", publishing)
    monkeypatch.setattr(pa, "_resume", lambda *a, **k: None)
    service = pa.make_service(ec, _params(ec, **GARAGE), TOKEN, logger)
    log = service.log
    log.begin("c1", [1.0], _HORIZON)
    res = pd.DataFrame({"P_deferrable0": 1.0}, index=pd.date_range(_now(), periods=2, freq="30min"))
    out = service._commit(None, Answer(np.zeros(2), 0.0, np.zeros(3), detail=res), "c1")
    assert seen["stamp"] == plan_store.stamp(plan_store.read(data_path))
    # the stamp followed is the one recorded as it was saved, not one read later
    plan_store.record(data_path, [], "2026-11-21T10:06:00Z", "1.0")
    assert out[2]["plan_stamp"] == seen["stamp"]


# --------------------------------------------------------------------------
# Eighth blind review
# --------------------------------------------------------------------------


def test_a_plan_is_published_under_its_commits_prefix_with_the_api_off(data_path, monkeypatch):
    """Switched off since the commit, the participant API's prefix is gone
    from the params: the plan is still published under the commit's, never
    over this EMHASS's own (unprefixed) entities."""
    import emhass.participant_api as pa

    case = _StepperCase(data_path)
    now = copy.deepcopy(case.params)
    now["optim_conf"].pop("participant_api")
    monkeypatch.setattr(pa, "_current_params", lambda conf, fallback: now)
    used = []
    out = case.stepper.tick(
        case.gen,
        case.stamp,
        case.ec,
        case.params,
        case.res,
        logger,
        None,
        lambda conf, params, *a: (used.append(params), ("accepted", ""))[1],
        pd.Timestamp("2026-11-21T10:30:01Z"),
    )
    assert out == "published"
    assert used[0]["optim_conf"]["participant_api"]["publish_prefix"] == "garage_"


def test_a_resume_with_the_api_off_publishes_under_the_commits_prefix(data_path, monkeypatch):
    import emhass.participant_api as pa

    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    stamp = plan_store.stamp(plan_store.read(data_path))
    conf = {"publish_prefix": "garage_", "key": "garage"}
    ec, stepper, followed, log = _resume_case(
        data_path,
        monkeypatch,
        [("a", "accepted", {"plan_stamp": stamp, "participant_api": conf})],
    )
    params = _params(ec, **GARAGE)
    params["optim_conf"].pop("participant_api")  # switched off
    resumed = []
    monkeypatch.setattr(stepper, "follow", lambda *a, **k: resumed.append(a[3]))
    pa._resume(ec, params, logger, stepper, None)
    assert resumed and resumed[0]["optim_conf"]["participant_api"]["publish_prefix"] == "garage_"


def test_web_server_start_up_resumes_stepping_even_with_the_api_off(data_path, monkeypatch):
    params = _params(_conf(data_path))  # no participant_api
    resumed = []

    async def read(conf):
        return params

    monkeypatch.setattr(participant_api, "read_params", read)
    monkeypatch.setattr(participant_api, "resume_steps", lambda *a, **k: resumed.append(a[1]))
    monkeypatch.setattr(web_server, "initialize", mock.AsyncMock())
    monkeypatch.setattr(web_server, "_configure_api_auth", mock.AsyncMock())
    asyncio.run(web_server.before_serving())
    assert resumed == [params]


def test_a_commit_cut_off_before_its_plan_was_kept_leaves_the_one_that_ran(data_path, monkeypatch):
    """Restarted between taking a commit and keeping its plan: nothing of it
    was published (that comes after), so it is refused, and the commit
    before it - which runs - is the one resumed."""
    import emhass.participant_api as pa

    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    stamp = plan_store.stamp(plan_store.read(data_path))
    ec, stepper, followed, log = _resume_case(
        data_path, monkeypatch, [("a", "accepted", {"plan_stamp": stamp})]
    )
    log.begin("x", [1.0], _HORIZON)
    log._records["x"]["state"] = "partial"  # as a restart reads a running one back
    assert not pa.committed_plan_path(ec, "x").exists()
    pa._resume(ec, _params(ec, **GARAGE), logger, stepper, None)
    assert followed == [("a", stamp)] and log.get("x")["state"] == "refused"


def test_a_resume_yields_to_a_commit_followed_meanwhile(data_path, monkeypatch):
    import emhass.participant_api as pa

    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    stamp = plan_store.stamp(plan_store.read(data_path))
    ec, stepper, followed, log = _resume_case(
        data_path, monkeypatch, [("a", "accepted", {"plan_stamp": stamp})]
    )
    real_read = pd.read_csv

    def read_while_a_commit_lands(*a, **k):
        stepper._gen += 1  # a newer commit is followed while the plan is read
        return real_read(*a, **k)

    monkeypatch.setattr(pa.pd, "read_csv", read_while_a_commit_lands)
    asked = []
    monkeypatch.setattr(stepper, "follow", lambda *a, **k: asked.append(k.get("expect_gen")))
    pa._resume(ec, _params(ec, **GARAGE), logger, stepper, None)
    # it asks to follow only as of the generation it began with, which the
    # commit has moved past: follow() then starts nothing (tested on its own)
    assert asked == [0] and stepper.generation() == 1


def test_the_participant_service_is_built_once_however_many_ask(data_path, monkeypatch):
    params = _params(_conf(data_path), **GARAGE)
    built = []

    async def read(conf):
        return params

    def build(*a, **k):
        built.append(1)
        threading.Event().wait(0.2)  # slow enough for every request to ask
        return "svc"

    monkeypatch.setattr(participant_api, "read_params", read)
    monkeypatch.setattr(participant_api, "make_service", build)
    monkeypatch.setattr(participant_api, "resume_steps", lambda *a, **k: None)
    monkeypatch.setattr(web_server, "_participant_service", None)
    monkeypatch.setitem(web_server.params_secrets, "participant_token", TOKEN)

    async def many():
        return await asyncio.gather(
            *(web_server._participant_service_for_params() for _ in range(5))
        )

    assert asyncio.run(many()) == ["svc"] * 5 and built == [1]


# --------------------------------------------------------------------------
# Ninth blind review
# --------------------------------------------------------------------------


def test_an_end_seen_while_the_api_is_off_is_recorded(data_path, monkeypatch):
    """Resumed with the participant API off (no service built), the stepper
    still records an end it sees: a later restart never revives the plan."""
    import emhass.participant_api as pa

    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    stamp = plan_store.stamp(plan_store.read(data_path))
    ec, stepper, followed, log = _resume_case(
        data_path, monkeypatch, [("a", "accepted", {"plan_stamp": stamp})]
    )
    params = _params(ec, **GARAGE)
    pa._resume(ec, params, logger, stepper, None)  # no make_service: the API is off
    assert stepper.log is log
    stepper.following, stepper._gen = "a", 1
    stepper._end(1)  # e.g. continual_publish seen by a step
    assert log.get("a")["superseded"] is True
    again = pa.PlanStepper()
    revived = []
    monkeypatch.setattr(again, "follow", lambda *a, **k: revived.append(1))
    pa._resume(ec, params, logger, again, None)
    assert revived == []


def test_a_commit_whose_plan_cannot_be_kept_is_refused_unpublished(data_path, monkeypatch):
    import emhass.participant_api as pa
    from emhass.coordinator.interface import Answer

    ec = _conf(data_path)
    published = []

    async def publishing(*a, **k):
        published.append(1)
        return "accepted", ""

    async def disk_full(*a, **k):
        raise OSError("No space left on device")

    monkeypatch.setattr(pa, "publish_commit", publishing)
    monkeypatch.setattr(pa.utils, "replace_file", disk_full)
    monkeypatch.setattr(pa, "_resume", lambda *a, **k: None)
    service = pa.make_service(ec, _params(ec, **GARAGE), TOKEN, logger)
    service.log.begin("c1", [1.0], _HORIZON)
    res = pd.DataFrame({"P_deferrable0": 1.0}, index=pd.date_range(_now(), periods=2, freq="30min"))
    out = service._commit(None, Answer(np.zeros(2), 0.0, np.zeros(3), detail=res), "c1")
    assert out[0] == "refused" and "nothing was published" in out[1] and published == []
    assert "OSError" in out[1] and "No space" not in out[1]  # no local error text sent


def test_a_resume_never_overtakes_a_commit_followed_meanwhile(data_path):
    from emhass.participant_api import PlanStepper

    case = _StepperCase(data_path)
    stepper = PlanStepper()
    gen = stepper.generation()
    stepper.follow("b", case.stamp, case.ec, case.params, case.res, logger)  # a commit lands
    stepper.follow(
        "a", case.stamp, case.ec, case.params, case.res, logger, expect_gen=gen
    )  # the resume, too late
    assert stepper.following == "b"
    stepper.stop()


def test_a_plan_store_it_cannot_read_now_is_tried_again_not_taken_as_replaced(
    data_path, monkeypatch
):
    case = _StepperCase(data_path)
    monkeypatch.setattr(plan_store, "read", lambda path: None)  # the file is there, unreadable
    assert case.tick("2026-11-21T10:30:01Z") == "busy"
    assert not case.stepper.superseded and case.published == []


def test_no_plan_is_stepped_without_a_prefix(data_path, monkeypatch):
    """The API switched off and the commit's prefix not known (a record from
    before it was kept): nothing is published over this EMHASS's own
    entities."""
    import emhass.participant_api as pa

    case = _StepperCase(data_path)
    now = copy.deepcopy(case.params)
    now["optim_conf"].pop("participant_api")
    committed = copy.deepcopy(now)  # no prefix known either
    monkeypatch.setattr(pa, "_current_params", lambda conf, fallback: now)
    out = case.stepper.tick(
        case.gen,
        case.stamp,
        case.ec,
        committed,
        case.res,
        logger,
        None,
        lambda *a: (case.published.append(1), ("accepted", ""))[1],
        pd.Timestamp("2026-11-21T10:30:01Z"),
    )
    assert out == "busy" and case.published == []  # tried again; published once known


def test_without_a_site_no_remote_is_held(tmp_path):
    from emhass.coordinator import backend

    index = pd.date_range("2026-11-21T10:00:00Z", periods=4, freq="30min")
    saved = {"old": {**_record([7.0, 7.0, 7.0, 7.0]), "url": "http://g", "steps": True}}
    (tmp_path / "coordinator_remote_commits.json").write_text(json.dumps(saved))
    opt = mock.MagicMock(
        time_step=0.5,
        emhass_conf={"data_path": tmp_path},
        optim_conf={"optimization_backend": "dantzig_wolfe"},
    )
    running, _, _ = backend.note_running(opt, pd.DataFrame(index=index))
    assert running == {}
    # control: a site it cannot read holds every remote it committed
    opt.optim_conf = {"optimization_backend": "dantzig_wolfe", "site": 5}
    running, _, _ = backend.note_running(opt, pd.DataFrame(index=index))
    assert list(running) == ["old"]


def test_a_plan_store_it_cannot_read_never_ends_a_plan_elsewhere(data_path, monkeypatch):
    """Unreadable for a moment, the plan store is never taken as "a plan of
    its own replaced it": not by a resume (which tries again later), not by
    what it reports, and a commit made then is refused, unpublished."""
    import emhass.participant_api as pa
    from emhass.coordinator.interface import Answer

    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    stamp = plan_store.stamp(plan_store.read(data_path))
    ec, stepper, followed, log = _resume_case(
        data_path, monkeypatch, [("a", "accepted", {"plan_stamp": stamp})]
    )
    monkeypatch.setattr(plan_store, "read", lambda path: None)  # the file is there, unreadable
    params = _params(ec, **GARAGE)
    pa._resume(ec, params, logger, stepper, None)
    assert followed == [] and not log.get("a").get("superseded") and stepper.resumed is None
    stepper.following, stepper.stamp = "a", stamp
    assert pa._active_commit(log, stepper, data_path)["commit_id"] == "a"
    assert not stepper.superseded
    published = []

    async def publishing(*a, **k):
        published.append(1)
        return "accepted", ""

    monkeypatch.setattr(pa, "publish_commit", publishing)
    monkeypatch.setattr(pa, "_resume", lambda *a, **k: None)
    service = pa.make_service(ec, params, TOKEN, logger)
    service.log.begin("c1", [1.0], _HORIZON)
    res = pd.DataFrame({"P_deferrable0": 1.0}, index=pd.date_range(_now(), periods=2, freq="30min"))
    out = service._commit(None, Answer(np.zeros(2), 0.0, np.zeros(3), detail=res), "c1")
    assert out[0] == "refused" and published == []


def test_continual_publish_ends_a_plan_even_with_its_prefix_unknown(data_path, monkeypatch):
    import emhass.participant_api as pa

    case = _StepperCase(data_path)
    now = copy.deepcopy(case.params)
    now["optim_conf"].pop("participant_api")
    now["retrieve_hass_conf"]["continual_publish"] = True
    committed = copy.deepcopy(now)
    monkeypatch.setattr(pa, "_current_params", lambda conf, fallback: now)
    out = case.stepper.tick(
        case.gen,
        case.stamp,
        case.ec,
        committed,
        case.res,
        logger,
        None,
        lambda *a: ("accepted", ""),
        pd.Timestamp("2026-11-21T10:30:01Z"),
    )
    assert out == "stop" and case.stepper.superseded


# --------------------------------------------------------------------------
# CodeRabbit's review of 9c21e919
# --------------------------------------------------------------------------


def test_a_carried_plan_is_averaged_over_each_slots_time():
    """Each window slot runs, on average, what the plan runs over its time:
    a 20-minute plan [0, 6, 0] gives a 30-minute slot 10 minutes at 6 kW (2 kW,
    not the 3 kW of its two samples), and a plan that starts within a slot
    counts from its start."""
    from emhass.coordinator.backend import _carried

    rec = _record([0.0, 6.0, 0.0], step=20.0)
    assert np.allclose(_carried(rec, _at("2026-11-21T10:00:00+00:00", 2), True), [2.0, 2.0])
    # aligned and equal slots: as before
    rec = _record([7.0, 0.0])
    assert _carried(rec, _at("2026-11-21T10:00:00+00:00", 3), True).tolist() == [7, 0, 0]
    # a plan from 10:10: the slot from 10:00 runs it for 20 of its 30 minutes
    rec = _record([6.0], start="2026-11-21T10:10:00+00:00", step=60.0)
    assert np.allclose(_carried(rec, _at("2026-11-21T10:00:00+00:00", 2), True), [4.0, 6.0])
    # a window that ends before the plan begins: nothing of it
    assert _carried(rec, _at("2026-11-21T09:00:00+00:00", 1), True) is None
    # a remote that does not step is weighted the same way in that slot
    assert np.allclose(_carried(rec, _at("2026-11-21T10:00:00+00:00", 2), False), [4.0, 6.0])
    assert _carried(rec, _at("2026-11-21T09:00:00+00:00", 1), False) is None


def test_a_carried_plan_of_tiny_slots_needs_no_sample_per_slot():
    """A record whose slots are tiny (any step is valid) is carried by its own
    boundaries, not one sample per plan slot: 48 slots of 30 minutes over a
    two-value plan of 0.001 minutes (30,000 plan slots in each) take a few kB,
    and run its last value."""
    import tracemalloc

    from emhass.coordinator.backend import _carried

    rec = _record([5.0, 3.0], step=1e-3)
    tracemalloc.start()
    try:
        out = _carried(rec, _at("2026-11-21T10:00:00+00:00", 48), True)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000, peak  # one sample per plan slot: over 10 MB
    # 0.001 of the first slot's 30 minutes at 5 kW, then 3 kW for good
    assert out[0] == pytest.approx(3.0 + 2.0 * 1e-3 / 30.0) and np.allclose(out[1:], 3.0)


def test_a_commits_outcome_is_never_asked_for_past_its_wait():
    """The wait for a commit's outcome bounds every call: a status poll to a
    remote that hangs is cut off when the wait ends, not at the call's
    timeout, and no call starts once it has passed."""
    import time as _time

    from emhass.coordinator.remote import RemoteParticipant

    polls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, body):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path.endswith("/v1/describe"):
                return self._send(
                    200,
                    {
                        "version": 1,
                        "kinds": ["price_response"],
                        "max_power_kw": 1.0,
                        "features": ["commit_status"],
                    },
                )
            polls.append(self.path)
            _time.sleep(3.0)  # a remote that hangs on the status poll
            with contextlib.suppress(OSError):
                self._send(202, _RUNNING)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self._send(202, _RUNNING)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        rp = RemoteParticipant(
            "r",
            f"http://127.0.0.1:{server.server_address[1]}",
            PlanWindow(30, 2, "x"),
            timeout=10.0,
            commit_wait_s=0.4,
        )
        t0 = _time.monotonic()
        ok, _ = rp.commit({"query_id": "q"}, np.zeros(2))
        took = _time.monotonic() - t0
        assert ok is None  # not known: its part stays in the plan
        assert polls and took < 2.0, took  # cut off at the wait, not after 3 s
    finally:
        server.shutdown()
        server.server_close()


def test_a_call_is_cut_off_at_what_is_left_of_the_wait():
    """The default transport takes a shorter time for one call: a poll near
    the end of a commit's wait is cut off there, not at the call's timeout."""
    import socket
    import time as _time

    from emhass.coordinator.remote import urllib_transport

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)  # accepts, never answers
    try:
        send = urllib_transport(10.0)
        url = f"http://127.0.0.1:{server.getsockname()[1]}/v1/commit/q"
        t0 = _time.monotonic()
        with pytest.raises(TimeoutError):
            send("GET", url, {}, None, 0.3)
        assert _time.monotonic() - t0 < 2.0
    finally:
        server.close()


def test_a_commit_whose_record_cannot_be_saved_never_runs(tmp_path, monkeypatch):
    """A commit that ran without its record on disk would be lost to a
    restart: its plan in Home Assistant, nothing stepping or reporting it. So
    a record that cannot be saved refuses the commit (409), and nothing runs."""
    from emhass.coordinator import remote
    from emhass.coordinator.remote import CommitLog, CommitNotRecorded

    log = CommitLog(tmp_path / "participant_commits.json")
    real = remote.os.replace

    def fails(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(remote.os, "replace", fails)
    with pytest.raises(CommitNotRecorded):
        log.begin("a", [1.0])
    assert log.get("a") is None
    ran = []
    service = _service(lambda p, a: ran.append(1) or ("accepted", ""), log=log)
    status, problem = service.handle("POST", "/v1/commit", {}, _commit_body())
    assert status == 409 and "could not be saved" in problem["detail"]
    assert ran == [] and log.get("q1") is None
    monkeypatch.setattr(remote.os, "replace", real)
    status, _ = service.handle("POST", "/v1/commit", {}, _commit_body())
    assert status == 202
    assert _until_final(service)[1]["state"] == "accepted" and ran == [1]


def test_a_committed_plan_that_cannot_be_read_is_resumed_later(data_path, monkeypatch):
    """A committed plan that cannot be read back at a resume is tried again at
    the next, rather than never."""
    import emhass.participant_api as pa

    ec, stepper, followed, _ = _resume_case(data_path, monkeypatch, [("a", "accepted", {})])
    params = _params(ec, **GARAGE)
    read = pd.read_csv
    fails = {"n": 1}

    def flaky(*a, **k):
        if fails["n"]:
            fails["n"] -= 1
            raise OSError("busy")
        return read(*a, **k)

    monkeypatch.setattr(pa.pd, "read_csv", flaky)
    pa._resume(ec, params, logger, stepper, None)
    assert followed == [] and stepper.resumed is None
    pa._resume(ec, params, logger, stepper, None)
    assert [f[0] for f in followed] == ["a"]


def test_the_active_commit_is_never_superseded_by_a_stamp_read_before_it(monkeypatch):
    """The saved stamp is read while a commit may be followed on another
    thread: a stamp read before that commit never marks it superseded."""
    import emhass.participant_api as pa
    from emhass.coordinator.remote import CommitLog

    log = CommitLog()
    for cid in ("a", "b"):
        log.begin(cid, [1.0], _HORIZON)
        log.end(cid, "accepted", "")
    stepper = pa.PlanStepper()
    stepper.following, stepper.stamp = "a", "s1"

    def read_then_followed(conf):
        # the stamp read is s1; meanwhile commit b is followed, made over s2
        with stepper._lock:
            stepper._gen += 1
            stepper.following, stepper.stamp, stepper.superseded = "b", "s2", False
        return "s1", None

    monkeypatch.setattr(pa, "_latest_saved", read_then_followed)
    assert pa._active_commit(log, stepper, "/nowhere")["commit_id"] in ("a", "b")
    assert not stepper.superseded
    # a plan of its own replacing the one followed is still seen
    monkeypatch.setattr(pa, "_latest_saved", lambda conf: ("s3", None))
    assert pa._active_commit(log, stepper, "/nowhere") is None and stepper.superseded


@pytest.mark.parametrize("pv_all_sold", [False, True])
def test_the_default_solver_takes_a_remotes_supply_as_less_load_never_below_zero(
    monkeypatch, pv_all_sold
):
    """A remote supplying more than the house draws (3 kW against 0.4 kW):
    the default solver's load stays >= 0 W, so self-consumption still
    solves. The surplus is planned as generation (P_PV still shows the
    house's own) where that is exact; with set_total_pv_sell on it is not,
    and the log says it goes to the grid unplanned."""
    from emhass import optimization

    rh, oc, pc, data, pv, load, buy, sell = _house(_now())
    oc = copy.deepcopy(oc)
    oc.update(optimization_backend="dantzig_wolfe", set_total_pv_sell=pv_all_sold)
    opt = Optimization(
        rh,
        oc,
        copy.deepcopy(pc),
        "unit_load_cost",
        "unit_prod_price",
        "self-consumption",
        EMHASS_CONF,
        logger,
    )
    monkeypatch.setattr(optimization, "_coordinated_plan", lambda *a, **k: (None, "declined"))
    monkeypatch.setattr(optimization, "_remotes_running_w", lambda o: np.full(len(load), -3000.0))
    with (
        mock.patch.object(opt.logger, "warning") as warned,
        mock.patch.object(opt.logger, "info") as told,
    ):
        res = opt.perform_optimization(data, pv, load, buy, sell, soc_init=0.5, soc_final=0.5)
    assert str(res["optim_status"].iloc[0]).lower() == "optimal"
    assert (res["P_Load"] >= 0).all()
    assert np.allclose(res["P_Load"].values, np.maximum(np.asarray(load) - 3000.0, 0.0))
    assert np.allclose(res["P_PV"].values, np.asarray(pv))  # the house's own PV
    as_generation = any("planned as generation" in str(c) for c in told.call_args_list)
    unplanned = any("supply more than the house" in str(c) for c in warned.call_args_list)
    assert (as_generation, unplanned) == ((False, True) if pv_all_sold else (True, False))


def test_a_remotes_surplus_meets_the_houses_other_loads_within_the_import_limit(monkeypatch):
    """A remote supplying 3 kW, the house drawing 0.4 kW, and a 2 kW load that
    must run, under a 1 kW import limit: the meter would export 0.6 kW, so the
    default solver must find that schedule rather than see 2 kW of import."""
    from emhass import optimization

    rh, oc, pc, data, pv, load, buy, sell = _house(_now())
    oc = copy.deepcopy(oc)
    oc.update(
        optimization_backend="dantzig_wolfe",
        number_of_deferrable_loads=1,
        nominal_power_of_deferrable_loads=[2000],
        operating_hours_of_each_deferrable_load=[24],
        treat_deferrable_load_as_semi_cont=[True],
        set_deferrable_load_single_constant=[False],
        start_timesteps_of_each_deferrable_load=[0],
        end_timesteps_of_each_deferrable_load=[0],
        set_use_battery=False,
    )
    pc = copy.deepcopy(pc)
    pc.update(maximum_power_from_grid=1000, maximum_power_to_grid=1000)
    opt = Optimization(
        rh, oc, pc, "unit_load_cost", "unit_prod_price", "profit", EMHASS_CONF, logger
    )
    monkeypatch.setattr(optimization, "_coordinated_plan", lambda *a, **k: (None, "declined"))
    monkeypatch.setattr(optimization, "_remotes_running_w", lambda o: np.full(len(load), -3000.0))
    house = np.full(len(load), 400.0)
    res = opt.perform_optimization(
        data, np.zeros(len(load)), house, buy, sell, soc_init=0.5, soc_final=0.5
    )
    # the meter: 0.4 + 2 - 3 = -0.6 kW, an export within its limit. The solver
    # counts the surplus as generation up to the export limit (1 kW), so it
    # plans 1 kW of import at most: within the import limit, where it saw 2 kW
    assert str(res["optim_status"].iloc[0]).lower() == "optimal"
    assert np.allclose(res["P_deferrable0"].values, 2000.0)
    assert (res["P_grid"].values <= 1000.0 + 1e-6).all()


def test_a_resume_that_finds_its_commit_superseded_yields_to_a_commit_followed_meanwhile(
    data_path, monkeypatch
):
    """A resume that finds the last commit superseded marks it so only if no
    commit was followed meanwhile: one followed while it read the plans keeps
    being stepped."""
    import emhass.participant_api as pa

    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    before = plan_store.stamp(plan_store.read(data_path))
    ec, stepper, followed, log = _resume_case(
        data_path, monkeypatch, [("b", "accepted", {"plan_stamp": before})]
    )
    plan_store.record(data_path, [], "2026-11-21T11:00:00Z", "1.0")  # its own since
    read = pa._latest_saved

    def commit_followed_meanwhile(conf):
        with stepper._lock:  # as follow("c") does
            stepper._gen += 1
            stepper.following, stepper.stamp, stepper.superseded = "c", "sc", False
        return read(conf)

    monkeypatch.setattr(pa, "_latest_saved", commit_followed_meanwhile)
    pa._resume(ec, _params(ec, **GARAGE), logger, stepper, None)
    _, following, stamp, superseded = stepper.view()
    assert (following, stamp, superseded) == ("c", "sc", False)


def test_a_commit_whose_record_cannot_be_annotated_is_refused_before_publishing(
    data_path, monkeypatch
):
    """The stamp and prefix a commit is made with are saved before anything
    is published: if they cannot be, nothing is published (refused)."""
    import emhass.participant_api as pa
    from emhass.coordinator.interface import Answer

    ec = _conf(data_path)
    called = []

    async def publishing(*a, **k):
        called.append(1)
        return "accepted", ""

    monkeypatch.setattr(pa, "publish_commit", publishing)
    monkeypatch.setattr(pa, "_resume", lambda *a, **k: None)
    service = pa.make_service(ec, _params(ec, **GARAGE), TOKEN, logger)
    service.log.begin("c1", [1.0], _HORIZON)
    monkeypatch.setattr(service.log, "annotate", lambda cid, fields: False)
    res = pd.DataFrame({"P_deferrable0": 1.0}, index=pd.date_range(_now(), periods=2, freq="30min"))
    out = service._commit(None, Answer(np.zeros(2), 0.0, np.zeros(3), detail=res), "c1")
    assert out[0] == "refused" and "nothing was published" in out[1] and called == []


def test_the_default_solver_heats_the_house_with_its_own_load_alone(monkeypatch):
    """Thermal internal gains come from the house's own load: what a remote
    draws (counted in P_Load on a fallback) is not heat in this house."""
    from emhass import optimization

    rh, oc, pc, data, pv, load, buy, sell = _house(_now())
    oc = copy.deepcopy(oc)
    oc.update(optimization_backend="dantzig_wolfe")
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
    monkeypatch.setattr(optimization, "_coordinated_plan", lambda *a, **k: (None, "declined"))
    monkeypatch.setattr(optimization, "_remotes_running_w", lambda o: np.full(len(load), 3000.0))
    seen = []
    build = Optimization._add_deferrable_load_constraints

    def spy(self, *a):
        seen.append(np.asarray(a[-1], dtype=float))
        return build(self, *a)

    monkeypatch.setattr(Optimization, "_add_deferrable_load_constraints", spy)
    gains = []
    update = Optimization.update_thermal_params

    def spy_update(self, optim_conf, data_opt, p_load, *a, **k):
        gains.append(np.asarray(p_load, dtype=float))
        return update(self, optim_conf, data_opt, p_load, *a, **k)

    monkeypatch.setattr(Optimization, "update_thermal_params", spy_update)
    res = opt.perform_optimization(data, pv, load, buy, sell, soc_init=0.5, soc_final=0.5)
    assert np.allclose(res["P_Load"].values, np.asarray(load) + 3000.0)
    assert seen and all(np.allclose(s, load) for s in seen)
    # planned again over the same problem: its thermal inputs refreshed alike
    opt.perform_optimization(data, pv, load, buy, sell, soc_init=0.5, soc_final=0.5)
    assert all(np.allclose(g, load) for g in gains)


def test_a_failed_note_leaves_the_record_as_it_was(tmp_path, monkeypatch):
    """A field that could not be saved is not kept in memory either, so a
    later save never writes it."""
    from emhass.coordinator import remote
    from emhass.coordinator.remote import CommitLog

    log = CommitLog(tmp_path / "participant_commits.json")
    log.begin("c1", [1.0])

    def fails(*a, **k):
        raise OSError("disk full")

    real = remote.os.replace
    monkeypatch.setattr(remote.os, "replace", fails)
    assert log.annotate("c1", {"note": 1.0}) is False
    assert "note" not in log.get("c1")
    monkeypatch.setattr(remote.os, "replace", real)
    assert log.annotate("c1", {"plan_stamp": "s"}) is True
    stored = json.loads((tmp_path / "participant_commits.json").read_text())
    assert "note" not in stored[0] and stored[0]["plan_stamp"] == "s"
    assert log.annotate("nope", {"x": 1}) is None


def test_a_commit_refused_for_its_record_keeps_no_plan_to_resume(data_path, monkeypatch):
    """Refused because its record could not be saved, the commit keeps no
    plan: its record may still read running on disk, and a restart then
    finds nothing of it to step."""
    import emhass.participant_api as pa
    from emhass.coordinator.interface import Answer

    ec = _conf(data_path)

    async def publishing(*a, **k):
        raise AssertionError("never published")

    monkeypatch.setattr(pa, "publish_commit", publishing)
    monkeypatch.setattr(pa, "_resume", lambda *a, **k: None)
    service = pa.make_service(ec, _params(ec, **GARAGE), TOKEN, logger)
    service.log.begin("c1", [1.0], _HORIZON)
    monkeypatch.setattr(service.log, "annotate", lambda cid, fields: False)
    res = pd.DataFrame({"P_deferrable0": 1.0}, index=pd.date_range(_now(), periods=2, freq="30min"))
    out = service._commit(None, Answer(np.zeros(2), 0.0, np.zeros(3), detail=res), "c1")
    assert out[0] == "refused" and not pa.committed_plan_path(ec, "c1").exists()


def test_a_commit_whose_plan_cannot_be_removed_is_not_reported_refused(data_path, monkeypatch):
    """Its record not saved and its kept plan not removable: a restart may
    step it, so it is not reported refused - it raises, which the service
    records as a commit that may have run (partial)."""
    import emhass.participant_api as pa
    from emhass.coordinator.interface import Answer

    ec = _conf(data_path)

    async def publishing(*a, **k):
        raise AssertionError("never published")

    monkeypatch.setattr(pa, "publish_commit", publishing)
    monkeypatch.setattr(pa, "_resume", lambda *a, **k: None)
    service = pa.make_service(ec, _params(ec, **GARAGE), TOKEN, logger)
    service.log.begin("c1", [1.0], _HORIZON)
    monkeypatch.setattr(service.log, "annotate", lambda cid, fields: False)

    def cannot(self, missing_ok=False):
        raise PermissionError("read-only")

    monkeypatch.setattr(pathlib.Path, "unlink", cannot)
    res = pd.DataFrame({"P_deferrable0": 1.0}, index=pd.date_range(_now(), periods=2, freq="30min"))
    with pytest.raises(OSError, match="a restart may run it"):
        service._commit(None, Answer(np.zeros(2), 0.0, np.zeros(3), detail=res), "c1")


# --------------------------------------------------------------------------
# A commit's plan knows its commit (plan_store's origin)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("annotated", [False, True])
def test_a_commits_own_plan_is_known_by_its_origin_after_a_stop(data_path, monkeypatch, annotated):
    """A commit stopped after its plan was saved - before its stamp was noted
    (`annotated=False`), or a record that could not be saved - is resumed:
    the plan says it is that commit's, however long the save took. Then a
    plan of this EMHASS's own supersedes it."""
    import emhass.participant_api as pa

    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    before = plan_store.stamp(plan_store.read(data_path))
    ec, stepper, followed, log = _resume_case(data_path, monkeypatch, [("b", "running", {})])
    log.annotate("b", {"plan_stamp": before})
    log._records["b"].update(state="partial", interrupted=annotated is False)
    plan_store.record(data_path, [], "2026-11-21T11:30:00Z", "1.0", origin="commit:b")
    own = plan_store.stamp(plan_store.read(data_path))
    params = _params(ec, **GARAGE)
    pa._resume(ec, params, logger, stepper, None)
    assert followed == [("b", own)] and log.get("b")["plan_stamp"] == own
    assert log.get("b").get("interrupted") in (None, False)
    plan_store.record(data_path, [], "2026-11-21T12:00:00Z", "1.0")  # its own run
    stepper.resumed = None
    pa._resume(ec, params, logger, stepper, None)
    assert followed == [("b", own)] and stepper.view()[3]


@pytest.mark.parametrize("origin", [None, "commit:other"])
def test_a_plan_without_the_commits_origin_supersedes_it(data_path, monkeypatch, origin):
    """A plan saved since the commit's stamp that is not that commit's - of
    this EMHASS's own, or another commit's - supersedes it, whatever its
    content or timing."""
    import emhass.participant_api as pa

    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    before = plan_store.stamp(plan_store.read(data_path))
    ec, stepper, followed, log = _resume_case(data_path, monkeypatch, [("b", "running", {})])
    log.annotate("b", {"plan_stamp": before})
    log._records["b"].update(state="partial", interrupted=True)
    plan_store.record(data_path, [], "2026-11-21T10:00:01Z", "1.0", origin=origin)
    pa._resume(ec, _params(ec, **GARAGE), logger, stepper, None)
    assert followed == [] and stepper.view()[3]


def test_a_stepper_keeps_stepping_its_commits_own_plan_under_another_stamp(data_path):
    """The stamp a stepper was given may be stale (its record could not be
    saved): a plan saved by its own commit is still its plan, stepped on; a
    plan of this EMHASS's own ends it."""
    case = _StepperCase(data_path)
    case.stepper.following = "c"
    plan_store.record(data_path, [], "2026-11-21T10:00:05Z", "1.0", origin="commit:c")
    assert case.tick("2026-11-21T10:30:01Z") == "published"
    plan_store.record(data_path, [], "2026-11-21T10:40:00Z", "1.0")
    assert case.tick("2026-11-21T11:00:01Z") == "stop" and case.stepper.superseded


def test_a_commit_names_its_plan_and_follows_its_stamp_even_if_its_record_fails(
    data_path, monkeypatch
):
    """The commit's plan is saved with its origin; the stepper follows the
    stamp it was saved under even when that stamp cannot be recorded (the
    record keeps the one before), and returns it for the final record."""
    import emhass.participant_api as pa
    from emhass.coordinator.interface import Answer

    ec = _conf(data_path)
    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    seen = {}

    async def publishing(conf, params, res, logger, on_recorded=None, origin=None):
        seen["origin"] = origin
        plan_store.record(data_path, [], "2026-11-21T10:05:00Z", "1.0", origin=origin)
        on_recorded(plan_store.stamp(plan_store.read(data_path)))
        return "accepted", ""

    monkeypatch.setattr(pa, "publish_commit", publishing)
    monkeypatch.setattr(pa, "_resume", lambda *a, **k: None)
    followed = []
    stepper = pa.plan_stepper(ec)
    monkeypatch.setattr(stepper, "follow", lambda *a, **k: followed.append(a[:2]))
    service = pa.make_service(ec, _params(ec, **GARAGE), TOKEN, logger)
    log = service.log
    log.begin("c1", [1.0], _HORIZON)
    annotate = log.annotate
    monkeypatch.setattr(
        log,
        "annotate",
        lambda cid, fields: False if set(fields) == {"plan_stamp"} else annotate(cid, fields),
    )
    res = pd.DataFrame({"P_deferrable0": 1.0}, index=pd.date_range(_now(), periods=2, freq="30min"))
    out = service._commit(None, Answer(np.zeros(2), 0.0, np.zeros(3), detail=res), "c1")
    saved = plan_store.stamp(plan_store.read(data_path))
    assert seen["origin"] == "commit:c1" and out[0] == "accepted"
    assert followed == [("c1", saved)] and out[2] == {"plan_stamp": saved}


def test_a_plan_not_saved_never_names_a_stamp(data_path, monkeypatch):
    """publish_commit reports a stamp only for a plan it saved: one that
    could not be saved never hands over the plan before's stamp."""
    from emhass import command_line

    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    res = _a_plan(ec, params)
    monkeypatch.setattr(command_line.plan_store, "record", lambda *a, **k: None)
    stamps = []
    (state, why), _ = _commit(ec, params, res, [], on_recorded=stamps.append)
    assert state == "accepted", why
    assert stamps == []


def test_the_plan_endpoint_never_serves_a_plans_origin(data_path, monkeypatch):
    """What made a plan (a commit) is this server's own bookkeeping."""
    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0", origin="commit:c1")
    monkeypatch.setattr(
        web_server, "emhass_conf", {**web_server.emhass_conf, "data_path": data_path}
    )

    async def get():
        async with web_server.app.app_context():
            response = await web_server.api_v1_plan()
            return json.loads(await response.get_data())

    body = asyncio.run(get())
    assert body["status"] == "ok" and "origin" not in body


# --------------------------------------------------------------------------
# CodeRabbit's review of 1d68bf81
# --------------------------------------------------------------------------


def test_the_remote_commit_file_keeps_every_other_runs_record(tmp_path):
    """Two runs sharing a data folder: each records only the remotes it
    committed, so one never drops the other's record (read and written under
    one lock)."""
    from emhass.coordinator import backend

    opt = mock.MagicMock(fed_dry_run=False)
    path = tmp_path / "coordinator_remote_commits.json"
    a, b = {"state": "committed", "url": "http://a"}, {"state": "committed", "url": "http://b"}
    snapshot = backend._read_remote_commits(opt)  # both runs read it empty
    assert snapshot == {}
    backend._write_remote_commits(opt, {"garage": a})
    backend._write_remote_commits(opt, {"shed": b})  # the other run, from its older view
    assert json.loads(path.read_text()) == {"garage": a, "shed": b}
    backend._write_remote_commits(opt, {"garage": None})
    assert json.loads(path.read_text()) == {"shed": b}
    assert (tmp_path / "coordinator_remote_commits.json.lock").exists()


@pytest.mark.parametrize("start", ["NaT", "not a time"])
def test_a_record_whose_start_is_no_time_carries_nothing(start):
    """A saved record with no time for its start never becomes non-finite
    load in the default solver: it is not carried."""
    from emhass.coordinator.backend import _carried

    rec = _record([7.0, 0.0], start=start)
    for steps in (True, False):
        assert _carried(rec, _at("2026-11-21T10:00:00+00:00"), steps) is None
    assert _carried(_record([7.0]), _at("NaT"), True) is None


@pytest.mark.parametrize("site", [{}, "invalid", 42])
def test_a_site_that_is_not_a_list_holds_what_was_committed(tmp_path, site):
    """A configured site that cannot be read (not a list) is not an absent
    one: the remotes this coordinator committed are still held."""
    from emhass.coordinator import backend

    record = {**_record([7.0, 7.0, 0.0, 0.0]), "url": "http://g", "steps": True}
    (tmp_path / "coordinator_remote_commits.json").write_text(json.dumps({"garage": record}))
    opt = mock.MagicMock(fed_dry_run=False, optim_conf={"site": site}, time_step=0.5)
    data_opt = pd.DataFrame(index=pd.date_range("2026-11-21T10:00:00Z", periods=4, freq="30min"))
    running, _, _ = backend.note_running(opt, data_opt)
    assert running["garage"].tolist() == [7.0, 7.0, 0.0, 0.0]
    for empty in (None, []):
        opt.optim_conf = {"site": empty}
        assert backend.note_running(opt, data_opt)[0] == {}


# 2 s: its last slot is tried until it ends; 0.5 s: the plan has ended by its
# first try (a slow runner, a try past its slot), which ends the loop at once
@pytest.mark.parametrize("step_s", [2.0, 0.5])
def test_a_stepped_plan_whose_last_slots_were_not_published_is_held_at_what_was(
    data_path, request, step_s
):
    """The final slot's publish failed: Home Assistant keeps the slot before.
    The commit records that slot, and a coordinator carries its value after
    the plan's end, not the plan's last."""
    import emhass.participant_api as pa
    from emhass.coordinator.backend import _carried
    from emhass.coordinator.remote import CommitLog, commit_view

    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    log = CommitLog()
    log.begin("c", [7.0, 7.0, 0.0], _HORIZON)
    log.end("c", "accepted", "")
    stepper = pa.PlanStepper()
    stepper.log = log
    # mid-minute, so the minute publishing rounds to stays the same throughout
    traveller = time_machine.travel(
        pd.Timestamp.now(tz="UTC").floor("min") + pd.Timedelta(seconds=20), tick=True
    )
    traveller.start()
    request.addfinalizer(traveller.stop)
    now = pd.Timestamp.now(tz="UTC")
    # slot 0 now (its commit published it), slot 1 - the last - in a moment
    res = pd.DataFrame(
        {"P_deferrable0": [7000.0, 7000.0]},
        index=pd.DatetimeIndex([now - pd.Timedelta(seconds=0.1), now + pd.Timedelta(seconds=0.9)]),
    )
    stepper.RETRY_S = 0.05
    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    stamp = plan_store.stamp(plan_store.read(data_path))

    def refuses(*a):
        return "partial", "Home Assistant refused it"

    # its commit published slot 0 (accepted); slot 1, the last, never publishes
    with mock.patch.object(pa, "_step", lambda p: pd.Timedelta(seconds=step_s)):
        stepper.follow("c", stamp, ec, params, res, logger, publish=refuses)
        # its last slot retried until the slot ends, then the plan ends: a
        # few seconds, longer on a slow runner
        for _ in range(1000):
            if (log.get("c") or {}).get("held_slot") is not None:
                break
            threading.Event().wait(0.02)
    stepper.stop()
    assert stepper.join(5.0)
    record = commit_view(log.get("c"))
    assert record["held_slot"] == 0
    rec = {**_record([7.0, 0.0]), "held_slot": 0}
    assert _carried(rec, _at("2026-11-21T11:00:00+00:00", 2), True).tolist() == [7.0, 7.0]
    assert _carried(_record([7.0, 0.0]), _at("2026-11-21T11:00:00+00:00", 2), True).tolist() == [
        0,
        0,
    ]


@pytest.mark.parametrize(
    "field, value",
    [("price_draw", ["0.1", 0.2]), ("price_draw", [True, 0.2]), ("price_supply", [None, 0.1])],
)
def test_a_query_with_arrays_of_anything_but_numbers_is_a_bad_request(field, value):
    service = _service(lambda p, a: ("accepted", ""))
    query = {
        "version": SCHEMA_VERSION,
        "kind": "price_response",
        "horizon": _HORIZON,
        "price_draw": [0.1, 0.2],
        field: value,
    }
    status, problem = service.handle("POST", "/v1/query", {}, json.dumps(query).encode())
    assert status == 400 and field in problem["detail"]


def test_a_commit_sent_again_for_another_horizon_is_refused():
    """A commit replayed with its plan but another horizon is not that
    commit: refused (409), the record unchanged; the same horizon, written
    another way, is the same."""
    service = _service(lambda p, a: ("accepted", ""))
    assert service.handle("POST", "/v1/commit", {}, _commit_body())[0] == 202
    _until_final(service)
    body = json.loads(_commit_body())
    body["horizon"] = {**_HORIZON, "start": "2026-11-21T10:30:00+00:00"}
    status, problem = service.handle("POST", "/v1/commit", {}, json.dumps(body).encode())
    assert status == 409 and "another horizon" in problem["detail"]
    body["horizon"] = {**_HORIZON, "start": "2026-11-21T11:00:00+01:00"}  # the same instant
    assert service.handle("POST", "/v1/commit", {}, json.dumps(body).encode())[0] == 200


def test_a_commit_whose_slot_passes_while_it_waits_for_the_lock_never_runs():
    """A commit waits for the live-optimization slot; if its horizon's first
    slot has passed by then, it is refused (409) and nothing runs."""
    from emhass.coordinator.remote import CommitLog, WindowRefused

    now = {"start": _HORIZON["start"]}
    released, ran = [], []

    def current(window):
        if window.start != now["start"]:
            raise WindowRefused("the coordinator and this EMHASS must agree on the slot")

    def begin():
        now["start"] = "2026-11-21T10:30:00+00:00"  # the slot passed while it waited
        return "ok", lambda: released.append(1)

    log = CommitLog()
    service = ParticipantService(
        lambda: {"key": "fixed", "max_power_kw": 1.0},
        lambda window: _Fixed(),
        lambda p, a: ran.append(1) or ("accepted", ""),
        begin=begin,
        log=log,
        current=current,
    )
    assert (
        service.handle("POST", "/v1/baseline", {"Query-Id": "q1"}, json.dumps(_HORIZON).encode())[0]
        == 200
    )
    status, _ = service.handle("POST", "/v1/commit", {}, _commit_body())
    assert status == 409 and ran == [] and released == [1] and log.get("q1") is None


def test_a_remote_token_with_the_value_of_one_of_emhass_own_secrets_is_never_sent():
    """A token_secret of its own name, holding Home Assistant's token (or the
    API token): never among the remotes' tokens."""
    site = [
        {"id": "grid"},
        {
            "id": "elsewhere",
            "parent": "grid",
            "solver": {"url": "http://elsewhere.example/x", "token_secret": "garage_token"},
        },
    ]

    async def build(secrets):
        config = orjson.loads((root / "src/emhass/data/config_defaults.json").read_bytes())
        config["site"] = site
        return await build_params(EMHASS_CONF, secrets, config, logger)

    for own in ({"long_lived_token": "HA"}, {"emhass_api_token": "HA"}):
        out = asyncio.run(build({**own, "garage_token": "HA"}))
        assert "remote_tokens" not in out["retrieve_hass_conf"]
    out = asyncio.run(build({"long_lived_token": "HA", "garage_token": "G"}))
    assert out["retrieve_hass_conf"]["remote_tokens"] == {"garage_token": "G"}


def test_each_remote_token_map_has_its_own_cached_optimizer():
    """The optimizer cache keys a salted digest of the remotes' tokens: a
    request with other tokens never shares the optimizer (nor its tokens)
    with one in flight, and no token appears in the key."""
    from emhass.command_line import OptimizationCache

    rh = {"optimization_time_step": pd.Timedelta(minutes=30), "remote_tokens": {"g": "s3cr3t-a"}}
    oc = {"number_of_deferrable_loads": 0}
    pc = {}
    a = OptimizationCache._compute_cache_key(oc, pc, "profit", rh)
    b = OptimizationCache._compute_cache_key(
        oc, pc, "profit", {**rh, "remote_tokens": {"g": "s3cr3t-b"}}
    )
    c = OptimizationCache._compute_cache_key(oc, pc, "profit", dict(rh))
    assert a != b and a == c
    assert "s3cr3t" not in repr(a) and "s3cr3t" not in repr(b)


@pytest.mark.parametrize("limit", ["scalar", "per slot"])
def test_a_remotes_surplus_planned_as_generation_stays_within_the_export_limit(monkeypatch, limit):
    """The export limit is hard and no PV can be curtailed: the surplus is
    planned as generation only up to what the grid can take beside the
    house's own PV, so a fallback that solved before still solves."""
    from emhass import optimization

    rh, oc, pc, data, pv, load, buy, sell = _house(_now())
    oc = copy.deepcopy(oc)
    oc.update(optimization_backend="dantzig_wolfe", set_use_battery=False)
    pc = copy.deepcopy(pc)
    # a scalar, or a limit per slot (5 kW, then 4.6 kW from the 10th slot)
    n = len(load)
    limits = [5000.0] * 10 + [4600.0] * (n - 10)
    pc.update(maximum_power_to_grid=5000 if limit == "scalar" else limits)
    opt = Optimization(
        rh, oc, pc, "unit_load_cost", "unit_prod_price", "profit", EMHASS_CONF, logger
    )
    monkeypatch.setattr(optimization, "_coordinated_plan", lambda *a, **k: (None, "declined"))
    monkeypatch.setattr(optimization, "_remotes_running_w", lambda o: np.full(len(load), -3000.0))
    house_pv = np.full(len(load), 4000.0)
    res = opt.perform_optimization(
        data, house_pv, np.full(len(load), 400.0), buy, sell, soc_init=0.5, soc_final=0.5
    )
    assert str(res["optim_status"].iloc[0]).lower() == "optimal"
    assert np.allclose(res["P_PV"].values, house_pv)
    bound = np.full(n, 5000.0) if limit == "scalar" else np.asarray(limits)
    assert (res["P_grid"].values >= -bound - 1e-6).all()


@pytest.mark.parametrize(
    "rounding, expected", [("nearest", 1), ("first", 1), ("last", 2), (None, 1)]
)
def test_the_slot_recorded_published_is_the_one_publishing_picks(rounding, expected):
    from emhass.participant_api import _slot_of

    index = pd.date_range("2026-11-21T10:00:00Z", periods=4, freq="30min")
    at = pd.Timestamp("2026-11-21T10:40:30Z")  # within slot 1; the minute: 10:40
    assert _slot_of(index, at, rounding) == expected
    assert _slot_of(index, pd.Timestamp("2026-11-21T10:30:01Z"), rounding) == 1  # a boundary


def test_an_ended_plan_published_again_clears_what_it_held(data_path):
    """A plan that ended with its last slot not published records the slot
    held; published again after a restart (its last value), that record is
    cleared, so a coordinator carries the last value again."""
    import emhass.participant_api as pa
    from emhass.coordinator.remote import CommitLog, commit_view

    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    log = CommitLog()
    log.begin("c", [7.0, 0.0], _HORIZON)
    log.end("c", "accepted", "", {"held_slot": 0})
    assert commit_view(log.get("c"))["held_slot"] == 0
    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    stamp = plan_store.stamp(plan_store.read(data_path))
    res = pd.DataFrame(
        {"P_deferrable0": [7000.0, 0.0]},
        index=pd.date_range("2026-11-21T09:00:00Z", periods=2, freq="30min"),  # ended
    )
    stepper = pa.PlanStepper()
    stepper.log = log
    stepper.follow(
        "c", stamp, ec, params, res, logger, publish=lambda *a: ("accepted", ""), now_first=True
    )
    for _ in range(200):
        if "held_slot" not in commit_view(log.get("c")):
            break
        threading.Event().wait(0.02)
    stepper.stop()
    assert stepper.join(5.0)
    assert "held_slot" not in commit_view(log.get("c"))


def test_an_unreadable_commit_file_is_never_rewritten(tmp_path):
    """A commit file there but not readable now is left as it is: rewriting
    it would lose every other remote's record."""
    from emhass.coordinator import backend

    opt = mock.MagicMock(fed_dry_run=False)
    path = tmp_path / "coordinator_remote_commits.json"
    path.write_text('{"garage": {"state": "committed"')  # cut short
    backend._write_remote_commits(opt, {"shed": {"state": "committed"}})
    assert path.read_text() == '{"garage": {"state": "committed"'
    assert any("cannot be read" in str(c) for c in opt.logger.warning.call_args_list)


def test_the_commit_file_is_written_without_a_file_lock_when_none_exists(tmp_path, monkeypatch):
    """No file-locking primitive: written under this process's lock alone,
    said once, rather than never recorded."""
    from emhass import forecast
    from emhass.coordinator import backend

    for name in ("_fcntl", "_msvcrt"):
        monkeypatch.delattr(forecast, name, raising=False)
    monkeypatch.setattr(backend, "_warned_no_file_lock", False)
    opt = mock.MagicMock(fed_dry_run=False)
    backend._write_remote_commits(opt, {"garage": {"state": "committed"}})
    backend._write_remote_commits(opt, {"shed": {"state": "committed"}})
    stored = json.loads((tmp_path / "coordinator_remote_commits.json").read_text())
    assert set(stored) == {"garage", "shed"}
    warnings = [c for c in opt.logger.warning.call_args_list if "no file lock" in str(c)]
    assert len(warnings) == 1


def test_a_remote_token_that_is_this_emhass_participant_token_is_sent():
    """Two EMHASS coordinating each other with one token: the remote expects
    the value this EMHASS's participant_token holds, which it already has."""
    site = [
        {"id": "grid"},
        {
            "id": "elsewhere",
            "parent": "grid",
            "solver": {"url": "http://elsewhere.example/x", "token_secret": "garage_token"},
        },
    ]

    async def build(secrets):
        config = orjson.loads((root / "src/emhass/data/config_defaults.json").read_bytes())
        config["site"] = site
        return await build_params(EMHASS_CONF, secrets, config, logger)

    out = asyncio.run(build({"participant_token": "shared", "garage_token": "shared"}))
    assert out["retrieve_hass_conf"]["remote_tokens"] == {"garage_token": "shared"}


def test_two_plans_saved_in_one_second_are_told_apart(data_path, monkeypatch):
    """A commit's plan and a plan of this EMHASS's own saved within the same
    second share a generated_at: each save has its own plan_id, so the later
    one supersedes the commit - on resume and while stepping - and a plan
    saved before plan ids is still known by its generated_at."""
    import emhass.participant_api as pa

    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0", origin="commit:b")
    own = plan_store.stamp(plan_store.read(data_path))
    assert own != "2026-11-21T10:00:00Z"
    ec, stepper, followed, log = _resume_case(
        data_path, monkeypatch, [("b", "accepted", {"plan_stamp": own})]
    )
    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")  # the same second
    pa._resume(ec, _params(ec, **GARAGE), logger, stepper, None)
    assert followed == [] and stepper.view()[3]
    case = _StepperCase(data_path)
    case.stepper.following = "c"
    case.stamp = plan_store.stamp(plan_store.read(data_path))
    plan_store.record(data_path, [], plan_store.read(data_path)["generated_at"], "1.0")
    assert case.tick("2026-11-21T10:30:01Z") == "stop" and case.stepper.superseded
    assert plan_store.stamp({"generated_at": "2026-11-21T10:00:00Z"}) == "2026-11-21T10:00:00Z"


def test_a_commit_refused_while_publishing_keeps_no_plan_to_resume(data_path, monkeypatch):
    """publish_commit refuses it (its inputs cannot be built): its kept plan
    is removed, so a restart - its final record not saved, read back as
    running - finds nothing of it to step, with a fresh log and stepper."""
    import emhass.participant_api as pa
    from emhass.coordinator.interface import Answer
    from emhass.coordinator.remote import CommitLog

    ec = _conf(data_path)

    async def refusing(*a, **k):
        return "refused", "EMHASS could not build its inputs to publish"

    monkeypatch.setattr(pa, "publish_commit", refusing)
    resume = pa._resume
    monkeypatch.setattr(pa, "_resume", lambda *a, **k: None)
    service = pa.make_service(ec, _params(ec, **GARAGE), TOKEN, logger)
    service.log.begin("c1", [1.0], _HORIZON)  # on disk: running
    res = pd.DataFrame({"P_deferrable0": 1.0}, index=pd.date_range(_now(), periods=2, freq="30min"))
    out = service._commit(None, Answer(np.zeros(2), 0.0, np.zeros(3), detail=res), "c1")
    assert out[0] == "refused" and not pa.committed_plan_path(ec, "c1").exists()
    # the final record never saved: a restart reads it back as running
    again = CommitLog(data_path / pa.COMMIT_LOG_FILE)
    assert again.get("c1")["state"] == "partial"
    monkeypatch.setattr(pa, "_commit_logs", {data_path / pa.COMMIT_LOG_FILE: again})
    stepper = pa.PlanStepper()
    followed = []
    monkeypatch.setattr(stepper, "follow", lambda *a, **k: followed.append(a[:2]))
    resume(ec, _params(ec, **GARAGE), logger, stepper, None)
    assert followed == [] and again.get("c1")["state"] == "refused"


def test_a_commits_stamp_is_its_own_snapshots_not_one_saved_after_it(data_path, monkeypatch):
    """A command-line run (which the live-optimization lock does not hold
    back) saves a plan right after the commit's: the commit is stamped with
    its own snapshot's id, as saved, never with the plan read back after."""
    import emhass.participant_api as pa

    ec = _conf(data_path)
    params = _params(ec, **GARAGE)
    res = _a_plan(ec, params)
    snapshot = pa._record_optim_snapshot

    own = []

    def then_a_local_run(*a, **k):
        stamp = snapshot(*a, **k)
        own.append(stamp)
        plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")  # the CLI's
        return stamp

    monkeypatch.setattr(pa, "_record_optim_snapshot", then_a_local_run)
    stamps = []
    (state, why), _ = _commit(ec, params, res, [], on_recorded=stamps.append, origin="commit:c1")
    assert state == "accepted", why
    latest = plan_store.read(data_path)
    assert stamps == own and own[0] is not None  # its own snapshot's, exactly, once
    assert stamps[0] != plan_store.stamp(latest)  # the latest is the local plan
    assert latest.get("origin") is None


class _Kept:
    """A participant whose one answer carries a plan to publish (detail)."""

    key, max_power_kw, modulating, onoff = "fixed", 1.0, False, True

    def __init__(self, res):
        self._answer = Answer(np.array([1.0, 0.0]), 0.0, np.zeros(3), detail=res)

    def baseline(self):
        return self._answer

    def respond(self, query):
        return self._answer


def test_a_refusal_whose_final_record_is_not_saved_is_never_resumed(data_path, monkeypatch):
    """Through the service's own commit flow: publish_commit refuses it, and
    the final record cannot be saved (the disk still says running). A
    restart, with a fresh log and stepper, finds no plan of it to step and
    ends it as refused."""
    import emhass.participant_api as pa
    from emhass.coordinator import remote
    from emhass.coordinator.remote import CommitLog

    ec = _conf(data_path)
    disk_full = {"on": False}
    replace = remote.os.replace

    def replacing(*a, **k):
        if disk_full["on"]:
            raise OSError("disk full")
        return replace(*a, **k)

    async def refusing(*a, **k):
        disk_full["on"] = True  # from here on, no record is saved
        return "refused", "EMHASS could not build its inputs to publish"

    monkeypatch.setattr(remote.os, "replace", replacing)
    monkeypatch.setattr(pa, "publish_commit", refusing)
    resume = pa._resume
    monkeypatch.setattr(pa, "_resume", lambda *a, **k: None)
    built = pa.make_service(ec, _params(ec, **GARAGE), TOKEN, logger)
    res = pd.DataFrame({"P_deferrable0": 1.0}, index=pd.date_range(_now(), periods=2, freq="30min"))
    service = ParticipantService(
        lambda: {"key": "fixed", "max_power_kw": 1.0},
        lambda window: _Kept(res),
        built._commit,
        log=built.log,
    )
    body = json.dumps(_HORIZON).encode()
    assert service.handle("POST", "/v1/baseline", {"Query-Id": "q1"}, body)[0] == 200
    assert service.handle("POST", "/v1/commit", {}, _commit_body())[0] == 202
    status, record = _until_final(service)
    assert record["state"] == "refused"  # as the coordinator is told
    disk_full["on"] = False
    stored = json.loads((data_path / pa.COMMIT_LOG_FILE).read_text())
    assert [r["state"] for r in stored if r["commit_id"] == "q1"] == ["running"]
    # the restart
    again = CommitLog(data_path / pa.COMMIT_LOG_FILE)
    monkeypatch.setattr(pa, "_commit_logs", {data_path / pa.COMMIT_LOG_FILE: again})
    stepper = pa.PlanStepper()
    followed = []
    monkeypatch.setattr(stepper, "follow", lambda *a, **k: followed.append(a[:2]))
    resume(ec, _params(ec, **GARAGE), logger, stepper, None)
    assert followed == [] and again.get("q1")["state"] == "refused"


# --------------------------------------------------------------------------
# CodeRabbit's review of 855e4e04
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reported", [None, "older"])
def test_what_a_remote_reports_never_overwrites_a_newer_commit(tmp_path, reported):
    """Run A reads the garage's record, asks the garage; run B commits it
    meanwhile. A's reconciliation - forgetting the record (the garage reports
    none) or replacing it with what it reported - leaves B's newer commit."""
    from emhass.coordinator import backend

    opt = mock.MagicMock(fed_dry_run=False)
    path = tmp_path / "coordinator_remote_commits.json"
    first = {**_record([7.0]), "url": "http://g", "steps": True, "max_power_kw": 8.0}
    path.write_text(json.dumps({"garage": first}))
    saved = backend._read_remote_commits(opt)  # what run A read
    newer = {**_record([0.0]), "state": "committed", "url": "http://g"}
    backend._write_remote_commits(opt, {"garage": newer})  # run B's commit
    # what it reports differs from what A read (A would write it)
    active = None if reported is None else _record([5.0])
    rp = mock.MagicMock(key="garage", url="http://g", active_commit=active, max_power_kw=8.0)
    rp.features = frozenset({"plan_steps"})
    backend._heard(opt, {}, saved, rp, _at("2026-11-21T10:00:00+00:00"))
    assert json.loads(path.read_text()) == {"garage": newer}
    # with nothing newer, the reconciliation is made
    saved = backend._read_remote_commits(opt)
    rp.active_commit = None
    backend._heard(opt, {}, saved, rp, _at("2026-11-21T10:00:00+00:00"))
    assert json.loads(path.read_text()) == {}


def test_a_naive_horizon_start_is_utc_for_the_participant(data_path):
    """A naive horizon start is UTC, as everywhere in EMHASS (and in
    _same_horizon), never this EMHASS's local time."""
    import time_machine as travel

    params = _params(_conf(data_path), **GARAGE)
    params["retrieve_hass_conf"]["time_zone"] = "Europe/Paris"
    check = participant_api.current_slot_check(params)
    with travel.travel(pd.Timestamp("2026-11-21T10:07:00Z").to_pydatetime(), tick=False):
        utc = _now()
        check(PlanWindow(30, 48, utc.tz_convert("UTC").tz_localize(None).isoformat()))
        with pytest.raises(WindowRefused):
            # the same wall-clock time read as Paris: an hour off
            check(PlanWindow(30, 48, utc.tz_convert("Europe/Paris").tz_localize(None).isoformat()))


def test_a_resume_takes_the_stamp_and_origin_from_one_read(data_path, monkeypatch):
    """A commit saving its own plan while a resume reads the plan store: the
    resume takes the stamp and the origin from one snapshot, so the commit is
    never recorded under another plan's stamp."""
    import emhass.participant_api as pa

    plan_store.record(data_path, [], "2026-11-21T10:00:00Z", "1.0")
    before = plan_store.stamp(plan_store.read(data_path))
    ec, stepper, followed, log = _resume_case(
        data_path, monkeypatch, [("b", "accepted", {"plan_stamp": before})]
    )
    plan_store.record(data_path, [], "2026-11-21T10:05:00Z", "1.0")  # of its own
    local = plan_store.stamp(plan_store.read(data_path))
    read = plan_store.read
    reads = []

    def then_the_commit_saves(path):
        out = read(path)
        reads.append(1)
        if len(reads) == 1:  # right after the resume's read: b saves its own
            plan_store.record(data_path, [], "2026-11-21T10:06:00Z", "1.0", origin="commit:b")
        return out

    monkeypatch.setattr(pa.plan_store, "read", then_the_commit_saves)
    pa._resume(ec, _params(ec, **GARAGE), logger, stepper, None)
    assert log.get("b")["plan_stamp"] != local  # never the local plan's stamp
    assert followed == [] and stepper.view()[3]  # what it read: a plan of its own


def test_a_remote_token_may_be_named_like_an_api_token():
    """A remote's token_secret called "garage_api_token" is that remote's
    token, not one of EMHASS's: it is sent. A token with the value of one of
    EMHASS's own secrets still is not."""
    site = [
        {"id": "grid"},
        {
            "id": "garage",
            "parent": "grid",
            "solver": {"url": "https://garage.lan/participant", "token_secret": "garage_api_token"},
        },
    ]

    async def build(secrets):
        config = orjson.loads((root / "src/emhass/data/config_defaults.json").read_bytes())
        config["site"] = site
        return await build_params(EMHASS_CONF, secrets, config, logger)

    out = asyncio.run(build({"long_lived_token": "HA", "garage_api_token": "G"}))
    assert out["retrieve_hass_conf"]["remote_tokens"] == {"garage_api_token": "G"}
    out = asyncio.run(build({"long_lived_token": "HA", "garage_api_token": "HA"}))
    assert "remote_tokens" not in out["retrieve_hass_conf"]


def test_the_web_servers_api_token_is_never_a_remotes_token_secret():
    """emhass_api_token is the web server's own token (and never among the
    secrets a site resolves): a site may not name it."""
    from emhass.coordinator.site import site_token_secret_usable

    assert not site_token_secret_usable("emhass_api_token")
    site = [
        {"id": "grid"},
        {
            "id": "garage",
            "parent": "grid",
            "solver": {"url": "https://garage.lan/participant", "token_secret": "emhass_api_token"},
        },
    ]
    assert "not one of EMHASS's secrets" in utils.site_problem(site)


def test_a_remote_token_with_the_api_tokens_value_is_never_sent(monkeypatch):
    """The web server's API token is kept out of the secrets (api_auth holds
    it): a remote token holding its value is still never sent."""
    from emhass import api_auth

    monkeypatch.setattr(api_auth, "current_token", lambda: "the-api-token-0123456789")
    site = [
        {"id": "grid"},
        {
            "id": "garage",
            "parent": "grid",
            "solver": {"url": "https://garage.lan/participant", "token_secret": "garage_api_token"},
        },
    ]

    async def build(secrets):
        config = orjson.loads((root / "src/emhass/data/config_defaults.json").read_bytes())
        config["site"] = site
        return await build_params(EMHASS_CONF, secrets, config, logger)

    out = asyncio.run(build({"garage_api_token": "the-api-token-0123456789"}))
    assert "remote_tokens" not in out["retrieve_hass_conf"]
    out = asyncio.run(build({"garage_api_token": "another-0123456789"}))
    assert out["retrieve_hass_conf"]["remote_tokens"] == {"garage_api_token": "another-0123456789"}


def test_a_remote_token_with_the_api_tokens_value_is_never_sent_from_the_command_line(
    tmp_path, monkeypatch
):
    """The command line does not load api_auth: the API token build_secrets
    read from the secrets file (and kept out of the secrets) is still never
    sent as a remote's token."""
    from emhass import api_auth

    monkeypatch.setattr(utils, "_own_api_tokens", set())
    monkeypatch.setattr(api_auth, "current_token", lambda: None)
    secrets_file = tmp_path / "secrets_emhass.yaml"
    secrets_file.write_text(
        "emhass_api_token: the-api-token-0123456789\n"
        "garage_api_token: the-api-token-0123456789\n"
        "shed_api_key: shed-token-0123456789\n"
    )
    site = [
        {"id": "grid"},
        {
            "id": "garage",
            "parent": "grid",
            "solver": {"url": "https://garage.lan/participant", "token_secret": "garage_api_token"},
        },
        {
            "id": "shed",
            "parent": "grid",
            "solver": {"url": "https://shed.lan/participant", "token_secret": "shed_api_key"},
        },
    ]

    async def build():
        _, secrets = await utils.build_secrets(
            EMHASS_CONF, logger, secrets_path=str(secrets_file), no_response=True
        )
        assert "emhass_api_token" not in secrets
        config = orjson.loads((root / "src/emhass/data/config_defaults.json").read_bytes())
        config["site"] = site
        return await build_params(EMHASS_CONF, secrets, config, logger)

    out = asyncio.run(build())
    assert out["retrieve_hass_conf"]["remote_tokens"] == {"shed_api_key": "shed-token-0123456789"}
