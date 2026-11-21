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
    the API's base, `shutdown()` stops it."""
    server = _Server(("127.0.0.1", 0), _Handler)
    server.service = service
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
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
    assert described["active_commit"]["commit_id"] == "q1"
    assert described["active_commit"]["state"] == state
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
    old = CommitLog(path, keep_s=0.0)  # past its keeping time
    assert old.get("done") is None and old.latest() is None


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

    def no_answer_to_commits(self, method, path, body=None, query_id=None):
        if path == "/v1/commit":
            sent.append(body)
            raise RemoteError(f"{self.key}: timed out")  # status 0: nothing came back
        if path.startswith("/v1/commit/"):
            asked.append(path)
            raise RemoteError(f"{self.key}: timed out")
        return call(self, method, path, body, query_id)

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


def test_a_failed_plan_after_a_commit_keeps_the_plan_that_runs(data_path, monkeypatch):
    """The garage takes its plan; the shed refuses; planning the house again
    without the shed fails outright. The garage runs already, so EMHASS's
    default solver must not plan the house around it as absent: the plan
    that counts the garage's part stands."""
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
        res = _coordinate(_house(_now()), site)
        assert (res["backend_used"] == "dantzig_wolfe").all(), res["backend_fallback_reason"].iloc[
            0
        ]
        assert (res["fed_remote_status_garage"] == "committed").all()
        assert res["fed_remote_status_shed"].iloc[0].startswith("refused")
        assert "fed_remote_power_garage" in res.columns
        assert len(built) == 3
    finally:
        garage.shutdown()
        shed.shutdown()


def test_the_plan_says_how_each_remotes_last_commit_ended(data_path):
    """A remote reports its last commit (describe's active_commit): the plan
    shows it, so a run starts from what the remote actually did - here, the
    commit of the run before."""
    ec = _conf(data_path)
    params = _params(ec, **GARAGE)

    def build(window):
        return asyncio.run(
            participant_api.build_session(ec, params, window, logger, get_data_from_file=True)
        )

    def describe():
        return emhass_description(build(PlanWindow(30, 48)))

    server = serve(ParticipantService(describe, build, lambda p, a: ("partial", "cut off")))
    try:
        site = [{"id": "garage", "parent": "grid", "solver": {"url": server.url}}]
        first = _coordinate(_house(_now()), site)
        assert "fed_remote_last_commit_garage" not in first.columns  # no commit before
        assert first["fed_remote_status_garage"].iloc[0].startswith("uncertain")
        second = _coordinate(_house(_now()), site)
        assert second["fed_remote_last_commit_garage"].iloc[0] == "partial: cut off"
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
    assert "lock" in path["post"]["responses"]["503"]["description"]


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
            return 200, b'{"version": 1, "kinds": ["price_response"], "max_power_kw": 1.0}'
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


@pytest.mark.parametrize(
    "power", ["", '"max_power_kw": null, ', '"max_power_kw": Infinity, ', '"max_power_kw": true, ']
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
