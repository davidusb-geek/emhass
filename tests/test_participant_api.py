"""This EMHASS's devices served to a coordinator elsewhere (participant_api),
and the coordinator's side: a remote solver named in `site`.

The remote side is served by home-energy-optimizer's ParticipantService on a
local port (remote.serve), building its model with EMHASS's own input path;
the coordinator is another EMHASS's optimize, as a live run calls it."""

import asyncio
import copy
import importlib.util
import pathlib
import pickle
import shutil
import tempfile

import numpy as np
import orjson
import pandas as pd
import pytest

from emhass import participant_api, web_server
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
    """This slot's start, in EMHASS's time zone: the remote plans from its own clock."""
    return pd.Timestamp.now(tz=_confs()[0]["time_zone"]).floor("30min")


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
