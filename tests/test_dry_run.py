"""The runtime flag dry_run: an optimisation action solves as usual, returns the
plan, and writes nothing (no results CSV, plan store, last-run record or
publish), so a coordinator outside EMHASS can probe it at trial prices. Nor
does it touch the in-memory OptimizationCache: the next live solve sees the
same cached problem, configuration and warm-start point as if the dry run
had never happened."""

import asyncio
import json
import pathlib
import shutil
import sys
import tempfile
import types
from unittest.mock import AsyncMock, patch

import orjson
import pandas as pd
import pytest

from emhass import last_run, web_server
from emhass.command_line import (
    OptimizationCache,
    dayahead_forecast_optim,
    is_dry_run,
    naive_mpc_optim,
    perfect_forecast_optim,
    set_input_data_dict,
)
from emhass.optimization import (
    _BACKENDS,
    _DEVICE,
    _LIMIT_NAME,
    _SOLVERS,
    _coordinated_config_problem,
)
from emhass.utils import (
    RuntimeParamError,
    build_config,
    build_params,
    build_secrets,
    get_logger,
    get_root,
    parse_dry_run,
)

root = pathlib.Path(get_root(__file__, num_parent=2))
logger, _ = get_logger(__name__, {"data_path": root / "data/"}, save_to_file=False)


def _conf(data_path: pathlib.Path) -> dict:
    """EMHASS paths with `data_path` as the data folder.

    Args:
        data_path: The data folder to read inputs from and write results to.

    Returns:
        dict: The emhass_conf paths.
    """
    return {
        "data_path": data_path,
        "root_path": root / "src/emhass/",
        "defaults_path": root / "src/emhass/data/config_defaults.json",
        "associations_path": root / "src/emhass/data/associations.csv",
    }


def test_is_dry_run_reads_the_runtime_flag():
    """The flag is read from params as a dict or as JSON, and is off by default."""
    assert is_dry_run({"params": {"passed_data": {"dry_run": True}}})
    assert is_dry_run({"params": orjson.dumps({"passed_data": {"dry_run": True}}).decode()})
    assert not is_dry_run({"params": {"passed_data": {}}})
    assert not is_dry_run({"params": {}})


def test_a_dry_run_writes_nothing():
    """A day-ahead optimisation with dry_run returns a plan and leaves the data
    folder exactly as it was."""

    async def run(data_path):
        """Run the day-ahead action as a dry run on the data in `data_path`;
        returns the plan (opt_res DataFrame)."""
        ec = _conf(data_path)
        _, secrets = await build_secrets(ec, logger, no_response=True)
        params = await build_params(
            ec, secrets, await build_config(ec, logger, ec["defaults_path"]), logger
        )
        runtime = {"dry_run": True}
        idd = await set_input_data_dict(
            ec,
            "profit",
            orjson.dumps(params).decode(),
            orjson.dumps(runtime).decode(),
            "dayahead-optim",
            logger,
            get_data_from_file=True,
        )
        return await dayahead_forecast_optim(idd, logger)

    with tempfile.TemporaryDirectory() as tmp:
        data_path = pathlib.Path(tmp) / "data"
        shutil.copytree(root / "data", data_path)
        before = {p: p.stat().st_mtime for p in data_path.rglob("*")}
        res = asyncio.run(run(data_path))
        after = {p: p.stat().st_mtime for p in data_path.rglob("*")}
    assert isinstance(res, pd.DataFrame) and len(res) > 0
    assert after == before


def test_the_web_action_returns_the_plan_on_a_dry_run():
    """On a dry run the action answers with the plan records (as /api/v1/plan
    serves them) and does not save the page's injection file."""
    index = pd.date_range("2026-10-01", periods=2, freq="30min", tz="UTC")
    plan = pd.DataFrame({"P_grid": [100.0, -50.0]}, index=index)
    idd = {"params": {"passed_data": {"dry_run": True}}}
    with (
        patch.object(web_server, "dayahead_forecast_optim", AsyncMock(return_value=plan)),
        patch.object(web_server, "_save_injection_dict", AsyncMock()) as save,
    ):
        body, status = asyncio.run(
            web_server._handle_action_dispatch(
                "dayahead-optim", idd, _conf(root / "data/"), {}, {}, logger
            )
        )
    assert status == 200
    assert body["dry_run"] is True
    assert [r["P_grid"] for r in body["plan"]] == [100.0, -50.0]
    save.assert_not_called()


def _optimise(data_path: pathlib.Path, costfun: str, runtime: dict) -> pd.DataFrame:
    """Run the day-ahead action on the data in `data_path` with `runtime`
    parameters; returns the plan (opt_res DataFrame)."""

    async def run():
        """Build the inputs and solve."""
        ec = _conf(data_path)
        _, secrets = await build_secrets(ec, logger, no_response=True)
        params = await build_params(
            ec, secrets, await build_config(ec, logger, ec["defaults_path"]), logger
        )
        idd = await set_input_data_dict(
            ec,
            costfun,
            orjson.dumps(params).decode(),
            orjson.dumps(runtime).decode(),
            "dayahead-optim",
            logger,
            get_data_from_file=True,
        )
        return await dayahead_forecast_optim(idd, logger)

    return asyncio.run(run())


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
    """A dry run at other prices, and one with another configuration, leave
    the cached problem as the live solve left it: same object, same key, same
    configuration, same solution to warm-start from. Without this, a dry run
    reconfigures and re-solves the live object (a cache hit) or replaces it
    (a miss), and the next live solve inherits the dry run."""
    OptimizationCache.clear()
    with tempfile.TemporaryDirectory() as tmp:
        data_path = pathlib.Path(tmp) / "data"
        shutil.copytree(root / "data", data_path)
        live = _optimise(data_path, "profit", {})
        assert isinstance(live, pd.DataFrame) and OptimizationCache._instance is not None
        before = _cache_state()

        # the same configuration, at very different prices: would be a cache hit
        prices = [0.5 if i % 2 else 0.01 for i in range(len(live))]
        trial = _optimise(data_path, "profit", {"dry_run": True, "load_cost_forecast": prices})
        assert isinstance(trial, pd.DataFrame) and len(trial) == len(live)
        assert _same(before, _cache_state())

        # another cost function: would be a cache miss, and evict the live problem
        other = _optimise(data_path, "cost", {"dry_run": True})
        assert isinstance(other, pd.DataFrame)
        assert _same(before, _cache_state())
    OptimizationCache.clear()


def _endpoints(data_path: pathlib.Path) -> tuple:
    """What GET /api/v1/plan and GET /api/v1/last-run serve for `data_path`."""

    async def get():
        """Ask the web app, as a client would."""
        saved = web_server.emhass_conf
        web_server.emhass_conf = {**saved, "data_path": data_path}
        try:
            client = web_server.app.test_client()
            plan = await client.get("/api/v1/plan")
            run = await client.get("/api/v1/last-run")
            return plan.status_code, await plan.get_json(), run.status_code, await run.get_json()
        finally:
            web_server.emhass_conf = saved

    return asyncio.run(get())


def test_a_dry_run_leaves_the_plan_and_last_run_endpoints_unchanged():
    """After a live run, /api/v1/plan and /api/v1/last-run serve that run.
    A dry run after it, at other prices, changes neither response."""
    OptimizationCache.clear()
    with tempfile.TemporaryDirectory() as tmp:
        data_path = pathlib.Path(tmp) / "data"
        shutil.copytree(root / "data", data_path)
        live = _optimise(data_path, "profit", {})
        before = _endpoints(data_path)
        plan_status, plan, run_status, run = before
        assert plan_status == 200 and plan["status"] == "ok" and len(plan["plan"]) == len(live)
        assert run_status == 200 and run["action"] == "dayahead-optim"

        prices = [0.5 if i % 2 else 0.01 for i in range(len(live))]
        trial = _optimise(data_path, "profit", {"dry_run": True, "load_cost_forecast": prices})
        assert isinstance(trial, pd.DataFrame) and not trial["P_grid"].equals(live["P_grid"])
        last_run._cache = None  # read from disk too, not only the in-memory copy
        assert _endpoints(data_path) == before
    OptimizationCache.clear()


def test_a_dry_run_plans_exactly_as_a_live_run():
    """The same inputs, each from a cold cache, give the same plan live and
    dry: a dry run changes what is written, never what is solved."""
    with tempfile.TemporaryDirectory() as tmp:
        data_path = pathlib.Path(tmp) / "data"
        shutil.copytree(root / "data", data_path)
        OptimizationCache.clear()
        live = _optimise(data_path, "profit", {})
        OptimizationCache.clear()
        dry = _optimise(data_path, "profit", {"dry_run": True})
    OptimizationCache.clear()
    assert isinstance(live, pd.DataFrame) and len(live) > 0
    pd.testing.assert_frame_equal(dry, live)


# --------------------------------------------------------------------------
# No results CSV, whichever action and however it saves
# --------------------------------------------------------------------------

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


def _run_action(data_path: pathlib.Path, action: str, runtime: dict, save: bool):
    """Run `action` on the data in `data_path`, not in debug mode (debug also
    skips the CSV, which would make the check below vacuous). Returns the
    wrapper's result."""

    async def run():
        """Build the inputs and run the action's wrapper."""
        ec = _conf(data_path)
        _, secrets = await build_secrets(ec, logger, no_response=True)
        params = await build_params(
            ec, secrets, await build_config(ec, logger, ec["defaults_path"]), logger
        )
        params["optim_conf"]["set_use_pv"] = True  # as tests/test_command_line_utils.py
        rt = {**FORECASTS, **runtime}
        if action == "naive-mpc-optim":
            rt["prediction_horizon"] = 48
        idd = await set_input_data_dict(
            ec,
            "profit",
            orjson.dumps(params).decode(),
            orjson.dumps(rt).decode(),
            action,
            logger,
            get_data_from_file=True,
        )
        assert idd, f"{action}: inputs could not be built"
        return await ACTIONS[action](idd, logger, save_data_to_file=save, debug=False)

    return asyncio.run(run())


def _files(data_path: pathlib.Path) -> dict:
    """Every file under `data_path`, with its size and modification time."""
    return {
        p.relative_to(data_path): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in data_path.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("save", [False, True], ids=["latest", "dated"])
@pytest.mark.parametrize("action", list(ACTIONS))
def test_a_live_run_writes_its_results_csv(action, save):
    """The control for the test below: the same action, live, does write a
    results CSV, so 'no CSV' there is the dry run's doing, not the setup's."""
    OptimizationCache.clear()
    with tempfile.TemporaryDirectory() as tmp:
        data_path = pathlib.Path(tmp) / "data"
        shutil.copytree(root / "data", data_path)
        before = _files(data_path)
        res = _run_action(data_path, action, {}, save)
        after = _files(data_path)
    assert isinstance(res, pd.DataFrame)
    written = [p for p in after if p.suffix == ".csv" and after[p] != before.get(p)]
    assert written, f"{action} (save_data_to_file={save}) wrote no CSV when live"
    OptimizationCache.clear()


@pytest.mark.parametrize("save", [False, True], ids=["latest", "dated"])
@pytest.mark.parametrize("action", list(ACTIONS))
def test_a_dry_run_writes_no_csv_and_no_file(action, save):
    """Every optimisation action, saving the latest results or a dated file,
    returns its plan on a dry run and leaves the data folder byte for byte as
    it was: no results CSV is created or modified, and no other file either."""
    OptimizationCache.clear()
    with tempfile.TemporaryDirectory() as tmp:
        data_path = pathlib.Path(tmp) / "data"
        shutil.copytree(root / "data", data_path)
        before = _files(data_path)
        csv_before = {p for p in before if p.suffix == ".csv"}
        res = _run_action(data_path, action, {"dry_run": True}, save)
        after = _files(data_path)
    assert isinstance(res, pd.DataFrame) and len(res) > 0
    assert {p for p in after if p.suffix == ".csv"} == csv_before, "a dry run created a CSV"
    assert after == before, f"a dry run changed {sorted(set(after.items()) ^ set(before.items()))}"
    OptimizationCache.clear()


# --------------------------------------------------------------------------
# The flag is read strictly; a value that cannot be read is refused
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
        ("False", False),
        ("no", False),
        ("off", False),
        ("0", False),
        ("", False),
    ],
)
def test_the_flag_is_read_strictly(value, expected):
    """In particular the string 'false' is False: bool('false') is True, which
    would turn a templated live request into a dry run that publishes nothing."""
    assert parse_dry_run(value) is expected


@pytest.mark.parametrize("value", ["maybe", "2", 2, -1, 1.0, [], {}, [True]])
def test_a_flag_that_cannot_be_read_is_refused(value):
    with pytest.raises(RuntimeParamError):
        parse_dry_run(value)


def test_a_request_with_an_unreadable_flag_is_refused_and_writes_nothing():
    """set_input_data_dict answers False (the action returns 400) - neither a
    live run nor a dry run is attempted - and the data folder is untouched."""

    async def build(data_path):
        """Build the inputs for a request carrying dry_run='maybe'."""
        ec = _conf(data_path)
        _, secrets = await build_secrets(ec, logger, no_response=True)
        params = await build_params(
            ec, secrets, await build_config(ec, logger, ec["defaults_path"]), logger
        )
        return await set_input_data_dict(
            ec,
            "profit",
            orjson.dumps(params).decode(),
            orjson.dumps({**FORECASTS, "dry_run": "maybe"}).decode(),
            "dayahead-optim",
            logger,
            get_data_from_file=True,
        )

    with tempfile.TemporaryDirectory() as tmp:
        data_path = pathlib.Path(tmp) / "data"
        shutil.copytree(root / "data", data_path)
        before = _files(data_path)
        assert asyncio.run(build(data_path)) is False
        assert _files(data_path) == before


def test_only_a_real_true_is_a_dry_run():
    """is_dry_run trusts nothing but True: a stray truthy value stays live."""
    for value in ("false", "yes", 1, [1]):
        assert not is_dry_run({"params": {"passed_data": {"dry_run": value}}})
    assert is_dry_run({"params": {"passed_data": {"dry_run": True}}})


# --------------------------------------------------------------------------
# Whatever the coordinator does, a plan is made: live and dry alike
# --------------------------------------------------------------------------


def _coordinated(data_path: pathlib.Path, runtime: dict, participants=None):
    """Day-ahead with the coordinated backend switched on (and `participants`
    if given); returns the plan."""

    async def run():
        """Build the inputs with the backend on, and solve."""
        ec = _conf(data_path)
        _, secrets = await build_secrets(ec, logger, no_response=True)
        params = await build_params(
            ec, secrets, await build_config(ec, logger, ec["defaults_path"]), logger
        )
        params["optim_conf"]["optimization_backend"] = "dantzig_wolfe"
        params["optim_conf"]["participants"] = participants
        idd = await set_input_data_dict(
            ec,
            "profit",
            orjson.dumps(params).decode(),
            orjson.dumps({**FORECASTS, **runtime}).decode(),
            "dayahead-optim",
            logger,
            get_data_from_file=True,
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
        res = _coordinated(data_path, {"dry_run": dry})
    assert isinstance(res, pd.DataFrame) and len(res) > 0
    assert (res["optim_status"] == "Optimal").all()
    assert "coordinator raised RuntimeError" in caplog.text
    # ...and the plan says which solver made it, and why
    assert (res["backend_used"] == "cvxpy").all()
    assert res["backend_fallback_reason"].str.contains("coordinator exploded").all()
    OptimizationCache.clear()


@pytest.mark.parametrize(
    "participants, problem",
    [
        ("battery", "must be a list"),
        ([{"nodevices": 1}], "must be an object with devices, solver and config only"),
        ([{"devices": []}], "non-empty list of device names"),
        ([{"devices": ["battery; rm -rf /"]}], "non-empty list of device names"),
        ([{"devices": ["__import__('os')"]}], "non-empty list of device names"),
        ([{"devices": ["battery"], "solver": "os.system"}], "solver must be one of"),
        ([{"devices": ["battery"]}, {"devices": ["battery"]}], "already has"),
        (
            [
                {
                    "devices": ["water_heater"],
                    "solver": "home_energy_optimizer",
                    "config": {"power_kw": {"__class__": 1}},
                }
            ],
            "finite number, a boolean or a short string",
        ),
        (
            [
                {
                    "devices": ["hvac"],
                    "solver": "home_energy_optimizer",
                    "config": {"cop": float("nan")},
                }
            ],
            "finite number, a boolean or a short string",
        ),
        (
            [
                {
                    "devices": ["water_heater"],
                    "solver": "home_energy_optimizer",
                    "config": {"t_comfort": "x" * 65},
                }
            ],
            "finite number, a boolean or a short string",
        ),
        ([{"devices": [f"deferrable{i}"]} for i in range(65)], "at most 64 groups"),
    ],
)
def test_participants_are_checked_before_anything_reads_them(participants, problem):
    """`participants` can arrive with a request, so its shape and types are
    checked first; anything unexpected means the default solver, never an
    exception or an unknown name reaching the coordinator."""
    reason = _coordinated_config_problem(
        {"optimization_backend": "dantzig_wolfe", "participants": participants}
    )
    assert reason and problem in reason


def test_a_known_backend_and_a_valid_spec_pass():
    assert _coordinated_config_problem({"optimization_backend": "dantzig_wolfe"}) is None
    assert (
        _coordinated_config_problem(
            {
                "optimization_backend": "dantzig_wolfe",
                "participants": [
                    {"devices": ["battery"], "solver": "home_energy_optimizer"},
                    {"devices": ["deferrable0", "deferrable1"], "solver": "emhass"},
                    {
                        "devices": ["water_heater"],
                        "solver": "home_energy_optimizer",
                        "config": {"power_kw": 3.0, "n_duty_levels": 2, "comfort_mode": "linear"},
                    },
                ],
            }
        )
        is None
    )
    assert "unknown backend" in _coordinated_config_problem({"optimization_backend": "rogue"})
    # documented as future work, not offered: it says so and the default solver plans
    assert "later release" in _coordinated_config_problem({"optimization_backend": "admm"})


def test_the_published_schema_is_the_check():
    """openapi.json describes participants as the check reads it - a list of
    objects with these device names and solvers - and offers only the
    backends EMHASS runs."""
    spec = json.loads((root / "src/emhass/static/openapi.json").read_text())
    props = spec["components"]["schemas"]["Config"]["properties"]
    assert props["optimization_backend"]["enum"] == list(_BACKENDS)
    group = props["participants"]["items"]
    assert props["participants"]["type"] == ["array", "null"]
    assert set(group["properties"]) == {"devices", "solver", "config"}
    assert group["additionalProperties"] is False
    assert group["properties"]["devices"]["items"]["pattern"] == _DEVICE.pattern
    assert group["properties"]["solver"]["enum"] == list(_SOLVERS)
    limit = props["group_limits"]["items"]
    assert set(limit["properties"]) == {"name", "devices", "max_power", "min_power"}
    assert limit["additionalProperties"] is False
    assert limit["properties"]["name"]["pattern"] == _LIMIT_NAME.pattern
    assert limit["properties"]["devices"]["items"]["pattern"] == _DEVICE.pattern
    assert limit["properties"]["min_power"]["maximum"] == 0


@pytest.mark.parametrize(
    "limits, problem",
    [
        ({"name": "garage"}, "must be a list"),
        (
            [{"name": "garage", "devices": ["hvac"], "max_power": 1, "rogue": 1}],
            "name, devices, max_power and min_power only",
        ),
        ([{"name": "Garage!", "devices": ["hvac"], "max_power": 1}], "lower-case name"),
        ([{"name": "inverter", "devices": ["hvac"], "max_power": 1}], "lower-case name"),
        ([{"name": "__import__('os')", "devices": ["hvac"], "max_power": 1}], "lower-case name"),
        (
            [{"name": "g", "devices": ["hvac; rm -rf /"], "max_power": 1}],
            "non-empty list of device names",
        ),
        ([{"name": "g", "devices": []}], "non-empty list of device names"),
        ([{"name": "g", "devices": ["hvac"]}], "needs max_power, min_power or both"),
        (
            [{"name": "g", "devices": ["hvac"], "max_power": "3000"}],
            "max_power must be a finite number",
        ),
        (
            [{"name": "g", "devices": ["hvac"], "max_power": float("inf")}],
            "max_power must be a finite number",
        ),
        (
            [{"name": "g", "devices": ["hvac"], "max_power": True}],
            "max_power must be a finite number",
        ),
        (
            [{"name": "g", "devices": ["hvac"], "min_power": 500}],
            "min_power must be a finite number of W, <= 0",
        ),
        (
            [
                {"name": "a", "devices": ["hvac"], "max_power": 1},
                {"name": "b", "devices": ["hvac"], "max_power": 1},
            ],
            "another limit already has",
        ),
        (
            [
                {"name": "a", "devices": ["hvac"], "max_power": 1},
                {"name": "a", "devices": ["battery"], "max_power": 1},
            ],
            "lower-case name",
        ),
        (
            [{"name": f"g{i}", "devices": [f"deferrable{i}"], "max_power": 1} for i in range(65)],
            "at most 64 limits",
        ),
    ],
)
def test_group_limits_are_checked_before_anything_reads_them(limits, problem):
    """`group_limits` can arrive with a request too: refused by shape and type,
    by name, before the coordinator sees it."""
    reason = _coordinated_config_problem(
        {"optimization_backend": "dantzig_wolfe", "group_limits": limits}
    )
    assert reason and problem in reason


def test_valid_group_limits_pass():
    assert (
        _coordinated_config_problem(
            {
                "optimization_backend": "dantzig_wolfe",
                "group_limits": [
                    {
                        "name": "garage",
                        "devices": ["water_heater", "hvac"],
                        "max_power": 3000,
                        "min_power": 0,
                    },
                    {"name": "ev_panel", "devices": ["deferrable1"], "max_power": 7400},
                ],
            }
        )
        is None
    )


@pytest.mark.parametrize("dry", [False, True], ids=["live", "dry"])
def test_a_malformed_spec_in_a_request_still_plans(dry, caplog):
    """The request that crashed with KeyError 'devices' before: now the
    default solver plans, live or dry, and the log says why."""
    OptimizationCache.clear()
    with tempfile.TemporaryDirectory() as tmp, caplog.at_level("WARNING"):
        data_path = pathlib.Path(tmp) / "data"
        shutil.copytree(root / "data", data_path)
        res = _coordinated(data_path, {"dry_run": dry}, participants=[{"nodevices": 1}])
    assert isinstance(res, pd.DataFrame) and (res["optim_status"] == "Optimal").all()
    assert (res["backend_used"] == "cvxpy").all()
    # refused by the check, by name - the coordinator never saw it
    assert "participants[0] must be an object with devices, solver and config only" in caplog.text
    assert "coordinator raised" not in caplog.text
    OptimizationCache.clear()
