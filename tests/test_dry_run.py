"""The runtime flag dry_run: an optimisation action solves as usual, returns the
plan, and writes nothing (no results CSV, plan store, last-run record or
publish), so a coordinator outside EMHASS can probe it at trial prices. Nor
does it touch the in-memory OptimizationCache: the next live solve sees the
same cached problem, configuration and warm-start point as if the dry run
had never happened."""

import asyncio
import pathlib
import shutil
import tempfile
from unittest.mock import AsyncMock, patch

import orjson
import pandas as pd

from emhass import web_server
from emhass.command_line import (
    OptimizationCache,
    dayahead_forecast_optim,
    is_dry_run,
    set_input_data_dict,
)
from emhass.utils import build_config, build_params, build_secrets, get_logger, get_root

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
    values = tuple(
        None if v.value is None else v.value.copy() for v in opt.prob.variables()
    )
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
