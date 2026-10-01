"""The runtime flag dry_run: an optimisation action solves as usual, returns the
plan, and writes nothing (no results CSV, plan store, last-run record or
publish), so a coordinator outside EMHASS can probe it at trial prices."""

import asyncio
import pathlib
import shutil
import tempfile
from unittest.mock import AsyncMock, patch

import orjson
import pandas as pd

from emhass import web_server
from emhass.command_line import dayahead_forecast_optim, is_dry_run, set_input_data_dict
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
