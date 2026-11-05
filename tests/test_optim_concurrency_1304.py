"""Regression tests for #1304: single-flight web optimizations and atomic files."""

import asyncio
import json
import logging
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import pandas as pd

from emhass import command_line, last_run, plan_store, utils, web_server
from emhass.atomic_file import atomic_write


class TestAtomicResults(unittest.TestCase):
    def test_reader_sees_old_complete_csv_until_new_complete_csv_is_replaced(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "opt_res_latest.csv"
            before = "timestamp,P_grid\n2026-01-01T00:00:00Z,1\n2026-01-01T00:05:00Z,2\n"
            after = "timestamp,P_grid\n2026-01-01T00:00:00Z,3\n2026-01-01T00:05:00Z,4\n"
            target.write_text(before)
            staged = threading.Event()
            proceed = threading.Event()
            exceptions = []

            def write(stream):
                stream.write("timestamp,P_grid\n2026-01-01T00:00:00Z,3\n")
                stream.flush()
                staged.set()
                if not proceed.wait(timeout=5):
                    raise TimeoutError("test writer was not released")
                stream.write("2026-01-01T00:05:00Z,4\n")

            def writer():
                try:
                    atomic_write(target, write)
                except Exception as exc:
                    exceptions.append(exc)

            worker = threading.Thread(target=writer, daemon=True)
            worker.start()
            try:
                self.assertTrue(staged.wait(timeout=5))
                # The temp file is still incomplete; the destination is not.
                for _ in range(25):
                    self.assertEqual(target.read_text(), before)
            finally:
                proceed.set()
                worker.join(timeout=5)

            self.assertFalse(worker.is_alive())
            self.assertEqual(exceptions, [])
            self.assertEqual(target.read_text(), after)
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_failed_staging_leaves_previous_csv_and_cleans_up(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "opt_res_latest.csv"
            target.write_text("complete")

            def broken(stream):
                stream.write("partial")
                raise ValueError("write interrupted")

            with self.assertRaisesRegex(ValueError, "interrupted"):
                atomic_write(target, broken)
            self.assertEqual(target.read_text(), "complete")
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_command_line_results_use_atomic_replace(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "opt_res_latest.csv"
            target.write_text("old")
            df = pd.DataFrame(
                {"P_grid": [100.0, 200.0]},
                index=pd.date_range("2026-11-05", periods=2, freq="5min", tz="UTC"),
            )
            command_line._write_results(
                {"emhass_conf": {"data_path": Path(directory)}, "params": {"passed_data": {}}},
                df,
                "opt_res_latest.csv",
            )
            actual = pd.read_csv(target, index_col="timestamp")
            self.assertEqual(actual["P_grid"].tolist(), [100.0, 200.0])
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_json_snapshots_are_valid_and_reads_follow_another_workers_write(self):
        with TemporaryDirectory() as directory:
            data = Path(directory)
            last_run._cache = None
            plan_store._cache = None
            try:
                ts = last_run.record(
                    data,
                    action=last_run.ACTION_NAIVE_MPC_OPTIM,
                    stage_times={},
                    optim_status="Optimal",
                    infeasible=False,
                    duration_total_seconds=1.0,
                    schema_version="1.0",
                )
                plan_store.record(
                    data, plan=[{"P_grid": 42}], generated_at=ts, schema_version="1.0"
                )
                # Simulate a second worker which has already cached a previous run.
                last_run._cache = {"timestamp": "stale"}
                plan_store._cache = {"generated_at": "stale"}
                self.assertEqual(last_run.read(data)["timestamp"], ts)
                self.assertEqual(plan_store.read(data)["generated_at"], ts)
                self.assertEqual(
                    json.loads((data / "last_run.json").read_text())["timestamp"], ts
                )
                self.assertEqual(
                    json.loads((data / "plan_latest.json").read_text())["generated_at"], ts
                )
            finally:
                last_run._cache = None
                plan_store._cache = None


class TestLiveOptimizationSlot(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = TemporaryDirectory()
        self.data_path = Path(self.tmp.name)
        self.conf_patch = patch.object(web_server, "emhass_conf", {"data_path": self.data_path})
        self.conf_patch.start()
        self.client = web_server.app.test_client()
        last_run._cache = None
        plan_store._cache = None

    async def asyncTearDown(self):
        self.conf_patch.stop()
        self.tmp.cleanup()
        last_run._cache = None
        plan_store._cache = None

    async def test_second_live_request_gets_429_and_cannot_overwrite_first_plan(self):
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []
        frame = pd.DataFrame(
            {"P_grid": [111], "optim_status": ["Optimal"]},
            index=pd.date_range("2026-11-05", periods=1, freq="5min", tz="UTC"),
        )

        async def build(*args, **kwargs):
            started.set()
            await release.wait()
            return {
                "emhass_conf": {"data_path": self.data_path},
                "params": {"passed_data": {}},
                "stage_times": {},
                "retrieve_hass_conf": {"continual_publish": False},
            }

        async def dispatch(action, inputs, *_):
            calls.append(action)
            command_line._write_results(inputs, frame, "opt_res_latest.csv")
            command_line._record_optim_snapshot(
                inputs, last_run.ACTION_NAIVE_MPC_OPTIM, frame, time.monotonic(), logging.getLogger(__name__)
            )
            return "OK", 200

        with (
            patch.object(web_server, "_load_params_and_runtime", AsyncMock(return_value=({}, "profit", "{}"))),
            patch.object(web_server, "set_input_data_dict", side_effect=build),
            patch.object(web_server, "_handle_action_dispatch", side_effect=dispatch),
            patch.object(web_server, "check_file_log", AsyncMock(return_value=False)),
        ):
            first_task = asyncio.create_task(
                self.client.post("/action/naive-mpc-optim", json={})
            )
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                blocked = await self.client.post("/action/dayahead-optim", json={})
                self.assertEqual(blocked.status_code, 429)
                self.assertEqual(blocked.headers["Retry-After"], "2")
                self.assertIn("not run", await blocked.get_data(as_text=True))
            finally:
                release.set()
                first = await asyncio.wait_for(first_task, timeout=5)

            self.assertEqual(first.status_code, 200)
            self.assertEqual(calls, ["naive-mpc-optim"])
            csv = pd.read_csv(self.data_path / "opt_res_latest.csv")
            self.assertEqual(csv["P_grid"].tolist(), [111])
            last = last_run.read(self.data_path)
            plan = plan_store.read(self.data_path)
            self.assertEqual(last["timestamp"], plan["generated_at"])
            self.assertEqual(plan["plan"][0]["P_grid"], 111)

    async def test_cross_worker_style_file_lock_rejects_request_without_running_it(self):
        slot = web_server._live_optim_slot()
        with (
            patch.object(web_server, "_load_params_and_runtime", new_callable=AsyncMock) as setup,
            open(slot, "a+") as other_worker,
        ):
            self.assertTrue(utils.try_lock_file(other_worker))
            try:
                blocked = await self.client.post("/action/naive-mpc-optim", json={})
                self.assertEqual(blocked.status_code, 429)
                setup.assert_not_awaited()
            finally:
                utils.unlock_file(other_worker)
