"""Regression tests for #1304: single-flight web optimizations and atomic files."""

import asyncio
import json
import logging
import os
import subprocess
import sys
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

    def test_separate_reader_process_never_sees_an_incomplete_csv(self):
        reader_code = (
            "import json, sys\n"
            "print('ready', flush=True)\n"
            "for line in sys.stdin:\n"
            "    with open(sys.argv[1], newline='', encoding='utf-8') as f:\n"
            "        print(json.dumps(f.read()), flush=True)\n"
        )
        with TemporaryDirectory() as directory:
            target = Path(directory) / "opt_res_latest.csv"
            before = "timestamp,P_grid\n2026-01-01T00:00:00Z,1\n2026-01-01T00:05:00Z,2\n"
            after = "timestamp,P_grid\n2026-01-01T00:00:00Z,3\n2026-01-01T00:05:00Z,4\n"
            target.write_text(before, newline="")
            staged = threading.Event()
            proceed = threading.Event()
            exceptions = []

            def write(stream):
                stream.write("timestamp,P_grid\n2026-01-01T00:00:00Z,3\n")
                stream.flush()
                staged.set()
                if not proceed.wait(timeout=10):
                    raise TimeoutError("test writer was not released")
                stream.write("2026-01-01T00:05:00Z,4\n")

            def writer():
                try:
                    atomic_write(target, write)
                except Exception as exc:
                    exceptions.append(exc)

            reader = subprocess.Popen(
                [sys.executable, "-c", reader_code, str(target)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            # Bound every blocking pipe read: killing the reader makes readline() return "".
            watchdog = threading.Timer(30, reader.kill)
            watchdog.start()
            worker = threading.Thread(target=writer, daemon=True)

            def read_in_reader_process():
                reader.stdin.write("read\n")
                reader.stdin.flush()
                return json.loads(reader.stdout.readline())

            try:
                self.assertEqual(reader.stdout.readline().strip(), "ready")
                worker.start()
                self.assertTrue(staged.wait(timeout=10))
                # A staged, incomplete replacement exists next to the target ...
                staging = list(Path(directory).glob(".opt_res_latest.csv.*.tmp"))
                self.assertEqual(len(staging), 1)
                self.assertNotEqual(staging[0].read_text(), after)
                # ... but the other process still reads the complete previous CSV.
                self.assertEqual(read_in_reader_process(), before)
                proceed.set()
                worker.join(timeout=10)
                self.assertFalse(worker.is_alive())
                self.assertEqual(exceptions, [])
                self.assertEqual(read_in_reader_process(), after)
                self.assertFalse(list(Path(directory).glob("*.tmp")))
            finally:
                proceed.set()
                if worker.is_alive():
                    worker.join(timeout=10)
                watchdog.cancel()
                reader.stdin.close()
                try:
                    reader.wait(timeout=10)
                finally:
                    if reader.poll() is None:
                        reader.kill()
                        reader.wait(timeout=5)
                    reader.stdout.close()

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

    @unittest.skipIf(sys.platform == "win32", "POSIX permission bits")
    def test_replaced_file_keeps_default_permissions_not_owner_only(self):
        # tempfile creates 0o600 files; other containers/users read the data folder.
        with TemporaryDirectory() as directory:
            target = Path(directory) / "opt_res_latest.csv"
            old_umask = os.umask(0o022)
            try:
                atomic_write(target, lambda stream: stream.write("x"))
            finally:
                os.umask(old_umask)
            self.assertEqual(target.stat().st_mode & 0o777, 0o644)

    @unittest.skipIf(sys.platform == "win32", "POSIX permission bits")
    def test_replacing_existing_file_preserves_its_permissions(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "last_run.json"
            target.write_text("old")
            target.chmod(0o664)
            atomic_write(target, lambda stream: stream.write("new"))
            self.assertEqual(target.read_text(), "new")
            self.assertEqual(target.stat().st_mode & 0o777, 0o664)

    def test_concurrent_writers_use_distinct_staging_names(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "opt_res_latest.csv"
            names = []

            def write(stream):
                names.extend(p.name for p in Path(directory).glob(".*.tmp"))
                stream.write("x")

            for _ in range(3):
                atomic_write(target, write)
            self.assertEqual(len(names), 3)
            self.assertEqual(len(set(names)), 3)

    @unittest.skipUnless(sys.platform == "win32", "Windows refuses to replace an open file")
    def test_replace_waits_for_a_reader_that_briefly_holds_the_file_open(self):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "opt_res_latest.csv"
            target.write_text("old")
            reader = open(target)
            threading.Timer(0.2, reader.close).start()
            atomic_write(target, lambda stream: stream.write("new"))
            self.assertEqual(target.read_text(), "new")

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
                self.assertEqual(json.loads((data / "last_run.json").read_text())["timestamp"], ts)
                self.assertEqual(
                    json.loads((data / "plan_latest.json").read_text())["generated_at"], ts
                )
            finally:
                last_run._cache = None
                plan_store._cache = None
                last_run._unpersisted = plan_store._unpersisted = False


class TestAtomicSymlinkDestination(unittest.TestCase):
    def test_a_symlinked_destination_keeps_pointing_at_the_rewritten_target(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "real.csv"
            link = root / "opt_res_latest.csv"
            target.write_text("old")
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("this account cannot create symlinks")
            atomic_write(link, lambda stream: stream.write("new"))
            self.assertTrue(link.is_symlink())
            self.assertEqual(target.read_text(), "new")
            self.assertEqual(link.read_text(), "new")
            self.assertEqual(
                sorted(p.name for p in root.iterdir()), ["opt_res_latest.csv", "real.csv"]
            )


class TestSnapshotOrdering(unittest.TestCase):
    """last_run / plan_store: the shared file wins unless a failed write is pending."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.data = Path(self.tmp.name)
        self._reset()
        self.addCleanup(self._reset)
        self.addCleanup(self.tmp.cleanup)

    @staticmethod
    def _reset():
        for module in (last_run, plan_store):
            module._cache = None
            module._unpersisted = False

    def _write(self, name, payload):
        (self.data / name).write_text(json.dumps(payload))

    def _record_run(self, action):
        last_run.record(self.data, action, {}, "Optimal", False, 1.0, "1")

    @staticmethod
    def _fail_writes():
        return (
            patch("emhass.last_run.atomic_write", side_effect=OSError("disk full")),
            patch("emhass.plan_store.atomic_write", side_effect=OSError("disk full")),
        )

    def test_failed_write_over_an_older_file_keeps_the_newer_snapshot(self):
        self._record_run(last_run.ACTION_DAYAHEAD_OPTIM)
        plan_store.record(self.data, [{"P_grid": 1}], "2026-01-01T00:00:00Z", "1")
        first, second = self._fail_writes()
        with first, second:
            self._record_run(last_run.ACTION_NAIVE_MPC_OPTIM)
            plan_store.record(self.data, [{"P_grid": 2}], "9999-01-01T00:00:00Z", "1")
        self.assertEqual(last_run.read(self.data)["action"], last_run.ACTION_NAIVE_MPC_OPTIM)
        self.assertEqual(plan_store.read(self.data)["plan"][0]["P_grid"], 2)
        # still the answer on the next read: the cache was not replaced by the old file
        self.assertEqual(last_run.read(self.data)["action"], last_run.ACTION_NAIVE_MPC_OPTIM)

    def test_same_second_file_from_another_worker_is_not_hidden_by_a_persisted_cache(self):
        self._record_run(last_run.ACTION_DAYAHEAD_OPTIM)
        cached = last_run.read(self.data)
        # another worker replaces the file within the same second
        self._write("last_run.json", {**cached, "action": "naive-mpc-optim"})
        self.assertEqual(last_run.read(self.data)["action"], "naive-mpc-optim")
        stamp = "2026-11-06T10:00:00Z"
        plan_store.record(self.data, [{"P_grid": 1}], stamp, "1")
        self._write("plan_latest.json", {"generated_at": stamp, "plan": [{"P_grid": 9}]})
        self.assertEqual(plan_store.read(self.data)["plan"][0]["P_grid"], 9)

    def test_equal_timestamp_with_a_pending_write_prefers_the_valid_file(self):
        stamp = "2026-11-06T10:00:00Z"
        last_run._cache = {"timestamp": stamp, "action": "cache"}
        last_run._unpersisted = True
        self._write("last_run.json", {"timestamp": stamp, "action": "disk"})
        self.assertEqual(last_run.read(self.data)["action"], "disk")
        self.assertFalse(last_run._unpersisted)

    def test_pending_cache_older_than_the_file_loses(self):
        last_run._cache = {"timestamp": "2026-11-06T09:00:00Z", "action": "cache"}
        last_run._unpersisted = True
        self._write("last_run.json", {"timestamp": "2026-11-06T10:00:00Z", "action": "disk"})
        self.assertEqual(last_run.read(self.data)["action"], "disk")

    def test_a_later_successful_write_clears_the_pending_state(self):
        first, second = self._fail_writes()
        with first, second:
            self._record_run(last_run.ACTION_NAIVE_MPC_OPTIM)
            plan_store.record(self.data, [{"P_grid": 1}], "2026-11-06T10:00:00Z", "1")
        self.assertTrue(last_run._unpersisted and plan_store._unpersisted)
        self._record_run(last_run.ACTION_DAYAHEAD_OPTIM)
        plan_store.record(self.data, [{"P_grid": 2}], "2026-11-06T10:00:01Z", "1")
        self.assertFalse(last_run._unpersisted or plan_store._unpersisted)

    def test_non_object_json_is_corrupt_for_pending_and_non_pending_reads(self):
        for root in ([], [1], "x", 7, None):
            for module, name in ((last_run, "last_run.json"), (plan_store, "plan_latest.json")):
                key = "timestamp" if module is last_run else "generated_at"
                cache = {key: "2026-11-06T10:00:00Z", "marker": "cache"}
                with self.subTest(module=module.__name__, root=root, pending=True):
                    self._reset()
                    module._cache, module._unpersisted = dict(cache), True
                    self._write(name, root)
                    self.assertEqual(module.read(self.data), cache)
                    self.assertTrue(module._unpersisted)
                with self.subTest(module=module.__name__, root=root, pending=False):
                    self._reset()
                    module._cache = dict(cache)
                    self._write(name, root)
                    self.assertIsNone(module.read(self.data))

    def test_mixed_naive_and_aware_timestamps_are_compared_as_utc(self):
        last_run._unpersisted = True
        last_run._cache = {"timestamp": "2026-11-06T10:00:01+00:00", "action": "cache"}
        self._write("last_run.json", {"timestamp": "2026-11-06T10:00:00", "action": "disk"})
        self.assertEqual(last_run.read(self.data)["action"], "cache")  # naive means UTC
        last_run._cache = {"timestamp": "2026-11-06T11:00:00+01:00", "action": "cache"}
        last_run._unpersisted = True
        self._write("last_run.json", {"timestamp": "2026-11-06T10:00:00", "action": "disk"})
        self.assertEqual(last_run.read(self.data)["action"], "disk")  # equal instants

    def test_invalid_timestamps_never_hide_a_valid_file_and_never_raise(self):
        for bad in ("stale", "", None, 7, "2026-13-45T00:00:00Z"):
            last_run._persisted = None
            last_run._cache = {"timestamp": bad, "action": "cache"}
            last_run._unpersisted = True
            self._write("last_run.json", {"timestamp": "2026-11-06T10:00:00Z", "action": "disk"})
            self.assertEqual(last_run.read(self.data)["action"], "disk", bad)
        # a pending valid snapshot beats a file whose timestamp is junk
        last_run._persisted = None
        last_run._cache = {"timestamp": "2026-11-06T10:00:00Z", "action": "cache"}
        last_run._unpersisted = True
        self._write("last_run.json", {"timestamp": "junk", "action": "disk"})
        self.assertEqual(last_run.read(self.data)["action"], "cache")

    def test_unreadable_file_serves_a_pending_snapshot_but_not_a_persisted_one(self):
        (self.data / "last_run.json").write_text("{not json")
        last_run._cache = {"timestamp": "2026-11-06T10:00:00Z", "action": "cache"}
        last_run._unpersisted = True
        self.assertEqual(last_run.read(self.data)["action"], "cache")
        last_run._unpersisted = False
        self.assertIsNone(last_run.read(self.data))

    def test_missing_file_still_falls_back_to_the_in_memory_snapshot(self):
        last_run._cache = {"timestamp": "2026-11-06T10:00:00Z", "action": "cache"}
        plan_store._cache = {"generated_at": "2026-11-06T10:00:00Z", "plan": []}
        self.assertEqual(last_run.read(self.data)["action"], "cache")
        self.assertEqual(plan_store.read(self.data)["plan"], [])
        self._reset()
        self.assertIsNone(last_run.read(self.data))
        self.assertIsNone(plan_store.read(self.data))


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
            # Bounded: if the guard were missing, a second request would reach here.
            await asyncio.wait_for(release.wait(), timeout=5)
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
                inputs,
                last_run.ACTION_NAIVE_MPC_OPTIM,
                frame,
                time.monotonic(),
                logging.getLogger(__name__),
            )
            return "OK", 200

        with (
            patch.object(
                web_server, "_load_params_and_runtime", AsyncMock(return_value=({}, "profit", "{}"))
            ),
            patch.object(web_server, "set_input_data_dict", side_effect=build),
            patch.object(web_server, "_handle_action_dispatch", side_effect=dispatch),
            patch.object(web_server, "check_file_log", AsyncMock(return_value=False)),
        ):
            first_task = asyncio.create_task(self.client.post("/action/naive-mpc-optim", json={}))
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                blocked = await asyncio.wait_for(
                    self.client.post("/action/dayahead-optim", json={}), timeout=5
                )
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

    async def test_unusable_lock_file_fails_closed_with_503_and_dispatches_nothing(self):
        with (
            patch.object(
                web_server, "_live_optim_slot", side_effect=lambda: Path("/nonexistent/x/y.lock")
            ),
            patch.object(web_server, "_action_call_unlocked", new_callable=AsyncMock) as unlocked,
            patch.object(web_server, "_load_params_and_runtime", new_callable=AsyncMock) as setup,
            patch.object(web_server, "_handle_action_dispatch", new_callable=AsyncMock) as run,
            self.assertLogs(web_server.app.logger, level="WARNING") as logs,
        ):
            for action in ("naive-mpc-optim", "dayahead-optim", "perfect-optim"):
                response = await self.client.post(f"/action/{action}", json={})
                self.assertEqual(response.status_code, 503, action)
                self.assertEqual(response.headers["Retry-After"], "2")
                self.assertIn("not run", await response.get_data(as_text=True))
        unlocked.assert_not_awaited()
        setup.assert_not_awaited()
        run.assert_not_awaited()
        self.assertTrue(any("lock unavailable" in line for line in logs.output))
        self.assertFalse((self.data_path / "opt_res_latest.csv").exists())

    async def test_unusable_lock_file_does_not_block_unguarded_actions(self):
        with (
            patch.object(
                web_server, "_live_optim_slot", side_effect=lambda: Path("/nonexistent/x/y.lock")
            ),
            patch.object(
                web_server, "_action_call_unlocked", AsyncMock(return_value="ok")
            ) as unlocked,
        ):
            response = await self.client.post("/action/publish-data", json={})
        self.assertEqual(response.status_code, 200)
        unlocked.assert_awaited_once()

    async def test_slot_is_released_after_a_failed_request(self):
        slot_path = web_server._live_optim_slot()
        with patch.object(
            web_server, "_load_params_and_runtime", AsyncMock(side_effect=RuntimeError("boom"))
        ):
            first = await self.client.post("/action/naive-mpc-optim", json={})
            self.assertEqual(first.status_code, 500)
        # The failed request released its lock: another holder can take it now ...
        with open(slot_path, "a+") as probe:
            self.assertTrue(utils.try_lock_file(probe))
            utils.unlock_file(probe)
        # ... and the next request genuinely reaches the loader instead of being rejected.
        loader = AsyncMock(side_effect=RuntimeError("reached"))
        with patch.object(web_server, "_load_params_and_runtime", loader):
            again = await self.client.post("/action/naive-mpc-optim", json={})
        self.assertEqual(again.status_code, 500)
        loader.assert_awaited_once()
        # While another holder owns the slot, the same request is rejected before the loader.
        loader.reset_mock()
        with (
            open(slot_path, "a+") as other_worker,
            patch.object(web_server, "_load_params_and_runtime", loader),
        ):
            self.assertTrue(utils.try_lock_file(other_worker))
            try:
                blocked = await self.client.post("/action/naive-mpc-optim", json={})
            finally:
                utils.unlock_file(other_worker)
        self.assertEqual(blocked.status_code, 429)
        loader.assert_not_awaited()

    async def test_unrelated_actions_are_not_blocked_by_a_running_optimization(self):
        with (
            patch.object(
                web_server, "_action_call_unlocked", AsyncMock(return_value="ok")
            ) as unlocked,
            open(web_server._live_optim_slot(), "a+") as other_worker,
        ):
            self.assertTrue(utils.try_lock_file(other_worker))
            try:
                for action in ("publish-data", "forecast-model-fit", "weather-forecast-cache"):
                    response = await self.client.post(f"/action/{action}", json={})
                    self.assertEqual(response.status_code, 200, action)
                self.assertEqual(unlocked.await_count, 3)
            finally:
                utils.unlock_file(other_worker)

    async def test_lock_held_by_another_process_rejects_request(self):
        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "from emhass.utils import try_lock_file\n"
                "f = open(sys.argv[1], 'a+')\n"
                "assert try_lock_file(f)\n"
                "print('locked', flush=True)\n"
                "sys.stdin.readline()",
                str(web_server._live_optim_slot()),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": str(Path(web_server.__file__).parents[1])},
        )
        try:
            self.assertEqual(holder.stdout.readline().strip(), "locked")
            with patch.object(
                web_server, "_load_params_and_runtime", new_callable=AsyncMock
            ) as setup:
                response = await self.client.post("/action/perfect-optim", json={})
                self.assertEqual(response.status_code, 429)
                setup.assert_not_awaited()
            holder.stdin.write("go\n")
            holder.stdin.flush()
            holder.wait(timeout=10)
            with patch.object(
                web_server, "_load_params_and_runtime", AsyncMock(return_value=(None, "", ""))
            ):
                response = await self.client.post("/action/perfect-optim", json={})
            self.assertNotEqual(response.status_code, 429)  # released when the holder exited
        finally:
            if holder.poll() is None:
                holder.kill()
            holder.stdin.close()
            holder.stdout.close()
