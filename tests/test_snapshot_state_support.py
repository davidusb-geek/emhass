"""The snapshot-isolation helper leaves no last-run / plan-store state behind."""

import tempfile
import unittest
import unittest.mock
from pathlib import Path

from emhass import last_run, plan_store

from .snapshot_state_support import isolated_snapshot_state

MODULES = (last_run, plan_store)


def _state():
    return [(m._cache, m._unpersisted, m._persisted) for m in MODULES]


class TestIsolatedSnapshotState(unittest.TestCase):
    def _dirty(self):
        for m in MODULES:
            self.enterContext(unittest.mock.patch.object(m, "_cache", {"marker": "before"}))
            self.enterContext(unittest.mock.patch.object(m, "_unpersisted", True))
            self.enterContext(unittest.mock.patch.object(m, "_persisted", {"marker": "before"}))

    def test_starts_cold_and_restores_all_three_fields(self):
        self._dirty()
        before = _state()
        with isolated_snapshot_state():
            self.assertEqual(_state(), [(None, False, None)] * 2)
        self.assertEqual(_state(), before)

    def test_a_successful_record_does_not_leave_a_persisted_snapshot(self):
        before = _state()
        with isolated_snapshot_state(), tempfile.TemporaryDirectory() as directory:
            ts = last_run.record(
                Path(directory), last_run.ACTION_NAIVE_MPC_OPTIM, {}, "Optimal", False, 1.0, "1"
            )
            plan_store.record(Path(directory), plan=[], generated_at=ts, schema_version="1")
            self.assertIsNotNone(last_run._persisted)
            self.assertIsNotNone(plan_store._persisted)
        self.assertEqual(_state(), before)

    def test_restores_state_when_the_body_raises(self):
        self._dirty()
        before = _state()
        with self.assertRaises(RuntimeError), isolated_snapshot_state():
            last_run._unpersisted = True
            raise RuntimeError("test body failed")
        self.assertEqual(_state(), before)
