"""Tests for the plan snapshot persistence layer (AC-6, GET /api/v1/plan).

Mirrors the unittest layout of the rest of tests/ so the CI gating job
(`python -m unittest`) collects them. Covers:
  - last_run.record() returning the stamped timestamp (Task 1)
  - plan_store serialize / record / read (Task 2)
  - _record_optim_snapshot writing the plan with the shared timestamp (Task 3)
"""

import tempfile
import unittest
from pathlib import Path

from emhass import last_run


class TestLastRunRecordReturn(unittest.TestCase):
    """last_run.record() returns the ISO timestamp it stamps (single source for AC-6)."""

    def setUp(self):
        self.tmp_path = Path(tempfile.mkdtemp())
        last_run._cache = None

    def tearDown(self):
        last_run._cache = None

    def test_record_returns_stamped_timestamp(self):
        ts = last_run.record(
            self.tmp_path,
            action=last_run.ACTION_DAYAHEAD_OPTIM,
            stage_times={},
            optim_status="Optimal",
            infeasible=False,
            duration_total_seconds=1.0,
            schema_version="1.0",
        )
        # the returned value is exactly what was persisted as the snapshot timestamp
        snap = last_run.read(self.tmp_path)
        self.assertIsNotNone(ts)
        self.assertEqual(ts, snap["timestamp"])
