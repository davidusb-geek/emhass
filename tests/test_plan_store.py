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

import numpy as np
import pandas as pd

from emhass import last_run, plan_store


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


class TestPlanStore(unittest.TestCase):
    """plan_store serialize / record / read (cache + write-through)."""

    def setUp(self):
        self.tmp_path = Path(tempfile.mkdtemp())
        plan_store._cache = None

    def tearDown(self):
        plan_store._cache = None

    def test_serialize_opt_res_records_iso_and_nan(self):
        idx = pd.to_datetime(["2026-06-17T00:00:00+00:00", "2026-06-17T00:30:00+00:00"], utc=True)
        idx.name = "timestamp"
        df = pd.DataFrame(
            {"P_Load": [100.0, np.nan], "optim_status": ["Optimal", "Optimal"]}, index=idx
        )
        records = plan_store.serialize(df)
        self.assertTrue(records[0]["timestamp"].startswith("2026-06-17T00:00:00"))
        self.assertEqual(records[0]["P_Load"], 100.0)
        self.assertIsNone(records[1]["P_Load"])  # NaN -> null

    def test_record_then_read_roundtrip(self):
        self.assertIsNone(plan_store.read(self.tmp_path))  # cold start, no run yet
        plan_store.record(
            self.tmp_path,
            plan=[{"timestamp": "2026-06-17T00:00:00Z", "P_Load": 100.0}],
            generated_at="2026-06-17T00:00:05Z",
            schema_version="1.0",
        )
        snap = plan_store.read(self.tmp_path)
        self.assertEqual(snap["generated_at"], "2026-06-17T00:00:05Z")
        self.assertEqual(snap["emhass_schema_version"], "1.0")
        self.assertEqual(snap["plan"][0]["P_Load"], 100.0)
        self.assertTrue((self.tmp_path / "plan_latest.json").exists())  # write-through


class TestRecordOptimSnapshotWritesPlan(unittest.TestCase):
    """_record_optim_snapshot writes the plan with generated_at == last_run timestamp."""

    def setUp(self):
        self.tmp_path = Path(tempfile.mkdtemp())
        last_run._cache = None
        plan_store._cache = None

    def tearDown(self):
        last_run._cache = None
        plan_store._cache = None

    def test_record_optim_snapshot_writes_plan_with_shared_timestamp(self):
        import logging

        from emhass import command_line

        idx = pd.to_datetime(["2026-06-17T00:00:00+00:00"], utc=True)
        idx.name = "timestamp"
        opt_res = pd.DataFrame({"P_Load": [100.0], "optim_status": ["Optimal"]}, index=idx)
        input_data_dict = {
            "emhass_conf": {"data_path": self.tmp_path},
            "stage_times": {},
        }
        command_line._record_optim_snapshot(
            input_data_dict,
            last_run.ACTION_DAYAHEAD_OPTIM,
            opt_res,
            0.0,
            logging.getLogger("test_plan_store"),
        )
        plan = plan_store.read(self.tmp_path)
        lr = last_run.read(self.tmp_path)
        self.assertIsNotNone(plan)
        self.assertEqual(plan["plan"][0]["P_Load"], 100.0)
        # shared timestamp: plan generated_at equals the last_run timestamp
        self.assertEqual(plan["generated_at"], lr["timestamp"])

    def test_record_optim_snapshot_skips_plan_on_non_optimal(self):
        """A failed/infeasible run is recorded by last_run but must NOT publish a
        plan: /api/v1/plan would otherwise report status='ok' for the same run
        that /api/v1/last-run reports as 'infeasible'. The plan is published iff
        the run is Optimal (i.e. iff last_run's status is 'ok')."""
        import logging

        from emhass import command_line

        idx = pd.to_datetime(["2026-06-17T00:00:00+00:00"], utc=True)
        idx.name = "timestamp"
        opt_res = pd.DataFrame({"P_Load": [100.0], "optim_status": ["Infeasible"]}, index=idx)
        input_data_dict = {
            "emhass_conf": {"data_path": self.tmp_path},
            "stage_times": {},
        }
        command_line._record_optim_snapshot(
            input_data_dict,
            last_run.ACTION_DAYAHEAD_OPTIM,
            opt_res,
            0.0,
            logging.getLogger("test_plan_store"),
        )
        # no plan published for a non-Optimal run
        self.assertIsNone(plan_store.read(self.tmp_path))
        # last_run still records the failed run, as 'infeasible'
        self.assertEqual(last_run.read(self.tmp_path)["status"], "infeasible")
