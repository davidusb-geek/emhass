"""Unit tests for src/emhass/last_run.py (AC-3)."""

import logging
from datetime import UTC, datetime

import pytest

from emhass import last_run


@pytest.fixture
def data_path(tmp_path):
    """Per-test scratch dir for last_run.json persistence."""
    return tmp_path


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear in-memory cache between tests to avoid cross-test bleed."""
    last_run._cache = None
    yield
    last_run._cache = None


def test_record_then_read_round_trip(data_path):
    last_run.record(
        data_path,
        action="naive-mpc-optim",
        stage_times={"pv_forecast": 1.2, "load_forecast": 0.5},
        optim_status="Optimal",
        infeasible=False,
        duration_total_seconds=3.7,
        schema_version="1.0",
    )
    snap = last_run.read(data_path)
    assert snap is not None
    assert snap["action"] == "naive-mpc-optim"
    assert snap["status"] == "ok"
    assert snap["stage_times"] == {"pv_forecast": 1.2, "load_forecast": 0.5}
    assert snap["duration_total_seconds"] == 3.7
    assert snap["infeasible"] is False
    assert snap["schema_version"] == "1.0"
    assert snap["error_message"] is None
    assert "timestamp" in snap
    assert "emhass_version" in snap


def test_read_returns_none_when_no_run(data_path):
    assert last_run.read(data_path) is None


def test_read_recovers_from_corrupt_file(data_path, caplog):
    last_run._path(data_path).write_text("not-json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="emhass.last_run"):
        result = last_run.read(data_path)
    assert result is None
    assert any("corrupt or unreadable" in rec.message for rec in caplog.records)


def test_is_recent_true_when_just_recorded(data_path):
    last_run.record(
        data_path,
        action="naive-mpc-optim",
        stage_times={},
        optim_status="Optimal",
        infeasible=False,
        duration_total_seconds=1.0,
        schema_version="1.0",
    )
    assert last_run.is_recent(data_path, max_age_seconds=60) is True


def test_is_recent_false_when_no_run(data_path):
    assert last_run.is_recent(data_path, max_age_seconds=60) is False


def test_is_recent_false_when_too_old(data_path):
    from datetime import timedelta

    old_ts = (
        (datetime.now(UTC) - timedelta(hours=2))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    last_run._cache = {
        "status": "ok",
        "timestamp": old_ts,
        "action": "naive-mpc-optim",
        "stage_times": {},
        "duration_total_seconds": 1.0,
        "emhass_version": "0.13.0",
        "schema_version": "1.0",
        "infeasible": False,
        "error_message": None,
    }
    assert last_run.is_recent(data_path, max_age_seconds=600) is False


def test_is_recent_logs_warning_on_malformed_timestamp(data_path, caplog):
    """A corrupt timestamp must surface as a warning, not silent False."""
    last_run._cache = {
        "status": "ok",
        "timestamp": "not-a-timestamp",
        "action": "naive-mpc-optim",
        "stage_times": {},
        "duration_total_seconds": 1.0,
        "emhass_version": "0.13.0",
        "schema_version": "1.0",
        "infeasible": False,
        "error_message": None,
    }
    with caplog.at_level(logging.WARNING, logger="emhass.last_run"):
        result = last_run.is_recent(data_path, max_age_seconds=60)
    assert result is False
    assert any("malformed timestamp" in rec.message for rec in caplog.records)


def test_record_unknown_status_maps_to_error(data_path):
    """Unrecognised solver status must be surfaced as 'error', not 'ok'."""
    last_run.record(
        data_path,
        action="naive-mpc-optim",
        stage_times={},
        optim_status="Unknown",
        infeasible=False,
        duration_total_seconds=1.0,
        schema_version="1.0",
    )
    snap = last_run.read(data_path)
    assert snap is not None
    assert snap["status"] == "error"


def test_record_infeasible_status(data_path):
    last_run.record(
        data_path,
        action="naive-mpc-optim",
        stage_times={},
        optim_status="Infeasible",
        infeasible=True,
        duration_total_seconds=1.0,
        schema_version="1.0",
    )
    snap = last_run.read(data_path)
    assert snap is not None
    assert snap["status"] == "infeasible"
    assert snap["infeasible"] is True


def test_record_is_thread_safe(data_path):
    """100 records across 10 threads converge to a consistent file + cache."""
    import threading as _t

    barrier = _t.Barrier(10)

    def worker(thread_id: int):
        barrier.wait()
        for i in range(10):
            last_run.record(
                data_path,
                action="naive-mpc-optim",
                stage_times={"t": float(thread_id * 10 + i)},
                optim_status="Optimal",
                infeasible=False,
                duration_total_seconds=0.1,
                schema_version="1.0",
            )

    threads = [_t.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = last_run.read(data_path)
    assert snap is not None
    assert snap["status"] == "ok"
    assert "t" in snap["stage_times"]
    assert isinstance(snap["stage_times"]["t"], float)
