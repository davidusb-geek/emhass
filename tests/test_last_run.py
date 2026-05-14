"""Unit tests for src/emhass/last_run.py (AC-3)."""

import logging
import pytest
from datetime import UTC, datetime

from emhass import last_run


@pytest.fixture
def emhass_conf(tmp_path):
    """Minimal emhass_conf with data_path pointing at a tmp dir."""
    return {"data_path": tmp_path}


@pytest.fixture(autouse=True)
def reset_cache():
    """Clear in-memory cache between tests to avoid cross-test bleed."""
    last_run._cache = None
    yield
    last_run._cache = None


def test_record_then_read_round_trip(emhass_conf):
    last_run.record(
        emhass_conf,
        action="naive-mpc-optim",
        stage_times={"pv_forecast": 1.2, "load_forecast": 0.5},
        optim_status="Optimal",
        infeasible=False,
        duration_total_seconds=3.7,
        schema_version="1.0",
    )
    snap = last_run.read(emhass_conf)
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


def test_read_returns_none_when_no_run(emhass_conf):
    assert last_run.read(emhass_conf) is None


def test_read_recovers_from_corrupt_file(emhass_conf, caplog):
    last_run._path(emhass_conf).write_text("not-json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="emhass.last_run"):
        result = last_run.read(emhass_conf)
    assert result is None
    assert any("corrupt or unreadable" in rec.message for rec in caplog.records)


def test_is_recent_true_when_just_recorded(emhass_conf):
    last_run.record(
        emhass_conf,
        action="naive-mpc-optim",
        stage_times={},
        optim_status="Optimal",
        infeasible=False,
        duration_total_seconds=1.0,
        schema_version="1.0",
    )
    assert last_run.is_recent(emhass_conf, max_age_seconds=60) is True


def test_is_recent_false_when_no_run(emhass_conf):
    assert last_run.is_recent(emhass_conf, max_age_seconds=60) is False


def test_is_recent_false_when_too_old(emhass_conf):
    from datetime import timedelta
    old_ts = (datetime.now(UTC) - timedelta(hours=2)).isoformat(timespec="seconds").replace("+00:00", "Z")
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
    assert last_run.is_recent(emhass_conf, max_age_seconds=600) is False


def test_record_is_thread_safe(emhass_conf):
    """100 records across 10 threads converge to a consistent file + cache."""
    import threading as _t

    barrier = _t.Barrier(10)

    def worker(thread_id: int):
        barrier.wait()
        for i in range(10):
            last_run.record(
                emhass_conf,
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

    snap = last_run.read(emhass_conf)
    assert snap is not None
    assert snap["status"] == "ok"
    assert "t" in snap["stage_times"]
    assert isinstance(snap["stage_times"]["t"], float)
