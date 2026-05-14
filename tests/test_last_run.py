"""Unit tests for src/emhass/last_run.py (AC-3)."""

import pytest

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
