"""Persistent state for the most-recent EMHASS optimization PLAN.

Backs the GET /api/v1/plan endpoint. Mirrors last_run.py: in-memory cache
(_cache) + write-through to <data_path>/plan_latest.json, lock-guarded for
multi-worker Quart deployments. The plan is the opt_res schedule serialized
to JSON records; generated_at is supplied by the caller (the same timestamp
last_run.record stamps) so the two endpoints agree.
"""

import json
import logging
import threading
from pathlib import Path

import pandas as pd

from emhass.atomic_file import atomic_write
from emhass.last_run import pending_wins

_PLAN_FILENAME = "plan_latest.json"
_lock = threading.Lock()
_cache: dict | None = None
# True while _cache holds a plan whose write failed (see last_run._unpersisted).
_unpersisted = False
# The plan this process last wrote successfully (see last_run._persisted).
_persisted: dict | None = None
_logger = logging.getLogger(__name__)


def _path(data_path: Path) -> Path:
    return Path(data_path) / _PLAN_FILENAME


def serialize(opt_res: pd.DataFrame) -> list[dict]:
    """opt_res DataFrame -> JSON-safe records (index as 'timestamp', NaN -> null)."""
    # to_json handles Timestamp -> ISO and NaN -> null; json.loads gives clean dicts.
    df = opt_res.copy()
    df.index.name = "timestamp"  # guarantee the reset_index column is 'timestamp', not 'index'
    return json.loads(df.reset_index().to_json(orient="records", date_format="iso"))


def record(data_path: Path, plan: list[dict], generated_at: str, schema_version: str) -> None:
    """Persist the latest plan snapshot to cache + <data_path>/plan_latest.json.

    Best-effort file write: an OSError is logged-and-swallowed so a disk error
    never breaks the optimization wrapper's return path.
    """
    global _cache, _unpersisted, _persisted
    snap = {
        "emhass_schema_version": schema_version,
        "generated_at": generated_at,
        "plan": plan,
    }
    with _lock:
        _cache = snap
        try:
            atomic_write(_path(data_path), lambda stream: json.dump(snap, stream))
            _unpersisted = False
            _persisted = snap
        except OSError as exc:
            _unpersisted = True
            _logger.warning("plan_store: failed to write snapshot file", exc_info=exc)


def read(data_path: Path) -> dict | None:
    """Return the latest plan snapshot, or None if no run yet.

    The shared file wins across workers. The in-memory plan is returned instead
    when the file is missing, or when its write failed and the file is either the
    plan this process wrote earlier or strictly older (or is unreadable).
    """
    global _cache, _unpersisted, _persisted
    with _lock:
        # Other workers can commit a newer plan; the process-local cache is
        # not authoritative in a multi-worker server.
        try:
            data = json.loads(_path(data_path).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("snapshot root is not a JSON object")
        except FileNotFoundError:
            return dict(_cache) if _cache is not None else None
        except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
            _logger.warning("plan_store: corrupt or unreadable snapshot file", exc_info=exc)
            return dict(_cache) if _unpersisted and _cache is not None else None
        if _unpersisted and (data == _persisted or pending_wins(_cache, data, "generated_at")):
            return dict(_cache)
        _cache = data
        _persisted = data
        _unpersisted = False
        return dict(data)
