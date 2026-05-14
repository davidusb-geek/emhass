"""Persistent state for the most-recent EMHASS optimization run.

Backs the GET /api/v1/last-run endpoint and the future GET /healthz endpoint
(AC-4). Single source of truth for both Quart Web and CLI surfaces.

State model: in-memory cache (_cache) + write-through to
<data_path>/last_run.json. Thread-safe via _lock to handle multi-worker
Quart deployments.
"""

import json
import threading
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_LAST_RUN_FILENAME = "last_run.json"
_lock = threading.Lock()
_cache: dict | None = None


def _path(emhass_conf: dict) -> Path:
    return Path(emhass_conf["data_path"]) / _LAST_RUN_FILENAME


def _emhass_version() -> str:
    try:
        return version("emhass")
    except PackageNotFoundError:
        return "unknown"


def record(
    emhass_conf: dict,
    action: str,
    stage_times: dict,
    optim_status: str,
    infeasible: bool,
    duration_total_seconds: float,
    schema_version: str,
    error_message: str | None = None,
) -> None:
    """Persist a snapshot after a completed optimization run.

    Writes to both the in-memory cache and <data_path>/last_run.json under
    a lock. File write failure is logged-and-swallowed so a disk error never
    breaks the caller's return path.
    """
    global _cache
    if optim_status == "Optimal":
        status = "ok"
    elif optim_status == "Infeasible" or infeasible:
        status = "infeasible"
    else:
        status = "error" if error_message else "ok"

    snap = {
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "action": action,
        "stage_times": stage_times,
        "duration_total_seconds": duration_total_seconds,
        "emhass_version": _emhass_version(),
        "schema_version": schema_version,
        "infeasible": infeasible,
        "error_message": error_message,
    }

    with _lock:
        _cache = snap
        try:
            _path(emhass_conf).write_text(json.dumps(snap, indent=2), encoding="utf-8")
        except OSError as exc:
            import logging
            logging.getLogger(__name__).warning(
                "last_run: failed to write %s: %s", _path(emhass_conf), exc
            )


def read(emhass_conf: dict) -> dict | None:
    """Return the most recent snapshot, or None if no run yet.

    Reads the in-memory cache first; falls back to the on-disk file for
    cold-start after a restart. A corrupted JSON file logs a warning and
    returns None.
    """
    global _cache
    with _lock:
        if _cache is not None:
            return dict(_cache)
        path = _path(emhass_conf)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _cache = data
            return dict(data)
        except (OSError, json.JSONDecodeError) as exc:
            import logging
            logging.getLogger(__name__).warning(
                "last_run: corrupt or unreadable %s: %s", path, exc
            )
            return None


def is_recent(emhass_conf: dict, max_age_seconds: int) -> bool:
    """Return True iff a snapshot exists and its timestamp is within max_age_seconds.

    Foundation for AC-4 /healthz. False when no run yet OR snapshot is too old.
    """
    snap = read(emhass_conf)
    if snap is None or snap.get("timestamp") is None:
        return False
    try:
        ts_str = snap["timestamp"].replace("Z", "+00:00")
        ts = datetime.fromisoformat(ts_str)
    except (ValueError, KeyError):
        return False
    age = (datetime.now(UTC) - ts).total_seconds()
    return age <= max_age_seconds
