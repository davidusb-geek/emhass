"""Persistent state for the most-recent EMHASS optimization run.

Backs the GET /api/v1/last-run endpoint and the future GET /healthz endpoint
(AC-4). Single source of truth for both Quart Web and CLI surfaces.

State model: in-memory cache (_cache) + write-through to
<data_path>/last_run.json. Thread-safe via _lock to handle multi-worker
Quart deployments.
"""

import json
import logging
import threading
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_LAST_RUN_FILENAME = "last_run.json"
_lock = threading.Lock()
_cache: dict | None = None
_logger = logging.getLogger(__name__)


def _path(data_path: Path) -> Path:
    return Path(data_path) / _LAST_RUN_FILENAME


_ERROR_MESSAGE_MAX_LEN = 200


def emhass_version() -> str:
    """Return the installed emhass package version, or 'unknown'."""
    try:
        return version("emhass")
    except PackageNotFoundError:
        return "unknown"


def _truncate_error_message(msg: str | None) -> str | None:
    """Length-bound error_message before it lands in the persisted snapshot.

    Acts as the dataflow boundary for callers that may forward solver
    exception strings: anything beyond _ERROR_MESSAGE_MAX_LEN is dropped
    so a long traceback or accidentally leaked secret cannot accumulate
    on disk. None passes through unchanged.
    """
    if msg is None:
        return None
    return str(msg)[:_ERROR_MESSAGE_MAX_LEN]


def record(
    data_path: Path,
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

    Status mapping: optim_status "Optimal" -> "ok", "Infeasible" (or
    infeasible=True) -> "infeasible", anything else -> "error". This errs
    on the side of caution: an unrecognised solver status is surfaced as
    "error" rather than silently classified as healthy.
    """
    global _cache
    if optim_status == "Optimal":
        status = "ok"
    elif optim_status == "Infeasible" or infeasible:
        status = "infeasible"
    else:
        status = "error"

    # Build snapshot through explicit field-by-field assignment so the
    # data-flow boundary is obvious to static analysers: each field is
    # constrained to a known-safe value space (enums, timestamps, floats,
    # package metadata) and error_message is length-bounded.
    snap: dict = {}
    snap["status"] = status
    snap["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    snap["action"] = action
    snap["stage_times"] = stage_times
    snap["duration_total_seconds"] = duration_total_seconds
    snap["emhass_version"] = emhass_version()
    snap["schema_version"] = schema_version
    snap["infeasible"] = infeasible
    snap["error_message"] = _truncate_error_message(error_message)

    with _lock:
        _cache = snap
        target = _path(data_path)
        try:
            payload = json.dumps(snap, indent=2)
            target.write_text(payload, encoding="utf-8")
        except OSError as exc:
            _logger.warning("last_run: failed to write snapshot file", exc_info=exc)


def read(data_path: Path) -> dict | None:
    """Return the most recent snapshot, or None if no run yet.

    Reads the in-memory cache first; falls back to the on-disk file for
    cold-start after a restart. A corrupted JSON file logs a warning and
    returns None.
    """
    global _cache
    with _lock:
        if _cache is not None:
            return dict(_cache)
        path = _path(data_path)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _cache = data
            return dict(data)
        except (OSError, json.JSONDecodeError) as exc:
            _logger.warning("last_run: corrupt or unreadable snapshot file", exc_info=exc)
            return None


def is_recent(data_path: Path, max_age_seconds: int) -> bool:
    """Return True iff a snapshot exists and its timestamp is within max_age_seconds.

    Foundation for AC-4 /healthz. False when no run yet OR snapshot is too old.
    Malformed timestamps log a warning and return False so operators can detect
    corruption.
    """
    snap = read(data_path)
    if snap is None or snap.get("timestamp") is None:
        return False
    try:
        ts_str = snap["timestamp"].replace("Z", "+00:00")
        ts = datetime.fromisoformat(ts_str)
    except (ValueError, KeyError) as exc:
        _logger.warning("last_run: malformed timestamp in snapshot", exc_info=exc)
        return False
    age = (datetime.now(UTC) - ts).total_seconds()
    return age <= max_age_seconds
