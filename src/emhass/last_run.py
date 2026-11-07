"""Persistent state for the most-recent EMHASS optimization run.

Backs the GET /api/v1/last-run endpoint and the future GET /healthz endpoint
(AC-4). Single source of truth for both Quart Web and CLI surfaces.

State model: in-memory cache (_cache) + write-through to
<data_path>/last_run.json. Thread-safe via _lock to handle multi-worker
Quart deployments. The file is the shared source of truth across workers; the
cache outranks it only while a snapshot is pending (its write failed) and is
strictly newer than the file.
"""

import json
import logging
import threading
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from emhass.atomic_file import atomic_write

_LAST_RUN_FILENAME = "last_run.json"
_lock = threading.Lock()
_cache: dict | None = None
# True while _cache holds a snapshot that could not be written to disk. Only then
# may the cache outrank an existing file: a successful write makes the file the
# single source of truth, so another worker's newer file is never hidden.
_unpersisted = False
# The snapshot this process last wrote successfully. While a later write is
# pending, a file that still equals it was not touched by another worker, so the
# pending snapshot is newer whatever the (one second) timestamps say.
_persisted: dict | None = None
_logger = logging.getLogger(__name__)

# Action names — single source of truth for `action` enum used by the
# /api/v1/last-run schema sidecar and the optim wrappers in command_line.py.
ACTION_PERFECT_OPTIM = "perfect-optim"
ACTION_DAYAHEAD_OPTIM = "dayahead-optim"
ACTION_NAIVE_MPC_OPTIM = "naive-mpc-optim"

# optim_status values whose plan is published and that count as a healthy run.
# Shared with the /api/v1/plan gate in command_line.py so the two endpoints agree.
# "Optimal_Inaccurate" is cvxpy's optimal_inaccurate, title-cased by the
# optimizer; CPLEX returns it (e.g. at a time limit with a feasible solution).
# HiGHS, the default, and Gurobi report a time-out as user_limit instead.
OK_OPTIM_STATUSES = frozenset(
    {"Optimal", "Optimal_Inaccurate", "Optimal (Relaxed)", "Optimal (Incumbent)"}
)


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
) -> str:
    """Persist a snapshot after a completed optimization run.

    Writes to both the in-memory cache and <data_path>/last_run.json under
    a lock. File write failure is logged-and-swallowed so a disk error never
    breaks the caller's return path. The snapshot is the one snapshot()
    builds.
    """
    global _cache, _unpersisted, _persisted
    snap = snapshot(
        action,
        stage_times,
        optim_status,
        infeasible,
        duration_total_seconds,
        schema_version,
        error_message,
    )

    with _lock:
        _cache = snap
        target = _path(data_path)
        try:
            atomic_write(target, lambda stream: json.dump(snap, stream, indent=2))
            _unpersisted = False
            _persisted = snap
        except OSError as exc:
            _unpersisted = True
            _logger.warning("last_run: failed to write snapshot file", exc_info=exc)

    # Return the stamped timestamp so callers (e.g. _record_optim_snapshot)
    # can reuse the exact same instant for the plan snapshot's generated_at,
    # keeping /api/v1/last-run and /api/v1/plan in agreement. Existing callers
    # that ignore the return value are unaffected.
    return snap["timestamp"]


def snapshot(
    action: str,
    stage_times: dict,
    optim_status: str,
    infeasible: bool,
    duration_total_seconds: float,
    schema_version: str,
    error_message: str | None = None,
) -> dict:
    """Build the snapshot of a completed optimization run, without storing it.

    record() stores it for /api/v1/last-run; a dry run returns it instead.

    Status mapping: every status whose plan the optimizer publishes -> "ok":
    "Optimal", "Optimal_Inaccurate" (CVXPY's optimal_inaccurate), "Optimal
    (Relaxed)" (the relaxed-LP fallback) and "Optimal (Incumbent)" (a
    time-limited solve's feasible incumbent). "Infeasible" (or
    infeasible=True) -> "infeasible", anything else -> "error". This errs
    on the side of caution: an unrecognised solver status is surfaced as
    "error" rather than silently classified as healthy.
    """
    if optim_status in OK_OPTIM_STATUSES:
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
    return snap


def instant(value: object) -> datetime | None:
    """Parse an ISO timestamp to an aware UTC datetime; naive means UTC, junk is None."""
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def pending_wins(cached: dict | None, on_disk: dict | None, key: str) -> bool:
    """True iff an unpersisted ``cached`` snapshot is strictly newer than ``on_disk``.

    ``on_disk`` None (missing or unreadable file) lets a valid pending snapshot win;
    an invalid or equal timestamp never does.
    """
    if cached is None:
        return False
    cached_at = instant(cached.get(key))
    if cached_at is None:
        return False
    if on_disk is None:
        return True
    disk_at = instant(on_disk.get(key))
    return disk_at is None or cached_at > disk_at


def read(data_path: Path) -> dict | None:
    """Return the most recent snapshot, or None if no run yet.

    Read the shared on-disk file each time: a cached value from another worker
    may have been superseded. The in-memory snapshot is returned instead when the
    file is missing, or when this process holds a snapshot whose write failed that
    is newer than the file (or the file is unreadable): the file is still the one
    this process wrote earlier, or its timestamp is strictly older. A corrupt file
    otherwise logs a warning and returns None.
    """
    global _cache, _unpersisted, _persisted
    with _lock:
        path = _path(data_path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("snapshot root is not a JSON object")
        except FileNotFoundError:
            return dict(_cache) if _cache is not None else None
        except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
            _logger.warning("last_run: corrupt or unreadable snapshot file", exc_info=exc)
            return dict(_cache) if _unpersisted and _cache is not None else None
        if _unpersisted and (data == _persisted or pending_wins(_cache, data, "timestamp")):
            return dict(_cache)
        _cache = data
        _persisted = data
        _unpersisted = False
        return dict(data)


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
