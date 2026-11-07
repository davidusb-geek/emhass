"""Replace an on-disk text result without exposing an incomplete file to readers."""

import os
import stat
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TextIO


def _replace(source: Path, destination: Path) -> None:
    """``os.replace``; on Windows retry briefly while a reader holds the destination open."""
    for attempt in range(10):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if sys.platform != "win32" or attempt == 9:
                raise
            time.sleep(0.05)


def atomic_write(path: Path, write: Callable[[TextIO], None]) -> None:
    """Stage a complete text file beside its destination, then replace it atomically.

    Unique temporary names allow concurrent writers. The caller must separately
    coordinate which writer is entitled to publish the latest result.

    The staging file is created with a plain ``open`` so it gets the permissions
    the destination would have had before (``0o666`` minus the umask), not the
    owner-only ``0o600`` of ``tempfile``: other containers or users that read
    the shared data folder keep working. When the destination already exists its
    permission bits are carried over, as an in-place rewrite would have kept them.
    A symlinked destination stays a symlink: its target is the file replaced.
    """
    # Replace what a symlink points to, not the link: an in-place rewrite followed it,
    # and utils.replace_file does the same.
    path = Path(os.path.realpath(path))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8", newline="") as stream:
            write(stream)
        try:
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        except FileNotFoundError:
            pass  # new file: keep the umask-derived default
        _replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
