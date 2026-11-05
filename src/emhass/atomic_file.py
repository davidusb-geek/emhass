"""Replace an on-disk text result without exposing an incomplete file to readers."""

import os
import stat
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TextIO


def atomic_write(path: Path, write: Callable[[TextIO], None]) -> None:
    """Stage a complete text file beside its destination, then replace it atomically.

    Unique temporary names allow concurrent writers. The caller must separately
    coordinate which writer is entitled to publish the latest result.

    The staging file is created with a plain ``open`` so it gets the permissions
    the destination would have had before (``0o666`` minus the umask), not the
    owner-only ``0o600`` of ``tempfile``: other containers or users that read
    the shared data folder keep working. When the destination already exists its
    permission bits are carried over, as an in-place rewrite would have kept them.
    """
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8", newline="") as stream:
            write(stream)
        try:
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        except FileNotFoundError:
            pass  # new file: keep the umask-derived default
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
