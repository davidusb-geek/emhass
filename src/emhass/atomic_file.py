"""Replace an on-disk text result without exposing an incomplete file to readers."""

import os
import tempfile
from pathlib import Path
from typing import Callable, TextIO


def atomic_write(path: Path, write: Callable[[TextIO], None]) -> None:
    """Stage a complete text file beside its destination, then replace it atomically.

    Unique temporary names allow concurrent writers. The caller must separately
    coordinate which writer is entitled to publish the latest result.
    """
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            write(stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
