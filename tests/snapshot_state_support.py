"""Isolation for the process-wide last-run and plan snapshot state.

``last_run`` and ``plan_store`` each keep three module globals: ``_cache`` (the in-memory
snapshot), ``_unpersisted`` (True while a failed write is pending) and ``_persisted`` (the
snapshot this process last wrote). They are compared with each other when a snapshot is
read, so clearing only ``_cache`` leaves a stale ``_persisted`` that changes the outcome of
a later test. Enter ``isolated_snapshot_state()`` to start from a cold process and restore
whatever was there before, even when the test fails.
"""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from emhass import last_run, plan_store

_COLD = {"_cache": None, "_unpersisted": False, "_persisted": None}


@contextmanager
def isolated_snapshot_state() -> Iterator[None]:
    """Run with cold ``last_run`` and ``plan_store`` state; restore the prior state on exit."""
    with ExitStack() as stack:
        for module in (last_run, plan_store):
            for name, cold in _COLD.items():
                stack.enter_context(patch.object(module, name, cold))
        yield
