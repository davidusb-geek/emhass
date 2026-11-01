"""Run the Node UI tests (tests/ui/*.test.cjs) with the Python suite so CI enforces them."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_NODE = shutil.which("node")
_ROOT = Path(__file__).resolve().parents[1]
_UI_TESTS = sorted((_ROOT / "tests" / "ui").glob("*.test.cjs"))


def test_node_ui_tests_pass():
    """The tests use paths relative to the repository root and need no packages (node:test)."""
    if _NODE is None:
        # GitHub-hosted runners ship Node: a missing node on CI is a failure, not a silent skip
        if os.environ.get("CI"):
            pytest.fail("node is required to run the UI tests on CI")
        pytest.skip("node not in PATH")
    assert _UI_TESTS, "no tests/ui/*.test.cjs found"
    result = subprocess.run(
        [_NODE, "--test", *map(str, _UI_TESTS)],
        cwd=_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]


def test_missing_node_fails_on_ci_and_skips_locally(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_NODE", None)
    monkeypatch.setenv("CI", "true")
    with pytest.raises(pytest.fail.Exception):
        test_node_ui_tests_pass()
    monkeypatch.delenv("CI")
    with pytest.raises(pytest.skip.Exception):
        test_node_ui_tests_pass()
