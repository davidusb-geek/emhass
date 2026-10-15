"""pytest header: which EMHASS / pandas / pytz the suite runs against."""

import importlib.metadata as md
import importlib.util
import pathlib
import sys
import warnings

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

# time-machine is in the "test" extra, so CI runs this folder. In an environment without it the
# folder is skipped instead of failing at collection.
if importlib.util.find_spec("time_machine") is None:
    collect_ignore_glob = ["test_*.py"]


def pytest_report_header(config):
    import emhass

    if "collect_ignore_glob" in globals():
        return ["emhass-dst-tests: skipped, time-machine not installed"]

    vers = {p: md.version(p) for p in ("emhass", "pandas", "pytz", "tzdata", "time-machine")}
    return [f"emhass-dst-tests: {vers}", f"emhass from: {pathlib.Path(emhass.__file__).parent}"]


@pytest.fixture(autouse=True)
def _dst_suite_isolation(monkeypatch):
    """Per-test (these tests only), restored afterwards: no SUPERVISOR_TOKEN, which would make
    get_ha_config() call the supervisor; and no RuntimeWarning noise, because the marker values
    (minutes since 2026) are absurd temperatures for pvlib. Neither leaks into other tests."""
    monkeypatch.delenv("SUPERVISOR_TOKEN", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield
