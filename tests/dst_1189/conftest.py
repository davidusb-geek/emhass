"""pytest header: which EMHASS / pandas / pytz the suite runs against."""

import importlib.metadata as md
import importlib.util
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

# time-machine is not in the "test" extra, so a plain `pytest` (upstream CI) skips this
# folder instead of failing at collection. Run it with `--with time-machine`.
if importlib.util.find_spec("time_machine") is None:
    collect_ignore_glob = ["test_*.py"]


def pytest_report_header(config):
    import emhass

    if "collect_ignore_glob" in globals():
        return ["emhass-dst-tests: skipped, time-machine not installed"]

    vers = {p: md.version(p) for p in ("emhass", "pandas", "pytz", "tzdata", "time-machine")}
    return [f"emhass-dst-tests: {vers}", f"emhass from: {pathlib.Path(emhass.__file__).parent}"]


def pytest_configure(config):
    # marker values (minutes since 2026) are absurd temperatures for pvlib; harmless here
    config.addinivalue_line("filterwarnings", "ignore::RuntimeWarning")
