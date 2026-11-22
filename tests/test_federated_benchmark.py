"""scripts/federated_benchmark.py: what it reports as a comparison, and what
it leaves out (a fallback, a solve that published no plan)."""

import importlib.util
import pathlib
import types

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "federated_benchmark", ROOT / "scripts/federated_benchmark.py"
)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def _res(used: str, reason: str = "") -> pd.DataFrame:
    return pd.DataFrame({"backend_used": [used], "backend_fallback_reason": [reason]})


def test_the_milp_is_read_from_the_plan_it_published():
    """The default solver's cost is the objective of the problem whose plan
    was published; a solve that published none is not compared."""
    capture = bench._Capture()
    opt = types.SimpleNamespace(solved_value=-5.0, optim_status="Optimal (Relaxed)")
    assert bench._objective(opt, capture, "cvxpy", None) == 5.0
    failed = types.SimpleNamespace(solved_value=-5.0, optim_status="Infeasible")
    assert bench._objective(failed, capture, "cvxpy", None) != bench._objective(
        failed, capture, "cvxpy", None
    )  # NaN


def test_a_coordinated_run_that_fell_back_is_not_compared(capsys):
    """A coordinated run that fell back to cvxpy is reported with its reason
    and left out of the summary, which says how many were left out."""
    capture = bench._Capture()
    capture.result = types.SimpleNamespace(upper=1.0)  # the coordinator had planned
    cost = bench._objective(None, capture, "dantzig_wolfe", _res("cvxpy", "a limit, 0.2 kWh"))
    assert cost != cost  # NaN
    out = capsys.readouterr().out
    assert "not compared" in out
    assert "a limit, 0.2 kWh" in out
    assert bench._objective(None, capture, "dantzig_wolfe", _res("dantzig_wolfe")) == 1.0
    rows = [bench._row("a", (1.0, 0.1), (1.0, 0.2)), bench._row("b", (1.0, 0.1), (cost, 0.2))]
    bench._summary("Runs", rows)
    out = capsys.readouterr().out
    assert "Runs (1 not compared): 2 runs" in out


def test_a_pair_that_straddles_a_slot_is_run_again(capsys):
    """Both backends plan the same slot: a pair that crosses a slot boundary
    between the two runs is run again, and one that keeps crossing is not
    compared."""
    starts = iter(["10:00", "10:30", "10:30", "10:30"])
    calls = []

    def run(backend):
        calls.append(backend)
        return (1.0 if backend == "cvxpy" else 1.1, 0.1, next(starts))

    milp, coord = bench._pair(run)
    assert calls == ["cvxpy", "dantzig_wolfe", "cvxpy", "dantzig_wolfe"]
    assert milp == (1.0, 0.1)
    assert coord == (1.1, 0.1)

    def drifting(backend, n=iter(range(100))):
        return (1.0, 0.1, next(n))

    milp, coord = bench._pair(drifting, tries=2)
    assert coord[0] != coord[0]  # NaN: not compared
    assert "not compared" in capsys.readouterr().out


def test_a_study_case_that_plans_nothing_is_not_compared(monkeypatch):
    """Inputs that could not be prepared, or an action that planned nothing,
    give no cost (NaN) and no first slot, instead of stopping the benchmark;
    the Home Assistant session a run opened is closed either way."""
    import asyncio

    closed = []

    class Session:
        async def close(self):
            closed.append(True)

    async def params():
        return {"optim_conf": {}}

    async def inputs(*args, **kwargs):
        return answer["idd"]

    async def action(idd, logger):
        return answer["res"]

    monkeypatch.setattr(bench, "_params", params)
    monkeypatch.setattr(bench, "set_input_data_dict", inputs)
    monkeypatch.setattr(bench, "dayahead_forecast_optim", action)
    answer = {"idd": False, "res": None}
    cost, _, start = asyncio.run(bench.study_case({}, "dayahead-optim", {}, "cvxpy", 0.0))
    assert cost != cost
    assert start is None
    answer = {"idd": {"rh": Session(), "opt": None}, "res": False}
    cost, _, start = asyncio.run(bench.study_case({}, "dayahead-optim", {}, "cvxpy", 0.0))
    assert cost != cost
    assert start is None
    assert closed == [True]
