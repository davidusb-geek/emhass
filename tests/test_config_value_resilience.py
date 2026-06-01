"""
Resilience tests: stringly-typed config values, null inputs, and mask-builder edge cases.

Parametrised suite (#901): for every optim_conf array parameter, inject bad runtime
values (scalar null, None element, string element) via treat_runtimeparams and assert
the optimizer survives without raising.  Guarded params MUST pass; unguarded bugs are
xfail(strict=True).

Specific regression guards (#873): def_current_state stringly-typed 'False' must not
fire the single-constant pin; None elements must coerce to False; windows outside the
horizon must deactivate the load.
"""

import asyncio
import csv
import json
import pathlib

import numpy as np
import orjson
import pandas as pd
import pytest

from emhass import utils
from emhass.optimization import Optimization

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "defaults_path": root / "src/emhass/data/config_defaults.json",
    "associations_path": root / "src/emhass/data/associations.csv",
}
logger, _ = utils.get_logger(__name__, emhass_conf, save_to_file=False)

_VAR_LOAD_COST = "unit_load_cost"
_VAR_PROD_PRICE = "unit_prod_price"


# ─────────────────────────── param inventory ────────────────────────────────


def _optim_array_param_names() -> list[str]:
    """Return all optim_conf params whose 'input' starts with 'array.'.

    Reads param_definitions.json (schema source of truth) and associations.csv
    (config-category mapping) so the inventory is derived, not hardcoded.
    """
    pd_path = root / "src/emhass/static/data/param_definitions.json"
    pd_data = json.loads(pd_path.read_text(encoding="utf-8"))

    optim_conf_params: set[str] = set()
    with open(emhass_conf["associations_path"], newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 3 and row[0] == "optim_conf":
                optim_conf_params.add(row[2])

    result: list[str] = []
    for section_entries in pd_data.values():
        for name, defn in section_entries.items():
            if defn.get("input", "").startswith("array.") and name in optim_conf_params:
                result.append(name)
    return result


_OPTIM_ARRAY_PARAMS: list[str] = _optim_array_param_names()

# Params whose guard is already merged — tests MUST pass and act as regression
# guards.  Add to this set when a new guard lands.
_GUARDED: frozenset[str] = frozenset({"cost_forecast_per_deferrable_load"})

# Params known to crash because no stringly-typed guard exists yet.
# Key = param name, value = reason string for xfail (include issue ref).
# When a guard lands, remove the entry; the test becomes a live regression guard.
_XFAIL_REASON: dict[str, str] = {
    "nominal_power_of_deferrable_loads": (
        "no stringly-typed guard; crashes at CVXPY constraint build (see #900)"
    ),
    "minimum_power_of_deferrable_loads": (
        "no stringly-typed guard; pad_list fails on non-list value (see #900)"
    ),
    "operating_hours_of_each_deferrable_load": (
        "no stringly-typed guard; pad_list fails on non-list value (see #900)"
    ),
    "set_deferrable_startup_penalty": (
        "no stringly-typed guard; string char used as float in penalty (see #900)"
    ),
    "deferrable_load_max_cost": (
        "no stringly-typed guard; string char used as float in penalty (see #900)"
    ),
    "set_deferrable_max_startups": (
        "no stringly-typed guard; string char used as int for max_starts (see #900)"
    ),
}

# Per-param, per-bad-value overrides for the xfail marker.
# Maps (param_name, value_id) — if present, this specific (param, value) combo
# is NOT marked xfail even if param is in _XFAIL_REASON; a partial guard exists
# that handles this specific bad-value shape.
_XFAIL_EXCLUDE: frozenset[tuple[str, str]] = frozenset(
    {
        # start timesteps: validate_def_timewindow(0.5, 0) produces start>end →
        # slice [def_start:def_end] is never reached.  None element handled by
        # the [s if s is not None else 0] guard (opt.py:2868).
        ("start_timesteps_of_each_deferrable_load", "none_element"),
        # set_deferrable_max_startups: `if max_starts and max_starts > 0:` guard
        # short-circuits on None (falsy) and 0.5 only appears as a CVXPY float
        # bound which is valid.  No crash for the [None, 0.5] shape.
        ("set_deferrable_max_startups", "none_element"),
    }
)

# Bad values to inject; (id_suffix, value) pairs.
_BAD_VALUES: list[tuple[str, object]] = [
    ("scalar_null", "null"),
    ("none_element", [None, 0.5]),
    ("str_element", ["0.0", "0.0"]),
]

# start/end timesteps crash with scalar_null and str_element (no guard there)
_XFAIL_REASON["start_timesteps_of_each_deferrable_load"] = (
    "no stringly-typed guard for scalar/string values; "
    "None-element path is guarded but scalar-null/string-element crash (see #900)"
)
_XFAIL_REASON["end_timesteps_of_each_deferrable_load"] = (
    "no stringly-typed guard for scalar/string values; "
    "None-element path is guarded but scalar-null/string-element crash (see #900)"
)


def _make_cases() -> list:
    cases = []
    for param in _OPTIM_ARRAY_PARAMS:
        reason = _XFAIL_REASON.get(param)
        for val_id, bad_val in _BAD_VALUES:
            marks: list = []
            if reason and (param, val_id) not in _XFAIL_EXCLUDE:
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            cases.append(pytest.param(param, bad_val, id=f"{param}-{val_id}", marks=marks))
    return cases


_CASES: list = _make_cases()

# ──────────────────────── module-level default params ───────────────────────


async def _build_default_params() -> str:
    config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
    _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
    params = await utils.build_params(emhass_conf, secrets, config, logger)
    return orjson.dumps(params).decode("utf-8")


# Built once at import; reused across all test cases for speed.
_PARAMS_JSON: str = asyncio.run(_build_default_params())
_RHCONF, _OPTCONF, _PCONF = utils.get_yaml_parse(_PARAMS_JSON, logger)


# ─────────────────────────────── helpers ────────────────────────────────────


def _make_forecast_inputs(rh_conf: dict, n: int = 48):
    freq = rh_conf["optimization_time_step"]
    tz = rh_conf["time_zone"]
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz=tz)
    ulc = np.full(n, 0.2)
    upp = np.full(n, 0.1)
    df = pd.DataFrame({_VAR_LOAD_COST: ulc, _VAR_PROD_PRICE: upp}, index=idx)
    return df, np.zeros(n), np.full(n, 1000.0), ulc, upp


def _treat(rp_dict: dict):
    """Run treat_runtimeparams with the given runtimeparams dict synchronously."""
    rp_json = orjson.dumps(rp_dict).decode("utf-8")

    async def _run():
        return await utils.treat_runtimeparams(
            rp_json,
            _PARAMS_JSON,
            _RHCONF,
            _OPTCONF,
            _PCONF,
            "dayahead-optim",
            logger,
            emhass_conf,
        )

    return asyncio.run(_run())


def _make_opt(rh_conf, opt_conf, pl_conf):
    return Optimization(
        rh_conf,
        opt_conf,
        pl_conf,
        _VAR_LOAD_COST,
        _VAR_PROD_PRICE,
        "profit",
        emhass_conf,
        logger,
    )


# ─────────────────── parametrised resilience tests (#901) ───────────────────


@pytest.mark.parametrize("param_name,bad_value", _CASES)
def test_stringly_typed_resilience(param_name: str, bad_value: object):
    """Injecting bad runtime values for param_name must not raise.

    pytest.mark.parametrize is not supported on unittest.TestCase methods, so
    this test lives as a module-level function.  The async treat_runtimeparams
    call is wrapped in asyncio.run() which is fine outside an active event loop.
    """
    runtimeparams_json = orjson.dumps({param_name: bad_value}).decode("utf-8")

    async def _run():
        return await utils.treat_runtimeparams(
            runtimeparams_json,
            _PARAMS_JSON,
            _RHCONF,
            _OPTCONF,
            _PCONF,
            "dayahead-optim",
            logger,
            emhass_conf,
        )

    _, rh_conf, opt_conf, pl_conf = asyncio.run(_run())

    df, p_pv, p_load, ulc, upp = _make_forecast_inputs(rh_conf)
    opt = Optimization(
        rh_conf,
        opt_conf,
        pl_conf,
        _VAR_LOAD_COST,
        _VAR_PROD_PRICE,
        "profit",
        emhass_conf,
        logger,
    )
    opt.perform_optimization(df, p_pv, p_load, ulc, upp)


# ── Test A: single-shot string-bool regression ───────────────────────────────


def test_def_current_state_stringly_false_does_not_pin_window_open():
    """A stringly-typed 'False' in def_current_state must NOT fire the pin.

    Before the fix: bool('False') = True → pin sets lb_mask[:required_steps] = 1
    → single-constant load starts at t=0 ignoring start_timesteps.
    After the fix: _cast_bool('False') = False → no pin → window respected.
    """
    rp = {
        # 2 loads (match default num_def_loads=2)
        "nominal_power_of_deferrable_loads": [3000, 700],
        "operating_hours_of_each_deferrable_load": [2, 1],
        "start_timesteps_of_each_deferrable_load": [10, 0],
        "end_timesteps_of_each_deferrable_load": [40, 0],
        "set_deferrable_load_single_constant": ["True", "False"],
        "treat_deferrable_load_as_semi_cont": [False, True],
        # BUG TRIGGER: stringly-typed 'False' — bool('False') = True in Python
        "def_current_state": ["False", "False"],
    }

    _, rh_conf, opt_conf, pl_conf = _treat(rp)

    n = 48
    df, p_pv, p_load, ulc, upp = _make_forecast_inputs(rh_conf, n)
    opt = _make_opt(rh_conf, opt_conf, pl_conf)
    res = opt.perform_optimization(df, p_pv, p_load, ulc, upp)

    p_def0 = res["P_deferrable0"].values
    pre_window_sum = p_def0[:10].sum()
    assert pre_window_sum == 0, (
        f"Load 0 must not run before its configured window (slot 10); "
        f"pre-window power sum = {pre_window_sum:.1f} W "
        f"(non-zero means spurious pin fired due to stringly-typed def_current_state)"
    )


# ── Test B: MPC rolling keystone ─────────────────────────────────────────────


def test_def_current_state_stringly_false_no_spurious_pin_across_mpc_ticks():
    """MPC rolling: stringly 'False' must not command load ON before the window start.

    Simulates 3 consecutive MPC ticks (advancing start_timesteps by 1 each tick).
    At each tick the first-slot result is the actual MPC command. If the pin fires
    spuriously, the load is commanded ON immediately; the loop checks all 3 ticks.

    This is the keystone test: a single-shot (Test A) might pass if the full-horizon
    optimizer happens not to start the load at slot 0, but MPC's per-tick command is
    unambiguous — slot 0 must be zero when the window hasn't opened yet.
    """
    for tick in range(3):
        start_ts = 10 - tick  # window start approaches by 1 slot each tick
        end_ts = 40 - tick

        rp = {
            "nominal_power_of_deferrable_loads": [3000, 700],
            "operating_hours_of_each_deferrable_load": [2, 1],
            "start_timesteps_of_each_deferrable_load": [start_ts, 0],
            "end_timesteps_of_each_deferrable_load": [end_ts, 0],
            "set_deferrable_load_single_constant": ["True", "False"],
            "treat_deferrable_load_as_semi_cont": [False, True],
            # Stringly-typed 'False' remains the same across ticks (not driven True)
            "def_current_state": ["False", "False"],
        }

        _, rh_conf, opt_conf, pl_conf = _treat(rp)

        # MPC state stays 'False' (no spurious start should have occurred)
        assert opt_conf["def_current_state"][0] is False, (
            f"Tick {tick}: def_current_state[0] should be False after coercion; "
            f"got {opt_conf['def_current_state'][0]!r}"
        )

        n = 48
        df, p_pv, p_load, ulc, upp = _make_forecast_inputs(rh_conf, n)
        opt = _make_opt(rh_conf, opt_conf, pl_conf)
        res = opt.perform_optimization(df, p_pv, p_load, ulc, upp)

        current_tick_cmd = res["P_deferrable0"].values[0]
        assert current_tick_cmd == 0, (
            f"Tick {tick}: MPC current-slot command must be 0 "
            f"(window starts at slot {start_ts}); got {current_tick_cmd:.1f} W"
        )


# ── Test D: null def_current_state coercion ──────────────────────────────────


def test_def_current_state_null_elements_do_not_crash_optimizer():
    """def_current_state arriving as [None, None] (HA JSON null) must coerce to
    [False, False], not [None, None].

    _cast_bool(None): str(None).capitalize() = 'None'; ast.literal_eval('None')
    returns Python None (not an exception) → no fallback → None propagates.
    Consumer _update_def_current_state_params (optimization.py:461-469) has an
    explicit isinstance(state, bool|int|float) guard; None hits the else branch
    and raises ValueError → optimization crash.
    """
    rp = {
        "nominal_power_of_deferrable_loads": [3000, 700],
        "operating_hours_of_each_deferrable_load": [2, 1],
        "start_timesteps_of_each_deferrable_load": [10, 0],
        "end_timesteps_of_each_deferrable_load": [40, 0],
        "set_deferrable_load_single_constant": ["True", "False"],
        "treat_deferrable_load_as_semi_cont": [False, True],
        "def_current_state": [None, None],
    }
    _, rh_conf, opt_conf, pl_conf = _treat(rp)

    assert opt_conf["def_current_state"] == [False, False], (
        f"None must coerce to False; got {opt_conf['def_current_state']!r}"
    )

    # Full round-trip: optimizer must not raise ValueError on None state.
    n = 48
    df, p_pv, p_load, ulc, upp = _make_forecast_inputs(rh_conf, n)
    opt = _make_opt(rh_conf, opt_conf, pl_conf)
    opt.perform_optimization(df, p_pv, p_load, ulc, upp)


# ── Test E: scalar runtimeparams coercion ────────────────────────────────────


def test_scalar_runtimeparams_coerce_to_bool():
    """Scalar (non-list) def_current_state and set_deferrable_load_single_constant
    must coerce correctly and pad to num_def_loads — covers the else-branches."""
    rp = {
        "nominal_power_of_deferrable_loads": [3000, 700],
        "operating_hours_of_each_deferrable_load": [2, 1],
        "start_timesteps_of_each_deferrable_load": [0, 0],
        "end_timesteps_of_each_deferrable_load": [0, 0],
        # Scalar bool (not list) — hits the else-branch in each handler
        "def_current_state": False,
        "set_deferrable_load_single_constant": False,
    }
    _, _, opt_conf, _ = _treat(rp)
    assert opt_conf["def_current_state"] == [False, False], (
        f"Scalar False must pad to [False, False] for 2 loads; got {opt_conf['def_current_state']!r}"
    )
    assert opt_conf["set_deferrable_load_single_constant"] == [False, False], (
        f"Scalar False must pad to [False, False] for 2 loads; "
        f"got {opt_conf['set_deferrable_load_single_constant']!r}"
    )


# ── Test C: horizon-outside-window regression guard ──────────────────────────


def test_window_entirely_outside_horizon_zeros_mask():
    """A window configured entirely outside the optimization horizon must deactivate the load.

    Regression guard for the fix merged in commit eea5d25 (upstream/master, May 27 2026):
    when start_timesteps and end_timesteps both exceed n, window_mask must stay zero
    (load cannot run this tick), NOT open the whole horizon.

    This test is intentionally GREEN on current master; it locks in the correct behavior
    that replaced the former `window_mask[:] = 1.0` fallback.
    """
    rp = {
        "nominal_power_of_deferrable_loads": [3000, 700],
        "operating_hours_of_each_deferrable_load": [2, 1],
        # Both loads: window far outside the 48-slot horizon
        "start_timesteps_of_each_deferrable_load": [200, 200],
        "end_timesteps_of_each_deferrable_load": [220, 220],
        "set_deferrable_load_single_constant": [False, False],
        "treat_deferrable_load_as_semi_cont": [False, True],
        "def_current_state": [False, False],
    }

    _, rh_conf, opt_conf, pl_conf = _treat(rp)

    n = 48
    df, p_pv, p_load, ulc, upp = _make_forecast_inputs(rh_conf, n)
    opt = _make_opt(rh_conf, opt_conf, pl_conf)
    res = opt.perform_optimization(df, p_pv, p_load, ulc, upp)

    p_def0_total = res["P_deferrable0"].values.sum()
    assert p_def0_total == 0, (
        f"Load 0 must not run anywhere when its window [{rp['start_timesteps_of_each_deferrable_load'][0]}, "
        f"{rp['end_timesteps_of_each_deferrable_load'][0]}] is outside the "
        f"horizon [0, {n}]; got total power = {p_def0_total:.1f} W"
    )


# ── Test F (#899): pin must not fire when window is outside horizon ───────────


def test_running_single_const_with_window_outside_horizon_stays_feasible():
    """A currently-running single-constant load whose window is outside the horizon
    must NOT be pinned ON — pinning forces param_running_lb while the load-active
    loop deactivates the load, yielding an infeasible MILP (#899).

    Trigger: set_deferrable_load_single_constant[0]=True, def_current_state[0]=True
    (load was running at the end of the previous MPC horizon), and a window entirely
    outside [0, n].  The energy and timestep blocks already carry the
    `k not in window_empty_loads` guard; the pin block did not, so it set
    param_running_lb[0][:pinned_steps]=1 while param_load_active[0]=0 conflicted.

    After the fix the pin block routes load 0 to its else branch (lb=0, sc=0),
    the solve stays Optimal, and load 0 is not scheduled.
    """
    rp = {
        "nominal_power_of_deferrable_loads": [3000, 700],
        "operating_hours_of_each_deferrable_load": [2, 1],
        # Load 0: window entirely outside the 48-slot horizon
        "start_timesteps_of_each_deferrable_load": [200, 0],
        "end_timesteps_of_each_deferrable_load": [220, 0],
        "set_deferrable_load_single_constant": [True, False],
        "treat_deferrable_load_as_semi_cont": [False, True],
        # Load 0 was running at the end of the previous MPC horizon
        "def_current_state": [True, False],
    }

    _, rh_conf, opt_conf, pl_conf = _treat(rp)

    n = 48
    df, p_pv, p_load, ulc, upp = _make_forecast_inputs(rh_conf, n)
    opt = _make_opt(rh_conf, opt_conf, pl_conf)
    res = opt.perform_optimization(df, p_pv, p_load, ulc, upp)

    status = res["optim_status"].iloc[0]
    assert status == "Optimal", (
        f"Single-const load running with window outside horizon must keep the MILP "
        f"feasible; got optim_status={status!r}. 'Optimal (Relaxed)' means the MILP "
        f"went infeasible and fell back to the continuous LP — the #899 bug, where the "
        f"pin forces param_running_lb while the load is deactivated."
    )

    p_def0_total = res["P_deferrable0"].values.sum()
    assert p_def0_total == 0, (
        f"Load 0 must not be scheduled when its window is outside the horizon; "
        f"got total power = {p_def0_total:.1f} W"
    )
