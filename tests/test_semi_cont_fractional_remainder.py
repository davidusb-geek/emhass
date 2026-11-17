#!/usr/bin/env python
"""Semi-continuous deferrable loads with a fractional operating requirement (#1105).

``operating_hours_of_each_deferrable_load`` that is not a whole number of
optimization timesteps used to make a strict ON/OFF load infeasible, so the whole
solve fell back to the relaxed LP (``Optimal (Relaxed)``) and the semi-continuous
and single-constant constraints of every load were dropped. The load now runs whole
steps at its nominal power plus exactly one remainder step whose interval-average
power is ``r * nominal``: the energy stays exact and the solve stays a strict MILP.

Synthetic and self-contained, built on the multi-battery test helper.
"""

import logging
import unittest

import cvxpy as cp
import numpy as np
import pandas as pd

from emhass.command_line import _deferrable_power_to_state
from tests.test_multi_battery_optimization import (
    build_optimization,
)

N = 12  # 6 h at 30 min
DT = 0.5
NOMINAL = 3000.0

BASE = {
    "set_use_battery": False,
    "number_of_deferrable_loads": 1,
    "nominal_power_of_deferrable_loads": [NOMINAL],
    "minimum_power_of_deferrable_loads": [0.0],
    "treat_deferrable_load_as_semi_cont": [True],
    "set_deferrable_load_single_constant": [False],
    "set_deferrable_startup_penalty": [0.0],
    "operating_hours_of_each_deferrable_load": [1.25],
    "start_timesteps_of_each_deferrable_load": [0],
    "end_timesteps_of_each_deferrable_load": [0],
}

CHEAP_FIRST = [0.05] * 4 + [0.30] * (N - 4)
RISING = [0.05 + 0.01 * t for t in range(N)]  # no ties: the next step is always cheapest


def make(**overrides):
    conf = dict(BASE)
    conf.update(overrides)
    return build_optimization(optim_overrides=conf)


def solve(opt, hours, prices=CHEAP_FIRST, n=N, pv=None, **kwargs):
    index = pd.date_range("2026-11-01", periods=n, freq="30min", tz="Europe/Brussels")
    loads = opt.optim_conf["number_of_deferrable_loads"]
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = list(prices)
    df_input["unit_prod_price"] = [0.0] * n
    df_input["outdoor_temperature_forecast"] = [5.0] * n
    return opt.perform_naive_mpc_optim(
        df_input,
        pd.Series([0.0] * n if pv is None else pv, index=index),
        pd.Series([500.0] * n, index=index),
        n,
        def_total_hours=hours,
        def_total_timestep=kwargs.pop("steps", None),
        def_start_timestep=kwargs.pop("start", [0] * loads),
        def_end_timestep=kwargs.pop("end", [0] * loads),
        **kwargs,
    )


def power(res, k=0):
    return res[f"P_deferrable{k}"].to_numpy()


def energy_wh(res, k=0):
    return float(power(res, k).sum() * DT)


def runs(p):
    """Index ranges (start, end exclusive) of the consecutive steps with power."""
    on = p > 1.0
    out, start = [], None
    for t, flag in enumerate(on):
        if flag and start is None:
            start = t
        if not flag and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, len(on)))
    return out


def assert_strict_on_off(testcase, p, nominal=NOMINAL, allowed_partial=0, level=None):
    """Every step is 0 or nominal, except at most ``allowed_partial`` steps, which
    carry exactly ``level`` (the remainder power) when it is given."""
    partial = [v for v in p if v > 1.0 and abs(v - nominal) > 1.0]
    testcase.assertLessEqual(len(partial), allowed_partial, p.round(0).tolist())
    if level is not None:
        for value in partial:
            testcase.assertAlmostEqual(value, level, delta=1e-3, msg=p.round(0).tolist())


class TestFractionalRemainder(unittest.TestCase):
    def test_issue_1105_reproduction_is_strict_and_exact(self):
        """1.25 h at 30 min = 2.5 steps: two full steps plus one half step, exactly
        3.75 kWh, solved as a strict MILP (not the relaxed fallback)."""
        opt = make()
        res = solve(opt, [1.25])
        self.assertEqual(opt.optim_status, "Optimal")
        p = power(res)
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 1.25, delta=1e-3)
        self.assertEqual(int((np.abs(p - NOMINAL) < 1.0).sum()), 2)
        self.assertEqual(int((np.abs(p - 0.5 * NOMINAL) < 1.0).sum()), 1)
        assert_strict_on_off(self, p, allowed_partial=1)

    def test_373_exact_fractional_energy_is_not_rounded(self):
        """The #373 regression (exact fractional energy) holds without rounding up:
        neither the floor nor the ceiling number of steps is delivered."""
        for hours in (2.5, 1.25, 0.75, 3.1):
            with self.subTest(hours=hours):
                opt = make()
                res = solve(opt, [hours])
                self.assertEqual(opt.optim_status, "Optimal")
                self.assertAlmostEqual(energy_wh(res), NOMINAL * hours, delta=1e-3)

    def test_sub_step_requirement_is_one_remainder_step(self):
        opt = make()
        res = solve(opt, [0.1])  # 0.2 of a 30 min step
        self.assertEqual(opt.optim_status, "Optimal")
        p = power(res)
        self.assertEqual(int((p > 1.0).sum()), 1)
        self.assertAlmostEqual(p.max(), 0.2 * NOMINAL, delta=1e-3)

    def test_whole_step_requirement_is_unchanged(self):
        """A whole number of steps keeps the strict ON/OFF model: no partial step."""
        for hours in (0.5, 1.5, 3.0):
            with self.subTest(hours=hours):
                opt = make()
                res = solve(opt, [hours])
                self.assertEqual(opt.optim_status, "Optimal")
                assert_strict_on_off(self, power(res), allowed_partial=0)
                self.assertAlmostEqual(energy_wh(res), NOMINAL * hours, delta=1e-3)

    def test_timestep_input_still_rounds_up_as_before(self):
        opt = make()
        res = solve(opt, [0], steps=[2.5])  # a fractional step count rounds up to 3
        self.assertEqual(opt.optim_status, "Optimal")
        assert_strict_on_off(self, power(res), allowed_partial=0)
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 3 * DT, delta=1e-3)

    def test_remainder_is_the_last_operating_step(self):
        """Prices that make the cheapest plan put the partial step first if it could
        go anywhere: the remainder still follows every full step, which keeps a
        rolling-MPC step counter exact."""
        prices = [0.30] * 3 + [0.08, 0.05, 0.06] + [0.30] * 6
        opt = make()
        res = solve(opt, [1.25], prices=prices)
        self.assertEqual(opt.optim_status, "Optimal")
        p = power(res)
        steps = np.flatnonzero(p > 1.0)
        self.assertAlmostEqual(p[steps[-1]], 0.5 * NOMINAL, delta=1e-3)
        np.testing.assert_allclose(p[steps[:-1]], NOMINAL, atol=1e-3)

    def test_completed_steps_leave_only_the_remainder(self):
        """2 of 2.5 steps completed: only the half step remains. 3 completed: nothing."""
        opt = make(def_current_operating_timesteps=[2])
        res = solve(opt, [1.25])
        self.assertEqual(opt.optim_status, "Optimal")
        p = power(res)
        self.assertEqual(int((p > 1.0).sum()), 1)
        self.assertAlmostEqual(p.max(), 0.5 * NOMINAL, delta=1e-3)

        opt = make(def_current_operating_timesteps=[3])
        res = solve(opt, [1.25])
        self.assertAlmostEqual(energy_wh(res), 0.0, delta=1e-3)

    def _rolling_execution(self, prices, hours=1.25, steps=6):
        """Replan after every executed step on a sliding price window, feeding back the
        completed-step counter exactly as a rolling MPC does."""
        executed, done, first_steps = 0.0, 0, []
        for s in range(steps):
            opt = make(def_current_operating_timesteps=[done])
            res = solve(opt, [hours], prices=prices[s : s + N])
            self.assertEqual(opt.optim_status, "Optimal")
            first = float(power(res)[0])
            first_steps.append(round(first))
            executed += first * DT
            done += 1 if first > 1.0 else 0
        return executed, first_steps

    def test_rolling_replan_delivers_exact_energy(self):
        """Prices where a free remainder would be placed FIRST (0.07 is the dearest of
        the three cheap steps): the executed total must still be exactly the request.
        Without the remainder-last rule the first executed step is the half step, the
        counter treats it as a whole step and only 3.0 of 3.75 kWh is delivered."""
        for prices in ([0.07, 0.05, 0.06] + [0.30] * 20, [0.05 + 0.01 * t for t in range(30)]):
            with self.subTest(prices=prices[:4]):
                executed, first_steps = self._rolling_execution(prices)
                self.assertAlmostEqual(executed, NOMINAL * 1.25, delta=1e-3, msg=first_steps)
                self.assertEqual(first_steps, [3000, 3000, 1500, 0, 0, 0])

    def test_single_constant_load_is_one_contiguous_run(self):
        prices = [0.3, 0.05] * 6
        opt = make(set_deferrable_load_single_constant=[True])
        res = solve(opt, [2.25], prices=prices)
        self.assertEqual(opt.optim_status, "Optimal")
        p = power(res)
        self.assertEqual(len(runs(p)), 1)
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 2.25, delta=1e-3)
        assert_strict_on_off(self, p, allowed_partial=1)
        self.assertAlmostEqual(p[runs(p)[0][1] - 1], 0.5 * NOMINAL, delta=1e-3)

    def test_minimum_on_time_counts_full_steps(self):
        """A run that contains the remainder step still has min-on full steps."""
        # 1.5 steps (one full step plus the remainder) cannot form a run of 2 full
        # steps: the strict solve must refuse it rather than count the remainder step
        # as a minimum on-time step.
        opt = make(def_minimum_on_time=[2])
        res = solve(opt, [0.75])
        self.assertEqual(opt.optim_status, "Optimal (Relaxed)")
        # The fallback must still deliver the requested energy, not drop the load.
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 0.75, delta=1e-3)

        opt = make(def_minimum_on_time=[2])
        res = solve(opt, [1.25], prices=[0.05, 0.3] * 6)
        self.assertEqual(opt.optim_status, "Optimal")
        p = power(res)
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 1.25, delta=1e-3)
        for start, end in runs(p):
            full = int((np.abs(p[start:end] - NOMINAL) < 1.0).sum())
            self.assertGreaterEqual(full, 2, p.round(0).tolist())

    def test_minimum_on_time_forbids_a_lone_pulse(self):
        """A requirement shorter than the minimum on-time cannot be met by a pulse:
        the strict solve is infeasible and the existing relaxed fallback handles it."""
        opt = make(def_minimum_on_time=[2])
        res = solve(opt, [0.25])
        self.assertEqual(opt.optim_status, "Optimal (Relaxed)")
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 0.25, delta=1e-3)

    def test_minimum_off_time_gap_is_kept(self):
        opt = make(def_minimum_off_time=[2])
        res = solve(opt, [1.25], prices=[0.05, 0.3] * 6)
        self.assertEqual(opt.optim_status, "Optimal")
        p = power(res)
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 1.25, delta=1e-3)
        found = runs(p)
        self.assertTrue(found, "requested energy must produce an active plan")
        for (_, end), (start, _) in zip(found, found[1:], strict=False):
            self.assertGreaterEqual(start - end, 2)

    def test_startup_count_includes_the_remainder_step(self):
        opt = make(set_deferrable_max_startups=[1])
        res = solve(opt, [1.25], prices=[0.05, 0.3] * 6)
        self.assertEqual(opt.optim_status, "Optimal")
        self.assertEqual(len(runs(power(res))), 1)

    def test_other_loads_keep_their_binary_constraints(self):
        opt = build_optimization(
            optim_overrides=dict(
                BASE,
                number_of_deferrable_loads=3,
                nominal_power_of_deferrable_loads=[3000.0, 2000.0, 1000.0],
                minimum_power_of_deferrable_loads=[0.0] * 3,
                treat_deferrable_load_as_semi_cont=[True, True, False],
                set_deferrable_load_single_constant=[False] * 3,
                set_deferrable_startup_penalty=[0.0] * 3,
                operating_hours_of_each_deferrable_load=[1.25, 1.0, 0.7],
                start_timesteps_of_each_deferrable_load=[0] * 3,
                end_timesteps_of_each_deferrable_load=[0] * 3,
            )
        )
        res = solve(opt, [1.25, 1.0, 0.7])
        self.assertEqual(opt.optim_status, "Optimal")
        assert_strict_on_off(self, power(res, 0), 3000.0, allowed_partial=1)
        assert_strict_on_off(self, power(res, 1), 2000.0, allowed_partial=0)
        # Continuous load: exact energy, free to modulate (not a semi-continuous one).
        self.assertAlmostEqual(energy_wh(res, 2), 1000.0 * 0.7, delta=1e-3)

    def test_cached_problem_follows_changing_hours(self):
        """Hours are a runtime input: the same problem object must follow whole ->
        fractional -> whole -> off without a stale remainder."""
        opt = make()
        for hours in (1.25, 1.5, 0.75, 1.25, 0.0, 1.0):
            with self.subTest(hours=hours):
                res = solve(opt, [hours])
                self.assertEqual(opt.optim_status, "Optimal")
                self.assertAlmostEqual(energy_wh(res), NOMINAL * hours, delta=1e-3)
                whole = abs(hours / DT - round(hours / DT)) < 1e-9
                assert_strict_on_off(self, power(res), allowed_partial=0 if whole else 1)

    def test_optional_load_with_max_cost_stays_all_or_nothing(self):
        """The cost cap is an all-or-nothing threshold on the cost of the whole plan.
        Cheapest plan: 3.75 kWh at 0.05 = 0.1875. A cap just below it leaves the load
        off, just above it schedules the full energy; nothing in between, and never a
        part of the energy."""
        for cap, expected in ((0.0001, 0.0), (0.1874, 0.0), (0.1876, 3750.0), (100.0, 3750.0)):
            with self.subTest(cap=cap):
                opt = make(deferrable_load_max_cost=[cap])
                res = solve(opt, [1.25])
                self.assertEqual(opt.optim_status, "Optimal")
                self.assertAlmostEqual(energy_wh(res), expected, delta=1e-3)

    def test_optional_load_threshold_uses_the_remainder_last_plan_cost(self):
        """Prices 0.07/0.05/0.06: a free remainder would sit on the dearest step (cost
        0.2175), the executed remainder-last plan costs 0.2250. The threshold is judged
        on the plan that is executed, and the load is never partial."""
        prices = [0.07, 0.05, 0.06] + [0.30] * 9
        for cap, expected in ((0.2174, 0.0), (0.22, 0.0), (0.2249, 0.0), (0.2251, 3750.0)):
            with self.subTest(cap=cap):
                opt = make(deferrable_load_max_cost=[cap])
                res = solve(opt, [1.25], prices=prices)
                self.assertEqual(opt.optim_status, "Optimal")
                self.assertAlmostEqual(energy_wh(res), expected, delta=1e-3)
                if expected:
                    p = power(res)
                    np.testing.assert_allclose(p[:3], [NOMINAL, NOMINAL, 0.5 * NOMINAL], atol=1e-3)

    def test_window_that_fits_the_ceiling_steps_is_feasible(self):
        opt = make()
        res = solve(opt, [1.25], start=[2], end=[5])
        self.assertEqual(opt.optim_status, "Optimal")
        p = power(res)
        self.assertTrue((p[:2] == 0).all() and (p[5:] == 0).all())
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 1.25, delta=1e-3)

    def test_window_shorter_than_the_energy_is_infeasible(self):
        """A 2-step window cannot hold 2.5 steps of energy at nominal power. As for a
        whole-step requirement that does not fit (1.5 h), the solve reports
        ``Infeasible`` and returns no plan: unchanged behaviour, no crash."""
        for hours in (1.25, 1.5):
            with self.subTest(hours=hours):
                opt = make()
                res = solve(opt, [hours], start=[2], end=[4])
                self.assertEqual(opt.optim_status, "Infeasible")
                self.assertNotIn("P_deferrable0", res.columns)

    def test_window_holding_one_step_and_a_remainder_is_feasible(self):
        """0.75 h = 1.5 steps fits a 2-step window as a full step plus the remainder."""
        opt = make()
        res = solve(opt, [0.75], start=[2], end=[4])
        self.assertEqual(opt.optim_status, "Optimal")
        p = power(res)
        self.assertTrue((p[:2] == 0).all() and (p[4:] == 0).all())
        np.testing.assert_allclose(p[2:4], [NOMINAL, 0.5 * NOMINAL], atol=1e-3)

    def test_shared_power_budget_uses_the_on_level(self):
        """A remainder step still switches the load to its full power, so a group
        budget is checked at the nominal power, not at the step average."""
        opt = build_optimization(
            optim_overrides=dict(
                BASE,
                number_of_deferrable_loads=2,
                nominal_power_of_deferrable_loads=[3000.0, 3000.0],
                minimum_power_of_deferrable_loads=[0.0] * 2,
                treat_deferrable_load_as_semi_cont=[True, True],
                set_deferrable_load_single_constant=[False] * 2,
                set_deferrable_startup_penalty=[0.0] * 2,
                operating_hours_of_each_deferrable_load=[1.25, 1.25],
                start_timesteps_of_each_deferrable_load=[0] * 2,
                end_timesteps_of_each_deferrable_load=[0] * 2,
                deferrable_load_groups=[
                    {
                        "names": ["deferrable0", "deferrable1"],
                        "max_power": 3500,
                        "mutual_exclusion": False,
                    }
                ],
            )
        )
        # Step 4 is strictly cheaper than the rest: sharing it would pay if the budget
        # were checked at the step average (1.5 + 1.5 kW) instead of the ON level.
        prices = [0.05] * 4 + [0.10] + [0.30] * (N - 5)
        res = solve(opt, [1.25, 1.25], prices=prices)
        self.assertEqual(opt.optim_status, "Optimal")
        both_on = (power(res, 0) > 1.0) & (power(res, 1) > 1.0)
        self.assertFalse(both_on.any(), "two loads share a 3.5 kW budget at 3 kW each")
        for k in (0, 1):
            self.assertAlmostEqual(energy_wh(res, k), NOMINAL * 1.25, delta=1e-3)

    def test_surplus_pv_cannot_be_chased_with_several_partial_steps(self):
        """1.5 kW of free PV surplus tempts a load into many half steps. Only ONE
        step may be partial: the rest must run at the nominal power."""
        pv = [2000.0] * 6 + [0.0] * (N - 6)  # 500 W base load -> 1.5 kW surplus
        opt = make()
        res = solve(opt, [1.25], prices=[0.30] * N, pv=pv)
        self.assertEqual(opt.optim_status, "Optimal")
        assert_strict_on_off(self, power(res), allowed_partial=1)
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 1.25, delta=1e-3)

    def test_minimum_power_applies_to_full_steps_only(self):
        """The remainder step averages below the load's minimum power but still runs
        at the nominal power while it is on: it must not make the solve infeasible."""
        opt = make(minimum_power_of_deferrable_loads=[2000.0])
        res = solve(opt, [1.25])
        self.assertEqual(opt.optim_status, "Optimal")
        assert_strict_on_off(self, power(res), allowed_partial=1, level=0.5 * NOMINAL)
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 1.25, delta=1e-3)

    def test_running_single_constant_load_finishes_its_block_with_the_remainder(self):
        opt = make(set_deferrable_load_single_constant=[True], def_current_state=[True])
        res = solve(opt, [1.25], prices=[0.30] * N)
        self.assertEqual(opt.optim_status, "Optimal")
        np.testing.assert_allclose(power(res)[:3], [NOMINAL, NOMINAL, 0.5 * NOMINAL], atol=1e-3)
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 1.25, delta=1e-3)

    def test_forced_minimum_on_steps_are_full_steps(self):
        """A run that still owes 3 minimum on-time steps cannot spend one of them on
        the remainder step: 2.5 steps of energy cannot fill them, so the strict solve
        is infeasible instead of publishing a run shorter than the minimum."""
        opt = make(def_minimum_on_time=[4], def_current_state=[True], def_current_on_timesteps=[1])
        res = solve(opt, [1.25])
        self.assertEqual(opt.optim_status, "Optimal (Relaxed)")
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 1.25, delta=1e-3)

        # Enough energy for the forced full steps plus the remainder: strict again.
        opt = make(def_minimum_on_time=[4], def_current_state=[True], def_current_on_timesteps=[1])
        res = solve(opt, [1.75])
        self.assertEqual(opt.optim_status, "Optimal")
        np.testing.assert_allclose(power(res)[:3], NOMINAL, atol=1e-3)
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 1.75, delta=1e-3)

    def test_forced_minimum_on_tail_ends_where_availability_blocks_the_load(self):
        """A still-owed minimum on-time tail is truncated at the first step the load
        cannot run (def_availability 0), and the tail's FULL-step floor with it:
        otherwise bin1[t] >= 1 meets a power ceiling of 0 and the strict solve is
        infeasible (negative check: the clamp is what keeps it Optimal)."""
        opt = make(def_minimum_on_time=[3], def_current_state=[True], def_current_on_timesteps=[1])
        # 2.25 h = 4.5 steps: the running load's one full step, then (the blocked step
        # having ended its forced run) a new run of 3 full steps plus the remainder.
        res = solve(opt, [2.25], def_availability=[[1, 0] + [1] * (N - 2)])
        self.assertEqual(opt.optim_status, "Optimal")
        p = power(res)
        self.assertAlmostEqual(p[1], 0.0, delta=1e-6)
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 2.25, delta=1e-3)
        self.assertEqual(float(opt.param_min_on_full_lb[0].value[1:].max()), 0.0)

    def test_partial_step_is_exactly_the_remainder_not_its_complement(self):
        """2.3 steps: full, full, 30 %. A surplus of exactly 70 % of the nominal power
        tempts a plan into a 70 % step (the complement) that is not an ON step."""
        pv = [2600.0] * 6 + [0.0] * (N - 6)  # 500 W base load -> 2.1 kW surplus
        opt = make()
        res = solve(opt, [1.15], prices=[0.30] * N, pv=pv)
        self.assertEqual(opt.optim_status, "Optimal")
        assert_strict_on_off(self, power(res), allowed_partial=1, level=0.3 * NOMINAL)
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 1.15, delta=1e-3)

    def test_current_power_forces_the_remainder_step_when_less_than_one_step_remains(self):
        """Documented def_current_power case: a running semi-continuous load with one
        remainder step of energy left is forced ON at t=0 AT THE REMAINDER'S AVERAGE
        power (not nominal, and not a full step), and the solve stays strict."""
        opt = make(def_current_power=[NOMINAL], def_current_operating_timesteps=[2])
        # t=0 is the DEAREST step: only the force-ON puts the remainder there.
        res = solve(opt, [1.25], prices=RISING[::-1])
        self.assertEqual(opt.optim_status, "Optimal")
        p = power(res)
        self.assertAlmostEqual(p[0], 0.5 * NOMINAL, delta=1e-3)
        np.testing.assert_allclose(p[1:], 0.0, atol=1e-6)
        self.assertAlmostEqual(energy_wh(res), 0.5 * DT * NOMINAL, delta=1e-3)

    def test_long_horizon_keeps_binaries_integral(self):
        """The remainder step stays ONE integral step on 96 and 288 step horizons
        (no spread of tiny binary values over many OFF steps), at several fractions
        including just above the resolution."""
        for n in (96, 288):
            prices = [0.05 + 0.002 * ((t * 7) % 23) for t in range(n)]
            for steps in (2 + 1.1e-5, 2.5, 1.5e-5):
                with self.subTest(n=n, steps=steps):
                    opt = build_optimization(optim_overrides=dict(BASE), opt_time_delta=int(n * DT))
                    res = solve(opt, [steps * DT], prices=prices, n=n)
                    self.assertEqual(opt.optim_status, "Optimal")
                    for name in ("p_def_bin1", "p_def_bin2"):
                        values = np.asarray(opt.vars[name][0].value, dtype=float)
                        np.testing.assert_allclose(values, np.rint(values), atol=1e-7)
                    self.assertEqual(self._decisions(opt), (int(np.floor(steps)), 1))
                    self.assertAlmostEqual(energy_wh(res), steps * DT * NOMINAL, delta=1e-3)

    def test_remainder_parameters_follow_each_call(self):
        opt = make()
        expected = {1.25: (1.0, 0.5 * NOMINAL), 1.5: (0.0, 0.0), 0.1: (1.0, 0.2 * NOMINAL)}
        for hours, (active, level) in expected.items():
            with self.subTest(hours=hours):
                solve(opt, [hours])
                self.assertEqual(opt.param_rem_active[0].value, active)
                self.assertAlmostEqual(opt.param_rem_level[0].value, level, places=6)
        solve(opt, [0.0])
        self.assertEqual(opt.param_rem_active[0].value, 0.0)
        self.assertEqual(opt.param_rem_level[0].value, 0.0)

    def test_remainder_above_solver_resolution_is_kept(self):
        """A remainder of 1e-4 steps (0.3 W of a 3 kW load) is a real, executable partial
        step: it is neither dropped nor rounded up, and the binary decisions show it."""
        for steps in (2.0001, 2.9999):
            with self.subTest(steps=steps):
                opt = make()
                res = solve(opt, [steps * DT])
                self.assertEqual(opt.optim_status, "Optimal")
                self.assertEqual(opt.param_rem_active[0].value, 1.0)
                expected = (steps - np.floor(steps)) * NOMINAL
                self.assertAlmostEqual(opt.param_rem_level[0].value, expected, delta=1e-6)
                self.assertAlmostEqual(energy_wh(res), steps * DT * NOMINAL, delta=1e-2)
                # Inspect the actual binary decisions, not only the energy.
                full = np.asarray(opt.vars["p_def_bin1"][0].value, dtype=float)
                on = np.asarray(opt.vars["p_def_bin2"][0].value, dtype=float)
                self.assertEqual(int(np.rint(full).sum()), int(np.floor(steps)))
                self.assertEqual(int(np.rint(on - full).sum()), 1)
                remainder_index = int(np.flatnonzero(on - full > 0.5)[0])
                self.assertGreater(remainder_index, int(np.flatnonzero(full > 0.5).max()))

    def test_sub_resolution_remainder_is_the_whole_step_count(self):
        """HiGHS accepts binaries within 1e-6 of an integer, so a remainder below that
        scale (0.0015 W here) cannot be told from integrality noise. Within
        FRACTIONAL_STEP_RESOLUTION of a whole number of steps the requirement, its
        energy target and the schedule are exactly that whole number of steps."""
        for steps, whole in ((2.0000005, 2), (2.9999995, 3), (2.000009, 2), (2.999991, 3)):
            with self.subTest(steps=steps):
                opt = make()
                res = solve(opt, [steps * DT])
                self.assertEqual(opt.optim_status, "Optimal")
                self.assertEqual(opt.param_rem_active[0].value, 0.0)
                self.assertAlmostEqual(opt.param_target_energy[0].value, whole * DT * NOMINAL)
                p = power(res)
                self.assertAlmostEqual(energy_wh(res), whole * DT * NOMINAL, delta=1e-3)
                self.assertEqual(int((np.abs(p - NOMINAL) < 1.0).sum()), whole)
                assert_strict_on_off(self, p, allowed_partial=0)

    def test_snap_never_turns_a_positive_requirement_into_none(self):
        from emhass.optimization import _snap_to_whole_steps

        self.assertIsNone(_snap_to_whole_steps(5e-6))
        self.assertIsNone(_snap_to_whole_steps(2.00002))
        self.assertEqual(_snap_to_whole_steps(2.0000099), 2)
        self.assertEqual(_snap_to_whole_steps(3.0), 3)

    def _decisions(self, opt):
        full = np.rint(np.asarray(opt.vars["p_def_bin1"][0].value, dtype=float))
        on = np.rint(np.asarray(opt.vars["p_def_bin2"][0].value, dtype=float))
        return int(full.sum()), int((on - full).sum())

    def test_snap_contract_on_both_sides_of_the_threshold(self):
        """Numerical contract (see FRACTIONAL_STEP_RESOLUTION): within 1e-5 steps of a whole
        number, in either direction, the requirement is that whole number, so the energy
        error is at most 1e-5 * nominal * dt (Wh) and 1e-5 * dt of run time. Beyond it the
        remainder step is kept and the energy is exact. Checked on the solver's binaries."""
        from emhass.optimization import FRACTIONAL_STEP_RESOLUTION

        for nominal in (1000.0, 22000.0):
            for base in (2, 3):
                for offset, snapped in ((5e-7, True), (9e-6, True), (1.1e-5, False), (1e-4, False)):
                    for sign in (1, -1):
                        steps = base + sign * offset
                        whole = round(steps)
                        with self.subTest(nominal=nominal, steps=steps):
                            opt = make(nominal_power_of_deferrable_loads=[nominal])
                            res = solve(opt, [steps * DT])
                            self.assertEqual(opt.optim_status, "Optimal")
                            full, remainder = self._decisions(opt)
                            requested = steps * DT * nominal
                            if snapped:
                                self.assertEqual(opt.param_rem_active[0].value, 0.0)
                                self.assertEqual((full, remainder), (whole, 0))
                                self.assertAlmostEqual(
                                    energy_wh(res), whole * DT * nominal, delta=1e-3
                                )
                                self.assertLessEqual(
                                    abs(energy_wh(res) - requested),
                                    FRACTIONAL_STEP_RESOLUTION * nominal * DT + 1e-3,
                                )
                            else:
                                self.assertEqual(opt.param_rem_active[0].value, 1.0)
                                self.assertEqual((full, remainder), (int(np.floor(steps)), 1))
                                self.assertAlmostEqual(energy_wh(res), requested, delta=1e-3)

    def test_requirement_below_one_step_is_never_snapped_to_nothing(self):
        """A positive requirement below one step keeps its own remainder step: it is
        neither dropped nor promoted to a full step, at any nominal power. One below the
        resolution (1e-5 steps) is raised to it, because a remainder of ~1e-8 steps or
        less made the strict solve numerically infeasible (22 kW at 1e-9 steps fell back
        to the relaxed solve); the error stays within 1e-5 * nominal * dt Wh. Run on
        HiGHS 1.10.0 and 1.16.0: 7 powers x 13 requests from 1e-14 to 2e-5 steps."""
        from emhass.optimization import FRACTIONAL_STEP_RESOLUTION

        for nominal in (100.0, 3000.0, 22000.0):
            for steps in (1e-12, 1e-9, 3e-9, 1e-6, 5e-6, 1e-5, 2e-5):
                with self.subTest(nominal=nominal, steps=steps):
                    opt = make(nominal_power_of_deferrable_loads=[nominal])
                    res = solve(opt, [steps * DT])
                    self.assertEqual(opt.optim_status, "Optimal")
                    effective = max(steps, FRACTIONAL_STEP_RESOLUTION)
                    self.assertEqual(opt.param_rem_active[0].value, 1.0)
                    self.assertAlmostEqual(
                        opt.param_rem_level[0].value, effective * nominal, delta=1e-9 * nominal
                    )
                    # No silent loss, no promotion to a (much larger) full step.
                    self.assertAlmostEqual(
                        energy_wh(res), effective * DT * nominal, delta=1e-6 * nominal * DT
                    )
                    self.assertLessEqual(
                        abs(energy_wh(res) - steps * DT * nominal),
                        FRACTIONAL_STEP_RESOLUTION * nominal * DT + 1e-9,
                    )
                    self.assertLess(float(power(res).max()), 1e-3 * nominal)
                    self.assertEqual(self._decisions(opt), (0, 1))

    def test_snap_is_limited_to_loads_with_a_remainder_step(self):
        """A continuous load keeps the exact energy target; only a semi-continuous
        load with a remainder step uses the resolution."""
        hours = (2 + 5e-6) * DT
        opt = make(treat_deferrable_load_as_semi_cont=[False])
        res = solve(opt, [hours])
        self.assertAlmostEqual(opt.param_target_energy[0].value, hours * NOMINAL, places=6)
        self.assertAlmostEqual(energy_wh(res), hours * NOMINAL, delta=1e-3)

    def test_required_steps_and_energy_agree_for_a_single_constant_load(self):
        for steps, required in ((2 + 5e-6, 2), (3 - 5e-6, 3), (2 + 2e-5, 3)):
            with self.subTest(steps=steps):
                opt = make(set_deferrable_load_single_constant=[True])
                res = solve(opt, [steps * DT])
                self.assertEqual(opt.optim_status, "Optimal")
                self.assertEqual(opt.param_required_timesteps[0].value, required)
                self.assertEqual(len(runs(power(res))), 1)
                self.assertEqual(int((power(res) > 1e-3).sum()), required)

    def test_rolling_accounting_with_a_snapped_requirement(self):
        opt = make(def_current_operating_timesteps=[1])
        res = solve(opt, [(3 - 5e-6) * DT])
        self.assertEqual(opt.optim_status, "Optimal")
        self.assertEqual(self._decisions(opt), (2, 0))
        self.assertAlmostEqual(energy_wh(res), 2 * DT * NOMINAL, delta=1e-3)

    def test_cached_problem_follows_a_snapped_requirement(self):
        opt = make()
        for steps, expected in ((2 + 5e-6, 2.0), (2.5, 2.5), (3 - 5e-6, 3.0), (0.0, 0.0)):
            with self.subTest(steps=steps):
                res = solve(opt, [steps * DT])
                self.assertEqual(opt.optim_status, "Optimal")
                self.assertAlmostEqual(energy_wh(res), expected * DT * NOMINAL, delta=1e-3)

    def test_minimum_on_time_with_a_snapped_requirement(self):
        opt = make(def_minimum_on_time=[2])
        res = solve(opt, [(3 - 5e-6) * DT], prices=[0.05, 0.3] * 6)
        self.assertEqual(opt.optim_status, "Optimal")
        self.assertEqual(self._decisions(opt), (3, 0))
        self.assertAlmostEqual(energy_wh(res), 3 * DT * NOMINAL, delta=1e-3)

    def test_relaxed_fallback_with_a_snapped_requirement(self):
        """The forced minimum-on tail cannot fit: the relaxed fallback still receives the
        snapped (whole-step) target."""
        opt = make(def_minimum_on_time=[4], def_current_state=[True], def_current_on_timesteps=[1])
        res = solve(opt, [(2 + 5e-6) * DT])
        self.assertEqual(opt.optim_status, "Optimal (Relaxed)")
        self.assertAlmostEqual(opt.param_target_energy[0].value, 2 * DT * NOMINAL, places=6)
        self.assertAlmostEqual(energy_wh(res), 2 * DT * NOMINAL, delta=1e-1)

    def test_group_power_limit_with_a_snapped_requirement(self):
        opt = build_optimization(
            optim_overrides=dict(
                BASE,
                number_of_deferrable_loads=2,
                nominal_power_of_deferrable_loads=[3000.0, 3000.0],
                minimum_power_of_deferrable_loads=[0.0] * 2,
                treat_deferrable_load_as_semi_cont=[True, True],
                set_deferrable_load_single_constant=[False] * 2,
                set_deferrable_startup_penalty=[0.0] * 2,
                operating_hours_of_each_deferrable_load=[1.0, 1.0],
                start_timesteps_of_each_deferrable_load=[0] * 2,
                end_timesteps_of_each_deferrable_load=[0] * 2,
                deferrable_load_groups=[
                    {
                        "names": ["deferrable0", "deferrable1"],
                        "max_power": 3500,
                        "mutual_exclusion": False,
                    }
                ],
            )
        )
        hours = (2 - 5e-6) * DT
        res = solve(opt, [hours, hours])
        self.assertEqual(opt.optim_status, "Optimal")
        total = power(res, 0) + power(res, 1)
        self.assertLessEqual(float(total.max()), 3500.0 + 1e-3)
        for k in (0, 1):
            self.assertAlmostEqual(energy_wh(res, k), 2 * DT * NOMINAL, delta=1e-3)

    def test_single_constant_at_the_resolution_boundary_stays_strict(self):
        """Hours and energy ratios can differ by an ulp exactly at the resolution; the
        required steps must still match the energy (3.7 kW, 2.500005 h = 5.00001 steps)."""
        for nominal, hours in ((3700.0, 2.500005), (3700.0, 3.000005), (7400.0, 2.500005)):
            with self.subTest(nominal=nominal, hours=hours):
                opt = make(
                    nominal_power_of_deferrable_loads=[nominal],
                    set_deferrable_load_single_constant=[True],
                )
                res = solve(opt, [hours])
                self.assertEqual(opt.optim_status, "Optimal")
                # 5.00001 and 6.00001 steps are inside the resolution (whole steps); the
                # independent oracle is the ON steps of the plan itself, which must be one
                # run of exactly the required steps.
                on_steps = int((power(res) > 1e-6).sum())
                self.assertEqual(opt.param_required_timesteps[0].value, on_steps)
                self.assertEqual(len(runs(power(res))), 1)

    def test_snap_removes_float_noise_for_a_remainder_load(self):
        from emhass.optimization import _snap_to_whole_steps

        self.assertEqual(_snap_to_whole_steps(249.00000000000003), 249)
        self.assertEqual(_snap_to_whole_steps(2.9999999999999996), 3)

    def test_loads_without_a_remainder_step_keep_the_previous_requirement(self):
        """No behavioural change for a load that cannot carry a remainder step: the
        required steps are the plain ceiling (an ulp above a whole number is one more
        step, exactly as on upstream) and the energy target is nominal * hours."""
        noise = float(np.nextafter(4.0, np.inf))
        for label, over in (
            ("continuous single-constant", {"treat_deferrable_load_as_semi_cont": [False]}),
            ("sequence", {"nominal_power_of_deferrable_loads": [[1000.0, 2000.0]]}),
        ):
            with self.subTest(load=label):
                opt = make(set_deferrable_load_single_constant=[True], **over)
                self.assertFalse(opt._has_remainder_step(0))
                hours = noise * DT
                self.assertEqual(opt._required_steps_for_hours(0, hours), 5)
                if label == "continuous single-constant":
                    solve(opt, [hours])
                    self.assertEqual(opt.param_required_timesteps[0].value, 5)
                    self.assertEqual(opt.param_target_energy[0].value, NOMINAL * hours)
                    self.assertEqual(opt.param_rem_active[0].value, 0.0)
                    self.assertEqual(opt.param_rem_level[0].value, 0.0)

    def test_window_check_uses_the_effective_requirement(self):
        """A requirement a hair above 4 steps is solved as 4 whole steps, so a 4-step
        window must not log 'Available timeframe is shorter' and must stay strict."""

        def window_warnings(opt, steps):
            captured = []

            class Capture(logging.Handler):
                def emit(self, record):
                    captured.append(record.getMessage())

            handler = Capture(level=logging.WARNING)
            opt.logger.addHandler(handler)
            try:
                res = solve(opt, [steps * DT], start=[0], end=[4])
            finally:
                opt.logger.removeHandler(handler)
            return res, [m for m in captured if "timeframe" in m]

        opt = make()
        res, warned = window_warnings(opt, 4 + 5e-6)
        self.assertEqual(warned, [])
        self.assertEqual(opt.optim_status, "Optimal")
        self.assertEqual(int((power(res) > 1.0).sum()), 4)
        # Beyond the resolution a fifth step is genuinely needed: the window check warns.
        _, warned = window_warnings(make(), 4 + 5e-5)
        self.assertGreaterEqual(len(warned), 1)

    def test_remainder_step_is_limited_to_plain_semi_continuous_loads(self):
        opt = make()
        self.assertTrue(opt._has_remainder_step(0))
        for conf in ({"thermal_config": {}}, {"thermal_battery": {}}):
            with self.subTest(conf=conf):
                opt.optim_conf["def_load_config"] = [conf]
                self.assertFalse(opt._has_remainder_step(0))
        opt.optim_conf["def_load_config"] = [{}]
        self.assertFalse(opt._has_remainder_step(0, shared_tank_membership={0: 0}))
        opt.optim_conf["nominal_power_of_deferrable_loads"] = [[1000.0, 2000.0]]
        self.assertFalse(opt._has_remainder_step(0))
        opt = make(treat_deferrable_load_as_semi_cont=[False])
        self.assertFalse(opt._has_remainder_step(0))

    def test_float_representation_noise_keeps_whole_steps(self):
        """An exact whole-step requirement with IEEE-754 roundoff must not
        become a spurious extra ON timestep or remainder."""
        for steps in (np.nextafter(2.0, np.inf), np.nextafter(3.0, -np.inf)):
            with self.subTest(steps=steps):
                opt = make()
                res = solve(opt, [float(steps * DT)])
                self.assertEqual(opt.optim_status, "Optimal")
                self.assertEqual(opt.param_rem_active[0].value, 0.0)
                self.assertAlmostEqual(energy_wh(res), round(steps) * DT * NOMINAL, delta=1e-3)
                self.assertEqual(int((power(res) > 1.0).sum()), round(steps))

    def test_on_state_covers_every_powered_step(self):
        """Startups, minimum on/off time and mutual exclusion read the ON state, so a
        remainder step must be ON and a step without power must be OFF."""
        opt = make()
        res = solve(opt, [1.25], prices=[0.05, 0.3] * 6)
        self.assertEqual(opt.optim_status, "Optimal")
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 1.25, delta=1e-3)
        on_state = opt.vars["p_def_bin2"][0].value > 0.5
        np.testing.assert_array_equal(on_state, power(res) > 1.0)

    def test_remainder_step_reads_as_variable_in_published_states(self):
        """The command-state sensor labels the optimizer's own remainder step
        ``variable`` (not ``on``): the label is derived from the planned power."""
        opt = make()
        res = solve(opt, [1.25])
        self.assertEqual(opt.optim_status, "Optimal")
        labels = [_deferrable_power_to_state(float(v), NOMINAL) for v in power(res)]
        self.assertEqual(labels.count("on"), 2)
        self.assertEqual(labels.count("variable"), 1)
        last_active = max(i for i, label in enumerate(labels) if label != "off")
        self.assertEqual(labels.index("variable"), last_active)

    def test_remainder_near_the_state_thresholds(self):
        """The command-state sensor keeps its generic bands: a remainder at or below
        1 % of the nominal power reads ``off`` and one at or above 99 % reads ``on``;
        in between it reads ``variable``. The plan itself stays exact and strict. The
        label can differ from the plan by at most 1 % of one step's energy."""
        expected = {0.005: "off", 0.009: "off", 0.011: "variable", 0.989: "variable", 0.991: "on"}
        for fraction, label in expected.items():
            with self.subTest(fraction=fraction):
                hours = (2 + fraction) * DT
                opt = make()
                res = solve(opt, [hours])
                self.assertEqual(opt.optim_status, "Optimal")
                self.assertAlmostEqual(energy_wh(res), NOMINAL * hours, delta=1e-3)
                partial = [v for v in power(res) if v > 1e-6 and abs(v - NOMINAL) > 1.0]
                if fraction >= 0.991:
                    partial = [v for v in power(res) if v > 1e-6][-1:]
                self.assertEqual(_deferrable_power_to_state(partial[0], NOMINAL), label)
                if label != "variable":
                    implied = NOMINAL if label == "on" else 0.0
                    self.assertLessEqual(abs(partial[0] - implied), 0.01 * NOMINAL + 1e-6)

    def test_raising_the_hours_after_the_remainder_ran_is_short_by_the_unused_part(self):
        """Documented limitation of the completed-step counter: the remainder step is
        counted as one whole step, so raising the requirement afterwards leaves the
        unused part of that step ((1 - r) * nominal * step) undelivered."""
        for new_hours in (1.75, 2.0, 3.0):
            with self.subTest(new_hours=new_hours):
                executed, done = 0.0, 0
                for i in range(6):
                    hours = 1.25 if i < 3 else new_hours
                    opt = make(def_current_operating_timesteps=[done])
                    res = solve(opt, [hours], prices=RISING)
                    first = float(power(res)[0])
                    executed += first * DT
                    done += 1 if first > 1.0 else 0
                shortfall = NOMINAL * new_hours - executed
                self.assertAlmostEqual(shortfall, 0.5 * NOMINAL * DT, delta=1e-3)

    def test_changed_requirement_is_exact_with_remaining_hours_accounting(self):
        """The supported way to change the requirement after a remainder step ran: the
        caller tracks the delivered energy and passes the REMAINING hours (new total -
        delivered / nominal power) with the step counter left at 0. A fresh optimizer
        per tick (a restart) holds no state, and the energy stays exact."""
        for single in (False, True):
            for new_hours in (1.75, 2.0, 2.6, 3.0):
                with self.subTest(single=single, new_hours=new_hours):
                    delivered, hours = 0.0, 1.25
                    for i in range(9):
                        if i == 3:  # the remainder step ran at i == 2
                            hours = new_hours
                        remaining = max(0.0, hours - delivered / NOMINAL)
                        opt = make(set_deferrable_load_single_constant=[single])
                        res = solve(opt, [0.0 if remaining < 1e-9 else remaining], prices=RISING)
                        self.assertEqual(opt.optim_status, "Optimal")
                        delivered += float(power(res)[0]) * DT
                    self.assertAlmostEqual(delivered, NOMINAL * new_hours, delta=1e-3)

    def test_label_driven_dispatch_deviates_by_at_most_one_percent_of_a_step(self):
        """An automation that executes the state label (off = nothing, on = the whole
        step at nominal power) differs from the plan by at most 1 % of one step's
        energy, and a label-driven step counter never counts an ``off`` remainder."""
        for fraction in (0.002, 0.009, 0.991, 0.998):
            with self.subTest(fraction=fraction):
                hours = (2 + fraction) * DT
                opt = make()
                res = solve(opt, [hours])
                p = power(res)
                labels = [_deferrable_power_to_state(v, NOMINAL) for v in p]
                self.assertNotIn("variable", labels)
                dispatched = sum(NOMINAL * DT for label in labels if label == "on")
                self.assertLessEqual(abs(dispatched - NOMINAL * hours), 0.01 * NOMINAL * DT + 1e-6)
                counted = sum(label != "off" for label in labels)
                if fraction < 0.5:
                    self.assertEqual(counted, 2)  # the off remainder is not counted
                    opt = make(def_current_operating_timesteps=[counted])
                    again = power(solve(opt, [hours]))
                    self.assertGreater(again.max(), 1.0)  # replanned every tick
                else:
                    self.assertEqual(counted, 3)

    def test_random_fractions_keep_exact_energy_and_one_partial_step(self):
        rng = np.random.default_rng(11)
        for _ in range(12):
            hours = float(np.round(rng.uniform(0.1, 4.0), 2))
            prices = list(np.round(rng.uniform(0.03, 0.4, N), 3))
            single = bool(rng.integers(0, 2))
            with self.subTest(hours=hours, single=single):
                opt = make(set_deferrable_load_single_constant=[single])
                res = solve(opt, [hours], prices=prices)
                self.assertEqual(opt.optim_status, "Optimal")
                p = power(res)
                self.assertAlmostEqual(energy_wh(res), NOMINAL * hours, delta=1e-2)
                steps_of_energy = hours / DT
                level = (steps_of_energy - int(steps_of_energy + 1e-9)) * NOMINAL
                assert_strict_on_off(self, p, allowed_partial=1, level=level)
                if single:
                    self.assertEqual(len(runs(p)), 1)


class TestRemainderSolverOptions(unittest.TestCase):
    """Gurobi/CPLEX default to a 1e-5 integrality tolerance, equal to the remainder
    resolution; it is tightened to 1e-7 only for them. No commercial solver is run."""

    def test_gurobi_and_cplex_get_a_tighter_integrality_tolerance(self):
        from emhass.optimization import REMAINDER_INTEGRALITY_TOL, Optimization

        base = {"verbose": False, "cplex_params": {"timelimit": 5}}
        gurobi = Optimization._remainder_solver_opts(cp.GUROBI, base)
        self.assertEqual(gurobi["IntFeasTol"], REMAINDER_INTEGRALITY_TOL)
        cplex = Optimization._remainder_solver_opts(cp.CPLEX, base)
        self.assertEqual(cplex["cplex_params"]["mip.tolerances.integrality"], 1e-7)
        self.assertEqual(cplex["cplex_params"]["timelimit"], 5)
        self.assertLess(REMAINDER_INTEGRALITY_TOL, 1e-5)
        # HiGHS is untouched and the input is never mutated.
        self.assertEqual(Optimization._remainder_solver_opts(cp.HIGHS, base), base)
        self.assertNotIn("IntFeasTol", base)
        self.assertNotIn("mip.tolerances.integrality", base["cplex_params"])

    def test_tolerance_is_only_applied_while_a_remainder_is_active(self):
        from unittest import mock

        for hours, expected in ((1.25, 1), (1.0, 0)):
            with self.subTest(hours=hours):
                opt = make()
                with mock.patch.object(
                    type(opt),
                    "_remainder_solver_opts",
                    side_effect=lambda s, o: o,
                ) as spy:
                    solve(opt, [hours])
                self.assertEqual(spy.call_count, expected)


class TestRemainderSolverOptionsOnReusedModel(unittest.TestCase):
    """The solver options that reach ``Problem.solve`` follow each call's requirement
    on one cached optimiser: fractional -> whole -> fractional (no stale tolerance).
    A fake Gurobi is selected only to inspect the options; HiGHS does the solving."""

    def test_integrality_option_follows_the_requirement_on_one_instance(self):
        import os
        from unittest import mock

        opt = make()
        captured = []
        real_solve = cp.Problem.solve

        def spy(problem, *args, **kwargs):
            captured.append(dict(kwargs))
            return real_solve(problem, solver=cp.HIGHS, verbose=False)

        installed = [*cp.installed_solvers(), "GUROBI"]
        with (
            mock.patch.dict(os.environ, {"LP_SOLVER": "GUROBI"}),
            mock.patch.object(cp, "installed_solvers", return_value=installed),
            mock.patch.object(cp.Problem, "solve", spy),
        ):
            for hours, fractional in ((1.25, True), (1.0, False), (1.25, True), (1.5, False)):
                with self.subTest(hours=hours):
                    captured.clear()
                    res = solve(opt, [hours])
                    self.assertEqual(opt.optim_status, "Optimal")
                    self.assertAlmostEqual(energy_wh(res), NOMINAL * hours, delta=1e-3)
                    self.assertTrue(captured)
                    self.assertEqual(captured[0].get("IntFeasTol"), 1e-7 if fractional else None)
                    self.assertIn("TimeLimit", captured[0])


if __name__ == "__main__":
    unittest.main()
