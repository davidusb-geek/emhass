#!/usr/bin/env python
"""Semi-continuous deferrable loads with a fractional operating requirement (#1105).

``operating_hours_of_each_deferrable_load`` that is not a whole number of
optimization timesteps used to make a strict ON/OFF load infeasible, so the whole
solve fell back to the relaxed LP (``Optimal (Relaxed)``) and every ON/OFF and
single-constant constraint of every load was dropped. The load now runs whole
steps at its nominal power plus exactly one remainder step whose interval-average
power is ``r * nominal``: the energy stays exact and the solve stays a strict MILP.

Synthetic and self-contained, built on the multi-battery test helper.
"""

import unittest

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
        res = solve(opt, [0], steps=[3])
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
        solve(opt, [0.25])
        self.assertEqual(opt.optim_status, "Optimal (Relaxed)")

    def test_minimum_off_time_gap_is_kept(self):
        opt = make(def_minimum_off_time=[2])
        res = solve(opt, [1.25], prices=[0.05, 0.3] * 6)
        self.assertEqual(opt.optim_status, "Optimal")
        p = power(res)
        found = runs(p)
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
        solve(opt, [1.25])
        self.assertEqual(opt.optim_status, "Optimal (Relaxed)")

        # Enough energy for the forced full steps plus the remainder: strict again.
        opt = make(def_minimum_on_time=[4], def_current_state=[True], def_current_on_timesteps=[1])
        res = solve(opt, [1.75])
        self.assertEqual(opt.optim_status, "Optimal")
        np.testing.assert_allclose(power(res)[:3], NOMINAL, atol=1e-3)
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 1.75, delta=1e-3)

    def test_partial_step_is_exactly_the_remainder_not_its_complement(self):
        """2.3 steps: full, full, 30 %. A surplus of exactly 70 % of the nominal power
        tempts a plan into a 70 % step (the complement) that is not an ON step."""
        pv = [2600.0] * 6 + [0.0] * (N - 6)  # 500 W base load -> 2.1 kW surplus
        opt = make()
        res = solve(opt, [1.15], prices=[0.30] * N, pv=pv)
        self.assertEqual(opt.optim_status, "Optimal")
        assert_strict_on_off(self, power(res), allowed_partial=1, level=0.3 * NOMINAL)
        self.assertAlmostEqual(energy_wh(res), NOMINAL * 1.15, delta=1e-3)

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

    def test_on_state_covers_every_powered_step(self):
        """Startups, minimum on/off time and mutual exclusion read the ON state, so a
        remainder step must be ON and a step without power must be OFF."""
        opt = make()
        res = solve(opt, [1.25], prices=[0.05, 0.3] * 6)
        on_state = opt.vars["p_def_bin2"][0].value > 0.5
        np.testing.assert_array_equal(on_state, power(res) > 1.0)

    def test_remainder_step_reads_as_variable_in_published_states(self):
        """The existing opt-in command-state sensor already labels an in-between
        power ``variable``, so the remainder step is never published as ``on``."""
        self.assertEqual(_deferrable_power_to_state(0.5 * NOMINAL, NOMINAL), "variable")
        self.assertEqual(_deferrable_power_to_state(NOMINAL, NOMINAL), "on")

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


if __name__ == "__main__":
    unittest.main()
