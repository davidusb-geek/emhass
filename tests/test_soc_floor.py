#!/usr/bin/env python
"""Tests for the soft per-step ``soc_floor`` runtime input of naive-mpc-optim (#1348).

``soc_floor`` asks for a minimum SoC per step. It is soft: a per-step shortfall
below it is priced in the objective, so it holds whenever the battery can hold it
and never makes the problem infeasible. A run that starts under the floor keeps that
shortfall instead of charging back up: it holds at ``soc_init`` (no discharge) and a
later rise is built by its own increase. A floor out of reach is lowered to what the
battery can reach by the start of its step. Without a floor the problem is the one
built without the feature.

Synthetic and self-contained, built on the multi-battery test helper.
"""

import inspect
import unittest
from unittest import mock

import cvxpy as cp
import numpy as np
import pandas as pd

import tests.test_multi_battery_optimization as multi_battery
from emhass import optimization
from tests.test_battery_availability import HOUSE_AND_EV
from tests.test_multi_battery_optimization import VALID_OPTIMAL_STATUSES
from tests.test_multi_battery_optimization import build_optimization as build_multi_battery

N = 12  # 6 h at 30 min
CAP = 10000.0  # build_optimization default battery: 10 kWh, 5 kW, soc 0.3..0.8, eff 1
# A real charge efficiency, so a cost-neutral charge and discharge at one price
# cannot tie with holding.
EFF_95 = {"battery_charge_efficiency": 0.95, "battery_discharge_efficiency": 0.95}

# The keys this file exercises. Older sources do not take them; the helpers
# drop them there so the behavioural assertions are what fails, not a TypeError.
NEW_KEYS = ("soc_floor", "battery_discharge_availability")

# Dearest first: without a floor the battery empties itself in steps 0-5.
RESERVE_PRICES = [0.50] * 6 + [0.40] * 2 + [0.20] * 4

ONE_SEMI_CONT_LOAD = {
    "number_of_deferrable_loads": 1,
    "nominal_power_of_deferrable_loads": [300.0],
    "minimum_power_of_deferrable_loads": [0.0],
    "treat_deferrable_load_as_semi_cont": [True],
    "set_deferrable_load_single_constant": [False],
    "set_deferrable_startup_penalty": [0.0],
    "operating_hours_of_each_deferrable_load": [2],
    "start_timesteps_of_each_deferrable_load": [0],
    "end_timesteps_of_each_deferrable_load": [0],
}


def build_optimization(optim_overrides=None, costfun="profit", **kwargs):
    """The multi-battery helper sized to the N-step test horizon (no resize)."""
    optim = {"delta_forecast_daily": pd.Timedelta(hours=N // 2)}
    optim.update(optim_overrides or {})
    if costfun == "profit":
        return build_multi_battery(optim_overrides=optim, **kwargs)
    real = multi_battery.Optimization

    def with_costfun(*args, **kw):
        # The helper passes Optimization's arguments by position, costfun sixth.
        costfun_index = list(inspect.signature(real).parameters).index("costfun")
        args = list(args)
        args[costfun_index] = costfun
        return real(*args, **kw)

    with mock.patch.object(multi_battery, "Optimization", with_costfun):
        return build_multi_battery(optim_overrides=optim, **kwargs)


def scenario(n=N, load_cost=None, prod_price=None, load=1500.0):
    index = pd.date_range("2026-11-01", periods=n, freq="30min", tz="Europe/Brussels")
    p_pv = pd.Series([0.0] * n, index=index)
    p_load = pd.Series(load if isinstance(load, list) else [load] * n, index=index)
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = load_cost if load_cost is not None else [0.30] * n
    df_input["unit_prod_price"] = prod_price if prod_price is not None else [0.0] * n
    return p_pv, p_load, df_input


def run(opt, n=N, load_cost=None, prod_price=None, load=1500.0, **kwargs):
    """One naive-MPC run of ``opt``; extra kwargs go to perform_naive_mpc_optim."""
    accepted = inspect.signature(opt.perform_naive_mpc_optim).parameters
    for key in NEW_KEYS:
        if key in kwargs and key not in accepted:
            kwargs.pop(key)
    p_pv, p_load, df_input = scenario(n, load_cost, prod_price, load)
    kwargs.setdefault("soc_init", 0.7)
    kwargs.setdefault("soc_final", 0.3)
    loads = opt.optim_conf["number_of_deferrable_loads"]
    hours = opt.optim_conf["operating_hours_of_each_deferrable_load"]
    return opt.perform_naive_mpc_optim(
        df_input,
        p_pv,
        p_load,
        n,
        def_total_hours=[0] * loads,
        def_total_timestep=[int(h * 2) for h in hours[:loads]],
        def_start_timestep=[0] * loads,
        def_end_timestep=[0] * loads,
        **kwargs,
    )


def run_counting_solves(opt, **kwargs):
    """``run`` that also returns how many times a cvxpy problem was solved."""
    original = cp.Problem.solve
    calls = []

    def counting(problem, *args, **kw):
        calls.append(problem)
        return original(problem, *args, **kw)

    with mock.patch.object(cp.Problem, "solve", counting):
        res = run(opt, **kwargs)
    return res, len(calls)


def floor_of(opt, k=0):
    """The floor (Wh) the last run built for battery ``k``; NaN on a source
    without the soft floor, so a value assertion fails there, not a lookup."""
    params = getattr(opt, "param_soc_floor", None)
    if params is None:
        return np.full(opt.num_timesteps, np.nan)
    return params[k].value


def has_floor_term(opt, k=0):
    shortfall = opt.vars.get("soc_floor_shortfall")
    return shortfall is not None and shortfall[k] is not None


def warnings_of(logs):
    return [line for line in logs.output if "WARNING" in line]


def missed_floor_warnings(logs):
    return [line for line in warnings_of(logs) if "stays under the soc_floor" in line]


def problem_signature(opt):
    """What the built problem is made of: constraint count, every variable's name
    and shape, and every Parameter's name and value."""
    variables = sorted((v.name(), v.shape) for v in opt.prob.variables())
    params = {p.name(): np.asarray(p.value).copy() for p in opt.prob.parameters()}
    return len(opt.prob.constraints), variables, params


class TestSocFloorDefault(unittest.TestCase):
    """Without an effective floor the problem and the plan are exactly today's."""

    def _assert_identical(self, a, res_a, b, res_b):
        n_a, vars_a, params_a = problem_signature(a)
        n_b, vars_b, params_b = problem_signature(b)
        self.assertEqual(n_a, n_b)
        self.assertEqual(vars_a, vars_b)
        self.assertEqual(sorted(params_a), sorted(params_b))
        for name, value in params_a.items():
            np.testing.assert_array_equal(value, params_b[name], err_msg=name)
        self.assertEqual(a.prob.value, b.prob.value)
        self.assertEqual(list(res_a.columns), list(res_b.columns))
        for name in res_a.columns:
            np.testing.assert_array_equal(
                res_a[name].to_numpy(), res_b[name].to_numpy(), err_msg=name
            )
        self.assertNotIn("soc_floor_shortfall", b.vars)

    def test_absent_null_and_ineffective_floors_are_a_no_op(self):
        a = build_optimization()
        res_a, solves_a = run_counting_solves(a, load_cost=RESERVE_PRICES)
        self.assertIn(a.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertEqual(solves_a, 1)
        for floor in (None, [None] * N, [float("nan")] * N, [0.3] * N, [0.1] * N, []):
            with self.subTest(soc_floor=floor):
                b = build_optimization()
                res_b, solves_b = run_counting_solves(b, load_cost=RESERVE_PRICES, soc_floor=floor)
                self.assertEqual(solves_b, 1)
                self._assert_identical(a, res_a, b, res_b)

    def test_floor_only_where_the_battery_is_unavailable_is_a_no_op(self):
        avail = [1] * 4 + [0] * (N - 4)
        a = build_optimization()
        res_a = run(a, load_cost=RESERVE_PRICES, battery_availability=avail)
        b = build_optimization()
        res_b = run(
            b,
            load_cost=RESERVE_PRICES,
            battery_availability=avail,
            soc_floor=[None] * 4 + [0.7] * (N - 4),
        )
        self._assert_identical(a, res_a, b, res_b)

    def test_hold_under_soc_min_is_a_no_op(self):
        """soc_init under soc_min: the hold caps the floor at soc_init, which is at
        or below soc_min, so there is nothing to add and the soc_min recovery runs
        as before."""
        a = build_optimization()
        res_a = run(a, soc_init=0.2, soc_final=0.3)
        b = build_optimization()
        res_b = run(b, soc_init=0.2, soc_final=0.3, soc_floor=[0.5] * N)
        self._assert_identical(a, res_a, b, res_b)
        self.assertEqual(a.param_soc_low_required[0].value, b.param_soc_low_required[0].value)

    def test_dayahead_after_a_floored_mpc_run_has_no_floor(self):
        fresh = build_optimization()
        p_pv, p_load, df_input = scenario(load_cost=RESERVE_PRICES)
        res_fresh = fresh.perform_dayahead_forecast_optim(
            df_input, p_pv, p_load, soc_init=0.7, soc_final=0.3
        )
        opt = build_optimization()
        run(opt, load_cost=RESERVE_PRICES, soc_floor=[0.7] * 6)
        res = opt.perform_dayahead_forecast_optim(
            df_input, p_pv, p_load, soc_init=0.7, soc_final=0.3
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        self._assert_identical(fresh, res_fresh, opt, res)


class TestSocFloorCache(unittest.TestCase):
    """One cached Optimization reused across MPC ticks builds the right problem:
    the floor term is there exactly when the run has a floor."""

    FLOOR = [0.7] * 6 + [None] * 6

    def test_floor_added_dropped_and_added_again_on_one_problem(self):
        fresh = build_optimization()
        res_fresh = run(fresh, load_cost=RESERVE_PRICES)
        fresh_floor = build_optimization()
        res_fresh_floor = run(fresh_floor, load_cost=RESERVE_PRICES, soc_floor=self.FLOOR)
        # Counterfactual: without the floor the battery drains in steps 0-5.
        self.assertLess(res_fresh["SOC_opt"].to_numpy()[:6].min(), 0.7 - 0.1)

        opt = build_optimization()
        results = []
        for floor in (None, self.FLOOR, None, self.FLOOR):
            res = run(opt, load_cost=RESERVE_PRICES, soc_floor=floor)
            self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
            self.assertEqual(has_floor_term(opt), floor is not None)
            self.assertEqual(
                problem_signature(opt)[0],
                problem_signature(fresh_floor if floor else fresh)[0],
            )
            results.append(res)
        self.assertGreaterEqual(results[1]["SOC_opt"].to_numpy()[:6].min(), 0.7 - 1e-4)
        for res_a, res_b in ((results[0], res_fresh), (results[2], res_fresh)):
            np.testing.assert_array_equal(res_a["P_batt"].to_numpy(), res_b["P_batt"].to_numpy())
        for res_a in (results[1], results[3]):
            np.testing.assert_allclose(res_a["P_batt"], res_fresh_floor["P_batt"], atol=1e-6)

    def test_a_new_floor_value_reuses_the_cached_problem(self):
        opt = build_optimization()
        run(opt, load_cost=RESERVE_PRICES, soc_floor=self.FLOOR)
        prob = opt.prob
        res = run(opt, load_cost=RESERVE_PRICES, soc_floor=[0.6] * 6 + [None] * 6)
        self.assertIs(opt.prob, prob)
        self.assertTrue(opt.prob.is_dpp())
        soc = res["SOC_opt"].to_numpy()
        self.assertGreaterEqual(soc[:6].min(), 0.6 - 1e-4)
        self.assertLess(soc[:6].min(), 0.7 - 1e-3)

    def test_per_battery_presence_rebuilds(self):
        opt = build_optimization(plant_overrides=HOUSE_AND_EV)
        soc = [0.5, 0.5]
        run(opt, soc_init=soc, soc_final=soc, soc_floor=[None, [0.6] * N])
        self.assertFalse(has_floor_term(opt, 0))
        self.assertTrue(has_floor_term(opt, 1))
        run(opt, soc_init=soc, soc_final=soc, soc_floor=[[0.6] * N, None])
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertTrue(has_floor_term(opt, 0))
        self.assertFalse(has_floor_term(opt, 1))

    def test_floor_follows_a_horizon_resize(self):
        """The cached problem was built for 10 steps; a 12-step run resizes it
        and the floor is built for the new horizon."""
        opt = build_multi_battery()
        self.assertEqual(opt.num_timesteps, 10)
        res = run(opt, load_cost=RESERVE_PRICES, soc_floor=[0.7] * 6)
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertEqual(floor_of(opt).shape, (N,))
        self.assertGreaterEqual(res["SOC_opt"].to_numpy()[:6].min(), 0.7 - 1e-4)


class TestSocFloorHolds(unittest.TestCase):
    def test_floor_holds_over_the_window_and_frees_the_battery_after(self):
        # Counterfactual: without the floor the battery is drained inside the window.
        free = build_optimization()
        soc_free = run(free, load_cost=RESERVE_PRICES)["SOC_opt"].to_numpy()
        self.assertLess(soc_free[:6].min(), 0.7 - 0.1)

        opt = build_optimization()
        res = run(opt, load_cost=RESERVE_PRICES, soc_floor=[0.7] * 6 + [None] * 6)
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        soc = res["SOC_opt"].to_numpy()
        self.assertGreaterEqual(soc[:6].min(), 0.7 - 1e-4)
        # free after the window: the reserve is spent in the cheaper steps
        self.assertLessEqual(soc[-1], 0.3 + 1e-3)

    def test_a_run_under_the_reserve_holds_and_does_not_charge_back_up(self):
        """soc_init 0.5 under a 0.7 floor at a price peak: the battery stops
        discharging and is not charged back up (#1348: hold, not climb)."""
        # Counterfactual 1: without the floor it drains in the dear steps.
        free = build_optimization(plant_overrides=EFF_95)
        soc_free = run(free, soc_init=0.5, load_cost=RESERVE_PRICES)["SOC_opt"].to_numpy()
        self.assertLess(soc_free[:6].min(), 0.5 - 0.1)
        # Counterfactual 2: a floor the plan can reach by the start of step 1 makes
        # it charge at the peak, which is what a climb back would do.
        climb = build_optimization(plant_overrides=EFF_95)
        res_climb = run(climb, soc_init=0.5, load_cost=RESERVE_PRICES, soc_floor=[None] + [0.7] * 5)
        self.assertLess(res_climb["P_batt"].to_numpy()[:2].min(), -1.0)

        opt = build_optimization(plant_overrides=EFF_95)
        with self.assertLogs(opt.logger, level="INFO") as logs:
            res, solves = run_counting_solves(
                opt, soc_init=0.5, load_cost=RESERVE_PRICES, soc_floor=[0.7] * 6
            )
        self.assertEqual(opt.optim_status, "Optimal")
        self.assertEqual(solves, 1)
        soc = res["SOC_opt"].to_numpy()
        np.testing.assert_allclose(soc[:6], 0.5, atol=1e-4)
        self.assertGreaterEqual(res["P_batt"].to_numpy()[:6].min(), -1e-6)
        np.testing.assert_allclose(floor_of(opt)[:6], 0.5 * CAP)
        self.assertTrue(any("held at soc_init" in line for line in logs.output), logs.output)
        self.assertEqual(missed_floor_warnings(logs), [])

    def test_floor_rule_cases(self):
        """The shortfall a run starts with is carried, never made up: a rise is
        built by its own increase, a lower value or a dip never lowers the floor
        under the held level, and only a step with no floor gives the plan time to
        build more. Each floor also binds the start of its step, so it is set on
        the step before it too (over a step where the battery is unavailable).
        5 kW at efficiency 1 adds 2500 Wh per 30-minute step."""
        hold = 0.5 * CAP
        unavailable_at_2 = {"battery_availability": [1, 1, 0] + [1] * 9}
        cases = {
            "flat": ([0.7] * N, {}, [hold] * N),
            "gap gives one step to build": (
                [0.7, 0.7, None] + [0.7] * 9,
                {},
                [hold, hold] + [7000] * 10,
            ),
            "dip to soc_init keeps the hold": ([0.7, 0.7, 0.5] + [0.7] * 9, {}, [hold] * N),
            "one-step dip below soc_init": ([0.7, 0.7, 0.4] + [0.7] * 9, {}, [hold] * N),
            "two-step dip below soc_init": (
                [0.7, 0.7, 0.4, 0.4] + [0.7] * 8,
                {},
                [hold, hold, 4000] + [hold] * 9,
            ),
            "rise built by its increase": ([0.6, 0.6] + [0.7] * 10, {}, [hold] + [6000] * 11),
            "one-step rise": ([0.6] + [0.61] * 11, {}, [5100] * N),
            "float noise": (
                [0.6, 0.6000001] + [0.6] * 10,
                {},
                [hold + 0.001, hold + 0.001] + [hold] * 10,
            ),
            "up-ramp": (
                [0.6 + 0.01 * t for t in range(N)],
                {},
                [hold + 100 * min(t + 1, N - 1) for t in range(N)],
            ),
            "lower but still above soc_init": ([0.7, 0.6] + [0.7] * 10, {}, [hold] * N),
            "window one step ahead": ([None] + [0.8] * 11, {}, [7500] * N),
            "unavailable step": (
                [0.7] * 6,
                unavailable_at_2,
                [hold, hold, 0, hold, hold, hold] + [0] * 6,
            ),
            "rise after an unavailable step": (
                [0.7] * 2 + [0.8] * 4,
                unavailable_at_2,
                [hold, 6000, 0, 6000, 6000, 6000] + [0] * 6,
            ),
            "value at an unavailable step is ignored": (
                [0.5, 0.8] + [0.6] * 4,
                {"battery_availability": [1, 0] + [1] * 10},
                [6000, 0, 6000, 6000, 6000, 6000] + [0] * 6,
            ),
            # soc_min is 0.3: a value at it is no floor, so it is a step to charge
            # in; a value just above it is a floor under the held level and is not.
            "value at soc_min inside a window": (
                [0.6] * 5 + [0.3] + [0.6] * 6,
                {"soc_init": 0.4},
                [4000] * 5 + [6000] * 7,
            ),
            "value just above soc_min inside a window": (
                [0.6] * 5 + [0.3001] + [0.6] * 6,
                {"soc_init": 0.4},
                [4000] * N,
            ),
        }
        for label, (soc_floor, extra, expected) in cases.items():
            with self.subTest(label):
                opt = build_optimization()
                run(opt, soc_floor=soc_floor, **{"soc_init": 0.5, **extra})
                self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
                np.testing.assert_allclose(floor_of(opt), expected, atol=1e-6)

    def test_a_value_at_an_unavailable_step_does_not_lower_later_floors(self):
        """An unreachable 0.8 at a step where the battery is unavailable gets no
        floor, so it must not count as a shortfall either. The 0.6 after it is
        built in step 0 and held, the same as with that step left null, and no
        clip warning names the unavailable step."""
        opt = build_optimization()
        with self.assertLogs(opt.logger, level="INFO") as logs:
            res = run(
                opt,
                soc_init=0.5,
                battery_availability=[1, 0] + [1] * 10,
                soc_floor=[0.5, 0.8] + [0.6] * 4,
            )
            opt.logger.warning("end of run")  # so assertLogs never sees an empty log
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        np.testing.assert_allclose(
            floor_of(opt), [6000, 0, 6000, 6000, 6000, 6000] + [0] * 6, atol=1e-6
        )
        self.assertGreaterEqual(res["SOC_opt"].to_numpy()[:6].min(), 0.6 - 1e-4)
        clip = [line for line in warnings_of(logs) if "cannot be reached" in line]
        self.assertEqual(clip, [])
        self.assertEqual(missed_floor_warnings(logs), [])

    def test_a_rising_reserve_is_not_climbed(self):
        """Under a reserve that rises from 0.5 to 0.6 at step 1, a run at 0.4 builds
        the 0.1 rise and holds the 0.1 it started under. Charging the whole 0.2 at the
        peak (the old climb back) is what the floor no longer asks for."""
        opt = build_optimization(plant_overrides=EFF_95)
        res = run(opt, soc_init=0.4, load_cost=RESERVE_PRICES, soc_floor=[0.5] + [0.6] * 11)
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        soc = res["SOC_opt"].to_numpy()
        self.assertGreaterEqual(soc[1:].min(), 0.5 - 1e-4)
        self.assertLessEqual(soc[:6].max(), 0.5 + 1e-3)
        # the 0.1 rise at step 1 is needed by the end of step 0
        np.testing.assert_allclose(floor_of(opt), [5000] * N, atol=1e-6)

    def test_a_later_reserve_is_built_toward(self):
        """A reserve that starts later is a target to build toward, not held."""
        floor = [None] * 6 + [0.75] * 6
        free = build_optimization()
        soc_free = run(free, soc_init=0.5, load_cost=RESERVE_PRICES)["SOC_opt"].to_numpy()
        self.assertLess(soc_free[6:].max(), 0.75 - 0.1)

        opt = build_optimization()
        with self.assertLogs(opt.logger, level="INFO") as logs:
            res = run(opt, soc_init=0.5, load_cost=RESERVE_PRICES, soc_floor=floor)
            opt.logger.warning("end of run")  # so assertLogs never sees an empty log
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertGreaterEqual(res["SOC_opt"].to_numpy()[6:].min(), 0.75 - 1e-4)
        self.assertFalse(any("held at soc_init" in line for line in logs.output))
        self.assertEqual(missed_floor_warnings(logs), [])

    def test_a_later_reserve_is_in_place_when_rolling_ticks_reach_it(self):
        """Receding horizon: each tick applies only step 0 of its plan and the next
        tick starts from that SoC, with the reserve sliced from absolute steps. A
        floor binds from the start of its step, so the plan cannot leave the charge
        for the window's first step, which the next tick would hold at soc_init."""
        plant = dict(EFF_95, battery_charge_power_max=3000.0)
        window = range(3, 3 + N)  # absolute steps with a 0.5 reserve
        cases = {
            "flat price": [0.30] * (N + 4),
            "cheap from the window on": [0.50] * 3 + [0.48] + [0.50] * N,
        }
        for label, prices in cases.items():
            with self.subTest(label):
                opt = build_optimization(plant_overrides=plant)
                soc = 0.4
                for tick in range(4):
                    floor = [0.5 if tick + s in window else None for s in range(N)]
                    res = run(opt, soc_init=soc, load_cost=prices[tick : tick + N], soc_floor=floor)
                    self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
                    soc = float(res["SOC_opt"].iloc[0])
                # tick 3 opens the window: it starts at the reserve and holds it
                self.assertGreaterEqual(res["SOC_opt"].to_numpy()[:6].min(), 0.5 - 1e-4)
                self.assertGreaterEqual(floor_of(opt)[0], 0.5 * CAP - 1e-6)

    def test_a_rise_after_the_hold_is_built_by_its_increase(self):
        """Held at soc_init under the first 0.6, then the 0.15 rise to 0.75 is
        built on top of the held level (0.65), not the whole way to 0.75."""
        opt = build_optimization(plant_overrides=EFF_95)
        res = run(opt, soc_init=0.5, load_cost=RESERVE_PRICES, soc_floor=[0.6] * 3 + [0.75] * 9)
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        soc = res["SOC_opt"].to_numpy()
        self.assertGreaterEqual(soc[:3].min(), 0.5 - 1e-4)
        self.assertGreaterEqual(soc[3:].min(), 0.65 - 1e-4)
        self.assertLessEqual(soc[3:6].max(), 0.65 + 1e-3)

    def test_reach_clip_counts_the_charge_efficiency(self):
        """At 95 % charge efficiency two steps add 2 * 0.95 * 5 kW * 0.5 h = 4750 Wh,
        so from 0.3 the reach by the start of step 2 is 0.775, not 0.8."""
        eff = {"battery_charge_efficiency": 0.95, "battery_discharge_efficiency": 0.95}
        reach = 0.3 + 2 * 0.95 * 5000 * 0.5 / CAP  # 0.775
        # Counterfactual: 0.8 by step 1 is not reachable at all.
        target = build_optimization(plant_overrides=eff)
        run(target, soc_init=0.3, soc_final=0.3, soc_target=0.8, soc_target_timestep=1)
        self.assertNotIn(target.optim_status, VALID_OPTIMAL_STATUSES)

        exact = build_optimization(plant_overrides=eff)
        with self.assertNoLogs(exact.logger, level="WARNING"):
            res = run(exact, soc_init=0.3, soc_final=0.3, soc_floor=[None, None, reach])
        self.assertIn(exact.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertGreaterEqual(res["SOC_opt"].to_numpy()[2], reach - 1e-4)

        over = build_optimization(plant_overrides=eff)
        with self.assertLogs(over.logger, level="WARNING") as logs:
            res = run(over, soc_init=0.3, soc_final=0.3, soc_floor=[None, None, 0.8])
        self.assertIn(over.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertAlmostEqual(floor_of(over)[2], reach * CAP, delta=1e-6)
        self.assertAlmostEqual(res["SOC_opt"].to_numpy()[2], reach, delta=1e-4)
        clip = [line for line in warnings_of(logs) if "cannot be reached" in line]
        self.assertEqual(len(clip), 1, logs.output)
        self.assertIn("timestep 2", clip[0])
        self.assertEqual(missed_floor_warnings(logs), [])

    def test_fractional_availability_limits_the_reach(self):
        """Half availability halves the charge ceiling, so the reach per step is
        0.125. A rise from 0.5 to 0.8 at step 1 is built to 0.625, what step 0
        can add, and then held there instead of climbed."""
        opt = build_optimization()
        with self.assertLogs(opt.logger, level="WARNING") as logs:
            res = run(
                opt,
                soc_init=0.5,
                soc_final=0.5,
                battery_availability=0.5,
                soc_floor=[0.5] + [0.8] * 5,
            )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        floor = floor_of(opt) / CAP
        # the rise at step 1 is needed by the end of step 0
        np.testing.assert_allclose(floor[:6], [0.625] * 6, atol=1e-6)
        self.assertGreaterEqual(res["SOC_opt"].to_numpy()[1:6].min(), 0.625 - 1e-4)
        clip = [line for line in warnings_of(logs) if "cannot be reached" in line]
        self.assertEqual(len(clip), 1, logs.output)
        self.assertIn("timestep 1", clip[0])

    def test_each_battery_is_clipped_against_its_own_limits(self):
        opt = build_optimization(plant_overrides=HOUSE_AND_EV)
        run(
            opt,
            soc_init=[0.5, 0.5],
            soc_final=[0.5, 0.5],
            soc_floor=[[None, 1.0], [None, 0.9]],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        # by the start of step 1: house 0.95 * 5 kW * 0.5 h on 10 kWh, EV
        # 0.95 * 11 kW * 0.5 h on 60 kWh
        self.assertAlmostEqual(floor_of(opt, 0)[1], 5000 + 0.95 * 5000 * 0.5, delta=1e-6)
        self.assertAlmostEqual(floor_of(opt, 1)[1], 30000 + 0.95 * 11000 * 0.5, delta=1e-6)


class TestSocFloorSoft(unittest.TestCase):
    """The floor never makes a run infeasible; a miss is priced and logged."""

    GRID_LIMIT = {"maximum_power_from_grid": 2000}

    def test_unreachable_floor_is_optimal_in_one_solve_with_a_warning(self):
        """The grid import limit leaves 500 W to charge, which the reach estimate
        does not count: 0.75 by step 2 is out of reach. The plan charges what it
        can, is Optimal from the first solve and the miss is logged."""
        # Counterfactual: the same level as a hard target is infeasible.
        target = build_optimization(plant_overrides=self.GRID_LIMIT)
        run(target, soc_init=0.5, soc_final=0.5, soc_target=0.75, soc_target_timestep=2)
        self.assertNotIn(target.optim_status, VALID_OPTIMAL_STATUSES)

        opt = build_optimization(plant_overrides=self.GRID_LIMIT)
        with self.assertLogs(opt.logger, level="WARNING") as logs:
            res, solves = run_counting_solves(
                opt, soc_init=0.5, soc_final=0.5, soc_floor=[None, None] + [0.75] * 10
            )
            opt.logger.warning("end of run")
        self.assertEqual(opt.optim_status, "Optimal")
        self.assertEqual(solves, 1)
        soc = res["SOC_opt"].to_numpy()
        # 500 W into 10 kWh for 30 min is 0.025 per step, charged from step 0
        np.testing.assert_allclose(soc[:3], [0.525, 0.55, 0.575], atol=1e-4)
        missed = missed_floor_warnings(logs)
        self.assertEqual(len(missed), 1, logs.output)
        # the 0.75 of step 2 also binds its start, the end of step 1
        self.assertIn("from timestep 1", missed[0])

    def test_forced_shortfall_inside_a_hold_discharges_only_what_it_must(self):
        """Grid limit 1.3 kW under 1.5 kW of load: holding soc_init needs no charge,
        but the battery has to cover 200 W, so the hold is missed by that much and
        no more."""
        plant = {"maximum_power_from_grid": 1300}
        free = build_optimization(plant_overrides=plant)
        soc_free = run(free, soc_init=0.5, load_cost=RESERVE_PRICES)["SOC_opt"].to_numpy()

        opt = build_optimization(plant_overrides=plant)
        with self.assertLogs(opt.logger, level="WARNING") as logs:
            res, solves = run_counting_solves(
                opt, soc_init=0.5, load_cost=RESERVE_PRICES, soc_floor=[0.7] * N
            )
            opt.logger.warning("end of run")
        self.assertEqual(opt.optim_status, "Optimal")
        self.assertEqual(solves, 1)
        soc = res["SOC_opt"].to_numpy()
        np.testing.assert_allclose(soc, 0.5 - 0.01 * np.arange(1, N + 1), atol=1e-4)
        self.assertLess(soc_free[-1], soc[-1] - 0.05)
        self.assertEqual(len(missed_floor_warnings(logs)), 1, logs.output)

    def test_floor_beats_soc_final(self):
        free = build_optimization()
        soc_free = run(free, load_cost=RESERVE_PRICES)["SOC_opt"].to_numpy()
        self.assertLess(soc_free[-1], 0.7 - 0.1)
        opt = build_optimization()
        res = run(opt, load_cost=RESERVE_PRICES, soc_final=0.3, soc_floor=[0.7] * N)
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertGreaterEqual(res["SOC_opt"].to_numpy().min(), 0.7 - 1e-4)

    def test_floor_beats_battery_first(self):
        optim = {"set_battery_first_priority": True}
        free = build_optimization(optim_overrides=optim)
        soc_free = run(free, load_cost=RESERVE_PRICES)["SOC_opt"].to_numpy()
        self.assertLess(soc_free[:6].min(), 0.7 - 0.1)
        opt = build_optimization(optim_overrides=optim)
        res = run(opt, load_cost=RESERVE_PRICES, soc_floor=[0.7] * 6)
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertGreaterEqual(res["SOC_opt"].to_numpy()[:6].min(), 0.7 - 1e-4)

    def test_floor_beats_an_export_spike_when_import_is_free(self):
        """Every import price is negative, so only the export price can size the
        penalty: a one-step reserve at a 3.0 export spike still holds."""
        load_cost = [-0.05] * N
        prod_price = [0.0] * 3 + [3.0] + [0.0] * (N - 3 - 1)
        free = build_optimization()
        soc_free = run(
            free, soc_init=0.8, soc_final=0.8, load_cost=load_cost, prod_price=prod_price
        )["SOC_opt"].to_numpy()
        self.assertLess(soc_free[3], 0.75 - 0.1)
        opt = build_optimization()
        res = run(
            opt,
            soc_init=0.8,
            soc_final=0.8,
            load_cost=load_cost,
            prod_price=prod_price,
            soc_floor=[None] * 3 + [0.75],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertGreaterEqual(res["SOC_opt"].to_numpy()[3], 0.75 - 1e-4)

    def test_floor_holds_under_self_consumption(self):
        free = build_optimization(costfun="self-consumption")
        soc_free = run(free, load_cost=RESERVE_PRICES)["SOC_opt"].to_numpy()
        self.assertLess(soc_free[:6].min(), 0.7 - 0.1)
        opt = build_optimization(costfun="self-consumption")
        res = run(opt, load_cost=RESERVE_PRICES, soc_floor=[0.7] * 6)
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertGreaterEqual(res["SOC_opt"].to_numpy()[:6].min(), 0.7 - 1e-4)

    def test_soc_target_and_floor_both_hold(self):
        """soc_target stays a hard target on its own Parameter; the floor is a
        separate soft one. Where both apply the higher binds."""
        opt = build_optimization()
        res = run(
            opt,
            load_cost=RESERVE_PRICES,
            soc_target=0.75,
            soc_target_timestep=9,
            soc_floor=[0.5] * N,
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        soc = res["SOC_opt"].to_numpy()
        self.assertGreaterEqual(soc.min(), 0.5 - 1e-4)
        self.assertGreaterEqual(soc[9], 0.75 - 1e-4)
        target_floor = opt.param_soc_target_floor[0].value
        self.assertEqual(int(np.count_nonzero(target_floor)), 1)
        np.testing.assert_allclose(floor_of(opt), 0.5 * CAP)

    def test_unreachable_soc_target_with_a_floor_stays_infeasible(self):
        """The floor does not soften soc_target."""
        alone = build_optimization()
        run(alone, soc_init=0.3, soc_final=0.3, soc_target=0.8, soc_target_timestep=0)
        self.assertNotIn(alone.optim_status, VALID_OPTIMAL_STATUSES)
        both = build_optimization()
        run(
            both,
            soc_init=0.3,
            soc_final=0.3,
            soc_target=0.8,
            soc_target_timestep=0,
            soc_floor=[0.4] * N,
        )
        self.assertEqual(both.optim_status, alone.optim_status)

    def test_relaxed_retry_carries_the_floor(self):
        """The semi-continuous load cannot run at 300 W (grid 1.7 kW, 1.5 kW load,
        50 W discharge), with or without the floor. The relaxed retry keeps the
        floor, and the next run without it plans as a fresh problem would."""
        plant = {"maximum_power_from_grid": 1700, "battery_discharge_power_max": 50.0}
        prices = [0.9] * N
        # Counterfactual: the relaxed plan without a floor ends below 0.69.
        free = build_optimization(optim_overrides=ONE_SEMI_CONT_LOAD, plant_overrides=plant)
        res_free = run(free, soc_init=0.7, load_cost=prices)
        self.assertEqual(free.optim_status, "Optimal (Relaxed)")
        self.assertLess(res_free["SOC_opt"].to_numpy().min(), 0.69 - 1e-3)

        opt = build_optimization(optim_overrides=ONE_SEMI_CONT_LOAD, plant_overrides=plant)
        res, solves = run_counting_solves(opt, soc_init=0.7, load_cost=prices, soc_floor=[0.69] * N)
        self.assertEqual(opt.optim_status, "Optimal (Relaxed)")
        self.assertEqual(solves, 2)
        self.assertGreaterEqual(res["SOC_opt"].to_numpy().min(), 0.69 - 1e-4)

        res_next = run(opt, soc_init=0.7, load_cost=prices)
        self.assertEqual(opt.optim_status, "Optimal (Relaxed)")
        self.assertFalse(has_floor_term(opt))
        np.testing.assert_allclose(res_next["SOC_opt"], res_free["SOC_opt"], atol=1e-6)

    def test_coordinator_declines_both_keys(self):
        """The coordinator backend does not model either key yet: it declines and
        the default solver plans, holding the floor."""
        opt = build_optimization(optim_overrides={"optimization_backend": "dantzig_wolfe"})
        res = run(opt, load_cost=RESERVE_PRICES, soc_floor=[0.7] * 6)
        self.assertTrue((res["backend_used"] == "cvxpy").all())
        self.assertTrue(res["backend_fallback_reason"].str.contains("soc_floor").all())
        self.assertGreaterEqual(res["SOC_opt"].to_numpy()[:6].min(), 0.7 - 1e-4)
        opt = build_optimization(optim_overrides={"optimization_backend": "dantzig_wolfe"})
        res = run(opt, battery_discharge_availability=[0] * N)
        self.assertTrue((res["backend_used"] == "cvxpy").all())
        self.assertTrue(
            res["backend_fallback_reason"].str.contains("battery_discharge_availability").all()
        )

    def test_penalty_is_priced_from_the_largest_price_by_size(self):
        factor = getattr(optimization, "SOC_FLOOR_SHORTFALL_PENALTY_FACTOR", None)
        self.assertEqual(factor, 1000.0)
        for label, load_cost, prod_price, ref in (
            ("export", RESERVE_PRICES, [0.8] + [0.0] * (N - 1), 0.8),
            ("negative import", [0.0] * 6 + [-2.0] + [0.0] * 5, [0.0] * N, 2.0),
        ):
            with self.subTest(label):
                opt = build_optimization()
                run(opt, load_cost=load_cost, prod_price=prod_price, soc_floor=[0.7])
                penalty = getattr(opt, "param_soc_floor_penalty", None)
                self.assertIsNotNone(penalty)
                self.assertAlmostEqual(penalty.value, 0.001 * 1000.0 * ref)

    def test_floor_holds_when_no_price_is_positive(self):
        """Every import price is 0 or a paid -2.0: the room to soak up the paid step
        is worth 2 per kWh, so the floor must be priced from the size of that price,
        not from a 1e-3 floor."""
        load_cost = [0.0] * 6 + [-2.0] + [0.0] * 5
        free = build_optimization(plant_overrides=EFF_95)
        soc_free = run(free, soc_init=0.8, soc_final=0.8, load_cost=load_cost)["SOC_opt"]
        self.assertLess(soc_free.to_numpy().min(), 0.7 - 0.1)
        opt = build_optimization(plant_overrides=EFF_95)
        with self.assertLogs(opt.logger, level="INFO") as logs:
            res = run(opt, soc_init=0.8, soc_final=0.8, load_cost=load_cost, soc_floor=[0.7] * N)
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertGreaterEqual(res["SOC_opt"].to_numpy().min(), 0.7 - 1e-4)
        self.assertEqual(missed_floor_warnings(logs), [])


class TestSocFloorShapes(unittest.TestCase):
    def test_null_nan_and_soc_min_entries_set_no_floor(self):
        opt = build_optimization()
        run(opt, soc_floor=[None, 0.6, float("nan"), 0.3, 0.2, 0.6])
        # each floor is also set on the step before it (start of its step)
        np.testing.assert_allclose(floor_of(opt), [6000, 6000, 0, 0, 6000, 6000] + [0] * (N - 6))

    def test_values_above_soc_max_are_clamped(self):
        opt = build_optimization()
        run(opt, soc_init=0.8, soc_final=0.8, soc_floor=[0.95, 0.95])
        np.testing.assert_allclose(floor_of(opt)[:2], 0.8 * CAP)

    def test_short_list_is_padded_with_no_floor_and_long_list_truncated(self):
        for soc_floor, expected in (
            ([0.6] * 3, [6000] * 3 + [0] * (N - 3)),
            ([0.6] * (N + 5), [6000] * N),
            ([0.6] * 10**6, [6000] * N),
        ):
            with self.subTest(length=len(soc_floor)):
                opt = build_optimization()
                run(opt, soc_floor=soc_floor)
                np.testing.assert_allclose(floor_of(opt), expected)

    def test_per_battery_lists(self):
        opt = build_optimization(plant_overrides=HOUSE_AND_EV)
        run(opt, soc_init=[0.5, 0.5], soc_final=[0.5, 0.5], soc_floor=[None, [0.5] * N])
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        np.testing.assert_array_equal(floor_of(opt, 0), np.zeros(N))
        np.testing.assert_allclose(floor_of(opt, 1), 0.5 * 60000)
        self.assertFalse(has_floor_term(opt, 0))

    def test_invalid_input_raises(self):
        cases = (
            ("scalar", {}, 0.6),
            ("0-d array", {}, np.array(0.6)),
            ("0-d array per battery", HOUSE_AND_EV, [None, np.array(0.6)]),
            ("percent", {}, [60] * N),
            ("negative", {}, [-0.1] * N),
            ("text", {}, ["high"] * N),
            ("scalar per battery", HOUSE_AND_EV, [None, 0.6]),
            ("flat list at N=2", HOUSE_AND_EV, [0.6] * N),
            ("wrong battery count", HOUSE_AND_EV, [None, None, [0.6] * N]),
            ("nested entry", {}, [[[0.6] * N]]),
            ("2-D entry at N=2", HOUSE_AND_EV, [None, [[0.7] * 6, [0.3] * 6]]),
            ("boolean", {}, [True] * N),
        )
        for label, plant, value in cases:
            with self.subTest(label):
                opt = build_optimization(plant_overrides=plant)
                soc = [0.5] * opt.n_batt if opt.n_batt > 1 else 0.5
                with self.assertRaises(ValueError):
                    run(opt, soc_init=soc, soc_final=soc, soc_floor=value)


if __name__ == "__main__":
    unittest.main()
