#!/usr/bin/env python3

"""Tests for the PV forecast bias self-tuning engine (issue #841, Phase 1).

Covers the adaptive-conformal-inference recursion in
``emhass.pv_bias_calibration``: convergence to a feasible target shortfall rate,
the exact recursion arithmetic, the static-curve / feasibility diagnostics, the
pluggable score hook, and input validation. The engine is side-effect free, so
these are pure unit tests over in-memory arrays.
"""

import logging
import unittest

import numpy as np

# Base-safe import: on the base branch the module does not exist, so the
# behavioural assertions below fail cleanly rather than erroring at import time.
try:
    from emhass import pv_bias_calibration as pbc
except Exception:  # pragma: no cover - only hit on the base branch
    pbc = None

logger = logging.getLogger("test_pv_bias_calibration")


def _hot_forecast_history(n=500, seed=42, p50=100.0, p10=60.0, mean=85.0, sd=15.0):
    """Synthetic history where P50 runs hot (realised PV below it most days).

    With realised ~ N(mean, sd), a plan at P50=100 is exceeded from below ~84%
    of the time and a plan at P10=60 ~5%, so a 10% target is well inside the
    feasible range and the recursion has room to converge to it.
    """
    rng = np.random.default_rng(seed)
    actual = np.clip(rng.normal(mean, sd, n), 0.0, None)
    return (
        np.full(n, p10, dtype=float),
        np.full(n, p50, dtype=float),
        actual,
    )


@unittest.skipIf(pbc is None, "pv_bias_calibration module not present")
class TestPvBiasCalibration(unittest.TestCase):
    # ── Convergence ──────────────────────────────────────────────────────────
    def test_converges_to_feasible_target(self):
        p10, p50, actual = _hot_forecast_history(n=600, seed=7)
        res = pbc.compute_pv_bias_calibration(
            p10, p50, actual, target_shortfall_rate=0.10, gamma=0.10
        )
        # The settled shortfall rate should sit near the 10% target...
        self.assertLess(abs(res["achieved_shortfall_rate_tail"] - 0.10), 0.06)
        # ...reached by lifting bias well off the pure-P50 default of 0.
        self.assertGreater(res["recommended_bias"], 0.5)
        self.assertLess(res["recommended_bias"], 1.0)
        self.assertTrue(res["target_feasible"])
        self.assertTrue(res["converged"])

    def test_well_calibrated_forecast_keeps_bias_low(self):
        # If P50 is already ~calibrated at the target (realised below P50 ~10%),
        # the recursion should not wander far from the pure-P50 default.
        rng = np.random.default_rng(1)
        n = 600
        p50 = np.full(n, 100.0)
        p10 = np.full(n, 60.0)
        # mean chosen so P(actual < 100) ~ 0.10  ->  z=-1.28 -> mean = 100 + 1.28*sd
        actual = np.clip(rng.normal(100.0 + 1.28 * 12.0, 12.0, n), 0.0, None)
        res = pbc.compute_pv_bias_calibration(
            p50=p50, p10=p10, actual=actual, target_shortfall_rate=0.10, gamma=0.10
        )
        self.assertLess(res["recommended_bias"], 0.25)

    # ── Exact recursion arithmetic (deterministic) ───────────────────────────
    def test_recursion_arithmetic(self):
        # p10=0, p50=10 -> plan = (1-bias)*10. bias0=0, gamma=0.1, alpha=0.1.
        # t0: plan=10, actual=5 -> shortfall=1 -> bias = 0 + 0.1*(1-0.1) = 0.09
        # t1: plan=9.1, actual=20 -> shortfall=0 -> bias = 0.09 + 0.1*(0-0.1) = 0.08
        res = pbc.compute_pv_bias_calibration(
            p10=[0.0, 0.0],
            p50=[10.0, 10.0],
            actual=[5.0, 20.0],
            target_shortfall_rate=0.10,
            gamma=0.10,
            bias0=0.0,
        )
        self.assertEqual(res["bias_trajectory"], [0.0, 0.09])
        self.assertAlmostEqual(res["final_bias"], 0.08, places=6)
        self.assertAlmostEqual(res["achieved_shortfall_rate"], 0.5, places=6)

    # ── Static-curve / feasibility diagnostics ───────────────────────────────
    def test_static_curve_is_monotone_non_increasing(self):
        p10, p50, actual = _hot_forecast_history(seed=3)
        curve = pbc.static_shortfall_curve(p10, p50, actual)
        rates = [r for _, r in curve]
        self.assertEqual(curve[0][0], 0.0)
        self.assertEqual(curve[-1][0], 1.0)
        for earlier, later in zip(rates, rates[1:]):
            self.assertLessEqual(later, earlier + 1e-9)

    def test_feasible_range_orientation(self):
        p10, p50, actual = _hot_forecast_history(seed=5)
        lo, hi = pbc.feasible_shortfall_range(p10, p50, actual)
        self.assertLessEqual(lo, hi)  # bias=1 rate <= bias=0 rate
        self.assertGreater(hi, 0.5)  # P50 runs hot -> many shortfalls at bias 0
        self.assertLess(lo, 0.15)  # P10 is conservative -> few shortfalls at bias 1

    def test_infeasible_target_saturates_and_warns(self):
        p10, p50, actual = _hot_forecast_history(seed=9)
        lo, _ = pbc.feasible_shortfall_range(p10, p50, actual)
        infeasible = max(1e-3, lo / 2.0)  # below what pure P10 can deliver
        with self.assertLogs("test_pv_bias_calibration", level="WARNING"):
            res = pbc.compute_pv_bias_calibration(
                p10,
                p50,
                actual,
                target_shortfall_rate=infeasible,
                gamma=0.10,
                logger=logger,
            )
        # Cannot reach an impossible target -> saturates high, not "converged".
        self.assertFalse(res["target_feasible"])
        self.assertFalse(res["converged"])
        self.assertGreater(res["recommended_bias"], 0.9)

    # ── Pluggable score hook ─────────────────────────────────────────────────
    def test_pluggable_score_fn(self):
        # A score that is always 0 (< alpha every step) pushes bias down to 0.
        p10, p50, actual = _hot_forecast_history(n=100, seed=2)
        res = pbc.compute_pv_bias_calibration(
            p10,
            p50,
            actual,
            target_shortfall_rate=0.10,
            gamma=0.20,
            bias0=0.5,
            score_fn=lambda planned, act: np.zeros_like(np.atleast_1d(act), dtype=float),
        )
        self.assertEqual(res["final_bias"], 0.0)

    # ── Input validation ─────────────────────────────────────────────────────
    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            pbc.compute_pv_bias_calibration([1.0, 2.0], [1.0], [1.0])

    def test_empty_history_raises(self):
        with self.assertRaises(ValueError):
            pbc.compute_pv_bias_calibration([], [], [])

    def test_non_finite_rows_dropped(self):
        # One NaN row is dropped rather than poisoning the recursion.
        res = pbc.compute_pv_bias_calibration(
            p10=[0.0, 0.0, np.nan],
            p50=[10.0, 10.0, 10.0],
            actual=[5.0, 20.0, 7.0],
            target_shortfall_rate=0.10,
            gamma=0.10,
        )
        self.assertEqual(res["n_observations"], 2)

    def test_bad_parameters_raise(self):
        p10, p50, actual = _hot_forecast_history(n=50)
        for kwargs in (
            {"target_shortfall_rate": 0.0},
            {"target_shortfall_rate": 1.0},
            {"gamma": 0.0},
            {"gamma": -0.1},
            {"bias0": 1.5},
        ):
            with self.assertRaises(ValueError):
                pbc.compute_pv_bias_calibration(p10, p50, actual, **kwargs)


if __name__ == "__main__":
    unittest.main()
