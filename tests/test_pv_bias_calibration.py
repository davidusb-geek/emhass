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


def _history_with_curtailed_days(n=200, seed=11, every=10):
    """A clean history with curtailed days spliced in every ``every`` steps.

    A curtailed day reports realised PV far below what the array could have
    made, so it looks like a severe forecast shortfall while carrying no
    information about the forecast at all.

    :return: ``(clean_triple, spliced_triple, censored_mask)``.
    """
    p10c, p50c, actualc = _hot_forecast_history(n=n, seed=seed)
    p10, p50, actual, censored = [], [], [], []
    for i in range(len(actualc)):
        p10.append(p10c[i])
        p50.append(p50c[i])
        actual.append(actualc[i])
        censored.append(False)
        if i % every == 0:
            p10.append(60.0)
            p50.append(100.0)
            actual.append(1.0)  # export-limited to nearly nothing
            censored.append(True)
    return (
        (p10c, p50c, actualc),
        (np.array(p10), np.array(p50), np.array(actual)),
        np.array(censored, dtype=bool),
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

    # ── Curtailment (censored observations) ──────────────────────────────────
    def test_masking_curtailed_steps_reproduces_the_clean_history(self):
        (p10c, p50c, actualc), (p10, p50, actual), curtailed = _history_with_curtailed_days()
        clean = pbc.compute_pv_bias_calibration(p10c, p50c, actualc, gamma=0.10)
        masked = pbc.compute_pv_bias_calibration(p10, p50, actual, curtailed=curtailed, gamma=0.10)
        # Masking the curtailed steps must recover the clean answer exactly.
        self.assertEqual(masked["n_observations"], clean["n_observations"])
        self.assertEqual(masked["recommended_bias"], clean["recommended_bias"])
        self.assertEqual(masked["bias_trajectory"], clean["bias_trajectory"])
        self.assertEqual(masked["n_curtailed_excluded"], int(np.count_nonzero(curtailed)))

    def test_unmasked_curtailment_over_tightens_the_bias(self):
        # The failure mode the mask exists to prevent: left unflagged, curtailed
        # steps read as forecast shortfalls and ratchet the bias up for the
        # wrong reason (reviewer point on PR #1043).
        (p10c, p50c, actualc), (p10, p50, actual), _ = _history_with_curtailed_days()
        clean = pbc.compute_pv_bias_calibration(p10c, p50c, actualc, gamma=0.10)
        unmasked = pbc.compute_pv_bias_calibration(p10, p50, actual, gamma=0.10)
        self.assertGreater(unmasked["recommended_bias"], clean["recommended_bias"])
        self.assertGreater(unmasked["achieved_shortfall_rate"], clean["achieved_shortfall_rate"])

    def test_accepts_a_curtailment_power_series(self):
        # EMHASS publishes curtailment as a power series (W), not a bool mask;
        # it is thresholded at > 0 exactly as the adjusted-PV path does.
        (_, _, _), (p10, p50, actual), curtailed = _history_with_curtailed_days()
        watts = np.where(curtailed, 1500.0, 0.0)
        watts[2] = np.nan  # a gap in the curtailment sensor is "not curtailed"
        from_bool = pbc.compute_pv_bias_calibration(p10, p50, actual, curtailed=curtailed)
        from_watts = pbc.compute_pv_bias_calibration(p10, p50, actual, curtailed=watts)
        self.assertEqual(from_watts["n_curtailed_excluded"], from_bool["n_curtailed_excluded"])
        self.assertEqual(from_watts["recommended_bias"], from_bool["recommended_bias"])

    def test_curtailment_margin_widens_the_exclusion(self):
        # Curtailment ramps in and out, so the neighbours can be partly capped.
        n = 9
        p10 = np.zeros(n)
        p50 = np.full(n, 10.0)
        actual = np.full(n, 20.0)
        curtailed = np.zeros(n, dtype=bool)
        curtailed[4] = True
        r0 = pbc.compute_pv_bias_calibration(p10, p50, actual, curtailed=curtailed)
        r1 = pbc.compute_pv_bias_calibration(
            p10, p50, actual, curtailed=curtailed, curtailment_margin=1
        )
        self.assertEqual(r0["n_curtailed_excluded"], 1)
        self.assertEqual(r0["n_observations"], n - 1)
        self.assertEqual(r1["n_curtailed_excluded"], 3)  # the step plus both neighbours
        self.assertEqual(r1["n_observations"], n - 3)

    def test_excluded_count_is_finite_rows_only(self):
        # A row that is BOTH non-finite and curtailed is dropped once, as a
        # non-finite row -- the curtailed count reports real data loss only.
        res = pbc.compute_pv_bias_calibration(
            p10=[0.0, 0.0, 0.0, np.nan],
            p50=[10.0, 10.0, 10.0, 10.0],
            actual=[5.0, 20.0, 5.0, 7.0],
            curtailed=[False, True, False, True],
        )
        self.assertEqual(res["n_curtailed_excluded"], 1)  # not 2
        self.assertEqual(res["n_observations"], 2)

    def test_heavy_curtailment_warns_and_is_reported(self):
        p10, p50, actual = _hot_forecast_history(n=100, seed=13)
        curtailed = np.zeros(100, dtype=bool)
        curtailed[:40] = True
        with self.assertLogs("test_pv_bias_calibration", level="WARNING"):
            res = pbc.compute_pv_bias_calibration(
                p10, p50, actual, curtailed=curtailed, gamma=0.10, logger=logger
            )
        self.assertEqual(res["n_curtailed_excluded"], 40)
        self.assertEqual(res["n_observations"], 60)
        self.assertAlmostEqual(res["curtailed_fraction"], 0.40, places=6)

    def test_curtailment_validation(self):
        p10, p50, actual = _hot_forecast_history(n=20)
        with self.assertRaises(ValueError):  # mask length mismatch
            pbc.compute_pv_bias_calibration(p10, p50, actual, curtailed=[True, False])
        with self.assertRaises(ValueError):  # nothing uncurtailed left
            pbc.compute_pv_bias_calibration(p10, p50, actual, curtailed=np.ones(20, dtype=bool))
        with self.assertRaises(ValueError):  # negative margin
            pbc.compute_pv_bias_calibration(
                p10, p50, actual, curtailed=np.zeros(20, dtype=bool), curtailment_margin=-1
            )

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
