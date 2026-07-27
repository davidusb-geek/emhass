#!/usr/bin/env python3

"""
Self-tuning of the PV forecast conservative-bias knob (issue #841, Phase 1).

EMHASS can blend Solcast's ``pv_estimate10`` (P10) with the central
``pv_estimate`` (P50) through the opt-in ``weather_forecast_pv_quantile_bias``
knob added in #961::

    planned_forecast = bias * P10 + (1 - bias) * P50      # bias in [0, 1]

Setting that knob by hand is awkward: the value that produces a given
conservativeness depends on the site, the season and the forecast pipeline, and
it drifts over time. This module turns the knob into a *self-calibrating*
parameter using **adaptive conformal inference (ACI)** — a one-scalar online
recursion that walks ``bias`` until the realised PV lands below the planned
forecast at a user-chosen target rate, and keeps it there as conditions drift.

It is a **reporting / recommendation engine with no side effects**: given a
history of ``(P10, P50, realised PV)`` observations it returns a recommended
``bias`` plus diagnostics. It never touches the optimization, the LP or the live
forecast. A caller (e.g. a CLI action, once the P10/P50-vs-actual history is
being logged) decides whether to surface the recommendation or apply it; the
default behaviour of EMHASS is unchanged.

The recursion (one update per resolved observation) is::

    planned_t  = bias_t * P10_t + (1 - bias_t) * P50_t
    shortfall_t = 1 if realised_t < planned_t else 0
    bias_{t+1} = clip(bias_t + gamma * (shortfall_t - alpha), 0, 1)

where ``alpha`` is the target one-sided shortfall rate. A run of shortfalls
pushes ``bias`` toward P10 (more conservative); quiet stretches relax it back
toward P50. The empirical shortfall rate converges to ``alpha`` provided the
target is inside the range the P10-P50 blend can express (see
``feasible_shortfall_range``). This is asymmetric by construction — the climb
per shortfall day is ``gamma * (1 - alpha)`` and the relax per quiet day is
``gamma * alpha`` — so protection ramps up quickly when the forecast starts
over-calling and releases slowly afterwards.

References
----------
* Gibbs, I. and Candès, E. (2021). *Adaptive Conformal Inference Under
  Distribution Shift.* NeurIPS 2021. arXiv:2106.00170. The fixed-``gamma``
  recursion implemented here.
* Gibbs, I. and Candès, E. (2024). *Conformal Inference for Online Prediction
  with Arbitrary Distribution Shifts.* JMLR. arXiv:2208.08401. The
  parameter-free (DtACI) refinement that removes the ``gamma`` choice by
  tracking an expert ensemble; documented here as a future extension.

Notes
-----
The shortfall indicator is the default score. The recursion only assumes the
score is (weakly) **non-increasing in ``bias``** — true for the shortfall rate,
because a larger bias lowers the planned forecast and can only make ``realised <
planned`` less likely. A different monotone score (for example a normalised
cost-regret or a threshold-weighted CRPS that prices *how much* the conservatism
costs on clear days) can be supplied through ``score_fn`` without changing the
recursion, which is the hook for the "merit-function" tie-breaker discussed on
the issue. This module intentionally ships only the single *shift* scalar (v1);
a second *shrink* degree of freedom on the band width is a separate follow-up.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Default target one-sided shortfall rate: realised PV is allowed to land below
# the planned (biased) forecast on this fraction of observations.
DEFAULT_TARGET_SHORTFALL_RATE = 0.10
# Default ACI learning rate. On daily updates 0.05-0.10 converges within a
# couple of months and stays calm; 0.20 is faster but jumpy (issue #841 log).
DEFAULT_GAMMA = 0.05
# Number of grid points used for the static shortfall-vs-bias diagnostic curve.
_STATIC_CURVE_POINTS = 21
# Fraction of the trajectory used to summarise the "settled" behaviour.
_TAIL_FRACTION = 0.5
# A run is called "converged" when the settled shortfall rate sits within this
# many percentage points of the target (and the bias is off both clip bounds).
_CONVERGENCE_TOL = 0.05


def _planned_forecast(p10: np.ndarray, p50: np.ndarray, bias: float) -> np.ndarray:
    """Blend P10 and P50 exactly as the #961 Solcast path does."""
    return bias * p10 + (1.0 - bias) * p50


def _default_shortfall_score(planned: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """One-sided shortfall indicator: 1 where realised PV fell below the plan.

    Non-increasing in ``bias`` (a larger bias lowers ``planned``), which is the
    property the recursion relies on to converge.
    """
    return (actual < planned).astype(float)


def static_shortfall_curve(
    p10: np.ndarray,
    p50: np.ndarray,
    actual: np.ndarray,
    n_points: int = _STATIC_CURVE_POINTS,
) -> list[tuple[float, float]]:
    """Realised shortfall rate at each *fixed* bias on a [0, 1] grid.

    This is the static reference curve: it shows the plateau structure and
    reveals whether a target rate is even reachable by the blend. Monotone
    non-increasing in ``bias`` by construction.

    :return: list of ``(bias, shortfall_rate)`` pairs.
    """
    grid = np.linspace(0.0, 1.0, n_points)
    out = []
    for b in grid:
        planned = _planned_forecast(p10, p50, float(b))
        rate = float(np.mean(_default_shortfall_score(planned, actual)))
        out.append((round(float(b), 4), round(rate, 4)))
    return out


def feasible_shortfall_range(
    p10: np.ndarray, p50: np.ndarray, actual: np.ndarray
) -> tuple[float, float]:
    """Shortfall rates achievable by the blend, as ``(min_at_bias1, max_at_bias0)``.

    A target outside this range cannot be met — the recursion will saturate at a
    clip bound rather than converge. Because P10 <= P50 the rate is highest at
    ``bias = 0`` (pure P50) and lowest at ``bias = 1`` (pure P10).
    """
    rate_hi = float(np.mean(_default_shortfall_score(_planned_forecast(p10, p50, 0.0), actual)))
    rate_lo = float(np.mean(_default_shortfall_score(_planned_forecast(p10, p50, 1.0), actual)))
    return (round(rate_lo, 4), round(rate_hi, 4))


def _coerce_history(p10, p50, actual) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and align the three history arrays.

    Rows with any non-finite value are dropped (a missing P10/actual is not a
    silent zero). Raises ``ValueError`` on length mismatch or empty input so a
    caller cannot get a confidently-wrong recommendation from garbage.
    """
    p10 = np.asarray(p10, dtype=float)
    p50 = np.asarray(p50, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if not (len(p10) == len(p50) == len(actual)):
        raise ValueError(f"p10/p50/actual length mismatch: {len(p10)}/{len(p50)}/{len(actual)}")
    if len(p10) == 0:
        raise ValueError("empty history: nothing to calibrate")
    finite = np.isfinite(p10) & np.isfinite(p50) & np.isfinite(actual)
    if not finite.any():
        raise ValueError("no finite observations in history")
    return p10[finite], p50[finite], actual[finite]


def compute_pv_bias_calibration(
    p10,
    p50,
    actual,
    *,
    target_shortfall_rate: float = DEFAULT_TARGET_SHORTFALL_RATE,
    gamma: float = DEFAULT_GAMMA,
    bias0: float = 0.0,
    score_fn=None,
    logger: logging.Logger | None = logger,
) -> dict:
    """Recommend ``weather_forecast_pv_quantile_bias`` from logged history via ACI.

    Replays the one-scalar ACI recursion over an ordered history of
    ``(P10, P50, realised)`` observations (whatever aggregation the caller uses
    for a single update step — e.g. whole-day PV energy totals) and returns the
    settled ``bias`` recommendation plus honest diagnostics.

    :param p10: ordered P10 (conservative) PV forecasts, one per update step.
    :param p50: ordered P50 (central) PV forecasts, same order/length.
    :param actual: ordered realised PV, same order/length.
    :param target_shortfall_rate: target one-sided shortfall rate ``alpha`` in
        (0, 1) — the fraction of steps realised PV may fall below the plan.
    :param gamma: ACI learning rate (> 0). 0.05-0.10 is the daily-update sweet
        spot; larger converges faster but reacts more to single events.
    :param bias0: initial bias in [0, 1] (0.0 = pure P50, the current default).
    :param score_fn: optional ``callable(planned, actual) -> array`` returning a
        per-step score in [0, 1] that is non-increasing in ``bias``. Defaults to
        the one-sided shortfall indicator. This is the hook for a cost-regret /
        threshold-weighted-CRPS tie-breaker; the recursion is unchanged.
    :param logger: optional logger for warnings; pass ``None`` to silence.
    :return: dict with the recommendation and diagnostics (see below), or a dict
        with ``recommended_bias=None`` and a ``reason`` when it cannot be
        computed.
    """
    if not 0.0 < target_shortfall_rate < 1.0:
        raise ValueError(f"target_shortfall_rate must be in (0, 1), got {target_shortfall_rate}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    if not 0.0 <= bias0 <= 1.0:
        raise ValueError(f"bias0 must be in [0, 1], got {bias0}")

    p10, p50, actual = _coerce_history(p10, p50, actual)
    score = score_fn if score_fn is not None else _default_shortfall_score
    alpha = float(target_shortfall_rate)
    n = len(p10)

    feasible = feasible_shortfall_range(p10, p50, actual)
    target_feasible = bool(feasible[0] <= alpha <= feasible[1])
    if not target_feasible and logger is not None:
        logger.warning(
            "target_shortfall_rate=%.3f is outside the feasible range %s the "
            "P10-P50 blend can express; the recursion will saturate at a clip "
            "bound (pure P10 or pure P50) rather than converge to it.",
            alpha,
            feasible,
        )

    # ACI recursion: one update per resolved observation. Clipping to [0, 1] is
    # expected during warm-up (a cold start at bias 0 dips against the lower
    # bound on the first quiet steps); it is not itself a problem, so it is not
    # flagged here — saturation is judged from the settled tail below.
    bias = float(bias0)
    trajectory = np.empty(n, dtype=float)
    scores = np.empty(n, dtype=float)
    for t in range(n):
        planned_t = _planned_forecast(p10[t], p50[t], bias)
        s_t = float(np.asarray(score(planned_t, actual[t])).reshape(-1)[0])
        trajectory[t] = bias
        scores[t] = s_t
        bias = float(min(1.0, max(0.0, bias + gamma * (s_t - alpha))))

    # Summarise the settled behaviour over the trajectory tail.
    tail_start = int(n * (1.0 - _TAIL_FRACTION))
    tail_bias = trajectory[tail_start:]
    tail_scores = scores[tail_start:]
    recommended_bias = round(float(np.mean(tail_bias)), 4)
    achieved_rate = round(float(np.mean(scores)), 4)
    achieved_rate_tail = round(float(np.mean(tail_scores)), 4)
    # "Converged" = the target is feasible for the blend AND the settled
    # shortfall rate sits within tolerance of it. An infeasible target
    # (target_feasible is False) saturates the recursion toward a clip bound
    # instead, so it is never reported as converged.
    converged = bool(target_feasible and abs(achieved_rate_tail - alpha) <= _CONVERGENCE_TOL)

    return {
        "recommended_bias": recommended_bias,
        "final_bias": round(bias, 4),
        "target_shortfall_rate": alpha,
        "achieved_shortfall_rate": achieved_rate,
        "achieved_shortfall_rate_tail": achieved_rate_tail,
        "feasible_shortfall_range": feasible,
        "target_feasible": target_feasible,
        "converged": converged,
        "n_observations": int(n),
        "gamma": float(gamma),
        "bias_trajectory": [round(float(b), 4) for b in trajectory],
        "static_shortfall_curve": static_shortfall_curve(p10, p50, actual),
    }
