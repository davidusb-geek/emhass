#!/usr/bin/env python3

"""
Opt-in battery self-identification.

Learn a battery's **usable capacity** and **round-trip efficiency** from Home
Assistant history (signed AC-side battery power plus measured state of charge),
the same way ``set_use_adjusted_pv`` learns a PV bias correction from history.

This module holds the pure estimators only. It performs no I/O, reads no
configuration files, and knows nothing about the :class:`Forecast` object. The
orchestration (cadence gate, history retrieval, JSON persistence, optional HA
sensor publishing, and the never-raise fallback) lives in ``command_line.py`` as
``identify_battery``, mirroring the ``adjust_pv_forecast`` orchestrator.

What is identifiable from a single AC-side power meter plus reported SoC:

* **Usable capacity** ``C`` (Wh per unit of *reported* SoC). This is capacity in
  whatever coordinate system the BMS uses to report SoC, which is exactly what
  the optimizer consumes, so the unknown BMS mapping cancels as long as it is
  consistent.
* **Round-trip efficiency** ``RTE`` (the lumped AC-side product
  ``eff_chg * eff_dis``).

What is **not** identifiable and is deliberately never invented here:

* The split of ``RTE`` into separate charge/discharge efficiencies. From one
  meter that split is structurally non-identifiable, so we set
  ``eff_chg = eff_dis = sqrt(RTE)`` and say so.
* Cell loss versus inverter conversion loss (needs a DC-side signal).
* Standby/idle draw (second order, deferred).

Identifiability algebra (matches the optimizer's SoC equation
``power_flow = p_sto_pos * (1/eff_dis) + p_sto_neg * eff_chg``):

* a charge run gives slope ``S_chg = throughput_Wh / dSoC = C / (100 * eff_chg)``
* a discharge run gives slope ``S_dis = throughput_Wh / |dSoC| = eff_dis * C / 100``

so ``S_dis / S_chg = eff_dis * eff_chg = RTE`` (capacity independent) and
``C = 100 * sqrt(S_chg * S_dis)`` under the symmetric split.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- Estimator constants (internal; deliberately NOT config params) ----------
# These are sufficiency floors and physics guardrails, not user tuning knobs.
# A user must not be able to widen the safety rails, and none of these changes
# externally visible behaviour the way the three optim_conf params do.

# Segmentation
POWER_DEADBAND_W = 25.0  # |power| below this is treated as idle (sign hysteresis)
SOC_REVERSAL_HYSTERESIS = 1.0  # SoC points of counter-move tolerated before a run splits
MIN_SEGMENT_DURATION = pd.Timedelta(minutes=20)
MIN_SOC_SWING_PER_SEGMENT = 10.0  # a segment must move at least this many SoC points to be "deep"

# Sufficiency floors
MIN_DEEP_SEGMENTS = 3  # per direction, before a capacity fit is attempted
MIN_EQUAL_SOC_CYCLES = 8  # closed cycles before the energy-balance RTE cross-check is trusted

# Confidence gates (observe -> suggest)
MAX_RELATIVE_CI = 0.15  # (ci_high - ci_low) / value must be tighter than this to publish
TLS_THEILSEN_MAX_DIVERGENCE = 0.15  # relative slope disagreement above this => stay observe-only
BOOTSTRAP_N_RESAMPLES = 800
# A near-zero-width bootstrap CI from only a handful of near-identical segments
# is false confidence, not real precision: it measures between-segment scatter,
# not accuracy, and would let a biased-but-consistent fit pass the CI gate. So an
# implausibly tight CI is only trusted once there are enough independent cycles.
CI_IMPLAUSIBLY_TIGHT_REL = 1e-4
CI_MIN_SEGMENTS_FOR_TIGHT = 6

# Physics sanity bounds
CAPACITY_SANITY_LOW = 0.3  # multiplier on configured Enom
CAPACITY_SANITY_HIGH = 1.5
# Absolute fallback band (Wh) when no configured capacity is available to scale.
ABS_CAPACITY_LOW_WH = 200.0
ABS_CAPACITY_HIGH_WH = 500000.0
SQRT_RTE_LOW = 0.80  # one-way efficiency = sqrt(RTE) plausible range
SQRT_RTE_HIGH = 0.999

# Energy-balance cross-check is only meaningful over an approximately CLOSED
# window: if the net SoC drift is large relative to the total SoC swung, the
# unbalanced tail biases the balance, so the cross-check is withheld.
CLOSURE_TOLERANCE_FRACTION = 0.15

SCHEMA_VERSION = 1


@dataclass
class _Segment:
    """A single monotonic charge or discharge run."""

    direction: str  # "charge" or "discharge"
    d_soc: float  # signed SoC change over the run (positive for charge)
    throughput_wh: float  # integrated |AC power| over the run, Wh (always >= 0)
    duration: pd.Timedelta
    start: pd.Timestamp
    end: pd.Timestamp
    energy_in_wh: float  # signed: charge energy delivered at AC (0 for discharge runs)
    energy_out_wh: float  # signed: discharge energy taken at AC (0 for charge runs)


@dataclass
class BatteryIdentificationResult:
    """
    Outcome of a single identification pass.

    ``status`` drives every downstream decision:

    * ``ok`` - estimate passed every guardrail; safe to publish / persist.
    * ``insufficient_data`` - not enough deep segments / cycles.
    * ``rejected_sanity_check`` - a value fell outside physical bounds.
    * ``low_confidence`` - CI too wide or TLS/Theil-Sen cross-checks diverged.

    Only an ``ok`` result should be written to disk or published to HA.
    """

    status: str
    messages: list[str] = field(default_factory=list)
    # Populated only when a fit ran (status may still be non-ok):
    capacity_wh: float | None = None
    capacity_ci_low: float | None = None
    capacity_ci_high: float | None = None
    capacity_theil_sen_wh: float | None = None
    capacity_intercept_diag: float | None = None
    round_trip_efficiency: float | None = None
    rte_ci_low: float | None = None
    rte_ci_high: float | None = None
    rte_crosscheck_energy_balance: float | None = None
    eta_symmetric: float | None = None  # sqrt(RTE); applied to both legs
    n_charge_segments: int = 0
    n_discharge_segments: int = 0
    n_equal_soc_cycles: int = 0

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict:
        """Serialisable payload for the ``data_path`` JSON (scalars only)."""
        cap_kwh = None if self.capacity_wh is None else round(self.capacity_wh / 1000.0, 4)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "messages": self.messages,
            "capacity_kwh": {
                "value": cap_kwh,
                "ci_low": None
                if self.capacity_ci_low is None
                else round(self.capacity_ci_low / 1000.0, 4),
                "ci_high": None
                if self.capacity_ci_high is None
                else round(self.capacity_ci_high / 1000.0, 4),
                "method": "tls_through_origin",
                "crosscheck_theil_sen_kwh": None
                if self.capacity_theil_sen_wh is None
                else round(self.capacity_theil_sen_wh / 1000.0, 4),
                "intercept_diag": self.capacity_intercept_diag,
            },
            "round_trip_efficiency": {
                "value": self.round_trip_efficiency,
                "ci_low": self.rte_ci_low,
                "ci_high": self.rte_ci_high,
                "method": "slope_ratio",
                "crosscheck_energy_balance": self.rte_crosscheck_energy_balance,
            },
            "eta_charge_symmetric": self.eta_symmetric,
            "eta_discharge_symmetric": self.eta_symmetric,
            "assumptions": {
                "symmetric_efficiency_split": True,
                "reported_soc_units": True,
            },
            "n_charge_segments": self.n_charge_segments,
            "n_discharge_segments": self.n_discharge_segments,
            "n_equal_soc_cycles": self.n_equal_soc_cycles,
        }


def _tls_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    """
    Total-least-squares slope through the origin (equal error variances, delta=1).

    Minimises perpendicular distance rather than vertical residual, so noise in
    the SoC channel (x) does not bias the slope the way ordinary least squares
    would. Closed form for the through-origin case.
    """
    sxx = float(np.sum(x * x))
    syy = float(np.sum(y * y))
    sxy = float(np.sum(x * y))
    if abs(sxy) < 1e-12:
        # Degenerate (no covariance); fall back to a ratio-of-means slope.
        return float(np.sum(y) / np.sum(x)) if np.sum(x) != 0 else float("nan")
    return ((syy - sxx) + np.sqrt((syy - sxx) ** 2 + 4.0 * sxy**2)) / (2.0 * sxy)


def _intercept_diag(x: np.ndarray, y: np.ndarray) -> float:
    """
    Diagnostic-only intercept from a TLS fit WITH intercept (first principal
    component of the centred cloud). A large intercept flags a fixed SoC/CT
    calibration offset that the through-origin slope would otherwise launder
    into the capacity estimate. We report it, we do not absorb it.
    """
    if len(x) < 2:
        return 0.0
    xm, ym = x.mean(), y.mean()
    xc, yc = x - xm, y - ym
    # Principal direction via the 2x2 covariance eigenvector.
    cov = np.cov(np.vstack([xc, yc]))
    if not np.all(np.isfinite(cov)):
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    vx, vy = eigvecs[:, int(np.argmax(eigvals))]
    if abs(vx) < 1e-12:
        return 0.0
    slope = vy / vx
    return float(ym - slope * xm)


class BatteryIdentification:
    """Pure estimator: history in, :class:`BatteryIdentificationResult` out."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    # -- segmentation ---------------------------------------------------------
    def _segment(self, df: pd.DataFrame, power_col: str, soc_col: str) -> list[_Segment]:
        """
        Split the series into monotonic charge/discharge runs.

        Robust to the two things that manufacture spurious segments: near-zero
        power flicker at the sign boundary (a ``POWER_DEADBAND_W`` deadband) and
        SoC quantisation jitter (a ``SOC_REVERSAL_HYSTERESIS`` counter-move
        tolerance before a run is considered to have reversed).
        """
        data = df[[power_col, soc_col]].dropna().sort_index()
        if len(data) < 3:
            return []
        power = data[power_col].to_numpy(dtype=float)
        soc = data[soc_col].to_numpy(dtype=float)
        times = data.index
        # Float hours since the first sample (tz-safe, resolution-independent).
        hours_axis = (times - times[0]).total_seconds().to_numpy() / 3600.0

        # Auto-detect the sign convention from the data itself. Each rising-SoC
        # interval [k, k+1] is CAUSED by the power held over it, power[k] (the
        # same convention the trapezoid integral uses), NOT power[k+1] - pairing
        # with the end sample mislabels pulsed charge-then-discharge data. Over
        # charging intervals the causing power must be positive; if the net vote
        # is negative the meter reports charge as negative and we flip so that
        # positive = charge.
        d_soc_step = np.diff(soc)
        rising = d_soc_step > 0
        if rising.any():
            charge_power_vote = float(np.sum(power[:-1][rising]))
            self.logger.debug(
                "battery_identification: sign vote=%.1f over %d rising steps",
                charge_power_vote,
                int(rising.sum()),
            )
            if charge_power_vote < 0:
                power = -power
                self.logger.debug("battery_identification: flipped power sign (positive=charge)")

        segments: list[_Segment] = []
        # Idle-aware direction per sample: +1 charge, -1 discharge, 0 idle.
        run_start = 0
        run_dir = 0  # current committed direction
        run_soc_extreme = soc[0]  # furthest SoC reached in the run's direction
        run_extreme_idx = 0  # index AT which that extreme was reached

        def _flush(i_end: int):
            if run_dir == 0:
                return
            seg = self._build_segment(power, soc, times, hours_axis, run_start, i_end, run_dir)
            if seg is not None:
                segments.append(seg)

        for i in range(1, len(soc)):
            step_dir = 0
            if abs(power[i]) >= POWER_DEADBAND_W:
                step_dir = 1 if power[i] > 0 else -1
            if run_dir == 0:
                if step_dir != 0:
                    run_dir = step_dir
                    run_start = i - 1
                    run_soc_extreme = soc[i - 1]
                    run_extreme_idx = i - 1
                continue
            # Extend the run while SoC keeps moving the committed way (allowing a
            # small counter-move within hysteresis without splitting). SoC lags
            # power by one interval, so a run truly ENDS at the SoC extreme, not
            # at the sample where the reversal is finally confirmed - flushing at
            # i-1 would fold an opposite-direction sample into this run and stretch
            # dSoC past the real turning point. Track the extreme index and split
            # there, then begin the next run at the same turning point.
            if run_dir == 1:
                if soc[i] > run_soc_extreme:
                    run_soc_extreme = soc[i]
                    run_extreme_idx = i
                reversed_ = soc[i] < run_soc_extreme - SOC_REVERSAL_HYSTERESIS
            else:
                if soc[i] < run_soc_extreme:
                    run_soc_extreme = soc[i]
                    run_extreme_idx = i
                reversed_ = soc[i] > run_soc_extreme + SOC_REVERSAL_HYSTERESIS
            if reversed_:
                _flush(run_extreme_idx)
                run_dir = step_dir
                run_start = run_extreme_idx
                run_soc_extreme = soc[run_extreme_idx]
        _flush(len(soc) - 1)
        return segments

    def _build_segment(
        self,
        power: np.ndarray,
        soc: np.ndarray,
        times: pd.DatetimeIndex,
        hours_axis: np.ndarray,
        i0: int,
        i1: int,
        direction_sign: int,
    ) -> _Segment | None:
        """Build one segment, keeping only deep, long-enough runs."""
        # Trim leading/trailing idle (sub-deadband) samples so the throughput
        # window and the SoC endpoints both bound the ACTIVE run: idle periods
        # must dilute neither the integral nor dSoC.
        while i0 < i1 and abs(power[i0]) < POWER_DEADBAND_W:
            i0 += 1
        while i1 > i0 and abs(power[i1]) < POWER_DEADBAND_W:
            i1 -= 1
        if i1 <= i0:
            return None
        d_soc = float(soc[i1] - soc[i0])
        duration = times[i1] - times[i0]
        if abs(d_soc) < MIN_SOC_SWING_PER_SEGMENT or duration < MIN_SEGMENT_DURATION:
            return None
        # Trapezoidal integral of AC power over real timestamps -> Wh.
        hours = np.diff(hours_axis[i0 : i1 + 1])
        p_seg = power[i0 : i1 + 1]
        signed_wh = float(np.sum((p_seg[:-1] + p_seg[1:]) / 2.0 * hours))
        direction = "charge" if direction_sign == 1 else "discharge"
        throughput = abs(signed_wh)
        return _Segment(
            direction=direction,
            d_soc=d_soc,
            throughput_wh=throughput,
            duration=duration,
            start=times[i0],
            end=times[i1],
            energy_in_wh=throughput if direction == "charge" else 0.0,
            energy_out_wh=throughput if direction == "discharge" else 0.0,
        )

    # -- fitting --------------------------------------------------------------
    def _fit_slopes(self, segments: list[_Segment]) -> tuple[float, float, float, float, float]:
        """
        Return (S_chg, S_dis, theil_sen_chg, theil_sen_dis, intercept_diag).

        ``S_chg``/``S_dis`` are TLS-through-origin slopes of throughput (Wh) vs
        |dSoC| (points). The Theil-Sen slopes are a robust median cross-check on
        a different residual structure (per-segment ratios), so correlated noise
        cannot drag both estimators into false agreement.
        """
        chg = [s for s in segments if s.direction == "charge"]
        dis = [s for s in segments if s.direction == "discharge"]
        x_c = np.array([abs(s.d_soc) for s in chg])
        y_c = np.array([s.throughput_wh for s in chg])
        x_d = np.array([abs(s.d_soc) for s in dis])
        y_d = np.array([s.throughput_wh for s in dis])
        s_chg = _tls_through_origin(x_c, y_c)
        s_dis = _tls_through_origin(x_d, y_d)
        ts_chg = float(np.median(y_c / x_c))
        ts_dis = float(np.median(y_d / x_d))
        # Diagnostic intercept on the combined slope structure (charge cloud).
        intercept = _intercept_diag(x_c, y_c)
        return s_chg, s_dis, ts_chg, ts_dis, intercept

    def _bootstrap_ci(
        self, segments: list[_Segment], n_resamples: int
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Block bootstrap over whole segments (never iid rows, which would
        understate the CI on an autocorrelated series). Returns
        ((cap_low, cap_high), (rte_low, rte_high)) as Wh and ratio.
        """
        chg = [s for s in segments if s.direction == "charge"]
        dis = [s for s in segments if s.direction == "discharge"]
        rng = np.random.default_rng(0)  # fixed seed: reproducible CIs across runs
        caps, rtes = [], []
        for _ in range(n_resamples):
            rc = rng.choice(len(chg), len(chg), replace=True)
            rd = rng.choice(len(dis), len(dis), replace=True)
            resampled = [chg[i] for i in rc] + [dis[i] for i in rd]
            s_chg, s_dis, *_ = self._fit_slopes(resampled)
            if s_chg <= 0 or s_dis <= 0:
                continue
            caps.append(100.0 * np.sqrt(s_chg * s_dis))
            rtes.append(s_dis / s_chg)
        if not caps:
            return (float("nan"), float("nan")), (float("nan"), float("nan"))
        cap_ci = (float(np.percentile(caps, 2.5)), float(np.percentile(caps, 97.5)))
        rte_ci = (float(np.percentile(rtes, 2.5)), float(np.percentile(rtes, 97.5)))
        return cap_ci, rte_ci

    def _energy_balance_rte(
        self, segments: list[_Segment], capacity_wh: float
    ) -> tuple[float | None, int]:
        """
        Independent RTE cross-check via the window energy balance
        ``eff^2 * E_in - eff * dE_stored - E_out = 0`` solved for eff = sqrt(RTE).

        This uses the capacity estimate (via ``dE_stored``), so it is weakly
        coupled to capacity error; it is a cross-check, not the primary RTE. It
        is only computed when there are at least ``MIN_EQUAL_SOC_CYCLES`` cycles
        AND the window is approximately closed (small net SoC drift relative to
        the total SoC swung), so an unbalanced tail cannot bias it.
        """
        n_cycles = min(
            sum(1 for s in segments if s.direction == "charge"),
            sum(1 for s in segments if s.direction == "discharge"),
        )
        if n_cycles < MIN_EQUAL_SOC_CYCLES:
            return None, n_cycles
        net_d_soc = sum(s.d_soc for s in segments)  # points
        total_swing = sum(abs(s.d_soc) for s in segments)
        if total_swing == 0 or abs(net_d_soc) > CLOSURE_TOLERANCE_FRACTION * total_swing:
            return None, n_cycles  # window not closed enough to trust the balance
        e_in = sum(s.energy_in_wh for s in segments)
        e_out = sum(s.energy_out_wh for s in segments)
        de_stored = capacity_wh * net_d_soc / 100.0
        if e_in <= 0 or e_out <= 0:
            return None, n_cycles
        disc = de_stored**2 + 4.0 * e_in * e_out
        if disc < 0:
            return None, n_cycles
        eff = (de_stored + np.sqrt(disc)) / (2.0 * e_in)
        return float(eff**2), n_cycles

    def identify(
        self,
        df: pd.DataFrame,
        power_col: str,
        soc_col: str,
        configured_capacity_wh: float,
    ) -> BatteryIdentificationResult:
        """
        Estimate usable capacity and round-trip efficiency from history.

        :param df: DataFrame indexed by timestamp with signed AC battery power
            and measured SoC (percent) columns.
        :param power_col: Name of the signed battery power column (W).
        :param soc_col: Name of the measured SoC column (percent).
        :param configured_capacity_wh: The user's currently configured
            ``battery_nominal_energy_capacity`` in Wh, used ONLY for the
            capacity sanity bound.
        :return: A :class:`BatteryIdentificationResult`; only ``status == "ok"``
            should be persisted or published.
        """
        segments = self._segment(df, power_col, soc_col)
        n_chg = sum(1 for s in segments if s.direction == "charge")
        n_dis = sum(1 for s in segments if s.direction == "discharge")
        if n_chg < MIN_DEEP_SEGMENTS or n_dis < MIN_DEEP_SEGMENTS:
            return BatteryIdentificationResult(
                status="insufficient_data",
                messages=[
                    f"Not enough deep segments to fit (charge={n_chg}, discharge={n_dis}, "
                    f"require >= {MIN_DEEP_SEGMENTS} each). Keeping configured battery values."
                ],
                n_charge_segments=n_chg,
                n_discharge_segments=n_dis,
            )

        s_chg, s_dis, ts_chg, ts_dis, intercept = self._fit_slopes(segments)
        result = BatteryIdentificationResult(
            status="ok",
            n_charge_segments=n_chg,
            n_discharge_segments=n_dis,
            capacity_intercept_diag=round(intercept, 3),
        )
        if not (s_chg > 0 and s_dis > 0 and ts_chg > 0 and ts_dis > 0):
            result.status = "rejected_sanity_check"
            result.messages.append("Non-positive slope estimate; keeping configured values.")
            return result

        capacity_wh = 100.0 * np.sqrt(s_chg * s_dis)
        rte = s_dis / s_chg
        cap_ts = 100.0 * np.sqrt(ts_chg * ts_dis)
        result.capacity_wh = capacity_wh
        result.capacity_theil_sen_wh = cap_ts
        result.round_trip_efficiency = round(rte, 4)

        # Cross-check divergence between TLS and Theil-Sen (different residual
        # structure). Wide disagreement means the estimate is not trustworthy.
        cap_divergence = abs(capacity_wh - cap_ts) / capacity_wh if capacity_wh else 1.0
        if cap_divergence > TLS_THEILSEN_MAX_DIVERGENCE:
            result.status = "low_confidence"
            result.messages.append(
                f"TLS vs Theil-Sen capacity diverge by {cap_divergence:.1%} "
                f"(> {TLS_THEILSEN_MAX_DIVERGENCE:.0%}); staying observe-only."
            )

        # Symmetric split: each leg gets sqrt(RTE) so they multiply back to RTE.
        if not (0.0 < rte <= 1.0):
            result.status = "rejected_sanity_check"
            result.messages.append(f"RTE={rte:.3f} outside (0, 1]; keeping configured values.")
            return result
        eta = float(np.sqrt(rte))
        result.eta_symmetric = round(eta, 4)

        # Physics sanity bounds. With a configured capacity we bound relative to
        # it; without one (Enom missing/0) we fall back to an absolute plausibility
        # band so the guardrail is never silently disabled.
        if configured_capacity_wh > 0:
            low = CAPACITY_SANITY_LOW * configured_capacity_wh
            high = CAPACITY_SANITY_HIGH * configured_capacity_wh
        else:
            low, high = ABS_CAPACITY_LOW_WH, ABS_CAPACITY_HIGH_WH
        if not (low <= capacity_wh <= high):
            result.status = "rejected_sanity_check"
            result.messages.append(
                f"Capacity {capacity_wh / 1000:.2f} kWh outside "
                f"[{low / 1000:.2f}, {high / 1000:.2f}] kWh sanity band; keeping configured value."
            )
            return result
        if not (SQRT_RTE_LOW <= eta <= SQRT_RTE_HIGH):
            result.status = "rejected_sanity_check"
            result.messages.append(
                f"One-way efficiency sqrt(RTE)={eta:.3f} outside "
                f"[{SQRT_RTE_LOW}, {SQRT_RTE_HIGH}]; keeping configured values."
            )
            return result

        # Confidence interval via block bootstrap.
        (cap_low, cap_high), (rte_low, rte_high) = self._bootstrap_ci(
            segments, BOOTSTRAP_N_RESAMPLES
        )
        finite_ci = np.isfinite(cap_low) and np.isfinite(cap_high)
        result.capacity_ci_low = cap_low if finite_ci else None
        result.capacity_ci_high = cap_high if finite_ci else None
        result.rte_ci_low = round(rte_low, 4) if np.isfinite(rte_low) else None
        result.rte_ci_high = round(rte_high, 4) if np.isfinite(rte_high) else None

        def _downgrade(msg: str) -> None:
            if result.status == "ok":
                result.status = "low_confidence"
            result.messages.append(msg)

        if not finite_ci or capacity_wh <= 0:
            # An undefined CI is the strongest signal of an untrustworthy fit -
            # fail CLOSED (withhold), never publish an estimate with no bound.
            _downgrade("Bootstrap CI is undefined; staying observe-only.")
        else:
            rel_ci = (cap_high - cap_low) / capacity_wh
            min_segs = min(n_chg, n_dis)
            if rel_ci > MAX_RELATIVE_CI:
                _downgrade(
                    f"Capacity CI width {rel_ci:.1%} exceeds {MAX_RELATIVE_CI:.0%}; "
                    "staying observe-only."
                )
            elif rel_ci < CI_IMPLAUSIBLY_TIGHT_REL and min_segs < CI_MIN_SEGMENTS_FOR_TIGHT:
                # Near-zero CI from too few near-identical cycles is false
                # confidence, not precision - do not let it reach 'suggest'.
                _downgrade(
                    f"Capacity CI implausibly tight ({rel_ci:.1e}) with only "
                    f"{min_segs} cycles per direction; staying observe-only."
                )

        # Independent energy-balance RTE cross-check.
        rte_eb, n_cycles = self._energy_balance_rte(segments, capacity_wh)
        result.rte_crosscheck_energy_balance = None if rte_eb is None else round(rte_eb, 4)
        result.n_equal_soc_cycles = n_cycles

        if result.status == "ok":
            result.messages.append(
                f"Identified capacity {capacity_wh / 1000:.2f} kWh "
                f"(CI [{cap_low / 1000:.2f}, {cap_high / 1000:.2f}]) and "
                f"round-trip efficiency {rte:.3f} (one-way sqrt {eta:.3f}) "
                f"from {n_chg} charge / {n_dis} discharge segments."
            )
        return result
