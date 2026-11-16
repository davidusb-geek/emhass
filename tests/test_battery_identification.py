#!/usr/bin/env python3

"""
Tests for opt-in battery self-identification (issue #963).

The estimator is pure (history in, result out), so ground truth is fully
controlled: synthetic charge/discharge cycles are generated from a KNOWN
capacity and round-trip efficiency, and the test asserts recovery within
tolerance. Base-safe: if the feature module is absent (running against the
base commit for a RED-on-base proof) the behavioural assertions fail rather
than erroring at import.
"""

import json
import logging
import os
import pathlib
import time
import unittest
import unittest.mock
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

try:
    from emhass import battery_identification as bi_module
    from emhass.battery_identification import (
        BatteryIdentification,
        BatteryIdentificationResult,
        apply_staleness_fill,
    )

    HAVE_FEATURE = True
except ImportError:  # base commit without the feature
    HAVE_FEATURE = False
    BatteryIdentification = None
    BatteryIdentificationResult = None
    apply_staleness_fill = None
    bi_module = None


def _make_history(
    capacity_wh: float,
    rte: float,
    n_cycles: int = 6,
    power_w: float = 3000.0,
    dt_minutes: int = 5,
    soc_low: float = 20.0,
    soc_high: float = 90.0,
    idle_steps: int = 6,
    noise_soc: float = 0.0,
    seed: int = 1,
) -> pd.DataFrame:
    """
    Generate signed-power + SoC history consistent with a given capacity and RTE.

    Sign convention: positive power = charging (into the battery). SoC is driven
    exactly by the physics so the estimator's recovered numbers can be checked
    against ``capacity_wh`` and ``rte``. An idle gap separates each half-cycle.
    """
    eta = float(np.sqrt(rte))
    dt_h = dt_minutes / 60.0
    rng = np.random.default_rng(seed)

    # Build the power schedule first: power[k] is held over the interval
    # [k, k+1]. SoC is then integrated interval-by-interval (n samples ->
    # n-1 increments), matching how the estimator integrates throughput, so a
    # clean fixture recovers ground truth to within discretisation only.
    per_charge_step = eta * power_w * dt_h / capacity_wh * 100.0
    per_disch_step = power_w * dt_h / (eta * capacity_wh) * 100.0
    n_charge = int(np.ceil((soc_high - soc_low) / per_charge_step))
    n_disch = int(np.ceil((soc_high - soc_low) / per_disch_step))
    powers: list[float] = []
    for _ in range(n_cycles):
        powers += [power_w] * n_charge  # charge run
        powers += [0.0] * idle_steps  # idle at top
        powers += [-power_w] * n_disch  # discharge run
        powers += [0.0] * idle_steps  # idle at bottom
    powers.append(0.0)  # final sample closes the last interval

    socs = [soc_low]
    for k in range(len(powers) - 1):
        p = powers[k]
        d_stored = eta * p * dt_h if p > 0 else p * dt_h / eta
        socs.append(socs[-1] + d_stored / capacity_wh * 100.0)
    n = len(powers)
    idx = pd.date_range("2026-01-01 00:00:00", periods=n, freq=f"{dt_minutes}min", tz="UTC")
    soc_arr = np.array(socs)
    if noise_soc > 0:
        soc_arr = soc_arr + rng.normal(0, noise_soc, n)
    return pd.DataFrame(
        {"sensor_power_battery": powers, "sensor_battery_state_of_charge": soc_arr}, index=idx
    )


def _make_multi_battery_df(specs: list[tuple[str, str, float, float]], **kwargs) -> pd.DataFrame:
    """
    Build one shared DataFrame holding N independent batteries' history.

    Each spec is ``(power_col, soc_col, capacity_wh, rte)``; ``_make_history``'s
    fixed column names are renamed per battery, then the frames are outer-join
    concatenated. Batteries with different cycle counts naturally produce
    different-length series; BatteryIdentification._segment does its own
    ``.dropna()`` on exactly the (power_col, soc_col) pair it is given, so the
    NaN padding an outer join introduces for a shorter battery is invisible to
    every other battery's fit.
    """
    frames = []
    for power_col, soc_col, capacity_wh, rte in specs:
        df = _make_history(capacity_wh, rte, **kwargs)
        df = df.rename(
            columns={
                "sensor_power_battery": power_col,
                "sensor_battery_state_of_charge": soc_col,
            }
        )
        frames.append(df)
    return pd.concat(frames, axis=1)


@unittest.skipUnless(HAVE_FEATURE, "battery_identification feature not present (base commit)")
class TestBatteryIdentification(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test_battery_id")
        self.bid = BatteryIdentification(self.logger)
        self.power_col = "sensor_power_battery"
        self.soc_col = "sensor_battery_state_of_charge"

    def _identify(self, df, configured_wh=10000.0):
        return self.bid.identify(df, self.power_col, self.soc_col, configured_wh)

    # -- acceptance slice: recover known ground truth --------------------------
    def test_recovers_known_capacity_and_rte(self):
        cap_true, rte_true = 10000.0, 0.90
        df = _make_history(cap_true, rte_true, n_cycles=6)
        res = self._identify(df)
        self.assertEqual(res.status, "ok", msg=str(res.messages))
        # Capacity within 3% of ground truth.
        self.assertAlmostEqual(res.capacity_wh, cap_true, delta=0.03 * cap_true)
        # RTE within 0.02 of ground truth.
        self.assertAlmostEqual(res.round_trip_efficiency, rte_true, delta=0.02)

    def test_symmetric_split_multiplies_back_to_rte(self):
        df = _make_history(12000.0, 0.88, n_cycles=6)
        res = self._identify(df, configured_wh=12000.0)
        self.assertEqual(res.status, "ok", msg=str(res.messages))
        # eta_ch == eta_dis == sqrt(RTE) and eta**2 == RTE (the pinned algebra).
        self.assertAlmostEqual(res.eta_symmetric**2, res.round_trip_efficiency, places=3)

    def test_different_capacity_value(self):
        cap_true, rte_true = 8000.0, 0.92
        df = _make_history(cap_true, rte_true, n_cycles=6)
        res = self._identify(df, configured_wh=8000.0)
        self.assertEqual(res.status, "ok", msg=str(res.messages))
        self.assertAlmostEqual(res.capacity_wh, cap_true, delta=0.03 * cap_true)

    # -- risk #1: segmentation robust to sign-boundary flicker ----------------
    def test_flicker_does_not_manufacture_segments(self):
        df = _make_history(10000.0, 0.90, n_cycles=6)
        clean = self._identify(df)
        # Inject sub-deadband flicker on the idle rows (power near zero).
        noisy = df.copy()
        idle_mask = noisy[self.power_col].abs() < 1.0
        rng = np.random.default_rng(3)
        noisy.loc[idle_mask, self.power_col] = rng.uniform(
            -20, 20, idle_mask.sum()
        )  # below POWER_DEADBAND_W
        res = self._identify(noisy)
        self.assertEqual(res.status, "ok", msg=str(res.messages))
        # Segment counts must be unchanged by flicker below the deadband.
        self.assertEqual(res.n_charge_segments, clean.n_charge_segments)
        self.assertEqual(res.n_discharge_segments, clean.n_discharge_segments)

    # -- guardrail: insufficient / shallow data does not publish --------------
    def test_insufficient_data_keeps_configured(self):
        # Only two shallow cycles, and shallow swings below MIN_SOC_SWING.
        df = _make_history(10000.0, 0.90, n_cycles=1, soc_low=50.0, soc_high=55.0)
        res = self._identify(df)
        self.assertNotEqual(res.status, "ok")
        self.assertIn(res.status, {"insufficient_data", "rejected_sanity_check", "low_confidence"})
        # Nothing publishable.
        self.assertFalse(res.is_ok)

    def test_capacity_outside_sanity_band_rejected(self):
        # True capacity 10 kWh but configured claims 3 kWh -> 10 kWh is > 1.5x -> reject.
        df = _make_history(10000.0, 0.90, n_cycles=6)
        res = self._identify(df, configured_wh=3000.0)
        self.assertEqual(res.status, "rejected_sanity_check")
        self.assertFalse(res.is_ok)

    def test_near_lossless_rte_rejected(self):
        # The #1069 case: change-only sampling produced a fit with RTE 0.991
        # (one-way sqrt 0.9955) on a pack independently metered at 0.83
        # round-trip efficiency. No real AC-coupled pack is that close to
        # lossless, so the efficiency guardrail must reject it rather than
        # publish. Asserting on the message pins the rejection to the
        # efficiency band, not the separate RTE > 1 gate.
        df = _make_history(10000.0, 0.991, n_cycles=6)
        res = self._identify(df)
        self.assertEqual(res.status, "rejected_sanity_check", msg=str(res.messages))
        self.assertFalse(res.is_ok)
        self.assertTrue(any("sqrt(RTE)" in m for m in res.messages), msg=str(res.messages))

    def test_plausible_high_rte_still_passes(self):
        # Counterfactual pin for the bound location: RTE 0.988 (one-way sqrt
        # 0.994) sits just below SQRT_RTE_HIGH and must still publish.
        df = _make_history(10000.0, 0.988, n_cycles=6)
        res = self._identify(df)
        self.assertEqual(res.status, "ok", msg=str(res.messages))
        self.assertTrue(res.is_ok)

    # -- #1072: same-source agreement near the efficiency ceiling --------------
    def test_near_ceiling_agreeing_crosscheck_is_low_confidence(self):
        # Same RTE 0.988 as above, but with enough closed cycles for the
        # energy-balance cross-check to run. On clean data it agrees with the
        # slope ratio, and since both come from the same samples that agreement
        # cannot vouch for a near-ceiling value: demote rather than publish.
        df = _make_history(10000.0, 0.988, n_cycles=8)
        res = self._identify(df)
        self.assertIsNotNone(res.rte_crosscheck_energy_balance)
        self.assertEqual(res.status, "low_confidence", msg=str(res.messages))
        self.assertFalse(res.is_ok)
        self.assertTrue(any("cross-check" in m for m in res.messages), msg=str(res.messages))

    def test_edge_band_is_on_one_way_efficiency(self):
        # RTE 0.975 is below 0.985 but its one-way sqrt (0.987) is above it.
        # The edge is defined on the one-way figure, so this must be held too.
        df = _make_history(10000.0, 0.975, n_cycles=8)
        res = self._identify(df)
        self.assertIsNotNone(res.rte_crosscheck_energy_balance)
        self.assertEqual(res.status, "low_confidence", msg=str(res.messages))

    def test_agreement_below_edge_band_still_publishes(self):
        # Counterfactual for the edge condition: RTE 0.96 (one-way 0.980) sits
        # below the band, where close agreement is just what a healthy fit
        # looks like, so the same fixture shape must still publish.
        df = _make_history(10000.0, 0.96, n_cycles=8)
        res = self._identify(df)
        self.assertIsNotNone(res.rte_crosscheck_energy_balance)
        self.assertEqual(res.status, "ok", msg=str(res.messages))

    def test_closeness_threshold_on_round_trip_gap(self):
        # Counterfactual for the closeness condition, on the RTE gap: 0.008
        # from the fitted 0.988 still counts as agreement and is held, 0.011
        # does not and is left to the other guardrails.
        df = _make_history(10000.0, 0.988, n_cycles=8)
        for rte_eb, expected in ((0.980, "low_confidence"), (0.977, "ok")):
            with self.subTest(rte_eb=rte_eb):
                bid = BatteryIdentification(self.logger)
                bid._energy_balance_rte = lambda segments, capacity_wh, v=rte_eb: (v, 8)
                res = bid.identify(df, self.power_col, self.soc_col, 10000.0)
                self.assertEqual(res.rte_crosscheck_energy_balance, rte_eb)
                self.assertEqual(res.status, expected, msg=str(res.messages))

    # -- default-no-op sanity: to_dict is JSON-serialisable scalars -----------
    def test_result_to_dict_is_scalar_json(self):
        import json

        df = _make_history(10000.0, 0.90, n_cycles=6)
        res = self._identify(df)
        payload = json.dumps(res.to_dict())  # must not raise
        self.assertIn("capacity_kwh", payload)
        self.assertIn("round_trip_efficiency", payload)

    # -- sign-convention auto-detect is robust to either polarity -------------
    def test_sign_convention_invariance(self):
        cap_true, rte_true = 10000.0, 0.90
        df = _make_history(cap_true, rte_true, n_cycles=8)
        res_pos = self._identify(df)
        # Negate the power channel (the opposite meter convention); auto-detect
        # must recover the same numbers, not an inverted RTE.
        flipped = df.copy()
        flipped[self.power_col] = -flipped[self.power_col]
        res_neg = self._identify(flipped)
        self.assertEqual(res_pos.status, "ok")
        self.assertEqual(res_neg.status, "ok")
        self.assertAlmostEqual(res_pos.capacity_wh, res_neg.capacity_wh, delta=1.0)
        self.assertAlmostEqual(
            res_pos.round_trip_efficiency, res_neg.round_trip_efficiency, places=4
        )

    def test_pulsed_charge_does_not_spuriously_flip(self):
        # Short charge pulses each followed by discharge is the pattern that a
        # wrong power/SoC pairing mislabels. Correctly signed (positive=charge),
        # it must recover RTE < 1, never an inverted 1/RTE.
        df = _make_history(10000.0, 0.90, n_cycles=8, power_w=6000.0, idle_steps=1)
        res = self._identify(df)
        self.assertEqual(res.status, "ok", msg=str(res.messages))
        self.assertLessEqual(res.round_trip_efficiency, 1.0)
        self.assertAlmostEqual(res.round_trip_efficiency, 0.90, delta=0.03)

    # -- risk #6: degenerate near-zero CI at the segment floor is NOT trusted --
    def test_degenerate_ci_at_floor_is_low_confidence(self):
        # Exactly the 3-segment floor of identical cycles -> zero-width CI. That
        # is false confidence, so it must stay observe-only, not reach 'ok'.
        df = _make_history(10000.0, 0.90, n_cycles=3)
        res = self._identify(df)
        self.assertEqual(res.status, "low_confidence", msg=str(res.messages))
        self.assertFalse(res.is_ok)

    # -- risk #3: an undefined (NaN) CI fails CLOSED, never publishes ----------
    def test_undefined_ci_fails_closed(self):
        df = _make_history(10000.0, 0.90, n_cycles=8)

        def _nan_ci(segments, n):
            return (float("nan"), float("nan")), (float("nan"), float("nan"))

        self.bid._bootstrap_ci = _nan_ci
        res = self._identify(df)
        self.assertNotEqual(res.status, "ok")
        self.assertFalse(res.is_ok)

    # -- tolerance to modest SoC quantisation noise ---------------------------
    def test_recovery_under_soc_noise(self):
        cap_true, rte_true = 10000.0, 0.90
        df = _make_history(cap_true, rte_true, n_cycles=8, noise_soc=0.5)
        res = self._identify(df)
        # Should still fit; capacity within a looser 6% under noise.
        self.assertIn(res.status, {"ok", "low_confidence"})
        if res.capacity_wh is not None:
            self.assertAlmostEqual(res.capacity_wh, cap_true, delta=0.06 * cap_true)


try:
    from emhass import command_line

    HAVE_CL = True
except ImportError:
    HAVE_CL = False


class _FakeRH:
    """Records published sensors; never touches the network."""

    def __init__(self):
        self.published = {}
        self.get_data_from_file = False

    async def post_scalar_sensor(self, entity_id, state, attributes):
        self.published[entity_id] = {"state": state, "attributes": attributes}
        return True


@unittest.skipUnless(
    HAVE_FEATURE and HAVE_CL, "battery_identification/command_line not present (base commit)"
)
class TestIdentifyBatteryOrchestrator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import tempfile

        self.logger = logging.getLogger("test_identify_battery")
        self.tmp = tempfile.mkdtemp()
        self.emhass_conf = {"data_path": pathlib.Path(self.tmp)}
        self.retrieve_hass_conf = {
            "sensor_power_battery": "sensor_power_battery",
            "sensor_battery_state_of_charge": "sensor_battery_state_of_charge",
        }
        self.plant_conf = {
            "battery_nominal_energy_capacity": 10000,
            "battery_charge_efficiency": 0.95,
            "battery_discharge_efficiency": 0.95,
        }
        self.df_good = _make_history(10000.0, 0.90, n_cycles=6)

    def _patch_retrieve(self, monkey_df, success=True):
        async def _fake_retrieve(*args, **kwargs):
            # args[2] is retrieve_hass_conf, mutated in place around this call
            # by _identify_battery_impl at N>1; snapshot what was actually
            # presented (copy, since it gets restored right after this
            # returns) so tests can pin the retrieval-shape contract.
            retrieve_hass_conf_arg = args[2]
            power_val = retrieve_hass_conf_arg.get("sensor_power_battery")
            soc_val = retrieve_hass_conf_arg.get("sensor_battery_state_of_charge")
            self.last_retrieve_sensor_config = (
                list(power_val) if isinstance(power_val, list) else power_val,
                list(soc_val) if isinstance(soc_val, list) else soc_val,
            )
            return success, monkey_df, None

        command_line.retrieve_home_assistant_data = _fake_retrieve

    async def _run(self, optim_conf, df=None, success=True):
        orig = command_line.retrieve_home_assistant_data
        self.last_retrieve_sensor_config = None
        self._patch_retrieve(self.df_good if df is None else df, success=success)
        rh = _FakeRH()
        try:
            await command_line.identify_battery(
                self.logger,
                optim_conf,
                self.plant_conf,
                self.retrieve_hass_conf,
                rh,
                self.emhass_conf,
                False,
                "test_df_final.pkl",
            )
        finally:
            command_line.retrieve_home_assistant_data = orig
        return rh

    def _json_path(self):
        return self.emhass_conf["data_path"] / "battery_identification.json"

    async def test_observe_writes_json_no_publish_no_plant_mutation(self):
        plant_before = dict(self.plant_conf)
        rh = await self._run({"battery_identification_trust_tier": "observe"})
        self.assertTrue(self._json_path().exists(), "observe must persist a JSON")
        self.assertEqual(rh.published, {}, "observe must NOT publish HA sensors")
        # v1 never mutates plant_conf.
        self.assertEqual(self.plant_conf, plant_before)

        payload = json.loads(self._json_path().read_text())
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["trust_tier"], "observe")

    async def test_suggest_publishes_two_sensors(self):
        with self.assertLogs("test_identify_battery", level="INFO") as cm:
            rh = await self._run({"battery_identification_trust_tier": "suggest"})
        self.assertIn("sensor.battery_identified_capacity", rh.published)
        self.assertIn("sensor.battery_identified_round_trip_efficiency", rh.published)
        # Attributes carry the CI and last-fit time.
        cap = rh.published["sensor.battery_identified_capacity"]["attributes"]
        self.assertIn("ci_low", cap)
        self.assertIsNotNone(cap["fitted_at"])
        # The docs promise the suggest tier logs a recommendation.
        self.assertTrue(any("recommendation" in m.lower() for m in cm.output))

    async def test_failed_fit_records_last_attempt_and_preserves_existing_record(self):
        # Seed an existing (stale-looking) file to prove it is not clobbered.
        self._json_path().write_text('{"status": "ok", "marker": "keep-me"}')
        # Shallow data -> insufficient -> non-ok.
        shallow = _make_history(10000.0, 0.90, n_cycles=1, soc_low=50.0, soc_high=54.0)
        # Force a re-fit by setting max age to 0.
        rh = await self._run(
            {
                "battery_identification_trust_tier": "suggest",
                "battery_identification_model_max_age": 0,
            },
            df=shallow,
        )

        payload = json.loads(self._json_path().read_text())
        self.assertEqual(
            payload.get("marker"), "keep-me", "failed fit must not discard the retained record"
        )
        self.assertEqual(rh.published, {}, "failed fit must not publish")
        self.assertEqual(payload.get("status"), "ok", "retained ok record must survive")
        attempt = payload["last_attempt"]
        self.assertEqual(attempt["status"], "insufficient_data")
        self.assertTrue(attempt["messages"], "the failure reason must be recorded")
        datetime.fromisoformat(attempt["attempted_at"])  # parses, same format as fitted_at

    async def test_n1_no_prior_fit_failure_writes_failure_only_payload_and_backs_off(self):
        """No prior file: a failed fit writes a failure-only payload (no
        top-level status, so it can never be mistaken for a fit result) and
        the very next run - still inside max_age - must serve the recorded
        backoff without retrieving history again. RED-on-base: base writes
        nothing on a failed fit, so this file would never even exist."""
        shallow = _make_history(10000.0, 0.90, n_cycles=1, soc_low=50.0, soc_high=54.0)
        rh1 = await self._run({"battery_identification_trust_tier": "suggest"}, df=shallow)
        self.assertEqual(rh1.published, {}, "a failed fit must not publish")
        self.assertTrue(self._json_path().exists(), "a failed fit must record the attempt")

        payload = json.loads(self._json_path().read_text())
        self.assertNotIn(
            "status", payload, "a failure-only payload must not look like a fit result"
        )
        attempt = payload["last_attempt"]
        self.assertEqual(attempt["status"], "insufficient_data")
        self.assertTrue(attempt["messages"], "the failure reason must be recorded")
        datetime.fromisoformat(attempt["attempted_at"])  # parses, same format as fitted_at

        rh2 = await self._run({"battery_identification_trust_tier": "suggest"}, df=shallow)
        self.assertIsNone(
            self.last_retrieve_sensor_config,
            "a second run inside max_age must not retrieve history for a fresh fit",
        )
        self.assertEqual(rh2.published, {})

    async def test_n1_backoff_expires_and_refits(self):
        """After the failure-only payload's mtime is aged past max_age (the
        mechanism the backoff rides on, since N=1 freshness is mtime-only via
        is_model_outdated), the next run must retrieve and re-fit rather than
        serving the stale backoff forever."""
        shallow = _make_history(10000.0, 0.90, n_cycles=1, soc_low=50.0, soc_high=54.0)
        rh1 = await self._run(
            {
                "battery_identification_trust_tier": "suggest",
                "battery_identification_model_max_age": 1,
            },
            df=shallow,
        )
        self.assertEqual(rh1.published, {})
        json_path = self._json_path()
        self.assertTrue(json_path.exists(), "a failed fit must record the attempt")
        aged = time.time() - 3 * 3600  # 3h old, past the 1h max_age
        os.utime(json_path, (aged, aged))

        rh2 = await self._run(
            {
                "battery_identification_trust_tier": "suggest",
                "battery_identification_model_max_age": 1,
            },
            df=self.df_good,
        )
        self.assertIsNotNone(
            self.last_retrieve_sensor_config,
            "an expired backoff must retrieve history for a fresh fit",
        )
        payload = json.loads(json_path.read_text())
        self.assertEqual(payload.get("status"), "ok", "the expired backoff's re-fit succeeded")
        self.assertIn("sensor.battery_identified_capacity", rh2.published)

    async def test_n1_success_after_failure_clears_last_attempt(self):
        """A later successful fit replaces the whole file, which is how a run
        of recorded failures clears automatically once the fit succeeds
        again."""
        shallow = _make_history(10000.0, 0.90, n_cycles=1, soc_low=50.0, soc_high=54.0)
        await self._run(
            {
                "battery_identification_trust_tier": "suggest",
                "battery_identification_model_max_age": 0,
            },
            df=shallow,
        )
        payload_after_fail = json.loads(self._json_path().read_text())
        self.assertIn("last_attempt", payload_after_fail)

        rh = await self._run(
            {
                "battery_identification_trust_tier": "suggest",
                "battery_identification_model_max_age": 0,
            },
            df=self.df_good,
        )
        payload_after_success = json.loads(self._json_path().read_text())
        self.assertEqual(payload_after_success.get("status"), "ok")
        self.assertNotIn(
            "last_attempt",
            payload_after_success,
            "a successful fit must clear any previously recorded failed attempt",
        )
        self.assertIn("sensor.battery_identified_capacity", rh.published)

    async def test_n1_retained_ok_keeps_being_served_across_failed_attempts(self):
        """A successful fit, then a failed attempt (which must not stop the
        retained estimate being served), then a THIRD run inside max_age
        that must serve the retained ok record without a re-fetch: once a
        battery has a good record, a later failing attempt must not make
        EMHASS go silent - it keeps serving the last known-good estimate
        while it quietly retries the fit in the background."""
        rh1 = await self._run(
            {
                "battery_identification_trust_tier": "suggest",
                "battery_identification_model_max_age": 24,
            },
            df=self.df_good,
        )
        self.assertIn("sensor.battery_identified_capacity", rh1.published)
        payload1 = json.loads(self._json_path().read_text())
        self.assertEqual(payload1.get("status"), "ok")

        shallow = _make_history(10000.0, 0.90, n_cycles=1, soc_low=50.0, soc_high=54.0)
        rh2 = await self._run(
            {
                "battery_identification_trust_tier": "suggest",
                "battery_identification_model_max_age": 0,
            },
            df=shallow,
        )
        self.assertEqual(rh2.published, {}, "the failing attempt itself must not publish")
        payload2 = json.loads(self._json_path().read_text())
        self.assertEqual(payload2.get("status"), "ok", "retained ok record must survive")
        self.assertIn("last_attempt", payload2)

        rh3 = await self._run(
            {
                "battery_identification_trust_tier": "suggest",
                "battery_identification_model_max_age": 24,
            },
            df=shallow,  # irrelevant: must never be retrieved
        )
        self.assertIsNone(
            self.last_retrieve_sensor_config,
            "a backed-off failure must not suppress serving the retained ok record",
        )
        self.assertIn(
            "sensor.battery_identified_capacity",
            rh3.published,
            "the retained ok record must keep being published across failed attempts",
        )

    async def test_missing_sensor_columns_is_graceful(self):
        bad = self.df_good.rename(columns={"sensor_power_battery": "something_else"})
        rh = await self._run(
            {
                "battery_identification_trust_tier": "suggest",
                "battery_identification_model_max_age": 0,
            },
            df=bad,
        )
        self.assertEqual(rh.published, {})
        self.assertFalse(self._json_path().exists())

    async def test_n2_config_skips_cleanly_with_warning(self):
        """number_of_batteries=2 but sensor_power_battery/
        sensor_battery_state_of_charge are still plain scalars (today's only
        supported shape). One sensor cannot identify two independent
        batteries, and there is deliberately no scalar broadcast for these two
        keys, so identification must skip with one clear warning naming the
        offending key instead of guessing or crashing."""
        multi_conf = dict(self.plant_conf)
        multi_conf["number_of_batteries"] = 2
        multi_conf["battery_nominal_energy_capacity"] = [10000, 12000]
        multi_conf["battery_charge_efficiency"] = [0.95, 0.9]
        original_plant_conf = self.plant_conf
        self.plant_conf = multi_conf
        try:
            with self.assertLogs("test_identify_battery", level="WARNING") as cm:
                rh = await self._run({"battery_identification_trust_tier": "suggest"})
        finally:
            self.plant_conf = original_plant_conf
        msg = " ".join(cm.output)
        self.assertIn("sensor_power_battery", msg, "must name the offending key")
        self.assertIn("number_of_batteries=2", msg, "must name the battery count")
        self.assertFalse(
            any("TypeError" in m for m in cm.output),
            "must skip cleanly, not swallow a TypeError as a generic failure",
        )
        self.assertEqual(rh.published, {}, "unresolved N>1 config must publish nothing")
        self.assertFalse(
            self._json_path().exists(), "unresolved N>1 config must not write a fit result"
        )

    async def test_n2_wrong_length_list_skips_with_warning_naming_key(self):
        """A list is present but the wrong length: still a skip, still one
        warning naming which key and its got-vs-expected length."""
        self.retrieve_hass_conf["sensor_power_battery"] = ["only_one_entity"]
        self.retrieve_hass_conf["sensor_battery_state_of_charge"] = ["soc_0", "soc_1"]
        multi_conf = dict(self.plant_conf)
        multi_conf["number_of_batteries"] = 2
        multi_conf["battery_nominal_energy_capacity"] = [10000, 12000]
        multi_conf["battery_charge_efficiency"] = [0.95, 0.9]
        self.plant_conf = multi_conf
        with self.assertLogs("test_identify_battery", level="WARNING") as cm:
            rh = await self._run({"battery_identification_trust_tier": "suggest"})
        msg = " ".join(cm.output)
        self.assertIn("sensor_power_battery", msg)
        self.assertIn("length 1", msg, "must name the got length")
        self.assertIn("number_of_batteries=2", msg, "must name the expected count")
        self.assertEqual(rh.published, {})
        self.assertFalse(self._json_path().exists())

    async def test_n2_scalar_sensor_key_skips_with_warning_no_broadcast(self):
        """One key resolved to a valid list, the other stayed a scalar: the
        warning must name the specific bad key, not a generic N>1 message."""
        self.retrieve_hass_conf["sensor_power_battery"] = "single_entity"
        self.retrieve_hass_conf["sensor_battery_state_of_charge"] = ["soc_0", "soc_1"]
        multi_conf = dict(self.plant_conf)
        multi_conf["number_of_batteries"] = 2
        multi_conf["battery_nominal_energy_capacity"] = [10000, 12000]
        multi_conf["battery_charge_efficiency"] = [0.95, 0.9]
        self.plant_conf = multi_conf
        with self.assertLogs("test_identify_battery", level="WARNING") as cm:
            rh = await self._run({"battery_identification_trust_tier": "suggest"})
        msg = " ".join(cm.output)
        self.assertIn("sensor_power_battery", msg)
        self.assertIn("single_entity", msg, "must name the got value")
        self.assertIn("number_of_batteries=2", msg)
        self.assertEqual(rh.published, {})
        self.assertFalse(self._json_path().exists())

    async def test_n2_duplicate_id_within_list_skips_with_warning(self):
        """Two batteries pointed at the same meter is exactly the case this
        feature exists to reject: one sensor cannot identify two packs."""
        self.retrieve_hass_conf["sensor_power_battery"] = ["shared_meter", "shared_meter"]
        self.retrieve_hass_conf["sensor_battery_state_of_charge"] = ["soc_0", "soc_1"]
        multi_conf = dict(self.plant_conf)
        multi_conf["number_of_batteries"] = 2
        multi_conf["battery_nominal_energy_capacity"] = [10000, 12000]
        multi_conf["battery_charge_efficiency"] = [0.95, 0.9]
        self.plant_conf = multi_conf
        with self.assertLogs("test_identify_battery", level="WARNING") as cm:
            rh = await self._run({"battery_identification_trust_tier": "suggest"})
        msg = " ".join(cm.output)
        self.assertIn("sensor_power_battery", msg)
        self.assertIn("shared_meter", msg, "must name the duplicated id")
        self.assertEqual(rh.published, {})
        self.assertFalse(self._json_path().exists())

    async def test_n2_non_string_list_entry_skips_with_warning(self):
        """A None (or otherwise non-string) list entry must skip precisely,
        not degrade into a generic wrapper crash further downstream."""
        self.retrieve_hass_conf["sensor_power_battery"] = ["p0", None]
        self.retrieve_hass_conf["sensor_battery_state_of_charge"] = ["soc_0", "soc_1"]
        multi_conf = dict(self.plant_conf)
        multi_conf["number_of_batteries"] = 2
        multi_conf["battery_nominal_energy_capacity"] = [10000, 12000]
        multi_conf["battery_charge_efficiency"] = [0.95, 0.9]
        self.plant_conf = multi_conf
        with self.assertLogs("test_identify_battery", level="WARNING") as cm:
            rh = await self._run({"battery_identification_trust_tier": "suggest"})
        msg = " ".join(cm.output)
        self.assertIn("sensor_power_battery", msg)
        self.assertIn("[1]", msg, "must name the offending index")
        self.assertFalse(
            any("TypeError" in m for m in cm.output),
            "must skip cleanly, not degrade into a generic wrapper crash",
        )
        self.assertEqual(rh.published, {})
        self.assertFalse(self._json_path().exists())

    async def test_n2_cross_list_overlap_skips_with_warning(self):
        """One entity id used for both battery 0's power AND battery 1's SOC:
        one entity cannot be both signals."""
        self.retrieve_hass_conf["sensor_power_battery"] = ["p0", "shared_id"]
        self.retrieve_hass_conf["sensor_battery_state_of_charge"] = ["shared_id", "soc_1"]
        multi_conf = dict(self.plant_conf)
        multi_conf["number_of_batteries"] = 2
        multi_conf["battery_nominal_energy_capacity"] = [10000, 12000]
        multi_conf["battery_charge_efficiency"] = [0.95, 0.9]
        self.plant_conf = multi_conf
        with self.assertLogs("test_identify_battery", level="WARNING") as cm:
            rh = await self._run({"battery_identification_trust_tier": "suggest"})
        msg = " ".join(cm.output)
        self.assertIn("shared_id", msg, "must name the overlapping id")
        self.assertEqual(rh.published, {})
        self.assertFalse(self._json_path().exists())

    def _set_n2_config(self, power_cols, soc_cols):
        self.retrieve_hass_conf["sensor_power_battery"] = power_cols
        self.retrieve_hass_conf["sensor_battery_state_of_charge"] = soc_cols
        multi_conf = dict(self.plant_conf)
        multi_conf["number_of_batteries"] = 2
        multi_conf["battery_nominal_energy_capacity"] = [10000, 10000]
        multi_conf["battery_charge_efficiency"] = [0.95, 0.95]
        multi_conf["battery_discharge_efficiency"] = [0.95, 0.95]
        self.plant_conf = multi_conf

    async def test_n2_exact_lists_publishes_per_battery_sensors(self):
        """RED-on-base: on base, N>1 skips unconditionally before ever
        retrieving/fitting/publishing (rh.published stays {} no matter the
        config), so this behavioural assertion fails on base and only passes
        with #1042's per-battery orchestrator loop."""
        self._set_n2_config(["p0", "p1"], ["soc0", "soc1"])
        df = _make_multi_battery_df(
            [("p0", "soc0", 10000.0, 0.90), ("p1", "soc1", 10000.0, 0.90)], n_cycles=6
        )
        rh = await self._run({"battery_identification_trust_tier": "suggest"}, df=df)
        self.assertEqual(
            set(rh.published.keys()),
            {
                "sensor.battery_identified_capacity_battery0",
                "sensor.battery_identified_round_trip_efficiency_battery0",
                "sensor.battery_identified_capacity_battery1",
                "sensor.battery_identified_round_trip_efficiency_battery1",
            },
        )

        payload = json.loads(self._json_path().read_text())
        self.assertEqual(payload.get("schema_version"), 2)
        self.assertEqual(payload["batteries"]["0"]["status"], "ok")
        self.assertEqual(payload["batteries"]["1"]["status"], "ok")

    async def test_n2_partial_failure_independence(self):
        """k=0 has good data and fits; k=1 has insufficient data. k=0's fit
        must persist and publish, k=1 is recorded as a failure-only slot (not
        omitted), and the run must not raise. A later successful fit for k=1
        then clears its recorded last_attempt, the same success-replaces-the-
        whole-slot clearing behaviour proven for N=1 elsewhere in this file."""
        self._set_n2_config(["p0", "p1"], ["soc0", "soc1"])
        good = _make_history(10000.0, 0.90, n_cycles=6).rename(
            columns={"sensor_power_battery": "p0", "sensor_battery_state_of_charge": "soc0"}
        )
        shallow = _make_history(10000.0, 0.90, n_cycles=1, soc_low=50.0, soc_high=54.0).rename(
            columns={"sensor_power_battery": "p1", "sensor_battery_state_of_charge": "soc1"}
        )
        df = pd.concat([good, shallow], axis=1)
        rh = await self._run({"battery_identification_trust_tier": "suggest"}, df=df)
        self.assertIn("sensor.battery_identified_capacity_battery0", rh.published)
        self.assertNotIn("sensor.battery_identified_capacity_battery1", rh.published)

        payload = json.loads(self._json_path().read_text())
        self.assertIn("0", payload["batteries"])
        self.assertIn("1", payload["batteries"], "the failed battery's attempt must be recorded")
        entry1 = payload["batteries"]["1"]
        self.assertNotIn("status", entry1, "failure-only slot must not look like a fit result")
        self.assertNotIn("fitted_at", entry1)
        self.assertNotIn("capacity_kwh", entry1)
        self.assertEqual(entry1["last_attempt"]["status"], "insufficient_data")
        self.assertEqual(entry1["last_attempt"]["sensors"], {"power": "p1", "soc": "soc1"})
        self.assertTrue(entry1["last_attempt"]["messages"])

        # A later successful fit for battery 1 clears its last_attempt.
        good1 = _make_history(10000.0, 0.90, n_cycles=6).rename(
            columns={"sensor_power_battery": "p1", "sensor_battery_state_of_charge": "soc1"}
        )
        df2 = pd.concat([good, good1], axis=1)
        rh2 = await self._run(
            {
                "battery_identification_trust_tier": "suggest",
                "battery_identification_model_max_age": 0,
            },
            df=df2,
        )
        payload2 = json.loads(self._json_path().read_text())
        self.assertEqual(payload2["batteries"]["1"]["status"], "ok")
        self.assertNotIn("last_attempt", payload2["batteries"]["1"])
        self.assertIn("sensor.battery_identified_capacity_battery1", rh2.published)

    async def test_n2_failing_battery_backs_off_and_is_excluded_from_next_retrieval(self):
        """A battery whose fit fails gains an in-slot last_attempt and, on
        the next cycle inside max_age, is excluded from the retrieval batch
        entirely (the acceptance slice: retries at max_age cadence, not
        every run) while its fresh sibling keeps being served from cache."""
        self._set_n2_config(["p0", "p1"], ["soc0", "soc1"])
        good0 = _make_history(10000.0, 0.90, n_cycles=6).rename(
            columns={"sensor_power_battery": "p0", "sensor_battery_state_of_charge": "soc0"}
        )
        shallow1 = _make_history(10000.0, 0.90, n_cycles=1, soc_low=50.0, soc_high=54.0).rename(
            columns={"sensor_power_battery": "p1", "sensor_battery_state_of_charge": "soc1"}
        )
        df1 = pd.concat([good0, shallow1], axis=1)
        await self._run({"battery_identification_trust_tier": "suggest"}, df=df1)

        payload = json.loads(self._json_path().read_text())
        self.assertIn("1", payload["batteries"], "the failed battery's attempt must be recorded")
        entry1 = payload["batteries"]["1"]
        self.assertNotIn("status", entry1)
        self.assertIn("last_attempt", entry1)

        # Second run: k=0 stays fresh and k=1's failure is still inside
        # max_age, so NEITHER battery is stale - no retrieval batch should
        # run at all (the existing M1 pin: only stale batteries' sensors are
        # ever presented to retrieval).
        rh2 = await self._run({"battery_identification_trust_tier": "suggest"}, df=df1)
        self.assertIsNone(
            self.last_retrieve_sensor_config,
            "neither battery is stale, so no retrieval batch should run at all",
        )
        self.assertIn("sensor.battery_identified_capacity_battery0", rh2.published)
        self.assertNotIn(
            "sensor.battery_identified_capacity_battery1",
            rh2.published,
            "a backed-off failure-only entry must never be published",
        )

    async def test_n2_sensor_change_on_failure_only_entry_forces_immediate_retry(self):
        """A failure-only slot's last_attempt records the sensor pair it was
        attempted against; if that pair changes, the backoff must not apply -
        the sensor-pair override (invariant 6) extends to a failure-only
        entry, not just a retained ok record."""
        self._set_n2_config(["p0", "p1"], ["soc0", "soc1"])
        good0 = _make_history(10000.0, 0.90, n_cycles=6).rename(
            columns={"sensor_power_battery": "p0", "sensor_battery_state_of_charge": "soc0"}
        )
        shallow1 = _make_history(10000.0, 0.90, n_cycles=1, soc_low=50.0, soc_high=54.0).rename(
            columns={"sensor_power_battery": "p1", "sensor_battery_state_of_charge": "soc1"}
        )
        df1 = pd.concat([good0, shallow1], axis=1)
        await self._run({"battery_identification_trust_tier": "suggest"}, df=df1)

        # Re-point battery 1 at a different sensor pair.
        self.retrieve_hass_conf["sensor_power_battery"] = ["p0", "p1_new"]
        self.retrieve_hass_conf["sensor_battery_state_of_charge"] = ["soc0", "soc1_new"]
        good1_new = _make_history(10000.0, 0.90, n_cycles=6).rename(
            columns={
                "sensor_power_battery": "p1_new",
                "sensor_battery_state_of_charge": "soc1_new",
            }
        )
        df2 = pd.concat([good0, good1_new], axis=1)
        rh2 = await self._run({"battery_identification_trust_tier": "suggest"}, df=df2)
        self.assertEqual(
            self.last_retrieve_sensor_config,
            (["p1_new"], ["soc1_new"]),
            "the sensor-pair change must force battery 1's retry against the NEW pair, "
            "not be suppressed by the old pair's backoff",
        )
        self.assertIn("sensor.battery_identified_capacity_battery1", rh2.published)

    async def test_n2_wrong_pair_last_attempt_never_publishes_old_sensor_capacity(self):
        """The single worst silent-wrong-result class this feature could
        introduce. A slot can hold a retained ok record fitted against an
        OLD sensor pair (A) plus a fresh, well-formed last_attempt against
        the CURRENTLY resolved pair (B) - e.g. after a sensor-list edit
        followed by a failed re-fit. _battery_fit_is_stale correctly treats
        this as not-stale (backed off against B), but the not-stale publish
        branch must NOT serve A's retained capacity as if it were B's."""
        self._set_n2_config(["p0", "pB"], ["soc0", "socB"])
        now = datetime.now(UTC)
        seed = {
            "schema_version": 2,
            "batteries": {
                "0": {
                    "status": "ok",
                    "marker": "keep-k0",
                    "fitted_at": now.isoformat(),
                    "sensors": {"power": "p0", "soc": "soc0"},
                    "capacity_kwh": {"value": 10.0},
                    "round_trip_efficiency": {"value": 0.9},
                },
                "1": {
                    "status": "ok",
                    "fitted_at": now.isoformat(),
                    "sensors": {"power": "pA", "soc": "socA"},
                    "capacity_kwh": {"value": 999.0},
                    "round_trip_efficiency": {"value": 0.5},
                    "last_attempt": {
                        "status": "insufficient_data",
                        "attempted_at": now.isoformat(),
                        "messages": ["not enough cycles"],
                        "sensors": {"power": "pB", "soc": "socB"},
                    },
                },
            },
        }
        self._json_path().write_text(json.dumps(seed))
        rh = await self._run({"battery_identification_trust_tier": "suggest"}, df=self.df_good)
        self.assertIsNone(
            self.last_retrieve_sensor_config,
            "battery 0 is fresh and battery 1 is backed off against the current pair B, "
            "so no retrieval batch should run at all",
        )
        self.assertIn("sensor.battery_identified_capacity_battery0", rh.published)
        self.assertNotIn(
            "sensor.battery_identified_capacity_battery1",
            rh.published,
            "battery A's retained capacity must never be published as battery B's",
        )

    async def test_n2_per_battery_freshness_only_stale_battery_refits(self):
        """A fresh k=0 entry must be reused untouched while a stale k=1 entry
        refits, proving freshness is keyed per battery, not per file mtime."""
        self._set_n2_config(["p0", "p1"], ["soc0", "soc1"])

        now = datetime.now(UTC)
        seed = {
            "schema_version": 2,
            "batteries": {
                "0": {
                    "status": "ok",
                    "marker": "keep-k0",
                    "fitted_at": now.isoformat(),
                    "sensors": {"power": "p0", "soc": "soc0"},
                    "capacity_kwh": {"value": 999.0},
                    "round_trip_efficiency": {"value": 0.5},
                },
                "1": {
                    "status": "ok",
                    "marker": "stale-k1",
                    "fitted_at": (now - timedelta(hours=48)).isoformat(),
                    "sensors": {"power": "p1", "soc": "soc1"},
                    "capacity_kwh": {"value": 1.0},
                    "round_trip_efficiency": {"value": 0.1},
                },
            },
        }
        self._json_path().write_text(json.dumps(seed))
        good0 = _make_history(10000.0, 0.90, n_cycles=6).rename(
            columns={"sensor_power_battery": "p0", "sensor_battery_state_of_charge": "soc0"}
        )
        good1 = _make_history(10000.0, 0.90, n_cycles=6).rename(
            columns={"sensor_power_battery": "p1", "sensor_battery_state_of_charge": "soc1"}
        )
        df = pd.concat([good0, good1], axis=1)
        rh = await self._run({"battery_identification_trust_tier": "suggest"}, df=df)
        payload = json.loads(self._json_path().read_text())
        self.assertEqual(
            payload["batteries"]["0"].get("marker"), "keep-k0", "fresh battery 0 must not refit"
        )
        self.assertNotIn(
            "marker", payload["batteries"]["1"], "stale battery 1 must have been refit"
        )
        self.assertEqual(payload["batteries"]["1"]["status"], "ok")
        self.assertEqual(
            rh.published["sensor.battery_identified_capacity_battery0"]["state"],
            999.0,
            "battery 0's published value must come from the cached entry, unrefit",
        )
        # M1 pin: the retrieval must have presented ONLY the stale battery's
        # sensors, not the full lists - battery 0's fresh sensors never enter
        # the batch, so a bad/unreachable sensor for battery 0 could never
        # affect battery 1's re-fit (and vice versa).
        self.assertEqual(self.last_retrieve_sensor_config, (["p1"], ["soc1"]))

    async def test_n2_swapped_sensor_lists_within_max_age_refit_correctly_attributed(self):
        """F2 pin: swapping the sensor lists (a legitimate re-indexing) while
        still inside battery_identification_model_max_age must re-fit both
        batteries against their NEW sensor pairing, not silently reuse the OLD
        index-keyed cached values (which would misattribute one pack's result
        to the other)."""
        self._set_n2_config(["p0", "p1"], ["soc0", "soc1"])
        self.plant_conf["battery_nominal_energy_capacity"] = [8000, 14000]
        df1 = _make_multi_battery_df(
            [("p0", "soc0", 8000.0, 0.90), ("p1", "soc1", 14000.0, 0.90)], n_cycles=6
        )
        rh1 = await self._run({"battery_identification_trust_tier": "suggest"}, df=df1)
        self.assertAlmostEqual(
            rh1.published["sensor.battery_identified_capacity_battery0"]["state"], 8.0, delta=0.3
        )
        self.assertAlmostEqual(
            rh1.published["sensor.battery_identified_capacity_battery1"]["state"], 14.0, delta=0.5
        )

        # Swap: index 0 now reads p1's sensors, index 1 now reads p0's.
        self.retrieve_hass_conf["sensor_power_battery"] = ["p1", "p0"]
        self.retrieve_hass_conf["sensor_battery_state_of_charge"] = ["soc1", "soc0"]
        self.plant_conf["battery_nominal_energy_capacity"] = [14000, 8000]
        df2 = _make_multi_battery_df(
            [("p1", "soc1", 14000.0, 0.90), ("p0", "soc0", 8000.0, 0.90)], n_cycles=6
        )
        rh2 = await self._run({"battery_identification_trust_tier": "suggest"}, df=df2)
        self.assertAlmostEqual(
            rh2.published["sensor.battery_identified_capacity_battery0"]["state"],
            14.0,
            delta=0.5,
            msg="index 0 now reads p1's data; must not serve the old p0-fitted cached value",
        )
        self.assertAlmostEqual(
            rh2.published["sensor.battery_identified_capacity_battery1"]["state"],
            8.0,
            delta=0.3,
            msg="index 1 now reads p0's data; must not serve the old p1-fitted cached value",
        )

    async def test_n2_corrupt_non_dict_entry_only_that_battery_refits(self):
        """F3 pin: a corrupted (non-dict) entry for one battery must not abort
        the whole cycle - the healthy fresh sibling still serves its cached
        result, and only the corrupt battery refits."""
        self._set_n2_config(["p0", "p1"], ["soc0", "soc1"])

        now = datetime.now(UTC)
        seed = {
            "schema_version": 2,
            "batteries": {
                "0": "garbage",
                "1": {
                    "status": "ok",
                    "marker": "keep-k1",
                    "fitted_at": now.isoformat(),
                    "sensors": {"power": "p1", "soc": "soc1"},
                    "capacity_kwh": {"value": 777.0},
                    "round_trip_efficiency": {"value": 0.5},
                },
            },
        }
        self._json_path().write_text(json.dumps(seed))
        df = _make_multi_battery_df(
            [("p0", "soc0", 10000.0, 0.90), ("p1", "soc1", 10000.0, 0.90)], n_cycles=6
        )
        rh = await self._run({"battery_identification_trust_tier": "suggest"}, df=df)
        # No global abort: battery 1's fresh cached entry still publishes untouched.
        self.assertEqual(
            rh.published["sensor.battery_identified_capacity_battery1"]["state"], 777.0
        )
        payload = json.loads(self._json_path().read_text())
        self.assertEqual(payload["batteries"]["1"].get("marker"), "keep-k1")
        # Battery 0's corrupt entry must have been refit, not left as "garbage".
        self.assertIsInstance(payload["batteries"]["0"], dict)
        self.assertEqual(payload["batteries"]["0"]["status"], "ok")
        self.assertIn("sensor.battery_identified_capacity_battery0", rh.published)

    async def test_n2_corrupt_int_fitted_at_only_that_battery_refits(self):
        """F3 pin, variant: a non-string fitted_at (TypeError from
        datetime.fromisoformat) must degrade the same way as any other
        corrupt entry - that battery refits, the healthy sibling is
        unaffected, and the scan never raises out of the per-k comprehension."""
        self._set_n2_config(["p0", "p1"], ["soc0", "soc1"])

        now = datetime.now(UTC)
        seed = {
            "schema_version": 2,
            "batteries": {
                "0": {
                    "status": "ok",
                    "fitted_at": 12345,
                    "sensors": {"power": "p0", "soc": "soc0"},
                    "capacity_kwh": {"value": 1.0},
                    "round_trip_efficiency": {"value": 0.1},
                },
                "1": {
                    "status": "ok",
                    "marker": "keep-k1",
                    "fitted_at": now.isoformat(),
                    "sensors": {"power": "p1", "soc": "soc1"},
                    "capacity_kwh": {"value": 777.0},
                    "round_trip_efficiency": {"value": 0.5},
                },
            },
        }
        self._json_path().write_text(json.dumps(seed))
        df = _make_multi_battery_df(
            [("p0", "soc0", 10000.0, 0.90), ("p1", "soc1", 10000.0, 0.90)], n_cycles=6
        )
        rh = await self._run({"battery_identification_trust_tier": "suggest"}, df=df)
        self.assertEqual(
            rh.published["sensor.battery_identified_capacity_battery1"]["state"], 777.0
        )
        payload = json.loads(self._json_path().read_text())
        self.assertEqual(payload["batteries"]["1"].get("marker"), "keep-k1")
        self.assertNotEqual(payload["batteries"]["0"].get("fitted_at"), 12345)
        self.assertIn("sensor.battery_identified_capacity_battery0", rh.published)

    async def test_n2_flat_v1_file_is_treated_as_absent_refits_all(self):
        """A flat v1 file left behind (e.g. after reverting number_of_batteries
        from 1 back to 2) must be treated as absent, not partially parsed."""
        self._set_n2_config(["p0", "p1"], ["soc0", "soc1"])
        self._json_path().write_text('{"status": "ok", "marker": "flat-v1"}')
        df = _make_multi_battery_df(
            [("p0", "soc0", 10000.0, 0.90), ("p1", "soc1", 10000.0, 0.90)], n_cycles=6
        )
        rh = await self._run({"battery_identification_trust_tier": "suggest"}, df=df)

        payload = json.loads(self._json_path().read_text())
        self.assertEqual(payload.get("schema_version"), 2)
        self.assertIn("0", payload["batteries"])
        self.assertIn("1", payload["batteries"])
        self.assertIn("sensor.battery_identified_capacity_battery0", rh.published)

    async def test_n1_v2_container_leftover_is_treated_as_absent_refits_flat(self):
        """A v2 container left behind (e.g. after reverting number_of_batteries
        from 2 back to 1) must be treated as absent, refit, and rewritten as
        the flat v1 shape."""

        seed = {"schema_version": 2, "batteries": {"0": {"status": "ok", "marker": "v2-leftover"}}}
        self._json_path().write_text(json.dumps(seed))
        rh = await self._run({"battery_identification_trust_tier": "suggest"})
        payload = json.loads(self._json_path().read_text())
        self.assertEqual(payload["status"], "ok")
        self.assertNotIn(
            "batteries", payload, "N=1 must rewrite the flat v1 shape, not keep the v2 container"
        )
        self.assertIn("sensor.battery_identified_capacity", rh.published)

    async def test_n1_failed_fit_does_not_destroy_a_v2_container_left_behind(self):
        """A v2 container left behind by a reverted number_of_batteries (e.g.
        2 -> 1), still holding at least one retained ok slot, must not be
        destroyed by a FAILED N=1 fit. Falling through to a re-fit every run
        (as the read-side v2-leftover test above accepts) is fine; silently
        wiping a retained ok slot is an invariant-2 violation, not an
        acceptable N-transition cost."""
        seed = {
            "schema_version": 2,
            "batteries": {
                "0": {"status": "ok", "marker": "keep-k0", "capacity_kwh": {"value": 10.0}},
                "1": {"status": "ok", "marker": "keep-k1", "capacity_kwh": {"value": 12.0}},
            },
        }
        self._json_path().write_text(json.dumps(seed))
        shallow = _make_history(10000.0, 0.90, n_cycles=1, soc_low=50.0, soc_high=54.0)
        rh = await self._run({"battery_identification_trust_tier": "suggest"}, df=shallow)
        self.assertEqual(rh.published, {}, "a failed fit must not publish")
        payload = json.loads(self._json_path().read_text())
        self.assertEqual(
            payload, seed, "a failed N=1 fit must not touch a foreign v2 container at all"
        )

    async def test_n1_failed_fit_overwrites_a_failure_only_v2_container(self):
        """The mirror case of the test above: a v2 container left behind by a
        reverted number_of_batteries whose slots are ALL failure-only (both
        batteries were failing before the revert) holds nothing worth
        protecting. A failed N=1 fit must overwrite it with the normal N=1
        failure-only payload and engage the backoff, not skip forever with no
        self-clearing path - skipping here would just re-create the every-run
        re-pull this feature exists to fix, since only a SUCCESSFUL N=1 fit
        would ever clear a protective skip, and that can't happen while the
        fit keeps failing."""
        seed = {
            "schema_version": 2,
            "batteries": {
                "0": {"last_attempt": {"status": "insufficient_data", "messages": ["x"]}},
                "1": {"last_attempt": {"status": "insufficient_data", "messages": ["y"]}},
            },
        }
        self._json_path().write_text(json.dumps(seed))
        shallow = _make_history(10000.0, 0.90, n_cycles=1, soc_low=50.0, soc_high=54.0)
        rh1 = await self._run({"battery_identification_trust_tier": "suggest"}, df=shallow)
        self.assertEqual(rh1.published, {})
        payload = json.loads(self._json_path().read_text())
        self.assertNotIn(
            "batteries", payload, "the failure-only container must be overwritten, not skipped"
        )
        self.assertIn("last_attempt", payload)

        rh2 = await self._run({"battery_identification_trust_tier": "suggest"}, df=shallow)
        self.assertIsNone(
            self.last_retrieve_sensor_config,
            "the backoff must engage on the very next run, not skip forever",
        )
        self.assertEqual(rh2.published, {})

    async def test_n2_failed_fits_do_not_destroy_a_v1_payload_left_behind(self):
        """Mirror direction: a flat v1 ok payload left behind by a reverted
        number_of_batteries (e.g. 1 -> 2) must not be destroyed by FAILED N>1
        fits, for either battery. The success path already treats a v1
        leftover as absent and rewrites it (unchanged, invariant 1) - only a
        FAILURE must refuse to touch it."""
        self._set_n2_config(["p0", "p1"], ["soc0", "soc1"])
        seed_text = json.dumps(
            {"status": "ok", "marker": "flat-v1", "capacity_kwh": {"value": 10.0}}
        )
        self._json_path().write_text(seed_text)
        shallow0 = _make_history(10000.0, 0.90, n_cycles=1, soc_low=50.0, soc_high=54.0).rename(
            columns={"sensor_power_battery": "p0", "sensor_battery_state_of_charge": "soc0"}
        )
        shallow1 = _make_history(10000.0, 0.90, n_cycles=1, soc_low=50.0, soc_high=54.0).rename(
            columns={"sensor_power_battery": "p1", "sensor_battery_state_of_charge": "soc1"}
        )
        df = pd.concat([shallow0, shallow1], axis=1)
        rh = await self._run({"battery_identification_trust_tier": "suggest"}, df=df)
        self.assertEqual(rh.published, {}, "failed fits must not publish")
        self.assertEqual(
            self._json_path().read_text(),
            seed_text,
            "failed N>1 fits must not touch a foreign v1 payload at all, byte-identical",
        )

    async def test_n1_publish_is_exact_todays_two_entities_pin(self):
        """N=1 regression pin: exactly today's two entity ids, and a flat JSON
        with no "batteries" key."""
        rh = await self._run({"battery_identification_trust_tier": "suggest"})
        self.assertEqual(
            set(rh.published.keys()),
            {
                "sensor.battery_identified_capacity",
                "sensor.battery_identified_round_trip_efficiency",
            },
        )

        payload = json.loads(self._json_path().read_text())
        self.assertNotIn("batteries", payload)

    async def test_n1_list_of_one_sensor_config_equivalent_to_bare_string(self):
        """M3 pin: param_definitions.json declares the two sensor keys
        array.string, so the config UI now saves a length-1 list instead of a
        bare string (CONTRACT.md's SCOPE NOTE on invariant 1). Both shapes
        must take a re-fit to the SAME published entities, the SAME flat JSON
        shape, and the SAME log wording."""
        opts = {
            "battery_identification_trust_tier": "suggest",
            "battery_identification_model_max_age": 0,  # force a real re-fit each run
        }
        with self.assertLogs("test_identify_battery", level="INFO") as cm_str:
            rh_str = await self._run(opts)
        payload_str = json.loads(self._json_path().read_text())
        # Delete before the second run so both runs see "no file yet" - with
        # max_age=0 the JSON's mere existence changes is_model_outdated's log
        # wording (a real, but here irrelevant, difference), independent of
        # whether the sensor config is a bare string or a list of one.
        self._json_path().unlink()

        # The config UI's saved shape: a length-1 list instead of a bare string.
        self.retrieve_hass_conf["sensor_power_battery"] = ["sensor_power_battery"]
        self.retrieve_hass_conf["sensor_battery_state_of_charge"] = [
            "sensor_battery_state_of_charge"
        ]
        with self.assertLogs("test_identify_battery", level="INFO") as cm_list:
            rh_list = await self._run(opts)
        payload_list = json.loads(self._json_path().read_text())

        self.assertEqual(set(rh_str.published.keys()), set(rh_list.published.keys()))
        self.assertEqual(
            {k: v["state"] for k, v in rh_str.published.items()},
            {k: v["state"] for k, v in rh_list.published.items()},
            "published sensor values must match between a bare-string and a list-of-one config",
        )
        self.assertNotIn("batteries", payload_str)
        self.assertNotIn("batteries", payload_list)
        self.assertEqual(payload_str.keys(), payload_list.keys())
        for key in payload_str:
            if key == "fitted_at":
                continue  # legitimately differs: real wall-clock time of each run
            self.assertEqual(
                payload_str[key],
                payload_list[key],
                f"JSON field {key!r} differs between the bare-string and list-of-one configs",
            )
        self.assertEqual(
            cm_str.output,
            cm_list.output,
            "log wording must be identical between the bare-string and list-of-one configs",
        )

    async def test_n1_misshapen_list_config_with_fresh_cache_still_publishes(self):
        """F4 pin: at N=1 a misshapen sensor_power_battery (e.g. a length-2
        list, invalid at N=1) must not suppress a fresh cached publish - base
        never read the sensor keys on the cache-hit path, and neither must
        this: the resolver only runs on the refit path."""

        seed = {
            "status": "ok",
            "fitted_at": datetime.now(UTC).isoformat(),
            "capacity_kwh": {"value": 42.0, "ci_low": None, "ci_high": None},
            "round_trip_efficiency": {"value": 0.9},
        }
        self._json_path().write_text(json.dumps(seed))
        self.retrieve_hass_conf["sensor_power_battery"] = ["a", "b"]  # invalid at N=1
        rh = await self._run({"battery_identification_trust_tier": "suggest"})
        self.assertEqual(
            rh.published["sensor.battery_identified_capacity"]["state"],
            42.0,
            "cached publish must be served without ever consulting the misshapen sensor config",
        )


@unittest.skipUnless(HAVE_CL, "command_line not present (base commit)")
class TestIsModelOutdatedLabel(unittest.TestCase):
    """The new label param must NOT change the existing adjusted-PV log text."""

    def test_default_label_preserves_pv_wording(self):
        missing = pathlib.Path("/nonexistent/never_here.pkl")
        logger = logging.getLogger("test_label_pv")
        with self.assertLogs("test_label_pv", level="INFO") as cm:
            command_line.is_model_outdated(missing, 24, logger)
        self.assertTrue(any("Adjusted PV model" in m for m in cm.output))

    def test_custom_label_used_for_battery(self):
        missing = pathlib.Path("/nonexistent/never_here.json")
        logger = logging.getLogger("test_label_batt")
        with self.assertLogs("test_label_batt", level="INFO") as cm:
            command_line.is_model_outdated(
                missing, 24, logger, label="Battery identification model"
            )
        self.assertTrue(any("Battery identification model" in m for m in cm.output))


@unittest.skipUnless(HAVE_FEATURE, "battery_identification feature not present (base commit)")
class TestGapHandling(unittest.TestCase):
    """
    Recorder gaps must contribute zero throughput and zero dSoC (#1051).

    After the retrieval resample, missing history becomes NaN buckets which the
    segmentation's dropna removes, leaving one oversized step spanning the gap.
    On the base commit the trapezoid integrates AC power across that step, so a
    multi-day gap fabricates p_avg * gap_hours of throughput (the reported 79 h
    outage became an 89,809 Wh "discharge segment" on a 10 kWh pack). The
    tests that avoid the ``time_step`` kwarg are base-safe: they drive the
    estimator through calls whose signatures exist on base too, so their RED
    assertions fail behaviourally there. The ones passing ``time_step``
    exercise fix-only surface and error on base instead.
    """

    def setUp(self):
        self.logger = logging.getLogger("test_battery_id_gaps")
        self.power_col = "sensor_power_battery"
        self.soc_col = "sensor_battery_state_of_charge"

    @staticmethod
    def _two_block_charge(
        gap_hours: float = 79.0,
        dt_minutes: int = 30,
        n_side: int = 10,
        power_w: float = 1000.0,
        cap_wh: float = 20000.0,
        rte: float = 0.90,
    ) -> pd.DataFrame:
        """
        Two charge blocks separated by a silent stretch with SoC flat across it:
        the recorder died mid-charge and came back while the battery (which sat
        idle in between) was charging again. The sample on each side of the gap
        carries active charge power, exactly the bridged shape from the issue.
        """
        eta = float(np.sqrt(rte))
        dt_h = dt_minutes / 60.0
        step_pts = eta * power_w * dt_h / cap_wh * 100.0
        idx1 = pd.date_range(
            "2026-01-01 00:00:00", periods=n_side, freq=f"{dt_minutes}min", tz="UTC"
        )
        idx2 = pd.date_range(
            idx1[-1] + pd.Timedelta(hours=gap_hours),
            periods=n_side,
            freq=f"{dt_minutes}min",
            tz="UTC",
        )
        soc1 = [20.0 + step_pts * k for k in range(n_side)]
        soc2 = [soc1[-1] + step_pts * k for k in range(n_side)]
        return pd.DataFrame(
            {
                "sensor_power_battery": [power_w] * (2 * n_side),
                "sensor_battery_state_of_charge": soc1 + soc2,
            },
            index=idx1.append(idx2),
        )

    # -- the #1051 bug itself (RED on base) -----------------------------------
    def test_gap_contributes_zero_throughput(self):
        """A 79 h recorder gap must add nothing to segment throughput."""
        df = self._two_block_charge()
        segs = BatteryIdentification(self.logger)._segment(df, self.power_col, self.soc_col)
        total_throughput = sum(s.throughput_wh for s in segs)
        # Ground truth: 9 half-hour intervals per block at 1000 W = 4500 Wh per
        # block, 9000 Wh measured in total. On base the bridged trapezoid adds
        # 1000 W * 79 h = 79,000 Wh on top, so this bound fails behaviourally.
        self.assertLess(
            total_throughput,
            9000.0 * 1.10,
            "segment throughput must not exceed the energy actually measured "
            f"(got {total_throughput:.0f} Wh)",
        )
        for s in segs:
            self.assertLess(
                s.end - s.start,
                pd.Timedelta(hours=79),
                "no segment may span the recorder gap",
            )

    def test_gap_contributes_zero_dsoc(self):
        """The run ends at the gap: unobserved SoC change is never credited."""
        df = self._two_block_charge()
        segs = BatteryIdentification(self.logger)._segment(df, self.power_col, self.soc_col)
        self.assertTrue(segs, "each observed block is deep enough to segment")
        eta = float(np.sqrt(0.90))
        per_block_swing = 9 * (eta * 1000.0 * 0.5 / 20000.0 * 100.0)
        for s in segs:
            # On base the single bridged segment carries both blocks' swing
            # (double this bound), so this fails behaviourally there.
            self.assertLessEqual(
                abs(s.d_soc),
                per_block_swing + 0.1,
                "a segment must not carry SoC change from beyond the gap",
            )

    def test_identify_recovers_truth_despite_gap(self):
        """End to end: a huge mid-run gap no longer corrupts the fit."""
        cap_true, rte_true = 10000.0, 0.90
        df = _make_history(cap_true, rte_true, n_cycles=20, dt_minutes=30, idle_steps=4)
        # Cut from mid-charge-run to mid-charge-run roughly 79 h later, so the
        # samples on BOTH sides of the gap carry active charge power and the
        # bridged trapezoid on base fabricates ~3 kW * ~80 h of throughput.
        run_rows = df[df[self.power_col] > 0]
        mid = run_rows.index[len(run_rows) // 2 + 2]
        later = run_rows.index[run_rows.index >= mid + pd.Timedelta(hours=79)]
        end = later[2]
        mask = (df.index >= mid) & (df.index < end)
        gapped = df.loc[~mask]
        res = BatteryIdentification(self.logger).identify(
            gapped, self.power_col, self.soc_col, cap_true
        )
        self.assertEqual(res.status, "ok", msg=str(res.messages))
        self.assertAlmostEqual(res.capacity_wh, cap_true, delta=0.05 * cap_true)
        self.assertAlmostEqual(res.round_trip_efficiency, rte_true, delta=0.03)

    # -- the fix's own risk: splitting must not eat healthy profiles ----------
    def test_idle_gaps_between_cycles_keep_segments(self):
        """
        The change-only profile from the issue: dense reporting while cycling,
        silent while idle. Gaps sit BETWEEN runs, so excluding them must not
        cost any segments and the fit must stay accurate.
        """
        cap_true, rte_true = 10000.0, 0.90
        df = _make_history(cap_true, rte_true, n_cycles=6, dt_minutes=30, idle_steps=12)
        # Silence every idle stretch (power == 0) except one edge sample on
        # each side, as a change-only sensor would.
        idle = df[self.power_col] == 0.0
        edges = idle & (~idle.shift(1, fill_value=False) | ~idle.shift(-1, fill_value=False))
        gapped = df[~idle | edges]
        res = BatteryIdentification(self.logger).identify(
            gapped, self.power_col, self.soc_col, cap_true
        )
        self.assertEqual(res.status, "ok", msg=str(res.messages))
        self.assertAlmostEqual(res.capacity_wh, cap_true, delta=0.05 * cap_true)

    def test_sub_threshold_dropout_still_bridged(self):
        """Up to two consecutive missing samples is jitter, not a gap: no split."""
        df = _make_history(10000.0, 0.90, n_cycles=1, dt_minutes=30, idle_steps=4)
        run_rows = df[df[self.power_col] > 0]
        mid_pos = len(run_rows) // 2
        # Drop exactly two consecutive in-run samples -> dt == 3x step, at the
        # documented tolerance boundary.
        to_drop = run_rows.index[mid_pos : mid_pos + 2]
        jittered = df.drop(index=to_drop)
        segs_ref = BatteryIdentification(self.logger)._segment(df, self.power_col, self.soc_col)
        segs_jit = BatteryIdentification(self.logger)._segment(
            jittered, self.power_col, self.soc_col
        )
        self.assertEqual(
            len(segs_jit),
            len(segs_ref),
            "a two-sample dropout must not split the run",
        )
        # Depth must survive too: a >=-threshold variant would split here and
        # the surviving half would shrink dSoC while the count stays equal.
        for ref, jit in zip(segs_ref, segs_jit):
            self.assertEqual(jit.direction, ref.direction)
            self.assertAlmostEqual(jit.d_soc, ref.d_soc, places=6)

    def test_sub_threshold_dropout_bridges_on_odd_grids(self):
        """
        The 3x tolerance must hold on grids whose step is not a binary
        fraction of an hour (float-hours comparison misclassified exactly-3x
        dropouts at 1/2/3/6/7/13 min steps; spacing is compared in integer
        nanoseconds precisely so this cannot happen).
        """
        for dt_minutes in (1, 3, 7, 13):
            df = _make_history(
                10000.0, 0.90, n_cycles=1, power_w=300.0, dt_minutes=dt_minutes, idle_steps=4
            )
            run_rows = df[df[self.power_col] > 0]
            mid_pos = len(run_rows) // 2
            to_drop = run_rows.index[mid_pos : mid_pos + 2]
            jittered = df.drop(index=to_drop)
            with self.subTest(dt_minutes=dt_minutes):
                segs_ref = BatteryIdentification(
                    self.logger, time_step=pd.Timedelta(minutes=dt_minutes)
                )._segment(df, self.power_col, self.soc_col)
                segs_jit = BatteryIdentification(
                    self.logger, time_step=pd.Timedelta(minutes=dt_minutes)
                )._segment(jittered, self.power_col, self.soc_col)
                self.assertEqual(
                    [(s.direction, round(s.d_soc, 6)) for s in segs_jit],
                    [(s.direction, round(s.d_soc, 6)) for s in segs_ref],
                    f"exactly-3x dropout must stay bridged at {dt_minutes} min",
                )

    def test_discharge_gap_contributes_nothing(self):
        """Same guarantee for the discharge direction: two discharge blocks."""
        eta = float(np.sqrt(0.90))
        step_pts = 1000.0 * 0.5 / (eta * 20000.0) * 100.0
        idx1 = pd.date_range("2026-01-01 00:00:00", periods=10, freq="30min", tz="UTC")
        idx2 = pd.date_range(idx1[-1] + pd.Timedelta(hours=79), periods=10, freq="30min", tz="UTC")
        soc1 = [90.0 - step_pts * k for k in range(10)]
        soc2 = [soc1[-1] - step_pts * k for k in range(10)]
        df = pd.DataFrame(
            {
                "sensor_power_battery": [-1000.0] * 20,
                "sensor_battery_state_of_charge": soc1 + soc2,
            },
            index=idx1.append(idx2),
        )
        segs = BatteryIdentification(self.logger)._segment(df, self.power_col, self.soc_col)
        self.assertEqual(len(segs), 2, "one discharge segment per observed block")
        for s in segs:
            self.assertEqual(s.direction, "discharge")
            self.assertLess(s.end - s.start, pd.Timedelta(hours=79))
        self.assertLess(
            sum(s.throughput_wh for s in segs),
            9000.0 * 1.10,
            "the bridged 79 h discharge trapezoid must not be counted",
        )

    def test_gap_at_series_edges_is_harmless(self):
        """A lone pre-history or post-history sample must change nothing."""
        df = _make_history(10000.0, 0.90, n_cycles=2, dt_minutes=30, idle_steps=4)
        lone_before = pd.DataFrame(
            {self.power_col: [3000.0], self.soc_col: [50.0]},
            index=pd.DatetimeIndex([df.index[0] - pd.Timedelta(hours=79)]),
        )
        lone_after = pd.DataFrame(
            {self.power_col: [-3000.0], self.soc_col: [30.0]},
            index=pd.DatetimeIndex([df.index[-1] + pd.Timedelta(hours=79)]),
        )
        padded = pd.concat([lone_before, df, lone_after])
        segs_ref = BatteryIdentification(self.logger)._segment(df, self.power_col, self.soc_col)
        segs_pad = BatteryIdentification(self.logger)._segment(padded, self.power_col, self.soc_col)
        self.assertEqual(
            [(s.direction, round(s.d_soc, 6)) for s in segs_pad],
            [(s.direction, round(s.d_soc, 6)) for s in segs_ref],
            "lone samples across an edge gap must contribute nothing",
        )

    def test_gap_detected_on_microsecond_index(self):
        """
        Gap detection must survive a non-nanosecond index resolution: pandas 3
        builds datetime64[us] indexes by default, where asi8 counts
        microseconds while Timedelta.value is nanoseconds. A unit mix-up makes
        the threshold 1000x too big and silently disables gap handling under
        an explicit time_step (the production configuration).
        """
        df = self._two_block_charge()
        df.index = df.index.as_unit("us")
        segs = BatteryIdentification(self.logger, time_step=pd.Timedelta(minutes=30))._segment(
            df, self.power_col, self.soc_col
        )
        self.assertEqual(len(segs), 2, "the 79 h gap must still split on a us-unit index")
        self.assertLess(sum(s.throughput_wh for s in segs), 9000.0 * 1.10)

    def test_explicit_time_step_overrides_inference(self):
        """
        With an explicit time_step the gap threshold follows the configured
        step, not the data spacing: a 65 min hole in 5 min data is a gap when
        inferring (threshold 15 min) but tolerated jitter at a 30 min
        configured step (threshold 90 min).
        """
        df = _make_history(10000.0, 0.90, n_cycles=1, power_w=1500.0, dt_minutes=5, idle_steps=4)
        run_rows = df[df[self.power_col] > 0]
        mid = run_rows.index[len(run_rows) // 2]
        mask = (df.index >= mid) & (df.index < mid + pd.Timedelta(minutes=60))
        gapped = df.loc[~mask]
        inferred = BatteryIdentification(self.logger)._segment(gapped, self.power_col, self.soc_col)
        explicit = BatteryIdentification(self.logger, time_step=pd.Timedelta(minutes=30))._segment(
            gapped, self.power_col, self.soc_col
        )
        n_charge_inferred = sum(1 for s in inferred if s.direction == "charge")
        n_charge_explicit = sum(1 for s in explicit if s.direction == "charge")
        self.assertEqual(n_charge_explicit, 1, "65 min hole within 3x30 min must bridge")
        self.assertEqual(n_charge_inferred, 2, "65 min hole beyond 3x5 min must split")

    def test_gap_steps_do_not_vote_on_sign_convention(self):
        """
        A gap step must not contribute its (stale) left-sample power to the
        sign auto-detection vote. Here the meter reports charge as negative,
        the battery started a hard discharge right before the recorder died,
        and SoC came back higher. Unmasked, that one stale high-power vote
        outweighs every real charging step and the convention flips the wrong
        way; masked, the real steps win.
        """
        step_pts = float(np.sqrt(0.90)) * 1000.0 * 0.5 / 20000.0 * 100.0
        idx1 = pd.date_range("2026-01-01 00:00:00", periods=11, freq="30min", tz="UTC")
        idx2 = pd.date_range(idx1[-1] + pd.Timedelta(hours=79), periods=10, freq="30min", tz="UTC")
        # Charge = NEGATIVE on this meter. Ten charging samples, then one
        # discharge spike sample right before the gap, then charging again
        # with SoC a little higher than where it went dark.
        powers = [-1000.0] * 10 + [20000.0] + [-1000.0] * 10
        soc1 = [20.0 + step_pts * k for k in range(11)]  # still rising into the spike
        soc2 = [soc1[-1] + 5.0 + step_pts * k for k in range(10)]
        df = pd.DataFrame(
            {
                "sensor_power_battery": powers,
                "sensor_battery_state_of_charge": soc1 + soc2,
            },
            index=idx1.append(idx2),
        )
        segs = BatteryIdentification(self.logger)._segment(df, self.power_col, self.soc_col)
        # Masked vote -> convention flips correctly -> one charge block on each
        # side of the gap. An unmasked vote leaves the convention inverted and
        # the post-gap block comes out labelled discharge instead.
        self.assertEqual(len(segs), 2, "one segment per observed block")
        for s in segs:
            self.assertEqual(
                s.direction,
                "charge",
                "sign convention must be detected from real steps, not the gap step",
            )


_STEP = pd.Timedelta("30min")
_P, _S, _L = "sensor.batt_power", "sensor.batt_soc", "sensor.load"


def _fast_counts(last):
    """Readings per bucket for sensors that report faster than the step: two in
    every bucket that has a real state, none elsewhere."""
    return {k: v.notna().astype("int64") * 2 for k, v in last.items()}


def _fill_case(power_last, soc_last, power_df=None, soc_df=None, tz="Europe/Paris", load_last=None):
    """A frame on a 30-min grid plus the aux series the REST retrieval records for it.

    ``*_last`` are the per-bucket last real states (NaN = no real recorded
    state in that bucket); ``*_df`` are the frame values (bucket means; they
    default to the last states). The aux is indexed in UTC like the retrieval
    records it, while the frame is converted to a local zone like prepare_data
    leaves it, so every test also covers the instant-based alignment. The load
    sensor has a reading in every bucket unless ``load_last`` says otherwise, so
    by default the recorder was running throughout.
    """
    n = len(power_last)
    idx = pd.date_range("2026-03-02 00:00", periods=n, freq=_STEP, tz="UTC")
    power_df = power_last if power_df is None else power_df
    soc_df = soc_last if soc_df is None else soc_df
    df = pd.DataFrame(
        {_P: np.asarray(power_df, dtype=float), _S: np.asarray(soc_df, dtype=float)}, index=idx
    )
    if tz is not None:
        df.index = df.index.tz_convert(tz)
    load_last = [500.0] * n if load_last is None else load_last
    last = {
        _P: pd.Series(np.asarray(power_last, dtype=float), index=idx),
        _S: pd.Series(np.asarray(soc_last, dtype=float), index=idx),
        _L: pd.Series(np.asarray(load_last, dtype=float), index=idx),
    }
    return df, last, idx


NAN = float("nan")


class TestStalenessFill(unittest.TestCase):
    """Staleness-aware fill of short battery sensor silences (#1073)."""

    def setUp(self):
        self.logger = logging.getLogger("test_staleness_fill")

    def _fill(self, df, last, markers=None, step=_STEP, counts=None):
        counts = _fast_counts(last) if counts is None else counts
        return apply_staleness_fill(
            df, _P, _S, last, markers or {}, step, self.logger, row_counts=counts
        )

    def _assert_untouched(self, out, df):
        pd.testing.assert_frame_equal(out, df)

    # -- a short bounded silence is held at the last raw state ---------------
    def test_short_silence_held_at_last_raw_state_not_mean_or_bridge(self):
        """The bucket before the silence holds a transition (raw
        rows 1000 W then 3500 W, mean 2250 W); the sensor then stayed at
        3500 W for two empty buckets. The fill must write 3500 W, not the
        bucket mean and not the straight line the trapezoid draws today, and
        the segment throughput must change by exactly that."""
        power_last = [0, 0, 3500, NAN, NAN, 3600, 3600, 3600, 0, 0]
        power_df = [0, 0, 2250, NAN, NAN, 3600, 3600, 3600, 0, 0]
        soc = [20, 20, 22, 28, 34, 40, 46, 52, 52, 52]
        df, last, _ = _fill_case(power_last, soc, power_df=power_df)
        out = self._fill(df, last)
        np.testing.assert_array_equal(out[_P].to_numpy()[3:5], [3500.0, 3500.0])
        # Everything else is untouched, and the input frame is not mutated.
        np.testing.assert_array_equal(
            np.delete(out[_P].to_numpy(), [3, 4]), np.delete(power_df, [3, 4])
        )
        np.testing.assert_array_equal(out[_S].to_numpy(), np.asarray(soc, dtype=float))
        self.assertTrue(df[_P].iloc[3:5].isna().all(), "the input frame must not be mutated")

        bid = BatteryIdentification(self.logger, time_step=_STEP)
        before = bid._segment(df, _P, _S)
        after = bid._segment(out, _P, _S)
        self.assertEqual(len(before), 1)
        self.assertEqual(len(after), 1)
        # Trapezoid over 0.5 h steps. Unfilled: 2250 -> 3600 bridged across
        # 1.5 h, then two 3600 W steps. Filled: 2250 -> 3500, two held steps at
        # 3500 W into 3600 W, then the same two 3600 W steps.
        self.assertAlmostEqual(before[0].throughput_wh, 4387.5 + 1800 + 1800)
        self.assertAlmostEqual(after[0].throughput_wh, 1437.5 + 1750 + 1775 + 1800 + 1800)
        self.assertEqual((after[0].start, after[0].end), (before[0].start, before[0].end))

    def test_one_bucket_silence_filled(self):
        df, last, _ = _fill_case([100, 200, NAN, 400], [10, 20, 30, 40])
        out = self._fill(df, last)
        self.assertEqual(out[_P].iloc[2], 200.0)

    def test_soc_silence_filled_too(self):
        df, last, _ = _fill_case([100, 200, 300, 400, 500], [10, 20, NAN, NAN, 50])
        out = self._fill(df, last)
        np.testing.assert_array_equal(out[_S].to_numpy(), [10, 20, 20, 20, 50])

    # -- length, bounds, never partial ----------------------------------------
    def test_run_of_h_plus_one_untouched(self):
        df, last, _ = _fill_case([100, 200, NAN, NAN, NAN, 600], [10, 20, 30, 40, 50, 60])
        self._assert_untouched(self._fill(df, last), df)

    def test_long_run_never_partially_filled(self):
        df, last, _ = _fill_case([100] + [NAN] * 7 + [900], list(range(9)))
        out = self._fill(df, last)
        self.assertTrue(out[_P].iloc[1:8].isna().all(), "a long silence must stay whole")

    def test_leading_and_trailing_runs_untouched(self):
        df, last, _ = _fill_case([NAN, NAN, 300, 400, NAN, NAN], [10, 20, 30, 40, 50, 60])
        self._assert_untouched(self._fill(df, last), df)

    def test_synthetic_start_row_bucket_counts_inside_a_run(self):
        """A bucket holding only Home Assistant's start-of-day row has a value in
        the frame but no real recorded state: it is part of the run, so a
        one-bucket run around it is filled (overwriting the carried value) ..."""
        df, last, _ = _fill_case(
            [100, 200, NAN, 400], [10, 20, 30, 40], power_df=[100, 200, 180, 400]
        )
        out = self._fill(df, last)
        self.assertEqual(out[_P].iloc[2], 200.0)

    def test_synthetic_start_row_bucket_is_never_a_bound(self):
        """... and it never splits a longer silence into two short ones:
        1 empty + synthetic + 1 empty = a run of 3, left exactly as it was."""
        df, last, _ = _fill_case(
            [100, NAN, NAN, NAN, 500],
            [10, 20, 30, 40, 50],
            power_df=[100, NAN, 180, NAN, 500],
        )
        self._assert_untouched(self._fill(df, last), df)

    def test_bucket_with_value_but_no_real_state_is_overwritten_only_in_a_qualifying_run(self):
        """A run is defined by the aux series, not by frame NaNs: a value
        prepare_data put there (sensor_linear_interp) is replaced by the held
        state inside a qualifying run and left alone outside one."""
        df, last, _ = _fill_case(
            [100, NAN, NAN, 400, NAN, NAN, NAN, 800],
            [10, 20, 30, 40, 50, 60, 70, 80],
            power_df=[100, 200, 300, 400, 500, 600, 700, 800],
        )
        out = self._fill(df, last)
        np.testing.assert_array_equal(out[_P].to_numpy(), [100, 100, 100, 400, 500, 600, 700, 800])

    # -- markers ------------------------------------------------------
    def _marker_case(self):
        # power silent at buckets 3-4 (bounds 2 and 5), soc silent at 3-4 too
        df, last, idx = _fill_case(
            [100, 200, 300, NAN, NAN, 600, 700, 800],
            [10, 20, 30, NAN, NAN, 60, 70, 80],
        )
        return df, last, idx

    def test_marker_inside_run_voids(self):
        df, last, idx = self._marker_case()
        out = self._fill(df, last, {_P: pd.DatetimeIndex([idx[4]])})
        self._assert_untouched(out, df)

    def test_marker_in_left_bounding_bucket_voids(self):
        """An integration death shares a bucket with its last healthy reading."""
        df, last, idx = self._marker_case()
        out = self._fill(df, last, {_P: pd.DatetimeIndex([idx[2]])})
        self._assert_untouched(out, df)

    def test_marker_in_right_bounding_bucket_voids(self):
        df, last, idx = self._marker_case()
        out = self._fill(df, last, {_P: pd.DatetimeIndex([idx[5]])})
        self._assert_untouched(out, df)

    def test_marker_on_pair_partner_voids_both(self):
        df, last, idx = self._marker_case()
        out = self._fill(df, last, {_S: pd.DatetimeIndex([idx[3]])})
        self._assert_untouched(out, df)

    def test_marker_outside_bounds_does_not_void(self):
        df, last, idx = self._marker_case()
        out = self._fill(df, last, {_P: pd.DatetimeIndex([idx[1], idx[6]])})
        np.testing.assert_array_equal(out[_P].to_numpy()[3:5], [300, 300])
        np.testing.assert_array_equal(out[_S].to_numpy()[3:5], [30, 30])

    def test_marker_on_unrelated_sensor_does_not_void(self):
        df, last, idx = self._marker_case()
        out = self._fill(df, last, {"sensor.load": pd.DatetimeIndex([idx[3], idx[4]])})
        np.testing.assert_array_equal(out[_P].to_numpy()[3:5], [300, 300])
        np.testing.assert_array_equal(out[_S].to_numpy()[3:5], [30, 30])

    def test_marker_voids_only_the_runs_it_touches(self):
        df, last, idx = _fill_case([100, NAN, 300, 400, NAN, 600], [10, 20, 30, 40, 50, 60])
        out = self._fill(df, last, {_S: pd.DatetimeIndex([idx[1]])})
        self.assertTrue(np.isnan(out[_P].iloc[1]), "the marked run stays unfilled")
        self.assertEqual(out[_P].iloc[4], 400.0, "an unmarked run elsewhere is still filled")

    # -- the horizon ----------------------------------------------------------
    def test_default_horizon_is_two_and_inside_the_gap_bound(self):
        """H=2 by default (#1073), and never past the buckets the #1063 gap test
        already bridges, so the fill cannot make a recorder gap of one sensor
        look observed."""
        h = getattr(bi_module, "STALENESS_FILL_MAX_BUCKETS", None)
        self.assertEqual(h, 2)
        self.assertLessEqual(h, bi_module.MAX_GAP_STEP_MULTIPLE - 1)

    def test_horizon_zero_means_no_fill(self):
        df, last, _ = _fill_case([100, 200, NAN, NAN, 500], [10, 20, NAN, 40, 50])
        with unittest.mock.patch.object(bi_module, "STALENESS_FILL_MAX_BUCKETS", 0, create=True):
            out = self._fill(df, last)
        self._assert_untouched(out, df)

    # -- the gap detector keeps firing for every silence longer than H -------
    def test_gap_detector_boundaries_unchanged_for_long_silences(self):
        """Silences of both sensors longer than H (3, 4 and 9 buckets) are
        untouched, so the #1063 gap detector ends and restarts runs there
        exactly as without the fill, while short silences elsewhere are filled."""
        hist = _make_history(10000.0, 0.90, n_cycles=6, power_w=1500.0, dt_minutes=30)
        hist = hist.rename(
            columns={"sensor_power_battery": _P, "sensor_battery_state_of_charge": _S}
        )
        n = len(hist)
        power_last = hist[_P].to_numpy().copy()
        soc_last = hist[_S].to_numpy().copy()
        long_silences = [(12, 3), (40, 4), (75, 9)]
        for start, length in long_silences:
            power_last[start : start + length] = NAN
            soc_last[start : start + length] = NAN
        for start in (25, 58, 101):  # short silences, one sensor at a time
            power_last[start : start + 2] = NAN
            soc_last[start + 6 : start + 7] = NAN
        self.assertLess(101 + 7, n)
        df, last, _ = _fill_case(power_last, soc_last, tz="UTC")
        out = self._fill(df, last)
        for start, length in long_silences:
            self.assertTrue(out[_P].iloc[start : start + length].isna().all())
            self.assertTrue(out[_S].iloc[start : start + length].isna().all())

        bid = BatteryIdentification(self.logger, time_step=_STEP)
        before = bid._segment(df, _P, _S)
        after = bid._segment(out, _P, _S)
        self.assertTrue(after, "the history must still segment")
        for start, length in long_silences:
            gap_open, gap_close = df.index[start - 1], df.index[start + length]
            for seg in after:
                self.assertFalse(
                    seg.start < gap_close and seg.end > gap_open,
                    f"no segment may span the {length}-bucket silence",
                )
            # The run ending at the silence and the run starting after it.
            self.assertEqual(
                max((s.end for s in after if s.end <= gap_open), default=None),
                max((s.end for s in before if s.end <= gap_open), default=None),
            )
            self.assertEqual(
                min((s.start for s in after if s.start >= gap_close), default=None),
                min((s.start for s in before if s.start >= gap_close), default=None),
            )
        self.assertFalse(out.equals(df), "the short silences must have been filled")

    def test_offset_silences_compose_with_one_member_observed(self):
        """Intended: power silent for two buckets, then
        SoC for the next two. Each bridged bucket keeps one really observed
        member, and neither held value is older than H buckets."""
        df, last, _ = _fill_case(
            [100, 200, NAN, NAN, 500, 600, 700],
            [10, 20, 30, 40, NAN, NAN, 70],
        )
        out = self._fill(df, last)
        self.assertFalse(out[[_P, _S]].isna().any().any())
        np.testing.assert_array_equal(out[_P].to_numpy(), [100, 200, 200, 200, 500, 600, 700])
        np.testing.assert_array_equal(out[_S].to_numpy(), [10, 20, 30, 40, 40, 40, 70])
        # On the joint rows the estimator's gap test sees, the two silences add
        # up to a step past the gap bound before the hold and to none after it.
        bound = bi_module.MAX_GAP_STEP_MULTIPLE * pd.Timedelta(_STEP)

        def joint_step(frame):
            return frame[[_P, _S]].dropna().index.to_series().diff().max()

        self.assertGreater(joint_step(df), bound)
        self.assertEqual(joint_step(out), pd.Timedelta(_STEP))

    # -- no-op paths ----------------------------------------------------------
    def test_missing_aux_is_identity(self):
        df, last, _ = _fill_case([100, NAN, 300], [10, NAN, 30])
        counts = _fast_counts(last)
        for aux in ({}, None, {_P: last[_P]}, {_S: last[_S]}, unittest.mock.MagicMock()):
            self.assertIs(apply_staleness_fill(df, _P, _S, aux, {}, _STEP, row_counts=counts), df)

    def test_no_time_step_is_identity(self):
        df, last, _ = _fill_case([100, NAN, 300], [10, NAN, 30])
        counts = _fast_counts(last)
        self.assertIs(apply_staleness_fill(df, _P, _S, last, {}, None, row_counts=counts), df)

    def test_misaligned_aux_is_identity(self):
        df, last, _ = _fill_case([100, NAN, 300], [10, NAN, 30])
        counts = _fast_counts(last)
        shifted = {k: v.shift(10, freq="min") for k, v in last.items()}
        self.assertIs(apply_staleness_fill(df, _P, _S, shifted, {}, _STEP, row_counts=counts), df)
        markers = {_P: pd.DatetimeIndex([last[_P].index[0] + pd.Timedelta("7min")])}
        self.assertIs(apply_staleness_fill(df, _P, _S, last, markers, _STEP, row_counts=counts), df)
        shifted_counts = {k: v.shift(10, freq="min") for k, v in counts.items()}
        self.assertIs(
            apply_staleness_fill(df, _P, _S, last, {}, _STEP, row_counts=shifted_counts), df
        )

    def test_tz_naive_frame_with_utc_aux_is_identity(self):
        df, last, _ = _fill_case([100, NAN, 300], [10, NAN, 30], tz=None)
        df.index = df.index.tz_localize(None)
        counts = _fast_counts(last)
        self.assertIs(apply_staleness_fill(df, _P, _S, last, {}, _STEP, row_counts=counts), df)

    # -- the pair must report faster than the step ------------
    def test_sensor_polled_once_per_step_or_slower_is_identity(self):
        """Regression: A power sensor polled every other step leaves
        one empty bucket between every two readings. Those buckets are changes
        nobody sampled, not an unchanged value, so nothing of the pair is held.
        The same history from a sensor that reports faster than the step is."""
        n = 24
        power = [100.0 + 10 * i if i % 2 == 0 else NAN for i in range(n)]
        soc = [20.0 + i for i in range(n)]
        df, last, _ = _fill_case(power, soc)
        polled = _fast_counts(last)
        polled[_P] = last[_P].notna().astype("int64")  # one reading per observed bucket
        self.assertIs(self._fill(df, last, counts=polled), df)
        self.assertEqual(self._fill(df, last)[_P].iloc[1], 100.0)

    def test_polled_soc_alone_also_stops_the_fill(self):
        df, last, _ = _fill_case(
            [100.0, 200.0, NAN, 400.0, 500.0, 600.0], [10.0, NAN, 30.0, NAN, 50.0, NAN]
        )
        polled = _fast_counts(last)
        polled[_S] = last[_S].notna().astype("int64")
        self.assertIs(self._fill(df, last, counts=polled), df)

    def test_reporting_rate_needs_two_readings_in_a_fifth_of_observed_buckets(self):
        df, last, _ = _fill_case(
            [100, 200, 300, 400, NAN, 600, 700, 800, 900, 1000, 1100], list(range(11))
        )
        counts = _fast_counts(last)
        power_counts = last[_P].notna().astype("int64")
        power_counts.iloc[[0, 1]] = 2  # 2 of the 10 observed buckets
        counts[_P] = power_counts
        self.assertEqual(self._fill(df, last, counts=counts)[_P].iloc[4], 400.0)
        power_counts.iloc[1] = 1  # 1 of 10
        self.assertIs(self._fill(df, last, counts=counts), df)

    def test_missing_row_counts_is_identity(self):
        """Without the readings per bucket the reporting rate cannot be checked."""
        df, last, _ = _fill_case([100, 200, NAN, 400], [10, 20, 30, 40])
        fast = _fast_counts(last)
        for counts in (None, {}, {_P: fast[_P]}, {_S: fast[_S]}, unittest.mock.MagicMock()):
            self.assertIs(apply_staleness_fill(df, _P, _S, last, {}, _STEP, row_counts=counts), df)

    # -- a run needs another sensor's reading in every bucket --
    def test_recorder_outage_is_not_held(self):
        """Regression: every sensor, the load included, is quiet
        for two buckets: the recorder was down, and the battery may well have
        changed meanwhile (it can go full inside the outage). Left as it was."""
        load = [500.0] * 8
        load[3] = load[4] = NAN
        df, last, _ = _fill_case(
            [100, 200, 300, NAN, NAN, 600, 700, 800],
            [10, 20, 30, NAN, NAN, 60, 70, 80],
            load_last=load,
        )
        self._assert_untouched(self._fill(df, last), df)

    def test_every_bucket_of_a_run_needs_a_witness(self):
        load = [500.0] * 8
        load[4] = NAN
        df, last, _ = _fill_case(
            [100, 200, 300, NAN, NAN, 600, 700, 800],
            [10, 20, 30, NAN, NAN, 60, 70, 80],
            load_last=load,
        )
        self._assert_untouched(self._fill(df, last), df)

    def test_partner_reading_witnesses_a_run(self):
        """Only the power sensor is quiet. The state of charge readings show the
        recorder was running, so the power run is held with no load reading."""
        load = [500.0] * 6
        load[2] = load[3] = NAN
        df, last, _ = _fill_case(
            [100, 200, NAN, NAN, 500, 600], [10, 20, 30, 40, 50, 60], load_last=load
        )
        np.testing.assert_array_equal(self._fill(df, last)[_P].to_numpy()[2:4], [200, 200])

    def test_witness_off_the_grid_counts_for_nothing(self):
        df, last, _ = _fill_case(
            [100, 200, 300, NAN, NAN, 600, 700, 800], [10, 20, 30, NAN, NAN, 60, 70, 80]
        )
        last[_L] = last[_L].shift(7, freq="min")
        self._assert_untouched(self._fill(df, last), df)

    # -- cost follows the history, not time_step --------------
    def test_time_step_finer_than_the_frame_is_identity(self):
        """The grid is the frame's span over time_step. A step far finer than the
        frame's own spacing must not size it: 1 ns would ask for about 10**13
        buckets on this four-row frame."""
        df, last, _ = _fill_case([100, 200, NAN, 400], [10, 20, 30, 40])
        counts = _fast_counts(last)
        for step in (pd.Timedelta("1ns"), pd.Timedelta("1ms"), pd.Timedelta("1s"), 1):
            started = time.perf_counter()
            self.assertIs(apply_staleness_fill(df, _P, _S, last, {}, step, row_counts=counts), df)
            self.assertLess(time.perf_counter() - started, 1.0)

    def test_degenerate_inputs_do_not_raise(self):
        df, last, _ = _fill_case([100, NAN, 300], [10, NAN, 30])

        def fill(frame, aux):
            return apply_staleness_fill(frame, _P, _S, aux, {}, _STEP, row_counts=_fast_counts(aux))

        empty = df.iloc[0:0]
        self.assertIs(fill(empty, last), empty)
        one = df.iloc[0:1]
        pd.testing.assert_frame_equal(fill(one, last), one)
        all_nan = {k: v * np.nan for k, v in last.items()}
        pd.testing.assert_frame_equal(fill(df, all_nan), df)
        empty_aux = {k: v.iloc[0:0] for k, v in last.items()}
        pd.testing.assert_frame_equal(fill(df, empty_aux), df)


# --- end to end through the real REST retrieval -----------------------------
_E2E_CAP_WH, _E2E_RTE = 10000.0, 0.9025
_E2E_LOAD = "sensor.power_load_no_var_loads"
_E2E_URL = "http://test.local/"


def _pack_truth(n_days: int, seed: int = 42):
    """Deterministic 1-min truth (index, power W, SoC %) for a 10 kWh pack with RTE
    0.9025: overnight base-load discharge plateaus, a solar charge (ramp,
    plateau, taper) and evening discharge plateaus each day."""
    rng = np.random.default_rng(seed)
    minutes = n_days * 24 * 60
    idx = pd.date_range("2026-07-01", periods=minutes, freq="1min", tz="UTC")
    p = np.zeros(minutes)
    for d in range(n_days):
        day0 = d * 24 * 60

        def span(h0, h1, day0=day0):
            return slice(day0 + int(h0 * 60), day0 + int(h1 * 60))

        t = 0.0
        while t < 6.0:
            dur = rng.uniform(2.0, 4.0)
            p[span(t, min(t + dur, 6.0))] = -rng.uniform(300, 800)
            t += dur
        p[span(9.0, 10.5)] = np.linspace(200, 2800, num=90) * rng.uniform(0.8, 1.2)
        p[span(10.5, 13.0)] = rng.uniform(2200, 3800)
        p[span(13.0, 15.0)] = np.linspace(1.0, 0.15, num=120) ** 1.5 * rng.uniform(2200, 3800)
        t = 17.0
        while t < 23.0:
            dur = rng.uniform(1.0, 2.0)
            p[span(t, min(t + dur, 23.0))] = -rng.uniform(1000, 3000)
            t += dur
    eta = np.sqrt(_E2E_RTE)
    soc = np.zeros(minutes)
    s = 50.0
    for i in range(minutes):
        step = eta * p[i] if p[i] > 0 else p[i] / eta
        s_new = s + step / 60.0 / _E2E_CAP_WH * 100.0
        if s_new > 95.0 or s_new < 12.0:
            p[i] = 0.0  # full or empty: the battery idles
            s_new = s
        soc[i] = s = s_new
    return idx, p, soc


def _change_only_pack(n_days: int, seed: int = 42, quant_w: float = 100.0, quant_soc: float = 1.0):
    """The :func:`_pack_truth` pack recorded the way a clean change-only Home
    Assistant sensor records it: a row is written only when the quantized value
    changes, so plateaus leave empty buckets between rows."""
    idx, p, soc = _pack_truth(n_days, seed)

    def change_only(values, quant):
        q = np.round(values / quant) * quant
        changed = np.r_[True, q[1:] != q[:-1]]
        return pd.Series(q[changed], index=idx[changed])

    return idx, change_only(p, quant_w), change_only(soc, quant_soc)


def _ha_history_rows(series: pd.Series, day: pd.Timestamp, entity_id: str) -> list:
    """What Home Assistant's /history/period/<day> returns for one entity: the
    state carried into the period, stamped at the day start, then every
    recorded change of that day."""
    end = day + pd.Timedelta(days=1)
    before = series[series.index < day]
    inside = series[(series.index >= day) & (series.index < end)]
    rows = []
    if len(before):
        rows.append((day, before.iloc[-1]))
    for ts, value in inside.items():
        if ts == day and rows:
            rows[0] = (ts, value)
        else:
            rows.append((ts, value))
    return [
        {
            "entity_id": entity_id,
            "state": str(value),
            "attributes": {},
            "last_changed": ts.isoformat(),
            "last_updated": ts.isoformat(),
        }
        for ts, value in rows
    ]


@unittest.skipUnless(HAVE_FEATURE and HAVE_CL, "battery_identification not present")
class TestStalenessFillEndToEnd(unittest.IsolatedAsyncioTestCase):
    """The real battery_id retrieval (``_retrieve_from_hass`` ->
    REST ``get_data`` -> ``prepare_data``) fed with HA-shaped history, then the
    fill, then the real estimator. The counterfactual (the same retrieved frame
    without the fill) is fitted in the same test, so it cannot false-green:
    on a source without the fill both fits are the same and the test fails."""

    @staticmethod
    async def _retrieve(series, days, logger):
        """The real battery_id retrieval over HA-shaped history of ``series``."""
        from tests.aioresponses_compat import aioresponses

        retrieve_hass_conf = {
            "optimization_time_step": _STEP,
            "sensor_power_load_no_var_loads": _E2E_LOAD,
            "sensor_power_battery": _P,
            "sensor_battery_state_of_charge": _S,
            "load_negative": False,
            "set_zero_min": True,
            "sensor_replace_zero": [],
            "sensor_linear_interp": [],
        }
        retrieve_hass_conf["historic_days_to_retrieve"] = len(days)
        rh = command_line.RetrieveHass(_E2E_URL, "token", _STEP, "UTC", None, {}, logger)
        with (
            aioresponses() as mocked,
            unittest.mock.patch.object(command_line.utils, "get_days_list", return_value=days),
        ):
            for day in days:
                for entity_id, rows in series.items():
                    url = (
                        f"{_E2E_URL}api/history/period/{day.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                        f"?filter_entity_id={entity_id}"
                    )
                    mocked.get(url, payload=[_ha_history_rows(rows, day, entity_id)], status=200)
            success, df, _ = await command_line.retrieve_home_assistant_data(
                "battery_id", False, retrieve_hass_conf, {}, rh, {}, "", logger
            )
        assert success
        return rh, df

    @staticmethod
    def _fill(rh, df, logger):
        return apply_staleness_fill(
            df,
            _P,
            _S,
            getattr(rh, "bucket_last_state", {}),
            getattr(rh, "marker_buckets", {}),
            _STEP,
            logger,
            row_counts=getattr(rh, "bucket_row_count", {}),
        )

    @staticmethod
    def _dense_load(idx):
        """A house load meter: a new reading every five minutes."""
        stamps = idx[::5]
        return pd.Series(500.0 + np.arange(len(stamps)) % 37, index=stamps)

    async def test_fill_shrinks_change_only_bias_through_real_rest_retrieval(self):
        logger = logging.getLogger("test_staleness_fill_e2e")
        n_days = 26
        idx, power_rows, soc_rows = _change_only_pack(n_days)
        load_rows = pd.Series(500.0, index=idx[::60])
        days = pd.date_range(idx[0], periods=n_days, freq="D")
        series = {_P: power_rows, _S: soc_rows, _E2E_LOAD: load_rows}
        rh, df = await self._retrieve(series, days, logger)

        filled = apply_staleness_fill(
            df,
            _P,
            _S,
            getattr(rh, "bucket_last_state", {}),
            getattr(rh, "marker_buckets", {}),
            _STEP,
            logger,
            row_counts=getattr(rh, "bucket_row_count", {}),
        )

        def errors(frame):
            res = BatteryIdentification(logger, time_step=_STEP).identify(
                frame, _P, _S, _E2E_CAP_WH
            )
            self.assertIsNotNone(res.capacity_wh, res.messages)
            self.assertIsNotNone(res.round_trip_efficiency, res.messages)
            cap_err = 100.0 * (res.capacity_wh - _E2E_CAP_WH) / _E2E_CAP_WH
            return cap_err, res.round_trip_efficiency - _E2E_RTE

        cap_raw, rte_raw = errors(df)
        cap_fill, rte_fill = errors(filled)
        # The change-only bias is there without the fill (it is what #1073 is
        # about), and the fill takes most of it out.
        self.assertGreater(abs(rte_raw), 0.04, "fixture must show the change-only RTE bias")
        self.assertLess(abs(rte_fill), 0.5 * abs(rte_raw), (rte_raw, rte_fill))
        self.assertLess(abs(cap_fill), abs(cap_raw) - 0.5, (cap_raw, cap_fill))
        self.assertLess(abs(cap_fill), 1.0)
        self.assertLess(abs(rte_fill), 0.02)

    async def test_polled_pair_is_left_as_retrieved(self):
        """Regression: the same pack read by an integration that
        polls once an hour, with the default 30 minute step: one empty bucket
        between every two readings while the battery works. Holding those would
        lag the pack by up to an hour (+15..+18% capacity in probes),
        so the frame reaches the fit exactly as retrieved."""
        logger = logging.getLogger("test_staleness_fill_e2e_polled")
        n_days = 8
        idx, p, soc = _pack_truth(n_days)
        polled = idx[59::60] + pd.Timedelta(seconds=7)

        def readings(values, quant):
            q = pd.Series(np.round(values[59::60] / quant) * quant, index=polled)
            return q[q.diff().ne(0)]  # HA writes a row only on a change

        series = {_P: readings(p, 100.0), _S: readings(soc, 1.0), _E2E_LOAD: self._dense_load(idx)}
        days = pd.date_range(idx[0], periods=n_days, freq="D")
        rh, df = await self._retrieve(series, days, logger)
        pd.testing.assert_frame_equal(self._fill(rh, df, logger), df)
        counts = getattr(rh, "bucket_row_count", {})
        self.assertEqual(int(counts[_P].max()), 1, "one reading per bucket at most")

    async def test_recorder_outage_is_left_as_retrieved(self):
        """Regression: every day the recorder is down from 12:10 to
        13:10, when the pack is near full: no row of any sensor, the load
        included, then each sensor reports its state again. The quiet bucket is
        left as retrieved, while short silences elsewhere are still held."""
        logger = logging.getLogger("test_staleness_fill_e2e_outage")
        n_days = 8
        idx, power_rows, soc_rows = _change_only_pack(n_days)
        series = {_P: power_rows, _S: soc_rows, _E2E_LOAD: self._dense_load(idx)}
        quiet = []
        for day in range(n_days):
            down = idx[0] + pd.Timedelta(days=day, hours=12, minutes=10)
            up = down + pd.Timedelta(hours=1)
            for key, rows in series.items():
                revived = rows.asof(up)
                rows = rows[(rows.index < down) | (rows.index >= up)]
                if not np.isnan(revived):
                    rows.loc[up] = revived
                series[key] = rows.sort_index()
            quiet.append(down + pd.Timedelta(minutes=20))  # the 12:30 bucket
        days = pd.date_range(idx[0], periods=n_days, freq="D")
        rh, df = await self._retrieve(series, days, logger)
        filled = self._fill(rh, df, logger)
        quiet = pd.DatetimeIndex(quiet).tz_convert(df.index.tz)
        self.assertTrue(df.loc[quiet, [_P, _S]].isna().all().all(), "fixture: nothing recorded")
        pd.testing.assert_frame_equal(filled.loc[quiet], df.loc[quiet])
        self.assertFalse(filled.equals(df), "short silences elsewhere must still be held")


class _FakeRHWithAux(_FakeRH):
    """A retrieval that recorded the staleness aux series (REST, battery_id)."""

    def __init__(self, bucket_last_state, marker_buckets=None):
        super().__init__()
        self.bucket_last_state = bucket_last_state
        self.bucket_row_count = _fast_counts(bucket_last_state)
        self.marker_buckets = marker_buckets or {}


@unittest.skipUnless(
    HAVE_FEATURE and HAVE_CL, "battery_identification/command_line not present (base commit)"
)
class TestStalenessFillWiring(unittest.IsolatedAsyncioTestCase):
    """Both identify call sites (N=1 and N>1) must hand the estimator the filled
    frame; a retrieval without aux (file mode, websocket, the existing fakes)
    must reach it unchanged. The frame identify receives still has
    the silence."""

    def setUp(self):
        import tempfile

        self.logger = logging.getLogger("test_staleness_fill_wiring")
        self.emhass_conf = {"data_path": pathlib.Path(tempfile.mkdtemp())}
        self.step = pd.Timedelta("5min")
        self.seen = {}
        self.plant_conf = {
            "battery_nominal_energy_capacity": 10000,
            "battery_charge_efficiency": 0.95,
            "battery_discharge_efficiency": 0.95,
        }

    @staticmethod
    def _with_hole(df, col, start, length=2):
        """Punch a short silence into ``col`` and return the frame plus its aux,
        whose last state before the silence differs from the frame's bucket
        mean so a hold at the last state is distinguishable from anything else."""
        holed = df.copy()
        holed.iloc[start : start + length, holed.columns.get_loc(col)] = np.nan
        last = holed[col].copy()
        last.iloc[start - 1] = holed[col].iloc[start - 1] + 7.0
        return holed, last

    async def _run(self, df, rh, retrieve_hass_conf, plant_conf):
        orig_retrieve = command_line.retrieve_home_assistant_data
        orig_bid = command_line.BatteryIdentification
        seen = self.seen

        async def _fake_retrieve(*args, **kwargs):
            return True, df, None

        class _Capture(orig_bid):
            def identify(self, frame, power_col, soc_col, cap):
                seen[power_col] = frame.copy()
                return super().identify(frame, power_col, soc_col, cap)

        command_line.retrieve_home_assistant_data = _fake_retrieve
        command_line.BatteryIdentification = _Capture
        try:
            await command_line.identify_battery(
                self.logger,
                {"battery_identification_trust_tier": "observe"},
                plant_conf,
                retrieve_hass_conf,
                rh,
                self.emhass_conf,
                False,
                "test_df_final.pkl",
            )
        finally:
            command_line.retrieve_home_assistant_data = orig_retrieve
            command_line.BatteryIdentification = orig_bid

    async def test_n1_identify_receives_filled_frame(self):
        base = _make_history(10000.0, 0.90, n_cycles=6).rename(
            columns={"sensor_power_battery": _P, "sensor_battery_state_of_charge": _S}
        )
        df, p_last = self._with_hole(base, _P, 30)
        rh = _FakeRHWithAux({_P: p_last, _S: df[_S]})
        conf = {
            "sensor_power_battery": _P,
            "sensor_battery_state_of_charge": _S,
            "optimization_time_step": self.step,
        }
        await self._run(df, rh, conf, self.plant_conf)
        frame = self.seen[_P]
        np.testing.assert_array_equal(frame[_P].iloc[30:32].to_numpy(), [p_last.iloc[29]] * 2)
        self.assertTrue(df[_P].iloc[30:32].isna().all(), "the retrieved frame must not be mutated")

    async def test_n1_without_aux_reaches_identify_unchanged(self):
        base = _make_history(10000.0, 0.90, n_cycles=6).rename(
            columns={"sensor_power_battery": _P, "sensor_battery_state_of_charge": _S}
        )
        df, _ = self._with_hole(base, _P, 30)
        conf = {
            "sensor_power_battery": _P,
            "sensor_battery_state_of_charge": _S,
            "optimization_time_step": self.step,
        }
        await self._run(df, _FakeRH(), conf, self.plant_conf)
        pd.testing.assert_frame_equal(self.seen[_P], df)

    async def test_n2_each_identify_receives_its_own_filled_columns(self):
        df = _make_multi_battery_df(
            [("p0", "soc0", 10000.0, 0.90), ("p1", "soc1", 10000.0, 0.90)], n_cycles=6
        )
        df, p0_last = self._with_hole(df, "p0", 30)
        df, soc1_last = self._with_hole(df, "soc1", 50)
        rh = _FakeRHWithAux({"p0": p0_last, "soc0": df["soc0"], "p1": df["p1"], "soc1": soc1_last})
        conf = {
            "sensor_power_battery": ["p0", "p1"],
            "sensor_battery_state_of_charge": ["soc0", "soc1"],
            "optimization_time_step": self.step,
        }
        plant_conf = {
            "number_of_batteries": 2,
            "battery_nominal_energy_capacity": [10000, 10000],
            "battery_charge_efficiency": [0.95, 0.95],
            "battery_discharge_efficiency": [0.95, 0.95],
        }
        await self._run(df, rh, conf, plant_conf)
        self.assertEqual(set(self.seen), {"p0", "p1"})
        np.testing.assert_array_equal(
            self.seen["p0"]["p0"].iloc[30:32].to_numpy(), [p0_last.iloc[29]] * 2
        )
        np.testing.assert_array_equal(
            self.seen["p1"]["soc1"].iloc[50:52].to_numpy(), [soc1_last.iloc[49]] * 2
        )
        self.assertTrue(df["p0"].iloc[30:32].isna().all(), "the shared frame must not be mutated")
        self.assertTrue(df["soc1"].iloc[50:52].isna().all(), "the shared frame must not be mutated")


if __name__ == "__main__":
    unittest.main()
