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

import logging
import pathlib
import unittest

import numpy as np
import pandas as pd

try:
    from emhass.battery_identification import (
        BatteryIdentification,
        BatteryIdentificationResult,
    )

    HAVE_FEATURE = True
except ImportError:  # base commit without the feature
    HAVE_FEATURE = False
    BatteryIdentification = None
    BatteryIdentificationResult = None


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
            return success, monkey_df, None

        command_line.retrieve_home_assistant_data = _fake_retrieve

    async def _run(self, optim_conf, df=None, success=True):
        orig = command_line.retrieve_home_assistant_data
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
        import json as _json

        payload = _json.loads(self._json_path().read_text())
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

    async def test_failed_fit_writes_nothing_and_leaves_existing_untouched(self):
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
        import json as _json

        payload = _json.loads(self._json_path().read_text())
        self.assertEqual(payload.get("marker"), "keep-me", "failed fit must not clobber the file")
        self.assertEqual(rh.published, {}, "failed fit must not publish")

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
        """Identification only knows how to fit a single battery. With
        number_of_batteries > 1 the per-battery config values are lists, so it
        must skip with one clear warning instead of coercing a list with
        float() and crashing (or masking that crash as a generic failure)."""
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
        self.assertTrue(
            any("number_of_batteries=2" in m for m in cm.output),
            "must name the battery count in a clear skip warning",
        )
        self.assertFalse(
            any("TypeError" in m for m in cm.output),
            "must skip cleanly, not swallow a TypeError as a generic failure",
        )
        self.assertEqual(rh.published, {}, "N>1 must publish nothing")
        self.assertFalse(self._json_path().exists(), "N>1 must not write a fit result")


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


if __name__ == "__main__":
    unittest.main()
