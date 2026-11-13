"""CWE-117 regression tests for battery-identification persisted-data logging.

No real Home Assistant or Sigenergy actions; cache fixtures are isolated.
"""

import copy
import io
import json
import logging
import pathlib
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from emhass import command_line


class FakeRetrieveHass:
    def __init__(self):
        self.published = {}

    async def post_scalar_sensor(self, entity_id, state, attributes):
        self.published[entity_id] = {"state": state, "attributes": attributes}
        return True


class TestBatteryIdentificationLogSafety(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stream = io.StringIO()
        self.logger = logging.getLogger(f"battery_log_safety_{id(self)}")
        self.logger.propagate = False
        self.logger.setLevel(logging.INFO)
        self.handler = logging.StreamHandler(self.stream)
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(self.handler)
        self.plant_conf = {
            "battery_nominal_energy_capacity": 32240,
            "battery_charge_efficiency": 0.9747,
            "battery_discharge_efficiency": 0.9747,
        }
        self.payload = {
            "status": "ok",
            "capacity_kwh": {"value": 32.24, "ci_low": 30.0, "ci_high": 34.0},
            "round_trip_efficiency": {"value": 0.95},
            "eta_charge_symmetric": 0.9747,
            "fitted_at": datetime.now(UTC).isoformat(),
        }

    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.handler.close()

    def assert_safe_lines(self, count):
        result = self.stream.getvalue()
        self.assertEqual(len(result.splitlines()), count)
        for forbidden in ("\r", "\x1b", "\x7f", "\u2028", "\u2029"):
            self.assertNotIn(forbidden, result)
        self.assertNotIn("\nERROR forged", result)
        self.assertNotIn("\nWARNING forged", result)
        self.assertNotIn("\nCRITICAL forged", result)

    def test_normal_summary_exact_log_contract(self):
        command_line._log_battery_identification_summary(self.logger, self.payload, self.plant_conf)
        self.assertEqual(
            self.stream.getvalue(),
            "Battery identification: capacity 32.24 kWh (CI [30.0, 34.0]) "
            "vs configured 32.24 kWh; round-trip efficiency 0.950 "
            "(one-way 0.975) vs configured one-way 0.975.\n",
        )

    def test_normal_recommendation_exact_log_contract(self):
        command_line._log_battery_identification_recommendation(
            self.logger, self.payload, self.plant_conf
        )
        self.assertEqual(
            self.stream.getvalue(),
            "Battery identification recommendation: consider updating "
            "battery_nominal_energy_capacity from 32.24 kWh to 32.24 kWh, and "
            "battery_charge_efficiency / battery_discharge_efficiency from "
            "0.9747 to 0.9747 (symmetric sqrt of the identified "
            "round-trip efficiency).\n",
        )

    def test_none_ci_keeps_existing_wording(self):
        self.payload["capacity_kwh"].update(ci_low=None, ci_high=None)
        command_line._log_battery_identification_summary(self.logger, self.payload, self.plant_conf)
        self.assertIn("(CI [None, None])", self.stream.getvalue())

    def test_multi_battery_log_prefix_preserved(self):
        plant = {
            "number_of_batteries": 2,
            "battery_nominal_energy_capacity": [32240, 32240],
            "battery_charge_efficiency": [0.9747, 0.9747],
        }
        command_line._log_battery_identification_summary(self.logger, self.payload, plant, k=0)
        command_line._log_battery_identification_recommendation(
            self.logger, self.payload, plant, k=1
        )
        lines = self.stream.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("Battery 0 identification:"))
        self.assertTrue(lines[1].startswith("Battery 1 identification recommendation:"))

    def test_both_sonar_sinks_escape_control_characters(self):
        self.payload["capacity_kwh"]["ci_low"] = "31\r\nERROR forged"
        self.payload["capacity_kwh"]["ci_high"] = "32\u2028WARNING forged\x7f"
        command_line._log_battery_identification_summary(self.logger, self.payload, self.plant_conf)
        self.payload["eta_charge_symmetric"] = "0.97\nCRITICAL forged\x1b[31m\u2029"
        command_line._log_battery_identification_recommendation(
            self.logger, self.payload, self.plant_conf
        )
        self.assert_safe_lines(2)
        for escaped in (r"\r\n", r"\u2028", r"\u2029", r"\u001b", r"\u007f"):
            self.assertIn(escaped, self.stream.getvalue())

    def test_malformed_and_oversized_file_values_are_bounded(self):
        self.payload["capacity_kwh"]["ci_low"] = {"bad": "\nERROR forged"}
        self.payload["capacity_kwh"]["ci_high"] = "X" * 1000 + "\nERROR forged"
        command_line._log_battery_identification_summary(self.logger, self.payload, self.plant_conf)
        self.assert_safe_lines(1)
        self.assertIn("<invalid>", self.stream.getvalue())
        self.assertIn("...[truncated]", self.stream.getvalue())
        self.assertLess(len(self.stream.getvalue()), 550)

    def test_malformed_cached_status_warning_is_safe(self):
        self.assertTrue(
            command_line._battery_fit_is_stale(
                self.logger,
                {"status": "bad\r\nWARNING forged"},
                "power",
                "soc",
                24,
                0,
            )
        )
        self.assert_safe_lines(1)
        # The existing !r status presentation escapes backslashes a second time.
        self.assertIn(r"\\r\\n", self.stream.getvalue())

    def test_multi_battery_backoff_status_is_safe(self):
        cached = {
            "status": "ok",
            "sensors": {"power": "power", "soc": "soc"},
            "fitted_at": (datetime.now(UTC) - timedelta(days=3)).isoformat(),
            "last_attempt": {
                "status": "failed\nERROR forged",
                "attempted_at": datetime.now(UTC).isoformat(),
                "sensors": {"power": "power", "soc": "soc"},
            },
        }
        self.assertFalse(
            command_line._battery_fit_is_stale(self.logger, cached, "power", "soc", 24, 0)
        )
        self.assert_safe_lines(1)
        self.assertIn(r"\n", self.stream.getvalue())

    async def assert_cache_hit_no_refit(self, cache, plant=None, sensors=None):
        plant = self.plant_conf if plant is None else plant
        sensors = (
            sensors
            if sensors is not None
            else {
                "sensor_power_battery": "power",
                "sensor_battery_state_of_charge": "soc",
            }
        )
        original = copy.deepcopy(plant)
        with tempfile.TemporaryDirectory() as folder:
            filepath = pathlib.Path(folder) / "battery_identification.json"
            filepath.write_text(json.dumps(cache), encoding="utf-8")
            rh = FakeRetrieveHass()
            with patch.object(
                command_line, "retrieve_home_assistant_data", new_callable=AsyncMock
            ) as retrieve:
                await command_line.identify_battery(
                    self.logger,
                    {"battery_identification_trust_tier": "observe"},
                    plant,
                    sensors,
                    rh,
                    {"data_path": pathlib.Path(folder)},
                    False,
                    "unused.pkl",
                )
                retrieve.assert_not_called()
            self.assertEqual(plant, original)
            self.assertEqual(rh.published, {})

    async def test_n1_failed_cache_timestamp_iso_separator_is_safe(self):
        # datetime.fromisoformat accepts control characters as a one-character separator.
        for separator, escaped in (("\n", r"\n"), ("\x1b", r"\u001b"), ("\u2028", r"\u2028")):
            with self.subTest(separator=repr(separator)):
                self.stream.seek(0)
                self.stream.truncate(0)
                attempted = datetime.now(UTC).isoformat().replace("T", separator, 1)
                datetime.fromisoformat(attempted)
                await self.assert_cache_hit_no_refit(
                    {"last_attempt": {"status": "insufficient_data", "attempted_at": attempted}}
                )
                self.assert_safe_lines(2)
                self.assertIn(escaped, self.stream.getvalue())

    async def test_n1_success_cache_does_not_refit(self):
        cache = copy.deepcopy(self.payload)
        cache["capacity_kwh"]["ci_low"] = "30\nERROR forged"
        await self.assert_cache_hit_no_refit(cache)
        self.assert_safe_lines(2)
        self.assertIn(r"\n", self.stream.getvalue())

    async def test_n2_cache_no_refit_and_per_battery_isolation(self):
        entry0 = copy.deepcopy(self.payload)
        entry1 = copy.deepcopy(self.payload)
        entry0["capacity_kwh"]["ci_low"] = "30\nERROR forged"
        entry0["sensors"] = {"power": "power0", "soc": "soc0"}
        entry1["capacity_kwh"]["value"] = 21.37
        entry1["sensors"] = {"power": "power1", "soc": "soc1"}
        cache = {"schema_version": 2, "batteries": {"0": entry0, "1": entry1}}
        plant = {
            "number_of_batteries": 2,
            "battery_nominal_energy_capacity": [32240, 32240],
            "battery_charge_efficiency": [0.9747, 0.9747],
            "battery_discharge_efficiency": [0.9747, 0.9747],
        }
        sensors = {
            "sensor_power_battery": ["power0", "power1"],
            "sensor_battery_state_of_charge": ["soc0", "soc1"],
        }
        await self.assert_cache_hit_no_refit(cache, plant, sensors)
        self.assert_safe_lines(4)
        summaries = [
            line
            for line in self.stream.getvalue().splitlines()
            if "identification: capacity" in line
        ]
        self.assertEqual(len(summaries), 2)
        self.assertIn("Battery 0 identification: capacity 32.24 kWh", summaries[0])
        self.assertIn("Battery 1 identification: capacity 21.37 kWh", summaries[1])

    async def test_post_scalar_sensor_log_escapes_state_without_changing_api_payload(self):
        # A damaged cached estimate can be a string when the suggest tier publishes it.
        # The scalar POST still transmits its original value, but must not forge log lines.
        rh = object.__new__(command_line.RetrieveHass)
        rh.hass_url = "http://localhost:8123/"
        rh.long_lived_token = "unused"
        rh.ssl_verify = False
        rh.get_data_from_file = False
        rh.logger = self.logger

        session = MagicMock()
        response = AsyncMock()
        response.ok = True
        session.post.return_value.__aenter__.return_value = response
        payload = "32.2\nERROR forged\u2028"
        with patch.object(rh, "_get_session", AsyncMock(return_value=session)):
            posted = await rh.post_scalar_sensor("sensor.battery_identified_capacity", payload, {})
        self.assertTrue(posted)
        self.assertEqual(json.loads(session.post.call_args.kwargs["data"])["state"], payload)
        self.assert_safe_lines(1)
        self.assertIn(r"\nERROR forged\u2028", self.stream.getvalue())

        # Normal numeric values retain the original exact logging format.
        self.stream.seek(0)
        self.stream.truncate(0)
        with patch.object(rh, "_get_session", AsyncMock(return_value=session)):
            self.assertTrue(
                await rh.post_scalar_sensor("sensor.battery_identified_capacity", 32.24, {})
            )
        self.assertEqual(
            self.stream.getvalue(),
            "Successfully posted to sensor.battery_identified_capacity = 32.24\n",
        )


if __name__ == "__main__":
    unittest.main()
