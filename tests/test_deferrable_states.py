#!/usr/bin/env python
"""Tests for the generic per-deferrable command-state sensors.

These exercise the opt-in ``publish_deferrable_load_states`` feature: the
``_deferrable_power_to_state`` power->label mapping and the ``categorical``
``post_data`` type that publishes a string state plus a full 'schedule'
attribute. They do not require any optional extra.
"""

import pathlib
import unittest

import pandas as pd

from emhass import utils
from emhass.retrieve_hass import RetrieveHass

# The root folder
root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)


class TestDeferrableStateInterpretation(unittest.IsolatedAsyncioTestCase):
    """Tests for the generic per-deferrable command-state interpretation sensors."""

    def test_power_to_state_mapping(self):
        from emhass.command_line import _deferrable_power_to_state

        # off when idle, on at nominal, variable in between.
        self.assertEqual(_deferrable_power_to_state(0.0, 3000.0), "off")
        self.assertEqual(_deferrable_power_to_state(3000.0, 3000.0), "on")
        self.assertEqual(_deferrable_power_to_state(1500.0, 3000.0), "variable")
        # No nominal known -> any running power is 'variable', idle is 'off'.
        self.assertEqual(_deferrable_power_to_state(1500.0, 0.0), "variable")
        self.assertEqual(_deferrable_power_to_state(0.0, 0.0), "off")
        # Non-finite is treated as off (defensive).
        self.assertEqual(_deferrable_power_to_state(float("nan"), 3000.0), "off")

    async def test_categorical_post_data(self):
        """post_data(type_var='categorical') yields a string state + schedule."""
        tz = "Europe/Paris"
        index = pd.date_range("2024-01-01 00:00", periods=6, freq="30min", tz=tz)
        states = pd.Series(["off", "off", "on", "on", "variable", "off"], index=index)
        rh = RetrieveHass(
            "http://localhost:8123/",
            "token",
            pd.Timedelta("30min"),
            tz,
            None,
            emhass_conf,
            logger,
            get_data_from_file=True,
        )
        _response, data = await rh.post_data(
            states,
            2,
            "sensor.p_deferrable0_state",
            "enum",
            "",
            "Deferrable Load 0 Command",
            type_var="categorical",
        )
        self.assertEqual(data["state"], "on")  # idx 2
        self.assertEqual(data["attributes"]["friendly_name"], "Deferrable Load 0 Command")
        schedule = data["attributes"]["schedule"]
        self.assertIsInstance(schedule, list)
        # Schedule runs from the current index to the end (4 remaining slots).
        self.assertEqual(len(schedule), 4)
        self.assertEqual(schedule[0]["value"], "on")
        self.assertEqual(schedule[-1]["value"], "off")
        self.assertIn("date", schedule[0])


if __name__ == "__main__":
    unittest.main()
