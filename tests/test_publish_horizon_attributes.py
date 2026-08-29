#!/usr/bin/env python
"""Tests for the opt-in ``publish_horizon_attributes`` publication control (#1077).

Default (unset) / explicit ``true`` keeps the historical behaviour: the
forecast and schedule entities that expose future values carry the
optimisation horizon as a list/schedule state attribute. Explicit ``false``
keeps those entities compact - the current scalar state plus the metadata they
already have only - so large MPC horizons stay under Home Assistant's 16 KiB
Recorder attribute-size limit. The optimisation, its horizon, the saved entity
series and ``/api/v1/plan`` are all unaffected.
"""

import json
import pathlib
import unittest

import orjson
import pandas as pd

from emhass import utils
from emhass.retrieve_hass import RetrieveHass

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

logger, ch = utils.get_logger(__name__, emhass_conf, save_to_file=False)

TZ = "Europe/Paris"

# Every full-horizon list attribute get_attr_data_dict / post_data can emit.
HORIZON_KEYS = {
    "forecasts",
    "deferrables_schedule",
    "predicted_temperatures",
    "battery_scheduled_power",
    "battery_scheduled_soc",
    "unit_load_cost_forecasts",
    "unit_prod_price_forecasts",
    "scheduled_forecast",
    "heating_demand_forecast",
    "schedule",
}


def _make_rh(publish_horizon_attributes=None):
    """Build a file-mode RetrieveHass, optionally with a static config value set."""
    params = None
    if publish_horizon_attributes is not None:
        params = orjson.dumps(
            {"retrieve_hass_conf": {"publish_horizon_attributes": publish_horizon_attributes}}
        ).decode()
    return RetrieveHass(
        "http://localhost:8123/",
        "token",
        pd.Timedelta("30min"),
        TZ,
        params,
        emhass_conf,
        logger,
        get_data_from_file=True,
    )


def _power_series(periods=6):
    index = pd.date_range("2024-01-01 00:00", periods=periods, freq="30min", tz=TZ)
    return pd.Series([100.0 + 10 * i for i in range(periods)], index=index, name="P_PV")


def _label_series():
    index = pd.date_range("2024-01-01 00:00", periods=6, freq="30min", tz=TZ)
    return pd.Series(["off", "off", "on", "on", "variable", "off"], index=index, name="def0")


class TestPublishHorizonAttributesResolution(unittest.TestCase):
    """The resolved boolean follows the normal EMHASS config/runtime precedence."""

    def test_config_default_is_true(self):
        defaults = json.loads(emhass_conf["defaults_path"].read_text(encoding="utf-8"))
        self.assertIs(defaults["publish_horizon_attributes"], True)
        self.assertTrue(_make_rh().publish_horizon_attributes)

    def test_unset_resolves_true(self):
        # No retrieve_hass_conf entry at all -> historical behaviour.
        self.assertTrue(_make_rh(publish_horizon_attributes=None).publish_horizon_attributes)

    def test_static_false_is_resolved(self):
        self.assertFalse(_make_rh(publish_horizon_attributes=False).publish_horizon_attributes)

    def test_static_true_is_resolved(self):
        self.assertTrue(_make_rh(publish_horizon_attributes=True).publish_horizon_attributes)

    def test_association_row_registered(self):
        rows = emhass_conf["associations_path"].read_text(encoding="utf-8").splitlines()
        self.assertIn(
            "retrieve_hass_conf,publish_horizon_attributes,publish_horizon_attributes", rows
        )


class TestPublishHorizonAttributesRuntime(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_override_follows_precedence(self):
        """A runtimeparam overrides the static value via the shared associations path."""
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
        params = await utils.build_params(emhass_conf, secrets, config, logger)
        params_json = orjson.dumps(params).decode()
        rhc, oc, pc = utils.get_yaml_parse(params_json, logger)
        params_out, rhc_out, _oc, _pc = await utils.treat_runtimeparams(
            orjson.dumps({"publish_horizon_attributes": False}).decode(),
            params_json,
            rhc,
            oc,
            pc,
            "naive-mpc-optim",
            logger,
            emhass_conf,
        )
        if isinstance(params_out, str):
            params_out = json.loads(params_out)
        self.assertIs(params_out["retrieve_hass_conf"]["publish_horizon_attributes"], False)
        # And the resolved instance built from those params is compact.
        rh = RetrieveHass(
            "http://localhost:8123/",
            "token",
            pd.Timedelta("5min"),
            TZ,
            orjson.dumps(params_out).decode(),
            emhass_conf,
            logger,
            get_data_from_file=True,
        )
        self.assertFalse(rh.publish_horizon_attributes)


class TestPublishHorizonAttributesBehaviour(unittest.IsolatedAsyncioTestCase):
    async def _post(self, rh, series, type_var, device_class="power", unit="W"):
        _resp, data = await rh.post_data(
            series,
            1,
            f"sensor.{series.name}",
            device_class,
            unit,
            f"{series.name} friendly",
            type_var=type_var,
        )
        return data

    async def test_default_keeps_horizon_attributes(self):
        rh = _make_rh()
        data = await self._post(rh, _power_series(), "power")
        self.assertIn("forecasts", data["attributes"])
        self.assertIsInstance(data["attributes"]["forecasts"], list)

    async def test_explicit_true_keeps_horizon_attributes(self):
        rh = _make_rh(publish_horizon_attributes=True)
        data = await self._post(rh, _power_series(), "power")
        self.assertIn("forecasts", data["attributes"])

    async def test_false_drops_only_the_horizon_list(self):
        rh = _make_rh(publish_horizon_attributes=False)
        series = _power_series()
        data = await self._post(rh, series, "power")
        attrs = data["attributes"]
        # Horizon list gone...
        self.assertNotIn("forecasts", attrs)
        self.assertFalse(HORIZON_KEYS & set(attrs))
        # ...but the current scalar state and normal metadata are untouched.
        self.assertEqual(data["state"], f"{series.iloc[1]:.2f}")
        self.assertEqual(attrs["device_class"], "power")
        self.assertEqual(attrs["unit_of_measurement"], "W")
        self.assertEqual(attrs["friendly_name"], "P_PV friendly")
        self.assertEqual(attrs["state_class"], "measurement")

    async def test_false_matches_true_except_for_horizon_key(self):
        series = _power_series()
        full = await self._post(_make_rh(publish_horizon_attributes=True), series, "power")
        compact = await self._post(_make_rh(publish_horizon_attributes=False), series, "power")
        self.assertEqual(full["state"], compact["state"])
        self.assertEqual(
            {k: v for k, v in full["attributes"].items() if k != "forecasts"},
            compact["attributes"],
        )

    async def test_false_covers_battery_and_price_types(self):
        rh = _make_rh(publish_horizon_attributes=False)
        for type_var, dc, unit, key in [
            ("batt", "power", "W", "battery_scheduled_power"),
            ("SOC", "battery", "%", "battery_scheduled_soc"),
            ("unit_load_cost", "monetary", "EUR/kWh", "unit_load_cost_forecasts"),
        ]:
            series = _power_series()
            series.name = key
            data = await self._post(rh, series, type_var, device_class=dc, unit=unit)
            self.assertNotIn(key, data["attributes"])
            self.assertEqual(data["attributes"]["friendly_name"], f"{key} friendly")

    async def test_categorical_true_keeps_schedule(self):
        rh = _make_rh(publish_horizon_attributes=True)
        _resp, data = await rh.post_data(
            _label_series(), 2, "sensor.def0_state", "enum", "", "Def 0", type_var="categorical"
        )
        self.assertIn("schedule", data["attributes"])
        self.assertEqual(data["state"], "on")

    async def test_categorical_false_drops_schedule_keeps_state(self):
        rh = _make_rh(publish_horizon_attributes=False)
        _resp, data = await rh.post_data(
            _label_series(), 2, "sensor.def0_state", "enum", "", "Def 0", type_var="categorical"
        )
        self.assertNotIn("schedule", data["attributes"])
        # No new metadata invented for the categorical entity - friendly_name only,
        # exactly as today.
        self.assertEqual(data["state"], "on")
        self.assertEqual(data["attributes"], {"friendly_name": "Def 0"})

    async def test_h288_compact_mode_does_not_serialize_the_horizon(self):
        """A 288-step (24 h @ 5 min) horizon: compact payload carries no arrays."""
        index = pd.date_range("2024-01-01 00:00", periods=288, freq="5min", tz=TZ)
        series = pd.Series(range(288), index=index, name="P_PV", dtype="float64")

        full = await RetrieveHass(
            "http://localhost:8123/",
            "token",
            pd.Timedelta("5min"),
            TZ,
            None,
            emhass_conf,
            logger,
            get_data_from_file=True,
        ).post_data(series, 0, "sensor.P_PV", "power", "W", "PV", type_var="power")
        compact = await _make_rh(publish_horizon_attributes=False).post_data(
            series, 0, "sensor.P_PV", "power", "W", "PV", type_var="power"
        )

        # The full 288-step horizon is still built when enabled...
        self.assertEqual(len(full[1]["attributes"]["forecasts"]), 288)
        # ...and entirely absent in compact mode - that absence is the contract.
        self.assertFalse(HORIZON_KEYS & set(compact[1]["attributes"]))
        # The compact payload is also comfortably under HA's 16 KiB Recorder limit
        # and far smaller than the full one.
        self.assertLess(len(orjson.dumps(compact[1]["attributes"])), 16 * 1024)
        self.assertLess(
            len(orjson.dumps(compact[1]["attributes"])) * 10,
            len(orjson.dumps(full[1]["attributes"])),
        )


class TestPublishHorizonAttributesStaticMethod(unittest.TestCase):
    """get_attr_data_dict stays backward compatible and honours include_horizon."""

    def _df(self):
        index = pd.date_range("2024-01-01 00:00", periods=4, freq="30min", tz=TZ)
        return pd.DataFrame({"sensor.p_pv_forecast": [1.0, 2.0, 3.0, 4.0]}, index=index)

    def test_default_includes_horizon(self):
        data = RetrieveHass.get_attr_data_dict(
            self._df(), 0, "sensor.p_pv_forecast", "power", "W", "PV", "forecasts", 1.0
        )
        self.assertEqual(len(data["attributes"]["forecasts"]), 4)

    def test_include_horizon_false_omits_list_only(self):
        data = RetrieveHass.get_attr_data_dict(
            self._df(),
            0,
            "sensor.p_pv_forecast",
            "power",
            "W",
            "PV",
            "forecasts",
            1.0,
            include_horizon=False,
        )
        self.assertNotIn("forecasts", data["attributes"])
        self.assertEqual(data["attributes"]["device_class"], "power")
        self.assertEqual(data["attributes"]["unit_of_measurement"], "W")
        self.assertEqual(data["attributes"]["friendly_name"], "PV")
        self.assertEqual(data["attributes"]["state_class"], "measurement")
        self.assertEqual(data["state"], "1.00")


if __name__ == "__main__":
    unittest.main()
