import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RP = REPO / "src" / "emhass" / "static" / "data" / "runtime_params.json"
# Config surfaces a runtime-only key must NOT appear in (see test_keys_disjoint_from_config).
PARAM_DEFINITIONS = REPO / "src" / "emhass" / "static" / "data" / "param_definitions.json"
CONFIG_DEFAULTS = REPO / "src" / "emhass" / "data" / "config_defaults.json"

VALID_INPUT = {
    "int",
    "float",
    "boolean",
    "string",
    "object",
    "select",
    "time",
    "array.int",
    "array.float",
    "array.boolean",
    "array.string",
    "array.time",
    "array.array.float",
}
VALID_UNIT = {
    "W",
    "Wh",
    "kWh",
    "¤/kWh",
    "€",
    "%",
    "fraction",
    "°C",
    "°",
    "min",
    "h",
    "days",
    "timesteps",
    "count",
    "s",
    "none",
}
# Locked by the AC-2b completeness scan of treat_runtimeparams (utils.py):
# the runtime-only optimization knobs that are in NEITHER param_definitions.json
# NOR config_defaults.json. adjusted_pv_model_max_age / open_meteo_cache_max_age
# were dropped after the scan found them to be config params (Bucket C).
EXPECTED_KEYS = {
    "prediction_horizon",
    "soc_init",
    "soc_final",
    "operating_timesteps_of_each_deferrable_load",
    "alpha",
    "beta",
    "weather_forecast_cache",
    "weather_forecast_cache_only",
    "def_current_state",
    "def_load_config",
}


def _entries():
    data = json.loads(RP.read_text(encoding="utf-8"))
    for section, params in data.items():
        assert isinstance(params, dict), f"section {section} is not an object"
        yield from params.items()


class TestRuntimeParams(unittest.TestCase):
    def test_file_parses(self):
        self.assertTrue(RP.exists(), f"missing {RP}")
        json.loads(RP.read_text(encoding="utf-8"))

    def test_required_fields(self):
        for key, val in _entries():
            for field in ("friendly_name", "Description", "input", "default_value", "unit"):
                self.assertIn(field, val, f"{key} missing {field}")

    def test_input_and_unit_enums(self):
        for key, val in _entries():
            self.assertIn(val["input"], VALID_INPUT, f"{key} bad input {val['input']!r}")
            self.assertIn(val["unit"], VALID_UNIT, f"{key} bad unit {val['unit']!r}")

    def test_no_applies_to(self):
        # applicability lives in the openapi operations (AM-1b), not the data file
        for key, val in _entries():
            self.assertNotIn("applies_to", val, f"{key} must not carry applies_to")

    def test_expected_keys_present(self):
        keys = {
            k for _s, params in json.loads(RP.read_text(encoding="utf-8")).items() for k in params
        }
        missing = EXPECTED_KEYS - keys
        self.assertEqual(missing, set(), f"missing expected keys: {missing}")

    def test_keys_disjoint_from_config(self):
        # Runtime-only contract: a runtime_params.json key must appear in NEITHER
        # param_definitions.json (the config/GUI form surface) NOR config_defaults.json.
        # Overlap means the key is a config parameter overridable at runtime (a Bucket-C
        # key), not a runtime-only knob, so it must not live here. Guards future drift.
        runtime_keys = {
            k for _s, params in json.loads(RP.read_text(encoding="utf-8")).items() for k in params
        }
        param_def = json.loads(PARAM_DEFINITIONS.read_text(encoding="utf-8"))
        config_keys = {k for _s, params in param_def.items() for k in params}
        config_keys |= set(json.loads(CONFIG_DEFAULTS.read_text(encoding="utf-8")).keys())
        overlap = runtime_keys & config_keys
        self.assertEqual(overlap, set(), f"runtime-only keys must not be config params: {overlap}")


if __name__ == "__main__":
    unittest.main()
