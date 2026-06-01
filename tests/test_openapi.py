import json
import sys
import unittest
from pathlib import Path

# scripts/ is not a package — add it to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_openapi as gen  # noqa: E402


class TestInputToSchema(unittest.TestCase):
    def test_select_becomes_string_enum(self):
        p = {"input": "select", "select_options": ["a", "b"], "default_value": "a",
             "friendly_name": "F", "Description": "D", "unit": "none"}
        s = gen._input_to_schema("select", p)
        self.assertEqual(s["type"], "string")
        self.assertEqual(s["enum"], ["a", "b"])
        self.assertEqual(s["default"], "a")

    def test_int_and_float(self):
        self.assertEqual(gen._input_to_schema("int", {"input": "int", "default_value": 5})["type"], "integer")
        self.assertEqual(gen._input_to_schema("float", {"input": "float", "default_value": 1.0})["type"], "number")

    def test_array_float_wraps_items_with_scalar_default(self):
        p = {"input": "array.float", "default_value": 3000, "unit": "W", "friendly_name": "N", "Description": "D"}
        s = gen._input_to_schema("array.float", p)
        self.assertEqual(s["type"], "array")
        self.assertEqual(s["items"]["type"], "number")
        self.assertEqual(s["items"]["default"], 3000)   # per-element template default
        self.assertNotIn("default", s)                  # not on the array itself
        self.assertEqual(s["x-unit"], "W")              # unit annotation on the property

    def test_nested_array_array_float(self):
        s = gen._input_to_schema("array.array.float", {"input": "array.array.float", "default_value": None})
        self.assertEqual(s["type"], "array")
        self.assertEqual(s["items"]["type"], "array")
        self.assertEqual(s["items"]["items"]["type"], "number")

    def test_object_and_time(self):
        self.assertEqual(gen._input_to_schema("object", {"input": "object", "default_value": None})["type"], "object")
        self.assertEqual(gen._input_to_schema("array.time", {"input": "array.time", "default_value": None})["items"]["type"], "string")

    def test_unknown_input_raises(self):
        with self.assertRaises(SystemExit):
            gen._input_to_schema("widget", {"input": "widget"})

    def test_null_default_omitted(self):
        s = gen._input_to_schema("float", {"input": "float", "default_value": None})
        self.assertNotIn("default", s)


class TestConfigComponent(unittest.TestCase):
    def _defs(self):
        return {
            "SectionA": {"costfun": {"input": "select", "select_options": ["profit", "cost"],
                                     "default_value": "profit", "friendly_name": "Cost", "Description": "d", "unit": "none"}},
            "SectionB": {"battery_power": {"input": "int", "default_value": 5, "friendly_name": "BP", "Description": "d", "unit": "W"}},
        }

    def test_flattens_all_sections_into_properties(self):
        comp = gen.build_config_component(self._defs())
        self.assertEqual(comp["type"], "object")
        self.assertIn("costfun", comp["properties"])
        self.assertIn("battery_power", comp["properties"])

    def test_duplicate_key_across_sections_raises(self):
        defs = self._defs()
        defs["SectionB"]["costfun"] = {"input": "int", "default_value": 1}
        with self.assertRaises(SystemExit):
            gen.build_config_component(defs)


class TestRouteGuard(unittest.TestCase):
    def test_undocumented_route_raises(self):
        routes = {("/healthz", "GET"), ("/brand-new", "GET")}
        curated = {"/healthz": {"GET"}}
        skip = {"/"}
        with self.assertRaises(SystemExit):
            gen.assert_no_undocumented(routes, curated, skip)

    def test_skiplisted_and_curated_ok(self):
        routes = {("/healthz", "GET"), ("/", "GET")}
        curated = {"/healthz": {"GET"}}
        skip = {"/"}
        gen.assert_no_undocumented(routes, curated, skip)  # no raise

    def test_discovered_routes_returns_pairs(self):
        routes = gen.discovered_routes()
        self.assertTrue(all(isinstance(p, tuple) and len(p) == 2 for p in routes))
        # the curated API routes must be discoverable (proves url_map import works)
        paths = {p for p, _ in routes}
        self.assertIn("/api/v1/last-run", paths)

    def test_curated_union_skip_covers_live_routes(self):
        # the live route set must be fully partitioned by CURATED ∪ SKIP
        gen.assert_no_undocumented(gen.discovered_routes(), gen.CURATED, gen.SKIP)


class TestBuildSpec(unittest.TestCase):
    def test_spec_has_core_shape_and_curated_paths(self):
        spec = gen.build_spec()
        self.assertEqual(spec["openapi"], "3.1.0")
        self.assertEqual(spec["info"]["title"], "EMHASS API")
        self.assertTrue(spec["info"]["version"])  # EMHASS_SCHEMA_VERSION
        for p in ("/get-config", "/set-config", "/get-json", "/action/{action_name}",
                  "/api/v1/last-run", "/healthz"):
            self.assertIn(p, spec["paths"], f"missing {p}")
        # HTML/UI routes absent
        for p in ("/", "/index", "/template", "/configuration", "/static/{filename}"):
            self.assertNotIn(p, spec["paths"])
        # config + inlined response components present
        self.assertIn("Config", spec["components"]["schemas"])
        self.assertIn("LastRun", spec["components"]["schemas"])
        self.assertIn("Healthz", spec["components"]["schemas"])

    def test_action_response_links_plan_output_doc(self):
        spec = gen.build_spec()
        action_201 = spec["paths"]["/action/{action_name}"]["post"]["responses"]["201"]
        self.assertIn("plan_output_schema.md", json.dumps(action_201))

    def test_action_request_is_open_object(self):
        spec = gen.build_spec()
        body = spec["paths"]["/action/{action_name}"]["post"]["requestBody"]
        schema = body["content"]["application/json"]["schema"]
        self.assertEqual(schema["type"], "object")
        self.assertTrue(schema["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
