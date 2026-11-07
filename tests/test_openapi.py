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
        p = {
            "input": "select",
            "select_options": ["a", "b"],
            "default_value": "a",
            "friendly_name": "F",
            "Description": "D",
            "unit": "none",
        }
        s = gen._input_to_schema("select", p)
        self.assertEqual(s["type"], "string")
        self.assertEqual(s["enum"], ["a", "b"])
        self.assertEqual(s["default"], "a")

    def test_int_and_float(self):
        self.assertEqual(
            gen._input_to_schema("int", {"input": "int", "default_value": 5})["type"], "integer"
        )
        self.assertEqual(
            gen._input_to_schema("float", {"input": "float", "default_value": 1.0})["type"],
            "number",
        )

    def test_array_float_wraps_items_with_scalar_default(self):
        p = {
            "input": "array.float",
            "default_value": 3000,
            "unit": "W",
            "friendly_name": "N",
            "Description": "D",
        }
        s = gen._input_to_schema("array.float", p)
        self.assertEqual(s["type"], "array")
        self.assertEqual(s["items"]["type"], "number")
        self.assertEqual(s["items"]["default"], 3000)  # per-element template default
        self.assertNotIn("default", s)  # not on the array itself
        self.assertEqual(s["x-unit"], "W")  # unit annotation on the property

    def test_nested_array_array_float(self):
        s = gen._input_to_schema(
            "array.array.float", {"input": "array.array.float", "default_value": None}
        )
        self.assertEqual(s["type"], "array")
        self.assertEqual(s["items"]["type"], "array")
        self.assertEqual(s["items"]["items"]["type"], "number")

    def test_object_and_time(self):
        self.assertEqual(
            gen._input_to_schema("object", {"input": "object", "default_value": None})["type"],
            "object",
        )
        self.assertEqual(
            gen._input_to_schema("array.time", {"input": "array.time", "default_value": None})[
                "items"
            ]["type"],
            "string",
        )

    def test_unknown_input_raises(self):
        with self.assertRaises(SystemExit):
            gen._input_to_schema("widget", {"input": "widget"})

    def test_null_default_omitted(self):
        s = gen._input_to_schema("float", {"input": "float", "default_value": None})
        self.assertNotIn("default", s)


class TestConfigComponent(unittest.TestCase):
    def _defs(self):
        return {
            "SectionA": {
                "costfun": {
                    "input": "select",
                    "select_options": ["profit", "cost"],
                    "default_value": "profit",
                    "friendly_name": "Cost",
                    "Description": "d",
                    "unit": "none",
                }
            },
            "SectionB": {
                "battery_power": {
                    "input": "int",
                    "default_value": 5,
                    "friendly_name": "BP",
                    "Description": "d",
                    "unit": "W",
                }
            },
        }

    def test_flattens_all_sections_into_properties(self):
        comp = gen.build_config_component(self._defs(), {})
        self.assertEqual(comp["type"], "object")
        self.assertIn("costfun", comp["properties"])
        self.assertIn("battery_power", comp["properties"])

    def test_duplicate_key_across_sections_raises(self):
        defs = self._defs()
        defs["SectionB"]["costfun"] = {"input": "int", "default_value": 1}
        with self.assertRaises(SystemExit):
            gen.build_config_component(defs, {})


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

    def test_normalise_path_converts_converters_and_args(self):
        self.assertEqual(gen._normalise_path("/action/<action_name>"), "/action/{action_name}")
        self.assertEqual(gen._normalise_path("/action/<int:action_id>"), "/action/{action_id}")
        self.assertEqual(gen._normalise_path("/static/<path:filename>"), "/static/{filename}")
        self.assertEqual(gen._normalise_path("/healthz"), "/healthz")  # no params untouched


class TestBuildSpec(unittest.TestCase):
    def test_build_spec_filters_paths_to_live_routes(self):
        # only routes present in the passed-in live set survive; other curated paths drop
        spec = gen.build_spec(routes={("/get-config", "GET"), ("/healthz", "GET")})
        self.assertEqual(set(spec["paths"]), {"/get-config", "/healthz"})
        self.assertNotIn("/get-json", spec["paths"])

    def test_spec_has_core_shape_and_curated_paths(self):
        spec = gen.build_spec()
        self.assertEqual(spec["openapi"], "3.1.0")
        self.assertEqual(spec["info"]["title"], "EMHASS API")
        self.assertTrue(spec["info"]["version"])  # EMHASS_SCHEMA_VERSION
        for p in (
            "/get-config",
            "/set-config",
            "/get-json",
            "/action/{action_name}",
            "/api/v1/last-run",
            "/healthz",
        ):
            self.assertIn(p, spec["paths"], f"missing {p}")
        # HTML/UI routes absent
        for p in ("/", "/index", "/template", "/configuration", "/static/{filename}"):
            self.assertNotIn(p, spec["paths"])
        # config + inlined response components present
        self.assertIn("Config", spec["components"]["schemas"])
        self.assertIn("LastRun", spec["components"]["schemas"])
        self.assertIn("Healthz", spec["components"]["schemas"])

    def test_action_operation_links_plan_output_doc(self):
        # externalDocs must sit on the Operation Object (valid in OpenAPI 3.1),
        # not on a Response Object (invalid — strict validators reject it).
        spec = gen.build_spec()
        action = spec["paths"]["/action/{action_name}"]["post"]
        self.assertIn("plan_output_schema.md", action["externalDocs"]["url"])
        self.assertNotIn("externalDocs", action["responses"]["200"])

    def test_action_documents_live_optimization_rejections(self):
        responses = gen.build_spec()["paths"]["/action/{action_name}"]["post"]["responses"]
        for code, phrase in (("429", "not run"), ("503", "failed closed")):
            self.assertIn(code, responses)
            description = responses[code]["description"]
            for action in ("naive-mpc-optim", "dayahead-optim", "perfect-optim"):
                self.assertIn(action, description)
            self.assertIn(phrase, description)
            header = responses[code]["headers"]["Retry-After"]
            self.assertEqual(header["schema"], {"type": "integer"})
        self.assertIn("not queued", responses["429"]["description"])
        for code in ("200", "400"):
            self.assertIn(code, responses)

    def test_committed_artifact_documents_live_optimization_rejections(self):
        committed = json.loads(
            (REPO_ROOT / "src" / "emhass" / "static" / "openapi.json").read_text(encoding="utf-8")
        )
        responses = committed["paths"]["/action/{action_name}"]["post"]["responses"]
        self.assertEqual(
            {c: responses[c] for c in ("429", "503")},
            {
                c: gen.build_spec()["paths"]["/action/{action_name}"]["post"]["responses"][c]
                for c in ("429", "503")
            },
        )
        self.assertIn("Retry-After", responses["429"]["headers"])
        self.assertIn("Retry-After", responses["503"]["headers"])

    def test_action_request_is_open_object(self):
        spec = gen.build_spec()
        body = spec["paths"]["/action/{action_name}"]["post"]["requestBody"]
        schema = body["content"]["application/json"]["schema"]
        self.assertEqual(schema["type"], "object")
        self.assertTrue(schema["additionalProperties"])


class TestOpenapiArtifact(unittest.TestCase):
    OUT = REPO_ROOT / "src" / "emhass" / "static" / "openapi.json"

    def test_load_json_strips_schema_and_id(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.schema.json"
            p.write_text(
                json.dumps({"$schema": "h", "$id": "i", "type": "object", "foo": 1}),
                encoding="utf-8",
            )
            loaded = gen._load_json(p)
        self.assertNotIn("$schema", loaded)
        self.assertNotIn("$id", loaded)
        self.assertEqual(loaded["foo"], 1)  # real content preserved

    def test_committed_file_is_valid(self):
        self.assertTrue(self.OUT.exists(), f"missing {self.OUT} — run scripts/generate_openapi.py")
        spec = json.loads(self.OUT.read_text(encoding="utf-8"))
        for key in ("openapi", "info", "paths", "components"):
            self.assertIn(key, spec)
        # every internal $ref resolves to a declared component
        names = set(spec["components"]["schemas"])
        text = json.dumps(spec)
        for ref in ("Config", "LastRun", "Healthz"):
            if f"#/components/schemas/{ref}" in text:
                self.assertIn(ref, names)

    def test_committed_matches_generated(self):
        committed = json.loads(self.OUT.read_text(encoding="utf-8"))
        fresh = gen.generate()
        self.assertEqual(
            committed,
            fresh,
            "openapi.json is stale — run `python scripts/generate_openapi.py` and commit",
        )

    def test_soc_fraction_fields_declare_maximum_one(self):
        spec = gen.build_spec()
        properties = spec["components"]["schemas"]["Config"]["properties"]
        for name in (
            "battery_minimum_state_of_charge",
            "battery_maximum_state_of_charge",
            "battery_target_state_of_charge",
            "battery_soc_deficit_threshold",
            "battery_soc_surplus_threshold",
        ):
            # A scalar default means /get-config returns a scalar at one battery, so the
            # property is "scalar or array"; the bound sits on both branches.
            scalar, array = properties[name]["anyOf"]
            self.assertEqual(scalar["maximum"], 1)
            self.assertEqual(array["items"]["maximum"], 1)

    def test_array_params_with_a_scalar_default_accept_a_scalar_or_an_array(self):
        properties = gen.build_spec()["components"]["schemas"]["Config"]["properties"]
        for name, element_type in (
            ("inverter_efficiency_dc_ac", "number"),
            ("inverter_ac_output_max", "integer"),
            ("compute_curtailment", "boolean"),
            ("battery_charge_power_max", "integer"),
        ):
            with self.subTest(name=name):
                prop = properties[name]
                self.assertNotIn("type", prop)
                scalar, array = prop["anyOf"]
                self.assertEqual(scalar["type"], element_type)
                self.assertEqual(array["type"], "array")
                self.assertEqual(array["items"]["type"], element_type)
                self.assertIn("default", array["items"])  # the element default stays where it was
                self.assertIn("title", prop)
                self.assertIn("description", prop)
        self.assertEqual(properties["battery_charge_power_max"]["x-unit"], "W")
        self.assertNotIn("x-unit", properties["battery_charge_power_max"]["anyOf"][0])

    def test_array_params_with_a_null_or_list_default_stay_plain_arrays(self):
        properties = gen.build_spec()["components"]["schemas"]["Config"]["properties"]
        for name in (
            "battery_inverter_index",  # null default
            "pv_plant_inverter_index",
            "inverter_power_curve_dc_ac",  # list default
            "nominal_power_of_deferrable_loads",
        ):
            with self.subTest(name=name):
                self.assertEqual(properties[name]["type"], "array")
                self.assertNotIn("anyOf", properties[name])


if __name__ == "__main__":
    unittest.main()
