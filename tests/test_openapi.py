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


if __name__ == "__main__":
    unittest.main()
