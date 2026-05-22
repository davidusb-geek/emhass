import json
import re
from pathlib import Path

# Keys in config_defaults.json with no param_definitions.json entry.
# These are ML-subsystem parameters not represented in the UI schema.
_KNOWN_UNDECLARED_DEFAULTS = frozenset(
    {
        "data_path",
        "deferrable_load_groups",
        "model_type",
        "num_lags",
        "perform_backtest",
        "sklearn_model",
        "split_date_delta",
        "var_model",
    }
)

# Keys whose default value type doesn't match their declared input type.
# Pre-existing mismatches outside the scope of the #876/#879 bug-wave.
_KNOWN_TYPE_MISMATCHES = frozenset(
    {
        "load_peak_hour_periods",  # declared array.time, default is dict
    }
)

# Mapping from param_definitions "input" type to acceptable Python types for defaults.
_INPUT_TYPE_TO_PY_TYPES: dict[str, tuple[type, ...]] = {
    "int": (int, type(None)),
    "float": (int, float, type(None)),
    "string": (str, type(None)),
    "boolean": (bool,),
    "time": (str, type(None)),
    "select": (str, type(None)),
    "array.int": (list, type(None)),
    "array.float": (list, type(None)),
    "array.string": (list, type(None)),
    "array.boolean": (list, type(None)),
    "array.time": (list, type(None)),
    "array.array.float": (list, type(None)),
    "object": (dict, type(None)),
}


def _extract_switch_body(js_src: str, fn_name: str) -> str:
    """Return the body of the first switch block inside fn_name.

    Narrower than full-function extraction: brace-counts only the switch
    block, so extra braces in the surrounding function don't matter.
    """
    fn_m = re.search(rf"function\s+{re.escape(fn_name)}\s*\([^)]*\)\s*\{{", js_src)
    if not fn_m:
        raise AssertionError(f"function {fn_name!r} not found in JS source")
    sw_m = re.search(r"\bswitch\s*\([^)]+\)\s*\{", js_src[fn_m.end() :])
    if not sw_m:
        raise AssertionError(f"no switch statement found in function {fn_name!r}")
    start = fn_m.end() + sw_m.end()
    depth = 1
    i = start
    while i < len(js_src) and depth > 0:
        if js_src[i] == "{":
            depth += 1
        elif js_src[i] == "}":
            depth -= 1
        i += 1
    return js_src[start : i - 1]


def test_param_definitions_input_types_have_renderer_cases():
    """Every distinct 'input' value in param_definitions.json must have a
    matching case in the buildParamElement switch in configuration_script.js.

    Pre-#872 master would have failed with {'array.array.float', 'object'}
    missing from the switch.
    """
    pd_path = Path("src/emhass/static/data/param_definitions.json")
    pd = json.loads(pd_path.read_text(encoding="utf-8"))

    declared_inputs: set[str] = set()
    for section_entries in pd.values():
        for name, definition in section_entries.items():
            input_type = definition.get("input")
            assert input_type is not None, f"{name!r} has no 'input' field"
            declared_inputs.add(input_type)

    js_path = Path("src/emhass/static/configuration_script.js")
    js_src = js_path.read_text(encoding="utf-8")
    switch_body = _extract_switch_body(js_src, "buildParamElement")
    cases = set(re.findall(r'case\s+"([^"]+)"\s*:', switch_body))

    missing = declared_inputs - cases
    assert not missing, (
        f"param_definitions.json declares input types with no matching case in "
        f"buildParamElement: {sorted(missing)}. "
        f"Add a case in src/emhass/static/configuration_script.js or "
        f"remove the input type from param_definitions.json."
    )


def test_config_defaults_keys_match_param_definitions():
    """Every key in config_defaults.json must have a param_definitions.json entry,
    and its default value must be type-compatible with the declared 'input' type.

    Targeted check: an 'object' input with default value 'null' (the string)
    instead of JSON null will fail — this is the #876 footgun.

    Known pre-existing exceptions are in _KNOWN_UNDECLARED_DEFAULTS and
    _KNOWN_TYPE_MISMATCHES; new violations will cause this test to fail.
    """
    pd = json.loads(
        Path("src/emhass/static/data/param_definitions.json").read_text(encoding="utf-8")
    )
    cd = json.loads(Path("src/emhass/data/config_defaults.json").read_text(encoding="utf-8"))

    pd_flat: dict[str, str] = {}
    for section_entries in pd.values():
        for name, definition in section_entries.items():
            pd_flat[name] = definition["input"]

    # Check 1: every config_defaults key must appear in param_definitions
    new_undeclared = (set(cd.keys()) - set(pd_flat.keys())) - _KNOWN_UNDECLARED_DEFAULTS
    assert not new_undeclared, (
        f"config_defaults.json keys with no entry in param_definitions.json: "
        f"{sorted(new_undeclared)}. "
        f"Add an entry to src/emhass/static/data/param_definitions.json."
    )

    # Check 2: every input type declared in param_definitions must be mapped
    unmapped = set(pd_flat.values()) - set(_INPUT_TYPE_TO_PY_TYPES.keys())
    assert not unmapped, (
        f"param_definitions.json uses input types not in _INPUT_TYPE_TO_PY_TYPES: "
        f"{sorted(unmapped)}. Add them to the mapping in test_schema_contract.py."
    )

    # Check 3: each default value's Python type must match the declared input type
    mismatches: list[str] = []
    for key, default_value in cd.items():
        if key not in pd_flat or key in _KNOWN_TYPE_MISMATCHES:
            continue
        input_type = pd_flat[key]
        acceptable = _INPUT_TYPE_TO_PY_TYPES[input_type]
        # Targeted #876 footgun: string "null" for an object-typed param
        if input_type == "object" and default_value == "null":
            mismatches.append(
                f"{key}: default is the string 'null', not JSON null — "
                f"callers expecting a dict will crash (the #876 footgun)"
            )
            continue
        if not isinstance(default_value, acceptable):
            mismatches.append(
                f"{key}: default is {type(default_value).__name__} {default_value!r}, "
                f"declared input is {input_type!r} "
                f"(expected one of {[t.__name__ for t in acceptable]})"
            )

    assert not mismatches, (
        "config_defaults.json has values incompatible with their declared input type:\n  - "
        + "\n  - ".join(mismatches)
    )
