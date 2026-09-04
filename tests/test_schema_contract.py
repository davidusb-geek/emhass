import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

_NODE = shutil.which("node")
# Node cold-start can be slow on loaded Windows CI runners; 30 s still flaked
# (test_minus_elements_does_not_crash_on_zero_inputs timed out). Give it headroom.
_NODE_TIMEOUT_S = 120

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
# The battery_* / weight_battery_* block (#610): declared array.* because they
# accept per-battery lists, but their defaults stay bare scalars on purpose -
# a scalar broadcasts to every battery and keeps existing single-battery
# configs byte-identical (number_of_batteries defaults to 1).
_KNOWN_TYPE_MISMATCHES = frozenset(
    {
        "load_peak_hour_periods",  # declared array.time, default is dict
        "battery_discharge_power_max",
        "battery_charge_power_max",
        "battery_discharge_efficiency",
        "battery_charge_efficiency",
        "battery_nominal_energy_capacity",
        "battery_minimum_state_of_charge",
        "battery_maximum_state_of_charge",
        "battery_target_state_of_charge",
        "battery_stress_cost",
        "weight_battery_discharge",
        "weight_battery_charge",
        "battery_soc_deficit_threshold",
        "battery_soc_deficit_cost",
        "battery_soc_surplus_threshold",
        "battery_soc_surplus_cost",
        # #1042: declared array.string because they accept per-battery entity-id
        # lists at number_of_batteries > 1, but their defaults stay bare scalar
        # strings on purpose - a single sensor name is not broadcast to every
        # battery (unlike the numeric params above), so the scalar default is
        # only ever valid at N=1 and keeps existing single-battery configs
        # byte-identical.
        "sensor_power_battery",
        "sensor_battery_state_of_charge",
        # #540 Part B: declared array.* because they accept a list of K
        # independent capacity/demand components, but their defaults stay bare
        # scalars on purpose - a scalar is the K=1 capacity charge and keeps
        # existing single-component configs byte-identical (same pattern as the
        # battery_* block above).
        "capacity_cost_per_kw",
        "capacity_charge_interval_timesteps",
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


# ---------------------------------------------------------------------------
# #880 / #904 regression tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def js_src() -> str:
    return Path("src/emhass/static/configuration_script.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def _build_param_fns(js_src: str) -> tuple[str, str]:
    """Extracted checkConfigParam + buildParamElement sources, shared across Node replay tests."""
    return (
        _extract_function_src(js_src, "checkConfigParam"),
        _extract_function_src(js_src, "buildParamElement"),
    )


def _extract_function_src(js_src: str, fn_name: str) -> str:
    """Extract a complete top-level ``[async] function <fn_name>(...) { ... }`` block."""
    fn_m = re.search(rf"(?:async\s+)?function\s+{re.escape(fn_name)}\s*\([^)]*\)\s*\{{", js_src)
    if not fn_m:
        raise AssertionError(f"function {fn_name!r} not found in JS source")
    brace_open = js_src.index("{", fn_m.start())
    depth = 1
    i = brace_open + 1
    while i < len(js_src) and depth > 0:
        if js_src[i] == "{":
            depth += 1
        elif js_src[i] == "}":
            depth -= 1
        i += 1
    return js_src[fn_m.start() : i]


def _run_node(script: str) -> subprocess.CompletedProcess:
    """Write *script* to a temp file and run it with Node.js."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False, encoding="utf-8") as f:
        f.write(script)
        tmp = f.name
    try:
        return subprocess.run([_NODE, tmp], capture_output=True, text=True, timeout=_NODE_TIMEOUT_S)
    finally:
        os.unlink(tmp)


def test_minus_elements_no_undefined_variable(js_src):
    """minusElements must not reference ``parameter_definition_name``.

    On master this FAILS: the variable appears at two console.log paths
    (lines 556 and 563), causing an uncaught ReferenceError that aborts
    window.onload and leaves the whole config UI dead (#880).
    After fix: PASSES — renamed to ``param`` everywhere in the function.
    """
    fn_src = _extract_function_src(js_src, "minusElements")
    assert "parameter_definition_name" not in fn_src, (
        "minusElements references `parameter_definition_name` (undefined in its scope). "
        "Rename to `param` (the function argument) at lines 556 and 563."
    )


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_minus_elements_does_not_crash_on_zero_inputs(js_src):
    """Node replay: minusElements must return cleanly when a param has no inputs.

    ThomasCZ's crash path (issue #880): heat_topology={"sources":[]} renders to
    "</br>" with no <input> elements; minusElements then references the undefined
    ``parameter_definition_name`` → ReferenceError → window.onload rejects.

    On master: exits 1 (ReferenceError).
    After fix (rename + early return): exits 0.
    """
    fn_src = _extract_function_src(js_src, "minusElements")

    node_script = (
        "const document = {\n"
        "  getElementById: (id) => {\n"
        "    if (id === 'heat_topology') {\n"
        "      return { getElementsByTagName: () => ({ length: 0 }) };\n"
        "    }\n"
        "    return null;\n"
        "  }\n"
        "};\n\n" + fn_src + "\n\ntry {\n"
        "  const r = minusElements('heat_topology');\n"
        "  process.stdout.write('OK result=' + r + '\\n');\n"
        "  process.exit(0);\n"
        "} catch (e) {\n"
        "  process.stderr.write('FAIL: ' + e.toString() + '\\n');\n"
        "  process.exit(1);\n"
        "}\n"
    )

    proc = _run_node(node_script)
    assert proc.returncode == 0, (
        f"minusElements threw on zero-input param (expected clean return):\n{proc.stderr.strip()}"
    )


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_build_param_element_object_type_renders_input(_build_param_fns):
    """Node replay: buildParamElement for 'object' type must return HTML with <input>.

    ThomasCZ's config has heat_topology={"sources":[]}.  On master the
    nested-object render path produces the string "</br>" (no <input>), so
    minusElements subsequently finds length==0 and crashes (#880).

    On master: exits 1.
    After fix (render as JSON text box): exits 0.
    """
    check_fn, build_fn = _build_param_fns

    node_script = f"""\
{check_fn}
{build_fn}

const paramDef = {{
  input: "object",
  default_value: null,
  friendly_name: "Heat Topology",
  Description: "test"
}};
const config = {{ heat_topology: {{ sources: [] }} }};
const html = buildParamElement(paramDef, "heat_topology", config);

if (!html.includes('<input')) {{
  process.stderr.write('FAIL: no <input> in rendered HTML: ' + JSON.stringify(html) + '\\n');
  process.exit(1);
}}
process.exit(0);
"""
    proc = _run_node(node_script)
    assert proc.returncode == 0, (
        f"buildParamElement produced no <input> for object type with value={{sources:[]}}:\n"
        f"{proc.stderr.strip()}"
    )


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_build_param_element_object_absent_config_no_null_string_value(_build_param_fns):
    """Node replay: object-type param absent from config must not render value="null".

    On master: placeholder=JSON.stringify(null)="null" → rendered as
    <input value=null>.  Saving that back stores the string "null" in
    config.json, causing a per-request backend warning (#904).

    On master: exits 1.
    After fix (placeholder="" for null defaults): exits 0.
    """
    check_fn, build_fn = _build_param_fns

    node_script = f"""\
{check_fn}
{build_fn}

const paramDef = {{
  input: "object",
  default_value: null,
  friendly_name: "Heat Topology",
  Description: "test"
}};
const config = {{}};  // heat_topology absent
const html = buildParamElement(paramDef, "heat_topology", config);

if (html.includes('value=null') || html.includes('value="null"')) {{
  process.stderr.write('FAIL: rendered string null as value attribute: ' + JSON.stringify(html) + '\\n');
  process.exit(1);
}}
process.exit(0);
"""
    proc = _run_node(node_script)
    assert proc.returncode == 0, (
        f"buildParamElement renders string 'null' as input value for absent config:\n"
        f"{proc.stderr.strip()}"
    )


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_build_param_element_object_apostrophe_value_escaping(_build_param_fns):
    """Node replay: object-type render must escape double quotes so apostrophes in
    string values don't corrupt the HTML attribute.

    Single-quote attribute wrapping (value='...') breaks when the JSON contains
    an apostrophe, e.g. {"label":"John's panel"} → value='{"label":"John's panel"}'
    terminates the attribute at the apostrophe.

    On current (broken) code: exits 1 — no &quot; escape, single-quote wrap.
    After fix (double-quote wrap + replaceAll): exits 0.
    """
    check_fn, build_fn = _build_param_fns

    node_script = f"""\
{check_fn}
{build_fn}

const paramDef = {{
  input: "object",
  default_value: null,
  friendly_name: "Heat Topology",
  Description: "test"
}};
const config = {{ heat_topology: {{ label: "John's panel" }} }};
const html = buildParamElement(paramDef, "heat_topology", config);

// Must use &quot; escaping so double quotes inside the JSON are safe
if (!html.includes('&quot;')) {{
  process.stderr.write('FAIL: no &quot; escaping in rendered HTML: ' + JSON.stringify(html) + '\\n');
  process.exit(1);
}}
// Must NOT use single-quote attribute wrapping (apostrophes break it)
if (html.includes("value='")) {{
  process.stderr.write('FAIL: single-quote attribute wrapping detected: ' + JSON.stringify(html) + '\\n');
  process.exit(1);
}}
// Round-trip: decode &quot; → " then JSON.parse must recover the original object
const m = html.match(/value="([^"]*)"/);
if (!m) {{
  process.stderr.write('FAIL: no double-quoted value= attribute in: ' + JSON.stringify(html) + '\\n');
  process.exit(1);
}}
const decoded = m[1].replaceAll('&quot;', '"');
let parsed;
try {{ parsed = JSON.parse(decoded); }} catch (e) {{
  process.stderr.write('FAIL: value attr does not round-trip via JSON.parse: ' + e + '\\n');
  process.exit(1);
}}
if (parsed.label !== "John's panel") {{
  process.stderr.write('FAIL: round-trip value mismatch: ' + JSON.stringify(parsed) + '\\n');
  process.exit(1);
}}
process.exit(0);
"""
    proc = _run_node(node_script)
    assert proc.returncode == 0, (
        f"buildParamElement broke on apostrophe in object value:\n{proc.stderr.strip()}"
    )


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_save_configuration_object_type_null_representations(js_src):
    """Node replay: saveConfiguration must store JSON null for '' and 'null' inputs.

    When heat_topology is absent from config, buildParamElement renders an empty
    text box (value="").  If saveConfiguration stores that as "" or as "null"
    (the string), the backend receives a non-null value and warns on every request.

    On original master (no object-type branch): exits 1 — param falls through to
    the generic path and config.heat_topology becomes "".
    After fix: exits 0 for both "" and "null" inputs.
    """
    save_fn_src = _extract_function_src(js_src, "saveConfiguration")

    # document must be a script-level var so saveConfiguration (outer scope) can see it.
    # We reassign it (without var/let) for each test case inside the IIFE.
    node_script = (
        "var capturedBody = null;\n"
        "var document = null;\n"
        "var fetch = async function(url, opts) {\n"
        "  capturedBody = opts.body;\n"
        "  return { status: 200, json: async function() { return {}; } };\n"
        "};\n"
        "function showChangeStatus() {}\n"
        "function errorAlert(msg) { throw new Error('errorAlert: ' + msg); }\n\n"
        + save_fn_src
        + "\n\n"
        "(async () => {\n"
        "  const paramDefs = {\n"
        "    General: {\n"
        "      heat_topology: { input: 'object', default_value: null,\n"
        "                       friendly_name: 'Heat Topology', Description: 'test' }\n"
        "    }\n"
        "  };\n"
        "  for (const [label, val] of [['empty string', ''], ['string null', 'null']]) {\n"
        "    capturedBody = null;\n"
        "    document = {\n"
        "      getElementsByClassName: function(cls) {\n"
        "        return cls === 'section-card' ? { length: 1 } : { length: 0 };\n"
        "      },\n"
        "      getElementById: function(id) {\n"
        "        if (id === 'config-box') return null;\n"
        "        if (id === 'heat_topology') return {\n"
        "          tagName: 'DIV',\n"
        "          getElementsByClassName: function(cls) {\n"
        "            return cls === 'param_input'\n"
        "              ? [{ type: 'text', value: val }]\n"
        "              : [];\n"
        "          }\n"
        "        };\n"
        "        return null;\n"
        "      }\n"
        "    };\n"
        "    await saveConfiguration(paramDefs);\n"
        "    const saved = JSON.parse(capturedBody);\n"
        "    if (saved.heat_topology !== null) {\n"
        "      process.stderr.write('FAIL (' + label + '): heat_topology should be null, got: '\n"
        "        + JSON.stringify(saved.heat_topology) + '\\n');\n"
        "      process.exit(1);\n"
        "    }\n"
        "  }\n"
        "  process.exit(0);\n"
        "})().catch(e => { process.stderr.write('FAIL: ' + e + '\\n'); process.exit(1); });\n"
    )

    proc = _run_node(node_script)
    assert proc.returncode == 0, (
        f"saveConfiguration did not store null for empty/null object input:\n{proc.stderr.strip()}"
    )


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_build_param_element_object_html_special_chars_escaped(_build_param_fns):
    """Node replay: object-type render must escape &, <, > in addition to double quotes.

    JSON values can contain arbitrary strings, e.g. {"label":"<b>bold</b> & more"}.
    Unescaped & is interpreted as an HTML entity reference; < and > are malformed
    in attribute context.  All four must be escaped: &→&amp; <→&lt; >→&gt; "→&quot;.

    On current code: exits 1 — only " is escaped, &/</> are not.
    After fix: exits 0.
    """
    check_fn, build_fn = _build_param_fns

    node_script = f"""\
{check_fn}
{build_fn}

const paramDef = {{
  input: "object",
  default_value: null,
  friendly_name: "Heat Topology",
  Description: "test"
}};
const config = {{ heat_topology: {{ label: "<b>bold</b> & 'text'" }} }};
const html = buildParamElement(paramDef, "heat_topology", config);

const checks = [
  ['&amp;',  html.includes('&amp;'),  'unescaped & in attribute'],
  ['&lt;',   html.includes('&lt;'),   'unescaped < in attribute'],
  ['&gt;',   html.includes('&gt;'),   'unescaped > in attribute'],
];
for (const [entity, found, msg] of checks) {{
  if (!found) {{
    process.stderr.write('FAIL: ' + entity + ' missing — ' + msg + ': ' + JSON.stringify(html) + '\\n');
    process.exit(1);
  }}
}}
// Round-trip: HTML-decode then JSON.parse must recover original object
const m = html.match(/value="([^"]*)"/);
if (!m) {{
  process.stderr.write('FAIL: no double-quoted value= attribute in: ' + JSON.stringify(html) + '\\n');
  process.exit(1);
}}
const decoded = m[1]
  .replaceAll('&quot;', '"')
  .replaceAll('&amp;', '&')
  .replaceAll('&lt;', '<')
  .replaceAll('&gt;', '>');
let parsed;
try {{ parsed = JSON.parse(decoded); }} catch (e) {{
  process.stderr.write('FAIL: value attr does not round-trip via JSON.parse: ' + e + '\\n');
  process.exit(1);
}}
if (parsed.label !== "<b>bold</b> & 'text'") {{
  process.stderr.write('FAIL: round-trip mismatch: ' + JSON.stringify(parsed) + '\\n');
  process.exit(1);
}}
process.exit(0);
"""
    proc = _run_node(node_script)
    assert proc.returncode == 0, (
        f"buildParamElement did not escape HTML special chars in object value:\n{proc.stderr.strip()}"
    )


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_save_configuration_object_type_invalid_json_shows_error(js_src):
    """Node replay: saveConfiguration must call errorAlert and not fetch when given invalid JSON.

    On current code: catch block stores raw string silently (fetch IS called,
    config gets garbage).
    After fix (errorAlert + return 0): fetch is NOT called, errorAlert is called.
    """
    save_fn_src = _extract_function_src(js_src, "saveConfiguration")

    node_script = (
        "var capturedBody = null;\n"
        "var errorAlertMsg = null;\n"
        "var document = null;\n"
        "var fetch = async function(url, opts) {\n"
        "  capturedBody = opts.body;\n"
        "  return { status: 200, json: async function() { return {}; } };\n"
        "};\n"
        "function showChangeStatus() {}\n"
        "function errorAlert(msg) { errorAlertMsg = msg; }\n\n" + save_fn_src + "\n\n"
        "(async () => {\n"
        "  const paramDefs = {\n"
        "    General: {\n"
        "      heat_topology: { input: 'object', default_value: null,\n"
        "                       friendly_name: 'Heat Topology', Description: 'test' }\n"
        "    }\n"
        "  };\n"
        "  document = {\n"
        "    getElementsByClassName: function(cls) {\n"
        "      return cls === 'section-card' ? { length: 1 } : { length: 0 };\n"
        "    },\n"
        "    getElementById: function(id) {\n"
        "      if (id === 'config-box') return null;\n"
        "      if (id === 'heat_topology') return {\n"
        "        tagName: 'DIV',\n"
        "        getElementsByClassName: function(cls) {\n"
        "          return cls === 'param_input'\n"
        "            ? [{ type: 'text', value: '{not valid json' }]\n"
        "            : [];\n"
        "        }\n"
        "      };\n"
        "      return null;\n"
        "    }\n"
        "  };\n"
        "  await saveConfiguration(paramDefs);\n"
        "  if (capturedBody !== null) {\n"
        "    process.stderr.write('FAIL: fetch was called despite invalid JSON; config stored: '\n"
        "      + capturedBody + '\\n');\n"
        "    process.exit(1);\n"
        "  }\n"
        "  if (!errorAlertMsg) {\n"
        "    process.stderr.write('FAIL: errorAlert was not called for invalid JSON\\n');\n"
        "    process.exit(1);\n"
        "  }\n"
        "  process.exit(0);\n"
        "})().catch(e => { process.stderr.write('FAIL: ' + e + '\\n'); process.exit(1); });\n"
    )

    proc = _run_node(node_script)
    assert proc.returncode == 0, (
        f"saveConfiguration silently stored invalid JSON instead of showing error:\n{proc.stderr.strip()}"
    )


# ---------------------------------------------------------------------------
# Generic config-form array (+/-) list controls.
#
# Three released-v0.18.2 defects on this generic path only surfaced once #540
# Part B routed capacity_cost_per_kw / capacity_charge_interval_timesteps
# through it (before, load_peak_hour_periods was the only array.* param outside
# the length-managed Deferrable Loads / Battery sections):
#   1. buildParamContainers excluded the whole "Battery" section, not just the
#      15 per-battery BATTERY_ARRAY_PARAMS.
#   2. the + click listener referenced `param_definitions`, undefined in
#      buildParamContainers scope -> ReferenceError on every click.
#   3. plusElements grew the list with `innerHTML +=`, re-serialising sibling
#      <input>s from their value *attribute* and dropping values the user had
#      typed (a live .value) but not saved.
#
# One shared Node harness (_run_array_control_probe) drives the real functions
# against a compact fake DOM and reports the observable behaviour; the tests
# below assert on that. Each fails on the released implementation.
# ---------------------------------------------------------------------------

_CAPACITY_ARRAY_PARAMS = ("capacity_cost_per_kw", "capacity_charge_interval_timesteps")

# Fake DOM + probe driver. Placeholders: __FNS__ (extracted JS), __SPEC__ (JSON:
# {section, params:{name:input}, click:name|null, typed:str|null}). Prints one
# JSON line: {buttons:{name:bool}, before, after, first_value, error}.
_ARRAY_CONTROL_HARNESS = r"""
__FNS__
const SPEC = __SPEC__;

function makeInput(v) {
  const el = { className: "param_input", _attr: String(v), value: String(v),
    parentNode: { tagName: "DIV" } };
  el.getAttribute = (k) => (k === "value" ? el._attr : null);
  el.setAttribute = (k, val) => { if (k === "value") el._attr = String(val); };
  el.remove = () => {};
  Object.defineProperty(el, "outerHTML", {
    get: () => '<input class="param_input" value=' + el._attr + ">",
  });
  return el;
}
// param-input container: an <input> list that honours BOTH mutation paths the
// production code might take. `innerHTML +=` reparses from the attribute only
// (drops typed .value); insertAdjacentHTML appends without touching siblings.
function makeInputContainer() {
  const c = { _inputs: [] };
  c.getElementsByTagName = (t) => (t === "input" ? c._inputs : []);
  Object.defineProperty(c, "innerHTML", {
    get: () => c._inputs.map((i) => i.outerHTML).join(""),
    set: (v) => {
      c._inputs = (v.match(/value=([^\s>]+)/g) || []).map((m) => makeInput(m.slice(6)));
    },
  });
  c.insertAdjacentHTML = (pos, html) => {
    const m = html.match(/value=([^\s>]+)/);
    if (pos === "beforeend") c._inputs.push(makeInput(m ? m[1] : ""));
  };
  return c;
}

const paramDivs = {};
let buttons = [];
const body = { _html: "" };
Object.defineProperty(body, "innerHTML", {
  get: () => body._html,
  set: (v) => {
    body._html = v;
    for (const [, id] of v.matchAll(/<div class="param" id="([^"]+)">/g)) {
      if (paramDivs[id]) continue;
      const c = makeInputContainer();
      c._inputs.push(makeInput(0)); // the row buildParamElement emitted
      paramDivs[id] = {
        id,
        _c: c,
        getElementsByClassName: (k) => (k === "param-input" ? [c] : []),
        getElementsByTagName: (t) => c.getElementsByTagName(t),
      };
    }
    buttons = [...v.matchAll(/class="input-(plus|minus) ([^"]+)"/g)].map(([, kind, name]) => ({
      classList: ["input-" + kind, name],
      addEventListener(ev, fn) { if (ev === "click") this._h = fn; },
    }));
  },
});
const container = {
  getElementsByClassName: (k) => (k === "section-body" ? [body] : []),
  querySelectorAll: (sel) => {
    if (sel === ".input-plus") return buttons.filter((b) => b.classList[0] === "input-plus");
    if (sel === ".input-minus") return buttons.filter((b) => b.classList[0] === "input-minus");
    return [];
  },
};
var document = {
  getElementById: (id) => (id === SPEC.section ? container : paramDivs[id] || null),
};

const defs = {};
for (const [name, input] of Object.entries(SPEC.params)) {
  defs[name] = { input, default_value: input.endsWith("int") ? 1 : 0, friendly_name: name, Description: "d" };
}
buildParamContainers(SPEC.section, defs, {}, []);

const out = { buttons: {}, before: null, after: null, first_value: null, error: null };
for (const name of Object.keys(SPEC.params)) {
  out.buttons[name] =
    body._html.includes('class="input-plus ' + name + '"') &&
    body._html.includes('class="input-minus ' + name + '"');
}
if (SPEC.click) {
  const div = paramDivs[SPEC.click];
  if (SPEC.typed != null) div._c._inputs[0].value = SPEC.typed;
  out.before = div._c._inputs.length;
  const btn = buttons.find((b) => b.classList[0] === "input-plus" && b.classList[1] === SPEC.click);
  try {
    btn._h();
    out.after = div._c._inputs.length;
    out.first_value = div._c._inputs[0].value;
  } catch (e) {
    out.error = String(e);
  }
}
process.stdout.write(JSON.stringify(out));
"""


def _run_array_control_probe(js_src, section, params, *, click=None, typed=None):
    """Render *section* through the real buildParamContainers/plusElements against
    a compact fake DOM; optionally type *typed* into *click*'s first row and fire
    its ``+``. Returns the probe's JSON dict."""
    fns = "\n".join(
        _extract_function_src(js_src, name)
        for name in (
            "checkConfigParam",
            "buildParamElement",
            "plusElements",
            "minusElements",
            "buildParamContainers",
        )
    )
    m = re.search(r"const\s+BATTERY_ARRAY_PARAMS\s*=\s*\[.*?\]\s*;", js_src, re.DOTALL)
    spec = {"section": section, "params": params, "click": click, "typed": typed}
    script = _ARRAY_CONTROL_HARNESS.replace("__FNS__", m.group(0) + "\n" + fns).replace(
        "__SPEC__", json.dumps(spec)
    )
    proc = _run_node(script)
    assert proc.returncode == 0, f"probe harness crashed:\n{proc.stderr.strip()}"
    return json.loads(proc.stdout)


def test_array_control_buttons_not_on_managed_battery_params(js_src):
    """Cheap (no-Node) guard for defect 1's invariant: the capacity params must
    stay out of BATTERY_ARRAY_PARAMS so they fall through to the generic +/-."""
    m = re.search(r"const\s+BATTERY_ARRAY_PARAMS\s*=\s*\[(.*?)\]\s*;", js_src, re.DOTALL)
    managed = re.findall(r'"([^"]+)"', m.group(1))
    for name in _CAPACITY_ARRAY_PARAMS:
        assert name not in managed, f"{name} in BATTERY_ARRAY_PARAMS -> would lose its generic +/-"


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_array_control_button_eligibility(js_src):
    """A: generic +/- present for the capacity params, absent for a per-battery
    array and for a Deferrable Loads managed array."""
    battery = _run_array_control_probe(
        js_src,
        "Battery",
        {
            "capacity_cost_per_kw": "array.float",
            "capacity_charge_interval_timesteps": "array.int",
            "weight_battery_discharge": "array.float",
        },
    )["buttons"]
    deferrable = _run_array_control_probe(
        js_src, "Deferrable Loads", {"nominal_power_of_deferrable_loads": "array.float"}
    )["buttons"]
    assert battery["capacity_cost_per_kw"], "capacity_cost_per_kw lost its generic +/-"
    assert battery["capacity_charge_interval_timesteps"], "interval param lost its generic +/-"
    assert not battery["weight_battery_discharge"], "a per-battery array got a free +/-"
    assert not deferrable["nominal_power_of_deferrable_loads"], (
        "a managed def-load array got a free +/-"
    )


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_array_control_plus_click_adds_row(js_src):
    """B: clicking + adds one row and raises nothing (guards the out-of-scope
    ``param_definitions`` ReferenceError in the click listener)."""
    r = _run_array_control_probe(
        js_src, "Battery", {"capacity_cost_per_kw": "array.float"}, click="capacity_cost_per_kw"
    )
    assert r["error"] is None, f"+ click raised: {r['error']}"
    assert (r["before"], r["after"]) == (1, 2), (
        f"row count {r['before']} -> {r['after']}, expected 1 -> 2"
    )


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_array_control_plus_click_preserves_typed_value(js_src):
    """C: a value typed but not saved in an existing row survives adding a row
    (guards the ``innerHTML +=`` re-serialisation that reset it to the default)."""
    r = _run_array_control_probe(
        js_src,
        "Battery",
        {"capacity_cost_per_kw": "array.float"},
        click="capacity_cost_per_kw",
        typed="3",
    )
    assert r["error"] is None, f"+ click raised: {r['error']}"
    assert r["after"] == 2, f"expected 2 rows, got {r['after']}"
    assert r["first_value"] == "3", f"first row reset to {r['first_value']!r} (should stay '3')"
