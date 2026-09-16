import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

_NODE = shutil.which("node")
_NODE_TIMEOUT_S = 120

_JS_PATH = Path("src/emhass/static/configuration_script.js")
_DEFS_PATH = Path("src/emhass/static/data/param_definitions.json")
_HTML_PATH = Path("src/emhass/static/configuration_list.html")


def _run_node(script: str) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".js", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(script)
        temp_path = handle.name
    try:
        return subprocess.run(
            [_NODE, temp_path],
            capture_output=True,
            text=True,
            timeout=_NODE_TIMEOUT_S,
        )
    finally:
        Path(temp_path).unlink(missing_ok=True)


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


def _vm_prelude() -> str:
    """Load the full script (constants + all functions) into a vm context named ``ctx``."""
    return f"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const source = fs.readFileSync({json.dumps(str(_JS_PATH))}, "utf8");
const defs = JSON.parse(fs.readFileSync({json.dumps(str(_DEFS_PATH))}, "utf8"));
const ctx = {{ window: {{}}, document: {{}}, console }};
vm.createContext(ctx);
vm.runInContext(source, ctx, {{ filename: "configuration_script.js" }});

const expectedPerLoad = Object.entries(defs["Deferrable Loads"])
  .filter(([, definition]) => definition.input.startsWith("array."))
  .map(([name]) => name)
  .sort();
"""


# ---------------------------------------------------------------------------
# 1 & 2: template minimums (pure static check, no Node needed)
# ---------------------------------------------------------------------------


def test_deferrable_count_template_minimum_is_zero():
    """number_of_deferrable_loads must allow 0 (EMHASS accepts num_def_loads=0,
    see upstream fea14881 'Added test to check when number of deferrable loads is 0')."""
    html = _HTML_PATH.read_text(encoding="utf-8")
    m = re.search(r'<input id="number_of_deferrable_loads"[^>]*\bmin="(\d+)"', html)
    assert m, "number_of_deferrable_loads input (with a min attribute) not found"
    assert m.group(1) == "0", f"number_of_deferrable_loads min is {m.group(1)!r}, expected '0'"


def test_battery_count_template_minimum_remains_one():
    """number_of_batteries must keep its min=1 floor - batteries are not part of
    this remediation and 0 batteries is not a supported EMHASS configuration."""
    html = _HTML_PATH.read_text(encoding="utf-8")
    m = re.search(r'<input id="number_of_batteries"[^>]*\bmin="(\d+)"', html)
    assert m, "number_of_batteries input (with a min attribute) not found"
    assert m.group(1) == "1", f"number_of_batteries min is {m.group(1)!r}, expected '1' (unchanged)"


# ---------------------------------------------------------------------------
# 3: DEFERRABLE_ARRAY_PARAMS matches schema, excludes heat_topology
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_deferrable_array_params_matches_schema_and_excludes_heat_topology():
    script = (
        _vm_prelude()
        + """
const allowList = Array.from(
  vm.runInContext("DEFERRABLE_ARRAY_PARAMS", ctx)
).sort();

assert.deepEqual(
  allowList,
  expectedPerLoad,
  "DEFERRABLE_ARRAY_PARAMS must match the section's per-load array fields"
);
assert.ok(!allowList.includes("heat_topology"));
"""
    )
    proc = _run_node(script)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# 4 & 5: headerElement grow/shrink touches only managed fields; shrink to 0
# requests the zero-minimum removal.
# ---------------------------------------------------------------------------


def _header_probe_script(target_value: str) -> str:
    return (
        _vm_prelude()
        + f"""
const inputCounts = Object.fromEntries(expectedPerLoad.map((name) => [name, 2]));
const params = expectedPerLoad.map((id) => ({{
  id,
  querySelectorAll: () => Array(inputCounts[id]).fill({{}}),
}}));
params.push({{ id: "heat_topology", querySelectorAll: () => [{{}}] }});

const sectionBody = {{
  getElementsByClassName: (name) => name === "param" ? params : [],
}};
const sectionCard = {{
  getElementsByClassName: (name) => name === "section-body" ? [sectionBody] : [],
}};
const header = {{
  id: "number_of_deferrable_loads",
  value: {json.dumps(target_value)},
  closest: () => sectionCard,
}};

let calls = [];
ctx.plusElements = (name) => calls.push(["plus", name, undefined]);
ctx.minusElements = (name, minimum_inputs) => calls.push(["minus", name, minimum_inputs]);

ctx.headerElement(header, defs, {{}});

const touched = Array.from(new Set(calls.map((c) => c[1]))).sort();
assert.deepEqual(
  touched,
  expectedPerLoad,
  "header count change must touch only the managed per-load fields, never heat_topology"
);
assert.ok(!touched.includes("heat_topology"));

process.stdout.write(JSON.stringify(calls));
"""
    )


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_growing_load_count_touches_only_managed_fields():
    proc = _run_node(_header_probe_script("3"))
    assert proc.returncode == 0, proc.stderr
    calls = json.loads(proc.stdout)
    assert all(kind == "plus" for kind, _name, _min in calls)


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_shrinking_to_zero_touches_only_managed_fields_with_zero_minimum():
    """Starting from 2 rows, dropping to 0 must call minusElements(name, 0) for
    every managed per-load field (not the default minimum-1 floor)."""
    proc = _run_node(_header_probe_script("0"))
    assert proc.returncode == 0, proc.stderr
    calls = json.loads(proc.stdout)
    assert all(kind == "minus" for kind, _name, _min in calls)
    assert all(minimum == 0 for _kind, _name, minimum in calls), (
        f"shrink-to-zero must pass minimum_inputs=0 to minusElements, got: {calls}"
    )


# ---------------------------------------------------------------------------
# 6 & 7: minusElements' own minimum-inputs contract, exercised directly (no
# headerElement involved) - the generic +/- and Battery paths rely on this
# still defaulting to 1.
# ---------------------------------------------------------------------------


def _minus_elements_probe(minimum_arg: str) -> str:
    call = (
        "ctx.minusElements('p')" if minimum_arg == "" else f"ctx.minusElements('p', {minimum_arg})"
    )
    return f"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync({json.dumps(str(_JS_PATH))}, "utf8");
const ctx = {{ window: {{}}, document: {{}}, console }};
vm.createContext(ctx);
vm.runInContext(source, ctx, {{ filename: "configuration_script.js" }});

const inputs = [];
function makeInput() {{
  const obj = {{ parentNode: {{ tagName: "DIV" }} }};
  obj.remove = () => {{
    const idx = inputs.indexOf(obj);
    if (idx > -1) inputs.splice(idx, 1);
  }};
  return obj;
}}
inputs.push(makeInput());

ctx.document = {{
  getElementById: (id) => id === "p" ? {{ getElementsByTagName: (t) => t === "input" ? inputs : [] }} : null,
}};

{call};
process.stdout.write(String(inputs.length));
"""


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_generic_minus_elements_default_keeps_last_row():
    """minusElements(param) with no second argument must preserve the existing
    generic/Battery minimum-one floor exactly as before."""
    proc = _run_node(_minus_elements_probe(""))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "1", "default call removed the last remaining row"


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_minus_elements_explicit_zero_removes_last_row():
    """minusElements(param, 0) must be able to remove the final row - this is
    what lets number_of_deferrable_loads reach 0."""
    proc = _run_node(_minus_elements_probe("0"))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "0", "explicit minimum 0 did not remove the last row"


# ---------------------------------------------------------------------------
# 8 & 9: zero-row serialization is owned only by the real Deferrable Loads
# section; unrelated zero-input arrays retain their pre-existing omission.
# ---------------------------------------------------------------------------


def _save_zero_input_array_probe(section: str) -> subprocess.CompletedProcess:
    js_src = _JS_PATH.read_text(encoding="utf-8")
    save_fn_src = _extract_function_src(js_src, "saveConfiguration")
    allow_list_src = re.search(
        r"const\s+DEFERRABLE_ARRAY_PARAMS\s*=\s*\[.*?\]\s*;", js_src, re.DOTALL
    ).group(0)
    node_script = (
        "var capturedBody = null;\n"
        "var document = null;\n"
        "var fetch = async function(url, opts) {\n"
        "  capturedBody = opts.body;\n"
        "  return { status: 200, json: async function() { return {}; } };\n"
        "};\n"
        "function showChangeStatus() {}\n"
        "function errorAlert(msg) { throw new Error('errorAlert: ' + msg); }\n\n"
        + allow_list_src
        + "\n\n"
        + save_fn_src
        + "\n\n"
        "(async () => {\n"
        "  const name = 'nominal_power_of_deferrable_loads';\n"
        f"  const paramDefs = {{ {json.dumps(section)}: {{\n"
        "    [name]: { input: 'array.float', default_value: 3000.0,\n"
        "              friendly_name: 'x', Description: 'd' }\n"
        "  } };\n"
        "  document = {\n"
        "    getElementsByClassName: (cls) => cls === 'section-card' ? { length: 1 } : { length: 0 },\n"
        "    getElementById: (id) => id === name ? {\n"
        "      tagName: 'DIV',\n"
        "      getElementsByClassName: (cls) => cls === 'param_input' ? [] : [],\n"
        "    } : null,\n"
        "  };\n"
        "  await saveConfiguration(paramDefs);\n"
        "  process.stdout.write(capturedBody);\n"
        "})().catch(e => { process.stderr.write('FAIL: ' + e + '\\n'); process.exit(1); });\n"
    )
    return _run_node(node_script)


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_save_configuration_zero_input_array_serializes_as_empty_list():
    proc = _save_zero_input_array_probe("Deferrable Loads")
    assert proc.returncode == 0, proc.stderr
    saved = json.loads(proc.stdout)
    assert saved["nominal_power_of_deferrable_loads"] == []


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_zero_input_array_outside_deferrable_section_remains_omitted():
    """Rebase guard for #1118: #1116 zero-row handling must not globally
    rewrite unrelated zero-input array parameters as []."""
    proc = _save_zero_input_array_probe("Other")
    assert proc.returncode == 0, proc.stderr
    saved = json.loads(proc.stdout)
    assert "nominal_power_of_deferrable_loads" not in saved


# ---------------------------------------------------------------------------
# 10, 11, 12: dynamic boolean row defaults + saved-value overrides
# ---------------------------------------------------------------------------


def _capture_dynamic_row_script(name: str) -> str:
    return (
        _vm_prelude()
        + f"""
let appended = "";
ctx.document = {{
  getElementById: (id) => id === {json.dumps(name)} ? {{
    getElementsByClassName: (className) =>
      className === "param-input" ? [{{
        insertAdjacentHTML: (_position, html) => {{ appended = html; }},
      }}] : [],
  }} : null,
}};
ctx.plusElements({json.dumps(name)}, defs, "Deferrable Loads", {{}});
process.stdout.write(appended);
"""
    )


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_dynamic_row_true_default_renders_checked():
    proc = _run_node(_capture_dynamic_row_script("is_electric_load"))
    assert proc.returncode == 0, proc.stderr
    assert "value=true" in proc.stdout
    assert re.search(r"\bchecked\b", proc.stdout)


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_dynamic_row_false_default_renders_unchecked():
    proc = _run_node(_capture_dynamic_row_script("set_deferrable_load_single_constant"))
    assert proc.returncode == 0, proc.stderr
    assert "value=false" in proc.stdout
    assert not re.search(r"\bchecked\b", proc.stdout)


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_saved_boolean_values_override_defaults():
    script = (
        _vm_prelude()
        + """
const savedFalse = ctx.buildParamElement(
  defs["Deferrable Loads"].is_electric_load,
  "is_electric_load",
  { is_electric_load: [false] }
);
assert.doesNotMatch(savedFalse, /\\bchecked\\b/);

const savedTrue = ctx.buildParamElement(
  defs["Deferrable Loads"].is_electric_load,
  "is_electric_load",
  { is_electric_load: [true] }
);
assert.match(savedTrue, /\\bchecked\\b/);
"""
    )
    proc = _run_node(script)
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# 13: heat_topology always remains exactly one input across count changes,
# including down to 0.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_NODE is None, reason="node not in PATH")
def test_heat_topology_untouched_across_grow_and_shrink_to_zero():
    """plusElements/minusElements are mocked to genuinely mutate simulated row
    counts (not just record which names were called), so this test detects the
    original defect: if headerElement() ever routes heat_topology through the
    per-load grow/shrink path, heatTopologyInputCount will drift away from 1."""
    script = (
        _vm_prelude()
        + """
const inputCounts = Object.fromEntries(expectedPerLoad.map((name) => [name, 2]));
let heatTopologyInputCount = 1;
const params = expectedPerLoad.map((id) => ({
  id,
  querySelectorAll: () => Array(inputCounts[id]).fill({}),
}));
params.push({ id: "heat_topology", querySelectorAll: () => Array(heatTopologyInputCount).fill({}) });

const sectionBody = {
  getElementsByClassName: (name) => name === "param" ? params : [],
  firstElementChild: params[0],
};
const sectionCard = { getElementsByClassName: (name) => name === "section-body" ? [sectionBody] : [] };
const header = { id: "number_of_deferrable_loads", value: "3", closest: () => sectionCard };

ctx.plusElements = (name) => {
  if (name === "heat_topology") {
    heatTopologyInputCount += 1;
  } else if (Object.prototype.hasOwnProperty.call(inputCounts, name)) {
    inputCounts[name] += 1;
  } else {
    throw new Error("plusElements received unexpected param name: " + name);
  }
};
ctx.minusElements = (name, minimum_inputs = 1) => {
  if (name === "heat_topology") {
    if (heatTopologyInputCount > minimum_inputs) heatTopologyInputCount -= 1;
  } else if (Object.prototype.hasOwnProperty.call(inputCounts, name)) {
    if (inputCounts[name] > minimum_inputs) inputCounts[name] -= 1;
  } else {
    throw new Error("minusElements received unexpected param name: " + name);
  }
};

ctx.headerElement(header, defs, {});
for (const name of expectedPerLoad) {
  assert.equal(inputCounts[name], 3, `${name} must have exactly 3 rows after growing to 3`);
}
assert.equal(heatTopologyInputCount, 1, "heat_topology must stay at 1 input after growing to 3");

header.value = "0";
ctx.headerElement(header, defs, {});
for (const name of expectedPerLoad) {
  assert.equal(inputCounts[name], 0, `${name} must have exactly 0 rows after shrinking to 0`);
}
assert.equal(heatTopologyInputCount, 1, "heat_topology must stay at 1 input after shrinking to 0");
"""
    )
    proc = _run_node(script)
    assert proc.returncode == 0, proc.stderr
