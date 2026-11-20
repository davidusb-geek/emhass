// Regression checks for the configuration UI's formerly dynamic HTML sinks.
// Run from repository root: node --test tests/ui/xss-rendering.test.cjs
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const source = fs.readFileSync("src/emhass/static/configuration_script.js", "utf8");
const malicious = '"><img src=x onerror=alert(1)>';

class FakeNode {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.dataset = {};
    this.attributes = {};
    this.className = "";
    this._type = "";
    this._value = "";
    this.textContent = "";
    this.style = {};
  }
  // Standards-aligned value sanitization (HTML Standard, input type=number): a value that is not a
  // valid floating-point number becomes "". Applied on assignment and again when the type changes.
  get type() { return this._type; }
  set type(type) { this._type = type; this._value = this._sanitize(this._value); }
  get value() { return this._value; }
  set value(value) { this._value = this._sanitize(String(value)); }
  _sanitize(value) {
    if (this._type !== "number") return value;
    return /^-?(\d+|\d*\.\d+)([eE][-+]?\d+)?$/.test(value) ? value : "";
  }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener() {}
  querySelectorAll() { return []; }
  querySelector() { return null; }
  set innerHTML(_value) {
    throw new Error("An HTML-parsing sink was used");
  }
}

function descendants(root) {
  const all = [];
  const walk = (node) => {
    if (!node || typeof node !== "object") return;
    all.push(node);
    for (const child of node.children || []) walk(child);
  };
  walk(root);
  return all;
}

function setup() {
  const overview = new FakeNode("div");
  const template = { content: { cloneNode: () => new FakeNode("fragment") } };
  const document = {
    createElement: (tag) => new FakeNode(tag),
    createTextNode: (text) => ({ text: String(text) }),
    getElementById: (id) => {
      if (id === "system-overview-badges") return overview;
      if (id === "configuration-list-template") return template;
      return null;
    },
    querySelectorAll: () => [],
  };
  const ctx = { document, window: {}, console };
  vm.createContext(ctx);
  vm.runInContext(source, ctx, { filename: "configuration_script.js" });
  return { ctx, overview, template };
}

test("thermal cards treat configuration values and extra keys as data, never markup", () => {
  const { ctx } = setup();
  const state = [
    {
      kind: "thermal_config",
      extra: { [malicious]: "kept" },
      thermal_config: {
        heating_rate: malicious,
        cooling_constant: 0.1,
        thermal_inertia: 0,
        start_temperature: 20,
        sense: "heat",
        min_temperatures: [malicious],
        max_temperatures: [24],
      },
    },
    {
      kind: "thermal_battery",
      extra: {},
      thermal_battery: {
        supply_temperature: malicious,
        min_temperatures: [21],
        max_temperatures: [26],
      },
    },
    { kind: "passthrough", extra: { [malicious]: true } },
  ];
  ctx.state = state;
  ctx.builder = new FakeNode("div");
  vm.runInContext("thermalLoadStates = state; renderThermalLoadCards(builder, 3);", ctx);
  const nodes = descendants(ctx.builder);
  const byField = (name) => nodes.find((n) => n.attributes?.["data-field"] === name);
  // type="number" inputs sanitize a non-numeric value to "" in a browser; they never keep or parse the markup
  assert.equal(byField("heating_rate").type, "number");
  assert.equal(byField("heating_rate").value, "");
  assert.equal(byField("supply_temperature").type, "number");
  assert.equal(byField("supply_temperature").value, "");
  assert.equal(byField("cooling_constant").value, "0.1");
  // a genuine text field keeps the markup-like string as an ordinary value
  const minTemperatures = nodes.filter((n) => n.attributes?.["data-field"] === "min_temperatures");
  assert.ok(minTemperatures.some((n) => n.type === "text" && n.value === malicious));
  assert.ok(nodes.some((n) => String(n.textContent ?? '').includes(malicious)));
  assert.ok(nodes.some((n) => String(n.className ?? '').includes("thermal-kind-select")));
  assert.ok(nodes.some((n) => String(n.className ?? '').includes("thermal-array-fields")));
  assert.equal(nodes.filter((n) => n.tag === "img" || n.tag === "script").length, 0);
});

test("system overview badges render malicious-looking values as text", () => {
  const { ctx, overview } = setup();
  ctx.saved = {
    set_use_pv: true,
    weather_forecast_method: malicious,
    costfun: malicious,
    load_forecast_method: malicious,
    number_of_deferrable_loads: 0,
  };
  vm.runInContext("updateConfigSystemOverviewBadges(saved);", ctx);
  assert.ok(descendants(overview).some((n) => String(n.textContent ?? '').includes(malicious)));
  assert.equal(descendants(overview).filter((n) => n.tag === "img").length, 0);
});

test("number inputs in the fixture sanitize like a browser", () => {
  const input = new FakeNode("input");
  input.value = malicious;
  input.type = "number";
  assert.equal(input.value, "");
  input.value = "1e3";
  assert.equal(input.value, "1e3");
  input.type = "text";
  input.value = malicious;
  assert.equal(input.value, malicious);
});

const SECTIONS = ["Local", "System", "Tariff", "Solar System (PV)", "Inverters", "Battery", "Deferrable Loads"];
const templateOf = (ids, { body = true } = {}) => ({
  content: {
    querySelectorAll: () => ids.map((id) => ({ id, querySelector: () => (body ? {} : null) })),
    cloneNode: () => new FakeNode("fragment"),
  },
});

test("the configuration fragment is validated before it is used as an inert template", () => {
  const { ctx, template } = setup();
  ctx.errors = [];
  vm.runInContext("errorAlert = (m) => errors.push(m);", ctx);
  template.content = templateOf(SECTIONS).content;
  assert.equal(vm.runInContext("getListTemplate()", ctx), template);
  assert.deepEqual(ctx.errors, []);
  for (const content of [templateOf([]).content, templateOf(["Local"]).content, templateOf(SECTIONS.slice(1)).content, templateOf(SECTIONS, { body: false }).content, {}]) {
    template.content = content;
    assert.equal(vm.runInContext("getListTemplate()", ctx), null);
  }
  assert.equal(ctx.errors.length, 5);
});
