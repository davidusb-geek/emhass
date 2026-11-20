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
    this.value = "";
    this.textContent = "";
    this.style = {};
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
  assert.equal(byField("heating_rate").value, malicious);
  assert.equal(byField("supply_temperature").value, malicious);
  assert.ok(nodes.some((n) => n.textContent.includes(malicious)));
  assert.ok(nodes.some((n) => n.className.includes("thermal-kind-select")));
  assert.ok(nodes.some((n) => n.className.includes("thermal-array-fields")));
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
  assert.ok(descendants(overview).some((n) => n.textContent.includes(malicious)));
  assert.equal(descendants(overview).filter((n) => n.tag === "img").length, 0);
});

test("the configuration fragment is cloned from an inert template", () => {
  const { ctx, template } = setup();
  assert.equal(vm.runInContext("getListTemplate()", ctx), template);
  assert.match(source, /replaceChildren\(list_template\.content\.cloneNode\(true\)\)/);
  assert.doesNotMatch(source, /innerHTML\s*=\s*list_template/);
});
