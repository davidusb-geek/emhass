// Run from repository root: node --test tests/ui/config-roundtrip.test.cjs
// Renders the real parameter definitions with the real configuration_script.js, then saves them
// back. The DOM is a minimal stand-in that reads the markup the script generates; it is not a
// browser rendering test.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/emhass/static/configuration_script.js', 'utf8');
const defsText = fs.readFileSync('src/emhass/static/data/param_definitions.json', 'utf8');
const defs = JSON.parse(defsText);
const listHtml = fs.readFileSync('src/emhass/static/configuration_list.html', 'utf8');

// unrelated customised settings: battery, inverter, grid, tariff cost function, optimisation
const CUSTOM = {
  battery_nominal_energy_capacity: 23456, battery_discharge_power_max: 4321, battery_charge_power_max: 3210,
  battery_minimum_state_of_charge: 0.11, battery_target_state_of_charge: 0.77, inverter_ac_output_max: 7777,
  inverter_ac_input_max: 6666, maximum_power_from_grid: 11111, set_use_battery: true, costfun: 'self-consumption',
  optimization_time_step: 15,
};
const DEFAULT_REPLY = { battery_nominal_energy_capacity: 9999, costfun: 'profit', optimization_time_step: 30 };

const decode = s => s.replaceAll('&quot;', '"').replaceAll('&lt;', '<').replaceAll('&gt;', '>').replaceAll('&amp;', '&');
function parseParams(html) {
  const params = {};
  const blocks = html.split('<div class="param" id="').slice(1);
  for (const block of blocks) {
    const name = block.slice(0, block.indexOf('"'));
    const inputs = [];
    for (const m of block.matchAll(/<(input|select|textarea)\b([^>]*)>(?:([\s\S]*?)<\/(?:select|textarea)>)?/g)) {
      const [, tag, attrs, inner] = m;
      if (tag === 'input') {
        const value = /\bvalue=\s*(?:"([^"]*)"|([^\s>]+))/.exec(attrs);
        inputs.push({ type: /\btype="(\w+)"/.exec(attrs)[1], value: value ? decode(value[1] ?? value[2]) : '', checked: /\bchecked\b/.test(attrs) });
      } else if (tag === 'select') {
        const options = [...inner.matchAll(/<option\b([^>]*)>([^<]*)/g)];
        const chosen = options.find(o => /selected/.test(o[1])) || options[0];
        inputs.push({ type: 'select-one', value: chosen[2], checked: false });
      } else {
        inputs.push({ type: 'textarea', value: decode(inner), checked: false });
      }
    }
    params[name] = inputs.map(input => ({ ...input, getAttribute: () => null, setAttribute() {}, addEventListener() {} }));
  }
  return params;
}

function setup({ getConfig = () => ({}) } = {}) {
  const bodies = Object.fromEntries(Object.keys(defs).map(section => [section, { innerHTML: '' }]));
  const sections = Object.fromEntries(Object.keys(defs).map(section => [section, {
    getElementsByClassName: () => [bodies[section]], querySelectorAll: () => [] }]));
  const header = (id, type) => ({ id, tagName: 'INPUT', type, value: '', checked: false, style: {}, addEventListener() {}, getElementsByClassName: () => [] });
  const headers = { set_use_battery: header('set_use_battery', 'checkbox'), set_use_pv: header('set_use_pv', 'checkbox'),
    number_of_deferrable_loads: header('number_of_deferrable_loads', 'number'), number_of_batteries: header('number_of_batteries', 'number') };
  const node = () => ({ style: {}, innerHTML: '', textContent: '', listeners: {}, classList: { add() {}, remove() {} },
    addEventListener(type, fn) { this.listeners[type] = fn; } });
  const nodes = Object.fromEntries(['alert', 'alert-text', 'loader', 'save', 'yaml', 'defaults', 'json-toggle'].map(id => [id, node()]));
  const container = node();
  let rendered = false, cache = { html: null, params: {} };
  Object.defineProperty(container, 'innerHTML', { get() { return ''; }, set(html) {
    rendered = html.includes('section-card');
    for (const body of Object.values(bodies)) body.innerHTML = ''; } });
  container.getElementsByClassName = () => (rendered ? [{}] : []);
  const posts = [], calls = [];
  const params = () => {
    const html = Object.values(bodies).map(b => b.innerHTML).join('');
    if (cache.html !== html) cache = { html, params: parseParams(html) };
    return cache.params;
  };
  const ctx = { window: {}, console: { ...console, debug() {}, log() {} }, setTimeout: () => 0, Response,
    document: {
      getElementById: id => {
        if (id === 'configuration-container') return container;
        if (headers[id]) return headers[id];
        if (sections[id]) return sections[id];
        if (nodes[id]) return nodes[id];
        const inputs = params()[id];
        return inputs ? { tagName: 'DIV', style: {}, getElementsByClassName: c => (c === 'param_input' ? inputs : []),
          querySelector: () => null, querySelectorAll: () => [], classList: { add() {}, remove() {}, contains: () => false } } : null;
      },
      getElementsByClassName: c => (c === 'section-card' && rendered ? [{}] : []) },
    fetch: async (url, options) => {
      calls.push(url);
      if (options && options.method === 'POST') { posts.push(JSON.parse(options.body)); return new Response('[]', { status: 200 }); }
      if (url.includes('param_definitions')) return new Response(defsText, { status: 200 });
      if (url.includes('configuration_list')) return new Response(listHtml, { status: 200 });
      if (url === 'get-config/defaults') return new Response(JSON.stringify(DEFAULT_REPLY), { status: 200 });
      return new Response(JSON.stringify(getConfig()), { status: 200 });
    } };
  vm.createContext(ctx); vm.runInContext(source, ctx);
  ctx.headerElement = () => {}; // header-driven row counts are not under test
  const shown = id => (params()[id] || [headers[id]])[0];
  return { ctx, nodes, posts, calls, shown };
}
// per-battery parameters are saved as one-element lists
const single = value => (Array.isArray(value) && value.length === 1 ? value[0] : value);
const display = input => (input.type === 'checkbox' ? input.checked : input.type === 'number' ? Number.parseFloat(input.value) : input.value);

test('loading a customised configuration and saving it preserves every customised setting', async () => {
  const s = setup();
  s.ctx.loadConfigurationListView(defs, CUSTOM, listHtml);
  for (const [name, value] of Object.entries(CUSTOM)) assert.equal(display(s.shown(name)), value, `${name} rendered`);
  await s.ctx.saveConfiguration(defs);
  assert.equal(s.posts.length, 1);
  for (const [name, value] of Object.entries(CUSTOM)) assert.deepEqual(single(s.posts[0][name]), value, `${name} saved`);
});

test('control: the same save from an empty configuration writes the parameter defaults, not the customised values', async () => {
  const s = setup();
  s.ctx.loadConfigurationListView(defs, {}, listHtml);
  await s.ctx.saveConfiguration(defs);
  assert.notEqual(single(s.posts[0].battery_nominal_energy_capacity), CUSTOM.battery_nominal_energy_capacity);
  assert.notEqual(s.posts[0].costfun, CUSTOM.costfun);
});

test('Defaults only changes the form; nothing is saved until the user presses Save', async () => {
  const s = setup({ getConfig: () => CUSTOM });
  await s.ctx.window.onload();
  assert.equal(display(s.shown('battery_nominal_energy_capacity')), CUSTOM.battery_nominal_energy_capacity);
  assert.equal(s.posts.length, 0, 'loading the page saves nothing');

  await s.nodes.defaults.listeners.click();
  assert.ok(s.calls.includes('get-config/defaults'));
  assert.equal(display(s.shown('battery_nominal_energy_capacity')), DEFAULT_REPLY.battery_nominal_energy_capacity);
  assert.equal(display(s.shown('costfun')), DEFAULT_REPLY.costfun);
  assert.equal(s.posts.length, 0, 'pressing Defaults saves nothing');

  await s.nodes.save.listeners.click();
  assert.equal(s.posts.length, 1);
  assert.equal(single(s.posts[0].battery_nominal_energy_capacity), DEFAULT_REPLY.battery_nominal_energy_capacity);
  assert.equal(s.posts[0].costfun, DEFAULT_REPLY.costfun);
});
