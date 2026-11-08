// Run from the repository root: node --test tests/ui/inverter-sync.test.cjs
// #1150: the per-inverter lists in the System section follow number_of_inverters.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/emhass/static/configuration_script.js', 'utf8');
const defs = JSON.parse(fs.readFileSync('src/emhass/static/data/param_definitions.json', 'utf8'));
const utilsSource = fs.readFileSync('src/emhass/utils.py', 'utf8');

const CAPS = ['inverter_ac_output_max', 'inverter_ac_input_max'];
const BROADCAST = ['inverter_efficiency_dc_ac', 'inverter_efficiency_ac_dc', 'inverter_stress_cost', 'compute_curtailment'];
const LISTS = [...CAPS, ...BROADCAST];
const CURVES = ['inverter_power_curve_dc_ac', 'inverter_power_curve_ac_dc'];
const RENDERED = [...LISTS, ...CURVES, 'number_of_inverters', 'inverter_is_hybrid'];

// a minimal DOM: each param div holds the inputs parsed from the HTML the real buildParamElement returns
function classList() {
  const set = new Set();
  return { add: c => set.add(c), remove: c => set.delete(c), contains: c => set.has(c) };
}
function paramElement(id) {
  const el = { id, tagName: 'DIV', inputs: [], classList: classList() };
  const container = { insertAdjacentHTML: (_pos, html) => addInputs(el, html) };
  el.getElementsByTagName = tag => (tag === 'input' ? el.inputs.slice() : []);
  el.querySelectorAll = sel => (sel === 'input' ? el.inputs.slice() : []);
  el.getElementsByClassName = cls => (cls === 'param-input' ? [container] : cls === 'param_input' ? el.inputs.slice() : []);
  return el;
}
function addInputs(el, html) {
  for (const m of html.matchAll(/<input ([^>]*)>/g)) {
    const attr = name => (m[1].match(new RegExp(`\\b${name}="([^"]*)"`)) || [])[1];
    const type = attr('type');
    const input = { type, value: (attr('value') ?? '').replace(/&quot;/g, '"').replace(/&amp;/g, '&'),
      checked: /\bchecked\b/.test(m[1].replace(/value="[^"]*"/, '')), listeners: {},
      addEventListener(event, fn) { (this.listeners[event] ||= []).push(fn); },
      fire(event) { for (const fn of this.listeners[event] || []) fn(); },
      remove() { el.inputs.splice(el.inputs.indexOf(input), 1); } };
    input.parentNode = type === 'checkbox' ? { tagName: 'LABEL', remove: () => input.remove() } : { tagName: 'DIV' };
    el.inputs.push(input);
  }
}
// realAlert keeps the script's own errorAlert; otherwise it is wrapped to record each message
function setup(config, { section = defs.System, realAlert = false } = {}) {
  const elements = { alert: { style: { display: 'none' } }, 'alert-text': { textContent: '' } };
  const ctx = { window: {}, console, AbortSignal, document: { getElementById: id => elements[id] ?? null } };
  vm.createContext(ctx); vm.runInContext(source, ctx);
  ctx.alerts = [];
  if (!realAlert) {
    const errorAlert = ctx.errorAlert;
    ctx.errorAlert = msg => { ctx.alerts.push(msg); return errorAlert(msg); };
  }
  for (const name of RENDERED) {
    if (!(name in section)) continue;
    elements[name] = paramElement(name);
    addInputs(elements[name], ctx.buildParamElement(section[name], name, config));
  }
  return { ctx, elements, count: () => elements.number_of_inverters.inputs[0] };
}
const values = (s, name) => s.elements[name].inputs.map(i => (i.type === 'checkbox' ? i.checked : i.value));
const lengths = s => Object.fromEntries(LISTS.map(n => [n, s.elements[n].inputs.length]));
const allAt = n => Object.fromEntries(LISTS.map(name => [name, n]));
// base-safe: on a build without the sync this does nothing, so the tests fail on what they assert, not on a TypeError
const start = (s, config) => { if (typeof s.ctx.setupInverterSync === 'function') s.ctx.setupInverterSync(defs, config); };
// a browser fires change only when the value differs from the last committed one, so tests only change it to a new value
function changeCount(s, value) {
  assert.notEqual(s.count().value, String(value), 'change fires only for a new value');
  s.count().value = String(value); s.count().fire('change');
}
const shown = s => s.elements.alert.style.display;

test('the JS lists are the backend INVERTER_CAP_PARAMS and INVERTER_BROADCAST_PARAMS', () => {
  const caps = utilsSource.match(/^INVERTER_CAP_PARAMS[^=]*= \(([^)]*)\)/m);
  const broadcast = utilsSource.match(/^INVERTER_BROADCAST_PARAMS[^=]*= \{([^}]*)\}/m);
  assert.ok(caps && broadcast, 'INVERTER_CAP_PARAMS and INVERTER_BROADCAST_PARAMS not found in utils.py');
  const s = setup({});
  assert.deepEqual([...vm.runInContext('INVERTER_CAP_PARAMS', s.ctx)], caps[1].match(/"(\w+)"/g).map(x => x.slice(1, -1)));
  assert.deepEqual([...vm.runInContext('INVERTER_BROADCAST_PARAMS', s.ctx)], broadcast[1].match(/"(\w+)":/g).map(x => x.slice(1, -2)));
  for (const name of LISTS) assert.ok(defs.System[name].input.startsWith('array.'), `${name} is a System array param`);
});

test('the synced lists lose their +/- buttons, the index lists keep them', () => {
  const body = { innerHTML: '' };
  const sectionEl = { getElementsByClassName: c => (c === 'section-body' ? [body] : []), querySelectorAll: () => [] };
  const ctx = { window: {}, console: { ...console, debug() {} }, document: { getElementById: id => (id === 'System' ? sectionEl : null) } };
  vm.createContext(ctx); vm.runInContext(source, ctx);
  ctx.buildParamContainers('System', defs.System, {}, []);
  for (const name of LISTS) assert.doesNotMatch(body.innerHTML, new RegExp(`input-plus ${name}"`), name);
  for (const name of ['battery_inverter_index', 'pv_plant_inverter_index', 'mlforecaster_weather_features'])
    assert.match(body.innerHTML, new RegExp(`input-plus ${name}"`), name);
});

test('one inverter is a no-op on the default and on a saved configuration', () => {
  for (const config of [{}, { number_of_inverters: 1, inverter_ac_output_max: 8000, inverter_efficiency_dc_ac: 0.97, compute_curtailment: true }]) {
    const s = setup(config);
    const before = Object.fromEntries(LISTS.map(n => [n, values(s, n)]));
    start(s, config);
    assert.deepEqual(Object.fromEntries(LISTS.map(n => [n, values(s, n)])), before);
    assert.deepEqual(lengths(s), allAt(1));
    assert.deepEqual(s.ctx.alerts, []);
  }
});

test('raising the count adds rows and keeps typed values; new cap rows start empty', () => {
  const s = setup({});
  start(s, {});
  s.elements.inverter_ac_output_max.inputs[0].value = '6000';
  changeCount(s, 3);
  assert.deepEqual(lengths(s), allAt(3));
  // a cap has no safe default above one inverter: the save asks for each one
  assert.deepEqual(values(s, 'inverter_ac_output_max'), ['6000', '', '']);
  assert.deepEqual(values(s, 'inverter_ac_input_max'), ['5000', '', '']);
  // a broadcast setting takes the definition default, as new battery rows do
  assert.deepEqual(values(s, 'inverter_efficiency_dc_ac'), ['1', '1', '1']);
  assert.deepEqual(values(s, 'inverter_stress_cost'), ['0', '0', '0']);
  assert.deepEqual(values(s, 'compute_curtailment'), [false, false, false]);
});

test('lowering the count drops rows from the end only, never below one', () => {
  const config = { number_of_inverters: 3, inverter_ac_output_max: [6000, 4000, 3000] };
  const s = setup(config);
  start(s, config);
  changeCount(s, 2);
  assert.deepEqual(values(s, 'inverter_ac_output_max'), ['6000', '4000']);
  assert.deepEqual(lengths(s), allAt(2));
  changeCount(s, 1);
  assert.deepEqual(lengths(s), allAt(1));
});

test('a stored configuration that already matches the count is left as it is', () => {
  const config = { number_of_inverters: 2, inverter_ac_output_max: [6000, 3000], inverter_ac_input_max: [6000, 3000],
    inverter_efficiency_dc_ac: [0.97, 0.95], inverter_efficiency_ac_dc: [0.96, 0.94], inverter_stress_cost: [0, 0.1], compute_curtailment: [true, false] };
  const s = setup(config);
  start(s, config);
  assert.deepEqual(values(s, 'inverter_efficiency_dc_ac'), ['0.97', '0.95']);
  assert.deepEqual(values(s, 'compute_curtailment'), [true, false]);
  assert.deepEqual(lengths(s), allAt(2));
  assert.deepEqual(s.ctx.alerts, []);
});

test('first render: a single stored value is expanded the way the backend reads it', () => {
  // what /get-config returns when the stored configuration has a fault (build_params then leaves it as written)
  const config = { number_of_inverters: 2, inverter_ac_output_max: 8000, inverter_ac_input_max: [5000, 3000],
    inverter_efficiency_dc_ac: 0.95, compute_curtailment: true };
  const s = setup(config);
  start(s, config);
  assert.deepEqual(values(s, 'inverter_efficiency_dc_ac'), ['0.95', '0.95']);
  assert.deepEqual(values(s, 'compute_curtailment'), [true, true]);
  assert.deepEqual(values(s, 'inverter_ac_output_max'), ['8000', '']);
  assert.deepEqual(values(s, 'inverter_ac_input_max'), ['5000', '3000']);
  // keys missing from the stored configuration: a broadcast one takes the definition default, a cap stays empty
  assert.deepEqual(values(s, 'inverter_efficiency_ac_dc'), ['1', '1']);
  assert.deepEqual(s.ctx.alerts, []);
  // after the first render a change uses the definition default again
  changeCount(s, 3);
  assert.deepEqual(values(s, 'inverter_efficiency_dc_ac'), ['0.95', '0.95', '1']);
  assert.deepEqual(values(s, 'inverter_ac_output_max'), ['8000', '', '']);
});

test('first render: one-entry lists from a one-inverter save, with the count then raised to 3', () => {
  // a one-inverter save writes one-entry lists; the count then raised by hand in config.json
  const config = { number_of_inverters: 3, inverter_ac_output_max: [6000], inverter_efficiency_dc_ac: [0.95], compute_curtailment: [true] };
  const s = setup(config);
  start(s, config);
  assert.deepEqual(values(s, 'inverter_efficiency_dc_ac'), ['0.95', '0.95', '0.95']);
  assert.deepEqual(values(s, 'compute_curtailment'), [true, true, true]);
  assert.deepEqual(values(s, 'inverter_ac_output_max'), ['6000', '', '']);
  assert.deepEqual(values(s, 'inverter_ac_input_max'), ['5000', '', '']);
});

test('first render: a stored cap key that is missing still gets empty rows', () => {
  const s = setup({ number_of_inverters: 2 });
  start(s, { number_of_inverters: 2 });
  assert.deepEqual(values(s, 'inverter_ac_output_max'), ['5000', '']);
});

test('first render: a list longer than the count is kept and named, never trimmed', () => {
  const config = { number_of_inverters: 1, inverter_ac_output_max: [6000, 4000] };
  const s = setup(config);
  start(s, config);
  assert.deepEqual(values(s, 'inverter_ac_output_max'), ['6000', '4000']);
  assert.equal(s.ctx.alerts.length, 1);
  assert.match(s.ctx.alerts[0], /number_of_inverters is 1 but the stored configuration has inverter_ac_output_max \(2 entries\)/);
  assert.match(s.ctx.alerts[0], /change number_of_inverters to another value and back/);
  // the remedy the alert gives: another value, then back
  changeCount(s, 2);
  changeCount(s, 1);
  assert.deepEqual(lengths(s), allAt(1));
  assert.deepEqual(values(s, 'inverter_ac_output_max'), ['6000']);
  assert.equal(shown(s), 'none', 'the working sync takes its own alert down');
});

test('first render: a short list of two or more is kept and named, never padded', () => {
  const config = { number_of_inverters: 3, inverter_ac_output_max: [6000, 3000], inverter_stress_cost: [0.1, 0.2],
    inverter_efficiency_dc_ac: [0.9, 0.9, 0.9, 0.9] };
  const s = setup(config);
  start(s, config);
  assert.deepEqual(values(s, 'inverter_ac_output_max'), ['6000', '3000']);
  assert.deepEqual(values(s, 'inverter_stress_cost'), ['0.1', '0.2']);
  assert.deepEqual(values(s, 'inverter_efficiency_dc_ac'), ['0.9', '0.9', '0.9', '0.9']);
  assert.equal(s.ctx.alerts.length, 1);
  for (const named of ['inverter_ac_output_max \\(2 entries\\)', 'inverter_stress_cost \\(2 entries\\)', 'inverter_efficiency_dc_ac \\(4 entries\\)'])
    assert.match(s.ctx.alerts[0], new RegExp(named));
  // single values next to them are still expanded
  assert.deepEqual(values(s, 'inverter_ac_input_max'), ['5000', '', '']);
  assert.doesNotMatch(s.ctx.alerts[0], /inverter_ac_input_max/);
});

test('a count that is not usable resizes nothing', () => {
  for (const bad of ['', ' ', '0', '-1', '2.5', 'abc']) {
    const s = setup({});
    start(s, {});
    changeCount(s, bad);
    assert.deepEqual(lengths(s), allAt(1), JSON.stringify(bad));
    assert.deepEqual(s.ctx.alerts, [], JSON.stringify(bad));
  }
});

test('a huge count is refused with a message, without building a row for it', () => {
  const s = setup({});
  start(s, {});
  let added = 0;
  const plusElements = s.ctx.plusElements;
  s.ctx.plusElements = (...args) => { added += 1; return plusElements(...args); };
  changeCount(s, 1e9);
  changeCount(s, '1e21');
  assert.equal(added, 0);
  assert.deepEqual(lengths(s), allAt(1));
  assert.match(s.ctx.alerts[0], /number_of_inverters is 1000000000: .*up to 100 inverters/);
  assert.match(s.ctx.alerts[1], /number_of_inverters is 1e21: /);
  assert.equal(shown(s), 'block');
  changeCount(s, 100);
  assert.deepEqual(lengths(s), allAt(100));
  assert.equal(shown(s), 'none', 'a corrected count takes its own alert down');
});

test('a working sync leaves an alert raised by something else in place', () => {
  const s = setup({}, { realAlert: true });
  start(s, {});
  s.ctx.errorAlert('set-config: some other problem');
  changeCount(s, 2);
  assert.equal(shown(s), 'block');
  changeCount(s, 500);
  assert.match(s.elements['alert-text'].textContent, /number_of_inverters is 500/);
  changeCount(s, 2);
  assert.equal(shown(s), 'none');
});

test('the lists follow the count on change, not on every keystroke', () => {
  const s = setup({ number_of_inverters: 3 });
  start(s, { number_of_inverters: 3 });
  s.elements.inverter_ac_output_max.inputs[2].value = '2500';
  assert.equal(s.count().listeners.input, undefined, 'no input listener on the count');
  // typing 13 passes through "1": nothing may shrink until the change event
  s.count().value = '1';
  changeCount(s, 13);
  assert.deepEqual(values(s, 'inverter_ac_output_max').slice(0, 3), ['5000', '', '2500']);
  assert.deepEqual(lengths(s), allAt(13));
});

// what buildParamContainers does for a param with "requires": disable it now, and again on every input event
function attachGenericRequires(s) {
  const hybrid = s.elements.inverter_is_hybrid.inputs[0];
  for (const name of CURVES) {
    const rule = () => s.ctx.checkRequirements(hybrid, s.elements[name], defs.System[name].requires.inverter_is_hybrid);
    hybrid.addEventListener('input', rule);
    rule();
  }
  return hybrid;
}
const disabled = s => CURVES.map(n => s.elements[n].classList.contains('requirement-disable'));

test('above one inverter the power curves stay editable whatever inverter_is_hybrid says', () => {
  const s = setup({ inverter_is_hybrid: false });
  const hybrid = attachGenericRequires(s);
  assert.deepEqual(disabled(s), [true, true]);
  start(s, {});
  assert.deepEqual(disabled(s), [true, true], 'one inverter keeps the requires rule');
  changeCount(s, 2);
  assert.deepEqual(disabled(s), [false, false]);
  // toggling inverter_is_hybrid fires the generic listener first, then the sync's
  hybrid.checked = false;
  hybrid.fire('input');
  assert.deepEqual(disabled(s), [false, false]);
  changeCount(s, 1);
  assert.deepEqual(disabled(s), [true, true]);
  hybrid.checked = true;
  hybrid.fire('input');
  assert.deepEqual(disabled(s), [false, false]);
});

test('a stored count above one also unlocks the curves on the first render', () => {
  const config = { number_of_inverters: 2, inverter_is_hybrid: false };
  const s = setup(config);
  attachGenericRequires(s);
  start(s, config);
  assert.deepEqual(disabled(s), [false, false]);
});

test('the save sends one entry per inverter for every synced list', async () => {
  const s = setup({});
  start(s, {});
  changeCount(s, 2);
  s.elements.inverter_ac_output_max.inputs[1].value = '3000';
  s.elements.inverter_ac_input_max.inputs[1].value = '2500';
  s.elements.inverter_efficiency_dc_ac.inputs[1].value = '0.95';
  s.elements.compute_curtailment.inputs[1].checked = true;
  let sent;
  s.ctx.document.getElementsByClassName = () => [{}];
  s.ctx.fetch = async (_url, options) => { sent = JSON.parse(options.body); return { status: 200, json: async () => [] }; };
  s.ctx.showChangeStatus = () => {};
  const section = Object.fromEntries(Object.entries(defs.System).filter(([n]) => n in s.elements));
  await s.ctx.saveConfiguration({ System: section });
  assert.equal(sent.number_of_inverters, 2);
  assert.deepEqual(sent.inverter_ac_output_max, [5000, 3000]);
  assert.deepEqual(sent.inverter_ac_input_max, [5000, 2500]);
  assert.deepEqual(sent.inverter_efficiency_dc_ac, [1, 0.95]);
  assert.deepEqual(sent.compute_curtailment, [false, true]);
  for (const name of LISTS) assert.equal(sent[name].length, 2, name);
});

test('no number_of_inverters row: nothing to sync, nothing thrown', () => {
  const section = Object.fromEntries(Object.entries(defs.System).filter(([n]) => n !== 'number_of_inverters'));
  const s = setup({}, { section });
  start(s, {});
  assert.deepEqual(lengths(s), allAt(1));
});

test('the list view runs the sync once the sections are built', () => {
  const calls = [];
  const container = { innerHTML: '' };
  const ctx = { window: {}, console: { ...console, debug() {}, log() {} },
    document: { getElementById: id => (id === 'configuration-container' ? container : null) } };
  vm.createContext(ctx); vm.runInContext(source, ctx);
  ctx.buildParamContainers = section => calls.push(['build', section]);
  ctx.setupInverterSync = (definitions, config) => calls.push(['sync', definitions, config]);
  const config = { number_of_inverters: 2 };
  const definitions = { Local: {}, System: {} };
  ctx.loadConfigurationListView(definitions, config, '<div></div>');
  assert.deepEqual(calls.map(c => c[0]), ['build', 'build', 'sync']);
  assert.equal(calls[2][1], definitions);
  assert.equal(calls[2][2], config);
});
