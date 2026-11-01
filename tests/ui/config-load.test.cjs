// Run from repository root: node --test tests/ui/config-load.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/emhass/static/configuration_script.js', 'utf8');
const saved = { battery_nominal_energy_capacity: 23456, costfun: 'self-consumption' };

function setup(getConfigReply, { box = '{}' } = {}) {
  const el = () => ({ style: {}, innerHTML: '', textContent: '', disabled: false, listeners: {},
    classList: { add() {}, remove() {} }, addEventListener(type, fn) { this.listeners[type] = fn; } });
  const nodes = {};
  for (const id of ['configuration-container', 'save', 'yaml', 'defaults', 'json-toggle', 'alert', 'alert-text', 'loader']) nodes[id] = el();
  const calls = [];
  const state = { rendered: undefined, view: 'list' };
  const boxEl = { value: box, innerHTML: '' };
  nodes['configuration-container'].getElementsByClassName = () => (state.view === 'list' ? [{}] : []);
  const ctx = { window: {}, console, setTimeout: () => 0, Response,
    document: { getElementById: id => (id === 'config-box' ? boxEl : nodes[id]),
      getElementsByClassName: () => (state.view === 'list' ? [{}] : []) },
    fetch: async (url, options) => {
      calls.push({ url, method: options && options.method });
      if (url.includes('param_definitions')) return new Response('{}', { status: 200 });
      if (url.includes('configuration_list')) return new Response('<div class="section-card"></div>', { status: 200 });
      return getConfigReply(url);
    } };
  vm.createContext(ctx); vm.runInContext(source, ctx);
  // the list renderer is not under test here; record what it was asked to render
  ctx.loadConfigurationListView = (defs, config) => { state.rendered = config; nodes['configuration-container'].innerHTML = 'LIST'; };
  return { ctx, nodes, calls, state, boxEl, load: () => ctx.window.onload() };
}
const json = (status, body) => new Response(JSON.stringify(body), { status });
const setConfigCalls = s => s.calls.filter(c => c.url === 'set-config');
const failClosed = (s) => {
  assert.equal(s.state.rendered, undefined, 'no form rendered from a failed load');
  assert.equal(s.nodes.save.listeners.click, undefined, 'Save not attached');
  assert.equal(s.nodes.save.disabled, true, 'Save disabled');
  assert.equal(s.nodes['configuration-container'].innerHTML, '');
  assert.equal(s.nodes.alert.style.display, 'block', 'failure reported in the alert box');
  assert.match(s.nodes['alert-text'].textContent, /Reload the page to try again/, 'a recovery step is offered');
  assert.deepEqual(setConfigCalls(s), [], 'nothing is sent to /set-config');
};

test('GET /get-config 500 with a JSON error array fails closed and shows the server message', async () => {
  const s = setup(() => json(500, ['failed to retrieve default config file']));
  await s.load(); failClosed(s);
  assert.match(s.nodes['alert-text'].textContent, /failed to retrieve default config file/);
});
test('GET /get-config 500 with an empty JSON list reports the HTTP status', async () => {
  const s = setup(() => json(500, []));
  await s.load(); failClosed(s);
  assert.match(s.nodes['alert-text'].textContent, /HTTP 500/);
});
test('GET /get-config 500 with an HTML body fails closed and reports the HTTP status', async () => {
  const s = setup(() => new Response('<html>Bad Gateway</html>', { status: 502 }));
  await s.load(); failClosed(s);
  assert.match(s.nodes['alert-text'].textContent, /HTTP 502/);
});
test('rejected GET /get-config request fails closed and reports the error', async () => {
  const s = setup(() => { throw new TypeError('Failed to fetch'); });
  await s.load(); failClosed(s);
  assert.match(s.nodes['alert-text'].textContent, /Failed to fetch/);
});
for (const [name, body] of [['malformed JSON', '{"costfun":'], ['a JSON array', '[]'], ['JSON null', 'null'], ['an empty object', '{}']]) {
  test(`HTTP 200 with ${name} fails closed`, async () => {
    const s = setup(() => new Response(body, { status: 200 }));
    await s.load(); failClosed(s);
  });
}
test('HTTP 200 renders the saved configuration and attaches the buttons', async () => {
  const s = setup(() => json(200, saved));
  await s.load();
  assert.deepEqual(s.state.rendered, saved);
  for (const id of ['save', 'yaml', 'defaults', 'json-toggle']) assert.equal(typeof s.nodes[id].listeners.click, 'function', id);
  assert.equal(s.nodes.save.disabled, false);
});
test('List/Box toggle keeps the current view when GET /get-config fails', async () => {
  const s = setup(() => json(500, ['failed to retrieve default config file']));
  s.nodes['configuration-container'].innerHTML = 'EDITED FORM';
  await s.ctx.ToggleView({}, '<div></div>', false);
  assert.equal(s.nodes['configuration-container'].innerHTML, 'EDITED FORM');
  assert.equal(s.state.rendered, undefined);
  assert.match(s.nodes['alert-text'].textContent, /failed to retrieve default config file/);
});
test('Defaults keeps the current form when GET /get-config/defaults fails with a JSON error', async () => {
  const s = setup(() => json(500, ['failed to retrieve default config file']));
  s.nodes['configuration-container'].innerHTML = 'EDITED FORM';
  await s.ctx.ToggleView({}, '<div></div>', true);
  assert.equal(s.nodes['configuration-container'].innerHTML, 'EDITED FORM');
  assert.equal(s.state.rendered, undefined);
  assert.match(s.nodes['alert-text'].textContent, /failed to retrieve default config file/);
  assert.deepEqual(setConfigCalls(s), []);
});
test('Defaults keeps the current form when GET /get-config/defaults fails with an HTML body', async () => {
  const s = setup(() => new Response('<html>Bad Gateway</html>', { status: 502 }));
  s.nodes['configuration-container'].innerHTML = 'EDITED FORM';
  await s.ctx.ToggleView({}, '<div></div>', true);
  assert.equal(s.nodes['configuration-container'].innerHTML, 'EDITED FORM');
  assert.equal(s.state.rendered, undefined);
  assert.match(s.nodes['alert-text'].textContent, /HTTP 502/);
});
test('Defaults keeps the current form when the GET /get-config/defaults request is rejected', async () => {
  const s = setup(() => { throw new TypeError('Failed to fetch'); });
  s.nodes['configuration-container'].innerHTML = 'EDITED FORM';
  await s.ctx.ToggleView({}, '<div></div>', true);
  assert.equal(s.nodes['configuration-container'].innerHTML, 'EDITED FORM');
  assert.equal(s.state.rendered, undefined);
  assert.match(s.nodes['alert-text'].textContent, /Failed to fetch/);
  assert.deepEqual(setConfigCalls(s), []);
});
for (const [name, body] of [['malformed JSON', '{"costfun":'], ['a JSON array', '[]'], ['JSON null', 'null'], ['an empty object', '{}']]) {
  test(`Defaults keeps the current form when GET /get-config/defaults returns 200 with ${name}`, async () => {
    const s = setup(() => new Response(body, { status: 200 }));
    s.nodes['configuration-container'].innerHTML = 'EDITED FORM';
    await s.ctx.ToggleView({}, '<div></div>', true);
    assert.equal(s.nodes['configuration-container'].innerHTML, 'EDITED FORM');
    assert.equal(s.state.rendered, undefined);
    assert.equal(s.nodes.alert.style.display, 'block');
    assert.deepEqual(setConfigCalls(s), []);
  });
}
test('List/Box toggle still switches views when GET /get-config succeeds', async () => {
  const s = setup(() => json(200, saved));
  await s.ctx.ToggleView({}, '<div></div>', false);
  assert.equal(s.state.rendered, undefined);
  assert.match(s.nodes['configuration-container'].innerHTML, /config-box/);
  assert.deepEqual(JSON.parse(s.boxEl.innerHTML), saved);
});

async function save(reply) {
  const s = setup(() => reply()); s.state.view = 'box';
  const out = {}; s.ctx.showChangeStatus = (st, log) => { out.status = st; out.log = log; };
  await s.ctx.saveConfiguration({});
  return { s, out };
}
test('/set-config 200 reports success', async () => {
  const { s, out } = await save(() => json(200, []));
  assert.equal(out.status, 200); assert.equal(setConfigCalls(s).length, 1);
});
test('/set-config JSON error reports the server messages', async () => {
  const { out } = await save(() => json(400, ['bad value']));
  assert.equal(out.status, 400); assert.deepEqual(out.log, ['bad value']);
});
test('/set-config non-JSON error reports the HTTP status', async () => {
  const { out } = await save(() => new Response('<html>Bad Gateway</html>', { status: 502 }));
  assert.equal(out.status, 502); assert.deepEqual([...out.log], ['HTTP 502']);
});
test('rejected /set-config request is reported with the error message', async () => {
  const { out } = await save(() => { throw new TypeError('Failed to fetch'); });
  assert.notEqual(out.status, 200); assert.match(out.log.join(' '), /Failed to fetch/);
});
