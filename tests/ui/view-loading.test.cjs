// Run from repository root: node --test tests/ui/view-loading.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/emhass/static/script.js', 'utf8');
const BASIC = fs.readFileSync('src/emhass/static/basic.html', 'utf8');
const ADVANCED = fs.readFileSync('src/emhass/static/advanced.html', 'utf8');
const idsOf = html => [...html.matchAll(/\bid=["']?([\w-]+)/g)].map(m => m[1]);

// Minimal DOM: setting innerHTML registers the elements it declares, so the real
// script wires the real templates' controls. Not a browser rendering test.
function setup({ stored, replies, savedInputs, realSavedData }) {
  const registry = {}, storage = stored ? { TabSelection: stored } : {};
  if (savedInputs !== undefined) storage.input_container_content = savedInputs;
  const element = id => ({ id, style: {}, textContent: '', innerHTML: '', value: 'List', listeners: {},
    classList: { add() {}, remove() {} }, addEventListener(type, fn) { this.listeners[type] = fn; } });
  const container = element('TabSelection');
  Object.defineProperty(container, 'innerHTML', {
    get() { return this._html || ''; },
    set(html) { this._html = html; this.firstElementChild = { id: idsOf(html)[0] || '' };
      for (const id of idsOf(html)) registry[id] = element(id); } });
  for (const id of ['alert', 'alert-text', 'basicOrAdvanced']) registry[id] = element(id);
  registry.TabSelection = container;
  const calls = [];
  const ctx = { window: {}, console, Response, setTimeout: () => 0,
    localStorage: { getItem: k => storage[k] ?? null, setItem: (k, v) => { storage[k] = String(v); }, removeItem: k => { delete storage[k]; } },
    document: { getElementById: id => registry[id] || null, querySelectorAll: () => [],
      createElement: () => ({ content: { querySelector: sel => (idsOf(ctx.probeHtml || '').includes(sel.slice(1)) ? {} : null),
          get firstElementChild() { const id = idsOf(ctx.probeHtml || '')[0]; return id === undefined ? null : { id }; } },
        set innerHTML(html) { ctx.probeHtml = html; } }) },
    fetch: async (url) => { calls.push(url); return replies[url.replace('static/', '')](); } };
  vm.createContext(ctx); vm.runInContext(source, ctx);
  if (!realSavedData) ctx.getSavedData = () => {};
  ctx.dictInputs = () => {}; ctx.initStickyTables = () => {};
  const rejections = [];
  const onRejection = e => rejections.push(e);
  process.on('unhandledRejection', onRejection);
  return { ctx, registry, container, storage, calls, rejections, load: () => ctx.window.onload(),
    toggle: () => registry.basicOrAdvanced.listeners.click(),
    done: () => process.off('unhandledRejection', onRejection) };
}
const ok = html => () => new Response(html, { status: 200 });
const healthy = { 'basic.html': ok(BASIC), 'advanced.html': ok(ADVANCED), '/get-config': ok('{}') };
const failures = {
  'a rejected request': () => { throw new TypeError('Failed to fetch'); },
  'an HTTP 502 error page': () => new Response('<html><h1>Bad Gateway</h1></html>', { status: 502 }),
  'an incomplete template': ok(BASIC.replace(/id=["']?dayahead-optim-basic["']?/, 'id=renamed')),
};
const alertText = s => s.registry['alert-text'].textContent;

test('first load shows the Basic view with its button wired and nothing stored', async () => {
  const s = setup({ replies: healthy }); await s.load();
  assert.equal(s.container.firstElementChild.id, 'basic');
  assert.equal(typeof s.registry['dayahead-optim-basic'].listeners.click, 'function');
  assert.notEqual(s.registry.alert.style.display, 'block'); assert.equal(s.storage.TabSelection, undefined); s.done();
});
test('stored Advanced selection loads the Advanced view with its controls wired', async () => {
  const s = setup({ stored: 'advanced', replies: healthy }); await s.load();
  assert.equal(s.container.firstElementChild.id, 'advance');
  for (const id of ['dayahead-optim', 'publish-data', 'input-plus', 'input-clear']) assert.equal(typeof s.registry[id].listeners.click, 'function', id);
  s.done();
});
test('a /get-config request that never completes does not delay the Advanced view setup', async () => {
  const s = setup({ stored: 'advanced', replies: { ...healthy, '/get-config': () => new Promise(() => {}) } });
  let restored = false; s.ctx.getSavedData = () => { restored = true; };
  const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error('view setup waited for /get-config')), 2000).unref());
  await Promise.race([s.load(), timeout]);
  assert.equal(restored, true);
  assert.equal(typeof s.registry.basicOrAdvanced.listeners.click, 'function'); s.done();
});
test('switching Advanced to Basic and back replaces the view and remembers the selection', async () => {
  const s = setup({ stored: 'advanced', replies: healthy }); await s.load();
  await s.toggle(); assert.equal(s.container.firstElementChild.id, 'basic'); assert.equal(s.storage.TabSelection, 'basic');
  await s.toggle(); assert.equal(s.container.firstElementChild.id, 'advance'); assert.equal(s.storage.TabSelection, 'advanced');
  assert.equal(s.rejections.length, 0); s.done();
});
for (const [kind, reply] of Object.entries(failures)) {
  test(`switch failing with ${kind} keeps the working view, its controls and the stored selection`, async () => {
    const s = setup({ stored: 'advanced', replies: { ...healthy, 'basic.html': reply } }); await s.load();
    const before = s.container.innerHTML, button = s.registry['dayahead-optim'];
    await s.toggle();
    assert.equal(s.container.innerHTML, before); assert.equal(s.registry['dayahead-optim'], button);
    assert.equal(typeof button.listeners.click, 'function');
    assert.equal(s.storage.TabSelection, 'advanced');
    assert.equal(s.registry.alert.style.display, 'block'); assert.notEqual(alertText(s), '');
    assert.equal(s.rejections.length, 0); s.done();
  });
  test(`first load failing with ${kind} is reported in the alert box and can be retried`, async () => {
    const s = setup({ replies: { ...healthy, 'basic.html': reply } });
    await s.load();
    assert.equal(s.container.innerHTML, ''); assert.equal(s.registry.alert.style.display, 'block');
    assert.notEqual(alertText(s), ''); assert.equal(s.rejections.length, 0);
    s.ctx.fetch = async (url) => healthy[url.replace('static/', '')]();
    await s.toggle();
    assert.equal(s.container.firstElementChild.id, 'basic');
    assert.equal(s.registry.alert.style.display, 'none', 'the failure message is cleared once the view loads'); s.done();
  });
}
for (const [kind, reply] of Object.entries({
  'a rejected request': () => { throw new TypeError('Failed to fetch'); },
  'a non-JSON 200 body': () => new Response('<html>not json</html>', { status: 200 }),
})) {
  test(`the export-section check failing with ${kind} neither rejects nor breaks the Advanced view`, async () => {
    const s = setup({ stored: 'advanced', replies: { ...healthy, '/get-config': reply } });
    await s.load();
    await new Promise(r => setImmediate(r)); // let any unhandled rejection surface
    assert.equal(s.rejections.length, 0);
    assert.equal(s.container.firstElementChild.id, 'advance');
    assert.equal(typeof s.registry['dayahead-optim'].listeners.click, 'function');
    assert.notEqual(s.registry.alert.style.display, 'block'); s.done();
  });
}
test('a successful view load keeps the error of an unrelated action visible', async () => {
  const s = setup({ replies: healthy }); await s.load();
  s.registry['alert-text'].textContent = 'optimisation failed'; s.registry.alert.style.display = 'block';
  await s.toggle();
  assert.equal(s.container.firstElementChild.id, 'advance');
  assert.equal(s.registry.alert.style.display, 'block'); assert.match(alertText(s), /optimisation failed/); s.done();
});
for (const [view, html, hidden, pattern] of [
  ['Advanced', () => ADVANCED, 'input-container', /missing input-container\)/],
  ['Advanced', () => ADVANCED, 'loader', /missing loader\)/],
  ['Basic', () => BASIC, 'loader', /missing loader\)/],
  ['Advanced', () => ADVANCED, 'advance', /missing advance[,)]/],
  ['Basic', () => BASIC, 'basic', /missing basic[,)]/],
]) {
  test(`${view} template without #${hidden} is rejected before it replaces the working view`, async () => {
    const broken = html().replace(new RegExp(`id=["']?${hidden}["']?(?=[\\s>])`), 'id=renamed');
    const other = view === 'Advanced' ? 'basic' : 'advanced';
    const s = setup({ stored: other, replies: { ...healthy, [`${view.toLowerCase()}.html`]: ok(broken) } });
    await s.load();
    const before = s.container.innerHTML, working = s.container.firstElementChild.id;
    await s.registry.basicOrAdvanced.listeners.click();
    assert.equal(s.container.innerHTML, before); assert.equal(s.container.firstElementChild.id, working);
    assert.equal(s.storage.TabSelection, other);
    assert.match(alertText(s), pattern); assert.equal(s.registry.alert.style.display, 'block'); s.done();
  });
}
test('invalid saved runtime data does not undo a successful switch to Advanced', async () => {
  const s = setup({ replies: healthy, savedInputs: '{not json', realSavedData: true });
  await s.load(); await s.toggle();
  assert.equal(s.container.firstElementChild.id, 'advance'); assert.equal(s.storage.TabSelection, 'advanced');
  assert.notEqual(s.registry.alert.style.display, 'block', 'no view-loading error for a view that did load');
  assert.equal(s.rejections.length, 0); s.done();
});
test('only the newest InfluxDB check may update the export section', async () => {
  const s = setup({ stored: 'advanced', replies: healthy }); await s.load();
  const pending = [];
  s.ctx.fetch = () => new Promise(resolve => pending.push(resolve));
  const first = s.ctx.checkInfluxDBAndShowExport(), second = s.ctx.checkInfluxDBAndShowExport();
  const reply = body => new Response(JSON.stringify(body), { status: 200 });
  pending[1](reply({})); await second;                                   // newer: not configured
  pending[0](reply({ influxdb_host: 'db', influxdb_port: 8086 })); await first; // older, resolves last
  assert.equal(s.registry['export-influxdb-section'].style.display, 'none'); s.done();
});
test('an incomplete Advanced template keeps the working Basic view', async () => {
  const incomplete = ADVANCED.replace(/id=["']?dayahead-optim["']?(?=[\s>])/, 'id=renamed');
  const s = setup({ replies: { ...healthy, 'advanced.html': ok(incomplete) } });
  await s.load();
  const before = s.container.innerHTML;
  await s.toggle();
  assert.equal(s.container.innerHTML, before);
  assert.equal(s.storage.TabSelection, undefined);
  assert.match(alertText(s), /advanced\.html is incomplete \(missing dayahead-optim\)/);
  assert.equal(s.rejections.length, 0); s.done();
});
test('a normal Advanced load requests /get-config for the export section and restores saved inputs', async () => {
  const s = setup({ stored: 'advanced', replies: healthy });
  let restored = false; s.ctx.getSavedData = () => { restored = true; };
  const requested = [];
  s.ctx.fetch = async (url) => { requested.push(url); return healthy[url.replace('static/', '')](); };
  await s.load();
  assert.ok(requested.includes('/get-config'));
  assert.equal(restored, true);
  s.done();
});
test('the alert names the HTTP status of a failed template request', async () => {
  const s = setup({ stored: 'advanced', replies: { ...healthy, 'basic.html': failures['an HTTP 502 error page'] } });
  await s.load(); await s.toggle(); assert.match(alertText(s), /HTTP 502/); s.done();
});
