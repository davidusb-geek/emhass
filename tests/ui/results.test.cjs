// Run from the repository root: node --test tests/ui/results.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/emhass/static/script.js', 'utf8');
function setup(response, rejects = false) {
  const classes = new Set();
  const loader = { innerHTML: '', classList: { add: x => classes.add(x), remove: x => classes.delete(x) } };
  const alert = { style: {} }, text = {};
  const ctx = { fetchCalls: 0, window: {}, document: { getElementById: id => ({ loader, alert, 'alert-text': text }[id] || null) },
    fetch: async () => { ctx.fetchCalls++; if (rejects) throw new Error('Offline'); return response; } };
  vm.createContext(ctx); vm.runInContext(source, ctx);
  ctx.refreshed = 0; ctx.saved = 0;
  ctx.getTemplate = () => { ctx.refreshed++; };
  ctx.saveStorage = () => { ctx.saved++; };
  ctx.inputToJson = () => ({});
  return { ctx, loader, classes, text };
}
for (const status of [200, 201, 202, 204, 299]) {
  test(`HTTP ${status}: text/empty success clears spinner and refreshes`, async () => {
    let parsed = false;
    const s = setup({ status, ok: true, json: async () => { parsed = true; throw new SyntaxError('Not JSON'); } });
    assert.equal(await s.ctx.formAction('dayahead-optim', 'advanced'), true);
    assert.equal(parsed, false); assert.equal(s.classes.has('loading'), false);
    assert.match(s.loader.innerHTML, /tick/); assert.equal(s.ctx.refreshed, 1); assert.equal(s.ctx.saved, 1);
  });
}
for (const payload of [['Backend failure'], { message: 'failure' }, null, 'not-json']) {
  test(`HTTP error clears spinner: ${JSON.stringify(payload)}`, async () => {
    const s = setup({ status: 500, ok: false, json: async () => {
      if (payload === 'not-json') throw new SyntaxError('Not JSON'); return payload;
    } });
    assert.equal(await s.ctx.formAction('dayahead-optim', 'basic'), false);
    assert.equal(s.classes.has('loading'), false); assert.match(s.loader.innerHTML, /cross/);
    assert.equal(s.ctx.refreshed, 0); assert.equal(s.ctx.saved, 0);
  });
}
test('network rejection clears spinner and displays error', async () => {
  const s = setup(null, true);
  assert.equal(await s.ctx.formAction('dayahead-optim', 'basic'), false);
  assert.equal(s.classes.has('loading'), false); assert.match(s.text.textContent, /Offline/);
});
test('invalid runtime input does not submit', async () => {
  const s = setup(null); s.ctx.inputToJson = () => 0;
  assert.equal(await s.ctx.formAction('dayahead-optim', 'advanced'), false);
  assert.equal(s.ctx.fetchCalls, 0);
  assert.equal(s.classes.has('loading'), false);
});
