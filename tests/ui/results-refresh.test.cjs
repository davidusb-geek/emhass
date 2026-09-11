// Run from repository root: node --test tests/ui/results-refresh.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/emhass/static/script.js', 'utf8');
function setup(templateReply) {
  const classes = new Set();
  const nodes = {
    loader: { innerHTML: '', classList: { add: v => classes.add(v), remove: v => classes.delete(v) } },
    alert: { style: {} }, 'alert-text': {},
    template: { innerHTML: 'previous results', getElementsByTagName: () => [] },
  };
  const calls = [];
  const ctx = { window: {}, Response, document: {
    getElementById: id => nodes[id], querySelectorAll: () => [],
  }, fetch: async (url) => {
    calls.push(url);
    if (url.startsWith('action/')) return new Response('Optimization completed', { status: 200 });
    return templateReply();
  } };
  vm.createContext(ctx); vm.runInContext(source, ctx);
  ctx.saveStorage = () => {}; ctx.inputToJson = () => ({});
  return { ctx, nodes, classes, calls };
}
test('success waits for the real template request before returning and showing tick', async () => {
  let release;
  const s = setup(() => new Promise(resolve => { release = resolve; }));
  let done = false;
  const action = s.ctx.formAction('dayahead-optim', 'basic').then(v => { done = true; return v; });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(done, false); assert.equal(s.classes.has('loading'), true);
  assert.doesNotMatch(s.nodes.loader.innerHTML, /tick/);
  release(new Response('<div>new results</div>', {status:200}));
  assert.equal(await action, true);
  assert.equal(s.nodes.template.innerHTML, '<div>new results</div>');
  assert.deepEqual(s.calls, ['action/dayahead-optim','template']);
  assert.equal(s.classes.has('loading'), false); assert.match(s.nodes.loader.innerHTML, /tick/);
});
for (const status of [400,401,404,500,503]) {
  test(`template HTTP ${status} rejects before replacing existing results`, async () => {
    const s = setup(() => new Response('Server error page', {status}));
    await assert.rejects(s.ctx.getTemplate(), new RegExp(`HTTP ${status}`));
    assert.equal(s.nodes.template.innerHTML, 'previous results');
  });
}
for (const kind of ['network', 'HTTP', 'body']) {
  test(`completed action with ${kind} refresh failure reports error, retains results, no resubmit`, async () => {
    const s = setup(async () => {
      if (kind === 'network') throw new Error('Offline');
      if (kind === 'HTTP') return new Response('Failure', {status:500});
      return {ok:true, blob: async () => {throw new Error('Body interrupted');}};
    });
    // Test the actual promise chain, not a stub of getTemplate/showChangeStatus.
    assert.equal(await s.ctx.formAction('dayahead-optim', 'advanced'), false);
    assert.equal(s.classes.has('loading'), false); assert.match(s.nodes.loader.innerHTML,/cross/);
    assert.match(s.nodes['alert-text'].textContent,/Action completed/);
    assert.equal(s.nodes.template.innerHTML,'previous results');
    assert.deepEqual(s.calls,['action/dayahead-optim','template']);
  });
}
test('retry after refresh error clears the previous error banner', async () => {
  let fail = true;
  const s = setup(() => new Response(fail ? 'Failure' : 'new results', {status: fail ? 500 : 200}));
  await s.ctx.formAction('dayahead-optim','basic');
  assert.equal(s.nodes.alert.style.display,'block'); fail = false;
  assert.equal(await s.ctx.formAction('dayahead-optim','basic'),true);
  assert.equal(s.nodes.alert.style.display,'none');
});
test('failed template body read preserves current results', async () => {
  const s = setup(() => ({ok:true,blob:async()=>{throw new Error('Read failed');}}));
  await assert.rejects(s.ctx.getTemplate(), /Read failed/);
  assert.equal(s.nodes.template.innerHTML,'previous results');
});
