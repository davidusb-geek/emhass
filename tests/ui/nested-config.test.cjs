// Run from the repository root: node --test tests/ui/nested-config.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/emhass/static/configuration_script.js', 'utf8');
const name = 'battery_charge_power_derating';
const definition = { input: 'array.array.float', default_value: [], friendly_name: 'Charge power derating', Description: 'Test' };
function setup(raw) {
  const input = { type: 'text', value: raw };
  const ctx = { window: {}, console, document: {
    getElementsByClassName: () => [{}],
    getElementById: id => id === name ? { tagName: 'DIV', getElementsByClassName: () => [input] } : null,
  }, fetch: async (url, options) => { ctx.sent = JSON.parse(options.body); return { status: 200, json: async () => [] }; } };
  vm.createContext(ctx); vm.runInContext(source, ctx);
  ctx.showChangeStatus = () => {}; ctx.errorAlert = msg => { ctx.error = msg; };
  return ctx;
}
for (const value of [[], [[0.5,1000],[0.9,200]], [[[0.5,1000]],[[0.8,500]]]]) {
  test(`render/save preserves ${JSON.stringify(value)}`, async () => {
    const ctx = setup('');
    const html = ctx.buildParamElement(definition, name, { [name]: value });
    assert.equal((html.match(/<input /g)||[]).length, 1);
    const raw = html.match(/value="([^"]*)"/)[1];
    assert.deepEqual(JSON.parse(raw), value);
    const save = setup(raw);
    await save.saveConfiguration({ Battery: { [name]: definition } });
    assert.deepEqual(save.sent[name], value);
  });
}
test('blank clears table to []', async () => {
  const ctx = setup('  '); await ctx.saveConfiguration({ Battery: { [name]: definition } });
  assert.deepEqual(ctx.sent[name], []);
});
for (const raw of ['[','[0.5,1000]','["[0.5,1000]"]','[["0.5",1000]]','null','{}','[[0.5,1e999]]']) {
  test(`invalid nested numeric JSON blocked: ${raw}`, async () => {
    const ctx = setup(raw); assert.equal(await ctx.saveConfiguration({ Battery: { [name]: definition } }), 0);
    assert.equal(ctx.sent, undefined); assert.match(ctx.error, /battery_charge_power_derating/);
  });
}
test('HTML characters escaped in existing malformed data', () => {
  const ctx = setup(''); const html = ctx.buildParamElement(definition, name, { [name]: '\"><img src=x>' });
  assert.equal(html.includes('<img'), false); assert.match(html, /&quot;/);
});
test('flat numeric arrays retain existing save behaviour', async () => {
  const ctx = setup('2'); await ctx.saveConfiguration({ Battery: { [name]: { input: 'array.string' } } });
  assert.deepEqual(ctx.sent[name], ['2']);
});
