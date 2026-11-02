// Run from the repository root: node --test tests/ui/html-encoding.test.cjs
// Regression tests for #1290: configuration-derived values must be rendered as data.
// The markup produced by buildParamElement() is read with a small tokenizer that follows the HTML
// start-tag attribute states (quoted, unquoted and unterminated values), so a value that leaks into
// the markup shows up as an extra attribute or element. Markers are inert text, nothing executes.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('src/emhass/static/configuration_script.js', 'utf8');

const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'" };
const decode = s => s.replace(/&(?:#(\d+)|#x([0-9a-f]+)|(amp|lt|gt|quot|apos));/gi,
  (_, dec, hex, name) => (name ? ENTITIES[name.toLowerCase()] : String.fromCodePoint(dec ? Number(dec) : parseInt(hex, 16))));

// Returns [{ tag, attrs: {name: value} }] for every start tag; textarea content is RCDATA (text only).
function parseElements(html) {
  const elements = [];
  let i = 0;
  while (i < html.length) {
    if (html[i] !== '<') { i++; continue; }
    if (html[i + 1] === '/') { i = html.indexOf('>', i); i = i < 0 ? html.length : i + 1; continue; }
    const start = /^<([a-zA-Z][^\s/>]*)/.exec(html.slice(i));
    if (!start) { i++; continue; }
    const element = { tag: start[1].toLowerCase(), attrs: {} };
    i += start[0].length;
    for (;;) {
      while (i < html.length && /[\s/]/.test(html[i])) i++;
      if (i >= html.length) break;
      if (html[i] === '>') { i++; break; }
      const name = /^[^\s/>=]+/.exec(html.slice(i))?.[0] ?? html[i];
      i += name.length;
      while (i < html.length && /\s/.test(html[i])) i++;
      let value = '';
      if (html[i] === '=') {
        i++;
        while (i < html.length && /\s/.test(html[i])) i++;
        if (html[i] === '"' || html[i] === "'") {
          const quote = html[i++];
          const end = html.indexOf(quote, i);
          value = html.slice(i, end < 0 ? html.length : end);
          i = end < 0 ? html.length : end + 1;
        } else {
          value = /^[^\s>]*/.exec(html.slice(i))[0];
          i += value.length;
        }
      }
      if (!(name.toLowerCase() in element.attrs)) element.attrs[name.toLowerCase()] = decode(value);
    }
    elements.push(element);
    if (element.tag === 'textarea') { const end = html.indexOf('</textarea>', i); i = end < 0 ? html.length : end; }
  }
  return elements;
}

function setup() {
  const ctx = { window: {}, console, AbortSignal, document: { getElementsByClassName: () => [], getElementById: () => null } };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  return ctx;
}

// Inert markers covering spaces, quotes, ampersands, angle brackets, entities, Unicode and control text.
const MARKERS = [
  'plain',
  'two words',
  'a" data-marker="1',
  "a' data-marker='1",
  'a data-marker=1',
  '"><b data-marker="1">x</b><i x="',
  '<marker-element data-marker="1">',
  '</select><marker-element>',
  '&amp; &lt;b&gt; &quot; &#39; &#x41; &unknown;',
  'Tom & Jerry <tag> "q" \'s\'',
  ' leading and trailing ',
  'tab\tnew\nline',
  'café 日本語 😀',
  '',
  '=',
  '/>',
];
const labelOf = marker => JSON.stringify(marker);

function inputsOf(html) {
  const elements = parseElements(html);
  assert.deepEqual([...new Set(elements.map(e => e.tag))].filter(t => t !== 'label' && t !== 'span'), elements.length ? ['input'] : []);
  return elements.filter(e => e.tag === 'input');
}
const allowed = ['class', 'type', 'placeholder', 'value', 'checked'];
function assertOnlyExpectedAttributes(input) {
  for (const name of Object.keys(input.attrs)) assert.ok(allowed.includes(name), `unexpected attribute ${name}`);
}

for (const marker of MARKERS) {
  const label = labelOf(marker);

  test(`scalar string value and placeholder stay data: ${label}`, () => {
    const ctx = setup();
    const definition = { input: 'string', default_value: marker, friendly_name: 'f', Description: 'd' };
    const inputs = inputsOf(ctx.buildParamElement(definition, 'name', { name: marker }));
    assert.equal(inputs.length, 1);
    assertOnlyExpectedAttributes(inputs[0]);
    assert.equal(inputs[0].attrs.value, marker);
    assert.equal(inputs[0].attrs.placeholder, marker);
    assert.equal(inputs[0].attrs.type, 'text');
  });

  test(`scalar default (no saved value) stays data: ${label}`, () => {
    const ctx = setup();
    const definition = { input: 'string', default_value: marker, friendly_name: 'f', Description: 'd' };
    const inputs = inputsOf(ctx.buildParamElement(definition, 'name', {}));
    assert.equal(inputs.length, 1);
    assertOnlyExpectedAttributes(inputs[0]);
    assert.equal(inputs[0].attrs.value, marker);
  });

  test(`null config value keeps an empty input and safe placeholder: ${label}`, () => {
    const ctx = setup();
    const definition = { input: 'string', default_value: marker, friendly_name: 'f', Description: 'd' };
    const inputs = inputsOf(ctx.buildParamElement(definition, 'name', { name: null }));
    assert.equal(inputs.length, 1);
    assertOnlyExpectedAttributes(inputs[0]);
    assert.equal(inputs[0].attrs.value, '');
    assert.equal(inputs[0].attrs.placeholder, marker);
  });

  test(`array.string values and placeholder stay data: ${label}`, () => {
    const ctx = setup();
    const definition = { input: 'array.string', default_value: marker, friendly_name: 'f', Description: 'd' };
    const values = [marker, 'second', marker];
    const inputs = inputsOf(ctx.buildParamElement(definition, 'name', { name: values }));
    assert.equal(inputs.length, 3);
    inputs.forEach((input, index) => {
      assertOnlyExpectedAttributes(input);
      assert.equal(input.attrs.value, values[index]);
      assert.equal(input.attrs.placeholder, marker);
    });
  });

  test(`nested array rows render one safe input per leaf: ${label}`, () => {
    // Arrays of arrays take the nested rendering path (one input per inner entry, first character of
    // string entries); only the markup safety and the entry count are asserted, not that quirk.
    const ctx = setup();
    const definition = { input: 'array.array.float', default_value: null, friendly_name: 'f', Description: 'd' };
    const inputs = inputsOf(ctx.buildParamElement(definition, 'name', { name: [[marker, 1], [2, marker]] }));
    assert.equal(inputs.length, 4);
    inputs.forEach(assertOnlyExpectedAttributes);
  });

  test(`load_peak_hour_periods nested values stay data: ${label}`, () => {
    const ctx = setup();
    const definition = { input: 'array.time', default_value: '00:00', friendly_name: 'f', Description: 'd' };
    const periods = { period_hp_1: [{ start: marker }, { end: marker }] };
    const inputs = inputsOf(ctx.buildParamElement(definition, 'load_peak_hour_periods', { load_peak_hour_periods: periods }));
    assert.equal(inputs.length, 2);
    for (const input of inputs) {
      assertOnlyExpectedAttributes(input);
      assert.equal(input.attrs.value, marker);
      assert.equal(input.attrs.placeholder, marker);
      assert.equal(input.attrs.type, 'time');
    }
  });

  test(`select option text and selection stay data: ${label}`, () => {
    const ctx = setup();
    const definition = { input: 'select', select_options: ['first', marker, 'last'], default_value: 'first', friendly_name: 'f', Description: 'd' };
    const html = ctx.buildParamElement(definition, 'name', { name: marker });
    const elements = parseElements(html);
    assert.deepEqual(elements.map(e => e.tag), ['select', 'option', 'option', 'option']);
    elements.filter(e => e.tag === 'option').forEach(option => assert.deepEqual(Object.keys(option.attrs).filter(a => a !== 'selected'), []));
    const options = [...html.matchAll(/<option\b([^>]*)>([^<]*)<\/option>/g)];
    assert.equal(options.length, 3);
    assert.deepEqual(options.map(o => decode(o[2])), ['first', marker, 'last']);
    const selected = options.filter(o => /\bselected="selected"/.test(o[1])).map(o => decode(o[2]));
    assert.deepEqual(selected, marker === 'first' ? ['first'] : [marker]);
  });

  test(`numeric table and object textarea stay data: ${label}`, () => {
    const ctx = setup();
    const table = { input: 'array.array.float', default_value: [], friendly_name: 'f', Description: 'd' };
    const tableInputs = inputsOf(ctx.buildParamElement(table, 'name', { name: marker }));
    assert.equal(tableInputs.length, 1);
    assertOnlyExpectedAttributes(tableInputs[0]);
    assert.equal(tableInputs[0].attrs.value, marker);
    const object = { input: 'object', default_value: { k: marker }, friendly_name: 'f', Description: 'd' };
    const html = ctx.buildParamElement(object, 'name', { name: { k: marker } });
    const elements = parseElements(html);
    assert.deepEqual(elements.map(e => e.tag), ['textarea']);
    assert.deepEqual(Object.keys(elements[0].attrs).sort(), ['class', 'placeholder', 'rows', 'spellcheck']);
    assert.deepEqual(JSON.parse(decode(/>([\s\S]*)<\/textarea>/.exec(html)[1])), { k: marker });
    assert.equal(elements[0].attrs.placeholder, JSON.stringify({ k: marker }));
  });
}

test('numbers, booleans and checkboxes keep their value and checked state', () => {
  const ctx = setup();
  const num = { input: 'float', default_value: 1.5, friendly_name: 'f', Description: 'd' };
  let [input] = inputsOf(ctx.buildParamElement(num, 'n', { n: 0 }));
  assert.equal(input.attrs.type, 'number');
  assert.equal(input.attrs.value, '0');
  assert.equal(input.attrs.placeholder, '1.5');
  [input] = inputsOf(ctx.buildParamElement(num, 'n', {}));
  assert.equal(input.attrs.value, '1.5');
  const flag = { input: 'boolean', default_value: false, friendly_name: 'f', Description: 'd' };
  [input] = inputsOf(ctx.buildParamElement(flag, 'b', { b: true }));
  assert.equal(input.attrs.type, 'checkbox');
  assert.equal(input.attrs.value, 'true');
  assert.ok('checked' in input.attrs);
  [input] = inputsOf(ctx.buildParamElement(flag, 'b', { b: false }));
  assert.equal(input.attrs.value, 'false');
  assert.ok(!('checked' in input.attrs));
  const flags = { input: 'array.boolean', default_value: false, friendly_name: 'f', Description: 'd' };
  const rows = inputsOf(ctx.buildParamElement(flags, 'b', { b: [true, false, true] }));
  assert.deepEqual(rows.map(r => 'checked' in r.attrs), [true, false, true]);
  const ints = { input: 'array.int', default_value: 3, friendly_name: 'f', Description: 'd' };
  assert.deepEqual(inputsOf(ctx.buildParamElement(ints, 'i', { i: [1, 0, -2] })).map(r => r.attrs.value), ['1', '0', '-2']);
});

test('an empty string value no longer swallows the following attribute', () => {
  const ctx = setup();
  const definition = { input: 'string', default_value: 'd', friendly_name: 'f', Description: 'd' };
  const [input] = inputsOf(ctx.buildParamElement(definition, 'name', { name: '' }));
  assert.equal(input.attrs.value, '');
  assert.equal(input.attrs.placeholder, 'd');
});

test('the tokenizer detects the reported unquoted-value leak', () => {
  const leaking = '<input class="param_input" type="text" placeholder=a value=x data-marker=1 y>';
  assert.ok('data-marker' in parseElements(leaking)[0].attrs);
  assert.equal(parseElements('<input value=a b><i>')[0].attrs.b, '');
  assert.deepEqual(parseElements('<input value="><b>"><i>').map(e => e.tag), ['input', 'i']);
});
