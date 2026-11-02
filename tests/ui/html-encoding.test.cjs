// Run from the repository root: node --test tests/ui/html-encoding.test.cjs
// controlsOf() models the value sanitization a browser applies to parsed form controls (text inputs drop
// CR/LF, time and number inputs reject invalid strings, an <option> without a value attribute trims and
// collapses its text, a <textarea> drops one leading newline), so tests assert the control's .value and the
// JSON sent by saveConfiguration(), not only the decoded attribute. See the calibration test below.
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
    if (element.tag === 'textarea') {
      const end = html.indexOf('</textarea>', i);
      element.text = decode(html.slice(i, end < 0 ? html.length : end));
      i = end < 0 ? html.length : end;
    } else if (element.tag === 'option') {
      const end = html.indexOf('<', i);
      element.text = decode(html.slice(i, end < 0 ? html.length : end));
    }
  }
  return elements;
}

function setup() {
  const ctx = { window: {}, console, AbortSignal, document: { getElementsByClassName: () => [], getElementById: () => null } };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  return ctx;
}


// ---- form-control model (HTML Standard value sanitization) --------------------------------------------
const TIME_RE = /^([01]\d|2[0-3]):[0-5]\d(:[0-5]\d(\.\d{1,3})?)?$/;
const NUMBER_RE = /^-?(\d+|\d*\.\d+)([eE][+-]?\d+)?$/;
const stripNewlines = s => s.replace(/[\r\n]/g, '');

function inputControl(e) {
  const type = e.attrs.type ?? 'text';
  let value = e.attrs.value ?? '';
  if (type === 'text') value = stripNewlines(value);
  else if (type === 'time') value = TIME_RE.test(value) ? value : '';
  else if (type === 'number') value = NUMBER_RE.test(value) ? value : '';
  else if (type === 'checkbox') value = e.attrs.value ?? 'on';
  return { tagName: 'INPUT', type, value, checked: 'checked' in e.attrs };
}

// Returns the form controls of a generated fragment, each with the .value / .checked a browser exposes.
function controlsOf(html) {
  const elements = parseElements(html.replace(/\r\n?/g, '\n'));  // input stream preprocessing
  const controls = [];
  for (let k = 0; k < elements.length; k++) {
    const e = elements[k];
    if (e.tag === 'input') controls.push(inputControl(e));
    else if (e.tag === 'textarea') {
      controls.push({ tagName: 'TEXTAREA', type: 'textarea', value: e.text.replace(/^\n/, ''), checked: false });
    } else if (e.tag === 'select') {
      const options = [];
      while (elements[k + 1]?.tag === 'option') {
        const o = elements[++k];
        options.push({ value: 'value' in o.attrs ? o.attrs.value : o.text.replace(/[\t\n\f\r ]+/g, ' ').trim(), selected: 'selected' in o.attrs });
      }
      const chosen = options.findLast(o => o.selected) ?? options[0];
      controls.push({ tagName: 'SELECT', type: 'select-one', value: chosen ? chosen.value : '', checked: false });
    }
  }
  return controls;
}

// A stand-in page for saveConfiguration(): each parameter container serves the controls of its markup.
function savePage(definitions, configs) {
  const posts = [];
  const node = () => ({ style: {}, innerHTML: '', textContent: '', classList: { add() {}, remove() {} } });
  const containers = {};
  const ctx = {
    window: {}, console: { ...console, debug() {}, log() {} }, setTimeout: () => 0, Response, AbortSignal,
    document: {
      getElementsByClassName: c => (c === 'section-card' && !ctx.box ? [{}] : []),
      getElementById: id => {
        if (id === 'config-box') return ctx.box ?? null;
        if (containers[id]) return containers[id];
        return ['alert', 'alert-text', 'loader'].includes(id) ? node() : null;
      },
    },
    fetch: async (url, options) => { posts.push(JSON.parse(options.body)); return new Response('[]', { status: 200 }); },
  };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  for (const [name, definition] of Object.entries(definitions)) {
    const container = {
      tagName: 'DIV', html: ctx.buildParamElement(definition, name, configs),
      getElementsByClassName(c) {
        if (c === 'param_input') return controlsOf(this.html);
        if (c === 'param-input') return [{ insertAdjacentHTML: (_, h) => { this.html += h; } }];
        return [];
      },
    };
    containers[name] = container;
  }
  const sections = { Section: definitions };
  return {
    ctx, containers, sections, posts,
    addRow: name => ctx.plusElements(name, sections, 'Section', {}),
    save: async () => { await ctx.saveConfiguration(sections); return posts.at(-1); },
  };
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
    elements.filter(e => e.tag === 'option').forEach(option => assert.deepEqual(Object.keys(option.attrs).filter(a => a !== 'selected' && a !== 'value'), []));
    assert.deepEqual(elements.filter(e => e.tag === 'option').map(o => o.attrs.value), ['first', marker, 'last']);
    const options = [...html.matchAll(/<option\b([^>]*)>([^<]*)<\/option>/g)];
    assert.equal(options.length, 3);
    assert.deepEqual(options.map(o => decode(o[2])), ['first', marker, 'last']);
    const selected = options.filter(o => /\bselected="selected"/.test(o[1])).map(o => decode(o[2]));
    assert.deepEqual(selected, marker === 'first' ? ['first'] : [marker]);
    // the browser-visible value of the select is the configured string, byte for byte (also what is saved)
    assert.equal(controlsOf(html)[0].value, marker);
  });

  test(`single-line input .value: only CR/LF are dropped, everything else is data: ${label}`, () => {
    const ctx = setup();
    const definition = { input: 'string', default_value: 'd', friendly_name: 'f', Description: 'd' };
    const [control, ...rest] = controlsOf(ctx.buildParamElement(definition, 'name', { name: marker }));
    assert.equal(rest.length, 0);
    assert.equal(control.value, stripNewlines(marker));
  });

  test(`string, array.string and select values reach the saved JSON: ${label}`, async () => {
    const definitions = {
      s: { input: 'string', default_value: 'd', friendly_name: 'f', Description: 'd' },
      a: { input: 'array.string', default_value: marker, friendly_name: 'f', Description: 'd' },
      sel: { input: 'select', select_options: ['first', marker, 'last'], default_value: 'first', friendly_name: 'f', Description: 'd' },
    };
    const page = savePage(definitions, { s: marker, a: [marker, 'second'], sel: marker });
    page.addRow('a');  // dynamically added row starts from the (escaped) definition default
    const saved = await page.save();
    assert.equal(saved.s, stripNewlines(marker));
    assert.deepEqual(saved.a, [stripNewlines(marker), 'second', stripNewlines(marker)]);
    assert.equal(saved.sel, marker);
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

test('an empty default no longer swallows the following value attribute', () => {
  const ctx = setup();
  const definition = { input: 'string', default_value: '', friendly_name: 'f', Description: 'd' };
  const [input] = inputsOf(ctx.buildParamElement(definition, 'name', { name: 'x' }));
  assert.equal(input.attrs.placeholder, '');
  assert.equal(input.attrs.value, 'x');
});

test('box view sets the textarea value, so character references are not decoded', async () => {
  // Assigning innerHTML to a textarea parses its text and decodes references such as &amp; or &lt.
  const ctx = setup();
  const box = { innerHTML: undefined, value: undefined };
  ctx.document.getElementById = id => (id === 'config-box' ? box : { innerHTML: '' });
  const config = { a: 'x &amp; y', url: 'https://h/?p=1&lt=2', q: '"<b>"' };
  await ctx.loadConfigurationBoxPage(config);
  assert.equal(box.innerHTML, undefined);
  assert.deepEqual(JSON.parse(box.value), config);
});

test('the tokenizer detects the reported unquoted-value leak', () => {
  const leaking = '<input class="param_input" type="text" placeholder=a value=x data-marker=1 y>';
  assert.ok('data-marker' in parseElements(leaking)[0].attrs);
  assert.equal(parseElements('<input value=a b><i>')[0].attrs.b, '');
  assert.deepEqual(parseElements('<input value="><b>"><i>').map(e => e.tag), ['input', 'i']);
});

test('model calibration: values a browser reports for the controls the script generates', () => {
  // Observed in a real DOM (jsdom 26) and specified by the HTML Standard; guards controlsOf() itself.
  const value = html => controlsOf(html)[0].value;
  assert.equal(value('<input type="text" value="a\nb">'), 'ab');
  assert.equal(value('<input type="text" value="a\r\nb">'), 'ab');
  assert.equal(value('<input type="text" value=" a  b\t&amp;lt; ">'), ' a  b\t&lt; ');
  assert.equal(value('<input type="time" value="02:54">'), '02:54');
  assert.equal(value('<input type="time" value="2:54">'), '');
  assert.equal(value('<input type="time" value="a\nb">'), '');
  assert.equal(value('<input type="number" value="1.5">'), '1.5');
  assert.equal(value('<input type="number" value="x">'), '');
  assert.equal(value('<select><option selected="selected"> a  b </option></select>'), 'a b');
  assert.equal(value('<select><option>x</option><option selected="selected" value=" a  b "> a  b </option></select>'), ' a  b ');
  assert.equal(value('<select><option>x</option><option>y</option></select>'), 'x');
  assert.equal(value('<textarea>\n{"a": 1}</textarea>'), '{"a": 1}');
  assert.equal(value('<textarea>a\r\nb</textarea>'), 'a\nb');
});

test('select options keep boundary and repeated whitespace in the selected value', async () => {
  const options = ['first', ' leading', 'trailing ', 'in  ner', 'tab\there', ' & <b> "q" '];
  const definition = { input: 'select', select_options: options, default_value: 'first', friendly_name: 'f', Description: 'd' };
  for (const option of options) {
    const page = savePage({ sel: definition }, { sel: option });
    assert.equal(controlsOf(page.containers.sel.html)[0].value, option);
    assert.equal((await page.save()).sel, option);
  }
});

test('single-line fields have no multiline contract: no string, time or select definition needs CR/LF', () => {
  // saveConfiguration reads input.value, which a browser strips of CR/LF for type=text (before and after #1291).
  // If a definition ever needs a multiline value it must render a <textarea>; this guard makes that visible.
  const definitions = JSON.parse(fs.readFileSync('src/emhass/static/data/param_definitions.json', 'utf8'));
  for (const [section, params] of Object.entries(definitions)) {
    for (const [name, definition] of Object.entries(params)) {
      if (!['string', 'array.string', 'time', 'array.time', 'select'].includes(definition.input)) continue;
      const strings = [definition.default_value, ...(definition.select_options ?? [])].filter(v => typeof v === 'string');
      for (const text of strings) assert.ok(!/[\r\n]/.test(text), `${section}/${name} contains a line break`);
    }
  }
});

test('numbers, booleans, checkboxes and valid times round trip through save', async () => {
  const definitions = {
    n: { input: 'float', default_value: 1.5, friendly_name: 'f', Description: 'd' },
    zero: { input: 'int', default_value: 3, friendly_name: 'f', Description: 'd' },
    ints: { input: 'array.int', default_value: 3, friendly_name: 'f', Description: 'd' },
    on: { input: 'boolean', default_value: false, friendly_name: 'f', Description: 'd' },
    off: { input: 'boolean', default_value: true, friendly_name: 'f', Description: 'd' },
    flags: { input: 'array.boolean', default_value: false, friendly_name: 'f', Description: 'd' },
    load_peak_hour_periods: { input: 'array.time', default_value: '00:00', friendly_name: 'f', Description: 'd' },
  };
  const periods = { period_hp_1: [{ start: '02:54' }, { end: '06:00' }], period_hp_2: [{ start: '17:30' }, { end: '23:59' }] };
  const config = { n: 0.25, zero: 0, ints: [1, 0, -2], on: true, off: false, flags: [true, false], load_peak_hour_periods: periods };
  assert.deepEqual(await savePage(definitions, config).save(), config);
});

test('a time input clears a non-time string: a native constraint, not a supported round trip', async () => {
  const definition = { input: 'array.time', default_value: '00:00', friendly_name: 'f', Description: 'd' };
  const periods = { period_hp_1: [{ start: 'not a time\n' }, { end: '2:5' }] };
  const page = savePage({ load_peak_hour_periods: definition }, { load_peak_hour_periods: periods });
  assert.deepEqual((await page.save()).load_peak_hour_periods, { period_hp_1: [{ start: '' }, { end: '' }] });
});

test('object textarea and numeric table JSON round trip through save', async () => {
  const value = { k: 'x &amp; y </textarea><b data-marker="1">', list: [1, 2, { nested: '"quoted"\nline' }], n: null };
  const table = [[0.5, 0.84], [1, 0.9]];
  const definitions = {
    obj: { input: 'object', default_value: null, friendly_name: 'f', Description: 'd' },
    objempty: { input: 'object', default_value: null, friendly_name: 'f', Description: 'd' },
    table: { input: 'array.array.float', default_value: [], friendly_name: 'f', Description: 'd' },
  };
  const config = { obj: value, objempty: null, table };
  assert.deepEqual(await savePage(definitions, config).save(), config);
});

test('JSON box view round trips configuration strings through the textarea value', async () => {
  const config = { a: 'x &amp; y', u: 'https://h/?p=1&lt=2', q: '"<b>"', m: 'two\nlines', s: ' padded ', n: 0, f: false, z: '', arr: ['&#39;', '日本'] };
  const page = savePage({}, {});
  const box = { value: undefined };
  page.ctx.box = box;
  page.ctx.document.getElementById = id => (id === 'config-box' ? box : id === 'configuration-container' ? { innerHTML: '' } : { style: {}, textContent: '', classList: { add() {}, remove() {} } });
  await page.ctx.loadConfigurationBoxPage(config);
  assert.deepEqual(JSON.parse(box.value), config);
  assert.deepEqual(await page.save(), config);
});
