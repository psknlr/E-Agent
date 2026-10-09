const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
if (!globalThis.crypto) globalThis.crypto = require('node:crypto').webcrypto;
const api = require('../web/reference-tools.js');
const bundle = JSON.parse(fs.readFileSync(path.join(__dirname, '../web/reference-data.json'), 'utf8'));
const tools = () => api.create(bundle);
const cite = (metadata, field, changes = {}) => {
  const value = {...metadata, field, ...changes};
  return `[cite artifact=${value.artifact} sha256=${value.sha256} row=${value.row} field=${value.field} method=${value.method}]`;
};

test('classic browser script exposes the same tools and original prompt', () => {
  const context = vm.createContext({crypto: globalThis.crypto, TextEncoder});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../web/reference-tools.js'), 'utf8'), context);
  const runtime = context.EAgentBrowserTools.create(bundle);
  assert.equal(runtime.systemPrompt, bundle.system_prompt);
  assert.deepEqual([...runtime.toolNames()], [...bundle.schemas.map(schema => schema.name), 'calculate']);
});

test('every exported local query returns its actual loader-vetted value or refusal', async () => {
  const runtime = tools();
  for (const [name, queries] of Object.entries(bundle.results)) {
    for (const [arguments, expected] of Object.entries(queries)) {
      const actual = await runtime.execute({interface: name, arguments: JSON.parse(arguments)});
      assert.equal(actual.ok, expected.ok);
      assert.deepEqual(actual.value, expected.value);
      assert.equal(actual.refusal, expected.refusal);
    }
  }
});

test('invalid IDs, arguments and prototype names refuse without modifying references', async () => {
  const runtime = tools();
  for (const call of [
    {interface: 'not_a_tool'}, {interface: 'kinetic_record', arguments: {label_id: 'missing'}},
    {interface: 'kinetic_record', arguments: {label_id: 4}},
    {interface: 'reference_summary', arguments: {extra: 'argument'}},
    {interface: 'activity_endpoint', arguments: {enzyme_id: 'Ssal-KRED', substrate_id: 'unknown'}},
    {interface: '__proto__'}, {interface: 'calculate', arguments: JSON.parse('{"__proto__":"bad"}')}
  ]) assert.equal((await runtime.execute(call)).ok, false);
  const first = await runtime.execute({interface: 'reference_summary'});
  first.value.counts.fake = 99;
  assert.equal((await runtime.execute({interface: 'reference_summary'})).value.counts.fake, undefined);
});

test('censored and insoluble endpoints retain their nulls and actual refusals', async () => {
  const runtime = tools();
  for (const status of ['below_detection', 'not_assayed']) {
    const [args] = Object.entries(bundle.results.activity_endpoint).find(([, result]) => result.value.status === status);
    const result = await runtime.execute({interface: 'activity_endpoint', arguments: JSON.parse(args)});
    assert.equal(result.value.total_product_mM, null);
    assert.ok(result.value.refusal);
    assert.equal(runtime.inspect('0 mM ' + cite(result.value.cite, 'total_product_mM')).clean, false);
    assert.equal(runtime.inspect('0 mM ' + cite(result.value.cite, 'product_r_mM_as_printed')).clean, false);
  }
});

test('bounded arithmetic computes precedence, scientific notation and approved functions', async () => {
  const runtime = tools();
  for (const [expression, expected] of [
    ['2 + 3 * 4', 14], ['(2 + 3) * 4', 20], ['-2^2', -4], ['2^3^2', 512],
    ['2^-2', 0.25], ['sqrt(9) + abs(-2)', 5], ['max(1, 4, 2) / 2', 2], ['1e-3 * 1000', 1]
  ]) {
    const result = await runtime.execute({interface: 'calculate', arguments: {expression}});
    assert.equal(result.ok, true, result.refusal);
    assert.equal(result.value.value, expected);
    assert.match(result.value.cite.sha256, /^[0-9a-f]{64}$/);
    const report = runtime.inspect(`${expected} ${cite(result.value.cite, 'value')}`);
    assert.equal(report.clean, true, report.summary);
    assert.equal(report.verified_quantities.length, 1);
    assert.equal(runtime.inspect(`${expected + 1} ${cite(result.value.cite, 'value')}`).clean, false);
  }
  const small = await runtime.execute({interface: 'calculate', arguments: {expression: '1e-3'}});
  assert.equal(runtime.inspect('1e-3 mM ' + cite(small.value.cite, 'value')).clean, true);
  assert.equal(runtime.inspect('1e-3 ' + cite(small.value.cite, 'value')).clean, true);
  assert.equal(runtime.inspect('1e-2 mM ' + cite(small.value.cite, 'value')).clean, false);
  assert.equal(runtime.inspect('1e-2 ' + cite(small.value.cite, 'value')).clean, false);
});

test('arithmetic refuses code execution, non-finite values and excessive work', async () => {
  const runtime = tools();
  for (const expression of [
    'globalThis.process.exit()', 'constructor.constructor("return process")()', 'Math.random()',
    'fetch("https://example.com")', '1;alert(1)', 'a=2', '[1,2]', '1/0', '0%0',
    'sqrt(-1)', 'exp(1000)', '1e101', 'pow(2)', 'min()', '2^^3', '1+2 trailing',
    '('.repeat(34) + '1' + ')'.repeat(34), '1+'.repeat(100) + '1'
  ]) assert.equal((await runtime.execute({interface: 'calculate', arguments: {expression}})).ok, false, expression);
});

test('citation integrity checks hashes, fields, plain counts and every citation', async () => {
  const runtime = tools();
  const kinetic = await runtime.execute({interface: 'kinetic_record', arguments: {label_id: '6ZZO_PaHBDH_AAE'}});
  const value = kinetic.value.kcat;
  assert.equal(runtime.inspect(`${value} s-1 ${cite(kinetic.value.cite, 'kcat')}`).clean, true);
  assert.equal(runtime.inspect(`${value + 100} s-1 ${cite(kinetic.value.cite, 'kcat')}`).clean, false);
  assert.equal(runtime.inspect(`${value} s-1 ${cite(kinetic.value.cite, 'kcat', {sha256: 'a'.repeat(64)})}`).clean, false);
  assert.equal(runtime.inspect(`Source ${cite(kinetic.value.cite, 'not_a_field')}`).clean, false);
  assert.equal(runtime.inspect(`Source ${cite(kinetic.value.cite, 'source_id')}`).clean, true);
  assert.equal(runtime.inspect(`Source ${cite(kinetic.value.cite, 'kcat', {row: 'missing'})}`).clean, false);
  assert.equal(runtime.inspect('Source [cite]').clean, false);
  const summary = await runtime.execute({interface: 'reference_summary'});
  const [field, number] = Object.entries(bundle.citation_rows.kred_reference.summary.fields).find(([, value]) => typeof value === 'number');
  assert.equal(runtime.inspect(`${number} records ${cite(summary.value.cite, field)}`).clean, true);
  assert.equal(runtime.inspect(`${number + 1} records ${cite(summary.value.cite, field)}`).clean, false);
  const [insolubleArgs] = Object.entries(bundle.results.activity_endpoint).find(([, result]) => result.value.status === 'not_assayed');
  const insoluble = await runtime.execute({interface: 'activity_endpoint', arguments: JSON.parse(insolubleArgs)});
  assert.equal(runtime.inspect(`${insoluble.value.enzyme_id} is insoluble ${cite(insoluble.value.cite, 'solubly_expressed')}`).clean, true);
  assert.equal(runtime.inspect(`Model round 2 reported an insoluble construct ${cite(insoluble.value.cite, 'status')}`).clean, true);
});

test('numeric guard preserves adjacency and blocks uncited measurements and fabricated derivations', async () => {
  const runtime = tools();
  const result = await runtime.execute({interface: 'calculate', arguments: {expression: '3.5987'}});
  const citation = cite(result.value.cite, 'value');
  assert.equal(runtime.inspect('3.6 A ' + citation).clean, true);
  assert.equal(runtime.inspect('3.59 A ' + citation).clean, false);
  assert.equal(runtime.inspect('3.6 A and 42 pLDDT ' + citation).clean, false);
  assert.equal(runtime.inspect(citation + ' 3.6 A').clean, false);
  assert.equal(runtime.inspect('30 s-1').clean, false);
  for (const value of ['1e3 mM', '1e-3 mM', '.5 mM', '-.5 mM', '+.5 mM']) {
    assert.equal(runtime.inspect(value).clean, false, value);
  }
  assert.equal(runtime.inspect('30 s-1 [artifact:fake]').clean, false);
  assert.equal(runtime.inspect('3.6 A ' + cite(result.value.cite, 'value', {method: 'derived:uncomputed'})).clean, false);
  assert.equal(tools().inspect('3.6 A ' + citation).clean, false);
});

test('citation rounding agrees with Python ties-to-even on exact binary values', async () => {
  const runtime = tools();
  for (const [expression, expected, wrong] of [['2.5', '2', '3'], ['1.25', '1.2', '1.3'], ['2.675', '2.67', '2.68']]) {
    const result = await runtime.execute({interface: 'calculate', arguments: {expression}});
    const citation = cite(result.value.cite, 'value', {method: 'rounded'});
    assert.equal(runtime.inspect(expected + ' mM ' + citation).clean, true, expression);
    assert.equal(runtime.inspect(wrong + ' mM ' + citation).clean, false, expression);
  }
});
