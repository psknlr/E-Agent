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

// ---- routing a question to the stored tools without a model -----------------
// This is a keyword lookup, so what matters is that it never offers a tool call
// the data cannot serve, that it explains each call, and that it stays quiet
// when it does not understand rather than guessing.
const planOf = (text) => tools().plan(text);
const names = (plan) => plan.calls.map(call => call.interface);

test('the three starter questions route to the tools a person would pick, each with a reason', () => {
  const structures = planOf('List the audited structures and inspect 6ZZO. What evidence and limitations are recorded for its cofactor and ligand?');
  assert.deepEqual(structures.calls.map(call => [call.interface, call.arguments]), [['list_structure_entries', {}], ['structure_entry', {pdb_id: '6ZZO'}]]);
  const sum = planOf('Use the browser calculator to calculate 2 * 3 + 4, then explain the result and show the calculation in the tool trace.');
  assert.deepEqual(sum.calls.map(call => [call.interface, call.arguments]), [['calculate', {expression: '2 * 3 + 4'}]]);
  const activity = planOf('Inspect Ssal-KRED activity on 2a. Explain missing values and detection limits before comparing it with other constructs.');
  assert.deepEqual(activity.calls.map(call => [call.interface, call.arguments]), [['activity_endpoint', {enzyme_id: 'Ssal-KRED', substrate_id: '2a'}]]);
  for (const call of [...structures.calls, ...sum.calls, ...activity.calls]) assert.ok(call.rationale.length > 10, call.interface + ' needs a stated reason');
});

test('every id the router can offer executes, and a prompt naming one plans exactly that call', async () => {
  const runtime = tools(), known = runtime.ids();
  assert.equal(known.pdb_ids.length, 19);
  assert.equal(known.label_ids.length, 28);
  assert.equal(known.enzyme_ids.length * known.substrate_ids.length, Object.keys(bundle.results.activity_endpoint).length);
  for (const id of known.pdb_ids) {
    for (const prompt of [id, id.toLowerCase(), `inspect ${id}, please`]) {
      const plan = runtime.plan(prompt);
      assert.deepEqual(plan.calls.map(call => call.arguments), [{pdb_id: id}], prompt);
      assert.equal((await runtime.execute(plan.calls[0])).ok, true, prompt);
    }
  }
  for (const id of known.label_ids) {
    const plan = runtime.plan(`record ${id}`);
    assert.deepEqual(plan.calls.map(call => [call.interface, call.arguments]), [['kinetic_record', {label_id: id}]], id);
    assert.equal((await runtime.execute(plan.calls[0])).ok, true, id);
  }
  for (const enzyme of known.enzyme_ids) {
    for (const substrate of known.substrate_ids) {
      const plan = runtime.plan(`${enzyme} activity on ${substrate}`);
      assert.deepEqual(plan.calls.map(call => call.arguments), [{enzyme_id: enzyme, substrate_id: substrate}], enzyme + substrate);
      assert.equal((await runtime.execute(plan.calls[0])).ok, true, enzyme + substrate);
    }
  }
});

test('arithmetic is taken from a question only when it is plausibly arithmetic', () => {
  for (const [prompt, expression] of [
    ['calculate (2.5 + 3.5) * 4', '(2.5 + 3.5) * 4'], ['2 * 3 + 4', '2 * 3 + 4'], ['2+2', '2+2'], ['what is 7 - 2?', '7 - 2'],
    ['sqrt(16) + 2', 'sqrt(16) + 2'], ['please compute 2 ^ 10 and explain', '2 ^ 10'], ['calculate 2 * 3 + 4, then explain', '2 * 3 + 4'],
  ]) assert.deepEqual(planOf(prompt).calls.filter(call => call.interface === 'calculate').map(call => call.arguments.expression), [expression], prompt);
  for (const prompt of ['2024-10-10', '10/3/2026', 'kcat 12.5 s-1', 'inspect 6ZZO', 'Ssal-KRED activity on 2a', 'Ort-EZM-1 on 1a', 'api * 2 spin', 'compute 1 +', 'list 1, 2, 3']) {
    assert.equal(names(planOf(prompt)).includes('calculate'), false, prompt);
  }
});

test('the router asks for nothing it cannot serve and says so when it understood nothing', () => {
  for (const prompt of ['9XYZ', '2024', '__proto__ constructor toString', '   ', 'tell me about enzymes', 'Why is the sky blue?']) {
    const plan = planOf(prompt);
    assert.deepEqual(plan.calls, [], prompt);
    assert.equal(plan.notes.length, 1, prompt);
  }
  const help = planOf('tell me about enzymes').notes[0], known = tools().ids();
  assert.match(help, /could not match/);
  assert.match(help, /does not interpret free text/);
  assert.ok(help.includes(known.pdb_ids[0]), 'the examples use ids that exist');
  assert.equal(planOf(undefined).calls.length, 0);
  assert.equal(planOf({}).calls.length, 0);
});

test('a broad question is bounded and the user is told it was clipped', () => {
  const plan = planOf('Ort-EZM-1 and Ort-EZM-2 activity, structures, kinetic records, audit verdict, summary, source verification');
  assert.equal(plan.calls.length, 8);
  assert.match(plan.notes[0], /Only the first 8 lookups were run/);
  assert.equal(new Set(plan.calls.map(call => call.interface + JSON.stringify(call.arguments))).size, plan.calls.length, 'no duplicate calls');
});

test('keywords pick the summary tools, and "audited" alone is not an audit-verdict request', () => {
  assert.deepEqual(names(planOf('what is the audit verdict?')), ['audit_verdict']);
  assert.deepEqual(names(planOf('overview of the reference set')), ['reference_summary']);
  assert.deepEqual(names(planOf('kinetic records in the core tier')), ['list_kinetic_records']);
  assert.deepEqual(planOf('kinetic records in the core tier').calls[0].arguments, {tier: 'core'});
  assert.deepEqual(names(planOf('are the numbers verified against their sources?')), ['source_verification']);
  assert.equal(names(planOf('list the audited structures')).includes('audit_verdict'), false);
  assert.deepEqual(names(planOf('how independent are the orthologs?')), ['activity_independence_groups']);
  assert.deepEqual(names(planOf('how independent is the set?')), ['reference_summary']);
});

test('ids() hands out copies, so a caller cannot change what the router will offer', () => {
  const runtime = tools();
  runtime.ids().pdb_ids.length = 0;
  runtime.ids().pdb_ids.push('0000');
  assert.equal(runtime.ids().pdb_ids.length, 19);
  assert.equal(runtime.ids().pdb_ids.includes('0000'), false);
});

// ---- identifiers are not measurements --------------------------------------
// The unit pattern is case-insensitive, which also made "A" (angstrom) and "M"
// (molar) match a lowercase letter, so the reference set's substrate ids 1a-5a
// read as "1 angstrom" and the guard refused every sentence that named one,
// including the one a model writes before it has called any tool. These cases
// mirror IdentifiersAreNotMeasurements in tests/test_numeric_guard.py.
const IDENTIFIER_SENTENCES = [
  'I will inspect Ssal-KRED activity on substrate 2a.', 'Checking substrates 1a, 3a and 5a next.',
  'Compare construct Ort-EZM-1 on 4a.', 'Buy 2 a day.', 'About 5 m away.',
];

test('a lowercase letter after a number is an identifier, not a unit', () => {
  const runtime = tools();
  for (const text of IDENTIFIER_SENTENCES) {
    const report = runtime.inspect(text);
    assert.deepEqual([...report.uncited_quantities], [], text);
    assert.equal(report.clean, true, text);
  }
});

test('the units themselves are still caught, in either case where more than one letter makes the unit', () => {
  const runtime = tools();
  for (const [text, expected] of [
    ['The resolution is 2.1 A.', ['2.1 A']], ['The distance is 3.5A.', ['3.5 A']], ['A 5 M solution.', ['5 M']],
    ['It is 10 nM.', ['10 nM']], ['It is 5 mM.', ['5 mM']], ['About 12 %.', ['12 %']], ['A 90 degrees turn.', ['90 degrees']],
    ['The resolution is 2.1 angstroms.', ['2.1 angstroms']], ['It is 10 NM.', ['10 NM']], ['A 5 Percent gain.', ['5 Percent']],
  ]) {
    assert.deepEqual([...runtime.inspect(text).uncited_quantities], expected, text);
  }
});

test('an identifier between a number and its citation does not steal the citation', async () => {
  const runtime = tools();
  const result = await runtime.execute({interface: 'calculate', arguments: {expression: '7.1974 / 2'}});
  const citation = cite(result.value.cite, 'value');
  const bound = runtime.inspect(`The distance is 3.6 A in substrate 2a ${citation}.`);
  assert.equal(bound.clean, true);
  assert.deepEqual([...bound.verified_quantities], ['3.6 A']);
  // A real quantity in between still does: the citation then belongs to the later number.
  assert.deepEqual([...runtime.inspect(`The distance is 3.6 A and 5 M ${citation}.`).uncited_quantities], ['3.6 A']);
});

test('the Ssal-KRED / 2a answer the user asked for passes when its numbers are cited, and only they are refused when not', async () => {
  const runtime = tools();
  const record = await runtime.execute({interface: 'activity_endpoint', arguments: {enzyme_id: 'Ssal-KRED', substrate_id: '2a'}});
  assert.equal(record.ok, true);
  const answer = field => cite(record.value.cite, field);
  const text = `Ssal-KRED on substrate 2a shows 12.5 ee ${answer('ee_reported')} with 7.2 mM total product ${answer('total_product_mM')}.`;
  const cited = runtime.inspect(text);
  assert.equal(cited.clean, true, JSON.stringify(cited));
  assert.deepEqual([...cited.verified_quantities], ['12.5 ee', '7.2 mM']);
  // Without the citations the guard must still refuse the real quantities, and not the id.
  const bare = runtime.inspect('Ssal-KRED on substrate 2a shows 12.5 ee with 7.2 mM total product.');
  assert.deepEqual([...bare.uncited_quantities], ['12.5 ee', '7.2 mM']);
});


// ---- units as they are actually written, and text a person reads differently ---
// The unit list had "A" for angstrom and "degrees" for an angle, which is not how
// either is written in a structure paper, and a percentage typed with a
// full-width sign (common in Chinese-language replies) was not seen at all.
// These mirror UnitsAsTheyAreActuallyWritten in tests/test_numeric_guard.py.
test('the angstrom sign, degree sign, micro sign, pct and per cent are units', () => {
  const runtime = tools();
  for (const [text, expected] of [
    ['The distance is 7.3 Å.', ['7.3 Å']], ['The distance is 7.3 Å.', ['7.3 Å']], ['The distance is 7.3Å.', ['7.3 Å']],
    ['The distance is 7.3 Å.', ['7.3 Å']], ['The angle is 104.5° at the vertex.', ['104.5 °']],
    ['The Km is 12.5 µM.', ['12.5 μM']], ['The Km is 12.5 μM.', ['12.5 μM']], ['Identity is 99.9 pct.', ['99.9 pct']],
    ['Identity is 99.9 per cent.', ['99.9 per cent']],
  ]) assert.deepEqual([...runtime.inspect(text).uncited_quantities], expected, text);
  assert.deepEqual([...runtime.inspect('Substrate 2å is not a distance.').uncited_quantities], [], 'a lowercase a-ring is still an identifier');
});

test('full-width and other-script digits are the number they spell', async () => {
  const runtime = tools();
  assert.deepEqual([...runtime.inspect('Identity is ９９．９％.').uncited_quantities], ['99.9 %']);
  assert.deepEqual([...runtime.inspect('Identity is ٩٩ %.').uncited_quantities], ['99 %']);
  assert.deepEqual([...runtime.inspect('Identity is ९९ %.').uncited_quantities], ['99 %']);
  assert.deepEqual([...runtime.inspect('Identity is 80-95 percent.').uncited_quantities], ['95 percent'], 'a hyphenated range holds a quantity');
  // The reader sees 3.6 Å and a citation; the guard sees the same.
  const result = await runtime.execute({interface: 'calculate', arguments: {expression: '7.1974 / 2'}});
  const citation = cite(result.value.cite, 'value');
  const right = runtime.inspect(`The distance is ３．６ Å ${citation}.`);
  assert.deepEqual([...right.verified_quantities], ['3.6 Å']);
  assert.equal(runtime.inspect(`The distance is ４．６ Å ${citation}.`).clean, false);
});

// ---- a number that no tool derived -----------------------------------------------
test('a calculation of a bare number restates it and verifies nothing', async () => {
  const runtime = tools();
  const bare = await runtime.execute({interface: 'calculate', arguments: {expression: '87.5'}});
  const report = runtime.inspect(`The identity is 87.5 % ${cite(bare.value.cite, 'value')}.`);
  assert.equal(report.clean, true, 'it is cited');
  assert.deepEqual([...report.verified_quantities], []);
  assert.deepEqual([...report.declared_quantities], ['87.5 %']);
  assert.equal(report.fully_verified, false);
  for (const expression of ['(87.5)', '+87.5', '-(-87.5)', '875e-1', 'pi']) {
    const result = await runtime.execute({interface: 'calculate', arguments: {expression}});
    const value = result.value.value;
    const again = runtime.inspect(`It is ${value} nM ${cite(result.value.cite, 'value')}.`);
    assert.deepEqual([...again.verified_quantities], [], expression);
  }
  // A derivation is checked: the number is what the arithmetic gives.
  const derived = await runtime.execute({interface: 'calculate', arguments: {expression: '350 / 4'}});
  const checked = runtime.inspect(`The identity is 87.5 % ${cite(derived.value.cite, 'value')}.`);
  assert.deepEqual([...checked.verified_quantities], ['87.5 %']);
  assert.deepEqual([...checked.declared_quantities], []);
  assert.equal(checked.fully_verified, true);
  assert.equal(runtime.inspect(`The identity is 88.5 % ${cite(derived.value.cite, 'value')}.`).clean, false);
});

test('a zero with a positive exponent says nothing about the value', async () => {
  const runtime = tools();
  const result = await runtime.execute({interface: 'calculate', arguments: {expression: '90.9 + 0'}});
  const citation = cite(result.value.cite, 'value');
  for (const written of ['0e5', '0E5', '0.0e3', '-0e2', '0e1']) {
    assert.equal(runtime.inspect(`Identity is ${written} percent ${citation}.`).clean, false, written);
  }
  // Scientific notation is still read at the precision it is written.
  assert.equal(runtime.inspect(`Identity is 9e1 percent ${citation}.`).clean, true);
  assert.equal(runtime.inspect(`Identity is 90.9 percent ${citation}.`).clean, true);
  assert.equal(runtime.inspect(`Identity is 0e0 percent ${citation}.`).clean, false);
});

// ---- whatever a model sends, the guard answers in time ---------------------------
test('the guard is linear in the length of a reply', async () => {
  const runtime = tools();
  const result = await runtime.execute({interface: 'calculate', arguments: {expression: '1 + 1'}});
  const real = cite(result.value.cite, 'value');
  const replies = {
    'a quantity every few characters, one citation at the end': '1 nM '.repeat(12000) + real,
    'openers that never close': '[cite '.repeat(12000),
    'one opener with a very long body, never closed': '[cite ' + 'a'.repeat(63000),
    'one opener with a very long body': '[cite a=' + 'a'.repeat(63000) + ']',
    'legacy openers that never close': '[artifact:'.repeat(6000),
    'many short citations': (`1 nM ${real} `).repeat(500),
    'digits of other scripts': '٩'.repeat(30000) + ' %',
  };
  for (const [label, reply] of Object.entries(replies)) {
    const started = Date.now();
    const report = runtime.inspect(reply);
    const took = Date.now() - started;
    assert.ok(took < 1500, `${label}: ${took} ms`);
    assert.equal(typeof report.clean, 'boolean', label);
  }
  const huge = runtime.inspect('[cite a=' + 'a'.repeat(5000) + ']');
  assert.match(huge.broken_citations.join(), /longer than any citation can be/);
});

// ---- whatever a model sends, the guard answers ---------------------------------------
test('a seeded fuzz of hostile text never makes the guard throw, hang, or contradict itself', async () => {
  const runtime = tools();
  const real = await runtime.execute({interface: 'calculate', arguments: {expression: '350 / 4'}});
  const citation = cite(real.value.cite, 'value');
  let state = 0x9e3779b9;
  const next = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  const pick = list => list[next() % list.length];
  const pieces = ['87.5', ' ', '%', ' percent', ' A', 'Å', 'Å', '°', ' nM', 'µM', ' mM', '1e5', '0e9', '-', '+', '.', ',', '[cite ', ']', '[', '=',
    'artifact=', 'sha256=', 'row=', 'field=', 'method=read', citation, '９９', '٩', '०', '①', '🧬', '\ud800', '\u0000', 'ee', 'pLDDT', ' s-1', 'x'.repeat(60), '\n',
    '[artifact:', '[cite artifact=browser_calculation', ' kcal/mol', '2a', '6ZZO', '​', '‮'];
  for (let round = 0; round < 3000; round += 1) {
    let text = '';
    for (let part = 0, parts = 1 + next() % 40; part < parts; part += 1) text += pick(pieces);
    const started = Date.now();
    const report = runtime.inspect(text);
    assert.ok(Date.now() - started < 500, `took too long on ${JSON.stringify(text.slice(0, 80))}`);
    assert.equal(report.clean, report.uncited_quantities.length === 0 && report.broken_citations.length === 0);
    assert.equal(report.fully_verified, report.clean && report.declared_quantities.length === 0);
    for (const list of [report.uncited_quantities, report.cited_quantities, report.verified_quantities, report.declared_quantities, report.broken_citations]) {
      assert.ok(Array.isArray(list) && list.every(item => typeof item === 'string'));
    }
    // A number is verified or declared only if it was cited.
    for (const token of [...report.verified_quantities, ...report.declared_quantities]) assert.ok(report.cited_quantities.includes(token), token);
  }
});
