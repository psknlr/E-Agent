"use strict";

// One registry for the controller: reference agents and compute agents. The
// engine here is a scripted stand-in (the real one is exercised in
// engine-agents.test.cjs), so these tests pin the page's side of the contract:
// what is checked before a call, what is trusted after it, and what a model can
// and cannot make citable.

const test = require("node:test");
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const project = path.resolve(__dirname, "..");
const bundle = JSON.parse(fs.readFileSync(path.join(project, "web/reference-data.json"), "utf8"));
const catalog = JSON.parse(fs.readFileSync(path.join(project, "web/agent-catalog.json"), "utf8"));
const Tools = require(path.join(project, "web/reference-tools.js"));
const Agents = require(path.join(project, "web/agents.js"));

const plain = (value) => JSON.parse(JSON.stringify(value));
const sha = (text) => crypto.createHash("sha256").update(text).digest("hex");

// An engine that answers from a table. `reply(agent, value)` builds a result
// the way the real engine's envelope does (digest of the value, row from it).
function envelope(agent, value) {
  const digest = sha(JSON.stringify(value));
  return { ok: true, agent, value, cite: { artifact: "engine_compute", sha256: digest, row: `${agent}-${digest.slice(0, 16)}`, method: "read" }, wraps: "project.function", elapsed_ms: 1.5 };
}
const refusal = (agent, kind, message) => ({ ok: false, agent, kind, refusal: message });
function engineOf(answer, status = "ready") {
  const calls = [];
  return {
    calls,
    state: { status },
    call: async (agent, args) => {
      calls.push([agent, JSON.parse(args)]);
      const out = await answer(agent, JSON.parse(args));
      return typeof out === "string" ? out : JSON.stringify(out);
    },
  };
}
const ALIGN = { identity_percent_over_aligned_columns: 90.9090909090909, identity_over_aligned_columns: 0.9090909090909091, identical_columns: 20, aligned_pair_columns: 22,
  score: 38, note: "Identity is counted over aligned columns only.", aligned_a: "MKTAYIAKQRQISFVKSHFSRQ", aligned_b: "MKTAYLAKQRQVSFVKSHFSRQ" };
const standard = (agent) => agent === "align_sequences" ? envelope(agent, ALIGN) : refusal(agent, "invalid", "not scripted");
const composed = (engine = engineOf(standard)) => {
  const tools = Tools.create(bundle);
  return { tools, engine, agents: Agents.compose({ tools, engine, catalog }) };
};
const SEQ_A = "MKTAYIAKQRQISFVKSHFSRQ", SEQ_B = "MKTAYLAKQRQVSFVKSHFSRQ";
const call = (name, args = {}) => ({ interface: name, arguments: args, rationale: "test" });
const citation = (result, field) => `[cite artifact=engine_compute sha256=${result.value.cite.sha256} row=${result.value.cite.row} field=${field} method=read]`;

test("the catalog the page ships is well formed and cannot shadow a reference tool", () => {
  assert.doesNotThrow(() => Agents.checkCatalog(catalog));
  const reference = new Set(bundle.schemas.map((schema) => schema.name).concat("calculate"));
  for (const agent of catalog.agents) assert.ok(!reference.has(agent.name), agent.name);
  for (const mutate of [
    (c) => { c.schema = "x"; }, (c) => { c.artifact = "other"; }, (c) => { c.agents = []; }, (c) => { c.agents[0].name = "Bad Name"; },
    (c) => { c.agents.push(c.agents[0]); }, (c) => { c.agents[0].parameters = { a: 1 }; }, (c) => { c.agents[0].max_chars = { nope: 5 }; },
    (c) => { c.limits.max_arg_chars = 0; }, (c) => { c.tasks = [1]; }, (c) => { delete c.pipeline; },
  ]) {
    const copy = plain(catalog);
    mutate(copy);
    assert.throws(() => Agents.checkCatalog(copy), /agent catalog is invalid/);
  }
});

test("composing offers reference and compute agents under one registry, and refuses a collision", () => {
  const { agents, tools } = composed();
  const names = agents.toolNames();
  assert.deepEqual(names.slice(0, tools.toolNames().length), tools.toolNames());
  assert.deepEqual(names.slice(tools.toolNames().length), catalog.agents.map((agent) => agent.name));
  assert.deepEqual(agents.computeNames(), catalog.agents.map((agent) => agent.name));
  const offered = agents.schemas().filter((schema) => agents.computeNames().includes(schema.name));
  assert.equal(offered.length, catalog.agents.length);
  assert.ok(offered.every((schema) => /WebAssembly/.test(schema.runs) && schema.description && schema.parameters));
  assert.match(agents.systemPrompt, /COMPUTE AGENTS/);
  assert.match(agents.systemPrompt, /artifact=engine_compute/);
  assert.match(agents.systemPrompt, /Agents never approve anything/);
  assert.ok(agents.systemPrompt.startsWith(tools.systemPrompt), "the reference rules stay as written");
  const clash = plain(catalog);
  clash.agents[0].name = "reference_summary";
  assert.throws(() => Agents.compose({ tools: Tools.create(bundle), engine: engineOf(standard), catalog: clash }), /would shadow a reference tool/);
  assert.throws(() => Agents.compose({ tools: { execute() {}, toolNames: () => [] }, engine: engineOf(standard), catalog }), /cannot register computed results/);
  assert.throws(() => Agents.compose({ tools: Tools.create(bundle), engine: {}, catalog }), /incompatible/);
});

test("a compute call runs in the engine and comes back cited, shaped like every other result", async () => {
  const { agents, engine } = composed();
  const result = await agents.execute(call("align_sequences", { sequence_a: SEQ_A, sequence_b: SEQ_B }));
  assert.equal(result.ok, true);
  assert.equal(result.tool, "align_sequences");
  assert.equal(result.executor, "browser-wasm");
  assert.equal(result.wraps, catalog.agents.find((agent) => agent.name === "align_sequences").wraps);
  assert.equal(result.compute_ms, 1.5);
  assert.equal(result.truncated, false);
  assert.deepEqual(plain(result.arguments), { sequence_a: SEQ_A, sequence_b: SEQ_B });
  assert.equal(result.value.identical_columns, 20);
  assert.equal(result.value.cite.artifact, "engine_compute");
  assert.match(result.value.cite.sha256, /^[0-9a-f]{64}$/);
  assert.equal(result.value.cite.row, `align_sequences-${result.value.cite.sha256.slice(0, 16)}`);
  assert.deepEqual(engine.calls, [["align_sequences", { sequence_a: SEQ_A, sequence_b: SEQ_B }]]);
  // Reference tools still work, labelled as the page's own JavaScript.
  const summary = await agents.execute(call("reference_summary"));
  assert.equal(summary.ok, true);
  assert.equal(summary.executor, "browser-js");
  assert.equal(engine.calls.length, 1, "a reference lookup never reaches the engine");
});

test("arguments are checked before the engine is asked", async () => {
  const { agents, engine } = composed();
  const refused = async (args, pattern, name = "align_sequences") => {
    const result = await agents.execute(call(name, args));
    assert.equal(result.ok, false);
    assert.equal(result.value, null);
    assert.match(result.refusal, pattern);
    assert.equal(result.cite, undefined);
    return result;
  };
  await refused({ sequence_a: SEQ_A, bogus: "x" }, /unsupported argument/);
  await refused({ sequence_a: 5 }, /must be strings/);
  const limit = catalog.agents.find((agent) => agent.name === "align_sequences").max_chars.sequence_a;
  await refused({ sequence_a: "M".repeat(limit + 1) }, new RegExp(`exceeds its ${limit}-character limit`));
  await refused({ task: "x".repeat(4001) }, /exceeds its 4000-character limit/, "task_gate_report");
  await refused({ task_yaml: "x".repeat(6001) }, /exceeds its 6000-character limit/, "task_gate_report");
  await refused([], /must be an object/);
  assert.equal(engine.calls.length, 0, "a refused call costs the engine nothing");
  // A long task file is allowed by its own limit, which is larger than the default.
  const { agents: roomy, engine: spy } = composed(engineOf((agent) => envelope(agent, { ok: true })));
  assert.equal((await roomy.execute(call("task_gate_report", { task_yaml: "x".repeat(5000) }))).ok, true);
  assert.equal(spy.calls.length, 1);
});

test("a stopped engine, a failed engine and a refusing agent are each reported, never turned into a value", async () => {
  const down = composed(engineOf(standard, "idle"));
  const stopped = await down.agents.execute(call("align_sequences", { sequence_a: SEQ_A, sequence_b: SEQ_B }));
  assert.equal(stopped.ok, false);
  assert.match(stopped.refusal, /compute engine is not running/);
  assert.equal(down.engine.calls.length, 0);

  const failing = composed(engineOf(() => { const error = new Error("took too long"); error.kind = "timeout"; throw error; }));
  const failed = await failing.agents.execute(call("align_sequences", { sequence_a: SEQ_A, sequence_b: SEQ_B }));
  assert.match(failed.refusal, /compute engine failed: took too long/);

  const declines = composed(engineOf((agent) => refusal(agent, "refused", "sequence_b may contain only one-letter residue codes (ASCII letters)")));
  const declined = await declines.agents.execute(call("align_sequences", { sequence_a: SEQ_A, sequence_b: "MK1" }));
  assert.equal(declined.ok, false);
  assert.equal(declined.value, null);
  assert.equal(declined.refusal, "sequence_b may contain only one-letter residue codes (ASCII letters)");
  assert.equal(declined.engine_kind, "refused");
  assert.equal(declined.executor, "browser-wasm");
});

test("a malformed or forged engine reply is rejected rather than trusted", async () => {
  const good = envelope("align_sequences", ALIGN);
  const cases = [
    ["not JSON", () => "{oops"],
    ["a different agent", () => ({ ...good, agent: "wilson_interval" })],
    ["no citation", () => ({ ...good, cite: undefined })],
    ["another artifact", () => ({ ...good, cite: { ...good.cite, artifact: "kred_activity" } })],
    ["a short digest", () => ({ ...good, cite: { ...good.cite, sha256: "ab12" } })],
    ["a row that is not derived from the digest", () => ({ ...good, cite: { ...good.cite, row: "align_sequences-0000000000000000" } })],
    ["a method that is not read", () => ({ ...good, cite: { ...good.cite, method: "derived" } })],
    ["a value that already holds a cite", () => ({ ...good, value: { ...ALIGN, cite: { artifact: "kred_activity" } } })],
    ["a value that is not an object", () => ({ ...good, value: [1, 2] })],
    ["ok that is not a boolean", () => ({ ...good, ok: "yes" })],
  ];
  for (const [label, make] of cases) {
    const { agents } = composed(engineOf(make));
    const result = await agents.execute(call("align_sequences", { sequence_a: SEQ_A, sequence_b: SEQ_B }));
    assert.equal(result.ok, false, label);
    assert.equal(result.value, null, label);
    assert.match(result.refusal, /compute engine returned/, label);
  }
});

test("the numeric guard checks a computed number against the field it cites, and nothing else", async () => {
  const { agents } = composed();
  const result = await agents.execute(call("align_sequences", { sequence_a: SEQ_A, sequence_b: SEQ_B }));
  const ok = agents.inspect(`The two sequences are 90.9 percent identical ${citation(result, "identity_percent_over_aligned_columns")}.`);
  assert.equal(ok.clean, true, ok.summary);
  assert.deepEqual(ok.verified_quantities, ["90.9 percent"]);
  assert.equal(agents.inspect(`${20} identical columns ${citation(result, "identical_columns")}.`).clean, true, "a plain count is checked too");

  const wrong = agents.inspect(`They are 95.0 percent identical ${citation(result, "identity_percent_over_aligned_columns")}.`);
  assert.equal(wrong.clean, false);
  assert.match(wrong.broken_citations.join(), /contradicts the reference value/);

  const wrongField = agents.inspect(`They are 90.9 percent identical ${citation(result, "score")}.`);
  assert.equal(wrongField.clean, false, "the number is checked against the field named, not any field");

  const absent = agents.inspect(`They are 90.9 percent identical ${citation(result, "no_such_field")}.`);
  assert.match(absent.broken_citations.join(), /field absent/);

  const bare = agents.inspect("They are 90.9 percent identical.");
  assert.deepEqual(bare.uncited_quantities, ["90.9 percent"]);

  const forged = agents.inspect("90.9 percent [cite artifact=engine_compute sha256=" + "ab".repeat(32) + " row=align_sequences-deadbeefdeadbeef field=score method=read]");
  assert.match(forged.broken_citations.join(), /unknown artifact\/row/, "a row that was never computed cannot be cited");

  const stolen = agents.inspect(`90.9 percent [cite artifact=engine_compute sha256=${"ab".repeat(32)} row=${result.value.cite.row} field=identity_percent_over_aligned_columns method=read]`);
  assert.match(stolen.broken_citations.join(), /mismatched data digest/, "a real row with the wrong digest is not accepted");

  const other = composed();
  const fresh = other.agents.inspect(`90.9 percent ${citation(result, "identity_percent_over_aligned_columns")}`);
  assert.equal(fresh.clean, false, "rows belong to the run that computed them");
});

test("nested results are citable by path, and withheld values cannot license a number", async () => {
  const nested = { n_variants: 2, variants: [{ label: "K2A", position: 2, valid: true }, { label: "T3S", position: 3, valid: true }], missing: null, flag: false };
  const { agents } = composed(engineOf((agent) => envelope(agent, nested)));
  const result = await agents.execute(call("variant_check", { sequence: "MKTAYIAKQR", variants: "K2A, T3S" }));
  assert.equal(result.ok, true);
  assert.equal(agents.inspect(`position 3 ${citation(result, "variants.1.position")}`).clean, true);
  assert.equal(agents.inspect(`position 2 ${citation(result, "variants.1.position")}`).clean, false);
  assert.equal(agents.inspect(`2 records ${citation(result, "n_variants")}`).clean, true);
  assert.equal(agents.inspect(`7 records ${citation(result, "missing")}`).clean, false, "a null field backs no number");
  assert.equal(agents.inspect(`7 records ${citation(result, "flag")}`).clean, false, "nor does a boolean");
});

test("computed rows are registered only by the page, and only in a bounded, immutable form", () => {
  const tools = Tools.create(bundle);
  const row = { sha256: "a".repeat(64), fields: { x: 1.5, y: null }, field_names: ["x", "y"] };
  assert.doesNotThrow(() => tools.registerComputed("engine_compute", "agent-0123456789abcdef", row));
  assert.doesNotThrow(() => tools.registerComputed("engine_compute", "agent-0123456789abcdef", row), "the same result again is idempotent");
  assert.throws(() => tools.registerComputed("engine_compute", "agent-0123456789abcdef", { ...row, sha256: "b".repeat(64) }), /different result already holds/);
  assert.throws(() => tools.registerComputed("kred_activity", "x", row), /Not a computed artifact/);
  assert.throws(() => tools.registerComputed("engine_compute", "has space", row), /short identifier/);
  assert.throws(() => tools.registerComputed("engine_compute", "ok", { ...row, sha256: "short" }), /bounded finite numeric fields/);
  assert.throws(() => tools.registerComputed("engine_compute", "ok", { ...row, fields: { x: Infinity } }), /bounded finite numeric fields/);
  assert.throws(() => tools.registerComputed("engine_compute", "ok", { ...row, fields: { x: "1" } }), /bounded finite numeric fields/);
  assert.throws(() => tools.registerComputed("engine_compute", "ok", { ...row, field_names: [1] }), /bounded finite numeric fields/);
  for (let index = 0; index < 499; index += 1) tools.registerComputed("engine_compute", `row-${index}`, row);
  assert.throws(() => tools.registerComputed("engine_compute", "one-too-many", row), /Too many computed rows/);
  const hostile = plain(bundle);
  hostile.citation_rows.engine_compute = {};
  assert.throws(() => Tools.create(hostile), /may not define an artifact that is computed/);
});

test("results are flattened to dotted field paths that the citation grammar can carry", () => {
  assert.deepEqual(Agents.leaves({ a: 1, b: { c: "x", d: [2, { e: null }] }, cite: { row: "skip" }, empty: [], none: {} }),
    [["a", 1], ["b.c", "x"], ["b.d.0", 2], ["b.d.1.e", null]]);
  const row = Agents.rowFor({ a: 1.25, b: "text", c: true, d: [3, null], e: Number.NaN }, "f".repeat(64));
  assert.deepEqual(plain(row.fields), { a: 1.25, "d.0": 3, "d.1": null });
  assert.deepEqual(row.field_names, ["a", "b", "c", "d.0", "d.1", "e"]);
  // Every name can sit inside a citation: none holds "]" or whitespace, which end one.
  for (const [name] of Agents.leaves({ list: [{ deep: [{ n: 1 }] }] })) assert.doesNotMatch(name, /[\]\s]/);
});

test("the live engine must offer exactly the agents the page was published with", () => {
  const live = () => ({ schema: "eagent.engine.v1", agents: plain(catalog.agents).map(({ name, description, parameters }) => ({ name, description, parameters })), limits: plain(catalog.limits) });
  assert.deepEqual(Agents.matchesEngine(catalog, live()), []);
  const missing = live(); missing.agents.pop();
  assert.match(Agents.matchesEngine(catalog, missing).join(), /has no agent/);
  const extra = live(); extra.agents.push({ name: "surprise", description: "x", parameters: {} });
  assert.match(Agents.matchesEngine(catalog, extra).join(), /surprise that this page does not list/);
  const changed = live(); changed.agents[0].parameters = { other: "x" };
  assert.match(Agents.matchesEngine(catalog, changed).join(), /different parameters/);
  const reworded = live(); reworded.agents[0].description += " (changed)";
  assert.match(Agents.matchesEngine(catalog, reworded).join(), /described differently/);
  const limits = live(); limits.limits.max_result_bytes = 1;
  assert.match(Agents.matchesEngine(catalog, limits).join(), /different size limits/);
  assert.match(Agents.matchesEngine(catalog, null).join(), /does not report/);
});

test("the agents panel lists every agent, where it runs and whether it can run now", () => {
  const off = Agents.registryView({ referenceSchemas: bundle.schemas.concat([{ name: "calculate", description: "c", parameters: {} }]), catalog, engineReady: false });
  assert.equal(off.length, bundle.schemas.length + 1 + catalog.agents.length);
  assert.ok(off.filter((agent) => agent.kind === "compute").every((agent) => agent.runs === "browser-wasm" && agent.available === false && agent.wraps));
  assert.ok(off.filter((agent) => agent.kind !== "compute").every((agent) => agent.runs === "browser-js" && agent.available));
  assert.equal(off.find((agent) => agent.name === "calculate").kind, "calculator");
  const on = Agents.registryView({ referenceSchemas: [], catalog, engineReady: true });
  assert.ok(on.every((agent) => agent.available));
});

// ---- routing a question to compute agents without a model -----------------
const route = (text, tasks = catalog.tasks) => Agents.planCompute(text, { ...catalog, tasks });
const names = (plan) => plan.calls.map((item) => item.interface);

test("a question that gives the inputs is routed to the agent that computes it", () => {
  const align = route(`Align ${SEQ_A} with ${SEQ_B} and tell me the identity`);
  assert.deepEqual(names(align), ["align_sequences"]);
  assert.deepEqual(plain(align.calls[0].arguments), { sequence_a: SEQ_A, sequence_b: SEQ_B });
  assert.match(align.calls[0].rationale, /22 and 22 residues/);

  const hamming = route(`hamming distance between ${SEQ_A} and ${SEQ_B}`);
  assert.deepEqual(names(hamming), ["sequence_distance"]);
  assert.equal(hamming.calls[0].arguments.method, "hamming");
  assert.equal(route(`k-mer distance ${SEQ_A} ${SEQ_B}`).calls[0].arguments.method, undefined, "the default method is the agent's own");

  const variants = route(`Check the substitutions K2A and T3S on ${SEQ_A}`);
  assert.deepEqual(names(variants), ["variant_check"]);
  assert.deepEqual(plain(variants.calls[0].arguments), { sequence: SEQ_A, variants: "K2A, T3S" });

  const wilson = route("Give me a Wilson interval for 7 of 20 poses at 99% confidence");
  assert.deepEqual(plain(wilson.calls[0].arguments), { successes: "7", trials: "20", confidence: "0.99" });
  assert.deepEqual(plain(route("5/5 poses succeeded, what is the interval?").calls[0].arguments), { successes: "5", trials: "5" });

  const distance = route("What is the distance between 0,0,0 and 3,4,0?");
  assert.deepEqual(plain(distance.calls[0].arguments), { measure: "distance", points: "0,0,0; 3,4,0" });
  assert.deepEqual(plain(route("angle at 0,0,0 between 1,0,0 and 0,1,0 — points 1,0,0; 0,0,0; 0,1,0").calls), [], "four triples are not an angle");
  assert.equal(route("dihedral of 1,0,0 0,0,0 0,1,0 0,1,1").calls[0].arguments.measure, "dihedral");
  assert.equal(route("centroid of 0,0,0 2,0,0 1,3,0").calls[0].arguments.measure, "centroid");

  const gate = route("What is still unresolved for KRED_PILOT_001?");
  assert.deepEqual(plain(gate.calls[0].arguments), { task: "KRED_PILOT_001" });
  assert.match(route("which gates are still open?").calls[0].rationale, /only shipped task/);
  assert.equal(route("which gates are still open?", ["A_1", "B_2"]).calls.length, 0, "with several tasks it does not guess which");

  assert.deepEqual(names(route("Which pipeline steps can run in a browser?")), ["pipeline_catalog"]);

  const run = route("Run the pipeline on KRED_PILOT_001 and tell me where it stops");
  assert.deepEqual(names(run), ["run_pipeline"]);
  assert.deepEqual(plain(run.calls[0].arguments), { task: "KRED_PILOT_001" });
  assert.match(run.calls[0].rationale, /shipped task “KRED_PILOT_001”/);
  assert.deepEqual(names(route("run KRED_PILOT_001")), ["run_pipeline"]);
  assert.match(route("please run the controller").calls[0].rationale, /only shipped task/);
  assert.ok(!names(route("run the pipeline", ["A_1", "B_2"])).includes("run_pipeline"), "with several tasks it does not guess which");
  assert.deepEqual(names(route("Run the pipeline for KRED_PILOT_001, then tell me what is unresolved")), ["task_gate_report", "run_pipeline"]);
  assert.deepEqual(names(route("Which pipeline steps can run in a browser and which need a program?")), ["pipeline_catalog"]);
  assert.deepEqual(names(route("Run the pipeline for KRED_PILOT_001 and say which steps need an external program")), ["run_pipeline", "pipeline_catalog"], "asking about both gets both");
});

test("a task id is matched literally, so a catalog cannot turn it into a pattern", () => {
  assert.deepEqual(names(route("what is unresolved for A.B?", ["A.B"])), ["task_gate_report"]);
  assert.deepEqual(names(route("what is unresolved for AXB? gates", ["A.B", "(a+)+$"])), [], "the dot is a dot, and a hostile id is just text");
  assert.doesNotThrow(() => route("unresolved gates for aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!", ["(a+)+$", "[", "*"]));
});

test("a message that merely resembles a request is not routed", () => {
  for (const text of [
    "Inspect Ssal-KRED activity on 2a. Explain missing values and detection limits.",
    "ALIGNMENT IDENTITY SIMILARITY DISTANCE ACTIVITY INTERVAL are words, not sequences",
    "Align these sequences, please",
    "I approve of this approach",
    "The 2024-10-11 release had 3 of the fixes",
    "list the structures",
    "Can KRED_PILOT_001 be explained without running anything?",
    "(2.5 + 3.5) * 4",
    "",
    "   ",
  ]) assert.deepEqual(names(route(text)), [], text);
  for (const text of ["I ran the pipeline last week", "the pipeline was running slowly", "Why does the pipeline run so rarely?"]) {
    assert.ok(!names(route(text)).includes("run_pipeline"), `${text}: a mention of the pipeline is not a request to run it`);
  }
  assert.match(route("Align MKTAYIAKQRQ please").notes.join(), /needs two sequences .* I found 1/);
  assert.match(route("Check variant K2A").notes.join(), /needs the parent sequence/);
  assert.deepEqual(route("Wilson interval for 8 of 7").notes, ["8 of 7 is not a count of successes out of trials."]);
  assert.match(route("distance 1,2,3").notes.join(), /needs exactly 2 points/);
  assert.deepEqual(names(route("approve the synthesis for KRED_PILOT_001")), ["task_gate_report"], "reading what is open is not approving it");
});

test("the registry plans reference and compute requests together, within one bound", () => {
  const { agents } = composed();
  const both = agents.plan(`List the structures, then align ${SEQ_A} with ${SEQ_B}`);
  assert.deepEqual(names(both), ["list_structure_entries", "align_sequences"]);
  assert.deepEqual(agents.plan("list the structures").notes, []);
  const neither = agents.plan("tell me a story");
  assert.deepEqual(neither.calls, []);
  assert.match(neither.notes.join("\n"), /could not match/, "the reference router's guidance is kept when nothing matched");
  assert.match(neither.notes.join("\n"), /align MKTAYIAKQR with/, "and compute examples are added");
  const onlyCompute = agents.plan(`align ${SEQ_A} with ${SEQ_B}`);
  assert.equal(onlyCompute.notes.some((note) => /could not match/.test(note)), false, "no complaint once something matched");
  assert.equal(agents.plan("").notes[0], "Enter a question or an id.");
});

// ---- the whole loop, through the real browser agent ------------------------
function runtimeWith(provider) {
  const requests = [];
  const context = vm.createContext({ window: {}, URL, TextEncoder, TextDecoder, AbortController, Response, Headers, Uint8Array, performance, crypto: crypto.webcrypto, console, setTimeout, clearTimeout,
    fetch: async (url, options) => {
      if (url === "./reference-data.json") return new Response(JSON.stringify(bundle), { headers: { "Content-Type": "application/json" } });
      requests.push(JSON.parse(options.body));
      return provider(requests.length, requests.at(-1));
    } });
  context.window = context;
  for (const file of ["endpoint.js", "reference-tools.js", "agents.js", "browser-agent.js"]) vm.runInContext(fs.readFileSync(path.join(project, "web", file), "utf8"), context);
  return { runtime: context.window.EAgentBrowserRuntime, requests };
}
const modelConfig = { provider: "custom", protocol: "openai_chat", api_key: "test-browser-only-api-key-secret", model: "model-example", api_url: "https://provider.example/v1/chat/completions" };
const turn = (value) => new Response(JSON.stringify({ choices: [{ finish_reason: "stop", message: { content: JSON.stringify(value) } }] }), { headers: { "Content-Type": "application/json" } });
const question = `Align ${SEQ_A} with ${SEQ_B} and report the identity.`;

test("the model routes the question to a compute agent, the browser computes it, and the cited answer is verified", async () => {
  const engine = engineOf(standard);
  const { runtime, requests } = runtimeWith((count, body) => {
    if (count === 1) return turn({ reasoning: "I will align the two sequences.", tool_calls: [{ interface: "align_sequences", arguments: { sequence_a: SEQ_A, sequence_b: SEQ_B }, rationale: "compute the identity" }] });
    const results = JSON.parse(body.messages.at(-1).content.slice("Tool results:\n".length));
    const cite = results[0].value.cite;
    return turn({ reasoning: `The sequences are 90.9 percent identical over aligned columns [cite artifact=${cite.artifact} sha256=${cite.sha256} row=${cite.row} field=identity_percent_over_aligned_columns method=read].` });
  });
  const result = await runtime.chat({ message: question, model_config: modelConfig, compute: { engine, catalog } });
  assert.equal(result.completion_verified, true);
  assert.match(result.answer, /90\.9 percent identical/);
  assert.deepEqual(engine.calls, [["align_sequences", { sequence_a: SEQ_A, sequence_b: SEQ_B }]]);
  assert.ok(result.transcript.tools_offered.includes("align_sequences"));
  assert.ok(result.transcript.tools_offered.includes("reference_summary"));
  assert.match(result.transcript.caveat, /Compute agents ran the project’s own Python code on the user’s device/);
  assert.equal(result.transcript.turns[0].results[0].executor, "browser-wasm");
  const system = requests[0].messages[0].content;
  assert.match(system, /COMPUTE AGENTS/);
  assert.match(system, /"name":"align_sequences"/);
  assert.match(system, /WebAssembly/);
});

test("without the compute option the model is offered the reference tools alone", async () => {
  const engine = engineOf(standard);
  const { runtime, requests } = runtimeWith(() => turn({ reasoning: "Nothing to look up." }));
  const result = await runtime.chat({ message: question, model_config: modelConfig });
  assert.equal(result.completion_verified, true);
  assert.equal(result.transcript.tools_offered.includes("align_sequences"), false);
  assert.doesNotMatch(requests[0].messages[0].content, /COMPUTE AGENTS|align_sequences/);
  assert.doesNotMatch(result.transcript.caveat, /Compute agents/);
  assert.equal(engine.calls.length, 0);
  const asked = await runtime.chat({ message: question, model_config: modelConfig, compute: { engine, catalog } }).catch(() => null);
  assert.ok(asked, "the same runtime serves both modes");
});

test("a model that misquotes a computed number is stopped by the guard", async () => {
  const engine = engineOf(standard);
  const { runtime } = runtimeWith((count, body) => {
    if (count === 1) return turn({ reasoning: "Aligning.", tool_calls: [{ interface: "align_sequences", arguments: { sequence_a: SEQ_A, sequence_b: SEQ_B } }] });
    const cite = JSON.parse(body.messages.at(-1).content.slice("Tool results:\n".length))[0].value.cite;
    return turn({ reasoning: `They are 99.0 percent identical [cite artifact=${cite.artifact} sha256=${cite.sha256} row=${cite.row} field=identity_percent_over_aligned_columns method=read].` });
  });
  await assert.rejects(runtime.chat({ message: question, model_config: modelConfig, compute: { engine, catalog } }),
    (error) => /numeric guard refused/.test(error.message) && error.data.completion_verified === false);
});

test("a model cannot cite a computation that never happened", async () => {
  const engine = engineOf(standard);
  const { runtime } = runtimeWith(() => turn({ reasoning: `They are 90.9 percent identical [cite artifact=engine_compute sha256=${"ab".repeat(32)} row=align_sequences-0123456789abcdef field=identity_percent_over_aligned_columns method=read].` }));
  await assert.rejects(runtime.chat({ message: question, model_config: modelConfig, compute: { engine, catalog } }), /numeric guard refused/);
  assert.equal(engine.calls.length, 0);
});

test("a model cannot reach past the registry: unknown agents and extra arguments are refused", async () => {
  const engine = engineOf(standard);
  const seen = [];
  const { runtime } = runtimeWith((count, body) => {
    if (count === 1) return turn({ reasoning: "Trying things.", tool_calls: [
      { interface: "approve_gate", arguments: { gate: "synthesis_authorized" } },
      { interface: "align_sequences", arguments: { sequence_a: SEQ_A, sequence_b: SEQ_B, shell: "rm -rf /" } },
    ] });
    seen.push(...JSON.parse(body.messages.at(-1).content.slice("Tool results:\n".length)));
    return turn({ reasoning: "Both were refused." });
  });
  const result = await runtime.chat({ message: question, model_config: modelConfig, compute: { engine, catalog } });
  assert.equal(result.completion_verified, true);
  assert.match(seen[0].refusal, /No such browser tool/);
  assert.match(seen[1].refusal, /unsupported argument/);
  assert.equal(engine.calls.length, 0);
});

test("Run in browser answers a compute question with no model and no network", async () => {
  const engine = engineOf(standard);
  const { runtime, requests } = runtimeWith(() => { throw new Error("no model request may be made"); });
  const run = await runtime.local({ message: question, compute: { engine, catalog } });
  assert.equal(run.completion_verified, false);
  assert.equal(run.execution, "browser-local");
  assert.equal(requests.length, 0);
  assert.deepEqual(engine.calls, [["align_sequences", { sequence_a: SEQ_A, sequence_b: SEQ_B }]]);
  const result = run.transcript.turns[0].results[0];
  assert.equal(result.ok, true);
  assert.equal(result.executor, "browser-wasm");
  assert.match(run.answer, /Ran 1 tool in this browser \(1 computed on this device/);
  assert.match(run.transcript.caveat, /Compute agents ran the project’s own Python code on this device/);
  const plain_ = await runtime.local({ message: "list the structures" });
  assert.match(plain_.answer, /Ran 1 stored-reference tool in this browser\./, "without compute the wording is unchanged");
});
