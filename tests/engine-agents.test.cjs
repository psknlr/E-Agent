"use strict";

// The whole stack with nothing scripted except the model: the real page code,
// the real client, the real worker installer and the real Python running as
// WebAssembly. A "model" that follows the protocol asks for each agent, and
// answers with citations copied from the real results, so this is where a field
// path, a rounding or a result shape that the unit tests could not anticipate
// would show up.

const test = require("node:test");
const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const project = path.resolve(__dirname, "..");
const { InProcessWorker, built, manifest } = require("./fixtures/engine_inprocess.cjs");
const skip = built() ? false : "web/engine is not built; run python scripts/build_engine.py";
const bundle = JSON.parse(fs.readFileSync(path.join(project, "web/reference-data.json"), "utf8"));
const catalog = JSON.parse(fs.readFileSync(path.join(project, "web/agent-catalog.json"), "utf8"));
const { create } = require(path.join(project, "web/engine-client.js"));
const install = require(path.join(project, "web/engine-install.js"));

let engine;
async function started() {
  if (engine) return engine;
  engine = create({ fetch: async () => ({ ok: true, status: 200, json: async () => manifest() }), Worker: InProcessWorker, WebAssembly, loadTimeoutMs: 180000, callTimeoutMs: 120000 });
  global.EAgentEngineInstall = install;
  await engine.probe();
  delete global.EAgentEngineInstall;
  await engine.load();
  return engine;
}

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
  return context.window.EAgentBrowserRuntime;
}
// A complete reaction: the controller normalises it for real, then waits for a person.
const SPECIFIED_TASK = [
  "task_id: SPECIFIED_001", "task_mode: enzyme_mining", "reaction:", "  reaction_class: ketone_to_secondary_alcohol",
  "  atom_mapped_reaction_smiles: '[CH3:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1>>[CH3:1][C@H:2]([OH:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1'",
  "  substrate:", "    name: acetophenone", "    isomeric_smiles: CC(=O)c1ccccc1", "  product:", "    name: (S)-1-phenylethanol",
  "    isomeric_smiles: C[C@H](O)c1ccccc1", "    creates_new_stereocenter: true", "",
].join("\n");
const modelConfig = { provider: "custom", protocol: "openai_chat", api_key: "test-browser-only-api-key-secret", model: "model-example", api_url: "https://provider.example/v1/chat/completions" };
const turn = (value) => new Response(JSON.stringify({ choices: [{ finish_reason: "stop", message: { content: JSON.stringify(value) } }] }), { headers: { "Content-Type": "application/json" } });
const cite = (c, field) => `[cite artifact=${c.artifact} sha256=${c.sha256} row=${c.row} field=${field} method=read]`;

// Ask for one agent, then answer with whatever `say(value, citeFor)` writes from
// the real result. Returns the finished chat and the real result.
async function ask(agent, args, say) {
  const live = await started();
  let seen;
  const runtime = runtimeWith((count, body) => {
    if (count === 1) return turn({ reasoning: `I will call ${agent}.`, tool_calls: [{ interface: agent, arguments: args, rationale: "compute it" }] });
    seen = JSON.parse(body.messages.at(-1).content.slice("Tool results:\n".length))[0];
    return turn({ reasoning: say(seen.value, (field) => cite(seen.value.cite, field)) });
  });
  const result = await runtime.chat({ message: `Use ${agent}.`, model_config: modelConfig, compute: { engine: live, catalog } });
  return { result, seen };
}

test("the engine is offered exactly the agents the page was published with", { skip }, async () => {
  const live = await started();
  assert.deepEqual(global.EAgentAgents ?? require(path.join(project, "web/agents.js")).matchesEngine(catalog, await live.catalog()), []);
});

test("an alignment is computed in the browser and its identity is quoted with a citation the guard verifies", { skip }, async () => {
  const { result, seen } = await ask("align_sequences", { sequence_a: "MKTAYIAKQRQISFVKSHFSRQ", sequence_b: "MKTAYLAKQRQVSFVKSHFSRQ" },
    (value, c) => `Over the aligned columns the sequences are ${value.identity_percent_over_aligned_columns.toFixed(1)} percent identical ${c("identity_percent_over_aligned_columns")}.`);
  assert.equal(result.completion_verified, true);
  assert.equal(seen.executor, "browser-wasm");
  assert.equal(seen.value.identical_columns, 20);
  assert.match(result.answer, /90\.9 percent identical/);
});

test("a Wilson interval's bounds are quoted from the computed fields, and a wrong bound is refused", { skip }, async () => {
  const good = await ask("wilson_interval", { successes: "7", trials: "20" },
    (value, c) => `At ${value.confidence_percent} percent ${c("confidence_percent")} confidence the interval runs from ${value.lower_percent.toFixed(1)} percent ${c("lower_percent")} `
      + `to ${value.upper_percent.toFixed(1)} percent ${c("upper_percent")}.`);
  assert.equal(good.result.completion_verified, true);
  assert.match(good.result.answer, /18\.1 percent .* to 56\.7 percent/);
  assert.equal(good.seen.value.lower_percent, good.seen.value.lower * 100);
  await assert.rejects(ask("wilson_interval", { successes: "7", trials: "20" },
    (value, c) => `The lower bound is ${(value.lower_percent + 5).toFixed(1)} percent ${c("lower_percent")}.`), /numeric guard refused/);
  // A fraction cannot back a percentage: the scale is part of the claim.
  await assert.rejects(ask("wilson_interval", { successes: "7", trials: "20" },
    (value, c) => `The lower bound is 18.1 percent ${c("lower")}.`), /numeric guard refused/);
});

test("a measured distance in angstroms is cited from the computed field", { skip }, async () => {
  const { result } = await ask("geometry_measure", { measure: "distance", points: "0,0,0; 3,4,0" },
    (value, c) => `The two atoms are 5.0 angstroms apart ${c("distance_angstrom")}.`);
  assert.equal(result.completion_verified, true);
});

test("counts inside nested results are cited by dotted path", { skip }, async () => {
  const { result, seen } = await ask("task_gate_report", { task: "KRED_PILOT_001" },
    (value, c) => `In total ${value.n_unresolved_fields_total} records ${c("n_unresolved_fields_total")} remain open, and the first gate has ${value.gates[0].n_unresolved} records ${c("gates.0.n_unresolved")}.`);
  assert.equal(result.completion_verified, true);
  assert.equal(seen.value.n_unresolved_fields_total, 10);
  assert.match(JSON.stringify(seen.value.statement), /only a named person can approve/);
});

test("the pipeline map says which steps need a program, and the model is held to it", { skip }, async () => {
  const { result, seen } = await ask("pipeline_catalog", {},
    (value, c) => `${value.n_need_a_program} records ${c("n_need_a_program")} need an external program.`);
  assert.equal(result.completion_verified, true);
  assert.equal(seen.value.n_need_a_program, 3);
  assert.equal(seen.value.n_run_in_browser, 5);
  await assert.rejects(ask("pipeline_catalog", {}, (value, c) => `4 records ${c("n_need_a_program")} need an external program.`), /numeric guard refused/);
});

test("the pipeline agent runs the real controller, and the model reports where it stopped with numbers the guard checks", { skip }, async () => {
  const { result, seen } = await ask("run_pipeline", { task: "KRED_PILOT_001" },
    (value, c) => `The controller ran ${value.n_steps_run} items ${c("n_steps_run")} and found ${value.n_blockers} items ${c("n_blockers")} that block the reaction spec; `
      + `it is waiting on ${value.waiting_on[0].gate}, which only a person can clear.`);
  assert.equal(result.completion_verified, true);
  assert.equal(seen.executor, "browser-wasm");
  assert.equal(seen.value.outcome, "awaiting_human");
  assert.deepEqual(seen.value.steps.map((step) => step.interface), ["normalize_reaction"]);
  assert.equal(seen.value.waiting_on[0].gate, "reaction_spec_confirmed");
  assert.match(seen.value.statement, /cannot clear a human decision point/);
  await assert.rejects(ask("run_pipeline", { task: "KRED_PILOT_001" },
    (value, c) => `It found ${value.n_blockers + 1} items ${c("n_blockers")} that block the reaction spec.`), /numeric guard refused/);
});

test("a flag in a pasted task is not an approval, and the run still stops at the first gate", { skip }, async () => {
  const task = SPECIFIED_TASK.replace("task_mode: enzyme_mining\n", "task_mode: enzyme_mining\napproval:\n  reaction_spec_confirmed: true\n");
  const { seen } = await ask("run_pipeline", { task_yaml: task }, () => "Done.");
  assert.equal(seen.ok, true, seen.refusal);
  assert.equal(seen.value.outcome, "awaiting_human");
  assert.equal(seen.value.n_steps_run, 1);
  assert.equal(seen.value.steps[0].status, "success");
  assert.equal(seen.value.waiting_on[0].gate, "reaction_spec_confirmed");
});

test("a refusal from the real project code reaches the model as a refusal with no value", { skip }, async () => {
  const { result, seen } = await ask("sequence_distance", { sequence_a: "ACDEFGHIK", sequence_b: "ACDEFGH", method: "hamming" },
    () => "The agent refused, so I have no distance to report.");
  assert.equal(result.completion_verified, true);
  assert.equal(seen.ok, false);
  assert.equal(seen.value, null);
  assert.match(seen.refusal, /./);
  assert.equal(seen.engine_kind, "refused");
});

test("Run in browser computes a question with no model, using the real engine", { skip }, async () => {
  const live = await started();
  const runtime = runtimeWith(() => { throw new Error("no model request may be made"); });
  const run = await runtime.local({ message: "Give me a Wilson interval for 7 of 20 poses and check the substitutions K2A and Y9F on MKTAYIAKQR", compute: { engine: live, catalog } });
  const results = run.transcript.turns[0].results;
  assert.deepEqual([...results.map((r) => r.tool)], ["variant_check", "wilson_interval"], "one message, two agents, each with the inputs the user wrote");
  assert.ok(results.every((r) => r.ok && r.executor === "browser-wasm"));
  assert.equal(results[0].value.n_wild_type_mismatches, 1, "position 9 is Q, not Y");
  assert.ok(Math.abs(results[1].value.lower - 0.18119182410108206) < 1e-12);
  assert.match(run.answer, /Ran 2 tools in this browser \(2 computed on this device/);
  assert.equal(run.completion_verified, false, "a local run is never presented as a model completion");
});

test("Run in browser starts the real controller from a plain sentence and shows where it stops", { skip }, async () => {
  const live = await started();
  const runtime = runtimeWith(() => { throw new Error("no model request may be made"); });
  const run = await runtime.local({ message: "Run the pipeline on KRED_PILOT_001 and tell me where it stops", compute: { engine: live, catalog } });
  const results = run.transcript.turns[0].results;
  assert.deepEqual([...results.map((r) => r.tool)], ["run_pipeline"]);
  assert.equal(results[0].ok, true, results[0].refusal);
  assert.equal(results[0].value.waiting_on[0].gate, "reaction_spec_confirmed");
  assert.ok(results[0].elapsed_ms < 60000);
  assert.match(run.answer, /1 computed on this device/);
});

test("a call that the engine cannot satisfy is refused by the page, not run", { skip }, async () => {
  const live = await started();
  const runtime = runtimeWith(() => turn({ reasoning: "x" }));
  const before = live.state.status;
  const run = await runtime.local({ message: "Align " + "M".repeat(1300) + " with " + "K".repeat(1300), compute: { engine: live, catalog } });
  const result = run.transcript.turns[0].results[0];
  assert.equal(result.ok, false);
  assert.match(result.refusal, /exceeds its 2400-character limit|accepts at most 1200/);
  assert.equal(live.state.status, before, "a refused request leaves the engine running");
});
