"use strict";

// The page's side of the engine. Most of these tests use a scripted stand-in for
// the worker, to hold the state machine to account without a 20 MB runtime; one
// drives the real built engine through the real client to prove they fit.

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const project = path.resolve(__dirname, "..");
const { InProcessWorker, built, manifest } = require("./fixtures/engine_inprocess.cjs");
const install = require(path.join(project, "web", "engine-install.js"));
const { create, describeDevice, describeGpu, EngineClientError } = require(path.join(project, "web", "engine-client.js"));

const HEX = "ab".repeat(32);
const fakeManifest = {
  schema: "eagent.engine.manifest.v1", engine_schema: "eagent.engine.v1", python: "3.14.2", pyodide: "314.0.7", total_bytes: 1234,
  install: { wheels: [], site_packages_zips: [], project: "eagent-project.zip" },
  selftest: [{ agent: "wilson_interval", arguments: "{}", sha256: HEX }],
  files: Object.fromEntries(["eagent-project.zip", "pyodide/pyodide.mjs", "pyodide/pyodide.asm.mjs", "pyodide/pyodide.asm.wasm", "pyodide/python_stdlib.zip"].map((name) => [name, { bytes: 1, sha256: HEX }])),
};
const json = (body, status = 200) => ({ ok: status >= 200 && status < 300, status, json: async () => body });
const served = (body = fakeManifest) => async (url) => {
  assert.equal(url, "engine/manifest.json");
  return json(body);
};

// A worker whose replies a test scripts: `script(message, post)` returns a
// reply, undefined to stay silent, or throws to simulate a dead worker.
function fakeWorkers(script) {
  const made = [];
  class FakeWorker {
    constructor(url, options) {
      this.url = url;
      this.options = options;
      this.terminated = false;
      made.push(this);
    }
    postMessage(message) {
      setImmediate(async () => {
        if (this.terminated) return;
        const reply = await script(message, (extra) => this.onmessage && this.onmessage({ data: { id: message.id, ...extra } }), this);
        if (reply && !this.terminated) this.onmessage({ data: { id: message.id, ...reply } });
      });
    }
    terminate() { this.terminated = true; }
  }
  return { Worker: FakeWorker, made };
}
const ok = (message, post) => {
  if (message.type === "init") { post({ type: "progress", stage: "runtime", label: "Starting" }); return { type: "ready", info: { python: "3.14.2", agents: ["wilson_interval"] } }; }
  if (message.type === "call") return { type: "result", json: JSON.stringify({ ok: true, agent: message.agent, args: message.args }) };
  if (message.type === "catalog") return { type: "result", json: JSON.stringify({ schema: "eagent.engine.v1", agents: [] }) };
  return { type: "error", kind: "protocol", message: "?" };
};
const client = (workers, extra = {}) => create({ fetch: served(), Worker: workers.Worker, WebAssembly, ...extra });

test("a site published without the engine reports it as absent, and load says so", async () => {
  for (const [label, fetch, pattern] of [
    ["404", async () => json({}, 404), /without the engine files/],
    ["500", async () => json({}, 500), /HTTP 500/],
    ["unreachable", async () => { throw new TypeError("Failed to fetch"); }, /could not be reached: Failed to fetch/],
  ]) {
    const workers = fakeWorkers(ok);
    const engine = create({ fetch, Worker: workers.Worker, WebAssembly });
    assert.equal((await engine.probe()).status, "absent", label);
    assert.match(engine.state.note, pattern, label);
    await assert.rejects(engine.load(), (error) => error.kind === "absent" && pattern.test(error.message), label);
    assert.equal(workers.made.length, 0, `${label}: no worker is started for an engine that is not there`);
  }
});

test("a browser without workers or WebAssembly is unsupported and nothing is fetched", async () => {
  let fetched = 0;
  const fetch = async () => { fetched += 1; return json(fakeManifest); };
  const noWorker = create({ fetch, Worker: undefined, WebAssembly });
  assert.equal((await noWorker.probe()).status, "unsupported");
  assert.match(noWorker.supported(), /no Web Workers/);
  const noWasm = create({ fetch, Worker: fakeWorkers(ok).Worker, WebAssembly: undefined });
  assert.equal((await noWasm.probe()).status, "unsupported");
  await assert.rejects(noWasm.load(), (error) => error.kind === "unsupported");
  assert.equal(fetched, 0);
});

test("a malformed manifest is an error, not an engine", async () => {
  global.EAgentEngineInstall = install;
  try {
    const engine = create({ fetch: served({ schema: "something.else" }), Worker: fakeWorkers(ok).Worker, WebAssembly });
    assert.equal((await engine.probe()).status, "error");
    assert.equal(engine.state.error.kind, "manifest");
    assert.match(engine.state.error.message, /expected schema/);
  } finally { delete global.EAgentEngineInstall; }
});

test("load starts one worker, reports progress, and a second load reuses it", async () => {
  const workers = fakeWorkers(ok);
  const engine = client(workers);
  assert.equal(engine.state.status, "unknown");
  assert.equal((await engine.probe()).status, "idle");
  const seen = [];
  engine.subscribe((state) => seen.push([state.status, state.progress && state.progress.stage]));
  const [first, second] = await Promise.all([engine.load(), engine.load()]);
  assert.equal(first, second);
  assert.equal(engine.state.status, "ready");
  assert.equal(workers.made.length, 1, "concurrent loads share one worker");
  assert.equal(workers.made[0].url, "engine-worker.js");
  assert.deepEqual(workers.made[0].options, { type: "module" }, "the runtime's loader needs a module worker");
  assert.deepEqual(seen.map(([status]) => status).filter((status, index, all) => all.indexOf(status) === index), ["loading", "ready"]);
  assert.ok(seen.some(([, stage]) => stage === "runtime"), "progress from the worker reaches subscribers");
  assert.equal(await engine.load(), first);
  assert.equal(workers.made.length, 1);
});

test("calls need a running engine, validate their shape, and keep their order", async () => {
  const workers = fakeWorkers(async (message, post) => {
    if (message.type === "call" && message.agent === "slow") await new Promise((resolve) => setTimeout(resolve, 40));
    return ok(message, post);
  });
  const engine = client(workers);
  await assert.rejects(engine.call("wilson_interval", "{}"), (error) => error.kind === "not_ready");
  await engine.load();
  await assert.rejects(engine.call(7, "{}"), (error) => error.kind === "protocol");
  await assert.rejects(engine.call("wilson_interval", {}), (error) => error.kind === "protocol");
  const order = [];
  const results = await Promise.all(["slow", "fast", "slow"].map((agent, index) =>
    engine.call(agent, `{"n":${index}}`).then((text) => { order.push(index); return JSON.parse(text); })));
  assert.deepEqual(order, [0, 1, 2], "a slow call does not let a later one overtake it");
  assert.deepEqual(results.map((result) => result.agent), ["slow", "fast", "slow"]);
  assert.deepEqual((await engine.catalog()).schema, "eagent.engine.v1");
});

test("a call that runs too long stops the worker, says so, and a fresh start runs a new one", async () => {
  let hang = true;
  const workers = fakeWorkers((message, post) => (message.type === "call" && hang ? undefined : ok(message, post)));
  const engine = client(workers, { callTimeoutMs: 30 });
  await engine.load();
  await assert.rejects(engine.call("align_sequences", "{}"), (error) => error.kind === "timeout" && /30 seconds|0 seconds|seconds/.test(error.message));
  assert.equal(workers.made[0].terminated, true, "the runaway worker is ended, not left running");
  assert.equal(engine.state.status, "idle");
  assert.match(engine.state.note, /Press Start to run it again/);
  assert.doesNotMatch(engine.state.note, /restarts/, "nothing restarts it by itself, so the card must not say so");
  await assert.rejects(engine.call("wilson_interval", "{}"), (error) => error.kind === "not_ready");
  hang = false;
  await engine.load();
  assert.equal(workers.made.length, 2);
  assert.equal(JSON.parse(await engine.call("wilson_interval", "{}")).ok, true);
});

test("calls queued behind a runaway call fail cleanly instead of hanging", async () => {
  const workers = fakeWorkers((message, post) => (message.type === "call" ? undefined : ok(message, post)));
  const engine = client(workers, { callTimeoutMs: 30 });
  await engine.load();
  const outcomes = await Promise.allSettled([engine.call("a", "{}"), engine.call("b", "{}")]);
  assert.deepEqual(outcomes.map((outcome) => outcome.status), ["rejected", "rejected"]);
  assert.equal(outcomes[0].reason.kind, "timeout");
  assert.equal(outcomes[1].reason.kind, "not_ready");
});

test("a worker script that cannot load fails at once, not after the time-out", async () => {
  class Broken {
    constructor() { setImmediate(() => this.onerror({ message: "Failed to load engine-worker.js" })); }
    postMessage() {}
    terminate() { this.terminated = true; }
  }
  const engine = create({ fetch: served(), Worker: Broken, WebAssembly, loadTimeoutMs: 60000 });
  const started = Date.now();
  await assert.rejects(engine.load(), (error) => error.kind === "worker" && /Failed to load engine-worker.js/.test(error.message));
  assert.ok(Date.now() - started < 2000);
  assert.equal(engine.state.status, "error");
  assert.equal(engine.state.error.kind, "worker");
});

test("an install that reports an error leaves the engine unusable, and load can be retried", async () => {
  let attempt = 0;
  const workers = fakeWorkers((message, post) => {
    if (message.type !== "init") return ok(message, post);
    attempt += 1;
    return attempt === 1 ? { type: "error", kind: "integrity", message: "wheels/x.whl does not match the build manifest" } : ok(message, post);
  });
  const engine = client(workers);
  await assert.rejects(engine.load(), (error) => error instanceof EngineClientError && error.kind === "integrity");
  assert.equal(engine.state.status, "error");
  assert.equal(engine.state.error.kind, "integrity");
  assert.equal(workers.made[0].terminated, true);
  await assert.rejects(engine.call("wilson_interval", "{}"), (error) => error.kind === "not_ready");
  await engine.load();
  assert.equal(engine.state.status, "ready");
});

test("an install that never finishes is abandoned at the load time-out", async () => {
  const workers = fakeWorkers(() => undefined);
  const engine = client(workers, { loadTimeoutMs: 30 });
  await assert.rejects(engine.load(), (error) => error.kind === "timeout");
  assert.equal(engine.state.status, "error");
  assert.equal(workers.made[0].terminated, true);
});

test("unload ends the worker and returns to idle", async () => {
  const workers = fakeWorkers(ok);
  const engine = client(workers);
  await engine.load();
  engine.unload();
  assert.equal(workers.made[0].terminated, true);
  assert.equal(engine.state.status, "idle");
  assert.equal(engine.state.info, null);
});

test("a worker that fails after it was running leaves the engine in error, not 'ready', and Try again works", async () => {
  const workers = fakeWorkers(ok);
  const engine = client(workers);
  await engine.load();
  assert.equal(engine.state.status, "ready");
  const seen = [];
  engine.subscribe((state) => seen.push(state.status));
  // The browser reports an uncaught error inside the worker.
  workers.made[0].onerror({ message: "Uncaught RangeError: out of memory" });
  assert.equal(engine.state.status, "error", "the card must stop offering agents that cannot answer");
  assert.equal(engine.state.error.kind, "worker");
  assert.match(engine.state.error.message, /out of memory/);
  assert.equal(engine.state.info, null);
  assert.equal(workers.made[0].terminated, true);
  await assert.rejects(engine.call("wilson_interval", "{}"), (error) => error.kind === "not_ready");
  assert.notEqual(await engine.load().then(() => "restarted"), undefined);
  assert.equal(engine.state.status, "ready", "load starts a fresh worker instead of returning the dead engine");
  assert.equal(workers.made.length, 2);
  // Starting again after a failure reads the manifest afresh (idle) before it loads.
  assert.deepEqual(seen.filter((status, index, all) => all.indexOf(status) === index), ["error", "idle", "loading", "ready"]);
});

test("calls in flight when the worker fails are rejected, not left waiting", async () => {
  const workers = fakeWorkers((message, post) => (message.type === "call" ? undefined : ok(message, post)));
  const engine = client(workers, { callTimeoutMs: 60000 });
  await engine.load();
  const pending = engine.call("align_sequences", "{}");
  await new Promise((resolve) => setImmediate(resolve));
  workers.made[0].onerror({ message: "crashed" });
  await assert.rejects(pending, (error) => error.kind === "worker");
});

test("cancelling a start returns to off, not to an error, and a start after it is not disturbed by it", async () => {
  const workers = fakeWorkers(async (message, post, worker) => {
    if (message.type !== "init") return ok(message, post);
    // The first worker never finishes installing; a later one does.
    return worker === workers.made[0] ? undefined : ok(message, post);
  });
  const engine = client(workers);
  await engine.probe();
  const seen = [];
  engine.subscribe((state) => seen.push(state.status));
  const first = engine.load();
  const outcome = first.then(() => "started", (error) => error.kind);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(engine.state.status, "loading");
  engine.unload();
  assert.equal(engine.state.status, "idle");
  assert.equal(workers.made[0].terminated, true);
  assert.equal(await outcome, "stopped", "the cancelled start reports that it was stopped");
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(engine.state.status, "idle", "and its failure handler does not turn the cancel into an error");
  assert.equal(engine.state.error, null);
  assert.ok(!seen.includes("error"), `the card never showed a failure: ${seen}`);

  // Cancel, then start again before the cancelled start has finished failing.
  const second = engine.load();
  engine.unload();
  const third = engine.load();
  assert.equal(await second.then(() => "started", (error) => error.kind), "stopped");
  const info = await third;
  assert.equal(engine.state.status, "ready");
  assert.equal(info.python, "3.14.2");
  assert.equal(workers.made.at(-1).terminated, false, "the cancelled start must not end the worker that replaced it");
});

test("the device readout reports what the browser exposes and claims nothing else", async () => {
  assert.deepEqual(describeDevice({ hardwareConcurrency: 8, deviceMemory: 4, gpu: {}, ml: undefined },
    { WebAssembly: {}, Worker() {}, isSecureContext: true }),
  { cpu_threads: 8, memory_gb: 4, webassembly: true, workers: true, webgpu: true, webnn: false, secure_context: true });
  assert.deepEqual(describeDevice({}, {}), { cpu_threads: null, memory_gb: null, webassembly: false, workers: false, webgpu: false, webnn: false, secure_context: true });
  assert.equal(describeDevice({ hardwareConcurrency: "8" }, {}).cpu_threads, null, "a value of the wrong type is not reported");

  assert.match((await describeGpu({})).reason, /not offered/);
  assert.match((await describeGpu({ gpu: { requestAdapter: async () => null } })).reason, /no adapter/);
  assert.match((await describeGpu({ gpu: { requestAdapter: async () => { throw new Error("denied"); } } })).reason, /denied/);
  assert.match((await describeGpu({ gpu: { requestAdapter: () => new Promise(() => {}) } }, 20)).reason, /timed out/);
  assert.deepEqual(await describeGpu({ gpu: { requestAdapter: async () => ({ info: { vendor: "apple", architecture: "metal-3", description: "M-series" } }) } }),
    { available: true, vendor: "apple", architecture: "metal-3", description: "M-series" });
  const bare = await describeGpu({ gpu: { requestAdapter: async () => ({}) } });
  assert.deepEqual(bare, { available: true, vendor: "", architecture: "", description: "" });
});

test("the real client drives the real built engine", { skip: built ? false : "web/engine is not built; run python scripts/build_engine.py" }, async () => {
  const read = new Set();
  class ReadingWorker extends InProcessWorker {
    constructor() {
      super();
      const inner = this.host;
      this.host = (message) => { if (message.type === "init") for (const name of message.manifest.install.wheels) read.add(name); return inner(message); };
    }
  }
  const engine = create({ fetch: served(manifest()), Worker: ReadingWorker, WebAssembly, loadTimeoutMs: 120000 });
  await engine.probe();
  const stages = new Set();
  engine.subscribe((state) => state.progress && stages.add(state.progress.stage));
  const info = await engine.load();
  assert.equal(info.integrity, "verified");
  assert.ok(["runtime", "packages", "project", "check"].every((stage) => stages.has(stage)), [...stages].join());
  const reply = JSON.parse(await engine.call("wilson_interval", JSON.stringify({ successes: "5", trials: "5" })));
  assert.equal(reply.ok, true);
  assert.ok(reply.value.lower > 0.5 && reply.value.lower < 0.6, "5 of 5 at 95% has a lower bound near 0.57");
  assert.equal(reply.value.upper, 1);
  assert.equal(reply.cite.artifact, "engine_compute");
  const refused = JSON.parse(await engine.call("wilson_interval", JSON.stringify({ successes: "6", trials: "5" })));
  assert.equal(refused.ok, false);
  assert.equal(refused.cite, undefined);
  assert.ok(read.size > 0);
  engine.unload();
});
