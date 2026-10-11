"use strict";

// The panels that say where the work happens. They are views of the engine's
// state and of the published catalog, so these tests drive both by hand and read
// the panels back: what the card claims in each state, what is offered to the
// model and when, and that nothing is described that the catalog does not hold.

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const project = path.resolve(__dirname, "..");
const catalog = JSON.parse(fs.readFileSync(path.join(project, "web/agent-catalog.json"), "utf8"));
const bundle = JSON.parse(fs.readFileSync(path.join(project, "web/reference-data.json"), "utf8"));
const Agents = require(path.join(project, "web/agents.js"));
const ComputeUI = require(path.join(project, "web/compute-ui.js"));

class Element {
  constructor(tag = "div", className = "", text = "") {
    this.tagName = tag; this.className = className; this.textContent = text; this.children = []; this.listeners = {};
    this.hidden = false; this.disabled = false; this.checked = false; this.title = "";
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(event, fn) { this.listeners[event] = fn; }
  setAttribute(name, value) { this[name] = value; }
  async click() { return this.listeners.click?.(); }
}
const text = (element) => [element.textContent, ...element.children.map(text)].join(" ").replace(/\s+/g, " ").trim();
const classes = (element) => element.className.split(" ");
const find = (element, predicate, found = []) => { if (predicate(element)) found.push(element); element.children.forEach((child) => find(child, predicate, found)); return found; };

const referenceSchemas = bundle.schemas.concat([{ name: "calculate", description: "Compute a finite arithmetic expression locally.", parameters: {} }]);
const manifest = { total_bytes: 19630000 };
const info = { python: "3.14.2", pyodide: "314.0.7", agents: catalog.agents.map((agent) => agent.name), integrity: "verified", selftest: 4, timings: { total_ms: 6400 }, bundle_sha256: "dc7634b33bb1" + "0".repeat(52) };

// An engine client whose state a test sets by hand.
function fakeEngine(initial = {}) {
  const listeners = new Set();
  const engine = {
    calls: [], state: { status: "idle", manifest, info: null, progress: null, error: null, note: "", ...initial },
    subscribe: (fn) => { listeners.add(fn); return () => listeners.delete(fn); },
    set(changes) { engine.state = { ...engine.state, ...changes }; for (const fn of listeners) fn(engine.state); },
    load: async () => { engine.calls.push("load"); },
    unload: () => { engine.calls.push("unload"); },
    catalog: async () => ({ schema: "eagent.engine.v1", agents: catalog.agents.map(({ name, description, parameters }) => ({ name, description, parameters })), limits: catalog.limits }),
    device: () => ({ cpu_threads: 8, memory_gb: 8, webassembly: true, workers: true, webgpu: true, webnn: false, secure_context: true }),
    gpu: async () => ({ available: true, vendor: "apple", architecture: "metal-3", description: "" }),
  };
  return engine;
}

function panels(engine, { stored = {}, catalogValue = catalog, schemas = referenceSchemas } = {}) {
  const elements = new Map();
  const byId = (id) => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  const make = (tag, className, content) => new Element(tag, className || "", content === undefined ? "" : String(content));
  const writes = [];
  const storage = { get: (key) => stored[key] ?? null, set: (key, value) => { writes.push([key, value]); stored[key] = value; } };
  let changes = 0;
  const ui = ComputeUI.create({ byId, make, engine, agents: Agents, storage, onChange: () => { changes += 1; } });
  if (catalogValue) ui.setCatalog(catalogValue);
  ui.setReference(schemas, schemas.map((schema) => schema.name));
  return { ui, byId, writes, changes: () => changes };
}
const settle = () => new Promise((resolve) => setImmediate(resolve));
async function running(engine, view) {
  engine.set({ status: "ready", info });
  await settle();
  return view;
}

test("the card says what each state is, and offers only the action that fits it", async () => {
  const cases = [
    [{ status: "unknown" }, "CHECKING", /Looking for the engine files/, { start: true, stop: true }],
    [{ status: "unsupported", error: { kind: "unsupported", message: "This browser has no WebAssembly." } }, "UNSUPPORTED", /no WebAssembly/, { start: true, stop: true }],
    [{ status: "absent", note: "This site was published without the engine files." }, "NOT PUBLISHED", /published without the engine files\. Evidence lookups and the calculator still work/, { start: true, stop: true }],
    [{ status: "idle" }, "OFF", /Runs the project's own Python on this device's CPU, as WebAssembly, in a worker thread.*19\.6 MB/, { start: false, stop: true }],
    [{ status: "loading", progress: { stage: "runtime", label: "Starting the Python runtime (WebAssembly)" } }, "STARTING", /checked against the build manifest/, { start: true, stop: false }],
    [{ status: "error", error: { kind: "integrity", message: "wheels/x.whl does not match the build manifest" } }, "FAILED", /did not start/, { start: false, stop: true }],
  ];
  for (const [state, chip, summary, hidden] of cases) {
    const engine = fakeEngine(state);
    const { byId } = panels(engine);
    assert.equal(byId("engine-chip").textContent, chip, chip);
    assert.match(byId("engine-summary").textContent, summary, chip);
    assert.equal(byId("engine-start").hidden, hidden.start, `${chip}: start button`);
    assert.equal(byId("engine-stop").hidden, hidden.stop, `${chip}: stop button`);
  }
  const loading = panels(fakeEngine({ status: "loading", progress: { stage: "packages", label: "Installing pydantic-2.14.0" } }));
  assert.equal(loading.byId("engine-meter").hidden, false);
  assert.equal(loading.byId("engine-stage").textContent, "Installing pydantic-2.14.0");
  assert.equal(loading.byId("engine-stop").textContent, "Cancel");
  const failed = panels(fakeEngine({ status: "error", error: { kind: "integrity", message: "pyodide/pyodide.asm.mjs could not be fetched, or it does not match the SHA-256 in the build manifest." } }));
  assert.equal(failed.byId("engine-error").hidden, false);
  assert.match(failed.byId("engine-error").textContent, /does not match the SHA-256 in the build manifest/);
  assert.equal(failed.byId("engine-start").textContent, "Try again");
  assert.equal(panels(fakeEngine({ status: "idle" })).byId("engine-start").textContent, "Start on this device · 19.6 MB");
});

test("without the published catalog the engine is not offered, and the card says why", () => {
  const engine = fakeEngine({ status: "idle" });
  const { byId, ui } = panels(engine, { catalogValue: null });
  assert.equal(byId("engine-chip").textContent, "NO CATALOG");
  assert.match(byId("engine-summary").textContent, /agent-catalog\.json\) did not load/);
  assert.equal(byId("engine-start").hidden, true);
  assert.equal(ui.option(), null);
  assert.equal(byId("flow-panel").hidden, true, "no diagram without a catalog to draw it from");
});

test("a running engine is offered only after it proves it holds the agents this page was published with", async () => {
  const engine = fakeEngine({ status: "idle" });
  const { byId, ui } = panels(engine);
  engine.set({ status: "loading", progress: { stage: "runtime", label: "x" } });
  engine.set({ status: "ready", info });
  assert.equal(byId("engine-chip").textContent, "CHECKING", "up is not yet verified");
  assert.equal(ui.option(), null, "nothing is offered while the check is in flight");
  await settle();
  assert.equal(byId("engine-chip").textContent, "RUNNING");
  assert.deepEqual(Object.keys(ui.option()), ["engine", "catalog"]);
  assert.equal(ui.option().engine, engine);
  assert.equal(byId("composer-compute").hidden, false);
  assert.equal(byId("composer-compute").textContent, `${catalog.agents.length} compute agents · this device`);
  assert.equal(byId("engine-offer-label").hidden, false);
  assert.equal(byId("engine-stop").textContent, "Stop");
  assert.deepEqual(byId("engine-facts").children.map((child) => child.textContent), [
    "Runtime", "Python 3.14.2 · Pyodide 314.0.7", "Agents", `${catalog.agents.length} compute agents`, "Integrity", "every engine file checked against its SHA-256",
    "Self-test", "4 calls match the native build", "Started in", "6.4 s", "Build", "dc7634b33bb1"]);
});

test("an engine that does not match the page is shown as a mismatch and never offered", async () => {
  for (const [label, mutate, pattern] of [
    ["a missing agent", (live) => { live.agents.pop(); }, /has no agent/],
    ["an unknown agent", (live) => { live.agents.push({ name: "surprise", description: "x", parameters: {} }); }, /surprise that this page does not list/],
    ["different parameters", (live) => { live.agents[0].parameters = { other: "x" }; }, /takes different parameters/],
  ]) {
    const engine = fakeEngine({ status: "idle" });
    const real = engine.catalog;
    engine.catalog = async () => { const live = await real(); mutate(live); return live; };
    const { byId, ui } = panels(engine);
    await running(engine, null);
    assert.equal(byId("engine-chip").textContent, "MISMATCH", label);
    assert.match(byId("engine-summary").textContent, pattern, label);
    assert.equal(ui.option(), null, label);
    assert.equal(byId("composer-compute").hidden, true, label);
    assert.equal(byId("engine-offer-label").hidden, true, label);
    assert.equal(byId("engine-stop").hidden, false, "it can still be stopped");
  }
  const silent = fakeEngine({ status: "idle" });
  silent.catalog = async () => { throw new Error("worker gone"); };
  const view = panels(silent);
  await running(silent, view);
  assert.equal(view.byId("engine-chip").textContent, "MISMATCH");
  assert.match(view.byId("engine-summary").textContent, /worker gone/);
});

test("the visitor decides whether the model is offered the compute agents", async () => {
  const engine = fakeEngine({ status: "idle" });
  const view = panels(engine);
  await running(engine, view);
  assert.ok(view.ui.option());
  view.byId("engine-offer").checked = false;
  await view.byId("engine-offer").listeners.change();
  assert.equal(view.ui.option(), null);
  assert.equal(view.byId("composer-compute").hidden, true);
  view.byId("engine-offer").checked = true;
  await view.byId("engine-offer").listeners.change();
  assert.ok(view.ui.option());
  assert.ok(view.changes() > 0, "the composer is told whenever what is offered changes");
});

test("starting and stopping go to the engine, and stopping takes the offer away", async () => {
  const engine = fakeEngine({ status: "idle" });
  const view = panels(engine);
  await view.byId("engine-start").click();
  assert.deepEqual(engine.calls, ["load"]);
  await running(engine, view);
  await view.byId("engine-stop").click();
  assert.deepEqual(engine.calls, ["load", "unload"]);
  engine.set({ status: "idle", info: null });
  assert.equal(view.ui.option(), null);
  assert.equal(view.byId("engine-chip").textContent, "OFF");
  const rejecting = fakeEngine({ status: "idle" });
  rejecting.load = async () => { throw new Error("boom"); };
  await assert.doesNotReject(panels(rejecting).ui.start(), "the card shows the failure; the click handler does not throw");
});

test("starting automatically is a remembered convenience, and a broken store changes nothing", () => {
  const stored = {};
  const first = panels(fakeEngine({ status: "idle" }), { stored });
  assert.equal(first.ui.autoStart(), false);
  first.byId("engine-auto").checked = true;
  first.byId("engine-auto").listeners.change();
  assert.deepEqual(first.writes, [["eagent.engine.auto", "on"]]);
  assert.equal(panels(fakeEngine({ status: "idle" }), { stored }).ui.autoStart(), true, "a later visit reads it back");
  first.byId("engine-auto").checked = false;
  first.byId("engine-auto").listeners.change();
  assert.equal(stored["eagent.engine.auto"], "off");
  const engine = fakeEngine({ status: "idle" });
  const elements = new Map();
  const byId = (id) => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  const hostile = { get() { throw new Error("blocked"); }, set() { throw new Error("blocked"); } };
  const ui = ComputeUI.create({ byId, make: (tag, c, t) => new Element(tag, c, t), engine, agents: Agents, storage: hostile });
  assert.equal(ui.autoStart(), false);
  assert.doesNotThrow(() => byId("engine-auto").listeners.change());
});

test("the agents panel lists every agent by kind, and compute agents are off until the engine runs", async () => {
  const engine = fakeEngine({ status: "idle" });
  const view = panels(engine);
  const titles = () => view.byId("agents-groups").children.map((group) => group.children[0].textContent);
  const lookups = bundle.schemas.length, total = lookups + 1 + catalog.agents.length;
  assert.deepEqual(titles(), [`Evidence · read-only lookups · ${lookups}`, "Calculator · 1", `Compute · Python on this device · ${catalog.agents.length}`]);
  assert.equal(view.byId("agents-tag").textContent, `${total} · ROUTED BY YOUR API`);
  const rows = () => find(view.byId("agents-groups"), (node) => classes(node).includes("agent"));
  assert.equal(rows().length, total);
  const off = rows().filter((row) => classes(row).includes("off"));
  assert.deepEqual(off.map((row) => row.children[0].children[1].textContent), catalog.agents.map((agent) => agent.name));
  assert.match(text(off[0]), /needs engine/);
  assert.match(text(off[0]), /Start the compute engine to use it/);
  assert.match(text(off[0]), /Runs eagent\./, "each compute agent says which project function it wraps");
  assert.match(view.byId("agents-note").textContent, /listed so you can see what the controller could route to/);
  const evidence = rows().find((row) => text(row).includes("reference_summary"));
  assert.match(text(evidence), /JS/);
  assert.equal(classes(evidence).includes("off"), false);
  await running(engine, view);
  assert.equal(rows().filter((row) => classes(row).includes("off")).length, 0);
  assert.match(text(rows().find((row) => text(row).includes("align_sequences"))), /WASM · CPU/);
  assert.match(view.byId("agents-note").textContent, /does not run them/);
});

test("the flow diagram and the pipeline map are drawn from the catalog, not written in", () => {
  const view = panels(fakeEngine({ status: "idle" }));
  assert.equal(view.byId("flow-agent-count").textContent, `${bundle.schemas.length + 1 + catalog.agents.length} agents`);
  assert.equal(view.byId("flow-js-count").textContent, String(bundle.schemas.length + 1), "the evidence tools and the calculator");
  assert.equal(view.byId("flow-wasm-count").textContent, String(catalog.agents.length));
  const steps = view.byId("flow-pipeline").children;
  assert.equal(steps.length, 10);
  assert.deepEqual(steps.map((step) => classes(step)[1]), catalog.pipeline.map((step) => step.runs_in_browser));
  const mine = steps.find((step) => text(step).includes("mine sequences"));
  assert.match(text(mine), /needs a program/);
  assert.match(mine.title, /tool_unavailable|never|nothing/i, "the note explains why");
  assert.match(view.byId("flow-pipeline-summary").textContent, /^5 run here, 2 run partly, 3 need an external program that no browser can run\./);
  const counts = {};
  for (const step of catalog.pipeline) counts[step.runs_in_browser] = (counts[step.runs_in_browser] || 0) + 1;
  assert.deepEqual(counts, { runs: 5, partial: 2, needs_program: 3 });
});

test("the device readout reports what the browser says and claims no use of the GPU or NPU", async () => {
  const view = panels(fakeEngine({ status: "idle" }));
  const facts = view.byId("device-facts").children.map((child) => child.textContent);
  assert.deepEqual(facts, ["CPU threads", "8", "Memory", "8 GB or more", "WebAssembly", "yes", "Web Workers", "yes", "WebGPU (GPU)", "available", "WebNN (NPU)", "not offered"]);
  assert.match(view.byId("device-help").textContent, /run on the CPU, as WebAssembly, in one worker thread/);
  assert.match(view.byId("device-help").textContent, /no agent here has a GPU or NPU kernel, so nothing is sent to either/);
  const pending = view.byId("device-gpu").click();
  assert.equal(view.byId("device-gpu").disabled, true, "one adapter request at a time");
  await pending;
  assert.equal(view.byId("device-gpu-result").hidden, false);
  assert.equal(view.byId("device-gpu-result").textContent, "Adapter: apple · metal-3. Present, and unused by the agents.");
  assert.equal(view.byId("device-gpu").disabled, false);
  const none = fakeEngine({ status: "idle" });
  none.gpu = async () => ({ available: false, reason: "WebGPU is present but the browser gave no adapter" });
  const missing = panels(none);
  await missing.byId("device-gpu").click();
  assert.equal(missing.byId("device-gpu-result").textContent, "WebGPU is present but the browser gave no adapter.");
  const odd = fakeEngine({ status: "idle" });
  odd.device = () => ({ cpu_threads: null, memory_gb: 2, webassembly: false, workers: false, webgpu: false, webnn: true, secure_context: true });
  assert.deepEqual(panels(odd).byId("device-facts").children.map((child) => child.textContent),
    ["CPU threads", "not reported", "Memory", "about 2 GB", "WebAssembly", "no", "Web Workers", "no", "WebGPU (GPU)", "not offered", "WebNN (NPU)", "available"]);
});

test("the Python backend mode hides every panel that describes the browser engine", async () => {
  const engine = fakeEngine({ status: "idle" });
  const view = panels(engine);
  const ids = ["engine-panel", "device-panel", "flow-panel", "agents-panel"];
  assert.deepEqual(ids.map((id) => view.byId(id).hidden), [false, false, false, false]);
  view.ui.setActive(false);
  assert.deepEqual(ids.map((id) => view.byId(id).hidden), [true, true, true, true]);
  await running(engine, view);
  assert.deepEqual(ids.map((id) => view.byId(id).hidden), [true, true, true, true], "a state change does not bring them back");
  view.ui.setActive(true);
  assert.deepEqual(ids.map((id) => view.byId(id).hidden), [false, false, false, false]);
});

test("the executor chips the trace uses are defined here, once", () => {
  assert.deepEqual(Object.keys(ComputeUI.CHIP), ["browser-js", "browser-wasm"]);
  assert.equal(ComputeUI.CHIP["browser-wasm"].label, "WASM · CPU");
  assert.match(ComputeUI.CHIP["browser-wasm"].title, /WebAssembly.*CPU.*worker/);
});

test("on a connection without Web Crypto the card says the browser still checked every file, and that the page did not check again", async () => {
  const engine = fakeEngine({ status: "idle" });
  const { byId } = panels(engine);
  engine.set({ status: "ready", info: { ...info, integrity: "unavailable" } });
  await settle();
  const row = byId("engine-facts").children.map((child) => child.textContent);
  const shown = row[row.indexOf("Integrity") + 1];
  assert.match(shown, /checked by the browser/);
  assert.match(shown, /no Web Crypto/);
  assert.doesNotMatch(shown, /^not checked/, "the browser's own check ran, so it must not say nothing was checked");
});
