"use strict";

// The browser engine is the project's own Python running as WebAssembly. These
// tests build nothing: they run the files `python scripts/build_engine.py`
// produced, through the same installer the page's worker uses, and hold the
// result to the native interpreter's answers for the same calls.
//
// Without a build (web/engine absent) the tests that need it are skipped, so a
// plain `node --test` works on a fresh checkout. CI sets EAGENT_REQUIRE_ENGINE=1
// so a deploy that failed to build the engine fails here instead of skipping.

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const { pathToFileURL } = require("node:url");

const project = path.resolve(__dirname, "..");
const engineDir = path.join(project, "web", "engine");
const manifestPath = path.join(engineDir, "manifest.json");
const built = fs.existsSync(manifestPath);
const required = Boolean(process.env.EAGENT_REQUIRE_ENGINE);
const skip = built ? false : "web/engine is not built; run python scripts/build_engine.py";
const install = require(path.join(project, "web", "engine-install.js"));

const readManifest = () => JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const readBytes = async (name) => new Uint8Array(fs.readFileSync(path.join(engineDir, name)));

// The installer hands the runtime's two JavaScript modules over as checked bytes
// (see tests/fixtures/engine_inprocess.cjs); this is the loader the tests share.
const { loadPyodide } = require("./fixtures/engine_inprocess.cjs").loaders;
const hostFor = (overrides = {}, events = []) => install.createHost({
  loadPyodide, indexURL: path.join(engineDir, "pyodide") + "/", readBytes, post: (message) => events.push(message), ...overrides,
});

// Pseudo-random residues that do not depend on the platform.
function residues(length, seed) {
  const alphabet = "ACDEFGHIKLMNPQRSTVWY";
  let state = seed >>> 0 || 1;
  let out = "";
  for (let index = 0; index < length; index += 1) {
    state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    out += alphabet[state % 20];
  }
  return out;
}
const base = residues(180, 7);
const edited = base.slice(0, 60) + "AAA" + base.slice(63, 120) + base.slice(124);

// A complete reaction: the controller normalises it for real, then waits for a person.
const SPECIFIED_TASK = [
  "task_id: SPECIFIED_001", "task_mode: enzyme_mining", "reaction:", "  reaction_class: ketone_to_secondary_alcohol",
  "  atom_mapped_reaction_smiles: '[CH3:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1>>[CH3:1][C@H:2]([OH:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1'",
  "  substrate:", "    name: acetophenone", "    isomeric_smiles: CC(=O)c1ccccc1", "  product:", "    name: (S)-1-phenylethanol",
  "    isomeric_smiles: C[C@H](O)c1ccccc1", "    creates_new_stereocenter: true", "",
].join("\n");

const J = JSON.stringify;
const battery = [
  ["align_sequences", { sequence_a: "MKTAYIAKQR", sequence_b: "MKTAYIAKQR" }],
  ["align_sequences", { sequence_a: "MKTAYIAKQRQISFVKSHFSRQ", sequence_b: "MKTAYLAKQRQVSFVKSHFSRQ" }],
  ["align_sequences", { sequence_a: "MKTAYIAKQRQISFVKSHFSRQ", sequence_b: "MKTAYIAKQRSFVKSHFSRQLLL", gap_open: "6", gap_extend: "1" }],
  ["align_sequences", { sequence_a: base, sequence_b: edited }],
  ["align_sequences", { sequence_a: "mkt ayi", sequence_b: "MKTAYI", match: "3", mismatch: "-2" }],
  ["align_sequences", { sequence_a: "MKT", sequence_b: "MK1" }],
  ["align_sequences", { sequence_a: "MKT" }],
  ["sequence_distance", { sequence_a: "ACDEFGHIK", sequence_b: "ACDEFGHIR", method: "hamming" }],
  ["sequence_distance", { sequence_a: "ACDEFGHIK", sequence_b: "ACDEFGH", method: "hamming" }],
  ["sequence_distance", { sequence_a: base, sequence_b: edited }],
  ["sequence_distance", { sequence_a: base, sequence_b: edited, k: "5" }],
  ["wilson_interval", { successes: "0", trials: "1" }],
  ["wilson_interval", { successes: "5", trials: "5" }],
  ["wilson_interval", { successes: "7", trials: "20", confidence: "0.99" }],
  ["wilson_interval", { successes: "8", trials: "7" }],
  ["wilson_interval", { successes: "٣", trials: "5" }],
  ["geometry_measure", { measure: "distance", points: "0,0,0; 3,4,0" }],
  ["geometry_measure", { measure: "angle", points: "1,0,0; 0,0,0; 0,1,0" }],
  ["geometry_measure", { measure: "dihedral", points: "1,0,0; 0,0,0; 0,1,0; 0,1,1" }],
  ["geometry_measure", { measure: "centroid", points: "0,0,0; 2,0,0; 1,3,0" }],
  ["geometry_measure", { measure: "distance", points: "0,0,0" }],
  ["variant_check", { sequence: "MKTAYIAKQR", variants: "K2A, T3S, Y9F" }],
  ["variant_check", { sequence: "MKTAYIAKQR", variants: "K12A; A1G", first_position: "10" }],
  ["variant_check", { sequence: "MKTAYIAKQR", variants: "nonsense" }],
  ["pipeline_catalog", {}],
  ["run_pipeline", { task: "KRED_PILOT_001" }],
  ["run_pipeline", { task_yaml: SPECIFIED_TASK }],
  ["run_pipeline", { task: "NO_SUCH_TASK" }],
  ["task_gate_report", { task: "KRED_PILOT_001" }],
  ["task_gate_report", { task: "NO_SUCH_TASK" }],
  ["task_gate_report", { task_yaml: "task_id: T1\n" }],
  ["task_gate_report", { task_yaml: "a: &x [1]\nb: *x\n" }],
  ["no_such_agent", {}],
  ["wilson_interval", { successes: "1", trials: "2", extra: "x" }],
].map(([agent, args]) => [agent, J(args)]);
battery.push(["wilson_interval", "{not json"], ["wilson_interval", "[]"], ["wilson_interval", J({ successes: 3, trials: "5" })],
  ["align_sequences", J({ sequence_a: "M".repeat(1300), sequence_b: "M" })]);

function native(calls) {
  const python = process.env.PYTHON || "python3";
  const run = spawnSync(python, ["-I", path.join(__dirname, "fixtures", "engine_native.py")], { input: J(calls), encoding: "utf8", maxBuffer: 1 << 26 });
  assert.equal(run.status, 0, run.stderr);
  return JSON.parse(run.stdout);
}

function nativeCatalog() {
  const run = spawnSync(process.env.PYTHON || "python3", ["-I", "-c",
    "import sys; sys.path.insert(0, 'src'); from eagent.engine import catalog; print(catalog())"], { cwd: project, encoding: "utf8" });
  assert.equal(run.status, 0, run.stderr);
  return JSON.parse(run.stdout);
}

// Same shape, floats equal to rounding: the two interpreters agree on every
// operation used here except that libm may differ in the last digit.
function same(left, right, where = "$") {
  if (typeof left === "number" && typeof right === "number") {
    const tolerance = 1e-9 * Math.max(1, Math.abs(left), Math.abs(right));
    assert.ok(Math.abs(left - right) <= tolerance, `${where}: ${left} against ${right}`);
  } else if (Array.isArray(left) && Array.isArray(right)) {
    assert.equal(left.length, right.length, `${where}: array length`);
    left.forEach((item, index) => same(item, right[index], `${where}[${index}]`));
  } else if (left && right && typeof left === "object" && typeof right === "object") {
    assert.deepEqual(Object.keys(left).sort(), Object.keys(right).sort(), `${where}: keys`);
    for (const key of Object.keys(left)) same(left[key], right[key], `${where}.${key}`);
  } else assert.equal(left, right, where);
}
const exact = (value) => typeof value === "number" ? Number.isInteger(value)
  : Array.isArray(value) ? value.every(exact)
    : value && typeof value === "object" ? Object.values(value).every(exact) : true;

test("the engine build is present when CI requires it", () => {
  if (required) assert.ok(built, "EAGENT_REQUIRE_ENGINE is set but web/engine/manifest.json does not exist");
});

test("the built engine installs, passes its self-test, and runs every agent as the native interpreter does", { skip }, async () => {
  const events = [];
  const handle = hostFor({}, events);
  const manifest = readManifest();
  const ready = await handle({ id: 1, type: "init", manifest });
  assert.equal(ready.type, "ready", ready.message);
  assert.equal(ready.info.integrity, "verified");
  assert.equal(ready.info.selftest, manifest.selftest.length);
  assert.match(ready.info.python, /^3\.14\./);
  assert.deepEqual(events.map((event) => event.stage).filter((stage, index, all) => all.indexOf(stage) === index),
    ["runtime", "packages", "project", "check"]);

  const listed = JSON.parse((await handle({ id: 2, type: "catalog" })).json);
  assert.equal(listed.schema, "eagent.engine.v1");
  assert.deepEqual(listed.agents.map((agent) => agent.name), ready.info.agents);
  same(listed, nativeCatalog());

  const expected = native(battery);
  assert.equal(expected.length, battery.length);
  let compared = 0;
  for (const [index, [agent, args]] of battery.entries()) {
    const reply = await handle({ id: 100 + index, type: "call", agent, args });
    assert.equal(reply.type, "result", `${agent} ${args.slice(0, 60)}: ${reply.message}`);
    const here = JSON.parse(reply.json);
    const there = JSON.parse(expected[index]);
    const label = `${agent} ${args.slice(0, 70)}`;
    assert.equal(here.ok, there.ok, label);
    if (!here.ok) {
      assert.equal(here.kind, there.kind, label);
      assert.equal(here.refusal, there.refusal, label);
      assert.equal(here.value, undefined, `${label}: a refusal carries no value`);
      assert.equal(here.cite, undefined, `${label}: a refusal carries no citation`);
    } else {
      same(here.value, there.value, label);
      assert.equal(here.cite.artifact, "engine_compute");
      assert.equal(here.cite.row, `${agent}-${here.cite.sha256.slice(0, 16)}`);
      assert.equal(here.wraps, there.wraps, label);
      if (exact(here.value)) assert.equal(here.cite.sha256, there.cite.sha256, `${label}: identical bytes give an identical hash`);
    }
    compared += 1;
  }
  assert.equal(compared, battery.length);
});

test("a call before the engine starts, and a malformed message, are refused", async () => {
  const handle = hostFor();
  assert.equal((await handle({ id: 1, type: "call", agent: "wilson_interval", args: "{}" })).kind, "not_ready");
  assert.equal((await handle({ id: 2, type: "catalog" })).kind, "not_ready");
  assert.equal((await handle({ id: 3, type: "bogus" })).kind, "not_ready");
  assert.equal((await handle(null)).kind, "protocol");
});

test("a corrupted package is refused before it is installed, and the engine stays unusable", { skip }, async () => {
  const wheel = readManifest().install.wheels.find((name) => name.includes("pydantic-2"));
  const flipped = async (name) => {
    const bytes = await readBytes(name);
    if (name === wheel) bytes[bytes.length - 40] ^= 0xff;
    return bytes;
  };
  const handle = hostFor({ readBytes: flipped });
  const refused = await handle({ id: 1, type: "init", manifest: readManifest() });
  assert.equal(refused.type, "error");
  assert.equal(refused.kind, "integrity");
  assert.ok(refused.message.includes(wheel), refused.message);
  assert.equal((await handle({ id: 2, type: "call", agent: "wilson_interval", args: J({ successes: "1", trials: "2" }) })).kind, "not_ready");

  const short = hostFor({ readBytes: async (name) => (await readBytes(name)).slice(0, name === wheel ? 100 : undefined) });
  const truncated = await short({ id: 1, type: "init", manifest: readManifest() });
  assert.equal(truncated.kind, "integrity");
  assert.match(truncated.message, /100 bytes/);
});

test("a runtime that computes a different answer from the build is not used", { skip }, async () => {
  const manifest = readManifest();
  manifest.selftest[1] = { ...manifest.selftest[1], sha256: "0".repeat(64) };
  const refused = await hostFor()({ id: 1, type: "init", manifest });
  assert.equal(refused.type, "error");
  assert.equal(refused.kind, "selftest");
  assert.match(refused.message, /align_sequences computed a different answer/);
});

test("an engine started twice is one engine", { skip }, async () => {
  const events = [];
  const handle = hostFor({}, events);
  const manifest = readManifest();
  const [first, second] = await Promise.all([handle({ id: 1, type: "init", manifest }), handle({ id: 2, type: "init", manifest })]);
  assert.equal(first.type, "ready");
  assert.equal(second.type, "ready");
  assert.equal(events.filter((event) => event.stage === "runtime").length, 1, "the runtime started once");
});

test("the manifest is validated before anything is read", () => {
  const good = built ? readManifest() : {
    schema: "eagent.engine.manifest.v1", engine_schema: "eagent.engine.v1",
    install: { wheels: ["wheels/a.whl"], site_packages_zips: [], project: "p.zip" },
    selftest: [{ agent: "x", arguments: "{}", sha256: "a".repeat(64) }],
    files: Object.fromEntries(["wheels/a.whl", "p.zip", "pyodide/pyodide.mjs", "pyodide/pyodide.asm.mjs", "pyodide/pyodide.asm.wasm", "pyodide/python_stdlib.zip"]
      .map((name) => [name, { bytes: 1, sha256: "b".repeat(64) }])),
  };
  assert.doesNotThrow(() => install.validateManifest(good));
  const broken = (change) => { const copy = structuredClone(good); change(copy); return copy; };
  for (const [label, manifest, pattern] of [
    ["not an object", null, /not an object/],
    ["wrong schema", broken((m) => { m.schema = "x"; }), /expected schema/],
    ["built for another page", broken((m) => { m.engine_schema = "eagent.engine.v0"; }), /this page speaks/],
    ["a path that climbs out", broken((m) => { m.files["../secret"] = { bytes: 1, sha256: "c".repeat(64) }; }), /unsafe path/],
    ["an absolute path", broken((m) => { m.files["/etc/passwd"] = { bytes: 1, sha256: "c".repeat(64) }; }), /unsafe path/],
    ["a bad hash", broken((m) => { m.files["p.zip"] = { bytes: 1, sha256: "zz" }; m.files[m.install.project] = { bytes: 1, sha256: "zz" }; }), /no valid size and SHA-256/],
    ["an install of an unlisted file", broken((m) => { m.install.wheels.push("wheels/missing.whl"); }), /which it does not list/],
    ["no second loader module", broken((m) => { delete m.files["pyodide/pyodide.asm.mjs"]; }), /does not list pyodide\/pyodide\.asm\.mjs/],
    ["no self-test", broken((m) => { m.selftest = []; }), /no self-test/],
    ["a malformed self-test", broken((m) => { m.selftest = [{ agent: 1 }]; }), /malformed self-test/],
  ]) assert.throws(() => install.validateManifest(manifest), pattern, label);
});

// ---- the runtime's two JavaScript modules ----------------------------------------
// An import() fetches a file itself, outside the installer's check, and runs
// whatever the host serves at that moment. So the installer reads and checks both
// modules like any other file and hands the runtime their bytes; a module that
// does not match the manifest is never given to it. These tests need no runtime:
// the loader is a stand-in that records what it was handed.
const crypto = require("node:crypto");
const sha256 = (bytes) => crypto.createHash("sha256").update(bytes).digest("hex");

function loaderFixture(tamper = {}) {
  const content = {};
  const names = ["wheels/a.whl", "p.zip", "pyodide/pyodide.mjs", "pyodide/pyodide.asm.mjs", "pyodide/pyodide.asm.wasm", "pyodide/python_stdlib.zip"];
  for (const name of names) content[name] = new TextEncoder().encode(`// ${name}\nexport default 1;\n`);
  const manifest = {
    schema: "eagent.engine.manifest.v1", engine_schema: "eagent.engine.v1",
    install: { wheels: ["wheels/a.whl"], site_packages_zips: [], project: "p.zip" },
    selftest: [{ agent: "x", arguments: "{}", sha256: "a".repeat(64) }],
    files: Object.fromEntries(names.map((name) => [name, { bytes: content[name].length, sha256: sha256(content[name]) }])),
  };
  const order = [];
  const served = (name) => (tamper[name] ? tamper[name](content[name]) : content[name]);
  const calls = [];
  const stop = new Error("stop after the runtime is handed its modules");
  return {
    manifest, order, calls, stop, content,
    run: () => install.install({
      manifest, indexURL: "https://site.example/engine/pyodide/",
      prepare: () => order.push("prepare"),
      readBytes: async (name) => { order.push(`read ${name}`); return served(name); },
      loadPyodide: async (options, given, modules) => { order.push("load"); calls.push({ options, given, modules }); throw stop; },
    }),
  };
}

test("the runtime is handed the exact bytes that were checked, after the browser's check was arranged", async () => {
  const fixture = loaderFixture();
  await assert.rejects(fixture.run(), (error) => error === fixture.stop);
  assert.deepEqual(fixture.order.slice(0, 4), ["prepare", "read pyodide/pyodide.mjs", "read pyodide/pyodide.asm.mjs", "load"],
    "fetch is wrapped before the first file is read, and the modules are read before the runtime starts");
  const { modules, given } = fixture.calls[0];
  assert.deepEqual(Object.keys(modules).sort(), ["asm", "runtime"]);
  assert.equal(sha256(modules.runtime), fixture.manifest.files["pyodide/pyodide.mjs"].sha256);
  assert.equal(sha256(modules.asm), fixture.manifest.files["pyodide/pyodide.asm.mjs"].sha256);
  assert.equal(given, fixture.manifest);
});

test("a loader module that does not match the manifest is never given to the runtime", async () => {
  const flip = (bytes) => { const copy = Uint8Array.from(bytes); copy[copy.length - 3] ^= 1; return copy; };
  const longer = (bytes) => Uint8Array.from([...bytes, ...new TextEncoder().encode("fetch('https://evil.example/');")]);
  for (const [label, tamper] of [
    ["the second module altered", { "pyodide/pyodide.asm.mjs": flip }],
    ["the second module extended", { "pyodide/pyodide.asm.mjs": longer }],
    ["the first module altered", { "pyodide/pyodide.mjs": flip }],
    ["the first module extended", { "pyodide/pyodide.mjs": longer }],
  ]) {
    const fixture = loaderFixture(tamper);
    await assert.rejects(fixture.run(), (error) => error.kind === "integrity" && /does not match the build manifest|bytes; the build recorded/.test(error.message), label);
    assert.equal(fixture.calls.length, 0, `${label}: the runtime must not have been started`);
  }
});

test("the runtime's own downloads get the browser's integrity check from the manifest", () => {
  const manifest = { files: { "pyodide/pyodide.asm.wasm": { bytes: 1, sha256: "00".repeat(32) } } };
  const seen = [];
  const wrapped = install.integrityFetch((input, init) => { seen.push([input, init]); return "response"; }, "https://site.example/engine/", manifest);
  assert.equal(wrapped("https://site.example/engine/pyodide/pyodide.asm.wasm?x=1", { credentials: "omit" }), "response");
  assert.deepEqual(seen[0][1], { credentials: "omit", integrity: "sha256-" + Buffer.alloc(32).toString("base64"), cache: "no-cache" });
  // The runtime asks for some files with a URL object and others with a string.
  wrapped(new URL("https://site.example/engine/pyodide/pyodide.asm.wasm"));
  assert.equal(seen[1][1].integrity, seen[0][1].integrity, "a URL object is covered too");
  wrapped({ url: "https://site.example/engine/pyodide/pyodide.asm.wasm" }, { method: "GET" });
  assert.equal(seen[2][1].integrity, seen[0][1].integrity, "and so is a Request");
  assert.equal(seen[2][1].method, "GET", "the caller's own options are kept");
  seen.length = 0;
  wrapped("https://site.example/engine/pyodide/other.bin");
  wrapped("https://elsewhere.example/engine/pyodide/pyodide.asm.wasm");
  assert.equal(seen[0][1], undefined, "a file the manifest does not list is left alone");
  assert.equal(seen[1][1], undefined, "and so is another host's");
  assert.equal(install.hexToBase64("ff00"), "/wA=");
});
