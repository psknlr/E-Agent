"use strict";

// The engine's worker, in a real browser.
//
// Everything else about the engine is tested in Node against the same built
// files, but a Web Worker, a Blob URL import and the browser's own integrity
// check exist only in a browser, and they are what a tampered or half-deployed
// engine has to get past. This starts the real page code against a small host
// that can change a file's bytes or type, and checks what the visitor would see.
//
// It is skipped unless Playwright with a Chromium is installed (nothing in the
// repository depends on it) and web/engine has been built. CI sets neither.

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const project = path.resolve(__dirname, "..");
const root = path.join(project, "web");

function findChromium() {
  let playwright;
  try { playwright = require("playwright"); } catch (_) { return { reason: "playwright is not installed" }; }
  const candidates = [process.env.EAGENT_CHROMIUM];
  try { candidates.push(playwright.chromium.executablePath()); } catch (_) { /* no default build */ }
  const browsers = process.env.PLAYWRIGHT_BROWSERS_PATH;
  if (browsers && fs.existsSync(browsers)) {
    for (const entry of fs.readdirSync(browsers).filter((name) => name.startsWith("chromium")).sort().reverse()) {
      candidates.push(path.join(browsers, entry, "chrome-linux", "chrome"), path.join(browsers, entry, "chrome-linux64", "chrome"));
    }
  }
  const executablePath = candidates.find((candidate) => candidate && fs.existsSync(candidate));
  return executablePath ? { playwright, executablePath } : { reason: "no Chromium was found (set EAGENT_CHROMIUM)" };
}

const found = findChromium();
const built = fs.existsSync(path.join(root, "engine", "manifest.json"));
const skip = !built ? "web/engine is not built; run python scripts/build_engine.py" : found.reason || false;

const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm",
  ".zip": "application/zip", ".whl": "application/octet-stream", ".css": "text/css", ".txt": "text/plain", ".svg": "image/svg+xml" };
const HARNESS = `<!doctype html><meta charset="utf-8"><title>engine harness</title>
<script src="engine-install.js"></script><script src="engine-client.js"></script>
<script>window.engine = EAgentEngineClient.create();</script>`;

// A static host that can be told to misbehave.
function host() {
  const faults = { mutate: {}, type: {}, headers: {}, requests: [] };
  const server = http.createServer((request, response) => {
    const relative = decodeURIComponent(new URL(request.url, "http://host.invalid").pathname);
    faults.requests.push(relative);
    if (relative === "/harness.html") { response.writeHead(200, { "content-type": TYPES[".html"] }); return response.end(HARNESS); }
    const file = path.join(root, relative);
    if (!file.startsWith(root + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) { response.writeHead(404); return response.end("not found"); }
    let body = fs.readFileSync(file);
    if (faults.mutate[relative]) body = faults.mutate[relative](Buffer.from(body));
    response.writeHead(200, { "content-type": faults.type[relative] || TYPES[path.extname(file)] || "application/octet-stream", "cache-control": "no-store", ...faults.headers[relative] });
    response.end(body);
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve({ server, faults, url: `http://127.0.0.1:${server.address().port}/` })));
}

let browser, site;
test.before(async () => {
  if (skip) return;
  browser = await found.playwright.chromium.launch({ executablePath: found.executablePath, args: ["--no-sandbox"] });
  site = await host();
});
test.after(async () => {
  if (browser) await browser.close();
  if (site) site.server.close();
});

// Open the page, start the engine, and report what the visitor would see. An
// optional async function (engine) => value runs in the page once it is ready.
async function start(afterwards) {
  const context = await browser.newContext();
  try {
    const page = await context.newPage();
    await page.goto(site.url + "harness.html");
    return await page.evaluate(async (script) => {
      await engine.probe();
      let outcome;
      try {
        const info = await engine.load();
        outcome = { ok: true, status: engine.state.status, integrity: info.integrity, selftest: info.selftest };
      } catch (error) {
        outcome = { ok: false, status: engine.state.status, kind: error.kind, message: error.message };
      }
      if (outcome.ok && script) outcome.then = await new Function("engine", `return (${script})(engine)`)(engine);
      return outcome;
    }, afterwards ? afterwards.toString() : "");
  } finally {
    await context.close();
  }
}

const withCode = (bytes) => Buffer.concat([bytes, Buffer.from("\n;fetch('/__injected_code_ran');\n")]);
const flipped = (bytes) => { bytes[5000] ^= 1; return bytes; };
const reset = () => { site.faults.mutate = {}; site.faults.type = {}; site.faults.headers = {}; site.faults.requests.length = 0; };

test("the real worker starts the engine, checks its files, and answers a call", { skip }, async () => {
  reset();
  const outcome = await start(async (engine) => JSON.parse(await engine.call("wilson_interval", JSON.stringify({ successes: "7", trials: "20" }))));
  assert.equal(outcome.ok, true, JSON.stringify(outcome));
  assert.equal(outcome.status, "ready");
  assert.equal(outcome.integrity, "verified");
  assert.equal(outcome.selftest, 4);
  assert.equal(outcome.then.ok, true);
  assert.equal(outcome.then.value.successes, 7);
  assert.match(outcome.then.cite.row, /^wilson_interval-[0-9a-f]{16}$/);
});

for (const [label, file] of [
  ["the runtime's second JavaScript module", "/engine/pyodide/pyodide.asm.mjs"],
  ["the runtime's first JavaScript module", "/engine/pyodide/pyodide.mjs"],
]) {
  test(`${label} with code appended is refused, and the code never runs`, { skip }, async () => {
    reset();
    site.faults.mutate[file] = withCode;
    const outcome = await start();
    assert.equal(outcome.ok, false, "a changed module must not start the engine");
    assert.equal(outcome.kind, "integrity");
    assert.equal(outcome.status, "error");
    assert.match(outcome.message, /does not match the SHA-256 in the build manifest/);
    assert.ok(!site.faults.requests.includes("/__injected_code_ran"), "the appended code ran");
  });
}

test("one flipped byte in the runtime's second module is refused", { skip }, async () => {
  reset();
  site.faults.mutate["/engine/pyodide/pyodide.asm.mjs"] = flipped;
  const outcome = await start();
  assert.equal(outcome.ok, false);
  assert.equal(outcome.kind, "integrity");
});

test("a package or the project altered on the host is refused", { skip }, async () => {
  for (const file of ["/engine/eagent-project.zip", "/engine/wheels/pydantic-2.14.0-py3-none-any.whl"]) {
    reset();
    site.faults.mutate[file] = (bytes) => { bytes[bytes.length - 40] ^= 1; return bytes; };
    const outcome = await start();
    assert.equal(outcome.ok, false, file);
    assert.equal(outcome.kind, "integrity", file);
  }
});

test("the type a host gives the .mjs files does not matter, because they run from checked bytes", { skip }, async () => {
  reset();
  site.faults.type["/engine/pyodide/pyodide.mjs"] = "application/octet-stream";
  site.faults.type["/engine/pyodide/pyodide.asm.mjs"] = "text/plain";
  const outcome = await start();
  assert.equal(outcome.ok, true, JSON.stringify(outcome));
});

test("cancelling a start in the real worker returns to off, and the next start still works", { skip }, async () => {
  reset();
  const context = await browser.newContext();
  try {
    const page = await context.newPage();
    await page.goto(site.url + "harness.html");
    const outcome = await page.evaluate(async () => {
      await engine.probe();
      const statuses = [];
      engine.subscribe((state) => statuses.push(state.status));
      const first = engine.load().then(() => "started", (error) => error.kind);
      await new Promise((resolve) => setTimeout(resolve, 600));
      const during = engine.state.status;
      engine.unload();
      const result = await first;
      await new Promise((resolve) => setTimeout(resolve, 50));
      const after = { status: engine.state.status, error: engine.state.error };
      const info = await engine.load();
      return { during, result, after, statuses, again: engine.state.status, selftest: info.selftest };
    });
    assert.equal(outcome.during, "loading");
    assert.equal(outcome.result, "stopped");
    assert.deepEqual(outcome.after, { status: "idle", error: null }, "a cancel is not a failure");
    assert.ok(!outcome.statuses.slice(0, outcome.statuses.indexOf("idle") + 1).includes("error"), outcome.statuses.join(","));
    assert.equal(outcome.again, "ready");
    assert.equal(outcome.selftest, 4);
  } finally {
    await context.close();
  }
});

test("a Content-Security-Policy that forbids blob: scripts is reported with its remedy, and one that allows them works", { skip }, async () => {
  reset();
  site.faults.headers["/engine-worker.js"] = { "content-security-policy": "script-src 'self' 'wasm-unsafe-eval'" };
  const blocked = await start();
  assert.equal(blocked.ok, false, JSON.stringify(blocked));
  assert.equal(blocked.kind, "runtime");
  assert.match(blocked.message, /Content-Security-Policy/);
  assert.match(blocked.message, /allow blob:/);
  reset();
  site.faults.headers["/engine-worker.js"] = { "content-security-policy": "script-src 'self' blob: 'wasm-unsafe-eval'" };
  const allowed = await start();
  assert.equal(allowed.ok, true, JSON.stringify(allowed));
});
