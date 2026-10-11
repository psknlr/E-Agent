"use strict";

// A stand-in for the engine's Web Worker that runs the real installer and the
// real built runtime in this Node process. Everything the page's client does
// with a worker (module options, message passing, termination) is the same;
// only the thread boundary is missing.

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

const project = path.resolve(__dirname, "..", "..");
const engineDir = path.join(project, "web", "engine");
const install = require(path.join(project, "web", "engine-install.js"));

// The installer hands the runtime's two JavaScript modules over as checked bytes,
// as it does in the real worker. A worker imports them from a Blob URL; Node
// cannot, and the asm module needs a file URL of its own, so the bytes go to a
// temporary directory and are imported from there.
let loaded;
const loadPyodide = async (options, manifest, modules) => {
  loaded ??= (async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "eagent-engine-"));
    process.on("exit", () => fs.rmSync(dir, { recursive: true, force: true }));
    const bring = async (name, bytes) => {
      const file = path.join(dir, name);
      fs.writeFileSync(file, bytes);
      return import(pathToFileURL(file).href);
    };
    const runtime = await bring("pyodide.mjs", modules.runtime);
    const asm = await bring("pyodide.asm.mjs", modules.asm);
    return { loadPyodide: runtime.loadPyodide, createPyodideModule: asm.default };
  })();
  const { loadPyodide: load, createPyodideModule } = await loaded;
  return load({ ...options, createPyodideModule });
};

class InProcessWorker {
  constructor() {
    this.terminated = false;
    this.host = install.createHost({
      indexURL: path.join(engineDir, "pyodide") + "/",
      loadPyodide,
      readBytes: async (name) => new Uint8Array(fs.readFileSync(path.join(engineDir, name))),
      post: (message) => setImmediate(() => !this.terminated && this.onmessage({ data: message })),
    });
  }

  postMessage(message) {
    // Structured clone, as a real worker boundary would do.
    const copy = JSON.parse(JSON.stringify(message));
    this.host(copy).then((reply) => setImmediate(() => !this.terminated && this.onmessage({ data: reply })));
  }

  terminate() { this.terminated = true; }
}

const manifest = () => JSON.parse(fs.readFileSync(path.join(engineDir, "manifest.json"), "utf8"));
const built = () => fs.existsSync(path.join(engineDir, "manifest.json"));

module.exports = { InProcessWorker, built, engineDir, manifest, loaders: { loadPyodide } };
