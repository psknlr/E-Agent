"use strict";

// A stand-in for the engine's Web Worker that runs the real installer and the
// real built runtime in this Node process. Everything the page's client does
// with a worker (module options, message passing, termination) is the same;
// only the thread boundary is missing.

const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

const project = path.resolve(__dirname, "..", "..");
const engineDir = path.join(project, "web", "engine");
const install = require(path.join(project, "web", "engine-install.js"));

let loader;
const loadPyodide = async (options) => {
  loader ??= (await import(pathToFileURL(path.join(engineDir, "pyodide", "pyodide.mjs")).href)).loadPyodide;
  return loader(options);
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

module.exports = { InProcessWorker, built, engineDir, manifest };
