/* The engine's Web Worker: the project's Python, as WebAssembly, off the page's thread.
 *
 * Everything that matters is in engine-install.js, which the tests run too. This
 * file only says where the build lives, how to read it, and how messages travel.
 *
 * It must be started as a module worker (the runtime's loader refuses to run in
 * a classic one), yet it has no static `import`: CI syntax-checks every web/*.js
 * as a plain script, and a worker that cannot be checked is a worker nobody
 * reviews. The installer is loaded with a dynamic import() instead, which sets
 * EAgentEngineInstall on this worker's global scope.
 */
'use strict';

const BASE = new URL('engine/', self.location.href).href;
const nativeFetch = self.fetch.bind(self);
let wrapped = false;

// The runtime's two JavaScript modules are run from the bytes the installer has
// already checked against the manifest, never imported from their URLs. An
// import() fetches the file itself, outside any check, and would run whatever
// the host served at that moment. This also means the host's MIME type for .mjs
// does not matter: the bytes are given a JavaScript type here.
async function importChecked(bytes) {
  const url = URL.createObjectURL(new Blob([bytes], {type: 'text/javascript'}));
  try {
    return await import(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

const ready = import('./engine-install.js').then(() => {
  const {EngineError, createHost, integrityFetch} = self.EAgentEngineInstall;
  return createHost({
    indexURL: BASE + 'pyodide/',
    // The runtime fetches its own WebAssembly binary and standard library.
    // Wrapping fetch for this worker makes the browser check those downloads, and
    // the ones in readBytes below, against the manifest's hashes. It is done before
    // the first file is read, so the loader modules are covered too.
    prepare: manifest => {
      if (!wrapped) self.fetch = integrityFetch(nativeFetch, BASE, manifest);
      wrapped = true;
    },
    loadPyodide: async (options, manifest, modules) => {
      let runtime, asm;
      try {
        runtime = await importChecked(modules.runtime);
        asm = await importChecked(modules.asm);
      } catch (error) {
        throw new EngineError('runtime', `The Python runtime could not be loaded: ${error.message}. If this host sends a `
          + 'Content-Security-Policy, its script-src must allow blob: (the runtime is run from bytes this page has checked) '
          + 'and \'wasm-unsafe-eval\' (it compiles WebAssembly).');
      }
      // When the WebAssembly binary cannot be fetched or compiled, the runtime
      // logs "wasm instantiation failed!" and then waits forever. Hear the log, and
      // fail now with the reason instead of after the page's five-minute time-out.
      const warn = console.warn;
      let stalled;
      const instantiation = new Promise((resolve, reject) => {
        stalled = reject;
        console.warn = (...args) => {
          warn.apply(console, args);
          if (String(args[0]).startsWith('wasm instantiation failed')) {
            reject(new EngineError('runtime', 'The WebAssembly runtime could not be started: its binary was refused '
              + '(it may not match the build manifest) or this browser cannot run it. The browser console has the detail.'));
          }
        };
      });
      try {
        // createPyodideModule hands the runtime its (checked) second module, so it
        // does not import pyodide.asm.mjs from a URL itself.
        return await Promise.race([runtime.loadPyodide({...options, createPyodideModule: asm.default}), instantiation]);
      } finally {
        console.warn = warn;
        stalled(new Error('settled'));
        instantiation.catch(() => {});
      }
    },
    readBytes: async name => {
      let response;
      try {
        response = await self.fetch(BASE + name);
      } catch (error) {
        // The browser reports a failed integrity check as a bare network error;
        // its console names the file and both hashes.
        throw new EngineError('integrity', `${name} could not be fetched, or it does not match the SHA-256 in the build `
          + 'manifest. The deployed engine files probably do not all come from one build; reload to fetch them again.');
      }
      if (!response.ok) throw new EngineError('missing', `${name} could not be fetched (HTTP ${response.status})`);
      return new Uint8Array(await response.arrayBuffer());
    },
    post: message => self.postMessage(message),
  });
});

self.onmessage = async event => {
  let host;
  try {
    host = await ready;
  } catch (error) {
    self.postMessage({id: event.data && event.data.id, type: 'error', kind: 'worker',
      message: 'engine-install.js could not be loaded next to engine-worker.js'});
    return;
  }
  self.postMessage(await host(event.data));
};
