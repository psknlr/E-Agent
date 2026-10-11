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

// A module that will not import is almost always a host serving .mjs with a
// type browsers refuse for scripts. Say so, rather than the browser's one line.
async function explainImportFailure(error, EngineError) {
  let type = '';
  try {
    type = (await nativeFetch(BASE + 'pyodide/pyodide.mjs', {method: 'HEAD'})).headers.get('content-type') || '';
  } catch (ignored) { /* fall through to the generic message */ }
  if (type && !/javascript|ecmascript/i.test(type)) {
    return new EngineError('mime', `This host serves .mjs files as "${type}", and browsers only run module scripts served `
      + 'as JavaScript. Map the .mjs extension to text/javascript in the host\'s settings '
      + '(nginx: types { text/javascript mjs; }  Apache: AddType text/javascript .mjs).');
  }
  return new EngineError('runtime', `The Python runtime could not be loaded: ${error.message}`);
}

const ready = import('./engine-install.js').then(() => {
  const {EngineError, createHost, integrityFetch} = self.EAgentEngineInstall;
  return createHost({
    indexURL: BASE + 'pyodide/',
    // The runtime fetches its own WebAssembly binary and standard library.
    // Wrapping fetch for this worker makes the browser check those downloads, and
    // the ones in readBytes below, against the manifest's hashes.
    loadPyodide: async (options, manifest) => {
      if (!wrapped) self.fetch = integrityFetch(nativeFetch, BASE, manifest);
      wrapped = true;
      let runtime;
      try {
        runtime = await import(BASE + 'pyodide/pyodide.mjs');
      } catch (error) {
        throw await explainImportFailure(error, EngineError);
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
        return await Promise.race([runtime.loadPyodide(options), instantiation]);
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
