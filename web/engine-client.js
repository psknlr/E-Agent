/* The page's side of the engine: find it, start it when asked, call it, stop it.
 *
 * The engine is the project's Python running as WebAssembly in a Web Worker, so
 * the page never blocks on it and a call that runs away can be cancelled by
 * ending the worker. Agents are stateless, which is what makes that safe: nothing
 * is lost, and pressing Start (load) runs a fresh worker. The client does not
 * restart one on its own; the card says the engine stopped and why.
 *
 * Nothing here talks to a model or a network other than this site's own files.
 */
(function (scope) {
  'use strict';

  class EngineClientError extends Error {
    constructor(kind, message) {
      super(message);
      this.name = 'EngineClientError';
      this.kind = kind;
    }
  }

  // What the visitor's device offers, read from the browser and never sent
  // anywhere. `webgpu` and `webnn` say the API exists, not that anything uses it.
  function describeDevice(nav = scope.navigator || {}, global = scope) {
    return {
      cpu_threads: Number.isInteger(nav.hardwareConcurrency) ? nav.hardwareConcurrency : null,
      // Chromium reports this rounded down to a power of two and capped at 8.
      memory_gb: typeof nav.deviceMemory === 'number' ? nav.deviceMemory : null,
      webassembly: typeof global.WebAssembly === 'object',
      workers: typeof global.Worker === 'function',
      webgpu: Boolean(nav.gpu),
      webnn: Boolean(nav.ml),
      secure_context: global.isSecureContext !== false,
    };
  }

  // The adapter the browser would give a WebGPU workload, if there is one. A
  // page that never asks for it never learns it, so this is only called when the
  // visitor opens the device readout.
  async function describeGpu(nav = scope.navigator || {}, timeoutMs = 3000) {
    if (!nav.gpu || typeof nav.gpu.requestAdapter !== 'function') return {available: false, reason: 'WebGPU is not offered by this browser'};
    let timer;
    try {
      const adapter = await Promise.race([
        nav.gpu.requestAdapter(),
        new Promise((resolve, reject) => { timer = setTimeout(() => reject(new Error('timed out')), timeoutMs); }),
      ]);
      if (!adapter) return {available: false, reason: 'WebGPU is present but the browser gave no adapter'};
      const info = adapter.info || {};
      return {
        available: true,
        vendor: String(info.vendor || ''),
        architecture: String(info.architecture || ''),
        description: String(info.description || ''),
      };
    } catch (error) {
      return {available: false, reason: `WebGPU adapter request failed: ${error.message}`};
    } finally {
      clearTimeout(timer);
    }
  }

  function create(options = {}) {
    // An option that is present wins even when it is undefined, so a caller (or a
    // test) can say "this browser has no workers" instead of getting the real ones.
    const given = name => Object.prototype.hasOwnProperty.call(options, name);
    const env = {
      fetch: given('fetch') ? options.fetch : (typeof scope.fetch === 'function' ? scope.fetch.bind(scope) : null),
      Worker: given('Worker') ? options.Worker : scope.Worker,
      WebAssembly: given('WebAssembly') ? options.WebAssembly : scope.WebAssembly,
      base: options.base || 'engine/',
      workerUrl: options.workerUrl || 'engine-worker.js',
      callTimeoutMs: options.callTimeoutMs || 60000,
      loadTimeoutMs: options.loadTimeoutMs || 300000,
    };
    const listeners = new Set();
    const pending = new Map();
    let worker = null;
    let sequence = 0;
    let loading = null;
    // Every start, and every unload, begins a new generation. A start still in
    // flight when the user cancels it belongs to an older one, and must not touch
    // the state or the worker of whatever came after.
    let generation = 0;
    let queue = Promise.resolve();
    let state = {status: 'unknown', manifest: null, info: null, progress: null, error: null, note: ''};

    const set = changes => {
      state = Object.assign({}, state, changes);
      for (const listener of listeners) {
        try { listener(state); } catch (error) { if (scope.console) scope.console.error(error); }
      }
    };

    const supported = () => {
      if (typeof env.Worker !== 'function') return 'This browser has no Web Workers.';
      if (!env.WebAssembly || typeof env.WebAssembly.instantiate !== 'function') return 'This browser has no WebAssembly.';
      return '';
    };

    function settleAll(error) {
      for (const [id, entry] of pending) {
        clearTimeout(entry.timer);
        pending.delete(id);
        entry.reject(error);
      }
    }

    function stop(error) {
      if (worker) {
        worker.onmessage = null;
        worker.onerror = null;
        worker.terminate();
        worker = null;
      }
      settleAll(error || new EngineClientError('stopped', 'The engine was stopped.'));
    }

    // The worker broke after it was up (or while it started): end it and say so,
    // so the card stops offering agents that can no longer answer.
    function fail(error) {
      stop(error);
      set({status: 'error', info: null, progress: null, error: {kind: error.kind || 'engine', message: error.message}});
    }

    function request(message, {timeoutMs, onTimeout}) {
      if (!worker) return Promise.reject(new EngineClientError('not_ready', 'The engine is not running.'));
      const id = ++sequence;
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          pending.delete(id);
          const error = new EngineClientError('timeout', `The engine did not answer within ${Math.round(timeoutMs / 1000)} seconds and was stopped.`);
          onTimeout(error);
          reject(error);
        }, timeoutMs);
        pending.set(id, {resolve, reject, timer});
        worker.postMessage(Object.assign({id}, message));
      });
    }

    function receive(event) {
      const reply = event.data;
      const entry = reply && pending.get(reply.id);
      if (!entry) return;
      if (reply.type === 'progress') {
        set({progress: {stage: reply.stage, label: reply.label, done: reply.done, total: reply.total}});
        return;
      }
      clearTimeout(entry.timer);
      pending.delete(reply.id);
      if (reply.type === 'error') entry.reject(new EngineClientError(reply.kind || 'engine', reply.message || 'The engine reported an error.'));
      else entry.resolve(reply);
    }

    async function probe() {
      const problem = supported();
      if (problem) { set({status: 'unsupported', error: {kind: 'unsupported', message: problem}}); return state; }
      if (state.status === 'loading' || state.status === 'ready') return state;
      if (!env.fetch) { set({status: 'absent', note: 'This page cannot fetch files.'}); return state; }
      let response;
      try {
        response = await env.fetch(env.base + 'manifest.json', {cache: 'no-cache', credentials: 'omit'});
      } catch (error) {
        set({status: 'absent', note: `The engine files could not be reached: ${error.message}`});
        return state;
      }
      if (!response.ok) {
        set({status: 'absent', note: response.status === 404
          ? 'This site was published without the engine files.'
          : `The engine manifest returned HTTP ${response.status}.`});
        return state;
      }
      try {
        const manifest = await response.json();
        if (scope.EAgentEngineInstall) scope.EAgentEngineInstall.validateManifest(manifest);
        set({status: 'idle', manifest, error: null, note: ''});
      } catch (error) {
        set({status: 'error', error: {kind: error.kind || 'manifest', message: error.message}});
      }
      return state;
    }

    async function load() {
      if (state.status === 'ready') return state.info;
      if (loading) return loading;
      const mine = ++generation;
      const current = () => mine === generation;
      loading = (async () => {
        const problem = supported();
        if (problem) throw new EngineClientError('unsupported', problem);
        if (!state.manifest || state.status === 'absent' || state.status === 'error') await probe();
        if (state.status !== 'idle' && state.status !== 'loading') {
          const detail = state.error ? state.error.message : state.note;
          throw new EngineClientError(state.status === 'absent' ? 'absent' : 'manifest', detail || 'The engine is not available here.');
        }
        set({status: 'loading', progress: {stage: 'download', label: 'Starting the engine worker'}, error: null});
        try {
          // A module worker: the Python runtime's loader will not run in a classic one.
          worker = new env.Worker(env.workerUrl, {type: 'module'});
        } catch (error) {
          throw new EngineClientError('worker', `The engine worker could not start: ${error.message}`);
        }
        worker.onmessage = receive;
        // A worker script that cannot load (wrong path, blocked, bad MIME type) fires
        // only this; without it the page would wait out the whole time-out.
        worker.onerror = event => {
          fail(new EngineClientError('worker', `The engine worker failed: ${event && event.message ? event.message
            : 'it could not be loaded. Check that engine-worker.js is published next to index.html.'}`));
        };
        const reply = await request({type: 'init', manifest: state.manifest}, {
          timeoutMs: env.loadTimeoutMs,
          onTimeout: error => { stop(error); },
        });
        set({status: 'ready', info: reply.info, progress: null, error: null, note: ''});
        return reply.info;
      })().catch(error => {
        // A start the user cancelled leaves the state the cancel set: off, not failed.
        if (current()) {
          stop(error);
          set({status: 'error', progress: null, error: {kind: error.kind || 'engine', message: error.message}});
        }
        throw error;
      }).finally(() => { if (current()) loading = null; });
      return loading;
    }

    // Calls run one at a time, each with its own clock. The interpreter is
    // single-threaded anyway; queueing here means a slow call cannot make the
    // ones behind it look like they timed out.
    function call(agent, args) {
      if (typeof agent !== 'string' || typeof args !== 'string') {
        return Promise.reject(new EngineClientError('protocol', 'A call needs an agent name and a JSON string.'));
      }
      const run = async () => {
        if (state.status !== 'ready') throw new EngineClientError('not_ready', 'The engine is not running.');
        const reply = await request({type: 'call', agent, args}, {
          timeoutMs: env.callTimeoutMs,
          onTimeout: error => {
            stop(error);
            set({status: 'idle', info: null, note: 'The engine was stopped after a call ran too long. Press Start to run it again.'});
          },
        });
        return reply.json;
      };
      const result = queue.then(run, run);
      queue = result.catch(() => {});
      return result;
    }

    async function catalog() {
      if (state.status !== 'ready') throw new EngineClientError('not_ready', 'The engine is not running.');
      const reply = await request({type: 'catalog'}, {timeoutMs: env.callTimeoutMs, onTimeout: error => {
        stop(error);
        set({status: 'idle', info: null, note: 'The engine stopped answering and was stopped. Press Start to run it again.'});
      }});
      return JSON.parse(reply.json);
    }

    function unload() {
      generation += 1;
      loading = null;
      stop(new EngineClientError('stopped', 'The engine was unloaded.'));
      if (state.manifest) set({status: 'idle', info: null, progress: null, error: null, note: ''});
    }

    return {
      get state() { return state; },
      subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); },
      supported,
      probe,
      load,
      call,
      catalog,
      unload,
      device: () => describeDevice(options.navigator || scope.navigator, options.global || scope),
      gpu: () => describeGpu(options.navigator || scope.navigator),
    };
  }

  const api = {EngineClientError, create, describeDevice, describeGpu};
  scope.EAgentEngineClient = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : self);
