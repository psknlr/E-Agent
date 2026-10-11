/* Install the engine into a Pyodide runtime, and answer calls on it.
 *
 * This file runs in two places: inside the engine's Web Worker, and in Node,
 * where the tests drive it against the same built files. It has no DOM and no
 * network code of its own. The caller says how to load the runtime and how to
 * read a file, so the install steps are one piece of code whichever side runs
 * them.
 *
 * What it guarantees: every file it reads is checked against the SHA-256 the
 * build recorded in manifest.json before anything from it is installed, and the
 * engine is not "ready" until a few real agent calls have returned exactly what
 * the build computed natively for the same inputs. A runtime that does the sums
 * differently is reported, not used.
 */
(function (scope) {
  'use strict';

  const MANIFEST_SCHEMA = 'eagent.engine.manifest.v1';
  const ENGINE_SCHEMA = 'eagent.engine.v1';
  const HEX64 = /^[0-9a-f]{64}$/;
  // A path inside the engine directory: no climbing out, no scheme, no query.
  const SAFE_PATH = /^(?!\/)(?!.*(?:^|\/)\.\.?(?:\/|$))[A-Za-z0-9._\-/]+$/;

  class EngineError extends Error {
    constructor(kind, message) {
      super(message);
      this.name = 'EngineError';
      this.kind = kind;
    }
  }

  const now = () => (scope.performance && scope.performance.now ? scope.performance.now() : Date.now());

  async function sha256Hex(bytes) {
    let subtle = scope.crypto && scope.crypto.subtle;
    if (!subtle && typeof require === 'function') subtle = require('node:crypto').webcrypto.subtle;
    // Browsers expose SubtleCrypto only on secure origins (HTTPS or localhost).
    if (!subtle) return null;
    const digest = new Uint8Array(await subtle.digest('SHA-256', bytes));
    return Array.from(digest, byte => byte.toString(16).padStart(2, '0')).join('');
  }

  function hexToBase64(hex) {
    const bytes = hex.match(/../g).map(pair => parseInt(pair, 16));
    return typeof btoa === 'function'
      ? btoa(String.fromCharCode(...bytes))
      : Buffer.from(bytes).toString('base64');
  }

  // The manifest is read from the same host as the page, but it is still input:
  // a truncated deploy or a stale cache must fail here with a sentence, not
  // later as a stack trace from inside the interpreter.
  function validateManifest(manifest) {
    const fail = message => { throw new EngineError('manifest', `engine manifest: ${message}`); };
    if (!manifest || typeof manifest !== 'object') fail('is not an object');
    if (manifest.schema !== MANIFEST_SCHEMA) fail(`expected schema ${MANIFEST_SCHEMA}, found ${manifest.schema}`);
    if (manifest.engine_schema !== ENGINE_SCHEMA) {
      fail(`built for ${manifest.engine_schema}, but this page speaks ${ENGINE_SCHEMA}`);
    }
    const files = manifest.files;
    if (!files || typeof files !== 'object') fail('has no file list');
    for (const [name, entry] of Object.entries(files)) {
      if (!SAFE_PATH.test(name)) fail(`lists an unsafe path ${JSON.stringify(name)}`);
      if (!entry || !HEX64.test(entry.sha256) || !Number.isSafeInteger(entry.bytes) || entry.bytes < 0) {
        fail(`${name} has no valid size and SHA-256`);
      }
    }
    const install = manifest.install;
    if (!install || !Array.isArray(install.wheels) || !Array.isArray(install.site_packages_zips)
        || typeof install.project !== 'string') fail('has no install plan');
    for (const name of [...install.wheels, ...install.site_packages_zips, install.project]) {
      if (!Object.prototype.hasOwnProperty.call(files, name)) fail(`installs ${name}, which it does not list`);
    }
    for (const name of ['pyodide/pyodide.mjs', 'pyodide/pyodide.asm.wasm', 'pyodide/python_stdlib.zip']) {
      if (!Object.prototype.hasOwnProperty.call(files, name)) fail(`does not list ${name}`);
    }
    if (!Array.isArray(manifest.selftest) || !manifest.selftest.length) fail('has no self-test');
    for (const check of manifest.selftest) {
      if (!check || typeof check.agent !== 'string' || typeof check.arguments !== 'string'
          || !HEX64.test(check.sha256)) fail('has a malformed self-test entry');
    }
    return manifest;
  }

  // `fetch` with the browser's own subresource-integrity check added for the
  // files the runtime loads for itself (the WebAssembly binary, the standard
  // library, the lock file). The installer verifies the files it reads; this
  // covers the ones it does not, so a half-updated deploy fails loudly instead
  // of as a LinkError deep inside WebAssembly. It also asks the cache to
  // revalidate, so a copy fetched before a deploy is replaced, not tripped over.
  function integrityFetch(nativeFetch, baseUrl, manifest) {
    const base = new URL(baseUrl).href;
    return function (input, init) {
      // The runtime passes a string for some files and a URL object for others
      // (including its WebAssembly binary); a Request carries .url.
      const url = typeof input === 'string' ? input : input && (input.href || input.url);
      if (typeof url === 'string' && url.startsWith(base)) {
        const name = url.slice(base.length).split(/[?#]/)[0];
        const entry = Object.prototype.hasOwnProperty.call(manifest.files, name) ? manifest.files[name] : null;
        if (entry) init = Object.assign({}, init, {integrity: `sha256-${hexToBase64(entry.sha256)}`, cache: 'no-cache'});
      }
      return nativeFetch(input, init);
    };
  }

  // Install the engine and return the two functions the rest of the page uses.
  //
  //   loadPyodide  async (options) => pyodide      the runtime's loader
  //   indexURL     where the runtime's files are
  //   readBytes    async (name) => Uint8Array      a file of the build, by manifest name
  //   manifest     the parsed manifest.json
  //   onProgress   ({stage, label}) => void
  async function install({loadPyodide, indexURL, readBytes, manifest, onProgress = () => {}}) {
    validateManifest(manifest);
    const started = now();
    const timings = {};
    let mark = started;
    const lap = name => { const t = now(); timings[name] = Math.round(t - mark); mark = t; };
    let integrity = 'verified';

    const read = async name => {
      const entry = manifest.files[name];
      const bytes = await readBytes(name);
      if (bytes.length !== entry.bytes) {
        throw new EngineError('integrity', `${name} is ${bytes.length} bytes; the build recorded ${entry.bytes}. `
          + 'The deployed engine files do not all come from one build; reload to fetch them again.');
      }
      const actual = await sha256Hex(bytes);
      if (actual === null) integrity = 'unavailable';
      else if (actual !== entry.sha256) {
        throw new EngineError('integrity', `${name} does not match the build manifest `
          + `(expected ${entry.sha256.slice(0, 12)}, received ${actual.slice(0, 12)}). `
          + 'The deployed engine files do not all come from one build; reload to fetch them again.');
      }
      return bytes;
    };

    onProgress({stage: 'runtime', label: 'Starting the Python runtime (WebAssembly)'});
    const output = [];
    const keep = line => { if (output.length < 40) output.push(String(line)); };
    const pyodide = await loadPyodide({indexURL, stdout: keep, stderr: keep}, manifest);
    lap('runtime_ms');

    const site = pyodide.runPython('import sysconfig\nsysconfig.get_path("purelib")');
    const wheels = manifest.install.wheels;
    for (let index = 0; index < wheels.length; index += 1) {
      onProgress({stage: 'packages', label: `Installing ${wheels[index].split('/').pop()}`, done: index, total: wheels.length});
      await pyodide.unpackArchive(await read(wheels[index]), 'wheel', {extractDir: site});
    }
    for (const name of manifest.install.site_packages_zips) {
      await pyodide.unpackArchive(await read(name), 'zip', {extractDir: site});
    }
    lap('packages_ms');

    onProgress({stage: 'project', label: 'Loading the E-Agent package'});
    await pyodide.unpackArchive(await read(manifest.install.project), 'zip', {extractDir: '/work'});
    pyodide.runPython([
      'import importlib, sys',
      'sys.path.insert(0, "/work/src")',
      'importlib.invalidate_caches()',
      'from eagent.engine import call as _eagent_call, catalog as _eagent_catalog',
    ].join('\n'));
    const pyCall = pyodide.globals.get('_eagent_call');
    const pyCatalog = pyodide.globals.get('_eagent_catalog');
    const call = (agent, argumentsJson) => String(pyCall(agent, argumentsJson));
    const catalog = () => String(pyCatalog());
    lap('project_ms');

    onProgress({stage: 'check', label: 'Checking the engine against the build'});
    const listed = JSON.parse(catalog());
    if (listed.schema !== manifest.engine_schema) {
      throw new EngineError('selftest', `the package reports ${listed.schema}, the build ${manifest.engine_schema}`);
    }
    for (const check of manifest.selftest) {
      const reply = JSON.parse(call(check.agent, check.arguments));
      if (!reply.ok) throw new EngineError('selftest', `${check.agent} refused its own self-test: ${reply.refusal}`);
      if (reply.cite.sha256 !== check.sha256) {
        throw new EngineError('selftest', `${check.agent} computed a different answer here than the build did natively `
          + `(${reply.cite.sha256.slice(0, 12)} against ${check.sha256.slice(0, 12)}); this runtime is not used.`);
      }
    }
    lap('selftest_ms');
    timings.total_ms = Math.round(now() - started);

    const version = pyodide.runPython('import sys\nsys.version.split()[0]');
    return {
      call,
      catalog,
      info: {
        python: String(version),
        pyodide: manifest.pyodide,
        bundle_sha256: manifest.bundle_sha256,
        engine_schema: manifest.engine_schema,
        agents: listed.agents.map(agent => agent.name),
        integrity,
        selftest: manifest.selftest.length,
        timings,
        output,
      },
    };
  }

  // The message handler both the worker and the tests use. One engine per host;
  // `post` carries progress, the returned value is the reply to one message.
  function createHost({loadPyodide, indexURL, readBytes, post = () => {}}) {
    let engine = null;
    let starting = null;
    const reply = (message, extra) => Object.assign({id: message.id}, extra);
    const fail = (message, error) => reply(message, {
      type: 'error',
      kind: error && error.kind ? error.kind : 'engine',
      message: error && error.message ? error.message : String(error),
    });
    return async function handle(message) {
      try {
        if (!message || typeof message !== 'object') throw new EngineError('protocol', 'a message is required');
        if (message.type === 'init') {
          if (engine) return reply(message, {type: 'ready', info: engine.info});
          if (!starting) {
            starting = install({
              loadPyodide, indexURL, readBytes, manifest: message.manifest,
              onProgress: progress => post(reply(message, {type: 'progress', ...progress})),
            }).then(
              ready => { engine = ready; return ready; },
              error => { starting = null; throw error; });
          }
          return reply(message, {type: 'ready', info: (await starting).info});
        }
        if (!engine) throw new EngineError('not_ready', 'the engine has not been started');
        if (message.type === 'call') {
          if (typeof message.agent !== 'string' || typeof message.args !== 'string') {
            throw new EngineError('protocol', 'a call needs an agent name and a JSON string');
          }
          return reply(message, {type: 'result', json: engine.call(message.agent, message.args)});
        }
        if (message.type === 'catalog') return reply(message, {type: 'result', json: engine.catalog()});
        throw new EngineError('protocol', `unknown message type ${JSON.stringify(message.type)}`);
      } catch (error) {
        return fail(message || {}, error);
      }
    };
  }

  const api = {EngineError, createHost, hexToBase64, install, integrityFetch, sha256Hex, validateManifest};
  scope.EAgentEngineInstall = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : self);
