/* The panels that show where the work happens: the compute engine, this device,
 * the agents the controller can route to, and how a question flows through them.
 *
 * Everything here is a view of state that lives elsewhere (the engine client and
 * the published agent catalog), and every claim it makes is read from one of
 * them. It says what the device reports and what the engine did, and nothing
 * about what either might be capable of.
 */
(function (scope) {
  'use strict';

  const AUTO_KEY = 'eagent.engine.auto';
  const CHIP = {
    'browser-js': {label: 'JS', title: 'Runs as JavaScript in this page'},
    'browser-wasm': {label: 'WASM · CPU', title: 'Runs as Python compiled to WebAssembly, on this device\'s CPU, in a worker thread'},
  };
  const STATUS_TEXT = {runs: 'runs here', partial: 'partly here', needs_program: 'needs a program'};
  const KIND_TITLE = {
    evidence: 'Evidence · read-only lookups',
    calculator: 'Calculator',
    compute: 'Compute · Python on this device',
  };

  const megabytes = bytes => (bytes / 1e6).toFixed(1);
  const seconds = ms => (ms / 1000).toFixed(1);

  function create({byId, make, engine, agents, storage = null, onChange = () => {}}) {
    const el = {
      chip: byId('engine-chip'), summary: byId('engine-summary'), meter: byId('engine-meter'), stage: byId('engine-stage'),
      start: byId('engine-start'), stop: byId('engine-stop'), offer: byId('engine-offer'), offerLabel: byId('engine-offer-label'),
      auto: byId('engine-auto'), facts: byId('engine-facts'), error: byId('engine-error'),
      deviceFacts: byId('device-facts'), deviceHelp: byId('device-help'), gpuButton: byId('device-gpu'), gpuResult: byId('device-gpu-result'),
      enginePanel: byId('engine-panel'), devicePanel: byId('device-panel'), flowPanel: byId('flow-panel'),
      agentsPanel: byId('agents-panel'), agentsGroups: byId('agents-groups'), agentsNote: byId('agents-note'), agentsTag: byId('agents-tag'),
      composerChip: byId('composer-compute'),
      flowCount: byId('flow-agent-count'), flowJs: byId('flow-js-count'), flowWasm: byId('flow-wasm-count'),
      flowPipeline: byId('flow-pipeline'), flowSummary: byId('flow-pipeline-summary'),
    };
    let catalog = null;
    let referenceSchemas = [];
    let referenceNames = [];
    let catalogProblem = '';
    let checkedLive = false;
    // Only the browser run mode uses these panels; the Python backend runs its own tools.
    let active = true;

    const remember = (key, value) => { try { if (storage) storage.set(key, value); } catch (_) { /* a per-viewer convenience only */ } };
    const recall = key => { try { return storage ? storage.get(key) : null; } catch (_) { return null; } };

    const state = () => engine.state;
    // What the card should say. The engine being up is not enough: it is offered
    // to the model only once the page has checked that it holds the agents this
    // page was published with.
    function phase() {
      const current = state();
      if (current.status === 'ready') return !checkedLive ? 'verifying' : catalogProblem ? 'mismatch' : 'ready';
      if (['idle', 'error'].includes(current.status) && !catalog) return 'nocatalog';
      return current.status;
    }
    const ready = () => phase() === 'ready';

    // ---- compute engine card ------------------------------------------------
    const CHIPS = {
      unknown: ['CHECKING', 'muted'], unsupported: ['UNSUPPORTED', 'bad'], absent: ['NOT PUBLISHED', 'muted'], nocatalog: ['NO CATALOG', 'bad'],
      idle: ['OFF', 'muted'], loading: ['STARTING', 'warn'], verifying: ['CHECKING', 'warn'], ready: ['RUNNING', 'ok'],
      mismatch: ['MISMATCH', 'bad'], error: ['FAILED', 'bad'],
    };

    function renderEngine() {
      const current = state();
      const manifest = current.manifest;
      const kind = phase();
      const size = manifest && Number.isFinite(manifest.total_bytes) ? `${megabytes(manifest.total_bytes)} MB` : 'about 20 MB';
      el.chip.textContent = CHIPS[kind][0];
      el.chip.className = `chip ${CHIPS[kind][1]}`;

      const summaries = {
        unknown: 'Looking for the engine files published with this site.',
        unsupported: current.error ? current.error.message : 'This browser cannot run the engine.',
        absent: `${current.note || 'The engine files are not part of this site.'} Evidence lookups and the calculator still work.`,
        nocatalog: 'The agent catalog (agent-catalog.json) did not load, so compute agents cannot be offered. Reload the page.',
        idle: 'Runs the project\'s own Python on this device\'s CPU, as WebAssembly, in a worker thread. '
          + `One download of ${size}; after that your browser keeps the files and only re-checks them. Nothing you type is sent anywhere by it.`,
        loading: 'Starting the Python runtime on this device. Every file is checked against the build manifest before it is used.',
        verifying: 'The engine started. Checking that it holds exactly the agents this page was published with.',
        ready: 'The compute agents run here, on this device\'s CPU. Your model API can route questions to them; it never sees your files and cannot approve anything.',
        mismatch: `The engine started but does not match this page (${catalogProblem}). It is not offered to the model; reload to fetch a matching build.`,
        error: 'The engine did not start. Evidence lookups and the calculator still work.',
      };
      el.summary.textContent = summaries[kind];

      const loading = kind === 'loading';
      el.meter.hidden = !(loading || kind === 'verifying');
      if (!el.meter.hidden) el.stage.textContent = loading ? (current.progress && current.progress.label ? current.progress.label : 'Starting') : 'Checking the agent catalog';

      el.start.hidden = !(kind === 'idle' || kind === 'error');
      el.start.textContent = kind === 'error' ? 'Try again' : `Start on this device · ${size}`;
      el.stop.hidden = !(loading || kind === 'verifying' || kind === 'ready' || kind === 'mismatch');
      el.stop.textContent = loading ? 'Cancel' : 'Stop';
      el.offerLabel.hidden = kind !== 'ready';
      el.auto.disabled = ['unsupported', 'absent', 'unknown', 'nocatalog'].includes(kind);

      const failed = kind === 'error' || kind === 'unsupported';
      el.error.hidden = !(failed && current.error);
      el.error.textContent = failed && current.error ? current.error.message : '';

      const showFacts = (kind === 'ready' || kind === 'mismatch') && current.info;
      el.facts.hidden = !showFacts;
      if (showFacts) {
        const info = current.info;
        const rows = [
          ['Runtime', `Python ${info.python} · Pyodide ${info.pyodide}`],
          ['Agents', `${info.agents.length} compute agents`],
          ['Integrity', info.integrity === 'verified'
            ? 'every file checked against its SHA-256'
            : 'not checked: this connection has no Web Crypto (use HTTPS)'],
          ['Self-test', `${info.selftest} calls match the native build`],
          ['Started in', `${seconds(info.timings.total_ms)} s`],
          ['Build', info.bundle_sha256.slice(0, 12)],
        ];
        el.facts.replaceChildren(...rows.flatMap(([name, value]) => [make('dt', '', name), make('dd', '', value)]));
      }
      renderComposerChip();
    }

    // ---- this device -----------------------------------------------------------
    function yesNo(value, yes = 'available', no = 'not offered') { return value ? yes : no; }
    function renderDevice() {
      const device = engine.device();
      const memory = device.memory_gb === null ? 'not reported' : device.memory_gb >= 8 ? '8 GB or more' : `about ${device.memory_gb} GB`;
      const rows = [
        ['CPU threads', device.cpu_threads === null ? 'not reported' : String(device.cpu_threads)],
        ['Memory', memory],
        ['WebAssembly', yesNo(device.webassembly, 'yes', 'no')],
        ['Web Workers', yesNo(device.workers, 'yes', 'no')],
        ['WebGPU (GPU)', yesNo(device.webgpu)],
        ['WebNN (NPU)', yesNo(device.webnn)],
      ];
      el.deviceFacts.replaceChildren(...rows.flatMap(([name, value]) => [make('dt', '', name), make('dd', '', value)]));
      el.deviceHelp.textContent = 'Compute agents run on the CPU, as WebAssembly, in one worker thread. WebGPU and WebNN are listed because the browser reports them; '
        + 'no agent here has a GPU or NPU kernel, so nothing is sent to either. The registry can route one to them when one exists.';
    }

    async function checkGpu() {
      el.gpuButton.disabled = true;
      el.gpuResult.hidden = false;
      el.gpuResult.textContent = 'Asking the browser for a GPU adapter…';
      try {
        const gpu = await engine.gpu();
        el.gpuResult.textContent = gpu.available
          ? `Adapter: ${[gpu.vendor, gpu.architecture, gpu.description].filter(Boolean).join(' · ') || 'the browser gave no details'}. Present, and unused by the agents.`
          : `${gpu.reason}.`;
      } finally {
        el.gpuButton.disabled = false;
      }
    }

    // ---- agents ------------------------------------------------------------------
    function agentItem(agent) {
      const item = make('details', `agent ${agent.available ? '' : 'off'}`.trim());
      const summary = make('summary', 'agent-summary');
      const chip = CHIP[agent.runs] || CHIP['browser-js'];
      summary.append(make('span', `lane ${agent.runs === 'browser-wasm' ? 'wasm' : 'js'}`), make('span', 'agent-name', agent.name),
        make('span', 'agent-run', agent.available ? chip.label : 'needs engine'));
      item.append(summary);
      if (agent.description) item.append(make('p', 'agent-desc', agent.description));
      item.append(make('p', 'agent-where', agent.available ? chip.title : `${chip.title}. Start the compute engine to use it.`));
      if (agent.wraps) item.append(make('p', 'agent-wraps', `Runs ${agent.wraps}`));
      return item;
    }

    function renderAgents() {
      if (!catalog) { el.agentsPanel.hidden = !active || referenceNames.length === 0; renderReferenceOnly(); return; }
      const view = agents.registryView({referenceSchemas: referenceSchemas.length ? referenceSchemas : referenceNames.map(name => ({name, description: ''})),
        catalog, engineReady: ready()});
      const groups = ['evidence', 'calculator', 'compute'];
      el.agentsGroups.replaceChildren(...groups.flatMap(kind => {
        const members = view.filter(agent => agent.kind === kind);
        if (!members.length) return [];
        const group = make('div', 'agent-group');
        group.append(make('h3', 'agent-group-title', `${KIND_TITLE[kind]} · ${members.length}`));
        const list = make('div', 'agent-list');
        list.append(...members.map(agentItem));
        group.append(list);
        return [group];
      }));
      el.agentsPanel.hidden = !active;
      el.agentsTag.textContent = `${view.length} · ROUTED BY YOUR API`;
      el.agentsNote.textContent = ready()
        ? 'Your model API sees these agents\' names, descriptions and parameters, picks which to call, and receives the results. It does not run them.'
        : 'Compute agents are listed so you can see what the controller could route to. They run once the compute engine is started.';
    }

    function renderReferenceOnly() {
      el.agentsGroups.replaceChildren(...referenceNames.map(name => agentItem({name, description: '', runs: 'browser-js', available: true, wraps: ''})));
      el.agentsNote.textContent = '';
    }

    // ---- the flow diagram on the welcome screen --------------------------------------
    function renderFlow() {
      if (!catalog) return;
      const evidence = Math.max(0, referenceNames.length);
      el.flowCount.textContent = `${evidence + catalog.agents.length} agents`;
      el.flowJs.textContent = evidence ? `${evidence}` : '';
      el.flowWasm.textContent = `${catalog.agents.length}`;
      const counts = {runs: 0, partial: 0, needs_program: 0};
      el.flowPipeline.replaceChildren(...catalog.pipeline.map(step => {
        counts[step.runs_in_browser] += 1;
        const item = make('li', `step ${step.runs_in_browser}`);
        item.title = step.note;
        item.append(make('span', 'step-order', String(step.order).padStart(2, '0')), make('span', 'step-name', step.interface.replace(/_/g, ' ')),
          make('span', 'step-status', STATUS_TEXT[step.runs_in_browser]));
        return item;
      }));
      el.flowSummary.textContent = `${counts.runs} run here, ${counts.partial} run partly, ${counts.needs_program} need an external program that no browser can run. `
        + 'Those are reported as unavailable and never imitated.';
    }

    function renderComposerChip() {
      const offered = offering();
      el.composerChip.hidden = !offered;
      if (offered) el.composerChip.textContent = `${catalog.agents.length} compute agents · this device`;
    }

    // ---- what is offered to the model ----------------------------------------------------
    function offering() { return ready() && el.offer.checked === true; }
    function option() { return offering() ? {engine, catalog} : null; }

    async function verifyLive() {
      // The page and the engine come from one build; if they do not, say so and
      // offer nothing rather than a mismatched set of agents.
      checkedLive = false;
      catalogProblem = '';
      if (state().status !== 'ready' || !catalog) return;
      try {
        const problems = agents.matchesEngine(catalog, await engine.catalog());
        catalogProblem = problems.join('; ');
      } catch (error) {
        catalogProblem = error.message;
      }
      checkedLive = true;
    }

    function renderAll() {
      el.enginePanel.hidden = !active;
      el.devicePanel.hidden = !active;
      el.flowPanel.hidden = !active || !catalog;
      renderEngine();
      renderAgents();
      renderFlow();
      onChange();
    }

    // ---- wiring ----------------------------------------------------------------------------
    async function start() {
      if (!catalog) return;
      try {
        await engine.load();
      } catch (_) { /* the state carries the error and the card shows it */ }
    }

    let lastStatus = '';
    engine.subscribe(async current => {
      if (current.status === 'ready' && lastStatus !== 'ready') {
        lastStatus = 'ready';
        checkedLive = false;
        catalogProblem = '';
        renderAll();
        await verifyLive();
      } else if (current.status !== 'ready') {
        lastStatus = current.status;
        checkedLive = false;
        catalogProblem = '';
      }
      renderAll();
    });

    el.start.addEventListener('click', start);
    el.stop.addEventListener('click', () => engine.unload());
    el.offer.addEventListener('change', renderAll);
    el.auto.addEventListener('change', () => remember(AUTO_KEY, el.auto.checked ? 'on' : 'off'));
    el.gpuButton.addEventListener('click', checkGpu);
    el.auto.checked = recall(AUTO_KEY) === 'on';
    el.offer.checked = true;
    renderDevice();
    renderEngine();

    return {
      option,
      offering,
      setActive(value) {
        active = Boolean(value);
        renderAll();
      },
      renderAll,
      start,
      autoStart: () => el.auto.checked === true,
      setCatalog(value) {
        catalog = agents.checkCatalog(value);
        renderAll();
      },
      setReference(schemas, names) {
        referenceSchemas = Array.isArray(schemas) ? schemas : [];
        referenceNames = Array.isArray(names) ? names : referenceSchemas.map(schema => schema.name);
        renderAll();
      },
      chipFor: executor => CHIP[executor] || null,
      get verified() { return ready(); },
      phase,
    };
  }

  const api = {CHIP, create};
  scope.EAgentComputeUI = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : window);
