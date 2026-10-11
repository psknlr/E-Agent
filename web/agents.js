/* One registry of agents for the controller to route over.
 *
 * The page has two kinds of agent. Reference agents read the published evidence
 * (reference-tools.js). Compute agents run the project's own Python in this
 * browser as WebAssembly (engine-client.js). A language model, reached through
 * the visitor's own API key, decides which agent to call and with what; it never
 * runs anything. This file is where a call is checked, sent to the executor that
 * owns it, and its result made citable.
 *
 * What makes a computed number trustworthy here is the same thing that makes a
 * stored one trustworthy: the model may only quote it with a citation, and the
 * numeric guard checks the number against the exact field it cites. A result is
 * registered with the guard only after the engine has produced it, so a model
 * can cite a computation that happened and cannot invent one.
 */
(function (scope) {
  'use strict';

  const ARTIFACT = 'engine_compute';
  const SCHEMA = 'eagent.engine.v1';
  const HEX64 = /^[0-9a-f]{64}$/;
  const AGENT_NAME = /^[a-z][a-z0-9_]{0,63}$/;
  const MAX_LOCAL_CALLS = 8;

  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const clone = value => JSON.parse(JSON.stringify(value));
  const plainObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);

  function checkCatalog(catalog) {
    const fail = message => { throw new Error(`The agent catalog is invalid: ${message}.`); };
    if (!plainObject(catalog) || catalog.schema !== SCHEMA || catalog.artifact !== ARTIFACT) fail('wrong schema');
    if (!plainObject(catalog.limits) || !Number.isInteger(catalog.limits.max_arg_chars) || catalog.limits.max_arg_chars < 1) fail('no argument limit');
    if (!Array.isArray(catalog.agents) || !catalog.agents.length) fail('no agents');
    const seen = new Set();
    for (const agent of catalog.agents) {
      if (!plainObject(agent) || !AGENT_NAME.test(agent.name || '') || seen.has(agent.name)) fail('an agent name is missing, malformed or repeated');
      seen.add(agent.name);
      if (typeof agent.description !== 'string' || !agent.description || typeof agent.wraps !== 'string' || !plainObject(agent.parameters)
          || Object.values(agent.parameters).some(help => typeof help !== 'string')) fail(`${agent.name} is malformed`);
      if (!plainObject(agent.max_chars) || Object.entries(agent.max_chars).some(([key, limit]) => !own(agent.parameters, key) || !Number.isInteger(limit) || limit < 1)) {
        fail(`${agent.name} has malformed argument limits`);
      }
    }
    if (!Array.isArray(catalog.pipeline) || !Array.isArray(catalog.tasks) || catalog.tasks.some(id => typeof id !== 'string')) fail('no pipeline map');
    return catalog;
  }

  // The leaf values of a result, named by dotted path (a.b, a.0.b). The citation
  // grammar ends a citation at the first "]", so a bracketed name such as a[0]
  // could never be cited; dots can. Non-numeric leaves are citable by name but
  // carry no number.
  function leaves(value, path = '', out = []) {
    if (Array.isArray(value)) value.forEach((item, index) => leaves(item, path ? `${path}.${index}` : String(index), out));
    else if (plainObject(value)) {
      for (const [name, item] of Object.entries(value)) if (name !== 'cite') leaves(item, path ? `${path}.${name}` : name, out);
    } else out.push([path, value]);
    return out;
  }

  function rowFor(value, sha256) {
    const fields = {}, names = [];
    for (const [path, leaf] of leaves(value)) {
      names.push(path);
      if (typeof leaf === 'number' && Number.isFinite(leaf)) fields[path] = leaf;
      else if (leaf === null) fields[path] = null;
    }
    return {sha256, fields, field_names: names};
  }

  // What the page promises the model about compute agents. It is appended to the
  // reference prompt, so the rules about citing and about refusals stay one voice.
  function addendum(catalog) {
    return [
      '',
      'COMPUTE AGENTS',
      'Besides the read-only reference tools, you can call compute agents. They run on the user\'s own device, in this',
      'browser, as the project\'s own Python code compiled to WebAssembly (a CPU computation). They take strings and',
      'return values computed there, for exactly the arguments you sent. You decide which agent to call; you do not',
      'run anything yourself, and you cannot add an agent.',
      '- Quote a number from an agent only with a citation placed straight after it:',
      `  [cite artifact=${ARTIFACT} sha256=<digest> row=<row> field=<field path> method=read]`,
      '  using the digest and row in that result\'s own "cite", and for field the name of the result field the number',
      '  came from (a path such as identity_percent_over_aligned_columns, or variants.0.position for the first item of a list).',
      '  One citation backs one number. Where a result holds a fraction it also holds the same value as a percentage (a',
      '  field ending _percent); quote a percentage from that field, and cite every number you write, including a confidence level.',
      '- If an agent refuses, report its refusal. Never compute the number yourself or substitute a nearby one.',
      '- Agents never approve anything. The three decision points (reaction_spec_confirmed, synthesis_authorized,',
      '  functional_criteria_confirmed) can be cleared only by a named person. task_gate_report tells you what a',
      '  task still leaves undecided; it cannot decide it. run_pipeline runs the project\'s own controller on a task',
      '  here (network off, no approvals): the reaction step runs for real and the run stops at the first decision',
      '  point. Say where it stopped and which decision it waits on, and never say a step ran that its report does not list.',
      '  If you write a task_yaml from the user\'s words, say that its structures are your reading of what they said and',
      '  that a person must confirm them at the first decision point.',
      '- Three of the ten pipeline interfaces need external programs that no browser can run (sequence search, structure',
      '  preparation, complex modelling) and two run only partly (evidence retrieval needs network access; mutation',
      '  proposal needs LigandMPNN for inverse folding). pipeline_catalog says which. Never imply a step ran that it says cannot.',
      `Compute agents: ${catalog.agents.map(agent => agent.name).join(', ')}.`,
    ].join('\n');
  }

  // ---- routing a question to compute agents without a model -----------------
  // Like the reference router, this is a keyword and pattern match, not language
  // understanding. It only passes on text the user wrote (a sequence, two
  // counts, coordinates, a task id), and says why each agent ran.
  const RESIDUES = 'ACDEFGHIKLMNPQRSTVWY';
  const SEQUENCE = new RegExp(`(?<![A-Za-z0-9_])[${RESIDUES}]{8,}(?![A-Za-z0-9_])`, 'g');
  const VARIANT = new RegExp(`(?<![A-Za-z0-9_])[${RESIDUES}]\\d{1,5}[${RESIDUES}](?![A-Za-z0-9_])`, 'g');
  // Ordinary capitalised words that happen to be spelled with residue letters.
  const NOT_SEQUENCES = new Set(['ALIGNMENT', 'IDENTITY', 'SIMILARITY', 'DISTANCE', 'ACTIVITY', 'INTERVAL']);
  const POINT = /(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)/g;
  const COUNTS = /(?<![\w.])(\d{1,6})\s*(?:\/|of|out of)\s*(\d{1,6})(?![\w.])/i;
  const unique = list => [...new Set(list)];
  const escapeRegExp = value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  function planCompute(text, catalog) {
    const calls = [], notes = [];
    if (typeof text !== 'string' || !text.trim()) return {calls, notes};
    const has = pattern => pattern.test(text);
    const add = (interfaceName, args, why) => calls.push({interface: interfaceName, arguments: args, rationale: why});
    const names = new Set(catalog.agents.map(agent => agent.name));
    const found = unique([...text.matchAll(SEQUENCE)].map(match => match[0]).filter(token => !NOT_SEQUENCES.has(token)));

    const wantsDistance = has(/\b(?:hamming|k-?mer|jaccard|sequence distance|distance between (?:the )?sequences)\b/i);
    const wantsAlignment = has(/\b(?:align\w*|identity|identical|homolog\w*|compare (?:the )?sequences|sequence comparison|percent identity)\b/i);
    if (names.has('sequence_distance') && wantsDistance) {
      if (found.length >= 2) {
        const args = {sequence_a: found[0], sequence_b: found[1]};
        if (has(/\bhamming\b/i)) args.method = 'hamming';
        add('sequence_distance', args, `The message asks for a sequence distance and names two sequences (${found[0].length} and ${found[1].length} residues).`);
      } else notes.push(`A sequence distance needs two sequences written in capital one-letter codes (at least 8 residues each); I found ${found.length}.`);
    } else if (names.has('align_sequences') && wantsAlignment) {
      if (found.length >= 2) {
        add('align_sequences', {sequence_a: found[0], sequence_b: found[1]},
          `The message asks to align or compare sequences and names two (${found[0].length} and ${found[1].length} residues).`);
      } else notes.push(`An alignment needs two sequences written in capital one-letter codes (at least 8 residues each); I found ${found.length}.`);
    }

    const variants = unique([...text.matchAll(VARIANT)].map(match => match[0]));
    if (names.has('variant_check') && variants.length && has(/\b(?:variants?|mutations?|mutants?|substitutions?|residues?|positions?|numbering)\b/i)) {
      if (found.length) {
        const parent = [...found].sort((a, b) => b.length - a.length)[0];
        add('variant_check', {sequence: parent, variants: variants.join(', ')},
          `The message lists substitutions (${variants.join(', ')}) and a parent sequence of ${parent.length} residues.`);
      } else notes.push('Checking variants needs the parent sequence, written in capital one-letter codes (at least 8 residues).');
    }

    const counts = COUNTS.exec(text);
    if (names.has('wilson_interval') && counts && has(/\b(?:wilson|confidence interval|interval|poses?|successes?|robust\w*|hits?|replicates?|binomial|proportion|fraction)\b/i)) {
      const successes = Number(counts[1]), trials = Number(counts[2]);
      if (trials >= 1 && successes <= trials) {
        const args = {successes: String(successes), trials: String(trials)};
        const level = /\b(90|95|99)\s*%/.exec(text) || /\b0\.(90|95|99)\b/.exec(text);
        if (level && has(/\bconfidence|\bCI\b/i)) args.confidence = `0.${level[1]}`;
        add('wilson_interval', args, `The message gives ${successes} of ${trials} and asks for an interval on that proportion.`);
      } else notes.push(`${counts[1]} of ${counts[2]} is not a count of successes out of trials.`);
    }

    const points = [...text.matchAll(POINT)].map(match => `${match[1]},${match[2]},${match[3]}`);
    const measure = has(/\b(?:dihedral|torsion)\b/i) ? 'dihedral' : has(/\bangle\b/i) ? 'angle'
      : has(/\bdistance\b/i) && !wantsDistance ? 'distance' : has(/\bcentroid|\bcent(?:er|re) of\b/i) ? 'centroid' : '';
    if (names.has('geometry_measure') && measure && points.length) {
      const needed = {distance: 2, angle: 3, dihedral: 4}[measure];
      if (!needed || points.length === needed) {
        add('geometry_measure', {measure, points: points.join('; ')}, `The message asks for a ${measure} and gives ${points.length} coordinate triple${points.length === 1 ? '' : 's'}.`);
      } else notes.push(`A ${measure} needs exactly ${needed} points written as x,y,z; I found ${points.length}.`);
    }

    const named = catalog.tasks.filter(id => new RegExp(`(?<![A-Za-z0-9_])${escapeRegExp(id)}(?![A-Za-z0-9_])`).test(text));

    // "Approve" and "decide" appear in many questions; they only mean a gate report
    // when the message also names a task or the thing being decided.
    if (names.has('task_gate_report')) {
      const strong = has(/\b(?:gates?|decision points?|unresolved|sign[- ]?off|reaction_spec_confirmed|synthesis_authorized|functional_criteria_confirmed)\b/i);
      const weak = has(/\b(?:approv\w*|decid\w*|decisions?|blocked|blocking)\b/i)
        && (named.length || has(/\b(?:task|pilot|synthesis|criteria|reaction spec)\b/i));
      if (strong || weak) {
        if (named.length) for (const id of named.slice(0, 2)) add('task_gate_report', {task: id}, `“${id}” is a shipped task named in the message.`);
        else if (catalog.tasks.length === 1) add('task_gate_report', {task: catalog.tasks[0]}, `No task was named, and “${catalog.tasks[0]}” is the only shipped task.`);
      }
    }

    // "Run" only means starting the controller when it is about the pipeline or a task,
    // so "which steps can run in a browser" does not start one.
    if (names.has('run_pipeline')) {
      const runs = has(/\b(?:run|execute|drive|start)\b[^.?!]{0,60}\b(?:pipeline|controller|protocol|workflow|task)\b/i)
        || (named.length > 0 && has(/\b(?:run|execute|drive|start)\b/i));
      if (runs) {
        if (named.length) for (const id of named.slice(0, 2)) add('run_pipeline', {task: id}, `The message asks to run the pipeline on the shipped task “${id}”.`);
        else if (catalog.tasks.length === 1) add('run_pipeline', {task: catalog.tasks[0]}, `The message asks to run the pipeline, and “${catalog.tasks[0]}” is the only shipped task.`);
      }
    }

    // The bare word "pipeline" asks for the capability map, unless the message is a request to run it.
    const ran = calls.some(item => item.interface === 'run_pipeline');
    if (names.has('pipeline_catalog')
        && (has(/\b(?:interfaces?|which steps?|each step|run in (?:the |my )?browser|external programs?|need (?:a |an )?(?:external )?program|what can (?:this|the) browser)\b/i)
          || (!ran && has(/\bpipeline\b/i)))) {
      add('pipeline_catalog', {}, 'The message asks what the pipeline can do in a browser.');
    }
    return {calls: calls.slice(0, MAX_LOCAL_CALLS), notes};
  }

  // Whether the live engine offers exactly the agents the page was published
  // with. A mismatch means the page and the engine come from different builds.
  function matchesEngine(catalog, live) {
    const problems = [];
    if (!plainObject(live) || live.schema !== SCHEMA || !Array.isArray(live.agents)) return ['the engine does not report its agents'];
    const here = new Map(catalog.agents.map(agent => [agent.name, agent]));
    const there = new Map(live.agents.map(agent => [agent.name, agent]));
    for (const name of here.keys()) if (!there.has(name)) problems.push(`the engine has no agent ${name}`);
    for (const name of there.keys()) if (!here.has(name)) problems.push(`the engine has an agent ${name} that this page does not list`);
    for (const [name, agent] of here) {
      const other = there.get(name);
      if (!other) continue;
      if (JSON.stringify(Object.keys(agent.parameters).sort()) !== JSON.stringify(Object.keys(other.parameters || {}).sort())) problems.push(`${name} takes different parameters`);
      if (agent.description !== other.description) problems.push(`${name} is described differently`);
    }
    if (live.limits && catalog.limits && (live.limits.max_arg_chars !== catalog.limits.max_arg_chars || live.limits.max_result_bytes !== catalog.limits.max_result_bytes)) {
      problems.push('the engine enforces different size limits');
    }
    return problems;
  }

  // What the Agents panel shows: every agent the controller can route to, where
  // each runs, and whether it can run right now.
  function registryView({referenceSchemas = [], catalog, engineReady = false}) {
    checkCatalog(catalog);
    const reference = referenceSchemas.map(schema => ({
      name: schema.name, description: schema.description, kind: schema.name === 'calculate' ? 'calculator' : 'evidence',
      runs: 'browser-js', available: true, wraps: '',
    }));
    const compute = catalog.agents.map(agent => ({
      name: agent.name, description: agent.description, kind: 'compute', runs: 'browser-wasm', available: Boolean(engineReady), wraps: agent.wraps,
    }));
    return [...reference, ...compute];
  }

  // Wrap a reference-tools instance so the same registry also serves compute agents.
  function compose({tools, engine, catalog}) {
    checkCatalog(catalog);
    if (!tools || typeof tools.execute !== 'function' || typeof tools.registerComputed !== 'function') {
      throw new Error('The reference tools cannot register computed results; reload the page.');
    }
    if (!engine || typeof engine.call !== 'function' || !engine.state) throw new Error('The compute engine client is incompatible with this page.');
    const referenceNames = tools.toolNames();
    const specs = new Map(catalog.agents.map(agent => [agent.name, agent]));
    for (const name of specs.keys()) if (referenceNames.includes(name)) throw new Error(`The compute agent ${name} would shadow a reference tool.`);
    const computeSchemas = () => catalog.agents.map(agent => ({
      name: agent.name, description: agent.description, parameters: clone(agent.parameters), runs: 'in this browser (WebAssembly, CPU)',
    }));

    async function runAgent(call) {
      const started = Date.now();
      const name = call.interface;
      const spec = specs.get(name);
      const args = call.arguments === undefined ? {} : call.arguments;
      const result = {tool: name, arguments: {}, rationale: typeof call.rationale === 'string' ? call.rationale : '', ok: false,
        value: null, refusal: '', truncated: false, elapsed_ms: 0, executor: 'browser-wasm', wraps: spec.wraps};
      try {
        if (!plainObject(args)) throw new Error('Tool arguments must be an object.');
        if (Object.keys(args).some(key => !own(spec.parameters, key))) throw new Error('The tool received an unsupported argument.');
        if (Object.values(args).some(value => typeof value !== 'string')) throw new Error('Tool arguments must be strings.');
        for (const [key, value] of Object.entries(args)) {
          const limit = own(spec.max_chars, key) ? spec.max_chars[key] : catalog.limits.max_arg_chars;
          if (value.length > limit) throw new Error(`Tool argument ${key} exceeds its ${limit}-character limit.`);
        }
        result.arguments = clone(args);
        if (engine.state.status !== 'ready') {
          throw new Error('The compute engine is not running in this browser. Start it from the Compute engine card, or ask without compute agents.');
        }
        let reply;
        try {
          reply = JSON.parse(await engine.call(name, JSON.stringify(args)));
        } catch (error) {
          throw new Error(error && error.kind ? `The compute engine failed: ${error.message}` : 'The compute engine returned unreadable output.');
        }
        if (!plainObject(reply) || reply.agent !== name || typeof reply.ok !== 'boolean') throw new Error('The compute engine returned a malformed result.');
        if (!reply.ok) {
          result.refusal = typeof reply.refusal === 'string' && reply.refusal ? reply.refusal : 'The agent declined to produce a value.';
          result.engine_kind = typeof reply.kind === 'string' ? reply.kind : 'refused';
        } else {
          const cite = reply.cite;
          if (!plainObject(reply.value) || own(reply.value, 'cite') || !plainObject(cite) || cite.artifact !== ARTIFACT || !HEX64.test(cite.sha256 || '')
              || cite.row !== `${name}-${cite.sha256.slice(0, 16)}` || cite.method !== 'read') {
            throw new Error('The compute engine returned a result without a valid citation.');
          }
          tools.registerComputed(ARTIFACT, cite.row, rowFor(reply.value, cite.sha256));
          result.value = Object.assign(clone(reply.value), {cite: {artifact: ARTIFACT, method: 'read', row: cite.row, sha256: cite.sha256}});
          result.ok = true;
          if (typeof reply.elapsed_ms === 'number' && Number.isFinite(reply.elapsed_ms)) result.compute_ms = reply.elapsed_ms;
        }
      } catch (error) {
        result.ok = false;
        result.value = null;
        result.refusal = error instanceof Error ? error.message : 'The compute agent refused the request.';
      }
      result.elapsed_ms = Date.now() - started;
      return result;
    }

    async function execute(call) {
      const name = call && typeof call.interface === 'string' ? call.interface : '';
      if (specs.has(name)) return runAgent(call);
      return Object.assign(await tools.execute(call), {executor: 'browser-js'});
    }

    function plan(text) {
      const reference = typeof tools.plan === 'function' ? tools.plan(text) : {calls: [], notes: []};
      const compute = planCompute(text, catalog);
      const calls = [...reference.calls];
      for (const call of compute.calls) if (calls.length < MAX_LOCAL_CALLS) calls.push(call);
      // The reference router's "could not match" note only appears when it found
      // nothing, so it is dropped once a compute agent has found something.
      const notes = [...(!reference.calls.length && compute.calls.length ? [] : reference.notes), ...compute.notes];
      if (!calls.length) {
        const task = catalog.tasks[0] || 'KRED_PILOT_001';
        notes.push('The compute engine is on, so you can also ask: “align MKTAYIAKQR with MKTAYLAKQR”, “Wilson interval for 7 of 20 poses”, '
          + `“what is unresolved for ${task}”, “run ${task} through the pipeline”, “distance between 0,0,0 and 3,4,0” or “which pipeline steps can run in a browser”.`);
      }
      return {calls, notes};
    }

    return {
      systemPrompt: tools.systemPrompt + '\n' + addendum(catalog),
      bundleDigest: tools.bundleDigest,
      names: [...referenceNames, ...specs.keys()],
      toolNames: () => [...tools.toolNames(), ...specs.keys()],
      schemas: () => [...tools.schemas(), ...computeSchemas()],
      execute,
      inspect: text => tools.inspect(text),
      guard: text => tools.inspect(text),
      ids: () => tools.ids(),
      plan,
      registerComputed: (...args) => tools.registerComputed(...args),
      computeNames: () => [...specs.keys()],
    };
  }

  const api = {ARTIFACT, addendum, checkCatalog, compose, leaves, matchesEngine, planCompute, registryView, rowFor};
  scope.EAgentAgents = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : window);
