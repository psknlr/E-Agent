/* Browser evidence tools over build-time exports from E-Agent's actual readers. */
(function (scope) {
  'use strict';

  const own = (value, key) => Object.prototype.hasOwnProperty.call(value, key);
  const clone = value => JSON.parse(JSON.stringify(value));
  const plainObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const canonical = args => JSON.stringify(Object.fromEntries(Object.keys(args).sort().map(key => [key, args[key]])));
  const calculateSchema = {
    name: 'calculate', description: 'Compute a finite arithmetic expression locally; no code execution or network.',
    parameters: {expression: 'Arithmetic expression using numbers, + - * / % ^, parentheses, pi, e and approved math functions.'}
  };
  // A number followed by one of these units is a measurement the model must cite.
  // The pattern is case-insensitive so "NM" and "Percent" are caught, but a
  // single-letter unit is a unit only in capitals: angstrom is written "A" and
  // molar "M". A lowercase letter after a number is an identifier (substrate
  // "2a" in the reference set), and reading it as "2 angstroms" refused every
  // sentence that named one. This must stay in step with MEASUREMENT_PATTERN in
  // src/eagent/harness/llm.py.
  const measurement = () => /(?<![\w.+-])([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*(%|percent|angstroms?|A\b|nm|kcal\/mol|kJ\/mol|s-1|s\^-1|\/s|mM|uM|nM|M\b|pLDDT|ee\b|kcat|Km|degrees?|deg\b)/gi;

  const quantities = text => [...text.matchAll(measurement())].filter(match => !/^[am]$/.test(match[2]));

  function bounded(value) {
    if (!Number.isFinite(value) || Math.abs(value) > 1e100) throw new Error('Arithmetic must remain finite and within the magnitude limit.');
    return Object.is(value, -0) ? 0 : value;
  }

  function arithmetic(source) {
    if (typeof source !== 'string' || !source.trim() || source.length > 512) throw new Error('expression must contain at most 512 characters.');
    const tokens = [];
    const tokenPattern = /(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z_0-9]*|\*\*|[()+\-*/%^,]/y;
    let offset = 0;
    while (offset < source.length) {
      if (/\s/.test(source[offset])) { offset += 1; continue; }
      tokenPattern.lastIndex = offset;
      const token = tokenPattern.exec(source);
      if (!token) throw new Error('The arithmetic expression contains an unsupported token.');
      tokens.push(token[0]); offset = tokenPattern.lastIndex;
      if (tokens.length > 128) throw new Error('The arithmetic expression has too many tokens.');
    }
    let position = 0, depth = 0, operations = 0;
    const functions = Object.freeze({
      abs: [1, 1, Math.abs], sqrt: [1, 1, Math.sqrt], log: [1, 1, Math.log],
      log10: [1, 1, Math.log10], exp: [1, 1, Math.exp], floor: [1, 1, Math.floor],
      ceil: [1, 1, Math.ceil], round: [1, 1, Math.round], sin: [1, 1, Math.sin],
      cos: [1, 1, Math.cos], tan: [1, 1, Math.tan],
      min: [1, 16, Math.min], max: [1, 16, Math.max], pow: [2, 2, Math.pow]
    });
    function checked(value) {
      if (++operations > 128) throw new Error('The arithmetic operation limit was reached.');
      return bounded(value);
    }
    function nested(run) {
      if (++depth > 32) throw new Error('The arithmetic expression is nested too deeply.');
      try { return run(); } finally { depth -= 1; }
    }
    function expect(token) {
      if (tokens[position++] !== token) throw new Error('The arithmetic expression is malformed.');
    }
    function primary() {
      const token = tokens[position++];
      if (token === '(') { const value = nested(sum); expect(')'); return value; }
      if (token && /^(?:\d|\.\d)/.test(token)) return bounded(Number(token));
      if (token === 'pi') return Math.PI;
      if (token === 'e') return Math.E;
      if (!token || !own(functions, token)) throw new Error('Only approved arithmetic functions and constants are allowed.');
      expect('(');
      const args = [];
      if (tokens[position] !== ')') {
        args.push(nested(sum));
        while (tokens[position] === ',') { position += 1; args.push(nested(sum)); }
      }
      expect(')');
      const [minimum, maximum, fn] = functions[token];
      if (args.length < minimum || args.length > maximum) throw new Error('The arithmetic function has the wrong number of arguments.');
      return checked(fn(...args));
    }
    function power() {
      const left = primary();
      if (tokens[position] === '^' || tokens[position] === '**') {
        position += 1;
        const right = nested(unary);
        if (Math.abs(right) > 1000) throw new Error('The arithmetic exponent is outside the supported range.');
        return checked(Math.pow(left, right));
      }
      return left;
    }
    function unary() {
      if (tokens[position] === '+' || tokens[position] === '-') {
        const sign = tokens[position++];
        return checked((sign === '-' ? -1 : 1) * nested(unary));
      }
      return power();
    }
    function product() {
      let value = unary();
      while (['*', '/', '%'].includes(tokens[position])) {
        const operator = tokens[position++], right = unary();
        if ((operator === '/' || operator === '%') && right === 0) throw new Error('Division by zero is undefined.');
        value = checked(operator === '*' ? value * right : operator === '/' ? value / right : value % right);
      }
      return value;
    }
    function sum() {
      let value = product();
      while (tokens[position] === '+' || tokens[position] === '-') {
        const operator = tokens[position++], right = product();
        value = checked(operator === '+' ? value + right : value - right);
      }
      return value;
    }
    const value = sum();
    if (position !== tokens.length) throw new Error('The arithmetic expression contains trailing input.');
    return bounded(value);
  }

  // Exact IEEE-754 scaling keeps Python's ties-to-even rounding for citations.
  function rounded(value, decimals) {
    const buffer = new ArrayBuffer(8), view = new DataView(buffer);
    view.setFloat64(0, value, false);
    const bits = view.getBigUint64(0, false), negative = (bits >> 63n) !== 0n;
    const exponent = Number((bits >> 52n) & 2047n);
    const fraction = bits & ((1n << 52n) - 1n);
    let numerator = exponent ? fraction | (1n << 52n) : fraction;
    if (decimals >= 0) numerator *= 10n ** BigInt(decimals);
    const shift = exponent ? exponent - 1075 : -1074;
    let denominator = 1n;
    if (shift < 0) denominator <<= BigInt(-shift); else numerator <<= BigInt(shift);
    if (decimals < 0) denominator *= 10n ** BigInt(-decimals);
    let quotient = numerator / denominator;
    const remainder = numerator % denominator;
    if (remainder * 2n > denominator || remainder * 2n === denominator && quotient % 2n) quotient += 1n;
    return (negative ? -1 : 1) * Number(quotient) / (10 ** decimals);
  }

  function agrees(cited, stored) {
    if (typeof stored !== 'number' || !Number.isFinite(stored)) return false;
    const number = Number(cited), components = cited.toLowerCase().split('e');
    const decimals = (components[0].includes('.') ? components[0].split('.')[1].length : 0) - Number(components[1] || 0);
    if (!Number.isFinite(number)) return false;
    if (Math.abs(decimals) > 100) return Math.abs(number - stored) <= Math.max(Number.MIN_VALUE, 1e-9 * Math.max(Math.abs(number), Math.abs(stored)));
    const actual = rounded(stored, decimals);
    return Math.abs(actual - number) <= Math.max(1e-9 * Math.max(Math.abs(actual), Math.abs(number)), 10 ** -(decimals + 6));
  }

  // ---- routing a question to the stored tools without a model ---------------
  // This is a keyword lookup, not language understanding: it matches ids and
  // words in the question to the tools that can answer them, and it says why
  // each tool ran. It never invents an argument; every id it offers comes from
  // the exported tool data itself.
  const MAX_LOCAL_CALLS = 8;
  const ARITHMETIC_TOKEN = /(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|\*\*|[()+\-*\/%^,]|(?:abs|sqrt|log10|log|exp|floor|ceil|round|sin|cos|tan|min|max|pow)(?=\s*\()|pi(?![A-Za-z0-9_])/y;
  const escapeRegExp = text => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const unique = list => [...new Set(list)];

  // Maximal runs of arithmetic tokens. A number or name glued to letters
  // ("6ZZO", "2a", "api") is part of a word, not an operand.
  function arithmeticRuns(text) {
    const runs = [];
    let current = '', index = 0;
    const flush = () => { const run = current.replace(/^[\s,]+|[\s,]+$/g, ''); if (run) runs.push(run); current = ''; };
    while (index < text.length) {
      if (/\s/.test(text[index])) { current += text[index++]; continue; }
      ARITHMETIC_TOKEN.lastIndex = index;
      const match = ARITHMETIC_TOKEN.exec(text);
      const end = match ? index + match[0].length : index;
      const word = match && /^[A-Za-z]/.test(match[0]), number = match && /^[\d.]/.test(match[0]);
      const glued = match && ((word && index > 0 && /[A-Za-z0-9_]/.test(text[index - 1]))
        || (number && ((index > 0 && /[A-Za-z_]/.test(text[index - 1])) || /[A-Za-z_]/.test(text[end] || ''))));
      if (match && !glued) { current += match[0]; index = end; }
      else { flush(); index = match ? end : index + 1; }
    }
    flush();
    return runs;
  }

  function arithmeticExpression(text) {
    const keyword = /\b(?:calculat\w*|comput\w*|evaluat\w*|calc|what\s+is|how\s+much|result\s+of)\b|=/i.test(text);
    const whole = text.trim().replace(/[\s?=.!]+$/g, '');
    let best = '';
    for (const run of arithmeticRuns(text)) {
      if (!/[-+*\/%^(]/.test(run.replace(/^[+-]\s*/, '')) || !/\d|pi/.test(run)) continue;
      if (/^\d{4}-\d{1,2}-\d{1,2}$|^\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4}$/.test(run)) continue;
      // A bare "2024-10-10" is a date, not a sum, unless the user asked to calculate.
      if (!(keyword || /[*\/%^(]/.test(run) || whole === run)) continue;
      try { arithmetic(run); } catch (_) { continue; }
      if (run.length > best.length) best = run;
    }
    return best;
  }

  function parseCitations(text) {
    const citations = [], problems = [];
    for (const match of text.matchAll(/\[cite(?:\s+([^\]]*))?\]/gi)) {
      const fields = Object.create(null);
      let duplicate = false;
      for (const pair of (match[1] || '').matchAll(/(\w+)\s*=\s*(?:"([^"]*)"|(\S+?))(?=\s+\w+\s*=|\s*$)/g)) {
        const key = pair[1].toLowerCase();
        if (own(fields, key)) duplicate = true;
        fields[key] = (pair[2] ?? pair[3]).trim();
      }
      const missing = ['artifact', 'sha256', 'row', 'field', 'method'].filter(key => !fields[key]);
      if (duplicate || missing.length || !/^[0-9a-f]{12,64}$/i.test(fields.sha256 || '')) {
        problems.push(`${match[0]}: malformed citation${missing.length ? '; missing ' + missing.join(', ') : ''}.`);
        continue;
      }
      citations.push({...fields, sha256: fields.sha256.toLowerCase(), start: match.index, end: match.index + match[0].length});
    }
    return {citations, problems};
  }

  function create(bundle) {
    if (!plainObject(bundle) || bundle.schema_version !== 1 || bundle.format !== 'eagent.browser-reference-tools'
        || typeof bundle.system_prompt !== 'string' || !Array.isArray(bundle.schemas)
        || !plainObject(bundle.results) || !plainObject(bundle.citation_rows)
        || !/^[0-9a-f]{64}$/.test(bundle.reference_digest || '') || !/^[0-9a-f]{64}$/.test(bundle.bundle_digest || '')) {
      throw new Error('The browser reference bundle is invalid or has an unsupported version.');
    }
    for (const artifact of Object.values(bundle.citation_rows)) {
      if (!plainObject(artifact)) throw new Error('The browser citation index is invalid.');
      for (const row of Object.values(artifact)) {
        if (!plainObject(row) || !/^[0-9a-f]{64}$/.test(row.sha256 || '') || !plainObject(row.fields)
            || !Array.isArray(row.field_names) || row.field_names.some(name => typeof name !== 'string')) {
          throw new Error('The browser citation index is invalid.');
        }
      }
    }
    // The snapshot and per-session computations never expose mutable source rows.
    const data = clone(bundle), schemaList = [...data.schemas, clone(calculateSchema)];
    const schemaMap = new Map(schemaList.map(schema => [schema.name, schema]));
    const computedRows = Object.create(null);
    api.systemPrompt = data.system_prompt;

    async function execute(call) {
      const started = Date.now();
      const name = typeof call?.interface === 'string' ? call.interface : '';
      const args = call?.arguments === undefined ? {} : call.arguments;
      const rationale = typeof call?.rationale === 'string' ? call.rationale : '';
      const result = {tool: name, arguments: {}, rationale,
        ok: false, value: null, refusal: '', truncated: false, elapsed_ms: 0};
      try {
        if (!schemaMap.has(name)) throw new Error('No such browser tool is registered.');
        if (!plainObject(args)) throw new Error('Tool arguments must be an object.');
        const parameters = schemaMap.get(name).parameters;
        if (Object.keys(args).some(key => !own(parameters, key))) throw new Error('The tool received an unsupported argument.');
        if (Object.values(args).some(value => typeof value !== 'string')) throw new Error('Tool arguments must be strings.');
        if (Object.values(args).some(value => value.length > 512)) throw new Error('Tool argument exceeds the character limit.');
        result.arguments = clone(args);
        if (name === 'calculate') {
          if (!own(args, 'expression')) throw new Error('calculate requires expression.');
          const value = arithmetic(args.expression);
          if (!scope.crypto?.subtle) throw new Error('Secure calculation hashing is unavailable in this browser.');
          const payload = JSON.stringify({expression: args.expression, value});
          const digest = [...new Uint8Array(await scope.crypto.subtle.digest('SHA-256', new TextEncoder().encode(payload)))]
            .map(byte => byte.toString(16).padStart(2, '0')).join('');
          const row = 'calculation-' + digest.slice(0, 16);
          computedRows[row] = {sha256: digest, fields: {value}, field_names: ['value']};
          result.value = {expression: args.expression, value, computed: true,
            method: 'bounded arithmetic parser; no scientific measurement',
            cite: {artifact: 'browser_calculation', sha256: digest, row, field: 'value', method: 'read'}};
          result.ok = true;
        } else {
          if (!own(data.results, name)) throw new Error('No data was exported for this tool.');
          const key = canonical(args), exported = data.results[name];
          if (!own(exported, key)) throw new Error('The requested ID, tier or argument combination is not in the curated reference set.');
          Object.assign(result, clone(exported[key]));
        }
      } catch (error) {
        result.refusal = error instanceof Error ? error.message : 'The browser tool refused the request.';
      }
      result.elapsed_ms = Date.now() - started;
      return result;
    }

    function inspect(text) {
      const report = {clean: false, fully_verified: false, summary: '', uncited_quantities: [],
        cited_quantities: [], verified_quantities: [], declared_quantities: [], broken_citations: []};
      if (typeof text !== 'string') { report.broken_citations.push('Model output must be text.'); text = ''; }
      const parsed = parseCitations(text);
      report.broken_citations.push(...parsed.problems);
      const validRows = new Map();
      for (const citation of parsed.citations) {
        let row;
        if (citation.artifact === 'browser_calculation') row = own(computedRows, citation.row) ? computedRows[citation.row] : null;
        else if (own(data.citation_rows, citation.artifact) && own(data.citation_rows[citation.artifact], citation.row)) {
          row = data.citation_rows[citation.artifact][citation.row];
        }
        const method = citation.method.split(':', 1)[0].trim().toLowerCase();
        if (!row || !row.sha256.startsWith(citation.sha256)) {
          report.broken_citations.push('Citation names an unknown artifact/row or mismatched data digest.');
        } else if (!row.field_names.includes(citation.field)) {
          report.broken_citations.push('Citation names a field absent from the exported source row.');
        } else if (!['read', 'rounded'].includes(method)) {
          report.broken_citations.push('An uncomputed derivation must first use calculate.');
        } else validRows.set(citation, row);
      }
      for (const old of text.matchAll(/\[(?:artifact|table|file|record|evidence):[^\]]+\]/gi)) {
        report.broken_citations.push(`${old[0]}: legacy citation does not identify a version, row and field.`);
      }
      const measuredCitations = new Set();
      const citationRanges = [...text.matchAll(/\[cite(?:\s+[^\]]*)?\]/gi)]
        .map(match => [match.index, match.index + match[0].length]);
      for (const match of quantities(text)) {
        // Row IDs, hashes and field names describe provenance; their digits
        // are not measurements in the model's prose.
        if (citationRanges.some(([start, end]) => match.index >= start && match.index < end)) continue;
        const token = `${match[1]} ${match[2]}`, end = match.index + match[0].length;
        const citation = parsed.citations.find(item => item.start >= end);
        if (!citation || quantities(text.slice(end, citation.start)).length) {
          report.uncited_quantities.push(token); continue;
        }
        report.cited_quantities.push(token);
        measuredCitations.add(citation);
        const row = validRows.get(citation);
        if (!row) continue;
        if (!own(row.fields, citation.field) || !agrees(match[1], row.fields[citation.field])) {
          report.broken_citations.push(`${token}: the cited field is withheld, ambiguous, absent or contradicts the reference value.`); continue;
        }
        report.verified_quantities.push(token);
      }
      // Arithmetic and record counts may be unitless. Validate a cited plain
      // number too, without treating every uncited conversational number as a
      // claimed experimental measurement.
      let previousEnd = 0;
      for (const citation of parsed.citations) {
        const segment = text.slice(previousEnd, citation.start);
        previousEnd = citation.end;
        if (measuredCitations.has(citation) || !validRows.has(citation)) continue;
        const direct = segment.match(/(?<![\w.+-])([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*(?:(?:records?|enzymes?|constructs?|entries|samples?|lineages?|groups?|items?|structures?|pairs?)\s*)?$/i);
        if (!direct) continue;
        const number = direct[1], row = validRows.get(citation);
        report.cited_quantities.push(number);
        if (!own(row.fields, citation.field) || !agrees(number, row.fields[citation.field])) {
          report.broken_citations.push(`${number}: the cited plain number contradicts or is withheld by its source.`);
        } else report.verified_quantities.push(number);
      }
      report.clean = report.uncited_quantities.length === 0 && report.broken_citations.length === 0;
      report.fully_verified = report.clean && report.declared_quantities.length === 0;
      report.summary = `${report.verified_quantities.length} verified, ${report.declared_quantities.length} declared but unchecked, `
        + `${report.uncited_quantities.length} uncited, ${report.broken_citations.length} broken citation(s)`;
      return report;
    }

    // The ids the exported tools can answer for, read from the exported
    // argument keys so a router can only ever offer what execute() can serve.
    function argumentValues(tool, field) {
      const values = [];
      for (const key of own(data.results, tool) ? Object.keys(data.results[tool]) : []) {
        try { const value = JSON.parse(key)[field]; if (typeof value === 'string' && value) values.push(value); } catch (_) { /* Not an argument key. */ }
      }
      return unique(values);
    }
    const ids = () => ({pdb_ids: argumentValues('structure_entry', 'pdb_id'), label_ids: argumentValues('kinetic_record', 'label_id'),
      enzyme_ids: argumentValues('activity_endpoint', 'enzyme_id'), substrate_ids: argumentValues('activity_endpoint', 'substrate_id'),
      tiers: argumentValues('list_kinetic_records', 'tier')});

    function plan(text) {
      const calls = [], notes = [];
      if (typeof text !== 'string' || !text.trim()) return {calls, notes: ['Enter a question or an id.']};
      const lower = text.toLowerCase(), known = ids();
      const mentions = id => new RegExp('(?<![A-Za-z0-9_-])' + escapeRegExp(id.toLowerCase()) + '(?![A-Za-z0-9_-])').test(lower);
      const add = (name, args, why) => {
        if (calls.some(call => call.interface === name && canonical(call.arguments) === canonical(args))) return;
        if (calls.length >= MAX_LOCAL_CALLS) { if (!notes.length) notes.push(`Only the first ${MAX_LOCAL_CALLS} lookups were run.`); return; }
        calls.push({interface: name, arguments: args, rationale: why});
      };

      const expression = arithmeticExpression(text);
      if (expression) add('calculate', {expression}, `The message contains the arithmetic expression “${expression}”.`);

      const pdbIds = unique((text.match(/(?<![A-Za-z0-9_])[0-9][A-Za-z0-9]{3}(?![A-Za-z0-9_])/g) || []).map(id => id.toUpperCase())).filter(id => known.pdb_ids.includes(id));
      if (/\b(?:structures?|pdb|crystal)\b/.test(lower) && (/\b(?:list|show|enumerate|available|which|all|audited)\b/.test(lower) || !pdbIds.length)) {
        add('list_structure_entries', {}, 'The message asks about the audited structures.');
      }
      for (const id of pdbIds) add('structure_entry', {pdb_id: id}, `“${id}” is an audited structure id named in the message.`);

      const labels = known.label_ids.filter(mentions);
      for (const id of labels) add('kinetic_record', {label_id: id}, `“${id}” is a kinetic record id named in the message.`);
      if (!labels.length && /\b(?:kinetics?|kcat|km|turnover|michaelis)\b/.test(lower)) {
        const tier = known.tiers.find(name => new RegExp('\\b' + escapeRegExp(name) + '\\b').test(lower)) || '';
        add('list_kinetic_records', {tier}, tier ? `The message asks about the ${tier}-tier kinetic records.` : 'The message asks about the kinetic records.');
      }

      const enzymes = known.enzyme_ids.filter(mentions), substrates = known.substrate_ids.filter(mentions);
      const activity = /\b(?:activity|activities|orthologs?|enantio\w*|ee|conversion|censor\w*|detection limit)\b/.test(lower);
      // An independence question is answered by its own tool, not by the generic activity summary too.
      const independence = /\bindependen\w*/.test(lower);
      if (enzymes.length) {
        for (const enzyme of enzymes.slice(0, 2)) {
          for (const substrate of substrates.length ? substrates : known.substrate_ids) {
            add('activity_endpoint', {enzyme_id: enzyme, substrate_id: substrate},
              `“${enzyme}” is an ortholog id and ${substrates.length ? `“${substrate}” a substrate id` : 'no substrate was named, so every substrate is shown'}.`);
          }
        }
      } else if (activity && !independence) {
        if (/\b(?:constructs?|list|which)\b/.test(lower)) add('list_activity_constructs', {}, 'The message asks which constructs have activity data.');
        else add('activity_summary', {}, 'The message asks about the ortholog activity data.');
      }
      if (independence) {
        if (activity || enzymes.length) add('activity_independence_groups', {}, 'The message asks about independence of the soluble orthologs.');
        else add('reference_summary', {}, 'The message asks about independence, which the reference summary reports.');
      }
      if (/\b(?:audit|eligib\w*|calibrat\w*|verdict)\b/.test(lower)) add('audit_verdict', {}, 'The message asks for the audit or calibration verdict.');
      if (/\b(?:summary|overview|reference set|how many|counts?)\b/.test(lower)) add('reference_summary', {}, 'The message asks for an overview of the reference set.');
      if (/\b(?:source verification|verif(?:y|ied|ication)|printed table)\b/.test(lower)) add('source_verification', {}, 'The message asks whether the numbers match their sources.');

      if (!calls.length) {
        const enzyme = known.enzyme_ids.includes('Ssal-KRED') ? 'Ssal-KRED' : known.enzyme_ids[0];
        notes.push('I could not match this to a stored tool. This looks words and ids up directly; it does not interpret free text the way a model does. Try a structure id such as '
          + `${known.pdb_ids[0] || '6ZZO'}, “list the structures”, “${enzyme} activity on ${known.substrate_ids[0] || '2a'}”, “kinetic records”, “audit verdict”, “summary”, or an expression such as (2.5 + 3.5) * 4.`);
      }
      return {calls, notes};
    }

    return {systemPrompt: data.system_prompt, bundleDigest: data.bundle_digest,
      names: schemaList.map(schema => schema.name), toolNames: () => schemaList.map(schema => schema.name),
      schemas: () => clone(schemaList), execute, inspect, guard: inspect, ids, plan};
  }

  const api = {create, systemPrompt: ''};
  scope.EAgentBrowserTools = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : window);
