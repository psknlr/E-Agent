"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { webcrypto } = require("node:crypto");

const project = path.resolve(__dirname, "..");
const bundle = JSON.parse(fs.readFileSync(path.join(project, "web/reference-data.json"), "utf8"));
const key = "test-browser-only-api-key-secret";
const config = (changes = {}) => ({ provider: "minimax", protocol: "openai_chat", api_key: key, model: "model-example", api_url: "https://provider.example/v1/chat/completions", ...changes });
const reply = (turn, { anthropic = false, finish = "stop", thinking = "" } = {}) => new Response(JSON.stringify(anthropic ? { content: [{ type: "thinking", thinking }, { type: "text", text: typeof turn === "string" ? turn : JSON.stringify(turn) }], stop_reason: finish } : { choices: [{ finish_reason: finish, message: { content: thinking + (typeof turn === "string" ? turn : JSON.stringify(turn)) } }] }), { headers: { "Content-Type": "application/json" } });

function harness(provider, { reference = bundle, timers, changeTools } = {}) {
  const calls = [];
  const scheduled = [];
  const context = vm.createContext({ window: {}, URL, TextEncoder, TextDecoder, AbortController, Response, Headers, Uint8Array, performance, crypto: webcrypto, console,
    setTimeout: (callback, delay) => { scheduled.push(delay); return setTimeout(callback, timers ? timers(delay) : delay); }, clearTimeout,
    fetch: async (url, options) => { calls.push({ url, options }); return url === "./reference-data.json" ? new Response(JSON.stringify(reference), { headers: { "Content-Type": "application/json" } }) : provider(url, options); },
  });
  context.window = context;
  vm.runInContext(fs.readFileSync(path.join(project, "web/endpoint.js"), "utf8"), context);
  vm.runInContext(fs.readFileSync(path.join(project, "web/reference-tools.js"), "utf8"), context);
  if (changeTools) {
    const original = context.window.EAgentBrowserTools;
    context.window.EAgentBrowserTools = { create: (data) => changeTools(original.create(data)) };
  }
  vm.runInContext(fs.readFileSync(path.join(project, "web/browser-agent.js"), "utf8"), context);
  return { runtime: context.window.EAgentBrowserRuntime, calls, scheduled, context };
}

test("prepare loads only the public same-origin reference bundle and validates its schema", async () => {
  const { runtime, calls } = harness(() => { throw new Error("No API request should occur."); });
  const metadata = await runtime.prepare();
  assert.equal(metadata.runtime_ready, true);
  assert.equal(metadata.execution, "browser");
  assert.ok(metadata.tools.includes("kinetic_record"));
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "./reference-data.json");
  assert.equal(calls[0].options.mode, "same-origin");
  assert.equal(calls[0].options.credentials, "omit");
  assert.equal(calls[0].options.headers.Authorization, undefined);
  await runtime.prepare();
  assert.equal(calls.length, 1);
  const invalid = harness(() => reply({ reasoning: "unused" }), { reference: { ...bundle, schema_version: 999 } });
  await assert.rejects(invalid.runtime.prepare(), /unsupported schema/);
});

test("MiniMax, GPT, Claude and both custom API formats send the selected model, key and URL", async (parent) => {
  for (const [provider, protocol] of [["minimax", "openai_chat"], ["minimax_cn", "openai_chat"], ["openai", "openai_chat"], ["anthropic", "anthropic_messages"], ["custom", "openai_chat"], ["custom", "anthropic_messages"]]) {
    const miniMax = provider.startsWith("minimax");
    await parent.test(provider + " " + protocol, async () => {
      const chosen = config({ provider, protocol });
      const { runtime, calls } = harness(() => reply({ reasoning: "Ready to inspect the reference evidence." }, { anthropic: protocol === "anthropic_messages", thinking: miniMax ? "<think>private thinking</think>" : "" }));
      const result = await runtime.chat({ message: "Can you inspect enzyme evidence?", model_config: chosen });
      assert.equal(result.completion_verified, true);
      assert.equal(result.answer, "Ready to inspect the reference evidence.");
      assert.equal(result.model_config.model, chosen.model);
      assert.equal(result.model_config.api_key, undefined);
      const request = calls.find((entry) => entry.url !== "./reference-data.json");
      const body = JSON.parse(request.options.body);
      assert.equal(request.url, chosen.api_url);
      assert.equal(body.model, chosen.model);
      assert.equal(body.api_key, undefined);
      assert.equal(body.temperature, undefined);
      assert.equal(body.seed, undefined);
      assert.equal(request.options.credentials, "omit");
      assert.equal(request.options.redirect, "error");
      assert.equal(request.options.mode, "cors");
      assert.equal(request.options.referrerPolicy, "no-referrer");
      if (protocol === "anthropic_messages") {
        assert.equal(request.options.headers["x-api-key"], key);
        assert.equal(request.options.headers["anthropic-version"], "2023-06-01");
        assert.equal(request.options.headers["anthropic-dangerous-direct-browser-access"], "true");
        assert.equal(request.options.headers.Authorization, undefined);
        assert.equal(body.max_tokens, 8192);
        assert.equal(body.messages[0].role, "user");
        assert.match(body.system, /Available tool interfaces/);
      } else {
        assert.equal(request.options.headers.Authorization, "Bearer " + key);
        assert.equal(body[miniMax ? "max_completion_tokens" : "max_tokens"], 8192);
        assert.equal(body.messages[0].role, "system");
        assert.equal(body.reasoning_split, miniMax ? true : undefined);
      }
    });
  }
});

test("a pasted base URL is completed, so the request reaches the real chat endpoint", async (parent) => {
  for (const [provider, protocol, typed, expected] of [
    ["minimax_cn", "openai_chat", "https://api.minimax.cn/v1", "https://api.minimax.cn/v1/chat/completions"],
    ["minimax_cn", "openai_chat", "https://api.minimax.cn/v1/", "https://api.minimax.cn/v1/chat/completions"],
    ["openai", "openai_chat", "https://api.openai.com/v1", "https://api.openai.com/v1/chat/completions"],
    ["custom", "openai_chat", "https://gateway.example/api/v4", "https://gateway.example/api/v4/chat/completions"],
    ["anthropic", "anthropic_messages", "https://api.anthropic.com/v1", "https://api.anthropic.com/v1/messages"],
    ["minimax_cn", "openai_chat", "https://api.minimax.cn/v1/chat/completions", "https://api.minimax.cn/v1/chat/completions"],
  ]) {
    await parent.test(provider + " " + typed, async () => {
      const { runtime, calls } = harness(() => reply({ reasoning: "Ready." }, { anthropic: protocol === "anthropic_messages" }));
      const result = await runtime.chat({ message: "Can you inspect enzyme evidence?", model_config: config({ provider, protocol, api_url: typed }) });
      const request = calls.find((entry) => entry.url !== "./reference-data.json");
      assert.equal(request.url, expected);
      assert.equal(result.model_config.api_url, expected);
    });
  }
});

// ---- connection test ----------------------------------------------------
// A real browser reports a CORS block, a wrong path with no CORS headers, a
// redirect and a dead host with one bare TypeError. These pin how the test
// tells them apart, and that it can only ever carry a placeholder credential.
const json = (body, status) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const refuses = () => { throw new TypeError("Failed to fetch"); };

test("connection test: an answered request means network and CORS are fine, whatever the status", async (parent) => {
  for (const [status, body, verdict] of [
    [401, { error: { message: "Incorrect API key provided" } }, "reachable"],
    [403, {}, "reachable"],
    [400, { base_resp: { status_code: 2013, status_msg: "invalid params" } }, "reachable"],
    [200, {}, "reachable"],
    [404, { error: { message: "no route" } }, "wrong_path"],
    [405, {}, "wrong_path"],
    [429, {}, "provider_error"],
    [503, {}, "provider_error"],
  ]) {
    await parent.test("HTTP " + status, async () => {
      const { runtime } = harness(() => json(body, status));
      const result = await runtime.testConnection({ api_url: "https://provider.example/v1/chat/completions", protocol: "openai_chat" });
      assert.equal(result.verdict, verdict);
      assert.equal(result.status, status);
      assert.equal(result.endpoint, "https://provider.example/v1/chat/completions");
      assert.match(result.summary, new RegExp("HTTP " + status));
    });
  }
});

test("connection test: the provider's own error text is shown, bounded", async () => {
  const { runtime } = harness(() => json({ error: { message: "Incorrect API key provided: " + "x".repeat(500) } }, 401));
  const result = await runtime.testConnection({ api_url: "https://provider.example/v1/chat/completions", protocol: "openai_chat" });
  assert.match(result.advice, /Incorrect API key provided/);
  assert.ok(result.advice.length < 600);
});

test("connection test: a refused reply from a reachable host is CORS or a wrong path, not an unreachable host", async () => {
  const { runtime, calls } = harness((url, options) => { if (options.mode === "no-cors") return new Response(null); return refuses(); });
  const result = await runtime.testConnection({ api_url: "https://provider.example/v1/nope", protocol: "openai_chat" });
  assert.equal(result.verdict, "blocked");
  assert.match(result.advice, /CORS/);
  assert.match(result.advice, /Run in browser/);
  const probe = calls.at(-1);
  assert.equal(probe.url, "https://provider.example/");
  assert.equal(probe.options.method, "GET");
  assert.equal(probe.options.mode, "no-cors");
  assert.equal(probe.options.credentials, "omit");
  assert.equal(probe.options.headers, undefined, "the reachability probe carries no headers at all");
  assert.equal(probe.options.body, undefined);
});

test("connection test: a host that does not answer even a bare request is unreachable", async () => {
  const { runtime, calls } = harness(refuses);
  const result = await runtime.testConnection({ api_url: "https://api.minimax.cn/v1", protocol: "openai_chat" });
  assert.equal(result.verdict, "unreachable");
  assert.match(result.summary, /api\.minimax\.cn/);
  assert.equal(result.endpoint, "https://api.minimax.cn/v1/chat/completions");
  assert.equal(calls.length, 2, "one real attempt and one reachability probe");
});

test("connection test: a silent host times out without a second probe", async () => {
  const { runtime, calls } = harness((url, options) => new Promise((resolve, reject) => options.signal.addEventListener("abort", () => reject(new TypeError("aborted")), { once: true })), { timers: () => 1 });
  const result = await runtime.testConnection({ api_url: "https://provider.example/v1/chat/completions", protocol: "openai_chat" });
  assert.equal(result.verdict, "timeout");
  assert.equal(calls.length, 1);
});

test("connection test: sends a placeholder, never a credential, in the chosen provider's format", async (parent) => {
  for (const [protocol, url] of [["openai_chat", "https://provider.example/v1/chat/completions"], ["anthropic_messages", "https://provider.example/v1/messages"]]) {
    await parent.test(protocol, async () => {
      const { runtime, calls } = harness(() => json({}, 401));
      await runtime.testConnection({ api_url: url, protocol });
      const request = calls.find((entry) => entry.url === url);
      assert.equal(request.options.method, "POST");
      assert.equal(request.options.body, "{}");
      assert.equal(request.options.mode, "cors");
      assert.equal(request.options.credentials, "omit");
      assert.equal(request.options.redirect, "error");
      assert.equal(request.options.referrerPolicy, "no-referrer");
      const sent = JSON.stringify(request.options.headers);
      assert.equal(sent.includes(key), false);
      if (protocol === "anthropic_messages") {
        assert.equal(request.options.headers["x-api-key"], "eagent-connection-test");
        assert.equal(request.options.headers["anthropic-dangerous-direct-browser-access"], "true");
        assert.equal(request.options.headers.Authorization, undefined);
      } else assert.equal(request.options.headers.Authorization, "Bearer eagent-connection-test");
    });
  }
});

test("connection test: completes a pasted base URL and rejects unsafe or unsupported input", async () => {
  const { runtime, calls } = harness(() => json({}, 401));
  const result = await runtime.testConnection({ api_url: "https://api.minimax.cn/v1", protocol: "openai_chat" });
  assert.equal(result.endpoint, "https://api.minimax.cn/v1/chat/completions");
  assert.equal(calls[0].url, "https://api.minimax.cn/v1/chat/completions");
  for (const changes of [{ api_url: "http://provider.example/v1" }, { api_url: "https://provider.example/" }, { api_url: "https://user:pw@provider.example/v1" }, { api_url: "https://provider.example/v1?key=x" }, { api_url: "" }, { protocol: "unsupported" }]) {
    // Errors come from the vm realm, so match by message rather than by class.
    await assert.rejects(runtime.testConnection({ api_url: "https://provider.example/v1", protocol: "openai_chat", ...changes }), /HTTPS API endpoint|complete model API URL|supported API format/, JSON.stringify(changes));
  }
});

test("connection test: a cancelled request is reported as cancelled, not as a network verdict", async () => {
  const controller = new AbortController();
  const { runtime } = harness((url, options) => new Promise((resolve, reject) => options.signal.addEventListener("abort", () => reject(new TypeError("aborted")), { once: true })));
  const pending = runtime.testConnection({ api_url: "https://provider.example/v1/chat/completions", protocol: "openai_chat", signal: controller.signal });
  controller.abort();
  await assert.rejects(pending, /cancelled/i);
});

// ---- Run in browser: the stored tools without any model --------------------
const neverCalled = () => { throw new Error("No model API request should occur."); };
// Values built inside the vm context have that realm's prototypes, which strict
// deep equality rejects; compare their data, not their realm.
const plain = (value) => JSON.parse(JSON.stringify(value));

test("local run answers the 6ZZO question from the stored tools with no model, key or API request", async () => {
  const { runtime, calls } = harness(neverCalled);
  const result = await runtime.local({ message: "List the audited structures and inspect 6ZZO. What evidence and limitations are recorded for its cofactor and ligand?" });
  assert.deepEqual(calls.map((entry) => entry.url), ["./reference-data.json"], "the only request is the same-origin bundle");
  assert.equal(result.completion_verified, false, "a local run is never a model completion");
  assert.equal(result.execution, "browser-local");
  const results = result.transcript.turns[0].results;
  assert.deepEqual(plain(results.map((entry) => entry.tool)), ["list_structure_entries", "structure_entry"]);
  assert.equal(result.transcript.tool_calls_made, 2);
  assert.equal(results.every((entry) => entry.ok === true && entry.rationale.length > 10), true);
  assert.deepEqual(plain(results[1].value), bundle.results.structure_entry['{"pdb_id":"6ZZO"}'].value, "the record is shown exactly as stored");
  assert.equal(result.transcript.provider, "none");
  assert.equal(result.transcript.runs_remotely, false);
  assert.match(result.answer, /^Ran 2 stored-reference tools in this browser\./);
  assert.match(result.answer, /No model or API was used and nothing was interpreted/);
  assert.match(result.transcript.caveat, /No model was called/);
});

test("local run computes arithmetic with a citation the numeric guard accepts", async () => {
  const { runtime, calls } = harness(neverCalled);
  const result = await runtime.local({ message: "calculate (2.5 + 3.5) * 4" });
  const [entry] = result.transcript.turns[0].results;
  assert.equal(entry.tool, "calculate");
  assert.equal(entry.value.value, 24);
  assert.equal(entry.value.computed, true);
  assert.equal(entry.value.cite.artifact, "browser_calculation");
  assert.equal(calls.length, 1);
});

test("local run says it did not understand, rather than guessing a tool", async () => {
  const { runtime } = harness(neverCalled);
  const result = await runtime.local({ message: "Why is the sky blue?" });
  assert.equal(result.transcript.tool_calls_made, 0);
  assert.deepEqual(plain(result.transcript.turns[0].results), []);
  assert.match(result.answer, /could not match this to a stored tool/);
  assert.equal(result.completion_verified, false);
});

test("local run reports a refused tool call as a refusal, and rejects bad input and cancellation", async () => {
  const refusing = harness(neverCalled, { changeTools: (tools) => ({ ...tools, execute: async () => { throw new Error("The tool refused."); } }) });
  const refused = await refusing.runtime.local({ message: "inspect 6ZZO" });
  assert.equal(refused.transcript.turns[0].results[0].ok, false);
  assert.match(refused.transcript.turns[0].results[0].refusal, /The tool refused/);

  const { runtime } = harness(neverCalled);
  for (const message of ["", "   ", "x".repeat(8001), undefined, 7]) await assert.rejects(runtime.local({ message }), /nonempty question/, String(message).slice(0, 20));
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(runtime.local({ message: "inspect 6ZZO", signal: controller.signal }), /cancelled/i);

  const stale = harness(neverCalled, { changeTools: (tools) => { const { plan, ...rest } = tools; return rest; } });
  await assert.rejects(stale.runtime.local({ message: "inspect 6ZZO" }), /cannot route questions/);
});

// The first real model run refused with 'Uncited quantities: ["2 a"]': the guard
// read substrate id "2a" as "2 angstroms" in the reasoning a model writes before
// it has called any tool, so no tool ever ran. This drives that exact exchange.
test("a model that names substrate 2a before calling the tool is not refused, and its cited answer verifies", async () => {
  let turn = 0;
  const { runtime } = harness((url, options) => {
    const body = JSON.parse(options.body);
    if (turn++ === 0) {
      return reply({ reasoning: "I will inspect Ssal-KRED activity on substrate 2a first.", tool_calls: [{ interface: "activity_endpoint", arguments: { enzyme_id: "Ssal-KRED", substrate_id: "2a" }, rationale: "Read the stored record for this pair." }] });
    }
    const latest = body.messages.at(-1).content;
    assert.ok(latest.startsWith("Tool results:\n"));
    const record = JSON.parse(latest.slice("Tool results:\n".length))[0];
    assert.equal(record.ok, true);
    const c = record.value.cite;
    return reply({ reasoning: `Ssal-KRED on substrate 2a shows 12.5 ee [cite artifact=${c.artifact} sha256=${c.sha256} row=${c.row} field=ee_reported method=read].` });
  });
  const result = await runtime.chat({ message: "Inspect Ssal-KRED activity on 2a. Explain missing values and detection limits before comparing it with other constructs.", model_config: config() });
  assert.equal(result.completion_verified, true);
  assert.equal(result.transcript.tool_calls_made, 1);
  assert.equal(result.transcript.turns[0].guard.clean, true, "the planning turn names 2a and must not be refused");
  assert.equal(result.transcript.turns[0].results[0].ok, true);
  assert.equal(result.transcript.turns[1].guard.clean, true);
  assert.deepEqual(plain(result.transcript.turns[1].guard.verified_quantities), ["12.5 ee"]);
  assert.match(result.answer, /substrate 2a shows 12\.5 ee/);
});

test("the model drives the real local loader, sees its withheld evidence, and retains follow-up context", async () => {
  let round = 0;
  const { runtime } = harness((url, options) => {
    const body = JSON.parse(options.body);
    if (round++ === 0) {
      assert.match(body.messages[1].content, /Previous conversation/);
      assert.match(body.messages[1].content, /Inspect PaHBDH/);
      return reply({ tool_calls: [{ interface: "kinetic_record", arguments: { label_id: "PaHBDH_H150N_AAE_activity_only" }, rationale: "Inspect the curated row." }] });
    }
    const results = JSON.parse(body.messages.at(-1).content.split("\n").slice(1).join("\n"));
    assert.equal(results[0].tool, "kinetic_record");
    assert.equal(results[0].ok, true);
    assert.equal(results[0].value.km, null);
    assert.match(results[0].value.refusals.km, /bound/i);
    return reply({ reasoning: "The loader withholds this Km because it is a bound." });
  });
  const result = await runtime.chat({ message: "What about its Km?", history: [{ role: "user", content: "Inspect PaHBDH." }, { role: "assistant", content: "I can inspect that record." }], model_config: config() });
  assert.equal(result.transcript.tool_calls_made, 1);
  assert.equal(result.transcript.turns[0].results[0].ok, true);
  assert.equal(result.transcript.stopped_because, "the model answered without asking for another tool");
  assert.equal(result.completion_verified, true);
});

test("the model can execute real bounded arithmetic and cite its instance-local calculation", async () => {
  let requests = 0;
  const { runtime } = harness((url, options) => {
    if (++requests === 1) return reply({ tool_calls: [{ interface: "calculate", arguments: { expression: "2 + 3" } }] });
    const result = JSON.parse(JSON.parse(options.body).messages.at(-1).content.slice("Tool results:\n".length))[0];
    assert.equal(result.ok, true);
    assert.equal(result.value.value, 5);
    const cite = result.value.cite;
    return reply({ reasoning: "Computed value: 5 [cite artifact=" + cite.artifact + " sha256=" + cite.sha256 + " row=" + cite.row + " field=value method=read]." });
  });
  const result = await runtime.chat({ message: "Compute the expression using the local arithmetic tool.", model_config: config() });
  assert.equal(result.completion_verified, true);
  assert.equal(result.transcript.turns.at(-1).guard.fully_verified, true);
  assert.equal(result.transcript.turns.at(-1).guard.verified_quantities.length, 1);
});

test("strict local guard refuses unsourced and forged-citation quantities", async (parent) => {
  for (const reasoning of ["Its kcat is 30 s-1.", "Its kcat is 30 s-1 [cite artifact=unknown sha256=" + "a".repeat(64) + " row=unknown field=kcat method=read]."]) {
    await parent.test(reasoning.slice(0, 30), async () => {
      const { runtime } = harness(() => reply({ reasoning }));
      await assert.rejects(runtime.chat({ message: "What is its kcat?", model_config: config() }), (error) => {
        assert.match(error.message, /numeric guard refused/);
        assert.equal(error.data.completion_verified, false);
        assert.equal(error.data.transcript.answer, "");
        assert.equal(error.data.transcript.turns[0].guard.refused, true);
        return true;
      });
    });
  }
});

test("failed model requests keep completed tool evidence and scrub literal and JSON-escaped keys", async () => {
  const specialKey = 'fake-quote"slash\\browser-key';
  let count = 0;
  const { runtime } = harness(() => count++ === 0 ? reply({ tool_calls: [{ interface: "reference_summary", arguments: {} }] }) : new Response(JSON.stringify({ error: specialKey }), { status: 401 }));
  await assert.rejects(runtime.chat({ message: "Read the references.", model_config: config({ api_key: specialKey }) }), (error) => {
    assert.equal(error.status, 401);
    assert.equal(error.data.completion_verified, false);
    assert.equal(error.data.transcript.tool_calls_made, 1);
    assert.equal(error.data.transcript.turns[0].results[0].ok, true);
    assert.ok(error.data.transcript.turns[1].provider_error);
    assert.equal(error.data.transcript.answer, "");
    assert.ok(!JSON.stringify(error.data).includes(specialKey));
    assert.ok(!JSON.stringify(error.data).includes(JSON.stringify(specialKey).slice(1, -1)));
    assert.match(error.message, /credential redacted/);
    return true;
  });
});

test("success answers, user text, progress and transcripts do not expose the supplied key", async () => {
  const specialKey = 'fake-quote"slash\\browser-key';
  const escaped = JSON.stringify(specialKey).slice(1, -1);
  const progress = [];
  const { runtime } = harness(() => reply({ reasoning: "Credential echoes: " + specialKey + " and " + escaped }));
  const result = await runtime.chat({ message: "Echo " + specialKey, model_config: config({ api_key: specialKey }), onProgress: (text) => progress.push(text) });
  const serialized = JSON.stringify(result);
  assert.ok(!serialized.includes(specialKey));
  assert.ok(!serialized.includes(escaped));
  assert.match(result.answer, /credential redacted/);
  assert.ok(progress.every((text) => !text.includes(specialKey)));
});

test("malformed agent output, empty answers and truncated provider responses never claim completion", async (parent) => {
  for (const turn of ["plain text", '{"questions":', ["Which enzyme?"], { reasoning: null }, { questions: "Which enzyme?" }, { reasoning: "Ignored null questions", questions: null }, { reasoning: "Ignored null calls", tool_calls: null }, { tool_calls: [{ interface: "kinetic_record", arguments: [] }] }, { answer: "unsupported" }, { reasoning: "" }]) {
    await parent.test(JSON.stringify(turn), async () => {
      const { runtime } = harness(() => reply(turn));
      await assert.rejects(runtime.chat({ message: "Can you inspect this?", model_config: config() }), (error) => {
        assert.equal(error.data.completion_verified, false);
        assert.equal(error.data.transcript.answer, "");
        return true;
      });
    });
  }
  for (const [provider, protocol, finish] of [["minimax", "openai_chat", "length"], ["anthropic", "anthropic_messages", "max_tokens"]]) {
    const { runtime } = harness(() => reply({ reasoning: "partial" }, { anthropic: protocol === "anthropic_messages", finish }));
    await assert.rejects(runtime.chat({ message: "Inspect evidence.", model_config: config({ provider, protocol }) }), /token limit/);
  }
});

test("per-turn clipping and total call limits stop the real loop without an intermediate answer", async () => {
  let requests = 0;
  const { runtime } = harness(() => { requests++; return reply({ reasoning: "Still reading evidence.", tool_calls: Array.from({ length: 8 }, () => ({ interface: "reference_summary", arguments: {} })) }); });
  await assert.rejects(runtime.chat({ message: "Inspect evidence.", model_config: config() }), (error) => {
    assert.match(error.message, /total limit of 20/);
    assert.equal(requests, 5);
    assert.equal(error.data.transcript.tool_calls_made, 20);
    assert.equal(error.data.transcript.turns.length, 5);
    assert.ok(error.data.transcript.turns.every((turn) => turn.results.length === 4 && turn.clipped));
    assert.equal(error.data.transcript.answer, "");
    return true;
  });
});

test("turn limit stops continued tool use and oversized values are bounded before the next API request", async () => {
  let requests = 0;
  const { runtime } = harness((url, options) => {
    requests++;
    if (requests > 1) {
      const result = JSON.parse(JSON.parse(options.body).messages.at(-1).content.slice("Tool results:\n".length))[0];
      assert.equal(result.truncated, true);
      assert.ok(new TextEncoder().encode(JSON.stringify(result.value)).length <= 8000);
    }
    return reply({ tool_calls: [{ interface: "reference_summary", arguments: {} }] });
  }, { changeTools: (tools) => ({ ...tools, execute: async (call) => ({ tool: call.interface, arguments: call.arguments, rationale: "", ok: true, value: "文".repeat(10000), truncated: false, elapsed_ms: 0 }) }) });
  await assert.rejects(runtime.chat({ message: "Inspect evidence.", model_config: config() }), (error) => {
    assert.match(error.message, /turn limit of 6/);
    assert.equal(requests, 6);
    assert.equal(error.data.transcript.tool_calls_made, 6);
    return true;
  });
});

test("cancellation aborts the direct fetch and keeps preceding real tool results", async () => {
  const controller = new AbortController();
  let requests = 0;
  let observedSignal;
  const { runtime } = harness((url, options) => {
    if (++requests === 1) return reply({ tool_calls: [{ interface: "reference_summary", arguments: {} }] });
    observedSignal = options.signal;
    queueMicrotask(() => controller.abort());
    return new Promise((resolve, reject) => options.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true }));
  });
  await assert.rejects(runtime.chat({ message: "Inspect evidence.", model_config: config(), signal: controller.signal }), (error) => {
    assert.match(error.message, /Request cancelled/);
    assert.equal(observedSignal.aborted, true);
    assert.equal(error.data.transcript.tool_calls_made, 1);
    assert.equal(error.data.transcript.turns[0].results[0].ok, true);
    assert.equal(error.data.completion_verified, false);
    return true;
  });
  assert.equal(requests, 2);
});

test("cancellation stops waiting for local asynchronous work before another model request", async () => {
  const controller = new AbortController();
  let requests = 0;
  const { runtime } = harness(() => { requests++; return reply({ tool_calls: [{ interface: "calculate", arguments: { expression: "2 + 3" } }] }); }, {
    changeTools: (tools) => ({ ...tools, execute: () => { queueMicrotask(() => controller.abort()); return new Promise(() => {}); } }),
  });
  await assert.rejects(runtime.chat({ message: "Compute using the local tool.", model_config: config(), signal: controller.signal }), /Request cancelled/);
  assert.equal(requests, 1);
});

test("request and total wall-clock timeouts abort without retries", async (parent) => {
  for (const total of [false, true]) {
    await parent.test(total ? "total wall clock" : "provider request", async () => {
      let requests = 0;
      const { runtime, scheduled } = harness((url, options) => {
        requests++;
        return new Promise((resolve, reject) => options.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true }));
      }, { timers: (delay) => delay === 180000 ? total ? 10 : 200 : total ? 100 : 10 });
      await assert.rejects(runtime.chat({ message: "Inspect evidence.", model_config: config() }), total ? /180 s wall-clock/ : /API request timed out/);
      assert.equal(requests, 1);
      assert.ok(scheduled.includes(60000));
      assert.ok(scheduled.includes(180000));
    });
  }
});

test("CORS, redirects and oversized API responses produce actionable failures, never proxies or retries", async (parent) => {
  for (const [name, provider, expected] of [
    ["CORS", () => { throw new TypeError("Network fetch failed"); }, /CORS.*CORS-enabled API URL.*Backend mode/],
    ["redirect", () => new Response(null, { status: 302 }), /redirected/],
    ["size cap", () => new Response("unused", { headers: { "Content-Length": "2097153" } }), /size limit/],
    ["stream size cap", () => new Response("x".repeat(2097153)), /size limit/],
  ]) {
    await parent.test(name, async () => {
      const { runtime, calls } = harness(provider);
      await assert.rejects(runtime.chat({ message: "Inspect evidence.", model_config: config() }), expected);
      assert.equal(calls.filter((entry) => entry.url !== "./reference-data.json").length, 1);
      assert.ok(calls.every((entry) => entry.options.mode !== "no-cors"));
    });
  }
});

test("provider context cap prevents oversized accumulated requests before fetch", async () => {
  let requests = 0;
  const { runtime } = harness(() => {
    requests++;
    return reply({ tool_calls: Array.from({ length: 4 }, () => ({ interface: "calculate", arguments: { expression: "x".repeat(7900) } })) });
  }, { changeTools: (tools) => ({ ...tools, execute: async (call) => ({ tool: call.interface, arguments: call.arguments, rationale: "", ok: true, value: "x".repeat(7900), truncated: false, elapsed_ms: 0 }) }) });
  await assert.rejects(runtime.chat({ message: "x".repeat(8000), history: [{ role: "user", content: "x".repeat(8000) }], model_config: config() }), (error) => {
    assert.match(error.message, /provider request size limit/);
    assert.equal(error.data.completion_verified, false);
    assert.equal(error.data.transcript.tool_calls_made, 12);
    assert.equal(requests, 3);
    return true;
  });
});

test("parallel browser chats keep model credentials and calculation state isolated", async () => {
  const pairs = [];
  const { runtime } = harness((url, options) => {
    const model = JSON.parse(options.body).model;
    const credential = options.headers.Authorization.slice("Bearer ".length);
    pairs.push([model, credential]);
    return reply({ reasoning: "Credential echo: " + credential });
  });
  const selected = [config({ model: "model-alpha", api_key: "fake-secret-alpha" }), config({ model: "model-beta", api_key: "fake-secret-beta" })];
  const results = await Promise.all(selected.map((model_config) => runtime.chat({ message: "Inspect evidence.", model_config })));
  assert.deepEqual(pairs.sort(), selected.map((value) => [value.model, value.api_key]).sort());
  results.forEach((result, index) => {
    assert.equal(result.model_config.model, selected[index].model);
    assert.ok(!JSON.stringify(result).includes(selected[index].api_key));
    assert.match(result.answer, /credential redacted/);
  });
});

test("invalid model settings, secret-bearing URLs and oversized history are refused before any network request", async (parent) => {
  for (const changes of [{ api_url: "http://provider.example/v1/messages" }, { api_url: "https://user:password@provider.example/v1/messages" }, { api_url: "https://provider.example/" }, { api_url: "https://provider.example/v1/messages?key=secret" }, { api_url: "https://provider.example/v1/messages#fragment" }, { api_url: "https://provider.example/" + key }, { protocol: "unsupported" }, { provider: "anthropic", protocol: "openai_chat" }, { api_key: "key\nHeader: injection" }, { model: "" }]) {
    await parent.test(JSON.stringify(changes), async () => {
      const { runtime, calls } = harness(() => reply({ reasoning: "unused" }));
      await assert.rejects(runtime.chat({ message: "Inspect evidence.", model_config: config(changes) }));
      assert.equal(calls.length, 0);
    });
  }
  const { runtime, calls } = harness(() => reply({ reasoning: "unused" }));
  await assert.rejects(runtime.chat({ message: "Inspect evidence.", history: Array.from({ length: 4 }, () => ({ role: "user", content: "x".repeat(8000) })), model_config: config() }), /input limits/);
  assert.equal(calls.length, 0);
  const percentKey = 'fake-quote"slash\\browser-key';
  const encoded = encodeURIComponent(percentKey).replace(/%[A-F0-9]{2}/g, (match) => match.toLowerCase());
  await assert.rejects(runtime.chat({ message: "Inspect evidence.", model_config: config({ api_key: percentKey, api_url: "https://provider.example/" + encoded }) }), /never in the API URL/);
  assert.equal(calls.length, 0);
});
