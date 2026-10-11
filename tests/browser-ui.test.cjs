"use strict";

const assert = require("node:assert/strict");
const { webcrypto } = require("node:crypto");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

// Exercise the shipped UI and its transport decisions without real API keys,
// browser extensions, or network access. The browser runtime is tested separately.
class Element {
  constructor(tag = "div") {
    this.tagName = tag; this.children = []; this.listeners = {}; this.value = "";
    this.hidden = false; this.style = {}; this.dataset = {}; this.textContent = "";
    this.scrollHeight = 50; this.files = [];
  }
  addEventListener(event, fn) { this.listeners[event] = fn; }
  setAttribute(name, value) { this[name] = value; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  click() { return this.listeners.click?.({ preventDefault() {} }); }
  focus() {}
  get childElementCount() { return this.children.length; }
  get firstElementChild() { return this.children[0]; }
  get lastElementChild() { return this.children.at(-1); }
  get selectedIndex() { return this.options ? this.options.findIndex((option) => option.value === this.value) : 0; }
}

const flush = () => new Promise((resolve) => setImmediate(resolve));
const event = { preventDefault() {} };

async function harness(options = {}) {
  const elements = new Map();
  const ui = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  ui("model-provider").options = ["minimax", "minimax_cn", "openai", "anthropic", "custom", "server_default"].map((value) => ({ value, text: value }));
  ui("model-provider").value = "minimax";
  const storage = new Map();
  const exports = [];
  const browserCalls = [];
  const requests = [];
  const testCalls = [];
  const localCalls = [];
  const backend = "https://test-backend.example";
  const runtime = {
    local: async (args) => {
      localCalls.push(args);
      if (options.local) return options.local(args);
      return { answer: "Ran 1 stored-reference tool in this browser. No model or API was used and nothing was interpreted.", notes: [], completion_verified: false, execution: "browser-local", transcript: { turns: [{ turn: 0, results: [{ tool: "calculate", ok: true, rationale: "The message contains an expression.", value: { value: 10 } }] }], tool_calls_made: 1 } };
    },
    testConnection: async (args) => {
      testCalls.push(args);
      if (options.testConnection) return options.testConnection(args);
      return { verdict: "reachable", endpoint: args.api_url, summary: "The API answered this browser with HTTP 401. Network and CORS are fine.", advice: "Your real key is checked on the first question." };
    },
    prepare: options.prepare || (async () => ({ runtime_ready: true, tools: ["calculate", "inspect_structure"], execution: "browser" })),
    chat: async (args) => {
      browserCalls.push(args);
      if (options.chat) return options.chat(args);
      args.onProgress("Running calculate");
      return { answer: "2 * 3 + 4 = 10", completion_verified: true, model_config: { provider: "minimax", model: "MiniMax-M2.7" }, transcript: { turns: [{ turn: 1, results: [{ tool: "calculate", ok: true, value: 10 }] }], tool_calls_made: 1 } };
    },
  };
  class TestURL extends URL {
    static createObjectURL(blob) { exports.push(blob); return "blob:test-export"; }
    static revokeObjectURL() {}
  }
  const fetch = async (url, request = {}) => {
    requests.push({ url, ...request });
    if (url === "./config.json") return { ok: true, json: async () => ({ backend_url: backend }) };
    if (options.realRuntime && url === "./reference-data.json") return new Response(fs.readFileSync(path.join(__dirname, "../web/reference-data.json"), "utf8"), { status: 200 });
    if (options.realRuntime && url === "https://api.minimax.io/v1/chat/completions") {
      if (options.providerError) throw new TypeError("Mock browser network/CORS failure");
      const body = JSON.parse(request.body);
      const latest = body.messages.at(-1).content;
      let turn;
      if (latest.startsWith("Tool results:\n")) {
        const result = JSON.parse(latest.slice("Tool results:\n".length))[0];
        assert.equal(result.tool, "calculate");
        assert.equal(result.ok, true);
        assert.equal(result.value.computed, true);
        assert.equal(result.value.value, 10);
        turn = { reasoning: "The local calculator returned ten.", tool_calls: [] };
      } else turn = { reasoning: "I will use the local calculator.", tool_calls: [{ interface: "calculate", arguments: { expression: "2 * 3 + 4" }, rationale: "Compute locally." }] };
      return new Response(JSON.stringify({ choices: [{ message: { content: JSON.stringify(turn) }, finish_reason: "stop" }] }), { status: 200 });
    }
    if (url === backend + "/api/health") return { ok: true, json: async () => ({ service: "eagent", runtime_ready: true, ready: true, authentication_required: true, provider: "minimax", model: "server-model", tools: ["python_tool"] }) };
    if (url === backend + "/api/chat") return { ok: true, json: async () => ({ answer: "Python backend response", completion_verified: true, transcript: { turns: [], tool_calls_made: 0 } }) };
    throw new Error("Unexpected network request: " + url);
  };
  const context = vm.createContext({
    document: { getElementById: ui, createElement: (tag) => new Element(tag), createTextNode: (text) => ({ textContent: text }), querySelectorAll: () => [] },
    window: { EAgentBrowserRuntime: runtime, matchMedia: () => ({ matches: false }), location: { hostname: "example.org", protocol: "https:", origin: "https://example.org" } },
    localStorage: { getItem: (key) => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) },
    URL: TestURL, Blob, TextEncoder, TextDecoder, AbortController, crypto: webcrypto, fetch,
    setTimeout: () => 1, clearTimeout() {}, requestAnimationFrame: (fn) => fn(),
  });
  // In a real page, window and globalThis address the same browser global.
  Object.assign(context, context.window);
  context.window = context;
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../web/endpoint.js"), "utf8"), context);
  if (options.realRuntime) {
    for (const filename of ["reference-tools.js", "browser-agent.js"]) vm.runInContext(fs.readFileSync(path.join(__dirname, "../web", filename), "utf8"), context);
  }
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../web/app.js"), "utf8"), context);
  await flush();
  const input = (id, value) => { ui(id).value = value; return ui(id).listeners.input?.(event); };
  const chooseMode = async (value) => { ui("execution-mode").value = value; await ui("execution-mode").listeners.change(event); await flush(); };
  const send = async (message) => { input("message", message); await ui("chat-form").listeners.submit(event); };
  return { ui, storage, exports, browserCalls, testCalls, localCalls, requests, backend, input, chooseMode, send };
}

test("default Browser + API prepares local tools and chats without a backend connection", async () => {
  const h = await harness();
  assert.equal(h.ui("execution-mode").value, "browser");
  assert.equal(h.ui("connection-settings").hidden, true);
  assert.equal(h.ui("status-label").textContent, "Browser tools loaded");
  assert.deepEqual(h.requests.map((request) => request.url), ["./config.json"]);
  assert.equal(h.ui("tools-list").children[0].textContent, "calculate");
  h.input("provider-key", "test-provider-key");
  assert.equal(h.ui("status-label").textContent, "Ready to verify");
  await h.send("Calculate 2 * 3 + 4");
  assert.equal(h.browserCalls.length, 1);
  assert.equal(h.browserCalls[0].model_config.api_key, "test-provider-key");
  assert.equal(h.browserCalls[0].model_config.api_url, "https://api.minimax.io/v1/chat/completions");
  assert.equal(h.ui("status-label").textContent, "Chat ready", h.ui("status-detail").textContent);
  assert.equal(h.ui("progress-label").textContent, "Running calculate");
  assert.equal(h.ui("messages").childElementCount, 2);
  assert.equal(h.requests.length, 1);
});

test("the MiniMax China preset fills M3 and its base URL, shows the endpoint it will call, and sends that", async () => {
  const h = await harness();
  assert.equal(h.ui("api-url-resolved").hidden, true, "a full endpoint needs no notice");
  h.ui("model-provider").value = "minimax_cn";
  h.ui("model-provider").listeners.change(event);
  assert.equal(h.ui("model-name").value, "MiniMax-M3");
  assert.equal(h.ui("api-url").value, "https://api.minimax.cn/v1");
  assert.equal(h.ui("api-url-resolved").hidden, false);
  assert.equal(h.ui("api-url-resolved").textContent, "Requests will be sent to https://api.minimax.cn/v1/chat/completions");
  h.input("provider-key", "test-provider-key");
  await h.send("Calculate 2 * 3 + 4");
  assert.equal(h.browserCalls[0].model_config.provider, "minimax_cn");
  assert.equal(h.browserCalls[0].model_config.api_url, "https://api.minimax.cn/v1/chat/completions");
  h.input("api-url", "https://api.minimax.cn/v1/chat/completions");
  assert.equal(h.ui("api-url-resolved").hidden, true);
});

test("templates keep the completed China endpoint and import back into the same preset", async () => {
  const h = await harness();
  h.ui("model-provider").value = "minimax_cn";
  h.ui("model-provider").listeners.change(event);
  h.ui("template-name").value = "China M3";
  h.ui("save-template").click();
  const saved = JSON.parse(h.storage.get("eagent.model_templates.v1"))[0];
  assert.equal(saved.provider, "minimax_cn");
  assert.equal(saved.model, "MiniMax-M3");
  assert.equal(saved.api_url, "https://api.minimax.cn/v1/chat/completions");
  h.ui("model-provider").value = "openai";
  h.ui("model-provider").listeners.change(event);
  h.ui("template-file").files = [{ size: 500, text: async () => JSON.stringify(saved) }];
  await h.ui("template-file").listeners.change(event);
  assert.equal(h.ui("model-provider").value, "minimax_cn");
  assert.equal(h.ui("api-url").value, "https://api.minimax.cn/v1/chat/completions");
});

const visibleText = (node) => [node.textContent, ...(node.children || []).map(visibleText)].join("\n");

test("Test connection sends only the URL and format, shows the verdict, and never touches the typed key", async () => {
  const h = await harness();
  h.input("provider-key", "typed-secret-key");
  await h.ui("test-connection").click();
  assert.equal(h.testCalls.length, 1);
  assert.deepEqual(Object.keys(h.testCalls[0]).sort(), ["api_url", "protocol", "signal"]);
  assert.equal(h.testCalls[0].api_url, "https://api.minimax.io/v1/chat/completions");
  assert.equal(h.testCalls[0].protocol, "openai_chat");
  assert.equal(JSON.stringify(h.testCalls[0]).includes("typed-secret-key"), false);
  assert.equal(h.ui("connection-test").hidden, false);
  assert.match(visibleText(h.ui("connection-test")), /Network and CORS are fine/);
  assert.match(h.ui("connection-test").className, /\bok\b/);
  assert.equal(h.ui("test-connection").textContent, "Test connection");
  assert.equal(h.ui("test-connection").disabled, false);
  assert.equal(h.requests.some((request) => request.url.includes("minimax")), false, "the page itself makes no provider request; the runtime does");
});

test("Test connection completes a pasted base URL before probing it", async () => {
  const h = await harness();
  h.ui("model-provider").value = "minimax_cn";
  h.ui("model-provider").listeners.change(event);
  await h.ui("test-connection").click();
  assert.equal(h.testCalls[0].api_url, "https://api.minimax.cn/v1/chat/completions");
});

test("Test connection shows each kind of failure distinctly and clears when the URL changes", async () => {
  for (const [verdict, kind] of [["blocked", "bad"], ["unreachable", "bad"], ["timeout", "bad"], ["wrong_path", "warn"], ["provider_error", "warn"]]) {
    const h = await harness({ testConnection: async () => ({ verdict, summary: "Summary for " + verdict, advice: "Advice for " + verdict }) });
    await h.ui("test-connection").click();
    assert.match(h.ui("connection-test").className, new RegExp("\\b" + kind + "\\b"), verdict);
    assert.match(visibleText(h.ui("connection-test")), new RegExp("Summary for " + verdict));
    assert.match(visibleText(h.ui("connection-test")), new RegExp("Advice for " + verdict));
    h.input("api-url", "https://api.minimax.io/v1/other");
    assert.equal(h.ui("connection-test").hidden, true, verdict + " result should be stale after the URL changed");
  }
});

test("typing a key does not discard a connection verdict, because the verdict does not involve the key", async () => {
  const h = await harness();
  await h.ui("test-connection").click();
  h.input("provider-key", "typed-after-the-test");
  assert.equal(h.ui("connection-test").hidden, false);
});

test("Test connection reports a runtime failure, and incomplete settings, without calling the runtime", async () => {
  const failing = await harness({ testConnection: async () => { throw new Error("Use a full HTTPS API endpoint."); } });
  await failing.ui("test-connection").click();
  assert.match(visibleText(failing.ui("connection-test")), /could not run/);
  assert.match(visibleText(failing.ui("connection-test")), /full HTTPS API endpoint/);
  assert.match(failing.ui("connection-test").className, /\bbad\b/);
  assert.equal(failing.ui("test-connection").disabled, false, "the control must recover after a failure");

  const incomplete = await harness();
  incomplete.input("model-name", "");
  await incomplete.ui("test-connection").click();
  assert.equal(incomplete.testCalls.length, 0);
  assert.match(visibleText(incomplete.ui("connection-test")), /Complete the settings first/);
});

test("Test connection is offered only in Browser + API mode and a mode switch discards its verdict", async () => {
  const h = await harness();
  assert.equal(h.ui("connection-test-box").hidden, false);
  await h.ui("test-connection").click();
  assert.equal(h.ui("connection-test").hidden, false);
  await h.chooseMode("backend");
  assert.equal(h.ui("connection-test-box").hidden, true);
  assert.equal(h.ui("connection-test").hidden, true);
  await h.chooseMode("browser");
  assert.equal(h.ui("connection-test-box").hidden, false);
  assert.equal(h.ui("connection-test").hidden, true);
});

const descendants = (node) => [node, ...(node.children || []).flatMap(descendants)];
const withClass = (root, name) => descendants(root).filter((node) => node.className === name || String(node.className).split(" ").includes(name));

test("Run in browser works with no API key, calls no model, and is neither verification nor model history", async () => {
  const h = await harness();
  assert.equal(h.ui("run-local").disabled, true, "nothing to run yet");
  h.input("message", "Calculate 2 * 3 + 4");
  assert.equal(h.ui("send-button").disabled, true, "sending to a model still needs a key");
  assert.equal(h.ui("run-local").disabled, false, "running locally does not");
  assert.match(h.ui("composer-notice").textContent, /Run in browser needs no API key/);
  await h.ui("run-local").click();
  assert.equal(h.localCalls.length, 1);
  assert.deepEqual(Object.keys(h.localCalls[0]), ["message"], "no key, model settings or URL are passed");
  assert.equal(h.localCalls[0].message, "Calculate 2 * 3 + 4");
  assert.equal(h.browserCalls.length, 0, "no model request");
  assert.equal(h.ui("messages").childElementCount, 2);
  assert.equal(h.ui("message").value, "");
  const shown = visibleText(h.ui("messages"));
  assert.match(shown, /BROWSER TOOLS · NO MODEL/);
  assert.match(shown, /View research trace · 1 tool call/);
  assert.match(shown, /nothing was interpreted/);
  assert.equal(withClass(h.ui("messages"), "trace").every((node) => node.open === true), true, "the evidence is the answer, so it opens");
  assert.notEqual(h.ui("status-label").textContent, "Chat ready");
  assert.equal(h.ui("run-local").disabled, true, "re-enabled only by a new question");
  h.input("provider-key", "test-provider-key");
  await h.send("Now ask the model");
  assert.equal(h.browserCalls[0].history.length, 0, "the local exchange is not part of the model's context");
});

test("after a failed model request one click runs the same question locally, without repeating it", async () => {
  const h = await harness({ chat: async () => { throw new Error("The browser could not reach this model API. CORS may block it."); } });
  h.input("provider-key", "test-provider-key");
  await h.send("List the audited structures");
  const actions = withClass(h.ui("messages"), "notice-action");
  assert.equal(actions.length, 1);
  assert.equal(h.ui("message").value, "List the audited structures", "the draft is kept so the model can be retried after fixing settings");
  await actions[0].click();
  await flush();
  assert.equal(actions[0].disabled, true);
  assert.equal(h.localCalls[0].message, "List the audited structures");
  assert.equal(withClass(h.ui("messages"), "user").length, 1, "the question is not shown twice");
  assert.match(visibleText(h.ui("messages")), /BROWSER TOOLS · NO MODEL/);
  assert.equal(h.ui("message").value, "List the audited structures", "a recovery run leaves the draft alone");
  assert.equal(h.ui("status-label").textContent, "Model request failed", "the model failure is still reported as a failure");
});

test("Run in browser is offered only in Browser + API mode, once the tools have loaded", async () => {
  const h = await harness();
  h.input("message", "inspect 6ZZO");
  assert.equal(h.ui("run-local").hidden, false);
  await h.chooseMode("backend");
  assert.equal(h.ui("run-local").hidden, true);
  await h.chooseMode("browser");
  assert.equal(h.ui("run-local").hidden, false);

  const unloaded = await harness({ prepare: async () => { throw new Error("The published reference bundle is unavailable (HTTP 404)."); } });
  unloaded.input("message", "inspect 6ZZO");
  assert.equal(unloaded.ui("run-local").disabled, true, "no tools, nothing to run");
  await unloaded.ui("run-local").click();
  assert.equal(unloaded.localCalls.length, 0);
});

test("a question the router cannot place shows its guidance and no empty trace", async () => {
  const h = await harness({ local: async () => ({ answer: "I could not match this to a stored tool. Try a structure id.", notes: ["x"], completion_verified: false, execution: "browser-local", transcript: { turns: [{ turn: 0, results: [] }], tool_calls_made: 0 } }) });
  h.input("message", "Why is the sky blue?");
  await h.ui("run-local").click();
  assert.match(visibleText(h.ui("messages")), /could not match this to a stored tool/);
  assert.equal(withClass(h.ui("messages"), "trace").length, 0);
});

test("a local run that fails is reported, and the control recovers", async () => {
  const h = await harness({ local: async () => { throw new Error("The browser reference tools cannot route questions. Reload the page."); } });
  h.input("message", "inspect 6ZZO");
  await h.ui("run-local").click();
  assert.match(visibleText(h.ui("messages")), /cannot route questions/);
  assert.equal(withClass(h.ui("messages"), "error").length, 1);
  h.input("message", "inspect 1IPF");
  assert.equal(h.ui("run-local").disabled, false);
});

test("a mode switch while a local run is pending drops its result", async () => {
  let finish;
  const h = await harness({ local: () => new Promise((resolve) => { finish = resolve; }) });
  h.input("message", "inspect 6ZZO");
  const pending = h.ui("run-local").click();
  await h.chooseMode("backend");
  finish({ answer: "Stale local answer", notes: [], completion_verified: false, execution: "browser-local", transcript: { turns: [], tool_calls_made: 0 } });
  await pending;
  assert.equal(h.ui("messages").childElementCount, 0);
});

test("the shipped UI and real runtime answer the 6ZZO question with no key and no model request", async () => {
  const h = await harness({ realRuntime: true });
  assert.equal(h.ui("status-label").textContent, "Browser tools loaded");
  h.input("message", "List the audited structures and inspect 6ZZO. What evidence and limitations are recorded for its cofactor and ligand?");
  await h.ui("run-local").click();
  await flush();
  const shown = visibleText(h.ui("messages"));
  assert.match(shown, /View research trace · 2 tool calls/);
  assert.match(shown, /list_structure_entries/);
  assert.match(shown, /structure_entry/);
  assert.match(shown, /"pdb_id": "6ZZO"/);
  assert.match(shown, /BROWSER TOOLS · NO MODEL/);
  assert.equal(h.requests.some((request) => /minimax|openai|anthropic/.test(request.url)), false, "no provider was contacted");
  assert.equal(h.ui("provider-key").value, "", "and no key was needed");
});

test("a reach or CORS failure points to Test connection once, and does not repeat itself as 'Run ended'", async () => {
  const sentence = "The browser could not reach this model API. CORS may block it.";
  const h = await harness({ chat: async () => {
    const error = new Error(sentence);
    error.data = { transcript: { turns: [{ turn: 0, provider_error: sentence }], tool_calls_made: 0, stopped_because: "The provider call failed: " + sentence } };
    throw error;
  } });
  h.input("provider-key", "test-provider-key");
  await h.send("Anything");
  // Only what is on screen: the message and its notices. The collapsed "Full
  // transcript" holds the same strings as JSON and is not displayed.
  const shown = [...withClass(h.ui("messages"), "message-body"), ...withClass(h.ui("messages"), "run-notice")].map((node) => node.textContent).join("\n");
  assert.equal(shown.split("could not reach this model API").length - 1, 2, "once as the notice and once as the provider error, not three times");
  assert.equal((shown.match(/Press Test connection/g) || []).length, 1);
  assert.equal(/Run ended/.test(shown), false);
});

test("a failure that is not about reaching the API gets no connection hint, and a different stop reason still shows", async () => {
  const h = await harness({ chat: async () => {
    const error = new Error("The provider returned malformed JSON in the agent protocol.");
    error.data = { transcript: { turns: [{ turn: 0, provider_error: "The provider returned malformed JSON in the agent protocol." }], tool_calls_made: 0, stopped_because: "The turn limit of 6 was reached." } };
    throw error;
  } });
  h.input("provider-key", "test-provider-key");
  await h.send("Anything");
  const shown = [...withClass(h.ui("messages"), "message-body"), ...withClass(h.ui("messages"), "run-notice")].map((node) => node.textContent).join("\n");
  assert.equal(/Press Test connection/.test(shown), false);
  assert.match(shown, /Run ended: The turn limit of 6 was reached\./);
});

test("provider failure retains browser tools and permits retry without a backend token", async () => {
  const h = await harness({ chat: async () => { const error = new Error("Model API rejected the API key."); error.status = 401; throw error; } });
  h.input("provider-key", "invalid-test-key");
  await h.send("Test failure");
  assert.equal(h.ui("status-label").textContent, "Model request failed");
  assert.match(h.ui("status-detail").textContent, /Browser tools remain loaded/);
  assert.equal(h.ui("send-button").disabled, false);
  assert.equal(h.ui("prepare-browser").hidden, true);
  assert.equal(h.ui("connection-settings").hidden, true);
  assert.equal(h.ui("message").value, "Test failure");
});

test("mode switch aborts a pending browser run and clears history and credentials", async () => {
  let resolveRun;
  const h = await harness({ chat: () => new Promise((resolve) => { resolveRun = resolve; }) });
  h.input("provider-key", "test-provider-key");
  h.ui("access-token").value = "test-backend-token";
  const pending = h.send("Pending calculation");
  assert.equal(h.ui("cancel-request").hidden, false);
  await h.chooseMode("backend");
  assert.equal(h.browserCalls[0].signal.aborted, true);
  assert.equal(h.ui("provider-key").value, "");
  assert.equal(h.ui("access-token").value, "");
  assert.equal(h.ui("messages").childElementCount, 0);
  assert.equal(h.ui("connection-settings").hidden, false);
  resolveRun({ answer: "Stale answer", completion_verified: true });
  await pending;
  assert.equal(h.ui("messages").childElementCount, 0);
  assert.equal(h.ui("status-label").textContent, "Not connected");
});

test("cancelling a browser request retains the runtime’s completed tool trace", async () => {
  const h = await harness({ chat: ({ signal }) => new Promise((resolve, reject) => {
    signal.addEventListener("abort", () => {
      const error = new Error("Request cancelled after local calculation.");
      error.status = 408;
      error.data = { completion_verified: false, transcript: { turns: [{ turn: 0, results: [{ tool: "calculate", ok: true, value: { computed: true, value: 10 } }] }], tool_calls_made: 1, stopped_because: "Request cancelled." } };
      reject(error);
    }, { once: true });
  }) });
  h.input("provider-key", "test-provider-key");
  const pending = h.send("Keep the calculation trace");
  h.ui("cancel-request").click();
  await pending;
  assert.equal(h.ui("status-label").textContent, "Request cancelled");
  assert.equal(h.ui("message").value, "Keep the calculation trace");
  const text = (node) => [node.textContent, ...(node.children || []).map(text)].join("\n");
  const visibleConversation = text(h.ui("messages"));
  assert.match(visibleConversation, /View research trace · 1 tool call/);
  assert.match(visibleConversation, /calculate/);
  assert.match(visibleConversation, /"computed": true/);
  assert.match(visibleConversation, /"value": 10/);
});

test("optional Python backend and server default still use authenticated backend transport", async () => {
  const h = await harness();
  await h.chooseMode("backend");
  h.ui("model-provider").value = "server_default";
  h.ui("model-provider").listeners.change(event);
  h.input("access-token", "test-backend-token");
  await h.ui("connection-settings").listeners.submit(event);
  await h.send("Ask the Python agent");
  const request = h.requests.find((entry) => entry.url.endsWith("/api/chat"));
  assert.equal(request.headers.Authorization, "Bearer test-backend-token");
  assert.equal(Object.hasOwn(JSON.parse(request.body), "model_config"), false);
  assert.equal(h.browserCalls.length, 0);
  await h.chooseMode("browser");
  assert.equal(h.ui("model-provider").value, "minimax");
  assert.equal(h.ui("manual-model-fields").hidden, false);
  assert.equal(h.ui("access-token").value, "");
});

test("version 2 templates preserve run mode, exclude credentials and import old templates", async () => {
  const h = await harness();
  h.input("provider-key", "test-provider-secret");
  h.ui("access-token").value = "test-backend-secret";
  h.ui("template-name").value = "Browser model";
  h.ui("save-template").click();
  const saved = JSON.parse(h.storage.get("eagent.model_templates.v1"))[0];
  assert.equal(saved.version, 2);
  assert.equal(saved.execution_mode, "browser");
  assert.equal(JSON.stringify(saved).includes("secret"), false);
  h.ui("export-template").click();
  assert.deepEqual(JSON.parse(await h.exports[0].text()), saved);
  const old = { ...saved, version: 1 };
  delete old.execution_mode;
  h.ui("template-file").files = [{ size: 500, text: async () => JSON.stringify(old) }];
  await h.ui("template-file").listeners.change(event);
  assert.equal(h.ui("execution-mode").value, "backend");
  assert.equal(h.ui("provider-key").value, "");
  assert.equal(h.ui("access-token").value, "");
  const secret = { ...saved, api_key: "do-not-import" };
  h.ui("template-file").files = [{ size: 500, text: async () => JSON.stringify(secret) }];
  await h.ui("template-file").listeners.change(event);
  assert.match(h.ui("model-feedback").textContent, /contains a key or token/);
  const invalidBrowserDefault = { ...saved, provider: "server_default", model: "", api_url: "" };
  h.ui("template-file").files = [{ size: 500, text: async () => JSON.stringify(invalidBrowserDefault) }];
  await h.ui("template-file").listeners.change(event);
  assert.match(h.ui("model-feedback").textContent, /only in Python backend mode/);
  assert.equal(h.ui("execution-mode").value, "backend");
});

test("save and export refuse a currently typed credential pasted into public template fields", async () => {
  const h = await harness();
  const key = 'test"provider\\key';
  h.input("provider-key", key);
  h.ui("template-name").value = "Profile " + key;
  h.ui("save-template").click();
  assert.match(h.ui("model-feedback").textContent, /Remove your API key or backend token/);
  assert.equal(h.storage.has("eagent.model_templates.v1"), false);
  h.ui("template-name").value = "Safe profile name";
  h.input("api-url", "https://provider.example/" + encodeURIComponent(key) + "/chat/completions");
  h.ui("export-template").click();
  assert.match(h.ui("model-feedback").textContent, /Remove your API key or backend token/);
  assert.equal(h.exports.length, 0);
  h.input("api-url", "https://provider.example/v1/chat/completions");
  h.ui("access-token").value = "test-backend-token";
  h.input("model-name", "model-test-backend-token");
  h.ui("save-template").click();
  assert.match(h.ui("model-feedback").textContent, /Remove your API key or backend token/);
  assert.equal(h.storage.has("eagent.model_templates.v1"), false);
});

test("shipped UI and real browser runtime calculate using the real reference tools; only provider transport is mocked", async () => {
  const h = await harness({ realRuntime: true });
  assert.equal(h.ui("status-label").textContent, "Browser tools loaded");
  assert.equal(h.ui("connection-settings").hidden, true);
  h.input("provider-key", "deliberately-invalid-test-key");
  await h.send("Calculate 2 * 3 + 4 using the local calculator");
  const modelRequests = h.requests.filter((request) => request.url === "https://api.minimax.io/v1/chat/completions");
  assert.equal(modelRequests.length, 2);
  assert.equal(modelRequests[0].headers.Authorization, "Bearer deliberately-invalid-test-key");
  assert.equal(h.ui("status-label").textContent, "Chat ready");
  assert.equal(h.ui("messages").childElementCount, 2);
  assert.equal(h.requests.some((request) => request.url.includes("/api/health") || request.url.includes("/api/chat")), false);
  const text = (node) => [node.textContent, ...(node.children || []).map(text)].join("\n");
  assert.match(text(h.ui("messages")), /View research trace · 1 tool call/);
  assert.match(text(h.ui("messages")), /"computed": true/);
  assert.match(text(h.ui("messages")), /"value": 10/);
});

test("real browser runtime reports CORS/network failure without a false ready status", async () => {
  const h = await harness({ realRuntime: true, providerError: true });
  h.input("provider-key", "deliberately-invalid-test-key");
  await h.send("Calculate a value");
  assert.equal(h.ui("status-label").textContent, "Model request failed");
  assert.match(h.ui("status-detail").textContent, /CORS/);
  assert.equal(h.ui("send-button").disabled, false);
  assert.equal(h.ui("prepare-browser").hidden, true);
});
