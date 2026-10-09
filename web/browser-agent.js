"use strict";

// Runs the reference agent locally; credentials go only to the selected API.
(() => {
  const root = typeof window === "object" ? window : globalThis;
  const LIMITS = Object.freeze({ max_turns: 6, max_calls_per_turn: 4, max_calls_total: 20, max_result_bytes: 8000, max_wall_seconds: 180 });
  const MAX_RESPONSE_BYTES = 2097152;
  const MAX_REFERENCE_BYTES = 3145728;
  const MAX_REQUEST_BYTES = 262144;
  const encoder = new TextEncoder();
  const normalStop = "the model answered without asking for another tool";
  const protocols = { minimax: "openai_chat", openai: "openai_chat", anthropic: "anthropic_messages", custom: null };
  let referenceBundle = null;

  const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
  const bytes = (text) => encoder.encode(text).length;
  const cancelled = () => new Error("Request cancelled. The provider may still finish work already started.");
  const checkSignal = (signal) => { if (signal?.aborted) throw signal.reason instanceof Error && signal.reason.name !== "AbortError" ? signal.reason : cancelled(); };

  async function withSignal(promise, signal) {
    checkSignal(signal);
    return new Promise((resolve, reject) => {
      const abort = () => { try { checkSignal(signal); } catch (error) { reject(error); } };
      signal?.addEventListener("abort", abort, { once: true });
      Promise.resolve(promise).then(resolve, reject).finally(() => signal?.removeEventListener("abort", abort));
    });
  }

  function redactString(value, key) {
    const variants = new Set();
    if (typeof key === "string" && key) {
      variants.add(key);
      variants.add(encodeURIComponent(key));
      variants.add(encodeURIComponent(key).replace(/%20/g, "+"));
      let escaped = key;
      for (let index = 0; index < 4; index++) { escaped = JSON.stringify(escaped).slice(1, -1); variants.add(escaped); }
    }
    for (const variant of [...variants].sort((a, b) => b.length - a.length)) value = value.split(variant).join("[credential redacted]");
    return value;
  }

  function redact(value, key) {
    if (typeof value === "string") return redactString(value, key);
    if (Array.isArray(value)) return value.map((item) => redact(item, key));
    if (object(value)) return Object.fromEntries(Object.entries(value).map(([name, item]) => [redactString(name, key), redact(item, key)]));
    return value;
  }

  function modelConfiguration(value) {
    const fields = ["provider", "protocol", "api_key", "model", "api_url"];
    if (!object(value) || Object.keys(value).length !== fields.length || fields.some((name) => !Object.hasOwn(value, name) || typeof value[name] !== "string" || !value[name].trim())) throw new Error("Model settings require provider, API format, API key, model name and full API URL.");
    const config = Object.fromEntries(fields.map((name) => [name, value[name].trim()]));
    if (!Object.hasOwn(protocols, config.provider) || !["openai_chat", "anthropic_messages"].includes(config.protocol) || (protocols[config.provider] && protocols[config.provider] !== config.protocol)) throw new Error("The model provider and API format do not match.");
    if (config.api_key.length > 8192 || /[^\x20-\x7e]/.test(config.api_key)) throw new Error("Enter a bounded printable API key without line breaks.");
    if (config.model.length > 256 || /[\x00-\x1f\x7f]/.test(config.model)) throw new Error("Enter a model name of up to 256 characters without control characters.");
    if (config.api_url.length > 2048 || /[\s\\\x00-\x1f]/.test(config.api_url)) throw new Error("Enter a complete API URL without whitespace or backslashes.");
    let url;
    try { url = new URL(config.api_url); } catch (_) { throw new Error("Enter a complete model API URL with its request path."); }
    const local = ["localhost", "[::1]"].includes(url.hostname) || /^127(?:\.\d{1,3}){3}$/.test(url.hostname);
    if (url.username || url.password || url.search || url.hash || url.pathname === "/" || (url.protocol !== "https:" && !(url.protocol === "http:" && local))) throw new Error("Use a full HTTPS API endpoint without URL credentials, query parameters or fragments. HTTP is allowed only on loopback.");
    let decoded;
    try { decoded = decodeURIComponent(config.api_url); } catch (_) { throw new Error("The API URL contains invalid percent encoding."); }
    if (redactString(config.api_url, config.api_key) !== config.api_url || decoded.includes(config.api_key)) throw new Error("Put the API key in its password field, never in the API URL.");
    config.api_url = url.href;
    return config;
  }

  function chatInput(message, history, config) {
    if (typeof message !== "string" || !message.trim() || message.length > 8000) throw new Error("Enter a nonempty question of up to 8000 characters.");
    if (!Array.isArray(history) || history.length > 20) throw new Error("Conversation context must contain at most 20 messages.");
    let total = 0;
    for (const entry of history) {
      if (!object(entry) || Object.keys(entry).length !== 2 || !["user", "assistant"].includes(entry.role) || typeof entry.content !== "string" || entry.content.length > 8000) throw new Error("Conversation context needs bounded user or assistant text messages.");
      total += entry.content.length;
    }
    if (total > 24000 || bytes(JSON.stringify({ message, history, model_config: config })) > 65536) throw new Error("The question and recent conversation exceed the agent’s input limits.");
    return message.trim();
  }

  function publicMetadata(config) {
    return { provider: config.provider, protocol: config.protocol, model: config.model, api_url: config.api_url };
  }

  async function boundedText(response, limit, signal) {
    const declared = Number(response.headers?.get("content-length"));
    if (Number.isFinite(declared) && declared > limit) throw new Error("The response exceeded the agent’s size limit.");
    if (!response.body?.getReader) {
      const text = await response.text();
      checkSignal(signal);
      if (bytes(text) > limit) throw new Error("The response exceeded the agent’s size limit.");
      return text;
    }
    const reader = response.body.getReader();
    const chunks = [];
    let size = 0;
    try {
      while (true) {
        checkSignal(signal);
        const { value, done } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > limit) throw new Error("The response exceeded the agent’s size limit.");
        chunks.push(value);
      }
      const joined = new Uint8Array(size);
      let offset = 0;
      for (const chunk of chunks) { joined.set(chunk, offset); offset += chunk.byteLength; }
      return new TextDecoder("utf-8", { fatal: true }).decode(joined);
    } catch (error) {
      try { await reader.cancel(); } catch (_) { /* An aborted stream may already be closed. */ }
      throw error;
    } finally { reader.releaseLock(); }
  }

  async function readRequest(url, options, { signal, timeout = 60000, limit = MAX_RESPONSE_BYTES, reference = false } = {}) {
    checkSignal(signal);
    const controller = new AbortController();
    let timedOut = false;
    let rejectAbort;
    const abortPromise = new Promise((_, reject) => { rejectAbort = reject; });
    const abort = () => { controller.abort(); rejectAbort(timedOut ? new Error("The model API request timed out; no completion was verified.") : signal?.reason instanceof Error && signal.reason.name !== "AbortError" ? signal.reason : cancelled()); };
    const timer = setTimeout(() => { timedOut = true; abort(); }, timeout);
    signal?.addEventListener("abort", abort, { once: true });
    try {
      return await Promise.race([abortPromise, (async () => {
        let response;
        try { response = await fetch(url, { ...options, signal: controller.signal, credentials: "omit", cache: "no-store", redirect: "error", referrerPolicy: "no-referrer", mode: reference ? "same-origin" : "cors" }); }
        catch (error) {
          if (controller.signal.aborted) { checkSignal(signal); throw timedOut ? new Error("The model API request timed out; no completion was verified.") : cancelled(); }
          throw new Error(reference ? "Could not load the local reference bundle. Reload the page and check its published files." : "The browser could not reach this model API. Its CORS policy, a blocked redirect, network access, or browser restrictions may prevent direct requests. Use a CORS-enabled API URL or choose Backend mode.");
        }
        if (response.redirected || response.status >= 300 && response.status < 400) throw new Error("The API redirected. Enter the final API endpoint URL; credentials were not forwarded.");
        return { response, text: await boundedText(response, limit, controller.signal) };
      })()]);
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    }
  }

  function referenceMetadata(bundle) {
    if (!root.EAgentBrowserTools || typeof root.EAgentBrowserTools.create !== "function") throw new Error("The browser reference tools did not load. Reload the page.");
    const tools = root.EAgentBrowserTools.create(bundle);
    if (!tools || typeof tools.schemas !== "function" || typeof tools.toolNames !== "function" || typeof tools.execute !== "function" || typeof tools.inspect !== "function" || typeof tools.systemPrompt !== "string") throw new Error("The browser reference tool runtime is incompatible with this page.");
    return { tools, metadata: { runtime_ready: true, tools: tools.toolNames(), execution: "browser" } };
  }

  async function prepare({ signal } = {}) {
    checkSignal(signal);
    if (referenceBundle) return referenceMetadata(referenceBundle).metadata;
    const { response, text } = await readRequest("./reference-data.json", { method: "GET", headers: { Accept: "application/json" } }, { signal, limit: MAX_REFERENCE_BYTES, reference: true });
    if (!response.ok) throw new Error("The published reference bundle is unavailable (HTTP " + response.status + ").");
    let bundle;
    try { bundle = JSON.parse(text); } catch (_) { throw new Error("The published reference bundle is unreadable JSON."); }
    if (!object(bundle) || bundle.schema_version !== 1 || typeof bundle.system_prompt !== "string" || !bundle.system_prompt.trim() || typeof bundle.reference_digest !== "string" || typeof bundle.bundle_digest !== "string" || !/^[a-f0-9]{64}$/.test(bundle.reference_digest) || !/^[a-f0-9]{64}$/.test(bundle.bundle_digest) || !Array.isArray(bundle.schemas) || !object(bundle.results) || !object(bundle.citation_rows)) throw new Error("The published reference bundle has an unsupported schema.");
    const { metadata } = referenceMetadata(bundle);
    checkSignal(signal);
    referenceBundle = bundle;
    return metadata;
  }

  async function complete(config, system, messages, signal) {
    const headers = { "Content-Type": "application/json", Accept: "application/json" };
    const body = { model: config.model, messages: messages.map((item) => ({ role: item.role, content: item.content })) };
    if (config.protocol === "anthropic_messages") {
      Object.assign(headers, { "x-api-key": config.api_key, "anthropic-version": "2023-06-01", "anthropic-dangerous-direct-browser-access": "true" });
      Object.assign(body, { system, max_tokens: 8192 });
    } else {
      headers.Authorization = "Bearer " + config.api_key;
      body.messages.unshift({ role: "system", content: system });
      body[config.provider === "custom" ? "max_tokens" : "max_completion_tokens"] = 8192;
      if (config.provider === "minimax") body.reasoning_split = true;
    }
    const serialized = JSON.stringify(body);
    if (bytes(serialized) > MAX_REQUEST_BYTES) throw new Error("The accumulated agent context exceeded the provider request size limit. Ask a narrower question.");
    const { response, text } = await readRequest(config.api_url, { method: "POST", headers, body: serialized }, { signal });
    if (!response.ok) {
      const error = new Error("HTTP " + response.status + " from " + config.provider + ": " + redactString(text, config.api_key).slice(0, 500));
      error.status = response.status;
      throw error;
    }
    let payload;
    try { payload = JSON.parse(text); } catch (_) { throw new Error("The provider returned unreadable JSON."); }
    if (!object(payload)) throw new Error("The provider returned an invalid response object.");
    if (payload.error || object(payload.base_resp) && payload.base_resp.status_code) throw new Error("The provider returned an error: " + redactString(JSON.stringify(payload.error || payload.base_resp.status_msg || payload.base_resp), config.api_key).slice(0, 500));
    let content;
    if (config.protocol === "anthropic_messages") {
      if (payload.stop_reason === "max_tokens") throw new Error("The provider reply reached its output token limit.");
      if (!Array.isArray(payload.content)) throw new Error("The provider returned no text content blocks.");
      content = payload.content.filter((part) => object(part) && part.type === "text" && typeof part.text === "string").map((part) => part.text).join("");
    } else {
      if (!Array.isArray(payload.choices) || !object(payload.choices[0])) throw new Error("The provider reply has no choices.");
      if (payload.choices[0].finish_reason === "length") throw new Error("The provider reply reached its output token limit.");
      content = payload.choices[0].message?.content;
      if (Array.isArray(content)) content = content.filter((part) => object(part) && typeof part.text === "string").map((part) => part.text).join("");
      if (config.provider === "minimax" && typeof content === "string") {
        content = content.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
        if (content.includes("<think>")) throw new Error("The MiniMax reply holds no final message content.");
      }
    }
    if (typeof content !== "string" || !content.trim()) throw new Error("The provider reply holds no final message content.");
    return content;
  }

  function parseTurn(raw) {
    let text = raw.trim();
    const fence = /^```(?:json)?\s*([\s\S]*?)\s*```$/.exec(text);
    if (fence) text = fence[1];
    let turn;
    try { turn = JSON.parse(text); } catch (_) { throw new Error("The provider returned malformed JSON in the agent protocol."); }
    if (!object(turn) || !Object.keys(turn).length || Object.keys(turn).some((name) => !["reasoning", "questions", "tool_calls"].includes(name))) throw new Error("The provider returned no supported agent JSON fields.");
    if (Object.hasOwn(turn, "reasoning") && (typeof turn.reasoning !== "string" || turn.reasoning.length > 64000)) throw new Error("The provider’s reasoning must be bounded text.");
    const questions = Object.hasOwn(turn, "questions") ? turn.questions : [];
    const calls = Object.hasOwn(turn, "tool_calls") ? turn.tool_calls : [];
    if (!Array.isArray(questions) || questions.length > 20 || questions.some((question) => typeof question !== "string" || question.length > 8000)) throw new Error("The provider’s questions must be a bounded array of strings.");
    if (!Array.isArray(calls) || calls.length > 64 || calls.some((call) => !object(call) || typeof call.interface !== "string" || !call.interface || call.interface.length > 200 || (call.arguments !== undefined && !object(call.arguments)) || bytes(JSON.stringify(call.arguments ?? {})) > 8192 || (call.rationale !== undefined && (typeof call.rationale !== "string" || call.rationale.length > 2000)))) throw new Error("The provider returned an invalid agent tool call.");
    return { reasoning: turn.reasoning ?? "", questions, tool_calls: calls.map((call) => ({ interface: call.interface, arguments: call.arguments ?? {}, rationale: call.rationale ?? "" })) };
  }

  function boundedResult(result, call) {
    if (!object(result) || typeof result.ok !== "boolean") return { tool: call.interface, arguments: call.arguments, rationale: call.rationale, ok: false, refusal: "The local tool returned an invalid result.", elapsed_ms: 0, truncated: false };
    const value = JSON.stringify(result.value ?? null);
    if (bytes(value) > LIMITS.max_result_bytes) {
      // Measure the serialized wrapper, including escaped and multi-byte text.
      const truncated = { truncated: true, note: "This result exceeded 8000 bytes. Ask a narrower question.", head: "" };
      let low = 0, high = value.length;
      while (low < high) {
        const middle = Math.ceil((low + high) / 2);
        truncated.head = value.slice(0, middle);
        if (bytes(JSON.stringify(truncated)) <= LIMITS.max_result_bytes) low = middle;
        else high = middle - 1;
      }
      truncated.head = value.slice(0, low);
      return { ...result, truncated: true, value: truncated };
    }
    return result;
  }

  async function chat({ message, history = [], model_config, signal, onProgress } = {}) {
    let config;
    let transcript = { question: "", provider: "", runs_remotely: true, execution: "browser", limits: { ...LIMITS }, tools_offered: [], turns: [], tool_calls_made: 0, stopped_because: "The agent did not complete.", answer: "", caveat: "The model used a browser port of E-Agent’s read-only reference tools and the published evidence bundle. No experiment was run and no reference data was changed. Review the evidence and tool trace." };
    const controller = new AbortController();
    const abort = () => controller.abort(signal?.reason);
    signal?.addEventListener("abort", abort, { once: true });
    let timer;
    const progress = (text) => { if (typeof onProgress === "function") onProgress(redactString(text, config?.api_key)); };
    const failure = (error) => {
      transcript.answer = "";
      const clean = redact({ error: error.message || String(error), transcript, completion_verified: false, ...(config ? { model_config: publicMetadata(config) } : {}) }, config?.api_key);
      const reported = new Error(clean.error);
      reported.data = clean;
      if (error.status) reported.status = error.status;
      return reported;
    };
    try {
      config = modelConfiguration(model_config);
      message = chatInput(message, history, config);
      const question = history.length ? "Previous conversation (context only; tool results must still support factual answers):\n" + JSON.stringify(history) + "\nCurrent user message:\n" + message : message;
      transcript.question = question;
      transcript.provider = config.provider + ":" + config.model;
      checkSignal(signal);
      timer = setTimeout(() => controller.abort(new Error("The agent’s 180 s wall-clock limit was reached; no completion was verified.")), LIMITS.max_wall_seconds * 1000);
      progress("Reading the published reference bundle…");
      await prepare({ signal: controller.signal });
      const tools = referenceMetadata(referenceBundle).tools;
      transcript.tools_offered = tools.toolNames();
      const system = tools.systemPrompt + "\nAvailable tool interfaces:\n" + JSON.stringify(tools.schemas());
      const messages = [{ role: "user", content: question }];
      for (let index = 0; index < LIMITS.max_turns; index++) {
        checkSignal(controller.signal);
        let turn;
        try {
          progress("Calling " + config.provider + " · agent turn " + (index + 1) + "…");
          turn = parseTurn(await complete(config, system, messages, controller.signal));
        } catch (error) {
          transcript.turns.push({ turn: index, provider_error: error.message || String(error) });
          transcript.stopped_because = "The provider call failed: " + (error.message || String(error));
          throw error;
        }
        checkSignal(controller.signal);
        const guard = tools.inspect([turn.reasoning, ...turn.questions].join("\n\n"));
        const entry = { turn: index, reasoning: turn.reasoning, questions: turn.questions, requested: turn.tool_calls.map((call) => call.interface), guard };
        transcript.turns.push(entry);
        if (guard.clean !== true) {
          entry.guard = { ...guard, refused: true };
          transcript.stopped_because = "The numeric guard refused the model’s output: " + guard.summary;
          throw new Error(transcript.stopped_because);
        }
        if (!turn.tool_calls.length) {
          const answer = [turn.reasoning.trim(), turn.questions.map((question) => question.trim()).filter(Boolean).join("\n")].filter(Boolean).join("\n\n");
          if (!answer) { transcript.stopped_because = "The provider returned no answer or clarification in the agent protocol."; throw new Error(transcript.stopped_because); }
          transcript.stopped_because = normalStop;
          transcript.answer = answer;
          return redact({ answer, transcript, completion_verified: true, model_config: publicMetadata(config) }, config.api_key);
        }
        if (turn.tool_calls.length > LIMITS.max_calls_per_turn) entry.clipped = "Only the first " + LIMITS.max_calls_per_turn + " tool calls were executed.";
        entry.results = [];
        for (const call of turn.tool_calls.slice(0, LIMITS.max_calls_per_turn)) {
          checkSignal(controller.signal);
          progress("Running local tool: " + call.interface + "…");
          let result;
          try { result = boundedResult(await withSignal(tools.execute(call), controller.signal), call); }
          catch (error) { result = { tool: call.interface, arguments: call.arguments, rationale: call.rationale, ok: false, refusal: error.message || String(error), truncated: false, elapsed_ms: 0 }; }
          entry.results.push(result);
          transcript.tool_calls_made++;
          checkSignal(controller.signal);
          if (transcript.tool_calls_made >= LIMITS.max_calls_total) {
            transcript.stopped_because = "The total limit of " + LIMITS.max_calls_total + " tool calls was reached.";
            throw new Error(transcript.stopped_because);
          }
        }
        messages.push({ role: "assistant", content: JSON.stringify(redact(turn, config.api_key)) }, { role: "user", content: "Tool results:\n" + JSON.stringify(redact(entry.results, config.api_key)) });
      }
      transcript.stopped_because = "The turn limit of " + LIMITS.max_turns + " was reached.";
      throw new Error(transcript.stopped_because);
    } catch (error) {
      if (transcript.stopped_because === "The agent did not complete.") transcript.stopped_because = error.message || String(error);
      throw failure(error);
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    }
  }

  root.EAgentBrowserRuntime = Object.freeze({ prepare, chat });
})();
