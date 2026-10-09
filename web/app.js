"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const ui = {
    url: byId("backend-url"), token: byId("access-token"), settings: byId("connection-settings"),
    toggle: byId("settings-toggle"), connect: byId("connect-button"), error: byId("connection-error"),
    dot: byId("status-dot"), label: byId("status-label"), detail: byId("status-detail"), meta: byId("backend-meta"),
    tools: byId("tools-list"), toolsPanel: byId("tools-panel"), welcome: byId("welcome"), messages: byId("messages"),
    progress: byId("request-progress"), form: byId("chat-form"), message: byId("message"), send: byId("send-button"),
    cancel: byId("cancel-request"), reset: byId("reset-chat"), notice: byId("composer-notice"), scroll: byId("conversation-scroll"),
    provider: byId("model-provider"), apiKey: byId("provider-key"), model: byId("model-name"), apiUrl: byId("api-url"),
    protocol: byId("model-protocol"), manual: byId("manual-model-fields"), protocolField: byId("custom-protocol-field"),
    serverHelp: byId("server-model-help"), feedback: byId("model-feedback"), modelSettings: byId("model-settings"),
    savedTemplate: byId("saved-template"), templateName: byId("template-name"), saveTemplate: byId("save-template"),
    exportTemplate: byId("export-template"), importTemplate: byId("import-template"), templateFile: byId("template-file"),
  };
  const state = { backend: "", token: "", health: null, history: [], verified: false, busy: false, connecting: false, controller: null, generation: 0 };
  const storageKey = "eagent.backend_url";
  const templatesKey = "eagent.model_templates.v1";
  const presets = {
    minimax: { protocol: "openai_chat", model: "MiniMax-M2.7", api_url: "https://api.minimax.io/v1/chat/completions" },
    openai: { protocol: "openai_chat", model: "", api_url: "https://api.openai.com/v1/chat/completions" },
    anthropic: { protocol: "anthropic_messages", model: "", api_url: "https://api.anthropic.com/v1/messages" },
    custom: { protocol: "openai_chat", model: "", api_url: "" },
    server_default: { protocol: "openai_chat", model: "", api_url: "" },
  };
  let templates = [];
  const localHost = (hostname) => ["localhost", "127.0.0.1", "[::1]"].includes(hostname.toLowerCase());
  const make = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  };

  function validateBackend(value) {
    const text = value.trim();
    if (!text) throw new Error("Enter the URL of your running E-Agent backend.");
    let url;
    try { url = new URL(text); } catch (_) { throw new Error("Enter a complete backend URL, including https://."); }
    if (url.username || url.password || url.search || url.hash) throw new Error("Backend URLs cannot contain credentials, query parameters, or fragments.");
    if (url.protocol !== "https:" && !(url.protocol === "http:" && localHost(url.hostname))) {
      throw new Error("Use HTTPS for a remote backend. HTTP is allowed only on localhost.");
    }
    return url.href.replace(/\/+$/, "");
  }

  function status(label, detail, kind = "") {
    ui.label.textContent = label;
    ui.detail.textContent = detail;
    ui.dot.className = "status-dot" + (kind ? " " + kind : "");
    syncComposer();
  }

  function requiresToken() {
    if (state.health && typeof state.health.authentication_required === "boolean") return state.health.authentication_required;
    return state.backend ? !localHost(new URL(state.backend).hostname) : true;
  }

  function validateApiUrl(value) {
    const input = value.trim();
    if (!input) throw new Error("Enter the full model API URL in Model settings.");
    if (input.length > 2048) throw new Error("The API URL is too long.");
    let url;
    try { url = new URL(input); } catch (_) { throw new Error("Enter a complete model API URL, including https:// and its request path."); }
    if (url.username || url.password || url.search || url.hash) throw new Error("API URLs cannot contain credentials, query parameters, or fragments.");
    if (url.protocol !== "https:" && !(url.protocol === "http:" && localHost(url.hostname))) throw new Error("Use HTTPS for a remote API. HTTP is allowed only on localhost.");
    if (url.pathname === "/") throw new Error("Include the full API request path, such as /v1/chat/completions.");
    return url.href;
  }

  function publicProfile() {
    const provider = ui.provider.value;
    if (!Object.hasOwn(presets, provider)) throw new Error("Choose a supported provider preset.");
    const protocol = provider === "custom" ? ui.protocol.value : presets[provider].protocol;
    if (!["openai_chat", "anthropic_messages"].includes(protocol)) throw new Error("Choose a supported API format.");
    if (provider === "server_default") return { provider, protocol, model: "", api_url: "" };
    const model = ui.model.value.trim();
    if (!model || model.length > 200 || /[\x00-\x1f\x7f]/.test(model)) throw new Error("Enter your exact model name in Model settings (up to 200 characters).");
    return { provider, protocol, model, api_url: validateApiUrl(ui.apiUrl.value) };
  }

  function modelConfig() {
    const profile = publicProfile();
    if (profile.provider === "server_default") return null;
    const apiKey = ui.apiKey.value.trim();
    if (!apiKey || apiKey.length > 4096 || /[\x00-\x1f\x7f]/.test(apiKey)) throw new Error("Enter your provider API key in Model settings.");
    return { ...profile, api_key: apiKey };
  }

  function configurationIssue() {
    if (!state.health) return "Connect your backend in Agent connection to begin.";
    if (ui.provider.value === "server_default") return state.health.ready ? "" : state.health.reason || "The server default model needs configuration on your backend.";
    if (state.health.runtime_ready !== true) return state.health.runtime_ready === false ? state.health.runtime_reason || "The backend’s Python tools are not ready." : "Update your E-Agent backend to enable DIY model settings.";
    try { modelConfig(); } catch (error) { return error.message; }
    return "";
  }

  function refreshConfigurationStatus() {
    if (!state.health || state.busy || state.connecting) { syncComposer(); return; }
    const issue = configurationIssue();
    const manual = ui.provider.value !== "server_default";
    ui.meta.hidden = false;
    ui.meta.textContent = manual ? [ui.provider.options[ui.provider.selectedIndex].text, ui.model.value.trim()].filter(Boolean).join(" · ") : [state.health.provider, state.health.model].filter(Boolean).join(" · ");
    if (issue) status("Needs configuration", issue, "error");
    else if (state.verified) status("Chat ready", "A successful response verified the selected model in this conversation.", "ready");
    else status("Ready to verify", "Backend connected. Send a question to verify the selected model and its API settings.", "configured");
  }

  function syncComposer() {
    const issue = configurationIssue();
    const configured = !issue;
    const hasToken = !requiresToken() || ui.token.value.trim().length > 0;
    ui.send.disabled = state.busy || state.connecting || !configured || !hasToken || !ui.message.value.trim();
    ui.cancel.hidden = !state.busy;
    ui.progress.hidden = !state.busy;
    ui.message.disabled = state.busy;
    ui.connect.disabled = state.busy || state.connecting;
    ui.url.disabled = state.busy;
    ui.token.disabled = state.busy;
    ui.reset.disabled = state.busy;
    for (const input of [ui.provider, ui.apiKey, ui.model, ui.apiUrl, ui.protocol, ui.savedTemplate, ui.templateName, ui.saveTemplate, ui.exportTemplate, ui.importTemplate]) input.disabled = state.busy || state.connecting;
    if (state.busy) ui.notice.textContent = "A model request is running. Its tool trace will appear with the response. You can cancel at any time.";
    else if (!configured) ui.notice.textContent = issue;
    else if (!hasToken) ui.notice.textContent = "Enter your backend access token in Agent connection to send a question.";
    else if (!state.verified) ui.notice.textContent = "Model settings complete. The first successful response will verify chat readiness.";
    else ui.notice.textContent = "Chat ready · Questions run through the connected backend and its registered Python tools.";
  }

  function hideSettings(hidden) {
    ui.settings.hidden = hidden;
    ui.toggle.setAttribute("aria-expanded", String(!hidden));
    ui.toggle.setAttribute("aria-label", hidden ? "Open connection settings" : "Close connection settings");
  }

  function clearConversation() {
    state.history = [];
    state.verified = false;
    ui.messages.replaceChildren();
    ui.welcome.hidden = false;
    ui.message.value = "";
    resizeInput();
    syncComposer();
  }

  function showHealth(data) {
    state.health = data;
    ui.tools.replaceChildren();
    for (const tool of Array.isArray(data.tools) ? data.tools : []) ui.tools.append(make("li", "", tool));
    ui.toolsPanel.hidden = ui.tools.childElementCount === 0;
    refreshConfigurationStatus();
  }

  function updateModelFields() {
    const manual = ui.provider.value !== "server_default";
    ui.manual.hidden = !manual;
    ui.serverHelp.hidden = manual;
    ui.protocolField.hidden = ui.provider.value !== "custom";
    ui.model.placeholder = ui.provider.value === "anthropic" ? "Your Claude model name" : ui.provider.value === "openai" ? "Your GPT model name" : "Your exact model name";
  }

  function modelChanged(clearHistory) {
    state.verified = false;
    if (clearHistory) {
      const draft = ui.message.value;
      clearConversation();
      ui.message.value = draft;
      resizeInput();
    }
    templateFeedback("Settings changed. Send a question to verify this model.");
    refreshConfigurationStatus();
  }

  function applyProfile(profile) {
    ui.provider.value = profile.provider;
    ui.protocol.value = profile.protocol;
    ui.model.value = profile.model;
    ui.apiUrl.value = profile.api_url;
    ui.apiKey.value = "";
    ui.templateName.value = profile.name || "";
    updateModelFields();
    modelChanged(true);
  }

  function currentTemplate() {
    const name = ui.templateName.value.trim() || "My " + ui.provider.options[ui.provider.selectedIndex].text + " settings";
    if (name.length > 80 || /[\x00-\x1f\x7f]/.test(name)) throw new Error("Use a template name of up to 80 characters.");
    return { schema: "eagent-model-template", version: 1, name, ...publicProfile() };
  }

  function validateTemplate(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Choose an exported E-Agent model template JSON file.");
    const allowed = ["schema", "version", "name", "provider", "protocol", "model", "api_url"];
    if (Object.keys(value).some((key) => /key|token|secret|password/i.test(key))) throw new Error("This file contains a key or token field. Remove secrets before importing; enter your API key in the password field instead.");
    if (Object.keys(value).some((key) => !allowed.includes(key)) || allowed.some((key) => !Object.hasOwn(value, key))) throw new Error("The template has unrecognized or missing fields. Import an exported E-Agent model template.");
    if (value.schema !== "eagent-model-template" || value.version !== 1 || typeof value.provider !== "string" || !Object.hasOwn(presets, value.provider)) throw new Error("This E-Agent template format or provider is not supported.");
    if (typeof value.name !== "string" || !value.name.trim() || value.name.length > 80 || /[\x00-\x1f\x7f]/.test(value.name)) throw new Error("The template needs a valid name of up to 80 characters.");
    if (!["openai_chat", "anthropic_messages"].includes(value.protocol)) throw new Error("This API format is not supported.");
    if (value.provider !== "custom" && value.protocol !== presets[value.provider].protocol) throw new Error("The template’s API format does not match its provider preset. Use Custom endpoint for another format.");
    if (typeof value.model !== "string" || typeof value.api_url !== "string") throw new Error("Model name and API URL must be text.");
    if (value.provider === "server_default") {
      if (value.model !== "" || value.api_url !== "") throw new Error("Server default templates must leave model and API URL empty.");
    } else {
      if (!value.model.trim() || value.model.length > 200 || /[\x00-\x1f\x7f]/.test(value.model)) throw new Error("The template needs a valid model name of up to 200 characters.");
      validateApiUrl(value.api_url);
    }
    // Copy only allowed public fields, even for files loaded from local storage.
    return { schema: value.schema, version: value.version, name: value.name.trim(), provider: value.provider, protocol: value.protocol, model: value.model.trim(), api_url: value.provider === "server_default" ? "" : validateApiUrl(value.api_url) };
  }

  function refreshTemplates(selected = "") {
    ui.savedTemplate.replaceChildren(make("option", "", "Choose a template…"));
    ui.savedTemplate.firstElementChild.value = "";
    templates.forEach((template, index) => {
      const option = make("option", "", template.name);
      option.value = String(index);
      ui.savedTemplate.append(option);
    });
    ui.savedTemplate.value = selected;
  }

  function templateFeedback(message, error = false) {
    ui.feedback.textContent = message;
    ui.feedback.className = error ? "field-error" : "field-help";
  }

  function loadTemplates() {
    try {
      const stored = JSON.parse(localStorage.getItem(templatesKey) || "[]");
      if (!Array.isArray(stored) || stored.length > 50) throw new Error("Invalid stored templates.");
      templates = stored.map(validateTemplate);
    } catch (_) { templates = []; }
    refreshTemplates();
  }

  function saveTemplate() {
    try {
      const template = currentTemplate();
      const index = templates.findIndex((entry) => entry.name === template.name);
      if (index < 0 && templates.length >= 50) throw new Error("You can save up to 50 templates. Use an existing name to replace one.");
      const updated = templates.slice();
      if (index < 0) updated.push(template); else updated[index] = template;
      localStorage.setItem(templatesKey, JSON.stringify(updated));
      templates = updated;
      refreshTemplates(String(index < 0 ? templates.length - 1 : index));
      templateFeedback("Template saved on this browser. API key and backend token were excluded.");
    } catch (error) { templateFeedback(error.message || "This browser could not save the template.", true); }
  }

  function exportTemplate() {
    try {
      const template = currentTemplate();
      const blobUrl = URL.createObjectURL(new Blob([safeJson(template) + "\n"], { type: "application/json" }));
      const link = make("a");
      link.href = blobUrl;
      link.download = "eagent-" + template.name.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").slice(0, 60) + ".json";
      link.click();
      setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
      templateFeedback("Template exported. API key and backend token were excluded.");
    } catch (error) { templateFeedback(error.message, true); }
  }

  async function importTemplate() {
    const file = ui.templateFile.files[0];
    ui.templateFile.value = "";
    if (!file || state.busy || state.connecting) return;
    try {
      if (file.size > 16384) throw new Error("Choose a model template JSON file smaller than 16 KB.");
      const profile = validateTemplate(JSON.parse(await file.text()));
      if (state.busy || state.connecting) return;
      applyProfile(profile);
      ui.savedTemplate.value = "";
      templateFeedback("Template loaded. Enter your API key again; Save keeps a copy on this browser.");
    } catch (error) { templateFeedback(error instanceof SyntaxError ? "This file is not valid JSON." : error.message, true); }
  }

  async function jsonRequest(url, options, timeout) {
    const controller = new AbortController();
    state.controller = controller;
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeout);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal, cache: "no-store", credentials: "omit", redirect: "error", referrerPolicy: "no-referrer" });
      let data;
      try { data = await response.json(); } catch (_) { throw new Error("The backend returned an unreadable response. Check the backend URL and server logs."); }
      if (!response.ok) {
        const error = new Error(typeof data.error === "string" ? data.error : "The backend rejected this request (HTTP " + response.status + ").");
        error.status = response.status;
        error.data = data;
        throw error;
      }
      return data;
    } catch (error) {
      if (error.name === "AbortError") throw new Error(timedOut ? "The backend request timed out. Check the server and reconnect before trying again." : "Request cancelled. The backend may still finish work already started.");
      if (error instanceof TypeError) throw new Error("Could not reach the backend. Check its URL, HTTPS connection, and allowed browser origin.");
      throw error;
    } finally {
      clearTimeout(timer);
      if (state.controller === controller) state.controller = null;
    }
  }

  async function connect(event) {
    if (event) event.preventDefault();
    if (state.busy || state.connecting) return;
    ui.error.hidden = true;
    let backend;
    try { backend = validateBackend(ui.url.value); } catch (error) { ui.error.textContent = error.message; ui.error.hidden = false; return; }
    if (backend !== state.backend) clearConversation();
    state.backend = backend;
    state.token = ui.token.value.trim();
    state.health = null;
    state.verified = false;
    ui.meta.hidden = true;
    ui.toolsPanel.hidden = true;
    state.connecting = true;
    ui.connect.textContent = "Checking backend…";
    status("Checking backend", "Checking the backend health endpoint. This does not verify a model response.", "checking");
    const generation = ++state.generation;
    try {
      const data = await jsonRequest(backend + "/api/health", { method: "GET", headers: { Accept: "application/json" } }, 30000);
      if (generation !== state.generation) return;
      if (!data || data.service !== "eagent" || typeof data.ready !== "boolean") throw new Error("This URL did not return an E-Agent health response.");
      try { localStorage.setItem(storageKey, backend); } catch (_) { /* URL storage is optional. */ }
      showHealth(data);
      if (!configurationIssue() && (!requiresToken() || state.token)) hideSettings(true);
    } catch (error) {
      if (generation !== state.generation) return;
      state.health = null;
      status(error.status === 401 || error.status === 403 ? "Needs configuration" : "Backend unavailable", error.message, "error");
      ui.error.textContent = error.message;
      ui.error.hidden = false;
    } finally {
      if (generation === state.generation) {
        state.connecting = false;
        ui.connect.replaceChildren(document.createTextNode("Connect backend "), make("span", "", "↗"));
        refreshConfigurationStatus();
      }
    }
  }

  function scrollToLatest() {
    requestAnimationFrame(() => {
      if (window.matchMedia("(max-width: 650px)").matches) ui.progress.hidden ? ui.messages.lastElementChild?.scrollIntoView({ behavior: "smooth", block: "start" }) : ui.progress.scrollIntoView({ behavior: "smooth", block: "nearest" });
      else ui.scroll.scrollTop = ui.scroll.scrollHeight;
    });
  }

  function appendMessage(role, content, error = false) {
    ui.welcome.hidden = true;
    const article = make("article", "message " + role + (error ? " error" : ""));
    const heading = make("div", "message-heading");
    heading.append(make("span", "message-avatar", role === "user" ? "Y" : "E·"), make("span", "", role === "user" ? "YOU" : error ? "REQUEST NOTICE" : "E-AGENT"));
    const time = make("time", "", new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    time.dateTime = new Date().toISOString();
    heading.append(time);
    article.append(heading, make("div", "message-body", content));
    ui.messages.append(article);
    return article;
  }

  function safeJson(value) {
    try { return JSON.stringify(value, null, 2); } catch (_) { return "Transcript could not be serialized."; }
  }

  function addDetail(container, title, value) {
    if (value === undefined || value === null || value === "") return;
    const details = make("details");
    details.append(make("summary", "", title), make("pre", "", typeof value === "string" ? value : safeJson(value)));
    container.append(details);
  }

  function addTrace(article, transcript, response) {
    if (!transcript || typeof transcript !== "object") {
      article.append(make("div", "run-notice", "The backend did not return a tool transcript. This response’s evidence trace is unavailable."));
      return;
    }
    const turns = Array.isArray(transcript.turns) ? transcript.turns : [];
    const results = turns.flatMap((turn) => Array.isArray(turn.results) ? turn.results : []);
    const failed = results.filter((result) => result.ok === false).length;
    const notices = [];
    for (const turn of turns) {
      if (turn.provider_error) notices.push("Provider error on turn " + turn.turn + ": " + (typeof turn.provider_error === "string" ? turn.provider_error : safeJson(turn.provider_error)));
      const guard = turn.guard;
      if (guard && (guard.clean === false || guard.refused)) {
        notices.push("Citation guard on turn " + turn.turn + ": " + (guard.summary || "The answer contained quantities or citations that could not be verified."));
        if (guard.uncited_quantities?.length) notices.push("Uncited quantities: " + safeJson(guard.uncited_quantities));
        if (guard.broken_citations?.length) notices.push("Broken citations: " + safeJson(guard.broken_citations));
      }
    }
    if (failed) notices.push(failed + " tool call" + (failed === 1 ? "" : "s") + " failed or refused. Review the trace below.");
    if (response.warnings) notices.push(typeof response.warnings === "string" ? response.warnings : safeJson(response.warnings));
    if (notices.length) article.append(make("div", "run-notice", notices.join("\n\n")));
    if (transcript.caveat) article.append(make("div", "run-notice neutral", transcript.caveat));
    if (transcript.stopped_because) {
      const normal = transcript.stopped_because === "the model answered without asking for another tool";
      article.append(make("div", "run-notice" + (normal ? " neutral" : ""), "Run ended: " + transcript.stopped_because));
    }
    const trace = make("details", "trace");
    const count = Number.isInteger(transcript.tool_calls_made) ? transcript.tool_calls_made : results.length;
    trace.append(make("summary", "", "View research trace · " + count + " tool call" + (count === 1 ? "" : "s") + (failed ? " · " + failed + " failed" : "")));
    const body = make("div", "trace-body");
    if (!results.length) body.append(make("p", "tool-reason", "The model did not call a Python tool in this request. Consult the full transcript for its reasoning and citation checks."));
    for (const turn of turns) {
      for (const result of Array.isArray(turn.results) ? turn.results : []) {
        const item = make("section", "tool-result");
        const heading = make("div", "tool-heading");
        const label = result.ok === false ? "Failed / refused" : result.ok === true ? "Completed" : "Status not reported";
        const elapsed = typeof result.elapsed_ms === "number" ? " · " + (result.elapsed_ms / 1000).toFixed(2) + "s" : "";
        heading.append(make("span", "", result.tool || "Unnamed tool"), make("span", "tool-state" + (result.ok === false ? " failed" : ""), label + elapsed));
        item.append(heading);
        if (result.rationale) item.append(make("p", "tool-reason", result.rationale));
        if (result.refusal) item.append(make("p", "field-error", typeof result.refusal === "string" ? result.refusal : safeJson(result.refusal)));
        if (result.truncated) item.append(make("p", "field-error", "The tool result was truncated. The visible output is incomplete."));
        addDetail(item, "Arguments", result.arguments);
        addDetail(item, "Returned evidence", result.value);
        body.append(item);
      }
    }
    const footer = make("div", "trace-footer");
    addDetail(footer, "Full transcript", transcript);
    const download = make("button", "", "Download transcript ↓");
    download.type = "button";
    download.addEventListener("click", () => {
      const blobUrl = URL.createObjectURL(new Blob([safeJson(transcript)], { type: "application/json" }));
      const link = make("a");
      link.href = blobUrl;
      link.download = "eagent-transcript-" + new Date().toISOString().replace(/[:.]/g, "-") + ".json";
      link.click();
      setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
    });
    footer.append(download);
    body.append(footer);
    trace.append(body);
    article.append(trace);
  }

  function requestPayload(message, history, configuration) {
    return configuration ? { message, history, model_config: configuration } : { message, history };
  }

  function recentHistory(message, configuration) {
    // Keep complete exchanges and the displayed conversation. The transport has
    // separate message-count, character-count, and UTF-8 request-size limits.
    const history = [];
    let characters = 0;
    for (let index = state.history.length - 2; index >= 0; index -= 2) {
      const exchange = state.history.slice(index, index + 2);
      if (history.length + exchange.length > 20) break;
      if (exchange.some((entry) => entry.content.length > 8000)) break;
      const additional = exchange.reduce((sum, entry) => sum + entry.content.length, 0);
      if (characters + additional > 24000) break;
      const candidate = [...exchange, ...history];
      if (new TextEncoder().encode(JSON.stringify(requestPayload(message, candidate, configuration))).length > 64000) break;
      history.unshift(...exchange);
      characters += additional;
    }
    return history;
  }

  async function send(event) {
    event.preventDefault();
    const message = ui.message.value.trim();
    if (state.busy || state.connecting || !message || configurationIssue() || (requiresToken() && !ui.token.value.trim())) return;
    let configuration;
    try { configuration = modelConfig(); } catch (error) { templateFeedback(error.message, true); refreshConfigurationStatus(); return; }
    state.token = ui.token.value.trim();
    const generation = state.generation;
    const history = recentHistory(message, configuration);
    const payload = JSON.stringify(requestPayload(message, history, configuration));
    if (new TextEncoder().encode(payload).length > 64000) {
      templateFeedback("The request exceeds the backend’s 64 KB limit. Shorten your question or model settings.", true);
      return;
    }
    const question = appendMessage("user", message);
    if (history.length < state.history.length) question.append(make("div", "run-notice neutral", "Older conversation context was omitted to fit the backend’s input limits. This request includes the most recent " + (history.length / 2) + " complete exchange" + (history.length === 2 ? "" : "s") + ". The full conversation remains visible here."));
    ui.message.value = "";
    resizeInput();
    state.busy = true;
    status("Request running", "Waiting for the model and E-Agent tool loop to finish.", "checking");
    scrollToLatest();
    try {
      const headers = { "Content-Type": "application/json", Accept: "application/json" };
      if (state.token) headers.Authorization = "Bearer " + state.token;
      const data = await jsonRequest(state.backend + "/api/chat", { method: "POST", headers, body: payload }, 245000);
      if (generation !== state.generation) return;
      if (typeof data.answer !== "string" || !data.answer.trim()) throw new Error("The backend returned no answer. Check the server trace before retrying.");
      const article = appendMessage("assistant", data.answer);
      addTrace(article, data.transcript, data);
      state.history.push({ role: "user", content: message }, { role: "assistant", content: data.answer });
      state.verified = data.completion_verified === true;
      if (data.model_config && typeof data.model_config === "object") ui.meta.textContent = [data.model_config.provider, data.model_config.model].filter((value) => typeof value === "string").join(" · ");
      if (state.verified) status("Chat ready", "A real model response completed through the connected E-Agent backend.", "ready");
      else status("Backend configured", "A response was returned, but the backend did not verify model completion. Review the trace.", "configured");
    } catch (error) {
      if (generation !== state.generation) return;
      const article = appendMessage("assistant", error.message + "\n\nYour question was not added to the model’s conversation history. You can edit it and retry after resolving the issue.", true);
      if (error.data?.transcript) addTrace(article, error.data.transcript, error.data);
      state.verified = false;
      if (error.status === 401 || error.status === 403) {
        state.health = null;
        status("Needs configuration", "The backend rejected the access token. Update the token and reconnect.", "error");
        hideSettings(false);
      } else if (error.status === 400 || error.status === 413 || error.status === 429) {
        status("Model request failed", error.message + " Check your settings and retry; model completion was not verified.", "error");
      } else if (error.status === 503) {
        state.health = null;
        status("Needs configuration", error.message, "error");
        hideSettings(false);
      } else if (error.message.startsWith("Request cancelled")) {
        status("Backend configured", "The request was cancelled. Chat readiness will be verified by the next successful model response.", "configured");
      } else if (error.status) {
        status("Model request failed", error.message + " Check Model settings and retry. The backend is still connected.", "error");
      } else {
        state.health = null;
        status("Backend unavailable", error.message, "error");
        hideSettings(false);
      }
      ui.message.value = message;
      resizeInput();
    } finally {
      if (generation === state.generation) {
        state.busy = false;
        syncComposer();
        scrollToLatest();
        ui.message.focus({ preventScroll: true });
      }
    }
  }

  function resizeInput() {
    ui.message.style.height = "auto";
    ui.message.style.height = Math.min(ui.message.scrollHeight, 150) + "px";
  }

  ui.settings.addEventListener("submit", connect);
  ui.toggle.addEventListener("click", () => hideSettings(!ui.settings.hidden));
  ui.form.addEventListener("submit", send);
  ui.message.addEventListener("input", () => { resizeInput(); syncComposer(); });
  ui.message.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!ui.send.disabled) ui.form.requestSubmit(); }
  });
  ui.token.addEventListener("input", () => { state.token = ui.token.value.trim(); state.verified = false; refreshConfigurationStatus(); });
  ui.provider.addEventListener("change", () => {
    const provider = ui.provider.value;
    applyProfile({ provider, ...presets[provider] });
    ui.savedTemplate.value = "";
  });
  for (const input of [ui.model, ui.apiUrl]) input.addEventListener("input", () => { ui.savedTemplate.value = ""; modelChanged(true); });
  ui.protocol.addEventListener("change", () => { ui.savedTemplate.value = ""; modelChanged(true); });
  ui.apiKey.addEventListener("input", () => modelChanged(false));
  ui.savedTemplate.addEventListener("change", () => {
    const template = templates[Number(ui.savedTemplate.value)];
    if (ui.savedTemplate.value !== "" && template) {
      applyProfile(template);
      templateFeedback("Template loaded. Enter your API key again.");
    }
  });
  ui.saveTemplate.addEventListener("click", saveTemplate);
  ui.exportTemplate.addEventListener("click", exportTemplate);
  ui.importTemplate.addEventListener("click", () => ui.templateFile.click());
  ui.templateFile.addEventListener("change", importTemplate);
  ui.url.addEventListener("input", () => {
    ui.apiKey.value = "";
    ui.token.value = "";
    state.token = "";
    state.verified = false;
    let edited;
    try { edited = validateBackend(ui.url.value); } catch (_) { edited = ""; }
    if (edited !== state.backend) {
      ++state.generation;
      state.controller?.abort();
      state.connecting = false;
      state.health = null;
      state.backend = "";
      state.token = "";
      ui.token.value = "";
      ui.meta.hidden = true;
      ui.toolsPanel.hidden = true;
      clearConversation();
      ui.connect.replaceChildren(document.createTextNode("Connect backend "), make("span", "", "↗"));
      status("Not connected", "Backend changed. The conversation, API key and access token were cleared. Connect to check this backend.");
    }
    refreshConfigurationStatus();
  });
  ui.cancel.addEventListener("click", () => state.controller?.abort());
  ui.reset.addEventListener("click", () => {
    if (state.busy) return;
    clearConversation();
    if (state.health) showHealth(state.health);
    ui.message.focus({ preventScroll: true });
    ui.scroll.scrollTop = 0;
  });
  document.querySelectorAll(".starter-card").forEach((button) => button.addEventListener("click", () => {
    ui.message.value = button.dataset.prompt;
    resizeInput();
    syncComposer();
    ui.message.focus();
  }));

  async function initialize() {
    applyProfile({ provider: "minimax", ...presets.minimax });
    ui.feedback.textContent = "";
    loadTemplates();
    if (window.matchMedia("(max-width: 650px)").matches) {
      hideSettings(true);
      ui.modelSettings.open = false;
    }
    let saved = "";
    try { saved = localStorage.getItem(storageKey) || ""; } catch (_) { /* URL storage is optional. */ }
    let configured = "";
    try {
      const response = await fetch("./config.json", { cache: "no-store", credentials: "omit" });
      if (response.ok) {
        const data = await response.json();
        if (typeof data.backend_url === "string") configured = data.backend_url;
      }
    } catch (_) { /* A missing static configuration leaves the UI disconnected. */ }
    const current = localHost(window.location.hostname) && ["http:", "https:"].includes(window.location.protocol) ? window.location.origin : "";
    const candidate = configured || saved || current;
    if (candidate) {
      try {
        ui.url.value = validateBackend(candidate);
        await connect();
      } catch (_) { status("Not connected", "The saved backend URL is invalid. Open connection settings to update it."); hideSettings(false); }
    }
    syncComposer();
    resizeInput();
  }
  initialize();
})();
