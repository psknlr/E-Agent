"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { resolve } = require("../web/endpoint.js");

// A base URL must be completed to the endpoint it stands for; a full endpoint
// must come back exactly as written. Wrongly rewriting a working URL would be
// worse than not helping, so the "left alone" cases matter as much.
test("a bare version base URL is completed for the OpenAI chat format", () => {
  for (const [input, expected] of [
    ["https://api.minimax.cn/v1", "https://api.minimax.cn/v1/chat/completions"],
    ["https://api.minimax.cn/v1/", "https://api.minimax.cn/v1/chat/completions"],
    ["https://api.openai.com/v1", "https://api.openai.com/v1/chat/completions"],
    ["https://open.bigmodel.cn/api/paas/v4", "https://open.bigmodel.cn/api/paas/v4/chat/completions"],
    ["https://gateway.example/openai/v1///", "https://gateway.example/openai/v1/chat/completions"],
    ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1/chat/completions"],
  ]) {
    const result = resolve(input, "openai_chat");
    assert.equal(result.url, expected, input);
    assert.equal(result.completed, true, input);
    assert.equal(result.appended, "/chat/completions");
  }
});

test("a bare version base URL is completed for the Anthropic Messages format", () => {
  const result = resolve("https://api.anthropic.com/v1", "anthropic_messages");
  assert.equal(result.url, "https://api.anthropic.com/v1/messages");
  assert.equal(result.completed, true);
});

test("a full endpoint is returned exactly as written", () => {
  for (const [input, protocol] of [
    ["https://api.minimax.io/v1/chat/completions", "openai_chat"],
    ["https://api.anthropic.com/v1/messages", "anthropic_messages"],
    ["https://provider.example/v1/chat/completions", "openai_chat"],
    ["https://provider.example/custom/path", "openai_chat"],
    ["https://provider.example/v1beta/openai", "openai_chat"],
    ["https://provider.example/version1", "openai_chat"],
    ["https://provider.example/v1/completions", "openai_chat"],
  ]) {
    const result = resolve(input, protocol);
    assert.equal(result.url, new URL(input).href, input);
    assert.equal(result.completed, false, input);
    assert.equal(result.appended, "");
  }
});

test("an unknown format never rewrites the URL", () => {
  const result = resolve("https://provider.example/v1", "unsupported");
  assert.equal(result.url, "https://provider.example/v1");
  assert.equal(result.completed, false);
  assert.equal(resolve("https://provider.example/v1", "constructor").completed, false);
});

test("resolving twice changes nothing the second time", () => {
  const once = resolve("https://api.minimax.cn/v1", "openai_chat");
  const twice = resolve(once.url, "openai_chat");
  assert.equal(twice.url, once.url);
  assert.equal(twice.completed, false);
});
