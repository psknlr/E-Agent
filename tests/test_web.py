"""HTTP transport tests with real loaders and loop, mocking only model completion."""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import http.client
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
import urllib.error

from eagent.harness.llm import LLMError
from eagent.harness.providers import OpenAIChatClient
from eagent.harness.request_client import RequestModelClient
from eagent.harness.toolloop import LoopLimits
from eagent.web import (
    AgentHTTPServer, AgentService, MAX_BODY_BYTES, WebConfig, _validate_chat,
)

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "test-agent-access-secret"
KEY = "test-minimax-api-key-secret"
DIY_KEY = "test-request-only-api-key-secret"
DIY_URL = "https://93.184.216.34/v1/chat/completions"


@contextmanager
def running_server(**config_overrides):
    with mock.patch.dict(os.environ, {"MINIMAX_API_KEY": KEY}, clear=True):
        config_values = {"model": "MiniMax-M2.7", "token": TOKEN, "project_root": ROOT}
        config_values.update(config_overrides)
        config = WebConfig(**config_values)
        service = AgentService(config)
        server = AgentHTTPServer(("127.0.0.1", 0), service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server, service
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def request(server, method, path, payload=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    sent_headers = dict(headers or {})
    body = None
    if payload is not None:
        body = json.dumps(payload).encode()
        sent_headers.setdefault("Content-Type", "application/json")
    connection.request(method, path, body, sent_headers)
    response = connection.getresponse()
    status, received_headers, body = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return status, received_headers, json.loads(body) if body else None


def authorized(**extra):
    return {"Authorization": "Bearer " + TOKEN, **extra}


def diy_config(**changes):
    values = {"provider": "openai", "protocol": "openai_chat", "api_key": DIY_KEY,
              "model": "gpt-example", "api_url": DIY_URL}
    values.update(changes)
    return values


class ProviderResponse(io.BytesIO):
    status = 200

    def __init__(self, turn):
        super().__init__(json.dumps({"choices": [{"message": {"content": json.dumps(turn)}}]}).encode())


class WebTransportTests(unittest.TestCase):
    def test_diy_chat_works_without_server_credentials_and_only_verifies_its_request(self):
        with running_server() as (server, service):
            with mock.patch.dict(os.environ, {}, clear=True):
                no_default = AgentService(WebConfig(project_root=ROOT, token=TOKEN))
                server.service = no_default
                health = request(server, "GET", "/api/health")[2]
                self.assertFalse(health["ready"])
                self.assertTrue(health["runtime_ready"])
                self.assertIn("custom", health["configurable_providers"])
                environment = dict(os.environ)
                seen = []

                def opener(provider_request, **kwargs):
                    seen.append(provider_request)
                    messages = json.loads(provider_request.data)["messages"]
                    if len(messages) == 2:
                        return ProviderResponse({"tool_calls": [{"interface": "kinetic_record",
                            "arguments": {"label_id": "PaHBDH_H150N_AAE_activity_only"}}]})
                    result = json.loads(messages[-1]["content"].split("\n", 1)[1])[0]
                    self.assertTrue(result["ok"])
                    self.assertIsNone(result["value"]["km"])
                    return ProviderResponse({"reasoning": "This Km is a bound, so the actual loader withholds it."})

                with mock.patch("eagent.harness.request_client._request_opener", return_value=opener):
                    status, _, body = request(server, "POST", "/api/chat", {
                        "message": "Inspect its Km.", "model_config": diy_config()}, authorized())
                self.assertEqual(status, 200, body)
                self.assertTrue(body["completion_verified"])
                self.assertEqual(body["transcript"]["tool_calls_made"], 1)
                self.assertEqual(body["model_config"], {key: value for key, value in diy_config().items()
                                                        if key != "api_key"})
                self.assertFalse(no_default.health()["completion_verified"])
                self.assertFalse(no_default.health()["ready"])
                self.assertEqual(dict(os.environ), environment)
                self.assertNotIn(DIY_KEY, repr(no_default.__dict__))
                self.assertNotIn(DIY_KEY, json.dumps(no_default.health()))
                self.assertEqual(seen[0].get_header("Authorization"), "Bearer " + DIY_KEY)

    def test_diy_credentials_are_scrubbed_from_success_failure_and_echoed_history(self):
        with running_server() as (server, service):
            for fail in (False, True):
                with self.subTest(fail=fail):
                    def opener(provider_request, **kwargs):
                        if fail:
                            raise RuntimeError(f"transport error {DIY_KEY} {TOKEN}")
                        return ProviderResponse({"reasoning": f"echo {DIY_KEY} {TOKEN}"})

                    with mock.patch("eagent.harness.request_client._request_opener", return_value=opener):
                        status, _, body = request(server, "POST", "/api/chat", {
                            "message": "echo " + DIY_KEY,
                            "history": [{"role": "user", "content": DIY_KEY}],
                            "model_config": diy_config()}, authorized())
                    self.assertEqual(status, 502 if fail else 200, body)
                    self.assertNotIn(DIY_KEY, json.dumps(body))
                    self.assertNotIn(TOKEN, json.dumps(body))
                    self.assertEqual(body["completion_verified"], not fail)
                    self.assertFalse(service.health()["completion_verified"])

    def test_diy_concurrent_chats_keep_models_keys_and_transcripts_separate(self):
        barrier = threading.Barrier(2)
        seen = []
        lock = threading.Lock()

        def opener(provider_request, **kwargs):
            body = json.loads(provider_request.data)
            model = body["model"]
            key = provider_request.get_header("Authorization")[7:]
            with lock:
                seen.append((model, key, body["messages"][-1]["content"]))
            barrier.wait(timeout=5)
            return ProviderResponse({"reasoning": f"Response for {model}; echo {key}"})

        with running_server() as (server, service):
            with mock.patch("eagent.harness.request_client._request_opener", return_value=opener):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(request, server, "POST", "/api/chat", {
                        "message": f"Chat {name}", "model_config": diy_config(model=name, api_key=key)},
                        authorized()) for name, key in (("model-a", "request-key-a"), ("model-b", "request-key-b"))]
                    results = [future.result() for future in futures]
            for result, name in zip(results, ("model-a", "model-b")):
                self.assertEqual(result[0], 200, result[2])
                self.assertEqual(result[2]["model_config"]["model"], name)
                self.assertIn("Response for " + name, result[2]["answer"])
                self.assertNotIn("request-key-", json.dumps(result[2]))
            self.assertEqual(set(seen), {("model-a", "request-key-a", "Chat model-a"),
                                         ("model-b", "request-key-b", "Chat model-b")})
            self.assertFalse(service.health()["completion_verified"])

    def test_json_escaped_diy_secret_is_scrubbed_from_errors_and_successful_transcript(self):
        key = 'fake-quote"slash\\credential'
        escaped = json.dumps(key)[1:-1]
        with running_server() as (server, _):
            for fail in (False, True):
                with self.subTest(fail=fail):
                    def opener(provider_request, **kwargs):
                        if fail:
                            raise urllib.error.HTTPError(DIY_URL, 401, "Unauthorized", {},
                                io.BytesIO(json.dumps({"error": key}).encode()))
                        return ProviderResponse({"reasoning": "An escaped credential: " + escaped})

                    with mock.patch("eagent.harness.request_client._request_opener", return_value=opener):
                        status, _, body = request(server, "POST", "/api/chat", {
                            "message": key, "model_config": diy_config(api_key=key)}, authorized())
                    self.assertEqual(status, 502 if fail else 200, body)
                    self.assertNotIn(key, json.dumps(body))
                    self.assertNotIn(escaped, json.dumps(body))
                    self.assertIn("[credential redacted]", body["error"] if fail else body["answer"])

    def test_invalid_diy_settings_never_borrow_environment_credentials_or_call_provider(self):
        invalid = (diy_config(api_key=""), diy_config(api_url=""), diy_config(model=""),
                   diy_config(provider="unknown"), diy_config(protocol="unsupported"),
                   diy_config(protocol="anthropic_messages"), {"provider": "custom"}, None,
                   diy_config(api_url="http://169.254.169.254/latest/meta-data"))
        with running_server() as (server, service):
            with mock.patch.object(RequestModelClient, "complete") as completion:
                for config in invalid:
                    with self.subTest(config=config):
                        status, _, body = request(server, "POST", "/api/chat", {
                            "message": "Hello", "model_config": config}, authorized())
                        self.assertEqual(status, 400, body)
                        self.assertNotIn(DIY_KEY, json.dumps(body))
                completion.assert_not_called()
                self.assertFalse(service.health()["completion_verified"])

    def test_local_model_urls_require_a_locally_bound_backend(self):
        with running_server() as (server, _):
            payload = {"message": "Hello", "model_config": diy_config(
                provider="custom", api_url="http://127.0.0.1:11434/v1/chat/completions")}
            with mock.patch("eagent.harness.request_client._request_opener",
                            return_value=lambda *a, **kw: ProviderResponse({"reasoning": "Local model answered."})):
                self.assertEqual(request(server, "POST", "/api/chat", payload, authorized())[0], 200)
                server.allow_local_api = False
                self.assertEqual(request(server, "POST", "/api/chat", payload, authorized())[0], 400)

    def test_health_distinguishes_configuration_from_verified_completion(self):
        with running_server() as (server, service):
            with mock.patch.object(OpenAIChatClient, "complete") as completion:
                status, _, body = request(server, "GET", "/api/health")
            self.assertEqual(status, 200)
            self.assertEqual(body["service"], "eagent")
            self.assertTrue(body["ready"])
            self.assertTrue(body["authentication_required"])
            self.assertFalse(body["completion_verified"])
            self.assertIn("kinetic_record", body["tools"])
            completion.assert_not_called()

    def test_unconfigured_provider_never_falls_back_or_reports_ready(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            service = AgentService(WebConfig(model="MiniMax-M2.7", project_root=ROOT))
            self.assertFalse(service.health()["ready"])
            status, body = service.chat("hello", [])
            self.assertEqual(status, 503)
            self.assertIn("MINIMAX_API_KEY", body["error"])
            self.assertFalse(service.health()["completion_verified"])
        with mock.patch.dict(os.environ, {"MINIMAX_API_KEY": KEY}, clear=True):
            service = AgentService(WebConfig(project_root=ROOT))
            self.assertFalse(service.health()["ready"])
            self.assertIn("EAGENT_MODEL", service.health()["reason"])

    def test_chat_uses_real_reference_tool_and_retains_followup_history(self):
        seen_messages = []

        def completion(client, system, messages, **kwargs):
            self.assertEqual(client.provider, "minimax")
            self.assertIn("Available tool interfaces:", system)
            self.assertIn("kinetic_record", system)
            seen_messages.append(messages[0]["content"])
            if len(messages) == 1:
                return json.dumps({"tool_calls": [{"interface": "kinetic_record",
                    "arguments": {"label_id": "PaHBDH_H150N_AAE_activity_only"}, "rationale": "Inspect the record."}]})
            result = json.loads(messages[-1]["content"].split("\n", 1)[1])[0]
            self.assertEqual(result["tool"], "kinetic_record")
            self.assertTrue(result["ok"])
            self.assertIsNone(result["value"]["km"])
            self.assertIn("bound", result["value"]["refusals"]["km"].lower())
            return json.dumps({"reasoning": "The loader withholds this Km because it is a bound."})

        with running_server() as (server, service):
            with mock.patch.object(OpenAIChatClient, "complete", autospec=True,
                                   side_effect=completion):
                status, _, body = request(server, "POST", "/api/chat", {
                    "message": "What about its Km?",
                    "history": [{"role": "user", "content": "Inspect PaHBDH_H150N."},
                                {"role": "assistant", "content": "I can inspect that record."}],
                }, authorized())
            self.assertEqual(status, 200, body)
            self.assertTrue(body["completion_verified"])
            self.assertTrue(service.health()["completion_verified"])
            self.assertEqual(body["transcript"]["tool_calls_made"], 1)
            self.assertIn("Previous conversation", seen_messages[0])
            self.assertIn("Inspect PaHBDH_H150N.", seen_messages[0])

    def test_authentication_and_origin_refusals_happen_before_model_call(self):
        with running_server() as (server, _):
            with mock.patch.object(OpenAIChatClient, "complete") as completion:
                self.assertEqual(request(server, "POST", "/api/chat",
                                         {"message": "hello"})[0], 401)
                status, headers, _ = request(server, "POST", "/api/chat", {"message": "hello"},
                    authorized(Origin="https://untrusted.example"))
                self.assertEqual(status, 403)
                self.assertNotIn("Access-Control-Allow-Origin", headers)
                completion.assert_not_called()

    def test_cors_preflight_and_same_origin_local_development(self):
        with running_server() as (server, _):
            status, headers, _ = request(server, "OPTIONS", "/api/chat", headers={
                "Origin": "https://psknlr.github.io",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            })
            self.assertEqual(status, 204)
            self.assertEqual(headers["Access-Control-Allow-Origin"], "https://psknlr.github.io")
            self.assertIn("Authorization", headers["Access-Control-Allow-Headers"])
            local_origin = f"http://127.0.0.1:{server.server_port}"
            status, headers, _ = request(server, "GET", "/api/health",
                                         headers={"Origin": local_origin})
            self.assertEqual(status, 200)
            self.assertEqual(headers["Access-Control-Allow-Origin"], local_origin)
            self.assertEqual(request(server, "GET", "/api/health", headers={
                "Origin": "http://127.0.0.1:9999"})[0], 403)

    def test_hosted_same_origin_https_requires_configured_authentication(self):
        headers = {"Origin": "https://eagent.example", "Host": "eagent.example"}
        with running_server() as (server, _):
            status, received, _ = request(server, "GET", "/api/health", headers=headers)
            self.assertEqual(status, 200)
            self.assertEqual(received["Access-Control-Allow-Origin"], "https://eagent.example")
            mismatched = {**headers, "Origin": "https://other.example"}
            self.assertEqual(request(server, "GET", "/api/health", headers=mismatched)[0], 403)
            insecure = {**headers, "Origin": "http://eagent.example"}
            self.assertEqual(request(server, "GET", "/api/health", headers=insecure)[0], 403)
        with running_server(token="") as (server, _):
            self.assertEqual(request(server, "GET", "/api/health", headers=headers)[0], 403)

    def test_public_binding_requires_access_token(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            service = AgentService(WebConfig(project_root=ROOT))
            with self.assertRaisesRegex(ValueError, "EAGENT_CHAT_TOKEN"):
                AgentHTTPServer(("0.0.0.0", 0), service)

    def test_provider_errors_remain_errors_and_scrub_credentials(self):
        with running_server() as (server, service):
            first_turn = json.dumps({"tool_calls": [{"interface": "reference_summary", "arguments": {}}]})
            with mock.patch.object(OpenAIChatClient, "complete", side_effect=[first_turn, LLMError(
                    f"Unknown model requested. debug {KEY} {TOKEN}")]):
                status, _, body = request(server, "POST", "/api/chat",
                                          {"message": "hello"}, authorized())
            self.assertEqual(status, 502)
            self.assertIn("Unknown model", body["error"])
            self.assertNotIn(KEY, json.dumps(body))
            self.assertNotIn(TOKEN, json.dumps(body))
            self.assertIn("provider_error", body["transcript"]["turns"][1])
            self.assertEqual(body["transcript"]["tool_calls_made"], 1)
            self.assertTrue(body["transcript"]["turns"][0]["results"][0]["ok"])
            self.assertFalse(service.health()["completion_verified"])

    def test_successful_trace_also_scrubs_keys_and_token(self):
        with running_server() as (server, _):
            with mock.patch.object(OpenAIChatClient, "complete", return_value=json.dumps({
                    "reasoning": f"An echoed credential {KEY} and token {TOKEN}."})):
                status, _, body = request(server, "POST", "/api/chat",
                    {"message": f"echo {KEY} {TOKEN}"}, authorized())
            self.assertEqual(status, 200)
            self.assertNotIn(KEY, json.dumps(body))
            self.assertNotIn(TOKEN, json.dumps(body))

    def test_numeric_guard_refuses_unsourced_answer(self):
        with running_server() as (server, service):
            with mock.patch.object(OpenAIChatClient, "complete",
                                   return_value='{"reasoning":"Its kcat is 30 s-1."}'):
                status, _, body = request(server, "POST", "/api/chat",
                                          {"message": "Its kcat?"}, authorized())
            self.assertEqual(status, 502)
            self.assertIn("numeric guard refused", body["error"])
            self.assertTrue(body["transcript"]["turns"][0]["guard"]["refused"])
            self.assertFalse(service.health()["completion_verified"])

    def test_questions_only_and_reasoning_with_questions_are_real_answers(self):
        responses = (
            ({"questions": ["Which enzyme do you want to inspect?", "Which substrate?"]},
             "Which enzyme do you want to inspect?\nWhich substrate?"),
            ({"reasoning": "The reference set can answer this.",
              "questions": ["Which enzyme do you want to inspect?"]},
             "The reference set can answer this.\n\nWhich enzyme do you want to inspect?"),
        )
        with running_server() as (server, _):
            for turn, expected in responses:
                with self.subTest(turn=turn):
                    fenced = "```json\n" + json.dumps(turn) + "\n```"
                    with mock.patch.object(OpenAIChatClient, "complete", return_value=fenced):
                        status, _, body = request(server, "POST", "/api/chat",
                                                  {"message": "Can you help?"}, authorized())
                    self.assertEqual(status, 200, body)
                    self.assertEqual(body["answer"], expected)
                    self.assertTrue(body["completion_verified"])
                    self.assertEqual(body["transcript"]["turns"][-1]["questions"], turn["questions"])

    def test_plain_text_and_invalid_json_cannot_masquerade_as_clarification(self):
        replies = ("Which enzyme?", '{"questions":', '["Which enzyme?"]',
                   '{"questions":"Which enzyme?"}', '{"reasoning":null}')
        with running_server() as (server, service):
            for raw in replies:
                with self.subTest(raw=raw):
                    with mock.patch.object(OpenAIChatClient, "complete", return_value=raw):
                        status, _, body = request(server, "POST", "/api/chat",
                                                  {"message": "Can you help?"}, authorized())
                    self.assertEqual(status, 502)
                    self.assertIn("provider_error", body["transcript"]["turns"][0])
                    self.assertFalse(service.health()["completion_verified"])

    def test_limit_and_empty_answer_failures_keep_full_actual_trace(self):
        with running_server() as (server, service):
            service.loop.limits = LoopLimits(max_turns=1)
            raw = json.dumps({"tool_calls": [{"interface": "reference_summary", "arguments": {}}]})
            with mock.patch.object(OpenAIChatClient, "complete", return_value=raw):
                status, _, body = request(server, "POST", "/api/chat",
                                          {"message": "Summarize the data."}, authorized())
            self.assertEqual(status, 502)
            self.assertIn("turn limit", body["error"])
            self.assertEqual(body["transcript"]["tool_calls_made"], 1)
            self.assertTrue(body["transcript"]["turns"][0]["results"][0]["ok"])
            with mock.patch.object(OpenAIChatClient, "complete", return_value='{"reasoning":""}'):
                status, _, body = request(server, "POST", "/api/chat",
                                          {"message": "Can you help?"}, authorized())
            self.assertEqual(status, 502)
            self.assertIn("no answer or clarification", body["error"])
            self.assertEqual(body["transcript"]["turns"][0]["reasoning"], "")
            self.assertFalse(service.health()["completion_verified"])

    def test_invalid_and_oversized_input_is_rejected_without_completion(self):
        with running_server() as (server, _):
            with mock.patch.object(OpenAIChatClient, "complete") as completion:
                for payload in ({"message": " "}, {"message": 4},
                                {"message": "hi", "history": [{"role": [], "content": "x"}]},
                                {"message": "hi", "provider": "openai"},
                                {"message": "hi", "history": [{"role": "system", "content": "x"}]}):
                    self.assertEqual(request(server, "POST", "/api/chat", payload,
                                             authorized())[0], 400)
                self.assertEqual(request(server, "POST", "/api/chat", {"message": "hi"},
                    authorized(**{"Content-Length": str(MAX_BODY_BYTES + 1)}))[0], 413)
                completion.assert_not_called()
        with self.assertRaises(ValueError):
            _validate_chat({"message": "hi", "history": [
                {"role": "user", "content": "x" * 8_000}] * 4})

    def test_model_concurrency_is_bounded(self):
        with running_server() as (server, service):
            service._slots.acquire()
            service._slots.acquire()
            try:
                with mock.patch.object(OpenAIChatClient, "complete") as completion:
                    status, _, body = request(server, "POST", "/api/chat",
                                              {"message": "hello"}, authorized())
                    self.assertEqual(status, 429)
                    self.assertIn("busy", body["error"])
                    completion.assert_not_called()
            finally:
                service._slots.release()
                service._slots.release()

    def test_static_server_restricts_files_to_web_directory(self):
        with running_server() as (server, service):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                web = root / "web"
                web.mkdir()
                (web / "index.html").write_text("<p>Agent chat</p>")
                secret = root / "server.env"
                secret.write_text(KEY)
                (web / "escape.env").symlink_to(secret)
                directory = web / "nested"
                directory.mkdir()
                (directory / "index.html").symlink_to(secret)
                service.static_root = web.resolve()
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
                connection.request("GET", "/")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn(b"Agent chat", response.read())
                connection.close()
                for path in ("/../server.env", "/%2e%2e/server.env", "/escape.env", "/nested/"):
                    self.assertEqual(request(server, "GET", path)[0], 404)


if __name__ == "__main__":
    unittest.main()
