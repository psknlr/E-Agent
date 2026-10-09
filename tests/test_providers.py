"""Several providers behind one boundary, and the credential discipline.

What is defended:

* **a key never leaves the environment.** It is read at call time, is not on
  the object, and does not appear in an exception, a URL or a transcript. The
  tests inspect the objects and the error strings for it rather than trusting
  that no code path prints it;
* **an unconfigured provider refuses** and does not fall back to a configured
  one, because a run whose answer depends on which key happened to be set is
  not reproducible;
* **no silent retries**, per the boundary's contract;
* **the two wire formats are built correctly** -- Anthropic takes the system
  prompt as a field, the chat-completions format takes it as a leading message
  -- and the MiniMax hazard of a failure arriving inside a 200 is caught;
* **the honest state is recorded**: every provider's ``shape_verified`` is
  ``False``, because no credential existed where this was written and not one
  real completion has been requested.

Nothing here touches the network; the HTTP layer is a fake opener. The live
route probes are exercised against a fake too, and the real ones are a
``eagent model providers --probe --allow-network`` away.
"""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
from typing import Any
from unittest import mock

from eagent.harness.llm import AnthropicMessagesClient, LLMClient, LLMError
from eagent.harness.providers import (
    PROVIDERS, ROUTE_MARKERS, OpenAIChatClient, ProviderNotConfiguredError,
    build_client, configured_providers, probe_route, provider_report, redact,
)

KEY = "sk-test-0123456789abcdef"


class _Reply:
    """A urlopen context manager over one canned body."""

    def __init__(self, body: Any, status: int = 200) -> None:
        self._raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.status = status

    def __enter__(self) -> "_Reply":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._raw


def opener_for(reply: Any, sink: list[Any] | None = None):
    def opener(request, timeout=None):            # noqa: ANN001
        if sink is not None:
            sink.append(request)
        if isinstance(reply, Exception):
            raise reply
        return _Reply(reply)
    return opener


# ==========================================================================
class TheRegistryIsHonestAboutWhatIsEstablished(unittest.TestCase):
    def test_three_providers_over_two_wire_formats(self) -> None:
        self.assertEqual(sorted(PROVIDERS), ["anthropic", "minimax", "openai"])
        self.assertEqual({s.wire for s in PROVIDERS.values()},
                         {"anthropic_messages", "openai_chat"})

    def test_every_route_answers_and_no_request_shape_is_verified(self) -> None:
        """The honest state: the hosts are real, nothing has been authenticated."""
        for key, spec in PROVIDERS.items():
            self.assertTrue(spec.route_answers, key)
            self.assertFalse(spec.shape_verified, key)

    def test_the_report_says_what_each_flag_means(self) -> None:
        report = provider_report({})
        self.assertIn("authentication error", report["what_route_answers_means"])
        self.assertIn("no credential was available",
                      report["what_shape_verified_means"])
        self.assertEqual(report["configured"], [])

    def test_every_provider_declares_a_key_variable_and_a_url_override(self) -> None:
        for key, spec in PROVIDERS.items():
            self.assertTrue(spec.api_key_env.endswith("_API_KEY"), key)
            self.assertTrue(spec.base_url.startswith("https://"), key)
            self.assertTrue(spec.base_url_env.startswith("EAGENT_"), key)
            self.assertTrue(spec.documentation.startswith("https://"), key)

    def test_the_base_url_can_be_moved_without_a_code_change(self) -> None:
        spec = PROVIDERS["minimax"]
        self.assertEqual(
            spec.resolved_base_url({spec.base_url_env: "https://elsewhere/v1/"}),
            "https://elsewhere/v1")
        self.assertEqual(spec.resolved_base_url({}), spec.base_url)

    def test_configured_means_the_variable_is_set(self) -> None:
        self.assertEqual(configured_providers({}), [])
        self.assertEqual(configured_providers({"OPENAI_API_KEY": KEY}), ["openai"])
        self.assertEqual(configured_providers({"OPENAI_API_KEY": "   "}), [],
                         "a blank variable is not a credential")

    def test_the_report_carries_no_key_even_when_one_is_set(self) -> None:
        report = provider_report({"OPENAI_API_KEY": KEY})
        self.assertNotIn(KEY, json.dumps(report))
        self.assertTrue(report["providers"]["openai"]["key_present"])


class AnUnconfiguredProviderRefuses(unittest.TestCase):
    def test_building_a_client_without_the_key_refuses_by_name(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            for key, spec in PROVIDERS.items():
                with self.assertRaises(ProviderNotConfiguredError) as ctx:
                    build_client(key, "some-model")
                self.assertEqual(ctx.exception.api_key_env, spec.api_key_env)
                message = str(ctx.exception)
                self.assertIn(spec.api_key_env, message)
                self.assertIn("Nothing falls back to another provider", message)
                self.assertIn("not reproducible", message)

    def test_an_unknown_provider_is_refused_rather_than_guessed(self) -> None:
        with self.assertRaises(LLMError) as ctx:
            build_client("some-new-service", "m")
        self.assertIn("does not guess a URL", str(ctx.exception))

    def test_a_configured_provider_does_not_substitute_for_another(self) -> None:
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": KEY}, clear=True):
            self.assertIsInstance(build_client("openai", "gpt-4o"), OpenAIChatClient)
            with self.assertRaises(ProviderNotConfiguredError):
                build_client("minimax", "abab6.5s-chat")

    def test_calling_without_a_key_makes_no_request(self) -> None:
        sent: list[Any] = []
        client = OpenAIChatClient("gpt-4o", opener=opener_for({}, sent))
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ProviderNotConfiguredError):
                client.complete("sys", [{"role": "user", "content": "hi"}])
        self.assertEqual(sent, [], "no request may be built without a credential")

    def test_a_client_requires_a_model_identifier(self) -> None:
        for build in (lambda: OpenAIChatClient(""),
                      lambda: AnthropicMessagesClient("  ")):
            with self.assertRaisesRegex(ValueError, "model identifier"):
                build()


class TheKeyStaysInTheEnvironment(unittest.TestCase):
    def test_it_is_not_stored_on_the_client(self) -> None:
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": KEY}, clear=True):
            client = build_client("openai", "gpt-4o")
            self.assertNotIn(KEY, json.dumps(client.__dict__, default=str))
            self.assertEqual(client.api_key_env, "OPENAI_API_KEY")

    def test_it_is_sent_as_a_bearer_header_and_nowhere_else(self) -> None:
        sent: list[Any] = []
        client = OpenAIChatClient(
            "gpt-4o", opener=opener_for(
                {"choices": [{"message": {"content": "ok"}}]}, sent))
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": KEY}, clear=True):
            client.complete("sys", [{"role": "user", "content": "hi"}])
        request = sent[0]
        self.assertEqual(request.get_header("Authorization"), f"Bearer {KEY}")
        self.assertNotIn(KEY, request.full_url)
        self.assertNotIn(KEY, request.data.decode("utf-8"))

    def test_a_provider_error_body_echoing_the_key_is_redacted(self) -> None:
        leaky = urllib.error.HTTPError(
            "https://api.openai.com/v1/chat/completions", 401, "Unauthorized",
            {}, io.BytesIO(f'{{"error":"bad key {KEY}"}}'.encode()))
        client = OpenAIChatClient("gpt-4o", opener=opener_for(leaky))
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": KEY}, clear=True):
            with self.assertRaises(LLMError) as ctx:
                client.complete("s", [{"role": "user", "content": "x"}])
        message = str(ctx.exception)
        self.assertNotIn(KEY, message)
        self.assertIn("redacted", message)

    def test_redact_replaces_every_configured_key(self) -> None:
        env = {"OPENAI_API_KEY": KEY, "MINIMAX_API_KEY": "mm-abcdefgh"}
        text = f"saw {KEY} and mm-abcdefgh"
        out = redact(text, env)
        self.assertNotIn(KEY, out)
        self.assertNotIn("mm-abcdefgh", out)

    def test_redact_leaves_a_too_short_value_alone(self) -> None:
        """A three-character 'key' would turn every occurrence of it into noise."""
        self.assertEqual(redact("abc def", {"OPENAI_API_KEY": "abc"}), "abc def")


class TheTwoWireFormatsAreBuiltCorrectly(unittest.TestCase):
    def body_of(self, client: LLMClient, env: dict[str, str]) -> dict[str, Any]:
        sent: list[Any] = []
        client._opener = opener_for(
            {"choices": [{"message": {"content": "ok"}}],
             "content": [{"type": "text", "text": "ok"}]}, sent)
        with mock.patch.dict("os.environ", env, clear=True):
            client.complete("SYSTEM", [{"role": "user", "content": "hello"}],
                            temperature=0.3, seed=7)
        return json.loads(sent[0].data.decode("utf-8"))

    def test_chat_completions_puts_the_system_prompt_in_a_leading_message(self) -> None:
        body = self.body_of(OpenAIChatClient("gpt-4o"), {"OPENAI_API_KEY": KEY})
        self.assertEqual(body["messages"][0], {"role": "system", "content": "SYSTEM"})
        self.assertEqual(body["messages"][1], {"role": "user", "content": "hello"})
        self.assertNotIn("system", body)
        self.assertEqual(body["temperature"], 0.3)
        self.assertEqual(body["seed"], 7)

    def test_the_messages_api_puts_it_in_its_own_field(self) -> None:
        body = self.body_of(AnthropicMessagesClient("claude-x"),
                            {"ANTHROPIC_API_KEY": KEY})
        self.assertEqual(body["system"], "SYSTEM")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hello"}])
        self.assertNotIn("seed", body, "the Messages API has no seed parameter")

    def test_an_empty_system_prompt_adds_no_message(self) -> None:
        sent: list[Any] = []
        client = OpenAIChatClient("gpt-4o", opener=opener_for(
            {"choices": [{"message": {"content": "ok"}}]}, sent))
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": KEY}, clear=True):
            client.complete("", [{"role": "user", "content": "hi"}])
        body = json.loads(sent[0].data.decode("utf-8"))
        self.assertEqual([m["role"] for m in body["messages"]], ["user"])

    def test_each_provider_builds_its_documented_endpoint(self) -> None:
        self.assertEqual(OpenAIChatClient("m", provider="openai").endpoint,
                         "https://api.openai.com/v1/chat/completions")
        self.assertEqual(OpenAIChatClient("m", provider="minimax").endpoint,
                         "https://api.minimax.io/v1/chat/completions")

    def test_minimax_separates_thinking_and_uses_current_token_budget(self) -> None:
        body = self.body_of(OpenAIChatClient("MiniMax-M2.7", provider="minimax"),
                            {"MINIMAX_API_KEY": KEY})
        self.assertTrue(body["reasoning_split"])
        self.assertEqual(body["max_completion_tokens"], 1024)
        self.assertNotIn("max_tokens", body)

    def test_minimax_legacy_thinking_is_not_returned_as_final_content(self) -> None:
        client = OpenAIChatClient("MiniMax-M2.7", provider="minimax", opener=opener_for(
            {"choices": [{"message": {"content": '<think>private</think>{"reasoning":"ok"}'}}]}))
        with mock.patch.dict("os.environ", {"MINIMAX_API_KEY": KEY}, clear=True):
            self.assertEqual(client.complete("s", [{"role": "user", "content": "x"}]),
                             '{"reasoning":"ok"}')

    def test_a_provider_with_no_entry_and_no_url_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not guess a URL"):
            OpenAIChatClient("m", provider="unknown-service")

    def test_a_self_hosted_base_url_is_accepted(self) -> None:
        client = OpenAIChatClient("local-model", provider="openai",
                                  base_url="http://127.0.0.1:8000/v1")
        self.assertEqual(client.endpoint, "http://127.0.0.1:8000/v1/chat/completions")

    def test_the_reply_is_read_from_the_message_content(self) -> None:
        client = OpenAIChatClient("gpt-4o", opener=opener_for(
            {"choices": [{"message": {"content": "the answer"}}]}))
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": KEY}, clear=True):
            self.assertEqual(
                client.complete("s", [{"role": "user", "content": "x"}]),
                "the answer")

    def test_content_delivered_as_parts_is_joined(self) -> None:
        client = OpenAIChatClient("gpt-4o", opener=opener_for(
            {"choices": [{"message": {"content": [{"text": "two "},
                                                  {"text": "parts"}]}}]}))
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": KEY}, clear=True):
            self.assertEqual(
                client.complete("s", [{"role": "user", "content": "x"}]),
                "two parts")


class AFailureIsRaisedNotRetriedAndNotParsedAsAnAnswer(unittest.TestCase):
    def call(self, reply: Any, provider: str = "openai",
             env: dict[str, str] | None = None) -> str:
        sent: list[Any] = []
        client = OpenAIChatClient("m", provider=provider,
                                  opener=opener_for(reply, sent))
        self.sent = sent
        with mock.patch.dict("os.environ",
                             env or {PROVIDERS[provider].api_key_env: KEY},
                             clear=True):
            return client.complete("s", [{"role": "user", "content": "x"}])

    def test_a_minimax_failure_inside_a_200_is_raised(self) -> None:
        """The hazard a live probe showed: MiniMax answers 200 with an error."""
        with self.assertRaises(LLMError) as ctx:
            self.call({"base_resp": {"status_code": 1004,
                                     "status_msg": "login fail"}},
                      provider="minimax")
        self.assertIn("1004", str(ctx.exception))
        self.assertIn("login fail", str(ctx.exception))

    def test_a_minimax_success_carries_status_code_zero(self) -> None:
        self.assertEqual(
            self.call({"base_resp": {"status_code": 0, "status_msg": ""},
                       "choices": [{"message": {"content": "fine"}}]},
                      provider="minimax"), "fine")

    def test_a_reply_with_no_choices_is_an_error_not_an_empty_answer(self) -> None:
        with self.assertRaisesRegex(LLMError, "no choices"):
            self.call({"id": "x"})

    def test_a_reply_with_empty_content_is_an_error(self) -> None:
        with self.assertRaisesRegex(LLMError, "no message content"):
            self.call({"choices": [{"message": {"content": "   "}}]})

    def test_a_truncated_reply_is_not_presented_as_a_complete_answer(self) -> None:
        with self.assertRaisesRegex(LLMError, "token limit"):
            self.call({"choices": [{"finish_reason": "length",
                                    "message": {"content": "partial answer"}}]},
                      provider="minimax")

    def test_a_transport_failure_is_raised_once_with_no_retry(self) -> None:
        attempts: list[Any] = []

        def opener(request, timeout=None):        # noqa: ANN001
            attempts.append(request)
            raise urllib.error.URLError("connection reset")

        client = OpenAIChatClient("m", opener=opener)
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": KEY}, clear=True):
            with self.assertRaisesRegex(LLMError, "call failed"):
                client.complete("s", [{"role": "user", "content": "x"}])
        self.assertEqual(len(attempts), 1, "implementations must not retry silently")

    def test_a_non_json_body_is_an_error(self) -> None:
        with self.assertRaises(LLMError):
            self.call(b"<html>gateway timeout</html>")


class TheRouteProbeChecksForTheProvidersOwnError(unittest.TestCase):
    def test_a_body_carrying_the_markers_passes(self) -> None:
        body = io.BytesIO(json.dumps(
            {"type": "error",
             "error": {"type": "authentication_error",
                       "message": "x-api-key header is required"}}).encode())
        failure = urllib.error.HTTPError("u", 401, "Unauthorized", {}, body)
        result = probe_route("anthropic", opener=opener_for(failure))
        self.assertTrue(result["ok"])
        self.assertEqual(sorted(result["markers_found"]),
                         sorted(ROUTE_MARKERS["anthropic"]))
        self.assertIn("NOT that the request body", result["establishes"])

    def test_the_current_minimax_authentication_error_is_recognized(self) -> None:
        result = probe_route("minimax", opener=opener_for(
            {"error": {"type": "authorized_error",
                       "message": "carry the key in Authorization"}}))
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], 200)

    def test_a_friendly_error_page_does_not_pass(self) -> None:
        body = io.BytesIO(b"<html><body>Service Unavailable</body></html>")
        failure = urllib.error.HTTPError("u", 503, "x", {}, body)
        result = probe_route("openai", opener=opener_for(failure))
        self.assertFalse(result["ok"])
        self.assertEqual(sorted(result["markers_missing"]),
                         sorted(ROUTE_MARKERS["openai"]))
        self.assertIn("it is not this provider's API", result["failure"])

    def test_a_request_that_does_not_complete_establishes_nothing(self) -> None:
        result = probe_route("openai", opener=opener_for(
            urllib.error.URLError("no route to host")))
        self.assertFalse(result["ok"])
        self.assertEqual(result["establishes"], "nothing")
        self.assertIsNone(result["status"])

    def test_the_probe_needs_no_credential(self) -> None:
        sent: list[Any] = []
        body = io.BytesIO(b'{"error": {"type": "authorized_error", '
                          b'"message": "Authorization"}}')
        failure = urllib.error.HTTPError("u", 401, "x", {}, body)
        with mock.patch.dict("os.environ", {}, clear=True):
            probe_route("minimax", opener=opener_for(failure, sent))
        self.assertIsNone(sent[0].get_header("Authorization"))

    def test_an_unknown_provider_cannot_be_probed(self) -> None:
        with self.assertRaises(LLMError):
            probe_route("nope")


if __name__ == "__main__":
    unittest.main()
