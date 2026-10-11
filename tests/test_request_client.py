"""DIY client wire contracts, URL isolation and credential-safe failures."""

from __future__ import annotations

import io
import json
import socket
import unittest
from unittest import mock
import urllib.error
import urllib.request

from eagent.harness.llm import LLMError
from eagent.harness.request_client import (
    RequestModelClient, RequestModelConfig, _PinnedHTTPSHandler, _RejectRedirects,
    validate_api_url,
)

KEY = "request-only-provider-key-secret"
URL = "https://93.184.216.34/v1/chat/completions"


def configuration(**changes):
    values = {"provider": "openai", "protocol": "openai_chat", "api_key": KEY,
              "model": "gpt-example", "api_url": URL}
    values.update(changes)
    return RequestModelConfig.from_payload(values)


class Response(io.BytesIO):
    status = 200

    def __init__(self, payload):
        super().__init__(json.dumps(payload).encode())


class RequestClientTests(unittest.TestCase):
    def test_each_provider_uses_the_selected_full_url_key_and_wire_format(self):
        variants = (
            ("openai", "openai_chat", {"choices": [{"message": {"content": "answer"}}]}),
            ("minimax", "openai_chat", {"choices": [{"message": {"content": "<think>private</think>answer"}}]}),
            ("minimax_cn", "openai_chat", {"choices": [{"message": {"content": "<think>private</think>answer"}}]}),
            ("custom", "openai_chat", {"choices": [{"message": {"content": [{"type": "text", "text": "answer"}]}}]}),
            ("anthropic", "anthropic_messages", {"content": [{"type": "thinking", "thinking": "private"},
                                                                       {"type": "text", "text": "answer"}]}),
            ("custom", "anthropic_messages", {"content": [{"type": "text", "text": "answer"}]}),
        )
        for provider, protocol, reply in variants:
            with self.subTest(provider=provider, protocol=protocol):
                seen = []

                def opener(request, **kwargs):
                    seen.append(request)
                    self.assertEqual(kwargs["timeout"], 60.0)
                    return Response(reply)

                client = RequestModelClient(configuration(provider=provider, protocol=protocol), opener=opener)
                self.assertEqual(client.complete("System instructions", [{"role": "user", "content": "Hello"}],
                                                 temperature=0.8, seed=4), "answer")
                request = seen[0]
                self.assertEqual(request.full_url, URL)
                self.assertEqual(request.method, "POST")
                body = json.loads(request.data)
                self.assertEqual(body["model"], "gpt-example")
                self.assertNotIn("api_key", body)
                self.assertNotIn("temperature", body)
                self.assertNotIn("seed", body)
                headers = dict(request.header_items())
                if protocol == "anthropic_messages":
                    self.assertEqual(headers["X-api-key"], KEY)
                    self.assertEqual(headers["Anthropic-version"], "2023-06-01")
                    self.assertNotIn("Authorization", headers)
                    self.assertEqual(body["system"], "System instructions")
                    self.assertEqual(body["messages"], [{"role": "user", "content": "Hello"}])
                    self.assertEqual(body["max_tokens"], 8192)
                else:
                    self.assertEqual(headers["Authorization"], "Bearer " + KEY)
                    self.assertEqual(body["messages"][0]["role"], "system")
                    token_field = "max_tokens" if provider == "custom" else "max_completion_tokens"
                    self.assertEqual(body[token_field], 8192)
                    self.assertEqual(body.get("reasoning_split"), True if provider.startswith("minimax") else None)

    def test_configuration_requires_every_field_and_compatible_protocol(self):
        original = configuration().public_metadata() | {"api_key": KEY}
        invalid = []
        for name in original:
            invalid.append({key: value for key, value in original.items() if key != name})
            invalid.append({**original, name: ""})
            invalid.append({**original, name: []})
        invalid.extend(({**original, "provider": "unknown"},
                        {**original, "protocol": "other"},
                        {**original, "protocol": "anthropic_messages"},
                        {**original, "extra": "field"},
                        {**original, "api_key": "key\nHeader: injection"},
                        {**original, "api_url": URL + "?api_key=" + KEY}))
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    RequestModelConfig.from_payload(payload)
        config = configuration()
        self.assertNotIn(KEY, repr(config))
        self.assertNotIn("api_key", config.public_metadata())

    def test_endpoint_policy_refuses_private_hosts_even_with_local_permission(self):
        for url in ("https://127.0.0.1/v1/chat/completions", "https://10.0.0.1/v1/chat/completions",
                    "http://169.254.169.254/latest/meta-data", "https://[::ffff:127.0.0.1]/v1/messages",
                    "http://93.184.216.34/v1/chat/completions", "https://user:password@example.com/v1/messages",
                    "https://224.0.0.1/v1/messages", "https://93.184.216.34/",
                    URL + "#fragment", URL + "?key=credential", URL + "\\alternate"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_api_url(url)
        for url in ("http://169.254.169.254/latest/meta-data", "https://10.0.0.1/v1/messages",
                    "http://93.184.216.34/v1/messages"):
            with self.assertRaises(ValueError):
                validate_api_url(url, allow_local=True)
        validate_api_url("http://127.0.0.1:11434/v1/chat/completions", allow_local=True)
        validate_api_url("https://[::1]/v1/messages", allow_local=True)
        validate_api_url(URL)

    def test_dns_is_checked_before_each_call_and_all_resolved_addresses_must_be_safe(self):
        def address(ip):
            return (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 443))

        hostname_url = "https://models.example/v1/chat/completions"
        with mock.patch("eagent.harness.request_client.socket.getaddrinfo",
                        side_effect=[[address("93.184.216.34")], [address("10.0.0.1")]]):
            config = configuration(api_url=hostname_url)
            opener = mock.Mock()
            with self.assertRaisesRegex(LLMError, "public HTTPS"):
                RequestModelClient(config, opener=opener).complete("", [])
            opener.assert_not_called()
        with mock.patch("eagent.harness.request_client.socket.getaddrinfo",
                        return_value=[address("93.184.216.34"), address("127.0.0.1")]):
            with self.assertRaises(ValueError):
                validate_api_url(hostname_url)

    def test_provider_and_network_errors_scrub_request_key_before_truncation(self):
        errors = (
            urllib.error.URLError("failed: " + KEY),
            urllib.error.HTTPError(URL, 401, "Unauthorized", {}, io.BytesIO(("x" * 495 + KEY).encode())),
            RuntimeError("unexpected transport debug: " + KEY),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                def opener(request, **kwargs):
                    raise error

                with self.assertRaises(LLMError) as raised:
                    RequestModelClient(configuration(), opener=opener).complete("", [])
                self.assertNotIn(KEY, str(raised.exception))
                self.assertNotIn(KEY[:5], str(raised.exception))
        for payload in ({"base_resp": {"status_code": 1004, "status_msg": KEY}},
                        {"error": {"message": KEY}}):
            with self.assertRaises(LLMError) as raised:
                RequestModelClient(configuration(), opener=lambda *a, **kw: Response(payload)).complete("", [])
            self.assertNotIn(KEY, str(raised.exception))

    def test_json_escaped_credentials_are_removed_from_provider_error_bodies(self):
        key = 'fake-quote"slash\\credential'
        error = urllib.error.HTTPError(URL, 401, "Unauthorized", {},
                                       io.BytesIO(json.dumps({"error": key}).encode()))
        with self.assertRaises(LLMError) as raised:
            RequestModelClient(configuration(api_key=key),
                               opener=mock.Mock(side_effect=error)).complete("", [])
        detail = str(raised.exception).split(": ", 1)[1]
        self.assertEqual(json.loads(detail)["error"], "[credential redacted]")
        key = 'fake-apostrophe\'quote"slash\\credential'
        payload = {"error": {"message": key, "details": [key]}}
        with self.assertRaises(LLMError) as raised:
            RequestModelClient(configuration(api_key=key),
                               opener=lambda *a, **kw: Response(payload)).complete("", [])
        detail = str(raised.exception).split(": ", 1)[1]
        self.assertEqual(json.loads(detail), {"message": "[credential redacted]",
                                             "details": ["[credential redacted]"]})

    def test_https_connection_pins_public_address_and_retains_original_tls_hostname(self):
        handler = _PinnedHTTPSHandler(("93.184.216.34",))
        transport_socket = mock.Mock()
        tls_context = mock.Mock()
        provider_request = urllib.request.Request(
            "https://models.example:8443/v1/chat/completions")

        def inspect_connection(factory, request, **kwargs):
            connection = factory(request.host, timeout=5.0, context=tls_context)
            self.assertEqual(connection.host, "models.example")
            connection.connect()
            return "connected"

        with mock.patch.object(handler, "do_open", side_effect=inspect_connection), \
                mock.patch("eagent.harness.request_client.socket.socket", return_value=transport_socket), \
                mock.patch("eagent.harness.request_client.socket.getaddrinfo",
                           return_value=[(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.1", 8443))]) as dns:
            self.assertEqual(handler.https_open(provider_request), "connected")
            dns.assert_not_called()
        transport_socket.connect.assert_called_once_with(("93.184.216.34", 8443))
        tls_context.wrap_socket.assert_called_once_with(transport_socket, server_hostname="models.example")

    def test_redirects_cannot_forward_the_key_or_masquerade_as_completion(self):
        with self.assertRaisesRegex(LLMError, "redirected"):
            _RejectRedirects().redirect_request(None, None, 302, "Found", {}, "https://other.example")
        reply = Response({"choices": [{"message": {"content": "answer"}}]})
        reply.status = 302
        with self.assertRaisesRegex(LLMError, "redirected"):
            RequestModelClient(configuration(), opener=lambda *a, **kw: reply).complete("", [])
        error = urllib.error.HTTPError(URL, 307, "Temporary Redirect", {"Location": "https://other.example"}, None)
        with self.assertRaisesRegex(LLMError, "redirected"):
            RequestModelClient(configuration(), opener=mock.Mock(side_effect=error)).complete("", [])

    def test_empty_invalid_and_truncated_responses_are_failures(self):
        for payload in ([], {"choices": []}, {"choices": [None]},
                        {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]},
                        {"choices": [{"message": {"content": ""}}]}):
            with self.subTest(payload=payload):
                with self.assertRaises(LLMError):
                    RequestModelClient(configuration(), opener=lambda *a, **kw: Response(payload)).complete("", [])
        with self.assertRaisesRegex(LLMError, "token limit"):
            RequestModelClient(configuration(provider="anthropic", protocol="anthropic_messages"),
                opener=lambda *a, **kw: Response({"stop_reason": "max_tokens", "content": []})).complete("", [])


if __name__ == "__main__":
    unittest.main()
