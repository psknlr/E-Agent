"""Request-scoped model connections for the browser's editable provider setup.

No request credential enters the process environment or shared service state.
Endpoints are explicit and redirects are refused before credentials can move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import http.client
import ipaddress
import json
import re
import socket
import time
from typing import Any, Callable, Iterable
from urllib.parse import quote, quote_plus, urlsplit
import urllib.error
import urllib.request

from .llm import LLMClient, LLMError

PROVIDER_PROTOCOLS = {
    "minimax": "openai_chat", "minimax_cn": "openai_chat", "openai": "openai_chat",
    "anthropic": "anthropic_messages", "custom": None,
}
#: MiniMax's global and China platforms are one API on two hosts, so they share
#: every request and reply quirk below.
MINIMAX_PROVIDERS = frozenset({"minimax", "minimax_cn"})
PROTOCOLS = {"openai_chat", "anthropic_messages"}
MAX_RESPONSE_BYTES = 2_097_152


def redact_credentials(text: str, secrets: Iterable[str]) -> str:
    """Remove literal, JSON-escaped and URL-escaped credential representations."""
    variants: set[str] = set()
    for secret in secrets:
        if not isinstance(secret, str) or not secret:
            continue
        variants.update((secret, quote(secret, safe=""), quote_plus(secret, safe="")))
        escaped = secret
        # Error bodies may themselves contain JSON-serialized error details.
        for _ in range(4):
            escaped = json.dumps(escaped, ensure_ascii=False)[1:-1]
            variants.add(escaped)
    for variant in sorted(variants, key=len, reverse=True):
        text = text.replace(variant, "[credential redacted]")
    return text


def _safe_address(address: str, *, allow_local: bool, scheme: str) -> bool:
    ip = ipaddress.ip_address(address)
    # Treat IPv4-mapped IPv6 by its embedded IPv4 classification too.
    ip = getattr(ip, "ipv4_mapped", None) or ip
    return (ip.is_global and not ip.is_multicast and not ip.is_reserved and scheme == "https"
            or allow_local and ip.is_loopback)


def validate_api_url(url: str, *, allow_local: bool = False) -> tuple[str, ...]:
    """Allow public HTTPS; a locally bound backend may also reach loopback.

    The returned addresses are pinned for the actual socket connection; the
    transport retains the original hostname for Host and TLS SNI verification.
    """
    if not isinstance(url, str) or not url or len(url) > 2048:
        raise ValueError("api_url must be a full endpoint URL of at most 2048 characters.")
    if any(character.isspace() or ord(character) < 32 for character in url) or "\\" in url:
        raise ValueError("api_url must not contain whitespace or backslashes.")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("api_url is not a valid endpoint URL.") from exc
    if (parsed.scheme not in {"https", "http"} or not hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or not parsed.path or parsed.path == "/"):
        raise ValueError("api_url must be a full HTTPS API endpoint without credentials, query or fragment.")
    if parsed.scheme == "http" and not allow_local:
        raise ValueError("Public backends require an HTTPS API endpoint.")
    try:
        addresses = [str(ipaddress.ip_address(hostname))]
    except ValueError:
        try:
            addresses = list({item[4][0] for item in socket.getaddrinfo(
                hostname, port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM)})
        except OSError as exc:
            raise ValueError("The API hostname could not be resolved.") from exc
    if not addresses or any(not _safe_address(address, allow_local=allow_local,
                                               scheme=parsed.scheme) for address in addresses):
        raise ValueError("The API endpoint must resolve to public HTTPS or permitted local loopback.")
    return tuple(addresses)


@dataclass(frozen=True)
class RequestModelConfig:
    provider: str
    protocol: str
    api_key: str = field(repr=False)
    model: str
    api_url: str

    @classmethod
    def from_payload(cls, payload: Any, *, allow_local: bool = False) -> "RequestModelConfig":
        fields = {"provider", "protocol", "api_key", "model", "api_url"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("model_config requires provider, protocol, api_key, model and api_url.")
        if any(not isinstance(payload[name], str) or not payload[name].strip() for name in fields):
            raise ValueError("All model_config fields must be nonempty strings.")
        values = {name: payload[name].strip() for name in fields}
        provider, protocol = values["provider"], values["protocol"]
        if provider not in PROVIDER_PROTOCOLS:
            raise ValueError("provider must be minimax, minimax_cn, openai, anthropic or custom.")
        if protocol not in PROTOCOLS:
            raise ValueError("protocol must be openai_chat or anthropic_messages.")
        expected = PROVIDER_PROTOCOLS[provider]
        if expected and protocol != expected:
            raise ValueError("The selected provider and protocol do not match.")
        if (len(values["api_key"]) > 8192 or any(ord(c) < 32 or ord(c) > 126
                                                for c in values["api_key"])):
            raise ValueError("api_key must be a bounded printable ASCII credential.")
        if len(values["model"]) > 256 or any(ord(c) < 32 for c in values["model"]):
            raise ValueError("model must be an identifier of at most 256 characters.")
        if values["api_key"] in values["api_url"]:
            raise ValueError("Put API credentials in api_key, not api_url.")
        validate_api_url(values["api_url"], allow_local=allow_local)
        return cls(**values)

    def public_metadata(self) -> dict[str, str]:
        return {"provider": self.provider, "protocol": self.protocol,
                "model": self.model, "api_url": self.api_url}


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        raise LLMError("The API endpoint redirected; use its final URL in model settings.")


def _connect_addresses(addresses: tuple[str, ...], port: int, timeout: float,
                       source_address: Any = None) -> socket.socket:
    """Dial validated numeric IPs without a second DNS resolution."""
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    for address in addresses:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("The provider connection timed out.")
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        connection = socket.socket(family, socket.SOCK_STREAM)
        try:
            connection.settimeout(remaining)
            if source_address:
                connection.bind(source_address)
            destination = (address, port, 0, 0) if family == socket.AF_INET6 else (address, port)
            connection.connect(destination)
            return connection
        except OSError as exc:
            last_error = exc
            connection.close()
    raise last_error or OSError("No validated API address was available.")


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, addresses: tuple[str, ...]) -> None:
        super().__init__()
        self.addresses = addresses

    def http_open(self, request: Any) -> Any:
        def connection(host: str, **kwargs: Any) -> http.client.HTTPConnection:
            client = http.client.HTTPConnection(host, **kwargs)
            client._create_connection = lambda address, timeout, source_address=None: _connect_addresses(
                self.addresses, address[1], timeout, source_address)
            return client
        return self.do_open(connection, request)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, addresses: tuple[str, ...]) -> None:
        super().__init__()
        self.addresses = addresses

    def https_open(self, request: Any) -> Any:
        def connection(host: str, **kwargs: Any) -> http.client.HTTPSConnection:
            client = http.client.HTTPSConnection(host, **kwargs)
            client._create_connection = lambda address, timeout, source_address=None: _connect_addresses(
                self.addresses, address[1], timeout, source_address)
            return client
        # HTTPSConnection performs TLS verification with the original hostname.
        return self.do_open(connection, request, context=self._context)


def _request_opener(addresses: tuple[str, ...]) -> Callable[..., Any]:
    # Direct requests make the URL address policy independent of proxy settings.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _RejectRedirects(),
                                       _PinnedHTTPHandler(addresses), _PinnedHTTPSHandler(addresses)).open


class RequestModelClient(LLMClient):
    """One explicit credential, model, protocol and endpoint for one chat run."""

    runs_remotely = True

    def __init__(self, config: RequestModelConfig, *, allow_local: bool = False,
                 max_tokens: int = 8192, timeout_s: float = 60.0,
                 opener: Callable[..., Any] | None = None) -> None:
        self.config = config
        self.allow_local = allow_local
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self._opener = opener
        self.name = f"{config.provider}:{config.model}"

    def _redact(self, text: Any) -> str:
        serialized = json.dumps(text, ensure_ascii=False) if isinstance(text, (dict, list)) else str(text)
        return redact_credentials(serialized, (self.config.api_key,))

    def complete(self, system: str, messages: list[dict[str, str]],
                 tools: list[dict[str, Any]] | None = None,
                 temperature: float = 0.0, seed: int | None = None) -> str:
        config = self.config
        try:
            addresses = validate_api_url(config.api_url, allow_local=self.allow_local)
        except ValueError as exc:
            raise LLMError(self._redact(exc)) from exc
        body: dict[str, Any] = {"model": config.model,
                               "messages": [{"role": m["role"], "content": m["content"]}
                                            for m in messages]}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if config.protocol == "anthropic_messages":
            body.update(system=system, max_tokens=self.max_tokens)
            headers.update({"x-api-key": config.api_key, "anthropic-version": "2023-06-01"})
        else:
            body["messages"] = ([{"role": "system", "content": system}] if system else []) + body["messages"]
            headers["Authorization"] = "Bearer " + config.api_key
            # New GPT reasoning models reject temperature and seed; omit both.
            if config.provider == "openai" or config.provider in MINIMAX_PROVIDERS:
                body["max_completion_tokens"] = self.max_tokens
            else:
                body["max_tokens"] = self.max_tokens
            if config.provider in MINIMAX_PROVIDERS:
                body["reasoning_split"] = True
        request = urllib.request.Request(config.api_url, data=json.dumps(body).encode("utf-8"),
                                         method="POST", headers=headers)
        try:
            opener = self._opener or _request_opener(addresses)
            with opener(request, timeout=self.timeout_s) as response:
                status = int(getattr(response, "status", 200) or 200)
                if 300 <= status < 400:
                    raise LLMError("The API endpoint redirected; use its final URL in model settings.")
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise LLMError("The provider response exceeded the size limit.")
                payload = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                raise LLMError("The API endpoint redirected; use its final URL in model settings.") from exc
            try:
                detail = exc.read(MAX_RESPONSE_BYTES).decode("utf-8", "replace")
            except Exception:
                detail = str(exc.reason)
            # Redact before truncation so a long error cannot reveal a key prefix.
            raise LLMError(f"HTTP {exc.code} from {config.provider}: {self._redact(detail)[:500]}") from exc
        except LLMError as exc:
            raise LLMError(self._redact(exc)) from exc
        except Exception as exc:
            raise LLMError(f"The {config.provider} call failed: {self._redact(exc)}") from exc
        if not isinstance(payload, dict):
            raise LLMError("The provider returned an invalid response object.")
        base_resp = payload.get("base_resp")
        if isinstance(base_resp, dict) and base_resp.get("status_code"):
            raise LLMError(f"The provider returned an error: {self._redact(base_resp.get('status_msg', ''))}")
        if payload.get("error"):
            raise LLMError(f"The provider returned an error: {self._redact(payload['error'])}")
        if config.protocol == "anthropic_messages":
            if payload.get("stop_reason") == "max_tokens":
                raise LLMError("The provider reply reached its output token limit.")
            content = payload.get("content")
            if not isinstance(content, list):
                raise LLMError("The provider returned no text content blocks.")
            text = "".join(part["text"] for part in content if isinstance(part, dict)
                           and part.get("type") == "text" and isinstance(part.get("text"), str))
        else:
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise LLMError("The provider reply has no choices.")
            if choices[0].get("finish_reason") == "length":
                raise LLMError("The provider reply reached its output token limit.")
            message = choices[0].get("message")
            text = message.get("content") if isinstance(message, dict) else None
            if isinstance(text, list):
                text = "".join(part["text"] for part in text if isinstance(part, dict)
                               and isinstance(part.get("text"), str))
            if config.provider in MINIMAX_PROVIDERS and isinstance(text, str):
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                if "<think>" in text:
                    raise LLMError("The MiniMax reply holds no final message content.")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("The provider reply holds no final message content.")
        return text
