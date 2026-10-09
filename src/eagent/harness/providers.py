"""Several model providers behind the one boundary, and what each is worth.

:mod:`eagent.harness.llm` defines the boundary: a client implements
:meth:`~eagent.harness.llm.LLMClient.complete` and everything else -- the
numeric guard, the restricted turn shape, the planner's refusal to send a
sequence to a remote model -- applies the same whichever provider answered.
This module adds the providers and, more importantly, keeps straight what has
actually been established about each one.

TWO DIFFERENT CLAIMS, AS ELSEWHERE IN THIS PROJECT
==================================================
The connector layer already separates *a route answers* from *this request
shape works* (see
:class:`eagent.connectors.chemistry.RequestShapeNotVerifiedError`). The same
split applies here, and the honest state is the same:

* **The route answers.** Every provider below was sent one unauthenticated
  request from this container on 2026-10-09, and every one replied with **its
  own** structured authentication error -- Anthropic's
  ``{"type":"error","error":{"type":"authentication_error"}}``, OpenAI's
  ``error.message`` naming Bearer auth, MiniMax's
  ``base_resp.status_code 1004``. A CDN error page and a login wall do not
  produce those, so the host, the path and the request parsing are the real API.
  :func:`probe_route` is that check, and :data:`ROUTE_MARKERS` is what it looks
  for.
* **The request shape is unverified.** Whether the body this module sends is
  accepted, and whether the reply is parsed correctly, cannot be checked
  without a credential, and **no provider credential exists in the environment
  this was written in**. Each client's shape is written from the provider's
  public API reference and driven in tests through a fake opener. Not one real
  completion has been requested from any of them.

So :attr:`ProviderSpec.shape_verified` is ``False`` everywhere, and it is a
field rather than a sentence in a docstring so that a run can record it.

WHAT IS SHARED AND WHAT IS NOT
==============================
Two wire formats cover the providers here. Anthropic's Messages API takes the
system prompt as its own top-level field; the OpenAI chat-completions format
takes it as a first message with role ``system``, and a long tail of services
speak that format at their own base URL. :class:`OpenAIChatClient` is therefore
one client with a configurable base URL rather than a class per vendor, because
a class per vendor would be four copies of one request with four chances to fix
a bug in three places.

RULES THAT DO NOT CHANGE PER PROVIDER
=====================================
* **The key is read from the environment at call time.** It is never stored on
  the object, never written to an audit file, never put in an exception
  message, and never logged. :func:`redact` is used on anything derived from a
  request before it is recorded.
* **A provider with no key refuses.** It does not fall back to another
  provider, and it does not fall back to a local stub. A silent fallback would
  make a run's results depend on which keys happened to be set.
* **No retries.** The boundary's contract is that implementations must not
  retry silently; a failure is raised with the provider's own message.
* **Which provider and model answered is recorded**, because "the model said"
  is not reproducible unless the model is named.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .llm import AnthropicMessagesClient, LLMClient, LLMError

__all__ = [
    "ProviderSpec",
    "PROVIDERS",
    "ROUTE_MARKERS",
    "OpenAIChatClient",
    "redact",
    "build_client",
    "configured_providers",
    "provider_report",
    "probe_route",
    "ProviderNotConfiguredError",
]


class ProviderNotConfiguredError(LLMError):
    """No credential for the provider that was asked for.

    Its own class so that a caller can tell "you have not configured this" from
    "the provider answered with an error". The first is fixed by setting an
    environment variable; the second is a fact about the service.
    """

    def __init__(self, provider: str, api_key_env: str) -> None:
        self.provider = provider
        self.api_key_env = api_key_env
        super().__init__(
            f"provider {provider!r} needs {api_key_env} in the environment and "
            f"it is not set, so no request was made. Nothing falls back to "
            f"another provider: a run whose answer depends on which key "
            f"happened to be set is not reproducible.")


@dataclass(frozen=True)
class ProviderSpec:
    """One provider: how to reach it, and what is known about doing so."""

    key: str
    display_name: str
    wire: str                       # "anthropic_messages" | "openai_chat"
    api_key_env: str
    base_url: str
    #: Environment variable that overrides :attr:`base_url`. A provider that
    #: moves a path, or a self-hosted service speaking the same wire format, is
    #: then a configuration change rather than a code change.
    base_url_env: str
    documentation: str
    #: Whether one unauthenticated request from this container got this
    #: provider's own structured auth error back. Says the host and path are
    #: real; says nothing about the body.
    route_answers: bool
    #: Whether a real completion has been requested and parsed. ``False``
    #: everywhere: no credential was available where this was written.
    shape_verified: bool = False
    #: Model identifiers the provider's own documentation listed. Examples for
    #: a ``--help``, not an endorsement and not a default: every client
    #: requires the model to be named.
    documented_models: tuple[str, ...] = ()
    notes: str = ""

    def resolved_base_url(self, env: Mapping[str, str] | None = None) -> str:
        source = os.environ if env is None else env
        return (source.get(self.base_url_env) or self.base_url).rstrip("/")

    def has_key(self, env: Mapping[str, str] | None = None) -> bool:
        source = os.environ if env is None else env
        return bool((source.get(self.api_key_env) or "").strip())

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "display_name": self.display_name,
                "wire": self.wire, "api_key_env": self.api_key_env,
                "base_url": self.base_url, "base_url_env": self.base_url_env,
                "documentation": self.documentation,
                "route_answers": self.route_answers,
                "shape_verified": self.shape_verified,
                "documented_models": list(self.documented_models),
                "notes": self.notes}


#: Strings that only the provider's own error body carries. A 401 with one of
#: these in it proves the real API answered; a CDN page or a login wall has
#: none of them. Used by :func:`probe_route`, in the spirit of
#: :mod:`eagent.datalayer.probe`: a status code is not a check.
ROUTE_MARKERS: Mapping[str, tuple[str, ...]] = {
    "anthropic": ("authentication_error", "x-api-key"),
    "openai": ("Authorization", "Bearer"),
    "minimax": ("base_resp", "Authorization"),
}


PROVIDERS: Mapping[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        key="anthropic", display_name="Anthropic Messages API",
        wire="anthropic_messages", api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com",
        base_url_env="EAGENT_ANTHROPIC_BASE_URL",
        documentation="https://docs.anthropic.com/en/api/messages",
        route_answers=True,
        documented_models=("claude-opus-4-20250514", "claude-sonnet-4-20250514"),
        notes=("The system prompt is a top-level field, not a message. There is "
               "no seed parameter, so seed is accepted and ignored and the "
               "planner stores the response itself. Model identifiers move "
               "faster than this file; pass the one you mean.")),
    "openai": ProviderSpec(
        key="openai", display_name="OpenAI Chat Completions",
        wire="openai_chat", api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        base_url_env="EAGENT_OPENAI_BASE_URL",
        documentation="https://platform.openai.com/docs/api-reference/chat",
        route_answers=True,
        documented_models=("gpt-4o", "gpt-4o-mini"),
        notes=("The system prompt is the first message with role 'system'. "
               "Accepts seed, which this client passes through; the provider "
               "documents it as best-effort, so it is not determinism.")),
    "minimax": ProviderSpec(
        key="minimax", display_name="MiniMax (OpenAI-compatible endpoint)",
        wire="openai_chat", api_key_env="MINIMAX_API_KEY",
        base_url="https://api.minimax.chat/v1",
        base_url_env="EAGENT_MINIMAX_BASE_URL",
        documentation="https://platform.minimaxi.com/document",
        route_answers=True,
        documented_models=("abab6.5s-chat",),
        notes=("Speaks the chat-completions format at "
               "text/chatcompletion_v2 rather than chat/completions, which is "
               "why the path is part of the client's configuration. Both "
               "api.minimax.chat and api.minimaxi.com answered; set "
               "EAGENT_MINIMAX_BASE_URL to pick one. Error bodies use "
               "base_resp.status_code rather than an HTTP status, so a failure "
               "can arrive inside a 200 -- this client checks for it.")),
}

#: Path appended to a provider's base URL for the chat-completions call.
#: MiniMax uses its own, so the path travels with the provider rather than
#: being assumed from the wire format.
_CHAT_PATHS: Mapping[str, str] = {
    "openai": "/chat/completions",
    "minimax": "/text/chatcompletion_v2",
}


def redact(text: str, env: Mapping[str, str] | None = None) -> str:
    """Replace any configured API key found in ``text`` with a marker.

    Belt and braces. No code path here puts a key into a string, but anything
    that records a request, an error body or a URL goes through this first,
    because a credential in an audit file is permanent and the cost of the
    check is nothing.
    """
    source = os.environ if env is None else env
    out = text
    for spec in PROVIDERS.values():
        value = (source.get(spec.api_key_env) or "").strip()
        if value and len(value) >= 8:
            out = out.replace(value, f"<{spec.api_key_env} redacted>")
    return out


class OpenAIChatClient(LLMClient):
    """The chat-completions wire format, at a configurable base URL.

    One client for every service that speaks it: OpenAI itself, MiniMax's
    compatible endpoint, and anything self-hosted. The differences that matter
    are the base URL, the path and the model identifier, and all three are
    configuration.

    NOT RUN AGAINST ANY LIVE SERVICE. The body (``model``, ``messages`` with
    the system prompt as a leading ``system`` message, ``temperature``,
    ``max_tokens``, optional ``seed``) and the reply path
    (``choices[0].message.content``) are written from the published reference.
    The route answers -- see the module docstring -- and the shape does not
    have a credential behind it.

    One provider-specific hazard is handled: MiniMax returns failures inside a
    200 response as ``base_resp.status_code``, so a non-zero status there is
    raised rather than parsed as an empty completion.
    """

    name = "openai-chat"
    runs_remotely = True

    def __init__(self, model: str, *, provider: str = "openai",
                 max_tokens: int = 1024, timeout_s: float = 60.0,
                 base_url: str | None = None, path: str | None = None,
                 api_key_env: str | None = None,
                 opener: Callable[..., Any] | None = None) -> None:
        if not model or not model.strip():
            raise ValueError("a model identifier is required; there is no default")
        spec = PROVIDERS.get(provider)
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.api_key_env = api_key_env or (spec.api_key_env if spec else "OPENAI_API_KEY")
        base = base_url or (spec.resolved_base_url() if spec else None)
        if not base:
            raise ValueError(
                f"provider {provider!r} is not in PROVIDERS and no base_url was "
                f"given; this client does not guess a URL")
        self.base_url = base.rstrip("/")
        self.path = path or _CHAT_PATHS.get(provider, "/chat/completions")
        self._opener = opener or urllib.request.urlopen
        self.name = f"{provider}:{model}"

    @property
    def endpoint(self) -> str:
        return self.base_url + self.path

    def complete(self, system: str, messages: list[dict[str, str]],
                 tools: list[dict[str, Any]] | None = None,
                 temperature: float = 0.0, seed: int | None = None) -> str:
        key = (os.environ.get(self.api_key_env) or "").strip()
        if not key:
            raise ProviderNotConfiguredError(self.provider, self.api_key_env)
        body: dict[str, Any] = {
            "model": self.model, "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": ([{"role": "system", "content": system}] if system else [])
                        + [{"role": m["role"], "content": m["content"]}
                           for m in messages],
        }
        if seed is not None:
            body["seed"] = seed
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {key}",
                     "content-type": "application/json"})
        try:
            with self._opener(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:                                    # noqa: BLE001
                pass
            raise LLMError(
                f"HTTP {exc.code} from {self.provider}: "
                f"{redact(detail or str(exc.reason))}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LLMError(f"the {self.provider} call failed: {redact(str(exc))}") from exc

        # MiniMax reports failures inside a 200. Checked before the reply is
        # read, so a refusal is not parsed as an empty completion.
        base_resp = payload.get("base_resp")
        if isinstance(base_resp, Mapping) and base_resp.get("status_code"):
            raise LLMError(
                f"{self.provider} returned status_code "
                f"{base_resp.get('status_code')}: "
                f"{redact(str(base_resp.get('status_msg', '')))}")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError(f"the {self.provider} reply has no choices")
        message = (choices[0] or {}).get("message") or {}
        text = message.get("content")
        if isinstance(text, list):          # some services return content parts
            text = "".join(part.get("text", "") for part in text
                           if isinstance(part, Mapping))
        if not isinstance(text, str) or not text.strip():
            raise LLMError(f"the {self.provider} reply holds no message content")
        return text


def build_client(provider: str, model: str, **kwargs: Any) -> LLMClient:
    """A client for one provider, or a refusal naming what is missing.

    Refuses an unknown provider rather than guessing a base URL, and refuses a
    configured provider with no key rather than falling back to another one.
    """
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise LLMError(
            f"{provider!r} is not a known provider; known: "
            f"{', '.join(sorted(PROVIDERS))}. A provider needs an entry with "
            f"its wire format, key variable and documented base URL -- this "
            f"function does not guess a URL from a name.")
    if not spec.has_key():
        raise ProviderNotConfiguredError(spec.key, spec.api_key_env)
    if spec.wire == "anthropic_messages":
        endpoint = spec.resolved_base_url() + "/v1/messages"
        return AnthropicMessagesClient(
            model, api_key_env=spec.api_key_env, endpoint=endpoint, **kwargs)
    if spec.wire == "openai_chat":
        return OpenAIChatClient(model, provider=spec.key, **kwargs)
    raise LLMError(f"provider {provider!r} declares wire format {spec.wire!r}, "
                   f"which this module has no client for")


def configured_providers(env: Mapping[str, str] | None = None) -> list[str]:
    """Providers whose key variable is set. Order is :data:`PROVIDERS`'."""
    return [k for k, spec in PROVIDERS.items() if spec.has_key(env)]


def provider_report(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """What is configured and what each provider is worth, for a run record.

    Carries no key and no fragment of one: only whether the variable is set.
    """
    return {
        "providers": {
            key: {**spec.to_dict(),
                  "key_present": spec.has_key(env),
                  "base_url_in_effect": spec.resolved_base_url(env)}
            for key, spec in PROVIDERS.items()},
        "configured": configured_providers(env),
        "what_route_answers_means": (
            "one unauthenticated request from the container this was written "
            "in came back as this provider's own structured authentication "
            "error, so the host, the path and the request parsing are the real "
            "API rather than a CDN page or a login wall"),
        "what_shape_verified_means": (
            "a real completion was requested and parsed. False for every "
            "provider here: no credential was available, so every request "
            "shape is written from the published reference and exercised only "
            "against a fake opener"),
    }


def probe_route(provider: str, *,
                opener: Callable[..., Any] | None = None,
                timeout_s: float = 20.0) -> dict[str, Any]:
    """Send one unauthenticated request and check the provider's own error back.

    What a pass establishes: the host resolves, the path exists, and the
    service parsed the request enough to complain about the credential. What it
    does not establish: that the body this module sends is accepted, or that
    the reply is parsed correctly. Both need a key.

    Deliberately not run on import and not part of any run: it is a check a
    person asks for.
    """
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise LLMError(f"{provider!r} is not a known provider")
    url = spec.resolved_base_url() + (
        "/v1/messages" if spec.wire == "anthropic_messages"
        else _CHAT_PATHS.get(spec.key, "/chat/completions"))
    request = urllib.request.Request(
        url, data=b"{}", method="POST", headers={"content-type": "application/json"})
    open_it = opener or urllib.request.urlopen
    status: int | None = None
    body = ""
    try:
        with open_it(request, timeout=timeout_s) as response:
            status = int(getattr(response, "status", 200) or 200)
            body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:                                        # noqa: BLE001
            body = ""
    except (urllib.error.URLError, OSError) as exc:
        return {"provider": provider, "url": url, "ok": False, "status": None,
                "failure": f"the request did not complete: {redact(str(exc))}",
                "markers_found": [], "establishes": "nothing"}
    markers = ROUTE_MARKERS.get(spec.key, ())
    found = [m for m in markers if m in body]
    missing = [m for m in markers if m not in body]
    return {
        "provider": provider, "url": url, "status": status,
        "ok": not missing,
        "markers_found": found, "markers_missing": missing,
        "failure": "" if not missing else (
            f"the service answered HTTP {status} but the body does not contain "
            f"{missing}; something replied and it is not this provider's API"),
        "establishes": (
            "the host, the path and the request parsing are this provider's "
            "own API. NOT that the request body this module sends is accepted, "
            "and NOT that the reply is parsed correctly -- both need a key."),
    }
