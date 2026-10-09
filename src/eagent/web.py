"""HTTP chat transport for the actual, read-only E-Agent reference console.

GitHub Pages serves the browser interface. This process keeps provider keys on
the server and runs the same loaders, tool loop and numeric guard as the CLI.
Readiness means configured; it does not claim a model has answered a request.
"""

from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import mimetypes
import os
from pathlib import Path
import re
import socket
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlsplit

from .harness.llm import LLMClient, LLMError, NumericGuard
from .harness.providers import PROVIDERS, build_client
from .harness.toolloop import LoopLimits, SYSTEM_PROMPT, ToolLoop, reference_tools

MAX_BODY_BYTES = 65_536
MAX_MESSAGE_CHARS = 8_000
MAX_HISTORY_MESSAGES = 20
MAX_HISTORY_CHARS = 24_000


def _project_root() -> Path:
    """Locate repository data independently of the process working directory."""
    configured = os.environ.get("EAGENT_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for location in (Path(__file__).resolve(), Path.cwd().resolve()):
        candidates = (location, *location.parents) if location.is_dir() else location.parents
        for candidate in candidates:
            if (candidate / "configs" / "references" / "kred_calibration").is_dir():
                return candidate
    return Path(__file__).resolve().parents[2]


def _loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _redact(value: Any, token: str = "") -> Any:
    """Scrub keys and the access token from every outgoing string, including traces."""
    secrets = {token} if token else set()
    secrets.update((os.environ.get(spec.api_key_env) or "").strip()
                   for spec in PROVIDERS.values())
    secrets.discard("")
    if isinstance(value, str):
        for secret in sorted(secrets, key=len, reverse=True):
            value = value.replace(secret, "[credential redacted]")
        return value
    if isinstance(value, dict):
        return {_redact(key, token): _redact(item, token)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, token) for item in value]
    return value


@dataclass(frozen=True)
class WebConfig:
    provider: str = "minimax"
    model: str = ""
    token: str = ""
    allowed_origins: tuple[str, ...] = ("https://psknlr.github.io",)
    project_root: Path | None = None

    @classmethod
    def from_environment(cls) -> "WebConfig":
        origins = tuple(origin.strip().rstrip("/") for origin in
                        os.environ.get("EAGENT_ALLOWED_ORIGINS",
                                       "https://psknlr.github.io").split(",")
                        if origin.strip())
        for origin in origins:
            parsed = urlsplit(origin)
            if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                    or parsed.username or parsed.password or parsed.path
                    or parsed.query or parsed.fragment):
                raise ValueError("EAGENT_ALLOWED_ORIGINS must contain exact HTTP origins")
        return cls(provider=os.environ.get("EAGENT_PROVIDER", "minimax").strip(),
                   model=os.environ.get("EAGENT_MODEL", "").strip(),
                   token=os.environ.get("EAGENT_CHAT_TOKEN", "").strip(),
                   allowed_origins=origins, project_root=_project_root())


class _JSONProtocolClient(LLMClient):
    """Validate provider turns before the core parser's plain-text fallback.

    The console parser preserves malformed output as a question for an
    operator. A web chat must distinguish that fallback from a model's genuine
    structured request for clarification. This adapter preserves valid raw
    replies and uses the loop's existing provider-error trace for invalid ones.
    """

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        self.name = getattr(client, "name", "unknown")
        self.runs_remotely = getattr(client, "runs_remotely", True)

    def complete(self, system: str, messages: list[dict[str, str]],
                 tools: list[dict[str, Any]] | None = None,
                 temperature: float = 0.0, seed: int | None = None) -> str:
        try:
            raw = self.client.complete(system, messages, tools=tools,
                                       temperature=temperature, seed=seed)
        except LLMError:
            raise
        except Exception as exc:
            # Keep unexpected provider response-shape failures inside the
            # existing loop transcript, including any preceding real tools.
            raise LLMError(f"The provider request failed: {exc}") from exc
        if not isinstance(raw, str):
            raise LLMError("The provider returned no text in the agent's JSON protocol.")
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        try:
            turn = json.loads(text)
        except (ValueError, RecursionError) as exc:
            raise LLMError("The provider returned malformed JSON in the agent protocol.") from exc
        fields = {"reasoning", "questions", "tool_calls"}
        if not isinstance(turn, dict) or not set(turn).intersection(fields):
            raise LLMError("The provider returned no recognized agent JSON fields.")
        if set(turn) - fields:
            raise LLMError("The provider returned unsupported agent JSON fields.")
        if "reasoning" in turn and not isinstance(turn["reasoning"], str):
            raise LLMError("The provider's agent reasoning must be a string.")
        questions = turn.get("questions", [])
        if not isinstance(questions, list) or any(not isinstance(q, str) for q in questions):
            raise LLMError("The provider's agent questions must be an array of strings.")
        calls = turn.get("tool_calls", [])
        if not isinstance(calls, list) or any(
                not isinstance(call, dict) or not isinstance(call.get("interface"), str)
                or not call["interface"] or not isinstance(call.get("arguments", {}), dict)
                or not isinstance(call.get("rationale", ""), str) for call in calls):
            raise LLMError("The provider returned an invalid agent tool call.")
        return raw


class AgentService:
    """One configured provider and the repository's actual registered readers."""

    def __init__(self, config: WebConfig) -> None:
        self.config = config
        self.project_root = config.project_root or _project_root()
        self.static_root = (self.project_root / "web").resolve()
        self._slots = threading.BoundedSemaphore(2)
        self._verified_lock = threading.Lock()
        self.completion_verified = False
        self.loop: ToolLoop | None = None
        self.tools: list[str] = []
        self._reason = ""
        try:
            tools = reference_tools(self.project_root / "configs" / "references"
                                    / "kred_calibration" / "v0.1")
            self.tools = sorted(tool.name for tool in tools)
            if not config.provider:
                self._reason = "Set EAGENT_PROVIDER to select a provider."
                return
            if not config.model:
                self._reason = "Set EAGENT_MODEL to an explicit provider model identifier."
                return
            client = build_client(config.provider, config.model,
                                  max_tokens=8192, timeout_s=60.0)
            loop = ToolLoop(_JSONProtocolClient(client), guard=NumericGuard(strict=True),
                            limits=LoopLimits(max_wall_seconds=180.0))
            loop.register_all(tools)
            # Providers return the harness JSON protocol; their HTTP clients
            # do not translate these schemas into native provider function calls.
            loop.system = SYSTEM_PROMPT + "\nAvailable tool interfaces:\n" + json.dumps(
                loop.schemas(), ensure_ascii=False)
            self.loop = loop
        except Exception as exc:  # configuration or loader failure: never pretend ready
            self._reason = str(_redact(f"Runtime initialization failed: {exc}", config.token))

    def health(self) -> dict[str, Any]:
        spec = PROVIDERS.get(self.config.provider)
        ready = self.loop is not None and spec is not None and spec.has_key()
        reason = self._reason
        if self.loop is not None and spec is not None and not spec.has_key():
            reason = f"Set {spec.api_key_env} on the server; no model request was made."
        if ready:
            reason = ("The runtime is configured. A provider completion has been verified."
                      if self.completion_verified else
                      "The runtime is configured; a provider completion has not yet been verified.")
        return {"service": "eagent", "ready": ready,
                "provider": self.config.provider, "model": self.config.model,
                "tools": self.tools, "reason": reason,
                "completion_verified": self.completion_verified,
                "authentication_required": bool(self.config.token)}

    def chat(self, message: str, history: list[dict[str, str]]) -> tuple[int, dict[str, Any]]:
        if not self.health()["ready"] or self.loop is None:
            return 503, {"error": self.health()["reason"]}
        if not self._slots.acquire(blocking=False):
            return 429, {"error": "The agent is busy. Try again when an active chat finishes."}
        try:
            question = message
            if history:
                question = ("Previous conversation (context only; tool results must still "
                            "support factual answers):\n" +
                            json.dumps(history, ensure_ascii=False) +
                            "\nCurrent user message:\n" + message)
            transcript = self.loop.run(question)

            def failure(error: str) -> tuple[int, dict[str, Any]]:
                return 502, {"error": error, "transcript": transcript,
                             "completion_verified": self.completion_verified}

            provider_error = next((turn["provider_error"] for turn in
                                   transcript["turns"] if "provider_error" in turn), None)
            if provider_error:
                return failure(f"The provider request failed: {provider_error}")
            if transcript["stopped_because"] != "the model answered without asking for another tool":
                return failure(transcript["stopped_because"])
            final_turn = transcript["turns"][-1] if transcript["turns"] else {}
            reasoning = final_turn.get("reasoning", "").strip()
            questions = "\n".join(q.strip() for q in final_turn.get("questions", []) if q.strip())
            answer = "\n\n".join(part for part in (reasoning, questions) if part)
            if not answer:
                return failure("The provider returned no answer or clarification in the agent's JSON protocol.")
            with self._verified_lock:
                self.completion_verified = True
            return 200, {"answer": answer, "transcript": transcript,
                         "completion_verified": True}
        except Exception as exc:
            # The provider's own error is useful for invalid model ids. Scrub
            # the entire response rather than logging request data or secrets.
            return 502, {"error": f"The agent request failed: {exc}"}
        finally:
            self._slots.release()


def _validate_chat(payload: Any) -> tuple[str, list[dict[str, str]]]:
    if not isinstance(payload, dict) or set(payload) - {"message", "history"}:
        raise ValueError("Expected a JSON object with message and optional history.")
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a nonempty string.")
    if len(message) > MAX_MESSAGE_CHARS:
        raise ValueError(f"message exceeds {MAX_MESSAGE_CHARS} characters.")
    history = payload.get("history", [])
    if not isinstance(history, list) or len(history) > MAX_HISTORY_MESSAGES:
        raise ValueError(f"history must contain at most {MAX_HISTORY_MESSAGES} messages.")
    total = 0
    for item in history:
        if (not isinstance(item, dict) or set(item) != {"role", "content"}
                or item.get("role") not in ("user", "assistant")
                or not isinstance(item.get("content"), str)
                or len(item["content"]) > MAX_MESSAGE_CHARS):
            raise ValueError("history entries require user/assistant role and bounded string content.")
        total += len(item["content"])
    if total > MAX_HISTORY_CHARS:
        raise ValueError(f"history exceeds {MAX_HISTORY_CHARS} characters in total.")
    return message.strip(), history


class AgentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, address: tuple[str, int], service: AgentService) -> None:
        if not _loopback(address[0]) and not service.config.token:
            raise ValueError("EAGENT_CHAT_TOKEN is required when binding beyond localhost.")
        self.service = service
        super().__init__(address, AgentRequestHandler)


class AgentRequestHandler(BaseHTTPRequestHandler):
    server: AgentHTTPServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format: str, *args: Any) -> None:
        # BaseHTTPRequestHandler logs request paths, which may contain user
        # content or accidental credentials. This transport records no requests.
        return

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        if origin in self.server.service.config.allowed_origins:
            return True
        try:
            parsed = urlsplit(origin)
            host = urlsplit("http://" + self.headers.get("Host", ""))
            valid_origin = (not parsed.username and not parsed.password
                            and not parsed.path and not parsed.query and not parsed.fragment
                            and bool(parsed.hostname) and parsed.netloc == host.netloc)
            local = (parsed.scheme == "http" and _loopback(parsed.hostname or "")
                     and (parsed.port or 80) == self.server.server_port)
            hosted = parsed.scheme == "https" and bool(self.server.service.config.token)
            return valid_origin and (local or hosted)
        except ValueError:
            return False

    def _headers(self, status: int, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Vary", "Origin")
        origin = self.headers.get("Origin")
        if origin and self._origin_allowed():
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        clean = _redact(payload, self.server.service.config.token)
        body = json.dumps(clean, ensure_ascii=False).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _allow_origin(self) -> bool:
        if self._origin_allowed():
            return True
        self._json(403, {"error": "This origin is not allowed to access the agent."})
        return False

    def do_OPTIONS(self) -> None:
        if not self._allow_origin():
            return
        if urlsplit(self.path).path not in {"/api/health", "/api/chat"}:
            self._json(404, {"error": "Unknown API route."})
            return
        self._headers(204, "application/json", 0)

    def do_GET(self) -> None:
        if not self._allow_origin():
            return
        path = urlsplit(self.path).path
        if path == "/api/health":
            self._json(200, self.server.service.health())
            return
        if path.startswith("/api/"):
            self._json(404, {"error": "Unknown API route."})
            return
        relative = unquote(path).lstrip("/") or "index.html"
        root = self.server.service.static_root
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            self._json(404, {"error": "File not found."})
            return
        if target.is_dir():
            target = (target / "index.html").resolve()
            try:
                target.relative_to(root)
            except ValueError:
                self._json(404, {"error": "File not found."})
                return
        if any(part.startswith(".") for part in Path(relative).parts) or not target.is_file():
            self._json(404, {"error": "File not found."})
            return
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        body = target.read_bytes()
        self._headers(200, content_type, len(body))
        self.wfile.write(body)

    def do_POST(self) -> None:
        if not self._allow_origin():
            return
        if urlsplit(self.path).path != "/api/chat":
            self._json(404, {"error": "Unknown API route."})
            return
        token = self.server.service.config.token
        authorization = self.headers.get("Authorization", "").split()
        if token and (len(authorization) != 2 or authorization[0].lower() != "bearer"
                      or not hmac.compare_digest(authorization[1].encode(), token.encode())):
            self._json(401, {"error": "A valid agent access token is required."})
            return
        if self.headers.get("Transfer-Encoding"):
            self._json(400, {"error": "Chunked requests are not supported."})
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1:
            self._json(411, {"error": "A single Content-Length header is required."})
            return
        try:
            length = int(lengths[0])
        except ValueError:
            self._json(400, {"error": "Content-Length must be an integer."})
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self._json(413, {"error": f"Request body must be at most {MAX_BODY_BYTES} bytes."})
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].lower() != "application/json":
            self._json(415, {"error": "Content-Type must be application/json."})
            return
        try:
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Request body was incomplete.")
            message, history = _validate_chat(json.loads(body.decode("utf-8")))
        except (ValueError, UnicodeDecodeError, socket.timeout) as exc:
            self._json(400, {"error": f"Invalid chat request: {exc}"})
            return
        status, payload = self.server.service.chat(message, history)
        self._json(status, payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve E-Agent's real reference chat runtime.")
    parser.add_argument("--host", default=os.environ.get(
        "EAGENT_HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"))
    parser.add_argument("--port", type=int, default=os.environ.get("PORT", "8787"))
    args = parser.parse_args(argv)
    try:
        service = AgentService(WebConfig.from_environment())
        server = AgentHTTPServer((args.host, args.port), service)
    except (ValueError, OSError) as exc:
        parser.error(str(_redact(str(exc), os.environ.get("EAGENT_CHAT_TOKEN", ""))))
    print(f"E-Agent chat listening on {args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
