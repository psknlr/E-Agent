"""The registry, the argument validation and the result envelope.

No science lives in this file. It is the part a reviewer should be able to read
in one sitting and be satisfied that a model cannot get anything out of it
except a bounded call to a fixed function.

THE RULES A CALL IS HELD TO
===========================
* The agent must exist. There is no dynamic dispatch on a model-supplied name
  beyond a dictionary lookup, and an unknown name is refused, not guessed.
* The arguments must be a JSON object of strings, every key declared by the
  agent, every value within the agent's own length bound. Numbers are strings
  here because that is the one type a model produces predictably, and because
  each agent parses its own with a strict pattern rather than ``float()``'s
  generous one (``float("1_0")`` is ``10.0``; ``float("nan")`` is not a number a
  result should ever carry).
* The result's value must be finite and small. A non-finite number is refused
  rather than serialised as ``NaN``, which no JSON parser agrees about, and an
  oversized value is refused because the page truncates what it feeds a model
  and a silently truncated result is a number the model never saw.

THE DIGEST
==========
``cite.sha256`` is the SHA-256 of the *canonical* JSON of the value: sorted
keys, no whitespace, no ASCII escaping. The page registers it so the numeric
guard can check a model's quoted number against the field it cites. It is a
provenance hash, not a security boundary: it ties a citation to a specific
result, so a number cannot be cited against a result that did not contain it.
Elapsed time is deliberately outside the hash, so two runs of the same agent on
the same input have the same row id.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..errors import EAgentError

__all__ = [
    "AGENTS", "ARTIFACT", "ENGINE_SCHEMA", "MAX_ARG_CHARS", "MAX_RESULT_BYTES",
    "AgentRefusal", "AgentSpec", "call", "canonical", "catalog", "choice_arg",
    "digest", "integer_arg", "number_arg", "register", "sequence_arg",
    "static_catalog", "text_arg",
]

ENGINE_SCHEMA = "eagent.engine.v1"

#: The artifact key in a ``cite``. The page's numeric guard treats rows under
#: this key as computed in this browser, like its calculator's.
ARTIFACT = "engine_compute"

#: Default bound on one argument. An agent may declare a different one.
MAX_ARG_CHARS = 4000

#: A result is refused above this many bytes of canonical JSON. The page feeds
#: a model at most 8000 bytes of any one result and says so when it clips; an
#: agent that could exceed that is built to stay below it instead.
MAX_RESULT_BYTES = 7000

# ASCII only, on purpose. Python's \d matches the digits of every script, and
# float() and int() accept them, so without re.ASCII the Arabic-Indic digit "٣"
# is a "plain" number here. A parser that is strict about format should be
# strict about alphabet too.
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)
_INTEGER = re.compile(r"\d{1,12}", re.ASCII)
_SEQUENCE = re.compile(r"[A-Z]+", re.ASCII)


class AgentRefusal(EAgentError):
    """An agent declines to produce a value, and says why.

    A refusal is a result, not a failure of the engine: "the sequences are
    different lengths, so a Hamming distance is undefined" is the correct answer
    to that question. It reaches the model as a refusal to report, never as a
    number.
    """


@dataclass(frozen=True)
class AgentSpec:
    """One agent: what it is called, what it takes, and which project code it runs."""

    name: str
    description: str
    #: Parameter name -> help text. Optional parameters say so in their help.
    parameters: Mapping[str, str]
    run: Callable[[dict[str, str]], dict[str, Any]]
    #: The project function(s) this wraps, for the result's provenance. An agent
    #: that wraps nothing says so rather than leaving it blank.
    wraps: str
    required: tuple[str, ...] = ()
    #: Per-parameter length bounds that differ from :data:`MAX_ARG_CHARS`.
    max_chars: Mapping[str, int] = field(default_factory=dict)

    def schema(self) -> dict[str, Any]:
        """The shape the controller model is shown, matching the reference tools'."""
        return {"name": self.name, "description": self.description,
                "parameters": dict(self.parameters)}

    def check(self, arguments: Any) -> str:
        """Why these arguments are refused, or ``""`` if they are acceptable."""
        if not isinstance(arguments, dict):
            return "arguments must be a JSON object"
        unknown = sorted(set(arguments) - set(self.parameters))
        if unknown:
            return f"unsupported argument(s): {', '.join(unknown)}"
        for key, value in arguments.items():
            if not isinstance(value, str):
                return f"argument {key} must be a string"
            limit = self.max_chars.get(key, MAX_ARG_CHARS)
            if len(value) > limit:
                return f"argument {key} has {len(value)} characters; the limit is {limit}"
        missing = [key for key in self.required if not arguments.get(key, "").strip()]
        if missing:
            return f"missing required argument(s): {', '.join(missing)}"
        return ""


#: Every agent, by name. Populated by importing the modules that define them.
AGENTS: dict[str, AgentSpec] = {}


def register(spec: AgentSpec) -> AgentSpec:
    """Add an agent. A duplicate name is a programming error, never a replacement."""
    if spec.name in AGENTS:
        raise ValueError(f"agent {spec.name!r} is already registered")
    AGENTS[spec.name] = spec
    return spec


# ---------------------------------------------------------------------------
# argument parsing: strict, and refusing with a sentence a person can act on
# ---------------------------------------------------------------------------

def text_arg(args: Mapping[str, str], name: str, *, required: bool = True) -> str | None:
    raw = args.get(name)
    if raw is None or not raw.strip():
        if required:
            raise AgentRefusal(f"{name} is required")
        return None
    return raw


def sequence_arg(args: Mapping[str, str], name: str, *, maximum: int) -> str:
    """A residue sequence: whitespace removed, upper-cased, letters only."""
    raw = text_arg(args, name)
    cleaned = "".join(raw.split())
    # Checked before upper-casing: "ı" (dotless i) upper-cases to a plain "I", and
    # "ſ" to "S", which would silently turn a non-sequence into a valid one.
    if not cleaned.isascii() or not _SEQUENCE.fullmatch(cleaned.upper()):
        raise AgentRefusal(f"{name} may contain only one-letter residue codes (ASCII letters)")
    sequence = cleaned.upper()
    if len(sequence) > maximum:
        raise AgentRefusal(f"{name} has {len(sequence)} residues; this agent accepts at "
                           f"most {maximum}")
    return sequence


def number_arg(args: Mapping[str, str], name: str, default: float | None, *,
               minimum: float, maximum: float) -> float:
    """A finite decimal number within bounds. ``default=None`` makes it required."""
    raw = args.get(name)
    if raw is None or not raw.strip():
        if default is None:
            raise AgentRefusal(f"{name} is required")
        return default
    text = raw.strip()
    if not _NUMBER.fullmatch(text):
        raise AgentRefusal(f"{name} must be a plain decimal number, got {raw!r}")
    value = float(text)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise AgentRefusal(f"{name} must be between {minimum:g} and {maximum:g}, got {text}")
    return value


def integer_arg(args: Mapping[str, str], name: str, default: int | None, *,
                minimum: int, maximum: int) -> int:
    """A non-negative whole number within bounds. ``default=None`` makes it required."""
    raw = args.get(name)
    if raw is None or not raw.strip():
        if default is None:
            raise AgentRefusal(f"{name} is required")
        return default
    text = raw.strip()
    if not _INTEGER.fullmatch(text):
        raise AgentRefusal(f"{name} must be a whole number of digits, got {raw!r}")
    value = int(text)
    if not minimum <= value <= maximum:
        raise AgentRefusal(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


def choice_arg(args: Mapping[str, str], name: str, choices: tuple[str, ...],
               default: str | None) -> str:
    """One of a fixed set, matched case-insensitively. ``default=None`` makes it required."""
    raw = args.get(name)
    if raw is None or not raw.strip():
        if default is None:
            raise AgentRefusal(f"{name} is required; choose one of {', '.join(choices)}")
        return default
    wanted = raw.strip().lower()
    for choice in choices:
        if choice.lower() == wanted:
            return choice
    raise AgentRefusal(f"{name} must be one of {', '.join(choices)}, got {raw.strip()!r}")


# ---------------------------------------------------------------------------
# the envelope
# ---------------------------------------------------------------------------

def canonical(value: Any) -> str:
    """The one serialisation the digest is taken over."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _fail(name: str, kind: str, message: str) -> str:
    return json.dumps({"ok": False, "agent": name, "kind": kind, "refusal": message},
                      ensure_ascii=False)


def call(name: str, arguments_json: str = "{}") -> str:
    """Run one agent. Always returns a JSON string, and never raises.

    ``{"ok": true, "agent", "value", "cite", "wraps", "elapsed_ms"}`` on success;
    ``{"ok": false, "agent", "kind", "refusal"}`` otherwise, where ``kind`` is
    ``unknown_agent``, ``invalid`` (the arguments were refused before anything
    ran), ``refused`` (the agent ran and declined to produce a value) or
    ``internal`` (a bug; reported, never turned into a number).
    """
    started = time.perf_counter()
    spec = AGENTS.get(name) if isinstance(name, str) else None
    if spec is None:
        label = name if isinstance(name, str) else type(name).__name__
        return _fail(label[:80], "unknown_agent",
                     f"no agent named {label[:80]!r}; the agents are {', '.join(sorted(AGENTS))}")
    try:
        arguments = json.loads(arguments_json) if arguments_json else {}
    except (TypeError, ValueError, RecursionError):
        # RecursionError is what a JSON array nested a hundred thousand deep raises.
        return _fail(name, "invalid", "arguments must be a JSON object")
    problem = spec.check(arguments)
    if problem:
        return _fail(name, "invalid", problem)
    try:
        value = spec.run(arguments)
        text = canonical(value)
    except AgentRefusal as exc:
        return _fail(name, "refused", str(exc))
    except (EAgentError, ValueError) as exc:
        # The project's own typed refusals, and its ValueErrors, are answers.
        return _fail(name, "refused", str(exc)[:600])
    except Exception as exc:  # noqa: BLE001 -- a bug is reported, never turned into a number
        return _fail(name, "internal", f"{type(exc).__name__}: {str(exc)[:200]}")
    if len(text.encode("utf-8")) > MAX_RESULT_BYTES:
        return _fail(name, "internal", f"the result is larger than {MAX_RESULT_BYTES} bytes, so it is "
                                       f"refused rather than clipped")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return json.dumps({
        "ok": True, "agent": name, "value": value,
        "cite": {"artifact": ARTIFACT, "sha256": sha, "row": f"{name}-{sha[:16]}",
                 "method": "read"},
        "wraps": spec.wraps,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }, ensure_ascii=False, allow_nan=False)


def catalog() -> str:
    """The agents' schemas as JSON, for the page to offer a controller model."""
    return json.dumps({"schema": ENGINE_SCHEMA,
                       "agents": [AGENTS[name].schema() for name in sorted(AGENTS)],
                       "limits": {"max_arg_chars": MAX_ARG_CHARS,
                                  "max_result_bytes": MAX_RESULT_BYTES}},
                      ensure_ascii=False)


def static_catalog() -> dict[str, Any]:
    """Everything the page shows without loading the engine, generated from code.

    The agents' schemas, and the ten pipeline interfaces with where each can run.
    Written to ``web/agent-catalog.json`` by ``scripts/build_browser_bundle.py``
    and checked for staleness in CI, the same way the evidence bundle is, so the
    page cannot describe a capability the code does not have.
    """
    from .pipeline import pipeline_table, shipped_tasks
    return {
        "schema": ENGINE_SCHEMA,
        "artifact": ARTIFACT,
        "limits": {"max_arg_chars": MAX_ARG_CHARS, "max_result_bytes": MAX_RESULT_BYTES},
        "agents": [{**AGENTS[name].schema(), "wraps": AGENTS[name].wraps,
                    "max_chars": dict(AGENTS[name].max_chars)} for name in sorted(AGENTS)],
        "pipeline": pipeline_table(),
        "tasks": shipped_tasks(),
    }
