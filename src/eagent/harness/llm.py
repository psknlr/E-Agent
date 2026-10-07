"""The language-model boundary.

The controller uses a model for four things only: understanding the task,
organising retrieval, proposing testable hypotheses, and interpreting evidence
into a plan. Everything measurable -- alignments, coordinates, atom mappings,
distances, model inference, statistics -- is computed by deterministic code or
by a dedicated scientific model.

That split is enforced here rather than requested in a prompt. :class:`NumericGuard`
inspects model output and rejects quantities that are presented as results
without an artifact to back them, because the characteristic failure of a
research agent is not refusing to answer: it is answering with a number that
reads like a measurement and was never measured.
"""

from __future__ import annotations

import abc
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..errors import EAgentError, FabricationGuardError
from . import citation as citation_mod
from .citation import ArtifactIndex, Citation


# Units whose appearance in model prose implies a measurement was made.
MEASUREMENT_PATTERN = re.compile(
    r"""(?<![\w.])
    (-?\d+(?:\.\d+)?)
    \s*
    (
        %|percent|
        angstrom|angstroms|A\b|nm|
        kcal/mol|kJ/mol|
        s-1|s\^-1|/s|
        mM|uM|nM|M\b|
        pLDDT|ee\b|
        kcat|Km|
        degrees?|deg\b
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

#: The old citation form: a bracketed file name and nothing else. Recognised
#: only so the guard can say what is wrong with it. It names no row, no field,
#: no version and no method, so there is nothing to check a number against.
LEGACY_CITATION_PATTERN = re.compile(
    r"\[(?:artifact|table|file|record|evidence):[^\]]+\]", re.IGNORECASE
)


@dataclass
class GuardReport:
    """What the guard found in one model response."""

    uncited_quantities: list[str] = field(default_factory=list)
    cited_quantities: list[str] = field(default_factory=list)
    #: Quantities whose citation was checked against the cell and matched.
    verified_quantities: list[str] = field(default_factory=list)
    #: Quantities whose citation is well formed but whose value this guard
    #: cannot check -- a derivation it was not given the arithmetic for.
    declared_quantities: list[str] = field(default_factory=list)
    #: Citations that are malformed, name an artifact the run never produced,
    #: cite the wrong version of it, or contradict the cell they point at.
    broken_citations: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.uncited_quantities or self.broken_citations)

    @property
    def fully_verified(self) -> bool:
        return self.clean and not self.declared_quantities

    def summary(self) -> str:
        return (f"{len(self.verified_quantities)} verified, "
                f"{len(self.declared_quantities)} declared but unchecked, "
                f"{len(self.uncited_quantities)} uncited, "
                f"{len(self.broken_citations)} broken citation(s)")


class NumericGuard:
    """Bind every quantity in model prose to the cell it came from, or refuse.

    The guard this replaces looked for a bracketed string on the same line. It
    accepted one citation covering two numbers from two different files, a
    citation naming a file that was never written, a citation with no row or
    column, a value contradicting the file it cited, and a citation placed
    before the number it was meant to back. A check that accepts all five is
    not stopping a fabricated measurement; it is teaching the model which
    punctuation to add.

    What is enforced now:

    * **A citation says where the number can be found again** -- artifact,
      version hash, row, field, method. See :mod:`eagent.harness.citation`
      for the grammar and for what each key is load-bearing against.
    * **Binding is by adjacency.** A quantity belongs to the first citation
      that follows it with no other quantity in between, so one citation can
      no longer license a line.
    * **The value is checked against the cell**, when an
      :class:`~eagent.harness.citation.ArtifactIndex` is supplied. Without one
      the grammar and the binding are still enforced: a syntax check is weak,
      but a syntax check nobody can satisfy by accident is not nothing.

    ``require_verified`` refuses a value whose method this guard cannot check
    -- a declared derivation. It is off by default because a run with no
    artifact index could never satisfy it, and a guard that always raises gets
    switched off.
    """

    def __init__(self, strict: bool = True,
                 index: ArtifactIndex | None = None,
                 require_verified: bool = False) -> None:
        self.strict = strict
        self.index = index
        self.require_verified = require_verified

    # -- inspection --------------------------------------------------------
    def inspect(self, text: str) -> GuardReport:
        report = GuardReport()
        citations, problems = citation_mod.parse_citations(text)
        report.broken_citations.extend(problems)
        for legacy in LEGACY_CITATION_PATTERN.finditer(text):
            report.broken_citations.append(
                f"{legacy.group(0)}: the old citation form names no version, "
                f"row, field or method, so no number can be checked against "
                f"it. Use [cite artifact=... sha256=... row=... field=... "
                f"method=read]")

        for match in MEASUREMENT_PATTERN.finditer(text):
            value, unit = match.group(1), match.group(2)
            token = f"{value} {unit}".strip()
            bound = self._binding(match, citations, text)
            if bound is None:
                report.uncited_quantities.append(token)
                continue
            report.cited_quantities.append(token)
            if self.index is None:
                report.declared_quantities.append(token)
                continue
            outcome = citation_mod.verify(value, bound, self.index)
            if not outcome.ok:
                report.broken_citations.append(outcome.describe())
            elif outcome.verified:
                report.verified_quantities.append(token)
            else:
                report.declared_quantities.append(token)
        return report

    @staticmethod
    def _binding(match: "re.Match[str]", citations: Sequence[Citation],
                 text: str) -> Citation | None:
        """The citation that backs this quantity, or ``None``.

        The first citation after the number, provided no other quantity comes
        between them. ``3.6 A [cite ...] and 42 pLDDT [cite ...]`` binds each
        number to its own; ``3.6 A and 42 pLDDT [cite ...]`` leaves the 3.6
        unbound, which is what the sentence actually says. A citation before
        the number backs nothing: it was written about something else.
        """
        after = [c for c in citations if c.start >= match.end()]
        if not after:
            return None
        nearest = min(after, key=lambda c: c.start)
        if MEASUREMENT_PATTERN.search(text[match.end():nearest.start]):
            return None
        return nearest

    # -- enforcement -------------------------------------------------------
    def check(self, text: str) -> str:
        report = self.inspect(text)
        if not self.strict:
            return text
        problems: list[str] = []
        if report.uncited_quantities:
            problems.append(
                "quantities with no citation bound to them: "
                + ", ".join(sorted(set(report.uncited_quantities))[:8]))
        if report.broken_citations:
            problems.append("broken citation(s): "
                            + "; ".join(report.broken_citations[:4]))
        if self.require_verified and report.declared_quantities:
            problems.append(
                "quantities whose citation could not be checked against the "
                "artifact: "
                + ", ".join(sorted(set(report.declared_quantities))[:8]))
        if problems:
            raise FabricationGuardError(
                "model output would state a measurement nobody can find "
                "again. " + " | ".join(problems)
                + ". Cite the cell the value came from -- [cite artifact=<key> "
                  "sha256=<digest> row=<id> field=<column> method=read] -- or "
                  "compute it with a tool.")
        return text


@dataclass
class ToolCall:
    """A model's request to run one registered interface."""

    interface: str
    arguments: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class Hypothesis:
    """A falsifiable statement the model proposes, with how to test it."""

    statement: str
    rationale: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    test: str = ""
    would_falsify: str = ""

    def is_testable(self) -> bool:
        return bool(self.test and self.would_falsify)


@dataclass
class ModelTurn:
    """The only three shapes a model response may take."""

    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def is_empty(self) -> bool:
        return not (self.tool_calls or self.hypotheses or self.questions)


class LLMClient(abc.ABC):
    """Provider-agnostic client. Implementations must not retry silently."""

    name: str = "abstract"
    #: Whether a prompt leaves this machine. Assumed true: a client that is
    #: local must say so, because a remote one sees everything in the prompt
    #: and disclosure cannot be taken back.
    runs_remotely: bool = True

    @abc.abstractmethod
    def complete(self, system: str, messages: list[dict[str, str]],
                 tools: list[dict[str, Any]] | None = None,
                 temperature: float = 0.0, seed: int | None = None) -> str:
        """Return the raw response text."""

    def propose(self, system: str, messages: list[dict[str, str]],
                tools: list[dict[str, Any]] | None = None,
                guard: NumericGuard | None = None,
                temperature: float = 0.0, seed: int | None = None) -> ModelTurn:
        """Complete, guard, and parse into the restricted turn shape."""
        raw = self.complete(system, messages, tools, temperature, seed)
        (guard or NumericGuard()).check(raw)
        return parse_turn(raw)


class EchoClient(LLMClient):
    """Deterministic offline client used in tests and dry runs.

    Returns a scripted sequence of responses. Having a real client in the test
    path matters: it keeps the parsing and guarding code on the executed path
    instead of only on the production one.
    """

    name = "echo"
    runs_remotely = False

    def __init__(self, scripted: Iterable[str] | None = None) -> None:
        self._scripted = list(scripted or [])
        self._i = 0
        self.calls: list[dict[str, Any]] = []

    def complete(self, system: str, messages: list[dict[str, str]],
                 tools: list[dict[str, Any]] | None = None,
                 temperature: float = 0.0, seed: int | None = None) -> str:
        self.calls.append({"system": system, "messages": messages,
                           "tools": [t.get("name") for t in (tools or [])],
                           "temperature": temperature, "seed": seed})
        if self._i < len(self._scripted):
            out = self._scripted[self._i]
            self._i += 1
            return out
        return json.dumps({"reasoning": "no scripted response", "questions":
                           ["What should happen next?"]})


class CallbackClient(LLMClient):
    """Wraps any callable, for embedding the harness in another runtime."""

    name = "callback"

    def __init__(self, fn: Callable[..., str], name: str = "callback",
                 runs_remotely: bool = True) -> None:
        self._fn = fn
        self.name = name
        self.runs_remotely = runs_remotely

    def complete(self, system: str, messages: list[dict[str, str]],
                 tools: list[dict[str, Any]] | None = None,
                 temperature: float = 0.0, seed: int | None = None) -> str:
        return self._fn(system=system, messages=messages, tools=tools,
                        temperature=temperature, seed=seed)


class LLMError(EAgentError):
    """A provider call failed. Carries the provider's own message."""


class AnthropicMessagesClient(LLMClient):
    """The Anthropic Messages API, over plain HTTPS.

    NOT YET RUN AGAINST THE LIVE API. The request shape (``POST /v1/messages``,
    ``x-api-key`` and ``anthropic-version`` headers, a ``system`` string,
    ``messages``, ``max_tokens``, ``temperature``; the reply's ``content``
    blocks of type ``text``) is written from the public API reference, and the
    tests drive it through a fake opener. Nothing here has been sent to the
    service, because no credential was available when this was written.

    * The key is read from the environment at call time and is never stored on
      the object, logged, or written to an audit file.
    * There is no ``seed`` parameter in the API, so ``seed`` is accepted and
      ignored; the planner stores the response itself for exactly that reason.
    * There are no retries. A failure is raised with the provider's message and
      the planner records it; "implementations must not retry silently".
    * It runs remotely, so the planner will not call it while the network is
      disabled and will not send it a sequence.
    """

    name = "anthropic-messages"
    runs_remotely = True
    ENDPOINT = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(self, model: str, *, max_tokens: int = 1024,
                 api_key_env: str = "ANTHROPIC_API_KEY",
                 endpoint: str | None = None, timeout_s: float = 60.0,
                 opener: Callable[..., Any] | None = None) -> None:
        if not model or not model.strip():
            raise ValueError("a model identifier is required; there is no default")
        self.model = model
        self.max_tokens = max_tokens
        self.api_key_env = api_key_env
        self.endpoint = endpoint or self.ENDPOINT
        self.timeout_s = timeout_s
        self._opener = opener or urllib.request.urlopen
        self.name = f"anthropic-messages:{model}"

    def complete(self, system: str, messages: list[dict[str, str]],
                 tools: list[dict[str, Any]] | None = None,
                 temperature: float = 0.0, seed: int | None = None) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise LLMError(f"{self.api_key_env} is not set; no request was made")
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "system": system, "messages": messages,
                "temperature": temperature}
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"x-api-key": key, "anthropic-version": self.API_VERSION,
                     "content-type": "application/json"})
        try:
            with self._opener(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:                        # noqa: BLE001
                pass
            raise LLMError(f"HTTP {exc.code} from the Messages API: "
                           f"{detail or exc.reason}") from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LLMError(f"the Messages API call failed: {exc}") from exc
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise LLMError("the reply has no content blocks")
        text = "".join(b.get("text", "") for b in blocks
                       if isinstance(b, dict) and b.get("type") == "text")
        if not text:
            raise LLMError("the reply holds no text block")
        return text


def parse_turn(raw: str) -> ModelTurn:
    """Parse a JSON model turn, tolerating a fenced code block around it.

    An unparseable response becomes a turn carrying the text as a question
    rather than an exception, so the controller can surface it to the operator
    instead of crashing a long run.
    """
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return ModelTurn(reasoning="", questions=[raw.strip()], raw=raw)
    if not isinstance(obj, dict):
        return ModelTurn(questions=[raw.strip()], raw=raw)

    calls = [
        ToolCall(interface=c.get("interface", ""),
                 arguments=c.get("arguments", {}) or {},
                 rationale=c.get("rationale", ""))
        for c in obj.get("tool_calls", []) or []
        if isinstance(c, dict) and c.get("interface")
    ]
    hyps = [
        Hypothesis(statement=h.get("statement", ""),
                   rationale=h.get("rationale", ""),
                   evidence_refs=list(h.get("evidence_refs", []) or []),
                   test=h.get("test", ""),
                   would_falsify=h.get("would_falsify", ""))
        for h in obj.get("hypotheses", []) or []
        if isinstance(h, dict) and h.get("statement")
    ]
    return ModelTurn(
        reasoning=str(obj.get("reasoning", "")),
        tool_calls=calls,
        hypotheses=hyps,
        questions=[str(q) for q in (obj.get("questions", []) or [])],
        raw=raw,
    )


def validate_turn(turn: ModelTurn, known_interfaces: Iterable[str]) -> list[str]:
    """Problems with a turn, as messages. Empty list means acceptable."""
    known = set(known_interfaces)
    problems: list[str] = []
    for call in turn.tool_calls:
        if call.interface not in known:
            problems.append(
                f"unknown interface '{call.interface}'; the model may only call "
                f"registered interfaces"
            )
    for h in turn.hypotheses:
        if not h.is_testable():
            problems.append(
                f"hypothesis is not falsifiable as stated: {h.statement!r}; "
                f"it needs both a test and what result would refute it"
            )
    if turn.is_empty:
        problems.append("turn contains no tool call, hypothesis or question")
    return problems


SYSTEM_PROMPT = """\
You are the research controller of an enzyme function mining and substrate-directed
engineering agent.

Your role is to understand the chemical task, organise evidence retrieval, propose
falsifiable hypotheses, interpret results and plan the next step.

You do not compute scientific quantities. Alignments, structural coordinates, atom
mappings, distances, angles, model confidences, docking results and statistics are
produced only by the registered interfaces. If you need a number, call the interface
that measures it.

Rules you must follow:
- A field whose value is unknown stays unknown. Never fill a substrate structure,
  stereochemistry, cofactor state, pH or temperature with a plausible value. Ask instead.
- Quote a quantity only with a citation that lets somebody find it again, written
  immediately after the number as
  [cite artifact=<key> sha256=<digest> row=<id> field=<column> method=read].
  One citation backs one number: the next quantity needs its own, and a citation
  placed before a number backs nothing. Use method=rounded when you have rounded
  the cell, and method=derived:<how> when the number is worked out rather than
  read -- a derived number is reported as unchecked, so prefer calling the
  interface that measures it.
- A modelling failure is not an experimental negative. An expression failure is not
  evidence about catalysis. An untested pair is not a negative.
- A hypothesis must come with the experiment that would refute it.
- Never claim that a candidate will work. Rank by the evidence that exists and say what
  is missing.

Respond with a single JSON object containing any of:
  "reasoning":   brief statement of what you concluded and why
  "tool_calls":  [{"interface": ..., "arguments": {...}, "rationale": ...}]
  "hypotheses":  [{"statement":..., "rationale":..., "evidence_refs":[...],
                   "test":..., "would_falsify":...}]
  "questions":   [strings, for decisions only a human operator may make]
"""
