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
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..errors import FabricationGuardError


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

#: Phrases that mark a number as quoted from an artifact rather than produced.
CITATION_PATTERN = re.compile(
    r"\[(?:artifact|table|file|record|evidence):[^\]]+\]", re.IGNORECASE
)


@dataclass
class GuardReport:
    """What the guard found in one model response."""

    uncited_quantities: list[str] = field(default_factory=list)
    cited_quantities: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.uncited_quantities


class NumericGuard:
    """Reject model-authored quantities that cite no artifact.

    A model is free to write "the hydride transfer distance in
    [artifact:catalytic_geometry.tsv] is 3.6 A" because that value came from a
    file the verifier can open. It may not write "the transfer distance is
    about 3.6 A", which is an estimate wearing the clothes of a measurement.
    """

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    def inspect(self, text: str) -> GuardReport:
        report = GuardReport()
        for line in text.splitlines():
            matches = MEASUREMENT_PATTERN.findall(line)
            if not matches:
                continue
            cited = bool(CITATION_PATTERN.search(line))
            for value, unit in matches:
                token = f"{value} {unit}".strip()
                (report.cited_quantities if cited
                 else report.uncited_quantities).append(token)
        return report

    def check(self, text: str) -> str:
        report = self.inspect(text)
        if self.strict and not report.clean:
            raise FabricationGuardError(
                "model output contains quantities with no artifact citation: "
                + ", ".join(sorted(set(report.uncited_quantities))[:8])
                + ". Cite the artifact the value came from, or compute it with a tool."
            )
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

    def __init__(self, fn: Callable[..., str], name: str = "callback") -> None:
        self._fn = fn
        self.name = name

    def complete(self, system: str, messages: list[dict[str, str]],
                 tools: list[dict[str, Any]] | None = None,
                 temperature: float = 0.0, seed: int | None = None) -> str:
        return self._fn(system=system, messages=messages, tools=tools,
                        temperature=temperature, seed=seed)


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
- Quote a quantity only with the artifact it came from, written as [artifact:<name>].
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
