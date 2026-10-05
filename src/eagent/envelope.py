"""The uniform return envelope every scientific interface must produce.

Each of the ten interfaces returns the same shape so the controller can route
on it without knowing the science:

    status / artifacts / provenance / qc_flags / uncertainty / next_actions

``uncertainty`` is deliberately a first-class field rather than a free-text
note: a step that produced an answer it cannot defend must say so in a place
the controller can branch on.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field, asdict
from typing import Any


class Status(str, enum.Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"

    @property
    def usable(self) -> bool:
        """Whether downstream steps may consume the artifacts."""
        return self in (Status.SUCCESS, Status.PARTIAL)


class Severity(str, enum.Enum):
    INFO = "info"
    WARN = "warn"
    BLOCKER = "blocker"


@dataclass(frozen=True)
class Artifact:
    """A file or structured object produced by a step."""

    key: str
    path: str | None = None
    kind: str = "file"          # file | table | object | structure | figure
    sha256: str | None = None
    n_records: int | None = None
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QCFlag:
    """A quality problem found by the step or by the independent verifier."""

    code: str
    severity: Severity
    message: str
    subject: str | None = None   # candidate id, residue, file ...

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass(frozen=True)
class Uncertainty:
    """An unresolved question, stated so the next step can act on it."""

    code: str
    question: str
    affects: list[str] = field(default_factory=list)
    resolvable_by: str = ""      # e.g. "operator input", "experiment", "better template"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NextAction:
    """A concrete, executable follow-up; not a vague suggestion."""

    action: str                  # name of an interface or an operator task
    rationale: str
    params: dict[str, Any] = field(default_factory=dict)
    requires_human: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Provenance:
    """Everything needed to recompute a step."""

    tool: str
    tool_version: str = "unknown"
    inputs_sha256: dict[str, str] = field(default_factory=dict)
    databases: dict[str, str] = field(default_factory=dict)   # name -> snapshot/version
    models: dict[str, str] = field(default_factory=dict)      # name -> version/weights id
    parameters: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    cost: dict[str, Any] = field(default_factory=dict)        # cpu_s, gpu_s, api_tokens, currency

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    """Uniform envelope returned by every interface."""

    status: Status
    artifacts: list[Artifact] = field(default_factory=list)
    provenance: Provenance | None = None
    qc_flags: list[QCFlag] = field(default_factory=list)
    uncertainty: list[Uncertainty] = field(default_factory=list)
    next_actions: list[NextAction] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    # -- queries -----------------------------------------------------------
    @property
    def blockers(self) -> list[QCFlag]:
        return [f for f in self.qc_flags if f.severity is Severity.BLOCKER]

    @property
    def ok(self) -> bool:
        """Usable *and* free of blocking QC problems."""
        return self.status.usable and not self.blockers

    def artifact(self, key: str) -> Artifact | None:
        for a in self.artifacts:
            if a.key == key:
                return a
        return None

    # -- construction helpers ---------------------------------------------
    def add_flag(self, code: str, severity: Severity, message: str,
                 subject: str | None = None) -> "ToolResult":
        self.qc_flags.append(QCFlag(code, severity, message, subject))
        return self

    def add_uncertainty(self, code: str, question: str, affects=None,
                        resolvable_by: str = "") -> "ToolResult":
        self.uncertainty.append(
            Uncertainty(code, question, list(affects or []), resolvable_by)
        )
        return self

    def add_next(self, action: str, rationale: str, params=None,
                 requires_human: bool = False) -> "ToolResult":
        self.next_actions.append(
            NextAction(action, rationale, dict(params or {}), requires_human)
        )
        return self

    @classmethod
    def failure(cls, tool: str, message: str, code: str = "step_failed") -> "ToolResult":
        r = cls(status=Status.FAILED, provenance=Provenance(tool=tool), message=message)
        r.add_flag(code, Severity.BLOCKER, message)
        return r

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "message": self.message,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "qc_flags": [f.to_dict() for f in self.qc_flags],
            "uncertainty": [u.to_dict() for u in self.uncertainty],
            "next_actions": [n.to_dict() for n in self.next_actions],
            "data": self.data,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)
