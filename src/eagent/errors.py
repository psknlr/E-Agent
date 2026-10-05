"""Typed failures for the enzyme-mining harness.

Design rule: a tool that cannot do its job raises or returns ``failed``.
It never substitutes a plausible-looking value for a missing one. Silent
imputation is the single most dangerous failure mode in this pipeline,
because a fabricated cofactor state or stereocentre propagates all the way
into a synthesis order.
"""

from __future__ import annotations


class EAgentError(Exception):
    """Base class for every error raised inside the harness."""


class UnresolvedFieldError(EAgentError):
    """A field required by the current gate is still unresolved (``null``).

    Raised instead of guessing. Carries the dotted paths so the controller can
    tell the operator exactly what to supply.
    """

    def __init__(self, paths, gate: str = ""):
        self.paths = list(paths)
        self.gate = gate
        joined = ", ".join(self.paths)
        suffix = f" required by gate '{gate}'" if gate else ""
        super().__init__(f"Unresolved required field(s){suffix}: {joined}")


class FabricationGuardError(EAgentError):
    """A component tried to emit a value it has no evidence for."""


class ToolUnavailableError(EAgentError):
    """An external executable, model weight set, or database is not present.

    The harness degrades loudly: the step is reported ``failed`` with this
    reason rather than falling back to a heuristic that imitates the tool.
    """

    def __init__(self, tool: str, hint: str = ""):
        self.tool = tool
        self.hint = hint
        msg = f"External tool unavailable: {tool}"
        if hint:
            msg += f" ({hint})"
        super().__init__(msg)


class TemplateError(EAgentError):
    """A catalytic/family/reaction template is missing, malformed, or unsourced."""


class ProvenanceError(EAgentError):
    """An artifact was produced without recordable provenance."""


class ApprovalRequiredError(EAgentError):
    """A human approval gate has not been cleared."""

    def __init__(self, gate: str, detail: str = ""):
        self.gate = gate
        super().__init__(f"Human approval required at gate '{gate}'. {detail}".strip())


class CircularEvidenceError(EAgentError):
    """A restraint used to build a model was re-used as independent evidence."""


class LicenseError(EAgentError):
    """A tool's licence forbids the requested use (e.g. commercial service)."""
