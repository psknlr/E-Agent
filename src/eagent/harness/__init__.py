"""Orchestration: one research controller, deterministic tools, an independent verifier.

Deliberately not a committee of role-playing agents arguing with each other. A
single controller plans and interprets; everything measurable is computed by
code or by a dedicated scientific model; a separate verifier checks the result
without reusing the producing step's conclusions.

The five modules divide as follows. :mod:`~eagent.harness.llm` is the only
place a language model is spoken to. :mod:`~eagent.harness.templates` is the
only place a threshold may come from. :mod:`~eagent.harness.registry` names
the ten interfaces that must exist for a run to be complete.
:mod:`~eagent.harness.approval` holds the three human decision points and the
block that makes them real. :mod:`~eagent.harness.controller` decides what
runs next and nothing scientific at all, and
:mod:`~eagent.harness.verifier` re-derives what the run claims.
"""

from __future__ import annotations

from .citation import ArtifactEntry, ArtifactIndex, Citation
from .llm import (
    CallbackClient, EchoClient, GuardReport, Hypothesis, LLMClient, ModelTurn,
    GuardReport, NumericGuard, SYSTEM_PROMPT, ToolCall, parse_turn,
    validate_turn,
)
from .templates import (
    CalibrationReport, ConstraintRecord, LoadedTemplate, TEMPLATE_KINDS,
    TemplateLibrary, WINDOW_AUTHORITIES, default_template_dir, normalise_family,
)
from .registry import (
    PROTOCOL_INTERFACES, PROTOCOL_ORDER, build_interface_registry,
    dependency_problems, registry_report, topological_order,
)
from .approval import (
    APPROVAL_GATES, ApprovalQueue, ApprovalRequest, BATCH_GATE, CRITERIA_GATE,
    DECISION_POINTS, Decision, DecisionPoint, REACTION_GATE, RequestKind,
    batch_cost_payload, guard_batch_selection,
)
from .controller import (
    Branch, ControllerHooks, Escalation, FailureKind, RETRY_POLICY,
    ResearchController, RetryRule, RunOutcome, Stage, StepAttempt,
    TERMINAL_STAGES, TRANSITIONS, classify_failure,
)
from .verifier import (
    Claim, Finding, IGNORED_PRODUCER_CONCLUSIONS, IndependentVerifier,
    LigandDeclaration, VERIFIER_NAME, VerificationReport,
)

__all__ = [
    # llm boundary
    "CallbackClient", "EchoClient", "GuardReport", "Hypothesis", "LLMClient",
    "ArtifactEntry", "ArtifactIndex", "Citation", "GuardReport",
    "ModelTurn", "NumericGuard", "SYSTEM_PROMPT", "ToolCall", "parse_turn",
    "validate_turn",
    # templates
    "CalibrationReport", "ConstraintRecord", "LoadedTemplate", "TEMPLATE_KINDS",
    "TemplateLibrary", "WINDOW_AUTHORITIES", "default_template_dir",
    "normalise_family",
    # interface registry
    "PROTOCOL_INTERFACES", "PROTOCOL_ORDER", "build_interface_registry",
    "dependency_problems", "registry_report", "topological_order",
    # approval
    "APPROVAL_GATES", "ApprovalQueue", "ApprovalRequest", "BATCH_GATE",
    "CRITERIA_GATE", "DECISION_POINTS", "Decision", "DecisionPoint",
    "REACTION_GATE", "RequestKind", "batch_cost_payload",
    "guard_batch_selection",
    # controller
    "Branch", "ControllerHooks", "Escalation", "FailureKind", "RETRY_POLICY",
    "ResearchController", "RetryRule", "RunOutcome", "Stage", "StepAttempt",
    "TERMINAL_STAGES", "TRANSITIONS", "classify_failure",
    # verifier
    "Claim", "Finding", "IGNORED_PRODUCER_CONCLUSIONS", "IndependentVerifier",
    "LigandDeclaration", "VERIFIER_NAME", "VerificationReport",
]
