"""The research controller, as an explicit state machine.

The controller decides *what runs next*. It does not decide anything
scientific. There is no distance, no score, no confidence and no ranking
computed anywhere in this module: every number it touches came out of an
interface's :class:`~eagent.envelope.ToolResult` or out of the manifest's own
bookkeeping. That separation is the point of the architecture, and it is
cheap to lose -- a single "if plddt > 70" here would plant a threshold that no
template authorised and no reviewer would find.

Shape of the machine
--------------------
::

    reaction spec confirmation
        -> evidence retrieval and family location
        -> candidate sequences and structures
        -> catalytic complex evaluation
             |-- input or mapping error  -> repair, re-run the affected step
             |-- insufficient evidence   -> widen retrieval, or carry the
             |                              candidate as an exploration probe
             '-- sufficient support      -> batch approval
        -> experimental results return
             |-- hit      -> local engineering
             '-- no hit   -> failure-cause diagnosis

Going back is a declared transition (:data:`TRANSITIONS`), not a recursive
call: the manifest has to be able to show that the run returned to
``prepare_structures`` after a repair, and a stack frame cannot be written to
a JSON file.

Retries
-------
Every failure type gets its own handling, because funnelling them into one
"try again" loop is how an agent spends a GPU budget re-running a step whose
input file does not exist. A missing binary is never retried -- re-running it
cannot succeed, and swapping in another model is how a missing tool becomes a
fabricated result. An unhandled exception is never retried -- that is a bug in
this codebase, and a retry hides it. Only a repairable input error and
insufficient evidence are re-run, each at most twice, each capped again by
``ctx.policy.max_retries``, and all of it stops as soon as a recorded cost
exceeds ``ctx.policy.cost_ceiling``.

Repairing and widening are *hooks*, not behaviours of this module. The
controller knows that an input error should be repaired; it does not know what
the right structure index or the right seed set is, and guessing would make it
the author of a scientific choice that belongs to the operator or to an
interface.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from pydantic import BaseModel

from ..context import RunContext
from ..envelope import NextAction, Severity, Status, ToolResult
from ..errors import ApprovalRequiredError
from ..provenance import canonical_json, sha256_file, sha256_obj, utc_now
from ..tools.base import InterfaceRegistry, ScientificInterface
from .approval import (
    APPROVAL_GATES, ApprovalQueue, BATCH_GATE, CRITERIA_GATE, REACTION_GATE,
    RequestKind, guard_batch_selection,
)
from .registry import build_interface_registry

__all__ = [
    "Branch",
    "CLASSIFIED_CODES",
    "ControllerHooks",
    "Escalation",
    "FailureKind",
    "INPUT_ERROR_CODES",
    "INSUFFICIENT_EVIDENCE_CODES",
    "INTERNAL_ERROR_CODES",
    "LINEAR_SUCCESSOR",
    "NEEDS_HUMAN_CODES",
    "POLICY_BLOCKED_CODES",
    "RETRY_POLICY",
    "ResearchController",
    "RetryRule",
    "RunOutcome",
    "Stage",
    "StepAttempt",
    "TERMINAL_STAGES",
    "TOOL_UNAVAILABLE_CODES",
    "TRANSITIONS",
    "UNDIAGNOSED_CODES",
    "classify_failure",
    "describe_failure",
]


# ---------------------------------------------------------------------------
# failure vocabulary
# ---------------------------------------------------------------------------

#: Blocking codes that mean "what was fed in is wrong or absent". These are
#: repairable: a structure index that does not parse, a pool that arrived
#: empty, a catalytic template that was never loaded. Taken from the codes the
#: ten interfaces actually emit, so the classifier is matched to the envelopes
#: it will see rather than to a vocabulary invented here. That claim is not
#: left to a comment: ``test_controller`` scans every
#: ``add_flag(..., Severity.BLOCKER)`` and ``ToolResult.failure(code=...)``
#: under ``src/eagent/tools/`` and fails when a code reaches none of these
#: sets, because a code nobody classified becomes an escalation with no
#: recovery route and the two halves drift apart silently.
INPUT_ERROR_CODES: frozenset[str] = frozenset({
    "missing_input", "missing_input_artifact", "no_sequences", "no_seeds",
    "no_candidates", "no_parents", "no_members", "empty_pool", "no_database",
    "no_search_adapter", "no_searchable_terms", "bad_structure_index",
    "missing_structure_index", "structure_unreadable", "receptor_unreadable",
    "unreadable_results", "schema_rejected", "record_rejected_by_schema",
    "malformed_catalytic_template", "catalytic_template_missing",
    "no_catalytic_template", "numbering_unavailable", "restraint_name_mismatch",
    "ligandmpnn_length_mismatch", "cofactor_wrong_identity",
    "cofactor_wrong_oxidation_state", "substrate_product_identical",
    "unresolved_subject_ids", "reaction_class_unset", "unresolvedfielderror",
    # ``prepare_structures`` reports a candidate with no usable coordinate
    # file here. A cache miss is routine, and leaving it unclassified turned
    # an ordinary "fetch the structure" into a dead-ended run.
    "no_usable_structure",
    # The spec contradicts itself about the cofactor or the stereocentre. The
    # operator edits one of the two declarations; nothing about the step is
    # at fault, so this is a repairable input and not an escalation.
    "cofactor_state_contradicts_ligand_code",
    "stereo_target_without_stereocentre", "stereo_achiral_with_stereocentre",
    # ``select_batch``: the declared control plan is not self-consistent, or
    # does not fit the gene budget it was given. Both are the plate plan the
    # operator supplied, and both are fixed by editing it.
    "control_claim_problem", "control_budget_overflow",
    # ``ingest_results``: the system control on the plate did not fire, so the
    # detection chain -- an input to every verdict on that plate -- is not
    # demonstrated. Re-reading the same file cannot fix it; a revalidated
    # plate can, which is the repair branch rather than an escalation.
    "assay_system_control_failed",
    # ``propose_mutations``: the catalytic freeze could not be verified
    # against this parent, or the designer returned a sequence that
    # contradicts the parent it was given. Both mean the inputs to the design
    # pass -- the residue map, the frozen set, the parent sequence -- do not
    # line up, and both are repaired upstream rather than retried here.
    "freeze_unverifiable", "ligandmpnn_touched_frozen_residue",
    "ligandmpnn_wild_type_unverified",
})

#: Codes a human must answer. Never retried: re-running the step reproduces
#: the same question.
NEEDS_HUMAN_CODES: frozenset[str] = frozenset({
    "approval_required", "unresolved_fields", "approvalrequirederror",
    "single_seed_not_authorised", "single_seed_unjustified",
    "no_pre_registered_criterion", "criterion_undecidable",
    "protocol_deviation_refused", "template_assumption_rejected",
    # ``normalize_reaction``: the substrate or the product is a name, or is
    # absent. A name fixes neither tautomer, salt form nor stereochemistry,
    # and this harness will not pick one -- only a chemist can supply the
    # structure, so there is nothing here for a repair hook to repair.
    "substrate_name_only", "substrate_unspecified",
    "product_name_only", "product_unspecified",
    # Whether the reaction creates a new stereocentre is not inferred from
    # the reaction class; it is asked.
    "stereocentre_undetermined",
    # ``propose_mutations``: the parent has not been shown to run the target
    # chemistry. Proceeding needs a ParentOverride with a stated reason, or
    # an assay of the parent -- both are decisions, not inputs.
    "unconfirmed_parent",
})

#: Codes meaning a tool, model or network resource is not available. Retrying
#: these burns budget for an outcome that cannot change, and substituting a
#: different model is exactly the failure mode the harness exists to prevent.
TOOL_UNAVAILABLE_CODES: frozenset[str] = frozenset({
    "tool_unavailable", "toolunavailableerror", "search_tool_unavailable",
    "ligandmpnn_unavailable", "network_disabled", "network_budget_exhausted",
    "ligandmpnn_network_blocked", "cache_miss",
    # ``mine_sequences`` raises this when every configured search crashed
    # rather than returning nothing. The sibling of search_tool_unavailable:
    # the search machinery did not run, so there is no pool to judge, and
    # reaching for whichever method did survive would silently change the
    # method behind the pool.
    "search_failed",
})

#: Codes where a policy or a licence refused the work. Escalated, never
#: retried and never worked around.
POLICY_BLOCKED_CODES: frozenset[str] = frozenset({
    "license_blocked", "licenseerror", "external_submission_refused",
    "ligandmpnn_disclosure_blocked", "identity_signal_withheld",
})

#: Codes meaning the step ran correctly and found too little to act on. These
#: are the "widen the inputs" branch -- never "lower the thresholds", which is
#: the same sentence with the evidence removed.
INSUFFICIENT_EVIDENCE_CODES: frozenset[str] = frozenset({
    "no_results", "no_rows", "no_evidence_retrieved", "evidence_gaps",
    "coverage_shortfall", "family_unassigned", "no_hypotheses",
    "no_supported_seed", "insufficient_evidence", "pool_not_composable",
    "no_actionable_sites", "evidence_entirely_circular", "layers_not_searched",
    "chemotype_unassigned", "catalytic_roles_incomplete",
    "family_signals_contradictory", "no_hits_in_round",
    # ``model_complexes``: the assembly is missing a component of the
    # catalytic system, or a modelling route produced nothing usable for this
    # candidate. Neither is a wrong input and neither is a broken tool -- the
    # run simply has too little structural evidence for that candidate, which
    # is the widen-or-carry-as-a-probe branch.
    "incomplete_catalytic_system", "route_failed",
})

#: Defects in this codebase. Never retried: a retry hides the bug, and a
#: hidden bug eventually produces a wrong number.
INTERNAL_ERROR_CODES: frozenset[str] = frozenset({"internal_error"})

#: Codes that name no recovery at all. ``step_failed`` is
#: :meth:`ToolResult.failure`'s default, so a step emitting it has not said
#: what went wrong; there is nothing to route on and the run escalates
#: carrying the envelope's own message.
UNDIAGNOSED_CODES: frozenset[str] = frozenset({"step_failed"})

#: Every code this module can place. The drift test in ``test_controller``
#: compares the interfaces' emitted vocabulary against this union.
CLASSIFIED_CODES: frozenset[str] = (
    INPUT_ERROR_CODES | NEEDS_HUMAN_CODES | TOOL_UNAVAILABLE_CODES
    | POLICY_BLOCKED_CODES | INSUFFICIENT_EVIDENCE_CODES
    | INTERNAL_ERROR_CODES | UNDIAGNOSED_CODES
)


class FailureKind(str, enum.Enum):
    """How a step failed, which is what decides whether it may be re-run."""

    NONE = "none"
    INPUT_ERROR = "input_error"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NEEDS_HUMAN = "needs_human"
    TOOL_UNAVAILABLE = "tool_unavailable"
    POLICY_BLOCKED = "policy_blocked"
    INTERNAL_ERROR = "internal_error"
    UNCLASSIFIED = "unclassified"


class Branch(str, enum.Enum):
    """The three-way branch after catalytic complex evaluation, plus the stops."""

    SUFFICIENT_SUPPORT = "sufficient_support"
    INPUT_OR_MAPPING_ERROR = "input_or_mapping_error"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class RetryRule:
    """What may be done about one failure kind, and how often."""

    max_attempts: int
    action: str            # repair_and_rerun | widen_inputs | await_human | escalate
    why: str


#: Per-failure-type policy. The asymmetry is the whole point: two of the seven
#: kinds are re-runnable and the rest are not.
RETRY_POLICY: dict[FailureKind, RetryRule] = {
    FailureKind.INPUT_ERROR: RetryRule(
        max_attempts=2, action="repair_and_rerun",
        why=("a malformed or missing input can be repaired once; the same "
             "error after a repair means the repair is not addressing it and "
             "a human has to look")),
    FailureKind.INSUFFICIENT_EVIDENCE: RetryRule(
        max_attempts=2, action="widen_inputs",
        why=("widening the inputs once is cheap; widening repeatedly is a "
             "search for a result that may simply not be there, and the "
             "candidates belong in the exploration bucket instead")),
    FailureKind.NEEDS_HUMAN: RetryRule(
        max_attempts=1, action="await_human",
        why=("re-running reproduces the same question; the decision is queued "
             "for an operator")),
    FailureKind.TOOL_UNAVAILABLE: RetryRule(
        max_attempts=1, action="escalate",
        why=("a binary or model that is not installed will not be installed by "
             "running the step again, and reaching for a different model "
             "substitutes a guess for a measurement")),
    FailureKind.POLICY_BLOCKED: RetryRule(
        max_attempts=1, action="escalate",
        why="a licence or disclosure refusal is a decision, not a transient fault"),
    FailureKind.INTERNAL_ERROR: RetryRule(
        max_attempts=1, action="escalate",
        why=("an unhandled exception is a defect in this codebase; retrying "
             "hides it and a hidden defect eventually produces a wrong number")),
    FailureKind.UNCLASSIFIED: RetryRule(
        max_attempts=1, action="escalate",
        why=("an unrecognised failure code has no known safe recovery, and "
             "guessing one is how an agent loops")),
}


#: The code sets in the order the classifier consults them, with the kind
#: each one means. The order is the policy: a step blocked on a human and
#: also short of evidence goes to the human, because widening the search
#: while the operator has not confirmed the substrate searches harder for the
#: wrong thing. Declared as data so the fallback pass over the non-blocking
#: codes cannot drift out of step with the blocking one.
_CODE_SETS: tuple[tuple[frozenset[str], FailureKind], ...] = (
    (NEEDS_HUMAN_CODES, FailureKind.NEEDS_HUMAN),
    (POLICY_BLOCKED_CODES, FailureKind.POLICY_BLOCKED),
    (TOOL_UNAVAILABLE_CODES, FailureKind.TOOL_UNAVAILABLE),
    (INPUT_ERROR_CODES, FailureKind.INPUT_ERROR),
    (INTERNAL_ERROR_CODES, FailureKind.INTERNAL_ERROR),
    (INSUFFICIENT_EVIDENCE_CODES, FailureKind.INSUFFICIENT_EVIDENCE),
)


def _failed(result: ToolResult) -> bool:
    """Whether this envelope is a failure at all, by status or by blocker."""
    return bool(result.blockers) or result.status is Status.FAILED


def classify_failure(result: ToolResult) -> FailureKind:
    """Decide what kind of failure an envelope represents.

    Reads only the envelope -- status, QC codes and the step's own
    ``next_actions`` -- never the science. The order matters: a step that is
    blocked on a human and also short of evidence must be routed to the human,
    because widening the search while the operator has not confirmed the
    substrate searches for the wrong thing.

    Every failed envelope leaves here with a kind that
    :data:`RETRY_POLICY` has a rule for, :attr:`FailureKind.UNCLASSIFIED`
    included. A step may fail with no blocking flag at all -- a structure
    cache miss that only sets ``status`` -- and that must still be routed
    somewhere declared, because a failure the machine has no bucket for is a
    run that stops with nothing to tell the operator.
    """
    codes = {f.code.lower() for f in result.qc_flags}
    blocking = {f.code.lower() for f in result.blockers}
    asks_for_a_human = any(a.requires_human for a in result.next_actions)

    if blocking & NEEDS_HUMAN_CODES or codes & {"approval_required",
                                                "unresolved_fields"}:
        return FailureKind.NEEDS_HUMAN
    if result.blockers and asks_for_a_human:
        # The step itself said a person has to act. Whatever the code says,
        # a retry cannot supply what it is asking for.
        return FailureKind.NEEDS_HUMAN
    for vocabulary, kind in _CODE_SETS[1:]:
        if blocking & vocabulary:
            return kind
    if (blocking | codes) & INSUFFICIENT_EVIDENCE_CODES:
        return FailureKind.INSUFFICIENT_EVIDENCE
    if not _failed(result):
        return FailureKind.NONE
    # A failure whose diagnosis was recorded below blocker severity, or on a
    # result that set no flag at all. Reading the warnings is still better
    # than escalating with "unclassified": the code is the step's own, and
    # the alternative is a human re-deriving it from the message.
    for vocabulary, kind in _CODE_SETS:
        if codes & vocabulary:
            return kind
    return FailureKind.UNCLASSIFIED


def describe_failure(result: ToolResult) -> str:
    """One line saying what the classifier read, for the escalation record.

    An escalation whose only content is "unclassified" asks the operator to
    go and find the envelope themselves. This states which codes were present
    and at what severity, so the reason recorded in the manifest is the
    reason, not a label.
    """
    kind = classify_failure(result)
    blocking = sorted({f.code for f in result.blockers})
    other = sorted({f.code for f in result.qc_flags
                    if f.severity is not Severity.BLOCKER})
    parts = [f"status {result.status.value}", f"classified {kind.value}"]
    if blocking:
        parts.append("blocking code(s): " + ", ".join(blocking))
    elif _failed(result):
        parts.append("no blocking qc_flag was set")
    if other:
        parts.append("other code(s): " + ", ".join(other))
    if result.message:
        parts.append(f"message: {result.message}")
    if kind is FailureKind.UNCLASSIFIED:
        parts.append(
            "no code on this envelope appears in any of the controller's "
            "classified vocabularies, so no recovery is declared for it and "
            "the run stops here rather than guessing one")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# the machine
# ---------------------------------------------------------------------------

class Stage(str, enum.Enum):
    """States of the run. One per decision the controller can be at."""

    START = "start"
    CONFIRM_REACTION_SPEC = "confirm_reaction_spec"
    AWAIT_REACTION_APPROVAL = "await_reaction_approval"
    RETRIEVE_EVIDENCE = "retrieve_evidence"
    MINE_SEQUENCES = "mine_sequences"
    ANNOTATE_FAMILY = "annotate_family"
    PREPARE_STRUCTURES = "prepare_structures"
    MODEL_COMPLEXES = "model_complexes"
    EVALUATE_CATALYSIS = "evaluate_catalysis"
    REPAIR_INPUTS = "repair_inputs"
    WIDEN_RETRIEVAL = "widen_retrieval"
    BATCH_APPROVAL = "batch_approval"
    SELECT_BATCH = "select_batch"
    AWAIT_RESULTS = "await_results"
    INGEST_RESULTS = "ingest_results"
    LOCAL_ENGINEERING = "local_engineering"
    DIAGNOSE_NO_HIT = "diagnose_no_hit"
    # terminal
    DONE = "done"
    ESCALATED = "escalated"
    HALTED = "halted"
    AWAITING_HUMAN = "awaiting_human"
    AWAITING_RESULTS = "awaiting_results"


#: Stages the run stops in. Each is a different thing to tell the operator,
#: which is why there is not one "finished" state: "waiting for your decision"
#: and "this failed and needs you" are not the same message.
TERMINAL_STAGES: frozenset[Stage] = frozenset({
    Stage.DONE, Stage.ESCALATED, Stage.HALTED, Stage.AWAITING_HUMAN,
    Stage.AWAITING_RESULTS,
})

#: Stage -> the interface it executes. Stages absent from this map are
#: decisions, not work.
STAGE_INTERFACE: dict[Stage, str] = {
    Stage.CONFIRM_REACTION_SPEC: "normalize_reaction",
    Stage.RETRIEVE_EVIDENCE: "retrieve_evidence",
    Stage.MINE_SEQUENCES: "mine_sequences",
    Stage.ANNOTATE_FAMILY: "annotate_family",
    Stage.PREPARE_STRUCTURES: "prepare_structures",
    Stage.MODEL_COMPLEXES: "model_complexes",
    Stage.EVALUATE_CATALYSIS: "evaluate_catalysis",
    Stage.SELECT_BATCH: "select_batch",
    Stage.INGEST_RESULTS: "ingest_results",
    Stage.LOCAL_ENGINEERING: "propose_mutations",
}

_TERMINALS: tuple[Stage, ...] = tuple(sorted(TERMINAL_STAGES, key=lambda s: s.value))

#: Every legal move. An undeclared move raises rather than silently happening,
#: so "how did the run get from modelling to ordering genes" is answerable
#: from this table instead of from reading the code.
TRANSITIONS: dict[Stage, tuple[Stage, ...]] = {
    Stage.START: (Stage.CONFIRM_REACTION_SPEC,) + _TERMINALS,
    Stage.CONFIRM_REACTION_SPEC: (
        Stage.AWAIT_REACTION_APPROVAL, Stage.RETRIEVE_EVIDENCE,
        Stage.REPAIR_INPUTS) + _TERMINALS,
    Stage.AWAIT_REACTION_APPROVAL: (Stage.RETRIEVE_EVIDENCE,
                                    Stage.CONFIRM_REACTION_SPEC) + _TERMINALS,
    Stage.RETRIEVE_EVIDENCE: (Stage.MINE_SEQUENCES, Stage.WIDEN_RETRIEVAL,
                              Stage.REPAIR_INPUTS) + _TERMINALS,
    Stage.MINE_SEQUENCES: (Stage.ANNOTATE_FAMILY, Stage.WIDEN_RETRIEVAL,
                           Stage.REPAIR_INPUTS) + _TERMINALS,
    Stage.ANNOTATE_FAMILY: (Stage.PREPARE_STRUCTURES, Stage.WIDEN_RETRIEVAL,
                            Stage.REPAIR_INPUTS) + _TERMINALS,
    Stage.PREPARE_STRUCTURES: (Stage.MODEL_COMPLEXES, Stage.WIDEN_RETRIEVAL,
                               Stage.REPAIR_INPUTS) + _TERMINALS,
    Stage.MODEL_COMPLEXES: (Stage.EVALUATE_CATALYSIS, Stage.WIDEN_RETRIEVAL,
                            Stage.REPAIR_INPUTS) + _TERMINALS,
    Stage.EVALUATE_CATALYSIS: (Stage.BATCH_APPROVAL, Stage.REPAIR_INPUTS,
                               Stage.WIDEN_RETRIEVAL) + _TERMINALS,
    # Going back: a repair returns to whichever stage failed.
    Stage.REPAIR_INPUTS: (
        Stage.CONFIRM_REACTION_SPEC, Stage.RETRIEVE_EVIDENCE,
        Stage.MINE_SEQUENCES, Stage.ANNOTATE_FAMILY, Stage.PREPARE_STRUCTURES,
        Stage.MODEL_COMPLEXES, Stage.EVALUATE_CATALYSIS, Stage.SELECT_BATCH,
        Stage.INGEST_RESULTS, Stage.LOCAL_ENGINEERING) + _TERMINALS,
    Stage.WIDEN_RETRIEVAL: (
        Stage.RETRIEVE_EVIDENCE, Stage.MINE_SEQUENCES, Stage.ANNOTATE_FAMILY,
        Stage.PREPARE_STRUCTURES, Stage.MODEL_COMPLEXES,
        Stage.EVALUATE_CATALYSIS, Stage.BATCH_APPROVAL) + _TERMINALS,
    Stage.BATCH_APPROVAL: (Stage.SELECT_BATCH,) + _TERMINALS,
    Stage.SELECT_BATCH: (Stage.AWAIT_RESULTS, Stage.REPAIR_INPUTS) + _TERMINALS,
    Stage.AWAIT_RESULTS: (Stage.INGEST_RESULTS,) + _TERMINALS,
    Stage.INGEST_RESULTS: (Stage.LOCAL_ENGINEERING, Stage.DIAGNOSE_NO_HIT,
                           Stage.REPAIR_INPUTS) + _TERMINALS,
    Stage.LOCAL_ENGINEERING: (Stage.DONE,) + _TERMINALS,
    Stage.DIAGNOSE_NO_HIT: (Stage.DONE, Stage.WIDEN_RETRIEVAL) + _TERMINALS,
}


#: Where a stage goes when its step produced usable output. Declared as data
#: so that "carry on with what exists" after an exhausted widening lands in
#: the same place a success would have, instead of jumping the protocol.
LINEAR_SUCCESSOR: dict[Stage, Stage] = {
    Stage.RETRIEVE_EVIDENCE: Stage.MINE_SEQUENCES,
    Stage.MINE_SEQUENCES: Stage.ANNOTATE_FAMILY,
    Stage.ANNOTATE_FAMILY: Stage.PREPARE_STRUCTURES,
    Stage.PREPARE_STRUCTURES: Stage.MODEL_COMPLEXES,
    Stage.MODEL_COMPLEXES: Stage.EVALUATE_CATALYSIS,
    Stage.EVALUATE_CATALYSIS: Stage.BATCH_APPROVAL,
    Stage.SELECT_BATCH: Stage.AWAIT_RESULTS,
}


class RunOutcome(str, enum.Enum):
    """How the run ended, in the words the operator needs."""

    COMPLETED = "completed"
    ESCALATED = "escalated"
    HALTED = "halted"
    AWAITING_HUMAN = "awaiting_human"
    AWAITING_RESULTS = "awaiting_results"


_STAGE_OUTCOME: dict[Stage, RunOutcome] = {
    Stage.DONE: RunOutcome.COMPLETED,
    Stage.ESCALATED: RunOutcome.ESCALATED,
    Stage.HALTED: RunOutcome.HALTED,
    Stage.AWAITING_HUMAN: RunOutcome.AWAITING_HUMAN,
    Stage.AWAITING_RESULTS: RunOutcome.AWAITING_RESULTS,
}


@dataclass
class StepAttempt:
    """One execution of one interface, as the controller saw it."""

    step_id: str
    stage: Stage
    interface: str
    attempt: int
    failure: FailureKind
    status: str
    started_at: str
    finished_at: str
    inputs_digest: str
    digest_stable: bool
    skipped: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id, "stage": self.stage.value,
            "interface": self.interface, "attempt": self.attempt,
            "failure": self.failure.value, "status": self.status,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "inputs_digest": self.inputs_digest,
            "digest_stable": self.digest_stable,
            "skipped": self.skipped, "message": self.message,
        }


@dataclass
class Escalation:
    """A failure a human has to take over, with why the controller stopped."""

    stage: Stage
    interface: str
    kind: FailureKind
    attempts: int
    reason: str
    blockers: tuple[str, ...] = ()
    at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage.value, "interface": self.interface,
                "kind": self.kind.value, "attempts": self.attempts,
                "reason": self.reason, "blockers": list(self.blockers),
                "at": self.at}


@dataclass
class ControllerHooks:
    """Where the run's judgement lives, kept out of the controller.

    Each hook answers a question the controller is deliberately unable to
    answer for itself. ``repair`` and ``widen`` return ``True`` when they
    actually changed something: a re-run after a no-op repair would reproduce
    the identical failure, so the controller escalates instead of looping.
    """

    #: Interface name -> keyword arguments for this run.
    arguments: Callable[["ResearchController", str], Mapping[str, Any]] | None = None
    #: Fix the inputs of a failed step. True if something changed.
    repair: Callable[["ResearchController", StepAttempt], bool] | None = None
    #: Broaden the search (more seeds, another database, a further family).
    widen: Callable[["ResearchController", StepAttempt], bool] | None = None
    #: What the operator must see to authorise the batch and its cost.
    batch_payload: Callable[["ResearchController"], Mapping[str, Any]] | None = None
    #: Whether the plate has come back.
    results_ready: Callable[["ResearchController"], bool] | None = None
    #: Override the branch decision after an otherwise successful step.
    branch: Callable[["ResearchController", ToolResult], Branch | None] | None = None
    #: Whether the ingested results contain a confirmed hit.
    hit_found: Callable[[ToolResult], bool] | None = None
    #: Told about an escalation after it is recorded. Observes only: it cannot
    #: undo the escalation, and a hook that raises is logged, not propagated,
    #: so a broken commentary step can never hide the failure it comments on.
    on_escalation: Callable[["ResearchController", Escalation], None] | None = None
    #: Told when a round ends (hit or no hit), before the run reaches DONE.
    #: Observes only, under the same rule.
    on_round_complete: Callable[["ResearchController"], None] | None = None


def _default_hit_found(result: ToolResult) -> bool:
    """Read the hit count the ingest step already computed.

    Reading a count out of an envelope is bookkeeping; deciding what counts as
    a hit is the ``functional_criteria_confirmed`` gate's business and
    ``ingest_results``'s implementation. The controller must never re-derive
    it, or there would be two definitions of a hit in one run.
    """
    counts = result.data.get("outcome_counts")
    if isinstance(counts, Mapping):
        return int(counts.get("confirmed_target_product", 0) or 0) > 0
    return False


#: Files hashed into a step's input digest, at most. Beyond this a directory
#: argument is marked unstable, which makes the step re-run: re-running a step
#: that did not need it costs time, while skipping one whose inputs changed
#: costs the result.
_MAX_HASHED_INPUT_FILES = 512


def _path_identity(path: Path) -> Any:
    """Identity of a path argument: what is in it, not where it is.

    Hashing the path string alone lets a step be skipped on resume after its
    input file was edited in place, because the only thing that changed is
    the bytes. The run then reports a step as reproduced from a cached result
    that the current inputs would not produce.

    A missing path is recorded as missing rather than as its string, so
    creating the file later invalidates the digest as it should.
    """
    text = str(path)
    try:
        if path.is_file():
            return {"path": text, "sha256": sha256_file(path)}
        if path.is_dir():
            files = sorted(q for q in path.rglob("*") if q.is_file())
            if len(files) > _MAX_HASHED_INPUT_FILES:
                return {"path": text, "__unstable__": "directory_too_large",
                        "n_files": len(files)}
            return {"path": text, "contents": [
                {"rel": str(q.relative_to(path)), "sha256": sha256_file(q)}
                for q in files]}
        return {"path": text, "state": "missing"}
    except OSError as exc:
        # Unreadable is not unchanged. Mark it unstable so the step re-runs
        # rather than being skipped on a digest that proves nothing.
        return {"path": text, "__unstable__": f"unreadable: {exc.__class__.__name__}"}


def _jsonable(value: Any) -> Any:
    """Render an argument for hashing, flagging anything with an unstable form.

    Returns the value itself where it is already canonical. A plain object
    whose ``repr`` carries its memory address is rendered as a marker rather
    than as that repr, because a digest containing ``0x7f...`` differs between
    processes and would make every resumed step look changed -- or, worse,
    look unchanged for the wrong reason.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return _path_identity(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_jsonable(v) for v in value]
        return sorted(items, key=canonical_json) if isinstance(
            value, (set, frozenset)) else items
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    text = repr(value)
    if " object at 0x" in text or " at 0x" in text:
        return {"__unstable__": type(value).__name__}
    return text


def _argument_digest(interface: str, arguments: Mapping[str, Any],
                     task_digest: str | None) -> tuple[str, bool]:
    """Digest of a step's inputs, and whether that digest is reproducible.

    The second element is what resume depends on. If any argument rendered to
    an unstable marker, the controller cannot prove the inputs are unchanged,
    so it re-runs the step rather than skipping it on a digest that happens to
    match.
    """
    rendered = {k: _jsonable(v) for k, v in sorted(arguments.items())}
    stable = "__unstable__" not in canonical_json(rendered)
    digest = sha256_obj({"interface": interface, "task": task_digest,
                         "arguments": rendered})
    return digest, stable


class ResearchController:
    """Walks the protocol, records every step, and knows when to stop.

    Not a planner that improvises: the stages and the legal moves between them
    are declared in :data:`TRANSITIONS`, so a run's path is auditable and a
    bug cannot invent a shortcut from modelling to ordering genes.
    """

    def __init__(
        self,
        ctx: RunContext,
        registry: InterfaceRegistry | None = None,
        queue: ApprovalQueue | None = None,
        hooks: ControllerHooks | None = None,
        *,
        arguments: Mapping[str, Mapping[str, Any]] | None = None,
        max_transitions: int = 200,
    ) -> None:
        self.ctx = ctx
        self.registry = registry if registry is not None else build_interface_registry()
        self.queue = queue if queue is not None else ApprovalQueue(
            ctx.workdir / "approvals.json", ctx.manifest)
        self.hooks = hooks or ControllerHooks()
        self.static_arguments: dict[str, dict[str, Any]] = {
            k: dict(v) for k, v in (arguments or {}).items()}
        self.max_transitions = max_transitions

        self.stage: Stage = Stage.START
        self.path: list[Stage] = [Stage.START]
        self.attempts: list[StepAttempt] = []
        self.results: dict[str, ToolResult] = {}
        self.escalations: list[Escalation] = []
        self.skipped_steps: list[str] = []
        self.exploration_candidates: list[str] = []
        self.human_actions: list[dict[str, Any]] = []
        self.suggested_actions: list[dict[str, Any]] = []
        self.unroutable_actions: list[dict[str, Any]] = []
        self.no_hit_diagnosis: Any = None
        #: Proposals a language model made (see harness/planner.py). Recorded
        #: here so the report shows them, and never acted on from here.
        self.llm_proposals: list[dict[str, Any]] = []
        self.outcome: RunOutcome | None = None
        self.stop_reason: str = ""

        self._attempt_counts: dict[str, int] = {}
        self._executed: set[str] = set()
        self._repair_target: Stage | None = None
        self._widen_target: Stage | None = None
        self._widenings: dict[Stage, int] = {}

    # -- small helpers -----------------------------------------------------
    def log(self, message: str) -> None:
        self.ctx.log(message)

    @property
    def manifest(self):
        return self.ctx.manifest

    def result_for(self, interface: str) -> ToolResult | None:
        """The latest envelope from one interface, for an argument provider."""
        return self.results.get(interface)

    def arguments_for(self, interface: str) -> dict[str, Any]:
        """Keyword arguments for one step: the hook first, then the static map."""
        if self.hooks.arguments is not None:
            supplied = self.hooks.arguments(self, interface)
            if supplied is not None:
                return dict(supplied)
        return dict(self.static_arguments.get(interface, {}))

    def budget_breaches(self) -> list[str]:
        """Cost keys whose recorded total has passed its ceiling.

        Reads :attr:`RunManifest.cost_total`, which the manifest accumulates
        from each step's provenance. The controller does not estimate what a
        step *will* cost -- it has no basis for that -- it stops once what was
        actually spent is over the line.
        """
        out: list[str] = []
        for key, ceiling in (self.ctx.policy.cost_ceiling or {}).items():
            spent = self.manifest.cost_total.get(key)
            if spent is None or ceiling is None:
                continue
            if spent > ceiling:
                out.append(f"{key}: {spent} recorded against a ceiling of {ceiling}")
        return out

    # -- transitions -------------------------------------------------------
    def goto(self, stage: Stage, reason: str = "") -> Stage:
        """Move to ``stage``, refusing any move the table does not declare."""
        allowed = TRANSITIONS.get(self.stage, ())
        if stage not in allowed and stage is not self.stage:
            raise ValueError(
                f"illegal transition {self.stage.value} -> {stage.value}; "
                f"declared moves are {[s.value for s in allowed]}. The control "
                f"flow is a declared graph so that a run's path can be audited")
        if reason:
            self.log(f"{self.stage.value} -> {stage.value}: {reason}")
        self.stage = stage
        self.path.append(stage)
        return stage

    # -- execution ---------------------------------------------------------
    def _step_id(self, stage: Stage, interface: str) -> str:
        return f"{stage.value}:{interface}"

    def _completed_record(self, step_id: str) -> dict[str, Any] | None:
        """The last manifest record for this step, if it finished usably."""
        for rec in reversed(self.manifest.steps):
            if rec.step_id != step_id:
                continue
            if rec.status in (Status.SUCCESS.value, Status.PARTIAL.value):
                return rec.to_dict()
            return None
        return None

    def execute(self, stage: Stage) -> ToolResult | None:
        """Run the interface for ``stage``, honouring resume and the retry caps.

        Returns the envelope, or ``None`` when the step was skipped on resume.
        Every executed attempt is appended to the manifest before any routing
        decision is made, so a crash between the call and the decision still
        leaves the evidence that the call happened.
        """
        interface_name = STAGE_INTERFACE[stage]
        iface: ScientificInterface = self.registry.get(interface_name)
        step_id = self._step_id(stage, interface_name)
        args = self.arguments_for(interface_name)
        digest, stable = _argument_digest(
            interface_name, args, self.manifest.task_input_sha256)

        previous = (None if step_id in self._executed
                    else self._completed_record(step_id))
        if previous is not None and stable:
            parameters = previous.get("provenance", {}).get("parameters", {})
            recorded = (previous.get("provenance", {})
                        .get("inputs_sha256", {})
                        .get("controller_step_inputs"))
            # Only a step the controller itself classified as clean may be
            # skipped. A record that was routed to widening or repair last time
            # is not "done", and a record written before this field existed
            # cannot prove either way, so it is re-run.
            # A step may only be skipped if its result can be handed to the
            # stages that read it. Skipping without restoring the payload is
            # how a round that found a hit reaches the no-hit diagnosis after
            # a restart: the step is marked done, the downstream stage reads
            # nothing, and nothing in the record says the conclusion changed
            # because of a resume rather than because of the data.
            restorable = bool(previous.get("data_restorable"))
            if recorded == digest and \
                    parameters.get("controller_failure") == FailureKind.NONE.value \
                    and not restorable:
                self.log(f"resume: re-running {step_id}; its result was not "
                         f"stored in a restorable form, and a skipped step "
                         f"whose output cannot be restored would silently "
                         f"change the branch taken")
            elif recorded == digest and \
                    parameters.get("controller_failure") == FailureKind.NONE.value:
                restored = ToolResult(
                    status=Status(previous.get("status", Status.SUCCESS.value)),
                    data=dict(previous.get("data") or {}),
                    message=previous.get("message", ""))
                self.results[interface_name] = restored
                self.skipped_steps.append(step_id)
                self.attempts.append(StepAttempt(
                    step_id=step_id, stage=stage, interface=interface_name,
                    attempt=0, failure=FailureKind.NONE,
                    status=previous.get("status", ""),
                    started_at=previous.get("started_at", ""),
                    finished_at=previous.get("finished_at", ""),
                    inputs_digest=digest, digest_stable=True, skipped=True,
                    message="resumed: inputs unchanged"))
                self.log(f"resume: skipping {step_id}; input hash unchanged")
                # Marked as seen so that re-entering this stage later in the
                # same run -- after a repair, say -- actually re-runs it.
                self._executed.add(step_id)
                return None

        attempt = self._attempt_counts.get(step_id, 0) + 1
        self._attempt_counts[step_id] = attempt
        self._executed.add(step_id)
        started = utc_now()
        result = iface.run(self.ctx, **args)

        failure = classify_failure(result)
        prov = result.provenance
        if prov is not None:
            prov.inputs_sha256.setdefault("controller_step_inputs", digest)
            prov.parameters.setdefault("controller_attempt", attempt)
            prov.parameters.setdefault("controller_stage", stage.value)
            prov.parameters.setdefault("controller_failure", failure.value)
        self.manifest.record(step_id, interface_name, result, started)
        self.results[interface_name] = result

        self.attempts.append(StepAttempt(
            step_id=step_id, stage=stage, interface=interface_name,
            attempt=attempt, failure=failure, status=result.status.value,
            started_at=started, finished_at=utc_now(), inputs_digest=digest,
            digest_stable=stable, message=result.message))
        self._consume_next_actions(result, step_id)
        return result

    # -- next actions ------------------------------------------------------
    def _consume_next_actions(self, result: ToolResult, step_id: str) -> None:
        """Route the envelope's follow-ups; execute none of them directly.

        A ``requires_human`` action goes to the approval queue even when it
        names something the harness could technically do, because the whole
        reason it is marked is that a person must decide. A non-human action
        naming a registered interface is recorded as a suggestion -- the
        controller still has to put it in the right place in the state
        machine, and executing it straight off the envelope would skip the
        gates that sit between here and there.
        """
        for action in result.next_actions:
            entry = {"step_id": step_id, "action": action.action,
                     "rationale": action.rationale, "params": dict(action.params)}
            if action.requires_human:
                self._queue_human_action(action, entry)
            elif action.action in self.registry:
                self.suggested_actions.append(entry)
            else:
                entry["note"] = (
                    "no registered interface and not marked for a human; "
                    "recorded for the operator rather than guessed at")
                self.unroutable_actions.append(entry)

    def _queue_human_action(self, action: NextAction,
                            entry: dict[str, Any]) -> None:
        gate = str(action.params.get("gate") or "")
        if gate in APPROVAL_GATES:
            request = self.queue.request(
                gate, requested_by=entry["step_id"], detail=action.rationale,
                payload=dict(action.params), kind=RequestKind.GATE)
        else:
            request = self.queue.request(
                action.action, requested_by=entry["step_id"],
                detail=action.rationale, payload=dict(action.params),
                kind=RequestKind.OPERATOR_TASK)
        entry["request_id"] = request.request_id
        entry["kind"] = request.kind.value
        self.human_actions.append(entry)

    # -- failure handling --------------------------------------------------
    def _effective_attempts(self, kind: FailureKind) -> int:
        """Attempts allowed for this failure kind under both caps.

        The policy's per-kind cap and ``ctx.policy.max_retries`` both apply,
        and the tighter wins. An operator who sets ``max_retries=0`` gets a
        single attempt at everything; one who sets it to 50 still does not get
        50 attempts at a missing binary.
        """
        rule = RETRY_POLICY[kind]
        ceiling = 1 + max(0, int(self.ctx.policy.max_retries))
        return max(1, min(rule.max_attempts, ceiling))

    def escalate(self, stage: Stage, interface: str, kind: FailureKind,
                 attempts: int, reason: str,
                 blockers: Iterable[str] = ()) -> Stage:
        """Hand the failure to a human and stop this branch of the run."""
        esc = Escalation(stage=stage, interface=interface, kind=kind,
                         attempts=attempts, reason=reason,
                         blockers=tuple(blockers))
        self.escalations.append(esc)
        self.manifest.notes.append(
            f"{esc.at} ESCALATION at {stage.value} ({interface}): {reason}")
        self.queue.request(
            "operator_review", requested_by=self._step_id(stage, interface),
            detail=reason, payload=esc.to_dict(),
            kind=RequestKind.OPERATOR_TASK)
        self.stop_reason = reason
        self._notify(self.hooks.on_escalation, esc)
        return self.goto(Stage.ESCALATED, reason)

    def _notify(self, hook: Callable[..., None] | None, *args: Any) -> None:
        """Call an observer hook; a failure in it is logged and nothing more."""
        if hook is None:
            return
        try:
            hook(self, *args)
        except Exception as exc:                      # noqa: BLE001
            self.manifest.notes.append(
                f"{utc_now()} an observer hook raised {type(exc).__name__}: "
                f"{exc}; the run is unaffected")

    def _handle_failure(self, stage: Stage, result: ToolResult) -> Stage:
        """Apply the retry policy for one failed step and return the next stage."""
        interface = STAGE_INTERFACE[stage]
        step_id = self._step_id(stage, interface)
        kind = classify_failure(result)
        # ``.get`` and not ``[]``: a kind added to the enum without a rule
        # would otherwise raise here, turning a classified failure into an
        # unhandled exception at the exact point the machine is supposed to
        # be deciding what to do about failures.
        rule = RETRY_POLICY.get(kind, RETRY_POLICY[FailureKind.UNCLASSIFIED])
        attempts = self._attempt_counts.get(step_id, 1)
        blockers = tuple(f"{f.code}: {f.message}" for f in result.blockers)
        if not blockers:
            # A step can fail with no blocking flag -- a structure cache miss
            # that only sets the status. Recording an empty blocker list would
            # hand the operator an escalation that names nothing.
            blockers = (describe_failure(result),)
        if kind is FailureKind.UNCLASSIFIED:
            return self.escalate(stage, interface, kind, attempts,
                                 f"{rule.why}. {describe_failure(result)}",
                                 blockers)

        breaches = self.budget_breaches()
        if breaches:
            return self.escalate(
                stage, interface, kind, attempts,
                "cost ceiling reached, so no further attempt was made ("
                + "; ".join(breaches) + ")", blockers)

        if rule.action == "await_human":
            self.stop_reason = (
                f"{interface} is waiting on a human decision: "
                + "; ".join(blockers or (result.message,)))
            return self.goto(Stage.AWAITING_HUMAN, self.stop_reason)

        if rule.action == "escalate":
            return self.escalate(stage, interface, kind, attempts, rule.why,
                                 blockers)

        if rule.action == "widen_inputs":
            # Thin evidence is a finding, not a fault: the widen stage decides
            # between another pass and carrying the candidates as exploration
            # probes, so the attempt cap is applied there rather than here.
            if Stage.WIDEN_RETRIEVAL not in TRANSITIONS.get(stage, ()):
                return self.escalate(
                    stage, interface, kind, attempts,
                    f"{interface} found too little to act on and the search "
                    f"cannot be widened from {stage.value}", blockers)
            self._widen_target = stage
            return self.goto(Stage.WIDEN_RETRIEVAL,
                             f"{interface} found too little to act on")

        if attempts >= self._effective_attempts(kind):
            return self.escalate(
                stage, interface, kind, attempts,
                f"{interface} failed {attempts} time(s) with {kind.value}; "
                f"{rule.why}", blockers)

        if Stage.REPAIR_INPUTS not in TRANSITIONS.get(stage, ()):
            return self.escalate(
                stage, interface, kind, attempts,
                f"{interface} reported a repairable error, but no repair "
                f"transition is declared from {stage.value}", blockers)
        self._repair_target = stage
        return self.goto(Stage.REPAIR_INPUTS,
                         f"{interface} reported an input or mapping error")

    # -- stage handlers ----------------------------------------------------
    def _run_step_stage(self, stage: Stage, on_success: Stage) -> Stage:
        """The common shape: execute, branch on the envelope, advance or recover."""
        result = self.execute(stage)
        if result is None:                       # skipped on resume
            return self.goto(on_success, "resumed")
        branch = self.branch_of(result)
        if branch is Branch.SUFFICIENT_SUPPORT:
            return self.goto(on_success, f"{STAGE_INTERFACE[stage]} usable")
        return self._handle_failure(stage, result)

    def branch_of(self, result: ToolResult) -> Branch:
        """Three-way branch for an envelope, with the hook given the last word."""
        if self.hooks.branch is not None:
            override = self.hooks.branch(self, result)
            if override is not None:
                return override
        kind = classify_failure(result)
        if kind is FailureKind.NONE:
            return Branch.SUFFICIENT_SUPPORT
        if kind is FailureKind.INPUT_ERROR:
            return Branch.INPUT_OR_MAPPING_ERROR
        if kind is FailureKind.INSUFFICIENT_EVIDENCE:
            return Branch.INSUFFICIENT_EVIDENCE
        return Branch.BLOCKED

    def _stage_confirm_reaction_spec(self) -> Stage:
        result = self.execute(Stage.CONFIRM_REACTION_SPEC)
        if result is not None:
            branch = self.branch_of(result)
            if branch is Branch.INPUT_OR_MAPPING_ERROR:
                return self._handle_failure(Stage.CONFIRM_REACTION_SPEC, result)
            if branch is Branch.BLOCKED and not self.queue.pending(REACTION_GATE):
                return self._handle_failure(Stage.CONFIRM_REACTION_SPEC, result)
        if self.queue.is_granted(REACTION_GATE):
            return self.goto(Stage.RETRIEVE_EVIDENCE,
                             "reaction spec confirmed by a recorded decision")
        self.queue.request(
            REACTION_GATE, requested_by="controller",
            detail=("confirm the substrate structure, the product structure "
                    "and the target configuration before any search starts"),
            payload={"task_id": self.ctx.task.task_id,
                     "unresolved": self.ctx.task.unresolved_for(REACTION_GATE)})
        return self.goto(Stage.AWAIT_REACTION_APPROVAL,
                         "waiting for the reaction spec to be confirmed")

    def _stage_await_reaction_approval(self) -> Stage:
        if self.queue.is_granted(REACTION_GATE):
            return self.goto(Stage.RETRIEVE_EVIDENCE, "reaction spec confirmed")
        if self.queue.is_denied(REACTION_GATE):
            self.stop_reason = "the operator rejected the reaction spec"
            return self.goto(Stage.HALTED, self.stop_reason)
        self.stop_reason = (
            "the reaction spec has not been confirmed; nothing downstream may "
            "run, because every later step would be evidence about an "
            "unconfirmed molecule")
        return self.goto(Stage.AWAITING_HUMAN, self.stop_reason)

    def _stage_evaluate_catalysis(self) -> Stage:
        result = self.execute(Stage.EVALUATE_CATALYSIS)
        if result is None:
            return self.goto(Stage.BATCH_APPROVAL, "resumed")
        branch = self.branch_of(result)
        if branch is Branch.SUFFICIENT_SUPPORT:
            return self.goto(Stage.BATCH_APPROVAL,
                             "catalytic complex evaluation produced usable support")
        return self._handle_failure(Stage.EVALUATE_CATALYSIS, result)

    def _stage_repair(self) -> Stage:
        """Re-run the affected step after the inputs were repaired.

        The controller does not repair anything itself. If no hook is wired,
        or the hook reports it changed nothing, re-running would reproduce the
        identical failure, so the run escalates instead of looping.
        """
        target = self._repair_target
        if target is None:
            return self.escalate(Stage.REPAIR_INPUTS, "controller",
                                 FailureKind.INTERNAL_ERROR, 0,
                                 "repair requested with no failed step recorded")
        last = self.attempts[-1] if self.attempts else None
        if self.hooks.repair is None or last is None:
            return self.escalate(
                target, STAGE_INTERFACE[target], FailureKind.INPUT_ERROR,
                self._attempt_counts.get(self._step_id(target,
                                                       STAGE_INTERFACE[target]), 1),
                ("the inputs need repair and this run has no repair hook; the "
                 "controller will not invent a replacement input"))
        changed = bool(self.hooks.repair(self, last))
        if not changed:
            return self.escalate(
                target, STAGE_INTERFACE[target], FailureKind.INPUT_ERROR,
                self._attempt_counts.get(last.step_id, 1),
                ("the repair hook changed nothing, so re-running would "
                 "reproduce the same error"))
        self._repair_target = None
        return self.goto(target, "inputs repaired; re-running the affected step")

    def _stage_widen(self) -> Stage:
        """Widen the inputs, or carry what is there as exploration candidates.

        Widening means more seeds, another database, a further family -- never
        a lowered threshold, which is the same step with the evidence removed.
        """
        target = self._widen_target or Stage.RETRIEVE_EVIDENCE
        count = self._widenings.get(target, 0)
        last = self.attempts[-1] if self.attempts else None
        kind = FailureKind.INSUFFICIENT_EVIDENCE
        allowed = self._effective_attempts(kind)
        changed = False
        if self.hooks.widen is not None and last is not None and count < allowed - 1:
            changed = bool(self.hooks.widen(self, last))
        if changed:
            self._widenings[target] = count + 1
            self._widen_target = None
            return self.goto(target, "inputs widened; re-running the search")

        reason = ("the search was not widened further; what exists is carried "
                  "forward as exploration probes, which is a declared use of "
                  "batch slots rather than a claim that the evidence is "
                  "sufficient")
        self.exploration_candidates.append(target.value)
        self._widen_target = None
        self.manifest.notes.append(f"{utc_now()} {reason} (at {target.value})")
        onward = LINEAR_SUCCESSOR.get(target)
        if onward is None:
            return self.escalate(
                target, STAGE_INTERFACE.get(target, "controller"), kind, count,
                "the search cannot be widened further and this stage has no "
                "declared successor to carry the shortfall into")
        return self.goto(onward, reason)

    def gate_payload(self, gate: str) -> dict[str, Any]:
        """What this controller would ask the operator to decide about.

        Public so that an operator interface, and a test, can authorise the
        same thing the controller will later check. A decision recorded
        against a different payload is not a decision about this work, and
        the gate is right to refuse it, so the way to pre-authorise is to
        approve the real payload rather than to loosen the check.
        """
        if gate == BATCH_GATE:
            return self._batch_gate_payload()
        return {"task_id": self.ctx.task.task_id, "gate": gate}

    def _batch_gate_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"task_id": self.ctx.task.task_id}
        if self.hooks.batch_payload is not None:
            payload.update(dict(self.hooks.batch_payload(self)))
        else:
            payload["note"] = (
                "no batch cost payload was supplied; the operator is being "
                "asked to authorise a batch whose size and price this run "
                "cannot state")
        return payload

    def _stage_batch_approval(self) -> Stage:
        payload: dict[str, Any] = self._batch_gate_payload()
        # The payload is what the operator is shown and therefore what they
        # decide about. Checking the gate alone would accept a grant given for
        # a different batch, at a different size and a different cost.
        self._batch_payload = dict(payload)
        if self.queue.is_granted(BATCH_GATE, payload):
            return self.goto(Stage.SELECT_BATCH,
                             "this batch authorised by a recorded decision")
        if self.queue.is_denied(BATCH_GATE, payload):
            self.stop_reason = "the operator declined to authorise this batch"
            return self.goto(Stage.HALTED, self.stop_reason)
        self.queue.request(
            BATCH_GATE, requested_by="controller",
            detail="authorise this batch and its cost", payload=payload)
        self.stop_reason = (
            "batch authorisation is pending; no construct may be selected "
            "until a named person approves this batch")
        return self.goto(Stage.AWAITING_HUMAN, self.stop_reason)

    def _stage_select_batch(self) -> Stage:
        """Compose the batch -- behind the hard approval block.

        :func:`~eagent.harness.approval.guard_batch_selection` is called here
        rather than relying on the interface's own ``required_approvals``,
        because the interface check also accepts the task-file flag. This one
        requires a grant with an actor in the manifest, so there is no path
        from "somebody edited the YAML" to "genes were ordered".
        """
        try:
            guard_batch_selection(self.manifest, self.queue,
                                  payload=getattr(self, "_batch_payload", None))
        except ApprovalRequiredError as exc:
            self.stop_reason = str(exc)
            return self.goto(Stage.AWAITING_HUMAN, self.stop_reason)
        return self._run_step_stage(Stage.SELECT_BATCH, Stage.AWAIT_RESULTS)

    def _stage_await_results(self) -> Stage:
        ready = bool(self.hooks.results_ready(self)) \
            if self.hooks.results_ready is not None else False
        if ready:
            return self.goto(Stage.INGEST_RESULTS, "assay results are available")
        self.stop_reason = (
            "the batch is out; the run resumes when the assay results arrive")
        return self.goto(Stage.AWAITING_RESULTS, self.stop_reason)

    def _stage_ingest_results(self) -> Stage:
        if not self.queue.is_granted(CRITERIA_GATE):
            self.queue.request(
                CRITERIA_GATE, requested_by="controller",
                detail=("confirm which experimental result counts as "
                        "supporting a functional claim, before the data are "
                        "read"),
                payload={"task_id": self.ctx.task.task_id})
            self.stop_reason = (
                "the hit definition has not been confirmed; reading the plate "
                "first would let the criterion be fitted to the data")
            return self.goto(Stage.AWAITING_HUMAN, self.stop_reason)
        result = self.execute(Stage.INGEST_RESULTS)
        if result is None:
            return self.goto(Stage.DIAGNOSE_NO_HIT, "resumed")
        if self.branch_of(result) is not Branch.SUFFICIENT_SUPPORT:
            return self._handle_failure(Stage.INGEST_RESULTS, result)
        decide = self.hooks.hit_found or _default_hit_found
        if decide(result):
            return self.goto(Stage.LOCAL_ENGINEERING,
                             "a confirmed hit was reported; engineering locally")
        self.no_hit_diagnosis = result.data.get("no_hit_diagnosis")
        return self.goto(Stage.DIAGNOSE_NO_HIT,
                         "no confirmed hit; reading the failure-cause diagnosis")

    def _stage_diagnose_no_hit(self) -> Stage:
        """Record the diagnosis the ingest step produced.

        The controller reads it; it does not compute it. Deciding whether a
        round failed because of expression, because of the assay, or because
        the family hypothesis was wrong is a scientific judgement that lives in
        ``ingest_results`` and in its templates.
        """
        if self.no_hit_diagnosis is None:
            ingest = self.results.get("ingest_results")
            if ingest is not None:
                self.no_hit_diagnosis = ingest.data.get("no_hit_diagnosis")
        if self.no_hit_diagnosis is None:
            self.manifest.notes.append(
                f"{utc_now()} no hit, and the ingest step reported no "
                f"failure-cause diagnosis; the cause is unknown rather than "
                f"'the enzymes do not work'")
        else:
            self.manifest.notes.append(
                f"{utc_now()} no-hit diagnosis recorded from ingest_results")
        self.stop_reason = "round complete: no confirmed hit, diagnosis recorded"
        self._notify(self.hooks.on_round_complete)
        return self.goto(Stage.DONE, self.stop_reason)

    def _stage_local_engineering(self) -> Stage:
        result = self.execute(Stage.LOCAL_ENGINEERING)
        if result is not None and self.branch_of(result) is not \
                Branch.SUFFICIENT_SUPPORT:
            return self._handle_failure(Stage.LOCAL_ENGINEERING, result)
        self.stop_reason = "round complete: hit confirmed, variants proposed"
        self._notify(self.hooks.on_round_complete)
        return self.goto(Stage.DONE, self.stop_reason)

    # -- driver ------------------------------------------------------------
    def step(self) -> Stage:
        """Advance one stage. Returns the stage the run is now in."""
        stage = self.stage
        if stage in TERMINAL_STAGES:
            return stage
        if stage is Stage.START:
            return self.goto(Stage.CONFIRM_REACTION_SPEC, "run started")
        handler = {
            Stage.CONFIRM_REACTION_SPEC: self._stage_confirm_reaction_spec,
            Stage.AWAIT_REACTION_APPROVAL: self._stage_await_reaction_approval,
            Stage.EVALUATE_CATALYSIS: self._stage_evaluate_catalysis,
            Stage.REPAIR_INPUTS: self._stage_repair,
            Stage.WIDEN_RETRIEVAL: self._stage_widen,
            Stage.BATCH_APPROVAL: self._stage_batch_approval,
            Stage.SELECT_BATCH: self._stage_select_batch,
            Stage.AWAIT_RESULTS: self._stage_await_results,
            Stage.INGEST_RESULTS: self._stage_ingest_results,
            Stage.DIAGNOSE_NO_HIT: self._stage_diagnose_no_hit,
            Stage.LOCAL_ENGINEERING: self._stage_local_engineering,
        }.get(stage)
        if handler is not None:
            return handler()
        linear = {
            Stage.RETRIEVE_EVIDENCE: Stage.MINE_SEQUENCES,
            Stage.MINE_SEQUENCES: Stage.ANNOTATE_FAMILY,
            Stage.ANNOTATE_FAMILY: Stage.PREPARE_STRUCTURES,
            Stage.PREPARE_STRUCTURES: Stage.MODEL_COMPLEXES,
            Stage.MODEL_COMPLEXES: Stage.EVALUATE_CATALYSIS,
        }[stage]
        return self._run_step_stage(stage, linear)

    def run(self, start: Stage | None = None) -> RunOutcome:
        """Walk the machine until it reaches a terminal stage.

        The transition cap is a liveness guard, not a budget: a controller that
        oscillates between two stages is a bug, and it must surface as an
        escalation rather than as a run that never returns.
        """
        if start is not None:
            self.stage = start
            self.path.append(start)
        moves = 0
        while self.stage not in TERMINAL_STAGES:
            moves += 1
            if moves > self.max_transitions:
                self.escalate(
                    self.stage, STAGE_INTERFACE.get(self.stage, "controller"),
                    FailureKind.INTERNAL_ERROR, moves,
                    f"the controller made {moves} transitions without "
                    f"terminating; this is a control-flow defect, not a long run")
                break
            self.step()
        self.outcome = _STAGE_OUTCOME[self.stage]
        self.manifest.notes.append(
            f"{utc_now()} run ended in {self.stage.value}: "
            f"{self.stop_reason or 'no reason recorded'}")
        return self.outcome

    # -- reporting ---------------------------------------------------------
    def report(self) -> dict[str, Any]:
        """Everything the operator needs to see what the run did and did not do."""
        return {
            "stage": self.stage.value,
            "outcome": self.outcome.value if self.outcome else None,
            "stop_reason": self.stop_reason,
            "path": [s.value for s in self.path],
            "attempts": [a.to_dict() for a in self.attempts],
            "skipped_steps": list(self.skipped_steps),
            "escalations": [e.to_dict() for e in self.escalations],
            "pending_approvals": [r.to_dict() for r in self.queue.pending()],
            "human_actions": list(self.human_actions),
            "suggested_actions": list(self.suggested_actions),
            "unroutable_actions": list(self.unroutable_actions),
            "exploration_candidates": list(self.exploration_candidates),
            "cost_total": dict(self.manifest.cost_total),
            "budget_breaches": self.budget_breaches(),
            "no_hit_diagnosis": self.no_hit_diagnosis,
            "llm_proposals": list(self.llm_proposals),
        }

    def write_report(self, path: str | Path | None = None) -> Path:
        """Persist the run report next to the manifest."""
        target = Path(path) if path is not None \
            else self.ctx.path("controller_report.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.report(), indent=2,
                                     ensure_ascii=False, default=str),
                          encoding="utf-8")
        return target
