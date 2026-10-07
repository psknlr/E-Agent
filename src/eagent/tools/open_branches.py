"""Two branches the protocol does not run -- open function discovery and de novo design -- as seams that refuse.

WHY THESE ARE SEAMS AND NOT FEATURES
====================================
The protocol is built for a *known* reaction: a substrate and a product are
fixed and confirmed by a person, and the question is which enzymes do it. Two
neighbouring questions are tempting and different in kind:

* **Open function discovery** -- given an enzyme, *what does it convert?* The
  reaction is not fixed in advance; it is the output. A predicted product is a
  hypothesis about annotation transfer or a learned model, and nothing in the
  pipeline can confirm it. It cannot be allowed to fill the reaction spec, whose
  confirmation is a human gate.
* **De novo design** -- *build an enzyme that does it.* The sequence has no
  accession, no homologue and no literature. Everything known about it is
  computational: ``EvidenceStrength.COMPUTATIONAL_CONSTRUCT``, the lowest rank.

Neither has an installed implementation here, and this module does not pretend
otherwise: the defaults raise :class:`~eagent.errors.ToolUnavailableError` with
an install hint. What it provides is the part that *can* be built before the
tools exist and that is expensive to retrofit: the conditions under which such a
branch may run at all, and the label its outputs must carry afterwards.

WHAT MUST HOLD BEFORE A BRANCH RUNS
===================================
:func:`authorize_branch` checks, and reports *every* failure rather than the
first, so an operator sees the whole list in one pass:

1. the reaction spec is confirmed by a person -- the target chemistry is fixed
   before anything is generated for it;
2. the licence of every artefact the branch uses -- code, weights, inputs,
   outputs -- permits this run (a reported non-commercial weights licence blocks
   a commercial run even when the code is Apache 2.0);
3. a workflow that would *modify and redistribute* third-party code has a
   derivative-works permission (a no-derivatives licence refuses it);
4. a remote generator is not called while the network is off, and no sequence
   reaches it without an authorisation;
5. GPU-model use is allowed when the generator has weights;
6. a named person approved *this* branch for *this* input -- the grant is bound
   to a payload, so approval for one design campaign does not release another.

WHAT THE OUTPUTS MAY CLAIM
==========================
Every output is a :class:`BranchProposal` whose evidence ceiling is fixed by the
branch kind, whatever the generator reports about its own confidence. A design
becomes a :class:`~eagent.schemas.Candidate` whose provenance says it was
designed (``source_database="de_novo:<generator>"``, ``search_method=
"de_novo_design"``, no accession). :func:`is_high_evidence_eligible` keeps it
out of the batch's high-evidence role -- a slot whose label says "evidence" --
while leaving it eligible for the probe and diversity roles, which are
exploration by definition.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from ..connectors.base import AccessPolicy
from ..context import RunContext
from ..errors import EAgentError, LicenseError, ToolUnavailableError
from ..harness.approval import REACTION_GATE, ApprovalQueue, RequestKind
from ..provenance import sha256_obj
from ..schemas import (
    Candidate, EvidenceStrength, FamilyAnnotation, SequenceRecord,
)
from .model_complexes import (
    ToolRegistry, check_license, check_modification,
)

__all__ = [
    "BranchKind",
    "BRANCH_CEILING",
    "BRANCH_FORBIDDEN_CLAIMS",
    "BranchRefusedError",
    "BranchRequest",
    "BranchAuthorization",
    "authorize_branch",
    "branch_gate",
    "FunctionHypothesis",
    "DiscoveryJob",
    "DesignJob",
    "RawDesign",
    "OpenFunctionPredictor",
    "DeNovoGenerator",
    "UnavailableOpenFunctionPredictor",
    "UnavailableDeNovoGenerator",
    "BranchProposal",
    "proposals_from_designs",
    "design_candidate",
    "is_high_evidence_eligible",
]


class BranchKind(str, enum.Enum):
    OPEN_FUNCTION_DISCOVERY = "open_function_discovery"
    DE_NOVO_DESIGN = "de_novo_design"


#: The strongest claim a branch's output may make, fixed by the branch and not
#: by what the generator says about itself.
BRANCH_CEILING: Mapping[BranchKind, EvidenceStrength] = {
    BranchKind.OPEN_FUNCTION_DISCOVERY: EvidenceStrength.ANNOTATION_ONLY,
    BranchKind.DE_NOVO_DESIGN: EvidenceStrength.COMPUTATIONAL_CONSTRUCT,
}

BRANCH_FORBIDDEN_CLAIMS: Mapping[BranchKind, tuple[str, ...]] = {
    BranchKind.OPEN_FUNCTION_DISCOVERY: (
        "that the enzyme catalyses the proposed reaction",
        "that the proposed reaction may stand in for the confirmed reaction spec",
        "that a model score is a probability of activity",
    ),
    BranchKind.DE_NOVO_DESIGN: (
        "that the design expresses, folds or is stable",
        "that the design is active on any substrate",
        "that a generator's confidence is evidence of function",
    ),
}


class BranchRefusedError(EAgentError):
    """A branch was refused; ``reasons`` lists every failed condition."""

    def __init__(self, kind: BranchKind, reasons: Sequence[str]) -> None:
        self.kind = kind
        self.reasons = tuple(reasons)
        super().__init__(f"{kind.value} refused: " + " | ".join(self.reasons))


def branch_gate(kind: BranchKind) -> str:
    """The operator-task key an approval for this branch is recorded under.

    An operator task and not a fourth approval gate: the three gates are the
    protocol's, and ``ApprovalQueue.request`` refuses to invent another.
    """
    return f"open_branch:{kind.value}"


# ==========================================================================
# authorisation
# ==========================================================================

@dataclass(frozen=True)
class BranchRequest:
    """Everything a person is asked to approve, and everything the checks need."""

    kind: BranchKind
    generator: str
    description: str
    registry_keys: tuple[str, ...]
    inputs_sha256: str
    uses_model_weights: bool = True
    runs_remotely: bool = False
    modifies_third_party_code: bool = False
    #: Sequence-shaped strings this branch would send off the machine.
    outbound_sequences: tuple[str, ...] = ()

    def payload(self) -> dict[str, Any]:
        """What the approval is *about*. A grant covers this and nothing else."""
        return {"branch": self.kind.value, "generator": self.generator,
                "description": self.description,
                "registry_keys": list(self.registry_keys),
                "inputs_sha256": self.inputs_sha256,
                "runs_remotely": self.runs_remotely,
                "modifies_third_party_code": self.modifies_third_party_code}


@dataclass(frozen=True)
class BranchAuthorization:
    kind: BranchKind
    payload_sha256: str
    approved_by: str
    licence_facts: Mapping[str, Any]
    disclosure_notes: tuple[str, ...]
    evidence_ceiling: EvidenceStrength
    forbidden_claims: tuple[str, ...]


def authorize_branch(request: BranchRequest, *, ctx: RunContext,
                     registry: ToolRegistry, queue: ApprovalQueue,
                     access_policy: AccessPolicy | None = None
                     ) -> BranchAuthorization:
    """Raise :class:`BranchRefusedError` listing every unmet condition, or authorise.

    A request for approval is queued as a side effect when -- and only when --
    every *other* condition holds, so a person is not asked to approve a branch
    the licences already forbid.
    """
    reasons: list[str] = []
    notes: list[str] = []
    facts: Mapping[str, Any] = {}

    if not queue.is_granted(REACTION_GATE):
        reasons.append(
            "the reaction spec has not been confirmed by a person; the target "
            "chemistry must be fixed before anything is generated for it")

    try:
        facts = check_license(
            registry, list(request.registry_keys),
            ctx.policy.allow_commercial_use, f"{request.kind.value} "
            f"generator '{request.generator}'",
            uses_model_weights=request.uses_model_weights)
    except LicenseError as exc:
        reasons.append(f"licence: {exc}")
    except EAgentError as exc:
        reasons.append(f"tool registry: {exc}")

    if request.modifies_third_party_code:
        try:
            check_modification(registry, list(request.registry_keys),
                               f"{request.kind.value} generator "
                               f"'{request.generator}'")
        except LicenseError as exc:
            reasons.append(f"licence: {exc}")

    if request.runs_remotely:
        if not ctx.policy.allow_network:
            reasons.append(
                f"{request.generator} runs off this machine and "
                f"ctx.policy.allow_network is False; nothing would be sent")
        else:
            policy = access_policy or AccessPolicy.from_execution_policy(
                ctx.policy)
            try:
                notes.extend(policy.check_outbound(
                    request.generator, list(request.outbound_sequences)))
            except EAgentError as exc:
                reasons.append(f"disclosure: {exc}")
    if request.uses_model_weights and not ctx.policy.allow_gpu_models:
        reasons.append("ctx.policy.allow_gpu_models is False, so no "
                       "weights-based generator may run")

    gate = branch_gate(request.kind)
    payload = request.payload()
    if not reasons:
        if queue.is_granted(gate, payload):
            granted = queue.latest_for(gate, payload)
            return BranchAuthorization(
                kind=request.kind, payload_sha256=sha256_obj(payload),
                approved_by=granted.actor if granted else "",
                licence_facts=facts, disclosure_notes=tuple(notes),
                evidence_ceiling=BRANCH_CEILING[request.kind],
                forbidden_claims=BRANCH_FORBIDDEN_CLAIMS[request.kind])
        queue.request(
            gate, requested_by="open_branches",
            detail=(f"approve running {request.kind.value} with "
                    f"'{request.generator}': {request.description}"),
            payload=payload, kind=RequestKind.OPERATOR_TASK)
        reasons.append(
            "no named person has approved this branch for this input; a "
            "request was queued")
    raise BranchRefusedError(request.kind, reasons)


# ==========================================================================
# the seams
# ==========================================================================

@dataclass(frozen=True)
class FunctionHypothesis:
    """A predicted conversion for one enzyme. A hypothesis, never a spec."""

    sequence_id: str
    proposed_reaction: str            # a Rhea id or a reaction SMILES
    score: float | None
    score_meaning: str                # what the number is; never "probability" unasked
    basis: str


@dataclass(frozen=True)
class DiscoveryJob:
    sequence_ids: tuple[str, ...]
    sequences: tuple[str, ...]
    seed: int
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawDesign:
    """What a generator returns for one design, before it is labelled."""

    design_id: str
    sequence: str
    generator_confidence: Mapping[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass(frozen=True)
class DesignJob:
    task_id: str
    reaction_spec_sha256: str
    n_designs: int
    seed: int
    params: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class OpenFunctionPredictor(Protocol):
    name: str
    version: str
    registry_keys: tuple[str, ...]
    uses_model_weights: bool
    runs_remotely: bool

    def is_available(self) -> bool: ...

    def predict(self, job: DiscoveryJob) -> Sequence[FunctionHypothesis]: ...


@runtime_checkable
class DeNovoGenerator(Protocol):
    name: str
    version: str
    registry_keys: tuple[str, ...]
    uses_model_weights: bool
    runs_remotely: bool

    def is_available(self) -> bool: ...

    def generate(self, job: DesignJob) -> Sequence[RawDesign]: ...


@dataclass(frozen=True)
class UnavailableOpenFunctionPredictor:
    """Absent, and loudly so. No heuristic stands in for a function predictor."""

    name: str = "open_function_predictor"
    version: str = "absent"
    registry_keys: tuple[str, ...] = ()
    uses_model_weights: bool = True
    runs_remotely: bool = False
    install_hint: str = ("install an enzyme-function predictor and pass it in; "
                         "no built-in guess is substituted, because an "
                         "annotation-transfer heuristic would produce "
                         "plausible reactions with no evidence behind them")

    def is_available(self) -> bool:
        return False

    def predict(self, job: DiscoveryJob) -> Sequence[FunctionHypothesis]:
        raise ToolUnavailableError(self.name, self.install_hint)


@dataclass(frozen=True)
class UnavailableDeNovoGenerator:
    """Absent, and loudly so. Nothing here invents a sequence."""

    name: str = "de_novo_generator"
    version: str = "absent"
    registry_keys: tuple[str, ...] = ()
    uses_model_weights: bool = True
    runs_remotely: bool = False
    install_hint: str = ("install a de novo enzyme design tool and register "
                         "its code, weights, inputs and outputs separately; "
                         "a random or mutated sequence is never substituted")

    def is_available(self) -> bool:
        return False

    def generate(self, job: DesignJob) -> Sequence[RawDesign]:
        raise ToolUnavailableError(self.name, self.install_hint)


# ==========================================================================
# labelling what comes back
# ==========================================================================

@dataclass(frozen=True)
class BranchProposal:
    """One generator output, with the ceiling its branch fixes for it."""

    proposal_id: str
    kind: BranchKind
    generator: str
    generator_version: str
    sequence: str
    evidence_ceiling: EvidenceStrength
    generator_confidence: Mapping[str, float]
    forbidden_claims: tuple[str, ...]
    notes: str = ""


def proposals_from_designs(designs: Iterable[RawDesign], generator: DeNovoGenerator
                           ) -> list[BranchProposal]:
    """Label raw designs. The generator's confidence is kept, never promoted."""
    seen: set[str] = set()
    out: list[BranchProposal] = []
    for d in designs:
        if d.design_id in seen:
            raise ValueError(f"design id {d.design_id!r} appears twice")
        seen.add(d.design_id)
        if not d.sequence.strip():
            raise ValueError(f"design {d.design_id!r} has no sequence")
        out.append(BranchProposal(
            proposal_id=d.design_id, kind=BranchKind.DE_NOVO_DESIGN,
            generator=generator.name, generator_version=generator.version,
            sequence=d.sequence.strip().upper(),
            evidence_ceiling=BRANCH_CEILING[BranchKind.DE_NOVO_DESIGN],
            generator_confidence=dict(d.generator_confidence),
            forbidden_claims=BRANCH_FORBIDDEN_CLAIMS[BranchKind.DE_NOVO_DESIGN],
            notes=d.notes))
    return out


DE_NOVO_SOURCE_PREFIX = "de_novo:"
DE_NOVO_SEARCH_METHOD = "de_novo_design"


def design_candidate(proposal: BranchProposal, family_name: str | None = None
                     ) -> Candidate:
    """A candidate whose provenance says it was designed, not mined.

    No accession and no seed: it has neither. ``annotation_confidence`` is the
    branch's ceiling, so a reader of the record cannot mistake it for a mined
    sequence with a database entry.
    """
    if proposal.kind is not BranchKind.DE_NOVO_DESIGN:
        raise ValueError("only a de novo proposal becomes a designed candidate")
    return Candidate(
        candidate_id=proposal.proposal_id,
        sequence_record=SequenceRecord(
            candidate_id=proposal.proposal_id, sequence=proposal.sequence,
            source_database=f"{DE_NOVO_SOURCE_PREFIX}{proposal.generator}",
            database_version=proposal.generator_version,
            search_method=DE_NOVO_SEARCH_METHOD, is_fragment=False,
            description=("designed sequence; computational construct with no "
                         "evidence of expression or activity"),
            annotation_confidence=proposal.evidence_ceiling),
        family=FamilyAnnotation(family_name=family_name))


def is_high_evidence_eligible(candidate: Candidate) -> bool:
    """False for a designed sequence: it is a hypothesis, not evidence.

    For :func:`eagent.science.diversity.compose_batch`'s
    ``high_evidence_eligible``. Reads the provenance the candidate carries
    rather than a flag someone could forget to set.
    """
    record = candidate.sequence_record
    designed = ((record.source_database or "").startswith(DE_NOVO_SOURCE_PREFIX)
                or record.search_method == DE_NOVO_SEARCH_METHOD)
    return not designed
