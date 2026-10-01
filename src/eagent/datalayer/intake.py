"""Staged promotion of evidence: four tiers that never merge on ingestion.

Why this module exists
----------------------
The single most destructive step in building an enzyme activity dataset is the
one nobody writes down: an ingest script reads a curated database row, a
machine-extracted sentence and a model prediction, writes all three into one
table with a ``label`` column, and every downstream number is then a weighted
average of an observation and a guess. Nothing in the resulting file records
which was which, so the mistake is unrecoverable.

This module keeps the four kinds apart *in storage* and makes the only path
upward an explicit human act:

* :class:`EvidenceTier` -- expert-verified primary, curated database,
  machine-extracted pending review, model-inferred. Partitioned in
  :class:`IntakeStore`; pooling requires naming the tiers you are pooling.
* :func:`promote` -- the only upward path. It demands a named human reviewer, a
  justification, and the registry, and it writes an :class:`AuditEntry`. There
  is deliberately no ``auto_promote``, no confidence threshold that promotes,
  and no promotion out of :data:`EvidenceTier.MODEL_INFERRED` at all: reviewing
  an inference does not turn it into an observation.
* the registry ceiling -- :attr:`DataSource.evidence_strength_ceiling` is
  enforced on both ingest and promotion, so an EC-number-plus-species mapping
  from BRENDA cannot acquire a sequence-level experimental label by passing
  through a pipeline. Exceeding the ceiling requires a human *and* the primary
  :class:`EvidenceRef` that actually supports the stronger claim.
* :func:`normalise_outcome` -- messy source statements onto the six outcome
  classes, refusing to guess. ``n.d.`` means "not detected" in one paper and
  "not determined" in the next, so it resolves to ``NOT_TESTED`` with a
  recorded uncertainty, never to a negative. Negation is read *structurally*,
  from the position of a negator relative to the phrase it governs, because a
  matcher that consumes phrases out of the text loses the negator and reads
  "0% conversion observed" as a confirmed product.
* :func:`direction_check` -- an alcohol oxidation measurement is not evidence
  for the ketone reduction, however good the enzyme looks.
* :func:`coverage` -- how many records reach each tier and how many are fully
  specified, so a campaign can tell evidence from rows.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import EAgentError
from ..provenance import utc_now
from ..schemas.record import (
    Detection,
    EvidenceRef,
    EvidenceStrength,
    ExperimentRecord,
    OutcomeClass,
    ReactionDirection,
)
from ..schemas.reaction import ReactionClass
from .registry import SourceRegistry, UnknownSourceError

__all__ = [
    "AuditEntry",
    "CoverageRequirement",
    "DIRECTION_AXIS_UNASSESSED",
    "DirectionVerdict",
    "EvidenceTier",
    "ExtractionMethod",
    "FULL_SPECIFICATION_REQUIREMENTS",
    "IntakeCoverage",
    "IntakeError",
    "IntakeRecord",
    "IntakeStore",
    "MODEL_INFERENCE_IS_NOT_EVIDENCE",
    "ModelInferenceNotEvidenceError",
    "NO_AUTOMATIC_PROMOTION",
    "OutcomeNormalisation",
    "REVERSIBLE_CLASS_PAIRS",
    "ReviewState",
    "ReviewerRequiredError",
    "SourceCeilingError",
    "TIER_ORDER",
    "TierMergeRefusedError",
    "TierPromotionError",
    "coverage",
    "direction_check",
    "ingest",
    "normalise_outcome",
    "promote",
    "requirements_for",
    "reverse_class_of",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class IntakeError(EAgentError):
    """Base class for every refusal raised while admitting evidence.

    Exists so a caller can distinguish "this record was refused on provenance
    grounds" from a validation error in the record's own fields; the two call
    for different fixes, and collapsing them produces an ingest script that
    retries the wrong thing.
    """


class TierPromotionError(IntakeError):
    """A promotion was requested that the tier ladder does not permit.

    Prevents the quiet downgrade-then-upgrade trick and the no-op promotion
    that writes an audit entry without changing anything, both of which make
    the audit log describe a review that never decided anything.
    """


class ReviewerRequiredError(TierPromotionError):
    """Promotion was attempted without a named human reviewer.

    This is the rule that stops an ingest pipeline promoting its own output.
    A reviewer string that names an automated actor is refused for the same
    reason an empty one is: the audit entry has to point at a person who can be
    asked why.
    """


class SourceCeilingError(IntakeError):
    """A claim exceeds what the registered source can support.

    Prevents the EC-number-plus-species to sequence-level-experimental jump,
    which is how a family annotation becomes a measurement with nobody having
    decided that it should.
    """


class ModelInferenceNotEvidenceError(TierPromotionError):
    """Something tried to promote a model inference into evidence about the world.

    Refused unconditionally rather than gated behind a reviewer: a human reading
    a prediction produces a reviewed prediction, not an observation. The correct
    action is to run the experiment and ingest the result as a new record.
    """


class TierMergeRefusedError(IntakeError):
    """Records from different tiers were pooled without the caller saying so.

    Prevents the implicit union that turns a store with four honest partitions
    into one table of rows of unknown provenance.
    """


#: Quoted in refusals and in the audit log, so the reason survives into files.
NO_AUTOMATIC_PROMOTION = (
    "evidence is never promoted automatically: no confidence score, no source "
    "agreement count and no re-curation raises a record's tier. The only path "
    "upward is promote(), which records a named human reviewer and a "
    "justification."
)

MODEL_INFERENCE_IS_NOT_EVIDENCE = (
    "a model inference is a hypothesis about the world, not a measurement of "
    "it. Reviewing it confirms that the model said so, which is not the same "
    "claim, so model_inferred records are never promoted; the experiment is "
    "ingested as a new record instead."
)


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------

class EvidenceTier(str, enum.Enum):
    """How a record came to exist, kept apart from how strong its claim is.

    Tier answers "who produced this row and did a person check it"; the
    :class:`~eagent.schemas.record.EvidenceStrength` ladder answers "how tightly
    does the claim bind to this exact sequence". They are different axes: a
    machine extraction can concern a specific sequence, and an expert can verify
    an EC-level mapping that is still only EC-level. Collapsing them is what
    produces a dataset where "verified" means nothing in particular.
    """

    EXPERT_VERIFIED_PRIMARY = "expert_verified_primary"
    CURATED_DATABASE = "curated_database"
    MACHINE_EXTRACTED_PENDING = "machine_extracted_pending"
    MODEL_INFERRED = "model_inferred"

    @property
    def rank(self) -> int:
        """Ordinal position. Used only for direction of travel, never averaged."""
        return _TIER_RANK[self]

    @property
    def is_evidence_about_the_world(self) -> bool:
        """False for model inference, which is a statement about a model."""
        return self is not EvidenceTier.MODEL_INFERRED

    @property
    def is_human_checked(self) -> bool:
        """Whether a person signed off on this specific record."""
        return self is EvidenceTier.EXPERT_VERIFIED_PRIMARY

    @property
    def storage_partition(self) -> str:
        """Partition name. Tiers are stored apart so a join cannot merge them."""
        return f"tier/{self.value}"

    @property
    def max_strength_on_ingest(self) -> EvidenceStrength:
        """Strongest claim an *automated* ingest may stamp at this tier.

        Machine extraction is capped below sequence-level experimental because
        :class:`~eagent.schemas.record.EvidenceRef` already refuses that
        combination without a verifier; capping here means the refusal happens
        before a half-built record exists.
        """
        return _TIER_INGEST_CEILING[self]

    def describe(self) -> str:
        return f"{self.value}: {_TIER_DOC[self]}"


_TIER_RANK: dict[EvidenceTier, int] = {
    EvidenceTier.EXPERT_VERIFIED_PRIMARY: 3,
    EvidenceTier.CURATED_DATABASE: 2,
    EvidenceTier.MACHINE_EXTRACTED_PENDING: 1,
    EvidenceTier.MODEL_INFERRED: 0,
}

_TIER_DOC: dict[EvidenceTier, str] = {
    EvidenceTier.EXPERT_VERIFIED_PRIMARY:
        "an original experimental record a named person read and checked",
    EvidenceTier.CURATED_DATABASE:
        "a record curated by a database's own staff; checked, but not by us, "
        "and usually not against the construct we care about",
    EvidenceTier.MACHINE_EXTRACTED_PENDING:
        "a relation pulled out of literature by software, awaiting review; "
        "usable for triage, never as a label",
    EvidenceTier.MODEL_INFERRED:
        "a model's own output; not evidence about the world at any confidence",
}

_TIER_INGEST_CEILING: dict[EvidenceTier, EvidenceStrength] = {
    EvidenceTier.EXPERT_VERIFIED_PRIMARY: EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL,
    EvidenceTier.CURATED_DATABASE: EvidenceStrength.HOMOLOG_EXPERIMENTAL,
    EvidenceTier.MACHINE_EXTRACTED_PENDING: EvidenceStrength.EC_SPECIES_MAPPED,
    EvidenceTier.MODEL_INFERRED: EvidenceStrength.COMPUTATIONAL_CONSTRUCT,
}

#: Display and storage order, strongest first. Reports list all four.
TIER_ORDER: tuple[EvidenceTier, ...] = (
    EvidenceTier.EXPERT_VERIFIED_PRIMARY,
    EvidenceTier.CURATED_DATABASE,
    EvidenceTier.MACHINE_EXTRACTED_PENDING,
    EvidenceTier.MODEL_INFERRED,
)


class ExtractionMethod(str, enum.Enum):
    """How the row was produced, recorded separately from the tier.

    Kept after promotion: a record promoted to expert-verified is still a record
    that began as a machine extraction, and knowing that is what lets a later
    audit re-check the extractions rather than the whole corpus.
    """

    HUMAN_READING_PRIMARY = "human_reading_primary"
    DATABASE_EXPORT = "database_export"
    MACHINE_EXTRACTION_LLM = "machine_extraction_llm"
    MACHINE_EXTRACTION_RULE = "machine_extraction_rule"
    MODEL_PREDICTION = "model_prediction"
    UNKNOWN = "unknown"

    @property
    def is_machine_extraction(self) -> bool:
        return self in (ExtractionMethod.MACHINE_EXTRACTION_LLM,
                        ExtractionMethod.MACHINE_EXTRACTION_RULE)

    @property
    def is_model_output(self) -> bool:
        return self is ExtractionMethod.MODEL_PREDICTION


class ReviewState(str, enum.Enum):
    """Where this record sits in human review.

    ``NOT_REVIEWED`` is the default and is never inferred away: a record that
    has been read by nobody looks identical to one that passed review unless
    the distinction is stored.
    """

    NOT_REVIEWED = "not_reviewed"
    UNDER_REVIEW = "under_review"
    REVIEWED_ACCEPTED = "reviewed_accepted"
    REVIEWED_REJECTED = "reviewed_rejected"

    @property
    def is_settled(self) -> bool:
        return self in (ReviewState.REVIEWED_ACCEPTED, ReviewState.REVIEWED_REJECTED)


#: Reviewer strings that name software rather than a person. A promotion signed
#: by one of these is the automatic promotion this module exists to prevent.
_AUTOMATED_REVIEWER_PATTERNS: tuple[str, ...] = (
    "model", "llm", "gpt", "claude", "agent", "auto", "automatic", "automated",
    "bot", "script", "pipeline", "system", "none", "n/a", "na", "unknown",
    "tbd", "self", "ingest", "default", "ai",
)


def _is_automated_reviewer(name: str) -> bool:
    tokens = [t for t in re.split(r"[^a-z0-9]+", name.strip().lower()) if t]
    if not tokens:
        return True
    return any(t in _AUTOMATED_REVIEWER_PATTERNS for t in tokens)


def _require_human_reviewer(reviewer: str | None) -> str:
    """Return a usable reviewer identity or refuse.

    Prevents the audit log from recording a promotion that no person can be
    asked about, which is indistinguishable from no review at all.
    """
    name = (reviewer or "").strip()
    if not name:
        raise ReviewerRequiredError(
            f"promotion requires a named human reviewer. {NO_AUTOMATIC_PROMOTION}")
    if len(name) < 3:
        raise ReviewerRequiredError(
            f"reviewer '{name}' is too short to identify a person; record the "
            f"name or ORCID of whoever did the review")
    if _is_automated_reviewer(name):
        raise ReviewerRequiredError(
            f"reviewer '{name}' names an automated actor, not a person. "
            f"{NO_AUTOMATIC_PROMOTION}")
    return name


#: Shortest justification that can plausibly say what was checked.
MIN_JUSTIFICATION_CHARS = 12


def _require_justification(justification: str | None) -> str:
    text = " ".join((justification or "").split())
    if len(text) < MIN_JUSTIFICATION_CHARS:
        raise TierPromotionError(
            f"a promotion justification must say what the reviewer checked "
            f"(at least {MIN_JUSTIFICATION_CHARS} characters); got {text!r}")
    return text


def _strength_rank(s: EvidenceStrength) -> int:
    rank = getattr(s, "rank", None)
    if isinstance(rank, int):
        return rank
    return EvidenceStrength(s).rank


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

class AuditEntry(BaseModel):
    """One recorded act on an intake record: ingest, promote, review, refuse.

    Frozen and append-only. Exists so that "this row is sequence-level
    experimental" can always be traced to the person who decided it was, which
    is the only thing that makes the claim checkable later.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: str
    at: str
    reviewer: str | None = None
    justification: str = ""
    from_tier: EvidenceTier | None = None
    to_tier: EvidenceTier | None = None
    from_strength: EvidenceStrength | None = None
    to_strength: EvidenceStrength | None = None
    source_id: str | None = None
    source_ceiling: EvidenceStrength | None = None
    ceiling_exceeded: bool = Field(
        False,
        description="True when the reviewer raised the claim above what the "
                    "registered source can support, using separately attached "
                    "primary evidence. Always visible in the log.")
    supporting_evidence: str | None = Field(
        None, description="Identifier of the primary EvidenceRef that justified "
                          "exceeding the source ceiling.")
    note: str = ""

    def render(self) -> str:
        bits = [f"{self.at} {self.action}"]
        if self.from_tier or self.to_tier:
            bits.append(f"tier {_v(self.from_tier)} -> {_v(self.to_tier)}")
        if self.from_strength or self.to_strength:
            bits.append(f"strength {_v(self.from_strength)} -> {_v(self.to_strength)}")
        if self.reviewer:
            bits.append(f"by {self.reviewer}")
        if self.ceiling_exceeded:
            bits.append(f"CEILING EXCEEDED (source ceiling "
                        f"{_v(self.source_ceiling)}, supported by "
                        f"{self.supporting_evidence})")
        if self.justification:
            bits.append(f'"{self.justification}"')
        if self.note:
            bits.append(f"[{self.note}]")
        return " | ".join(bits)


def _v(x: Any) -> str:
    return "-" if x is None else getattr(x, "value", str(x))


# ---------------------------------------------------------------------------
# IntakeRecord
# ---------------------------------------------------------------------------

class IntakeRecord(BaseModel):
    """One :class:`ExperimentRecord` wrapped in everything needed to trust it.

    The wrapper exists because the record itself cannot answer "where did this
    come from and did anybody check it". Without the wrapper those facts live in
    the ingest script, which is thrown away, and the record then looks the same
    whether it was read out of a paper or produced by a model.
    """

    model_config = ConfigDict(extra="forbid")

    intake_id: str
    record: ExperimentRecord
    tier: EvidenceTier
    source_id: str = Field(
        ..., min_length=2,
        description="Registry id of the resource this came from, so the "
                    "evidence_strength_ceiling can be enforced against it.")
    extraction_method: ExtractionMethod = ExtractionMethod.UNKNOWN
    claimed_strength: EvidenceStrength = EvidenceStrength.ANNOTATION_ONLY
    reviewer: str | None = None
    review_state: ReviewState = ReviewState.NOT_REVIEWED
    uncertainties: list[str] = Field(
        default_factory=list,
        description="What is unresolved about this row. Carried rather than "
                    "resolved, because resolving it by guessing is the failure.")
    needs_curation: bool = False
    audit: list[AuditEntry] = Field(default_factory=list)
    ingested_at: str | None = None

    @model_validator(mode="after")
    def _tier_invariants(self) -> "IntakeRecord":
        if self.tier is EvidenceTier.EXPERT_VERIFIED_PRIMARY:
            if not (self.reviewer or "").strip():
                raise IntakeError(
                    f"{self.intake_id}: expert_verified_primary without a named "
                    f"reviewer is the same row as an unreviewed one with a nicer "
                    f"label")
            if self.review_state is not ReviewState.REVIEWED_ACCEPTED:
                raise IntakeError(
                    f"{self.intake_id}: expert_verified_primary requires "
                    f"review_state=reviewed_accepted, got "
                    f"{self.review_state.value}")
        if self.tier is EvidenceTier.MODEL_INFERRED:
            if self.claimed_strength is not EvidenceStrength.COMPUTATIONAL_CONSTRUCT:
                raise IntakeError(
                    f"{self.intake_id}: a model_inferred row may only claim "
                    f"computational_construct strength; {MODEL_INFERENCE_IS_NOT_EVIDENCE}")
            if self.record.outcome.is_experimental:
                raise IntakeError(
                    f"{self.intake_id}: a model_inferred row carries the "
                    f"experimental outcome '{self.record.outcome.value}'. "
                    f"{MODEL_INFERENCE_IS_NOT_EVIDENCE}")
        if self.claimed_strength.is_sequence_level and not (self.reviewer or "").strip():
            raise IntakeError(
                f"{self.intake_id}: a sequence-level experimental claim needs a "
                f"recorded reviewer. {NO_AUTOMATIC_PROMOTION}")
        if self.extraction_method.is_model_output \
                and self.tier is not EvidenceTier.MODEL_INFERRED:
            raise IntakeError(
                f"{self.intake_id}: extraction_method=model_prediction cannot sit "
                f"at tier {self.tier.value}. {MODEL_INFERENCE_IS_NOT_EVIDENCE}")
        return self

    # -- queries -----------------------------------------------------------
    @property
    def partition(self) -> str:
        """Storage partition. Two tiers never share one."""
        return self.tier.storage_partition

    @property
    def is_usable_as_label(self) -> bool:
        """Whether this row may train or evaluate anything as a ground truth.

        Only a human-checked experimental outcome qualifies. A curated database
        row is good triage and a poor label, because the curator checked the
        paper, not our construct and not our substrate.
        """
        return (self.tier.is_human_checked
                and self.record.outcome.informs_catalytic_ability
                and self.review_state is ReviewState.REVIEWED_ACCEPTED)

    def with_audit(self, entry: AuditEntry, **updates: Any) -> "IntakeRecord":
        """Return a copy with ``entry`` appended and ``updates`` applied.

        Returns a new object rather than mutating, so a refusal part-way through
        a promotion cannot leave a half-promoted record behind.
        """
        data = self.model_dump()
        data.update(updates)
        data["audit"] = list(self.audit) + [entry]
        return IntakeRecord.model_validate(data)

    def history(self) -> list[str]:
        return [e.render() for e in self.audit]


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest(
    record: ExperimentRecord,
    *,
    tier: EvidenceTier,
    source_id: str,
    registry: SourceRegistry,
    extraction_method: ExtractionMethod = ExtractionMethod.UNKNOWN,
    claimed_strength: EvidenceStrength | None = None,
    reviewer: str | None = None,
    uncertainties: Sequence[str] = (),
    intake_id: str | None = None,
    at: str | None = None,
) -> IntakeRecord:
    """Admit a record at a tier, refusing any claim the source cannot support.

    This is the *only* constructor used by connectors. It refuses rather than
    clamps: silently lowering a caller's ``claimed_strength`` to the source
    ceiling would hide a connector that believes it is reading sequence-level
    data out of an EC-level resource, and that belief is the bug.

    ``tier`` is never inferred from ``extraction_method``, and
    ``extraction_method`` is never inferred from ``tier``; a caller that cannot
    say which it has passes ``UNKNOWN`` and gets a curation flag.

    A caller that passes no ``claimed_strength`` gets the *floor* --
    ``annotation_only``, or ``computational_construct`` at the model-inferred
    tier -- with an uncertainty recording that the connector said nothing. The
    source and tier ceilings remain what they are, a cap on what may be
    claimed; defaulting to a cap would stamp the strongest permitted label on
    the rows that assert the least, which is a guess upward and the exact move
    the rest of this module refuses.
    """
    source = registry.get(source_id)           # raises UnknownSourceError
    ceiling: EvidenceStrength = source.evidence_strength_ceiling
    tier_cap = tier.max_strength_on_ingest

    assumed_strength_note: str | None = None
    if claimed_strength is None:
        # Default to the FLOOR, not to either cap. The caps bound what this
        # source and tier are allowed to support; a row that states nothing
        # supports nothing, and stamping the maximum it would be permitted is
        # the upward guess this module exists to prevent. The ceiling stays the
        # ceiling: it is checked below, exactly as for a stated claim.
        claimed_strength = (EvidenceStrength.COMPUTATIONAL_CONSTRUCT
                            if tier is EvidenceTier.MODEL_INFERRED
                            else EvidenceStrength.ANNOTATION_ONLY)
        assumed_strength_note = (
            f"the connector stated no evidence strength for this row, so the "
            f"weakest ({claimed_strength.value}) was assumed; the source "
            f"ceiling ({ceiling.value}) and the tier ceiling "
            f"({tier_cap.value}) are the most it could be raised to, and "
            f"raising it requires promote() with a named human reviewer")

    if _strength_rank(claimed_strength) > _strength_rank(ceiling):
        raise SourceCeilingError(
            f"{source_id} is registered with evidence_strength_ceiling="
            f"{ceiling.value}; an automated ingest may not stamp "
            f"{claimed_strength.value} on its records. Raising it requires "
            f"promote() with a human reviewer and the primary evidence that "
            f"supports the stronger claim.")
    if _strength_rank(claimed_strength) > _strength_rank(tier_cap):
        raise SourceCeilingError(
            f"tier {tier.value} may not be ingested at {claimed_strength.value}; "
            f"its ingest ceiling is {tier_cap.value}. {NO_AUTOMATIC_PROMOTION}")

    if tier is EvidenceTier.EXPERT_VERIFIED_PRIMARY:
        reviewer = _require_human_reviewer(reviewer)
        review_state = ReviewState.REVIEWED_ACCEPTED
    else:
        review_state = ReviewState.NOT_REVIEWED

    if extraction_method.is_machine_extraction \
            and tier.rank > EvidenceTier.MACHINE_EXTRACTED_PENDING.rank:
        raise TierPromotionError(
            f"a machine extraction cannot be ingested at tier {tier.value}; "
            f"ingest it as machine_extracted_pending and promote it. "
            f"{NO_AUTOMATIC_PROMOTION}")

    notes = list(uncertainties)
    needs_curation = bool(source.needs_curation)
    if assumed_strength_note is not None:
        notes.append(assumed_strength_note)
    if extraction_method is ExtractionMethod.UNKNOWN:
        notes.append("extraction method not recorded by the connector; a curator "
                     "must determine whether this row was read, exported or "
                     "extracted")
        needs_curation = True
    if source.needs_curation:
        notes.append(f"source '{source_id}' is flagged needs_curation in the "
                     f"registry: {'; '.join(source.curation_notes) or 'see registry'}")

    ts = at or utc_now()
    entry = AuditEntry(
        action="ingest",
        at=ts,
        to_tier=tier,
        to_strength=claimed_strength,
        source_id=source_id,
        source_ceiling=ceiling,
        reviewer=reviewer,
        note=f"extraction={extraction_method.value}",
    )
    return IntakeRecord(
        intake_id=intake_id or f"intake:{source_id}:{record.record_id}",
        record=record,
        tier=tier,
        source_id=source_id,
        extraction_method=extraction_method,
        claimed_strength=claimed_strength,
        reviewer=reviewer,
        review_state=review_state,
        uncertainties=notes,
        needs_curation=needs_curation,
        audit=[entry],
        ingested_at=ts,
    )


# ---------------------------------------------------------------------------
# Promotion: the only path upward
# ---------------------------------------------------------------------------

def promote(
    record: IntakeRecord,
    reviewer: str,
    justification: str,
    *,
    registry: SourceRegistry,
    to_tier: EvidenceTier | None = None,
    to_strength: EvidenceStrength | None = None,
    supporting_evidence: EvidenceRef | None = None,
    at: str | None = None,
) -> IntakeRecord:
    """Raise a record's tier and/or evidence strength, with a human on record.

    The only upward path in this module. It exists to make the cost of the claim
    visible: somebody has to type their name and say what they checked, and the
    result is written into an append-only audit log that travels with the row.

    Refusals, each preventing a specific silent corruption:

    * no reviewer, or a reviewer naming software -- prevents a pipeline
      promoting its own output (:class:`ReviewerRequiredError`);
    * a model-inferred record -- prevents a prediction becoming an observation
      (:class:`ModelInferenceNotEvidenceError`);
    * a target at or below the current level -- prevents an audit entry that
      records a decision nobody made (:class:`TierPromotionError`);
    * a strength above the registered source's
      ``evidence_strength_ceiling`` without a primary ``EvidenceRef`` --
      prevents an EC-number-plus-species mapping acquiring a sequence-level
      experimental label (:class:`SourceCeilingError`).

    To exceed the source ceiling the reviewer must attach ``supporting_evidence``:
    the record they actually read. BRENDA cannot be used to justify a claim
    stronger than BRENDA supports; the paper behind it can.
    """
    name = _require_human_reviewer(reviewer)
    reason = _require_justification(justification)

    if to_tier is None and to_strength is None:
        raise TierPromotionError(
            "promote() needs a target: pass to_tier, to_strength, or both. "
            "A promotion with no target would write an audit entry for a "
            "decision that was never made.")

    if record.tier is EvidenceTier.MODEL_INFERRED:
        raise ModelInferenceNotEvidenceError(
            f"{record.intake_id} sits at tier model_inferred. "
            f"{MODEL_INFERENCE_IS_NOT_EVIDENCE}")
    if to_tier is EvidenceTier.MODEL_INFERRED:
        raise TierPromotionError(
            f"model_inferred is the bottom of the ladder; promote() only moves "
            f"records upward. {NO_AUTOMATIC_PROMOTION}")

    new_tier = to_tier or record.tier
    if to_tier is not None and to_tier.rank <= record.tier.rank:
        raise TierPromotionError(
            f"{record.intake_id}: promote() moves records upward only; "
            f"{record.tier.value} -> {to_tier.value} is not a promotion. "
            f"A record that should be weakened is re-ingested at the correct "
            f"tier so the downgrade is visible as such.")

    new_strength = to_strength or record.claimed_strength
    if to_strength is not None \
            and _strength_rank(to_strength) <= _strength_rank(record.claimed_strength):
        raise TierPromotionError(
            f"{record.intake_id}: {record.claimed_strength.value} -> "
            f"{to_strength.value} is not an increase in evidence strength.")

    try:
        source = registry.get(record.source_id)
    except UnknownSourceError as exc:
        raise SourceCeilingError(
            f"{record.intake_id}: source '{record.source_id}' is not registered, "
            f"so its evidence_strength_ceiling cannot be checked. Register the "
            f"source before promoting records from it."
        ) from exc
    ceiling: EvidenceStrength = source.evidence_strength_ceiling

    ceiling_exceeded = _strength_rank(new_strength) > _strength_rank(ceiling)
    if ceiling_exceeded:
        if supporting_evidence is None:
            raise SourceCeilingError(
                f"{record.intake_id}: '{record.source_id}' is registered with "
                f"evidence_strength_ceiling={ceiling.value}, which cannot "
                f"support a {new_strength.value} claim. To go above it, attach "
                f"supporting_evidence: the primary record the reviewer read. "
                f"A re-reading of the same database row is not new evidence.")
        if _strength_rank(supporting_evidence.strength) < _strength_rank(new_strength):
            raise SourceCeilingError(
                f"{record.intake_id}: supporting evidence "
                f"'{supporting_evidence.identifier}' is itself only "
                f"{supporting_evidence.strength.value}; it cannot justify a "
                f"{new_strength.value} claim.")
        if not (supporting_evidence.verified_by or "").strip():
            raise SourceCeilingError(
                f"{record.intake_id}: supporting evidence "
                f"'{supporting_evidence.identifier}' names no verifier; the "
                f"reviewer who read it must be recorded on the evidence too.")

    ts = at or utc_now()
    entry = AuditEntry(
        action="promote",
        at=ts,
        reviewer=name,
        justification=reason,
        from_tier=record.tier,
        to_tier=new_tier,
        from_strength=record.claimed_strength,
        to_strength=new_strength,
        source_id=record.source_id,
        source_ceiling=ceiling,
        ceiling_exceeded=ceiling_exceeded,
        supporting_evidence=(supporting_evidence.identifier
                             if supporting_evidence is not None else None),
    )

    updates: dict[str, Any] = {
        "tier": new_tier,
        "claimed_strength": new_strength,
        "reviewer": name,
    }
    if new_tier is EvidenceTier.EXPERT_VERIFIED_PRIMARY:
        updates["review_state"] = ReviewState.REVIEWED_ACCEPTED
    elif record.review_state is ReviewState.NOT_REVIEWED:
        updates["review_state"] = ReviewState.UNDER_REVIEW

    if supporting_evidence is not None:
        inner = record.record.model_copy(
            update={"evidence": list(record.record.evidence) + [supporting_evidence]})
        updates["record"] = inner
    if ceiling_exceeded:
        updates["uncertainties"] = list(record.uncertainties) + [
            f"claim raised above the registered ceiling of "
            f"'{record.source_id}' ({ceiling.value}) by {name}, on the strength "
            f"of {supporting_evidence.identifier if supporting_evidence else '?'}"
        ]
    return record.with_audit(entry, **updates)


# ---------------------------------------------------------------------------
# Outcome normalisation
# ---------------------------------------------------------------------------

#: ``(rule name, outcome or None for "known-ambiguous", phrases)``.
#: ``None`` marks a phrase whose meaning genuinely varies between sources; it
#: resolves to NOT_TESTED with an uncertainty rather than being guessed at.
_OUTCOME_RULES: tuple[tuple[str, OutcomeClass | None, tuple[str, ...]], ...] = (
    ("explicit_not_tested", OutcomeClass.NOT_TESTED, (
        "not tested", "untested", "no assay performed", "not assayed",
        "not screened", "not measured", "not attempted",
    )),
    ("expression_failure", OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE, (
        "no soluble protein", "insoluble", "inclusion bodies",
        "did not express", "no expression", "expression failed",
        "failed to express", "not soluble", "no purified protein",
        "protein could not be purified",
    )),
    ("computational_failure", OutcomeClass.COMPUTATIONAL_FAILURE, (
        "docking failed", "no pose generated", "no poses generated",
        "modelling failed", "modeling failed", "prediction failed",
        "simulation failed", "did not converge", "job timed out", "timed out",
        "structure prediction failed", "no model produced", "tool error",
    )),
    ("computational_negative", OutcomeClass.COMPUTATIONAL_NEGATIVE, (
        "predicted inactive", "predicted no activity", "model predicts inactive",
        "in silico negative", "predicted non substrate", "predicted negative",
    )),
    ("wrong_product_or_configuration", OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION, (
        "wrong enantiomer", "opposite enantiomer", "opposite configuration",
        "other product", "different product", "side product only",
        "over reduction", "overreduction", "wrong regiochemistry",
        "incorrect stereochemistry", "wrong product",
    )),
    ("negative_detected", OutcomeClass.NO_TARGET_PRODUCT_DETECTED, (
        "no target product detected", "no product detected", "not detected",
        "no activity detected", "no detectable activity", "no conversion",
        "no turnover", "below detection limit", "below the limit of detection",
        "no activity", "inactive", "not active", "zero conversion",
    )),
    ("positive", OutcomeClass.CONFIRMED_TARGET_PRODUCT, (
        "target product confirmed", "product confirmed", "product formed",
        "product observed", "product detected", "product identified",
        "activity confirmed", "activity detected", "conversion observed",
        "active", "activity", "converted", "turnover observed", "positive",
    )),
    # ``no`` is deliberately absent here. It is a *negator*, not an outcome
    # token, and listing it as a phrase made its meaning depend on whether some
    # other phrase happened to consume the words around it. It is scanned for
    # position instead (:func:`_scan_negators`); a statement that is nothing but
    # a bare negator still resolves to NOT_TESTED, with the same reason.
    ("known_ambiguous", None, (
        "nd", "na", "trace", "racemic", "no data", "not reported",
        "not determined", "not available", "unclear", "ambiguous", "unknown",
        "n r", "variable", "low", "weak", "some activity", "slight activity",
        "poor", "marginal", "borderline", "yes", "negative",
    )),
)

#: Why each known-ambiguous phrase is refused, so the uncertainty is specific.
_AMBIGUITY_NOTES: Mapping[str, str] = {
    "nd": "'n.d.' means 'not detected' in some sources and 'not determined' in "
          "others; the two are a negative and an absence of a test",
    "na": "'n/a' can mean not applicable, not assayed or not available",
    "trace": "'trace' is a positive observation below a quantitation limit in "
             "some papers and a blank-level artefact in others",
    "racemic": "a racemic product is a positive for the transformation and a "
               "failure for the configuration; which one it is depends on the "
               "target spec, which this function is not given",
    "no data": "absence of a value is not a measured absence of activity",
    "not reported": "the assay may or may not have been run",
    "not determined": "the assay may or may not have been run",
    "not available": "the assay may or may not have been run",
    "unclear": "the source statement does not resolve to an outcome class",
    "ambiguous": "the source statement does not resolve to an outcome class",
    "unknown": "the source statement does not resolve to an outcome class",
    "n r": "'n.r.' means 'not reported' or 'no reaction' depending on the source",
    "variable": "a range of behaviours was reported without a single outcome",
    "low": "'low' is a magnitude, not an outcome class; it may be a weak "
           "positive or a value at the noise floor",
    "weak": "'weak' is a magnitude, not an outcome class",
    "some activity": "a magnitude without a detection method or limit",
    "slight activity": "a magnitude without a detection method or limit",
    "poor": "'poor' is a magnitude, not an outcome class",
    "marginal": "'marginal' is a magnitude, not an outcome class",
    "borderline": "'borderline' is a magnitude, not an outcome class",
    "yes": "'yes' does not say what was observed, or that the product was "
           "identified rather than a cofactor signal",
    "negative": "'negative' is used for a failed assay, a failed expression and "
                "a measured absence of product",
}


# -- tokenisation and negation scope ----------------------------------------
#
# Matching runs over a *token list with positions*, not over a string that each
# match deletes from. Negation is a question about position: "0% conversion
# observed" and "conversion observed" contain the same positive phrase and mean
# opposite things, and a matcher that consumes the longest phrase anywhere in
# the string and throws the offsets away cannot tell them apart. The negator is
# then left over as a stray token belonging to no rule, is discarded, and a row
# with no conversion at all is stored as a confirmed product.

#: Stands in for a clause boundary (punctuation or a contrastive conjunction).
#: Negation does not cross one, so "no standard available, product observed"
#: keeps its positive while "no product observed" does not.
_CLAUSE_BREAK = "|"

#: Words that begin a new clause. A negator before one of these governs the
#: clause it is in, not the claim after it.
_CLAUSE_CONJUNCTIONS: frozenset[str] = frozenset({
    "but", "however", "although", "though", "whereas", "while", "yet",
    "despite", "nevertheless", "nonetheless", "except",
})

#: Tokens that assert absence outright. A statement they govern is a negative,
#: not a positive.
_DEFINITE_NEGATORS: frozenset[str] = frozenset({
    "no", "not", "non", "without", "absent", "absence", "undetectable",
    "undetected", "nil", "none", "zero", "never", "neither", "nor", "lacking",
})

#: Tokens that put a statement at or near the noise floor. These resolve to
#: neither a positive nor a negative: "trace" and "<1%" are a real small
#: conversion in one paper and a blank-level artefact in the next, and choosing
#: between them is the guess this module exists to refuse.
_NEAR_ZERO_NEGATORS: frozenset[str] = frozenset({
    "negligible", "trace", "traces", "barely", "insignificant", "minimal",
    "nd", "borderline",
})

#: Comparison qualifiers. With a quantity or a limit noun after them they bound
#: a statement from above, which is the "<1%" case.
_QUANTITY_QUALIFIERS: frozenset[str] = frozenset({
    "lessthan", "below", "under", "beneath",
})

#: Nouns that make a bare qualifier into a bound ("under the detection limit").
_LIMIT_NOUNS: frozenset[str] = frozenset({
    "detection", "limit", "limits", "lod", "loq", "background", "noise",
    "baseline", "quantitation", "quantification",
})

#: Negators that may also govern a statement they *follow* ("activity nil").
#: Determiners and prepositions are excluded: in "product formed without
#: cofactor" the "without" takes the cofactor as its object, not the product.
_TRAILING_ABSENCE: frozenset[str] = frozenset({
    "nil", "none", "zero", "absent", "absence", "undetectable", "undetected",
    "nd",
})

#: Tokens that carry a measurement but make no claim of their own, so a clause
#: built only from these is a bare quantity ("conversion observed: 0%").
_MEASUREMENT_WORDS: frozenset[str] = frozenset({
    "percent", "pct", "ee", "conversion", "conv", "yield", "mm", "um", "nm",
    "mg", "ml", "mol", "mmol", "min", "h", "s", "u", "the", "a", "of", "and",
    "or", "was", "is", "were", "are", "at",
})

#: Prepositions that mark a trailing quantity as an experimental parameter
#: rather than the measured outcome: "product formed with 0 mM NADPH" is a
#: control, not a zero-conversion row.
_PARAMETER_PREPOSITIONS: frozenset[str] = frozenset({
    "with", "without", "in", "on", "using", "from", "per", "over", "after",
    "containing", "against", "versus", "vs", "by", "into",
})

#: How many tokens may sit between a negator and the phrase it governs before
#: the link stops being readable from the text alone.
_NEGATION_WINDOW = 2


#: Why a statement that is nothing but a negator or a quantity is refused.
#: "no" and "0%" in an outcome column are each used for all three of "not
#: detected", "not determined" and "not applicable", and nothing in the cell
#: says which.
_BARE_NEGATION_NOTE = (
    "a bare negator or quantity does not say what was measured, whether the "
    "assay ran, whether the protein expressed, or at what detection limit, so "
    "it is not an outcome"
)

#: Internal spellings put back into readable form for the recorded reasons.
_TOKEN_DISPLAY: Mapping[str, str] = {
    "lessthan": "<", "greaterthan": ">", "nd": "n.d.",
}


def _display(token: str) -> str:
    return _TOKEN_DISPLAY.get(token, token)


def _tokenise(raw: str) -> list[str]:
    """Normalise a source statement into tokens, keeping clause boundaries.

    Punctuation becomes :data:`_CLAUSE_BREAK` rather than whitespace, because
    the boundary is the thing that stops a negator in one clause reaching a
    claim in the next. ``<`` and ``%`` survive as words, and a decimal point is
    kept inside its number, so "<0.5%" is still readable as a bounded quantity
    after the alphanumeric squeeze.
    """
    s = str(raw).lower()
    for dash in ("‐", "‑", "‒", "–", "—", "−"):
        s = s.replace(dash, "-")
    s = re.sub(r"\bn\s*/\s*a\b", " na ", s)
    s = re.sub(r"\bn\s*\.\s*d\s*\.?", " nd ", s)
    s = re.sub(r"\bn\s*\.\s*r\s*\.?", " n r ", s)
    s = re.sub(r"\b(?:less|fewer)\s+than\b", " lessthan ", s)
    s = re.sub(r"(\d)\s*\.\s*(\d)", r"\1point\2", s)
    for op in ("≤", "<="):
        s = s.replace(op, " lessthan ")
    s = s.replace("<", " lessthan ")
    for op in ("≥", ">="):
        s = s.replace(op, " greaterthan ")
    s = s.replace(">", " greaterthan ")
    s = s.replace("%", " percent ")
    s = re.sub(r"[;:,.()\[\]{}/\\\n\r\t|]+", f" {_CLAUSE_BREAK} ", s)
    s = re.sub(r"[^a-z0-9|]+", " ", s)

    tokens: list[str] = []
    for tok in s.split():
        if tok in _CLAUSE_CONJUNCTIONS:
            tok = _CLAUSE_BREAK
        if tok == _CLAUSE_BREAK and (not tokens or tokens[-1] == _CLAUSE_BREAK):
            continue
        tokens.append(tok)
    while tokens and tokens[-1] == _CLAUSE_BREAK:
        tokens.pop()
    return tokens


def _clause_ids(tokens: Sequence[str]) -> list[int]:
    """Clause number of each token; a break token keeps the clause it closes."""
    out: list[int] = []
    clause = 0
    for tok in tokens:
        out.append(clause)
        if tok == _CLAUSE_BREAK:
            clause += 1
    return out


def _number_value(tok: str) -> float | None:
    """Numeric value of a token, or ``None``. ``0point5`` is ``0.5``."""
    if not tok or not tok[0].isdigit():
        return None
    try:
        return float(tok.replace("point", "."))
    except ValueError:
        return None


@dataclass(frozen=True)
class _PhraseMatch:
    """One rule phrase found at a known position in the token list."""

    rule: str
    outcome: OutcomeClass | None
    phrase: str
    start: int
    end: int


def _build_phrase_index() -> tuple[tuple[tuple[str, ...], str, OutcomeClass | None, str], ...]:
    index: list[tuple[tuple[str, ...], str, OutcomeClass | None, str]] = []
    for rule_name, outcome, phrases in _OUTCOME_RULES:
        for phrase in phrases:
            toks = tuple(t for t in _tokenise(phrase) if t != _CLAUSE_BREAK)
            if toks:
                index.append((toks, rule_name, outcome, phrase))
    index.sort(key=lambda e: (len(e[0]), len(e[3])), reverse=True)
    return tuple(index)


#: Rule phrases as token tuples, longest first.
_PHRASE_INDEX = _build_phrase_index()


def _match_phrases(tokens: Sequence[str]) -> list[_PhraseMatch]:
    """Leftmost-longest, non-overlapping phrase matches, with their positions.

    Longest-first at each position is what makes "not active" a negative rather
    than a negative *and* a positive; leftmost rather than globally-longest is
    what keeps the match positions meaningful, so a negator can be tested
    against the span it actually governs.
    """
    out: list[_PhraseMatch] = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i] == _CLAUSE_BREAK:
            i += 1
            continue
        hit: _PhraseMatch | None = None
        for toks, rule_name, outcome, phrase in _PHRASE_INDEX:
            k = len(toks)
            if k > n - i:
                continue
            if tuple(tokens[i:i + k]) == toks:
                hit = _PhraseMatch(rule_name, outcome, phrase, i, i + k)
                break
        if hit is None:
            i += 1
            continue
        out.append(hit)
        i = hit.end
    return out


@dataclass(frozen=True)
class _Negator:
    """A token that denies, or bounds to nothing, whatever it governs."""

    index: int
    token: str
    kind: str          # "definite" | "near_zero"
    may_trail: bool    # may it govern a statement it follows?
    note: str


def _scan_negators(tokens: Sequence[str]) -> list[_Negator]:
    """Every negator and none-meaning quantity in the text, with its position.

    Run before any phrase is consumed, so negation is structural rather than
    whatever token happens to be left over once the phrase matcher has eaten
    the rest of the sentence.
    """
    out: list[_Negator] = []
    for i, tok in enumerate(tokens):
        if tok == _CLAUSE_BREAK:
            continue
        value = _number_value(tok)
        if value is not None:
            if any(tokens[j] in _QUANTITY_QUALIFIERS
                   for j in range(max(0, i - _NEGATION_WINDOW), i)):
                continue                      # the qualifier is recorded instead
            if value == 0.0:
                out.append(_Negator(
                    i, tok, "definite", True,
                    f"the quantity '{_display(tok)}' is zero"))
            continue
        if tok in _QUANTITY_QUALIFIERS:
            after = tokens[i + 1:i + 1 + _NEGATION_WINDOW + 1]
            bounded = any(_number_value(t) is not None or t in _LIMIT_NOUNS
                          for t in after)
            out.append(_Negator(
                i, tok, "near_zero", bounded,
                f"'{_display(tok)}' bounds the statement from above rather "
                f"than measuring it"))
            continue
        if tok in _DEFINITE_NEGATORS:
            out.append(_Negator(
                i, tok, "definite", tok in _TRAILING_ABSENCE,
                f"'{_display(tok)}' denies what it governs"))
        elif tok in _NEAR_ZERO_NEGATORS:
            out.append(_Negator(
                i, tok, "near_zero", tok in _TRAILING_ABSENCE,
                f"'{_display(tok)}' places the statement at or near the noise "
                f"floor"))
    return out


def _bare_quantity_clauses(
    tokens: Sequence[str],
    clauses: Sequence[int],
    spans: Sequence[tuple[int, int]],
) -> set[int]:
    """Clauses that are only a quantity or a bare negator, e.g. ``": 0%"``.

    Such a clause makes no claim of its own, so it qualifies the neighbouring
    one: "conversion observed: 0%" is a zero-conversion row however the source
    punctuated it, and treating the punctuation as a wall is how the negator
    gets lost.
    """
    covered = {i for s, e in spans for i in range(s, e)}
    content: dict[int, list[int]] = {}
    for i, tok in enumerate(tokens):
        if tok == _CLAUSE_BREAK:
            continue
        content.setdefault(clauses[i], []).append(i)
    bare: set[int] = set()
    for cid, idxs in content.items():
        if any(i in covered for i in idxs):
            continue
        has_signal = False
        ok = True
        for i in idxs:
            tok = tokens[i]
            if _number_value(tok) is not None or tok in _DEFINITE_NEGATORS \
                    or tok in _NEAR_ZERO_NEGATORS or tok in _QUANTITY_QUALIFIERS:
                has_signal = True
                continue
            if tok in _MEASUREMENT_WORDS:
                continue
            ok = False
            break
        if ok and has_signal:
            bare.add(cid)
    return bare


def _governance(
    neg: _Negator,
    start: int,
    end: int,
    tokens: Sequence[str],
    clauses: Sequence[int],
    bare: set[int],
) -> str | None:
    """Whether ``neg`` governs the span ``[start, end)``, and how closely.

    ``"close"`` means the link is readable from the text: the negator sits
    beside the statement, leads it as a quantity, or is the whole of an
    adjoining clause. ``"far"`` means a negator is loose in the same clause but
    not attached to this statement, which is not a reading, it is a reason to
    stop and say so.
    """
    if start <= neg.index < end:
        return None
    if clauses[neg.index] == clauses[start]:
        if neg.index < start:
            gap = start - neg.index - 1
            return "close" if gap <= _NEGATION_WINDOW else "far"
        if not neg.may_trail:
            return None
        gap = neg.index - end
        if gap > _NEGATION_WINDOW:
            return None
        if any(t in _PARAMETER_PREPOSITIONS for t in tokens[end:neg.index]):
            return None                   # a reaction parameter, not the outcome
        return "close"
    if clauses[neg.index] in bare \
            and abs(clauses[neg.index] - clauses[start]) == 1:
        return "close"
    return None


@dataclass(frozen=True)
class OutcomeNormalisation:
    """The result of reading a source's outcome statement, with its caveats.

    Two outcome fields on purpose.

    ``outcome`` is the class it is *safe to store*: when the statement is
    ambiguous, or when the record cannot support the class the text implies, it
    degrades to ``NOT_TESTED``. A caller may always write this one.

    ``proposed_outcome`` is what the text appeared to say. It is kept so the
    degradation is visible and a curator can act on it, and so nobody later
    "recovers" the lost signal by re-parsing the string with worse rules.

    A blocked normalisation is not a failure of the source; it is a statement
    that this row is not yet interpretable.
    """

    raw: str | None
    outcome: OutcomeClass
    proposed_outcome: OutcomeClass | None
    confident: bool
    matched_rules: tuple[str, ...]
    uncertainties: tuple[str, ...]
    blocked_reason: str | None = None

    @property
    def is_storable(self) -> bool:
        """Whether ``proposed_outcome`` may be written onto a record as-is."""
        return self.blocked_reason is None

    @property
    def is_negative(self) -> bool:
        return self.outcome is OutcomeClass.NO_TARGET_PRODUCT_DETECTED

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "outcome": self.outcome.value,
            "proposed_outcome": (self.proposed_outcome.value
                                 if self.proposed_outcome else None),
            "confident": self.confident,
            "matched_rules": list(self.matched_rules),
            "uncertainties": list(self.uncertainties),
            "blocked_reason": self.blocked_reason,
            "storable": self.is_storable,
        }

    def describe(self) -> str:
        head = (f"{self.raw!r} -> {self.outcome.value}"
                f"{'' if self.confident else ' (not confident)'}")
        if self.proposed_outcome and self.proposed_outcome is not self.outcome:
            head += f" [text suggested {self.proposed_outcome.value}, blocked]"
        lines = [head]
        if self.blocked_reason:
            lines.append(f"  blocked: {self.blocked_reason}")
        lines.extend(f"  uncertainty: {u}" for u in self.uncertainties)
        return "\n".join(lines)


def normalise_outcome(
    raw: str | None,
    *,
    detection: Detection | None = None,
) -> OutcomeNormalisation:
    """Map a source's outcome statement onto the six outcome classes, or refuse.

    Prevents the two failures that make public activity data unusable.

    *An ambiguous statement must never become a negative.* A model trained on a
    table where ``n.d.`` was read as "no activity" learns that a large class of
    perfectly good enzymes does not work, and the error is invisible because the
    column looks complete. Every statement this function cannot resolve becomes
    ``NOT_TESTED`` with a recorded uncertainty naming why.

    *A negative without a detection limit is not interpretable.* "No product
    detected" at an unknown sensitivity excludes nothing, so such a statement is
    blocked rather than stored; ``proposed_outcome`` keeps the fact that the
    source said so. ``detection`` is accepted here, rather than looked up later,
    because the limit has to exist at the moment the class is assigned.

    A positive is blocked the same way when ``detection`` does not identify the
    product: a cofactor absorbance change is consistent with turnover on an
    impurity.

    *A negated positive is not a positive.* "0% conversion observed", "<1%
    conversion observed" and "no conversion observed" all contain a positive
    phrase, and a matcher that consumes the longest phrase and throws its
    position away reads all three as confirmed product. Negators and
    none-meaning quantities are therefore scanned first, with their positions,
    and a positive whose span a negator governs is never stored as a positive:
    a definite negator beside it makes it a negative (still subject to the
    detection-limit rule above), and anything near-zero or out of reach makes
    it ``NOT_TESTED`` with the reason recorded. "trace" and "<1%" are
    deliberately in the second group: at the noise floor a real small
    conversion and a blank are the same number.
    """
    if raw is None or not str(raw).strip():
        return OutcomeNormalisation(
            raw=raw,
            outcome=OutcomeClass.NOT_TESTED,
            proposed_outcome=None,
            confident=False,
            matched_rules=(),
            uncertainties=("the source carries no outcome statement; absence of "
                           "a statement is not a measured absence of activity",),
        )

    tokens = _tokenise(raw)
    clauses = _clause_ids(tokens)
    matches = _match_phrases(tokens)
    spans = [(m.start, m.end) for m in matches]
    # A negator swallowed by a phrase ("no" inside "no product detected") is
    # that phrase's own business; only the free ones can govern another span.
    negators = [g for g in _scan_negators(tokens)
                if not any(s <= g.index < e for s, e in spans)]
    bare = _bare_quantity_clauses(tokens, clauses, spans)

    uncertainties: list[str] = []

    if not matches:
        content_clauses = {clauses[i] for i, t in enumerate(tokens)
                           if t != _CLAUSE_BREAK}
        if negators and content_clauses and content_clauses <= bare:
            uncertainties.append(
                f"the source statement {raw!r} is a bare negator or quantity: "
                f"{_BARE_NEGATION_NOTE}")
        elif negators:
            named = ", ".join(sorted({f"'{_display(g.token)}'" for g in negators}))
            uncertainties.append(
                f"no rule matched the source statement {raw!r}; it carries a "
                f"negator or none-meaning quantity ({named}) but nothing this "
                f"function recognises as a statement of what was measured, so "
                f"it was not mapped onto an outcome class and a human must "
                f"read it")
        else:
            uncertainties.append(
                f"no rule matched the source statement {raw!r}; it was not mapped "
                f"onto an outcome class, and a human must read it")
        return OutcomeNormalisation(
            raw=raw, outcome=OutcomeClass.NOT_TESTED, proposed_outcome=None,
            confident=False, matched_rules=(),
            uncertainties=tuple(uncertainties))

    # -- negation scope ----------------------------------------------------
    # Each matched class is tested against the negators that govern its span.
    # A governed positive is never a positive: it is a negative when a definite
    # negator sits on it, and not_tested when the sense cannot be read off the
    # text. Nothing is decided by which token happened to survive the matcher.
    kept: list[_PhraseMatch] = []
    rule_names: list[str] = []
    negation_notes: list[str] = []
    negation_blocked: str | None = None

    for m in matches:
        if m.outcome is None:
            kept.append(m)
            rule_names.append(m.rule)
            continue
        governing = [(g, prox) for g, prox in
                     ((g, _governance(g, m.start, m.end, tokens, clauses, bare))
                      for g in negators) if prox]
        if not governing:
            kept.append(m)
            rule_names.append(m.rule)
            continue

        tokens_named = ", ".join(
            sorted({f"'{_display(g.token)}'" for g, _ in governing}))
        near = [g for g, _ in governing if g.kind == "near_zero"]
        definite_close = [g for g, prox in governing
                          if g.kind == "definite" and prox == "close"]

        if near:
            rule_names.append("negation_ambiguous")
            negation_notes.append(
                f"{tokens_named} governs '{m.phrase}' in {raw!r}: a quantity "
                f"at or near the noise floor is ambiguous between a small real "
                f"conversion and background, so the statement is neither a "
                f"positive nor a measured absence")
            negation_blocked = (
                "a near-zero or bounded quantity governs the statement; it "
                "resolves to neither a positive nor a negative")
        elif definite_close and m.outcome is OutcomeClass.CONFIRMED_TARGET_PRODUCT:
            kept.append(_PhraseMatch(
                rule="negated_positive",
                outcome=OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                phrase=m.phrase, start=m.start, end=m.end))
            rule_names.append("negated_positive")
            negation_notes.append(
                f"{tokens_named} negates '{m.phrase}' in {raw!r}; the statement "
                f"was read as an absence of the target product, never as the "
                f"positive the phrase alone would suggest")
        elif m.outcome is OutcomeClass.CONFIRMED_TARGET_PRODUCT:
            rule_names.append("negation_unresolved")
            negation_notes.append(
                f"{tokens_named} appears in the same clause as '{m.phrase}' in "
                f"{raw!r} but not beside it, so whether the positive is negated "
                f"cannot be read off the text; a human must read the source")
            negation_blocked = (
                "a negator shares the clause with the positive statement and "
                "the scope cannot be resolved by rule")
        else:
            rule_names.append("negation_unresolved")
            negation_notes.append(
                f"{tokens_named} governs '{m.phrase}' in {raw!r}, which is "
                f"itself a {m.outcome.value} statement; a double negation is "
                f"not resolved by rule")
            negation_blocked = (
                f"a negator governs a {m.outcome.value} statement; the double "
                f"negation is not resolved by rule")

    rules = tuple(dict.fromkeys(rule_names))
    uncertainties.extend(negation_notes)

    classes = {m.outcome for m in kept if m.outcome is not None}
    ambiguous_hits = [m.phrase for m in kept if m.outcome is None]

    if len(classes) > 1:
        names = ", ".join(sorted(c.value for c in classes))
        uncertainties.append(
            f"the source statement {raw!r} matches more than one outcome class "
            f"({names}); it is recorded as not_tested rather than resolved by "
            f"preferring one")
        return OutcomeNormalisation(
            raw=raw, outcome=OutcomeClass.NOT_TESTED, proposed_outcome=None,
            confident=False, matched_rules=rules,
            uncertainties=tuple(uncertainties))

    if not classes:
        for phrase in ambiguous_hits:
            uncertainties.append(
                f"{phrase!r}: {_AMBIGUITY_NOTES.get(phrase, 'meaning varies between sources')}")
        if not ambiguous_hits and negation_blocked is None:
            uncertainties.append(
                f"no outcome class survived reading {raw!r}; a human must read it")
        return OutcomeNormalisation(
            raw=raw, outcome=OutcomeClass.NOT_TESTED, proposed_outcome=None,
            confident=False, matched_rules=rules,
            uncertainties=tuple(uncertainties),
            blocked_reason=negation_blocked)

    proposed = next(iter(classes))

    # A known-ambiguous token alongside a resolved class still poisons it:
    # "no activity (n.d.)" is not a cleaner negative than "n.d." alone.
    if ambiguous_hits:
        for phrase in ambiguous_hits:
            uncertainties.append(
                f"{phrase!r} also appears in {raw!r}: "
                f"{_AMBIGUITY_NOTES.get(phrase, 'meaning varies between sources')}")
        return OutcomeNormalisation(
            raw=raw, outcome=OutcomeClass.NOT_TESTED, proposed_outcome=proposed,
            confident=False, matched_rules=rules,
            uncertainties=tuple(uncertainties),
            blocked_reason=(f"the statement resolves to {proposed.value} only if "
                            f"an ambiguous token is ignored"))

    # An unresolved negation elsewhere in the text cannot be ignored just
    # because some other phrase did resolve.
    if negation_blocked is not None:
        return OutcomeNormalisation(
            raw=raw, outcome=OutcomeClass.NOT_TESTED, proposed_outcome=proposed,
            confident=False, matched_rules=rules,
            uncertainties=tuple(uncertainties),
            blocked_reason=negation_blocked)

    blocked: str | None = None
    outcome = proposed

    if proposed is OutcomeClass.NO_TARGET_PRODUCT_DETECTED:
        if detection is None or detection.limit_of_detection is None:
            blocked = ("a negative record must carry the detection limit it is "
                       "negative at; without one the statement excludes nothing")
            uncertainties.append(
                "record the limit of detection and its unit, then re-run "
                "normalise_outcome; until then this row is not_tested, which is "
                "weaker than the source's own statement and deliberately so")
            outcome = OutcomeClass.NOT_TESTED
        elif not detection.limit_unit:
            blocked = "the detection limit has no unit, so it is not a limit"
            uncertainties.append("record limit_unit for the detection limit")
            outcome = OutcomeClass.NOT_TESTED
        elif not detection.method:
            uncertainties.append(
                "a detection limit is recorded but not the method that produced "
                "it; a curator must confirm which assay the limit belongs to")

    elif proposed is OutcomeClass.CONFIRMED_TARGET_PRODUCT:
        if detection is None or not detection.confirms_product_identity:
            blocked = ("a positive needs a detection method that identifies the "
                       "product; an indirect signal such as cofactor absorbance "
                       "is consistent with turnover on something else")
            uncertainties.append(
                "confirm the product with a method that identifies it "
                "(chiral GC/HPLC against an authentic standard, MS) before this "
                "row may be stored as confirmed_target_product")
            outcome = OutcomeClass.NOT_TESTED

    return OutcomeNormalisation(
        raw=raw, outcome=outcome, proposed_outcome=proposed,
        confident=blocked is None, matched_rules=rules,
        uncertainties=tuple(uncertainties), blocked_reason=blocked)


# ---------------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------------

#: Reaction classes that are chemically the reverse of one another. A record of
#: one is not evidence for the other, whatever its declared direction says.
REVERSIBLE_CLASS_PAIRS: tuple[frozenset[ReactionClass], ...] = (
    frozenset({ReactionClass.KETONE_TO_SECONDARY_ALCOHOL,
               ReactionClass.ALCOHOL_OXIDATION}),
    frozenset({ReactionClass.ALDEHYDE_TO_PRIMARY_ALCOHOL,
               ReactionClass.ALCOHOL_OXIDATION}),
)


def reverse_class_of(cls: ReactionClass) -> tuple[ReactionClass, ...]:
    """Reaction classes that run the opposite way to ``cls``.

    Exists so the oxidation/reduction confusion is a lookup rather than a
    condition repeated at every call site, where one copy will be forgotten.
    """
    out: list[ReactionClass] = []
    for pair in REVERSIBLE_CLASS_PAIRS:
        if cls in pair:
            out.extend(sorted((c for c in pair if c is not cls), key=lambda c: c.value))
    return tuple(dict.fromkeys(out))


@dataclass(frozen=True)
class DirectionVerdict:
    """Whether a record's measured direction supports the target reaction."""

    record_id: str
    measured_direction: ReactionDirection
    target_class: ReactionClass | None
    measured_class: ReactionClass | None
    supports: bool
    is_reverse: bool
    is_unspecified: bool
    reasons: tuple[str, ...]

    @property
    def non_supporting(self) -> bool:
        """Convenience for filters: the record may not be counted as support."""
        return not self.supports

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "measured_direction": self.measured_direction.value,
            "target_class": self.target_class.value if self.target_class else None,
            "measured_class": self.measured_class.value if self.measured_class else None,
            "supports": self.supports,
            "is_reverse": self.is_reverse,
            "is_unspecified": self.is_unspecified,
            "reasons": list(self.reasons),
        }

    def describe(self) -> str:
        head = "supports target direction" if self.supports \
            else "DOES NOT support the target direction"
        return f"{self.record_id}: {head}\n" + "\n".join(
            f"  - {r}" for r in self.reasons)


def _target_reaction_class(target: Any) -> ReactionClass | None:
    """Read a reaction class out of a ReactionSpec, TaskSpec, enum or string."""
    if target is None:
        return None
    if isinstance(target, ReactionClass):
        return target
    for attr in ("reaction_class",):
        v = getattr(target, attr, None)
        if isinstance(v, ReactionClass):
            return v
        if isinstance(v, str):
            try:
                return ReactionClass(v)
            except ValueError:
                return None
    inner = getattr(target, "reaction", None)
    if inner is not None and inner is not target:
        return _target_reaction_class(inner)
    if isinstance(target, str):
        try:
            return ReactionClass(target)
        except ValueError:
            return None
    return None


def direction_check(record: Any, target_reaction: Any) -> DirectionVerdict:
    """Flag records measured in the reverse of the target direction.

    An alcohol dehydrogenase assayed on the alcohol, following NAD+ reduction,
    is a superb record of the oxidation and says nothing dependable about the
    ketone reduction at the target pH, with the target cofactor, at the target
    substrate concentration. Thermodynamics and the enzyme's kinetics both
    differ. Those records nevertheless arrive from curated resources with the
    same EC number and the same substrate name, so without this check they
    enter a seed set as positives.

    Two independent signals are used, and a disagreement between them is itself
    a refusal: the declared :class:`~eagent.schemas.record.ReactionDirection` on
    the record, and whether the record's reaction class is the chemical reverse
    of the target's. ``UNSPECIFIED`` is non-supporting, because an unrecorded
    direction is not a forward one.
    """
    rid = str(getattr(record, "record_id", None)
              or getattr(record, "intake_id", None)
              or getattr(record, "candidate_id", None) or "<record>")
    inner = getattr(record, "record", None)
    if inner is not None and hasattr(inner, "reaction_direction"):
        record = inner

    measured_dir = getattr(record, "reaction_direction", ReactionDirection.UNSPECIFIED)
    if not isinstance(measured_dir, ReactionDirection):
        try:
            measured_dir = ReactionDirection(str(measured_dir))
        except ValueError:
            measured_dir = ReactionDirection.UNSPECIFIED

    measured_class = getattr(record, "reaction_class", None)
    if not isinstance(measured_class, ReactionClass):
        measured_class = None
    target_class = _target_reaction_class(target_reaction)

    reasons: list[str] = []
    is_reverse = False
    is_unspecified = False

    class_is_reverse = bool(
        measured_class and target_class
        and measured_class in reverse_class_of(target_class))
    if class_is_reverse:
        is_reverse = True
        reasons.append(
            f"the record measures {measured_class.value}, which is the chemical "
            f"reverse of the target {target_class.value}; a measurement of the "
            f"reverse reaction is not evidence for the forward one under the "
            f"target conditions")

    if measured_dir is ReactionDirection.REVERSE_OF_TARGET:
        is_reverse = True
        reasons.append(
            "the record declares reaction_direction=reverse_of_target")
    elif measured_dir is ReactionDirection.UNSPECIFIED:
        is_unspecified = True
        reasons.append(
            "the record declares no reaction direction; an unrecorded direction "
            "is not a forward one, so this row cannot be counted as support "
            "until a curator reads the assay description")
    elif measured_dir is ReactionDirection.REVERSIBLE_BOTH_SHOWN:
        reasons.append(
            "both directions were demonstrated on this record")
    elif measured_dir is ReactionDirection.FORWARD_AS_TARGET:
        reasons.append("the record declares reaction_direction=forward_as_target")

    if class_is_reverse and measured_dir.supports_target_direction:
        reasons.append(
            f"the declared direction ({measured_dir.value}) contradicts the "
            f"reaction class pair; the contradiction is itself a reason to "
            f"refuse the record until a curator resolves it")

    if target_class is None:
        reasons.append(
            "the target reaction class could not be read from the task, so the "
            "class-pair check was not performed")
    if measured_class is None:
        reasons.append(
            "the record carries no reaction class, so only its declared "
            "direction was checked")

    supports = (measured_dir.supports_target_direction
                and not is_reverse
                and not is_unspecified)
    return DirectionVerdict(
        record_id=rid,
        measured_direction=measured_dir,
        target_class=target_class,
        measured_class=measured_class,
        supports=supports,
        is_reverse=is_reverse,
        is_unspecified=is_unspecified,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class IntakeStore:
    """Tier-partitioned storage. Nothing here returns a merged table by accident.

    Exists because the usual container -- a list of records with a ``tier``
    column -- makes merging the default and keeping them apart the thing you
    have to remember. Here iteration is per partition, and the one method that
    pools (:meth:`pool`) requires the caller to name the tiers and refuses to
    put model inferences in the same bag as measurements unless asked in so many
    words.
    """

    def __init__(self, records: Iterable[IntakeRecord] = ()) -> None:
        self._partitions: dict[EvidenceTier, list[IntakeRecord]] = {
            t: [] for t in TIER_ORDER}
        self._seen: set[str] = set()
        for r in records:
            self.add(r)

    def add(self, record: IntakeRecord) -> IntakeRecord:
        """File a record in its tier's partition, refusing a duplicate id."""
        if record.intake_id in self._seen:
            raise IntakeError(
                f"intake_id '{record.intake_id}' is already stored; a second row "
                f"under the same id would make the audit trail ambiguous")
        self._seen.add(record.intake_id)
        self._partitions[record.tier].append(record)
        return record

    def replace(self, record: IntakeRecord) -> IntakeRecord:
        """Re-file a record after a promotion changed its tier.

        Separate from :meth:`add` so that moving a row between partitions is an
        explicit act with the old row removed, rather than the same record
        existing at two tiers at once.
        """
        for tier, bucket in self._partitions.items():
            for i, existing in enumerate(bucket):
                if existing.intake_id == record.intake_id:
                    del bucket[i]
                    self._partitions[record.tier].append(record)
                    return record
        raise IntakeError(
            f"intake_id '{record.intake_id}' is not stored, so there is nothing "
            f"to replace; use add()")

    def of_tier(self, tier: EvidenceTier) -> tuple[IntakeRecord, ...]:
        return tuple(self._partitions[EvidenceTier(tier)])

    def partitions(self) -> dict[str, tuple[IntakeRecord, ...]]:
        """Storage partitions by name, in tier order, including empty ones."""
        return {t.storage_partition: tuple(self._partitions[t]) for t in TIER_ORDER}

    def counts(self) -> dict[EvidenceTier, int]:
        return {t: len(self._partitions[t]) for t in TIER_ORDER}

    def pool(
        self,
        tiers: Sequence[EvidenceTier],
        *,
        justification: str,
        allow_model_inferred: bool = False,
    ) -> list[IntakeRecord]:
        """Return records from the named tiers, pooled on purpose.

        The caller names the tiers and says why. Pooling model inferences with
        measurements additionally needs ``allow_model_inferred=True``, because
        that is the merge that turns a dataset into a mixture of observations
        and guesses with no column that records which is which.
        """
        wanted = [EvidenceTier(t) for t in tiers]
        if not wanted:
            raise TierMergeRefusedError(
                "pool() needs the tiers named explicitly; there is no default "
                "set, because the default would become a silent merge")
        if not " ".join((justification or "").split()):
            raise TierMergeRefusedError(
                "pooling tiers requires a justification recording what the "
                "pooled set is for")
        if EvidenceTier.MODEL_INFERRED in wanted and not allow_model_inferred:
            raise TierMergeRefusedError(
                f"refusing to pool model_inferred rows with evidence. "
                f"{MODEL_INFERENCE_IS_NOT_EVIDENCE} Pass "
                f"allow_model_inferred=True if the pooled set is explicitly a "
                f"mixture for triage.")
        out: list[IntakeRecord] = []
        for t in TIER_ORDER:
            if t in wanted:
                out.extend(self._partitions[t])
        return out

    def __len__(self) -> int:
        return sum(len(v) for v in self._partitions.values())

    def __iter__(self) -> Iterator[IntakeRecord]:
        """Iterate in tier order. Iteration never merges: the tier is on the row."""
        for t in TIER_ORDER:
            yield from self._partitions[t]

    def report_lines(self) -> list[str]:
        lines = [f"{len(self)} intake record(s) in {len(TIER_ORDER)} partitions"]
        for t in TIER_ORDER:
            lines.append(f"  {t.storage_partition:<40} {len(self._partitions[t]):>4}")
        return lines


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverageRequirement:
    """One thing a record needs before it counts as fully specified.

    Named and carried as data so the coverage report can say *which* field is
    missing across the corpus. "Forty records, three of them usable" is only
    actionable when the report also says what the other thirty-seven lack.
    """

    name: str
    why: str
    test: Callable[[IntakeRecord], bool]


def _has_sequence(r: IntakeRecord) -> bool:
    rec = r.record
    return bool(rec.construct_sequence or rec.sequence or rec.sequence_sha256)


def _has_substrate_structure(r: IntakeRecord) -> bool:
    sub = r.record.substrate
    return bool(getattr(sub, "inchikey", None) or getattr(sub, "isomeric_smiles", None))


def _direction_recorded(r: IntakeRecord) -> bool:
    """Whether a direction is written down at all. A *statistic*, not a check.

    Kept because "how many rows say nothing about direction" is worth knowing,
    and kept separate because presence is not support: a row declaring
    ``reverse_of_target`` has a direction recorded and is still a measurement
    of the opposite reaction.
    """
    return r.record.reaction_direction is not ReactionDirection.UNSPECIFIED


def _direction_requirement(target_reaction: Any = None) -> CoverageRequirement:
    """The direction axis, bound to a target reaction when one is supplied.

    With a target it asks :func:`direction_check` whether the row *supports*
    that reaction. Without one it can only ask whether a direction is recorded,
    which is why :func:`coverage` additionally reports that the axis went
    unassessed: an oxidation measurement passes the presence test and the
    requirement's own text says an oxidation measurement is not reduction
    evidence.
    """
    if target_reaction is None:
        return CoverageRequirement(
            name="reaction_direction",
            why="an oxidation measurement is not reduction evidence, and an "
                "unrecorded direction cannot be assumed to be the target one; "
                "with no target reaction supplied only the presence of a "
                "direction could be checked, not whether it supports anything",
            test=_direction_recorded)

    def _supports(r: IntakeRecord, _target: Any = target_reaction) -> bool:
        try:
            return bool(direction_check(r.record, _target).supports)
        except Exception:
            return False

    return CoverageRequirement(
        name="reaction_direction",
        why="an oxidation measurement is not reduction evidence, and an "
            "unrecorded direction cannot be assumed to be the target one; the "
            "row must be measured in a direction that supports the target "
            "reaction, not merely carry a direction field",
        test=_supports)


def _has_conditions(r: IntakeRecord) -> bool:
    c = r.record.conditions
    return c.pH is not None and c.temperature_C is not None


def _has_detection(r: IntakeRecord) -> bool:
    d = r.record.detection
    if r.record.outcome is OutcomeClass.NO_TARGET_PRODUCT_DETECTED:
        return bool(d.method) and d.limit_of_detection is not None
    if r.record.outcome.is_positive:
        return bool(d.method) and d.confirms_product_identity
    return bool(d.method)


def _informative_outcome(r: IntakeRecord) -> bool:
    return r.record.outcome.informs_catalytic_ability


def _has_traceable_evidence(r: IntakeRecord) -> bool:
    return any((e.identifier or "").strip() for e in r.record.evidence)


#: The closed checklist used by :func:`coverage`.
FULL_SPECIFICATION_REQUIREMENTS: tuple[CoverageRequirement, ...] = (
    CoverageRequirement(
        name="sequence_identity",
        why="without the sequence actually expressed, the row cannot be joined "
            "to a candidate; an engineered construct has no accession to fall "
            "back on",
        test=_has_sequence),
    CoverageRequirement(
        name="substrate_structure",
        why="a substrate name is not a structure; two papers using the same "
            "trivial name routinely mean different compounds",
        test=_has_substrate_structure),
    # Replaced by a target-bound version in requirements_for(); on its own it
    # can only report that a direction was recorded.
    _direction_requirement(None),
    CoverageRequirement(
        name="conditions",
        why="pH and temperature are part of the record key; an activity without "
            "them cannot be compared with another activity",
        test=_has_conditions),
    CoverageRequirement(
        name="detection",
        why="the outcome class is only as good as the method behind it: a "
            "negative needs a limit and a positive needs product identification",
        test=_has_detection),
    CoverageRequirement(
        name="informative_outcome",
        why="not_tested, a computational failure and a computational negative "
            "say nothing about catalytic ability",
        test=_informative_outcome),
    CoverageRequirement(
        name="traceable_evidence",
        why="a claim nobody can re-read is not checkable, so it cannot be "
            "defended when the batch fails",
        test=_has_traceable_evidence),
)


def requirements_for(target_reaction: Any = None) -> tuple[CoverageRequirement, ...]:
    """The full-specification checklist, with the direction axis bound.

    :data:`FULL_SPECIFICATION_REQUIREMENTS` is the closed list of axes; this is
    the list as it applies to a particular campaign. Only the direction axis
    differs, and it differs in the only way that matters: with a target it
    tests support, without one it can test nothing more than presence.
    """
    direction = _direction_requirement(target_reaction)
    return tuple(direction if req.name == "reaction_direction" else req
                 for req in FULL_SPECIFICATION_REQUIREMENTS)


#: Said in the coverage report, and in the refusal to call such a report clean.
DIRECTION_AXIS_UNASSESSED = (
    "the direction axis could not be assessed: no target reaction was supplied, "
    "so rows were only checked for carrying a direction at all. A record "
    "declaring reverse_of_target passes that check and is a measurement of the "
    "opposite reaction. Pass target_reaction= to assess it."
)


@dataclass(frozen=True)
class IntakeCoverage:
    """Evidence or rows: how many records reach each tier and how many are usable.

    Exists because a count of records is the number everybody quotes and the
    number that means least. Four hundred rows of which none carry a detection
    method and none a structure-defined substrate is a corpus with no evidence
    in it, and it looks identical to a good one in every summary that reports
    only a total.
    """

    n_records: int
    by_tier: Mapping[EvidenceTier, int]
    usable_as_label_by_tier: Mapping[EvidenceTier, int]
    fully_specified_ids: tuple[str, ...]
    missing_by_requirement: Mapping[str, int]
    missing_by_record: Mapping[str, tuple[str, ...]]
    direction_non_supporting_ids: tuple[str, ...]
    needs_curation_ids: tuple[str, ...]
    #: Whether the direction axis was tested for *support* of a target
    #: reaction. False means no target was supplied and the axis reports only
    #: that a direction exists, which a reverse-direction row satisfies.
    direction_axis_assessed: bool = False
    #: Records carrying any direction at all. Kept apart from the axis above,
    #: because presence is not support.
    direction_recorded_ids: tuple[str, ...] = ()
    #: Coverage-level caveats: things this report could not check.
    notes: tuple[str, ...] = ()

    @property
    def n_fully_specified(self) -> int:
        """The intersection: records meeting every requirement at once."""
        return len(self.fully_specified_ids)

    @property
    def n_direction_recorded(self) -> int:
        """How many rows record a direction at all, supporting or not."""
        return len(self.direction_recorded_ids)

    @property
    def has_evidence(self) -> bool:
        """True only when at least one fully-specified, human-checked row exists.

        False whenever the direction axis went unassessed, however good the
        rest of the report looks. A corpus whose rows might all be measurements
        of the reverse reaction is not a corpus with evidence in it, and a
        report that cannot tell is not entitled to say otherwise.
        """
        return (self.direction_axis_assessed
                and self.n_fully_specified > 0
                and self.usable_as_label_by_tier.get(
                    EvidenceTier.EXPERT_VERIFIED_PRIMARY, 0) > 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_records": self.n_records,
            "by_tier": {t.value: self.by_tier.get(t, 0) for t in TIER_ORDER},
            "usable_as_label_by_tier": {
                t.value: self.usable_as_label_by_tier.get(t, 0) for t in TIER_ORDER},
            "n_fully_specified": self.n_fully_specified,
            "fully_specified_ids": list(self.fully_specified_ids),
            "missing_by_requirement": dict(self.missing_by_requirement),
            "missing_by_record": {k: list(v) for k, v in self.missing_by_record.items()},
            "direction_non_supporting_ids": list(self.direction_non_supporting_ids),
            "direction_axis_assessed": self.direction_axis_assessed,
            "direction_recorded_ids": list(self.direction_recorded_ids),
            "n_direction_recorded": self.n_direction_recorded,
            "needs_curation_ids": list(self.needs_curation_ids),
            "notes": list(self.notes),
            "has_evidence": self.has_evidence,
        }

    def report_lines(self) -> list[str]:
        lines = [f"intake coverage: {self.n_records} record(s)"]
        for t in TIER_ORDER:
            lines.append(
                f"  {t.value:<28} {self.by_tier.get(t, 0):>4} "
                f"({self.usable_as_label_by_tier.get(t, 0)} usable as labels)")
        lines.append(f"  fully specified (all {len(FULL_SPECIFICATION_REQUIREMENTS)} "
                     f"requirements): {self.n_fully_specified}")
        for req in FULL_SPECIFICATION_REQUIREMENTS:
            n = self.missing_by_requirement.get(req.name, 0)
            if n:
                lines.append(f"    missing {req.name:<22} {n:>4}  ({req.why})")
        lines.append(f"  direction recorded at all: {self.n_direction_recorded} "
                     f"(a recorded direction is not a supporting one)")
        if self.direction_non_supporting_ids:
            lines.append(f"  measured in a non-supporting direction: "
                         f"{len(self.direction_non_supporting_ids)}")
        if self.needs_curation_ids:
            lines.append(f"  flagged needs_curation: {len(self.needs_curation_ids)}")
        for note in self.notes:
            lines.append(f"  NOT ASSESSED: {note}")
        if not self.has_evidence:
            if not self.direction_axis_assessed:
                lines.append("  VERDICT: rows, not evidence -- the direction "
                             "axis was not assessed, so no row can be called "
                             "fully specified for a target reaction")
            else:
                lines.append("  VERDICT: rows, not evidence -- no fully-specified, "
                             "human-checked record is present")
        return lines

    def describe(self) -> str:
        return "\n".join(self.report_lines())


def coverage(
    records: Iterable[IntakeRecord],
    *,
    target_reaction: Any = None,
) -> IntakeCoverage:
    """Report tier counts and the fully-specified intersection for a record set.

    The intersection is the point. Per-field completeness percentages look
    healthy when every field is eighty per cent complete and no single record
    has all of them, which is the normal state of a corpus assembled from
    several databases. A campaign needs to know how many rows it can actually
    reason with, which is the size of the intersection, not the average.

    Passing ``target_reaction`` is what makes the direction axis assessable.
    Without it the axis can only ask whether a direction is recorded -- which a
    ``reverse_of_target`` row satisfies while being a measurement of the
    opposite reaction -- so the report says the axis went unassessed and
    refuses to call the corpus evidence. The weaker statistic, how many rows
    record a direction at all, is reported separately and named as such.
    """
    items = list(records)
    requirements = requirements_for(target_reaction)
    assessed = target_reaction is not None
    notes: list[str] = [] if assessed else [DIRECTION_AXIS_UNASSESSED]
    direction_recorded: list[str] = []
    by_tier: dict[EvidenceTier, int] = {t: 0 for t in TIER_ORDER}
    usable: dict[EvidenceTier, int] = {t: 0 for t in TIER_ORDER}
    missing_counts: dict[str, int] = {r.name: 0 for r in requirements}
    missing_by_record: dict[str, tuple[str, ...]] = {}
    full: list[str] = []
    non_supporting: list[str] = []
    needs_curation: list[str] = []

    for item in items:
        by_tier[item.tier] = by_tier.get(item.tier, 0) + 1
        if item.is_usable_as_label:
            usable[item.tier] = usable.get(item.tier, 0) + 1
        if item.needs_curation:
            needs_curation.append(item.intake_id)

        if _direction_recorded(item):
            direction_recorded.append(item.intake_id)

        missing: list[str] = []
        for req in requirements:
            try:
                ok = bool(req.test(item))
            except Exception:
                ok = False
            if not ok:
                missing.append(req.name)
                missing_counts[req.name] = missing_counts.get(req.name, 0) + 1
        if missing:
            missing_by_record[item.intake_id] = tuple(missing)
        else:
            full.append(item.intake_id)

        if target_reaction is not None:
            if direction_check(item.record, target_reaction).non_supporting:
                non_supporting.append(item.intake_id)

    return IntakeCoverage(
        n_records=len(items),
        by_tier=by_tier,
        usable_as_label_by_tier=usable,
        fully_specified_ids=tuple(full),
        missing_by_requirement=missing_counts,
        missing_by_record=missing_by_record,
        direction_non_supporting_ids=tuple(non_supporting),
        needs_curation_ids=tuple(needs_curation),
        direction_axis_assessed=assessed,
        direction_recorded_ids=tuple(direction_recorded),
        notes=tuple(notes),
    )
