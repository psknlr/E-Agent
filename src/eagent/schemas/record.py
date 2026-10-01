"""The minimal data unit: one sequence, one substrate, one condition set, one outcome.

The same protein measured against a second substrate, a second cofactor, or a
second solvent is a *different* record. Collapsing those into one activity
label is what makes public enzyme datasets unusable for substrate-level
prediction, so the model forbids it structurally.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..provenance import sequence_hash
from .chem import CofactorSpec, ProductSpec, Stereochemistry, SubstrateSpec
from .reaction import Conditions, ReactionClass


class OutcomeClass(str, enum.Enum):
    """What a record is entitled to claim.

    The distinction between "not tested", "tested and not detected", and
    "the protein never expressed" carries most of the information in a
    screening campaign, and all three collapse to 0 in a binary label.
    """

    CONFIRMED_TARGET_PRODUCT = "confirmed_target_product"
    NO_TARGET_PRODUCT_DETECTED = "no_target_product_detected"
    EXPRESSION_OR_SOLUBILITY_FAILURE = "expression_or_solubility_failure"
    OTHER_PRODUCT_OR_WRONG_CONFIGURATION = "other_product_or_wrong_configuration"
    NOT_TESTED = "not_tested"
    COMPUTATIONAL_FAILURE = "computational_failure"
    COMPUTATIONAL_NEGATIVE = "computational_negative"

    @property
    def is_experimental(self) -> bool:
        return self not in (
            OutcomeClass.NOT_TESTED,
            OutcomeClass.COMPUTATIONAL_FAILURE,
            OutcomeClass.COMPUTATIONAL_NEGATIVE,
        )

    @property
    def is_positive(self) -> bool:
        return self is OutcomeClass.CONFIRMED_TARGET_PRODUCT

    @property
    def informs_catalytic_ability(self) -> bool:
        """Expression failure says nothing about whether the enzyme can catalyse."""
        return self in (
            OutcomeClass.CONFIRMED_TARGET_PRODUCT,
            OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
            OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION,
        )

    def claim(self) -> str:
        return _CLAIMS[self]


_CLAIMS: dict[OutcomeClass, str] = {
    OutcomeClass.CONFIRMED_TARGET_PRODUCT:
        "target reaction activity exists under the recorded conditions",
    OutcomeClass.NO_TARGET_PRODUCT_DETECTED:
        "no activity detected under these conditions at this detection limit",
    OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE:
        "this construct did not pass expression; catalytic ability undetermined",
    OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION:
        "turnover occurred but does not meet the target reaction requirement",
    OutcomeClass.NOT_TESTED: "unknown",
    OutcomeClass.COMPUTATIONAL_FAILURE:
        "modelling or tooling produced no usable result; says nothing about the enzyme",
    OutcomeClass.COMPUTATIONAL_NEGATIVE:
        "model training label only; not an experimental negative",
}


class ReactionDirection(str, enum.Enum):
    """Which way the measured reaction ran.

    A record of alcohol oxidation is not evidence that the same enzyme performs
    the reduction under the target conditions. Many curated resources store a
    reference direction that differs from the direction actually assayed, so the
    direction travels with the record rather than being inferred from a label.
    """

    FORWARD_AS_TARGET = "forward_as_target"
    REVERSE_OF_TARGET = "reverse_of_target"
    REVERSIBLE_BOTH_SHOWN = "reversible_both_shown"
    UNSPECIFIED = "unspecified"

    @property
    def supports_target_direction(self) -> bool:
        return self in (ReactionDirection.FORWARD_AS_TARGET,
                        ReactionDirection.REVERSIBLE_BOTH_SHOWN)


class EvidenceStrength(str, enum.Enum):
    """How tightly the claim is bound to this exact sequence."""

    SEQUENCE_LEVEL_EXPERIMENTAL = "sequence_level_experimental"
    HOMOLOG_EXPERIMENTAL = "homolog_experimental"
    EC_SPECIES_MAPPED = "ec_species_mapped"
    ANNOTATION_ONLY = "annotation_only"
    COMPUTATIONAL_CONSTRUCT = "computational_construct"

    @property
    def rank(self) -> int:
        return _STRENGTH_RANK[self]

    @property
    def is_sequence_level(self) -> bool:
        return self is EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL


_STRENGTH_RANK = {
    EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL: 4,
    EvidenceStrength.HOMOLOG_EXPERIMENTAL: 3,
    EvidenceStrength.EC_SPECIES_MAPPED: 2,
    EvidenceStrength.ANNOTATION_ONLY: 1,
    EvidenceStrength.COMPUTATIONAL_CONSTRUCT: 0,
}


class EvidenceRef(BaseModel):
    """Where a claim came from, precise enough to re-read."""

    model_config = ConfigDict(extra="forbid")

    source_type: str                      # publication | database | internal_experiment
    identifier: str                       # PMID / DOI / accession / assay run id
    locator: str | None = Field(
        None, description="Supplementary table, figure, or page within the source"
    )
    strength: EvidenceStrength = EvidenceStrength.ANNOTATION_ONLY
    extracted_by: str | None = None       # human | model-assisted
    verified_by: str | None = None
    quote: str | None = None
    retrieved_at: str | None = None
    database_version: str | None = None
    source_doi: str | None = None
    source_record_id: str | None = None
    license: str | None = Field(
        None, description="Licence of the source record, carried so redistribution "
                          "terms travel with the data rather than being looked up later."
    )
    upstream_sources: list[str] = Field(
        default_factory=list,
        description="Resources this record was re-integrated from. Four databases "
                    "carrying one re-curated measurement are one piece of evidence, "
                    "not four, and the lineage layer uses this to say so.",
    )
    experiment_activity_id: str | None = Field(
        None, description="Identifier of the measurement campaign, used to group "
                          "records that are not independent."
    )

    @model_validator(mode="after")
    def _model_extraction_needs_review(self) -> "EvidenceRef":
        if (self.extracted_by or "").startswith("model") and self.strength.is_sequence_level \
                and not self.verified_by:
            raise ValueError(
                "a model-extracted record may not be promoted to sequence-level "
                "experimental evidence without a human verifier"
            )
        return self


class Detection(BaseModel):
    """How the outcome was measured. A negative is meaningless without this."""

    model_config = ConfigDict(extra="forbid")

    method: str | None = None             # GC-MS, chiral HPLC, NADPH A340, ...
    limit_of_detection: float | None = None
    limit_unit: str | None = None
    authentic_standard: bool | None = None
    confirms_product_identity: bool = Field(
        False,
        description="A cofactor absorbance change alone does not confirm the "
                    "product; only a method that identifies the product does.",
    )
    chiral_method_validated: bool | None = None


class ExperimentRecord(BaseModel):
    """Record = (Sequence, Substrate, Reaction, Cofactor, Conditions, Outcome, Evidence)."""

    model_config = ConfigDict(extra="forbid")

    record_id: str
    # -- sequence identity
    sequence: str | None = None
    sequence_sha256: str | None = None
    accession: str | None = None
    database_version: str | None = None
    is_variant: bool = False
    parent_sequence_sha256: str | None = None
    mutations: list[str] = Field(default_factory=list)
    construct_description: str | None = Field(
        None, description="Tags, truncations and fusions actually expressed"
    )
    construct_sequence: str | None = Field(
        None, description="The sequence actually expressed. An engineered enzyme "
                          "often has no accession, so a database identifier cannot "
                          "serve as its identity."
    )
    # -- chemistry
    substrate: SubstrateSpec = Field(default_factory=SubstrateSpec)
    product_observed: ProductSpec | None = None
    reaction_class: ReactionClass = ReactionClass.OTHER
    reaction_direction: ReactionDirection = ReactionDirection.UNSPECIFIED
    reaction_id: str | None = Field(None, description="e.g. a Rhea identifier")
    cofactor: CofactorSpec | None = None
    conditions: Conditions = Field(default_factory=Conditions)
    # -- outcome
    outcome: OutcomeClass = OutcomeClass.NOT_TESTED
    detection: Detection = Field(default_factory=Detection)
    conversion_pct: float | None = None
    ee_target_pct: float | None = Field(
        None, ge=-100.0, le=100.0,
        description="Signed toward the target enantiomer; negative means the "
                    "opposite configuration dominated.",
    )
    specific_activity: float | None = None
    specific_activity_unit: str | None = None
    measurement_type: str | None = Field(
        None, description="What was actually measured: conversion, initial rate, "
                          "specific activity, growth, binding. Endpoints differ and "
                          "must not be pooled onto one numeric scale."
    )
    measurement_value: float | None = None
    measurement_unit: str | None = None
    kcat_s: float | None = None
    km_mM: float | None = None
    soluble_expression: bool | None = None
    # -- evidence
    evidence: list[EvidenceRef] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _derive_and_check(self) -> "ExperimentRecord":
        if self.sequence and not self.sequence_sha256:
            object.__setattr__(self, "sequence_sha256", sequence_hash(self.sequence))
        if self.outcome.is_positive and not self.detection.confirms_product_identity:
            raise ValueError(
                "confirmed_target_product requires a detection method that "
                "identifies the product, not an indirect signal"
            )
        if self.outcome is OutcomeClass.NO_TARGET_PRODUCT_DETECTED \
                and self.detection.limit_of_detection is None:
            raise ValueError(
                "a negative record must carry the detection limit it is negative at"
            )
        if self.is_variant and not self.parent_sequence_sha256:
            raise ValueError("a variant record must name its parent sequence hash")
        if (self.kcat_s is not None or self.km_mM is not None) \
                and self.conversion_pct is not None and self.specific_activity is None:
            # Not fatal, but worth refusing silently-derived constants.
            pass
        return self

    @property
    def max_strength(self) -> EvidenceStrength:
        if not self.evidence:
            return EvidenceStrength.COMPUTATIONAL_CONSTRUCT
        return max((e.strength for e in self.evidence), key=lambda s: s.rank)

    def group_key(self) -> tuple:
        """Identity used for leakage-controlled splits and de-duplication."""
        return (
            self.parent_sequence_sha256 or self.sequence_sha256,
            self.substrate.inchikey or self.substrate.isomeric_smiles,
            self.cofactor.describe() if self.cofactor else None,
            self.conditions.key(),
        )


def ee_target(n_target: float, n_opposite: float) -> float:
    """Signed enantiomeric excess toward the target configuration, in percent.

    Unlike an absolute ee, this goes negative when the wrong enantiomer wins,
    so a highly selective failure cannot be reported as a success.
    """
    total = n_target + n_opposite
    if total <= 0:
        raise ValueError("no product quantified; ee is undefined")
    return (n_target - n_opposite) / total * 100.0
