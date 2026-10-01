"""Mutation proposals for substrate-directed engineering.

A proposal must state what it is expected to improve *and* what it may cost,
because a variant that gains activity and loses expression is a different
result from one that gains both, and only a pre-stated expectation makes the
difference interpretable.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .candidate import ConfidenceLevel


class Mutation(BaseModel):
    """A single substitution, carried in both numbering systems."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    wild_type: str
    position_author: int = Field(..., description="Position in the reference numbering")
    position_index: int = Field(..., description="0-based index in the parent sequence")
    mutant: str

    def __str__(self) -> str:
        return f"{self.wild_type}{self.position_author}{self.mutant}"

    @model_validator(mode="after")
    def _single_letters(self) -> "Mutation":
        for aa in (self.wild_type, self.mutant):
            if len(aa) != 1 or aa not in "ACDEFGHIKLMNPQRSTVWY":
                raise ValueError(f"not a standard residue letter: {aa!r}")
        if self.wild_type == self.mutant:
            raise ValueError(f"{self} is not a substitution")
        return self


class SiteEvidence(BaseModel):
    """Why a position is worth mutating. Proximity alone is not a reason."""

    model_config = ConfigDict(extra="forbid")

    structural: list[str] = Field(
        default_factory=list,
        description="contact with substrate, pocket entrance, orienting loop, "
                    "steric clash, cofactor-adjacent",
    )
    family: list[str] = Field(
        default_factory=list,
        description="systematic differences between subfamilies with different "
                    "substrate ranges; co-varying positions in active clades",
    )
    experimental: list[str] = Field(
        default_factory=list,
        description="mutations already shown to change activity, selectivity, "
                    "stability or cofactor preference",
    )
    distance_to_substrate_A: float | None = None

    @property
    def n_evidence_classes(self) -> int:
        return sum(1 for s in (self.structural, self.family, self.experimental) if s)

    @property
    def is_proximity_only(self) -> bool:
        return self.n_evidence_classes <= 1 and bool(self.structural) \
            and not self.family and not self.experimental


class MutationProposal(BaseModel):
    """One variant to build and test."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    parent_candidate_id: str
    parent_sequence_sha256: str
    mutations: list[Mutation] = Field(..., min_length=1)
    numbering_reference: str = Field(
        ..., description="Which numbering the author positions refer to, e.g. "
                         "'PDB 1XYZ chain A' or 'parent sequence 1-based'"
    )
    site_evidence: dict[str, SiteEvidence] = Field(
        default_factory=dict, description="keyed by str(mutation position_author)"
    )
    intended_improvement: list[str] = Field(..., min_length=1)
    possible_cost: list[str] = Field(default_factory=list)
    supporting_models: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    frozen_roles_respected: bool = True
    generator: str | None = Field(None, description="e.g. ligandmpnn, rational, literature")
    confidence: ConfidenceLevel = ConfidenceLevel.WEAK
    experimental_priority: int = Field(99, ge=1)
    decomposition_controls: list[str] = Field(
        default_factory=list,
        description="Single-mutant proposal ids needed to attribute a combination's effect",
    )

    @property
    def is_combination(self) -> bool:
        return len(self.mutations) > 1

    @model_validator(mode="after")
    def _combination_needs_decomposition(self) -> "MutationProposal":
        if self.is_combination and not self.decomposition_controls:
            raise ValueError(
                f"{self.proposal_id}: a combination variant needs single-mutant "
                f"controls, otherwise an improvement cannot be attributed"
            )
        if not self.possible_cost:
            raise ValueError(
                f"{self.proposal_id}: state at least one property this change "
                f"might damage"
            )
        return self

    def label(self) -> str:
        return "/".join(str(m) for m in self.mutations)

    def apply_to(self, parent_sequence: str) -> str:
        """Build the variant sequence, verifying every wild-type letter first."""
        seq = list(parent_sequence)
        for m in self.mutations:
            if not 0 <= m.position_index < len(seq):
                raise ValueError(f"{m}: index {m.position_index} outside parent")
            if seq[m.position_index] != m.wild_type:
                raise ValueError(
                    f"{m}: parent has {seq[m.position_index]} at index "
                    f"{m.position_index}, not {m.wild_type}; numbering is wrong"
                )
            seq[m.position_index] = m.mutant
        return "".join(seq)
