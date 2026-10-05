"""Candidates: sequence, family call, structure, complex poses, scorecard.

The scorecard deliberately has no total. Dimensions carry different units and
error structures, so an unvalidated weighted sum such as
``0.4*pLDDT + 0.3*dock + 0.3*distance`` produces a precise ranking with no
basis. Ranking happens through hard feasibility gates, ordinal evidence
levels, within-family comparison and Pareto non-domination instead.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..provenance import sequence_hash
from .chem import CofactorSpec, CofactorState, LigandSource, Stereochemistry
from .record import EvidenceRef, EvidenceStrength
from .templates import GeometryConstraint


class ConfidenceLevel(str, enum.Enum):
    """Ordinal evidence level. Ordinal because the inputs are not commensurable."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    INSUFFICIENT = "insufficient"
    CONTRADICTORY = "contradictory"

    @property
    def rank(self) -> int:
        return {"strong": 4, "moderate": 3, "weak": 2,
                "insufficient": 1, "contradictory": 0}[self.value]


class SequenceRecord(BaseModel):
    """A mined sequence with its retrieval provenance."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    sequence: str
    sequence_sha256: str | None = None
    accession: str | None = None
    source_database: str | None = None
    database_version: str | None = None
    organism: str | None = None
    description: str | None = None
    length: int | None = None
    # retrieval provenance
    seed_accession: str | None = None
    search_method: str | None = None          # blastp | mmseqs2 | hmmsearch
    percent_identity: float | None = None
    query_coverage: float | None = None
    evalue: float | None = None
    # quality
    is_fragment: bool | None = None
    has_nonstandard_residues: bool = False
    annotation_confidence: EvidenceStrength = EvidenceStrength.ANNOTATION_ONLY

    @model_validator(mode="after")
    def _derive(self) -> "SequenceRecord":
        if not self.sequence_sha256:
            object.__setattr__(self, "sequence_sha256", sequence_hash(self.sequence))
        if self.length is None:
            object.__setattr__(self, "length", len(self.sequence.strip()))
        bad = set(self.sequence.upper()) - set("ACDEFGHIKLMNPQRSTVWY")
        if bad:
            object.__setattr__(self, "has_nonstandard_residues", True)
        return self


class FamilyAnnotation(BaseModel):
    """A family call built from several independent signals, not one motif."""

    model_config = ConfigDict(extra="forbid")

    family_name: str | None = None
    subfamily: str | None = None
    family_template_id: str | None = None
    overall_identity_to_seed: float | None = None
    domains: list[dict[str, Any]] = Field(default_factory=list)   # {id, start, end, source}
    matched_motifs: list[dict[str, Any]] = Field(default_factory=list)
    cofactor_preference: str | None = None
    cofactor_preference_evidence: str | None = None
    sequence_cluster_id: str | None = None
    phylogenetic_clade: str | None = None
    signals_supporting: list[str] = Field(default_factory=list)
    signals_conflicting: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT

    @property
    def n_independent_signals(self) -> int:
        return len(set(self.signals_supporting))

    def recompute_confidence(self, min_signals: int = 3) -> ConfidenceLevel:
        """Family confidence from signal agreement, not from one hit."""
        n = self.n_independent_signals
        if self.signals_conflicting:
            level = ConfidenceLevel.CONTRADICTORY if n <= len(self.signals_conflicting) \
                else ConfidenceLevel.WEAK
        elif n >= min_signals:
            level = ConfidenceLevel.STRONG
        elif n == min_signals - 1:
            level = ConfidenceLevel.MODERATE
        elif n >= 1:
            level = ConfidenceLevel.WEAK
        else:
            level = ConfidenceLevel.INSUFFICIENT
        object.__setattr__(self, "confidence", level)
        return level


class CatalyticMapping(BaseModel):
    """Where the template's catalytic roles land on this sequence."""

    model_config = ConfigDict(extra="forbid")

    catalytic_template_id: str | None = None
    role_to_residue: dict[str, str] = Field(
        default_factory=dict,
        description="role label -> residue token like 'Y155' in author numbering",
    )
    role_to_index: dict[str, int] = Field(
        default_factory=dict, description="role label -> 0-based index in this sequence"
    )
    missing_roles: list[str] = Field(default_factory=list)
    substituted_roles: dict[str, str] = Field(default_factory=dict)
    alignment_quality: float | None = None

    @property
    def is_complete(self) -> bool:
        return not self.missing_roles

    @property
    def mechanism_compatible(self) -> bool:
        """Complete catalytic machinery present, allowing conservative swaps."""
        return self.is_complete or not any(
            r for r in self.missing_roles if r not in self.substituted_roles
        )


class StructureRecord(BaseModel):
    """A structure chosen for a candidate, with why it was chosen."""

    model_config = ConfigDict(extra="forbid")

    structure_id: str
    source: str                       # pdb_complex | pdb_apo | afdb | predicted | homology
    path: str | None = None
    format: str = "mmcif"
    priority_rank: int = 99
    sequence_identity_to_candidate: float | None = None
    covers_residues: tuple[int, int] | None = None
    missing_regions: list[tuple[int, int]] = Field(default_factory=list)
    is_mutant_relative_to_candidate: bool | None = None
    mutations_in_structure: list[str] = Field(default_factory=list)
    assembly: str | None = None
    bound_ligands: list[str] = Field(default_factory=list)
    cofactor_in_structure: str | None = None
    cofactor_state_in_structure: CofactorState = CofactorState.UNKNOWN
    ligand_source: LigandSource = LigandSource.UNKNOWN
    mean_plddt: float | None = None
    pocket_plddt: float | None = None
    numbering_offset: int = 0
    qc_notes: list[str] = Field(default_factory=list)

    @property
    def pocket_confidence(self) -> ConfidenceLevel:
        """Pocket-local confidence; the whole-chain mean hides a disordered loop."""
        v = self.pocket_plddt
        if v is None:
            return ConfidenceLevel.INSUFFICIENT
        if v >= 85:
            return ConfidenceLevel.STRONG
        if v >= 70:
            return ConfidenceLevel.MODERATE
        return ConfidenceLevel.WEAK


class ComplexPose(BaseModel):
    """One modelled enzyme-substrate-cofactor arrangement."""

    model_config = ConfigDict(extra="forbid")

    pose_id: str
    method: str                       # template_docking | af3 | boltz | manual
    rank: int | None = None
    path: str | None = None
    substrate_present: bool = True
    cofactor_present: bool = False
    cofactor_state: CofactorState = CofactorState.UNKNOWN
    substrate_source: LigandSource = LigandSource.UNKNOWN
    cofactor_source: LigandSource = LigandSource.UNKNOWN
    metals_present: list[str] = Field(default_factory=list)
    docking_score: float | None = None
    docking_score_function: str | None = None
    model_confidence: dict[str, float] = Field(
        default_factory=dict, description="e.g. ranking_score, iptm, pae_interface"
    )
    restrained_constraints: list[str] = Field(
        default_factory=list,
        description="Constraint names enforced while building this pose. These "
                    "may not be counted as independent evidence afterwards.",
    )
    clash_count: int | None = None
    is_valid: bool = True
    invalid_reason: str | None = None


class GeometryReport(BaseModel):
    """Measured geometry for one pose against one catalytic template."""

    model_config = ConfigDict(extra="forbid")

    pose_id: str
    measurements: dict[str, float | None] = Field(default_factory=dict)
    satisfied: dict[str, bool | None] = Field(default_factory=dict)
    gating_passed: bool | None = None
    circular_constraints: list[str] = Field(
        default_factory=list,
        description="Constraints that were restrained during modelling and so "
                    "were excluded from the independent evidence count.",
    )
    independent_satisfied: int = 0
    independent_total: int = 0
    clash_count: int | None = None
    notes: str = ""

    @property
    def independent_fraction(self) -> float | None:
        if self.independent_total == 0:
            return None
        return self.independent_satisfied / self.independent_total


#: Share of the *minority* face at or above which a split pose set is reported
#: as ``competing_poses`` instead of being resolved by majority vote.
#:
#: WHY THIS IS NAMED AND WHY THE DEFAULT IS 0.0
#: --------------------------------------------
#: This module's position is that a split pose set is **not** resolved by
#: majority vote: pose counts are an artefact of how the sampler ran, not a
#: Boltzmann population, so "seven re and one si" is not "87% towards the
#: target", it is an ensemble that disagrees with itself. A bare literal cut
#: buried in the comparison contradicted that position silently, because every
#: split under the cut was reported as a clean ``favors_target`` with nothing
#: in the record to say a minority pose had pointed the other way.
#:
#: The default of ``0.0`` therefore means: *any* genuine split -- at least one
#: pose on each face -- is reported as ``competing_poses``. That is the
#: conservative direction. The failure it prevents is a confident single-
#: enantiomer prediction handed to a chemist who then runs a chiral assay
#: against it; the cost of being conservative is only that a reviewer is asked
#: to look at an ensemble that genuinely disagrees.
#:
#: NEEDS CALIBRATION BEFORE IT IS RAISED. A non-zero value is a claim that a
#: minority below it is sampling noise rather than a second binding mode, and
#: that claim belongs to a specific pose generator, pose count and clustering
#: radius -- never to this file. Calibrate it against ensembles whose
#: experimental ee is known, pass it explicitly through
#: :meth:`StereoCall.from_counts` or
#: :func:`eagent.science.stereo.call_stereochemistry`, and let it reach
#: provenance so the reader knows which cut produced the call.
DEFAULT_COMPETING_FACE_FRACTION: float = 0.0


class StereoCall(BaseModel):
    """Direction of the predicted product configuration. Never a fake ee value."""

    model_config = ConfigDict(extra="forbid")

    call: str = Field(
        "insufficient_evidence",
        description="favors_target | favors_opposite | competing_poses | "
                    "insufficient_evidence | not_applicable",
    )
    target_face_poses: int = 0
    opposite_face_poses: int = 0
    undetermined_poses: int = 0
    basis: str = ""
    predicted_ee_pct: float | None = Field(
        None,
        description="Populated only by a model calibrated on this reaction class; "
                    "stays None otherwise.",
    )
    calibration_source: str | None = None

    @model_validator(mode="after")
    def _no_uncalibrated_ee(self) -> "StereoCall":
        if self.predicted_ee_pct is not None and not self.calibration_source:
            raise ValueError(
                "a numeric ee prediction requires a named calibration source"
            )
        return self

    @classmethod
    def from_counts(
        cls,
        target: int,
        opposite: int,
        undetermined: int,
        basis: str = "",
        *,
        competing_face_fraction: float = DEFAULT_COMPETING_FACE_FRACTION,
    ) -> "StereoCall":
        """Direction from per-pose counts, reporting a split rather than voting.

        ``competing_face_fraction`` is the minority share at or above which the
        call is ``competing_poses``; see
        :data:`DEFAULT_COMPETING_FACE_FRACTION` for why it defaults to 0.0 and
        what calibrating it would mean. It is a parameter rather than a literal
        so that the value which decided the call travels with the call, in
        ``basis``, instead of living unnamed in this method.

        Raises
        ------
        ValueError
            When the cut is outside ``[0, 0.5]``. Above 0.5 the minority share
            can never reach it, so ``competing_poses`` would become
            unreachable -- a silent disabling of the split report rather than a
            calibration of it.
        """
        if not (0.0 <= competing_face_fraction <= 0.5):
            raise ValueError(
                f"competing_face_fraction must lie in [0, 0.5], got "
                f"{competing_face_fraction}; the minority share of a split can "
                f"never exceed 0.5, so a larger cut would silently make "
                f"'competing_poses' unreachable"
            )
        note = ""
        if target == 0 and opposite == 0:
            call = "insufficient_evidence"
        elif target > 0 and opposite > 0:
            minority = min(target, opposite) / (target + opposite)
            competing = minority >= competing_face_fraction
            call = "competing_poses" if competing else (
                "favors_target" if target > opposite else "favors_opposite")
            note = (
                f"split pose set: minority face share {minority:.3f} against a "
                f"reporting cut of {competing_face_fraction:.3f}"
                + ("" if competing else
                   "; the minority was below the calibrated cut, so the majority "
                   "face was reported -- the minority poses are still counted "
                   "above and were not discarded")
            )
        elif target > 0:
            call = "favors_target"
        else:
            call = "favors_opposite"
        full_basis = f"{basis}; {note}" if (basis and note) else (basis or note)
        return cls(call=call, target_face_poses=target, opposite_face_poses=opposite,
                   undetermined_poses=undetermined, basis=full_basis)


class ScoreDimension(BaseModel):
    """One axis of the scorecard, kept separate and self-describing."""

    model_config = ConfigDict(extra="forbid")

    name: str
    level: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT
    value: float | None = None
    unit: str | None = None
    direction: str = Field("higher_is_better",
                           description="higher_is_better | lower_is_better | categorical")
    basis: str = ""
    is_gate: bool = False
    gate_passed: bool | None = None


#: The nine scorecard axes.
SCORE_DIMENSIONS: tuple[str, ...] = (
    "functional_literature_evidence",
    "family_mechanism_compatibility",
    "substrate_specificity_model",
    "catalytic_geometry",
    "local_structure_confidence",
    "docking_result",
    "expression_developability_risk",
    "model_uncertainty",
    "sequence_pocket_novelty",
)


class Candidate(BaseModel):
    """Everything known about one candidate enzyme for this task."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    sequence_record: SequenceRecord
    family: FamilyAnnotation = Field(default_factory=FamilyAnnotation)
    catalytic_mapping: CatalyticMapping = Field(default_factory=CatalyticMapping)
    structures: list[StructureRecord] = Field(default_factory=list)
    poses: list[ComplexPose] = Field(default_factory=list)
    geometry: list[GeometryReport] = Field(default_factory=list)
    stereo: StereoCall = Field(default_factory=StereoCall)
    robustness_G: float | None = Field(
        None,
        description="Circularity-corrected robustness: of the decided poses that "
                    "carried at least one gating constraint which was NOT "
                    "restrained during modelling, the fraction whose independent "
                    "gating constraints were all satisfied. None -- never 1.0 and "
                    "never 0.0 -- when no pose carried independent evidence at "
                    "all. A sampling statistic, not a probability of catalysis.",
    )
    scorecard: dict[str, ScoreDimension] = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    input_errors: list[str] = Field(
        default_factory=list,
        description="Wrong sequence, wrong ligand, wrong chirality: must be fixed, "
                    "never traded off against other dimensions.",
    )
    disqualified: bool = False
    disqualification_reason: str | None = None

    @property
    def sequence(self) -> str:
        return self.sequence_record.sequence

    @property
    def best_structure(self) -> StructureRecord | None:
        if not self.structures:
            return None
        return sorted(self.structures, key=lambda s: s.priority_rank)[0]

    @property
    def valid_poses(self) -> list[ComplexPose]:
        return [p for p in self.poses if p.is_valid]

    def gates(self) -> list[ScoreDimension]:
        return [d for d in self.scorecard.values() if d.is_gate]

    @property
    def passes_gates(self) -> bool:
        if self.input_errors or self.disqualified:
            return False
        gs = self.gates()
        return bool(gs) and all(g.gate_passed is True for g in gs)

    @property
    def has_unresolved_gate(self) -> bool:
        """A gate that could not be evaluated is not a failed gate."""
        return any(g.gate_passed is None for g in self.gates())

    def dimension(self, name: str) -> ScoreDimension | None:
        return self.scorecard.get(name)

    def set_dimension(self, dim: ScoreDimension) -> None:
        self.scorecard[dim.name] = dim
