"""Five machine-readable template types.

These are data, not prompts. A catalytic template in particular must trace to
an experimental structure, a mechanism publication, or an explicitly labelled
theoretical model. A set of coordinates a language model produced from memory
is not a mechanism, and the loader rejects it.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import TemplateError
from .chem import CofactorState, Stereochemistry


class TemplateSourceType(str, enum.Enum):
    EXPERIMENTAL_STRUCTURE = "experimental_structure"   # PDB entry with the relevant ligands
    MECHANISM_LITERATURE = "mechanism_literature"       # M-CSA entry or primary paper
    CURATED_DATABASE = "curated_database"
    THEORETICAL_MODEL = "theoretical_model"             # theozyme; must be labelled
    UNSOURCED = "unsourced"                             # rejected by the loader

    @property
    def admissible(self) -> bool:
        return self is not TemplateSourceType.UNSOURCED


class TemplateProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: TemplateSourceType = TemplateSourceType.UNSOURCED
    identifiers: list[str] = Field(default_factory=list)   # PDB ids, PMIDs, M-CSA ids
    curated_by: str | None = None
    curated_at: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _must_be_sourced(self) -> "TemplateProvenance":
        if not self.source_type.admissible:
            raise TemplateError("template provenance is unsourced")
        if not self.identifiers:
            raise TemplateError(
                f"template of type {self.source_type.value} carries no identifiers"
            )
        return self

    @property
    def is_theoretical(self) -> bool:
        return self.source_type is TemplateSourceType.THEORETICAL_MODEL


class GeometryConstraint(BaseModel):
    """One checkable geometric relationship between named atoms.

    ``calibrated_on`` records the systems whose known activity set the window.
    An uncalibrated constraint is still usable, but the evaluator downgrades
    confidence instead of rejecting candidates against it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str = Field("distance", description="distance | angle | dihedral")
    atom_a: str                       # role token, e.g. "cofactor.hydride_donor_C4"
    atom_b: str                       # e.g. "substrate.electrophile"
    atom_c: str | None = None
    atom_d: str | None = None
    target: float | None = None
    tolerance: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    unit: str = "angstrom"
    severity: str = Field("scoring", description="gating | scoring | advisory")
    calibrated_on: list[str] = Field(default_factory=list)
    source: str = ""

    @model_validator(mode="after")
    def _window_defined(self) -> "GeometryConstraint":
        if self.target is None and self.min_value is None and self.max_value is None:
            raise TemplateError(f"constraint '{self.name}' defines no window")
        if self.kind == "angle" and self.unit == "angstrom":
            object.__setattr__(self, "unit", "degree")
        if self.kind in ("angle", "dihedral") and self.atom_c is None:
            raise TemplateError(f"constraint '{self.name}' of kind {self.kind} needs atom_c")
        return self

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibrated_on)

    def window(self) -> tuple[float, float]:
        if self.min_value is not None or self.max_value is not None:
            lo = self.min_value if self.min_value is not None else float("-inf")
            hi = self.max_value if self.max_value is not None else float("inf")
            return lo, hi
        tol = self.tolerance if self.tolerance is not None else 0.0
        return self.target - tol, self.target + tol

    def satisfied_by(self, value: float | None) -> bool | None:
        """True / False, or None when the value could not be measured."""
        if value is None:
            return None
        lo, hi = self.window()
        return lo <= value <= hi


class ReactionTemplate(BaseModel):
    """What counts as the target transformation."""

    model_config = ConfigDict(extra="forbid")

    template_id: str
    reaction_class: str
    description: str = ""
    required_substrate_motif: str | None = Field(
        None, description="SMARTS the substrate must match"
    )
    forbidden_substrate_motif: list[str] = Field(default_factory=list)
    product_motif: str | None = None
    chemoselectivity_notes: str = ""
    stereo_requirement: Stereochemistry = Stereochemistry.UNSPECIFIED
    creates_stereocenter: bool | None = None
    provenance: TemplateProvenance


class FamilyTemplate(BaseModel):
    """How to recognise a family and what it implies about mechanism."""

    model_config = ConfigDict(extra="forbid")

    template_id: str
    family_name: str                                  # SDR, MDR/ADH, AKR, ...
    pfam_ids: list[str] = Field(default_factory=list)
    interpro_ids: list[str] = Field(default_factory=list)
    domain_architecture: list[str] = Field(default_factory=list)
    conserved_motifs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Each: {name, pattern, role, evidence}. A motif supports "
                    "family assignment; it is not an activity prediction.",
    )
    cofactor_preference: dict[str, str] = Field(
        default_factory=dict, description="cofactor -> evidence string"
    )
    typical_fold: str | None = None
    oligomeric_state: str | None = None
    catalytic_template_ids: list[str] = Field(default_factory=list)
    seed_accessions: list[str] = Field(default_factory=list)
    provenance: TemplateProvenance
    caveats: str = ""

    @property
    def min_independent_signals(self) -> int:
        """Family calls need agreement across several signal types."""
        return 3


class CatalyticTemplate(BaseModel):
    """Catalytic residues, functional atoms, cofactor state and geometry."""

    model_config = ConfigDict(extra="forbid")

    template_id: str
    family_name: str
    mechanism_summary: str = ""
    catalytic_residues: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Each: {label, residue_types, role, functional_atoms, evidence}",
    )
    required_cofactor: str | None = None
    required_cofactor_state: CofactorState = CofactorState.UNKNOWN
    cofactor_ligand_codes: list[str] = Field(default_factory=list)
    metals: list[str] = Field(default_factory=list)
    assembly_state: str | None = None
    geometry_constraints: list[GeometryConstraint] = Field(default_factory=list)
    reference_structures: list[str] = Field(default_factory=list)
    provenance: TemplateProvenance

    @model_validator(mode="after")
    def _reduced_state_is_explicit(self) -> "CatalyticTemplate":
        if self.required_cofactor and self.required_cofactor_state is CofactorState.UNKNOWN:
            raise TemplateError(
                f"{self.template_id}: cofactor {self.required_cofactor} declared "
                f"without an oxidation state; NAD(P)+ and NAD(P)H are not "
                f"interchangeable in a hydride-transfer model"
            )
        return self

    def gating_constraints(self) -> list[GeometryConstraint]:
        return [c for c in self.geometry_constraints if c.severity == "gating"]

    def uncalibrated(self) -> list[GeometryConstraint]:
        return [c for c in self.geometry_constraints if not c.is_calibrated]


class EngineeringTemplate(BaseModel):
    """What may be changed, what is frozen, and what has already failed."""

    model_config = ConfigDict(extra="forbid")

    template_id: str
    family_name: str
    frozen_roles: list[str] = Field(
        default_factory=list,
        description="Catalytic and cofactor-anchoring roles frozen in round 1",
    )
    mutable_zones: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Each: {name, selector, rationale}. e.g. substrate shell 4-8 A",
    )
    default_shell_min_angstrom: float = 4.0
    default_shell_max_angstrom: float = 8.0
    known_beneficial_mutations: list[dict[str, Any]] = Field(default_factory=list)
    known_failure_modes: list[str] = Field(default_factory=list)
    max_simultaneous_mutations_round1: int = 2
    provenance: TemplateProvenance


class AssayTemplate(BaseModel):
    """How the result will be measured and what counts as a hit."""

    model_config = ConfigDict(extra="forbid")

    template_id: str
    tier: int = Field(1, ge=1, le=3)
    method: str
    confirms_product_identity: bool = False
    chiral_capable: bool = False
    requires_authentic_standard: bool = False
    controls_required: list[str] = Field(default_factory=list)
    positive_criteria: dict[str, Any] = Field(
        default_factory=dict,
        description="Pre-registered. Changing this after seeing data invalidates "
                    "the primary endpoint.",
    )
    limit_of_detection: float | None = None
    limit_unit: str | None = None
    replicates: int = 3
    readout_fields: list[str] = Field(default_factory=list)
    provenance: TemplateProvenance

    @model_validator(mode="after")
    def _tier2_must_identify_product(self) -> "AssayTemplate":
        if self.tier >= 2 and not self.confirms_product_identity:
            raise TemplateError(
                f"{self.template_id}: a tier-{self.tier} assay must identify the "
                f"product; an indirect cofactor signal cannot confirm a hit"
            )
        return self
