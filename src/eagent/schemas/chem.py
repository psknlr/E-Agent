"""Chemical identity: substrates, products, cofactors, reactive atoms.

Small molecules enter the system as explicit structures (isomeric SMILES or a
molfile/SDF block), never as prose names and never as "substrate sequence".
Peptide, protein and nucleic-acid substrates use the sequence interface in
``candidate.py`` instead.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Stereochemistry(str, enum.Enum):
    """Configuration at the centre created or consumed by the reaction."""

    R = "R"
    S = "S"
    RACEMIC = "racemic"
    ACHIRAL = "achiral"            # e.g. aldehyde or symmetric ketone reduction
    UNSPECIFIED = "unspecified"    # not yet decided by the operator


class LigandSource(str, enum.Enum):
    """How a ligand came to sit where it sits.

    These four origins carry completely different authority, and collapsing them
    into "binding site confirmed" is how a computational placement becomes a
    reported fact. A cofactor transplanted from a homologue is a hypothesis worth
    checking, not a measurement of this enzyme.
    """

    EXPERIMENTAL_OBSERVED = "experimental_observed"
    HOMOLOGY_TRANSPLANTED = "homology_transplanted"
    DOCKING_PREDICTED = "docking_predicted"
    JOINT_STRUCTURE_PREDICTION = "joint_structure_prediction"
    MANUAL_PLACEMENT = "manual_placement"
    UNKNOWN = "unknown"

    @property
    def is_experimental(self) -> bool:
        return self is LigandSource.EXPERIMENTAL_OBSERVED

    def claim(self) -> str:
        return {
            LigandSource.EXPERIMENTAL_OBSERVED:
                "observed in an experimental structure",
            LigandSource.HOMOLOGY_TRANSPLANTED:
                "inferred by transfer from a homologous structure; not measured here",
            LigandSource.DOCKING_PREDICTED:
                "a docking pose; a hypothesis about placement",
            LigandSource.JOINT_STRUCTURE_PREDICTION:
                "a co-folded prediction; a hypothesis about placement",
            LigandSource.MANUAL_PLACEMENT: "placed by hand",
            LigandSource.UNKNOWN: "origin not recorded",
        }[self]


class CofactorState(str, enum.Enum):
    """Oxidation state matters: NAD(P)+ in a template is not NAD(P)H."""

    REDUCED = "reduced"            # NADH / NADPH / FADH2 -- hydride donor
    OXIDIZED = "oxidized"          # NAD+ / NADP+ / FAD
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class AtomRef(BaseModel):
    """A specific atom, addressed by atom-map id rather than by element name."""

    model_config = ConfigDict(frozen=True)

    atom_map_id: int = Field(..., description="Atom map number in the mapped SMILES")
    element: str | None = None
    role: str | None = Field(
        None, description="e.g. carbonyl_C, carbonyl_O, hydride_donor_C4, nucleophile"
    )

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.element or '?'}:{self.atom_map_id}({self.role or '-'})"


class ReactiveAtoms(BaseModel):
    """Which atoms the reaction actually touches.

    Distance checks are defined against these atoms. A distance from the
    substrate centroid to the protein centroid is not a catalytic criterion.
    """

    electrophile: AtomRef | None = None     # e.g. ketone carbonyl carbon
    nucleophile: AtomRef | None = None
    leaving_group: AtomRef | None = None
    stabilised_atoms: list[AtomRef] = Field(default_factory=list)  # e.g. carbonyl O
    prochiral_center: AtomRef | None = None

    def all_refs(self) -> list[AtomRef]:
        refs = [self.electrophile, self.nucleophile, self.leaving_group,
                self.prochiral_center]
        return [r for r in refs if r is not None] + list(self.stabilised_atoms)


class SubstrateSpec(BaseModel):
    """A small-molecule substrate."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    isomeric_smiles: str | None = None
    molfile: str | None = None
    inchikey: str | None = None
    reactive_atoms: ReactiveAtoms = Field(default_factory=ReactiveAtoms)
    is_prochiral: bool | None = None
    chemical_purity: float | None = Field(None, ge=0.0, le=1.0)
    supplier_or_synthesis: str | None = None
    notes: str = ""

    @field_validator("isomeric_smiles")
    @classmethod
    def _no_placeholder(cls, v: str | None) -> str | None:
        if v is not None and v.strip() in {"", "null", "None", "TBD"}:
            raise ValueError("use None for an unresolved SMILES, not a placeholder string")
        return v

    @property
    def is_structurally_defined(self) -> bool:
        return bool(self.isomeric_smiles or self.molfile)


class ProductSpec(BaseModel):
    """The product the assay must actually detect."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    isomeric_smiles: str | None = None
    inchikey: str | None = None
    target_stereochemistry: Stereochemistry = Stereochemistry.UNSPECIFIED
    creates_new_stereocenter: bool | None = Field(
        None,
        description="False for aldehyde or symmetric-ketone reduction; must be "
                    "decided per substrate rather than assumed from reaction class.",
    )
    authentic_standard_available: bool | None = None
    notes: str = ""

    @property
    def is_structurally_defined(self) -> bool:
        return bool(self.isomeric_smiles)

    @property
    def requires_chiral_analysis(self) -> bool:
        return (
            self.creates_new_stereocenter is True
            and self.target_stereochemistry in (Stereochemistry.R, Stereochemistry.S)
        )


class CofactorSpec(BaseModel):
    """A cofactor with its chemical state and the atom that does the chemistry."""

    model_config = ConfigDict(extra="forbid")

    name: str                                  # NADH, NADPH, FAD, Zn2+, ...
    state: CofactorState = CofactorState.UNKNOWN
    smiles: str | None = None
    ligand_code: str | None = None             # PDB chemical component id, e.g. NAI/NAP
    transfer_atom: AtomRef | None = Field(
        None, description="Atom that transfers (e.g. nicotinamide C4 of NAD(P)H)"
    )
    stoichiometry: float | None = None
    recycling_system: str | None = None
    source: LigandSource = LigandSource.UNKNOWN
    evidence: str = ""

    @property
    def is_hydride_donor(self) -> bool:
        return self.state is CofactorState.REDUCED and self.transfer_atom is not None

    def describe(self) -> str:
        return f"{self.name}[{self.state.value}]"


# PDB chemical component ids that are routinely confused. Kept here so the
# verifier can flag a template that claims NADPH but carries an oxidised ligand.
OXIDISED_NICOTINAMIDE_CODES: dict[str, str] = {"NAD": "NAD+", "NAP": "NADP+"}
REDUCED_NICOTINAMIDE_CODES: dict[str, str] = {"NAI": "NADH", "NDP": "NADPH"}


def cofactor_state_from_ligand_code(code: str | None) -> CofactorState:
    """Infer oxidation state from a PDB ligand code, or report UNKNOWN.

    Returns UNKNOWN rather than guessing for any code not in the two tables
    above; the caller must then resolve it from the source structure.
    """
    if not code:
        return CofactorState.UNKNOWN
    c = code.strip().upper()
    if c in OXIDISED_NICOTINAMIDE_CODES:
        return CofactorState.OXIDIZED
    if c in REDUCED_NICOTINAMIDE_CODES:
        return CofactorState.REDUCED
    return CofactorState.UNKNOWN
