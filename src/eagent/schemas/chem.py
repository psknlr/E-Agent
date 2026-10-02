"""Chemical identity: substrates, products, cofactors, reactive atoms.

A substrate enters the system by one of two paths, and which one is a declared
fact (:class:`SubstrateKind`) rather than something a reader infers from which
fields happen to be filled:

* **Small molecules** are explicit structures -- isomeric SMILES or a
  molfile/SDF block -- never prose names and never a "substrate sequence". The
  reactive site is an atom, addressed by atom-map id
  (:class:`SubstrateSpec`, :class:`ReactiveAtoms`).
* **Biopolymers** -- peptide, protein, nucleic acid -- are a sequence, and the
  reactive site is a *residue*, addressed by position in that sequence
  (:class:`BiopolymerSubstrateSpec`, :class:`ReactiveResidues`).

The second path exists because the first one cannot be stretched to cover it.
A 30-mer peptide substrate has no isomeric SMILES anybody will write, no
InChIKey, and no atom-map id for the residue a kinase phosphorylates. A system
offering only the small-molecule path leaves two options, and both are
failures: block such a task forever on a structure it can never have, or let a
prose name stand in for the substrate, which is the thing this module exists to
prevent. Declaring the kind lets the gates ask each task for what that kind of
substrate can actually be pinned down by -- see
:data:`eagent.schemas.reaction.BIOPOLYMER_GATE_REQUIREMENTS`.

Note that :class:`~eagent.schemas.candidate.SequenceRecord` is **not** this
path: it is the mined *enzyme*. The substrate is the thing the enzyme acts on,
and conflating the two makes the record of an assay unreadable.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..provenance import sequence_hash


class SubstrateKind(str, enum.Enum):
    """Which path a substrate enters by, declared rather than inferred.

    Inferring the kind from which fields happen to be filled is how a task
    with an unresolved SMILES becomes indistinguishable from a task that can
    never have one. The kind decides which gate requirements apply.
    """

    SMALL_MOLECULE = "small_molecule"
    PEPTIDE = "peptide"
    PROTEIN = "protein"
    NUCLEIC_ACID = "nucleic_acid"

    @property
    def is_biopolymer(self) -> bool:
        return self is not SubstrateKind.SMALL_MOLECULE

    @property
    def alphabet(self) -> str:
        """Residue letters this kind is written in."""
        if self is SubstrateKind.NUCLEIC_ACID:
            return "ACGTU"
        if self.is_biopolymer:
            return "ACDEFGHIKLMNPQRSTVWY"
        return ""


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


class ResidueRef(BaseModel):
    """A residue in a biopolymer substrate, addressed by position.

    The small-molecule path addresses a reactive site by atom-map id, which a
    peptide substrate has none of. A residue plus the letter expected there is
    the equivalent, and carrying the letter means a position that has drifted
    against the sequence is caught instead of silently pointing at a
    neighbour.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    position: int = Field(..., ge=1, description="1-based position in the substrate")
    residue: str | None = Field(None, description="Expected one-letter residue")
    role: str | None = Field(
        None, description="e.g. phosphoacceptor, scissile_P1, hydroxylated"
    )

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.residue or '?'}{self.position}({self.role or '-'})"


class ReactiveResidues(BaseModel):
    """Which residues the reaction touches, for a biopolymer substrate.

    The analogue of :class:`ReactiveAtoms`. Kept as a separate type rather
    than overloading the atom one, because a geometry check written against
    atom-map ids silently measures nothing when handed residue positions.
    """

    model_config = ConfigDict(extra="forbid")

    modified: list[ResidueRef] = Field(default_factory=list)
    scissile_bond: tuple[ResidueRef, ResidueRef] | None = None
    recognition_motif: str | None = None

    def all_refs(self) -> list[ResidueRef]:
        refs = list(self.modified)
        if self.scissile_bond:
            refs.extend(self.scissile_bond)
        return refs


class BiopolymerSubstrateSpec(BaseModel):
    """A peptide, protein or nucleic-acid substrate.

    This is not :class:`~eagent.schemas.candidate.SequenceRecord`, which is the
    mined *enzyme*. Conflating the acting enzyme with the thing acted on makes
    the record of an assay unreadable, so they are separate types.
    """

    model_config = ConfigDict(extra="forbid")

    kind: SubstrateKind = SubstrateKind.PEPTIDE
    name: str | None = None
    sequence: str | None = None
    sequence_sha256: str | None = None
    reactive_residues: ReactiveResidues = Field(default_factory=ReactiveResidues)
    modifications: list[str] = Field(
        default_factory=list,
        description="Non-standard residues, caps, labels. A synthetic peptide "
                    "is rarely just its one-letter sequence.",
    )
    length: int | None = None
    source: str | None = None
    purity: float | None = Field(None, ge=0.0, le=1.0)
    notes: str = ""

    @model_validator(mode="after")
    def _derive_and_check(self) -> "BiopolymerSubstrateSpec":
        if self.kind is SubstrateKind.SMALL_MOLECULE:
            raise ValueError(
                "a small molecule belongs in SubstrateSpec, not in the "
                "biopolymer path")
        if self.sequence:
            norm = "".join(self.sequence.split()).upper()
            object.__setattr__(self, "sequence", norm)
            if self.sequence_sha256 is None:
                object.__setattr__(self, "sequence_sha256", sequence_hash(norm))
            if self.length is None:
                object.__setattr__(self, "length", len(norm))
            bad = set(norm) - set(self.kind.alphabet)
            if bad and not self.modifications:
                raise ValueError(
                    f"sequence carries {sorted(bad)}, which are outside the "
                    f"{self.kind.value} alphabet, and no modifications are "
                    f"recorded to account for them")
            for ref in self.reactive_residues.all_refs():
                if ref.position > len(norm):
                    raise ValueError(
                        f"reactive residue {ref} lies beyond the "
                        f"{len(norm)}-residue substrate")
                if ref.residue and norm[ref.position - 1] != ref.residue.upper():
                    raise ValueError(
                        f"reactive residue {ref} does not match the sequence, "
                        f"which has {norm[ref.position - 1]} at that position; "
                        f"the numbering is wrong")
        return self

    @property
    def is_structurally_defined(self) -> bool:
        """A biopolymer is pinned down by its sequence, not by a SMILES."""
        return bool(self.sequence)


class SubstrateSpec(BaseModel):
    """A small-molecule substrate."""

    model_config = ConfigDict(extra="forbid")

    kind: SubstrateKind = SubstrateKind.SMALL_MOLECULE
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

    @model_validator(mode="after")
    def _kind_matches_path(self) -> "SubstrateSpec":
        if self.kind.is_biopolymer:
            raise ValueError(
                f"kind '{self.kind.value}' belongs in BiopolymerSubstrateSpec; "
                f"a biopolymer has no isomeric SMILES and its reactive site is "
                f"a residue, not an atom-map id")
        return self

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
