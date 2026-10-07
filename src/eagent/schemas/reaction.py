"""Task and reaction specification, with the unresolved-field discipline.

A ``None`` field means "not yet determined". It is never filled by a language
model. It is filled only by :meth:`TaskSpec.resolve`, which demands a source
and writes an :class:`Assumption` into the ledger. Gates then refuse to open
while fields they depend on are still unresolved.
"""

from __future__ import annotations

import enum
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from ..errors import FabricationGuardError, UnresolvedFieldError
from .chem import (
    BiopolymerSubstrateSpec, CofactorSpec, ProductSpec, Stereochemistry,
    SubstrateKind, SubstrateSpec,
)


class ReactionClass(str, enum.Enum):
    """Extensible registry of supported transformations."""

    KETONE_TO_SECONDARY_ALCOHOL = "ketone_to_secondary_alcohol"
    ALDEHYDE_TO_PRIMARY_ALCOHOL = "aldehyde_to_primary_alcohol"
    IMINE_REDUCTION = "imine_reduction"
    ALCOHOL_OXIDATION = "alcohol_oxidation"
    HALOGENATION = "halogenation"
    TRANSAMINATION = "transamination"
    ESTER_HYDROLYSIS = "ester_hydrolysis"
    HYDROXYLATION = "hydroxylation"
    OTHER = "other"


class TaskMode(str, enum.Enum):
    """The three supported input modes."""

    ENZYME_MINING = "enzyme_mining"              # A: substrate + reaction known
    REACTION_SPACE_EXPLORATION = "reaction_space_exploration"  # B: reaction known only
    SUBSTRATE_DIRECTED_ENGINEERING = "substrate_directed_engineering"  # C: enzyme + substrate


class Assumption(BaseModel):
    """A field that was filled in, and by what authority."""

    model_config = ConfigDict(frozen=True)

    field_path: str
    value: Any
    source: str = Field(
        ...,
        description="operator:<name> | literature:<PMID/DOI> | template:<id> | "
                    "database:<name>@<version>. Never 'model' alone.",
    )
    justification: str = ""
    at: str | None = None

    @model_validator(mode="after")
    def _reject_unsourced(self) -> "Assumption":
        src = (self.source or "").strip().lower()
        if not src:
            raise FabricationGuardError(f"{self.field_path}: assumption needs a source")
        allowed = ("operator:", "literature:", "template:", "database:", "experiment:")
        if not src.startswith(allowed):
            raise FabricationGuardError(
                f"{self.field_path}: source '{self.source}' is not an accepted "
                f"authority; a model's own guess is not a source"
            )
        return self


class Conditions(BaseModel):
    """Reaction and expression context. Part of the record key, not metadata."""

    model_config = ConfigDict(extra="forbid")

    cofactor_options: list[CofactorSpec] = Field(default_factory=list)
    pH: float | None = None
    temperature_C: float | None = None
    solvent_system: str | None = None
    cosolvent_fraction: float | None = None
    buffer: str | None = None
    expression_host: str | None = None
    substrate_concentration_mM: float | None = None
    enzyme_loading: str | None = None
    reaction_time_h: float | None = None

    def key(self) -> tuple:
        """Hashable identity: two records differ if any of these differ."""
        return (
            tuple(sorted(c.describe() for c in self.cofactor_options)),
            self.pH, self.temperature_C, self.solvent_system,
            self.cosolvent_fraction, self.buffer, self.expression_host,
            self.substrate_concentration_mM, self.enzyme_loading,
            self.reaction_time_h,
        )


class Budget(BaseModel):
    """Resource plan. These are planning targets, not predetermined pass rates."""

    model_config = ConfigDict(extra="forbid")

    initial_sequence_target: int = 2000
    family_qc_pool_target: int = 600
    structure_pool_target: int = 600
    detailed_complex_target: int = 300
    new_constructs_round_1: int = 96
    constructs_include_controls: bool = Field(
        False,
        description="If the construct budget is a hard synthesis cap, controls "
                    "that need new genes must be reserved from it up front.",
    )
    reserved_control_slots: int = 0

    @property
    def candidate_slots(self) -> int:
        if self.constructs_include_controls:
            return max(0, self.new_constructs_round_1 - self.reserved_control_slots)
        return self.new_constructs_round_1

    @model_validator(mode="after")
    def _pool_must_exceed_batch(self) -> "Budget":
        if self.detailed_complex_target < self.candidate_slots:
            raise ValueError(
                "detailed_complex_target must exceed the construct slots, "
                "otherwise selection has nothing to select from"
            )
        return self


class Objectives(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary: str = "experimentally_confirmed_target_product"
    secondary: list[str] = Field(
        default_factory=lambda: [
            "target_enantiomer_selectivity", "activity", "soluble_expression",
        ]
    )


class Approval(BaseModel):
    """The three human decision points."""

    model_config = ConfigDict(extra="forbid")

    reaction_spec_confirmed: bool = False
    synthesis_authorized: bool = False
    functional_criteria_confirmed: bool = False

    def state(self, gate: str) -> bool:
        """Whether this gate's flag is strictly ``True``.

        ``bool(value)`` would be true for any non-empty string, so a field
        holding "false", "no" or "pending" would read as approved. A gate
        opens on a real boolean or it does not open.
        """
        return getattr(self, gate, False) is True


class ReactionSpec(BaseModel):
    """What chemistry must happen, stated precisely enough to be falsifiable."""

    #: ``validate_assignment`` because this model carries an invariant between
    #: two of its fields -- exactly one substrate path -- and the spec is
    #: filled in by assignment as an operator resolves it. Without it the
    #: invariant holds only at construction, which is the one moment a task
    #: spec is empty.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    reaction_class: ReactionClass = ReactionClass.OTHER
    atom_mapped_reaction_smiles: str | None = None
    substrate: SubstrateSpec = Field(default_factory=SubstrateSpec)
    biopolymer_substrate: BiopolymerSubstrateSpec | None = Field(
        None,
        description="Set instead of `substrate` when the substrate is a "
                    "peptide, protein or nucleic acid. Exactly one path is "
                    "used; the gates follow whichever it is.",
    )
    product: ProductSpec = Field(default_factory=ProductSpec)
    rhea_id: str | None = None
    ec_hint: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _exactly_one_substrate_path(self) -> "ReactionSpec":
        """Refuse a spec that fills both substrate paths.

        The field says exactly one path is used, and every reader picks one:
        the gate table follows :attr:`substrate_kind`, which prefers the
        biopolymer, while the structure checks used to read ``substrate``
        regardless. A task carrying both therefore satisfied each reader with a
        different molecule -- and adding an unrelated small molecule to a
        peptide task *removed* a blocker, because the two readers disagreed
        about what the substrate was.

        Refused here, where both fields are visible at once, rather than in
        each reader. A name alone counts: it is what somebody believes the
        substrate is, and two of those is still two substrates.
        """
        if self.biopolymer_substrate is None:
            return self
        small = self.substrate
        declared = [f for f, v in (("name", small.name),
                                   ("isomeric_smiles", small.isomeric_smiles),
                                   ("molfile", small.molfile),
                                   ("inchikey", small.inchikey)) if v]
        if declared:
            raise ValueError(
                f"reaction.substrate declares {', '.join(declared)} while "
                f"reaction.biopolymer_substrate declares a "
                f"{self.biopolymer_substrate.kind.value}. Exactly one "
                f"substrate path is used, and every reader picks one: the gate "
                f"table would follow the biopolymer and the structure checks "
                f"the small molecule, so the task would pass by describing two "
                f"different substrates. Clear whichever is not the substrate."
            )
        return self

    @property
    def substrate_kind(self) -> SubstrateKind:
        """Which substrate path this reaction uses."""
        if self.biopolymer_substrate is not None:
            return self.biopolymer_substrate.kind
        return self.substrate.kind

    @property
    def target_stereochemistry(self) -> Stereochemistry:
        return self.product.target_stereochemistry

    def unresolved(self, required: Iterable[str]) -> list[str]:
        out = []
        for path in required:
            if _get_path(self, path) is None:
                out.append(path)
        return out


#: Fields that must be resolved before each gate opens, for a small-molecule
#: substrate. A biopolymer task uses :data:`BIOPOLYMER_GATE_REQUIREMENTS`
#: instead, because blocking it on a structure it can never have would leave
#: the gate permanently shut for a perfectly well-specified task.
GATE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "reaction_spec_confirmed": (
        "reaction.substrate.isomeric_smiles",
        "reaction.product.isomeric_smiles",
        "reaction.product.creates_new_stereocenter",
        "reaction.atom_mapped_reaction_smiles",
    ),
    "synthesis_authorized": (
        "reaction.substrate.isomeric_smiles",
        "reaction.product.isomeric_smiles",
        "conditions.pH",
        "conditions.temperature_C",
        "conditions.expression_host",
    ),
    "functional_criteria_confirmed": (
        "reaction.product.isomeric_smiles",
    ),
}


#: The same gates, asked of a peptide, protein or nucleic-acid substrate.
#: A biopolymer is pinned down by its sequence and the residue the reaction
#: touches, so those are what the gate demands; an atom-mapped reaction SMILES
#: and an isomeric SMILES are not available and are not asked for.
BIOPOLYMER_GATE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "reaction_spec_confirmed": (
        "reaction.biopolymer_substrate.sequence",
        "reaction.biopolymer_substrate.reactive_residues.modified",
        "reaction.product.name",
    ),
    "synthesis_authorized": (
        "reaction.biopolymer_substrate.sequence",
        "conditions.pH",
        "conditions.temperature_C",
        "conditions.expression_host",
    ),
    "functional_criteria_confirmed": (
        "reaction.product.name",
    ),
}


def _get_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        cur = getattr(cur, part, None)
    return cur


def _set_path(obj: Any, path: str, value: Any) -> Any:
    """Assign a nested field, validating it against the field's own type.

    A bare ``setattr`` on a nested model skips validation entirely, because
    pydantic validates on construction and on assignment to the model it was
    configured for, not to a child reached through attribute access. The
    consequences are not cosmetic. A field declared ``bool`` accepted the
    string ``"false"``, and every later reader asked ``bool(value)``, which
    is ``True`` for any non-empty string -- so writing "false" into an
    approval flag read as approved. A field declared as an enum kept a plain
    string, and the next reader to ask for ``.value`` raised.

    Validating through the declared type coerces what is coercible, rejects
    what is not, and returns the stored value so the caller can record what
    was actually written rather than what was offered.
    """
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = getattr(cur, part)
        if cur is None:
            raise UnresolvedFieldError([".".join(parts[:-1])])
    name = parts[-1]
    coerced = value
    if isinstance(cur, BaseModel):
        info = type(cur).model_fields.get(name)
        if info is None:
            raise UnresolvedFieldError(
                [path], f"{type(cur).__name__} has no field '{name}'")
        if info.annotation is not None:
            # Raises pydantic.ValidationError on anything the field's type
            # cannot accept, which is the point: a refusal here is cheaper
            # than a wrong value read by three modules downstream.
            coerced = TypeAdapter(info.annotation).validate_python(value)
    setattr(cur, name, coerced)
    return coerced


class TaskSpec(BaseModel):
    """Top-level task. Serialised to and from ``reaction_spec.yaml``."""

    model_config = ConfigDict(extra="forbid", validate_assignment=False)

    task_id: str
    task_mode: TaskMode = TaskMode.ENZYME_MINING
    reaction: ReactionSpec = Field(default_factory=ReactionSpec)
    conditions: Conditions = Field(default_factory=Conditions)
    budget: Budget = Field(default_factory=Budget)
    objectives: Objectives = Field(default_factory=Objectives)
    approval: Approval = Field(default_factory=Approval)
    parent_enzymes: list[str] = Field(
        default_factory=list,
        description="Mode C only: accessions or sequence hashes of the parents.",
    )
    assumptions: list[Assumption] = Field(default_factory=list)

    # -- unresolved-field discipline --------------------------------------
    def gate_requirements(self, gate: str) -> tuple[str, ...]:
        """The fields this gate needs, chosen by the substrate's kind."""
        table = (BIOPOLYMER_GATE_REQUIREMENTS
                 if self.reaction.substrate_kind.is_biopolymer
                 else GATE_REQUIREMENTS)
        return table.get(gate, ())

    def unresolved_for(self, gate: str) -> list[str]:
        out: list[str] = []
        for path in self.gate_requirements(gate):
            value = _get_path(self, path)
            if value is None or (isinstance(value, (list, tuple, dict))
                                 and len(value) == 0):
                out.append(path)
        return out

    def require(self, gate: str) -> None:
        missing = self.unresolved_for(gate)
        if missing:
            raise UnresolvedFieldError(missing, gate)

    def resolve(self, field_path: str, value: Any, source: str,
                justification: str = "", at: str | None = None) -> Assumption:
        """Fill an unresolved field, recording who said so.

        Raises if ``source`` is not a recognised authority, which is what stops
        a language model from quietly promoting its own guess into the spec.
        """
        coerced = _set_path(self, field_path, value)
        # The ledger records what was stored, not what was offered. If the
        # field's type coerced the value, the entry has to say so or the
        # audit trail describes a different task from the one that runs.
        a = Assumption(field_path=field_path, value=coerced, source=source,
                       justification=justification, at=at)
        self.assumptions.append(a)
        return a

    # -- mode-specific sanity ---------------------------------------------
    @model_validator(mode="after")
    def _mode_consistency(self) -> "TaskSpec":
        if self.task_mode is TaskMode.SUBSTRATE_DIRECTED_ENGINEERING and not self.parent_enzymes:
            raise ValueError("substrate_directed_engineering requires parent_enzymes")
        if self.task_mode is TaskMode.ENZYME_MINING and self.reaction.substrate.name is None \
                and not self.reaction.substrate.is_structurally_defined:
            # Allowed at creation time: the spec starts mostly null by design.
            pass
        return self

    @property
    def stereo_task(self) -> bool:
        """Whether this task actually creates a stereocentre.

        Reduction of an aldehyde or a symmetric ketone does not, and must not
        be scored against an enantioselectivity objective.
        """
        return self.reaction.product.requires_chiral_analysis
