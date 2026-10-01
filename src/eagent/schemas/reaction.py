"""Task and reaction specification, with the unresolved-field discipline.

A ``None`` field means "not yet determined". It is never filled by a language
model. It is filled only by :meth:`TaskSpec.resolve`, which demands a source
and writes an :class:`Assumption` into the ledger. Gates then refuse to open
while fields they depend on are still unresolved.
"""

from __future__ import annotations

import enum
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import FabricationGuardError, UnresolvedFieldError
from .chem import CofactorSpec, ProductSpec, Stereochemistry, SubstrateSpec


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
        return bool(getattr(self, gate, False))


class ReactionSpec(BaseModel):
    """What chemistry must happen, stated precisely enough to be falsifiable."""

    model_config = ConfigDict(extra="forbid")

    reaction_class: ReactionClass = ReactionClass.OTHER
    atom_mapped_reaction_smiles: str | None = None
    substrate: SubstrateSpec = Field(default_factory=SubstrateSpec)
    product: ProductSpec = Field(default_factory=ProductSpec)
    rhea_id: str | None = None
    ec_hint: str | None = None
    notes: str = ""

    @property
    def target_stereochemistry(self) -> Stereochemistry:
        return self.product.target_stereochemistry

    def unresolved(self, required: Iterable[str]) -> list[str]:
        out = []
        for path in required:
            if _get_path(self, path) is None:
                out.append(path)
        return out


#: Fields that must be resolved before each gate opens.
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


def _get_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        cur = getattr(cur, part, None)
    return cur


def _set_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = getattr(cur, part)
        if cur is None:
            raise UnresolvedFieldError([".".join(parts[:-1])])
    setattr(cur, parts[-1], value)


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
    def unresolved_for(self, gate: str) -> list[str]:
        req = GATE_REQUIREMENTS.get(gate, ())
        return [p for p in req if _get_path(self, p) is None]

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
        a = Assumption(field_path=field_path, value=value, source=source,
                       justification=justification, at=at)
        _set_path(self, field_path, value)
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
