"""The staged rollout, written as data the code can check rather than prose.

Why this module exists
----------------------
A rollout plan that lives in a design document is a plan nobody can fail. It
says "stage 2 needs experimental data for the parent family" and no component
ever evaluates that sentence, so stage 2 starts on the day somebody feels ready
and the model it produces is reported with an accuracy figure that measures
re-testing near neighbours of its own training set.

So the plan here is typed:

* :class:`WorkPackage` carries the sources it needs, the deliverables it
  produces, and the preconditions it needs from earlier packages. A package
  cannot silently depend on something no earlier package produces, because
  :func:`validate_plan` refuses to build the plan in that case.
* :func:`readiness` answers "what can actually run now", given what is
  registered and what the operator says is reachable. Nothing is assumed
  reachable: every registry entry carries ``connectivity_verified=false``, so a
  source counts as available only when a caller asserts it.
* :func:`extrapolation_check` and :func:`answer_stage2_gate` make stage 2's
  gating question computable: is there experimental data matching this parent
  family and substrate space, and would using it be genuine extrapolation
  rather than re-testing near neighbours of the training set?
* :func:`check_novelty_budget` encodes the stage 3 rule that distant sequences
  without functional evidence may trickle in early but must never consume a
  whole experimental round.

Stage ordering is the point of the whole file. Stage 1 closes the natural
enzyme discovery loop; stage 2 only becomes meaningful once stage 1 has
produced a parent family and real records; stage 3 is breadth, which is the
most expensive thing to reach for first.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..errors import EAgentError
from ..provenance import sequence_hash
from .layers import DataLayer
from .registry import SourceRegistry, UnknownSourceError

try:  # lineage is a sibling module; guarded so the plan imports standalone
    from .lineage import ClusterLookup as _ClusterLookup
except Exception:  # pragma: no cover - sibling may be mid-write
    _ClusterLookup = None  # type: ignore[assignment]

#: A sequence-id to cluster-id mapping or callable; mirrors
#: :data:`eagent.datalayer.lineage.ClusterLookup` and falls back to the same
#: shape when that module cannot be imported.
ClusterLookup = (_ClusterLookup if _ClusterLookup is not None
                 else Mapping[str, str] | Callable[[str], str | None] | None)

__all__ = [
    "BlockReason",
    "Blocker",
    "ClusterLookup",
    "Deliverable",
    "ExtrapolationReport",
    "ExtrapolationVerdict",
    "MGNIFY_EARLY_ENTRY_RULE",
    "NoveltyBudgetPolicy",
    "NoveltyBudgetVerdict",
    "PackageReadiness",
    "PlanError",
    "Precondition",
    "STAGES",
    "Stage2GateAnswer",
    "StagePlan",
    "StageReadiness",
    "UnknownPackageError",
    "WorkPackage",
    "all_packages",
    "answer_stage2_gate",
    "check_novelty_budget",
    "extrapolation_check",
    "get_stage",
    "package",
    "readiness",
    "validate_plan",
]


class PlanError(EAgentError):
    """The rollout plan itself is inconsistent.

    Raised at import time rather than at run time: a package depending on a
    deliverable nobody produces is a bug in the plan, and discovering it half
    way through a campaign costs a round.
    """


class UnknownPackageError(PlanError):
    """A work package id was requested that the plan does not define."""


# ---------------------------------------------------------------------------
# Plan model
# ---------------------------------------------------------------------------

class Deliverable(BaseModel):
    """One concrete thing a work package hands to later packages.

    Named rather than described in prose so :func:`readiness` can check that a
    package's preconditions are actually produced by something upstream. A
    deliverable that nothing consumes and a precondition nothing produces are
    both plan bugs, and both are invisible in a prose roadmap.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., min_length=2)
    description: str = Field(..., min_length=10)
    is_blocking: bool = Field(
        True,
        description="False for a deliverable later stages can proceed without, "
                    "such as a supplementary hypothesis.")


class Precondition(BaseModel):
    """A named deliverable a package needs from an earlier package.

    Carrying the reason as well as the dependency is deliberate: when
    :func:`readiness` reports a block, the operator needs to know what breaks if
    they run the package anyway, not merely that an arrow is missing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    package_id: str
    deliverable: str
    why: str = Field(..., min_length=10)


class WorkPackage(BaseModel):
    """One unit of the rollout: its sources, deliverables and preconditions.

    Exists so the plan can be executed and audited by the same object. The
    ``must_not_claim`` field mirrors the registry's ``not_good_for``: a package
    that collects family annotations must not report them as activity, and the
    place to write that down is next to the package that produces them.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=3, pattern=r"^s[123]_[a-z0-9_]+$")
    stage: int = Field(..., ge=1, le=3)
    title: str = Field(..., min_length=5)
    question: str = Field(..., min_length=10)
    layers: tuple[DataLayer, ...] = Field(..., min_length=1)
    source_ids: tuple[str, ...] = Field(
        ..., min_length=1,
        description="Registry ids this package cannot run without.")
    supplementary_source_ids: tuple[str, ...] = Field(
        (), description="Registry ids that improve the package but do not block "
                        "it; their absence is a warning, not a block.")
    deliverables: tuple[Deliverable, ...] = Field(..., min_length=1)
    preconditions: tuple[Precondition, ...] = ()
    must_not_claim: tuple[str, ...] = Field(
        ..., min_length=1,
        description="Required. What this package's output must never be read as.")
    notes: str = ""

    @model_validator(mode="after")
    def _coherent(self) -> "WorkPackage":
        if not self.id.startswith(f"s{self.stage}_"):
            raise PlanError(
                f"package '{self.id}' declares stage {self.stage}; the id prefix "
                f"and the stage must agree so a reader cannot mistake one for "
                f"the other")
        names = [d.name for d in self.deliverables]
        if len(set(names)) != len(names):
            raise PlanError(f"package '{self.id}' repeats a deliverable name")
        overlap = set(self.source_ids) & set(self.supplementary_source_ids)
        if overlap:
            raise PlanError(
                f"package '{self.id}' lists {sorted(overlap)} as both required "
                f"and supplementary; a source is one or the other")
        if self.id in {p.package_id for p in self.preconditions}:
            raise PlanError(f"package '{self.id}' depends on itself")
        return self

    def deliverable(self, name: str) -> Deliverable | None:
        for d in self.deliverables:
            if d.name == name:
                return d
        return None

    @property
    def blocking_deliverables(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.deliverables if d.is_blocking)

    def describe(self) -> str:
        lines = [f"[{self.id}] stage {self.stage}: {self.title}",
                 f"  question: {self.question}",
                 f"  sources: {', '.join(self.source_ids)}"]
        if self.supplementary_source_ids:
            lines.append(f"  supplementary: "
                         f"{', '.join(self.supplementary_source_ids)}")
        for d in self.deliverables:
            lines.append(f"  delivers: {d.name} -- {d.description}")
        for p in self.preconditions:
            lines.append(f"  needs: {p.package_id}.{p.deliverable} -- {p.why}")
        for m in self.must_not_claim:
            lines.append(f"  must not claim: {m}")
        return "\n".join(lines)


class StagePlan(BaseModel):
    """One rollout stage: its intent, its gating question and its packages."""

    model_config = ConfigDict(extra="forbid")

    stage: int = Field(..., ge=1, le=3)
    title: str = Field(..., min_length=5)
    intent: str = Field(..., min_length=20)
    gating_question: str = Field(
        ..., min_length=20,
        description="The question that must be answered before the stage is "
                    "worth running. Stage 2's is answered by "
                    ":func:`answer_stage2_gate`, not by assertion.")
    packages: tuple[WorkPackage, ...] = Field(..., min_length=1)
    notes: str = ""

    @model_validator(mode="after")
    def _ids_unique_and_in_stage(self) -> "StagePlan":
        ids = [p.id for p in self.packages]
        if len(set(ids)) != len(ids):
            raise PlanError(f"stage {self.stage} repeats a package id")
        for p in self.packages:
            if p.stage != self.stage:
                raise PlanError(
                    f"package '{p.id}' declares stage {p.stage} but sits in the "
                    f"stage {self.stage} plan")
        return self

    def package(self, package_id: str) -> WorkPackage:
        for p in self.packages:
            if p.id == package_id:
                return p
        raise UnknownPackageError(
            f"stage {self.stage} has no package '{package_id}'; it defines "
            f"{', '.join(p.id for p in self.packages)}")

    def source_ids(self) -> tuple[str, ...]:
        out: list[str] = []
        for p in self.packages:
            out.extend(p.source_ids)
            out.extend(p.supplementary_source_ids)
        return tuple(dict.fromkeys(out))

    def describe(self) -> str:
        lines = [f"Stage {self.stage}: {self.title}",
                 f"  intent: {self.intent}",
                 f"  gate:   {self.gating_question}"]
        for p in self.packages:
            lines.append("")
            lines.append(p.describe())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 1 -- closing the natural-enzyme discovery loop
# ---------------------------------------------------------------------------

_S1_SUBSTRATE = WorkPackage(
    id="s1_substrate_and_reaction",
    stage=1,
    title="Substrate and reaction definition",
    question="What exactly is the substrate, which bond changes, which way does "
             "the reaction run, and what is the product configuration?",
    layers=(DataLayer.REACTION_AND_CHEMISTRY,),
    source_ids=("pubchem", "chebi", "rhea"),
    deliverables=(
        Deliverable(
            name="normalised_substrate",
            description="One structure with an InChIKey and isomeric SMILES, "
                        "replacing the trivial name the task arrived with."),
        Deliverable(
            name="target_product",
            description="The product structure including the configuration "
                        "being asked for, so selectivity is falsifiable."),
        Deliverable(
            name="reaction_direction",
            description="Which direction counts as the target, recorded so an "
                        "oxidation measurement cannot be read as support."),
        Deliverable(
            name="atom_labelling",
            description="Atom-mapped reaction SMILES identifying the carbonyl "
                        "carbon, the hydride acceptor and the prochiral face."),
    ),
    must_not_claim=(
        "a reaction record says nothing about whether any protein catalyses it",
        "a ChEBI or PubChem entry for a name is not proof the task meant that "
        "compound; the operator confirms the structure at the reaction gate",
    ),
    notes="Runs first because every later package keys off the normalised "
          "substrate. Searching for the wrong reaction efficiently is the most "
          "expensive failure in the pipeline.",
)

_S1_SEEDS = WorkPackage(
    id="s1_known_activity_seeds",
    stage=1,
    title="Known-activity seed collection",
    question="Which enzymes have actually been measured on this substrate or a "
             "near neighbour, under what conditions, and with what detection?",
    layers=(DataLayer.ENZYMOLOGY_EVIDENCE, DataLayer.LITERATURE_AND_FEEDBACK),
    source_ids=("brenda", "oed", "pubmed", "europe_pmc"),
    supplementary_source_ids=("retrobiocat_db", "sabio_rk"),
    deliverables=(
        Deliverable(
            name="seed_enzymes",
            description="Sourced seed enzymes, each carrying the source id it "
                        "came from and the evidence strength that source can "
                        "support."),
        Deliverable(
            name="experimental_records",
            description="ExperimentRecords with substrate, conditions, outcome "
                        "class and detection, ingested at their honest tier."),
    ),
    preconditions=(
        Precondition(
            package_id="s1_substrate_and_reaction",
            deliverable="normalised_substrate",
            why="searching BRENDA by trivial name returns records for a "
                "different compound with the same common name, and nothing "
                "downstream can detect that afterwards"),
        Precondition(
            package_id="s1_substrate_and_reaction",
            deliverable="reaction_direction",
            why="without the target direction, alcohol oxidation records enter "
                "the seed set as positives"),
    ),
    must_not_claim=(
        "an EC-number-plus-species mapping is not a sequence-level experimental "
        "label; promotion needs a recorded human review",
        "BRENDA, OED and a re-publishing resource agreeing is not three-fold "
        "corroboration when they share an upstream",
        "RetroBioCat is a supplement for coverage, not an activity measurement "
        "on this substrate",
    ),
)

_S1_EXPANSION = WorkPackage(
    id="s1_sequence_family_expansion",
    stage=1,
    title="Sequence and family expansion",
    question="Which sequences are worth expanding to, and which belong to the "
             "same mechanism family as the seeds?",
    layers=(DataLayer.SEQUENCE_FAMILY_EVOLUTION,),
    source_ids=("uniprotkb", "interpro", "pfam", "ncbi_protein"),
    supplementary_source_ids=("sdred", "akr_superfamily", "uniref", "uniparc"),
    deliverables=(
        Deliverable(
            name="candidate_set",
            description="A deduplicated candidate set: one row per distinct "
                        "sequence, not one row per database that holds it."),
        Deliverable(
            name="family_labels",
            description="Family and subfamily calls built from several "
                        "independent signals, with conflicts recorded."),
        Deliverable(
            name="numbering_maps",
            description="Per-candidate maps from the family reference numbering "
                        "to the candidate's own residue indices."),
    ),
    preconditions=(
        Precondition(
            package_id="s1_known_activity_seeds",
            deliverable="seed_enzymes",
            why="expansion without seeds is an untargeted sweep of sequence "
                "space; the seeds are what make a hit interpretable"),
    ),
    must_not_claim=(
        "family membership is not activity on the target substrate",
        "cluster co-membership is not functional equivalence",
        "SDRED and the AKR superfamily resource supply family structure and "
        "numbering conventions, not measured activity",
    ),
    notes="NCBI identical-protein groups are used to collapse the same protein "
          "appearing under many accessions; without that the candidate set is "
          "one protein sampled many times.",
)

_S1_STRUCTURE = WorkPackage(
    id="s1_structure_and_mechanism",
    stage=1,
    title="Structure and mechanism preparation",
    question="Can the substrate, the cofactor and the catalytic residues form a "
             "sensible arrangement in these candidates?",
    layers=(DataLayer.STRUCTURE_AND_MECHANISM,),
    source_ids=("rcsb_pdb", "alphafold_db", "sifts", "wwpdb_ccd"),
    supplementary_source_ids=("mcsa",),
    deliverables=(
        Deliverable(
            name="checkable_structures",
            description="Experimental or predicted structures paired with the "
                        "candidate sequence they actually correspond to."),
        Deliverable(
            name="ligand_identities",
            description="Chemical component identities and atom naming for "
                        "every bound ligand, so cofactor state is read rather "
                        "than assumed."),
        Deliverable(
            name="catalytic_templates",
            description="Catalytic residue roles and geometric constraints, "
                        "sourced rather than inferred from a sequence motif."),
    ),
    preconditions=(
        Precondition(
            package_id="s1_sequence_family_expansion",
            deliverable="candidate_set",
            why="a structure is only usable once it is bound to the candidate "
                "whose sequence it represents; a deposited point mutant looks "
                "identical otherwise"),
        Precondition(
            package_id="s1_sequence_family_expansion",
            deliverable="numbering_maps",
            why="a catalytic residue assignment without a numbering map lands "
                "on the wrong residue and the resulting geometry still looks "
                "plausible"),
    ),
    must_not_claim=(
        "a plausible geometry is a hypothesis, not a turnover number",
        "a predicted complex is not an observed one",
        "a ligand present in a crystal is not evidence the cofactor state is "
        "the catalytically relevant one",
    ),
)

_S1_SUPPLEMENTARY = WorkPackage(
    id="s1_supplementary_context",
    stage=1,
    title="Supplementary cofactor and assay context",
    question="Where does the cofactor sit, and what assay context have others "
             "used for this chemistry?",
    layers=(DataLayer.STRUCTURE_AND_MECHANISM, DataLayer.ENZYMOLOGY_EVIDENCE),
    source_ids=("alphafill", "sabio_rk"),
    deliverables=(
        Deliverable(
            name="cofactor_placement_hypotheses",
            description="Transplanted cofactor placements, labelled with the "
                        "LigandSource that produced them so they can never be "
                        "reported as an observed binding site.",
            is_blocking=False),
        Deliverable(
            name="assay_context",
            description="Reported buffers, pH, temperature and cofactor "
                        "recycling systems for comparable chemistry, as "
                        "starting points for the operator to confirm.",
            is_blocking=False),
    ),
    preconditions=(
        Precondition(
            package_id="s1_structure_and_mechanism",
            deliverable="checkable_structures",
            why="a transplanted cofactor needs a structure to be transplanted "
                "into, and the transplant must be recorded against that exact "
                "structure"),
    ),
    must_not_claim=(
        "a homology-transplanted ligand is not a confirmed binding site",
        "an assay condition taken from another substrate is a starting point "
        "for the operator, not a validated protocol",
    ),
    notes="Both deliverables are non-blocking: stage 1 can close without them, "
          "and a run that treats them as required will stall on resources that "
          "may not be reachable.",
)

STAGE_1 = StagePlan(
    stage=1,
    title="Close the natural-enzyme discovery loop",
    intent="Get from a substrate name to a ranked, mechanistically checkable "
           "candidate set with a real evidence trail, using the smallest set of "
           "resources that makes the pilot task answerable.",
    gating_question="Is the reaction specified precisely enough to be "
                    "falsifiable, and is there at least one sourced record of "
                    "the target chemistry to anchor the search?",
    packages=(_S1_SUBSTRATE, _S1_SEEDS, _S1_EXPANSION, _S1_STRUCTURE,
              _S1_SUPPLEMENTARY),
    notes="Stage 1 is the only stage that must complete. Stages 2 and 3 are "
          "worth running only on top of its output.",
)


# ---------------------------------------------------------------------------
# Stage 2 -- mutation optimisation and model evaluation
# ---------------------------------------------------------------------------

_S2_VARIANT_EFFECTS = WorkPackage(
    id="s2_variant_effect_corpus",
    stage=2,
    title="Variant-effect and engineering-campaign corpus",
    question="Which positions have been changed before in this family, and what "
             "did the change cost?",
    layers=(DataLayer.MUTATION_AND_PERFORMANCE,),
    source_ids=("fireprotdb", "mavedb", "enzengdb"),
    deliverables=(
        Deliverable(
            name="variant_effect_records",
            description="Mutation-level records with the measured endpoint "
                        "named, so stability, binding and activity are not "
                        "pooled onto one scale."),
        Deliverable(
            name="position_priors",
            description="Positions with prior evidence of tolerance or "
                        "intolerance, mapped through the family numbering."),
    ),
    preconditions=(
        Precondition(
            package_id="s1_sequence_family_expansion",
            deliverable="family_labels",
            why="a variant-effect record from another family is not a prior for "
                "this one, and the family label is the only thing that says so"),
        Precondition(
            package_id="s1_sequence_family_expansion",
            deliverable="numbering_maps",
            why="a position number without a numbering map points at a "
                "different residue in every homolog"),
    ),
    must_not_claim=(
        "a stability or binding measurement is not a measurement of activity on "
        "the target substrate",
        "a deep mutational scan's fitness score is not a kinetic constant",
    ),
)

_S2_KINETICS = WorkPackage(
    id="s2_kinetics_and_reaction_scope",
    stage=2,
    title="Kinetics and reaction-scope corpus for model evaluation",
    question="What quantitative enzyme-substrate data exists that could train or "
             "evaluate a model for this chemistry?",
    layers=(DataLayer.MUTATION_AND_PERFORMANCE, DataLayer.ENZYMOLOGY_EVIDENCE),
    source_ids=("skid", "intenzydb", "reactzyme"),
    supplementary_source_ids=("esibank",),
    deliverables=(
        Deliverable(
            name="training_records",
            description="Enzyme-substrate records usable as model training or "
                        "evaluation rows, each carrying its origin so a split "
                        "can be made leakage-safe."),
        Deliverable(
            name="substrate_space_map",
            description="Which substrate chemistries these corpora actually "
                        "cover, as opposed to which EC numbers they mention."),
    ),
    preconditions=(
        Precondition(
            package_id="s1_substrate_and_reaction",
            deliverable="normalised_substrate",
            why="substrate-space overlap cannot be computed against a name"),
    ),
    must_not_claim=(
        "a benchmark set is not an activity set for this substrate",
        "coverage of an EC number is not coverage of the substrate space",
        "records re-curated from a shared upstream are not independent rows",
    ),
)

_S2_MODEL_GATE = WorkPackage(
    id="s2_model_evaluation_gate",
    stage=2,
    title="Model-use gate: extrapolation or re-testing?",
    question="Is there experimental data matching the current parent family and "
             "substrate space, and would using it be genuine extrapolation "
             "rather than re-testing near neighbours of the training set?",
    layers=(DataLayer.MUTATION_AND_PERFORMANCE,),
    source_ids=("skid", "intenzydb"),
    supplementary_source_ids=("reactzyme", "esibank", "mavedb"),
    deliverables=(
        Deliverable(
            name="extrapolation_report",
            description="The computed overlap between the candidate set and the "
                        "training corpus, with the recommendation on whether "
                        "model outputs stay marked uncalibrated."),
        Deliverable(
            name="uncalibrated_marking",
            description="The marking itself, attached to every model-derived "
                        "score so a deliverable cannot drop it."),
    ),
    preconditions=(
        Precondition(
            package_id="s2_kinetics_and_reaction_scope",
            deliverable="training_records",
            why="the gate compares the candidate set against the training rows; "
                "without them the overlap is undefined, not zero"),
        Precondition(
            package_id="s1_sequence_family_expansion",
            deliverable="candidate_set",
            why="there is nothing to compute an overlap against until the "
                "candidate set exists"),
    ),
    must_not_claim=(
        "a low overlap is not evidence the model generalises; it means no "
        "in-domain validation data exists",
        "a high accuracy on an overlapping evaluation set is memorisation, not "
        "calibration",
    ),
    notes="Implemented by answer_stage2_gate() and extrapolation_check(); the "
          "gating question is computed, not asserted.",
)

STAGE_2 = StagePlan(
    stage=2,
    title="Support mutation optimisation and model evaluation",
    intent="Add the variant-effect and kinetics corpora needed to propose "
           "mutations and to say honestly what a model's output is worth, on "
           "top of a parent family that stage 1 has already identified.",
    gating_question="Is there experimental data matching the current parent "
                    "family and substrate space, and would using it be genuine "
                    "extrapolation rather than re-testing near neighbours of "
                    "the training set?",
    packages=(_S2_VARIANT_EFFECTS, _S2_KINETICS, _S2_MODEL_GATE),
    notes="Running stage 2 before stage 1 has produced a parent family gives a "
          "model fitted to whichever family the public corpora happen to "
          "over-represent.",
)


# ---------------------------------------------------------------------------
# Stage 3 -- novelty and generality
# ---------------------------------------------------------------------------

#: The stage 3 rule, quoted in refusals so it survives into logs and reports.
MGNIFY_EARLY_ENTRY_RULE = (
    "metagenomic candidates may enter an earlier round in small numbers, as a "
    "deliberately bounded exploration slot, but a whole experimental round must "
    "not be spent on distant sequences with no functional evidence: a round "
    "that returns nothing teaches nothing, because a negative on an unexpressed "
    "or untestable protein does not even tell you the sequence was wrong."
)

_S3_NOVEL_SPACE = WorkPackage(
    id="s3_novel_sequence_space",
    stage=3,
    title="Novel and distant sequence space",
    question="What lies outside the characterised families, and which parts of "
             "it are worth a bounded exploration slot?",
    layers=(DataLayer.SEQUENCE_FAMILY_EVOLUTION,),
    source_ids=("mgnify_proteins", "cath_funfam"),
    supplementary_source_ids=("eggnog",),
    deliverables=(
        Deliverable(
            name="exploratory_candidates",
            description="Distant candidates, each labelled with the fact that "
                        "it carries no functional evidence, and counted against "
                        "the exploration budget."),
        Deliverable(
            name="fold_level_context",
            description="Structural-family context for distant hits, so an "
                        "unannotated sequence is at least placed.",
            is_blocking=False),
    ),
    preconditions=(
        Precondition(
            package_id="s1_sequence_family_expansion",
            deliverable="candidate_set",
            why="the exploration budget is a fraction of a real candidate set; "
                "without one there is nothing to take a fraction of"),
    ),
    must_not_claim=(
        "a metagenomic protein-sequence prediction is not an observed protein",
        "fold or FunFam membership is not activity on the target substrate",
        MGNIFY_EARLY_ENTRY_RULE,
    ),
)

_S3_GENERALITY = WorkPackage(
    id="s3_reaction_generality",
    stage=3,
    title="Reaction generality and literature-scale relations",
    question="Does this chemistry generalise beyond the pilot substrate, and "
             "what has the literature already related to it at scale?",
    layers=(DataLayer.REACTION_AND_CHEMISTRY, DataLayer.LITERATURE_AND_FEEDBACK),
    source_ids=("retrorules", "enzchemred"),
    deliverables=(
        Deliverable(
            name="generalised_reaction_rules",
            description="Reaction rules covering neighbouring substrates, as "
                        "hypotheses for the next task, not as evidence."),
        Deliverable(
            name="literature_relation_set",
            description="Machine-extracted enzyme-chemistry relations, ingested "
                        "as machine_extracted_pending and never as labels."),
    ),
    preconditions=(
        Precondition(
            package_id="s1_substrate_and_reaction",
            deliverable="atom_labelling",
            why="a reaction rule can only be matched against a mapped reaction"),
    ),
    must_not_claim=(
        "a reaction rule firing is not a prediction that an enzyme performs it",
        "a machine-extracted relation is not a curated record",
    ),
)

_S3_STRAIN_CONTEXT = WorkPackage(
    id="s3_strain_and_growth_context",
    stage=3,
    title="Strain and growth context",
    question="What is known about the organisms the distant candidates came "
             "from, and does that constrain expression?",
    layers=(DataLayer.LITERATURE_AND_FEEDBACK,),
    source_ids=("bacdive",),
    deliverables=(
        Deliverable(
            name="strain_context",
            description="Growth temperature, oxygen requirement and culture "
                        "notes for source organisms, as context for expression "
                        "host choice.",
            is_blocking=False),
    ),
    preconditions=(
        Precondition(
            package_id="s3_novel_sequence_space",
            deliverable="exploratory_candidates",
            why="strain context is only worth retrieving for candidates that "
                "are actually under consideration"),
    ),
    must_not_claim=(
        "a strain's growth temperature is not its enzyme's optimum, and neither "
        "is evidence of soluble expression in a heterologous host",
    ),
)

STAGE_3 = StagePlan(
    stage=3,
    title="Novelty and generality",
    intent="Reach outside the characterised families and beyond the pilot "
           "substrate, under an explicit budget, once stages 1 and 2 have "
           "produced something to compare against.",
    gating_question="Has a round already produced confirmed activity, so that a "
                    "bounded exploration slot can be afforded without risking "
                    "an entire round on sequences with no functional evidence?",
    packages=(_S3_NOVEL_SPACE, _S3_GENERALITY, _S3_STRAIN_CONTEXT),
    notes=MGNIFY_EARLY_ENTRY_RULE,
)


#: The whole plan, by stage number.
STAGES: dict[int, StagePlan] = {1: STAGE_1, 2: STAGE_2, 3: STAGE_3}


def all_packages() -> dict[str, WorkPackage]:
    """Every work package in the plan, keyed by id, in stage order."""
    out: dict[str, WorkPackage] = {}
    for n in (1, 2, 3):
        for p in STAGES[n].packages:
            out[p.id] = p
    return out


def get_stage(stage: int) -> StagePlan:
    """One stage plan. Raises rather than returning None on a bad stage number."""
    try:
        return STAGES[int(stage)]
    except (KeyError, ValueError, TypeError) as exc:
        raise PlanError(f"no such stage {stage!r}; the plan has stages 1, 2 and 3") \
            from exc


def package(package_id: str) -> WorkPackage:
    """One work package by id, from any stage."""
    try:
        return all_packages()[package_id]
    except KeyError as exc:
        raise UnknownPackageError(
            f"unknown work package '{package_id}'; the plan defines "
            f"{', '.join(sorted(all_packages()))}") from exc


def validate_plan(registry: SourceRegistry | None = None) -> list[str]:
    """Check the plan is internally consistent; return advisory warnings.

    Raises :class:`PlanError` on a precondition naming a package or deliverable
    that does not exist, or on a dependency cycle. Those are plan bugs that
    would otherwise surface as a readiness report that is quietly wrong.

    When a ``registry`` is supplied, source ids are checked against it and the
    unregistered ones are returned as warnings rather than raised, because the
    registry is curated separately and may legitimately lag the plan.
    """
    packages = all_packages()
    warnings: list[str] = []

    for pid, pkg in packages.items():
        for pre in pkg.preconditions:
            upstream = packages.get(pre.package_id)
            if upstream is None:
                raise PlanError(
                    f"package '{pid}' requires '{pre.package_id}', which the "
                    f"plan does not define")
            if upstream.stage > pkg.stage:
                raise PlanError(
                    f"package '{pid}' (stage {pkg.stage}) requires "
                    f"'{pre.package_id}' from the later stage {upstream.stage}; "
                    f"stages would deadlock")
            if upstream.deliverable(pre.deliverable) is None:
                raise PlanError(
                    f"package '{pid}' requires deliverable "
                    f"'{pre.package_id}.{pre.deliverable}', which that package "
                    f"does not produce")

    # Cycle detection over the precondition graph.
    state: dict[str, int] = {}

    def visit(pid: str, path: tuple[str, ...]) -> None:
        if state.get(pid) == 2:
            return
        if state.get(pid) == 1:
            raise PlanError(f"cyclic preconditions: {' -> '.join(path + (pid,))}")
        state[pid] = 1
        for pre in packages[pid].preconditions:
            visit(pre.package_id, path + (pid,))
        state[pid] = 2

    for pid in packages:
        visit(pid, ())

    if registry is not None:
        for pid, pkg in packages.items():
            for sid in pkg.source_ids + pkg.supplementary_source_ids:
                if sid not in registry:
                    warnings.append(
                        f"package '{pid}' names source '{sid}', which is not in "
                        f"the registry")
    return warnings


validate_plan()  # plan bugs are import-time failures, not run-time surprises


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------

class BlockReason(str, enum.Enum):
    """Why a package cannot run. Typed so a controller can route on it."""

    SOURCE_NOT_REGISTERED = "source_not_registered"
    SOURCE_NOT_AVAILABLE = "source_not_available"
    UPSTREAM_PACKAGE_BLOCKED = "upstream_package_blocked"
    UPSTREAM_DELIVERABLE_MISSING = "upstream_deliverable_missing"

    def describe(self) -> str:
        return _BLOCK_DOC[self]


_BLOCK_DOC: dict[BlockReason, str] = {
    BlockReason.SOURCE_NOT_REGISTERED:
        "the plan names a source the registry does not define; a curator must "
        "register it before anything can call it",
    BlockReason.SOURCE_NOT_AVAILABLE:
        "the source is registered but the operator has not confirmed it is "
        "reachable; nothing in the registry has been connectivity-tested",
    BlockReason.UPSTREAM_PACKAGE_BLOCKED:
        "an earlier package this one depends on cannot run",
    BlockReason.UPSTREAM_DELIVERABLE_MISSING:
        "an earlier package can run but has not produced the deliverable this "
        "one consumes",
}


@dataclass(frozen=True)
class Blocker:
    """One specific reason a package is blocked, with the thing that blocks it."""

    reason: BlockReason
    subject: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"reason": self.reason.value, "subject": self.subject,
                "detail": self.detail}

    def render(self) -> str:
        return f"{self.reason.value}: {self.subject} -- {self.detail}"


@dataclass(frozen=True)
class PackageReadiness:
    """Whether one package can run now, and on what it is waiting."""

    package_id: str
    stage: int
    runnable: bool
    blockers: tuple[Blocker, ...]
    warnings: tuple[str, ...]
    available_sources: tuple[str, ...]
    missing_sources: tuple[str, ...]

    @property
    def blocked(self) -> bool:
        return not self.runnable

    def blocked_on(self) -> tuple[str, ...]:
        """The subjects blocking this package, for a one-line status."""
        return tuple(b.subject for b in self.blockers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "stage": self.stage,
            "runnable": self.runnable,
            "blockers": [b.as_dict() for b in self.blockers],
            "warnings": list(self.warnings),
            "available_sources": list(self.available_sources),
            "missing_sources": list(self.missing_sources),
        }


@dataclass(frozen=True)
class StageReadiness:
    """What a stage can actually run, given what is registered and reachable."""

    stage: int
    packages: tuple[PackageReadiness, ...]
    upstream_blocked: tuple[str, ...]

    @property
    def runnable_ids(self) -> tuple[str, ...]:
        return tuple(p.package_id for p in self.packages if p.runnable)

    @property
    def blocked_ids(self) -> tuple[str, ...]:
        return tuple(p.package_id for p in self.packages if p.blocked)

    @property
    def stage_runnable(self) -> bool:
        """True only when every package in the stage can run.

        A partially runnable stage is reported as blocked on purpose: the
        deliverables of the packages that cannot run are exactly what the later
        stages consume, so "three of five ran" is not a stage that completed.
        """
        return bool(self.packages) and not self.blocked_ids

    def for_package(self, package_id: str) -> PackageReadiness:
        for p in self.packages:
            if p.package_id == package_id:
                return p
        raise UnknownPackageError(
            f"stage {self.stage} readiness has no package '{package_id}'")

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "stage_runnable": self.stage_runnable,
            "runnable": list(self.runnable_ids),
            "blocked": list(self.blocked_ids),
            "upstream_blocked": list(self.upstream_blocked),
            "packages": [p.as_dict() for p in self.packages],
        }

    def report_lines(self) -> list[str]:
        lines = [f"stage {self.stage} readiness: "
                 f"{len(self.runnable_ids)} runnable, "
                 f"{len(self.blocked_ids)} blocked"]
        for p in self.packages:
            mark = "ok     " if p.runnable else "BLOCKED"
            lines.append(f"  {mark} {p.package_id}")
            for b in p.blockers:
                lines.append(f"            {b.render()}")
            for w in p.warnings:
                lines.append(f"            warning: {w}")
        if self.upstream_blocked:
            lines.append(f"  earlier-stage packages also blocked: "
                         f"{', '.join(self.upstream_blocked)}")
        return lines

    def describe(self) -> str:
        return "\n".join(self.report_lines())


def readiness(
    stage: int | StagePlan,
    registry: SourceRegistry,
    available_sources: Iterable[str] | None = None,
    *,
    completed_packages: Iterable[str] = (),
    available_deliverables: Mapping[str, Iterable[str]] | None = None,
) -> StageReadiness:
    """Report which packages of a stage can run, and what blocks the others.

    ``available_sources`` is what the operator asserts is reachable *in this
    environment, right now*. It is not defaulted to "everything registered",
    and it is not derived from the registry's access modes: every registry
    entry carries ``connectivity_verified=false``, so deriving availability from
    it would manufacture exactly the confidence the registry refuses to state.
    Passing ``None`` therefore means nothing has been confirmed reachable, and
    every package that needs a source is reported blocked. That is the honest
    starting state of a fresh checkout.

    ``completed_packages`` lets a later stage be assessed without re-running the
    earlier ones: a package listed there counts as having produced all of its
    deliverables. ``available_deliverables`` narrows that to specific
    deliverables when an earlier package only partly completed, which is the
    normal case when a supplementary resource was unreachable.

    Blocks cascade: a package whose upstream is blocked is blocked, so the
    report names the root cause rather than a wall of symptoms.
    """
    plan = stage if isinstance(stage, StagePlan) else get_stage(int(stage))
    available = {str(s) for s in (available_sources or ())}
    done = {str(p) for p in completed_packages}
    delivered: dict[str, set[str]] = {
        k: {str(x) for x in v} for k, v in (available_deliverables or {}).items()}

    packages = all_packages()
    results: dict[str, PackageReadiness] = {}
    upstream_blocked: list[str] = []

    def assess(pid: str) -> PackageReadiness:
        if pid in results:
            return results[pid]
        pkg = packages[pid]
        blockers: list[Blocker] = []
        warnings: list[str] = []
        have: list[str] = []
        missing: list[str] = []

        for sid in pkg.source_ids:
            try:
                src = registry.get(sid)
            except UnknownSourceError:
                missing.append(sid)
                blockers.append(Blocker(
                    BlockReason.SOURCE_NOT_REGISTERED, sid,
                    f"package '{pid}' needs '{sid}', which is not registered"))
                continue
            if sid in available:
                have.append(sid)
                if src.is_human_import_only:
                    warnings.append(
                        f"'{sid}' has no programmatic route registered; the "
                        f"operator has asserted it is available, which means a "
                        f"person fetched or curated the records")
                if src.needs_curation:
                    warnings.append(
                        f"'{sid}' is flagged needs_curation: "
                        f"{'; '.join(src.curation_notes) or 'see registry'}")
            else:
                missing.append(sid)
                blockers.append(Blocker(
                    BlockReason.SOURCE_NOT_AVAILABLE, sid,
                    f"'{sid}' is registered but has not been confirmed "
                    f"reachable; nothing in the registry is connectivity-tested"))

        for sid in pkg.supplementary_source_ids:
            if sid not in registry:
                warnings.append(
                    f"supplementary source '{sid}' is not registered; the "
                    f"package can run without it, with reduced coverage")
            elif sid not in available:
                warnings.append(
                    f"supplementary source '{sid}' is not available; the "
                    f"package can run without it, with reduced coverage")
            else:
                have.append(sid)

        for pre in pkg.preconditions:
            if pre.package_id in done or pre.deliverable in delivered.get(
                    pre.package_id, set()):
                continue
            up = assess(pre.package_id)
            if up.blocked:
                if pre.package_id not in upstream_blocked \
                        and packages[pre.package_id].stage != pkg.stage:
                    upstream_blocked.append(pre.package_id)
                blockers.append(Blocker(
                    BlockReason.UPSTREAM_PACKAGE_BLOCKED, pre.package_id,
                    f"'{pid}' needs {pre.package_id}.{pre.deliverable}: "
                    f"{pre.why}"))
            else:
                blockers.append(Blocker(
                    BlockReason.UPSTREAM_DELIVERABLE_MISSING,
                    f"{pre.package_id}.{pre.deliverable}",
                    f"'{pre.package_id}' can run but has not produced "
                    f"'{pre.deliverable}' yet: {pre.why}"))

        res = PackageReadiness(
            package_id=pid,
            stage=pkg.stage,
            runnable=not blockers,
            blockers=tuple(blockers),
            warnings=tuple(dict.fromkeys(warnings)),
            available_sources=tuple(dict.fromkeys(have)),
            missing_sources=tuple(dict.fromkeys(missing)),
        )
        results[pid] = res
        return res

    ordered = tuple(assess(p.id) for p in plan.packages)
    return StageReadiness(
        stage=plan.stage,
        packages=ordered,
        upstream_blocked=tuple(upstream_blocked),
    )


# ---------------------------------------------------------------------------
# Stage 2 gate: extrapolation or re-testing?
# ---------------------------------------------------------------------------

def _sequence_keys(obj: Any) -> tuple[str, ...]:
    """Every identifier a cluster lookup might be keyed by, strongest first.

    Several keys are tried because a candidate is keyed by sequence hash in one
    clustering run and by accession in the next; silently failing to resolve a
    cluster would count a near neighbour as novel, which is the error that makes
    an evaluation look like extrapolation when it is memorisation.
    """
    keys: list[str] = []
    inner = getattr(obj, "sequence_record", None)
    holders = [obj] + ([inner] if inner is not None else [])
    for h in holders:
        for attr in ("sequence_sha256", "parent_sequence_sha256", "accession"):
            v = getattr(h, attr, None)
            if v:
                keys.append(str(v))
        seq = getattr(h, "construct_sequence", None) or getattr(h, "sequence", None)
        if isinstance(seq, str) and seq.strip():
            keys.append(sequence_hash(seq))
        for attr in ("candidate_id", "record_id", "id"):
            v = getattr(h, attr, None)
            if v:
                keys.append(str(v))
    return tuple(dict.fromkeys(keys))


def _lookup_cluster(keys: Sequence[str], lookup: Any) -> str | None:
    if lookup is None:
        return None
    for k in keys:
        try:
            v = lookup(k) if callable(lookup) else lookup.get(k)
        except Exception:
            v = None
        if v:
            return str(v)
    return None


def _item_id(obj: Any, fallback: str) -> str:
    for attr in ("candidate_id", "record_id", "intake_id", "id"):
        v = getattr(obj, attr, None)
        if v:
            return str(v)
    return fallback


class ExtrapolationVerdict(str, enum.Enum):
    """What applying a model to this candidate set would actually be.

    ``UNDETERMINED`` is a distinct verdict rather than a default to one of the
    others, because "most candidates have no resolvable cluster" is a statement
    about the clustering, not about the candidates, and treating it as novelty
    is how an unclustered set gets reported as a frontier.
    """

    RETESTING_NEIGHBOURS = "retesting_near_neighbours"
    MIXED = "mixed"
    GENUINE_EXTRAPOLATION = "genuine_extrapolation"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True)
class ExtrapolationReport:
    """Overlap between a candidate set and a model's training rows.

    ``mark_model_outputs_uncalibrated`` is True in every branch, and that is
    deliberate rather than an oversight. There is no overlap value at which an
    unvalidated model becomes calibrated: a high overlap means the apparent
    accuracy measures memorisation, and a low overlap means no in-domain
    validation data exists at all. The field exists so that the marking is a
    value a downstream deliverable has to carry, not a caveat in a paragraph
    somebody deletes; ``uncalibrated_reason`` says which of the two situations
    this is, which is what actually changes what the operator should do.
    """

    n_candidates: int
    n_resolved: int
    n_unresolved: int
    n_in_training_cluster: int
    n_identical_to_training: int
    n_training_clusters: int
    overlap_fraction: float | None
    overlap_fraction_of_all: float
    verdict: ExtrapolationVerdict
    mark_model_outputs_uncalibrated: bool
    uncalibrated_reason: str
    calibration_possible: bool
    overlapping_candidate_ids: tuple[str, ...]
    unresolved_candidate_ids: tuple[str, ...]
    recommendations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_candidates": self.n_candidates,
            "n_resolved": self.n_resolved,
            "n_unresolved": self.n_unresolved,
            "n_in_training_cluster": self.n_in_training_cluster,
            "n_identical_to_training": self.n_identical_to_training,
            "n_training_clusters": self.n_training_clusters,
            "overlap_fraction": self.overlap_fraction,
            "overlap_fraction_of_all": self.overlap_fraction_of_all,
            "verdict": self.verdict.value,
            "mark_model_outputs_uncalibrated": self.mark_model_outputs_uncalibrated,
            "uncalibrated_reason": self.uncalibrated_reason,
            "calibration_possible": self.calibration_possible,
            "overlapping_candidate_ids": list(self.overlapping_candidate_ids),
            "unresolved_candidate_ids": list(self.unresolved_candidate_ids),
            "recommendations": list(self.recommendations),
        }

    def report_lines(self) -> list[str]:
        frac = ("n/a" if self.overlap_fraction is None
                else f"{self.overlap_fraction:.2f}")
        lines = [
            f"extrapolation check: {self.verdict.value}",
            f"  candidates {self.n_candidates} "
            f"(resolved {self.n_resolved}, unresolved {self.n_unresolved})",
            f"  in a training cluster: {self.n_in_training_cluster} "
            f"(overlap over resolved: {frac})",
            f"  identical to a training sequence: {self.n_identical_to_training}",
            f"  model outputs stay marked uncalibrated: "
            f"{self.mark_model_outputs_uncalibrated} "
            f"-- {self.uncalibrated_reason}",
        ]
        lines.extend(f"  recommendation: {r}" for r in self.recommendations)
        return lines

    def describe(self) -> str:
        return "\n".join(self.report_lines())


def extrapolation_check(
    candidate_set: Sequence[Any],
    training_records: Sequence[Any],
    cluster_lookup: Any = None,
    *,
    high_overlap: float = 0.5,
    low_overlap: float = 0.1,
    min_resolved_fraction: float = 0.5,
) -> ExtrapolationReport:
    """Measure how much of a candidate set the training corpus already covers.

    This answers the second half of the stage 2 gate. Without it, a model
    trained on public enzyme data is applied to candidates drawn from the same
    few well-studied clusters, scores well, and the score is reported as
    evidence the model works. It is not: it is evidence the candidates were
    near neighbours of the training rows. The failure is invisible because the
    candidate accessions are genuinely different from the training accessions.

    Unresolved clusters are counted apart and never counted as novel. A
    candidate whose cluster could not be determined might be the nearest
    neighbour in the corpus; calling it extrapolation because the lookup was
    incomplete is the same error with extra steps, so a largely unresolved set
    returns ``UNDETERMINED``.

    ``overlap_fraction`` is over the *resolved* candidates, so it is not
    deflated by a poor lookup; ``overlap_fraction_of_all`` is given beside it so
    neither denominator has to be guessed at by the reader.
    """
    cands = list(candidate_set)
    train = list(training_records)

    training_clusters: set[str] = set()
    training_keys: set[str] = set()
    for r in train:
        keys = _sequence_keys(r)
        training_keys.update(keys)
        c = _lookup_cluster(keys, cluster_lookup)
        if c:
            training_clusters.add(c)

    overlapping: list[str] = []
    unresolved: list[str] = []
    identical = 0
    resolved = 0

    for i, cand in enumerate(cands):
        cid = _item_id(cand, f"candidate{i}")
        keys = _sequence_keys(cand)
        if any(k in training_keys for k in keys):
            identical += 1
        cluster = _lookup_cluster(keys, cluster_lookup)
        if cluster is None:
            unresolved.append(cid)
            # An identical sequence is overlap whether or not it clustered.
            if any(k in training_keys for k in keys):
                overlapping.append(cid)
            continue
        resolved += 1
        if cluster in training_clusters:
            overlapping.append(cid)

    n = len(cands)
    overlap_fraction = (len(overlapping) / resolved) if resolved else None
    overlap_of_all = (len(overlapping) / n) if n else 0.0
    resolved_fraction = (resolved / n) if n else 0.0

    recommendations: list[str] = []

    if n == 0:
        verdict = ExtrapolationVerdict.UNDETERMINED
        reason = ("the candidate set is empty, so the overlap is undefined "
                  "rather than zero")
        calibration_possible = False
        recommendations.append(
            "build the candidate set before asking whether using a model on it "
            "would be extrapolation")
    elif not train:
        verdict = ExtrapolationVerdict.UNDETERMINED
        reason = ("no training rows were supplied, so nothing is known about "
                  "what the model has already seen; an empty training set is "
                  "not evidence of novelty")
        calibration_possible = False
        recommendations.append(
            "supply the model's training rows, or mark the model as having an "
            "undocumented training set, which is itself a reason not to rely "
            "on its outputs")
    elif resolved_fraction < min_resolved_fraction:
        verdict = ExtrapolationVerdict.UNDETERMINED
        reason = (f"only {resolved}/{n} candidates resolved to a sequence "
                  f"cluster; an unresolved cluster is a gap in the clustering, "
                  f"not evidence that the candidate is novel")
        calibration_possible = False
        recommendations.append(
            "cluster the candidate set and the training rows together, with one "
            "clustering run and one identifier scheme, before reading any "
            "overlap number")
    elif overlap_fraction is not None and overlap_fraction >= high_overlap:
        verdict = ExtrapolationVerdict.RETESTING_NEIGHBOURS
        reason = (f"{len(overlapping)}/{resolved} resolved candidates sit in a "
                  f"cluster the training corpus already contains, so an "
                  f"evaluation on this set measures memorisation rather than "
                  f"calibration")
        calibration_possible = False
        recommendations.append(
            "keep every model-derived score marked uncalibrated: a good score "
            "here is the expected result of re-testing near neighbours")
        recommendations.append(
            "if the model is to be evaluated, hold out whole clusters, not "
            "sequences, and report the held-out number separately")
        recommendations.append(
            "spend the experimental round on candidates outside the training "
            "clusters, where the result is informative either way")
    elif overlap_fraction is not None and overlap_fraction <= low_overlap:
        verdict = ExtrapolationVerdict.GENUINE_EXTRAPOLATION
        reason = (f"only {len(overlapping)}/{resolved} resolved candidates "
                  f"overlap the training corpus, so the model is being asked to "
                  f"extrapolate and there is no in-domain data to calibrate it "
                  f"against")
        calibration_possible = False
        recommendations.append(
            "keep every model-derived score marked uncalibrated: there is no "
            "in-domain validation set, so its accuracy here is unmeasured")
        recommendations.append(
            "use the model to order candidates within the mechanistic gates, "
            "not to replace them")
    else:
        verdict = ExtrapolationVerdict.MIXED
        reason = (f"{len(overlapping)}/{resolved} resolved candidates overlap "
                  f"the training corpus; the overlapping subset can be used to "
                  f"check the model, and the rest is extrapolation it cannot "
                  f"speak to")
        calibration_possible = True
        recommendations.append(
            "keep model-derived scores marked uncalibrated for the "
            "non-overlapping candidates, and report any accuracy figure as "
            "applying only to the overlapping subset")

    if unresolved:
        recommendations.append(
            f"{len(unresolved)} candidate(s) had no resolvable cluster and were "
            f"excluded from the overlap denominator; they are not counted as "
            f"novel")
    if identical:
        recommendations.append(
            f"{identical} candidate(s) are the same sequence as a training row; "
            f"testing them measures nothing new")

    return ExtrapolationReport(
        n_candidates=n,
        n_resolved=resolved,
        n_unresolved=len(unresolved),
        n_in_training_cluster=len(overlapping),
        n_identical_to_training=identical,
        n_training_clusters=len(training_clusters),
        overlap_fraction=overlap_fraction,
        overlap_fraction_of_all=overlap_of_all,
        verdict=verdict,
        mark_model_outputs_uncalibrated=True,
        uncalibrated_reason=reason,
        calibration_possible=calibration_possible,
        overlapping_candidate_ids=tuple(dict.fromkeys(overlapping)),
        unresolved_candidate_ids=tuple(dict.fromkeys(unresolved)),
        recommendations=tuple(recommendations),
    )


def _family_labels(obj: Any) -> set[str]:
    out: set[str] = set()
    holders = [obj]
    for attr in ("family", "family_annotation", "record", "sequence_record"):
        v = getattr(obj, attr, None)
        if v is not None:
            holders.append(v)
    for h in holders:
        for attr in ("family_name", "subfamily", "family_template_id",
                     "family", "superfamily"):
            v = getattr(h, attr, None)
            if isinstance(v, str) and v.strip():
                out.add(v.strip().lower())
    return out


def _substrate_keys(obj: Any) -> set[str]:
    out: set[str] = set()
    holders = [obj]
    inner = getattr(obj, "record", None)
    if inner is not None:
        holders.append(inner)
    for h in holders:
        sub = getattr(h, "substrate", None)
        if sub is None:
            continue
        for attr in ("inchikey", "isomeric_smiles"):
            v = getattr(sub, attr, None)
            if isinstance(v, str) and v.strip():
                out.add(v.strip())
    return out


@dataclass(frozen=True)
class Stage2GateAnswer:
    """The computed answer to stage 2's gating question.

    Both halves are needed and neither implies the other: a corpus can match the
    parent family perfectly and still be useless because every candidate is a
    near neighbour of it, and a corpus can be genuinely out-of-domain and
    useless because it does not cover the family at all.
    """

    has_matching_experimental_data: bool
    n_matching_records: int
    n_matching_sequence_level: int
    matched_on_family: int
    matched_on_substrate: int
    extrapolation: ExtrapolationReport
    proceed: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_matching_experimental_data": self.has_matching_experimental_data,
            "n_matching_records": self.n_matching_records,
            "n_matching_sequence_level": self.n_matching_sequence_level,
            "matched_on_family": self.matched_on_family,
            "matched_on_substrate": self.matched_on_substrate,
            "proceed": self.proceed,
            "reasons": list(self.reasons),
            "extrapolation": self.extrapolation.as_dict(),
        }

    def describe(self) -> str:
        lines = [f"stage 2 gate: {'proceed' if self.proceed else 'do not proceed'}",
                 f"  matching experimental records: {self.n_matching_records} "
                 f"({self.n_matching_sequence_level} sequence-level)"]
        lines.extend(f"  {r}" for r in self.reasons)
        lines.extend(f"  {l}" for l in self.extrapolation.report_lines())
        return "\n".join(lines)


def answer_stage2_gate(
    *,
    parent_families: Iterable[str],
    substrate_keys: Iterable[str],
    candidate_set: Sequence[Any],
    training_records: Sequence[Any],
    cluster_lookup: Any = None,
    min_matching_records: int = 1,
) -> Stage2GateAnswer:
    """Compute, rather than assert, whether stage 2 is worth running.

    The gate has two halves. First, does experimental data exist that matches
    the parent family *and* the substrate space we care about? Records matching
    on family alone are a different chemistry on a related scaffold; records
    matching on substrate alone are a different scaffold on our chemistry.
    Neither is a basis for fitting anything, and counting them together is how
    a corpus of a thousand irrelevant rows is reported as coverage.

    Second, :func:`extrapolation_check` on the candidate set, so a "yes" to the
    first half cannot be read as a licence to trust the model's numbers.
    """
    families = {f.strip().lower() for f in parent_families if str(f).strip()}
    substrates = {s.strip() for s in substrate_keys if str(s).strip()}

    matched_family = 0
    matched_substrate = 0
    matched_both = 0
    sequence_level = 0

    for r in training_records:
        fam_hit = bool(families & _family_labels(r)) if families else False
        sub_hit = bool(substrates & _substrate_keys(r)) if substrates else False
        if fam_hit:
            matched_family += 1
        if sub_hit:
            matched_substrate += 1
        if fam_hit and sub_hit:
            matched_both += 1
            strength = getattr(r, "max_strength", None) \
                or getattr(r, "claimed_strength", None)
            if getattr(strength, "is_sequence_level", False):
                sequence_level += 1

    reasons: list[str] = []
    if not families:
        reasons.append(
            "no parent family was supplied, so the family half of the gate "
            "could not be evaluated; it is reported as unmet rather than passed")
    if not substrates:
        reasons.append(
            "no substrate key was supplied, so the substrate half of the gate "
            "could not be evaluated; it is reported as unmet rather than passed")

    has_match = bool(families and substrates and matched_both >= min_matching_records)
    if has_match:
        reasons.append(
            f"{matched_both} record(s) match both the parent family and the "
            f"substrate space, {sequence_level} of them at sequence level")
    else:
        reasons.append(
            f"only {matched_both} record(s) match both the parent family and "
            f"the substrate space (family-only {matched_family}, "
            f"substrate-only {matched_substrate}); fitting or evaluating on "
            f"those would be fitting to a different question")

    report = extrapolation_check(candidate_set, training_records, cluster_lookup)
    reasons.append(f"extrapolation verdict: {report.verdict.value}")
    reasons.append(f"model outputs stay marked uncalibrated: "
                   f"{report.uncalibrated_reason}")

    return Stage2GateAnswer(
        has_matching_experimental_data=has_match,
        n_matching_records=matched_both,
        n_matching_sequence_level=sequence_level,
        matched_on_family=matched_family,
        matched_on_substrate=matched_substrate,
        extrapolation=report,
        proceed=has_match,
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Stage 3 novelty budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoveltyBudgetPolicy:
    """How much of a round may go to distant sequences with no functional evidence.

    These numbers are *project policy*, not measured quantities: nobody has
    determined an optimal exploration fraction for this chemistry, and this
    module will not pretend otherwise. They are defaults an operator changes,
    and they exist because the alternative to a stated default is an unstated
    one applied by whoever assembles the batch.
    """

    max_fraction: float = 0.10
    max_count: int = 8
    min_round_size_for_exploration: int = 24

    def allowance(self, round_size: int) -> int:
        """Slots an exploration may take in a round of ``round_size``."""
        if round_size < self.min_round_size_for_exploration:
            return 0
        return max(0, min(self.max_count, int(round_size * self.max_fraction)))


@dataclass(frozen=True)
class NoveltyBudgetVerdict:
    """Whether a proposed batch respects the exploration budget."""

    round_size: int
    allowance: int
    n_exploratory: int
    n_exploratory_without_evidence: int
    within_budget: bool
    over_by: int
    offending_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    #: Members whose origin source could not be read. Counted as exploratory
    #: and charged against the allowance: a candidate with no provenance is
    #: the least likely to have functional evidence, not the most.
    unknown_origin_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_size": self.round_size,
            "allowance": self.allowance,
            "n_exploratory": self.n_exploratory,
            "n_exploratory_without_evidence": self.n_exploratory_without_evidence,
            "within_budget": self.within_budget,
            "over_by": self.over_by,
            "offending_ids": list(self.offending_ids),
            "unknown_origin_ids": list(self.unknown_origin_ids),
            "n_unknown_origin": self.n_unknown_origin,
            "reasons": list(self.reasons),
        }

    @property
    def n_unknown_origin(self) -> int:
        return len(self.unknown_origin_ids)

    def describe(self) -> str:
        head = ("within the exploration budget" if self.within_budget
                else f"OVER the exploration budget by {self.over_by} slot(s)")
        return "\n".join([f"novelty budget: {head}"]
                         + [f"  {r}" for r in self.reasons])


#: Registry ids whose candidates carry no functional evidence by construction.
_DEFAULT_EXPLORATORY_SOURCES: tuple[str, ...] = ("mgnify_proteins", "uniparc",
                                                 "uniref")


def _origin_source(obj: Any) -> str | None:
    inner = getattr(obj, "sequence_record", None)
    for h in (obj, inner):
        if h is None:
            continue
        for attr in ("source_id", "source_database", "source"):
            v = getattr(h, attr, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _has_functional_evidence(obj: Any) -> bool:
    """Whether anything measured supports this candidate doing anything.

    Annotation and family membership deliberately do not count: the stage 3
    rule is about sequences nobody has ever shown to function, and a predicted
    domain hit is exactly the kind of signal that makes such a sequence look
    supported.
    """
    holders = [obj]
    for attr in ("record", "sequence_record"):
        v = getattr(obj, attr, None)
        if v is not None:
            holders.append(v)
    for h in holders:
        for ev in (getattr(h, "evidence", None) or []):
            strength = getattr(ev, "strength", None)
            rank = getattr(strength, "rank", None)
            if isinstance(rank, int) and rank >= 3:   # homolog- or sequence-level
                return True
        outcome = getattr(h, "outcome", None)
        if getattr(outcome, "informs_catalytic_ability", False):
            return True
    return False


def check_novelty_budget(
    batch: Sequence[Any],
    *,
    policy: NoveltyBudgetPolicy | None = None,
    exploratory_source_ids: Iterable[str] = _DEFAULT_EXPLORATORY_SOURCES,
    has_functional_evidence: Callable[[Any], bool] | None = None,
) -> NoveltyBudgetVerdict:
    """Enforce the stage 3 rule on a proposed experimental round.

    Encodes :data:`MGNIFY_EARLY_ENTRY_RULE`: metagenomic and other distant
    candidates may enter an earlier round in small numbers, but a whole round
    must not be spent on them. The reason is not conservatism. A round of
    distant sequences that returns nothing is uninterpretable -- no expression,
    no soluble protein and no activity are three different results and the
    cheap assays cannot tell them apart -- so the round costs a full cycle and
    produces no information to plan the next one with.

    A member whose origin source cannot be read is counted as exploratory with
    unknown origin and charged against the allowance, and named in the
    reasons. It is not the same case as a member from a known, non-exploratory
    source: no provenance is the weakest position a candidate can be in, and
    passing it through as known traffic produces a clean verdict on a round
    nobody can account for.

    Returns a verdict rather than raising, because the operator may knowingly
    spend the round that way; what must not happen is spending it without the
    cost being stated.
    """
    pol = policy or NoveltyBudgetPolicy()
    evidence_fn = has_functional_evidence or _has_functional_evidence
    exploratory = {str(s) for s in exploratory_source_ids}

    n = len(batch)
    allowance = pol.allowance(n)
    offenders: list[str] = []
    unknown_origin: list[str] = []
    n_exploratory = 0
    n_without = 0

    for i, item in enumerate(batch):
        src = _origin_source(item)
        mid = _item_id(item, f"member{i}")
        if src is None:
            # Not the same case as "from a known, non-exploratory source".
            # A member whose origin cannot be read has no provenance at all,
            # which is the weakest position a candidate can be in, and
            # treating it as known traffic is how an unaccountable round
            # passes the budget check with a clean verdict.
            unknown_origin.append(mid)
        elif src not in exploratory:
            continue
        n_exploratory += 1
        if not evidence_fn(item):
            n_without += 1
            offenders.append(mid)

    over_by = max(0, n_without - allowance)
    reasons = [
        f"round size {n}; exploration allowance {allowance} slot(s) "
        f"(policy: {pol.max_fraction:.0%} of the round, at most {pol.max_count}, "
        f"and none below a round of {pol.min_round_size_for_exploration})",
        f"{n_exploratory} member(s) come from an exploratory source or have "
        f"no readable origin, {n_without} of them with no functional evidence",
    ]
    if unknown_origin:
        shown = ", ".join(unknown_origin[:10])
        if len(unknown_origin) > 10:
            shown += f", ... (+{len(unknown_origin) - 10} more)"
        reasons.append(
            f"{len(unknown_origin)} member(s) record no origin source and are "
            f"counted as exploratory with unknown origin, charged against the "
            f"allowance: {shown}. Record where each came from; a candidate "
            f"nobody can trace is not a known candidate")
    if over_by:
        reasons.append(MGNIFY_EARLY_ENTRY_RULE)
        reasons.append(
            f"remove {over_by} distant candidate(s) without functional "
            f"evidence, or record the operator's decision to spend the round "
            f"this way")
    if n < pol.min_round_size_for_exploration and n_without:
        reasons.append(
            f"a round of {n} is too small to carry an exploration slot at all; "
            f"every member has to be a candidate the round can learn from")

    return NoveltyBudgetVerdict(
        round_size=n,
        allowance=allowance,
        n_exploratory=n_exploratory,
        n_exploratory_without_evidence=n_without,
        within_budget=over_by == 0,
        over_by=over_by,
        offending_ids=tuple(offenders),
        unknown_origin_ids=tuple(unknown_origin),
        reasons=tuple(reasons),
    )
