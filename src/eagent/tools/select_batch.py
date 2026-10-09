"""Interface ``select_batch`` -- what the round actually costs, before it is ordered.

WHY THIS STEP EXISTS SEPARATELY FROM RANKING
============================================
Ranking says which candidates look best. It does not say what to build, and
the gap between the two is where screening budgets die:

* **Genes are not wells.** "96 constructs" is a synthesis order. The plate it
  implies is ``candidates x cofactor conditions x replicates``, plus controls
  at the same multiplicity. For 96 genes under two cofactor conditions in
  triplicate that is 576 candidate wells before a single control -- six
  plates, not one. Discovering that after the order is placed means either
  dropping replicates (and losing the ability to call anything) or dropping
  conditions (and losing the cofactor question). The footprint is therefore
  computed and reported *here*, next to the order.
* **Controls are genes too.** A positive-control enzyme has to be
  synthesised like everything else. If the construct count is a hard cap and
  the controls were not reserved up front, the batch is either 98 constructs
  (the cap was not a cap) or 94 chosen candidates plus 2 the selection never
  saw (the selection was not the selection). ``Budget.constructs_include_controls``
  makes that choice explicit and
  :func:`~eagent.science.diversity.reserve_control_slots` takes the slots
  first.
* **A combination variant without its single mutants is unattributable.**
  Round-2 variants are admitted as whole groups: a double mutant drags its two
  single-mutant controls into the same plate, or it does not go in at all.

WHAT A CONTROL IS ENTITLED TO PROVE
===================================
The most expensive confusion in a screening round is between

    "the assay system works"        and        "the target substrate is turned over"

A positive-control enzyme acting on its own known substrate demonstrates the
first and says nothing at all about the second -- it is there so that a plate
of negatives can be attributed to the enzymes rather than to a dead reagent.
:class:`ControlClaim` makes the two different values of an enum, and
:func:`validate_control_claims` refuses a plan in which a background or
system control has been written up as evidence of target turnover.

NO PADDING, AND THE SHORTFALL IS THE RESULT
===========================================
If the gate-passing pool cannot fill the round, the batch comes back short
with :attr:`~eagent.schemas.batch.BatchPlan.shortfall_reason` naming the
arithmetic. There is no branch here that relaxes a gate, a family quota or a
cluster cap to reach a round number. A short batch with a reason is a finding
about the pool; a full batch reached by lowering the bar is a fabricated
experiment.

THE PRE-REGISTERED ENDPOINT IS WRITTEN HERE, BEFORE THE DATA
============================================================
``experiment_plan.yaml`` carries the :class:`~eagent.schemas.templates.AssayTemplate`'s
``positive_criteria`` verbatim, hashed. That is the pre-registration: the
criterion exists, in a file, before a plate is run, and ``ingest_results``
uses that one and no other.

OFFLINE
=======
This step reads objects and writes three files. It opens no socket and
transmits nothing; ``submit_to`` is refused outright, because a selected batch
is a list of unpublished sequences about to be ordered.
"""

from __future__ import annotations

import csv
import enum
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import yaml

from ..context import RunContext
from ..envelope import Artifact, Provenance, Severity, Status, ToolResult
from ..provenance import sha256_file, sha256_obj
from ..schemas import (
    AssayTemplate,
    BatchMember,
    BatchPlan,
    BatchRole,
    Budget,
    Candidate,
    CofactorSpec,
    ControlItem,
    MutationProposal,
)
from ..science.diversity import (
    DEFAULT_LAMBDA_WEIGHT,
    DEFAULT_ROLE_TARGETS,
    compose_batch,
    reserve_control_slots,
)
from .base import ScientificInterface
from .handoff import CANDIDATES_KEY, as_candidates, serialise_candidates

__all__ = [
    "SYNTHESIS_GATE",
    "DEFAULT_WELLS_PER_PLATE",
    "DEFAULT_COFACTOR_CONDITIONS",
    "ASSAY_RESULT_COLUMNS",
    "ASSAY_IDENTITY_COLUMNS",
    "ASSAY_PLAN_COLUMNS",
    "ASSAY_MEASURED_COLUMNS",
    "BATCH_CSV_COLUMNS",
    "ControlClaim",
    "ControlSpec",
    "default_control_plan",
    "validate_control_claims",
    "MeasurementFootprint",
    "measurement_footprint",
    "well_label",
    "LAYOUT_STRATEGY",
    "WellAssignment",
    "PlateLayout",
    "plan_plate_layout",
    "VariantSelection",
    "select_variant_groups",
    "SelectBatch",
]


#: The human decision point this step is gated on. Composing a batch is the
#: last act before money is spent on genes.
SYNTHESIS_GATE: str = "synthesis_authorized"

#: Wells per microplate. A labware constant, not a scientific threshold; it
#: converts a measurement count into a plate count and nothing else. Pass a
#: different value for 384-well work.
DEFAULT_WELLS_PER_PLATE: int = 96

#: Cofactor conditions assumed when the task does not list any. Used only as a
#: fallback and always flagged: for an NAD(P)H-dependent reduction the right
#: number comes from ``TaskSpec.conditions.cofactor_options``, because testing
#: an NADPH-preferring enzyme only with NADH produces a plate of true negatives
#: that look like the chemistry failed.
DEFAULT_COFACTOR_CONDITIONS: int = 2

#: Columns identifying a well, written by this step and expected back unchanged.
#: Reconstructing a plate map afterwards from slot numbers is how a transposed
#: column of results gets attributed to the wrong enzymes.
ASSAY_IDENTITY_COLUMNS: tuple[str, ...] = (
    "plan_id", "plate", "well", "slot",
    "candidate_id", "construct_id", "kind", "role",
    "parent_candidate_id", "mutations",
    "cofactor", "cofactor_state", "replicate",
)

#: Columns stating what the plan ASKED FOR. Pre-filled by this step, read by
#: nobody downstream, and present so the bench can see the intended protocol
#: beside the well it applies to.
#:
#: They exist because the same facts used to be written into the *measured*
#: columns. ``detection_method``, ``confirms_product_identity``,
#: ``authentic_standard``, ``chiral_method_validated`` and the detection limit
#: came out of the AssayTemplate pre-filled, so a sheet returned with only the
#: numbers typed in asserted that a GC-MS reading had identified the product
#: against an authentic standard -- in every well, including the ones nobody
#: ran. ``ingest_results`` then had a record that met the bar on a confirmed
#: identity, and the round reported hits. An intention and an observation are
#: different claims and are now in different columns.
ASSAY_PLAN_COLUMNS: tuple[str, ...] = (
    "plan_detection_method",
    "plan_confirms_product_identity",
    "plan_authentic_standard",
    "plan_chiral_method_validated",
    "plan_limit_of_detection",
    "plan_limit_unit",
    "plan_replicates",
)

#: Columns the bench fills in. Every one of them is blank in the template.
#:
#: There is deliberately **no "hit" or "outcome" column**. The operator records
#: what was measured; the pre-registered criterion in the AssayTemplate decides
#: what it means. A spreadsheet column where a human writes "positive" is how a
#: criterion gets adjusted after the data are seen, without anyone deciding to
#: adjust it.
ASSAY_MEASURED_COLUMNS: tuple[str, ...] = (
    "tested",                       # yes | no  -- blank is NOT yes
    "expressed_soluble",            # yes | no | unknown
    "detection_method",
    "confirms_product_identity",    # yes | no
    "authentic_standard",           # yes | no | unknown
    "chiral_method_validated",      # yes | no | unknown
    "limit_of_detection", "limit_unit",
    "measurement_type", "measurement_value", "measurement_unit",
    "conversion_pct",
    "product_identity_observed",    # target | other | none | unknown
    "peak_area_target_enantiomer", "peak_area_opposite_enantiomer",
    "notes",
)

#: Columns of ``assay_results_template.csv``, which is the contract between
#: this step and ``ingest_results``.
ASSAY_RESULT_COLUMNS: tuple[str, ...] = (
    ASSAY_IDENTITY_COLUMNS + ASSAY_PLAN_COLUMNS + ASSAY_MEASURED_COLUMNS
)

#: Columns of ``selected_batch_<n>.csv``: the order form, with the reason each
#: slot was spent in the same row as the slot.
BATCH_CSV_COLUMNS: tuple[str, ...] = (
    "slot", "kind", "candidate_id", "role", "family", "sequence_cluster_id",
    "utility", "selection_reason", "parent_candidate_id", "mutations",
    "numbering_reference", "decomposition_controls",
    "requires_new_construct", "demonstrates",
)


# ==========================================================================
# Controls
# ==========================================================================

class ControlClaim(str, enum.Enum):
    """What one control is capable of demonstrating. One claim per control.

    This is an enum rather than free text because the two claims at the top of
    the list are routinely merged in practice, and merging them turns "our
    detection chain is alive" into "this enzyme class turns the substrate
    over". Keeping them as different values means a plan cannot assert the
    second by writing a loose sentence.
    """

    ASSAY_SYSTEM_WORKS = "assay_system_works"
    TARGET_SUBSTRATE_TURNED_OVER = "target_substrate_turned_over"
    BACKGROUND_WITHOUT_ENZYME = "background_without_enzyme"
    BACKGROUND_FROM_HOST = "background_from_host"
    COFACTOR_DEPENDENCE = "cofactor_dependence"
    DETECTION_LIMIT_ESTABLISHED = "detection_limit_established"

    def statement(self) -> str:
        return {
            ControlClaim.ASSAY_SYSTEM_WORKS:
                "the assay system works -- reagents, cofactor recycling, "
                "extraction and detection are alive end to end. Says NOTHING "
                "about the target substrate.",
            ControlClaim.TARGET_SUBSTRATE_TURNED_OVER:
                "the target substrate is turned over to the target product "
                "under these conditions, confirmed by a method that identifies "
                "the product.",
            ControlClaim.BACKGROUND_WITHOUT_ENZYME:
                "how much product appears with no enzyme present: the abiotic "
                "and reagent background a hit must exceed.",
            ControlClaim.BACKGROUND_FROM_HOST:
                "how much product appears from host proteins alone, which is "
                "the background an expressed construct must exceed.",
            ControlClaim.COFACTOR_DEPENDENCE:
                "whether the signal depends on the cofactor, separating "
                "cofactor-driven chemistry from a cofactor-independent artefact.",
            ControlClaim.DETECTION_LIMIT_ESTABLISHED:
                "the concentration at which the method would have seen the "
                "product, which is what makes a negative record mean anything.",
        }[self]

    @property
    def is_target_turnover(self) -> bool:
        return self is ControlClaim.TARGET_SUBSTRATE_TURNED_OVER


@dataclass(frozen=True)
class ControlSpec:
    """A control before it becomes a :class:`ControlItem`.

    Carrying the claim as an enum and rendering ``demonstrates`` from it means
    the text in the plan and the machine-checkable claim cannot drift apart.
    """

    name: str
    kind: str                       # no_enzyme | empty_vector | no_cofactor | positive_enzyme
    claim: ControlClaim
    rationale: str = ""
    requires_new_construct: bool = False
    substrate_is_target: bool = False
    product_identity_confirmed: bool = False

    def to_control_item(self) -> ControlItem:
        text = f"{self.claim.value}: {self.claim.statement()}"
        if self.rationale:
            text = f"{text} ({self.rationale})"
        return ControlItem(
            name=self.name, kind=self.kind, demonstrates=text,
            requires_new_construct=self.requires_new_construct,
            # reserve_control_slots normalises this, but setting it here keeps
            # a hand-built plan valid without going through the reservation.
            occupies_batch_slot=self.requires_new_construct,
        )


def default_control_plan(
    *,
    positive_control_enzyme_available: bool,
    positive_control_needs_new_construct: bool = True,
    cofactor_name: str | None = None,
) -> list[ControlSpec]:
    """The minimum control set for a round-1 screen, with honest claims.

    Four controls, each answering a different question, and none of them
    claiming the round's result:

    * **no enzyme** -- abiotic background. A ketone that reduces slowly in the
      buffer will otherwise be read as weak activity across the whole plate.
    * **empty vector** -- host background. *E. coli* lysate contains endogenous
      ketoreductases, and a lysate screen without this control attributes them
      to the construct.
    * **no cofactor** -- cofactor dependence. Without it, a cofactor-independent
      artefact in the detection chain is indistinguishable from catalysis.
    * **positive enzyme** -- the assay system works. Omitted when no such enzyme
      is available, and in that case the shortfall is reported rather than the
      claim being quietly transferred to another control: a plate of negatives
      with no system control cannot be interpreted at all.

    None of the four is given
    :attr:`ControlClaim.TARGET_SUBSTRATE_TURNED_OVER`. In round 1 nothing is
    known to turn the target substrate over -- that is the question the round
    asks -- and a control that claimed it would be assuming the answer.
    """
    specs = [
        ControlSpec(
            name="no_enzyme", kind="no_enzyme",
            claim=ControlClaim.BACKGROUND_WITHOUT_ENZYME,
            rationale="buffer, substrate, cofactor and recycling system only",
        ),
        ControlSpec(
            name="empty_vector", kind="empty_vector",
            claim=ControlClaim.BACKGROUND_FROM_HOST,
            rationale="host lysate carries endogenous reductases",
        ),
        ControlSpec(
            name="no_cofactor", kind="no_cofactor",
            claim=ControlClaim.COFACTOR_DEPENDENCE,
            rationale=(f"omits {cofactor_name}" if cofactor_name
                       else "omits the nicotinamide cofactor"),
        ),
    ]
    if positive_control_enzyme_available:
        specs.append(ControlSpec(
            name="positive_enzyme", kind="positive_enzyme",
            claim=ControlClaim.ASSAY_SYSTEM_WORKS,
            rationale=("a characterised enzyme on its own known substrate; it "
                       "validates the detection chain, not this substrate"),
            requires_new_construct=positive_control_needs_new_construct,
        ))
    return specs


def validate_control_claims(specs: Sequence[ControlSpec]) -> list[str]:
    """Problems with a control plan, as sentences. Empty means it holds up.

    Three checks, each for a mistake that survives peer review because it is
    made in prose:

    1. a background or system control claiming target turnover;
    2. a target-turnover claim with no authentic-standard-backed product
       identification behind it, which is the claim that most needs one;
    3. no control claiming the assay system works, which leaves a plate of
       negatives uninterpretable -- dead reagents and inactive enzymes look
       identical.
    """
    problems: list[str] = []
    for spec in specs:
        if spec.claim.is_target_turnover:
            if spec.kind != "positive_enzyme":
                problems.append(
                    f"control '{spec.name}' is a {spec.kind} control but claims "
                    f"the target substrate is turned over; a background control "
                    f"cannot demonstrate turnover")
            if not spec.substrate_is_target:
                problems.append(
                    f"control '{spec.name}' claims target turnover but is not "
                    f"run on the target substrate")
            if not spec.product_identity_confirmed:
                problems.append(
                    f"control '{spec.name}' claims target turnover without a "
                    f"method that identifies the product; an indirect signal "
                    f"cannot establish it")
        elif spec.kind == "positive_enzyme" and spec.substrate_is_target \
                and spec.product_identity_confirmed:
            problems.append(
                f"control '{spec.name}' runs a characterised enzyme on the "
                f"target substrate with product identification, which would "
                f"demonstrate target turnover, but claims only "
                f"'{spec.claim.value}'; state the stronger claim deliberately")
    if not any(c.claim is ControlClaim.ASSAY_SYSTEM_WORKS for c in specs):
        problems.append(
            "no control demonstrates that the assay system works; a round that "
            "returns nothing will not be able to separate inactive enzymes from "
            "a dead detection chain")
    return problems


# ==========================================================================
# Measurement footprint
# ==========================================================================

@dataclass(frozen=True)
class MeasurementFootprint:
    """Genes ordered versus wells consumed, computed before the order.

    Both numbers are reported because they are paid for by different budgets
    and are routinely confused: the synthesis quote scales with
    :attr:`n_genes`, the plate, reagent and instrument time scale with
    :attr:`total_wells`, and the ratio between them is
    ``cofactor_conditions x replicates`` -- typically six.
    """

    n_candidate_genes: int
    n_variant_genes: int
    n_control_genes: int
    n_controls: int
    cofactor_conditions: int
    replicates: int
    wells_per_plate: int

    @property
    def n_genes(self) -> int:
        """Constructs that must be synthesised, controls included."""
        return self.n_candidate_genes + self.n_variant_genes + self.n_control_genes

    @property
    def n_measured_constructs(self) -> int:
        return self.n_candidate_genes + self.n_variant_genes

    @property
    def candidate_wells(self) -> int:
        return (self.n_measured_constructs * max(1, self.cofactor_conditions)
                * max(1, self.replicates))

    @property
    def control_wells_per_plate(self) -> int:
        """The control set, replicated on every plate.

        Not once per round. A background is a property of the plate it was
        read on -- the lysate batch, the reader's lamp, the hour it sat on the
        bench -- so a plate with no empty-vector well has no background, and
        every fold-over-background bar on it is undecidable rather than met.
        Laying the controls out once and letting the last plate carry them is
        how a round comes back with half its wells unusable, and the cost of
        the controls is only visible if it is counted here.
        """
        return (self.n_controls * max(1, self.cofactor_conditions)
                * max(1, self.replicates))

    @property
    def candidate_wells_per_plate(self) -> int:
        """What is left of a plate once its own controls are on it."""
        return self.wells_per_plate - self.control_wells_per_plate

    @property
    def layout_feasible(self) -> bool:
        """Whether a plate can hold its controls and at least one candidate.

        False means the control set at this replication fills a plate on its
        own. That is a design decision to take deliberately -- fewer controls,
        fewer replicates, fewer cofactor conditions, or a bigger plate -- and
        not something to resolve by dropping controls from some plates.
        """
        return self.candidate_wells_per_plate > 0

    @property
    def plates(self) -> int:
        """Plates needed with each plate carrying its own controls.

        ``0`` when :attr:`layout_feasible` is false: there is no plate count
        that satisfies the design, and returning a number anyway would hide
        that behind an arithmetic answer.
        """
        if not self.layout_feasible:
            return 0
        if self.candidate_wells == 0:
            return 1 if self.control_wells_per_plate else 0
        per = self.candidate_wells_per_plate
        return (self.candidate_wells + per - 1) // per

    @property
    def control_wells(self) -> int:
        return self.plates * self.control_wells_per_plate

    @property
    def total_wells(self) -> int:
        return self.candidate_wells + self.control_wells

    @property
    def control_overhead_fraction(self) -> float | None:
        """Share of the round's wells spent on controls, reported not judged.

        No threshold: whether a quarter of the plate on controls is too much
        depends on how expensive the round is and on how much the controls
        are being asked to establish. The number is surfaced so the choice is
        made with it in view.
        """
        if not self.total_wells:
            return None
        return self.control_wells / self.total_wells

    def describe(self) -> str:
        if not self.layout_feasible:
            return (
                f"{self.n_controls} control(s) at "
                f"{self.cofactor_conditions} cofactor condition(s) x "
                f"{self.replicates} replicate(s) is "
                f"{self.control_wells_per_plate} well(s) per plate, which "
                f"fills a plate of {self.wells_per_plate} on its own. No "
                f"layout carries both the controls and a candidate: reduce "
                f"the control replication or use a larger plate."
            )
        overhead = self.control_overhead_fraction
        return (
            f"{self.n_genes} gene(s) to synthesise "
            f"({self.n_candidate_genes} mined candidate(s), "
            f"{self.n_variant_genes} variant(s), "
            f"{self.n_control_genes} control construct(s)) -> "
            f"{self.total_wells} well(s) "
            f"({self.candidate_wells} candidate + {self.control_wells} "
            f"control) = {self.plates} plate(s) of {self.wells_per_plate}, at "
            f"{self.cofactor_conditions} cofactor condition(s) x "
            f"{self.replicates} replicate(s). Genes are not wells: the "
            f"multiplier is "
            f"{max(1, self.cofactor_conditions) * max(1, self.replicates)}x. "
            f"The control set is repeated on every plate "
            f"({self.control_wells_per_plate} well(s) each, "
            f"{overhead:.0%} of the round) because a background belongs to "
            f"the plate it was read on."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_genes": self.n_genes,
            "n_candidate_genes": self.n_candidate_genes,
            "n_variant_genes": self.n_variant_genes,
            "n_control_genes": self.n_control_genes,
            "n_controls_measured": self.n_controls,
            "cofactor_conditions": self.cofactor_conditions,
            "replicates": self.replicates,
            "candidate_wells": self.candidate_wells,
            "control_wells": self.control_wells,
            "control_wells_per_plate": self.control_wells_per_plate,
            "candidate_wells_per_plate": self.candidate_wells_per_plate,
            "control_overhead_fraction": self.control_overhead_fraction,
            "total_wells": self.total_wells,
            "wells_per_plate": self.wells_per_plate,
            "plates": self.plates,
            "layout_feasible": self.layout_feasible,
            "note": self.describe(),
        }


def measurement_footprint(
    plan: BatchPlan, *, n_variant_genes: int = 0,
    wells_per_plate: int = DEFAULT_WELLS_PER_PLATE,
) -> MeasurementFootprint:
    """Footprint of a composed plan, counting controls that need their own gene."""
    control_genes = sum(1 for c in plan.controls if c.requires_new_construct)
    n_candidates = plan.n_candidates - n_variant_genes
    return MeasurementFootprint(
        n_candidate_genes=max(0, n_candidates),
        n_variant_genes=n_variant_genes,
        n_control_genes=control_genes,
        n_controls=len(plan.controls),
        cofactor_conditions=plan.cofactor_conditions,
        replicates=plan.replicates,
        wells_per_plate=wells_per_plate,
    )


def _yes_no(value: bool | None) -> str:
    """``yes``/``no``/``unknown`` for a plan cell, never a bare blank.

    A blank in a plan column would look exactly like a measured column nobody
    filled in, which is the confusion these columns exist to end.
    """
    if value is None:
        return "unknown"
    return "yes" if value else "no"


def well_label(position: int, wells_per_plate: int = DEFAULT_WELLS_PER_PLATE) -> str:
    """``A1`` ... ``H12`` for a 0-based position inside a plate.

    Row-major, 12 columns for a 96-well plate and 24 for a 384. Returned as a
    label rather than an index because a result file that comes back from a
    plate reader is keyed on the label, and reconstructing it later from a
    slot number is where plate maps get transposed.
    """
    if position < 0:
        raise ValueError("well position must be >= 0")
    columns = 24 if wells_per_plate > 96 else 12
    within = position % max(1, wells_per_plate)
    row = within // columns
    col = within % columns
    return f"{chr(ord('A') + row)}{col + 1}"


# ==========================================================================
# Plate layout
# ==========================================================================

#: Name of the layout algorithm, written into the plan and the provenance.
#: Versioned because a layout is only reproducible if the recipe that
#: produced it is identified as precisely as the seed is.
LAYOUT_STRATEGY: str = "stratified-random-v1"


@dataclass(frozen=True)
class WellAssignment:
    """One well: what is in it, where it is, and which stratum it came from."""

    plate: int
    well: str
    position: int                   # 0-based index within the plate
    slot: int
    identifier: str
    kind: str                       # candidate | control
    role: str
    stratum: str
    cofactor_index: int
    replicate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "plate": self.plate, "well": self.well, "position": self.position,
            "slot": self.slot, "identifier": self.identifier,
            "kind": self.kind, "role": self.role, "stratum": self.stratum,
            "cofactor_index": self.cofactor_index, "replicate": self.replicate,
        }


@dataclass(frozen=True)
class PlateLayout:
    """Where every measurement sits, and the seed that put it there.

    WHY THE LAYOUT IS NOT THE SELECTION ORDER
    =========================================
    Filling wells in the order candidates were ranked makes plate position a
    function of rank. Microplates have real position effects -- edge wells
    evaporate faster, the reader's optics are not uniform, a thermal gradient
    runs across the block -- so a layout that follows the ranking confounds
    "ranked highly" with "sat in the middle of plate 1". The round then
    confirms its own prior, and nothing in the data can show that it did.

    Cheap to avoid, and only avoidable before the plate is run: randomise
    position within each plate, and spread each construct's replicates across
    plates so that a plate effect lands inside a construct as noise rather
    than between constructs as a difference.

    STRATIFIED, NOT PLAIN RANDOM
    ============================
    An unstratified shuffle leaves plate composition to luck, and with a
    handful of families and a handful of plates the luck is often bad enough
    to put a whole family on one plate. Dealing each stratum round-robin
    across plates keeps the families and roles balanced, so a per-plate
    comparison stays meaningful.

    REPRODUCIBLE
    ============
    :attr:`seed` and :attr:`strategy` are written into the plan and into
    provenance. A randomised layout nobody can reconstruct is worse than a
    fixed one: the plate map becomes unverifiable, and a transposition in the
    returned sheet cannot be detected by re-deriving the map.
    """

    assignments: tuple[WellAssignment, ...]
    plates: int
    wells_per_plate: int
    seed: int
    strategy: str = LAYOUT_STRATEGY
    strata: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def by_plate(self) -> dict[int, list[WellAssignment]]:
        out: dict[int, list[WellAssignment]] = {}
        for a in self.assignments:
            out.setdefault(a.plate, []).append(a)
        for plate in out:
            out[plate].sort(key=lambda a: a.position)
        return out

    def controls_on_every_plate(self) -> bool:
        """Whether each plate carries at least one well of each control."""
        names = {a.identifier for a in self.assignments if a.kind == "control"}
        if not names:
            return False
        for wells in self.by_plate().values():
            present = {a.identifier for a in wells if a.kind == "control"}
            if present != names:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "plates": self.plates,
            "wells_per_plate": self.wells_per_plate,
            "strata": list(self.strata),
            "controls_on_every_plate": self.controls_on_every_plate(),
            "n_wells": len(self.assignments),
            "notes": list(self.notes),
        }


def plan_plate_layout(
    plan: BatchPlan, footprint: MeasurementFootprint, *, seed: int,
) -> PlateLayout:
    """Assign every measurement to a plate and a well.

    Controls go onto every plate, at the plan's replication and under every
    cofactor condition, because that is what makes a per-plate background
    exist. Candidate wells are dealt round-robin across the plates within
    each stratum, so a construct's replicates do not sit together and the
    families stay balanced. Positions within a plate are then shuffled with
    the given seed.

    Raises
    ------
    ValueError
        When the control set fills a plate on its own. There is no layout in
        that case, and silently dropping controls from some plates would
        produce one that cannot be analysed per plate.
    """
    if not footprint.layout_feasible:
        raise ValueError(footprint.describe())

    conditions = max(1, plan.cofactor_conditions)
    replicates = max(1, plan.replicates)
    plates = max(1, footprint.plates)
    rng = random.Random(seed)
    notes: list[str] = []

    # -- candidate wells, dealt across plates within each stratum ---------
    strata: dict[str, list[tuple[Any, int, int]]] = {}
    for member in plan.members:
        key = f"{member.role.value}|{member.family or 'unassigned'}"
        for condition in range(conditions):
            for replicate in range(1, replicates + 1):
                strata.setdefault(key, []).append((member, condition, replicate))

    capacity = {plate: footprint.candidate_wells_per_plate
                for plate in range(1, plates + 1)}
    dealt: dict[int, list[tuple[Any, int, int, str]]] = {
        plate: [] for plate in range(1, plates + 1)}
    # Each stratum starts its round-robin one plate further along, so the
    # first plate does not collect the first unit of every stratum.
    for offset, key in enumerate(sorted(strata)):
        units = list(strata[key])
        rng.shuffle(units)
        cursor = offset % plates
        for unit in units:
            for _ in range(plates):
                plate = 1 + (cursor % plates)
                cursor += 1
                if capacity[plate] > 0:
                    capacity[plate] -= 1
                    dealt[plate].append(unit + (key,))
                    break
            else:                                   # pragma: no cover
                raise ValueError(
                    "the plates are full before every candidate well was "
                    "placed; the footprint and the layout disagree")

    # -- control wells, repeated on every plate ---------------------------
    control_units: dict[int, list[tuple[Any, int, int, str]]] = {}
    for plate in range(1, plates + 1):
        for control in plan.controls:
            for condition in range(conditions):
                for replicate in range(1, replicates + 1):
                    control_units.setdefault(plate, []).append(
                        (control, condition, replicate, "control"))

    # -- positions within each plate, shuffled ----------------------------
    assignments: list[WellAssignment] = []
    slots: dict[str, int] = {}
    for n, member in enumerate(plan.members, start=1):
        slots[member.candidate_id] = n
    for n, control in enumerate(plan.controls, start=len(plan.members) + 1):
        slots[control.name] = n

    for plate in range(1, plates + 1):
        units = dealt.get(plate, []) + control_units.get(plate, [])
        positions = list(range(footprint.wells_per_plate))
        rng.shuffle(positions)
        for position, (item, condition, replicate, stratum) in zip(positions, units):
            is_control = stratum == "control"
            identifier = item.name if is_control else item.candidate_id
            assignments.append(WellAssignment(
                plate=plate,
                well=well_label(position, footprint.wells_per_plate),
                position=position,
                slot=slots.get(identifier, 0),
                identifier=identifier,
                kind="control" if is_control else "candidate",
                role=(f"{item.kind} control" if is_control
                      else item.role.value),
                stratum=stratum,
                cofactor_index=condition,
                replicate=replicate))

    assignments.sort(key=lambda a: (a.plate, a.position))
    if plates > 1:
        notes.append(
            f"each construct's {replicates} replicate(s) are dealt across "
            f"{plates} plates, so a plate effect falls inside a construct as "
            f"noise rather than between constructs as a difference")
    notes.append(
        f"the control set is repeated on every plate "
        f"({footprint.control_wells_per_plate} well(s) each), so every plate "
        f"carries its own background")
    notes.append(
        f"well positions are shuffled with seed {seed} under "
        f"{LAYOUT_STRATEGY}, so plate position does not follow selection rank")
    return PlateLayout(
        assignments=tuple(assignments), plates=plates,
        wells_per_plate=footprint.wells_per_plate, seed=seed,
        strata=tuple(sorted(strata)), notes=tuple(notes))


# ==========================================================================
# Variant groups
# ==========================================================================

@dataclass(frozen=True)
class VariantSelection:
    """Variant proposals that fit the gene budget, as whole attributable groups."""

    selected: tuple[MutationProposal, ...]
    dropped: tuple[MutationProposal, ...]
    reason: str

    @property
    def n_selected(self) -> int:
        return len(self.selected)


def select_variant_groups(
    proposals: Sequence[MutationProposal], slots: int
) -> VariantSelection:
    """Admit variants in attributable groups, never a combination on its own.

    A combination variant is only interpretable alongside its single-mutant
    decomposition controls, so the unit of admission is the group: the
    combination plus every proposal id it names in
    ``decomposition_controls``. A group that does not fit in the remaining
    slots is dropped whole. Taking the combination and leaving the singles
    would fill the plate and produce a result nobody can attribute, which is
    the expensive version of running out of room.

    Groups are considered in ``experimental_priority`` order, then by id, so
    the selection is deterministic.
    """
    if slots < 0:
        raise ValueError(f"slots must be >= 0, got {slots}")
    by_id = {p.proposal_id: p for p in proposals}
    ordered = sorted(proposals, key=lambda p: (p.experimental_priority,
                                               p.is_combination, p.proposal_id))
    selected: dict[str, MutationProposal] = {}
    dropped: list[MutationProposal] = []
    missing_controls: list[str] = []

    for proposal in ordered:
        if proposal.proposal_id in selected:
            continue
        group = [proposal]
        incomplete = False
        for control_id in proposal.decomposition_controls:
            control = by_id.get(control_id)
            if control is None:
                incomplete = True
                missing_controls.append(
                    f"{proposal.proposal_id} names control {control_id}, which "
                    f"is not among the supplied proposals")
                break
            if control.proposal_id not in selected:
                group.append(control)
        if incomplete:
            dropped.append(proposal)
            continue
        new = [g for g in group if g.proposal_id not in selected]
        if len(selected) + len(new) > slots:
            dropped.extend(new)
            continue
        for member in new:
            selected[member.proposal_id] = member

    parts = [f"{len(selected)} variant gene(s) admitted into {slots} slot(s) as "
             f"{'complete' if selected else 'no'} attributable group(s)"]
    if dropped:
        parts.append(
            f"{len(dropped)} proposal(s) dropped whole rather than split from "
            f"their single-mutant controls")
    if missing_controls:
        parts.append("; ".join(missing_controls))
    return VariantSelection(tuple(selected.values()), tuple(dropped),
                            ". ".join(parts))


# ==========================================================================
# The interface
# ==========================================================================

class SelectBatch(ScientificInterface):
    """Compose the round, price it in wells, and write the order form.

    Responsibilities, each enforced rather than described:

    * gated on ``synthesis_authorized`` -- this is the step that spends money;
    * control genes reserved from the construct cap before candidates are
      chosen;
    * role split (high evidence / diversity / uncertainty probe) delegated to
      :func:`~eagent.science.diversity.compose_batch`, which refuses to pad;
    * the measurement footprint reported in wells and plates, next to the gene
      count;
    * the control plan's claims checked against :class:`ControlClaim`;
    * the pre-registered positive criterion written into the plan file before
      any data exist.
    """

    name: ClassVar[str] = "select_batch"
    description: ClassVar[str] = (
        "Compose a round under the construct budget: three candidate roles, "
        "family and clade quotas, attributable variant groups, a claim-checked "
        "control plan, and the plate cost made visible before the order."
    )
    required_fields: ClassVar[tuple[str, ...]] = (
        "reaction.substrate.isomeric_smiles",
        "reaction.product.isomeric_smiles",
        "conditions.pH",
        "conditions.temperature_C",
        "conditions.expression_host",
    )
    required_approvals: ClassVar[tuple[str, ...]] = (SYNTHESIS_GATE,)
    depends_on: ClassVar[tuple[str, ...]] = ("evaluate_catalysis",)
    version: ClassVar[str] = "0.1.0"

    def execute(
        self,
        ctx: RunContext,
        *,
        candidates: Sequence[Candidate] | Sequence[Mapping[str, Any]]
        | Mapping[str, Any] | None = None,
        variant_proposals: Sequence[MutationProposal] = (),
        assay_template: AssayTemplate | None = None,
        controls: Sequence[ControlSpec] | None = None,
        positive_control_enzyme_available: bool = False,
        positive_control_needs_new_construct: bool = True,
        role_targets: Mapping[BatchRole, int] | None = None,
        family_quotas: Mapping[str, int] | None = None,
        cluster_cap: int | None = None,
        extra_pocket_residues: Mapping[str, Sequence[str]] | None = None,
        order_of_dimensions: Sequence[str] | None = None,
        lambda_weight: float = DEFAULT_LAMBDA_WEIGHT,
        reallocate_unfilled_roles: bool = False,
        cofactor_conditions: int | None = None,
        replicates: int | None = None,
        wells_per_plate: int = DEFAULT_WELLS_PER_PLATE,
        round_number: int = 1,
        plan_id: str | None = None,
        submit_to: str | None = None,
        **_: Any,
    ) -> ToolResult:
        """Compose the batch and write the three artifacts.

        ``candidates`` may arrive as models or as the serialised mapping an
        earlier step published; it is coerced here so a dict never reaches
        :func:`~eagent.science.diversity.compose_batch`, where it would fail
        as a missing attribute with no indication of which step produced it.
        """
        refusal = self._refuse_external_submission(ctx, submit_to)
        if refusal is not None:
            return refusal
        candidates = as_candidates(candidates, source=self.name)
        if not candidates and not variant_proposals:
            return ToolResult.failure(
                self.name,
                "nothing to select from: pass candidates=[Candidate, ...] "
                "and/or variant_proposals=[MutationProposal, ...]. An empty "
                "batch is not a plan.",
                code="empty_pool",
            )

        result = ToolResult(status=Status.SUCCESS)
        task = ctx.task
        budget: Budget = task.budget
        plan_id = plan_id or f"{task.task_id}-round-{round_number}"

        n_cofactor, cofactor_note = self._cofactor_conditions(
            task.conditions.cofactor_options, cofactor_conditions)
        if cofactor_note:
            result.add_flag("cofactor_conditions_assumed", Severity.WARN,
                            cofactor_note)
        n_replicates, replicate_note = self._replicates(assay_template, replicates)
        if replicate_note:
            result.add_flag("replicates_assumed", Severity.WARN, replicate_note)

        # -- controls, and their claims ------------------------------------
        cofactor_name = (task.conditions.cofactor_options[0].name
                         if task.conditions.cofactor_options else None)
        specs = list(controls) if controls is not None else default_control_plan(
            positive_control_enzyme_available=positive_control_enzyme_available,
            positive_control_needs_new_construct=positive_control_needs_new_construct,
            cofactor_name=cofactor_name,
        )
        for problem in validate_control_claims(specs):
            result.add_flag("control_claim_problem", Severity.BLOCKER, problem)
        control_items = [s.to_control_item() for s in specs]

        # -- gene budget: controls first, then variants, then candidates ----
        try:
            reservation = reserve_control_slots(budget, control_items)
        except ValueError as exc:
            return ToolResult.failure(self.name, str(exc),
                                      code="control_budget_overflow")
        if reservation.note:
            result.add_flag("control_slot_reservation", Severity.INFO,
                            reservation.note)
        candidate_slots = reservation.candidate_slots
        control_items = list(reservation.controls)

        variants = select_variant_groups(variant_proposals, candidate_slots)
        mining_slots = max(0, candidate_slots - variants.n_selected)
        if variants.dropped:
            result.add_flag(
                "variant_groups_dropped", Severity.WARN, variants.reason)

        # -- mined candidates ----------------------------------------------
        base_members: list[BatchMember] = []
        compose_shortfall: str | None = None
        used_quotas: dict[str, int] = dict(family_quotas or {})
        if candidates and mining_slots > 0:
            mining_budget = Budget(**{
                **reservation.budget.model_dump(),
                "new_constructs_round_1": mining_slots,
                "constructs_include_controls": False,
                "reserved_control_slots": 0,
            })
            try:
                base_plan = compose_batch(
                    candidates, mining_budget, role_targets or DEFAULT_ROLE_TARGETS,
                    family_quotas, controls=(), plan_id=plan_id,
                    round_number=round_number, cofactor_conditions=n_cofactor,
                    replicates=n_replicates, cluster_cap=cluster_cap,
                    extra_pocket_residues=extra_pocket_residues,
                    order_of_dimensions=order_of_dimensions,
                    lambda_weight=lambda_weight,
                    reallocate_unfilled_roles=reallocate_unfilled_roles,
                )
            except ValueError as exc:
                return ToolResult.failure(
                    self.name,
                    f"batch composition refused the pool: {exc}",
                    code="pool_not_composable")
            base_members = list(base_plan.members)
            compose_shortfall = base_plan.shortfall_reason
            used_quotas = dict(base_plan.family_quota)
        elif candidates and mining_slots == 0:
            compose_shortfall = (
                f"0 of the {candidate_slots} candidate slot(s) were available to "
                f"mined candidates: {variants.n_selected} went to round-2 "
                f"variant groups.")

        members = self._renumber(base_members, variants.selected)
        plan = self._build_plan(
            plan_id=plan_id, round_number=round_number, members=members,
            controls=control_items, cofactor_conditions=n_cofactor,
            replicates=n_replicates, requested_slots=candidate_slots,
            family_quota=used_quotas,
            compose_shortfall=compose_shortfall, variants=variants,
            n_candidates_offered=len(candidates or ()),
        )
        footprint = measurement_footprint(
            plan, n_variant_genes=variants.n_selected,
            wells_per_plate=wells_per_plate)

        if plan.is_short:
            result.add_flag(
                "batch_short", Severity.WARN,
                f"{plan.n_candidates} of {plan.requested_slots} slot(s) filled. "
                f"{plan.shortfall_reason}")
        if plan.n_candidates == 0:
            result.add_flag(
                "no_members", Severity.BLOCKER,
                "no candidate or variant qualified for a slot; there is nothing "
                "to order. The pool, not the plate, is the problem.")

        if not footprint.layout_feasible:
            return ToolResult.failure(
                self.name, footprint.describe(), code="layout_infeasible")

        # The layout is randomised, so it needs a seed that is recorded and a
        # strategy that is named; both go into the plan and into provenance.
        # A plate map nobody can re-derive cannot be checked against the
        # sheet that comes back.
        layout_seed = ctx.seed_for(f"{self.name}:plate_layout")
        layout = plan_plate_layout(plan, footprint, seed=layout_seed)
        if plan.controls and not layout.controls_on_every_plate():
            result.add_flag(
                "controls_missing_from_a_plate", Severity.BLOCKER,
                "a plate was laid out without the full control set; every "
                "fold-over-background bar on it would be undecidable",
                subject="layout")
        for note in layout.notes:
            result.add_flag("plate_layout", Severity.INFO, note,
                            subject="layout")

        # -- artifacts ------------------------------------------------------
        batch_path = self._write_batch_csv(ctx, plan, variants)
        plan_path = self._write_plan_yaml(
            ctx, plan, footprint, specs, assay_template, variants, task,
            wells_per_plate, layout)
        template_path = self._write_results_template(
            ctx, plan, assay_template, task, layout, variants)

        result.artifacts.append(Artifact(
            key="selected_batch", path=str(batch_path), kind="table",
            sha256=sha256_file(batch_path), n_records=len(plan.members),
            summary=("the order form: one row per gene, with the role it fills "
                     "and the reason the slot was spent"),
        ))
        result.artifacts.append(Artifact(
            key="experiment_plan", path=str(plan_path), kind="file",
            sha256=sha256_file(plan_path),
            summary=("roles, quotas, control claims, the measurement footprint "
                     "in wells and plates, and the pre-registered positive "
                     "criterion recorded before any data exist"),
        ))
        result.artifacts.append(Artifact(
            key="assay_results_template", path=str(template_path), kind="table",
            sha256=sha256_file(template_path), n_records=footprint.total_wells,
            summary=("one row per well, with the columns ingest_results reads; "
                     "deliberately no 'hit' column"),
        ))

        result.data.update({
            CANDIDATES_KEY: serialise_candidates(candidates),
            "plan": plan.model_dump(mode="json"),
            "footprint": footprint.to_dict(),
            "role_counts": plan.role_counts(),
            "n_variant_genes": variants.n_selected,
            "variant_proposal_ids": [p.proposal_id for p in variants.selected],
            "dropped_variant_ids": [p.proposal_id for p in variants.dropped],
            "control_claims": [
                {"name": s.name, "kind": s.kind, "claim": s.claim.value,
                 "requires_new_construct": s.requires_new_construct}
                for s in specs
            ],
            "assay_results_columns": list(ASSAY_RESULT_COLUMNS),
        })

        criterion = dict(assay_template.positive_criteria) if assay_template else {}
        if not criterion:
            result.add_flag(
                "no_pre_registered_criterion", Severity.BLOCKER,
                "no AssayTemplate with positive_criteria was supplied, so the "
                "round has no pre-registered endpoint. Without one, whatever "
                "the data look like will become the criterion.")
            result.add_next("confirm_functional_criteria",
                            "Fix the hit definition before the plate is run",
                            {"gate": "functional_criteria_confirmed"},
                            requires_human=True)
        result.data["positive_criteria"] = criterion
        result.data["positive_criteria_sha256"] = sha256_obj(criterion)

        result.add_uncertainty(
            "role_split_is_a_choice",
            "Is this the right split between paying wells and informative "
            "wells? The 48/24/24 default is a stated starting configuration, "
            "not a validated optimum; round 2 should revise it against round-1 "
            "outcomes.",
            affects=[plan.plan_id], resolvable_by="round-1 outcomes")
        if footprint.plates > 1:
            result.add_uncertainty(
                "plate_effects",
                f"The round spans {footprint.plates} plates. Are controls "
                f"replicated on every plate, and is the plate assignment "
                f"randomised with respect to family? Otherwise a plate effect "
                f"is confounded with a family effect.",
                affects=[plan.plan_id], resolvable_by="plate layout decision")
        result.add_next(
            "ingest_results",
            "Return the filled template; the pre-registered criterion recorded "
            "in experiment_plan.yaml is the only one that will be applied",
            {"template": str(template_path),
             "positive_criteria_sha256": result.data["positive_criteria_sha256"]},
            requires_human=True)

        result.provenance = Provenance(
            tool=self.name, tool_version=self.version,
            inputs_sha256={
                "candidates": sha256_obj([c.candidate_id for c in (candidates or ())]),
                "variant_proposals": sha256_obj(
                    [p.proposal_id for p in variant_proposals]),
                "budget": sha256_obj(budget.model_dump(mode="json")),
                "positive_criteria": result.data["positive_criteria_sha256"],
            },
            databases={},
            models={},
            parameters={
                "plan_id": plan_id,
                "round_number": round_number,
                "role_targets": {k.value: v for k, v in
                                 (role_targets or DEFAULT_ROLE_TARGETS).items()},
                "family_quotas": dict(family_quotas or {}),
                "cluster_cap": cluster_cap,
                "lambda_weight": lambda_weight,
                "reallocate_unfilled_roles": reallocate_unfilled_roles,
                "cofactor_conditions": n_cofactor,
                "replicates": n_replicates,
                "wells_per_plate": wells_per_plate,
                "candidate_slots": candidate_slots,
                "reserved_control_slots": reservation.reserved_slots,
                "assay_template_id": (assay_template.template_id
                                      if assay_template else None),
                "allow_network": ctx.policy.allow_network,
            },
            random_seed=ctx.seed_for(self.name),
        )

        # A short batch is PARTIAL, not SUCCESS: the plan is usable and the
        # arithmetic is honest, but the round it describes is not the round
        # that was asked for, and the controller has to see that rather than
        # read a green status and place the order.
        if result.blockers:
            result.status = Status.PARTIAL
            result.message = (
                f"batch composed with {len(result.blockers)} blocking problem(s); "
                f"{footprint.describe()}")
        elif plan.is_short:
            result.status = Status.PARTIAL
            result.message = (
                f"batch is short ({plan.n_candidates} of "
                f"{plan.requested_slots} slot(s)) and was not padded; "
                f"{footprint.describe()}")
        else:
            result.message = footprint.describe()
        return result

    # -- guards ------------------------------------------------------------
    def _refuse_external_submission(
        self, ctx: RunContext, submit_to: str | None
    ) -> ToolResult | None:
        """Refuse to send the selected sequences anywhere.

        A selected batch is a list of unpublished sequences about to be
        ordered, which is the most disclosure-sensitive object the pipeline
        produces. Synthesis itself is an operator transaction conducted
        outside this interface, deliberately: automating it would make a
        disclosure decision implicit in a tool call.
        """
        if not submit_to:
            return None
        return ToolResult.failure(
            self.name,
            f"refused to transmit the selected batch to '{submit_to}'. Placing "
            f"a synthesis order discloses every sequence in it and is an "
            f"operator transaction, not a tool call"
            + ("" if ctx.policy.allow_network
               else " (the run policy also has allow_network=False)"),
            code="external_submission_refused",
        )

    # -- inputs -------------------------------------------------------------
    @staticmethod
    def _cofactor_conditions(
        options: Sequence[CofactorSpec], override: int | None
    ) -> tuple[int, str]:
        """Number of cofactor conditions, preferring the task's own list.

        An NADPH-preferring ketoreductase screened only with NADH returns a
        true negative that reads as "this enzyme does not do the chemistry",
        so the condition count is a scientific decision and comes from the
        task where it exists.
        """
        if override is not None:
            if override < 1:
                raise ValueError("cofactor_conditions must be >= 1")
            if options and override != len(options):
                return override, (
                    f"cofactor_conditions={override} overrides the "
                    f"{len(options)} cofactor option(s) in the task; the plate "
                    f"will not cover every declared condition")
            return override, ""
        if options:
            return len(options), ""
        return DEFAULT_COFACTOR_CONDITIONS, (
            f"the task lists no cofactor options, so the footprint assumes "
            f"{DEFAULT_COFACTOR_CONDITIONS} condition(s). Resolve "
            f"conditions.cofactor_options: a cofactor mismatch is one of the "
            f"five diagnoses a no-hit round has to rule out.")

    @staticmethod
    def _replicates(template: AssayTemplate | None,
                    override: int | None) -> tuple[int, str]:
        """Replicates, preferring the assay template's pre-registered value."""
        if override is not None:
            if override < 1:
                raise ValueError("replicates must be >= 1")
            if template is not None and override != template.replicates:
                return override, (
                    f"replicates={override} overrides the assay template's "
                    f"pre-registered {template.replicates}; the change is "
                    f"recorded in provenance")
            return override, ""
        if template is not None:
            return template.replicates, ""
        return 3, ("no AssayTemplate supplied, so 3 replicates are assumed; the "
                   "replicate count belongs to the pre-registered protocol")

    # -- plan assembly ------------------------------------------------------
    @staticmethod
    def _renumber(mined: Sequence[BatchMember],
                  variants: Sequence[MutationProposal]) -> list[BatchMember]:
        """Concatenate mined members and variant members into one slot series."""
        out: list[BatchMember] = []
        for member in mined:
            out.append(member.model_copy(update={"slot": len(out) + 1}))
        for proposal in variants:
            out.append(BatchMember(
                slot=len(out) + 1,
                candidate_id=proposal.proposal_id,
                role=BatchRole.VARIANT,
                family=None,
                sequence_cluster_id=None,
                utility=None,
                selection_reason=(
                    f"round-2 variant of {proposal.parent_candidate_id} at "
                    f"priority {proposal.experimental_priority}; "
                    + ("admitted with its single-mutant decomposition controls"
                       if proposal.is_combination else
                       "single mutant, also serving as a decomposition control")),
                construct_notes=(
                    f"{proposal.label()} in {proposal.numbering_reference}; "
                    f"parent {proposal.parent_sequence_sha256[:19]}"),
            ))
        return out

    @staticmethod
    def _build_plan(
        *, plan_id: str, round_number: int, members: Sequence[BatchMember],
        controls: Sequence[ControlItem], cofactor_conditions: int,
        replicates: int, requested_slots: int, family_quota: Mapping[str, int],
        compose_shortfall: str | None, variants: VariantSelection,
        n_candidates_offered: int,
    ) -> BatchPlan:
        """Build the :class:`BatchPlan`, composing an honest shortfall reason.

        ``BatchPlan`` refuses a short batch with no reason, which is the point:
        the only way past the validator is to say why, and the only thing the
        reason is allowed to be is arithmetic.
        """
        n_members = len([m for m in members if m.role is not BatchRole.CONTROL])
        reason: str | None = None
        if n_members < requested_slots:
            parts = [
                f"{n_members} of {requested_slots} candidate slot(s) filled "
                f"({variants.n_selected} variant gene(s), "
                f"{n_members - variants.n_selected} mined candidate(s), from a "
                f"pool of {n_candidates_offered} mined candidate(s) offered)."
            ]
            if compose_shortfall:
                parts.append(compose_shortfall)
            if variants.reason:
                parts.append(variants.reason + ".")
            parts.append(
                "The batch is reported short rather than padded: no candidate "
                "was admitted by relaxing a gate, a family quota or a cluster cap.")
            reason = " ".join(parts)
        return BatchPlan(
            plan_id=plan_id, round_number=round_number, members=list(members),
            controls=list(controls), cofactor_conditions=cofactor_conditions,
            replicates=replicates, requested_slots=requested_slots,
            family_quota=dict(family_quota), shortfall_reason=reason,
        )

    # -- artifacts ----------------------------------------------------------
    def _write_batch_csv(self, ctx: RunContext, plan: BatchPlan,
                         variants: VariantSelection) -> Path:
        """Write ``selected_batch_<n>.csv``, named from the count actually ordered.

        The filename carries the real number. A file called
        ``selected_batch_96.csv`` holding 71 rows is the kind of artefact that
        gets quoted as "we screened 96" two meetings later.
        """
        by_id = {p.proposal_id: p for p in variants.selected}
        path = ctx.path("select_batch",
                        f"selected_batch_{plan.n_candidates}.csv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(BATCH_CSV_COLUMNS)
            for member in plan.members:
                proposal = by_id.get(member.candidate_id)
                writer.writerow([
                    member.slot,
                    "variant" if proposal is not None else "candidate",
                    member.candidate_id, member.role.value,
                    member.family or "", member.sequence_cluster_id or "",
                    "" if member.utility is None else f"{member.utility:.4f}",
                    member.selection_reason,
                    proposal.parent_candidate_id if proposal else "",
                    proposal.label() if proposal else "",
                    proposal.numbering_reference if proposal else "",
                    ";".join(proposal.decomposition_controls) if proposal else "",
                    "", "",
                ])
            slot = len(plan.members)
            for control in plan.controls:
                slot += 1
                writer.writerow([
                    slot, "control", control.name, BatchRole.CONTROL.value,
                    "", "", "", f"{control.kind} control", "", "", "", "",
                    control.requires_new_construct, control.demonstrates,
                ])
        return path

    def _write_plan_yaml(
        self, ctx: RunContext, plan: BatchPlan, footprint: MeasurementFootprint,
        specs: Sequence[ControlSpec], assay_template: AssayTemplate | None,
        variants: VariantSelection, task: Any, wells_per_plate: int,
        layout: PlateLayout,
    ) -> Path:
        """Write ``experiment_plan.yaml``, including the pre-registered endpoint."""
        criterion = dict(assay_template.positive_criteria) if assay_template else {}
        document: dict[str, Any] = {
            "plan_id": plan.plan_id,
            "task_id": task.task_id,
            "round_number": plan.round_number,
            "composed_by": self.name,
            "candidates": {
                "requested_slots": plan.requested_slots,
                "selected": plan.n_candidates,
                "role_counts": plan.role_counts(),
                "family_quota": plan.family_quota,
                "shortfall_reason": plan.shortfall_reason,
            },
            "variants": {
                "selected": [p.proposal_id for p in variants.selected],
                "dropped": [p.proposal_id for p in variants.dropped],
                "admission_rule": (
                    "a combination variant is admitted only together with every "
                    "single-mutant decomposition control it names; a group that "
                    "does not fit is dropped whole"),
                "note": variants.reason,
            },
            "controls": [
                {"name": s.name, "kind": s.kind, "claim": s.claim.value,
                 "demonstrates": s.claim.statement(),
                 "requires_new_construct": s.requires_new_construct,
                 "rationale": s.rationale}
                for s in specs
            ],
            "control_claim_separation": (
                "'the assay system works' and 'the target substrate is turned "
                "over' are different claims. No control in this plan asserts "
                "the second; that is what the round is for."),
            "measurement_footprint": footprint.to_dict(),
            "plate_layout": {
                **layout.to_dict(),
                "rationale": (
                    "Filling wells in ranking order makes plate position a "
                    "function of rank, and microplates have real position "
                    "effects. Positions are shuffled under a recorded seed, "
                    "each stratum is dealt across the plates so the families "
                    "and roles stay balanced, and the control set is repeated "
                    "on every plate so every plate has its own background."),
                "wells": [a.to_dict() for a in layout.assignments],
            },
            "conditions": {
                "cofactor_conditions": plan.cofactor_conditions,
                "cofactor_options": [c.describe()
                                     for c in task.conditions.cofactor_options],
                "replicates": plan.replicates,
                "pH": task.conditions.pH,
                "temperature_C": task.conditions.temperature_C,
                "expression_host": task.conditions.expression_host,
                "wells_per_plate": wells_per_plate,
            },
            "pre_registered_endpoint": {
                "assay_template_id": (assay_template.template_id
                                      if assay_template else None),
                "method": assay_template.method if assay_template else None,
                "tier": assay_template.tier if assay_template else None,
                "confirms_product_identity": (
                    assay_template.confirms_product_identity
                    if assay_template else None),
                "chiral_capable": (assay_template.chiral_capable
                                   if assay_template else None),
                "limit_of_detection": (assay_template.limit_of_detection
                                       if assay_template else None),
                "limit_unit": (assay_template.limit_unit
                               if assay_template else None),
                "positive_criteria": criterion,
                "positive_criteria_sha256": sha256_obj(criterion),
                "binding_statement": (
                    "This is the pre-registration. ingest_results applies this "
                    "criterion and no other; any attempt to supply a different "
                    "one after the data arrive is recorded as a protocol "
                    "deviation and refused."),
            },
            "results_template_columns": list(ASSAY_RESULT_COLUMNS),
        }
        path = ctx.path("select_batch", "experiment_plan.yaml")
        path.write_text(
            yaml.safe_dump(document, sort_keys=False, allow_unicode=True,
                           default_flow_style=False),
            encoding="utf-8")
        return path

    def _write_results_template(
        self, ctx: RunContext, plan: BatchPlan,
        assay_template: AssayTemplate | None, task: Any,
        layout: PlateLayout, variants: VariantSelection | None = None,
    ) -> Path:
        """Write ``assay_results_template.csv``: one pre-addressed row per well.

        Three kinds of column, and the separation is the point:

        * **identity** -- filled in, and expected back unchanged. Pre-addressing
          the wells is what keeps the returned file joinable; a plate map
          reconstructed afterwards from slot numbers is how a transposed
          column of results gets attributed to the wrong enzymes.
        * **plan** -- what the protocol asks for, prefixed ``plan_`` and read by
          nothing downstream. Reference for the bench, not evidence.
        * **measured** -- blank, every one of them. ``tested`` included: the
          sheet has to say which wells were run, because a template returned
          untouched would otherwise parse as a full plate of results.

        The addresses come from :class:`PlateLayout`, so where a construct
        sits is not where its ranking put it.
        """
        path = ctx.path("select_batch", "assay_results_template.csv")
        cofactors = list(task.conditions.cofactor_options)
        parents: dict[str, str] = {}
        mutations: dict[str, str] = {}
        for proposal in (variants.selected if variants else ()):
            parents[proposal.proposal_id] = proposal.parent_candidate_id or ""
            mutations[proposal.proposal_id] = proposal.label() or ""

        plan_values: list[Any] = [
            assay_template.method if assay_template else "",
            _yes_no(assay_template.confirms_product_identity
                    if assay_template else None),
            _yes_no(assay_template.requires_authentic_standard
                    if assay_template else None),
            _yes_no(assay_template.chiral_capable if assay_template else None),
            ("" if not assay_template
             or assay_template.limit_of_detection is None
             else assay_template.limit_of_detection),
            (assay_template.limit_unit or "") if assay_template else "",
            plan.replicates,
        ]
        assert len(plan_values) == len(ASSAY_PLAN_COLUMNS)
        blanks: list[Any] = [""] * len(ASSAY_MEASURED_COLUMNS)

        rows: list[list[Any]] = []
        for a in layout.assignments:
            spec = (cofactors[a.cofactor_index]
                    if a.cofactor_index < len(cofactors) else None)
            rows.append([
                plan.plan_id, a.plate, a.well, a.slot,
                a.identifier, "", a.kind, a.role,
                parents.get(a.identifier, ""), mutations.get(a.identifier, ""),
                spec.name if spec else f"condition_{a.cofactor_index + 1}",
                spec.state.value if spec else "unknown",
                a.replicate,
            ] + plan_values + blanks)

        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(ASSAY_RESULT_COLUMNS)
            writer.writerows(rows)
        return path
