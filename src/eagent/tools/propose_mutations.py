"""Interface ``propose_mutations`` -- substrate-directed engineering, round 2.

WHY THIS STEP IS SECOND AND NOT FIRST
=====================================
It would be easy to run a design model over every mined candidate and hand
back a library. That ordering destroys the only diagnosis the campaign can
make. When a round-1 mining batch comes back empty, there are two very
different worlds:

* **we never found the right family** -- the scaffold does not do this
  chemistry, and no amount of pocket redesign on it will help; or
* **we found a workable scaffold whose substrate fit is poor** -- the
  chemistry runs, the pocket is the wrong shape, and that is exactly what
  engineering fixes.

Only an experimentally confirmed parent separates them. So this interface
starts from :class:`~eagent.schemas.record.ExperimentRecord` objects that
*confirm* a parent, and it refuses to run on an unconfirmed candidate unless
an operator records an override with a reason (:class:`ParentOverride`). The
override is not a bypass: it is attached to every proposal it produced, in
``contradicting_evidence``, so the weakness travels with the variant into the
plate map and into whatever is eventually written up.

WHY THREE EVIDENCE CLASSES, AND WHY PROXIMITY IS NOT ONE OF THEM
================================================================
A 4-8 angstrom shell around the substrate in a single model is a **search
scope**. It is where a mutation *could* matter. It is not a claim that every
residue in it is worth changing: a typical shell holds 25-40 residues, most of
which are scaffold, and a library built by saturating all of them spends a
plate learning that a protein tolerates conservative surface-adjacent
substitutions.

:class:`~eagent.schemas.variant.SiteEvidence` therefore keeps three classes
apart and this module ranks on how many of them agree:

``structural``
    substrate contact, pocket entrance, orienting loop, steric clash,
    cofactor-adjacent -- derived from measured geometry in a *named* zone
    whose window comes from the :class:`EngineeringTemplate`, never from a
    constant in this file.
``family``
    positions that differ systematically between subfamilies with different
    substrate ranges, and positions that co-vary inside an active clade. Each
    signal carries the alignment or publication it came from.
``experimental``
    substitutions already shown to move activity, selectivity, stability or
    cofactor preference, each with a PMID/DOI/assay id.

A site supported only by shell membership is emitted (it is still a lead) but
is marked proximity-only, ranked last, and carries the measured distance so a
reviewer can see exactly how thin the justification is.

WHY THE CATALYTIC MACHINERY IS FROZEN IN ROUND 1
================================================
Mutating a catalytic residue and a pocket residue in the same round produces a
dead variant and no information: the single-mutant controls that would
attribute the loss do not exist, and "it stopped working" is consistent with
both changes. Round 1 therefore freezes the roles the
:class:`~eagent.schemas.templates.EngineeringTemplate` lists -- catalytic and
cofactor-anchoring -- and concentrates variation on substrate recognition and
conformational control.

This is enforced, not advised. A frozen role that cannot be located on a
parent (because the catalytic mapping is incomplete) means the freeze cannot
be checked, and the parent is **skipped with a blocker** rather than designed
on. Every emitted proposal is re-checked against the frozen index set before
it leaves the function, and a violation raises
:class:`~eagent.errors.FabricationGuardError`.

WHERE A MUTANT RESIDUE IDENTITY IS ALLOWED TO COME FROM
=======================================================
Choosing "which amino acid" is a hypothesis, and this module will only form
one from a stated source (:class:`SubstitutionSource`):

* an experimental precedent that names the substitution,
* a residue actually observed at that position in a subfamily with the wanted
  substrate range,
* a smaller residue, **only where a steric clash was measured**, ordered by
  the Bondi-era residue volumes in :data:`SIDE_CHAIN_VOLUME_A3`, or
* a ligand-aware design model (see below).

There is no path that invents a substitution for a site whose evidence does
not suggest one. A site with evidence but no sourced alternative is reported
in the excluded-sites table with that exact reason.

THE LIGANDMPNN SEAM
===================
:class:`LigandMPNNAdapter` is the hook for ligand-aware sequence design with
explicit fixed/redesigned residue specification. **It is not installed in this
environment**, so the default :class:`MissingLigandMPNN` reports unavailable
and the step degrades loudly: ``require_ligandmpnn=True`` fails the step with
:class:`~eagent.errors.ToolUnavailableError`, and a merely-requested adapter
yields a ``PARTIAL`` result with a warning, an uncertainty and a next action.
Nothing in this module imitates the model's output.

What a design model returns is **candidate sequences**. It is not a
demonstrated improvement, and every proposal it generates says so in
``contradicting_evidence``. Its output is also checked rather than trusted: a
returned sequence that changed a position declared fixed is rejected with a
blocker, because a design model that silently edited a catalytic residue would
otherwise walk straight past the freeze policy.

OFFLINE, AND NO SEQUENCE LEAVES THE MACHINE
===========================================
A parent here is typically an unpublished in-house variant. ``submit_to`` is
refused outright, and an adapter that declares ``reaches_network`` needs both
``ExecutionPolicy.allow_network`` and the ``external_sequence_submission``
approval before it is called. Publication status is assumed unknown, and
unknown counts as unpublished.
"""

from __future__ import annotations

import abc
import csv
import enum
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..context import RunContext
from ..envelope import Artifact, Provenance, Severity, Status, ToolResult
from ..errors import (
    ApprovalRequiredError,
    FabricationGuardError,
    TemplateError,
    ToolUnavailableError,
)
from ..provenance import sha256_file, sha256_obj
from ..schemas import (
    AxisExpectation,
    Candidate,
    ConfidenceLevel,
    EffectDirection,
    EngineeringTemplate,
    ExperimentRecord,
    Mutation,
    MutationProposal,
    PerformanceAxis,
    SiteEvidence,
)
from ..science import geometry as geom
from ..science.numbering import (
    STANDARD_ONE_LETTER,
    NumberingError,
    ResidueMap,
    build_map,
    verify_residue,
)
from ..science.structure_io import Atom, Chain, Structure
from .base import ScientificInterface
# The disclosure gate is named once, in mine_sequences, so the controller, the
# manifest and every interface that could leak a sequence agree on the string.
from .mine_sequences import EXTERNAL_SUBMISSION_GATE

__all__ = [
    "EXTERNAL_SUBMISSION_GATE",
    "SIDE_CHAIN_VOLUME_A3",
    "STERIC_RELIEF_LADDER",
    "DEFAULT_MAX_SUBSTITUTIONS_PER_SITE",
    "DEFAULT_MAX_COMBINATIONS_PER_PARENT",
    "PROPOSALS_TSV_COLUMNS",
    "EXCLUDED_TSV_COLUMNS",
    "StructuralRole",
    "SubstitutionSource",
    "SitePriority",
    "StructuralObservation",
    "FamilySignal",
    "ExperimentalPrecedent",
    "ParentOverride",
    "LigandMPNNRequest",
    "LigandMPNNResult",
    "LigandMPNNAdapter",
    "MissingLigandMPNN",
    "SubstitutionOption",
    "SiteCandidate",
    "ExcludedSite",
    "ParentConfirmation",
    "confirm_parent",
    "frozen_indices",
    "derive_structural_observations",
    "ProposeMutations",
]


# ==========================================================================
# Named constants.
#
# None of these is a catalytic threshold. The only geometric windows used by
# this module come from the EngineeringTemplate's mutable zones and shell
# bounds; the numbers here are library-design knobs and a physical volume
# table.
# ==========================================================================

#: Mean residue volumes in cubic angstroms (Zamyatnin, *Annu. Rev. Biophys.
#: Bioeng.* 1 (1972) 145-165). A measured physical property of the amino
#: acids, used for one purpose only: ordering candidate substitutions by size
#: at a position where a steric clash was *measured*. It is never a threshold,
#: and no proposal is generated from it at a site with no clash observation.
SIDE_CHAIN_VOLUME_A3: dict[str, float] = {
    "G": 60.1, "A": 88.6, "S": 89.0, "C": 108.5, "D": 111.1, "P": 112.7,
    "N": 114.1, "T": 116.1, "E": 138.4, "V": 140.0, "Q": 143.8, "H": 153.2,
    "M": 162.9, "I": 166.7, "L": 166.7, "K": 168.6, "R": 173.4, "F": 189.9,
    "Y": 193.6, "W": 227.8,
}

#: Order in which smaller residues are offered when a steric clash has been
#: measured at a position. A **design convention**, not a prediction: it
#: prefers residues that are small, conformationally unremarkable and common
#: in buried positions, and it deliberately omits G and P, whose backbone
#: effects make them a separate hypothesis about loop conformation rather than
#: a volume change. CALIBRATION: families differ in what they tolerate; an
#: engineering template that knows better should supply explicit precedents.
STERIC_RELIEF_LADDER: tuple[str, ...] = ("A", "S", "V", "T", "C")

#: How many sourced substitutions one site may contribute, so that a position
#: with six precedents does not consume the whole library. Library-design knob.
DEFAULT_MAX_SUBSTITUTIONS_PER_SITE: int = 3

#: Cap on combination variants per parent. Combinations need single-mutant
#: controls, so each one costs at least three wells; the cap keeps a round-1
#: library from becoming mostly controls. Library-design knob.
DEFAULT_MAX_COMBINATIONS_PER_PARENT: int = 8

#: Columns of ``mutation_proposals.tsv``. Every field a reviewer needs to
#: challenge a proposal is a column, including the ones that argue against it.
PROPOSALS_TSV_COLUMNS: tuple[str, ...] = (
    "proposal_id", "parent_candidate_id", "parent_sequence_sha256",
    "label", "n_mutations", "is_combination",
    "numbering_reference", "positions_author", "positions_index",
    "wild_type", "mutant",
    "experimental_priority", "confidence", "generator",
    "evidence_classes", "proximity_only",
    "structural_evidence", "family_evidence", "experimental_evidence",
    "distance_to_substrate_A",
    "intended_improvement", "possible_cost",
    "supporting_models", "contradicting_evidence",
    "axis_substrate_fit", "axis_catalytic_function",
    "axis_stability_expression_risk",
    "decomposition_controls", "frozen_roles_respected",
)

#: Columns of ``excluded_sites.tsv``. Refusals stay in a table: "we considered
#: 31 shell positions and proposed at 6" and "we proposed at the 6 we found"
#: are different claims, and only this file distinguishes them.
EXCLUDED_TSV_COLUMNS: tuple[str, ...] = (
    "parent_candidate_id", "position_index", "position_author", "wild_type",
    "evidence_classes", "reason_excluded", "detail",
)


# ==========================================================================
# Evidence vocabulary
# ==========================================================================

class StructuralRole(str, enum.Enum):
    """What a structural observation actually asserts about a position.

    Kept as an enum because the difference between "an atom of this residue
    lies in the shell" and "this residue contacts the substrate" is the whole
    argument. The first is scope; the rest are reasons.
    """

    POCKET_SHELL_MEMBER = "pocket_shell_member"
    SUBSTRATE_CONTACT = "substrate_contact"
    POCKET_ENTRANCE = "pocket_entrance"
    ORIENTING_LOOP = "orienting_loop"
    STERIC_CLASH = "steric_clash"
    COFACTOR_ADJACENT = "cofactor_adjacent"

    @property
    def is_mere_proximity(self) -> bool:
        """True only for bare shell membership, which justifies nothing by itself."""
        return self is StructuralRole.POCKET_SHELL_MEMBER


class SubstitutionSource(str, enum.Enum):
    """Where the identity of a proposed replacement residue came from.

    Recorded per substitution because the four sources carry different
    authority, and a library whose rationale reads "model suggested it" for
    every member cannot be prioritised.
    """

    EXPERIMENTAL_PRECEDENT = "experimental_precedent"
    FAMILY_VARIATION = "family_variation"
    STERIC_RELIEF_RULE = "steric_relief_rule"
    LIGANDMPNN = "ligandmpnn"


class SitePriority(int, enum.Enum):
    """Ordinal tier for experimental priority. A tier, deliberately not a score.

    There is no arithmetic on these values anywhere in the module: they order
    the library and nothing else. Combining "two evidence classes" with "4.2
    angstroms from the substrate" into one number would claim an exchange rate
    between an argument and a measurement.
    """

    MULTI_EVIDENCE_WITH_EXPERIMENT = 1
    TWO_EVIDENCE_CLASSES = 2
    SINGLE_SPECIFIC_EVIDENCE = 3
    PROXIMITY_ONLY = 4

    def describe(self) -> str:
        return {
            SitePriority.MULTI_EVIDENCE_WITH_EXPERIMENT:
                "an experimental precedent agrees with structure or family signal",
            SitePriority.TWO_EVIDENCE_CLASSES:
                "two independent evidence classes agree",
            SitePriority.SINGLE_SPECIFIC_EVIDENCE:
                "one evidence class, but a specific claim rather than proximity",
            SitePriority.PROXIMITY_ONLY:
                "shell membership only: a search scope, not a reason",
        }[self]


# ==========================================================================
# Inputs
# ==========================================================================

class StructuralObservation(BaseModel):
    """One measured structural statement about one candidate position."""

    model_config = ConfigDict(extra="forbid")

    candidate_index: int = Field(..., ge=0, description="0-based parent index")
    role: StructuralRole
    detail: str = Field(..., min_length=1)
    distance_to_substrate_A: float | None = None
    zone: str | None = Field(
        None, description="Name of the EngineeringTemplate mutable zone this "
                          "observation was made inside, when it came from one."
    )
    source: str = Field(
        ..., min_length=1,
        description="Structure id / pose id the measurement was made on. A "
                    "structural claim with no structure behind it is not one."
    )


class FamilySignal(BaseModel):
    """A position that family comparison says is worth varying.

    ``alternative_residues`` is the operative field: a position that differs
    between subfamilies is only actionable when the residue the *other*
    subfamily uses is known, because that residue is the hypothesis.
    """

    model_config = ConfigDict(extra="forbid")

    observation: str = Field(..., min_length=1)
    source: str = Field(
        ..., min_length=1,
        description="Alignment id, clade analysis id, PMID or DOI. A family "
                    "signal with no alignment behind it is an assertion."
    )
    candidate_index: int | None = Field(None, ge=0)
    reference_position: int | None = Field(
        None, description="1-based position in the family reference numbering, "
                          "resolved through ResidueMap.from_reference."
    )
    alternative_residues: list[str] = Field(default_factory=list)
    covaries_with: list[int] = Field(
        default_factory=list,
        description="Other reference positions this one co-varies with in the "
                    "active clade. Recorded, never silently co-mutated.",
    )

    @model_validator(mode="after")
    def _addressable(self) -> "FamilySignal":
        if self.candidate_index is None and self.reference_position is None:
            raise ValueError(
                "a family signal must name either a candidate_index or a "
                "reference_position; an unaddressed signal cannot be checked "
                "against the parent sequence"
            )
        bad = [r for r in self.alternative_residues
               if len(r) != 1 or r.upper() not in STANDARD_ONE_LETTER]
        if bad:
            raise ValueError(f"alternative_residues must be single standard "
                             f"residue letters; got {bad}")
        return self


class ExperimentalPrecedent(BaseModel):
    """A substitution already shown to move one axis, with its citation."""

    model_config = ConfigDict(extra="forbid")

    detail: str = Field(..., min_length=1)
    source: str = Field(
        ..., min_length=1,
        description="PMID, DOI, or internal assay run id. Required: an "
                    "'already known to help' with no citation is folklore.",
    )
    axis: PerformanceAxis = PerformanceAxis.SUBSTRATE_FIT
    direction: EffectDirection = EffectDirection.UNKNOWN
    candidate_index: int | None = Field(None, ge=0)
    reference_position: int | None = None
    wild_type: str | None = None
    mutant: str | None = None
    on_this_sequence: bool = Field(
        False,
        description="True when the precedent was measured on this exact parent "
                    "rather than on a homologue. A homologue precedent is still "
                    "evidence, but it is weaker and is reported as such.",
    )

    @model_validator(mode="after")
    def _addressable(self) -> "ExperimentalPrecedent":
        if self.candidate_index is None and self.reference_position is None:
            raise ValueError(
                "an experimental precedent must name a candidate_index or a "
                "reference_position"
            )
        for letter in (self.wild_type, self.mutant):
            if letter is not None and (len(letter) != 1
                                       or letter.upper() not in STANDARD_ONE_LETTER):
                raise ValueError(f"not a standard residue letter: {letter!r}")
        return self


class ParentOverride(BaseModel):
    """Operator authorisation to design on a parent that is not confirmed.

    Exists so the refusal can be overridden *on the record* instead of by
    deleting the check. Both fields are mandatory, and the text is copied into
    every proposal the parent produces.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(..., min_length=1)
    reason: str = Field(
        ..., min_length=10,
        description="Why designing on an unconfirmed parent is justified here. "
                    "Ten characters minimum, because 'ok' is not a reason.",
    )
    authorised_by: str = Field(
        ..., min_length=1,
        description="Named person. 'the model decided' is not an authority.",
    )

    def as_contradiction(self) -> str:
        return (f"parent {self.candidate_id} has no sequence-level experimental "
                f"confirmation; designed under an override recorded by "
                f"{self.authorised_by}: {self.reason}")


# ==========================================================================
# The LigandMPNN seam
# ==========================================================================

class LigandMPNNRequest(BaseModel):
    """What a ligand-aware design model is asked for, stated explicitly.

    ``fixed_indices`` is the safety-critical field: it carries the frozen
    catalytic and cofactor-anchoring positions. It is passed to the model and
    then *re-checked on the way back*, because a design tool silently editing
    a catalytic residue is the failure this interface cannot afford.
    """

    model_config = ConfigDict(extra="forbid")

    parent_candidate_id: str
    sequence: str = Field(..., min_length=1)
    fixed_indices: list[int] = Field(default_factory=list)
    redesign_indices: list[int] = Field(default_factory=list)
    structure_path: str | None = None
    ligand_codes: list[str] = Field(default_factory=list)
    n_sequences: int = Field(8, ge=1)
    temperature: float | None = None
    random_seed: int | None = None

    @model_validator(mode="after")
    def _disjoint(self) -> "LigandMPNNRequest":
        overlap = sorted(set(self.fixed_indices) & set(self.redesign_indices))
        if overlap:
            raise ValueError(
                f"indices {overlap} are both fixed and redesigned; the request "
                f"is contradictory and the model's behaviour would be undefined"
            )
        n = len(self.sequence)
        bad = [i for i in list(self.fixed_indices) + list(self.redesign_indices)
               if not 0 <= i < n]
        if bad:
            raise ValueError(f"indices outside the parent sequence: {bad}")
        return self


class LigandMPNNResult(BaseModel):
    """Candidate sequences from a design model. Not improvements."""

    model_config = ConfigDict(extra="forbid")

    sequences: list[str] = Field(default_factory=list)
    model_version: str = Field(
        ..., min_length=1,
        description="Weights identifier, recorded in provenance. A design run "
                    "whose weights are unknown cannot be reproduced.",
    )
    per_sequence_notes: list[str] = Field(default_factory=list)
    notes: str = ""


class LigandMPNNAdapter(abc.ABC):
    """Seam for ligand-aware sequence design with fixed/redesigned residues.

    An abstract class rather than a function so a real deployment can inject a
    local LigandMPNN install, a container runner, or a test double, without
    this module ever learning how the model is executed. ``reaches_network``
    is declared by the adapter and gates the disclosure checks, so a remote
    implementation cannot quietly be treated as local.
    """

    #: Identifier recorded in provenance.
    name: ClassVar[str] = "ligandmpnn"
    #: True when calling this adapter sends the parent sequence off the machine.
    reaches_network: ClassVar[bool] = False

    @abc.abstractmethod
    def available(self) -> bool:
        """Whether the weights and the runtime are actually present here."""

    @abc.abstractmethod
    def propose(self, request: LigandMPNNRequest) -> LigandMPNNResult:
        """Return candidate sequences, or raise ``ToolUnavailableError``."""


class MissingLigandMPNN(LigandMPNNAdapter):
    """The default adapter: LigandMPNN is not installed in this environment.

    It reports unavailable and raises when called. There is deliberately no
    "approximate" fallback: a hand-rolled substitution heuristic wearing the
    name of a ligand-aware network would produce designs a reviewer would
    attribute to the network's training, and that attribution would be false.
    """

    def available(self) -> bool:
        return False

    def propose(self, request: LigandMPNNRequest) -> LigandMPNNResult:
        raise ToolUnavailableError(
            "LigandMPNN",
            "weights and runtime are absent from this environment; no "
            "substitute heuristic is offered, because its output would be "
            "mistaken for a ligand-aware design",
        )


# ==========================================================================
# Internal working types
# ==========================================================================

@dataclass(frozen=True)
class SubstitutionOption:
    """One candidate replacement residue at one position, with its authority."""

    mutant: str
    source: SubstitutionSource
    rationale: str
    citation: str
    intended_improvement: str
    possible_cost: str
    axis: PerformanceAxis = PerformanceAxis.SUBSTRATE_FIT
    direction: EffectDirection = EffectDirection.UNKNOWN
    on_this_sequence: bool = False


@dataclass
class SiteCandidate:
    """A position that survived the freeze check, with everything known about it."""

    index: int
    wild_type: str
    evidence: SiteEvidence
    structural_roles: tuple[StructuralRole, ...] = ()
    options: list[SubstitutionOption] = field(default_factory=list)
    author_token: str | None = None
    observed: bool = False

    @property
    def n_classes(self) -> int:
        return self.evidence.n_evidence_classes

    @property
    def shell_membership_only(self) -> bool:
        """Only bare shell membership argues for this site.

        Stricter than :attr:`SiteEvidence.is_proximity_only`, which fires for
        any structural-only site: a measured steric clash is structural-only
        but is a specific claim, while "an atom lies in the 4-8 A shell" is
        not. Both are reported; this one drives the priority tier.
        """
        if self.evidence.family or self.evidence.experimental:
            return False
        if not self.structural_roles:
            return True
        return all(r.is_mere_proximity for r in self.structural_roles)

    def priority(self) -> SitePriority:
        """Ordinal tier from evidence agreement. No arithmetic, by design."""
        has_exp = bool(self.evidence.experimental)
        if self.n_classes >= 2 and has_exp:
            return SitePriority.MULTI_EVIDENCE_WITH_EXPERIMENT
        if self.n_classes >= 2:
            return SitePriority.TWO_EVIDENCE_CLASSES
        if self.shell_membership_only:
            return SitePriority.PROXIMITY_ONLY
        return SitePriority.SINGLE_SPECIFIC_EVIDENCE

    def confidence(self) -> ConfidenceLevel:
        """Ordinal confidence in the *site*, not in any particular substitution."""
        tier = self.priority()
        if tier is SitePriority.MULTI_EVIDENCE_WITH_EXPERIMENT:
            return ConfidenceLevel.MODERATE
        if tier is SitePriority.TWO_EVIDENCE_CLASSES:
            return ConfidenceLevel.WEAK
        if tier is SitePriority.SINGLE_SPECIFIC_EVIDENCE:
            return ConfidenceLevel.WEAK
        return ConfidenceLevel.INSUFFICIENT


@dataclass(frozen=True)
class ExcludedSite:
    """A position that was considered and not proposed on, with the reason."""

    parent_candidate_id: str
    position_index: int | None
    position_author: str
    wild_type: str
    evidence_classes: int
    reason: str
    detail: str


@dataclass(frozen=True)
class ParentConfirmation:
    """Whether a parent has earned round-2 engineering, and on what evidence."""

    candidate_id: str
    confirmed: bool
    supporting_record_ids: tuple[str, ...]
    reason: str

    @property
    def blocked(self) -> bool:
        return not self.confirmed


# ==========================================================================
# Confirmation, freezing, and structural derivation
# ==========================================================================

def confirm_parent(
    candidate: Candidate, records: Sequence[ExperimentRecord]
) -> ParentConfirmation:
    """Decide whether ``candidate`` is an experimentally confirmed parent.

    Three conditions, all of which have been quietly skipped in real
    campaigns:

    1. the record is about **this sequence**, matched on
       ``sequence_sha256`` rather than on an accession, because accessions get
       re-annotated and an engineered parent usually has none;
    2. the outcome is ``CONFIRMED_TARGET_PRODUCT``, which the schema already
       ties to a detection method that identifies the product -- so a parent
       cannot be confirmed by a cofactor absorbance trace; and
    3. the supporting evidence is **sequence-level experimental**. A homologue
       that works is a reason to mine, not a licence to spend a plate
       engineering this protein.

    Returns the verdict with the record ids behind it, so a reviewer can go
    read them.
    """
    want = candidate.sequence_record.sequence_sha256
    if not want:
        return ParentConfirmation(
            candidate.candidate_id, False, (),
            "parent has no sequence hash, so no record can be matched to it",
        )
    matched = [r for r in records if r.sequence_sha256 == want]
    if not matched:
        return ParentConfirmation(
            candidate.candidate_id, False, (),
            f"no ExperimentRecord carries sequence hash {want[:19]}...; this "
            f"parent has not been tested",
        )
    positive = [r for r in matched if r.outcome.is_positive]
    if not positive:
        outcomes = sorted({r.outcome.value for r in matched})
        return ParentConfirmation(
            candidate.candidate_id, False,
            tuple(r.record_id for r in matched),
            f"{len(matched)} record(s) for this sequence, none confirming the "
            f"target product (outcomes: {', '.join(outcomes)}). Engineering a "
            f"scaffold that has not been shown to run the chemistry cannot "
            f"separate 'wrong family' from 'poor substrate fit'",
        )
    sequence_level = [r for r in positive if r.max_strength.is_sequence_level]
    if not sequence_level:
        return ParentConfirmation(
            candidate.candidate_id, False,
            tuple(r.record_id for r in positive),
            f"{len(positive)} positive record(s), but the strongest evidence is "
            f"{max(r.max_strength for r in positive).value}, not sequence-level "
            f"experimental; a homologue's activity does not confirm this parent",
        )
    return ParentConfirmation(
        candidate.candidate_id, True,
        tuple(r.record_id for r in sequence_level),
        f"confirmed by {len(sequence_level)} sequence-level experimental "
        f"record(s) with product identity established",
    )


def frozen_indices(
    candidate: Candidate,
    template: EngineeringTemplate,
    extra: Iterable[int] = (),
) -> tuple[set[int], list[str]]:
    """Resolve the template's frozen roles to 0-based indices on this parent.

    Returns ``(indices, unresolved_roles)``. An unresolved role is **not** a
    licence to proceed: the caller refuses to design on a parent whose freeze
    cannot be verified, because "we could not find the catalytic tyrosine" and
    "the catalytic tyrosine is safe" look identical in the output otherwise.

    ``extra`` lets a caller freeze additional positions (an engineered
    disulfide, a tag junction) without editing the template.
    """
    mapping = candidate.catalytic_mapping
    out: set[int] = set(int(i) for i in extra)
    unresolved: list[str] = []
    for role in template.frozen_roles:
        idx = mapping.role_to_index.get(role)
        if idx is None:
            unresolved.append(role)
            continue
        out.add(int(idx))
    return out, unresolved


def derive_structural_observations(
    candidate: Candidate,
    structure: Structure,
    residue_map: ResidueMap,
    template: EngineeringTemplate,
    substrate_atoms: Sequence[Atom],
    cofactor_atoms: Sequence[Atom] = (),
    *,
    vdw_overlap_tolerance: float = geom.DEFAULT_VDW_OVERLAP_TOLERANCE_A,
    source: str = "",
) -> tuple[list[StructuralObservation], list[str]]:
    """Measure structural evidence inside the template's declared zones.

    Every window used here comes from the :class:`EngineeringTemplate`:
    each entry of ``mutable_zones`` may carry ``min_angstrom``/``max_angstrom``
    and a ``role``; zones that do not are measured in the template's
    ``default_shell_min_angstrom``/``default_shell_max_angstrom`` and recorded
    as bare :attr:`StructuralRole.POCKET_SHELL_MEMBER` -- scope, not a reason.
    No window is written in this file.

    Cofactor adjacency is measured from zero to the template's outer shell
    bound, because the point of the cofactor check is to catch positions that
    would perturb cofactor binding, and those are the close ones that a shell
    with an inner cut-out would hide.

    Steric clashes come from :func:`eagent.science.geometry.clash_pairs`, a
    hard-sphere screen. The second return value lists honest problems --
    elements with no van der Waals radius, residues the map could not place --
    so a caller cannot read "no clashes" off a measurement that did not happen.
    """
    notes: list[str] = []
    observations: list[StructuralObservation] = []
    label = source or (candidate.best_structure.structure_id
                       if candidate.best_structure else "unnamed structure")
    if not substrate_atoms:
        raise ValueError(
            "derive_structural_observations needs the substrate's atoms; an "
            "empty selection usually means the ligand code matched nothing, "
            "and an empty shell would be reported as 'no pocket residues'"
        )

    def index_of(res: Any) -> int | None:
        return residue_map.to_index(res.author_key)

    zones: list[Mapping[str, Any]] = [z for z in template.mutable_zones
                                      if isinstance(z, Mapping)] or []
    if not zones:
        zones = [{
            "name": "default_substrate_shell",
            "role": StructuralRole.POCKET_SHELL_MEMBER.value,
            "rationale": "the engineering template declares no named zone, so "
                         "the default shell bounds are used as a search scope",
        }]

    for zone in zones:
        name = str(zone.get("name") or "unnamed_zone")
        lo = float(zone.get("min_angstrom", template.default_shell_min_angstrom))
        hi = float(zone.get("max_angstrom", template.default_shell_max_angstrom))
        try:
            role = StructuralRole(str(zone.get("role")
                                      or StructuralRole.POCKET_SHELL_MEMBER.value))
        except ValueError:
            notes.append(
                f"zone '{name}' declares role {zone.get('role')!r}, which is not "
                f"a StructuralRole; recorded as shell membership only")
            role = StructuralRole.POCKET_SHELL_MEMBER
        try:
            shell = geom.pocket_shell(structure, substrate_atoms, lo, hi)
        except geom.GeometryError as exc:
            notes.append(f"zone '{name}' [{lo}, {hi}] A could not be measured: {exc}")
            continue
        for res in shell:
            idx = index_of(res)
            if idx is None:
                notes.append(
                    f"zone '{name}': observed residue {res} has no counterpart "
                    f"in the parent sequence and was not proposed on")
                continue
            try:
                dist = geom.min_distance(res.heavy_atoms(), substrate_atoms)
            except geom.GeometryError:
                dist = None
            observations.append(StructuralObservation(
                candidate_index=idx, role=role,
                detail=(f"zone '{name}' [{lo:g}, {hi:g}] A from the substrate; "
                        f"{zone.get('rationale', 'no rationale recorded in the template')}"),
                distance_to_substrate_A=dist, zone=name, source=label,
            ))

    if cofactor_atoms:
        try:
            cof_shell = geom.pocket_shell(
                structure, cofactor_atoms, 0.0,
                float(template.default_shell_max_angstrom))
        except geom.GeometryError as exc:
            notes.append(f"cofactor shell could not be measured: {exc}")
            cof_shell = []
        for res in cof_shell:
            idx = index_of(res)
            if idx is None:
                continue
            try:
                dist = geom.min_distance(res.heavy_atoms(), cofactor_atoms)
            except geom.GeometryError:
                dist = None
            observations.append(StructuralObservation(
                candidate_index=idx, role=StructuralRole.COFACTOR_ADJACENT,
                detail=(f"within {template.default_shell_max_angstrom:g} A of the "
                        f"cofactor (closest heavy-atom approach "
                        f"{'%.2f' % dist if dist is not None else 'unmeasured'} A)"),
                distance_to_substrate_A=None, zone="cofactor_shell", source=label,
            ))

    pairs, unscreened = geom.clash_pairs(
        structure.atoms(), substrate_atoms,
        vdw_overlap_tolerance=vdw_overlap_tolerance)
    if unscreened:
        elements = sorted({a.element.strip().upper() for a in unscreened})
        notes.append(
            f"{len(unscreened)} atom(s) of element(s) {', '.join(elements)} have "
            f"no van der Waals radius and were not screened for clashes; "
            f"'no clash' does not cover them")
    clashed: dict[int, float] = {}
    for pair in pairs:
        # clash_pairs builds ClashPair(protein_atom, ligand_atom), so atom_a is
        # always the residue side of the pair.
        key = pair.atom_a.residue_key
        idx = residue_map.to_index((key[0], key[1], key[2]))
        if idx is None:
            continue
        clashed[idx] = max(clashed.get(idx, 0.0), pair.overlap_A)
    for idx, overlap in sorted(clashed.items()):
        observations.append(StructuralObservation(
            candidate_index=idx, role=StructuralRole.STERIC_CLASH,
            detail=(f"hard-sphere overlap of {overlap:.2f} A with the substrate "
                    f"at tolerance {vdw_overlap_tolerance:g} A; a localisation "
                    f"screen, not an energy"),
            zone="clash_screen", source=label,
        ))
    return observations, notes


# ==========================================================================
# The interface
# ==========================================================================

class ProposeMutations(ScientificInterface):
    """Build a round-1 variant library from a confirmed parent.

    The guarantees this class is responsible for, each enforced in code:

    * no design on an unconfirmed parent without a recorded override, and the
      override text attached to every proposal it produced;
    * no proposal at a frozen catalytic or cofactor-anchoring position, and no
      proposals at all for a parent whose freeze cannot be verified;
    * every wild-type letter checked through
      :func:`~eagent.science.numbering.verify_residue` before a proposal
      exists, with ``require_observed`` for structure-only arguments;
    * one numbering convention per parent, chosen explicitly and named in
      ``numbering_reference``;
    * a single-mutant decomposition control for every combination;
    * a loud failure, never an imitation, when LigandMPNN is absent;
    * nothing transmitted off this machine.
    """

    name: ClassVar[str] = "propose_mutations"
    description: ClassVar[str] = (
        "Round-2 mutation proposals from experimentally confirmed parents: "
        "three evidence classes, a verified freeze policy, per-axis "
        "expectations and single-mutant decomposition controls."
    )
    required_fields: ClassVar[tuple[str, ...]] = (
        "reaction.substrate.isomeric_smiles",
        "reaction.product.isomeric_smiles",
    )
    required_approvals: ClassVar[tuple[str, ...]] = ()
    depends_on: ClassVar[tuple[str, ...]] = ("evaluate_catalysis", "ingest_results")
    version: ClassVar[str] = "0.1.0"

    # -- entry point -------------------------------------------------------
    def execute(
        self,
        ctx: RunContext,
        *,
        parents: Sequence[Candidate] | None = None,
        records: Sequence[ExperimentRecord] = (),
        engineering_template: EngineeringTemplate | Mapping[str, EngineeringTemplate] | None = None,
        residue_maps: Mapping[str, ResidueMap] | None = None,
        structure_chains: Mapping[str, Chain] | None = None,
        structural_observations: Mapping[str, Sequence[StructuralObservation]] | None = None,
        family_signals: Mapping[str, Sequence[FamilySignal]] | None = None,
        experimental_precedents: Mapping[str, Sequence[ExperimentalPrecedent]] | None = None,
        overrides: Sequence[ParentOverride] = (),
        extra_frozen_indices: Mapping[str, Sequence[int]] | None = None,
        ligandmpnn: LigandMPNNAdapter | None = None,
        require_ligandmpnn: bool = False,
        include_proximity_only: bool = True,
        max_substitutions_per_site: int = DEFAULT_MAX_SUBSTITUTIONS_PER_SITE,
        max_combinations_per_parent: int = DEFAULT_MAX_COMBINATIONS_PER_PARENT,
        submit_to: str | None = None,
        **_: Any,
    ) -> ToolResult:
        """Propose variants for every supplied parent and write the tables.

        ``structural_observations`` may be passed directly (the usual case
        when :func:`derive_structural_observations` was run by an earlier
        step) or omitted; it is never fabricated from the sequence alone.
        """
        refusal = self._refuse_external_submission(ctx, submit_to)
        if refusal is not None:
            return refusal

        if not parents:
            return ToolResult.failure(
                self.name,
                "no parents supplied: pass parents=[Candidate, ...] that an "
                "earlier round confirmed. This step does not mine candidates, "
                "and an empty library is not a design result.",
                code="no_parents",
            )

        result = ToolResult(status=Status.SUCCESS)
        override_by_id = {o.candidate_id: o for o in overrides}
        residue_maps = dict(residue_maps or {})
        structure_chains = dict(structure_chains or {})
        structural_observations = dict(structural_observations or {})
        family_signals = dict(family_signals or {})
        experimental_precedents = dict(experimental_precedents or {})
        extra_frozen_indices = dict(extra_frozen_indices or {})

        mpnn_state = self._prepare_ligandmpnn(ctx, ligandmpnn, require_ligandmpnn,
                                              result)

        proposals: list[MutationProposal] = []
        excluded: list[ExcludedSite] = []
        confirmations: list[ParentConfirmation] = []
        per_parent_numbering: dict[str, str] = {}
        designed_parents = 0

        for parent in parents:
            template = self._resolve_template(ctx, parent, engineering_template)
            confirmation = confirm_parent(parent, records)
            confirmations.append(confirmation)
            override = override_by_id.get(parent.candidate_id)

            if confirmation.blocked and override is None:
                result.add_flag(
                    "unconfirmed_parent", Severity.BLOCKER,
                    f"{parent.candidate_id}: {confirmation.reason}. Refusing to "
                    f"design on it. Supply a ParentOverride with a reason to "
                    f"proceed anyway, or test the parent first.",
                    subject=parent.candidate_id)
                result.add_uncertainty(
                    "parent_not_confirmed",
                    f"Does {parent.candidate_id} actually run the target "
                    f"chemistry at all? Without that, a failed variant library "
                    f"cannot distinguish a wrong scaffold from a poor fit.",
                    affects=[parent.candidate_id],
                    resolvable_by="assay the parent against the target substrate")
                result.add_next(
                    "select_batch",
                    "Test the parent itself before engineering it",
                    {"candidate_ids": [parent.candidate_id]}, requires_human=True)
                continue
            if confirmation.blocked and override is not None:
                result.add_flag(
                    "unconfirmed_parent_override", Severity.WARN,
                    override.as_contradiction(), subject=parent.candidate_id)

            rmap, map_error = self._residue_map_for(parent, residue_maps,
                                                    structure_chains)
            if rmap is None:
                result.add_flag(
                    "numbering_unavailable", Severity.BLOCKER,
                    f"{parent.candidate_id}: {map_error}. No proposal is emitted: "
                    f"a mutation written against an unverified numbering mutates "
                    f"some other residue, expresses, folds, and reads as a failed "
                    f"hypothesis.",
                    subject=parent.candidate_id)
                continue

            frozen, unresolved = frozen_indices(
                parent, template, extra_frozen_indices.get(parent.candidate_id, ()))
            if unresolved:
                result.add_flag(
                    "freeze_unverifiable", Severity.BLOCKER,
                    f"{parent.candidate_id}: frozen role(s) "
                    f"{', '.join(unresolved)} are not mapped onto this parent, so "
                    f"the round-1 freeze cannot be verified. Map the catalytic "
                    f"site first; designing without the check risks mutating the "
                    f"machinery and losing attribution for everything else.",
                    subject=parent.candidate_id)
                excluded.append(ExcludedSite(
                    parent.candidate_id, None, "-", "-", 0,
                    "freeze_unverifiable",
                    f"unmapped frozen roles: {', '.join(unresolved)}"))
                continue

            sites, site_exclusions = self._collect_sites(
                parent, rmap, template, frozen,
                structural_observations.get(parent.candidate_id, ()),
                family_signals.get(parent.candidate_id, ()),
                experimental_precedents.get(parent.candidate_id, ()),
                include_proximity_only=include_proximity_only,
                max_substitutions_per_site=max_substitutions_per_site,
            )
            excluded.extend(site_exclusions)
            if not sites:
                result.add_flag(
                    "no_actionable_sites", Severity.WARN,
                    f"{parent.candidate_id}: no position carried both evidence "
                    f"and a sourced replacement residue. See excluded_sites.tsv; "
                    f"the library is short rather than padded.",
                    subject=parent.candidate_id)
                continue

            mode, reference_label, mode_note = self._numbering_decision(
                parent, rmap, sites)
            per_parent_numbering[parent.candidate_id] = reference_label
            if mode_note:
                result.add_flag("numbering_convention", Severity.INFO, mode_note,
                                subject=parent.candidate_id)

            singles, single_exclusions = self._build_singles(
                parent, rmap, sites, mode, reference_label, override)
            excluded.extend(single_exclusions)
            combos = self._build_combinations(
                parent, singles, sites, template, reference_label, override,
                max_combinations_per_parent)
            mpnn_proposals = self._build_ligandmpnn(
                ctx, parent, rmap, template, frozen, sites, mode, reference_label,
                override, mpnn_state, result)

            parent_proposals = singles + combos + mpnn_proposals
            if parent_proposals:
                designed_parents += 1
            proposals.extend(parent_proposals)

        self._assert_freeze_respected(parents, proposals, engineering_template,
                                      ctx, extra_frozen_indices)

        proposals.sort(key=lambda p: (p.experimental_priority, p.is_combination,
                                      p.parent_candidate_id, p.proposal_id))

        tsv_path = self._write_proposals(ctx, proposals)
        excluded_path = self._write_excluded(ctx, excluded)

        result.artifacts.append(Artifact(
            key="mutation_proposals", path=str(tsv_path), kind="table",
            sha256=sha256_file(tsv_path), n_records=len(proposals),
            summary=("one row per variant: both numbering systems, the three "
                     "evidence classes, the intended gain, the property at "
                     "risk, and the decomposition controls"),
        ))
        result.artifacts.append(Artifact(
            key="excluded_sites", path=str(excluded_path), kind="table",
            sha256=sha256_file(excluded_path), n_records=len(excluded),
            summary=("positions considered and not proposed on, with the reason; "
                     "frozen roles, unverifiable numbering and sites with no "
                     "sourced substitution"),
        ))

        result.data.update({
            "n_proposals": len(proposals),
            "n_singles": sum(1 for p in proposals if not p.is_combination),
            "n_combinations": sum(1 for p in proposals if p.is_combination),
            "n_parents_designed": designed_parents,
            "n_parents_refused": len(parents) - designed_parents,
            "proposals": [p.model_dump(mode="json") for p in proposals],
            "confirmations": [
                {"candidate_id": c.candidate_id, "confirmed": c.confirmed,
                 "records": list(c.supporting_record_ids), "reason": c.reason}
                for c in confirmations
            ],
            "numbering_reference": per_parent_numbering,
            "excluded_sites": [
                {"parent_candidate_id": e.parent_candidate_id,
                 "position_index": e.position_index,
                 "position_author": e.position_author,
                 "wild_type": e.wild_type,
                 "evidence_classes": e.evidence_classes,
                 "reason": e.reason, "detail": e.detail}
                for e in excluded
            ],
            "ligandmpnn": mpnn_state,
        })

        if not proposals:
            result.status = Status.FAILED
            result.message = (
                "no proposal survived the parent-confirmation, freeze and "
                "wild-type verification checks; see excluded_sites.tsv")
        elif result.blockers or mpnn_state.get("degraded"):
            result.status = Status.PARTIAL
            result.message = (
                f"{len(proposals)} proposal(s) for {designed_parents} parent(s); "
                f"{len(parents) - designed_parents} parent(s) refused")
        else:
            result.message = (
                f"{len(proposals)} proposal(s) for {designed_parents} parent(s)")

        result.add_uncertainty(
            "designs_are_hypotheses",
            "Which of these substitutions actually improves substrate fit? "
            "Every row is a pre-stated expectation, not a prediction with a "
            "calibrated error bar; only the assay settles it.",
            affects=[p.proposal_id for p in proposals[:20]],
            resolvable_by="experiment")
        result.add_next(
            "select_batch",
            "Compose the variant round under the construct budget, keeping "
            "every combination's single-mutant controls in the same plate",
            {"n_proposals": len(proposals)}, requires_human=False)

        result.provenance = Provenance(
            tool=self.name, tool_version=self.version,
            inputs_sha256={
                "parents": sha256_obj([p.candidate_id for p in parents]),
                "records": sha256_obj([r.record_id for r in records]),
                "structural_observations": sha256_obj(
                    {k: [o.model_dump(mode="json") for o in v]
                     for k, v in structural_observations.items()}),
                "family_signals": sha256_obj(
                    {k: [s.model_dump(mode="json") for s in v]
                     for k, v in family_signals.items()}),
                "experimental_precedents": sha256_obj(
                    {k: [e.model_dump(mode="json") for e in v]
                     for k, v in experimental_precedents.items()}),
            },
            databases={},
            models={"ligandmpnn": str(mpnn_state.get("model_version") or "absent")},
            parameters={
                "include_proximity_only": include_proximity_only,
                "max_substitutions_per_site": max_substitutions_per_site,
                "max_combinations_per_parent": max_combinations_per_parent,
                "require_ligandmpnn": require_ligandmpnn,
                "steric_relief_ladder": list(STERIC_RELIEF_LADDER),
                "numbering_reference": per_parent_numbering,
                "overrides": [o.model_dump(mode="json") for o in overrides],
                "allow_network": ctx.policy.allow_network,
            },
            random_seed=ctx.seed_for(self.name),
        )
        return result

    # -- guards ------------------------------------------------------------
    def _refuse_external_submission(
        self, ctx: RunContext, submit_to: str | None
    ) -> ToolResult | None:
        """Refuse to hand a parent sequence to any outside service.

        A round-2 parent is almost always unpublished: an in-house isolate or
        a variant from the previous plate. Posting it to a web design server
        discloses it irreversibly and can forfeit novelty, and this step has no
        legitimate need to transmit anything -- the design seam is local by
        contract. So the request is refused here rather than gated.
        """
        if not submit_to:
            return None
        suffix = ("" if ctx.policy.allow_network
                  else " (the run policy also has allow_network=False)")
        return ToolResult.failure(
            self.name,
            f"refused to submit parent sequences to '{submit_to}'. "
            f"propose_mutations is a local computation; disclosing an "
            f"unpublished parent is an operator decision taken outside this "
            f"interface{suffix}",
            code="external_submission_refused",
        )

    def _prepare_ligandmpnn(
        self, ctx: RunContext, adapter: LigandMPNNAdapter | None,
        required: bool, result: ToolResult,
    ) -> dict[str, Any]:
        """Decide whether the design seam may be used, and say so loudly.

        Four outcomes, kept apart because they mean different things: not
        requested; requested and usable; requested, present, but blocked by
        the disclosure policy; requested and absent. Only the second one ever
        produces a sequence, and the last one is the state of this
        environment.
        """
        state: dict[str, Any] = {
            "requested": adapter is not None or required,
            "available": False,
            "degraded": False,
            "adapter": None if adapter is None else type(adapter).__name__,
            "model_version": None,
            "reason": "",
        }
        if adapter is None and not required:
            state["reason"] = "no adapter supplied; rational proposals only"
            return state
        if adapter is None:
            adapter = MissingLigandMPNN()
            state["adapter"] = type(adapter).__name__

        if getattr(adapter, "reaches_network", False):
            if not ctx.policy.allow_network:
                state["degraded"] = True
                state["reason"] = (
                    f"adapter {type(adapter).__name__} declares reaches_network "
                    f"and ExecutionPolicy.allow_network is False; not called")
                result.add_flag("ligandmpnn_network_blocked", Severity.WARN,
                                state["reason"])
                return state
            try:
                ctx.require_approval(
                    EXTERNAL_SUBMISSION_GATE,
                    detail=("a remote LigandMPNN adapter would send parent "
                            "sequences and coordinates off this machine; "
                            "publication status is unknown and unknown counts "
                            "as unpublished"))
            except ApprovalRequiredError as exc:
                state["degraded"] = True
                state["reason"] = str(exc)
                result.add_flag("ligandmpnn_disclosure_blocked", Severity.BLOCKER,
                                state["reason"])
                result.add_next("request_approval",
                                "Remote design would disclose an unpublished parent",
                                {"gate": EXTERNAL_SUBMISSION_GATE},
                                requires_human=True)
                return state

        if not adapter.available():
            state["degraded"] = True
            state["reason"] = (
                f"{type(adapter).__name__} reports the LigandMPNN weights and "
                f"runtime are not present; no substitute is generated")
            if required:
                raise ToolUnavailableError("LigandMPNN", state["reason"])
            result.add_flag("ligandmpnn_unavailable", Severity.WARN, state["reason"])
            result.add_uncertainty(
                "ligandmpnn_unavailable",
                "Would a ligand-aware design model propose pocket substitutions "
                "the evidence-driven route missed? Unanswered: the model is not "
                "installed and nothing here imitates it.",
                affects=["propose_mutations"],
                resolvable_by="install LigandMPNN and rerun with an adapter")
            result.add_next("install_tool",
                            "Ligand-aware design would widen the library",
                            {"tool": "LigandMPNN"}, requires_human=True)
            return state

        state["available"] = True
        state["instance"] = adapter
        return state

    # -- template and numbering -------------------------------------------
    def _resolve_template(
        self, ctx: RunContext, parent: Candidate,
        supplied: EngineeringTemplate | Mapping[str, EngineeringTemplate] | None,
    ) -> EngineeringTemplate:
        """Find the engineering template for this parent, or raise.

        Raises :class:`~eagent.errors.TemplateError` rather than falling back
        to a default policy. A default freeze list would be the worst possible
        guess: too short and the machinery gets mutated, too long and the
        library has nothing to vary.
        """
        if isinstance(supplied, EngineeringTemplate):
            return supplied
        family = parent.family.family_name
        if isinstance(supplied, Mapping):
            for key in (parent.candidate_id, family, "*"):
                if key and key in supplied:
                    found = supplied[key]
                    if isinstance(found, EngineeringTemplate):
                        return found
                    raise TemplateError(
                        f"'{key}' resolved to {type(found).__name__}, not an "
                        f"EngineeringTemplate")
        library = ctx.templates
        if library is not None:
            found: Any = None
            if isinstance(library, Mapping):
                store = library.get("engineering")
                if isinstance(store, Mapping):
                    found = store.get(family) or store.get(parent.candidate_id)
            else:
                store = getattr(library, "engineering_templates", None)
                if isinstance(store, Mapping):
                    found = store.get(family)
                if found is None:
                    getter = getattr(library, "engineering", None)
                    if callable(getter):
                        found = getter(family)
            if isinstance(found, EngineeringTemplate):
                return found
        raise TemplateError(
            f"no EngineeringTemplate for parent {parent.candidate_id} "
            f"(family {family!r}); the frozen roles, the mutable zones and the "
            f"round-1 mutation cap all come from it, and there is no defensible "
            f"default for any of them"
        )

    def _residue_map_for(
        self, parent: Candidate, residue_maps: Mapping[str, ResidueMap],
        chains: Mapping[str, Chain],
    ) -> tuple[ResidueMap | None, str]:
        """Get or build the parent's residue map, or explain why there is none."""
        existing = residue_maps.get(parent.candidate_id)
        if existing is not None:
            if existing.candidate_sequence != parent.sequence.upper():
                return None, (
                    "the supplied ResidueMap was built against a different "
                    "sequence than this parent carries; using it would shift "
                    "every position")
            return existing, ""
        chain = chains.get(parent.candidate_id)
        if chain is None:
            return None, (
                "no ResidueMap and no structure chain supplied; author "
                "numbering cannot be established and wild-type letters cannot "
                "be verified against coordinates")
        try:
            return build_map(parent.sequence, chain), ""
        except NumberingError as exc:
            return None, f"residue map could not be built: {exc}"

    def _numbering_decision(
        self, parent: Candidate, rmap: ResidueMap, sites: Sequence[SiteCandidate],
    ) -> tuple[str, str, str]:
        """Pick ONE numbering convention for this parent's whole library.

        Mixing conventions inside a library is the off-by-N disaster in its
        most deniable form: half the tubes are right. ``Mutation`` carries an
        integer author position, so a position with an insertion code or with
        no coordinates cannot be expressed in author numbering at all. Rather
        than expressing some proposals one way and some the other, the whole
        parent falls back to 1-based parent-sequence numbering and
        ``numbering_reference`` says so.
        """
        unobserved = [s for s in sites if not s.observed]
        icoded = [s for s in sites
                  if s.author_token and not s.author_token.rstrip().isdigit()]
        if unobserved or icoded:
            reason_parts = []
            if unobserved:
                reason_parts.append(
                    f"{len(unobserved)} site(s) have no coordinates")
            if icoded:
                reason_parts.append(
                    f"{len(icoded)} site(s) carry an insertion code")
            note = (
                f"{parent.candidate_id}: author numbering cannot express every "
                f"site ({'; '.join(reason_parts)}), so the whole library is "
                f"written in 1-based parent-sequence numbering. Mixing the two "
                f"inside one library is how half a plate ends up at the wrong "
                f"residue.")
            return ("parent_1_based",
                    f"parent sequence 1-based ({parent.candidate_id})", note)
        structure_id = (parent.best_structure.structure_id
                        if parent.best_structure else "structure")
        return ("author",
                f"{structure_id} chain {rmap.chain_id} author numbering", "")

    # -- site collection ---------------------------------------------------
    def _collect_sites(
        self,
        parent: Candidate,
        rmap: ResidueMap,
        template: EngineeringTemplate,
        frozen: set[int],
        structural: Sequence[StructuralObservation],
        family: Sequence[FamilySignal],
        precedents: Sequence[ExperimentalPrecedent],
        *,
        include_proximity_only: bool,
        max_substitutions_per_site: int,
    ) -> tuple[list[SiteCandidate], list[ExcludedSite]]:
        """Fuse the three evidence classes onto positions, honouring the freeze.

        Everything that is dropped is dropped into ``excluded`` with a reason:
        a frozen role, a position outside the parent, a signal that could not
        be addressed through the reference numbering, or -- the common one --
        a position with evidence but no sourced replacement residue. A design
        step whose refusals are invisible cannot be audited.
        """
        excluded: list[ExcludedSite] = []
        by_index: dict[int, SiteCandidate] = {}
        seq = parent.sequence.upper()

        def author_token(index: int) -> tuple[str | None, bool]:
            pos = rmap.index_to_author.get(index)
            return (pos.token if pos is not None else None, pos is not None)

        def site_for(index: int) -> SiteCandidate | None:
            if not 0 <= index < len(seq):
                excluded.append(ExcludedSite(
                    parent.candidate_id, index, "-", "-", 0, "index_out_of_range",
                    f"index {index} is outside the parent sequence (len {len(seq)})"))
                return None
            if index in frozen:
                token, _ = author_token(index)
                excluded.append(ExcludedSite(
                    parent.candidate_id, index, token or "-", seq[index], 0,
                    "frozen_role",
                    "catalytic or cofactor-anchoring role frozen for round 1 by "
                    f"EngineeringTemplate {template.template_id}"))
                return None
            site = by_index.get(index)
            if site is None:
                token, observed = author_token(index)
                site = SiteCandidate(index=index, wild_type=seq[index],
                                     evidence=SiteEvidence(), author_token=token,
                                     observed=observed)
                by_index[index] = site
            return site

        def resolve_reference(ref: int | None, direct: int | None,
                              what: str, detail: str) -> int | None:
            if direct is not None:
                return direct
            if ref is None:
                return None
            idx = rmap.from_reference(int(ref))
            if idx is None:
                excluded.append(ExcludedSite(
                    parent.candidate_id, None, f"ref:{ref}", "-", 0,
                    "reference_position_unmapped",
                    f"{what} at reference position {ref} has no aligned residue "
                    f"in this parent (a deletion relative to the reference); "
                    f"{detail}"))
            return idx

        roles_by_index: dict[int, set[StructuralRole]] = {}

        # -- structural ----------------------------------------------------
        for obs in structural:
            site = site_for(obs.candidate_index)
            if site is None:
                continue
            site.evidence.structural.append(
                f"{obs.role.value}: {obs.detail} [{obs.source}]")
            roles_by_index.setdefault(obs.candidate_index, set()).add(obs.role)
            if obs.distance_to_substrate_A is not None:
                current = site.evidence.distance_to_substrate_A
                if current is None or obs.distance_to_substrate_A < current:
                    site.evidence.distance_to_substrate_A = obs.distance_to_substrate_A

        # -- family --------------------------------------------------------
        for sig in family:
            idx = resolve_reference(sig.reference_position, sig.candidate_index,
                                    "family signal", sig.observation)
            if idx is None:
                continue
            site = site_for(idx)
            if site is None:
                continue
            covary = (f"; co-varies with reference position(s) "
                      f"{', '.join(str(p) for p in sig.covaries_with)}"
                      if sig.covaries_with else "")
            site.evidence.family.append(
                f"{sig.observation} [{sig.source}]{covary}")
            for alt in sig.alternative_residues:
                alt = alt.upper()
                if alt == site.wild_type:
                    continue
                site.options.append(SubstitutionOption(
                    mutant=alt, source=SubstitutionSource.FAMILY_VARIATION,
                    rationale=(f"residue observed at this position in a subfamily "
                               f"with a different substrate range: {sig.observation}"),
                    citation=sig.source,
                    intended_improvement=(
                        "shift substrate preference toward the target by adopting "
                        "the residue the other subfamily uses"),
                    possible_cost=(
                        "the donor subfamily differs at other positions too, so "
                        "the residue may be incompatible with this scaffold's "
                        "packing"),
                    axis=PerformanceAxis.SUBSTRATE_FIT,
                    direction=EffectDirection.IMPROVE,
                ))

        # -- experimental --------------------------------------------------
        for prec in precedents:
            idx = resolve_reference(prec.reference_position, prec.candidate_index,
                                    "experimental precedent", prec.detail)
            if idx is None:
                continue
            site = site_for(idx)
            if site is None:
                continue
            scope = ("measured on this sequence" if prec.on_this_sequence
                     else "measured on a homologue")
            site.evidence.experimental.append(
                f"{prec.detail} ({prec.axis.value} {prec.direction.value}, "
                f"{scope}) [{prec.source}]")
            if prec.wild_type and prec.wild_type.upper() != site.wild_type:
                excluded.append(ExcludedSite(
                    parent.candidate_id, idx, site.author_token or "-",
                    site.wild_type, site.n_classes, "precedent_wild_type_mismatch",
                    f"precedent expects {prec.wild_type} at index {idx}, parent "
                    f"has {site.wild_type}; the precedent addresses a different "
                    f"residue and its substitution is not offered"))
                continue
            if prec.mutant and prec.mutant.upper() != site.wild_type:
                site.options.append(SubstitutionOption(
                    mutant=prec.mutant.upper(),
                    source=SubstitutionSource.EXPERIMENTAL_PRECEDENT,
                    rationale=f"{prec.detail} ({scope})",
                    citation=prec.source,
                    intended_improvement=(
                        f"reproduce a reported {prec.direction.value} on "
                        f"{prec.axis.value}"),
                    possible_cost=(
                        "an effect measured on a homologue need not transfer; "
                        "scaffold context differs"
                        if not prec.on_this_sequence else
                        "a reported gain on one substrate need not hold for this one"),
                    axis=prec.axis, direction=prec.direction,
                    on_this_sequence=prec.on_this_sequence,
                ))

        # -- template precedents ------------------------------------------
        for entry in template.known_beneficial_mutations:
            if not isinstance(entry, Mapping):
                continue
            idx = resolve_reference(
                _as_int(entry.get("reference_position")),
                _as_int(entry.get("candidate_index")),
                "template beneficial mutation", str(entry.get("notes", "")))
            if idx is None:
                continue
            site = site_for(idx)
            if site is None:
                continue
            mutant = str(entry.get("mutant", "")).strip().upper()
            citation = str(entry.get("evidence", "")).strip()
            if not citation:
                excluded.append(ExcludedSite(
                    parent.candidate_id, idx, site.author_token or "-",
                    site.wild_type, site.n_classes, "template_entry_unsourced",
                    f"EngineeringTemplate {template.template_id} lists a "
                    f"beneficial mutation at index {idx} with no evidence field; "
                    f"an unsourced precedent is not evidence"))
                continue
            note = str(entry.get("notes")
                       or "listed as beneficial by the engineering template")
            site.evidence.experimental.append(f"{note} [{citation}]")
            if mutant and mutant in STANDARD_ONE_LETTER and mutant != site.wild_type:
                site.options.append(SubstitutionOption(
                    mutant=mutant,
                    source=SubstitutionSource.EXPERIMENTAL_PRECEDENT,
                    rationale=str(entry.get("notes",
                                            "listed in the engineering template")),
                    citation=citation,
                    intended_improvement=str(entry.get(
                        "intended_improvement",
                        "reproduce the reported benefit in this scaffold")),
                    possible_cost=str(entry.get(
                        "possible_cost",
                        "the template's precedent was established in another "
                        "member of the family")),
                    axis=PerformanceAxis.SUBSTRATE_FIT,
                    direction=EffectDirection.IMPROVE,
                ))

        # -- steric relief, only where a clash was measured -----------------
        for index, roles in roles_by_index.items():
            site = by_index.get(index)
            if site is None or StructuralRole.STERIC_CLASH not in roles:
                continue
            wt_volume = SIDE_CHAIN_VOLUME_A3.get(site.wild_type)
            if wt_volume is None:
                continue
            for letter in STERIC_RELIEF_LADDER:
                volume = SIDE_CHAIN_VOLUME_A3[letter]
                if letter == site.wild_type or volume >= wt_volume:
                    continue
                site.options.append(SubstitutionOption(
                    mutant=letter, source=SubstitutionSource.STERIC_RELIEF_RULE,
                    rationale=(f"a hard-sphere overlap with the substrate was "
                               f"measured here; {letter} is smaller than "
                               f"{site.wild_type} ({volume:.0f} vs "
                               f"{wt_volume:.0f} A^3, Zamyatnin 1972)"),
                    citation="design rule: volume reduction at a measured clash",
                    intended_improvement=(
                        "relieve the measured steric overlap so the substrate can "
                        "adopt a productive orientation"),
                    possible_cost=(
                        "removing side-chain volume can leave a cavity, lose "
                        "packing and shape complementarity, and cost both "
                        "stability and selectivity"),
                    axis=PerformanceAxis.SUBSTRATE_FIT,
                    direction=EffectDirection.IMPROVE,
                ))

        # -- finalise ------------------------------------------------------
        sites: list[SiteCandidate] = []
        for index in sorted(by_index):
            site = by_index[index]
            site.structural_roles = tuple(sorted(roles_by_index.get(index, ()),
                                                 key=lambda r: r.value))
            if site.n_classes == 0:
                excluded.append(ExcludedSite(
                    parent.candidate_id, index, site.author_token or "-",
                    site.wild_type, 0, "no_evidence",
                    "position reached the site table with no evidence attached"))
                continue
            if site.shell_membership_only and not include_proximity_only:
                excluded.append(ExcludedSite(
                    parent.candidate_id, index, site.author_token or "-",
                    site.wild_type, site.n_classes, "proximity_only_excluded",
                    "only shell membership supports this position and "
                    "include_proximity_only is False"))
                continue
            site.options = _dedupe_options(site.options)[:max_substitutions_per_site]
            if not site.options:
                excluded.append(ExcludedSite(
                    parent.candidate_id, index, site.author_token or "-",
                    site.wild_type, site.n_classes, "no_sourced_substitution",
                    "the position has evidence but nothing names a replacement "
                    "residue; a mutant identity is not invented here"))
                continue
            sites.append(site)

        sites.sort(key=lambda s: (s.priority().value, s.index))
        return sites, excluded

    # -- proposal construction --------------------------------------------
    def _author_position(self, mode: str, site: SiteCandidate) -> int:
        """Author position under the chosen convention.

        In ``parent_1_based`` mode the author field carries ``index + 1``, and
        ``numbering_reference`` names that convention explicitly, so nothing
        downstream has to guess which system a bare integer belongs to.
        """
        if mode == "author":
            token = site.author_token or ""
            return int(token)
        return site.index + 1

    def _build_singles(
        self, parent: Candidate, rmap: ResidueMap, sites: Sequence[SiteCandidate],
        mode: str, reference_label: str, override: ParentOverride | None,
    ) -> tuple[list[MutationProposal], list[ExcludedSite]]:
        """One proposal per (site, sourced substitution), wild type verified first."""
        out: list[MutationProposal] = []
        excluded: list[ExcludedSite] = []
        for site in sites:
            structure_only = not (site.evidence.family or site.evidence.experimental)
            try:
                verify_residue(rmap, site.index, site.wild_type,
                               require_observed=structure_only)
            except NumberingError as exc:
                excluded.append(ExcludedSite(
                    parent.candidate_id, site.index, site.author_token or "-",
                    site.wild_type, site.n_classes, "wild_type_unverified", str(exc)))
                continue
            for option in site.options:
                mutation = Mutation(
                    wild_type=site.wild_type,
                    position_author=self._author_position(mode, site),
                    position_index=site.index,
                    mutant=option.mutant,
                )
                out.append(self._make_proposal(
                    parent, [mutation], [site], [option], reference_label,
                    override, generator=f"rational:{option.source.value}",
                    priority=site.priority().value,
                    confidence=self._option_confidence(site, option),
                ))
        return out, excluded

    def _build_combinations(
        self, parent: Candidate, singles: Sequence[MutationProposal],
        sites: Sequence[SiteCandidate], template: EngineeringTemplate,
        reference_label: str, override: ParentOverride | None, cap: int,
    ) -> list[MutationProposal]:
        """Pair the best-evidenced sites, always with their single-mutant controls.

        Only sites that already produced a single proposal are combined, which
        is what makes the decomposition control a real plate member rather than
        an id in a column. The arity cap comes from the engineering template
        (``max_simultaneous_mutations_round1``), not from this file: how many
        changes a family tolerates at once is a family fact.
        """
        if template.max_simultaneous_mutations_round1 < 2 or cap <= 0:
            return []
        by_site: dict[int, list[MutationProposal]] = {}
        for proposal in singles:
            by_site.setdefault(proposal.mutations[0].position_index,
                               []).append(proposal)
        site_by_index = {s.index: s for s in sites}
        eligible = [s for s in sites
                    if s.index in by_site
                    and s.priority().value <= SitePriority.TWO_EVIDENCE_CLASSES.value]
        eligible.sort(key=lambda s: (s.priority().value, s.index))

        out: list[MutationProposal] = []
        for left, right in itertools.combinations(eligible, 2):
            if len(out) >= cap:
                break
            a = by_site[left.index][0]
            b = by_site[right.index][0]
            mutations = [a.mutations[0], b.mutations[0]]
            combo_sites = [site_by_index[left.index], site_by_index[right.index]]
            options = [_option_for(left, a.mutations[0].mutant),
                       _option_for(right, b.mutations[0].mutant)]
            if any(o is None for o in options):
                continue
            out.append(self._make_proposal(
                parent, mutations, combo_sites, [o for o in options if o],
                reference_label, override, generator="rational:combination",
                priority=max(left.priority().value, right.priority().value),
                confidence=ConfidenceLevel.WEAK,
                decomposition_controls=[a.proposal_id, b.proposal_id],
            ))
        return out

    def _build_ligandmpnn(
        self, ctx: RunContext, parent: Candidate, rmap: ResidueMap,
        template: EngineeringTemplate, frozen: set[int],
        sites: Sequence[SiteCandidate], mode: str, reference_label: str,
        override: ParentOverride | None, state: dict[str, Any],
        result: ToolResult,
    ) -> list[MutationProposal]:
        """Turn design-model sequences into proposals, after checking the freeze.

        The returned sequences are *candidates*, never improvements, and every
        proposal says so. A sequence that edited a fixed position is rejected
        with a blocker rather than trimmed back to the allowed positions:
        trimming would hide that the model ignored the specification, and the
        next run would hit the same problem invisibly.
        """
        adapter = state.get("instance")
        if not state.get("available") or adapter is None:
            return []
        seq = parent.sequence.upper()
        redesign = sorted({s.index for s in sites} - frozen)
        if not redesign:
            return []
        request = LigandMPNNRequest(
            parent_candidate_id=parent.candidate_id,
            sequence=seq,
            fixed_indices=sorted(frozen),
            redesign_indices=redesign,
            structure_path=(parent.best_structure.path
                            if parent.best_structure else None),
            ligand_codes=sorted({c for s in parent.structures
                                 for c in s.bound_ligands}),
            random_seed=ctx.seed_for(f"{self.name}:ligandmpnn"),
        )
        try:
            designed = adapter.propose(request)
        except ToolUnavailableError as exc:
            result.add_flag("ligandmpnn_unavailable", Severity.WARN, str(exc),
                            subject=parent.candidate_id)
            state["degraded"] = True
            state["reason"] = str(exc)
            return []
        state["model_version"] = designed.model_version
        site_by_index = {s.index: s for s in sites}
        out: list[MutationProposal] = []

        for n, variant in enumerate(designed.sequences):
            variant = variant.strip().upper()
            if len(variant) != len(seq):
                result.add_flag(
                    "ligandmpnn_length_mismatch", Severity.BLOCKER,
                    f"{parent.candidate_id}: design {n} is {len(variant)} "
                    f"residues against a parent of {len(seq)}; indices would not "
                    f"line up and the sequence is discarded",
                    subject=parent.candidate_id)
                continue
            diffs = [i for i, (a, b) in enumerate(zip(seq, variant)) if a != b]
            violated = sorted(set(diffs) & frozen)
            if violated:
                result.add_flag(
                    "ligandmpnn_touched_frozen_residue", Severity.BLOCKER,
                    f"{parent.candidate_id}: design {n} changed frozen "
                    f"position(s) {violated} despite being declared fixed; the "
                    f"whole sequence is discarded rather than trimmed, because a "
                    f"design tool that ignores the specification must be visible",
                    subject=parent.candidate_id)
                continue
            if not diffs:
                continue
            if len(diffs) > template.max_simultaneous_mutations_round1:
                result.add_flag(
                    "ligandmpnn_exceeds_round1_cap", Severity.WARN,
                    f"{parent.candidate_id}: design {n} carries {len(diffs)} "
                    f"substitutions, over the template cap of "
                    f"{template.max_simultaneous_mutations_round1} for round 1; "
                    f"not proposed, because its single-mutant decomposition "
                    f"would consume the library",
                    subject=parent.candidate_id)
                continue
            unknown = [i for i in diffs if i not in site_by_index]
            if unknown:
                for i in unknown:
                    site_by_index[i] = SiteCandidate(
                        index=i, wild_type=seq[i],
                        evidence=SiteEvidence(structural=[
                            f"redesigned by {adapter.name} "
                            f"{designed.model_version} in the ligand-aware pass "
                            f"[{adapter.name}]"]),
                        author_token=(rmap.index_to_author[i].token
                                      if i in rmap.index_to_author else None),
                        observed=i in rmap.index_to_author,
                    )
            members: list[Mutation] = []
            member_sites: list[SiteCandidate] = []
            member_options: list[SubstitutionOption] = []
            failed = False
            for i in diffs:
                site = site_by_index[i]
                try:
                    verify_residue(rmap, i, seq[i], require_observed=False)
                except NumberingError as exc:
                    result.add_flag("ligandmpnn_wild_type_unverified",
                                    Severity.BLOCKER, str(exc),
                                    subject=parent.candidate_id)
                    failed = True
                    break
                members.append(Mutation(
                    wild_type=seq[i],
                    position_author=self._author_position(mode, site),
                    position_index=i, mutant=variant[i]))
                member_sites.append(site)
                member_options.append(SubstitutionOption(
                    mutant=variant[i], source=SubstitutionSource.LIGANDMPNN,
                    rationale=(f"proposed by {adapter.name} "
                               f"{designed.model_version} with the catalytic and "
                               f"cofactor-anchoring positions declared fixed"),
                    citation=f"{adapter.name}:{designed.model_version}",
                    intended_improvement=(
                        "a ligand-aware candidate sequence for the redesigned "
                        "pocket; the model scores sequence recovery given the "
                        "ligand, which is not a claim about turnover"),
                    possible_cost=(
                        "sequence-recovery models are trained on native "
                        "complexes, so a design can be well-packed and "
                        "catalytically dead; expression and selectivity are both "
                        "at risk"),
                    axis=PerformanceAxis.SUBSTRATE_FIT,
                    direction=EffectDirection.UNKNOWN,
                ))
            if failed:
                continue

            controls: list[str] = []
            if len(members) > 1:
                for mutation, site, option in zip(members, member_sites,
                                                  member_options):
                    control = self._make_proposal(
                        parent, [mutation], [site], [option], reference_label,
                        override, generator=f"ligandmpnn:{designed.model_version}",
                        priority=SitePriority.SINGLE_SPECIFIC_EVIDENCE.value,
                        confidence=ConfidenceLevel.INSUFFICIENT)
                    out.append(control)
                    controls.append(control.proposal_id)
            out.append(self._make_proposal(
                parent, members, member_sites, member_options, reference_label,
                override, generator=f"ligandmpnn:{designed.model_version}",
                priority=SitePriority.SINGLE_SPECIFIC_EVIDENCE.value,
                confidence=ConfidenceLevel.INSUFFICIENT,
                decomposition_controls=controls))
        return _dedupe_proposals(out)

    def _make_proposal(
        self, parent: Candidate, mutations: Sequence[Mutation],
        sites: Sequence[SiteCandidate], options: Sequence[SubstitutionOption],
        reference_label: str, override: ParentOverride | None, *,
        generator: str, priority: int, confidence: ConfidenceLevel,
        decomposition_controls: Sequence[str] = (),
    ) -> MutationProposal:
        """Assemble one proposal with everything a reviewer needs to argue with it."""
        label = "/".join(f"{m.wild_type}{m.position_author}{m.mutant}"
                         for m in mutations)
        proposal_id = f"{parent.candidate_id}|{label}"
        evidence = {str(m.position_author): site.evidence
                    for m, site in zip(mutations, sites)}
        intended = [o.intended_improvement for o in options]
        costs = [o.possible_cost for o in options]
        costs.append(
            "soluble expression: any substitution in a folded protein is a "
            "folding risk until measured, and an expression failure says "
            "nothing about catalysis")
        supporting = sorted({o.citation for o in options if o.citation})
        contradicting: list[str] = []
        if override is not None:
            contradicting.append(override.as_contradiction())
        for site, option in zip(sites, options):
            if site.shell_membership_only:
                distance = site.evidence.distance_to_substrate_A
                contradicting.append(
                    f"position {site.index}: supported by shell membership only"
                    + (f" (closest approach {distance:.2f} A)"
                       if distance is not None else "")
                    + "; proximity is a search scope, not a reason to mutate")
            if option.source is SubstitutionSource.STERIC_RELIEF_RULE:
                contradicting.append(
                    f"position {site.index}: the replacement residue "
                    f"{option.mutant} comes from a volume-reduction design rule, "
                    f"not from a precedent; no experiment supports this identity")
            if option.source is SubstitutionSource.LIGANDMPNN:
                contradicting.append(
                    f"position {site.index}: a design-model candidate sequence, "
                    f"not a demonstrated improvement")
            if (option.source is SubstitutionSource.EXPERIMENTAL_PRECEDENT
                    and not option.on_this_sequence):
                contradicting.append(
                    f"position {site.index}: the precedent was measured on a "
                    f"homologue, so transfer to this scaffold is untested")
        if len(mutations) > 1:
            contradicting.append(
                "a combination's effect is not the sum of its parts; the "
                "single-mutant controls listed here are what make it attributable")
        return MutationProposal(
            proposal_id=proposal_id,
            parent_candidate_id=parent.candidate_id,
            parent_sequence_sha256=parent.sequence_record.sequence_sha256 or "",
            mutations=list(mutations),
            numbering_reference=reference_label,
            site_evidence=evidence,
            intended_improvement=sorted(set(intended)) or ["unstated"],
            possible_cost=sorted(set(costs)),
            supporting_models=supporting,
            contradicting_evidence=sorted(set(contradicting)),
            frozen_roles_respected=True,
            generator=generator,
            confidence=confidence,
            experimental_priority=max(1, int(priority)),
            decomposition_controls=list(decomposition_controls),
            axis_expectations=self._axis_expectations(sites, options),
        )

    def _option_confidence(self, site: SiteCandidate,
                           option: SubstitutionOption) -> ConfidenceLevel:
        """Confidence in this substitution, which is never above the site's.

        A strong site argument plus an invented residue identity is a weak
        proposal, so the substitution's own authority caps the result.
        """
        site_level = site.confidence()
        cap = {
            SubstitutionSource.EXPERIMENTAL_PRECEDENT:
                ConfidenceLevel.MODERATE if option.on_this_sequence
                else ConfidenceLevel.WEAK,
            SubstitutionSource.FAMILY_VARIATION: ConfidenceLevel.WEAK,
            SubstitutionSource.STERIC_RELIEF_RULE: ConfidenceLevel.INSUFFICIENT,
            SubstitutionSource.LIGANDMPNN: ConfidenceLevel.INSUFFICIENT,
        }[option.source]
        return site_level if site_level.rank <= cap.rank else cap

    def _axis_expectations(
        self, sites: Sequence[SiteCandidate], options: Sequence[SubstitutionOption],
    ) -> list[AxisExpectation]:
        """State an expectation on each of the three axes, before testing.

        All three are always present, including the ones nobody has an opinion
        about: an axis left out of the record is an axis that gets narrated
        after the result arrives. There is deliberately no method anywhere that
        combines them.
        """
        rationale_fit = "; ".join(sorted({o.rationale for o in options}))
        evidence = sorted({o.citation for o in options if o.citation})
        fit_direction = EffectDirection.IMPROVE if any(
            o.direction is EffectDirection.IMPROVE for o in options
        ) else EffectDirection.UNKNOWN

        cofactor_adjacent = any(
            StructuralRole.COFACTOR_ADJACENT in s.structural_roles for s in sites)
        catalytic_direction = (EffectDirection.DEGRADE if cofactor_adjacent
                               else EffectDirection.UNKNOWN)
        catalytic_rationale = (
            "a cofactor-adjacent position was changed, so cofactor binding and "
            "hydride-transfer geometry are both at risk even though the "
            "catalytic residues themselves are frozen"
            if cofactor_adjacent else
            "the catalytic residues are frozen, but a pocket change can still "
            "reposition the substrate relative to them; direction unknown")

        risky = [o.mutant for o in options if o.mutant in ("P", "G")]
        charge_in = [o.mutant for o in options if o.mutant in ("D", "E", "K", "R")]
        buried_loss = [s.wild_type for s in sites
                       if s.wild_type in ("W", "F", "Y", "L", "I", "M")]
        stability_direction = (
            EffectDirection.DEGRADE
            if risky or (charge_in and buried_loss) else EffectDirection.UNKNOWN)
        stability_rationale = "; ".join(filter(None, [
            (f"introduces backbone-perturbing residue(s) {', '.join(risky)}"
             if risky else ""),
            (f"buries charge ({', '.join(charge_in)}) while removing a large "
             f"hydrophobic residue ({', '.join(sorted(set(buried_loss)))})"
             if charge_in and buried_loss else ""),
        ])) or ("no specific destabilising feature identified; the risk is the "
                "generic one that any pocket substitution carries")

        return [
            AxisExpectation(
                axis=PerformanceAxis.SUBSTRATE_FIT, direction=fit_direction,
                rationale=rationale_fit or "no rationale recorded",
                evidence=evidence, confidence=ConfidenceLevel.WEAK),
            AxisExpectation(
                axis=PerformanceAxis.CATALYTIC_FUNCTION,
                direction=catalytic_direction, rationale=catalytic_rationale,
                evidence=evidence, confidence=ConfidenceLevel.INSUFFICIENT),
            AxisExpectation(
                axis=PerformanceAxis.STABILITY_EXPRESSION_RISK,
                direction=stability_direction, rationale=stability_rationale,
                evidence=[], confidence=ConfidenceLevel.INSUFFICIENT),
        ]

    # -- final checks and output ------------------------------------------
    def _assert_freeze_respected(
        self, parents: Sequence[Candidate], proposals: Sequence[MutationProposal],
        supplied: EngineeringTemplate | Mapping[str, EngineeringTemplate] | None,
        ctx: RunContext, extra: Mapping[str, Sequence[int]],
    ) -> None:
        """Re-derive the frozen set and check every emitted proposal against it.

        Belt and braces, deliberately. The site collector already filters
        frozen positions, but a later edit that adds a generation path -- the
        design-model branch is exactly that -- could bypass it. A silent breach
        here mutates the catalytic machinery and makes the whole library
        uninterpretable, so it raises rather than warns.
        """
        by_id = {p.candidate_id: p for p in parents}
        for proposal in proposals:
            parent = by_id.get(proposal.parent_candidate_id)
            if parent is None:
                continue
            template = self._resolve_template(ctx, parent, supplied)
            frozen, _ = frozen_indices(parent, template,
                                       extra.get(parent.candidate_id, ()))
            touched = sorted({m.position_index for m in proposal.mutations} & frozen)
            if touched:
                raise FabricationGuardError(
                    f"{proposal.proposal_id} mutates frozen position(s) "
                    f"{touched}; the round-1 freeze from EngineeringTemplate "
                    f"{template.template_id} was breached")

    def _write_proposals(self, ctx: RunContext,
                         proposals: Sequence[MutationProposal]) -> Path:
        """Write ``mutation_proposals.tsv``: one row per variant, nothing hidden."""
        path = ctx.path("propose_mutations", "mutation_proposals.tsv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
            writer.writerow(PROPOSALS_TSV_COLUMNS)
            for p in proposals:
                evidence = list(p.site_evidence.values())
                structural = [e for ev in evidence for e in ev.structural]
                family = [e for ev in evidence for e in ev.family]
                experimental = [e for ev in evidence for e in ev.experimental]
                classes = max((ev.n_evidence_classes for ev in evidence),
                              default=0)
                distances = [ev.distance_to_substrate_A for ev in evidence
                             if ev.distance_to_substrate_A is not None]
                axes = p.axes_summary()
                writer.writerow([
                    p.proposal_id, p.parent_candidate_id, p.parent_sequence_sha256,
                    p.label(), len(p.mutations), p.is_combination,
                    p.numbering_reference,
                    ";".join(str(m.position_author) for m in p.mutations),
                    ";".join(str(m.position_index) for m in p.mutations),
                    ";".join(m.wild_type for m in p.mutations),
                    ";".join(m.mutant for m in p.mutations),
                    p.experimental_priority, p.confidence.value, p.generator or "",
                    classes,
                    any(ev.is_proximity_only for ev in evidence),
                    " | ".join(structural), " | ".join(family),
                    " | ".join(experimental),
                    f"{min(distances):.2f}" if distances else "",
                    " | ".join(p.intended_improvement),
                    " | ".join(p.possible_cost),
                    " | ".join(p.supporting_models),
                    " | ".join(p.contradicting_evidence),
                    axes[PerformanceAxis.SUBSTRATE_FIT.value],
                    axes[PerformanceAxis.CATALYTIC_FUNCTION.value],
                    axes[PerformanceAxis.STABILITY_EXPRESSION_RISK.value],
                    ";".join(p.decomposition_controls),
                    p.frozen_roles_respected,
                ])
        return path

    def _write_excluded(self, ctx: RunContext,
                        excluded: Sequence[ExcludedSite]) -> Path:
        """Write ``excluded_sites.tsv``: the refusals, with their reasons."""
        path = ctx.path("propose_mutations", "excluded_sites.tsv")
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
            writer.writerow(EXCLUDED_TSV_COLUMNS)
            for e in excluded:
                writer.writerow([
                    e.parent_candidate_id,
                    "" if e.position_index is None else e.position_index,
                    e.position_author, e.wild_type, e.evidence_classes,
                    e.reason, e.detail,
                ])
        return path


# ==========================================================================
# Small helpers
# ==========================================================================

def _as_int(value: Any) -> int | None:
    """Int or ``None``; never a silent 0, which would address residue one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe_options(options: Sequence[SubstitutionOption]) -> list[SubstitutionOption]:
    """Keep one option per mutant letter, preferring the strongest source.

    Ordered experimental > family > steric rule > design model, so a position
    with both a precedent and a volume-reduction guess reports the precedent.
    """
    rank = {
        SubstitutionSource.EXPERIMENTAL_PRECEDENT: 0,
        SubstitutionSource.FAMILY_VARIATION: 1,
        SubstitutionSource.STERIC_RELIEF_RULE: 2,
        SubstitutionSource.LIGANDMPNN: 3,
    }
    best: dict[str, SubstitutionOption] = {}
    for option in options:
        current = best.get(option.mutant)
        if current is None or rank[option.source] < rank[current.source]:
            best[option.mutant] = option
    return sorted(best.values(), key=lambda o: (rank[o.source], o.mutant))


def _option_for(site: SiteCandidate, mutant: str) -> SubstitutionOption | None:
    for option in site.options:
        if option.mutant == mutant:
            return option
    return None


def _dedupe_proposals(proposals: Sequence[MutationProposal]) -> list[MutationProposal]:
    """Drop duplicate proposal ids, keeping the first. Ids are deterministic."""
    seen: set[str] = set()
    out: list[MutationProposal] = []
    for p in proposals:
        if p.proposal_id in seen:
            continue
        seen.add(p.proposal_id)
        out.append(p)
    return out
