"""Interface ``evaluate_catalysis``: mechanism-aware feasibility, never a score.

WHAT THIS STEP REPLACES
=======================
The standard computational screen is "dock the substrate, keep the poses whose
score is good and whose nearest-atom distance is under 10 angstroms". Both
halves of that are unsound, and they are unsound in ways that produce a
confident, well-formatted, completely wrong shortlist:

* A docking score is a pseudo-energy whose sign convention, units and dynamic
  range belong to one scoring function. It ranks pocket volume and ligand
  heavy-atom count at least as strongly as it ranks catalysis, so comparing it
  across families ranks the families.
* A bare distance cutoff is usually measured between the wrong things. The
  substrate centroid to the protein centroid, or a ligand atom to a residue
  alpha carbon, says where the molecule *is*. Catalysis is a statement about
  the atoms the chemistry touches: in an NAD(P)H-dependent carbonyl reduction,
  the nicotinamide C4 and the carbonyl carbon, with the carbonyl oxygen
  engaged by the family's stabilising residues and the donor approaching from
  the face that gives the wanted enantiomer.

This module therefore asks a mechanism-shaped question of every pose, against
the family's sourced :class:`~eagent.schemas.templates.CatalyticTemplate`, and
reports it in a form a reviewer can disagree with: every measurement, the
window it was compared against, where that window came from, whether the
constraint was one the modelling run had already enforced, and -- when a
question could not be answered -- that it could not be answered.

THE THREE SITUATIONS A NAIVE PIPELINE COLLAPSES
===============================================
:class:`PoseOutcome` keeps apart, structurally, three things that all look
like "no" by the end of a pipeline:

``MECHANISM_VIOLATED``
    The geometry was measured and falls outside a window we trust. A
    *computational negative*: a statement about this pose under this template,
    and still not an experimental result.
``NOT_MEASURABLE``
    The pose could not be built, parsed or measured; an atom role was never
    bound; the cofactor is absent from the file. **Nothing was tested.** An
    enzyme whose docking run crashed has not been shown to be inactive.
``INPUT_ERROR``
    The inputs were wrong -- the oxidised cofactor where the mechanism needs
    the reduced one, the wrong cofactor entirely. A defect to repair, not a
    property of the enzyme.

The confusion is made *unexpressible* rather than merely discouraged:
:meth:`PoseOutcome.as_record_outcome` maps every member into the
non-experimental half of :class:`~eagent.schemas.record.OutcomeClass` and
raises :class:`~eagent.errors.FabricationGuardError` if a mapping ever lands on
an experimental class. There is no code path from this module to
``NO_TARGET_PRODUCT_DETECTED``. The counts are likewise kept in separate
fields (:attr:`CandidateEvaluation.n_violated` versus
:attr:`CandidateEvaluation.n_not_measurable`) and the robustness denominator
counts only decided poses, so a failed modelling run cannot be averaged into a
rate of failure.

ROBUSTNESS IS CORRECTED FOR CIRCULARITY, NOT ANNOTATED AFTERWARDS
=================================================================
A pose ensemble built under restraints satisfies exactly the constraints it
was built to satisfy. Computing ``G = satisfied / decided`` over it and
flagging the circularity afterwards still emits ``G = 1.0``, and that figure
is what reaches the scorecard, the tables and the batch. So the correction is
in the definition instead: :meth:`EvaluateCatalysis._rollup` counts a pose in
the numerator only when it satisfies a gating constraint that was **not**
restrained while it was built, counts it in the denominator only when it
carried at least one such constraint, and reports ``G = None`` with a stated
reason -- never 1.0, never 0.0 -- for a candidate that carries no independent
evidence at all. The uncorrected sampling fraction survives under the separate
name :attr:`CandidateEvaluation.sampling_G`, so the correction is auditable
and the two numbers can never be mistaken for each other.

WINDOWS COME FROM THE TEMPLATE, AND SO DOES THE AUTHORITY TO REJECT
===================================================================
No catalytic threshold is written in this file. Every pass/fail decision is
made by :meth:`~eagent.schemas.templates.GeometryConstraint.satisfied_by` on a
constraint that carries its own ``calibrated_on`` provenance.

Where a family has no trustworthy geometric template -- the constraint is
uncalibrated, or the whole template is a labelled theoretical model (a
theozyme) -- the correct response is to raise the uncertainty, not to reject
the family wholesale against a window that is probably wrong. That is
implemented, not merely documented: :class:`WindowAuthority` decides whether a
failed window may *reject*. A failure against a provisional window yields
:attr:`PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW`, which leaves
``GeometryReport.gating_passed`` at ``None`` (undecided, not failed), is
excluded from the robustness denominator, and raises an explicit
:class:`~eagent.envelope.Uncertainty` naming the constraints that need
calibration. Symmetrically, a *pass* against a provisional window does not get
to be strong evidence: the reported robustness level and the
``catalytic_geometry`` axis are both downgraded one ordinal step.

THE 10 ANGSTROM NUMBER
======================
It appears exactly once, as :data:`DEFAULT_POCKET_LOCALISATION_A`, and it is a
**QC localisation screen, not evidence of catalytic ability**. It is measured
between the substrate's designated reactive atom and the template's catalytic
functional atoms -- never centroid to centroid, never an arbitrary atom to an
alpha carbon. The template may override it with its own
``pocket_localisation`` constraint, and it needs per-family calibration. A
pose that fails it gets a QC flag and nothing else; the mechanism constraints
decide the outcome.

OFFLINE AND LOCAL BY CONSTRUCTION
=================================
This step reads coordinate files from disk and computes. It opens no socket,
and it refuses outright to hand a candidate sequence or structure to an
external service: see
:meth:`EvaluateCatalysis._refuse_external_submission`. An unpublished sequence
leaving the machine is not a networking detail.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..context import RunContext
from ..envelope import Artifact, Provenance, Severity, Status, ToolResult
from ..errors import CircularEvidenceError, FabricationGuardError, TemplateError
from ..provenance import sha256_file, sha256_obj
from ..schemas import (
    Candidate,
    CatalyticTemplate,
    CofactorState,
    ComplexPose,
    ConfidenceLevel,
    GeometryConstraint,
    GeometryReport,
    OutcomeClass,
    ScoreDimension,
    StereoCall,
    cofactor_state_from_ligand_code,
)
from ..science import geometry as geom
from ..science.calibration import CalibrationContext, CalibrationStore
from ..science.family_numbering import FamilyNumberingScheme
from ..science.pocket import PocketResidues, pocket_residues_for_pose
from ..science import stereo as stereo_mod
from ..science.robustness import (
    DEFAULT_MIN_VALID_POSES,
    CircularityGuard,
    classify_robustness,
    pose_robustness,
    wilson_interval,
)
from ..science.scorecard import (
    build_scorecard,
    cofactor_identity,
    evaluate_feasibility_gates,
    explain,
    gate_uncertainties,
    scorecard_qc_flags,
)
from ..science.structure_io import (
    Atom,
    Residue,
    Structure,
    StructureParseError,
    read_structure,
)
from .base import ScientificInterface

__all__ = [
    "DEFAULT_POCKET_LOCALISATION_A",
    "DEFAULT_COMPETING_GROUP_MARGIN_A",
    "DEFAULT_CLASH_TOLERANCE_A",
    "POCKET_LOCALISATION_CONSTRAINT_NAME",
    "FAMILY_MATCH_GATE",
    "NICOTINAMIDE_HYDRIDE_DONOR_ATOM",
    "NS_SUBSTRATE",
    "NS_COFACTOR",
    "NS_PROTEIN",
    "PoseOutcome",
    "WindowAuthority",
    "ResidueSelector",
    "ProteinAtomRef",
    "PoseBinding",
    "RoleContextResult",
    "PocketLocalisation",
    "ChemoselectivityCheck",
    "CofactorCheck",
    "ClashScreen",
    "PoseEvaluation",
    "CandidateEvaluation",
    "template_authority",
    "constraint_authority",
    "pocket_localisation_window",
    "classify_aspect",
    "build_role_context",
    "EvaluateCatalysis",
]


# ==========================================================================
# Named defaults.
#
# Every number below is a QC-localisation parameter: it decides how loudly a
# problem is reported, never whether a candidate is catalytically competent.
# The windows that decide competence live in a sourced CatalyticTemplate's
# GeometryConstraint objects and are applied by GeometryConstraint.satisfied_by.
# All of them need per-family (or per-protocol) calibration, all are
# overridable per call, and the value actually used is written to provenance.
# ==========================================================================

#: Largest distance, in angstroms, from the substrate's **reactive atom** to
#: the nearest **catalytic functional atom named by the template** at which the
#: substrate still counts as localised in the catalytic region.
#:
#: This is the familiar 10 A figure, and this is where it belongs: a coarse "is
#: the molecule even in the right part of the protein" screen, defined between
#: the atoms the chemistry touches. It is NOT a catalytic criterion. A
#: substrate 4 A away in the wrong orientation cannot react, and a flexible
#: substrate whose reactive atom sits 11 A out in one member of an ensemble is
#: not thereby disqualified. Failing this screen raises a QC flag and changes
#: no verdict.
#:
#: NEEDS PER-FAMILY CALIBRATION. A deep AKR barrel and a shallow SDR cleft do
#: not have the same pocket depth. A family template may override it with a
#: constraint named :data:`POCKET_LOCALISATION_CONSTRAINT_NAME`, which is the
#: preferred route because such a constraint carries its own ``calibrated_on``.
DEFAULT_POCKET_LOCALISATION_A: float = 10.0

#: How much closer, in angstroms, a non-target electrophilic atom must be to
#: the hydride donor than the designated reactive atom before it counts as the
#: group actually occupying the reactive position.
#:
#: A margin rather than a bare comparison because both distances carry the same
#: coordinate noise and a 0.05 A difference is not a chemoselectivity finding.
#: NEEDS PER-PROTOCOL CALIBRATION: set it from the positional scatter of the
#: pose generator, and say so through ``competing_group_margin_source``.
#: Until a caller does, the comparison carries
#: :attr:`WindowAuthority.UNCALIBRATED` and a displacement is reported as
#: :attr:`PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW` rather than rejecting the
#: pose -- the same rule every other window in this module follows.
DEFAULT_COMPETING_GROUP_MARGIN_A: float = 0.5

#: Van der Waals overlap tolerated before a contact counts as a clash. Passed
#: straight to :func:`eagent.science.geometry.clash_count`, which documents it
#: as a screen for localising steric problems rather than as an energy.
DEFAULT_CLASH_TOLERANCE_A: float = geom.DEFAULT_VDW_OVERLAP_TOLERANCE_A

#: Radius around the substrate within which a residue is collected as part of
#: the pocket, for the diversity signature only.
#:
#: NOT a claim and NOT calibrated: membership of this shell says a residue has
#: a heavy atom within the radius in this one model, nothing more. It is a
#: *comparison scope* -- two candidates' pockets are compared over the same
#: radius -- and it never gates, scores or rejects anything. The value is the
#: conventional first-and-second-shell radius; a campaign with a reason to use
#: another one passes it.
DEFAULT_POCKET_SHELL_A: float = 6.0

#: Constraint name a catalytic template may use to state its own, sourced
#: pocket-localisation window in place of :data:`DEFAULT_POCKET_LOCALISATION_A`.
POCKET_LOCALISATION_CONSTRAINT_NAME: str = "pocket_localisation"

#: Nicotinamide C4 atom names per PDB chemical component, from the wwPDB
#: Chemical Component Dictionary. Used **only** as a fallback when a pose
#: binding does not name the hydride-donor atom, and only for components in
#: this table: an unlisted component yields an unresolved role -- and therefore
#: an unmeasured constraint -- never a guess at a nearby carbon.
NICOTINAMIDE_HYDRIDE_DONOR_ATOM: dict[str, str] = {
    "NAI": "C4N",   # 1,4-dihydronicotinamide adenine dinucleotide (NADH)
    "NDP": "C4N",   # 1,4-dihydro NADP (NADPH)
    "NAD": "C4N",   # NAD+ -- listed so the *absence* of a hydride stays locatable
    "NAP": "C4N",   # NADP+
    "NAJ": "C4N",
    "NDC": "C4N",
}

#: Scorecard key under which the "is this template even this candidate's
#: template" check is recorded. A gate rather than an axis: applying one
#: family's catalytic rules to another family's candidate is an input defect to
#: repair, not a weakness to trade off against a good docking score.
FAMILY_MATCH_GATE: str = "catalytic_template_family_match"

#: Role-token namespaces understood by :func:`build_role_context`.
NS_SUBSTRATE: str = "substrate"
NS_COFACTOR: str = "cofactor"
NS_PROTEIN: str = "protein"


# ==========================================================================
# Outcome taxonomy
# ==========================================================================

class WindowAuthority(str, enum.Enum):
    """Whether a geometric window is trustworthy enough to reject a candidate.

    The distinction exists because both alternatives are worse. Treating every
    template as authoritative rejects whole families against a theozyme someone
    sketched; treating every template as advisory means nothing is ever
    excluded and the gates are decorative. So the template says which it is,
    through :attr:`~eagent.schemas.templates.GeometryConstraint.calibrated_on`
    and :attr:`~eagent.schemas.templates.TemplateProvenance.source_type`, and
    only a calibrated window from a non-theoretical template may turn a
    measurement into a rejection.
    """

    CALIBRATED = "calibrated"
    UNCALIBRATED = "uncalibrated"
    THEORETICAL_MODEL = "theoretical_model"

    @property
    def may_reject(self) -> bool:
        """Only a window fitted to systems of known activity may disqualify."""
        return self is WindowAuthority.CALIBRATED

    def caveat(self) -> str:
        """One line a report can quote next to a verdict from this window."""
        return {
            WindowAuthority.CALIBRATED:
                "window calibrated on named systems",
            WindowAuthority.UNCALIBRATED:
                "window carries no calibration set; it may not drive a rejection, "
                "and a pass against it is not strong evidence",
            WindowAuthority.THEORETICAL_MODEL:
                "template is a labelled theoretical model (theozyme); its windows "
                "may not drive rejections and passes against them are provisional",
        }[self]


class PoseOutcome(str, enum.Enum):
    """What one modelled pose established, in the only five states allowed.

    The important property of this enum is what it cannot say. There is no
    member meaning "inactive", and :meth:`as_record_outcome` cannot return an
    experimental :class:`~eagent.schemas.record.OutcomeClass`. A modelling
    failure therefore cannot be written into the record layer as an
    experimental negative, which is the confusion that turns a crashed docking
    run into a published "no activity detected".
    """

    MECHANISM_SATISFIED = "mechanism_satisfied"
    MECHANISM_VIOLATED = "mechanism_violated"
    OUTSIDE_UNCALIBRATED_WINDOW = "outside_uncalibrated_window"
    NOT_MEASURABLE = "not_measurable"
    INPUT_ERROR = "input_error"

    @property
    def decided(self) -> bool:
        """Whether this pose belongs in the robustness denominator.

        Only the two outcomes resting on a window we trust. A pose outside a
        provisional window is deliberately excluded rather than counted as a
        failure: counting it would let an uncalibrated template drive G to zero
        for an entire family, which is the mass rejection this design forbids.
        """
        return self in (PoseOutcome.MECHANISM_SATISFIED,
                        PoseOutcome.MECHANISM_VIOLATED)

    @property
    def is_mechanism_negative(self) -> bool:
        """True only for a measured failure against a trusted window."""
        return self is PoseOutcome.MECHANISM_VIOLATED

    @property
    def is_modelling_failure(self) -> bool:
        """True when nothing about the enzyme was tested."""
        return self in (PoseOutcome.NOT_MEASURABLE, PoseOutcome.INPUT_ERROR)

    def as_record_outcome(self) -> OutcomeClass:
        """Map to the record layer, which may only receive non-experimental classes.

        The guard at the end is the point of the method. If a future edit maps a
        modelling outcome onto an experimental class, this raises instead of
        letting a computed verdict enter the dataset as a measurement.
        """
        mapping = {
            # A satisfied pose is a hypothesis, not an observation: nothing was
            # assayed, so the record layer must still read "not tested".
            PoseOutcome.MECHANISM_SATISFIED: OutcomeClass.NOT_TESTED,
            PoseOutcome.MECHANISM_VIOLATED: OutcomeClass.COMPUTATIONAL_NEGATIVE,
            PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW: OutcomeClass.NOT_TESTED,
            PoseOutcome.NOT_MEASURABLE: OutcomeClass.COMPUTATIONAL_FAILURE,
            PoseOutcome.INPUT_ERROR: OutcomeClass.COMPUTATIONAL_FAILURE,
        }
        out = mapping[self]
        if out.is_experimental:  # pragma: no cover - guard, must stay unreachable
            raise FabricationGuardError(
                f"{self.value} was mapped to the experimental outcome "
                f"{out.value}; a modelled pose is never an experimental result"
            )
        return out

    def claim(self) -> str:
        """The sentence this outcome entitles a report to write."""
        return {
            PoseOutcome.MECHANISM_SATISFIED:
                "this pose places the catalytic atoms inside the template's "
                "windows; a hypothesis about geometry, not a measured activity",
            PoseOutcome.MECHANISM_VIOLATED:
                "this pose falls outside a calibrated mechanism window; a "
                "computational negative about this pose, not about the enzyme",
            PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW:
                "this pose falls outside a window that carries no calibration; "
                "the window is not trusted enough to reject on",
            PoseOutcome.NOT_MEASURABLE:
                "the mechanism conditions could not be measured for this pose; "
                "nothing was tested and no negative may be recorded",
            PoseOutcome.INPUT_ERROR:
                "the inputs to this pose are wrong; repair them and re-run",
        }[self]


# ==========================================================================
# Pose bindings: which atom of the file plays which mechanistic role
# ==========================================================================

class ResidueSelector(BaseModel):
    """Which residue of a coordinate file a ligand role refers to.

    A selector rather than a bare component name because a file routinely holds
    two copies of the cofactor (one per protomer), and because the substrate's
    component id is whatever the pose builder called it. Ambiguity is reported,
    never resolved by taking the first match: measuring the hydride distance to
    the nicotinamide of the *other* protomer is a wrong answer that looks right.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chain: str | None = None
    resname: str | None = None
    resseq: int | None = None
    icode: str = ""

    def describe(self) -> str:
        """Readable form for a QC message, so a reviewer can re-run the selection."""
        seq = "*" if self.resseq is None else str(self.resseq)
        parts = [f"chain={self.chain or '*'}", f"resname={self.resname or '*'}",
                 f"resseq={seq}"]
        if self.icode.strip():
            parts.append(f"icode={self.icode.strip()}")
        return "(" + ", ".join(parts) + ")"

    def matches(self, residue: Residue) -> bool:
        """Whether one residue satisfies every stated criterion."""
        if self.chain is not None \
                and residue.chain.strip().upper() != self.chain.strip().upper():
            return False
        if self.resname is not None \
                and residue.resname.strip().upper() != self.resname.strip().upper():
            return False
        if self.resseq is not None and residue.resseq != self.resseq:
            return False
        if self.icode.strip() and residue.icode.strip() != self.icode.strip():
            return False
        return True

    def select(self, structure: Structure) -> list[Residue]:
        """Every matching residue. The caller decides what 0 or 2 matches mean."""
        return [r for r in structure.residues() if self.matches(r)]


class ProteinAtomRef(BaseModel):
    """One protein atom, addressed in **structure author numbering**.

    Author numbering, explicitly, because
    :attr:`~eagent.schemas.candidate.CatalyticMapping.role_to_residue` is in
    1-based *candidate-sequence* numbering and the two axes differ by whatever
    offset the construct and the crystallographer introduced. This module never
    converts between them: a caller holding only sequence numbering must run it
    through :class:`~eagent.science.numbering.ResidueMap` first. Measuring a
    catalytic distance to the residue at the wrong position is an off-by-N
    error that produces an entirely plausible number.

    ``expected_resname`` is verified when supplied. A mismatch leaves the role
    unresolved with a reason, rather than measuring to whatever residue happens
    to occupy that author position -- which is the signature of exactly the
    numbering confusion above.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chain: str
    resseq: int
    atom: str
    icode: str = ""
    expected_resname: str | None = None

    def describe(self) -> str:
        """Readable handle used in unresolved-role reasons."""
        return (f"{self.chain}/{self.expected_resname or '???'}{self.resseq}"
                f"{self.icode.strip()}/{self.atom}")


class PoseBinding(BaseModel):
    """The map from mechanistic role tokens to atoms of one pose's file.

    WHY THIS IS AN INPUT AND NOT AN INFERENCE
    -----------------------------------------
    A catalytic template names atoms by role: ``cofactor.hydride_donor_C4``,
    ``substrate.electrophile``, ``protein.catalytic_Tyr.OH``. Turning those into
    coordinates needs knowledge this step does not have and must not invent:

    * :class:`~eagent.schemas.chem.ReactiveAtoms` addresses substrate atoms by
      *atom-map id in the mapped reaction SMILES*. No coordinate file carries
      that id. Only the step that generated the pose knows which atom of the
      file is atom-map 3, so the correspondence arrives here as
      ``substrate_atoms``. Without it, substrate roles are unresolved, every
      constraint naming them is **unmeasured** (``None``), and the pose is
      reported as a modelling gap -- which is honest, and is precisely the state
      a naive implementation papers over by taking "the nearest oxygen".
    * Protein roles arrive as :class:`ProteinAtomRef` in author numbering, for
      the reason given there.

    Anything left unbound stays unbound. There is no best-guess branch.
    """

    model_config = ConfigDict(extra="forbid")

    pose_id: str
    structure_path: str | None = Field(
        None, description="Overrides ComplexPose.path when the pose file moved."
    )
    substrate: ResidueSelector | None = None
    substrate_atoms: dict[str, str] = Field(
        default_factory=dict,
        description="Role key (the part after 'substrate.') -> atom name in the file.",
    )
    cofactor: ResidueSelector | None = None
    cofactor_atoms: dict[str, str] = Field(default_factory=dict)
    protein_atoms: dict[str, ProteinAtomRef] = Field(
        default_factory=dict,
        description="Role key (the part after 'protein.') -> atom in author numbering.",
    )
    competing_electrophiles: dict[str, str] = Field(
        default_factory=dict,
        description="Label -> atom name of a non-target electrophilic atom of the "
                    "substrate (a second ketone, an aldehyde, an ester carbonyl). "
                    "An empty map means chemoselectivity was not tested, which is "
                    "not the same as its having passed.",
    )
    reactive_atom_role: str = Field(
        "electrophile",
        description="Key into substrate_atoms for the atom the reaction attacks.",
    )
    carbonyl_oxygen_role: str = "carbonyl_O"
    prochiral_substituent_roles: list[str] = Field(
        default_factory=list,
        description="The two carbon substituents of the prochiral carbonyl carbon, "
                    "as keys into substrate_atoms. Required for a face call, and "
                    "the same keys must carry CIP ranks.",
    )
    hydride_donor_role: str = "hydride_donor_C4"
    notes: str = ""


@dataclass
class RoleContextResult:
    """Namespaces for a :class:`~eagent.science.geometry.RoleResolver`, plus gaps.

    ``problems`` and ``misses`` are returned alongside the context rather than
    logged, because "three of eight constraints measured" and "five constraints
    failed" look identical in a report that does not say which roles could not
    be bound -- and they mean opposite things about the candidate.
    """

    structure: Structure
    namespaces: dict[str, Any] = field(default_factory=dict)
    misses: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    substrate_residue: Residue | None = None
    cofactor_residue: Residue | None = None
    protein_atoms: dict[str, Atom] = field(default_factory=dict)
    substrate_role_atoms: dict[str, Atom] = field(default_factory=dict)
    cofactor_role_atoms: dict[str, Atom] = field(default_factory=dict)

    def resolver(self) -> geom.RoleResolver:
        """A caching resolver over the three namespaces."""
        return geom.RoleResolver(self.namespaces)

    def gaps(self) -> dict[str, str]:
        """Every unbindable role and selection problem, keyed for a report."""
        out = dict(self.misses)
        for i, problem in enumerate(self.problems):
            out[f"selection[{i}]"] = problem
        return out


class _AtomNamespace:
    """A role-key -> :class:`Atom` lookup that records *why* a key missed.

    :func:`eagent.science.geometry.resolve_role` already returns a reason, but
    it can only say "the namespace has no atom for this role". This records the
    specific reason -- the residue is absent from the file, the atom name is not
    in the residue, the binding never named it -- which is the difference
    between re-running the pose builder and fixing a typo in a template.
    """

    def __init__(self, label: str, atoms: Mapping[str, Atom],
                 reasons: Mapping[str, str], sink: dict[str, str]) -> None:
        self.label = label
        self._atoms = dict(atoms)
        self._reasons = dict(reasons)
        self._sink = sink

    def __call__(self, key: str) -> Atom | None:
        hit = self._atoms.get(key)
        if hit is None:
            self._sink.setdefault(
                f"{self.label}.{key}",
                self._reasons.get(
                    key,
                    f"no atom is bound to role '{key}' in namespace '{self.label}'",
                ),
            )
        return hit


# ==========================================================================
# Template authority and windows
# ==========================================================================

def template_authority(template: CatalyticTemplate,
                       calibration: CalibrationContext | None = None
                       ) -> WindowAuthority:
    """Whole-template authority, used for the per-candidate confidence downgrade.

    A theoretical-model template is provisional however well calibrated an
    individual constraint claims to be, because the coordinates the window was
    fitted against were themselves modelled.
    """
    if template.provenance.is_theoretical:
        return WindowAuthority.THEORETICAL_MODEL
    context = calibration or CalibrationContext()
    if any(not context.status(c).calibrated for c in template.geometry_constraints):
        return WindowAuthority.UNCALIBRATED
    return WindowAuthority.CALIBRATED


def constraint_authority(constraint: GeometryConstraint,
                         template: CatalyticTemplate,
                         calibration: CalibrationContext | None = None,
                         ) -> WindowAuthority:
    """Authority of one window. Per-constraint, because real templates are mixed.

    A working family template usually has one or two windows fitted on a handful
    of crystal complexes and several more written down from a mechanism review.
    Judging the whole template at the level of its weakest constraint would
    throw away the calibrated ones; judging it at the level of its strongest
    would let an uncalibrated angle reject candidates.

    ``calibrated_on`` used to be free text, so one hand-typed string granted a
    window the power to reject enzymes. A ``calibration:`` entry is now
    checked against a stored record (see :mod:`eagent.science.calibration`),
    and fails closed -- to uncalibrated -- whenever it cannot be: no store, no
    such record, an edited record, a window changed since, a verdict that no
    longer meets its policy. Prose entries keep their old meaning unless the
    context is strict.
    """
    if template.provenance.is_theoretical:
        return WindowAuthority.THEORETICAL_MODEL
    context = calibration or CalibrationContext()
    if not context.status(constraint).calibrated:
        return WindowAuthority.UNCALIBRATED
    return WindowAuthority.CALIBRATED


def pocket_localisation_window(
    template: CatalyticTemplate | None,
    override_angstrom: float | None = None,
) -> tuple[float, str]:
    """``(max distance, where it came from)`` for the localisation QC screen.

    Preference order, and the reason for it: a template-supplied constraint
    carries its own provenance and calibration set; an operator override is at
    least a deliberate, recorded choice; the module default is a last resort,
    and the caller is told it was used so the fact reaches the report instead of
    nobody's head.
    """
    if template is not None:
        for constraint in template.geometry_constraints:
            if constraint.name != POCKET_LOCALISATION_CONSTRAINT_NAME:
                continue
            _, hi = constraint.window()
            if not math.isfinite(hi):
                continue
            if constraint.is_calibrated:
                calib = "calibrated on " + ", ".join(constraint.calibrated_on)
            else:
                calib = "uncalibrated"
            label = (f"template:{template.template_id}:{constraint.name} "
                     f"[{calib}]")
            return hi, label
    if override_angstrom is not None:
        return float(override_angstrom), "operator override"
    return (
        DEFAULT_POCKET_LOCALISATION_A,
        "module default DEFAULT_POCKET_LOCALISATION_A (needs per-family "
        "calibration; a QC screen, not a catalytic criterion)",
    )


def classify_aspect(constraint: GeometryConstraint) -> str:
    """Which mechanistic question a constraint belongs to. Reporting only.

    Derived from the role tokens rather than from the constraint's name, because
    names are free text and a template author's ``d1`` would otherwise vanish
    from the summary. This classification has **no** bearing on pass/fail; it
    exists so a reviewer can see at a glance that, say, every satisfied
    constraint concerned cofactor placement and none concerned the hydride
    trajectory.
    """
    tokens = [t for t in (constraint.atom_a, constraint.atom_b,
                          constraint.atom_c, constraint.atom_d) if t]
    spaces = {t.split(".", 1)[0] for t in tokens}
    keys = " ".join(t.split(".", 1)[1].lower() for t in tokens if "." in t)

    donor = ("hydride" in keys) or ("c4" in keys)
    oxygen = ("carbonyl_o" in keys) or ("oxyanion" in keys) or ("stabilis" in keys)
    if donor and NS_SUBSTRATE in spaces:
        return "hydride_transfer" if constraint.kind == "distance" \
            else "approach_trajectory"
    if NS_PROTEIN in spaces and NS_SUBSTRATE in spaces and oxygen:
        return "oxyanion_stabilisation"
    if NS_PROTEIN in spaces and NS_SUBSTRATE in spaces:
        return "substrate_contact"
    if NS_COFACTOR in spaces and NS_PROTEIN in spaces:
        return "cofactor_placement"
    if spaces == {NS_COFACTOR}:
        return "cofactor_internal"
    if spaces == {NS_SUBSTRATE}:
        return "substrate_internal"
    return "other"


# ==========================================================================
# Per-pose findings
# ==========================================================================

@dataclass(frozen=True)
class PocketLocalisation:
    """Coarse "is the substrate in the catalytic region" screen.

    Explicitly labelled a QC result. ``within=False`` asks a human to look at the
    pose; it is not a statement that the enzyme cannot catalyse the reaction,
    and nothing in this module rejects a pose on it.

    ``missing_references`` is the half of the reference set that could not be
    built. It exists because the alternative -- substituting whatever atom
    happened to be bound under the bare residue label -- measures to an alpha
    carbon and then labels the result with the template's functional atom. The
    distance is then real, the label is wrong, and nothing downstream can tell:
    exactly the sloppy distance definition this screen was specified to avoid.
    A reference that cannot be built is left out and listed here instead.
    """

    distance_A: float | None
    threshold_A: float
    threshold_source: str
    nearest_role: str | None = None
    reference_roles: tuple[str, ...] = ()
    missing_references: tuple[str, ...] = ()
    reason: str = ""

    @property
    def within(self) -> bool | None:
        """``None`` when the screen could not run -- never ``False``."""
        if self.distance_A is None:
            return None
        return self.distance_A <= self.threshold_A


@dataclass(frozen=True)
class ChemoselectivityCheck:
    """Is the *target* group the one sitting in the reactive position?

    The failure this catches is specific and common: a substrate with two
    electrophilic carbons (a keto-ester, a diketone, an aldehyde beside a
    ketone) docks with the wrong one presented to the nicotinamide. Every
    distance and angle then measures beautifully, the pose satisfies every
    constraint, and the product is not the target product. A pipeline that
    measures "the carbonyl" cannot see this, because it never asked *which*
    carbonyl.

    ``tested=False`` means no competing electrophile was declared. That is not a
    pass; it is an untested question, and it is reported as one.
    """

    tested: bool
    target_distance_A: float | None = None
    displacing_label: str | None = None
    displacing_distance_A: float | None = None
    margin_A: float = DEFAULT_COMPETING_GROUP_MARGIN_A
    #: Whether the margin this comparison used was fitted to anything.
    #:
    #: The comparison is two distances and a margin, and the margin is the
    #: whole content of it: both distances carry the pose generator's
    #: positional scatter, so the question "is the competitor closer" has no
    #: answer until somebody says how much closer counts. The module default
    #: is a placeholder that says so in its own docstring, and a rejection
    #: resting on it is a rejection resting on a number nobody measured.
    margin_authority: WindowAuthority = WindowAuthority.UNCALIBRATED
    #: Where the margin came from, for the record.
    margin_source: str = ""
    reason: str = ""

    @property
    def target_in_reactive_position(self) -> bool | None:
        """``True`` / ``False`` / ``None`` when the comparison was impossible."""
        if not self.tested or self.target_distance_A is None:
            return None
        return self.displacing_label is None

    @property
    def displaced_on_a_calibrated_margin(self) -> bool:
        """A displacement that may reject a pose.

        Separate from :attr:`target_in_reactive_position` so that the measured
        fact and the authority to act on it stay apart: the displacement is
        reported either way, and only this one rejects.
        """
        return (self.target_in_reactive_position is False
                and self.margin_authority.may_reject)


@dataclass(frozen=True)
class CofactorCheck:
    """Right cofactor, right oxidation state, actually present in the file.

    The oxidation state is the trap. NADP+ and NADPH differ by two electrons and
    a proton, share a name stem and adjacent PDB component codes, and are
    routinely swapped when a pose is built by transplanting the ligand from
    whatever crystal structure was available -- and dehydrogenase structures are
    very often the oxidised, product-complex form. A hydride-transfer distance
    measured to NAD+ is a distance to a nicotinamide with no hydride to give.
    That is an input defect, not a geometry failure, and it is classified as one
    here so it cannot be traded off against a good score on another axis.
    """

    required_identity: str | None
    required_state: CofactorState
    present: bool
    observed_identity: str | None = None
    observed_state: CofactorState = CofactorState.UNKNOWN
    observed_component: str | None = None
    state_source: str = ""
    placement_constraints: tuple[str, ...] = ()
    placement_satisfied: bool | None = None

    @property
    def identity_ok(self) -> bool | None:
        """``None`` when either side is unknown: unknown is unchecked, not wrong."""
        if self.required_identity is None or self.observed_identity is None:
            return None
        return self.observed_identity == self.required_identity

    @property
    def state_ok(self) -> bool | None:
        """``None`` when either state is UNKNOWN; never assumed to be the right one."""
        if self.required_state is CofactorState.UNKNOWN \
                or self.observed_state is CofactorState.UNKNOWN:
            return None
        return self.observed_state is self.required_state

    def unresolved_requirements(self) -> list[str]:
        """Cofactor questions that were never answered, as opposed to answered badly.

        An unknown oxidation state is the important one. The mechanism needs a
        hydride; a nicotinamide whose state nobody recorded may or may not have
        one. Treating "unknown" as "reduced" is the substitution this harness
        forbids, so a pose whose geometry is otherwise perfect is reported as
        unmeasured rather than promoted to satisfied. The asymmetry is
        deliberate: an unresolved input may block a promotion, and never causes
        a rejection.
        """
        out: list[str] = []
        if self.required_state in (CofactorState.REDUCED, CofactorState.OXIDIZED) \
                and self.state_ok is None:
            out.append(
                f"the cofactor's oxidation state could not be established "
                f"({self.state_source}), so whether a hydride is available to "
                f"transfer was never tested; it was not assumed to be "
                f"{self.required_state.value}"
            )
        if self.required_identity is not None and self.present \
                and self.identity_ok is None:
            out.append(
                f"the cofactor in this pose (component {self.observed_component}) "
                f"could not be resolved to a known identity, so it was not "
                f"compared with the required {self.required_identity}"
            )
        return out

    def input_defects(self) -> list[str]:
        """Defects that disqualify this pose until the inputs are repaired."""
        out: list[str] = []
        if self.identity_ok is False:
            out.append(
                f"pose carries cofactor {self.observed_identity} "
                f"(component {self.observed_component}) but the mechanism requires "
                f"{self.required_identity}"
            )
        if self.state_ok is False:
            out.append(
                f"pose carries {self.observed_identity or 'the cofactor'} in the "
                f"{self.observed_state.value} state (component "
                f"{self.observed_component}, {self.state_source}) but the mechanism "
                f"requires the {self.required_state.value} form; the oxidised "
                f"nicotinamide has no hydride to transfer, so every "
                f"hydride-transfer measurement on this pose is meaningless"
            )
        return out


@dataclass(frozen=True)
class ClashScreen:
    """Hard-sphere overlap count, with an explicit statement of coverage.

    ``complete=False`` means some atom had no tabulated van der Waals radius and
    was not screened, so a count of 0 would otherwise be ambiguous between "no
    clashes" and "nothing was tested" -- the ambiguity
    :func:`eagent.science.geometry.clash_count` raises about by default.
    """

    count: int | None
    tolerance_A: float
    complete: bool = True
    unscreened_elements: tuple[str, ...] = ()
    reason: str = ""


@dataclass
class PoseEvaluation:
    """Everything this step established about one candidate-pose pair."""

    candidate_id: str
    pose_id: str
    method: str
    outcome: PoseOutcome
    reason: str = ""
    template_id: str | None = None
    authority: WindowAuthority = WindowAuthority.CALIBRATED
    measurements: dict[str, float | None] = field(default_factory=dict)
    satisfied: dict[str, bool | None] = field(default_factory=dict)
    aspects: dict[str, str] = field(default_factory=dict)
    authorities: dict[str, WindowAuthority] = field(default_factory=dict)
    gating_passed: bool | None = None
    hard_failures: tuple[str, ...] = ()
    provisional_failures: tuple[str, ...] = ()
    unmeasured: tuple[str, ...] = ()
    circular_constraints: tuple[str, ...] = ()
    circular_satisfied: tuple[str, ...] = ()
    independent_satisfied: int = 0
    independent_total: int = 0
    #: Gating constraints that were NOT restrained while this pose was built.
    #: Kept apart from ``independent_total`` (which counts every evaluated
    #: constraint, gating or scoring) because only the gating ones decide the
    #: outcome, and so only they can corroborate it.
    independent_gating: tuple[str, ...] = ()
    independent_gating_satisfied: tuple[str, ...] = ()
    restrained_gating: tuple[str, ...] = ()
    entirely_circular: bool = False
    restraint_name_mismatch: tuple[str, ...] = ()
    pocket: PocketLocalisation | None = None
    chemoselectivity: ChemoselectivityCheck | None = None
    cofactor: CofactorCheck | None = None
    clash: ClashScreen | None = None
    face: str | None = None
    product_configuration: str | None = None
    stereo_note: str = ""
    unresolved_roles: dict[str, str] = field(default_factory=dict)
    input_errors: tuple[str, ...] = ()

    @property
    def carries_independent_test(self) -> bool:
        """Whether this pose could corroborate anything it was not built to show.

        A pose every one of whose gating constraints was enforced during
        modelling was never a test: it satisfies them because it was built to,
        and re-measuring them is a readback of the input file. Such a pose is
        excluded from the robustness denominator entirely rather than counted
        as a failure, for the same reason a pose outside an uncalibrated window
        is excluded -- it did not answer the question either way.
        """
        return self.outcome.decided and bool(self.independent_gating)

    @property
    def independently_satisfied(self) -> bool:
        """Numerator test: satisfied, and satisfied on evidence nobody imposed.

        Requires at least one independent gating constraint and *all* of them
        satisfied. This is the predicate that makes "circularity-corrected"
        true of :attr:`CandidateEvaluation.robustness_G` rather than merely
        advertised by it.
        """
        return (self.outcome is PoseOutcome.MECHANISM_SATISFIED
                and bool(self.independent_gating)
                and len(self.independent_gating_satisfied)
                == len(self.independent_gating))

    def to_report(self) -> GeometryReport:
        """Render as the schema object downstream steps consume.

        ``gating_passed`` stays three-valued for the same reason it is
        three-valued in the schema: an unevaluated gate is routed to an
        uncertainty, while a failed one removes the candidate. Writing ``False``
        for a pose that could not be measured would silently convert a modelling
        gap into a rejection.
        """
        notes = [f"{self.outcome.value}: {self.reason or self.outcome.claim()}"]
        if self.authority is not WindowAuthority.CALIBRATED:
            notes.append(self.authority.caveat())
        if self.unresolved_roles:
            notes.append("unresolved: " + "; ".join(
                f"{k} ({v})" for k, v in sorted(self.unresolved_roles.items())))
        return GeometryReport(
            pose_id=self.pose_id,
            measurements=dict(self.measurements),
            satisfied=dict(self.satisfied),
            gating_passed=self.gating_passed,
            circular_constraints=list(self.circular_constraints),
            independent_satisfied=self.independent_satisfied,
            independent_total=self.independent_total,
            clash_count=None if self.clash is None else self.clash.count,
            notes=" | ".join(notes),
        )


@dataclass
class CandidateEvaluation:
    """Pose evaluations rolled up to the candidate, without a total score."""

    candidate_id: str
    template_id: str | None
    authority: WindowAuthority
    #: False when no catalytic template could be resolved at all. Kept apart
    #: from ``authority`` because "the window is not calibrated" and "there is
    #: no window" call for different follow-ups: one needs a calibration set,
    #: the other needs a curated template.
    has_template: bool = True
    #: Whether the resolved template's ``family_name`` is this candidate's
    #: family. ``True`` matched, ``False`` refused (a different family's
    #: mechanism), ``None`` the question could not be asked -- the candidate's
    #: family is unknown, or no template was resolved. Three-valued because an
    #: unknown family is an unevaluated gate, not a mismatch, and conflating
    #: the two would either reject every unannotated candidate or wave it
    #: through as though the template had been checked.
    family_match: bool | None = None
    family_note: str = ""
    poses: list[PoseEvaluation] = field(default_factory=list)
    #: Circularity-corrected robustness. See :meth:`EvaluateCatalysis._rollup`
    #: for the numerator, the denominator and why an ensemble built entirely
    #: under restraints reports ``None`` here rather than 1.0.
    robustness_G: float | None = None
    #: Competent poses whose every gating constraint was restrained while
    #: they were built, and which therefore did not vote on the
    #: stereochemical call. Reported rather than silently dropped: "no
    #: independent pose had an opinion" and "no pose was competent" lead to
    #: different next experiments.
    stereo_poses_excluded_as_circular: int = 0
    #: The uncorrected sampling fraction (satisfied / decided), kept because it
    #: describes how the sampler behaved and because hiding it would make the
    #: correction unauditable. It is NOT evidence about the enzyme when the
    #: constraints it counts were imposed, which is the whole point of
    #: reporting the two numbers in two fields under two names.
    sampling_G: float | None = None
    robustness_basis: str = ""
    wilson_lo: float | None = None
    wilson_hi: float | None = None
    robustness_level: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT
    stereo: StereoCall = field(default_factory=StereoCall)
    input_errors: list[str] = field(default_factory=list)
    explanation: str = ""

    # -- the counts that must never be added together ----------------------
    @property
    def n_poses(self) -> int:
        """Poses attempted. Never the denominator of anything."""
        return len(self.poses)

    @property
    def n_decided(self) -> int:
        """Poses that produced a verdict against a trusted window."""
        return sum(1 for p in self.poses if p.outcome.decided)

    @property
    def n_satisfied(self) -> int:
        """Poses inside every gating window of the template."""
        return sum(1 for p in self.poses
                   if p.outcome is PoseOutcome.MECHANISM_SATISFIED)

    @property
    def n_violated(self) -> int:
        """Measured failures against a calibrated window -- the only negatives."""
        return sum(1 for p in self.poses
                   if p.outcome is PoseOutcome.MECHANISM_VIOLATED)

    @property
    def n_provisional(self) -> int:
        """Measured failures against a window not trusted enough to reject on."""
        return sum(1 for p in self.poses
                   if p.outcome is PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW)

    @property
    def n_not_measurable(self) -> int:
        """Modelling failures. Kept in their own field so they cannot be summed
        into the negatives by a downstream reader."""
        return sum(1 for p in self.poses
                   if p.outcome is PoseOutcome.NOT_MEASURABLE)

    @property
    def n_input_error(self) -> int:
        """Poses built on wrong inputs. Repairable; never an enzyme property."""
        return sum(1 for p in self.poses if p.outcome is PoseOutcome.INPUT_ERROR)

    @property
    def n_independently_tested(self) -> int:
        """Decided poses that carried at least one unrestrained gating constraint.

        The denominator of :attr:`robustness_G`. Poses whose every gating
        constraint was imposed are not in it: they are not failures, they are
        non-tests.
        """
        return sum(1 for p in self.poses if p.carries_independent_test)

    @property
    def n_independently_satisfied(self) -> int:
        """The numerator of :attr:`robustness_G`."""
        return sum(1 for p in self.poses if p.independently_satisfied)

    @property
    def was_tested(self) -> bool:
        """Whether any pose produced a verdict at all.

        Read this before reading :attr:`robustness_G`, and note that the two
        now distinguish three states rather than two:

        * ``was_tested=False``, ``robustness_G=None`` -- the pipeline failed on
          this candidate. Nothing was measured; it must not be ranked, plotted
          or averaged beside a candidate whose G is a genuine 0.0.
        * ``was_tested=True``, ``robustness_G=None`` -- poses were measured and
          decided, but no decided pose carried a gating constraint that had not
          been restrained during modelling. There is a sampling fraction
          (:attr:`sampling_G`) and there is no independent evidence, so no
          corrected G exists. This is the state a restrained ensemble used to
          report as ``G = 1.0``.
        * ``was_tested=True`` with a number -- a corrected fraction over the
          poses that could actually have come out otherwise.
        """
        return self.n_decided > 0

    def entirely_circular_poses(self) -> list[str]:
        """Pose ids whose every satisfied constraint had been imposed."""
        return [p.pose_id for p in self.poses if p.entirely_circular]


# ==========================================================================
# Role context construction
# ==========================================================================

def _pick_residue(
    structure: Structure, selector: ResidueSelector, what: str,
) -> tuple[Residue | None, str]:
    """Exactly one match, or a reason. Never "the first one"."""
    hits = selector.select(structure)
    if not hits:
        return None, (f"no residue in {structure.structure_id} matches the {what} "
                      f"selector {selector.describe()}")
    if len(hits) > 1:
        shown = ", ".join(str(r) for r in hits[:4])
        return None, (f"{len(hits)} residues match the {what} selector "
                      f"{selector.describe()} ({shown}); the selector must name one "
                      f"copy, because measuring to the wrong protomer's ligand is a "
                      f"wrong answer that looks right")
    return hits[0], ""


def build_role_context(
    structure: Structure,
    binding: PoseBinding,
    template: CatalyticTemplate | None = None,
    *,
    infer_single_ligand_substrate: bool = True,
) -> RoleContextResult:
    """Turn a pose binding plus a parsed structure into resolver namespaces.

    Every lookup that fails leaves a *named reason* behind instead of a
    fallback. The three namespaces are built independently, so a pose with a
    bound cofactor and an unbound substrate measures its cofactor-internal
    constraints and reports the substrate ones as unmeasured -- the distinction
    :func:`eagent.science.geometry.measure_constraint` exists to preserve.

    ``infer_single_ligand_substrate`` covers the ordinary case where the pose
    file holds exactly one non-water, non-cofactor polyatomic ligand. Choosing
    the only candidate is a selection, not a guess, and it is recorded in
    ``problems`` so that it reaches provenance; with zero or several such
    residues the substrate stays unbound and the reason says which.
    """
    problems: list[str] = []
    reasons_sub: dict[str, str] = {}
    reasons_cof: dict[str, str] = {}
    reasons_pro: dict[str, str] = {}
    misses: dict[str, str] = {}

    # -- cofactor residue ---------------------------------------------------
    cof_res: Residue | None = None
    if binding.cofactor is not None:
        cof_res, why = _pick_residue(structure, binding.cofactor, "cofactor")
        if why:
            problems.append(why)
    elif template is not None and template.cofactor_ligand_codes:
        codes = {c.strip().upper() for c in template.cofactor_ligand_codes}
        hits = [r for r in structure.ligands() if r.resname.strip().upper() in codes]
        if len(hits) == 1:
            cof_res = hits[0]
        elif not hits:
            problems.append(
                f"no residue in {structure.structure_id} carries one of the "
                f"template's cofactor component ids {sorted(codes)}; the cofactor "
                f"is absent from this pose"
            )
        else:
            shown = ", ".join(str(r) for r in hits[:4])
            problems.append(
                f"{len(hits)} residues carry a template cofactor component id "
                f"({shown}); bind one explicitly"
            )
    else:
        problems.append(
            "no cofactor selector in the binding and no cofactor_ligand_codes in "
            "the template, so the cofactor could not be located"
        )

    # -- substrate residue --------------------------------------------------
    sub_res: Residue | None = None
    if binding.substrate is not None:
        sub_res, why = _pick_residue(structure, binding.substrate, "substrate")
        if why:
            problems.append(why)
    elif infer_single_ligand_substrate:
        cof_key = cof_res.key if cof_res is not None else None
        others = [r for r in structure.ligands()
                  if r.key != cof_key and len(r.heavy_atoms()) > 1]
        if len(others) == 1:
            sub_res = others[0]
            problems.append(
                f"substrate not bound explicitly; selected the only non-cofactor "
                f"polyatomic ligand residue {sub_res} in {structure.structure_id}"
            )
        elif not others:
            problems.append(
                "no non-cofactor ligand residue in this pose, so the substrate "
                "could not be located; the file may have been written without it"
            )
        else:
            shown = ", ".join(str(r) for r in others[:4])
            problems.append(
                f"{len(others)} non-cofactor ligand residues ({shown}); the "
                f"substrate must be named in the binding rather than chosen here"
            )
    else:
        problems.append("no substrate selector supplied and inference is disabled")

    # -- substrate atoms ----------------------------------------------------
    sub_atoms: dict[str, Atom] = {}
    if sub_res is None:
        for key in binding.substrate_atoms:
            reasons_sub[key] = "the substrate residue itself could not be located"
    else:
        if not binding.substrate_atoms:
            problems.append(
                "the binding names no substrate atom roles. Atom-map ids from the "
                "mapped reaction SMILES do not appear in a coordinate file, so the "
                "pose builder must supply the correspondence; without it every "
                "substrate constraint is unmeasured, not failed"
            )
        for key, atom_name in binding.substrate_atoms.items():
            atom = sub_res.atom(atom_name)
            if atom is None:
                reasons_sub[key] = f"residue {sub_res} has no atom named '{atom_name}'"
            else:
                sub_atoms[key] = atom

    # -- cofactor atoms -----------------------------------------------------
    cof_atoms: dict[str, Atom] = {}
    donor_key = binding.hydride_donor_role
    if cof_res is None:
        for key in list(binding.cofactor_atoms) + [donor_key]:
            reasons_cof[key] = "the cofactor residue itself could not be located"
    else:
        wanted = dict(binding.cofactor_atoms)
        if donor_key not in wanted:
            component = cof_res.resname.strip().upper()
            fallback = NICOTINAMIDE_HYDRIDE_DONOR_ATOM.get(component)
            if fallback is not None:
                wanted[donor_key] = fallback
                problems.append(
                    f"hydride-donor atom not bound; used the wwPDB Chemical "
                    f"Component Dictionary atom name '{fallback}' for component "
                    f"{component}"
                )
            else:
                reasons_cof[donor_key] = (
                    f"component {component} is not in "
                    f"NICOTINAMIDE_HYDRIDE_DONOR_ATOM, so the donor atom name is "
                    f"unknown and will not be guessed"
                )
        for key, atom_name in wanted.items():
            atom = cof_res.atom(atom_name)
            if atom is None:
                reasons_cof[key] = f"residue {cof_res} has no atom named '{atom_name}'"
            else:
                cof_atoms[key] = atom

    # -- protein atoms ------------------------------------------------------
    pro_atoms: dict[str, Atom] = {}
    if not binding.protein_atoms:
        problems.append(
            "the binding names no protein atom roles. CatalyticMapping is in "
            "1-based candidate-sequence numbering and this step will not convert "
            "it to author numbering; run eagent.science.numbering.ResidueMap and "
            "supply ProteinAtomRef entries"
        )
    for key, ref in binding.protein_atoms.items():
        hits = ResidueSelector(chain=ref.chain, resseq=ref.resseq,
                               icode=ref.icode).select(structure)
        if not hits:
            reasons_pro[key] = (
                f"author position {ref.chain}/{ref.resseq}{ref.icode.strip()} is not "
                f"present in {structure.structure_id}")
            continue
        if len(hits) > 1:
            reasons_pro[key] = (
                f"author position {ref.chain}/{ref.resseq} carries more than one "
                f"component name; the handle is ambiguous in this file")
            continue
        residue = hits[0]
        if ref.expected_resname is not None and \
                residue.resname.strip().upper() != ref.expected_resname.strip().upper():
            reasons_pro[key] = (
                f"author position {ref.describe()} holds "
                f"{residue.resname.strip()}, not the expected {ref.expected_resname}; "
                f"this is the off-by-N signature of sequence numbering used as "
                f"author numbering")
            continue
        atom = residue.atom(ref.atom)
        if atom is None:
            reasons_pro[key] = f"residue {residue} has no atom named '{ref.atom}'"
            continue
        pro_atoms[key] = atom

    # Seed the miss sink with every gap already known, rather than only with
    # the ones a constraint happens to ask about. A role nobody measured is
    # still a role that could not be bound, and leaving it out of the report
    # would make a thin template look like a complete evaluation.
    for namespace, reasons in ((NS_SUBSTRATE, reasons_sub),
                               (NS_COFACTOR, reasons_cof),
                               (NS_PROTEIN, reasons_pro)):
        for key, why in reasons.items():
            misses.setdefault(f"{namespace}.{key}", why)

    namespaces: dict[str, Any] = {
        NS_SUBSTRATE: _AtomNamespace(NS_SUBSTRATE, sub_atoms, reasons_sub, misses),
        NS_COFACTOR: _AtomNamespace(NS_COFACTOR, cof_atoms, reasons_cof, misses),
        NS_PROTEIN: _AtomNamespace(NS_PROTEIN, pro_atoms, reasons_pro, misses),
    }
    return RoleContextResult(
        structure=structure,
        namespaces=namespaces,
        misses=misses,
        problems=problems,
        substrate_residue=sub_res,
        cofactor_residue=cof_res,
        protein_atoms=pro_atoms,
        substrate_role_atoms=sub_atoms,
        cofactor_role_atoms=cof_atoms,
    )


# ==========================================================================
# TSV helpers (local on purpose: this module owns its artifact format)
# ==========================================================================

def _tsv_cell(value: Any) -> str:
    """Render one cell. An empty cell means "not measured", never zero."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, enum.Enum):
        return str(value.value)
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, set, frozenset)):
        return ";".join(_tsv_cell(v) for v in value)
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def _write_tsv(path: Path, header: Sequence[str],
               rows: Iterable[Sequence[Any]]) -> Path:
    """Write a tab-separated table, one row per record, no index column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for row in rows:
            fh.write("\t".join(_tsv_cell(c) for c in row) + "\n")
    return path


# ==========================================================================
# The interface
# ==========================================================================

class EvaluateCatalysis(ScientificInterface):
    """Mechanism-aware catalytic feasibility for every candidate-pose pair.

    One pass over the poses answers, per pose and against the family's sourced
    catalytic template: is the substrate localised in the catalytic region (QC
    only); is the *target* group the one in the reactive position; is the
    hydride-donor-to-carbonyl-carbon arrangement inside the template's windows;
    is the carbonyl oxygen engaged by the family's stabilising residues; does
    the trajectory correspond to the target configuration; is the cofactor the
    right molecule in the right oxidation state and correctly placed; are there
    steric clashes; and does any of it survive across poses once the restraints
    that were *imposed* during modelling are removed from the evidence.

    What it refuses to do, and why
    ------------------------------
    * **No total score.** The scorecard is gates plus ordinal levels; see
      :mod:`eagent.science.scorecard`.
    * **No numeric ee.** :class:`~eagent.schemas.candidate.StereoCall` demands a
      named calibration source for one, this interface never supplies one, and
      :meth:`_guard_no_uncalibrated_ee` re-checks before the result leaves.
    * **No hard rejection on an uncalibrated window.** See
      :class:`WindowAuthority`.
    * **No network, and no sequence disclosure.** See
      :meth:`_refuse_external_submission`.
    """

    name: ClassVar[str] = "evaluate_catalysis"
    description: ClassVar[str] = (
        "Per-pose mechanism geometry against a sourced catalytic template, with "
        "pocket-localisation QC, chemoselectivity and cofactor-state checks, "
        "circularity-corrected robustness (G counted only over gating "
        "constraints that were not restrained during modelling, and undefined "
        "rather than 1.0 when none were), a directional stereochemical call and "
        "a per-candidate scorecard with no total."
    )
    required_fields: ClassVar[tuple[str, ...]] = (
        "reaction.substrate.isomeric_smiles",
        "reaction.product.creates_new_stereocenter",
    )
    required_approvals: ClassVar[tuple[str, ...]] = ("reaction_spec_confirmed",)
    depends_on: ClassVar[tuple[str, ...]] = ("annotate_family", "model_complexes")
    version: ClassVar[str] = "0.1.0"

    # -- entry point --------------------------------------------------------
    def execute(
        self,
        ctx: RunContext,
        *,
        candidates: Sequence[Candidate] | None = None,
        bindings: Mapping[str, PoseBinding] | Sequence[PoseBinding] | None = None,
        structures: Mapping[str, Structure] | None = None,
        catalytic_templates: Mapping[str, CatalyticTemplate] | CatalyticTemplate | None = None,
        cip_ranks: Any = None,
        pocket_localisation_angstrom: float | None = None,
        clash_tolerance_angstrom: float = DEFAULT_CLASH_TOLERANCE_A,
        competing_group_margin_angstrom: float = DEFAULT_COMPETING_GROUP_MARGIN_A,
        competing_group_margin_source: str = "",
        min_valid_poses: int = DEFAULT_MIN_VALID_POSES,
        in_plane_tolerance_deg: float = stereo_mod.DEFAULT_IN_PLANE_TOLERANCE_DEG,
        infer_single_ligand_substrate: bool = True,
        pocket_shell_angstrom: float = DEFAULT_POCKET_SHELL_A,
        numbering_schemes: Mapping[str, FamilyNumberingScheme] | None = None,
        calibration_store: CalibrationStore | None = None,
        strict_calibration: bool = False,
        submit_to: str | None = None,
        **_: Any,
    ) -> ToolResult:
        """Evaluate every pose of every candidate and write the two tables.

        ``structures`` lets a caller hand in already-parsed
        :class:`~eagent.science.structure_io.Structure` objects keyed by pose id,
        for in-memory pipelines and tests; otherwise each pose is read from
        ``PoseBinding.structure_path`` or ``ComplexPose.path``. A pose with
        neither is a modelling gap, reported as
        :attr:`PoseOutcome.NOT_MEASURABLE` -- never as a candidate that failed.
        """
        refusal = self._refuse_external_submission(ctx, submit_to)
        if refusal is not None:
            return refusal

        if not candidates:
            return ToolResult.failure(
                self.name,
                "no candidates supplied: pass candidates=[Candidate, ...] from "
                "annotate_family / model_complexes. This step will not re-derive "
                "them, and an empty evaluation is not a result.",
                code="no_candidates",
            )

        self._pockets = {}
        self._pocket_notes = {}
        # How this run decides whether a window's calibration counts. Held on
        # the instance for the length of one execute, like the pockets, so the
        # many call sites that ask for an authority all ask the same question.
        self._calibration = CalibrationContext(
            store=calibration_store, strict=strict_calibration)
        binding_map = self._index_bindings(bindings)
        lookup = self._template_lookup(ctx, catalytic_templates)
        result = ToolResult(status=Status.SUCCESS)
        cip, cip_note = self._resolve_cip(ctx, cip_ranks, result)

        evaluations: list[CandidateEvaluation] = []
        used_templates: dict[str, CatalyticTemplate] = {}
        for candidate in candidates:
            evaluations.append(self._evaluate_candidate(
                candidate, binding_map, structures or {}, lookup, used_templates,
                result,
                cip=cip, cip_note=cip_note,
                target_configuration=ctx.task.reaction.product.target_stereochemistry,
                pocket_localisation_angstrom=pocket_localisation_angstrom,
                clash_tolerance_angstrom=clash_tolerance_angstrom,
                competing_group_margin_angstrom=competing_group_margin_angstrom,
                competing_group_margin_source=competing_group_margin_source,
                min_valid_poses=min_valid_poses,
                in_plane_tolerance_deg=in_plane_tolerance_deg,
                infer_single_ligand_substrate=infer_single_ligand_substrate,
                pocket_shell_angstrom=pocket_shell_angstrom,
                numbering_schemes=dict(numbering_schemes or {}),
            ))

        self._score_candidates(ctx, candidates, evaluations, used_templates,
                               result, min_valid_poses)
        geometry_path, scorecard_path, explain_path = self._write_artifacts(
            ctx, evaluations, candidates
        )
        result.artifacts.extend([
            Artifact(
                key="catalytic_geometry", path=str(geometry_path), kind="table",
                sha256=sha256_file(geometry_path),
                n_records=sum(e.n_poses for e in evaluations),
                summary=("one row per candidate-pose: every measurement, the "
                         "satisfied flag and window authority per constraint, which "
                         "constraints were circular, and the clash count"),
            ),
            Artifact(
                key="candidate_scorecards", path=str(scorecard_path), kind="table",
                sha256=sha256_file(scorecard_path), n_records=len(evaluations),
                summary=("per-candidate gates, ordinal axis levels, robustness with "
                         "its Wilson interval and the stereochemical call; there is "
                         "no total-score column and none may be derived"),
            ),
            Artifact(
                key="candidate_explanations", path=str(explain_path), kind="file",
                sha256=sha256_file(explain_path), n_records=len(evaluations),
                summary=("the human-readable justification a reviewer reads instead "
                         "of a score"),
            ),
        ])

        result.provenance = Provenance(
            tool=self.name,
            tool_version=self.version,
            inputs_sha256={
                "candidates": sha256_obj([c.candidate_id for c in candidates]),
                "poses": sha256_obj([[p.pose_id for p in c.poses]
                                     for c in candidates]),
                "bindings": sha256_obj({
                    k: v.model_dump(mode="json")
                    for k, v in sorted(binding_map.items())
                }),
            },
            databases={},
            models={
                f"catalytic_template:{tid}": (
                    f"{t.provenance.source_type.value}:"
                    f"{','.join(t.provenance.identifiers)}"
                )
                for tid, t in sorted(used_templates.items())
            },
            parameters={
                "pocket_localisation_default_A": DEFAULT_POCKET_LOCALISATION_A,
                "pocket_localisation_override_A": pocket_localisation_angstrom,
                "pocket_localisation_is_qc_only": True,
                "clash_tolerance_A": clash_tolerance_angstrom,
                "competing_group_margin_A": competing_group_margin_angstrom,
                "min_valid_poses": min_valid_poses,
                "in_plane_tolerance_deg": in_plane_tolerance_deg,
                "infer_single_ligand_substrate": infer_single_ligand_substrate,
                "cip_source": None if cip is None else cip.source,
                "n_candidates": len(candidates),
                "n_poses": sum(e.n_poses for e in evaluations),
                "template_authority": {
                    tid: template_authority(t, self._calibration).value
                    for tid, t in sorted(used_templates.items())
                },
                "weighted_total_score": None,
                "ranking_method": ("gates + ordinal levels + Pareto non-domination; "
                                   "no weighted total exists"),
            },
            random_seed=ctx.seed_for(self.name),
        )

        self._summarise(result, evaluations)
        result.data.update({
            "evaluations": [self._evaluation_payload(e) for e in evaluations],
            "pocket_residues": {
                cid: p.to_dict() for cid, p in sorted(self._pockets.items())
            },
            "pocket_frame_note": (
                "pocket residues are reported in the frame they were placed "
                "in. Two candidates' positions mean the same thing only when "
                "their frames match: author numbering shifts with construct "
                "boundaries and tags, so an unframed pocket is comparable by "
                "composition and by nothing else. Pass numbering_schemes to "
                "put them in a family frame."
            ),
            "pose_outcome_legend": {o.value: o.claim() for o in PoseOutcome},
            "pocket_localisation_note": (
                "the pocket-localisation distance is a QC screen between the "
                "substrate's reactive atom and the template's catalytic functional "
                "atoms; it is not evidence of catalytic ability and never rejects a "
                "pose"
            ),
            "no_total_score": (
                "ranking is gates, ordinal levels and Pareto non-domination; this "
                "step emits no weighted total and none may be derived from it"
            ),
        })
        return result

    # -- guards -------------------------------------------------------------
    def _refuse_external_submission(
        self, ctx: RunContext, submit_to: str | None
    ) -> ToolResult | None:
        """Refuse to hand candidate sequences or structures to any service.

        Enforced in code rather than promised in a comment. Candidate sequences
        at this stage are unpublished: a mined ORF from an in-house metagenome,
        or a designed variant. Posting one to a web service for a "quick
        structure check" discloses it irreversibly and can forfeit novelty. This
        step has no legitimate need to transmit anything -- it reads local
        coordinates and computes -- so the request is refused outright rather
        than gated on ``allow_network``, which exists for database reads.
        """
        if not submit_to:
            return None
        suffix = ("" if ctx.policy.allow_network
                  else " (the run policy also has allow_network=False)")
        return ToolResult.failure(
            self.name,
            f"refused to submit candidate sequences or structures to "
            f"'{submit_to}'. evaluate_catalysis is a local computation; "
            f"transmitting an unpublished sequence needs an explicit disclosure "
            f"decision taken by the operator outside this interface{suffix}",
            code="external_submission_refused",
        )

    @staticmethod
    def _guard_no_uncalibrated_ee(call: StereoCall) -> StereoCall:
        """Re-check that no numeric ee escaped without a named calibration source.

        :class:`StereoCall` validates this on construction and
        :func:`~eagent.science.stereo.call_stereochemistry` never sets the field,
        so this is belt and braces -- which is proportionate, because a
        fabricated "92% ee" is the most quotable number this pipeline could emit
        and the one a reader is least likely to check.
        """
        if call.predicted_ee_pct is not None and not call.calibration_source:
            raise FabricationGuardError(
                "a numeric ee reached the result without a named calibration "
                "source; pose counts are a sampling artefact and cannot support one"
            )
        return call

    # -- input marshalling --------------------------------------------------
    @staticmethod
    def _index_bindings(
        bindings: Mapping[str, PoseBinding] | Sequence[PoseBinding] | None,
    ) -> dict[str, PoseBinding]:
        """Bindings keyed by pose id, however the caller passed them."""
        if bindings is None:
            return {}
        if isinstance(bindings, Mapping):
            return {str(k): v for k, v in bindings.items()}
        return {b.pose_id: b for b in bindings}

    def _template_lookup(
        self, ctx: RunContext,
        catalytic_templates: Mapping[str, CatalyticTemplate] | CatalyticTemplate | None,
    ) -> Callable[[str | None], CatalyticTemplate | None]:
        """Resolve a template id against the explicit argument, then ``ctx.templates``.

        A named template that no library holds raises
        :class:`~eagent.errors.TemplateError`: that is a broken run
        configuration, and degrading it to "this candidate has no known
        mechanism" would turn a missing file into a property of the enzyme.
        """
        explicit: dict[str, CatalyticTemplate] = {}
        if isinstance(catalytic_templates, CatalyticTemplate):
            explicit[catalytic_templates.template_id] = catalytic_templates
        elif isinstance(catalytic_templates, Mapping):
            for key, value in catalytic_templates.items():
                if isinstance(value, CatalyticTemplate):
                    explicit[str(key)] = value
        library = ctx.templates

        def lookup(template_id: str | None) -> CatalyticTemplate | None:
            if not template_id:
                return None
            if template_id in explicit:
                return explicit[template_id]
            if library is None:
                if explicit:
                    raise TemplateError(
                        f"catalytic template '{template_id}' was not among the "
                        f"templates supplied to evaluate_catalysis "
                        f"({sorted(explicit)}); a candidate may not be evaluated "
                        f"against a template that was not loaded"
                    )
                return None
            found: Any = None
            if isinstance(library, Mapping):
                found = library.get(template_id)
                nested = library.get("catalytic")
                if found is None and isinstance(nested, Mapping):
                    found = nested.get(template_id)
            else:
                store = getattr(library, "catalytic_templates", None)
                if isinstance(store, Mapping):
                    found = store.get(template_id)
                if found is None:
                    for attr in ("catalytic", "get_catalytic", "catalytic_template",
                                 "get"):
                        fn = getattr(library, attr, None)
                        if callable(fn):
                            found = fn(template_id)
                            if found is not None:
                                break
            if found is None:
                raise TemplateError(
                    f"catalytic template '{template_id}' is not in the injected "
                    f"library; a candidate may not be evaluated against a template "
                    f"that was not loaded"
                )
            if not isinstance(found, CatalyticTemplate):
                raise TemplateError(
                    f"'{template_id}' resolved to {type(found).__name__}, not a "
                    f"CatalyticTemplate"
                )
            return found

        return lookup

    def _resolve_cip(
        self, ctx: RunContext, cip_ranks: Any, result: ToolResult,
    ) -> tuple[stereo_mod.CIPPriority | None, str]:
        """Obtain supplied CIP ranks, or state why there is no face call.

        CIP priorities are *data*, never perceived here: a swapped pair of ranks
        inverts every R/S call silently while every distance, angle and docking
        score still looks perfect, and there is no downstream check that would
        catch it. When no ranks are available the step still measures geometry;
        it simply reports every pose's configuration as undetermined, which
        aggregates to ``insufficient_evidence`` rather than to a guess.
        """
        source = cip_ranks
        if source is None:
            cfg = ctx.config or {}
            if stereo_mod.CIP_CONFIG_KEY in cfg or isinstance(cfg.get("stereo"), Mapping):
                source = cfg
        if source is None:
            note = ("no CIP priority block was supplied, so no pose could be turned "
                    "into a product configuration; this pipeline does not perceive "
                    "CIP ranks")
            result.add_uncertainty(
                "cip_ranks_absent", note + ". Which CIP ranks apply to the three "
                "ligands of the prochiral carbonyl carbon?",
                affects=[self.name],
                resolvable_by="supply cip_ranks (ligand -> rank, plus the incoming "
                              "group's rank) from the reaction template or an "
                              "operator-reviewed config",
            )
            return None, note
        try:
            cip = stereo_mod.cip_from_template(source)
        except TemplateError as exc:
            note = f"CIP ranks could not be read: {exc}"
            result.add_flag("cip_ranks_unusable", Severity.WARN, note)
            result.add_uncertainty(
                "cip_ranks_unusable", note,
                affects=[self.name],
                resolvable_by="fix the CIP priority block in the reaction template "
                              "or the task config",
            )
            return None, note
        if cip.incoming_rank is None:
            note = ("CIP ranks were supplied without the incoming group's rank, so "
                    "a face cannot be converted into a configuration")
            result.add_uncertainty(
                "cip_incoming_rank_absent", note,
                affects=[self.name],
                resolvable_by="add incoming_group_rank (4 for hydride delivery) to "
                              "the CIP priority block",
            )
        return cip, ""

    # -- per candidate ------------------------------------------------------
    def _evaluate_candidate(
        self,
        candidate: Candidate,
        binding_map: Mapping[str, PoseBinding],
        structures: Mapping[str, Structure],
        lookup: Callable[[str | None], CatalyticTemplate | None],
        used_templates: dict[str, CatalyticTemplate],
        result: ToolResult,
        *,
        cip: stereo_mod.CIPPriority | None,
        cip_note: str,
        target_configuration: Any,
        pocket_localisation_angstrom: float | None,
        clash_tolerance_angstrom: float,
        competing_group_margin_angstrom: float,
        competing_group_margin_source: str,
        min_valid_poses: int,
        in_plane_tolerance_deg: float,
        infer_single_ligand_substrate: bool,
        pocket_shell_angstrom: float,
        numbering_schemes: Mapping[str, FamilyNumberingScheme] | None,
    ) -> CandidateEvaluation:
        """Evaluate one candidate's poses and roll them up without a total."""
        template_id = candidate.catalytic_mapping.catalytic_template_id
        template: CatalyticTemplate | None = None
        template_problem = ""
        try:
            template = lookup(template_id)
        except TemplateError as exc:
            template_problem = str(exc)
            result.add_flag("catalytic_template_missing", Severity.BLOCKER,
                            template_problem, candidate.candidate_id)
        family_match: bool | None = None
        family_note = ""
        if template is not None:
            family_match, family_note = self._family_agreement(candidate, template)
            if family_match is not False:
                # Registered only when it is actually applied: a template from
                # another family must not reach provenance as a model this
                # candidate was evaluated against, nor reach the scorecard's
                # template-dependent gates.
                used_templates[template.template_id] = template

        authority = (template_authority(template, self._calibration)
                     if template is not None
                     else WindowAuthority.UNCALIBRATED)
        evaluation = CandidateEvaluation(
            candidate_id=candidate.candidate_id,
            template_id=template.template_id if template is not None else template_id,
            authority=authority,
            has_template=template is not None,
            family_match=family_match,
            family_note=family_note,
        )

        if template is None:
            reason = template_problem or (
                "no catalytic template is attached to this candidate, so there is "
                "no sourced mechanism to measure against"
            )
            result.add_uncertainty(
                "no_catalytic_template",
                f"{candidate.candidate_id}: {reason}. Which sourced "
                f"CatalyticTemplate describes this family's mechanism?",
                affects=[candidate.candidate_id],
                resolvable_by="curate or attach a CatalyticTemplate for the family",
            )
            for pose in candidate.poses:
                evaluation.poses.append(self._gap(
                    result,
                    dict(candidate_id=candidate.candidate_id, pose_id=pose.pose_id,
                         method=pose.method, template_id=template_id,
                         authority=WindowAuthority.UNCALIBRATED),
                    reason,
                ))
        elif family_match is False:
            self._refuse_family_mismatch(
                result, candidate, evaluation,
                template.template_id, family_note,
            )
        else:
            if family_match is None:
                # An unevaluated gate, not a mismatch: the geometry is still
                # worth measuring, and the open question is routed as one
                # rather than resolved in either direction here.
                result.add_uncertainty(
                    "candidate_family_unverified",
                    f"{candidate.candidate_id}: {family_note}. Is "
                    f"{template.template_id} the right mechanistic hypothesis "
                    f"for this sequence?",
                    affects=[candidate.candidate_id],
                    resolvable_by=("run annotate_family to obtain a family call, "
                                   "or name the family on the catalytic template, "
                                   "so the template can be checked against it"),
                )
            self._flag_template_authority(result, candidate, template, authority)
            for pose in candidate.poses:
                evaluation.poses.append(self._evaluate_pose(
                    candidate, pose, template, authority,
                    binding_map.get(pose.pose_id), structures.get(pose.pose_id),
                    result,
                    cip=cip,
                    pocket_localisation_angstrom=pocket_localisation_angstrom,
                    clash_tolerance_angstrom=clash_tolerance_angstrom,
                    competing_group_margin_angstrom=competing_group_margin_angstrom,
                    competing_group_margin_source=competing_group_margin_source,
                    in_plane_tolerance_deg=in_plane_tolerance_deg,
                    infer_single_ligand_substrate=infer_single_ligand_substrate,
                    pocket_shell_angstrom=pocket_shell_angstrom,
                    numbering_schemes=numbering_schemes,
                ))

        self._rollup(evaluation, min_valid_poses)
        self._stereo_rollup(evaluation, target_configuration, cip_note)
        self._attach(candidate, evaluation)
        return evaluation

    @staticmethod
    def _family_agreement(
        candidate: Candidate, template: CatalyticTemplate,
    ) -> tuple[bool | None, str]:
        """``(verdict, note)``: does this template describe this candidate's family?

        WHY AN ID LOOKUP IS NOT ENOUGH
        ------------------------------
        The catalytic template is resolved by id from
        :attr:`~eagent.schemas.candidate.CatalyticMapping.catalytic_template_id`,
        and an id is just a string that some earlier step wrote. Nothing in the
        lookup notices that ``ct_sdr_ketoreductase_v1`` was attached to an
        aldo-keto reductase: the SDR template's catalytic Tyr/Lys roles map
        cleanly onto *some* residues, its NADPH windows measure, and the
        candidate scores. The whole design treats the short-chain
        dehydrogenase/reductase, aldo-keto reductase and zinc-dependent
        medium-chain alcohol dehydrogenase families as separate mechanistic
        hypotheses -- different catalytic residues, different cofactor
        recognition, different stereochemical logic -- and silently applying
        one family's rules to another family's candidate produces a confident
        verdict about a mechanism nobody proposed.

        The comparison is case-insensitive and whitespace-insensitive because
        family names arrive from annotation steps and curated template files
        that disagree about capitalisation, and a case difference is not a
        mechanistic difference. It is deliberately **not** fuzzy beyond that:
        "SDR" and "SDR-like" are left to a curator, because guessing which
        near-matches are the same mechanism is precisely the judgement this
        step must not make on its own.
        """
        declared = (candidate.family.family_name or "").strip()
        from_template = (template.family_name or "").strip()
        if not declared:
            return None, (
                f"the candidate carries no family call, so whether catalytic "
                f"template {template.template_id} (family "
                f"'{from_template or 'unnamed'}') describes this sequence's "
                f"mechanism could not be checked"
            )
        if not from_template:
            return None, (
                f"catalytic template {template.template_id} names no family, so "
                f"it could not be compared with the candidate's family "
                f"'{declared}'"
            )
        if declared.casefold() == from_template.casefold():
            return True, (
                f"catalytic template {template.template_id} describes family "
                f"'{from_template}', which is the candidate's family '{declared}'"
            )
        return False, (
            f"catalytic template {template.template_id} describes family "
            f"'{from_template}', but this candidate was annotated as "
            f"'{declared}'. These are separate mechanistic hypotheses with "
            f"different catalytic residues and different cofactor recognition, "
            f"so the template's windows cannot be applied to this sequence; "
            f"doing so would score a mechanism nobody proposed for it"
        )

    def _refuse_family_mismatch(
        self, result: ToolResult, candidate: Candidate,
        evaluation: CandidateEvaluation, template_id: str | None, reason: str,
    ) -> None:
        """Disqualify rather than score, and record every pose as unmeasured.

        A refusal, not a low score. There is no number this step could emit for
        a candidate measured against the wrong family's mechanism that would be
        less misleading than emitting none: the geometry would be real, the
        windows would be real, and the conclusion would be about a reaction
        this enzyme was never hypothesised to run. So the poses are recorded as
        unmeasured -- nothing *was* tested -- the candidate is disqualified with
        the reason attached, and the fix (attach the right template, or correct
        the family call) is stated as a BLOCKER.
        """
        result.add_flag("catalytic_template_family_mismatch", Severity.BLOCKER,
                        reason, candidate.candidate_id)
        if candidate.disqualified and candidate.disqualification_reason:
            candidate.disqualification_reason = (
                f"{candidate.disqualification_reason}; {reason}")
        else:
            candidate.disqualification_reason = reason
        candidate.disqualified = True
        result.add_next(
            "annotate_family",
            "Resolve the family call and the catalytic template against each "
            "other before any geometry is measured; a template from another "
            "family cannot be applied to this candidate",
            {"candidate_id": candidate.candidate_id,
             "catalytic_template_id": template_id},
        )
        for pose in candidate.poses:
            evaluation.poses.append(self._gap(
                result,
                dict(candidate_id=candidate.candidate_id, pose_id=pose.pose_id,
                     method=pose.method, template_id=template_id,
                     authority=WindowAuthority.UNCALIBRATED),
                f"refused to measure this pose: {reason}",
            ))

    def _flag_template_authority(
        self, result: ToolResult, candidate: Candidate,
        template: CatalyticTemplate, authority: WindowAuthority,
    ) -> None:
        """Say, per candidate, that the windows are provisional -- and which ones."""
        # Windows that cite a calibration which cannot be shown are named
        # whether or not the template as a whole ends up calibrated: a window
        # holding the power to reject on the strength of a prose string is the
        # thing a reader of this report most needs to be told.
        for constraint in template.geometry_constraints:
            status = self._calibration.status(constraint)
            if status.is_unverified_claim:
                result.add_flag(
                    "calibration_unverified", Severity.WARN,
                    f"{template.template_id}.{constraint.name}: calibrated_on "
                    f"cites {', '.join(status.evidence)[:120]!r} "
                    + ("and the window is therefore NOT treated as calibrated: "
                       if not status.calibrated else
                       "which no record backs; the window is treated as "
                       "calibrated only because this run is not strict: ")
                    + "; ".join(status.problems or (
                        "free-text calibrated_on is not a record anyone can "
                        "open",))[:300],
                    candidate.candidate_id)
        if authority is WindowAuthority.CALIBRATED:
            return
        uncalibrated = [c.name for c in template.geometry_constraints
                        if not self._calibration.status(c).calibrated]
        listed = ", ".join(uncalibrated) if uncalibrated \
            else "the whole template is a theoretical model"
        result.add_flag(
            "template_window_provisional", Severity.WARN,
            f"{template.template_id}: {authority.caveat()}; {listed}",
            candidate.candidate_id,
        )
        result.add_uncertainty(
            "uncalibrated_geometry_window",
            f"{template.template_id} supplies windows that were never fitted to "
            f"systems of known activity ({listed}). What are the correct windows "
            f"for this family? Until they exist these windows may not reject "
            f"candidates, and passes against them are reported at a downgraded "
            f"confidence.",
            affects=[candidate.candidate_id],
            resolvable_by=("calibrate the window on known-active and known-inactive "
                           "members of this family, or cite an experimental complex"),
        )

    # -- per pose -----------------------------------------------------------
    def _evaluate_pose(
        self,
        candidate: Candidate,
        pose: ComplexPose,
        template: CatalyticTemplate,
        authority: WindowAuthority,
        binding: PoseBinding | None,
        structure: Structure | None,
        result: ToolResult,
        *,
        cip: stereo_mod.CIPPriority | None,
        pocket_localisation_angstrom: float | None,
        clash_tolerance_angstrom: float,
        competing_group_margin_angstrom: float,
        competing_group_margin_source: str,
        in_plane_tolerance_deg: float,
        infer_single_ligand_substrate: bool,
        pocket_shell_angstrom: float,
        numbering_schemes: Mapping[str, FamilyNumberingScheme] | None,
    ) -> PoseEvaluation:
        """Measure one pose. Every early return is a *gap*, never a verdict."""
        base: dict[str, Any] = dict(
            candidate_id=candidate.candidate_id, pose_id=pose.pose_id,
            method=pose.method, template_id=template.template_id,
            authority=authority,
        )
        if not pose.is_valid:
            return self._gap(
                result, base,
                "the pose was marked invalid by the step that built it: "
                + (pose.invalid_reason or "no reason recorded"),
            )
        if binding is None:
            return self._gap(
                result, base,
                "no PoseBinding for this pose: the correspondence between "
                "mechanistic role tokens and coordinate atoms is knowledge only "
                "the pose builder has, and this step will not invent it",
            )
        structure, parse_error = self._load_structure(pose, binding, structure)
        if structure is None:
            return self._gap(result, base, parse_error)

        context = build_role_context(
            structure, binding, template,
            infer_single_ligand_substrate=infer_single_ligand_substrate,
        )
        resolver = context.resolver()
        # The pocket is extracted here because this is the one place that
        # holds the coordinates and knows which residue is the substrate.
        # Leaving it to the selector meant nothing ever extracted it, and the
        # diversity layer fell back to catalytic roles -- which are the
        # conserved positions of a family and so nearly constant within one.
        self._record_pocket(candidate, pose, context, template,
                            pocket_shell_angstrom, numbering_schemes, result)

        # -- mechanism geometry, measured by the science layer ---------------
        try:
            measurements = geom.measure_all(template.geometry_constraints, resolver)
        except TemplateError as exc:
            result.add_flag("malformed_catalytic_template", Severity.BLOCKER,
                            f"{template.template_id}: {exc}", candidate.candidate_id)
            return self._gap(
                result, base,
                f"the template could not be applied to this pose: {exc}",
                unresolved_roles=context.gaps(),
            )
        satisfied = {c.name: c.satisfied_by(measurements.get(c.name))
                     for c in template.geometry_constraints}
        aspects = {c.name: classify_aspect(c) for c in template.geometry_constraints}
        authorities = {c.name: constraint_authority(c, template, self._calibration)
                       for c in template.geometry_constraints}

        cofactor = self._check_cofactor(pose, template, context, satisfied, aspects)
        pocket = self._pocket_localisation(template, binding, context,
                                           pocket_localisation_angstrom)
        chemo = self._chemoselectivity(binding, context,
                                       competing_group_margin_angstrom,
                                       competing_group_margin_source)
        clash = self._clash_screen(context, clash_tolerance_angstrom)
        face, configuration, stereo_note = self._face_call(
            binding, context, cip, in_plane_tolerance_deg
        )

        verdict, hard, provisional, unmeasured = self._gating_verdict(
            template, satisfied, authorities
        )
        input_errors = tuple(cofactor.input_defects())
        outcome, reason = self._classify_pose(
            template, verdict, hard, provisional, unmeasured, chemo, input_errors,
            cofactor.unresolved_requirements(),
        )
        if outcome in (PoseOutcome.INPUT_ERROR,
                       PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW,
                       PoseOutcome.NOT_MEASURABLE):
            # Nothing was decided against a window we trust, so the schema-level
            # gate must read "undecided" rather than "failed".
            verdict = None
        if chemo.displaced_on_a_calibrated_margin:
            verdict = False

        evaluation = PoseEvaluation(
            outcome=outcome, reason=reason,
            measurements=dict(measurements), satisfied=satisfied,
            aspects=aspects, authorities=authorities,
            gating_passed=verdict,
            hard_failures=tuple(hard), provisional_failures=tuple(provisional),
            unmeasured=tuple(unmeasured),
            pocket=pocket, chemoselectivity=chemo, cofactor=cofactor, clash=clash,
            face=face, product_configuration=configuration, stereo_note=stereo_note,
            unresolved_roles=context.gaps(), input_errors=input_errors,
            **base,
        )
        self._apply_circularity(
            evaluation, pose, result, candidate.candidate_id,
            gating_names=[c.name for c in template.gating_constraints()],
        )
        self._flag_pose(result, candidate.candidate_id, evaluation)
        return evaluation

    # -- pose sub-checks ----------------------------------------------------
    def _gap(self, result: ToolResult, base: dict[str, Any], reason: str,
             unresolved_roles: dict[str, str] | None = None) -> PoseEvaluation:
        """Record a pose nothing could be measured on, and say so out loud.

        Routed through one helper so that *every* early exit raises the
        "this is a modelling failure, not an inactive enzyme" flag. An early
        return that quietly produced an unflagged NOT_MEASURABLE would leave the
        candidate looking merely unremarkable in the QC log, which is how a
        broken pipeline stage becomes a biological conclusion.
        """
        evaluation = PoseEvaluation(
            outcome=PoseOutcome.NOT_MEASURABLE, reason=reason,
            unresolved_roles=dict(unresolved_roles or {}), **base,
        )
        self._flag_pose(result, str(base["candidate_id"]), evaluation)
        return evaluation

    #: candidate id -> the pocket extracted for it, and the note describing
    #: how. Collected across poses during one ``execute`` and reset at its
    #: start, so a second run cannot inherit the first one's pockets.
    _pockets: dict[str, PocketResidues]
    _pocket_notes: dict[str, str]
    #: Set at the start of every ``execute``; see there.
    _calibration: CalibrationContext = CalibrationContext()

    def _record_pocket(
        self, candidate: Candidate, pose: ComplexPose,
        context: RoleContextResult, template: CatalyticTemplate,
        radius: float,
        schemes: Mapping[str, FamilyNumberingScheme] | None,
        result: ToolResult,
    ) -> None:
        """Extract this candidate's pocket once, from the first usable pose.

        One pose per candidate, not an average over poses: the pocket is a
        property of the protein, and a union over poses would grow with how
        many poses happened to be modelled, which would make a
        well-sampled candidate look as though it had a larger pocket.

        Failures are recorded as notes and nothing else. An unmeasured pocket
        must stay unmeasured: an empty one would read as "no residues line
        this site", which is a claim, and :func:`pocket_distance` treats the
        two differently on purpose.
        """
        if candidate.candidate_id in self._pockets:
            return
        residue = context.substrate_residue
        if residue is None:
            self._pocket_notes.setdefault(
                candidate.candidate_id,
                f"{pose.pose_id}: the substrate residue could not be located, "
                f"so no pocket was extracted")
            return
        scheme = (schemes or {}).get(candidate.family.family_name or "")
        pocket, note = pocket_residues_for_pose(
            candidate.candidate_id, candidate.sequence, context.structure,
            residue.heavy_atoms(), max_angstrom=radius, scheme=scheme)
        self._pocket_notes[candidate.candidate_id] = f"{pose.pose_id}: {note}"
        if pocket is None:
            return
        self._pockets[candidate.candidate_id] = pocket
        if pocket.n_lost:
            result.add_flag(
                "pocket_residues_unplaced", Severity.WARN,
                f"{pocket.n_lost} of "
                f"{len(pocket.tokens) + pocket.n_lost} pocket residue(s) could "
                f"not be placed in frame {pocket.frame}: "
                + "; ".join(pocket.notes[:3]),
                candidate.candidate_id)

    @staticmethod
    def _load_structure(
        pose: ComplexPose, binding: PoseBinding, supplied: Structure | None,
    ) -> tuple[Structure | None, str]:
        """Parsed coordinates, or the reason there are none.

        A parse failure comes back as text rather than as an exception: one
        unreadable pose file must not abort the evaluation of every other
        candidate, and the pose it belongs to is then recorded as unmeasurable,
        which is exactly what happened.
        """
        if supplied is not None:
            return supplied, ""
        path = binding.structure_path or pose.path
        if not path:
            return None, ("neither the pose nor its binding names a coordinate "
                          "file, so nothing could be measured")
        p = Path(path)
        if not p.is_file():
            return None, f"pose coordinate file {p} does not exist"
        try:
            return read_structure(p, structure_id=pose.pose_id), ""
        except StructureParseError as exc:
            return None, f"pose coordinate file {p} could not be parsed: {exc}"

    @staticmethod
    def _check_cofactor(
        pose: ComplexPose, template: CatalyticTemplate, context: RoleContextResult,
        satisfied: Mapping[str, bool | None], aspects: Mapping[str, str],
    ) -> CofactorCheck:
        """Identity, oxidation state and placement of the cofactor in this pose.

        The state is read from the pose's own declaration first and from the PDB
        component id second, and it stays ``UNKNOWN`` when neither resolves. An
        unknown state is an unchecked state, never an assumed reduced one --
        assuming reduction is how a hydride-transfer distance gets measured to a
        nicotinamide that cannot donate.
        """
        required_identity = cofactor_identity(
            template.required_cofactor, *template.cofactor_ligand_codes
        )
        residue = context.cofactor_residue
        component = residue.resname.strip().upper() if residue is not None else None

        observed_state = pose.cofactor_state
        state_source = "declared on the pose"
        if observed_state is CofactorState.UNKNOWN:
            observed_state = cofactor_state_from_ligand_code(component)
            state_source = (
                f"inferred from component id {component}"
                if observed_state is not CofactorState.UNKNOWN
                else "neither the pose nor the component id states it"
            )

        placement = tuple(
            name for name, aspect in aspects.items()
            if aspect in ("cofactor_placement", "cofactor_internal")
        )
        values = [satisfied.get(n) for n in placement]
        if not values or all(v is None for v in values):
            placement_ok: bool | None = None
        elif any(v is False for v in values):
            placement_ok = False
        elif any(v is None for v in values):
            placement_ok = None
        else:
            placement_ok = True

        return CofactorCheck(
            required_identity=required_identity,
            required_state=template.required_cofactor_state,
            present=residue is not None and bool(context.cofactor_role_atoms),
            observed_identity=cofactor_identity(component),
            observed_state=observed_state,
            observed_component=component,
            state_source=state_source,
            placement_constraints=placement,
            placement_satisfied=placement_ok,
        )

    @staticmethod
    def _pocket_localisation(
        template: CatalyticTemplate, binding: PoseBinding,
        context: RoleContextResult, override: float | None,
    ) -> PocketLocalisation:
        """Reactive atom to nearest catalytic functional atom, in angstroms.

        The reference set is the template's own catalytic functional atoms plus
        the hydride donor -- the atoms that do the chemistry. Deliberately not a
        centroid, and deliberately not an alpha carbon: an alpha carbon can sit
        6 A from its own side-chain hydroxyl, which is most of the window.

        A reference is accepted only when the atom bound for it really is one of
        the atoms the template named. The canonical ``label.ATOM`` key is tried
        first; a binding made under the bare ``label`` is accepted only after
        the resolved atom's own ``name`` is checked against that entry's
        ``functional_atoms``, and is otherwise dropped into
        ``missing_references``. Without that check the screen silently measures
        to a backbone atom and reports the distance as the template's catalytic
        functional atom -- a number that is right about the coordinates and
        wrong about the chemistry, which is worse than no number at all.
        """
        threshold, source = pocket_localisation_window(template, override)
        reactive = context.substrate_role_atoms.get(binding.reactive_atom_role)
        if reactive is None:
            return PocketLocalisation(
                None, threshold, source,
                reason=(f"the substrate's reactive atom role "
                        f"'{binding.reactive_atom_role}' is not bound to a "
                        f"coordinate atom, so the screen could not run"),
            )
        refs: dict[str, Atom] = {}
        gaps: list[str] = []
        for entry in template.catalytic_residues:
            label = entry.get("label")
            if label is None:
                continue
            functional = [str(a) for a in (entry.get("functional_atoms", ()) or ())]
            permitted = {a.strip().upper() for a in functional}
            for atom_name in functional:
                key = f"{label}.{atom_name}"
                atom = context.protein_atoms.get(key)
                if atom is not None:
                    refs[key] = atom
                    continue
                bare = context.protein_atoms.get(str(label))
                if bare is None:
                    gaps.append(
                        f"{key} (no atom is bound under this role, and the bare "
                        f"label '{label}' is unbound too)"
                    )
                    continue
                if bare.name.strip().upper() in permitted:
                    # The bare-label binding happens to name a functional atom;
                    # key it by the atom actually resolved, never by the atom
                    # name we were looking for.
                    refs[f"{label}.{bare.name.strip()}"] = bare
                else:
                    gaps.append(
                        f"{key} (unbound; the bare label '{label}' resolves to "
                        f"atom '{bare.name.strip()}', which is not among this "
                        f"template entry's functional_atoms {sorted(permitted)}, "
                        f"so it was NOT substituted)"
                    )
        donor = context.cofactor_role_atoms.get(binding.hydride_donor_role)
        if donor is not None:
            refs[f"{NS_COFACTOR}.{binding.hydride_donor_role}"] = donor
        missing = tuple(sorted(set(gaps)))
        if not refs:
            return PocketLocalisation(
                None, threshold, source, missing_references=missing,
                reason=("none of the template's catalytic functional atoms could be "
                        "located in this pose, so there is nothing to measure the "
                        "reactive atom against"
                        + (": " + "; ".join(missing) if missing else "")),
            )
        best_key, best_d = min(
            ((k, geom.distance(reactive, a)) for k, a in refs.items()),
            key=lambda kv: kv[1],
        )
        return PocketLocalisation(
            distance_A=best_d, threshold_A=threshold, threshold_source=source,
            nearest_role=best_key, reference_roles=tuple(sorted(refs)),
            missing_references=missing,
            reason=("reference atoms left out of the screen: " + "; ".join(missing)
                    if missing else ""),
        )

    @staticmethod
    def _chemoselectivity(
        binding: PoseBinding, context: RoleContextResult, margin: float,
        margin_source: str = "",
    ) -> ChemoselectivityCheck:
        """Compare the target reactive atom against the declared competing groups.

        ``margin_source`` names what the margin was fitted to. Without one the
        comparison is marked uncalibrated and cannot reject a pose: the margin
        is the entire content of the comparison -- both distances carry the
        pose generator's positional scatter -- and the module default is a
        placeholder. Rejecting on it produced a hard computational negative
        from a number nobody measured, while every template window in this
        module with the same provenance is explicitly not allowed to reject.
        """
        authority = (WindowAuthority.CALIBRATED if margin_source.strip()
                     else WindowAuthority.UNCALIBRATED)
        source = margin_source.strip() or (
            "module default DEFAULT_COMPETING_GROUP_MARGIN_A, fitted to "
            "nothing; pass competing_group_margin_source once it has been set "
            "from this pose generator's positional scatter")
        common = dict(margin_A=margin, margin_authority=authority,
                      margin_source=source)
        donor = context.cofactor_role_atoms.get(binding.hydride_donor_role)
        target = context.substrate_role_atoms.get(binding.reactive_atom_role)
        if donor is None or target is None:
            return ChemoselectivityCheck(
                tested=False, **common,
                reason=("the hydride donor or the target reactive atom is unbound, "
                        "so which group occupies the reactive position could not be "
                        "compared"),
            )
        d_target = geom.distance(donor, target)
        if not binding.competing_electrophiles:
            return ChemoselectivityCheck(
                tested=False, target_distance_A=d_target, **common,
                reason=("no competing electrophilic atom was declared for this "
                        "substrate; chemoselectivity was not tested, which is not "
                        "the same as its having passed"),
            )
        residue = context.substrate_residue
        best_label: str | None = None
        best_d: float | None = None
        unresolved: list[str] = []
        for label, atom_name in binding.competing_electrophiles.items():
            atom = residue.atom(atom_name) if residue is not None else None
            if atom is None:
                unresolved.append(f"{label}({atom_name})")
                continue
            d = geom.distance(donor, atom)
            if best_d is None or d < best_d:
                best_d, best_label = d, label
        if best_d is None:
            return ChemoselectivityCheck(
                tested=False, target_distance_A=d_target, **common,
                reason=("none of the declared competing atoms is present in the "
                        "substrate residue: " + ", ".join(unresolved)),
            )
        displaced = best_d < d_target - margin
        extra = f"; unresolved: {', '.join(unresolved)}" if unresolved else ""
        return ChemoselectivityCheck(
            tested=True, target_distance_A=d_target,
            displacing_label=best_label if displaced else None,
            displacing_distance_A=best_d, **common,
            reason=(f"nearest competing group {best_label} at {best_d:.2f} A versus "
                    f"the target reactive atom at {d_target:.2f} A "
                    f"(margin {margin} A from {source}){extra}"),
        )

    @staticmethod
    def _clash_screen(context: RoleContextResult, tolerance: float) -> ClashScreen:
        """Count hard-sphere overlaps between the substrate and its surroundings.

        A screen, not an energy, and never a rejection on its own: it localises a
        steric problem for a human or for a real force field. Coverage is
        reported because a count of 0 from a partial screen means nothing, which
        is why :func:`eagent.science.geometry.clash_count` refuses a partial
        screen unless it is asked for explicitly.
        """
        residue = context.substrate_residue
        if residue is None:
            return ClashScreen(None, tolerance, True, (),
                               "the substrate residue could not be located")
        ligand_atoms = residue.heavy_atoms()
        if not ligand_atoms:
            return ClashScreen(None, tolerance, True, (),
                               f"substrate residue {residue} has no heavy atoms")
        surroundings = [a for a in context.structure.atoms()
                        if a.residue_key != residue.key and a.is_heavy
                        and not a.is_water]
        if not surroundings:
            return ClashScreen(None, tolerance, True, (),
                               "the pose holds nothing but the substrate, so there "
                               "is no contact partner to screen against")
        try:
            return ClashScreen(
                geom.clash_count(surroundings, ligand_atoms,
                                 vdw_overlap_tolerance=tolerance),
                tolerance, True,
            )
        except geom.GeometryError as exc:
            count = geom.clash_count(surroundings, ligand_atoms,
                                     vdw_overlap_tolerance=tolerance,
                                     allow_unscreened=True)
            elements = sorted({
                a.element.strip().upper() for a in surroundings + ligand_atoms
                if a.element.strip().upper() not in geom.VDW_RADII_A
            })
            return ClashScreen(count, tolerance, False, tuple(elements), str(exc))

    @staticmethod
    def _face_call(
        binding: PoseBinding, context: RoleContextResult,
        cip: stereo_mod.CIPPriority | None, in_plane_tolerance_deg: float,
    ) -> tuple[str | None, str | None, str]:
        """``(face, product configuration, note)`` for one pose.

        Returns ``(None, None, reason)`` rather than a guess whenever the ranks,
        the donor or any of the three sp2 ligands is missing. The face is
        recorded even for poses that will not be aggregated, because a reviewer
        comparing a rejected pose's face against the accepted ones is doing
        useful work; the aggregation in :meth:`_stereo_rollup` is what restricts
        the call to geometrically competent poses.
        """
        if cip is None:
            return None, None, "no CIP ranks supplied"
        donor = context.cofactor_role_atoms.get(binding.hydride_donor_role)
        carbon = context.substrate_role_atoms.get(binding.reactive_atom_role)
        oxygen = context.substrate_role_atoms.get(binding.carbonyl_oxygen_role)
        if donor is None or carbon is None or oxygen is None:
            return None, None, ("the donor, the carbonyl carbon or the carbonyl "
                                "oxygen is unbound, so no face can be assigned")
        subs: dict[str, Atom] = {}
        for key in binding.prochiral_substituent_roles:
            atom = context.substrate_role_atoms.get(key)
            if atom is None:
                return None, None, (f"prochiral substituent role '{key}' is unbound, "
                                    f"so the sp2 plane is undefined")
            subs[key] = atom
        if len(subs) != 2:
            return None, None, ("a trigonal carbonyl carbon needs exactly two carbon "
                                "substituents declared; the binding declares "
                                f"{len(subs)}")
        ligands: dict[str, Atom] = {binding.carbonyl_oxygen_role: oxygen, **subs}
        try:
            face = stereo_mod.face_of_approach_with_cip(
                donor, carbon, ligands, cip,
                in_plane_tolerance_deg=in_plane_tolerance_deg,
            )
        except (ValueError, TemplateError) as exc:
            return None, None, f"face assignment refused: {exc}"
        try:
            configuration = stereo_mod.face_to_configuration(face, cip)
        except ValueError as exc:
            return face, None, f"configuration not derivable: {exc}"
        note = "" if configuration is not None else (
            "face assigned but no configuration followed (ambiguous face, or no "
            "incoming-group rank)"
        )
        return face, configuration, note

    @staticmethod
    def _gating_verdict(
        template: CatalyticTemplate,
        satisfied: Mapping[str, bool | None],
        authorities: Mapping[str, WindowAuthority],
    ) -> tuple[bool | None, list[str], list[str], list[str]]:
        """Three-valued gating, partitioned by whether a failure may reject.

        Returns ``(verdict, hard_failures, provisional_failures, unmeasured)``.
        ``verdict`` is ``True`` only when every gating constraint was measured and
        satisfied, ``False`` only when a *calibrated* gating constraint of a
        non-theoretical template failed, and ``None`` otherwise -- which covers
        both "we could not measure it" and "it failed a window we do not trust".
        """
        gating = template.gating_constraints()
        if not gating:
            return None, [], [], []
        hard: list[str] = []
        provisional: list[str] = []
        unmeasured: list[str] = []
        for constraint in gating:
            value = satisfied.get(constraint.name)
            if value is True:
                continue
            if value is False:
                authority = authorities.get(constraint.name,
                                            WindowAuthority.CALIBRATED)
                (hard if authority.may_reject else provisional).append(constraint.name)
            else:
                unmeasured.append(constraint.name)
        if hard:
            return False, hard, provisional, unmeasured
        if provisional or unmeasured:
            return None, hard, provisional, unmeasured
        return True, [], [], []

    @staticmethod
    def _classify_pose(
        template: CatalyticTemplate,
        verdict: bool | None,
        hard: Sequence[str],
        provisional: Sequence[str],
        unmeasured: Sequence[str],
        chemo: ChemoselectivityCheck,
        input_errors: Sequence[str],
        cofactor_unresolved: Sequence[str] = (),
    ) -> tuple[PoseOutcome, str]:
        """Collapse the sub-checks into one outcome, in a stated precedence.

        The precedence is argued rather than assumed. An input error comes first
        because geometry measured on the wrong molecule is not a weaker
        measurement, it is a measurement of something else. A chemoselectivity
        failure comes next because a pose presenting the wrong group is not a
        pose of the target reaction however good its numbers are. Only then do
        the template's own windows speak, and a window we do not trust speaks
        last and without the power to reject. An unresolved cofactor requirement
        comes last of all: it cannot reject a pose, but it does withhold the
        promotion to MECHANISM_SATISFIED, because "the state was never recorded"
        is not "the state is the one we need".
        """
        if input_errors:
            return PoseOutcome.INPUT_ERROR, "; ".join(input_errors)
        if chemo.target_in_reactive_position is False:
            displacement = (
                f"a non-target group occupies the reactive position: "
                f"{chemo.displacing_label} sits {chemo.displacing_distance_A:.2f} A "
                f"from the hydride donor while the target reactive atom is at "
                f"{chemo.target_distance_A:.2f} A (margin {chemo.margin_A} A)"
            )
            if chemo.displaced_on_a_calibrated_margin:
                return PoseOutcome.MECHANISM_VIOLATED, (
                    f"{displacement}. This pose would give a different product, "
                    f"so it is not evidence for the target reaction"
                )
            # The displacement is real and reported; the authority to reject on
            # it is not. The margin is the whole content of the comparison and
            # this one was fitted to nothing, so the pose lands where every
            # other uncalibrated window in this module lands.
            return PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW, (
                f"{displacement}, judged against {chemo.margin_source}. The "
                f"measurement is reported; it is not used to reject the "
                f"candidate"
            )
        if verdict is False:
            return PoseOutcome.MECHANISM_VIOLATED, (
                f"calibrated gating constraint(s) outside their window: "
                f"{', '.join(hard)}"
            )
        if provisional:
            return PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW, (
                f"outside the window of {', '.join(provisional)}, which carries no "
                f"calibration (or belongs to a theoretical-model template). The "
                f"measurement is reported; it is not used to reject the candidate"
            )
        if unmeasured:
            return PoseOutcome.NOT_MEASURABLE, (
                f"gating constraint(s) could not be measured: "
                f"{', '.join(unmeasured)}. Nothing was tested for this pose"
            )
        if verdict is None:
            return PoseOutcome.NOT_MEASURABLE, (
                f"template {template.template_id} declares no gating constraints, "
                f"so no mechanism condition could be decided"
            )
        if cofactor_unresolved:
            # Every window is satisfied, but a precondition of the mechanism was
            # never established. Promoting the pose here would be the "unknown
            # means the convenient value" substitution, so it is withheld --
            # without rejecting the candidate, which is why this is NOT_MEASURABLE
            # and not MECHANISM_VIOLATED.
            return PoseOutcome.NOT_MEASURABLE, (
                "every gating constraint is satisfied, but the pose cannot be "
                "called mechanism-competent: " + "; ".join(cofactor_unresolved)
            )
        return PoseOutcome.MECHANISM_SATISFIED, (
            "every gating constraint of the template is satisfied by this pose"
        )

    def _apply_circularity(
        self, evaluation: PoseEvaluation, pose: ComplexPose,
        result: ToolResult, candidate_id: str,
        *, gating_names: Sequence[str] = (),
    ) -> None:
        """Remove the restraints that were imposed from the evidence that counts.

        Uses :class:`~eagent.science.robustness.CircularityGuard`, built from the
        pose's own ``restrained_constraints`` rather than from anything this step
        believes, so the guard cannot be defeated by a caller that forgets to
        pass the restraint list. When every satisfied constraint turns out to
        have been enforced, the guard raises; the exception is caught and turned
        into a loud QC flag plus an uncertainty, and the independent evidence
        count stays at zero.
        """
        report = GeometryReport(
            pose_id=evaluation.pose_id,
            measurements=dict(evaluation.measurements),
            satisfied=dict(evaluation.satisfied),
            gating_passed=evaluation.gating_passed,
        )
        guard = CircularityGuard.for_report(pose, report)
        try:
            evidence = guard.raise_if_only_circular(report, subject=candidate_id)
        except CircularEvidenceError as exc:
            evidence = guard.independent_evidence(report)
            evaluation.entirely_circular = True
            result.add_flag("evidence_entirely_circular", Severity.WARN,
                            str(exc), candidate_id)
            result.add_uncertainty(
                "evidence_entirely_circular",
                f"{candidate_id}/{evaluation.pose_id}: every satisfied constraint "
                f"was enforced during modelling, so the independent evidence count "
                f"is zero. What does this pose look like when it is rebuilt without "
                f"those restraints?",
                affects=[candidate_id],
                resolvable_by=("re-run pose generation with the restraints released, "
                               "or with an independent modelling route"),
            )
        evaluation.circular_constraints = evidence.circular_all
        evaluation.circular_satisfied = evidence.circular_satisfied
        evaluation.independent_satisfied = evidence.n_independent_satisfied
        evaluation.independent_total = len(evidence.satisfied) + len(evidence.unsatisfied)
        # The gating subset, which is what the outcome rests on and therefore
        # the only subset that can corroborate the outcome. Computed from the
        # same guard, so a constraint cannot be independent here and circular
        # three lines above.
        restrained = set(evidence.circular_all)
        independent_gating = [n for n in gating_names if n not in restrained]
        evaluation.independent_gating = tuple(sorted(independent_gating))
        evaluation.independent_gating_satisfied = tuple(sorted(
            n for n in independent_gating if evaluation.satisfied.get(n) is True))
        evaluation.restrained_gating = tuple(sorted(
            n for n in gating_names if n in restrained))
        mismatch = guard.restrained_but_not_evaluated
        if mismatch:
            evaluation.restraint_name_mismatch = tuple(sorted(mismatch))
            result.add_flag(
                "restraint_name_mismatch", Severity.WARN,
                f"pose {evaluation.pose_id}: restraint(s) {sorted(mismatch)} match "
                f"no evaluated constraint name, so the circularity check may be "
                f"under-counting",
                candidate_id,
            )

    def _flag_pose(self, result: ToolResult, candidate_id: str,
                   evaluation: PoseEvaluation) -> None:
        """Per-pose QC, each finding at the severity it actually deserves."""
        pocket = evaluation.pocket
        if pocket is not None and pocket.within is False:
            result.add_flag(
                "pocket_localisation_outside_window", Severity.WARN,
                f"pose {evaluation.pose_id}: the substrate's reactive atom is "
                f"{pocket.distance_A:.2f} A from the nearest catalytic functional "
                f"atom ({pocket.nearest_role}), outside the {pocket.threshold_A:.1f} "
                f"A localisation screen [{pocket.threshold_source}]. This is a QC "
                f"screen, not evidence that the enzyme cannot catalyse the reaction",
                candidate_id,
            )
        elif pocket is not None and pocket.within is None and pocket.reason:
            result.add_flag("pocket_localisation_unmeasured", Severity.INFO,
                            f"pose {evaluation.pose_id}: {pocket.reason}",
                            candidate_id)
        if pocket is not None and pocket.missing_references:
            result.add_flag(
                "pocket_reference_atom_unbound", Severity.INFO,
                f"pose {evaluation.pose_id}: the localisation screen ran against "
                f"{len(pocket.reference_roles)} reference atom(s); these catalytic "
                f"functional atoms named by the template were left out rather than "
                f"substituted: " + "; ".join(pocket.missing_references),
                candidate_id,
            )
        chemo = evaluation.chemoselectivity
        if chemo is not None and chemo.target_in_reactive_position is False:
            result.add_flag("non_target_group_in_reactive_position", Severity.WARN,
                            f"pose {evaluation.pose_id}: {chemo.reason}", candidate_id)
            if not chemo.margin_authority.may_reject:
                # Reported, and reported as not acted on. An operator who sees
                # the displacement but not this line would reasonably assume
                # the pose had been excluded.
                result.add_uncertainty(
                    "chemoselectivity_margin_uncalibrated",
                    f"pose {evaluation.pose_id}: a non-target group is closer "
                    f"to the donor, but the margin that decides it was fitted "
                    f"to nothing ({chemo.margin_source}), so the pose was not "
                    f"rejected. How much closer counts for this pose "
                    f"generator?",
                    affects=[candidate_id],
                    resolvable_by=("measure the generator's positional scatter "
                                   "and pass competing_group_margin_angstrom "
                                   "with competing_group_margin_source"))
        elif chemo is not None and not chemo.tested:
            result.add_flag("chemoselectivity_untested", Severity.INFO,
                            f"pose {evaluation.pose_id}: {chemo.reason}", candidate_id)
        cof = evaluation.cofactor
        if cof is not None and cof.identity_ok is False:
            result.add_flag("cofactor_wrong_identity", Severity.BLOCKER,
                            f"pose {evaluation.pose_id}: "
                            + "; ".join(cof.input_defects()), candidate_id)
        if cof is not None and cof.state_ok is False:
            result.add_flag("cofactor_wrong_oxidation_state", Severity.BLOCKER,
                            f"pose {evaluation.pose_id}: "
                            + "; ".join(cof.input_defects()), candidate_id)
        if cof is not None and cof.present and cof.state_ok is None:
            result.add_flag(
                "cofactor_state_unknown", Severity.WARN,
                f"pose {evaluation.pose_id}: the cofactor's oxidation state is not "
                f"stated and could not be read from the component id "
                f"({cof.state_source}); it was NOT assumed to be "
                f"{cof.required_state.value}",
                candidate_id,
            )
        clash = evaluation.clash
        if clash is not None and clash.count:
            extra = ("" if clash.complete else
                     f"; screen incomplete, untabulated element(s) "
                     f"{list(clash.unscreened_elements)}")
            result.add_flag(
                "steric_clash", Severity.WARN,
                f"pose {evaluation.pose_id}: {clash.count} hard-sphere overlap(s) at "
                f"{clash.tolerance_A} A tolerance{extra}",
                candidate_id,
            )
        if evaluation.outcome.is_modelling_failure:
            record = evaluation.outcome.as_record_outcome()
            result.add_flag(
                "modelling_failure_not_a_negative", Severity.INFO,
                f"pose {evaluation.pose_id}: {evaluation.reason}. Recorded as "
                f"{record.value}, which means: {record.claim()}",
                candidate_id,
            )

    # -- rollups ------------------------------------------------------------
    def _rollup(self, evaluation: CandidateEvaluation, min_valid_poses: int) -> None:
        """Circularity-corrected robustness, over evidence nobody imposed.

        WHY G IS NOT THE SATISFIED FRACTION
        -----------------------------------
        The obvious definition -- satisfied poses over decided poses -- lets an
        ensemble built entirely under restraints report ``G = 1.0``, because
        every pose satisfies exactly the constraints it was built to satisfy.
        That number then flows into the scorecard, the Wilson interval, the
        tables and the batch selection as though the model had corroborated
        itself, which is the self-justification
        :class:`~eagent.science.robustness.CircularityGuard` exists to prevent.
        Flagging it afterwards does not help: by then the figure has already
        been computed and reported.

        So the correction is applied *in the definition*:

        * **Denominator** -- decided poses that carried at least one gating
          constraint which was not restrained while the pose was built
          (:attr:`PoseEvaluation.carries_independent_test`). A pose whose every
          gating constraint was imposed is excluded rather than counted as a
          failure: it is a non-test, not a negative.
        * **Numerator** -- poses satisfying the mechanism whose independent
          gating constraints were *all* satisfied
          (:attr:`PoseEvaluation.independently_satisfied`).
        * **No independent evidence at all** -- ``G = None`` with a stated
          reason, never 1.0 and never 0.0. ``None`` is the only honest value:
          nothing was corroborated and nothing was refuted.

        The uncorrected sampling fraction is still computed, under the separate
        name :attr:`CandidateEvaluation.sampling_G`, so the correction can be
        audited and so nobody has to re-derive it from the pose table.

        The denominator remains the number of poses that *could* have come out
        otherwise, never the number attempted: a candidate whose five poses all
        failed to parse keeps ``G = None`` with ``was_tested = False``, which is
        why :func:`~eagent.science.robustness.pose_robustness` returns ``None``
        rather than ``0.0`` for an empty denominator.
        """
        n_decided = evaluation.n_decided
        evaluation.sampling_G = pose_robustness(evaluation.n_satisfied, n_decided)

        n_valid = evaluation.n_independently_tested
        n_sat = evaluation.n_independently_satisfied
        evaluation.robustness_G = pose_robustness(n_sat, n_valid)
        if n_valid > 0:
            evaluation.wilson_lo, evaluation.wilson_hi = wilson_interval(n_sat, n_valid)

        sampling_text = ("none (no pose was decided)"
                         if evaluation.sampling_G is None
                         else f"{evaluation.sampling_G:.3f}")
        if n_valid == 0 and n_decided > 0:
            circular = sorted({c for p in evaluation.poses
                               for c in p.restrained_gating})
            evaluation.robustness_basis = (
                f"G is undefined: of {n_decided} decided pose(s), none carried a "
                f"gating constraint that had not been restrained during "
                f"modelling"
                + (f" (restrained gating constraints: {', '.join(circular)})"
                   if circular else "")
                + f". A distance that was enforced cannot corroborate the model "
                  f"that enforced it, so no corrected fraction exists; the "
                  f"uncorrected sampling fraction was {sampling_text} and is "
                  f"reported separately as sampling_G, which is not evidence "
                  f"about the enzyme"
            )
        elif n_valid == 0:
            evaluation.robustness_basis = (
                "G is undefined: no pose was decided against a window this "
                "step trusts, so there was nothing to correct for circularity "
                "and nothing to count. This candidate was not tested"
            )
        else:
            evaluation.robustness_basis = (
                f"G = {n_sat}/{n_valid}: of the decided poses carrying at least "
                f"one gating constraint that was not restrained during "
                f"modelling, this many had all of those independent constraints "
                f"satisfied. {n_decided - n_valid} decided pose(s) were excluded "
                f"from the denominator as entirely restrained non-tests. The "
                f"uncorrected sampling fraction over all {n_decided} decided "
                f"pose(s) was {sampling_text}"
            )

        level = classify_robustness(evaluation.robustness_G, n_valid,
                                    min_valid_poses=min_valid_poses)
        if evaluation.authority is not WindowAuthority.CALIBRATED:
            level = self._downgrade(level)
        evaluation.robustness_level = level

    @staticmethod
    def _downgrade(level: ConfidenceLevel) -> ConfidenceLevel:
        """One ordinal step down, floored at WEAK.

        Downgrading rather than rejecting is the whole point of the
        uncalibrated-template rule. The floor is WEAK rather than INSUFFICIENT
        because the measurement really was made; it is the window that is in
        doubt, and INSUFFICIENT would wrongly say nothing was measured.
        """
        ladder = [ConfidenceLevel.STRONG, ConfidenceLevel.MODERATE,
                  ConfidenceLevel.WEAK]
        if level in ladder:
            return ladder[min(ladder.index(level) + 1, len(ladder) - 1)]
        return level

    def _stereo_rollup(self, evaluation: CandidateEvaluation,
                       target_configuration: Any, cip_note: str) -> None:
        """Directional stereochemical call over the geometrically competent poses.

        Only poses with :attr:`PoseOutcome.MECHANISM_SATISFIED` contribute. A face
        computed from a pose that never satisfied the catalytic windows describes
        an arrangement that cannot react; letting it vote would have an
        unreactive pose decide the product's configuration. This mirrors the
        scope note at the top of :mod:`eagent.science.stereo`.

        The result is a direction -- favours the target, favours the opposite,
        competing poses, or insufficient evidence -- and never a number. Pose
        counts are an artefact of how the sampler ran, not a Boltzmann
        population, so no ee can be derived from them; see
        :meth:`_guard_no_uncalibrated_ee`.
        """
        competent = [p for p in evaluation.poses
                     if p.outcome is PoseOutcome.MECHANISM_SATISFIED]
        # The same circularity correction the robustness figure gets, applied
        # in the same place: in the definition. A pose whose every gating
        # constraint was restrained while it was built satisfies exactly what
        # it was built to satisfy, so it is not evidence that the enzyme
        # presents this face -- it is evidence that the restraints were
        # applied. Letting it vote produced a candidate reporting
        # robustness_G = None ("nothing was corroborated") beside
        # stereo = favors_target, from one and the same pose.
        voting = [p for p in competent if p.independently_satisfied]
        circular = [p for p in competent if not p.independently_satisfied]
        per_pose: dict[str, str | None] = {
            p.pose_id: p.product_configuration for p in voting
        }
        basis = (f"faces computed for {len(per_pose)} geometrically competent "
                 f"pose(s) of {evaluation.n_poses} attempted")
        if circular:
            basis = (
                f"{basis}; {len(circular)} further competent pose(s) were "
                f"excluded because every gating constraint they satisfy was "
                f"restrained while they were built, so their face is a "
                f"property of the restraints rather than of the enzyme"
                + (" -- no pose carries independent evidence, so there is no "
                   "stereochemical call to make" if not voting else ""))
        if cip_note:
            basis = f"{basis}; {cip_note}"
        call = stereo_mod.call_stereochemistry(per_pose, target_configuration,
                                               basis=basis)
        evaluation.stereo = self._guard_no_uncalibrated_ee(call)
        evaluation.stereo_poses_excluded_as_circular = len(circular)

    @staticmethod
    def _attach(candidate: Candidate, evaluation: CandidateEvaluation) -> None:
        """Write the per-pose reports and the roll-ups back onto the candidate.

        Input defects found at the pose level are appended to
        :attr:`~eagent.schemas.candidate.Candidate.input_errors`, which is the
        field the scorecard's input-integrity gate reads. Routing them there
        rather than onto a geometry axis is what stops a wrong cofactor from
        being traded off against a good docking score.
        """
        candidate.geometry = [p.to_report() for p in evaluation.poses]
        candidate.robustness_G = evaluation.robustness_G
        candidate.stereo = evaluation.stereo
        for defect in sorted({e for p in evaluation.poses for e in p.input_errors}):
            if defect not in candidate.input_errors:
                candidate.input_errors.append(defect)
        evaluation.input_errors = list(candidate.input_errors)

    # -- scorecards ---------------------------------------------------------
    def _score_candidates(
        self, ctx: RunContext, candidates: Sequence[Candidate],
        evaluations: Sequence[CandidateEvaluation],
        used_templates: Mapping[str, CatalyticTemplate],
        result: ToolResult, min_valid_poses: int,
    ) -> None:
        """Build each scorecard and attach its explanation. No total anywhere."""
        config = {"min_valid_poses": min_valid_poses}
        for candidate, evaluation in zip(candidates, evaluations):
            tid = candidate.catalytic_mapping.catalytic_template_id
            # Pass a library only when it holds this candidate's template: the
            # scorecard raises on a named-but-absent template, and a template we
            # already reported as missing must not fail the whole step twice.
            # A template belonging to another family is withheld for a stronger
            # reason: its cofactor and machinery gates would otherwise decide
            # this candidate against a mechanism nobody proposed for it.
            library = dict(used_templates) if (
                tid and tid in used_templates and evaluation.family_match is not False
            ) else None
            gates = evaluate_feasibility_gates(candidate, ctx.task, library)
            candidate.scorecard = build_scorecard(candidate, ctx.task, library, config)
            self._attach_family_gate(candidate, evaluation)
            if evaluation.has_template \
                    and evaluation.authority is not WindowAuthority.CALIBRATED:
                self._downgrade_axis(candidate, "catalytic_geometry",
                                     evaluation.authority)
            result.uncertainty.extend(
                gate_uncertainties(gates, subject=candidate.candidate_id))
            result.qc_flags.extend(scorecard_qc_flags(candidate, gates))
            evaluation.explanation = explain(candidate)

    @staticmethod
    def _attach_family_gate(
        candidate: Candidate, evaluation: CandidateEvaluation,
    ) -> None:
        """Record the family check as a three-valued gate on the scorecard.

        A gate rather than an axis, because there is no amount of good geometry
        that compensates for having measured the wrong mechanism. Three-valued
        because :attr:`~eagent.schemas.candidate.Candidate.has_unresolved_gate`
        is how this codebase says "this question was never answered": an
        unknown family leaves ``gate_passed=None``, which keeps the candidate
        out of the eligible set without recording it as a failure, and the
        matching uncertainty says what would resolve it.
        """
        if not evaluation.has_template:
            return
        verdict = evaluation.family_match
        level = {
            True: ConfidenceLevel.STRONG,
            False: ConfidenceLevel.CONTRADICTORY,
            None: ConfidenceLevel.INSUFFICIENT,
        }[verdict]
        candidate.set_dimension(ScoreDimension(
            name=FAMILY_MATCH_GATE, level=level, direction="categorical",
            basis=evaluation.family_note or "family agreement was not assessed",
            is_gate=True, gate_passed=verdict,
        ))

    @staticmethod
    def _downgrade_axis(candidate: Candidate, axis: str,
                        authority: WindowAuthority) -> None:
        """Lower one axis's level because the window behind it is provisional.

        Only ever downgrades. The scorecard builder cannot see the template's
        calibration state -- it reads the measured reports -- so a pass against an
        uncalibrated window would otherwise arrive as STRONG geometric evidence.
        Inflation is the failure mode that matters here, so the correction runs
        in one direction only.
        """
        dim = candidate.scorecard.get(axis)
        if dim is None:
            return
        ladder = [ConfidenceLevel.STRONG, ConfidenceLevel.MODERATE,
                  ConfidenceLevel.WEAK]
        level = dim.level
        if level in ladder:
            level = ladder[min(ladder.index(level) + 1, len(ladder) - 1)]
        candidate.set_dimension(ScoreDimension(
            name=dim.name, level=level, value=dim.value, unit=dim.unit,
            direction=dim.direction,
            basis=f"{dim.basis} | downgraded: {authority.caveat()}",
            is_gate=dim.is_gate, gate_passed=dim.gate_passed,
        ))

    # -- artifacts ----------------------------------------------------------
    def _write_artifacts(
        self, ctx: RunContext, evaluations: Sequence[CandidateEvaluation],
        candidates: Sequence[Candidate],
    ) -> tuple[Path, Path, Path]:
        """Write the geometry table, the scorecard table and the explanations."""
        names = sorted({n for e in evaluations for p in e.poses
                        for n in p.measurements})
        header = [
            "candidate_id", "pose_id", "method", "outcome", "record_outcome",
            "decided", "reason", "template_id", "template_authority",
            "gating_passed", "hard_failures", "provisional_failures",
            "unmeasured_constraints",
            "pocket_localisation_A", "pocket_threshold_A", "pocket_threshold_source",
            "pocket_within", "pocket_nearest_role", "pocket_missing_references",
            "chemoselectivity_tested", "target_in_reactive_position",
            "target_to_donor_A", "displacing_group", "displacing_to_donor_A",
            "chemoselectivity_margin_A", "chemoselectivity_margin_authority",
            "chemoselectivity_margin_source",
            "cofactor_present", "cofactor_component", "cofactor_identity",
            "cofactor_required_identity", "cofactor_state", "cofactor_required_state",
            "cofactor_identity_ok", "cofactor_state_ok", "cofactor_placement_ok",
            "clash_count", "clash_tolerance_A", "clash_screen_complete",
            "circular_constraints", "circular_satisfied", "independent_satisfied",
            "independent_total", "independent_gating", "independent_gating_satisfied",
            "restrained_gating", "carries_independent_test",
            "counts_toward_robustness",
            "entirely_circular", "restraint_name_mismatch",
            "face", "product_configuration", "stereo_note", "unresolved_roles",
        ]
        header += [f"measure:{n}" for n in names]
        header += [f"satisfied:{n}" for n in names]
        header += [f"authority:{n}" for n in names]
        header += [f"aspect:{n}" for n in names]

        rows: list[list[Any]] = []
        for e in evaluations:
            for p in e.poses:
                pocket, chemo = p.pocket, p.chemoselectivity
                cof, clash = p.cofactor, p.clash
                row: list[Any] = [
                    p.candidate_id, p.pose_id, p.method, p.outcome.value,
                    p.outcome.as_record_outcome().value, p.outcome.decided, p.reason,
                    p.template_id, p.authority.value, p.gating_passed,
                    p.hard_failures, p.provisional_failures, p.unmeasured,
                    None if pocket is None else pocket.distance_A,
                    None if pocket is None else pocket.threshold_A,
                    None if pocket is None else pocket.threshold_source,
                    None if pocket is None else pocket.within,
                    None if pocket is None else pocket.nearest_role,
                    None if pocket is None else pocket.missing_references,
                    None if chemo is None else chemo.tested,
                    None if chemo is None else chemo.target_in_reactive_position,
                    None if chemo is None else chemo.target_distance_A,
                    None if chemo is None else chemo.displacing_label,
                    None if chemo is None else chemo.displacing_distance_A,
                    None if chemo is None else chemo.margin_A,
                    None if chemo is None else chemo.margin_authority.value,
                    None if chemo is None else chemo.margin_source,
                    None if cof is None else cof.present,
                    None if cof is None else cof.observed_component,
                    None if cof is None else cof.observed_identity,
                    None if cof is None else cof.required_identity,
                    None if cof is None else cof.observed_state.value,
                    None if cof is None else cof.required_state.value,
                    None if cof is None else cof.identity_ok,
                    None if cof is None else cof.state_ok,
                    None if cof is None else cof.placement_satisfied,
                    None if clash is None else clash.count,
                    None if clash is None else clash.tolerance_A,
                    None if clash is None else clash.complete,
                    p.circular_constraints, p.circular_satisfied,
                    p.independent_satisfied, p.independent_total,
                    p.independent_gating, p.independent_gating_satisfied,
                    p.restrained_gating, p.carries_independent_test,
                    p.independently_satisfied,
                    p.entirely_circular, p.restraint_name_mismatch,
                    p.face, p.product_configuration, p.stereo_note,
                    "; ".join(f"{k}={v}" for k, v in sorted(p.unresolved_roles.items())),
                ]
                row += [p.measurements.get(n) for n in names]
                row += [p.satisfied.get(n) for n in names]
                row += [p.authorities[n].value if n in p.authorities else None
                        for n in names]
                row += [p.aspects.get(n) for n in names]
                rows.append(row)
        geometry_path = _write_tsv(ctx.path(self.name, "catalytic_geometry.tsv"),
                                   header, rows)

        axes = sorted({name for c in candidates for name in c.scorecard})
        sc_header = [
            "candidate_id", "family", "catalytic_template_id", "template_authority",
            "template_family_match", "template_family_note",
            "passes_gates", "has_unresolved_gate", "disqualified",
            "disqualification_reason", "input_errors",
            "n_poses", "n_decided", "n_mechanism_satisfied", "n_mechanism_violated",
            "n_outside_uncalibrated_window", "n_not_measurable", "n_input_error",
            "was_tested", "robustness_G", "robustness_basis",
            "n_independently_tested", "n_independently_satisfied",
            "sampling_G_uncorrected", "wilson_lower", "wilson_upper",
            "robustness_level", "stereo_call", "stereo_target_poses",
            "stereo_opposite_poses", "stereo_undetermined_poses",
            "predicted_ee_pct", "ee_calibration_source",
        ]
        for axis in axes:
            sc_header += [f"level:{axis}", f"value:{axis}", f"gate:{axis}"]

        sc_rows: list[list[Any]] = []
        for candidate, e in zip(candidates, evaluations):
            row = [
                candidate.candidate_id, candidate.family.family_name, e.template_id,
                e.authority.value, e.family_match, e.family_note,
                candidate.passes_gates,
                candidate.has_unresolved_gate, candidate.disqualified,
                candidate.disqualification_reason,
                candidate.input_errors, e.n_poses, e.n_decided, e.n_satisfied,
                e.n_violated, e.n_provisional, e.n_not_measurable, e.n_input_error,
                e.was_tested, e.robustness_G, e.robustness_basis,
                e.n_independently_tested, e.n_independently_satisfied,
                e.sampling_G, e.wilson_lo, e.wilson_hi,
                e.robustness_level.value, e.stereo.call, e.stereo.target_face_poses,
                e.stereo.opposite_face_poses, e.stereo.undetermined_poses,
                e.stereo.predicted_ee_pct, e.stereo.calibration_source,
            ]
            for axis in axes:
                dim = candidate.scorecard.get(axis)
                row += [None if dim is None else dim.level.value,
                        None if dim is None else dim.value,
                        None if dim is None else dim.gate_passed]
            sc_rows.append(row)
        scorecard_path = _write_tsv(ctx.path(self.name, "candidate_scorecards.tsv"),
                                    sc_header, sc_rows)

        explain_path = ctx.path(self.name, "candidate_explanations.txt")
        blocks: list[str] = []
        for e in evaluations:
            g = "not computed" if e.robustness_G is None else f"{e.robustness_G:.3f}"
            blocks.append(
                "=" * 70 + "\n" + e.explanation + "\n"
                f"Pose outcomes: satisfied={e.n_satisfied}, "
                f"violated={e.n_violated}, "
                f"outside-uncalibrated-window={e.n_provisional}, "
                f"not-measurable={e.n_not_measurable}, "
                f"input-error={e.n_input_error}\n"
                f"Robustness G (circularity-corrected)={g} over "
                f"{e.n_independently_tested} independently testable pose(s) of "
                f"{e.n_decided} decided; level {e.robustness_level.value}; window "
                f"authority {e.authority.value}\n"
                f"  {e.robustness_basis}\n"
                f"Stereochemistry: {e.stereo.call} -- {e.stereo.basis}\n"
            )
        explain_path.write_text("\n".join(blocks), encoding="utf-8")
        return geometry_path, scorecard_path, explain_path

    # -- reporting ----------------------------------------------------------
    @staticmethod
    def _evaluation_payload(e: CandidateEvaluation) -> dict[str, Any]:
        """Machine-readable roll-up, with the failure kinds kept in separate keys."""
        return {
            "candidate_id": e.candidate_id,
            "catalytic_template_id": e.template_id,
            "template_authority": e.authority.value,
            "has_catalytic_template": e.has_template,
            "template_family_matches_candidate": e.family_match,
            "template_family_note": e.family_note,
            "n_poses": e.n_poses,
            "n_decided": e.n_decided,
            "counts": {
                "mechanism_satisfied": e.n_satisfied,
                "mechanism_violated": e.n_violated,
                "outside_uncalibrated_window": e.n_provisional,
                "not_measurable": e.n_not_measurable,
                "input_error": e.n_input_error,
            },
            "was_tested": e.was_tested,
            "robustness_G": e.robustness_G,
            "robustness_is_circularity_corrected": True,
            "robustness_basis": e.robustness_basis,
            "n_independently_tested": e.n_independently_tested,
            "n_independently_satisfied": e.n_independently_satisfied,
            "sampling_G": e.sampling_G,
            "sampling_G_note": (
                "uncorrected satisfied/decided fraction, reported for audit only; "
                "constraints that were restrained during modelling are counted in "
                "it, so it is not evidence about the enzyme"
            ),
            "robustness_wilson_95": (None if e.wilson_lo is None
                                     else [e.wilson_lo, e.wilson_hi]),
            "robustness_level": e.robustness_level.value,
            "stereo": e.stereo.model_dump(mode="json"),
            "stereo_is_circularity_corrected": True,
            "stereo_poses_excluded_as_circular": e.stereo_poses_excluded_as_circular,
            "input_errors": list(e.input_errors),
            "entirely_circular_poses": e.entirely_circular_poses(),
            "poses": [
                {
                    "pose_id": p.pose_id,
                    "outcome": p.outcome.value,
                    "record_outcome": p.outcome.as_record_outcome().value,
                    "reason": p.reason,
                    "gating_passed": p.gating_passed,
                    "measurements": p.measurements,
                    "satisfied": p.satisfied,
                    "circular_constraints": list(p.circular_constraints),
                    "independent_satisfied": p.independent_satisfied,
                    "independent_total": p.independent_total,
                    "independent_gating": list(p.independent_gating),
                    "independent_gating_satisfied": list(p.independent_gating_satisfied),
                    "restrained_gating": list(p.restrained_gating),
                    "carries_independent_test": p.carries_independent_test,
                    "counts_toward_robustness": p.independently_satisfied,
                    "entirely_circular": p.entirely_circular,
                    "target_in_reactive_position": (
                        None if p.chemoselectivity is None
                        else p.chemoselectivity.target_in_reactive_position),
                    "chemoselectivity_margin_authority": (
                        None if p.chemoselectivity is None
                        else p.chemoselectivity.margin_authority.value),
                    "chemoselectivity_margin_source": (
                        None if p.chemoselectivity is None
                        else p.chemoselectivity.margin_source),
                    "clash_count": None if p.clash is None else p.clash.count,
                    "face": p.face,
                    "product_configuration": p.product_configuration,
                    "unresolved_roles": p.unresolved_roles,
                }
                for p in e.poses
            ],
            "explanation": e.explanation,
        }

    def _summarise(self, result: ToolResult,
                   evaluations: Sequence[CandidateEvaluation]) -> None:
        """Set the status and a message that never rounds a gap into a negative."""
        n_cand = len(evaluations)
        tested = [e for e in evaluations if e.was_tested]
        refused = [e for e in evaluations if e.family_match is False]
        # A refused candidate is untested too, but "re-run the modelling" is the
        # wrong next step for it: nothing is wrong with its poses. Keeping the
        # two apart means each gets the follow-up that would actually fix it.
        untested = [e for e in evaluations
                    if not e.was_tested and e.family_match is not False]
        n_poses = sum(e.n_poses for e in evaluations)
        n_failed = sum(e.n_not_measurable + e.n_input_error for e in evaluations)

        if refused:
            result.status = Status.PARTIAL
            shown = ", ".join(e.candidate_id for e in refused[:8])
            more = " ..." if len(refused) > 8 else ""
            result.add_flag(
                "candidates_refused_on_family_mismatch", Severity.BLOCKER,
                f"{len(refused)} of {n_cand} candidate(s) were refused because "
                f"the catalytic template attached to them describes another "
                f"family: {shown}{more}. They were disqualified rather than "
                f"scored; nothing was measured about their mechanism.",
            )

        if untested:
            result.status = Status.PARTIAL
            shown = ", ".join(e.candidate_id for e in untested[:8])
            more = " ..." if len(untested) > 8 else ""
            result.add_flag(
                "candidates_not_tested", Severity.WARN,
                f"{len(untested)} of {n_cand} candidate(s) produced no decidable "
                f"pose: {shown}{more}. These are modelling failures, not inactive "
                f"enzymes, and they must not be ranked below candidates with a "
                f"genuine G of 0.",
            )
            result.add_next(
                "model_complexes",
                "Re-run complex generation for the candidates whose poses could not "
                "be measured; an unmeasured candidate is untested, not negative",
                {"candidate_ids": [e.candidate_id for e in untested]},
            )
        for e in evaluations:
            circular = e.entirely_circular_poses()
            if circular:
                result.add_next(
                    "model_complexes",
                    "Rebuild without restraints so that satisfied constraints can "
                    "become independent evidence",
                    {"candidate_id": e.candidate_id, "pose_ids": circular,
                     "release_restraints": True},
                )
        result.message = (
            f"{n_cand} candidate(s), {n_poses} pose(s): "
            f"{sum(e.n_satisfied for e in evaluations)} satisfied the mechanism, "
            f"{sum(e.n_violated for e in evaluations)} violated it against a "
            f"calibrated window, "
            f"{sum(e.n_provisional for e in evaluations)} fell outside an "
            f"uncalibrated window (reported, not rejected), and {n_failed} could "
            f"not be measured or carried wrong inputs. {len(tested)} candidate(s) "
            f"were actually tested. No total score was computed; ranking is gates, "
            f"ordinal levels and Pareto non-domination."
        )
