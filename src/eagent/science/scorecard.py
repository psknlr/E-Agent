"""Ranking candidates without inventing a total score.

WHY THIS MODULE REFUSES TO PRODUCE A NUMBER
-------------------------------------------
The nine axes in :data:`~eagent.schemas.candidate.SCORE_DIMENSIONS` are not
measurements of one quantity on nine scales. A pLDDT is a per-residue
confidence in ``[0, 100]`` produced by a structure predictor; a docking score
is an unbounded pseudo-energy whose sign convention depends on the scoring
function; a hydride-transfer distance is an angstrom measurement with a
mechanism-derived window; "functional literature evidence" is an ordinal
statement about *attribution*, not about magnitude. There exists no calibration
set in this repository mapping any weighted combination of those onto the
probability that a given enzyme turns over a given substrate. A weighted sum
would therefore manufacture a precise, reproducible, completely unjustified
ordering -- and because it is precise and reproducible it would survive review.
:func:`refuse_linear_blend` exists so that the next contributor who types
``0.4 * pLDDT + 0.3 * dock + 0.3 * distance`` is stopped with an explanation
rather than a merge conflict.

WHAT IS USED INSTEAD, IN ORDER
------------------------------
1. :class:`FeasibilityGate` -- hard, mechanism-derived eligibility checks.
   A candidate that cannot be given the catalytic machinery the template
   requires, or that cannot be given the cofactor the mechanism needs, is not
   a low-scoring candidate; it is not a candidate. Gates are three-valued:
   ``PASS`` / ``FAIL`` / ``UNEVALUATED``. ``UNEVALUATED`` is never folded into
   ``FAIL``; it is routed to an :class:`~eagent.envelope.Uncertainty` so the
   controller can go and evaluate it.
2. Ordinal levels -- :class:`~eagent.schemas.candidate.ConfidenceLevel` per
   axis, because "better supported" is a defensible ordering while "1.7 times
   as good" is not.
3. Within-family comparison -- :func:`rank_within_family`. A docking score from
   a short-chain dehydrogenase pocket and one from an aldo-keto reductase
   pocket are not on the same scale; comparing them ranks pocket volume.
4. :func:`pareto_front` -- non-domination, which is the only multi-objective
   ordering that needs no exchange rate between the objectives.
5. :func:`lexicographic_rank` -- the defensible default *total* order when a
   single list is required and no trained model exists. The priority order is a
   stated configuration, visible and arguable, not a hidden set of weights.

TWO ERROR KINDS THAT MUST NOT BE MIXED
--------------------------------------
``INPUT_DEFECT``
    A wrong sequence (fragment, non-standard letters, a structure of a
    different protein), a wrong ligand (the oxidised cofactor where the
    mechanism needs the reduced one), or an unresolved/contradictory
    stereochemical specification. These are **defects to be repaired**. They
    disqualify the candidate until fixed and they are never traded off against
    a good docking score. See :func:`input_defects`.

``EVIDENCE_WEAKNESS``
    Low pocket pLDDT, few sampled poses, two modelling routes disagreeing, an
    uncalibrated geometric window, no literature at the sequence level. These
    lower the *strength of the evidence* and nothing else. They are **not**
    evidence that the enzyme is inactive. A candidate with weak evidence is a
    candidate we have not looked at hard enough, which is precisely the
    population an uncertainty-probe slot exists to sample. Reading weakness as
    a negative is how a screening campaign ends up testing only the proteins
    the models already understood. See :func:`evidence_weaknesses`.

Nothing in this module hard-codes a catalytic threshold. Geometric windows are
read from the :class:`~eagent.schemas.templates.GeometryConstraint` objects of a
sourced :class:`~eagent.schemas.templates.CatalyticTemplate`, applied by
:mod:`eagent.science.geometry` before anything arrives here. The only
module-level numbers are sampling-adequacy defaults re-exported from
:mod:`eagent.science.robustness`, which describe the modelling protocol rather
than the enzyme.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

from ..envelope import QCFlag, Severity, Uncertainty
from ..errors import FabricationGuardError, TemplateError
from ..schemas import (
    Candidate,
    CatalyticTemplate,
    CofactorState,
    ConfidenceLevel,
    EvidenceStrength,
    GeometryReport,
    SCORE_DIMENSIONS,
    ScoreDimension,
    Stereochemistry,
    TaskSpec,
    cofactor_state_from_ligand_code,
)
from .robustness import (
    DEFAULT_MIN_VALID_POSES,
    classify_robustness,
    cross_method_agreement,
)

__all__ = [
    "COFACTOR_IDENTITY",
    "DEFAULT_LEXICOGRAPHIC_ORDER",
    "GATE_NAMES",
    "ErrorKind",
    "GateOutcome",
    "FeasibilityGate",
    "cofactor_identity",
    "input_defects",
    "evidence_weaknesses",
    "gate_catalytic_machinery",
    "gate_cofactor_compatibility",
    "gate_input_integrity",
    "gate_stereochemistry_resolved",
    "evaluate_feasibility_gates",
    "gate_uncertainties",
    "scorecard_qc_flags",
    "build_scorecard",
    "apply_scorecard",
    "Comparable",
    "MEASURED_SCALE",
    "ORDINAL_SCALE",
    "NO_SCALE",
    "comparable",
    "objective_comparable",
    "objective_value",
    "dominates",
    "pareto_front",
    "rank_within_family",
    "lexicographic_rank",
    "refuse_linear_blend",
    "explain",
]


# ==========================================================================
# Named defaults. None of these is a catalytic threshold.
# ==========================================================================

#: Base cofactor identity for the nicotinamide and flavin names and PDB
#: chemical-component codes this pipeline actually sees. The *oxidation state*
#: is deliberately not encoded here: it travels separately so that NADP+ and
#: NADPH compare as the same identity in a different state, which is what makes
#: the "right cofactor, wrong state" failure visible instead of silent.
#:
#: Anything absent from this table resolves to ``None`` and sends the cofactor
#: gate to ``UNEVALUATED``. Guessing that an unknown string beginning with
#: "NAD" means NAD is exactly the substitution the harness forbids.
#: CALIBRATION: extend per family as new cofactor classes enter the pipeline.
COFACTOR_IDENTITY: dict[str, str] = {
    "NAD": "NAD", "NAD+": "NAD", "NADH": "NAD", "NAI": "NAD",
    "NADP": "NADP", "NADP+": "NADP", "NADPH": "NADP",
    "NAP": "NADP", "NDP": "NADP",
    "FAD": "FAD", "FADH2": "FAD",
    "FMN": "FMN", "FMNH2": "FMN",
}

#: Priority order used by :func:`lexicographic_rank` and
#: :func:`rank_within_family` when the caller states none.
#:
#: The order encodes two stated beliefs, both arguable and both visible:
#: *measured beats modelled* (a sequence-level experimental record outranks any
#: prediction) and *mechanism beats score* (catalytic geometry against a sourced
#: template outranks a docking number). Novelty is last because it is an
#: exploration axis, not a quality axis -- a sequence being unlike everything
#: known is a reason to spend a diversity slot, not a reason to believe it
#: works.
#:
#: CALIBRATION: this is a configuration, not a validated optimum. It should be
#: revisited against round-1 outcomes for the family in hand.
DEFAULT_LEXICOGRAPHIC_ORDER: tuple[str, ...] = (
    "functional_literature_evidence",
    "family_mechanism_compatibility",
    "catalytic_geometry",
    "local_structure_confidence",
    "model_uncertainty",
    "substrate_specificity_model",
    "docking_result",
    "expression_developability_risk",
    "sequence_pocket_novelty",
)

#: Names of the feasibility-gate entries that :func:`build_scorecard` adds to
#: the scorecard dictionary alongside the nine axes. Kept distinct from
#: :data:`~eagent.schemas.candidate.SCORE_DIMENSIONS` so a gate can never be
#: mistaken for a tradeable objective.
GATE_NAMES: tuple[str, ...] = (
    "catalytic_machinery_mappable",
    "cofactor_compatible",
    "inputs_free_of_defects",
    "stereochemistry_resolved",
)

_EXPERIMENTAL_STRUCTURE_SOURCES: frozenset[str] = frozenset({"pdb_complex", "pdb_apo"})

_EVIDENCE_LEVEL_BY_STRENGTH: dict[EvidenceStrength, ConfidenceLevel] = {
    EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL: ConfidenceLevel.STRONG,
    EvidenceStrength.HOMOLOG_EXPERIMENTAL: ConfidenceLevel.MODERATE,
    EvidenceStrength.EC_SPECIES_MAPPED: ConfidenceLevel.WEAK,
    EvidenceStrength.ANNOTATION_ONLY: ConfidenceLevel.WEAK,
    EvidenceStrength.COMPUTATIONAL_CONSTRUCT: ConfidenceLevel.INSUFFICIENT,
}


# ==========================================================================
# The two error kinds
# ==========================================================================

class ErrorKind(str, enum.Enum):
    """The two categories of problem, which must never be summed together.

    Keeping them apart is the whole point. An input defect is a bug in what we
    fed the pipeline and it is repairable; evidence weakness is a statement
    about how much we know and it is informative. A ranking that mixes them
    will demote an enzyme for being unstudied and promote one whose inputs were
    quietly wrong.
    """

    INPUT_DEFECT = "input_defect"
    EVIDENCE_WEAKNESS = "evidence_weakness"

    def disqualifies(self) -> bool:
        """Only a defect disqualifies; weakness never does."""
        return self is ErrorKind.INPUT_DEFECT


class GateOutcome(str, enum.Enum):
    """Three-valued gate result.

    ``UNEVALUATED`` is a distinct state rather than a ``False`` because the
    difference between "the mechanism says no" and "we did not check" is the
    difference between discarding a candidate and scheduling a computation.
    Collapsing the two silently discards everything the pipeline has not yet
    got round to looking at, which biases the batch toward the easy cases.
    """

    PASS = "pass"
    FAIL = "fail"
    UNEVALUATED = "unevaluated"

    @property
    def as_bool(self) -> bool | None:
        """``True`` / ``False`` / ``None`` for ``ScoreDimension.gate_passed``."""
        return {"pass": True, "fail": False, "unevaluated": None}[self.value]


@dataclass(frozen=True)
class FeasibilityGate:
    """One hard eligibility check, with its verdict and what to do about it.

    A gate carries more than a boolean because the three verdicts need three
    different follow-ups: a pass needs a basis a reviewer can check, a failure
    needs a remedy (defects are repairable), and an unevaluated gate needs a
    question someone can go and answer. Compressing that to ``bool`` loses
    exactly the part the controller routes on.
    """

    name: str
    question: str
    outcome: GateOutcome
    basis: str = ""
    kind: ErrorKind | None = None
    remedy: str = ""
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.outcome is GateOutcome.PASS

    @property
    def failed(self) -> bool:
        return self.outcome is GateOutcome.FAIL

    @property
    def unevaluated(self) -> bool:
        return self.outcome is GateOutcome.UNEVALUATED

    def to_dimension(self) -> ScoreDimension:
        """Render the gate as a scorecard entry with ``is_gate=True``.

        ``level`` mirrors ``gate_passed`` so that code which only reads ordinal
        levels still sorts a failed gate to the bottom: ``CONTRADICTORY`` has
        rank 0 and ``INSUFFICIENT`` rank 1, so an unevaluated gate never sorts
        below a failed one. ``direction`` is ``categorical`` because a gate is
        not an objective to be maximised, and :func:`objective_value` refuses
        gate dimensions outright.
        """
        level = {
            GateOutcome.PASS: ConfidenceLevel.STRONG,
            GateOutcome.FAIL: ConfidenceLevel.CONTRADICTORY,
            GateOutcome.UNEVALUATED: ConfidenceLevel.INSUFFICIENT,
        }[self.outcome]
        parts = [self.basis] if self.basis else []
        if self.remedy:
            parts.append(f"remedy: {self.remedy}")
        parts.extend(self.notes)
        return ScoreDimension(
            name=self.name,
            level=level,
            value=None,
            unit=None,
            direction="categorical",
            basis=" | ".join(parts),
            is_gate=True,
            gate_passed=self.outcome.as_bool,
        )

    def to_uncertainty(self) -> Uncertainty | None:
        """The question an unevaluated gate poses, or ``None`` if it was decided.

        This is the route that keeps ``UNEVALUATED`` from degenerating into a
        silent rejection: every undecided gate leaves the module as an explicit,
        actionable question attached to the candidate.
        """
        if not self.unevaluated:
            return None
        return Uncertainty(
            code=f"gate_unevaluated:{self.name}",
            question=f"{self.question} -- not decided: {self.basis}",
            affects=[],
            resolvable_by=self.remedy or "run the missing computation or supply the input",
        )


# ==========================================================================
# Template resolution
# ==========================================================================

def _resolve_catalytic_template(templates: Any, template_id: str | None) -> CatalyticTemplate | None:
    """Fetch a :class:`CatalyticTemplate` from whatever template library is injected.

    The harness injects ``RunContext.templates`` as an opaque object, so this
    accepts a mapping, a library exposing ``catalytic``/``get_catalytic``/``get``,
    or a single template. It returns ``None`` only when there is genuinely no
    library to ask. When a library *is* present and the candidate names a
    template it does not hold, that is a broken run configuration and raises
    :class:`~eagent.errors.TemplateError` rather than quietly degrading the
    candidate to "no mechanism known" -- the second behaviour would turn a
    missing file into an apparent property of the enzyme.
    """
    if template_id is None:
        return None
    if templates is None:
        return None
    if isinstance(templates, CatalyticTemplate):
        if templates.template_id == template_id:
            return templates
        raise TemplateError(
            f"catalytic template '{template_id}' requested but the injected "
            f"template is '{templates.template_id}'"
        )

    found: Any = None
    try:
        if isinstance(templates, Mapping):
            found = templates.get(template_id)
            if found is None:
                sub = templates.get("catalytic")
                if isinstance(sub, Mapping):
                    found = sub.get(template_id)
        else:
            store = getattr(templates, "catalytic_templates", None)
            if isinstance(store, Mapping):
                found = store.get(template_id)
            if found is None:
                for attr in ("catalytic", "get_catalytic", "catalytic_template", "get"):
                    fn = getattr(templates, attr, None)
                    if callable(fn):
                        found = fn(template_id)
                        if found is not None:
                            break
    except KeyError as exc:
        raise TemplateError(
            f"catalytic template '{template_id}' is not in the injected library"
        ) from exc

    if found is None:
        raise TemplateError(
            f"catalytic template '{template_id}' is not in the injected library; "
            f"a candidate may not be scored against a template that was not loaded"
        )
    if not isinstance(found, CatalyticTemplate):
        raise TemplateError(
            f"'{template_id}' resolved to {type(found).__name__}, not a "
            f"CatalyticTemplate"
        )
    return found


def cofactor_identity(*names: str | None) -> str | None:
    """Base cofactor identity from a name or PDB ligand code, or ``None``.

    Returns ``None`` for anything not in :data:`COFACTOR_IDENTITY` instead of
    pattern-matching the string, because a near-miss match here would silently
    declare an enzyme cofactor-compatible with a molecule it has never seen.
    """
    for name in names:
        if not name:
            continue
        key = str(name).strip().upper().replace(" ", "")
        if key in COFACTOR_IDENTITY:
            return COFACTOR_IDENTITY[key]
    return None


# ==========================================================================
# The two error kinds, as detectors
# ==========================================================================

def input_defects(candidate: Candidate, task: TaskSpec | None = None) -> list[str]:
    """Detected defects in what was fed in. Each one disqualifies until repaired.

    Deliberately short and literal: every entry names something that is *wrong*,
    not something that is *unknown*. A sequence with non-standard letters is a
    defect (a model built on it is a model of a different molecule); a sequence
    whose fragment status was never annotated is not a defect, it is a gap, and
    it belongs in :func:`evidence_weaknesses`.
    """
    out: list[str] = list(candidate.input_errors)
    if candidate.disqualified and candidate.disqualification_reason:
        out.append(f"disqualified: {candidate.disqualification_reason}")
    elif candidate.disqualified:
        out.append("disqualified (no reason recorded)")

    rec = candidate.sequence_record
    if rec.has_nonstandard_residues:
        out.append(
            "sequence contains letters outside the 20 standard residues; any "
            "structure built from it models a different molecule"
        )
    if rec.is_fragment is True:
        out.append(
            "sequence is annotated as a fragment; a truncated chain cannot be "
            "scored for pocket geometry or ordered as a construct"
        )

    if task is not None:
        sub = task.reaction.substrate
        if not sub.is_structurally_defined:
            out.append(
                "task substrate has no isomeric SMILES or molfile; a prose name "
                "cannot be docked, measured against, or assayed"
            )
        prod = task.reaction.product
        if prod.creates_new_stereocenter is True \
                and sub.is_prochiral is False:
            out.append(
                "task declares that the product gains a stereocentre while the "
                "substrate is marked not prochiral; these cannot both be true"
            )
    return out


def evidence_weaknesses(candidate: Candidate) -> list[str]:
    """Things we do not know well enough. **None of these is a negative result.**

    Every entry here lowers the strength of a claim and must leave the
    candidate eligible. The scientific trap this function exists to label is the
    one that silently converts "the predictor was unconfident about this loop"
    into "this enzyme does not work" -- which systematically removes exactly the
    uncharacterised proteins a mining campaign is run to find.
    """
    out: list[str] = []
    rec = candidate.sequence_record
    if rec.is_fragment is None:
        out.append("fragment status not annotated; completeness unverified")
    if rec.percent_identity is None:
        out.append("no percent identity to the seed recorded; retrieval distance unknown")

    st = candidate.best_structure
    if st is None:
        out.append("no structure selected; all structural axes are unevaluated")
    else:
        if st.pocket_plddt is None and st.source not in _EXPERIMENTAL_STRUCTURE_SOURCES:
            out.append(
                "pocket-local pLDDT not computed; a whole-chain mean cannot stand "
                "in for it because the mean hides a disordered active-site loop"
            )
        if st.missing_regions:
            out.append(f"structure has {len(st.missing_regions)} unmodelled region(s)")
        if st.is_mutant_relative_to_candidate:
            out.append(
                "chosen structure is a mutant relative to the candidate sequence: "
                f"{', '.join(st.mutations_in_structure) or 'mutations not listed'}"
            )

    if not candidate.poses:
        out.append("no complex pose built; geometry and docking axes are unevaluated")
    if not candidate.evidence:
        out.append("no literature or database evidence attached to this sequence")

    for report in candidate.geometry:
        if report.circular_constraints and report.independent_total == 0:
            out.append(
                f"pose {report.pose_id}: every satisfied constraint was restrained "
                f"during modelling, so none of them is independent evidence"
            )
    if candidate.family.signals_conflicting:
        out.append(
            "family signals conflict: "
            + ", ".join(candidate.family.signals_conflicting)
        )
    return out


# ==========================================================================
# The feasibility gates
# ==========================================================================

def gate_catalytic_machinery(candidate: Candidate, templates: Any = None) -> FeasibilityGate:
    """Can the template's catalytic roles be placed on this sequence at all?

    This is the first hard gate because every downstream geometric criterion is
    defined *relative to named catalytic atoms*. If the catalytic tyrosine has
    no counterpart in the candidate, a hydride-transfer distance measured to
    whatever residue happens to sit nearby is not a weaker version of the right
    measurement; it is a measurement of something else.

    A conservative substitution recorded in
    :attr:`~eagent.schemas.candidate.CatalyticMapping.substituted_roles` is
    accepted, because families do tolerate them and the substitution is written
    down where a reviewer can disagree with it.
    """
    name = "catalytic_machinery_mappable"
    question = "Can every catalytic role of the template be placed on this sequence?"
    mapping = candidate.catalytic_mapping

    if not mapping.catalytic_template_id and not mapping.role_to_residue:
        return FeasibilityGate(
            name, question, GateOutcome.UNEVALUATED,
            basis="no catalytic mapping has been attempted for this candidate",
            remedy="run the catalytic-site mapping interface against a sourced "
                   "CatalyticTemplate",
        )

    unexplained = [r for r in mapping.missing_roles if r not in mapping.substituted_roles]
    if unexplained:
        return FeasibilityGate(
            name, question, GateOutcome.FAIL, kind=ErrorKind.INPUT_DEFECT,
            basis=f"catalytic role(s) absent with no recorded substitution: "
                  f"{', '.join(sorted(unexplained))}",
            remedy="verify the alignment; if the substitution is real and the "
                   "family tolerates it, record it in substituted_roles with its "
                   "evidence",
        )

    template = _resolve_catalytic_template(templates, mapping.catalytic_template_id)
    notes: list[str] = []
    if template is not None:
        wanted = {str(r.get("label")) for r in template.catalytic_residues
                  if r.get("label") is not None}
        covered = set(mapping.role_to_residue) | set(mapping.substituted_roles) \
            | set(mapping.missing_roles)
        uncovered = sorted(wanted - covered)
        if uncovered:
            return FeasibilityGate(
                name, question, GateOutcome.UNEVALUATED,
                basis=f"template {template.template_id} declares role(s) the mapping "
                      f"never reports on: {', '.join(uncovered)}",
                remedy="re-run the mapping against the full template role set; an "
                       "unreported role is not an absent role",
            )
        if template.provenance.is_theoretical:
            notes.append(
                f"template {template.template_id} is a labelled theoretical model, "
                f"not an experimental structure"
            )
    elif mapping.catalytic_template_id:
        notes.append(
            "no template library was injected, so the mapping was checked only "
            "for internal consistency"
        )

    if mapping.substituted_roles:
        notes.append(
            "accepted with substitution(s): "
            + ", ".join(f"{k}->{v}" for k, v in sorted(mapping.substituted_roles.items()))
        )
    if not mapping.role_to_residue:
        return FeasibilityGate(
            name, question, GateOutcome.UNEVALUATED,
            basis="mapping names a template but assigns no residues",
            remedy="re-run the catalytic-site mapping interface",
        )
    return FeasibilityGate(
        name, question, GateOutcome.PASS,
        basis=f"{len(mapping.role_to_residue)} catalytic role(s) placed"
              + (f", alignment quality {mapping.alignment_quality:.3g}"
                 if mapping.alignment_quality is not None else ""),
        notes=tuple(notes),
    )


def gate_cofactor_compatibility(
    candidate: Candidate, task: TaskSpec, templates: Any = None
) -> FeasibilityGate:
    """Does the run offer the cofactor, in the oxidation state, the mechanism needs?

    The trap is the oxidation state. ``NADP+`` and ``NADPH`` share a name stem,
    a PDB-adjacent ligand code and most of their atoms, and a hydride-transfer
    mechanism can only run on the reduced form. A pipeline that matches on the
    name alone will declare a ketoreductase cofactor-compatible with the
    oxidised cofactor and then measure a hydride-transfer distance to a
    nicotinamide that has no hydride to give.

    Both sides are resolved through :data:`COFACTOR_IDENTITY` and
    :func:`~eagent.schemas.chem.cofactor_state_from_ligand_code`, and anything
    either table cannot resolve leaves the gate ``UNEVALUATED`` rather than
    matched by similarity.
    """
    name = "cofactor_compatible"
    question = ("Is the cofactor this mechanism requires, in the required oxidation "
                "state, among the ones the run will supply?")
    template = _resolve_catalytic_template(
        templates, candidate.catalytic_mapping.catalytic_template_id
    )
    if template is None:
        return FeasibilityGate(
            name, question, GateOutcome.UNEVALUATED,
            basis="no catalytic template resolved, so the cofactor requirement of "
                  "this candidate's mechanism is unknown",
            remedy="assign a sourced CatalyticTemplate to this candidate",
        )
    if template.required_cofactor is None:
        return FeasibilityGate(
            name, question, GateOutcome.UNEVALUATED,
            basis=f"template {template.template_id} declares no required cofactor; "
                  f"'needs none' and 'never filled in' are not distinguishable here",
            remedy="state required_cofactor (or an explicit 'none') in the template",
        )
    required = cofactor_identity(template.required_cofactor, *template.cofactor_ligand_codes)
    if required is None:
        return FeasibilityGate(
            name, question, GateOutcome.UNEVALUATED,
            basis=f"cofactor '{template.required_cofactor}' is not in the identity "
                  f"table, so it cannot be compared without guessing",
            remedy="add the cofactor to COFACTOR_IDENTITY with its source",
        )
    required_state = template.required_cofactor_state
    if required_state is CofactorState.UNKNOWN:
        return FeasibilityGate(
            name, question, GateOutcome.UNEVALUATED,
            basis=f"template {template.template_id} gives no oxidation state for "
                  f"{template.required_cofactor}",
            remedy="record required_cofactor_state in the template",
        )

    offered = task.conditions.cofactor_options
    if not offered:
        return FeasibilityGate(
            name, question, GateOutcome.UNEVALUATED,
            basis="the task lists no cofactor options, so there is nothing to "
                  "compare the requirement against",
            remedy="fill conditions.cofactor_options via TaskSpec.resolve",
        )

    unresolved: list[str] = []
    state_mismatch: list[str] = []
    for spec in offered:
        ident = cofactor_identity(spec.name, spec.ligand_code)
        state = spec.state
        if state is CofactorState.UNKNOWN:
            state = cofactor_state_from_ligand_code(spec.ligand_code)
        if ident is None or state is CofactorState.UNKNOWN:
            unresolved.append(f"{spec.name}({spec.ligand_code or 'no code'})")
            continue
        if ident == required and state is required_state:
            notes: list[str] = []
            pref = candidate.family.cofactor_preference
            pref_id = cofactor_identity(pref)
            if pref_id is not None and pref_id != required:
                notes.append(
                    f"family annotation prefers {pref} while the catalytic template "
                    f"requires {template.required_cofactor}; the template was used "
                    f"and the disagreement is recorded, not averaged"
                )
            for st in candidate.structures:
                if st.cofactor_state_in_structure not in (CofactorState.UNKNOWN,
                                                          required_state):
                    notes.append(
                        f"structure {st.structure_id} holds the "
                        f"{st.cofactor_state_in_structure.value} cofactor; that is a "
                        f"common crystallographic state and is not a gate failure, "
                        f"but the pose must not inherit it"
                    )
            return FeasibilityGate(
                name, question, GateOutcome.PASS,
                basis=f"{spec.describe()} supplies {required} in the "
                      f"{required_state.value} state required by "
                      f"{template.template_id}",
                notes=tuple(notes),
            )
        if ident == required:
            state_mismatch.append(f"{spec.name} is {state.value}")

    if state_mismatch:
        return FeasibilityGate(
            name, question, GateOutcome.FAIL, kind=ErrorKind.INPUT_DEFECT,
            basis=f"the run supplies {required} only in the wrong oxidation state "
                  f"({'; '.join(state_mismatch)}); the mechanism in "
                  f"{template.template_id} needs {required_state.value}",
            remedy="add the correctly-stated cofactor (and its recycling system) to "
                   "conditions.cofactor_options",
        )
    if unresolved and len(unresolved) == len(offered):
        return FeasibilityGate(
            name, question, GateOutcome.UNEVALUATED,
            basis="no offered cofactor could be resolved to an identity and an "
                  "oxidation state: " + ", ".join(unresolved),
            remedy="give each cofactor option a known name or PDB ligand code and "
                   "an explicit state",
        )
    return FeasibilityGate(
        name, question, GateOutcome.FAIL, kind=ErrorKind.INPUT_DEFECT,
        basis=f"this candidate's mechanism requires {required} "
              f"({required_state.value}); the run offers only "
              + ", ".join(s.describe() for s in offered),
        remedy="either supply the required cofactor or drop candidates of this "
               "cofactor class from the pool deliberately",
        notes=tuple(f"unresolved option: {u}" for u in unresolved),
    )


def gate_input_integrity(candidate: Candidate, task: TaskSpec | None = None) -> FeasibilityGate:
    """Are the inputs free of *detected* defects, and were the checks runnable?

    A pass here asserts two things, and the second is the one usually skipped:
    no defect was found, **and** the checks that would have found one could
    actually run. An unrun check is not a pass. That is why an unannotated
    fragment status produces ``UNEVALUATED`` rather than ``PASS`` -- annotating
    it costs one database lookup, and pretending it was checked costs a
    construct.
    """
    name = "inputs_free_of_defects"
    question = "Are this candidate's sequence, structure and ligand inputs free of defects?"
    defects = input_defects(candidate, task)
    if defects:
        return FeasibilityGate(
            name, question, GateOutcome.FAIL, kind=ErrorKind.INPUT_DEFECT,
            basis="; ".join(defects),
            remedy="repair the input and re-run; a defect is never traded off "
                   "against a favourable score on another axis",
        )
    unrun: list[str] = []
    if candidate.sequence_record.is_fragment is None:
        unrun.append("fragment status never annotated")
    if unrun:
        return FeasibilityGate(
            name, question, GateOutcome.UNEVALUATED,
            basis="no defect found, but a check could not run: " + "; ".join(unrun),
            remedy="annotate the missing field from the source database record",
        )
    return FeasibilityGate(
        name, question, GateOutcome.PASS,
        basis="sequence is full-length with standard residues, no recorded input "
              "error, and the task substrate is structurally defined",
    )


def gate_stereochemistry_resolved(candidate: Candidate, task: TaskSpec) -> FeasibilityGate:
    """Is the stereochemical requirement of the task actually decided?

    Two distinct traps live here, and only the first is about the candidate.

    First, whether the reaction creates a stereocentre at all must be decided
    *per substrate*, never inferred from the reaction class: reduction of an
    aldehyde or of a symmetric ketone creates none, and scoring such a task
    against an enantioselectivity objective invents a requirement the chemistry
    does not have.

    Second, if a stereocentre *is* created and the operator has not said which
    configuration is wanted, then there is no target to rank against. That is an
    unresolved input, so this gate fails for every candidate -- loudly and
    identically -- rather than letting the pipeline rank enzymes against a
    preference nobody stated.

    The candidate's own :class:`~eagent.schemas.candidate.StereoCall` is
    deliberately *not* gated on: a model predicting the opposite face is a
    prediction with its own uncertainty, which belongs on a scorecard axis, not
    in an eligibility test.
    """
    name = "stereochemistry_resolved"
    question = "Is the stereochemical target of this task resolved?"
    product = task.reaction.product
    creates = product.creates_new_stereocenter
    if creates is None:
        return FeasibilityGate(
            name, question, GateOutcome.UNEVALUATED,
            basis="reaction.product.creates_new_stereocenter is unresolved; it must "
                  "be decided per substrate and never assumed from the reaction class",
            remedy="resolve reaction.product.creates_new_stereocenter via "
                   "TaskSpec.resolve with an operator or literature source",
        )
    if creates is False:
        return FeasibilityGate(
            name, question, GateOutcome.PASS,
            basis="the product gains no stereocentre, so no enantiopreference "
                  "requirement applies to this task",
        )
    target = product.target_stereochemistry
    if target is Stereochemistry.UNSPECIFIED:
        return FeasibilityGate(
            name, question, GateOutcome.FAIL, kind=ErrorKind.INPUT_DEFECT,
            basis="the product gains a stereocentre but no target configuration is "
                  "specified; candidates cannot be ranked against an unstated "
                  "preference",
            remedy="resolve reaction.product.target_stereochemistry to R, S or "
                   "racemic via TaskSpec.resolve",
        )
    if target is Stereochemistry.ACHIRAL:
        return FeasibilityGate(
            name, question, GateOutcome.FAIL, kind=ErrorKind.INPUT_DEFECT,
            basis="the task states that the product gains a stereocentre and that "
                  "the target configuration is achiral; these contradict",
            remedy="correct creates_new_stereocenter or target_stereochemistry",
        )
    notes: list[str] = []
    if task.reaction.substrate.is_prochiral is None:
        notes.append(
            "substrate prochirality not annotated; the face assignment rests on the "
            "atom mapping alone"
        )
    if candidate.stereo.call == "insufficient_evidence":
        notes.append(
            "this candidate has no stereochemical call yet; that is missing evidence, "
            "not a predicted loss of selectivity"
        )
    return FeasibilityGate(
        name, question, GateOutcome.PASS,
        basis=f"target configuration is {target.value} and the product is declared "
              f"to gain a stereocentre",
        notes=tuple(notes),
    )


def evaluate_feasibility_gates(
    candidate: Candidate, task: TaskSpec, templates: Any = None
) -> list[FeasibilityGate]:
    """Run every hard eligibility check, in a fixed order, for one candidate.

    Fixed order so two runs produce byte-identical scorecards, and *all* gates
    are always run rather than short-circuiting on the first failure: a
    candidate with two defects should be reported with two defects, because the
    operator fixing them wants the whole list, not one at a time.
    """
    return [
        gate_catalytic_machinery(candidate, templates),
        gate_cofactor_compatibility(candidate, task, templates),
        gate_input_integrity(candidate, task),
        gate_stereochemistry_resolved(candidate, task),
    ]


def gate_uncertainties(
    gates: Iterable[FeasibilityGate], subject: str | None = None
) -> list[Uncertainty]:
    """Turn every ``UNEVALUATED`` gate into an actionable question.

    This is the function that makes "never silently treat UNEVALUATED as FAIL"
    enforceable rather than aspirational: an undecided gate always leaves the
    module as an :class:`~eagent.envelope.Uncertainty` the controller can route.
    """
    out: list[Uncertainty] = []
    for gate in gates:
        unc = gate.to_uncertainty()
        if unc is None:
            continue
        if subject:
            unc = Uncertainty(code=unc.code, question=unc.question,
                              affects=[subject], resolvable_by=unc.resolvable_by)
        out.append(unc)
    return out


def scorecard_qc_flags(
    candidate: Candidate, gates: Iterable[FeasibilityGate]
) -> list[QCFlag]:
    """QC flags separating blocking defects from non-blocking weaknesses.

    The severity split is the whole content: an input defect is a
    ``BLOCKER`` because downstream steps must not consume the candidate, while
    every evidence weakness is at most a ``WARN`` because consuming a
    weakly-supported candidate is a legitimate, deliberate choice (that is what
    an uncertainty-probe slot is for).
    """
    flags: list[QCFlag] = []
    for gate in gates:
        if gate.failed:
            flags.append(QCFlag(f"gate_failed:{gate.name}", Severity.BLOCKER,
                                gate.basis, candidate.candidate_id))
        elif gate.unevaluated:
            flags.append(QCFlag(f"gate_unevaluated:{gate.name}", Severity.INFO,
                                gate.basis, candidate.candidate_id))
    for weakness in evidence_weaknesses(candidate):
        flags.append(QCFlag("evidence_weakness", Severity.WARN, weakness,
                            candidate.candidate_id))
    return flags


# ==========================================================================
# The nine axes
# ==========================================================================

def _dim(name: str, level: ConfidenceLevel, basis: str, *,
         value: float | None = None, unit: str | None = None,
         direction: str = "higher_is_better") -> ScoreDimension:
    return ScoreDimension(name=name, level=level, value=value, unit=unit,
                          direction=direction, basis=basis, is_gate=False,
                          gate_passed=None)


def _external_dimension(
    name: str, candidate_id: str, config: Mapping[str, Any]
) -> ScoreDimension | None:
    """Read an axis supplied by a model outside this repository, or ``None``.

    Two axes (substrate specificity, developability risk) require a trained
    predictor that this repository does not contain. Rather than approximate
    them from family membership or sequence length -- which would be fabrication
    wearing a number -- they are left ``INSUFFICIENT`` unless an external model
    supplies them through ``config["external_dimensions"]``.

    An external entry must carry a non-empty ``basis`` naming the model and its
    version; an unsourced number is rejected with
    :class:`~eagent.errors.FabricationGuardError`, because an anonymous float is
    indistinguishable from a guess once it is in the scorecard.
    """
    table = config.get("external_dimensions") or {}
    per_candidate = table.get(candidate_id) if isinstance(table, Mapping) else None
    if not isinstance(per_candidate, Mapping):
        return None
    entry = per_candidate.get(name)
    if entry is None:
        return None
    if not isinstance(entry, Mapping):
        raise FabricationGuardError(
            f"external dimension '{name}' for {candidate_id} must be a mapping "
            f"with at least a 'level' and a 'basis'"
        )
    basis = str(entry.get("basis") or "").strip()
    if not basis:
        raise FabricationGuardError(
            f"external dimension '{name}' for {candidate_id} carries no basis; "
            f"name the model and its version or do not supply the value"
        )
    level_raw = entry.get("level", ConfidenceLevel.INSUFFICIENT)
    level = level_raw if isinstance(level_raw, ConfidenceLevel) \
        else ConfidenceLevel(str(level_raw))
    value = entry.get("value")
    return ScoreDimension(
        name=name, level=level,
        value=None if value is None else float(value),
        unit=entry.get("unit"),
        direction=str(entry.get("direction", "higher_is_better")),
        basis=basis, is_gate=False, gate_passed=None,
    )


def _axis_functional_literature_evidence(candidate: Candidate) -> ScoreDimension:
    name = "functional_literature_evidence"
    refs = candidate.evidence
    if not refs:
        return _dim(name, ConfidenceLevel.INSUFFICIENT,
                    "no evidence reference attached; absence of literature is "
                    "absence of information, not a negative result",
                    value=0.0, unit="independent evidence groups")
    # Four databases re-curating one measurement are one piece of evidence.
    groups: set[str] = set()
    for ref in refs:
        groups.add(ref.experiment_activity_id or ref.source_doi or ref.identifier)
    best = max((r.strength for r in refs), key=lambda s: s.rank)
    level = _EVIDENCE_LEVEL_BY_STRENGTH[best]
    return _dim(
        name, level,
        f"strongest attribution is {best.value} over {len(groups)} independent "
        f"evidence group(s) from {len(refs)} reference(s); records sharing an "
        f"experiment id or DOI are counted once. Reaction direction is not "
        f"visible at this level and is checked per ExperimentRecord.",
        value=float(len(groups)), unit="independent evidence groups",
    )


def _axis_family_mechanism_compatibility(candidate: Candidate) -> ScoreDimension:
    name = "family_mechanism_compatibility"
    fam = candidate.family
    mapping = candidate.catalytic_mapping
    level = fam.confidence
    notes = [f"family call {fam.family_name or 'unassigned'} with "
             f"{fam.n_independent_signals} independent signal(s)"]
    if fam.signals_conflicting:
        level = ConfidenceLevel.CONTRADICTORY
        notes.append("conflicting signals: " + ", ".join(fam.signals_conflicting))
    elif not mapping.is_complete:
        level = min((level, ConfidenceLevel.WEAK), key=lambda lv: lv.rank)
        notes.append("catalytic mapping incomplete, so family membership alone "
                     "does not carry the mechanism")
    elif mapping.substituted_roles:
        level = min((level, ConfidenceLevel.MODERATE), key=lambda lv: lv.rank)
        notes.append("mechanism accepted with recorded role substitutions")
    return _dim(name, level, "; ".join(notes),
                value=float(fam.n_independent_signals), unit="independent signals")


def _axis_substrate_specificity_model(candidate: Candidate,
                                      config: Mapping[str, Any]) -> ScoreDimension:
    name = "substrate_specificity_model"
    external = _external_dimension(name, candidate.candidate_id, config)
    if external is not None:
        return external
    return _dim(
        name, ConfidenceLevel.INSUFFICIENT,
        "no calibrated substrate-specificity model is attached to this run. "
        "Family membership is deliberately not used as a substitute: 'it is an "
        "SDR, so it reduces this ketone' is the inference this axis exists to "
        "keep out of the ranking.",
    )


def _axis_catalytic_geometry(candidate: Candidate, min_valid_poses: int) -> ScoreDimension:
    name = "catalytic_geometry"
    reports = list(candidate.geometry)
    if not reports:
        return _dim(name, ConfidenceLevel.INSUFFICIENT,
                    "no geometry report; the mechanism conditions were never "
                    "measured for this candidate")
    decided = [r for r in reports if r.gating_passed is not None]
    if not decided:
        return _dim(name, ConfidenceLevel.INSUFFICIENT,
                    f"{len(reports)} report(s) present but none could decide the "
                    f"gating constraints (atoms unresolvable or values missing)")
    passing = [r for r in decided if r.gating_passed is True]
    well_sampled = len(decided) >= min_valid_poses
    if not passing:
        return _dim(
            name,
            ConfidenceLevel.STRONG if well_sampled else ConfidenceLevel.WEAK,
            f"no pose satisfied the template's gating constraints over "
            f"{len(decided)} decided pose(s). This is a negative geometric result "
            f"at the sampling depth achieved"
            + ("" if well_sampled else
               f" -- below the {min_valid_poses}-pose adequacy default, so it is "
               f"reported weakly"),
            value=0.0, unit="fraction of independent constraints satisfied",
        )

    def _key(r: GeometryReport) -> tuple[float, int]:
        frac = r.independent_fraction
        return (-1.0 if frac is None else frac, r.independent_total)

    best = max(passing, key=_key)
    frac = best.independent_fraction
    if best.independent_total == 0:
        return _dim(
            name, ConfidenceLevel.WEAK,
            f"pose {best.pose_id} passes gating, but every satisfied constraint was "
            f"restrained during modelling ({', '.join(best.circular_constraints) or 'all'}); "
            f"a distance that was enforced cannot corroborate the model that enforced it",
        )
    if frac is not None and frac >= 1.0:
        level = ConfidenceLevel.STRONG
    elif frac is not None and frac > 0.0:
        level = ConfidenceLevel.MODERATE
    else:
        level = ConfidenceLevel.WEAK
    if not well_sampled:
        level = min((level, ConfidenceLevel.MODERATE), key=lambda lv: lv.rank)
    return _dim(
        name, level,
        f"pose {best.pose_id} satisfies {best.independent_satisfied}/"
        f"{best.independent_total} constraints that were not restrained during "
        f"modelling, over {len(decided)} decided pose(s)"
        + ("" if well_sampled else
           f"; capped at MODERATE because fewer than {min_valid_poses} poses were "
           f"decided"),
        value=frac, unit="fraction of independent constraints satisfied",
    )


def _axis_local_structure_confidence(candidate: Candidate) -> ScoreDimension:
    name = "local_structure_confidence"
    st = candidate.best_structure
    if st is None:
        return _dim(name, ConfidenceLevel.INSUFFICIENT,
                    "no structure selected for this candidate")
    if st.source in _EXPERIMENTAL_STRUCTURE_SOURCES:
        level = ConfidenceLevel.STRONG
        notes = [f"experimental coordinates ({st.source}, {st.structure_id})"]
        if st.is_mutant_relative_to_candidate:
            level = ConfidenceLevel.MODERATE
            notes.append("structure is a mutant relative to the candidate sequence")
        if st.missing_regions:
            level = min((level, ConfidenceLevel.MODERATE), key=lambda lv: lv.rank)
            notes.append(f"{len(st.missing_regions)} unmodelled region(s)")
        return _dim(name, level, "; ".join(notes), value=st.pocket_plddt,
                    unit="pLDDT" if st.pocket_plddt is not None else None)
    if st.pocket_plddt is None:
        extra = (f" A whole-chain mean of {st.mean_plddt:.1f} is present but is not "
                 f"used: the mean is dominated by the well-ordered core and hides a "
                 f"disordered active-site loop." if st.mean_plddt is not None else "")
        return _dim(name, ConfidenceLevel.INSUFFICIENT,
                    "pocket-local pLDDT not computed for the predicted structure." + extra)
    level = st.pocket_confidence
    if st.missing_regions:
        level = min((level, ConfidenceLevel.MODERATE), key=lambda lv: lv.rank)
    return _dim(name, level,
                f"pocket pLDDT {st.pocket_plddt:.1f} on {st.source} structure "
                f"{st.structure_id}; the pocket value is used rather than the "
                f"chain mean because the pocket is where the chemistry happens",
                value=st.pocket_plddt, unit="pLDDT")


def _axis_docking_result(candidate: Candidate, config: Mapping[str, Any]) -> ScoreDimension:
    """Docking score, or an explicit refusal to rank one whose polarity is unstated.

    Scoring functions disagree about sign: some report an estimated binding free
    energy where more negative is better, others a positive goodness score. The
    direction is therefore read from
    ``config["docking_score_direction"]``; when the function is not declared
    there the axis is reported ``INSUFFICIENT`` rather than assumed, because a
    sign error here inverts the entire ranking silently.
    """
    name = "docking_result"
    scored = [p for p in candidate.valid_poses if p.docking_score is not None]
    if not scored:
        return _dim(name, ConfidenceLevel.INSUFFICIENT,
                    "no valid pose carries a docking score", direction="categorical")
    functions = {p.docking_score_function for p in scored}
    if len(functions) > 1:
        return _dim(name, ConfidenceLevel.INSUFFICIENT,
                    f"poses carry scores from {len(functions)} different functions "
                    f"({', '.join(sorted(str(f) for f in functions))}); scores from "
                    f"different functions are not on one scale and are not pooled",
                    direction="categorical")
    function = next(iter(functions))
    polarity = (config.get("docking_score_direction") or {})
    direction = polarity.get(function) if isinstance(polarity, Mapping) else None
    if direction not in ("higher_is_better", "lower_is_better"):
        return _dim(name, ConfidenceLevel.INSUFFICIENT,
                    f"scoring function '{function}' has no declared polarity in "
                    f"config['docking_score_direction']; a score whose direction is "
                    f"unknown cannot be ranked without risking a silent inversion",
                    direction="categorical")
    best = (min if direction == "lower_is_better" else max)(
        scored, key=lambda p: (p.docking_score, p.pose_id)
    )
    return _dim(
        name, ConfidenceLevel.WEAK,
        f"best of {len(scored)} pose(s) under '{function}' ({direction}). A docking "
        f"score is a weak signal by construction and is comparable only within a "
        f"family and a scoring function -- see rank_within_family",
        value=best.docking_score, unit=str(function), direction=direction,
    )


def _axis_expression_developability_risk(candidate: Candidate,
                                         config: Mapping[str, Any]) -> ScoreDimension:
    name = "expression_developability_risk"
    external = _external_dimension(name, candidate.candidate_id, config)
    if external is not None:
        return external
    return _dim(
        name, ConfidenceLevel.INSUFFICIENT,
        "no solubility, aggregation or host-expression predictor is attached to "
        "this run. Sequence length and organism are not used as proxies: they "
        "would produce a confident ordering with no relationship to whether the "
        "protein expresses in the chosen host.",
        direction="lower_is_better",
    )


def _axis_model_uncertainty(candidate: Candidate, min_valid_poses: int) -> ScoreDimension:
    """How well-corroborated the structural model is, as an ordinal level.

    No scalar, on purpose. "Uncertainty" here is two unrelated things -- how
    thinly the pose ensemble was sampled, and whether independent modelling
    routes agree -- and a single float would imply they are one quantity with an
    exchange rate. ``CONTRADICTORY`` is reserved for genuine disagreement
    between routes, which is a finding rather than noise and must not be
    resolved by majority vote.
    """
    name = "model_uncertainty"
    reports_by_pose = {r.pose_id: r for r in candidate.geometry}
    by_method: dict[str, list[GeometryReport]] = {}
    for pose in candidate.valid_poses:
        report = reports_by_pose.get(pose.pose_id)
        if report is not None:
            by_method.setdefault(pose.method, []).append(report)
    agreement = cross_method_agreement(by_method) if by_method \
        else ConfidenceLevel.INSUFFICIENT
    n_valid = len(candidate.valid_poses)
    robustness = classify_robustness(candidate.robustness_G, n_valid,
                                     min_valid_poses=min_valid_poses)
    if agreement is ConfidenceLevel.CONTRADICTORY:
        level = ConfidenceLevel.CONTRADICTORY
    else:
        level = min((agreement, robustness), key=lambda lv: lv.rank)
    g_text = "unset" if candidate.robustness_G is None else f"{candidate.robustness_G:.2f}"
    return _dim(
        name, level,
        f"cross-method agreement over {len(by_method)} route(s) is "
        f"{agreement.value}; pose robustness G={g_text} over {n_valid} valid pose(s) "
        f"classifies as {robustness.value}. Low corroboration lowers evidence "
        f"strength only -- it is never read as evidence that the enzyme is inactive.",
        direction="categorical",
    )


def _axis_sequence_pocket_novelty(candidate: Candidate) -> ScoreDimension:
    """Distance from what was already known. An exploration axis, not a quality axis.

    ``level`` here answers "how well established is this number", not "how good
    is this candidate": a novelty of 60% measured from a recorded search is
    well established and says nothing at all about activity. Whether novelty is
    desirable depends entirely on which batch role the slot is being spent on.
    """
    name = "sequence_pocket_novelty"
    rec = candidate.sequence_record
    pocket_substitutions = len(candidate.catalytic_mapping.substituted_roles)
    if rec.percent_identity is None:
        return _dim(name, ConfidenceLevel.INSUFFICIENT,
                    "no percent identity to the seed recorded, so distance from the "
                    "known set cannot be stated; "
                    f"{pocket_substitutions} catalytic-role substitution(s) mapped")
    level = ConfidenceLevel.STRONG if rec.search_method else ConfidenceLevel.WEAK
    return _dim(
        name, level,
        f"{100.0 - rec.percent_identity:.1f}% divergent from seed "
        f"{rec.seed_accession or 'unrecorded'} via "
        f"{rec.search_method or 'an unrecorded search'}; "
        f"{pocket_substitutions} catalytic-role substitution(s) mapped. Novelty is "
        f"a reason to spend a diversity slot, not a reason to expect activity.",
        value=100.0 - rec.percent_identity, unit="% divergence from seed",
    )


def build_scorecard(
    candidate: Candidate,
    task: TaskSpec,
    templates: Any = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, ScoreDimension]:
    """Fill every axis from what is actually known, marking the rest INSUFFICIENT.

    Returns the nine axes of
    :data:`~eagent.schemas.candidate.SCORE_DIMENSIONS` **plus** the four
    feasibility-gate entries named in :data:`GATE_NAMES`, all keyed by name so
    the result can be assigned straight onto
    :attr:`~eagent.schemas.candidate.Candidate.scorecard`. There is deliberately
    no total: see :func:`refuse_linear_blend`.

    THE TWO ERROR KINDS, IN CODE
    ----------------------------
    *Input errors* -- a wrong sequence, a wrong ligand identity or oxidation
    state, an unresolved or self-contradictory stereochemical specification --
    are detected by :func:`input_defects` and routed to **gates**. A failed gate
    sets ``gate_passed=False``, which makes
    :attr:`~eagent.schemas.candidate.Candidate.passes_gates` false, which
    removes the candidate from every batch role. They are defects to be
    repaired, and no score on another axis compensates for one.

    *Evidence weakness* -- thin pose sampling, a low pocket pLDDT, two modelling
    routes disagreeing, no literature -- is detected by
    :func:`evidence_weaknesses` and routed to **levels**. It lowers
    ``ScoreDimension.level`` and never touches ``gate_passed``. The asymmetry is
    the point: a weakly-supported candidate stays eligible, because "we do not
    know" is not "it does not work", and the candidates we know least about are
    the ones a mining campaign exists to find.

    Axes whose evidence does not exist in this repository (a trained substrate
    specificity model, a developability predictor) stay ``INSUFFICIENT`` unless
    an external model supplies them with a named basis through
    ``config["external_dimensions"]``.

    Parameters
    ----------
    templates:
        The injected template library. ``None`` is allowed and leaves the
        template-dependent gates ``UNEVALUATED``; a library that is present but
        missing a template the candidate names raises
        :class:`~eagent.errors.TemplateError`, because that is a broken run, not
        a property of the enzyme.
    config:
        ``docking_score_direction`` (scoring function -> polarity),
        ``min_valid_poses`` (sampling adequacy, default
        :data:`~eagent.science.robustness.DEFAULT_MIN_VALID_POSES`) and
        ``external_dimensions``.
    """
    cfg: Mapping[str, Any] = config or {}
    min_valid_poses = int(cfg.get("min_valid_poses", DEFAULT_MIN_VALID_POSES))

    card: dict[str, ScoreDimension] = {}
    for gate in evaluate_feasibility_gates(candidate, task, templates):
        card[gate.name] = gate.to_dimension()

    card["functional_literature_evidence"] = _axis_functional_literature_evidence(candidate)
    card["family_mechanism_compatibility"] = _axis_family_mechanism_compatibility(candidate)
    card["substrate_specificity_model"] = _axis_substrate_specificity_model(candidate, cfg)
    card["catalytic_geometry"] = _axis_catalytic_geometry(candidate, min_valid_poses)
    card["local_structure_confidence"] = _axis_local_structure_confidence(candidate)
    card["docking_result"] = _axis_docking_result(candidate, cfg)
    card["expression_developability_risk"] = _axis_expression_developability_risk(candidate, cfg)
    card["model_uncertainty"] = _axis_model_uncertainty(candidate, min_valid_poses)
    card["sequence_pocket_novelty"] = _axis_sequence_pocket_novelty(candidate)

    missing = [n for n in SCORE_DIMENSIONS if n not in card]
    if missing:  # pragma: no cover - guards against an axis being added upstream
        raise FabricationGuardError(
            f"scorecard is missing axes {missing}; every declared axis must be "
            f"filled or explicitly marked INSUFFICIENT, never omitted"
        )
    return card


def apply_scorecard(
    candidate: Candidate,
    task: TaskSpec,
    templates: Any = None,
    config: Mapping[str, Any] | None = None,
) -> list[Uncertainty]:
    """Write the scorecard onto the candidate and hand back its open questions.

    Convenience for the interface layer. The uncertainties are returned rather
    than stored because :class:`~eagent.schemas.candidate.Candidate` has no
    field for them: they belong in the step's
    :class:`~eagent.envelope.ToolResult`, where the controller can act on them.
    """
    gates = evaluate_feasibility_gates(candidate, task, templates)
    candidate.scorecard = build_scorecard(candidate, task, templates, config)
    return gate_uncertainties(gates, subject=candidate.candidate_id)


# ==========================================================================
# Comparison primitives
# ==========================================================================

#: The scale a comparable number lives on. Two numbers on different scales are
#: not two values of one quantity, and the module refuses to order them.
MEASURED_SCALE: str = "measured"
ORDINAL_SCALE: str = "evidence_ordinal"
NO_SCALE: str = "none"


class Comparable(NamedTuple):
    """One dimension reduced to a number, with the scale that number is on.

    The scale is the field this used to be missing. A dimension with a value
    is compared on that value; a dimension without one falls back to its
    ordinal evidence rank, which runs 0-4. Those are different quantities, and
    a comparison that puts them on one axis says a candidate with no
    measurement and a STRONG level (4.0) beats one measured at
    ``robustness_G = 0.9`` -- the measured candidate is then dominated and
    drops off the Pareto front, which is the opposite of what the evidence
    supports.
    """

    value: float | None
    direction: str
    scale: str = NO_SCALE

    @property
    def usable(self) -> bool:
        return self.value is not None

    def commensurable_with(self, other: "Comparable") -> bool:
        """Whether these two numbers are two values of the same quantity."""
        return (self.usable and other.usable and self.scale == other.scale
                and self.direction == other.direction)


def comparable(dim: ScoreDimension | None) -> Comparable:
    """The comparable number of a dimension, its direction, and its scale.

    Returns a ``value`` of ``None`` when the dimension cannot be compared at
    all. Four rules, each avoiding a specific trap:

    * ``INSUFFICIENT`` and ``CONTRADICTORY`` yield ``None`` even when a
      ``value`` is present. An axis we could not establish must not sort as a
      mid-range number, and models that disagree must not be averaged into one.
    * A dimension with a ``value`` is compared on that value, in its own
      declared direction, on the ``measured`` scale.
    * A dimension with no ``value`` falls back to its ordinal level rank, and
      the direction of that fallback is **always** ``higher_is_better``,
      because an evidence level only increases with evidence. Returning the
      dimension's declared direction here would invert a ``lower_is_better``
      axis the moment its scalar went missing -- a sign flip that no test on
      the happy path would catch.
    * The fallback is marked ``evidence_ordinal``, and callers must not order
      it against a ``measured`` number. A pLDDT of 40 and an evidence rank of
      4 are not 40 and 4 of one thing.
    """
    if dim is None:
        return Comparable(None, "higher_is_better", NO_SCALE)
    if dim.level in (ConfidenceLevel.INSUFFICIENT, ConfidenceLevel.CONTRADICTORY):
        return Comparable(None, dim.direction, NO_SCALE)
    if dim.value is not None and dim.direction in ("higher_is_better", "lower_is_better"):
        return Comparable(dim.value, dim.direction, MEASURED_SCALE)
    return Comparable(float(dim.level.rank), "higher_is_better", ORDINAL_SCALE)


def objective_value(
    candidate: Candidate, dimension_name: str, direction: str
) -> float | None:
    """The candidate's number for one stated objective, or ``None`` if incomparable.

    Raises rather than guessing in the two cases where a caller has made a
    category error:

    * naming a gate as an objective -- a gate is an eligibility test, and
      trading a failed gate against a good docking score is exactly what gates
      exist to prevent;
    * stating a direction the dimension disagrees with -- an inverted objective
      produces a complete, plausible, exactly-backwards Pareto front.
    """
    return objective_comparable(candidate, dimension_name, direction).value


def objective_comparable(
    candidate: Candidate, dimension_name: str, direction: str
) -> Comparable:
    """:func:`objective_value` with the scale kept, for callers that compare.

    Anything that orders two candidates needs the scale: an evidence ordinal
    and a measured value are not two readings of one quantity, and ordering
    them lets a candidate nobody measured dominate one that was.
    """
    if direction not in ("higher_is_better", "lower_is_better"):
        raise ValueError(
            f"direction must be 'higher_is_better' or 'lower_is_better', "
            f"got {direction!r}"
        )
    dim = candidate.dimension(dimension_name)
    if dim is not None and dim.is_gate:
        raise ValueError(
            f"'{dimension_name}' is a feasibility gate, not an objective. Filter on "
            f"Candidate.passes_gates first; a gate is not something to trade away."
        )
    reduced = comparable(dim)
    if reduced.value is None:
        return reduced
    if reduced.direction != direction:
        raise ValueError(
            f"objective '{dimension_name}' was requested as {direction} but the "
            f"dimension declares {reduced.direction}; one of the two is a sign "
            f"error and guessing which would invert the ranking silently"
        )
    return reduced


def dominates(
    a: Candidate, b: Candidate, objectives: Sequence[tuple[str, str]]
) -> bool:
    """Does ``a`` Pareto-dominate ``b``: at least as good everywhere, better somewhere?

    A ``None`` on either side of any objective makes the pair **incomparable**
    and the answer ``False``. The alternative -- treating ``None`` as the worst
    possible value -- would let a candidate that was simply never measured be
    dominated and dropped from the front, converting a gap in the pipeline's
    coverage into a judgement about the enzyme. Incomparability keeps it on the
    front, where its missing axis is visible and can be filled.
    """
    if not objectives:
        raise ValueError("Pareto domination needs at least one objective")
    strictly_better = False
    for dimension_name, direction in objectives:
        ca = objective_comparable(a, dimension_name, direction)
        cb = objective_comparable(b, dimension_name, direction)
        # Not just "both present": both on the same scale. One candidate
        # reduced to a measured pLDDT of 40 and the other to an evidence rank
        # of 4 are not 40 and 4 of one quantity, and ordering them lets the
        # candidate nobody measured dominate the one that was -- which then
        # drops off the front with its measurement intact and unused.
        if not ca.commensurable_with(cb):
            return False
        va, vb = ca.value, cb.value
        assert va is not None and vb is not None
        if direction == "lower_is_better":
            va, vb = -va, -vb
        if va < vb:
            return False
        if va > vb:
            strictly_better = True
    return strictly_better


def pareto_front(
    candidates: Sequence[Candidate], objectives: Sequence[tuple[str, str]]
) -> list[Candidate]:
    """Candidates no other candidate dominates, in input order.

    Pareto non-domination is used because it is the only multi-objective
    ordering that requires no exchange rate between the objectives -- which is
    precisely the thing this pipeline cannot justify (see
    :func:`refuse_linear_blend`). The front is usually wider than a weighted
    sum's top-k, and that width is honest: it is the set of candidates that
    cannot be separated without asserting a tradeoff nobody has calibrated.

    Input order is preserved so the result is deterministic and so an upstream
    ordering (for example a lexicographic rank) survives into the front.
    """
    if not objectives:
        raise ValueError("pareto_front needs at least one (dimension, direction) pair")
    front: list[Candidate] = []
    for cand in candidates:
        if any(other is not cand and dominates(other, cand, objectives)
               for other in candidates):
            continue
        front.append(cand)
    return front


def _gate_tier(candidate: Candidate, require_gates: bool) -> int:
    """0 eligible, 1 undecided, 2 ineligible, with undecided ranked above failed.

    Three tiers rather than a boolean so an unevaluated gate never shares a
    bucket with a failed one; see :class:`GateOutcome`.
    """
    gates = candidate.gates()
    if not gates:
        if require_gates:
            raise ValueError(
                f"{candidate.candidate_id} has no feasibility gates on its "
                f"scorecard; run build_scorecard first. Ranking a candidate whose "
                f"eligibility was never tested silently treats 'unchecked' as 'fine'."
            )
        return 1
    if candidate.input_errors or candidate.disqualified:
        return 2
    if any(g.gate_passed is False for g in gates):
        return 2
    if any(g.gate_passed is None for g in gates):
        return 1
    return 0


def lexicographic_rank(
    candidates: Sequence[Candidate],
    order_of_dimensions: Sequence[str] | None = None,
    *,
    require_gates: bool = True,
) -> list[Candidate]:
    """Total order by priority of axes: the defensible default when no model exists.

    WHY LEXICOGRAPHIC AND NOT WEIGHTED
    ----------------------------------
    A weighted sum claims an exchange rate ("0.1 of a pLDDT point is worth 0.03
    kcal/mol of docking score") that nothing in this repository can support. A
    lexicographic order claims only a *priority*: that a sequence-level
    experimental record settles the question before a docking score is consulted
    at all. That claim is visible in
    :data:`DEFAULT_LEXICOGRAPHIC_ORDER`, arguable, and changeable in one place --
    which is the whole difference between a stated assumption and a hidden one.

    Candidates are grouped first by gate tier (eligible, then undecided, then
    ineligible), so a failed gate can never be out-ranked by a strong score.

    WHAT THE UNKNOWN-LAST CONVENTION DOES AND DOES NOT MEAN
    -------------------------------------------------------
    Within a dimension, a candidate with no comparable value is ordered after
    one with a value, and the comparison then continues to the next dimension.
    That is a *ranking convention needed to produce a list*, not a claim that
    the unknown candidate is worse. :func:`pareto_front` is the function that
    refuses to make that call; both are provided so the convention is a choice
    the caller makes rather than a default they never see.

    The final tie-break is ``candidate_id``, so the order is deterministic.
    """
    order = tuple(order_of_dimensions or DEFAULT_LEXICOGRAPHIC_ORDER)
    if not order:
        raise ValueError("lexicographic_rank needs at least one dimension")

    def key(cand: Candidate) -> tuple:
        parts: list[Any] = [_gate_tier(cand, require_gates)]
        for dimension_name in order:
            dim = cand.dimension(dimension_name)
            if dim is not None and dim.is_gate:
                raise ValueError(
                    f"'{dimension_name}' is a feasibility gate and cannot be a "
                    f"ranking dimension; gates are handled by the tier instead"
                )
            # Level first, scalar second. Every dimension has a level and
            # every level is on one ordinal scale, so this comparison is
            # always between like and like; the scalar then separates
            # candidates whose evidence is equally strong. Sorting on
            # comparable()'s number alone put a measured 0.9 against an
            # evidence rank of 4 whenever one candidate had a scalar and
            # another did not.
            reduced = comparable(dim)
            if reduced.value is None:
                parts.append((1, 0.0, 0.0))
                continue
            level_rank = float(dim.level.rank) if dim is not None else 0.0
            scalar = (reduced.value
                      if reduced.scale == MEASURED_SCALE else None)
            parts.append((
                0, -level_rank,
                0.0 if scalar is None
                else (-scalar if reduced.direction == "higher_is_better"
                      else scalar)))
        parts.append(cand.candidate_id)
        return tuple(parts)

    return sorted(candidates, key=key)


def rank_within_family(
    candidates: Sequence[Candidate],
    order_of_dimensions: Sequence[str] | None = None,
    *,
    require_gates: bool = True,
    unassigned_key: str = "unassigned",
) -> dict[str, list[Candidate]]:
    """Group by family, then rank inside each group. Never across groups.

    WHY THE GROUPING IS NOT OPTIONAL
    --------------------------------
    The axes that discriminate best within a family are the ones that travel
    worst between families. A docking score depends on the scoring function's
    implicit assumptions about pocket size, desolvation and the ligand's
    rotatable bonds; comparing an SDR pocket's score with an aldo-keto
    reductase's largely ranks pocket volume. Pocket pLDDT depends on how much of
    the fold is in the predictor's training set. Percent identity to the seed is
    measured against a seed chosen inside one family. A single global ranking
    over these would be dominated by family effects and would be read as a
    statement about individual enzymes.

    Returning a dict of per-family orders forces the caller to decide explicitly
    how many slots each family gets -- which is what
    :func:`~eagent.science.diversity.family_quota_select` is for.
    """
    groups: dict[str, list[Candidate]] = {}
    for cand in candidates:
        key = cand.family.family_name or unassigned_key
        groups.setdefault(key, []).append(cand)
    return {
        family: lexicographic_rank(members, order_of_dimensions,
                                   require_gates=require_gates)
        for family, members in sorted(groups.items())
    }


def refuse_linear_blend(weights: Mapping[str, float] | Sequence[float] | None = None) -> None:
    """Always raises. Present so the weighted-sum shortcut fails loudly, with a reason.

    Someone will eventually want ``0.4 * pLDDT + 0.3 * dock + 0.3 * distance``.
    It is quick, it produces a clean ordering, and it is indefensible. The three
    quantities are not commensurable:

    * **pLDDT** is a predicted per-residue confidence in ``[0, 100]``. It is a
      statement about the *structure predictor*, not about the enzyme, and it
      saturates: 95 and 99 differ far less than 65 and 69.
    * **A docking score** is an unbounded pseudo-energy whose units and sign
      convention belong to one scoring function. It is not calibrated against
      measured affinity, let alone against turnover, and it is not comparable
      between scoring functions or between pockets.
    * **A catalytic distance** in angstroms is meaningful only against a window
      from a sourced :class:`~eagent.schemas.templates.GeometryConstraint`. It is
      not monotonic: 2.6 A can be better than 3.4 A *and* better than 1.8 A,
      because too close is a clash. Multiplying a non-monotonic quantity by a
      positive weight asserts something false before any arithmetic happens.

    Adding them requires an exchange rate. No dataset in this repository
    provides one, so any weights chosen are the author's intuition rendered as
    three decimal places -- and once a number exists, reviewers argue about the
    weights instead of about the evidence.

    WHAT TO DO INSTEAD
    ------------------
    1. Make the hard requirements gates (:class:`FeasibilityGate`), not terms.
    2. Compare on ordinal levels, which need no exchange rate.
    3. Compare within a family (:func:`rank_within_family`).
    4. Use :func:`pareto_front` when several objectives genuinely conflict, and
       accept that the front is wide.
    5. Use :func:`lexicographic_rank` when one list is required; its priority
       order is a visible, arguable configuration.
    6. If a scalar really is needed, *earn* it: fit a model on held-out
       :class:`~eagent.schemas.record.ExperimentRecord` outcomes with a
       leakage-controlled split, report its calibration, and register it with
       provenance. Then it is a prediction with an error bar, not a blend.

    A related-looking but legitimate case:
    :func:`~eagent.science.diversity.greedy_submodular_select` does combine a
    utility term with a coverage term through ``lambda_weight``. That knob
    trades two *batch-design* quantities (how many slots go to the best-supported
    candidates versus how much of the pocket space the batch covers) and its
    effect is visible in the resulting batch. It never produces a number
    reported as a property of an enzyme, which is the line this function
    defends.
    """
    named: str
    if isinstance(weights, Mapping):
        named = ", ".join(f"{k}={v}" for k, v in weights.items())
    elif weights is None:
        named = "(none given)"
    else:
        named = ", ".join(str(w) for w in weights)
    raise FabricationGuardError(
        "refusing to combine scorecard dimensions into a weighted total "
        f"[{named}]. pLDDT (bounded predictor confidence), a docking score "
        "(unbounded, function-specific pseudo-energy) and a catalytic distance "
        "(angstroms, non-monotonic against a template window) have no common "
        "unit and no calibrated exchange rate in this repository, so any weights "
        "are an opinion with decimal places. Use feasibility gates, ordinal "
        "levels, rank_within_family, pareto_front or lexicographic_rank; or fit "
        "and validate a model on held-out experimental records and register it "
        "with provenance. See refuse_linear_blend's docstring."
    )


# ==========================================================================
# Per-candidate justification
# ==========================================================================

def _level_line(dim: ScoreDimension | None, label: str) -> str:
    if dim is None:
        return f"  - {label}: not on the scorecard"
    value = "" if dim.value is None else f" (value {dim.value:g}{' ' + dim.unit if dim.unit else ''})"
    return f"  - {label}: {dim.level.value}{value} -- {dim.basis}"


def explain(candidate: Candidate) -> str:
    """The per-candidate justification a reviewer reads instead of a score.

    Answers, in order: why was this sequence retrieved, which family is it and
    on what evidence, what supports the target reaction specifically, how are
    the substrate and cofactor positioned and *by what authority*, what is still
    uncertain, and why it deserves a slot.

    The authority question is why
    :meth:`~eagent.schemas.chem.LigandSource.claim` is used verbatim for every
    ligand: "the cofactor sits 3.1 A from the carbonyl carbon" means something
    entirely different when the cofactor was observed crystallographically than
    when it was transplanted from a homologue, and a justification that does not
    say which has not justified anything.
    """
    rec = candidate.sequence_record
    lines: list[str] = [f"Candidate {candidate.candidate_id}"]

    lines.append("Why it was retrieved:")
    retrieval = (
        f"  {rec.search_method or 'an unrecorded search method'} against "
        f"{rec.source_database or 'an unrecorded database'}"
        f"{'@' + rec.database_version if rec.database_version else ''}"
        f" from seed {rec.seed_accession or 'unrecorded'}"
    )
    if rec.percent_identity is not None:
        retrieval += f", {rec.percent_identity:.1f}% identity"
    if rec.query_coverage is not None:
        retrieval += f", {rec.query_coverage:.0f}% coverage"
    if rec.evalue is not None:
        retrieval += f", E={rec.evalue:.2g}"
    lines.append(retrieval)
    lines.append(f"  accession {rec.accession or 'none'}; organism "
                 f"{rec.organism or 'unrecorded'}; length {rec.length}")

    fam = candidate.family
    lines.append("Family and mechanism:")
    lines.append(f"  {fam.family_name or 'unassigned'}"
                 f"{'/' + fam.subfamily if fam.subfamily else ''}, call confidence "
                 f"{fam.confidence.value} from {fam.n_independent_signals} "
                 f"independent signal(s)"
                 + (f"; conflicting: {', '.join(fam.signals_conflicting)}"
                    if fam.signals_conflicting else ""))
    mapping = candidate.catalytic_mapping
    if mapping.role_to_residue:
        lines.append("  catalytic roles mapped: "
                     + ", ".join(f"{k}={v}" for k, v in sorted(mapping.role_to_residue.items())))
    else:
        lines.append("  no catalytic roles mapped")
    if mapping.missing_roles:
        lines.append(f"  missing roles: {', '.join(mapping.missing_roles)}")

    lines.append("Evidence for the target reaction:")
    if candidate.evidence:
        for ref in candidate.evidence:
            lines.append(f"  - {ref.source_type}:{ref.identifier} "
                         f"[{ref.strength.value}]"
                         + (f" {ref.locator}" if ref.locator else ""))
    else:
        lines.append("  - none attached to this sequence; the case rests on "
                     "structure and family inference alone")

    lines.append("Substrate and cofactor placement:")
    if candidate.poses:
        for pose in candidate.poses[:5]:
            bits = [f"pose {pose.pose_id} via {pose.method}"]
            bits.append("substrate " + pose.substrate_source.claim()
                        if pose.substrate_present else "no substrate in this pose")
            if pose.cofactor_present:
                bits.append(f"cofactor {pose.cofactor_state.value}, "
                            + pose.cofactor_source.claim())
            else:
                bits.append("no cofactor in this pose")
            if pose.restrained_constraints:
                bits.append("restrained during modelling: "
                            + ", ".join(pose.restrained_constraints)
                            + " (not independent evidence)")
            lines.append("  - " + "; ".join(bits))
    else:
        lines.append("  - no complex pose has been built")
    st = candidate.best_structure
    if st is not None:
        lines.append(f"  - structure {st.structure_id} ({st.source}); ligands "
                     f"{', '.join(st.bound_ligands) or 'none'}; "
                     f"{st.ligand_source.claim()}")
    lines.append(f"  - stereochemical call: {candidate.stereo.call}"
                 + (f" ({candidate.stereo.basis})" if candidate.stereo.basis else ""))

    lines.append("Scorecard (no total, by design):")
    for dimension_name in SCORE_DIMENSIONS:
        lines.append(_level_line(candidate.dimension(dimension_name), dimension_name))

    lines.append("Feasibility gates:")
    gates = candidate.gates()
    if not gates:
        lines.append("  - none evaluated; this candidate has not been gated")
    for gate_dim in gates:
        verdict = {True: "PASS", False: "FAIL", None: "UNEVALUATED"}[gate_dim.gate_passed]
        lines.append(f"  - {gate_dim.name}: {verdict} -- {gate_dim.basis}")

    lines.append("What is uncertain:")
    weaknesses = evidence_weaknesses(candidate)
    if weaknesses:
        for weakness in weaknesses:
            lines.append(f"  - {weakness}")
    else:
        lines.append("  - no outstanding evidence gap recorded")
    lines.append("  (none of the above is evidence of inactivity; each lowers the "
                 "strength of a claim, which is what an uncertainty probe exists "
                 "to resolve)")

    lines.append("Why it deserves a slot:")
    if candidate.input_errors or candidate.disqualified:
        lines.append("  - it does not. Input defects must be repaired first: "
                     + "; ".join(input_defects(candidate)))
    elif candidate.has_unresolved_gate:
        lines.append("  - not yet. A feasibility gate is undecided; deciding it is "
                     "cheaper than a construct, and until then the candidate is "
                     "neither eligible nor rejected.")
    elif candidate.passes_gates:
        lines.append("  - the catalytic machinery maps, the required cofactor in the "
                     "required oxidation state is supplied, the inputs carry no "
                     "detected defect and the stereochemical target is resolved. "
                     "Its place in the batch depends on which role the slot serves; "
                     "see the role tag on its BatchMember.")
    else:
        lines.append("  - it does not: a mechanism gate failed. See the gate list above.")
    return "\n".join(lines)
