"""The gate chain that must pass before any catalytic geometry is ranked.

Why this module exists
----------------------
A final candidate ranking is a list of numbers: hydride-transfer distances,
attack angles, clash counts. Those numbers are produced by measuring a model,
and a model can be measured perfectly while describing the wrong thing. Four
mapping errors account for most of it:

1. the structure is not the candidate's sequence (a homologue, a point mutant,
   a construct with a tag, or simply the wrong chain);
2. residue numbering was never mapped, so the template's "Tyr155" lands on
   whatever residue happens to be numbered 155 in this file;
3. the ligand was matched by position in the coordinate file rather than by
   atom name, so the "hydride donor carbon" is some other atom of the
   cofactor;
4. the cofactor is in the wrong oxidation state, so a hydride-transfer
   geometry is being measured against a molecule that cannot donate a hydride.

Each of those produces a *plausible number*. Nothing about a 3.1 angstrom
distance says which atoms it was measured between. The errors therefore do not
show up as noise; they show up as signal, and they reorder the final ranking.

**Steps 1 to 4 add no scientific insight by themselves.** Passing them tells
nobody anything about whether an enzyme works. Their entire value is negative:
they remove the mapping errors that would otherwise change the ranking while
looking like real evidence. That is why they run first, in a fixed order, and
why the fifth step -- the only one that measures chemistry -- is not permitted
to run until they have passed.

How the chain behaves
---------------------
* The pipeline is ordered and inspectable: :class:`PreconditionChainReport`
  holds one :class:`PreconditionResult` per step, so a candidate's failure
  point is visible rather than summarised into a boolean.
* Three outcomes per step, never two: ``passed``, ``failed`` and
  ``not_evaluated``. "We could not check" is not "we checked and it is fine",
  and it is not "we checked and it is broken" either.
* A candidate that fails an early step is **repairable**: every failure carries
  a machine-readable :class:`RepairAction` and the step to re-run from. It must
  not be pushed into a distance ranking, and it must not be silently dropped;
  :attr:`PreconditionChainReport.discard_allowed` is always ``False``.

This module measures nothing itself. It consumes sequences, numbering maps,
atom names, cofactor identities and geometry measurements that other modules
produce, and decides whether they are coherent enough for a ranking to mean
anything.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

try:  # harness error base; guarded so this module imports standalone
    from ..errors import EAgentError
except Exception:  # pragma: no cover - only when eagent.errors is unavailable
    class EAgentError(Exception):  # type: ignore[no-redef]
        """Fallback base error used when :mod:`eagent.errors` cannot be imported."""

try:  # canonical sequence hash, for reporting; comparison never depends on it
    from ..provenance import sequence_hash as _sequence_hash
except Exception:  # pragma: no cover
    _sequence_hash = None  # type: ignore[assignment]

from .identity import (
    CofactorIdentity,
    CofactorRequirementCheck,
    CofactorSpecies,
    RedoxState,
)


__all__ = [
    "PreconditionError",
    "PreconditionStep",
    "STEP_ORDER",
    "MAPPING_GATES_RATIONALE",
    "why_mapping_gates_matter",
    "GateOutcome",
    "RepairAction",
    "PreconditionResult",
    "ChainDisposition",
    "PreconditionChainReport",
    "AtomMappingCheck",
    "check_atom_mapping",
    "check_sequence_consistency",
    "check_residue_numbering",
    "check_ligand_atom_mapping",
    "check_cofactor_state",
    "check_catalytic_geometry",
    "gate_chain",
    "POSITIONAL_MAPPING_BASES",
]


class PreconditionError(EAgentError):
    """A gate could not be constructed at all, e.g. a malformed template.

    Distinct from a gate *failing*: a failure is information about a candidate,
    whereas this is a bug in the inputs that must not be reported as a
    candidate's fault.
    """


#: Why the first four gates exist, quoted verbatim in reports so a reader who
#: sees four "passed" rows does not mistake them for evidence of activity.
MAPPING_GATES_RATIONALE: str = (
    "Steps 1 to 4 (sequence consistency, residue numbering, ligand atom "
    "mapping, cofactor state) add no scientific insight by themselves. Passing "
    "them says nothing about whether the enzyme catalyses the reaction. Their "
    "entire value is removing mapping errors that would otherwise change the "
    "final ranking while looking like real signal: a distance measured to the "
    "wrong atom, a residue identified by the wrong number, or a hydride "
    "transfer modelled from an oxidised cofactor all produce plausible numbers "
    "that no downstream statistic can detect."
)


def why_mapping_gates_matter() -> str:
    """Return :data:`MAPPING_GATES_RATIONALE`.

    Exists so report writers and the CLI quote one text rather than each
    paraphrasing it, which is how a careful caveat degrades into a slogan.
    """
    return MAPPING_GATES_RATIONALE


class PreconditionStep(str, enum.Enum):
    """The ordered gates, with what each establishes and what it does not.

    The order is not arbitrary: each gate presupposes the previous one. A
    residue-numbering map built against the wrong sequence is a correct map of
    the wrong protein, and an atom mapping validated against the wrong ligand
    is worse than no mapping at all, because it looks checked.
    """

    def __new__(cls, value: str, order: int, title: str,
                adds_scientific_insight: bool, doc: str,
                failure_mode: str) -> "PreconditionStep":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.order = order                                       # type: ignore[attr-defined]
        obj.title = title                                       # type: ignore[attr-defined]
        obj.adds_scientific_insight = adds_scientific_insight   # type: ignore[attr-defined]
        obj.failure_mode = failure_mode                         # type: ignore[attr-defined]
        obj.__doc__ = doc
        return obj

    SEQUENCE_CONSISTENCY = (
        "sequence_consistency", 1, "actual sequence consistency", False,
        "Is the structure's sequence the candidate's sequence? Compared on the "
        "residues actually present, not on the accession the file claims.",
        "A homologue, a point mutant or a tagged construct is measured and "
        "reported as the candidate; every later number describes a different "
        "protein.",
    )
    RESIDUE_NUMBERING = (
        "residue_numbering", 2, "residue numbering mapping established", False,
        "Is there an explicit map between author numbering in the file, the "
        "reference (UniProt) numbering a template speaks in, and the "
        "candidate's own 0-based index?",
        "A template residue lands on whichever residue carries that number in "
        "this file, so the 'catalytic tyrosine' is a different residue and the "
        "mutation order names the wrong position.",
    )
    LIGAND_ATOM_MAPPING = (
        "ligand_atom_mapping", 3, "ligand identity and atom-level mapping", False,
        "Is the ligand the right chemical component, and are its atoms matched "
        "by component id and atom name rather than by their order in the file?",
        "Two files listing the same ligand's atoms in different orders are "
        "mapped positionally, so the hydride-donor carbon is silently some "
        "other atom and every distance to it is meaningless.",
    )
    COFACTOR_STATE = (
        "cofactor_state", 4, "cofactor chemical state verified", False,
        "Does the modelled cofactor's species and oxidation state match what "
        "the mechanism requires?",
        "A hydride-transfer geometry is measured against NAD(P)+, which cannot "
        "donate a hydride; the geometry is fine and the chemistry is "
        "impossible.",
    )
    CATALYTIC_GEOMETRY = (
        "catalytic_geometry", 5, "catalytic geometry", True,
        "Do the measured distances and angles fall inside the mechanism's "
        "gating windows? This is the only step that adds scientific content, "
        "and it is only meaningful once the four above have passed.",
        "A geometry verdict is produced on a model whose atoms, residues or "
        "cofactor were mis-assigned, and it reorders the final ranking.",
    )

    @property
    def doc(self) -> str:
        """The member docstring, exposed for reports and the CLI."""
        return self.__doc__ or ""

    def describe(self) -> str:
        """One-line rendering used in the chain report."""
        tag = "measures chemistry" if self.adds_scientific_insight \
            else "removes mapping error only"
        return f"{self.order}. {self.title} [{tag}]: {self.doc}"


#: The gates in execution order. Exported so a caller can show the whole chain
#: before running it, including the steps a candidate never reached.
STEP_ORDER: tuple[PreconditionStep, ...] = tuple(
    sorted(PreconditionStep, key=lambda s: s.order)
)


class GateOutcome(str, enum.Enum):
    """Three outcomes, kept apart on purpose.

    ``NOT_EVALUATED`` is the one that matters. Folding it into ``FAILED``
    discards candidates for missing inputs; folding it into ``PASSED`` lets an
    unchecked candidate into the ranking. Both have happened in real pipelines
    and both are invisible afterwards.
    """

    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"

    @property
    def clears_gate(self) -> bool:
        """Whether the chain may proceed to the next step."""
        return self is GateOutcome.PASSED


@dataclass(frozen=True)
class RepairAction:
    """A concrete, machine-readable instruction for fixing one gate failure.

    ``action`` is a stable token a controller can dispatch on;
    ``params`` carries what it needs. A free-text suggestion would make the
    repair plan unexecutable, and an unexecutable repair plan means the
    candidate quietly disappears instead of coming back.
    """

    action: str
    detail: str
    params: dict[str, Any] = field(default_factory=dict)
    requires_human: bool = False
    rerun_from: PreconditionStep | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form used in the repair plan."""
        return {
            "action": self.action,
            "detail": self.detail,
            "params": dict(self.params),
            "requires_human": self.requires_human,
            "rerun_from": self.rerun_from.value if self.rerun_from else None,
        }


@dataclass(frozen=True)
class PreconditionResult:
    """The outcome of one gate, with the evidence it was decided on.

    ``evidence`` holds the actual values compared -- hashes, counts, atom
    names, measured distances -- so a reviewer can re-derive the verdict
    without re-running the step. A gate that reports only pass or fail cannot
    be audited, and an unauditable gate is eventually trusted for the wrong
    reason.
    """

    step: PreconditionStep
    outcome: GateOutcome
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    repair: RepairAction | None = None
    warnings: tuple[str, ...] = ()
    not_reached: bool = False

    @property
    def passed(self) -> bool:
        """Whether this gate cleared."""
        return self.outcome is GateOutcome.PASSED

    @property
    def blocks_chain(self) -> bool:
        """Whether the chain must stop here."""
        return self.outcome is not GateOutcome.PASSED

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the manifest."""
        return {
            "step": self.step.value,
            "order": self.step.order,
            "outcome": self.outcome.value,
            "adds_scientific_insight": self.step.adds_scientific_insight,
            "summary": self.summary,
            "evidence": dict(self.evidence),
            "repair": self.repair.as_dict() if self.repair else None,
            "warnings": list(self.warnings),
            "not_reached": self.not_reached,
        }


class ChainDisposition(str, enum.Enum):
    """What may be done with a candidate once the chain has run.

    There is deliberately no ``DISCARD``. A candidate that fails a mapping gate
    has not been shown to be a bad enzyme; it has been shown to have a broken
    model, and dropping it silently removes a possibly good candidate on the
    basis of a file-handling error.
    """

    READY_FOR_RANKING = "ready_for_ranking"
    REPAIR_AND_RERUN = "repair_and_rerun"
    HOLD_FOR_OPERATOR = "hold_for_operator"

    @property
    def may_enter_ranking(self) -> bool:
        """Whether the candidate's geometry may be ranked against others."""
        return self is ChainDisposition.READY_FOR_RANKING


@dataclass
class PreconditionChainReport:
    """The ordered results of one candidate's gate chain.

    Holds every step, including the ones never reached, so the report shows
    where the candidate stopped rather than implying the later gates were
    checked and passed.
    """

    candidate_id: str
    results: list[PreconditionResult] = field(default_factory=list)
    structure_id: str | None = None
    pose_id: str | None = None
    template_id: str | None = None

    # -- queries -----------------------------------------------------------
    def result_for(self, step: PreconditionStep | str) -> PreconditionResult | None:
        """The result for one step, or ``None`` if the chain never recorded it."""
        s = PreconditionStep(step)
        for r in self.results:
            if r.step is s:
                return r
        return None

    @property
    def first_failure_result(self) -> PreconditionResult | None:
        """The first step that did not pass, or ``None`` when all passed."""
        for r in self.results:
            if r.blocks_chain and not r.not_reached:
                return r
        return None

    @property
    def first_failure(self) -> PreconditionStep | None:
        """The step the candidate stopped at, or ``None`` when all passed."""
        r = self.first_failure_result
        return r.step if r else None

    @property
    def all_passed(self) -> bool:
        """Whether every gate, including catalytic geometry, cleared."""
        return bool(self.results) and all(r.passed for r in self.results)

    @property
    def mapping_gates_passed(self) -> bool:
        """Whether steps 1 to 4 cleared, i.e. the model is coherent."""
        return all(r.passed for r in self.results
                   if not r.step.adds_scientific_insight)

    @property
    def disposition(self) -> ChainDisposition:
        """What the controller may do with this candidate next."""
        if self.all_passed:
            return ChainDisposition.READY_FOR_RANKING
        first = self.first_failure_result
        if first is not None and first.repair is not None \
                and first.repair.requires_human:
            return ChainDisposition.HOLD_FOR_OPERATOR
        return ChainDisposition.REPAIR_AND_RERUN

    @property
    def may_enter_ranking(self) -> bool:
        """Whether this candidate's geometry may join a distance ranking."""
        return self.disposition.may_enter_ranking

    @property
    def discard_allowed(self) -> bool:
        """Always ``False``.

        A gate failure is a statement about the model, not about the enzyme. The
        candidate goes back for repair and stays in the report either way, so
        that a shortlist can say "nine ranked, four held for repair" instead of
        quietly presenting nine.
        """
        return False

    def rerun_from(self) -> PreconditionStep | None:
        """The step to resume at after the repair, or ``None`` if complete."""
        first = self.first_failure_result
        if first is None:
            return None
        if first.repair is not None and first.repair.rerun_from is not None:
            return first.repair.rerun_from
        return first.step

    def repair_plan(self) -> list[dict[str, Any]]:
        """Machine-readable repair plan: one entry per non-passing gate."""
        plan: list[dict[str, Any]] = []
        for r in self.results:
            if r.passed or r.repair is None:
                continue
            entry = r.repair.as_dict()
            entry["step"] = r.step.value
            entry["order"] = r.step.order
            entry["outcome"] = r.outcome.value
            entry["blocking"] = not r.not_reached
            plan.append(entry)
        return plan

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the run manifest."""
        return {
            "candidate_id": self.candidate_id,
            "structure_id": self.structure_id,
            "pose_id": self.pose_id,
            "template_id": self.template_id,
            "all_passed": self.all_passed,
            "mapping_gates_passed": self.mapping_gates_passed,
            "first_failure": self.first_failure.value if self.first_failure else None,
            "disposition": self.disposition.value,
            "may_enter_ranking": self.may_enter_ranking,
            "discard_allowed": self.discard_allowed,
            "rerun_from": self.rerun_from().value if self.rerun_from() else None,
            "results": [r.as_dict() for r in self.results],
            "repair_plan": self.repair_plan(),
            "rationale": MAPPING_GATES_RATIONALE,
        }

    def report_lines(self) -> list[str]:
        """Human-readable rendering that always lists all five gates."""
        lines = [f"precondition chain for candidate {self.candidate_id} "
                 f"(structure={self.structure_id}, pose={self.pose_id}, "
                 f"template={self.template_id})"]
        for r in self.results:
            mark = {"passed": "PASS", "failed": "FAIL",
                    "not_evaluated": "----"}[r.outcome.value]
            suffix = " (not reached)" if r.not_reached else ""
            lines.append(f"  {mark} {r.step.order}. {r.step.title}{suffix}: "
                         f"{r.summary}")
            for w in r.warnings:
                lines.append(f"       warning: {w}")
            if r.repair is not None and not r.passed:
                lines.append(f"       repair: {r.repair.action} -- "
                             f"{r.repair.detail}")
        lines.append(f"  disposition: {self.disposition.value} "
                     f"(ranking allowed: {self.may_enter_ranking}; "
                     f"discard allowed: {self.discard_allowed})")
        return lines

    def describe(self) -> str:
        """Report lines joined for printing."""
        return "\n".join(self.report_lines())


# ---------------------------------------------------------------------------
# Step 1 -- actual sequence consistency
# ---------------------------------------------------------------------------

def check_sequence_consistency(
    candidate: Any, structure: Any, *,
    allow_declared_mutations: bool = False,
) -> PreconditionResult:
    """Compare the candidate's sequence with the structure's actual sequence.

    "Actual" means the residues present in the coordinates, not the accession
    the file cites. A structure selected by accession is routinely a homologue,
    a point mutant, a different isoform or a construct carrying a tag, and all
    four produce a model whose numbers describe a protein the project is not
    proposing.

    ``allow_declared_mutations`` lets a structure that is a *declared* point
    mutant pass, provided the number of differences does not exceed the number
    of mutations the record declares. The differences are still reported,
    because the mutated residue may be the catalytic one being measured.
    """
    cand_seq = _first_str(candidate, "sequence") or _nested_str(
        candidate, "sequence_record", "sequence")
    struct_seq = _first_str(structure, "observed_sequence", "sequence",
                            "modelled_sequence", "seqres")
    missing = [name for name, seq in (("candidate", cand_seq),
                                      ("structure", struct_seq)) if not seq]
    if missing:
        return PreconditionResult(
            step=PreconditionStep.SEQUENCE_CONSISTENCY,
            outcome=GateOutcome.NOT_EVALUATED,
            summary=(f"no sequence available for {', '.join(missing)}; the gate "
                     f"was not evaluated and the candidate is neither passed "
                     f"nor discarded"),
            evidence={"missing": missing},
            repair=RepairAction(
                action="supply_sequence",
                detail=("extract the residues actually present in the "
                        "coordinates and attach the candidate's sequence, then "
                        "re-run from step 1"),
                params={"missing": missing},
                rerun_from=PreconditionStep.SEQUENCE_CONSISTENCY))

    a, b = _norm_seq(cand_seq), _norm_seq(struct_seq)
    evidence: dict[str, Any] = {
        "candidate_length": len(a), "structure_length": len(b),
        "candidate_sha256": _hash_or_none(a), "structure_sha256": _hash_or_none(b),
    }
    warnings: list[str] = []

    construct = _first_str(candidate, "construct_sequence")
    if construct and _norm_seq(construct) != a:
        warnings.append(
            "the candidate's expressed construct differs from its catalytic "
            "sequence; activity was measured on the construct, while this gate "
            "compares the catalytic sequence")
        evidence["construct_sha256"] = _hash_or_none(_norm_seq(construct))

    if a == b:
        return PreconditionResult(
            step=PreconditionStep.SEQUENCE_CONSISTENCY,
            outcome=GateOutcome.PASSED,
            summary=(f"the structure's {len(b)} observed residues are the "
                     f"candidate's sequence"),
            evidence=evidence, warnings=tuple(warnings))

    declared = list(_get(structure, "mutations_in_structure") or [])
    evidence["declared_mutations"] = declared

    if len(a) == len(b):
        diffs = [(i, a[i], b[i]) for i in range(len(a)) if a[i] != b[i]]
        evidence["n_mismatches"] = len(diffs)
        evidence["mismatches"] = [
            {"index": i, "candidate": x, "structure": y}
            for i, x, y in diffs[:20]]
        if allow_declared_mutations and declared and len(diffs) <= len(declared):
            warnings.append(
                f"the structure is a declared mutant ({', '.join(map(str, declared))}); "
                f"every geometric number describes the mutant, and if a mutated "
                f"position is one of the catalytic residues the measurement is "
                f"about a different active site")
            return PreconditionResult(
                step=PreconditionStep.SEQUENCE_CONSISTENCY,
                outcome=GateOutcome.PASSED,
                summary=(f"{len(diffs)} declared mutation(s) separate the "
                         f"structure from the candidate; accepted because the "
                         f"caller allowed declared mutations"),
                evidence=evidence, warnings=tuple(warnings))
        return PreconditionResult(
            step=PreconditionStep.SEQUENCE_CONSISTENCY,
            outcome=GateOutcome.FAILED,
            summary=(f"same length but {len(diffs)} residue(s) differ between "
                     f"the candidate and the structure"),
            evidence=evidence, warnings=tuple(warnings),
            repair=RepairAction(
                action="reselect_or_declare_mutant_structure",
                detail=("select a structure matching the candidate sequence, or "
                        "record the differences as declared mutations and "
                        "re-run allowing them; do not rank this model until one "
                        "of the two is done"),
                params={"n_mismatches": len(diffs),
                        "first_mismatches": evidence["mismatches"][:5]},
                rerun_from=PreconditionStep.SEQUENCE_CONSISTENCY))

    relation = ""
    if a and b:
        if a in b:
            relation = ("the candidate sequence is a contiguous subsequence of "
                        "the structure's, consistent with a tag, a fusion "
                        "partner or extra chain content in the file")
        elif b in a:
            relation = ("the structure's sequence is a contiguous subsequence "
                        "of the candidate's, consistent with a truncated "
                        "construct or unresolved termini")
    evidence["length_relation"] = relation or "no containment relation"
    return PreconditionResult(
        step=PreconditionStep.SEQUENCE_CONSISTENCY,
        outcome=GateOutcome.FAILED,
        summary=(f"length mismatch: candidate {len(a)} residues, structure "
                 f"{len(b)} residues"
                 + (f"; {relation}" if relation else "")),
        evidence=evidence, warnings=tuple(warnings),
        repair=RepairAction(
            action="reconcile_construct_and_structure",
            detail=("record the construct actually present in the coordinates "
                    "(tags, truncations, fusion partners, chain selection) and "
                    "either trim it or select another structure"),
            params={"candidate_length": len(a), "structure_length": len(b)},
            rerun_from=PreconditionStep.SEQUENCE_CONSISTENCY))


# ---------------------------------------------------------------------------
# Step 2 -- residue numbering mapping
# ---------------------------------------------------------------------------

def check_residue_numbering(
    candidate: Any, structure: Any, *, residue_map: Any = None,
    template: Any = None,
) -> PreconditionResult:
    """Require an explicit map between author, reference and candidate numbering.

    Three numbering systems are in play: the author numbering written in the
    coordinate file, the reference (UniProt) numbering a family or catalytic
    template speaks in, and the candidate's own 0-based sequence index. They
    diverge routinely -- construct offsets, insertion codes, disordered
    regions, historical renumbering -- and a position compared across two of
    them without a map is a different position.

    The reference axis is demanded only when something actually speaks in
    reference numbering: an accession on the candidate or structure, or a
    template declaring reference positions. Demanding it unconditionally would
    block *de novo* candidates that have no accession at all, which are exactly
    the ones this project produces.
    """
    rmap = residue_map
    if rmap is None:
        rmap = _get(structure, "residue_map", "numbering_map") \
            or _get(candidate, "residue_map", "numbering_map")
    if rmap is None:
        return PreconditionResult(
            step=PreconditionStep.RESIDUE_NUMBERING,
            outcome=GateOutcome.NOT_EVALUATED,
            summary=("no residue-numbering map was supplied; author numbering, "
                     "reference numbering and candidate index cannot be "
                     "related"),
            evidence={},
            repair=RepairAction(
                action="build_residue_map",
                detail=("align the candidate sequence to the structure chain and "
                        "build the author <-> index map (and the reference axis "
                        "when an accession exists) before any residue is named"),
                params={"structure_id": _get(structure, "structure_id", "id")},
                rerun_from=PreconditionStep.RESIDUE_NUMBERING))

    index_to_author = _mapping_of(rmap, "index_to_author")
    author_to_index = _mapping_of(rmap, "author_to_index")
    index_to_reference = _mapping_of(rmap, "index_to_reference",
                                     "index_to_uniprot")
    evidence: dict[str, Any] = {
        "n_author_mapped": len(index_to_author),
        "n_reference_mapped": len(index_to_reference),
        "reference_label": _get(rmap, "reference_label"),
    }
    warnings: list[str] = []

    if not index_to_author or not author_to_index:
        return PreconditionResult(
            step=PreconditionStep.RESIDUE_NUMBERING,
            outcome=GateOutcome.FAILED,
            summary=("the numbering map carries no author <-> index entries; it "
                     "cannot resolve a single residue"),
            evidence=evidence,
            repair=RepairAction(
                action="rebuild_residue_map",
                detail=("the map is empty: re-run the alignment against the "
                        "correct chain of the correct structure"),
                params={}, rerun_from=PreconditionStep.RESIDUE_NUMBERING))

    mismatches = list(_get(rmap, "mismatches") or [])
    if mismatches:
        evidence["n_map_mismatches"] = len(mismatches)
        warnings.append(
            f"{len(mismatches)} mapped position(s) disagree between the "
            f"candidate sequence and the structure; the map is usable but the "
            f"structure is not the candidate at those positions")

    unmapped_structure = list(_get(rmap, "unmapped_structure_positions") or [])
    if unmapped_structure:
        evidence["n_unmapped_structure_positions"] = len(unmapped_structure)
        warnings.append(
            f"{len(unmapped_structure)} observed residue(s) have no place in "
            f"the candidate sequence (tag, fusion partner, or the wrong chain)")

    # Catalytic roles must resolve through the map, not past it.
    roles = _role_indices(candidate, template)
    unresolved_roles = {
        role: idx for role, idx in roles.items()
        if _author_for_index(index_to_author, idx) is None
    }
    evidence["catalytic_roles_checked"] = sorted(roles)
    if unresolved_roles:
        return PreconditionResult(
            step=PreconditionStep.RESIDUE_NUMBERING,
            outcome=GateOutcome.FAILED,
            summary=(f"catalytic role(s) {', '.join(sorted(unresolved_roles))} "
                     f"have no observed position in the structure, so they "
                     f"cannot be measured or mutated"),
            evidence={**evidence, "unresolved_roles": unresolved_roles},
            warnings=tuple(warnings),
            repair=RepairAction(
                action="resolve_catalytic_positions",
                detail=("the catalytic residues are unobserved in this "
                        "structure (disordered or absent); choose a structure "
                        "that resolves them, or mark the geometry as "
                        "unmeasurable rather than measuring a neighbour"),
                params={"roles": sorted(unresolved_roles)},
                rerun_from=PreconditionStep.RESIDUE_NUMBERING))

    needs_reference, why = _reference_axis_required(candidate, structure, template)
    if needs_reference and not index_to_reference:
        return PreconditionResult(
            step=PreconditionStep.RESIDUE_NUMBERING,
            outcome=GateOutcome.FAILED,
            summary=(f"the reference (UniProt) numbering axis is missing while "
                     f"{why}; template positions cannot be placed on this "
                     f"candidate"),
            evidence=evidence, warnings=tuple(warnings),
            repair=RepairAction(
                action="establish_reference_numbering",
                detail=("build the candidate <-> reference mapping (SIFTS for a "
                        "PDB entry, or an alignment to the cited accession's "
                        "sequence) so template positions resolve explicitly"),
                params={"reason": why},
                rerun_from=PreconditionStep.RESIDUE_NUMBERING))

    axes = ["author", "candidate_index"] + (["reference"] if index_to_reference
                                            else [])
    return PreconditionResult(
        step=PreconditionStep.RESIDUE_NUMBERING,
        outcome=GateOutcome.PASSED,
        summary=(f"numbering established across {', '.join(axes)} for "
                 f"{len(index_to_author)} residue(s)"),
        evidence={**evidence, "axes": axes}, warnings=tuple(warnings))


# ---------------------------------------------------------------------------
# Step 3 -- ligand identity and atom-level mapping
# ---------------------------------------------------------------------------

#: Mapping bases that pair atoms by their position in a file. Named so the gate
#: can refuse them explicitly instead of hoping nobody used one.
POSITIONAL_MAPPING_BASES: frozenset[str] = frozenset({
    "file_order", "positional", "index", "atom_index", "serial", "order",
})


@dataclass(frozen=True)
class AtomMappingCheck:
    """Whether a ligand's atoms were matched by name rather than by position.

    ``positional_would_mismatch`` is the evidence that makes the refusal
    concrete: it lists the pairs a position-based reading would have produced,
    so a reviewer sees "C4 would have been paired with N1" rather than an
    abstract warning about atom order.
    """

    component_id: str | None
    ok: bool
    reason: str
    matched: dict[str, int] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    duplicated: tuple[str, ...] = ()
    order_differs: bool = False
    positional_would_mismatch: tuple[tuple[int, str, str], ...] = ()
    declared_conflicts: tuple[str, ...] = ()
    basis_used: str = "atom_name"
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Serialisable form for the gate evidence."""
        return {
            "component_id": self.component_id,
            "ok": self.ok,
            "reason": self.reason,
            "matched": dict(self.matched),
            "missing": list(self.missing),
            "duplicated": list(self.duplicated),
            "order_differs": self.order_differs,
            "positional_would_mismatch": [
                {"index": i, "reference_atom": r, "pose_atom": p}
                for i, r, p in self.positional_would_mismatch],
            "declared_conflicts": list(self.declared_conflicts),
            "basis_used": self.basis_used,
            "warnings": list(self.warnings),
        }


def check_atom_mapping(
    reference_atoms: Sequence[str],
    pose_atoms: Sequence[str],
    *,
    declared_mapping: Mapping[str, Any] | None = None,
    mapping_basis: str | None = None,
    component_id: str | None = None,
) -> AtomMappingCheck:
    """Map a ligand's atoms by name and catch any position-based mapping.

    The failure this prevents is the classic silent one: two files list the
    same chemical component's atoms in different orders, something pairs them
    by index, and the "hydride donor C4" is now the nicotinamide N1. Every
    distance measured afterwards is a real number measured between the wrong
    atoms, and nothing downstream can detect it.

    The check therefore does three things: it maps by atom name, it reports
    whether the two orders differ at all (and exactly which pairs a positional
    read would have produced), and it refuses a mapping that was *declared* on
    file order or that pairs atoms with different names.
    """
    ref = tuple(str(a).strip() for a in reference_atoms)
    pose = tuple(str(a).strip() for a in pose_atoms)
    warnings: list[str] = []

    seen: dict[str, int] = {}
    duplicated: list[str] = []
    for i, name in enumerate(pose):
        if name in seen:
            duplicated.append(name)
        else:
            seen[name] = i

    order_differs = len(ref) != len(pose) or any(
        ref[i] != pose[i] for i in range(min(len(ref), len(pose))))
    would_mismatch = tuple(
        (i, ref[i], pose[i]) for i in range(min(len(ref), len(pose)))
        if ref[i] != pose[i])

    missing = tuple(name for name in ref if name not in seen)
    matched = {name: seen[name] for name in ref if name in seen}

    if duplicated:
        return AtomMappingCheck(
            component_id=component_id, ok=False,
            reason=(f"atom names are not unique in the pose "
                    f"({', '.join(sorted(set(duplicated)))}); mapping by name "
                    f"is ambiguous and mapping by position is unsafe"),
            matched=matched, missing=missing,
            duplicated=tuple(sorted(set(duplicated))),
            order_differs=order_differs,
            positional_would_mismatch=would_mismatch)

    if missing:
        return AtomMappingCheck(
            component_id=component_id, ok=False,
            reason=(f"the pose does not contain required atom name(s) "
                    f"{', '.join(missing)}; the functional atoms the mechanism "
                    f"names cannot be located"),
            matched=matched, missing=missing, order_differs=order_differs,
            positional_would_mismatch=would_mismatch)

    conflicts: list[str] = []
    if declared_mapping:
        for ref_name, target in declared_mapping.items():
            ref_name = str(ref_name).strip()
            if isinstance(target, int) and not isinstance(target, bool):
                actual = pose[target] if 0 <= target < len(pose) else None
                if actual != ref_name:
                    conflicts.append(
                        f"'{ref_name}' was mapped to pose index {target} "
                        f"({actual or 'out of range'}), which is a different "
                        f"atom")
            else:
                tname = str(target).strip()
                if tname != ref_name:
                    conflicts.append(
                        f"'{ref_name}' was mapped to pose atom '{tname}'; the "
                        f"names differ, so this pairs two different atoms of "
                        f"the component")
    if conflicts:
        return AtomMappingCheck(
            component_id=component_id, ok=False,
            reason=("the declared atom mapping pairs atoms that do not share a "
                    "name: " + "; ".join(conflicts)),
            matched=matched, order_differs=order_differs,
            positional_would_mismatch=would_mismatch,
            declared_conflicts=tuple(conflicts))

    basis = (str(mapping_basis).strip().lower() if mapping_basis else "")
    if basis in POSITIONAL_MAPPING_BASES:
        if order_differs:
            return AtomMappingCheck(
                component_id=component_id, ok=False,
                reason=(f"the mapping was declared on '{basis}' while the two "
                        f"atom orders differ; a positional read would pair "
                        + ", ".join(f"{r}->{p}" for _, r, p
                                    in would_mismatch[:5])
                        + " -- these are different atoms of the same component"),
                matched=matched, order_differs=True,
                positional_would_mismatch=would_mismatch,
                basis_used=basis)
        warnings.append(
            f"the mapping was declared on '{basis}'; the orders happen to agree "
            f"today, but a re-written file reorders atoms without warning, so "
            f"the mapping must be re-expressed by atom name")

    if order_differs:
        warnings.append(
            "the pose lists this component's atoms in a different order from "
            "the reference; mapping was done by atom name, and a positional "
            "read would have been wrong for "
            + ", ".join(f"{r}/{p}" for _, r, p in would_mismatch[:5]))

    return AtomMappingCheck(
        component_id=component_id, ok=True,
        reason=(f"all {len(ref)} required atom(s) matched by name"),
        matched=matched, order_differs=order_differs,
        positional_would_mismatch=would_mismatch,
        basis_used=basis or "atom_name", warnings=tuple(warnings))


def check_ligand_atom_mapping(pose: Any, template: Any) -> PreconditionResult:
    """Check every ligand the mechanism needs by component id and atom name.

    A ligand identified only by a prose label ("NADPH", "the substrate") cannot
    be atom-mapped at all, and a ligand whose component id differs from the
    template's is a different molecule however similar the label -- NAP and NDP
    differ by two hydrogens and an entire oxidation state.
    """
    expectations = _required_ligands(template)
    if not expectations:
        return PreconditionResult(
            step=PreconditionStep.LIGAND_ATOM_MAPPING,
            outcome=GateOutcome.NOT_EVALUATED,
            summary=("the template declares no ligand component ids or "
                     "functional atom names, so no atom mapping can be checked"),
            evidence={},
            repair=RepairAction(
                action="declare_required_ligands",
                detail=("amend the catalytic template to name each required "
                        "ligand by chemical component id together with the "
                        "functional atom names the geometry refers to"),
                params={"template_id": _get(template, "template_id", "id")},
                requires_human=True,
                rerun_from=PreconditionStep.LIGAND_ATOM_MAPPING))

    pose_ligands = list(_iter_ligands(pose))
    if not pose_ligands:
        return PreconditionResult(
            step=PreconditionStep.LIGAND_ATOM_MAPPING,
            outcome=GateOutcome.NOT_EVALUATED,
            summary="the pose carries no ligand records to map",
            evidence={"required": [e["role"] for e in expectations]},
            repair=RepairAction(
                action="attach_pose_ligands",
                detail=("export each ligand from the pose with its chemical "
                        "component id and its atom names, not just coordinates"),
                params={}, rerun_from=PreconditionStep.LIGAND_ATOM_MAPPING))

    checks: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    failures: list[str] = []
    unevaluated: list[str] = []

    for exp in expectations:
        role = exp["role"]
        want_id = (exp["component_id"] or "").upper() or None
        found = _match_ligand(pose_ligands, role, want_id)
        if found is None:
            failures.append(
                f"{role}: no ligand in the pose matches component id "
                f"{want_id or '(unspecified)'}")
            checks[role] = {"ok": False, "reason": "ligand absent from pose"}
            continue
        got_id = (_first_str(found, "component_id", "ccd_component_id",
                             "ligand_code", "comp_id") or "").upper() or None
        if not got_id:
            failures.append(
                f"{role}: the pose ligand has no chemical component id, so its "
                f"atoms cannot be identified; a prose label is not an identity")
            checks[role] = {"ok": False, "reason": "no component id on pose ligand"}
            continue
        if want_id and got_id != want_id:
            failures.append(
                f"{role}: component id mismatch, template requires {want_id} "
                f"and the pose carries {got_id}; these are different chemical "
                f"entities even when the prose label is the same")
            checks[role] = {"ok": False, "reason": f"{got_id} != {want_id}"}
            continue

        ref_atoms = exp["atom_names"]
        pose_atoms = [str(x) for x in (_get(found, "atom_names", "atoms") or [])]
        if not ref_atoms:
            unevaluated.append(
                f"{role}: the template names no functional atoms for this "
                f"component, so atom-level mapping cannot be verified")
            checks[role] = {"ok": None,
                            "reason": "no reference atom names in template"}
            continue
        if not pose_atoms:
            failures.append(
                f"{role}: the pose ligand lists no atom names; atoms can then "
                f"only be addressed by position, which is the error this gate "
                f"exists to stop")
            checks[role] = {"ok": False, "reason": "no atom names on pose ligand"}
            continue

        result = check_atom_mapping(
            ref_atoms, pose_atoms,
            declared_mapping=_mapping_of(found, "atom_mapping") or None,
            mapping_basis=_first_str(found, "mapping_basis", "atom_mapping_basis"),
            component_id=got_id)
        checks[role] = result.as_dict()
        warnings.extend(f"{role}: {w}" for w in result.warnings)
        if not result.ok:
            failures.append(f"{role}: {result.reason}")

    evidence = {"ligands": checks,
                "required_roles": [e["role"] for e in expectations]}
    if failures:
        return PreconditionResult(
            step=PreconditionStep.LIGAND_ATOM_MAPPING,
            outcome=GateOutcome.FAILED,
            summary="; ".join(failures),
            evidence=evidence, warnings=tuple(warnings),
            repair=RepairAction(
                action="remap_ligand_atoms_by_name",
                detail=("re-export the pose ligands with their chemical "
                        "component ids and atom names, and express every "
                        "mapping by atom name; never by the order atoms appear "
                        "in a file"),
                params={"failed_roles": [f.split(":", 1)[0] for f in failures]},
                rerun_from=PreconditionStep.LIGAND_ATOM_MAPPING))
    if unevaluated:
        return PreconditionResult(
            step=PreconditionStep.LIGAND_ATOM_MAPPING,
            outcome=GateOutcome.NOT_EVALUATED,
            summary="; ".join(unevaluated),
            evidence=evidence, warnings=tuple(warnings),
            repair=RepairAction(
                action="declare_functional_atom_names",
                detail=("name the functional atoms in the catalytic template "
                        "(for example the nicotinamide C4 of the cofactor) so "
                        "the mapping can be verified atom by atom"),
                params={}, requires_human=True,
                rerun_from=PreconditionStep.LIGAND_ATOM_MAPPING))

    return PreconditionResult(
        step=PreconditionStep.LIGAND_ATOM_MAPPING,
        outcome=GateOutcome.PASSED,
        summary=(f"{len(expectations)} ligand(s) identified by component id and "
                 f"mapped atom by atom by name"),
        evidence=evidence, warnings=tuple(warnings))


# ---------------------------------------------------------------------------
# Step 4 -- cofactor chemical state
# ---------------------------------------------------------------------------

def check_cofactor_state(pose: Any, template: Any, *,
                         allow_backbone_substitution: bool = False) \
        -> PreconditionResult:
    """Verify the modelled cofactor against the mechanism's requirement.

    Compares species *and* oxidation state. An oxidation-state mismatch is a
    mechanism error, not a preference: a hydride-transfer geometry measured
    with NAD(P)+ in the site is measuring a reaction that cannot occur in that
    direction. An unknown state is reported as ``not_evaluated``, because
    guessing it in either direction silently decides the mechanism.
    """
    required = _required_cofactor(template)
    observed = _observed_cofactor(pose)

    if required is None:
        return PreconditionResult(
            step=PreconditionStep.COFACTOR_STATE,
            outcome=GateOutcome.NOT_EVALUATED,
            summary=("the catalytic template states no cofactor requirement; "
                     "whether this mechanism needs one cannot be assumed"),
            evidence={"observed": observed.as_dict() if observed else None},
            repair=RepairAction(
                action="declare_cofactor_requirement",
                detail=("state the required cofactor species and oxidation "
                        "state on the catalytic template, or mark it "
                        "explicitly as not applicable"),
                params={"template_id": _get(template, "template_id", "id")},
                requires_human=True,
                rerun_from=PreconditionStep.COFACTOR_STATE))

    if required.state is RedoxState.NOT_APPLICABLE \
            and required.species is CofactorSpecies.UNKNOWN:
        return PreconditionResult(
            step=PreconditionStep.COFACTOR_STATE,
            outcome=GateOutcome.PASSED,
            summary="the mechanism declares no cofactor requirement",
            evidence={"required": required.as_dict()})

    if observed is None:
        return PreconditionResult(
            step=PreconditionStep.COFACTOR_STATE,
            outcome=GateOutcome.NOT_EVALUATED,
            summary=(f"the mechanism requires {required.describe()} but the "
                     f"pose records no cofactor; the gate cannot be evaluated"),
            evidence={"required": required.as_dict()},
            repair=RepairAction(
                action="model_required_cofactor",
                detail=("rebuild the complex including the required cofactor in "
                        "the required oxidation state, and record its chemical "
                        "component id"),
                params={"required": required.describe()},
                rerun_from=PreconditionStep.LIGAND_ATOM_MAPPING))

    check: CofactorRequirementCheck = observed.satisfies(
        required, allow_backbone_substitution=allow_backbone_substitution)
    evidence = {"required": required.as_dict(), "observed": observed.as_dict(),
                "check": check.as_dict()}
    warnings: list[str] = []

    if _requires_hydride_donor(template):
        answer = observed.hydride_donor_answer()
        evidence["hydride_donor"] = answer.as_dict()
        if not answer.answered:
            return PreconditionResult(
                step=PreconditionStep.COFACTOR_STATE,
                outcome=GateOutcome.NOT_EVALUATED,
                summary=(f"the mechanism is a hydride transfer and the "
                         f"cofactor's oxidation state is unknown: "
                         f"{answer.reason}"),
                evidence=evidence,
                repair=RepairAction(
                    action="determine_cofactor_oxidation_state",
                    detail=answer.question,
                    params={"ligand_code": observed.ligand_code},
                    rerun_from=PreconditionStep.COFACTOR_STATE))
        if answer.value is False:
            return PreconditionResult(
                step=PreconditionStep.COFACTOR_STATE,
                outcome=GateOutcome.FAILED,
                summary=(f"the mechanism needs a hydride donor and the model "
                         f"carries {observed.describe()}, which cannot donate "
                         f"one"),
                evidence=evidence,
                repair=RepairAction(
                    action="rebuild_with_reduced_cofactor",
                    detail=("re-model the complex with the reduced cofactor, or "
                            "select a source structure whose component id is "
                            "the reduced form"),
                    params={"observed": observed.describe(),
                            "required": required.describe()},
                    rerun_from=PreconditionStep.LIGAND_ATOM_MAPPING))

    if check.satisfied is True:
        return PreconditionResult(
            step=PreconditionStep.COFACTOR_STATE,
            outcome=GateOutcome.PASSED,
            summary=check.reason, evidence=evidence, warnings=tuple(warnings))
    if check.satisfied is False:
        return PreconditionResult(
            step=PreconditionStep.COFACTOR_STATE,
            outcome=GateOutcome.FAILED, summary=check.reason, evidence=evidence,
            repair=RepairAction(
                action="correct_cofactor",
                detail=check.repair_hint,
                params={"required": check.required, "observed": check.observed},
                rerun_from=PreconditionStep.LIGAND_ATOM_MAPPING))
    return PreconditionResult(
        step=PreconditionStep.COFACTOR_STATE,
        outcome=GateOutcome.NOT_EVALUATED, summary=check.reason,
        evidence=evidence,
        repair=RepairAction(
            action="resolve_cofactor_identity",
            detail=check.repair_hint or (observed.question or
                                         "resolve the cofactor species and "
                                         "oxidation state"),
            params={"ligand_code": observed.ligand_code},
            rerun_from=PreconditionStep.COFACTOR_STATE))


# ---------------------------------------------------------------------------
# Step 5 -- catalytic geometry
# ---------------------------------------------------------------------------

def check_catalytic_geometry(pose: Any, template: Any, *,
                             measurements: Mapping[str, Any] | None = None) \
        -> PreconditionResult:
    """Evaluate the mechanism's gating geometry windows.

    This step adds the only scientific content in the chain, and it consumes
    measurements rather than computing them, so the numbers are produced by one
    module and judged by another.

    Two refusals are built in. A constraint with no measurement is
    ``not_evaluated``, never "not satisfied". And a constraint that was
    *restrained* while the pose was being built is marked circular and excluded
    from the independent count, because a distance that was held at 3.0
    angstroms by the modelling protocol is not evidence that the enzyme holds
    it there.
    """
    constraints = _gating_constraints(template)
    if not constraints:
        return PreconditionResult(
            step=PreconditionStep.CATALYTIC_GEOMETRY,
            outcome=GateOutcome.NOT_EVALUATED,
            summary=("the catalytic template declares no gating geometry "
                     "constraint, so there is nothing to gate on"),
            evidence={},
            repair=RepairAction(
                action="declare_gating_constraints",
                detail=("add at least one gating geometry constraint with a "
                        "window and its calibration source to the catalytic "
                        "template"),
                params={}, requires_human=True,
                rerun_from=PreconditionStep.CATALYTIC_GEOMETRY))

    values = dict(measurements or _mapping_of(pose, "measurements",
                                              "geometry_measurements"))
    restrained = {str(x) for x in (_get(pose, "restrained_constraints") or [])}

    per_constraint: dict[str, Any] = {}
    unresolved: list[str] = []
    violated: list[str] = []
    independent_satisfied = 0
    independent_total = 0
    warnings: list[str] = []

    for c in constraints:
        name = _constraint_name(c)
        value = values.get(name)
        satisfied = _constraint_satisfied(c, value)
        circular = name in restrained
        per_constraint[name] = {
            "value": value, "satisfied": satisfied, "circular": circular,
            "window": _constraint_window(c),
            "calibrated": bool(_get(c, "calibrated_on")),
        }
        if not _get(c, "calibrated_on"):
            warnings.append(
                f"constraint '{name}' has no recorded calibration, so passing "
                f"it is weaker evidence than it looks")
        if circular:
            warnings.append(
                f"constraint '{name}' was restrained while the pose was built; "
                f"it is excluded from the independent evidence count")
        else:
            independent_total += 1
        if satisfied is None:
            unresolved.append(name)
        elif satisfied:
            if not circular:
                independent_satisfied += 1
        else:
            violated.append(name)

    evidence = {
        "constraints": per_constraint,
        "independent_satisfied": independent_satisfied,
        "independent_total": independent_total,
        "circular_constraints": sorted(restrained & set(per_constraint)),
    }

    if violated:
        return PreconditionResult(
            step=PreconditionStep.CATALYTIC_GEOMETRY,
            outcome=GateOutcome.FAILED,
            summary=(f"gating constraint(s) {', '.join(violated)} fall outside "
                     f"the mechanism window"),
            evidence=evidence, warnings=tuple(warnings),
            repair=RepairAction(
                action="resample_or_reject_pose",
                detail=("the arrangement does not meet the mechanism's gating "
                        "geometry; sample further poses or record this "
                        "candidate as geometrically infeasible with the "
                        "measured values"),
                params={"violated": violated},
                rerun_from=PreconditionStep.CATALYTIC_GEOMETRY))
    if unresolved:
        return PreconditionResult(
            step=PreconditionStep.CATALYTIC_GEOMETRY,
            outcome=GateOutcome.NOT_EVALUATED,
            summary=(f"gating constraint(s) {', '.join(unresolved)} were not "
                     f"measured; an unmeasured constraint is not a satisfied "
                     f"one"),
            evidence=evidence, warnings=tuple(warnings),
            repair=RepairAction(
                action="measure_missing_geometry",
                detail=("measure the named constraints on this pose; if the "
                        "atoms they refer to are absent, the pose cannot be "
                        "gated and must be reported as unmeasurable"),
                params={"unresolved": unresolved},
                rerun_from=PreconditionStep.CATALYTIC_GEOMETRY))

    return PreconditionResult(
        step=PreconditionStep.CATALYTIC_GEOMETRY,
        outcome=GateOutcome.PASSED,
        summary=(f"all {len(constraints)} gating constraint(s) satisfied; "
                 f"{independent_satisfied}/{independent_total} of them "
                 f"independent of the modelling restraints"),
        evidence=evidence, warnings=tuple(warnings))


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------

def gate_chain(candidate: Any, structure: Any, pose: Any, template: Any, *,
               residue_map: Any = None,
               measurements: Mapping[str, Any] | None = None,
               allow_declared_mutations: bool = False,
               allow_backbone_substitution: bool = False) \
        -> PreconditionChainReport:
    """Run the five gates in order and stop at the first that does not pass.

    Returns a :class:`PreconditionChainReport` holding one result per step.
    Steps after the first failure are recorded as ``not_evaluated`` with
    ``not_reached=True``, so the report never implies that a gate the chain
    skipped was checked.

    Stopping early is not an optimisation. Running the later gates on a model
    that failed an earlier one produces verdicts about the wrong protein, the
    wrong residue or the wrong atom, and those verdicts are indistinguishable
    from real ones once they are in a table.

    The candidate is neither ranked nor discarded on a failure: the report
    carries a machine-readable repair plan and the step to resume from.
    """
    report = PreconditionChainReport(
        candidate_id=str(_get(candidate, "candidate_id", "id") or "unknown"),
        structure_id=_first_str(structure, "structure_id", "id"),
        pose_id=_first_str(pose, "pose_id", "id"),
        template_id=_first_str(template, "template_id", "id"),
    )

    runners = {
        PreconditionStep.SEQUENCE_CONSISTENCY: lambda: check_sequence_consistency(
            candidate, structure,
            allow_declared_mutations=allow_declared_mutations),
        PreconditionStep.RESIDUE_NUMBERING: lambda: check_residue_numbering(
            candidate, structure, residue_map=residue_map, template=template),
        PreconditionStep.LIGAND_ATOM_MAPPING: lambda: check_ligand_atom_mapping(
            pose, template),
        PreconditionStep.COFACTOR_STATE: lambda: check_cofactor_state(
            pose, template,
            allow_backbone_substitution=allow_backbone_substitution),
        PreconditionStep.CATALYTIC_GEOMETRY: lambda: check_catalytic_geometry(
            pose, template, measurements=measurements),
    }

    stopped_at: PreconditionStep | None = None
    for step in STEP_ORDER:
        if stopped_at is not None:
            report.results.append(PreconditionResult(
                step=step, outcome=GateOutcome.NOT_EVALUATED,
                summary=(f"not reached: the chain stopped at step "
                         f"{stopped_at.order} ({stopped_at.title})"),
                not_reached=True))
            continue
        result = runners[step]()
        report.results.append(result)
        if result.blocks_chain:
            stopped_at = step
    return report


# ---------------------------------------------------------------------------
# Input readers -- deliberately tolerant, because the producers of these
# records are written by other modules and may be mappings or models.
# ---------------------------------------------------------------------------

def _get(record: Any, *names: str) -> Any:
    """First present, non-empty field read from a mapping or an object."""
    for name in names:
        if isinstance(record, Mapping):
            if name in record and record[name] not in (None, ""):
                return record[name]
            continue
        value = getattr(record, name, None)
        if value not in (None, ""):
            return value
    return None


def _first_str(record: Any, *names: str) -> str | None:
    """First present field coerced to a stripped string, or ``None``."""
    v = _get(record, *names)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _nested_str(record: Any, outer: str, inner: str) -> str | None:
    """Read ``record.outer.inner`` defensively, returning ``None`` on any gap."""
    mid = _get(record, outer)
    if mid is None:
        return None
    return _first_str(mid, inner)


def _mapping_of(record: Any, *names: str) -> dict[Any, Any]:
    """A field read as a dict, or an empty dict when absent or not mapping-like."""
    v = _get(record, *names)
    if isinstance(v, Mapping):
        return dict(v)
    if v is None:
        return {}
    try:
        return dict(v)  # e.g. a list of pairs
    except Exception:
        return {}


def _norm_seq(seq: str) -> str:
    """Whitespace- and case-normalised sequence used for in-process comparison."""
    return "".join(str(seq).split()).upper()


def _hash_or_none(seq: str) -> str | None:
    """Canonical sequence hash for the report, or ``None`` when unavailable.

    Reporting ``None`` is deliberate: the gate's verdict never depends on the
    hash, so a missing :mod:`eagent.provenance` weakens the report rather than
    inviting a locally normalised substitute.
    """
    if _sequence_hash is None or not seq:
        return None
    try:
        return _sequence_hash(seq)
    except Exception:  # pragma: no cover - empty or invalid sequence
        return None


def _role_indices(candidate: Any, template: Any) -> dict[str, int]:
    """Catalytic role -> candidate index, from the candidate's mapping."""
    mapping = _get(candidate, "catalytic_mapping")
    roles = _mapping_of(mapping, "role_to_index") if mapping is not None else {}
    if not roles:
        roles = _mapping_of(candidate, "role_to_index")
    out: dict[str, int] = {}
    for role, idx in roles.items():
        try:
            out[str(role)] = int(idx)
        except (TypeError, ValueError):
            continue
    return out


def _author_for_index(index_to_author: Mapping[Any, Any], index: int) -> Any:
    """Author position for a candidate index, tolerating str/int keys."""
    if index in index_to_author:
        return index_to_author[index]
    return index_to_author.get(str(index))


def _reference_axis_required(candidate: Any, structure: Any,
                             template: Any) -> tuple[bool, str]:
    """Whether the reference-numbering axis is needed here, and why."""
    if _get(template, "reference_numbering_scheme") or \
            _get(template, "uses_reference_numbering"):
        return True, "the template declares positions in reference numbering"
    acc = _first_str(candidate, "accession") or _first_str(structure, "accession")
    if acc:
        return True, f"the records cite accession {acc}"
    return False, ""


def _required_ligands(template: Any) -> list[dict[str, Any]]:
    """Normalise the template's ligand expectations into role/id/atom-name rows."""
    rows: list[dict[str, Any]] = []
    declared = _get(template, "required_ligands")
    if declared:
        for entry in declared:
            rows.append({
                "role": _first_str(entry, "role", "name", "label") or "ligand",
                "component_id": _first_str(entry, "component_id",
                                           "ccd_component_id", "ligand_code"),
                "atom_names": [str(a) for a in
                               (_get(entry, "atom_names", "functional_atoms")
                                or [])],
            })
        return rows
    codes = _get(template, "cofactor_ligand_codes") or []
    atoms = [str(a) for a in (_get(template, "cofactor_functional_atoms") or [])]
    for code in codes:
        rows.append({"role": "cofactor", "component_id": str(code),
                     "atom_names": list(atoms)})
    return rows


def _iter_ligands(pose: Any) -> Iterable[Any]:
    """Ligand records attached to a pose, in whatever container they arrive."""
    ligs = _get(pose, "ligands", "ligand_records")
    if not ligs:
        return []
    if isinstance(ligs, Mapping):
        return list(ligs.values())
    return list(ligs)


def _match_ligand(ligands: Sequence[Any], role: str,
                  component_id: str | None) -> Any:
    """Find the pose ligand for a role, by role label first, then component id."""
    for lig in ligands:
        if (_first_str(lig, "role") or "").lower() == role.lower():
            return lig
    if component_id:
        for lig in ligands:
            got = _first_str(lig, "component_id", "ccd_component_id",
                             "ligand_code", "comp_id")
            if got and got.upper() == component_id.upper():
                return lig
    return None


def _required_cofactor(template: Any) -> CofactorIdentity | None:
    """The mechanism's cofactor requirement as a :class:`CofactorIdentity`."""
    req = _get(template, "required_cofactor_identity")
    if isinstance(req, CofactorIdentity):
        return req
    name = _first_str(template, "required_cofactor")
    state = _get(template, "required_cofactor_state")
    codes = _get(template, "cofactor_ligand_codes") or []
    if name is None and state is None and not codes:
        return None
    ident: CofactorIdentity | None = None
    if name is not None:
        ident = CofactorIdentity.from_label(
            name, determined_from="catalytic template")
    if (ident is None or not ident.is_known) and codes:
        ident = CofactorIdentity.from_ligand_code(
            str(codes[0]), determined_from="catalytic template ligand code")
    if ident is None:
        ident = CofactorIdentity(determined_from="catalytic template")
    declared = _coerce_state(state)
    if declared is not None and declared is not RedoxState.UNKNOWN:
        ident = replace(ident, state=declared)
    return ident


def _observed_cofactor(pose: Any) -> CofactorIdentity | None:
    """The cofactor actually modelled in the pose, or ``None`` if absent."""
    direct = _get(pose, "cofactor_identity")
    if isinstance(direct, CofactorIdentity):
        return direct
    cof = _get(pose, "cofactor")
    if isinstance(cof, CofactorIdentity):
        return cof
    code = None
    label = None
    state = None
    if cof is not None:
        code = _first_str(cof, "ligand_code", "component_id", "ccd_component_id")
        label = _first_str(cof, "species", "name", "label")
        state = _get(cof, "state")
    code = code or _first_str(pose, "cofactor_ligand_code", "cofactor_component_id")
    label = label or _first_str(pose, "cofactor_name")
    state = state if state is not None else _get(pose, "cofactor_state")
    if not code and not label and state is None:
        return None
    ident = (CofactorIdentity.from_ligand_code(code) if code
             else CofactorIdentity.from_label(label))
    declared = _coerce_state(state)
    if declared is not None and declared is not RedoxState.UNKNOWN \
            and not ident.state.is_known:
        ident = replace(ident, state=declared)
    return ident


def _coerce_state(value: Any) -> RedoxState | None:
    """Read an oxidation state written as a RedoxState, a string, or an enum."""
    if value is None:
        return None
    if isinstance(value, RedoxState):
        return value
    raw = getattr(value, "value", value)
    try:
        return RedoxState(str(raw).strip().lower())
    except ValueError:
        return None


def _requires_hydride_donor(template: Any) -> bool:
    """Whether the template declares a hydride-transfer mechanism."""
    flag = _get(template, "requires_hydride_donor")
    if flag is not None:
        return bool(flag)
    summary = (_first_str(template, "mechanism_summary") or "").lower()
    return "hydride" in summary


def _gating_constraints(template: Any) -> list[Any]:
    """Gating geometry constraints declared by the template."""
    getter = getattr(template, "gating_constraints", None)
    if callable(getter):
        try:
            return list(getter())
        except Exception:  # pragma: no cover - defensive
            pass
    out: list[Any] = []
    for c in (_get(template, "geometry_constraints") or []):
        sev = (_first_str(c, "severity") or "scoring").lower()
        if sev == "gating":
            out.append(c)
    return out


def _constraint_name(c: Any) -> str:
    """A constraint's name, which is also the key its measurement arrives under."""
    name = _first_str(c, "name")
    if not name:
        raise PreconditionError(
            "a geometry constraint has no name; measurements are keyed by name, "
            "so an unnamed constraint cannot be evaluated")
    return name


def _constraint_window(c: Any) -> tuple[float | None, float | None]:
    """The constraint's (low, high) window, read from either declaration style."""
    window = getattr(c, "window", None)
    if callable(window):
        try:
            lo, hi = window()
            return float(lo), float(hi)
        except Exception:  # pragma: no cover - defensive
            pass
    lo = _get(c, "min_value")
    hi = _get(c, "max_value")
    if lo is None and hi is None:
        target = _get(c, "target")
        tol = _get(c, "tolerance") or 0.0
        if target is None:
            return None, None
        return float(target) - float(tol), float(target) + float(tol)
    return (float(lo) if lo is not None else None,
            float(hi) if hi is not None else None)


def _constraint_satisfied(c: Any, value: Any) -> bool | None:
    """Tri-state satisfaction: inside the window, outside it, or unmeasured."""
    if value is None:
        return None
    checker = getattr(c, "satisfied_by", None)
    if callable(checker):
        try:
            return checker(value)
        except Exception:  # pragma: no cover - defensive
            pass
    lo, hi = _constraint_window(c)
    if lo is None and hi is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if lo is not None and v < lo:
        return False
    if hi is not None and v > hi:
        return False
    return True
