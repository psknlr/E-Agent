"""Comparators for the agent, on one pool, one budget and one hit definition.

WHY A BASELINE MODULE AND NOT A PARAGRAPH IN THE PAPER
======================================================
"The agent found four hits; homology search would not have" is the claim the
whole project rests on, and it is the easiest claim in computational biology
to make badly. Three things have to be identical across the comparators or the
comparison measures those things instead of the method:

* **The candidate pool.** A baseline run over a different set of sequences is
  not a baseline. :class:`CandidatePool` carries a digest over its members and
  :func:`compare_baselines` refuses any selection that names a candidate the
  pool does not contain, or that was produced against a different pool digest.
* **The budget.** Picking 96 when the comparator picked 20 compares budgets.
* **The hit definition.** Scoring one comparator against 20% conversion and
  another against 10% compares criteria. Every selection records the
  :class:`~eagent.eval.metrics.PreRegistration` digest it was made under, and
  the comparison refuses a mixture.

Every comparator therefore has the *same signature* --
``(pool, budget, registration) -> RankedSelection`` -- and anything a
particular comparator needs beyond that (seed accessions, a trained model, a
docking sign convention) is bound by a factory before the comparison starts,
where it is visible, rather than passed in at call time where it could differ
between comparators.

THE COMPARATORS, AND WHAT EACH ONE IS ENTITLED TO ANSWER
========================================================
``homology_multi_seed``
    What a competent human does first: search from several seeds and spread
    the picks so the batch is not one clade. The honest baseline to beat.
``family_function_prediction``
    A family or EC-class predictor. It is here as a *seam*, and its
    ``question_answered`` says what it is: "which family does this sequence
    belong to". That is **a different question** from "does this enzyme turn
    over this substrate under these conditions". A family call cannot
    distinguish two members of one family with opposite substrate ranges,
    which is the entire problem a substrate-directed campaign has. Reporting
    its number beside the agent's without that sentence attached would compare
    two methods on a task only one of them was built for.
``docking_score_ranking``
    Rank by the docking number. Honest only within one scoring function and
    one pocket class, which this comparator enforces rather than assumes.
``substrate_specificity_model``
    A trained substrate-specificity predictor: the right comparator, and a
    seam, because no such model ships here.
``full_agent``
    The pipeline's own selection -- gates, lexicographic rank, family and
    clade quotas, diversity -- through
    :func:`eagent.science.diversity.compose_batch`, so the thing being
    evaluated is the code that would actually run.

A SEAM RETURNS "UNAVAILABLE", IT DOES NOT IMPROVISE
===================================================
Where no model is installed, the comparator returns a
:class:`RankedSelection` with ``unavailable_reason`` set and no picks. It does
not fall back to a heuristic that imitates the model: a stand-in baseline that
the agent then beats is the most flattering possible result and means nothing.
:meth:`BaselineComparison.render` lists the unavailable comparators by name so
the gap in the comparison is on the page.

NOTHING HERE PRODUCES A TOTAL SCORE
===================================
Comparators are ordered by the single pre-registered primary endpoint, which
is one measured quantity, not a weighted blend of several
(:func:`eagent.science.scorecard.refuse_linear_blend` explains why the blend
is refused). :meth:`BaselineComparison.indistinguishable_pairs` lists the
pairs whose Wilson intervals overlap, because at 96 wells most pairs do, and
an ordering without that list reads as a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from ..errors import EAgentError
from ..provenance import sha256_obj
from ..schemas import Budget, Candidate
from ..science.diversity import (
    DEFAULT_LAMBDA_WEIGHT,
    compose_batch,
    greedy_submodular_select,
    sequence_distance,
)
from .metrics import OutcomeRow, PrecisionAtK, PreRegistration, precision_at_k

__all__ = [
    "PoolMismatchError",
    "CandidatePool",
    "comparator_name",
    "RankedSelection",
    "Comparator",
    "ComparatorScore",
    "BaselineComparison",
    "homology_multi_seed",
    "docking_score_ranking",
    "family_function_prediction",
    "substrate_specificity_model",
    "full_agent",
    "make_docking_comparator",
    "make_family_function_comparator",
    "make_specificity_model_comparator",
    "DEFAULT_COMPARATORS",
    "compare_baselines",
    "score_selection",
]


class PoolMismatchError(EAgentError):
    """A comparator was run, or scored, against something other than the shared pool.

    Raised rather than reported, because a comparison over two pools is not a
    weaker comparison -- it is a different experiment wearing the same table.
    """


# ==========================================================================
# The shared pool
# ==========================================================================

@dataclass(frozen=True)
class CandidatePool:
    """The one set of candidates every comparator must choose from.

    The digest covers each member's id *and* its sequence hash, so neither
    swapping a candidate for another with the same id nor adding one extra
    candidate can go unnoticed. It is the mechanism that turns "we used the
    same pool" from an assurance into a check.
    """

    pool_id: str
    candidates: tuple[Candidate, ...]
    digest: str

    @classmethod
    def build(cls, candidates: Sequence[Candidate],
              pool_id: str = "pool") -> "CandidatePool":
        items = tuple(candidates)
        ids = [c.candidate_id for c in items]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise PoolMismatchError(
                f"candidate id(s) {duplicates} appear twice in the pool; a "
                f"duplicated candidate occupies two slots and is counted twice "
                f"in every denominator")
        payload = sorted((c.candidate_id, c.sequence_record.sequence_sha256)
                         for c in items)
        return cls(pool_id=pool_id, candidates=items, digest=sha256_obj(payload))

    def ids(self) -> list[str]:
        return [c.candidate_id for c in self.candidates]

    def get(self, candidate_id: str) -> Candidate:
        for cand in self.candidates:
            if cand.candidate_id == candidate_id:
                return cand
        raise PoolMismatchError(
            f"'{candidate_id}' is not in pool {self.pool_id}")

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)


@dataclass(frozen=True)
class RankedSelection:
    """What one comparator would have ordered, and under what conditions.

    ``question_answered`` is a required field rather than documentation,
    because the most misleading row in a comparison table is the comparator
    that was answering a different question and is read as a weaker answer to
    this one.
    """

    comparator: str
    question_answered: str
    pool_digest: str
    criterion_digest: str
    budget: int
    ranked_candidate_ids: tuple[str, ...] = ()
    basis: str = ""
    unavailable_reason: str | None = None
    shortfall_reason: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def n_selected(self) -> int:
        return len(self.ranked_candidate_ids)

    @property
    def is_available(self) -> bool:
        return self.unavailable_reason is None

    def describe(self) -> str:
        if not self.is_available:
            return f"{self.comparator}: NOT RUN -- {self.unavailable_reason}"
        line = (f"{self.comparator}: {self.n_selected}/{self.budget} slot(s) "
                f"filled; answers: {self.question_answered}")
        if self.shortfall_reason:
            line += f"; SHORT: {self.shortfall_reason}"
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparator": self.comparator,
            "question_answered": self.question_answered,
            "pool_digest": self.pool_digest,
            "criterion_digest": self.criterion_digest,
            "budget": self.budget,
            "ranked_candidate_ids": list(self.ranked_candidate_ids),
            "basis": self.basis,
            "unavailable_reason": self.unavailable_reason,
            "shortfall_reason": self.shortfall_reason,
            "notes": list(self.notes),
            "n_selected": self.n_selected,
        }


class Comparator(Protocol):
    """The one signature every comparator has.

    Uniform so that a comparison cannot accidentally hand two comparators
    different inputs. Per-comparator configuration is bound by a factory
    before the run, where a reader can see it, rather than passed at call
    time where two comparators could receive different values.
    """

    def __call__(self, pool: CandidatePool, budget: int,
                 registration: PreRegistration) -> RankedSelection:
        ...


def comparator_name(comparator: Any) -> str:
    """Name a comparator reports itself by, for the row it occupies in a table.

    A factory-built comparator carries an explicit ``name`` so two models of
    the same kind can appear side by side; a plain function falls back to its
    ``__name__``. Named rather than positional because a comparison table
    whose rows are "comparator 1" and "comparator 2" cannot be read.
    """
    return str(getattr(comparator, "name", None)
               or getattr(comparator, "__name__", repr(comparator)))


def _selection(
    name: str, question: str, pool: CandidatePool, budget: int,
    registration: PreRegistration, ids: Sequence[str], basis: str,
    *, shortfall: str | None = None, notes: Sequence[str] = (),
) -> RankedSelection:
    """Build a selection, re-verifying the registration on the way out."""
    registration.assert_unchanged()
    return RankedSelection(
        comparator=name, question_answered=question, pool_digest=pool.digest,
        criterion_digest=registration.registered_digest, budget=budget,
        ranked_candidate_ids=tuple(ids), basis=basis,
        shortfall_reason=shortfall, notes=tuple(notes),
    )


def _unavailable(name: str, question: str, pool: CandidatePool, budget: int,
                 registration: PreRegistration, reason: str) -> RankedSelection:
    return RankedSelection(
        comparator=name, question_answered=question, pool_digest=pool.digest,
        criterion_digest=registration.registered_digest, budget=budget,
        unavailable_reason=reason,
    )


# ==========================================================================
# 1. Multi-seed homology search with diversity selection
# ==========================================================================

_HOMOLOGY_QUESTION = (
    "which pool sequences are most similar to a characterised seed, spread so "
    "the batch is not one clade")


def homology_multi_seed(
    pool: CandidatePool, budget: int, registration: PreRegistration,
) -> RankedSelection:
    """Rank by identity to the seeds, then spread the picks. The baseline to beat.

    WHY SEVERAL SEEDS ARE REQUIRED
    ------------------------------
    A search from one seed returns that seed's neighbourhood, and a batch
    drawn from it tests one clade with 96 slightly different constructs. A
    negative round then cannot distinguish "this chemistry does not work" from
    "this clade does not work". The pipeline refuses an unjustified single-seed
    search elsewhere (``single_seed_not_authorised``), and so does this
    comparator: with fewer than two distinct seeds it reports itself
    unavailable rather than quietly becoming a one-clade baseline that the
    agent then beats.

    Diversity comes from :func:`~eagent.science.diversity.greedy_submodular_select`
    with k-mer Jaccard distance -- the same selector the agent uses, so the
    comparison is between *what is ranked*, not between one method having a
    diversity step and the other not.
    """
    candidates = [c for c in pool.candidates
                  if c.sequence_record.percent_identity is not None]
    seeds = {c.sequence_record.seed_accession for c in pool.candidates
             if c.sequence_record.seed_accession}
    if len(seeds) < 2:
        return _unavailable(
            "homology_multi_seed", _HOMOLOGY_QUESTION, pool, budget, registration,
            f"only {len(seeds)} distinct seed accession(s) are recorded on this "
            f"pool. A single-seed homology baseline samples one clade, and a "
            f"comparison against it flatters whatever it is compared with; "
            f"record the seeds each sequence was found from, or drop this "
            f"comparator from the table")
    if not candidates:
        return _unavailable(
            "homology_multi_seed", _HOMOLOGY_QUESTION, pool, budget, registration,
            "no candidate in the pool records a percent identity to its seed, "
            "so there is nothing for a homology ranking to rank")

    utilities = {c.candidate_id: float(c.sequence_record.percent_identity) / 100.0
                 for c in candidates}
    picked = greedy_submodular_select(
        candidates, budget, utilities,
        lambda a, b: sequence_distance(a, b),
        DEFAULT_LAMBDA_WEIGHT,
    )
    shortfall = None
    if len(picked) < budget:
        shortfall = (f"{len(picked)} of {budget} slot(s): only {len(candidates)} "
                     f"pool member(s) carry a percent identity to a seed")
    return _selection(
        "homology_multi_seed", _HOMOLOGY_QUESTION, pool, budget, registration,
        [c.candidate_id for c in picked],
        basis=(f"percent identity to {len(seeds)} seed(s) as utility, k-mer "
               f"Jaccard distance for coverage, lambda={DEFAULT_LAMBDA_WEIGHT:g}"),
        shortfall=shortfall,
        notes=("identity to a characterised enzyme is evidence about the fold "
               "and only weak evidence about substrate scope; this comparator "
               "is strong precisely because that weak evidence is often "
               "enough",),
    )


# ==========================================================================
# 2. Family-level function prediction -- a different question
# ==========================================================================

_FAMILY_QUESTION = (
    "which family or EC class does this sequence belong to -- NOT whether it "
    "turns over this substrate under these conditions")

#: Stated once so it travels with every report rather than living in a
#: docstring nobody reads next to the table.
FAMILY_PREDICTION_CAVEAT: str = (
    "A family-level or EC-level call answers a different question from "
    "substrate-level specificity. Two members of one family routinely have "
    "non-overlapping substrate ranges, and the ketoreductase case is the "
    "textbook one: SDR, AKR and MDR/ADH all reduce ketones and none of them "
    "tells you which ketone. A family predictor scoring poorly on this "
    "endpoint has not been shown to be a poor family predictor."
)


def make_family_function_comparator(
    predictor: Callable[[Candidate], float | None] | None = None,
    *, name: str = "family_function_prediction",
) -> Comparator:
    """Bind a family-level function predictor, or leave the seam open.

    ``predictor`` returns a per-candidate score in the family-call sense (how
    confidently the sequence belongs to a family known to perform the reaction
    class). With no predictor the comparator reports itself unavailable: there
    is no fallback heuristic here, because a stand-in that imitates a family
    predictor would be compared with the agent and reported as a family
    predictor.
    """

    def comparator(pool: CandidatePool, budget: int,
                   registration: PreRegistration) -> RankedSelection:
        if predictor is None:
            return _unavailable(
                name, _FAMILY_QUESTION, pool, budget, registration,
                "no family-level function predictor is installed in this "
                "environment. " + FAMILY_PREDICTION_CAVEAT)
        scored: list[tuple[float, str]] = []
        unscored: list[str] = []
        for cand in pool.candidates:
            value = predictor(cand)
            if value is None:
                unscored.append(cand.candidate_id)
            else:
                scored.append((float(value), cand.candidate_id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        picked = [cid for _, cid in scored[:budget]]
        shortfall = None
        if len(picked) < budget:
            shortfall = (f"{len(picked)} of {budget} slot(s): the predictor "
                         f"returned no call for {len(unscored)} candidate(s), "
                         f"which are not ranked rather than ranked last")
        return _selection(
            name, _FAMILY_QUESTION, pool, budget, registration, picked,
            basis="family-call confidence from the injected predictor, "
                  "descending, ties broken by candidate id",
            shortfall=shortfall,
            notes=(FAMILY_PREDICTION_CAVEAT,),
        )

    comparator.name = name              # type: ignore[attr-defined]
    return comparator                   # type: ignore[return-value]


#: The seam as it stands in this environment: no predictor installed.
family_function_prediction: Comparator = make_family_function_comparator()


# ==========================================================================
# 3. Docking-score ranking
# ==========================================================================

_DOCKING_QUESTION = (
    "which candidate's best pose scores best under one docking scoring "
    "function")


def _best_docking_score(cand: Candidate,
                        lower_is_better: bool) -> tuple[float | None, set[str]]:
    """Best score over the candidate's valid poses, and the functions involved."""
    scores: list[float] = []
    functions: set[str] = set()
    for pose in cand.valid_poses:
        if pose.docking_score is None:
            continue
        scores.append(float(pose.docking_score))
        functions.add(pose.docking_score_function or "unnamed_scoring_function")
    if not scores:
        return None, functions
    return (min(scores) if lower_is_better else max(scores)), functions


def make_docking_comparator(
    *, lower_is_better: bool = True, name: str = "docking_score_ranking",
) -> Comparator:
    """Bind the sign convention, because guessing it inverts the ranking.

    Docking scoring functions disagree about sign: most pseudo-energies are
    "more negative is better", several machine-learned scores are the
    opposite, and a wrong assumption here produces a complete, plausible,
    exactly-backwards ranking that no downstream check would catch. It is
    therefore a named argument with no default discovery.
    """

    def comparator(pool: CandidatePool, budget: int,
                   registration: PreRegistration) -> RankedSelection:
        scored: list[tuple[float, str]] = []
        functions: set[str] = set()
        undocked: list[str] = []
        for cand in pool.candidates:
            score, funcs = _best_docking_score(cand, lower_is_better)
            functions |= funcs
            if score is None:
                undocked.append(cand.candidate_id)
            else:
                scored.append((score, cand.candidate_id))
        if len(functions) > 1:
            return _unavailable(
                name, _DOCKING_QUESTION, pool, budget, registration,
                f"this pool carries docking scores from more than one scoring "
                f"function ({', '.join(sorted(functions))}). Their values are "
                f"not on one scale, and ranking across them ranks the choice "
                f"of program; re-dock the pool with one function")
        if not scored:
            return _unavailable(
                name, _DOCKING_QUESTION, pool, budget, registration,
                "no candidate in the pool has a valid pose carrying a docking "
                "score")
        scored.sort(key=lambda pair: (pair[0] if lower_is_better else -pair[0],
                                      pair[1]))
        picked = [cid for _, cid in scored[:budget]]
        shortfall = None
        if len(picked) < budget:
            shortfall = (f"{len(picked)} of {budget} slot(s): "
                         f"{len(undocked)} candidate(s) have no scored pose and "
                         f"are left unranked rather than ranked last")
        return _selection(
            name, _DOCKING_QUESTION, pool, budget, registration, picked,
            basis=(f"best score per candidate over valid poses, "
                   f"{'lower' if lower_is_better else 'higher'} is better, "
                   f"function: {', '.join(sorted(functions)) or 'unnamed'}"),
            shortfall=shortfall,
            notes=("docking scores are comparable within one pocket class and "
                   "one scoring function; across families this ranking is "
                   "substantially a ranking of pocket volume",),
        )

    comparator.name = name              # type: ignore[attr-defined]
    return comparator                   # type: ignore[return-value]


docking_score_ranking: Comparator = make_docking_comparator()


# ==========================================================================
# 4. Substrate-specificity model -- the right comparator, and a seam
# ==========================================================================

_SPECIFICITY_QUESTION = (
    "how likely is this enzyme to turn over THIS substrate -- the same "
    "question the agent is asked")


def make_specificity_model_comparator(
    model: Callable[[Candidate], float | None] | None = None,
    *, name: str = "substrate_specificity_model", model_id: str | None = None,
) -> Comparator:
    """Bind a trained substrate-specificity model, or leave the seam open.

    This is the comparator that matters: unlike the family predictor it is
    answering the agent's own question, so a difference here is a difference in
    method rather than in task. ``model_id`` is recorded in the selection's
    basis, because a specificity number means nothing without knowing what the
    model was trained on -- and whether it was trained on data that overlaps
    this pool, which is what :mod:`eagent.eval.splits` exists to check.
    """

    def comparator(pool: CandidatePool, budget: int,
                   registration: PreRegistration) -> RankedSelection:
        if model is None:
            return _unavailable(
                name, _SPECIFICITY_QUESTION, pool, budget, registration,
                "no substrate-specificity model is installed in this "
                "environment. The seam is left open rather than filled with a "
                "heuristic: a stand-in the agent beats would be the most "
                "flattering possible comparison and would mean nothing")
        scored: list[tuple[float, str]] = []
        unscored: list[str] = []
        for cand in pool.candidates:
            value = model(cand)
            if value is None:
                unscored.append(cand.candidate_id)
            else:
                scored.append((float(value), cand.candidate_id))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        picked = [cid for _, cid in scored[:budget]]
        shortfall = None
        if len(picked) < budget:
            shortfall = (f"{len(picked)} of {budget} slot(s): the model "
                         f"returned no prediction for {len(unscored)} "
                         f"candidate(s)")
        return _selection(
            name, _SPECIFICITY_QUESTION, pool, budget, registration, picked,
            basis=f"substrate-specificity model '{model_id or 'unnamed'}', "
                  f"descending score, ties broken by candidate id",
            shortfall=shortfall,
            notes=("a specificity model's score on this pool is only "
                   "interpretable once the pool has been shown absent from its "
                   "training data; see eagent.eval.splits.audit_leakage",),
        )

    comparator.name = name              # type: ignore[attr-defined]
    return comparator                   # type: ignore[return-value]


substrate_specificity_model: Comparator = make_specificity_model_comparator()


# ==========================================================================
# 5. The agent itself
# ==========================================================================

_AGENT_QUESTION = (
    "which candidates should this round's slots be spent on, given mechanism "
    "gates, evidence priority, family and clade quotas and pocket coverage")


def full_agent(
    pool: CandidatePool, budget: int, registration: PreRegistration,
) -> RankedSelection:
    """The pipeline's own selection, through the code that would actually run.

    Calls :func:`eagent.science.diversity.compose_batch` rather than
    re-implementing its logic, so what is evaluated is the selection the
    harness would make, including its refusal to admit an ungated candidate
    and its refusal to pad a short batch.

    An ungated pool is reported as unavailable rather than silently ranked:
    composing a batch from candidates whose feasibility was never evaluated
    would treat "never checked" as "eligible", and the comparison would credit
    the agent with gates it did not apply.
    """
    try:
        plan = compose_batch(
            list(pool.candidates),
            Budget(new_constructs_round_1=budget,
                   detailed_complex_target=max(budget, len(pool))),
            plan_id=f"baseline-{pool.pool_id}",
        )
    except ValueError as exc:
        return _unavailable(
            "full_agent", _AGENT_QUESTION, pool, budget, registration,
            f"the agent's selection refused this pool: {exc}")
    members = sorted(plan.members, key=lambda m: m.slot)
    return _selection(
        "full_agent", _AGENT_QUESTION, pool, budget, registration,
        [m.candidate_id for m in members],
        basis=("feasibility gates, then lexicographic rank over the stated "
               "dimension priority, then uncertainty probes, then pocket-"
               "coverage diversity, under family and clade quotas"),
        shortfall=plan.shortfall_reason,
        notes=tuple(f"role {role}: {n}"
                    for role, n in sorted(plan.role_counts().items())),
    )


#: The comparators run by default. Order is presentation only.
DEFAULT_COMPARATORS: tuple[Comparator, ...] = (
    homology_multi_seed,
    family_function_prediction,
    docking_score_ranking,
    substrate_specificity_model,
    full_agent,
)


# ==========================================================================
# Running and scoring the comparison
# ==========================================================================

@dataclass(frozen=True)
class ComparatorScore:
    """One comparator's selection and what it would have been worth.

    ``n_without_outcome`` is first-class because a retrospective comparison is
    only fair over candidates whose outcome is known. A comparator that picked
    ten candidates nobody ever tested has not scored badly; it has not been
    scored, and ``fair`` says so.
    """

    comparator: str
    selection: RankedSelection
    endpoint: PrecisionAtK | None
    n_without_outcome: int
    note: str = ""

    @property
    def fair(self) -> bool:
        return (self.selection.is_available and self.endpoint is not None
                and self.n_without_outcome == 0)

    @property
    def n_hits(self) -> int:
        return 0 if self.endpoint is None else self.endpoint.n_hits

    def describe(self) -> str:
        if self.endpoint is None:
            if not self.selection.is_available:
                return self.selection.describe()
            return f"{self.selection.describe()} -- not scored: {self.note}"
        suffix = ("" if self.fair else
                  f" [NOT A FAIR SCORE: {self.n_without_outcome} pick(s) have "
                  f"no recorded outcome]")
        return (f"{self.comparator}: "
                f"{self.endpoint.rate_over_slots_spent.describe()}{suffix}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparator": self.comparator,
            "selection": self.selection.to_dict(),
            "endpoint": self.endpoint.to_dict() if self.endpoint else None,
            "n_without_outcome": self.n_without_outcome,
            "fair": self.fair,
            "note": self.note,
        }


def score_selection(
    selection: RankedSelection,
    outcomes: Mapping[str, OutcomeRow],
    registration: PreRegistration,
    *,
    known_candidate_ids: Iterable[str] = (),
) -> ComparatorScore:
    """Score one selection against recorded outcomes, under the registered bar.

    Picks with no recorded outcome are counted and reported rather than
    treated as failures: an untested candidate is not a negative, and scoring
    it as one would reward whichever comparator happened to pick the
    candidates someone already ran.
    """
    registration.assert_unchanged()
    if selection.criterion_digest != registration.registered_digest:
        raise PoolMismatchError(
            f"selection '{selection.comparator}' was made under criterion "
            f"{selection.criterion_digest} and is being scored under "
            f"{registration.registered_digest}; one comparison, one hit "
            f"definition")
    if not selection.is_available:
        return ComparatorScore(
            comparator=selection.comparator, selection=selection, endpoint=None,
            n_without_outcome=0,
            note=selection.unavailable_reason or "comparator did not run")

    rows: list[OutcomeRow] = []
    missing = 0
    for cid in selection.ranked_candidate_ids:
        row = outcomes.get(cid)
        if row is None:
            missing += 1
            continue
        rows.append(row)
    if not rows:
        return ComparatorScore(
            comparator=selection.comparator, selection=selection, endpoint=None,
            n_without_outcome=missing,
            note=("none of this comparator's picks has a recorded outcome, so "
                  "it cannot be scored on this round"))
    endpoint = precision_at_k(rows, registration,
                              known_candidate_ids=known_candidate_ids)
    return ComparatorScore(
        comparator=selection.comparator, selection=selection, endpoint=endpoint,
        n_without_outcome=missing,
        note=("" if missing == 0 else
              f"{missing} pick(s) were never tested and are outside both the "
              f"numerator and the denominator"))


@dataclass(frozen=True)
class BaselineComparison:
    """Every comparator on one pool, one budget and one criterion."""

    pool_id: str
    pool_digest: str
    criterion_digest: str
    budget: int
    scores: tuple[ComparatorScore, ...]

    def ordered_by_primary_endpoint(self) -> list[ComparatorScore]:
        """Scored comparators, most registered hits first, then by name.

        The ordering key is the single pre-registered endpoint -- one measured
        quantity, not a blend -- and the order is not a verdict: see
        :meth:`indistinguishable_pairs`.
        """
        scored = [s for s in self.scores if s.endpoint is not None]
        return sorted(scored, key=lambda s: (-s.n_hits, s.comparator))

    def indistinguishable_pairs(self) -> list[tuple[str, str]]:
        """Comparator pairs whose Wilson intervals overlap.

        At a 96-construct budget almost every pair lands here, which is the
        honest headline: one round separates only large differences, and a
        table ordered by point estimate invites the opposite reading.
        """
        out: list[tuple[str, str]] = []
        scored = [s for s in self.scores if s.endpoint is not None]
        for i, a in enumerate(scored):
            for b in scored[i + 1:]:
                ia = a.endpoint.rate_over_slots_spent.interval   # type: ignore[union-attr]
                ib = b.endpoint.rate_over_slots_spent.interval   # type: ignore[union-attr]
                if ia is None or ib is None:
                    continue
                if ia[0] <= ib[1] and ib[0] <= ia[1]:
                    out.append(tuple(sorted((a.comparator, b.comparator))))
        return sorted(set(out))

    @property
    def unavailable(self) -> tuple[str, ...]:
        return tuple(s.comparator for s in self.scores
                     if not s.selection.is_available)

    def render(self) -> str:
        lines = [
            f"baseline comparison on pool {self.pool_id} "
            f"({self.pool_digest[:12]}), budget {self.budget}, criterion "
            f"{self.criterion_digest[:12]}",
        ]
        for score in self.ordered_by_primary_endpoint():
            lines.append("  " + score.describe())
        for score in self.scores:
            if score.endpoint is None:
                lines.append("  " + score.describe())
        pairs = self.indistinguishable_pairs()
        if pairs:
            lines.append("  intervals overlap (not separated by this round): "
                         + "; ".join(f"{a} vs {b}" for a, b in pairs))
        if self.unavailable:
            lines.append("  comparators not run: " + ", ".join(self.unavailable))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_id": self.pool_id,
            "pool_digest": self.pool_digest,
            "criterion_digest": self.criterion_digest,
            "budget": self.budget,
            "scores": [s.to_dict() for s in self.scores],
            "indistinguishable_pairs": [list(p)
                                        for p in self.indistinguishable_pairs()],
            "unavailable": list(self.unavailable),
        }


def compare_baselines(
    pool: CandidatePool,
    budget: int,
    registration: PreRegistration,
    comparators: Sequence[Comparator] = DEFAULT_COMPARATORS,
    *,
    outcomes: Mapping[str, OutcomeRow] | None = None,
    known_candidate_ids: Iterable[str] = (),
) -> BaselineComparison:
    """Run every comparator on the same pool, budget and criterion, and check it.

    The shared pool is enforced here rather than trusted. Each returned
    selection must declare this pool's digest and this registration's digest,
    must name only candidates the pool contains, must not name one twice, and
    must not exceed the budget. Any violation raises
    :class:`PoolMismatchError`, because a comparison in which one comparator
    saw a different pool is not a weaker result, it is a different experiment.

    Raises
    ------
    PoolMismatchError
    ValueError
        If ``budget`` is not positive, or two comparators share a name -- which
        would silently overwrite one of them in the result.
    """
    if budget < 1:
        raise ValueError(f"budget must be at least one slot, got {budget}")
    registration.assert_unchanged()
    pool_ids = set(pool.ids())

    names = [comparator_name(c) for c in comparators]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"comparator name(s) {duplicates} appear twice; two rows with one "
            f"name hide one of the two comparators")

    scores: list[ComparatorScore] = []
    for comparator, name in zip(comparators, names):
        selection = comparator(pool, budget, registration)
        if selection.pool_digest != pool.digest:
            raise PoolMismatchError(
                f"comparator '{name}' returned a selection made against pool "
                f"digest {selection.pool_digest}, not the shared pool "
                f"{pool.digest}")
        if selection.criterion_digest != registration.registered_digest:
            raise PoolMismatchError(
                f"comparator '{name}' recorded criterion "
                f"{selection.criterion_digest}, not the registered "
                f"{registration.registered_digest}")
        if selection.budget != budget:
            raise PoolMismatchError(
                f"comparator '{name}' recorded budget {selection.budget}, not "
                f"the shared {budget}")
        picked = list(selection.ranked_candidate_ids)
        outside = sorted(set(picked) - pool_ids)
        if outside:
            raise PoolMismatchError(
                f"comparator '{name}' selected {outside}, which is not in the "
                f"shared pool; a comparator that can reach outside the pool is "
                f"not being compared on the same problem")
        repeated = sorted({c for c in picked if picked.count(c) > 1})
        if repeated:
            raise PoolMismatchError(
                f"comparator '{name}' selected {repeated} more than once; a "
                f"repeated pick spends two slots on one construct")
        if len(picked) > budget:
            raise PoolMismatchError(
                f"comparator '{name}' selected {len(picked)} candidates against "
                f"a budget of {budget}; comparing methods at different budgets "
                f"compares budgets")
        if outcomes is None:
            scores.append(ComparatorScore(
                comparator=name, selection=selection, endpoint=None,
                n_without_outcome=len(picked),
                note=("no outcomes were supplied, so the selections are "
                      "recorded and not scored")))
        else:
            score = score_selection(selection, outcomes, registration,
                                    known_candidate_ids=known_candidate_ids)
            scores.append(ComparatorScore(
                comparator=name, selection=score.selection,
                endpoint=score.endpoint,
                n_without_outcome=score.n_without_outcome, note=score.note))

    return BaselineComparison(
        pool_id=pool.pool_id, pool_digest=pool.digest,
        criterion_digest=registration.registered_digest, budget=budget,
        scores=tuple(scores),
    )
