"""Removing one module at a time, over a fixed pool, and seeing what changes.

WHAT A 96-WELL ROUND CAN AND CANNOT ESTABLISH
=============================================
**96 experiments on a single substrate can establish a first application and
cannot establish generality across reaction types.** Everything this module
produces is subject to that sentence. One round, one ketone, one family set
and one laboratory's conditions can show that this pipeline found working
enzymes for this reaction, which is a real result. It cannot show that the
geometry module "contributes 8 percent", that the cofactor gate is necessary
in general, or that the same ablation ordering would hold for a transaminase
or a halogenase. A difference of one or two hits out of 96 is inside the
Wilson interval of almost any comparison, and
:attr:`AblationResult.intervals_overlap` says so for every pair rather than
leaving the point estimate to be read as a finding.

HOW AN ABLATION IS RUN HERE
===========================
The pool is fixed, the budget is fixed, and the pre-registered hit definition
is fixed (:class:`~eagent.eval.metrics.PreRegistration`). One module is
removed, the *same* selection code re-runs
(:func:`eagent.science.diversity.compose_batch`), and the primary endpoint is
recomputed over whichever of the picked candidates have recorded outcomes.
Nothing is simulated: a candidate a selection picks that nobody ever tested
is counted as unscorable and named, because treating it as a failure would
reward whichever variant happened to pick the candidates someone had already
run.

WHICH MODULE EACH ABLATION ACTUALLY REMOVES
===========================================
The mapping from "module" to "what is switched off" is stated as data in
:class:`AblatedModule`, not left to the reader:

``catalytic_geometry``
    the ``catalytic_geometry`` ranking axis and the
    ``catalytic_machinery_mappable`` feasibility gate -- everything the
    template-driven geometry layer contributes to selection.
``cofactor_constraints``
    the ``cofactor_compatible`` gate. Removing it admits candidates whose
    mechanism needs a cofactor the construct cannot use.
``family_diversity_quota``
    the family quota and the sequence-cluster cap, so an over-sequenced clade
    can take the whole plate.
``active_learning``
    the between-round update. **This one is not evaluable from a single
    round**, and says so rather than producing a number: with no prior round
    there is nothing for the module to have learned from. Supply a
    :class:`PriorRound` -- whose quota update comes from
    :func:`eagent.tools.ingest_results.next_round_quotas` -- and the ablation
    becomes meaningful.

REMOVING A GATE IS NOT THE SAME AS FAILING IT
=============================================
An ablation strips the gate from a *copy* of each candidate's scorecard, so a
candidate that would have been excluded becomes eligible. That is the point:
the question is what the module kept out of the plate. The originals are
never mutated, so the full-system selection and the ablated one are computed
from the same inputs.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..schemas import Budget, Candidate
from ..science.diversity import DEFAULT_LAMBDA_WEIGHT, compose_batch
from ..science.scorecard import DEFAULT_LEXICOGRAPHIC_ORDER, GATE_NAMES
from .baselines import CandidatePool, RankedSelection, score_selection
from .metrics import OutcomeRow, PrecisionAtK, PreRegistration

__all__ = [
    "SCOPE_STATEMENT",
    "AblatedModule",
    "PriorRound",
    "SelectionSettings",
    "AblationResult",
    "AblationReport",
    "select_under",
    "run_ablations",
]

#: Repeated into every rendered report, because the caveat is the result's
#: scope and a scope that lives only in a docstring does not travel with the
#: table someone copies into a slide.
SCOPE_STATEMENT: str = (
    "96 experiments on a single substrate can establish a first application "
    "and cannot establish generality across reaction types. These deltas "
    "describe this pool, this substrate and this round; they are not "
    "measurements of how much each module is worth in general, and most of "
    "them are smaller than the Wilson interval around either endpoint."
)


class AblatedModule(str, enum.Enum):
    """One removable module, and exactly what removing it switches off."""

    CATALYTIC_GEOMETRY = "catalytic_geometry"
    COFACTOR_CONSTRAINTS = "cofactor_constraints"
    FAMILY_DIVERSITY_QUOTA = "family_diversity_quota"
    ACTIVE_LEARNING = "active_learning"

    @property
    def gates_removed(self) -> tuple[str, ...]:
        """Feasibility gates stripped from the scorecard copy."""
        return {
            AblatedModule.CATALYTIC_GEOMETRY: ("catalytic_machinery_mappable",),
            AblatedModule.COFACTOR_CONSTRAINTS: ("cofactor_compatible",),
            AblatedModule.FAMILY_DIVERSITY_QUOTA: (),
            AblatedModule.ACTIVE_LEARNING: (),
        }[self]

    @property
    def dimensions_removed(self) -> tuple[str, ...]:
        """Ranking axes dropped from the lexicographic priority order."""
        return {
            AblatedModule.CATALYTIC_GEOMETRY: ("catalytic_geometry",),
            AblatedModule.COFACTOR_CONSTRAINTS: (),
            AblatedModule.FAMILY_DIVERSITY_QUOTA: (),
            AblatedModule.ACTIVE_LEARNING: (),
        }[self]

    def what_it_removes(self) -> str:
        return {
            AblatedModule.CATALYTIC_GEOMETRY:
                "the template-driven catalytic geometry: its ranking axis and "
                "the gate that requires the catalytic machinery to be mappable",
            AblatedModule.COFACTOR_CONSTRAINTS:
                "the cofactor compatibility gate, so a mechanism that needs a "
                "cofactor the construct cannot use is no longer excluded",
            AblatedModule.FAMILY_DIVERSITY_QUOTA:
                "the family quota and the sequence-cluster cap, so one "
                "over-sequenced clade may take the whole plate",
            AblatedModule.ACTIVE_LEARNING:
                "the between-round update: the prior round's outcomes no "
                "longer exclude tested constructs or reshape family quotas",
        }[self]

    @property
    def needs_prior_round(self) -> bool:
        """Whether this module only acts between rounds, so one round cannot test it."""
        return self is AblatedModule.ACTIVE_LEARNING


@dataclass(frozen=True)
class PriorRound:
    """What an earlier round contributed to this round's selection.

    This is the input the active-learning ablation needs, and it is required
    rather than inferred: the pipeline's between-round update is
    :func:`eagent.tools.ingest_results.next_round_quotas` plus the exclusion
    of constructs already tested, and both of those are decisions a previous
    round recorded. Reconstructing them here from outcomes would be this
    module guessing what the previous round did.
    """

    tested_candidate_ids: frozenset[str] = frozenset()
    quotas_before: Mapping[str, int] | None = None
    quotas_after: Mapping[str, int] | None = None

    @property
    def is_informative(self) -> bool:
        """Whether anything actually came back from the earlier round."""
        return bool(self.tested_candidate_ids) or self.quotas_after is not None


@dataclass(frozen=True)
class SelectionSettings:
    """Everything a selection run depends on, so an ablation differs in one thing.

    Held as a value object rather than as keyword arguments scattered through
    the call sites, because the credibility of an ablation rests entirely on
    the two runs being identical apart from the module under test.
    """

    quotas: Mapping[str, int] | None = None
    cluster_cap: int | None = None
    order_of_dimensions: tuple[str, ...] = DEFAULT_LEXICOGRAPHIC_ORDER
    gates_applied: tuple[str, ...] = GATE_NAMES
    excluded_candidate_ids: frozenset[str] = frozenset()
    lambda_weight: float = DEFAULT_LAMBDA_WEIGHT

    def without(self, module: AblatedModule,
                prior: "PriorRound | None" = None) -> "SelectionSettings":
        """The same settings with one module switched off."""
        gates = tuple(g for g in self.gates_applied
                      if g not in module.gates_removed)
        order = tuple(d for d in self.order_of_dimensions
                      if d not in module.dimensions_removed)
        quotas = self.quotas
        cluster_cap = self.cluster_cap
        excluded = self.excluded_candidate_ids
        if module is AblatedModule.FAMILY_DIVERSITY_QUOTA:
            quotas = None
            cluster_cap = None
        if module is AblatedModule.ACTIVE_LEARNING:
            # Without the between-round update the round is planned as if the
            # previous one had not reported: its quotas revert and the
            # constructs it already tested are candidates again.
            quotas = prior.quotas_before if prior is not None else self.quotas
            excluded = frozenset()
        return SelectionSettings(
            quotas=quotas, cluster_cap=cluster_cap, order_of_dimensions=order,
            gates_applied=gates, excluded_candidate_ids=excluded,
            lambda_weight=self.lambda_weight,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "quotas": dict(self.quotas) if self.quotas else None,
            "cluster_cap": self.cluster_cap,
            "order_of_dimensions": list(self.order_of_dimensions),
            "gates_applied": list(self.gates_applied),
            "excluded_candidate_ids": sorted(self.excluded_candidate_ids),
            "lambda_weight": self.lambda_weight,
        }


def _strip_gates(candidate: Candidate, keep: Sequence[str]) -> Candidate:
    """A deep copy whose scorecard keeps only the named gates.

    A copy, never an edit: the full-system run and the ablated run must see
    the same original candidates, or the comparison measures the order the two
    runs happened to execute in.
    """
    copy = candidate.model_copy(deep=True)
    for name in list(copy.scorecard):
        dim = copy.scorecard[name]
        if dim.is_gate and name not in keep:
            del copy.scorecard[name]
    return copy


def select_under(
    pool: CandidatePool,
    budget: int,
    settings: SelectionSettings,
    *,
    plan_id: str = "ablation",
) -> tuple[list[str], str | None, tuple[str, ...]]:
    """Run the pipeline's own selection under these settings.

    Returns ``(candidate_ids_in_slot_order, shortfall_reason, notes)``. The
    selection is :func:`eagent.science.diversity.compose_batch` -- the code the
    harness would run -- so an ablation cannot accidentally compare the agent
    with a reimplementation of itself.

    A candidate left with no gate at all after stripping is dropped rather than
    admitted: ``compose_batch`` refuses an ungated pool outright, and admitting
    it here would turn "every gate was ablated" into "every candidate is
    eligible", which is a different experiment.
    """
    members = [c for c in pool.candidates
               if c.candidate_id not in settings.excluded_candidate_ids]
    stripped = [_strip_gates(c, settings.gates_applied) for c in members]
    ungated = [c.candidate_id for c in stripped if not c.gates()]
    usable = [c for c in stripped if c.gates()]
    notes: list[str] = []
    if ungated:
        notes.append(
            f"{len(ungated)} candidate(s) were left with no feasibility gate "
            f"once this module was removed and are not selectable; "
            f"compose_batch refuses an ungated candidate because 'never "
            f"checked' is not 'eligible' (first: {sorted(ungated)[0]})")
    if not usable:
        return [], ("no candidate carries a feasibility gate under these "
                    "settings, so no selection was made"), tuple(notes)
    plan = compose_batch(
        usable,
        Budget(new_constructs_round_1=budget,
               detailed_complex_target=max(budget, len(usable))),
        quotas=settings.quotas,
        cluster_cap=settings.cluster_cap,
        order_of_dimensions=settings.order_of_dimensions,
        lambda_weight=settings.lambda_weight,
        plan_id=plan_id,
    )
    ordered = [m.candidate_id for m in sorted(plan.members, key=lambda m: m.slot)]
    return ordered, plan.shortfall_reason, tuple(notes)


@dataclass(frozen=True)
class AblationResult:
    """One module removed: what the selection and the primary endpoint did.

    ``delta_hits`` is a count, not a rate difference with a confidence claim.
    ``intervals_overlap`` is the number that decides whether the delta is
    readable at all, and it is ``True`` for most module/round combinations at
    this budget.
    """

    module: AblatedModule
    applicable: bool
    full_system_ids: tuple[str, ...] = ()
    ablated_ids: tuple[str, ...] = ()
    full_system_endpoint: PrecisionAtK | None = None
    ablated_endpoint: PrecisionAtK | None = None
    reason_not_applicable: str | None = None
    n_unscorable_full: int = 0
    n_unscorable_ablated: int = 0
    notes: tuple[str, ...] = ()

    @property
    def candidates_added(self) -> tuple[str, ...]:
        """Candidates the ablated selection took that the full system did not."""
        return tuple(sorted(set(self.ablated_ids) - set(self.full_system_ids)))

    @property
    def candidates_dropped(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.full_system_ids) - set(self.ablated_ids)))

    @property
    def selection_changed(self) -> bool:
        return bool(self.candidates_added or self.candidates_dropped)

    @property
    def delta_hits(self) -> int | None:
        """Ablated minus full-system registered hits, or ``None`` if unscored."""
        if self.full_system_endpoint is None or self.ablated_endpoint is None:
            return None
        return self.ablated_endpoint.n_hits - self.full_system_endpoint.n_hits

    @property
    def delta_rate(self) -> float | None:
        """Difference of the two point estimates over slots spent.

        A difference of point estimates and nothing more. It carries no
        interval of its own, which is why :attr:`intervals_overlap` is
        reported beside it and why :meth:`describe` never prints one without
        the other.
        """
        if self.full_system_endpoint is None or self.ablated_endpoint is None:
            return None
        a = self.ablated_endpoint.rate_over_slots_spent.point
        b = self.full_system_endpoint.rate_over_slots_spent.point
        if a is None or b is None:
            return None
        return a - b

    @property
    def intervals_overlap(self) -> bool | None:
        """Whether this round can separate the two endpoints at all."""
        if self.full_system_endpoint is None or self.ablated_endpoint is None:
            return None
        ia = self.full_system_endpoint.rate_over_slots_spent.interval
        ib = self.ablated_endpoint.rate_over_slots_spent.interval
        if ia is None or ib is None:
            return None
        return ia[0] <= ib[1] and ib[0] <= ia[1]

    def describe(self) -> str:
        if not self.applicable:
            return (f"{self.module.value}: NOT EVALUABLE -- "
                    f"{self.reason_not_applicable}")
        if self.delta_hits is None:
            return (f"{self.module.value}: selection "
                    f"{'changed' if self.selection_changed else 'unchanged'}; "
                    f"not scored (no outcomes for the picks)")
        verdict = ("the intervals overlap, so this round does not separate the "
                   "two" if self.intervals_overlap
                   else "the intervals do not overlap")
        return (f"{self.module.value}: removing {self.module.what_it_removes()} "
                f"changed the primary endpoint by {self.delta_hits:+d} hit(s) "
                f"({(self.delta_rate or 0.0) * 100:+.1f} points); "
                f"{len(self.candidates_added)} candidate(s) entered the plate "
                f"and {len(self.candidates_dropped)} left it; {verdict}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module.value,
            "removes": self.module.what_it_removes(),
            "applicable": self.applicable,
            "reason_not_applicable": self.reason_not_applicable,
            "full_system_ids": list(self.full_system_ids),
            "ablated_ids": list(self.ablated_ids),
            "candidates_added": list(self.candidates_added),
            "candidates_dropped": list(self.candidates_dropped),
            "full_system_endpoint": (self.full_system_endpoint.to_dict()
                                     if self.full_system_endpoint else None),
            "ablated_endpoint": (self.ablated_endpoint.to_dict()
                                 if self.ablated_endpoint else None),
            "delta_hits": self.delta_hits,
            "delta_rate": self.delta_rate,
            "intervals_overlap": self.intervals_overlap,
            "n_unscorable_full": self.n_unscorable_full,
            "n_unscorable_ablated": self.n_unscorable_ablated,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class AblationReport:
    """Every ablation on one pool, with the scope statement attached."""

    pool_id: str
    pool_digest: str
    criterion_digest: str
    budget: int
    settings: SelectionSettings
    full_system_ids: tuple[str, ...]
    full_system_endpoint: PrecisionAtK | None
    results: tuple[AblationResult, ...]
    #: Shortfalls and refusals of the full-system run. Kept here rather than
    #: copied into every result, where the same paragraph repeated four times
    #: hides the one note that differs.
    full_system_notes: tuple[str, ...] = ()
    scope_statement: str = SCOPE_STATEMENT

    @property
    def evaluable(self) -> tuple[AblationResult, ...]:
        return tuple(r for r in self.results if r.applicable)

    @property
    def separated(self) -> tuple[AblationResult, ...]:
        """Ablations whose endpoint intervals this round actually separates."""
        return tuple(r for r in self.results if r.intervals_overlap is False)

    def render(self) -> str:
        lines = [
            f"ablations on pool {self.pool_id} ({self.pool_digest[:12]}), "
            f"budget {self.budget}, criterion {self.criterion_digest[:12]}",
        ]
        if self.full_system_endpoint is not None:
            lines.append("  full system: "
                         + self.full_system_endpoint.describe())
        for note in self.full_system_notes:
            lines.append(f"  full-system note: {note}")
        for result in self.results:
            lines.append("  " + result.describe())
            for note in result.notes:
                lines.append(f"      note: {note}")
        if not self.separated:
            lines.append("  no ablation was separated from the full system by "
                         "this round")
        lines.append("  SCOPE: " + self.scope_statement)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pool_id": self.pool_id,
            "pool_digest": self.pool_digest,
            "criterion_digest": self.criterion_digest,
            "budget": self.budget,
            "settings": self.settings.to_dict(),
            "full_system_ids": list(self.full_system_ids),
            "full_system_endpoint": (self.full_system_endpoint.to_dict()
                                     if self.full_system_endpoint else None),
            "full_system_notes": list(self.full_system_notes),
            "results": [r.to_dict() for r in self.results],
            "scope_statement": self.scope_statement,
        }


def _score(ids: Sequence[str], pool: CandidatePool, budget: int,
           registration: PreRegistration, outcomes: Mapping[str, OutcomeRow],
           label: str, known: Iterable[str]) -> tuple[PrecisionAtK | None, int]:
    """Score one selection with the shared, guarded endpoint code."""
    selection = RankedSelection(
        comparator=label,
        question_answered=("which candidates this round's slots are spent on, "
                           "under the stated ablation"),
        pool_digest=pool.digest,
        criterion_digest=registration.registered_digest,
        budget=budget,
        ranked_candidate_ids=tuple(ids),
    )
    score = score_selection(selection, outcomes, registration,
                            known_candidate_ids=known)
    return score.endpoint, score.n_without_outcome


def run_ablations(
    pool: CandidatePool,
    budget: int,
    registration: PreRegistration,
    outcomes: Mapping[str, OutcomeRow],
    *,
    settings: SelectionSettings | None = None,
    modules: Sequence[AblatedModule] = tuple(AblatedModule),
    prior_round: PriorRound | None = None,
    known_candidate_ids: Iterable[str] = (),
) -> AblationReport:
    """Remove each module in turn over a fixed pool and report the endpoint change.

    The full-system selection is computed once from ``settings`` and every
    ablation differs from it in exactly one thing. The endpoint is the
    pre-registered one and is recomputed through
    :func:`eagent.eval.metrics.precision_at_k`, so an ablation cannot be
    scored against a criterion the round did not register.

    ``prior_round`` is required for the active-learning ablation and ignored by
    the others. Without it that ablation is reported **not evaluable**, with
    the reason, instead of returning a delta of zero -- a zero would read as
    "active learning contributed nothing", when the truth is that a single
    round contains nothing for a between-round module to have done.

    Raises
    ------
    ValueError
        If the budget exceeds the registered one. These would then not be the
        round that was pre-registered, and the endpoint's denominator would
        answer a question nobody asked.
    """
    if budget < 1:
        raise ValueError(f"budget must be at least one slot, got {budget}")
    if budget > registration.k_slots:
        raise ValueError(
            f"ablating at a budget of {budget} against a pre-registered "
            f"{registration.k_slots} slot(s) compares two different rounds")
    registration.assert_unchanged()
    base = settings or SelectionSettings()

    full_ids, full_shortfall, full_notes = select_under(
        pool, budget, base, plan_id=f"{pool.pool_id}-full")
    full_endpoint, full_unscorable = _score(
        full_ids, pool, budget, registration, outcomes, "full_system",
        known_candidate_ids)

    results: list[AblationResult] = []
    for module in modules:
        if module.needs_prior_round and (prior_round is None
                                         or not prior_round.is_informative):
            results.append(AblationResult(
                module=module, applicable=False,
                full_system_ids=tuple(full_ids),
                full_system_endpoint=full_endpoint,
                n_unscorable_full=full_unscorable,
                reason_not_applicable=(
                    "active learning acts between rounds, and this evaluation "
                    "has one round. With no prior round there is nothing the "
                    "module could have learned from, so removing it cannot "
                    "change anything and a delta of zero would be read as "
                    "'this module contributes nothing'. Supply a PriorRound "
                    "(its quota update comes from "
                    "ingest_results.next_round_quotas) to make this ablation "
                    "meaningful"),
            ))
            continue
        ablated_settings = base.without(module, prior_round)
        ablated_ids, shortfall, notes = select_under(
            pool, budget, ablated_settings,
            plan_id=f"{pool.pool_id}-without-{module.value}")
        ablated_endpoint, unscorable = _score(
            ablated_ids, pool, budget, registration, outcomes,
            f"without_{module.value}", known_candidate_ids)
        all_notes = list(notes)
        if shortfall:
            all_notes.append(f"ablated selection short: {shortfall}")
        if unscorable or full_unscorable:
            all_notes.append(
                f"{unscorable} ablated pick(s) and {full_unscorable} "
                f"full-system pick(s) have no recorded outcome; they are "
                f"outside both numerator and denominator, not counted as "
                f"failures")
        results.append(AblationResult(
            module=module, applicable=True,
            full_system_ids=tuple(full_ids),
            ablated_ids=tuple(ablated_ids),
            full_system_endpoint=full_endpoint,
            ablated_endpoint=ablated_endpoint,
            n_unscorable_full=full_unscorable,
            n_unscorable_ablated=unscorable,
            notes=tuple(dict.fromkeys(all_notes)),
        ))

    return AblationReport(
        pool_id=pool.pool_id,
        pool_digest=pool.digest,
        criterion_digest=registration.registered_digest,
        budget=budget,
        settings=base,
        full_system_ids=tuple(full_ids),
        full_system_endpoint=full_endpoint,
        results=tuple(results),
        full_system_notes=tuple(
            list(full_notes)
            + ([f"full-system selection short: {full_shortfall}"]
               if full_shortfall else [])),
    )
