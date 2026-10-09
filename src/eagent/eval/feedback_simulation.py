"""Does feeding measured labels back into selection change what a round finds?

THE QUESTION, AND THE ONLY DATA THAT CAN ASK IT HERE
====================================================
The protocol's central loop is: select, measure, update, select again. The
claim "the agent learns from experiments" is a claim about the *update*, and it
is checkable only by an experiment that holds everything else fixed and turns
the update on and off. No ketoreductase campaign data exists in this
repository, so this simulation runs on the one dataset that does: the SDR
substrate-class deposit, whose labels are annotation-derived
(:mod:`eagent.eval.retrospective`). "Measuring" a candidate reveals its
annotated class. What the simulation can show is therefore whether
**similarity-guided, feedback-updated selection finds annotated-class members
faster than random, faster than selection with no feedback, and faster than
pure exploration**. It cannot show the same of a real assay, and every report
says so.

WHAT IS HELD FIXED, SO A DIFFERENCE IS A DIFFERENCE IN METHOD
==============================================================
Per replicate: the same pool, the same initial labelled set, the same label
oracle, the same number of rounds and the same slots per round for every
strategy. Randomness is derived from ``(replicate seed, strategy name)`` so
adding a strategy never changes another's draws. The initial labelled set is
stratified (at least three of each class) because a method cannot start from
no positives, and that requirement is stated rather than hidden.

THE STRATEGIES
==============
``random``
    the floor.
``nearest_neighbour[feedback]`` / ``nearest_neighbour[frozen]``
    similarity to known positives minus similarity to known negatives, by the
    same k-mer cosine the model uses. The homology-guided baseline; *frozen*
    uses only the initial labels and never learns from the rounds.
``model[feedback]`` / ``model[frozen]``
    the spectrum-kernel model, refitted each round on everything measured so
    far, or fitted once.
``model+explore[feedback]``
    :class:`~eagent.science.acquisition.AcquisitionPolicy`: most slots to the
    model's top scores, the rest to the candidates least similar to anything
    measured.
``explore_only``
    farthest-first by novelty, the exploration-only floor.

REDUNDANCY IS A LEVER, NOT A DETAIL
===================================
A pool with clusters of near-identical sequences rewards any method that
reads similarity: one revealed hit makes its neighbours certain. That is also
what happens in a real campaign built from a homology search, so the full pool
is reported -- and so is a **non-redundant** pool (one sequence per cluster at
the stated identity), where that advantage is mostly gone. Both are shown
because the gap between them is the finding.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..errors import EAgentError
from ..provenance import sha256_obj
from ..science.acquisition import AcquisitionPolicy, select_by_acquisition
from ..science.enzyme_substrate import (
    EnzymeSubstrateModel, Example, KernelCache,
)

__all__ = [
    "SimulationError",
    "SimulationProtocol",
    "StrategyResult",
    "TargetSimulation",
    "SimulationReport",
    "STRATEGIES",
    "simulate_target",
    "run_simulation",
    "non_redundant",
    "FeedbackSuite",
    "run_sdr_feedback",
]

CLAIM = (
    "This simulation reveals ANNOTATION-DERIVED labels as if they were assay "
    "results. It shows how selection strategies compare at recovering an "
    "annotated class; it is not evidence about a real assay, the agent's gates, "
    "or ketoreductase activity.")


class SimulationError(EAgentError):
    """The simulation's inputs cannot support the experiment it was asked for."""


@dataclass(frozen=True)
class SimulationProtocol:
    rounds: int = 5
    per_round: int = 8
    initial_labelled: int = 24
    min_initial_per_class: int = 3
    n_replicates: int = 20
    explore_slots_per_round: int = 2
    seed: int = 0
    ridge: float = 0.1
    ks: tuple[int, ...] = (2, 3)
    n_boot: int = 500

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__, ks=list(self.ks))


# --------------------------------------------------------------------------
# state shared by every strategy in one replicate
# --------------------------------------------------------------------------

@dataclass
class _State:
    ids: list[str]
    seqs: dict[str, str]
    labels: dict[str, bool]               # the oracle; strategies never read it
    labelled: list[str]                   # revealed so far, in order
    initial: list[str]
    kernel: KernelCache
    protocol: SimulationProtocol
    target: str = "t"

    def unlabelled(self) -> list[str]:
        known = set(self.labelled)
        return [i for i in self.ids if i not in known]

    def examples(self, ids: Sequence[str]) -> list[Example]:
        return [Example(i, self.seqs[i], self.target, self.labels[i],
                        "annotation") for i in ids]


def _sim(state: _State, a: str, b: str) -> float:
    return state.kernel.similarity(state.seqs[a], state.seqs[b])


def _neighbour_scores(state: _State, known: Sequence[str]) -> dict[str, float | None]:
    pos = [i for i in known if state.labels[i]]
    neg = [i for i in known if not state.labels[i]]
    out: dict[str, float | None] = {}
    for cid in state.unlabelled():
        sp = max((_sim(state, cid, p) for p in pos), default=0.0)
        sn = max((_sim(state, cid, n) for n in neg), default=0.0)
        out[cid] = sp - sn
    return out


def _model_scores(state: _State, known: Sequence[str]) -> dict[str, float | None]:
    model = EnzymeSubstrateModel(
        state.examples(known), ridge=state.protocol.ridge,
        ks=state.protocol.ks, kernel=state.kernel, factor_cache={},
        min_calibration_per_class=10 ** 9)          # ranks only; no probabilities
    return {cid: model.predict(state.seqs[cid], state.target).score
            for cid in state.unlabelled()}


def _novelty(state: _State, known: Sequence[str]) -> dict[str, float | None]:
    return {cid: 1.0 - max((_sim(state, cid, k) for k in known), default=0.0)
            for cid in state.unlabelled()}


def _top(scores: Mapping[str, float | None], k: int) -> list[str]:
    ranked = sorted((c for c, v in scores.items() if v is not None),
                    key=lambda c: (-float(scores[c]), c))   # type: ignore[arg-type]
    return ranked[:k]


Selector = Callable[[_State, int, random.Random, Sequence[str]], list[str]]


def _random(state: _State, k: int, rng: random.Random,
            known: Sequence[str]) -> list[str]:
    pool = sorted(state.unlabelled())
    rng.shuffle(pool)
    return pool[:k]


def _nn(state: _State, k: int, rng: random.Random,
        known: Sequence[str]) -> list[str]:
    return _top(_neighbour_scores(state, known), k)


def _model(state: _State, k: int, rng: random.Random,
           known: Sequence[str]) -> list[str]:
    return _top(_model_scores(state, known), k)


def _model_explore(state: _State, k: int, rng: random.Random,
                   known: Sequence[str]) -> list[str]:
    n_explore = min(state.protocol.explore_slots_per_round, k)
    picks = select_by_acquisition(
        state.unlabelled(), _model_scores(state, known), _novelty(state, known),
        AcquisitionPolicy(n_exploit=k - n_explore, n_explore=n_explore))
    return [p.candidate_id for p in picks]


def _explore_only(state: _State, k: int, rng: random.Random,
                  known: Sequence[str]) -> list[str]:
    return _top(_novelty(state, known), k)


@dataclass(frozen=True)
class _Strategy:
    name: str
    select: Selector
    feedback: bool
    description: str


STRATEGIES: tuple[_Strategy, ...] = (
    _Strategy("random", _random, False, "uniform draw from the unlabelled pool"),
    _Strategy("nearest_neighbour[feedback]", _nn, True,
              "similarity to known positives minus negatives, relearned each round"),
    _Strategy("nearest_neighbour[frozen]", _nn, False,
              "the same, using only the initial labels"),
    _Strategy("model[feedback]", _model, True,
              "spectrum-kernel model refitted on everything measured"),
    _Strategy("model[frozen]", _model, False,
              "the model fitted once on the initial labels"),
    _Strategy("model+explore[feedback]", _model_explore, True,
              "model top scores plus novelty-chosen exploration slots"),
    _Strategy("explore_only", _explore_only, True,
              "farthest from everything measured"),
)


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass
class StrategyResult:
    name: str
    description: str
    feedback: bool
    #: hits found per replicate after each round, cumulative (selected only)
    cumulative_hits: list[list[int]]

    def final(self) -> list[int]:
        return [c[-1] for c in self.cumulative_hits]


@dataclass
class TargetSimulation:
    target: str
    pool_size: int
    prevalence: float
    n_replicates: int
    results: dict[str, StrategyResult]
    summary: dict[str, dict[str, Any]] = field(default_factory=dict)
    paired: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class SimulationReport:
    protocol: SimulationProtocol
    claim: str
    pool_kind: str
    targets: list[TargetSimulation]
    skipped: dict[str, str]
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol.to_dict(), "claim": self.claim,
            "pool_kind": self.pool_kind, "skipped": self.skipped,
            "targets": [{
                "target": t.target, "pool_size": t.pool_size,
                "prevalence": t.prevalence, "n_replicates": t.n_replicates,
                "summary": t.summary, "paired": t.paired} for t in self.targets],
            "digest": self.digest,
        }

    def render(self) -> str:
        p = self.protocol
        budget = p.rounds * p.per_round
        lines = [
            f"FEEDBACK SIMULATION ({self.pool_kind} pool)", self.claim, "",
            f"{p.rounds} rounds x {p.per_round} slots = {budget} selections per "
            f"replicate, {p.n_replicates} replicates, initial labelled "
            f"{p.initial_labelled}, seed {p.seed}", ""]
        for t in self.targets:
            lines.append(f"{t.target}  (pool {t.pool_size}, prevalence "
                         f"{t.prevalence:.2f}; random expects "
                         f"{budget * t.prevalence:.1f} hits)")
            for name, s in t.summary.items():
                lines.append(
                    f"  {name:<30} hits {s['mean_hits']:5.1f} "
                    f"[{s['ci_low']:.1f},{s['ci_high']:.1f}]  "
                    f"enrichment x{s['enrichment_vs_random']:.2f}")
            for name, d in t.paired.items():
                lines.append(
                    f"  {name:<46} {d['mean_difference']:+.1f} hits "
                    f"[{d['ci_low']:+.1f},{d['ci_high']:+.1f}]"
                    + ("  *" if d["interval_excludes_zero"] else ""))
            lines.append("")
        lines.append("* paired bootstrap interval over replicates excludes zero")
        for t, why in sorted(self.skipped.items()):
            lines.append(f"skipped {t}: {why}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------

def _initial_set(ids: Sequence[str], labels: Mapping[str, bool],
                 protocol: SimulationProtocol, rng: random.Random) -> list[str]:
    pos = sorted(i for i in ids if labels[i])
    neg = sorted(i for i in ids if not labels[i])
    prevalence = len(pos) / len(ids)
    n_pos = max(protocol.min_initial_per_class,
                round(prevalence * protocol.initial_labelled))
    n_neg = max(protocol.min_initial_per_class,
                protocol.initial_labelled - n_pos)
    if len(pos) < n_pos + protocol.per_round or len(neg) < n_neg:
        raise SimulationError(
            f"{len(pos)} positive / {len(neg)} negative cannot supply an "
            f"initial set of {n_pos}+{n_neg} and still leave positives to find")
    rng.shuffle(pos)
    rng.shuffle(neg)
    return sorted(pos[:n_pos] + neg[:n_neg])


def _bootstrap_mean(values: Sequence[float], n_boot: int, seed: int
                    ) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[min(n_boot - 1, int(0.975 * n_boot))]


def simulate_target(
    examples: Sequence[Example], protocol: SimulationProtocol, *,
    kernel: KernelCache | None = None, strategies: Sequence[_Strategy] | None = None,
) -> TargetSimulation:
    """Run every strategy over identical replicates for one target."""
    rows = list(examples)
    if not rows:
        raise SimulationError("no examples")
    target = rows[0].target
    kernel = kernel or KernelCache(protocol.ks)
    ids = sorted({e.example_id for e in rows})
    seqs = {e.example_id: e.sequence for e in rows}
    labels = {e.example_id: e.label for e in rows}
    strategies = list(strategies or STRATEGIES)
    prevalence = sum(labels.values()) / len(labels)
    results = {s.name: StrategyResult(s.name, s.description, s.feedback, [])
               for s in strategies}

    for rep in range(protocol.n_replicates):
        rep_seed = protocol.seed * 1_000_003 + rep
        initial = _initial_set(ids, labels, protocol, random.Random(rep_seed))
        for strat in strategies:
            rng = random.Random(sha256_obj([rep_seed, strat.name])[:16])
            state = _State(ids=ids, seqs=seqs, labels=labels,
                           labelled=list(initial), initial=list(initial),
                           kernel=kernel, protocol=protocol, target=target)
            cumulative: list[int] = []
            hits = 0
            for _ in range(protocol.rounds):
                known = state.labelled if strat.feedback else state.initial
                picks = strat.select(state, protocol.per_round, rng, known)
                picks = [p for p in picks if p not in set(state.labelled)]
                hits += sum(1 for p in picks if labels[p])
                state.labelled.extend(picks)           # revealed either way
                cumulative.append(hits)
            results[strat.name].cumulative_hits.append(cumulative)

    sim = TargetSimulation(target=target, pool_size=len(ids),
                           prevalence=prevalence,
                           n_replicates=protocol.n_replicates, results=results)
    random_final = results["random"].final() if "random" in results else None
    random_mean = (sum(random_final) / len(random_final)
                   if random_final else None)
    for name, res in results.items():
        final = res.final()
        mean = sum(final) / len(final)
        lo, hi = _bootstrap_mean(final, protocol.n_boot, protocol.seed + 1)
        sim.summary[name] = {
            "mean_hits": mean, "ci_low": lo, "ci_high": hi,
            "enrichment_vs_random": (mean / random_mean
                                     if random_mean else float("nan"))}

    def paired(a: str, b: str) -> None:
        if a not in results or b not in results:
            return
        diffs = [x - y for x, y in zip(results[a].final(), results[b].final())]
        mean = sum(diffs) / len(diffs)
        lo, hi = _bootstrap_mean(diffs, protocol.n_boot, protocol.seed + 2)
        sim.paired[f"{a} - {b}"] = {
            "mean_difference": mean, "ci_low": lo, "ci_high": hi,
            "interval_excludes_zero": lo > 0 or hi < 0}

    paired("model[feedback]", "random")
    paired("nearest_neighbour[feedback]", "random")
    paired("model[feedback]", "model[frozen]")
    paired("nearest_neighbour[feedback]", "nearest_neighbour[frozen]")
    paired("model[feedback]", "nearest_neighbour[feedback]")
    paired("model+explore[feedback]", "model[feedback]")
    paired("model+explore[feedback]", "explore_only")
    return sim


def run_simulation(
    examples: Sequence[Example], protocol: SimulationProtocol | None = None, *,
    pool_kind: str = "full", min_per_class: int = 15,
    kernel: KernelCache | None = None,
) -> SimulationReport:
    """Simulate every target with enough of both classes."""
    protocol = protocol or SimulationProtocol()
    kernel = kernel or KernelCache(protocol.ks)
    by_target: dict[str, list[Example]] = {}
    for e in examples:
        by_target.setdefault(e.target, []).append(e)
    targets: list[TargetSimulation] = []
    skipped: dict[str, str] = {}
    for target, rows in sorted(by_target.items()):
        n_pos = sum(1 for e in rows if e.label)
        n_neg = len(rows) - n_pos
        if n_pos < min_per_class or n_neg < min_per_class:
            skipped[target] = (f"{n_pos} positive / {n_neg} negative; "
                               f"{min_per_class} of each are required")
            continue
        try:
            targets.append(simulate_target(rows, protocol, kernel=kernel))
        except SimulationError as exc:
            skipped[target] = str(exc)
    report = SimulationReport(protocol=protocol, claim=CLAIM, pool_kind=pool_kind,
                              targets=targets, skipped=skipped)
    report.digest = sha256_obj({k: v for k, v in report.to_dict().items()
                                if k != "digest"})
    return report


def non_redundant(examples: Sequence[Example], assignment: Mapping[str, str]
                  ) -> list[Example]:
    """One sequence per cluster: the first by id, the same one for every target.

    Choosing the representative globally rather than per target keeps the pool
    the same set of enzymes whichever class is being searched for.
    """
    chosen: dict[str, str] = {}
    for ident in sorted({e.example_id for e in examples}):
        chosen.setdefault(assignment.get(ident, ident), ident)
    keep = set(chosen.values())
    return [e for e in examples if e.example_id in keep]


@dataclass
class FeedbackSuite:
    full: SimulationReport
    non_redundant: SimulationReport
    clustering: Mapping[str, Any]

    def render(self) -> str:
        return "\n\n".join([
            self.full.render(), self.non_redundant.render(),
            f"redundancy removed by one-per-cluster at identity "
            f"{self.clustering.get('identity_threshold')} "
            f"({self.clustering.get('method')}); the gap between the two "
            f"pools is how much of any similarity-guided method's advantage "
            f"is the pool's redundancy"])


def run_sdr_feedback(data_dir, protocol: SimulationProtocol | None = None,
                     *, identity_threshold: float = 0.4,
                     targets: Sequence[str] | None = None) -> FeedbackSuite:
    """Load the SDR deposit and run the simulation on the full and non-redundant pools."""
    from ..tools.mine_sequences import cluster_sequences
    from .retrospective import load_sdr, sdr_examples

    protocol = protocol or SimulationProtocol()
    examples = sdr_examples(load_sdr(data_dir))
    if targets is not None:
        examples = [e for e in examples if e.target in set(targets)]
    items = sorted({(e.example_id, e.sequence) for e in examples})
    clustering = cluster_sequences(items, identity_threshold)
    note = {"method": clustering.method,
            "identity_threshold": clustering.identity_threshold,
            "fallback_reason": clustering.fallback_reason}
    kernel = KernelCache(protocol.ks)
    full = run_simulation(examples, protocol, pool_kind="full", kernel=kernel)
    reduced = run_simulation(
        non_redundant(examples, dict(clustering.assignment)), protocol,
        pool_kind="non-redundant", kernel=kernel)
    return FeedbackSuite(full, reduced, note)
