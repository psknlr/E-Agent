"""Tests for the acquisition policy, the ties-only tiebreak, and the feedback simulation.

The simulation is where a claim about learning from experiments can be checked
or faked, so most of these tests are about fairness: strategies sharing initial
sets, no strategy reading a label it has not been shown, and adding a strategy
not changing another's draws.
"""

from __future__ import annotations

import random
import unittest
from typing import Any

from eagent.eval.feedback_simulation import (
    CLAIM, STRATEGIES, SimulationError, SimulationProtocol, _Strategy,
    non_redundant, run_simulation, simulate_target,
)
from eagent.schemas import ConfidenceLevel
from eagent.science.acquisition import AcquisitionPolicy, select_by_acquisition
from eagent.science.diversity import compose_batch
from eagent.science.enzyme_substrate import STANDARD_AA, Example, KernelCache
from eagent.science.scorecard import lexicographic_rank
from eagent.schemas import BatchRole, Budget

from test_scorecard import candidate, dim


# ==========================================================================
# acquisition
# ==========================================================================

class TestAcquisition(unittest.TestCase):

    POOL = ["a", "b", "c", "d", "e", "f"]

    def test_exploit_takes_the_highest_scores_then_explore_takes_the_most_novel(self) -> None:
        scores = {"a": 0.9, "b": 0.8, "c": 0.1, "d": 0.2, "e": 0.3, "f": 0.0}
        novelty = {"a": 0.1, "b": 0.1, "c": 0.9, "d": 0.5, "e": 0.2, "f": 0.7}
        picks = select_by_acquisition(self.POOL, scores, novelty,
                                      AcquisitionPolicy(2, 2))
        self.assertEqual([(p.candidate_id, p.role) for p in picks],
                         [("a", "exploit"), ("b", "exploit"),
                          ("c", "explore"), ("f", "explore")])

    def test_an_unscored_candidate_is_never_exploited_but_may_be_explored(self) -> None:
        scores = {"a": 0.5, "b": None, "c": 0.4, "d": None, "e": None, "f": None}
        novelty = {c: 1.0 if c == "b" else 0.1 for c in self.POOL}
        picks = select_by_acquisition(self.POOL, scores, novelty,
                                      AcquisitionPolicy(5, 1))
        exploit = [p.candidate_id for p in picks if p.role == "exploit"]
        explore = [p.candidate_id for p in picks if p.role == "explore"]
        self.assertEqual(exploit, ["a", "c"])           # only the two it scored
        self.assertEqual(explore, ["b"])

    def test_the_policy_never_pads(self) -> None:
        picks = select_by_acquisition(["a"], {"a": 1.0}, {"a": 0.5},
                                      AcquisitionPolicy(3, 3))
        self.assertEqual([p.candidate_id for p in picks], ["a"])

    def test_ties_break_on_the_id_so_the_order_is_deterministic(self) -> None:
        scores = {c: 0.5 for c in self.POOL}
        picks = select_by_acquisition(self.POOL, scores, scores,
                                      AcquisitionPolicy(3, 0))
        self.assertEqual([p.candidate_id for p in picks], ["a", "b", "c"])

    def test_exploration_does_not_read_the_models_opinion(self) -> None:
        scores = {c: 0.0 for c in self.POOL}
        novelty = {c: float(i) for i, c in enumerate(self.POOL)}
        picks = select_by_acquisition(self.POOL, scores, novelty,
                                      AcquisitionPolicy(0, 2))
        self.assertEqual([p.candidate_id for p in picks], ["f", "e"])
        self.assertIn("without reading the model's opinion", picks[0].reason)

    def test_uncertainty_exploration_needs_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            select_by_acquisition(self.POOL, {}, {}, AcquisitionPolicy(
                0, 1, explore_by="uncertainty"))

    def test_uncertainty_exploration_picks_the_closest_to_one_half(self) -> None:
        probs = {"a": 0.95, "b": 0.52, "c": 0.1, "d": 0.45, "e": None, "f": 0.5}
        picks = select_by_acquisition(
            self.POOL, {}, {}, AcquisitionPolicy(0, 2, explore_by="uncertainty"),
            probabilities=probs)
        self.assertEqual([p.candidate_id for p in picks], ["f", "b"])

    def test_bad_policies_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            AcquisitionPolicy(-1, 0)
        with self.assertRaises(ValueError):
            AcquisitionPolicy(1, 1, explore_by="vibes")


# ==========================================================================
# the ties-only tiebreak
# ==========================================================================

def by_id(values):
    """A tiebreak is called with the candidate; these tests key on its id."""
    return lambda cand: values.get(cand.candidate_id)


class TestTiebreak(unittest.TestCase):

    def pool(self):
        strong = dim("functional_literature_evidence", ConfidenceLevel.STRONG)
        weak = dim("functional_literature_evidence", ConfidenceLevel.WEAK)
        return [
            candidate("c1", dims={"x": strong}),
            candidate("c2", dims={"x": strong}),
            candidate("c3", dims={"x": strong}),
            candidate("c4", dims={"x": weak}),
        ]

    def order(self, tiebreak=None):
        return [c.candidate_id for c in lexicographic_rank(
            self.pool(), ["functional_literature_evidence"], tiebreak=tiebreak)]

    def test_without_a_tiebreak_ties_fall_back_to_the_id(self) -> None:
        self.assertEqual(self.order(), ["c1", "c2", "c3", "c4"])

    def test_a_tiebreak_reorders_ties(self) -> None:
        values = {"c1": 0.1, "c2": 0.9, "c3": 0.5, "c4": 0.0}
        self.assertEqual(self.order(by_id(values)), ["c2", "c3", "c1", "c4"])

    def test_a_tiebreak_cannot_lift_a_candidate_over_stronger_evidence(self) -> None:
        values = {"c1": 0.0, "c2": 0.0, "c3": 0.0, "c4": 1.0}   # weak one adored
        self.assertEqual(self.order(by_id(values))[-1], "c4")

    def test_none_sorts_after_any_value_among_ties(self) -> None:
        values = {"c1": None, "c2": 0.0, "c3": None}
        self.assertEqual(self.order(by_id(values))[:3], ["c2", "c1", "c3"])

    def test_compose_batch_passes_it_through_and_says_so(self) -> None:
        strong = dim("functional_literature_evidence", ConfidenceLevel.STRONG)
        pool = [candidate(f"c{i}", dims={"x": strong}) for i in range(6)]
        budget = Budget(new_constructs_round_1=2, detailed_complex_target=6)
        values = {"c0": 0.1, "c1": 0.2, "c2": 0.3, "c3": 0.4, "c4": 0.5, "c5": 0.9}
        plain = compose_batch(
            pool, budget, role_targets={BatchRole.HIGH_EVIDENCE: 2},
            order_of_dimensions=["functional_literature_evidence"])
        tied = compose_batch(
            pool, budget, role_targets={BatchRole.HIGH_EVIDENCE: 2},
            order_of_dimensions=["functional_literature_evidence"],
            rank_tiebreak=by_id(values))
        self.assertEqual([m.candidate_id for m in plain.members], ["c0", "c1"])
        self.assertEqual([m.candidate_id for m in tied.members], ["c5", "c4"])
        self.assertIn("tie-break", tied.members[0].selection_reason)
        self.assertNotIn("tie-break", plain.members[0].selection_reason)


# ==========================================================================
# the simulation
# ==========================================================================

def families(n_families: int = 10, per: int = 6, n_positive_families: int = 3,
             seed: int = 3) -> list[Example]:
    """Families of near-duplicates; whole families are positive or negative."""
    rng = random.Random(seed)
    out: list[Example] = []
    for f in range(n_families):
        base = "".join(rng.choice(STANDARD_AA) for _ in range(180))
        for c in range(per):
            seq = list(base)
            for _ in range(2):
                seq[rng.randrange(len(seq))] = rng.choice(STANDARD_AA)
            out.append(Example(f"f{f}_{c}", "".join(seq), "t",
                               f < n_positive_families, "annotation"))
    return out


PROTO = SimulationProtocol(rounds=3, per_round=4, initial_labelled=12,
                           n_replicates=6, n_boot=50)


class TestSimulationFairness(unittest.TestCase):

    def test_similarity_guided_feedback_beats_random_on_clustered_labels(self) -> None:
        sim = simulate_target(families(), PROTO)
        diff = sim.paired["model[feedback] - random"]
        self.assertGreater(diff["mean_difference"], 1.0)
        self.assertTrue(diff["interval_excludes_zero"])

    def test_random_is_unchanged_by_the_presence_of_other_strategies(self) -> None:
        alone = simulate_target(families(), PROTO, strategies=[STRATEGIES[0]])
        together = simulate_target(families(), PROTO)
        self.assertEqual(alone.results["random"].cumulative_hits,
                         together.results["random"].cumulative_hits)

    def test_the_run_is_deterministic_and_the_seed_matters(self) -> None:
        a = simulate_target(families(), PROTO)
        b = simulate_target(families(), PROTO)
        c = simulate_target(families(), SimulationProtocol(
            **{**PROTO.__dict__, "seed": 5}))
        self.assertEqual(a.results["random"].cumulative_hits,
                         b.results["random"].cumulative_hits)
        self.assertNotEqual(a.results["random"].cumulative_hits,
                            c.results["random"].cumulative_hits)

    def test_no_strategy_reads_a_label_it_has_not_been_shown(self) -> None:
        violations: list[tuple[str, str]] = []

        class Spy(dict):
            def __init__(self, base, allowed, who):
                super().__init__(base)
                self.allowed, self.who = set(allowed), who

            def __getitem__(self, key):
                if key not in self.allowed:
                    violations.append((self.who, key))
                return super().__getitem__(key)

            def get(self, key, default=None):
                if key not in self.allowed:
                    violations.append((self.who, key))
                return super().get(key, default)

        def wrap(strategy: _Strategy) -> _Strategy:
            def select(state, k, rng, known):
                real = state.labels
                state.labels = Spy(real, known, strategy.name)
                try:
                    return strategy.select(state, k, rng, known)
                finally:
                    state.labels = real
            return _Strategy(strategy.name, select, strategy.feedback,
                             strategy.description)

        simulate_target(families(), PROTO, strategies=[wrap(s) for s in STRATEGIES])
        self.assertEqual(violations, [])

    def test_frozen_strategies_never_learn_from_the_rounds(self) -> None:
        seen: list[int] = []

        def select(state, k, rng, known):
            seen.append(len(known))
            return sorted(state.unlabelled())[:k]
        frozen = _Strategy("probe[frozen]", select, False, "x")
        live = _Strategy("probe[feedback]", select, True, "x")
        simulate_target(families(), SimulationProtocol(
            **{**PROTO.__dict__, "n_replicates": 1}), strategies=[
                _Strategy("random", STRATEGIES[0].select, False, "r"),
                frozen])
        self.assertEqual(set(seen), {12})                  # always the initial set
        seen.clear()
        simulate_target(families(), SimulationProtocol(
            **{**PROTO.__dict__, "n_replicates": 1}), strategies=[
                _Strategy("random", STRATEGIES[0].select, False, "r"), live])
        self.assertEqual(seen, [12, 16, 20])               # grows each round

    def test_a_selection_is_never_repeated_within_a_replicate(self) -> None:
        sim = simulate_target(families(), PROTO)
        for res in sim.results.values():
            for cumulative in res.cumulative_hits:
                self.assertEqual(cumulative, sorted(cumulative))   # monotone
                self.assertLessEqual(cumulative[-1],
                                     PROTO.rounds * PROTO.per_round)

    def test_the_initial_set_is_stratified_so_a_method_can_start(self) -> None:
        sim = simulate_target(families(n_positive_families=2), PROTO)
        self.assertGreater(sim.pool_size, 0)

    def test_a_target_that_cannot_supply_the_initial_set_is_refused(self) -> None:
        rows = families(n_families=4, per=3, n_positive_families=1)
        with self.assertRaises(SimulationError):
            simulate_target(rows, PROTO)

    def test_the_report_carries_the_claim_and_names_skipped_targets(self) -> None:
        rows = families() + [Example(f"r{i}", "ACDEFGHIK" * 20 + str(i % 1) * 0,
                                     "rare", i == 0, "annotation")
                             for i in range(20)]
        report = run_simulation(rows, PROTO, min_per_class=5)
        self.assertEqual(report.claim, CLAIM)
        self.assertIn("rare", report.skipped)
        text = report.render()
        self.assertIn("ANNOTATION-DERIVED", text)
        self.assertIn("random expects", text)

    def test_the_digest_is_stable(self) -> None:
        a = run_simulation(families(), PROTO, min_per_class=5)
        b = run_simulation(families(), PROTO, min_per_class=5)
        self.assertEqual(a.digest, b.digest)


class TestNonRedundantPool(unittest.TestCase):

    def test_one_representative_per_cluster_and_the_same_one_for_every_target(self) -> None:
        rows = families(n_families=4, per=3)
        other = [Example(e.example_id, e.sequence, "u", not e.label, "annotation")
                 for e in rows]
        assignment = {e.example_id: e.example_id.split("_")[0] for e in rows}
        kept = non_redundant(rows + other, assignment)
        ids_t = sorted(e.example_id for e in kept if e.target == "t")
        ids_u = sorted(e.example_id for e in kept if e.target == "u")
        self.assertEqual(ids_t, ["f0_0", "f1_0", "f2_0", "f3_0"])
        self.assertEqual(ids_t, ids_u)

    def test_a_sequence_with_no_cluster_is_its_own(self) -> None:
        rows = families(n_families=2, per=2)
        kept = non_redundant(rows, {})
        self.assertEqual(len(kept), len(rows))


class TestAgainstTheRealDeposit(unittest.TestCase):
    """The recorded result is reproducible from the fixture, at small scale."""

    def test_feedback_beats_random_on_the_rarest_class(self) -> None:
        from pathlib import Path
        from eagent.eval.retrospective import load_sdr, sdr_examples
        data = load_sdr(Path(__file__).resolve().parent / "fixtures" / "sdr")
        rows = [e for e in sdr_examples(data)
                if e.target == "sdr:substrate_cluster_3"]
        sim = simulate_target(rows, SimulationProtocol(
            n_replicates=4, n_boot=50), kernel=KernelCache())
        self.assertGreater(sim.paired["model[feedback] - random"]["mean_difference"],
                           5.0)
        self.assertAlmostEqual(sim.prevalence, 24 / 309, places=3)


if __name__ == "__main__":
    unittest.main()
