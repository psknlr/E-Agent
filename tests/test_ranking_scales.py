"""Regression tests: two ways the selector spent a slot on nothing.

Both from the follow-up audit, reproduced before being fixed.

A dimension reduces to a number for comparison. With a scalar it reduces to
that scalar; without one it falls back to its ordinal evidence rank, 0 to 4.
Those were put on one axis, so a candidate with no measurement and a STRONG
level (4.0) beat one measured at ``robustness_G = 0.9``. The measured
candidate was dominated and dropped off the Pareto front with its measurement
intact and unused.

Coverage was local to one call of the greedy selector, and a batch is filled
in stages: high evidence, then uncertainty probes, then diversity. The
diversity stage started from "nothing covered", so it could spend its slot on
a near duplicate of a candidate the first stage had already put in the batch.
The slot bought no coverage and the plan recorded it as having been spent on
diversity.
"""

from __future__ import annotations

import unittest

from eagent.schemas.candidate import ConfidenceLevel
from eagent.science.diversity import greedy_submodular_select
from eagent.science.scorecard import (
    MEASURED_SCALE, NO_SCALE, ORDINAL_SCALE, comparable, dominates,
    lexicographic_rank, objective_comparable, pareto_front,
)

from test_scorecard import candidate, dim

GEOMETRY = [("catalytic_geometry", "higher_is_better")]


def measured(cid: str, value: float,
             level: ConfidenceLevel = ConfidenceLevel.MODERATE):
    return candidate(cid, dims={"g": dim("catalytic_geometry", level, value)})


def ordinal(cid: str, level: ConfidenceLevel = ConfidenceLevel.STRONG):
    return candidate(cid, dims={"g": dim("catalytic_geometry", level, None)})


class ScalesAreNotInterchangeable(unittest.TestCase):
    def test_the_two_reductions_say_which_scale_they_are_on(self) -> None:
        self.assertEqual(comparable(measured("a", 0.9).dimension(
            "catalytic_geometry")).scale, MEASURED_SCALE)
        self.assertEqual(comparable(ordinal("b").dimension(
            "catalytic_geometry")).scale, ORDINAL_SCALE)

    def test_an_incomparable_dimension_is_on_no_scale(self) -> None:
        contradictory = candidate("c", dims={
            "g": dim("catalytic_geometry", ConfidenceLevel.CONTRADICTORY, 0.9)})
        reduced = comparable(contradictory.dimension("catalytic_geometry"))
        self.assertIsNone(reduced.value)
        self.assertEqual(reduced.scale, NO_SCALE)

    def test_the_premise_the_numbers_really_do_cross(self) -> None:
        """Guards against the test passing because the values happen not to."""
        a = comparable(measured("a", 0.9).dimension("catalytic_geometry"))
        b = comparable(ordinal("b").dimension("catalytic_geometry"))
        assert a.value is not None and b.value is not None
        self.assertGreater(b.value, a.value,
                           "the ordinal must outrank the measured value, or "
                           "this fixture does not reproduce the finding")

    def test_an_unmeasured_candidate_no_longer_dominates_a_measured_one(self) -> None:
        self.assertFalse(dominates(ordinal("b"), measured("a", 0.9), GEOMETRY))

    def test_nor_the_other_way_round(self) -> None:
        self.assertFalse(dominates(measured("a", 0.9), ordinal("b"), GEOMETRY))

    def test_the_measured_candidate_stays_on_the_front(self) -> None:
        front = pareto_front([measured("a", 0.9), ordinal("b")], GEOMETRY)
        self.assertEqual([c.candidate_id for c in front], ["a", "b"])

    def test_two_measured_candidates_still_compare(self) -> None:
        self.assertTrue(dominates(measured("a", 0.9), measured("b", 0.2),
                                  GEOMETRY))
        front = pareto_front([measured("a", 0.9), measured("b", 0.2)], GEOMETRY)
        self.assertEqual([c.candidate_id for c in front], ["a"])

    def test_two_ordinal_candidates_still_compare(self) -> None:
        strong = ordinal("a", ConfidenceLevel.STRONG)
        weak = ordinal("b", ConfidenceLevel.WEAK)
        self.assertTrue(dominates(strong, weak, GEOMETRY))

    def test_the_lexicographic_rank_orders_on_the_level_first(self) -> None:
        """Which is one scale for every candidate, measured or not."""
        order = lexicographic_rank(
            [measured("measured", 0.9, ConfidenceLevel.MODERATE),
             ordinal("ordinal", ConfidenceLevel.STRONG)],
            ["catalytic_geometry"], require_gates=False)
        self.assertEqual([c.candidate_id for c in order],
                         ["ordinal", "measured"])

    def test_the_scalar_separates_candidates_of_one_level(self) -> None:
        order = lexicographic_rank(
            [measured("low", 0.2, ConfidenceLevel.MODERATE),
             measured("high", 0.9, ConfidenceLevel.MODERATE)],
            ["catalytic_geometry"], require_gates=False)
        self.assertEqual([c.candidate_id for c in order], ["high", "low"])

    def test_objective_comparable_keeps_the_scale(self) -> None:
        reduced = objective_comparable(measured("a", 0.9),
                                       "catalytic_geometry", "higher_is_better")
        self.assertEqual(reduced.scale, MEASURED_SCALE)
        self.assertEqual(reduced.value, 0.9)


class CoverageIsAPropertyOfTheBatch(unittest.TestCase):
    """A later stage must know what the earlier ones already covered."""

    ITEMS = ["a", "a2", "b", "c"]
    #: a and a2 are near duplicates; everything else is distinct.
    UTILITIES = {"a": 1.0, "a2": 0.99, "b": 0.5, "c": 0.4}

    @staticmethod
    def distance(x, y) -> float:
        if x == y:
            return 0.0
        return 0.05 if {x, y} == {"a", "a2"} else 1.0

    def select(self, pool, k, already=()):
        return greedy_submodular_select(pool, k, self.UTILITIES, self.distance,
                                        1.0, already_selected=already)

    def test_the_near_duplicate_wins_on_utility_alone(self) -> None:
        """The premise: without a baseline, a2 is what gets picked."""
        self.assertEqual(self.select(["a2", "b", "c"], 1), ["a2"])

    def test_a_seeded_baseline_passes_over_the_near_duplicate(self) -> None:
        self.assertEqual(self.select(["a2", "b", "c"], 1, already=["a"]), ["b"])

    def test_the_baseline_does_not_change_an_unrelated_pick(self) -> None:
        self.assertEqual(self.select(["b", "c"], 1, already=["a"]), ["b"])

    def test_an_empty_baseline_is_the_old_behaviour(self) -> None:
        self.assertEqual(self.select(["a2", "b", "c"], 1, already=[]), ["a2"])

    def test_selecting_more_than_one_still_covers(self) -> None:
        self.assertEqual(self.select(["a2", "b", "c"], 2, already=["a"]),
                         ["b", "c"])

    def test_a_baseline_item_outside_the_pool_is_fine(self) -> None:
        """The already-selected items are not candidates for re-selection."""
        picked = self.select(["b", "c"], 2, already=["a", "a2"])
        self.assertEqual(sorted(picked), ["b", "c"])


class CoverageAcrossRoles(unittest.TestCase):
    """The composed batch must not spend a diversity slot on a duplicate."""

    def setUp(self) -> None:
        from eagent.schemas import BatchRole, Budget
        self.BatchRole = BatchRole
        self.budget = Budget(new_constructs_round_1=2,
                             detailed_complex_target=300)
        self.targets = {BatchRole.HIGH_EVIDENCE: 1, BatchRole.DIVERSITY: 1,
                        BatchRole.UNCERTAINTY_PROBE: 0}

    def pool(self):
        """Three candidates: the twin of the best one, and a distinct third.

        ``twin`` shares the top candidate's pocket exactly and outranks
        ``other`` on evidence, so an uninformed diversity stage takes it.
        """
        from test_diversity import candidate
        return [
            candidate("best", pocket={"s": "R1", "t": "Q1"}, evidence_value=3.0),
            candidate("twin", pocket={"s": "R1", "t": "Q1"}, evidence_value=2.0),
            candidate("other", pocket={"s": "W9", "t": "K9"}, evidence_value=1.0),
        ]

    def test_the_diversity_slot_does_not_go_to_the_twin(self) -> None:
        from eagent.science.diversity import compose_batch
        plan = compose_batch(self.pool(), self.budget, self.targets, None)
        roles = {m.candidate_id: m.role for m in plan.members}
        self.assertEqual(roles.get("best"), self.BatchRole.HIGH_EVIDENCE)
        self.assertNotIn("twin", roles,
                         "the twin covers a pocket the batch already has")
        self.assertEqual(roles.get("other"), self.BatchRole.DIVERSITY)

    def test_the_premise_the_twin_outranks_the_alternative(self) -> None:
        """Without inheritance, utility alone would pick the twin."""
        from eagent.science.diversity import (
            greedy_submodular_select, candidate_distance, rank_utility,
        )
        from eagent.science.scorecard import lexicographic_rank
        pool = self.pool()
        ranked = lexicographic_rank(pool)
        utilities = rank_utility(ranked)
        self.assertGreater(utilities["twin"], utilities["other"])
        rest = [c for c in ranked if c.candidate_id != "best"]
        blind = greedy_submodular_select(rest, 1, utilities, candidate_distance,
                                         1.0)
        self.assertEqual([c.candidate_id for c in blind], ["twin"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
