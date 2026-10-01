"""Tests for :mod:`eagent.science.robustness`.

The cases here are chosen around the failure modes the module exists to
prevent, not around its happy path: zero valid poses rendered as a zero
fraction, a bare 5/5 read as certainty, and a restrained constraint counted
afterwards as if the model had discovered it.
"""

from __future__ import annotations

import unittest

from eagent.errors import CircularEvidenceError
from eagent.schemas import ComplexPose, ConfidenceLevel, GeometryReport
from eagent.science.robustness import (
    DEFAULT_MIN_VALID_POSES,
    CircularityGuard,
    bootstrap_fraction_ci,
    classify_robustness,
    cross_method_agreement,
    cross_method_verdicts,
    pose_robustness,
    wilson_interval,
)


def report(pose_id: str, satisfied: dict[str, bool | None],
           gating_passed: bool | None = None) -> GeometryReport:
    return GeometryReport(pose_id=pose_id, satisfied=dict(satisfied),
                          gating_passed=gating_passed)


class TestPoseRobustness(unittest.TestCase):

    def test_zero_valid_poses_returns_none_not_zero(self) -> None:
        """No valid pose is a modelling failure, never a measured zero."""
        self.assertIsNone(pose_robustness(0, 0))

    def test_plain_fraction(self) -> None:
        self.assertEqual(pose_robustness(3, 12), 0.25)
        self.assertEqual(pose_robustness(0, 12), 0.0)
        self.assertEqual(pose_robustness(12, 12), 1.0)

    def test_negative_counts_raise(self) -> None:
        with self.assertRaises(ValueError):
            pose_robustness(-1, 5)
        with self.assertRaises(ValueError):
            pose_robustness(1, -5)

    def test_more_satisfying_than_valid_raises(self) -> None:
        with self.assertRaises(ValueError):
            pose_robustness(6, 5)


class TestWilsonInterval(unittest.TestCase):

    def test_interval_brackets_the_point_estimate(self) -> None:
        for trials in (1, 2, 5, 10, 37, 200):
            for successes in range(trials + 1):
                lo, hi = wilson_interval(successes, trials)
                p = successes / trials
                self.assertLessEqual(lo, p + 1e-12,
                                     msg=f"{successes}/{trials}: lo {lo} > p {p}")
                self.assertGreaterEqual(hi, p - 1e-12,
                                        msg=f"{successes}/{trials}: hi {hi} < p {p}")
                self.assertGreaterEqual(lo, 0.0)
                self.assertLessEqual(hi, 1.0)

    def test_five_out_of_five_is_not_certainty(self) -> None:
        """The whole point: a perfect small sample must not report +/- 0."""
        lo, hi = wilson_interval(5, 5)
        self.assertEqual(hi, 1.0)
        self.assertLess(lo, 0.8)
        self.assertGreater(lo, 0.0)

    def test_wider_for_smaller_samples(self) -> None:
        narrow = wilson_interval(50, 100)
        wide = wilson_interval(5, 10)
        self.assertLess(narrow[1] - narrow[0], wide[1] - wide[0])

    def test_zero_trials_raise(self) -> None:
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)

    def test_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            wilson_interval(6, 5)
        with self.assertRaises(ValueError):
            wilson_interval(1, 5, z=0.0)


class TestClassifyRobustness(unittest.TestCase):

    def test_no_poses_is_insufficient(self) -> None:
        self.assertIs(classify_robustness(None, 0), ConfidenceLevel.INSUFFICIENT)

    def test_small_sample_is_insufficient_even_when_perfect(self) -> None:
        """1/1 and 0/1 are both noise, and must not rank above each other."""
        n = DEFAULT_MIN_VALID_POSES - 1
        self.assertIs(classify_robustness(1.0, n), ConfidenceLevel.INSUFFICIENT)
        self.assertIs(classify_robustness(0.0, n), ConfidenceLevel.INSUFFICIENT)

    def test_large_unanimous_sample_is_strong(self) -> None:
        self.assertIs(classify_robustness(1.0, 20), ConfidenceLevel.STRONG)

    def test_middling_fraction_is_moderate_or_weak(self) -> None:
        self.assertIs(classify_robustness(0.40, 40), ConfidenceLevel.MODERATE)
        self.assertIs(classify_robustness(0.05, 40), ConfidenceLevel.WEAK)

    def test_level_reads_the_lower_bound_not_the_point_estimate(self) -> None:
        """The same fraction from more poses is stronger evidence, and says so."""
        self.assertIs(classify_robustness(0.6, 5), ConfidenceLevel.MODERATE)
        self.assertIs(classify_robustness(0.6, 200), ConfidenceLevel.STRONG)

    def test_zero_with_a_real_sample_is_weak_not_insufficient(self) -> None:
        self.assertIs(classify_robustness(0.0, 40), ConfidenceLevel.WEAK)

    def test_never_returns_contradictory(self) -> None:
        for n in (0, 3, 5, 40):
            for g in (None, 0.0, 0.5, 1.0):
                if g is None and n:
                    continue
                level = classify_robustness(g, n)
                self.assertIsNot(level, ConfidenceLevel.CONTRADICTORY)

    def test_thresholds_are_parameters(self) -> None:
        strict = classify_robustness(1.0, 6, strong_lower_bound=0.9,
                                     moderate_lower_bound=0.3)
        self.assertIs(strict, ConfidenceLevel.MODERATE)
        lenient = classify_robustness(1.0, 6, strong_lower_bound=0.5,
                                      moderate_lower_bound=0.2)
        self.assertIs(lenient, ConfidenceLevel.STRONG)

    def test_bad_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            classify_robustness(1.5, 10)
        with self.assertRaises(ValueError):
            classify_robustness(0.5, -1)
        with self.assertRaises(ValueError):
            classify_robustness(0.5, 10, min_valid_poses=0)
        with self.assertRaises(ValueError):
            classify_robustness(0.5, 10, strong_lower_bound=0.1,
                                moderate_lower_bound=0.4)


class TestBootstrap(unittest.TestCase):

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(bootstrap_fraction_ci([], 100, 7))

    def test_deterministic_for_a_given_seed(self) -> None:
        flags = [True, True, False, True, False, False, True]
        a = bootstrap_fraction_ci(flags, 500, 11)
        b = bootstrap_fraction_ci(flags, 500, 11)
        self.assertEqual(a, b)

    def test_different_seeds_may_differ_but_stay_sane(self) -> None:
        flags = [True] * 4 + [False] * 6
        lo, hi = bootstrap_fraction_ci(flags, 500, 3)
        self.assertLessEqual(0.0, lo)
        self.assertLessEqual(lo, 0.4)
        self.assertLessEqual(0.4, hi)
        self.assertLessEqual(hi, 1.0)

    def test_unanimous_sample_collapses(self) -> None:
        lo, hi = bootstrap_fraction_ci([True] * 8, 200, 1)
        self.assertEqual((lo, hi), (1.0, 1.0))

    def test_bad_parameters_raise(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap_fraction_ci([True], 0, 1)
        with self.assertRaises(ValueError):
            bootstrap_fraction_ci([True], 10, 1, alpha=0.0)


class TestCircularityGuard(unittest.TestCase):

    def setUp(self) -> None:
        self.pose = ComplexPose(
            pose_id="p1",
            method="template_docking",
            restrained_constraints=["d_hydride", "hbond_tyr"],
        )
        self.evaluated = ["d_hydride", "hbond_tyr", "bd_angle", "hbond_ser"]

    def test_partition(self) -> None:
        guard = CircularityGuard.from_pose(self.pose, self.evaluated)
        circular, independent = guard.partition()
        self.assertEqual(circular, frozenset({"d_hydride", "hbond_tyr"}))
        self.assertEqual(independent, frozenset({"bd_angle", "hbond_ser"}))

    def test_raises_when_every_satisfied_constraint_was_restrained(self) -> None:
        """The central case: a model corroborating only what it was told."""
        guard = CircularityGuard.from_pose(self.pose, self.evaluated)
        rep = report("p1", {"d_hydride": True, "hbond_tyr": True,
                            "bd_angle": False, "hbond_ser": False})
        with self.assertRaises(CircularEvidenceError) as cm:
            guard.raise_if_only_circular(rep)
        self.assertIn("restrained", str(cm.exception))

    def test_does_not_raise_when_an_independent_constraint_is_satisfied(self) -> None:
        guard = CircularityGuard.from_pose(self.pose, self.evaluated)
        rep = report("p1", {"d_hydride": True, "hbond_tyr": True,
                            "bd_angle": True, "hbond_ser": False})
        ev = guard.raise_if_only_circular(rep)
        self.assertEqual(ev.satisfied, ("bd_angle",))
        self.assertEqual(ev.n_independent_satisfied, 1)

    def test_nothing_satisfied_is_a_failed_gate_not_a_circularity(self) -> None:
        guard = CircularityGuard.from_pose(self.pose, self.evaluated)
        rep = report("p1", {"d_hydride": False, "hbond_tyr": False,
                            "bd_angle": False, "hbond_ser": False})
        ev = guard.raise_if_only_circular(rep)
        self.assertEqual(ev.satisfied, ())
        with self.assertRaises(CircularEvidenceError):
            guard.raise_if_only_circular(rep, allow_no_evidence=False)

    def test_independent_evidence_buckets(self) -> None:
        guard = CircularityGuard.from_pose(self.pose, self.evaluated)
        rep = report("p1", {"d_hydride": True, "hbond_tyr": False,
                            "bd_angle": True, "hbond_ser": None,
                            "not_in_template": True})
        ev = guard.independent_evidence(rep)
        self.assertEqual(ev.satisfied, ("bd_angle",))
        self.assertEqual(ev.unsatisfied, ())
        self.assertEqual(ev.unmeasured, ("hbond_ser",))
        self.assertEqual(ev.circular_satisfied, ("d_hydride",))
        self.assertEqual(ev.circular_all, ("d_hydride", "hbond_tyr"))
        self.assertEqual(ev.fraction, 1.0)
        self.assertFalse(ev.is_entirely_circular)

    def test_fraction_is_none_when_nothing_independent_was_measured(self) -> None:
        guard = CircularityGuard(["a", "b"], ["a", "b"])
        ev = guard.independent_evidence(report("p1", {"a": True, "b": True}))
        self.assertIsNone(ev.fraction)
        self.assertTrue(ev.is_entirely_circular)

    def test_unmatched_restraint_names_are_visible(self) -> None:
        guard = CircularityGuard(["d_hydride", "typo_name"], self.evaluated)
        self.assertEqual(guard.restrained_but_not_evaluated, frozenset({"typo_name"}))

    def test_annotate_fills_the_report_without_mutating_it(self) -> None:
        guard = CircularityGuard.from_pose(self.pose, self.evaluated)
        rep = report("p1", {"d_hydride": True, "hbond_tyr": True,
                            "bd_angle": True, "hbond_ser": False})
        annotated = guard.annotate(rep)
        self.assertEqual(annotated.independent_satisfied, 1)
        self.assertEqual(annotated.independent_total, 2)
        self.assertEqual(sorted(annotated.circular_constraints),
                         ["d_hydride", "hbond_tyr"])
        self.assertEqual(annotated.independent_fraction, 0.5)
        # original untouched
        self.assertEqual(rep.independent_satisfied, 0)
        self.assertEqual(rep.circular_constraints, [])

    def test_for_report_uses_the_reports_own_constraint_names(self) -> None:
        rep = report("p1", {"d_hydride": True, "bd_angle": True})
        guard = CircularityGuard.for_report(self.pose, rep)
        self.assertEqual(guard.evaluated, frozenset({"d_hydride", "bd_angle"}))
        self.assertEqual(guard.independent, frozenset({"bd_angle"}))


class TestCrossMethodAgreement(unittest.TestCase):

    def test_two_independent_routes_agreeing_is_strong(self) -> None:
        level = cross_method_agreement({
            "template_docking": report("p1", {}, gating_passed=True),
            "af3": report("p2", {}, gating_passed=True),
        })
        self.assertIs(level, ConfidenceLevel.STRONG)

    def test_agreement_on_a_negative_is_also_strong_agreement(self) -> None:
        level = cross_method_agreement({"template_docking": False, "af3": False})
        self.assertIs(level, ConfidenceLevel.STRONG)
        self.assertEqual(
            cross_method_verdicts({"template_docking": False, "af3": False}),
            {"template_docking": False, "af3": False},
        )

    def test_disagreement_is_contradictory_not_a_majority_vote(self) -> None:
        level = cross_method_agreement({
            "template_docking": True, "af3": False, "boltz": True,
        })
        self.assertIs(level, ConfidenceLevel.CONTRADICTORY)

    def test_one_route_alone_is_weak(self) -> None:
        self.assertIs(
            cross_method_agreement({"template_docking": True, "af3": None}),
            ConfidenceLevel.WEAK,
        )

    def test_no_route_decided_is_insufficient(self) -> None:
        self.assertIs(
            cross_method_agreement({"template_docking": None, "af3": None}),
            ConfidenceLevel.INSUFFICIENT,
        )
        self.assertIs(cross_method_agreement({}), ConfidenceLevel.INSUFFICIENT)

    def test_an_undecided_third_route_downgrades_to_moderate(self) -> None:
        level = cross_method_agreement({"docking": True, "af3": True, "boltz": None})
        self.assertIs(level, ConfidenceLevel.MODERATE)

    def test_reruns_of_one_program_are_not_two_routes(self) -> None:
        """Two seeds of the same docking run must not manufacture STRONG."""
        reports = {"vina_run1": True, "vina_run2": True}
        self.assertIs(cross_method_agreement(reports), ConfidenceLevel.STRONG)
        grouped = cross_method_agreement(
            reports, independent_groups={"vina_run1": "vina", "vina_run2": "vina"}
        )
        self.assertIs(grouped, ConfidenceLevel.WEAK)

    def test_pose_lists_collapse_to_any_passing(self) -> None:
        level = cross_method_agreement({
            "docking": [report("p1", {}, gating_passed=False),
                        report("p2", {}, gating_passed=True)],
            "af3": [report("p3", {}, gating_passed=True)],
        })
        self.assertIs(level, ConfidenceLevel.STRONG)

    def test_all_poses_failing_is_a_negative_verdict(self) -> None:
        verdicts = cross_method_verdicts({
            "docking": [report("p1", {}, gating_passed=False),
                        report("p2", {}, gating_passed=None)],
        })
        self.assertEqual(verdicts, {"docking": False})

    def test_unreadable_input_raises(self) -> None:
        with self.assertRaises(TypeError):
            cross_method_agreement({"docking": "looks fine"})

    def test_min_methods_below_two_raises(self) -> None:
        with self.assertRaises(ValueError):
            cross_method_agreement({"docking": True}, min_methods=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
