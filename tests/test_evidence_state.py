"""Regression tests: a computed verdict must match the evidence behind it.

Two findings from an external review, each reproduced before being fixed.
They are the same error twice: a conclusion stated more strongly than the
thing it rests on.

A pose whose every gating constraint was restrained while it was built
satisfies exactly what it was built to satisfy. The robustness figure says so
-- ``robustness_G = None``, nothing was corroborated -- and the stereochemical
call, computed from the very same pose, said ``favors_target``. One candidate,
two numbers, opposite claims.

The chemoselectivity comparison is two distances and a margin, and the margin
is the whole content of it: both distances carry the pose generator's
positional scatter, so "is the competitor closer" has no answer until somebody
says how much closer counts. The module default says in its own docstring that
it is fitted to nothing, and a displacement measured against it produced
``MECHANISM_VIOLATED`` -- a hard computational negative -- while every template
window with the same provenance is explicitly not allowed to reject.
"""

from __future__ import annotations

import unittest

from eagent.schemas.chem import CofactorState
from eagent.schemas.record import OutcomeClass
from eagent.tools.evaluate_catalysis import (
    ChemoselectivityCheck, PoseOutcome, WindowAuthority,
)

from test_evaluate_catalysis import (
    _Harness, make_binding, make_candidate, make_pose, make_structure,
)

EVERY_CONSTRAINT = ["hydride_transfer_distance", "burgi_dunitz_angle",
                    "oxyanion_Tyr_OH", "cofactor_anchor_Lys_NZ"]


class StereoRestsOnIndependentEvidence(_Harness):
    """A face a restraint put there is not a face the enzyme chose."""

    def run_with(self, restrained):
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED, restrained=restrained)])
        _, result = self.run_step([candidate], [make_binding()],
                                  {"p1": make_structure()})
        return candidate, result

    def test_a_fully_restrained_pose_makes_no_stereochemical_call(self) -> None:
        candidate, _ = self.run_with(EVERY_CONSTRAINT)
        self.assertEqual(candidate.stereo.call, "insufficient_evidence")

    def test_the_two_numbers_now_agree(self) -> None:
        """robustness_G = None and favors_target cannot both be honest."""
        candidate, result = self.run_with(EVERY_CONSTRAINT)
        evaluation = self.candidate_eval(result)
        self.assertIsNone(evaluation["robustness_G"])
        self.assertEqual(candidate.stereo.call, "insufficient_evidence")

    def test_the_exclusion_is_counted_and_explained(self) -> None:
        candidate, result = self.run_with(EVERY_CONSTRAINT)
        evaluation = self.candidate_eval(result)
        self.assertEqual(evaluation["stereo_poses_excluded_as_circular"], 1)
        self.assertIn("restrained while they were built", candidate.stereo.basis)

    def test_no_pose_voting_is_not_the_same_as_no_pose_competent(self) -> None:
        """Both give insufficient_evidence; only the basis distinguishes them."""
        candidate, _ = self.run_with(EVERY_CONSTRAINT)
        self.assertIn("no pose carries independent evidence",
                      candidate.stereo.basis)

    def test_a_partly_restrained_pose_still_votes(self) -> None:
        candidate, result = self.run_with(["hydride_transfer_distance"])
        self.assertEqual(candidate.stereo.call, "favors_target")
        self.assertEqual(
            self.candidate_eval(result)["stereo_poses_excluded_as_circular"], 0)

    def test_an_unrestrained_pose_is_untouched(self) -> None:
        candidate, _ = self.run_with([])
        self.assertEqual(candidate.stereo.call, "favors_target")

    def test_the_counts_exclude_the_circular_pose(self) -> None:
        candidate, _ = self.run_with(EVERY_CONSTRAINT)
        self.assertEqual(candidate.stereo.target_face_poses, 0)
        self.assertEqual(candidate.stereo.opposite_face_poses, 0)


class ChemoselectivityMarginAuthority(_Harness):
    """A margin fitted to nothing cannot turn a measurement into a negative."""

    SOURCE = "positional scatter of the in-house pose generator, n=240 redocks"

    def run_with(self, source: str):
        candidate = make_candidate(poses=[make_pose(
            cofactor_state=CofactorState.REDUCED)])
        _, result = self.run_step(
            [candidate], [make_binding(competing=True)],
            {"p1": make_structure(with_competing_carbonyl=True)},
            competing_group_margin_angstrom=0.2,
            competing_group_margin_source=source)
        return candidate, result

    def test_the_default_margin_is_marked_uncalibrated(self) -> None:
        _, result = self.run_with("")
        pose = self.pose_eval(result)
        self.assertEqual(pose["chemoselectivity_margin_authority"],
                         WindowAuthority.UNCALIBRATED.value)

    def test_an_uncalibrated_margin_does_not_produce_a_negative(self) -> None:
        _, result = self.run_with("")
        pose = self.pose_eval(result)
        self.assertEqual(pose["outcome"],
                         PoseOutcome.OUTSIDE_UNCALIBRATED_WINDOW.value)
        self.assertNotEqual(pose["record_outcome"],
                            OutcomeClass.COMPUTATIONAL_NEGATIVE.value)

    def test_the_displacement_is_still_measured(self) -> None:
        _, result = self.run_with("")
        pose = self.pose_eval(result)
        self.assertIs(pose["target_in_reactive_position"], False)
        self.assertIn("second_carbonyl_C9", pose["reason"])

    def test_not_acting_on_it_is_itself_reported(self) -> None:
        """An operator seeing the displacement would assume it was acted on."""
        _, result = self.run_with("")
        self.assertIn("chemoselectivity_margin_uncalibrated",
                      [u.code for u in result.uncertainty])

    def test_a_calibrated_margin_still_rejects(self) -> None:
        _, result = self.run_with(self.SOURCE)
        pose = self.pose_eval(result)
        self.assertEqual(pose["outcome"], PoseOutcome.MECHANISM_VIOLATED.value)
        self.assertEqual(pose["record_outcome"],
                         OutcomeClass.COMPUTATIONAL_NEGATIVE.value)
        self.assertIs(pose["gating_passed"], False)

    def test_the_margin_source_reaches_the_record(self) -> None:
        _, result = self.run_with(self.SOURCE)
        pose = self.pose_eval(result)
        self.assertEqual(pose["chemoselectivity_margin_source"], self.SOURCE)
        self.assertEqual(pose["chemoselectivity_margin_authority"],
                         WindowAuthority.CALIBRATED.value)

    def test_the_two_properties_say_different_things(self) -> None:
        """The fact and the authority to act on it are kept apart."""
        check = ChemoselectivityCheck(
            tested=True, target_distance_A=3.4, displacing_label="C9",
            displacing_distance_A=2.9, margin_A=0.2,
            margin_authority=WindowAuthority.UNCALIBRATED)
        self.assertIs(check.target_in_reactive_position, False)
        self.assertFalse(check.displaced_on_a_calibrated_margin)

    def test_an_untested_check_displaces_nothing(self) -> None:
        check = ChemoselectivityCheck(tested=False, target_distance_A=3.4,
                                      margin_authority=WindowAuthority.CALIBRATED)
        self.assertIsNone(check.target_in_reactive_position)
        self.assertFalse(check.displaced_on_a_calibrated_margin)


if __name__ == "__main__":
    unittest.main(verbosity=2)
