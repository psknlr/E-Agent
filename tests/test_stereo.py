"""Tests for :mod:`eagent.science.stereo`.

The face tests are built on a synthetic trigonal centre whose answer is known
by construction rather than by running the code and recording what it said:
the carbonyl carbon sits at the origin with its three ligands at 0, 120 and 240
degrees in the xy-plane, so "which side is the donor on" is decidable by hand.

The face-to-configuration test does not trust the module's own derivation
either. It re-derives every answer from an independent signed-volume
implementation of the R/S rule written in this file, for all four possible
incoming-group ranks and both faces.
"""

from __future__ import annotations

import math
import unittest

from eagent.errors import TemplateError
from eagent.schemas import Stereochemistry
from eagent.science.stereo import (
    CIPPriority,
    call_stereochemistry,
    centre_planarity_deviation,
    cip_from_template,
    face_of_approach,
    face_of_approach_with_cip,
    face_to_configuration,
    out_of_plane_angle_deg,
)

try:  # the sibling agent's modules may not be on disk yet
    from eagent.science.structure_io import Atom  # type: ignore
    _HAS_STRUCTURE_IO = True
except Exception:  # pragma: no cover - availability depends on parallel work
    Atom = None  # type: ignore[assignment]
    _HAS_STRUCTURE_IO = False


# -- the synthetic centre ---------------------------------------------------
# Trigonal carbon at the origin; ligands at 0 / 120 / 240 degrees in z = 0.
C = (0.0, 0.0, 0.0)
O = (1.0, 0.0, 0.0)                      # rank 1 among the three sp2 ligands
A = (-0.5, math.sqrt(3) / 2, 0.0)        # rank 2
B = (-0.5, -math.sqrt(3) / 2, 0.0)       # rank 3

# Looking down from +z (x right, y up) the sequence O -> A -> B runs 0 -> 120 ->
# 240 degrees, i.e. counterclockwise. Counterclockwise 1 -> 2 -> 3 means the
# viewer is on the si face. So a donor at +z sees si, a donor at -z sees re.
DONOR_ABOVE = (0.0, 0.0, 1.0)
DONOR_BELOW = (0.0, 0.0, -1.0)


def _sub(p, q):
    return (p[0] - q[0], p[1] - q[1], p[2] - q[2])


def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


def _dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def reference_descriptor(p1, p2, p3, p4) -> str:
    """Independent R/S reference from the signed volume of the ligand tetrahedron.

    ``p1..p4`` are ligand positions in descending CIP priority. Calibrated on a
    case worked out by hand: with the lowest-priority ligand at -z and
    1 -> 2 -> 3 counterclockwise seen from +z (lowest pointing away), the centre
    is S, and that arrangement gives a positive triple product.
    """
    v = _dot(_sub(p1, p4), _cross(_sub(p2, p4), _sub(p3, p4)))
    if abs(v) < 1e-12:
        raise AssertionError("degenerate reference geometry")
    return "S" if v > 0 else "R"


class TestReferenceImplementation(unittest.TestCase):
    """Pin the reference itself before using it to judge the module."""

    def test_reference_matches_hand_worked_case(self) -> None:
        # O, A, B in the plane; H (lowest priority) below. Viewed from +z the
        # lowest ligand points away and O -> A -> B is counterclockwise => S.
        self.assertEqual(reference_descriptor(O, A, B, (0.0, 0.0, -1.0)), "S")
        self.assertEqual(reference_descriptor(O, A, B, (0.0, 0.0, 1.0)), "R")


class TestFaceOfApproach(unittest.TestCase):

    def test_known_construction_gives_si_from_above(self) -> None:
        self.assertEqual(face_of_approach(DONOR_ABOVE, C, A, B, O), "si")

    def test_known_construction_gives_re_from_below(self) -> None:
        self.assertEqual(face_of_approach(DONOR_BELOW, C, A, B, O), "re")

    def test_inverting_one_coordinate_flips_the_call(self) -> None:
        """Negating the donor's z moves it to the other face and must flip."""
        dx, dy, dz = DONOR_ABOVE
        first = face_of_approach((dx, dy, dz), C, A, B, O)
        second = face_of_approach((dx, dy, -dz), C, A, B, O)
        self.assertEqual(first, "si")
        self.assertEqual(second, "re")
        self.assertNotEqual(first, second)

    def test_swapping_the_two_substituents_flips_the_call(self) -> None:
        """Reordering the triple is a parity change, so the label must invert."""
        self.assertEqual(face_of_approach(DONOR_ABOVE, C, A, B, O), "si")
        self.assertEqual(face_of_approach(DONOR_ABOVE, C, B, A, O), "re")

    def test_in_plane_donor_is_ambiguous_rather_than_forced(self) -> None:
        near_plane = (0.5, 0.3, 0.02)        # elevation ~ 2 degrees
        self.assertLess(out_of_plane_angle_deg(near_plane, C, A, B, O), 5.0)
        self.assertEqual(
            face_of_approach(near_plane, C, A, B, O), "in_plane_ambiguous"
        )

    def test_tolerance_is_configurable(self) -> None:
        """The same coordinates give different labels under different tolerances."""
        near_plane = (0.5, 0.3, 0.02)        # elevation ~ 2 degrees
        # A tighter tolerance than the protocol's noise lets the call through.
        self.assertEqual(
            face_of_approach(near_plane, C, A, B, O, in_plane_tolerance_deg=0.5),
            "si",
        )
        oblique = (1.0, 0.0, 1.0)            # elevation exactly 45 degrees
        self.assertAlmostEqual(
            out_of_plane_angle_deg(oblique, C, A, B, O), 45.0, places=9
        )
        self.assertEqual(
            face_of_approach(oblique, C, A, B, O, in_plane_tolerance_deg=30.0), "si"
        )
        self.assertEqual(
            face_of_approach(oblique, C, A, B, O, in_plane_tolerance_deg=60.0),
            "in_plane_ambiguous",
        )

    def test_elevation_is_ninety_degrees_straight_above(self) -> None:
        self.assertAlmostEqual(
            out_of_plane_angle_deg(DONOR_ABOVE, C, A, B, O), 90.0, places=9
        )

    def test_collinear_ligands_raise_rather_than_guess(self) -> None:
        with self.assertRaises(ValueError):
            face_of_approach(DONOR_ABOVE, C, (1.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                             (3.0, 0.0, 0.0))

    def test_donor_on_the_carbonyl_carbon_raises(self) -> None:
        with self.assertRaises(ValueError):
            face_of_approach(C, C, A, B, O)

    def test_bad_tolerance_raises(self) -> None:
        with self.assertRaises(ValueError):
            face_of_approach(DONOR_ABOVE, C, A, B, O, in_plane_tolerance_deg=90.0)

    def test_planarity_diagnostic(self) -> None:
        self.assertAlmostEqual(centre_planarity_deviation(C, A, B, O), 0.0, places=12)
        self.assertAlmostEqual(
            centre_planarity_deviation((0.0, 0.0, 0.3), A, B, O), 0.3, places=12
        )

    def test_with_cip_sorts_the_ligands(self) -> None:
        """Passing the substituents in the wrong dict order must not matter."""
        cip = CIPPriority(ranks={"o": 1, "a": 2, "b": 3}, source="operator:test")
        subs = {"b": B, "o": O, "a": A}       # deliberately scrambled
        self.assertEqual(
            face_of_approach_with_cip(DONOR_ABOVE, C, subs, cip), "si"
        )
        # Swap which carbon substituent is senior: the face label inverts.
        cip_swapped = CIPPriority(ranks={"o": 1, "a": 3, "b": 2},
                                  source="operator:test")
        self.assertEqual(
            face_of_approach_with_cip(DONOR_ABOVE, C, subs, cip_swapped), "re"
        )

    def test_with_cip_refuses_unranked_ligand(self) -> None:
        cip = CIPPriority(ranks={"o": 1, "a": 2, "b": 3}, source="operator:test")
        with self.assertRaises(TemplateError):
            face_of_approach_with_cip(DONOR_ABOVE, C, {"o": O, "a": A, "z": B}, cip)


class TestCIPPriority(unittest.TestCase):

    def test_unsourced_ranks_are_rejected(self) -> None:
        with self.assertRaises(TemplateError):
            CIPPriority(ranks={"o": 1, "a": 2, "b": 3}, source="")

    def test_duplicate_ranks_are_rejected(self) -> None:
        with self.assertRaises(TemplateError):
            CIPPriority(ranks={"o": 1, "a": 2, "b": 2}, source="operator:test")

    def test_priority_order(self) -> None:
        cip = CIPPriority(ranks={"b": 3, "o": 1, "a": 2}, source="operator:test")
        self.assertEqual(cip.keys_in_priority_order(), ("o", "a", "b"))
        self.assertTrue(cip.is_complete)
        self.assertEqual(cip.missing_keys(["o", "zz"]), ["zz"])

    def test_cip_from_template_reads_a_config_block(self) -> None:
        cfg = {
            "template_id": "rt_ketone_v1",
            "cip_ranks": {"o": 1, "a": 2, "b": 3,
                          "incoming_group_rank": 4,
                          "source": "literature:10.1000/example"},
        }
        cip = cip_from_template(cfg)
        self.assertEqual(cip.as_dict(), {"o": 1, "a": 2, "b": 3})
        self.assertEqual(cip.incoming_rank, 4)
        self.assertEqual(cip.source, "literature:10.1000/example")

    def test_cip_from_template_reads_a_nested_block(self) -> None:
        cfg = {"id": "task1", "stereo": {"cip_ranks": {"o": 1, "a": 2, "b": 3}}}
        cip = cip_from_template(cfg)
        self.assertEqual(cip.as_dict(), {"o": 1, "a": 2, "b": 3})
        self.assertIn("task1", cip.source)

    def test_cip_from_template_refuses_to_improvise(self) -> None:
        """A template without ranks must raise, not fall back to element order."""
        with self.assertRaises(TemplateError):
            cip_from_template({"template_id": "rt_no_ranks"})
        with self.assertRaises(TemplateError):
            cip_from_template(None)


class TestFaceToConfiguration(unittest.TestCase):

    def _ranks_for(self, incoming_rank: int) -> dict[str, int]:
        """Give O, A, B the remaining ranks, keeping O > A > B in seniority."""
        remaining = [r for r in (1, 2, 3, 4) if r != incoming_rank]
        return {"o": remaining[0], "a": remaining[1], "b": remaining[2]}

    def test_matches_independent_signed_volume_for_every_incoming_rank(self) -> None:
        positions = {"o": O, "a": A, "b": B}
        for donor, expected_face in ((DONOR_BELOW, "re"), (DONOR_ABOVE, "si")):
            face = face_of_approach(donor, C, A, B, O)
            self.assertEqual(face, expected_face)
            # The incoming group ends up where the donor came from.
            incoming_pos = donor
            for g in (1, 2, 3, 4):
                ranks = self._ranks_for(g)
                cip = CIPPriority(ranks=ranks, source="operator:test")

                by_rank = {ranks[k]: positions[k] for k in ranks}
                by_rank[g] = incoming_pos
                ordered = [by_rank[i] for i in (1, 2, 3, 4)]
                expected = reference_descriptor(*ordered)

                got = face_to_configuration(face, cip, g)
                self.assertEqual(
                    got, expected,
                    msg=f"face={face} incoming_rank={g}: got {got}, "
                        f"signed-volume reference says {expected}",
                )

    def test_hydride_on_the_re_face_gives_s(self) -> None:
        """The textbook case: re-face hydride delivery gives the S alcohol."""
        cip = CIPPriority(ranks={"o": 1, "a": 2, "b": 3}, source="operator:test")
        self.assertEqual(face_to_configuration("re", cip, 4), "S")
        self.assertEqual(face_to_configuration("si", cip, 4), "R")

    def test_missing_cip_ranks_produce_no_configuration(self) -> None:
        self.assertIsNone(face_to_configuration("re", None, 4))

    def test_incomplete_cip_ranks_produce_no_configuration(self) -> None:
        partial = CIPPriority(ranks={"o": 1, "a": 2}, source="operator:test")
        self.assertIsNone(face_to_configuration("re", partial, 4))

    def test_missing_incoming_rank_produces_no_configuration(self) -> None:
        cip = CIPPriority(ranks={"o": 1, "a": 2, "b": 3}, source="operator:test")
        self.assertIsNone(face_to_configuration("re", cip, None))

    def test_ambiguous_face_produces_no_configuration(self) -> None:
        cip = CIPPriority(ranks={"o": 1, "a": 2, "b": 3}, source="operator:test")
        self.assertIsNone(face_to_configuration("in_plane_ambiguous", cip, 4))
        self.assertIsNone(face_to_configuration(None, cip, 4))

    def test_inconsistent_rank_set_raises(self) -> None:
        cip = CIPPriority(ranks={"o": 1, "a": 2, "b": 3}, source="operator:test")
        with self.assertRaises(ValueError):
            face_to_configuration("re", cip, 5)

    def test_incoming_rank_may_come_from_the_container(self) -> None:
        cip = CIPPriority(ranks={"o": 1, "a": 2, "b": 3}, source="operator:test",
                          incoming_rank=4)
        self.assertEqual(face_to_configuration("re", cip), "S")


class TestCallStereochemistry(unittest.TestCase):

    def test_counts_and_favours_target(self) -> None:
        call = call_stereochemistry(
            {"p1": "S", "p2": "S", "p3": "S", "p4": None}, "S", basis="unit test"
        )
        self.assertEqual(call.call, "favors_target")
        self.assertEqual(call.target_face_poses, 3)
        self.assertEqual(call.opposite_face_poses, 0)
        self.assertEqual(call.undetermined_poses, 1)
        self.assertIsNone(call.predicted_ee_pct)
        self.assertIn("sampling artefact", call.basis)

    def test_competing_poses_are_not_resolved_by_majority(self) -> None:
        call = call_stereochemistry(["S", "S", "S", "R"], "S")
        self.assertEqual(call.call, "competing_poses")
        self.assertEqual(call.target_face_poses, 3)
        self.assertEqual(call.opposite_face_poses, 1)

    def test_favours_opposite(self) -> None:
        call = call_stereochemistry(["R", "R", "R"], Stereochemistry.S)
        self.assertEqual(call.call, "favors_opposite")
        self.assertEqual(call.opposite_face_poses, 3)

    def test_all_undetermined_is_insufficient_evidence(self) -> None:
        call = call_stereochemistry([None, None, None], "R")
        self.assertEqual(call.call, "insufficient_evidence")
        self.assertEqual(call.undetermined_poses, 3)
        self.assertIsNone(call.predicted_ee_pct)

    def test_no_poses_is_insufficient_evidence(self) -> None:
        call = call_stereochemistry([], "R")
        self.assertEqual(call.call, "insufficient_evidence")

    def test_never_populates_predicted_ee(self) -> None:
        """A unanimous 10/0 split must still refuse to name an ee."""
        call = call_stereochemistry(["S"] * 10, "S")
        self.assertEqual(call.call, "favors_target")
        self.assertIsNone(call.predicted_ee_pct)
        self.assertIsNone(call.calibration_source)

    def test_achiral_target_is_not_applicable(self) -> None:
        call = call_stereochemistry(["R", "S"], Stereochemistry.ACHIRAL)
        self.assertEqual(call.call, "not_applicable")

    def test_unspecified_target_is_insufficient_not_a_guess(self) -> None:
        call = call_stereochemistry(["S", "S"], Stereochemistry.UNSPECIFIED)
        self.assertEqual(call.call, "insufficient_evidence")
        self.assertEqual(call.target_face_poses, 0)

    def test_pairs_are_accepted(self) -> None:
        call = call_stereochemistry([("p1", "R"), ("p2", None)], "R")
        self.assertEqual(call.target_face_poses, 1)
        self.assertEqual(call.undetermined_poses, 1)

    def test_unknown_label_raises_instead_of_being_dropped(self) -> None:
        with self.assertRaises(ValueError):
            call_stereochemistry(["S", "probably S"], "S")


@unittest.skipUnless(_HAS_STRUCTURE_IO, "eagent.science.structure_io not on disk yet")
class TestAtomInterop(unittest.TestCase):
    """Atom objects must work as coordinate arguments without any conversion."""

    @staticmethod
    def _atom(serial, name, element, xyz):
        return Atom(serial=serial, name=name, element=element, resname="LIG",
                    chain="A", resseq=1, icode=" ", altloc=" ",
                    x=xyz[0], y=xyz[1], z=xyz[2], is_hetatm=True)

    def test_atoms_are_accepted_directly(self) -> None:
        face = face_of_approach(
            self._atom(1, "H4", "H", DONOR_ABOVE),
            self._atom(2, "C1", "C", C),
            self._atom(3, "CA", "C", A),
            self._atom(4, "CB", "C", B),
            self._atom(5, "O1", "O", O),
        )
        self.assertEqual(face, "si")

    def test_atoms_and_tuples_agree(self) -> None:
        """Duck-typing must not change the answer relative to plain tuples."""
        for donor, expected in ((DONOR_ABOVE, "si"), (DONOR_BELOW, "re")):
            atoms = face_of_approach(
                self._atom(1, "H4", "H", donor),
                self._atom(2, "C1", "C", C),
                self._atom(3, "CA", "C", A),
                self._atom(4, "CB", "C", B),
                self._atom(5, "O1", "O", O),
            )
            self.assertEqual(atoms, expected)
            self.assertEqual(atoms, face_of_approach(donor, C, A, B, O))


if __name__ == "__main__":
    unittest.main(verbosity=2)
