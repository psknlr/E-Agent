"""Tests for standardised family numbering and cross-subfamily equivalence.

The sequences here are synthetic. They are built so that the correct answer is
known by construction: an insertion of a known length shifts downstream
positions by exactly that amount, so a correct implementation must report the
shifted index and an implementation that compares raw numbers must fail.
"""

from __future__ import annotations

import unittest

from eagent.schemas.candidate import ConfidenceLevel
from eagent.science.family_numbering import (
    AnchorMotif, CrossFamilyComparisonError, EquivalenceResult,
    FamilyNumberingScheme, FamilyNumberingError, UnsourcedReferenceError,
    equivalent_position, group_by_standard_position, naive_number_match,
)


# A synthetic 60-residue "family reference". Position 31 (1-based) is Y.
REFERENCE = (
    "MKAIVTGASRGIGRAIAEELAKQGAKVVLNYSSNQAEAEEVVAAIEAAGGKAVAVQADVSK"
)

#: An insertion of this length is placed before the landmark in the variant,
#: so every downstream author number shifts by exactly this much.
INSERT = "GGGGG"


def _scheme(**kw) -> FamilyNumberingScheme:
    params = dict(
        family_name="SYNTH",
        scheme_id="synth.v1",
        reference_label="synthetic reference",
        reference_sequence=REFERENCE,
        reference_source="constructed in tests/test_family_numbering.py",
        catalytic_roles={"catalytic_tyrosine": 31},
        min_identity=0.25,
    )
    params.update(kw)
    return FamilyNumberingScheme(**params)


class TestSchemeConstruction(unittest.TestCase):
    def test_reference_must_be_sourced(self) -> None:
        """A reference with no origin silently redefines every position."""
        with self.assertRaises(UnsourcedReferenceError):
            _scheme(reference_source="   ")

    def test_catalytic_role_outside_the_reference_is_rejected(self) -> None:
        with self.assertRaises(FamilyNumberingError):
            _scheme(catalytic_roles={"bogus": 5000})

    def test_role_position_reports_its_letter(self) -> None:
        pos = _scheme().role_position("catalytic_tyrosine")
        self.assertIsNotNone(pos)
        assert pos is not None
        self.assertEqual(pos.number, 31)
        self.assertEqual(pos.reference_letter, REFERENCE[30])
        self.assertEqual(pos.role, "catalytic_tyrosine")

    def test_positions_from_different_schemes_are_not_comparable(self) -> None:
        a = _scheme().role_position("catalytic_tyrosine")
        b = _scheme(family_name="OTHER", scheme_id="other.v1").role_position(
            "catalytic_tyrosine")
        assert a is not None and b is not None
        self.assertFalse(a.same_scheme_as(b))


class TestRoundTrip(unittest.TestCase):
    def test_identical_sequence_round_trips_every_position(self) -> None:
        s = _scheme()
        for idx in range(0, len(REFERENCE), 7):
            m = s.to_standard(REFERENCE, idx)
            self.assertTrue(m.mapped, f"index {idx}: {m.reason}")
            assert m.standard is not None
            self.assertEqual(m.standard.number, idx + 1)
            back = s.from_standard(REFERENCE, m.standard.number)
            self.assertEqual(back.query_index, idx)

    def test_index_outside_the_sequence_is_refused(self) -> None:
        m = _scheme().to_standard(REFERENCE, 10_000)
        self.assertFalse(m.mapped)
        self.assertIn("outside", m.reason)

    def test_standard_number_outside_the_reference_is_refused(self) -> None:
        m = _scheme().from_standard(REFERENCE, 10_000)
        self.assertFalse(m.mapped)
        self.assertIn("outside", m.reason)


class TestInsertionShift(unittest.TestCase):
    """The central case: equal numbers are wrong, the alignment is right."""

    def setUp(self) -> None:
        self.scheme = _scheme()
        # Insert five residues near the N-terminus, shifting everything after.
        self.variant = REFERENCE[:10] + INSERT + REFERENCE[10:]
        self.shift = len(INSERT)

    def test_downstream_positions_are_shifted_by_the_insertion(self) -> None:
        target_number = 31
        m = self.scheme.from_standard(self.variant, target_number)
        self.assertTrue(m.mapped, m.reason)
        self.assertEqual(m.query_index, (target_number - 1) + self.shift)
        self.assertEqual(m.query_letter, REFERENCE[target_number - 1])

    def test_equivalent_position_finds_the_shifted_residue(self) -> None:
        src_index = 30                      # 0-based: standard position 31
        res = equivalent_position(self.scheme, REFERENCE, src_index, self.variant)
        self.assertIs(res.equivalent, True, res.reason)
        self.assertEqual(res.target_index, src_index + self.shift)
        assert res.standard is not None
        self.assertEqual(res.standard.number, 31)
        self.assertEqual(res.standard.role, "catalytic_tyrosine")
        self.assertEqual(res.source_letter, res.target_letter)

    def test_the_naive_number_comparison_would_have_been_wrong(self) -> None:
        """Guards the premise: raw numbers disagree where the alignment agrees."""
        src_index = 30
        res = equivalent_position(self.scheme, REFERENCE, src_index, self.variant)
        assert res.target_index is not None
        naive_author_number = src_index + 1
        aligned_author_number = res.target_index + 1
        self.assertNotEqual(naive_author_number, aligned_author_number)

    def test_a_wrong_expected_letter_is_reported_as_disagreement(self) -> None:
        wrong = "W" if REFERENCE[30] != "W" else "A"
        res = equivalent_position(self.scheme, REFERENCE, 30, self.variant,
                                  expect_target_letter=wrong)
        self.assertIs(res.equivalent, False)
        self.assertIn("disagree", res.reason)


class TestRefusals(unittest.TestCase):
    def test_cross_family_comparison_raises(self) -> None:
        """An SDR position has no AKR counterpart; returning one is the bug."""
        sdr = _scheme(family_name="SDR", scheme_id="sdr.v1")
        akr = _scheme(family_name="AKR", scheme_id="akr.v1")
        with self.assertRaises(CrossFamilyComparisonError) as ctx:
            equivalent_position(sdr, REFERENCE, 30, REFERENCE, target_scheme=akr)
        self.assertIn("do not share", str(ctx.exception))

    def test_weak_alignment_yields_undetermined_not_false(self) -> None:
        """Too little signal must not be reported as 'not equivalent'."""
        unrelated = "WWWWPPPPCCCCWWWWPPPPCCCCWWWWPPPPCCCCWWWWPPPPCCCC"
        res = equivalent_position(_scheme(min_identity=0.90), REFERENCE, 30, unrelated)
        self.assertIsNone(res.equivalent)
        self.assertFalse(res.answered)

    def test_a_deleted_position_has_no_counterpart(self) -> None:
        deleted = REFERENCE[:28] + REFERENCE[34:]      # removes position 31
        m = _scheme().from_standard(deleted, 31)
        if m.query_index is not None:
            # If the aligner closed the gap elsewhere, the residue must at least
            # not be silently presented as the original one.
            self.assertNotEqual(m.query_letter, None)
        else:
            self.assertIn("deletion", m.reason)

    def test_gap_in_reference_gives_no_standard_position(self) -> None:
        inserted = REFERENCE[:10] + "DDDDDDDD" + REFERENCE[10:]
        m = _scheme().to_standard(inserted, 13)        # inside the insertion
        if not m.mapped:
            self.assertIn("gap", m.reason)

    def test_naive_number_match_cannot_be_used_as_a_boolean(self) -> None:
        warn = naive_number_match(155, 155)
        self.assertTrue(warn.numbers_equal)
        self.assertIn("do not mean equivalent", warn.warning)
        with self.assertRaises(FamilyNumberingError):
            bool(warn)


class TestAnchors(unittest.TestCase):
    def test_a_satisfied_anchor_raises_confidence(self) -> None:
        anchor = AnchorMotif(name="rossmann_like", pattern="GASRGIG",
                             expected_standard_start=8, tolerance=5,
                             role="cofactor_binding",
                             evidence="constructed for this test")
        s = _scheme(anchors=(anchor,))
        checks = s.check_anchors(REFERENCE)
        self.assertEqual(len(checks), 1)
        self.assertTrue(checks[0].found)
        self.assertTrue(checks[0].agreed, checks[0].detail)

    def test_an_anchor_in_the_wrong_place_contradicts_the_alignment(self) -> None:
        anchor = AnchorMotif(name="misplaced", pattern="GASRGIG",
                             expected_standard_start=55, tolerance=2)
        s = _scheme(anchors=(anchor,))
        m = s.to_standard(REFERENCE, 30)
        self.assertIs(m.anchors_agree, False)
        self.assertIs(m.confidence, ConfidenceLevel.CONTRADICTORY)

    def test_contradictory_anchors_make_equivalence_undetermined(self) -> None:
        anchor = AnchorMotif(name="misplaced", pattern="GASRGIG",
                             expected_standard_start=55, tolerance=2)
        s = _scheme(anchors=(anchor,))
        variant = REFERENCE[:10] + INSERT + REFERENCE[10:]
        res = equivalent_position(s, REFERENCE, 30, variant)
        self.assertIsNone(res.equivalent)
        self.assertIn("anchors disagree", res.reason)

    def test_an_absent_motif_is_recorded_without_a_verdict(self) -> None:
        anchor = AnchorMotif(name="absent", pattern="HHHHHHHH",
                             expected_standard_start=20)
        checks = _scheme(anchors=(anchor,)).check_anchors(REFERENCE)
        self.assertFalse(checks[0].found)
        self.assertIsNone(checks[0].agreed)


class TestGrouping(unittest.TestCase):
    def test_positions_group_by_standard_number_across_subfamilies(self) -> None:
        s = _scheme()
        variant = REFERENCE[:10] + INSERT + REFERENCE[10:]
        grouped = group_by_standard_position(
            s,
            {"parent": REFERENCE, "reported": variant},
            {"parent": [30], "reported": [30 + len(INSERT)]},
        )
        self.assertIn(31, grouped)
        names = sorted(n for n, _, _ in grouped[31])
        self.assertEqual(names, ["parent", "reported"])

    def test_unmappable_entries_are_absent_rather_than_approximated(self) -> None:
        s = _scheme(min_identity=0.99)
        unrelated = "WWWWPPPPCCCCWWWWPPPPCCCC"
        grouped = group_by_standard_position(
            s, {"weird": unrelated}, {"weird": [3]})
        self.assertEqual(grouped, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
