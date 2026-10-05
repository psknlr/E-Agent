"""Tests for residue numbering, the layer that prevents off-by-one mutations.

The fixtures are inline and small enough that every expected index can be
counted by eye. The structure chain deliberately contains:

* a numbering gap (author 13 and 14 are missing, i.e. a disordered loop),
* a cofactor, a metal and a water interleaved on the same author chain,
* a selenomethionine written as HETATM,

because each of those is a documented way for a sequence derived from a
structure to come out the wrong length -- and a sequence that is one residue
short misnumbers everything after it.
"""

from __future__ import annotations

import unittest

from eagent.science.numbering import (
    AuthorPosition,
    NumberingError,
    alignment_score,
    build_map,
    needleman_wunsch,
    one_to_three,
    residue_one_letter,
    three_to_one,
    verify_residue,
)
from eagent.science.structure_io import read_pdb

#: Chain A: MET10 LYS11 VAL12 [gap 13-14] TRP15 TYR16, plus ZN, NAP and HOH.
PDB_TEXT = """\
ATOM      1  N   MET A  10      11.104   6.134  -6.504  1.00 20.00           N
ATOM      2  CA  MET A  10      11.639   6.071  -5.147  1.00 20.00           C
ATOM      3  N   LYS A  11      13.800   5.400  -4.400  1.00 22.00           N
ATOM      4  CA  LYS A  11      15.250   5.450  -4.300  1.00 22.00           C
ATOM      5  N   VAL A  12      15.900   6.500  -4.000  1.00 21.00           N
ATOM      6  CA  VAL A  12      17.350   6.600  -3.950  1.00 21.00           C
ATOM      7  N   TRP A  15       4.000   5.000   5.000  1.00 19.00           N
ATOM      8  CA  TRP A  15       4.500   5.500   6.200  1.00 19.00           C
ATOM      9  N   TYR A  16       7.000   5.000   5.000  1.00 19.00           N
ATOM     10  CA  TYR A  16       7.800   5.600   6.000  1.00 19.00           C
HETATM  900 ZN    ZN A 400       5.000   5.000   5.000  1.00 15.00          ZN
HETATM  901  C4  NAP A 401       3.000   4.000   5.000  1.00 18.00           C
HETATM  950  O   HOH A 500       1.000   2.000   3.000  1.00 30.00           O
END
"""

#: Selenomethionine recorded as HETATM, as many SeMet-phased entries do.
PDB_WITH_MSE = """\
ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00 20.00           C
HETATM    2  CA  MSE B   2       3.800   0.000   0.000  1.00 20.00           C
ATOM      3  CA  GLY B   3       7.600   0.000   0.000  1.00 20.00           C
END
"""


def _chain_a():
    chain = read_pdb(PDB_TEXT).chain("A")
    assert chain is not None
    return chain


class TestResidueTables(unittest.TestCase):
    def test_standard_round_trip(self) -> None:
        self.assertEqual(three_to_one("TYR"), "Y")
        self.assertEqual(three_to_one("gly"), "G")
        self.assertEqual(one_to_three("Y"), "TYR")
        self.assertEqual(one_to_three("w"), "TRP")
        for three in ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
                      "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
                      "THR", "TRP", "TYR", "VAL"):
            self.assertEqual(one_to_three(three_to_one(three)), three)

    def test_modified_residues_map_to_their_parent(self) -> None:
        self.assertEqual(three_to_one("MSE"), "M")
        self.assertEqual(three_to_one("SEP"), "S")
        self.assertEqual(three_to_one("KCX"), "K")

    def test_unknown_residue_is_strict_by_default(self) -> None:
        # Returning "X" here would turn a bound ligand into a residue and
        # shift every position after it.
        with self.assertRaises(NumberingError):
            three_to_one("NAP")
        self.assertEqual(three_to_one("NAP", strict=False), "X")
        self.assertIsNone(residue_one_letter("NAP"))
        self.assertIsNone(residue_one_letter("HOH"))
        self.assertEqual(residue_one_letter("UNK"), "X")

    def test_one_to_three_rejects_junk(self) -> None:
        for bad in ("", "XX", "B", "1"):
            with self.assertRaises(NumberingError):
                one_to_three(bad)


class TestNeedlemanWunsch(unittest.TestCase):
    def test_identical_sequences(self) -> None:
        aln = needleman_wunsch("MKVLAYG", "MKVLAYG")
        self.assertEqual(aln.aligned_a, "MKVLAYG")
        self.assertEqual(aln.aligned_b, "MKVLAYG")
        self.assertEqual(aln.identity, 1.0)
        self.assertEqual(aln.n_gaps, 0)

    def test_tuple_unpacking_still_works(self) -> None:
        a, b, identity = needleman_wunsch("ACDE", "ACDE")
        self.assertEqual((a, b, identity), ("ACDE", "ACDE", 1.0))

    def test_internal_deletion_stays_one_block(self) -> None:
        # Affine gaps: one three-residue gap, not three scattered ones.
        aln = needleman_wunsch("MKVLAYGWHTRQ", "MKVLWHTRQ")
        self.assertEqual(aln.aligned_a, "MKVLAYGWHTRQ")
        self.assertEqual(aln.aligned_b, "MKVL---WHTRQ")
        self.assertEqual(aln.identity, 1.0)
        self.assertEqual(aln.n_aligned_columns, 9)
        self.assertEqual(aln.n_gaps, 3)
        self.assertAlmostEqual(aln.coverage_b(), 1.0)

    def test_terminal_extension(self) -> None:
        aln = needleman_wunsch("MKVQAWY", "GMKVQAWY")
        self.assertEqual(aln.aligned_a, "-MKVQAWY")
        self.assertEqual(aln.aligned_b, "GMKVQAWY")

    def test_mismatches_are_counted_not_gapped(self) -> None:
        aln = needleman_wunsch("MKVLAYG", "MKVLGYG")
        self.assertEqual(aln.aligned_a, "MKVLAYG")
        self.assertEqual(aln.aligned_b, "MKVLGYG")
        self.assertAlmostEqual(aln.identity, 6 / 7)
        self.assertEqual(aln.n_identities, 6)

    def test_affine_model_prefers_one_long_gap(self) -> None:
        one_block = alignment_score("ACDEFGHIK", "AC-----IK")
        scattered = alignment_score("ACDEFGHIK", "A-C-D-F-K")
        self.assertGreater(one_block, scattered)

    def test_score_matches_the_alignment_it_produced(self) -> None:
        aln = needleman_wunsch("MKVLAYGWHTRQ", "MKVLWHTRQ")
        self.assertAlmostEqual(
            alignment_score(aln.aligned_a, aln.aligned_b), 9 * 2.0 - (10.0 + 1.5)
        )

    def test_empty_sequence_raises(self) -> None:
        with self.assertRaises(NumberingError):
            needleman_wunsch("", "ACDE")

    def test_negative_gap_penalties_rejected(self) -> None:
        # They are magnitudes, subtracted. A negative value would reward gaps.
        with self.assertRaises(NumberingError):
            needleman_wunsch("ACDE", "ACDE", gap_open=-10.0)

    def test_moderate_length_alignment_completes(self) -> None:
        a = ("MSTAGKVIKCKAAVLWEEKKPFSIEEVEVAPPKAHEVRIKMVATGICRSDDHVVSGTLVTP"
             "LPVIAGHEAAGIVESIGEGVTTVRPGDKVIPLFTPQCGKCRVCKHPEGNFCLKNDLSMPRG")
        b = a[:40] + a[48:]          # an eight-residue internal deletion
        aln = needleman_wunsch(a, b)
        self.assertEqual(aln.identity, 1.0)
        self.assertEqual(aln.n_gaps, 8)
        self.assertEqual(aln.aligned_b.count("-"), 8)


class TestBuildMap(unittest.TestCase):
    def setUp(self) -> None:
        # Candidate has two extra residues (Q, A) where the structure has its
        # disordered 13/14 gap.
        self.candidate = "MKVQAWY"
        self.map = build_map(self.candidate, _chain_a())

    def test_non_polymer_groups_do_not_enter_the_sequence(self) -> None:
        # ZN, NAP and HOH sit on the same author chain. If any of them were
        # counted as a residue the whole map would shift.
        self.assertEqual(len(self.map.index_to_author), 5)
        self.assertEqual(self.map.structure_identity, 1.0)
        self.assertTrue(
            any("non-polymer group" in n for n in self.map.notes)
        )

    def test_author_mapping_both_directions(self) -> None:
        self.assertEqual(self.map.to_author(0), AuthorPosition("A", 10, ""))
        self.assertEqual(self.map.to_author(2), AuthorPosition("A", 12, ""))
        self.assertEqual(self.map.to_author(5), AuthorPosition("A", 15, ""))
        self.assertEqual(self.map.to_index(AuthorPosition("A", 15, "")), 5)
        self.assertEqual(self.map.to_index(15), 5)          # bare author number
        self.assertEqual(self.map.to_index((15, "")), 5)
        self.assertEqual(self.map.to_index(("A", 16, "")), 6)
        self.assertIsNone(self.map.to_index(13))            # not observed

    def test_unobserved_region_is_explicit(self) -> None:
        self.assertFalse(self.map.is_observed(3))
        self.assertFalse(self.map.is_observed(4))
        self.assertTrue(self.map.is_observed(2))
        self.assertIsNone(self.map.to_author(3))
        self.assertEqual(self.map.unobserved_regions(), [(3, 4)])
        self.assertAlmostEqual(self.map.coverage, 5 / 7)
        self.assertTrue(any("no coordinates" in n for n in self.map.notes))

    def test_index_out_of_range_raises(self) -> None:
        with self.assertRaises(NumberingError):
            self.map.to_author(7)
        with self.assertRaises(NumberingError):
            self.map.is_observed(-1)

    def test_terminal_gaps_are_reported_too(self) -> None:
        m = build_map("GGMKVWYGG", _chain_a())
        self.assertEqual(m.unobserved_regions(), [(0, 1), (7, 8)])

    def test_structure_residue_absent_from_candidate_is_reported(self) -> None:
        # A tag, a fusion partner, or -- most often -- the wrong chain.
        m = build_map("MKV", _chain_a())
        self.assertEqual(
            [str(p) for p in m.unmapped_structure_positions], ["A/15", "A/16"]
        )
        self.assertTrue(any("no counterpart" in n for n in m.notes))

    def test_sequence_mismatch_is_recorded_not_corrected(self) -> None:
        m = build_map("MKVQAFY", _chain_a())     # candidate F where PDB has W
        self.assertEqual(len(m.mismatches), 1)
        index, cand_letter, struct_letter, pos = m.mismatches[0]
        self.assertEqual((index, cand_letter, struct_letter), (5, "F", "W"))
        self.assertEqual(pos, AuthorPosition("A", 15, ""))
        self.assertTrue(any("variant or a homologue" in n for n in m.notes))

    def test_modified_residue_keeps_the_numbering(self) -> None:
        chain = read_pdb(PDB_WITH_MSE).chain("B")
        assert chain is not None
        m = build_map("AMG", chain)
        self.assertEqual(m.coverage, 1.0)
        self.assertEqual(m.to_author(1), AuthorPosition("B", 2, ""))
        self.assertTrue(any("modified residue" in n for n in m.notes))

    def test_describe_is_a_single_line(self) -> None:
        text = self.map.describe()
        self.assertNotIn("\n", text)
        self.assertIn("5/7", text)

    def test_empty_inputs_raise(self) -> None:
        with self.assertRaises(NumberingError):
            build_map("", _chain_a())
        with self.assertRaises(NumberingError):
            build_map("MKV", None)  # type: ignore[arg-type]


class TestReferenceNumbering(unittest.TestCase):
    def test_reference_axis_is_one_based(self) -> None:
        # The reference carries one extra N-terminal residue, so candidate
        # index 0 is reference position 2 -- exactly the kind of shift that
        # makes "the catalytic Tyr155" land on the wrong residue.
        m = build_map("MKVQAWY", _chain_a(),
                      reference_sequence="GMKVQAWY",
                      reference_label="SDR reference")
        self.assertEqual(m.to_reference(0), 2)
        self.assertEqual(m.to_reference(6), 8)
        self.assertEqual(m.from_reference(2), 0)
        self.assertIsNone(m.from_reference(1))      # candidate lacks the Gly
        self.assertEqual(m.reference_identity, 1.0)
        self.assertEqual(m.reference_label, "SDR reference")

    def test_no_reference_means_no_reference_mapping(self) -> None:
        m = build_map("MKVQAWY", _chain_a())
        self.assertIsNone(m.to_reference(0))
        self.assertIsNone(m.from_reference(1))

    def test_empty_reference_string_raises(self) -> None:
        with self.assertRaises(NumberingError):
            build_map("MKVQAWY", _chain_a(), reference_sequence="")


class TestVerifyResidue(unittest.TestCase):
    def setUp(self) -> None:
        self.map = build_map("MKVQAWY", _chain_a())

    def test_correct_wild_type_returns_the_author_position(self) -> None:
        self.assertEqual(verify_residue(self.map, 5, "W"),
                         AuthorPosition("A", 15, ""))
        self.assertEqual(verify_residue(self.map, 0, "m"),
                         AuthorPosition("A", 10, ""))

    def test_wrong_wild_type_letter_is_rejected(self) -> None:
        # The failure this whole module exists to prevent.
        with self.assertRaises(NumberingError) as cm:
            verify_residue(self.map, 5, "F")
        msg = str(cm.exception)
        self.assertIn("wild-type mismatch", msg)
        self.assertIn("A/15", msg)
        self.assertIn("W", msg)

    def test_off_by_one_is_caught(self) -> None:
        # Passing the author number 15 where a 0-based index was wanted.
        with self.assertRaises(NumberingError):
            verify_residue(self.map, 15, "W")
        # Passing index 6 (Y) while meaning index 5 (W).
        with self.assertRaises(NumberingError):
            verify_residue(self.map, 6, "W")

    def test_unobserved_position_is_allowed_unless_required(self) -> None:
        self.assertIsNone(verify_residue(self.map, 3, "Q"))
        with self.assertRaises(NumberingError) as cm:
            verify_residue(self.map, 3, "Q", require_observed=True)
        self.assertIn("no coordinates", str(cm.exception))

    def test_non_standard_expected_letter_is_rejected(self) -> None:
        for bad in ("X", "", "AL", "z"):
            with self.assertRaises(NumberingError):
                verify_residue(self.map, 0, bad)

    def test_mutation_proposal_style_use(self) -> None:
        # What a mutation generator does: resolve a template position through
        # the reference axis, verify the wild type, then name the mutation in
        # author numbering.
        m = build_map("MKVQAWY", _chain_a(),
                      reference_sequence="GMKVQAWY", reference_label="ref")
        index = m.from_reference(7)          # reference W
        assert index is not None
        pos = verify_residue(m, index, "W", require_observed=True)
        assert pos is not None
        self.assertEqual(f"W{pos.token}F", "W15F")


if __name__ == "__main__":  # pragma: no cover - pytest may not be installed
    unittest.main(verbosity=2)
