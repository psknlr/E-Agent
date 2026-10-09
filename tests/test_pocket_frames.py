"""Regression tests: a numbering offset is not a structural difference.

Two ketoreductases with the same pocket, one deposited with an extra
N-terminal methionine. Every author number in the second is one higher. The
diversity layer compared ``F98`` with ``F99``, found nothing in common, and
returned a Jaccard distance of 1.0 -- maximally different, for two proteins
that would behave identically toward the substrate.

That number is spent: a batch composed to maximise pocket diversity buys two
slots for one experiment, and the round reports pocket coverage it did not
achieve. Positions are only comparable inside a frame both sides have been
mapped into, so a pocket now carries the frame it is numbered in, and a
comparison across frames is refused instead of computed.
"""

from __future__ import annotations

import unittest

from eagent.science.diversity import (
    SignatureBasis, pocket_distance, pocket_signature,
)
from eagent.science.family_numbering import FamilyNumberingScheme
from eagent.science.numbering import build_map
from eagent.science.pocket import (
    OWN_NUMBERING, best_chain_map, pocket_residues_for_pose,
    pocket_residues_from_shell, pocket_residues_from_tokens,
)
from eagent.science.structure_io import Atom, Chain, Residue, Structure

from test_diversity import candidate

SEQ = ("MKAAVLYEFGKPLEIKEVEVAPPKAHEVRIKIAYTGVCHTDAYTLSGADPEGLFPVILGHEGAGIVESVGEGVT"
       "NVKPGDHVIPLYTPECGECKFCKSGKTNLCQKIRATQGKGLMPDGTSRFTCKGKEILHYMGCSTFSEYTVVAD")

THREE = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
         "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
         "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
         "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"}


def atom(serial, name, element, resname, chain, resseq, x, y, z,
         hetatm=False) -> Atom:
    return Atom(serial=serial, name=name, element=element, resname=resname,
                chain=chain, resseq=resseq, icode="", altloc="",
                x=x, y=y, z=z, occupancy=1.0, bfactor_or_plddt=80.0,
                is_hetatm=hetatm)


def protein_chain(sequence: str, first_number: int, chain_id: str = "A") -> Chain:
    """One CA per residue, laid out on a line 3.8 A apart."""
    residues = []
    for n, letter in enumerate(sequence):
        number = first_number + n
        residues.append(Residue(
            chain=chain_id, resname=THREE[letter], resseq=number, icode="",
            atoms=[atom(number, "CA", "C", THREE[letter], chain_id, number,
                        3.8 * n, 0.0, 0.0)],
            is_hetatm=False))
    return Chain(chain_id=chain_id, residues=residues)


def ligand_at(x: float) -> Residue:
    return Residue(chain="L", resname="LIG", resseq=900, icode="",
                   atoms=[atom(9000, "C1", "C", "LIG", "L", 900, x, 1.0, 0.0,
                               hetatm=True)],
                   is_hetatm=True)


def structure_with(sequence: str, first_number: int, ligand_x: float) -> Structure:
    return Structure(
        structure_id="s", source_format="pdb",
        chains=[protein_chain(sequence, first_number),
                Chain(chain_id="L", residues=[ligand_at(ligand_x)])])


def scheme() -> FamilyNumberingScheme:
    return FamilyNumberingScheme(
        family_name="SDR", scheme_id="sdr.test.v1",
        reference_label="synthetic reference",
        reference_sequence=SEQ,
        reference_source="constructed in tests/test_pocket_frames.py",
        min_identity=0.25)


class ShellToFrameTests(unittest.TestCase):
    """The same pocket in two numberings must come out the same."""

    def extract(self, first_number: int, with_scheme: bool):
        structure = structure_with(SEQ, first_number, ligand_x=3.8 * 20)
        pocket, note = pocket_residues_for_pose(
            "c", SEQ, structure, ligand_at(3.8 * 20).atoms,
            max_angstrom=6.0, scheme=scheme() if with_scheme else None)
        self.assertIsNotNone(pocket, note)
        assert pocket is not None
        return pocket

    def test_without_a_scheme_the_frame_is_this_candidate_s_own(self) -> None:
        pocket = self.extract(1, with_scheme=False)
        self.assertEqual(pocket.frame, OWN_NUMBERING)
        self.assertFalse(pocket.positional)

    def test_without_a_scheme_an_offset_changes_every_token(self) -> None:
        """The premise: this is why a frame is needed, not a bug introduced here."""
        self.assertNotEqual(self.extract(1, with_scheme=False).tokens,
                            self.extract(2, with_scheme=False).tokens)

    def test_with_a_scheme_an_offset_changes_nothing(self) -> None:
        self.assertEqual(self.extract(1, with_scheme=True).tokens,
                         self.extract(2, with_scheme=True).tokens)

    def test_with_a_scheme_the_frame_names_the_family(self) -> None:
        pocket = self.extract(1, with_scheme=True)
        self.assertEqual(pocket.frame, "SDR/sdr.test.v1")
        self.assertTrue(pocket.positional)

    def test_the_pocket_is_not_empty(self) -> None:
        """A test that compares two empty sets would pass for the wrong reason."""
        self.assertTrue(self.extract(1, with_scheme=True).tokens)

    def test_a_residue_outside_the_candidate_is_reported_not_dropped(self) -> None:
        tagged = structure_with(SEQ, 1, ligand_x=3.8 * 20)
        chain = tagged.chains[0]
        # A purification tag the candidate sequence does not contain, sitting
        # right next to the ligand so the shell picks it up.
        extra = [Residue(chain="A", resname="HIS", resseq=500 + n, icode="",
                         atoms=[atom(500 + n, "CA", "C", "HIS", "A", 500 + n,
                                     3.8 * 20, 1.5, 0.0)],
                         is_hetatm=False)
                 for n in range(6)]
        chain.residues.extend(extra)
        residue_map, _ = best_chain_map(SEQ, tagged)
        assert residue_map is not None
        pocket = pocket_residues_from_shell(
            "c", SEQ, residue_map, chain.residues, scheme=None)
        self.assertTrue(pocket.unmapped)
        self.assertTrue([n for n in pocket.notes if "not in the candidate" in n])

    def test_no_ligand_is_an_unmeasured_pocket_not_an_empty_one(self) -> None:
        pocket, note = pocket_residues_for_pose(
            "c", SEQ, structure_with(SEQ, 1, 0.0), [], max_angstrom=6.0)
        self.assertIsNone(pocket)
        self.assertIn("no ligand atoms", note)

    def test_the_candidate_s_own_chain_is_chosen_over_a_partner(self) -> None:
        mixed = structure_with(SEQ, 1, 0.0)
        other = protein_chain("WWWWWWWWWWWWWWWWWWWW", 1, chain_id="B")
        mixed.chains.append(other)
        residue_map, note = best_chain_map(SEQ, mixed)
        assert residue_map is not None
        self.assertEqual(residue_map.chain_id, "A")
        self.assertIn("chain A chosen", note)


class SignatureFrameTests(unittest.TestCase):
    """A distance across two frames means nothing and is not returned."""

    def test_an_offset_no_longer_makes_two_pockets_maximally_distant(self) -> None:
        a = candidate("a", pocket={"catalytic_Tyr": "Y155"})
        b = candidate("b", pocket={"catalytic_Tyr": "Y156"})
        self.assertEqual(pocket_distance(a, b), 0.0)

    def test_a_real_catalytic_substitution_is_still_a_difference(self) -> None:
        a = candidate("a", pocket={"catalytic_Tyr": "Y155"})
        b = candidate("b", pocket={"catalytic_Tyr": "F155"})
        distance = pocket_distance(a, b)
        assert distance is not None
        self.assertGreater(distance, 0.0)

    def test_two_pockets_in_one_frame_compare(self) -> None:
        a = pocket_residues_from_tokens("a", ["W110", "F147"], frame="SDR/v1")
        b = pocket_residues_from_tokens("b", ["W110", "A147"], frame="SDR/v1")
        distance = pocket_distance(candidate("a"), candidate("b"),
                                   extra_pocket_residues={"a": a, "b": b})
        assert distance is not None
        self.assertGreater(distance, 0.0)
        self.assertLess(distance, 1.0)

    def test_two_pockets_in_different_frames_are_incomparable(self) -> None:
        a = pocket_residues_from_tokens("a", ["W110"], frame="SDR/v1")
        b = pocket_residues_from_tokens("b", ["W110"], frame="AKR/v1")
        self.assertIsNone(pocket_distance(
            candidate("a"), candidate("b"), extra_pocket_residues={"a": a, "b": b}))

    def test_a_framed_and_an_unframed_pocket_are_incomparable(self) -> None:
        a = pocket_residues_from_tokens("a", ["W110"], frame="SDR/v1")
        b = pocket_residues_from_tokens("b", ["W110"])
        self.assertIsNone(pocket_distance(
            candidate("a"), candidate("b"), extra_pocket_residues={"a": a, "b": b}))

    def test_composition_counts_multiplicity(self) -> None:
        """Three tyrosines and one tyrosine are not the same pocket."""
        a = pocket_residues_from_tokens("a", ["Y10", "Y20", "Y30"])
        b = pocket_residues_from_tokens("b", ["Y10"])
        sa = pocket_signature(candidate("a"), extra_pocket_residues={"a": a})
        sb = pocket_signature(candidate("b"), extra_pocket_residues={"b": b})
        self.assertIs(sa.basis, SignatureBasis.POCKET_COMPOSITION)
        distance = pocket_distance(candidate("a"), candidate("b"),
                                   extra_pocket_residues={"a": a, "b": b})
        assert distance is not None
        self.assertGreater(distance, 0.0)

    def test_the_unplaced_count_travels_with_the_signature(self) -> None:
        from eagent.science.pocket import PocketResidues
        shell = PocketResidues("a", ("Y10",), frame="SDR/v1",
                               unmapped=("F99", "W120"))
        sig = pocket_signature(candidate("a"), extra_pocket_residues={"a": shell})
        self.assertEqual(sig.n_unplaced, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
