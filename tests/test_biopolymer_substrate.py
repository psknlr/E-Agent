"""Tests for the biopolymer substrate path and its gate requirements.

The failure these guard against is a task that can never open a gate. A
30-mer peptide substrate has no isomeric SMILES anybody will write and no
atom-map id for the residue a kinase phosphorylates, so a system offering only
the small-molecule path either blocks such a task forever on a structure it
cannot have, or lets a prose name stand in for the substrate.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from eagent.schemas import (
    BiopolymerSubstrateSpec, ReactionSpec, ReactiveResidues, ResidueRef,
    SubstrateKind, SubstrateSpec, TaskSpec,
)


class TestSubstrateKind(unittest.TestCase):
    def test_small_molecule_is_not_a_biopolymer(self) -> None:
        self.assertFalse(SubstrateKind.SMALL_MOLECULE.is_biopolymer)
        for k in (SubstrateKind.PEPTIDE, SubstrateKind.PROTEIN,
                  SubstrateKind.NUCLEIC_ACID):
            self.assertTrue(k.is_biopolymer)

    def test_nucleic_acid_has_its_own_alphabet(self) -> None:
        self.assertEqual(SubstrateKind.NUCLEIC_ACID.alphabet, "ACGTU")
        self.assertIn("W", SubstrateKind.PROTEIN.alphabet)
        self.assertNotIn("W", SubstrateKind.NUCLEIC_ACID.alphabet)


class TestPathsDoNotMix(unittest.TestCase):
    def test_a_biopolymer_kind_is_refused_on_the_small_molecule_type(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            SubstrateSpec(kind=SubstrateKind.PROTEIN)
        self.assertIn("BiopolymerSubstrateSpec", str(ctx.exception))

    def test_a_small_molecule_is_refused_on_the_biopolymer_type(self) -> None:
        with self.assertRaises(ValidationError):
            BiopolymerSubstrateSpec(kind=SubstrateKind.SMALL_MOLECULE)


class TestBiopolymerValidation(unittest.TestCase):
    def test_sequence_is_hashed_measured_and_normalised(self) -> None:
        p = BiopolymerSubstrateSpec(kind=SubstrateKind.PEPTIDE,
                                    sequence=" acdefg hiky ")
        self.assertEqual(p.sequence, "ACDEFGHIKY")
        self.assertEqual(p.length, 10)
        self.assertTrue(p.sequence_sha256.startswith("sha256:"))
        self.assertTrue(p.is_structurally_defined)

    def test_a_reactive_residue_that_disagrees_with_the_sequence_is_refused(self) -> None:
        """A drifted position silently points at a neighbour; catch it here."""
        with self.assertRaises(ValidationError) as ctx:
            BiopolymerSubstrateSpec(
                kind=SubstrateKind.PEPTIDE, sequence="ACDEFGHIKY",
                reactive_residues=ReactiveResidues(
                    modified=[ResidueRef(position=10, residue="W")]))
        self.assertIn("numbering is wrong", str(ctx.exception))

    def test_a_reactive_residue_beyond_the_substrate_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            BiopolymerSubstrateSpec(
                kind=SubstrateKind.PEPTIDE, sequence="ACDEF",
                reactive_residues=ReactiveResidues(
                    modified=[ResidueRef(position=99)]))
        self.assertIn("beyond", str(ctx.exception))

    def test_a_matching_reactive_residue_is_accepted(self) -> None:
        p = BiopolymerSubstrateSpec(
            kind=SubstrateKind.PEPTIDE, sequence="ACDEFGHIKY",
            reactive_residues=ReactiveResidues(
                modified=[ResidueRef(position=10, residue="Y",
                                     role="phosphoacceptor")]))
        self.assertEqual(p.reactive_residues.modified[0].role, "phosphoacceptor")

    def test_off_alphabet_residues_need_a_recorded_modification(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            BiopolymerSubstrateSpec(kind=SubstrateKind.NUCLEIC_ACID,
                                    sequence="ACGTX")
        self.assertIn("alphabet", str(ctx.exception))
        ok = BiopolymerSubstrateSpec(kind=SubstrateKind.NUCLEIC_ACID,
                                     sequence="ACGTX",
                                     modifications=["X = abasic site"])
        self.assertEqual(ok.length, 5)


class TestGateRequirementsFollowTheKind(unittest.TestCase):
    def test_small_molecule_requirements_are_unchanged(self) -> None:
        t = TaskSpec(task_id="SMALL")
        self.assertEqual(
            t.unresolved_for("reaction_spec_confirmed"),
            ["reaction.substrate.isomeric_smiles",
             "reaction.product.isomeric_smiles",
             "reaction.product.creates_new_stereocenter",
             "reaction.atom_mapped_reaction_smiles"])

    def test_a_peptide_task_is_never_asked_for_a_smiles(self) -> None:
        t = TaskSpec(task_id="PEP", reaction=ReactionSpec(
            biopolymer_substrate=BiopolymerSubstrateSpec(
                kind=SubstrateKind.PEPTIDE)))
        unresolved = t.unresolved_for("reaction_spec_confirmed")
        self.assertTrue(unresolved, "a blank peptide task has unresolved fields")
        self.assertFalse([p for p in unresolved if "smiles" in p.lower()])
        self.assertIn("reaction.biopolymer_substrate.sequence", unresolved)

    def test_a_filled_peptide_task_can_actually_open_the_gate(self) -> None:
        """The point of the second path: this gate must be reachable."""
        t = TaskSpec(task_id="PEP", reaction=ReactionSpec(
            biopolymer_substrate=BiopolymerSubstrateSpec(
                kind=SubstrateKind.PEPTIDE, sequence="ACDEFGHIKY",
                reactive_residues=ReactiveResidues(
                    modified=[ResidueRef(position=10, residue="Y")]))))
        t.reaction.product.name = "phospho-peptide"
        self.assertEqual(t.unresolved_for("reaction_spec_confirmed"), [])

    def test_an_empty_reactive_residue_list_counts_as_unresolved(self) -> None:
        """An empty list is not an answer; it is a field nobody filled in."""
        t = TaskSpec(task_id="PEP", reaction=ReactionSpec(
            biopolymer_substrate=BiopolymerSubstrateSpec(
                kind=SubstrateKind.PEPTIDE, sequence="ACDEFGHIKY")))
        t.reaction.product.name = "phospho-peptide"
        self.assertIn("reaction.biopolymer_substrate.reactive_residues.modified",
                      t.unresolved_for("reaction_spec_confirmed"))

    def test_substrate_kind_reports_the_path_in_use(self) -> None:
        self.assertIs(ReactionSpec().substrate_kind, SubstrateKind.SMALL_MOLECULE)
        r = ReactionSpec(biopolymer_substrate=BiopolymerSubstrateSpec(
            kind=SubstrateKind.NUCLEIC_ACID))
        self.assertIs(r.substrate_kind, SubstrateKind.NUCLEIC_ACID)


if __name__ == "__main__":
    unittest.main(verbosity=2)
