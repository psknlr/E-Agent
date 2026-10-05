"""Tests that the scaffold key keeps the reacting group and drops decoration.

The two-core alone reduced acetophenone, 4-chloroacetophenone and plain
benzene to one ring and therefore to one scaffold key. In a
carbonyl-reduction campaign that collapses almost the whole substrate axis of
a split: holding out "aromatic ketones" would also hold out every other
six-ring compound, and holding out nothing useful is indistinguishable from
having no substrate regime at all.

The fix has to satisfy three properties at once, and the obvious repairs each
break one of them:

* the reacting group must survive into the key, so a ketone is not the same
  scaffold as the bare ring it hangs off;
* a peripheral substituent must NOT create a new scaffold, or a chloro
  analogue of the training substrate counts as a novel substrate;
* every spelling of one molecule must give one key, or the same compound
  lands on both sides of the split, which is leakage.
"""

from __future__ import annotations

import unittest

from eagent.eval.splits import ScaffoldBasis, scaffold_key

BENZENE_AROMATIC = "c1ccccc1"
BENZENE_KEKULE = "C1=CC=CC=C1"
CYCLOHEXANE = "C1CCCCC1"
ACETOPHENONE = "CC(=O)c1ccccc1"
ACETOPHENONE_KEKULE = "CC(=O)C1=CC=CC=C1"
CHLORO_ACETOPHENONE = "CC(=O)c1ccc(Cl)cc1"
BROMO_ACETOPHENONE = "CC(=O)c1ccc(Br)cc1"
PROPIOPHENONE = "CCC(=O)c1ccccc1"
NAPHTHALENE = "c1ccc2ccccc2c1"


def key(smiles: str) -> str:
    return scaffold_key(smiles).key


class SpellingInvarianceTests(unittest.TestCase):
    """The property no improvement may cost: one molecule, one key."""

    def test_aromatic_and_kekulised_benzene_agree(self) -> None:
        self.assertEqual(key(BENZENE_AROMATIC), key(BENZENE_KEKULE))

    def test_aromatic_and_kekulised_acetophenone_agree(self) -> None:
        self.assertEqual(key(ACETOPHENONE), key(ACETOPHENONE_KEKULE))

    def test_the_key_is_stable_across_repeated_calls(self) -> None:
        self.assertEqual(key(ACETOPHENONE), key(ACETOPHENONE))


class FunctionalGroupSurvivesTests(unittest.TestCase):
    def test_a_ketone_is_not_the_same_scaffold_as_its_bare_ring(self) -> None:
        """The regression: the two-core used to throw the carbonyl away."""
        self.assertNotEqual(key(ACETOPHENONE), key(BENZENE_AROMATIC))

    def test_a_phenol_is_not_the_same_scaffold_as_benzene(self) -> None:
        self.assertNotEqual(key("Oc1ccccc1"), key(BENZENE_AROMATIC))

    def test_an_aniline_is_not_the_same_scaffold_as_benzene(self) -> None:
        self.assertNotEqual(key("Nc1ccccc1"), key(BENZENE_AROMATIC))


class PeripheralSubstituentsAreIgnoredTests(unittest.TestCase):
    def test_a_halogenated_analogue_shares_the_scaffold(self) -> None:
        """A chloro analogue of the training substrate is not a new scaffold."""
        self.assertEqual(key(ACETOPHENONE), key(CHLORO_ACETOPHENONE))

    def test_different_halogens_share_the_scaffold(self) -> None:
        self.assertEqual(key(CHLORO_ACETOPHENONE), key(BROMO_ACETOPHENONE))

    def test_a_longer_alkyl_chain_shares_the_scaffold(self) -> None:
        self.assertEqual(key(ACETOPHENONE), key(PROPIOPHENONE))


class RingSystemStillMattersTests(unittest.TestCase):
    def test_a_fused_ring_system_differs_from_a_single_ring(self) -> None:
        self.assertNotEqual(key(NAPHTHALENE), key(BENZENE_AROMATIC))

    def test_a_heterocycle_differs_from_the_carbocycle(self) -> None:
        self.assertNotEqual(key("c1ccncc1"), key(BENZENE_AROMATIC))


class DocumentedLimitsTests(unittest.TestCase):
    """Limits that are real, conservative, and stated rather than hidden."""

    def test_aromaticity_is_not_perceived_so_same_composition_rings_collide(self) -> None:
        self.assertEqual(key(BENZENE_AROMATIC), key(CYCLOHEXANE))

    def test_the_collision_is_declared_in_the_key_s_limitations(self) -> None:
        limits = " ".join(scaffold_key(BENZENE_AROMATIC).limitations).lower()
        self.assertIn("aromatic", limits)
        self.assertIn("collide", limits)

    def test_the_limitations_state_what_the_key_does_keep(self) -> None:
        limits = " ".join(scaffold_key(ACETOPHENONE).limitations).lower()
        self.assertIn("carbonyl", limits)
        self.assertIn("halogen", limits)

    def test_over_grouping_is_the_direction_it_errs_in(self) -> None:
        """Two unlike things sharing a key costs data; it cannot inflate a score."""
        self.assertEqual(scaffold_key(ACETOPHENONE).basis,
                         ScaffoldBasis.RING_AND_LINKER_SKELETON)

    def test_an_unparseable_substrate_is_unresolved_not_guessed(self) -> None:
        result = scaffold_key("this is not a smiles {{{")
        self.assertFalse(result.resolved)
        self.assertTrue(result.needs_curator)


if __name__ == "__main__":
    unittest.main(verbosity=2)
