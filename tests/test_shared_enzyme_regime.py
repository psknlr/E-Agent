"""Tests for the shared-enzyme, novel-substrate split regime.

Why it exists. The three leakage-safe regimes group by lineage, which is
correct for them and has a side effect: an enzyme measured on a held-out
substrate drags all of its other measurements into the same fold, so a
substrate-scope split also becomes enzyme-disjoint. That answers a harder
question than "does this model handle a new substrate for an enzyme it knows",
which is the commonest substrate-scope question there is and could not be
expressed at all.

The regime is deliberately weaker, and the tests below pin both halves of
that: the substrate separation is real, and the enzyme overlap it permits is
reported as the regime's design rather than quietly excused for everything
else.
"""

from __future__ import annotations

import unittest

from eagent.eval.splits import (
    LeakageCategory, SplitRegime, audit_leakage, grouped_split,
)
from eagent.schemas import ExperimentRecord, OutcomeClass, SubstrateSpec

SEQUENCES = {
    "e1": "MKAIVTGASRGIGRAIAEELAKQGAKVVLNYSSNQ",
    "e2": "MTDKLSGKVALVTGGASGIGEATARLFAEHGAKVV",
}
SUBSTRATES = {
    "s1": "CC(=O)c1ccccc1",
    "s2": "CCCCC(=O)CC",
    "s3": "O=C1CCCCC1",
    "s4": "CC(=O)CCc1ccccc1",
}


def _grid() -> list[ExperimentRecord]:
    """Every enzyme measured on every substrate: the arrangement at issue."""
    return [
        ExperimentRecord(
            record_id=f"{e}_{s}", sequence=seq,
            substrate=SubstrateSpec(isomeric_smiles=smiles),
            outcome=OutcomeClass.NOT_TESTED)
        for e, seq in SEQUENCES.items()
        for s, smiles in SUBSTRATES.items()
    ]


def _sides(split):
    train = set(split.train_record_ids)
    test = set(split.test_record_ids)
    return train, test


def _enzymes(ids):
    return {i.split("_")[0] for i in ids}


def _substrates(ids):
    return {i.split("_")[1] for i in ids}


class RegimeSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.regime = SplitRegime.SHARED_ENZYME_NOVEL_SUBSTRATE

    def test_it_holds_out_substrates_but_not_sequences(self) -> None:
        self.assertTrue(self.regime.holds_out_substrates)
        self.assertFalse(self.regime.holds_out_sequences)

    def test_it_declares_itself_weaker(self) -> None:
        self.assertTrue(self.regime.permits_shared_enzymes)
        self.assertTrue(self.regime.is_weaker_than_leakage_safe)
        for other in (SplitRegime.NOVEL_ENZYME, SplitRegime.NOVEL_SUBSTRATE,
                      SplitRegime.DUAL_EXTRAPOLATION):
            self.assertFalse(other.permits_shared_enzymes, other)

    def test_its_claim_states_the_qualifier(self) -> None:
        claim = self.regime.claim()
        self.assertIn("already seen", claim)
        self.assertIn("weaker", claim)

    def test_enzyme_overlap_is_expected_but_publication_overlap_is_not(self) -> None:
        """A regime may excuse its own design, never someone else's leak."""
        for excused in (LeakageCategory.SHARED_SEQUENCE_CLUSTER,
                        LeakageCategory.SHARED_PARENT_LINEAGE,
                        LeakageCategory.SHARED_SPLIT_GROUP):
            self.assertTrue(excused.is_expected_under(self.regime), excused)
        for never in (LeakageCategory.SHARED_PUBLICATION,
                      LeakageCategory.RECURATED_SOURCE,
                      LeakageCategory.SHARED_SCAFFOLD):
            self.assertFalse(never.is_expected_under(self.regime), never)


class SplitBehaviourTests(unittest.TestCase):
    def test_substrates_are_disjoint_and_enzymes_are_shared(self) -> None:
        split = grouped_split(_grid(), SplitRegime.SHARED_ENZYME_NOVEL_SUBSTRATE,
                              test_fraction=0.5, seed=1)
        train, test = _sides(split)
        self.assertTrue(train and test)
        self.assertEqual(_substrates(train) & _substrates(test), set(),
                         "the substrate separation must be real")
        self.assertTrue(_enzymes(train) & _enzymes(test),
                        "the shared enzyme is the point of this regime")

    def test_the_strict_regime_separates_enzymes_too(self) -> None:
        """The contrast that motivates the new regime existing."""
        strict = grouped_split(_grid(), SplitRegime.NOVEL_SUBSTRATE,
                               test_fraction=0.5, seed=1)
        train, test = _sides(strict)
        if train and test:
            self.assertEqual(
                _enzymes(train) & _enzymes(test), set(),
                "grouping by lineage makes a substrate split enzyme-disjoint; "
                "that is correct for this regime and is why the weaker one "
                "had to be added rather than loosening this one")

    def test_the_split_records_why_enzymes_appear_on_both_sides(self) -> None:
        split = grouped_split(_grid(), SplitRegime.SHARED_ENZYME_NOVEL_SUBSTRATE,
                              test_fraction=0.5, seed=1)
        text = " ".join(split.notes)
        self.assertIn("both sides", text)
        self.assertIn("not evidence about a new enzyme", text)

    def test_it_is_deterministic_for_a_seed_and_varies_across_seeds(self) -> None:
        a = grouped_split(_grid(), SplitRegime.SHARED_ENZYME_NOVEL_SUBSTRATE,
                          test_fraction=0.5, seed=1)
        b = grouped_split(_grid(), SplitRegime.SHARED_ENZYME_NOVEL_SUBSTRATE,
                          test_fraction=0.5, seed=1)
        self.assertEqual(a.test_record_ids, b.test_record_ids)
        draws = {
            tuple(grouped_split(_grid(),
                                SplitRegime.SHARED_ENZYME_NOVEL_SUBSTRATE,
                                test_fraction=0.5, seed=s).test_record_ids)
            for s in range(6)
        }
        self.assertGreater(len(draws), 1, "every seed gave the same fold")

    def test_it_needs_no_sequence_clustering(self) -> None:
        """It makes no claim about unseen enzymes, so it demands no clustering."""
        split = grouped_split(_grid(), SplitRegime.SHARED_ENZYME_NOVEL_SUBSTRATE,
                              test_fraction=0.5, seed=2)
        self.assertTrue(split.test_record_ids)

    def test_the_audit_does_not_call_a_shared_scaffold_expected_here(self) -> None:
        """If a scaffold did straddle the boundary, that is still leakage."""
        rows = _grid()
        train = [r for r in rows if r.record_id.endswith(("s1", "s2"))]
        test = [r for r in rows if r.record_id.endswith("s1")]
        audit = audit_leakage(
            train, test, regime=SplitRegime.SHARED_ENZYME_NOVEL_SUBSTRATE)
        scaffold_overlaps = [o for o in audit.overlaps
                             if o.category is LeakageCategory.SHARED_SCAFFOLD]
        self.assertTrue(scaffold_overlaps,
                        "a shared substrate is leakage under this regime too")


if __name__ == "__main__":
    unittest.main(verbosity=2)
