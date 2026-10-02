"""Tests that a re-curation split is not reported as clean.

"Train on database A, test on database B" reads like a clean separation and is
not one when B re-published A. Before an evidence reference recorded the
resource it was read from, the audit could only see this if the caller handed
it a lookup, so the default behaviour was to call such a split clean.
"""

from __future__ import annotations

import pathlib
import unittest

from eagent.datalayer.registry import SourceRegistry
from eagent.eval.splits import LeakageCategory, audit_leakage, source_tokens_of
from eagent.schemas import (
    EvidenceRef, ExperimentRecord, OutcomeClass, SubstrateSpec,
)

CONFIGS = pathlib.Path(__file__).resolve().parent.parent / "configs" / "datasources"

# Distinct sequences and substrates, so the only thing the two sides can share
# is their provenance. Otherwise a shared scaffold would mask the result.
SEQ_A = "MKAIVTGASRGIGRAIAEELAKQGAKVVLNY"
SEQ_B = "MTDKLSGKVALVTGGASGIGEATARLFAEHG"
SMILES_A = "CC(=O)c1ccccc1"
SMILES_B = "CCCCC(=O)CC"


def _record(rid: str, *, source_id: str, upstream: list[str],
            sequence: str, smiles: str) -> ExperimentRecord:
    return ExperimentRecord(
        record_id=rid,
        sequence=sequence,
        substrate=SubstrateSpec(isomeric_smiles=smiles),
        outcome=OutcomeClass.NOT_TESTED,
        evidence=[EvidenceRef(source_type="database", identifier=rid,
                              source_id=source_id,
                              upstream_sources=list(upstream))],
    )


class SourceTokenTests(unittest.TestCase):
    def test_a_row_reports_both_its_source_and_its_upstreams(self) -> None:
        row = _record("r", source_id="oed", upstream=["brenda", "sabio_rk"],
                      sequence=SEQ_A, smiles=SMILES_A)
        self.assertEqual(sorted(source_tokens_of(row)),
                         ["brenda", "oed", "sabio_rk"])

    def test_a_row_with_no_recorded_source_reports_nothing_invented(self) -> None:
        row = ExperimentRecord(record_id="r", sequence=SEQ_A,
                               substrate=SubstrateSpec(isomeric_smiles=SMILES_A),
                               outcome=OutcomeClass.NOT_TESTED)
        self.assertEqual(source_tokens_of(row), set())


class RecurationAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = SourceRegistry.from_directory(CONFIGS)

    def _audit(self, train_source, train_upstream, test_source, test_upstream):
        train = [_record("t1", source_id=train_source, upstream=train_upstream,
                         sequence=SEQ_A, smiles=SMILES_A)]
        test = [_record("s1", source_id=test_source, upstream=test_upstream,
                        sequence=SEQ_B, smiles=SMILES_B)]
        return audit_leakage(train, test, registry=self.registry)

    def _recuration(self, audit):
        return [o for o in audit.overlaps
                if o.category is LeakageCategory.RECURATED_SOURCE]

    def test_training_on_brenda_and_testing_on_oed_is_not_clean(self) -> None:
        """The headline case, detected with no caller-supplied lookup."""
        audit = self._audit("brenda", [], "oed", ["brenda", "sabio_rk"])
        self.assertTrue(self._recuration(audit),
                        "OED re-curated BRENDA; this split shares rows")
        self.assertFalse(audit.proven_clean)

    def test_the_symmetric_direction_is_caught_too(self) -> None:
        audit = self._audit("oed", ["brenda", "sabio_rk"], "brenda", [])
        self.assertTrue(self._recuration(audit))

    def test_two_sources_sharing_an_upstream_are_caught(self) -> None:
        """Neither re-published the other; both re-published a third."""
        audit = self._audit("oed", ["brenda"], "catpred_db", ["brenda"])
        self.assertTrue(self._recuration(audit))

    def test_genuinely_unrelated_sources_raise_no_recuration_overlap(self) -> None:
        """The check must not fire on every pair, or it says nothing."""
        audit = self._audit("fireprotdb", [], "mavedb", [])
        self.assertEqual(self._recuration(audit), [])

    def test_a_split_with_no_recorded_provenance_is_not_called_clean(self) -> None:
        """Unknown provenance is unchecked, which is not the same as clean."""
        train = [ExperimentRecord(record_id="t", sequence=SEQ_A,
                                  substrate=SubstrateSpec(isomeric_smiles=SMILES_A),
                                  outcome=OutcomeClass.NOT_TESTED)]
        test = [ExperimentRecord(record_id="s", sequence=SEQ_B,
                                 substrate=SubstrateSpec(isomeric_smiles=SMILES_B),
                                 outcome=OutcomeClass.NOT_TESTED)]
        audit = audit_leakage(train, test, registry=self.registry)
        self.assertEqual(self._recuration(audit), [])
        self.assertTrue(
            audit.unchecked or audit.notes,
            "a split whose provenance was never recorded must say the "
            "independence of its sources was not checked")


if __name__ == "__main__":
    unittest.main(verbosity=2)
