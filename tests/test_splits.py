"""Tests for :mod:`eagent.eval.splits`.

Every expected value here follows from how the fixture was built rather than
from running the code and writing down what it printed. The central fixtures
are the three ways a retrospective enzyme benchmark leaks:

* a parent and its variants, which must land in one fold;
* one publication appearing on both sides;
* "train on database A, test on database B" where B re-curated A.

A regression that starts calling any of those a clean split fails here instead
of redefining what clean means.

Runs under pytest, or standalone with ``python3 tests/test_splits.py``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - standalone execution
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from eagent.datalayer import AccessMode, DataLayer, DataSource, SourceRegistry
from eagent.eval.splits import (
    LeakageCategory,
    ScaffoldBasis,
    SCAFFOLD_KEY_ALGORITHM,
    SplitNotPossibleError,
    SplitRegime,
    audit_leakage,
    grouped_split,
    rdkit_available,
    scaffold_key,
    source_tokens_of,
)
from eagent.provenance import sequence_hash
from eagent.schemas.chem import SubstrateSpec
from eagent.schemas.record import EvidenceRef, ExperimentRecord

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

BASE = ("MKAAVLYEFGKPLEIKEVEVAPPKAHEVRIKIAYTGVCHTDAYTLSGADPEGLFPVILGHEGAGIVES"
        "VGEGVTNVKPGDHVIPLYTPECGECKFCKSGKTNLCQKIRATQGKGLMPDGTSRFTCKGKEILHYMG")

#: Reserved 10.5555 test prefix, and suffixes that cannot be read as a title:
#: a plausible-looking DOI in a fixture is eventually printed as a citation to
#: a paper that does not exist.
DOI_ONE = "10.5555/example-split-fixture-one"
DOI_TWO = "10.5555/example-split-fixture-two"

ACETOPHENONE = "CC(=O)c1ccccc1"
CHLOROACETOPHENONE = "CC(=O)c1ccc(Cl)cc1"
NAPHTHYL_KETONE = "CC(=O)c1ccc2ccccc2c1"
HEXANONE = "CCCC(=O)CC"


def sequence_for(tag: str) -> str:
    """A distinct sequence per tag, same length, so hashes differ by one letter."""
    letters = "ACDEFGHIKLMNPQRSTVWY"
    index = sum(ord(c) for c in tag) % (len(BASE) - 1)
    replacement = letters[sum(ord(c) for c in tag) % len(letters)]
    return BASE[:index] + replacement + BASE[index + 1:]


def record(
    record_id: str,
    *,
    sequence: str | None = None,
    parent_sequence: str | None = None,
    doi: str | None = None,
    activity: str | None = None,
    smiles: str | None = ACETOPHENONE,
    inchikey: str | None = None,
    substrate_name: str | None = None,
    upstream: tuple[str, ...] = (),
    database_id: str | None = None,
) -> ExperimentRecord:
    """One record with exactly the facets a test means to plant."""
    evidence: list[EvidenceRef] = []
    if doi or activity:
        evidence.append(EvidenceRef(
            source_type="publication",
            identifier=doi or f"activity-only:{record_id}",
            source_doi=doi,
            experiment_activity_id=activity,
        ))
    if database_id or upstream:
        evidence.append(EvidenceRef(
            source_type="database",
            identifier=database_id or f"acc:{record_id}",
            upstream_sources=list(upstream),
        ))
    seq = sequence if sequence is not None else sequence_for(record_id)
    return ExperimentRecord(
        record_id=record_id,
        sequence=seq,
        is_variant=parent_sequence is not None,
        parent_sequence_sha256=(sequence_hash(parent_sequence)
                                if parent_sequence else None),
        substrate=SubstrateSpec(name=substrate_name, isomeric_smiles=smiles,
                                inchikey=inchikey),
        evidence=evidence,
    )


def cluster_lookup(records) -> dict[str, str]:
    """One cluster per record's own sequence, named after the record id."""
    out: dict[str, str] = {}
    for rec in records:
        out[rec.sequence_sha256] = f"clu-{rec.record_id}"
    return out


# --------------------------------------------------------------------------
# scaffolds without rdkit
# --------------------------------------------------------------------------

class TestScaffoldKey(unittest.TestCase):
    """The pure-Python key must not split one molecule across the boundary."""

    def test_this_environment_has_no_rdkit_and_the_import_is_guarded(self) -> None:
        self.assertFalse(rdkit_available())
        key = scaffold_key(ACETOPHENONE)
        self.assertEqual(key.basis, ScaffoldBasis.RING_AND_LINKER_SKELETON)
        self.assertIn(SCAFFOLD_KEY_ALGORITHM, key.key or "")

    def test_aromatic_and_kekulised_spellings_give_one_key(self) -> None:
        """The same molecule written two ways must not land in two folds."""
        self.assertEqual(scaffold_key("CC(=O)c1ccccc1").key,
                         scaffold_key("CC(=O)C1=CC=CC=C1").key)

    def test_a_substituted_analogue_shares_the_core(self) -> None:
        self.assertEqual(scaffold_key(ACETOPHENONE).key,
                         scaffold_key(CHLOROACETOPHENONE).key)

    def test_a_different_ring_system_gives_a_different_key(self) -> None:
        self.assertNotEqual(scaffold_key(ACETOPHENONE).key,
                            scaffold_key(NAPHTHYL_KETONE).key)

    def test_a_counterion_does_not_change_the_scaffold(self) -> None:
        self.assertEqual(scaffold_key(ACETOPHENONE).key,
                         scaffold_key(ACETOPHENONE + ".Cl").key)

    def test_an_acyclic_substrate_is_not_fused_with_every_other_one(self) -> None:
        key = scaffold_key(HEXANONE)
        self.assertEqual(key.basis, ScaffoldBasis.ACYCLIC_SKELETON)
        self.assertNotEqual(key.key, scaffold_key("CCC(=O)CC").key)
        self.assertFalse(key.is_true_scaffold)

    def test_an_unparseable_smiles_is_unresolved_and_names_the_remedy(self) -> None:
        key = scaffold_key("C1CC")                    # unclosed ring digit
        self.assertIsNone(key.key)
        self.assertEqual(key.basis, ScaffoldBasis.UNRESOLVED)
        self.assertTrue(key.needs_curator)

    def test_a_prose_name_is_never_turned_into_a_key(self) -> None:
        """One character separates 4-chloroacetophenone from the 2- isomer."""
        key = scaffold_key(SubstrateSpec(name="4-chloroacetophenone"))
        self.assertIsNone(key.key)
        self.assertIn("isomeric SMILES", key.needs_curator or "")

    def test_an_inchikey_block_is_flagged_as_not_a_scaffold(self) -> None:
        # An obvious fixture string rather than a real compound's InChIKey: a
        # plausible-looking key in a test is copied into documentation and
        # eventually read as an identification of a specific molecule.
        key = scaffold_key(SubstrateSpec(inchikey="EXAMPLEFIXTURE-EXAMPLEFIXT-N"))
        self.assertEqual(key.basis, ScaffoldBasis.INCHIKEY_CONSTITUTION_BLOCK)
        self.assertFalse(key.is_true_scaffold)
        self.assertTrue(any("UNDER-states" in lim for lim in key.limitations))

    def test_the_known_over_grouping_is_documented_rather_than_hidden(self) -> None:
        """Benzene and cyclohexane collide; the key must say that it can."""
        key = scaffold_key("c1ccccc1")
        self.assertEqual(key.key, scaffold_key("C1CCCCC1").key)
        self.assertTrue(any("cyclohexane" in lim for lim in key.limitations))

    def test_a_record_can_be_passed_directly(self) -> None:
        rec = record("r1", smiles=ACETOPHENONE)
        self.assertEqual(scaffold_key(rec).key, scaffold_key(ACETOPHENONE).key)


# --------------------------------------------------------------------------
# grouped splitting
# --------------------------------------------------------------------------

class TestGroupedSplit(unittest.TestCase):

    def setUp(self) -> None:
        parent_seq = sequence_for("parent")
        self.siblings = [
            record("p1", sequence=parent_seq, doi=DOI_ONE),
            record("v1", sequence=sequence_for("v1"),
                   parent_sequence=parent_seq, doi=DOI_ONE),
            record("v2", sequence=sequence_for("v2"),
                   parent_sequence=parent_seq, doi=DOI_ONE),
        ]
        self.others = [record(f"o{i}", doi=f"{DOI_TWO}-{i}") for i in range(9)]
        self.records = self.siblings + self.others
        self.clusters = cluster_lookup(self.records)
        # The variants cluster with their parent, as a real clustering would.
        for rec in self.siblings[1:]:
            self.clusters[rec.sequence_sha256] = "clu-p1"

    def test_sibling_variants_never_straddle_the_boundary(self) -> None:
        """A parent in train and its mutant in test is interpolation, not a test."""
        for seed in range(8):
            split = grouped_split(
                self.records, SplitRegime.NOVEL_ENZYME, seed=seed,
                sequence_cluster_lookup=self.clusters)
            train = set(split.train_record_ids)
            test = set(split.test_record_ids)
            family = {"p1", "v1", "v2"}
            self.assertTrue(family <= train or family <= test,
                            f"seed {seed} split the sibling group: "
                            f"train={sorted(family & train)} "
                            f"test={sorted(family & test)}")

    def test_a_novel_enzyme_split_is_proven_clean(self) -> None:
        split = grouped_split(self.records, SplitRegime.NOVEL_ENZYME, seed=3,
                              sequence_cluster_lookup=self.clusters)
        self.assertTrue(split.is_usable)
        self.assertFalse(split.audit.has_leakage, split.audit.render())
        self.assertTrue(split.audit.proven_clean, split.audit.render())

    def test_a_shared_substrate_is_expected_under_the_novel_enzyme_regime(self) -> None:
        """Every fixture record uses acetophenone; that is the regime's design."""
        split = grouped_split(self.records, SplitRegime.NOVEL_ENZYME, seed=1,
                              sequence_cluster_lookup=self.clusters)
        shared = split.audit.by_category()[
            LeakageCategory.SHARED_SCAFFOLD.value]
        self.assertTrue(shared, "the fixture shares one substrate")
        self.assertNotIn(LeakageCategory.SHARED_SCAFFOLD,
                         {o.category for o in split.audit.blocking_overlaps})

    def test_novel_enzyme_without_a_clustering_is_refused(self) -> None:
        """An unseen-cluster claim nobody can check must not be producible."""
        with self.assertRaises(SplitNotPossibleError):
            grouped_split(self.records, SplitRegime.NOVEL_ENZYME)

    def test_the_split_is_deterministic_for_one_seed_and_moves_with_another(self) -> None:
        first = grouped_split(self.records, SplitRegime.NOVEL_ENZYME, seed=5,
                              sequence_cluster_lookup=self.clusters)
        again = grouped_split(self.records, SplitRegime.NOVEL_ENZYME, seed=5,
                              sequence_cluster_lookup=self.clusters)
        self.assertEqual(first.test_record_ids, again.test_record_ids)
        seen = {grouped_split(self.records, SplitRegime.NOVEL_ENZYME, seed=s,
                              sequence_cluster_lookup=self.clusters
                              ).test_record_ids
                for s in range(8)}
        self.assertGreater(len(seen), 1)

    def test_partition_returns_the_rows_and_drops_the_excluded_ones(self) -> None:
        split = grouped_split(self.records, SplitRegime.NOVEL_ENZYME, seed=2,
                              sequence_cluster_lookup=self.clusters)
        train, test = split.partition(self.records)
        self.assertEqual({r.record_id for r in train},
                         set(split.train_record_ids))
        self.assertEqual({r.record_id for r in test},
                         set(split.test_record_ids))
        self.assertEqual(len(train) + len(test) + len(split.excluded_record_ids),
                         len(self.records))

    def test_duplicate_record_ids_cannot_form_a_split(self) -> None:
        """A repeated id silently drops a row from the fold map."""
        rows = self.records + [record("p1", doi=DOI_ONE)]
        with self.assertRaises(SplitNotPossibleError):
            grouped_split(rows, SplitRegime.NOVEL_ENZYME,
                          sequence_cluster_lookup=self.clusters)

    def test_a_bad_test_fraction_is_refused(self) -> None:
        for fraction in (0.0, 1.0, -0.2, 1.5):
            with self.assertRaises(SplitNotPossibleError):
                grouped_split(self.records, SplitRegime.NOVEL_ENZYME,
                              test_fraction=fraction,
                              sequence_cluster_lookup=self.clusters)


class TestNovelSubstrateSplit(unittest.TestCase):

    def setUp(self) -> None:
        self.records = []
        for i, smiles in enumerate([ACETOPHENONE] * 3 + [NAPHTHYL_KETONE] * 3
                                   + [HEXANONE] * 3):
            self.records.append(
                record(f"s{i}", doi=f"{DOI_TWO}-{i}", smiles=smiles))

    def test_held_out_scaffolds_are_absent_from_training(self) -> None:
        split = grouped_split(self.records, SplitRegime.NOVEL_SUBSTRATE, seed=0)
        self.assertTrue(split.is_usable, split.render())
        train, test = split.partition(self.records)
        train_keys = {scaffold_key(r).key for r in train}
        test_keys = {scaffold_key(r).key for r in test}
        self.assertTrue(test_keys)
        self.assertFalse(train_keys & test_keys, split.audit.render())
        self.assertFalse(split.audit.has_leakage, split.audit.render())

    def test_a_row_with_no_structure_is_excluded_and_names_the_remedy(self) -> None:
        rows = self.records + [record("unknown", doi=f"{DOI_TWO}-x",
                                      smiles=None,
                                      substrate_name="an unregistered ketone")]
        split = grouped_split(rows, SplitRegime.NOVEL_SUBSTRATE, seed=0)
        self.assertIn("unknown", split.excluded_record_ids)
        self.assertIn("isomeric SMILES", split.exclusion_reasons["unknown"])
        self.assertNotIn("unknown", split.train_record_ids)
        self.assertNotIn("unknown", split.test_record_ids)

    def test_one_scaffold_everywhere_cannot_be_split_and_is_reported_short(self) -> None:
        """No fold can be formed without a scaffold on both sides, so none is."""
        rows = [record(f"same{i}", doi=f"{DOI_TWO}-{i}", smiles=ACETOPHENONE)
                for i in range(4)]
        split = grouped_split(rows, SplitRegime.NOVEL_SUBSTRATE, seed=0)
        self.assertFalse(split.is_usable)
        self.assertTrue(split.shortfall_reason)
        self.assertEqual(split.test_record_ids, ())

    def test_the_lineage_invariant_and_its_cost_are_stated(self) -> None:
        split = grouped_split(self.records, SplitRegime.NOVEL_SUBSTRATE, seed=0)
        self.assertTrue(any("enzyme-disjoint" in note for note in split.notes),
                        split.notes)

    def test_dual_extrapolation_holds_out_both_facets(self) -> None:
        clusters = cluster_lookup(self.records)
        split = grouped_split(self.records, SplitRegime.DUAL_EXTRAPOLATION,
                              seed=0, sequence_cluster_lookup=clusters)
        self.assertTrue(split.is_usable, split.render())
        self.assertFalse(split.audit.has_leakage, split.audit.render())
        self.assertTrue(split.held_out_scaffolds)
        self.assertTrue(split.held_out_sequence_clusters)


# --------------------------------------------------------------------------
# the audit
# --------------------------------------------------------------------------

class TestAuditLeakage(unittest.TestCase):

    def test_a_planted_publication_overlap_is_found(self) -> None:
        train = [record("t1", doi=DOI_ONE), record("t2", doi=DOI_TWO)]
        test = [record("e1", doi=DOI_ONE)]        # same paper as t1
        audit = audit_leakage(train, test)
        categories = {o.category for o in audit.overlaps}
        self.assertIn(LeakageCategory.SHARED_PUBLICATION, categories)
        self.assertIn(LeakageCategory.SHARED_SPLIT_GROUP, categories)
        self.assertTrue(audit.has_leakage)
        self.assertFalse(audit.proven_clean)
        shared = audit.by_category()[LeakageCategory.SHARED_PUBLICATION.value]
        self.assertEqual(shared[0].train_record_ids, ("t1",))
        self.assertEqual(shared[0].test_record_ids, ("e1",))

    def test_a_planted_parent_lineage_overlap_is_found(self) -> None:
        parent_seq = sequence_for("parent")
        train = [record("t1", sequence=parent_seq, doi=DOI_ONE)]
        test = [record("e1", sequence=sequence_for("v1"),
                       parent_sequence=parent_seq, doi=DOI_TWO)]
        audit = audit_leakage(train, test)
        self.assertIn(LeakageCategory.SHARED_PARENT_LINEAGE,
                      {o.category for o in audit.overlaps})

    def test_a_planted_sequence_cluster_overlap_is_found(self) -> None:
        train = [record("t1", doi=DOI_ONE)]
        test = [record("e1", doi=DOI_TWO)]
        clusters = {train[0].sequence_sha256: "clu-shared",
                    test[0].sequence_sha256: "clu-shared"}
        audit = audit_leakage(train, test, sequence_cluster_lookup=clusters)
        self.assertIn(LeakageCategory.SHARED_SEQUENCE_CLUSTER,
                      {o.category for o in audit.overlaps})

    def test_a_planted_scaffold_overlap_is_found(self) -> None:
        train = [record("t1", doi=DOI_ONE, smiles=ACETOPHENONE)]
        test = [record("e1", doi=DOI_TWO, smiles=CHLOROACETOPHENONE)]
        audit = audit_leakage(train, test, regime=SplitRegime.NOVEL_SUBSTRATE)
        self.assertIn(LeakageCategory.SHARED_SCAFFOLD,
                      {o.category for o in audit.blocking_overlaps})

    def test_a_clean_split_is_proven_clean(self) -> None:
        train = [record("t1", doi=DOI_ONE, smiles=ACETOPHENONE)]
        test = [record("e1", doi=DOI_TWO, smiles=ACETOPHENONE)]
        clusters = {train[0].sequence_sha256: "clu-a",
                    test[0].sequence_sha256: "clu-b"}
        audit = audit_leakage(train, test, regime=SplitRegime.NOVEL_ENZYME,
                              sequence_cluster_lookup=clusters)
        self.assertFalse(audit.has_leakage, audit.render())
        self.assertTrue(audit.proven_clean, audit.render())

    def test_an_unresolved_facet_means_not_proven_clean_rather_than_clean(self) -> None:
        train = [record("t1", doi=DOI_ONE, smiles=None)]
        test = [record("e1", doi=DOI_TWO, smiles=None)]
        audit = audit_leakage(train, test)
        self.assertFalse(audit.has_leakage)
        self.assertFalse(audit.proven_clean)
        self.assertTrue(audit.unchecked)

    def test_a_novel_enzyme_claim_without_a_clustering_is_not_proven(self) -> None:
        train = [record("t1", doi=DOI_ONE)]
        test = [record("e1", doi=DOI_TWO)]
        audit = audit_leakage(train, test, regime=SplitRegime.NOVEL_ENZYME)
        self.assertFalse(audit.proven_clean)
        self.assertTrue(any("clustering" in item for item in audit.unchecked))

    def test_duplicate_record_ids_are_declared_rather_than_absorbed(self) -> None:
        audit = audit_leakage([record("t1", doi=DOI_ONE),
                               record("t1", doi=DOI_TWO)],
                              [record("e1", doi=f"{DOI_TWO}-x")])
        self.assertFalse(audit.proven_clean)
        self.assertTrue(any("more than once" in item
                            for item in audit.unchecked))

    def test_an_unknown_substrate_does_not_block_a_novel_enzyme_claim(self) -> None:
        """That regime shares substrates on purpose, so the gap is a note."""
        train = [record("t1", doi=DOI_ONE, smiles=None)]
        test = [record("e1", doi=DOI_TWO, smiles=None)]
        clusters = {train[0].sequence_sha256: "clu-a",
                    test[0].sequence_sha256: "clu-b"}
        audit = audit_leakage(train, test, regime=SplitRegime.NOVEL_ENZYME,
                              sequence_cluster_lookup=clusters)
        self.assertTrue(audit.proven_clean, audit.render())
        self.assertTrue(any("scaffold unresolved" in note
                            for note in audit.notes))

    def test_every_category_is_listed_even_when_empty(self) -> None:
        """A missing line reads as 'not checked'; a zero reads as checked."""
        audit = audit_leakage([record("t1", doi=DOI_ONE)],
                              [record("e1", doi=DOI_TWO)])
        self.assertEqual(set(audit.by_category()),
                         {c.value for c in LeakageCategory})


class TestRecurationDetection(unittest.TestCase):
    """'Train on database A, test on database B' when B re-curated A."""

    @staticmethod
    def registry() -> SourceRegistry:
        common = dict(
            layers=[DataLayer.REACTION_AND_CHEMISTRY],
            good_for=["an activity record"],
            not_good_for=["sequence-level activity for an uncharacterised protein"],
            access_modes=[AccessMode.REST_API],
            curation_notes=["confirm the endpoint"],
        )
        return SourceRegistry([
            DataSource(id="source_a", display_name="Source A",
                       derived_from=[], **common),
            DataSource(id="source_b", display_name="Source B",
                       derived_from=["source_a"], **common),
        ])

    def test_the_registry_derived_from_edge_is_followed(self) -> None:
        train = [record("t1", doi=DOI_ONE, database_id="a:1")]
        test = [record("e1", doi=DOI_TWO, database_id="b:1")]
        audit = audit_leakage(
            train, test, registry=self.registry(),
            source_lookup={"t1": "source_a", "e1": "source_b"})
        recuration = audit.by_category()[LeakageCategory.RECURATED_SOURCE.value]
        self.assertTrue(recuration, audit.render())
        self.assertIn("source_a", recuration[0].key)
        self.assertIn("source_b", recuration[0].key)
        self.assertTrue(audit.has_leakage)

    def test_two_genuinely_separate_resources_are_not_flagged(self) -> None:
        common = dict(
            layers=[DataLayer.REACTION_AND_CHEMISTRY],
            good_for=["an activity record"],
            not_good_for=["sequence-level activity for an uncharacterised protein"],
            access_modes=[AccessMode.REST_API],
            curation_notes=["confirm the endpoint"],
        )
        registry = SourceRegistry([
            DataSource(id="source_a", display_name="Source A", **common),
            DataSource(id="source_c", display_name="Source C", **common),
        ])
        audit = audit_leakage(
            [record("t1", doi=DOI_ONE)], [record("e1", doi=DOI_TWO)],
            registry=registry,
            source_lookup={"t1": "source_a", "e1": "source_c"})
        self.assertFalse(
            audit.by_category()[LeakageCategory.RECURATED_SOURCE.value])

    def test_a_shared_upstream_is_caught_without_any_registry(self) -> None:
        """Both sides re-integrate one resource, so neither side is independent."""
        train = [record("t1", doi=DOI_ONE, upstream=("source_a",))]
        test = [record("e1", doi=DOI_TWO, upstream=("source_a",))]
        audit = audit_leakage(train, test)
        recuration = audit.by_category()[LeakageCategory.RECURATED_SOURCE.value]
        self.assertTrue(recuration, audit.render())
        self.assertEqual(recuration[0].key, "source_a")

    def test_without_a_registry_the_claim_of_independence_is_unproven(self) -> None:
        audit = audit_leakage(
            [record("t1", doi=DOI_ONE)], [record("e1", doi=DOI_TWO)],
            source_lookup={"t1": "source_a", "e1": "source_b"})
        self.assertTrue(any("derived_from" in note for note in audit.notes))

    def test_source_tokens_read_the_three_places_they_can_live(self) -> None:
        rec = record("t1", doi=DOI_ONE, upstream=("source_a", "Source A"))
        tokens = source_tokens_of(rec, {"t1": "source_b"}, "t1")
        self.assertEqual(tokens, {"source_a", "source a", "source_b"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
