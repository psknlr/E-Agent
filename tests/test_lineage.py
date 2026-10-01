"""Tests for :mod:`eagent.datalayer.lineage`.

Every expected value here is decided from the construction of the fixture, not
from running the code and recording what it printed. The central fixture is the
re-curation chain the module exists to defeat: one measurement, in one paper,
copied into four databases. The answer "one independent measurement" is true by
construction of the fixture, so a regression that starts reporting four fails
the test rather than redefining it.
"""

from __future__ import annotations

import unittest

from eagent.datalayer.lineage import (
    DiscountedRow,
    EvidenceGroup,
    LineageReport,
    ProvenanceGraph,
    assay_fingerprint,
    corroboration_level,
    count_independent,
    grouping_key,
    identity_tokens,
    independent_evidence_groups,
    leakage_safe_groups,
    normalise_doi,
    publication_ids,
    split_leakage,
)
from eagent.schemas.candidate import ConfidenceLevel
from eagent.schemas.chem import CofactorSpec, CofactorState, SubstrateSpec
from eagent.schemas.reaction import Conditions, ReactionClass
from eagent.schemas.record import (
    Detection,
    EvidenceRef,
    EvidenceStrength,
    ExperimentRecord,
    OutcomeClass,
    ReactionDirection,
)

PARENT_SEQ = "MKAVVLSGFGGLDNVKLEEVPKPTPGPGQVLVKVEAAGVCHSDLHLIDGDLP"
VARIANT_SEQ = "MKAVVLSGFGGLDNVKLEEVPKPTPGPGQVLVKVEAAGVCHSDLHLIDGDLA"
OTHER_SEQ = "MSTQLFKPLTIGSLELKNRIVMAPMTRSRAENGVPGELMAEYYAQRASAGLI"

DOI_A = "10.1021/acscatal.0c00001"
DOI_B = "10.1038/s41929-021-00001-2"

SUBSTRATE = SubstrateSpec(name="acetophenone", isomeric_smiles="CC(=O)c1ccccc1",
                          inchikey="KWOLFJPFCHCOCG-UHFFFAOYSA-N")
CONDITIONS = Conditions(pH=7.0, temperature_C=30.0, buffer="phosphate",
                        substrate_concentration_mM=10.0)
NADPH = CofactorSpec(name="NADPH", state=CofactorState.REDUCED)
POSITIVE_DETECTION = Detection(method="chiral GC-MS", authentic_standard=True,
                               confirms_product_identity=True,
                               chiral_method_validated=True)


def _ref(
    *,
    source_type: str = "database",
    identifier: str,
    doi: str | None = None,
    activity: str | None = None,
    upstream: tuple[str, ...] = (),
    strength: EvidenceStrength = EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL,
) -> EvidenceRef:
    return EvidenceRef(
        source_type=source_type,
        identifier=identifier,
        source_doi=doi,
        experiment_activity_id=activity,
        upstream_sources=list(upstream),
        strength=strength,
        extracted_by="human",
    )


def _record(
    record_id: str,
    *,
    evidence: list[EvidenceRef],
    sequence: str = PARENT_SEQ,
    outcome: OutcomeClass = OutcomeClass.CONFIRMED_TARGET_PRODUCT,
    conversion: float | None = 92.0,
    ee: float | None = 98.0,
    parent: str | None = None,
    detection: Detection | None = None,
) -> ExperimentRecord:
    kwargs: dict = {}
    if parent is not None:
        kwargs["is_variant"] = True
        kwargs["parent_sequence_sha256"] = parent
    return ExperimentRecord(
        record_id=record_id,
        sequence=sequence,
        substrate=SUBSTRATE,
        conditions=CONDITIONS,
        cofactor=NADPH,
        reaction_class=ReactionClass.KETONE_TO_SECONDARY_ALCOHOL,
        reaction_direction=ReactionDirection.FORWARD_AS_TARGET,
        outcome=outcome,
        detection=detection if detection is not None else POSITIVE_DETECTION,
        conversion_pct=conversion,
        ee_target_pct=ee,
        evidence=evidence,
        **kwargs,
    )


def _recuration_chain() -> list[ExperimentRecord]:
    """One measurement, one paper, re-integrated by four resources.

    BRENDA curates it from the paper. OED re-integrates BRENDA. SKiD
    re-integrates BRENDA and OED. CatPred-DB re-integrates all three. Every row
    carries the same DOI, which is the only honest reason to believe they are
    one measurement -- and they are, by construction.
    """
    chain = [
        ("rec-brenda", "BRENDA:1.1.1.2:ent9001", ()),
        ("rec-oed", "OED:9001", ("BRENDA",)),
        ("rec-skid", "SKiD:77-9001", ("BRENDA", "OED")),
        ("rec-catpred", "CATPRED-DB:kcat-9001", ("BRENDA", "OED", "SKiD")),
    ]
    return [
        _record(rid, evidence=[_ref(identifier=ident, doi=DOI_A, upstream=up)])
        for rid, ident, up in chain
    ]


class TestDoiNormalisation(unittest.TestCase):
    def test_spellings_collapse(self) -> None:
        """Four spellings of one DOI must not look like four papers."""
        for spelling in (
            DOI_A,
            DOI_A.upper(),
            f"https://doi.org/{DOI_A}",
            f"doi:{DOI_A}",
            f" http://dx.doi.org/{DOI_A} ",
        ):
            self.assertEqual(normalise_doi(spelling), DOI_A)

    def test_non_doi_returns_none(self) -> None:
        """A PMID is not a DOI; guessing would merge two different papers."""
        for junk in (None, "", "12345678", "PMID:12345678", "BRENDA:1.1.1.2"):
            self.assertIsNone(normalise_doi(junk))


class TestRecurationCollapsesToOneMeasurement(unittest.TestCase):
    def setUp(self) -> None:
        self.records = _recuration_chain()

    def test_four_rows_one_doi_is_one_independent_measurement(self) -> None:
        self.assertEqual(len(self.records), 4)
        self.assertEqual(count_independent(self.records), 1)

    def test_the_single_group_holds_all_four_rows(self) -> None:
        groups = independent_evidence_groups(self.records)
        self.assertEqual(len(groups), 1)
        g = groups[0]
        self.assertEqual(g.key_kind, "publication")
        self.assertEqual(g.n_rows, 4)
        self.assertEqual(set(g.record_ids),
                         {"rec-brenda", "rec-oed", "rec-skid", "rec-catpred"})
        self.assertIn(f"doi:{DOI_A}", g.key_values)

    def test_most_upstream_source_is_the_publication_not_a_database(self) -> None:
        """BRENDA copied the number; the paper produced it."""
        g = independent_evidence_groups(self.records)[0]
        self.assertEqual(g.most_upstream_source, f"doi:{DOI_A}")
        self.assertNotIn("BRENDA", str(g.most_upstream_source))
        self.assertIn("BRENDA", g.upstream_resources)
        self.assertIn("SKiD", g.upstream_resources)

    def test_report_discounts_three_of_four_rows_with_reasons(self) -> None:
        report = LineageReport.build("ADH-X reduces acetophenone", self.records)
        self.assertEqual(report.n_rows, 4)
        self.assertEqual(report.n_independent, 1)
        self.assertEqual(len(report.discounted), 3)
        for d in report.discounted:
            self.assertIsInstance(d, DiscountedRow)
            self.assertIn("re-report", d.reason)
        counted = {d.record_id for d in report.discounted}
        rep = independent_evidence_groups(self.records)[0].representative_record_id
        self.assertNotIn(rep, counted)
        rendered = report.render()
        self.assertIn("rows retrieved:           4", rendered)
        self.assertIn("independent measurements: 1", rendered)

    def test_copies_cannot_raise_corroboration(self) -> None:
        """Ten copies of one experiment are still one experiment."""
        base = corroboration_level(self.records)
        self.assertEqual(base, ConfidenceLevel.MODERATE)
        inflated = list(self.records)
        for i in range(10):
            inflated.append(_record(
                f"rec-copy-{i}",
                evidence=[_ref(identifier=f"MIRROR:{i}", doi=DOI_A,
                               upstream=("CATPRED-DB",))],
            ))
        self.assertEqual(count_independent(inflated), 1)
        self.assertEqual(corroboration_level(inflated), base)


class TestCorroborationRisesOnlyWithDistinctSources(unittest.TestCase):
    def test_second_genuinely_distinct_paper_raises_the_level(self) -> None:
        one_paper = _recuration_chain()
        self.assertEqual(corroboration_level(one_paper), ConfidenceLevel.MODERATE)

        second = _record(
            "rec-paper-b",
            evidence=[_ref(source_type="publication", identifier=DOI_B, doi=DOI_B)],
        )
        two_papers = one_paper + [second]
        self.assertEqual(count_independent(two_papers), 2)
        self.assertEqual(corroboration_level(two_papers), ConfidenceLevel.STRONG)

    def test_weak_evidence_cannot_reach_strong_by_repetition(self) -> None:
        """Annotation-grade rows from five papers are still not an experiment."""
        rows = [
            _record(
                f"rec-annot-{i}",
                evidence=[_ref(source_type="publication",
                               identifier=f"10.1000/paper.{i}",
                               doi=f"10.1000/paper.{i}",
                               strength=EvidenceStrength.EC_SPECIES_MAPPED)],
            )
            for i in range(5)
        ]
        self.assertEqual(count_independent(rows), 5)
        self.assertEqual(corroboration_level(rows), ConfidenceLevel.WEAK)

    def test_two_homolog_level_papers_reach_moderate_not_strong(self) -> None:
        rows = [
            _record(
                f"rec-hom-{i}",
                evidence=[_ref(source_type="publication",
                               identifier=f"10.1000/hom.{i}",
                               doi=f"10.1000/hom.{i}",
                               strength=EvidenceStrength.HOMOLOG_EXPERIMENTAL)],
            )
            for i in range(2)
        ]
        self.assertEqual(corroboration_level(rows), ConfidenceLevel.MODERATE)

    def test_computational_failure_is_not_evidence_either_way(self) -> None:
        """A modelling failure says nothing about the enzyme, however often it repeats."""
        rows = [
            _record(
                f"rec-comp-{i}",
                evidence=[_ref(source_type="publication",
                               identifier=f"10.1000/comp.{i}",
                               doi=f"10.1000/comp.{i}")],
                outcome=OutcomeClass.COMPUTATIONAL_FAILURE,
                conversion=None, ee=None,
                detection=Detection(),
            )
            for i in range(3)
        ]
        self.assertEqual(count_independent(rows), 3)
        self.assertEqual(corroboration_level(rows), ConfidenceLevel.INSUFFICIENT)

    def test_expression_failure_is_not_a_catalytic_negative(self) -> None:
        rows = [_record(
            "rec-noexpress",
            evidence=[_ref(source_type="publication", identifier=DOI_A, doi=DOI_A)],
            outcome=OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE,
            conversion=None, ee=None, detection=Detection(),
        )]
        self.assertEqual(corroboration_level(rows), ConfidenceLevel.INSUFFICIENT)

    def test_independent_disagreement_is_contradictory_not_averaged(self) -> None:
        positive = _record(
            "rec-pos",
            evidence=[_ref(source_type="publication", identifier=DOI_A, doi=DOI_A)],
        )
        negative = _record(
            "rec-neg",
            evidence=[_ref(source_type="publication", identifier=DOI_B, doi=DOI_B)],
            outcome=OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
            conversion=0.0, ee=None,
            detection=Detection(method="chiral GC-MS", limit_of_detection=0.5,
                                limit_unit="%", confirms_product_identity=True),
        )
        self.assertEqual(corroboration_level([positive, negative]),
                         ConfidenceLevel.CONTRADICTORY)


class TestKeyPriority(unittest.TestCase):
    def test_activity_id_is_the_first_priority_key(self) -> None:
        rec = _record(
            "rec-act",
            evidence=[_ref(identifier="BRENDA:1", doi=DOI_A, activity="assay-7")],
        )
        tokens, tier = identity_tokens(rec)
        self.assertEqual(tier, "experiment_activity")
        self.assertIn("activity:assay-7", tokens)

    def test_strict_priority_separates_two_activities_in_one_paper(self) -> None:
        a = _record("rec-a1", evidence=[_ref(identifier="X:1", doi=DOI_A,
                                             activity="assay-1")])
        b = _record("rec-a2", evidence=[_ref(identifier="X:2", doi=DOI_A,
                                             activity="assay-2")])
        self.assertEqual(count_independent([a, b], link_across_tiers=False), 2)

    def test_default_links_a_copy_that_lost_the_activity_id(self) -> None:
        """The activity id is the first field a re-integrating resource drops."""
        full = _record("rec-full", evidence=[_ref(identifier="BRENDA:1", doi=DOI_A,
                                                  activity="assay-1")])
        copy = _record("rec-copy", evidence=[_ref(identifier="SKiD:1", doi=DOI_A,
                                                  upstream=("BRENDA",))])
        groups = independent_evidence_groups([full, copy])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].key_kind, "experiment_activity")
        self.assertEqual(set(groups[0].record_ids), {"rec-full", "rec-copy"})

    def test_fingerprint_only_applies_when_no_identifier_exists(self) -> None:
        """Two papers may report the same number; that is replication, not copying."""
        a = _record("rec-fp-a", evidence=[_ref(source_type="publication",
                                               identifier=DOI_A, doi=DOI_A)])
        b = _record("rec-fp-b", evidence=[_ref(source_type="publication",
                                               identifier=DOI_B, doi=DOI_B)])
        self.assertEqual(assay_fingerprint(a), assay_fingerprint(b))
        self.assertEqual(count_independent([a, b]), 2)

    def test_anonymous_rows_with_one_fingerprint_collapse(self) -> None:
        a = _record("rec-anon-a", evidence=[])
        b = _record("rec-anon-b", evidence=[])
        self.assertIsNotNone(assay_fingerprint(a))
        groups = independent_evidence_groups([a, b])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].key_kind, "assay_fingerprint")

    def test_rows_with_nothing_to_match_on_stay_separate_and_are_flagged(self) -> None:
        """An unmatched row is reported, never merged on suspicion."""
        a = ExperimentRecord(record_id="rec-blank-a", sequence=PARENT_SEQ)
        b = ExperimentRecord(record_id="rec-blank-b", sequence=OTHER_SEQ)
        self.assertIsNone(assay_fingerprint(a))
        groups = independent_evidence_groups([a, b])
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(g.key_kind == "unlinkable" for g in groups))
        report = LineageReport.build("untested rows", [a, b])
        self.assertEqual(sorted(report.unlinkable_record_ids),
                         ["rec-blank-a", "rec-blank-b"])
        self.assertEqual(report.corroboration, ConfidenceLevel.INSUFFICIENT)


class TestProvenanceGraph(unittest.TestCase):
    def test_graph_records_derivation_from_resources(self) -> None:
        g = ProvenanceGraph.from_records(_recuration_chain())
        self.assertIn(f"doi:{DOI_A}", g.nodes)
        self.assertIn("resource:brenda", g.nodes)
        self.assertEqual(g.node("resource:brenda").kind, "resource")
        self.assertEqual(g.node(f"doi:{DOI_A}").kind, "publication")
        self.assertIn(f"doi:{DOI_A}", g.ancestors("resource:skid"))

    def test_mutually_citing_resources_do_not_hang_the_walk(self) -> None:
        """Resources cite each other in the real world; the walk must terminate."""
        g = ProvenanceGraph()
        g.add_edge("resource:a", "resource:b")
        g.add_edge("resource:b", "resource:a")
        # The walk stops at the cycle instead of following it, so "a" is not
        # reported as its own ancestor and the call returns at all.
        self.assertEqual(set(g.ancestors("resource:a")), {"resource:b"})
        self.assertEqual(set(g.ancestors("resource:b")), {"resource:a"})
        self.assertIn(("resource:a", "resource:b"), g.cycles())
        # Neither is upstream of the other, so neither may be named the origin.
        self.assertEqual(g.roots_among(["resource:a", "resource:b"]), [])

    def test_ambiguous_origin_is_reported_not_guessed(self) -> None:
        rec = _record(
            "rec-two-papers",
            evidence=[
                _ref(source_type="publication", identifier=DOI_A, doi=DOI_A),
                _ref(source_type="publication", identifier=DOI_B, doi=DOI_B),
            ],
        )
        g = independent_evidence_groups([rec])[0]
        self.assertIsNone(g.most_upstream_source)
        self.assertEqual(len(g.upstream_candidates), 2)
        self.assertTrue(any("ambiguous origin" in n for n in g.notes))


class TestGroupingKeyForSplitting(unittest.TestCase):
    def setUp(self) -> None:
        from eagent.provenance import sequence_hash

        self.parent_hash = sequence_hash(PARENT_SEQ)
        self.parent = _record(
            "rec-parent",
            evidence=[_ref(source_type="publication", identifier=DOI_A, doi=DOI_A)],
        )
        self.variant_1 = _record(
            "rec-var1", sequence=VARIANT_SEQ, parent=self.parent_hash,
            evidence=[_ref(source_type="publication", identifier=DOI_B, doi=DOI_B)],
        )
        self.variant_2 = _record(
            "rec-var2", sequence=VARIANT_SEQ, parent=self.parent_hash,
            evidence=[_ref(source_type="publication", identifier="10.1000/other",
                           doi="10.1000/other")],
        )

    def test_sibling_variants_share_a_split_group(self) -> None:
        """A variant in train and its sibling in test is leakage, not generalisation."""
        assign = leakage_safe_groups([self.variant_1, self.variant_2])
        self.assertEqual(assign["rec-var1"], assign["rec-var2"])

    def test_variant_and_parent_share_a_split_group(self) -> None:
        assign = leakage_safe_groups([self.parent, self.variant_1])
        self.assertEqual(assign["rec-parent"], assign["rec-var1"])

    def test_same_paper_records_share_a_split_group(self) -> None:
        other_enzyme = _record(
            "rec-same-paper", sequence=OTHER_SEQ,
            evidence=[_ref(source_type="publication", identifier=DOI_A, doi=DOI_A)],
        )
        assign = leakage_safe_groups([self.parent, other_enzyme])
        self.assertEqual(assign["rec-parent"], assign["rec-same-paper"])

    def test_the_source_database_is_not_part_of_the_key(self) -> None:
        """'Train on BRENDA, test on SKiD' must not look like a clean split."""
        brenda = _record("rec-b", evidence=[_ref(identifier="BRENDA:1", doi=DOI_A)])
        skid = _record("rec-s", evidence=[_ref(identifier="SKiD:1", doi=DOI_A,
                                               upstream=("BRENDA",))])
        for rec in (brenda, skid):
            for facet in grouping_key(rec):
                self.assertNotIn("brenda", facet.lower())
                self.assertNotIn("skid", facet.lower())
        assign = leakage_safe_groups([brenda, skid])
        self.assertEqual(assign["rec-b"], assign["rec-s"])
        leak = split_leakage([brenda], [skid])
        self.assertFalse(leak.ok)
        self.assertIn("SPLIT LEAKS", leak.render())

    def test_unrelated_sequences_in_different_papers_do_not_share_a_group(self) -> None:
        a = _record("rec-u1", sequence=PARENT_SEQ,
                    evidence=[_ref(source_type="publication", identifier=DOI_A,
                                   doi=DOI_A)])
        b = _record("rec-u2", sequence=OTHER_SEQ,
                    evidence=[_ref(source_type="publication", identifier=DOI_B,
                                   doi=DOI_B)])
        assign = leakage_safe_groups([a, b])
        self.assertNotEqual(assign["rec-u1"], assign["rec-u2"])
        self.assertTrue(split_leakage([a], [b]).ok)

    def test_unresolved_clusters_do_not_fuse_into_one_group(self) -> None:
        """A shared 'cluster:unknown' placeholder would make every split vacuous."""
        a = _record("rec-c1", sequence=PARENT_SEQ,
                    evidence=[_ref(source_type="publication", identifier=DOI_A,
                                   doi=DOI_A)])
        b = _record("rec-c2", sequence=OTHER_SEQ,
                    evidence=[_ref(source_type="publication", identifier=DOI_B,
                                   doi=DOI_B)])
        facets_a = [f for f in grouping_key(a, None) if f.startswith("cluster:")]
        facets_b = [f for f in grouping_key(b, None) if f.startswith("cluster:")]
        self.assertTrue(facets_a and facets_b)
        self.assertNotEqual(facets_a, facets_b)

    def test_cluster_lookup_merges_homologous_sequences(self) -> None:
        from eagent.provenance import sequence_hash

        lookup = {sequence_hash(PARENT_SEQ): "clust-1",
                  sequence_hash(OTHER_SEQ): "clust-1"}
        a = _record("rec-h1", sequence=PARENT_SEQ,
                    evidence=[_ref(source_type="publication", identifier=DOI_A,
                                   doi=DOI_A)])
        b = _record("rec-h2", sequence=OTHER_SEQ,
                    evidence=[_ref(source_type="publication", identifier=DOI_B,
                                   doi=DOI_B)])
        self.assertTrue(split_leakage([a], [b]).ok,
                        "without the cluster lookup there is no link to find")
        assign = leakage_safe_groups([a, b], lookup)
        self.assertEqual(assign["rec-h1"], assign["rec-h2"])
        self.assertFalse(split_leakage([a], [b], lookup).ok)

    def test_transitive_closure_catches_an_indirect_link(self) -> None:
        """A shares a paper with B; B shares a cluster with C; all three must stay together."""
        from eagent.provenance import sequence_hash

        lookup = {sequence_hash(VARIANT_SEQ): "clust-9",
                  sequence_hash(OTHER_SEQ): "clust-9"}
        a = _record("rec-t-a", sequence=PARENT_SEQ,
                    evidence=[_ref(source_type="publication", identifier=DOI_A,
                                   doi=DOI_A)])
        b = _record("rec-t-b", sequence=VARIANT_SEQ,
                    evidence=[_ref(source_type="publication", identifier=DOI_A,
                                   doi=DOI_A)])
        c = _record("rec-t-c", sequence=OTHER_SEQ,
                    evidence=[_ref(source_type="publication", identifier=DOI_B,
                                   doi=DOI_B)])
        # a and c share no facet directly: only the chain through b links them.
        self.assertFalse(set(grouping_key(a, lookup)) & set(grouping_key(c, lookup)))
        assign = leakage_safe_groups([a, b, c], lookup)
        self.assertEqual(assign["rec-t-a"], assign["rec-t-c"])
        self.assertEqual(assign["rec-t-a"], assign["rec-t-b"])
        leak = split_leakage([a, b], [c], lookup)
        self.assertFalse(leak.ok)
        self.assertIn("rec-t-c", leak.test_record_ids)
        # Dropping the bridging row removes the link, and nothing is invented.
        self.assertTrue(split_leakage([a], [c], lookup).ok)


class TestReportSerialisation(unittest.TestCase):
    def test_report_round_trips_to_plain_data(self) -> None:
        report = LineageReport.build("ADH-X reduces acetophenone", _recuration_chain())
        d = report.to_dict()
        self.assertEqual(d["n_rows"], 4)
        self.assertEqual(d["n_independent"], 1)
        self.assertEqual(d["corroboration"], ConfidenceLevel.MODERATE.value)
        self.assertEqual(len(d["discounted"]), 3)
        self.assertEqual(len(d["groups"]), 1)
        self.assertIsInstance(d["groups"][0]["record_ids"], list)

    def test_group_ids_are_stable_across_runs(self) -> None:
        a = independent_evidence_groups(_recuration_chain())
        b = independent_evidence_groups(list(reversed(_recuration_chain())))
        self.assertEqual([g.group_id for g in a], [g.group_id for g in b])
        self.assertIsInstance(a[0], EvidenceGroup)

    def test_publication_ids_are_deduplicated(self) -> None:
        rec = _record("rec-dup", evidence=[
            _ref(identifier="BRENDA:1", doi=DOI_A),
            _ref(identifier="OED:1", doi=f"https://doi.org/{DOI_A.upper()}"),
        ])
        self.assertEqual(publication_ids(rec), [f"doi:{DOI_A}"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
