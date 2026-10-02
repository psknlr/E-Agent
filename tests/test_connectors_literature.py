"""Tests for the literature connectors.

The failures here are partly scientific and partly legal, and all four are
silent:

* an abstract is ingested as a measurement, so a seed set fills with numbers
  nobody reported;
* an article's full text is carried into this project's artifacts under a
  licence that does not permit it;
* an expert-annotated corpus and a text-mined one are concatenated, after
  which which-was-which cannot be recovered;
* a run is pinned to a Zenodo concept DOI, so its inputs change whenever the
  authors upload a new version and nothing records that they did.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eagent.connectors.base import AccessPolicy, FileCache, ResponseStatus
from eagent.connectors.chemistry import (
    UNVERIFIED_CAPABILITY,
    EndpointNotEstablishedError,
    LayerSemanticsError,
    datasource_registry,
)
from eagent.connectors.literature import (
    OPEN_REDISTRIBUTABLE_LICENCES,
    CitationNotEvidenceError,
    CorpusMergeRefusedError,
    EnzChemREDAsset,
    EnzChemREDConnector,
    EuropePMCConnector,
    NotRedistributableError,
    PubMedConnector,
    ZenodoConnector,
    is_open_redistributable,
)
from eagent.datalayer.intake import EvidenceTier, ExtractionMethod
from eagent.datalayer.registry import CapabilityState
from eagent.schemas.record import EvidenceStrength, OutcomeClass


class LiteratureTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cache = FileCache(self.root)
        self.registry = datasource_registry()
        self.addCleanup(self._tmp.cleanup)


# ---------------------------------------------------------------------------
# offline-first, endpoints, capabilities
# ---------------------------------------------------------------------------

class TestOfflineAndRefusals(LiteratureTestCase):

    def test_a_cache_miss_is_structured_not_an_empty_bibliography(self) -> None:
        response = PubMedConnector(cache=self.cache).search({"query": "KRED"})
        self.assertIs(response.status, ResponseStatus.MISS)
        self.assertIsNone(response.payload)
        self.assertTrue(response.needed)
        self.assertEqual(
            PubMedConnector(cache=self.cache).citations({"query": "KRED"}), ())
        self.assertIsNone(PubMedConnector(cache=self.cache).citation("1"))

    def test_a_cache_hit_returns_the_curated_citations(self) -> None:
        connector = PubMedConnector(cache=self.cache)
        connector.store_import("search", {"query": "KRED"}, {"records": [
            {"identifier": "12345678", "title": "A ketoreductase",
             "journal": "J. Test", "year": 2021, "doi": "10.1000/abc"}]})
        citations = connector.citations({"query": "KRED"})
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].identifier, "12345678")
        self.assertEqual(citations[0].year, 2021)
        self.assertEqual(citations[0].doi, "10.1000/abc")

    def test_every_literature_source_still_records_a_null_endpoint(self) -> None:
        for source_id in ("pubmed", "europe_pmc", "enzchemred", "zenodo"):
            with self.subTest(source=source_id):
                self.assertIsNone(self.registry.get(source_id).endpoint)

    def test_require_endpoint_is_a_typed_refusal(self) -> None:
        for connector in (PubMedConnector(cache=self.cache),
                          EuropePMCConnector(cache=self.cache),
                          ZenodoConnector(cache=self.cache)):
            with self.subTest(source=connector.source_id):
                with self.assertRaises(EndpointNotEstablishedError) as ctx:
                    connector.require_endpoint()
                self.assertEqual(ctx.exception.source_id, connector.source_id)
                self.assertTrue(ctx.exception.curation_notes)

    def test_the_europe_pmc_curation_note_is_a_hint_not_a_configured_url(self) -> None:
        """The registry's note mentions a candidate base; the code must not."""
        source = self.registry.get("europe_pmc")
        self.assertIsNone(source.endpoint)
        self.assertTrue(any("Confirm the real base URL" in note
                            for note in source.curation_notes))

    def test_enzchemred_refuses_a_keyword_query(self) -> None:
        connector = EnzChemREDConnector(cache=self.cache)
        self.assertIs(connector.capability("keyword_query").state,
                      CapabilityState.NOT_SUPPORTED)
        response = connector.search({"query": "ketoreductase"})
        self.assertIs(response.status, ResponseStatus.REFUSED)
        self.assertIsNone(response.payload)
        self.assertIn("not_supported", response.miss_reason or "")

    def test_an_unknown_capability_is_marked_unverified(self) -> None:
        connector = ZenodoConnector(cache=self.cache)
        self.assertIs(connector.capability("exact_record_fetch").state,
                      CapabilityState.UNKNOWN)
        connector.store_import("fetch", "Z1",
                               {"records": [{"deposit_id": "Z1"}]})
        response = connector.fetch("Z1")
        self.assertTrue(any(n.startswith(UNVERIFIED_CAPABILITY)
                            for n in response.notes))

    def test_a_supported_capability_is_not_marked_unverified(self) -> None:
        connector = PubMedConnector(cache=self.cache)
        self.assertIs(connector.capability("exact_record_fetch").state,
                      CapabilityState.SUPPORTED)
        connector.store_import("fetch", "1",
                               {"records": [{"identifier": "1"}]})
        self.assertFalse(any(n.startswith(UNVERIFIED_CAPABILITY)
                             for n in connector.fetch("1").notes))

    def test_network_is_off_by_default(self) -> None:
        for factory in (PubMedConnector, EuropePMCConnector,
                        EnzChemREDConnector, ZenodoConnector):
            with self.subTest(connector=factory.__name__):
                self.assertFalse(factory(cache=self.cache).access.allow_network)

    def test_a_network_enabled_run_still_misses_rather_than_dialling(self) -> None:
        connector = PubMedConnector(cache=self.cache,
                                    access=AccessPolicy(allow_network=True))
        self.assertIs(connector.fetch("404").status, ResponseStatus.MISS)


# ---------------------------------------------------------------------------
# citations are not evidence
# ---------------------------------------------------------------------------

class TestCitationsAreNotEvidence(LiteratureTestCase):

    def _citation(self):
        connector = PubMedConnector(cache=self.cache)
        connector.store_import("fetch", "12345678", {"records": [
            {"identifier": "12345678", "title": "High activity towards ketones",
             "abstract": "conversion of 98% was observed"}]})
        return connector.citation("12345678")

    def test_an_abstract_is_never_a_measurement(self) -> None:
        with self.assertRaises(CitationNotEvidenceError) as ctx:
            self._citation().as_measurement()
        self.assertIn("not data", str(ctx.exception))

    def test_a_citation_record_carries_no_outcome(self) -> None:
        record = self._citation().to_experiment_record()
        self.assertIs(record.outcome, OutcomeClass.NOT_TESTED)
        self.assertIsNone(record.measurement_value)
        self.assertIsNone(record.conversion_pct)

    def test_a_citation_carries_the_source_ceiling_and_lineage(self) -> None:
        citation = self._citation()
        self.assertIs(citation.evidence.strength,
                      EvidenceStrength.ANNOTATION_ONLY)
        self.assertIn("pubmed", citation.evidence.upstream_sources)

    def test_absence_of_a_hit_is_not_absence_of_work(self) -> None:
        self.assertIn("absence of a hit", " ".join(self._citation().notes))

    def test_an_unpinnable_search_says_so(self) -> None:
        self.assertIs(self.registry.get("pubmed").capabilities.version_information,
                      CapabilityState.NOT_SUPPORTED)
        self.assertIn("cannot be pinned", " ".join(self._citation().notes))

    def test_europe_pmc_shares_pubmeds_lineage(self) -> None:
        connector = EuropePMCConnector(cache=self.cache)
        connector.store_import("fetch", "PMC1",
                               {"records": [{"identifier": "PMC1"}]})
        citation = connector.citation("PMC1")
        self.assertIn("pubmed", citation.evidence.upstream_sources)
        self.assertIn("europe_pmc", citation.evidence.upstream_sources)


# ---------------------------------------------------------------------------
# full-text licensing
# ---------------------------------------------------------------------------

class TestFullTextLicensing(LiteratureTestCase):

    def _connector(self, **rows: dict) -> EuropePMCConnector:
        connector = EuropePMCConnector(cache=self.cache)
        for article_id, row in rows.items():
            payload = {"identifier": article_id}
            payload.update(row)
            connector.store_import("fetch", article_id,
                                   {"records": [payload]})
        return connector

    def test_the_allow_list_is_explicit(self) -> None:
        self.assertIn("cc-by-4.0", OPEN_REDISTRIBUTABLE_LICENCES)
        self.assertTrue(is_open_redistributable("CC BY 4.0"))
        self.assertTrue(is_open_redistributable("cc0-1.0"))
        self.assertFalse(is_open_redistributable("CC BY-NC 4.0"))
        self.assertFalse(is_open_redistributable("CC BY-ND 4.0"))
        self.assertFalse(is_open_redistributable(""))
        self.assertFalse(is_open_redistributable(None))

    def test_an_open_licence_returns_the_text(self) -> None:
        connector = self._connector(PMC1={"license": "CC BY 4.0",
                                          "full_text": "the body"})
        result = connector.full_text("PMC1")
        self.assertTrue(result.redistributable)
        self.assertEqual(result.require_text(), "the body")

    def test_a_closed_licence_withholds_the_text_and_marks_it(self) -> None:
        connector = self._connector(PMC2={"license": "CC BY-NC 4.0",
                                          "full_text": "the body"})
        result = connector.full_text("PMC2")
        self.assertFalse(result.redistributable)
        self.assertIsNone(result.text)
        self.assertIn("allow-list", result.reason)
        with self.assertRaises(NotRedistributableError):
            result.require_text()

    def test_an_unstated_licence_is_not_an_open_one(self) -> None:
        connector = self._connector(PMC3={"full_text": "the body"})
        result = connector.full_text("PMC3")
        self.assertFalse(result.redistributable)
        self.assertIsNone(result.text)
        self.assertIn("unstated licence is not an open one", result.reason)

    def test_an_open_access_flag_without_a_licence_is_not_enough(self) -> None:
        connector = self._connector(PMC4={"is_open_access": True,
                                          "full_text": "the body"})
        result = connector.full_text("PMC4")
        self.assertFalse(result.redistributable)
        self.assertIsNone(result.text)

    def test_a_missing_article_is_a_miss_not_a_licence_decision(self) -> None:
        result = EuropePMCConnector(cache=self.cache).full_text("PMC9")
        self.assertFalse(result.redistributable)
        self.assertIsNone(result.licence)
        self.assertIn("no text was synthesised", result.reason)

    def test_an_open_licence_with_no_text_returns_no_text(self) -> None:
        connector = self._connector(PMC5={"license": "cc0-1.0"})
        result = connector.full_text("PMC5")
        self.assertTrue(result.redistributable)
        self.assertIsNone(result.text)
        with self.assertRaises(NotRedistributableError):
            result.require_text()

    def test_the_source_level_redistribution_flag_is_still_unknown(self) -> None:
        self.assertIs(
            self.registry.get("europe_pmc").capabilities.redistribution_allowed,
            CapabilityState.UNKNOWN)

    def test_the_result_serialises_without_leaking_the_text(self) -> None:
        connector = self._connector(PMC1={"license": "CC BY 4.0",
                                          "full_text": "the body"})
        encoded = connector.full_text("PMC1").to_dict()
        self.assertNotIn("the body", str(encoded))
        self.assertTrue(encoded["text_present"])


# ---------------------------------------------------------------------------
# EnzChemRED assets
# ---------------------------------------------------------------------------

class TestEnzChemREDAssetsStaySeparate(LiteratureTestCase):

    def _connector(self) -> EnzChemREDConnector:
        connector = EnzChemREDConnector(cache=self.cache)
        connector.store_import(
            "search", {"asset": "expert_annotated", "slice": "all"},
            {"records": [{"annotation_id": "e1", "publication_id": "PMID:1",
                          "uniprot_accession": "P00000",
                          "chebi_ids": ["CHEBI:1"], "rhea_ids": ["RHEA:1"]}]})
        connector.store_import(
            "search", {"asset": "text_mined", "slice": "all"},
            {"records": [{"annotation_id": "t1", "publication_id": "PMID:2",
                          "uniprot_accession": "P00001",
                          "text_span": "reduced the ketone"}]})
        return connector

    def test_the_two_assets_are_different_cache_keys(self) -> None:
        connector = self._connector()
        expert = connector.corpus(EnzChemREDAsset.EXPERT_ANNOTATED)
        mined = connector.corpus(EnzChemREDAsset.TEXT_MINED)
        self.assertNotEqual(expert.response.cache_key, mined.response.cache_key)
        self.assertNotEqual(expert.storage_partition, mined.storage_partition)
        self.assertEqual([a.annotation_id for a in expert.annotations], ["e1"])
        self.assertEqual([a.annotation_id for a in mined.annotations], ["t1"])

    def test_merging_the_two_assets_is_refused(self) -> None:
        connector = self._connector()
        expert = connector.corpus(EnzChemREDAsset.EXPERT_ANNOTATED)
        mined = connector.corpus(EnzChemREDAsset.TEXT_MINED)
        with self.assertRaises(CorpusMergeRefusedError) as ctx:
            expert.merged_with(mined)
        self.assertIn("cannot be recovered", str(ctx.exception))
        with self.assertRaises(CorpusMergeRefusedError):
            mined.merged_with(expert)

    def test_concatenating_one_asset_with_itself_is_also_refused_here(self) -> None:
        connector = self._connector()
        expert = connector.corpus(EnzChemREDAsset.EXPERT_ANNOTATED)
        with self.assertRaises(CorpusMergeRefusedError):
            expert.merged_with(expert)

    def test_machine_extracted_rows_enter_at_the_pending_review_tier(self) -> None:
        mined = self._connector().corpus(EnzChemREDAsset.TEXT_MINED)
        self.assertIs(mined.asset.tier, EvidenceTier.MACHINE_EXTRACTED_PENDING)
        self.assertIs(mined.asset.extraction_method,
                      ExtractionMethod.MACHINE_EXTRACTION_LLM)
        records = mined.to_intake_records(self.registry)
        self.assertEqual(len(records), 1)
        self.assertIs(records[0].tier, EvidenceTier.MACHINE_EXTRACTED_PENDING)
        self.assertIs(records[0].claimed_strength,
                      EvidenceStrength.ANNOTATION_ONLY)
        self.assertFalse(records[0].is_usable_as_label)
        self.assertIn("triage", " ".join(records[0].uncertainties))

    def test_expert_rows_enter_at_the_curated_tier_not_above_it(self) -> None:
        """Somebody else's expert is not this project's named reviewer."""
        expert = self._connector().corpus(EnzChemREDAsset.EXPERT_ANNOTATED)
        records = expert.to_intake_records(self.registry)
        self.assertIs(records[0].tier, EvidenceTier.CURATED_DATABASE)
        self.assertNotEqual(records[0].tier,
                            EvidenceTier.EXPERT_VERIFIED_PRIMARY)
        self.assertIsNone(records[0].reviewer)

    def test_the_two_tiers_never_share_a_storage_partition(self) -> None:
        connector = self._connector()
        expert = connector.corpus(EnzChemREDAsset.EXPERT_ANNOTATED)
        mined = connector.corpus(EnzChemREDAsset.TEXT_MINED)
        self.assertNotEqual(
            expert.to_intake_records(self.registry)[0].partition,
            mined.to_intake_records(self.registry)[0].partition)

    def test_an_annotation_is_a_relation_not_a_result(self) -> None:
        expert = self._connector().corpus(EnzChemREDAsset.EXPERT_ANNOTATED)
        record = expert.annotations[0].to_experiment_record()
        self.assertIs(record.outcome, OutcomeClass.NOT_TESTED)
        self.assertEqual(record.reaction_id, "RHEA:1")
        self.assertIn("not that an assay was performed", record.notes)

    def test_the_lineage_and_the_asset_travel_with_the_corpus(self) -> None:
        expert = self._connector().corpus(EnzChemREDAsset.EXPERT_ANNOTATED)
        self.assertIn("enzchemred", expert.upstream_sources)
        self.assertIn("pubmed", expert.upstream_sources)
        self.assertIn("incomplete", " ".join(expert.notes))
        self.assertEqual(expert.annotations[0].evidence.locator,
                         "asset=expert_annotated")

    def test_an_unknown_asset_name_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._connector().corpus("semi_curated")

    def test_a_missing_slice_misses_rather_than_returning_the_other_asset(self) -> None:
        connector = self._connector()
        empty = connector.corpus(EnzChemREDAsset.TEXT_MINED, slice_id="2024")
        self.assertIs(empty.response.status, ResponseStatus.MISS)
        self.assertEqual(empty.annotations, ())


# ---------------------------------------------------------------------------
# Zenodo
# ---------------------------------------------------------------------------

class TestZenodoDeposits(LiteratureTestCase):

    def _deposit(self, **row: object):
        connector = ZenodoConnector(cache=self.cache)
        base = {"deposit_id": "1234567", "concept_doi": "10.5281/zenodo.1234566",
                "version_doi": "10.5281/zenodo.1234567", "version": "v2",
                "title": "KRED screening data", "license": "cc-by-4.0",
                "files": [{"filename": "data.csv", "checksum": "md5:abc"}]}
        base.update(row)
        connector.store_import("fetch", str(base["deposit_id"]),
                               {"records": [base]})
        return connector.deposit(str(base["deposit_id"]))

    def test_a_version_doi_is_what_pins_a_run(self) -> None:
        deposit = self._deposit()
        self.assertEqual(deposit.require_version_doi(),
                         "10.5281/zenodo.1234567")
        self.assertEqual(deposit.evidence.database_version, "v2")

    def test_a_concept_doi_alone_is_refused(self) -> None:
        deposit = self._deposit(deposit_id="7654321", version_doi=None)
        with self.assertRaises(LayerSemanticsError) as ctx:
            deposit.require_version_doi()
        self.assertIn("latest version", str(ctx.exception))
        self.assertIn("no version DOI recorded", " ".join(deposit.notes))

    def test_the_licence_is_read_per_deposit(self) -> None:
        self.assertTrue(self._deposit().redistributable)
        closed = self._deposit(deposit_id="2", license="CC BY-NC 4.0")
        self.assertFalse(closed.redistributable)
        self.assertIn("per deposit", " ".join(closed.notes))

    def test_a_deposit_without_a_licence_is_not_redistributable(self) -> None:
        deposit = self._deposit(deposit_id="3", license=None)
        self.assertFalse(deposit.redistributable)
        self.assertIn("not redistributable", " ".join(deposit.notes))

    def test_a_file_without_a_checksum_is_flagged(self) -> None:
        deposit = self._deposit(deposit_id="4",
                                files=[{"filename": "data.csv"}])
        self.assertIn("no recorded checksum", " ".join(deposit.notes))

    def test_a_missing_deposit_is_none_not_a_placeholder(self) -> None:
        self.assertIsNone(ZenodoConnector(cache=self.cache).deposit("99"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
