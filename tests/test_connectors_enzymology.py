"""Tests for the enzymology-evidence connectors and importers.

The failures these guard against all produce a table that looks fine:

* four databases republishing one curated row are counted as four
  confirmations, so a weak candidate reads as corroborated;
* a row keyed to an EC number and an organism is read as evidence about one
  protein, so a family annotation becomes a measurement;
* an alcohol oxidation enters a ketone-reduction seed set as a positive;
* ``n.d.`` is read as "no activity", so a column of negatives appears that
  nobody measured;
* a browse-only resource acquires an imaginary REST client, and the run then
  schedules a step that will never happen.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eagent.connectors.base import AccessPolicy, FileCache, ResponseStatus
from eagent.connectors.chemistry import (
    UNVERIFIED_CAPABILITY,
    ConnectorConfigurationError,
    CuratedImportError,
    EndpointNotEstablishedError,
    OfflineImportOnlyError,
    RegistryBackedConnector,
    datasource_registry,
)
from eagent.connectors.enzymology import (
    BRENDAConnector,
    EnzEngDBConnector,
    MeasurementNotSequenceLevelError,
    OEDImporter,
    ProtaBankImporter,
    RetroBioCatDBImporter,
    SABIORKConnector,
    STRENDADBImporter,
    parse_reaction_direction,
)
from eagent.datalayer.intake import EvidenceTier, direction_check
from eagent.datalayer.lineage import count_independent
from eagent.datalayer.registry import CapabilityState
from eagent.schemas.reaction import ReactionClass
from eagent.schemas.record import (
    EvidenceStrength,
    OutcomeClass,
    ReactionDirection,
)

TARGET = ReactionClass.KETONE_TO_SECONDARY_ALCOHOL


class EnzymologyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cache = FileCache(self.root)
        self.registry = datasource_registry()
        self.addCleanup(self._tmp.cleanup)

    def brenda_with(self, *rows: dict) -> BRENDAConnector:
        connector = BRENDAConnector(cache=self.cache)
        connector.store_import("search", {"ec": "1.1.1.1"},
                               {"records": list(rows)})
        return connector


# ---------------------------------------------------------------------------
# offline-first
# ---------------------------------------------------------------------------

class TestOfflineFirst(EnzymologyTestCase):

    def test_a_miss_is_structured_and_carries_no_rows(self) -> None:
        response = BRENDAConnector(cache=self.cache).search({"ec": "1.1.1.1"})
        self.assertIs(response.status, ResponseStatus.MISS)
        self.assertIsNone(response.payload)
        self.assertTrue(response.needed)
        self.assertEqual(
            BRENDAConnector(cache=self.cache).measurements_for_ec("1.1.1.1"), ())

    def test_a_hit_returns_the_curated_rows(self) -> None:
        connector = self.brenda_with({"record_id": "r1", "ec": "1.1.1.1",
                                      "organism": "Escherichia coli",
                                      "substrate": "acetophenone"})
        measurements = connector.measurements_for_ec("1.1.1.1")
        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0].record_id, "r1")
        self.assertEqual(measurements[0].organism, "Escherichia coli")

    def test_the_null_endpoint_is_a_typed_refusal(self) -> None:
        for connector in (BRENDAConnector(cache=self.cache),
                          SABIORKConnector(cache=self.cache),
                          EnzEngDBConnector(cache=self.cache)):
            with self.subTest(source=connector.source_id):
                self.assertIsNone(connector.source.endpoint)
                with self.assertRaises(EndpointNotEstablishedError) as ctx:
                    connector.require_endpoint()
                self.assertEqual(ctx.exception.source_id, connector.source_id)
                self.assertTrue(ctx.exception.curation_notes)


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------

class TestCapabilities(EnzymologyTestCase):

    def test_sabio_rk_refuses_a_sequence_query(self) -> None:
        connector = SABIORKConnector(cache=self.cache)
        self.assertIs(connector.capability("sequence_query").state,
                      CapabilityState.NOT_SUPPORTED)
        response = connector.guarded("search", {"sequence": "MKV"},
                                     "sequence_query")
        self.assertIs(response.status, ResponseStatus.REFUSED)
        self.assertIsNone(response.payload)

    def test_brendas_unknown_fetch_is_marked_unverified(self) -> None:
        connector = BRENDAConnector(cache=self.cache)
        self.assertIs(connector.capability("exact_record_fetch").state,
                      CapabilityState.UNKNOWN)
        connector.store_import("fetch", "r1", {"records": [{"record_id": "r1"}]})
        response = connector.fetch("r1")
        self.assertIs(response.status, ResponseStatus.HIT)
        self.assertTrue(any(n.startswith(UNVERIFIED_CAPABILITY)
                            for n in response.notes))

    def test_brendas_supported_keyword_query_is_not_marked_unverified(self) -> None:
        connector = self.brenda_with({"record_id": "r1"})
        response = connector.search({"ec": "1.1.1.1"})
        self.assertIs(connector.capability("keyword_query").state,
                      CapabilityState.SUPPORTED)
        self.assertFalse(any(n.startswith(UNVERIFIED_CAPABILITY)
                             for n in response.notes))

    def test_enzengdb_has_no_verified_capability_at_all(self) -> None:
        connector = EnzEngDBConnector(cache=self.cache)
        connector.store_import("search", {"parent_accession": "P1"},
                               {"records": [{"record_id": "c1",
                                             "parent_accession": "P1",
                                             "mutations": ["A123V"]}]})
        campaigns = connector.campaigns_for({"parent_accession": "P1"})
        self.assertEqual(len(campaigns), 1)
        for name in ("keyword_query", "exact_record_fetch", "bulk_snapshot",
                     "version_information"):
            self.assertIs(connector.capability(name).state,
                          CapabilityState.UNKNOWN)
        response = connector.guarded("search", {"parent_accession": "P1"},
                                     "keyword_query")
        self.assertTrue(any(n.startswith(UNVERIFIED_CAPABILITY)
                            for n in response.notes))


# ---------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------

class TestUpstreamSourcesTravel(EnzymologyTestCase):

    def test_every_measurement_carries_its_lineage(self) -> None:
        measurement = self.brenda_with(
            {"record_id": "r1", "ec": "1.1.1.1"}).measurements_for_ec("1.1.1.1")[0]
        self.assertIn("brenda", measurement.upstream_sources)
        self.assertIn("brenda", measurement.evidence.upstream_sources)

    def test_a_rebublished_row_shares_the_upstream_of_its_origin(self) -> None:
        """OED re-integrates BRENDA, so the two are not independent."""
        self.assertIn("brenda", self.registry.get("oed").derived_from)
        oed_lineage = set(self.registry.lineage("oed"))
        self.assertIn("brenda", oed_lineage)
        self.assertIn("sabio_rk", oed_lineage)

    def test_lineage_collapses_two_rows_from_one_measurement(self) -> None:
        """The counting failure this layer exists to prevent."""
        brenda = self.brenda_with(
            {"record_id": "r1", "ec": "1.1.1.1", "uniprot_accession": "P00000",
             "substrate": "acetophenone", "substrate_smiles": "CC(=O)c1ccccc1",
             "measurement_type": "kcat", "value": 2.0, "unit": "s^-1",
             "pubmed_id": "PMID:1", "direction": "forward_as_target"})
        sabio = SABIORKConnector(cache=self.cache)
        sabio.store_import("search", {"ec": "1.1.1.1"}, {"records": [
            {"record_id": "s1", "ec": "1.1.1.1", "uniprot_accession": "P00000",
             "substrate": "acetophenone", "substrate_smiles": "CC(=O)c1ccccc1",
             "measurement_type": "kcat", "value": 2.0, "unit": "s^-1",
             "pubmed_id": "PMID:1", "direction": "forward_as_target",
             "pH": 7.0, "temperature_C": 30.0, "buffer": "phosphate"}]})
        rows = [brenda.measurements_for_ec("1.1.1.1")[0].to_experiment_record(),
                sabio.measurements_for_ec("1.1.1.1")[0].to_experiment_record()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(count_independent(rows), 1)


# ---------------------------------------------------------------------------
# direction
# ---------------------------------------------------------------------------

class TestDirectionIsPreserved(EnzymologyTestCase):

    def test_an_unrecorded_direction_is_not_a_forward_one(self) -> None:
        self.assertIs(parse_reaction_direction(None),
                      ReactionDirection.UNSPECIFIED)
        self.assertIs(parse_reaction_direction(""),
                      ReactionDirection.UNSPECIFIED)
        self.assertIs(parse_reaction_direction("whatever"),
                      ReactionDirection.UNSPECIFIED)
        measurement = self.brenda_with(
            {"record_id": "r1", "ec": "1.1.1.1"}).measurements_for_ec("1.1.1.1")[0]
        self.assertIs(measurement.reaction_direction,
                      ReactionDirection.UNSPECIFIED)
        self.assertFalse(measurement.supports_direction(TARGET))
        self.assertIn("no reaction direction", " ".join(measurement.caveats))

    def test_an_oxidation_row_does_not_support_the_reduction(self) -> None:
        measurement = self.brenda_with(
            {"record_id": "r1", "ec": "1.1.1.1", "direction": "reverse",
             "reaction_class": "alcohol_oxidation"}).measurements_for_ec("1.1.1.1")[0]
        self.assertIs(measurement.reaction_direction,
                      ReactionDirection.REVERSE_OF_TARGET)
        self.assertFalse(measurement.supports_direction(TARGET))
        verdict = direction_check(measurement.to_experiment_record(), TARGET)
        self.assertTrue(verdict.is_reverse)

    def test_a_forward_row_supports_the_target(self) -> None:
        measurement = self.brenda_with(
            {"record_id": "r1", "ec": "1.1.1.1", "direction": "forward",
             "reaction_class": "ketone_to_secondary_alcohol"}
        ).measurements_for_ec("1.1.1.1")[0]
        self.assertTrue(measurement.supports_direction(TARGET))


# ---------------------------------------------------------------------------
# sequence-level claims
# ---------------------------------------------------------------------------

class TestSequenceLevelClaims(EnzymologyTestCase):

    def test_an_ec_keyed_row_refuses_to_be_sequence_level(self) -> None:
        measurement = self.brenda_with(
            {"record_id": "r1", "ec": "1.1.1.1",
             "organism": "Escherichia coli"}).measurements_for_ec("1.1.1.1")[0]
        self.assertFalse(measurement.sequence_resolved)
        with self.assertRaises(MeasurementNotSequenceLevelError):
            measurement.require_sequence_level()

    def test_an_accession_keyed_row_names_its_protein(self) -> None:
        measurement = self.brenda_with(
            {"record_id": "r1", "ec": "1.1.1.1",
             "uniprot_accession": "P00000"}).measurements_for_ec("1.1.1.1")[0]
        self.assertEqual(measurement.require_sequence_level(), "P00000")

    def test_intake_stamps_the_floor_not_the_ceiling(self) -> None:
        """The ceiling is a cap; using it as a default is a guess upward."""
        measurement = self.brenda_with(
            {"record_id": "r1", "ec": "1.1.1.1",
             "uniprot_accession": "P00000"}).measurements_for_ec("1.1.1.1")[0]
        record = measurement.to_intake_record(self.registry)
        self.assertEqual(self.registry.get("brenda").evidence_strength_ceiling,
                         EvidenceStrength.EC_SPECIES_MAPPED)
        self.assertIs(record.claimed_strength, EvidenceStrength.ANNOTATION_ONLY)
        self.assertIs(record.tier, EvidenceTier.CURATED_DATABASE)
        self.assertFalse(record.is_usable_as_label)

    def test_a_kinetic_constant_is_not_an_outcome(self) -> None:
        measurement = self.brenda_with(
            {"record_id": "r1", "ec": "1.1.1.1", "measurement_type": "kcat",
             "value": 7.0, "unit": "s^-1"}).measurements_for_ec("1.1.1.1")[0]
        record = measurement.to_experiment_record()
        self.assertIs(record.outcome, OutcomeClass.NOT_TESTED)
        self.assertEqual(record.measurement_value, 7.0)


class TestSABIORKConditions(EnzymologyTestCase):

    def test_a_row_without_conditions_is_flagged_not_completed(self) -> None:
        connector = SABIORKConnector(cache=self.cache)
        connector.store_import("search", {"ec": "1.1.1.1"},
                               {"records": [{"record_id": "s1", "ec": "1.1.1.1"}]})
        measurement = connector.measurements_for_ec("1.1.1.1")[0]
        self.assertIsNone(measurement.conditions.pH)
        joined = " ".join(measurement.caveats)
        self.assertIn("pH", joined)
        self.assertIn("may not be compared", joined)

    def test_absence_of_an_entry_is_not_absence_of_activity(self) -> None:
        connector = SABIORKConnector(cache=self.cache)
        connector.store_import("search", {"ec": "9.9.9.9"},
                               {"records": [{"record_id": "s1"}]})
        measurement = connector.measurements_for_ec("9.9.9.9")[0]
        self.assertIn("absence of an entry", " ".join(measurement.caveats))


# ---------------------------------------------------------------------------
# importers
# ---------------------------------------------------------------------------

class TestImportersAreNotClients(EnzymologyTestCase):

    def test_an_offline_import_source_rejects_a_fetch(self) -> None:
        for importer in (OEDImporter(), RetroBioCatDBImporter(),
                         STRENDADBImporter(), ProtaBankImporter()):
            with self.subTest(source=importer.source_id):
                with self.assertRaises(OfflineImportOnlyError) as ctx:
                    importer.fetch("anything")
                self.assertIn("import_file", str(ctx.exception))
                with self.assertRaises(OfflineImportOnlyError):
                    importer.search({"ec": "1.1.1.1"})
                with self.assertRaises(EndpointNotEstablishedError):
                    importer.require_endpoint()

    def test_none_of_them_can_be_built_as_a_connector(self) -> None:
        for source_id in ("oed", "retrobiocat_db", "strenda_db", "protabank"):
            with self.subTest(source=source_id):
                self.assertTrue(
                    self.registry.get(source_id).is_human_import_only)

                namespace = {"source_id": source_id}
                faked = type(f"Fake_{source_id}", (RegistryBackedConnector,),
                             namespace)
                with self.assertRaises(ConnectorConfigurationError):
                    faked(cache=self.cache)

    def test_an_import_records_who_did_it_and_from_what(self) -> None:
        path = self.root / "oed.csv"
        path.write_text(
            "record_id,ec,substrate,measurement_type,value,unit\n"
            "A1,1.1.1.1,acetophenone,kcat,3.0,s^-1\n",
            encoding="utf-8")
        result = OEDImporter().import_file(path, imported_by="operator:Jane Doe",
                                           database_version="2024-02")
        self.assertEqual(result.provenance.imported_by, "operator:Jane Doe")
        self.assertEqual(result.provenance.n_rows_accepted, 1)
        self.assertEqual(result.provenance.database_version, "2024-02")
        self.assertEqual(len(result.provenance.file_sha256), 64)
        self.assertTrue(result.complete)
        self.assertIn("unknown", result.provenance.access_modes)

    def test_an_import_refuses_an_anonymous_importer(self) -> None:
        path = self.root / "oed.csv"
        path.write_text("record_id,ec\nA1,1.1.1.1\n", encoding="utf-8")
        with self.assertRaises(CuratedImportError):
            OEDImporter().import_file(path, imported_by="")

    def test_rows_are_rejected_loudly_not_dropped(self) -> None:
        path = self.root / "oed.csv"
        path.write_text("record_id,ec\nA1,1.1.1.1\n,1.1.1.2\n", encoding="utf-8")
        result = OEDImporter().import_file(path, imported_by="operator:Jane Doe")
        self.assertEqual(result.provenance.n_rows_read, 2)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(len(result.rejected), 1)
        self.assertFalse(result.complete)
        self.assertIn("record_id", result.rejected[0].reason)
        with self.assertRaises(CuratedImportError):
            OEDImporter().import_file(path, imported_by="operator:Jane Doe",
                                      strict=True)

    def test_an_unknown_file_format_is_refused(self) -> None:
        path = self.root / "oed.xlsx"
        path.write_text("not really a spreadsheet", encoding="utf-8")
        with self.assertRaises(CuratedImportError):
            OEDImporter().import_file(path, imported_by="operator:Jane Doe")

    def test_a_headerless_table_is_refused(self) -> None:
        path = self.root / "oed.csv"
        path.write_text("", encoding="utf-8")
        with self.assertRaises(CuratedImportError):
            OEDImporter().import_file(path, imported_by="operator:Jane Doe")

    def test_imported_rows_are_stamped_at_or_below_the_ceiling(self) -> None:
        path = self.root / "oed.csv"
        path.write_text("record_id,ec\nA1,1.1.1.1\n", encoding="utf-8")
        record = OEDImporter().import_file(
            path, imported_by="operator:Jane Doe").records[0]
        ceiling = self.registry.get("oed").evidence_strength_ceiling
        self.assertLessEqual(record.claimed_strength.rank, ceiling.rank)
        self.assertIs(record.claimed_strength, EvidenceStrength.ANNOTATION_ONLY)
        self.assertIn("imported by operator:Jane Doe",
                      " ".join(record.uncertainties))

    def test_the_incomplete_lineage_is_warned_about(self) -> None:
        path = self.root / "oed.csv"
        path.write_text("record_id,ec\nA1,1.1.1.1\n", encoding="utf-8")
        result = OEDImporter().import_file(path, imported_by="operator:Jane Doe")
        self.assertIn("incomplete lineage", " ".join(result.warnings))
        self.assertIn("de-duplicated",
                      " ".join(result.records[0].uncertainties))


class TestProtaBankImport(EnzymologyTestCase):
    """ProtaBank has no established route at all, so it is an importer."""

    def _import(self, body: str):
        path = self.root / "protabank.csv"
        path.write_text(body, encoding="utf-8")
        return ProtaBankImporter().import_file(
            path, imported_by="operator:Jane Doe")

    def test_a_mutation_without_a_parent_is_rejected(self) -> None:
        result = self._import("record_id,mutations\nP1,A12V\n")
        self.assertEqual(len(result.records), 0)
        self.assertIn("names no parent accession", result.rejected[0].reason)

    def test_rows_enter_at_the_floor_below_the_homolog_ceiling(self) -> None:
        result = self._import(
            "record_id,parent_accession,mutations\nP1,Q00001,A12V;L30F\n")
        self.assertIs(self.registry.get("protabank").evidence_strength_ceiling,
                      EvidenceStrength.HOMOLOG_EXPERIMENTAL)
        record = result.records[0]
        self.assertIs(record.claimed_strength, EvidenceStrength.ANNOTATION_ONLY)
        self.assertEqual(record.record.mutations, ["A12V", "L30F"])
        self.assertIn("does not transfer", " ".join(record.uncertainties))

    def test_double_counting_against_the_other_collections_is_flagged(self) -> None:
        result = self._import("record_id,parent_accession\nP1,Q00001\n")
        self.assertIn("may already have been counted",
                      " ".join(result.records[0].uncertainties))


class TestRetroBioCatImport(EnzymologyTestCase):

    def _import(self, body: str):
        path = self.root / "rbc.csv"
        path.write_text(body, encoding="utf-8")
        return RetroBioCatDBImporter().import_file(
            path, imported_by="operator:Jane Doe")

    def test_nd_does_not_become_a_negative(self) -> None:
        result = self._import(
            "record_id,enzyme_name,substrate,outcome\n"
            "R1,CbADH,acetophenone,n.d.\n")
        record = result.records[0].record
        self.assertIs(record.outcome, OutcomeClass.NOT_TESTED)
        self.assertNotEqual(record.outcome,
                            OutcomeClass.NO_TARGET_PRODUCT_DETECTED)

    def test_a_negative_without_a_detection_limit_degrades_to_not_tested(self) -> None:
        result = self._import(
            "record_id,enzyme_name,substrate,outcome\n"
            "R1,CbADH,acetophenone,no product detected\n")
        record = result.records[0].record
        self.assertIs(record.outcome, OutcomeClass.NOT_TESTED)
        self.assertIn("not storable", result.records[0].record.notes)

    def test_a_confirmed_product_needs_a_confirming_method(self) -> None:
        result = self._import(
            "record_id,enzyme_name,substrate,outcome,detection_method,"
            "confirms_product_identity\n"
            "R1,CbADH,acetophenone,product confirmed,chiral HPLC,yes\n")
        record = result.records[0].record
        self.assertIs(record.outcome, OutcomeClass.CONFIRMED_TARGET_PRODUCT)
        self.assertTrue(record.detection.confirms_product_identity)

    def test_a_prose_substrate_is_flagged(self) -> None:
        result = self._import(
            "record_id,enzyme_name,substrate\nR1,CbADH,acetophenone\n")
        self.assertIn("prose name",
                      " ".join(result.records[0].uncertainties))


class TestSTRENDAPerRowCeiling(EnzymologyTestCase):

    SEQ = ("MKAVVLYESNGPEVLQLKEVPKPEPGPGEVLIKVEAAGVCHSDLHLIDGELPFPLPVVLGH"
           "EGAGVVEAVGPGVTHVKPGDHVVLSW")

    def _import(self, body: str):
        path = self.root / "strenda.csv"
        path.write_text(body, encoding="utf-8")
        return STRENDADBImporter().import_file(
            path, imported_by="operator:Jane Doe")

    def test_the_registry_ceiling_is_sequence_level(self) -> None:
        self.assertIs(self.registry.get("strenda_db").evidence_strength_ceiling,
                      EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL)

    def test_a_row_with_no_sequence_is_capped_lower_and_says_so(self) -> None:
        result = self._import("record_id,uniprot_accession\nS1,P00000\n")
        notes = " ".join(result.records[0].uncertainties)
        self.assertIn("ec_species_mapped", notes)
        self.assertIn("no construct sequence", notes)

    def test_a_row_with_a_sequence_keeps_the_registry_ceiling(self) -> None:
        result = self._import(f"record_id,sequence\nS2,{self.SEQ}\n")
        notes = " ".join(result.records[0].uncertainties)
        self.assertNotIn("capped at ec_species_mapped", notes)
        self.assertIn("could later promote it", notes)

    def test_even_a_sequence_bearing_row_enters_at_the_floor(self) -> None:
        result = self._import(f"record_id,sequence\nS2,{self.SEQ}\n")
        record = result.records[0]
        self.assertIs(record.claimed_strength, EvidenceStrength.ANNOTATION_ONLY)
        self.assertIsNone(record.reviewer)
        self.assertFalse(record.is_usable_as_label)

    def test_every_row_needs_a_named_reviewer_before_use(self) -> None:
        result = self._import("record_id\nS3\n")
        self.assertTrue(
            self.registry.get("strenda_db").requires_human_review_per_record)
        self.assertIn("named reviewer", " ".join(result.warnings))


class TestEnzEngDBCampaigns(EnzymologyTestCase):

    def test_a_mutation_without_a_parent_is_refused(self) -> None:
        connector = EnzEngDBConnector(cache=self.cache)
        connector.store_import("search", {"q": "adh"}, {"records": [
            {"record_id": "c1", "mutations": ["A123V"]}]})
        campaign = connector.campaigns_for({"q": "adh"})[0]
        with self.assertRaises(Exception):
            campaign.to_experiment_record()

    def test_an_effect_does_not_transfer_to_another_parent(self) -> None:
        connector = EnzEngDBConnector(cache=self.cache)
        connector.store_import("search", {"q": "adh"}, {"records": [
            {"record_id": "c1", "parent_accession": "P1",
             "mutations": ["A123V"], "effect": "3-fold kcat"}]})
        campaign = connector.campaigns_for({"q": "adh"})[0]
        self.assertIn("does not transfer", " ".join(campaign.caveats))
        record = campaign.to_experiment_record()
        self.assertEqual(record.mutations, ["A123V"])
        self.assertIs(record.outcome, OutcomeClass.NOT_TESTED)


class TestPolicyIsHonoured(EnzymologyTestCase):

    def test_network_stays_off_by_default_for_every_connector_here(self) -> None:
        for connector in (BRENDAConnector(cache=self.cache),
                          SABIORKConnector(cache=self.cache),
                          EnzEngDBConnector(cache=self.cache)):
            with self.subTest(source=connector.source_id):
                self.assertFalse(connector.access.allow_network)

    def test_an_execution_policy_without_the_flag_reads_as_offline(self) -> None:
        class LegacyPolicy:
            pass

        access = AccessPolicy.from_execution_policy(LegacyPolicy())
        self.assertFalse(access.allow_network)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
