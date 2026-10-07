"""Tests for the sequence and family connectors.

The failures guarded here are the ones that are invisible afterwards:

* an unpublished construct is sent to a remote search service, which is
  irreversible and which no later policy decision undoes;
* a UniProt annotation is read as an experimental result for that sequence, so
  a seed set fills with proteins nobody assayed;
* a UniRef cluster or an MGnify gene call is counted as data rather than as a
  computational construct;
* a hit in InterPro and a hit in Pfam are counted as two family signals;
* a browse-only family resource acquires an imaginary client.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eagent.connectors.base import (
    AccessPolicy,
    FileCache,
    ResponseStatus,
    SubmissionAuthorization,
    UnauthorizedSubmissionError,
    looks_like_biological_sequence,
)
from eagent.connectors.chemistry import (
    UNVERIFIED_CAPABILITY,
    ConnectorConfigurationError,
    CuratedImportError,
    EndpointNotEstablishedError,
    LayerSemanticsError,
    OfflineImportOnlyError,
    RegistryBackedConnector,
    datasource_registry,
)
from eagent.connectors.sequence import (
    AKRSuperfamilyImporter,
    AnnotationEvidence,
    AnnotationNotEvidenceError,
    InterProConnector,
    MGnifyProteinsConnector,
    NCBIIdenticalProteinGroupsConnector,
    PfamConnector,
    SDREDImporter,
    UniParcConnector,
    UniProtKBConnector,
    UniRefConnector,
)
from eagent.datalayer.registry import CapabilityState
from eagent.provenance import sequence_hash
from eagent.schemas.record import EvidenceStrength, OutcomeClass

#: A sequence-shaped string used throughout. Long and varied enough that
#: ``looks_like_biological_sequence`` classifies it as a sequence, which is the
#: precondition for the disclosure guard to fire at all.
SEQ = ("MKAVVLYESNGPEVLQLKEVPKPEPGPGEVLIKVEAAGVCHSDLHLIDGELPFPLPVVLGHEG"
       "AGVVEAVGPGVTHVKPGDHVVLSWIPACGKCRACLNPQSNLCLKNDLSQPTGLMQDGTSRFT"
       "CRGKPIHHFLGTSTFSQYTVVSEISLAKIDPEAPLDKVCLLGCGVTTGYGAAVNTAKVEPGS")

OTHER_SEQ = ("MSNQVALVTGASRGIGRAIALELARRGFDLALNDRSAEAVEATAAEIRALGRRALAVAGDVS"
             "DEAFCRELVARTVEEFGRLDILVNNAGITRDTLLLRMKDEDWDAVLDTNLKGAFNCIRAATP")


class SequenceTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cache = FileCache(self.root)
        self.registry = datasource_registry()
        self.addCleanup(self._tmp.cleanup)

    def authorised_policy(self, *sequences: str,
                          scope: str = "*") -> AccessPolicy:
        grant = SubmissionAuthorization(
            authorized_by="operator:Jane Doe", scope=scope,
            justification="published reference sequences, cleared for search",
            sequence_sha256=tuple(sequence_hash(s) for s in sequences))
        return AccessPolicy(authorizations=(grant,))


# ---------------------------------------------------------------------------
# disclosure
# ---------------------------------------------------------------------------

class TestSequenceDisclosure(SequenceTestCase):

    def test_the_test_sequence_is_recognised_as_a_sequence(self) -> None:
        self.assertTrue(looks_like_biological_sequence(SEQ))
        self.assertTrue(looks_like_biological_sequence(OTHER_SEQ))

    def test_an_unauthorised_sequence_cannot_leave_the_process(self) -> None:
        connector = UniProtKBConnector(cache=self.cache)
        with self.assertRaises(UnauthorizedSubmissionError) as ctx:
            connector.sequence_query(SEQ)
        self.assertEqual(ctx.exception.sequence_sha256, sequence_hash(SEQ))
        self.assertEqual(ctx.exception.connector, "uniprotkb")
        self.assertIn("not covered by any submission authorisation",
                      str(ctx.exception))

    def test_the_default_policy_authorises_nothing(self) -> None:
        policy = AccessPolicy()
        self.assertEqual(policy.authorizations, ())
        self.assertEqual(policy.public_sequence_sha256, frozenset())
        with self.assertRaises(UnauthorizedSubmissionError):
            UniProtKBConnector(cache=self.cache,
                               access=policy).sequence_query(SEQ)

    def test_the_refusal_happens_even_though_the_network_is_off(self) -> None:
        """A guard that only fires online is never exercised before it matters."""
        connector = UniProtKBConnector(cache=self.cache)
        self.assertFalse(connector.access.allow_network)
        with self.assertRaises(UnauthorizedSubmissionError):
            connector.sequence_query(SEQ)

    def test_a_grant_for_another_sequence_does_not_cover_this_one(self) -> None:
        connector = UniProtKBConnector(
            cache=self.cache, access=self.authorised_policy(OTHER_SEQ))
        with self.assertRaises(UnauthorizedSubmissionError):
            connector.sequence_query(SEQ)

    def test_a_grant_scoped_to_another_connector_does_not_cover_this_one(self) -> None:
        connector = UniProtKBConnector(
            cache=self.cache,
            access=self.authorised_policy(SEQ, scope="uniparc"))
        with self.assertRaises(UnauthorizedSubmissionError):
            connector.sequence_query(SEQ)

    def test_a_named_grant_lets_the_query_through_and_is_recorded(self) -> None:
        connector = UniProtKBConnector(
            cache=self.cache,
            access=self.authorised_policy(SEQ, scope="uniprotkb"))
        response = connector.sequence_query(SEQ)
        self.assertIs(response.status, ResponseStatus.MISS)
        disclosure = [n for n in response.notes if n.startswith("disclosed to")]
        self.assertTrue(disclosure)
        self.assertIn("operator:Jane Doe", disclosure[0])
        self.assertIn(sequence_hash(SEQ), disclosure[0])

    def test_an_already_public_sequence_is_recorded_as_such(self) -> None:
        connector = UniProtKBConnector(
            cache=self.cache,
            access=AccessPolicy(public_sequence_sha256={sequence_hash(SEQ)}))
        response = connector.sequence_query(SEQ)
        self.assertIn("already public",
                      " ".join(n for n in response.notes
                               if n.startswith("disclosed to")))

    def test_an_authorisation_covering_nothing_is_refused_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            SubmissionAuthorization(authorized_by="operator:Jane Doe",
                                    scope="*", justification="because")
        with self.assertRaises(ValueError):
            SubmissionAuthorization(authorized_by="", scope="*",
                                    justification="because",
                                    any_sequence=True)

    def test_the_query_is_keyed_on_the_hash_not_the_sequence(self) -> None:
        """The cache key and the run manifest both retain whatever is in it."""
        connector = UniProtKBConnector(
            cache=self.cache, access=self.authorised_policy(SEQ))
        response = connector.sequence_query(SEQ)
        self.assertNotIn(SEQ, str(response.query))
        self.assertEqual(response.query["sequence_sha256"], sequence_hash(SEQ))

    def test_every_sequence_capable_connector_guards_the_same_way(self) -> None:
        for factory in (UniProtKBConnector, UniParcConnector, UniRefConnector,
                        InterProConnector, PfamConnector,
                        NCBIIdenticalProteinGroupsConnector,
                        MGnifyProteinsConnector):
            with self.subTest(connector=factory.__name__):
                with self.assertRaises(UnauthorizedSubmissionError):
                    factory(cache=self.cache).sequence_query(SEQ)

    def test_an_empty_sequence_query_is_refused(self) -> None:
        with self.assertRaises(LayerSemanticsError):
            UniProtKBConnector(cache=self.cache).sequence_query("   ")


# ---------------------------------------------------------------------------
# offline-first and endpoints
# ---------------------------------------------------------------------------

class TestOfflineAndEndpoints(SequenceTestCase):

    def test_a_cache_miss_is_structured(self) -> None:
        response = UniProtKBConnector(cache=self.cache).fetch("P00000")
        self.assertIs(response.status, ResponseStatus.MISS)
        self.assertIsNone(response.payload)
        self.assertTrue(response.needed)
        self.assertIsNone(UniProtKBConnector(cache=self.cache).entry("P00000"))

    def test_a_cache_hit_returns_the_curated_entry(self) -> None:
        connector = UniProtKBConnector(cache=self.cache)
        connector.store_import("fetch", "P00000", {"records": [
            {"accession": "P00000", "entry_name": "ADH_TEST", "sequence": SEQ,
             "organism": "Escherichia coli", "reviewed": True}]},
            database_version="2024_01")
        entry = connector.entry("P00000")
        self.assertEqual(entry.accession, "P00000")
        self.assertEqual(entry.sequence, SEQ)
        self.assertEqual(entry.sequence_sha256, sequence_hash(SEQ))
        self.assertTrue(entry.reviewed)
        self.assertEqual(entry.response.database_version, "2024_01")

    def test_an_endpoint_appears_only_with_a_verified_route(self) -> None:
        for source_id in ("uniprotkb", "uniref", "uniparc", "interpro", "pfam",
                          "ncbi_protein", "mgnify_proteins"):
            with self.subTest(source=source_id):
                source = self.registry.get(source_id)
                if source.endpoint is None:
                    continue
                self.assertTrue(source.connectivity_verified, source_id)
                self.assertTrue(source.verified_capabilities, source_id)

    def test_most_of_them_still_have_no_route_at_all(self) -> None:
        for source_id in ("uniref", "uniparc", "interpro", "pfam",
                          "ncbi_protein", "mgnify_proteins"):
            with self.subTest(source=source_id):
                self.assertIsNone(self.registry.get(source_id).endpoint)

    def test_require_endpoint_is_a_typed_refusal(self) -> None:
        with self.assertRaises(EndpointNotEstablishedError) as ctx:
            UniRefConnector(cache=self.cache).require_endpoint()
        self.assertEqual(ctx.exception.source_id, "uniref")
        self.assertTrue(ctx.exception.curation_notes)

    def test_a_verified_base_url_does_not_license_a_call(self) -> None:
        """UniProt's REST base is established; this client was not checked
        against it, and the generic <base>/<key> shape is not its API."""
        from eagent.connectors.chemistry import RequestShapeNotVerifiedError
        connector = UniProtKBConnector(cache=self.cache)
        self.assertEqual(connector.source.endpoint, "https://rest.uniprot.org")
        self.assertIsNone(connector.verified_route_capability)
        with self.assertRaises(RequestShapeNotVerifiedError) as ctx:
            connector.require_endpoint()
        self.assertIn("not a verified request", str(ctx.exception))


# ---------------------------------------------------------------------------
# capabilities
# ---------------------------------------------------------------------------

class TestCapabilities(SequenceTestCase):

    def test_a_not_supported_capability_is_refused(self) -> None:
        connector = UniProtKBConnector(cache=self.cache)
        self.assertIs(connector.capability("chemical_structure_query").state,
                      CapabilityState.NOT_SUPPORTED)
        response = connector.guarded("search", {"smiles": "CC(=O)c1ccccc1"},
                                     "chemical_structure_query")
        self.assertIs(response.status, ResponseStatus.REFUSED)
        self.assertIsNone(response.payload)
        self.assertIn("not_supported", response.miss_reason or "")

    def test_an_unknown_sequence_capability_is_marked_unverified(self) -> None:
        connector = UniRefConnector(
            cache=self.cache,
            access=AccessPolicy(public_sequence_sha256={sequence_hash(SEQ)}))
        self.assertIs(connector.capability("sequence_query").state,
                      CapabilityState.UNKNOWN)
        response = connector.sequence_query(SEQ)
        self.assertTrue(any(n.startswith(UNVERIFIED_CAPABILITY)
                            for n in response.notes))

    def test_a_supported_sequence_capability_is_not_marked_unverified(self) -> None:
        connector = UniProtKBConnector(
            cache=self.cache,
            access=AccessPolicy(public_sequence_sha256={sequence_hash(SEQ)}))
        self.assertIs(connector.capability("sequence_query").state,
                      CapabilityState.SUPPORTED)
        response = connector.sequence_query(SEQ)
        self.assertFalse(any(n.startswith(UNVERIFIED_CAPABILITY)
                             for n in response.notes))

    def test_unknown_and_not_supported_never_collapse(self) -> None:
        interpro = InterProConnector(cache=self.cache)
        self.assertIs(interpro.capability("sequence_query").state,
                      CapabilityState.UNKNOWN)
        self.assertIs(interpro.capability("chemical_structure_query").state,
                      CapabilityState.NOT_SUPPORTED)
        self.assertTrue(interpro.capability("sequence_query").allowed)
        self.assertFalse(interpro.capability("chemical_structure_query").allowed)


# ---------------------------------------------------------------------------
# UniProt annotations
# ---------------------------------------------------------------------------

class TestUniProtAnnotations(SequenceTestCase):

    def _entry(self, *annotations: dict):
        connector = UniProtKBConnector(cache=self.cache)
        connector.store_import("fetch", "P00000", {"records": [
            {"accession": "P00000", "sequence": SEQ,
             "ec_numbers": ["1.1.1.1"], "rhea_ids": ["RHEA:1"],
             "annotations": list(annotations)}]})
        return connector.entry("P00000")

    def test_the_evidence_code_is_kept_per_statement(self) -> None:
        entry = self._entry(
            {"kind": "catalytic_activity", "text": "reduces ketones",
             "evidence_code": "ECO:0000269"},
            {"kind": "function", "text": "alcohol dehydrogenase",
             "evidence_code": "ECO:0000256"},
            {"kind": "similarity", "text": "belongs to the SDR family",
             "evidence_code": "ECO:0000250"})
        self.assertEqual([a.evidence for a in entry.annotations],
                         [AnnotationEvidence.EXPERIMENTAL,
                          AnnotationEvidence.AUTOMATIC_ASSERTION,
                          AnnotationEvidence.SEQUENCE_SIMILARITY])
        self.assertEqual(entry.annotations[0].evidence_code, "ECO:0000269")

    def test_an_unknown_code_fails_closed_to_unstated(self) -> None:
        entry = self._entry({"kind": "function", "evidence_code": "ECO:9999999"})
        self.assertIs(entry.annotations[0].evidence, AnnotationEvidence.UNSTATED)
        self.assertFalse(entry.annotations[0].evidence.is_experimental_in_source)
        self.assertIn("no recognised evidence code", " ".join(entry.notes))

    def test_a_missing_code_is_not_experimental(self) -> None:
        entry = self._entry({"kind": "function", "text": "an ADH"})
        self.assertIs(entry.annotations[0].evidence, AnnotationEvidence.UNSTATED)

    def test_an_annotation_is_never_an_experimental_result(self) -> None:
        entry = self._entry({"kind": "catalytic_activity",
                             "evidence_code": "ECO:0000269"})
        self.assertEqual(len(entry.experimentally_evidenced()), 1)
        with self.assertRaises(AnnotationNotEvidenceError) as ctx:
            entry.as_experimental_evidence()
        self.assertIn("not a measurement on this sequence", str(ctx.exception))

    def test_an_annotated_ec_number_does_not_become_an_outcome(self) -> None:
        entry = self._entry({"kind": "catalytic_activity",
                             "evidence_code": "ECO:0000269"})
        record = entry.to_sequence_record()
        self.assertIs(record.outcome, OutcomeClass.NOT_TESTED)
        self.assertEqual(record.sequence_sha256, sequence_hash(SEQ))

    def test_projected_annotations_are_flagged(self) -> None:
        entry = self._entry({"kind": "function", "evidence_code": "ECO:0000250"})
        self.assertTrue(entry.annotations[0].evidence.is_projected)
        self.assertIn("projected by similarity", " ".join(entry.notes))

    def test_the_source_ceiling_is_annotation_only(self) -> None:
        self.assertIs(self.registry.get("uniprotkb").evidence_strength_ceiling,
                      EvidenceStrength.ANNOTATION_ONLY)
        self.assertIs(UniProtKBConnector(cache=self.cache).evidence_strength_ceiling,
                      EvidenceStrength.ANNOTATION_ONLY)


# ---------------------------------------------------------------------------
# computational constructs
# ---------------------------------------------------------------------------

class TestComputationalConstructs(SequenceTestCase):

    def test_a_uniref_cluster_is_not_function_evidence(self) -> None:
        connector = UniRefConnector(cache=self.cache)
        connector.store_import("fetch", "UniRef90_X", {"records": [
            {"cluster_id": "UniRef90_X", "identity": "90",
             "representative": "P00000", "members": ["P00000", "P00001"]}]})
        cluster = connector.cluster("UniRef90_X")
        self.assertEqual(cluster.member_accessions, ("P00000", "P00001"))
        with self.assertRaises(LayerSemanticsError):
            cluster.as_function_evidence()
        self.assertIs(self.registry.get("uniref").evidence_strength_ceiling,
                      EvidenceStrength.COMPUTATIONAL_CONSTRUCT)

    def test_a_metagenomic_call_needs_a_recorded_synthesis_gate(self) -> None:
        connector = MGnifyProteinsConnector(cache=self.cache)
        connector.store_import("fetch", "MGYP1", {"records": [
            {"mgnify_id": "MGYP1", "sequence": SEQ}]})
        record = connector.protein("MGYP1")
        self.assertIs(self.registry.get("mgnify_proteins").evidence_strength_ceiling,
                      EvidenceStrength.COMPUTATIONAL_CONSTRUCT)
        with self.assertRaises(LayerSemanticsError):
            record.require_synthesis_gate(None, True)
        with self.assertRaises(LayerSemanticsError):
            record.require_synthesis_gate("expression_screen", None)
        self.assertEqual(record.require_synthesis_gate("expression_screen", True),
                         "expression_screen")

    def test_missing_quality_flags_block_staging(self) -> None:
        connector = MGnifyProteinsConnector(cache=self.cache)
        connector.store_import("fetch", "MGYP2", {"records": [
            {"mgnify_id": "MGYP2", "sequence": SEQ}]})
        record = connector.protein("MGYP2")
        self.assertIsNone(record.completeness_flag)
        self.assertIn("cannot be staged", " ".join(record.notes))


# ---------------------------------------------------------------------------
# family signatures
# ---------------------------------------------------------------------------

class TestFamilySignatures(SequenceTestCase):

    def _hits(self, connector):
        connector.store_import("search", {"accession": "P00000"}, {"records": [
            {"entry_id": "IPR000001", "name": "SDR family", "type": "family",
             "accession": "P00000", "start": 5, "end": 240}]})
        return connector.signatures_for("P00000")

    def test_interpro_and_pfam_are_not_independent_signals(self) -> None:
        interpro = self._hits(InterProConnector(cache=self.cache))[0]
        pfam = self._hits(PfamConnector(cache=self.cache))[0]
        self.assertEqual(interpro.counts_independently_of, ("pfam",))
        self.assertEqual(pfam.counts_independently_of, ("interpro",))

    def test_a_signature_is_not_substrate_scope(self) -> None:
        hit = self._hits(InterProConnector(cache=self.cache))[0]
        with self.assertRaises(LayerSemanticsError) as ctx:
            hit.as_substrate_scope()
        self.assertIn("not substrate scope", str(ctx.exception))

    def test_positions_are_none_rather_than_defaulted(self) -> None:
        connector = PfamConnector(cache=self.cache)
        connector.store_import("search", {"accession": "P1"},
                               {"records": [{"entry_id": "PF00001"}]})
        hit = connector.signatures_for("P1")[0]
        self.assertIsNone(hit.start)
        self.assertIsNone(hit.end)


# ---------------------------------------------------------------------------
# identical protein groups
# ---------------------------------------------------------------------------

class TestIdenticalProteinGroups(SequenceTestCase):

    def _group(self):
        connector = NCBIIdenticalProteinGroupsConnector(cache=self.cache)
        connector.store_import("fetch", "IPG1", {"records": [
            {"group_id": "IPG1", "sequence": SEQ,
             "members": ["WP_000000001.1", "NP_000000001.1"],
             "databases": ["RefSeq", "GenBank"]}]})
        return connector.identical_protein_group("IPG1")

    def test_the_group_joins_on_the_sequence_hash(self) -> None:
        group = self._group()
        self.assertEqual(group.sequence_sha256, sequence_hash(SEQ))
        self.assertEqual(len(group.member_accessions), 2)

    def test_identical_sequence_does_not_license_merging_annotations(self) -> None:
        with self.assertRaises(LayerSemanticsError) as ctx:
            self._group().merge_annotations()
        self.assertIn("does not justify merging their annotations",
                      str(ctx.exception))


# ---------------------------------------------------------------------------
# importers
# ---------------------------------------------------------------------------

class TestFamilyImporters(SequenceTestCase):

    def test_both_resources_are_human_import_only(self) -> None:
        for source_id in ("sdred", "akr_superfamily"):
            with self.subTest(source=source_id):
                self.assertTrue(
                    self.registry.get(source_id).is_human_import_only)

    def test_an_offline_import_source_rejects_a_fetch(self) -> None:
        for importer in (SDREDImporter(), AKRSuperfamilyImporter()):
            with self.subTest(source=importer.source_id):
                with self.assertRaises(OfflineImportOnlyError) as ctx:
                    importer.fetch("AKR1A1")
                self.assertIn("import_file", str(ctx.exception))
                with self.assertRaises(OfflineImportOnlyError):
                    importer.search({"family": "AKR1"})

    def test_they_cannot_be_built_as_connectors(self) -> None:
        for source_id in ("sdred", "akr_superfamily"):
            with self.subTest(source=source_id):
                faked = type(f"Fake_{source_id}", (RegistryBackedConnector,),
                             {"source_id": source_id})
                with self.assertRaises(ConnectorConfigurationError):
                    faked(cache=self.cache)

    def test_a_fasta_export_imports_at_the_floor(self) -> None:
        path = self.root / "sdred.fasta"
        path.write_text(f">SDR001 short-chain dehydrogenase\n{SEQ}\n"
                        f">SDR002 another one\n{OTHER_SEQ}\n", encoding="utf-8")
        result = SDREDImporter().import_file(path,
                                             imported_by="operator:Jane Doe")
        self.assertEqual(len(result.records), 2)
        for record in result.records:
            self.assertIs(record.claimed_strength,
                          EvidenceStrength.ANNOTATION_ONLY)
            self.assertLessEqual(
                record.claimed_strength.rank,
                self.registry.get("sdred").evidence_strength_ceiling.rank)
        self.assertEqual(result.records[0].record.sequence_sha256,
                         sequence_hash(SEQ))

    def test_the_import_records_the_curator_and_the_file(self) -> None:
        path = self.root / "akr.fasta"
        path.write_text(f">AKR1A1 aldo-keto reductase\n{SEQ}\n", encoding="utf-8")
        result = AKRSuperfamilyImporter().import_file(
            path, imported_by="operator:Sam Patel")
        self.assertEqual(result.provenance.imported_by, "operator:Sam Patel")
        self.assertEqual(result.provenance.source_id, "akr_superfamily")
        self.assertEqual(result.provenance.file_format, ".fasta")
        self.assertIn("manual_review_import", result.provenance.access_modes)
        self.assertTrue(result.provenance.notes)

    def test_a_truncated_sequence_is_rejected_not_hashed(self) -> None:
        path = self.root / "sdred.fasta"
        path.write_text(">SDR003 truncated\nMKAV\n", encoding="utf-8")
        result = SDREDImporter().import_file(path,
                                             imported_by="operator:Jane Doe")
        self.assertEqual(len(result.records), 0)
        self.assertEqual(len(result.rejected), 1)
        self.assertIn("too short", result.rejected[0].reason)

    def test_a_headerless_fasta_is_refused(self) -> None:
        path = self.root / "bad.fasta"
        path.write_text(f"{SEQ}\n", encoding="utf-8")
        with self.assertRaises(CuratedImportError):
            SDREDImporter().import_file(path, imported_by="operator:Jane Doe")

    def test_every_row_is_flagged_for_per_record_review(self) -> None:
        path = self.root / "sdred.fasta"
        path.write_text(f">SDR001 x\n{SEQ}\n", encoding="utf-8")
        result = SDREDImporter().import_file(path,
                                             imported_by="operator:Jane Doe")
        self.assertIn("named reviewer", " ".join(result.warnings))
        self.assertIn("incomplete lineage", " ".join(result.warnings))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
