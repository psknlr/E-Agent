"""The first data route in this project that a run can actually use.

UniProtKB's REST base was reachable long before anything could read it. The
connector was written against a normalised payload so that a curator's export
and a live response reach one code path, and the half that turns a real
UniProt response into that payload had never existed -- so the live route was
unreachable however reachable the host was.

These tests run against responses recorded from the real service
(``tests/fixtures/uniprot``), not against a shape written from memory. Nothing
here touches the network.

The property they are really protecting is the one the registry's own
``not_good_for`` states: an inferred catalytic activity annotation reads
identically to a measured one. ``ECO:0000269`` cites a published experiment
and ``ECO:0000250`` says the statement was carried over from a homologue, and
if the translator drops the code the two become the same sentence.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from eagent.connectors.base import AccessPolicy, FileCache
from eagent.connectors.chemistry import RequestShapeNotVerifiedError
from eagent.connectors.sequence import (
    UNIPROT_ENTRY_FIELDS, AnnotationEvidence, AnnotationNotEvidenceError,
    UniProtKBConnector, uniprot_entry_payload,
)
from eagent.datalayer.registry import SourceRegistry

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "uniprot"


def recorded(accession: str) -> dict:
    return json.loads((FIXTURES / f"{accession}.json").read_text(encoding="utf-8"))


class TheTranslatorReadsARealResponse(unittest.TestCase):
    def setUp(self) -> None:
        self.sorbitol = uniprot_entry_payload(recorded("P07846"))
        self.aldo_keto = uniprot_entry_payload(recorded("P14550"))

    def test_the_identifying_fields_come_through(self) -> None:
        self.assertEqual(self.sorbitol["accession"], "P07846")
        self.assertEqual(self.sorbitol["entry_name"], "DHSO_SHEEP")
        self.assertEqual(self.sorbitol["organism"], "Ovis aries")
        self.assertEqual(len(self.sorbitol["sequence"]), 354)

    def test_reviewed_is_read_not_assumed(self) -> None:
        """A default of True would promote every TrEMBL record to Swiss-Prot."""
        self.assertIs(self.sorbitol["reviewed"], True)
        unreviewed = dict(recorded("P07846"))
        unreviewed["entryType"] = "UniProtKB unreviewed (TrEMBL)"
        self.assertIs(uniprot_entry_payload(unreviewed)["reviewed"], False)

    def test_an_entry_with_no_type_says_nothing_about_review(self) -> None:
        bare = {k: v for k, v in recorded("P07846").items() if k != "entryType"}
        self.assertIsNone(uniprot_entry_payload(bare)["reviewed"])

    def test_every_catalytic_activity_keeps_its_evidence_code(self) -> None:
        codes = {a["evidence_code"] for a in self.aldo_keto["annotations"]}
        self.assertIn("ECO:0000269", codes)
        self.assertIn("ECO:0000250", codes)

    def test_the_cited_experiment_travels_with_the_statement(self) -> None:
        sources = {a["source_identifier"] for a in self.sorbitol["annotations"]}
        self.assertTrue(any(s and s.startswith("PubMed:") for s in sources))

    def test_a_statement_with_several_codes_is_not_collapsed(self) -> None:
        """The strongest code would otherwise speak for the weakest."""
        entry = {
            "primaryAccession": "X", "entryType": "UniProtKB reviewed",
            "sequence": {"value": "MK"},
            "comments": [{
                "commentType": "CATALYTIC ACTIVITY",
                "reaction": {
                    "name": "a + b = c", "ecNumber": "1.1.1.1",
                    "evidences": [
                        {"evidenceCode": "ECO:0000269", "source": "PubMed",
                         "id": "1"},
                        {"evidenceCode": "ECO:0000250", "source": "UniProtKB",
                         "id": "P00000"},
                    ]}}]}
        payload = uniprot_entry_payload(entry)
        self.assertEqual(len(payload["annotations"]), 2)
        self.assertEqual({a["evidence_code"] for a in payload["annotations"]},
                         {"ECO:0000269", "ECO:0000250"})

    def test_a_statement_with_no_code_is_not_given_one(self) -> None:
        entry = {
            "primaryAccession": "X", "entryType": "UniProtKB reviewed",
            "sequence": {"value": "MK"},
            "comments": [{"commentType": "CATALYTIC ACTIVITY",
                          "reaction": {"name": "a = b"}}]}
        payload = uniprot_entry_payload(entry)
        self.assertEqual(payload["annotations"][0]["evidence_code"], None)

    def test_the_ec_numbers_are_the_ones_on_a_reaction(self) -> None:
        self.assertIn("1.1.1.9", self.sorbitol["ec_numbers"])
        self.assertIn("1.1.1.2", self.aldo_keto["ec_numbers"])

    def test_rhea_identifiers_are_carried_and_nothing_else_is(self) -> None:
        self.assertTrue(self.sorbitol["rhea_ids"])
        for rid in self.sorbitol["rhea_ids"]:
            self.assertTrue(rid.startswith("RHEA:"), rid)

    def test_pdb_cross_references_are_carried(self) -> None:
        self.assertIn("3QE3", self.sorbitol["pdb_ids"])

    def test_an_empty_entry_produces_empty_fields_not_invented_ones(self) -> None:
        payload = uniprot_entry_payload({})
        self.assertIsNone(payload["accession"])
        self.assertIsNone(payload["sequence"])
        self.assertEqual(payload["annotations"], [])


class TheConnectorBuildsTheProbedRequest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache = FileCache(pathlib.Path(self._tmp.name))
        self.calls: list[str] = []

    def connector(self, payload=None, *, status_version=None):
        connector = UniProtKBConnector(
            cache=self.cache, access=AccessPolicy(allow_network=True))
        body = recorded("P07846") if payload is None else payload

        def http_json(url, timeout=30.0):
            self.calls.append(url)
            return body, status_version

        connector._http_json = http_json          # the one outbound call
        return connector

    def test_the_url_matches_the_shape_the_probe_checked(self) -> None:
        self.connector()._fetch_remote("P07846")
        self.assertEqual(len(self.calls), 1)
        url = self.calls[0]
        self.assertTrue(url.startswith(
            "https://rest.uniprot.org/uniprotkb/P07846.json?fields="), url)

    def test_the_field_list_is_pinned_not_taken_from_the_caller(self) -> None:
        self.connector()._fetch_remote("P07846")
        for field in UNIPROT_ENTRY_FIELDS:
            self.assertIn(field, self.calls[0])

    def test_an_accession_is_quoted_into_the_path(self) -> None:
        self.connector()._fetch_remote("P 07846/../x")
        self.assertNotIn("/../", self.calls[0])

    def test_a_body_that_is_not_an_entry_is_not_translated(self) -> None:
        """An error object would otherwise become an entry with empty fields."""
        payload, _ = self.connector({"messages": ["not found"]})._fetch_remote("X")
        self.assertIsNone(payload)

    def test_the_release_is_whatever_the_service_sent(self) -> None:
        payload, version = self.connector(status_version=None)._fetch_remote("P07846")
        self.assertIsNone(version, "a run pinned to an invented string is not pinned")

    def test_the_entry_round_trips_through_the_connector(self) -> None:
        entry = self.connector().entry("P07846")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.accession, "P07846")
        self.assertEqual(len(entry.sequence or ""), 354)
        self.assertTrue((entry.sequence_sha256 or "").startswith("sha256:"))

    def test_the_hash_is_computed_from_the_sequence_that_came_back(self) -> None:
        """Not from the accession that was asked for: entries get re-annotated."""
        other = dict(recorded("P07846"))
        other["sequence"] = {"value": "MKAIV"}
        entry = self.connector(other).entry("P07846")
        assert entry is not None
        self.assertEqual(entry.sequence, "MKAIV")

    def test_projected_and_experimental_statements_stay_apart(self) -> None:
        entry = self.connector(recorded("P14550")).entry("P14550")
        assert entry is not None
        experimental = entry.experimentally_evidenced()
        projected = [a for a in entry.annotations if a.evidence.is_projected]
        self.assertTrue(experimental)
        self.assertTrue(projected)
        self.assertEqual(
            {a.evidence for a in projected},
            {AnnotationEvidence.SEQUENCE_SIMILARITY})

    def test_the_projection_is_said_out_loud(self) -> None:
        entry = self.connector(recorded("P14550")).entry("P14550")
        assert entry is not None
        self.assertTrue([n for n in entry.notes if "projected" in n])

    def test_an_experimental_code_is_still_not_experimental_evidence(self) -> None:
        """The whole point: a code is a pointer to somebody else's experiment."""
        entry = self.connector().entry("P07846")
        assert entry is not None
        self.assertTrue(entry.experimentally_evidenced())
        with self.assertRaises(AnnotationNotEvidenceError):
            entry.as_experimental_evidence()


class TheRouteIsDeclaredAndVerified(unittest.TestCase):
    def test_the_connector_declares_the_shape_it_was_checked_against(self) -> None:
        self.assertEqual(UniProtKBConnector.verified_route_capability,
                         "exact_record_fetch")

    def test_the_registry_holds_a_passing_check_for_that_capability(self) -> None:
        source = SourceRegistry.from_directory().get("uniprotkb")
        self.assertIn("exact_record_fetch", source.verified_capabilities)
        self.assertEqual(source.endpoint, "https://rest.uniprot.org")

    def test_the_probe_and_the_client_ask_for_the_same_thing(self) -> None:
        """If they drifted, the check would be verifying a route nobody uses."""
        from eagent.datalayer.probe import probes_for
        probe = next(p for p in probes_for("uniprotkb")
                     if p.capability == "exact_record_fetch")
        self.assertIn("/uniprotkb/", probe.url)
        self.assertIn(".json", probe.url)
        self.assertIn("fields=", probe.url)

    def test_a_verified_url_without_a_checked_client_still_refuses(self) -> None:
        """RCSB's base URL is verified. A client that declares no probed shape
        is refused anyway: a verified base is not a verified request."""
        from eagent.connectors.structure import RCSBPDBConnector
        with tempfile.TemporaryDirectory() as tmp:
            connector = RCSBPDBConnector(cache=FileCache(pathlib.Path(tmp)))
            connector.verified_route_capability = None     # an unchecked client
            self.assertIsNotNone(connector.source.endpoint)
            with self.assertRaises(RequestShapeNotVerifiedError):
                connector.require_endpoint()


if __name__ == "__main__":
    unittest.main(verbosity=2)
