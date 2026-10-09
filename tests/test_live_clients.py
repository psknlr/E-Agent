"""RCSB and Rhea: clients written against recorded responses, and held to
the requests the probes actually checked.

A probe shows that one request, spelled one way, answered correctly. A client
that builds a different request has not been verified by it, and the registry
used to have no way to say so -- a verified base URL read as a working route.
These tests close the loop from the other side: every URL a checked client
builds must be one of the URLs a shipped probe fetched, so the two cannot
drift apart without a test failing.

Responses are recorded from the real services (``tests/fixtures``). Nothing
here touches the network.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from eagent.connectors.base import (
    AccessPolicy, FileCache, RemoteCallFailedError, ResponseStatus,
)
from eagent.connectors.chemistry import (
    RHEA_COLUMNS, RHEA_HEADER, RHEA_QUERY_LIMIT, RheaConnector,
    rhea_tsv_payload,
)
from eagent.connectors.sequence import (
    UNIPROT_ENTRY_FIELDS, UniProtKBConnector, uniprot_entry_payload,
)
from eagent.connectors.structure import RCSBPDBConnector, rcsb_entry_payload
from eagent.datalayer.probe import PROBES

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def rcsb(name: str) -> dict:
    return json.loads((FIXTURES / "rcsb" / name).read_text(encoding="utf-8"))


def rhea(name: str) -> str:
    return (FIXTURES / "rhea" / name).read_text(encoding="utf-8")


def probe_urls(source_id: str) -> set[str]:
    return {p.url for p in PROBES if p.source_id == source_id}


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache = FileCache(pathlib.Path(self._tmp.name))


# --------------------------------------------------------------------------
# RCSB
# --------------------------------------------------------------------------

class RCSBTranslator(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = rcsb_entry_payload(
            rcsb("entry_1CDO.json"),
            {"1": rcsb("polymer_entity_1CDO_1.json")},
            {"2": rcsb("nonpolymer_entity_1CDO_2.json"),
             "3": rcsb("nonpolymer_entity_1CDO_3.json")})

    def test_the_entry_level_facts(self) -> None:
        self.assertEqual(self.payload["pdb_id"], "1CDO")
        self.assertEqual(self.payload["method"], "X-RAY DIFFRACTION")
        self.assertEqual(self.payload["resolution"], 2.05)

    def test_every_chain_gets_the_sequence_of_its_entity(self) -> None:
        self.assertEqual(self.payload["chains"], ["A", "B"])
        self.assertEqual(set(self.payload["sequences"]), {"A", "B"})
        self.assertEqual(len(self.payload["sequences"]["A"]), 374)

    def test_a_ligand_on_two_chains_is_two_ligands(self) -> None:
        """NAD sits on both chains; one row per component would hide chain B's."""
        nad = [l for l in self.payload["ligands"] if l["component_id"] == "NAD"]
        self.assertEqual(sorted(l["chain"] for l in nad), ["A", "B"])

    def test_zinc_is_reported_alongside_the_cofactor(self) -> None:
        self.assertEqual({l["component_id"] for l in self.payload["ligands"]},
                         {"ZN", "NAD"})

    def test_occupancy_is_not_invented(self) -> None:
        """The core API does not carry it; 1.0 would silence the connector's
        own warning with a number nobody read."""
        for ligand in self.payload["ligands"]:
            self.assertIsNone(ligand["occupancy"])
            self.assertIsNone(ligand["author_seq_id"])

    def test_a_structure_with_no_resolution_has_none_not_zero(self) -> None:
        entry = rcsb("entry_1CDO.json")
        entry["rcsb_entry_info"] = {k: v for k, v in entry["rcsb_entry_info"].items()
                                    if k != "resolution_combined"}
        payload = rcsb_entry_payload(entry, {}, {})
        self.assertIsNone(payload["resolution"])

    def test_an_empty_entry_produces_empty_fields(self) -> None:
        payload = rcsb_entry_payload({}, {}, {})
        self.assertIsNone(payload["pdb_id"])
        self.assertEqual(payload["ligands"], [])
        self.assertEqual(payload["sequences"], {})


class RCSBClient(_Tmp):
    def connector(self, responses: dict[str, object]):
        connector = RCSBPDBConnector(
            cache=self.cache, access=AccessPolicy(allow_network=True))
        self.urls: list[str] = []

        def http_json(url, timeout=30.0):
            self.urls.append(url)
            for suffix, body in responses.items():
                if url.endswith(suffix):
                    return body, None
            return None, None

        connector._http_json = http_json
        return connector

    def full(self):
        return {
            "/core/entry/1CDO": rcsb("entry_1CDO.json"),
            "/core/polymer_entity/1CDO/1": rcsb("polymer_entity_1CDO_1.json"),
            "/core/nonpolymer_entity/1CDO/2": rcsb("nonpolymer_entity_1CDO_2.json"),
            "/core/nonpolymer_entity/1CDO/3": rcsb("nonpolymer_entity_1CDO_3.json"),
        }

    def test_the_entry_round_trips(self) -> None:
        record = self.connector(self.full()).entry("1cdo")
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.pdb_id, "1CDO")
        self.assertEqual({l.component_id for l in record.ligands}, {"ZN", "NAD"})

    def test_every_url_built_is_a_shape_a_probe_fetched(self) -> None:
        """The loop closed from the other side: if the client drifts from the
        probes, the probes verify a request nobody makes."""
        self.connector(self.full()).entry("1CDO")
        probed_shapes = {u.split("1CDO")[0] for u in probe_urls("rcsb_pdb")}
        self.assertEqual(len(self.urls), 4)
        for url in self.urls:
            self.assertIn(url.split("1CDO")[0], probed_shapes, url)

    def test_the_declared_route_matches_the_registry(self) -> None:
        self.assertEqual(RCSBPDBConnector.verified_route_capability,
                         "exact_record_fetch")

    def test_a_missing_entity_fails_the_whole_fetch(self) -> None:
        """A protein assembled without its cofactor is a different protein."""
        responses = self.full()
        del responses["/core/nonpolymer_entity/1CDO/3"]
        self.assertIsNone(self.connector(responses).entry("1CDO"))

    def test_an_unknown_entry_is_none_not_an_exception(self) -> None:
        self.assertIsNone(self.connector({}).entry("9ZZZ"))

    def test_the_id_is_quoted_into_the_path(self) -> None:
        self.connector({}).entry("1C/../DO")
        self.assertNotIn("/../", self.urls[0])


# --------------------------------------------------------------------------
# Rhea
# --------------------------------------------------------------------------

class RheaTranslator(unittest.TestCase):
    def test_the_ec_query_response(self) -> None:
        payload = rhea_tsv_payload(rhea("ec_1.1.1.1.tsv"))
        self.assertEqual([r["rhea_id"] for r in payload["records"]],
                         ["RHEA:10736", "RHEA:10740", "RHEA:25290"])

    def test_ec_numbers_are_split_and_stripped_of_their_prefix(self) -> None:
        first = rhea_tsv_payload(rhea("ec_1.1.1.1.tsv"))["records"][0]
        self.assertEqual(first["ec_numbers"], ["1.1.1.1", "1.1.1.71"])

    def test_direction_is_unstated_because_the_response_cannot_say(self) -> None:
        for record in rhea_tsv_payload(rhea("ec_1.1.1.1.tsv"))["records"]:
            self.assertEqual(record["direction"], "unstated")
            self.assertIsNone(record["master_id"])
            self.assertIsNone(record["reaction_class"])

    def test_participants_are_not_assigned_to_a_side(self) -> None:
        record = rhea_tsv_payload(rhea("rhea_10740.tsv"))["records"][0]
        self.assertEqual(record["participants_left"], [])
        self.assertEqual(record["participants_right"], [])
        self.assertIn("CHEBI:17087", record["participants"])

    def test_the_equation_is_kept_as_written(self) -> None:
        record = rhea_tsv_payload(rhea("rhea_10740.tsv"))["records"][0]
        self.assertEqual(
            record["equation"], "a secondary alcohol + NAD(+) = a ketone + NADH + H(+)")
        self.assertEqual(record["equation_right_text"], "a ketone + NADH + H(+)")

    def test_an_absent_reaction_is_an_empty_table_not_an_error(self) -> None:
        self.assertEqual(rhea_tsv_payload(rhea("rhea_absent.tsv"))["records"], [])

    def test_a_changed_header_is_refused(self) -> None:
        """The column list is the parser: a renamed column must not be read
        from the wrong field."""
        text = rhea("rhea_10740.tsv").replace("EC number", "Enzyme code", 1)
        with self.assertRaises(RemoteCallFailedError) as ctx:
            rhea_tsv_payload(text)
        self.assertIn("diverged", str(ctx.exception))

    def test_a_short_row_is_refused(self) -> None:
        text = rhea("rhea_10740.tsv").rstrip("\n") + "\nRHEA:1\tx\n"
        with self.assertRaises(RemoteCallFailedError):
            rhea_tsv_payload(text)

    def test_a_non_rhea_identifier_is_refused(self) -> None:
        header = "\t".join(RHEA_HEADER)
        with self.assertRaises(RemoteCallFailedError):
            rhea_tsv_payload(f"{header}\nCHEBI:1\ta = b\tEC:1.1.1.1\tCHEBI:2\n")

    def test_a_result_that_fills_the_limit_is_marked_truncated(self) -> None:
        header = "\t".join(RHEA_HEADER)
        rows = "\n".join(f"RHEA:{n}\ta = b\tEC:1.1.1.1\tCHEBI:1"
                         for n in range(RHEA_QUERY_LIMIT))
        self.assertTrue(rhea_tsv_payload(f"{header}\n{rows}\n")["truncated"])

    def test_a_short_result_is_not(self) -> None:
        self.assertFalse(rhea_tsv_payload(rhea("ec_1.1.1.1.tsv"))["truncated"])


class RheaClient(_Tmp):
    def connector(self, text_by_query: dict[str, str | None]):
        connector = RheaConnector(
            cache=self.cache, access=AccessPolicy(allow_network=True))
        self.urls: list[str] = []

        def http_text(url, timeout=30.0, accept="*/*"):
            self.urls.append(url)
            for fragment, body in text_by_query.items():
                if f"query={fragment}&" in url:
                    return body
            return None

        connector._http_text = http_text
        return connector

    def test_a_fetch_returns_exactly_the_reaction_asked_for(self) -> None:
        reaction = self.connector({"RHEA:10740": rhea("rhea_10740.tsv")}
                                  ).reaction("RHEA:10740")
        self.assertIsNotNone(reaction)
        assert reaction is not None
        self.assertEqual(reaction.rhea_id, "RHEA:10740")

    def test_a_query_that_returns_another_reaction_is_not_cached_as_this_one(self) -> None:
        """A fuzzy match under this key would put the wrong reaction in the
        cache for every later run."""
        connector = self.connector({"RHEA:99999": rhea("rhea_10740.tsv")})
        response = connector.fetch("RHEA:99999")
        self.assertIs(response.status, ResponseStatus.MISS)

    def test_an_absent_reaction_is_a_miss(self) -> None:
        response = self.connector({"RHEA:99999999": rhea("rhea_absent.tsv")}
                                  ).fetch("RHEA:99999999")
        self.assertIs(response.status, ResponseStatus.MISS)

    def test_an_ec_search_returns_every_reaction(self) -> None:
        reactions = self.connector({"ec:1.1.1.1": rhea("ec_1.1.1.1.tsv")}
                                   ).reactions_for_ec("1.1.1.1")
        self.assertEqual(len(reactions), 3)

    def test_the_direction_is_reported_as_unrecorded_not_forward(self) -> None:
        reaction = self.connector({"RHEA:10740": rhea("rhea_10740.tsv")}
                                  ).reaction("RHEA:10740")
        assert reaction is not None
        self.assertTrue([n for n in reaction.notes if "no direction" in n])

    def test_every_url_built_is_a_shape_a_probe_fetched(self) -> None:
        connector = self.connector({"RHEA:10740": rhea("rhea_10740.tsv"),
                                    "ec:1.1.1.1": rhea("ec_1.1.1.1.tsv")})
        connector.reaction("RHEA:10740")
        connector.reactions_for_ec("1.1.1.1")
        probed = probe_urls("rhea")
        self.assertEqual(len(self.urls), 2)
        for url in self.urls:
            self.assertIn(url, probed)

    def test_the_pinned_columns_are_the_ones_in_the_probe_url(self) -> None:
        for url in probe_urls("rhea"):
            self.assertIn(f"columns={','.join(RHEA_COLUMNS)}", url)
            self.assertIn(f"limit={RHEA_QUERY_LIMIT}", url)


# --------------------------------------------------------------------------
# UniProt, closed the same way
# --------------------------------------------------------------------------

class UniProtProbeMatchesClient(_Tmp):
    def test_the_probe_asks_for_exactly_the_clients_field_list(self) -> None:
        """It used to ask for three fields while the client asked for ten --
        verifying a request nobody makes."""
        entry_probe = next(p for p in PROBES
                           if p.source_id == "uniprotkb"
                           and p.capability == "exact_record_fetch")
        self.assertTrue(entry_probe.url.endswith(
            "fields=" + ",".join(UNIPROT_ENTRY_FIELDS)), entry_probe.url)

    def test_the_client_builds_a_probed_url(self) -> None:
        connector = UniProtKBConnector(
            cache=self.cache, access=AccessPolicy(allow_network=True))
        seen: list[str] = []
        connector._http_json = lambda url, timeout=30.0: (
            seen.append(url) or (None, None))
        connector._fetch_remote("P07846")
        self.assertIn(seen[0], probe_urls("uniprotkb"))


# --------------------------------------------------------------------------
# The failure the old client would have crashed on
# --------------------------------------------------------------------------

class UnknownIdentifiersAndOutages(_Tmp):
    """404 is an answer; a timeout is the absence of one."""

    def test_an_unknown_accession_is_a_miss(self) -> None:
        from eagent.connectors.sequence import UniProtKBConnector
        connector = UniProtKBConnector(
            cache=self.cache, access=AccessPolicy(allow_network=True))
        connector._http_json = lambda url, timeout=30.0: (None, None)
        response = connector.fetch("NOTANACCESSION")
        self.assertIs(response.status, ResponseStatus.MISS)

    def test_an_outage_is_an_error_not_a_miss(self) -> None:
        """Reported as "the service returned nothing", a transient failure
        would read as evidence a record does not exist."""
        connector = UniProtKBConnector(
            cache=self.cache, access=AccessPolicy(allow_network=True))

        def down(url, timeout=30.0):
            raise RemoteCallFailedError("uniprotkb", url, "HTTP 503", status=503)

        connector._http_json = down
        response = connector.fetch("P07846")
        self.assertIs(response.status, ResponseStatus.ERROR)
        self.assertIn("503", response.miss_reason)

    def test_an_outage_is_not_cached(self) -> None:
        """The next run must ask again instead of inheriting the outage."""
        connector = UniProtKBConnector(
            cache=self.cache, access=AccessPolicy(allow_network=True))

        def down(url, timeout=30.0):
            raise RemoteCallFailedError("uniprotkb", url, "timed out")

        connector._http_json = down
        connector.fetch("P07846")
        connector._http_json = lambda url, timeout=30.0: (
            {"primaryAccession": "P07846", "sequence": {"value": "MK"},
             "entryType": "UniProtKB reviewed"}, None)
        self.assertIs(connector.fetch("P07846").status, ResponseStatus.FETCHED)


# --------------------------------------------------------------------------
# Three things only a live run found
# --------------------------------------------------------------------------

class WhatOnlyALiveRunFound(_Tmp):
    def test_the_client_and_the_probe_identify_themselves_identically(self) -> None:
        """Rhea's CDN answers the default Python-urllib agent with 403 and ours
        with 200. A probe sending one agent while the client sent the other
        passed while the client failed, so the agent is one shared constant."""
        import urllib.request
        from unittest import mock
        from eagent.datalayer.probe import USER_AGENT, urllib_fetcher

        sent: list[dict] = []

        class Handle:
            status = 200
            def read(self): return b"{}"
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_urlopen(request, timeout=None):
            sent.append(dict(request.header_items()))
            return Handle()

        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            urllib_fetcher("https://example.org/x", 5.0)
            RheaConnector(cache=self.cache,
                          access=AccessPolicy(allow_network=True)
                          )._http_text("https://example.org/x")
        agents = {h.get("User-agent") for h in sent}
        self.assertEqual(agents, {USER_AGENT})
        self.assertNotIn("Python-urllib", " ".join(agents))

    def test_an_inactive_accession_is_an_answer_not_an_entry(self) -> None:
        """UniProt answers a deleted accession with HTTP 200 and an entry that
        has an accession and no sequence."""
        payload = uniprot_entry_payload(json.loads(
            (FIXTURES / "uniprot" / "Q9ZZZ9_inactive.json").read_text()))
        self.assertTrue(payload["inactive"])
        self.assertIn("DELETED", payload["inactive_reason"])
        self.assertIsNone(payload["sequence"])

    def test_an_inactive_entry_says_it_must_not_be_a_seed(self) -> None:
        connector = UniProtKBConnector(
            cache=self.cache, access=AccessPolicy(allow_network=True))
        body = json.loads((FIXTURES / "uniprot" / "Q9ZZZ9_inactive.json").read_text())
        connector._http_json = lambda url, timeout=30.0: (body, None)
        entry = connector.entry("Q9ZZZ9")
        assert entry is not None
        self.assertTrue(entry.inactive)
        self.assertIsNone(entry.sequence)
        self.assertIsNone(entry.sequence_sha256)
        self.assertTrue([n for n in entry.notes if "INACTIVE" in n])
        self.assertTrue([n for n in entry.notes if "seed" in n])

    def test_a_malformed_identifier_is_an_error_that_says_why(self) -> None:
        """A bare HTTP 400 sends somebody to curl to find out what was wrong."""
        import io
        import urllib.error
        import urllib.request
        from unittest import mock

        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url, 400, "Bad Request", {},
                io.BytesIO(b'{"messages":["The accession value has invalid '
                           b'format"]}'))

        connector = UniProtKBConnector(
            cache=self.cache, access=AccessPolicy(allow_network=True))
        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            response = connector.fetch("ZZZZZZ")
        self.assertIs(response.status, ResponseStatus.ERROR)
        self.assertIn("HTTP 400", response.miss_reason)
        self.assertIn("invalid format", response.miss_reason)

    def test_a_404_is_still_a_clean_miss(self) -> None:
        import io
        import urllib.error
        import urllib.request
        from unittest import mock

        def fake_urlopen(request, timeout=None):
            raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {},
                                         io.BytesIO(b""))

        connector = UniProtKBConnector(
            cache=self.cache, access=AccessPolicy(allow_network=True))
        with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
            self.assertIs(connector.fetch("P99999").status, ResponseStatus.MISS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
