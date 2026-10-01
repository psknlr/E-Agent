"""Tests for :mod:`eagent.tools.retrieve_evidence` and the connector contract.

The data layer and the step that uses it are tested together because the
property that matters spans both: a record that is not in the cache must come
back as a *gap*, with the file that would close it named, and must never come
back as an absence of activity. A test of either half alone would miss that.

The other claims under test are the ones a reviewer would attack first:

* the query plan exists before retrieval and says what was not searched;
* an evidence strength is capped at what its source may assert;
* a record the schema rejects is reported, never repaired;
* the matrix keeps confirmed, not-detected, expression-failure and untested
  apart, and separates wild-type success from engineered success;
* a chemotype is never perceived from a SMILES string;
* no sequence leaves the process without a named authorisation.

Runs under pytest, or standalone with
``PYTHONPATH=src python3 tests/test_retrieve_evidence.py``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import yaml

from eagent.connectors import (
    AccessPolicy, CONNECTOR_REGISTRY, CacheIntegrityError, ConnectorLayer,
    FileCache, OfflineConnector, ResponseStatus, SubmissionAuthorization,
    UnauthorizedSubmissionError, connector_keys, connector_spec,
    cross_check_source_registry, looks_like_biological_sequence,
    offline_connectors, resolve_strength, substitution,
)
from eagent.context import ExecutionPolicy, RunContext
from eagent.envelope import Severity, Status
from eagent.provenance import RunManifest
from eagent.schemas import (
    EvidenceStrength, OutcomeClass, ReactionClass, TaskSpec,
)
from eagent.tools.normalize_reaction import SubstrateChemotype
from eagent.tools.retrieve_evidence import (
    MIN_INDEPENDENT_SOURCES_FOR_MATURE, Maturity, RetrieveEvidence,
    assign_chemotype, build_evidence_matrix, build_query_plan, extract_records,
)

ACETOPHENONE = "CC(=O)c1ccccc1"
A_REAL_SEQUENCE = (
    "MKAAVLYEFGKPLSIEEVEVAPPKAHEVRIKIEATGLCHSDLHVIDGELPFPLPAILGHEGAGIVESVG"
)


def _task(**kw: Any) -> TaskSpec:
    task = TaskSpec(task_id="pilot")
    task.reaction.reaction_class = ReactionClass.KETONE_TO_SECONDARY_ALCOHOL
    task.reaction.substrate.name = kw.get("substrate_name", "acetophenone")
    task.reaction.substrate.isomeric_smiles = ACETOPHENONE
    task.reaction.product.isomeric_smiles = "C[C@H](O)c1ccccc1"
    task.reaction.ec_hint = kw.get("ec_hint", "1.1.1.-")
    return task


def _ctx(task: TaskSpec, **policy_kw: Any) -> RunContext:
    return RunContext(
        task=task,
        workdir=Path(tempfile.mkdtemp(prefix="eagent-evidence-")),
        manifest=RunManifest(run_id="test-run", task_id=task.task_id),
        policy=ExecutionPolicy(**policy_kw),
    )


def _cache() -> FileCache:
    return FileCache(Path(tempfile.mkdtemp(prefix="eagent-cache-")))


def _record(**over: Any) -> dict[str, Any]:
    """A minimally valid curated payload record."""
    base: dict[str, Any] = {
        "record_id": "r1",
        "accession": "P00001",
        "family": "SDR",
        "chemotype": "aromatic_ketone",
        "substrate": {"name": "acetophenone", "isomeric_smiles": ACETOPHENONE},
        "outcome": "confirmed_target_product",
        "reaction_direction": "forward_as_target",
        "detection": {"method": "chiral HPLC",
                      "confirms_product_identity": True,
                      "authentic_standard": True},
        "evidence": [{"source_type": "publication", "identifier": "PMID:1"}],
    }
    base.update(over)
    return base


def _seed(conn: OfflineConnector, plan: Any, records: list[dict[str, Any]],
          *, database_version: str | None = "2024.1") -> int:
    """Answer every planned query for one connector with the same payload."""
    n = 0
    for pq in plan.queries:
        if pq.connector == conn.name:
            conn.store_import("search", pq.query, {"records": records},
                              database_version=database_version)
            n += 1
    return n


def _codes(result: Any) -> set[str]:
    return {f.code for f in result.qc_flags}


# ---------------------------------------------------------------------------
# the connector contract
# ---------------------------------------------------------------------------

class ConnectorRegistryTests(unittest.TestCase):

    def test_the_ten_named_resources_are_registered(self) -> None:
        self.assertEqual(
            {"uniprot", "pdb", "alphafold", "interpro", "rhea", "enzymemap",
             "brenda", "sabio_rk", "mcsa", "pubmed"},
            set(connector_keys()))

    def test_every_entry_says_what_it_is_not_good_for(self) -> None:
        for key in connector_keys():
            spec = connector_spec(key)
            self.assertTrue(spec.not_good_for, key)
            self.assertTrue(spec.cannot_substitute_for, key)
            self.assertTrue(spec.datasource_ids, key)

    def test_the_layers_are_the_declared_ones(self) -> None:
        self.assertIs(connector_spec("rhea").data_layer, ConnectorLayer.REACTION)
        self.assertIs(connector_spec("enzymemap").data_layer,
                      ConnectorLayer.REACTION)
        self.assertIs(connector_spec("brenda").data_layer,
                      ConnectorLayer.KINETICS)
        self.assertIs(connector_spec("sabio_rk").data_layer,
                      ConnectorLayer.KINETICS)
        self.assertIs(connector_spec("mcsa").data_layer,
                      ConnectorLayer.MECHANISM)

    def test_layers_do_not_substitute_for_one_another(self) -> None:
        """The exact claim in the brief, encoded so a planner can act on it."""
        for a, b in [("rhea", "brenda"), ("enzymemap", "sabio_rk"),
                     ("brenda", "mcsa"), ("mcsa", "rhea"),
                     ("sabio_rk", "enzymemap")]:
            verdict = substitution(a, b)
            self.assertFalse(verdict, f"{a} must not substitute for {b}")
            self.assertTrue(verdict.reason)

    def test_same_layer_substitution_carries_the_caveats(self) -> None:
        verdict = substitution("brenda", "sabio_rk")
        self.assertTrue(verdict)
        self.assertTrue(verdict.caveats)

    def test_an_ec_mapping_is_not_promoted_to_sequence_level(self) -> None:
        decision = resolve_strength(
            "brenda", EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL)
        self.assertTrue(decision.capped)
        self.assertIs(decision.granted, EvidenceStrength.EC_SPECIES_MAPPED)

    def test_a_named_reviewer_may_promote_and_is_recorded(self) -> None:
        decision = resolve_strength(
            "brenda", EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL,
            human_reviewer="curator:ada")
        self.assertFalse(decision.capped)
        self.assertIn("curator:ada", decision.reason)

    def test_a_prediction_source_cannot_claim_evidence(self) -> None:
        self.assertIs(connector_spec("alphafold").evidence_strength_ceiling,
                      EvidenceStrength.COMPUTATIONAL_CONSTRUCT)

    def test_no_registered_resource_may_assert_sequence_level_evidence(self) -> None:
        """None of these ten identifies a sequence on its own authority."""
        for key, spec in CONNECTOR_REGISTRY.items():
            self.assertFalse(
                spec.may_claim(EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL),
                f"{key} would let an automated ingest claim sequence-level "
                f"evidence without human review")

    def test_it_agrees_with_the_datasource_registry(self) -> None:
        self.assertEqual([], cross_check_source_registry())

    def test_unknown_keys_raise_rather_than_narrowing_a_search(self) -> None:
        with self.assertRaises(Exception):
            connector_spec("not_a_database")


class OfflineConnectorTests(unittest.TestCase):

    def setUp(self) -> None:
        self.cache = _cache()
        self.conn = OfflineConnector("brenda", cache=self.cache)

    def test_a_miss_carries_no_payload_and_names_what_is_needed(self) -> None:
        response = self.conn.search({"substrate": "acetophenone"})
        self.assertIs(response.status, ResponseStatus.MISS)
        self.assertIsNone(response.payload)      # nothing was invented
        self.assertFalse(response.ok)
        self.assertTrue(response.needed)
        self.assertIn(response.cache_path, " ".join(response.needed))

    def test_a_curated_import_is_found_by_the_same_query(self) -> None:
        query = {"substrate": "acetophenone"}
        self.conn.store_import("search", query, {"records": []},
                               database_version="2024.1")
        response = self.conn.search(query)
        self.assertIs(response.status, ResponseStatus.HIT)
        self.assertEqual("2024.1", response.database_version)
        self.assertEqual("2024.1", response.version_for_provenance)

    def test_an_unpinned_run_says_so_instead_of_guessing_a_release(self) -> None:
        self.assertFalse(self.conn.is_pinned)
        self.conn.store_import("fetch", "EC1", {"records": []})
        response = self.conn.fetch("EC1")
        self.assertIsNone(response.database_version)
        self.assertEqual("unpinned", response.version_for_provenance)
        self.assertTrue(response.notes)

    def test_a_pinned_snapshot_has_its_own_cache_namespace(self) -> None:
        """Two releases answer the same question differently; keys must differ."""
        pinned = OfflineConnector("brenda", cache=self.cache, snapshot="2024.1")
        self.conn.store_import("fetch", "EC1", {"records": [_record()]})
        self.assertIs(pinned.fetch("EC1").status, ResponseStatus.MISS)

    def test_a_tampered_cache_entry_is_an_error_not_a_hit(self) -> None:
        path = self.conn.store_import("fetch", "EC1", {"records": []})
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["payload"] = {"records": [_record()]}
        path.write_text(json.dumps(envelope), encoding="utf-8")
        response = self.conn.fetch("EC1")
        self.assertIs(response.status, ResponseStatus.ERROR)
        self.assertIsNone(response.payload)

    def test_the_cache_refuses_a_file_answering_another_query(self) -> None:
        path = self.conn.store_import("fetch", "EC1", {"records": []})
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["query"] = {"op": "fetch", "key": "EC2"}
        path.write_text(json.dumps(envelope), encoding="utf-8")
        with self.assertRaises(CacheIntegrityError):
            self.cache.read("brenda", "unpinned", {"op": "fetch", "key": "EC1"})

    def test_offline_is_the_default_for_a_run_policy(self) -> None:
        policy = AccessPolicy.from_execution_policy(ExecutionPolicy())
        self.assertFalse(policy.allow_network)
        self.assertFalse(
            AccessPolicy.from_execution_policy(object()).allow_network)


class DisclosureGuardTests(unittest.TestCase):

    def test_a_protein_sequence_is_recognised(self) -> None:
        self.assertTrue(looks_like_biological_sequence(A_REAL_SEQUENCE))

    def test_short_strings_and_structures_are_not(self) -> None:
        self.assertFalse(looks_like_biological_sequence("acetophenone"))
        self.assertFalse(looks_like_biological_sequence(ACETOPHENONE))
        self.assertFalse(looks_like_biological_sequence("C" * 40))
        self.assertFalse(looks_like_biological_sequence("P00001"))

    def test_an_unauthorised_sequence_is_refused_before_it_leaves(self) -> None:
        policy = AccessPolicy(allow_network=True)
        with self.assertRaises(UnauthorizedSubmissionError):
            policy.check_outbound("pdb", {"sequence": A_REAL_SEQUENCE})

    def test_a_sequence_hidden_in_a_key_is_also_refused(self) -> None:
        policy = AccessPolicy(allow_network=True)
        with self.assertRaises(UnauthorizedSubmissionError):
            policy.check_outbound("pdb", {A_REAL_SEQUENCE: {"n": 1}})

    def test_a_named_authorisation_permits_it_and_is_recorded(self) -> None:
        from eagent.provenance import sequence_hash

        auth = SubmissionAuthorization(
            authorized_by="operator:ada", scope="pdb",
            justification="published in 2019",
            sequence_sha256=(sequence_hash(A_REAL_SEQUENCE),))
        policy = AccessPolicy(allow_network=True, authorizations=(auth,))
        notes = policy.check_outbound("pdb", {"sequence": A_REAL_SEQUENCE})
        self.assertTrue(any("operator:ada" in n for n in notes))

    def test_an_authorisation_for_another_connector_does_not_carry_over(self) -> None:
        from eagent.provenance import sequence_hash

        auth = SubmissionAuthorization(
            authorized_by="operator:ada", scope="pdb", justification="ok",
            sequence_sha256=(sequence_hash(A_REAL_SEQUENCE),))
        policy = AccessPolicy(allow_network=True, authorizations=(auth,))
        with self.assertRaises(UnauthorizedSubmissionError):
            policy.check_outbound("uniprot", {"sequence": A_REAL_SEQUENCE})

    def test_an_empty_authorisation_is_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            SubmissionAuthorization(authorized_by="operator:ada", scope="*",
                                    justification="because")

    def test_offline_never_reaches_the_guard_because_it_never_dials(self) -> None:
        conn = OfflineConnector("pdb", cache=_cache())
        response = conn.search({"sequence": A_REAL_SEQUENCE})
        self.assertIs(response.status, ResponseStatus.MISS)


# ---------------------------------------------------------------------------
# the query plan
# ---------------------------------------------------------------------------

class QueryPlanTests(unittest.TestCase):

    def test_the_plan_names_the_limits_of_each_query(self) -> None:
        plan = build_query_plan(_task(), families=("SDR",))
        self.assertTrue(plan.queries)
        for pq in plan.queries:
            self.assertTrue(pq.purpose)
            self.assertTrue(pq.must_not_be_used_for)
            self.assertTrue(pq.evidence_strength_ceiling)

    def test_unsearchable_connectors_are_skipped_with_a_reason(self) -> None:
        plan = build_query_plan(_task())
        skipped = {s.connector: s for s in plan.skipped}
        self.assertIn("pdb", skipped)
        self.assertIn("alphafold", skipped)
        self.assertTrue(skipped["pdb"].unblocked_by)

    def test_no_query_is_built_from_a_placeholder(self) -> None:
        """A search for 'None' finds nothing and looks like a real negative."""
        bare = TaskSpec(task_id="t")
        plan = build_query_plan(bare)
        self.assertEqual((), plan.queries)
        self.assertTrue(plan.skipped)
        for pq in plan.queries:
            self.assertNotIn("None", json.dumps(dict(pq.query)))

    def test_missing_families_become_a_recall_caveat(self) -> None:
        plan = build_query_plan(_task())
        joined = " ".join(plan.recall_caveats).lower()
        self.assertIn("family", joined)
        # the caveat a reader must not miss: a miss is not an absence of activity
        self.assertIn("never evidence", joined)

    def test_supplying_families_removes_that_caveat(self) -> None:
        with_families = " ".join(
            build_query_plan(_task(), families=("SDR",)).recall_caveats)
        self.assertNotIn("No candidate family names were supplied",
                         with_families)

    def test_the_plan_hash_is_stable_and_changes_with_the_plan(self) -> None:
        a = build_query_plan(_task(), families=("SDR",))
        b = build_query_plan(_task(), families=("SDR",))
        c = build_query_plan(_task(), families=("SDR", "AKR"))
        self.assertEqual(a.plan_sha256, b.plan_sha256)
        self.assertNotEqual(a.plan_sha256, c.plan_sha256)


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

class ExtractionTests(unittest.TestCase):

    def _response(self, records: list[dict[str, Any]], connector: str = "brenda",
                  database_version: str | None = "2024.1") -> Any:
        conn = OfflineConnector(connector, cache=_cache())
        conn.store_import("fetch", "q", {"records": records},
                          database_version=database_version)
        return conn.fetch("q")

    def test_a_claimed_strength_is_capped_at_the_source_ceiling(self) -> None:
        raw = _record(evidence=[{"source_type": "publication",
                                 "identifier": "PMID:1",
                                 "strength": "sequence_level_experimental"}])
        rows, failures = extract_records("brenda", self._response([raw]))
        self.assertEqual([], failures)
        self.assertIs(rows[0].record.evidence[0].strength,
                      EvidenceStrength.EC_SPECIES_MAPPED)
        self.assertTrue(rows[0].strength_decisions[0]["capped"])

    def test_shared_upstreams_travel_with_the_record(self) -> None:
        """EnzymeMap derives from BRENDA; two hits are one witness."""
        rows, _ = extract_records("enzymemap",
                                  self._response([_record()], "enzymemap"))
        self.assertIn("brenda", rows[0].record.evidence[0].upstream_sources)

    def test_a_negative_without_a_detection_limit_is_rejected(self) -> None:
        raw = _record(outcome="no_target_product_detected",
                      detection={"method": "A340"})
        rows, failures = extract_records("brenda", self._response([raw]))
        self.assertEqual([], rows)
        self.assertEqual(1, len(failures))
        self.assertIn("detection limit", failures[0].reason.lower())

    def test_a_positive_on_an_indirect_signal_is_rejected(self) -> None:
        raw = _record(detection={"method": "NADPH A340 decrease",
                                 "confirms_product_identity": False})
        rows, failures = extract_records("brenda", self._response([raw]))
        self.assertEqual([], rows)
        self.assertEqual(1, len(failures))

    def test_a_record_without_an_outcome_is_rejected_not_defaulted(self) -> None:
        raw = _record()
        raw.pop("outcome")
        rows, failures = extract_records("brenda", self._response([raw]))
        self.assertEqual([], rows)
        self.assertIn("outcome", failures[0].reason)

    def test_a_record_without_a_citation_is_rejected(self) -> None:
        raw = _record(evidence=[])
        _, failures = extract_records("brenda", self._response([raw]))
        self.assertIn("evidence reference", failures[0].reason)

    def test_a_payload_that_is_not_records_yields_nothing(self) -> None:
        conn = OfflineConnector("rhea", cache=_cache())
        conn.store_import("fetch", "q", {"reaction": "RHEA:1"})
        rows, failures = extract_records("rhea", conn.fetch("q"))
        self.assertEqual(([], []), (rows, failures))


# ---------------------------------------------------------------------------
# chemotype binning
# ---------------------------------------------------------------------------

class ChemotypeTests(unittest.TestCase):

    def test_a_declared_chemotype_is_used(self) -> None:
        call = assign_chemotype({"chemotype": "cyclic_ketone"})
        self.assertIs(call.chemotype, SubstrateChemotype.CYCLIC_KETONE)
        self.assertEqual("payload_declared", call.basis)

    def test_an_operator_table_is_used_and_keyed(self) -> None:
        raw = {"substrate": {"inchikey": "KWOLFJPFCHCOCG-UHFFFAOYSA-N"}}
        call = assign_chemotype(raw, {"KWOLFJPFCHCOCG-UHFFFAOYSA-N":
                                      "aromatic_ketone"})
        self.assertIs(call.chemotype, SubstrateChemotype.AROMATIC_KETONE)
        self.assertEqual("operator_table", call.basis)

    def test_nothing_is_perceived_from_a_smiles(self) -> None:
        """A phenyl ring three carbons away does not make a ketone aromatic."""
        raw = {"substrate": {"isomeric_smiles": "CC(=O)CCc1ccccc1"}}
        call = assign_chemotype(raw)
        self.assertIs(call.chemotype, SubstrateChemotype.UNCLASSIFIED)
        self.assertEqual("unassigned", call.basis)

    def test_an_unknown_label_is_not_coerced_into_a_known_one(self) -> None:
        call = assign_chemotype({"chemotype": "beta_keto_ester"})
        self.assertIs(call.chemotype, SubstrateChemotype.UNCLASSIFIED)


# ---------------------------------------------------------------------------
# the evidence matrix
# ---------------------------------------------------------------------------

class EvidenceMatrixTests(unittest.TestCase):

    def _rows(self, records: list[dict[str, Any]]) -> Any:
        conn = OfflineConnector("brenda", cache=_cache())
        conn.store_import("fetch", "q", {"records": records},
                          database_version="2024.1")
        rows, failures = extract_records("brenda", conn.fetch("q"))
        self.assertEqual([], [f.reason for f in failures])
        return rows

    def test_the_four_outcomes_stay_apart(self) -> None:
        rows = self._rows([
            _record(record_id="a"),
            _record(record_id="b", outcome="no_target_product_detected",
                    detection={"method": "GC-MS", "limit_of_detection": 0.5,
                               "limit_unit": "%"}),
            _record(record_id="c",
                    outcome="expression_or_solubility_failure",
                    detection={"method": "SDS-PAGE"}),
            _record(record_id="d", outcome="not_tested",
                    detection={"method": "n/a"}),
        ])
        matrix = build_evidence_matrix(rows)
        cell = matrix.cell("SDR", SubstrateChemotype.AROMATIC_KETONE)
        self.assertEqual((1, 1, 1, 1),
                         (cell.confirmed, cell.not_detected,
                          cell.expression_failure, cell.not_tested))

    def test_no_outcome_class_is_silently_dropped(self) -> None:
        """Every outcome has to land in a counter, or the matrix loses records."""
        outcomes = [
            ("a", OutcomeClass.CONFIRMED_TARGET_PRODUCT,
             {"method": "chiral HPLC", "confirms_product_identity": True}),
            ("b", OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
             {"method": "GC-MS", "limit_of_detection": 0.5, "limit_unit": "%"}),
            ("c", OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE,
             {"method": "SDS-PAGE"}),
            ("d", OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION,
             {"method": "GC-MS", "confirms_product_identity": True}),
            ("e", OutcomeClass.NOT_TESTED, {"method": "n/a"}),
            ("f", OutcomeClass.COMPUTATIONAL_NEGATIVE, {"method": "docking"}),
            ("g", OutcomeClass.COMPUTATIONAL_FAILURE, {"method": "docking"}),
        ]
        self.assertEqual(len(OutcomeClass), len(outcomes))
        rows = self._rows([_record(record_id=rid, outcome=o.value,
                                   detection=det)
                           for rid, o, det in outcomes])
        cell = build_evidence_matrix(rows).cell(
            "SDR", SubstrateChemotype.AROMATIC_KETONE)
        counted = (cell.confirmed + cell.confirmed_reverse_direction
                   + cell.not_detected + cell.expression_failure
                   + cell.other_product + cell.not_tested
                   + cell.computational_only)
        self.assertEqual(cell.n_records, counted)
        self.assertEqual(2, cell.computational_only)

    def test_an_untested_pair_is_not_a_negative(self) -> None:
        matrix = build_evidence_matrix(self._rows([_record()]))
        empty = matrix.cell("SDR", SubstrateChemotype.CYCLIC_KETONE)
        self.assertIs(empty.maturity(), Maturity.UNTESTED)
        self.assertIn("not a negative result", matrix.to_tsv())

    def test_engineered_success_is_not_natural_success(self) -> None:
        rows = self._rows([
            _record(record_id="v", is_variant=True,
                    parent_sequence_sha256="sha256:abc", mutations=["A94T"]),
        ])
        matrix = build_evidence_matrix(rows)
        cell = matrix.cell("SDR", SubstrateChemotype.AROMATIC_KETONE)
        self.assertIs(cell.maturity(), Maturity.ENGINEERED_ONLY)
        self.assertEqual(0, cell.confirmed_wild_type)

    def test_one_report_is_not_a_mature_chemotype(self) -> None:
        matrix = build_evidence_matrix(self._rows([_record()]))
        cell = matrix.cell("SDR", SubstrateChemotype.AROMATIC_KETONE)
        self.assertIs(cell.maturity(), Maturity.NATURAL_SINGLE_REPORT)
        self.assertLess(cell.independent_sources,
                        MIN_INDEPENDENT_SOURCES_FOR_MATURE)

    def test_independent_reports_make_a_chemotype_mature(self) -> None:
        rows = self._rows([
            _record(record_id="a", accession="P1",
                    evidence=[{"source_type": "publication",
                               "identifier": "PMID:1"}]),
            _record(record_id="b", accession="P2",
                    evidence=[{"source_type": "publication",
                               "identifier": "PMID:2"}]),
        ])
        matrix = build_evidence_matrix(rows)
        cell = matrix.cell("SDR", SubstrateChemotype.AROMATIC_KETONE)
        self.assertGreaterEqual(cell.independent_sources,
                                MIN_INDEPENDENT_SOURCES_FOR_MATURE)
        self.assertIs(cell.maturity(), Maturity.MATURE_NATURAL)

    def test_a_reverse_direction_measurement_is_not_confirmation(self) -> None:
        """An oxidation measurement is not evidence of the reduction."""
        rows = self._rows([_record(reaction_direction="reverse_of_target")])
        cell = build_evidence_matrix(rows).cell(
            "SDR", SubstrateChemotype.AROMATIC_KETONE)
        self.assertEqual(0, cell.confirmed)
        self.assertEqual(1, cell.confirmed_reverse_direction)
        self.assertIs(cell.maturity(), Maturity.REVERSE_DIRECTION_ONLY)

    def test_expression_failure_does_not_read_as_inactive(self) -> None:
        rows = self._rows([
            _record(record_id="e", outcome="expression_or_solubility_failure",
                    detection={"method": "SDS-PAGE"})])
        cell = build_evidence_matrix(rows).cell(
            "SDR", SubstrateChemotype.AROMATIC_KETONE)
        self.assertIs(cell.maturity(), Maturity.EXPRESSION_LIMITED)
        self.assertIn("undetermined", cell.maturity().claim())

    def test_the_tsv_is_rectangular_and_carries_its_legend(self) -> None:
        rows = self._rows([_record(), _record(record_id="x", family="AKR",
                                              chemotype="cyclic_ketone")])
        tsv = build_evidence_matrix(rows).to_tsv()
        body = [l for l in tsv.splitlines() if l and not l.startswith("#")]
        widths = {len(l.split("\t")) for l in body}
        self.assertEqual(1, len(widths))
        self.assertIn("maturity vocabulary", tsv.lower())

    def test_records_without_a_family_get_their_own_visible_row(self) -> None:
        rows = self._rows([_record(family="")])
        matrix = build_evidence_matrix(rows)
        self.assertIn("unassigned_family", matrix.families)


# ---------------------------------------------------------------------------
# the interface end to end
# ---------------------------------------------------------------------------

class RetrieveEvidenceTests(unittest.TestCase):

    def test_an_empty_cache_produces_gaps_not_negatives(self) -> None:
        ctx = _ctx(_task())
        conns = offline_connectors(cache=_cache())
        result = RetrieveEvidence().run(ctx, connectors=conns,
                                        families=("SDR",))

        self.assertIs(result.status, Status.PARTIAL)
        self.assertIn("no_evidence_retrieved",
                      {f.code for f in result.qc_flags
                       if f.severity is Severity.BLOCKER})
        self.assertFalse(result.ok)       # downstream must not consume this
        self.assertEqual(0, result.data["n_records"])

        gaps = Path(result.artifact("evidence_gaps").path).read_text(
            encoding="utf-8")
        self.assertIn("not a negative result", gaps)
        self.assertIn("query_gap", gaps)

    def test_the_plan_is_written_even_when_nothing_is_retrieved(self) -> None:
        ctx = _ctx(_task())
        result = RetrieveEvidence().run(ctx, connectors=offline_connectors(
            cache=_cache()))
        plan_path = Path(result.artifact("evidence_query_plan").path)
        document = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        self.assertTrue(document["queries"])
        self.assertTrue(document["skipped_connectors"])
        self.assertTrue(document["recall_caveats"])
        self.assertTrue(document["plan_sha256"])

    def test_nothing_searchable_fails_instead_of_searching_for_nothing(self) -> None:
        bare = TaskSpec(task_id="t")
        result = RetrieveEvidence().run(_ctx(bare))
        self.assertIs(result.status, Status.FAILED)
        self.assertIn("no_searchable_terms", _codes(result))

    def test_a_seeded_cache_produces_records_matrix_and_provenance(self) -> None:
        task = _task()
        cache = _cache()
        conns = offline_connectors(cache=cache)
        plan = build_query_plan(task, families=("SDR",))
        seeded = _seed(conns["brenda"], plan, [
            _record(record_id="a", accession="P1",
                    evidence=[{"source_type": "publication",
                               "identifier": "PMID:1",
                               "strength": "sequence_level_experimental"}]),
            _record(record_id="b", accession="P2", family="AKR",
                    chemotype="functionalised_ketone", is_variant=True,
                    parent_sequence_sha256="sha256:abc",
                    evidence=[{"source_type": "publication",
                               "identifier": "PMID:2"}]),
        ])
        self.assertGreater(seeded, 0)

        ctx = _ctx(task)
        result = RetrieveEvidence().run(ctx, connectors=conns,
                                        families=("SDR",))

        self.assertIs(result.status, Status.PARTIAL)   # other layers still gap
        self.assertEqual(2, result.data["n_records"])
        self.assertEqual("2024.1", result.provenance.databases["brenda"])

        lines = Path(result.artifact("evidence_records").path).read_text(
            encoding="utf-8").strip().splitlines()
        self.assertEqual(2, len(lines))
        first = json.loads(lines[0])
        self.assertIn("record", first)
        self.assertIn("annotation", first)
        self.assertEqual("ec_species_mapped",
                         first["record"]["evidence"][0]["strength"])

        tsv = Path(result.artifact("evidence_matrix").path).read_text(
            encoding="utf-8")
        self.assertIn("engineered_only", tsv)
        self.assertIn("natural_single_report", tsv)
        self.assertIn("evidence_strength_capped", _codes(result))

    def test_every_source_touched_is_recorded_in_provenance(self) -> None:
        task = _task()
        conns = offline_connectors(cache=_cache())
        result = RetrieveEvidence().run(ctx=_ctx(task), connectors=conns,
                                        families=("SDR",))
        planned = {q.connector for q in
                   build_query_plan(task, families=("SDR",)).queries}
        self.assertEqual(planned, set(result.provenance.databases))

    def test_an_unpinned_source_is_flagged_not_hidden(self) -> None:
        task = _task()
        conns = offline_connectors(cache=_cache())
        plan = build_query_plan(task, families=("SDR",))
        _seed(conns["brenda"], plan, [_record()], database_version=None)
        result = RetrieveEvidence().run(_ctx(task), connectors=conns,
                                        families=("SDR",))
        self.assertIn("database_version_unknown", _codes(result))
        self.assertIn("brenda", result.data["unpinned_sources"])

    def test_rejected_records_are_reported_on_one_tsv_line_each(self) -> None:
        task = _task()
        conns = offline_connectors(cache=_cache())
        plan = build_query_plan(task, families=("SDR",))
        _seed(conns["brenda"], plan, [
            _record(record_id="bad", outcome="no_target_product_detected",
                    detection={"method": "A340"})])
        result = RetrieveEvidence().run(_ctx(task), connectors=conns,
                                        families=("SDR",))

        self.assertIn("records_rejected", _codes(result))
        text = Path(result.artifact("evidence_gaps").path).read_text(
            encoding="utf-8")
        rejected = [l for l in text.splitlines()
                    if l.startswith("record_rejected")]
        self.assertEqual(1, len(rejected))
        self.assertEqual(5, len(rejected[0].split("\t")))

    def test_the_run_never_dials_out_with_the_default_policy(self) -> None:
        """No interface may reach the network unless the policy says so."""
        task = _task()
        ctx = _ctx(task)
        self.assertFalse(ctx.policy.allow_network)

        class _Tripwire(OfflineConnector):
            def _search_remote(self, query: Mapping[str, Any]) -> Any:
                raise AssertionError("the step attempted a network call")

        conns = {k: _Tripwire(k, cache=_cache()) for k in connector_keys()}
        result = RetrieveEvidence().run(ctx, connectors=conns,
                                        families=("SDR",))
        self.assertIs(result.status, Status.PARTIAL)

    def test_unclassified_chemotypes_are_surfaced_as_a_curation_task(self) -> None:
        task = _task()
        conns = offline_connectors(cache=_cache())
        plan = build_query_plan(task, families=("SDR",))
        raw = _record()
        raw.pop("chemotype")
        _seed(conns["brenda"], plan, [raw])
        result = RetrieveEvidence().run(_ctx(task), connectors=conns,
                                        families=("SDR",))
        self.assertIn("chemotype_unassigned", _codes(result))
        self.assertTrue(any(u.code == "chemotype_assignment"
                            for u in result.uncertainty))


if __name__ == "__main__":
    unittest.main(verbosity=2)
