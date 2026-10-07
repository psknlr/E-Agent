"""Tests for :mod:`eagent.datalayer`.

These tests are written against the two claims the data layer makes, rather
than against the current contents of the YAML:

* the registry is honest -- nothing claims to have been connectivity-tested,
  every entry says what it must not be used for, every endpoint is traceable to
  a citation, and lineage resolves so that re-published data cannot be counted
  twice;
* records are joined by identifiers only -- every resemblance-based "join" is
  refused by name, and a missing identifier is a different outcome from a
  mismatched one.

Validator behaviour is exercised with synthetic sources as well as with the
shipped files, so a test keeps its teeth even if a future curator removes the
last entry of some kind from the registry.

Runs under pytest, or standalone with ``python3 tests/test_datasource_registry.py``.
"""

from __future__ import annotations

import pathlib
import unittest
from pathlib import Path

import yaml

from eagent.datalayer import (
    AccessMode,
    CAPABILITY_NAMES,
    CapabilityFlags,
    CapabilityState,
    DataLayer,
    DataSource,
    DuplicateSourceError,
    EvidenceStrength,
    JoinFieldMissingError,
    JoinKey,
    LAYER_ORDER,
    LayerCoverage,
    RegistryError,
    RegistryIntegrityError,
    SimilarityJoinRefusedError,
    SourceRegistry,
    UnknownJoinKeyError,
    UnknownSourceError,
    is_similarity_pseudo_key,
    permitted_keys,
    refuse_similarity_join,
    validate_join,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASOURCE_DIR = REPO_ROOT / "configs" / "datasources"

#: Loaded once; every registry test reads this.
REGISTRY = SourceRegistry.from_directory(DATASOURCE_DIR)


def _curated_files() -> list[pathlib.Path]:
    """The hand-curated source files, excluding the machine-written overlay.

    The observation file lives in the same directory and is written by
    ``eagent sources verify``; counting it as a layer file would make "one
    file per layer" fail for a reason that has nothing to do with the layers.
    """
    from eagent.datalayer.registry import OBSERVED_CONNECTIVITY_FILE
    return sorted(p for p in DATASOURCE_DIR.glob("*.yaml")
                  if p.name != OBSERVED_CONNECTIVITY_FILE)


def _minimal_source(**overrides) -> dict:
    """A valid source dict, so a test can break exactly one thing at a time."""
    base = dict(
        id="test_source",
        display_name="Test source",
        layers=[DataLayer.REACTION_AND_CHEMISTRY],
        good_for=["something specific"],
        not_good_for=["something it must not be used to claim"],
        access_modes=[AccessMode.REST_API],
        curation_notes=["confirm the endpoint"],
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# The six YAML files
# ---------------------------------------------------------------------------

class TestYamlFilesLoad(unittest.TestCase):
    """Every shipped file must parse and validate into the typed model."""

    def test_six_files_one_per_layer(self) -> None:
        files = sorted(p.name for p in _curated_files())
        self.assertEqual(len(files), 6, f"expected one file per layer, got {files}")
        declared = set()
        for p in _curated_files():
            doc = yaml.safe_load(p.read_text(encoding="utf-8"))
            declared.add(DataLayer(doc["layer"]))
        self.assertEqual(declared, set(LAYER_ORDER))

    def test_every_entry_validates(self) -> None:
        self.assertGreater(len(REGISTRY), 20)
        for src in REGISTRY:
            self.assertIsInstance(src, DataSource)
            self.assertTrue(src.layers)

    def test_each_file_declares_the_layer_its_sources_serve(self) -> None:
        for p in _curated_files():
            doc = yaml.safe_load(p.read_text(encoding="utf-8"))
            layer = DataLayer(doc["layer"])
            for entry in doc["sources"]:
                self.assertIn(layer, REGISTRY.get(entry["id"]).layers,
                              f"{p.name}: {entry['id']} does not declare {layer}")

    def test_every_layer_has_at_least_one_source(self) -> None:
        for layer, n in REGISTRY.layer_counts().items():
            self.assertGreater(n, 0, f"no source registered for {layer.value}")

    def test_a_source_serving_two_layers_is_defined_once(self) -> None:
        # EnzEngDB is referenced from the mutation file and defined in the
        # enzymology file; a second definition would break lineage accounting.
        enzeng = REGISTRY.get("enzengdb")
        self.assertIn(DataLayer.ENZYMOLOGY_EVIDENCE, enzeng.layers)
        self.assertIn(DataLayer.MUTATION_AND_PERFORMANCE, enzeng.layers)
        self.assertIn(enzeng, REGISTRY.by_layer(DataLayer.MUTATION_AND_PERFORMANCE))


# ---------------------------------------------------------------------------
# Honesty invariants
# ---------------------------------------------------------------------------

class TestHonestyInvariants(unittest.TestCase):
    """The invariants that make this registry safe for code to act on."""

    def test_every_source_states_what_it_is_not_good_for(self) -> None:
        for src in REGISTRY:
            self.assertTrue(src.not_good_for,
                            f"{src.id} has no not_good_for entries")
            for item in src.not_good_for:
                self.assertTrue(item.strip(), f"{src.id} has a blank not_good_for")

    def test_not_good_for_is_required_by_the_model(self) -> None:
        with self.assertRaises(Exception):
            DataSource(**_minimal_source(not_good_for=[]))

    def test_every_source_states_what_it_is_good_for(self) -> None:
        for src in REGISTRY:
            self.assertTrue(src.good_for, f"{src.id} has no good_for entries")

    def test_a_verified_source_carries_the_call_that_earned_it(self) -> None:
        """The invariant is not "nothing is verified": it is "nothing is
        verified without the record of a call"."""
        for src in REGISTRY:
            if not src.connectivity_verified:
                continue
            self.assertTrue(src.connectivity_checks,
                            f"{src.id} claims connectivity with no recorded call")
            self.assertTrue(any(c.ok for c in src.connectivity_checks), src.id)
            self.assertTrue(src.verified_capabilities, src.id)

    def test_the_model_refuses_a_bare_verified_flag(self) -> None:
        with self.assertRaises(Exception):
            DataSource(**_minimal_source(connectivity_verified=True))

    def test_a_check_recorded_against_an_unsupported_capability_is_refused(self) -> None:
        """One half of an entry must not describe a different service."""
        from eagent.datalayer.registry import CapabilityFlags, CapabilityState
        check = {
            "capability": "chemical_structure_query",
            "url": "https://example.org/api/x", "checked_at": "2026-01-01T00:00:00Z",
            "checked_by": "test", "ok": True, "status_code": 200,
            "markers": ["x"],
        }
        with self.assertRaises(Exception):
            DataSource(**_minimal_source(
                endpoint="https://example.org/api",
                citations=["https://example.org/docs (fetched HTTP 200)"],
                capabilities=CapabilityFlags(
                    chemical_structure_query=CapabilityState.NOT_SUPPORTED),
                connectivity_verified=True, connectivity_checks=[check]))

    def test_a_passing_check_must_name_what_it_found(self) -> None:
        """A 200 is not a working API: a 404 page and a login wall return one."""
        from eagent.datalayer.registry import ConnectivityCheck
        with self.assertRaises(Exception):
            ConnectivityCheck(capability="exact_record_fetch",
                              url="https://example.org/x",
                              checked_at="2026-01-01T00:00:00Z",
                              checked_by="test", ok=True, status_code=200,
                              markers=[])

    def test_a_failed_check_must_say_why(self) -> None:
        from eagent.datalayer.registry import ConnectivityCheck
        with self.assertRaises(Exception):
            ConnectivityCheck(capability="exact_record_fetch",
                              url="https://example.org/x",
                              checked_at="2026-01-01T00:00:00Z",
                              checked_by="test", ok=False, failure="")

    def test_every_endpoint_has_a_citation(self) -> None:
        """An endpoint code will dial must be traceable to documentation.

        Vacuous over the shipped files while no entry records an endpoint; the
        teeth are in ``test_the_model_refuses_an_endpoint_without_a_citation``,
        which holds whatever a future curator adds.
        """
        for src in REGISTRY:
            if src.endpoint is not None:
                self.assertTrue(
                    src.citations,
                    f"{src.id} records an endpoint with no citation")

    def test_no_entry_asserts_an_endpoint_nobody_has_called(self) -> None:
        """Entries used to record a base URL recalled from memory.

        A registry consumed by code must not carry a URL nobody has called: the
        failure then surfaces at call time, inside a run that has already spent
        its budget, and a planner will have preferred that route over one that
        honestly reported uncertainty. An endpoint is admissible exactly when a
        recorded call reached it.
        """
        for src in REGISTRY:
            if src.endpoint is None:
                continue
            reached = [c for c in src.connectivity_checks
                       if c.ok and c.url.startswith(src.endpoint)]
            self.assertTrue(
                reached,
                f"{src.id} asserts the endpoint {src.endpoint} with no "
                f"recorded call that reached it; a URL belongs in "
                f"curation_notes as a hint until somebody dials it")

    def test_the_recalled_urls_survive_as_curator_hints(self) -> None:
        """Nulling the endpoint must not throw the starting point away.

        The URL is still the most useful thing a curator can be handed; it is
        kept as prose that says it was never called, where no consumer can read
        it as an established route.
        """
        for sid in ("pubmed", "europe_pmc", "pubchem", "ncbi_protein"):
            src = REGISTRY.get(sid)
            notes = " ".join(src.curation_notes)
            self.assertIsNone(src.endpoint, sid)
            self.assertTrue(src.needs_curation, sid)
            self.assertIn("starting hint", notes, sid)
            self.assertIn("from this environment", notes, sid)

    def test_a_hint_that_has_since_been_called_keeps_its_history(self) -> None:
        """uniprotkb was a hint and is now a verified route.

        The hint stays in the curation notes. It records that the URL was once
        only recalled, which is how a reader can tell the difference between a
        route somebody established and one somebody remembered.
        """
        src = REGISTRY.get("uniprotkb")
        self.assertEqual(src.endpoint, "https://rest.uniprot.org")
        self.assertIn("starting hint", " ".join(src.curation_notes))
        self.assertTrue(src.connectivity_verified)

    def test_no_bare_url_is_asserted_as_a_citation(self) -> None:
        """A documentation URL nobody opened is an assertion, not a citation.

        The five endpoints above were each backed by a bare documentation URL
        recalled the same way. Those are gone; a URL may come back only when it
        is marked as unopened or carries an identifier, so that a reader cannot
        mistake it for a page somebody read.
        """
        for src in REGISTRY:
            for cite in src.citations:
                if cite.startswith(("http://", "https://")):
                    self.assertIn(
                        "fetched HTTP", cite,
                        f"{src.id} cites the bare URL {cite!r}; a citation "
                        f"must be an identifier, or say when it was opened, or "
                        f"say explicitly that it was not read")

    def test_the_model_refuses_an_endpoint_without_a_citation(self) -> None:
        with self.assertRaises(Exception):
            DataSource(**_minimal_source(endpoint="https://example.org/api",
                                         citations=[]))

    def test_an_endpoint_needs_a_network_access_mode(self) -> None:
        # A web-only or author-archive resource must not be dressed up as a service.
        with self.assertRaises(Exception):
            DataSource(**_minimal_source(
                access_modes=[AccessMode.MANUAL_REVIEW_IMPORT],
                endpoint="https://example.org/api",
                citations=["doi:10.0000/placeholder"]))

    def test_human_import_only_sources_record_no_endpoint(self) -> None:
        for src in REGISTRY:
            if src.is_human_import_only:
                self.assertIsNone(src.endpoint,
                                  f"{src.id} is import-only but records an endpoint")

    def test_uncertain_specifics_are_null_not_guessed(self) -> None:
        """A specific value and an open curation flag cannot both be true.

        ``endpoint`` is covered here as well as ``version`` and
        ``approximate_record_count``: the module's own docstring calls a guessed
        base URL the most dangerous value in the file, because code will call
        it, so it is the last field that may sit beside "nobody has checked".
        """
        for src in REGISTRY:
            notes = " ".join(src.curation_notes)
            if src.version is not None or src.approximate_record_count is not None:
                self.assertFalse(
                    src.needs_curation,
                    f"{src.id} asserts a version or a record count while still "
                    f"flagged for curation")
            if src.endpoint is None:
                continue
            # An endpoint may stand while other things are still open -- the
            # licence usually is. What it may not do is stand while a note
            # still asks for the endpoint itself to be established: that would
            # be the entry contradicting itself about its own most dangerous
            # field.
            self.assertNotIn(
                "No endpoint", notes,
                f"{src.id} records {src.endpoint} while a curation note still "
                f"says no endpoint is established")
            self.assertTrue(
                src.connectivity_verified,
                f"{src.id} records an endpoint without a verified route")

    def test_needs_curation_entries_say_what_to_confirm(self) -> None:
        for src in REGISTRY:
            if src.needs_curation:
                self.assertTrue(src.curation_notes,
                                f"{src.id} needs curation but says nothing about what")

    def test_the_model_refuses_an_unexplained_curation_flag(self) -> None:
        with self.assertRaises(Exception):
            DataSource(**_minimal_source(needs_curation=True, curation_notes=[]))

    def test_missing_licence_forces_legal_review(self) -> None:
        for src in REGISTRY:
            if src.license is None:
                self.assertTrue(src.needs_legal_review,
                                f"{src.id} has no licence and no legal review flag")
            else:
                self.assertIsNotNone(
                    src.license_source,
                    f"{src.id} states a licence without saying where it came from")

    def test_brenda_licence_is_recorded_as_a_claim_not_a_verified_fact(self) -> None:
        brenda = REGISTRY.get("brenda")
        self.assertEqual(brenda.license, "CC BY 4.0")
        self.assertIsNotNone(brenda.license_source)
        self.assertTrue(brenda.needs_legal_review)
        self.assertIs(brenda.capabilities.redistribution_allowed,
                      CapabilityState.UNKNOWN)

    def test_priority_stages_partition_the_registry(self) -> None:
        staged = {s.id for n in (1, 2, 3) for s in REGISTRY.stage_plan(n)}
        self.assertEqual(staged, set(REGISTRY.ids()))
        for n in (1, 2, 3):
            self.assertTrue(REGISTRY.stage_plan(n), f"stage {n} is empty")
        with self.assertRaises(ValueError):
            REGISTRY.stage_plan(4)


# ---------------------------------------------------------------------------
# Capability tri-state
# ---------------------------------------------------------------------------

class TestCapabilities(unittest.TestCase):
    """UNKNOWN must stay visibly different from NOT_SUPPORTED."""

    def test_default_is_unknown_for_every_flag(self) -> None:
        flags = CapabilityFlags()
        self.assertEqual(len(CAPABILITY_NAMES), 7)
        for name in CAPABILITY_NAMES:
            self.assertIs(flags.get(name), CapabilityState.UNKNOWN)
        self.assertEqual(set(flags.unknowns()), set(CAPABILITY_NAMES))

    def test_unknown_is_not_treated_as_supported(self) -> None:
        flags = CapabilityFlags()
        self.assertFalse(flags.supported("keyword_query"))
        self.assertIsNot(CapabilityState.UNKNOWN, CapabilityState.NOT_SUPPORTED)

    def test_not_supported_and_unknown_round_trip_separately(self) -> None:
        flags = CapabilityFlags(keyword_query="not_supported",
                                sequence_query="unknown")
        self.assertIs(flags.keyword_query, CapabilityState.NOT_SUPPORTED)
        self.assertIs(flags.sequence_query, CapabilityState.UNKNOWN)
        self.assertEqual(flags.as_dict()["keyword_query"], "not_supported")

    def test_unknown_access_mode_is_available_and_used_honestly(self) -> None:
        self.assertFalse(AccessMode.UNKNOWN.is_programmatic)
        for src in REGISTRY:
            if AccessMode.UNKNOWN in src.access_modes:
                self.assertTrue(src.needs_curation)
                self.assertIsNone(src.endpoint)

    def test_import_only_modes_are_marked_as_needing_a_human(self) -> None:
        self.assertTrue(AccessMode.OFFLINE_IMPORT.requires_human_step)
        self.assertTrue(AccessMode.MANUAL_REVIEW_IMPORT.requires_human_step)
        self.assertFalse(AccessMode.REST_API.requires_human_step)

    def test_sdred_akr_and_retrobiocat_are_registered_as_human_imports(self) -> None:
        """Three resources whose records a person must fetch before any ingest.

        ``retrobiocat_db`` is the one that is easy to get wrong: its code is
        published as an installable package, but installing it yields only the
        example specificity data. Registering it as ``local_package`` would make
        it programmatically reachable in this taxonomy and suppress the warning
        in ``plan.readiness()`` that somebody has to obtain and check the real
        records first.
        """
        for sid in ("sdred", "akr_superfamily", "retrobiocat_db"):
            src = REGISTRY.get(sid)
            self.assertTrue(src.is_human_import_only,
                            f"{sid} must not be registered as a live API")
            self.assertFalse(src.is_programmatically_reachable, sid)
            self.assertIsNone(src.endpoint)

    def test_local_package_is_programmatic_and_not_a_human_step(self) -> None:
        """The taxonomy fact the retrobiocat_db entry turned on."""
        self.assertTrue(AccessMode.LOCAL_PACKAGE.is_programmatic)
        self.assertFalse(AccessMode.LOCAL_PACKAGE.requires_human_step)
        self.assertFalse(
            DataSource(**_minimal_source(
                access_modes=[AccessMode.MANUAL_REVIEW_IMPORT,
                              AccessMode.LOCAL_PACKAGE])).is_human_import_only)


# ---------------------------------------------------------------------------
# Lineage and independence
# ---------------------------------------------------------------------------

class TestLineage(unittest.TestCase):
    """Re-published data must not be counted as independent confirmation."""

    def test_derived_sources_resolve_to_their_upstream_closure(self) -> None:
        self.assertEqual(REGISTRY.upstream_closure("oed"), {"brenda", "sabio_rk"})
        self.assertEqual(REGISTRY.upstream_closure("catpred_db"),
                         {"brenda", "sabio_rk"})
        self.assertEqual(REGISTRY.upstream_closure("brenda"), frozenset())

    def test_closure_is_transitive(self) -> None:
        reg = SourceRegistry([
            DataSource(**_minimal_source(id="root_db", display_name="Root")),
            DataSource(**_minimal_source(id="mid_db", display_name="Mid",
                                         derived_from=["root_db"])),
            DataSource(**_minimal_source(id="leaf_db", display_name="Leaf",
                                         derived_from=["mid_db"])),
        ])
        self.assertEqual(reg.upstream_closure("leaf_db"), {"mid_db", "root_db"})
        self.assertEqual(reg.lineage("leaf_db"), {"leaf_db", "mid_db", "root_db"})

    def test_every_declared_upstream_is_registered(self) -> None:
        known = set(REGISTRY.ids())
        for src in REGISTRY:
            for up in src.derived_from:
                self.assertIn(up, known, f"{src.id} points at unregistered {up}")

    def test_dangling_lineage_is_rejected_at_load(self) -> None:
        with self.assertRaises(RegistryIntegrityError):
            SourceRegistry([
                DataSource(**_minimal_source(id="lonely_db",
                                             display_name="Lonely",
                                             derived_from=["not_registered"]))
            ]).validate()

    def test_cyclic_lineage_is_rejected(self) -> None:
        reg = SourceRegistry([
            DataSource(**_minimal_source(id="a_db", display_name="Alpha",
                                         derived_from=["b_db"])),
            DataSource(**_minimal_source(id="b_db", display_name="Beta",
                                         derived_from=["a_db"])),
        ])
        with self.assertRaises(RegistryIntegrityError):
            reg.upstream_closure("a_db")

    def test_brenda_oed_and_catpred_collapse_into_one_group(self) -> None:
        groups = REGISTRY.independent_source_groups(
            ["brenda", "oed", "catpred_db"])
        self.assertEqual(len(groups), 1,
                         f"expected one group, got {groups}")
        self.assertEqual(set(groups[0]), {"brenda", "oed", "catpred_db"})

    def test_the_collapse_survives_unrelated_company(self) -> None:
        groups = REGISTRY.independent_source_groups(
            ["brenda", "oed", "catpred_db", "uniprotkb", "wwpdb_ccd"])
        containing = [g for g in groups if "brenda" in g]
        self.assertEqual(len(containing), 1)
        self.assertTrue({"brenda", "oed", "catpred_db"}.issubset(set(containing[0])))
        self.assertEqual(len(groups), 3, f"unrelated sources merged: {groups}")

    def test_independence_report_names_the_shared_upstream(self) -> None:
        report = REGISTRY.independence_report(["brenda", "oed", "catpred_db"])
        self.assertEqual(report.n_groups, 1)
        shared = report.shared_upstreams[report.groups[0][0]]
        self.assertIn("brenda", shared)

    def test_shared_upstreams_explain_a_collapse_made_through_a_chain(self) -> None:
        """A group formed transitively must still name what tied it together.

        BRENDA and SABIO-RK share nothing with each other; OED and CatPred-DB
        each re-integrate both, so all four collapse into one group through
        them. The intersection across every member is therefore empty, and a
        report that printed that empty set would leave the collapse unexplained
        exactly where a reader most needs to see the cause.
        """
        ids = ["brenda", "sabio_rk", "oed", "catpred_db"]
        report = REGISTRY.independence_report(ids)
        self.assertEqual(report.groups,
                         (("brenda", "catpred_db", "oed", "sabio_rk"),))
        shared = report.shared_upstreams[report.groups[0][0]]
        self.assertTrue(shared, "the collapse is reported without a cause")
        self.assertEqual(set(shared), {"brenda", "sabio_rk"})

        # The old global-intersection rule would have produced nothing here.
        common = set(REGISTRY.lineage(ids[0]))
        for sid in ids[1:]:
            common &= set(REGISTRY.lineage(sid))
        self.assertEqual(common, set())

        self.assertIn("shared upstream: brenda, sabio_rk",
                      "\n".join(report.report_lines()))

    def test_a_single_source_group_claims_no_shared_upstream(self) -> None:
        report = REGISTRY.independence_report(["brenda", "wwpdb_ccd"])
        self.assertEqual(report.shared_upstreams, {})

    def test_incomplete_lineage_is_flagged_rather_than_assumed_independent(self) -> None:
        skid = REGISTRY.get("skid")
        self.assertFalse(skid.derived_from_complete)
        report = REGISTRY.independence_report(["skid", "wwpdb_ccd"])
        self.assertIn("skid", report.incomplete_lineage)
        self.assertEqual(report.n_groups, 2)
        self.assertEqual(report.n_independent, 1, "skid was counted")
        self.assertEqual(report.counted_groups, (("wwpdb_ccd",),))
        self.assertEqual(report.withheld_groups, (("skid",),))

    def test_n_independent_excludes_every_group_with_an_unknown_lineage(self) -> None:
        """The count must not absorb a source that admits it is untraced.

        SKiD and IntEnzyDB declare ``derived_from: []`` with a note saying their
        upstreams are not fully known, and OED and CatPred-DB say the same about
        the upstreams beyond the two they list. An untraced source cannot be
        shown to be separate from anything, so counting its group would report
        re-publication as corroboration -- which is the one thing this registry
        exists to prevent.
        """
        ids = ["brenda", "sabio_rk", "oed", "catpred_db", "skid", "intenzydb"]
        report = REGISTRY.independence_report(ids)
        self.assertEqual(report.n_groups, 3)
        self.assertEqual(len(REGISTRY.independent_source_groups(ids)), 3)
        self.assertEqual(
            report.incomplete_lineage,
            ("catpred_db", "intenzydb", "oed", "skid"))
        self.assertEqual(report.n_independent, 0,
                         f"counted {report.counted_groups} as independent")
        self.assertEqual(report.counted_groups, ())
        text = "\n".join(report.report_lines())
        self.assertIn("0 of 3 source group(s) countable as independent", text)
        for sid in report.incomplete_lineage:
            self.assertIn(f"! {sid}: lineage is known to be incomplete", text)

    def test_one_untraced_member_withholds_the_whole_group(self) -> None:
        """Synthetic, so the invariant survives a curator completing a lineage."""
        reg = SourceRegistry([
            DataSource(**_minimal_source(id="traced_db", display_name="Traced")),
            DataSource(**_minimal_source(id="vague_db", display_name="Vague",
                                         derived_from_complete=False)),
            DataSource(**_minimal_source(id="child_db", display_name="Child",
                                         derived_from=["traced_db"])),
        ])
        report = reg.independence_report()
        self.assertEqual(report.n_groups, 2)
        self.assertEqual(report.counted_groups, (("child_db", "traced_db"),))
        self.assertEqual(report.n_independent, 1)
        self.assertEqual(report.incomplete_lineage, ("vague_db",))

        completed = SourceRegistry([
            DataSource(**_minimal_source(id="traced_db", display_name="Traced")),
            DataSource(**_minimal_source(id="vague_db", display_name="Vague")),
        ])
        self.assertEqual(completed.independence_report().n_independent, 2)

    def test_the_whole_registry_counts_fewer_groups_than_it_has(self) -> None:
        """The two numbers must stay visibly different on the shipped files."""
        report = REGISTRY.independence_report()
        self.assertEqual(report.n_groups, len(REGISTRY.independent_source_groups()))
        self.assertLess(report.n_independent, report.n_groups)
        self.assertEqual(len(report.incomplete_lineage),
                         sum(1 for s in REGISTRY if not s.derived_from_complete))

    def test_unknown_source_id_raises(self) -> None:
        with self.assertRaises(UnknownSourceError):
            REGISTRY.get("no_such_database")

    def test_duplicate_ids_are_refused(self) -> None:
        with self.assertRaises(DuplicateSourceError):
            SourceRegistry([
                DataSource(**_minimal_source(id="twice_db", display_name="One")),
                DataSource(**_minimal_source(id="twice_db", display_name="Two")),
            ])

    def test_cross_layer_reference_to_an_unknown_id_is_refused(self) -> None:
        doc = {
            "layer": "mutation_and_performance",
            "sources": [_yaml_source(id="only_db",
                                     layers=["mutation_and_performance"])],
            "cross_layer_refs": ["nowhere_db"],
        }
        with self.assertRaises(RegistryIntegrityError):
            SourceRegistry.from_documents([("synthetic.yaml", doc)])

    def test_a_source_defined_under_the_wrong_layer_is_refused(self) -> None:
        doc = {
            "layer": "mutation_and_performance",
            "sources": [_yaml_source(id="misfiled_db",
                                     layers=["reaction_and_chemistry"])],
        }
        with self.assertRaises(RegistryError):
            SourceRegistry.from_documents([("synthetic.yaml", doc)])


def _yaml_source(**overrides) -> dict:
    """A YAML-shaped source dict (plain strings, as a file would hold)."""
    base = dict(
        id="synthetic_db",
        display_name="Synthetic",
        layers=["reaction_and_chemistry"],
        good_for=["a specific job"],
        not_good_for=["a claim it cannot support"],
        access_modes=["offline_import"],
        needs_curation=True,
        curation_notes=["confirm everything"],
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Evidence ceilings
# ---------------------------------------------------------------------------

class TestEvidenceCeilings(unittest.TestCase):
    """A source may not hand out a stronger claim than it can support."""

    def test_predicted_structure_sources_cannot_claim_experimental(self) -> None:
        for sid in ("alphafold_db", "alphafill", "retrorules", "mgnify_proteins"):
            src = REGISTRY.get(sid)
            self.assertIs(src.evidence_strength_ceiling,
                          EvidenceStrength.COMPUTATIONAL_CONSTRUCT, sid)
            self.assertFalse(
                src.may_claim(EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL), sid)

    def test_breadth_enzymology_is_capped_below_sequence_level(self) -> None:
        brenda = REGISTRY.get("brenda")
        self.assertFalse(
            brenda.may_claim(EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL))
        self.assertTrue(brenda.may_claim(EvidenceStrength.EC_SPECIES_MAPPED))

    def test_machine_extraction_is_always_pending_review(self) -> None:
        src = REGISTRY.get("machine_literature_extraction")
        self.assertTrue(src.requires_human_review_per_record)
        self.assertFalse(
            src.may_claim(EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL))

    def test_reaction_layer_never_supplies_sequence_level_evidence(self) -> None:
        for src in REGISTRY.by_layer(DataLayer.REACTION_AND_CHEMISTRY):
            self.assertFalse(
                src.may_claim(EvidenceStrength.HOMOLOG_EXPERIMENTAL),
                f"{src.id} would hand out experimental strength from the "
                f"reaction layer")


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class TestDataLayer(unittest.TestCase):
    """Six layers, each documented with its question and the stage it serves."""

    def test_exactly_six_members(self) -> None:
        self.assertEqual(len(list(DataLayer)), 6)
        self.assertEqual(len(LAYER_ORDER), 6)
        self.assertEqual(set(LAYER_ORDER), set(DataLayer))

    def test_each_member_documents_its_question_and_stage(self) -> None:
        for layer in DataLayer:
            self.assertTrue(layer.question.strip(), layer.value)
            self.assertTrue(layer.agent_stage.strip(), layer.value)
            self.assertTrue(layer.cannot_substitute.strip(), layer.value)
            self.assertTrue((layer.doc or "").strip(), layer.value)
            self.assertIn(layer.value, layer.describe())

    def test_values_are_stable_strings(self) -> None:
        self.assertEqual(DataLayer("enzymology_evidence"),
                         DataLayer.ENZYMOLOGY_EVIDENCE)


# ---------------------------------------------------------------------------
# Joins
# ---------------------------------------------------------------------------

class TestSimilarityJoinsAreRefused(unittest.TestCase):
    """Resemblance may rank candidates; it may never merge records."""

    def test_named_refusal_path_always_raises(self) -> None:
        with self.assertRaises(SimilarityJoinRefusedError):
            refuse_similarity_join("embedding_cosine", "two ADH entries")

    def test_validate_join_refuses_similarity_keys(self) -> None:
        a = {"name": "alcohol dehydrogenase"}
        b = {"name": "alcohol dehydrogenase A"}
        for pseudo in ("name_similarity", "sequence_identity", "embedding_cosine",
                       "fuzzy_name", "tanimoto", "structure_rmsd",
                       "nearest_neighbour", "vector_search", "blast_hit",
                       "homology_transfer"):
            with self.assertRaises(SimilarityJoinRefusedError, msg=pseudo):
                validate_join(a, b, pseudo)

    def test_the_refusal_explains_itself(self) -> None:
        try:
            validate_join({}, {}, "name_similarity")
        except SimilarityJoinRefusedError as exc:
            self.assertIn("resemblance is not identity", str(exc))
        else:  # pragma: no cover - the call above must raise
            self.fail("similarity join was not refused")

    def test_a_similar_name_is_not_a_join_key(self) -> None:
        self.assertTrue(is_similarity_pseudo_key("looks_like_the_same_enzyme"))
        self.assertFalse(is_similarity_pseudo_key("sequence_sha256"))
        self.assertFalse(is_similarity_pseudo_key("inchikey"))

    def test_an_unknown_non_similarity_key_is_rejected_separately(self) -> None:
        with self.assertRaises(UnknownJoinKeyError):
            validate_join({"x": 1}, {"x": 1}, "my_own_key")


class TestIdentifierJoins(unittest.TestCase):
    """The permitted joins, and the three outcomes they must keep apart."""

    def test_matching_identifiers_join(self) -> None:
        res = validate_join({"sequence_sha256": "sha256:abc"},
                            {"sequence_sha256": "SHA256:ABC"},
                            JoinKey.SEQUENCE_SHA256)
        self.assertTrue(res.joined)
        self.assertTrue(bool(res))
        self.assertTrue(res.establishes)
        self.assertTrue(res.does_not_establish)

    def test_different_identifiers_do_not_join(self) -> None:
        res = validate_join({"inchikey": "AAAAAAAAAAAAAA-BBBBBBBBBB-N"},
                            {"inchikey": "CCCCCCCCCCCCCC-DDDDDDDDDD-N"},
                            JoinKey.INCHIKEY)
        self.assertFalse(res.joined)
        self.assertFalse(res.partial)

    def test_a_missing_identifier_raises_rather_than_answering_no(self) -> None:
        with self.assertRaises(JoinFieldMissingError):
            validate_join({"inchikey": "AAAAAAAAAAAAAA-BBBBBBBBBB-N"},
                          {"name": "some ketone"},
                          JoinKey.INCHIKEY)

    def test_accession_without_matching_release_is_only_partial(self) -> None:
        res = validate_join({"accession": "P00000", "database_version": "2024_01"},
                            {"accession": "p00000", "database_version": "2025_02"},
                            JoinKey.ACCESSION_WITH_DB_VERSION)
        self.assertFalse(res.joined)
        self.assertTrue(res.partial)
        self.assertIn("release", res.reason)

    def test_prefixed_identifiers_normalise(self) -> None:
        self.assertTrue(validate_join({"chebi_id": "CHEBI:15378"},
                                      {"chebi_id": "15378"},
                                      JoinKey.CHEBI_ID).joined)
        self.assertTrue(validate_join({"rhea_id": "RHEA:10000"},
                                      {"reaction_id": "10000"},
                                      JoinKey.RHEA_REACTION_ID).joined)
        self.assertTrue(validate_join({"pubchem_cid": "CID:7847"},
                                      {"pubchem_cid": "7847"},
                                      JoinKey.PUBCHEM_CID).joined)

    def test_composite_keys_need_every_component(self) -> None:
        a = {"pdb_id": "1abc", "chain_id": "A", "sifts_residue": 143}
        b = {"pdb_id": "1ABC", "chain_id": "A", "sifts_residue": "143"}
        self.assertTrue(validate_join(a, b, JoinKey.PDB_CHAIN_SIFTS_RESIDUE).joined)
        c = dict(b, chain_id="B")
        res = validate_join(a, c, JoinKey.PDB_CHAIN_SIFTS_RESIDUE)
        self.assertFalse(res.joined)
        self.assertTrue(res.partial)
        with self.assertRaises(JoinFieldMissingError):
            validate_join(a, {"pdb_id": "1ABC", "chain_id": "A"},
                          JoinKey.PDB_CHAIN_SIFTS_RESIDUE)

    def test_doi_and_campaign_identify_one_measurement(self) -> None:
        a = {"doi": "https://doi.org/10.0000/Example", "experiment_activity_id": "T3"}
        b = {"source_doi": "10.0000/example", "experiment_activity_id": "T3"}
        res = validate_join(a, b, JoinKey.DOI_WITH_EXPERIMENT_ACTIVITY_ID)
        self.assertTrue(res.joined)
        self.assertIn("not independent", res.establishes)

    def test_cofactor_component_atoms_join_but_do_not_imply_state(self) -> None:
        res = validate_join({"ligand_code": "NAI", "atom_name": "C4N"},
                            {"ccd_component_id": "nai", "atom_name": "C4N"},
                            JoinKey.CCD_COMPONENT_ATOM)
        self.assertTrue(res.joined)
        self.assertIn("oxidation state", res.does_not_establish)

    def test_joins_work_on_objects_as_well_as_mappings(self) -> None:
        class Rec:
            def __init__(self, cid):
                self.pubchem_cid = cid

        self.assertTrue(validate_join(Rec("CID123"), Rec("123"),
                                      JoinKey.PUBCHEM_CID).joined)

    def test_permitted_keys_reports_no_bridge_rather_than_inventing_one(self) -> None:
        keys = permitted_keys(DataLayer.REACTION_AND_CHEMISTRY,
                              DataLayer.MUTATION_AND_PERFORMANCE)
        self.assertIn(JoinKey.INCHIKEY, keys)
        self.assertNotIn(JoinKey.SEQUENCE_SHA256, keys)
        self.assertEqual(
            permitted_keys(DataLayer.STRUCTURE_AND_MECHANISM,
                           DataLayer.LITERATURE_AND_FEEDBACK),
            tuple(k for k in JoinKey
                  if DataLayer.STRUCTURE_AND_MECHANISM in k.spec.connects
                  and DataLayer.LITERATURE_AND_FEEDBACK in k.spec.connects))

    def test_every_key_documents_what_it_does_not_establish(self) -> None:
        for key in JoinKey:
            self.assertTrue(key.spec.establishes.strip(), key.value)
            self.assertTrue(key.spec.does_not_establish.strip(), key.value)
            self.assertTrue(key.spec.fields, key.value)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

class TestLayerCoverage(unittest.TestCase):
    """Thinness must be reported, not implied away."""

    def test_all_six_layers_appear_even_when_empty(self) -> None:
        cov = LayerCoverage(task_id="pilot")
        self.assertEqual(len(cov.as_dict()["layers"]), 6)
        self.assertEqual(len(cov.empty_layers()), 6)
        text = cov.describe()
        for layer in DataLayer:
            self.assertIn(layer.value, text)

    def test_counts_are_per_layer_and_per_source(self) -> None:
        cov = LayerCoverage(task_id="pilot", thin_threshold=3)
        cov.add(DataLayer.ENZYMOLOGY_EVIDENCE, 2, "brenda", experimental=True)
        cov.add(DataLayer.ENZYMOLOGY_EVIDENCE, 1, "sabio_rk")
        cov.add(DataLayer.SEQUENCE_FAMILY_EVOLUTION, 40, "uniprotkb")
        self.assertEqual(cov.count(DataLayer.ENZYMOLOGY_EVIDENCE), 3)
        self.assertEqual(cov.experimental_count(DataLayer.ENZYMOLOGY_EVIDENCE), 2)
        self.assertEqual(cov.sources(DataLayer.ENZYMOLOGY_EVIDENCE),
                         ["brenda", "sabio_rk"])
        self.assertEqual(cov.total(), 43)

    def test_a_rich_layer_does_not_hide_an_empty_one(self) -> None:
        cov = LayerCoverage(task_id="pilot")
        cov.add(DataLayer.SEQUENCE_FAMILY_EVOLUTION, 500, "uniprotkb")
        self.assertIn(DataLayer.MUTATION_AND_PERFORMANCE, cov.empty_layers())
        self.assertIn(DataLayer.ENZYMOLOGY_EVIDENCE, cov.thin_layers())
        self.assertTrue(cov.is_single_source(DataLayer.SEQUENCE_FAMILY_EVOLUTION))

    def test_gaps_are_recorded_with_a_reason(self) -> None:
        cov = LayerCoverage(task_id="pilot")
        cov.record_gap(DataLayer.MUTATION_AND_PERFORMANCE,
                       "no variant data for this family")
        self.assertIn("no variant data for this family",
                      cov.as_dict()["layers"]["mutation_and_performance"]["gaps"])

    def test_negative_counts_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            LayerCoverage(task_id="pilot").add(DataLayer.ENZYMOLOGY_EVIDENCE, -1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
