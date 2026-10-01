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
        files = sorted(p.name for p in DATASOURCE_DIR.glob("*.yaml"))
        self.assertEqual(len(files), 6, f"expected one file per layer, got {files}")
        declared = set()
        for p in sorted(DATASOURCE_DIR.glob("*.yaml")):
            doc = yaml.safe_load(p.read_text(encoding="utf-8"))
            declared.add(DataLayer(doc["layer"]))
        self.assertEqual(declared, set(LAYER_ORDER))

    def test_every_entry_validates(self) -> None:
        self.assertGreater(len(REGISTRY), 20)
        for src in REGISTRY:
            self.assertIsInstance(src, DataSource)
            self.assertTrue(src.layers)

    def test_each_file_declares_the_layer_its_sources_serve(self) -> None:
        for p in sorted(DATASOURCE_DIR.glob("*.yaml")):
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

    def test_nothing_claims_connectivity_was_verified(self) -> None:
        for src in REGISTRY:
            self.assertFalse(src.connectivity_verified,
                             f"{src.id} claims connectivity_verified")

    def test_the_model_refuses_connectivity_verified(self) -> None:
        with self.assertRaises(Exception):
            DataSource(**_minimal_source(connectivity_verified=True))

    def test_every_endpoint_has_a_citation(self) -> None:
        checked = 0
        for src in REGISTRY:
            if src.endpoint is not None:
                checked += 1
                self.assertTrue(
                    src.citations,
                    f"{src.id} records an endpoint with no citation")
        self.assertGreater(checked, 0,
                           "no endpoint-bearing entry; the check would be vacuous")

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
        for src in REGISTRY:
            if src.version is not None or src.approximate_record_count is not None:
                self.assertFalse(
                    src.needs_curation,
                    f"{src.id} asserts a version or record count while still "
                    f"flagged for curation")

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

    def test_sdred_and_akr_are_registered_as_human_imports(self) -> None:
        for sid in ("sdred", "akr_superfamily"):
            src = REGISTRY.get(sid)
            self.assertTrue(src.is_human_import_only,
                            f"{sid} must not be registered as a live API")
            self.assertIsNone(src.endpoint)


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
        self.assertEqual(report.n_independent, 1)
        shared = report.shared_upstreams[report.groups[0][0]]
        self.assertIn("brenda", shared)

    def test_incomplete_lineage_is_flagged_rather_than_assumed_independent(self) -> None:
        skid = REGISTRY.get("skid")
        self.assertFalse(skid.derived_from_complete)
        report = REGISTRY.independence_report(["skid", "wwpdb_ccd"])
        self.assertIn("skid", report.incomplete_lineage)

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
