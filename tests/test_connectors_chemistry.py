"""Tests for the chemistry-layer connectors and the shared connector foundation.

Each case here is a way the reaction-and-chemistry layer quietly stops meaning
anything:

* a cache miss comes back looking like an empty result, so a gap in the
  evidence reads as a finding;
* a connector dials a URL nobody established, or invents one;
* an operation runs against a capability the source does not have, or against
  one nobody has checked, and the two are indistinguishable afterwards;
* a chemical class is accepted as a substrate, so a whole campaign aims at a
  set rather than a compound;
* a Rhea entry written in the oxidation direction is counted as evidence for
  the reduction;
* a cross-reference edge is treated as permission to merge chemical states.

In every case the broken result looks exactly like a good one once it has been
written down, which is why the refusal has to happen in the connector.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eagent.connectors.base import (
    AccessPolicy,
    CachedResponse,
    FileCache,
    ResponseStatus,
)
from eagent.connectors.chemistry import (
    UNVERIFIED_CAPABILITY,
    ChEBIClassStatus,
    ChEBIConnector,
    ChemicalClassRefusedError,
    ConnectorConfigurationError,
    CrossReferenceNotAMergeError,
    CuratedFileImporter,
    EndpointNotEstablishedError,
    EnzymeMapConnector,
    LayerSemanticsError,
    MetaNetXConnector,
    PubChemConnector,
    ReactionLevelOnlyError,
    RegistryBackedConnector,
    RheaConnector,
    RheaDirection,
    datasource_registry,
)
from eagent.datalayer.identity import IdentityRung
from eagent.datalayer.registry import CapabilityState
from eagent.schemas.reaction import ReactionClass
from eagent.schemas.record import ReactionDirection


class ConnectorTestCase(unittest.TestCase):
    """Base class: an empty cache per test, and the real source registry.

    The registry is the real one on purpose. These connectors are meant to be
    governed by what ``configs/datasources`` actually says, and a test fixture
    registry would let the code pass while the shipped configuration disagreed
    with it.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cache = FileCache(self.root)
        self.registry = datasource_registry()
        self.addCleanup(self._tmp.cleanup)

    def assertOfflineByDefault(self, connector: RegistryBackedConnector) -> None:
        self.assertFalse(connector.access.allow_network)


# ---------------------------------------------------------------------------
# the shared foundation
# ---------------------------------------------------------------------------

class TestOfflineFirstBehaviour(ConnectorTestCase):

    def test_default_policy_is_offline(self) -> None:
        self.assertFalse(AccessPolicy().allow_network)
        self.assertOfflineByDefault(PubChemConnector(cache=self.cache))

    def test_cache_miss_returns_a_structured_miss_not_content(self) -> None:
        """A miss must be a fact about this machine, never an empty record."""
        connector = RheaConnector(cache=self.cache)
        response = connector.fetch("RHEA:00000")
        self.assertIs(response.status, ResponseStatus.MISS)
        self.assertIsNone(response.payload)
        self.assertFalse(response.ok)
        self.assertTrue(response.is_miss)
        self.assertTrue(response.needed, "a miss must say what would fix it")
        self.assertIn("rhea", response.needed[0])
        self.assertIn(str(self.root), response.cache_path)
        # and nothing was written as a side effect of asking
        self.assertFalse(Path(response.cache_path).exists())

    def test_cache_hit_returns_the_curated_payload_verbatim(self) -> None:
        connector = RheaConnector(cache=self.cache)
        payload = {"records": [{"rhea_id": "RHEA:00001", "equation": "A = B"}]}
        path = connector.store_import("fetch", "RHEA:00001", payload,
                                      database_version="2024-01")
        self.assertTrue(Path(path).is_file())

        response = connector.fetch("RHEA:00001")
        self.assertIs(response.status, ResponseStatus.HIT)
        self.assertTrue(response.ok)
        self.assertEqual(response.payload, payload)
        self.assertEqual(response.database_version, "2024-01")
        self.assertEqual(response.version_for_provenance, "2024-01")

    def test_miss_names_the_import_call_that_would_fix_it(self) -> None:
        connector = PubChemConnector(cache=self.cache)
        response = connector.fetch("7410")
        joined = " ".join(response.needed)
        self.assertIn("store_import", joined)
        self.assertIn("allow_network=True", joined)

    def test_unresolved_name_never_invents_a_structure(self) -> None:
        resolution = PubChemConnector(cache=self.cache).resolve_name("acetophenone")
        self.assertTrue(resolution.unresolved)
        self.assertIsNone(resolution.ladder.get(IdentityRung.NORMALISED_STRUCTURE))
        self.assertIsNone(resolution.ladder.get(
            IdentityRung.STEREO_DEFINED_STRUCTURE))


class TestEndpointRefusal(ConnectorTestCase):

    def test_an_endpoint_appears_only_with_a_verified_route(self) -> None:
        """An endpoint is admissible exactly when a recorded call reached it."""
        for source_id in ("pubchem", "chebi", "rhea", "metanetx", "enzymemap"):
            with self.subTest(source=source_id):
                source = self.registry.get(source_id)
                if source.endpoint is None:
                    continue
                self.assertTrue(source.connectivity_verified, source_id)
                self.assertTrue(source.verified_capabilities, source_id)

    def test_the_rest_still_record_no_endpoint(self) -> None:
        for source_id in ("pubchem", "chebi", "metanetx", "enzymemap"):
            with self.subTest(source=source_id):
                self.assertIsNone(self.registry.get(source_id).endpoint)

    def test_require_endpoint_raises_a_typed_refusal(self) -> None:
        connector = PubChemConnector(cache=self.cache)
        with self.assertRaises(EndpointNotEstablishedError) as ctx:
            connector.require_endpoint()
        self.assertEqual(ctx.exception.source_id, "pubchem")
        self.assertTrue(ctx.exception.curation_notes)
        self.assertIn("pubchem", str(ctx.exception))
        self.assertIn("curation note", str(ctx.exception).lower())

    def test_endpoint_status_is_actionable(self) -> None:
        status = ChEBIConnector(cache=self.cache).endpoint_status()
        self.assertFalse(status.established)
        self.assertTrue(status.has_network_mode)
        self.assertTrue(status.curation_notes)
        self.assertIn("endpoint", json.dumps(status.to_dict()))

    def test_remote_hooks_refuse_before_building_any_url(self) -> None:
        """Even with the network allowed, there is no checked request to make.

        Rhea's base URL is established -- a probe ran a TSV keyword query
        against it. This client was written against nothing: its fetch builds
        ``<base>/<key>``, which is not Rhea's API, and calling it would return
        a 404 the resolver reports as a miss. A route silently not working is
        worse than a refusal, so the refusal names the real gap.
        """
        from eagent.connectors.chemistry import RequestShapeNotVerifiedError
        connector = RheaConnector(cache=self.cache,
                                  access=AccessPolicy(allow_network=True))
        self.assertIsNotNone(connector.source.endpoint)
        with self.assertRaises(RequestShapeNotVerifiedError):
            connector._fetch_remote("RHEA:00001")
        with self.assertRaises(RequestShapeNotVerifiedError):
            connector._search_remote({"op": "search", "ec": "1.1.1.1"})

    def test_a_source_with_no_endpoint_still_refuses_on_the_endpoint(self) -> None:
        connector = PubChemConnector(cache=self.cache,
                                     access=AccessPolicy(allow_network=True))
        with self.assertRaises(EndpointNotEstablishedError):
            connector._fetch_remote("2244")

    def test_a_network_enabled_run_still_reports_a_miss_not_a_crash(self) -> None:
        connector = RheaConnector(cache=self.cache,
                                  access=AccessPolicy(allow_network=True))
        response = connector.fetch("RHEA:00002")
        self.assertIs(response.status, ResponseStatus.MISS)
        self.assertIsNone(response.payload)

    def test_no_module_in_the_package_hard_codes_a_url(self) -> None:
        """A URL literal anywhere here is a route nobody verified."""
        package = Path(__file__).resolve().parents[1] / "src" / "eagent" / "connectors"
        for path in sorted(package.glob("*.py")):
            with self.subTest(module=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("http://", text)
                self.assertNotIn("https://", text)


class TestCapabilityGating(ConnectorTestCase):

    def test_not_supported_and_unknown_stay_distinct(self) -> None:
        chebi = ChEBIConnector(cache=self.cache)
        self.assertIs(chebi.capability("sequence_query").state,
                      CapabilityState.NOT_SUPPORTED)
        self.assertIs(chebi.capability("chemical_structure_query").state,
                      CapabilityState.UNKNOWN)
        self.assertFalse(chebi.capability("sequence_query").allowed)
        self.assertTrue(chebi.capability("chemical_structure_query").allowed)
        self.assertTrue(chebi.capability("chemical_structure_query").unverified)

    def test_a_not_supported_operation_is_refused_without_touching_the_cache(self) -> None:
        connector = MetaNetXConnector(cache=self.cache)
        response = connector.guarded("search", {"sequence": "MKV"},
                                     "sequence_query")
        self.assertIs(response.status, ResponseStatus.REFUSED)
        self.assertIsNone(response.payload)
        self.assertIn("not_supported", response.miss_reason or "")
        self.assertTrue(response.needed)
        self.assertFalse(Path(response.cache_path).exists())

    def test_a_refusal_is_not_curable_by_populating_the_cache(self) -> None:
        """REFUSED and MISS are different states and must stay different."""
        connector = MetaNetXConnector(cache=self.cache)
        connector.store_import("search", {"sequence": "MKV"},
                               {"records": [{"to_id": "nonsense"}]})
        response = connector.guarded("search", {"sequence": "MKV"},
                                     "sequence_query")
        self.assertIs(response.status, ResponseStatus.REFUSED)
        self.assertIsNone(response.payload)

    def test_an_unknown_capability_is_marked_unverified_not_pretended(self) -> None:
        connector = EnzymeMapConnector(cache=self.cache)
        self.assertIs(connector.capability("keyword_query").state,
                      CapabilityState.UNKNOWN)
        connector.store_import("search", {"ec": "1.1.1.1"},
                               {"records": [{"enzymemap_id": "EM1"}]})
        response = connector.guarded("search", {"ec": "1.1.1.1"},
                                     "keyword_query")
        self.assertIs(response.status, ResponseStatus.HIT)
        self.assertTrue(any(n.startswith(UNVERIFIED_CAPABILITY)
                            for n in response.notes))

    def test_a_supported_capability_is_not_marked_unverified(self) -> None:
        connector = ChEBIConnector(cache=self.cache)
        self.assertIs(connector.capability("exact_record_fetch").state,
                      CapabilityState.SUPPORTED)
        connector.store_import("fetch", "CHEBI:1",
                               {"records": [{"chebi_id": "CHEBI:1",
                                             "entity_type": "instance",
                                             "smiles": "CC"}]})
        response = connector.fetch("CHEBI:1")
        self.assertFalse(any(n.startswith(UNVERIFIED_CAPABILITY)
                             for n in response.notes))

    def test_an_unregistered_capability_name_is_a_configuration_error(self) -> None:
        with self.assertRaises(ConnectorConfigurationError):
            ChEBIConnector(cache=self.cache).capability("telepathy")


class TestAccessModeGating(ConnectorTestCase):

    def test_a_human_import_only_source_cannot_be_built_as_a_connector(self) -> None:
        """The structural guarantee behind "no fake clients"."""

        class SDREDAsAClient(RegistryBackedConnector):
            source_id = "sdred"

        with self.assertRaises(ConnectorConfigurationError) as ctx:
            SDREDAsAClient(cache=self.cache)
        self.assertIn("CuratedFileImporter", str(ctx.exception))

    def test_an_importer_refuses_a_source_that_has_a_programmatic_route(self) -> None:
        class RheaAsAnImporter(CuratedFileImporter):
            source_id = "rhea"
            required_fields = ("id",)

            def _build_record(self, row, row_number):  # pragma: no cover
                raise AssertionError("must not be reached")

        with self.assertRaises(ConnectorConfigurationError):
            RheaAsAnImporter()

    def test_a_connector_without_a_source_id_is_refused(self) -> None:
        class Nameless(RegistryBackedConnector):
            pass

        with self.assertRaises(ConnectorConfigurationError):
            Nameless(cache=self.cache)


# ---------------------------------------------------------------------------
# PubChem
# ---------------------------------------------------------------------------

class TestPubChemIdentityLadder(ConnectorTestCase):

    def _store(self, **fields: object) -> PubChemConnector:
        connector = PubChemConnector(cache=self.cache)
        row = {"cid": "7410", "iupac_name": "1-phenylethan-1-one"}
        row.update(fields)
        connector.store_import("search", {"name": "acetophenone"},
                               {"records": [row]})
        return connector

    def test_a_name_resolves_onto_rungs_not_into_one_field(self) -> None:
        connector = self._store(connectivity_smiles="CC(=O)c1ccccc1",
                                isomeric_smiles="CC(=O)c1ccccc1",
                                charge=0)
        resolution = connector.resolve_name("acetophenone")
        self.assertTrue(resolution.resolved)
        as_written = resolution.ladder.require(IdentityRung.AS_WRITTEN)
        self.assertEqual(as_written.value, "acetophenone")
        self.assertEqual(
            resolution.ladder.require(IdentityRung.NORMALISED_STRUCTURE).value,
            "CC(=O)c1ccccc1")

    def test_a_connectivity_only_answer_leaves_the_stereo_rung_absent(self) -> None:
        """The whole objective of an asymmetric reduction lives on that rung."""
        connector = self._store(connectivity_smiles="CC(O)c1ccccc1",
                                isomeric_smiles="CC(O)c1ccccc1")
        resolution = connector.resolve_name("acetophenone")
        self.assertFalse(resolution.stereo_defined)
        self.assertIsNone(resolution.ladder.get(
            IdentityRung.STEREO_DEFINED_STRUCTURE))
        with self.assertRaises(LayerSemanticsError):
            resolution.require_stereo_defined()

    def test_a_stereo_descriptor_in_the_answer_populates_the_stereo_rung(self) -> None:
        connector = self._store(connectivity_smiles="CC(O)c1ccccc1",
                                isomeric_smiles="C[C@H](O)c1ccccc1")
        resolution = connector.resolve_name("acetophenone")
        self.assertTrue(resolution.stereo_defined)
        self.assertEqual(resolution.require_stereo_defined().value,
                         "C[C@H](O)c1ccccc1")

    def test_chirality_salt_and_charge_are_always_flagged(self) -> None:
        connector = self._store(connectivity_smiles="CC(=O)c1ccccc1.Cl",
                                isomeric_smiles="CC(=O)c1ccccc1.Cl")
        caveats = " ".join(connector.resolve_name("acetophenone").caveats)
        self.assertIn("stereo descriptor", caveats)
        self.assertIn("salt", caveats)
        self.assertIn("charge", caveats)

    def test_an_ambiguous_name_is_unresolved_rather_than_first_wins(self) -> None:
        connector = PubChemConnector(cache=self.cache)
        connector.store_import("search", {"name": "xylene"}, {"records": [
            {"cid": "1", "connectivity_smiles": "Cc1ccccc1C"},
            {"cid": "2", "connectivity_smiles": "Cc1cccc(C)c1"},
        ]})
        resolution = connector.resolve_name("xylene")
        self.assertTrue(resolution.unresolved)
        self.assertIn("2 records", " ".join(resolution.caveats))

    def test_a_record_with_no_structure_resolves_to_nothing(self) -> None:
        connector = self._store()
        resolution = connector.resolve_name("acetophenone")
        self.assertTrue(resolution.unresolved)
        self.assertIsNone(resolution.ladder.get(IdentityRung.NORMALISED_STRUCTURE))


# ---------------------------------------------------------------------------
# ChEBI
# ---------------------------------------------------------------------------

class TestChEBIClassVersusInstance(ConnectorTestCase):

    def _connector(self, **row: object) -> ChEBIConnector:
        connector = ChEBIConnector(cache=self.cache)
        base = {"chebi_id": "CHEBI:17087", "name": "ketone"}
        base.update(row)
        connector.store_import("fetch", str(base["chebi_id"]),
                               {"records": [base]})
        return connector

    def test_a_class_entry_is_refused_as_a_substrate_structure(self) -> None:
        connector = self._connector(entity_type="class", smiles=None)
        entry = connector.entry("CHEBI:17087")
        self.assertIs(entry.class_status, ChEBIClassStatus.CLASS)
        self.assertTrue(entry.is_class)
        with self.assertRaises(ChemicalClassRefusedError) as ctx:
            entry.as_substrate_structure()
        self.assertIn("class", str(ctx.exception))
        with self.assertRaises(ChemicalClassRefusedError):
            connector.require_substrate_entry("CHEBI:17087")

    def test_a_class_entry_that_carries_a_structure_is_still_refused(self) -> None:
        """A class with a representative structure is still a set."""
        connector = self._connector(entity_type="class", smiles="CC(=O)C")
        with self.assertRaises(ChemicalClassRefusedError):
            connector.entry("CHEBI:17087").as_substrate_structure()

    def test_an_undetermined_entry_is_refused_too(self) -> None:
        """"The record does not say" is not the same claim as "it is a compound"."""
        connector = self._connector(smiles="CC(=O)c1ccccc1")
        entry = connector.entry("CHEBI:17087")
        self.assertIs(entry.class_status, ChEBIClassStatus.UNDETERMINED)
        self.assertFalse(entry.class_status.usable_as_substrate)
        with self.assertRaises(ChemicalClassRefusedError):
            entry.as_substrate_structure()
        self.assertIn("entity_type", " ".join(entry.notes))

    def test_an_instance_entry_yields_its_structure(self) -> None:
        connector = self._connector(chebi_id="CHEBI:1", name="acetophenone",
                                    entity_type="instance",
                                    smiles="CC(=O)c1ccccc1")
        entry = connector.entry("CHEBI:1")
        self.assertEqual(entry.as_substrate_structure(), "CC(=O)c1ccccc1")
        self.assertIs(entry.stereo_defined, False)

    def test_an_instance_entry_without_a_structure_is_refused(self) -> None:
        connector = self._connector(chebi_id="CHEBI:2", entity_type="instance")
        with self.assertRaises(ChemicalClassRefusedError):
            connector.entry("CHEBI:2").as_substrate_structure()

    def test_the_boolean_form_is_read_too(self) -> None:
        connector = self._connector(chebi_id="CHEBI:3", is_class=True)
        self.assertIs(connector.entry("CHEBI:3").class_status,
                      ChEBIClassStatus.CLASS)

    def test_a_missing_entry_is_none_not_a_placeholder(self) -> None:
        self.assertIsNone(ChEBIConnector(cache=self.cache).entry("CHEBI:99999"))


# ---------------------------------------------------------------------------
# Rhea
# ---------------------------------------------------------------------------

class TestRheaDirection(ConnectorTestCase):

    TARGET = ReactionClass.KETONE_TO_SECONDARY_ALCOHOL

    def _reaction(self, **row: object):
        connector = RheaConnector(cache=self.cache)
        base = {"rhea_id": "RHEA:10000", "equation": "an alcohol + NAD+ = a ketone"}
        base.update(row)
        connector.store_import("fetch", str(base["rhea_id"]),
                               {"records": [base]})
        return connector.reaction(str(base["rhea_id"]))

    def test_an_oxidation_entry_is_not_reduction_evidence(self) -> None:
        reaction = self._reaction(direction="left_to_right",
                                  reaction_class="alcohol_oxidation")
        self.assertIs(reaction.rhea_direction, RheaDirection.LEFT_TO_RIGHT)
        self.assertIs(reaction.direction_for(self.TARGET),
                      ReactionDirection.REVERSE_OF_TARGET)
        self.assertFalse(reaction.supports_target_direction(self.TARGET))

        verdict = reaction.direction_verdict(self.TARGET)
        self.assertFalse(verdict.supports)
        self.assertTrue(verdict.is_reverse)
        self.assertTrue(verdict.non_supporting)
        self.assertIn("reverse", verdict.describe().lower())

    def test_the_matching_class_supports_the_target(self) -> None:
        reaction = self._reaction(rhea_id="RHEA:10002", direction="left_to_right",
                                  reaction_class="ketone_to_secondary_alcohol")
        self.assertIs(reaction.direction_for(self.TARGET),
                      ReactionDirection.FORWARD_AS_TARGET)
        self.assertTrue(reaction.supports_target_direction(self.TARGET))

    def test_a_bidirectional_entry_is_unspecified_not_both_shown(self) -> None:
        """A direction group is a curatorial convention, not an observation."""
        reaction = self._reaction(rhea_id="RHEA:10004", direction="bidirectional",
                                  reaction_class="ketone_to_secondary_alcohol")
        self.assertIs(reaction.direction_for(self.TARGET),
                      ReactionDirection.UNSPECIFIED)
        self.assertNotEqual(reaction.direction_for(self.TARGET),
                            ReactionDirection.REVERSIBLE_BOTH_SHOWN)
        self.assertFalse(reaction.supports_target_direction(self.TARGET))
        self.assertIn("curatorial convention", " ".join(reaction.notes))

    def test_an_unstated_direction_is_not_a_forward_one(self) -> None:
        reaction = self._reaction(rhea_id="RHEA:10006",
                                  reaction_class="ketone_to_secondary_alcohol")
        self.assertIs(reaction.rhea_direction, RheaDirection.UNSTATED)
        self.assertIs(reaction.direction_for(self.TARGET),
                      ReactionDirection.UNSPECIFIED)
        self.assertFalse(reaction.supports_target_direction(self.TARGET))

    def test_an_unstated_reaction_class_is_not_guessed_from_participants(self) -> None:
        reaction = self._reaction(rhea_id="RHEA:10008", direction="left_to_right",
                                  participants_left=["CHEBI:1"],
                                  participants_right=["CHEBI:2"])
        self.assertIsNone(reaction.reaction_class)
        self.assertIs(reaction.direction_for(self.TARGET),
                      ReactionDirection.UNSPECIFIED)
        self.assertIn("states no reaction class", " ".join(reaction.notes))

    def test_a_rhea_entry_is_never_sequence_evidence(self) -> None:
        reaction = self._reaction(rhea_id="RHEA:10010",
                                  reaction_class="ketone_to_secondary_alcohol")
        with self.assertRaises(ReactionLevelOnlyError):
            reaction.as_sequence_evidence()

    def test_both_directions_of_an_ec_come_back_separately(self) -> None:
        connector = RheaConnector(cache=self.cache)
        connector.store_import("search", {"ec": "1.1.1.1"}, {"records": [
            {"rhea_id": "RHEA:1", "direction": "left_to_right",
             "reaction_class": "alcohol_oxidation"},
            {"rhea_id": "RHEA:2", "direction": "right_to_left",
             "reaction_class": "ketone_to_secondary_alcohol"},
        ]})
        reactions = connector.reactions_for_ec("1.1.1.1")
        self.assertEqual(len(reactions), 2)
        supporting = [r.rhea_id for r in reactions
                      if r.supports_target_direction(self.TARGET)]
        self.assertEqual(supporting, ["RHEA:2"])


# ---------------------------------------------------------------------------
# MetaNetX
# ---------------------------------------------------------------------------

class TestMetaNetXIsAMappingNotAMerge(ConnectorTestCase):

    def _mapping(self):
        connector = MetaNetXConnector(cache=self.cache)
        connector.store_import("search", {"identifier": "CHEBI:1"}, {"records": [
            {"mnx_id": "MNXM1", "from_id": "CHEBI:1", "from_namespace": "chebi",
             "to_id": "C00001", "to_namespace": "kegg"},
            {"mnx_id": "MNXM1", "from_id": "CHEBI:1", "from_namespace": "chebi",
             "to_id": "CHEBI:2", "to_namespace": "chebi"},
        ]})
        return connector.cross_references("CHEBI:1")

    def test_edges_come_back_with_their_caveat(self) -> None:
        mapping = self._mapping()
        self.assertEqual(len(mapping.edges), 2)
        self.assertEqual(mapping.targets_in("kegg"), ("C00001",))
        for edge in mapping.edges:
            self.assertIn("stereochemistry", edge.caveat)

    def test_merging_identities_is_refused(self) -> None:
        with self.assertRaises(CrossReferenceNotAMergeError) as ctx:
            self._mapping().merge_identities()
        self.assertIn("not a licence to merge", str(ctx.exception))

    def test_the_incomplete_lineage_travels_with_the_mapping(self) -> None:
        mapping = self._mapping()
        self.assertFalse(self.registry.get("metanetx").derived_from_complete)
        self.assertIn("incomplete", " ".join(mapping.notes))
        self.assertIn("metanetx", mapping.upstream_sources)
        self.assertIn("rhea", mapping.upstream_sources)


# ---------------------------------------------------------------------------
# EnzymeMap
# ---------------------------------------------------------------------------

class TestEnzymeMapIsReactionLevel(ConnectorTestCase):

    def _reactions(self, **row: object):
        connector = EnzymeMapConnector(cache=self.cache)
        base = {"enzymemap_id": "EM1", "ec": "1.1.1.1",
                "atom_mapped_reaction_smiles": "[CH3:1][C:2](=[O:3])>>[CH3:1][CH:2][OH:3]"}
        base.update(row)
        connector.store_import("search", {"ec": "1.1.1.1"}, {"records": [base]})
        return connector.reactions_for_ec("1.1.1.1")

    def test_a_hit_is_not_per_sequence_validation(self) -> None:
        reaction = self._reactions()[0]
        with self.assertRaises(ReactionLevelOnlyError) as ctx:
            reaction.as_sequence_evidence()
        self.assertIn("reaction-level", str(ctx.exception))

    def test_the_upstream_corpus_travels_so_it_cannot_be_double_counted(self) -> None:
        reaction = self._reactions()[0]
        self.assertIn("brenda", reaction.upstream_sources)
        self.assertIn("enzymemap", reaction.upstream_sources)

    def test_a_mapping_with_no_confidence_is_refused(self) -> None:
        reaction = self._reactions()[0]
        with self.assertRaises(ReactionLevelOnlyError):
            reaction.require_reviewed_mapping(0.9)

    def test_a_low_confidence_mapping_is_refused(self) -> None:
        reaction = self._reactions(mapping_confidence=0.4)[0]
        with self.assertRaises(ReactionLevelOnlyError):
            reaction.require_reviewed_mapping(0.9)

    def test_a_confident_mapping_is_returned(self) -> None:
        reaction = self._reactions(mapping_confidence=0.95)[0]
        self.assertIn(">>", reaction.require_reviewed_mapping(0.9))


# ---------------------------------------------------------------------------
# the registry contract the connectors depend on
# ---------------------------------------------------------------------------

class TestRegistryContract(ConnectorTestCase):

    def test_the_connectors_read_the_registry_not_a_copy(self) -> None:
        connector = ChEBIConnector(cache=self.cache)
        self.assertIs(connector.source, self.registry.get("chebi"))
        self.assertEqual(connector.evidence_strength_ceiling,
                         self.registry.get("chebi").evidence_strength_ceiling)

    def test_a_verified_source_carries_the_call_that_earned_it(self) -> None:
        for source in self.registry:
            with self.subTest(source=source.id):
                if not source.connectivity_verified:
                    continue
                self.assertTrue(source.connectivity_checks, source.id)
                self.assertTrue(any(c.ok for c in source.connectivity_checks))

    #: Connectors whose client has been checked against a recorded probe.
    #: Everything else is generic -- ``<base>/<key>`` for a fetch, query
    #: parameters for a search -- and was written against no service, so a
    #: verified base URL licenses nothing for it. The list grows one connector
    #: at a time, by whoever does the checking.
    CHECKED_CLIENTS = {"uniprotkb": "exact_record_fetch"}

    def test_only_a_checked_client_declares_a_request_shape(self) -> None:
        from eagent.connectors.chemistry import RegistryBackedConnector

        def walk(cls):
            for sub in cls.__subclasses__():
                yield sub
                yield from walk(sub)

        for cls in walk(RegistryBackedConnector):
            sid = getattr(cls, "source_id", "")
            if not sid:
                continue
            with self.subTest(connector=cls.__name__):
                self.assertEqual(cls.verified_route_capability,
                                 self.CHECKED_CLIENTS.get(sid))

    def test_a_checked_client_names_a_capability_its_source_has_verified(self) -> None:
        """Declaring a shape that nothing probed would be the same assertion
        the probe exists to replace."""
        for sid, capability in self.CHECKED_CLIENTS.items():
            with self.subTest(source=sid):
                source = self.registry.get(sid)
                self.assertIn(capability, source.verified_capabilities)

    def test_an_unpinned_connector_says_so(self) -> None:
        connector = ChEBIConnector(cache=self.cache)
        self.assertFalse(connector.is_pinned)
        self.assertEqual(connector.version, "unpinned")

    def test_a_response_round_trips_to_json_for_the_manifest(self) -> None:
        response = ChEBIConnector(cache=self.cache).fetch("CHEBI:1")
        self.assertIsInstance(response, CachedResponse)
        encoded = json.dumps(response.to_dict())
        self.assertIn("payload_present", encoded)
        self.assertIn('"payload_present": false', encoded)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
