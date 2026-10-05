"""Tests for the structure and mechanism connectors.

Each case is a number that would come out looking perfectly reasonable:

* a predicted apo model scored as though it were a crystal structure;
* a residue number obtained by adding a constant, which lands on a neighbour
  of the intended residue and produces a construct nobody notices is wrong;
* ligand atoms matched by their position in a file, so the hydride-transfer
  distance is measured to a different carbon;
* a cofactor transplanted from a homologue reported as one observed here;
* a catalytic template built from residues with no recorded source.
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
from eagent.connectors.structure import (
    AlphaFillConnector,
    AlphaFoldDBConnector,
    AtomOrderMatchRefusedError,
    MCSAConnector,
    NumberingOffsetRefusedError,
    PredictionNotObservationError,
    RCSBPDBConnector,
    SIFTSConnector,
    WwPDBChemicalComponentConnector,
)
from eagent.datalayer.registry import CapabilityState
from eagent.schemas.chem import CofactorState, LigandSource
from eagent.schemas.record import EvidenceStrength
from eagent.schemas.templates import TemplateSourceType
from eagent.science.numbering import AuthorPosition


class StructureTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cache = FileCache(self.root)
        self.registry = datasource_registry()
        self.addCleanup(self._tmp.cleanup)


# ---------------------------------------------------------------------------
# offline-first, endpoints, capabilities
# ---------------------------------------------------------------------------

class TestOfflineAndRefusals(StructureTestCase):

    def test_a_cache_miss_is_structured_not_an_empty_structure(self) -> None:
        response = RCSBPDBConnector(cache=self.cache).fetch("0XYZ")
        self.assertIs(response.status, ResponseStatus.MISS)
        self.assertIsNone(response.payload)
        self.assertTrue(response.needed)
        self.assertIsNone(RCSBPDBConnector(cache=self.cache).entry("0XYZ"))
        self.assertEqual(AlphaFillConnector(cache=self.cache).transplants("P0"),
                         ())

    def test_a_cache_hit_returns_the_curated_entry(self) -> None:
        connector = RCSBPDBConnector(cache=self.cache)
        connector.store_import("fetch", "0XYZ", {"records": [
            {"pdb_id": "0XYZ", "method": "X-RAY DIFFRACTION", "resolution": 1.8,
             "chains": ["A"],
             "ligands": [{"component_id": "NAP", "chain": "A",
                          "author_seq_id": 301, "occupancy": 1.0}]}]},
            database_version="2024-02-01")
        entry = connector.entry("0XYZ")
        self.assertEqual(entry.pdb_id, "0XYZ")
        self.assertAlmostEqual(entry.resolution_angstrom, 1.8)
        self.assertEqual(entry.ligand("nap").component_id, "NAP")
        self.assertIs(entry.ligand("NAP").source,
                      LigandSource.EXPERIMENTAL_OBSERVED)
        self.assertIs(entry.ligand("NAP").cofactor_state, CofactorState.OXIDIZED)

    def test_an_unstated_occupancy_stays_none_and_is_flagged(self) -> None:
        connector = RCSBPDBConnector(cache=self.cache)
        connector.store_import("fetch", "0ABC", {"records": [
            {"pdb_id": "0ABC",
             "ligands": [{"component_id": "NAI", "chain": "A"}]}]})
        entry = connector.entry("0ABC")
        self.assertIsNone(entry.ligands[0].occupancy)
        self.assertIn("full occupancy", " ".join(entry.notes))

    def test_coordinates_are_not_activity(self) -> None:
        connector = RCSBPDBConnector(cache=self.cache)
        connector.store_import("fetch", "0XYZ",
                               {"records": [{"pdb_id": "0XYZ"}]})
        with self.assertRaises(LayerSemanticsError):
            connector.entry("0XYZ").as_activity_evidence()

    def test_every_structure_source_still_records_a_null_endpoint(self) -> None:
        for source_id in ("rcsb_pdb", "alphafold_db", "sifts", "wwpdb_ccd",
                          "mcsa", "alphafill"):
            with self.subTest(source=source_id):
                self.assertIsNone(self.registry.get(source_id).endpoint)

    def test_require_endpoint_is_a_typed_refusal_naming_the_curation_note(self) -> None:
        for connector in (RCSBPDBConnector(cache=self.cache),
                          AlphaFoldDBConnector(cache=self.cache),
                          SIFTSConnector(cache=self.cache),
                          MCSAConnector(cache=self.cache),
                          AlphaFillConnector(cache=self.cache)):
            with self.subTest(source=connector.source_id):
                with self.assertRaises(EndpointNotEstablishedError) as ctx:
                    connector.require_endpoint()
                self.assertEqual(ctx.exception.source_id, connector.source_id)
                self.assertTrue(ctx.exception.curation_notes)

    def test_a_bulk_only_source_refuses_a_per_record_call_even_with_a_url(self) -> None:
        """wwPDB CCD is registered bulk_download only: there is no service."""
        connector = WwPDBChemicalComponentConnector(cache=self.cache)
        modes = connector.source.access_modes
        self.assertFalse(any(m.is_network_endpoint for m in modes))
        with self.assertRaises(EndpointNotEstablishedError):
            connector.require_endpoint()

    def test_sifts_refuses_a_keyword_query(self) -> None:
        connector = SIFTSConnector(cache=self.cache)
        self.assertIs(connector.capability("keyword_query").state,
                      CapabilityState.NOT_SUPPORTED)
        response = connector.search({"text": "dehydrogenase"})
        self.assertIs(response.status, ResponseStatus.REFUSED)
        self.assertIsNone(response.payload)
        self.assertIn("not_supported", response.miss_reason or "")

    def test_an_unknown_fetch_capability_is_marked_unverified(self) -> None:
        connector = SIFTSConnector(cache=self.cache)
        self.assertIs(connector.capability("exact_record_fetch").state,
                      CapabilityState.UNKNOWN)
        connector.store_import("fetch", "P0:0XYZ:A", {"records": [
            {"uniprot_accession": "P0", "pdb_id": "0XYZ", "chain_id": "A",
             "residues": []}]})
        response = connector.fetch("P0:0XYZ:A")
        self.assertIs(response.status, ResponseStatus.HIT)
        self.assertTrue(any(n.startswith(UNVERIFIED_CAPABILITY)
                            for n in response.notes))

    def test_a_supported_capability_is_not_marked_unverified(self) -> None:
        connector = RCSBPDBConnector(cache=self.cache)
        self.assertIs(connector.capability("exact_record_fetch").state,
                      CapabilityState.SUPPORTED)
        connector.store_import("fetch", "0XYZ",
                               {"records": [{"pdb_id": "0XYZ"}]})
        self.assertFalse(any(n.startswith(UNVERIFIED_CAPABILITY)
                             for n in connector.fetch("0XYZ").notes))


# ---------------------------------------------------------------------------
# AlphaFold
# ---------------------------------------------------------------------------

class TestPredictionsStayPredictions(StructureTestCase):

    def _model(self, **row: object):
        connector = AlphaFoldDBConnector(cache=self.cache)
        base = {"accession": "P00000", "model_id": "AF-P00000-F1",
                "plddt": [95.0, 91.0, 42.0, 88.0]}
        base.update(row)
        connector.store_import("fetch", "P00000", {"records": [base]})
        return connector.model("P00000")

    def test_a_model_refuses_to_be_an_observation(self) -> None:
        model = self._model()
        with self.assertRaises(PredictionNotObservationError) as ctx:
            model.as_observed_structure()
        self.assertIn("computational_construct", str(ctx.exception))
        self.assertIs(self.registry.get("alphafold_db").evidence_strength_ceiling,
                      EvidenceStrength.COMPUTATIONAL_CONSTRUCT)

    def test_a_model_carries_no_ligands(self) -> None:
        self.assertEqual(self._model().ligands, ())

    def test_confidence_is_per_residue_not_a_chain_mean(self) -> None:
        model = self._model()
        self.assertAlmostEqual(model.plddt_at(2), 42.0)
        self.assertAlmostEqual(model.pocket_plddt([0, 1]), 93.0)
        self.assertAlmostEqual(model.pocket_plddt([2]), 42.0)

    def test_a_partial_pocket_yields_none_rather_than_a_partial_mean(self) -> None:
        self.assertIsNone(self._model().pocket_plddt([0, 99]))
        self.assertIsNone(self._model().plddt_at(99))

    def test_a_model_with_no_confidence_says_so(self) -> None:
        model = self._model(plddt=[])
        self.assertIn("no way to localise", " ".join(model.notes))


# ---------------------------------------------------------------------------
# SIFTS
# ---------------------------------------------------------------------------

class TestSiftsResidueMapping(StructureTestCase):

    #: Deliberately non-contiguous: a construct truncated at the N terminus,
    #: an insertion code, and a gap in the middle. Every one of those breaks a
    #: constant offset, and all three occur routinely.
    RESIDUES = [
        {"uniprot_position": 10, "author_seq_id": 3, "pdb_residue_name": "MET",
         "uniprot_residue": "M"},
        {"uniprot_position": 11, "author_seq_id": 3, "insertion_code": "A",
         "pdb_residue_name": "LYS"},
        {"uniprot_position": 12, "author_seq_id": 4, "pdb_residue_name": "ALA"},
        {"uniprot_position": 20, "author_seq_id": 30, "pdb_residue_name": "TYR"},
    ]

    def _mapping(self, residues=None):
        connector = SIFTSConnector(cache=self.cache)
        connector.store_import("fetch", "P00000:0XYZ:A", {"records": [
            {"uniprot_accession": "P00000", "pdb_id": "0XYZ", "chain_id": "A",
             "residues": list(self.RESIDUES if residues is None else residues)}]})
        return connector.mapping("P00000", "0XYZ", "A")

    def test_the_mapping_is_per_residue(self) -> None:
        mapping = self._mapping()
        self.assertEqual(len(mapping.pairs), 4)
        self.assertEqual(mapping.author_position_for(10),
                         AuthorPosition("A", 3, ""))
        self.assertEqual(mapping.author_position_for(20),
                         AuthorPosition("A", 30, ""))

    def test_an_insertion_code_is_part_of_the_identity(self) -> None:
        mapping = self._mapping()
        self.assertEqual(mapping.author_position_for(11),
                         AuthorPosition("A", 3, "A"))
        self.assertNotEqual(mapping.author_position_for(10),
                            mapping.author_position_for(11))
        self.assertEqual(mapping.uniprot_position_for(3), 10)
        self.assertEqual(mapping.uniprot_position_for(3, insertion_code="A"), 11)

    def test_a_single_numeric_offset_is_refused(self) -> None:
        mapping = self._mapping()
        self.assertFalse(mapping.is_contiguous)
        with self.assertRaises(NumberingOffsetRefusedError) as ctx:
            mapping.numbering_offset()
        self.assertIn("per residue", str(ctx.exception))

    def test_an_offset_is_refused_even_when_the_map_happens_to_be_simple(self) -> None:
        mapping = self._mapping([
            {"uniprot_position": 1, "author_seq_id": 1},
            {"uniprot_position": 2, "author_seq_id": 2},
        ])
        self.assertTrue(mapping.is_contiguous)
        with self.assertRaises(NumberingOffsetRefusedError):
            mapping.numbering_offset()

    def test_an_unresolved_residue_is_reported_not_interpolated(self) -> None:
        mapping = self._mapping()
        self.assertIsNone(mapping.author_position_for(15))
        with self.assertRaises(LayerSemanticsError) as ctx:
            mapping.require_author_position(15)
        self.assertIn("not resolved in this structure", str(ctx.exception))
        self.assertEqual(mapping.unmapped_uniprot_positions(10, 13), (13,))

    def test_the_map_is_consumable_as_author_positions(self) -> None:
        mapping = self._mapping()
        forward = mapping.uniprot_to_author
        reverse = mapping.author_to_uniprot
        self.assertTrue(all(isinstance(v, AuthorPosition)
                            for v in forward.values()))
        for position, author in forward.items():
            self.assertEqual(reverse[author], position)
        self.assertEqual(str(forward[11]), "A/3A")
        self.assertEqual(forward[11].token, "3A")

    def test_the_mapping_is_keyed_on_all_three_identifiers(self) -> None:
        """A mapping is only valid for the pair it was built from."""
        self._mapping()
        connector = SIFTSConnector(cache=self.cache)
        self.assertIsNone(connector.mapping("P00000", "0XYZ", "B"))
        self.assertIsNone(connector.mapping("P00001", "0XYZ", "A"))
        self.assertIsNotNone(connector.mapping("P00000", "0XYZ", "A"))

    def test_rows_without_both_numbers_are_dropped_not_guessed(self) -> None:
        mapping = self._mapping([
            {"uniprot_position": 1, "author_seq_id": 1},
            {"uniprot_position": 2},
            {"author_seq_id": 3},
        ])
        self.assertEqual(len(mapping.pairs), 1)


# ---------------------------------------------------------------------------
# wwPDB CCD
# ---------------------------------------------------------------------------

class TestChemicalComponentDictionary(StructureTestCase):

    ATOMS = [
        {"atom_id": "C4N", "element": "C"},
        {"atom_id": "C3N", "element": "C"},
        {"atom_id": "N1N", "element": "N"},
        {"atom_id": "O7N", "element": "O"},
    ]
    BONDS = [
        {"atom_id_1": "C4N", "atom_id_2": "C3N", "order": "SING"},
        {"atom_id_1": "C3N", "atom_id_2": "N1N", "order": "DOUB"},
    ]

    def _component(self, component_id: str = "NAP", **row: object):
        connector = WwPDBChemicalComponentConnector(cache=self.cache)
        base = {"component_id": component_id, "name": "NADP nicotinamide",
                "atoms": list(self.ATOMS), "bonds": list(self.BONDS)}
        base.update(row)
        connector.store_import("fetch", component_id, {"records": [base]},
                               database_version="2024-02")
        return connector.component(component_id)

    def test_atoms_are_matched_by_name_whatever_the_file_order(self) -> None:
        """Two depositions list the same ligand's atoms in different orders."""
        component = self._component()
        first_order = [{"name": "C4N"}, {"name": "C3N"},
                       {"name": "N1N"}, {"name": "O7N"}]
        second_order = [{"name": "O7N"}, {"name": "N1N"},
                        {"name": "C3N"}, {"name": "C4N"}]
        a = component.match_atoms(first_order)
        b = component.match_atoms(second_order)

        self.assertNotEqual([x["name"] for x in first_order],
                            [x["name"] for x in second_order])
        self.assertEqual(sorted(a.matched), sorted(b.matched))
        self.assertTrue(a.complete)
        self.assertTrue(b.complete)
        for name in ("C4N", "C3N", "N1N", "O7N"):
            with self.subTest(atom=name):
                self.assertIs(a.matched[name], b.matched[name])
                self.assertEqual(a.require_atom(name).element,
                                 b.require_atom(name).element)
        # the element at a given index differs between the two orderings, which
        # is exactly what an index-based match would have got wrong
        self.assertNotEqual(first_order[0]["name"], second_order[0]["name"])
        self.assertEqual(a.require_atom("C4N").element, "C")
        self.assertEqual(a.require_atom("O7N").element, "O")

    def test_matching_by_order_is_refused_outright(self) -> None:
        component = self._component()
        with self.assertRaises(AtomOrderMatchRefusedError) as ctx:
            component.match_by_order([{"name": "O7N"}, {"name": "C4N"}])
        self.assertIn("match_atoms()", str(ctx.exception))

    def test_a_different_component_id_is_refused(self) -> None:
        """NAP and NAI share atom names and differ in oxidation state."""
        component = self._component("NAP")
        with self.assertRaises(LayerSemanticsError) as ctx:
            component.match_atoms([{"name": "C4N"}], component_id="NAI")
        self.assertIn("oxidation state", str(ctx.exception))
        match = component.match_atoms([{"name": "C4N"}], component_id="nap")
        self.assertEqual(sorted(match.matched), ["C4N"])

    def test_missing_and_unexpected_atoms_are_reported_separately(self) -> None:
        component = self._component()
        match = component.match_atoms([{"name": "C4N"}, {"name": "ZZZ"}])
        self.assertEqual(sorted(match.matched), ["C4N"])
        self.assertEqual(match.missing, ("C3N", "N1N", "O7N"))
        self.assertEqual(match.unexpected, ("ZZZ",))
        self.assertFalse(match.complete)

    def test_an_absent_atom_is_refused_not_substituted(self) -> None:
        match = self._component().match_atoms([{"name": "C4N"}])
        with self.assertRaises(LayerSemanticsError) as ctx:
            match.require_atom("O7N")
        self.assertIn("No substitute atom", str(ctx.exception))

    def test_bonds_are_addressed_by_name_too(self) -> None:
        component = self._component()
        self.assertEqual(sorted(component.bonded_to("C3N")), ["C4N", "N1N"])
        self.assertEqual(component.bonded_to("O7N"), ())

    def test_plain_strings_and_objects_are_accepted_as_file_atoms(self) -> None:
        component = self._component()

        class FakeAtom:
            def __init__(self, name: str) -> None:
                self.name = name

        by_string = component.match_atoms(["O7N", "C4N"])
        by_object = component.match_atoms([FakeAtom("C4N"), FakeAtom("O7N")])
        self.assertEqual(sorted(by_string.matched), sorted(by_object.matched))

    def test_an_unpinned_dictionary_is_flagged(self) -> None:
        connector = WwPDBChemicalComponentConnector(cache=self.cache)
        connector.store_import("fetch", "NAI",
                               {"records": [{"component_id": "NAI",
                                             "atoms": list(self.ATOMS)}]})
        component = connector.component("NAI")
        self.assertIn("no dictionary release", " ".join(component.notes))

    def test_a_repeated_atom_name_is_flagged_as_ambiguous(self) -> None:
        component = self._component("NDP", atoms=[
            {"atom_id": "C4N", "element": "C"},
            {"atom_id": "C4N", "element": "C"}])
        self.assertIn("repeats atom name", " ".join(component.notes))


# ---------------------------------------------------------------------------
# M-CSA
# ---------------------------------------------------------------------------

class TestMCSACatalyticResidues(StructureTestCase):

    def _entry(self, **row: object):
        connector = MCSAConnector(cache=self.cache)
        base = {"mcsa_id": "MCSA-1", "reference_uniprot": "P00000",
                "reference_pdb": "0XYZ", "ec": "1.1.1.1",
                "literature_ids": ["PMID:111"],
                "residues": [
                    {"residue_name": "TYR", "role": "acid/base",
                     "uniprot_position": 155, "chain_id": "A",
                     "author_seq_id": 155, "mechanism_step": "proton transfer",
                     "source_reference": "PMID:111"},
                    {"residue_name": "LYS", "role": "pKa modulator",
                     "uniprot_position": 159, "chain_id": "A",
                     "author_seq_id": 159, "source_reference": "PMID:111"}]}
        base.update(row)
        connector.store_import("fetch", str(base["mcsa_id"]),
                               {"records": [base]})
        return connector.entry(str(base["mcsa_id"]))

    def test_residues_come_back_with_their_roles_and_sources(self) -> None:
        entry = self._entry()
        self.assertEqual(len(entry.residues), 2)
        self.assertEqual(entry.roles(),
                         {"acid/base": ["TYR155"], "pKa modulator": ["LYS159"]})
        self.assertEqual(entry.residues[0].source_reference, "PMID:111")
        self.assertEqual(entry.residues[0].author_position,
                         AuthorPosition("A", 155, ""))

    def test_a_template_entry_carries_its_evidence(self) -> None:
        item = self._entry().residues[0].as_template_entry()
        self.assertEqual(set(item),
                         {"label", "residue_types", "role", "functional_atoms",
                          "evidence"})
        self.assertEqual(item["evidence"], "PMID:111")

    def test_provenance_is_mechanism_literature_with_identifiers(self) -> None:
        provenance = self._entry().to_template_provenance()
        self.assertIs(provenance.source_type,
                      TemplateSourceType.MECHANISM_LITERATURE)
        self.assertTrue(provenance.source_type.admissible)
        self.assertIn("MCSA-1", provenance.identifiers)
        self.assertIn("PMID:111", provenance.identifiers)
        self.assertIn("residue mapping", provenance.notes)

    def test_an_entry_citing_nothing_cannot_become_a_template(self) -> None:
        entry = self._entry(mcsa_id="MCSA-2", reference_pdb=None,
                            literature_ids=[])
        with self.assertRaises(LayerSemanticsError) as ctx:
            entry.to_template_provenance()
        self.assertIn("unsourced", str(ctx.exception))

    def test_unsourced_residues_are_flagged(self) -> None:
        entry = self._entry(mcsa_id="MCSA-3", residues=[
            {"residue_name": "TYR", "role": "acid/base"}])
        self.assertIn("no source reference", " ".join(entry.notes))

    def test_mechanism_is_not_substrate_scope(self) -> None:
        with self.assertRaises(LayerSemanticsError):
            self._entry().as_substrate_scope()

    def test_the_transfer_caveat_always_travels(self) -> None:
        self.assertIn("evidenced for the reference enzyme",
                      " ".join(self._entry().notes))


# ---------------------------------------------------------------------------
# AlphaFill
# ---------------------------------------------------------------------------

class TestAlphaFillTransplants(StructureTestCase):

    def _transplants(self, *rows: dict):
        connector = AlphaFillConnector(cache=self.cache)
        connector.store_import("fetch", "P00000", {"records": list(rows)})
        return connector, connector.transplants("P00000")

    def test_every_transplanted_ligand_is_tagged_as_transplanted(self) -> None:
        _, ligands = self._transplants(
            {"transplant_id": "t1", "component_id": "NAP",
             "donor_pdb_id": "1ABC", "donor_chain": "A",
             "local_rmsd": 0.3, "sequence_identity": 0.62},
            {"transplant_id": "t2", "component_id": "nai",
             "donor_pdb_id": "2DEF"})
        self.assertEqual(len(ligands), 2)
        for ligand in ligands:
            with self.subTest(ligand=ligand.component_id):
                self.assertIs(ligand.source,
                              LigandSource.HOMOLOGY_TRANSPLANTED)
                self.assertFalse(ligand.source.is_experimental)
                self.assertIn("not measured here", ligand.source.claim())

    def test_the_tag_cannot_be_overridden(self) -> None:
        """There is no constructor argument and no setter for the source."""
        _, (ligand,) = self._transplants({"component_id": "NAP"})
        with self.assertRaises(AttributeError):
            ligand.source = LigandSource.EXPERIMENTAL_OBSERVED  # type: ignore[misc]
        self.assertIs(ligand.source, LigandSource.HOMOLOGY_TRANSPLANTED)

    def test_the_cofactor_spec_keeps_the_tag_and_names_the_donor(self) -> None:
        _, (ligand,) = self._transplants(
            {"component_id": "NDP", "donor_pdb_id": "1ABC", "donor_chain": "B"})
        spec = ligand.to_cofactor_spec(name="NADPH")
        self.assertIs(spec.source, LigandSource.HOMOLOGY_TRANSPLANTED)
        self.assertIs(spec.state, CofactorState.REDUCED)
        self.assertEqual(spec.ligand_code, "NDP")
        self.assertIn("1ABC", spec.evidence)
        self.assertIn("not an observation", spec.evidence)

    def test_the_component_id_still_decides_the_oxidation_state(self) -> None:
        _, ligands = self._transplants({"component_id": "NAP"},
                                       {"component_id": "NDP"},
                                       {"component_id": "XYZ"})
        self.assertIs(ligands[0].cofactor_state, CofactorState.OXIDIZED)
        self.assertIs(ligands[1].cofactor_state, CofactorState.REDUCED)
        self.assertIs(ligands[2].cofactor_state, CofactorState.UNKNOWN)

    def test_thresholds_must_be_stated_and_a_missing_metric_fails(self) -> None:
        connector, _ = self._transplants(
            {"component_id": "NAP", "local_rmsd": 0.3,
             "sequence_identity": 0.62},
            {"component_id": "NAI", "local_rmsd": 2.5,
             "sequence_identity": 0.62},
            {"component_id": "NDP"})
        accepted = connector.accepted_transplants(
            "P00000", max_local_rmsd=1.0, min_sequence_identity=0.5)
        self.assertEqual([t.component_id for t in accepted], ["NAP"])
        self.assertEqual(
            len(connector.accepted_transplants("P00000", max_local_rmsd=None,
                                               min_sequence_identity=None)), 3)

    def test_alphafill_is_not_independent_of_its_upstreams(self) -> None:
        lineage = self.registry.lineage("alphafill")
        self.assertIn("alphafold_db", lineage)
        self.assertIn("rcsb_pdb", lineage)
        self.assertIs(self.registry.get("alphafill").evidence_strength_ceiling,
                      EvidenceStrength.COMPUTATIONAL_CONSTRUCT)

    def test_a_row_without_a_component_id_is_dropped(self) -> None:
        _, ligands = self._transplants({"transplant_id": "t1"},
                                       {"component_id": "NAP"})
        self.assertEqual([l.component_id for l in ligands], ["NAP"])


class TestPolicyIsHonoured(StructureTestCase):

    def test_network_is_off_by_default(self) -> None:
        for factory in (RCSBPDBConnector, AlphaFoldDBConnector, SIFTSConnector,
                        WwPDBChemicalComponentConnector, MCSAConnector,
                        AlphaFillConnector):
            with self.subTest(connector=factory.__name__):
                self.assertFalse(factory(cache=self.cache).access.allow_network)

    def test_enabling_the_network_still_produces_a_miss_not_a_call(self) -> None:
        connector = MCSAConnector(cache=self.cache,
                                  access=AccessPolicy(allow_network=True))
        response = connector.fetch("MCSA-404")
        self.assertIs(response.status, ResponseStatus.MISS)
        self.assertIsNone(response.payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
