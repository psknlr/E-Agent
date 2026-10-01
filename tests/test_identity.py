"""Tests for layered identity: the layer that stops records being over-merged.

Each test targets one documented failure mode rather than one function:

* a stereocentre that appeared during "normalisation" (an assumption dressed
  as a reading of the source);
* a hydride-donor question answered for a cofactor whose oxidation state
  nobody recorded;
* two different sequences carrying one accession;
* a merge justified by name similarity.

All four are silent in a real pipeline: each produces a well-formed record
that is simply about the wrong thing.
"""

from __future__ import annotations

import unittest

from eagent.datalayer.identity import (
    AccessionRef,
    ChemicalIdentityLadder,
    CofactorIdentity,
    CofactorSpecies,
    CofactorStateUnknownError,
    EntityMergePolicy,
    IdentityRung,
    MergeGround,
    NameGuessRefusedError,
    NameResolver,
    ProteinIdentity,
    ProteinRelation,
    RedoxState,
    Representation,
    ResolutionStatus,
    ResolverSource,
    RungValue,
    UnresolvedIdentityError,
    normalise_query_name,
    same_protein,
    stereo_descriptors,
)

#: Fixture structures. The identifiers used below are synthetic placeholders,
#: not real database keys: these tests check the resolver's behaviour, and a
#: plausible-looking real identifier in a fixture is exactly the kind of value
#: that later gets copied into production configuration.
KETONE_SMILES = "CC(=O)c1ccccc1"
S_ALCOHOL_SMILES = "C[C@H](O)c1ccccc1"
R_ALCOHOL_SMILES = "C[C@@H](O)c1ccccc1"


class TestChemicalIdentityLadder(unittest.TestCase):
    """The ladder must keep every rung and name every added assumption."""

    def test_rungs_are_retained_separately(self) -> None:
        lad = ChemicalIdentityLadder("substrate-1")
        lad.add(IdentityRung.AS_WRITTEN, "acetophenone",
                representation=Representation.FREE_TEXT,
                source="fixture source record")
        lad.add(IdentityRung.NORMALISED_STRUCTURE, KETONE_SMILES,
                representation=Representation.SMILES, source="fixture registry entry")
        self.assertEqual(lad.get(IdentityRung.AS_WRITTEN).value, "acetophenone")
        self.assertEqual(lad.get(IdentityRung.NORMALISED_STRUCTURE).value,
                         KETONE_SMILES)
        # No rung stands in for another.
        self.assertIsNone(lad.get(IdentityRung.STEREO_DEFINED_STRUCTURE))
        self.assertIsNone(lad.structure_for_modelling())
        report = lad.is_consistent()
        self.assertIn(IdentityRung.MODELLED_STRUCTURE, report.rungs_absent)

    def test_stereocentre_introduced_during_normalisation_is_flagged(self) -> None:
        # The author wrote a name with no configuration; "normalisation"
        # produced a single enantiomer. That is a decision, not a reading.
        lad = ChemicalIdentityLadder("substrate-2")
        lad.add(IdentityRung.AS_WRITTEN, "1-phenylethanol",
                representation=Representation.FREE_TEXT, source="fixture source record")
        lad.add(IdentityRung.NORMALISED_STRUCTURE, S_ALCOHOL_SMILES,
                representation=Representation.SMILES, produced_by="toolkit")

        report = lad.is_consistent()
        self.assertTrue(report.stereocentre_introduced_during_normalisation)
        stereo_adds = report.additions_of_kind("stereochemistry")
        self.assertEqual(len(stereo_adds), 1)
        self.assertTrue(stereo_adds[0].is_assumption)
        self.assertIs(stereo_adds[0].to_rung, IdentityRung.NORMALISED_STRUCTURE)
        self.assertTrue(any("assumption" in n for n in report.stereocentre_notes))
        self.assertTrue(report.needs_curation)
        # The flag does not fire when the source itself stated the configuration.
        lad2 = ChemicalIdentityLadder("substrate-3")
        lad2.add(IdentityRung.AS_WRITTEN, "(S)-1-phenylethanol",
                 representation=Representation.FREE_TEXT, declared_stereo=True)
        lad2.add(IdentityRung.NORMALISED_STRUCTURE, S_ALCOHOL_SMILES,
                 representation=Representation.SMILES, source="fixture registry entry")
        self.assertFalse(
            lad2.is_consistent().stereocentre_introduced_during_normalisation)

    def test_stereocentre_at_its_own_rung_is_an_addition_not_a_normalisation_flag(self) -> None:
        lad = ChemicalIdentityLadder("substrate-4")
        lad.add(IdentityRung.AS_WRITTEN, "1-phenylethanol",
                representation=Representation.FREE_TEXT)
        lad.add(IdentityRung.NORMALISED_STRUCTURE, "CC(O)c1ccccc1",
                representation=Representation.SMILES, source="fixture registry entry")
        lad.add(IdentityRung.STEREO_DEFINED_STRUCTURE, S_ALCOHOL_SMILES,
                representation=Representation.SMILES,
                source="paper table 2, chiral HPLC assignment")
        report = lad.is_consistent()
        self.assertFalse(report.stereocentre_introduced_during_normalisation)
        adds = report.additions_of_kind("stereochemistry")
        self.assertEqual(len(adds), 1)
        self.assertIs(adds[0].to_rung, IdentityRung.STEREO_DEFINED_STRUCTURE)
        self.assertFalse(adds[0].is_assumption)   # it cites a source

    def test_losing_stereochemistry_higher_up_is_a_problem(self) -> None:
        lad = ChemicalIdentityLadder("substrate-5")
        lad.add(IdentityRung.STEREO_DEFINED_STRUCTURE, S_ALCOHOL_SMILES,
                representation=Representation.SMILES, source="paper")
        lad.add(IdentityRung.MODELLED_STRUCTURE, "CC(O)c1ccccc1",
                representation=Representation.SMILES, produced_by="prep tool")
        report = lad.is_consistent()
        self.assertFalse(report.consistent)
        self.assertTrue(any("drops the stereochemistry" in p
                            for p in report.problems))

    def test_modelling_edit_is_reported(self) -> None:
        lad = ChemicalIdentityLadder("substrate-6")
        lad.add(IdentityRung.STEREO_DEFINED_STRUCTURE, S_ALCOHOL_SMILES,
                representation=Representation.SMILES, source="paper")
        lad.add(IdentityRung.MODELLED_STRUCTURE, R_ALCOHOL_SMILES,
                representation=Representation.SMILES, produced_by="docking prep")
        edits = lad.is_consistent().additions_of_kind("modelling_edit")
        self.assertEqual(len(edits), 1)
        self.assertTrue(edits[0].needs_curation)

    def test_charge_assignment_is_recorded_once_and_zero_is_not_noise(self) -> None:
        neutral = ChemicalIdentityLadder("neutral")
        neutral.add(IdentityRung.AS_WRITTEN, "acetophenone",
                    representation=Representation.FREE_TEXT)
        neutral.add(IdentityRung.NORMALISED_STRUCTURE, KETONE_SMILES,
                    representation=Representation.SMILES, source="fixture registry entry")
        self.assertEqual(
            neutral.is_consistent().additions_of_kind("charge_state"), ())

        charged = ChemicalIdentityLadder("charged")
        charged.add(IdentityRung.NORMALISED_STRUCTURE, "CC(=O)[O-]",
                    representation=Representation.SMILES, source="fixture registry entry")
        charged.add(IdentityRung.CHARGE_AND_PROTONATION_STATE, "CC(=O)O",
                    representation=Representation.SMILES,
                    produced_by="protonation model at pH 7")
        adds = charged.is_consistent().additions_of_kind("charge_state")
        self.assertEqual(len(adds), 1)
        self.assertIn("charge changed from -1 to +0", adds[0].detail)

    def test_empty_rung_value_is_refused(self) -> None:
        with self.assertRaises(Exception):
            RungValue(IdentityRung.AS_WRITTEN, "   ")

    def test_stereo_descriptors_say_unknown_for_prose(self) -> None:
        self.assertIsNone(stereo_descriptors("1-phenylethanol",
                                             Representation.FREE_TEXT))
        d = stereo_descriptors(S_ALCOHOL_SMILES, Representation.SMILES)
        self.assertIsNotNone(d)
        self.assertEqual(d.tetrahedral, 1)
        self.assertEqual(d.double_bond, 0)


class TestNameResolver(unittest.TestCase):
    """Names resolve through registered sources, or not at all."""

    def setUp(self) -> None:
        self.resolver = NameResolver([
            ResolverSource(source_id="fixture_local", identifier_type="inchikey",
                           version="fixture-v0",
                           table={"(s)-1-phenylethanol": "FIXTUREKEYAAAA-FIXTUREONLYA-N"}),
        ])

    def test_exact_registered_name_resolves_with_provenance(self) -> None:
        res = self.resolver.resolve("(S)-1-Phenylethanol")
        self.assertTrue(res.resolved)
        hit = res.require()
        self.assertEqual(hit.source_id, "fixture_local")
        self.assertEqual(hit.source_version, "fixture-v0")
        self.assertFalse(res.needs_curation)

    def test_unregistered_name_returns_an_unresolved_marker_with_a_question(self) -> None:
        res = self.resolver.resolve("the ketone from entry 3")
        self.assertFalse(res)
        self.assertIs(res.status, ResolutionStatus.UNRESOLVED_NOT_FOUND)
        self.assertIsNone(res.identifier)
        self.assertTrue(res.question)
        self.assertTrue(res.needs_curation)
        with self.assertRaises(UnresolvedIdentityError):
            res.require()

    def test_stereo_prefix_is_never_stripped_to_force_a_match(self) -> None:
        # The racemate must not resolve to the (S) entry.
        self.assertFalse(self.resolver.resolve("1-phenylethanol").resolved)
        self.assertFalse(self.resolver.resolve("rac-1-phenylethanol").resolved)
        self.assertEqual(normalise_query_name("  (S)-1-Phenylethanol "),
                         "(s)-1-phenylethanol")

    def test_conflicting_sources_do_not_pick_a_winner(self) -> None:
        self.resolver.register(ResolverSource(
            source_id="other_local", identifier_type="inchikey",
            table={"(s)-1-phenylethanol": "FIXTUREKEYBBBB-FIXTUREONLYB-N"}))
        res = self.resolver.resolve("(S)-1-phenylethanol")
        self.assertIs(res.status, ResolutionStatus.UNRESOLVED_CONFLICTING_SOURCES)
        self.assertEqual(len(res.hits), 2)
        self.assertIn("disagree", res.reason)

    def test_guessing_is_a_named_refusal(self) -> None:
        with self.assertRaises(NameGuessRefusedError):
            self.resolver.guess("ADH substrate 3")


class TestCofactorIdentity(unittest.TestCase):
    """A cofactor is a specific molecule in a specific oxidation state."""

    def test_family_label_does_not_become_a_species(self) -> None:
        cof = CofactorIdentity.from_label("NAD-type cofactor")
        self.assertIs(cof.species, CofactorSpecies.UNKNOWN)
        self.assertIs(cof.state, RedoxState.UNKNOWN)
        self.assertTrue(cof.needs_curation)
        self.assertIn("family", cof.question)
        # A bare "NAD" is ambiguous in prose and is treated the same way.
        self.assertIs(CofactorIdentity.from_label("NAD").species,
                      CofactorSpecies.UNKNOWN)

    def test_hydride_donor_question_is_refused_when_state_unknown(self) -> None:
        cof = CofactorIdentity.from_label("NAD(P)H")
        answer = cof.hydride_donor_answer()
        self.assertFalse(answer.answered)
        self.assertIsNone(answer.value)          # not False
        self.assertTrue(answer.question)
        self.assertTrue(answer.needs_curation)
        with self.assertRaises(CofactorStateUnknownError) as cm:
            cof.is_hydride_donor()
        self.assertIn("oxidation state", str(cm.exception))

    def test_known_states_answer_in_both_directions(self) -> None:
        self.assertTrue(CofactorIdentity.from_label("NADPH").is_hydride_donor())
        self.assertFalse(CofactorIdentity.from_label("NADP+").is_hydride_donor())

    def test_component_ids_fix_the_oxidation_state(self) -> None:
        self.assertIs(CofactorIdentity.from_ligand_code("NDP").species,
                      CofactorSpecies.NADPH)
        self.assertIs(CofactorIdentity.from_ligand_code("NAP").species,
                      CofactorSpecies.NADP_PLUS)
        unknown = CofactorIdentity.from_ligand_code("XYZ")
        self.assertIs(unknown.species, CofactorSpecies.UNKNOWN)
        self.assertTrue(unknown.needs_curation)

    def test_requirement_check_separates_mismatch_from_unknown(self) -> None:
        oxidised = CofactorIdentity.from_ligand_code("NAP")       # NADP+
        wrong_state = oxidised.satisfies(CofactorSpecies.NADPH)
        self.assertIs(wrong_state.satisfied, False)
        self.assertIn("oxidation state", wrong_state.reason)

        unknown = CofactorIdentity.from_label("nicotinamide cofactor")
        cannot_tell = unknown.satisfies(CofactorSpecies.NADPH)
        self.assertIsNone(cannot_tell.satisfied)                  # not False
        self.assertTrue(cannot_tell.repair_hint)

        specificity = CofactorIdentity.from_ligand_code("NAI").satisfies(
            CofactorSpecies.NADPH)
        self.assertIs(specificity.satisfied, False)
        self.assertIs(CofactorIdentity.from_ligand_code("NAI").satisfies(
            CofactorSpecies.NADPH, allow_backbone_substitution=True).satisfied,
            True)


class TestProteinIdentity(unittest.TestCase):
    """Identity is the sequence hash. An accession is a pointer."""

    def test_same_accession_different_sequence_is_not_the_same_protein(self) -> None:
        acc = AccessionRef("P12345", database="uniprot", version="2026_02")
        wild = ProteinIdentity(label="wild type", sequence="MKVQAWYTGSDL",
                               accessions=(acc,))
        variant = ProteinIdentity(label="deposited variant",
                                  sequence="MKVQAWFTGSDL", accessions=(acc,))
        verdict = same_protein(wild, variant)
        self.assertIs(verdict.relation, ProteinRelation.RELATED_NOT_IDENTICAL)
        self.assertFalse(verdict)
        self.assertEqual(verdict.shared_accessions, ("P12345",))
        self.assertIn("not an identity", verdict.reason)
        self.assertTrue(verdict.question)
        self.assertTrue(verdict.needs_curation)

    def test_identical_sequences_are_identical_even_without_accessions(self) -> None:
        a = ProteinIdentity(label="engineered round 2",
                            sequence="MKVQAWYTGSDL", is_engineered=True)
        b = ProteinIdentity(label="same protein, another record",
                            sequence="mkvqawytgsdl")
        verdict = same_protein(a, b)
        self.assertTrue(verdict)
        self.assertIs(verdict.relation, ProteinRelation.IDENTICAL_SEQUENCE)
        self.assertEqual(a.accessions, ())      # an engineered enzyme has none

    def test_matching_accession_without_sequences_is_undetermined(self) -> None:
        acc = AccessionRef("P12345", version="2026_02")
        a = ProteinIdentity(label="a", accessions=(acc,))
        b = ProteinIdentity(label="b", accessions=(acc,))
        verdict = same_protein(a, b)
        self.assertIs(verdict.relation, ProteinRelation.UNDETERMINED)
        self.assertFalse(verdict)
        self.assertIn("not an answer", verdict.reason)

    def test_construct_is_recorded_separately_from_the_catalytic_sequence(self) -> None:
        tagged = ProteinIdentity(
            label="his-tagged", sequence="MKVQAWYTGSDL",
            construct_sequence="MHHHHHHMKVQAWYTGSDL",
            construct_description="N-terminal His6")
        plain = ProteinIdentity(label="untagged", sequence="MKVQAWYTGSDL",
                                construct_sequence="MKVQAWYTGSDL")
        self.assertTrue(tagged.construct_differs_from_catalytic_sequence)
        verdict = same_protein(tagged, plain)
        self.assertTrue(verdict)                       # same catalytic sequence
        self.assertIn("constructs differ", verdict.construct_note)

    def test_primary_key_refuses_to_fall_back_to_an_accession(self) -> None:
        only_accession = ProteinIdentity(
            label="annotation only",
            accessions=(AccessionRef("P12345", version="2026_02"),))
        self.assertFalse(only_accession.has_identity)
        with self.assertRaises(Exception):
            only_accession.primary_key()

    def test_unversioned_accessions_are_flagged(self) -> None:
        p = ProteinIdentity(label="p", sequence="MKVA",
                            accessions=(AccessionRef("P12345"),
                                        AccessionRef("Q9ABC1", version="2026_02")))
        self.assertEqual([a.accession for a in p.unversioned_accessions()],
                         ["P12345"])


class TestEntityMergePolicy(unittest.TestCase):
    """Resemblance may rank records. It may never merge them."""

    def setUp(self) -> None:
        self.policy = EntityMergePolicy()

    def test_name_similarity_is_refused_with_a_reason(self) -> None:
        a = {"name": "alcohol dehydrogenase A", "accession": "P11111"}
        b = {"name": "alcohol dehydrogenase A", "accession": "Q22222"}
        decision = self.policy.may_merge(a, b, MergeGround.NAME_SIMILARITY)
        self.assertFalse(decision)
        self.assertIsNone(decision.ground)
        self.assertIn("not grounds for merging", decision.reason)
        self.assertEqual(decision.rejected[0].ground,
                         MergeGround.NAME_SIMILARITY)
        self.assertIn("not identifiers", decision.rejected[0].ground.reason)
        self.assertTrue(decision.question)

    def test_structural_and_substrate_name_resemblance_are_refused_too(self) -> None:
        for ground in (MergeGround.STRUCTURAL_RESEMBLANCE,
                       MergeGround.SUBSTRATE_NAME_SIMILARITY,
                       MergeGround.SEQUENCE_IDENTITY_THRESHOLD,
                       MergeGround.EMBEDDING_PROXIMITY,
                       MergeGround.SAME_EC_AND_ORGANISM):
            decision = self.policy.may_merge({}, {}, ground)
            self.assertFalse(decision, f"{ground.value} must be refused")
            self.assertIn("inadmissible", self.policy.explain(ground))

    def test_sequence_hash_equality_is_the_admissible_ground(self) -> None:
        a = {"sequence_sha256": "sha256:abc", "name": "ADH-A"}
        b = {"sequence_sha256": "sha256:abc", "name": "ketoreductase 7"}
        decision = self.policy.evaluate(a, b)
        self.assertTrue(decision)
        self.assertIs(decision.ground, MergeGround.SEQUENCE_SHA256_EQUAL)

    def test_evaluate_names_the_resemblance_trap_it_refused(self) -> None:
        a = {"name": "alcohol dehydrogenase", "accession": "P12345",
             "substrate_name": "4-chloroacetophenone"}
        b = {"name": "alcohol dehydrogenase", "accession": "P12345",
             "substrate_name": "2-chloroacetophenone"}
        decision = self.policy.evaluate(a, b)
        self.assertFalse(decision)
        grounds = {r.ground for r in decision.rejected}
        self.assertIn(MergeGround.NAME_SIMILARITY, grounds)
        self.assertIn(MergeGround.SUBSTRATE_NAME_SIMILARITY, grounds)
        self.assertIn(MergeGround.ACCESSION_WITHOUT_VERSION, grounds)

    def test_accession_merge_needs_the_database_release(self) -> None:
        a = {"accession": "P12345"}
        b = {"accession": "P12345"}
        undecidable = self.policy.may_merge(
            a, b, MergeGround.ACCESSION_WITH_VERSION_EQUAL)
        self.assertFalse(undecidable)
        self.assertTrue(undecidable.needs_curation)
        self.assertIn("could not be evaluated", undecidable.reason)

        a2 = {"accession": "P12345", "database_version": "2026_02"}
        b2 = {"accession": "P12345", "database_version": "2026_02"}
        self.assertTrue(self.policy.may_merge(
            a2, b2, MergeGround.ACCESSION_WITH_VERSION_EQUAL))

    def test_every_inadmissible_ground_states_its_reason(self) -> None:
        for ground in self.policy.inadmissible_grounds():
            self.assertTrue(ground.reason.strip(),
                            f"{ground.value} must carry a reason")


if __name__ == "__main__":  # pragma: no cover - pytest may not be installed
    unittest.main(verbosity=2)
