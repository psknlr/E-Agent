"""Regression tests for the follow-up audit's P2 findings.

Five findings, each reproduced before being fixed. The sixth of the set --
diversity selection not inheriting coverage -- is in
``test_ranking_scales.py`` with the ranking work it belongs to.

* a freeze policy verified against indices from a different sequence;
* an engineered variant's success counted as wild-type corroboration;
* a campaign row converted into a wild-type record with its PMID dropped;
* a reaction template chosen by which filename sorted first;
* a peptide substrate read from one object by the gate and another by the
  structure checks.
"""

from __future__ import annotations

import types
import unittest

from pydantic import ValidationError

from eagent.schemas import (
    BiopolymerSubstrateSpec, ReactionClass, ReactionSpec, ReactiveResidues,
    ResidueRef, SubstrateKind, SubstrateSpec,
)
from eagent.schemas.candidate import CatalyticMapping, Candidate, SequenceRecord
from eagent.schemas.templates import (
    EngineeringTemplate, TemplateProvenance, TemplateSourceType,
)
from eagent.science.numbering import letter_of_residue_token
from eagent.tools.normalize_reaction import _reaction_template_from_context
from eagent.tools.propose_mutations import frozen_indices

SEQ = "MKAIVTYGASRGIGYEAA"      # the catalytic Tyr sits at index 6
TAGGED = "HHHHHH" + SEQ         # every index shifts by six


# --------------------------------------------------------------------------
# FUP-07: a freeze verified against the wrong sequence
# --------------------------------------------------------------------------

class FreezeIsCheckedAgainstTheSequence(unittest.TestCase):
    TEMPLATE = EngineeringTemplate(
        template_id="eng:test", family_name="SDR",
        frozen_roles=["catalytic_Tyr"], mutable_zones=[],
        provenance=TemplateProvenance(
            source_type=TemplateSourceType.CURATED_DATABASE,
            identifiers=["SOP"]))

    def candidate(self, sequence: str, *, index: int = 6,
                  residue: str = "Y7") -> Candidate:
        return Candidate(
            candidate_id="c1",
            sequence_record=SequenceRecord(candidate_id="c1",
                                           sequence=sequence, is_fragment=False),
            catalytic_mapping=CatalyticMapping(
                catalytic_template_id="cat:sdr",
                role_to_residue={"catalytic_Tyr": residue},
                role_to_index={"catalytic_Tyr": index}))

    def test_a_current_annotation_still_verifies(self) -> None:
        indices, unresolved = frozen_indices(self.candidate(SEQ), self.TEMPLATE)
        self.assertEqual(sorted(indices), [6])
        self.assertEqual(unresolved, [])

    def test_the_premise_a_tag_moves_the_residue(self) -> None:
        self.assertEqual(SEQ[6], "Y")
        self.assertEqual(TAGGED[6], "M")

    def test_a_stale_annotation_does_not_verify(self) -> None:
        """It used to freeze the initiator methionine and call that a freeze."""
        indices, unresolved = frozen_indices(self.candidate(TAGGED),
                                             self.TEMPLATE)
        self.assertEqual(indices, set())
        self.assertEqual(len(unresolved), 1)
        self.assertIn("different sequence", unresolved[0])

    def test_an_index_past_the_end_is_unresolved(self) -> None:
        _, unresolved = frozen_indices(self.candidate(SEQ, index=5000),
                                       self.TEMPLATE)
        self.assertIn("outside a sequence", unresolved[0])

    def test_an_unknown_residue_letter_asserts_nothing(self) -> None:
        indices, unresolved = frozen_indices(
            self.candidate(SEQ, residue="X7"), self.TEMPLATE)
        self.assertEqual(sorted(indices), [6])
        self.assertEqual(unresolved, [])

    def test_a_role_with_no_recorded_letter_is_still_frozen(self) -> None:
        """The check tightens what can be verified and invents nothing."""
        cand = self.candidate(SEQ)
        cand.catalytic_mapping.role_to_residue = {}
        indices, unresolved = frozen_indices(cand, self.TEMPLATE)
        self.assertEqual(sorted(indices), [6])
        self.assertEqual(unresolved, [])

    def test_the_token_helper_reads_both_spellings(self) -> None:
        self.assertEqual(letter_of_residue_token("Y155"), "Y")
        self.assertEqual(letter_of_residue_token("TYR155"), "Y")
        self.assertIsNone(letter_of_residue_token("HOH1"))


# --------------------------------------------------------------------------
# FUP-08: a variant's success counted as wild-type corroboration
# --------------------------------------------------------------------------

class MaturityCountsWildTypeSources(unittest.TestCase):
    def cell(self, **kw):
        from eagent.tools.retrieve_evidence import MatrixCell
        base = dict(confirmed=2, confirmed_wild_type=1, confirmed_variant=1,
                    independent_sources=2, independent_wild_type_sources=1,
                    record_ids=["a", "b"])
        base.update(kw)
        return MatrixCell(**base)

    def test_one_wild_type_report_is_not_mature(self) -> None:
        from eagent.tools.retrieve_evidence import Maturity
        self.assertIs(self.cell().maturity(), Maturity.NATURAL_SINGLE_REPORT)

    def test_two_independent_wild_type_reports_are(self) -> None:
        from eagent.tools.retrieve_evidence import Maturity
        self.assertIs(self.cell(independent_wild_type_sources=2).maturity(),
                      Maturity.MATURE_NATURAL)

    def test_a_variant_only_cell_is_engineered_only(self) -> None:
        from eagent.tools.retrieve_evidence import Maturity
        cell = self.cell(confirmed_wild_type=0, confirmed_variant=2,
                         independent_wild_type_sources=0)
        self.assertIs(cell.maturity(), Maturity.ENGINEERED_ONLY)

    def test_both_counts_are_reported(self) -> None:
        payload = self.cell().to_dict()
        self.assertEqual(payload["independent_sources"], 2)
        self.assertEqual(payload["independent_wild_type_sources"], 1)
        self.assertIn("indwt=1", self.cell().render())

    def test_the_matrix_counts_wild_type_sources_separately(self) -> None:
        """End to end: one wild-type record plus one variant from another source."""
        from test_followup_batch3 import _row
        from eagent.schemas import OutcomeClass, ReactionDirection, ReactionSpec
        from eagent.tools.retrieve_evidence import Maturity, build_evidence_matrix
        target = ReactionSpec(
            reaction_class=ReactionClass.KETONE_TO_SECONDARY_ALCOHOL)
        wild = _row(ReactionDirection.FORWARD_AS_TARGET,
                    ReactionClass.KETONE_TO_SECONDARY_ALCOHOL,
                    OutcomeClass.CONFIRMED_TARGET_PRODUCT)
        variant = _row(ReactionDirection.FORWARD_AS_TARGET,
                       ReactionClass.KETONE_TO_SECONDARY_ALCOHOL,
                       OutcomeClass.CONFIRMED_TARGET_PRODUCT)
        from eagent.schemas import EvidenceRef, EvidenceStrength
        for row, source in ((wild, "brenda"), (variant, "other_database")):
            row.record.evidence = [EvidenceRef(
                source_type="database", identifier=f"{source}:1",
                source_id=source, strength=EvidenceStrength.ANNOTATION_ONLY)]
        variant.record.record_id = "variant"
        variant.record.is_variant = True
        variant.record.parent_sequence_sha256 = "sha256:" + "cd" * 32
        matrix = build_evidence_matrix([wild, variant], target)
        cell = list(matrix.cells.values())[0]
        self.assertEqual(cell.confirmed_wild_type, 1)
        self.assertEqual(cell.confirmed_variant, 1)
        self.assertIsNot(cell.maturity(), Maturity.MATURE_NATURAL)


# --------------------------------------------------------------------------
# FUP-12: a template chosen by filename order
# --------------------------------------------------------------------------

class TemplateAmbiguityIsRefused(unittest.TestCase):
    class Tpl:
        def __init__(self, tid: str, creates: bool) -> None:
            self.template_id = tid
            self.reaction_class = ReactionClass.KETONE_TO_SECONDARY_ALCOHOL.value
            self.creates_stereocenter = creates

    def ctx(self, *templates):
        lib = types.SimpleNamespace(
            reaction_templates={t.template_id: t for t in templates})
        return types.SimpleNamespace(templates=lib)

    def test_one_template_is_used(self) -> None:
        only = self.Tpl("rxn.aryl_ketone", True)
        tpl, origin = _reaction_template_from_context(
            self.ctx(only), ReactionClass.KETONE_TO_SECONDARY_ALCOHOL)
        self.assertIs(tpl, only)
        self.assertIn("reaction_templates", origin)

    def test_two_templates_for_one_class_are_refused(self) -> None:
        a = self.Tpl("rxn.a_symmetric_ketone", False)
        b = self.Tpl("rxn.b_aryl_ketone", True)
        tpl, origin = _reaction_template_from_context(
            self.ctx(a, b), ReactionClass.KETONE_TO_SECONDARY_ALCOHOL)
        self.assertIsNone(tpl)
        self.assertIn("whichever file sorted first", origin)
        self.assertIn("rxn.a_symmetric_ketone", origin)
        self.assertIn("rxn.b_aryl_ketone", origin)

    def test_the_answer_does_not_depend_on_insertion_order(self) -> None:
        a = self.Tpl("rxn.a_symmetric_ketone", False)
        b = self.Tpl("rxn.b_aryl_ketone", True)
        first, _ = _reaction_template_from_context(
            self.ctx(a, b), ReactionClass.KETONE_TO_SECONDARY_ALCOHOL)
        second, _ = _reaction_template_from_context(
            self.ctx(b, a), ReactionClass.KETONE_TO_SECONDARY_ALCOHOL)
        self.assertIs(first, second)

    def test_no_template_still_reports_that(self) -> None:
        tpl, origin = _reaction_template_from_context(
            self.ctx(), ReactionClass.KETONE_TO_SECONDARY_ALCOHOL)
        self.assertIsNone(tpl)
        self.assertIn("no ReactionTemplate", origin)


# --------------------------------------------------------------------------
# FUP-13: two readers, two substrates
# --------------------------------------------------------------------------

PEPTIDE = BiopolymerSubstrateSpec(
    kind=SubstrateKind.PEPTIDE, sequence="ACDEFGHIKY",
    reactive_residues=ReactiveResidues(
        modified=[ResidueRef(position=3, residue="D")]))


class ExactlyOneSubstratePath(unittest.TestCase):
    def test_a_biopolymer_task_alone_is_valid(self) -> None:
        spec = ReactionSpec(biopolymer_substrate=PEPTIDE)
        self.assertTrue(spec.substrate_kind.is_biopolymer)

    def test_a_small_molecule_task_alone_is_valid(self) -> None:
        spec = ReactionSpec(substrate=SubstrateSpec(
            name="acetophenone", isomeric_smiles="CC(=O)c1ccccc1"))
        self.assertFalse(spec.substrate_kind.is_biopolymer)

    def test_both_at_once_is_refused_at_construction(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            ReactionSpec(biopolymer_substrate=PEPTIDE,
                         substrate=SubstrateSpec(
                             name="acetophenone",
                             isomeric_smiles="CC(=O)c1ccccc1"))
        self.assertIn("Exactly one substrate path", str(ctx.exception))

    def test_both_at_once_is_refused_on_assignment(self) -> None:
        """The spec is filled in by assignment, so construction is not enough."""
        spec = ReactionSpec(biopolymer_substrate=PEPTIDE)
        with self.assertRaises(ValidationError):
            spec.substrate = SubstrateSpec(name="acetophenone",
                                           isomeric_smiles="CC(=O)c1ccccc1")

    def test_a_name_alone_is_enough_to_be_a_second_substrate(self) -> None:
        with self.assertRaises(ValidationError):
            ReactionSpec(biopolymer_substrate=PEPTIDE,
                         substrate=SubstrateSpec(name="acetophenone"))


class NormalizeReadsTheDeclaredSubstrate(unittest.TestCase):
    def run_on(self, reaction_kwargs):
        from eagent.schemas import ProductSpec, TaskSpec
        from eagent.tools.normalize_reaction import NormalizeReaction
        from test_normalize_reaction import _ctx
        task = TaskSpec(task_id="T-peptide")
        for field, value in reaction_kwargs.items():
            setattr(task.reaction, field, value)
        task.reaction.product = ProductSpec(
            name="phospho-peptide",
            isomeric_smiles="CC(=O)OP(=O)(O)O")
        return NormalizeReaction().run(_ctx(task))

    def blockers(self, result) -> set[str]:
        return {f.code for f in result.qc_flags
                if f.severity.value == "blocker"}

    def test_a_peptide_task_is_not_asked_for_a_smiles(self) -> None:
        blockers = self.blockers(self.run_on({"biopolymer_substrate": PEPTIDE}))
        self.assertNotIn("substrate_unspecified", blockers)
        self.assertNotIn("substrate_name_only", blockers)

    def test_a_peptide_task_is_not_asked_for_an_atom_map(self) -> None:
        blockers = self.blockers(self.run_on({"biopolymer_substrate": PEPTIDE}))
        self.assertNotIn("atom_map_missing", blockers)

    def test_the_absence_is_recorded_rather_than_silent(self) -> None:
        result = self.run_on({"biopolymer_substrate": PEPTIDE})
        self.assertIn("atom_map_absent_for_biopolymer",
                      {f.code for f in result.qc_flags})

    def test_a_peptide_with_no_sequence_is_blocked(self) -> None:
        bare = BiopolymerSubstrateSpec(kind=SubstrateKind.PEPTIDE,
                                       name="a kinase substrate peptide")
        blockers = self.blockers(self.run_on({"biopolymer_substrate": bare}))
        self.assertIn("biopolymer_substrate_unspecified", blockers)

    def test_a_peptide_with_no_reactive_residue_is_blocked(self) -> None:
        no_residue = BiopolymerSubstrateSpec(kind=SubstrateKind.PEPTIDE,
                                            sequence="ACDEFGHIKY")
        blockers = self.blockers(self.run_on(
            {"biopolymer_substrate": no_residue}))
        self.assertIn("biopolymer_reactive_residue_unspecified", blockers)

    def test_a_small_molecule_task_is_unaffected(self) -> None:
        blockers = self.blockers(self.run_on({}))
        self.assertIn("substrate_unspecified", blockers)
        self.assertIn("atom_map_missing", blockers)


if __name__ == "__main__":
    unittest.main(verbosity=2)
