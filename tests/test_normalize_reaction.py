"""Tests for :mod:`eagent.tools.normalize_reaction`.

These are written against the refusals, not against the happy path. The step is
valuable only insofar as it declines to complete a specification, so each test
asserts that something was *not* filled in, or that an inconsistency that a
human would miss was caught.

Runs under pytest, or standalone with
``PYTHONPATH=src python3 tests/test_normalize_reaction.py``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from eagent.envelope import Severity, Status
from eagent.context import ExecutionPolicy, RunContext
from eagent.provenance import RunManifest
from eagent.schemas import (
    AtomRef, CofactorSpec, CofactorState, ReactionClass, ReactiveAtoms,
    Stereochemistry, TaskMode, TaskSpec,
)
from eagent.tools.normalize_reaction import (
    PROCHIRAL_CENTRE_IS_ELECTROPHILE, SUBSTRATE_CLASS_SCAFFOLD,
    NormalizeReaction, StereocentreBasis, SubstrateChemotype,
    parse_atom_map, perceive_new_stereocenter, split_reaction_smiles,
    validate_atom_map,
)

ACETOPHENONE = "CC(=O)c1ccccc1"
PHENYLETHANOL = "C[C@H](O)c1ccccc1"
MAPPED = ("[CH3:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1>>"
          "[CH3:1][CH:2]([OH:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1")


def _ctx(task: TaskSpec, **policy_kw) -> RunContext:
    """A run context on a throwaway working directory."""
    return RunContext(
        task=task,
        workdir=Path(tempfile.mkdtemp(prefix="eagent-normalize-")),
        manifest=RunManifest(run_id="test-run", task_id=task.task_id),
        policy=ExecutionPolicy(**policy_kw),
    )


def _codes(result) -> set[str]:
    return {f.code for f in result.qc_flags}


def _blocker_codes(result) -> set[str]:
    return {f.code for f in result.qc_flags if f.severity is Severity.BLOCKER}


def _full_task() -> TaskSpec:
    """A task whose reaction spec is complete and internally consistent."""
    task = TaskSpec(task_id="pilot")
    rxn = task.reaction
    rxn.reaction_class = ReactionClass.KETONE_TO_SECONDARY_ALCOHOL
    rxn.atom_mapped_reaction_smiles = MAPPED
    rxn.substrate.name = "acetophenone"
    rxn.substrate.isomeric_smiles = ACETOPHENONE
    rxn.substrate.reactive_atoms = ReactiveAtoms(
        electrophile=AtomRef(atom_map_id=2, element="C", role="carbonyl_C"),
        prochiral_center=AtomRef(atom_map_id=2, element="C"),
        stabilised_atoms=[AtomRef(atom_map_id=3, element="O",
                                  role="carbonyl_O")],
    )
    rxn.product.name = "(S)-1-phenylethanol"
    rxn.product.isomeric_smiles = PHENYLETHANOL
    rxn.product.target_stereochemistry = Stereochemistry.S
    rxn.product.creates_new_stereocenter = True
    rxn.product.authentic_standard_available = True
    task.conditions.cofactor_options = [
        CofactorSpec(name="NADPH", state=CofactorState.REDUCED,
                     ligand_code="NDP",
                     transfer_atom=AtomRef(atom_map_id=40, element="C",
                                           role="hydride_donor_C4"))
    ]
    return task


# ---------------------------------------------------------------------------
# atom-map primitives
# ---------------------------------------------------------------------------

class AtomMapParsingTests(unittest.TestCase):

    def test_only_bracket_atoms_carry_map_ids(self) -> None:
        ids, dupes = parse_atom_map("[CH3:1][C:2](=[O:3])c1ccccc1")
        self.assertEqual(ids, {1: "C", 2: "C", 3: "O"})
        self.assertEqual(dupes, [])

    def test_aromatic_and_charged_atoms_parse(self) -> None:
        ids, _ = parse_atom_map("[nH+:7][c:8]1[cH:9]1")
        self.assertEqual(ids, {7: "n", 8: "c", 9: "c"})

    def test_duplicate_ids_are_reported(self) -> None:
        _, dupes = parse_atom_map("[C:1][C:1]")
        self.assertEqual(dupes, [1])

    def test_a_molecule_is_not_a_reaction(self) -> None:
        self.assertIsNone(split_reaction_smiles(ACETOPHENONE))
        self.assertIsNotNone(split_reaction_smiles("A>>B"))


class AtomMapValidationTests(unittest.TestCase):

    def test_missing_mapping_blocks(self) -> None:
        report = validate_atom_map(None, {})
        self.assertIn("atom_map_missing",
                      {i.code for i in report.blockers})

    def test_referenced_id_absent_is_a_blocker(self) -> None:
        refs = {"substrate.electrophile": AtomRef(atom_map_id=99, element="C")}
        report = validate_atom_map(MAPPED, refs)
        codes = {i.code for i in report.blockers}
        self.assertIn("atom_map_missing_id", codes)

    def test_element_mismatch_is_a_blocker(self) -> None:
        refs = {"substrate.electrophile": AtomRef(atom_map_id=3, element="C")}
        report = validate_atom_map(MAPPED, refs)
        self.assertIn("atom_map_element_mismatch",
                      {i.code for i in report.blockers})

    def test_product_only_id_is_a_blocker(self) -> None:
        bad = "[CH3:1][C:2]=[O:3]>>[CH3:1][CH:2]([OH:3])[CH3:77]"
        report = validate_atom_map(bad, {})
        self.assertIn("atom_map_product_only_id",
                      {i.code for i in report.blockers})

    def test_reactant_only_id_is_a_warning_not_a_blocker(self) -> None:
        partial = "[CH3:1][C:2]=[O:3]>>[CH3:1][CH:2][OH:3].[Cl:9]"
        report = validate_atom_map("[Cl:9][CH3:1][C:2]=[O:3]>>[CH3:1][CH:2][OH:3]",
                                   {})
        codes = {i.code for i in report.issues}
        self.assertIn("atom_map_reactant_only_id", codes)
        self.assertNotIn("atom_map_reactant_only_id",
                         {i.code for i in report.blockers})
        self.assertIsNotNone(split_reaction_smiles(partial))

    def test_cofactor_in_the_agent_position_is_addressable(self) -> None:
        """A cofactor written as an agent still has to be referenceable."""
        rxn = "[C:2]=[O:3]>[C:40]>[C:2][O:3]"
        refs = {"cofactor.NADPH.transfer_atom": AtomRef(atom_map_id=40,
                                                        element="C")}
        report = validate_atom_map(rxn, refs)
        self.assertNotIn("atom_map_missing_id",
                         {i.code for i in report.blockers})

    def test_electrophile_and_prochiral_centre_must_coincide(self) -> None:
        refs = {
            "substrate.electrophile": AtomRef(atom_map_id=2, element="C"),
            "substrate.prochiral_center": AtomRef(atom_map_id=1, element="C"),
        }
        report = validate_atom_map(
            MAPPED, refs,
            reaction_class=ReactionClass.KETONE_TO_SECONDARY_ALCOHOL)
        self.assertIn("electrophile_prochiral_mismatch",
                      {i.code for i in report.blockers})

    def test_the_coincidence_rule_is_class_scoped(self) -> None:
        """Only the classes in the table demand it; others get a warning."""
        self.assertIn(ReactionClass.KETONE_TO_SECONDARY_ALCOHOL,
                      PROCHIRAL_CENTRE_IS_ELECTROPHILE)
        self.assertNotIn(ReactionClass.HYDROXYLATION,
                         PROCHIRAL_CENTRE_IS_ELECTROPHILE)
        refs = {
            "substrate.electrophile": AtomRef(atom_map_id=2),
            "substrate.prochiral_center": AtomRef(atom_map_id=1),
        }
        report = validate_atom_map(MAPPED, refs,
                                   reaction_class=ReactionClass.HYDROXYLATION)
        self.assertEqual([], report.blockers)
        self.assertIn("electrophile_prochiral_differ",
                      {i.code for i in report.issues})


# ---------------------------------------------------------------------------
# the stereocentre refusal
# ---------------------------------------------------------------------------

class StereocentrePerceptionTests(unittest.TestCase):

    def test_no_toolkit_means_no_answer(self) -> None:
        call = perceive_new_stereocenter(ACETOPHENONE, PHENYLETHANOL)
        if call.basis is StereocentreBasis.UNDETERMINED:
            self.assertIsNone(call.value)
            self.assertTrue(call.detail)
        else:  # RDKit present: still only ever a proposal
            self.assertIs(call.basis, StereocentreBasis.TOOLKIT_PROPOSAL)
            self.assertFalse(call.writable_to_spec)

    def test_missing_structures_cannot_be_answered(self) -> None:
        self.assertIsNone(perceive_new_stereocenter(None, PHENYLETHANOL).value)
        self.assertIsNone(perceive_new_stereocenter(ACETOPHENONE, None).value)

    def test_a_toolkit_proposal_may_not_write_into_the_spec(self) -> None:
        """The assumption ledger has no 'a tool said so' authority."""
        from eagent.tools.normalize_reaction import StereocentreCall

        proposal = StereocentreCall(True, StereocentreBasis.TOOLKIT_PROPOSAL,
                                    "perceived")
        self.assertFalse(proposal.writable_to_spec)
        template = StereocentreCall(True, StereocentreBasis.TEMPLATE, "",
                                    source="template:rt-1")
        self.assertTrue(template.writable_to_spec)


# ---------------------------------------------------------------------------
# the interface
# ---------------------------------------------------------------------------

class NormalizeReactionTests(unittest.TestCase):

    def test_a_name_is_never_resolved_into_a_structure(self) -> None:
        task = TaskSpec(task_id="t")
        task.reaction.substrate.name = "acetophenone"
        task.reaction.product.name = "1-phenylethanol"
        result = NormalizeReaction().run(_ctx(task))

        self.assertIs(result.status, Status.PARTIAL)
        self.assertIn("substrate_name_only", _blocker_codes(result))
        self.assertIn("product_name_only", _blocker_codes(result))
        # and nothing was invented on the way
        self.assertIsNone(task.reaction.substrate.isomeric_smiles)
        self.assertIsNone(task.reaction.product.isomeric_smiles)
        self.assertEqual([], task.assumptions)

    def test_stereocentre_is_not_inferred_from_the_reaction_class(self) -> None:
        """An aldehyde and a symmetric ketone both reduce to an achiral alcohol."""
        task = TaskSpec(task_id="t")
        task.reaction.reaction_class = ReactionClass.KETONE_TO_SECONDARY_ALCOHOL
        task.reaction.substrate.isomeric_smiles = "CC(=O)C"   # acetone: symmetric
        task.reaction.product.isomeric_smiles = "CC(O)C"
        result = NormalizeReaction().run(_ctx(task))

        self.assertIsNone(task.reaction.product.creates_new_stereocenter)
        self.assertIn("creates_new_stereocenter",
                      " ".join(result.data["unresolved_for_gate"]))
        self.assertTrue(any(u.code == "stereocentre_undetermined"
                            for u in result.uncertainty))

    def test_a_sourced_template_may_resolve_the_stereocentre(self) -> None:
        class _Template:
            template_id = "rt-ketone-1"
            reaction_class = "ketone_to_secondary_alcohol"
            creates_stereocenter = True

        task = TaskSpec(task_id="t")
        task.reaction.reaction_class = ReactionClass.KETONE_TO_SECONDARY_ALCOHOL
        task.reaction.substrate.isomeric_smiles = ACETOPHENONE
        task.reaction.product.isomeric_smiles = PHENYLETHANOL
        task.reaction.atom_mapped_reaction_smiles = MAPPED
        NormalizeReaction().run(_ctx(task), reaction_template=_Template())

        self.assertTrue(task.reaction.product.creates_new_stereocenter)
        self.assertEqual(1, len(task.assumptions))
        self.assertEqual("template:rt-ketone-1", task.assumptions[0].source)

    def test_target_configuration_without_a_stereocentre_is_a_blocker(self) -> None:
        task = _full_task()
        task.reaction.product.creates_new_stereocenter = False
        result = NormalizeReaction().run(_ctx(task))
        self.assertIn("stereo_target_without_stereocentre",
                      _blocker_codes(result))

    def test_cofactor_state_contradicting_its_ligand_code_is_a_blocker(self) -> None:
        """NAP is NADP+; declaring it reduced would license a false hydride donor."""
        task = _full_task()
        task.conditions.cofactor_options = [
            CofactorSpec(name="NADPH", state=CofactorState.REDUCED,
                         ligand_code="NAP",
                         transfer_atom=AtomRef(atom_map_id=40))
        ]
        result = NormalizeReaction().run(_ctx(task))
        self.assertIn("cofactor_state_contradicts_ligand_code",
                      _blocker_codes(result))

    def test_a_consistent_complete_spec_succeeds(self) -> None:
        task = _full_task()
        # The cofactor transfer atom must be addressable in the mapping.
        task.reaction.atom_mapped_reaction_smiles = MAPPED.replace(
            ">>", ">[C:40]>")
        result = NormalizeReaction().run(_ctx(task))

        self.assertEqual([], result.blockers, msg=[f.code for f in result.blockers])
        self.assertEqual([], result.data["unresolved_for_gate"])
        self.assertIs(result.status, Status.SUCCESS)
        self.assertTrue(result.ok)
        self.assertTrue(any(a.action == "request_approval"
                            for a in result.next_actions))

    def test_mode_b_offers_sub_spaces_and_refuses_a_best_enzyme(self) -> None:
        task = TaskSpec(task_id="t",
                        task_mode=TaskMode.REACTION_SPACE_EXPLORATION)
        task.reaction.reaction_class = ReactionClass.KETONE_TO_SECONDARY_ALCOHOL
        result = NormalizeReaction().run(_ctx(task))

        codes = _codes(result)
        self.assertIn("substrate_class_undecided", codes)
        self.assertIn("best_enzyme_claim_refused", codes)
        # the missing substrate is the point of mode B, not a blocking error
        self.assertNotIn("substrate_unspecified", _blocker_codes(result))

        scaffold = result.data["checks"]["substrate_class_decision"]
        named = {s["chemotype"] for s in scaffold["sub_spaces"]}
        self.assertEqual(
            {SubstrateChemotype.AROMATIC_KETONE.value,
             SubstrateChemotype.ALIPHATIC_KETONE.value,
             SubstrateChemotype.CYCLIC_KETONE.value,
             SubstrateChemotype.FUNCTIONALISED_KETONE.value},
            named)
        self.assertTrue(scaffold["refusals"])

    def test_the_scaffold_states_decisions_not_conclusions(self) -> None:
        """Every sub-space must ask something, not assert enzyme availability."""
        for sub in SUBSTRATE_CLASS_SCAFFOLD:
            self.assertTrue(sub.decision_required)
            self.assertTrue(sub.evidence_to_gather)
            self.assertTrue(sub.representative_question.endswith("?"))

    def test_the_artifact_carries_the_spec_and_its_gaps(self) -> None:
        task = TaskSpec(task_id="t")
        task.reaction.substrate.name = "acetophenone"
        ctx = _ctx(task)
        result = NormalizeReaction().run(ctx)

        artifact = result.artifact("reaction_spec")
        self.assertIsNotNone(artifact)
        self.assertTrue(artifact.sha256)
        document = yaml.safe_load(Path(artifact.path).read_text(encoding="utf-8"))
        self.assertEqual("t", document["task_id"])
        self.assertIn("unresolved_for_gate", document["checks"])
        self.assertIsNone(document["reaction"]["substrate"]["isomeric_smiles"])

    def test_identical_substrate_and_product_describe_no_reaction(self) -> None:
        task = TaskSpec(task_id="t")
        task.reaction.substrate.isomeric_smiles = ACETOPHENONE
        task.reaction.product.isomeric_smiles = ACETOPHENONE
        result = NormalizeReaction().run(_ctx(task))
        self.assertIn("substrate_product_identical", _blocker_codes(result))

    def test_provenance_records_no_database_and_the_toolkit_state(self) -> None:
        task = _full_task()
        result = NormalizeReaction().run(_ctx(task))
        prov = result.provenance
        self.assertEqual({}, prov.databases)        # this step reads nothing
        self.assertIn("rdkit", prov.models)
        self.assertIsNotNone(prov.random_seed)
        self.assertIn("task_spec", prov.inputs_sha256)

    def test_the_step_runs_on_an_empty_spec_rather_than_refusing(self) -> None:
        """Its job is triage, so it must not demand the fields it checks."""
        self.assertEqual((), NormalizeReaction.required_fields)
        task = TaskSpec(task_id="t")
        result = NormalizeReaction().run(_ctx(task, strict=True))
        self.assertIs(result.status, Status.PARTIAL)
        self.assertTrue(result.data["unresolved_for_gate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
