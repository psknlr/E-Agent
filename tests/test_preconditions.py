"""Tests for the precondition gate chain.

The fixtures are small enough to read by eye, and each one encodes a specific
way a ranking gets quietly corrupted:

* the structure is a point mutant of the candidate (step 1);
* the catalytic residue is unobserved, or the reference numbering axis was
  never built (step 2);
* two files list a cofactor's atoms in different orders and something mapped
  them positionally (step 3) -- the classic silent mis-mapping;
* the bound cofactor is the oxidised form while the mechanism needs a hydride
  donor (step 4);
* a gating distance was never measured (step 5), which is not the same as a
  gating distance that failed.

In every case the point is the same: the broken model produces a perfectly
plausible number, so the error has to be caught before the number exists.
"""

from __future__ import annotations

import unittest

from eagent.datalayer.preconditions import (
    ChainDisposition,
    GateOutcome,
    MAPPING_GATES_RATIONALE,
    PreconditionStep,
    STEP_ORDER,
    check_atom_mapping,
    check_catalytic_geometry,
    check_cofactor_state,
    check_ligand_atom_mapping,
    check_residue_numbering,
    check_sequence_consistency,
    gate_chain,
)

SEQ = "MKVQAWYTGSDLNPFR"
#: Same length, one substitution at index 6 (Y -> F): a deposited point mutant.
MUTANT_SEQ = "MKVQAWFTGSDLNPFR"

#: Reference atom names for the nicotinamide half of the reduced cofactor, in
#: the order the catalytic template lists them.
REF_ATOMS = ["C4N", "N1N", "C2N", "C3N"]
#: The same four atoms as a second file happens to order them. Nothing is
#: wrong with this file; it is only wrong to pair it by position.
REORDERED_ATOMS = ["N1N", "C2N", "C3N", "C4N"]


def _candidate(sequence: str = SEQ, **extra: object) -> dict:
    """A minimal candidate record: the chain reads fields, not model classes."""
    rec = {"candidate_id": "cand-1", "sequence": sequence,
           "catalytic_mapping": {"role_to_index": {"catalytic_tyr": 6}}}
    rec.update(extra)
    return rec


def _structure(sequence: str = SEQ, **extra: object) -> dict:
    rec = {"structure_id": "struct-1", "observed_sequence": sequence}
    rec.update(extra)
    return rec


def _residue_map(length: int = len(SEQ), with_reference: bool = True) -> dict:
    """Author numbering offset by 10, plus the reference axis when asked."""
    return {
        "index_to_author": {i: ("A", i + 10) for i in range(length)},
        "author_to_index": {("A", i + 10): i for i in range(length)},
        "index_to_reference": ({i: i + 1 for i in range(length)}
                               if with_reference else {}),
        "reference_label": "reference-sequence" if with_reference else None,
    }


def _template(**extra: object) -> dict:
    rec = {
        "template_id": "tmpl-1",
        "mechanism_summary": "NADPH-dependent hydride transfer to a ketone",
        "requires_hydride_donor": True,
        "required_cofactor": "NADPH",
        "required_cofactor_state": "reduced",
        "required_ligands": [
            {"role": "cofactor", "component_id": "NDP", "atom_names": REF_ATOMS},
        ],
        "geometry_constraints": [
            {"name": "hydride_distance", "severity": "gating",
             "min_value": 2.4, "max_value": 3.6, "calibrated_on": ["fixture"]},
            {"name": "burial", "severity": "scoring", "min_value": 0.0},
        ],
    }
    rec.update(extra)
    return rec


def _pose(atom_names=None, **extra: object) -> dict:
    rec = {
        "pose_id": "pose-1",
        "ligands": [{"role": "cofactor", "component_id": "NDP",
                     "atom_names": list(atom_names or REF_ATOMS)}],
        "cofactor": {"ligand_code": "NDP"},
        "measurements": {"hydride_distance": 3.0},
    }
    rec.update(extra)
    return rec


class TestChainShape(unittest.TestCase):
    """The chain is ordered, inspectable, and honest about what it proves."""

    def test_step_order_is_fixed_and_geometry_is_last(self) -> None:
        self.assertEqual([s.order for s in STEP_ORDER], [1, 2, 3, 4, 5])
        self.assertIs(STEP_ORDER[-1], PreconditionStep.CATALYTIC_GEOMETRY)

    def test_only_the_last_step_claims_scientific_insight(self) -> None:
        for step in STEP_ORDER[:4]:
            self.assertFalse(step.adds_scientific_insight, step.value)
        self.assertTrue(PreconditionStep.CATALYTIC_GEOMETRY.adds_scientific_insight)
        self.assertIn("no scientific insight", MAPPING_GATES_RATIONALE)

    def test_a_clean_candidate_passes_all_five_gates(self) -> None:
        report = gate_chain(_candidate(), _structure(), _pose(), _template(),
                            residue_map=_residue_map())
        self.assertTrue(report.all_passed, report.describe())
        self.assertIsNone(report.first_failure)
        self.assertIs(report.disposition, ChainDisposition.READY_FOR_RANKING)
        self.assertTrue(report.may_enter_ranking)
        self.assertEqual(report.repair_plan(), [])
        self.assertIsNone(report.rerun_from())


class TestStopsAtFirstFailure(unittest.TestCase):
    """A failure stops the chain, and the later gates are not claimed as checked."""

    def test_sequence_mismatch_stops_the_chain_with_a_repair_plan(self) -> None:
        report = gate_chain(_candidate(), _structure(MUTANT_SEQ), _pose(),
                            _template(), residue_map=_residue_map())

        self.assertIs(report.first_failure, PreconditionStep.SEQUENCE_CONSISTENCY)
        first = report.first_failure_result
        self.assertIs(first.outcome, GateOutcome.FAILED)
        self.assertEqual(first.evidence["n_mismatches"], 1)
        self.assertEqual(first.evidence["mismatches"][0],
                         {"index": 6, "candidate": "Y", "structure": "F"})

        # Everything after step 1 is explicitly "not reached", never "passed".
        later = [r for r in report.results if r.step.order > 1]
        self.assertEqual(len(later), 4)
        for r in later:
            self.assertIs(r.outcome, GateOutcome.NOT_EVALUATED)
            self.assertTrue(r.not_reached)

        plan = report.repair_plan()
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["step"], "sequence_consistency")
        self.assertEqual(plan[0]["action"],
                         "reselect_or_declare_mutant_structure")
        self.assertTrue(plan[0]["blocking"])
        self.assertEqual(plan[0]["rerun_from"], "sequence_consistency")
        self.assertIs(report.rerun_from(), PreconditionStep.SEQUENCE_CONSISTENCY)

        # Repairable, not rankable, and never silently dropped.
        self.assertIs(report.disposition, ChainDisposition.REPAIR_AND_RERUN)
        self.assertFalse(report.may_enter_ranking)
        self.assertFalse(report.discard_allowed)

    def test_a_declared_mutant_may_be_accepted_but_is_reported(self) -> None:
        structure = _structure(MUTANT_SEQ, mutations_in_structure=["Y16F"])
        blocked = gate_chain(_candidate(), structure, _pose(), _template(),
                             residue_map=_residue_map())
        self.assertIs(blocked.first_failure,
                      PreconditionStep.SEQUENCE_CONSISTENCY)

        allowed = gate_chain(_candidate(), structure, _pose(), _template(),
                             residue_map=_residue_map(),
                             allow_declared_mutations=True)
        self.assertTrue(allowed.all_passed, allowed.describe())
        warnings = allowed.result_for(
            PreconditionStep.SEQUENCE_CONSISTENCY).warnings
        self.assertTrue(any("declared mutant" in w for w in warnings))

    def test_missing_sequence_is_not_evaluated_rather_than_failed(self) -> None:
        result = check_sequence_consistency(_candidate(), {"structure_id": "s"})
        self.assertIs(result.outcome, GateOutcome.NOT_EVALUATED)
        self.assertEqual(result.repair.action, "supply_sequence")
        self.assertIn("neither passed nor discarded", result.summary)

    def test_length_mismatch_names_the_containment_relation(self) -> None:
        tagged = _structure("MHHHHHH" + SEQ)
        result = check_sequence_consistency(_candidate(), tagged)
        self.assertIs(result.outcome, GateOutcome.FAILED)
        self.assertIn("contiguous subsequence of the structure",
                      result.evidence["length_relation"])


class TestResidueNumbering(unittest.TestCase):
    """Step 2: three numbering axes, related explicitly or not at all."""

    def test_absent_map_is_not_evaluated_with_a_repair(self) -> None:
        result = check_residue_numbering(_candidate(), _structure())
        self.assertIs(result.outcome, GateOutcome.NOT_EVALUATED)
        self.assertEqual(result.repair.action, "build_residue_map")

    def test_unobserved_catalytic_residue_fails(self) -> None:
        rmap = _residue_map()
        del rmap["index_to_author"][6]           # the catalytic tyrosine
        result = check_residue_numbering(_candidate(), _structure(),
                                         residue_map=rmap)
        self.assertIs(result.outcome, GateOutcome.FAILED)
        self.assertIn("catalytic_tyr", result.summary)
        self.assertEqual(result.repair.action, "resolve_catalytic_positions")

    def test_reference_axis_is_required_only_when_something_cites_one(self) -> None:
        no_ref = _residue_map(with_reference=False)
        engineered = check_residue_numbering(_candidate(), _structure(),
                                             residue_map=no_ref)
        self.assertIs(engineered.outcome, GateOutcome.PASSED)
        self.assertNotIn("reference", engineered.evidence["axes"])

        with_accession = check_residue_numbering(
            _candidate(accession="P12345"), _structure(), residue_map=no_ref)
        self.assertIs(with_accession.outcome, GateOutcome.FAILED)
        self.assertEqual(with_accession.repair.action,
                         "establish_reference_numbering")


class TestAtomMapping(unittest.TestCase):
    """Step 3: atoms are matched by name, and file order is never an identity."""

    def test_differing_atom_order_is_detected_and_mapped_by_name(self) -> None:
        check = check_atom_mapping(REF_ATOMS, REORDERED_ATOMS,
                                   component_id="NDP")
        self.assertTrue(check.ok)
        self.assertTrue(check.order_differs)
        # The pairs a positional read would have produced, named explicitly.
        self.assertEqual(check.positional_would_mismatch[0], (0, "C4N", "N1N"))
        self.assertEqual(check.matched["C4N"], 3)
        self.assertTrue(any("positional read would have been wrong" in w
                            for w in check.warnings))

    def test_mapping_declared_on_file_order_is_refused_when_orders_differ(self) -> None:
        check = check_atom_mapping(REF_ATOMS, REORDERED_ATOMS,
                                   mapping_basis="file_order",
                                   component_id="NDP")
        self.assertFalse(check.ok)
        self.assertIn("different atoms of the same component", check.reason)

    def test_declared_mapping_pairing_different_names_is_refused(self) -> None:
        check = check_atom_mapping(
            REF_ATOMS, REORDERED_ATOMS,
            declared_mapping={"C4N": "N1N"}, component_id="NDP")
        self.assertFalse(check.ok)
        self.assertTrue(check.declared_conflicts)

    def test_missing_and_duplicated_atom_names_are_distinguished(self) -> None:
        missing = check_atom_mapping(REF_ATOMS, ["N1N", "C2N", "C3N"])
        self.assertFalse(missing.ok)
        self.assertEqual(missing.missing, ("C4N",))

        duplicated = check_atom_mapping(REF_ATOMS,
                                        ["C4N", "C4N", "N1N", "C2N", "C3N"])
        self.assertFalse(duplicated.ok)
        self.assertEqual(duplicated.duplicated, ("C4N",))

    def test_order_difference_is_caught_at_step_three_of_the_chain(self) -> None:
        # Two files, same ligand, different atom order, and a pose that says it
        # mapped them by file order. The chain must stop here -- before any
        # distance is measured to what would be the wrong atom.
        pose = _pose()
        pose["ligands"] = [{"role": "cofactor", "component_id": "NDP",
                            "atom_names": REORDERED_ATOMS,
                            "mapping_basis": "file_order"}]
        report = gate_chain(_candidate(), _structure(), pose, _template(),
                            residue_map=_residue_map())

        self.assertIs(report.first_failure, PreconditionStep.LIGAND_ATOM_MAPPING)
        result = report.result_for(PreconditionStep.LIGAND_ATOM_MAPPING)
        self.assertIs(result.outcome, GateOutcome.FAILED)
        self.assertIn("C4N->N1N", result.summary)
        evidence = result.evidence["ligands"]["cofactor"]
        self.assertTrue(evidence["order_differs"])
        self.assertEqual(evidence["positional_would_mismatch"][0],
                         {"index": 0, "reference_atom": "C4N",
                          "pose_atom": "N1N"})
        self.assertEqual(report.repair_plan()[0]["action"],
                         "remap_ligand_atoms_by_name")
        # The geometry gate was never run, so no distance exists to rank.
        self.assertIs(report.result_for(
            PreconditionStep.CATALYTIC_GEOMETRY).outcome,
            GateOutcome.NOT_EVALUATED)
        self.assertFalse(report.may_enter_ranking)

    def test_ligand_without_a_component_id_cannot_be_mapped(self) -> None:
        pose = _pose()
        pose["ligands"] = [{"role": "cofactor", "atom_names": REF_ATOMS}]
        result = check_ligand_atom_mapping(pose, _template())
        self.assertIs(result.outcome, GateOutcome.FAILED)
        self.assertIn("no chemical component id", result.summary)

    def test_wrong_component_id_is_a_different_molecule(self) -> None:
        pose = _pose()
        pose["ligands"] = [{"role": "cofactor", "component_id": "NAP",
                            "atom_names": REF_ATOMS}]
        result = check_ligand_atom_mapping(pose, _template())
        self.assertIs(result.outcome, GateOutcome.FAILED)
        self.assertIn("component id mismatch", result.summary)


class TestCofactorStateGate(unittest.TestCase):
    """Step 4: the oxidation state is verified, never assumed."""

    def test_oxidised_cofactor_fails_a_hydride_transfer_mechanism(self) -> None:
        pose = _pose()
        pose["ligands"] = [{"role": "cofactor", "component_id": "NAP",
                            "atom_names": REF_ATOMS}]
        pose["cofactor"] = {"ligand_code": "NAP"}        # NADP+, not NADPH
        template = _template(required_ligands=[
            {"role": "cofactor", "component_id": "NAP", "atom_names": REF_ATOMS}])
        result = check_cofactor_state(pose, template)
        self.assertIs(result.outcome, GateOutcome.FAILED)
        self.assertIn("cannot donate", result.summary)
        self.assertEqual(result.repair.action, "rebuild_with_reduced_cofactor")

    def test_unknown_state_is_not_evaluated_rather_than_failed(self) -> None:
        pose = _pose(cofactor={"name": "NAD(P)H"})
        result = check_cofactor_state(pose, _template())
        self.assertIs(result.outcome, GateOutcome.NOT_EVALUATED)
        self.assertEqual(result.repair.action,
                         "determine_cofactor_oxidation_state")
        self.assertTrue(result.repair.detail)

    def test_template_without_a_requirement_is_not_evaluated(self) -> None:
        bare = {"template_id": "t", "geometry_constraints": []}
        result = check_cofactor_state(_pose(), bare)
        self.assertIs(result.outcome, GateOutcome.NOT_EVALUATED)
        self.assertTrue(result.repair.requires_human)


class TestGeometryGate(unittest.TestCase):
    """Step 5: the only step that measures chemistry, and it still refuses."""

    def test_unmeasured_constraint_is_not_a_satisfied_one(self) -> None:
        pose = _pose(measurements={})
        result = check_catalytic_geometry(pose, _template())
        self.assertIs(result.outcome, GateOutcome.NOT_EVALUATED)
        self.assertEqual(result.repair.action, "measure_missing_geometry")

    def test_violated_window_fails(self) -> None:
        pose = _pose(measurements={"hydride_distance": 7.9})
        result = check_catalytic_geometry(pose, _template())
        self.assertIs(result.outcome, GateOutcome.FAILED)
        self.assertEqual(result.repair.params["violated"], ["hydride_distance"])

    def test_restrained_constraints_are_not_counted_as_independent(self) -> None:
        pose = _pose(restrained_constraints=["hydride_distance"])
        result = check_catalytic_geometry(pose, _template())
        self.assertIs(result.outcome, GateOutcome.PASSED)
        self.assertEqual(result.evidence["independent_total"], 0)
        self.assertEqual(result.evidence["circular_constraints"],
                         ["hydride_distance"])
        self.assertTrue(any("restrained" in w for w in result.warnings))

    def test_only_gating_constraints_gate(self) -> None:
        result = check_catalytic_geometry(_pose(), _template())
        self.assertIs(result.outcome, GateOutcome.PASSED)
        self.assertEqual(list(result.evidence["constraints"]),
                         ["hydride_distance"])


class TestReportRendering(unittest.TestCase):
    """The report must show the whole chain, including what was skipped."""

    def test_report_lists_every_gate_and_the_disposition(self) -> None:
        report = gate_chain(_candidate(), _structure(MUTANT_SEQ), _pose(),
                            _template(), residue_map=_residue_map())
        text = report.describe()
        for step in STEP_ORDER:
            self.assertIn(step.title, text)
        self.assertIn("not reached", text)
        self.assertIn("discard allowed: False", text)

        payload = report.as_dict()
        self.assertEqual(len(payload["results"]), 5)
        self.assertEqual(payload["first_failure"], "sequence_consistency")
        self.assertFalse(payload["may_enter_ranking"])
        self.assertFalse(payload["discard_allowed"])
        self.assertIn("no scientific insight", payload["rationale"])


if __name__ == "__main__":  # pragma: no cover - pytest may not be installed
    unittest.main(verbosity=2)
