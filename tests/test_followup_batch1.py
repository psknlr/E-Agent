"""Regression tests for the follow-up audit's first batch.

Three findings, each reproduced before being fixed. They share the shape the
auditor identified as the dominant one: the numbers survive a module boundary
and the scientific identity attached to them does not.

* the plate a measurement came from is dropped, so a signal is divided by a
  background it never shared a plate with;
* the substrate is dropped, so a parent and a variant measured on different
  chemistry are subtracted from each other;
* the cofactor's oxidation state is dropped, so two deliberately different
  conditions become one contradictory record.
"""

from __future__ import annotations

import unittest

from eagent.eval.metrics import (
    ChemicalTaskMismatchError, OutcomeRow, variant_versus_parent,
)
from eagent.schemas import OutcomeClass
from eagent.schemas.chem import CofactorSpec, CofactorState
from eagent.tools.ingest_results import (
    AssayRow, IngestResults, attach_empty_vector_baselines, group_rows,
)


def well(plate: str, name: str, kind: str, value: float, rep: int,
         cofactor: str = "NADPH", state: str = "reduced") -> AssayRow:
    return AssayRow(
        plan_id="p", slot=1, plate=plate, well=name, candidate_id="c1",
        construct_id="c1", kind=kind, role="", parent_candidate_id="",
        mutations=(), cofactor=cofactor, cofactor_state=state, replicate=rep,
        tested=True, expressed_soluble=True, detection_method="GC-MS",
        confirms_product_identity=True, authentic_standard=True,
        chiral_method_validated=True, limit_of_detection=0.1,
        limit_unit="U/mg", measurement_type="specific_activity",
        measurement_value=value, measurement_unit="U/mg", conversion_pct=None,
        product_identity_observed="target", peak_area_target=None,
        peak_area_opposite=None, notes="")


class CrossPlateFoldTests(unittest.TestCase):
    """A fold must be taken against the background of its own plate."""

    def setUp(self) -> None:
        # Each plate reaches exactly 2x. Pooling turned this into 10x,
        # because the plate with the high signal had three wells and the
        # plate with the low background had three controls.
        self.rows = [
            well("P1", "A1", "candidate", 10.0, 1),
            well("P1", "A2", "candidate", 10.0, 2),
            well("P1", "A3", "candidate", 10.0, 3),
            well("P1", "H1", "empty_vector", 5.0, 1),
            well("P2", "B1", "candidate", 2.0, 4),
            well("P2", "H1", "empty_vector", 1.0, 1),
            well("P2", "H2", "empty_vector", 1.0, 2),
            well("P2", "H3", "empty_vector", 1.0, 3),
        ]
        self.group = group_rows([r for r in self.rows
                                 if r.kind == "candidate"])[0]
        self.notes = attach_empty_vector_baselines([self.group], self.rows)

    def test_each_plate_keeps_its_own_background(self) -> None:
        self.assertEqual(self.group.empty_vector_by_plate,
                         {"P1": 5.0, "P2": 1.0})

    def test_the_per_plate_folds_are_both_two(self) -> None:
        self.assertEqual(self.group.fold_by_plate(), {"P1": 2.0, "P2": 2.0})

    def test_the_combined_fold_does_not_exceed_either_plate(self) -> None:
        fold = self.group.fold_over_empty_vector
        self.assertAlmostEqual(fold, 2.0)
        self.assertLessEqual(fold, max(self.group.fold_by_plate().values()))

    def test_a_three_fold_bar_is_not_met(self) -> None:
        self.assertLess(self.group.fold_over_empty_vector, 3.0,
                        "neither plate reached 3x, so the round did not")

    def test_spanning_plates_is_recorded(self) -> None:
        self.assertTrue([n for n in self.notes if "never pooled" in n])

    def test_a_plate_without_a_control_contributes_nothing(self) -> None:
        rows = [well("P1", "A1", "candidate", 10.0, 1),
                well("P1", "H1", "empty_vector", 5.0, 1),
                well("P3", "C1", "candidate", 99.0, 2)]
        group = group_rows([r for r in rows if r.kind == "candidate"])[0]
        attach_empty_vector_baselines([group], rows)
        self.assertEqual(set(group.fold_by_plate()), {"P1"})
        self.assertIn("P3", group.plates_without_background)
        self.assertAlmostEqual(group.fold_over_empty_vector, 2.0)

    def test_a_single_plate_is_unaffected(self) -> None:
        rows = [well("P1", "A1", "candidate", 30.0, 1),
                well("P1", "H1", "empty_vector", 10.0, 1)]
        group = group_rows([r for r in rows if r.kind == "candidate"])[0]
        attach_empty_vector_baselines([group], rows)
        self.assertAlmostEqual(group.fold_over_empty_vector, 3.0)


class ChemicalTaskTests(unittest.TestCase):
    """An engineering delta needs both sides on the same chemistry."""

    def row(self, cid: str, value: float, substrate: str, product: str) -> OutcomeRow:
        return OutcomeRow(
            candidate_id=cid, outcome=OutcomeClass.CONFIRMED_TARGET_PRODUCT,
            soluble_expression=True, measurement_type="specific_activity",
            measurement_value=value, measurement_unit="U/mg",
            replicate_values=(value * 0.99, value, value * 1.01),
            conditions={"pH": 7.0}, substrate_key=substrate,
            product_key=product, reaction_direction="forward_as_target")

    def test_a_cross_substrate_pair_is_refused(self) -> None:
        with self.assertRaises(ChemicalTaskMismatchError) as ctx:
            variant_versus_parent(
                self.row("v", 100.0, "acetaldehyde", "ethanol"),
                self.row("p", 1.0, "acetophenone", "1-phenylethanol"))
        self.assertIn("no baseline for a difference", str(ctx.exception))

    def test_a_cross_product_pair_is_refused(self) -> None:
        with self.assertRaises(ChemicalTaskMismatchError):
            variant_versus_parent(
                self.row("v", 100.0, "acetophenone", "(R)-1-phenylethanol"),
                self.row("p", 1.0, "acetophenone", "(S)-1-phenylethanol"))

    def test_the_same_chemistry_still_compares(self) -> None:
        comparison = variant_versus_parent(
            self.row("v", 3.0, "acetophenone", "1-phenylethanol"),
            self.row("p", 1.0, "acetophenone", "1-phenylethanol"))
        self.assertAlmostEqual(comparison.delta_measurement, 2.0)

    def test_a_reverse_direction_pair_is_refused(self) -> None:
        forward = self.row("v", 3.0, "acetophenone", "1-phenylethanol")
        reverse = OutcomeRow(
            candidate_id="p", outcome=OutcomeClass.CONFIRMED_TARGET_PRODUCT,
            soluble_expression=True, measurement_type="specific_activity",
            measurement_value=1.0, measurement_unit="U/mg",
            conditions={"pH": 7.0}, substrate_key="acetophenone",
            product_key="1-phenylethanol",
            reaction_direction="reverse_of_target")
        with self.assertRaises(ChemicalTaskMismatchError):
            variant_versus_parent(forward, reverse)


class CofactorStateTests(unittest.TestCase):
    """Two oxidation states are two conditions, not one."""

    DECLARED = [CofactorSpec(name="FMN", state=CofactorState.REDUCED,
                             recycling_system="GDH")]

    def group(self, state: str):
        return group_rows([well("P1", "A1", "candidate", 1.0, 1,
                                cofactor="FMN", state=state)])[0]

    def test_the_plate_state_is_not_overwritten_by_a_declared_option(self) -> None:
        spec = IngestResults._cofactor_for(self.group("oxidized"), self.DECLARED)
        self.assertIs(spec.state, CofactorState.OXIDIZED)

    def test_a_matching_state_still_inherits_the_declared_spec(self) -> None:
        """Matching on both keeps the recycling system attached."""
        spec = IngestResults._cofactor_for(self.group("reduced"), self.DECLARED)
        self.assertIs(spec.state, CofactorState.REDUCED)
        self.assertEqual(spec.recycling_system, "GDH")

    def test_an_unknown_state_is_not_promoted_to_reduced(self) -> None:
        spec = IngestResults._cofactor_for(self.group("nonsense"), self.DECLARED)
        self.assertIs(spec.state, CofactorState.UNKNOWN)


if __name__ == "__main__":
    unittest.main(verbosity=2)
