"""Regression tests for the follow-up audit's third batch.

Two findings, each reproduced before being fixed.

Assigning a nested field through ``resolve`` skipped validation, so a field
declared ``bool`` could hold the string ``"false"``. Every later reader asked
``bool(value)``, which is true for any non-empty string, and the approval gate
read "false" as approved.

The evidence matrix read each record's direction *label* and never checked it
against the record's own chemistry, although the verifier that does exactly
that already existed one module away. A record declaring the target direction
while describing the reverse reaction counted as confirmed support.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from eagent.datalayer.intake import direction_check
from eagent.schemas import (
    Detection, ExperimentRecord, OutcomeClass, ProductSpec, ReactionClass,
    ReactionDirection, ReactionSpec, SubstrateSpec, TaskSpec,
)
from eagent.tools.retrieve_evidence import (
    ChemotypeAssignment, EvidenceRow, SubstrateChemotype, build_evidence_matrix,
)

TARGET = ReactionSpec(reaction_class=ReactionClass.KETONE_TO_SECONDARY_ALCOHOL)
CHEMOTYPE = ChemotypeAssignment(chemotype=SubstrateChemotype.AROMATIC_KETONE,
                                basis="test fixture", key="k")


class ResolveValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = TaskSpec(task_id="T")

    def test_the_string_false_does_not_approve_a_gate(self) -> None:
        self.task.resolve("approval.synthesis_authorized", "false",
                          source="operator:test")
        self.assertIs(self.task.approval.synthesis_authorized, False)
        self.assertFalse(self.task.approval.state("synthesis_authorized"))

    def test_other_falsey_strings_do_not_approve_either(self) -> None:
        for text in ("False", "no", "0", "off"):
            task = TaskSpec(task_id="T")
            try:
                task.resolve("approval.synthesis_authorized", text,
                             source="operator:test")
            except ValidationError:
                continue          # refusing outright is also correct
            self.assertFalse(task.approval.state("synthesis_authorized"), text)

    def test_a_real_approval_still_opens_the_gate(self) -> None:
        self.task.resolve("approval.synthesis_authorized", True,
                          source="operator:test")
        self.assertTrue(self.task.approval.state("synthesis_authorized"))

    def test_a_non_boolean_in_the_field_is_not_truthy(self) -> None:
        """Defence behind the validation: state() demands a real True."""
        object.__setattr__(self.task.approval, "synthesis_authorized", "yes please")
        self.assertFalse(self.task.approval.state("synthesis_authorized"))

    def test_an_enum_field_is_coerced_not_left_as_a_string(self) -> None:
        self.task.resolve("reaction.reaction_class",
                          "ketone_to_secondary_alcohol", source="operator:test")
        self.assertIs(self.task.reaction.reaction_class,
                      ReactionClass.KETONE_TO_SECONDARY_ALCOHOL)
        self.assertEqual(self.task.reaction.reaction_class.value,
                         "ketone_to_secondary_alcohol")

    def test_a_value_the_field_cannot_hold_is_refused(self) -> None:
        with self.assertRaises(ValidationError):
            self.task.resolve("reaction.reaction_class", "not_a_reaction",
                              source="operator:test")
        with self.assertRaises(ValidationError):
            self.task.resolve("conditions.pH", "not a number",
                              source="operator:test")

    def test_a_coercible_value_is_coerced(self) -> None:
        self.task.resolve("conditions.pH", "7.4", source="operator:test")
        self.assertEqual(self.task.conditions.pH, 7.4)

    def test_the_ledger_records_what_was_stored(self) -> None:
        """An audit trail describing a different value is worse than none."""
        self.task.resolve("conditions.pH", "7.4", source="operator:test")
        self.assertEqual(self.task.assumptions[-1].value, 7.4)


def _row(direction: ReactionDirection, reaction_class: ReactionClass,
         outcome: OutcomeClass) -> EvidenceRow:
    record = ExperimentRecord(
        record_id=f"r-{direction.value}-{reaction_class.value}",
        sequence="MKAIVTGASRGIG",
        substrate=SubstrateSpec(name="1-phenylethanol",
                                isomeric_smiles="CC(O)c1ccccc1"),
        product_observed=(ProductSpec(name="acetophenone",
                                      isomeric_smiles="CC(=O)c1ccccc1")
                          if outcome is OutcomeClass.CONFIRMED_TARGET_PRODUCT
                          else None),
        reaction_class=reaction_class, reaction_direction=direction,
        outcome=outcome,
        detection=Detection(method="GC-MS", confirms_product_identity=True,
                            limit_of_detection=0.1))
    return EvidenceRow(record=record, connector="test", family="SDR",
                       family_basis="test fixture", chemotype=CHEMOTYPE,
                       strength_decisions=(), database_version=None,
                       retrieved_at=None)


def _cell(row: EvidenceRow):
    matrix = build_evidence_matrix([row], TARGET)
    return list(matrix.cells.values())[0]


class EvidenceDirectionTests(unittest.TestCase):
    def test_the_existing_verifier_sees_the_contradiction(self) -> None:
        """Guards the premise: the check existed and was simply not called."""
        row = _row(ReactionDirection.FORWARD_AS_TARGET,
                   ReactionClass.ALCOHOL_OXIDATION,
                   OutcomeClass.CONFIRMED_TARGET_PRODUCT)
        verdict = direction_check(row.record, TARGET)
        self.assertFalse(verdict.supports)
        self.assertTrue(verdict.is_reverse)

    def test_a_contradicting_label_is_not_counted_as_confirmed(self) -> None:
        cell = _cell(_row(ReactionDirection.FORWARD_AS_TARGET,
                          ReactionClass.ALCOHOL_OXIDATION,
                          OutcomeClass.CONFIRMED_TARGET_PRODUCT))
        self.assertEqual(cell.confirmed, 0)
        self.assertEqual(cell.confirmed_reverse_direction, 1)

    def test_a_genuine_forward_record_still_counts(self) -> None:
        cell = _cell(_row(ReactionDirection.FORWARD_AS_TARGET,
                          ReactionClass.KETONE_TO_SECONDARY_ALCOHOL,
                          OutcomeClass.CONFIRMED_TARGET_PRODUCT))
        self.assertEqual(cell.confirmed, 1)

    def test_an_unrecorded_direction_is_not_called_reverse(self) -> None:
        """Not knowing which way it ran differs from knowing it ran back."""
        cell = _cell(_row(ReactionDirection.UNSPECIFIED,
                          ReactionClass.KETONE_TO_SECONDARY_ALCOHOL,
                          OutcomeClass.CONFIRMED_TARGET_PRODUCT))
        self.assertEqual(cell.confirmed, 0)
        self.assertEqual(cell.confirmed_reverse_direction, 0)
        self.assertEqual(cell.confirmed_direction_unknown, 1)

    def test_a_reverse_negative_is_not_a_target_negative(self) -> None:
        """Failing to see the oxidation is not failing to see the reduction."""
        cell = _cell(_row(ReactionDirection.REVERSE_OF_TARGET,
                          ReactionClass.ALCOHOL_OXIDATION,
                          OutcomeClass.NO_TARGET_PRODUCT_DETECTED))
        self.assertEqual(cell.not_detected, 0)
        self.assertEqual(cell.not_detected_reverse_direction, 1)

    def test_a_forward_negative_still_counts_as_not_detected(self) -> None:
        cell = _cell(_row(ReactionDirection.FORWARD_AS_TARGET,
                          ReactionClass.KETONE_TO_SECONDARY_ALCOHOL,
                          OutcomeClass.NO_TARGET_PRODUCT_DETECTED))
        self.assertEqual(cell.not_detected, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
