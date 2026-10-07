"""Regression tests: what a plate has to agree about before it is one number.

Five findings from an external review, each reproduced before being fixed.
They are one failure wearing five hats: an aggregate was taken over wells
that were not measurements of the same thing, and the result reads like a
measurement afterwards.

* a well that never says it was run still voted;
* an untested well's leftover ``expressed_soluble=yes`` overruled the tested
  well that failed, turning an expression failure into a catalytic negative;
* ``0.01 mM`` and ``10 uM`` -- one concentration reported twice -- were
  averaged into ``5.005``;
* two constructs of one candidate were averaged into a performance neither
  construct had;
* a control well nobody ran set the denominator of every fold on its plate.

Labels are what the next model trains on, so each of these propagates.
"""

from __future__ import annotations

import unittest

from eagent.science.units import (
    Reconciliation, canonical_unit, convert_measurement, reconcile_units,
)
from eagent.schemas import OutcomeClass
from eagent.schemas.templates import (
    AssayTemplate, TemplateProvenance, TemplateSourceType,
)
from eagent.tools.ingest_results import (
    AssayRow, MeasurementGroup, PositiveCriterion, attach_empty_vector_baselines,
    build_record_id, classify_group, group_rows,
)

CRITERION = PositiveCriterion.from_template(AssayTemplate(
    template_id="assay:test", tier=2, method="chiral GC-MS",
    confirms_product_identity=True, chiral_capable=True,
    positive_criteria={"min_conversion_pct": 20.0},
    limit_of_detection=0.1, limit_unit="uM",
    provenance=TemplateProvenance(
        source_type=TemplateSourceType.CURATED_DATABASE,
        identifiers=["SOP-TEST"])))


def well(**kw) -> AssayRow:
    base = dict(plan_id="p", slot=1, plate="P1", well="A1", candidate_id="c1",
                construct_id="c1", kind="candidate", role="",
                parent_candidate_id="", mutations=(), cofactor="NADPH",
                cofactor_state="reduced", replicate=1, tested=True,
                expressed_soluble=True, detection_method="GC-MS",
                confirms_product_identity=True, authentic_standard=True,
                chiral_method_validated=True, limit_of_detection=0.1,
                limit_unit="uM", measurement_type="specific_activity",
                measurement_value=1.0, measurement_unit="U/mg",
                conversion_pct=None, product_identity_observed="target",
                peak_area_target=None, peak_area_opposite=None, notes="")
    base.update(kw)
    return AssayRow(**base)


def one_group(rows) -> MeasurementGroup:
    groups = group_rows(rows)
    assert len(groups) == 1, f"expected one group, got {len(groups)}"
    return groups[0]


class UntestedWellTests(unittest.TestCase):
    """Only a well that says it was run contributes a fact."""

    def test_a_blank_tested_cell_is_not_a_measurement(self) -> None:
        group = one_group([well(well="A1", tested=None, conversion_pct=90.0)])
        self.assertEqual(group.tested_rows, [])
        self.assertIsNone(group.conversion_pct)

    def test_a_blank_tested_cell_is_named_not_hidden(self) -> None:
        group = one_group([well(well="A1", tested=None, conversion_pct=90.0)])
        self.assertEqual(len(group.rows_without_tested_flag), 1)
        self.assertIn("do not say whether they were run", group.disagreement)

    def test_a_group_of_untested_wells_is_not_tested_not_negative(self) -> None:
        group = one_group([well(well="A1", tested=None, conversion_pct=0.0)])
        verdict = classify_group(group, CRITERION)
        self.assertIs(verdict.outcome, OutcomeClass.NOT_TESTED)
        self.assertIn("blank", verdict.reasons[0])

    def test_an_untested_well_does_not_outvote_the_tested_one(self) -> None:
        """The commonest plate is last round's sheet with stale numbers in it."""
        group = one_group([well(well="A1", tested=True, conversion_pct=5.0),
                           well(well="A2", tested=False, conversion_pct=95.0),
                           well(well="A3", tested=False, conversion_pct=95.0)])
        self.assertEqual(group.conversion_pct, 5.0)

    def test_a_tested_well_still_counts_normally(self) -> None:
        group = one_group([well(well="A1", conversion_pct=30.0),
                           well(well="A2", conversion_pct=40.0)])
        self.assertEqual(group.conversion_pct, 35.0)

    def test_the_plan_s_identification_claim_needs_a_run_well(self) -> None:
        """confirms_product_identity is pre-filled from the plan, not observed."""
        group = one_group([
            well(well="A1", tested=True, confirms_product_identity=False,
                 detection_method="A340"),
            well(well="A2", tested=False, confirms_product_identity=True,
                 detection_method="A340")])
        self.assertFalse(group.confirms_product_identity)


class ExpressionTests(unittest.TestCase):
    """An expression failure must not be promoted by a well nobody ran."""

    def setUp(self) -> None:
        self.group = one_group([
            well(well="A1", tested=True, expressed_soluble=False),
            well(well="A2", tested=False, expressed_soluble=True)])

    def test_the_tested_failure_stands(self) -> None:
        self.assertIs(self.group.expressed_soluble, False)

    def test_it_is_classified_as_an_expression_failure(self) -> None:
        verdict = classify_group(self.group, CRITERION)
        self.assertIs(verdict.outcome,
                      OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE)

    def test_an_expression_failure_is_not_a_catalytic_negative(self) -> None:
        verdict = classify_group(self.group, CRITERION)
        assert verdict.outcome is not None
        self.assertFalse(verdict.outcome.informs_catalytic_ability)

    def test_one_soluble_prep_among_tested_wells_is_still_enough(self) -> None:
        group = one_group([well(well="A1", expressed_soluble=False),
                           well(well="A2", expressed_soluble=True)])
        self.assertIs(group.expressed_soluble, True)

    def test_unreported_expression_stays_unknown(self) -> None:
        group = one_group([well(well="A1", expressed_soluble=None)])
        self.assertIsNone(group.expressed_soluble)


class UnitReconciliationTests(unittest.TestCase):
    """Two spellings of one concentration are one measurement."""

    def group(self, *pairs) -> MeasurementGroup:
        return one_group([
            well(well=f"A{n}", measurement_value=v, measurement_unit=u,
                 measurement_type="km")
            for n, (v, u) in enumerate(pairs, start=1)])

    def test_the_same_concentration_twice_aggregates_to_itself(self) -> None:
        group = self.group((0.01, "mM"), (10.0, "uM"))
        self.assertAlmostEqual(group.measurement_value, 0.01)
        self.assertEqual(group.measurement_unit, "mM")

    def test_the_unit_reported_is_the_unit_aggregated_in(self) -> None:
        group = self.group((0.01, "mM"), (10.0, "uM"))
        self.assertTrue(group.measurement_reconciliation.converted)

    def test_one_spelling_is_left_untouched(self) -> None:
        group = self.group((10.0, "AU/min"), (12.0, "AU/min"))
        self.assertEqual(group.measurement_value, 11.0)
        self.assertEqual(group.measurement_unit, "AU/min")
        self.assertFalse(group.measurement_reconciliation.converted)

    def test_an_unconvertible_pair_produces_no_number(self) -> None:
        group = self.group((1.0, "U/mg"), (5.0, "widgets"))
        self.assertIsNone(group.measurement_value)
        self.assertIn("conversion table", group.measurement_unit_conflict)

    def test_two_different_quantities_produce_no_number(self) -> None:
        group = self.group((1.0, "U/mg"), (50.0, "%"))
        self.assertIsNone(group.measurement_value)
        self.assertIn("different quantities", group.measurement_unit_conflict)

    def test_a_conflict_is_unresolved_rather_than_undecided_silently(self) -> None:
        group = self.group((1.0, "U/mg"), (5.0, "widgets"))
        verdict = classify_group(group, CRITERION)
        self.assertIsNone(verdict.outcome)
        assert verdict.unresolved is not None
        self.assertEqual(verdict.unresolved.reason_code,
                         "measurement_unit_conflict")
        self.assertEqual(sorted(verdict.unresolved.evidence["units"]),
                         ["U/mg", "widgets"])

    def test_detection_limits_are_reconciled_too(self) -> None:
        """'The weakest limit binds' is arithmetic only on one scale."""
        group = one_group([
            well(well="A1", limit_of_detection=0.5, limit_unit="uM"),
            well(well="A2", limit_of_detection=0.1, limit_unit="mM")])
        self.assertAlmostEqual(group.limit_of_detection, 0.1)
        self.assertEqual(group.limit_unit, "mM")

    def test_irreconcilable_limits_produce_no_limit(self) -> None:
        group = one_group([
            well(well="A1", limit_of_detection=0.5, limit_unit="uM"),
            well(well="A2", limit_of_detection=0.1, limit_unit="furlongs")])
        self.assertIsNone(group.limit_of_detection)


class ConstructIdentityTests(unittest.TestCase):
    """One candidate id covers several constructs; they are not replicates."""

    def setUp(self) -> None:
        self.groups = group_rows([
            well(well="A1", construct_id="pET28-c1", conversion_pct=90.0),
            well(well="A2", construct_id="pET22-c1", conversion_pct=10.0)])

    def test_two_constructs_make_two_groups(self) -> None:
        self.assertEqual(len(self.groups), 2)

    def test_neither_group_reports_the_average(self) -> None:
        self.assertEqual(sorted(g.conversion_pct for g in self.groups),
                         [10.0, 90.0])

    def test_each_group_names_its_construct(self) -> None:
        self.assertEqual(sorted(g.construct_id for g in self.groups),
                         ["pET22-c1", "pET28-c1"])

    def test_two_detection_methods_are_two_measurements(self) -> None:
        groups = group_rows([well(well="A1", detection_method="GC-MS"),
                             well(well="A2", detection_method="A340")])
        self.assertEqual(len(groups), 2)

    def test_two_endpoints_are_two_measurements(self) -> None:
        groups = group_rows([well(well="A1", measurement_type="kcat"),
                             well(well="A2", measurement_type="initial_rate")])
        self.assertEqual(len(groups), 2)

    def test_spelling_alone_does_not_split_a_group(self) -> None:
        groups = group_rows([well(well="A1", detection_method="GC-MS"),
                             well(well="A2", detection_method=" gc-ms ")])
        self.assertEqual(len(groups), 1)

    def test_the_record_id_separates_what_the_group_key_separates(self) -> None:
        ids = {build_record_id("RUN", g) for g in self.groups}
        self.assertEqual(len(ids), 2)

    def test_the_record_id_survives_a_separator_in_an_id(self) -> None:
        from eagent.tools.ingest_results import _split_record_id
        group = MeasurementGroup(candidate_id="c|1", construct_id="pET|28",
                                 cofactor="NADPH", cofactor_state="reduced")
        fields = _split_record_id(build_record_id("RUN", group))
        self.assertIsNotNone(fields)
        assert fields is not None
        self.assertEqual(fields[1], "c|1")
        self.assertEqual(fields[2], "pET|28")


class ControlWellTests(unittest.TestCase):
    """A background taken from a well nobody ran scales every fold."""

    def test_an_untested_control_sets_no_background(self) -> None:
        rows = [well(well="A1", measurement_value=10.0),
                well(well="H1", kind="empty_vector", measurement_value=5.0,
                     tested=False)]
        group = one_group([r for r in rows if r.kind == "candidate"])
        attach_empty_vector_baselines([group], rows)
        self.assertEqual(group.empty_vector_by_plate, {})
        self.assertIsNone(group.fold_over_empty_vector)

    def test_the_refusal_is_reported(self) -> None:
        rows = [well(well="A1", measurement_value=10.0),
                well(well="H1", kind="empty_vector", measurement_value=5.0,
                     tested=None)]
        group = one_group([r for r in rows if r.kind == "candidate"])
        notes = attach_empty_vector_baselines([group], rows)
        self.assertTrue([n for n in notes if "do not state that they were run" in n])

    def test_a_tested_control_still_sets_the_background(self) -> None:
        rows = [well(well="A1", measurement_value=10.0),
                well(well="H1", kind="empty_vector", measurement_value=5.0)]
        group = one_group([r for r in rows if r.kind == "candidate"])
        attach_empty_vector_baselines([group], rows)
        self.assertAlmostEqual(group.fold_over_empty_vector, 2.0)

    def test_a_control_read_on_another_method_is_not_this_plate_s_background(self) -> None:
        rows = [well(well="A1", measurement_value=10.0, detection_method="GC-MS"),
                well(well="H1", kind="empty_vector", measurement_value=5.0,
                     detection_method="A340")]
        group = one_group([r for r in rows if r.kind == "candidate"])
        attach_empty_vector_baselines([group], rows)
        self.assertEqual(group.empty_vector_by_plate, {})

    def test_a_control_on_another_scale_is_converted_not_divided_raw(self) -> None:
        rows = [well(well="A1", measurement_value=10.0, measurement_unit="mM",
                     measurement_type="km"),
                well(well="H1", kind="empty_vector", measurement_value=5000.0,
                     measurement_unit="uM", measurement_type="km")]
        group = one_group([r for r in rows if r.kind == "candidate"])
        attach_empty_vector_baselines([group], rows)
        self.assertAlmostEqual(group.fold_over_empty_vector, 2.0)

    def test_an_unconvertible_control_refuses_rather_than_divides(self) -> None:
        rows = [well(well="A1", measurement_value=10.0, measurement_unit="U/mg"),
                well(well="H1", kind="empty_vector", measurement_value=5.0,
                     measurement_unit="widgets")]
        group = one_group([r for r in rows if r.kind == "candidate"])
        attach_empty_vector_baselines([group], rows)
        self.assertIsNone(group.fold_over_empty_vector)
        self.assertTrue(group.fold_conflicts())


class ControlRecognitionTests(unittest.TestCase):
    """The pipeline's own spelling of a control has to be recognised.

    ``select_batch`` writes ``kind="control"`` and ``role="empty_vector
    control"``. Matching the bare token ``empty_vector`` against those two
    fields recognised neither, so on every plate the pipeline itself produced
    there was no background and every fold-over-background bar was
    undecidable -- while hand-built rows in the tests, which set
    ``kind="empty_vector"``, matched.
    """

    def baseline(self, kind: str, role: str):
        rows = [well(well="A1", measurement_value=10.0),
                well(well="H1", kind=kind, role=role, measurement_value=5.0)]
        group = one_group([r for r in rows if r.kind == "candidate"])
        attach_empty_vector_baselines([group], rows)
        return group.fold_over_empty_vector

    def test_the_role_select_batch_writes_is_recognised(self) -> None:
        self.assertAlmostEqual(
            self.baseline("control", "empty_vector control"), 2.0)

    def test_spelling_variants_are_recognised(self) -> None:
        for role in ("empty-vector control", "Empty Vector Control",
                     "vector_only", "no_insert control"):
            self.assertAlmostEqual(self.baseline("control", role), 2.0, msg=role)

    def test_another_control_is_not_the_host_background(self) -> None:
        """A no-enzyme well is abiotic background, not host background."""
        self.assertIsNone(self.baseline("control", "no_enzyme control"))

    def test_a_candidate_is_never_its_own_background(self) -> None:
        self.assertIsNone(self.baseline("candidate", "mined candidate"))


class UnitTableTests(unittest.TestCase):
    """The shared table, used by ingest and by evaluation alike."""

    def test_the_table_is_self_consistent(self) -> None:
        from eagent.science.units import UNIT_TABLE
        for spelling, (canon, _) in UNIT_TABLE.items():
            self.assertIsNotNone(canonical_unit(canon),
                                 f"{spelling} canonicalises to {canon!r}, "
                                 f"which is not itself in the table")

    def test_every_canonical_unit_is_its_own_identity(self) -> None:
        from eagent.science.units import UNIT_TABLE
        for canon, _ in UNIT_TABLE.values():
            self.assertAlmostEqual(canonical_unit(canon)[1], 1.0, places=12)

    def test_specific_activity_factors_are_definitional(self) -> None:
        self.assertEqual(convert_measurement(1000.0, "nmol/min/mg"),
                         (1.0, "U/mg"))
        self.assertEqual(convert_measurement(1.0, "umol/min/mg"), (1.0, "U/mg"))

    def test_concentration_and_specific_activity_do_not_mix(self) -> None:
        self.assertNotEqual(canonical_unit("mM")[0], canonical_unit("U/mg")[0])

    def test_no_values_keeps_the_declared_unit(self) -> None:
        rec = reconcile_units([(None, "U/mg")])
        self.assertEqual(rec, Reconciliation(values=(), unit="U/mg"))

    def test_an_empty_input_is_usable_and_empty(self) -> None:
        rec = reconcile_units([])
        self.assertTrue(rec.usable)
        self.assertEqual(rec.values, ())


if __name__ == "__main__":
    unittest.main(verbosity=2)
