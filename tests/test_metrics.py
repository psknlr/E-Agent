"""Tests for :mod:`eagent.eval.metrics`.

The fixture is one six-construct round whose answer is known by construction:
two constructs meet the registered 20% conversion bar with the target product,
one makes product below the bar, one makes the wrong enantiomer, one never
expressed, and one well was never run. So the registered hit count is two, the
rate over the six slots is 2/6 and the rate over the three constructs that
expressed is 2/3. Both are true, and a regression that starts reporting only
the flattering one fails here instead of redefining the endpoint.

The other half of the file is the guard: the criterion cannot be swapped,
mutated, re-registered under a different digest, or routed around by handing
the endpoint a different shape of the same question.

Runs under pytest, or standalone with ``python3 tests/test_metrics.py``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - standalone execution
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from eagent.datalayer.house_db import ConditionMismatchError, ExpressionStatus
from eagent.eval.metrics import (
    CriterionChangedError,
    EndpointMismatchError,
    HitRateReport,
    OutcomeRow,
    PreRegistration,
    Rate,
    configuration_compliance,
    criterion_from_mapping,
    hit_rates,
    precision_at_k,
    replicate_reproducibility,
    round_two_report,
    signed_ee_aggregate,
    variant_versus_parent,
)
from eagent.schemas.chem import CofactorSpec, CofactorState, SubstrateSpec
from eagent.schemas.reaction import Conditions
from eagent.schemas.record import Detection, ExperimentRecord, OutcomeClass
from eagent.schemas.templates import (
    AssayTemplate, TemplateProvenance, TemplateSourceType,
)

REGISTERED = {"min_conversion_pct": 20.0, "measurement_type": "conversion_pct"}
MOVED = {"min_conversion_pct": 10.0, "measurement_type": "conversion_pct"}


def registration(criteria=None, k_slots: int = 6) -> PreRegistration:
    return PreRegistration.from_plan(
        dict(criteria or REGISTERED), k_slots=k_slots,
        registered_by="operator:ada", registered_at="2026-01-01T00:00:00Z")


def row(candidate_id: str, outcome: OutcomeClass, **kwargs) -> OutcomeRow:
    kwargs.setdefault("expression_status", ExpressionStatus.SOLUBLE)
    return OutcomeRow(candidate_id, outcome, **kwargs)


def round_of_six() -> list[OutcomeRow]:
    """Six constructs whose verdicts are fixed by construction, not by the code."""
    return [
        row("c1", OutcomeClass.CONFIRMED_TARGET_PRODUCT, conversion_pct=55.0,
            target_peak_area=97.0, opposite_peak_area=3.0),
        row("c2", OutcomeClass.CONFIRMED_TARGET_PRODUCT, conversion_pct=31.0,
            target_peak_area=90.0, opposite_peak_area=10.0),
        row("c3", OutcomeClass.CONFIRMED_TARGET_PRODUCT, conversion_pct=4.0),
        row("c4", OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION,
            conversion_pct=61.0, target_peak_area=4.0, opposite_peak_area=96.0),
        row("c5", OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE,
            expression_status=ExpressionStatus.INSOLUBLE),
        row("c6", OutcomeClass.NOT_TESTED,
            expression_status=ExpressionStatus.NOT_ASSESSED),
    ]


# --------------------------------------------------------------------------
# rates
# --------------------------------------------------------------------------

class TestRate(unittest.TestCase):

    def test_an_empty_denominator_is_undefined_not_zero(self) -> None:
        rate = Rate("x", 0, 0, "constructs that expressed")
        self.assertIsNone(rate.point)
        self.assertIsNone(rate.interval)
        self.assertIn("undefined", rate.describe())

    def test_the_interval_brackets_the_point(self) -> None:
        rate = Rate("x", 2, 6, "slots")
        low, high = rate.interval
        self.assertLessEqual(low, rate.point)
        self.assertLessEqual(rate.point, high)
        self.assertLess(low, 1 / 3)
        self.assertGreater(high, 1 / 3)

    def test_impossible_counts_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            Rate("x", 7, 6, "slots")
        with self.assertRaises(ValueError):
            Rate("x", -1, 6, "slots")

    def test_a_rate_must_say_what_its_denominator_counts(self) -> None:
        with self.assertRaises(ValueError):
            Rate("x", 1, 2, "   ")


# --------------------------------------------------------------------------
# the pre-registration guard
# --------------------------------------------------------------------------

class TestPreRegistrationGuard(unittest.TestCase):

    def test_a_changed_criterion_raises_rather_than_rescoring(self) -> None:
        reg = registration()
        with self.assertRaises(CriterionChangedError):
            precision_at_k(round_of_six(), reg, criterion_in_use=MOVED)

    def test_the_refusal_names_what_moved(self) -> None:
        reg = registration()
        with self.assertRaises(CriterionChangedError) as caught:
            reg.assert_matches(MOVED)
        message = str(caught.exception)
        self.assertIn("min_conversion_pct", message)
        self.assertIn("20.0", message)
        self.assertIn("10.0", message)

    def test_every_shape_of_the_same_question_is_checked(self) -> None:
        """A mapping, a criterion object, a template and a digest all go through."""
        reg = registration()
        reg.assert_matches(REGISTERED)
        reg.assert_matches(criterion_from_mapping(REGISTERED))
        reg.assert_matches(reg.registered_digest)
        for moved in (MOVED, criterion_from_mapping(MOVED),
                      "0" * 64):
            with self.assertRaises(CriterionChangedError):
                reg.assert_matches(moved)

    def test_the_caller_s_own_mapping_cannot_change_the_registration(self) -> None:
        """Editing the dict a registration was built from must not move the bar."""
        criteria = dict(REGISTERED)
        reg = PreRegistration.from_plan(
            criteria, k_slots=6, registered_by="operator:ada",
            registered_at="2026-01-01T00:00:00Z")
        criteria["min_conversion_pct"] = 10.0
        reg.assert_unchanged()
        self.assertEqual(reg.criterion.min_conversion_pct, 20.0)
        self.assertEqual(precision_at_k(round_of_six(), reg).n_hits, 2)

    def test_editing_the_criterion_in_place_is_caught(self) -> None:
        """The other way round: mutate the registration itself and be refused."""
        reg = registration()
        reg.criterion.raw["min_conversion_pct"] = 10.0
        with self.assertRaises(CriterionChangedError):
            reg.assert_unchanged()
        with self.assertRaises(CriterionChangedError):
            precision_at_k(round_of_six(), reg)
        with self.assertRaises(CriterionChangedError):
            hit_rates(round_of_six(), reg)

    def test_a_plan_whose_recorded_digest_disagrees_is_refused(self) -> None:
        with self.assertRaises(CriterionChangedError):
            PreRegistration.from_plan(
                REGISTERED, k_slots=6, registered_by="operator:ada",
                recorded_digest="0" * 64)

    def test_an_empty_criterion_is_not_a_pre_registration(self) -> None:
        with self.assertRaises(CriterionChangedError):
            criterion_from_mapping({})

    def test_an_unimplemented_criterion_key_is_refused(self) -> None:
        """An ignored criterion key is a criterion that passes everything."""
        with self.assertRaises(CriterionChangedError):
            criterion_from_mapping({"min_converion_pct": 20.0})

    def test_a_registration_needs_an_actor_a_time_and_a_budget(self) -> None:
        for kwargs in ({"registered_by": "  "}, {"registered_at": ""},
                       {"k_slots": 0}):
            base = dict(k_slots=6, registered_by="operator:ada",
                        registered_at="2026-01-01T00:00:00Z")
            base.update(kwargs)
            with self.assertRaises(ValueError):
                PreRegistration.from_plan(REGISTERED, **base)

    def test_a_template_registration_uses_the_same_digest_as_the_plan(self) -> None:
        template = AssayTemplate(
            template_id="assay.test.v1", tier=1, method="a plate reader",
            positive_criteria=dict(REGISTERED),
            provenance=TemplateProvenance(
                source_type=TemplateSourceType.CURATED_DATABASE,
                identifiers=["eagent:test"]),
        )
        from_template = PreRegistration.from_template(
            template, k_slots=6, registered_by="operator:ada",
            registered_at="2026-01-01T00:00:00Z")
        self.assertEqual(from_template.registered_digest,
                         registration().registered_digest)

    def test_there_is_no_argument_that_supplies_hit_labels(self) -> None:
        """The hit definition is applied here, or the number is not produced."""
        import inspect
        signature = inspect.signature(precision_at_k)
        for name in signature.parameters:
            self.assertNotIn("hit", name)
            self.assertNotIn("label", name)


# --------------------------------------------------------------------------
# the primary endpoint
# --------------------------------------------------------------------------

class TestPrecisionAtK(unittest.TestCase):

    def test_the_hand_built_round_scores_exactly_as_constructed(self) -> None:
        result = precision_at_k(round_of_six(), registration())
        self.assertEqual(result.n_hits, 2)                 # c1 and c2
        self.assertEqual(result.slots_spent, 6)
        self.assertEqual(result.k_registered, 6)
        self.assertEqual(result.n_undecidable, 1)          # c6 was never run
        self.assertEqual(result.n_expression_failures, 1)  # c5
        self.assertAlmostEqual(result.rate_over_slots_spent.point, 2 / 6)
        self.assertAlmostEqual(result.rate_over_registered_k.point, 2 / 6)
        self.assertFalse(result.denominators_differ)

    def test_a_short_round_reports_both_denominators(self) -> None:
        rows = round_of_six()[:4]
        result = precision_at_k(rows, registration())
        self.assertEqual(result.slots_spent, 4)
        self.assertAlmostEqual(result.rate_over_slots_spent.point, 2 / 4)
        self.assertAlmostEqual(result.rate_over_registered_k.point, 2 / 6)
        self.assertTrue(result.denominators_differ)
        self.assertIn("slots_spent", result.to_dict())

    def test_a_control_is_not_a_new_candidate(self) -> None:
        rows = round_of_six() + [
            OutcomeRow("positive_control", OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                       conversion_pct=95.0, is_control=True,
                       expression_status=ExpressionStatus.SOLUBLE)]
        result = precision_at_k(rows, registration())
        self.assertEqual(result.n_hits, 2)
        self.assertEqual(result.n_controls_excluded, 1)
        self.assertEqual(result.slots_spent, 6)

    def test_a_known_candidate_stays_in_the_denominator(self) -> None:
        """Stacking the batch with sure things must not raise the endpoint."""
        result = precision_at_k(round_of_six(), registration(),
                                known_candidate_ids=["c1"])
        self.assertEqual(result.n_hits, 1)
        self.assertEqual(result.n_known_excluded_from_numerator, 1)
        self.assertEqual(result.slots_spent, 6)

    def test_a_wrong_configuration_product_is_not_a_hit(self) -> None:
        verdicts = {v.candidate_id: v
                    for v in precision_at_k(round_of_six(),
                                            registration()).verdicts}
        self.assertFalse(verdicts["c4"].is_hit)
        self.assertFalse(verdicts["c4"].undecidable)

    def test_an_untested_well_is_undecidable_not_a_negative(self) -> None:
        verdicts = {v.candidate_id: v
                    for v in precision_at_k(round_of_six(),
                                            registration()).verdicts}
        self.assertTrue(verdicts["c6"].undecidable)
        self.assertFalse(verdicts["c5"].undecidable)   # expression failure is a miss

    def test_more_constructs_than_the_registered_budget_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            precision_at_k(round_of_six(), registration(k_slots=4))


# --------------------------------------------------------------------------
# hit rate, both denominators
# --------------------------------------------------------------------------

class TestHitRates(unittest.TestCase):

    def test_both_denominators_differ_when_a_construct_fails_to_express(self) -> None:
        report = hit_rates(round_of_six(), registration())
        self.assertEqual(report.n_submitted, 6)
        self.assertEqual(report.n_expressed, 4)      # c1-c4 expressed
        self.assertEqual(report.n_expression_failed, 1)
        self.assertEqual(report.n_expression_unknown, 1)
        self.assertEqual(report.n_hits, 2)
        self.assertTrue(report.denominators_differ)
        self.assertAlmostEqual(report.rate_all_submitted.point, 2 / 6)
        self.assertAlmostEqual(report.rate_expressed_only.point, 2 / 4)

    def test_one_line_always_carries_both(self) -> None:
        text = hit_rates(round_of_six(), registration()).describe()
        self.assertIn("all_submitted", text)
        self.assertIn("expressed_only", text)

    def test_there_is_no_single_hit_rate_attribute_to_quote(self) -> None:
        report = hit_rates(round_of_six(), registration())
        for forbidden in ("hit_rate", "rate", "value", "score"):
            self.assertFalse(hasattr(report, forbidden),
                             f"HitRateReport.{forbidden} would be quoted alone")
        self.assertNotIn("rate", {f for f in HitRateReport.__dataclass_fields__})

    def test_the_expressed_rate_is_undefined_when_nothing_expressed(self) -> None:
        rows = [row("c1", OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE,
                    expression_status=ExpressionStatus.INSOLUBLE)]
        report = hit_rates(rows, registration())
        self.assertIsNone(report.rate_expressed_only.point)

    def test_the_hit_count_matches_the_primary_endpoint(self) -> None:
        rows = round_of_six()
        self.assertEqual(hit_rates(rows, registration()).n_hits,
                         precision_at_k(rows, registration()).n_hits)


# --------------------------------------------------------------------------
# configuration and ee
# --------------------------------------------------------------------------

class TestConfiguration(unittest.TestCase):

    def test_the_wrong_enantiomer_fails_the_configuration_requirement(self) -> None:
        result = configuration_compliance(round_of_six(), registration())
        self.assertTrue(result.applicable)
        self.assertEqual(result.rate.trials, 3)        # c1, c2, c4 carry areas
        self.assertEqual(result.rate.successes, 2)
        self.assertEqual(result.n_wrong_configuration, 1)
        self.assertEqual(result.n_unassessable, 3)

    def test_an_unmeasured_configuration_is_not_a_failure(self) -> None:
        result = configuration_compliance(round_of_six(), registration())
        self.assertTrue(any("c3" in reason
                            for reason in result.reasons_unassessable))

    def test_a_pre_registered_ee_bar_is_used_when_there_is_one(self) -> None:
        reg = registration({"min_conversion_pct": 20.0,
                            "min_ee_target_pct": 90.0})
        result = configuration_compliance(round_of_six(), reg)
        self.assertTrue(result.bar_is_pre_registered)
        self.assertEqual(result.bar_pct, 90.0)
        self.assertEqual(result.rate.successes, 1)      # only c1 reaches +94%

    def test_without_a_registered_bar_the_claim_is_only_a_direction(self) -> None:
        result = configuration_compliance(round_of_six(), registration())
        self.assertFalse(result.bar_is_pre_registered)
        self.assertIn("direction", result.describe())

    def test_a_reaction_with_no_stereocentre_reports_inapplicable(self) -> None:
        result = configuration_compliance(round_of_six(), registration(),
                                          stereo_task=False)
        self.assertFalse(result.applicable)
        self.assertIn("no new stereocentre", result.note)

    def test_a_criterion_demanding_a_validated_chiral_method_excludes_rows(self) -> None:
        reg = registration({"min_conversion_pct": 20.0,
                            "requires_chiral_method_validated": True})
        result = configuration_compliance(round_of_six(), reg)
        self.assertEqual(result.rate.trials, 0)
        self.assertEqual(result.n_unassessable, 6)


class TestSignedEE(unittest.TestCase):

    def test_ee_is_derived_from_areas_through_the_schema_definition(self) -> None:
        single = OutcomeRow("c", OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                            target_peak_area=97.0, opposite_peak_area=3.0)
        self.assertAlmostEqual(single.ee_target_pct, 94.0)

    def test_areas_that_contradict_a_stated_ee_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            OutcomeRow("c", OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                       ee_target_pct=94.0,
                       target_peak_area=50.0, opposite_peak_area=50.0)

    def test_the_aggregate_keeps_the_sign(self) -> None:
        aggregate = signed_ee_aggregate(round_of_six())
        self.assertEqual(aggregate.n_with_signed_ee, 3)
        self.assertEqual(aggregate.n_wrong_configuration, 1)   # c4 at -92%
        self.assertAlmostEqual(aggregate.min_signed_ee_pct, -92.0)
        self.assertAlmostEqual(aggregate.max_signed_ee_pct, 94.0)
        # Pooled over all areas: (97+90+4) - (3+10+96) = 82 of 300.
        self.assertAlmostEqual(aggregate.pooled_ee_pct, 82 / 300 * 100)

    def test_there_is_no_mean_of_absolute_ee(self) -> None:
        aggregate = signed_ee_aggregate(round_of_six())
        for forbidden in ("mean_ee_pct", "mean_absolute_ee_pct", "average_ee"):
            self.assertFalse(hasattr(aggregate, forbidden))

    def test_a_round_with_no_ee_says_so_rather_than_reporting_zero(self) -> None:
        aggregate = signed_ee_aggregate([row("c", OutcomeClass.NOT_TESTED)])
        self.assertIsNone(aggregate.median_signed_ee_pct)
        self.assertIsNone(aggregate.pooled_ee_pct)


# --------------------------------------------------------------------------
# round two
# --------------------------------------------------------------------------

def conditions(ph: float = 7.0) -> dict:
    return {"pH": ph, "temperature_C": 30.0, "buffer": "KPi",
            "solvent_system": "aqueous", "cosolvent_fraction": 0.05,
            "substrate_concentration_mM": 10.0, "enzyme_loading": "2 g/L",
            "reaction_time_h": 24.0, "expression_host": "E. coli BL21(DE3)"}


def pair_row(candidate_id: str, value: float, *, ph: float = 7.0,
             replicates=(), ee: float | None = None,
             measurement_type: str = "conversion_pct") -> OutcomeRow:
    return OutcomeRow(
        candidate_id, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
        expression_status=ExpressionStatus.SOLUBLE,
        conversion_pct=value, measurement_value=value,
        measurement_type=measurement_type, measurement_unit="%",
        ee_target_pct=ee, conditions=conditions(ph),
        cofactor_species="NADPH", cofactor_state="reduced",
        replicate_values=tuple(replicates), mutations=("Y155F",))


class TestRoundTwo(unittest.TestCase):

    def test_a_like_for_like_pair_is_compared(self) -> None:
        comparison = variant_versus_parent(
            pair_row("v1", 60.0, replicates=(58.0, 60.0, 62.0)),
            pair_row("p1", 40.0, replicates=(39.0, 40.0, 41.0)))
        self.assertAlmostEqual(comparison.delta_measurement, 20.0)
        self.assertAlmostEqual(comparison.delta_conversion_pct, 20.0)
        self.assertIsNone(comparison.delta_ee_pct)
        self.assertTrue(comparison.replicate_check.reproducible)

    def test_a_cross_condition_comparison_is_refused(self) -> None:
        with self.assertRaises(ConditionMismatchError) as caught:
            variant_versus_parent(pair_row("v1", 60.0, ph=8.0),
                                  pair_row("p1", 40.0, ph=7.0))
        self.assertIn("pH", str(caught.exception))

    def test_a_cofactor_change_is_also_a_condition_change(self) -> None:
        variant = OutcomeRow("v1", OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                             conversion_pct=60.0, conditions=conditions(),
                             cofactor_species="NADH", cofactor_state="reduced")
        parent = OutcomeRow("p1", OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                            conversion_pct=40.0, conditions=conditions(),
                            cofactor_species="NADPH", cofactor_state="reduced")
        with self.assertRaises(ConditionMismatchError) as caught:
            variant_versus_parent(variant, parent)
        self.assertIn("cofactor_species", str(caught.exception))

    def test_two_endpoints_are_not_subtracted(self) -> None:
        with self.assertRaises(EndpointMismatchError):
            variant_versus_parent(
                pair_row("v1", 60.0, measurement_type="initial_rate"),
                pair_row("p1", 40.0, measurement_type="conversion_pct"))

    def test_a_difference_inside_the_replicate_spread_is_not_reproducible(self) -> None:
        comparison = variant_versus_parent(
            pair_row("v1", 42.0, replicates=(30.0, 42.0, 54.0)),
            pair_row("p1", 40.0, replicates=(28.0, 40.0, 52.0)))
        self.assertFalse(comparison.replicate_check.reproducible)

    def test_one_replicate_cannot_decide_and_says_so(self) -> None:
        check = replicate_reproducibility([40.0], [60.0], 20.0)
        self.assertIsNone(check.reproducible)
        self.assertIn("not checkable", check.basis)

    def test_refusals_are_returned_rather_than_dropped(self) -> None:
        report = round_two_report([
            (pair_row("v1", 60.0, replicates=(59.0, 60.0, 61.0)),
             pair_row("p1", 40.0, replicates=(39.0, 40.0, 41.0))),
            (pair_row("v2", 70.0, ph=8.0), pair_row("p1", 40.0)),
        ])
        self.assertEqual(len(report.comparisons), 1)
        self.assertEqual(len(report.refusals), 1)
        self.assertEqual(report.n_pairs, 2)
        self.assertIn("pH", report.refusals[0].reason)
        self.assertEqual(report.n_improved_on_measurement, 1)

    def test_an_improvement_inside_the_noise_is_not_counted_as_one(self) -> None:
        report = round_two_report([
            (pair_row("v1", 41.0, replicates=(30.0, 41.0, 52.0)),
             pair_row("p1", 40.0, replicates=(29.0, 40.0, 51.0))),
        ])
        self.assertEqual(report.n_improved_on_measurement, 0)

    def test_there_is_no_combined_improvement_number(self) -> None:
        comparison = variant_versus_parent(pair_row("v1", 60.0),
                                           pair_row("p1", 40.0))
        for forbidden in ("total", "score", "improvement", "combined"):
            self.assertFalse(hasattr(comparison, forbidden))


# --------------------------------------------------------------------------
# conversion from the canonical record type
# --------------------------------------------------------------------------

class TestFromRecord(unittest.TestCase):

    def test_a_record_carries_its_conditions_into_the_row(self) -> None:
        rec = ExperimentRecord(
            record_id="r1",
            sequence="MKAVVLSGFGGLDNVKLEEVPKPTPGPGQVLVKVEAAGVCHSDLHLIDGDLP",
            substrate=SubstrateSpec(isomeric_smiles="CC(=O)c1ccccc1"),
            outcome=OutcomeClass.CONFIRMED_TARGET_PRODUCT,
            detection=Detection(method="chiral GC-MS",
                                confirms_product_identity=True,
                                chiral_method_validated=True,
                                authentic_standard=True),
            conversion_pct=45.0,
            ee_target_pct=92.0,
            cofactor=CofactorSpec(name="NADPH", state=CofactorState.REDUCED),
            conditions=Conditions(pH=7.0, temperature_C=30.0, buffer="KPi"),
        )
        row_ = OutcomeRow.from_record(rec)
        self.assertEqual(row_.candidate_id, "r1")
        self.assertEqual(row_.conversion_pct, 45.0)
        self.assertTrue(row_.chiral_method_validated)
        self.assertEqual(row_.cofactor_species, "NADPH")
        self.assertTrue(row_.expressed)
        self.assertEqual(row_.conditions["pH"], 7.0)

    def test_an_unassessed_negative_keeps_expression_unknown(self) -> None:
        rec = ExperimentRecord(
            record_id="r2",
            sequence="MKAVVLSGFGGLDNVKLEEVPKPTPGPGQVLVKVEAAGVCHSDLHLIDGDLA",
            outcome=OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
            detection=Detection(method="chiral GC-MS", limit_of_detection=0.5,
                                limit_unit="uM", confirms_product_identity=True),
        )
        self.assertIsNone(OutcomeRow.from_record(rec).expressed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
