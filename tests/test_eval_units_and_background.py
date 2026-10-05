"""Regression tests: two ways the evaluation layer reported the wrong thing.

Both from an external audit, reproduced before being fixed.

A parent measured at one per second and a variant at sixty per minute are the
same activity. The comparison subtracted the raw numbers and reported a
fifty-nine-fold gain, and because the replicate spreads were in the same
unconverted units the "larger than the replicate scatter" check agreed with
it. An engineering campaign would have called that a success.

Separately, a perfectly legal pre-registered plan naming a
fold-over-background bar crashed the primary endpoint outright, because the
criterion object and the row it scores were defined in two modules that
disagreed about which fields a row has.
"""

from __future__ import annotations

import unittest

from eagent.eval.metrics import (
    OutcomeRow, PreRegistration, UnitMismatchError, canonical_unit,
    convert_measurement, precision_at_k, variant_versus_parent,
)
from eagent.schemas import OutcomeClass


def row(cid: str, **kw) -> OutcomeRow:
    base = dict(candidate_id=cid,
                outcome=OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                soluble_expression=True, measurement_type="kcat",
                conditions={"pH": 7.0})
    base.update(kw)
    return OutcomeRow(**base)


class UnitConversionTests(unittest.TestCase):
    def test_equivalent_rates_in_different_units_differ_by_zero(self) -> None:
        comparison = variant_versus_parent(
            row("v", measurement_value=60.0, measurement_unit="min-1",
                replicate_values=(59.0, 60.0, 61.0)),
            row("p", measurement_value=1.0, measurement_unit="s-1",
                replicate_values=(0.98, 1.0, 1.02)))
        self.assertAlmostEqual(comparison.delta_measurement, 0.0, places=9)

    def test_an_equivalent_pair_is_not_called_reproducible(self) -> None:
        """The check must not confirm an improvement that is a unit artefact."""
        comparison = variant_versus_parent(
            row("v", measurement_value=60.0, measurement_unit="min-1",
                replicate_values=(59.0, 60.0, 61.0)),
            row("p", measurement_value=1.0, measurement_unit="s-1",
                replicate_values=(0.98, 1.0, 1.02)))
        self.assertFalse(comparison.replicate_check.reproducible)

    def test_a_real_doubling_is_still_detected(self) -> None:
        comparison = variant_versus_parent(
            row("v", measurement_value=120.0, measurement_unit="min-1",
                replicate_values=(119.0, 120.0, 121.0)),
            row("p", measurement_value=1.0, measurement_unit="s-1",
                replicate_values=(0.98, 1.0, 1.02)))
        self.assertAlmostEqual(comparison.delta_measurement, 1.0, places=9)
        self.assertTrue(comparison.replicate_check.reproducible)

    def test_the_comparison_reports_the_unit_it_actually_used(self) -> None:
        comparison = variant_versus_parent(
            row("v", measurement_value=60.0, measurement_unit="min-1"),
            row("p", measurement_value=1.0, measurement_unit="s-1"))
        self.assertEqual(comparison.measurement_unit, "s-1")

    def test_an_unconvertible_unit_pair_is_refused(self) -> None:
        with self.assertRaises(UnitMismatchError) as ctx:
            variant_versus_parent(
                row("v", measurement_value=5.0, measurement_unit="widgets"),
                row("p", measurement_value=1.0, measurement_unit="s-1"))
        self.assertIn("will not guess a conversion", str(ctx.exception))

    def test_identical_units_are_untouched(self) -> None:
        comparison = variant_versus_parent(
            row("v", measurement_value=3.0, measurement_unit="s-1"),
            row("p", measurement_value=1.0, measurement_unit="s-1"))
        self.assertAlmostEqual(comparison.delta_measurement, 2.0)

    def test_the_conversion_table_is_self_consistent(self) -> None:
        """Every entry maps onto a canonical unit that is itself an entry."""
        for unit in ("s-1", "min-1", "h-1", "mM", "uM", "%"):
            canon = canonical_unit(unit)
            self.assertIsNotNone(canon, unit)
            assert canon is not None
            self.assertIsNotNone(canonical_unit(canon[0]), canon[0])

    def test_a_known_conversion_is_arithmetically_right(self) -> None:
        converted = convert_measurement(60.0, "min-1")
        self.assertIsNotNone(converted)
        assert converted is not None
        self.assertAlmostEqual(converted[0], 1.0)
        self.assertEqual(converted[1], "s-1")

    def test_an_unknown_unit_converts_to_nothing(self) -> None:
        self.assertIsNone(convert_measurement(1.0, "furlongs per fortnight"))


class FoldOverBackgroundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registration = PreRegistration.from_plan(
            {"min_fold_over_empty_vector": 3.0}, k_slots=2,
            registered_by="operator:test")

    def _row(self, cid: str, value, baseline) -> OutcomeRow:
        return OutcomeRow(
            candidate_id=cid, outcome=OutcomeClass.CONFIRMED_TARGET_PRODUCT,
            soluble_expression=True, measurement_type="initial_rate",
            measurement_value=value, measurement_unit="AU/min",
            empty_vector_baseline=baseline)

    def test_the_primary_endpoint_no_longer_crashes(self) -> None:
        result = precision_at_k([self._row("c1", 30.0, 10.0)], self.registration)
        self.assertEqual(result.n_hits, 1)

    def test_a_row_below_the_bar_is_not_a_hit(self) -> None:
        result = precision_at_k([self._row("c2", 12.0, 10.0)], self.registration)
        self.assertEqual(result.n_hits, 0)

    def test_no_control_makes_the_row_undecidable_not_negative(self) -> None:
        result = precision_at_k([self._row("c3", 30.0, None)], self.registration)
        self.assertEqual(result.n_hits, 0)
        self.assertEqual(result.n_undecidable, 1)

    def test_a_zero_background_is_undecidable(self) -> None:
        result = precision_at_k([self._row("c4", 30.0, 0.0)], self.registration)
        self.assertEqual(result.n_undecidable, 1)

    def test_the_fold_matches_the_ingest_definition(self) -> None:
        """One bar must mean one thing wherever it is evaluated."""
        self.assertAlmostEqual(self._row("c", 30.0, 10.0).fold_over_empty_vector, 3.0)
        self.assertIsNone(self._row("c", 30.0, None).fold_over_empty_vector)
        self.assertIsNone(self._row("c", None, 10.0).fold_over_empty_vector)


if __name__ == "__main__":
    unittest.main(verbosity=2)
