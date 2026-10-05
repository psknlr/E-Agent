"""Regression tests: three ways the result reader could manufacture a hit.

All three came from an external audit and were reproduced here before being
fixed. They share one shape: a piece of missing or unasserted information was
converted into a definite experimental claim. That is the most damaging class
of bug in this system, because an experimental label is what the next round of
design and the final effect estimate are both built on.
"""

from __future__ import annotations

import unittest

from eagent.schemas import OutcomeClass
from eagent.tools.ingest_results import (
    AssayRow, PositiveCriterion, classify_group, group_rows,
)


def row(**kw) -> AssayRow:
    base = dict(
        plan_id="p", slot=1, plate="P1", well="A1", candidate_id="c1",
        construct_id="c1", kind="candidate", role="", parent_candidate_id="",
        mutations=(), cofactor="NADPH", cofactor_state="reduced", replicate=1,
        tested=True, expressed_soluble=True, detection_method="chiral HPLC",
        confirms_product_identity=True, authentic_standard=True,
        chiral_method_validated=True, limit_of_detection=0.1, limit_unit="%",
        measurement_type="conversion", measurement_value=None,
        measurement_unit="%", conversion_pct=None,
        product_identity_observed="target", peak_area_target=None,
        peak_area_opposite=None, notes="",
    )
    base.update(kw)
    return AssayRow(**base)


def criterion(**kw) -> PositiveCriterion:
    return PositiveCriterion(raw=dict(kw), source_template_id="t.test", **kw)


class BlankOppositePeakTests(unittest.TestCase):
    """A blank cell is missing data, not an absent enantiomer."""

    def test_a_blank_opposite_peak_yields_no_ee(self) -> None:
        g = group_rows([row(peak_area_target=100.0, peak_area_opposite=None,
                            conversion_pct=30.0)])[0]
        self.assertIsNone(
            g.ee_target_pct,
            "summing a blank opposite peak as zero reports an exact 100% ee, "
            "which is the strongest stereochemical claim the assay can make, "
            "manufactured from an empty cell")

    def test_the_incomplete_chiral_analysis_is_surfaced(self) -> None:
        """No ee must not be allowed to read as 'not a chiral assay'."""
        g = group_rows([row(peak_area_target=100.0, peak_area_opposite=None)])[0]
        self.assertTrue(g.ee_incomplete_peaks)

    def test_both_peaks_present_still_computes_normally(self) -> None:
        g = group_rows([row(peak_area_target=90.0, peak_area_opposite=10.0)])[0]
        self.assertAlmostEqual(g.ee_target_pct, 80.0)

    def test_a_wrong_handed_result_is_still_negative(self) -> None:
        g = group_rows([row(peak_area_target=10.0, peak_area_opposite=90.0)])[0]
        self.assertAlmostEqual(g.ee_target_pct, -80.0)

    def test_an_explicit_non_detection_gives_a_bound_not_an_exact_value(self) -> None:
        """A recorded limit supports a bound; it never supports an exact 100%."""
        g = group_rows([row(peak_area_target=100.0, peak_area_opposite=None,
                            limit_of_detection=0.1)])[0]
        bound = g.ee_target_lower_bound()
        self.assertIsNotNone(bound)
        assert bound is not None
        value, basis = bound
        self.assertLess(value, 100.0, "a bound at the limit cannot be exactly 100%")
        self.assertGreater(value, 99.0)
        self.assertIn("limit", basis)

    def test_no_recorded_limit_supports_no_bound_at_all(self) -> None:
        g = group_rows([row(peak_area_target=100.0, peak_area_opposite=None,
                            limit_of_detection=None)])[0]
        self.assertIsNone(g.ee_target_lower_bound())


class UntestedWellTests(unittest.TestCase):
    """A well marked tested=no routinely still carries last round's number."""

    def test_an_untested_well_does_not_enter_the_conversion_median(self) -> None:
        rows = [row(well="A1", tested=True, conversion_pct=0.0),
                row(well="A2", tested=False, conversion_pct=100.0, replicate=2)]
        g = group_rows(rows)[0]
        self.assertEqual(
            g.conversion_pct, 0.0,
            "the only well that was run read 0%; a stale value from a well "
            "nobody ran must not outvote it")

    def test_an_untested_well_does_not_enter_the_measurement_median(self) -> None:
        rows = [row(well="A1", tested=True, measurement_value=1.0),
                row(well="A2", tested=False, measurement_value=99.0, replicate=2)]
        self.assertEqual(group_rows(rows)[0].measurement_value, 1.0)

    def test_an_untested_well_does_not_contribute_peak_areas(self) -> None:
        rows = [row(well="A1", tested=True, peak_area_target=50.0,
                    peak_area_opposite=50.0),
                row(well="A2", tested=False, peak_area_target=100.0,
                    peak_area_opposite=0.0, replicate=2)]
        self.assertAlmostEqual(group_rows(rows)[0].ee_target_pct, 0.0)

    def test_an_untested_well_does_not_contribute_an_identity(self) -> None:
        rows = [row(well="A1", tested=True, product_identity_observed="none"),
                row(well="A2", tested=False, product_identity_observed="target",
                    replicate=2)]
        self.assertEqual(group_rows(rows)[0].product_identity_observed, "none")

    def test_the_untested_row_is_still_retained_on_the_group(self) -> None:
        """Excluded from aggregates, not deleted: the plate record stays whole."""
        rows = [row(well="A1", tested=True, conversion_pct=0.0),
                row(well="A2", tested=False, conversion_pct=100.0, replicate=2)]
        g = group_rows(rows)[0]
        self.assertEqual(g.n_replicates, 2)
        self.assertEqual(len(g.tested_rows), 1)


class ProductIdentityTests(unittest.TestCase):
    """Being able to identify the product is not having identified it."""

    def setUp(self) -> None:
        self.criterion = criterion(min_conversion_pct=50.0)

    def _classify(self, identity: str):
        g = group_rows([row(product_identity_observed=identity,
                            conversion_pct=90.0, measurement_value=90.0,
                            detection_method="GC-MS")])[0]
        return classify_group(g, self.criterion)

    def test_no_product_identified_is_not_a_confirmed_hit(self) -> None:
        result = self._classify("none")
        self.assertIsNone(result.outcome)
        assert result.unresolved is not None
        self.assertEqual(result.unresolved.reason_code,
                         "bars_met_without_identified_product")

    def test_an_unknown_identity_is_not_a_confirmed_hit(self) -> None:
        self.assertIsNone(self._classify("unknown").outcome)

    def test_an_identified_target_is_still_a_hit(self) -> None:
        self.assertIs(self._classify("target").outcome,
                      OutcomeClass.CONFIRMED_TARGET_PRODUCT)

    def test_another_product_remains_wrong_chemoselectivity(self) -> None:
        self.assertIs(self._classify("other").outcome,
                      OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION)

    def test_the_unresolved_row_says_what_would_settle_it(self) -> None:
        result = self._classify("none")
        assert result.unresolved is not None
        self.assertIn("consumption is not product formation",
                      result.unresolved.reason)
        self.assertTrue(result.unresolved.required_to_resolve)


if __name__ == "__main__":
    unittest.main(verbosity=2)
