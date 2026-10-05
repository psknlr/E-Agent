"""Tests for the fold-over-empty-vector criterion.

The bar was previously accepted as a recognised criterion key and then never
evaluated. That is the precise failure the ingest step's own error message
warns about for unrecognised keys: an ignored criterion is a criterion that
passes everything. A plan reading "at least 20% conversion and at least
five-fold over the empty vector" would admit a well that converted 25% while
sitting 1.2-fold above its own background.
"""

from __future__ import annotations

import unittest

from eagent.tools.ingest_results import (
    AssayRow, MeasurementGroup, PositiveCriterion,
    attach_empty_vector_baselines, group_rows,
)


def _row(candidate: str, *, plate: str = "P1", kind: str = "candidate",
         role: str = "", value: float | None = None, well: str = "A1",
         cofactor: str = "NADPH", state: str = "reduced",
         replicate: int = 1) -> AssayRow:
    return AssayRow(
        plan_id="plan", slot=1, plate=plate, well=well,
        candidate_id=candidate, construct_id=candidate, kind=kind, role=role,
        parent_candidate_id="", mutations=(), cofactor=cofactor,
        cofactor_state=state, replicate=replicate, tested=True,
        expressed_soluble=True, detection_method="chiral HPLC",
        confirms_product_identity=True, authentic_standard=True,
        chiral_method_validated=True, limit_of_detection=0.1,
        limit_unit="AU/min", measurement_type="initial_rate",
        measurement_value=value, measurement_unit="AU/min",
        conversion_pct=None, product_identity_observed="target",
        peak_area_target=None, peak_area_opposite=None, notes="",
    )


def _criterion(**kw) -> PositiveCriterion:
    raw = dict(kw)
    return PositiveCriterion(raw=raw, source_template_id="t.test", **kw)


class TestBaselineAttachment(unittest.TestCase):
    def test_baseline_comes_from_the_same_plate_and_cofactor(self) -> None:
        rows = [
            _row("c1", plate="P1", value=10.0),
            _row("ev", plate="P1", kind="empty_vector", value=2.0, well="H12"),
            _row("ev", plate="P2", kind="empty_vector", value=50.0, well="H12"),
        ]
        groups = group_rows([r for r in rows if r.kind == "candidate"])
        attach_empty_vector_baselines(groups, rows)
        self.assertEqual(groups[0].empty_vector_baseline, 2.0)
        self.assertEqual(groups[0].fold_over_empty_vector, 5.0)

    def test_a_control_from_another_cofactor_condition_is_not_borrowed(self) -> None:
        """Different condition, different background; borrowing hides that."""
        rows = [
            _row("c1", plate="P1", cofactor="NADPH", value=10.0),
            _row("ev", plate="P1", kind="empty_vector", cofactor="NADH",
                 value=2.0, well="H12"),
        ]
        groups = group_rows([r for r in rows if r.kind == "candidate"])
        notes = attach_empty_vector_baselines(groups, rows)
        self.assertIsNone(groups[0].empty_vector_baseline)
        self.assertIn("no empty-vector control", groups[0].empty_vector_basis)
        self.assertTrue(notes)

    def test_replicate_controls_are_reduced_to_a_median(self) -> None:
        rows = [
            _row("c1", plate="P1", value=12.0),
            _row("ev", plate="P1", kind="empty_vector", value=1.0, well="H10"),
            _row("ev", plate="P1", kind="empty_vector", value=3.0, well="H11"),
            _row("ev", plate="P1", kind="empty_vector", value=2.0, well="H12"),
        ]
        groups = group_rows([r for r in rows if r.kind == "candidate"])
        attach_empty_vector_baselines(groups, rows)
        self.assertEqual(groups[0].empty_vector_baseline, 2.0)


class TestTheBarIsActuallyEvaluated(unittest.TestCase):
    def test_a_well_below_the_fold_bar_fails_even_when_conversion_passes(self) -> None:
        """The regression: this used to be reported as a hit."""
        rows = [
            _row("c1", plate="P1", value=12.0),
            _row("ev", plate="P1", kind="empty_vector", value=10.0, well="H12"),
        ]
        groups = group_rows([r for r in rows if r.kind == "candidate"])
        attach_empty_vector_baselines(groups, rows)
        crit = _criterion(min_measurement_value=5.0,
                          min_fold_over_empty_vector=5.0,
                          measurement_type="initial_rate")
        met, reasons = crit.evaluate(groups[0])
        self.assertIs(met, False)
        self.assertTrue([r for r in reasons if "fold over empty vector" in r])

    def test_a_well_above_the_fold_bar_passes(self) -> None:
        rows = [
            _row("c1", plate="P1", value=60.0),
            _row("ev", plate="P1", kind="empty_vector", value=10.0, well="H12"),
        ]
        groups = group_rows([r for r in rows if r.kind == "candidate"])
        attach_empty_vector_baselines(groups, rows)
        crit = _criterion(min_measurement_value=5.0,
                          min_fold_over_empty_vector=5.0,
                          measurement_type="initial_rate")
        met, _ = crit.evaluate(groups[0])
        self.assertIs(met, True)

    def test_a_missing_control_makes_the_bar_undecidable_not_met(self) -> None:
        rows = [_row("c1", plate="P1", value=60.0)]
        groups = group_rows(rows)
        attach_empty_vector_baselines(groups, rows)
        crit = _criterion(min_fold_over_empty_vector=5.0)
        met, reasons = crit.evaluate(groups[0])
        self.assertIsNone(met, "no control must not be read as the bar being met")
        self.assertTrue([r for r in reasons if "no such control was run" in r])

    def test_a_zero_background_is_undecidable_rather_than_infinite_fold(self) -> None:
        rows = [
            _row("c1", plate="P1", value=60.0),
            _row("ev", plate="P1", kind="empty_vector", value=0.0, well="H12"),
        ]
        groups = group_rows([r for r in rows if r.kind == "candidate"])
        attach_empty_vector_baselines(groups, rows)
        crit = _criterion(min_fold_over_empty_vector=5.0)
        met, reasons = crit.evaluate(groups[0])
        self.assertIsNone(met)
        self.assertTrue([r for r in reasons if "not defined" in r])

    def test_a_criterion_of_only_this_bar_is_now_a_numeric_bar(self) -> None:
        """It used to fall through to 'no numeric bar', deciding nothing."""
        rows = [
            _row("c1", plate="P1", value=60.0),
            _row("ev", plate="P1", kind="empty_vector", value=10.0, well="H12"),
        ]
        groups = group_rows([r for r in rows if r.kind == "candidate"])
        attach_empty_vector_baselines(groups, rows)
        met, reasons = _criterion(min_fold_over_empty_vector=5.0).evaluate(groups[0])
        self.assertIs(met, True)
        self.assertFalse([r for r in reasons if "sets no numeric bar" in r])

    def test_a_definite_miss_still_beats_an_unmeasurable_bar(self) -> None:
        """A well that definitively failed is False, not undecidable."""
        rows = [_row("c1", plate="P1", value=1.0)]
        groups = group_rows(rows)
        attach_empty_vector_baselines(groups, rows)
        crit = _criterion(min_measurement_value=5.0,
                          min_fold_over_empty_vector=5.0,
                          measurement_type="initial_rate")
        met, _ = crit.evaluate(groups[0])
        self.assertIs(met, False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
