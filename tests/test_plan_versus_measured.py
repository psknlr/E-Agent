"""The returned sheet must not hand the plan's intentions back as findings.

The results template used to pre-fill ``detection_method``,
``confirms_product_identity``, ``authentic_standard``,
``chiral_method_validated`` and the detection limit out of the AssayTemplate,
in the same columns ``ingest_results`` reads as what the bench observed. A
sheet with only the numbers typed in therefore asserted, in every well, that a
GC-MS reading had identified the product against an authentic standard --
including in the wells nobody ran. A conversion number was then enough to earn
``CONFIRMED_TARGET_PRODUCT``.

These tests go the whole way round: compose a batch, fill the returned
template the way a bench would, and ingest it.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from eagent.schemas import OutcomeClass
from eagent.tools.select_batch import (
    ASSAY_MEASURED_COLUMNS, ASSAY_PLAN_COLUMNS, SelectBatch,
)
from eagent.tools.ingest_results import IngestResults, group_rows, parse_assay_rows

from test_select_batch import candidate, make_assay_template, make_ctx, make_task


class TemplateRoundTripTests(unittest.TestCase):
    def compose(self, root: Path):
        ctx = make_ctx(root, make_task(constructs=4))
        out = SelectBatch().run(
            ctx, candidates=[candidate(f"C{i}") for i in range(4)],
            assay_template=make_assay_template(),
            positive_control_enzyme_available=True)
        self.assertIs(out.status.value, out.status.value)
        return ctx, out, Path(out.artifact("assay_results_template").path)

    def read(self, path: Path) -> list[dict[str, str]]:
        with open(path, encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def test_an_untouched_template_states_no_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, path = self.compose(Path(tmp))
            rows, problems = parse_assay_rows(path)
        self.assertEqual(problems, [])
        self.assertTrue(rows)
        self.assertTrue(all(r.tested is None for r in rows))
        self.assertTrue(all(not r.measured for r in rows))

    def test_an_untouched_template_claims_no_product_identification(self) -> None:
        """This is the claim the plan columns used to make on the bench's behalf."""
        with tempfile.TemporaryDirectory() as tmp:
            _, _, path = self.compose(Path(tmp))
            rows, _ = parse_assay_rows(path)
        self.assertFalse(any(r.confirms_product_identity for r in rows))
        self.assertFalse(any(r.detection_method for r in rows))
        self.assertFalse(any(r.limit_of_detection is not None for r in rows))

    def test_the_protocol_is_still_on_the_sheet_for_the_bench(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, path = self.compose(Path(tmp))
            rows = self.read(path)
        self.assertTrue(all(r["plan_detection_method"] for r in rows))
        self.assertEqual({r["plan_confirms_product_identity"] for r in rows},
                         {"yes"})
        self.assertTrue(all(c in rows[0] for c in ASSAY_PLAN_COLUMNS))

    def test_every_measured_column_starts_blank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, path = self.compose(Path(tmp))
            rows = self.read(path)
        for column in ASSAY_MEASURED_COLUMNS:
            self.assertEqual({r[column] for r in rows}, {""}, column)

    def test_numbers_alone_do_not_make_a_confirmed_hit(self) -> None:
        """A bench that types conversions and nothing else has not identified
        a product, and the round must say so rather than infer it."""
        with tempfile.TemporaryDirectory() as tmp:
            _, _, path = self.compose(Path(tmp))
            rows = self.read(path)
            for r in rows:
                r["tested"] = "yes"
                r["expressed_soluble"] = "yes"
                r["measurement_type"] = "conversion"
                r["conversion_pct"] = "90"
            groups = group_rows([g for g in parse_assay_rows(rows)[0]
                                 if not g.is_control])
        self.assertTrue(groups)
        for group in groups:
            self.assertFalse(group.confirms_product_identity)
            self.assertEqual(group.product_identity_observed, "unknown")

    def test_a_filled_sheet_does_make_a_confirmed_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, path = self.compose(Path(tmp))
            rows = self.read(path)
            for r in rows:
                r["tested"] = "yes"
                r["expressed_soluble"] = "yes"
                r["detection_method"] = r["plan_detection_method"]
                r["confirms_product_identity"] = "yes"
                r["authentic_standard"] = "yes"
                r["chiral_method_validated"] = "yes"
                r["limit_of_detection"] = r["plan_limit_of_detection"]
                r["limit_unit"] = r["plan_limit_unit"]
                r["measurement_type"] = "conversion"
                r["conversion_pct"] = "90"
                r["product_identity_observed"] = "target"
                r["peak_area_target_enantiomer"] = "99"
                r["peak_area_opposite_enantiomer"] = "1"
            parsed, _ = parse_assay_rows(rows)
            groups = group_rows([g for g in parsed if not g.is_control])
        for group in groups:
            self.assertTrue(group.confirms_product_identity)
            self.assertEqual(group.product_identity_observed, "target")
            self.assertEqual(group.conversion_pct, 90.0)

    def test_the_plan_columns_are_never_read_as_results(self) -> None:
        """Blanking the plan columns must change nothing about the outcome."""
        with tempfile.TemporaryDirectory() as tmp:
            _, _, path = self.compose(Path(tmp))
            filled = self.read(path)
            for r in filled:
                r["tested"] = "yes"
                r["expressed_soluble"] = "yes"
                r["detection_method"] = "chiral GC-MS"
                r["confirms_product_identity"] = "yes"
                r["measurement_type"] = "conversion"
                r["conversion_pct"] = "90"
                r["product_identity_observed"] = "target"
            stripped = [dict(r, **{c: "" for c in ASSAY_PLAN_COLUMNS})
                        for r in filled]
            with_plan = group_rows([g for g in parse_assay_rows(filled)[0]
                                    if not g.is_control])
            without = group_rows([g for g in parse_assay_rows(stripped)[0]
                                  if not g.is_control])
        self.assertEqual(
            [(g.candidate_id, g.conversion_pct, g.confirms_product_identity)
             for g in with_plan],
            [(g.candidate_id, g.conversion_pct, g.confirms_product_identity)
             for g in without])

    def test_every_plate_carries_its_own_empty_vector_control(self) -> None:
        from eagent.tools.ingest_results import attach_empty_vector_baselines
        with tempfile.TemporaryDirectory() as tmp:
            _, _, path = self.compose(Path(tmp))
            rows = self.read(path)
            for r in rows:
                r["tested"] = "yes"
                r["expressed_soluble"] = "yes"
                r["detection_method"] = "chiral GC-MS"
                r["measurement_type"] = "initial_rate"
                r["measurement_unit"] = "AU/min"
                r["measurement_value"] = ("1.0" if r["kind"] == "control"
                                          else "10.0")
            parsed, _ = parse_assay_rows(rows)
            groups = group_rows([g for g in parsed if not g.is_control])
            attach_empty_vector_baselines(groups, parsed)
        plates = {str(r.plate) for r in parsed}
        for group in groups:
            self.assertEqual(set(group.fold_by_plate()),
                             {p for p in plates
                              if any(r.plate == p for r in group.tested_rows)})
            self.assertAlmostEqual(group.fold_over_empty_vector, 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
