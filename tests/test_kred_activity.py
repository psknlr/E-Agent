"""Measured labels, and the three different kinds of absence in them.

This is the first label set in the repository where a positive means somebody
detected a product, so what the tests defend is the arithmetic around the
absences rather than the positives:

* an insoluble construct was **never assayed** and is not a catalytic negative;
* a soluble construct with no detectable product is **left-censored** at a limit
  of detection the source does not state, and is not a measured zero;
* a blank enantiomeric excess is **undefined** (it is 0/0), and is not zero.

Each of the three refuses when asked for a number, with a different reason, and
``endpoint_table`` leaves the first two out unless a caller says otherwise.

Also defended: the reported ee is kept apart from the ee the printed products
imply, because the two disagree by up to several points and are two different
measurements; the two rows that print an ee on a 0/0 are found and withheld;
the two quarantined substrate structures are not handed out; and the 69 soluble
orthologs are counted as the handful of identity groups they are rather than as
69 independent enzymes.
"""

from __future__ import annotations

import json
import math
import unittest

from eagent.eval.kred_activity import (
    BELOW_DETECTION, ENDPOINT_SUBSTRATES, LABEL_TYPES, MEASURED, NOT_ASSAYED,
    SUBSTRATES, ActivityError, endpoint_table, independence, load_activity_set,
)
from eagent.eval.kred_reference import LINEAGE_IDENTITY_THRESHOLD


class _Loaded(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.a = load_activity_set()


# ==========================================================================
class TheDataIsWhatTheSourceSummarySays(_Loaded):
    def test_the_construct_counts(self) -> None:
        c = self.a.counts()
        self.assertEqual((c["constructs"], c["soluble"], c["insoluble"]), (109, 69, 40))
        self.assertEqual(c["endpoint_pairs_assayed"], 276)
        self.assertEqual(c["relative_slope_rows"], 69)

    def test_the_per_substrate_tallies(self) -> None:
        """69 x 4 endpoint pairs, split as the archive's own summary states."""
        self.assertEqual(self.a.counts()["by_substrate"], {
            "2a": {MEASURED: 33, BELOW_DETECTION: 36, NOT_ASSAYED: 40},
            "3a": {MEASURED: 68, BELOW_DETECTION: 1, NOT_ASSAYED: 40},
            "4a": {MEASURED: 69, BELOW_DETECTION: 0, NOT_ASSAYED: 40},
            "5a": {MEASURED: 46, BELOW_DETECTION: 23, NOT_ASSAYED: 40},
        })

    def test_the_largest_product_per_substrate(self) -> None:
        for sub, expected in (("2a", 7.7), ("3a", 6.5), ("4a", 3.4), ("5a", 7.6)):
            rows = endpoint_table(self.a, sub, "total_product_mM")
            self.assertAlmostEqual(max(v for _, v in rows), expected, places=6, msg=sub)

    def test_it_loads_strictly_with_no_errors(self) -> None:
        self.assertEqual([f for f in self.a.findings if f.severity == "error"], [])

    def test_the_parent_is_the_normalisation_point_and_appears_once(self) -> None:
        parent = self.a.record("Ssal-KRED")
        self.assertTrue(parent.is_parent)
        self.assertEqual(parent.require("relative_depletion_slope"), 1.0)
        self.assertEqual(sorted(parent.rounds), ["round_1", "round_2"])
        self.assertEqual(sum(1 for r in self.a.records if r.is_parent), 1)

    def test_the_rows_that_appear_in_both_rounds_agree(self) -> None:
        both = [r for r in self.a.records if len(r.rounds) > 1]
        self.assertEqual(len(both), 6)
        self.assertEqual([f for f in self.a.findings
                          if f.code == "activity.round_disagreement"], [])


class AnInsolubleConstructWasNeverAssayed(_Loaded):
    def test_the_forty_are_not_assayed_on_every_substrate(self) -> None:
        for record in self.a.insoluble:
            self.assertIsNone(record.relative_depletion_slope, record.enzyme_id)
            for sub in ENDPOINT_SUBSTRATES:
                self.assertEqual(record.endpoint(sub).status, NOT_ASSAYED)

    def test_asking_for_a_product_says_it_is_not_a_catalytic_negative(self) -> None:
        record = self.a.insoluble[0]
        with self.assertRaises(ActivityError) as ctx:
            record.endpoint("2a").require("total_product_mM")
        self.assertIn("not a catalytic negative", str(ctx.exception))
        with self.assertRaisesRegex(ActivityError, "never assayed"):
            record.require("relative_depletion_slope")

    def test_no_insoluble_row_carries_an_activity_value(self) -> None:
        self.assertEqual([f for f in self.a.findings
                          if f.code == "activity.insoluble_carries_values"], [])

    def test_they_are_absent_from_every_endpoint_table(self) -> None:
        insoluble = {r.enzyme_id for r in self.a.insoluble}
        for sub in ENDPOINT_SUBSTRATES:
            for include in (False, True):
                served = {r.enzyme_id for r, _ in endpoint_table(
                    self.a, sub, "total_product_mM", include_censored=include)}
                self.assertEqual(served & insoluble, set(), f"{sub} {include}")


class NoDetectableProductIsCensoredNotZero(_Loaded):
    def censored(self, sub: str):
        return [r for r in self.a.soluble if r.endpoint(sub).status == BELOW_DETECTION]

    def test_both_products_are_exactly_zero(self) -> None:
        for record in self.censored("2a"):
            endpoint = record.endpoint("2a")
            self.assertEqual((endpoint.product_r_mM, endpoint.product_s_mM), (0.0, 0.0))

    def test_asking_for_a_product_says_left_censored_not_zero(self) -> None:
        record = self.censored("2a")[0]
        with self.assertRaises(ActivityError) as ctx:
            record.endpoint("2a").require("total_product_mM")
        message = str(ctx.exception)
        self.assertIn("left-censored", message)
        self.assertIn("not a measured zero", message)

    def test_they_are_left_out_of_an_endpoint_table_by_default(self) -> None:
        plain = endpoint_table(self.a, "2a", "total_product_mM")
        with_censored = endpoint_table(self.a, "2a", "total_product_mM",
                                       include_censored=True)
        self.assertEqual(len(plain), 33)
        self.assertEqual(len(with_censored), 69)
        self.assertEqual(sorted(v for _, v in with_censored)[:36], [0.0] * 36)

    def test_a_censored_row_has_no_ee_from_its_products(self) -> None:
        for record in self.censored("2a"):
            endpoint = record.endpoint("2a")
            self.assertIsNone(endpoint.ee_from_products)
            with self.assertRaisesRegex(ActivityError, "0/0"):
                endpoint.require("ee_from_products")

    def test_a_soluble_construct_can_be_censored_on_one_substrate_and_not_another(self) -> None:
        """Which is why a label may not be carried across substrates."""
        mixed = [r for r in self.a.soluble
                 if r.endpoint("2a").status == BELOW_DETECTION
                 and r.endpoint("4a").status == MEASURED]
        self.assertTrue(mixed)
        record = mixed[0]
        self.assertGreater(record.endpoint("4a").require("total_product_mM"), 0)


class AnEnantiomericExcessIsReportedNotDerived(_Loaded):
    def test_the_printed_ee_and_the_one_the_products_imply_both_exist(self) -> None:
        record = self.a.record("Ssal-KRED")
        endpoint = record.endpoint("2a")
        self.assertAlmostEqual(endpoint.require("ee_reported"), 12.5, places=6)
        self.assertAlmostEqual(endpoint.require("ee_from_products"), 12.5, places=4)

    def test_the_two_disagree_on_many_rows_and_both_are_kept(self) -> None:
        disagreeing = [
            (r.enzyme_id, s) for r in self.a.soluble for s in ENDPOINT_SUBSTRATES
            if (e := r.endpoint(s)).ee_residual is not None
            and abs(e.ee_residual) > 1.0]
        self.assertGreater(len(disagreeing), 30,
                           "the printed ee is a separate measurement from the "
                           "products; if these agreed everywhere it would be derived")
        for enzyme_id, sub in disagreeing:
            endpoint = self.a.record(enzyme_id).endpoint(sub)
            self.assertIsNotNone(endpoint.ee_reported)
            self.assertIsNotNone(endpoint.ee_from_products)

    def test_every_disagreement_is_reported_as_a_finding(self) -> None:
        noted = {f.subject for f in self.a.findings
                 if f.code == "activity.ee_reported_vs_products"}
        for record in self.a.soluble:
            for sub in ENDPOINT_SUBSTRATES:
                residual = record.endpoint(sub).ee_residual
                if residual is not None and abs(residual) > 1.0:
                    self.assertIn(f"{record.enzyme_id}/{sub}", noted)

    def test_the_ee_the_products_imply_is_the_textbook_formula(self) -> None:
        for record in self.a.soluble:
            for sub in ENDPOINT_SUBSTRATES:
                endpoint = record.endpoint(sub)
                if endpoint.ee_from_products is None:
                    continue
                r, s = endpoint.product_r_mM, endpoint.product_s_mM
                self.assertAlmostEqual(endpoint.ee_from_products,
                                       (r - s) / (r + s) * 100.0, places=9)

    def test_the_two_rows_printing_an_ee_on_a_zero_over_zero_are_withheld(self) -> None:
        """A finding this loader made that the source's own audit did not list."""
        flagged = sorted(f.subject for f in self.a.findings
                         if f.code == "activity.ee_on_zero_over_zero")
        self.assertEqual(flagged, ["Ort-EZM-24/2a", "Ort-EZM-6/2a"])
        for subject in flagged:
            enzyme_id, sub = subject.split("/")
            endpoint = self.a.record(enzyme_id).endpoint(sub)
            self.assertEqual(endpoint.status, BELOW_DETECTION)
            self.assertIsNone(endpoint.ee_reported)
            with self.assertRaisesRegex(ActivityError, "0/0"):
                endpoint.require("ee_reported")

    def test_a_detected_product_with_no_reported_ee_is_noted_not_filled_in(self) -> None:
        noted = [f.subject for f in self.a.findings
                 if f.code == "activity.product_without_ee"]
        self.assertEqual(noted, ["Ort-EZM-23/2a"])
        endpoint = self.a.record("Ort-EZM-23").endpoint("2a")
        self.assertEqual(endpoint.status, MEASURED)
        self.assertGreater(endpoint.require("total_product_mM"), 0)
        self.assertIsNone(endpoint.ee_reported)


class TheLabelTypesDoNotPool(_Loaded):
    def test_there_are_three_of_them_and_the_substrates_declare_which(self) -> None:
        self.assertEqual(set(LABEL_TYPES),
                         {"relative_depletion_slope", "product_concentration",
                          "signed_ee"})
        self.assertEqual(SUBSTRATES["1a"].label_type, "relative_depletion_slope")
        for sub in ENDPOINT_SUBSTRATES:
            self.assertEqual(SUBSTRATES[sub].label_type, "product_concentration")

    def test_the_relative_slope_is_a_ratio_and_not_a_concentration(self) -> None:
        """1b is normalised to the parent, so it has no units and no mM."""
        self.assertEqual(self.a.record("Ssal-KRED").require("relative_depletion_slope"),
                         1.0)
        with self.assertRaises(ActivityError):
            self.a.record("Ssal-KRED").require("total_product_mM")

    def test_an_endpoint_table_serves_one_substrate_at_a_time(self) -> None:
        for sub in ENDPOINT_SUBSTRATES:
            for record, _ in endpoint_table(self.a, sub, "total_product_mM"):
                self.assertEqual(record.endpoint(sub).substrate_id, sub)

    def test_an_unknown_substrate_or_quantity_is_refused(self) -> None:
        with self.assertRaisesRegex(ActivityError, "not an assayed substrate"):
            endpoint_table(self.a, "9a", "total_product_mM")
        with self.assertRaisesRegex(ActivityError, "not a quantity"):
            self.a.record("Ssal-KRED").endpoint("2a").require("kcat")
        with self.assertRaisesRegex(ActivityError, "not a record-level quantity"):
            self.a.record("Ssal-KRED").require("product_r_mM")

    def test_the_melting_temperature_is_not_an_activity_label(self) -> None:
        tm = self.a.record("Ssal-KRED").require("melting_temperature_c")
        self.assertAlmostEqual(tm, 59.7, places=6)
        self.assertNotIn("melting_temperature", LABEL_TYPES)


class TheTwoQuarantinedStructuresAreNotHandedOut(unittest.TestCase):
    def test_the_deposited_smiles_for_1a_is_an_alcohol_and_is_refused(self) -> None:
        substrate = SUBSTRATES["1a"]
        self.assertTrue(substrate.quarantined)
        with self.assertRaises(ActivityError) as ctx:
            substrate.require_smiles()
        self.assertIn("alcohol", str(ctx.exception))
        self.assertIn("C(O)", substrate.smiles_as_deposited)
        self.assertIn("C(=O)", substrate.proposed_smiles)

    def test_the_deposited_smiles_for_5a_is_a_carboxylate_and_is_refused(self) -> None:
        substrate = SUBSTRATES["5a"]
        self.assertTrue(substrate.quarantined)
        with self.assertRaisesRegex(ActivityError, "carboxylate"):
            substrate.require_smiles()
        self.assertIn("[O-]", substrate.smiles_as_deposited)
        self.assertEqual(substrate.proposed_smiles, "CCOC(=O)CC(=O)CCl")

    def test_the_source_value_is_preserved_beside_the_proposal(self) -> None:
        for key in ("1a", "5a"):
            substrate = SUBSTRATES[key]
            self.assertNotEqual(substrate.smiles_as_deposited,
                                substrate.proposed_smiles)
            self.assertTrue(substrate.quarantine_reason)

    def test_the_three_unquarantined_substrates_are_handed_out(self) -> None:
        self.assertEqual(SUBSTRATES["2a"].require_smiles(), "CC(=O)C1=CC=CC=C1")
        for key in ("3a", "4a"):
            self.assertTrue(SUBSTRATES[key].require_smiles())
            self.assertFalse(SUBSTRATES[key].quarantined)


class SixtyNineOrthologsAreNotSixtyNineEnzymes(_Loaded):
    def test_the_group_count_is_cited_at_this_projects_lineage_threshold(self) -> None:
        groups = independence(self.a)
        self.assertEqual(groups["cited_threshold"],
                         f"{LINEAGE_IDENTITY_THRESHOLD:.2f}")
        self.assertEqual(groups["soluble_constructs"], 69)
        self.assertEqual(groups["n_independent_at_cited_threshold"], 5)

    def test_at_the_cited_threshold_almost_all_of_them_are_one_group(self) -> None:
        at40 = independence(self.a)["groups_by_threshold"]["0.40"]
        self.assertEqual(at40["n_groups"], 5)
        self.assertEqual(at40["largest_group"], 59)

    def test_the_count_moves_a_long_way_with_the_threshold(self) -> None:
        """Which is why the threshold has to be stated with the number."""
        by = independence(self.a)["groups_by_threshold"]
        self.assertEqual([by[t]["n_groups"] for t in ("0.40", "0.70", "0.90", "0.95")],
                         [5, 49, 67, 68])

    def test_the_stored_groups_were_computed_for_these_constructs(self) -> None:
        stored = json.loads(
            (self.a.directory / "activity_expansion_2026"
             / "derived_identity_groups.json").read_text(encoding="utf-8"))
        self.assertEqual(sorted(stored["members"]),
                         sorted(r.enzyme_id for r in self.a.soluble))
        self.assertEqual(stored["n_pairs"], 69 * 68 // 2)
        self.assertLess(stored["identity_median"], 0.5)
        self.assertGreater(stored["identity_max"], 0.95)

    def test_a_group_count_computed_for_another_set_is_refused(self) -> None:
        import dataclasses
        trimmed = dataclasses.replace(
            self.a, records=tuple(r for r in self.a.records
                                  if r.enzyme_id != "Ort-RDM-1"))
        with self.assertRaisesRegex(ActivityError, "different set of constructs"):
            independence(trimmed)

    def test_no_two_constructs_share_a_sequence(self) -> None:
        self.assertEqual([f for f in self.a.findings
                          if f.code == "activity.identical_sequences"], [])
        self.assertEqual(len({r.sequence for r in self.a.records}), 109)


if __name__ == "__main__":
    unittest.main()
