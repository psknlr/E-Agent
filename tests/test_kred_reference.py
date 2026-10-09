"""The KRED calibration reference set: what the delivery says, and what holds.

The set arrived as one spreadsheet compiled by an AI assistant, with a cover note
that makes specific claims (19 structures, 28 kinetic records of which 26 are
numeric and 2 are ND, 22/3/1 in the core/secondary/sensitivity tiers). These
tests defend three things:

* **the files are the delivery.** A stored table cannot differ from the stored
  workbook it was converted from, and a changed byte anywhere is visible;
* **the numbers are recomputed, not read.** Every SI-unit column is a formula
  cell in the workbook; here each is rebuilt from the original value and unit
  through the project's own unit table, and a doctored unit is caught;
* **a missing number stays missing.** ``ND`` is not zero, a ``<`` bound is not
  a point, a reported efficiency is not a ``kcat/Km`` quotient, and the four
  label types are never pooled.

Everything here is offline and reads only what is committed.
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from eagent.eval.kred_reference import (
    GEOMETRY_USE, LINEAGE_GROUPS, STATED_COUNTS, TIERS, Finding, ReferenceSetError,
    build_manifest, default_reference_dir, load_reference_set, verify_conversion,
    verify_manifest, write_manifest,
)
from eagent.eval.kred_workbook import SHEET_TABLES, WorkbookError, read_workbook
from eagent.science.units import canonical_unit, convert_measurement, reconcile_units

#: The digest of the data files (workbook, converted tables, bindings, the
#: coordinate pins). Pinned here so that changing any of them -- a corrected
#: number, a new binding, a re-pinned coordinate file -- also changes this file
#: in the same commit, where a reviewer will see both. Update it with
#: ``eagent reference manifest --write`` after reading the diff, not before.
EXPECTED_DATA_DIGEST = "73a65de287bcac85d2d7fb79c489e717fd98382ca4da3c874b4ae60ebbfbb32e"

DELIVERED_WORKBOOK_SHA256 = (
    "c0eac78f178ecf0a82c90f3b451ddddaef706ccd3a763ce0529a679ad25fbf52")


def _copy_set(tmp: Path) -> Path:
    """A writable copy of the shipped set, for tests that damage it."""
    target = tmp / "set"
    shutil.copytree(default_reference_dir(), target)
    return target


def _rewrite_csv(path: Path, edit) -> None:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    edit(rows)
    with path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)


def _cell(rows, key_col: str, key: str, col: str):
    head = rows[0]
    for row in rows[1:]:
        if row[head.index(key_col)] == key:
            return row, head.index(col)
    raise KeyError(key)


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def findings(self, directory: Path, code: str) -> list[Finding]:
        rs = load_reference_set(directory, verify=False, strict=False)
        return [f for f in rs.findings if f.code == code]


# ==========================================================================
class TheFilesAreTheDelivery(_Tmp):
    def test_the_shipped_set_matches_its_manifest(self) -> None:
        self.assertEqual(verify_manifest(default_reference_dir()), [])

    def test_the_stored_workbook_is_the_one_that_was_uploaded(self) -> None:
        import hashlib

        path = default_reference_dir() / "source" / "KRED_Calibration_Reference_v0.1.xlsx"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         DELIVERED_WORKBOOK_SHA256)

    def test_the_stored_tables_are_what_the_converter_makes_of_the_workbook(self) -> None:
        self.assertEqual(verify_conversion(default_reference_dir()), [])

    def test_the_data_digest_is_the_pinned_one(self) -> None:
        manifest = json.loads((default_reference_dir() / "MANIFEST.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(manifest["data_digest"], EXPECTED_DATA_DIGEST,
                         "a data file changed. Read the diff, then update "
                         "EXPECTED_DATA_DIGEST in this test in the same commit")

    def test_one_edited_byte_in_a_table_is_reported_by_name(self) -> None:
        base = _copy_set(self.tmp)
        path = base / "tables" / "kinetics.csv"
        path.write_bytes(path.read_bytes().replace(b"0.49", b"0.94", 1))
        problems = verify_manifest(base)
        self.assertEqual([(f.code, f.subject) for f in problems],
                         [("manifest.file_changed", "tables/kinetics.csv")])

    def test_a_table_edited_after_conversion_no_longer_matches_the_workbook(self) -> None:
        base = _copy_set(self.tmp)
        path = base / "tables" / "kinetics.csv"
        path.write_bytes(path.read_bytes().replace(b"0.49", b"0.94", 1))
        problems = verify_conversion(base)
        self.assertEqual([f.code for f in problems], ["conversion.table_differs"])

    def test_a_loader_refuses_an_edited_set_under_strict_verification(self) -> None:
        base = _copy_set(self.tmp)
        (base / "tables" / "sources.csv").write_bytes(b"corrupt")
        with self.assertRaises(ReferenceSetError) as ctx:
            load_reference_set(base)
        codes = {f.code for f in ctx.exception.findings}
        self.assertIn("manifest.file_changed", codes)

    def test_a_changed_readme_is_a_warning_that_does_not_stop_the_set_loading(self) -> None:
        base = _copy_set(self.tmp)
        (base / "README.md").write_text("a typo fix\n", encoding="utf-8")
        found = verify_manifest(base)
        self.assertEqual([(f.severity, f.code, f.subject) for f in found],
                         [("warning", "manifest.file_changed", "README.md")])
        rs = load_reference_set(base)            # strict, with verification: still loads
        self.assertEqual(rs.counts(), STATED_COUNTS)
        self.assertTrue(any(f.code == "manifest.file_changed" and f.severity == "warning"
                            for f in rs.findings))

    def test_the_verification_record_is_guarded_like_data(self) -> None:
        base = _copy_set(self.tmp)
        path = base / "verification" / "results.json"
        path.write_bytes(path.read_bytes().replace(b'"mismatch": 0', b'"mismatch": 1', 1))
        self.assertEqual([(f.severity, f.subject) for f in verify_manifest(base)],
                         [("error", "verification/results.json")])

    def test_an_added_file_is_unlisted_and_a_removed_one_is_missing(self) -> None:
        base = _copy_set(self.tmp)
        (base / "tables" / "extra.csv").write_text("a,b\n", encoding="utf-8")
        (base / "tables" / "usage_notes.csv").unlink()
        codes = {(f.code, f.subject) for f in verify_manifest(base)}
        self.assertIn(("manifest.file_unlisted", "tables/extra.csv"), codes)
        self.assertIn(("manifest.file_missing", "tables/usage_notes.csv"), codes)

    def test_a_missing_manifest_is_an_error_not_a_pass(self) -> None:
        base = _copy_set(self.tmp)
        (base / "MANIFEST.json").unlink()
        self.assertEqual([f.code for f in verify_manifest(base)], ["manifest.missing"])

    def test_documentation_is_listed_but_is_not_part_of_the_data_digest(self) -> None:
        base = _copy_set(self.tmp)
        before = build_manifest(base)["data_digest"]
        (base / "NOTICE.md").write_text("a typo fix\n", encoding="utf-8")
        after = build_manifest(base)
        self.assertEqual(after["data_digest"], before)
        self.assertIn("NOTICE.md", after["files"])

    def test_the_directory_is_not_subject_to_line_ending_rewriting(self) -> None:
        text = (default_reference_dir() / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("-text", text)


# ==========================================================================
class TheWorkbookReaderIsStrict(_Tmp):
    def test_every_expected_sheet_has_a_table_and_a_matching_header(self) -> None:
        tables = read_workbook(default_reference_dir() / "source"
                               / "KRED_Calibration_Reference_v0.1.xlsx")
        self.assertEqual({t.sheet for t in tables}, set(SHEET_TABLES))
        for t in tables:
            self.assertEqual(tuple(t.header), SHEET_TABLES[t.sheet][1])

    def test_a_file_that_is_not_a_workbook_is_refused(self) -> None:
        bad = self.tmp / "x.xlsx"
        bad.write_bytes(b"not a zip")
        with self.assertRaises(WorkbookError):
            read_workbook(bad)

    def test_a_workbook_with_a_dtd_is_refused_before_it_is_parsed(self) -> None:
        import zipfile

        source = default_reference_dir() / "source" / "KRED_Calibration_Reference_v0.1.xlsx"
        evil = self.tmp / "evil.xlsx"
        with zipfile.ZipFile(source) as src, zipfile.ZipFile(evil, "w") as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename == "xl/sharedStrings.xml":
                    data = (b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]>'
                            + data.split(b"?>", 1)[1])
                dst.writestr(item, data)
        with self.assertRaisesRegex(WorkbookError, "DTD or an entity"):
            read_workbook(evil)

    def test_formula_text_is_kept_apart_from_values_and_never_evaluated(self) -> None:
        tables = {t.stem: t for t in read_workbook(
            default_reference_dir() / "source" / "KRED_Calibration_Reference_v0.1.xlsx")}
        kin = tables["kinetics"]
        self.assertEqual(kin.formulas["P26"], "F26/60")        # kcat min^-1 -> s^-1
        self.assertEqual(kin.formulas["Q26"], "I26/1000000")   # Km uM -> M
        self.assertEqual(kin.formulas["R16"], "M16*1000/60")   # mM^-1 min^-1 -> M^-1 s^-1


# ==========================================================================
class TheCoverNoteIsTrue(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rs = load_reference_set()

    def test_the_set_loads_with_no_error_findings(self) -> None:
        self.assertEqual([f for f in self.rs.findings if f.severity == "error"], [])

    def test_every_count_the_delivery_states_is_what_the_tables_hold(self) -> None:
        self.assertEqual(self.rs.counts(), STATED_COUNTS)

    def test_the_counts_are_the_ones_in_the_report(self) -> None:
        c = self.rs.counts()
        self.assertEqual((c["structures"], c["kinetic_records"], c["numeric_records"],
                          c["not_determined_records"], c["records_with_kcat"],
                          c["efficiency_only_records"], c["selectivity_records"]),
                         (19, 28, 26, 2, 18, 8, 10))

    def test_the_tiers_are_the_22_3_1_2_of_the_usage_notes(self) -> None:
        by_tier = {t: [k.label_id for k in self.rs.kinetics if k.tier == t]
                   for t in ("core", "secondary", "sensitivity", "nd")}
        self.assertEqual([len(by_tier[t]) for t in ("core", "secondary", "sensitivity", "nd")],
                         [22, 3, 1, 2])
        self.assertEqual(by_tier["sensitivity"], ["Ssal2024_M4_1a"])
        self.assertEqual(sorted(by_tier["secondary"]),
                         ["DSTRII_TROP_NADPH_2003", "LB_G37D_AP_NADH_2005",
                          "LB_WT_AP_NADPH_2005"])
        self.assertEqual(sorted(by_tier["nd"]),
                         ["PNAS2015_Y190F_1", "PNAS2015_Y190F_2"])

    def test_the_core_set_is_hbdh_8_lkkred_8_ssal_6(self) -> None:
        by: dict[str, int] = {}
        for k in self.rs.kinetics:
            if k.tier == "core":
                by[k.enzyme] = by.get(k.enzyme, 0) + 1
        self.assertEqual(by, {"PaHBDH": 4, "AbHBDH": 4, "LkKRED": 8, "Ssal-KRED": 6})

    def test_eighteen_records_have_a_kcat_and_eight_have_only_an_efficiency(self) -> None:
        numeric = [k for k in self.rs.kinetics if k.record_status != "not_determined"]
        self.assertEqual(sum(k.kcat_original is not None for k in numeric), 18)
        self.assertEqual(sum(k.kcat_original is None for k in numeric), 8)

    def test_no_entry_is_graded_a_strict_geometry_reference(self) -> None:
        """The audit found none, and the grade vocabulary has no place to put one."""
        self.assertFalse([g for g in GEOMETRY_USE if "strict" in g or "gold" in g])
        for s in self.rs.structures:
            self.assertIn(s.geometry_use, GEOMETRY_USE)
        self.assertEqual(
            sorted(s.pdb_id for s in self.rs.structures
                   if s.geometry_use == "conditional_pose_reference"),
            ["1IPF", "6ZZO"])


# ==========================================================================
class EveryNumberIsRecomputed(_Tmp):
    def test_every_cached_si_value_equals_the_recomputation(self) -> None:
        rs = load_reference_set()
        self.assertEqual([f for f in rs.findings if f.code == "kinetics.recomputation"], [])

    def test_a_per_minute_constant_read_as_per_second_is_caught(self) -> None:
        """The 60-fold error that makes a variant look sixty times better."""
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "label_id", "Ssal2024_WT_1a", "kcat_unit")
            row[col] = "s^-1"
        _rewrite_csv(base / "tables" / "kinetics.csv", edit)
        found = self.findings(base, "kinetics.recomputation")
        self.assertTrue(found)
        self.assertTrue(any("Ssal2024_WT_1a" == f.subject and "kcat_s_1" in f.message
                            for f in found))

    def test_a_micromolar_value_labelled_millimolar_is_caught(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "label_id", "Ssal2024_M1_1a", "Km_unit")
            row[col] = "mM"
        _rewrite_csv(base / "tables" / "kinetics.csv", edit)
        self.assertTrue(any(f.subject == "Ssal2024_M1_1a" and "Km_value_or_bound_M" in f.message
                            for f in self.findings(base, "kinetics.recomputation")))

    def test_an_efficiency_unit_the_table_does_not_know_is_refused_not_guessed(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "label_id", "PNAS2015_WT_1", "efficiency_unit")
            row[col] = "mM^-1 fortnight^-1"
        _rewrite_csv(base / "tables" / "kinetics.csv", edit)
        self.assertTrue(self.findings(base, "unit.unknown"))

    def test_a_rate_constant_in_a_concentration_unit_is_the_wrong_quantity(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "label_id", "6ZZO_PaHBDH_AAE", "kcat_unit")
            row[col] = "mM"
        _rewrite_csv(base / "tables" / "kinetics.csv", edit)
        self.assertTrue(self.findings(base, "unit.wrong_quantity"))

    def test_the_efficiency_units_are_in_the_shared_table_with_the_right_factors(self) -> None:
        self.assertEqual(canonical_unit("mM^-1 min^-1"), ("M-1 s-1", 1000.0 / 60.0))
        self.assertEqual(canonical_unit("min^-1 mM^-1"), ("M-1 s-1", 1000.0 / 60.0))
        self.assertEqual(canonical_unit("M^-1 s^-1"), ("M-1 s-1", 1.0))
        self.assertEqual(canonical_unit("uM-1 s-1"), ("M-1 s-1", 1e6))
        # 0.025 mM^-1 min^-1 is 0.41666... M^-1 s^-1
        value, unit = convert_measurement(0.025, "mM^-1 min^-1")
        self.assertAlmostEqual(value, 0.4166666666666667, places=12)
        self.assertEqual(unit, "M-1 s-1")

    def test_an_efficiency_never_pools_with_a_first_order_constant_or_a_concentration(self) -> None:
        for other in ("s^-1", "mM"):
            result = reconcile_units([(1.0, "mM^-1 min^-1"), (1.0, other)])
            self.assertFalse(result.usable, other)

    def test_converted_values_can_be_reconciled_across_spellings(self) -> None:
        result = reconcile_units([(0.025, "mM^-1 min^-1"), (0.4166666666666667, "M^-1 s^-1")])
        self.assertTrue(result.usable)
        self.assertAlmostEqual(result.values[0], result.values[1], places=12)


# ==========================================================================
class AMissingNumberStaysMissing(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rs = load_reference_set()

    def test_nd_is_not_zero(self) -> None:
        for lid in ("PNAS2015_Y190F_1", "PNAS2015_Y190F_2"):
            rec = self.rs.kinetic(lid)
            self.assertEqual(rec.label_type, "not_determined")
            self.assertEqual(rec.point_quantities(), ())
            for q in ("kcat", "km", "efficiency_reported", "efficiency_derived"):
                with self.assertRaisesRegex(ReferenceSetError, "not determined"):
                    rec.require(q)

    def test_a_less_than_bound_is_not_a_point_and_gives_no_efficiency(self) -> None:
        rec = self.rs.kinetic("PaHBDH_H150N_AAE_activity_only")
        self.assertEqual(rec.label_type, "kcat_only_km_bounded")
        self.assertEqual(rec.require("kcat"), 0.014)
        with self.assertRaisesRegex(ReferenceSetError, "bound is not a point"):
            rec.require("km")
        with self.assertRaisesRegex(ReferenceSetError, "bound"):
            rec.require("efficiency_derived")
        self.assertEqual(rec.km_comparator, "<")
        self.assertEqual(rec.km_original, 5400.0)
        self.assertEqual(rec.km_unit, "mM")          # kept as printed, not 'corrected'
        self.assertEqual(rec.default_use, "kcat_only_Km_bound_quarantined")

    def test_an_efficiency_only_record_has_no_kcat_to_hand_out(self) -> None:
        rec = self.rs.kinetic("PNAS2015_WT_1")
        self.assertEqual(rec.label_type, "reported_efficiency")
        with self.assertRaisesRegex(ReferenceSetError, "efficiency, not kcat"):
            rec.require("kcat")
        self.assertAlmostEqual(rec.require("efficiency_reported"), 0.025 * 1000 / 60,
                               places=12)

    def test_reported_and_derived_efficiency_are_two_fields_that_may_differ(self) -> None:
        rec = self.rs.kinetic("6ZZO_PaHBDH_AAE")
        self.assertEqual(rec.require("efficiency_reported"), 630000.0)
        self.assertAlmostEqual(rec.require("efficiency_derived"), 600000.0, places=6)
        self.assertGreater(rec.efficiency_discrepancy, 0.04)

    def test_a_row_without_a_reported_efficiency_has_only_the_derived_one(self) -> None:
        rec = self.rs.kinetic("DSTRII_TROP_NADPH_2003")
        with self.assertRaisesRegex(ReferenceSetError, "derived quotient is a different field"):
            rec.require("efficiency_reported")
        self.assertAlmostEqual(rec.require("efficiency_derived"), 71.4 / 9.49e-5, places=3)

    def test_the_four_label_types_are_distinct_and_never_share_a_unit_of_use(self) -> None:
        types = {k.label_id: k.label_type for k in self.rs.kinetics}
        self.assertEqual(types["6ZZO_PaHBDH_AAE"], "kcat_km_steady_state")
        self.assertEqual(types["Ssal2024_WT_1a"], "apparent_kcat_km_regeneration_system")
        self.assertEqual(types["PNAS2015_E145S_2"], "reported_efficiency")
        self.assertEqual(types["PaHBDH_H150N_AAE_activity_only"], "kcat_only_km_bounded")
        self.assertEqual(set(types.values()),
                         {"kcat_km_steady_state", "apparent_kcat_km_regeneration_system",
                          "reported_efficiency", "kcat_only_km_bounded", "not_determined"})

    def test_ranking_is_only_within_one_assay_group(self) -> None:
        groups = {k.assay_group for k in self.rs.kinetics}
        wt = self.rs.kinetic("6ZZO_PaHBDH_AAE").assay_group
        mutant = self.rs.kinetic("PaHBDH_H150A_AAE_activity_only").assay_group
        self.assertNotEqual(wt, mutant, "WT (2018 SI) and mutant (2020) rows are "
                                        "different papers and must not be ranked as one assay")
        self.assertGreater(len(groups), 5)

    def test_the_unsaturated_ssal_variant_is_a_sensitivity_record_only(self) -> None:
        rec = self.rs.kinetic("Ssal2024_M4_1a")
        self.assertEqual(rec.tier, "sensitivity")
        self.assertIn("saturation was not reached", rec.quality_flags)

    def test_crystal_cofactor_and_assay_cofactor_differences_are_stated_on_the_hbdh_rows(self) -> None:
        for lid in ("6ZZO_PaHBDH_AAE", "6ZZP_PaHBDH_QT8", "6ZZQ_AbHBDH_AAE", "6ZZS_AbHBDH_QT8"):
            self.assertIn("Crystal NAD+ differs from assay NADH", self.rs.kinetic(lid).quality_flags)

    def test_a_variant_with_no_matched_complex_has_none_listed(self) -> None:
        for lid in ("PaHBDH_H150A_AAE_activity_only", "AbHBDH_N145H_AAE_activity_only",
                    "PNAS2015_A94F_1", "Ssal2024_M3_1a"):
            self.assertFalse(self.rs.kinetic(lid).has_matched_complex, lid)


# ==========================================================================
class TheCrossReferencesAreFollowedBothWays(_Tmp):
    def test_a_structure_that_names_a_label_the_label_does_not_name_back_is_an_error(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "pdb_id", "1IPF", "linked_label_ids")
            row[col] = "DSTRII_TROP_NADPH_2003; LB_WT_AP_NADPH_2005"
        _rewrite_csv(base / "tables" / "structures.csv", edit)
        self.assertTrue(any("1IPF->LB_WT_AP_NADPH_2005" == f.subject
                            for f in self.findings(base, "link.one_way")))

    def test_a_label_that_names_a_structure_that_does_not_name_it_back_is_an_error(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "pdb_id", "6ZZO", "linked_label_ids")
            row[col] = "6ZZO_PaHBDH_AAE"
        _rewrite_csv(base / "tables" / "structures.csv", edit)
        found = self.findings(base, "link.one_way")
        self.assertTrue(any("PaHBDH_H150A_AAE_activity_only->6ZZO" == f.subject for f in found))

    def test_a_dangling_label_is_an_error(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "pdb_id", "1IPF", "linked_label_ids")
            row[col] = "DSTRII_TROP_NADPH_2003; NO_SUCH_LABEL"
        _rewrite_csv(base / "tables" / "structures.csv", edit)
        self.assertTrue(self.findings(base, "link.dangling"))

    def test_the_one_label_that_lives_in_the_auxiliary_sheet_is_the_one_listed(self) -> None:
        rs = load_reference_set()
        self.assertIn("SMBDH_WT_ACETOIN_2020", rs.structure("6XEW").linked_label_ids)

    def test_a_selectivity_row_cannot_claim_a_match_it_does_not_have(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "selectivity_id", "PNAS2015_Sph_1_er", "kinetic_variant_identity_match")
            row[col] = "1"
            row[rows[0].index("kinetic_label_id")] = "PNAS2015_Sph_1"
        _rewrite_csv(base / "tables" / "selectivity.csv", edit)
        self.assertTrue(self.findings(base, "selectivity.identity"))

    def test_the_nine_mutation_sph_er_is_not_joined_to_the_ten_mutation_efficiency(self) -> None:
        rs = load_reference_set()
        sel = next(s for s in rs.selectivity if s.selectivity_id == "PNAS2015_Sph_1_er")
        self.assertFalse(sel.variant_identity_match)
        self.assertEqual(sel.kinetic_label_id, "")
        self.assertEqual(sel.variant, "Sph_without_P194N")

    def test_a_density_summary_must_equal_the_validation_row_it_came_from(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "pdb_id", "1IPF", "selected_ligand_cofactor_RSCC_summary")
            row[col] = "0.999;0.965"
        _rewrite_csv(base / "tables" / "structures.csv", edit)
        self.assertTrue(self.findings(base, "rscc.unsupported"))

    def test_a_changed_count_contradicts_the_cover_note(self) -> None:
        base = _copy_set(self.tmp)
        _rewrite_csv(base / "tables" / "selectivity.csv", lambda rows: rows.pop())
        self.assertTrue(any(f.subject == "selectivity_records"
                            for f in self.findings(base, "counts.mismatch")))

    def test_an_unknown_grade_or_an_unknown_enzyme_is_refused_not_defaulted(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "pdb_id", "1IPF", "geometry_use")
            row[col] = "strict_geometry_gold"
        _rewrite_csv(base / "tables" / "structures.csv", edit)
        self.assertTrue(self.findings(base, "structures.geometry_use"))

        base2 = self.tmp / "set2"
        shutil.copytree(default_reference_dir(), base2)

        def edit2(rows):
            row, col = _cell(rows, "pdb_id", "1IPF", "enzyme")
            row[col] = "SomeNewEnzyme"
        _rewrite_csv(base2 / "tables" / "structures.csv", edit2)
        self.assertTrue(self.findings(base2, "structures.lineage"))

    def test_a_row_marked_as_not_matching_cannot_name_a_kinetic_record(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, _ = _cell(rows, "selectivity_id", "PNAS2015_Sph_1_er", "kinetic_label_id")
            row[rows[0].index("kinetic_label_id")] = "PNAS2015_Sph_1"
        _rewrite_csv(base / "tables" / "selectivity.csv", edit)
        self.assertTrue(self.findings(base, "selectivity.unmatched_linked"))

    def test_a_selectivity_row_claiming_a_record_that_does_not_exist_is_an_error(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, _ = _cell(rows, "selectivity_id", "PNAS2015_WT_1_er", "kinetic_label_id")
            row[rows[0].index("kinetic_label_id")] = "PNAS2015_WT_9"
        _rewrite_csv(base / "tables" / "selectivity.csv", edit)
        self.assertTrue(self.findings(base, "selectivity.link"))

    def test_the_auxiliary_link_must_exist_where_the_code_says_it_does(self) -> None:
        base = _copy_set(self.tmp)
        _rewrite_csv(base / "tables" / "auxiliary.csv",
                     lambda rows: [rows.remove(r) for r in list(rows)
                                   if r and r[0] == "SmBdh / 6XEW / WT"])
        self.assertTrue(self.findings(base, "link.auxiliary_missing"))

    def test_a_quarantined_bound_that_becomes_a_point_is_an_error(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "label_id", "PaHBDH_H150N_AAE_activity_only", "Km_comparator")
            row[col] = "eq"
        _rewrite_csv(base / "tables" / "kinetics.csv", edit)
        self.assertTrue(self.findings(base, "bound.use"))

    def test_a_point_row_that_is_quarantined_as_a_bound_is_an_error(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "label_id", "PaHBDH_H150A_AAE_activity_only", "default_use")
            row[col] = "kcat_only_Km_bound_quarantined"
        _rewrite_csv(base / "tables" / "kinetics.csv", edit)
        self.assertTrue(self.findings(base, "bound.use"))

    def test_assay_conditions_must_agree_with_the_kinetic_row(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "label_id", "6ZZO_PaHBDH_AAE", "temperature_C_nominal")
            row[col] = "25"
        _rewrite_csv(base / "tables" / "assay_conditions.csv", edit)
        self.assertTrue(self.findings(base, "assay.disagrees"))

        base2 = self.tmp / "set2"
        shutil.copytree(default_reference_dir(), base2)

        def edit2(rows):
            row, col = _cell(rows, "label_id", "6ZZO_PaHBDH_AAE", "cofactor")
            row[col] = "NADPH"
        _rewrite_csv(base2 / "tables" / "assay_conditions.csv", edit2)
        self.assertTrue(self.findings(base2, "assay.cofactor"))

    def test_a_duplicated_row_would_count_one_complex_twice_and_is_an_error(self) -> None:
        base = _copy_set(self.tmp)
        _rewrite_csv(base / "tables" / "structures.csv", lambda rows: rows.append(list(rows[1])))
        self.assertTrue(self.findings(base, "ids.duplicate"))

    def test_a_kinetic_record_citing_an_unlisted_source_is_an_error(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "label_id", "6ZZO_PaHBDH_AAE", "source_id")
            row[col] = "NoSuchSource"
        _rewrite_csv(base / "tables" / "kinetics.csv", edit)
        self.assertTrue(self.findings(base, "source.unknown"))

    def test_a_validation_row_for_an_entry_that_is_not_in_the_set_is_an_error(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            rows.append(list(rows[1]))
            rows[-1][0] = "9ZZZ"
        _rewrite_csv(base / "tables" / "ligand_validation.csv", edit)
        self.assertTrue(self.findings(base, "validation.dangling"))

    def test_a_scaffold_entry_that_names_a_reaction_ligand_is_an_error(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "pdb_id", "4RF2", "reaction_ligand_ccd")
            row[col] = "KET"
        _rewrite_csv(base / "tables" / "structures.csv", edit)
        self.assertTrue(self.findings(base, "grade.scaffold"))

    def test_a_pose_reference_without_a_reaction_ligand_is_an_error(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "pdb_id", "1IPF", "reaction_ligand_ccd")
            row[col] = ""
        _rewrite_csv(base / "tables" / "structures.csv", edit)
        self.assertTrue(self.findings(base, "grade.no_ligand"))

    def test_a_header_that_is_not_the_one_the_loader_was_written_for_is_refused(self) -> None:
        base = _copy_set(self.tmp)
        path = base / "tables" / "selectivity.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("er_S_over_R", "er_R_over_S", 1),
                        encoding="utf-8")
        with self.assertRaisesRegex(ReferenceSetError, "header is not the one"):
            load_reference_set(base, verify=False)

    def test_a_not_determined_row_that_acquires_a_number_is_an_error(self) -> None:
        base = _copy_set(self.tmp)

        def edit(rows):
            row, col = _cell(rows, "label_id", "PNAS2015_Y190F_1", "efficiency_original")
            row[col] = "0"
        _rewrite_csv(base / "tables" / "kinetics.csv", edit)
        self.assertTrue(self.findings(base, "nd.has_value"))


# ==========================================================================
class IndependenceIsComputedNotAssumed(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rs = load_reference_set()

    def test_nineteen_entries_are_six_lineages(self) -> None:
        ind = self.rs.independence()
        self.assertEqual(ind["structure_entries"], 19)
        self.assertEqual(ind["n_independent_structure_lineages"], 6)
        self.assertEqual(ind["structure_lineages"],
                         {"HBDH": 4, "LbADH/LkKRED": 9, "SmBdh": 1, "Ssal-KRED": 1,
                          "TR-II": 2, "TeSADH": 2})

    def test_twenty_two_core_records_are_three_lineages(self) -> None:
        ind = self.rs.independence()
        self.assertEqual(ind["core_records"], 22)
        self.assertEqual(ind["n_independent_core_lineages"], 3)

    def test_only_two_lineages_have_a_substrate_or_product_placed_in_the_site(self) -> None:
        self.assertEqual(self.rs.independence()["n_independent_pose_lineages"], 2)

    def test_the_lactobacillus_enzymes_share_a_lineage(self) -> None:
        self.assertEqual(LINEAGE_GROUPS["LbADH"], LINEAGE_GROUPS["LkKRED"])

    def test_every_enzyme_named_anywhere_has_a_lineage(self) -> None:
        names = ({s.enzyme for s in self.rs.structures} | {k.enzyme for k in self.rs.kinetics}
                 | {s.enzyme for s in self.rs.selectivity})
        self.assertEqual(names - set(LINEAGE_GROUPS), set())

    def test_every_tier_value_in_the_data_is_known(self) -> None:
        self.assertEqual({k.default_use for k in self.rs.kinetics} - set(TIERS), set())


if __name__ == "__main__":
    unittest.main()
