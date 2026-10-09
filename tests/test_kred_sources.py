"""The workbook's kinetic numbers against the tables they were taken from.

Two layers. The parsers and the comparison are tested on small documents built
here (from the workbook's own values, so that "everything matches" is a real
expectation, and then damaged one number at a time to show each kind of
disagreement is caught). The committed ``verification/results.json`` -- the
record of the real run on real documents -- is tested for what it claims and for
agreeing with the tables it is about.

No test fetches anything, and no document is committed: the record holds the
sha256 of the bytes that were read.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
import unittest
from pathlib import Path

from eagent.eval.kred_reference import default_reference_dir, load_reference_set
from eagent.eval.kred_sources import (
    DOCUMENTS, SourceVerificationError, brenda_entries, html_tables, jats_tables,
    parse_value, si_rows, summarise, verify_sources,
)

RESULTS = default_reference_dir() / "verification" / "results.json"
DELIVERED_WORKBOOK_SHA256 = (
    "c0eac78f178ecf0a82c90f3b451ddddaef706ccd3a763ce0529a679ad25fbf52")
PM = "±"
SYMBOL_PM = ""          # what pdftotext leaves of a Symbol-font plus-minus


# ==========================================================================
# printed numbers
# ==========================================================================
class PrintedValuesAreReadAsPrinted(unittest.TestCase):
    def test_a_value_with_its_error(self) -> None:
        c = parse_value(f"30 {PM} 1")
        self.assertEqual((c.value, c.error, c.comparator, c.nd), (30.0, 1.0, "", False))

    def test_trailing_zeros_are_the_same_number(self) -> None:
        self.assertEqual(parse_value(f"0.0140 {PM} 0.0004").value, 0.014)

    def test_a_less_than_bound_keeps_its_comparator_and_has_no_error(self) -> None:
        c = parse_value("<5400")
        self.assertEqual((c.value, c.comparator, c.error), (5400.0, "<", None))

    def test_nd_is_not_a_number(self) -> None:
        for text in ("ND", "N.D.", "n.d."):
            c = parse_value(text)
            self.assertTrue(c.nd, text)
            self.assertIsNone(c.value)

    def test_a_bare_value_has_no_error(self) -> None:
        self.assertEqual(parse_value("41").error, None)

    def test_footnote_markers_do_not_become_part_of_the_number(self) -> None:
        c = parse_value("1.3 * , †")
        self.assertEqual(c.value, 1.3)
        self.assertIn("*", c.text)

    def test_unicode_minus_is_a_minus(self) -> None:
        self.assertEqual(parse_value("−0.85").value, -0.85)

    def test_text_that_is_not_a_number_has_no_value(self) -> None:
        self.assertIsNone(parse_value("WT").value)


class TheParsersReadOneLayoutEach(unittest.TestCase):
    def test_a_jats_table_is_found_by_its_label(self) -> None:
        xml = ("<article><table-wrap><label>Table 2</label><table>"
               "<tr><th>variant</th><th>KM</th></tr>"
               f"<tr><td>WT</td><td>11.9 {PM} 1.1</td></tr>"
               "</table></table-wrap></article>")
        rows = jats_tables(xml)["Table 2"]
        self.assertEqual(rows[1], ["WT", f"11.9 {PM} 1.1"])

    def test_a_jats_document_that_declares_an_entity_is_refused(self) -> None:
        with self.assertRaises(SourceVerificationError):
            jats_tables('<!DOCTYPE x [<!ENTITY a "b">]><article/>')

    def test_an_html_table_is_read_cell_by_cell_with_nested_markup_flattened(self) -> None:
        page = ("<html><body><table><tr><th>Variant</th><th>x</th></tr>"
                "<tr><td>WT</td><td>k<sub>cat</sub> 0.025</td></tr></table>"
                "<table><tr><td>second</td></tr></table></body></html>")
        tables = html_tables(page)
        self.assertEqual(len(tables), 2)
        self.assertEqual(tables[0][1], ["WT", "kcat 0.025"])

    def test_an_si_row_is_read_through_the_symbol_font_plus_minus(self) -> None:
        text = ("\fTable S2. Steady-state kinetic constants\n\n"
                "    Parameter    kcat (s-1)\n"
                f"      283 K      30 {SYMBOL_PM} 1        0.05 {SYMBOL_PM} 0.01     "
                f"0.0064 {SYMBOL_PM} 0.0002     6.3 {SYMBOL_PM} 0.8 × 105   "
                f"4.6 {SYMBOL_PM} 0.2 × 106\n"
                f"      288 K      42 {SYMBOL_PM} 1        0.06 {SYMBOL_PM} 0.01     "
                f"0.010 {SYMBOL_PM} 0.001     7.8 {SYMBOL_PM} 1.1 × 105\n")
        row = si_rows(text, "S2")
        self.assertEqual((row["kcat"], row["kcat_error"], row["km"], row["km_error"]),
                         (30.0, 1.0, 0.05, 0.01))
        self.assertAlmostEqual(row["efficiency"], 630000.0, places=6)
        self.assertAlmostEqual(row["efficiency_error"], 80000.0, places=6)

    def test_the_283_k_row_is_the_one_read_not_the_next(self) -> None:
        text = ("Table S9. x\n      288 K   1  1  1  1  1  1  1  1 × 105\n"
                "      283 K   2  1  2  1  2  1  2  1 × 105\n")
        self.assertEqual(si_rows(text, "S9")["kcat"], 2.0)

    def test_a_table_that_is_not_there_gives_nothing_not_a_neighbour(self) -> None:
        self.assertEqual(si_rows("Table S2. x\n      283 K  1  1\n", "S5"), {})
        # a row with too few numbers is not guessed at
        self.assertEqual(si_rows("Table S2. x\n      283 K  1  1\n", "S2"), {})

    def test_brenda_entries_carry_their_substrate_and_conditions(self) -> None:
        page = ("<div>KM Value [mM]</div><div>KM Value Maximum [mM]</div>"
                "<div>0.0949</div><div>tropinone</div><div>pH 7.5, 15°C</div>"
                "<div>Turnover Number [1/s]</div><div>Turnover Number Minimum [1/s]</div>"
                "<div>71.4</div><div>tropinone</div><div>pH 7.5, 15°C</div>"
                "<div>pH Optimum</div><div>8.0</div>")
        entries = brenda_entries(page)
        self.assertEqual([(e.section, e.value) for e in entries],
                         [("km", 0.0949), ("kcat", 71.4)])
        self.assertIn("pH 7.5, 15°C", entries[0].text())
        self.assertNotIn(8.0, [e.value for e in entries])


# ==========================================================================
# the comparison, on documents built from the workbook's own numbers
# ==========================================================================
def _g(x: float) -> str:
    return f"{x:g}"


def build_documents(rs, folder: Path) -> None:
    kin = {k.label_id: k for k in rs.kinetics}
    # Ssal-KRED Table 2
    rows = ["<tr><th>variant</th><th>KM</th><th>kcat</th><th>eff</th><th>rel</th></tr>"]
    for k in rs.kinetics:
        if k.source_id == "Ssal2024":
            name = k.variant + ("a" if k.variant == "M4" else "")
            rows.append(f"<tr><td>{name}</td><td>{_g(k.km_original)} {PM} "
                        f"{k.km_uncertainty_original}</td><td>{_g(k.kcat_original)} {PM} "
                        f"{k.kcat_uncertainty_original}</td>"
                        f"<td>{_g(k.efficiency_original)}</td><td>1</td></tr>")
    (folder / "PMC10902378.xml").write_text(
        "<article><table-wrap><label>Table 2</label><table>" + "".join(rows)
        + "</table></table-wrap></article>", encoding="utf-8")
    # HBDH 2020 Table 4 (mutants, and the two WT rows that cite the 2018 paper)
    rows = ["<tr><th>HBDH</th><th>kcat</th><th>KM</th><th>KM NADH</th></tr>"]
    for k in rs.kinetics:
        if k.source_id == "HBDH2020":
            km = (f"&lt;{_g(k.km_original)}" if k.km_comparator == "<"
                  else f"{_g(k.km_original)} {PM} {k.km_uncertainty_original}")
            rows.append(f"<tr><td>{k.variant}-{k.enzyme}</td><td>{_g(k.kcat_original)} {PM} "
                        f"{k.kcat_uncertainty_original}</td><td>{km}</td><td>x</td></tr>")
    (folder / "PMC7773212.xml").write_text(
        "<article><table-wrap><label>Table 4</label><table>" + "".join(rows)
        + "</table></table-wrap></article>", encoding="utf-8")
    # PNAS 2015 Table 1
    sel = {s.selectivity_id: s for s in rs.selectivity}
    out = ["<tr><th>Variant</th><th>a</th><th>b</th></tr>",
           "<tr><td>er S/R</td><td>ddG</td><td>k</td><td>er S/R</td><td>ddG</td><td>k</td></tr>"]
    for variant in ("WT", "A94F", "Y190F", "E145S", "Sph"):
        cells = [variant]
        for n in (1, 2):
            er = sel[f"PNAS2015_{variant}_{n}_er"].er_s_over_r
            rec = kin[f"PNAS2015_{variant}_{n}"]
            eff = "ND" if rec.record_status == "not_determined" else _g(rec.efficiency_original)
            cells += [_g(er) + (" * , †" if variant == "Sph" and n == 1 else ""), "0", eff]
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    (folder / "PMC4697376.html").write_text(
        "<html><table>" + "".join(out) + "</table></html>", encoding="utf-8")
    # HBDH 2018 SI, 283 K rows
    parts = []
    for table, label in (("S1", "6ZZQ_AbHBDH_AAE"), ("S2", "6ZZO_PaHBDH_AAE"),
                         ("S3", "6ZZS_AbHBDH_QT8"), ("S4", "6ZZP_PaHBDH_QT8")):
        k = kin[label]
        exp = int(math.floor(math.log10(k.efficiency_original)))
        mant = k.efficiency_original / 10 ** exp
        err = float(k.efficiency_uncertainty_original) / 10 ** exp
        parts.append(
            f"\fTable {table}. Steady-state kinetic constants\n\n    Parameter\n"
            f"      283 K     {_g(k.kcat_original)} {SYMBOL_PM} {k.kcat_uncertainty_original}   "
            f"{_g(k.km_original)} {SYMBOL_PM} {k.km_uncertainty_original}   "
            f"0.01 {SYMBOL_PM} 0.001   {round(mant, 6):g} {SYMBOL_PM} {round(err, 6):g} "
            f"× 10{exp}   1.0 {SYMBOL_PM} 0.1 × 106\n")
    (folder / "hbdh2018_si.txt").write_text("\n".join(parts), encoding="utf-8")
    # BRENDA
    def brenda(entries_km, entries_kcat) -> str:
        def block(title, items):
            return (f"<div>{title}</div>" + "".join(
                f"<div>{v}</div><div>{s}</div><div>{c}</div>" for v, s, c in items))
        return (block("KM Value [mM]", entries_km)
                + block("Turnover Number [1/s]", entries_kcat))
    tr = kin["DSTRII_TROP_NADPH_2003"]
    (folder / "brenda_654707.html").write_text(
        brenda([(_g(tr.km_original), "tropinone", "pH 7.5, 15°C")],
               [(_g(tr.kcat_original), "tropinone", "pH 7.5, 15°C")]), encoding="utf-8")
    wt, mu = kin["LB_WT_AP_NADPH_2005"], kin["LB_G37D_AP_NADH_2005"]
    (folder / "brenda_675348.html").write_text(
        brenda([(_g(wt.km_original), "acetophenone",
                 "cofactor NADPH, reduction, recombinante wild-type, 30°C, pH 7.0"),
                (_g(mu.km_original), "acetophenone",
                 "cofactor NADH, reduction, recombinante mutant G37D, 30°C, pH 7.0")],
               [(_g(wt.kcat_original), "acetophenone",
                 "cofactor NADPH, reduction, recombinante wild-type, 30°C, pH 7.0"),
                (_g(mu.kcat_original), "acetophenone",
                 "cofactor NADH, reduction, recombinante mutant G37Dm 30°C, pH 7.0")]),
        encoding="utf-8")


class TheComparisonCatchesEachKindOfDisagreement(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rs = load_reference_set()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.docs = Path(self._tmp.name)
        build_documents(self.rs, self.docs)

    def run_check(self) -> dict:
        return verify_sources(self.rs, self.docs)

    def statuses(self, report: dict, label: str) -> dict[str, str]:
        return {c["quantity"]: c["status"] for c in report["kinetic_records"][label]}

    def test_documents_built_from_the_workbooks_own_values_match_everywhere(self) -> None:
        report = self.run_check()
        self.assertEqual(report["summary"]["kinetic_quantities"],
                         {"matches": 57, "mismatch": 0, "not_found": 0, "not_checked": 6})
        self.assertEqual(report["summary"]["selectivity_quantities"]["matches"], 10)

    def test_a_changed_michaelis_constant_is_a_mismatch_naming_the_quantity(self) -> None:
        path = self.docs / "PMC10902378.xml"
        path.write_text(path.read_text(encoding="utf-8").replace(f"162.8 {PM}", f"182.8 {PM}"),
                        encoding="utf-8")
        report = self.run_check()
        self.assertEqual(self.statuses(report, "Ssal2024_M1_1a")["Km"], "mismatch")
        self.assertEqual(self.statuses(report, "Ssal2024_M1_1a")["kcat"], "matches")
        self.assertEqual(report["summary"]["records_with_a_mismatch_or_missing_value"],
                         ["Ssal2024_M1_1a"])

    def test_a_changed_error_term_is_a_mismatch_too(self) -> None:
        path = self.docs / "PMC7773212.xml"
        path.write_text(path.read_text(encoding="utf-8").replace(f"0.54 {PM} 0.01", f"0.54 {PM} 0.02"),
                        encoding="utf-8")
        self.assertEqual(self.statuses(self.run_check(), "PaHBDH_H150A_AAE_activity_only")["kcat"],
                         "mismatch")

    def test_a_bound_printed_as_a_point_is_a_mismatch(self) -> None:
        path = self.docs / "PMC7773212.xml"
        path.write_text(path.read_text(encoding="utf-8").replace("<td>&lt;5400</td>", "<td>5400</td>"),
                        encoding="utf-8")
        self.assertEqual(self.statuses(self.run_check(), "PaHBDH_H150N_AAE_activity_only")["Km"],
                         "mismatch")

    def test_nd_in_the_source_against_a_number_in_the_workbook_is_a_mismatch(self) -> None:
        path = self.docs / "PMC4697376.html"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("<td>ND</td>", "<td>0.5</td>", 1), encoding="utf-8")
        report = self.run_check()
        self.assertEqual(self.statuses(report, "PNAS2015_Y190F_1")["efficiency"], "mismatch")

    def test_a_source_number_against_nd_in_the_workbook_is_a_mismatch_too(self) -> None:
        path = self.docs / "PMC4697376.html"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("<td>0.025</td>", "<td>ND</td>", 1), encoding="utf-8")
        self.assertEqual(self.statuses(self.run_check(), "PNAS2015_WT_1")["efficiency"],
                         "mismatch")

    def test_a_missing_row_is_not_found_and_never_matches(self) -> None:
        path = self.docs / "PMC10902378.xml"
        text = path.read_text(encoding="utf-8")
        path.write_text(re.sub(r"<tr><td>M3</td>.*?</tr>", "", text), encoding="utf-8")
        report = self.run_check()
        self.assertEqual(set(self.statuses(report, "Ssal2024_M3_1a").values()), {"not_found"})

    def test_a_missing_si_row_is_not_found(self) -> None:
        path = self.docs / "hbdh2018_si.txt"
        path.write_text(path.read_text(encoding="utf-8").replace("283 K", "293 K"), encoding="utf-8")
        report = self.run_check()
        self.assertEqual(set(self.statuses(report, "6ZZO_PaHBDH_AAE").values()), {"not_found"})

    def test_a_changed_efficiency_exponent_in_the_si_is_a_mismatch(self) -> None:
        path = self.docs / "hbdh2018_si.txt"
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("× 105", "× 106", 1), encoding="utf-8")
        report = self.run_check()
        self.assertIn("mismatch", self.statuses(report, "6ZZO_PaHBDH_AAE").values())

    def test_a_brenda_value_that_is_not_on_the_page_is_not_found(self) -> None:
        path = self.docs / "brenda_654707.html"
        path.write_text(path.read_text(encoding="utf-8").replace("71.4", "17.4"), encoding="utf-8")
        report = self.run_check()
        self.assertEqual(self.statuses(report, "DSTRII_TROP_NADPH_2003")["kcat"], "not_found")

    def test_the_wild_type_and_mutant_lbadh_values_are_not_swapped(self) -> None:
        path = self.docs / "brenda_675348.html"
        text = path.read_text(encoding="utf-8")
        swapped = text.replace("wild-type", "@@").replace("mutant G37D", "wild-type").replace("@@", "mutant G37D")
        path.write_text(swapped, encoding="utf-8")
        report = self.run_check()
        self.assertEqual(self.statuses(report, "LB_WT_AP_NADPH_2005")["Km"], "not_found")

    def test_what_was_never_read_is_not_checked_and_says_why(self) -> None:
        checks = self.run_check()["kinetic_records"]["LB_WT_AP_NADPH_2005"]
        unread = [c for c in checks if c["status"] == "not_checked"]
        self.assertEqual({c["quantity"] for c in unread},
                         {"efficiency", "kcat_uncertainty", "Km_uncertainty"})
        self.assertTrue(all(c["note"] for c in unread))

    def test_a_curated_database_is_not_counted_as_the_papers_own_table(self) -> None:
        summary = self.run_check()["summary"]
        self.assertEqual(summary["records_fully_matched_against_a_curated_database_only"],
                         ["DSTRII_TROP_NADPH_2003"])
        self.assertNotIn("DSTRII_TROP_NADPH_2003",
                         summary["records_fully_matched_against_the_papers_own_table"])
        self.assertEqual(len(summary["records_fully_matched_against_the_papers_own_table"]), 25)

    def test_a_missing_document_stops_the_run_instead_of_being_skipped(self) -> None:
        (self.docs / "brenda_675348.html").unlink()
        with self.assertRaisesRegex(SourceVerificationError, "does not fetch them"):
            self.run_check()

    def test_the_record_names_the_workbook_and_hashes_every_document_read(self) -> None:
        report = self.run_check()
        self.assertEqual(report["workbook_sha256"], DELIVERED_WORKBOOK_SHA256)
        self.assertEqual(set(report["documents"]), {d.key for d in DOCUMENTS})
        for doc in report["documents"].values():
            self.assertRegex(doc["sha256"], r"^[0-9a-f]{64}$")


# ==========================================================================
# the committed record
# ==========================================================================
class TheCommittedRecordIsTheRealRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(RESULTS.read_text(encoding="utf-8"))
        cls.rs = load_reference_set()

    def test_it_is_about_the_delivered_workbook(self) -> None:
        self.assertEqual(self.report["workbook_sha256"], DELIVERED_WORKBOOK_SHA256)

    def test_every_kinetic_record_and_every_er_value_has_an_entry(self) -> None:
        self.assertEqual(set(self.report["kinetic_records"]), {k.label_id for k in self.rs.kinetics})
        self.assertEqual(set(self.report["selectivity_records"]),
                         {s.selectivity_id for s in self.rs.selectivity})

    def test_nothing_mismatched_and_nothing_was_missing(self) -> None:
        q = self.report["summary"]["kinetic_quantities"]
        self.assertEqual((q["mismatch"], q["not_found"]), (0, 0))
        self.assertEqual((q["matches"], q["not_checked"]), (57, 6))
        self.assertEqual(self.report["summary"]["selectivity_quantities"],
                         {"matches": 10, "mismatch": 0, "not_found": 0, "not_checked": 0})

    def test_the_split_between_the_papers_table_and_a_database_is_as_stated(self) -> None:
        s = self.report["summary"]
        self.assertEqual(len(s["records_fully_matched_against_the_papers_own_table"]), 25)
        self.assertEqual(s["records_fully_matched_against_a_curated_database_only"],
                         ["DSTRII_TROP_NADPH_2003"])
        self.assertEqual(s["records_matched_in_part_rest_not_checked"],
                         ["LB_G37D_AP_NADH_2005", "LB_WT_AP_NADPH_2005"])
        self.assertEqual(s["records_with_a_mismatch_or_missing_value"], [])

    def test_the_summary_is_what_the_checks_add_up_to(self) -> None:
        again = summarise(self.report["kinetic_records"], self.report["selectivity_records"])
        self.assertEqual(again, self.report["summary"])

    def test_six_documents_each_with_a_hash_and_a_primary_flag(self) -> None:
        docs = self.report["documents"]
        self.assertEqual(set(docs), {d.key for d in DOCUMENTS})
        self.assertEqual({k for k, d in docs.items() if not d["primary"]},
                         {"TRII_BRENDA", "LbADH_BRENDA"})
        for d in docs.values():
            self.assertRegex(d["sha256"], r"^[0-9a-f]{64}$")

    def test_what_the_record_says_the_workbook_holds_is_what_the_tables_hold(self) -> None:
        for label, checks in self.report["kinetic_records"].items():
            rec = self.rs.kinetic(label)
            for c in checks:
                if c["status"] != "matches":
                    continue
                wb = c["workbook"].replace("<", "").split(PM)[0].strip()
                expected = {"kcat": rec.kcat_original, "Km": rec.km_original,
                            "efficiency": rec.efficiency_original}.get(c["quantity"])
                if wb == "ND":
                    self.assertIsNone(expected, f"{label} {c['quantity']}")
                elif wb:
                    self.assertAlmostEqual(float(wb), expected, places=9, msg=f"{label} {c['quantity']}")

    def test_the_lbadh_gaps_are_named_not_hidden(self) -> None:
        for label in ("LB_WT_AP_NADPH_2005", "LB_G37D_AP_NADH_2005"):
            unread = {c["quantity"]: c for c in self.report["kinetic_records"][label]
                      if c["status"] == "not_checked"}
            self.assertEqual(set(unread), {"efficiency", "kcat_uncertainty", "Km_uncertainty"})
            self.assertIn("thesis", unread["efficiency"]["note"])
            self.assertIn("neither was opened", unread["efficiency"]["note"])
            self.assertIn("no error terms", unread["kcat_uncertainty"]["note"])


if __name__ == "__main__":
    unittest.main()
