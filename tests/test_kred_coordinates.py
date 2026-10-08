"""The 19 pinned coordinate files, and what the workbook claims about them.

The reference set's grades are claims about files, and the RCSB revises files.
These tests defend the pins -- a hash and the facts read from the file for every
entry -- the family each entry's RCSB annotation places it in, the lineage groups
re-derived from the pinned sequences, and the checks that tie each cached file
back to the workbook (every one of the 118 validation rows must name a residue
that is really in the file). The download route itself is tested in
``test_structure_download``.

No test here touches the network. The full files are in the cache only if
somebody fetched them, and the tests that need them are skipped otherwise.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eagent.eval.kred_coordinates import (
    default_cache_dir, entry_facts, family_class, lineage_counts, lineage_findings,
    load_coordinates_manifest, verify_coordinates,
)
from eagent.eval.kred_reference import LINEAGE_GROUPS, default_reference_dir, load_reference_set

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "kred"
EXCERPT = (FIXTURES / "1IPF_excerpt.cif").read_bytes()


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)


# ==========================================================================
class FactsAreReadFromTheFileNotLookedUp(unittest.TestCase):
    def test_ligands_and_alternate_locations_are_listed_per_residue(self) -> None:
        text = (FIXTURES / "6ZZO_excerpt.cif").read_text(encoding="utf-8")
        facts = entry_facts(text, "6ZZO")
        by = {(l["chain"], l["resseq"], l["ccd"]): l for l in facts["ligands"]}
        self.assertEqual(by[("B", 302, "AAE")]["altlocs"], ["A", "B"])
        self.assertEqual(by[("B", 302, "AAE")]["occupancy_min"], 0.5)
        self.assertEqual(by[("B", 301, "NAD")]["altlocs"], [])

    def test_the_committed_pins_carry_a_hash_and_facts_for_every_entry(self) -> None:
        rs = load_reference_set()
        pins = load_coordinates_manifest(default_reference_dir())
        self.assertEqual(set(pins["files"]), {s.pdb_id for s in rs.structures})
        for pid, pin in pins["files"].items():
            self.assertRegex(pin["sha256"], r"^[0-9a-f]{64}$", pid)
            self.assertEqual(pin["facts"]["entry_id"], pid)
            self.assertTrue(pin["url"].endswith(f"/{pid}.cif"))

    def test_the_resolutions_in_the_files_are_the_workbooks(self) -> None:
        rs = load_reference_set()
        pins = load_coordinates_manifest(default_reference_dir())
        for s in rs.structures:
            self.assertAlmostEqual(pins["files"][s.pdb_id]["facts"]["resolution_a"],
                                   s.resolution_a, places=3, msg=s.pdb_id)


class FamilyIsTheRcsbsAnnotationNotAGuess(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pins = load_coordinates_manifest(default_reference_dir())

    def test_seventeen_entries_are_classical_sdr_one_is_not_and_two_are_zinc_adh(self) -> None:
        classes = {pid: p["family_class"] for pid, p in self.pins["files"].items()}
        self.assertEqual(sorted(p for p, c in classes.items() if c == "classical_sdr"),
                         sorted(set(classes) - {"1Y1P", "7UUT", "7UTC"}))
        self.assertEqual(classes["1Y1P"], "extended_sdr_epimerase_dehydratase")
        self.assertEqual(classes["7UUT"], "zinc_adh_mdr")
        self.assertEqual(classes["7UTC"], "zinc_adh_mdr")

    def test_the_stored_class_is_what_the_stored_annotation_gives(self) -> None:
        for pid, pin in self.pins["files"].items():
            self.assertEqual(pin["family_class"], family_class(pin["annotations"]), pid)

    def test_no_annotation_means_unclassified_not_a_member(self) -> None:
        self.assertEqual(family_class(None), "unclassified")
        self.assertEqual(family_class({"interpro": [], "pfam": [], "go": []}), "unclassified")

    def test_the_classes_are_by_identifier_and_zinc_binding_alone(self) -> None:
        self.assertEqual(family_class({"interpro": [{"id": "IPR002347", "name": ""}],
                                       "pfam": [], "go": []}), "classical_sdr")
        self.assertEqual(family_class({"interpro": [], "pfam": [{"id": "PF00106", "name": ""}],
                                       "go": []}), "classical_sdr")
        self.assertEqual(family_class({"interpro": [], "pfam": [],
                                       "go": [{"id": "GO:0008270", "name": "zinc ion binding"}]}),
                         "zinc_adh_mdr")
        self.assertEqual(family_class({"interpro": [{"id": "IPR023210", "name": ""}],
                                       "pfam": [], "go": []}), "aldo_keto_reductase")


# ==========================================================================
class LineagesAreDerivedFromThePinnedSequences(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rs = load_reference_set()
        cls.pins = load_coordinates_manifest(default_reference_dir())

    def test_the_declared_groups_agree_with_the_measured_identities(self) -> None:
        errors = [f for f in lineage_findings(self.rs, self.pins) if f.severity == "error"]
        self.assertEqual(errors, [])

    def test_leaving_the_lactobacillus_enzymes_apart_is_an_error(self) -> None:
        """LbADH and LkKRED are about 80 % identical: two draws of one lineage."""
        with mock.patch.dict(LINEAGE_GROUPS, {"LkKRED": "LkKRED"}):
            errors = [f for f in lineage_findings(self.rs, self.pins)
                      if f.code == "lineage.merge_required"]
        self.assertEqual(len(errors), 1)
        self.assertIn("LbADH / LkKRED", errors[0].subject)
        self.assertRegex(errors[0].message, r"8\d% identical")

    def test_entries_of_one_enzyme_are_the_same_protein(self) -> None:
        self.assertEqual([f for f in lineage_findings(self.rs, self.pins)
                          if f.code == "lineage.same_enzyme_differs"], [])

    def test_the_count_of_independent_groups_at_three_identity_levels(self) -> None:
        counts = lineage_counts(self.rs, self.pins)
        self.assertEqual({k: v["n_groups"] for k, v in counts.items()},
                         {"0.30": 4, "0.40": 6, "0.50": 6})
        self.assertIn(["LbADH", "LkKRED"], counts["0.40"]["groups"])
        self.assertIn(["AbHBDH", "PaHBDH"], counts["0.40"]["groups"])

    def test_the_pairs_the_threshold_separates_are_reported_as_such(self) -> None:
        near = [f for f in lineage_findings(self.rs, self.pins)
                if f.code == "lineage.near_threshold"]
        self.assertTrue(near)


# ==========================================================================
class TheCachedFilesAreCheckedAgainstTheirPinsAndTheWorkbook(_Tmp):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rs = load_reference_set()
        cls.pins = load_coordinates_manifest(default_reference_dir())

    def test_a_missing_file_is_a_finding_for_that_entry_only(self) -> None:
        findings, structures = verify_coordinates(self.rs, self.root / "nowhere", self.pins)
        self.assertEqual(structures, {})
        self.assertEqual({f.code for f in findings}, {"coordinates.absent"})
        self.assertEqual(len(findings), 19)

    def test_a_different_file_under_the_right_name_is_a_hash_error(self) -> None:
        cache = self.root / "cache"
        cache.mkdir()
        (cache / "1IPF.cif").write_bytes(EXCERPT)
        findings, structures = verify_coordinates(self.rs, cache, self.pins)
        mine = [f for f in findings if f.subject == "1IPF"]
        self.assertEqual([f.code for f in mine], ["coordinates.hash"])
        self.assertNotIn("1IPF", structures)

    @unittest.skipUnless((default_cache_dir() / "1IPF.cif").is_file()
                         and len(list(default_cache_dir().glob("*.cif"))) == 19,
                         "the 19 coordinate files have not been fetched")
    def test_the_fetched_files_satisfy_every_claim_the_workbook_makes(self) -> None:
        findings, structures = verify_coordinates(self.rs, default_cache_dir(), self.pins)
        self.assertEqual([f for f in findings if f.severity == "error"], [])
        self.assertEqual(len(structures), 19)
        # the one thing worth a line: 7UUT's deposited cofactor is oxidised NADP
        self.assertEqual([(f.severity, f.subject) for f in findings],
                         [("warning", "7UUT")])

    @unittest.skipUnless((default_cache_dir() / "6ZZO.cif").is_file(),
                         "the coordinate files have not been fetched")
    def test_a_validation_row_for_an_altloc_the_residue_does_not_have_is_an_error(self) -> None:
        import dataclasses

        rows = tuple(dataclasses.replace(r, altloc="C")
                     if (r.pdb_id, r.ccd, r.altloc) == ("6ZZO", "AAE", "A") else r
                     for r in self.rs.ligand_validation)
        rs = dataclasses.replace(self.rs, ligand_validation=rows)
        findings, _ = verify_coordinates(rs, default_cache_dir(), self.pins, only=["6ZZO"])
        self.assertTrue(any(f.code == "coordinates.altloc" and f.subject.startswith("6ZZO:")
                            for f in findings))

    @unittest.skipUnless((default_cache_dir() / "1IPF.cif").is_file(),
                         "the coordinate files have not been fetched")
    def test_a_validation_row_for_a_residue_the_file_does_not_have_is_an_error(self) -> None:
        import dataclasses

        rows = tuple(dataclasses.replace(r, residue_number=999)
                     if (r.pdb_id, r.ccd) == ("1IPF", "TNE") and r.chain == "A" else r
                     for r in self.rs.ligand_validation)
        rs = dataclasses.replace(self.rs, ligand_validation=rows)
        findings, _ = verify_coordinates(rs, default_cache_dir(), self.pins, only=["1IPF"])
        self.assertTrue(any(f.code == "coordinates.validation_row" for f in findings))

    @unittest.skipUnless((default_cache_dir() / "1IPF.cif").is_file(),
                         "the coordinate files have not been fetched")
    def test_a_workbook_that_misdescribes_the_cofactor_is_an_error_in_state_and_in_identity(self) -> None:
        import dataclasses

        for wrong, why in (("NADP+", "oxidation state"), ("NADH", "identity")):
            structures = tuple(dataclasses.replace(s, cofactor_state=wrong)
                               if s.pdb_id == "1IPF" else s for s in self.rs.structures)
            rs = dataclasses.replace(self.rs, structures=structures)
            findings, _ = verify_coordinates(rs, default_cache_dir(), self.pins, only=["1IPF"])
            self.assertTrue(any(f.code == "coordinates.cofactor_state" and f.severity == "error"
                                and f.subject == "1IPF" for f in findings), why)

    @unittest.skipUnless((default_cache_dir() / "1IPF.cif").is_file(),
                         "the coordinate files have not been fetched")
    def test_a_one_byte_edit_to_a_cached_file_is_a_hash_error(self) -> None:
        cache = self.root / "cache"
        cache.mkdir()
        data = (default_cache_dir() / "1IPF.cif").read_bytes()
        (cache / "1IPF.cif").write_bytes(data.replace(b"2.50", b"2.51", 1))
        findings, _ = verify_coordinates(self.rs, cache, self.pins, only=["1IPF"])
        self.assertEqual([f.code for f in findings if f.subject == "1IPF"],
                         ["coordinates.hash"])


if __name__ == "__main__":
    unittest.main()
