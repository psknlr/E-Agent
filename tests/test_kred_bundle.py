"""The second delivery: ingested by a stated rule, and checked against the first.

The archive is not committed (46 MB of it is somebody else's re-fetchable
files), so these tests work two ways: against the committed bundle for what it
holds, and against small synthetic archives built here for the import path and
for every way an archive can fail to be the one the importer was written for.

What is pinned:

* the disposition rule is a function of the archive's own label, with one
  explicit list of exceptions, and nothing else;
* the import refuses rather than writes when a check fails -- a spreadsheet that
  is not the committed one, an aggregate that is not really a duplicate, a
  member outside the single root, an unsafe member name;
* the two exports of the five shared tables agree in every shared field, and
  the three representation conventions that differ are normalised rather than
  special-cased per value;
* the 19 coordinate files agree with this project's independent pins, and a
  disagreement would be reported as a revision rather than silently preferred
  one way.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from eagent.eval.kred_bundle import (
    ARCHIVE_MANIFEST, ARCHIVE_ROOT, ARCHIVE_SHA256, BUNDLE_DIRNAME,
    COMMITTED_UPSTREAM, DERIVED_FILES, DROPPED_DUPLICATES, BundleError,
    bundle_dir, coordinate_agreement, cross_check_tables, disposition_for,
    import_archive, read_archive, verify_archive_manifest, verify_bundle,
)
from eagent.eval.kred_reference import default_reference_dir, load_reference_set

BUNDLE = bundle_dir()
MANIFEST = json.loads((BUNDLE / ARCHIVE_MANIFEST).read_text(encoding="utf-8"))
MEMBERS = MANIFEST["members"]


def member_bytes(path: str) -> bytes:
    return (BUNDLE / path).read_bytes()


class _Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def build_archive(self, members: dict[str, bytes], *,
                      kinds: dict[str, str] | None = None,
                      urls: dict[str, str] | None = None,
                      root: str = ARCHIVE_ROOT,
                      manifest: object = "auto") -> Path:
        """A small archive shaped like the delivered one."""
        kinds = kinds or {}
        urls = urls or {}
        if manifest == "auto":
            manifest = {
                "audit_date": "2026-10-07", "hash_algorithm": "SHA256",
                "paths_relative_to": "archive root",
                "files": [
                    {"path": p, "bytes": len(b),
                     "sha256": hashlib.sha256(b).hexdigest(),
                     "source_url": urls.get(p),
                     "kind": kinds.get(p, "curated_or_supporting_artifact")}
                    for p, b in sorted(members.items())],
            }
        path = self.tmp / "archive.zip"
        with zipfile.ZipFile(path, "w") as zf:
            for p, b in members.items():
                zf.writestr(root + p, b)
            if manifest is not None:
                zf.writestr(root + "file_manifest.json",
                            json.dumps(manifest).encode("utf-8"))
        return path

    def reference_copy(self) -> Path:
        """A reference set with source/ and tables/ but no bundle/."""
        target = self.tmp / "set"
        shutil.copytree(default_reference_dir(), target)
        shutil.rmtree(target / BUNDLE_DIRNAME)
        return target


# ==========================================================================
class TheDispositionRuleIsTheArchivesOwnLabel(unittest.TestCase):
    def test_curated_is_committed_and_upstream_is_pinned(self) -> None:
        self.assertEqual(disposition_for("data/x.json", "curated_or_supporting_artifact"),
                         "committed")
        self.assertEqual(disposition_for("structures/1IPF/1IPF.cif", "upstream_original"),
                         "pinned")

    def test_the_committed_upstream_exception_is_an_explicit_list(self) -> None:
        for path in COMMITTED_UPSTREAM:
            self.assertEqual(disposition_for(path, "upstream_original"),
                             "committed_upstream", path)
        self.assertEqual(disposition_for("activity_expansion_2026/other.csv",
                                         "upstream_original"), "pinned")

    def test_the_spreadsheet_is_a_duplicate_of_source_whatever_its_label(self) -> None:
        for kind in ("curated_or_supporting_artifact", "upstream_original"):
            self.assertEqual(
                disposition_for("KRED_Calibration_Reference_v0.1.xlsx", kind),
                "duplicate_of_source")

    def test_a_label_the_importer_has_no_rule_for_is_refused(self) -> None:
        with self.assertRaisesRegex(BundleError, "no rule for"):
            disposition_for("data/x.json", "something_new")

    def test_every_committed_file_in_the_bundle_was_dispositioned_that_way(self) -> None:
        for rel, info in MEMBERS.items():
            on_disk = (BUNDLE / rel).is_file()
            if info["disposition"] in ("committed", "committed_upstream"):
                self.assertTrue(on_disk, f"{rel} should be committed")
            else:
                self.assertFalse(on_disk, f"{rel} should not be committed")

    def test_the_rule_is_what_produced_the_stored_dispositions(self) -> None:
        for rel, info in MEMBERS.items():
            self.assertEqual(disposition_for(rel, info["kind"]), info["disposition"], rel)


class TheBundleIsWhatWasDelivered(unittest.TestCase):
    def test_the_manifest_records_the_archive_this_importer_ran_on(self) -> None:
        self.assertEqual(MANIFEST["archive_sha256"], ARCHIVE_SHA256)
        self.assertEqual(MANIFEST["archive_audit_date"], "2026-10-07")

    def test_the_counts_are_the_ones_the_documentation_states(self) -> None:
        self.assertEqual(MANIFEST["disposition_counts"],
                         {"committed": 54, "committed_upstream": 8,
                          "dropped_duplicate": 1, "duplicate_of_source": 1,
                          "pinned": 129})
        self.assertEqual(sum(MANIFEST["disposition_counts"].values()), 193)

    def test_every_committed_file_matches_its_recorded_hash(self) -> None:
        self.assertEqual([f for f in verify_bundle() if f.severity == "error"], [])

    def test_a_changed_committed_file_is_reported_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "set"
            shutil.copytree(default_reference_dir(), base)
            path = base / BUNDLE_DIRNAME / "data" / "sources.json"
            path.write_bytes(path.read_bytes().replace(b"BRENDA", b"BRANDA", 1))
            found = verify_bundle(base)
            self.assertEqual([(f.code, f.subject) for f in found
                              if f.severity == "error"],
                             [("bundle.changed", "data/sources.json")])

    def test_a_missing_committed_file_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "set"
            shutil.copytree(default_reference_dir(), base)
            (base / BUNDLE_DIRNAME / "NOTICE.txt").unlink()
            self.assertIn(("bundle.missing", "NOTICE.txt"),
                          [(f.code, f.subject) for f in verify_bundle(base)])

    def test_a_file_from_nowhere_is_an_error_but_a_derived_one_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "set"
            shutil.copytree(default_reference_dir(), base)
            (base / BUNDLE_DIRNAME / "stray.json").write_text("{}", encoding="utf-8")
            found = verify_bundle(base)
            self.assertEqual([(f.code, f.subject) for f in found
                              if f.severity == "error"],
                             [("bundle.unlisted", "stray.json")])
            self.assertTrue(any(f.code == "bundle.derived" for f in found))

    def test_the_derived_file_is_present_and_names_its_command(self) -> None:
        for rel, command in DERIVED_FILES.items():
            self.assertTrue((BUNDLE / rel).is_file(), rel)
            self.assertTrue(command.startswith("eagent "), command)

    def test_the_dropped_aggregate_is_not_committed_but_its_hash_is_kept(self) -> None:
        for rel in DROPPED_DUPLICATES:
            self.assertFalse((BUNDLE / rel).is_file())
            self.assertIn(rel, MEMBERS)
            self.assertRegex(MEMBERS[rel]["sha256"], r"^[0-9a-f]{64}$")
            self.assertIn("dropped_because", MEMBERS[rel])

    def test_every_pinned_member_keeps_a_url_to_fetch_it_from(self) -> None:
        pinned = [i for i in MEMBERS.values() if i["disposition"] == "pinned"]
        self.assertEqual(len(pinned), 129)
        for info in pinned:
            self.assertTrue(info.get("source_url", "").startswith("https://"), info)
        self.assertEqual({i["source_url"].split("/")[2] for i in pinned},
                         {"data.rcsb.org", "files.rcsb.org"},
                         "only the two RCSB hosts are pinned; the GitHub and "
                         "Zenodo files are small and open, so they are committed")
        upstream = [i for i in MEMBERS.values()
                    if i["disposition"] == "committed_upstream"]
        self.assertEqual({i["source_url"].split("/")[2] for i in upstream},
                         {"raw.githubusercontent.com", "zenodo.org"})

    def test_the_committed_bundle_stays_small(self) -> None:
        total = sum(p.stat().st_size for p in BUNDLE.rglob("*") if p.is_file())
        self.assertLess(total, 4e6, "the bundle is meant to be the compilation, "
                                    "not the 46 MB of upstream originals")


class TheTwoExportsAgree(unittest.TestCase):
    def test_all_shared_fields_of_the_five_tables_agree(self) -> None:
        findings = cross_check_tables(BUNDLE, default_reference_dir() / "tables")
        self.assertEqual([f for f in findings if f.severity == "error"], [])

    def test_the_agreement_is_reported_with_its_size(self) -> None:
        findings = cross_check_tables(BUNDLE, default_reference_dir() / "tables")
        note = next(f for f in findings if f.code == "cross.compared")
        self.assertIn("1632 shared fields", note.message)

    def test_a_changed_value_in_either_export_is_caught(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "set"
            shutil.copytree(default_reference_dir(), base)
            path = base / BUNDLE_DIRNAME / "data" / "activity_labels.json"
            records = json.loads(path.read_text(encoding="utf-8"))
            for r in records:
                if r["label_id"] == "6ZZO_PaHBDH_AAE":
                    r["kcat_original"] = 31.0
            path.write_text(json.dumps(records), encoding="utf-8")
            errors = [f for f in cross_check_tables(base / BUNDLE_DIRNAME,
                                                    base / "tables")
                      if f.severity == "error"]
            self.assertEqual([f.subject for f in errors],
                             ["kinetics[6ZZO_PaHBDH_AAE].kcat_original"])

    def test_the_three_representation_conventions_are_not_disagreements(self) -> None:
        """A JSON list against a joined string, '=' against 'eq', bool against 1/0."""
        records = json.loads(
            (BUNDLE / "data" / "activity_labels.json").read_text(encoding="utf-8"))
        by = {r["label_id"]: r for r in records}
        self.assertIsInstance(by["6ZZO_PaHBDH_AAE"]["quality_flags"], list)
        self.assertEqual(by["6ZZQ_AbHBDH_AAE"]["Km_comparator"], "=")
        selectivity = json.loads(
            (BUNDLE / "data" / "selectivity_labels.json").read_text(encoding="utf-8"))
        self.assertIsInstance(selectivity[0]["kinetic_variant_identity_match"], bool)
        # and the repo's CSVs spell the same three the other way
        rs = load_reference_set()
        self.assertEqual(rs.kinetic("6ZZQ_AbHBDH_AAE").km_comparator, "eq")

    def test_a_key_only_one_export_has_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "set"
            shutil.copytree(default_reference_dir(), base)
            path = base / BUNDLE_DIRNAME / "data" / "selectivity_labels.json"
            records = json.loads(path.read_text(encoding="utf-8"))
            path.write_text(json.dumps(records[:-1]), encoding="utf-8")
            errors = [f for f in cross_check_tables(base / BUNDLE_DIRNAME,
                                                    base / "tables")
                      if f.severity == "error"]
            self.assertEqual([f.code for f in errors], ["cross.keys"])


class TheCoordinatesAgreeWithAnIndependentFetch(unittest.TestCase):
    def test_all_nineteen_are_recorded_as_independently_confirmed(self) -> None:
        """The archive's files and this project's own RCSB fetch, byte for byte."""
        members = {}
        for pid in [s.pdb_id for s in load_reference_set().structures]:
            info = MEMBERS[f"structures/{pid}/{pid}.cif"]
            members[f"structures/{pid}/{pid}.cif"] = type(
                "M", (), {"sha256": info["sha256"]})()
        findings = coordinate_agreement(members, default_reference_dir())
        errors = [f for f in findings if f.severity == "error"]
        self.assertEqual(errors, [])
        note = next(f for f in findings
                    if f.code == "coordinates.independently_confirmed")
        self.assertIn("19 entries", note.subject)

    def test_a_revision_between_the_two_fetches_would_be_reported_not_preferred(self) -> None:
        members = {"structures/1IPF/1IPF.cif": type("M", (), {"sha256": "0" * 64})()}
        findings = coordinate_agreement(members, default_reference_dir())
        errors = [f for f in findings if f.severity == "error"]
        self.assertEqual([f.code for f in errors], ["coordinates.archive_differs"])
        self.assertIn("revised it between them", errors[0].message)


# ==========================================================================
class TheImportRefusesRatherThanWrites(_Tmp):
    def test_an_archive_with_no_manifest_cannot_be_dispositioned(self) -> None:
        path = self.build_archive({"data/x.json": b"{}"}, manifest=None)
        with self.assertRaisesRegex(BundleError, "no file_manifest"):
            read_archive(path)

    def test_a_member_outside_the_single_root_is_refused(self) -> None:
        path = self.tmp / "bad.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(ARCHIVE_ROOT + "file_manifest.json", b'{"files": []}')
            zf.writestr("elsewhere/x.json", b"{}")
        with self.assertRaisesRegex(BundleError, "outside"):
            read_archive(path)

    def test_an_unsafe_member_name_is_refused(self) -> None:
        path = self.tmp / "bad.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr(ARCHIVE_ROOT + "file_manifest.json", b'{"files": []}')
            zf.writestr(ARCHIVE_ROOT + "../escape.json", b"{}")
        with self.assertRaisesRegex(BundleError, "safe member name"):
            read_archive(path)

    def test_a_manifest_hash_that_does_not_match_its_member_is_an_error(self) -> None:
        body = b'{"a": 1}'
        manifest = {"audit_date": "x", "files": [
            {"path": "data/x.json", "bytes": len(body), "sha256": "0" * 64,
             "source_url": None, "kind": "curated_or_supporting_artifact"}]}
        path = self.build_archive({"data/x.json": body}, manifest=manifest)
        members, got = read_archive(path)
        findings = verify_archive_manifest(members, got)
        self.assertEqual([f.code for f in findings], ["archive.hash"])

    def test_a_member_the_manifest_does_not_list_is_an_error(self) -> None:
        manifest = {"audit_date": "x", "files": []}
        path = self.build_archive({"data/x.json": b"{}"}, manifest=manifest)
        members, got = read_archive(path)
        self.assertEqual([f.code for f in verify_archive_manifest(members, got)],
                         ["archive.unlisted"])

    def test_a_spreadsheet_that_is_not_the_committed_one_stops_the_import(self) -> None:
        base = self.reference_copy()
        path = self.build_archive({
            "KRED_Calibration_Reference_v0.1.xlsx": b"a different workbook"})
        result = import_archive(path, base, write=False)
        codes = [f.code for f in result["findings"]]
        self.assertIn("archive.workbook_differs", codes)
        with self.assertRaisesRegex(BundleError, "did not pass its checks"):
            import_archive(path, base, write=True)
        self.assertFalse((base / BUNDLE_DIRNAME).exists())

    def test_an_aggregate_that_is_not_really_a_duplicate_stops_the_import(self) -> None:
        base = self.reference_copy()
        per = {"pdb_id": "1IPF", "resolution_A": 2.5}
        other = {"pdb_id": "1IPF", "resolution_A": 9.9}       # differs
        path = self.build_archive({
            "data/structure_audit_raw.json": json.dumps([other]).encode(),
            "structures/1IPF/audit_metadata.json": json.dumps(per).encode()})
        result = import_archive(path, base, write=False)
        self.assertIn("archive.dropped_not_duplicate",
                      [f.code for f in result["findings"]])
        with self.assertRaises(BundleError):
            import_archive(path, base, write=True)

    def test_an_aggregate_entry_with_no_per_structure_file_stops_the_import(self) -> None:
        base = self.reference_copy()
        path = self.build_archive({
            "data/structure_audit_raw.json":
                json.dumps([{"pdb_id": "9XYZ"}]).encode()})
        self.assertIn("archive.dropped_not_duplicate",
                      [f.code for f in import_archive(path, base, write=False)["findings"]])

    def test_a_different_archive_is_a_warning_and_the_hash_is_recorded(self) -> None:
        base = self.reference_copy()
        workbook = (default_reference_dir() / "source"
                    / "KRED_Calibration_Reference_v0.1.xlsx").read_bytes()
        path = self.build_archive({
            "KRED_Calibration_Reference_v0.1.xlsx": workbook,
            "data/x.json": b"{}"})
        result = import_archive(path, base, write=False)
        warning = next(f for f in result["findings"]
                       if f.code == "archive.another_delivery")
        self.assertIn("not the", warning.message)
        self.assertNotEqual(result["report"]["archive_sha256"], ARCHIVE_SHA256)

    def test_a_clean_small_archive_writes_only_its_committed_members(self) -> None:
        base = self.reference_copy()
        workbook = (default_reference_dir() / "source"
                    / "KRED_Calibration_Reference_v0.1.xlsx").read_bytes()
        path = self.build_archive(
            {"KRED_Calibration_Reference_v0.1.xlsx": workbook,
             "data/curated.json": b'{"kept": true}',
             "NOTICE.txt": b"terms\n",
             # an entry this project does not pin, so the coordinate check has
             # nothing to compare and the import is about the rule, not the bytes
             "structures/9XYZ/9XYZ.cif": b"data_9XYZ\n"},
            kinds={"structures/9XYZ/9XYZ.cif": "upstream_original"},
            urls={"structures/9XYZ/9XYZ.cif":
                  "https://files.rcsb.org/download/9XYZ.cif"})
        result = import_archive(path, base, write=True)
        out = base / BUNDLE_DIRNAME
        self.assertTrue((out / "data" / "curated.json").is_file())
        self.assertTrue((out / "NOTICE.txt").is_file())
        self.assertFalse((out / "structures" / "9XYZ" / "9XYZ.cif").is_file(),
                         "an upstream original is pinned, not committed")
        self.assertFalse((out / "KRED_Calibration_Reference_v0.1.xlsx").is_file(),
                         "the spreadsheet is already under source/")
        stored = json.loads((out / ARCHIVE_MANIFEST).read_text(encoding="utf-8"))
        self.assertEqual(
            stored["members"]["structures/9XYZ/9XYZ.cif"]["source_url"],
            "https://files.rcsb.org/download/9XYZ.cif")
        self.assertEqual([f for f in verify_bundle(base) if f.severity == "error"], [])

    def test_a_coordinate_file_disagreeing_with_a_pin_stops_the_import(self) -> None:
        """The check that caught a synthetic file would catch a real revision."""
        base = self.reference_copy()
        path = self.build_archive(
            {"structures/1IPF/1IPF.cif": b"data_1IPF\n"},
            kinds={"structures/1IPF/1IPF.cif": "upstream_original"},
            urls={"structures/1IPF/1IPF.cif":
                  "https://files.rcsb.org/download/1IPF.cif"})
        with self.assertRaisesRegex(BundleError, "Two retrievals of one entry"):
            import_archive(path, base, write=True)
        self.assertFalse((base / BUNDLE_DIRNAME).exists())

    def test_a_dry_run_writes_nothing(self) -> None:
        base = self.reference_copy()
        path = self.build_archive({"data/curated.json": b"{}"})
        import_archive(path, base, write=False)
        self.assertFalse((base / BUNDLE_DIRNAME).exists())


if __name__ == "__main__":
    unittest.main()
