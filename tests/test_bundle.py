"""Tests for :mod:`eagent.deliverables.bundle`.

The cases are the ways a research package stops being a record:

* an item that was never produced quietly not appearing, so the package reads
  as complete to anyone who does not know what should be in it;
* a batch file whose name states a number the file does not contain;
* a converted PDB standing in for the mmCIF it came from, or a picture of a
  confidence plot standing in for the confidence file;
* a declared file that has been edited, deleted or added since assembly, with
  the manifest still vouching for it;
* a report quoting a count with nothing behind it.

Every fixture here is built by hand rather than by running the pipeline: the
bundler must work on a run directory it did not produce, including a
half-finished one, and a test that needed a successful run to exercise it
would only ever test the happy path.

Runs under pytest, or standalone with ``python3 tests/test_bundle.py``.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

from eagent.deliverables.bundle import (
    BATCH_ITEM_NAME,
    BUNDLE_ITEMS,
    BUNDLE_MANIFEST_NAME,
    BundleManifest,
    ItemStatus,
    RESEARCH_REPORT_NAME,
    RUN_MANIFEST_NAME,
    _fact,
    assemble_bundle,
    verify_bundle,
)
from eagent.envelope import Artifact, Provenance, Status, ToolResult
from eagent.provenance import RunManifest, utc_now
from eagent.schemas.candidate import Candidate, SequenceRecord

#: The package the brief specifies, by the name each item is declared under.
STANDARD_ITEMS = (
    "reaction_spec.yaml", "evidence_records.jsonl", "candidate_sequences.fasta",
    "sequence_annotations.tsv", "family_analysis", "structures", "complexes",
    "confidence_metrics", "residue_atom_mapping.tsv", "catalytic_geometry.tsv",
    "candidate_scorecards.tsv", BATCH_ITEM_NAME, "mutation_proposals.tsv",
    "experiment_plan.yaml", "assay_results_template.csv", RUN_MANIFEST_NAME,
    RESEARCH_REPORT_NAME,
)

MINIMAL_CIF = """data_TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
ATOM 1 N N ALA A 1 1.000 2.000 3.000 1.00 50.00 1 A
ATOM 2 C CA ALA A 1 2.000 3.000 4.000 1.00 50.00 1 A
"""

MINIMAL_PDB = (
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 50.00           N\n"
    "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 50.00           C\n"
    "END\n"
)

#: A PDB with one atom fewer than MINIMAL_CIF: the shape a conversion failure
#: takes in practice, where residues or chains silently do not fit the format.
TRUNCATED_PDB = (
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 50.00           N\n"
    "END\n"
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

class BundleCase(unittest.TestCase):
    """A temporary run directory, torn down afterwards."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="eagent-bundle-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.run_dir = self.tmp / "run"
        self.run_dir.mkdir(parents=True)
        self.manifest = RunManifest(run_id="R-1", task_id="T-1")

    # -- helpers -----------------------------------------------------------
    def write(self, relative: str, text: str) -> Path:
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def record_step(self, step_id: str, interface: str,
                    artifacts: Sequence[Artifact] = ()) -> None:
        result = ToolResult(status=Status.SUCCESS,
                            provenance=Provenance(tool=interface),
                            artifacts=list(artifacts))
        self.manifest.record(step_id, interface, result, utc_now())

    def save_manifest(self) -> Path:
        return self.manifest.write(self.run_dir / RUN_MANIFEST_NAME)

    def assemble(self, **kwargs: Any):
        return assemble_bundle(self.run_dir, **kwargs)

    def entry(self, result: Any, name: str):
        found = result.manifest.entry(name)
        self.assertIsNotNone(found, f"{name} is not declared in the manifest")
        return found

    def batch_entry(self, result: Any):
        return self.entry(result, BATCH_ITEM_NAME)


# ---------------------------------------------------------------------------
# the declared set
# ---------------------------------------------------------------------------

class DeclaredSetTests(BundleCase):
    """Nothing the package promises may be absent from the index."""

    def test_the_standard_package_is_declared_in_full(self) -> None:
        declared = {item.name for item in BUNDLE_ITEMS}
        for name in STANDARD_ITEMS:
            self.assertIn(name, declared)

    def test_an_empty_run_lists_every_item_as_missing_rather_than_omitting_it(self) -> None:
        self.save_manifest()
        result = self.assemble()
        names = [e.name for e in result.manifest.entries]
        self.assertEqual(names, [item.name for item in BUNDLE_ITEMS])
        # Everything but the manifest and the generated report is absent.
        missing = {e.name for e in result.manifest.missing}
        self.assertIn("candidate_sequences.fasta", missing)
        self.assertIn("structures", missing)
        self.assertIn(BATCH_ITEM_NAME, missing)
        self.assertNotIn(RUN_MANIFEST_NAME, missing)
        self.assertFalse(result.complete)

    def test_a_missing_item_carries_a_reason_and_what_a_curator_must_supply(self) -> None:
        self.save_manifest()
        result = self.assemble()
        document = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertFalse(document["complete"])
        self.assertTrue(document["missing"])
        for record in document["missing"]:
            self.assertTrue(record["reason"],
                            f"{record['name']} is missing with no reason")
            self.assertTrue(record["why_it_matters"])
            self.assertTrue(record["curator_must_supply"],
                            f"{record['name']} does not say how to supply it")

    def test_the_counts_add_up_so_a_reader_can_check_them(self) -> None:
        self.save_manifest()
        result = self.assemble()
        counts = result.manifest.counts()
        self.assertEqual(counts["declared"], len(BUNDLE_ITEMS))
        self.assertEqual(
            counts["present"] + counts["partial"] + counts["missing"],
            counts["declared"])

    def test_an_artifact_the_manifest_records_but_that_is_absent_is_reported(self) -> None:
        """The run believed it wrote the file. That is worth saying out loud."""
        self.record_step("s1", "mine_sequences", [Artifact(
            key="candidate_sequences",
            path=str(self.run_dir / "mine_sequences" / "candidate_sequences.fasta"),
            kind="file")])
        self.save_manifest()
        result = self.assemble()
        entry = self.entry(result, "candidate_sequences.fasta")
        self.assertIs(entry.status, ItemStatus.MISSING)
        self.assertTrue(any("records artifact" in note for note in entry.notes),
                        entry.notes)

    def test_a_run_without_a_manifest_is_bundled_and_the_gap_is_recorded(self) -> None:
        self.write("reaction_spec.yaml", "task_id: T-1\nreaction: {}\n")
        result = self.assemble()
        self.assertIs(self.entry(result, "reaction_spec.yaml").status,
                      ItemStatus.PRESENT)
        self.assertIs(self.entry(result, RUN_MANIFEST_NAME).status,
                      ItemStatus.MISSING)
        self.assertTrue(any(RUN_MANIFEST_NAME in note
                            for note in result.manifest.notes))


# ---------------------------------------------------------------------------
# the batch filename
# ---------------------------------------------------------------------------

def _batch_csv(n_constructs: int, n_controls: int = 0) -> str:
    header = "slot,kind,candidate_id,role,family,selection_reason\n"
    rows = [f"{i + 1},candidate,C{i + 1},high_evidence,SDR,best supported\n"
            for i in range(n_constructs)]
    rows += [f"{n_constructs + i + 1},control,no_enzyme_{i + 1},control,,"
             f"assay system works\n" for i in range(n_controls)]
    return header + "".join(rows)


class BatchNamingTests(BundleCase):
    """A filename that states a count must state the count the file holds."""

    def _assemble_with_batch(self, declared_name: str, n_constructs: int,
                             n_controls: int = 0):
        self.write(f"select_batch/{declared_name}",
                   _batch_csv(n_constructs, n_controls))
        self.save_manifest()
        return self.assemble()

    def test_a_short_batch_is_named_by_its_real_count(self) -> None:
        result = self._assemble_with_batch("selected_batch_96.csv", 71, 4)
        entry = self.batch_entry(result)
        self.assertIs(entry.status, ItemStatus.PRESENT)
        self.assertEqual(entry.path, "selected_batch_71.csv")
        self.assertEqual(entry.n_records, 71)
        self.assertTrue((result.bundle_dir / "selected_batch_71.csv").is_file())
        self.assertFalse((result.bundle_dir / "selected_batch_96.csv").exists())
        self.assertTrue(entry.renamed)
        self.assertTrue(any("actual construct count" in note
                            for note in entry.notes), entry.notes)

    def test_the_control_rows_are_not_counted_as_constructs(self) -> None:
        result = self._assemble_with_batch("selected_batch_96.csv", 10, 5)
        entry = self.batch_entry(result)
        self.assertEqual(entry.n_records, 10)
        self.assertTrue(any("are constructs" in note for note in entry.notes),
                        entry.notes)

    def test_a_full_batch_keeps_the_declared_name(self) -> None:
        result = self._assemble_with_batch("selected_batch_96.csv", 96)
        entry = self.batch_entry(result)
        self.assertEqual(entry.path, "selected_batch_96.csv")
        self.assertFalse(entry.renamed)

    def test_the_file_is_found_even_when_the_run_already_named_it_short(self) -> None:
        result = self._assemble_with_batch("selected_batch_12.csv", 12)
        entry = self.batch_entry(result)
        self.assertEqual(entry.path, "selected_batch_12.csv")
        self.assertEqual(entry.n_records, 12)

    def test_the_report_quotes_the_real_count_with_its_source(self) -> None:
        result = self._assemble_with_batch("selected_batch_96.csv", 71, 4)
        report = (result.bundle_dir / RESEARCH_REPORT_NAME).read_text(
            encoding="utf-8")
        self.assertIn("71 construct(s) in the batch "
                      "[source: selected_batch_71.csv]", report)
        self.assertNotIn("96 construct(s)", report)


# ---------------------------------------------------------------------------
# structures and confidence
# ---------------------------------------------------------------------------

class PrimaryRecordTests(BundleCase):
    """mmCIF is the record; a conversion or a picture is not."""

    def test_mmcif_with_a_matching_pdb_is_present_and_the_pdb_is_an_extra(self) -> None:
        self.write("prepare_structures/structures/C1.cif", MINIMAL_CIF)
        self.write("prepare_structures/structures/C1.pdb", MINIMAL_PDB)
        self.save_manifest()
        result = self.assemble()
        entry = self.entry(result, "structures")
        self.assertIs(entry.status, ItemStatus.PRESENT)
        roles = {f.path: f.role for f in entry.files}
        self.assertEqual(roles["structures/C1.cif"], "primary")
        self.assertEqual(roles["structures/C1.pdb"], "derived")
        derived = next(f for f in entry.files if f.role == "derived")
        self.assertTrue(derived.note.startswith("validated"), derived.note)

    def test_a_directory_of_converted_pdb_only_is_partial_not_present(self) -> None:
        self.write("prepare_structures/structures/C1.pdb", MINIMAL_PDB)
        self.save_manifest()
        result = self.assemble()
        entry = self.entry(result, "structures")
        self.assertIs(entry.status, ItemStatus.PARTIAL)
        self.assertIn("mmCIF", entry.reason)
        self.assertFalse(result.complete)

    def test_a_pdb_that_disagrees_with_its_mmcif_is_not_silently_accepted(self) -> None:
        self.write("prepare_structures/structures/C1.cif", MINIMAL_CIF)
        self.write("prepare_structures/structures/C1.pdb", TRUNCATED_PDB)
        self.save_manifest()
        result = self.assemble()
        entry = self.entry(result, "structures")
        self.assertIs(entry.status, ItemStatus.PARTIAL)
        self.assertIn("could not be validated", entry.reason)
        derived = next(f for f in entry.files if f.role == "derived")
        self.assertIn("disagrees with the mmCIF record", derived.note)

    def test_an_image_is_not_a_substitute_for_the_confidence_files(self) -> None:
        self.write("prepare_structures/confidence_metrics/C1_plddt.png", "not a plot")
        self.save_manifest()
        result = self.assemble()
        entry = self.entry(result, "confidence_metrics")
        self.assertIs(entry.status, ItemStatus.PARTIAL)
        self.assertIn("screenshot", entry.reason)
        self.assertFalse(result.complete)
        # The image is still carried, labelled for what it is.
        self.assertEqual([f.role for f in entry.files], ["rejected_substitute"])

    def test_a_real_confidence_file_makes_the_directory_present(self) -> None:
        self.write("prepare_structures/confidence_metrics/C1.json",
                   json.dumps({"mean_plddt": 82.1, "pocket_plddt": 74.0}))
        self.save_manifest()
        result = self.assemble()
        self.assertIs(self.entry(result, "confidence_metrics").status,
                      ItemStatus.PRESENT)

    def test_an_empty_directory_is_missing_not_present(self) -> None:
        (self.run_dir / "prepare_structures" / "structures").mkdir(parents=True)
        self.save_manifest()
        result = self.assemble()
        entry = self.entry(result, "structures")
        self.assertIs(entry.status, ItemStatus.MISSING)
        self.assertIn("no files", entry.reason)

    def test_a_supporting_directory_needs_no_primary_format(self) -> None:
        self.write("annotate_family/family_analysis/alignment.txt", "MKAV\nMKTV\n")
        self.save_manifest()
        result = self.assemble()
        self.assertIs(self.entry(result, "family_analysis").status,
                      ItemStatus.PRESENT)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

class VerifyBundleTests(BundleCase):
    """A package vouches for its files, so the vouching has to be checkable."""

    def _complete_enough_bundle(self):
        self.write("reaction_spec.yaml", "task_id: T-1\n")
        self.write("evidence_records.jsonl", '{"record_id": "E1"}\n')
        self.write("prepare_structures/structures/C1.cif", MINIMAL_CIF)
        self.save_manifest()
        return self.assemble()

    def test_a_freshly_assembled_bundle_verifies(self) -> None:
        result = self._complete_enough_bundle()
        verification = verify_bundle(result.bundle_dir)
        self.assertTrue(verification.ok, verification.problems)
        self.assertGreater(verification.n_files_checked, 0)
        self.assertEqual(verification.n_items_checked, len(BUNDLE_ITEMS))

    def test_a_tampered_file_is_caught(self) -> None:
        result = self._complete_enough_bundle()
        target = result.bundle_dir / "reaction_spec.yaml"
        target.write_text(target.read_text(encoding="utf-8") + "# edited\n",
                          encoding="utf-8")
        verification = verify_bundle(result.bundle_dir)
        self.assertFalse(verification.ok)
        self.assertTrue(any("reaction_spec.yaml" in p and "does not match" in p
                            for p in verification.problems),
                        verification.problems)

    def test_a_tampered_file_inside_a_directory_is_caught(self) -> None:
        result = self._complete_enough_bundle()
        target = result.bundle_dir / "structures" / "C1.cif"
        target.write_text(MINIMAL_CIF.replace("1.000", "9.000"),
                          encoding="utf-8")
        verification = verify_bundle(result.bundle_dir)
        self.assertFalse(verification.ok)
        self.assertTrue(any("structures/C1.cif" in p
                            for p in verification.problems),
                        verification.problems)

    def test_a_deleted_file_is_caught(self) -> None:
        result = self._complete_enough_bundle()
        (result.bundle_dir / "evidence_records.jsonl").unlink()
        verification = verify_bundle(result.bundle_dir)
        self.assertFalse(verification.ok)
        self.assertTrue(any("absent from the bundle" in p
                            for p in verification.problems),
                        verification.problems)

    def test_an_unrecorded_file_is_caught(self) -> None:
        """An extra file has no provenance, which is its own kind of problem."""
        result = self._complete_enough_bundle()
        (result.bundle_dir / "structures" / "sneaked_in.cif").write_text(
            MINIMAL_CIF, encoding="utf-8")
        verification = verify_bundle(result.bundle_dir)
        self.assertFalse(verification.ok)
        self.assertTrue(any("not recorded" in p for p in verification.problems),
                        verification.problems)

    def test_a_directory_without_a_manifest_is_refused(self) -> None:
        empty = self.tmp / "not-a-bundle"
        empty.mkdir()
        verification = verify_bundle(empty)
        self.assertFalse(verification.ok)
        self.assertIn(BUNDLE_MANIFEST_NAME, verification.problems[0])

    def test_the_manifest_path_itself_may_be_passed(self) -> None:
        result = self._complete_enough_bundle()
        verification = verify_bundle(result.bundle_dir / BUNDLE_MANIFEST_NAME)
        self.assertTrue(verification.ok, verification.problems)

    def test_verification_reports_incompleteness_separately_from_integrity(self) -> None:
        """An intact package can still be an incomplete one, and says so."""
        result = self._complete_enough_bundle()
        verification = verify_bundle(result.bundle_dir)
        self.assertTrue(verification.ok)
        self.assertFalse(verification.complete)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

_COUNT_LINE = re.compile(
    r"^-?\s*\**\s*\d+ (step|construct|candidate|field|record)")


class ReportTests(BundleCase):
    """Every number in the narrative has a file behind it."""

    def _report(self, **kwargs: Any) -> str:
        result = self.assemble(**kwargs)
        self.report_result = result
        return (result.bundle_dir / RESEARCH_REPORT_NAME).read_text(
            encoding="utf-8")

    def test_a_quantitative_claim_without_a_source_is_a_programming_error(self) -> None:
        with self.assertRaises(ValueError):
            _fact("17 candidates passed", "")
        self.assertIn("[source: x.tsv]", _fact("3 rows", "x.tsv"))

    def test_counts_in_the_report_cite_their_artifact(self) -> None:
        self.record_step("s1", "normalize_reaction")
        self.save_manifest()
        report = self._report()
        quantitative = [line for line in report.splitlines()
                        if _COUNT_LINE.match(line.strip())]
        self.assertTrue(quantitative, "the report states no counts at all")
        for line in quantitative:
            self.assertIn("[source:", line, line)

    def test_the_report_says_the_package_is_incomplete_when_it_is(self) -> None:
        self.save_manifest()
        report = self._report()
        self.assertIn("This package is incomplete", report)
        self.assertIn("## What is missing", report)
        self.assertNotIn("All 18 declared items are present", report)

    def test_the_report_does_not_list_itself_as_missing(self) -> None:
        self.save_manifest()
        report = self._report()
        self.assertNotIn(f"**{RESEARCH_REPORT_NAME}** -- MISSING", report)

    def test_an_absent_batch_is_stated_as_unavailable_not_as_zero(self) -> None:
        self.save_manifest()
        report = self._report()
        self.assertIn("unavailable", report)
        self.assertNotIn("0 construct(s)", report)

    def test_candidate_blocks_are_generated_by_the_scorecard_explainer(self) -> None:
        self.save_manifest()
        candidate = Candidate(
            candidate_id="CAND_1",
            sequence_record=SequenceRecord(
                candidate_id="CAND_1", sequence="MKAVVTGAAQGIG",
                accession="UPI0001", source_database="uniprotkb",
                search_method="hmmsearch", seed_accession="P00001"))
        report = self._report(candidates=[candidate])
        self.assertIn("### CAND_1", report)
        self.assertIn("Why it was retrieved", report)
        self.assertIn("Why it deserves a slot", report)
        self.assertIn("eagent.science.scorecard.explain", report)

    def test_candidate_blocks_fall_back_to_the_stored_explanations(self) -> None:
        self.write("evaluate_catalysis/candidate_explanations.txt",
                   "=" * 70 + "\nCandidate CAND_2\nWhy it was retrieved:\n"
                   "  hmmsearch against uniprotkb\n")
        self.save_manifest()
        report = self._report()
        self.assertIn("### CAND_2", report)
        self.assertIn("candidate_explanations.txt", report)

    def test_the_report_does_not_cite_a_backing_table_that_is_absent(self) -> None:
        """An invitation to check a file that is not there reads as a check."""
        self.write("evaluate_catalysis/candidate_explanations.txt",
                   "=" * 70 + "\nCandidate CAND_3\nWhy it was retrieved:\n  seed\n")
        self.save_manifest()
        report = self._report()
        self.assertIn("cannot be checked against them", report)
        self.assertIn("catalytic_geometry.tsv", report)

    def test_the_report_cites_the_backing_tables_when_they_are_there(self) -> None:
        self.write("evaluate_catalysis/candidate_explanations.txt",
                   "=" * 70 + "\nCandidate CAND_4\nWhy it was retrieved:\n  seed\n")
        self.write("evaluate_catalysis/catalytic_geometry.tsv",
                   "candidate_id\tpose_id\n CAND_4\tP1\n")
        self.write("evaluate_catalysis/candidate_scorecards.tsv",
                   "candidate_id\tpasses_gates\nCAND_4\tTrue\n")
        self.save_manifest()
        report = self._report()
        self.assertIn("The measurements behind these lines are in", report)
        self.assertNotIn("cannot be checked against them", report)

    def test_the_report_states_the_no_total_score_rule(self) -> None:
        self.save_manifest()
        report = self._report()
        self.assertIn("no total score", report)

    def test_a_run_manifest_with_no_cost_is_not_reported_as_zero(self) -> None:
        self.record_step("s1", "normalize_reaction")
        self.save_manifest()
        report = self._report()
        self.assertIn("cost: nothing was recorded by any step", report)
        self.assertNotIn("cost: 0", report)


# ---------------------------------------------------------------------------
# output directory handling
# ---------------------------------------------------------------------------

class OutputDirectoryTests(BundleCase):
    def test_an_unrelated_non_empty_directory_is_refused(self) -> None:
        self.save_manifest()
        out = self.tmp / "out"
        out.mkdir()
        (out / "someone_elses_work.txt").write_text("keep me", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            assemble_bundle(self.run_dir, out)

    def test_overwrite_replaces_an_unrelated_directory_only_when_asked(self) -> None:
        self.save_manifest()
        out = self.tmp / "out"
        out.mkdir()
        (out / "someone_elses_work.txt").write_text("keep me", encoding="utf-8")
        result = assemble_bundle(self.run_dir, out, overwrite=True)
        self.assertFalse((out / "someone_elses_work.txt").exists())
        self.assertTrue(result.manifest_path.is_file())

    def test_a_bundle_may_be_reassembled_in_place(self) -> None:
        self.save_manifest()
        first = self.assemble()
        again = assemble_bundle(self.run_dir, first.bundle_dir)
        self.assertTrue(verify_bundle(again.bundle_dir).ok)

    def test_a_missing_run_directory_is_refused(self) -> None:
        with self.assertRaises(FileNotFoundError):
            assemble_bundle(self.tmp / "no-such-run")


class ManifestRoundTripTests(BundleCase):
    def test_the_manifest_round_trips(self) -> None:
        self.write("reaction_spec.yaml", "task_id: T-1\n")
        self.save_manifest()
        result = self.assemble()
        reloaded = BundleManifest.load(result.manifest_path)
        self.assertEqual([e.name for e in reloaded.entries],
                         [e.name for e in result.manifest.entries])
        self.assertEqual(reloaded.complete, result.manifest.complete)
        self.assertEqual(reloaded.run_id, "R-1")
        self.assertEqual(reloaded.task_id, "T-1")


if __name__ == "__main__":  # pragma: no cover - standalone runner
    unittest.main(verbosity=2)
