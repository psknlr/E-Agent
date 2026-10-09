"""Regression tests for the follow-up audit's second batch: identity of things.

Four findings, each reproduced before being fixed. They are about a construct,
a residue, a file and a run being the thing they claim to be.

* an alignment tie-break named one residue and would have changed another;
* two candidates encoded onto one structure filename, so the second silently
  replaced the first;
* a package adopted a same-named file belonging to a different task;
* assembling a package into its own run directory deleted the run first.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import unittest

from eagent.deliverables.bundle import BUNDLE_ITEMS, _resolve_source, assemble_bundle
from eagent.envelope import Artifact, Status, ToolResult
from eagent.provenance import RunManifest, sha256_file
from eagent.science.numbering import NumberingError, build_map, verify_residue
from eagent.science.structure_io import Atom, Chain, Residue
from eagent.tools.prepare_structures import PrepareStructures

THREE = {"M": "MET", "K": "LYS", "A": "ALA", "I": "ILE", "V": "VAL",
         "T": "THR", "Y": "TYR", "G": "GLY", "S": "SER", "R": "ARG"}
#: Two adjacent tyrosines at 1-based positions 7 and 8.
SEQ = "MKAIVTYYGASRG"


def residue(letter: str, number: int) -> Residue:
    atom = Atom(serial=number, name="CA", element="C", resname=THREE[letter],
                chain="A", resseq=number, icode="", altloc="",
                x=float(number), y=0.0, z=0.0, occupancy=1.0,
                bfactor_or_plddt=90.0, is_hetatm=False)
    return Residue(chain="A", resname=THREE[letter], resseq=number, icode="",
                   atoms=[atom], is_hetatm=False)


def chain(observed) -> Chain:
    return Chain(chain_id="A",
                 residues=[residue(SEQ[n - 1], n) for n in observed])


class AlignmentAmbiguityTests(unittest.TestCase):
    """A partly observed run of identical residues determines nothing."""

    def setUp(self) -> None:
        # Coordinates for the first tyrosine only; the second is absent.
        self.partial = build_map(SEQ, chain([1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]))
        self.full = build_map(SEQ, chain(range(1, 14)))

    def test_the_ambiguity_is_detected(self) -> None:
        index = self.partial.to_index(7)
        self.assertTrue(self.partial.is_ambiguous(index))

    def test_a_fully_observed_run_is_not_flagged(self) -> None:
        self.assertEqual(self.full.ambiguous_indices, frozenset())

    def test_the_map_says_why_in_its_notes(self) -> None:
        self.assertTrue([n for n in self.partial.notes
                         if "equally-scoring" in n or "identical residues" in n])

    def test_an_ambiguous_position_cannot_be_called_observed(self) -> None:
        index = self.partial.to_index(7)
        with self.assertRaises(NumberingError) as ctx:
            verify_residue(self.partial, index, "Y", require_observed=True)
        self.assertIn("cannot be confirmed as observed", str(ctx.exception))

    def test_an_unambiguous_position_still_verifies(self) -> None:
        """The guard must not refuse ordinary, well-determined positions."""
        self.assertIsNotNone(
            verify_residue(self.full, 6, "Y", require_observed=True))

    def test_the_wild_type_check_still_works_without_the_flag(self) -> None:
        index = self.partial.to_index(7)
        self.assertIsNotNone(verify_residue(self.partial, index, "Y"))
        with self.assertRaises(NumberingError):
            verify_residue(self.partial, index, "A")


class StructureFilenameTests(unittest.TestCase):
    """One file per (candidate, structure) pair, and inside the run."""

    def stem(self, candidate: str, structure: str) -> str:
        return PrepareStructures._storage_stem(candidate, structure)

    def test_the_ambiguous_pair_no_longer_collides(self) -> None:
        self.assertNotEqual(self.stem("enzyme__model", "v1"),
                            self.stem("enzyme", "model__v1"))

    def test_distinct_pairs_stay_distinct(self) -> None:
        pairs = [("enzyme__model", "v1"), ("enzyme", "model__v1"),
                 ("enzyme", "v1"), ("enzyme_model", "v1")]
        stems = {self.stem(c, s) for c, s in pairs}
        self.assertEqual(len(stems), len(pairs))

    def test_the_same_pair_is_stable(self) -> None:
        self.assertEqual(self.stem("c1", "s1"), self.stem("c1", "s1"))

    def test_a_traversing_id_cannot_escape(self) -> None:
        for bad in ("../../escape", "a/b", "..", "/absolute"):
            stem = self.stem(bad, "v1")
            self.assertNotIn("/", stem)
            self.assertNotIn("..", stem)

    def test_a_path_outside_the_run_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            with self.assertRaises(Exception):
                PrepareStructures._within(root, root / ".." / "outside.cif")

    def test_a_path_inside_the_run_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            self.assertIsNotNone(PrepareStructures._within(root, root / "a.cif"))


class BundleIdentityTests(unittest.TestCase):
    """A package contains this run's evidence or records it as missing."""

    def setUp(self) -> None:
        self._cwd = os.getcwd()
        self.addCleanup(os.chdir, self._cwd)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)

    def _run_with_manifest(self):
        run = self.root / "RUN-A"
        run.mkdir()
        evidence = run / "evidence_records.jsonl"
        evidence.write_text('{"task":"TASK-A"}\n')
        manifest = RunManifest(run_id="RUN-A", task_id="TASK-A")
        manifest.record("s1", "retrieve_evidence", ToolResult(
            status=Status.SUCCESS,
            artifacts=[Artifact(key="evidence_records",
                                path="evidence_records.jsonl",
                                sha256=sha256_file(evidence))]), "T")
        return run, manifest

    def test_a_same_named_file_from_another_task_is_refused(self) -> None:
        _, manifest = self._run_with_manifest()
        decoy_dir = self.root / "OTHER"
        decoy_dir.mkdir()
        (decoy_dir / "evidence_records.jsonl").write_text('{"task":"TASK-B"}\n')
        moved = self.root / "RUN-A-MOVED"
        moved.mkdir()
        os.chdir(decoy_dir)
        item = next(i for i in BUNDLE_ITEMS
                    if "evidence_records" in i.artifact_keys)
        path, _, notes = _resolve_source(item, moved, manifest)
        self.assertIsNone(path, "a file of the right name is not this run's evidence")
        self.assertTrue([n for n in notes if "does not match the hash" in n])

    def test_the_correct_file_is_still_found(self) -> None:
        run, manifest = self._run_with_manifest()
        item = next(i for i in BUNDLE_ITEMS
                    if "evidence_records" in i.artifact_keys)
        path, how, _ = _resolve_source(item, run, manifest)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertIn("TASK-A", path.read_text())

    def test_assembling_into_the_run_directory_is_refused(self) -> None:
        run, _ = self._run_with_manifest()
        (run / "run_manifest.json").write_text(json.dumps({"run_id": "RUN-A"}))
        with self.assertRaises(ValueError) as ctx:
            assemble_bundle(run, out_dir=run, overwrite=True)
        self.assertIn("run directory itself", str(ctx.exception))
        self.assertTrue((run / "evidence_records.jsonl").exists(),
                        "the source must survive a refused assembly")

    def test_assembling_into_a_parent_of_the_run_is_refused(self) -> None:
        run, _ = self._run_with_manifest()
        with self.assertRaises(ValueError) as ctx:
            assemble_bundle(run, out_dir=self.root, overwrite=True)
        self.assertIn("contains the run directory", str(ctx.exception))
        self.assertTrue((run / "evidence_records.jsonl").exists())

    def test_the_default_nested_output_still_works(self) -> None:
        run, _ = self._run_with_manifest()
        (run / "run_manifest.json").write_text(json.dumps({"run_id": "RUN-A"}))
        assemble_bundle(run, overwrite=True)
        self.assertTrue((run / "bundle").is_dir())
        self.assertTrue((run / "evidence_records.jsonl").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
