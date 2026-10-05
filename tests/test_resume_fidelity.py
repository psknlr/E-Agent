"""Regression tests: a restart must not change what a round concluded.

Two findings from an external audit, reproduced before being fixed.

The first: a step's input digest hashed a path as its string, so editing the
file in place left the digest unchanged and the step was skipped. The run then
reported a result reproduced from inputs that no longer existed.

The second, and worse: a skipped step never repopulated the controller's
result map. A round whose ingest found a confirmed hit, and which correctly
routed to local engineering, was saved and restarted; the step was skipped,
the downstream stage read nothing, and the run went to the no-hit diagnosis
and recorded that nothing was confirmed. The experimental record never
changed. The conclusion did.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from eagent.envelope import Status, ToolResult
from eagent.harness.controller import _argument_digest
from eagent.provenance import RunManifest


class InputDigestTests(unittest.TestCase):
    """A path argument is identified by what is in it, not where it is."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = pathlib.Path(self._tmp.name)

    def digest(self, **args) -> str:
        return _argument_digest("step", args, "task-digest")[0]

    def test_editing_a_file_in_place_invalidates_the_digest(self) -> None:
        f = self.tmp / "input.txt"
        f.write_text("original-content")
        before = self.digest(path=f)
        f.write_text("changed-content")
        self.assertNotEqual(
            before, self.digest(path=f),
            "the same path with different bytes is a different input")

    def test_restoring_the_content_restores_the_digest(self) -> None:
        f = self.tmp / "input.txt"
        f.write_text("a")
        first = self.digest(path=f)
        f.write_text("b")
        f.write_text("a")
        self.assertEqual(first, self.digest(path=f),
                         "identical content must resume, or nothing ever does")

    def test_creating_a_previously_missing_file_invalidates(self) -> None:
        f = self.tmp / "absent.txt"
        before = self.digest(path=f)
        f.write_text("now it exists")
        self.assertNotEqual(before, self.digest(path=f))

    def test_a_change_inside_a_directory_argument_invalidates(self) -> None:
        d = self.tmp / "dir"
        d.mkdir()
        (d / "a.txt").write_text("a")
        before = self.digest(path=d)
        (d / "a.txt").write_text("b")
        self.assertNotEqual(before, self.digest(path=d))

    def test_a_new_file_in_a_directory_argument_invalidates(self) -> None:
        d = self.tmp / "dir"
        d.mkdir()
        (d / "a.txt").write_text("a")
        before = self.digest(path=d)
        (d / "b.txt").write_text("b")
        self.assertNotEqual(before, self.digest(path=d))

    def test_the_digest_is_stable_for_an_unchanged_file(self) -> None:
        f = self.tmp / "input.txt"
        f.write_text("steady")
        _, stable = _argument_digest("step", {"path": f}, "t")
        self.assertTrue(stable, "a readable file must not mark a step unstable")


class ResultPersistenceTests(unittest.TestCase):
    """A step may only be skipped if its result can be handed on."""

    def setUp(self) -> None:
        self.manifest = RunManifest(run_id="r", task_id="t")

    def record(self, data):
        return self.manifest.record(
            "s1", "ingest_results",
            ToolResult(status=Status.SUCCESS, data=data), "2026-01-01T00:00:00Z")

    def test_a_plain_payload_is_stored_and_marked_restorable(self) -> None:
        rec = self.record({"confirmed_target_product": 1, "hits": ["c1"]})
        self.assertTrue(rec.data_restorable)
        self.assertEqual(rec.data, {"confirmed_target_product": 1, "hits": ["c1"]})

    def test_a_live_object_is_not_claimed_restorable(self) -> None:
        """str() coercion would store a repr with a memory address in it."""
        rec = self.record({"obj": object()})
        self.assertFalse(rec.data_restorable)
        self.assertIsNone(rec.data)

    def test_a_tuple_is_not_claimed_restorable(self) -> None:
        """It would come back a list, which is not what the step returned."""
        rec = self.record({"pair": (1, 2)})
        self.assertFalse(rec.data_restorable)

    def test_an_oversized_payload_is_not_stored(self) -> None:
        rec = self.record({"big": "x" * (RunManifest.MAX_RESUMABLE_DATA_BYTES + 10)})
        self.assertFalse(rec.data_restorable)
        self.assertIsNone(rec.data)

    def test_an_empty_payload_is_restorable(self) -> None:
        self.assertTrue(self.record({}).data_restorable)

    def test_the_record_survives_a_manifest_round_trip(self) -> None:
        self.record({"confirmed_target_product": 1})
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "manifest.json"
            self.manifest.write(path)
            loaded = RunManifest.load(path)
        step = loaded.steps[-1]
        self.assertTrue(step.data_restorable)
        self.assertEqual(step.data, {"confirmed_target_product": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
