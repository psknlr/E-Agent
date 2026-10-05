"""Regression tests: an approval is an approval *of something*.

From an external audit, reproduced before being fixed. A grant recorded for a
small pilot released a later batch that a human had explicitly refused,
because the only thing the check matched on was the gate name, and every gate
name is shared by every batch that will ever be proposed under it. The queue
simultaneously reported the gate as granted and as denied.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from eagent.errors import ApprovalRequiredError
from eagent.harness.approval import (
    ApprovalQueue, BATCH_GATE, guard_batch_selection,
)
from eagent.provenance import RunManifest

PILOT = {"batch_id": "A", "constructs": 8, "cost": 100}
EXPENSIVE = {"batch_id": "B", "constructs": 960, "cost": 10000}


class QueueBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.manifest = RunManifest(run_id="r", task_id="t")
        self.queue = ApprovalQueue(
            pathlib.Path(self._tmp.name) / "approvals.json", self.manifest)

    def _grant(self, payload) -> None:
        req = self.queue.request(BATCH_GATE, payload=payload)
        self.queue.grant(req.request_id, actor="scientist", reason="approved")

    def _deny(self, payload) -> None:
        req = self.queue.request(BATCH_GATE, payload=payload)
        self.queue.deny(req.request_id, actor="scientist", reason="too costly")

    def test_a_grant_does_not_carry_to_a_different_batch(self) -> None:
        self._grant(PILOT)
        self.assertTrue(self.queue.is_granted(BATCH_GATE, PILOT))
        self.assertFalse(
            self.queue.is_granted(BATCH_GATE, EXPENSIVE),
            "a pilot approval must not release a hundredfold larger batch")

    def test_an_explicitly_denied_batch_stays_denied(self) -> None:
        self._grant(PILOT)
        self._deny(EXPENSIVE)
        self.assertFalse(self.queue.is_granted(BATCH_GATE, EXPENSIVE))
        self.assertTrue(self.queue.is_denied(BATCH_GATE, EXPENSIVE))

    def test_the_earlier_grant_survives_for_its_own_batch(self) -> None:
        """Refusing B must not retroactively revoke the decision about A."""
        self._grant(PILOT)
        self._deny(EXPENSIVE)
        self.assertTrue(self.queue.is_granted(BATCH_GATE, PILOT))

    def test_granted_and_denied_are_never_both_true_for_one_payload(self) -> None:
        self._grant(PILOT)
        self._deny(EXPENSIVE)
        for payload in (PILOT, EXPENSIVE):
            self.assertFalse(
                self.queue.is_granted(BATCH_GATE, payload)
                and self.queue.is_denied(BATCH_GATE, payload))

    def test_any_change_to_the_payload_needs_a_new_decision(self) -> None:
        self._grant(PILOT)
        for field, value in (("cost", 101), ("constructs", 9), ("batch_id", "A2")):
            changed = dict(PILOT, **{field: value})
            self.assertFalse(
                self.queue.is_granted(BATCH_GATE, changed),
                f"changing {field} must invalidate the grant")

    def test_the_latest_decision_governs_at_gate_level_too(self) -> None:
        """Without a payload, a newer refusal still shuts the gate."""
        self._grant(PILOT)
        self.assertTrue(self.queue.is_granted(BATCH_GATE))
        self._deny(EXPENSIVE)
        self.assertFalse(self.queue.is_granted(BATCH_GATE))

    def test_require_names_why_an_unrelated_grant_does_not_help(self) -> None:
        self._grant(PILOT)
        with self.assertRaises(ApprovalRequiredError) as ctx:
            self.queue.require(BATCH_GATE, "ordering genes: ", payload=EXPENSIVE)
        message = str(ctx.exception)
        self.assertIn("DIFFERENT work", message)
        self.assertIn("scientist", message)
        self.assertNotIn(
            "no request has even been raised", message,
            "telling an operator who just approved something that no request "
            "was raised reads as the tool losing their decision")

    def test_require_passes_for_the_payload_that_was_approved(self) -> None:
        self._grant(PILOT)
        self.queue.require(BATCH_GATE, payload=PILOT)


class ManifestGuardTests(unittest.TestCase):
    """The hard block in front of batch selection reads the manifest."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.manifest = RunManifest(run_id="r", task_id="t")
        self.queue = ApprovalQueue(
            pathlib.Path(self._tmp.name) / "approvals.json", self.manifest)
        req = self.queue.request(BATCH_GATE, payload=PILOT)
        self.queue.grant(req.request_id, actor="scientist")

    def test_the_decision_records_what_it_was_about(self) -> None:
        entry = self.manifest.approvals[-1]
        self.assertTrue(entry.get("payload_sha256"))
        self.assertTrue(entry.get("request_id"))

    def test_the_guard_passes_for_the_approved_batch(self) -> None:
        guard_batch_selection(self.manifest, self.queue, payload=PILOT)

    def test_the_guard_blocks_a_batch_nobody_decided(self) -> None:
        with self.assertRaises(ApprovalRequiredError) as ctx:
            guard_batch_selection(self.manifest, self.queue, payload=EXPENSIVE)
        self.assertIn("DIFFERENT batch", str(ctx.exception))

    def test_the_guard_blocks_a_denied_batch(self) -> None:
        req = self.queue.request(BATCH_GATE, payload=EXPENSIVE)
        self.queue.deny(req.request_id, actor="scientist", reason="too costly")
        with self.assertRaises(ApprovalRequiredError) as ctx:
            guard_batch_selection(self.manifest, self.queue, payload=EXPENSIVE)
        self.assertIn("denied", str(ctx.exception))

    def test_manifest_approved_honours_the_payload_hash(self) -> None:
        wanted = self.manifest.approvals[-1]["payload_sha256"]
        self.assertTrue(self.manifest.approved(BATCH_GATE, wanted))
        self.assertFalse(self.manifest.approved(BATCH_GATE, "sha256:other"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
