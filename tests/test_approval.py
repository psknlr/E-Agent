"""Tests for :mod:`eagent.harness.approval`.

The cases are the ways a human decision point degrades into a formality:

* a gate cleared by a boolean somebody typed into a YAML file rather than by a
  person deciding;
* an approval with nobody's name on it;
* a decision quietly re-made after the fact, so the record no longer says what
  was agreed;
* a grant given for one batch inherited by a different, larger one;
* a queue that lives in memory, so a resumed run re-asks or, worse, proceeds;
* an authorisation request that states a cost nobody computed.

The batch gate gets the most attention because it is the one that spends
money.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eagent.errors import ApprovalRequiredError
from eagent.harness.approval import (
    APPROVAL_GATES,
    BATCH_GATE,
    CRITERIA_GATE,
    DECISION_POINTS,
    REACTION_GATE,
    ApprovalQueue,
    Decision,
    RequestKind,
    batch_cost_payload,
    guard_batch_selection,
)
from eagent.provenance import RunManifest
from eagent.schemas import Approval, TaskSpec


def make_queue(tmp: Path) -> tuple[ApprovalQueue, RunManifest]:
    manifest = RunManifest(run_id="R1", task_id="T1")
    return ApprovalQueue(tmp / "approvals.json", manifest), manifest


class GateVocabularyTests(unittest.TestCase):
    """There are three gates, and they are the ones the schema declares."""

    def test_three_gates_match_the_task_schema(self):
        self.assertEqual(set(APPROVAL_GATES),
                         set(Approval.model_fields),
                         "a gate name that the TaskSpec does not carry is a "
                         "gate nothing ever checks")

    def test_every_gate_has_a_decision_point_describing_it(self):
        self.assertEqual(set(DECISION_POINTS), set(APPROVAL_GATES))
        for gate, point in DECISION_POINTS.items():
            self.assertEqual(point.gate, gate)
            self.assertTrue(point.must_show)
            self.assertTrue(point.consequence_if_wrong)

    def test_a_fourth_gate_cannot_be_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue, _ = make_queue(Path(tmp))
            with self.assertRaises(ValueError):
                queue.request("looks_fine_to_me")

    def test_an_operator_task_may_use_any_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue, _ = make_queue(Path(tmp))
            req = queue.request("operator_fetch_structures",
                                kind=RequestKind.OPERATOR_TASK)
            self.assertEqual(req.kind, RequestKind.OPERATOR_TASK)
            self.assertFalse(queue.is_granted("operator_fetch_structures"))


class RequestGrantDenyTests(unittest.TestCase):
    """Who decided, when, and about what."""

    def test_grant_records_actor_and_timestamp_in_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue, manifest = make_queue(Path(tmp))
            req = queue.request(REACTION_GATE, detail="confirm the substrate")
            self.assertTrue(req.is_pending)
            queue.grant(req.request_id, actor="r.chemist",
                        reason="structures checked against the order sheet")
            self.assertEqual(len(manifest.approvals), 1)
            entry = manifest.approvals[0]
            self.assertEqual(entry["gate"], REACTION_GATE)
            self.assertEqual(entry["decision"], "approve")
            self.assertEqual(entry["actor"], "r.chemist")
            self.assertTrue(entry["at"])
            self.assertTrue(manifest.approved(REACTION_GATE))
            self.assertTrue(queue.is_granted(REACTION_GATE))

    def test_an_anonymous_decision_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue, _ = make_queue(Path(tmp))
            req = queue.request(BATCH_GATE)
            with self.assertRaises(ValueError):
                queue.grant(req.request_id, actor="   ")
            self.assertTrue(queue.get(req.request_id).is_pending)

    def test_a_decision_cannot_be_re_made(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue, _ = make_queue(Path(tmp))
            req = queue.request(CRITERIA_GATE)
            queue.deny(req.request_id, actor="s.lead", reason="criterion vague")
            with self.assertRaises(ValueError):
                queue.grant(req.request_id, actor="s.lead")

    def test_deny_leaves_the_gate_shut(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue, manifest = make_queue(Path(tmp))
            req = queue.request(BATCH_GATE)
            queue.deny(req.request_id, actor="s.lead", reason="too expensive")
            self.assertFalse(queue.is_granted(BATCH_GATE))
            self.assertTrue(queue.is_denied(BATCH_GATE))
            self.assertFalse(manifest.approved(BATCH_GATE))

    def test_deciding_without_a_request_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue, _ = make_queue(Path(tmp))
            with self.assertRaises(KeyError):
                queue.grant(BATCH_GATE, actor="s.lead")

    def test_a_gate_may_be_decided_by_naming_the_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue, _ = make_queue(Path(tmp))
            queue.request(REACTION_GATE)
            queue.grant(REACTION_GATE, actor="r.chemist")
            self.assertTrue(queue.is_granted(REACTION_GATE))

    def test_an_identical_pending_request_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue, _ = make_queue(Path(tmp))
            a = queue.request(BATCH_GATE, payload={"n_constructs": 96})
            b = queue.request(BATCH_GATE, payload={"n_constructs": 96})
            self.assertEqual(a.request_id, b.request_id)
            self.assertEqual(len(queue.pending(BATCH_GATE)), 1)

    def test_a_changed_batch_raises_a_new_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue, _ = make_queue(Path(tmp))
            a = queue.request(BATCH_GATE, payload={"n_constructs": 96})
            b = queue.request(BATCH_GATE, payload={"n_constructs": 384})
            self.assertNotEqual(a.request_id, b.request_id,
                                "an approval is an approval of something; a "
                                "bigger batch is a different thing")

    def test_require_names_the_pending_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue, _ = make_queue(Path(tmp))
            req = queue.request(BATCH_GATE)
            with self.assertRaises(ApprovalRequiredError) as caught:
                queue.require(BATCH_GATE)
            self.assertIn(req.request_id, str(caught.exception))


class PersistenceTests(unittest.TestCase):
    """The decision outlives the process that asked for it."""

    def test_round_trips_through_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approvals.json"
            manifest = RunManifest(run_id="R1", task_id="T1")
            queue = ApprovalQueue(path, manifest)
            req = queue.request(BATCH_GATE, payload={"n_constructs": 96},
                                detail="round 1")
            queue.grant(req.request_id, actor="s.lead", reason="budget ok")
            self.assertTrue(path.exists())

            reloaded = ApprovalQueue.load(path)
            self.assertEqual(len(reloaded), 1)
            restored = reloaded.get(req.request_id)
            self.assertEqual(restored.decision, Decision.APPROVE)
            self.assertEqual(restored.actor, "s.lead")
            self.assertEqual(restored.payload, {"n_constructs": 96})
            self.assertTrue(reloaded.is_granted(BATCH_GATE))

    def test_loading_a_queue_from_the_future_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approvals.json"
            path.write_text(json.dumps({"requests": [
                {"request_id": "x", "gate": BATCH_GATE,
                 "auto_approved_by_policy": True}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                ApprovalQueue.load(path)

    def test_a_manifest_grant_survives_a_lost_queue_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "approvals.json"
            manifest = RunManifest(run_id="R1", task_id="T1")
            manifest.record_approval(BATCH_GATE, "approve", "s.lead", "round 1")
            queue = ApprovalQueue.load(path, manifest)
            self.assertEqual(len(queue), 0)
            self.assertTrue(queue.is_granted(BATCH_GATE),
                            "the manifest is the record of record")


class BatchBlockTests(unittest.TestCase):
    """No path reaches batch selection without a recorded grant."""

    def test_blocks_before_any_request_is_raised(self):
        manifest = RunManifest(run_id="R1", task_id="T1")
        with self.assertRaises(ApprovalRequiredError) as caught:
            guard_batch_selection(manifest)
        self.assertIn("no request has been raised", str(caught.exception))

    def test_blocks_while_the_request_is_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue, manifest = make_queue(Path(tmp))
            queue.request(BATCH_GATE, payload={"n_constructs": 96})
            with self.assertRaises(ApprovalRequiredError) as caught:
                guard_batch_selection(manifest, queue)
            self.assertIn("waiting for a decision", str(caught.exception))

    def test_opens_once_a_named_person_grants_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue, manifest = make_queue(Path(tmp))
            req = queue.request(BATCH_GATE, payload={"n_constructs": 96})
            queue.grant(req.request_id, actor="s.lead")
            guard_batch_selection(manifest, queue)   # must not raise

    def test_a_task_file_flag_does_not_open_the_gate(self):
        """The whole point: ``synthesis_authorized: true`` is not a decision."""
        task = TaskSpec(task_id="T1")
        task.approval.synthesis_authorized = True
        manifest = RunManifest(run_id="R1", task_id=task.task_id)
        self.assertTrue(task.approval.state(BATCH_GATE))
        with self.assertRaises(ApprovalRequiredError):
            guard_batch_selection(manifest)

    def test_an_approval_with_no_actor_does_not_open_the_gate(self):
        manifest = RunManifest(run_id="R1", task_id="T1")
        manifest.record_approval(BATCH_GATE, "approve", "", "")
        self.assertTrue(manifest.approved(BATCH_GATE))
        with self.assertRaises(ApprovalRequiredError):
            guard_batch_selection(manifest)

    def test_a_denial_is_reported_with_who_denied_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue, manifest = make_queue(Path(tmp))
            req = queue.request(BATCH_GATE)
            queue.deny(req.request_id, actor="s.lead", reason="no budget")
            with self.assertRaises(ApprovalRequiredError) as caught:
                guard_batch_selection(manifest, queue)
            self.assertIn("s.lead", str(caught.exception))

    def test_a_grant_for_another_gate_does_not_open_this_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue, manifest = make_queue(Path(tmp))
            req = queue.request(REACTION_GATE)
            queue.grant(req.request_id, actor="r.chemist")
            with self.assertRaises(ApprovalRequiredError):
                guard_batch_selection(manifest, queue)


class CostPayloadTests(unittest.TestCase):
    """What the operator is shown before authorising spend."""

    def test_unknown_prices_stay_null_with_a_note(self):
        payload = batch_cost_payload(
            n_constructs=96, n_candidate_wells=576, n_control_wells=24,
            n_plates=7, cofactor_conditions=2, replicates=3,
            controls=["no_enzyme", "empty_vector"])
        self.assertIsNone(payload["cost"]["estimated_total"])
        self.assertIsNone(payload["cost"]["currency"])
        self.assertIn("note", payload["cost"])
        self.assertIn("curator", payload["cost"]["note"])
        self.assertEqual(payload["n_wells_total"], 600)

    def test_supplied_prices_are_multiplied_not_invented(self):
        payload = batch_cost_payload(
            n_constructs=10, n_candidate_wells=60, n_control_wells=6,
            n_plates=1, cofactor_conditions=2, replicates=3,
            currency="EUR", cost_per_construct=30.0, cost_per_plate=120.0)
        self.assertEqual(payload["cost"]["estimated_total"], 420.0)
        self.assertEqual(payload["cost"]["currency"], "EUR")
        self.assertNotIn("note", payload["cost"])

    def test_half_a_price_list_does_not_produce_a_total(self):
        payload = batch_cost_payload(
            n_constructs=96, n_candidate_wells=576, n_control_wells=24,
            n_plates=7, cofactor_conditions=2, replicates=3,
            currency="EUR", cost_per_construct=30.0)
        self.assertIsNone(payload["cost"]["estimated_total"],
                          "treating the unknown plate price as zero would "
                          "quote a total that is confidently too low")
        self.assertIn("per_plate", payload["cost"]["note"])

    def test_a_price_is_not_needed_for_a_count_of_zero(self):
        payload = batch_cost_payload(
            n_constructs=4, n_candidate_wells=8, n_control_wells=0, n_plates=0,
            cofactor_conditions=1, replicates=2, currency="EUR",
            cost_per_construct=25.0)
        self.assertEqual(payload["cost"]["estimated_total"], 100.0)

    def test_the_payload_covers_what_the_decision_point_demands(self):
        payload = batch_cost_payload(
            n_constructs=1, n_candidate_wells=1, n_control_wells=0, n_plates=1,
            cofactor_conditions=1, replicates=1)
        missing = [f for f in DECISION_POINTS[BATCH_GATE].must_show
                   if f not in payload]
        self.assertEqual(missing, [],
                         "the operator cannot authorise a cost they were not "
                         "shown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
