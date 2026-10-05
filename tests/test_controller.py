"""Tests for :mod:`eagent.harness.controller` and :mod:`eagent.harness.registry`.

The cases are the ways an agent loop stops being a protocol:

* every failure funnelled into one "try again", so a missing binary is
  re-run until the compute budget is gone;
* a retry loop with no cap, or one capped only by cost;
* a repair that repairs nothing, re-running to the identical failure;
* widening the search forever instead of admitting the evidence is thin;
* a resumed run redoing work it already did, or skipping work whose inputs
  changed underneath it;
* a ``next_action`` marked ``requires_human`` being executed because it
  happened to name an interface;
* reaching batch selection without a recorded grant;
* a transition nobody declared.

The interfaces are scripted stand-ins. That is deliberate: the controller is
supposed to route on the envelope and never on the chemistry, so a test that
needed real chemistry to exercise it would be evidence of a leak.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

from eagent.context import ExecutionPolicy, RunContext
from eagent.envelope import Provenance, Severity, Status, ToolResult
from eagent.harness.approval import ApprovalQueue, BATCH_GATE, CRITERIA_GATE, REACTION_GATE
from eagent.harness.controller import (
    Branch,
    ControllerHooks,
    FailureKind,
    RETRY_POLICY,
    ResearchController,
    RunOutcome,
    Stage,
    TERMINAL_STAGES,
    TRANSITIONS,
    classify_failure,
)
from eagent.harness.registry import (
    PROTOCOL_ORDER,
    build_interface_registry,
    dependency_problems,
    registry_report,
    topological_order,
)
from eagent.provenance import RunManifest
from eagent.schemas import TaskSpec
from eagent.tools.base import InterfaceRegistry, ScientificInterface


# ---------------------------------------------------------------------------
# scripted interfaces
# ---------------------------------------------------------------------------

class ScriptedInterface(ScientificInterface):
    """An interface that returns prepared envelopes and counts its calls."""

    def __init__(self, name: str, script: Sequence[Any] | None = None) -> None:
        self.name = name                     # shadows the ClassVar on purpose
        self.depends_on: tuple[str, ...] = ()
        self.required_fields: tuple[str, ...] = ()
        self.required_approvals: tuple[str, ...] = ()
        self.calls: list[dict[str, Any]] = []
        self._script = list(script or [])

    def execute(self, ctx: RunContext, **kwargs: Any) -> ToolResult:
        self.calls.append(dict(kwargs))
        index = min(len(self.calls) - 1, len(self._script) - 1)
        if not self._script:
            return ok(self.name)
        item = self._script[index]
        return item(ctx, **kwargs) if callable(item) else item


def ok(name: str, **data: Any) -> ToolResult:
    return ToolResult(status=Status.SUCCESS,
                      provenance=Provenance(tool=name), data=dict(data))


def input_error(name: str, code: str = "missing_input") -> ToolResult:
    return ToolResult.failure(name, f"{name}: nothing to work on", code=code)


def insufficient(name: str) -> ToolResult:
    result = ToolResult(status=Status.PARTIAL, provenance=Provenance(tool=name),
                        message="thin")
    result.add_flag("evidence_gaps", Severity.WARN,
                    "two of three planned queries returned nothing")
    return result


def needs_human(name: str, gate: str = REACTION_GATE) -> ToolResult:
    result = ToolResult.failure(name, "gate not cleared", code="approval_required")
    result.add_next("request_approval", "a human must decide", {"gate": gate},
                    requires_human=True)
    return result


def unavailable(name: str) -> ToolResult:
    return ToolResult.failure(name, "mmseqs2 is not installed",
                              code="tool_unavailable")


def internal(name: str) -> ToolResult:
    return ToolResult.failure(name, "unhandled KeyError: 'chain'",
                              code="internal_error")


def costly(name: str, **cost: float) -> ToolResult:
    result = input_error(name)
    result.provenance = Provenance(tool=name, cost=dict(cost))
    return result


def scripted_registry(**overrides: Sequence[Any]) -> InterfaceRegistry:
    """All ten protocol steps, succeeding unless a script says otherwise."""
    registry = InterfaceRegistry()
    for name in PROTOCOL_ORDER:
        registry.register(ScriptedInterface(name, overrides.get(name)))
    registry.missing_interfaces = []         # type: ignore[attr-defined]
    return registry


def make_ctx(tmp: Path, *, max_retries: int = 1,
             cost_ceiling: dict[str, float] | None = None,
             manifest: RunManifest | None = None) -> RunContext:
    task = TaskSpec(task_id="T1")
    return RunContext(
        task=task, workdir=tmp,
        manifest=manifest or RunManifest(run_id="R1", task_id="T1"),
        policy=ExecutionPolicy(max_retries=max_retries,
                               cost_ceiling=dict(cost_ceiling or {})))


def make_controller(tmp: Path, registry: InterfaceRegistry | None = None,
                    *, hooks: ControllerHooks | None = None,
                    arguments: dict[str, dict[str, Any]] | None = None,
                    granted: Sequence[str] = (),
                    ctx: RunContext | None = None,
                    **ctx_kwargs: Any) -> ResearchController:
    ctx = ctx or make_ctx(tmp, **ctx_kwargs)
    queue = ApprovalQueue(Path(tmp) / "approvals.json", ctx.manifest)
    controller = ResearchController(ctx, registry or scripted_registry(), queue,
                                    hooks=hooks, arguments=arguments)
    # Pre-authorise the payload the controller will actually present, not a
    # placeholder. A grant recorded against different content is not a grant
    # for this work, and the gate is right to refuse it -- so the fixture
    # approves the real thing rather than the check being loosened for it.
    for gate in granted:
        queue.request(gate, payload=controller.gate_payload(gate))
        queue.grant(gate, actor="test.operator", reason="fixture grant")
    return controller


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

class RegistryTests(unittest.TestCase):
    """The ten real interfaces, assembled the way a run assembles them."""

    def test_all_ten_protocol_interfaces_register(self):
        registry = build_interface_registry()
        self.assertEqual(set(registry.names()), set(PROTOCOL_ORDER))
        self.assertEqual(len(PROTOCOL_ORDER), 10)

    def test_topological_order_covers_every_interface_and_respects_depends_on(self):
        registry = build_interface_registry()
        order = topological_order(registry)
        self.assertEqual(set(order), set(PROTOCOL_ORDER))
        position = {name: i for i, name in enumerate(order)}
        for iface in registry:
            for dep in iface.depends_on:
                self.assertLess(position[dep], position[iface.name],
                                f"{dep} must be orderable before {iface.name}")

    def test_no_declared_dependency_is_missing(self):
        self.assertEqual(dependency_problems(build_interface_registry()), [])

    def test_strict_build_refuses_an_interface_that_does_not_exist(self):
        with self.assertRaises(LookupError):
            build_interface_registry(["normalize_reaction", "screen_geometry"],
                                     strict=True)

    def test_non_strict_build_reports_the_gap_instead_of_hiding_it(self):
        registry = build_interface_registry(
            ["normalize_reaction", "screen_geometry"], strict=False)
        self.assertEqual(registry.names(), ["normalize_reaction"])
        self.assertTrue(any("screen_geometry" in m
                            for m in registry.missing_interfaces))

    def test_report_is_manifest_ready(self):
        report = registry_report(build_interface_registry())
        self.assertEqual(report["unregistered_protocol_steps"], [])
        self.assertEqual(report["missing"], [])
        self.assertEqual(len(report["interfaces"]), 10)


class TransitionTableTests(unittest.TestCase):
    """Control flow is a declared graph, not an emergent one."""

    def test_every_non_terminal_stage_declares_its_moves(self):
        for stage in Stage:
            if stage in TERMINAL_STAGES:
                continue
            self.assertIn(stage, TRANSITIONS, f"{stage} declares no moves")

    def test_every_declared_target_is_a_real_stage(self):
        for source, targets in TRANSITIONS.items():
            for target in targets:
                self.assertIsInstance(target, Stage, source)

    def test_an_undeclared_move_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = make_controller(Path(tmp))
            controller.stage = Stage.START
            with self.assertRaises(ValueError) as caught:
                controller.goto(Stage.SELECT_BATCH)
            self.assertIn("illegal transition", str(caught.exception))

    def test_batch_selection_is_only_reachable_through_batch_approval(self):
        predecessors = {source for source, targets in TRANSITIONS.items()
                        if Stage.SELECT_BATCH in targets}
        self.assertEqual(predecessors,
                         {Stage.BATCH_APPROVAL, Stage.REPAIR_INPUTS},
                         "the only way in is the approval stage; the repair "
                         "edge can only be taken after select_batch has "
                         "already run once behind the same block")

    def test_going_back_is_a_declared_move(self):
        self.assertIn(Stage.PREPARE_STRUCTURES,
                      TRANSITIONS[Stage.REPAIR_INPUTS])
        self.assertIn(Stage.RETRIEVE_EVIDENCE,
                      TRANSITIONS[Stage.WIDEN_RETRIEVAL])


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

class ClassificationTests(unittest.TestCase):
    """One failure kind per recovery, because they are not interchangeable."""

    def test_each_family_of_codes_maps_to_its_kind(self):
        cases = {
            FailureKind.INPUT_ERROR: input_error("x"),
            FailureKind.NEEDS_HUMAN: needs_human("x"),
            FailureKind.TOOL_UNAVAILABLE: unavailable("x"),
            FailureKind.INTERNAL_ERROR: internal("x"),
            FailureKind.INSUFFICIENT_EVIDENCE: insufficient("x"),
            FailureKind.NONE: ok("x"),
        }
        for kind, result in cases.items():
            self.assertIs(classify_failure(result), kind, kind.value)

    def test_a_human_decision_outranks_thin_evidence(self):
        result = needs_human("x")
        result.add_flag("evidence_gaps", Severity.WARN, "also thin")
        self.assertIs(classify_failure(result), FailureKind.NEEDS_HUMAN,
                      "widening a search while the substrate is unconfirmed "
                      "searches harder for the wrong thing")

    def test_a_step_that_asks_for_a_person_is_not_repaired_behind_their_back(self):
        result = input_error("x")
        result.add_next("operator_fetch_structures",
                        "only the operator can put the files on this machine",
                        {}, requires_human=True)
        self.assertIs(classify_failure(result), FailureKind.NEEDS_HUMAN,
                      "the step said a person has to act; a retry cannot "
                      "supply what it is asking for")

    def test_an_unrecognised_blocker_is_not_quietly_retried(self):
        result = ToolResult.failure("x", "something new", code="who_knows")
        self.assertIs(classify_failure(result), FailureKind.UNCLASSIFIED)
        self.assertEqual(RETRY_POLICY[FailureKind.UNCLASSIFIED].action,
                         "escalate")

    def test_only_two_kinds_are_re_runnable(self):
        rerunnable = {k for k, rule in RETRY_POLICY.items()
                      if rule.max_attempts > 1}
        self.assertEqual(rerunnable, {FailureKind.INPUT_ERROR,
                                      FailureKind.INSUFFICIENT_EVIDENCE})


# ---------------------------------------------------------------------------
# gates along the path
# ---------------------------------------------------------------------------

class GatedFlowTests(unittest.TestCase):
    """The run stops at each human decision point."""

    def test_stops_before_retrieval_until_the_spec_is_confirmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = make_controller(Path(tmp))
            outcome = controller.run()
            self.assertIs(outcome, RunOutcome.AWAITING_HUMAN)
            self.assertIs(controller.stage, Stage.AWAITING_HUMAN)
            self.assertTrue(controller.queue.pending(REACTION_GATE))
            self.assertNotIn(Stage.MINE_SEQUENCES, controller.path)

    def test_halts_when_the_operator_rejects_the_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = make_controller(Path(tmp))
            controller.run()
            request = controller.queue.pending(REACTION_GATE)[-1]
            controller.queue.deny(request.request_id, actor="r.chemist",
                                  reason="wrong substrate")
            controller.stage = Stage.AWAIT_REACTION_APPROVAL
            self.assertIs(controller.run(), RunOutcome.HALTED)

    def test_stops_at_batch_approval_with_the_cost_payload_queued(self):
        with tempfile.TemporaryDirectory() as tmp:
            hooks = ControllerHooks(
                batch_payload=lambda c: {"n_constructs": 96, "n_plates": 7})
            controller = make_controller(Path(tmp), granted=[REACTION_GATE],
                                         hooks=hooks)
            self.assertIs(controller.run(), RunOutcome.AWAITING_HUMAN)
            self.assertIs(controller.stage, Stage.AWAITING_HUMAN)
            pending = controller.queue.pending(BATCH_GATE)
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].payload["n_constructs"], 96)
            self.assertNotIn(Stage.SELECT_BATCH, controller.path)

    def test_batch_selection_runs_only_after_a_recorded_grant(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry()
            controller = make_controller(
                Path(tmp), registry, granted=[REACTION_GATE, BATCH_GATE])
            self.assertIs(controller.run(), RunOutcome.AWAITING_RESULTS)
            self.assertIn(Stage.SELECT_BATCH, controller.path)
            self.assertEqual(len(registry.get("select_batch").calls), 1)

    def test_a_task_flag_alone_never_reaches_batch_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            ctx.task.approval.synthesis_authorized = True
            registry = scripted_registry()
            controller = make_controller(Path(tmp), registry, ctx=ctx,
                                         granted=[REACTION_GATE])
            controller.run()
            self.assertEqual(registry.get("select_batch").calls, [])
            self.assertIn("named person", controller.stop_reason)
            self.assertTrue(controller.queue.pending(BATCH_GATE))

    def test_results_are_not_read_before_the_hit_definition_is_fixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry()
            hooks = ControllerHooks(results_ready=lambda c: True)
            controller = make_controller(
                Path(tmp), registry, hooks=hooks,
                granted=[REACTION_GATE, BATCH_GATE])
            self.assertIs(controller.run(), RunOutcome.AWAITING_HUMAN)
            self.assertEqual(registry.get("ingest_results").calls, [])
            self.assertTrue(controller.queue.pending(CRITERIA_GATE))


class HitBranchTests(unittest.TestCase):
    """What happens when the plate comes back."""

    def _run(self, ingest_result: ToolResult) -> ResearchController:
        tmp = self._tmp
        registry = scripted_registry(ingest_results=[ingest_result])
        hooks = ControllerHooks(results_ready=lambda c: True)
        controller = make_controller(
            Path(tmp), registry, hooks=hooks,
            granted=[REACTION_GATE, BATCH_GATE, CRITERIA_GATE])
        controller.run()
        self.registry = registry
        return controller

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self._tmp = self._dir.name

    def tearDown(self):
        self._dir.cleanup()

    def test_a_confirmed_hit_goes_to_local_engineering(self):
        controller = self._run(ok("ingest_results", outcome_counts={
            "confirmed_target_product": 2, "no_target_product_detected": 90}))
        self.assertIs(controller.outcome, RunOutcome.COMPLETED)
        self.assertIn(Stage.LOCAL_ENGINEERING, controller.path)
        self.assertEqual(len(self.registry.get("propose_mutations").calls), 1)

    def test_no_hit_goes_to_failure_cause_diagnosis(self):
        controller = self._run(ok(
            "ingest_results",
            outcome_counts={"no_target_product_detected": 96},
            no_hit_diagnosis={"leading": "expression", "evidence": ["..."]}))
        self.assertIs(controller.outcome, RunOutcome.COMPLETED)
        self.assertIn(Stage.DIAGNOSE_NO_HIT, controller.path)
        self.assertEqual(controller.no_hit_diagnosis["leading"], "expression")
        self.assertEqual(self.registry.get("propose_mutations").calls, [])

    def test_the_controller_does_not_invent_a_diagnosis(self):
        controller = self._run(ok("ingest_results", outcome_counts={}))
        self.assertIsNone(controller.no_hit_diagnosis)
        self.assertTrue(any("cause is unknown" in n
                            for n in controller.manifest.notes))


# ---------------------------------------------------------------------------
# branching and retries
# ---------------------------------------------------------------------------

class BranchingTests(unittest.TestCase):
    """The three-way branch after catalytic complex evaluation."""

    def test_input_error_repairs_and_re_runs_the_affected_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry(evaluate_catalysis=[
                input_error("evaluate_catalysis"), ok("evaluate_catalysis")])
            repairs: list[str] = []

            def repair(controller, attempt):
                repairs.append(attempt.step_id)
                return True

            controller = make_controller(
                Path(tmp), registry, hooks=ControllerHooks(repair=repair),
                granted=[REACTION_GATE])
            controller.run()
            self.assertIn(Stage.REPAIR_INPUTS, controller.path)
            self.assertEqual(len(registry.get("evaluate_catalysis").calls), 2)
            self.assertEqual(len(repairs), 1)
            self.assertIn(Stage.BATCH_APPROVAL, controller.path)
            self.assertEqual(controller.escalations, [])

    def test_insufficient_evidence_widens_rather_than_repairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry(evaluate_catalysis=[
                insufficient("evaluate_catalysis"), ok("evaluate_catalysis")])
            widened: list[str] = []
            repaired: list[str] = []

            controller = make_controller(
                Path(tmp), registry,
                hooks=ControllerHooks(
                    widen=lambda c, a: (widened.append(a.step_id) or True),
                    repair=lambda c, a: (repaired.append(a.step_id) or True)),
                granted=[REACTION_GATE])
            controller.run()
            self.assertIn(Stage.WIDEN_RETRIEVAL, controller.path)
            self.assertNotIn(Stage.REPAIR_INPUTS, controller.path)
            self.assertEqual(len(widened), 1)
            self.assertEqual(repaired, [])
            self.assertEqual(len(registry.get("evaluate_catalysis").calls), 2)

    def test_the_two_branches_are_distinguished_by_the_envelope_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = make_controller(Path(tmp))
            self.assertIs(controller.branch_of(input_error("x")),
                          Branch.INPUT_OR_MAPPING_ERROR)
            self.assertIs(controller.branch_of(insufficient("x")),
                          Branch.INSUFFICIENT_EVIDENCE)
            self.assertIs(controller.branch_of(ok("x")),
                          Branch.SUFFICIENT_SUPPORT)

    def test_exhausted_widening_keeps_the_candidates_as_exploration_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry(evaluate_catalysis=[
                insufficient("evaluate_catalysis")])
            controller = make_controller(
                Path(tmp), registry,
                hooks=ControllerHooks(widen=lambda c, a: True),
                granted=[REACTION_GATE])
            controller.run()
            self.assertIn(Stage.BATCH_APPROVAL, controller.path)
            self.assertTrue(controller.exploration_candidates)
            self.assertLessEqual(
                len(registry.get("evaluate_catalysis").calls), 2,
                "widening is capped; a search that keeps widening is looking "
                "for a result that may not be there")

    def test_widening_with_no_hook_does_not_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry(evaluate_catalysis=[
                insufficient("evaluate_catalysis")])
            controller = make_controller(Path(tmp), registry,
                                         granted=[REACTION_GATE])
            controller.run()
            self.assertEqual(len(registry.get("evaluate_catalysis").calls), 1)
            self.assertTrue(controller.exploration_candidates)


class RetryPolicyTests(unittest.TestCase):
    """Bounded, per-kind, and escalating."""

    def test_a_repeated_input_error_escalates_after_the_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry(evaluate_catalysis=[
                input_error("evaluate_catalysis")])
            controller = make_controller(
                Path(tmp), registry,
                hooks=ControllerHooks(repair=lambda c, a: True),
                granted=[REACTION_GATE])
            self.assertIs(controller.run(), RunOutcome.ESCALATED)
            self.assertEqual(len(registry.get("evaluate_catalysis").calls), 2)
            escalation = controller.escalations[-1]
            self.assertIs(escalation.kind, FailureKind.INPUT_ERROR)
            self.assertEqual(escalation.attempts, 2)
            self.assertTrue(controller.queue.pending("operator_review"))

    def test_max_retries_zero_means_one_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry(evaluate_catalysis=[
                input_error("evaluate_catalysis")])
            controller = make_controller(
                Path(tmp), registry, max_retries=0,
                hooks=ControllerHooks(repair=lambda c, a: True),
                granted=[REACTION_GATE])
            self.assertIs(controller.run(), RunOutcome.ESCALATED)
            self.assertEqual(len(registry.get("evaluate_catalysis").calls), 1)

    def test_a_generous_max_retries_does_not_raise_the_per_kind_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry(evaluate_catalysis=[
                input_error("evaluate_catalysis")])
            controller = make_controller(
                Path(tmp), registry, max_retries=50,
                hooks=ControllerHooks(repair=lambda c, a: True),
                granted=[REACTION_GATE])
            controller.run()
            self.assertEqual(len(registry.get("evaluate_catalysis").calls), 2)

    def test_a_repair_that_changes_nothing_escalates_immediately(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry(evaluate_catalysis=[
                input_error("evaluate_catalysis")])
            controller = make_controller(
                Path(tmp), registry,
                hooks=ControllerHooks(repair=lambda c, a: False),
                granted=[REACTION_GATE])
            self.assertIs(controller.run(), RunOutcome.ESCALATED)
            self.assertEqual(len(registry.get("evaluate_catalysis").calls), 1)
            self.assertIn("changed nothing",
                          controller.escalations[-1].reason)

    def test_without_a_repair_hook_the_controller_refuses_to_invent_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry(prepare_structures=[
                input_error("prepare_structures", "missing_structure_index")])
            controller = make_controller(Path(tmp), registry,
                                         granted=[REACTION_GATE])
            self.assertIs(controller.run(), RunOutcome.ESCALATED)
            self.assertIn("will not invent", controller.escalations[-1].reason)
            self.assertEqual(len(registry.get("prepare_structures").calls), 1)

    def test_a_missing_tool_is_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry(mine_sequences=[
                unavailable("mine_sequences")])
            controller = make_controller(
                Path(tmp), registry, max_retries=5,
                hooks=ControllerHooks(repair=lambda c, a: True,
                                      widen=lambda c, a: True),
                granted=[REACTION_GATE])
            self.assertIs(controller.run(), RunOutcome.ESCALATED)
            self.assertEqual(len(registry.get("mine_sequences").calls), 1)
            self.assertIs(controller.escalations[-1].kind,
                          FailureKind.TOOL_UNAVAILABLE)
            self.assertIn("not installed", controller.escalations[-1].reason)

    def test_an_unhandled_exception_is_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            def boom(ctx, **kwargs):
                raise KeyError("chain")

            registry = scripted_registry(annotate_family=[boom])
            controller = make_controller(
                Path(tmp), registry,
                hooks=ControllerHooks(repair=lambda c, a: True),
                granted=[REACTION_GATE])
            self.assertIs(controller.run(), RunOutcome.ESCALATED)
            self.assertEqual(len(registry.get("annotate_family").calls), 1)
            self.assertIs(controller.escalations[-1].kind,
                          FailureKind.INTERNAL_ERROR)

    def test_a_human_question_waits_instead_of_retrying(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry(mine_sequences=[
                needs_human("mine_sequences", REACTION_GATE)])
            controller = make_controller(Path(tmp), registry,
                                         granted=[REACTION_GATE])
            self.assertIs(controller.run(), RunOutcome.AWAITING_HUMAN)
            self.assertEqual(len(registry.get("mine_sequences").calls), 1)
            self.assertEqual(controller.escalations, [])

    def test_the_cost_ceiling_stops_further_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry(evaluate_catalysis=[
                costly("evaluate_catalysis", gpu_s=500.0)])
            controller = make_controller(
                Path(tmp), registry, cost_ceiling={"gpu_s": 100.0},
                hooks=ControllerHooks(repair=lambda c, a: True),
                granted=[REACTION_GATE])
            self.assertIs(controller.run(), RunOutcome.ESCALATED)
            self.assertEqual(len(registry.get("evaluate_catalysis").calls), 1,
                             "a repairable error is still not retried once the "
                             "recorded spend is over the ceiling")
            self.assertIn("cost ceiling", controller.escalations[-1].reason)
            self.assertEqual(controller.budget_breaches()[0].split(":")[0],
                             "gpu_s")

    def test_spend_under_the_ceiling_does_not_stop_the_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry(evaluate_catalysis=[
                costly("evaluate_catalysis", gpu_s=10.0),
                ok("evaluate_catalysis")])
            controller = make_controller(
                Path(tmp), registry, cost_ceiling={"gpu_s": 100.0},
                hooks=ControllerHooks(repair=lambda c, a: True),
                granted=[REACTION_GATE])
            controller.run()
            self.assertEqual(len(registry.get("evaluate_catalysis").calls), 2)
            self.assertEqual(controller.budget_breaches(), [])

    def test_the_controller_records_only_costs_a_step_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry(evaluate_catalysis=[
                costly("evaluate_catalysis", gpu_s=10.0),
                ok("evaluate_catalysis")])
            controller = make_controller(
                Path(tmp), registry,
                hooks=ControllerHooks(repair=lambda c, a: True),
                granted=[REACTION_GATE])
            controller.run()
            self.assertEqual(set(controller.manifest.cost_total), {"gpu_s"},
                             "the controller estimates nothing; every cost key "
                             "comes from a step's own provenance")

    def test_a_control_flow_loop_surfaces_as_an_escalation(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = make_controller(Path(tmp), granted=[REACTION_GATE])
            controller.max_transitions = 2
            self.assertIs(controller.run(), RunOutcome.ESCALATED)
            self.assertIn("control-flow defect",
                          controller.escalations[-1].reason)


# ---------------------------------------------------------------------------
# next actions
# ---------------------------------------------------------------------------

class NextActionRoutingTests(unittest.TestCase):
    """``next_actions`` are consumed, and none of them are executed."""

    def test_a_human_action_goes_to_the_queue_and_is_not_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ok("retrieve_evidence")
            result.add_next("confirm_cofactor_state",
                            "only the operator may fix the cofactor",
                            {"candidate": "C1"}, requires_human=True)
            registry = scripted_registry(retrieve_evidence=[result])
            controller = make_controller(Path(tmp), registry,
                                         granted=[REACTION_GATE])
            controller.run()
            self.assertEqual(len(controller.human_actions), 1)
            queued = controller.queue.for_gate("confirm_cofactor_state")
            self.assertEqual(len(queued), 1)
            self.assertEqual(queued[0].kind.value, "operator_task")

    def test_a_human_action_naming_a_gate_queues_that_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ok("select_batch")
            result.add_next("confirm_functional_criteria", "fix the endpoint",
                            {"gate": CRITERIA_GATE}, requires_human=True)
            registry = scripted_registry(select_batch=[result])
            controller = make_controller(
                Path(tmp), registry, granted=[REACTION_GATE, BATCH_GATE])
            controller.run()
            self.assertTrue(controller.queue.pending(CRITERIA_GATE))

    def test_a_human_action_naming_an_interface_is_still_not_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ok("mine_sequences")
            result.add_next("mine_sequences", "re-run with more seeds",
                            {"seeds": 10}, requires_human=True)
            registry = scripted_registry(mine_sequences=[result])
            controller = make_controller(Path(tmp), registry,
                                         granted=[REACTION_GATE])
            controller.run()
            self.assertEqual(len(registry.get("mine_sequences").calls), 1)
            self.assertEqual(controller.human_actions[0]["action"],
                             "mine_sequences")

    def test_a_machine_action_naming_an_interface_is_recorded_as_a_suggestion(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ok("prepare_structures")
            result.add_next("model_complexes", "structures are ready", {})
            registry = scripted_registry(prepare_structures=[result])
            controller = make_controller(Path(tmp), registry,
                                         granted=[REACTION_GATE])
            controller.run()
            self.assertEqual([a["action"] for a in controller.suggested_actions],
                             ["model_complexes"])

    def test_an_unknown_action_is_recorded_rather_than_guessed_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = ok("annotate_family")
            result.add_next("screen_geometry", "a step that no longer exists",
                            {})
            registry = scripted_registry(annotate_family=[result])
            controller = make_controller(Path(tmp), registry,
                                         granted=[REACTION_GATE])
            controller.run()
            self.assertEqual(len(controller.unroutable_actions), 1)
            self.assertIn("recorded for the operator",
                          controller.unroutable_actions[0]["note"])


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------

class ResumeTests(unittest.TestCase):
    """A resumed run redoes nothing whose inputs it can prove unchanged."""

    def _first_pass(self, tmp: Path, arguments=None):
        ctx = make_ctx(tmp)
        registry = scripted_registry()
        controller = make_controller(tmp, registry, ctx=ctx,
                                     arguments=arguments,
                                     granted=[REACTION_GATE])
        controller.run()
        return ctx.manifest, registry, controller

    def test_completed_steps_are_skipped_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest, first, _ = self._first_pass(path)
            self.assertEqual(len(first.get("evaluate_catalysis").calls), 1)

            second = scripted_registry()
            ctx = make_ctx(path, manifest=manifest)
            resumed = make_controller(path, second, ctx=ctx,
                                      granted=[REACTION_GATE, BATCH_GATE])
            resumed.run()
            self.assertEqual(second.get("evaluate_catalysis").calls, [],
                             "the inputs are unchanged, so the step is skipped")
            self.assertIn("evaluate_catalysis:evaluate_catalysis",
                          resumed.skipped_steps)
            self.assertEqual(len(second.get("select_batch").calls), 1,
                             "the newly unblocked step still runs")

    def test_a_changed_input_hash_re_runs_the_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest, _, _ = self._first_pass(
                path, arguments={"mine_sequences": {"seeds": ["A"]}})

            second = scripted_registry()
            ctx = make_ctx(path, manifest=manifest)
            resumed = make_controller(
                path, second, ctx=ctx, granted=[REACTION_GATE],
                arguments={"mine_sequences": {"seeds": ["A", "B"]}})
            resumed.run()
            self.assertEqual(len(second.get("mine_sequences").calls), 1,
                             "a different seed set is a different step")
            self.assertNotIn("mine_sequences:mine_sequences",
                             resumed.skipped_steps)

    def test_an_unhashable_argument_forces_a_re_run(self):
        class Opaque:
            pass

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            args = {"annotate_family": {"adapter": Opaque()}}
            manifest, _, first_controller = self._first_pass(path, args)
            attempt = [a for a in first_controller.attempts
                       if a.interface == "annotate_family"][0]
            self.assertFalse(attempt.digest_stable)

            second = scripted_registry()
            ctx = make_ctx(path, manifest=manifest)
            resumed = make_controller(path, second, ctx=ctx,
                                      granted=[REACTION_GATE],
                                      arguments={"annotate_family":
                                                 {"adapter": Opaque()}})
            resumed.run()
            self.assertEqual(len(second.get("annotate_family").calls), 1,
                             "an input whose identity cannot be hashed cannot "
                             "be proved unchanged, so the step re-runs")

    def test_a_failed_step_is_not_skipped_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            ctx = make_ctx(path)
            first = scripted_registry(model_complexes=[
                unavailable("model_complexes")])
            controller = make_controller(path, first, ctx=ctx,
                                         granted=[REACTION_GATE])
            controller.run()

            second = scripted_registry()
            resumed = make_controller(path, second,
                                      ctx=make_ctx(path, manifest=ctx.manifest),
                                      granted=[REACTION_GATE])
            resumed.run()
            self.assertEqual(len(second.get("model_complexes").calls), 1)

    def test_the_report_names_what_was_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest, _, _ = self._first_pass(path)
            second = scripted_registry()
            resumed = make_controller(path, second,
                                      ctx=make_ctx(path, manifest=manifest),
                                      granted=[REACTION_GATE, BATCH_GATE])
            resumed.run()
            report = resumed.report()
            self.assertTrue(report["skipped_steps"])
            written = resumed.write_report()
            self.assertTrue(written.exists())


class ManifestRecordingTests(unittest.TestCase):
    """Every executed step is in the manifest before any routing happens."""

    def test_each_attempt_appends_a_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = scripted_registry(evaluate_catalysis=[
                input_error("evaluate_catalysis"), ok("evaluate_catalysis")])
            controller = make_controller(
                Path(tmp), registry,
                hooks=ControllerHooks(repair=lambda c, a: True),
                granted=[REACTION_GATE])
            controller.run()
            records = [s for s in controller.manifest.steps
                       if s.interface == "evaluate_catalysis"]
            self.assertEqual(len(records), 2)
            self.assertEqual([r.status for r in records],
                             ["failed", "success"])
            self.assertEqual(records[1].provenance["parameters"]
                             ["controller_attempt"], 2)
            self.assertIn("controller_step_inputs",
                          records[0].provenance["inputs_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
