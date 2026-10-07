"""Tests for the registry's reported-licence entries and the two refusing branches.

What is defended: a restriction that was only reported is still honoured; a
permission that was only reported is not claimed; every failed condition for a
branch is listed together; an approval is bound to the exact branch and input;
and a designed sequence cannot occupy a slot whose label says "evidence".
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from eagent.connectors.base import AccessPolicy, SubmissionAuthorization
from eagent.context import ExecutionPolicy, RunContext
from eagent.errors import LicenseError, ToolUnavailableError
from eagent.harness.approval import REACTION_GATE, ApprovalQueue
from eagent.provenance import RunManifest, sequence_hash
from eagent.schemas import (
    BatchRole, Budget, ConfidenceLevel, EvidenceStrength, ScoreDimension,
    TaskSpec,
)
from eagent.science.diversity import compose_batch
from eagent.tools.model_complexes import (
    ToolKind, ToolRegistry, ToolRegistryEntry, ToolRegistryError,
    check_license, check_modification,
)
from eagent.tools.open_branches import (
    BRANCH_CEILING, BranchKind, BranchRefusedError, BranchRequest, DesignJob,
    DiscoveryJob, RawDesign, UnavailableDeNovoGenerator,
    UnavailableOpenFunctionPredictor, authorize_branch, branch_gate,
    design_candidate, is_high_evidence_eligible, proposals_from_designs,
)

from test_scorecard import candidate, dim

REPO = Path(__file__).resolve().parents[1]
SEQ = "MKAIVTGGAQGIGRAIAERLAADGYNVAVLDRNEEGLKAVAEEIKAAG"


def shipped_registry() -> ToolRegistry:
    raw = yaml.safe_load((REPO / "configs" / "tool_registry.yaml").read_text())
    return ToolRegistry.from_config(raw)


class TestReportedLicences(unittest.TestCase):

    def setUp(self) -> None:
        self.registry = shipped_registry()

    def test_every_new_tool_has_all_four_facets(self) -> None:
        for tool in ("venusmine", "ap_novo"):
            for facet in ("code", "model_weights", "input_database", "output"):
                self.assertIn(f"{tool}.{facet}", self.registry)

    def test_venusmine_is_recorded_as_a_restriction_and_says_it_was_not_read(self) -> None:
        entry = self.registry.get("venusmine.code")
        self.assertIn("CC BY-NC-ND 4.0", entry.license)
        self.assertIn("not read", entry.license)
        self.assertIn("NOT opened", entry.license_source)
        self.assertIs(entry.permits_commercial_use, False)
        self.assertIs(entry.permits_derivative_works, False)
        self.assertTrue(entry.needs_legal_review)

    def test_the_reported_apache_code_licence_is_not_turned_into_a_permission(self) -> None:
        entry = self.registry.get("ap_novo.code")
        self.assertIn("Apache-2.0", entry.license)
        self.assertIsNone(entry.permits_commercial_use)
        self.assertIsNone(entry.permits_derivative_works)

    def test_ap_novo_weights_and_outputs_are_restricted_whatever_the_code_says(self) -> None:
        self.assertIs(self.registry.get("ap_novo.model_weights")
                      .permits_commercial_use, False)
        self.assertIs(self.registry.get("ap_novo.output")
                      .permits_commercial_use, False)

    def test_a_commercial_run_is_blocked_for_both_tools(self) -> None:
        for keys in (["venusmine.code"],
                     ["ap_novo.code", "ap_novo.model_weights", "ap_novo.output"]):
            with self.subTest(keys=keys):
                with self.assertRaises(LicenseError):
                    check_license(self.registry, keys, True, "test",
                                  uses_model_weights="ap_novo.model_weights" in keys)

    def test_a_non_commercial_run_may_use_the_restricted_weights(self) -> None:
        facts = check_license(
            self.registry,
            ["ap_novo.code", "ap_novo.model_weights", "ap_novo.output"], False,
            "test", uses_model_weights=True)
        self.assertEqual(len(facts["entries"]), 3)

    def test_modifying_and_redistributing_venusmine_is_refused(self) -> None:
        with self.assertRaises(LicenseError) as caught:
            check_modification(self.registry, ["venusmine.code"], "vendoring")
        self.assertIn("derivative-works", str(caught.exception))

    def test_unread_terms_are_not_a_derivative_permission_either(self) -> None:
        with self.assertRaises(LicenseError):
            check_modification(self.registry, ["ap_novo.code"], "vendoring")

    def test_a_tool_whose_terms_were_read_and_allow_it_may_be_modified(self) -> None:
        registry = ToolRegistry([ToolRegistryEntry(
            key="x.code", kind=ToolKind.CODE, display_name="x",
            license="Apache-2.0", license_source="LICENSE read 2026-10-07",
            permits_commercial_use=True, permits_derivative_works=True)])
        check_modification(registry, ["x.code"], "vendoring")

    def test_a_derivative_position_with_no_licence_rests_on_nothing(self) -> None:
        with self.assertRaises(ToolRegistryError):
            ToolRegistryEntry(key="x", kind=ToolKind.CODE, display_name="x",
                              permits_derivative_works=False)

    def test_the_check_license_facts_carry_the_derivative_position(self) -> None:
        facts = check_license(self.registry, ["venusmine.code"], False, "test")
        self.assertIs(facts["entries"][0]["permits_derivative_works"], False)


class _Case(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.registry = shipped_registry()

    def ctx(self, **policy) -> RunContext:
        return RunContext(task=TaskSpec(task_id="T1"), workdir=self.tmp,
                          manifest=RunManifest(run_id="R1", task_id="T1"),
                          policy=ExecutionPolicy(**policy))

    def queue(self, ctx: RunContext, *, reaction_confirmed: bool = True
              ) -> ApprovalQueue:
        queue = ApprovalQueue(self.tmp / "approvals.json", ctx.manifest)
        if reaction_confirmed:
            queue.request(REACTION_GATE, payload={"spec": "x"})
            queue.grant(REACTION_GATE, actor="r.chemist", reason="confirmed")
        return queue

    def request(self, **kw) -> BranchRequest:
        base = dict(
            kind=BranchKind.DE_NOVO_DESIGN, generator="ap_novo",
            description="designs for the confirmed ketone reduction",
            registry_keys=("ap_novo.code", "ap_novo.model_weights",
                           "ap_novo.output"),
            inputs_sha256="ab" * 32)
        base.update(kw)
        return BranchRequest(**base)


class TestAuthorization(_Case):

    def test_a_clean_request_is_queued_for_a_person_and_refused_until_granted(self) -> None:
        ctx = self.ctx()
        queue = self.queue(ctx)
        with self.assertRaises(BranchRefusedError) as caught:
            authorize_branch(self.request(), ctx=ctx, registry=self.registry,
                             queue=queue)
        self.assertEqual(len(caught.exception.reasons), 1)
        self.assertIn("no named person", caught.exception.reasons[0])
        pending = queue.pending(branch_gate(BranchKind.DE_NOVO_DESIGN))
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].payload["generator"], "ap_novo")

    def test_a_grant_for_exactly_this_payload_authorises(self) -> None:
        ctx = self.ctx()
        queue = self.queue(ctx)
        req = self.request()
        with self.assertRaises(BranchRefusedError):
            authorize_branch(req, ctx=ctx, registry=self.registry, queue=queue)
        queue.grant(branch_gate(BranchKind.DE_NOVO_DESIGN), actor="p.investigator",
                    reason="approved for this input")
        auth = authorize_branch(req, ctx=ctx, registry=self.registry, queue=queue)
        self.assertEqual(auth.approved_by, "p.investigator")
        self.assertIs(auth.evidence_ceiling, EvidenceStrength.COMPUTATIONAL_CONSTRUCT)
        self.assertTrue(auth.forbidden_claims)
        self.assertEqual(len(auth.licence_facts["entries"]), 3)

    def test_a_grant_does_not_carry_to_a_different_input(self) -> None:
        ctx = self.ctx()
        queue = self.queue(ctx)
        with self.assertRaises(BranchRefusedError):
            authorize_branch(self.request(), ctx=ctx, registry=self.registry,
                             queue=queue)
        queue.grant(branch_gate(BranchKind.DE_NOVO_DESIGN), actor="p.investigator")
        with self.assertRaises(BranchRefusedError) as caught:
            authorize_branch(self.request(inputs_sha256="cd" * 32), ctx=ctx,
                             registry=self.registry, queue=queue)
        self.assertIn("no named person", " ".join(caught.exception.reasons))

    def test_every_failed_condition_is_listed_together(self) -> None:
        ctx = self.ctx(allow_commercial_use=True, allow_network=False,
                       allow_gpu_models=False)
        queue = self.queue(ctx, reaction_confirmed=False)
        req = self.request(runs_remotely=True, modifies_third_party_code=True,
                           registry_keys=("venusmine.code", "ap_novo.model_weights"))
        with self.assertRaises(BranchRefusedError) as caught:
            authorize_branch(req, ctx=ctx, registry=self.registry, queue=queue)
        text = " | ".join(caught.exception.reasons)
        for fragment in ("reaction spec has not been confirmed",
                         "does not record a positive commercial-use",
                         "derivative-works", "allow_network is False",
                         "allow_gpu_models is False"):
            self.assertIn(fragment, text)
        # nobody is asked to approve what the licences already forbid
        self.assertEqual(queue.pending(branch_gate(BranchKind.DE_NOVO_DESIGN)), [])

    def test_no_reaction_confirmation_means_no_branch(self) -> None:
        ctx = self.ctx()
        queue = self.queue(ctx, reaction_confirmed=False)
        with self.assertRaises(BranchRefusedError) as caught:
            authorize_branch(self.request(), ctx=ctx, registry=self.registry,
                             queue=queue)
        self.assertIn("reaction spec has not been confirmed",
                      caught.exception.reasons[0])

    def test_a_remote_generator_does_not_get_an_unauthorised_sequence(self) -> None:
        ctx = self.ctx(allow_network=True)
        queue = self.queue(ctx)
        req = self.request(
            kind=BranchKind.OPEN_FUNCTION_DISCOVERY, generator="remote-fn",
            registry_keys=("boltz.code", "boltz.model_weights"),
            runs_remotely=True, outbound_sequences=(SEQ,))
        with self.assertRaises(BranchRefusedError) as caught:
            authorize_branch(req, ctx=ctx, registry=self.registry, queue=queue)
        self.assertIn("disclosure", " | ".join(caught.exception.reasons))

    def test_a_named_authorisation_lets_the_sequence_through_and_is_recorded(self) -> None:
        ctx = self.ctx(allow_network=True)
        queue = self.queue(ctx)
        req = self.request(
            kind=BranchKind.OPEN_FUNCTION_DISCOVERY, generator="remote-fn",
            registry_keys=("boltz.code", "boltz.model_weights"),
            runs_remotely=True, outbound_sequences=(SEQ,))
        access = AccessPolicy(allow_network=True, authorizations=(
            SubmissionAuthorization(
                authorized_by="operator:p.investigator", scope="remote-fn",
                justification="released for this prediction",
                sequence_sha256=(sequence_hash(SEQ),),
                at="2026-10-07T00:00:00Z"),))
        with self.assertRaises(BranchRefusedError):          # still needs approval
            authorize_branch(req, ctx=ctx, registry=self.registry, queue=queue,
                             access_policy=access)
        queue.grant(branch_gate(BranchKind.OPEN_FUNCTION_DISCOVERY), actor="p.i")
        auth = authorize_branch(req, ctx=ctx, registry=self.registry, queue=queue,
                                access_policy=access)
        self.assertTrue(any("p.investigator" in n for n in auth.disclosure_notes))

    def test_an_unregistered_key_is_a_refusal_not_a_crash(self) -> None:
        ctx = self.ctx()
        with self.assertRaises(BranchRefusedError) as caught:
            authorize_branch(self.request(registry_keys=("no_such.code",)),
                             ctx=ctx, registry=self.registry,
                             queue=self.queue(ctx))
        self.assertIn("tool registry", " | ".join(caught.exception.reasons))

    def test_the_approval_is_an_operator_task_not_a_fourth_gate(self) -> None:
        self.assertEqual(branch_gate(BranchKind.DE_NOVO_DESIGN),
                         "open_branch:de_novo_design")
        ctx = self.ctx()
        queue = self.queue(ctx)
        with self.assertRaises(BranchRefusedError):
            authorize_branch(self.request(), ctx=ctx, registry=self.registry,
                             queue=queue)
        self.assertEqual(queue.pending(branch_gate(BranchKind.DE_NOVO_DESIGN))[0]
                         .kind.value, "operator_task")


class TestSeams(unittest.TestCase):

    def test_the_defaults_refuse_with_an_install_hint_and_invent_nothing(self) -> None:
        with self.assertRaises(ToolUnavailableError) as g:
            UnavailableDeNovoGenerator().generate(DesignJob("t", "ab", 3, 0))
        self.assertIn("never substituted", str(g.exception))
        with self.assertRaises(ToolUnavailableError):
            UnavailableOpenFunctionPredictor().predict(
                DiscoveryJob(("s1",), (SEQ,), 0))
        self.assertFalse(UnavailableDeNovoGenerator().is_available())
        self.assertFalse(UnavailableOpenFunctionPredictor().is_available())

    def test_the_ceilings_are_fixed_by_the_branch(self) -> None:
        self.assertIs(BRANCH_CEILING[BranchKind.DE_NOVO_DESIGN],
                      EvidenceStrength.COMPUTATIONAL_CONSTRUCT)
        self.assertIs(BRANCH_CEILING[BranchKind.OPEN_FUNCTION_DISCOVERY],
                      EvidenceStrength.ANNOTATION_ONLY)


class _Gen:
    name, version = "fake-designer", "0.1"
    registry_keys = ("ap_novo.code",)
    uses_model_weights, runs_remotely = True, False

    def is_available(self) -> bool:
        return True


class TestDesignedCandidates(unittest.TestCase):

    def proposals(self, confidence=0.99):
        return proposals_from_designs([
            RawDesign("d1", SEQ, {"plddt": confidence}),
            RawDesign("d2", SEQ[::-1], {"plddt": 0.2})], _Gen())

    def test_a_generators_confidence_is_kept_and_never_promoted(self) -> None:
        proposals = self.proposals(confidence=1.0)
        self.assertEqual(proposals[0].generator_confidence, {"plddt": 1.0})
        self.assertIs(proposals[0].evidence_ceiling,
                      EvidenceStrength.COMPUTATIONAL_CONSTRUCT)
        self.assertIn("that the design is active on any substrate",
                      proposals[0].forbidden_claims)

    def test_duplicate_ids_and_empty_sequences_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            proposals_from_designs([RawDesign("d", SEQ), RawDesign("d", SEQ)],
                                   _Gen())
        with self.assertRaises(ValueError):
            proposals_from_designs([RawDesign("d", "  ")], _Gen())

    def test_a_designed_candidate_says_it_was_designed(self) -> None:
        cand = design_candidate(self.proposals()[0], "SDR")
        record = cand.sequence_record
        self.assertEqual(record.source_database, "de_novo:fake-designer")
        self.assertEqual(record.search_method, "de_novo_design")
        self.assertIsNone(record.accession)
        self.assertIsNone(record.seed_accession)
        self.assertIs(record.annotation_confidence,
                      EvidenceStrength.COMPUTATIONAL_CONSTRUCT)

    def test_only_a_de_novo_proposal_becomes_a_designed_candidate(self) -> None:
        from dataclasses import replace
        wrong = replace(self.proposals()[0],
                        kind=BranchKind.OPEN_FUNCTION_DISCOVERY)
        with self.assertRaises(ValueError):
            design_candidate(wrong)

    def test_a_designed_candidate_is_not_high_evidence_eligible_and_a_mined_one_is(self) -> None:
        self.assertFalse(is_high_evidence_eligible(
            design_candidate(self.proposals()[0])))
        self.assertTrue(is_high_evidence_eligible(candidate("mined1")))

    def gated(self, cand):
        cand.set_dimension(ScoreDimension(
            name="catalytic_machinery_mappable", is_gate=True, gate_passed=True,
            direction="categorical", level=ConfidenceLevel.STRONG))
        return cand

    def test_compose_batch_keeps_a_design_out_of_the_high_evidence_role_only(self) -> None:
        strong = dim("functional_literature_evidence", ConfidenceLevel.STRONG)
        design = self.gated(design_candidate(self.proposals()[0]))
        design.set_dimension(strong)           # even a design that outranks everything
        mined = [candidate(f"m{i}", dims={"x": dim(
            "functional_literature_evidence", ConfidenceLevel.WEAK)})
            for i in range(5)]
        pool = [design, *mined]
        budget = Budget(new_constructs_round_1=3, detailed_complex_target=6)
        order = ["functional_literature_evidence"]
        plain = compose_batch(pool, budget, order_of_dimensions=order,
                              role_targets={BatchRole.HIGH_EVIDENCE: 3})
        filtered = compose_batch(pool, budget, order_of_dimensions=order,
                                 role_targets={BatchRole.HIGH_EVIDENCE: 3},
                                 high_evidence_eligible=is_high_evidence_eligible)
        self.assertEqual(plain.members[0].candidate_id, design.candidate_id)
        high = [m.candidate_id for m in filtered.members
                if m.role is BatchRole.HIGH_EVIDENCE]
        self.assertNotIn(design.candidate_id, high)
        self.assertEqual(len(high), 3)

    def test_a_design_may_still_take_a_diversity_slot(self) -> None:
        design = self.gated(design_candidate(self.proposals()[0]))
        mined = [candidate(f"m{i}") for i in range(2)]
        plan = compose_batch(
            [design, *mined], Budget(new_constructs_round_1=3,
                                     detailed_complex_target=3),
            role_targets={BatchRole.HIGH_EVIDENCE: 1, BatchRole.DIVERSITY: 2},
            high_evidence_eligible=is_high_evidence_eligible)
        roles = {m.candidate_id: m.role for m in plan.members}
        self.assertIn(design.candidate_id, roles)
        self.assertIsNot(roles[design.candidate_id], BatchRole.HIGH_EVIDENCE)


if __name__ == "__main__":
    unittest.main()
