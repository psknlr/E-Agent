"""Tests for :mod:`eagent.tools.select_batch`.

The cases are the ways a batch plan stops being honest between the ranking and
the purchase order:

* the synthesis gate being cleared implicitly;
* "96 constructs" quietly meaning 576 wells;
* control genes discovered after the construct cap was already spent;
* a background control written up as evidence that the substrate is turned
  over;
* a short pool padded back to the round number so the plan looks complete;
* a combination variant admitted without the single mutants that make it
  attributable;
* a results template with a ``hit`` column, which is where a pre-registered
  endpoint goes to die.

Everything is in memory except the three artifacts, which are written to a
temporary directory and read back.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import yaml

from eagent.context import ExecutionPolicy, RunContext
from eagent.envelope import Severity, Status
from eagent.provenance import RunManifest
from eagent.schemas import (
    AssayTemplate,
    BatchRole,
    Budget,
    Candidate,
    CatalyticMapping,
    CofactorSpec,
    CofactorState,
    Conditions,
    ConfidenceLevel,
    FamilyAnnotation,
    Mutation,
    MutationProposal,
    ProductSpec,
    ReactionSpec,
    ScoreDimension,
    SequenceRecord,
    Stereochemistry,
    SubstrateSpec,
    TaskSpec,
    TemplateProvenance,
    TemplateSourceType,
)
from eagent.tools.select_batch import (
    ASSAY_RESULT_COLUMNS,
    ControlClaim,
    ControlSpec,
    MeasurementFootprint,
    SelectBatch,
    default_control_plan,
    measurement_footprint,
    select_variant_groups,
    validate_control_claims,
    well_label,
)

BASE = ("MKAAVLYEFGKPLEIKEVEVAPPKAHEVRIKIAYTGVCHTDAYTLSGADPEGLFPVILGHEGAGIVESVGEGVT"
        "NVKPGDHVIPLYTPECGECKFCKSGKTNLCQKIRATQGKGLMPDGTSRFTCKGKEILHYMGCSTFSEYTVVAD")


def mutate(sequence: str, positions: dict[int, str]) -> str:
    chars = list(sequence)
    for index, letter in positions.items():
        chars[index] = letter
    return "".join(chars)


def candidate(cid: str, *, family: str = "SDR", cluster: str | None = None,
              gates_pass: bool | None = True, evidence_value: float = 1.0,
              uncertainty: ConfidenceLevel = ConfidenceLevel.STRONG,
              sequence: str | None = None) -> Candidate:
    """A candidate with just enough scorecard for composition to be legal."""
    cand = Candidate(
        candidate_id=cid,
        sequence_record=SequenceRecord(
            candidate_id=cid,
            sequence=sequence or mutate(BASE, {7: "ACDEFGHIKLMNPQRSTVWY"[
                sum(ord(c) for c in cid) % 20]})),
        family=FamilyAnnotation(family_name=family, sequence_cluster_id=cluster),
        catalytic_mapping=CatalyticMapping(
            catalytic_template_id="cat:sdr:kred",
            role_to_residue={"catalytic_Tyr": f"Y{155 + len(cid)}"}),
    )
    cand.set_dimension(ScoreDimension(
        name="catalytic_machinery_mappable", is_gate=True,
        gate_passed=gates_pass, direction="categorical",
        level=ConfidenceLevel.STRONG if gates_pass else ConfidenceLevel.WEAK))
    cand.set_dimension(ScoreDimension(
        name="functional_literature_evidence", level=ConfidenceLevel.MODERATE,
        value=evidence_value))
    cand.set_dimension(ScoreDimension(
        name="model_uncertainty", level=uncertainty, direction="categorical"))
    return cand


def make_task(*, authorized: bool = True, constructs: int = 6,
              include_controls: bool = False,
              cofactors: int = 2) -> TaskSpec:
    options = [
        CofactorSpec(name="NADPH", state=CofactorState.REDUCED),
        CofactorSpec(name="NADH", state=CofactorState.REDUCED),
    ][:cofactors]
    task = TaskSpec(
        task_id="T1",
        reaction=ReactionSpec(
            substrate=SubstrateSpec(name="acetophenone",
                                    isomeric_smiles="CC(=O)c1ccccc1"),
            product=ProductSpec(name="(R)-1-phenylethanol",
                                isomeric_smiles="C[C@@H](O)c1ccccc1",
                                target_stereochemistry=Stereochemistry.R,
                                creates_new_stereocenter=True)),
        conditions=Conditions(pH=7.0, temperature_C=30.0,
                              expression_host="E. coli BL21(DE3)",
                              cofactor_options=options),
        budget=Budget(new_constructs_round_1=constructs,
                      constructs_include_controls=include_controls),
    )
    task.approval.synthesis_authorized = authorized
    return task


def make_ctx(tmp: Path, task: TaskSpec | None = None) -> RunContext:
    task = task or make_task()
    return RunContext(task=task, workdir=tmp,
                      manifest=RunManifest(run_id="R1", task_id=task.task_id),
                      policy=ExecutionPolicy(allow_network=False))


def make_assay_template(**overrides) -> AssayTemplate:
    kwargs = dict(
        template_id="assay:kred:tier2", tier=2, method="chiral GC-MS",
        confirms_product_identity=True, chiral_capable=True,
        requires_authentic_standard=True, replicates=3,
        positive_criteria={"min_conversion_pct": 5.0,
                           "min_ee_target_pct": 80.0},
        limit_of_detection=0.5, limit_unit="uM",
        provenance=TemplateProvenance(
            source_type=TemplateSourceType.CURATED_DATABASE,
            identifiers=["SOP-KRED-01"]))
    kwargs.update(overrides)
    return AssayTemplate(**kwargs)


def proposal(pid: str, mutations: list[tuple[str, int, int, str]], *,
             controls: list[str] | None = None,
             priority: int = 1) -> MutationProposal:
    return MutationProposal(
        proposal_id=pid, parent_candidate_id="C1",
        parent_sequence_sha256="sha256:" + "0" * 64,
        mutations=[Mutation(wild_type=w, position_author=a, position_index=i,
                            mutant=m) for w, a, i, m in mutations],
        numbering_reference="1XYZ chain A author numbering",
        intended_improvement=["widen the pocket for the aryl group"],
        possible_cost=["packing and therefore soluble expression"],
        decomposition_controls=list(controls or []),
        experimental_priority=priority,
    )


class WellLabelTests(unittest.TestCase):
    def test_row_major_96(self):
        self.assertEqual(well_label(0), "A1")
        self.assertEqual(well_label(11), "A12")
        self.assertEqual(well_label(12), "B1")
        self.assertEqual(well_label(95), "H12")

    def test_wraps_onto_the_next_plate(self):
        self.assertEqual(well_label(96), "A1")

    def test_384_uses_24_columns(self):
        self.assertEqual(well_label(23, 384), "A24")
        self.assertEqual(well_label(24, 384), "B1")


class ControlClaimTests(unittest.TestCase):
    def test_default_plan_never_claims_target_turnover(self):
        specs = default_control_plan(positive_control_enzyme_available=True)
        claims = {s.claim for s in specs}
        self.assertIn(ControlClaim.ASSAY_SYSTEM_WORKS, claims)
        self.assertNotIn(ControlClaim.TARGET_SUBSTRATE_TURNED_OVER, claims)
        self.assertEqual(validate_control_claims(specs), [])

    def test_background_control_may_not_claim_turnover(self):
        specs = [ControlSpec(name="no_enzyme", kind="no_enzyme",
                             claim=ControlClaim.TARGET_SUBSTRATE_TURNED_OVER)]
        problems = validate_control_claims(specs)
        self.assertTrue(any("cannot demonstrate turnover" in p for p in problems))

    def test_turnover_claim_needs_product_identification(self):
        specs = [
            ControlSpec(name="positive", kind="positive_enzyme",
                        claim=ControlClaim.TARGET_SUBSTRATE_TURNED_OVER,
                        substrate_is_target=True,
                        product_identity_confirmed=False),
        ]
        problems = validate_control_claims(specs)
        self.assertTrue(any("identifies the product" in p for p in problems))

    def test_missing_system_control_is_a_problem(self):
        specs = default_control_plan(positive_control_enzyme_available=False)
        problems = validate_control_claims(specs)
        self.assertTrue(any("assay system works" in p for p in problems))

    def test_demonstrates_text_separates_the_two_claims(self):
        item = ControlSpec(name="positive", kind="positive_enzyme",
                           claim=ControlClaim.ASSAY_SYSTEM_WORKS).to_control_item()
        self.assertIn("Says NOTHING about the target substrate",
                      item.demonstrates)


class FootprintTests(unittest.TestCase):
    def test_genes_are_not_wells(self):
        footprint = MeasurementFootprint(
            n_candidate_genes=96, n_variant_genes=0, n_control_genes=1,
            n_controls=4, cofactor_conditions=2, replicates=3,
            wells_per_plate=96)
        self.assertEqual(footprint.n_genes, 97)
        self.assertEqual(footprint.candidate_wells, 576)
        self.assertEqual(footprint.control_wells, 24)
        self.assertEqual(footprint.total_wells, 600)
        self.assertEqual(footprint.plates, 7)
        self.assertIn("Genes are not wells", footprint.describe())


class VariantGroupTests(unittest.TestCase):
    def setUp(self):
        self.single_a = proposal("P:A", [("L", 112, 11, "A")], priority=1)
        self.single_b = proposal("P:B", [("Y", 134, 33, "F")], priority=1)
        self.combo = proposal(
            "P:AB", [("L", 112, 11, "A"), ("Y", 134, 33, "F")],
            controls=["P:A", "P:B"], priority=1)

    def test_whole_group_fits(self):
        picked = select_variant_groups(
            [self.combo, self.single_a, self.single_b], 3)
        self.assertEqual(picked.n_selected, 3)
        self.assertEqual(picked.dropped, ())

    def test_combination_is_dropped_rather_than_split(self):
        picked = select_variant_groups(
            [self.combo, self.single_a, self.single_b], 2)
        ids = {p.proposal_id for p in picked.selected}
        self.assertEqual(ids, {"P:A", "P:B"})
        self.assertIn("P:AB", {p.proposal_id for p in picked.dropped})
        self.assertIn("dropped whole", picked.reason)

    def test_combination_with_a_missing_control_is_refused(self):
        orphan = proposal("P:CD", [("L", 112, 11, "A"), ("Y", 134, 33, "F")],
                          controls=["P:C", "P:D"])
        picked = select_variant_groups([orphan], 10)
        self.assertEqual(picked.n_selected, 0)
        self.assertIn("not among the supplied proposals", picked.reason)


class GateTests(unittest.TestCase):
    def test_unauthorized_synthesis_blocks_the_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp), make_task(authorized=False))
            out = SelectBatch().run(ctx, candidates=[candidate("C1")],
                                    assay_template=make_assay_template())
            self.assertIs(out.status, Status.FAILED)
            self.assertTrue(any(f.code == "approval_required"
                                for f in out.blockers))

    def test_submit_to_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = SelectBatch().run(
                ctx, candidates=[candidate("C1")],
                assay_template=make_assay_template(),
                submit_to="https://genes.example.com")
            self.assertIs(out.status, Status.FAILED)
            self.assertTrue(any(f.code == "external_submission_refused"
                                for f in out.blockers))

    def test_empty_pool_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = SelectBatch().run(make_ctx(Path(tmp)), candidates=[],
                                    assay_template=make_assay_template())
            self.assertIs(out.status, Status.FAILED)

    def test_ungated_candidate_pool_is_refused_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            bare = Candidate(
                candidate_id="C9",
                sequence_record=SequenceRecord(candidate_id="C9", sequence=BASE))
            out = SelectBatch().run(ctx, candidates=[bare],
                                    assay_template=make_assay_template())
            self.assertIs(out.status, Status.FAILED)
            self.assertTrue(any(f.code == "pool_not_composable"
                                for f in out.blockers))


class CompositionTests(unittest.TestCase):
    def _run(self, tmp: Path, **kwargs):
        pool = [candidate(f"C{i}", family="SDR" if i % 2 else "AKR",
                          cluster=f"cl{i % 3}", evidence_value=1.0 - i / 100)
                for i in range(12)]
        ctx = make_ctx(tmp, kwargs.pop("task", None) or make_task())
        return ctx, SelectBatch().run(
            ctx, candidates=kwargs.pop("candidates", pool),
            assay_template=make_assay_template(),
            positive_control_enzyme_available=True, **kwargs)

    def test_footprint_is_reported_in_wells_and_plates(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = self._run(Path(tmp))
            footprint = out.data["footprint"]
            plan = out.data["plan"]
            self.assertEqual(
                footprint["candidate_wells"],
                len(plan["members"]) * plan["cofactor_conditions"]
                * plan["replicates"])
            self.assertEqual(
                footprint["total_wells"],
                footprint["candidate_wells"] + footprint["control_wells"])
            self.assertGreaterEqual(footprint["plates"], 1)
            self.assertIn("Genes are not wells", out.message)

    def test_controls_are_reserved_from_a_hard_construct_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = make_task(constructs=6, include_controls=True)
            _, out = self._run(Path(tmp), task=task)
            self.assertEqual(out.provenance.parameters["reserved_control_slots"], 1)
            self.assertEqual(out.provenance.parameters["candidate_slots"], 5)
            self.assertLessEqual(out.data["plan"]["requested_slots"], 5)
            self.assertEqual(out.data["footprint"]["n_genes"],
                             out.data["footprint"]["n_candidate_genes"]
                             + out.data["footprint"]["n_variant_genes"] + 1)

    def test_separate_control_budget_reserves_nothing_but_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = self._run(Path(tmp), task=make_task(include_controls=False))
            self.assertEqual(out.provenance.parameters["reserved_control_slots"], 0)
            self.assertTrue(any(
                f.code == "control_slot_reservation"
                and "separate synthesis budget line" in f.message
                for f in out.qc_flags))

    def test_short_batch_is_reported_not_padded(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = [candidate("C1"), candidate("C2")]
            _, out = self._run(Path(tmp), candidates=pool)
            plan = out.data["plan"]
            self.assertEqual(len(plan["members"]), 2)
            self.assertTrue(plan["shortfall_reason"])
            self.assertIn("reported short", plan["shortfall_reason"])
            short = [f for f in out.qc_flags if f.code == "batch_short"]
            self.assertTrue(short)
            # A short batch is a finding about the pool, not a broken step:
            # warned and usable, but never reported as a complete round.
            self.assertIs(short[0].severity, Severity.WARN)
            self.assertIs(out.status, Status.PARTIAL)
            self.assertTrue(out.status.usable)

    def test_failing_gate_never_enters_the_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            pool = [candidate("C1"), candidate("C2", gates_pass=False)]
            _, out = self._run(Path(tmp), candidates=pool)
            ids = {m["candidate_id"] for m in out.data["plan"]["members"]}
            self.assertNotIn("C2", ids)

    def test_missing_pre_registered_criterion_is_a_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = SelectBatch().run(
                ctx, candidates=[candidate("C1")], assay_template=None,
                positive_control_enzyme_available=True)
            self.assertTrue(any(f.code == "no_pre_registered_criterion"
                                for f in out.blockers))

    def test_no_cofactor_options_is_flagged_not_assumed_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = self._run(Path(tmp), task=make_task(cofactors=0))
            self.assertTrue(any(f.code == "cofactor_conditions_assumed"
                                for f in out.qc_flags))


class FootprintFromPlanTests(unittest.TestCase):
    def test_footprint_counts_control_genes_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = SelectBatch().run(
                ctx, candidates=[candidate(f"C{i}") for i in range(4)],
                assay_template=make_assay_template(),
                positive_control_enzyme_available=True)
            from eagent.schemas import BatchPlan
            plan = BatchPlan(**out.data["plan"])
            footprint = measurement_footprint(plan)
            self.assertEqual(footprint.n_control_genes, 1)
            self.assertEqual(footprint.n_candidate_genes, plan.n_candidates)
            self.assertEqual(footprint.total_wells, plan.total_units)


class VariantBatchTests(unittest.TestCase):
    def test_variants_consume_gene_slots_alongside_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp), make_task(constructs=4))
            singles = [proposal("P:A", [("L", 112, 11, "A")]),
                       proposal("P:B", [("Y", 134, 33, "F")])]
            combo = proposal("P:AB",
                             [("L", 112, 11, "A"), ("Y", 134, 33, "F")],
                             controls=["P:A", "P:B"])
            out = SelectBatch().run(
                ctx, candidates=[candidate(f"C{i}") for i in range(6)],
                variant_proposals=[combo] + singles,
                assay_template=make_assay_template(),
                positive_control_enzyme_available=True)
            plan = out.data["plan"]
            self.assertEqual(out.data["n_variant_genes"], 3)
            self.assertEqual(len(plan["members"]), 4)
            roles = {m["candidate_id"]: m["role"] for m in plan["members"]}
            self.assertEqual(roles.get("P:AB"), BatchRole.VARIANT.value)
            self.assertEqual(sum(1 for r in roles.values()
                                 if r == BatchRole.VARIANT.value), 3)


class ArtifactTests(unittest.TestCase):
    def _compose(self, tmp: Path):
        ctx = make_ctx(tmp)
        out = SelectBatch().run(
            ctx, candidates=[candidate(f"C{i}") for i in range(12)],
            assay_template=make_assay_template(),
            positive_control_enzyme_available=True)
        return ctx, out

    def test_batch_file_is_named_from_the_real_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = self._compose(Path(tmp))
            path = Path(out.artifact("selected_batch").path)
            n_members = len([m for m in out.data["plan"]["members"]
                             if m["role"] != BatchRole.CONTROL.value])
            n_controls = len(out.data["plan"]["controls"])
            self.assertEqual(path.name, f"selected_batch_{n_members}.csv")
            self.assertNotEqual(n_members, 96)   # the fixture pool is smaller
            with open(path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), n_members + n_controls)

    def test_plan_yaml_carries_the_pre_registered_criterion(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = self._compose(Path(tmp))
            document = yaml.safe_load(
                Path(out.artifact("experiment_plan").path).read_text())
            endpoint = document["pre_registered_endpoint"]
            self.assertEqual(endpoint["positive_criteria"],
                             {"min_conversion_pct": 5.0,
                              "min_ee_target_pct": 80.0})
            self.assertEqual(endpoint["positive_criteria_sha256"],
                             out.data["positive_criteria_sha256"])
            self.assertIn("and no other", endpoint["binding_statement"])
            self.assertIn("different claims",
                          document["control_claim_separation"])

    def test_results_template_matches_the_footprint_and_has_no_hit_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = self._compose(Path(tmp))
            path = Path(out.artifact("assay_results_template").path)
            with open(path, encoding="utf-8") as fh:
                reader = csv.reader(fh)
                header = next(reader)
                rows = list(reader)
            self.assertEqual(tuple(header), ASSAY_RESULT_COLUMNS)
            self.assertNotIn("hit", header)
            self.assertNotIn("outcome", header)
            self.assertEqual(len(rows), out.data["footprint"]["total_wells"])

    def test_results_template_pre_addresses_the_wells(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = self._compose(Path(tmp))
            with open(out.artifact("assay_results_template").path,
                      encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(rows[0]["well"], "A1")
            self.assertTrue(all(r["candidate_id"] for r in rows))
            self.assertTrue(all(r["cofactor"] for r in rows))
            self.assertEqual({r["kind"] for r in rows}, {"candidate", "control"})

    def test_provenance_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, out = self._compose(Path(tmp))
            self.assertEqual(out.provenance.tool, "select_batch")
            self.assertIsNotNone(out.provenance.random_seed)
            self.assertIn("positive_criteria", out.provenance.inputs_sha256)
            self.assertFalse(out.provenance.parameters["allow_network"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
