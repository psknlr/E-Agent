"""Tests for :mod:`eagent.tools.ingest_results`.

The cases are the ways a returned plate loses its information on the way into
the database:

* four distinct outcomes collapsing into one binary label;
* a negative recorded without the limit it is negative at;
* a positive called off a cofactor absorbance trace;
* an absolute ee hiding a selective failure;
* an expression failure counted as a catalytic negative;
* negatives dropped from the learning update;
* the positivity criterion drifting after the data arrive;
* an empty round being written up as "this substrate has no catalyst".

Everything is built in memory; two cases write and read back a CSV.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from eagent.context import ExecutionPolicy, RunContext
from eagent.envelope import Severity, Status
from eagent.errors import TemplateError
from eagent.provenance import RunManifest
from eagent.schemas import (
    AssayTemplate,
    CofactorSpec,
    CofactorState,
    Conditions,
    Detection,
    ExperimentRecord,
    OutcomeClass,
    ProductSpec,
    ReactionSpec,
    Stereochemistry,
    SubstrateSpec,
    TaskSpec,
    TemplateProvenance,
    TemplateSourceType,
    ee_target,
)
from eagent.tools.ingest_results import (
    MIN_FAMILY_QUOTA_AFTER_NEGATIVE,
    ASSAY_RESULT_COLUMNS,
    IngestResults,
    MeasurementGroup,
    NoHitHypothesis,
    PositiveCriterion,
    build_active_learning_update,
    build_record_id,
    classify_group,
    diagnose_no_hits,
    group_rows,
    next_round_quotas,
    parse_assay_rows,
)


def record_id_for(candidate: str, cofactor: str = "NADPH") -> str:
    """A record id in the shape the module actually writes.

    Built through :func:`build_record_id` rather than typed out, so these
    tests cannot keep passing against an id format the code no longer
    produces -- which is how a record-id change gets through a green suite.
    """
    return build_record_id("run", MeasurementGroup(
        candidate_id=candidate, cofactor=cofactor, cofactor_state="reduced"))

SEQ_A = "MKAAVLYEFGKPLEIKEVEVAPPKAHEVRIKIAYTGVCHT"
SEQ_B = "MKAAVLYEFGKPLEIKEVEVAPPKAHEVRIKIAYTGVCHA"
SEQ_C = "MKAAVLYEFGKPLEIKEVEVAPPKAHEVRIKIAYTGVCHW"


def make_task(*, confirmed: bool = True) -> TaskSpec:
    task = TaskSpec(
        task_id="T1",
        reaction=ReactionSpec(
            substrate=SubstrateSpec(name="acetophenone",
                                    isomeric_smiles="CC(=O)c1ccccc1"),
            product=ProductSpec(name="(R)-1-phenylethanol",
                                isomeric_smiles="C[C@@H](O)c1ccccc1",
                                target_stereochemistry=Stereochemistry.R,
                                creates_new_stereocenter=True)),
        conditions=Conditions(
            pH=7.0, temperature_C=30.0, expression_host="E. coli BL21(DE3)",
            cofactor_options=[
                CofactorSpec(name="NADPH", state=CofactorState.REDUCED),
                CofactorSpec(name="NADH", state=CofactorState.REDUCED)]),
    )
    task.approval.functional_criteria_confirmed = confirmed
    return task


def make_ctx(tmp: Path, task: TaskSpec | None = None) -> RunContext:
    task = task or make_task()
    return RunContext(task=task, workdir=tmp,
                      manifest=RunManifest(run_id="R1", task_id=task.task_id),
                      policy=ExecutionPolicy(allow_network=False))


def tier2_template(**overrides) -> AssayTemplate:
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


def tier1_template(**overrides) -> AssayTemplate:
    kwargs = dict(
        template_id="assay:kred:tier1", tier=1, method="NADPH depletion A340",
        confirms_product_identity=False, chiral_capable=False, replicates=2,
        positive_criteria={"min_conversion_pct": 5.0},
        limit_of_detection=2.0, limit_unit="uM",
        provenance=TemplateProvenance(
            source_type=TemplateSourceType.CURATED_DATABASE,
            identifiers=["SOP-KRED-00"]))
    kwargs.update(overrides)
    return AssayTemplate(**kwargs)


def row(candidate_id: str, **overrides) -> dict[str, object]:
    base: dict[str, object] = {c: "" for c in ASSAY_RESULT_COLUMNS}
    base.update({
        "plan_id": "T1-round-1", "plate": "1", "well": "A1", "slot": "1",
        "candidate_id": candidate_id, "kind": "candidate",
        "role": "high_evidence", "cofactor": "NADPH",
        "cofactor_state": "reduced", "replicate": "1",
        "tested": "yes", "expressed_soluble": "yes",
        "detection_method": "chiral GC-MS",
        "confirms_product_identity": "yes", "authentic_standard": "yes",
        "chiral_method_validated": "yes",
        "limit_of_detection": "0.5", "limit_unit": "uM",
        "measurement_type": "conversion", "measurement_unit": "%",
        "product_identity_observed": "target",
    })
    base.update(overrides)
    return base


def hit_rows(candidate_id: str = "C1", n: int = 2) -> list[dict[str, object]]:
    return [row(candidate_id, replicate=str(i + 1), conversion_pct="41",
                peak_area_target_enantiomer="960",
                peak_area_opposite_enantiomer="40") for i in range(n)]


def negative_rows(candidate_id: str = "C2", n: int = 2,
                  **overrides) -> list[dict[str, object]]:
    return [row(candidate_id, replicate=str(i + 1), conversion_pct="0",
                product_identity_observed="none", **overrides)
            for i in range(n)]


def run(ctx: RunContext, rows, template=None, **kwargs):
    return IngestResults().run(ctx, rows=rows,
                               assay_template=template or tier2_template(),
                               **kwargs)


class CriterionTests(unittest.TestCase):
    def test_unknown_criterion_key_raises(self):
        template = tier2_template(
            positive_criteria={"min_converion_pct": 5.0})   # typo on purpose
        with self.assertRaises(TemplateError) as ctx:
            PositiveCriterion.from_template(template)
        self.assertIn("min_converion_pct", str(ctx.exception))

    def test_empty_criteria_raises(self):
        with self.assertRaises(TemplateError):
            PositiveCriterion.from_template(
                tier2_template(positive_criteria={}))

    def test_missing_quantity_is_undecidable_not_a_negative(self):
        criterion = PositiveCriterion.from_template(tier2_template())
        group = group_rows(parse_assay_rows(
            [row("C1", conversion_pct="40")])[0])[0]
        met, reasons = criterion.evaluate(group)
        self.assertIsNone(met)
        self.assertTrue(any("peak areas were not reported" in r
                            for r in reasons))

    def test_digest_is_stable(self):
        a = PositiveCriterion.from_template(tier2_template())
        b = PositiveCriterion.from_template(tier2_template())
        self.assertEqual(a.digest(), b.digest())


class EeTests(unittest.TestCase):
    def test_signed_ee_goes_negative_for_the_wrong_enantiomer(self):
        self.assertAlmostEqual(ee_target(40.0, 960.0), -92.0)

    def test_group_reports_signed_ee_from_summed_peaks(self):
        rows, _ = parse_assay_rows(hit_rows())
        group = group_rows(rows)[0]
        self.assertAlmostEqual(group.ee_target_pct, 92.0)

    def test_no_product_quantified_gives_none_not_zero(self):
        rows, _ = parse_assay_rows(negative_rows())
        self.assertIsNone(group_rows(rows)[0].ee_target_pct)


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.criterion = PositiveCriterion.from_template(tier2_template())

    def _classify(self, rows, **kwargs):
        parsed, _ = parse_assay_rows(rows)
        return classify_group(group_rows(parsed)[0], self.criterion, **kwargs)

    def test_confirmed_hit(self):
        result = self._classify(hit_rows())
        self.assertIs(result.outcome, OutcomeClass.CONFIRMED_TARGET_PRODUCT)

    def test_untested_well_is_not_a_negative(self):
        result = self._classify([row("C1", tested="no")])
        self.assertIs(result.outcome, OutcomeClass.NOT_TESTED)

    def test_expression_failure_outranks_the_criterion(self):
        result = self._classify(
            [row("C1", expressed_soluble="no", conversion_pct="0")])
        self.assertIs(result.outcome,
                      OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE)
        self.assertFalse(result.outcome.informs_catalytic_ability)

    def test_other_product_is_turnover_not_absence(self):
        result = self._classify(
            [row("C1", conversion_pct="30", product_identity_observed="other")])
        self.assertIs(result.outcome,
                      OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION)

    def test_wrong_configuration_is_kept_apart_from_no_product(self):
        result = self._classify(
            [row("C1", conversion_pct="55",
                 peak_area_target_enantiomer="40",
                 peak_area_opposite_enantiomer="960")])
        self.assertIs(result.outcome,
                      OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION)
        self.assertTrue(any("opposite configuration dominated" in r
                            for r in result.reasons))

    def test_target_product_below_the_selectivity_bar(self):
        result = self._classify(
            [row("C1", conversion_pct="55",
                 peak_area_target_enantiomer="600",
                 peak_area_opposite_enantiomer="400")])
        self.assertIs(result.outcome,
                      OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION)

    def test_negative_uses_the_reported_detection_limit(self):
        result = self._classify(
            negative_rows(peak_area_target_enantiomer="0",
                          peak_area_opposite_enantiomer="0"))
        self.assertIs(result.outcome, OutcomeClass.NO_TARGET_PRODUCT_DETECTED)
        self.assertTrue(any("down to 0.5 uM" in r for r in result.reasons))

    def test_negative_without_any_limit_is_held_unresolved(self):
        result = self._classify(
            negative_rows(limit_of_detection="", limit_unit="",
                          peak_area_target_enantiomer="0",
                          peak_area_opposite_enantiomer="0"))
        self.assertIsNone(result.outcome)
        self.assertEqual(result.unresolved.reason_code,
                         "negative_without_detection_limit")

    def test_negative_falls_back_to_the_pre_registered_limit(self):
        result = self._classify(
            negative_rows(limit_of_detection="", limit_unit="",
                          peak_area_target_enantiomer="0",
                          peak_area_opposite_enantiomer="0"),
            fallback_limit_of_detection=0.5, fallback_limit_unit="uM")
        self.assertIs(result.outcome, OutcomeClass.NO_TARGET_PRODUCT_DETECTED)
        self.assertTrue(any("pre-registered limit" in r for r in result.reasons))

    def test_indirect_signal_cannot_make_a_positive(self):
        criterion = PositiveCriterion.from_template(tier1_template())
        parsed, _ = parse_assay_rows([row(
            "C1", conversion_pct="40", detection_method="NADPH A340",
            confirms_product_identity="no", chiral_method_validated="",
            product_identity_observed="unknown")])
        result = classify_group(group_rows(parsed)[0], criterion)
        self.assertIsNone(result.outcome)
        self.assertEqual(result.unresolved.reason_code,
                         "indirect_signal_positive")
        self.assertIn("lysate does that", result.unresolved.reason)


class ActiveLearningTests(unittest.TestCase):
    @staticmethod
    def _record(rid: str, outcome: OutcomeClass) -> ExperimentRecord:
        detection = Detection(method="chiral GC-MS",
                              confirms_product_identity=True,
                              limit_of_detection=0.5, limit_unit="uM")
        return ExperimentRecord(record_id=rid, outcome=outcome,
                                detection=detection)

    def test_partitions_sum_to_the_input(self):
        records = [
            self._record(record_id_for("C1"), OutcomeClass.CONFIRMED_TARGET_PRODUCT),
            self._record(record_id_for("C2"), OutcomeClass.NO_TARGET_PRODUCT_DETECTED),
            self._record(record_id_for("C3"),
                         OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE),
            self._record(record_id_for("C4"),
                         OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION),
            self._record(record_id_for("C5"), OutcomeClass.NOT_TESTED),
        ]
        update = build_active_learning_update(records)
        self.assertEqual(update.n_total, 5)
        self.assertEqual(len(update.catalytic_negatives), 1)
        self.assertEqual(len(update.expression_failures), 1)
        self.assertEqual(update.n_catalytically_informative, 3)
        self.assertIn("never dropped", update.note)

    def test_expression_failures_are_not_catalytic_negatives(self):
        records = [self._record(record_id_for("C3"),
                                OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE)]
        update = build_active_learning_update(records)
        self.assertEqual(update.catalytic_negatives, ())
        self.assertEqual(len(update.expression_failures), 1)

    def test_partition_is_exhaustive_over_the_whole_taxonomy(self):
        """Every outcome class lands somewhere, so the guard can never fire."""
        records = [self._record(f"run:C{i}:NADPH", outcome)
                   for i, outcome in enumerate(OutcomeClass)]
        update = build_active_learning_update(records)
        self.assertEqual(update.n_total, len(list(OutcomeClass)))

    def test_the_guard_is_unreachable_by_construction(self):
        """The arithmetic check cannot fire today, and that is the assertion.

        Every outcome class falls through to a partition, so
        ``n_total == len(records)`` always holds. The guard exists so that an
        edit which adds a branch -- "skip the expression failures here" is the
        tempting one -- fails loudly instead of quietly shrinking the training
        set. This case pins the invariant the guard protects.
        """
        for outcome in OutcomeClass:
            update = build_active_learning_update(
                [self._record(record_id_for("C1"), outcome)])
            self.assertEqual(update.n_total, 1, outcome.value)


class QuotaTests(unittest.TestCase):
    @staticmethod
    def _record(rid: str, outcome: OutcomeClass) -> ExperimentRecord:
        return ExperimentRecord(
            record_id=rid, outcome=outcome,
            detection=Detection(method="chiral GC-MS",
                                confirms_product_identity=True,
                                limit_of_detection=0.5, limit_unit="uM"))

    def test_hit_family_expands_and_negative_family_is_floored(self):
        records = [
            self._record(record_id_for("C1"), OutcomeClass.CONFIRMED_TARGET_PRODUCT),
            self._record(record_id_for("C2"), OutcomeClass.NO_TARGET_PRODUCT_DETECTED),
        ]
        verdicts = {v.family: v for v in next_round_quotas(
            records, {"C1": "SDR", "C2": "AKR"}, {"SDR": 10, "AKR": 10})}
        self.assertEqual(verdicts["SDR"].tier, "expand")
        self.assertEqual(verdicts["SDR"].suggested_quota, 20)
        self.assertEqual(verdicts["AKR"].tier, "reduce")
        self.assertEqual(verdicts["AKR"].suggested_quota, 5)
        self.assertGreaterEqual(verdicts["AKR"].suggested_quota,
                                MIN_FAMILY_QUOTA_AFTER_NEGATIVE)

    def test_negative_family_is_never_cut_to_zero(self):
        records = [self._record(record_id_for("C2"),
                                OutcomeClass.NO_TARGET_PRODUCT_DETECTED)]
        verdict = next_round_quotas(records, {"C2": "AKR"}, {"AKR": 1})[0]
        self.assertEqual(verdict.suggested_quota, MIN_FAMILY_QUOTA_AFTER_NEGATIVE)
        self.assertIn("unlucky round", verdict.rationale)

    def test_expression_failure_family_holds_rather_than_shrinks(self):
        records = [self._record(record_id_for("C3"),
                                OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE)]
        verdict = next_round_quotas(records, {"C3": "MDR"}, {"MDR": 8})[0]
        self.assertEqual(verdict.tier, "re-express")
        self.assertEqual(verdict.suggested_quota, 8)

    def test_wrong_configuration_expands_like_a_hit(self):
        records = [self._record(
            record_id_for("C4"), OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION)]
        verdict = next_round_quotas(records, {"C4": "SDR"}, {"SDR": 4})[0]
        self.assertEqual(verdict.tier, "expand")
        self.assertIn("working scaffold", verdict.rationale)

    def test_hit_rate_carries_a_wilson_interval(self):
        records = [self._record(f"run:C{i}:NADPH",
                                OutcomeClass.NO_TARGET_PRODUCT_DETECTED)
                   for i in range(4)]
        verdict = next_round_quotas(
            records, {f"C{i}": "AKR" for i in range(4)})[0]
        self.assertEqual(verdict.hit_rate, 0.0)
        self.assertIsNotNone(verdict.hit_rate_ci)
        self.assertGreater(verdict.hit_rate_ci[1], 0.0)


class NoHitDiagnosisTests(unittest.TestCase):
    @staticmethod
    def _record(rid: str, outcome: OutcomeClass) -> ExperimentRecord:
        return ExperimentRecord(
            record_id=rid, outcome=outcome,
            detection=Detection(method="chiral GC-MS",
                                confirms_product_identity=True,
                                limit_of_detection=0.5, limit_unit="uM"))

    def test_five_hypotheses_and_no_natural_catalyst_is_never_concluded(self):
        records = [self._record(record_id_for("C1"),
                                OutcomeClass.NO_TARGET_PRODUCT_DETECTED)]
        parsed, _ = parse_assay_rows(negative_rows("C1"))
        diagnosis = diagnose_no_hits(records, group_rows(parsed), {"C1": "SDR"},
                                     cofactor_conditions_run=["NADPH"])
        self.assertTrue(diagnosis.triggered)
        self.assertEqual(len(diagnosis.assessments), 5)
        by_name = {a.hypothesis: a for a in diagnosis.assessments}
        self.assertEqual(by_name[NoHitHypothesis.NO_NATURAL_CATALYST].status,
                         "undetermined")
        self.assertIn("cannot establish the absence",
                      by_name[NoHitHypothesis.NO_NATURAL_CATALYST]
                      .contradicted_by[0])

    def test_single_cofactor_condition_supports_the_mismatch_hypothesis(self):
        records = [self._record(record_id_for("C1"),
                                OutcomeClass.NO_TARGET_PRODUCT_DETECTED)]
        parsed, _ = parse_assay_rows(negative_rows("C1"))
        diagnosis = diagnose_no_hits(
            records, group_rows(parsed), {"C1": "SDR"},
            cofactor_conditions_run=["NADPH"],
            declared_cofactor_preferences={"SDR": "NADH"})
        mismatch = {a.hypothesis: a for a in diagnosis.assessments}[
            NoHitHypothesis.COFACTOR_MISMATCH]
        self.assertEqual(mismatch.status, "consistent")
        self.assertTrue(any("NADH" in s for s in mismatch.supported_by))

    def test_failed_system_control_invalidates_the_negatives(self):
        records = [self._record(record_id_for("C1"),
                                OutcomeClass.NO_TARGET_PRODUCT_DETECTED)]
        parsed, _ = parse_assay_rows(negative_rows("C1"))
        diagnosis = diagnose_no_hits(records, group_rows(parsed), {"C1": "SDR"},
                                     system_control_worked=False)
        detection = {a.hypothesis: a for a in diagnosis.assessments}[
            NoHitHypothesis.DETECTION_CONDITIONS_UNSUITABLE]
        self.assertEqual(detection.status, "consistent")

    def test_not_triggered_when_there_is_a_hit(self):
        records = [self._record(record_id_for("C1"),
                                OutcomeClass.CONFIRMED_TARGET_PRODUCT)]
        parsed, _ = parse_assay_rows(hit_rows("C1"))
        diagnosis = diagnose_no_hits(records, group_rows(parsed), {"C1": "SDR"})
        self.assertFalse(diagnosis.triggered)


class InterfaceTests(unittest.TestCase):
    def test_gate_blocks_without_confirmed_criteria(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp), make_task(confirmed=False))
            out = run(ctx, hit_rows())
            self.assertIs(out.status, Status.FAILED)
            self.assertTrue(any(f.code == "approval_required"
                                for f in out.blockers))

    def test_no_assay_template_is_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = IngestResults().run(make_ctx(Path(tmp)), rows=hit_rows(),
                                      assay_template=None)
            self.assertIs(out.status, Status.FAILED)
            self.assertTrue(any(f.code == "no_pre_registered_criterion"
                                for f in out.blockers))

    def test_four_outcomes_survive_ingestion_as_four(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            rows = (hit_rows("C1")
                    + negative_rows("C2", peak_area_target_enantiomer="0",
                                    peak_area_opposite_enantiomer="0")
                    + [row("C3", expressed_soluble="no", conversion_pct="0",
                           product_identity_observed="none")]
                    + [row("C4", conversion_pct="30",
                           product_identity_observed="other")]
                    + [row("C5", tested="no")])
            out = run(ctx, rows,
                      sequences={"C1": SEQ_A, "C2": SEQ_B, "C3": SEQ_C})
            counts = out.data["outcome_counts"]
            self.assertEqual(counts["confirmed_target_product"], 1)
            self.assertEqual(counts["no_target_product_detected"], 1)
            self.assertEqual(counts["expression_or_solubility_failure"], 1)
            self.assertEqual(counts["other_product_or_wrong_configuration"], 1)
            self.assertEqual(counts["not_tested"], 1)

    def test_indirect_positive_is_a_blocker_and_is_not_a_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = run(ctx, [row("C1", conversion_pct="40",
                                detection_method="NADPH A340",
                                confirms_product_identity="no",
                                product_identity_observed="unknown")],
                      template=tier1_template())
            self.assertTrue(any(f.code == "indirect_signal_positive"
                                and f.severity is Severity.BLOCKER
                                for f in out.qc_flags))
            self.assertEqual(out.data["n_records"], 0)
            self.assertEqual(out.data["n_unresolved"], 1)
            self.assertNotIn("no_target_product_detected",
                             out.data["outcome_counts"])

    def test_criterion_override_is_refused_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = run(ctx, hit_rows(),
                      criterion_override={"min_conversion_pct": 0.1})
            deviations = out.data["protocol_deviations"]
            self.assertEqual(len(deviations), 1)
            self.assertEqual(deviations[0]["applied"],
                             {"min_conversion_pct": 5.0,
                              "min_ee_target_pct": 80.0})
            self.assertEqual(deviations[0]["attempted"],
                             {"min_conversion_pct": 0.1})
            self.assertTrue(any(f.code == "protocol_deviation_refused"
                                for f in out.qc_flags))
            self.assertTrue(out.provenance.parameters
                            ["criterion_override_refused"])

    def test_signed_ee_reaches_the_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = run(ctx, [row("C1", conversion_pct="55",
                                peak_area_target_enantiomer="40",
                                peak_area_opposite_enantiomer="960")],
                      sequences={"C1": SEQ_A})
            record = out.data["records"][0]
            self.assertLess(record["ee_target_pct"], 0)
            self.assertEqual(record["outcome"],
                             "other_product_or_wrong_configuration")

    def test_replicate_disagreement_is_surfaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            rows = [row("C1", replicate="1", conversion_pct="40",
                        peak_area_target_enantiomer="960",
                        peak_area_opposite_enantiomer="40"),
                    row("C1", replicate="2", conversion_pct="1",
                        peak_area_target_enantiomer="960",
                        peak_area_opposite_enantiomer="40")]
            out = run(ctx, rows, sequences={"C1": SEQ_A})
            self.assertTrue(any(f.code == "replicate_disagreement"
                                for f in out.qc_flags))
            self.assertEqual(out.data["n_groups"], 1)

    def test_cofactor_condition_is_part_of_the_record_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            rows = (hit_rows("C1")
                    + negative_rows("C1", cofactor="NADH",
                                    peak_area_target_enantiomer="0",
                                    peak_area_opposite_enantiomer="0"))
            out = run(ctx, rows, sequences={"C1": SEQ_A})
            self.assertEqual(out.data["n_records"], 2)
            outcomes = {r["cofactor"]["name"]: r["outcome"]
                        for r in out.data["records"]}
            self.assertEqual(outcomes["NADPH"], "confirmed_target_product")
            self.assertEqual(outcomes["NADH"], "no_target_product_detected")

    def test_three_layers_report_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = run(ctx, hit_rows() + negative_rows(
                "C2", peak_area_target_enantiomer="0",
                peak_area_opposite_enantiomer="0"),
                sequences={"C1": SEQ_A, "C2": SEQ_B},
                families={"C1": "SDR", "C2": "AKR"})
            layers = out.data["layers"]
            self.assertTrue(layers["data_layer"]["changed"])
            self.assertTrue(layers["model_layer"]["changed"])
            self.assertTrue(layers["decision_layer"]["changed"])
            self.assertIn("next_round_family_quotas",
                          layers["decision_layer"]["state"])
            self.assertEqual(
                layers["model_layer"]["state"]["specificity"]
                ["n_informative_records"], 2)

    def test_negatives_are_retained_in_the_learning_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = run(ctx, hit_rows() + negative_rows(
                "C2", peak_area_target_enantiomer="0",
                peak_area_opposite_enantiomer="0"),
                sequences={"C1": SEQ_A, "C2": SEQ_B})
            learning = out.data["active_learning"]
            self.assertEqual(learning["negatives_retained"], 1)
            self.assertEqual(learning["n_total"], out.data["n_records"])

    def test_failed_system_control_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            control = row("positive_enzyme", kind="control",
                          role="positive_enzyme control", conversion_pct="0",
                          product_identity_observed="none",
                          peak_area_target_enantiomer="0",
                          peak_area_opposite_enantiomer="0")
            out = run(ctx, negative_rows("C2",
                                         peak_area_target_enantiomer="0",
                                         peak_area_opposite_enantiomer="0")
                      + [control], sequences={"C2": SEQ_B})
            self.assertFalse(out.data["system_control_worked"])
            self.assertTrue(any(f.code == "assay_system_control_failed"
                                for f in out.blockers))

    def test_no_hit_round_is_preserved_in_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = run(ctx, negative_rows("C2",
                                         peak_area_target_enantiomer="0",
                                         peak_area_opposite_enantiomer="0"),
                      sequences={"C2": SEQ_B}, families={"C2": "AKR"})
            diagnosis = out.data["no_hit_diagnosis"]
            self.assertTrue(diagnosis["triggered"])
            self.assertEqual(len(diagnosis["hypotheses"]), 5)
            self.assertTrue(any(f.code == "no_hits_in_round"
                                for f in out.qc_flags))
            self.assertEqual(out.data["n_records"], 1)


class FileAndArtifactTests(unittest.TestCase):
    def test_missing_column_refuses_the_whole_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            columns = [c for c in ASSAY_RESULT_COLUMNS if c != "notes"]
            with open(path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=columns)
                writer.writeheader()
                writer.writerow({c: hit_rows()[0].get(c, "") for c in columns})
            ctx = make_ctx(Path(tmp))
            out = IngestResults().run(ctx, results_csv=path,
                                      assay_template=tier2_template())
            self.assertIs(out.status, Status.FAILED)
            self.assertTrue(any(f.code == "unreadable_results"
                                for f in out.blockers))

    def test_round_trip_through_a_csv_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            with open(path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(ASSAY_RESULT_COLUMNS))
                writer.writeheader()
                writer.writerows(hit_rows())
            ctx = make_ctx(Path(tmp))
            out = IngestResults().run(ctx, results_csv=path,
                                      assay_template=tier2_template(),
                                      sequences={"C1": SEQ_A})
            self.assertEqual(out.data["n_records"], 1)

    def test_artifacts_are_written_and_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = run(ctx, hit_rows() + negative_rows(
                "C2", peak_area_target_enantiomer="0",
                peak_area_opposite_enantiomer="0"),
                sequences={"C1": SEQ_A, "C2": SEQ_B},
                families={"C1": "SDR", "C2": "AKR"})
            records_path = Path(out.artifact("experiment_records").path)
            lines = [json.loads(line) for line
                     in records_path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(lines), out.data["n_records"])

            pending = Path(out.artifact("pending_confirmation").path)
            self.assertTrue(pending.exists())

            summary = Path(out.artifact("round_summary").path).read_text()
            self.assertIn("Pre-registered criterion", summary)
            self.assertIn("no_target_product_detected", summary)
            self.assertIn("No-hit differential", summary)

            proposal = yaml.safe_load(
                Path(out.artifact("next_round_task").path).read_text())
            self.assertEqual(proposal["proposed_round"], 2)
            self.assertIn("SDR", proposal["family_quotas"])
            self.assertIn("negatives_retained", proposal["derived_from"])

    def test_provenance_pins_the_criterion(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            out = run(ctx, hit_rows(), sequences={"C1": SEQ_A})
            self.assertEqual(out.provenance.tool, "ingest_results")
            self.assertEqual(out.provenance.parameters["positive_criteria"],
                             {"min_conversion_pct": 5.0,
                              "min_ee_target_pct": 80.0})
            self.assertEqual(
                out.provenance.parameters["positive_criteria_sha256"],
                out.data["positive_criteria_sha256"])
            self.assertIsNotNone(out.provenance.random_seed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
