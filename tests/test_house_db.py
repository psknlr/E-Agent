"""Tests for :mod:`eagent.datalayer.house_db`.

Every expected value here is decided by the construction of the fixture rather
than by running the code and recording what it printed. The central fixture is
one screening round of six constructs in which three fail in three different
ways: one hit out of six submitted, one hit out of three that actually
expressed. Both numbers are true by construction, so a regression that starts
reporting only the flattering one fails the test instead of redefining it.

The guards under test are the ones that cost something to keep: a prediction
cannot be written once its results exist, a failed construct cannot be deleted,
a variant cannot be compared with its parent across differing conditions, and
the three performance axes have nowhere to be collapsed into one number.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - standalone execution
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from eagent.datalayer.house_db import (
    FORBIDDEN_COMBINED_SCORE_NAMES,
    SCHEMA_VERSION,
    THERE_IS_NO_COMBINED_MUTATION_SCORE,
    AxisReadout,
    AxisRecordKind,
    CollapsedScoreError,
    ConditionMismatchError,
    DeletionRefusedError,
    EnrichmentResult,
    ExpressionStatus,
    FrozenPredictionError,
    HitRate,
    HouseDB,
    PredictionEntry,
    PredictionOutcomeReport,
    RecordOverwriteError,
    SchemaVersionError,
    UnknownRecordError,
    condition_key,
)
from eagent.errors import FabricationGuardError, UnresolvedFieldError
from eagent.provenance import sequence_hash
from eagent.schemas.chem import CofactorSpec, CofactorState, ProductSpec, SubstrateSpec
from eagent.schemas.reaction import Conditions
from eagent.schemas.record import (
    Detection,
    EvidenceRef,
    EvidenceStrength,
    ExperimentRecord,
    OutcomeClass,
    ReactionDirection,
)
from eagent.schemas.variant import EffectDirection, PerformanceAxis

TARGET = "acetophenone-to-R-1-phenylethanol"

#: Two condition sets that differ in exactly one field, so a refusal has one
#: thing to name.
COND_A: dict[str, object] = {
    "pH": 7.0,
    "temperature_C": 30.0,
    "buffer": "potassium phosphate 100 mM",
    "solvent_system": "aqueous",
    "cosolvent_fraction": 0.05,
    "substrate_concentration_mM": 10.0,
    "enzyme_loading": "1 mg/mL lysate",
    "reaction_time_h": 24.0,
    "expression_host": "E. coli BL21(DE3)",
}
COND_B: dict[str, object] = dict(COND_A, pH=8.0)


def _seq(tag: str) -> str:
    """A distinct, deterministic test sequence per tag."""
    return "MKALVTGAAGFIGSHLVDRLL" * 3 + tag * 4


def _detection(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "method": "chiral GC-MS against authentic standard",
        "confirms_product_identity": True,
        "chiral_method_validated": True,
        "authentic_standard": True,
        "limit_of_detection": 0.05,
        "limit_unit": "mM",
    }
    base.update(kw)
    return base


def _result(record_id: str, sha: str, outcome: OutcomeClass, **kw: object) -> dict:
    row: dict[str, object] = {
        "record_id": record_id,
        "round_id": kw.pop("round_id", "round-1"),
        "sequence_sha256": sha,
        "outcome": outcome,
        "reaction_direction": ReactionDirection.FORWARD_AS_TARGET,
        "cofactor_species": "NADPH",
        "cofactor_state": CofactorState.REDUCED,
        "conditions": dict(kw.pop("conditions", COND_A)),
        "detection": _detection(**(kw.pop("detection", {}) or {})),
    }
    row.update(kw)
    return row


def _fixture_db() -> tuple[HouseDB, dict[str, str]]:
    """One substrate target, six constructs, one round, predictions frozen first.

    Composition of the batch, fixed by hand so the expected rates are
    arithmetic rather than observation:

    ======  ==================================  =========  =====
    tag     outcome                             expressed  hit
    ======  ==================================  =========  =====
    c1      confirmed_target_product            yes        yes
    c2      no_target_product_detected          yes        no
    c3      other_product_or_wrong_config.      yes        no
    c4      expression_or_solubility_failure    no         no
    c5      expression_or_solubility_failure    no         no
    c6      not_tested                          unknown    no
    ======  ==================================  =========  =====
    """
    db = HouseDB(":memory:")
    db.register_substrate_target(
        TARGET,
        label="acetophenone -> (R)-1-phenylethanol",
        substrate_ladder={"rungs": [{"rung": "isomeric_smiles",
                                     "value": "CC(=O)c1ccccc1"}]},
        product_ladder={"rungs": [{"rung": "isomeric_smiles",
                                   "value": "C[C@@H](O)c1ccccc1"}]},
        target_stereochemistry="R",
        creates_new_stereocenter=True,
        # No Rhea identifier is asserted here: inventing one in a fixture is
        # the same fabrication the module exists to prevent.
        reaction_id=None,
    )
    tags = ["c1", "c2", "c3", "c4", "c5", "c6"]
    sha = {}
    for tag in tags:
        sha[tag] = db.upsert_candidate(
            sequence=_seq(tag), candidate_id=f"ADH-{tag}", origin="natural",
            family="SDR", family_basis="HMM hit to PF00106",
            organism="test organism",
        )
    db.create_round("round-1", TARGET, round_number=1)
    db.freeze_predictions("round-1", [
        PredictionEntry(sha["c3"], 1, "high_evidence", "closest characterised homolog",
                        "docking used a transplanted cofactor",
                        stereo_call="favors_target",
                        scorecard={"docking_result": -8.1, "catalytic_geometry": 1}),
        PredictionEntry(sha["c1"], 2, "diversity", "different subfamily, same motif",
                        "no structure for this clade",
                        stereo_call="favors_target"),
        PredictionEntry(sha["c2"], 3, "uncertainty_probe", "mechanism plausible",
                        "model disagrees with family annotation"),
        PredictionEntry(sha["c4"], 4, "diversity", "thermophile clade",
                        "expression untested in this host"),
        PredictionEntry(sha["c5"], 5, "diversity", "halophile clade",
                        "expression untested in this host"),
        PredictionEntry(sha["c6"], 6, "uncertainty_probe", "long shot",
                        "no evidence at all"),
    ], snapshot_id="snapshot-round-1")

    db.ingest_round([
        _result("rec-c1", sha["c1"], OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                ee_target_pct=92.0, conversion_pct=61.0,
                measurement_type="conversion", measurement_value=61.0,
                measurement_unit="%", soluble_expression=True,
                product_smiles="C[C@@H](O)c1ccccc1"),
        _result("rec-c2", sha["c2"], OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                measurement_type="conversion", measurement_value=0.0,
                measurement_unit="%", soluble_expression=True,
                detection={"confirms_product_identity": False}),
        _result("rec-c3", sha["c3"], OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION,
                ee_target_pct=-95.0, conversion_pct=70.0,
                measurement_type="conversion", measurement_value=70.0,
                measurement_unit="%", soluble_expression=True,
                product_smiles="C[C@H](O)c1ccccc1"),
        _result("rec-c4", sha["c4"], OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE,
                expression_status=ExpressionStatus.INSOLUBLE,
                detection={"confirms_product_identity": False}),
        _result("rec-c5", sha["c5"], OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE,
                expression_status=ExpressionStatus.NOT_DETECTED,
                detection={"confirms_product_identity": False}),
        _result("rec-c6", sha["c6"], OutcomeClass.NOT_TESTED,
                detection={"confirms_product_identity": False,
                           "limit_of_detection": None}),
    ])
    return db, sha


# --------------------------------------------------------------------------


class TestSchema(unittest.TestCase):
    def test_fresh_database_is_at_the_current_schema_version(self) -> None:
        db = HouseDB(":memory:")
        self.assertEqual(db.schema_version(), SCHEMA_VERSION)
        for table in ("substrate_target", "candidate", "lineage", "prediction",
                      "experiment_round", "experiment_record", "performance_axes",
                      "evidence", "lineage_group", "schema_version"):
            self.assertIn(table, db.table_names())

    def test_migrate_is_idempotent(self) -> None:
        db = HouseDB(":memory:")
        self.assertEqual(db.migrate(), SCHEMA_VERSION)
        self.assertEqual(db.migrate(), SCHEMA_VERSION)

    def test_data_survives_reopening_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "house.sqlite")
            db = HouseDB(path)
            db.register_substrate_target(
                TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
                product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
            db.close()
            again = HouseDB(path)
            self.assertEqual(again.schema_version(), SCHEMA_VERSION)
            self.assertEqual(again.substrate_target(TARGET)["substrate_target_id"],
                             TARGET)
            again.close()

    def test_a_file_from_a_newer_schema_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "house.sqlite")
            HouseDB(path).close()
            raw = sqlite3.connect(path)
            raw.execute("INSERT INTO schema_version (version, applied_at, "
                        "description) VALUES (999, '2999-01-01', 'from the future')")
            raw.commit()
            raw.close()
            with self.assertRaises(SchemaVersionError):
                HouseDB(path)


class TestFrozenPredictions(unittest.TestCase):
    def test_a_posthoc_prediction_edit_is_rejected(self) -> None:
        db, sha = _fixture_db()
        before = db.predictions("round-1")
        self.assertEqual([p["predicted_rank"] for p in before], [1, 2, 3, 4, 5, 6])
        with self.assertRaises(FrozenPredictionError):
            db.freeze_predictions("round-1", [
                PredictionEntry(sha["c1"], 1, "high_evidence",
                                "obviously the best one", "none at all"),
            ], snapshot_id="snapshot-round-1-rewritten",
                replace_reason="tidying up after the fact")
        after = db.predictions("round-1")
        self.assertEqual([p["sequence_sha256"] for p in before],
                         [p["sequence_sha256"] for p in after])
        self.assertEqual([p["content_sha256"] for p in before],
                         [p["content_sha256"] for p in after])

    def test_predictions_are_frozen_before_the_results_arrive(self) -> None:
        db, _ = _fixture_db()
        batch = db.batch_outcomes("round-1")
        self.assertIsNotNone(batch.predictions_frozen_at)
        self.assertIsNotNone(batch.results_ingested_at)
        self.assertLessEqual(batch.predictions_frozen_at, batch.results_ingested_at)
        self.assertTrue(db.prediction_vs_outcome("round-1").frozen_before_results)

    def test_raw_sql_cannot_edit_a_frozen_prediction_either(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "house.sqlite")
            db = HouseDB(path)
            db.register_substrate_target(
                TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
                product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
            sha = db.upsert_candidate(sequence=_seq("z"))
            db.create_round("r", TARGET)
            db.freeze_predictions(
                "r", [PredictionEntry(sha, 1, "high_evidence", "only candidate",
                                      "single pose")], snapshot_id="snap")
            db.ingest_round([_result("rec", sha,
                                     OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                                     round_id="r", soluble_expression=True)])
            raw = sqlite3.connect(path)
            with self.assertRaises(sqlite3.IntegrityError):
                raw.execute("UPDATE prediction SET predicted_rank = 99")
                raw.commit()
            raw.rollback()
            with self.assertRaises(sqlite3.IntegrityError):
                raw.execute("DELETE FROM prediction")
                raw.commit()
            raw.rollback()
            raw.close()
            self.assertEqual(db.predictions("r")[0]["predicted_rank"], 1)
            db.close()

    def test_refreezing_before_results_needs_a_reason_and_is_logged(self) -> None:
        db = HouseDB(":memory:")
        db.register_substrate_target(
            TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
        a = db.upsert_candidate(sequence=_seq("a"))
        b = db.upsert_candidate(sequence=_seq("b"))
        db.create_round("r", TARGET)
        db.freeze_predictions("r", [PredictionEntry(a, 1, "diversity", "r", "u")],
                              snapshot_id="s1")
        with self.assertRaises(FrozenPredictionError):
            db.freeze_predictions("r", [PredictionEntry(b, 1, "diversity", "r", "u")],
                                  snapshot_id="s2")
        db.freeze_predictions("r", [PredictionEntry(b, 1, "diversity", "r", "u")],
                              snapshot_id="s2", replace_reason="plate changed")
        log = db.freeze_log("r")
        self.assertEqual(len(log), 2)
        self.assertEqual(log[1]["n_replaced"], 1)
        self.assertEqual(log[1]["reason"], "plate changed")
        self.assertEqual(log[0]["reason"], "initial freeze")

    def test_a_numeric_predicted_ee_requires_a_calibration_source(self) -> None:
        with self.assertRaises(FabricationGuardError):
            PredictionEntry("sha256:x", 1, "diversity", "reason", "uncertainty",
                            predicted_ee_pct=95.0)

    def test_duplicate_ranks_are_refused(self) -> None:
        db, sha = _fixture_db()
        db.create_round("round-2", TARGET, round_number=2)
        with self.assertRaises(ValueError):
            db.freeze_predictions("round-2", [
                PredictionEntry(sha["c1"], 1, "diversity", "r", "u"),
                PredictionEntry(sha["c2"], 1, "diversity", "r", "u"),
            ], snapshot_id="s")

    def test_a_prediction_without_a_stated_uncertainty_is_refused(self) -> None:
        with self.assertRaises(UnresolvedFieldError):
            PredictionEntry("sha256:x", 1, "diversity", "a reason", "")


class TestFullBatch(unittest.TestCase):
    def test_every_submitted_construct_including_failures_is_retrievable(self) -> None:
        db, sha = _fixture_db()
        batch = db.batch_outcomes("round-1")
        self.assertEqual(batch.n_submitted, 6)
        self.assertEqual({r.record_id for r in batch.rows},
                         {f"rec-c{i}" for i in range(1, 7)})
        self.assertEqual(batch.n_expression_failures, 2)
        self.assertEqual(batch.n_not_tested, 1)
        counts = batch.by_outcome()
        self.assertEqual(counts["confirmed_target_product"], 1)
        self.assertEqual(counts["expression_or_solubility_failure"], 2)
        self.assertEqual(counts["not_tested"], 1)
        # every outcome class is reported, including the ones at zero
        self.assertEqual(set(counts), {o.value for o in OutcomeClass})

    def test_the_failures_a_paper_would_drop_are_listed(self) -> None:
        db, _ = _fixture_db()
        failures = db.batch_outcomes("round-1").failures()
        self.assertEqual({r.record_id for r in failures},
                         {"rec-c2", "rec-c3", "rec-c4", "rec-c5", "rec-c6"})

    def test_expression_tri_state_keeps_unassessed_separate(self) -> None:
        db, _ = _fixture_db()
        rows = {r.record_id: r for r in db.batch_outcomes("round-1").rows}
        self.assertTrue(rows["rec-c1"].expressed)
        self.assertFalse(rows["rec-c4"].expressed)
        self.assertIsNone(rows["rec-c6"].expressed)


class TestHitRate(unittest.TestCase):
    def test_both_denominators_are_returned_and_differ(self) -> None:
        db, _ = _fixture_db()
        hr = db.hit_rate("round-1")
        self.assertEqual(hr.n_submitted, 6)
        self.assertEqual(hr.n_expressed, 3)
        self.assertEqual(hr.n_expression_failed, 2)
        self.assertEqual(hr.n_expression_unknown, 1)
        self.assertEqual(hr.n_hits, 1)
        self.assertAlmostEqual(hr.rate_all_submitted, 1 / 6)
        self.assertAlmostEqual(hr.rate_expressed_only, 1 / 3)
        self.assertNotAlmostEqual(hr.rate_all_submitted, hr.rate_expressed_only)
        d = hr.as_dict()
        self.assertTrue(d["denominators_differ"])
        self.assertIn("rate_all_submitted", d)
        self.assertIn("rate_expressed_only", d)
        self.assertIn("of 6 submitted", hr.describe())
        self.assertIn("of 3 expressed", hr.describe())

    def test_expressed_only_rate_is_undefined_when_nothing_expressed(self) -> None:
        db = HouseDB(":memory:")
        db.register_substrate_target(
            TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
        sha = db.upsert_candidate(sequence=_seq("q"))
        db.create_round("r", TARGET)
        db.ingest_round([_result("rec", sha,
                                 OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE,
                                 round_id="r",
                                 detection={"confirms_product_identity": False})])
        hr = db.hit_rate("r")
        self.assertEqual(hr.n_expressed, 0)
        self.assertIsNone(hr.rate_expressed_only)
        self.assertEqual(hr.rate_all_submitted, 0.0)

    def test_there_is_no_single_hit_rate_attribute_to_quote(self) -> None:
        db, _ = _fixture_db()
        hr = db.hit_rate("round-1")
        for forbidden in ("hit_rate", "rate", "value", "percent"):
            self.assertFalse(hasattr(hr, forbidden),
                             f"HitRate must not expose a single {forbidden!r}")
        self.assertIsInstance(hr, HitRate)


class TestExpressionIsNeverInvented(unittest.TestCase):
    """Regression guard for the expressed-only denominator.

    ``RecordRow.expressed`` used to fall through to
    ``outcome.informs_catalytic_ability``, which is true for
    ``no_target_product_detected`` and
    ``other_product_or_wrong_configuration`` as well as for a hit. A negative
    whose expression nobody assessed was therefore reported as successfully
    expressed, the expression-unknown count went to zero, and the
    expressed-only hit rate was computed over a denominator containing
    constructs no one had ever seen on a gel.
    """

    def _unassessed_db(self) -> HouseDB:
        """One hit with expression data; a negative and an other-product row
        with none at all. All three inform catalysis; only one expressed."""
        db = HouseDB(":memory:")
        db.register_substrate_target(
            TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
        hit = db.upsert_candidate(sequence=_seq("h"))
        neg = db.upsert_candidate(sequence=_seq("n"))
        oth = db.upsert_candidate(sequence=_seq("o"))
        db.create_round("r", TARGET)
        db.ingest_round([
            _result("rec-hit", hit, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="r", soluble_expression=True, conversion_pct=55.0,
                    measurement_type="conversion", measurement_value=55.0,
                    measurement_unit="%",
                    product_smiles="C[C@@H](O)c1ccccc1"),
            # expression never assessed: no soluble_expression, no status
            _result("rec-neg", neg, OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                    round_id="r",
                    detection={"confirms_product_identity": False}),
            _result("rec-other", oth,
                    OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION,
                    round_id="r", conversion_pct=40.0,
                    measurement_type="conversion", measurement_value=40.0,
                    measurement_unit="%",
                    detection={"confirms_product_identity": False}),
        ])
        return db

    def test_an_unassessed_negative_counts_as_expression_unknown(self) -> None:
        db = self._unassessed_db()
        row = {r.record_id: r for r in db.batch_outcomes("r").rows}["rec-neg"]
        self.assertIs(row.expression_status, ExpressionStatus.NOT_ASSESSED)
        self.assertIsNone(row.soluble_expression)
        self.assertTrue(row.outcome.informs_catalytic_ability,
                        "the fixture must exercise the informative-outcome path")
        self.assertIsNone(row.expressed,
                          "a negative nobody ran a gel on is expression-unknown")

    def test_an_unassessed_other_product_row_counts_as_expression_unknown(
            self) -> None:
        db = self._unassessed_db()
        row = {r.record_id: r for r in db.batch_outcomes("r").rows}["rec-other"]
        self.assertIs(row.expression_status, ExpressionStatus.NOT_ASSESSED)
        self.assertTrue(row.outcome.informs_catalytic_ability)
        self.assertIsNone(row.expressed)

    def test_the_unassessed_rows_stay_out_of_the_expressed_denominator(self) -> None:
        db = self._unassessed_db()
        hr = db.hit_rate("r")
        self.assertEqual(hr.n_submitted, 3)
        self.assertEqual(hr.n_informative, 3)
        self.assertEqual(hr.n_expressed, 1)
        self.assertEqual(hr.n_expression_unknown, 2)
        self.assertEqual(hr.n_expression_failed, 0)
        self.assertEqual(hr.n_hits, 1)
        self.assertAlmostEqual(hr.rate_expressed_only, 1.0)
        self.assertAlmostEqual(hr.rate_all_submitted, 1 / 3)
        # the counts still add up: nothing was invented into either bucket
        self.assertEqual(
            hr.n_expressed + hr.n_expression_failed + hr.n_expression_unknown,
            hr.n_submitted)
        self.assertIn("2 expression unassessed", hr.describe())

    def test_a_confirmed_product_still_implies_the_protein_existed(self) -> None:
        """The one sound inference is kept: product was made, so protein was there."""
        db = HouseDB(":memory:")
        db.register_substrate_target(
            TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
        sha = db.upsert_candidate(sequence=_seq("h"))
        db.create_round("r", TARGET)
        db.ingest_round([
            _result("rec", sha, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="r", conversion_pct=55.0,
                    measurement_type="conversion", measurement_value=55.0,
                    measurement_unit="%",
                    product_smiles="C[C@@H](O)c1ccccc1")])
        row = db.batch_outcomes("r").rows[0]
        self.assertIs(row.expression_status, ExpressionStatus.NOT_ASSESSED)
        self.assertTrue(row.expressed)
        self.assertEqual(db.hit_rate("r").n_expression_unknown, 0)


class TestVariantVsParent(unittest.TestCase):
    def _two_variants(self) -> tuple[HouseDB, str, str, str]:
        """Parent plus two variants: one assayed with it, one at a different pH."""
        db = HouseDB(":memory:")
        db.register_substrate_target(
            TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
        parent = db.upsert_candidate(sequence=_seq("p"), candidate_id="parent")
        same = db.upsert_candidate(sequence=_seq("s"), candidate_id="v-same",
                                   origin="engineered",
                                   construct_sequence=_seq("s"))
        other = db.upsert_candidate(sequence=_seq("o"), candidate_id="v-other",
                                    origin="engineered",
                                    construct_sequence=_seq("o"))
        for v, label in ((same, "W110A"), (other, "F285L")):
            db.add_lineage_edge(parent_sha256=parent, variant_sha256=v,
                                mutations=[label],
                                numbering_reference="parent sequence 1-based")
        db.create_round("r", TARGET)
        db.ingest_round([
            _result("rec-parent", parent, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="r", ee_target_pct=40.0, conversion_pct=30.0,
                    measurement_type="conversion", measurement_value=30.0,
                    measurement_unit="%", soluble_expression=True,
                    product_smiles="C[C@@H](O)c1ccccc1"),
            _result("rec-same", same, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="r", ee_target_pct=88.0, conversion_pct=55.0,
                    measurement_type="conversion", measurement_value=55.0,
                    measurement_unit="%", soluble_expression=True,
                    product_smiles="C[C@@H](O)c1ccccc1"),
            _result("rec-other", other, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="r", conditions=COND_B, ee_target_pct=99.0,
                    conversion_pct=95.0, measurement_type="conversion",
                    measurement_value=95.0, measurement_unit="%",
                    soluble_expression=True,
                    product_smiles="C[C@@H](O)c1ccccc1"),
        ])
        return db, parent, same, other

    def test_identical_conditions_produce_a_comparison(self) -> None:
        db, parent, same, _ = self._two_variants()
        report = db.variant_vs_parent(parent)
        pair = report.require(same)
        self.assertEqual(pair.mutations, ("W110A",))
        self.assertAlmostEqual(pair.delta_measurement, 25.0)
        self.assertAlmostEqual(pair.delta_ee_pct, 48.0)
        self.assertAlmostEqual(pair.delta_conversion_pct, 25.0)
        self.assertEqual(pair.measurement_unit, "%")

    def test_a_cross_condition_comparison_is_refused_and_names_the_field(self) -> None:
        db, parent, _, other = self._two_variants()
        report = db.variant_vs_parent(parent)
        refusals = {r.variant_sha256: r for r in report.refusals}
        self.assertIn(other, refusals,
                      "a variant assayed at a different pH must not be compared")
        self.assertIn("pH", refusals[other].differing_fields)
        self.assertNotIn(other, {c.variant_sha256 for c in report.comparisons})

    def test_requiring_a_refused_comparison_raises(self) -> None:
        db, parent, _, other = self._two_variants()
        report = db.variant_vs_parent(parent)
        with self.assertRaises(ConditionMismatchError) as ctx:
            report.require(other)
        self.assertIn("pH", str(ctx.exception))

    def test_an_unexpressed_variant_is_refused_as_undetermined(self) -> None:
        db = HouseDB(":memory:")
        db.register_substrate_target(
            TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
        parent = db.upsert_candidate(sequence=_seq("p"))
        var = db.upsert_candidate(sequence=_seq("v"), origin="engineered",
                                  construct_sequence=_seq("v"))
        db.add_lineage_edge(parent_sha256=parent, variant_sha256=var,
                            mutations=["A1G"], numbering_reference="parent 1-based")
        db.create_round("r", TARGET)
        db.ingest_round([
            _result("rec-p", parent, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="r", conversion_pct=30.0, measurement_type="conversion",
                    measurement_value=30.0, measurement_unit="%",
                    soluble_expression=True),
            _result("rec-v", var, OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE,
                    round_id="r", detection={"confirms_product_identity": False}),
        ])
        report = db.variant_vs_parent(parent)
        self.assertEqual(report.comparisons, ())
        self.assertEqual(len(report.refusals), 1)
        self.assertIn("undetermined", report.refusals[0].reason)

    def test_comparison_refused_when_the_endpoint_differs(self) -> None:
        db = HouseDB(":memory:")
        db.register_substrate_target(
            TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
        parent = db.upsert_candidate(sequence=_seq("p"))
        var = db.upsert_candidate(sequence=_seq("v"), origin="engineered",
                                  construct_sequence=_seq("v"))
        db.add_lineage_edge(parent_sha256=parent, variant_sha256=var,
                            mutations=["A1G"], numbering_reference="parent 1-based")
        db.create_round("r", TARGET)
        db.ingest_round([
            _result("rec-p", parent, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="r", measurement_type="conversion",
                    measurement_value=30.0, measurement_unit="%",
                    soluble_expression=True),
            _result("rec-v", var, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="r", measurement_type="initial_rate",
                    measurement_value=1.2, measurement_unit="U/mg",
                    soluble_expression=True),
        ])
        report = db.variant_vs_parent(parent)
        self.assertEqual(report.comparisons, ())
        self.assertIn("measurement_type", report.refusals[0].differing_fields)


class TestCoverageAudit(unittest.TestCase):
    def _coverage_db(self) -> HouseDB:
        """Six records, each complete except one -- a different one each time.

        Every individual facet is satisfied by five of six records and the
        union is six, while exactly one record satisfies all six at once. The
        gap between 6 and 1 is the thing the audit exists to show.
        """
        db = HouseDB(":memory:")
        db.register_substrate_target(
            "cov-target",
            substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"rungs": [{"rung": "name", "value": "the alcohol"}]},
            needs_curation=True,
            curation_notes=["product structure not yet resolved"],
        )
        full = dict(COND_A)
        no_ph = dict(COND_A, pH=None)
        shas = {t: db.upsert_candidate(sequence=_seq(t))
                for t in ("a", "b", "c", "d", "e")}
        # A candidate known only by its hash: no sequence text anywhere.
        hashed_only = db.upsert_candidate(
            sequence_sha256=sequence_hash(_seq("f")),
            candidate_id="hash-only", needs_curation=True,
            curation_notes=["sequence text never retrieved"])
        db.create_round("cov-r", "cov-target")
        rows = [
            _result("cov-a", shas["a"], OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="cov-r", conditions=full,
                    product_smiles="C[C@@H](O)c1ccccc1",
                    measurement_type="conversion", measurement_value=61.0,
                    measurement_unit="%", soluble_expression=True),
            # missing: defined_product
            _result("cov-b", shas["b"], OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                    round_id="cov-r", conditions=full,
                    measurement_type="conversion", measurement_value=0.0,
                    measurement_unit="%", soluble_expression=True,
                    detection={"confirms_product_identity": False}),
            # missing: cofactor state
            _result("cov-c", shas["c"], OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="cov-r", conditions=full,
                    cofactor_state=CofactorState.UNKNOWN,
                    product_smiles="C[C@@H](O)c1ccccc1",
                    measurement_type="conversion", measurement_value=20.0,
                    measurement_unit="%", soluble_expression=True),
            # missing: full reaction conditions
            _result("cov-d", shas["d"], OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="cov-r", conditions=no_ph,
                    product_smiles="C[C@@H](O)c1ccccc1",
                    measurement_type="conversion", measurement_value=15.0,
                    measurement_unit="%", soluble_expression=True),
            # missing: quantitative result
            _result("cov-e", shas["e"], OutcomeClass.NOT_TESTED,
                    round_id="cov-r", conditions=full,
                    product_smiles="C[C@@H](O)c1ccccc1",
                    detection={"confirms_product_identity": False,
                               "limit_of_detection": None}),
            # missing: defined sequence
            _result("cov-f", hashed_only, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="cov-r", conditions=full,
                    product_smiles="C[C@@H](O)c1ccccc1",
                    measurement_type="conversion", measurement_value=44.0,
                    measurement_unit="%", soluble_expression=True),
        ]
        db.ingest_round(rows)
        return db

    def test_audit_reports_the_intersection_not_the_union(self) -> None:
        db = self._coverage_db()
        audit = db.coverage_audit("cov-target")
        self.assertEqual(audit.n_records, 6)
        self.assertEqual(audit.n_union, 6)
        self.assertEqual(audit.n_complete, 1)
        self.assertEqual(audit.complete_record_ids, ("cov-a",))
        self.assertAlmostEqual(audit.completeness_fraction, 1 / 6)
        self.assertLess(audit.n_complete, audit.n_union)
        self.assertLess(audit.n_complete, min(audit.facet_counts.values()))

    def test_each_facet_alone_looks_much_healthier_than_the_intersection(self) -> None:
        db = self._coverage_db()
        audit = db.coverage_audit("cov-target")
        self.assertEqual(audit.facet_counts["defined_sequence"], 5)
        self.assertEqual(audit.facet_counts["defined_substrate_structure"], 6)
        self.assertEqual(audit.facet_counts["cofactor_identity_and_state"], 5)
        self.assertEqual(audit.facet_counts["defined_product"], 5)
        self.assertEqual(audit.facet_counts["full_reaction_conditions"], 5)
        self.assertEqual(audit.facet_counts["quantitative_result"], 5)
        self.assertEqual(sum(audit.missing_counts.values()), 5)
        self.assertIn("intersection", audit.as_dict()["note"])

    def test_a_negative_bounded_by_a_detection_limit_counts_as_quantitative(
            self) -> None:
        """A negative with an LOD bounds the value, which is what makes it usable."""
        db = HouseDB(":memory:")
        db.register_substrate_target(
            "neg-target",
            substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
        sha = db.upsert_candidate(sequence=_seq("n"))
        db.create_round("neg-r", "neg-target")
        db.ingest_round([
            _result("neg-1", sha, OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                    round_id="neg-r", conditions=dict(COND_A),
                    soluble_expression=True,
                    detection={"confirms_product_identity": False,
                               "limit_of_detection": 0.02})])
        row = db.batch_outcomes("neg-r").rows[0]
        self.assertIsNone(row.quantitative_value())
        audit = db.coverage_audit("neg-target")
        self.assertEqual(audit.facet_counts["quantitative_result"], 1)
        self.assertEqual(audit.n_complete, 1)

    def test_each_missing_facet_is_attributed_to_the_right_record(self) -> None:
        db = self._coverage_db()
        audit = db.coverage_audit("cov-target")
        self.assertEqual(audit.missing_counts["quantitative_result"], 1)
        self.assertEqual(audit.missing_counts["defined_product"], 1)
        self.assertEqual(audit.missing_counts["defined_sequence"], 1)
        self.assertEqual(audit.missing_counts["defined_substrate_structure"], 0)


class TestPerformanceAxes(unittest.TestCase):
    def _axes_db(self) -> tuple[HouseDB, str]:
        db = HouseDB(":memory:")
        db.register_substrate_target(
            TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
        sha = db.upsert_candidate(sequence=_seq("v"), origin="engineered",
                                  construct_sequence=_seq("v"))
        db.record_performance_axis(
            sequence_sha256=sha, axis=PerformanceAxis.SUBSTRATE_FIT,
            direction=EffectDirection.IMPROVE, value=-1.4, unit="kcal/mol (docking)",
            basis="pocket widened; more productive poses")
        db.record_performance_axis(
            sequence_sha256=sha, axis=PerformanceAxis.CATALYTIC_FUNCTION,
            direction=EffectDirection.DEGRADE, value=0.4, unit="relative kcat",
            basis="hydride distance lengthened")
        db.record_performance_axis(
            sequence_sha256=sha, axis=PerformanceAxis.STABILITY_EXPRESSION_RISK,
            direction=EffectDirection.DEGRADE, value=0.3,
            unit="relative soluble yield", basis="buried polar substitution")
        return db, sha

    def test_the_three_axes_are_stored_separately(self) -> None:
        db, sha = self._axes_db()
        readout = db.performance_axes(sha)
        self.assertIsInstance(readout, AxisReadout)
        self.assertEqual(readout.axes_covered, 3)
        by_axis = readout.by_axis()
        self.assertEqual(set(by_axis), {a.value for a in PerformanceAxis})
        self.assertEqual(len(by_axis["substrate_fit"]), 1)
        self.assertEqual(
            readout.directions(AxisRecordKind.OBSERVED),
            {"substrate_fit": "improve",
             "catalytic_function": "degrade",
             "stability_expression_risk": "degrade"},
        )
        self.assertEqual(len(readout.observations), 3)
        self.assertEqual({o.unit for o in readout.observations},
                         {"kcal/mol (docking)", "relative kcat",
                          "relative soluble yield"})
        # the axes carry different units, which is why no sum of them exists
        self.assertEqual(len({o.unit for o in readout.observations}), 3)

    def test_no_combined_score_column_or_view_exists_anywhere(self) -> None:
        db, _ = self._axes_db()
        for table in db.table_names():
            for col in db.column_names(table):
                low = col.lower()
                self.assertNotIn(low, FORBIDDEN_COMBINED_SCORE_NAMES,
                                 f"{table}.{col} collapses the axes")
                self.assertFalse(low.endswith("_score"), f"{table}.{col}")
                self.assertNotIn("quality", low, f"{table}.{col}")
        views = db._all("SELECT name FROM sqlite_master WHERE type = 'view'")
        self.assertEqual([v["name"] for v in views], [],
                         "a view could reintroduce a combined score")

    def test_no_public_helper_offers_a_combined_score(self) -> None:
        banned = ("combined", "composite", "aggregate", "overall", "quality",
                  "total", "_score")
        for cls in (HouseDB, AxisReadout):
            for name in dir(cls):
                if name.startswith("_"):
                    continue
                low = name.lower()
                for token in banned:
                    self.assertNotIn(token, low,
                                     f"{cls.__name__}.{name} looks like a "
                                     f"collapsed score")
        self.assertIn("no column", THERE_IS_NO_COMBINED_MUTATION_SCORE)

    def test_a_payload_carrying_a_combined_score_is_refused(self) -> None:
        db, sha = self._axes_db()
        with self.assertRaises(CollapsedScoreError):
            db.record_performance_axis(
                sequence_sha256=sha, axis=PerformanceAxis.SUBSTRATE_FIT,
                direction=EffectDirection.IMPROVE, basis="overall_score")
        with self.assertRaises(CollapsedScoreError):
            PredictionEntry("sha256:x", 1, "diversity", "reason", "uncertainty",
                            scorecard={"docking_result": -8.0,
                                       "overall_score": 0.91})


class TestDeletionGuard(unittest.TestCase):
    def test_a_negative_record_survives_an_attempted_delete(self) -> None:
        db, _ = _fixture_db()
        negative = "rec-c2"
        with self.assertRaises(DeletionRefusedError) as ctx:
            db.delete_record(negative)
        self.assertIn("deprecate_record", str(ctx.exception))
        rows = {r.record_id for r in db.batch_outcomes("round-1").rows}
        self.assertIn(negative, rows)
        self.assertEqual(db.batch_outcomes("round-1").n_submitted, 6)

    def test_an_expression_failure_also_survives(self) -> None:
        db, _ = _fixture_db()
        with self.assertRaises(DeletionRefusedError):
            db.delete_record("rec-c4")
        self.assertIn("rec-c4",
                      {r.record_id for r in db.batch_outcomes("round-1").rows})

    def test_raw_sql_delete_is_refused_by_the_database_itself(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "house.sqlite")
            db = HouseDB(path)
            db.register_substrate_target(
                TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
                product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
            sha = db.upsert_candidate(sequence=_seq("n"))
            db.create_round("r", TARGET)
            db.ingest_round([_result("rec-neg", sha,
                                     OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                                     round_id="r", soluble_expression=True,
                                     detection={"confirms_product_identity": False})])
            raw = sqlite3.connect(path)
            with self.assertRaises(sqlite3.IntegrityError):
                raw.execute("DELETE FROM experiment_record WHERE record_id = ?",
                            ("rec-neg",))
                raw.commit()
            raw.rollback()
            raw.close()
            self.assertEqual(db.batch_outcomes("r").n_submitted, 1)
            db.close()

    def test_deprecation_keeps_the_row_and_records_the_reason(self) -> None:
        db, _ = _fixture_db()
        db.deprecate_record("rec-c2", "chiral column was out of calibration")
        rows = {r.record_id: r for r in db.batch_outcomes("round-1").rows}
        self.assertTrue(rows["rec-c2"].deprecated)
        self.assertEqual(rows["rec-c2"].deprecation_reason,
                         "chiral column was out of calibration")
        self.assertEqual(db.batch_outcomes("round-1").n_deprecated, 1)
        kept = db.batch_outcomes("round-1", include_deprecated=False)
        self.assertEqual(kept.n_submitted, 5)
        # the row is still in the database, only filtered out of that one view
        self.assertEqual(db.batch_outcomes("round-1").n_submitted, 6)

    def test_deprecation_without_a_reason_is_refused(self) -> None:
        db, _ = _fixture_db()
        with self.assertRaises(UnresolvedFieldError):
            db.deprecate_record("rec-c2", "   ")


class TestPredictionVsOutcome(unittest.TestCase):
    def test_the_report_measures_the_frozen_ranking(self) -> None:
        db, _ = _fixture_db()
        report = db.prediction_vs_outcome("round-1")
        self.assertTrue(report.interpretable)
        self.assertIsNone(report.non_interpretable_reason)
        self.assertEqual(report.n_predicted, 6)
        self.assertEqual(report.n_submitted, 6)
        self.assertEqual(report.n_informative, 3)
        self.assertEqual(report.n_hits, 1)
        self.assertEqual(report.n_excluded_uninformative, 3)
        self.assertAlmostEqual(report.rate_over_informative, 1 / 3)
        self.assertAlmostEqual(report.rate_over_submitted, 1 / 6)
        # ranks 1,2,3 over informative rows are c3, c1, c2; c1 is the only hit
        self.assertEqual(report.rank_of_first_hit, 2)
        top2 = report.top_k(2)
        self.assertEqual((top2.n_considered, top2.n_hits), (2, 1))
        self.assertAlmostEqual(top2.rate, 0.5)
        self.assertAlmostEqual(report.enrichment_vs_baseline(2).ratio, 1.5)

    def test_stereo_calls_are_scored_against_the_sign_of_the_measured_ee(self) -> None:
        db, _ = _fixture_db()
        counts = db.prediction_vs_outcome("round-1").stereo_agreement_counts()
        self.assertEqual(counts["agree"], 1)      # c1: favors_target, ee +92
        self.assertEqual(counts["disagree"], 1)   # c3: favors_target, ee -95
        self.assertEqual(counts["undetermined"], 4)

    def test_a_round_whose_predictions_came_after_the_results_is_not_interpretable(
            self) -> None:
        db = HouseDB(":memory:")
        db.register_substrate_target(
            TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
        sha = db.upsert_candidate(sequence=_seq("w"))
        db.create_round("r", TARGET)
        db.ingest_round([_result("rec", sha, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                                 round_id="r", conversion_pct=50.0,
                                 measurement_type="conversion",
                                 measurement_value=50.0, measurement_unit="%",
                                 soluble_expression=True)])
        report = db.prediction_vs_outcome("r")
        self.assertFalse(report.interpretable)
        self.assertIn("no predictions", report.non_interpretable_reason)
        self.assertFalse(report.frozen_before_results)

    def test_constructs_submitted_without_a_prediction_are_counted(self) -> None:
        db, sha = _fixture_db()
        db.create_round("round-2", TARGET, round_number=2)
        extra = db.upsert_candidate(sequence=_seq("x"))
        db.freeze_predictions("round-2", [
            PredictionEntry(sha["c1"], 1, "high_evidence", "best in round 1",
                            "single replicate"),
        ], snapshot_id="snapshot-round-2")
        db.ingest_round([
            _result("r2-c1", sha["c1"], OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                    round_id="round-2", soluble_expression=True,
                    detection={"confirms_product_identity": False}),
            _result("r2-x", extra, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="round-2", conversion_pct=80.0,
                    measurement_type="conversion", measurement_value=80.0,
                    measurement_unit="%", soluble_expression=True),
        ])
        report = db.prediction_vs_outcome("round-2")
        self.assertEqual(report.n_submitted_without_prediction, 1)
        self.assertEqual(report.n_predicted_not_submitted, 0)
        # the unpredicted construct is the only hit, so the top-1 is empty of hits
        self.assertEqual(report.top_k(1).n_hits, 0)


class TestPredictionRateDenominators(unittest.TestCase):
    """Regression guard: the report must not expose one rate on its own.

    ``PredictionOutcomeReport`` used to publish ``baseline_rate``, a single
    hits-over-informative number -- the narrow denominator ``HitRate`` was
    built to stop anyone quoting alone, reintroduced one class further down
    the module. Both rates are now emitted together, and the enrichment says
    which one it divided by.
    """

    def test_there_is_no_single_baseline_rate_to_quote(self) -> None:
        db, _ = _fixture_db()
        report = db.prediction_vs_outcome("round-1")
        for forbidden in ("baseline_rate", "hit_rate", "rate", "value"):
            self.assertFalse(
                hasattr(report, forbidden),
                f"PredictionOutcomeReport must not expose a single {forbidden!r}")
        self.assertFalse(hasattr(PredictionOutcomeReport, "baseline_rate"))

    def test_both_rates_appear_together_and_differ(self) -> None:
        db, _ = _fixture_db()
        report = db.prediction_vs_outcome("round-1")
        self.assertAlmostEqual(report.rate_over_informative, 1 / 3)
        self.assertAlmostEqual(report.rate_over_submitted, 1 / 6)
        self.assertNotAlmostEqual(report.rate_over_informative,
                                  report.rate_over_submitted)
        d = report.as_dict()
        self.assertNotIn("baseline_rate", d)
        self.assertIn("rate_over_informative", d)
        self.assertIn("rate_over_submitted", d)
        self.assertTrue(d["denominators_differ"])

    def test_both_rates_are_undefined_rather_than_zero_on_an_empty_round(
            self) -> None:
        db = HouseDB(":memory:")
        db.register_substrate_target(
            TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
        db.create_round("empty", TARGET)
        report = db.prediction_vs_outcome("empty")
        self.assertIsNone(report.rate_over_informative)
        self.assertIsNone(report.rate_over_submitted)

    def test_the_enrichment_names_the_denominator_it_divided_by(self) -> None:
        db, _ = _fixture_db()
        report = db.prediction_vs_outcome("round-1")
        enr = report.enrichment_vs_baseline(2)
        self.assertIsInstance(enr, EnrichmentResult)
        self.assertEqual(enr.baseline_denominator, "informative")
        self.assertEqual(enr.n_baseline_denominator, report.n_informative)
        self.assertAlmostEqual(enr.baseline_rate, report.rate_over_informative)
        self.assertAlmostEqual(enr.top_k_rate, 0.5)
        self.assertAlmostEqual(enr.ratio, 1.5)
        self.assertIsNone(enr.undefined_reason)
        self.assertIn("informative", enr.describe())
        self.assertEqual(enr.as_dict()["baseline_denominator"], "informative")
        # the other denominator would have given a different, larger number
        self.assertNotAlmostEqual(
            enr.ratio, enr.top_k_rate / report.rate_over_submitted)

    def test_an_enrichment_with_no_hits_is_undefined_with_a_reason(self) -> None:
        db = HouseDB(":memory:")
        db.register_substrate_target(
            TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
        a = db.upsert_candidate(sequence=_seq("a"))
        b = db.upsert_candidate(sequence=_seq("b"))
        db.create_round("r", TARGET)
        db.freeze_predictions("r", [
            PredictionEntry(a, 1, "diversity", "closest homolog", "no structure"),
            PredictionEntry(b, 2, "diversity", "second clade", "no structure"),
        ], snapshot_id="snap")
        db.ingest_round([
            _result("rec-a", a, OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                    round_id="r", soluble_expression=True,
                    detection={"confirms_product_identity": False}),
            _result("rec-b", b, OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                    round_id="r", soluble_expression=True,
                    detection={"confirms_product_identity": False}),
        ])
        enr = db.prediction_vs_outcome("r").enrichment_vs_baseline(2)
        self.assertIsNone(enr.ratio)
        self.assertIn("no hits", enr.undefined_reason)
        self.assertEqual(enr.baseline_denominator, "informative")
        self.assertIn("undefined", enr.describe())


class TestIngestGuards(unittest.TestCase):
    def setUp(self) -> None:
        self.db = HouseDB(":memory:")
        self.db.register_substrate_target(
            TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
        self.sha = self.db.upsert_candidate(sequence=_seq("g"))
        self.db.create_round("r", TARGET)

    def test_a_positive_without_product_confirmation_is_refused(self) -> None:
        with self.assertRaises(FabricationGuardError):
            self.db.ingest_round([
                _result("bad", self.sha, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                        round_id="r", conversion_pct=50.0,
                        detection={"confirms_product_identity": False})])

    def test_a_negative_without_a_detection_limit_is_refused(self) -> None:
        with self.assertRaises(UnresolvedFieldError):
            self.db.ingest_round([
                _result("bad", self.sha, OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                        round_id="r",
                        detection={"confirms_product_identity": False,
                                   "limit_of_detection": None})])

    def test_an_expression_failure_carrying_a_measurement_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.db.ingest_round([
                _result("bad", self.sha,
                        OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE,
                        round_id="r", conversion_pct=12.0,
                        detection={"confirms_product_identity": False})])

    def test_an_unregistered_candidate_is_refused_rather_than_stubbed(self) -> None:
        with self.assertRaises(UnknownRecordError):
            self.db.ingest_round([
                _result("bad", "sha256:never-seen",
                        OutcomeClass.NOT_TESTED, round_id="r",
                        detection={"confirms_product_identity": False,
                                   "limit_of_detection": None})])

    def test_an_unknown_cofactor_state_becomes_a_curation_note_not_a_guess(self) -> None:
        report = self.db.ingest_round([
            _result("rec", self.sha, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="r", cofactor_state=CofactorState.UNKNOWN,
                    conversion_pct=50.0, measurement_type="conversion",
                    measurement_value=50.0, measurement_unit="%",
                    soluble_expression=True)])
        self.assertTrue(report.needs_curation)
        self.assertTrue(any("oxidation state unknown" in n
                            for n in report.curation_notes))
        row = self.db.batch_outcomes("r").rows[0]
        self.assertIs(row.cofactor_state, CofactorState.UNKNOWN)
        self.assertTrue(row.needs_curation)

    def test_a_pydantic_experiment_record_can_be_ingested(self) -> None:
        rec = ExperimentRecord(
            record_id="pyd-1",
            sequence=_seq("g"),
            substrate=SubstrateSpec(name="acetophenone",
                                    isomeric_smiles="CC(=O)c1ccccc1"),
            product_observed=ProductSpec(name="(R)-1-phenylethanol",
                                         isomeric_smiles="C[C@@H](O)c1ccccc1"),
            reaction_direction=ReactionDirection.FORWARD_AS_TARGET,
            cofactor=CofactorSpec(name="NADPH", state=CofactorState.REDUCED),
            conditions=Conditions(pH=7.0, temperature_C=30.0, buffer="KPi",
                                  substrate_concentration_mM=10.0,
                                  reaction_time_h=24.0),
            outcome=OutcomeClass.CONFIRMED_TARGET_PRODUCT,
            detection=Detection(method="chiral GC-MS",
                                confirms_product_identity=True,
                                chiral_method_validated=True,
                                limit_of_detection=0.05, limit_unit="mM"),
            ee_target_pct=97.0, conversion_pct=88.0,
            measurement_type="conversion", measurement_value=88.0,
            measurement_unit="%", soluble_expression=True,
            evidence=[EvidenceRef(source_type="internal_experiment",
                                  identifier="run-2026-01",
                                  strength=EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL,
                                  experiment_activity_id="act-1")],
        )
        report = self.db.ingest_round([rec], round_id="r")
        self.assertEqual(report.n_written, 1)
        row = self.db.batch_outcomes("r").rows[0]
        self.assertEqual(row.product_inchikey, None)
        self.assertEqual(row.product_smiles, "C[C@@H](O)c1ccccc1")
        self.assertAlmostEqual(row.ee_target_pct, 97.0)
        self.assertEqual(row.conditions["pH"], 7.0)
        evidence = self.db.evidence_for(record_id="pyd-1")
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["experiment_activity_id"], "act-1")
        self.assertEqual(evidence[0]["upstream_sources"], [])

    def test_re_ingesting_an_unchanged_row_is_allowed(self) -> None:
        row = _result("rec", self.sha, OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                      round_id="r", soluble_expression=True,
                      detection={"confirms_product_identity": False})
        self.db.ingest_round([row])
        self.db.ingest_round([dict(row)])
        self.assertEqual(self.db.batch_outcomes("r").n_submitted, 1)

    def test_a_negative_cannot_be_rewritten_into_a_positive(self) -> None:
        """Overwriting a record id is the forbidden delete, done without a delete."""
        self.db.ingest_round([
            _result("rec", self.sha, OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                    round_id="r", soluble_expression=True,
                    detection={"confirms_product_identity": False})])
        with self.assertRaises(RecordOverwriteError) as ctx:
            self.db.ingest_round([
                _result("rec", self.sha, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                        round_id="r", soluble_expression=True,
                        conversion_pct=70.0, measurement_type="conversion",
                        measurement_value=70.0, measurement_unit="%")])
        self.assertIn("new record_id", str(ctx.exception))
        row = self.db.batch_outcomes("r").rows[0]
        self.assertIs(row.outcome, OutcomeClass.NO_TARGET_PRODUCT_DETECTED)

    def test_deprecating_a_record_does_not_unlock_an_in_place_rewrite(self) -> None:
        """Regression: the deprecated escape destroyed the original reading.

        ``_assert_result_not_silently_rewritten`` used to return early for a
        deprecated row, and the ``ON CONFLICT(record_id) DO UPDATE`` behind it
        then rewrote every result column with no audit row and no log entry.
        Deprecate-then-reingest was therefore the delete-and-replace the
        module forbids, reachable through the public API. The reading must
        still be retrievable afterwards.
        """
        self.db.ingest_round([
            _result("rec", self.sha, OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                    round_id="r", soluble_expression=True,
                    measurement_type="conversion", measurement_value=0.0,
                    measurement_unit="%", conversion_pct=0.0,
                    detection={"confirms_product_identity": False})])
        self.db.deprecate_record("rec", "wrong plate read")
        with self.assertRaises(RecordOverwriteError) as ctx:
            self.db.ingest_round([
                _result("rec", self.sha, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                        round_id="r", soluble_expression=True,
                        conversion_pct=70.0, ee_target_pct=95.0,
                        measurement_type="conversion", measurement_value=70.0,
                        measurement_unit="%")])
        self.assertIn("new record_id", str(ctx.exception))
        self.assertIn("deprecat", str(ctx.exception).lower())

        # the original negative reading survives, numbers and all
        rows = {r.record_id: r for r in self.db.batch_outcomes("r").rows}
        self.assertEqual(set(rows), {"rec"})
        kept = rows["rec"]
        self.assertIs(kept.outcome, OutcomeClass.NO_TARGET_PRODUCT_DETECTED)
        self.assertAlmostEqual(kept.conversion_pct, 0.0)
        self.assertAlmostEqual(kept.measurement_value, 0.0)
        self.assertIsNone(kept.ee_target_pct)
        self.assertTrue(kept.deprecated)
        self.assertEqual(kept.deprecation_reason, "wrong plate read")

    def test_every_result_column_of_a_deprecated_row_is_protected(self) -> None:
        """Not only the outcome: kcat, Km and ee are rewritable columns too."""
        self.db.ingest_round([
            _result("kin", self.sha, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="r", soluble_expression=True, ee_target_pct=80.0,
                    kcat_s=2.5, km_mM=0.4,
                    measurement_type="kcat", measurement_value=2.5,
                    measurement_unit="1/s",
                    product_smiles="C[C@@H](O)c1ccccc1")])
        self.db.deprecate_record("kin", "cuvette path length wrong")
        unchanged: dict[str, object] = {
            "round_id": "r", "soluble_expression": True, "ee_target_pct": 80.0,
            "kcat_s": 2.5, "km_mM": 0.4, "measurement_type": "kcat",
            "measurement_value": 2.5, "measurement_unit": "1/s",
            "product_smiles": "C[C@@H](O)c1ccccc1",
        }
        for change in ({"kcat_s": 9.9}, {"km_mM": 0.01}, {"ee_target_pct": 99.0}):
            with self.assertRaises(RecordOverwriteError):
                self.db.ingest_round([
                    _result("kin", self.sha,
                            OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                            **{**unchanged, **change})])
        row = self.db.batch_outcomes("r").rows[0]
        self.assertAlmostEqual(row.kcat_s, 2.5)
        self.assertAlmostEqual(row.km_mM, 0.4)
        self.assertAlmostEqual(row.ee_target_pct, 80.0)

    def test_the_correction_goes_under_a_new_id_beside_the_original(self) -> None:
        """The route the refusal names actually works, and keeps both rows."""
        self.db.ingest_round([
            _result("rec", self.sha, OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                    round_id="r", soluble_expression=True,
                    detection={"confirms_product_identity": False})])
        self.db.deprecate_record("rec", "wrong plate read")
        self.db.ingest_round([
            _result("rec-v2", self.sha, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="r", soluble_expression=True, conversion_pct=70.0,
                    measurement_type="conversion", measurement_value=70.0,
                    measurement_unit="%",
                    product_smiles="C[C@@H](O)c1ccccc1")])
        rows = {r.record_id: r for r in self.db.batch_outcomes("r").rows}
        self.assertEqual(set(rows), {"rec", "rec-v2"})
        self.assertIs(rows["rec"].outcome, OutcomeClass.NO_TARGET_PRODUCT_DETECTED)
        self.assertTrue(rows["rec"].deprecated)
        self.assertIs(rows["rec-v2"].outcome, OutcomeClass.CONFIRMED_TARGET_PRODUCT)
        self.assertFalse(rows["rec-v2"].deprecated)

    def test_re_ingesting_an_unchanged_deprecated_row_is_still_allowed(self) -> None:
        """A replayed batch must not fail merely because one row was retired."""
        row = _result("rec", self.sha, OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                      round_id="r", soluble_expression=True,
                      detection={"confirms_product_identity": False})
        self.db.ingest_round([row])
        self.db.deprecate_record("rec", "wrong plate read")
        self.db.ingest_round([dict(row)])
        kept = self.db.batch_outcomes("r").rows[0]
        self.assertTrue(kept.deprecated)
        self.assertEqual(kept.deprecation_reason, "wrong plate read")

    def test_one_ingest_writes_one_round(self) -> None:
        self.db.create_round("r2", TARGET, round_number=2)
        with self.assertRaises(ValueError):
            self.db.ingest_round([
                _result("a", self.sha, OutcomeClass.NOT_TESTED, round_id="r",
                        detection={"confirms_product_identity": False,
                                   "limit_of_detection": None}),
                _result("b", self.sha, OutcomeClass.NOT_TESTED, round_id="r2",
                        detection={"confirms_product_identity": False,
                                   "limit_of_detection": None}),
            ])


class TestRegistrationGuards(unittest.TestCase):
    def test_a_substrate_target_without_a_structure_needs_curation(self) -> None:
        db = HouseDB(":memory:")
        with self.assertRaises(UnresolvedFieldError):
            db.register_substrate_target("vague", label="some ketone")
        db.register_substrate_target(
            "vague", label="some ketone", needs_curation=True,
            curation_notes=["operator has not named the ketone yet"])
        row = db.substrate_target("vague")
        self.assertTrue(row["needs_curation"])
        self.assertIsNone(row["substrate_ladder"])
        self.assertTrue(any("structural representation" in n
                            for n in row["curation_notes"]))

    def test_a_sequence_that_does_not_match_its_hash_is_refused(self) -> None:
        db = HouseDB(":memory:")
        with self.assertRaises(FabricationGuardError):
            db.upsert_candidate(sequence=_seq("a"),
                                sequence_sha256=sequence_hash(_seq("b")))

    def test_accessions_are_versioned_secondary_attributes(self) -> None:
        db = HouseDB(":memory:")
        sha = db.upsert_candidate(
            sequence=_seq("a"),
            accessions=[{"database": "uniprot", "accession": "Q00000",
                         "version": "2"},
                        {"database": "uniprot", "accession": "Q00000",
                         "version": "3"}])
        accs = sorted(db.candidate(sha)["accessions"],
                      key=lambda a: a["version"])
        self.assertEqual([a["version"] for a in accs], ["2", "3"])
        self.assertNotIn("accession", db.column_names("candidate"))

    def test_upserting_never_erases_a_known_value_with_none(self) -> None:
        db = HouseDB(":memory:")
        sha = db.upsert_candidate(sequence=_seq("a"), organism="Thermus sp.",
                                  family="SDR")
        db.upsert_candidate(sequence_sha256=sha, cluster_id="clu-7")
        row = db.candidate(sha)
        self.assertEqual(row["organism"], "Thermus sp.")
        self.assertEqual(row["family"], "SDR")
        self.assertEqual(row["cluster_id"], "clu-7")

    def test_lineage_edges_refuse_cycles_and_self_edges(self) -> None:
        db = HouseDB(":memory:")
        a = db.upsert_candidate(sequence=_seq("a"))
        b = db.upsert_candidate(sequence=_seq("b"))
        db.add_lineage_edge(parent_sha256=a, variant_sha256=b, mutations=["A1G"],
                            numbering_reference="parent 1-based")
        with self.assertRaises(ValueError):
            db.add_lineage_edge(parent_sha256=a, variant_sha256=a,
                                mutations=["A1G"], numbering_reference="x")
        with self.assertRaises(ValueError):
            db.add_lineage_edge(parent_sha256=b, variant_sha256=a,
                                mutations=["G1A"], numbering_reference="x")
        with self.assertRaises(UnresolvedFieldError):
            db.add_lineage_edge(parent_sha256=a, variant_sha256=b, mutations=[],
                                numbering_reference="x")
        self.assertEqual(db.descendants(a), {b})
        self.assertEqual(db.ancestors(b), [a])


class TestExportAndEvidence(unittest.TestCase):
    def test_the_export_is_labelled_as_not_certified_enzymeml(self) -> None:
        db, _ = _fixture_db()
        doc = db.export_enzymeml_like("round-1")
        self.assertEqual(doc["format"], "enzymeml_like_export")
        self.assertFalse(doc["is_certified_enzymeml"])
        self.assertIsNone(doc["enzymeml_version"])
        self.assertTrue(doc["needs_curation"])
        self.assertTrue(any("EnzymeML version" in n for n in doc["curation_notes"]))
        self.assertIn("NOT been validated", doc["disclaimer"])

    def test_the_export_carries_the_failures_too(self) -> None:
        db, _ = _fixture_db()
        doc = db.export_enzymeml_like("round-1")
        self.assertEqual(len(doc["measurements"]), 6)
        self.assertTrue(doc["batch_completeness"]["includes_expression_failures"])
        self.assertTrue(doc["batch_completeness"]["includes_untested"])
        excluded = [m for m in doc["measurements"] if m["excluded_from_kinetics"]]
        self.assertEqual(len(excluded), 3)
        self.assertEqual(len(doc["proteins"]), 6)

    def test_the_export_does_not_invent_an_ec_number_or_a_vessel(self) -> None:
        db, _ = _fixture_db()
        doc = db.export_enzymeml_like("round-1")
        self.assertEqual(doc["vessels"], [])
        self.assertTrue(all(p["ecnumber"] is None for p in doc["proteins"]))
        self.assertIsNone(doc["reactions"][0]["reversible"])

    def test_independent_evidence_discounts_re_curated_copies(self) -> None:
        db = HouseDB(":memory:")
        db.register_substrate_target(
            TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
        sha = db.upsert_candidate(sequence=_seq("a"))
        other = db.upsert_candidate(sequence=_seq("b"))
        db.create_round("r", TARGET)
        shared = [EvidenceRef(source_type="database", identifier="BRENDA:1",
                              source_doi="10.1000/one",
                              experiment_activity_id="act-1")]
        copy = [EvidenceRef(source_type="database", identifier="SKiD:9",
                            source_doi="10.1000/one",
                            upstream_sources=["brenda"])]
        db.ingest_round([
            _result("rec-1", sha, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="r", conversion_pct=50.0,
                    measurement_type="conversion", measurement_value=50.0,
                    measurement_unit="%", soluble_expression=True,
                    evidence=shared),
            _result("rec-2", other, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="r", conversion_pct=50.0,
                    measurement_type="conversion", measurement_value=50.0,
                    measurement_unit="%", soluble_expression=True,
                    evidence=copy),
        ])
        summary = db.independent_evidence(round_id="r")
        self.assertEqual(summary.unresolved_record_ids, ())
        self.assertTrue(summary.complete)
        self.assertEqual(summary.n_rows, 2)
        self.assertEqual(summary.n_resolved, 2)
        if summary.available:
            self.assertEqual(summary.n_independent, 1)
            self.assertEqual(summary.n_discounted, 1)
            self.assertEqual(len(summary.groups), 1)
            self.assertEqual(len(summary.groups[0]["record_ids"]), 2)
        else:  # pragma: no cover - only when the lineage module is absent
            self.assertIsNone(summary.n_independent)


class TestEvidenceShimsCarryLineage(unittest.TestCase):
    """Regression guard for the shims handed to :mod:`eagent.datalayer.lineage`.

    ``parent_sequence_sha256`` was hardcoded to ``None`` even though
    ``add_lineage_edge`` had already stored the edge, so a variant and its
    parent never shared a lineage facet and an engineering series was split
    into unrelated rows. Separately, a requested ``record_id`` that matched no
    row was skipped while ``n_rows`` was taken from the shims that were found,
    so a partially missing group reported itself as complete.
    """

    def _lineage_db(self) -> tuple[HouseDB, str, str]:
        db = HouseDB(":memory:")
        db.register_substrate_target(
            TARGET, substrate_ladder={"isomeric_smiles": "CC(=O)c1ccccc1"},
            product_ladder={"isomeric_smiles": "C[C@@H](O)c1ccccc1"})
        parent = db.upsert_candidate(sequence=_seq("p"), candidate_id="parent")
        variant = db.upsert_candidate(sequence=_seq("v"), candidate_id="variant",
                                      origin="engineered",
                                      construct_sequence=_seq("v"))
        db.add_lineage_edge(parent_sha256=parent, variant_sha256=variant,
                            mutations=["W110A"],
                            numbering_reference="parent sequence 1-based")
        db.create_round("r", TARGET)
        db.ingest_round([
            _result("rec-p", parent, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="r", conversion_pct=30.0,
                    measurement_type="conversion", measurement_value=30.0,
                    measurement_unit="%", soluble_expression=True,
                    product_smiles="C[C@@H](O)c1ccccc1"),
            _result("rec-v", variant, OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                    round_id="r", conversion_pct=70.0,
                    measurement_type="conversion", measurement_value=70.0,
                    measurement_unit="%", soluble_expression=True,
                    product_smiles="C[C@@H](O)c1ccccc1"),
        ])
        return db, parent, variant

    def test_the_shim_reads_the_parent_from_the_lineage_table(self) -> None:
        db, parent, _ = self._lineage_db()
        shims, unresolved = db._evidence_shims(["rec-p", "rec-v"])
        self.assertEqual(unresolved, ())
        by_id = {s.record_id: s for s in shims}
        self.assertEqual(by_id["rec-v"].parent_sequence_sha256, parent)
        self.assertIsNone(by_id["rec-p"].parent_sequence_sha256,
                          "a row with no incoming edge has no parent to claim")

    def test_a_variant_and_its_parent_stay_in_one_split_group(self) -> None:
        """The consequence of the dropped edge: a leaking train/test split."""
        from eagent.datalayer.lineage import leakage_safe_groups
        db, _, _ = self._lineage_db()
        shims, _unresolved = db._evidence_shims(["rec-p", "rec-v"])
        groups = leakage_safe_groups(shims)
        self.assertEqual(groups["rec-p"], groups["rec-v"],
                         "a variant and its parent must not straddle a split")

    def test_an_unknown_record_id_is_returned_not_dropped(self) -> None:
        db, _, _ = self._lineage_db()
        summary = db.independent_evidence(
            record_ids=["rec-p", "rec-missing"], persist=False)
        self.assertEqual(summary.n_rows, 2, "n_rows counts what was asked about")
        self.assertEqual(summary.unresolved_record_ids, ("rec-missing",))
        self.assertFalse(summary.complete)
        self.assertEqual(summary.n_resolved, 1)
        if summary.available:
            self.assertEqual(summary.n_independent, 1)
            # the missing id is a curation gap, not a discounted duplicate
            self.assertEqual(summary.n_discounted, 0)
        else:  # pragma: no cover - only when the lineage module is absent
            self.assertIsNone(summary.n_independent)
            self.assertIsNone(summary.n_discounted)

    def test_the_shim_helper_reports_the_missing_id_too(self) -> None:
        db, _, _ = self._lineage_db()
        shims, unresolved = db._evidence_shims(["rec-p", "nope", "rec-v"])
        self.assertEqual(unresolved, ("nope",))
        self.assertEqual(len(shims), 2)


class TestConditionKey(unittest.TestCase):
    def test_the_cofactor_is_part_of_the_comparability_key(self) -> None:
        a = condition_key(COND_A, "NADPH", "reduced")
        b = condition_key(COND_A, "NADH", "reduced")
        c = condition_key(COND_A, "NADPH", "oxidized")
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(a, condition_key(dict(COND_A), "NADPH", "reduced"))

    def test_fields_outside_the_key_do_not_change_it(self) -> None:
        a = condition_key(COND_A, "NADPH", "reduced")
        b = condition_key(dict(COND_A, operator="someone else"),
                          "NADPH", "reduced")
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
