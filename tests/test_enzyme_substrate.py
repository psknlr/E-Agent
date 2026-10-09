"""Tests for the enzyme -> target model and its calibration.

The synthetic problems plant a signal the model should find (a motif that
decides the label) or deliberately plant none (so any confidence it shows is
over-confidence). Both are needed: a calibration test that only uses learnable
data cannot tell a calibrated model from a lucky one.
"""

from __future__ import annotations

import math
import random
import unittest

from eagent.envelope import Status
from eagent.schemas import (
    Detection, ExperimentRecord, OutcomeClass, SubstrateSpec,
)
from eagent.science.enzyme_substrate import (
    EnzymeSubstrateError, EnzymeSubstrateModel, Example, KernelCache,
    PlattCalibrator, STANDARD_AA, _cholesky, _solve, auroc, average_precision,
    brier, candidate_scorer, cluster_bootstrap_ci, cosine, examples_from_records,
    expected_calibration_error, grouped_folds, log_loss, reliability_table,
    spectrum_vector, training_digest,
)

MOTIF = "GGGAVGG"


def random_sequence(rng: random.Random, motif: str | None = None,
                    length: int = 220) -> str:
    s = [rng.choice(STANDARD_AA) for _ in range(length)]
    if motif:
        i = rng.randrange(40, length - 40)
        s[i:i + len(motif)] = motif
    return "".join(s)


def motif_examples(n: int = 100, seed: int = 1, target: str = "t",
                   source: str = "annotation") -> list[Example]:
    rng = random.Random(seed)
    return [Example(f"s{i}", random_sequence(rng, MOTIF if i % 2 == 0 else None),
                    target, i % 2 == 0, source, group=f"g{i}")
            for i in range(n)]


class TestSpectrum(unittest.TestCase):

    def test_vectors_are_unit_length_per_block(self) -> None:
        v = spectrum_vector("MKAIVTGGAQGIGRAIAERLAADGYNVAVL")
        norm = math.sqrt(sum(x * x for x in v.entries.values()))
        self.assertAlmostEqual(norm, 1.0, places=9)

    def test_a_sequence_is_identical_to_itself_and_unlike_a_disjoint_one(self) -> None:
        a = spectrum_vector("ACDEFGHIKLMNPQRSTVWYACDEFGHIK")
        b = spectrum_vector("WWWWWWWWWWWWWWWWWWWW")
        self.assertAlmostEqual(cosine(a, a), 1.0, places=9)
        self.assertEqual(cosine(a, b), 0.0)

    def test_non_standard_residues_are_skipped_and_counted_not_imputed(self) -> None:
        clean = spectrum_vector("ACDEFGHIK")
        dirty = spectrum_vector("ACDEXGHIK")
        self.assertGreater(dirty.n_skipped, 0)
        self.assertEqual(clean.n_skipped, 0)
        self.assertLess(cosine(clean, dirty), 1.0)

    def test_the_cache_is_symmetric_and_memoised(self) -> None:
        cache = KernelCache()
        s1, s2 = "ACDEFGHIKLMNPQ", "KLMNPQRSTVWYAC"
        self.assertEqual(cache.similarity(s1, s2), cache.similarity(s2, s1))
        self.assertEqual(len(cache._sim), 1)


class TestLinearAlgebra(unittest.TestCase):

    def test_cholesky_solves_a_known_system(self) -> None:
        a = [[4.0, 2.0, 0.6], [2.0, 5.0, 1.0], [0.6, 1.0, 3.0]]
        x_true = [1.0, -2.0, 0.5]
        b = [sum(a[i][j] * x_true[j] for j in range(3)) for i in range(3)]
        x = _solve(_cholesky(a), b)
        for got, want in zip(x, x_true):
            self.assertAlmostEqual(got, want, places=9)

    def test_an_indefinite_matrix_is_refused(self) -> None:
        with self.assertRaises(EnzymeSubstrateError):
            _cholesky([[1.0, 2.0], [2.0, 1.0]])


class TestMetrics(unittest.TestCase):

    def test_auroc(self) -> None:
        self.assertEqual(auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]), 1.0)
        self.assertEqual(auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]), 0.0)
        self.assertEqual(auroc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]), 0.5)
        self.assertIsNone(auroc([0.1, 0.2], [True, True]))

    def test_average_precision(self) -> None:
        self.assertEqual(average_precision([3, 2, 1], [True, True, False]), 1.0)
        self.assertAlmostEqual(average_precision([3, 2, 1], [True, False, True]),
                               (1 + 2 / 3) / 2)
        self.assertIsNone(average_precision([1, 2], [False, False]))

    def test_brier_and_log_loss(self) -> None:
        self.assertEqual(brier([1.0, 0.0], [True, False]), 0.0)
        self.assertEqual(brier([0.5, 0.5], [True, False]), 0.25)
        self.assertAlmostEqual(log_loss([0.5, 0.5], [True, False]), math.log(2))
        self.assertTrue(math.isfinite(log_loss([0.0], [True])))

    def test_a_calibrated_sample_has_zero_ece(self) -> None:
        probs = [0.2] * 10 + [0.8] * 10
        labels = [True] * 2 + [False] * 8 + [True] * 8 + [False] * 2
        self.assertAlmostEqual(expected_calibration_error(probs, labels), 0.0)

    def test_an_overconfident_sample_has_large_ece(self) -> None:
        self.assertAlmostEqual(
            expected_calibration_error([0.95] * 10, [True] * 5 + [False] * 5), 0.45)

    def test_reliability_bins_account_for_every_item(self) -> None:
        probs = [0.0, 0.05, 0.5, 0.99, 1.0]
        table = reliability_table(probs, [False, False, True, True, True])
        self.assertEqual(sum(row["n"] for row in table), 5)
        self.assertEqual(table[-1]["n"], 2)       # 0.99 and 1.0 share the top bin

    def test_the_bootstrap_resamples_clusters_not_rows(self) -> None:
        rng = random.Random(0)
        n_clusters, per = 12, 5
        scores, labels, clusters = [], [], []
        for c in range(n_clusters):
            label = c % 2 == 0
            for _ in range(per):
                scores.append(rng.random() + (0.5 if label else 0.0))
                labels.append(label)
                clusters.append(f"c{c}")
        ci = cluster_bootstrap_ci(auroc, scores, labels, clusters, n_boot=200)
        point = auroc(scores, labels)
        self.assertIsNotNone(ci)
        self.assertLessEqual(ci[0], point)
        self.assertGreaterEqual(ci[1], point)
        row_ci = cluster_bootstrap_ci(auroc, scores, labels,
                                      [f"r{i}" for i in range(len(scores))],
                                      n_boot=200)
        # fully clustered data is less informative than the same rows treated
        # as independent, so its interval must be at least as wide
        self.assertGreaterEqual(ci[1] - ci[0], (row_ci[1] - row_ci[0]) * 0.9)


class TestGroupedFolds(unittest.TestCase):

    def test_no_group_is_split_and_every_fold_is_used(self) -> None:
        groups = [f"g{i // 3}" for i in range(60)]
        folds = grouped_folds(groups, 5, seed=2)
        seen: dict[str, set[int]] = {}
        for g, f in zip(groups, folds):
            seen.setdefault(g, set()).add(f)
        self.assertTrue(all(len(v) == 1 for v in seen.values()))
        self.assertEqual(set(folds), set(range(5)))

    def test_folds_are_balanced_and_deterministic(self) -> None:
        groups = [f"g{i // 2}" for i in range(100)]
        a = grouped_folds(groups, 4, seed=7)
        self.assertEqual(a, grouped_folds(groups, 4, seed=7))
        sizes = [a.count(k) for k in range(4)]
        self.assertLessEqual(max(sizes) - min(sizes), 2)

    def test_one_fold_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            grouped_folds(["a", "b"], 1)


class TestPlatt(unittest.TestCase):

    def test_separable_scores_do_not_produce_certainty(self) -> None:
        cal = PlattCalibrator.fit([-1.0] * 5 + [1.0] * 5, [False] * 5 + [True] * 5)
        self.assertTrue(math.isfinite(cal.a))
        self.assertLess(cal(1.0), 1.0)
        self.assertGreater(cal(-1.0), 0.0)

    def test_the_map_is_monotone_in_the_score(self) -> None:
        rng = random.Random(3)
        scores = [rng.gauss(0, 1) for _ in range(200)]
        labels = [rng.random() < 1 / (1 + math.exp(-2 * s)) for s in scores]
        cal = PlattCalibrator.fit(scores, labels)
        self.assertGreater(cal.a, 0)
        self.assertLess(cal(-1.0), cal(0.0))
        self.assertLess(cal(0.0), cal(1.0))

    def test_one_class_cannot_be_calibrated(self) -> None:
        with self.assertRaises(EnzymeSubstrateError):
            PlattCalibrator.fit([0.1, 0.2], [True, True])


class TestModel(unittest.TestCase):

    def test_a_planted_motif_is_found_on_held_out_sequences(self) -> None:
        model = EnzymeSubstrateModel(motif_examples(100, seed=1))
        rng = random.Random(99)
        test = [(random_sequence(rng, MOTIF), True) for _ in range(15)] + \
               [(random_sequence(rng), False) for _ in range(15)]
        preds = [model.predict(s, "t") for s, _ in test]
        self.assertGreater(auroc([p.score for p in preds], [y for _, y in test]), 0.95)
        self.assertGreater(preds[0].probability, 0.5)
        self.assertLess(preds[-1].probability, 0.5)

    def test_a_target_with_no_examples_is_refused_not_extrapolated(self) -> None:
        pred = EnzymeSubstrateModel(motif_examples(40)).predict("MKAIV", "other")
        self.assertIsNone(pred.score)
        self.assertIsNone(pred.probability)
        self.assertEqual(pred.claim_ceiling, "none")
        self.assertIn("never given", pred.basis)

    def test_too_few_examples_rank_but_report_no_probability(self) -> None:
        few = motif_examples(10, seed=4)       # 5 + 5, below 8 per class
        pred = EnzymeSubstrateModel(few).predict(few[0].sequence, "t")
        self.assertIsNotNone(pred.score)
        self.assertIsNone(pred.probability)
        self.assertFalse(pred.calibrated)
        self.assertIn("reports no probability", pred.basis)

    def test_annotation_labels_cap_the_claim_at_annotated_class(self) -> None:
        pred = EnzymeSubstrateModel(motif_examples(60)).predict("MKAIV" * 10, "t")
        self.assertEqual(pred.claim_ceiling, "annotation_class")

    def test_measured_labels_carry_a_measured_ceiling(self) -> None:
        model = EnzymeSubstrateModel(motif_examples(60, source="measured"))
        self.assertIn("measured_activity", model.predict("MKAIV" * 10, "t").claim_ceiling)

    def test_mixing_label_sources_in_one_target_is_refused(self) -> None:
        mixed = motif_examples(20) + [Example(
            "m1", "ACDEFGHIK" * 10, "t", True, "measured")]
        with self.assertRaises(EnzymeSubstrateError) as caught:
            EnzymeSubstrateModel(mixed)
        self.assertIn("different questions", str(caught.exception))

    def test_a_sequence_labelled_both_ways_is_refused(self) -> None:
        a = Example("a", "ACDEFGHIK" * 10, "t", True, "annotation")
        b = Example("b", "ACDEFGHIK" * 10, "t", False, "annotation")
        with self.assertRaises(EnzymeSubstrateError):
            EnzymeSubstrateModel([a, b])

    def test_an_unknown_label_source_is_refused(self) -> None:
        with self.assertRaises(EnzymeSubstrateError):
            Example("a", "ACD", "t", True, "vibes")

    def test_the_training_digest_ignores_order_and_group(self) -> None:
        ex = motif_examples(20)
        regrouped = [Example(e.example_id, e.sequence, e.target, e.label,
                             e.label_source, e.evidence, "same") for e in ex]
        self.assertEqual(training_digest(ex), training_digest(list(reversed(regrouped))))

    def test_the_nearest_training_sequence_is_reported(self) -> None:
        ex = motif_examples(40)
        pred = EnzymeSubstrateModel(ex).predict(ex[3].sequence, "t")
        self.assertEqual(pred.nearest_id, ex[3].example_id)
        self.assertAlmostEqual(pred.nearest_similarity, 1.0, places=9)

    def test_the_model_card_names_what_it_was_trained_on(self) -> None:
        card = EnzymeSubstrateModel(motif_examples(60)).model_card()
        self.assertEqual(card["n_examples"], 60)
        self.assertEqual(card["heads"]["t"]["label_source"], "annotation")
        self.assertEqual(len(card["training_digest"]), 64)


class TestCalibrationIsOutOfFoldAndGrouped(unittest.TestCase):
    """A calibration fitted on leaked scores is confident about nothing real."""

    @staticmethod
    def near_duplicates_of_noise(grouped: bool, seed: int = 5) -> list[Example]:
        """40 sequences x 3 close variants, labels unrelated to the sequence."""
        rng = random.Random(seed)
        out: list[Example] = []
        for i in range(40):
            base = random_sequence(rng)
            label = rng.random() < 0.5
            for copy in range(3):
                seq = list(base)
                for _ in range(3):                       # ~1% point mutations
                    seq[rng.randrange(len(seq))] = rng.choice(STANDARD_AA)
                out.append(Example(f"s{i}_{copy}", "".join(seq), "t", label,
                                   "annotation",
                                   group=f"g{i}" if grouped else None))
        return out

    def test_noise_with_grouped_folds_earns_no_probability(self) -> None:
        for seed in (5, 6, 7, 8):
            with self.subTest(seed=seed):
                model = EnzymeSubstrateModel(
                    self.near_duplicates_of_noise(True, seed))
                report = model.report("t")
                self.assertFalse(report.calibrated, report.calibration_note)
                self.assertIsNone(model.predict("ACDEFGHIK" * 12, "t").probability)

    def test_the_same_noise_with_leaky_folds_earns_a_confident_probability(self) -> None:
        confident = 0
        for seed in (5, 6, 7, 8):
            model = EnzymeSubstrateModel(self.near_duplicates_of_noise(False, seed))
            report = model.report("t")
            if report.calibrated and report.calibration["a"] > 1.0:
                confident += 1
        self.assertGreaterEqual(confident, 3)

    def test_the_refusal_says_why(self) -> None:
        note = EnzymeSubstrateModel(
            self.near_duplicates_of_noise(True)).report("t").calibration_note
        self.assertIn("no demonstrated skill", note)

    def test_a_real_signal_survives_grouping_and_is_calibrated(self) -> None:
        report = EnzymeSubstrateModel(motif_examples(100)).report("t")
        self.assertTrue(report.calibrated)
        self.assertGreater(report.oof_auroc, 0.9)


class TestUpdate(unittest.TestCase):

    def test_an_update_changes_the_digest_and_leaves_the_original_alone(self) -> None:
        base = EnzymeSubstrateModel(motif_examples(40))
        digest_before = training_digest(base.examples)
        new = motif_examples(10, seed=50)
        updated, record = base.update(new)
        self.assertTrue(record.changed)
        self.assertEqual(record.before_digest, digest_before)
        self.assertEqual(record.n_added, 10)
        self.assertEqual(training_digest(base.examples), digest_before)
        self.assertEqual(len(updated.examples), 50)
        self.assertEqual(updated.updates[-1], record)

    def test_duplicates_are_ignored_and_counted_not_double_counted(self) -> None:
        base = EnzymeSubstrateModel(motif_examples(40))
        _, record = base.update(motif_examples(40))
        self.assertEqual(record.n_added, 0)
        self.assertEqual(record.n_duplicates_ignored, 40)
        self.assertFalse(record.changed)

    def test_a_new_target_appears_only_after_measured_examples_arrive(self) -> None:
        base = EnzymeSubstrateModel(motif_examples(40))
        seq = "ACDEFGHIKLMNPQRSTVWY" * 8
        self.assertIsNone(base.predict(seq, "substrate:X").score)
        measured = motif_examples(24, seed=60, target="substrate:X",
                                  source="measured")
        updated, record = base.update(measured)
        self.assertEqual(record.new_targets, ("substrate:X",))
        self.assertIsNotNone(updated.predict(seq, "substrate:X").score)
        self.assertTrue(updated.predict(seq, "substrate:X").calibrated)

    def test_a_measured_active_pulls_its_neighbours_up(self) -> None:
        rng = random.Random(8)
        parent = random_sequence(rng)
        neighbour = parent[:100] + "W" + parent[101:]
        labelled = [Example(f"n{i}", random_sequence(rng), "t", i % 2 == 0,
                            "measured", group=f"g{i}") for i in range(30)]
        base = EnzymeSubstrateModel(labelled)
        before = base.predict(neighbour, "t").score
        updated, _ = base.update([Example("hit", parent, "t", True, "measured")])
        after = updated.predict(neighbour, "t").score
        self.assertGreater(after, before)

    def test_a_measured_inactive_pulls_its_neighbours_down(self) -> None:
        rng = random.Random(9)
        parent = random_sequence(rng)
        neighbour = parent[:100] + "W" + parent[101:]
        labelled = [Example(f"n{i}", random_sequence(rng), "t", i % 2 == 0,
                            "measured", group=f"g{i}") for i in range(30)]
        base = EnzymeSubstrateModel(labelled)
        updated, _ = base.update([Example("miss", parent, "t", False, "measured")])
        self.assertLess(updated.predict(neighbour, "t").score,
                        base.predict(neighbour, "t").score)


def record(rid: str, outcome: OutcomeClass, *, seq: str | None = "ACDEFGHIK" * 8,
           smiles: str | None = "CC(=O)c1ccccc1", is_variant: bool = False,
           parent: str | None = None) -> ExperimentRecord:
    detection = Detection(
        method="chiral GC-MS", confirms_product_identity=True,
        limit_of_detection=0.5, limit_unit="percent")
    return ExperimentRecord(
        record_id=rid, sequence=seq, substrate=SubstrateSpec(
            name="acetophenone", isomeric_smiles=smiles),
        outcome=outcome, detection=detection, is_variant=is_variant,
        parent_sequence_sha256=parent)


class TestExamplesFromRecords(unittest.TestCase):

    def test_positives_and_negatives_become_measured_examples(self) -> None:
        recs = [record("r1", OutcomeClass.CONFIRMED_TARGET_PRODUCT, seq="A" * 40),
                record("r2", OutcomeClass.NO_TARGET_PRODUCT_DETECTED, seq="C" * 40),
                record("r3", OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION,
                       seq="D" * 40)]
        examples, skipped = examples_from_records(recs)
        self.assertEqual([e.label for e in examples], [True, False, False])
        self.assertTrue(all(e.label_source == "measured" for e in examples))
        self.assertEqual({e.target for e in examples},
                         {"substrate:CC(=O)c1ccccc1"})
        self.assertEqual(skipped, [])

    def test_an_expression_failure_is_undetermined_not_negative(self) -> None:
        examples, skipped = examples_from_records(
            [record("r1", OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE)])
        self.assertEqual(examples, [])
        self.assertIn("undetermined", skipped[0])

    def test_untested_and_computational_outcomes_are_not_examples(self) -> None:
        recs = [record("r1", OutcomeClass.NOT_TESTED),
                record("r2", OutcomeClass.COMPUTATIONAL_NEGATIVE),
                record("r3", OutcomeClass.COMPUTATIONAL_FAILURE)]
        examples, skipped = examples_from_records(recs)
        self.assertEqual(examples, [])
        self.assertEqual(len(skipped), 3)

    def test_a_record_with_no_sequence_or_no_substrate_is_left_out_with_a_reason(self) -> None:
        recs = [record("r1", OutcomeClass.NO_TARGET_PRODUCT_DETECTED, seq=None),
                record("r2", OutcomeClass.NO_TARGET_PRODUCT_DETECTED, smiles=None)]
        examples, skipped = examples_from_records(recs)
        self.assertEqual(examples, [])
        self.assertIn("no sequence", skipped[0])
        self.assertIn("no substrate identity", skipped[1])

    def test_a_variant_is_grouped_with_its_parent(self) -> None:
        recs = [record("r1", OutcomeClass.NO_TARGET_PRODUCT_DETECTED, seq="A" * 40,
                       is_variant=True, parent="abc123")]
        examples, _ = examples_from_records(recs)
        self.assertEqual(examples[0].group, "abc123")

    def test_a_custom_target_naming_is_honoured(self) -> None:
        examples, _ = examples_from_records(
            [record("r1", OutcomeClass.NO_TARGET_PRODUCT_DETECTED)],
            target_of=lambda r: "ketone-class")
        self.assertEqual(examples[0].target, "ketone-class")


class TestComparatorSeam(unittest.TestCase):
    """The model plugs into the baseline comparison the agent is judged against."""

    def pool(self, sequences):
        from eagent.eval.baselines import CandidatePool
        from eagent.schemas import Candidate, SequenceRecord
        return CandidatePool.build([
            Candidate(candidate_id=f"c{i}",
                      sequence_record=SequenceRecord(candidate_id=f"c{i}",
                                                     sequence=s))
            for i, s in enumerate(sequences)])

    def registration(self):
        from eagent.eval.metrics import PreRegistration
        return PreRegistration.from_plan(
            {"min_conversion_pct": 20.0}, k_slots=3,
            registered_by="operator:test", registered_at="2026-01-01T00:00:00Z")

    def test_a_calibrated_head_ranks_the_pool_and_names_itself(self) -> None:
        from eagent.eval.baselines import make_specificity_model_comparator
        model = EnzymeSubstrateModel(motif_examples(100))
        rng = random.Random(77)
        seqs = [random_sequence(rng, MOTIF if i < 3 else None) for i in range(12)]
        scorer = candidate_scorer(model, "t")
        selection = make_specificity_model_comparator(
            scorer, name="spectrum_krr", model_id=scorer.model_id)(
                self.pool(seqs), 3, self.registration())
        self.assertTrue(selection.is_available)
        self.assertEqual(set(selection.ranked_candidate_ids), {"c0", "c1", "c2"})
        self.assertIn(model_digest_prefix(model), selection.basis)

    def test_an_uncalibrated_head_gives_no_prediction_under_probability_ranking(self) -> None:
        from eagent.eval.baselines import make_specificity_model_comparator
        model = EnzymeSubstrateModel(motif_examples(10))     # too few to calibrate
        scorer = candidate_scorer(model, "t")
        selection = make_specificity_model_comparator(scorer, name="m")(
            self.pool(["ACDEFGHIK" * 10] * 1 + ["MKVLA" * 20]), 2,
            self.registration())
        self.assertEqual(selection.ranked_candidate_ids, ())
        self.assertIn("no prediction", selection.shortfall_reason)

    def test_score_ranking_is_available_and_labelled(self) -> None:
        model = EnzymeSubstrateModel(motif_examples(10))
        scorer = candidate_scorer(model, "t", ranking="score")
        self.assertIn("[score]", scorer.model_id)
        self.assertIsNotNone(scorer(type("C", (), {"sequence": "ACDEFGHIK" * 10})()))

    def test_a_bad_ranking_mode_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            candidate_scorer(EnzymeSubstrateModel(motif_examples(10)), "t",
                             ranking="vibes")


def model_digest_prefix(model) -> str:
    return training_digest(model.examples)[:12]


if __name__ == "__main__":
    unittest.main()
