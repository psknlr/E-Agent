"""Tests for :mod:`eagent.eval.baselines`.

The claim these tests defend is narrow and load-bearing: a comparison between
the agent and a baseline is only a comparison if both saw the same candidate
pool, the same budget and the same hit definition. Every check below plants a
comparator that breaks exactly one of those three and expects a refusal, plus
the ordinary case where all comparators agree on all three.

The seams -- family-level function prediction and the substrate-specificity
model -- are expected to report themselves unavailable in this environment.
A test that let a seam fall back to a heuristic would be testing that the
agent can beat a strawman.

Runs under pytest, or standalone with ``python3 tests/test_baselines.py``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - standalone execution
    _SRC = Path(__file__).resolve().parents[1] / "src"
    if _SRC.is_dir() and str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from eagent.datalayer.house_db import ExpressionStatus
from eagent.eval.baselines import (
    CandidatePool,
    DEFAULT_COMPARATORS,
    PoolMismatchError,
    RankedSelection,
    compare_baselines,
    comparator_name,
    docking_score_ranking,
    family_function_prediction,
    full_agent,
    homology_multi_seed,
    make_docking_comparator,
    make_family_function_comparator,
    make_specificity_model_comparator,
    score_selection,
    substrate_specificity_model,
)
from eagent.eval.metrics import OutcomeRow, PreRegistration
from eagent.schemas import (
    Candidate,
    CatalyticMapping,
    ComplexPose,
    ConfidenceLevel,
    FamilyAnnotation,
    ScoreDimension,
    SequenceRecord,
)
from eagent.schemas.record import OutcomeClass
from eagent.science.scorecard import GATE_NAMES

BASE = ("MKAAVLYEFGKPLEIKEVEVAPPKAHEVRIKIAYTGVCHTDAYTLSGADPEGLFPVILGHEGAGIVES"
        "VGEGVTNVKPGDHVIPLYTPECGECKFCKSGKTNLCQKIRATQGKGLMPDGTSRFTCKGKEILHYMG")

CRITERION = {"min_conversion_pct": 20.0}


def registration(k_slots: int = 8) -> PreRegistration:
    return PreRegistration.from_plan(
        dict(CRITERION), k_slots=k_slots, registered_by="operator:ada",
        registered_at="2026-01-01T00:00:00Z")


def mutate(sequence: str, index: int, letter: str) -> str:
    chars = list(sequence)
    chars[index] = letter
    return "".join(chars)


def candidate(
    cid: str,
    index: int,
    *,
    family: str = "SDR",
    cluster: str | None = None,
    identity: float = 60.0,
    seed: str = "SEED_A",
    gates: dict[str, bool | None] | None = None,
    docking: float | None = -8.0,
    scoring_function: str = "vina",
) -> Candidate:
    """A candidate with just enough scorecard for every comparator to run."""
    cand = Candidate(
        candidate_id=cid,
        sequence_record=SequenceRecord(
            candidate_id=cid, sequence=mutate(BASE, index, "W"),
            percent_identity=identity, seed_accession=seed,
            search_method="mmseqs2"),
        family=FamilyAnnotation(family_name=family,
                                sequence_cluster_id=cluster or f"clu{index % 3}"),
        catalytic_mapping=CatalyticMapping(
            catalytic_template_id="cat:sdr:kred",
            role_to_residue={"catalytic_Tyr": f"Y{150 + index}"}),
        poses=([ComplexPose(pose_id=f"{cid}-p1", method="template_docking",
                            docking_score=docking,
                            docking_score_function=scoring_function)]
               if docking is not None else []),
    )
    verdicts = dict.fromkeys(GATE_NAMES, True)
    verdicts.update(gates or {})
    for name, passed in verdicts.items():
        cand.set_dimension(ScoreDimension(
            name=name, is_gate=True, gate_passed=passed, direction="categorical",
            level=(ConfidenceLevel.STRONG if passed
                   else ConfidenceLevel.CONTRADICTORY if passed is False
                   else ConfidenceLevel.INSUFFICIENT)))
    cand.set_dimension(ScoreDimension(name="functional_literature_evidence",
                                      level=ConfidenceLevel.MODERATE,
                                      value=identity / 100.0))
    cand.set_dimension(ScoreDimension(name="catalytic_geometry",
                                      level=ConfidenceLevel.STRONG, value=1.0))
    cand.set_dimension(ScoreDimension(name="model_uncertainty",
                                      level=ConfidenceLevel.STRONG,
                                      direction="categorical"))
    return cand


def pool_of(n: int = 9) -> CandidatePool:
    return CandidatePool.build(
        [candidate(f"c{i}", i, family="SDR" if i < 6 else "AKR",
                   identity=90.0 - i, seed="SEED_A" if i % 2 else "SEED_B",
                   docking=-9.0 + 0.3 * i)
         for i in range(n)],
        pool_id="kred-pilot")


def outcomes_for(pool: CandidatePool, hits=("c1", "c4")) -> dict[str, OutcomeRow]:
    return {
        cand.candidate_id: OutcomeRow(
            cand.candidate_id,
            (OutcomeClass.CONFIRMED_TARGET_PRODUCT
             if cand.candidate_id in hits
             else OutcomeClass.NO_TARGET_PRODUCT_DETECTED),
            conversion_pct=55.0 if cand.candidate_id in hits else 1.0,
            expression_status=ExpressionStatus.SOLUBLE)
        for cand in pool.candidates
    }


def rogue(name: str, **overrides):
    """A comparator that breaks exactly one rule of the shared comparison."""
    def comparator(pool: CandidatePool, budget: int,
                   registration_: PreRegistration) -> RankedSelection:
        fields = dict(
            comparator=name, question_answered="a rogue comparator",
            pool_digest=pool.digest,
            criterion_digest=registration_.registered_digest,
            budget=budget, ranked_candidate_ids=tuple(pool.ids()[:2]))
        fields.update(overrides)
        return RankedSelection(**fields)
    comparator.name = name
    return comparator


# --------------------------------------------------------------------------
# the shared pool
# --------------------------------------------------------------------------

class TestSharedPool(unittest.TestCase):

    def test_every_default_comparator_records_the_one_pool_and_criterion(self) -> None:
        pool, reg = pool_of(), registration()
        comparison = compare_baselines(pool, 4, reg, outcomes=outcomes_for(pool))
        self.assertEqual(len(comparison.scores), len(DEFAULT_COMPARATORS))
        for score in comparison.scores:
            self.assertEqual(score.selection.pool_digest, pool.digest)
            self.assertEqual(score.selection.criterion_digest,
                             reg.registered_digest)
            self.assertEqual(score.selection.budget, 4)
            self.assertTrue(set(score.selection.ranked_candidate_ids)
                            <= set(pool.ids()))

    def test_a_comparator_reaching_outside_the_pool_is_refused(self) -> None:
        pool, reg = pool_of(), registration()
        with self.assertRaises(PoolMismatchError) as caught:
            compare_baselines(pool, 4, reg,
                              [rogue("outsider",
                                     ranked_candidate_ids=("not_in_pool",))])
        self.assertIn("not_in_pool", str(caught.exception))

    def test_a_comparator_with_another_pool_digest_is_refused(self) -> None:
        pool, reg = pool_of(), registration()
        with self.assertRaises(PoolMismatchError):
            compare_baselines(pool, 4, reg,
                              [rogue("stale", pool_digest="0" * 64)])

    def test_a_comparator_with_another_criterion_is_refused(self) -> None:
        pool, reg = pool_of(), registration()
        with self.assertRaises(PoolMismatchError):
            compare_baselines(pool, 4, reg,
                              [rogue("rescored", criterion_digest="0" * 64)])

    def test_a_comparator_overspending_the_budget_is_refused(self) -> None:
        pool, reg = pool_of(), registration()
        with self.assertRaises(PoolMismatchError):
            compare_baselines(
                pool, 2, reg,
                [rogue("greedy", ranked_candidate_ids=tuple(pool.ids()[:5]))])

    def test_a_comparator_recording_another_budget_is_refused(self) -> None:
        pool, reg = pool_of(), registration()
        with self.assertRaises(PoolMismatchError):
            compare_baselines(pool, 4, reg, [rogue("drifted", budget=9)])

    def test_a_repeated_pick_is_refused(self) -> None:
        pool, reg = pool_of(), registration()
        with self.assertRaises(PoolMismatchError):
            compare_baselines(pool, 4, reg,
                              [rogue("double", ranked_candidate_ids=("c1", "c1"))])

    def test_the_pool_digest_changes_when_the_pool_does(self) -> None:
        first = pool_of(9)
        second = CandidatePool.build(list(first.candidates)
                                     + [candidate("extra", 20)], "kred-pilot")
        self.assertNotEqual(first.digest, second.digest)

    def test_a_duplicated_candidate_id_cannot_form_a_pool(self) -> None:
        with self.assertRaises(PoolMismatchError):
            CandidatePool.build([candidate("c1", 1), candidate("c1", 2)])

    def test_two_comparators_may_not_share_a_name(self) -> None:
        pool, reg = pool_of(), registration()
        with self.assertRaises(ValueError):
            compare_baselines(pool, 4, reg, [rogue("same"), rogue("same")])

    def test_scoring_a_selection_under_another_criterion_is_refused(self) -> None:
        pool, reg = pool_of(), registration()
        other = PreRegistration.from_plan(
            {"min_conversion_pct": 10.0}, k_slots=8,
            registered_by="operator:ada", registered_at="2026-01-01T00:00:00Z")
        selection = homology_multi_seed(pool, 4, reg)
        with self.assertRaises(PoolMismatchError):
            score_selection(selection, outcomes_for(pool), other)


# --------------------------------------------------------------------------
# the comparators themselves
# --------------------------------------------------------------------------

class TestComparators(unittest.TestCase):

    def test_homology_needs_more_than_one_seed(self) -> None:
        """A single-seed baseline samples one clade and flatters whatever beats it."""
        pool = CandidatePool.build(
            [candidate(f"c{i}", i, seed="SEED_ONLY") for i in range(5)])
        selection = homology_multi_seed(pool, 3, registration())
        self.assertFalse(selection.is_available)
        self.assertIn("seed", selection.unavailable_reason or "")

    def test_homology_picks_from_the_pool_and_spreads_them(self) -> None:
        pool = pool_of()
        selection = homology_multi_seed(pool, 4, registration())
        self.assertTrue(selection.is_available)
        self.assertEqual(selection.n_selected, 4)
        self.assertEqual(len(set(selection.ranked_candidate_ids)), 4)

    def test_the_family_seam_reports_itself_unavailable_and_says_why(self) -> None:
        selection = family_function_prediction(pool_of(), 4, registration())
        self.assertFalse(selection.is_available)
        self.assertIn("NOT whether it turns over this substrate",
                      selection.question_answered)
        self.assertIn("non-overlapping substrate ranges",
                      selection.unavailable_reason or "")

    def test_an_injected_family_predictor_carries_the_caveat_into_its_notes(self) -> None:
        comparator = make_family_function_comparator(
            lambda cand: 1.0 if cand.family.family_name == "SDR" else 0.0)
        selection = comparator(pool_of(), 3, registration())
        self.assertTrue(selection.is_available)
        self.assertEqual(selection.n_selected, 3)
        self.assertTrue(any("different question" in note
                            for note in selection.notes))

    def test_the_specificity_seam_is_left_open_rather_than_imitated(self) -> None:
        selection = substrate_specificity_model(pool_of(), 4, registration())
        self.assertFalse(selection.is_available)
        self.assertIn("heuristic", selection.unavailable_reason or "")

    def test_an_injected_specificity_model_names_itself(self) -> None:
        comparator = make_specificity_model_comparator(
            lambda cand: cand.sequence_record.percent_identity,
            model_id="toy-model-v0")
        selection = comparator(pool_of(), 3, registration())
        self.assertIn("toy-model-v0", selection.basis)
        self.assertEqual(selection.ranked_candidate_ids, ("c0", "c1", "c2"))

    def test_docking_ranks_the_best_score_first(self) -> None:
        selection = docking_score_ranking(pool_of(), 3, registration())
        self.assertEqual(selection.ranked_candidate_ids, ("c0", "c1", "c2"))

    def test_docking_refuses_to_pool_two_scoring_functions(self) -> None:
        pool = CandidatePool.build([
            candidate("c0", 0, docking=-9.0, scoring_function="vina"),
            candidate("c1", 1, docking=-40.0, scoring_function="chemplp"),
        ])
        selection = docking_score_ranking(pool, 2, registration())
        self.assertFalse(selection.is_available)
        self.assertIn("one scale", selection.unavailable_reason or "")

    def test_the_docking_sign_convention_is_an_argument_not_a_guess(self) -> None:
        pool = pool_of()
        higher = make_docking_comparator(lower_is_better=False,
                                         name="docking_higher_is_better")
        self.assertEqual(higher(pool, 1, registration()).ranked_candidate_ids,
                         ("c8",))
        self.assertEqual(docking_score_ranking(pool, 1,
                                               registration()).ranked_candidate_ids,
                         ("c0",))

    def test_a_pool_with_no_docking_scores_is_reported_not_guessed(self) -> None:
        pool = CandidatePool.build([candidate(f"c{i}", i, docking=None)
                                    for i in range(3)])
        selection = docking_score_ranking(pool, 2, registration())
        self.assertFalse(selection.is_available)

    def test_the_agent_runs_the_pipeline_selection(self) -> None:
        selection = full_agent(pool_of(), 4, registration())
        self.assertTrue(selection.is_available)
        self.assertTrue(selection.n_selected <= 4)
        self.assertTrue(any(note.startswith("role ") for note in selection.notes))

    def test_the_agent_refuses_an_ungated_pool(self) -> None:
        """Composing from unchecked candidates would read 'never checked' as 'fine'."""
        bare = Candidate(
            candidate_id="bare",
            sequence_record=SequenceRecord(candidate_id="bare", sequence=BASE))
        pool = CandidatePool.build([bare])
        selection = full_agent(pool, 1, registration())
        self.assertFalse(selection.is_available)
        self.assertIn("refused", selection.unavailable_reason or "")

    def test_a_comparator_is_named_by_its_own_name_or_its_function_name(self) -> None:
        self.assertEqual(comparator_name(full_agent), "full_agent")
        self.assertEqual(comparator_name(family_function_prediction),
                         "family_function_prediction")


# --------------------------------------------------------------------------
# scoring the comparison
# --------------------------------------------------------------------------

class TestScoring(unittest.TestCase):

    def test_picks_with_no_recorded_outcome_are_not_counted_as_failures(self) -> None:
        pool, reg = pool_of(), registration()
        selection = homology_multi_seed(pool, 4, reg)
        partial = {cid: row for cid, row in outcomes_for(pool).items()
                   if cid in selection.ranked_candidate_ids[:2]}
        score = score_selection(selection, partial, reg)
        self.assertEqual(score.n_without_outcome, 2)
        self.assertFalse(score.fair)
        self.assertEqual(score.endpoint.slots_spent, 2)

    def test_a_comparator_none_of_whose_picks_was_tested_is_not_scored(self) -> None:
        pool, reg = pool_of(), registration()
        selection = homology_multi_seed(pool, 4, reg)
        score = score_selection(selection, {}, reg)
        self.assertIsNone(score.endpoint)
        self.assertIn("cannot be scored", score.note)

    def test_an_unavailable_comparator_is_listed_rather_than_scored_zero(self) -> None:
        pool, reg = pool_of(), registration()
        comparison = compare_baselines(pool, 4, reg, outcomes=outcomes_for(pool))
        self.assertIn("family_function_prediction", comparison.unavailable)
        self.assertIn("substrate_specificity_model", comparison.unavailable)
        for score in comparison.scores:
            if not score.selection.is_available:
                self.assertIsNone(score.endpoint)

    def test_overlapping_intervals_are_reported_beside_the_ordering(self) -> None:
        """At this budget the comparators are not separated, and it must say so."""
        pool, reg = pool_of(), registration()
        comparison = compare_baselines(pool, 4, reg, outcomes=outcomes_for(pool))
        ordered = comparison.ordered_by_primary_endpoint()
        self.assertTrue(ordered)
        self.assertTrue(comparison.indistinguishable_pairs())
        self.assertIn("intervals overlap", comparison.render())

    def test_without_outcomes_the_selections_are_recorded_and_not_scored(self) -> None:
        pool, reg = pool_of(), registration()
        comparison = compare_baselines(pool, 4, reg)
        for score in comparison.scores:
            self.assertIsNone(score.endpoint)
        self.assertIn("not scored", comparison.render())

    def test_a_zero_budget_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            compare_baselines(pool_of(), 0, registration())

    def test_the_comparison_serialises_with_everything_needed_to_recheck_it(self) -> None:
        pool, reg = pool_of(), registration()
        payload = compare_baselines(pool, 4, reg,
                                    outcomes=outcomes_for(pool)).to_dict()
        self.assertEqual(payload["pool_digest"], pool.digest)
        self.assertEqual(payload["criterion_digest"], reg.registered_digest)
        self.assertEqual(len(payload["scores"]), len(DEFAULT_COMPARATORS))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
