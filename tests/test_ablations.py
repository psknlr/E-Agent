"""Tests for :mod:`eagent.eval.ablations`.

Three things are checked, and they are the three ways an ablation table
misleads:

* a module whose removal is *not evaluable* from one round must say so rather
  than report a delta of zero, which reads as "this module contributes
  nothing";
* the ablated run and the full-system run must differ in exactly one thing,
  so the originals are never mutated and the pool, the budget and the
  registered criterion are shared;
* the scope statement -- 96 experiments on a single substrate establish a
  first application and not generality across reaction types -- has to travel
  with the numbers.

Runs under pytest, or standalone with ``python3 tests/test_ablations.py``.
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
from eagent.eval import ablations as ablations_module
from eagent.eval.ablations import (
    AblatedModule,
    PriorRound,
    SCOPE_STATEMENT,
    SelectionSettings,
    run_ablations,
    select_under,
)
from eagent.eval.baselines import CandidatePool
from eagent.eval.metrics import OutcomeRow, PreRegistration
from eagent.schemas import (
    Candidate,
    CatalyticMapping,
    ConfidenceLevel,
    FamilyAnnotation,
    ScoreDimension,
    SequenceRecord,
)
from eagent.schemas.record import OutcomeClass
from eagent.science.scorecard import GATE_NAMES

BASE = ("MKAAVLYEFGKPLEIKEVEVAPPKAHEVRIKIAYTGVCHTDAYTLSGADPEGLFPVILGHEGAGIVES"
        "VGEGVTNVKPGDHVIPLYTPECGECKFCKSGKTNLCQKIRATQGKGLMPDGTSRFTCKGKEILHYMG")


def registration(k_slots: int = 8) -> PreRegistration:
    return PreRegistration.from_plan(
        {"min_conversion_pct": 20.0}, k_slots=k_slots,
        registered_by="operator:ada", registered_at="2026-01-01T00:00:00Z")


def candidate(cid: str, index: int, *, family: str = "SDR",
              identity: float = 60.0,
              gates: dict[str, bool | None] | None = None) -> Candidate:
    chars = list(BASE)
    chars[index % (len(BASE) - 1)] = "ACDEFGHIKLMNPQRSTVWY"[index % 20]
    cand = Candidate(
        candidate_id=cid,
        sequence_record=SequenceRecord(candidate_id=cid, sequence="".join(chars),
                                       percent_identity=identity,
                                       seed_accession="SEED_A",
                                       search_method="mmseqs2"),
        family=FamilyAnnotation(family_name=family,
                                sequence_cluster_id=f"clu{index % 4}"),
        catalytic_mapping=CatalyticMapping(
            catalytic_template_id="cat:sdr:kred",
            role_to_residue={"catalytic_Tyr": f"Y{150 + index}"}),
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


#: ``cofactor_blocked`` fails only the cofactor gate and is the round's one
#: hit, so dropping that gate must change the primary endpoint by exactly one.
def pool_with_a_cofactor_block() -> CandidatePool:
    members = [candidate(f"c{i}", i, family="SDR" if i < 4 else "AKR",
                         identity=60.0 - i)
               for i in range(6)]
    members.append(candidate("cofactor_blocked", 9, family="MDR", identity=99.0,
                             gates={"cofactor_compatible": False}))
    return CandidatePool.build(members, pool_id="kred-ablation")


def outcomes_for(pool: CandidatePool, hits=("cofactor_blocked",)) -> dict:
    return {
        cand.candidate_id: OutcomeRow(
            cand.candidate_id,
            (OutcomeClass.CONFIRMED_TARGET_PRODUCT
             if cand.candidate_id in hits
             else OutcomeClass.NO_TARGET_PRODUCT_DETECTED),
            conversion_pct=70.0 if cand.candidate_id in hits else 2.0,
            expression_status=ExpressionStatus.SOLUBLE)
        for cand in pool.candidates
    }


# --------------------------------------------------------------------------
# scope
# --------------------------------------------------------------------------

class TestScope(unittest.TestCase):

    def test_the_scope_statement_says_what_96_experiments_cannot_establish(self) -> None:
        self.assertIn("96 experiments on a single substrate", SCOPE_STATEMENT)
        self.assertIn("cannot establish generality across reaction types",
                      SCOPE_STATEMENT)

    def test_the_module_docstring_states_it_too(self) -> None:
        doc = ablations_module.__doc__ or ""
        self.assertIn("96 experiments on a single substrate", doc)
        self.assertIn("cannot establish generality across reaction types", doc)

    def test_a_rendered_report_carries_the_scope(self) -> None:
        pool = pool_with_a_cofactor_block()
        report = run_ablations(pool, 4, registration(), outcomes_for(pool))
        self.assertIn("96 experiments on a single substrate", report.render())


# --------------------------------------------------------------------------
# removing one module at a time
# --------------------------------------------------------------------------

class TestAblations(unittest.TestCase):

    def test_dropping_the_cofactor_gate_admits_the_blocked_candidate(self) -> None:
        pool = pool_with_a_cofactor_block()
        report = run_ablations(pool, 4, registration(), outcomes_for(pool))
        result = next(r for r in report.results
                      if r.module is AblatedModule.COFACTOR_CONSTRAINTS)
        self.assertTrue(result.applicable)
        self.assertNotIn("cofactor_blocked", result.full_system_ids)
        self.assertIn("cofactor_blocked", result.ablated_ids)
        self.assertIn("cofactor_blocked", result.candidates_added)
        self.assertEqual(result.delta_hits, 1)
        self.assertGreater(result.delta_rate or 0.0, 0.0)

    def test_a_small_round_cannot_separate_the_two_endpoints(self) -> None:
        pool = pool_with_a_cofactor_block()
        report = run_ablations(pool, 4, registration(), outcomes_for(pool))
        result = next(r for r in report.results
                      if r.module is AblatedModule.COFACTOR_CONSTRAINTS)
        self.assertTrue(result.intervals_overlap)
        self.assertIn("intervals overlap", result.describe())

    def test_dropping_the_family_quota_changes_which_clade_fills_the_plate(self) -> None:
        pool = pool_with_a_cofactor_block()
        settings = SelectionSettings(quotas={"SDR": 1, "*": 4}, cluster_cap=1)
        report = run_ablations(pool, 4, registration(), outcomes_for(pool),
                               settings=settings)
        result = next(r for r in report.results
                      if r.module is AblatedModule.FAMILY_DIVERSITY_QUOTA)
        self.assertTrue(result.applicable)
        self.assertTrue(result.selection_changed,
                        f"full={result.full_system_ids} "
                        f"ablated={result.ablated_ids}")
        self.assertGreaterEqual(len(result.ablated_ids),
                                len(result.full_system_ids))

    def test_the_geometry_ablation_drops_its_gate_and_its_axis(self) -> None:
        self.assertEqual(AblatedModule.CATALYTIC_GEOMETRY.gates_removed,
                         ("catalytic_machinery_mappable",))
        self.assertEqual(AblatedModule.CATALYTIC_GEOMETRY.dimensions_removed,
                         ("catalytic_geometry",))
        pool = pool_with_a_cofactor_block()
        report = run_ablations(pool, 4, registration(), outcomes_for(pool))
        result = next(r for r in report.results
                      if r.module is AblatedModule.CATALYTIC_GEOMETRY)
        self.assertTrue(result.applicable)
        self.assertIsNotNone(result.ablated_endpoint)

    def test_the_originals_are_never_mutated(self) -> None:
        """The two runs must see the same inputs, not inputs one of them edited."""
        pool = pool_with_a_cofactor_block()
        before = {c.candidate_id: sorted(d.name for d in c.gates())
                  for c in pool.candidates}
        run_ablations(pool, 4, registration(), outcomes_for(pool))
        after = {c.candidate_id: sorted(d.name for d in c.gates())
                 for c in pool.candidates}
        self.assertEqual(before, after)

    def test_every_ablation_is_scored_against_the_registered_criterion(self) -> None:
        pool, reg = pool_with_a_cofactor_block(), registration()
        report = run_ablations(pool, 4, reg, outcomes_for(pool))
        self.assertEqual(report.criterion_digest, reg.registered_digest)
        for result in report.results:
            if result.ablated_endpoint is not None:
                self.assertEqual(result.ablated_endpoint.criterion_digest,
                                 reg.registered_digest)


# --------------------------------------------------------------------------
# active learning: a between-round module
# --------------------------------------------------------------------------

class TestActiveLearning(unittest.TestCase):

    def test_one_round_cannot_evaluate_a_between_round_module(self) -> None:
        pool = pool_with_a_cofactor_block()
        report = run_ablations(pool, 4, registration(), outcomes_for(pool))
        result = next(r for r in report.results
                      if r.module is AblatedModule.ACTIVE_LEARNING)
        self.assertFalse(result.applicable)
        self.assertIsNone(result.delta_hits)
        self.assertIn("between rounds", result.reason_not_applicable or "")
        self.assertIn("NOT EVALUABLE", result.describe())

    def test_a_prior_round_makes_the_ablation_meaningful(self) -> None:
        pool = pool_with_a_cofactor_block()
        prior = PriorRound(tested_candidate_ids=frozenset({"c0"}),
                           quotas_before={"SDR": 1, "*": 4},
                           quotas_after={"SDR": 4, "*": 4})
        settings = SelectionSettings(quotas=prior.quotas_after,
                                     excluded_candidate_ids=frozenset({"c0"}))
        report = run_ablations(pool, 4, registration(), outcomes_for(pool),
                               settings=settings, prior_round=prior)
        result = next(r for r in report.results
                      if r.module is AblatedModule.ACTIVE_LEARNING)
        self.assertTrue(result.applicable)
        self.assertNotIn("c0", result.full_system_ids)
        self.assertIn("c0", result.ablated_ids)

    def test_an_empty_prior_round_is_still_not_evaluable(self) -> None:
        pool = pool_with_a_cofactor_block()
        report = run_ablations(pool, 4, registration(), outcomes_for(pool),
                               prior_round=PriorRound())
        result = next(r for r in report.results
                      if r.module is AblatedModule.ACTIVE_LEARNING)
        self.assertFalse(result.applicable)


# --------------------------------------------------------------------------
# the selection runner and its refusals
# --------------------------------------------------------------------------

class TestSelectUnder(unittest.TestCase):

    def test_removing_every_gate_leaves_nothing_selectable(self) -> None:
        """'No gate at all' is not 'everything is eligible'."""
        pool = pool_with_a_cofactor_block()
        picked, shortfall, notes = select_under(
            pool, 4, SelectionSettings(gates_applied=()))
        self.assertEqual(picked, [])
        self.assertIn("carries a feasibility gate", shortfall or "")
        self.assertTrue(notes)

    def test_an_excluded_candidate_is_not_offered_to_the_selection(self) -> None:
        pool = pool_with_a_cofactor_block()
        picked, _, _ = select_under(
            pool, 4, SelectionSettings(excluded_candidate_ids=frozenset({"c0"})))
        self.assertNotIn("c0", picked)

    def test_a_budget_above_the_registered_one_is_refused(self) -> None:
        pool = pool_with_a_cofactor_block()
        with self.assertRaises(ValueError):
            run_ablations(pool, 9, registration(k_slots=8), outcomes_for(pool))

    def test_a_zero_budget_is_refused(self) -> None:
        pool = pool_with_a_cofactor_block()
        with self.assertRaises(ValueError):
            run_ablations(pool, 0, registration(), outcomes_for(pool))

    def test_the_settings_differ_from_the_full_system_in_one_thing_only(self) -> None:
        base = SelectionSettings(quotas={"SDR": 2}, cluster_cap=2)
        ablated = base.without(AblatedModule.COFACTOR_CONSTRAINTS)
        self.assertEqual(ablated.quotas, base.quotas)
        self.assertEqual(ablated.cluster_cap, base.cluster_cap)
        self.assertEqual(ablated.order_of_dimensions, base.order_of_dimensions)
        self.assertNotIn("cofactor_compatible", ablated.gates_applied)
        self.assertEqual(len(ablated.gates_applied),
                         len(base.gates_applied) - 1)

    def test_the_report_serialises_with_its_scope_and_its_settings(self) -> None:
        pool = pool_with_a_cofactor_block()
        payload = run_ablations(pool, 4, registration(),
                                outcomes_for(pool)).to_dict()
        self.assertIn("96 experiments", payload["scope_statement"])
        self.assertEqual(payload["pool_digest"], pool.digest)
        self.assertEqual(len(payload["results"]), len(AblatedModule))
        self.assertIn("gates_applied", payload["settings"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
