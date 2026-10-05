"""Tests for the staged rollout plan.

The plan's value is that it is checkable, so these tests check it:

* the precondition graph is acyclic and every precondition names a deliverable
  something upstream actually produces -- a prose roadmap cannot be wrong in
  this way because nothing ever evaluates it;
* readiness reports blocked packages and the root cause, and refuses to treat
  "registered" as "reachable", because every registry entry carries
  ``connectivity_verified=false``;
* stage 2's gating question is computed: a candidate set drawn from the
  training corpus's own clusters is reported as re-testing near neighbours, and
  model outputs stay marked uncalibrated;
* an unresolved sequence cluster is never counted as novelty;
* a round spent on distant sequences with no functional evidence is flagged
  against the exploration budget.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from eagent.datalayer.layers import DataLayer
from eagent.datalayer.plan import (
    MGNIFY_EARLY_ENTRY_RULE,
    STAGES,
    BlockReason,
    ExtrapolationVerdict,
    NoveltyBudgetPolicy,
    PlanError,
    UnknownPackageError,
    all_packages,
    answer_stage2_gate,
    check_novelty_budget,
    extrapolation_check,
    get_stage,
    package,
    readiness,
    validate_plan,
)
from eagent.datalayer.registry import AccessMode, DataSource, SourceRegistry
from eagent.schemas.record import EvidenceRef, EvidenceStrength


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _registry_for(source_ids: list[str]) -> SourceRegistry:
    """A registry holding exactly the named ids, so tests stay hermetic."""
    return SourceRegistry([
        DataSource(
            id=sid,
            display_name=f"test source {sid}",
            layers=[DataLayer.REACTION_AND_CHEMISTRY],
            good_for=["test fixture"],
            not_good_for=["anything real"],
            access_modes=[AccessMode.UNKNOWN],
            needs_curation=False,
        )
        for sid in sorted(set(source_ids))
    ])


def _full_registry() -> SourceRegistry:
    ids: list[str] = []
    for plan in STAGES.values():
        ids.extend(plan.source_ids())
    return _registry_for(ids)


@dataclass
class _Seq:
    """A minimal candidate: just enough identity for the cluster lookup."""

    candidate_id: str
    accession: str
    source_id: str | None = None
    evidence: list[EvidenceRef] = field(default_factory=list)


def _training(n: int, prefix: str = "T") -> list[_Seq]:
    return [_Seq(candidate_id=f"{prefix}{i}", accession=f"{prefix}ACC{i}")
            for i in range(n)]


# ---------------------------------------------------------------------------
# plan integrity
# ---------------------------------------------------------------------------

class TestPlanIntegrity(unittest.TestCase):

    def test_the_plan_validates_at_import(self) -> None:
        self.assertEqual(validate_plan(), [])

    def test_every_precondition_names_a_real_deliverable(self) -> None:
        pkgs = all_packages()
        self.assertTrue(pkgs)
        for pid, pkg in pkgs.items():
            for pre in pkg.preconditions:
                up = pkgs[pre.package_id]
                self.assertIsNotNone(
                    up.deliverable(pre.deliverable),
                    f"{pid} requires {pre.package_id}.{pre.deliverable}")
                self.assertLessEqual(up.stage, pkg.stage)

    def test_every_package_states_what_it_must_not_claim(self) -> None:
        for pid, pkg in all_packages().items():
            self.assertTrue(pkg.must_not_claim, pid)
            self.assertTrue(pkg.deliverables, pid)
            self.assertTrue(pkg.source_ids, pid)

    def test_the_three_stages_cover_the_named_resources(self) -> None:
        s1 = set(STAGES[1].source_ids())
        for sid in ("pubchem", "chebi", "rhea", "brenda", "oed",
                    "retrobiocat_db", "uniprotkb", "interpro", "pfam",
                    "ncbi_protein", "sdred", "akr_superfamily", "rcsb_pdb",
                    "alphafold_db", "sifts", "wwpdb_ccd", "mcsa", "alphafill",
                    "sabio_rk"):
            self.assertIn(sid, s1, sid)
        s2 = set(STAGES[2].source_ids())
        for sid in ("enzengdb", "fireprotdb", "mavedb", "skid", "intenzydb",
                    "esibank", "reactzyme"):
            self.assertIn(sid, s2, sid)
        s3 = set(STAGES[3].source_ids())
        for sid in ("mgnify_proteins", "cath_funfam", "eggnog", "retrorules",
                    "enzchemred", "bacdive"):
            self.assertIn(sid, s3, sid)

    def test_lookup_helpers_raise_rather_than_return_none(self) -> None:
        with self.assertRaises(PlanError):
            get_stage(4)
        with self.assertRaises(UnknownPackageError):
            package("s1_does_not_exist")
        self.assertEqual(package("s1_known_activity_seeds").stage, 1)

    def test_stage_3_encodes_the_metagenome_rule(self) -> None:
        pkg = package("s3_novel_sequence_space")
        self.assertIn(MGNIFY_EARLY_ENTRY_RULE, pkg.must_not_claim)
        self.assertIn("mgnify_proteins", pkg.source_ids)
        self.assertIn("whole experimental round must not be spent",
                      MGNIFY_EARLY_ENTRY_RULE)


# ---------------------------------------------------------------------------
# readiness
# ---------------------------------------------------------------------------

class TestReadiness(unittest.TestCase):
    """Registered is not reachable, and blocks cascade to the root cause."""

    def test_nothing_is_runnable_when_nothing_is_confirmed_reachable(self) -> None:
        r = readiness(1, _full_registry(), None)
        self.assertEqual(r.runnable_ids, ())
        self.assertEqual(len(r.blocked_ids), len(STAGES[1].packages))
        self.assertFalse(r.stage_runnable)
        first = r.for_package("s1_substrate_and_reaction")
        self.assertTrue(all(b.reason is BlockReason.SOURCE_NOT_AVAILABLE
                            for b in first.blockers))
        self.assertIn("connectivity-tested", first.blockers[0].detail)

    def test_an_unregistered_source_is_a_different_block(self) -> None:
        reg = _registry_for(["chebi", "rhea"])          # pubchem missing
        r = readiness(1, reg, ["chebi", "rhea"])
        first = r.for_package("s1_substrate_and_reaction")
        reasons = {b.reason for b in first.blockers}
        self.assertIn(BlockReason.SOURCE_NOT_REGISTERED, reasons)
        self.assertIn("pubchem", first.missing_sources)

    def test_the_first_package_runs_once_its_sources_are_available(self) -> None:
        reg = _full_registry()
        r = readiness(1, reg, ["pubchem", "chebi", "rhea"])
        first = r.for_package("s1_substrate_and_reaction")
        self.assertTrue(first.runnable)
        self.assertEqual(first.blockers, ())
        self.assertFalse(r.stage_runnable)

    def test_downstream_blocks_name_the_upstream_package(self) -> None:
        reg = _full_registry()
        r = readiness(1, reg, ["pubchem", "chebi", "rhea", "brenda", "oed",
                               "pubmed", "europe_pmc"])
        seeds = r.for_package("s1_known_activity_seeds")
        self.assertTrue(seeds.blocked)
        reasons = {b.reason for b in seeds.blockers}
        self.assertIn(BlockReason.UPSTREAM_DELIVERABLE_MISSING, reasons)
        self.assertIn("s1_substrate_and_reaction.normalised_substrate",
                      seeds.blocked_on())

        expansion = r.for_package("s1_sequence_family_expansion")
        self.assertTrue(expansion.blocked)
        self.assertTrue(any(b.reason is BlockReason.UPSTREAM_PACKAGE_BLOCKED
                            for b in expansion.blockers))

    def test_completed_packages_unblock_their_consumers(self) -> None:
        reg = _full_registry()
        r = readiness(
            1, reg,
            ["pubchem", "chebi", "rhea", "brenda", "oed", "pubmed",
             "europe_pmc"],
            completed_packages=["s1_substrate_and_reaction"])
        seeds = r.for_package("s1_known_activity_seeds")
        self.assertTrue(seeds.runnable, seeds.blockers)
        self.assertTrue(any("retrobiocat_db" in w for w in seeds.warnings))

    def test_a_partly_completed_upstream_only_unblocks_what_it_delivered(self) -> None:
        reg = _full_registry()
        r = readiness(
            1, reg,
            ["pubchem", "chebi", "rhea", "brenda", "oed", "pubmed",
             "europe_pmc"],
            available_deliverables={
                "s1_substrate_and_reaction": ["normalised_substrate"]})
        seeds = r.for_package("s1_known_activity_seeds")
        self.assertTrue(seeds.blocked)
        self.assertIn("s1_substrate_and_reaction.reaction_direction",
                      seeds.blocked_on())

    def test_stage_two_reports_stage_one_as_the_root_cause(self) -> None:
        reg = _full_registry()
        r = readiness(2, reg, ["fireprotdb", "mavedb", "enzengdb", "skid",
                               "intenzydb", "reactzyme"])
        self.assertEqual(r.runnable_ids, ())
        self.assertIn("s1_sequence_family_expansion", r.upstream_blocked)
        text = r.describe()
        self.assertIn("BLOCKED", text)
        self.assertIn("s2_model_evaluation_gate", text)

    def test_readiness_serialises_without_losing_the_reasons(self) -> None:
        payload = readiness(3, _full_registry(), []).as_dict()
        self.assertEqual(payload["stage"], 3)
        self.assertFalse(payload["stage_runnable"])
        blockers = payload["packages"][0]["blockers"]
        self.assertTrue(blockers)
        self.assertIn("reason", blockers[0])
        self.assertIn("detail", blockers[0])


# ---------------------------------------------------------------------------
# extrapolation
# ---------------------------------------------------------------------------

class TestExtrapolationCheck(unittest.TestCase):
    """High overlap means memorisation, and the marking stays on."""

    def test_high_overlap_recommends_the_uncalibrated_marking(self) -> None:
        train = _training(5)
        # Eight of ten candidates sit in clusters the training rows occupy.
        cands = [_Seq(f"C{i}", f"CACC{i}") for i in range(10)]
        lookup = {f"TACC{i}": f"cluster{i % 2}" for i in range(5)}
        lookup.update({f"CACC{i}": ("cluster0" if i < 8 else f"far{i}")
                       for i in range(10)})

        rep = extrapolation_check(cands, train, lookup)

        self.assertIs(rep.verdict, ExtrapolationVerdict.RETESTING_NEIGHBOURS)
        self.assertEqual(rep.n_resolved, 10)
        self.assertEqual(rep.n_in_training_cluster, 8)
        self.assertAlmostEqual(rep.overlap_fraction or 0.0, 0.8)
        self.assertTrue(rep.mark_model_outputs_uncalibrated)
        self.assertFalse(rep.calibration_possible)
        self.assertIn("memorisation", rep.uncalibrated_reason)
        self.assertTrue(any("marked uncalibrated" in r
                            for r in rep.recommendations))
        self.assertTrue(any("hold out whole clusters" in r
                            for r in rep.recommendations))
        self.assertIn("retesting_near_neighbours", rep.describe())

    def test_low_overlap_is_extrapolation_and_still_uncalibrated(self) -> None:
        train = _training(5)
        cands = [_Seq(f"C{i}", f"CACC{i}") for i in range(10)]
        lookup = {f"TACC{i}": "home" for i in range(5)}
        lookup.update({f"CACC{i}": f"far{i}" for i in range(10)})

        rep = extrapolation_check(cands, train, lookup)

        self.assertIs(rep.verdict, ExtrapolationVerdict.GENUINE_EXTRAPOLATION)
        self.assertEqual(rep.n_in_training_cluster, 0)
        self.assertTrue(rep.mark_model_outputs_uncalibrated)
        self.assertIn("no in-domain data", rep.uncalibrated_reason)

    def test_a_mixed_set_allows_calibration_only_on_the_overlap(self) -> None:
        train = _training(4)
        cands = [_Seq(f"C{i}", f"CACC{i}") for i in range(10)]
        lookup = {f"TACC{i}": "home" for i in range(4)}
        lookup.update({f"CACC{i}": ("home" if i < 3 else f"far{i}")
                       for i in range(10)})

        rep = extrapolation_check(cands, train, lookup)
        self.assertIs(rep.verdict, ExtrapolationVerdict.MIXED)
        self.assertTrue(rep.calibration_possible)
        self.assertTrue(rep.mark_model_outputs_uncalibrated)

    def test_unresolved_clusters_are_never_counted_as_novelty(self) -> None:
        train = _training(3)
        cands = [_Seq(f"C{i}", f"CACC{i}") for i in range(10)]
        lookup = {f"TACC{i}": "home" for i in range(3)}   # candidates unresolved

        rep = extrapolation_check(cands, train, lookup)
        self.assertIs(rep.verdict, ExtrapolationVerdict.UNDETERMINED)
        self.assertEqual(rep.n_unresolved, 10)
        self.assertEqual(rep.n_resolved, 0)
        self.assertIsNone(rep.overlap_fraction)
        self.assertTrue(rep.mark_model_outputs_uncalibrated)
        self.assertIn("not evidence that the candidate is novel",
                      rep.uncalibrated_reason)

    def test_an_identical_sequence_counts_as_overlap_without_a_cluster(self) -> None:
        train = [_Seq("T0", "SHARED")]
        cands = [_Seq("C0", "SHARED")]
        rep = extrapolation_check(cands, train, None)
        self.assertEqual(rep.n_identical_to_training, 1)
        self.assertIn("C0", rep.overlapping_candidate_ids)
        self.assertTrue(any("same sequence as a training row" in r
                            for r in rep.recommendations))

    def test_an_empty_training_set_is_undetermined_not_novel(self) -> None:
        rep = extrapolation_check([_Seq("C0", "A")], [], {"A": "c1"})
        self.assertIs(rep.verdict, ExtrapolationVerdict.UNDETERMINED)
        self.assertIn("undocumented training set",
                      " ".join(rep.recommendations))

    def test_a_callable_cluster_lookup_is_accepted(self) -> None:
        train = _training(2)
        cands = [_Seq("C0", "CACC0")]
        rep = extrapolation_check(cands, train, lambda k: "one")
        self.assertIs(rep.verdict, ExtrapolationVerdict.RETESTING_NEIGHBOURS)


# ---------------------------------------------------------------------------
# stage 2 gate
# ---------------------------------------------------------------------------

@dataclass
class _TrainingRow:
    """A training row carrying a family label and a substrate structure."""

    record_id: str
    accession: str
    family_name: str
    substrate: Any
    max_strength: EvidenceStrength = EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL


@dataclass
class _Sub:
    inchikey: str | None = None
    isomeric_smiles: str | None = None


class TestStage2Gate(unittest.TestCase):

    KEY = "KWOLFJPFCHCOCG-UHFFFAOYSA-N"

    def _rows(self, n: int, family: str, key: str | None) -> list[_TrainingRow]:
        return [_TrainingRow(f"R{i}", f"TACC{i}", family, _Sub(inchikey=key))
                for i in range(n)]

    def test_matching_data_plus_low_overlap_lets_stage_two_proceed(self) -> None:
        train = self._rows(3, "sdr", self.KEY)
        cands = [_Seq(f"C{i}", f"CACC{i}") for i in range(6)]
        lookup = {f"TACC{i}": "home" for i in range(3)}
        lookup.update({f"CACC{i}": f"far{i}" for i in range(6)})

        ans = answer_stage2_gate(
            parent_families=["SDR"], substrate_keys=[self.KEY],
            candidate_set=cands, training_records=train,
            cluster_lookup=lookup)

        self.assertTrue(ans.has_matching_experimental_data)
        self.assertTrue(ans.proceed)
        self.assertEqual(ans.n_matching_records, 3)
        self.assertEqual(ans.n_matching_sequence_level, 3)
        self.assertIs(ans.extrapolation.verdict,
                      ExtrapolationVerdict.GENUINE_EXTRAPOLATION)
        self.assertTrue(ans.extrapolation.mark_model_outputs_uncalibrated)

    def test_family_match_without_substrate_match_does_not_pass(self) -> None:
        train = self._rows(5, "sdr", "OTHERKEY-UHFFFAOYSA-N")
        ans = answer_stage2_gate(
            parent_families=["SDR"], substrate_keys=[self.KEY],
            candidate_set=[_Seq("C0", "A")], training_records=train,
            cluster_lookup={"A": "c"})
        self.assertFalse(ans.proceed)
        self.assertEqual(ans.matched_on_family, 5)
        self.assertEqual(ans.matched_on_substrate, 0)
        self.assertEqual(ans.n_matching_records, 0)
        self.assertTrue(any("different question" in r for r in ans.reasons))

    def test_a_missing_parent_family_is_unmet_not_passed(self) -> None:
        ans = answer_stage2_gate(
            parent_families=[], substrate_keys=[self.KEY],
            candidate_set=[], training_records=self._rows(2, "sdr", self.KEY))
        self.assertFalse(ans.proceed)
        self.assertTrue(any("could not be evaluated" in r for r in ans.reasons))

    def test_high_overlap_is_reported_even_when_data_matches(self) -> None:
        train = self._rows(4, "sdr", self.KEY)
        cands = [_Seq(f"C{i}", f"CACC{i}") for i in range(4)]
        lookup = {f"TACC{i}": "home" for i in range(4)}
        lookup.update({f"CACC{i}": "home" for i in range(4)})
        ans = answer_stage2_gate(
            parent_families=["sdr"], substrate_keys=[self.KEY],
            candidate_set=cands, training_records=train, cluster_lookup=lookup)
        self.assertTrue(ans.proceed)
        self.assertIs(ans.extrapolation.verdict,
                      ExtrapolationVerdict.RETESTING_NEIGHBOURS)
        self.assertIn("retesting_near_neighbours", ans.describe())


# ---------------------------------------------------------------------------
# novelty budget
# ---------------------------------------------------------------------------

class TestNoveltyBudget(unittest.TestCase):

    def _mgnify(self, n: int) -> list[_Seq]:
        return [_Seq(f"M{i}", f"MACC{i}", source_id="mgnify_proteins")
                for i in range(n)]

    def _characterised(self, n: int) -> list[_Seq]:
        ev = [EvidenceRef(source_type="publication", identifier="PMID:1",
                          strength=EvidenceStrength.HOMOLOG_EXPERIMENTAL)]
        return [_Seq(f"K{i}", f"KACC{i}", source_id="brenda", evidence=list(ev))
                for i in range(n)]

    def test_a_small_exploration_slot_is_within_budget(self) -> None:
        batch = self._characterised(90) + self._mgnify(6)
        v = check_novelty_budget(batch)
        self.assertTrue(v.within_budget)
        self.assertEqual(v.n_exploratory, 6)
        self.assertEqual(v.n_exploratory_without_evidence, 6)
        self.assertEqual(v.allowance, 8)

    def test_a_whole_round_of_distant_sequences_is_refused(self) -> None:
        batch = self._mgnify(96)
        v = check_novelty_budget(batch)
        self.assertFalse(v.within_budget)
        self.assertEqual(v.over_by, 96 - v.allowance)
        self.assertIn(MGNIFY_EARLY_ENTRY_RULE, v.reasons)
        self.assertEqual(len(v.offending_ids), 96)
        self.assertIn("OVER the exploration budget", v.describe())

    def test_a_tiny_round_carries_no_exploration_slot(self) -> None:
        v = check_novelty_budget(self._mgnify(4) + self._characterised(4))
        self.assertEqual(v.allowance, 0)
        self.assertFalse(v.within_budget)
        self.assertTrue(any("too small to carry an exploration slot" in r
                            for r in v.reasons))

    def test_distant_candidates_with_functional_evidence_do_not_count(self) -> None:
        ev = [EvidenceRef(source_type="publication", identifier="PMID:2",
                          strength=EvidenceStrength.HOMOLOG_EXPERIMENTAL)]
        batch = self._characterised(80) + [
            _Seq(f"M{i}", f"MACC{i}", source_id="mgnify_proteins",
                 evidence=list(ev)) for i in range(16)]
        v = check_novelty_budget(batch)
        self.assertEqual(v.n_exploratory, 16)
        self.assertEqual(v.n_exploratory_without_evidence, 0)
        self.assertTrue(v.within_budget)

    def test_the_policy_is_overridable_and_says_so(self) -> None:
        pol = NoveltyBudgetPolicy(max_fraction=0.25, max_count=24,
                                  min_round_size_for_exploration=8)
        self.assertEqual(pol.allowance(96), 24)
        self.assertEqual(pol.allowance(4), 0)
        v = check_novelty_budget(self._characterised(72) + self._mgnify(24),
                                 policy=pol)
        self.assertTrue(v.within_budget)
        self.assertIn("project policy", NoveltyBudgetPolicy.__doc__ or "")


# ---------------------------------------------------------------------------
# the plan against the real registry, when it is present
# ---------------------------------------------------------------------------

class TestAgainstRealRegistry(unittest.TestCase):

    def test_plan_source_ids_exist_in_the_shipped_registry(self) -> None:
        try:
            reg = SourceRegistry.from_directory()
        except Exception as exc:  # pragma: no cover - registry may be absent
            self.skipTest(f"shipped datasource registry unavailable: {exc}")
        self.assertEqual(
            validate_plan(reg), [],
            "the plan names sources the shipped registry does not define")


if __name__ == "__main__":  # pragma: no cover - pytest may not be installed
    unittest.main(verbosity=2)
