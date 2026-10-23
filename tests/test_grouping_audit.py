"""Tests for the fold-grouping audit and the single-linkage grouping it checks.

The audit's whole value is the precision of what it claims, so most of what is
tested here is the *limits*: that it refuses a single-fold pool, that it
refuses a screen that reaches outside the cross-fold relation, that it counts
the pairs it did not look at, and that it says so in words. A test that only
checked "it finds the planted breach" would pass just as well on a version
that claimed the grouping was clean.
"""

from __future__ import annotations

import itertools
import unittest
from pathlib import Path

from click.testing import CliRunner

from eagent.cli import main
from eagent.eval.retrospective import (
    AUDIT_SCREEN,
    GROUPING_SCREEN,
    PINNED_GROUPING,
    BenchmarkProtocol,
    GroupingPinError,
    RetrospectiveError,
    grouping_input_digest,
    load_pinned_grouping,
    run_benchmark,
    run_sdr_suite,
)

from eagent.eval.grouping_audit import (
    GroupingAudit,
    GroupingAuditError,
    COMPLETE_GROUPING_NOTE,
    SCREEN_CAVEAT,
    audit_grouping,
    jaccard_screen,
    top_k_similarity_screen,
)
from eagent.tools.mine_sequences import (
    PAIRWISE_IDENTITY_DEFINITION,
    cluster_sequences,
    pairwise_identity,
    single_linkage_clusters,
)


def exact_identity(a: str, b: str) -> float:
    """A stand-in for the aligner: 1.0 for equal strings, 0.0 otherwise.

    Used wherever the test is about the audit's bookkeeping rather than about
    sequence identity, so that the expected answer is unarguable.
    """
    return 1.0 if a == b else 0.0


def everything() -> object:
    """A screen that admits every cross-fold pair."""
    from eagent.eval.grouping_audit import CandidateScreen, _cross_fold_pairs

    return CandidateScreen(
        name="all", description="every cross-fold pair",
        select=lambda seqs, folds: set(_cross_fold_pairs(seqs, folds)))


class RefusalTests(unittest.TestCase):
    """What the audit will not do, and why each refusal matters."""

    def setUp(self) -> None:
        self.seqs = {"a": "AAAA", "b": "CCCC", "c": "DDDD"}
        self.folds = {"a": 0, "b": 1, "c": 1}

    def _audit(self, **kw):
        base = dict(identity=exact_identity, threshold=0.4,
                    screens=[everything()],
                    identity_name="exact string equality")
        base.update(kw)
        return audit_grouping(self.seqs, self.folds, **base)

    def test_no_screen_is_refused_rather_than_silently_complete(self) -> None:
        # The complete audit is a different, far more expensive function. If a
        # caller passes no screen, the dangerous thing is to quietly align
        # everything; the honest thing is to make them name the screen.
        with self.assertRaises(GroupingAuditError) as caught:
            self._audit(screens=[])
        self.assertIn("screen", str(caught.exception))

    def test_a_single_fold_is_refused(self) -> None:
        with self.assertRaises(GroupingAuditError) as caught:
            audit_grouping(self.seqs, {k: 0 for k in self.seqs},
                           identity=exact_identity, threshold=0.4,
                           screens=[everything()], identity_name="x")
        self.assertIn("cross-fold", str(caught.exception))

    def test_a_sequence_with_no_fold_is_refused(self) -> None:
        with self.assertRaises(GroupingAuditError) as caught:
            audit_grouping(self.seqs, {"a": 0, "b": 1},
                           identity=exact_identity, threshold=0.4,
                           screens=[everything()], identity_name="x")
        self.assertIn("no fold", str(caught.exception))

    def test_an_empty_pool_is_refused(self) -> None:
        with self.assertRaises(GroupingAuditError):
            audit_grouping({}, {}, identity=exact_identity, threshold=0.4,
                           screens=[everything()], identity_name="x")

    def test_a_screen_may_not_widen_the_relation_under_audit(self) -> None:
        # A screen that nominated a within-fold pair would quietly change the
        # claim: within-fold similarity is not leakage, so a "breach" found
        # there would be reported as one that isn't.
        from eagent.eval.grouping_audit import CandidateScreen

        rogue = CandidateScreen(
            name="rogue", description="also within-fold pairs",
            select=lambda seqs, folds: {("b", "c")})
        with self.assertRaises(GroupingAuditError) as caught:
            self._audit(screens=[rogue])
        self.assertIn("cross-fold pair of this pool", str(caught.exception))

    def test_a_screen_may_not_nominate_an_unknown_sequence(self) -> None:
        from eagent.eval.grouping_audit import CandidateScreen

        rogue = CandidateScreen(
            name="rogue", description="a pair from somewhere else",
            select=lambda seqs, folds: {("a", "zzz")})
        with self.assertRaises(GroupingAuditError):
            self._audit(screens=[rogue])

    def test_a_jaccard_screen_outside_zero_to_one_is_refused(self) -> None:
        with self.assertRaises(GroupingAuditError):
            jaccard_screen(1.4)

    def test_a_top_k_screen_of_zero_is_refused(self) -> None:
        with self.assertRaises(GroupingAuditError):
            top_k_similarity_screen(lambda a, b: 0.0, 0, name="n",
                                    description="d")


class BookkeepingTests(unittest.TestCase):
    """The counts, which are what distinguish a lower bound from a claim."""

    def setUp(self) -> None:
        # Four sequences, two folds, so four cross-fold pairs: a-c, a-d, b-c,
        # b-d. 'a' and 'c' are the same sequence, which is the planted breach.
        self.seqs = {"a": "AAAA", "b": "BBBB", "c": "AAAA", "d": "DDDD"}
        self.folds = {"a": 0, "b": 0, "c": 1, "d": 1}

    def _audit(self, screens):
        return audit_grouping(
            self.seqs, self.folds, identity=exact_identity, threshold=0.4,
            screens=screens, identity_name="exact string equality",
            clustering_identity_name="exact string equality")

    def test_every_cross_fold_pair_is_counted_whether_or_not_it_is_aligned(self) -> None:
        only_ac = self._audit([_only({("a", "c")})])
        self.assertEqual(only_ac.n_cross_fold_pairs, 4)
        self.assertEqual(only_ac.n_pairs_aligned, 1)
        self.assertEqual(only_ac.n_pairs_unscreened, 3)
        self.assertAlmostEqual(only_ac.to_dict()["fraction_of_pairs_aligned"],
                               0.25)

    def test_within_fold_pairs_are_not_cross_fold_pairs(self) -> None:
        # 6 pairs in all, 2 of them within a fold.
        self.assertEqual(self._audit([everything()]).n_cross_fold_pairs, 4)
        self.assertEqual(len(list(itertools.combinations(self.seqs, 2))), 6)

    def test_the_planted_breach_is_found_and_attributed_to_its_screen(self) -> None:
        audit = self._audit([_only({("a", "c")})])
        self.assertEqual(len(audit.breaches), 1)
        breach = audit.breaches[0]
        self.assertEqual((breach.example_id, breach.neighbour_id), ("a", "c"))
        self.assertEqual((breach.fold, breach.neighbour_fold), (0, 1))
        self.assertEqual(breach.screens, ("only",))
        self.assertFalse(audit.clean_among_checked)

    def test_a_breach_the_screen_skipped_is_not_found_and_not_claimed_absent(self) -> None:
        # The audit looks only at a-d, so it reports nothing -- and reports
        # that it looked at one pair out of four. The absence is not a clean
        # bill of health and the fields say which it is.
        audit = self._audit([_only({("a", "d")})])
        self.assertEqual(audit.breaches, [])
        self.assertTrue(audit.clean_among_checked)
        self.assertEqual(audit.n_pairs_unscreened, 3)
        self.assertIn("LOWER BOUND", audit.caveat)
        self.assertIn("LOWER BOUND", "\n".join(audit.render()))

    def test_a_sequence_breaching_twice_counts_once_in_the_sequence_count(self) -> None:
        seqs = {"a": "AAAA", "b": "AAAA", "c": "AAAA"}
        folds = {"a": 0, "b": 1, "c": 2}
        audit = audit_grouping(seqs, folds, identity=exact_identity,
                               threshold=0.4, screens=[everything()],
                               identity_name="x")
        self.assertEqual(len(audit.breaches), 3)       # every pair
        self.assertEqual(audit.n_breaching_sequences, 3)

    def test_a_breach_counts_both_members_as_breaching(self) -> None:
        # Leakage is symmetric: in the fold where 'a' is held out, 'c' is in
        # its training set, and in the fold where 'c' is held out, 'a' is.
        audit = self._audit([_only({("a", "c")})])
        self.assertEqual(audit.n_breaching_sequences, 2)

    def test_two_screens_nominating_the_same_pair_align_it_once(self) -> None:
        audit = self._audit([_only({("a", "c")}, name="first"),
                             _only({("a", "c")}, name="second")])
        self.assertEqual(audit.n_pairs_aligned, 1)
        self.assertEqual(audit.breaches[0].screens, ("first", "second"))
        rows = {s["name"]: s for s in audit.screens}
        self.assertEqual(rows["first"]["n_breaches"], 1)
        self.assertEqual(rows["second"]["n_breaches"], 1)

    def test_a_screen_that_found_nothing_is_reported_as_having_found_nothing(self) -> None:
        audit = self._audit([_only({("a", "c")}, name="useful"),
                             _only({("b", "d")}, name="useless")])
        rows = {s["name"]: s for s in audit.screens}
        self.assertEqual(rows["useful"]["n_breaches"], 1)
        self.assertEqual(rows["useless"]["n_admitted"], 1)
        self.assertEqual(rows["useless"]["n_breaches"], 0)

    def test_near_misses_are_counted_separately_from_breaches(self) -> None:
        # identity 0.37 against a threshold of 0.4 and a band of 0.05.
        audit = audit_grouping(
            self.seqs, self.folds, identity=lambda a, b: 0.37, threshold=0.4,
            screens=[everything()], identity_name="x", near_miss_band=0.05)
        self.assertEqual(audit.breaches, [])
        self.assertEqual(audit.n_near_misses, 4)

    def test_identity_exactly_at_the_threshold_is_a_breach(self) -> None:
        # The clustering uses >=, so the audit must too, or the two disagree
        # about the boundary they are both talking about.
        audit = audit_grouping(
            self.seqs, self.folds, identity=lambda a, b: 0.4, threshold=0.4,
            screens=[everything()], identity_name="x")
        self.assertEqual(len(audit.breaches), 4)

    def test_the_quantiles_are_observed_values(self) -> None:
        audit = audit_grouping(
            self.seqs, self.folds,
            identity=lambda a, b: 0.1 if a == b else 0.2, threshold=0.9,
            screens=[everything()], identity_name="x")
        for value in audit.identity_quantiles.values():
            self.assertIn(value, (0.1, 0.2))

    def test_the_fold_count_is_the_number_of_distinct_folds(self) -> None:
        self.assertEqual(self._audit([everything()]).n_folds, 2)


class DefinitionAgreementTests(unittest.TestCase):
    """An audit using a different identity definition says so."""

    def _audit(self, clustering_name):
        return audit_grouping(
            {"a": "AAAA", "b": "CCCC"}, {"a": 0, "b": 1},
            identity=exact_identity, threshold=0.4, screens=[everything()],
            identity_name="exact string equality",
            clustering_identity_name=clustering_name)

    def test_agreement_is_reported_when_the_definitions_match(self) -> None:
        audit = self._audit("exact string equality")
        self.assertIs(audit.definitions_agree, True)
        self.assertNotIn("WARNING", "\n".join(audit.render()))

    def test_a_mismatch_is_warned_about_in_the_rendered_report(self) -> None:
        audit = self._audit("mmseqs2 --min-seq-id over the aligned region")
        self.assertIs(audit.definitions_agree, False)
        rendered = "\n".join(audit.render())
        self.assertIn("WARNING", rendered)
        self.assertIn("second opinion", rendered)

    def test_an_unknown_clustering_definition_claims_neither(self) -> None:
        audit = self._audit(None)
        self.assertIsNone(audit.definitions_agree)
        self.assertNotIn("WARNING", "\n".join(audit.render()))


class ScreenTests(unittest.TestCase):
    """The two shipped screens, on sequences whose answer is known."""

    def test_the_jaccard_screen_admits_similar_pairs_and_drops_the_rest(self) -> None:
        seqs = {"a": "MKVLAAGIVG" * 8, "b": "MKVLAAGIVG" * 8,
                "c": "WWYYWWYYWW" * 8}
        folds = {"a": 0, "b": 1, "c": 1}
        admitted = jaccard_screen(0.5).select(seqs, folds)
        self.assertEqual(admitted, {("a", "b")})

    def test_a_sequence_shorter_than_k_is_admitted_rather_than_screened_out(self) -> None:
        # kmer_set returns an empty set, jaccard_distance returns None. An
        # empty signature is "nothing was measured", and screening a pair out
        # on the strength of nothing is exactly the silent miss this module is
        # for.
        seqs = {"a": "AA", "b": "MKVLAAGIVG" * 8}
        admitted = jaccard_screen(0.9).select(seqs, {"a": 0, "b": 1})
        self.assertEqual(admitted, {("a", "b")})

    def test_the_top_k_screen_takes_k_per_sequence_from_other_folds_only(self) -> None:
        seqs = {"a": "A", "b": "B", "c": "C", "d": "D"}
        folds = {"a": 0, "b": 0, "c": 1, "d": 1}
        rank = {"A": 0.0, "B": 0.1, "C": 0.9, "D": 0.5}
        screen = top_k_similarity_screen(
            lambda x, y: rank[y], 1, name="top1", description="d")
        # a and b each pick c (0.9); c and d each pick b (0.1 > 0.0).
        self.assertEqual(screen.select(seqs, folds),
                         {("a", "c"), ("b", "c"), ("b", "d")})

    def test_the_top_k_screen_breaks_ties_by_id_so_the_cut_is_deterministic(self) -> None:
        seqs = {"a": "A", "b": "B", "c": "C"}
        folds = {"a": 0, "b": 1, "c": 1}
        screen = top_k_similarity_screen(
            lambda x, y: 0.5, 1, name="top1", description="d")
        first = screen.select(seqs, folds)
        self.assertEqual(first, screen.select(seqs, folds))
        self.assertIn(("a", "b"), first)       # 'b' sorts before 'c'

    def test_the_screens_name_their_own_blind_spot(self) -> None:
        # The description is printed verbatim in the report, and it is the only
        # place a reader learns what the audit could not see.
        self.assertIn("not a bound", jaccard_screen(0.1).description)
        self.assertIn("0.1", jaccard_screen(0.1).name)


class SingleLinkageTests(unittest.TestCase):
    """The grouping the audit exists to check."""

    def test_a_chain_lands_in_one_component_even_where_the_ends_are_far(self) -> None:
        # A and C share nothing; both are close to B. For a leakage control
        # they must share a fold, because B's fold cannot hold a neighbour of
        # either without also holding one of the other.
        a = "M" + "AKLVGTSEDA" * 12
        b = "M" + "AKLVGTSEDA" * 6 + "WYWYCWYWYC" * 6
        c = "M" + "WYWYCWYWYC" * 12
        report = single_linkage_clusters(
            [("a", a), ("b", b), ("c", c)], 0.4, kmer_prefilter=None)
        self.assertEqual(len(report.clusters), 1)
        self.assertLess(pairwise_identity(a, c), 0.4)
        self.assertGreaterEqual(pairwise_identity(a, b), 0.4)

    def test_greedy_clustering_splits_that_chain_which_is_why_this_exists(self) -> None:
        a = "M" + "AKLVGTSEDA" * 12
        b = "M" + "AKLVGTSEDA" * 6 + "WYWYCWYWYC" * 6
        c = "M" + "WYWYCWYWYC" * 12
        items = [("a", a), ("b", b), ("c", c)]
        greedy = cluster_sequences(items, 0.4, kmer_prefilter=None)
        single = single_linkage_clusters(items, 0.4, kmer_prefilter=None)
        self.assertGreater(len(greedy.clusters), len(single.clusters))

    def test_unrelated_sequences_stay_in_their_own_components(self) -> None:
        report = single_linkage_clusters(
            [("a", "AKLVGTSEDA" * 12), ("b", "WYWYCWYWYC" * 12)], 0.4,
            kmer_prefilter=None)
        self.assertEqual(len(report.clusters), 2)

    def test_every_input_appears_in_the_assignment_exactly_once(self) -> None:
        items = [(f"s{i}", "AKLVGTSEDA" * (i + 2)) for i in range(6)]
        report = single_linkage_clusters(items, 0.4)
        self.assertEqual(sorted(report.assignment), sorted(k for k, _ in items))
        members = [m for c in report.clusters for m in c.members]
        self.assertEqual(sorted(members), sorted(report.assignment))

    def test_the_method_and_its_identity_definition_travel_with_the_report(self) -> None:
        report = single_linkage_clusters([("a", "AAAAAAAAAA" * 4)], 0.4)
        self.assertEqual(report.method, "single_linkage_pure_python")
        self.assertEqual(report.identity_definition,
                         PAIRWISE_IDENTITY_DEFINITION)

    def test_screened_out_pairs_are_counted_not_hidden(self) -> None:
        report = single_linkage_clusters(
            [("a", "AKLVGTSEDA" * 12), ("b", "WYWYCWYWYC" * 12)], 0.4,
            kmer_prefilter=0.9)
        self.assertEqual(report.n_alignments, 0)
        self.assertEqual(report.n_prefiltered_pairs, 1)

    def test_an_exhausted_budget_says_the_grouping_may_leak(self) -> None:
        # An unfound link leaves two components that should be one, which is
        # leakage rather than a coarser grain, so it must not read as a
        # performance note.
        items = [(f"s{i}", "AKLVGTSEDA" * 12) for i in range(5)]
        report = single_linkage_clusters(items, 0.4, kmer_prefilter=None,
                                        max_alignments=1)
        self.assertTrue(report.budget_exhausted)
        self.assertIn("leak", report.fallback_reason)

    def test_one_key_naming_two_sequences_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            single_linkage_clusters([("a", "AKLVGTSEDA" * 4),
                                     ("a", "WYWYCWYWYC" * 4)], 0.4)
        self.assertIn("'a'", str(caught.exception))

    def test_the_same_pair_twice_is_not_a_duplicate_key(self) -> None:
        report = single_linkage_clusters([("a", "AKLVGTSEDA" * 4),
                                          ("a", "AKLVGTSEDA" * 4)], 0.4)
        self.assertEqual(sorted(report.assignment), ["a"])

    def test_an_impossible_threshold_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            single_linkage_clusters([("a", "AAAA")], 0.0)
        with self.assertRaises(ValueError):
            single_linkage_clusters([("a", "AAAA")], 1.5)

    def test_the_ordering_of_the_input_does_not_change_the_components(self) -> None:
        items = [(f"s{i}", "AKLVGTSEDA" * (i + 4)) for i in range(6)]
        forward = single_linkage_clusters(items, 0.4, kmer_prefilter=None)
        backward = single_linkage_clusters(list(reversed(items)), 0.4,
                                           kmer_prefilter=None)
        def groups(report):
            return sorted(tuple(sorted(c.members)) for c in report.clusters)
        self.assertEqual(groups(forward), groups(backward))


class TheGuaranteeTests(unittest.TestCase):
    """Single linkage's one promise, checked by the audit on the same pool.

    This is the pair of tests that matter: the grouping claims every
    cross-component pair is below the threshold among the pairs the screen
    admitted, and the audit is the thing that would catch it if that were
    false. Running them against each other on a pool with a known chain is the
    closest this suite gets to an end-to-end check of the control.
    """

    def setUp(self) -> None:
        # Nine sequences in three independent chains of three, each chain's
        # ends related only through its middle and no block shared between
        # chains -- so single linkage must give three components and greedy
        # must give more.
        chains = (("AKLVGTSEDA", "WYWYCWYWYC"),
                  ("PQRSTNHIMF", "GGCCDDEEFF"),
                  ("HHIIKKLLMM", "NNPPQQRRSS"))
        self.items = []
        for n, (left, right) in enumerate(chains):
            self.items += [
                (f"c{n}a", "M" + left * 12),
                (f"c{n}b", "M" + left * 6 + right * 6),
                (f"c{n}c", "M" + right * 12)]
        self.seqs = dict(self.items)

    def _folds(self, report):
        # One fold per component, round-robin, so that any component the
        # clustering failed to merge is split across folds and the audit has
        # something to find.
        order = sorted(report.clusters, key=lambda c: c.cluster_id)
        return {m: n % 2 for n, c in enumerate(order) for m in c.members}

    def test_the_audit_finds_no_breach_in_a_single_linkage_grouping(self) -> None:
        report = single_linkage_clusters(self.items, 0.4, kmer_prefilter=None)
        self.assertEqual(len(report.clusters), 3, "fixture is not three chains")
        folds = self._folds(report)
        audit = audit_grouping(
            self.seqs, folds, identity=pairwise_identity, threshold=0.4,
            screens=[everything()], identity_name=PAIRWISE_IDENTITY_DEFINITION,
            clustering_identity_name=report.identity_definition)
        self.assertEqual(audit.breaches, [], "\n".join(audit.render()))
        # With no screen dropping anything, this one time the audit is
        # complete and the caveat's "pairs checked" is every pair.
        self.assertEqual(audit.n_pairs_unscreened, 0)

    def test_the_audit_finds_the_breach_a_greedy_grouping_leaves(self) -> None:
        report = cluster_sequences(self.items, 0.4, kmer_prefilter=None)
        folds = self._folds(report)
        audit = audit_grouping(
            self.seqs, folds, identity=pairwise_identity, threshold=0.4,
            screens=[everything()], identity_name=PAIRWISE_IDENTITY_DEFINITION,
            clustering_identity_name=report.identity_definition)
        self.assertTrue(audit.breaches,
                        "the greedy grouping split a chain and the audit "
                        "missed it, which would make the audit useless")
        self.assertFalse(audit.clean_among_checked)
        self.assertIn("BREACH", "\n".join(audit.render()))


class WhatACleanResultMeansTests(unittest.TestCase):
    """A clean audit means two different things, and must say which.

    Against a screened grouping it is a lower bound: the pairs no screen
    admitted were never looked at. Against a complete grouping it cannot be
    anything else, because the construction already settled every pair -- so
    presenting it as reassurance about the design would be presenting a
    tautology as evidence.
    """

    SEQS = {"a": "AAAA", "b": "CCCC", "c": "DDDD"}
    FOLDS = {"a": 0, "b": 1, "c": 1}

    def _audit(self, complete):
        return audit_grouping(
            self.SEQS, self.FOLDS, identity=lambda x, y: 0.0, threshold=0.4,
            screens=[everything()], identity_name="x",
            grouping_was_complete=complete)

    def test_a_screened_grouping_gets_the_lower_bound_caveat(self) -> None:
        audit = self._audit(False)
        self.assertEqual(audit.caveat, SCREEN_CAVEAT)
        self.assertIn("LOWER BOUND", "\n".join(audit.render()))

    def test_an_unknown_grouping_gets_the_lower_bound_caveat(self) -> None:
        self.assertEqual(self._audit(None).caveat, SCREEN_CAVEAT)

    def test_a_complete_grouping_is_told_it_cannot_discover_a_leak(self) -> None:
        audit = self._audit(True)
        self.assertEqual(audit.caveat, COMPLETE_GROUPING_NOTE)
        rendered = "\n".join(audit.render())
        self.assertIn("cannot discover a leak", rendered)
        self.assertIn("check of the clustering code", rendered)
        self.assertNotIn("LOWER BOUND", rendered)

    def test_the_completeness_flag_survives_the_dict(self) -> None:
        self.assertIs(self._audit(True).to_dict()["grouping_was_complete"],
                      True)

    def test_the_flag_changes_nothing_about_the_measurement(self) -> None:
        screened, complete = self._audit(False), self._audit(True)
        for field_name in ("n_pairs_aligned", "n_cross_fold_pairs",
                           "identity_quantiles", "n_breaching_sequences"):
            self.assertEqual(getattr(screened, field_name),
                             getattr(complete, field_name))


class RenderTests(unittest.TestCase):
    """The report lines, which are what anyone actually reads."""

    def _audit(self, identity):
        return audit_grouping(
            {"a": "AAAA", "b": "CCCC", "c": "AAAA"}, {"a": 0, "b": 1, "c": 1},
            identity=identity, threshold=0.4, screens=[everything()],
            identity_name="exact string equality")

    def test_a_clean_run_still_prints_the_caveat_and_the_unchecked_count(self) -> None:
        rendered = "\n".join(self._audit(lambda a, b: 0.0).render())
        self.assertIn("no aligned cross-fold pair reached 0.4", rendered)
        self.assertIn(SCREEN_CAVEAT, rendered)
        self.assertIn("admitted by no screen", rendered)

    def test_a_breach_is_stated_as_affecting_every_number_in_the_report(self) -> None:
        rendered = "\n".join(self._audit(exact_identity).render())
        self.assertIn("BREACH", rendered)
        self.assertIn("optimistic", rendered)

    def test_only_the_ten_worst_breaches_are_listed_and_the_rest_counted(self) -> None:
        seqs = {f"s{i}": "AAAA" for i in range(8)}
        folds = {f"s{i}": i % 2 for i in range(8)}
        audit = audit_grouping(seqs, folds, identity=exact_identity,
                               threshold=0.4, screens=[everything()],
                               identity_name="x")
        self.assertEqual(len(audit.breaches), 16)
        rendered = "\n".join(audit.render())
        self.assertIn("and 6 more", rendered)

    def test_an_empty_audit_renders_nothing_rather_than_a_header(self) -> None:
        self.assertEqual(GroupingAudit(
            threshold=0.4, identity_name="x", n_sequences=0, n_folds=0,
            n_cross_fold_pairs=0, n_pairs_aligned=0).render()[1],
            "FOLD-GROUPING AUDIT (the clustering's own identity function, not "
            "the k-mer proxy the report otherwise uses):")

    def test_the_dict_round_trips_through_json(self) -> None:
        import json

        audit = self._audit(exact_identity)
        reloaded = json.loads(json.dumps(audit.to_dict()))
        self.assertEqual(reloaded["n_breaching_pairs"], 1)
        self.assertFalse(reloaded["clean_among_checked"])
        self.assertEqual(reloaded["breaches"][0]["admitted_by"], ["all"])


class TheBenchmarkWiringTests(unittest.TestCase):
    """The suite's own use of the audit and the grouping it audits."""

    FIXTURE = Path(__file__).parent / "fixtures" / "sdr"

    def test_the_report_records_the_fold_assignment_it_used(self) -> None:
        # Without this the run is not reproducible from the report, and the
        # audit is a claim about a fold map nobody can see.
        examples, clusters = _tiny_families()
        report = run_benchmark(
            examples, BenchmarkProtocol(n_folds=3, n_boot=20, min_per_class=3),
            cluster_assignment=clusters)
        self.assertEqual(sorted(report.fold_assignment),
                         sorted({e.example_id for e in examples}))
        self.assertEqual(set(report.fold_assignment.values()), {0, 1, 2})
        self.assertEqual(report.to_dict()["fold_assignment"],
                         report.fold_assignment)

    def test_an_unknown_grouping_is_refused_by_name(self) -> None:
        with self.assertRaises(RetrospectiveError) as caught:
            run_sdr_suite(self.FIXTURE,
                          BenchmarkProtocol(grouping="whatever"),
                          sweep=(), audit_screen=None)
        self.assertIn("whatever", str(caught.exception))
        self.assertIn("pinned", str(caught.exception))

    def test_auditing_a_recomputed_grouping_at_its_own_screen_is_refused(self) -> None:
        # An audit at the screen the grouping was built at reports a clean
        # result whatever the truth is, so the CLI refuses rather than
        # printing a reassuring tautology.
        result = CliRunner().invoke(main, [
            "benchmark", "sdr", "--data", str(self.FIXTURE),
            "--grouping", "single_linkage",
            "--audit-screen", str(GROUPING_SCREEN)])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("clean result whatever the truth is", result.output)
        self.assertIn("--no-audit", result.output)

    def test_the_pinned_grouping_has_no_screen_to_be_tautological_against(self) -> None:
        # It aligned every pair, so no set of pairs was ruled out by
        # assumption and any audit screen says something about it. The guard
        # must not fire here, so the run gets past it -- whatever it then does.
        result = CliRunner().invoke(main, [
            "benchmark", "sdr", "--data", str(self.FIXTURE),
            "--grouping", "pinned", "--audit-screen", str(GROUPING_SCREEN)])
        self.assertNotIn("clean result whatever the truth is", result.output)

    def test_the_shipped_audit_screen_is_below_the_shipped_grouping_screen(self) -> None:
        self.assertLess(AUDIT_SCREEN, GROUPING_SCREEN)

    def test_the_pinned_grouping_is_the_default(self) -> None:
        self.assertEqual(BenchmarkProtocol().grouping, "pinned")


class ThePinnedGroupingTests(unittest.TestCase):
    """Loading the checked-in grouping, and every way it is refused.

    All five refusals matter for the same reason: the alternative to refusing
    is running the benchmark against a grouping that does not establish what
    the folds promise, and reporting the resulting optimism as a result.
    """

    ITEMS = [("a", "MKVLAAGIVG" * 4), ("b", "MWYWYCWYWY" * 4)]
    THRESHOLD = 0.4

    def _pin(self, **overrides):
        payload = {
            "schema": "eagent.sdr_grouping.v1",
            "input_digest": grouping_input_digest(self.ITEMS, self.THRESHOLD),
            "identity_threshold": self.THRESHOLD,
            "method": "single_linkage_pure_python",
            "identity_definition": PAIRWISE_IDENTITY_DEFINITION,
            "screen": None, "complete": True,
            "n_alignments": 1, "n_prefiltered_pairs": 0, "n_clusters": 2,
            "assignment": {"a": "clu_00001", "b": "clu_00002"},
        }
        payload.update(overrides)
        import json
        import tempfile
        path = Path(tempfile.mkdtemp()) / "grouping.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _load(self, path, items=None, threshold=None):
        return load_pinned_grouping(items or self.ITEMS,
                                    self.THRESHOLD if threshold is None
                                    else threshold, path)

    def test_a_matching_pin_loads(self) -> None:
        pinned = self._load(self._pin())
        self.assertEqual(pinned["n_clusters"], 2)
        self.assertEqual(sorted(pinned["assignment"]), ["a", "b"])

    def test_an_absent_pin_is_refused_with_the_command_that_makes_one(self) -> None:
        with self.assertRaises(GroupingPinError) as caught:
            self._load(Path("/nonexistent/grouping.json"))
        self.assertIn("pin_sdr_grouping.py", str(caught.exception))

    def test_a_pin_computed_from_other_sequences_is_refused(self) -> None:
        other = [("a", "MKVLAAGIVG" * 4), ("b", "MPQRSTNHIM" * 4)]
        with self.assertRaises(GroupingPinError) as caught:
            self._load(self._pin(), items=other)
        self.assertIn("different input", str(caught.exception))

    def test_a_pin_computed_at_another_threshold_is_refused(self) -> None:
        with self.assertRaises(GroupingPinError) as caught:
            self._load(self._pin(), threshold=0.9)
        self.assertIn("different input", str(caught.exception))

    def test_a_pin_computed_behind_a_screen_is_refused(self) -> None:
        # The point of pinning is that the expensive complete computation ran.
        # A screened pin would be cheap and would not establish the property,
        # so accepting it would make the pin decorative.
        with self.assertRaises(GroupingPinError) as caught:
            self._load(self._pin(screen=0.1, complete=False))
        self.assertIn("behind a screen", str(caught.exception))

    def test_a_pin_claiming_completeness_while_naming_a_screen_is_refused(self) -> None:
        with self.assertRaises(GroupingPinError):
            self._load(self._pin(screen=0.1, complete=True))

    def test_a_pin_missing_a_sequence_is_refused_with_the_counts(self) -> None:
        with self.assertRaises(GroupingPinError) as caught:
            self._load(self._pin(assignment={"a": "clu_00001"}))
        self.assertIn("1 of 2", str(caught.exception))

    def test_unreadable_json_is_refused_rather_than_crashing(self) -> None:
        import tempfile
        path = Path(tempfile.mkdtemp()) / "grouping.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(GroupingPinError) as caught:
            self._load(path)
        self.assertIn("unreadable", str(caught.exception))

    def test_the_digest_is_a_function_of_the_sequences_not_their_order(self) -> None:
        self.assertEqual(
            grouping_input_digest(self.ITEMS, self.THRESHOLD),
            grouping_input_digest(list(reversed(self.ITEMS)), self.THRESHOLD))

    def test_the_digest_changes_with_a_sequence_and_with_the_threshold(self) -> None:
        base = grouping_input_digest(self.ITEMS, self.THRESHOLD)
        self.assertNotEqual(base, grouping_input_digest(
            [("a", "MKVLAAGIVG" * 4), ("b", "MWYWYCWYWA" * 4)], self.THRESHOLD))
        self.assertNotEqual(base, grouping_input_digest(self.ITEMS, 0.5))

    def test_the_digest_changes_when_an_id_changes(self) -> None:
        # The assignment is keyed by id, so a pin whose ids moved describes a
        # different grouping even with the same sequences.
        self.assertNotEqual(
            grouping_input_digest(self.ITEMS, self.THRESHOLD),
            grouping_input_digest([("z", self.ITEMS[0][1]),
                                   ("b", self.ITEMS[1][1])], self.THRESHOLD))


class TheShippedPinTests(unittest.TestCase):
    """The pin that is actually checked in, against the fixture it is for."""

    FIXTURE = Path(__file__).parent / "fixtures" / "sdr"

    def setUp(self) -> None:
        if not PINNED_GROUPING.exists():
            self.skipTest(f"no pin at {PINNED_GROUPING}")

    def _items(self):
        from eagent.eval.retrospective import load_sdr, sdr_examples

        return sorted({(e.example_id, e.sequence)
                       for e in sdr_examples(load_sdr(self.FIXTURE))})

    def test_it_matches_the_checked_in_dataset_at_the_protocol_threshold(self) -> None:
        pinned = load_pinned_grouping(
            self._items(), BenchmarkProtocol().identity_threshold)
        self.assertTrue(pinned["complete"])
        self.assertIsNone(pinned["screen"])
        self.assertEqual(pinned["n_prefiltered_pairs"], 0)

    def test_it_records_the_identity_definition_the_audit_checks_against(self) -> None:
        pinned = load_pinned_grouping(
            self._items(), BenchmarkProtocol().identity_threshold)
        self.assertEqual(pinned["identity_definition"],
                         PAIRWISE_IDENTITY_DEFINITION)

    def test_no_cross_cluster_pair_is_over_the_threshold(self) -> None:
        """The pin's one promise, re-derived here instead of trusted.

        The pin says it closed the identity relation over every pair, which
        means no two sequences in different clusters reach the threshold. That
        is checkable in a couple of seconds by aligning only the cross-cluster
        pairs a k-mer screen admits -- a subset of what the pin aligned, so a
        pass is not independent evidence about the *design*, but it would
        catch a union-find bug, a mis-sorted assignment or a pin regenerated
        from the wrong file, and those are the ways a checked-in artefact goes
        wrong between commits.
        """
        items = self._items()
        pinned = load_pinned_grouping(
            items, BenchmarkProtocol().identity_threshold)
        sequences = dict(items)
        cluster_ids = sorted(set(pinned["assignment"].values()))
        index = {cid: n for n, cid in enumerate(cluster_ids)}
        as_folds = {k: index[v] for k, v in pinned["assignment"].items()}
        audit = audit_grouping(
            sequences, as_folds, identity=pairwise_identity,
            threshold=pinned["identity_threshold"],
            screens=[jaccard_screen(GROUPING_SCREEN)],
            identity_name=PAIRWISE_IDENTITY_DEFINITION,
            clustering_identity_name=pinned["identity_definition"],
            grouping_was_complete=True)
        self.assertEqual(audit.breaches, [], "\n".join(audit.render()))
        self.assertGreater(audit.n_pairs_aligned, 0,
                           "the screen admitted nothing, so this checked "
                           "nothing")

    def test_every_pinned_cluster_id_is_used(self) -> None:
        pinned = load_pinned_grouping(
            self._items(), BenchmarkProtocol().identity_threshold)
        self.assertEqual(len(set(pinned["assignment"].values())),
                         pinned["n_clusters"])


def _tiny_families():
    """Six sequences in three families, enough to exercise the fold map."""
    from eagent.science.enzyme_substrate import Example

    examples, clusters = [], {}
    for f in range(3):
        for n in range(4):
            ident = f"fam{f}_{n}"
            seq = "M" + ("AKLVGTSEDA", "WYWYCWYWYC", "PQRSTNHIMF")[f] * 10
            clusters[ident] = f"family{f}"
            examples.append(Example(ident, seq, "t", f == 0,
                                    "annotation", ()))
    return examples, clusters


def _only(pairs, name="only"):
    """A screen that admits exactly ``pairs``."""
    from eagent.eval.grouping_audit import CandidateScreen

    return CandidateScreen(name=name, description=f"exactly {sorted(pairs)}",
                           select=lambda seqs, folds: set(pairs))


if __name__ == "__main__":      # pragma: no cover
    unittest.main()
