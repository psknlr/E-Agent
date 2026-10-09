"""Regression tests: one more search must never remove a candidate.

A sequence found by two routes was found by two independent pieces of
evidence. The filter collapsed the routes to one "best" hit before judging
any of them, so the verdict depended on which route happened to rank highest.
A protein kept on a blastp hit at 80% identity over 95% of the query was
dropped when an hmmsearch route also found it at 20% coverage: a profile hit
outranks a pairwise one, and the profile hit failed the coverage floor.

Running one more search removed a qualifying candidate. Nothing about adding
evidence should be able to do that, and the loss is invisible -- the pool is
simply smaller, and the row says the sequence failed the coverage bar.
"""

from __future__ import annotations

import unittest

from eagent.tools.mine_sequences import (
    FastaEntry, LengthExpectation, MineSequences, RetentionPolicy, SearchHit,
)

SEQ = ("MKAAVLYEFGKPLEIKEVEVAPPKAHEVRIKIAYTGVCHTDAYTLSGADPEGLFPVILGHEGAGIVESVGEGVT"
       "NVKPGDHVIPLYTPECGECKFCKSGKTNLCQKIRATQGKGLMPDGTSRFTCKGKEILHYMGCSTFSEYTVVAD")
ENTRY = FastaEntry("P00001", "an enzyme", SEQ)
INDICES = {"db@v1": {"P00001": ENTRY}}
WINDOW = LengthExpectation(min_length=20, max_length=500, source="test fixture")


def blast(**kw) -> SearchHit:
    base = dict(query_id="SEED1", query_kind="sequence", subject_id="P00001",
                search_method="blastp", database_name="db",
                database_version="v1", percent_identity=80.0,
                query_coverage=0.95, evalue=1e-50, bitscore=400.0,
                family_template_id="SDR")
    base.update(kw)
    return SearchHit(**base)


def profile(**kw) -> SearchHit:
    base = dict(query_id="PF00106", query_kind="profile", subject_id="P00001",
                search_method="hmmsearch", database_name="db",
                database_version="v1", percent_identity=None,
                query_coverage=0.95, evalue=1e-40, bitscore=300.0,
                family_template_id="SDR")
    base.update(kw)
    return SearchHit(**base)


class RouteIndependenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.step = MineSequences()
        self.retention = RetentionPolicy(
            max_evalue=1e-5, min_query_coverage=0.60,
            require_family_evidence=True)

    def filter(self, hits):
        return self.step._filter_hits(hits, INDICES, self.retention, WINDOW)

    def test_a_good_blast_hit_is_retained(self) -> None:
        rows, records, _ = self.filter([blast()])
        self.assertEqual(len(records), 1)
        self.assertEqual(rows[0].status, "retained")

    def test_adding_a_failing_profile_route_does_not_remove_it(self) -> None:
        rows, records, _ = self.filter(
            [blast(), profile(query_coverage=0.20, evalue=1e-3)])
        self.assertEqual(len(records), 1,
                         "one more search must not shrink the pool")
        self.assertEqual(rows[0].status, "retained")

    def test_the_route_that_travels_with_it_is_one_that_retained_it(self) -> None:
        rows, _, _ = self.filter(
            [blast(), profile(query_coverage=0.20, evalue=1e-3)])
        self.assertEqual(rows[0].search_method, "blastp")
        self.assertEqual(rows[0].seed_accession, "SEED1")

    def test_adding_a_failing_blast_route_does_not_remove_a_profile_hit(self) -> None:
        rows, records, _ = self.filter(
            [profile(), blast(query_coverage=0.10, evalue=1.0)])
        self.assertEqual(len(records), 1)
        self.assertEqual(rows[0].search_method, "hmmsearch")
        self.assertEqual(rows[0].family_evidence, "profile_hmm")

    def test_the_better_of_two_qualifying_routes_still_wins(self) -> None:
        """The existing ranking is unchanged where both routes qualify."""
        rows, _, _ = self.filter([blast(), profile()])
        self.assertEqual(rows[0].search_method, "hmmsearch")
        self.assertEqual(rows[0].family_evidence, "profile_hmm")

    def test_the_disagreement_is_recorded(self) -> None:
        rows, _, _ = self.filter(
            [blast(), profile(query_coverage=0.20, evalue=1e-3)])
        self.assertEqual(rows[0].n_routes, 2)
        self.assertEqual(rows[0].n_routes_retaining, 1)
        self.assertIn("hmmsearch", rows[0].dissenting_routes)
        self.assertIn("coverage", rows[0].dissenting_routes)

    def test_a_unanimous_retention_has_no_dissent(self) -> None:
        rows, _, _ = self.filter([blast(), profile()])
        self.assertEqual(rows[0].n_routes_retaining, 2)
        self.assertEqual(rows[0].dissenting_routes, "")

    def test_a_sequence_no_route_retains_is_still_excluded(self) -> None:
        rows, records, _ = self.filter(
            [blast(query_coverage=0.10), profile(query_coverage=0.20)])
        self.assertEqual(records, [])
        self.assertEqual(rows[0].status, "excluded")
        self.assertEqual(rows[0].n_routes_retaining, 0)

    def test_a_rejection_names_the_best_case_against_the_sequence(self) -> None:
        """Not an arbitrary route's objection: the fewest objections wins."""
        rows, _, _ = self.filter([
            blast(query_coverage=0.10, evalue=1.0, family_template_id=None),
            profile(query_coverage=0.20)])
        self.assertIn("coverage:0.200", rows[0].reason)
        self.assertNotIn("evalue", rows[0].reason)

    def test_a_sequence_level_failure_is_not_route_dependent(self) -> None:
        """A length outside the window is a fact about the sequence."""
        narrow = LengthExpectation(min_length=400, max_length=500,
                                   source="test fixture")
        rows, records, _ = self.step._filter_hits(
            [blast(), profile()], INDICES, self.retention, narrow)
        self.assertEqual(records, [])
        self.assertEqual(rows[0].status, "excluded")
        self.assertIn("len=", rows[0].reason)

    def test_the_new_columns_are_in_the_table(self) -> None:
        from eagent.tools.mine_sequences import RetrievalRow
        for column in ("n_routes", "n_routes_retaining", "dissenting_routes"):
            self.assertIn(column, RetrievalRow.columns)
        rows, _, _ = self.filter([blast()])
        self.assertEqual(len(rows[0].as_row()), len(RetrievalRow.columns))


if __name__ == "__main__":
    unittest.main(verbosity=2)
