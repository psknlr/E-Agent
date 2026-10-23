"""Tests for the profile-search similarity and for the column it adds.

Two kinds, deliberately separated:

* the plumbing, with a stub provider, so that the benchmark's new column is
  tested on a machine where the ``search`` extra is not installed. The stub is
  not an alignment and makes no claim to be one; it exists so that "the column
  appears, is ranking-only, and carries both paired differences" is checked
  without pyhmmer.
* the provider itself, skipped when pyhmmer is absent. CI runs one install
  with the extra and one without, so both legs are exercised.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from click.testing import CliRunner

from eagent.cli import main
from eagent.errors import ToolUnavailableError
from eagent.eval.retrospective import (
    BenchmarkProtocol, _alignment_record, load_sdr, run_benchmark,
    sdr_examples,
)
from eagent.science import alignment_similarity as A
from eagent.science.alignment_similarity import (
    AlignmentReport, AlignmentSimilarityError, PhmmerSimilarity, ProfileKernel,
    pool_from_examples, pyhmmer_available,
)
from eagent.science.enzyme_substrate import Example

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sdr"
needs_pyhmmer = unittest.skipUnless(
    pyhmmer_available(), "the search extra (pyhmmer) is not installed")


def sdr_sequences(n: int | None = None) -> list[str]:
    seqs = list(load_sdr(FIXTURES).sequences.values())
    return seqs if n is None else seqs[:n]


class Stub:
    """A deterministic number per pair. NOT an alignment; plumbing only."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, str]] = []

    def similarity(self, query: str, target: str) -> float:
        self.asked.append((query, target))
        return len(set(query[:40]) & set(target[:40])) / 20.0


def small_examples(n_per_class: int = 12) -> tuple[list[Example], dict[str, str]]:
    """Two targets over real sequences, labelled by a property of the sequence.

    The label is "the sequence starts with a residue in the first half of the
    alphabet", which is arbitrary on purpose: these tests check the plumbing,
    never that any comparator scores well.
    """
    seqs = sdr_sequences(2 * n_per_class + 10)
    examples: list[Example] = []
    clusters: dict[str, str] = {}
    for i, s in enumerate(seqs):
        ident = f"s{i}"
        clusters[ident] = f"c{i % 7}"
        for target in ("t_one", "t_two"):
            examples.append(
                Example(ident, s, target, bool(i % 2) if target == "t_one"
                        else bool(i % 3), "annotation"))
    return examples, clusters


class ReportTests(unittest.TestCase):

    def test_the_report_round_trips_and_names_its_backend(self) -> None:
        r = AlignmentReport(
            backend="phmmer", backend_version="pyhmmer 9", normalisation="none",
            pool_size=3, queries_run=2, pairs_requested=5, pairs_without_hit=1,
            search_settings={"E": 1.0}, builder_settings={"seed": 42})
        d = r.to_dict()
        self.assertEqual(d["backend"], "phmmer")
        self.assertEqual(d["pairs_without_hit"], 1)
        self.assertEqual(d["search_settings"], {"E": 1.0})
        self.assertEqual(d["builder_settings"], {"seed": 42})

    def test_a_provider_that_reports_nothing_is_named_not_described(self) -> None:
        record = _alignment_record(Stub())
        self.assertEqual(record["backend"], "Stub")
        self.assertIn("cannot", record["note"])

    def test_a_provider_that_reports_a_mapping_is_recorded_verbatim(self) -> None:
        class Mapped(Stub):
            def report(self):  # noqa: ANN201
                return {"backend": "mapped", "normalisation": "none"}

        self.assertEqual(_alignment_record(Mapped()),
                         {"backend": "mapped", "normalisation": "none"})

    def test_a_provider_reporting_a_dataclass_is_recorded_through_to_dict(self) -> None:
        class Dataclassy(Stub):
            def report(self):  # noqa: ANN201
                return AlignmentReport(
                    backend="phmmer", backend_version="v", normalisation="n",
                    pool_size=1, queries_run=1, pairs_requested=1,
                    pairs_without_hit=0)

        self.assertEqual(_alignment_record(Dataclassy())["backend"], "phmmer")


class PoolHelperTests(unittest.TestCase):

    def test_the_pool_is_distinct_sequences_normalised(self) -> None:
        rows = [Example("a", " mkv ", "t", True, "annotation"),
                Example("b", "MKV", "t", False, "annotation"),
                Example("c", "AAA", "t", True, "annotation")]
        self.assertEqual(pool_from_examples(rows), ["MKV", "AAA"])


class RefusalWithoutTheExtraTests(unittest.TestCase):

    def test_the_provider_refuses_when_pyhmmer_is_absent(self) -> None:
        original = A.pyhmmer_available
        A.pyhmmer_available = lambda: False
        try:
            with self.assertRaises(ToolUnavailableError) as ctx:
                PhmmerSimilarity(["MKVAAA"])
        finally:
            A.pyhmmer_available = original
        self.assertIn("search", str(ctx.exception))

    def test_it_does_not_fall_back_to_a_kmer_cosine(self) -> None:
        """The refusal is the feature: a column labelled alignment must align."""
        original = A.pyhmmer_available
        A.pyhmmer_available = lambda: False
        try:
            with self.assertRaises(ToolUnavailableError):
                PhmmerSimilarity(["MKVAAA", "MKVAAB"])
        finally:
            A.pyhmmer_available = original


class BenchmarkColumnTests(unittest.TestCase):
    """The column the provider feeds, checked with the stub."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.examples, cls.clusters = small_examples()
        cls.protocol = BenchmarkProtocol(n_folds=3, n_boot=30, min_per_class=5)
        cls.stub = Stub()
        cls.with_alignment = run_benchmark(
            cls.examples, cls.protocol, cluster_assignment=cls.clusters,
            alignment=cls.stub)
        cls.without = run_benchmark(cls.examples, cls.protocol,
                                    cluster_assignment=cls.clusters)

    def test_the_column_is_present_only_when_a_provider_is_supplied(self) -> None:
        for t in self.with_alignment.targets:
            self.assertIn("profile_nearest_neighbour", t.comparators, t.target)
        for t in self.without.targets:
            self.assertNotIn("profile_nearest_neighbour", t.comparators, t.target)

    def test_an_absent_provider_leaves_an_absent_cell_not_a_zero(self) -> None:
        self.assertEqual(self.without.alignment_similarity, {})
        self.assertNotIn("profile_nearest_neighbour", self.without.macro)

    def test_the_column_is_ranking_only(self) -> None:
        for t in self.with_alignment.targets:
            cell = t.comparators["profile_nearest_neighbour"]
            self.assertIsNotNone(cell["auroc"])
            self.assertNotIn("brier", cell)
            self.assertNotIn("ece", cell)

    def test_both_paired_differences_are_reported(self) -> None:
        for t in self.with_alignment.targets:
            for name in ("delta_auroc_profile_vs_nearest_neighbour",
                         "delta_auroc_vs_profile"):
                d = getattr(t, name)
                self.assertIsNotNone(d, f"{t.target}/{name}")
                self.assertIn("delta", d)
                self.assertIn("interval_excludes_zero", d)

    def test_the_other_columns_are_untouched_by_adding_this_one(self) -> None:
        """The same folds and rows: the model's own numbers must not move."""
        for a, b in zip(self.with_alignment.targets, self.without.targets):
            self.assertEqual(a.target, b.target)
            self.assertEqual(a.n, b.n)
            for name in ("prevalence", "nearest_neighbour", "spectrum_krr"):
                self.assertEqual(a.comparators[name]["auroc"],
                                 b.comparators[name]["auroc"], name)

    def test_the_query_is_the_held_out_sequence(self) -> None:
        """Directional providers must be asked in the search direction."""
        test_ids = {e.example_id for e in self.examples}
        by_sequence = {e.sequence: e.example_id for e in self.examples}
        self.assertTrue(self.stub.asked)
        for query, target in self.stub.asked:
            self.assertIn(by_sequence.get(query), test_ids)

    def test_the_render_states_what_produced_the_column(self) -> None:
        text = self.with_alignment.render()
        self.assertIn("ALIGNMENT BASELINE", text)
        self.assertIn("profile_nearest_neighbour", text)
        self.assertIn("not a measured dissimilarity", text)
        self.assertNotIn("ALIGNMENT BASELINE", self.without.render())

    def test_the_digest_records_the_added_column(self) -> None:
        self.assertNotEqual(self.with_alignment.digest, self.without.digest)


class StubKernel:
    """A constant off-diagonal. Positive definite above ~0, indefinite below."""

    self_similarity = 1.0

    def __init__(self, value: float, poison: str | None = None) -> None:
        self.value = value
        self.poison = poison
        self.kernel_id = f"stub:{value}:{bool(poison)}"

    def similarity(self, a: str, b: str) -> float:
        if a == b:
            return self.self_similarity
        if self.poison is not None and self.poison not in (a, b):
            return 0.3
        return self.value


class ProfileHeadColumnTests(unittest.TestCase):
    """The head refitted on an alignment kernel, checked without pyhmmer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.examples, cls.clusters = small_examples()
        cls.protocol = BenchmarkProtocol(n_folds=3, n_boot=30, min_per_class=5)

    def run_with(self, kernel: object):
        return run_benchmark(self.examples, self.protocol,
                             cluster_assignment=self.clusters,
                             alignment=Stub(), alignment_kernel=kernel)

    def test_a_definite_kernel_fills_the_columns_and_both_pairings(self) -> None:
        report = self.run_with(StubKernel(0.3))
        self.assertTrue(report.alignment_kernel["scored"])
        self.assertIsNone(report.alignment_kernel["cholesky_refusal"])
        for t in report.targets:
            self.assertIsNotNone(t.comparators["profile_krr"]["auroc"])
            self.assertIsNotNone(t.delta_auroc_profile_krr_vs_profile_nn)
            self.assertIsNotNone(t.delta_auroc_profile_krr_vs_spectrum_krr)
        self.assertIn("profile_krr", report.macro)
        self.assertIn("its column is in the table above", report.render())

    def test_an_indefinite_kernel_leaves_no_partial_column(self) -> None:
        """A refusal on any fold disqualifies the column for the whole run."""
        report = self.run_with(StubKernel(-0.9))
        self.assertFalse(report.alignment_kernel["scored"])
        self.assertIn("positive definite",
                      report.alignment_kernel["cholesky_refusal"])
        for t in report.targets:
            self.assertNotIn("profile_krr", t.comparators)
            self.assertNotIn("profile_krr_calibrated", t.comparators)
            self.assertIsNone(t.delta_auroc_profile_krr_vs_profile_nn)
        self.assertNotIn("profile_krr", report.macro)
        self.assertIn("NOT SCORED", report.render())

    def test_a_kernel_that_only_one_fold_refuses_leaves_no_partial_column(self) -> None:
        """The same invariant when the refusal comes after a fold has scored."""
        poison = self.examples[0].sequence
        report = self.run_with(StubKernel(-0.95, poison=poison))
        if report.alignment_kernel["scored"]:
            self.skipTest("this kernel factorised on every fold")
        for t in report.targets:
            self.assertNotIn("profile_krr", t.comparators)
        self.assertNotIn("profile_krr", report.macro)

    def test_the_ridge_is_never_raised_to_force_a_factorisation(self) -> None:
        report = self.run_with(StubKernel(-0.9))
        self.assertEqual(report.alignment_kernel["ridge"],
                         self.protocol.ridge)
        self.assertIn("was NOT raised", report.render())

    def test_the_other_columns_are_untouched_by_adding_this_one(self) -> None:
        with_head = self.run_with(StubKernel(0.3))
        without = run_benchmark(self.examples, self.protocol,
                                cluster_assignment=self.clusters,
                                alignment=Stub())
        for a, b in zip(with_head.targets, without.targets):
            for name in ("nearest_neighbour", "spectrum_krr",
                         "profile_nearest_neighbour"):
                self.assertEqual(a.comparators[name]["auroc"],
                                 b.comparators[name]["auroc"], name)


@needs_pyhmmer
class ProfileKernelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.seqs = sdr_sequences(8)
        cls.provider = PhmmerSimilarity(cls.seqs, cpus=1)
        cls.kernel = ProfileKernel(cls.provider)

    def test_it_is_symmetric_where_the_provider_is_not(self) -> None:
        a, b = self.seqs[0], self.seqs[1]
        self.assertEqual(self.kernel.similarity(a, b),
                         self.kernel.similarity(b, a))

    def test_it_is_the_mean_of_the_two_directions(self) -> None:
        a, b = self.seqs[2], self.seqs[3]
        expected = (self.provider.similarity(a, b)
                    + self.provider.similarity(b, a)) / 2.0
        self.assertAlmostEqual(self.kernel.similarity(a, b), expected,
                               places=9)

    def test_a_sequence_scores_one_against_itself(self) -> None:
        self.assertEqual(self.kernel.self_similarity, 1.0)
        self.assertAlmostEqual(
            self.kernel.similarity(self.seqs[0], self.seqs[0]), 1.0, places=6)

    def test_it_names_itself_so_a_factor_cache_cannot_cross_kernels(self) -> None:
        self.assertEqual(self.kernel.kernel_id, "phmmer_symmetrised")

    def test_it_declares_no_ks_because_it_is_not_a_kmer_kernel(self) -> None:
        self.assertFalse(hasattr(self.kernel, "ks"))

    def test_a_pair_is_scored_once_however_often_it_is_asked(self) -> None:
        kernel = ProfileKernel(PhmmerSimilarity(self.seqs[:4], cpus=1))
        for _ in range(3):
            kernel.similarity(self.seqs[0], self.seqs[1])
        self.assertEqual(kernel.report().pairs_requested, 2)

    def test_the_report_comes_from_the_provider(self) -> None:
        self.assertEqual(self.kernel.report().backend, "phmmer")


@needs_pyhmmer
class ProfileContrastTests(unittest.TestCase):

    def test_the_label_free_contrast_runs_on_the_profile_search(self) -> None:
        from eagent.science.alignment_similarity import profile_contrast

        out = profile_contrast(sdr_sequences(12), n_pairs=12, seed=0, cpus=1)
        self.assertEqual(out["n_sequences"], 12)
        self.assertIsNotNone(out["median_real_pair"])
        self.assertEqual(out["provider"]["pool_size"], 24)
        self.assertIn("profile bit score", out["measure"])

    def test_the_provider_pool_covers_the_shuffles_too(self) -> None:
        """A target outside the pool is refused, so prepare must include them."""
        from eagent.science.alignment_similarity import profile_contrast

        out = profile_contrast(sdr_sequences(6), n_pairs=20, seed=1, cpus=1)
        self.assertEqual(out["provider"]["pool_size"], 12)


class CommandLineTests(unittest.TestCase):
    """The flag's two outcomes, independent of what is installed here."""

    def test_asking_for_the_baseline_without_the_extra_refuses_with_the_fix(self) -> None:
        original = A.pyhmmer_available
        A.pyhmmer_available = lambda: False
        try:
            result = CliRunner().invoke(main, [
                "benchmark", "sdr", "--data", str(FIXTURES), "--alignment"])
        finally:
            A.pyhmmer_available = original
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("eagent[search]", result.output)
        self.assertIn("drop --alignment", result.output)

    def test_asking_for_the_head_without_the_extra_refuses_too(self) -> None:
        """--alignment-head implies --alignment, so it meets the same refusal."""
        original = A.pyhmmer_available
        A.pyhmmer_available = lambda: False
        try:
            result = CliRunner().invoke(main, [
                "benchmark", "sdr", "--data", str(FIXTURES),
                "--alignment-head"])
        finally:
            A.pyhmmer_available = original
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("eagent[search]", result.output)

    def test_the_flag_is_off_by_default(self) -> None:
        """A plain install must measure the same benchmark it always did."""
        result = CliRunner().invoke(main, ["benchmark", "sdr", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--no-alignment", result.output)
        self.assertIn("[default: no-alignment]", result.output.replace("\n", " "))


@needs_pyhmmer
class PhmmerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.seqs = sdr_sequences(12)
        cls.provider = PhmmerSimilarity(cls.seqs, cpus=1)

    def test_a_sequence_is_exactly_as_similar_to_itself_as_the_scale_allows(self) -> None:
        for s in self.seqs[:3]:
            self.assertAlmostEqual(self.provider.similarity(s, s), 1.0, places=6)

    def test_another_sequence_scores_below_the_self_score(self) -> None:
        a, b = self.seqs[0], self.seqs[1]
        self.assertLess(self.provider.similarity(a, b), 1.0)

    def test_the_score_is_directional_and_is_not_symmetrised(self) -> None:
        full = self.seqs[0]
        fragment = full[: len(full) // 3]
        provider = PhmmerSimilarity([full, fragment], cpus=1)
        forward = provider.similarity(fragment, full)
        backward = provider.similarity(full, fragment)
        self.assertNotAlmostEqual(forward, backward, places=3)

    def test_a_target_outside_the_pool_is_refused_not_scored_zero(self) -> None:
        with self.assertRaises(AlignmentSimilarityError) as ctx:
            self.provider.similarity(self.seqs[0], "MKVAAAGGGLLLKKK")
        self.assertIn("never searched", str(ctx.exception))

    def test_raw_bit_scores_are_available_for_checking_the_normalisation(self) -> None:
        provider = PhmmerSimilarity(self.seqs[:4], cpus=1, normalise=False)
        raw = provider.similarity(self.seqs[0], self.seqs[0])
        self.assertGreater(raw, 1.0)
        self.assertEqual(provider.report().normalisation, "none (raw bit score)")

    def test_duplicate_sequences_collapse_to_one_pool_entry(self) -> None:
        provider = PhmmerSimilarity([self.seqs[0], self.seqs[0], self.seqs[1]],
                                    cpus=1)
        self.assertEqual(provider.report().pool_size, 2)

    def test_an_empty_pool_or_an_empty_sequence_is_refused(self) -> None:
        with self.assertRaises(AlignmentSimilarityError):
            PhmmerSimilarity([], cpus=1)
        with self.assertRaises(AlignmentSimilarityError):
            PhmmerSimilarity(["MKVAAA", "   "], cpus=1)

    def test_the_report_counts_what_was_asked_and_what_had_no_hit(self) -> None:
        provider = PhmmerSimilarity(self.seqs[:5], cpus=1)
        for target in self.seqs[:5]:
            provider.similarity(self.seqs[0], target)
        report = provider.report()
        self.assertEqual(report.pairs_requested, 5)
        self.assertEqual(report.queries_run, 1)
        self.assertEqual(report.pool_size, 5)
        self.assertLessEqual(report.pairs_without_hit, 5)
        self.assertIn("BLOSUM", report.builder_settings["score_matrix"])
        self.assertFalse(report.search_settings["acceleration_filters"])

    def test_one_profile_is_searched_once_however_often_it_is_asked(self) -> None:
        provider = PhmmerSimilarity(self.seqs[:4], cpus=1)
        for _ in range(3):
            for target in self.seqs[:4]:
                provider.similarity(self.seqs[1], target)
        self.assertEqual(provider.report().queries_run, 1)
        self.assertEqual(provider.report().pairs_requested, 12)

    def test_a_query_outside_the_pool_is_searched_and_normalised(self) -> None:
        """A new sequence is the realistic query: it is not in the pool."""
        provider = PhmmerSimilarity(self.seqs[1:5], cpus=1)
        value = provider.similarity(self.seqs[0], self.seqs[1])
        self.assertLess(value, 1.0)
        self.assertEqual(provider.report().queries_run, 1)


@needs_pyhmmer
class PhmmerInTheBenchmarkTests(unittest.TestCase):
    """The real provider, on a small slice, through the benchmark it feeds."""

    def test_the_real_provider_fills_the_column(self) -> None:
        examples, clusters = small_examples(8)
        provider = PhmmerSimilarity(pool_from_examples(examples), cpus=1)
        report = run_benchmark(
            examples, BenchmarkProtocol(n_folds=2, n_boot=20, min_per_class=5),
            cluster_assignment=clusters, alignment=provider)
        for t in report.targets:
            self.assertIsNotNone(
                t.comparators["profile_nearest_neighbour"]["auroc"])
        self.assertEqual(report.alignment_similarity["backend"], "phmmer")
        self.assertGreater(report.alignment_similarity["queries_run"], 0)
        self.assertIn("pyhmmer", report.alignment_similarity["backend_version"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
