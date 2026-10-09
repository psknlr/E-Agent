"""Tests for the SDR label-recovery benchmark.

Two kinds: the loader and the protocol on the real recorded dataset (a verbatim
copy under ``tests/fixtures/sdr``, CC BY 4.0, md5-checked), and the protocol's
leakage behaviour on synthetic data built to make leakage visible.
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from eagent.eval.retrospective import (
    CLAIM, SDR_FILE_MD5, BenchmarkProtocol, RetrospectiveError, load_sdr,
    precision_at_k, run_benchmark, sdr_examples,
)
from eagent.science.enzyme_substrate import STANDARD_AA, Example

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sdr"
RECORD = json.loads((Path(__file__).resolve().parent / "fixtures" / "zenodo"
                     / "record_7141435.json").read_text(encoding="utf-8"))


def mutate(rng: random.Random, seq: str, n: int = 3) -> str:
    s = list(seq)
    for _ in range(n):
        s[rng.randrange(len(s))] = rng.choice(STANDARD_AA)
    return "".join(s)


def noisy_families(n_families: int = 40, per: int = 3, seed: int = 11
                   ) -> tuple[list[Example], dict[str, str]]:
    """Families of near-duplicates whose label has nothing to do with sequence."""
    rng = random.Random(seed)
    examples: list[Example] = []
    clusters: dict[str, str] = {}
    for f in range(n_families):
        base = "".join(rng.choice(STANDARD_AA) for _ in range(200))
        label = rng.random() < 0.5
        for c in range(per):
            ident = f"f{f}_{c}"
            examples.append(Example(ident, mutate(rng, base), "t", label,
                                    "annotation"))
            clusters[ident] = f"family{f}"
    return examples, clusters


class TestTheRecordedDataset(unittest.TestCase):

    def test_the_expected_md5s_are_the_ones_the_zenodo_record_states(self) -> None:
        stated = {f["key"]: f["checksum"].removeprefix("md5:")
                  for f in RECORD["files"]}
        for name, md5 in SDR_FILE_MD5.items():
            self.assertEqual(stated[name], md5, name)

    def test_the_fixture_loads_and_matches_the_readme_counts(self) -> None:
        data = load_sdr(FIXTURES)
        self.assertEqual(len(data.sequences), 358)       # "358 SDRs" in the README
        self.assertEqual(len(data.cofactor), 338)
        self.assertEqual(set(data.substructure["A0A024R7X6"]),
                         {"phenol_substructure", "sterol_substructure",
                          "CoA_substructure"})
        self.assertEqual(len(next(iter(data.clusters.values()))), 9)  # 9 clusters

    def test_examples_are_annotation_sourced_and_name_the_deposit(self) -> None:
        examples = sdr_examples(load_sdr(FIXTURES))
        self.assertTrue(examples)
        self.assertTrue(all(e.label_source == "annotation" for e in examples))
        self.assertTrue(all("10.5281/zenodo.7141435" in e.evidence for e in examples))
        self.assertEqual(len({e.target for e in examples}), 13)

    def test_cofactor_label_follows_the_readme_convention(self) -> None:
        # README: 1 = NADP, 0 = NAD; A0A024R7X6 is 1.0 in the file
        examples = {(e.example_id, e.target): e for e in
                    sdr_examples(load_sdr(FIXTURES))}
        self.assertTrue(examples[("A0A024R7X6", "sdr:cofactor_NADP")].label)

    def test_an_edited_file_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name in list(SDR_FILE_MD5):
                shutil.copy(FIXTURES / name, tmp)
            path = Path(tmp) / "SDR_cofactor_classifications.csv"
            path.write_text(path.read_text().replace("1.0", "0.0", 1))
            with self.assertRaises(RetrospectiveError) as caught:
                load_sdr(tmp)
            self.assertIn("Zenodo record states", str(caught.exception))

    def test_a_missing_file_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RetrospectiveError):
                load_sdr(tmp)


class TestLeakageIsVisible(unittest.TestCase):

    def run_both(self):
        examples, clusters = noisy_families()
        proto = BenchmarkProtocol(n_boot=30, min_per_class=5)
        grouped = run_benchmark(examples, proto, cluster_assignment=clusters)
        ungrouped = run_benchmark(
            examples, BenchmarkProtocol(**{**proto.__dict__, "grouped": False}),
            cluster_assignment=clusters)
        return grouped, ungrouped

    def test_noise_is_not_recovered_when_families_are_held_out_together(self) -> None:
        grouped, _ = self.run_both()
        self.assertLess(grouped.macro["nearest_neighbour"]["auroc"], 0.7)

    def test_the_same_noise_looks_perfect_when_families_straddle_folds(self) -> None:
        _, ungrouped = self.run_both()
        self.assertGreater(ungrouped.macro["nearest_neighbour"]["auroc"], 0.9)

    def test_the_similarity_to_training_exposes_the_difference(self) -> None:
        grouped, ungrouped = self.run_both()
        self.assertGreater(ungrouped.similarity_to_training["median"],
                           grouped.similarity_to_training["median"] + 0.3)

    def test_prevalence_ranks_nothing(self) -> None:
        grouped, _ = self.run_both()
        for t in grouped.targets:
            self.assertEqual(t.comparators["prevalence"]["auroc"], 0.5)

    def test_every_comparator_scored_the_same_items(self) -> None:
        grouped, _ = self.run_both()
        target = grouped.targets[0]
        self.assertEqual(target.n, 120)
        self.assertEqual(
            {c: v["prevalence"] for c, v in target.comparators.items()
             if v.get("prevalence") is not None},
            {c: target.n_positive / target.n for c in
             ("prevalence", "nearest_neighbour", "spectrum_krr",
              "spectrum_krr_calibrated")})


class TestProtocol(unittest.TestCase):

    def test_the_claim_travels_with_the_report(self) -> None:
        examples, clusters = noisy_families(20)
        report = run_benchmark(examples, BenchmarkProtocol(n_boot=20,
                               min_per_class=3), cluster_assignment=clusters)
        self.assertEqual(report.claim, CLAIM)
        self.assertIn("ANNOTATION-DERIVED", report.render())
        self.assertIn("not evidence about ketoreductase activity", report.render())

    def test_a_target_with_too_few_of_one_class_is_skipped_with_the_counts(self) -> None:
        examples, clusters = noisy_families(20)
        rng = random.Random(1)
        rare = [Example(f"r{i}", "".join(rng.choice(STANDARD_AA) for _ in range(150)),
                        "rare", i == 0, "annotation") for i in range(30)]
        for e in rare:
            clusters[e.example_id] = e.example_id
        report = run_benchmark(examples + rare, BenchmarkProtocol(
            n_boot=20, min_per_class=5), cluster_assignment=clusters)
        self.assertIn("rare", report.skipped_targets)
        self.assertIn("1 positive", report.skipped_targets["rare"])
        self.assertNotIn("rare", [t.target for t in report.targets])

    def test_the_run_is_deterministic_and_the_digest_tracks_the_seed(self) -> None:
        examples, clusters = noisy_families(20)
        proto = BenchmarkProtocol(n_boot=20, min_per_class=3)
        a = run_benchmark(examples, proto, cluster_assignment=clusters)
        b = run_benchmark(examples, proto, cluster_assignment=clusters)
        c = run_benchmark(examples, BenchmarkProtocol(**{**proto.__dict__, "seed": 9}),
                          cluster_assignment=clusters)
        self.assertEqual(a.digest, b.digest)
        self.assertNotEqual(a.digest, c.digest)

    def test_a_missing_cluster_is_refused(self) -> None:
        examples, clusters = noisy_families(10)
        clusters.pop(examples[0].example_id)
        with self.assertRaises(RetrospectiveError):
            run_benchmark(examples, cluster_assignment=clusters)

    def test_one_id_naming_two_sequences_is_refused(self) -> None:
        examples, clusters = noisy_families(10)
        bad = Example(examples[0].example_id, "ACDEFGHIK" * 20, "other", True,
                      "annotation")
        with self.assertRaises(RetrospectiveError):
            run_benchmark(examples + [bad], cluster_assignment=clusters)

    def test_the_macro_says_how_many_targets_each_average_is_over(self) -> None:
        examples, clusters = noisy_families(20)
        report = run_benchmark(examples, BenchmarkProtocol(
            n_boot=20, min_per_class=3), cluster_assignment=clusters)
        self.assertEqual(report.macro["prevalence"]["n_targets_auroc"], 1.0)
        self.assertIn("over 1 targets", report.render())


class TestOnTheRealData(unittest.TestCase):
    """One real target, cluster-held-out: the claim the README can support."""

    def test_the_cofactor_preference_is_recoverable_from_sequence(self) -> None:
        from eagent.tools.mine_sequences import cluster_sequences
        data = load_sdr(FIXTURES)
        rng = random.Random(0)
        accessions = sorted(data.cofactor)
        rng.shuffle(accessions)
        keep = set(accessions[:140])
        examples = [e for e in sdr_examples(data)
                    if e.target == "sdr:cofactor_NADP" and e.example_id in keep]
        items = sorted({(e.example_id, e.sequence) for e in examples})
        clustering = cluster_sequences(items, 0.4)
        report = run_benchmark(
            examples, BenchmarkProtocol(n_boot=30),
            cluster_assignment=dict(clustering.assignment),
            clustering_note={"method": clustering.method})
        target = report.targets[0]
        self.assertGreater(target.comparators["spectrum_krr"]["auroc"], 0.7)
        self.assertGreater(target.comparators["nearest_neighbour"]["auroc"], 0.6)
        self.assertEqual(target.comparators["prevalence"]["auroc"], 0.5)


class TestPrecisionAtK(unittest.TestCase):

    def test_hits_and_interval(self) -> None:
        out = precision_at_k([0.9, 0.8, 0.7, 0.1], [True, False, True, False], 3)
        self.assertEqual((out["k"], out["hits"]), (3, 2))
        self.assertAlmostEqual(out["precision"], 2 / 3)
        lo, hi = out["wilson_95"]
        self.assertLess(lo, 2 / 3)
        self.assertGreater(hi, 2 / 3)

    def test_ties_break_by_index_not_by_luck(self) -> None:
        out = precision_at_k([0.5, 0.5, 0.5], [True, False, False], 1)
        self.assertEqual(out["hits"], 1)

    def test_k_larger_than_the_pool_reports_the_pool(self) -> None:
        self.assertEqual(precision_at_k([1.0], [True], 5)["k"], 1)


class TestCli(unittest.TestCase):

    def test_a_directory_without_the_dataset_is_refused_with_a_next_step(self) -> None:
        from click.testing import CliRunner
        from eagent.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            result = CliRunner().invoke(main, ["benchmark", "sdr", "--data", tmp])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("missing", result.output + str(result.exception or ""))

    def test_an_edited_file_is_refused_by_the_cli_too(self) -> None:
        from click.testing import CliRunner
        from eagent.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            for name in SDR_FILE_MD5:
                shutil.copy(FIXTURES / name, tmp)
            path = Path(tmp) / "SDR_sequences.fasta"
            path.write_text(path.read_text().replace("M", "K", 1))
            result = CliRunner().invoke(main, ["benchmark", "sdr", "--data", tmp])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("md5", result.output)


if __name__ == "__main__":
    unittest.main()
