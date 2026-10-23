"""The sequence-search adapter, run against real HMMER rather than a fake.

The point of this file is one guarantee the rest of the suite cannot give. Every
other test of ``mine_sequences`` drives the adapters through a fake runner that
writes output in the documented shape, so the parser and the fixture agree by
construction: both were written from the same format documentation by the same
hand. What that can never catch is the real program writing something the
parser does not expect.

So the tests here that matter build a real HMM, run a real search, and hand the
shipped ``HmmsearchAdapter`` **HMMER's own** ``--domtblout``. They skip where
the ``search`` extra is absent; the argv-contract and refusal tests do not need
it and always run.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eagent.errors import ToolUnavailableError
from eagent.tools import mine_sequences as M
from eagent.tools import pyhmmer_search as P
from eagent.tools.mine_sequences import (
    HmmsearchAdapter, SearchExecutionError, SearchQuery, SequenceDatabase,
    parse_fasta,
)
from eagent.tools.pyhmmer_search import (
    PyhmmerError, PyhmmerRunner, build_profile_hmm, hmmer_version,
    pyhmmer_available,
)

FIXTURE = (Path(__file__).resolve().parent / "fixtures" / "sdr"
           / "SDR_sequences.fasta")
needs_pyhmmer = unittest.skipUnless(
    pyhmmer_available(), "the search extra (pyhmmer) is not installed")


def sdr_entries(n: int | None = None):
    entries = parse_fasta(FIXTURE.read_text(encoding="utf-8"),
                          source=str(FIXTURE))
    return entries if n is None else entries[:n]


class RelativeTargetTests(unittest.TestCase):
    """The defect the real runner exposed: one field with two meanings."""

    def test_a_relative_fasta_path_is_refused_as_a_search_target(self) -> None:
        db = SequenceDatabase(name="d", version="v1",
                              fasta_path="tests/fixtures/sdr/SDR_sequences.fasta")
        with self.assertRaises(SearchExecutionError) as ctx:
            db.target()
        self.assertIn("scratch working directory", str(ctx.exception))

    def test_resolve_fasta_still_reads_a_relative_path_against_the_cache(self) -> None:
        """The other meaning is intact; only conflating them is refused."""
        db = SequenceDatabase(name="d", version="v1", fasta_path="sdr.fasta")
        self.assertEqual(db.resolve_fasta(Path("/cache")),
                         Path("/cache/sdr.fasta"))

    def test_an_absolute_path_is_the_target(self) -> None:
        db = SequenceDatabase(name="d", version="v1", fasta_path=str(FIXTURE))
        self.assertEqual(db.target(), str(FIXTURE))

    def test_an_index_alias_is_not_interpreted_as_a_path(self) -> None:
        db = SequenceDatabase(name="d", version="v1", fasta_path="rel.fasta",
                              search_target="uniprot_db")
        self.assertEqual(db.target(), "uniprot_db")


class RefusalWithoutTheExtraTests(unittest.TestCase):

    def test_the_runner_refuses_to_exist_without_pyhmmer(self) -> None:
        original = P.pyhmmer_available
        P.pyhmmer_available = lambda: False
        try:
            with self.assertRaises(ToolUnavailableError) as ctx:
                PyhmmerRunner()
        finally:
            P.pyhmmer_available = original
        self.assertIn("search", str(ctx.exception))

    def test_building_a_profile_refuses_without_pyhmmer(self) -> None:
        original = P.pyhmmer_available
        P.pyhmmer_available = lambda: False
        try:
            with self.assertRaises(ToolUnavailableError):
                build_profile_hmm([("a", "MKVAAA")], "/tmp/none.hmm", name="x")
        finally:
            P.pyhmmer_available = original


@needs_pyhmmer
class ArgvContractTests(unittest.TestCase):
    """What the runner implements, and what it refuses rather than ignores."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="eagent-pyhmmer-"))
        self.runner = PyhmmerRunner()

    def run_argv(self, *args: str):
        return self.runner(["hmmsearch", *args], cwd=self.tmp)

    def test_an_unimplemented_option_is_refused_not_dropped(self) -> None:
        result = self.run_argv("--domtblout", "o.tbl", "--incT", "20",
                               "p.hmm", "db.fasta")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--incT", result.stderr)
        self.assertIn("not implemented", result.stderr)

    def test_a_missing_domtblout_is_refused(self) -> None:
        result = self.run_argv("p.hmm", "db.fasta")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--domtblout", result.stderr)

    def test_a_non_numeric_evalue_is_refused(self) -> None:
        result = self.run_argv("--domtblout", "o.tbl", "-E", "soon",
                               "p.hmm", "db.fasta")
        self.assertEqual(result.returncode, 2)
        self.assertIn("not a number", result.stderr)

    def test_the_wrong_number_of_positionals_is_refused(self) -> None:
        result = self.run_argv("--domtblout", "o.tbl", "p.hmm")
        self.assertEqual(result.returncode, 2)
        self.assertIn("positional", result.stderr)

    def test_a_missing_profile_exits_non_zero_like_the_binary_would(self) -> None:
        result = self.run_argv("--domtblout", "o.tbl", "-E", "10.0",
                               "absent.hmm", str(FIXTURE))
        self.assertEqual(result.returncode, 1)
        self.assertIn("no such profile", result.stderr)

    def test_a_missing_database_exits_non_zero(self) -> None:
        record = build_profile_hmm([("seed", sdr_entries(1)[0].sequence)],
                                   self.tmp / "seed.hmm", name="seed")
        result = self.run_argv("--domtblout", "o.tbl", "-E", "10.0",
                               record["path"], "absent.fasta")
        self.assertEqual(result.returncode, 1)
        self.assertIn("no such sequence database", result.stderr)

    def test_nothing_is_recorded_as_run_when_the_argv_is_refused(self) -> None:
        self.run_argv("--domtblout", "o.tbl", "--incT", "20", "p.hmm", "d.fa")
        self.assertEqual(self.runner.invocations, [])


@needs_pyhmmer
class ProfileBuildingTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="eagent-hmm-"))

    def test_a_single_sequence_builds_a_profile_of_its_length(self) -> None:
        entry = sdr_entries(1)[0]
        record = build_profile_hmm([(entry.identifier, entry.sequence)],
                                   self.tmp / "one.hmm", name="one")
        self.assertEqual(record["profile_length"], len(entry.sequence))
        self.assertEqual(record["built_from"], "single sequence")
        self.assertTrue(Path(record["path"]).exists())

    def test_the_record_carries_the_hmmer_version_that_wrote_the_file(self) -> None:
        record = build_profile_hmm([("a", "MKVAAAGGGLLLKKKDDDEEE")],
                                   self.tmp / "v.hmm", name="v")
        self.assertIn("HMMER", record["hmmer_version"])
        self.assertEqual(record["hmmer_version"], hmmer_version())

    def test_an_aligned_set_builds_from_the_alignment(self) -> None:
        rows = [("a", "MKVAA-GGG"), ("b", "MKVAAGGGG"), ("c", "MKV-AGGGG")]
        record = build_profile_hmm(rows, self.tmp / "msa.hmm", name="msa")
        self.assertIn("alignment of 3 sequences", record["built_from"])
        self.assertEqual(record["n_sequences"], 3)

    def test_unaligned_sequences_are_refused_not_aligned(self) -> None:
        rows = [("a", "MKVAAAGGG"), ("b", "MKVAAAGGGTTT")]
        with self.assertRaises(PyhmmerError) as ctx:
            build_profile_hmm(rows, self.tmp / "no.hmm", name="no")
        self.assertIn("does not align them", str(ctx.exception))

    def test_a_lone_gapped_sequence_is_refused(self) -> None:
        with self.assertRaises(PyhmmerError):
            build_profile_hmm([("a", "MKV--AAA")], self.tmp / "g.hmm", name="g")

    def test_an_empty_input_is_refused(self) -> None:
        with self.assertRaises(PyhmmerError):
            build_profile_hmm([], self.tmp / "e.hmm", name="e")
        with self.assertRaises(PyhmmerError):
            build_profile_hmm([("a", "")], self.tmp / "e.hmm", name="e")


@needs_pyhmmer
class TheShippedParserOnRealOutputTests(unittest.TestCase):
    """The guarantee this file exists for."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="eagent-real-"))
        cls.entries = sdr_entries()
        cls.record = build_profile_hmm(
            [(cls.entries[0].identifier, cls.entries[0].sequence)],
            cls.tmp / "sdr.hmm", name="sdr_seed")
        cls.runner = PyhmmerRunner()
        cls.adapter = HmmsearchAdapter(
            runner=cls.runner,
            executable_finder=lambda name: "hmmsearch (pyhmmer)",
            threads=1)
        cls.query = SearchQuery(identifier="sdr_seed", kind="profile",
                                profile_path=cls.record["path"])
        cls.database = SequenceDatabase(name="sdr_fixture",
                                        version="zenodo-7141435",
                                        fasta_path=str(FIXTURE))
        cls.hits = cls.adapter.search(cls.query, cls.database,
                                      cls.tmp / "work", max_evalue=1e-5)

    def test_the_shipped_parser_reads_real_hmmer_output(self) -> None:
        self.assertGreater(len(self.hits), 100)
        self.assertLessEqual(len(self.hits), len(self.entries))

    def test_every_hit_names_a_sequence_that_is_in_the_database(self) -> None:
        known = {e.identifier for e in self.entries}
        for hit in self.hits:
            self.assertIn(hit.subject_id, known)

    def test_the_seed_finds_itself_with_the_best_score(self) -> None:
        best = max(self.hits, key=lambda h: h.bitscore or 0.0)
        self.assertEqual(best.subject_id, self.entries[0].identifier)

    def test_the_numbers_real_hmmer_wrote_parse_as_numbers(self) -> None:
        for hit in self.hits:
            self.assertIsInstance(hit.evalue, float)
            self.assertIsInstance(hit.bitscore, float)
            self.assertLessEqual(hit.evalue, 1e-5)

    def test_coverage_never_exceeds_one_on_real_domain_envelopes(self) -> None:
        """The union-of-intervals rule, held to output HMMER actually wrote."""
        for hit in self.hits:
            self.assertIsNotNone(hit.query_coverage)
            self.assertGreaterEqual(hit.query_coverage, 0.0)
            self.assertLessEqual(hit.query_coverage, 1.0)

    def test_no_percent_identity_is_invented_from_a_profile_hit(self) -> None:
        for hit in self.hits:
            self.assertIsNone(hit.percent_identity)

    def test_the_hits_carry_the_database_label_they_came_from(self) -> None:
        for hit in self.hits:
            self.assertEqual(hit.database_name, "sdr_fixture")
            self.assertEqual(hit.database_version, "zenodo-7141435")
            self.assertEqual(hit.search_method, "hmmsearch")

    def test_the_run_is_recorded_with_the_threshold_it_used(self) -> None:
        # Order-independent: another test in this class runs a second search,
        # and a test that depends on which ran first is a flake waiting to
        # happen.
        thresholds = {i.max_evalue for i in self.runner.invocations}
        self.assertIn(1e-5, thresholds)
        for invocation in self.runner.invocations:
            self.assertEqual(invocation.target, FIXTURE)
            self.assertEqual(invocation.profile, Path(self.record["path"]))

    def test_the_table_on_disk_is_hmmer_s_own_format(self) -> None:
        written = list((self.tmp / "work").glob("*.domtbl"))
        self.assertEqual(len(written), 1)
        lines = written[0].read_text(encoding="utf-8").splitlines()
        self.assertTrue(lines[0].startswith("#"))
        data = [l for l in lines if l and not l.startswith("#")]
        self.assertTrue(data)
        for line in data:
            self.assertGreaterEqual(len(line.split()), 23)

    def test_a_tighter_threshold_returns_fewer_hits(self) -> None:
        """The -E the adapter passes reaches the search, rather than being dropped."""
        tight = self.adapter.search(self.query, self.database,
                                    self.tmp / "tight", max_evalue=1e-200)
        self.assertLess(len(tight), len(self.hits))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
