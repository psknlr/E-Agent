"""Tests for the ``mine_sequences`` interface.

The three things these tests are really protecting:

* a single-seed search cannot happen by accident,
* no code path loosens retention to reach ``initial_sequence_target``,
* none of blastp, mmseqs2 or hmmsearch is installed in this environment, so
  the adapters are exercised through both supported seams -- an injected
  runner, and a real fake executable found by ``shutil.which`` and run through
  a real subprocess.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from eagent.context import ExecutionPolicy, RunContext
from eagent.envelope import Status
from eagent.errors import ApprovalRequiredError, ToolUnavailableError
from eagent.provenance import RunManifest, sequence_hash
from eagent.schemas import Budget, EvidenceRef, EvidenceStrength, TaskSpec
from eagent.tools.mine_sequences import (
    EXTERNAL_SUBMISSION_GATE,
    BlastpAdapter,
    CommandResult,
    FamilyProfile,
    FastaEntry,
    HmmsearchAdapter,
    LengthExpectation,
    MMseqs2Adapter,
    MineSequences,
    RetentionPolicy,
    SearchExecutionError,
    SearchQuery,
    SeedSequence,
    SequenceDatabase,
    cluster_sequences,
    guard_external_submission,
    index_fasta,
    parse_fasta,
    read_fasta,
    write_fasta,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

#: A 300-residue scaffold, repeated so that every test sequence has a
#: predictable length and the length window can be counted by eye.
BASE = ("MKAVLTGAASGIGRATALLFAREGAKVVLADRNEEGLKETAELVRAEGGEAIAVKADVSKEEDV"
        "RALVDATVEKFGRLDILVNNAGITRDNLLMRMKDEEWDAVIDVNLKGVFNCTQAVARPMMKQRS"
        "GSIVNISSVVGLMGNAGQANYAAAKAGVIGFTKSLAREVASRGITVNAVAPGFIETDMTDALSE"
        "DLKEQMLTQIPLGRLGQPEEIAAAVAFLASDEAAYITGQTLHVNGGMYMV")


def _seq(tag: str, length: int = 300) -> str:
    """Deterministic sequence of a given length, distinct per ``tag``.

    The tag rotates the scaffold rather than changing one letter, so two tags
    never produce sequences that collapse to the same ``sequence_sha256`` --
    which would silently turn a de-duplication test into a no-op.
    """
    offset = sum(ord(c) * (i + 1) for i, c in enumerate(tag)) % len(BASE)
    body = (BASE * 6)[offset:offset + length]
    return "M" + body[1:]


def _evidence(strength: EvidenceStrength = EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL
              ) -> EvidenceRef:
    return EvidenceRef(source_type="publication", identifier="PMID:00000001",
                       strength=strength, extracted_by="human")


def _seed(accession: str, length: int = 300, *, supported: bool = True,
          published: bool | None = True) -> SeedSequence:
    strength = (EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL if supported
                else EvidenceStrength.ANNOTATION_ONLY)
    return SeedSequence(
        accession=accession, sequence=_seq(accession, length),
        family_template_id="fam.sdr.v1", evidence=[_evidence(strength)],
        is_published=published,
    )


def _ctx(tmp: Path, *, target: int = 1, allow_network: bool = False) -> RunContext:
    task = TaskSpec(task_id="t-mine", budget=Budget(initial_sequence_target=target))
    manifest = RunManifest(run_id="r1", task_id="t-mine", global_seed=7)
    return RunContext(task=task, workdir=tmp, manifest=manifest,
                      policy=ExecutionPolicy(allow_network=allow_network))


#: Columns are qseqid sseqid pident length qstart qend sstart send evalue bits qcovs
BLAST_ROWS = [
    ("seedA", "s1", "88.0", "300", "1", "300", "1", "300", "1e-170", "600", "98"),
    ("seedA", "s2", "90.0", "100", "1", "100", "1", "100", "1e-40", "200", "99"),
    ("seedA", "s3", "70.0", "300", "1", "300", "1", "300", "1e-120", "420", "97"),
    ("seedA", "s4", "65.0", "90", "1", "90", "1", "90", "1e-20", "120", "30"),
    ("seedB", "s5", "85.0", "300", "1", "300", "1", "300", "1e-160", "580", "96"),
    ("seedB", "s6", "40.0", "300", "1", "300", "1", "300", "2.0", "40", "95"),
]
BLAST_TSV = "".join("\t".join(r) + "\n" for r in BLAST_ROWS)


class FakeRunner:
    """Injected command runner: records argv and serves canned tool output.

    This is the seam the task description asks for. It writes to the output
    path the adapter chose, so the adapter's own file handling is exercised
    rather than bypassed.
    """

    def __init__(self, stdout: str = "", file_text: str = "",
                 returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.file_text = file_text
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def __call__(self, argv, *, cwd=None, timeout=None) -> CommandResult:
        argv = list(argv)
        self.calls.append(argv)
        name = Path(argv[0]).name
        if self.returncode == 0:
            if name == "mmseqs":
                Path(argv[4]).write_text(self.file_text, encoding="utf-8")
            elif name == "hmmsearch":
                Path(argv[argv.index("--domtblout") + 1]).write_text(
                    self.file_text, encoding="utf-8")
        return CommandResult(tuple(argv), self.returncode, self.stdout, self.stderr)


def _fake_which(name: str) -> str:
    return f"/fake/bin/{name}"


def _database(tmp: Path, entries: dict[str, str], *, name: str = "uniprotkb",
              version: str = "2024_01", remote: bool = False) -> SequenceDatabase:
    path = tmp / "db.fasta"
    write_fasta([FastaEntry(k, "test entry", v) for k, v in entries.items()], path)
    return SequenceDatabase(name=name, version=version, fasta_path=str(path),
                            is_remote=remote)


def _default_entries() -> dict[str, str]:
    good = _seq("s1")
    return {
        "s1": good,
        "s2": _seq("s2", 100),                   # fragment
        "s3": _seq("s3")[:299] + "X",            # non-standard residue
        "s4": _seq("s4"),                        # low coverage hit
        "s5": good,                              # duplicate of s1 by sequence
        "s6": _seq("s6"),                        # bad e-value
    }


# ---------------------------------------------------------------------------
# FASTA
# ---------------------------------------------------------------------------

class TestFasta(unittest.TestCase):
    def test_round_trip(self) -> None:
        entries = [FastaEntry("a", "first", "MKVLA"), FastaEntry("b", "", "GGWW")]
        with tempfile.TemporaryDirectory() as d:
            path = write_fasta(entries, Path(d) / "x.fasta", line_width=3)
            back = read_fasta(path)
        self.assertEqual([e.identifier for e in back], ["a", "b"])
        self.assertEqual([e.sequence for e in back], ["MKVLA", "GGWW"])
        self.assertEqual(back[0].description, "first")

    def test_softmask_is_recorded_not_discarded(self) -> None:
        entries = parse_fasta(">a\nMKvla\n")
        self.assertEqual(entries[0].sequence, "MKVLA")
        self.assertTrue(entries[0].was_softmasked)

    def test_trailing_stop_codon_removed_internal_kept(self) -> None:
        self.assertEqual(parse_fasta(">a\nMKV*\n")[0].sequence, "MKV")
        self.assertEqual(parse_fasta(">a\nMK*V\n")[0].sequence, "MK*V")

    def test_sequence_before_header_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            parse_fasta("MKVLA\n>a\nMK\n")

    def test_empty_header_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            parse_fasta(">\nMKVLA\n")

    def test_index_reports_duplicates_and_adds_pipe_keys(self) -> None:
        entries = [FastaEntry("sp|P1|AAA", "", "MKV"), FastaEntry("sp|P1|AAA", "", "MKW"),
                   FastaEntry("sp|P2|BBB", "", "MKY")]
        index, dups = index_fasta(entries)
        self.assertEqual(dups, ["sp|P1|AAA"])
        self.assertIn("P2", index)
        self.assertEqual(index["P2"].sequence, "MKY")

    def test_hash_matches_pipeline_join_key(self) -> None:
        e = FastaEntry("a", "", "mkvla")
        self.assertEqual(parse_fasta(">a\nmkvla\n")[0].sequence_sha256,
                         sequence_hash("MKVLA"))
        self.assertEqual(e.sequence_sha256, sequence_hash("mkvla"))


# ---------------------------------------------------------------------------
# adapters
# ---------------------------------------------------------------------------

class TestAdapters(unittest.TestCase):
    def test_missing_binary_raises_with_install_hint(self) -> None:
        for cls in (BlastpAdapter, MMseqs2Adapter, HmmsearchAdapter):
            adapter = cls(runner=FakeRunner(), executable_finder=lambda n: None)
            self.assertFalse(adapter.is_available())
            with self.assertRaises(ToolUnavailableError) as cm:
                adapter.resolve_binary()
            self.assertIn(cls.binary, str(cm.exception))
            self.assertTrue(cm.exception.hint, "every adapter must offer an install hint")

    def test_these_binaries_are_genuinely_absent_here(self) -> None:
        # The premise of the injected-runner seam. If this ever fails, the
        # adapters should be re-tested against the real tools as well.
        for binary in ("blastp", "mmseqs", "hmmsearch"):
            self.assertIsNone(shutil.which(binary))

    def test_blastp_argv_and_parsing(self) -> None:
        runner = FakeRunner(stdout=BLAST_TSV)
        adapter = BlastpAdapter(runner=runner, executable_finder=_fake_which)
        with tempfile.TemporaryDirectory() as d:
            db = _database(Path(d), {"s1": _seq("s1")})
            hits = adapter.search(SearchQuery("seedA", "sequence", sequence=_seq("a")),
                                  db, Path(d), max_evalue=1e-5)
        argv = runner.calls[0]
        self.assertEqual(Path(argv[0]).name, "blastp")
        self.assertIn("-outfmt", argv)
        self.assertIn("qcovs", argv[argv.index("-outfmt") + 1])
        self.assertEqual(len(hits), len(BLAST_ROWS))
        first = hits[0]
        self.assertEqual(first.subject_id, "s1")
        self.assertAlmostEqual(first.percent_identity, 88.0)
        # BLAST qcovs is a percent; the pool stores a fraction.
        self.assertAlmostEqual(first.query_coverage, 0.98)
        self.assertAlmostEqual(first.evalue, 1e-170)

    def test_blastp_short_row_is_an_error_not_a_silent_shift(self) -> None:
        runner = FakeRunner(stdout="seedA\ts1\t88.0\n")
        adapter = BlastpAdapter(runner=runner, executable_finder=_fake_which)
        with tempfile.TemporaryDirectory() as d:
            db = _database(Path(d), {"s1": _seq("s1")})
            with self.assertRaises(SearchExecutionError):
                adapter.search(SearchQuery("seedA", "sequence", sequence=_seq("a")),
                               db, Path(d), max_evalue=1e-5)

    def test_nonzero_exit_is_an_error(self) -> None:
        runner = FakeRunner(stdout="", returncode=2, stderr="BLAST Database error")
        adapter = BlastpAdapter(runner=runner, executable_finder=_fake_which)
        with tempfile.TemporaryDirectory() as d:
            db = _database(Path(d), {"s1": _seq("s1")})
            with self.assertRaises(SearchExecutionError) as cm:
                adapter.search(SearchQuery("seedA", "sequence", sequence=_seq("a")),
                               db, Path(d), max_evalue=1e-5)
        self.assertIn("BLAST Database error", str(cm.exception))

    def test_mmseqs_fraction_identity_is_converted_to_percent(self) -> None:
        text = "seedA\ts1\t0.873\t0.95\t1e-90\t400\n"
        runner = FakeRunner(file_text=text)
        adapter = MMseqs2Adapter(runner=runner, executable_finder=_fake_which)
        with tempfile.TemporaryDirectory() as d:
            db = _database(Path(d), {"s1": _seq("s1")})
            hits = adapter.search(SearchQuery("seedA", "sequence", sequence=_seq("a")),
                                  db, Path(d), max_evalue=1e-5)
        self.assertAlmostEqual(hits[0].percent_identity, 87.3)
        self.assertAlmostEqual(hits[0].query_coverage, 0.95)

    def test_hmmsearch_reports_no_identity_and_unions_domain_envelopes(self) -> None:
        # qlen = 200; two domains covering 1-100 and 51-150 -> union 150/200.
        cols_a = ["s1", "-", "300", "fam.sdr.hmm", "-", "200", "1e-60", "210.0", "0.0",
                  "1", "2", "1e-30", "1e-28", "105.0", "0.0", "1", "100",
                  "10", "110", "10", "110", "0.95", "desc"]
        cols_b = list(cols_a)
        cols_b[15], cols_b[16] = "51", "150"
        text = ("#  comment line\n" + " ".join(cols_a) + "\n" + " ".join(cols_b) + "\n")
        runner = FakeRunner(file_text=text)
        adapter = HmmsearchAdapter(runner=runner, executable_finder=_fake_which)
        with tempfile.TemporaryDirectory() as d:
            db = _database(Path(d), {"s1": _seq("s1")})
            hits = adapter.search(
                SearchQuery("fam.sdr.hmm", "profile", profile_path="x.hmm",
                            family_template_id="fam.sdr.v1"),
                db, Path(d), max_evalue=1e-5)
        self.assertEqual(len(hits), 1)
        self.assertIsNone(hits[0].percent_identity)
        self.assertAlmostEqual(hits[0].query_coverage, 150 / 200)
        self.assertEqual(hits[0].family_template_id, "fam.sdr.v1")

    def test_wrong_query_kind_is_refused(self) -> None:
        adapter = HmmsearchAdapter(runner=FakeRunner(), executable_finder=_fake_which)
        with tempfile.TemporaryDirectory() as d:
            db = _database(Path(d), {"s1": _seq("s1")})
            with self.assertRaises(SearchExecutionError):
                adapter.search(SearchQuery("seedA", "sequence", sequence="MK"),
                               db, Path(d), max_evalue=1e-5)

    def test_remote_flag_is_refused_when_network_is_off(self) -> None:
        adapter = BlastpAdapter(runner=FakeRunner(stdout=""),
                                executable_finder=_fake_which,
                                extra_args=["-remote"])
        with tempfile.TemporaryDirectory() as d:
            db = _database(Path(d), {"s1": _seq("s1")})
            with self.assertRaises(ToolUnavailableError):
                adapter.search(SearchQuery("seedA", "sequence", sequence=_seq("a")),
                               db, Path(d), max_evalue=1e-5, allow_network=False)

    def test_fake_executable_on_path_through_a_real_subprocess(self) -> None:
        """The second seam: a real binary, found by ``shutil.which``, really run."""
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            bin_dir = tmp / "bin"
            bin_dir.mkdir()
            script = bin_dir / "blastp"
            script.write_text(
                f"#!{sys.executable}\n"
                "import sys\n"
                f"sys.stdout.write({BLAST_TSV!r})\n",
                encoding="utf-8",
            )
            script.chmod(0o755)
            adapter = BlastpAdapter(
                executable_finder=lambda n: shutil.which(n, path=str(bin_dir)),
            )
            self.assertTrue(adapter.is_available())
            db = _database(tmp, {"s1": _seq("s1")})
            hits = adapter.search(SearchQuery("seedA", "sequence", sequence=_seq("a")),
                                  db, tmp, max_evalue=1e-5)
        self.assertEqual(len(hits), len(BLAST_ROWS))
        self.assertEqual(hits[0].subject_id, "s1")


# ---------------------------------------------------------------------------
# policy objects and clustering
# ---------------------------------------------------------------------------

class TestPolicyAndClustering(unittest.TestCase):
    def test_length_window_comes_from_measured_seeds(self) -> None:
        exp = LengthExpectation.from_seeds([_seed("a", 300), _seed("b", 320)])
        self.assertEqual(exp.min_length, 240)
        self.assertEqual(exp.max_length, 400)
        self.assertIn("seed length", exp.source)
        self.assertEqual(exp.classify(100), "fragment")
        self.assertEqual(exp.classify(900), "implausibly_long")
        self.assertIsNone(exp.classify(300))

    def test_length_window_needs_a_source(self) -> None:
        with self.assertRaises(ValueError):
            LengthExpectation(min_length=10, max_length=20, source="  ")

    def test_database_version_is_mandatory(self) -> None:
        with self.assertRaises(ValueError):
            SequenceDatabase(name="uniprotkb", version="  ", fasta_path="/tmp/x.fasta")

    def test_impossible_identity_window_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RetentionPolicy(min_percent_identity=90.0, max_percent_identity=50.0)

    def test_clustering_groups_near_identical_and_separates_distant(self) -> None:
        a = _seq("a")
        b = a[:290] + "AAAAAAAAAA"          # ~97% identical to a
        c = "".join(reversed(a))            # same composition, different order
        report = cluster_sequences([("a", a), ("b", b), ("c", c)],
                                   identity_threshold=0.9)
        self.assertEqual(report.assignment["a"], report.assignment["b"])
        self.assertNotEqual(report.assignment["a"], report.assignment["c"])
        self.assertEqual(len(report.assignment), 3, "clustering removes nothing")

    def test_clustering_budget_is_reported_not_hidden(self) -> None:
        seqs = [(f"k{i}", _seq(chr(65 + i))) for i in range(4)]
        report = cluster_sequences(seqs, identity_threshold=0.99,
                                   kmer_prefilter=None, max_alignments=1)
        self.assertTrue(report.budget_exhausted)
        self.assertEqual(len(report.assignment), 4)


# ---------------------------------------------------------------------------
# disclosure guard
# ---------------------------------------------------------------------------

class TestExternalSubmissionGuard(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_local_only_search_needs_no_approval(self) -> None:
        ctx = _ctx(self.tmp)
        db = _database(self.tmp, {"s1": _seq("s1")})
        guard_external_submission(
            ctx, [_seed("seedA", published=None).to_query()],
            [BlastpAdapter(executable_finder=_fake_which)], [db],
        )

    def test_unpublished_sequence_to_remote_service_needs_approval(self) -> None:
        ctx = _ctx(self.tmp, allow_network=True)
        db = _database(self.tmp, {"s1": _seq("s1")}, remote=True)
        with self.assertRaises(ApprovalRequiredError) as cm:
            guard_external_submission(
                ctx, [_seed("seedA", published=None).to_query()],
                [BlastpAdapter(executable_finder=_fake_which)], [db],
            )
        self.assertEqual(cm.exception.gate, EXTERNAL_SUBMISSION_GATE)

    def test_unknown_publication_status_is_treated_as_unpublished(self) -> None:
        ctx = _ctx(self.tmp, allow_network=True)
        db = _database(self.tmp, {"s1": _seq("s1")}, remote=True)
        with self.assertRaises(ApprovalRequiredError):
            guard_external_submission(ctx, [_seed("x", published=None).to_query()],
                                      [], [db])
        # A published seed needs no approval for the same destination.
        guard_external_submission(ctx, [_seed("x", published=True).to_query()],
                                  [], [db])

    def test_recorded_approval_clears_the_gate(self) -> None:
        ctx = _ctx(self.tmp, allow_network=True)
        db = _database(self.tmp, {"s1": _seq("s1")}, remote=True)
        ctx.manifest.record_approval(EXTERNAL_SUBMISSION_GATE, "approve", "tester")
        guard_external_submission(ctx, [_seed("x", published=False).to_query()],
                                  [], [db])

    def test_profile_query_is_publishable_by_construction(self) -> None:
        q = FamilyProfile(profile_id="PF00106", profile_path="p.hmm",
                          family_template_id="fam.sdr.v1", source="Pfam 36.0").to_query()
        self.assertTrue(q.is_published)
        self.assertEqual(q.kind, "profile")


# ---------------------------------------------------------------------------
# the interface
# ---------------------------------------------------------------------------

class TestMineSequencesInterface(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.iface = MineSequences()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, *, target: int = 1, seeds=None, allow_network: bool = False,
             entries=None, remote_db: bool = False, **kwargs):
        ctx = _ctx(self.tmp, target=target, allow_network=allow_network)
        db = _database(self.tmp, entries or _default_entries(), remote=remote_db)
        runner = FakeRunner(stdout=BLAST_TSV)
        adapter = BlastpAdapter(runner=runner, executable_finder=_fake_which)
        kwargs.setdefault("adapters", [adapter])
        kwargs.setdefault("databases", [db])
        result = self.iface.run(
            ctx, seeds=seeds if seeds is not None else [_seed("seedA"), _seed("seedB")],
            **kwargs,
        )
        return ctx, result

    # -- multi-seed discipline ------------------------------------------
    def test_single_seed_is_refused_by_default(self) -> None:
        _, result = self._run(seeds=[_seed("seedA")])
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(result.blockers[0].code, "single_seed_not_authorised")

    def test_single_seed_needs_a_justification(self) -> None:
        _, result = self._run(seeds=[_seed("seedA")], allow_single_seed=True)
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(result.blockers[0].code, "single_seed_unjustified")

    def test_single_seed_with_justification_runs_and_is_flagged(self) -> None:
        _, result = self._run(
            seeds=[_seed("seedA")], allow_single_seed=True,
            single_seed_justification="only characterised member of this clade",
        )
        self.assertTrue(result.status.usable)
        codes = {f.code for f in result.qc_flags}
        self.assertIn("single_seed_mining", codes)
        self.assertTrue(result.provenance.parameters["single_seed_mode"])
        self.assertIn("only characterised member",
                      result.provenance.parameters["single_seed_justification"])

    def test_no_seeds_fails(self) -> None:
        _, result = self._run(seeds=[])
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(result.blockers[0].code, "no_seeds")

    def test_unsupported_seeds_are_dropped_and_an_all_unsupported_set_fails(self) -> None:
        _, result = self._run(seeds=[_seed("seedA", supported=False),
                                     _seed("seedB", supported=False)])
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(result.blockers[0].code, "no_supported_seed")

        _, mixed = self._run(seeds=[_seed("seedA"), _seed("seedB", supported=False)])
        self.assertTrue(mixed.status.usable)
        self.assertIn("seed_without_experimental_support",
                      {f.code for f in mixed.qc_flags})
        self.assertEqual(mixed.provenance.parameters["n_seeds_dropped_unsupported"], 1)

    # -- offline behaviour -------------------------------------------------
    def test_remote_database_is_refused_offline(self) -> None:
        _, result = self._run(remote_db=True)
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(result.blockers[0].code, "network_disabled")
        self.assertIn("remote", result.message)

    def test_offline_is_reported_before_an_approval_that_cannot_matter(self) -> None:
        # Remote database, unpublished seeds, network off: the operator needs
        # to hear about the offline policy, not be sent to clear a gate for a
        # transfer that could not happen.
        _, result = self._run(remote_db=True,
                              seeds=[_seed("seedA", published=None),
                                     _seed("seedB", published=None)])
        self.assertEqual(result.blockers[0].code, "network_disabled")

    def test_unpublished_seed_to_a_remote_database_needs_approval(self) -> None:
        _, result = self._run(remote_db=True, allow_network=True,
                              seeds=[_seed("seedA", published=None),
                                     _seed("seedB", published=True)])
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(result.blockers[0].code, "approval_required")
        self.assertIn(EXTERNAL_SUBMISSION_GATE,
                      [a.params.get("gate") for a in result.next_actions])
        self.assertTrue(any(a.requires_human for a in result.next_actions))

    def test_cache_miss_names_the_file_it_needed(self) -> None:
        ctx = _ctx(self.tmp)
        db = SequenceDatabase(name="uniprotkb", version="2024_01",
                              fasta_path=str(self.tmp / "absent.fasta"))
        result = self.iface.run(
            ctx, seeds=[_seed("seedA"), _seed("seedB")], databases=[db],
            adapters=[BlastpAdapter(runner=FakeRunner(stdout=BLAST_TSV),
                                    executable_finder=_fake_which)],
        )
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(result.blockers[0].code, "cache_miss")
        self.assertIn("absent.fasta", result.message)
        self.assertIn("uniprotkb@2024_01", result.message)

    def test_missing_binary_is_reported_not_substituted(self) -> None:
        ctx = _ctx(self.tmp)
        db = _database(self.tmp, _default_entries())
        result = self.iface.run(
            ctx, seeds=[_seed("seedA"), _seed("seedB")], databases=[db],
            adapters=[BlastpAdapter(runner=FakeRunner(),
                                    executable_finder=lambda n: None)],
        )
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(result.blockers[0].code, "search_failed")
        self.assertIn("blastp", result.message)

    # -- QC and retention ---------------------------------------------------
    def test_pool_qc_dedup_and_exclusions(self) -> None:
        ctx, result = self._run(target=1)
        self.assertTrue(result.status.usable, result.message)
        records = result.data["sequence_records"]
        accessions = {r["accession"] for r in records}
        # s1/s5 are the same sequence -> one record; s2 fragment, s3 X,
        # s4 coverage, s6 e-value -> all excluded.
        self.assertEqual(len(records), 1, accessions)
        self.assertIn(accessions.pop(), {"s1", "s5"})

        rows = {r["accession"]: r for r in result.data["retrieval_rows"]}
        self.assertEqual(rows["s2"]["status"], "excluded")
        self.assertIn("fragment", rows["s2"]["reason"])
        self.assertEqual(rows["s3"]["status"], "excluded")
        self.assertIn("nonstandard_residues", rows["s3"]["reason"])
        self.assertEqual(rows["s4"]["status"], "excluded")
        self.assertIn("coverage", rows["s4"]["reason"])
        self.assertEqual(rows["s6"]["status"], "excluded")
        self.assertIn("evalue", rows["s6"]["reason"])

    def test_every_retained_record_carries_full_retrieval_provenance(self) -> None:
        _, result = self._run(target=1)
        for rec in result.data["sequence_records"]:
            self.assertTrue(rec["seed_accession"])
            self.assertEqual(rec["search_method"], "blastp")
            self.assertIsNotNone(rec["percent_identity"])
            self.assertIsNotNone(rec["query_coverage"])
            self.assertIsNotNone(rec["evalue"])
            self.assertEqual(rec["source_database"], "uniprotkb")
            self.assertEqual(rec["database_version"], "2024_01")

    def test_artifacts_are_written_and_hashed(self) -> None:
        ctx, result = self._run(target=1)
        fasta = result.artifact("candidate_sequences")
        tsv = result.artifact("retrieval_provenance")
        self.assertTrue(Path(fasta.path).is_file())
        self.assertTrue(Path(tsv.path).is_file())
        self.assertTrue(fasta.sha256 and tsv.sha256)
        self.assertEqual(fasta.n_records, len(result.data["sequence_records"]))
        header = Path(tsv.path).read_text(encoding="utf-8").splitlines()[0]
        for column in ("seed_accession", "search_method", "percent_identity",
                       "query_coverage", "evalue", "database_version"):
            self.assertIn(column, header)
        # Excluded rows are kept in the table so the pool can be audited.
        body = Path(tsv.path).read_text(encoding="utf-8").splitlines()[1:]
        self.assertEqual(len(body), len(result.data["retrieval_rows"]))
        self.assertTrue(any("excluded" in line for line in body))

    def test_provenance_records_databases_parameters_and_seed(self) -> None:
        ctx, result = self._run(target=1)
        prov = result.provenance
        self.assertEqual(prov.tool, "mine_sequences")
        self.assertEqual(prov.databases, {"uniprotkb": "2024_01"})
        self.assertEqual(prov.random_seed, ctx.seed_for("mine_sequences"))
        self.assertIn("retention_policy", prov.parameters)
        self.assertIn("inputs_sha256", prov.to_dict())
        self.assertIn("seeds", prov.inputs_sha256)

    # -- coverage honesty -----------------------------------------------------
    def test_short_pool_is_reported_not_padded(self) -> None:
        _, small = self._run(target=1)
        _, large = self._run(target=50)
        self.assertIs(small.status, Status.SUCCESS)
        self.assertIs(large.status, Status.PARTIAL)
        self.assertIn("coverage_shortfall", {f.code for f in large.qc_flags})
        coverage = large.data["coverage"]
        self.assertEqual(coverage["target"], 50)
        self.assertEqual(coverage["retained"], 1)
        self.assertFalse(coverage["policy_relaxation_performed"])
        self.assertIn("excluded_by_reason", coverage)
        # The pool is identical either way: the target did not move a threshold.
        self.assertEqual([r["sequence_sha256"] for r in small.data["sequence_records"]],
                         [r["sequence_sha256"] for r in large.data["sequence_records"]])
        self.assertEqual(small.provenance.parameters["retention_policy"],
                         large.provenance.parameters["retention_policy"])

    def test_empty_pool_fails_rather_than_returning_nothing_quietly(self) -> None:
        entries = {"s2": _seq("s2", 100)}       # only a fragment is retrievable
        rows = [r for r in BLAST_ROWS if r[1] == "s2"]
        runner = FakeRunner(stdout="".join("\t".join(r) + "\n" for r in rows))
        ctx = _ctx(self.tmp, target=10)
        db = _database(self.tmp, entries)
        result = self.iface.run(
            ctx, seeds=[_seed("seedA"), _seed("seedB")], databases=[db],
            adapters=[BlastpAdapter(runner=runner, executable_finder=_fake_which)],
        )
        self.assertIs(result.status, Status.FAILED)
        self.assertIn("empty_pool", {f.code for f in result.qc_flags})

    def test_family_evidence_absent_excludes_the_hit(self) -> None:
        seeds = [SeedSequence(accession="seedA", sequence=_seq("a"),
                              evidence=[_evidence()], is_published=True),
                 SeedSequence(accession="seedB", sequence=_seq("b"),
                              evidence=[_evidence()], is_published=True)]
        _, result = self._run(target=1, seeds=seeds)
        self.assertIs(result.status, Status.FAILED)
        rows = result.data["retrieval_rows"]
        self.assertTrue(all(r["family_evidence"] == "none" for r in rows))
        self.assertTrue(all("family_evidence_absent" in r["reason"] for r in rows))

    def test_unresolved_subject_ids_are_dropped_not_invented(self) -> None:
        entries = {"s1": _seq("s1")}            # the other hit ids are not in the db
        _, result = self._run(target=1, entries=entries)
        self.assertIn("unresolved_subject_ids", {f.code for f in result.qc_flags})
        self.assertEqual(len(result.data["sequence_records"]), 1)

    def test_cluster_ids_are_assigned_to_every_retained_row(self) -> None:
        entries = {"s1": _seq("s1"), "s3": _seq("s3")}
        rows = [r for r in BLAST_ROWS if r[1] in ("s1", "s3")]
        runner = FakeRunner(stdout="".join("\t".join(r) + "\n" for r in rows))
        ctx = _ctx(self.tmp, target=1)
        db = _database(self.tmp, entries)
        result = self.iface.run(
            ctx, seeds=[_seed("seedA"), _seed("seedB")], databases=[db],
            adapters=[BlastpAdapter(runner=runner, executable_finder=_fake_which)],
        )
        retained = [r for r in result.data["retrieval_rows"] if r["status"] == "retained"]
        self.assertTrue(retained)
        for row in retained:
            self.assertTrue(row["cluster_id"])
        self.assertEqual(result.provenance.parameters["cluster_method"],
                         "greedy_pure_python")

    def test_next_actions_point_at_inputs_never_at_thresholds(self) -> None:
        _, result = self._run(target=50)
        rationales = " ".join(a.rationale for a in result.next_actions).lower()
        self.assertIn("widen the *inputs*", rationales)
        self.assertNotIn("lower the threshold", rationales)
        self.assertIn("annotate_family", {a.action for a in result.next_actions})


if __name__ == "__main__":  # pragma: no cover - pytest may not be installed
    unittest.main(verbosity=2)
