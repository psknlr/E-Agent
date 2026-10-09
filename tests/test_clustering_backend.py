"""Tests for the real clusterer, and for what happens when it is absent.

``cluster_sequences`` documented mmseqs2 ``easy-cluster`` as "the preferred
clusterer whenever it is installed" and there was no adapter for it: every run
got the greedy pure-Python fallback, and nothing in the output said so.

That is not a cosmetic difference. Cluster size is the weight the diversity
step selects by and the unit the cluster cap limits, so a pool whose tail sits
in spurious singletons -- which is what the fallback produces once its
alignment budget runs out -- spends slots on near-duplicates and reports them
as pocket coverage.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from eagent.errors import ToolUnavailableError
from eagent.tools.mine_sequences import (
    CommandResult, IDENTITY_DEFINITIONS, MMseqs2ClusterAdapter,
    SearchExecutionError, cluster_sequences, read_fasta,
)

A = "MKAAVLYEFGKPLEIKEVEVAPPKAHEVRIKIAYTGVCHTDAYTLSGADPEGLFPVILGHEGAGIVESVGEGVT"
B = A[:-1] + "A"          # one substitution from A
C = "MSTNPKPQRKTKRNTNRRPQDVKFPGGGQIVGGVYLLPRRGPRLGVRATRKTSERSQPRGRRQPIPKARRPEG"

ITEMS = [("sha256:aaa", A), ("sha256:bbb", B), ("sha256:ccc", C)]


class ClusterRunner:
    """A runner that behaves like ``mmseqs easy-cluster``: writes a TSV."""

    def __init__(self, rows: str | None = None, returncode: int = 0,
                 write: bool = True, stderr: str = "") -> None:
        self.rows = rows
        self.returncode = returncode
        self.write = write
        self.stderr = stderr
        self.calls: list[list[str]] = []

    def __call__(self, argv, *, cwd=None, timeout=None) -> CommandResult:
        argv = list(argv)
        self.calls.append(argv)
        if self.returncode == 0 and self.write:
            prefix = pathlib.Path(argv[3])
            rows = self.rows
            if rows is None:
                # Default: A and B together under A, C on its own. Surrogate
                # ids are what the adapter writes into the FASTA.
                rows = "s0\ts0\ns0\ts1\ns2\ts2\n"
            pathlib.Path(f"{prefix}_cluster.tsv").write_text(rows,
                                                             encoding="utf-8")
        return CommandResult(tuple(argv), self.returncode, "", self.stderr)


def installed(name: str) -> str:
    return f"/fake/bin/{name}"


def absent(name: str) -> None:
    return None


class _InTemp(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = pathlib.Path(self._tmp.name)


class TheAdapterRuns(_InTemp):
    def adapter(self, runner=None, **kw) -> MMseqs2ClusterAdapter:
        return MMseqs2ClusterAdapter(runner=runner or ClusterRunner(),
                                     executable_finder=installed, **kw)

    def test_the_clusters_come_from_the_tool(self) -> None:
        report = self.adapter().cluster(ITEMS, 0.7, self.tmp)
        self.assertEqual(report.method, "mmseqs2_easy_cluster")
        self.assertEqual(len(report.clusters), 2)
        sizes = sorted(c.size for c in report.clusters)
        self.assertEqual(sizes, [1, 2])

    def test_every_input_key_is_assigned(self) -> None:
        report = self.adapter().cluster(ITEMS, 0.7, self.tmp)
        self.assertEqual(set(report.assignment), {k for k, _ in ITEMS})

    def test_the_representative_is_a_pool_key_not_a_surrogate(self) -> None:
        report = self.adapter().cluster(ITEMS, 0.7, self.tmp)
        for cluster in report.clusters:
            self.assertTrue(cluster.representative.startswith("sha256:"),
                            cluster.representative)
            self.assertIn(cluster.representative, cluster.members)

    def test_the_command_line_is_recorded(self) -> None:
        report = self.adapter().cluster(ITEMS, 0.7, self.tmp)
        self.assertIn("easy-cluster", report.command)
        self.assertIn("--min-seq-id", report.command)
        self.assertIn("--cov-mode", report.command)
        self.assertIn("-c", report.command)

    def test_the_threshold_reaches_the_command(self) -> None:
        runner = ClusterRunner()
        self.adapter(runner).cluster(ITEMS, 0.42, self.tmp)
        argv = runner.calls[0]
        self.assertEqual(argv[argv.index("--min-seq-id") + 1], "0.42")

    def test_the_identity_definition_travels_with_the_report(self) -> None:
        """0.7 under one definition is not 0.7 under another."""
        report = self.adapter().cluster(ITEMS, 0.7, self.tmp)
        self.assertEqual(report.identity_definition,
                         IDENTITY_DEFINITIONS["mmseqs2_easy_cluster"])
        fallback = cluster_sequences(ITEMS, 0.7)
        self.assertNotEqual(fallback.identity_definition,
                            report.identity_definition)

    def test_ids_do_not_travel_through_the_fasta(self) -> None:
        """A key with whitespace in it must not become two keys."""
        awkward = [("sha256:a a", A), ("sha256:b\tb", B)]
        runner = ClusterRunner(rows="s0\ts0\ns0\ts1\n")
        report = self.adapter(runner).cluster(awkward, 0.7, self.tmp)
        self.assertEqual(set(report.assignment), {"sha256:a a", "sha256:b\tb"})
        written = read_fasta(self.tmp / "cluster_input.fasta")
        self.assertEqual([e.identifier for e in written], ["s0", "s1"])

    def test_an_empty_pool_is_an_empty_clustering(self) -> None:
        report = self.adapter().cluster([], 0.7, self.tmp)
        self.assertEqual(report.clusters, [])
        self.assertEqual(report.assignment, {})


class TheAdapterRefuses(_InTemp):
    def adapter(self, runner, finder=installed) -> MMseqs2ClusterAdapter:
        return MMseqs2ClusterAdapter(runner=runner, executable_finder=finder)

    def test_a_missing_binary_raises_with_an_install_hint(self) -> None:
        with self.assertRaises(ToolUnavailableError) as ctx:
            self.adapter(ClusterRunner(), absent).cluster(ITEMS, 0.7, self.tmp)
        self.assertIn("mmseqs2", str(ctx.exception))

    def test_a_non_zero_exit_is_not_an_empty_clustering(self) -> None:
        with self.assertRaises(SearchExecutionError) as ctx:
            self.adapter(ClusterRunner(returncode=1, stderr="out of memory")
                         ).cluster(ITEMS, 0.7, self.tmp)
        self.assertIn("out of memory", str(ctx.exception))

    def test_a_missing_output_file_is_not_a_pool_of_singletons(self) -> None:
        with self.assertRaises(SearchExecutionError) as ctx:
            self.adapter(ClusterRunner(write=False)).cluster(ITEMS, 0.7, self.tmp)
        self.assertIn("wrote no", str(ctx.exception))

    def test_a_sequence_missing_from_the_output_is_refused(self) -> None:
        """An unassigned sequence reads as a singleton cluster downstream."""
        with self.assertRaises(SearchExecutionError) as ctx:
            self.adapter(ClusterRunner(rows="s0\ts0\ns0\ts1\n")
                         ).cluster(ITEMS, 0.7, self.tmp)
        self.assertIn("absent from its output", str(ctx.exception))

    def test_an_unknown_surrogate_is_refused(self) -> None:
        with self.assertRaises(SearchExecutionError) as ctx:
            self.adapter(ClusterRunner(rows="s0\ts0\ns9\ts9\n")
                         ).cluster(ITEMS, 0.7, self.tmp)
        self.assertIn("another clustering", str(ctx.exception))

    def test_a_malformed_row_is_refused(self) -> None:
        with self.assertRaises(SearchExecutionError) as ctx:
            self.adapter(ClusterRunner(rows="s0\ts0\ts1\n")
                         ).cluster(ITEMS, 0.7, self.tmp)
        self.assertIn("expected 2", str(ctx.exception))

    def test_a_duplicate_key_is_refused(self) -> None:
        with self.assertRaises(SearchExecutionError) as ctx:
            self.adapter(ClusterRunner()).cluster(
                [("sha256:aaa", A), ("sha256:aaa", B)], 0.7, self.tmp)
        self.assertIn("twice", str(ctx.exception))


class TheFallbackAnnouncesItself(_InTemp):
    def test_no_adapter_means_no_announcement_is_needed(self) -> None:
        report = cluster_sequences(ITEMS, 0.7)
        self.assertEqual(report.method, "greedy_pure_python")
        self.assertEqual(report.fallback_reason, "")

    def test_an_uninstalled_adapter_is_reported(self) -> None:
        report = cluster_sequences(
            ITEMS, 0.7, workdir=self.tmp,
            adapter=MMseqs2ClusterAdapter(runner=ClusterRunner(),
                                          executable_finder=absent))
        self.assertEqual(report.method, "greedy_pure_python")
        self.assertIn("not on PATH", report.fallback_reason)
        self.assertIn("coarser", report.fallback_reason)

    def test_an_installed_adapter_is_used(self) -> None:
        report = cluster_sequences(
            ITEMS, 0.7, workdir=self.tmp,
            adapter=MMseqs2ClusterAdapter(runner=ClusterRunner(),
                                          executable_finder=installed))
        self.assertEqual(report.method, "mmseqs2_easy_cluster")
        self.assertEqual(report.fallback_reason, "")

    def test_an_adapter_with_no_workdir_falls_back_and_says_so(self) -> None:
        report = cluster_sequences(
            ITEMS, 0.7,
            adapter=MMseqs2ClusterAdapter(runner=ClusterRunner(),
                                          executable_finder=installed))
        self.assertEqual(report.method, "greedy_pure_python")
        self.assertIn("working directory", report.fallback_reason)

    def test_require_adapter_refuses_rather_than_substituting(self) -> None:
        with self.assertRaises(ToolUnavailableError) as ctx:
            cluster_sequences(
                ITEMS, 0.7, workdir=self.tmp, require_adapter=True,
                adapter=MMseqs2ClusterAdapter(runner=ClusterRunner(),
                                              executable_finder=absent))
        self.assertIn("will not substitute", str(ctx.exception))

    def test_require_adapter_with_no_adapter_at_all_refuses(self) -> None:
        with self.assertRaises(ToolUnavailableError):
            cluster_sequences(ITEMS, 0.7, require_adapter=True)

    def test_the_report_serialises_for_provenance(self) -> None:
        report = cluster_sequences(
            ITEMS, 0.7, workdir=self.tmp,
            adapter=MMseqs2ClusterAdapter(runner=ClusterRunner(),
                                          executable_finder=installed))
        payload = report.to_dict()
        self.assertEqual(payload["method"], "mmseqs2_easy_cluster")
        self.assertTrue(payload["identity_definition"])
        self.assertIn("easy-cluster", payload["command"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TheStepReportsWhichClustererRan(unittest.TestCase):
    """End to end: the pool's grain is a fact about the run, so it is recorded."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = pathlib.Path(self._tmp.name)

    def run_step(self, **kwargs):
        from test_mine_sequences import (
            BLAST_TSV, BlastpAdapter, FakeRunner, _ctx, _database,
            _default_entries, _fake_which, _seed,
        )
        from eagent.tools.mine_sequences import MineSequences
        ctx = _ctx(self.tmp, target=1)
        db = _database(self.tmp, _default_entries())
        search = BlastpAdapter(runner=FakeRunner(stdout=BLAST_TSV),
                               executable_finder=_fake_which)
        return MineSequences().run(
            ctx, seeds=[_seed("seedA"), _seed("seedB")],
            adapters=[search], databases=[db], **kwargs)

    def test_the_fallback_is_flagged_when_mmseqs_is_absent(self) -> None:
        result = self.run_step(
            clustering_adapter=MMseqs2ClusterAdapter(
                runner=ClusterRunner(), executable_finder=absent))
        self.assertIn("clustering_fallback", {f.code for f in result.qc_flags})
        self.assertEqual(
            result.provenance.parameters["clustering"]["method"],
            "greedy_pure_python")

    def test_the_fallback_is_an_open_question_not_just_a_note(self) -> None:
        result = self.run_step(
            clustering_adapter=MMseqs2ClusterAdapter(
                runner=ClusterRunner(), executable_finder=absent))
        self.assertIn("clustering_fallback",
                      {u.code for u in result.uncertainty})

    def test_requiring_the_adapter_fails_the_step_rather_than_degrading(self) -> None:
        from eagent.envelope import Status
        result = self.run_step(
            clustering_adapter=MMseqs2ClusterAdapter(
                runner=ClusterRunner(), executable_finder=absent),
            require_clustering_adapter=True)
        self.assertIs(result.status, Status.FAILED)
        self.assertEqual(result.blockers[0].code,
                         "clustering_backend_unavailable")

    def test_an_installed_clusterer_is_used_and_recorded(self) -> None:
        runner = ClusterRunner(rows="s0\ts0\n")
        result = self.run_step(
            clustering_adapter=MMseqs2ClusterAdapter(
                runner=runner, executable_finder=installed))
        self.assertTrue(result.status.usable)
        self.assertNotIn("clustering_fallback", {f.code for f in result.qc_flags})
        recorded = result.provenance.parameters["clustering"]
        self.assertEqual(recorded["method"], "mmseqs2_easy_cluster")
        self.assertIn("easy-cluster", recorded["command"])
