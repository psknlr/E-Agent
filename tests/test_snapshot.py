"""Tests for :mod:`eagent.datalayer.snapshot`.

The fixtures write real files into a temporary directory and then really
change them, so the drift tests exercise the checksum path rather than a
mocked one. Every expected value follows from the fixture: a file written with
different bytes must verify as modified, a snapshot built with ``version="v1"``
against one built with ``version="v2"`` must diff on the version, and a
pipeline that drops four of nine records must leave four log entries.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from eagent.datalayer.snapshot import (
    CleaningPipeline,
    CleaningRule,
    DatasetSnapshot,
    EvidenceChain,
    ExclusionLog,
    SourceSpec,
    diff,
    freeze,
    verify,
)
from eagent.errors import ProvenanceError

DROP_FRAGMENTS = CleaningRule(
    name="drop_fragments", version="1.2",
    description="remove sequences annotated as fragments",
    params={"min_length": 100},
)
DROP_NO_SUBSTRATE = CleaningRule(
    name="drop_rows_without_substrate_structure", version="1.0",
    description="a row with no SMILES cannot be matched to the target substrate",
)


@dataclass
class _Row:
    """Minimal stand-in for an ingested row; the pipeline is duck-typed."""

    record_id: str
    length: int = 300
    smiles: str | None = "CC(=O)c1ccccc1"


def _rows() -> list[_Row]:
    """Nine rows: two fragments, two without a substrate structure, five clean."""
    return [
        _Row("r1"), _Row("r2"),
        _Row("r3", length=40), _Row("r4", length=12),
        _Row("r5", smiles=None), _Row("r6", smiles=None),
        _Row("r7"), _Row("r8"), _Row("r9"),
    ]


def _pipeline(source_id: str) -> CleaningPipeline:
    pipe = CleaningPipeline(source_id)
    pipe.add(DROP_FRAGMENTS,
             lambda r: f"length {r.length} below 100 aa; annotated fragment"
             if r.length < 100 else None)
    pipe.add(DROP_NO_SUBSTRATE,
             lambda r: "no substrate structure recorded" if not r.smiles else None)
    return pipe


class _Tmp(unittest.TestCase):
    """Base class giving each test its own working directory."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.wd = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, name: str, text: str) -> Path:
        p = self.wd / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p


class TestExclusionLog(_Tmp):
    def test_every_dropped_record_is_logged(self) -> None:
        """Nine in, five out, and the log accounts for all four that left."""
        rows = _rows()
        log = ExclusionLog()
        kept = _pipeline("brenda").run(rows, log)

        self.assertEqual(len(rows), 9)
        self.assertEqual(len(kept), 5)
        self.assertEqual(len(log), 4)
        self.assertEqual(len(rows) - len(log), len(kept))
        self.assertEqual(log.excluded_ids("brenda"), {"r3", "r4", "r5", "r6"})
        self.assertEqual({r.record_id for r in kept}, {"r1", "r2", "r7", "r8", "r9"})

    def test_each_entry_names_the_rule_and_the_reason(self) -> None:
        log = ExclusionLog()
        _pipeline("brenda").run(_rows(), log)
        by_id = {e.record_id: e for e in log.entries}
        self.assertEqual(by_id["r3"].rule_name, "drop_fragments")
        self.assertEqual(by_id["r3"].rule_version, "1.2")
        self.assertIn("fragment", by_id["r3"].reason)
        self.assertEqual(by_id["r5"].rule_name,
                         "drop_rows_without_substrate_structure")
        self.assertIn("no substrate structure", by_id["r5"].reason)
        self.assertTrue(all(e.source_id == "brenda" for e in log.entries))

    def test_counts_are_grouped_by_reason_and_by_rule(self) -> None:
        log = ExclusionLog()
        _pipeline("brenda").run(_rows(), log)
        self.assertEqual(sum(log.counts_by_reason().values()), 4)
        self.assertEqual(log.counts_by_rule(),
                         {"drop_fragments@1.2": 2,
                          "drop_rows_without_substrate_structure@1.0": 2})

    def test_one_rule_is_credited_per_drop(self) -> None:
        """A row failing two rules is attributed to the first, not counted twice."""
        log = ExclusionLog()
        kept = _pipeline("brenda").run([_Row("rX", length=10, smiles=None)], log)
        self.assertEqual(kept, [])
        self.assertEqual(len(log), 1)
        self.assertEqual(log.entries[0].rule_name, "drop_fragments")

    def test_a_drop_without_a_reason_is_refused(self) -> None:
        """An exclusion log full of blank reasons is the same as no log."""
        log = ExclusionLog()
        with self.assertRaises(ProvenanceError):
            log.record_exclusion("r1", DROP_FRAGMENTS, "   ", source_id="brenda")
        with self.assertRaises(ProvenanceError):
            log.record_exclusion("", DROP_FRAGMENTS, "a real reason")
        self.assertEqual(len(log), 0)

    def test_log_round_trips(self) -> None:
        log = ExclusionLog()
        _pipeline("brenda").run(_rows(), log)
        back = ExclusionLog.from_list(json.loads(json.dumps(log.to_list())))
        self.assertEqual([e.to_dict() for e in back.entries], log.to_list())

    def test_rows_without_an_id_get_a_positional_id_not_an_invented_one(self) -> None:
        log = ExclusionLog()
        pipe = CleaningPipeline("oed")
        pipe.add(DROP_FRAGMENTS, lambda r: "always dropped in this test")
        pipe.run([{"length": 10}], log)
        self.assertEqual(log.entries[0].record_id, "oed#row0")


class TestFreeze(_Tmp):
    def _spec(self, **over) -> SourceSpec:
        self.write("brenda/dump.tsv", "ec\tsequence\n1.1.1.1\tMKAV\n")
        log = ExclusionLog()
        _pipeline("brenda").run(_rows(), log)
        base = dict(
            source_id="brenda",
            paths=["brenda/dump.tsv"],
            version="2024.1",
            retrieved_at="2026-09-01T10:00:00+00:00",
            cleaning_rules=[DROP_FRAGMENTS, DROP_NO_SUBSTRATE],
            n_records_before=9,
            n_records_after=5,
            exclusion_log=log,
            license="CC BY 4.0",
        )
        base.update(over)
        return SourceSpec(**base)

    def test_freeze_pins_checksums_and_counts(self) -> None:
        snap = freeze([self._spec()], self.wd, round_label="round-1",
                      model_versions={"af3": "1.0.0"}, random_seed=7,
                      selection_policy_id="policy-v1")
        src = snap.source("brenda")
        self.assertIsNotNone(src)
        self.assertEqual(len(src.files), 1)
        self.assertEqual(len(src.files[0].sha256), 64)
        self.assertEqual(src.n_records_before, 9)
        self.assertEqual(src.n_records_after, 5)
        self.assertEqual(src.n_excluded, 4)
        self.assertEqual([r.label for r in src.cleaning_rules],
                         ["drop_fragments@1.2",
                          "drop_rows_without_substrate_structure@1.0"])
        self.assertFalse(snap.needs_curation)
        self.assertTrue(snap.snapshot_id.startswith("snap:"))

    def test_identical_data_yields_an_identical_content_hash(self) -> None:
        a = freeze([self._spec()], self.wd, random_seed=7,
                   model_versions={"af3": "1.0.0"}, selection_policy_id="p1")
        b = freeze([self._spec()], self.wd, random_seed=7,
                   model_versions={"af3": "1.0.0"}, selection_policy_id="p1")
        self.assertEqual(a.content_sha256, b.content_sha256)
        self.assertEqual(a.snapshot_id, b.snapshot_id)

    def test_missing_declared_file_is_refused(self) -> None:
        spec = self._spec(paths=["brenda/dump.tsv", "brenda/absent.tsv"])
        with self.assertRaises(ProvenanceError) as ctx:
            freeze([spec], self.wd)
        self.assertIn("absent.tsv", str(ctx.exception))

    def test_duplicate_source_ids_are_refused(self) -> None:
        with self.assertRaises(ProvenanceError):
            freeze([self._spec(), self._spec()], self.wd)

    def test_silent_filtering_is_caught_by_the_arithmetic(self) -> None:
        """9 - 4 logged exclusions is 5; claiming 3 survivors means 2 vanished."""
        with self.assertRaises(ProvenanceError) as ctx:
            freeze([self._spec(n_records_after=3)], self.wd)
        msg = str(ctx.exception)
        self.assertIn("without an exclusion log entry", msg)

    def test_non_strict_records_the_gap_instead_of_hiding_it(self) -> None:
        snap = freeze([self._spec(n_records_after=3)], self.wd, strict=False)
        src = snap.source("brenda")
        self.assertTrue(src.needs_curation)
        self.assertTrue(any("without an exclusion log entry" in n
                            for n in src.curation_notes))
        self.assertTrue(snap.needs_curation)

    def test_unknown_version_stays_null_and_asks_for_curation(self) -> None:
        """An unknown release is never replaced with 'latest' or today's date."""
        snap = freeze([self._spec(version=None, retrieved_at=None)], self.wd,
                      random_seed=1, selection_policy_id="p", model_versions={"m": "1"})
        src = snap.source("brenda")
        self.assertIsNone(src.version)
        self.assertIsNone(src.retrieved_at)
        self.assertTrue(src.needs_curation)
        self.assertTrue(any("version is null" in n for n in src.curation_notes))
        self.assertTrue(any("retrieval date is null" in n for n in src.curation_notes))

    def test_missing_seed_and_policy_are_flagged_at_snapshot_level(self) -> None:
        snap = freeze([self._spec()], self.wd)
        self.assertTrue(snap.needs_curation)
        joined = " ".join(snap.curation_notes)
        self.assertIn("random_seed is null", joined)
        self.assertIn("selection_policy_id is null", joined)
        self.assertIn("no model versions recorded", joined)

    def test_evidence_chain_is_carried_in_the_snapshot(self) -> None:
        chain = EvidenceChain(candidate_id="cand-1", record_ids=["rec-a", "rec-b"],
                              independent_measurements=1,
                              upstream_resources=["10.1021/acscatal.0c00001"],
                              corroboration="moderate")
        snap = freeze([self._spec()], self.wd, evidence_chains=[chain])
        self.assertEqual(snap.evidence_chains[0].candidate_id, "cand-1")
        self.assertEqual(snap.evidence_chains[0].independent_measurements, 1)


class TestVerify(_Tmp):
    def _frozen(self) -> DatasetSnapshot:
        self.write("oed/records.json", '{"rows": 3}')
        self.write("oed/notes.txt", "downloaded by hand\n")
        spec = SourceSpec(
            source_id="oed", paths=["oed/records.json", "oed/notes.txt"],
            version="2025-03", retrieved_at="2026-09-01T00:00:00+00:00",
            n_records_before=3, n_records_after=3, exclusion_log=ExclusionLog(),
        )
        return freeze([spec], self.wd, random_seed=1, selection_policy_id="p1",
                      model_versions={"m": "1"})

    def test_untouched_snapshot_verifies(self) -> None:
        report = verify(self._frozen())
        self.assertTrue(report.ok)
        self.assertEqual(report.drift, [])
        self.assertIn("verified", report.render())

    def test_a_modified_file_is_detected(self) -> None:
        snap = self._frozen()
        self.write("oed/records.json", '{"rows": 4}')
        report = verify(snap)
        self.assertFalse(report.ok)
        self.assertEqual(len(report.drift), 1)
        drift = report.drift[0]
        self.assertEqual(drift.path, "oed/records.json")
        self.assertIn(drift.status, ("modified", "size_changed"))
        self.assertNotEqual(drift.expected_sha256, drift.actual_sha256)
        self.assertIn("DRIFT DETECTED", report.render())

    def test_an_edit_of_the_same_length_is_still_detected(self) -> None:
        """Same byte count, different bytes: only the checksum catches this."""
        snap = self._frozen()
        self.write("oed/records.json", '{"rows": 9}')
        report = verify(snap)
        self.assertFalse(report.ok)
        self.assertEqual(report.drift[0].status, "modified")

    def test_a_deleted_file_is_detected(self) -> None:
        snap = self._frozen()
        (self.wd / "oed/notes.txt").unlink()
        report = verify(snap)
        self.assertFalse(report.ok)
        self.assertEqual([d.status for d in report.drift], ["missing"])

    def test_verification_reports_rather_than_raising(self) -> None:
        """The caller decides whether drift invalidates the round."""
        snap = self._frozen()
        self.write("oed/records.json", "tampered")
        report = verify(snap)          # must not raise
        self.assertFalse(report.ok)
        self.assertIn("snapshot_id", report.to_dict())

    def test_a_source_with_no_pinned_file_says_so(self) -> None:
        spec = SourceSpec(source_id="manual", paths=[], n_records_before=0,
                          n_records_after=0, exclusion_log=ExclusionLog())
        snap = freeze([spec], self.wd)
        report = verify(snap)
        self.assertTrue(any("cannot be verified" in m for m in report.messages))


class TestJsonRoundTrip(_Tmp):
    def _frozen(self) -> DatasetSnapshot:
        self.write("skid/data.csv", "a,b\n1,2\n")
        spec = SourceSpec(source_id="skid", paths=["skid/data.csv"],
                          version="v1", retrieved_at="2026-09-02T00:00:00+00:00",
                          n_records_before=1, n_records_after=1,
                          exclusion_log=ExclusionLog())
        return freeze([spec], self.wd, random_seed=3, selection_policy_id="p",
                      model_versions={"af3": "1.0.0"})

    def test_write_and_load(self) -> None:
        snap = self._frozen()
        path = snap.write(self.wd / "snapshot.json")
        back = DatasetSnapshot.load(path)
        self.assertEqual(back.snapshot_id, snap.snapshot_id)
        self.assertEqual(back.content_sha256, snap.content_sha256)
        self.assertEqual(back.to_dict(), snap.to_dict())
        self.assertTrue(verify(back, self.wd).ok)

    def test_a_hand_edited_snapshot_is_refused(self) -> None:
        """A snapshot edited after the fact has authority it has not earned."""
        snap = self._frozen()
        path = snap.write(self.wd / "snapshot.json")
        raw = json.loads(path.read_text())
        raw["sources"][0]["n_records_after"] = 999
        path.write_text(json.dumps(raw))
        with self.assertRaises(ProvenanceError) as ctx:
            DatasetSnapshot.load(path)
        self.assertIn("modified since it was written", str(ctx.exception))


class TestDiff(_Tmp):
    def _snap(self, *, version: str, content: str, model: str = "1.0.0",
              seed: int = 7, policy: str = "policy-v1") -> DatasetSnapshot:
        self.write("src/data.tsv", content)
        spec = SourceSpec(source_id="brenda", paths=["src/data.tsv"],
                          version=version, retrieved_at="2026-09-01T00:00:00+00:00",
                          cleaning_rules=[DROP_FRAGMENTS],
                          n_records_before=2, n_records_after=2,
                          exclusion_log=ExclusionLog())
        return freeze([spec], self.wd, model_versions={"af3": model},
                      random_seed=seed, selection_policy_id=policy)

    def test_a_version_change_is_reported_as_a_data_change(self) -> None:
        a = self._snap(version="2024.1", content="x\n")
        b = self._snap(version="2024.2", content="x\n")
        d = diff(a, b)
        kinds = [c.kind for c in d.changes]
        self.assertIn("version", kinds)
        change = next(c for c in d.changes if c.kind == "version")
        self.assertEqual((change.before, change.after), ("2024.1", "2024.2"))
        self.assertTrue(change.is_data_change)
        self.assertEqual(d.method_changes, [])
        self.assertIn("attributable to the data", d.attribution)
        self.assertIn("2024.2", d.render())

    def test_identical_snapshots_diff_to_nothing(self) -> None:
        a = self._snap(version="2024.1", content="x\n")
        b = self._snap(version="2024.1", content="x\n")
        d = diff(a, b)
        self.assertTrue(d.identical)
        self.assertIn("identical", d.attribution)

    def test_a_changed_file_is_reported_even_at_the_same_version(self) -> None:
        """A resource that edits a release in place must not slip through."""
        a = self._snap(version="2024.1", content="x\n")
        b = self._snap(version="2024.1", content="x\ny\n")
        d = diff(a, b)
        file_changes = [c for c in d.changes if c.kind == "file"]
        self.assertEqual(len(file_changes), 1)
        self.assertNotEqual(file_changes[0].before, file_changes[0].after)
        self.assertIn("attributable to the data", d.attribution)

    def test_a_model_change_alone_is_a_method_change(self) -> None:
        a = self._snap(version="2024.1", content="x\n", model="1.0.0")
        b = self._snap(version="2024.1", content="x\n", model="2.0.0")
        d = diff(a, b)
        self.assertEqual([c.kind for c in d.changes], ["model_version"])
        self.assertEqual(d.data_changes, [])
        self.assertIn("attributable to the method", d.attribution)

    def test_changing_both_makes_the_rounds_incomparable(self) -> None:
        a = self._snap(version="2024.1", content="x\n", model="1.0.0")
        b = self._snap(version="2024.2", content="x\ny\n", model="2.0.0")
        d = diff(a, b)
        self.assertTrue(d.data_changes)
        self.assertTrue(d.method_changes)
        self.assertIn("attributable to neither", d.attribution)

    def test_seed_and_policy_changes_are_reported(self) -> None:
        a = self._snap(version="v", content="x\n", seed=1, policy="p1")
        b = self._snap(version="v", content="x\n", seed=2, policy="p2")
        kinds = {c.kind for c in diff(a, b).changes}
        self.assertEqual(kinds, {"seed", "selection_policy"})

    def test_added_and_removed_sources_are_reported(self) -> None:
        a = self._snap(version="v", content="x\n")
        self.write("extra/more.tsv", "z\n")
        extra = SourceSpec(source_id="catpred", paths=["extra/more.tsv"],
                           version="v0", retrieved_at="2026-09-01T00:00:00+00:00",
                           n_records_before=1, n_records_after=1,
                           exclusion_log=ExclusionLog())
        base = SourceSpec(source_id="brenda", paths=["src/data.tsv"],
                          version="v", retrieved_at="2026-09-01T00:00:00+00:00",
                          cleaning_rules=[DROP_FRAGMENTS],
                          n_records_before=2, n_records_after=2,
                          exclusion_log=ExclusionLog())
        b = freeze([base, extra], self.wd, model_versions={"af3": "1.0.0"},
                   random_seed=7, selection_policy_id="policy-v1")
        d = diff(a, b)
        self.assertEqual([c.kind for c in d.changes], ["source_added"])
        self.assertEqual(diff(b, a).changes[0].kind, "source_removed")

    def test_a_reordered_cleaning_pipeline_is_a_change(self) -> None:
        """Order decides which rule is credited with each drop."""
        self.write("src/data.tsv", "x\n")
        def mk(rules):
            spec = SourceSpec(source_id="brenda", paths=["src/data.tsv"],
                              version="v", retrieved_at="2026-09-01T00:00:00+00:00",
                              cleaning_rules=rules, n_records_before=2,
                              n_records_after=2, exclusion_log=ExclusionLog())
            return freeze([spec], self.wd, model_versions={"af3": "1"},
                          random_seed=1, selection_policy_id="p")
        a = mk([DROP_FRAGMENTS, DROP_NO_SUBSTRATE])
        b = mk([DROP_NO_SUBSTRATE, DROP_FRAGMENTS])
        d = diff(a, b)
        self.assertEqual([c.kind for c in d.changes], ["cleaning_rules"])
        self.assertIn("rule order", d.changes[0].detail)
        # A different pipeline is a different dataset, not a better method.
        self.assertEqual(d.method_changes, [])
        self.assertIn("attributable to the data", d.attribution)

    def test_different_exclusion_reasons_are_reported(self) -> None:
        self.write("src/data.tsv", "x\n")

        def mk(reason: str | None):
            log = ExclusionLog()
            if reason:
                log.record_exclusion("r1", DROP_FRAGMENTS, reason, source_id="brenda")
            spec = SourceSpec(source_id="brenda", paths=["src/data.tsv"],
                              version="v", retrieved_at="2026-09-01T00:00:00+00:00",
                              n_records_before=3,
                              n_records_after=3 - (1 if reason else 0),
                              exclusion_log=log)
            return freeze([spec], self.wd, model_versions={"af3": "1"},
                          random_seed=1, selection_policy_id="p")

        d = diff(mk(None), mk("length 40 below 100 aa"))
        kinds = {c.kind for c in d.changes}
        self.assertIn("exclusions", kinds)
        self.assertIn("counts", kinds)


class TestEvidenceChainFromLineage(_Tmp):
    def test_chain_is_built_from_a_lineage_report_when_available(self) -> None:
        try:
            from eagent.datalayer.lineage import LineageReport
        except Exception:  # pragma: no cover - lineage is a sibling module
            self.skipTest("lineage module unavailable")

        class _Group:
            group_id = "grp:abc"
            record_ids = ("rec-a", "rec-b")

        from eagent.schemas.candidate import ConfidenceLevel

        report = LineageReport(claim="ADH-X reduces acetophenone", n_rows=4,
                               n_independent=1,
                               corroboration=ConfidenceLevel.MODERATE,
                               groups=[_Group()],
                               upstream_resources=["10.1021/acscatal.0c00001"])
        chain = EvidenceChain.from_lineage_report("cand-1", report)
        self.assertEqual(chain.candidate_id, "cand-1")
        self.assertEqual(chain.record_ids, ["rec-a", "rec-b"])
        self.assertEqual(chain.independent_measurements, 1)
        self.assertEqual(chain.corroboration, "moderate")

    def test_a_report_missing_fields_leaves_nulls_not_guesses(self) -> None:
        class _Empty:
            pass

        chain = EvidenceChain.from_lineage_report("cand-2", _Empty())
        self.assertIsNone(chain.independent_measurements)
        self.assertEqual(chain.record_ids, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
