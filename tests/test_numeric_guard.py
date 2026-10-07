"""Regression tests: a cited number must be findable in the file it cites.

The guard this replaces looked for a bracketed string on the same line. Five
sentences that passed it, each reproduced here before the fix:

* one citation licensing two numbers from two different files;
* a citation naming a file the run never produced;
* a citation with no row and no column;
* a value contradicting the cell it cites;
* a citation placed before the number it is meant to back.

A check that accepts all five is not stopping a fabricated measurement; it is
teaching the model which punctuation to add. The characteristic failure of a
research agent is answering with a number that reads like a measurement and
was never measured, and a syntax check cannot tell the two apart.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from eagent.errors import FabricationGuardError
from eagent.envelope import Artifact, Status, ToolResult
from eagent.harness.citation import (
    ArtifactEntry, ArtifactIndex, parse_citations, values_agree,
)
from eagent.harness.llm import NumericGuard
from eagent.provenance import RunManifest, sha256_file

DIGEST = "9f2c1a7b4e55" + "0" * 52


def cite(field: str = "plddt", method: str = "read", *,
         artifact: str = "candidate_scorecards", row: str = "cand_0a1b",
         sha: str = "9f2c1a7b4e55") -> str:
    return (f"[cite artifact={artifact} sha256={sha} row={row} "
            f"field={field} method={method}]")


class _WithTable(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.table = pathlib.Path(self._tmp.name) / "scorecards.tsv"
        self.table.write_text(
            "candidate_id\tplddt\tdistance_A\n"
            "cand_0a1b\t42.0\t3.5987\n"
            "cand_7c3d\t88.5\t3.1000\n", encoding="utf-8")
        self.index = ArtifactIndex(
            [ArtifactEntry("candidate_scorecards", self.table, DIGEST)])
        self.guard = NumericGuard(strict=True, index=self.index)


class TheFiveSentencesThatUsedToPass(_WithTable):
    def refuses(self, text: str) -> str:
        with self.assertRaises(FabricationGuardError) as ctx:
            self.guard.check(text)
        return str(ctx.exception)

    def test_one_citation_does_not_license_two_numbers(self) -> None:
        message = self.refuses(
            f"The distance is 3.6 A and the pLDDT is 42 pLDDT {cite()}.")
        self.assertIn("3.6 A", message)

    def test_an_artifact_the_run_never_produced_is_refused(self) -> None:
        message = self.refuses(
            f"Conversion reached 85 % {cite(artifact='does_not_exist')}.")
        self.assertIn("no artifact called", message)

    def test_a_citation_with_no_row_or_field_is_refused(self) -> None:
        message = self.refuses(
            "kcat is 12 s-1 [cite artifact=candidate_scorecards "
            "sha256=9f2c1a7b4e55].")
        self.assertIn("missing", message)

    def test_a_value_contradicting_the_cell_is_refused(self) -> None:
        message = self.refuses(f"The pLDDT is 95 pLDDT {cite()}.")
        self.assertIn("the text says 95", message)
        self.assertIn("42.0", message)

    def test_a_citation_before_the_number_backs_nothing(self) -> None:
        message = self.refuses(
            f"{cite()} says the distance; we estimate 3.6 A.")
        self.assertIn("3.6 A", message)

    def test_the_old_citation_form_is_refused_with_the_new_one(self) -> None:
        message = self.refuses("The pLDDT is 42 pLDDT [artifact:scorecards.tsv].")
        self.assertIn("names no version, row, field or method", message)
        self.assertIn("method=read", message)


class WhatIsAccepted(_WithTable):
    def test_a_value_read_straight_from_the_cell(self) -> None:
        self.guard.check(f"The pLDDT is 42 pLDDT {cite()}.")

    def test_two_numbers_with_two_citations(self) -> None:
        self.guard.check(
            f"pLDDT 42 pLDDT {cite()} and distance 3.6 A "
            f"{cite('distance_A', 'rounded')}.")

    def test_rounding_at_the_precision_written(self) -> None:
        self.guard.check(
            f"The distance is 3.6 A {cite('distance_A', 'rounded')}.")

    def test_more_digits_narrow_the_claim(self) -> None:
        """3.60 is the cell to two places; 3.59 is not."""
        self.guard.check(
            f"The distance is 3.60 A {cite('distance_A', 'rounded')}.")
        with self.assertRaises(FabricationGuardError):
            self.guard.check(
                f"The distance is 3.59 A {cite('distance_A', 'rounded')}.")

    def test_prose_with_no_quantities_passes(self) -> None:
        self.guard.check("The SDR hypothesis is the one to test first.")

    def test_a_second_row_is_found_by_its_own_id(self) -> None:
        self.guard.check(
            f"The pLDDT is 88.5 pLDDT {cite(row='cand_7c3d')}.")

    def test_a_row_the_table_does_not_have_is_refused(self) -> None:
        with self.assertRaises(FabricationGuardError) as ctx:
            self.guard.check(f"The pLDDT is 42 pLDDT {cite(row='ghost')}.")
        self.assertIn("no row 'ghost'", str(ctx.exception))

    def test_a_field_the_table_does_not_have_is_refused(self) -> None:
        with self.assertRaises(FabricationGuardError) as ctx:
            self.guard.check(f"The pLDDT is 42 pLDDT {cite('invented')}.")
        self.assertIn("no field 'invented'", str(ctx.exception))

    def test_the_wrong_version_of_the_file_is_refused(self) -> None:
        with self.assertRaises(FabricationGuardError) as ctx:
            self.guard.check(f"The pLDDT is 42 pLDDT {cite(sha='deadbeefcafe')}.")
        self.assertIn("not the one the number was read from", str(ctx.exception))

    def test_a_short_hash_is_refused_as_no_version_at_all(self) -> None:
        with self.assertRaises(FabricationGuardError) as ctx:
            self.guard.check(f"The pLDDT is 42 pLDDT {cite(sha='9f2c')}.")
        self.assertIn("hex characters", str(ctx.exception))


class DerivedValues(_WithTable):
    def test_a_derivation_is_accepted_but_counted_as_unchecked(self) -> None:
        text = f"The ratio is 2 % {cite('plddt', 'derived:plddt/21')}."
        self.guard.check(text)
        report = self.guard.inspect(text)
        self.assertEqual(report.declared_quantities, ["2 %"])
        self.assertEqual(report.verified_quantities, [])
        self.assertFalse(report.fully_verified)

    def test_require_verified_refuses_it(self) -> None:
        guard = NumericGuard(strict=True, index=self.index,
                             require_verified=True)
        with self.assertRaises(FabricationGuardError) as ctx:
            guard.check(f"The ratio is 2 % {cite('plddt', 'derived:plddt/21')}.")
        self.assertIn("could not be checked against the artifact",
                      str(ctx.exception))

    def test_a_read_value_satisfies_require_verified(self) -> None:
        guard = NumericGuard(strict=True, index=self.index,
                             require_verified=True)
        guard.check(f"The pLDDT is 42 pLDDT {cite()}.")

    def test_the_report_says_what_it_checked(self) -> None:
        report = self.guard.inspect(f"The pLDDT is 42 pLDDT {cite()}.")
        self.assertTrue(report.fully_verified)
        self.assertIn("1 verified", report.summary())


class WithoutAnIndex(unittest.TestCase):
    """Grammar and binding are enforced even when nothing can be opened."""

    def setUp(self) -> None:
        self.guard = NumericGuard(strict=True)

    def test_a_well_formed_citation_passes(self) -> None:
        self.guard.check(f"The pLDDT is 42 pLDDT {cite()}.")

    def test_an_unbound_number_is_still_refused(self) -> None:
        with self.assertRaises(FabricationGuardError):
            self.guard.check("The pLDDT is about 42 pLDDT.")

    def test_nothing_is_reported_as_verified(self) -> None:
        report = self.guard.inspect(f"The pLDDT is 42 pLDDT {cite()}.")
        self.assertEqual(report.verified_quantities, [])
        self.assertEqual(report.declared_quantities, ["42 pLDDT"])

    def test_a_lenient_guard_still_reports(self) -> None:
        guard = NumericGuard(strict=False)
        report = guard.inspect("The pLDDT is about 42 pLDDT.")
        guard.check("The pLDDT is about 42 pLDDT.")
        self.assertEqual(report.uncited_quantities, ["42 pLDDT"])


class CitationGrammar(unittest.TestCase):
    def test_keys_may_be_in_any_order(self) -> None:
        citations, problems = parse_citations(
            "[cite method=read field=plddt row=c1 sha256=9f2c1a7b4e55 "
            "artifact=scores]")
        self.assertEqual(problems, [])
        self.assertEqual(citations[0].field_name, "plddt")
        self.assertEqual(citations[0].artifact, "scores")

    def test_a_quoted_value_may_contain_spaces(self) -> None:
        citations, problems = parse_citations(
            '[cite artifact=scores sha256=9f2c1a7b4e55 row="well A1" '
            'field=plddt method=read]')
        self.assertEqual(problems, [])
        self.assertEqual(citations[0].row, "well A1")

    def test_every_key_is_required(self) -> None:
        for dropped in ("artifact", "sha256", "row", "field", "method"):
            pairs = {"artifact": "scores", "sha256": "9f2c1a7b4e55",
                     "row": "c1", "field": "plddt", "method": "read"}
            pairs.pop(dropped)
            body = " ".join(f"{k}={v}" for k, v in pairs.items())
            citations, problems = parse_citations(f"[cite {body}]")
            self.assertEqual(citations, [], dropped)
            self.assertIn(dropped, problems[0])

    def test_values_agree_handles_text_cells(self) -> None:
        self.assertTrue(values_agree("SDR", "SDR"))
        self.assertFalse(values_agree("SDR", "AKR"))
        self.assertFalse(values_agree("3.6", "not a number"))


class IndexFromManifest(unittest.TestCase):
    def test_the_hashes_come_from_the_run_not_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "t.tsv"
            path.write_text("candidate_id\tplddt\nc1\t42\n", encoding="utf-8")
            digest = sha256_file(path)
            manifest = RunManifest(run_id="r", task_id="t")
            manifest.record("s1", "evaluate_catalysis", ToolResult(
                status=Status.SUCCESS,
                artifacts=[Artifact(key="candidate_scorecards", path=str(path),
                                    sha256=digest)]), "2026-01-01T00:00:00Z")
            index = ArtifactIndex.from_manifest(manifest)
            self.assertEqual(index.keys, ["candidate_scorecards"])
            entry = index.resolve("candidate_scorecards")
            assert entry is not None
            self.assertEqual(entry.sha256, digest)

            guard = NumericGuard(strict=True, index=index)
            guard.check(
                f"The pLDDT is 42 pLDDT [cite artifact=candidate_scorecards "
                f"sha256={digest[:12]} row=c1 field=plddt method=read].")

    def test_an_edit_after_the_run_invalidates_the_citation(self) -> None:
        """The recorded hash is the point: the file changed, the number may not."""
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "t.tsv"
            path.write_text("candidate_id\tplddt\nc1\t42\n", encoding="utf-8")
            digest = sha256_file(path)
            path.write_text("candidate_id\tplddt\nc1\t95\n", encoding="utf-8")
            index = ArtifactIndex([ArtifactEntry("scores", path, digest)])
            guard = NumericGuard(strict=True, index=index)
            with self.assertRaises(FabricationGuardError):
                guard.check(
                    f"The pLDDT is 42 pLDDT [cite artifact=scores "
                    f"sha256={sha256_file(path)[:12]} row=c1 field=plddt "
                    f"method=read].")


if __name__ == "__main__":
    unittest.main(verbosity=2)
