"""The motif-anchored cofactor determinant: what it reads, and what it refuses.

The property these tests exist to hold is the last one in the file: a signal
with no demonstrated skill may be recorded and may not become a call. The
determinant shipped in ``fam.sdr.v1`` scores AUROC 0.58 [0.46, 0.69] against
0.74 for nearest-neighbour homology transfer on the only cofactor-labelled
dataset this repository has, so anything that promoted it to
``cofactor_preference`` would be asserting a preference the measurement does
not support -- and the cost of that error is an enzyme assayed with the wrong
nicotinamide, recorded as inactive, and dropped from the pool.
"""

from __future__ import annotations

import unittest

from eagent.harness.templates import TemplateLibrary, default_template_dir
from eagent.science.cofactor_determinant import (
    AnchoredDeterminantRule,
    DeterminantError,
    DeterminantOutcome,
    read_determinant,
    rules_from_template,
)
from eagent.tools.annotate_family import compile_motif

GLY_RICH = "G-x(3)-G-x-G"

#: A seven-residue GxxxGxG at offset 0, then ``RAA`` in the window 7..9.
WITH_BASIC = "GAGSGLG" + "RAA" + "WWWWWWWWWW"
#: The same anchor, then no basic residue in the window.
WITHOUT_BASIC = "GAGSGLG" + "AAA" + "WWWWWWWWWW"


def rule(window: tuple[int, int] = (7, 9),
         residues: str = "RK",
         cofactor: str = "NADPH") -> AnchoredDeterminantRule:
    return AnchoredDeterminantRule(
        cofactor=cofactor,
        anchor_motif="rossmann_glycine_rich_sdr",
        anchor_pattern=GLY_RICH,
        window=window,
        accepted_residues=frozenset(residues),
        evidence="fixture evidence string",
        source_template_id="fixture.template.v1",
    )


def read(sequence: str, r: AnchoredDeterminantRule | None = None):
    return read_determinant(sequence, r or rule(), compile_pattern=compile_motif)


class OutcomeTests(unittest.TestCase):

    def test_a_basic_residue_in_the_window_is_present_with_its_position(self) -> None:
        call = read(WITH_BASIC)
        self.assertIs(call.outcome, DeterminantOutcome.PRESENT)
        self.assertEqual(call.hits, ((8, "R"),))
        self.assertEqual((call.window_start, call.window_end), (8, 10))
        self.assertEqual(call.window_residues, "RAA")
        self.assertEqual(call.anchor_start, 1)

    def test_no_basic_residue_in_a_readable_window_is_absent(self) -> None:
        call = read(WITHOUT_BASIC)
        self.assertIs(call.outcome, DeterminantOutcome.ABSENT)
        self.assertEqual(call.hits, ())
        self.assertTrue(call.is_determined)

    def test_absence_is_not_evidence_for_the_other_cofactor(self) -> None:
        """The templates state the asymmetry; the call has to carry it."""
        self.assertIsNone(read(WITHOUT_BASIC).supports_cofactor)
        self.assertEqual(read(WITH_BASIC).supports_cofactor, "NADPH")

    def test_a_missing_anchor_is_its_own_outcome_not_an_absence(self) -> None:
        call = read("W" * 40)
        self.assertIs(call.outcome, DeterminantOutcome.ANCHOR_ABSENT)
        self.assertFalse(call.is_determined)
        self.assertIsNone(call.supports_cofactor)
        self.assertIn("does not match", call.note)

    def test_a_repeated_anchor_refuses_rather_than_taking_the_first(self) -> None:
        """Which occurrence binds the cofactor is not decidable from sequence."""
        call = read(WITH_BASIC + "GAGSGLGRAA")
        self.assertIs(call.outcome, DeterminantOutcome.ANCHOR_NOT_UNIQUE)
        self.assertFalse(call.is_determined)
        self.assertIn("matches 2 times", call.note)

    def test_a_window_running_past_the_end_is_truncated_not_absent(self) -> None:
        """A partial window would under-report the determinant."""
        call = read("GAGSGLGR", rule(window=(7, 20)))
        self.assertIs(call.outcome, DeterminantOutcome.WINDOW_TRUNCATED)
        self.assertEqual(call.anchor_start, 1)
        self.assertIn("outside a sequence of 8 residues", call.note)

    def test_an_empty_sequence_is_an_absent_anchor(self) -> None:
        self.assertIs(read("").outcome, DeterminantOutcome.ANCHOR_ABSENT)

    def test_a_negative_offset_reads_upstream_of_the_anchor(self) -> None:
        call = read("KK" + WITH_BASIC, rule(window=(-2, -1)))
        self.assertIs(call.outcome, DeterminantOutcome.PRESENT)
        self.assertEqual(call.hits, ((1, "K"), (2, "K")))

    def test_a_negative_offset_before_the_sequence_start_is_truncated(self) -> None:
        self.assertIs(read(WITH_BASIC, rule(window=(-2, -1))).outcome,
                      DeterminantOutcome.WINDOW_TRUNCATED)

    def test_every_outcome_round_trips_through_to_dict(self) -> None:
        for sequence in (WITH_BASIC, WITHOUT_BASIC, "W" * 40,
                         WITH_BASIC + "GAGSGLGRAA"):
            d = read(sequence).to_dict()
            self.assertIn(d["outcome"], {o.value for o in DeterminantOutcome})
            self.assertEqual(d["rule_source"], "fixture.template.v1")
            self.assertEqual(d["needs_curation"], True)
            self.assertEqual(d["determined"],
                             DeterminantOutcome(d["outcome"]).is_determined)


class RuleValidationTests(unittest.TestCase):
    """An unusable rule raises where it is written, not where it is applied."""

    def test_a_rule_without_evidence_may_not_be_built(self) -> None:
        with self.assertRaises(DeterminantError) as caught:
            AnchoredDeterminantRule(
                cofactor="NADPH", anchor_motif="m", anchor_pattern=GLY_RICH,
                window=(7, 9), accepted_residues=frozenset("RK"),
                evidence="   ", source_template_id="t")
        self.assertIn("no evidence", str(caught.exception))

    def test_a_rule_without_a_source_template_may_not_be_built(self) -> None:
        with self.assertRaises(DeterminantError):
            AnchoredDeterminantRule(
                cofactor="NADPH", anchor_motif="m", anchor_pattern=GLY_RICH,
                window=(7, 9), accepted_residues=frozenset("RK"),
                evidence="e", source_template_id="")

    def test_a_non_standard_residue_code_is_a_typo_and_raises(self) -> None:
        with self.assertRaises(DeterminantError) as caught:
            rule(residues="RZ")
        self.assertIn("one-letter codes", str(caught.exception))

    def test_an_empty_window_raises(self) -> None:
        with self.assertRaises(DeterminantError):
            rule(window=(9, 7))

    def test_an_empty_residue_set_raises(self) -> None:
        with self.assertRaises(DeterminantError):
            rule(residues="")


class TemplateRuleTests(unittest.TestCase):

    def setUp(self) -> None:
        self.library = TemplateLibrary.load(default_template_dir())
        self.sdr = self.library.get("fam.sdr.v1")

    def test_the_shipped_sdr_template_declares_one_anchored_rule(self) -> None:
        rules = rules_from_template(self.sdr)
        self.assertEqual([r.cofactor for r in rules], ["NADPH"])
        only = rules[0]
        self.assertEqual(only.anchor_motif, "rossmann_glycine_rich_sdr")
        self.assertEqual(only.window, (7, 9))
        self.assertEqual(sorted(only.accepted_residues), ["K", "R"])
        self.assertEqual(only.source_template_id, "fam.sdr.v1")
        self.assertTrue(only.needs_curation)

    def test_its_anchor_pattern_is_the_templates_own_motif(self) -> None:
        """The rule names a motif; it does not carry a second copy of it."""
        motif = next(m for m in self.sdr.conserved_motifs
                     if m["name"] == "rossmann_glycine_rich_sdr")
        self.assertEqual(rules_from_template(self.sdr)[0].anchor_pattern,
                         motif["pattern"])

    def test_the_shipped_rule_records_that_it_was_scored_and_lost(self) -> None:
        """The measurement travels with the rule, not only with the report."""
        raw = self.sdr.cofactor_recognition_rules[0]
        measured = str(raw.get("measured_on") or "")
        self.assertIn("0.58", measured)
        self.assertIn("MAY NOT BE READ AS A COFACTOR CALL", measured)

    def test_a_rule_anchored_on_an_undefined_motif_raises(self) -> None:
        class Broken:
            template_id = "broken.v1"
            conserved_motifs = [{"name": "something_else", "pattern": GLY_RICH}]
            cofactor_recognition_rules = [{
                "cofactor": "NADPH", "anchor_motif": "not_defined",
                "window": [7, 9], "accepted_residues": ["R"],
                "evidence": "e"}]

        with self.assertRaises(DeterminantError) as caught:
            rules_from_template(Broken())
        self.assertIn("does not define", str(caught.exception))

    def test_a_rule_without_a_two_offset_window_raises(self) -> None:
        class Broken:
            template_id = "broken.v1"
            conserved_motifs = [{"name": "m", "pattern": GLY_RICH}]
            cofactor_recognition_rules = [{
                "cofactor": "NADPH", "anchor_motif": "m", "window": [7],
                "accepted_residues": ["R"], "evidence": "e"}]

        with self.assertRaises(DeterminantError) as caught:
            rules_from_template(Broken())
        self.assertIn("window [first, last]", str(caught.exception))

    def test_a_template_declaring_no_rules_yields_none(self) -> None:
        for template_id in ("fam.akr.v1", "fam.mdr_adh.v1"):
            template = self.library.get(template_id)
            if template is not None:
                self.assertEqual(rules_from_template(template), [])


class NotACallTests(unittest.TestCase):
    """The honesty property: the determinant is recorded, never asserted."""

    def test_read_cofactor_determinants_marks_every_entry_as_not_a_call(self) -> None:
        from eagent.tools.annotate_family import read_cofactor_determinants

        library = TemplateLibrary.load(default_template_dir())
        entries = read_cofactor_determinants(WITH_BASIC, library.get("fam.sdr.v1"))
        self.assertEqual(len(entries), 1)
        self.assertFalse(entries[0]["is_cofactor_call"])
        self.assertEqual(entries[0]["outcome"], "present")
        self.assertEqual(entries[0]["window_offsets"], [7, 9])

    def test_a_refusal_is_reported_rather_than_dropped(self) -> None:
        from eagent.tools.annotate_family import read_cofactor_determinants

        library = TemplateLibrary.load(default_template_dir())
        entries = read_cofactor_determinants("W" * 50, library.get("fam.sdr.v1"))
        self.assertEqual([e["outcome"] for e in entries], ["anchor_absent"])

    def test_the_determinant_does_not_set_a_cofactor_preference(self) -> None:
        """Measured AUROC 0.58 with an interval containing 0.5 sets no field."""
        from eagent.schemas.candidate import FamilyAnnotation
        from eagent.tools.annotate_family import read_cofactor_determinants

        library = TemplateLibrary.load(default_template_dir())
        annotation = FamilyAnnotation(
            family_name="SDR", family_template_id="fam.sdr.v1",
            cofactor_determinants=read_cofactor_determinants(
                WITH_BASIC, library.get("fam.sdr.v1")))
        self.assertEqual(annotation.cofactor_determinants[0]["outcome"],
                         "present")
        self.assertIsNone(annotation.cofactor_preference)
        self.assertIsNone(annotation.cofactor_preference_evidence)


class BenchmarkComparatorTests(unittest.TestCase):
    """How the comparator reaches the report, and where it must not."""

    def examples(self, target: str):
        from eagent.science.enzyme_substrate import Example

        rows, clusters = [], {}
        for i in range(24):
            # Half the pool carries the determinant, and the label agrees with
            # it for exactly half of those -- so the comparator has something
            # to rank and no fixture can make it look better than it is.
            sequence = (WITH_BASIC if i % 2 else WITHOUT_BASIC) + "A" * i
            rows.append(Example(f"s{i}", sequence, target, i % 4 in (1, 2),
                                "annotation"))
            clusters[f"s{i}"] = f"c{i}"
        return rows, clusters

    def run_one(self, target: str, rules=()):
        from eagent.eval.retrospective import BenchmarkProtocol, run_benchmark

        rows, clusters = self.examples(target)
        return run_benchmark(rows, BenchmarkProtocol(n_boot=20, min_per_class=3),
                             cluster_assignment=clusters,
                             determinant_rules=rules)

    def test_without_rules_the_comparator_is_absent_not_zero(self) -> None:
        report = self.run_one("sdr:cofactor_NADP")
        self.assertEqual(report.determinant_rules, [])
        self.assertNotIn("cofactor_determinant", report.targets[0].comparators)
        self.assertNotIn("cofactor_determinant", report.macro)
        self.assertNotIn("COFACTOR DETERMINANT", report.render())

    def test_with_rules_it_is_scored_on_the_cofactor_target(self) -> None:
        report = self.run_one("sdr:cofactor_NADP", (rule(),))
        cell = report.targets[0].comparators["cofactor_determinant"]
        self.assertIsNotNone(cell["auroc"])
        self.assertIsNone(cell.get("brier"),
                          "a rule reports no probability, so no Brier score")
        self.assertIn("COFACTOR DETERMINANT", report.render())
        self.assertIn("a rule, not a fit", report.render())

    def test_it_is_not_scored_on_a_target_it_does_not_answer(self) -> None:
        """A 2'-phosphate determinant says nothing about a sterol label."""
        report = self.run_one("sdr:sterol_substructure", (rule(),))
        self.assertNotIn("cofactor_determinant",
                         report.targets[0].comparators)
        self.assertEqual(report.determinant_outcomes, {})

    def test_the_report_records_the_rule_it_ran(self) -> None:
        report = self.run_one("sdr:cofactor_NADP", (rule(),))
        self.assertEqual(report.determinant_rules[0]["window"], [7, 9])
        self.assertEqual(report.determinant_rules[0]["accepted_residues"],
                         ["K", "R"])
        self.assertIn("fixture.template.v1", report.render())

    def test_outcome_counts_are_reported_so_coverage_is_visible(self) -> None:
        report = self.run_one("sdr:cofactor_NADP", (rule(),))
        counts = report.determinant_outcomes["sdr:cofactor_NADP"]
        self.assertEqual(sum(counts.values()), 24)
        self.assertEqual(set(counts), {"present", "absent"})
        self.assertIn("read on 24/24 sequences", report.render())

    def test_an_unreadable_sequence_takes_the_neutral_score(self) -> None:
        from eagent.eval.retrospective import _DeterminantScorer

        scorer = _DeterminantScorer((rule(),))
        self.assertEqual(scorer.score(WITH_BASIC, "sdr:cofactor_NADP"), 1.0)
        self.assertEqual(scorer.score(WITHOUT_BASIC, "sdr:cofactor_NADP"), 0.0)
        self.assertEqual(scorer.score("W" * 40, "sdr:cofactor_NADP"), 0.5)
        self.assertIsNone(scorer.score(WITH_BASIC, "sdr:phenol_substructure"))

    def test_two_rules_for_one_cofactor_are_refused(self) -> None:
        from eagent.eval.retrospective import RetrospectiveError, _DeterminantScorer

        with self.assertRaises(RetrospectiveError):
            _DeterminantScorer((rule(), rule(window=(0, 6))))

    def test_the_score_does_not_depend_on_the_fold_assignment(self) -> None:
        """Nothing is fitted, so leakage cannot flatter this comparator."""
        from eagent.eval.retrospective import BenchmarkProtocol, run_benchmark

        rows, clusters = self.examples("sdr:cofactor_NADP")
        grouped = run_benchmark(
            rows, BenchmarkProtocol(n_boot=20, min_per_class=3),
            cluster_assignment=clusters, determinant_rules=(rule(),))
        ungrouped = run_benchmark(
            rows, BenchmarkProtocol(n_boot=20, min_per_class=3, grouped=False),
            cluster_assignment=clusters, determinant_rules=(rule(),))
        self.assertEqual(
            grouped.targets[0].comparators["cofactor_determinant"]["auroc"],
            ungrouped.targets[0].comparators["cofactor_determinant"]["auroc"])


if __name__ == "__main__":                                 # pragma: no cover
    unittest.main()
