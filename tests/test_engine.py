"""The browser engine: what a call may do, and that each agent says what the project says.

The engine is the one surface a language model reaches the project's Python code
through, so these tests hold three things:

* the *envelope*: a call never raises, a refusal carries no number, every success
  carries a digest tied to its own value, and nothing a model can send reaches
  anything but a fixed function;
* each *agent* agrees with the project function it wraps, to the last digit,
  because the engine adds parsing and shaping and must add no arithmetic;
* the engine's *reach*: it imports nothing that could start a process or open a
  socket, and it can express no approval, so a model cannot approve a gate even
  by being asked to.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import pathlib
import random
import re
import time
import unittest

import eagent.engine as engine
from eagent.engine import core
from eagent.engine.core import AgentRefusal
from eagent.engine.agents import MAX_ALIGNMENT_TEXT, MAX_SEQUENCE
from eagent.engine.pipeline import BROWSER_STATUS, pipeline_table
from eagent.harness.approval import DECISION_POINTS
from eagent.harness.registry import PROTOCOL_ORDER
from eagent.schemas import TaskSpec
from eagent.science.diversity import sequence_distance
from eagent.science.geometry import angle, dihedral, distance
from eagent.science.numbering import needleman_wunsch
from eagent.science.robustness import DEFAULT_WILSON_Z, wilson_interval

ENGINE_DIR = pathlib.Path(engine.__file__).resolve().parent
ROOT_DIR = ENGINE_DIR.parents[2]
PILOT = "KRED_PILOT_001"


def run(name, **arguments):
    return json.loads(engine.call(name, json.dumps(arguments)))


def value_of(name, **arguments):
    result = run(name, **arguments)
    assert result["ok"], result
    return result["value"]


class TheEnvelope(unittest.TestCase):
    def test_the_agents_are_exactly_the_reviewed_set(self) -> None:
        # Adding an agent is a decision about what a model may reach. It should
        # take a change to this list, in review, and not only a new file.
        self.assertEqual(sorted(engine.AGENTS), [
            "align_sequences", "geometry_measure", "pipeline_catalog", "run_pipeline",
            "sequence_distance", "task_gate_report", "variant_check", "wilson_interval"])

    def test_a_success_carries_a_digest_of_its_own_value(self) -> None:
        result = run("wilson_interval", successes="3", trials="10")
        sha = hashlib.sha256(core.canonical(result["value"]).encode("utf-8")).hexdigest()
        self.assertEqual(result["cite"], {"artifact": engine.ARTIFACT, "sha256": sha,
                                          "row": f"wilson_interval-{sha[:16]}", "method": "read"})
        self.assertEqual(result["agent"], "wilson_interval")
        self.assertIn("wilson_interval", result["wraps"])
        self.assertGreaterEqual(result["elapsed_ms"], 0)

    def test_the_same_question_has_the_same_row_and_a_different_question_does_not(self) -> None:
        first = run("wilson_interval", successes="3", trials="10")
        again = run("wilson_interval", successes="3", trials="10")
        other = run("wilson_interval", successes="4", trials="10")
        self.assertEqual(first["cite"], again["cite"], "elapsed time must not be in the digest")
        self.assertNotEqual(first["cite"]["row"], other["cite"]["row"])

    def test_a_refusal_carries_no_value_and_no_citation(self) -> None:
        result = run("wilson_interval", successes="7", trials="3")
        self.assertFalse(result["ok"])
        self.assertEqual(result["kind"], "refused")
        self.assertNotIn("value", result)
        self.assertNotIn("cite", result)
        self.assertIn("out of range", result["refusal"])

    def test_arguments_are_refused_before_anything_runs(self) -> None:
        for arguments, fragment in (
                ("[]", "JSON object"), ("not json", "JSON object"), ('"text"', "JSON object"),
                ('{"successes": 3, "trials": "10"}', "must be a string"),
                ('{"successes": "3", "trials": "10", "z": "9"}', "unsupported argument"),
                ('{"successes": "3"}', "missing required"),
                ('{"successes": "3", "trials": "   "}', "missing required"),
                (json.dumps({"successes": "3", "trials": "1" * 5000}), "limit is 4000")):
            with self.subTest(arguments=arguments[:40]):
                result = json.loads(engine.call("wilson_interval", arguments))
                self.assertEqual((result["ok"], result["kind"]), (False, "invalid"))
                self.assertIn(fragment, result["refusal"])

    def test_an_unknown_agent_is_refused_not_guessed(self) -> None:
        for name in ("approve", "run_interface", "eval", "", "wilson_interval ", None, 7):
            with self.subTest(name=name):
                result = json.loads(engine.call(name, "{}"))
                self.assertEqual((result["ok"], result["kind"]), (False, "unknown_agent"))

    def test_a_call_never_raises(self) -> None:
        for name, arguments in (("wilson_interval", None), ("wilson_interval", ""),
                                ("align_sequences", '{"sequence_a": "\\u0000", "sequence_b": "A"}'),
                                (object(), "{}"), ("pipeline_catalog", "{}")):
            with self.subTest(name=str(name)[:20]):
                json.loads(engine.call(name, arguments))

    def test_a_non_finite_number_is_never_reported(self) -> None:
        spec = core.AgentSpec(name="_nan", description="", parameters={}, wraps="test",
                              run=lambda args: {"x": float("nan")})
        self.addCleanup(core.AGENTS.pop, "_nan", None)
        core.register(spec)
        result = run("_nan")
        self.assertEqual((result["ok"], result["kind"]), (False, "refused"))
        self.assertNotIn("NaN", json.dumps(result))

    def test_an_oversized_result_is_refused_rather_than_clipped(self) -> None:
        spec = core.AgentSpec(name="_big", description="", parameters={}, wraps="test",
                              run=lambda args: {"text": "x" * (core.MAX_RESULT_BYTES + 1)})
        self.addCleanup(core.AGENTS.pop, "_big", None)
        core.register(spec)
        result = run("_big")
        self.assertEqual((result["ok"], result["kind"]), (False, "internal"))
        self.assertIn("refused rather than clipped", result["refusal"])

    def test_a_bug_is_reported_as_one_and_never_becomes_a_number(self) -> None:
        def broken(args):
            raise RuntimeError("boom")
        spec = core.AgentSpec(name="_bug", description="", parameters={}, wraps="test", run=broken)
        self.addCleanup(core.AGENTS.pop, "_bug", None)
        core.register(spec)
        result = run("_bug")
        self.assertEqual((result["ok"], result["kind"]), (False, "internal"))
        self.assertIn("RuntimeError", result["refusal"])
        self.assertNotIn("value", result)

    def test_registering_a_name_twice_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            core.register(engine.AGENTS["wilson_interval"])


class StrictArguments(unittest.TestCase):
    def test_numbers_are_plain_decimals(self) -> None:
        for bad in ("1_0", "nan", "inf", "-inf", "0x10", "1e400", "1,5", "ten", "1 2", "--1"):
            with self.subTest(bad=bad):
                result = run("align_sequences", sequence_a="MKT", sequence_b="MKT", gap_open=bad)
                self.assertFalse(result["ok"], bad)
        for good in ("10", "1e1", "10.0", ".5e1", "+10"):
            with self.subTest(good=good):
                self.assertTrue(run("align_sequences", sequence_a="MKT", sequence_b="MKT",
                                    gap_open=good)["ok"], good)

    def test_numbers_are_bounded(self) -> None:
        self.assertFalse(run("align_sequences", sequence_a="MKT", sequence_b="MKT", gap_open="-1")["ok"])
        self.assertFalse(run("align_sequences", sequence_a="MKT", sequence_b="MKT", gap_open="101")["ok"])

    def test_integers_are_digits_only(self) -> None:
        for bad in ("-1", "1.5", "1e3", "1_000", "٣", " "):
            with self.subTest(bad=bad):
                self.assertFalse(run("wilson_interval", successes=bad, trials="10")["ok"], bad)

    def test_only_ascii_digits_and_letters_are_accepted(self) -> None:
        # "٣" is Arabic-Indic three, "３" is fullwidth three, "ı" upper-cases to a plain I.
        for bad in ("٣", "３", "१२"):
            self.assertFalse(run("wilson_interval", successes=bad, trials="10")["ok"], bad)
            self.assertFalse(run("align_sequences", sequence_a="MKT", sequence_b="MKT", gap_open=bad)["ok"], bad)
        for bad in ("MKTAYıAK", "MKTAſIAK", "MKTAYﬁAK", "MKTAYÅAK", "ΜΚΤ"):
            refused = run("align_sequences", sequence_a=bad, sequence_b="MKTAYIAK")
            self.assertFalse(refused["ok"], bad)
            self.assertIn("ASCII", refused["refusal"])
        self.assertFalse(run("geometry_measure", measure="distance", points="0,0,0; ٣,0,0")["ok"])
        self.assertEqual(value_of("variant_check", sequence="MKTAY", variants="K٢R")["n_valid_substitutions"], 0)

    def test_choices_ignore_case_and_refuse_the_unknown(self) -> None:
        self.assertEqual(value_of("sequence_distance", sequence_a="MKTAYIAK", sequence_b="MKTAYIAK",
                                  method="HAMMING")["method"], "hamming")
        refused = run("sequence_distance", sequence_a="MKTAYIAK", sequence_b="MKTAYIAK", method="blast")
        self.assertFalse(refused["ok"])
        self.assertIn("kmer_jaccard, hamming", refused["refusal"])

    def test_sequences_are_cleaned_and_policed(self) -> None:
        cleaned = value_of("align_sequences", sequence_a=" mkt ay\niak ", sequence_b="MKTAYIAK")
        self.assertEqual(cleaned["identity_over_aligned_columns"], 1.0)
        for bad in ("MKT1", "MK-T", "MK*", "MK T!", ""):
            with self.subTest(bad=bad):
                self.assertFalse(run("align_sequences", sequence_a=bad, sequence_b="MKT")["ok"])
        too_long = run("align_sequences", sequence_a="A" * (MAX_SEQUENCE + 1), sequence_b="A")
        self.assertFalse(too_long["ok"])
        self.assertIn(str(MAX_SEQUENCE), too_long["refusal"])


class AlignSequences(unittest.TestCase):
    A, B = "MKTAYIAKQRQISFVKSHFSRQ", "MKTAYIKQRQLSFVKSHFSRQ"

    def test_it_reports_what_the_projects_aligner_computes(self) -> None:
        direct = needleman_wunsch(self.A, self.B)
        value = value_of("align_sequences", sequence_a=self.A, sequence_b=self.B)
        self.assertEqual(value["aligned_a"], direct.aligned_a)
        self.assertEqual(value["aligned_b"], direct.aligned_b)
        self.assertEqual(value["identity_over_aligned_columns"], direct.identity)
        self.assertEqual(value["identity_percent_over_aligned_columns"], direct.identity * 100.0)
        self.assertEqual(value["identical_columns"], direct.n_identities)
        self.assertEqual(value["gap_columns"], direct.n_gaps)
        self.assertEqual(value["aligned_pair_columns"], direct.n_aligned_columns)
        self.assertEqual(value["coverage_a"], direct.coverage_a())
        self.assertEqual(value["coverage_b"], direct.coverage_b())

    def test_the_scoring_parameters_reach_the_aligner(self) -> None:
        default = value_of("align_sequences", sequence_a=self.A, sequence_b=self.B)
        harsh = value_of("align_sequences", sequence_a=self.A, sequence_b=self.B,
                         gap_open="50", gap_extend="5", match="3", mismatch="-2")
        self.assertEqual((harsh["match"], harsh["mismatch"], harsh["gap_open"], harsh["gap_extend"]),
                         (3.0, -2.0, 50.0, 5.0))
        self.assertNotEqual(default["score"], harsh["score"])

    def test_identity_is_reported_next_to_the_coverage_it_must_be_read_with(self) -> None:
        value = value_of("align_sequences", sequence_a="MKTAYIAKQRQISFVKSHFSRQ", sequence_b="SHFSRQ")
        self.assertLess(value["coverage_a"], 0.5)
        self.assertEqual(value["coverage_b"], 1.0)
        self.assertIn("coverage", value["note"])

    def test_a_long_alignment_keeps_its_numbers_and_drops_only_its_text(self) -> None:
        a = ("ACDEFGHIKLMNPQRSTVWY" * 30)[:500]
        value = value_of("align_sequences", sequence_a=a, sequence_b=a)
        self.assertGreater(value["alignment_columns"], MAX_ALIGNMENT_TEXT)
        self.assertNotIn("aligned_a", value)
        self.assertIn("omitted", value["alignment_text"])
        self.assertEqual(value["identity_over_aligned_columns"], 1.0)

    def test_the_largest_accepted_input_still_fits_the_result_bound(self) -> None:
        a = ("ACDEFGHIKLMNPQRSTVWY" * 61)[:MAX_SEQUENCE]
        b = a[:600] + ("WYVTSRQPNMLKIHGFEDCA" * 31)[:MAX_SEQUENCE - 600]
        result = run("align_sequences", sequence_a=a, sequence_b=b)
        self.assertTrue(result["ok"], result)
        self.assertLessEqual(len(core.canonical(result["value"]).encode("utf-8")), core.MAX_RESULT_BYTES)


class SequenceDistance(unittest.TestCase):
    def test_it_reports_what_the_projects_function_computes(self) -> None:
        a, b = "MKTAYIAKQRQISFVKSHFSRQ", "MKTAYIKQRQLSFVKSHFSRQ"
        for method in ("kmer_jaccard", "hamming"):
            if method == "hamming":
                b2 = b + "A"
            else:
                b2 = b
            value = value_of("sequence_distance", sequence_a=a, sequence_b=b2, method=method)
            self.assertEqual(value["distance"], sequence_distance(a, b2, method=method))
            self.assertEqual(value["similarity"], 1.0 - value["distance"])
        self.assertEqual(value_of("sequence_distance", sequence_a=a, sequence_b=b, k="4")["k"], 4)

    def test_the_projects_own_refusals_pass_through(self) -> None:
        unequal = run("sequence_distance", sequence_a="MKTAY", sequence_b="MKTA", method="hamming")
        self.assertEqual(unequal["kind"], "refused")
        self.assertIn("equal-length", unequal["refusal"])
        short = run("sequence_distance", sequence_a="MK", sequence_b="MKTAY")
        self.assertEqual(short["kind"], "refused")
        self.assertIn("shorter than k", short["refusal"])
        self.assertFalse(run("sequence_distance", sequence_a="MKTAY", sequence_b="MKTAY", k="9")["ok"])


class WilsonInterval(unittest.TestCase):
    def test_it_reports_what_the_projects_function_computes(self) -> None:
        for successes, trials in ((0, 1), (5, 5), (3, 10), (17, 40), (0, 12), (999, 1000)):
            with self.subTest(successes=successes, trials=trials):
                value = value_of("wilson_interval", successes=str(successes), trials=str(trials))
                lower, upper = wilson_interval(successes, trials, DEFAULT_WILSON_Z)
                self.assertEqual((value["lower"], value["upper"]), (lower, upper))
                self.assertEqual(value["proportion"], successes / trials)
                self.assertLessEqual(value["lower"], value["proportion"])
                self.assertGreaterEqual(value["upper"], value["proportion"])

    def test_every_fraction_also_comes_as_the_percentage_it_is_quoted_as(self) -> None:
        # "A 95% interval" must be citable against a field that says 95: a model that
        # writes a percentage in prose is held to a cited field, and a fraction cannot
        # back it.
        for confidence in ("0.90", "0.95", "0.99"):
            value = value_of("wilson_interval", successes="7", trials="20", confidence=confidence)
            self.assertEqual(value["confidence_percent"], float(confidence) * 100.0)
            for name in ("proportion", "lower", "upper"):
                self.assertEqual(value[f"{name}_percent"], value[name] * 100.0, name)
        align = value_of("align_sequences", sequence_a="MKTAYIAKQRQISFVKSHFSRQ", sequence_b="MKTAYLAKQRQ")
        self.assertEqual(align["coverage_a_percent"], align["coverage_a"] * 100.0)
        self.assertEqual(align["coverage_b_percent"], align["coverage_b"] * 100.0)
        distance = value_of("sequence_distance", sequence_a="ACDEFGHIK", sequence_b="ACDEFGHIR", method="hamming")
        self.assertEqual(distance["distance_percent"], distance["distance"] * 100.0)
        self.assertAlmostEqual(distance["similarity_percent"] + distance["distance_percent"], 100.0)

    def test_five_for_five_is_not_a_certainty(self) -> None:
        value = value_of("wilson_interval", successes="5", trials="5")
        self.assertAlmostEqual(value["lower"], 0.5655, places=3)
        self.assertEqual(value["upper"], 1.0)

    def test_confidence_widens_the_interval(self) -> None:
        widths = []
        for confidence in ("0.90", "0.95", "0.99"):
            value = value_of("wilson_interval", successes="7", trials="20", confidence=confidence)
            widths.append(value["upper"] - value["lower"])
        self.assertLess(widths[0], widths[1])
        self.assertLess(widths[1], widths[2])
        self.assertFalse(run("wilson_interval", successes="7", trials="20", confidence="0.5")["ok"])

    def test_no_trials_is_a_refusal_not_a_proportion_of_zero(self) -> None:
        zero = run("wilson_interval", successes="0", trials="0")
        self.assertEqual(zero["kind"], "refused")
        self.assertIn("at least one trial", zero["refusal"])


class GeometryMeasure(unittest.TestCase):
    def test_it_reports_what_the_projects_functions_compute(self) -> None:
        p = [(0.0, 0.0, 0.0), (1.5, 0.2, -0.3), (2.0, 1.4, 0.1), (3.1, 1.9, 1.2)]
        text = "; ".join(", ".join(repr(c) for c in point) for point in p)
        self.assertEqual(value_of("geometry_measure", measure="distance", points="; ".join(
            ", ".join(repr(c) for c in point) for point in p[:2]))["distance_angstrom"], distance(p[0], p[1]))
        self.assertEqual(value_of("geometry_measure", measure="angle", points="; ".join(
            ", ".join(repr(c) for c in point) for point in p[:3]))["angle_degrees"], angle(*p[:3]))
        self.assertEqual(value_of("geometry_measure", measure="dihedral", points=text)["dihedral_degrees"],
                         dihedral(*p))

    def test_known_answers(self) -> None:
        self.assertEqual(value_of("geometry_measure", measure="distance", points="0,0,0; 3,4,0")["distance_angstrom"], 5.0)
        self.assertAlmostEqual(value_of("geometry_measure", measure="angle", points="1,0,0; 0,0,0; 0,1,0")["angle_degrees"], 90.0)
        centre = value_of("geometry_measure", measure="centroid", points="0,0,0; 2,0,0; 0,4,0")
        self.assertEqual((centre["centroid_x_angstrom"], centre["centroid_y_angstrom"], centre["centroid_z_angstrom"]), (2 / 3, 4 / 3, 0.0))

    def test_the_sign_of_a_torsion_is_kept(self) -> None:
        up = value_of("geometry_measure", measure="dihedral", points="0,0,0; 1,0,0; 1,1,0; 1,1,1")["dihedral_degrees"]
        down = value_of("geometry_measure", measure="dihedral", points="0,0,0; 1,0,0; 1,1,0; 1,1,-1")["dihedral_degrees"]
        self.assertAlmostEqual(up, -down)
        self.assertNotEqual(up, 0.0)

    def test_the_wrong_number_of_points_and_unreadable_points_are_refused(self) -> None:
        for measure, points in (("distance", "0,0,0"), ("angle", "0,0,0; 1,1,1"), ("dihedral", "0,0,0; 1,0,0; 1,1,0"),
                                ("centroid", ""), ("distance", "0,0,0; 1,2"), ("distance", "0,0,0; a,b,c"),
                                ("distance", "0,0,0; 1e7,0,0"), ("distance", "0,0,0; nan,0,0"), ("volume", "0,0,0; 1,1,1")):
            with self.subTest(measure=measure, points=points):
                self.assertFalse(run("geometry_measure", measure=measure, points=points)["ok"])

    def test_degenerate_geometry_is_a_refusal_with_the_projects_reason(self) -> None:
        coincident = run("geometry_measure", measure="angle", points="1,1,1; 1,1,1; 2,2,2")
        self.assertEqual(coincident["kind"], "refused")
        self.assertIn("coincident", coincident["refusal"])


class VariantCheck(unittest.TestCase):
    SEQUENCE = "MKTAYIAKQR"

    def test_a_substitution_is_checked_against_the_residue_at_that_position(self) -> None:
        value = value_of("variant_check", sequence=self.SEQUENCE, variants="K2R, A7G, T4S")
        by_label = {item["label"]: item for item in value["variants"]}
        self.assertTrue(by_label["K2R"]["matches_wild_type"])
        self.assertTrue(by_label["A7G"]["matches_wild_type"])
        self.assertFalse(by_label["T4S"]["matches_wild_type"])
        self.assertEqual(by_label["T4S"]["sequence_residue"], "A")

    def test_the_counts_partition_the_valid_substitutions(self) -> None:
        value = value_of("variant_check", sequence=self.SEQUENCE,
                         variants="K2R, A7G, X5Y, L45V, T3T, Y5F, T4S, bad")
        self.assertEqual(value["n_variants"], 8)
        self.assertEqual(value["n_valid_substitutions"] + value["n_invalid"], value["n_variants"])
        self.assertEqual(value["n_valid_substitutions"],
                         value["n_wild_type_matches"] + value["n_wild_type_mismatches"] + value["n_out_of_range"])
        self.assertEqual(value["n_out_of_range"], 1)
        self.assertEqual(value["n_invalid"], 3)

    def test_invalid_tokens_say_why_and_do_not_sink_the_rest(self) -> None:
        value = value_of("variant_check", sequence=self.SEQUENCE, variants="X5Y T3T 12 K2R")
        errors = {item["label"]: item["error"] for item in value["variants"] if not item["valid"]}
        self.assertIn("standard residue", errors["X5Y"])
        self.assertIn("not a substitution", errors["T3T"])
        self.assertIn("<residue><position><residue>", errors["12"])
        self.assertEqual(value["n_wild_type_matches"], 1)

    def test_numbering_can_start_anywhere(self) -> None:
        shifted = value_of("variant_check", sequence=self.SEQUENCE, variants="K11R", first_position="10")
        self.assertTrue(shifted["variants"][0]["matches_wild_type"])
        self.assertEqual((shifted["first_position"], shifted["last_position"]), (10, 19))

    def test_separators_and_case(self) -> None:
        for variants in ("K2R,A7G", "k2r a7g", "K2R/A7G", "K2R;A7G", "K2R\nA7G"):
            with self.subTest(variants=variants):
                self.assertEqual(value_of("variant_check", sequence=self.SEQUENCE,
                                          variants=variants)["n_wild_type_matches"], 2)

    def test_bounds(self) -> None:
        self.assertFalse(run("variant_check", sequence=self.SEQUENCE, variants="")["ok"])
        self.assertFalse(run("variant_check", sequence=self.SEQUENCE,
                             variants=", ".join(["K2R"] * 41))["ok"])
        self.assertTrue(run("variant_check", sequence=self.SEQUENCE, variants=", ".join(["K2R"] * 40))["ok"])
        widest = run("variant_check", sequence="M" * MAX_SEQUENCE, variants=", ".join(f"M{n}A" for n in range(1, 41)))
        self.assertTrue(widest["ok"])
        self.assertLessEqual(len(core.canonical(widest["value"]).encode("utf-8")), core.MAX_RESULT_BYTES)


class ThePipelineTable(unittest.TestCase):
    def test_it_covers_exactly_the_protocols_ten_interfaces_in_order(self) -> None:
        self.assertEqual(set(BROWSER_STATUS), set(PROTOCOL_ORDER))
        self.assertEqual([row["interface"] for row in pipeline_table()], list(PROTOCOL_ORDER))
        self.assertEqual([row["order"] for row in pipeline_table()], list(range(1, 11)))

    def test_the_statuses_are_the_roadmaps(self) -> None:
        by = {row["interface"]: row["runs_in_browser"] for row in pipeline_table()}
        self.assertEqual({n for n, s in by.items() if s == "needs_program"},
                         {"mine_sequences", "prepare_structures", "model_complexes"})
        self.assertEqual({n for n, s in by.items() if s == "partial"},
                         {"retrieve_evidence", "propose_mutations"})
        self.assertEqual({n for n, s in by.items() if s == "runs"},
                         {"normalize_reaction", "annotate_family", "evaluate_catalysis",
                          "select_batch", "ingest_results"})

    def test_a_row_that_needs_something_says_what(self) -> None:
        for row in pipeline_table():
            self.assertEqual(bool(row["needs"]), row["runs_in_browser"] != "runs", row["interface"])
            self.assertTrue(row["note"], row["interface"])

    def test_the_report_agent_counts_agree_with_the_table(self) -> None:
        value = value_of("pipeline_catalog")
        self.assertEqual(value["n_interfaces"], 10)
        self.assertEqual(value["n_run_in_browser"] + value["n_partial"] + value["n_need_a_program"], 10)
        self.assertEqual((value["n_run_in_browser"], value["n_partial"], value["n_need_a_program"]), (5, 2, 3))
        self.assertIn("read-only", value["statement"])


class TaskGateReport(unittest.TestCase):
    def test_the_shipped_pilot_reports_what_the_validator_reports(self) -> None:
        import yaml
        task = TaskSpec(**yaml.safe_load((ENGINE_DIR.parents[2] / "configs" / "tasks" / f"{PILOT}.yaml").read_text()))
        value = value_of("task_gate_report", task=PILOT)
        self.assertEqual(value["task_id"], PILOT)
        for gate in value["gates"]:
            self.assertEqual(gate["unresolved_fields"], task.unresolved_for(gate["gate"]))
            self.assertEqual(gate["n_unresolved"], len(gate["unresolved_fields"]))
        self.assertEqual(value["n_unresolved_fields_total"], sum(g["n_unresolved"] for g in value["gates"]))
        self.assertEqual([g["n_unresolved"] for g in value["gates"]], [4, 5, 1])

    def test_inline_text_gives_the_same_answer_as_the_same_task_by_id(self) -> None:
        inline = value_of("task_gate_report", task_yaml="task_id: T1\n")
        self.assertEqual(inline["task_id"], "T1")
        self.assertEqual(inline["n_unresolved_fields_total"], 10)
        self.assertEqual(inline["n_gates_with_unresolved_fields"], 3)

    def test_a_blocked_step_names_the_fields_blocking_it(self) -> None:
        steps = {s["interface"]: s for s in value_of("task_gate_report", task=PILOT)["steps"]}
        self.assertIn("reaction.substrate.isomeric_smiles", steps["evaluate_catalysis"]["unresolved_required_fields"])
        self.assertEqual(steps["mine_sequences"]["runs_in_browser"], "needs_program")

    def test_it_says_it_cannot_decide_anything(self) -> None:
        statement = value_of("task_gate_report", task=PILOT)["statement"]
        self.assertIn("resolves no field", statement)
        self.assertIn("only a named person can approve", statement)

    def test_what_it_refuses(self) -> None:
        for arguments, fragment in (
                ({}, "exactly one of"),
                ({"task": PILOT, "task_yaml": "task_id: T1"}, "exactly one of"),
                ({"task": "../../etc/passwd"}, "no shipped task"),
                ({"task": "/etc/passwd"}, "no shipped task"),
                ({"task": "KRED_PILOT_001.yaml"}, "no shipped task"),
                ({"task_yaml": "a: &x [1]\nb: *x"}, "anchors and aliases"),
                ({"task_yaml": "a: [unclosed"}, "not valid YAML"),
                ({"task_yaml": "- a\n- b"}, "mapping"),
                ({"task_yaml": "task_id: 5\nnope: 1"}, "does not validate"),
                ({"task_yaml": "x" * 6001}, "limit is 6000")):
            with self.subTest(arguments=str(arguments)[:50]):
                result = run("task_gate_report", **arguments)
                self.assertFalse(result["ok"], arguments)
                self.assertIn(fragment, result["refusal"])


class RunPipeline(unittest.TestCase):
    """The agent that runs the project's own controller. What it reports is what the
    controller did; what it must never do is decide."""

    @staticmethod
    def specified(**approval) -> str:
        import yaml
        doc = yaml.safe_load((ENGINE_DIR.parents[2] / "configs" / "tasks" / f"{PILOT}.yaml").read_text())
        doc["reaction"]["substrate"] = {"name": "acetophenone", "isomeric_smiles": "CC(=O)c1ccccc1"}
        doc["reaction"]["product"] = {"name": "(S)-1-phenylethanol", "isomeric_smiles": "C[C@H](O)c1ccccc1",
                                      "creates_new_stereocenter": True}
        doc["reaction"]["atom_mapped_reaction_smiles"] = (
            "[CH3:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1>>"
            "[CH3:1][C@H:2]([OH:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1")
        doc["approval"].update(approval)
        return yaml.safe_dump(doc, sort_keys=False)

    @staticmethod
    def controller_for(task: TaskSpec):
        """The same run, made directly, for the agent's report to be held against."""
        import tempfile
        from eagent.context import ExecutionPolicy, RunContext
        from eagent.harness.approval import ApprovalQueue
        from eagent.harness.controller import ResearchController
        from eagent.harness.registry import build_interface_registry
        from eagent.harness.templates import TemplateLibrary
        from eagent.provenance import RunManifest
        with tempfile.TemporaryDirectory() as scratch:
            run_dir = pathlib.Path(scratch)
            manifest = RunManifest(run_id="reference", task_id=task.task_id, global_seed=0)
            ctx = RunContext(task=task, workdir=run_dir, manifest=manifest,
                             policy=ExecutionPolicy(allow_network=False, dry_run=False, max_retries=1),
                             templates=TemplateLibrary.load(None), logger=lambda message: None)
            queue = ApprovalQueue.load(run_dir / "approvals.json", manifest)
            controller = ResearchController(ctx, build_interface_registry(), queue, hooks=None, arguments={})
            controller.run()
            return controller, sorted({request.gate for request in queue.pending()})

    def test_the_pilot_runs_the_reaction_step_for_real_and_stops_at_the_first_gate(self) -> None:
        import yaml
        task = TaskSpec(**yaml.safe_load((ENGINE_DIR.parents[2] / "configs" / "tasks" / f"{PILOT}.yaml").read_text()))
        controller, gates = self.controller_for(task)
        value = value_of("run_pipeline", task=PILOT)
        self.assertEqual(value["outcome"], "awaiting_human")
        self.assertEqual(value["outcome"], controller.outcome.value)
        self.assertEqual(value["stage"], controller.stage.value)
        self.assertEqual(value["reason"], controller.stop_reason)
        self.assertEqual([step["interface"] for step in value["steps"]], ["normalize_reaction"])
        step = value["steps"][0]
        result = controller.results["normalize_reaction"]
        self.assertEqual((step["status"], step["failure"]), ("partial", "needs_human"))
        self.assertEqual([item["code"] for item in step["blockers"]],
                         [flag.code for flag in result.qc_flags if flag.severity.value == "blocker"])
        self.assertEqual(step["n_blockers"], len(step["blockers"]))
        self.assertEqual(value["n_blockers"], step["n_blockers"])
        self.assertEqual([item["code"] for item in step["uncertainties"]], [u.code for u in result.uncertainty])
        self.assertEqual([item["gate"] for item in value["waiting_on"]], gates)
        self.assertEqual(value["waiting_on"][0]["question"], DECISION_POINTS[gates[0]].question)
        self.assertEqual((value["n_steps_run"], value["n_steps_partial"], value["n_steps_failed"]), (1, 1, 0))
        self.assertEqual(value["n_gates_waiting"], 1)
        self.assertEqual(value["network"], "disabled")
        self.assertFalse(value["trimmed"], "the pilot's report fits whole")
        self.assertNotIn("mine_sequences", json.dumps(value["steps"]), "nothing behind the gate ran")

    def test_a_complete_reaction_normalises_and_then_waits_for_a_person(self) -> None:
        value = value_of("run_pipeline", task_yaml=self.specified())
        step = value["steps"][0]
        self.assertEqual((step["interface"], step["status"], step["failure"]), ("normalize_reaction", "success", "none"))
        self.assertEqual(value["n_blockers"], 0)
        self.assertGreaterEqual(value["n_warnings"], 1)
        self.assertEqual(value["outcome"], "awaiting_human")
        self.assertEqual([item["gate"] for item in value["waiting_on"]], ["reaction_spec_confirmed"])
        self.assertIn(("a person", "request_approval"), [(a["who"], a["action"]) for a in step["next_actions"]])

    def test_a_flag_in_the_file_is_not_an_approval(self) -> None:
        value = value_of("run_pipeline", task_yaml=self.specified(reaction_spec_confirmed=True))
        self.assertEqual(value["outcome"], "awaiting_human")
        self.assertEqual(value["n_steps_run"], 1)
        self.assertEqual([item["gate"] for item in value["waiting_on"]], ["reaction_spec_confirmed"])

    def test_nor_are_flags_for_every_decision_point(self) -> None:
        # An interface's own `require_approval` honours a flag in the file, so what
        # keeps a flag from doing anything here is the controller stopping at its
        # first decision point, before any later interface is reached. Setting all
        # three flags tests exactly that dependency on the order of the stages.
        value = value_of("run_pipeline", task_yaml=self.specified(
            reaction_spec_confirmed=True, synthesis_authorized=True, functional_criteria_confirmed=True))
        self.assertEqual(value["outcome"], "awaiting_human")
        self.assertEqual(value["n_steps_run"], 1)
        self.assertEqual([step["interface"] for step in value["steps"]], ["normalize_reaction"])
        self.assertEqual([item["gate"] for item in value["waiting_on"]], ["reaction_spec_confirmed"])

    def test_the_same_call_gives_the_same_answer_and_the_same_hash(self) -> None:
        first, second = run("run_pipeline", task=PILOT), run("run_pipeline", task=PILOT)
        self.assertEqual(first["value"], second["value"])
        self.assertEqual(first["cite"]["sha256"], second["cite"]["sha256"])
        text = json.dumps(first["value"])
        for volatile in ("request_id", "requested_at", "decided_at", "run_id"):
            self.assertNotIn(volatile, text)
        self.assertNotRegex(text, r"\d{4}-\d{2}-\d{2}T\d{2}:", "no timestamps: the hash must not depend on when it ran")

    def test_it_is_refused_what_the_gate_report_is_refused(self) -> None:
        for arguments, fragment in (
                ({}, "exactly one of"),
                ({"task": PILOT, "task_yaml": "task_id: T1"}, "exactly one of"),
                ({"task": "../../etc/passwd"}, "no shipped task"),
                ({"task_yaml": "a: &x [1]\nb: *x"}, "anchors and aliases"),
                ({"task_yaml": "a: [unclosed"}, "not valid YAML"),
                ({"task_yaml": "task_id: 5\nnope: 1"}, "does not validate"),
                ({"task_yaml": "x" * 6001}, "limit is 6000")):
            with self.subTest(arguments=str(arguments)[:50]):
                result = run("run_pipeline", **arguments)
                self.assertFalse(result["ok"], arguments)
                self.assertIn(fragment, result["refusal"])
                self.assertNotIn("cite", result)

    def test_nothing_is_left_behind_whether_the_run_finishes_or_fails(self) -> None:
        import tempfile
        from unittest import mock
        from eagent.harness.controller import ResearchController

        def leftovers():
            return sorted(pathlib.Path(tempfile.gettempdir()).glob("eagent-engine-*"))

        before = leftovers()
        self.assertTrue(run("run_pipeline", task=PILOT)["ok"])
        self.assertEqual(leftovers(), before)
        with mock.patch.object(ResearchController, "run", side_effect=RuntimeError("boom")):
            failed = run("run_pipeline", task=PILOT)
        self.assertEqual((failed["ok"], failed["kind"]), (False, "internal"))
        self.assertIn("RuntimeError: boom", failed["refusal"])
        self.assertEqual(leftovers(), before)

    def test_a_report_too_large_for_a_result_is_trimmed_to_fit_and_says_so(self) -> None:
        from types import SimpleNamespace as NS
        from eagent.engine import pipeline

        def flag(index, severity):
            return NS(code=f"code_{index}", severity=NS(value=severity), subject=f"subject.{index}",
                      message="a long finding " * 60)
        severity = __import__("eagent.envelope", fromlist=["Severity"]).Severity
        flags = [NS(code=f"code_{i}", severity=severity.BLOCKER, subject=f"s.{i}", message="finding " * 80) for i in range(40)]
        result = NS(qc_flags=flags, uncertainty=[NS(code=f"u{i}", question="why " * 80, resolvable_by="a person " * 30) for i in range(40)],
                    next_actions=[NS(action=f"a{i}", rationale="because " * 60, requires_human=True) for i in range(40)],
                    artifacts=[NS(key=f"k{i}", kind="file") for i in range(40)])
        attempts = [NS(stage=NS(value=f"stage_{i}"), interface=f"interface_{i}", status=NS(value="partial"), attempt=1,
                       failure=NS(value="needs_human"), message="message " * 80, skipped=False) for i in range(9)]
        controller = NS(attempts=attempts, results={a.interface: result for a in attempts}, stage=NS(value="awaiting_human"),
                        outcome=NS(value="awaiting_human"), stop_reason="reason " * 200, path=[NS(value=f"p{i}") for i in range(30)],
                        escalations=[])
        queue = NS(pending=lambda: [NS(gate="reaction_spec_confirmed", detail="detail " * 100)])
        task = NS(task_id="T", task_mode=NS(value="enzyme_mining"))
        report = pipeline._fit_report(task, controller, queue)
        self.assertLessEqual(len(core.canonical(report).encode("utf-8")), core.MAX_RESULT_BYTES)
        self.assertTrue(report["trimmed"])
        self.assertEqual(report["n_steps_run"], 9, "counts are taken before anything is trimmed")
        self.assertEqual(report["n_blockers"], 9 * 40)
        self.assertEqual(report["n_uncertainties"], 9 * 40)
        self.assertLess(len(report["steps"]), 9)
        self.assertLessEqual(len(report["path"]), 12)


class TheResultCarriesNothingAboutTheMachine(unittest.TestCase):
    """The controller records where it ran; the engine's reply must not repeat it."""

    def test_run_pipeline_does_not_report_the_host_the_platform_or_a_path(self) -> None:
        import platform
        import socket
        import tempfile
        text = engine.call("run_pipeline", json.dumps({"task": PILOT}))
        for needle in (socket.gethostname(), platform.platform(), platform.node(), tempfile.gettempdir(),
                       str(pathlib.Path.home()), str(ROOT_DIR), "rev-parse", "hostname"):
            if len(needle) > 3:
                self.assertNotIn(needle, text, needle)


class AnArgumentCannotBurnTheWorker(unittest.TestCase):
    """What a model sends is bounded in length, and must be bounded in time too.

    The argument limit stops a long argument, not a slow one: an ambiguous
    regular expression or a quadratic scanner turns a few thousand characters
    into many seconds, and a call past the page's 60 s clock kills the worker and
    turns the engine off. Each case below was measured before it was fixed.
    """

    def timed(self, name, **arguments):
        started = time.perf_counter()
        result = run(name, **arguments)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 1.5, f"{name} took {elapsed:.1f} s on {str(arguments)[:60]!r}")
        return result

    def test_runs_of_spaces_between_coordinates_are_linear(self) -> None:
        for points in ("1" + " " * 700 + "2" + " " * 700 + "x", "1" + " " * 1900 + "2" + " " * 1900,
                       " " * 3990 + "1", ",".join(["1"] * 1900), "1 " * 1900 + "x", "1" + " ," * 1000 + "2"):
            result = self.timed("geometry_measure", measure="distance", points=points)
            self.assertFalse(result["ok"])
            self.assertEqual(result["kind"], "refused", points[:20])

    def test_the_point_grammar_accepts_and_refuses_what_the_old_expression_did(self) -> None:
        from eagent.engine.agents import _COORDINATE, _points
        coordinate = _COORDINATE.pattern
        old = re.compile(rf"\s*({coordinate})\s*[,\s]\s*({coordinate})\s*[,\s]\s*({coordinate})\s*", re.ASCII)

        def new(chunk):
            try:
                return [list(point) for point in _points(chunk)]
            except AgentRefusal as exc:
                if "could not read" in str(exc):
                    return None
                return "out of range"

        def agree(chunk):
            found = old.fullmatch(chunk)
            got = new(chunk)
            if not chunk.strip(" \t\n\r\f\v") or ";" in chunk:
                return
            self.assertEqual(found is not None, got is not None, repr(chunk))
            if found and got != "out of range":
                self.assertEqual(got, [[float(part) for part in found.groups()]], repr(chunk))

        alphabet = ["1", "-", ".", "e", ",", " ", "\t"]
        for length in range(1, 7):
            for combination in itertools.product(alphabet, repeat=length):
                agree("".join(combination))
        rng = random.Random(20261011)
        for _ in range(20000):
            agree("".join(rng.choice(alphabet + ["12", "0.5", " , ", "1e3"]) for _ in range(rng.randrange(1, 12))))
        for chunk in ("1,2,3", " 1 2 3 ", "1 , 2 , 3", "1,\t2,\n3".replace("\n", " "), "1.5e2 -3 +4", "1,,2,3", ",1,2,3", "1,2,3,", "1 2", "1 2 3 4", "1.5.5 2 3", "٣ 2 1"):
            agree(chunk)

    def test_a_task_file_nested_thousands_deep_is_refused_at_once(self) -> None:
        for text in ("[" * 6000, "{" * 6000, "a: " + "[" * 5000, "[" * 3000 + "]" * 3000, "[{" * 1500 + "]}" * 1500,
                     "[" * 65, "k: " + "[" * 65 + "]" * 65):
            for agent in ("task_gate_report", "run_pipeline"):
                result = self.timed(agent, task_yaml=text)
                self.assertFalse(result["ok"], agent)
                self.assertEqual(result["kind"], "refused", f"{agent}: {text[:12]}")
                self.assertIn("block style", result["refusal"])

    def test_many_brackets_are_fine_when_they_do_not_nest(self) -> None:
        # What costs time is depth, not count: a SMILES string closes every bracket
        # at once, and a flow list is as cheap at 2,000 items as at 20.
        smiles = "[CH3:1][C:2](=[O:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1>>[CH3:1][C@H:2]([OH:3])[c:4]1[cH:5][cH:6][cH:7][cH:8][cH:9]1"
        for text in ("task_id: SMILES_001\ntask_mode: enzyme_mining\nnote: '" + smiles * 40 + "'\n",
                     "task_id: LIST_001\ntask_mode: enzyme_mining\nnotes: [" + "a, " * 1500 + "]\n",
                     "task_id: DEEP_001\ntask_mode: enzyme_mining\nnotes: " + "[" * 64 + "]" * 64 + "\n"):
            for agent in ("task_gate_report",):
                result = self.timed(agent, task_yaml=text)
                self.assertNotIn("block style", result.get("refusal", ""), text[:40])

    def test_a_task_nested_without_brackets_is_refused_not_an_internal_error(self) -> None:
        for text in ("- " * 2900, "? " * 2900, "- - - - " * 700):
            result = self.timed("task_gate_report", task_yaml=text)
            self.assertFalse(result["ok"])
            self.assertEqual(result["kind"], "refused", text[:8])

    def test_json_nested_a_hundred_thousand_deep_is_an_invalid_argument(self) -> None:
        for text in ("[" * 100000, "[" * 100000 + "]" * 100000, '{"a":' * 50000):
            reply = json.loads(engine.call("wilson_interval", text))
            self.assertEqual((reply["ok"], reply["kind"]), (False, "invalid"))

    def test_invalid_variants_are_quoted_briefly_so_the_reply_always_fits(self) -> None:
        for token in ('"' * 98, "\u0001" * 98, "\\" * 98, "é" * 98, "x" * 98):
            result = run("variant_check", sequence="MKTAYIAKQR", variants=" ".join([token] * 40))
            self.assertTrue(result["ok"], token[:4])
            value = result["value"]
            self.assertEqual(value["n_invalid"], 40)
            self.assertTrue(all(len(item["label"]) <= 19 for item in value["variants"]))
            self.assertLessEqual(len(core.canonical(value).encode("utf-8")), core.MAX_RESULT_BYTES)
        # A real substitution is still shown whole, and plain text as typed; only
        # what JSON would have to escape is replaced.
        shown = value_of("variant_check", sequence="MKTAYIAKQR", variants="K2A, bad! caf\u00e9")["variants"]
        self.assertEqual([item["label"] for item in shown], ["K2A", "bad!", "caf?"])


class TheCatalog(unittest.TestCase):
    def test_the_controller_is_shown_every_agent_with_every_parameter_described(self) -> None:
        listed = json.loads(engine.catalog())
        self.assertEqual(listed["schema"], engine.ENGINE_SCHEMA)
        self.assertEqual([a["name"] for a in listed["agents"]], sorted(engine.AGENTS))
        for agent in listed["agents"]:
            self.assertTrue(agent["description"], agent["name"])
            for parameter, help_text in agent["parameters"].items():
                self.assertIsInstance(help_text, str)
                self.assertTrue(help_text, f"{agent['name']}.{parameter}")

    def test_the_documentation_names_every_agent_and_every_pipeline_interface(self) -> None:
        text = (ROOT_DIR / "docs" / "ENGINE.md").read_text(encoding="utf-8")
        for name in engine.AGENTS:
            self.assertIn(f"`{name}`", text, f"docs/ENGINE.md does not describe {name}")
        for row in engine.static_catalog()["pipeline"]:
            self.assertIn(row["interface"], text, f"docs/ENGINE.md omits the {row['interface']} interface")

    def test_the_static_catalog_is_json_and_carries_the_pipeline(self) -> None:
        static = engine.static_catalog()
        text = json.dumps(static)
        self.assertEqual(json.loads(text), static)
        self.assertEqual(len(static["pipeline"]), 10)
        self.assertEqual(sorted(a["name"] for a in static["agents"]), sorted(engine.AGENTS))
        self.assertTrue(all(a["wraps"] for a in static["agents"]))
        self.assertLess(len(text), 40000)
        self.assertEqual(static["tasks"], ["KRED_PILOT_001"], "the router and the page know which tasks ship")


class TheEnginesReach(unittest.TestCase):
    # "js" and "pyodide" are the bridge from Python back into the page: code that
    # could import them could call fetch or touch the worker. No agent has any use
    # for either, so neither may appear.
    FORBIDDEN_IMPORTS = {"subprocess", "socket", "ssl", "urllib", "http", "ctypes", "multiprocessing",
                         "importlib", "pickle", "marshal", "shutil", "os", "sys", "builtins", "js",
                         "pyodide", "_pyodide", "pyodide_js"}
    FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open", "getattr_static", "setattr", "delattr"}

    def sources(self):
        for path in sorted(ENGINE_DIR.glob("*.py")):
            yield path, ast.parse(path.read_text(encoding="utf-8"))

    def test_it_imports_nothing_that_could_reach_a_process_a_socket_or_the_filesystem_at_large(self) -> None:
        for path, tree in self.sources():
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]
                for name in names:
                    self.assertNotIn(name, self.FORBIDDEN_IMPORTS, f"{path.name} imports {name}")

    def test_it_never_evaluates_or_opens_anything_by_name(self) -> None:
        for path, tree in self.sources():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, self.FORBIDDEN_CALLS, f"{path.name}: {node.func.id}()")

    # What decides, in the project. Only the command line's `approve` calls grant or
    # deny; only a person-supplied authority calls TaskSpec.resolve. The engine runs
    # the controller, which requests an approval and waits, so none of these may
    # appear in any engine module: nothing a model can reach can answer a request.
    DECIDING_CALLS = {"grant", "deny", "_decide", "record_approval", "approve"}

    def test_no_agent_is_named_for_deciding_writing_or_ordering(self) -> None:
        for name in engine.AGENTS:
            for word in ("approve", "resolve", "execute", "submit", "order", "write", "grant"):
                self.assertNotIn(word, name, name)
        # The one agent that runs anything runs the controller, and says exactly that.
        self.assertEqual([name for name in engine.AGENTS if "run" in name], ["run_pipeline"])

    def test_it_never_calls_anything_that_grants_denies_or_resolves(self) -> None:
        for path, tree in self.sources():
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in self.DECIDING_CALLS:
                    self.fail(f"{path.name} reaches .{node.attr}, which only a person may do")
                if isinstance(node, ast.Name) and node.id in self.DECIDING_CALLS:
                    self.fail(f"{path.name} uses {node.id}, which only a person may do")
                # TaskSpec.resolve(field, value, source) fills a field. pathlib's
                # Path.resolve() shares the name but takes nothing, so it is allowed.
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "resolve"
                        and (node.args or any(keyword.arg != "strict" for keyword in node.keywords))):
                    self.fail(f"{path.name} calls .resolve with arguments, which fills a task field")

    def test_the_resolve_check_tells_a_path_from_a_task(self) -> None:
        def offends(source: str) -> bool:
            return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "resolve"
                       and (node.args or any(keyword.arg != "strict" for keyword in node.keywords))
                       for node in ast.walk(ast.parse(source)))
        self.assertFalse(offends("Path(__file__).resolve().parents[3]"))
        self.assertFalse(offends("p.resolve(strict=True)"))
        self.assertTrue(offends("task.resolve('reaction.substrate', value, 'operator:x')"))
        self.assertTrue(offends("task.resolve(field_path='a', value=1, source='b')"))

    def test_only_the_pipeline_module_touches_a_directory_and_only_a_throwaway_one(self) -> None:
        users = []
        for path, tree in self.sources():
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) and any(alias.name == "tempfile" for alias in node.names):
                    users.append(path.name)
        self.assertEqual(users, ["pipeline.py"])
        text = (ENGINE_DIR / "pipeline.py").read_text(encoding="utf-8")
        self.assertEqual(text.count("tempfile."), 1, "one throwaway directory, made in one place")
        self.assertIn("tempfile.TemporaryDirectory(", text, "and removed when the run ends")

    def test_no_agent_description_claims_to_approve_anything(self) -> None:
        for name, spec in engine.AGENTS.items():
            text = spec.description.lower()
            for phrase in ("approves", "will approve", "clears the gate", "grants"):
                self.assertNotIn(phrase, text, name)

    def test_the_only_file_it_reads_is_a_shipped_task(self) -> None:
        reads = [(path.name, node.attr) for path, tree in self.sources() for node in ast.walk(tree)
                 if isinstance(node, ast.Attribute) and node.attr in {"read_text", "read_bytes", "write_text", "write_bytes"}]
        self.assertEqual(reads, [("pipeline.py", "read_text")])


if __name__ == "__main__":
    unittest.main()
