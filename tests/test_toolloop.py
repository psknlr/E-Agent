"""A model reading the project's loaders, inside a fence that is code.

The point of giving a model tools here is that the tools know the rules. So the
tests check the rules survive the round trip: a bounded ``Km``, a censored
product, an insoluble construct and an ND all come back as a refusal with the
loader's own reason, and none of them comes back as a number.

The fence is checked structurally rather than by inspection of prose: a tool
that declares it writes or reaches the network cannot be registered, the turn
and call limits all stop the loop, an oversized result is cut, and a refusal is
recorded rather than raised so that the transcript is the output.

The client is the offline echo client throughout, scripted turn by turn, which
keeps the parsing, guarding and recording code on the executed path.
"""

from __future__ import annotations

import json
import unittest

from eagent.harness.llm import EchoClient, LLMError, NumericGuard
from eagent.harness.toolloop import (
    LoopLimits, ReadOnlyTool, ToolLoop, ToolLoopError, ToolRefusal,
    reference_tools,
)

TOOLS = reference_tools()


def script(*turns: dict) -> EchoClient:
    return EchoClient([json.dumps(t) for t in turns])


def call(name: str, **arguments) -> dict:
    return {"interface": name, "arguments": arguments, "rationale": "because"}


def loop_with(client: EchoClient, **limit_kwargs) -> ToolLoop:
    loop = ToolLoop(client, limits=LoopLimits(**limit_kwargs))
    loop.register_all(reference_tools())
    return loop


def results_of(transcript: dict) -> list[dict]:
    return [r for turn in transcript["turns"] for r in turn.get("results", [])]


# ==========================================================================
class TheFenceIsCodeNotAnInstruction(unittest.TestCase):
    def test_a_tool_that_writes_cannot_be_registered(self) -> None:
        loop = ToolLoop(script())
        with self.assertRaisesRegex(ToolLoopError, "read-only"):
            loop.register(ReadOnlyTool("w", "writes", {}, lambda: 1, writes=True))

    def test_a_tool_that_reaches_the_network_cannot_be_registered(self) -> None:
        loop = ToolLoop(script())
        with self.assertRaisesRegex(ToolLoopError, "read-only"):
            loop.register(ReadOnlyTool("n", "fetches", {}, lambda: 1,
                                       reaches_network=True))

    def test_every_shipped_tool_is_read_only_and_local(self) -> None:
        for tool in TOOLS:
            self.assertFalse(tool.writes, tool.name)
            self.assertFalse(tool.reaches_network, tool.name)
            self.assertTrue(tool.description.strip(), tool.name)

    def test_a_tool_with_no_description_is_refused(self) -> None:
        loop = ToolLoop(script())
        with self.assertRaisesRegex(ToolLoopError, "no description"):
            loop.register(ReadOnlyTool("x", "   ", {}, lambda: 1))

    def test_a_duplicate_name_is_refused(self) -> None:
        loop = ToolLoop(script())
        loop.register(ReadOnlyTool("x", "does a thing", {}, lambda: 1))
        with self.assertRaisesRegex(ToolLoopError, "already registered"):
            loop.register(ReadOnlyTool("x", "does another", {}, lambda: 2))

    def test_a_loop_with_no_tools_refuses_to_run(self) -> None:
        with self.assertRaisesRegex(ToolLoopError, "no tools are registered"):
            ToolLoop(script()).run("anything")

    def test_the_tools_offered_are_named_in_the_transcript(self) -> None:
        transcript = loop_with(script({"reasoning": "done"})).run("q")
        self.assertEqual(transcript["tools_offered"], sorted(t.name for t in TOOLS))


class TheLoopIsBounded(unittest.TestCase):
    def test_it_stops_at_the_turn_limit(self) -> None:
        asking = {"tool_calls": [call("reference_summary")]}
        transcript = loop_with(script(asking, asking, asking, asking),
                               max_turns=2).run("q")
        self.assertEqual(len(transcript["turns"]), 2)
        self.assertIn("turn limit of 2", transcript["stopped_because"])

    def test_it_stops_at_the_total_call_limit(self) -> None:
        asking = {"tool_calls": [call("reference_summary"),
                                 call("activity_summary")]}
        transcript = loop_with(script(asking, asking, asking),
                               max_turns=5, max_calls_total=3).run("q")
        self.assertEqual(transcript["tool_calls_made"], 3)
        self.assertIn("total limit of 3", transcript["stopped_because"])

    def test_it_clips_at_the_per_turn_call_limit(self) -> None:
        asking = {"tool_calls": [call("reference_summary"), call("activity_summary"),
                                 call("audit_verdict"), call("source_verification")]}
        transcript = loop_with(script(asking, {"reasoning": "done"}),
                               max_calls_per_turn=2).run("q")
        self.assertEqual(transcript["tool_calls_made"], 2)
        self.assertIn("per-turn limit", transcript["turns"][0]["clipped"])

    def test_an_oversized_result_is_cut_and_says_so(self) -> None:
        transcript = loop_with(
            script({"tool_calls": [call("list_kinetic_records")]},
                   {"reasoning": "done"}),
            max_result_bytes=50).run("q")
        result = results_of(transcript)[0]
        self.assertTrue(result["truncated"])
        self.assertTrue(result["value"]["truncated"])
        self.assertIn("ask a narrower question", result["value"]["note"])

    def test_a_model_that_answers_immediately_makes_no_call(self) -> None:
        transcript = loop_with(script({"reasoning": "I already know."})).run("q")
        self.assertEqual(transcript["tool_calls_made"], 0)
        self.assertEqual(transcript["answer"], "I already know.")
        self.assertIn("without asking for another tool",
                      transcript["stopped_because"])

    def test_the_limits_must_be_sane(self) -> None:
        for bad in ({"max_turns": 0}, {"max_calls_per_turn": 0},
                    {"max_calls_total": 0}, {"max_result_bytes": 0},
                    {"max_wall_seconds": 0}):
            with self.assertRaises(ToolLoopError):
                LoopLimits(**bad)

    def test_the_limits_in_force_are_recorded(self) -> None:
        transcript = loop_with(script({"reasoning": "x"}), max_turns=3).run("q")
        self.assertEqual(transcript["limits"]["max_turns"], 3)


class ARefusalIsRecordedNotRaised(unittest.TestCase):
    def test_structure_ids_can_be_discovered_then_read_from_actual_data(self) -> None:
        tools = {tool.name: tool for tool in reference_tools()}
        entries = tools["list_structure_entries"].run()["entries"]
        self.assertTrue(entries)
        for entry in entries:
            record = tools["structure_entry"].run(pdb_id=entry["pdb_id"])
            self.assertEqual(record["enzyme"], entry["enzyme"])

    def test_activity_ids_can_be_discovered_before_a_real_endpoint_read(self) -> None:
        tools = {tool.name: tool for tool in reference_tools()}
        catalog = tools["list_activity_constructs"].run()
        self.assertTrue(catalog["enzyme_ids"])
        self.assertIn("2a", catalog["substrate_ids"])
        for enzyme_id in catalog["enzyme_ids"]:
            endpoint = tools["activity_endpoint"].run(enzyme_id=enzyme_id, substrate_id="2a")
            self.assertEqual(endpoint["enzyme_id"], enzyme_id)

    def test_an_unknown_tool_lists_the_real_ones(self) -> None:
        transcript = loop_with(script({"tool_calls": [call("no_such_tool")]},
                                      {"reasoning": "done"})).run("q")
        result = results_of(transcript)[0]
        self.assertFalse(result["ok"])
        self.assertIn("there is no tool named", result["refusal"])
        self.assertIn("reference_summary", result["refusal"])

    def test_an_unexpected_argument_names_the_real_parameters(self) -> None:
        transcript = loop_with(
            script({"tool_calls": [call("kinetic_record", nonsense="x")]},
                   {"reasoning": "done"})).run("q")
        result = results_of(transcript)[0]
        self.assertFalse(result["ok"])
        self.assertIn("has no parameter(s) ['nonsense']", result["refusal"])
        self.assertIn("label_id", result["refusal"])

    def test_a_missing_argument_is_reported_not_crashed(self) -> None:
        transcript = loop_with(script({"tool_calls": [call("kinetic_record")]},
                                      {"reasoning": "done"})).run("q")
        self.assertFalse(results_of(transcript)[0]["ok"])

    def test_a_tool_raising_unexpectedly_is_contained(self) -> None:
        loop = ToolLoop(script({"tool_calls": [call("boom")]},
                               {"reasoning": "done"}))
        def boom() -> None:
            raise RuntimeError("something broke")
        loop.register(ReadOnlyTool("boom", "raises", {}, boom))
        transcript = loop.run("q")
        result = results_of(transcript)[0]
        self.assertFalse(result["ok"])
        self.assertIn("RuntimeError", result["refusal"])

    def test_a_provider_failure_ends_the_loop_with_the_reason(self) -> None:
        class Failing(EchoClient):
            def complete(self, *a, **k):          # noqa: ANN002, ANN003
                raise LLMError("HTTP 429 from the provider")
        loop = ToolLoop(Failing())
        loop.register_all(reference_tools())
        transcript = loop.run("q")
        self.assertIn("429", transcript["stopped_because"])
        self.assertEqual(transcript["tool_calls_made"], 0)

    def test_a_tool_refusal_class_reaches_the_model_as_a_refusal(self) -> None:
        loop = ToolLoop(script({"tool_calls": [call("picky")]},
                               {"reasoning": "done"}))
        def picky() -> None:
            raise ToolRefusal("I will not answer that, because reasons")
        loop.register(ReadOnlyTool("picky", "declines", {}, picky))
        result = results_of(loop.run("q"))[0]
        self.assertFalse(result["ok"])
        self.assertEqual(result["refusal"], "I will not answer that, because reasons")


class TheLoadersRulesSurviveTheRoundTrip(unittest.TestCase):
    """The reason to give a model these tools rather than the CSVs."""

    def one(self, name: str, **arguments) -> dict:
        transcript = loop_with(script({"tool_calls": [call(name, **arguments)]},
                                      {"reasoning": "done"})).run("q")
        return results_of(transcript)[0]

    def test_a_bounded_michaelis_constant_comes_back_as_a_refusal(self) -> None:
        value = self.one("kinetic_record",
                         label_id="PaHBDH_H150N_AAE_activity_only")["value"]
        self.assertIsNone(value["km"])
        self.assertIn("bound is not a point", value["refusals"]["km"])
        self.assertEqual(value["label_type"], "kcat_only_km_bounded")
        self.assertEqual(value["kcat"], 0.014)

    def test_a_not_determined_record_yields_no_quantity_at_all(self) -> None:
        value = self.one("kinetic_record", label_id="PNAS2015_Y190F_1")["value"]
        self.assertEqual(value["record_status"], "not_determined")
        self.assertEqual(value["available_quantities"], [])
        for quantity in ("kcat", "km", "efficiency_reported"):
            self.assertIsNone(value[quantity])
            self.assertIn("not determined", value["refusals"][quantity])

    def test_a_censored_endpoint_refuses_rather_than_returning_zero(self) -> None:
        value = self.one("activity_endpoint", enzyme_id="Ort-EZM-6",
                         substrate_id="2a")["value"]
        self.assertEqual(value["status"], "below_detection")
        self.assertIsNone(value["total_product_mM"])
        self.assertIn("left-censored", value["refusal"])
        self.assertIn("not a measured zero", value["refusal"])
        self.assertEqual(value["product_r_mM_as_printed"], 0.0,
                         "the source's printed value is still visible")

    def test_an_insoluble_construct_is_never_assayed_not_inactive(self) -> None:
        value = self.one("activity_endpoint", enzyme_id="Ort-RDM-2",
                         substrate_id="3a")["value"]
        self.assertFalse(value["solubly_expressed"])
        self.assertEqual(value["status"], "not_assayed")
        self.assertIn("not a catalytic negative", value["refusal"])

    def test_the_two_bad_ee_cells_are_withheld_from_the_model_too(self) -> None:
        value = self.one("activity_endpoint", enzyme_id="Ort-EZM-24",
                         substrate_id="2a")["value"]
        self.assertIsNone(value["ee_reported"])
        self.assertIn("ee_reported", value["withheld"])

    def test_the_relative_slope_is_flagged_as_a_different_label_type(self) -> None:
        value = self.one("activity_endpoint", enzyme_id="Ssal-KRED",
                         substrate_id="1a")["value"]
        self.assertEqual(value["label_type"], "relative_depletion_slope")
        self.assertIn("do not pool", value["note"])

    def test_a_quarantined_substrate_is_marked_as_such(self) -> None:
        """5a's deposited structure is wrong, and the model is told before it reasons."""
        flagged = self.one("activity_endpoint", enzyme_id="Ssal-KRED",
                           substrate_id="5a")["value"]
        self.assertTrue(flagged["substrate_smiles_quarantined"])
        clean = self.one("activity_endpoint", enzyme_id="Ssal-KRED",
                         substrate_id="4a")["value"]
        self.assertFalse(clean["substrate_smiles_quarantined"])

    def test_counts_come_with_the_warning_that_they_are_not_enzymes(self) -> None:
        value = self.one("reference_summary")["value"]
        self.assertIn("not counts of independent enzymes", value["note"])
        self.assertEqual(value["independence"]["n_independent_structure_lineages"], 6)

    def test_the_ortholog_group_count_is_available_to_the_model(self) -> None:
        value = self.one("activity_independence_groups")["value"]
        self.assertEqual(value["n_independent_at_cited_threshold"], 5)
        self.assertEqual(value["soluble_constructs"], 69)

    def test_the_audit_verdict_says_nothing_is_calibrated(self) -> None:
        value = self.one("audit_verdict")["value"]
        self.assertTrue(value["scenarios"])
        for scenario in value["scenarios"]:
            self.assertFalse(scenario["any_constraint_calibrated"],
                             scenario["scenario"])
            self.assertEqual(scenario["n_known_inactives"], 0)

    def test_the_source_verification_is_available_with_its_split(self) -> None:
        value = self.one("source_verification")["value"]
        self.assertEqual(value["summary"]["kinetic_quantities"]["mismatch"], 0)
        self.assertEqual(
            len(value["summary"]["records_fully_matched_against_the_papers_own_table"]),
            25)

    def test_an_unknown_identifier_is_refused_with_a_way_forward(self) -> None:
        self.assertIn("call list_kinetic_records",
                      self.one("kinetic_record", label_id="NOPE")["refusal"])
        self.assertIn("it holds",
                      self.one("structure_entry", pdb_id="9ZZZ")["refusal"])
        self.assertIn("the substrates are",
                      self.one("activity_endpoint", enzyme_id="Ssal-KRED",
                               substrate_id="9z")["refusal"])

    def test_every_result_carries_the_citation_keys_for_its_rows(self) -> None:
        for name, arguments in (("reference_summary", {}),
                                ("kinetic_record", {"label_id": "6ZZO_PaHBDH_AAE"}),
                                ("structure_entry", {"pdb_id": "1IPF"}),
                                ("activity_summary", {})):
            value = self.one(name, **arguments)["value"]
            self.assertIn("cite", value, name)
            self.assertEqual(set(value["cite"]),
                             {"artifact", "sha256", "row", "method"}, name)
            self.assertRegex(value["cite"]["sha256"], r"^[0-9a-f]{64}$", name)


class TheGuardStillAppliesToWhatTheModelSays(unittest.TestCase):
    def test_a_strict_guard_refuses_an_uncited_quantity_and_withholds_the_answer(self) -> None:
        loop = ToolLoop(script({"reasoning": "The kcat is 30 s-1, clearly."}),
                        guard=NumericGuard(strict=True))
        loop.register_all(reference_tools())
        transcript = loop.run("q")
        self.assertIn("numeric guard refused", transcript["stopped_because"])
        self.assertEqual(transcript["answer"], "")
        self.assertTrue(transcript["turns"][0]["guard"]["refused"])
        self.assertIn("30 s-1", transcript["turns"][0]["guard"]["uncited_quantities"])

    def test_a_lenient_guard_records_the_problem_and_keeps_the_answer(self) -> None:
        loop = ToolLoop(script({"reasoning": "The kcat is 30 s-1, clearly."}),
                        guard=NumericGuard(strict=False))
        loop.register_all(reference_tools())
        transcript = loop.run("q")
        self.assertEqual(transcript["answer"], "The kcat is 30 s-1, clearly.")
        self.assertFalse(transcript["turns"][0]["guard"]["clean"])
        self.assertIn("30 s-1", transcript["turns"][0]["guard"]["uncited_quantities"])

    def test_prose_with_no_quantity_is_clean(self) -> None:
        loop = ToolLoop(script({"reasoning": "No record carries that value."}),
                        guard=NumericGuard(strict=True))
        loop.register_all(reference_tools())
        transcript = loop.run("q")
        self.assertTrue(transcript["turns"][0]["guard"]["clean"])
        self.assertEqual(transcript["answer"], "No record carries that value.")

    def test_the_transcript_says_what_it_is_worth(self) -> None:
        transcript = loop_with(script({"reasoning": "ok"})).run("q")
        self.assertIn("nothing was written", transcript["caveat"])
        self.assertIn("without a citation naming its row", transcript["caveat"])
        self.assertIn("provider", transcript)
        self.assertIn("runs_remotely", transcript)


class TheTranscriptIsTheOutput(unittest.TestCase):
    def test_it_records_the_question_every_call_and_why_it_stopped(self) -> None:
        transcript = loop_with(
            script({"reasoning": "looking", "tool_calls": [call("reference_summary")]},
                   {"reasoning": "answered"})).run("how many lineages?")
        self.assertEqual(transcript["question"], "how many lineages?")
        self.assertEqual(transcript["turns"][0]["requested"], ["reference_summary"])
        self.assertEqual(transcript["turns"][0]["reasoning"], "looking")
        self.assertEqual(transcript["answer"], "answered")
        self.assertEqual(transcript["tool_calls_made"], 1)

    def test_it_is_json_serialisable(self) -> None:
        transcript = loop_with(
            script({"tool_calls": [call("kinetic_record",
                                        label_id="6ZZO_PaHBDH_AAE")]},
                   {"reasoning": "done"})).run("q")
        json.loads(json.dumps(transcript, default=str))

    def test_a_question_the_model_cannot_settle_is_carried_through(self) -> None:
        transcript = loop_with(
            script({"questions": ["Which conformer should be used?"]})).run("q")
        self.assertEqual(transcript["turns"][0]["questions"],
                         ["Which conformer should be used?"])

    def test_the_model_is_shown_the_tool_schemas(self) -> None:
        client = script({"reasoning": "done"})
        loop = ToolLoop(client)
        loop.register_all(reference_tools())
        loop.run("q")
        offered = client.calls[0]["tools"]
        self.assertEqual(sorted(offered), sorted(t.name for t in TOOLS))


if __name__ == "__main__":
    unittest.main()
