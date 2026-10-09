"""Tests for :mod:`eagent.cli` and the top-level package surface.

What is being defended here:

* the CLI runs with the network removed from under it -- help, validation,
  planning and the registry readers must never reach out, and a test that
  merely trusts them not to would not notice the day one does;
* ``validate`` reports exactly the fields each gate declares, and fails when
  a gate the user asked about cannot be satisfied, because a validator that
  prints problems and exits zero is a validator nobody's CI notices;
* ``--dry-run`` plans without executing a single interface, which is the only
  thing that makes "cost this run before spending anything" true;
* nothing printed is a fabricated number: an unrecorded cost prints as such,
  not as zero;
* the output carries no ANSI escapes, so it survives a pipe, a log file and
  an email;
* ``import eagent`` stays cheap, so the console script and any tool reading
  ``__version__`` do not drag in pydantic and the structure parser.

Runs under pytest, or standalone with ``python3 tests/test_cli.py``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence
from unittest import mock

import yaml
from click.testing import CliRunner

from eagent.cli import (
    EXIT_BLOCKED, EXIT_FAILED, EXIT_OK, EXIT_UNRESOLVED, EXIT_USAGE, main,
)
from eagent.harness.approval import APPROVAL_GATES
from eagent.schemas import GATE_REQUIREMENTS, TaskSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_TASK = REPO_ROOT / "configs" / "tasks" / "KRED_PILOT_001.yaml"
SRC_ROOT = REPO_ROOT / "src"

#: Any ANSI control sequence. The CLI must emit none of them.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@contextmanager
def no_network() -> Iterator[None]:
    """Make any attempt to open a socket fail loudly.

    Patched rather than asserted after the fact: a command that silently
    falls back to a cached value when the network is unavailable would pass
    an "it worked offline" test while still reaching out in production.
    """
    def refuse(*args: Any, **kwargs: Any):
        raise AssertionError("this command opened a network socket")

    with mock.patch.object(socket, "socket", refuse), \
            mock.patch.object(socket, "create_connection", refuse):
        yield


class CLICase(unittest.TestCase):
    """A runner and a scratch directory."""

    def setUp(self) -> None:
        self.runner = CliRunner()
        self.tmp = Path(tempfile.mkdtemp(prefix="eagent-cli-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def invoke(self, args: Sequence[str], *, offline: bool = True):
        if offline:
            with no_network():
                result = self.runner.invoke(main, list(args))
        else:
            result = self.runner.invoke(main, list(args))
        if result.exception is not None and not isinstance(
                result.exception, SystemExit):
            raise AssertionError(
                f"eagent {' '.join(args)} raised {result.exception!r}\n"
                f"{result.output}")
        self.assertIsNone(_ANSI.search(result.output),
                          "the output carries ANSI escapes; it must be "
                          "readable in a terminal with no colour support")
        return result


# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------

COMMANDS = ("init", "validate", "run", "approve", "status", "verify",
            "bundle", "bundle-verify", "templates", "sources", "reference",
            "model")


class HelpTests(CLICase):
    def test_the_group_help_runs_offline(self) -> None:
        result = self.invoke(["--help"])
        self.assertEqual(result.exit_code, EXIT_OK)
        for command in COMMANDS:
            self.assertIn(command, result.output)

    def test_every_command_has_help_and_it_runs_offline(self) -> None:
        for command in COMMANDS:
            with self.subTest(command=command):
                result = self.invoke([command, "--help"])
                self.assertEqual(result.exit_code, EXIT_OK)
                self.assertIn("Usage:", result.output)

    def test_the_subgroups_list_their_subcommands(self) -> None:
        for group, subcommands in (("templates", ("list", "show", "lint")),
                                   ("sources", ("list", "show", "independence"))):
            with self.subTest(group=group):
                result = self.invoke([group, "--help"])
                for sub in subcommands:
                    self.assertIn(sub, result.output)

    def test_the_help_states_the_exit_codes(self) -> None:
        result = self.invoke(["--help"])
        self.assertIn("Exit codes", result.output)

    def test_version_runs(self) -> None:
        result = self.invoke(["--version"])
        self.assertEqual(result.exit_code, EXIT_OK)
        self.assertIn("eagent", result.output)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

class ValidateTests(CLICase):
    def test_the_pilot_task_reports_exactly_the_fields_each_gate_declares(self) -> None:
        result = self.invoke(["validate", str(PILOT_TASK)])
        self.assertEqual(result.exit_code, EXIT_OK,
                         "a report with no gate asked about is not a failure")
        task = TaskSpec(**yaml.safe_load(PILOT_TASK.read_text(encoding="utf-8")))
        for gate in APPROVAL_GATES:
            self.assertIn(f"gate {gate}", result.output)
            for path in task.unresolved_for(gate):
                self.assertIn(path, result.output)
        # Exactly the declared requirements, nothing invented alongside them.
        self.assertEqual(task.unresolved_for("reaction_spec_confirmed"),
                         list(GATE_REQUIREMENTS["reaction_spec_confirmed"]))

    def test_asking_about_a_gate_that_cannot_be_satisfied_exits_non_zero(self) -> None:
        result = self.invoke(["validate", str(PILOT_TASK),
                              "--gate", "reaction_spec_confirmed"])
        self.assertEqual(result.exit_code, EXIT_UNRESOLVED)
        for path in GATE_REQUIREMENTS["reaction_spec_confirmed"]:
            self.assertIn(path, result.output)
        self.assertIn("blocked on a human decision: reaction_spec_confirmed",
                      result.output)

    def test_all_gates_fails_on_the_pilot_task(self) -> None:
        result = self.invoke(["validate", str(PILOT_TASK), "--all-gates"])
        self.assertEqual(result.exit_code, EXIT_UNRESOLVED)
        self.assertIn("3 gate(s) you asked about cannot be satisfied",
                      result.output)

    def test_a_satisfiable_gate_passes(self) -> None:
        task = yaml.safe_load(PILOT_TASK.read_text(encoding="utf-8"))
        task["reaction"]["product"]["isomeric_smiles"] = "C[C@H](O)c1ccccc1"
        path = self.tmp / "resolved.yaml"
        path.write_text(yaml.safe_dump(task), encoding="utf-8")
        result = self.invoke(["validate", str(path),
                              "--gate", "functional_criteria_confirmed"])
        self.assertEqual(result.exit_code, EXIT_OK)
        self.assertIn("has its fields resolved", result.output)
        self.assertIn("The decision itself is still a human's", result.output)

    def test_a_task_file_flag_is_not_treated_as_an_approval(self) -> None:
        task = yaml.safe_load(PILOT_TASK.read_text(encoding="utf-8"))
        task["approval"]["synthesis_authorized"] = True
        path = self.tmp / "flagged.yaml"
        path.write_text(yaml.safe_dump(task), encoding="utf-8")
        result = self.invoke(["validate", str(path)])
        self.assertIn("a flag in a file is not an approval", result.output)

    def test_an_unknown_gate_is_a_usage_error(self) -> None:
        result = self.invoke(["validate", str(PILOT_TASK), "--gate", "ship_it"])
        self.assertEqual(result.exit_code, EXIT_USAGE)
        self.assertIn("not an approval gate", result.output)

    def test_a_malformed_task_file_names_the_field(self) -> None:
        path = self.tmp / "broken.yaml"
        path.write_text("task_id: B1\nreaction:\n  substrate:\n"
                        "    isomeric_smiles: TBD\n", encoding="utf-8")
        result = self.invoke(["validate", str(path)])
        self.assertEqual(result.exit_code, EXIT_UNRESOLVED)
        self.assertIn("isomeric_smiles", result.output)

    def test_invalid_yaml_is_reported_as_such(self) -> None:
        path = self.tmp / "bad.yaml"
        path.write_text("task_id: [unclosed\n", encoding="utf-8")
        result = self.invoke(["validate", str(path)])
        self.assertEqual(result.exit_code, EXIT_UNRESOLVED)
        self.assertIn("not valid YAML", result.output)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

class InitTests(CLICase):
    def _init(self, *args: str):
        out = self.tmp / "T_NEW.yaml"
        result = self.invoke(["init", "T_NEW", "--out", str(out), *args])
        return result, out

    def test_the_scaffold_loads_and_keeps_its_nulls(self) -> None:
        result, out = self._init()
        self.assertEqual(result.exit_code, EXIT_OK)
        task = TaskSpec(**yaml.safe_load(out.read_text(encoding="utf-8")))
        self.assertEqual(task.task_id, "T_NEW")
        self.assertIsNone(task.reaction.substrate.isomeric_smiles)
        self.assertIsNone(task.reaction.product.isomeric_smiles)
        self.assertIsNone(task.reaction.product.creates_new_stereocenter)
        self.assertIsNone(task.reaction.substrate.is_prochiral)
        self.assertIsNone(task.reaction.atom_mapped_reaction_smiles)
        self.assertIsNone(task.conditions.pH)
        self.assertIsNone(task.conditions.expression_host)
        self.assertEqual(task.assumptions, [])
        self.assertFalse(task.stereo_task)

    def test_the_scaffold_explains_why_the_nulls_are_there(self) -> None:
        _, out = self._init()
        text = out.read_text(encoding="utf-8").lower()
        self.assertIn("deliberate", text)
        self.assertIn("operator", text)

    def test_the_scaffold_blocks_every_gate_it_should(self) -> None:
        _, out = self._init("--reaction-class", "ketone_to_secondary_alcohol")
        task = TaskSpec(**yaml.safe_load(out.read_text(encoding="utf-8")))
        self.assertEqual(task.unresolved_for("reaction_spec_confirmed"),
                         list(GATE_REQUIREMENTS["reaction_spec_confirmed"]))
        result = self.invoke(["validate", str(out), "--all-gates"])
        self.assertEqual(result.exit_code, EXIT_UNRESOLVED)

    def test_it_refuses_to_overwrite_without_force(self) -> None:
        _, out = self._init()
        again = self.invoke(["init", "T_NEW", "--out", str(out)])
        self.assertEqual(again.exit_code, EXIT_USAGE)
        self.assertIn("already exists", again.output)
        forced = self.invoke(["init", "T_NEW", "--out", str(out), "--force"])
        self.assertEqual(forced.exit_code, EXIT_OK)

    def test_engineering_mode_without_a_parent_is_refused(self) -> None:
        result, _ = self._init("--mode", "substrate_directed_engineering")
        self.assertEqual(result.exit_code, EXIT_USAGE)
        self.assertIn("--parent", result.output)

    def test_engineering_mode_with_a_parent_scaffolds(self) -> None:
        result, out = self._init("--mode", "substrate_directed_engineering",
                                 "--parent", "P12345")
        self.assertEqual(result.exit_code, EXIT_OK)
        task = TaskSpec(**yaml.safe_load(out.read_text(encoding="utf-8")))
        self.assertEqual(task.parent_enzymes, ["P12345"])

    def test_an_unknown_reaction_class_lists_the_known_ones(self) -> None:
        result, _ = self._init("--reaction-class", "alchemy")
        self.assertEqual(result.exit_code, EXIT_USAGE)
        self.assertIn("ketone_to_secondary_alcohol", result.output)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class DryRunTests(CLICase):
    """--dry-run must plan without executing anything."""

    @contextmanager
    def _no_execution(self) -> Iterator[None]:
        """Make executing any interface an error.

        Both ``run`` and ``execute`` are poisoned rather than ``subprocess``:
        an external scientific tool is only ever launched from inside an
        interface, while a blanket subprocess ban would instead catch the
        standard library probing the machine for the run manifest, which is
        not what "executes no external tool" is about.
        """
        from eagent.tools.base import ScientificInterface

        def refuse(*args: Any, **kwargs: Any):
            raise AssertionError("a dry run executed a scientific interface")

        with mock.patch.object(ScientificInterface, "run", refuse), \
                mock.patch.object(ScientificInterface, "execute", refuse):
            yield

    def _dry_run(self, *extra: str):
        rundir = self.tmp / "dry"
        with self._no_execution():
            result = self.invoke(["run", str(PILOT_TASK), "--rundir",
                                  str(rundir), "--dry-run", *extra])
        return result, rundir

    def test_a_dry_run_executes_no_interface(self) -> None:
        result, rundir = self._dry_run()
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        plan = json.loads((rundir / "dry_run_plan.json").read_text(
            encoding="utf-8"))
        self.assertFalse(plan["executed"])
        self.assertIn("no interface was executed", plan["statement"])

    def test_a_dry_run_writes_no_run_manifest(self) -> None:
        """A manifest for a run that did not happen reads like one that did."""
        _, rundir = self._dry_run()
        self.assertFalse((rundir / "run_manifest.json").exists())

    def test_a_dry_run_plans_every_protocol_step(self) -> None:
        from eagent.harness.registry import PROTOCOL_ORDER

        result, rundir = self._dry_run()
        plan = json.loads((rundir / "dry_run_plan.json").read_text(
            encoding="utf-8"))
        self.assertEqual([s["interface"] for s in plan["steps"]],
                         list(PROTOCOL_ORDER))
        for name in PROTOCOL_ORDER:
            self.assertIn(name, result.output)

    def test_a_dry_run_states_no_monetary_total(self) -> None:
        result, rundir = self._dry_run()
        plan = json.loads((rundir / "dry_run_plan.json").read_text(
            encoding="utf-8"))
        self.assertIsNone(plan["cost"]["estimated_total"])
        self.assertIsNone(plan["cost"]["currency"])
        self.assertIn("no price list", plan["cost"]["note"])
        self.assertIn("nothing has been recorded", result.output)
        self.assertNotIn("recorded so far:         0", result.output)

    def test_a_dry_run_reports_every_gate_and_what_blocks_it(self) -> None:
        result, rundir = self._dry_run()
        plan = json.loads((rundir / "dry_run_plan.json").read_text(
            encoding="utf-8"))
        self.assertEqual(sorted(plan["gates"]), sorted(APPROVAL_GATES))
        self.assertEqual(plan["gates"]["reaction_spec_confirmed"],
                         list(GATE_REQUIREMENTS["reaction_spec_confirmed"]))

    def test_a_dry_run_never_permits_the_network(self) -> None:
        _, rundir = self._dry_run()
        plan = json.loads((rundir / "dry_run_plan.json").read_text(
            encoding="utf-8"))
        self.assertFalse(plan["policy"]["allow_network"])
        self.assertTrue(plan["policy"]["dry_run"])

    def test_offline_beats_allow_network(self) -> None:
        result, rundir = self._dry_run("--offline", "--allow-network")
        plan = json.loads((rundir / "dry_run_plan.json").read_text(
            encoding="utf-8"))
        self.assertFalse(plan["policy"]["allow_network"])
        self.assertIn("--offline", result.output)

    def test_the_planned_scale_is_the_task_s_own_budget(self) -> None:
        _, rundir = self._dry_run()
        plan = json.loads((rundir / "dry_run_plan.json").read_text(
            encoding="utf-8"))
        task = TaskSpec(**yaml.safe_load(PILOT_TASK.read_text(encoding="utf-8")))
        self.assertEqual(plan["budget"]["new_constructs_round_1"],
                         task.budget.new_constructs_round_1)

    def test_the_uncalibrated_windows_are_reported_up_front(self) -> None:
        result, _ = self._dry_run()
        self.assertIn("carry no calibration set", result.output)


class RunTests(CLICase):
    """A real run of the pilot task, which must stop at the first gate."""

    def test_the_pilot_run_stops_at_the_reaction_gate(self) -> None:
        rundir = self.tmp / "run"
        result = self.invoke(["run", str(PILOT_TASK), "--rundir", str(rundir)])
        self.assertEqual(result.exit_code, EXIT_BLOCKED, result.output)
        self.assertIn("blocked on a human decision: reaction_spec_confirmed",
                      result.output)
        self.assertIn("eagent approve reaction_spec_confirmed", result.output)
        self.assertTrue((rundir / "run_manifest.json").is_file())
        self.assertTrue((rundir / "controller_report.json").is_file())
        self.assertTrue((rundir / "approvals.json").is_file())

    def test_the_run_streams_the_step_its_qc_flags_and_uncertainties(self) -> None:
        rundir = self.tmp / "run"
        result = self.invoke(["run", str(PILOT_TASK), "--rundir", str(rundir)])
        self.assertIn("[confirm_reaction_spec] normalize_reaction", result.output)
        self.assertIn("qc BLOCKER:", result.output)
        self.assertIn("uncertainty ", result.output)
        self.assertIn("next (human):", result.output)

    def test_an_unrecorded_cost_is_not_printed_as_zero(self) -> None:
        rundir = self.tmp / "run"
        result = self.invoke(["run", str(PILOT_TASK), "--rundir", str(rundir)])
        self.assertIn("nothing recorded by any step", result.output)

    def test_a_single_step_runs_only_that_step(self) -> None:
        rundir = self.tmp / "step"
        result = self.invoke(["run", str(PILOT_TASK), "--rundir", str(rundir),
                              "--step", "confirm_reaction_spec"])
        self.assertIn("normalize_reaction", result.output)
        manifest = json.loads((rundir / "run_manifest.json").read_text(
            encoding="utf-8"))
        self.assertEqual([s["interface"] for s in manifest["steps"]],
                         ["normalize_reaction"])

    def test_step_cannot_be_combined_with_from_or_until(self) -> None:
        result = self.invoke(["run", str(PILOT_TASK), "--rundir",
                              str(self.tmp / "x"), "--step", "retrieve_evidence",
                              "--until", "mine_sequences"])
        self.assertEqual(result.exit_code, EXIT_USAGE)

    def test_an_unknown_stage_lists_the_real_ones(self) -> None:
        result = self.invoke(["run", str(PILOT_TASK), "--rundir",
                              str(self.tmp / "x"), "--step", "do_the_science"])
        self.assertEqual(result.exit_code, EXIT_USAGE)
        self.assertIn("confirm_reaction_spec", result.output)
        self.assertIn("stages that execute a step", result.output)

    def test_the_run_says_it_invents_no_data_flow_between_steps(self) -> None:
        rundir = self.tmp / "run"
        result = self.invoke(["run", str(PILOT_TASK), "--rundir", str(rundir)])
        self.assertIn("does not invent the data flow", result.output)


# ---------------------------------------------------------------------------
# approve / status
# ---------------------------------------------------------------------------

class ApproveTests(CLICase):
    def _blocked_run(self) -> Path:
        rundir = self.tmp / "run"
        self.invoke(["run", str(PILOT_TASK), "--rundir", str(rundir)])
        return rundir

    def test_an_approval_needs_an_actor(self) -> None:
        rundir = self._blocked_run()
        result = self.invoke(["approve", "reaction_spec_confirmed",
                              "--rundir", str(rundir)])
        self.assertEqual(result.exit_code, EXIT_USAGE)
        self.assertIn("--actor", result.output)

    def test_an_approval_is_recorded_with_the_actor_and_the_time(self) -> None:
        rundir = self._blocked_run()
        result = self.invoke(["approve", "reaction_spec_confirmed",
                              "--rundir", str(rundir), "--actor", "R. Operator",
                              "--note", "structures confirmed by the chemist"])
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        manifest = json.loads((rundir / "run_manifest.json").read_text(
            encoding="utf-8"))
        approvals = manifest["approvals"]
        self.assertEqual(len(approvals), 1)
        self.assertEqual(approvals[0]["gate"], "reaction_spec_confirmed")
        self.assertEqual(approvals[0]["decision"], "approve")
        self.assertEqual(approvals[0]["actor"], "R. Operator")
        self.assertTrue(approvals[0]["at"])

    def test_the_decision_point_and_its_consequence_are_shown_first(self) -> None:
        rundir = self._blocked_run()
        result = self.invoke(["approve", "reaction_spec_confirmed",
                              "--rundir", str(rundir), "--actor", "R. Operator"])
        self.assertIn("Is this the reaction the project is about", result.output)
        self.assertIn("if this is wrong:", result.output)

    def test_a_denial_is_recorded_too(self) -> None:
        rundir = self._blocked_run()
        result = self.invoke(["approve", "reaction_spec_confirmed",
                              "--rundir", str(rundir), "--actor", "R. Operator",
                              "--deny", "--note", "wrong substrate"])
        self.assertEqual(result.exit_code, EXIT_OK)
        manifest = json.loads((rundir / "run_manifest.json").read_text(
            encoding="utf-8"))
        self.assertEqual(manifest["approvals"][0]["decision"], "deny")

    def test_a_gate_with_no_request_cannot_be_approved(self) -> None:
        rundir = self._blocked_run()
        result = self.invoke(["approve", "synthesis_authorized",
                              "--rundir", str(rundir), "--actor", "R. Operator"])
        self.assertEqual(result.exit_code, EXIT_UNRESOLVED)
        self.assertIn("no pending request", result.output)
        self.assertIn("must answer a request", result.output)

    def test_approving_without_a_run_is_refused(self) -> None:
        empty = self.tmp / "nowhere"
        empty.mkdir()
        result = self.invoke(["approve", "reaction_spec_confirmed",
                              "--rundir", str(empty), "--actor", "R. Operator"])
        self.assertEqual(result.exit_code, EXIT_UNRESOLVED)

    def test_the_run_resumes_past_the_gate_once_it_is_granted(self) -> None:
        rundir = self._blocked_run()
        self.invoke(["approve", "reaction_spec_confirmed", "--rundir",
                     str(rundir), "--actor", "R. Operator"])
        result = self.invoke(["run", str(PILOT_TASK), "--rundir", str(rundir)])
        self.assertIn("retrieve_evidence", result.output)
        self.assertIn("resumed", result.output)


class StatusTests(CLICase):
    def test_status_reports_the_stage_costs_and_open_uncertainties(self) -> None:
        rundir = self.tmp / "run"
        self.invoke(["run", str(PILOT_TASK), "--rundir", str(rundir)])
        result = self.invoke(["status", str(rundir)])
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        self.assertIn("stage:", result.output)
        self.assertIn("awaiting_human", result.output)
        self.assertIn("open uncertainties", result.output)
        self.assertIn("pending decisions", result.output)
        self.assertIn("nothing was recorded by any step", result.output)
        self.assertIn("blocking QC flags", result.output)

    def test_status_on_a_directory_with_no_manifest_is_refused(self) -> None:
        empty = self.tmp / "empty"
        empty.mkdir()
        result = self.invoke(["status", str(empty)])
        self.assertEqual(result.exit_code, EXIT_UNRESOLVED)
        self.assertIn("run_manifest.json", result.output)


# ---------------------------------------------------------------------------
# verify / bundle
# ---------------------------------------------------------------------------

class VerifyCommandTests(CLICase):
    def test_a_run_with_nothing_to_check_is_not_reported_as_verified(self) -> None:
        rundir = self.tmp / "run"
        self.invoke(["run", str(PILOT_TASK), "--rundir", str(rundir)])
        result = self.invoke(["verify", str(rundir)])
        self.assertEqual(result.exit_code, EXIT_UNRESOLVED)
        self.assertIn("nothing in", result.output)
        self.assertIn("an empty verification is not a passed one", result.output)

    def test_the_verifier_runs_over_serialised_candidates(self) -> None:
        from eagent.schemas.candidate import Candidate, SequenceRecord

        rundir = self.tmp / "run"
        self.invoke(["run", str(PILOT_TASK), "--rundir", str(rundir)])
        candidate = Candidate(
            candidate_id="C1",
            sequence_record=SequenceRecord(candidate_id="C1",
                                           sequence="MKAVVTGAAQGIG"))
        (rundir / "candidates.json").write_text(
            json.dumps([candidate.model_dump(mode="json")]), encoding="utf-8")
        result = self.invoke(["verify", str(rundir)])
        self.assertIn("candidates checked", result.output)
        self.assertIn("checks run", result.output)
        self.assertIn("could not be checked", result.output)


class BundleCommandTests(CLICase):
    def _run_dir(self) -> Path:
        rundir = self.tmp / "run"
        self.invoke(["run", str(PILOT_TASK), "--rundir", str(rundir)])
        return rundir

    def test_bundle_reports_the_missing_items_and_does_not_claim_completeness(self) -> None:
        rundir = self._run_dir()
        result = self.invoke(["bundle", str(rundir)])
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        self.assertIn("complete: NO", result.output)
        self.assertIn("[MISSING] candidate_sequences.fasta", result.output)
        self.assertIn("the package is incomplete", result.output)

    def test_strict_makes_an_incomplete_package_a_failure(self) -> None:
        rundir = self._run_dir()
        result = self.invoke(["bundle", str(rundir), "--strict"])
        self.assertEqual(result.exit_code, EXIT_FAILED)

    def test_the_bundle_verifies_straight_after_assembly(self) -> None:
        rundir = self._run_dir()
        self.invoke(["bundle", str(rundir)])
        result = self.invoke(["bundle-verify", str(rundir / "bundle")])
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        self.assertIn("every declared file is present", result.output)

    def test_bundle_verify_fails_on_a_tampered_package(self) -> None:
        rundir = self._run_dir()
        self.invoke(["bundle", str(rundir)])
        target = rundir / "bundle" / "run_manifest.json"
        target.write_text(target.read_text(encoding="utf-8") + "\n",
                          encoding="utf-8")
        result = self.invoke(["bundle-verify", str(rundir / "bundle")])
        self.assertEqual(result.exit_code, EXIT_FAILED)
        self.assertIn("does not match the manifest", result.output)


# ---------------------------------------------------------------------------
# templates / sources
# ---------------------------------------------------------------------------

class TemplatesTests(CLICase):
    def test_list_runs_offline_and_names_every_shipped_template(self) -> None:
        from eagent.harness.templates import TemplateLibrary

        result = self.invoke(["templates", "list"])
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        for template_id in TemplateLibrary.load().ids():
            self.assertIn(template_id, result.output)

    def test_show_prints_the_caveats_a_report_must_carry(self) -> None:
        result = self.invoke(["templates", "show",
                              "cat.sdr.nadph_carbonyl_reduction.v1"])
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        self.assertIn("caveats", result.output)
        self.assertIn("uncalibrated", result.output)

    def test_show_refuses_an_unknown_template(self) -> None:
        result = self.invoke(["templates", "show", "cat.nope.v9"])
        self.assertEqual(result.exit_code, EXIT_UNRESOLVED)

    def test_lint_reports_the_gaps_without_failing_by_default(self) -> None:
        result = self.invoke(["templates", "lint"])
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        self.assertIn("integrity", result.output)
        self.assertIn("geometry windows", result.output)

    def test_lint_strict_follows_the_library_s_own_verdict(self) -> None:
        from eagent.harness.templates import TemplateLibrary

        problems = TemplateLibrary.load().integrity_problems()
        result = self.invoke(["templates", "lint", "--strict"])
        self.assertEqual(result.exit_code,
                         EXIT_FAILED if problems else EXIT_OK, result.output)
        for problem in problems:
            self.assertIn(problem, result.output)

    def test_lint_says_an_unfitted_window_may_not_reject_a_candidate(self) -> None:
        result = self.invoke(["templates", "lint"])
        self.assertIn("a failure may not disqualify a candidate", result.output)


class SourcesTests(CLICase):
    def test_list_runs_offline(self) -> None:
        from eagent.datalayer.registry import SourceRegistry

        result = self.invoke(["sources", "list"])
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        registry = SourceRegistry.from_directory()
        self.assertIn(f"{len(registry)} registered sources", result.output)
        self.assertIn("connectivity_verified=false", result.output)

    def test_show_prints_what_a_source_may_not_claim(self) -> None:
        result = self.invoke(["sources", "show", "brenda"])
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        self.assertIn("not good for", result.output)
        self.assertIn("evidence ceiling", result.output)
        self.assertIn("connectivity verified:   no", result.output)

    def test_an_unknown_source_is_refused(self) -> None:
        result = self.invoke(["sources", "show", "definitely_not_a_database"])
        self.assertEqual(result.exit_code, EXIT_UNRESOLVED)

    def test_independence_separates_the_two_counts(self) -> None:
        from eagent.datalayer.registry import SourceRegistry

        report = SourceRegistry.from_directory().independence_report()
        result = self.invoke(["sources", "independence"])
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        self.assertIn(f"{report.n_independent} of {report.n_groups}",
                      result.output)
        self.assertIn("lineage incomplete", result.output)

    def test_an_unknown_layer_lists_the_real_ones(self) -> None:
        result = self.invoke(["sources", "list", "--layer", "vibes"])
        self.assertEqual(result.exit_code, EXIT_USAGE)


# ---------------------------------------------------------------------------
# the KRED reference set
# ---------------------------------------------------------------------------

class ReferenceTests(CLICase):
    """The reference-set commands run offline and refuse in the places they should."""

    def setUp(self) -> None:
        super().setUp()
        self.empty_cache = self.tmp / "no_coordinates"
        self.empty_cache.mkdir()

    def copy_set(self) -> Path:
        from eagent.eval.kred_reference import default_reference_dir

        target = self.tmp / "set"
        shutil.copytree(default_reference_dir(), target)
        return target

    def test_the_subcommands_are_listed(self) -> None:
        result = self.invoke(["reference", "--help"])
        for sub in ("verify", "manifest", "fetch-coordinates", "bindings", "audit",
                    "verify-sources", "import-workbook"):
            self.assertIn(sub, result.output)

    def test_verify_passes_on_the_shipped_set_and_prints_the_counts(self) -> None:
        result = self.invoke(["reference", "verify", "--cache-dir", str(self.empty_cache)])
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        self.assertIn("reference set intact", result.output)
        self.assertIn("structures:", result.output)
        self.assertIn("independent lineages:    6", result.output)
        self.assertIn("coordinates not checked", result.output)

    def test_verify_fails_on_one_edited_byte_and_names_the_file(self) -> None:
        base = self.copy_set()
        path = base / "tables" / "kinetics.csv"
        path.write_bytes(path.read_bytes().replace(b"0.49", b"0.94", 1))
        result = self.invoke(["reference", "verify", "--dir", str(base),
                              "--cache-dir", str(self.empty_cache)])
        self.assertEqual(result.exit_code, EXIT_FAILED)
        self.assertIn("tables/kinetics.csv", result.output)

    def test_manifest_check_reports_a_match_and_a_mismatch(self) -> None:
        self.assertEqual(self.invoke(["reference", "manifest"]).exit_code, EXIT_OK)
        base = self.copy_set()
        (base / "tables" / "sources.csv").write_bytes(b"x")
        result = self.invoke(["reference", "manifest", "--dir", str(base)])
        self.assertEqual(result.exit_code, EXIT_FAILED)
        self.assertIn("manifest.file_changed", result.output)

    def test_manifest_write_is_a_separate_deliberate_flag(self) -> None:
        base = self.copy_set()
        (base / "tables" / "sources.csv").write_bytes(b"x")
        result = self.invoke(["reference", "manifest", "--dir", str(base), "--write"])
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        self.assertEqual(self.invoke(["reference", "manifest", "--dir", str(base)]).exit_code,
                         EXIT_OK)

    def test_fetching_coordinates_needs_the_network_flag_and_says_so(self) -> None:
        result = self.invoke(["reference", "fetch-coordinates",
                              "--cache-dir", str(self.empty_cache)])
        self.assertEqual(result.exit_code, EXIT_BLOCKED)
        self.assertIn("--allow-network", result.output)
        self.assertEqual(list(self.empty_cache.iterdir()), [])

    def test_audit_without_the_coordinate_files_points_at_how_to_get_them(self) -> None:
        result = self.invoke(["reference", "audit", "--cache-dir", str(self.empty_cache)])
        self.assertEqual(result.exit_code, EXIT_UNRESOLVED)
        self.assertIn("fetch-coordinates", result.output)

    def test_bindings_without_the_coordinate_files_refuses_the_same_way(self) -> None:
        result = self.invoke(["reference", "bindings", "--cache-dir", str(self.empty_cache)])
        self.assertEqual(result.exit_code, EXIT_UNRESOLVED)

    def test_verify_sources_without_the_documents_says_it_does_not_fetch_them(self) -> None:
        docs = self.tmp / "docs"
        docs.mkdir()
        result = self.invoke(["reference", "verify-sources", "--docs", str(docs)])
        self.assertEqual(result.exit_code, EXIT_USAGE)
        self.assertIn("does not fetch them", result.output)

    def test_import_workbook_makes_a_set_that_loads_and_verifies(self) -> None:
        from eagent.eval.kred_reference import default_reference_dir

        workbook = next((default_reference_dir() / "source").glob("*.xlsx"))
        out = self.tmp / "v0.2"
        result = self.invoke(["reference", "import-workbook", str(workbook), "--out", str(out)])
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        self.assertTrue((out / "source" / workbook.name).is_file())
        verified = self.invoke(["reference", "verify", "--dir", str(out),
                                "--cache-dir", str(self.empty_cache)])
        self.assertEqual(verified.exit_code, EXIT_OK, verified.output)
        self.assertIn("no coordinates manifest", verified.output)

    def test_import_workbook_refuses_a_directory_that_is_not_empty(self) -> None:
        from eagent.eval.kred_reference import default_reference_dir

        workbook = next((default_reference_dir() / "source").glob("*.xlsx"))
        out = self.tmp / "occupied"
        out.mkdir()
        (out / "x").write_text("x", encoding="utf-8")
        result = self.invoke(["reference", "import-workbook", str(workbook), "--out", str(out)])
        self.assertEqual(result.exit_code, EXIT_USAGE)

    def test_import_workbook_refuses_a_file_that_is_not_the_workbook_it_knows(self) -> None:
        bad = self.tmp / "other.xlsx"
        bad.write_bytes(b"not a workbook")
        result = self.invoke(["reference", "import-workbook", str(bad),
                              "--out", str(self.tmp / "new")])
        self.assertEqual(result.exit_code, EXIT_USAGE)


# ---------------------------------------------------------------------------
# model providers and the read-only tool loop
# ---------------------------------------------------------------------------

class ModelTests(CLICase):
    """The provider surface runs offline; anything remote needs the flag."""

    def test_the_subcommands_are_listed(self) -> None:
        result = self.invoke(["model", "--help"])
        for sub in ("providers", "ask"):
            self.assertIn(sub, result.output)

    def test_providers_runs_offline_and_states_what_is_unverified(self) -> None:
        result = self.invoke(["model", "providers"])
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        for provider in ("anthropic", "openai", "minimax"):
            self.assertIn(provider, result.output)
        self.assertIn("shape verified:    no", result.output)
        self.assertIn("route answers:     yes", result.output)

    def test_providers_names_the_variable_to_set(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False) as environ:
            for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MINIMAX_API_KEY"):
                environ.pop(name, None)
            result = self.invoke(["model", "providers"])
        self.assertIn("set OPENAI_API_KEY", result.output)
        self.assertIn("no provider key is set", result.output)

    def test_probing_needs_the_network_flag(self) -> None:
        result = self.invoke(["model", "providers", "--probe"])
        self.assertEqual(result.exit_code, EXIT_BLOCKED)
        self.assertIn("--allow-network", result.output)

    def test_asking_needs_the_network_flag_and_says_what_leaves(self) -> None:
        result = self.invoke(["model", "ask", "how many lineages?",
                              "--provider", "openai", "--model", "gpt-4o"])
        self.assertEqual(result.exit_code, EXIT_BLOCKED)
        self.assertIn("leave this", result.output)

    def test_asking_an_unconfigured_provider_refuses_with_the_variable(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False) as environ:
            environ.pop("OPENAI_API_KEY", None)
            result = self.invoke(["model", "ask", "q", "--provider", "openai",
                                  "--model", "gpt-4o", "--allow-network"],
                                 offline=False)
        self.assertEqual(result.exit_code, EXIT_UNRESOLVED)
        self.assertIn("OPENAI_API_KEY", result.output)

    def test_asking_an_unknown_provider_is_a_usage_error(self) -> None:
        result = self.invoke(["model", "ask", "q", "--provider", "nope",
                              "--model", "m", "--allow-network"], offline=False)
        self.assertEqual(result.exit_code, EXIT_USAGE)

    def test_the_model_must_be_named(self) -> None:
        result = self.invoke(["model", "ask", "q", "--provider", "openai",
                              "--allow-network"], offline=False)
        self.assertEqual(result.exit_code, EXIT_USAGE)

    def test_the_whole_loop_runs_against_a_scripted_client(self) -> None:
        """No network: the client is replaced, so the CLI path itself is exercised."""
        import json as _json

        from eagent.harness.llm import EchoClient

        scripted = EchoClient([
            _json.dumps({"reasoning": "checking",
                         "tool_calls": [{"interface": "reference_summary",
                                         "arguments": {}}]}),
            _json.dumps({"reasoning": "Six lineages among the entries."}),
        ])
        out_path = self.tmp / "transcript.json"
        with mock.patch("eagent.harness.providers.build_client",
                        return_value=scripted):
            result = self.invoke(["model", "ask", "how many lineages?",
                                  "--provider", "openai", "--model", "gpt-4o",
                                  "--allow-network", "--out", str(out_path)],
                                 offline=False)
        self.assertEqual(result.exit_code, EXIT_OK, result.output)
        self.assertIn("reference_summary", result.output)
        self.assertIn("Six lineages among the entries.", result.output)
        self.assertIn("nothing was written", result.output)
        transcript = _json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(transcript["tool_calls_made"], 1)
        self.assertEqual(transcript["question"], "how many lineages?")


# ---------------------------------------------------------------------------
# the package surface
# ---------------------------------------------------------------------------

class PackageSurfaceTests(unittest.TestCase):
    """``import eagent`` must stay cheap and must not guess its own version."""

    def _child(self, code: str) -> str:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC_ROOT)
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env=env, cwd=str(REPO_ROOT), timeout=120)
        self.assertEqual(completed.returncode, 0,
                         completed.stdout + completed.stderr)
        return completed.stdout.strip()

    def test_importing_the_package_pulls_in_no_submodule(self) -> None:
        loaded = self._child(
            "import sys, eagent; "
            "print([m for m in sys.modules if m.startswith('eagent.')])")
        self.assertEqual(loaded, "[]",
                         "importing eagent dragged in submodules; the console "
                         "script and anything reading __version__ pay for that")

    def test_importing_the_package_pulls_in_no_pydantic(self) -> None:
        loaded = self._child("import sys, eagent; "
                             "print('pydantic' in sys.modules)")
        self.assertEqual(loaded, "False")

    def test_the_version_matches_the_declared_one(self) -> None:
        import tomllib

        import eagent
        declared = tomllib.loads(
            (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["version"]
        self.assertIsInstance(eagent.__version__, str)
        self.assertEqual(eagent.__version__, declared)

    def test_public_names_resolve_lazily(self) -> None:
        import eagent

        self.assertIs(eagent.ToolResult,
                      __import__("eagent.envelope", fromlist=["ToolResult"]
                                 ).ToolResult)
        self.assertIs(eagent.assemble_bundle,
                      __import__("eagent.deliverables.bundle",
                                 fromlist=["assemble_bundle"]).assemble_bundle)
        self.assertIn("TaskSpec", dir(eagent))

    def test_an_unknown_attribute_names_what_is_exported(self) -> None:
        import eagent

        with self.assertRaises(AttributeError) as caught:
            eagent.definitely_not_exported
        self.assertIn("exported names", str(caught.exception))


if __name__ == "__main__":  # pragma: no cover - standalone runner
    unittest.main(verbosity=2)
