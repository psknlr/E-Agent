"""What the pipeline needs from a person, where each of its steps can run, and a run of it.

Three agents, and the table behind two of them.

WHY THESE ARE AGENTS AT ALL
===========================
A model asked "what do I still have to decide?" or "can this run in my browser?"
will answer fluently from nothing. These agents are how it reads the project's
own answer instead: the real validator's list of unresolved fields, the real
registry's ten interfaces with the status the roadmap records for each, and --
for ``run_pipeline`` -- what the real controller does when it is given a task.

WHAT ``run_pipeline`` DOES, AND WHAT NO AGENT HERE DOES
=======================================================
``run_pipeline`` drives the project's own deterministic controller, in a
throwaway directory, with the network off and no approvals on record. The
controller runs each interface that can run and stops at the first closed human
decision point, which on any task that has not been through ``eagent approve``
is the first one. That is the controller doing exactly what it does on the
command line; the browser is only where it happens.

What nothing in this package does is decide. No agent grants or denies an
approval request, resolves a task field or records an assumption: the three
decision points are cleared only by a named person through ``eagent approve``,
and a flag in a YAML file is not an approval. The controller *requests* an
approval and waits; nothing reachable from a model can answer it, by
construction rather than by instruction. A model may read where a task is
blocked and watch the real steps run; deciding is a person's.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from ..context import ExecutionPolicy, RunContext
from ..envelope import Severity
from ..harness.approval import APPROVAL_GATES, DECISION_POINTS, ApprovalQueue
from ..harness.controller import ResearchController
from ..harness.registry import PROTOCOL_ORDER, build_interface_registry
from ..harness.templates import TemplateLibrary
from ..provenance import RunManifest
from ..schemas import TaskSpec
from .core import MAX_RESULT_BYTES, AgentRefusal, AgentSpec, canonical, register, text_arg

#: Where each of the ten interfaces can run when the interpreter is a browser's.
#:
#: ``runs``           -- pure computation; nothing outside the interpreter is needed.
#: ``partial``        -- some paths run here; the rest need something a browser lacks.
#: ``needs_program``  -- an external program is required and none can run in a
#:                       browser. The step reports ``tool_unavailable`` and
#:                       produces nothing: it never substitutes a heuristic that
#:                       imitates the program.
#:
#: Written down rather than derived, because it is a statement about the world
#: (which programs exist for WebAssembly) and not about the code. It mirrors the
#: roadmap's component table, and a test pins it to the registry's ten names.
BROWSER_STATUS: Mapping[str, tuple[str, str, str]] = {
    "normalize_reaction": (
        "runs", "",
        "Runs fully offline. A structure must be confirmed by a person; a prose name is a "
        "blocker, not a guess."),
    "retrieve_evidence": (
        "partial", "network access to public databases",
        "Cache-first. The engine makes no network calls, so it reads cached evidence only; "
        "with an empty cache it reports its query gaps and no records."),
    "mine_sequences": (
        "needs_program", "blastp, mmseqs2 or hmmsearch",
        "Reports tool_unavailable and produces nothing rather than a heuristic that imitates "
        "the search."),
    "annotate_family": (
        "runs", "",
        "Needs seed accessions and references that sequence mining supplies."),
    "prepare_structures": (
        "needs_program", "a structure predictor and a local structure index",
        "Reports tool_unavailable; no structure is invented."),
    "model_complexes": (
        "needs_program", "a docking program or a joint complex predictor",
        "Reports tool_unavailable; a failed or absent model is never recorded as an "
        "experimental negative."),
    "evaluate_catalysis": (
        "runs", "",
        "Reads poses it is given. Every verdict is caveated, because the template windows "
        "are uncalibrated."),
    "select_batch": (
        "runs", "",
        "Pure selection. Opens no socket, and submitting an order is refused outright."),
    "ingest_results": (
        "runs", "",
        "Reads plate results into typed records. It has never read a real plate."),
    "propose_mutations": (
        "partial", "LigandMPNN (inverse folding)",
        "The rational and literature-precedent paths run here; the inverse-folding path "
        "needs the program."),
}

_STATUSES = ("runs", "partial", "needs_program")

#: Shipped task files, by id. Looked up in the project tree, never by a path a
#: model supplies: the id must match the pattern below and the file must sit
#: directly in this directory.
_TASKS_DIR = Path(__file__).resolve().parents[3] / "configs" / "tasks"
_MAX_TASK_YAML = 6000

#: PyYAML's scanner takes time quadratic in how deeply flow collections ("[" and
#: "{") nest: depth 60 costs 3 ms, depth 500 about 0.1 s and depth 3,000 almost 6 s,
#: and 6,000 characters can nest 6,000 deep. How many brackets there are does not
#: matter (a flow list of 2,000 items is 40 ms, and a SMILES string is full of
#: brackets that close at once), so this bounds depth. A task file nests two or
#: three deep.
_MAX_FLOW_DEPTH = 64


def _flow_depth(text: str) -> int:
    """The deepest nesting of "[" and "{" in ``text``, counting every bracket."""
    depth = deepest = 0
    for char in text:
        if char in "[{":
            depth += 1
            deepest = max(deepest, depth)
        elif char in "]}" and depth:
            depth -= 1
    return deepest


def shipped_tasks() -> list[str]:
    """Ids of the task files this installation ships, in sorted order."""
    return sorted(path.stem for path in _TASKS_DIR.glob("*.yaml")) if _TASKS_DIR.is_dir() else []


def pipeline_table() -> list[dict[str, Any]]:
    """The ten interfaces in protocol order, with where each can run."""
    registry = build_interface_registry()
    rows: list[dict[str, Any]] = []
    for order, name in enumerate(PROTOCOL_ORDER, start=1):
        interface = registry.get(name)
        status, needs, note = BROWSER_STATUS[name]
        rows.append({
            "order": order,
            "interface": name,
            "runs_in_browser": status,
            "needs": needs,
            "note": note,
            "description": interface.description,
            "depends_on": list(interface.depends_on),
            "required_fields": list(interface.required_fields),
            "required_approvals": list(interface.required_approvals),
        })
    return rows


# ---------------------------------------------------------------------------
# pipeline_catalog
# ---------------------------------------------------------------------------

def _pipeline_catalog(args: dict[str, str]) -> dict[str, Any]:
    rows = pipeline_table()
    counts = {status: sum(1 for row in rows if row["runs_in_browser"] == status)
              for status in _STATUSES}
    return {
        "n_interfaces": len(rows),
        "n_run_in_browser": counts["runs"],
        "n_partial": counts["partial"],
        "n_need_a_program": counts["needs_program"],
        "interfaces": [{k: row[k] for k in ("order", "interface", "runs_in_browser", "needs",
                                            "note", "depends_on", "required_approvals")}
                       for row in rows],
        "statement": ("Where each of the ten interfaces can run, from the project's own status "
                      "table. An interface that needs a program fails loudly when the program "
                      "is absent and nothing substitutes for it. This report is read-only."),
    }


register(AgentSpec(
    name="pipeline_catalog",
    description=("The ten pipeline interfaces in protocol order, with which can run in this "
                 "browser, which only partly, and which need an external program that a "
                 "browser cannot run. Read-only."),
    parameters={},
    run=_pipeline_catalog,
    wraps="eagent.harness.registry (interface registry) and the roadmap's status table",
))


# ---------------------------------------------------------------------------
# task_gate_report
# ---------------------------------------------------------------------------

def _is_unset(task: TaskSpec, dotted: str) -> bool:
    """Whether a dotted field path is null or empty on the task."""
    current: Any = task
    for part in dotted.split("."):
        current = current.get(part) if isinstance(current, dict) else getattr(current, part, None)
        if current is None:
            return True
    return isinstance(current, (list, tuple, dict)) and len(current) == 0


def _load_task(args: dict[str, str]) -> TaskSpec:
    shipped = text_arg(args, "task", required=False)
    inline = text_arg(args, "task_yaml", required=False)
    if bool(shipped) == bool(inline):
        raise AgentRefusal("give exactly one of task (a shipped task id) or task_yaml "
                           "(the task file's text)")
    if shipped:
        name = shipped.strip()
        known = shipped_tasks()
        if name not in known:
            raise AgentRefusal(f"no shipped task {name!r}; the shipped tasks are "
                               f"{', '.join(known) or '(none in this installation)'}")
        text = (_TASKS_DIR / f"{name}.yaml").read_text(encoding="utf-8")
    else:
        text = inline
        if len(text) > _MAX_TASK_YAML:
            raise AgentRefusal(f"task_yaml has {len(text)} characters; the limit is {_MAX_TASK_YAML}")
    depth = _flow_depth(text)
    if depth > _MAX_FLOW_DEPTH:
        raise AgentRefusal(f"the task nests '[' and '{{' {depth} deep; write it in block style "
                           f"(the limit is {_MAX_FLOW_DEPTH})")
    try:
        # Anchors and aliases are how a small document becomes a very large one; a
        # task file has no use for them, so they are refused before anything expands.
        for event in yaml.parse(text, Loader=yaml.SafeLoader):
            if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None) is not None:
                raise AgentRefusal("YAML anchors and aliases are not accepted in a task file")
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise AgentRefusal(f"the task is not valid YAML: {str(exc).splitlines()[0][:160]}") from None
    except RecursionError:
        raise AgentRefusal("the task is nested too deeply to read") from None
    if not isinstance(document, Mapping):
        raise AgentRefusal("a task file holds exactly one task, as a mapping of fields")
    try:
        return TaskSpec(**dict(document))
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                             for e in exc.errors()[:8])
        raise AgentRefusal(f"the task does not validate as a TaskSpec: {problems}") from None


def _task_gate_report(args: dict[str, str]) -> dict[str, Any]:
    task = _load_task(args)
    gates: list[dict[str, Any]] = []
    for gate in APPROVAL_GATES:
        unresolved = task.unresolved_for(gate)
        point = DECISION_POINTS.get(gate)
        gates.append({
            "gate": gate,
            "question": point.question if point else "",
            "required_fields": list(task.gate_requirements(gate)),
            "unresolved_fields": unresolved,
            "n_unresolved": len(unresolved),
            "flag_in_file": bool(task.approval.state(gate)),
        })
    steps: list[dict[str, Any]] = []
    for row in pipeline_table():
        blocked = [field for field in row["required_fields"] if _is_unset(task, field)]
        steps.append({"interface": row["interface"], "runs_in_browser": row["runs_in_browser"],
                      "unresolved_required_fields": blocked,
                      "required_approvals": row["required_approvals"]})
    return {
        "task_id": task.task_id,
        "task_mode": task.task_mode.value,
        "reaction_class": task.reaction.reaction_class.value,
        "n_gates": len(gates),
        "n_gates_with_unresolved_fields": sum(1 for gate in gates if gate["n_unresolved"]),
        "n_unresolved_fields_total": sum(gate["n_unresolved"] for gate in gates),
        "gates": gates,
        "n_steps_blocked_by_unresolved_fields": sum(1 for step in steps
                                                     if step["unresolved_required_fields"]),
        "steps": steps,
        "n_assumptions": len(task.assumptions),
        "assumptions": [{"field_path": item.field_path, "source": item.source}
                        for item in task.assumptions[:20]],
        "statement": ("A read-only report from the project's own validator. It resolves no "
                      "field and clears no gate: a flag in a file is not an approval, and only "
                      "a named person can approve."),
    }


register(AgentSpec(
    name="task_gate_report",
    description=("Read-only: which fields of a task are still unresolved for each of the three "
                 "human decision points, and which pipeline steps they block. Uses the "
                 "project's own validator. It cannot resolve a field or approve anything."),
    parameters={
        "task": "optional: the id of a shipped task, for example KRED_PILOT_001",
        "task_yaml": "optional: the text of a task file (instead of task), up to 6000 characters",
    },
    max_chars={"task_yaml": _MAX_TASK_YAML},
    run=_task_gate_report,
    wraps="eagent.schemas.TaskSpec.unresolved_for and the interface registry",
))


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

#: How a report is shortened when it would not fit in a result: (steps kept, items
#: kept per list, characters kept per text). The counts at the top of the report are
#: taken before any of this, so they stay exact when the detail is trimmed.
_TRIM = ((6, 8, 400), (5, 6, 200), (4, 4, 120), (3, 3, 70), (2, 2, 40))

#: Head-room under the engine's result bound for the envelope around the value.
_REPORT_BUDGET = MAX_RESULT_BYTES - 600


def _value(item: Any) -> Any:
    """An enum's value, or the thing itself: the report never holds an enum."""
    return getattr(item, "value", item)


def _pipeline_report(task: TaskSpec, controller: Any, queue: Any, trim: tuple[int, int, int]) -> dict[str, Any]:
    steps_cap, items_cap, chars_cap = trim
    trimmed = False

    def clip(text: Any, limit: int = chars_cap) -> str:
        nonlocal trimmed
        flat = " ".join(str(text or "").split())
        if len(flat) <= limit:
            return flat
        trimmed = True
        return flat[: limit - 1].rstrip() + "\u2026"

    def first(items: list[Any]) -> list[Any]:
        nonlocal trimmed
        if len(items) > items_cap:
            trimmed = True
        return items[:items_cap]

    attempts = [attempt for attempt in controller.attempts if not attempt.skipped]
    counts = {"success": 0, "partial": 0, "failed": 0}
    totals = {"blockers": 0, "warnings": 0, "uncertainties": 0}
    steps: list[dict[str, Any]] = []
    for attempt in attempts:
        status = str(_value(attempt.status))
        counts[status] = counts.get(status, 0) + 1
        result = controller.results.get(attempt.interface)
        blockers = [flag for flag in (result.qc_flags if result else []) if flag.severity is Severity.BLOCKER]
        warnings = [flag for flag in (result.qc_flags if result else []) if flag.severity is Severity.WARN]
        uncertainties = list(result.uncertainty) if result else []
        totals["blockers"] += len(blockers)
        totals["warnings"] += len(warnings)
        totals["uncertainties"] += len(uncertainties)
        steps.append({
            "stage": str(_value(attempt.stage)),
            "interface": attempt.interface,
            "status": status,
            "attempt": attempt.attempt,
            "failure": str(_value(attempt.failure)),
            "message": clip(attempt.message),
            "n_blockers": len(blockers),
            "n_warnings": len(warnings),
            "n_uncertainties": len(uncertainties),
            "blockers": [{"code": flag.code, "subject": flag.subject or "", "message": clip(flag.message)}
                         for flag in first(blockers)],
            "warnings": [{"code": flag.code, "subject": flag.subject or "", "message": clip(flag.message)}
                         for flag in first(warnings)],
            "uncertainties": [{"code": item.code, "question": clip(item.question),
                               "resolvable_by": clip(item.resolvable_by, 80)}
                              for item in first(uncertainties)],
            "next_actions": [{"who": "a person" if action.requires_human else "the harness",
                              "action": action.action, "rationale": clip(action.rationale)}
                             for action in first(list(result.next_actions) if result else [])],
            "artifacts": [{"key": item.key, "kind": item.kind} for item in first(list(result.artifacts) if result else [])],
        })
    if len(steps) > steps_cap:
        trimmed = True

    waiting: dict[str, dict[str, str]] = {}
    for request in queue.pending():
        if request.gate not in waiting:
            point = DECISION_POINTS.get(request.gate)
            waiting[request.gate] = {"gate": request.gate, "question": clip(point.question if point else ""),
                                     "detail": clip(request.detail)}
    stage = controller.stage
    outcome = controller.outcome
    return {
        "task_id": task.task_id,
        "task_mode": task.task_mode.value,
        "outcome": str(_value(outcome)) if outcome else str(_value(stage)),
        "stage": str(_value(stage)),
        "reason": clip(controller.stop_reason, 400),
        "path": [str(_value(item)) for item in controller.path][:12],
        "n_steps_run": len(attempts),
        "n_steps_success": counts["success"],
        "n_steps_partial": counts["partial"],
        "n_steps_failed": counts["failed"],
        "n_blockers": totals["blockers"],
        "n_warnings": totals["warnings"],
        "n_uncertainties": totals["uncertainties"],
        "n_gates_waiting": len(waiting),
        "steps": steps[:steps_cap],
        "waiting_on": list(waiting.values())[:items_cap],
        "escalations": [{"interface": item.interface, "kind": str(_value(item.kind)), "reason": clip(item.reason)}
                        for item in first(list(controller.escalations))],
        "network": "disabled",
        "trimmed": trimmed,
        "statement": ("The project's own controller, run here in a throwaway directory with the network "
                      "off and no approvals on record. It ran what could run and stopped where it must. "
                      "It cannot clear a human decision point or fill a field: only a named person can, "
                      "with eagent approve."),
    }


def _run_pipeline(args: dict[str, str]) -> dict[str, Any]:
    task = _load_task(args)
    with tempfile.TemporaryDirectory(prefix="eagent-engine-") as scratch:
        run_dir = Path(scratch)
        manifest = RunManifest(run_id=f"{task.task_id}-engine", task_id=task.task_id, global_seed=0)
        policy = ExecutionPolicy(allow_network=False, dry_run=False, max_retries=1)
        context = RunContext(task=task, workdir=run_dir, manifest=manifest, policy=policy,
                             templates=TemplateLibrary.load(None), logger=lambda message: None)
        queue = ApprovalQueue.load(run_dir / "approvals.json", manifest)
        controller = ResearchController(context, build_interface_registry(), queue, hooks=None, arguments={})
        controller.run()
        return _fit_report(task, controller, queue)


def _fit_report(task: TaskSpec, controller: Any, queue: Any) -> dict[str, Any]:
    """The fullest report that fits a result, trimmed one step at a time if it must be."""
    for trim in _TRIM:
        report = _pipeline_report(task, controller, queue, trim)
        if len(canonical(report).encode("utf-8")) <= _REPORT_BUDGET:
            return report
    raise AgentRefusal("the run's report is too large to return whole, even trimmed")


register(AgentSpec(
    name="run_pipeline",
    description=("Run the project's own deterministic controller on a task, on this device with the "
                 "network off and no approvals: it runs each interface that can run here (the reaction "
                 "step first) until the first human decision point, then reports what each step found "
                 "(status, blocking problems, open questions) and which decision it is waiting on. It "
                 "cannot clear that decision or fill a field; only a named person can."),
    parameters={
        "task": "optional: the id of a shipped task, for example KRED_PILOT_001",
        "task_yaml": "optional: the text of a task file (instead of task), up to 6000 characters",
    },
    max_chars={"task_yaml": _MAX_TASK_YAML},
    run=_run_pipeline,
    wraps="eagent.harness.controller.ResearchController and the interfaces it runs",
))
