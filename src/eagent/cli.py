"""The terminal interface: plain text, no colour, no invented numbers.

Three rules shape everything below, and each one is here because of a way a
research CLI quietly lies to the person reading it.

*No colour, no progress widgets, no rich.* The output of a run of this tool
ends up pasted into a lab notebook, an email and a ticket. ANSI escapes
survive none of those legibly, a spinner leaves nothing behind at all, and a
table drawn with box characters reflows into noise at eighty columns. So the
output is ASCII that means the same thing in a terminal, in a pipe and in a
text file, and ``rich`` is not imported -- it is not installed, and a CLI
whose readability depends on an optional dependency is a CLI that is
unreadable exactly when something has gone wrong.

*A number printed here came from a file.* Costs, counts, identities and
distances are read from the manifest or from an artifact and printed with
their source, or they are printed as ``unknown`` with what is missing. There
is no place in this module where a value is estimated, averaged into
existence or defaulted to zero. A cost ceiling with nothing recorded against
it prints "nothing recorded", not "0.0" -- those are different facts.

*A failure names the next action and the human who has to take it.* "Step
failed" is not an error message. Every stop prints the typed reason from
:mod:`eagent.errors` or the QC code from the envelope, then the concrete next
action, and when a human decision point is what is blocking, the gate's name
and the question that gate asks.

Imports are deferred into the command bodies on purpose: ``eagent --help``
and ``eagent templates list`` must not pay for pydantic, the structure parser
and the ten interface modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import click

__all__ = [
    "EXIT_BLOCKED",
    "EXIT_FAILED",
    "EXIT_OK",
    "EXIT_UNRESOLVED",
    "EXIT_USAGE",
    "Refusal",
    "main",
]

# ---------------------------------------------------------------------------
# exit codes -- a script wrapping this tool must be able to tell the four
# outcomes apart without parsing prose.
# ---------------------------------------------------------------------------

#: Everything asked for was done.
EXIT_OK = 0
#: Bad invocation. Click's own convention; kept so ``eagent run`` with a typo
#: does not look like a scientific failure.
EXIT_USAGE = 2
#: A required field, template or input is unresolved. Nothing ran, and
#: nothing was guessed.
EXIT_UNRESOLVED = 3
#: A human decision point is blocking. The run is intact and resumable.
EXIT_BLOCKED = 4
#: The work ran and failed, or a check found a problem.
EXIT_FAILED = 5

#: Printed wherever a value is genuinely not known. Never substituted with a
#: plausible default, and deliberately not an empty string, which reads as
#: "zero" in a column.
UNKNOWN = "unknown"


class Refusal(click.ClickException):
    """A typed reason the command stopped, with the next concrete action.

    Subclasses :class:`click.ClickException` so click's own machinery prints
    it and sets the exit status, and overrides ``show`` because the default
    one-line ``Error: ...`` has nowhere to put the next action or the name of
    the gate a person has to clear.
    """

    def __init__(self, reason: str, *, next_action: str = "",
                 gate: str | None = None, exit_code: int = EXIT_FAILED) -> None:
        super().__init__(reason)
        self.exit_code = exit_code
        self.next_action = next_action
        self.gate = gate

    def show(self, file: Any = None) -> None:       # noqa: D102 - click's hook
        click.echo(f"error: {self.format_message()}", err=True)
        if self.gate:
            click.echo(f"blocked on a human decision: {self.gate}", err=True)
        if self.next_action:
            click.echo(f"next: {self.next_action}", err=True)


# ---------------------------------------------------------------------------
# plain-text output
# ---------------------------------------------------------------------------

class Out:
    """Plain-text writer. No colour, no cursor control, no optional deps."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet

    def line(self, text: str = "") -> None:
        if not self.quiet:
            click.echo(text)

    def heading(self, text: str) -> None:
        self.line("")
        self.line(text)
        self.line("-" * len(text))

    def kv(self, key: str, value: Any, width: int = 24) -> None:
        self.line(f"{key + ':':<{width}} {fmt(value)}")

    def bullet(self, text: str, indent: int = 2) -> None:
        self.line(" " * indent + "- " + text)

    def note(self, text: str, indent: int = 2) -> None:
        self.line(" " * indent + text)

    def warn(self, text: str) -> None:
        click.echo(f"warning: {text}", err=True)

    def error(self, text: str) -> None:
        click.echo(f"error: {text}", err=True)


def fmt(value: Any) -> str:
    """Render a value for a terminal without ever inventing one.

    ``None`` becomes ``unknown`` rather than ``0``, ``-`` or an empty cell,
    because a reader scanning a column of numbers will read any of those as a
    measurement.
    """
    if value is None:
        return UNKNOWN
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) if value else "none"
    if isinstance(value, Mapping):
        return ", ".join(f"{k}={v}" for k, v in sorted(value.items())) \
            if value else "none"
    text = str(value)
    return text if text.strip() else UNKNOWN


def _version() -> str:
    from . import __version__
    return __version__


# ---------------------------------------------------------------------------
# shared loading helpers
# ---------------------------------------------------------------------------

def _read_yaml(path: Path, what: str) -> Any:
    import yaml
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Refusal(f"{what} {path} could not be read: {exc}",
                      next_action="check the path and the file permissions",
                      exit_code=EXIT_UNRESOLVED) from exc
    except yaml.YAMLError as exc:
        raise Refusal(f"{what} {path} is not valid YAML: {exc}",
                      next_action="fix the YAML syntax and run this again",
                      exit_code=EXIT_UNRESOLVED) from exc


def _load_task(path: Path):
    """Load a task file into a :class:`~eagent.schemas.TaskSpec`.

    A validation error is reported as a refusal rather than a traceback: the
    person running this is a chemist with a YAML file, and the fields pydantic
    names are the fields they have to fix.
    """
    from pydantic import ValidationError
    from .schemas import TaskSpec

    document = _read_yaml(path, "task file")
    if not isinstance(document, Mapping):
        raise Refusal(f"{path} did not parse into a mapping; a task file holds "
                      f"exactly one task",
                      next_action="see configs/tasks/KRED_PILOT_001.yaml for the shape",
                      exit_code=EXIT_UNRESOLVED)
    try:
        return TaskSpec(**dict(document))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in exc.errors())
        raise Refusal(f"{path} does not validate as a TaskSpec -- {problems}",
                      next_action=("correct the named fields; a field that is "
                                   "undecided must be null, not a placeholder"),
                      exit_code=EXIT_UNRESOLVED) from exc


def _load_run_manifest(run_dir: Path, *, required: bool = True):
    from .deliverables.bundle import RUN_MANIFEST_NAME
    from .provenance import RunManifest

    path = run_dir / RUN_MANIFEST_NAME
    if not path.exists():
        if not required:
            return None
        raise Refusal(
            f"no {RUN_MANIFEST_NAME} in {run_dir}; without it this run cannot "
            f"be described, because every count and cost would have to be "
            f"guessed at",
            next_action=f"run `eagent run <task.yaml> --rundir {run_dir}` first",
            exit_code=EXIT_UNRESOLVED)
    try:
        return RunManifest.load(path)
    except Exception as exc:
        raise Refusal(f"{path} could not be read ({type(exc).__name__}: {exc})",
                      next_action="restore the manifest from the run's backup",
                      exit_code=EXIT_FAILED) from exc


def _load_templates(root: Path | None):
    from .errors import TemplateError
    from .harness.templates import TemplateLibrary

    try:
        return TemplateLibrary.load(root)
    except TemplateError as exc:
        raise Refusal(f"the template library did not load: {exc}",
                      next_action=("fix or restore the named template file; "
                                   "every threshold in a run comes from these, "
                                   "and none of them has a defensible default"),
                      exit_code=EXIT_UNRESOLVED) from exc


def _parse_stage(value: str | None):
    """Turn a ``--step``/``--from``/``--until`` name into a Stage.

    Validated here instead of with :class:`click.Choice` so that ``--help``
    does not have to import the controller, and so the error can say which
    stages actually execute something.
    """
    if value is None:
        return None
    from .harness.controller import STAGE_INTERFACE, Stage

    try:
        stage = Stage(value)
    except ValueError as exc:
        executable = ", ".join(sorted(s.value for s in STAGE_INTERFACE))
        others = ", ".join(sorted(s.value for s in Stage
                                  if s not in STAGE_INTERFACE))
        raise Refusal(f"'{value}' is not a stage of this protocol",
                      next_action=(f"stages that execute a step: {executable}. "
                                   f"Decision stages: {others}"),
                      exit_code=EXIT_USAGE) from exc
    return stage


def _gate_question(gate: str) -> str:
    from .harness.approval import DECISION_POINTS
    point = DECISION_POINTS.get(gate)
    return point.question if point else ""


# ---------------------------------------------------------------------------
# the group
# ---------------------------------------------------------------------------

_HELP = """\
Enzyme function mining and substrate-directed engineering.

A run walks a declared protocol: confirm the reaction, retrieve evidence,
mine sequences, locate the family, prepare structures, model complexes,
evaluate catalysis, select a batch behind a human approval, ingest results
and propose variants.

Three decisions are a human's and are never made here: what the reaction is,
whether to spend the synthesis budget, and what counts as a hit. Unknown
values are reported as unknown; nothing in this tool fills one in.

Exit codes: 0 done, 2 bad invocation, 3 something is unresolved, 4 a human
decision point is blocking, 5 the work ran and failed.
"""


@click.group(help=_HELP, context_settings={
    "help_option_names": ["-h", "--help"], "max_content_width": 96})
@click.version_option(version=_version(), prog_name="eagent")
def main() -> None:
    """Entry point registered as the ``eagent`` console script."""


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

_TASK_SCAFFOLD = '''\
# ===========================================================================
# {task_id} -- scaffolded by `eagent init`.
#
# EVERY null BELOW IS DELIBERATE. A null means "nobody has decided yet", and
# no step that depends on it may run. It is not a gap for a language model to
# close with a reasonable default:
#
#   * the substrate and the product ARE the task. A prose name is not a
#     structure -- it loses stereochemistry, salt form and tautomer -- and a
#     guessed structure silently redefines the project, so every sequence
#     mined afterwards would be evidence about a different molecule;
#   * the target configuration is a commercial decision. Guessing it inverts
#     the objective: the best variant for one enantiomer is the worst
#     possible choice for the other;
#   * whether a new stereocentre is created at all is a per-substrate fact.
#     An aldehyde or a symmetric ketone gives an achiral alcohol, and then no
#     ee may be reported;
#   * pH, temperature, solvent, cosolvent and host are part of the record KEY,
#     not metadata. Two measurements under different conditions are two
#     different facts, and inventing conditions merges records that are not
#     comparable.
#
# A null is filled through TaskSpec.resolve(), which demands an authority --
# operator:<name>, literature:<PMID/DOI>, database:<name>@<version>,
# template:<id> or experiment:<id> -- and records an Assumption below. A
# model's own guess is not an authority and is refused.
#
# Check what is still outstanding with:
#     eagent validate {filename} --all-gates
# ===========================================================================

task_id: {task_id}
task_mode: {task_mode}

reaction:
  reaction_class: {reaction_class}

  # Writable only once both structures exist: the mapping is what ties the
  # reactive atoms to the geometry checks. Required by reaction_spec_confirmed.
  atom_mapped_reaction_smiles: null

  substrate:
    name: null
    isomeric_smiles: null      # the operator supplies this; required by the gate
    molfile: null
    inchikey: null
    reactive_atoms:
      electrophile: null       # atom-map ids are meaningless before the mapping
      nucleophile: null
      leaving_group: null
      stabilised_atoms: []
      prochiral_center: null
    is_prochiral: null         # null, NOT false: false switches off chiral analysis
    chemical_purity: null
    supplier_or_synthesis: null
    notes: ""

  product:
    name: null
    isomeric_smiles: null      # what the assay must actually detect
    inchikey: null
    target_stereochemistry: unspecified
    creates_new_stereocenter: null   # null, NOT true; drives the ee criterion
    authentic_standard_available: null   # a procurement fact, not a chemical one
    notes: ""

  rhea_id: null                # an invented identifier would be used as a join key
  ec_hint: null                # a search hint only; it must widen, never narrow
  notes: ""

conditions:
  cofactor_options: []         # empty, not guessed: this decides the controls too
  pH: null
  temperature_C: null
  solvent_system: null
  cosolvent_fraction: null
  buffer: null
  expression_host: null
  substrate_concentration_mM: null
  enzyme_loading: null
  reaction_time_h: null

budget:
  # Resource plan for round 1. These are NOT predicted pass rates: nothing
  # here asserts how many sequences will survive a stage, only how the stage
  # is sized. Change them to match the site.
  initial_sequence_target: 2000
  family_qc_pool_target: 600
  structure_pool_target: 600
  detailed_complex_target: 300
  new_constructs_round_1: 96
  # If the construct count is a hard synthesis cap, set this true and reserve
  # slots: a positive control that needs a new gene comes out of the same cap,
  # and discovering that after the order is placed costs a round.
  constructs_include_controls: false
  reserved_control_slots: 0

objectives:
  primary: experimentally_confirmed_target_product
  # Separate named axes, never combined into one total score.
  secondary:
    - target_enantiomer_selectivity
    - activity
    - soluble_expression

approval:
  # A flag here is not an approval. The gates are cleared by `eagent approve`,
  # which records a named actor and a timestamp in the run manifest.
  reaction_spec_confirmed: false
  synthesis_authorized: false
  functional_criteria_confirmed: false

parent_enzymes: {parent_enzymes}

# Every future entry carries the authority that supplied the value.
assumptions: []
'''


@main.command("init")
@click.argument("task_id")
@click.option("--out", "-o", "out_path", type=click.Path(path_type=Path),
              default=None, help="Where to write the task file. "
                                 "Default: <task_id>.yaml in the current directory.")
@click.option("--mode", "task_mode", default="enzyme_mining",
              help="enzyme_mining | reaction_space_exploration | "
                   "substrate_directed_engineering.")
@click.option("--reaction-class", "reaction_class", default="other",
              help="The transformation, if it is already decided. Defaults to "
                   "'other', which is an explicit 'not classified yet'.")
@click.option("--parent", "parents", multiple=True,
              help="Parent enzyme accession or sequence hash. Required by "
                   "substrate_directed_engineering, repeatable.")
@click.option("--force", is_flag=True, help="Overwrite an existing file.")
def init_command(task_id: str, out_path: Path | None, task_mode: str,
                 reaction_class: str, parents: tuple[str, ...],
                 force: bool) -> None:
    """Scaffold a task YAML with its null fields intact.

    The scaffold is deliberately unfinished. Writing a plausible substrate
    into it would produce a file that validates, runs, and answers a question
    nobody asked.
    """
    from .schemas import ReactionClass, TaskMode

    out = Out()
    try:
        mode = TaskMode(task_mode)
    except ValueError as exc:
        raise Refusal(f"'{task_mode}' is not a task mode",
                      next_action="one of: "
                                  + ", ".join(m.value for m in TaskMode),
                      exit_code=EXIT_USAGE) from exc
    try:
        rxn = ReactionClass(reaction_class)
    except ValueError as exc:
        raise Refusal(f"'{reaction_class}' is not a known reaction class",
                      next_action="one of: "
                                  + ", ".join(c.value for c in ReactionClass),
                      exit_code=EXIT_USAGE) from exc

    target = out_path if out_path is not None else Path(f"{task_id}.yaml")
    if target.exists() and not force:
        raise Refusal(f"{target} already exists",
                      next_action="pass --force to overwrite it, or choose "
                                  "another path with --out",
                      exit_code=EXIT_USAGE)

    parent_list = "[]" if not parents else \
        "\n" + "\n".join(f"  - {p}" for p in parents)
    text = _TASK_SCAFFOLD.format(
        task_id=task_id, filename=target.name, task_mode=mode.value,
        reaction_class=rxn.value, parent_enzymes=parent_list)

    # Validate before writing: a scaffold that does not load is worse than no
    # scaffold, because the person will edit it for an hour first.
    import yaml
    from pydantic import ValidationError
    from .schemas import TaskSpec
    try:
        TaskSpec(**yaml.safe_load(text))
    except ValidationError as exc:
        detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                           for e in exc.errors())
        raise Refusal(f"the scaffold for mode '{mode.value}' does not validate "
                      f"-- {detail}",
                      next_action=("substrate_directed_engineering needs at "
                                   "least one --parent; it is engineering *of* "
                                   "something"),
                      exit_code=EXIT_USAGE) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    out.line(f"wrote {target}")
    out.line("")
    out.line("Every field that is null is a decision waiting for an operator. "
             "Nothing in this file was inferred.")
    out.line(f"Next: fill the substrate and product structures, then run "
             f"`eagent validate {target} --all-gates`.")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@main.command("validate")
@click.argument("task_file", type=click.Path(exists=True, dir_okay=False,
                                             path_type=Path))
@click.option("--gate", "gates", multiple=True,
              help="Report on this gate and fail if it cannot be satisfied. "
                   "Repeatable.")
@click.option("--all-gates", is_flag=True,
              help="Treat all three gates as asked about, so any unresolved "
                   "field fails the command.")
def validate_command(task_file: Path, gates: tuple[str, ...],
                     all_gates: bool) -> None:
    """Report unresolved fields per gate.

    Exits non-zero when a gate you asked about cannot be satisfied. Without
    ``--gate`` or ``--all-gates`` this is a report and exits zero: listing
    what is outstanding is not the same as being asked whether a particular
    decision can be taken.
    """
    from .harness.approval import APPROVAL_GATES
    from .schemas import GATE_REQUIREMENTS

    out = Out()
    unknown_gates = [g for g in gates if g not in APPROVAL_GATES]
    if unknown_gates:
        raise Refusal(f"not an approval gate: {', '.join(unknown_gates)}",
                      next_action="the three gates are "
                                  + ", ".join(APPROVAL_GATES),
                      exit_code=EXIT_USAGE)

    task = _load_task(task_file)
    asked = set(APPROVAL_GATES) if all_gates else set(gates)

    out.line(f"task {task.task_id}  ({task.task_mode.value})")
    out.line(f"file {task_file}")
    out.line(f"reaction class: {task.reaction.reaction_class.value}")

    blocked: list[str] = []
    for gate in APPROVAL_GATES:
        required = GATE_REQUIREMENTS.get(gate, ())
        unresolved = task.unresolved_for(gate)
        out.heading(f"gate {gate}"
                    + ("  [asked about]" if gate in asked else ""))
        question = _gate_question(gate)
        if question:
            out.note(question)
        out.kv("  required fields", len(required), width=26)
        out.kv("  unresolved", len(unresolved), width=26)
        for path in unresolved:
            out.bullet(f"{path}: null -- an operator must supply this", indent=4)
        if not unresolved:
            out.note("  every field this gate depends on is resolved", indent=2)
        flag = task.approval.state(gate)
        out.kv("  flag in the task file", flag, width=26)
        out.note("  a flag in a file is not an approval: the gate is cleared "
                 "only by `eagent approve`, which records who decided and "
                 "when in the run manifest", indent=2)
        if gate in asked and unresolved:
            blocked.append(gate)

    out.heading("assumptions on record")
    if task.assumptions:
        for assumption in task.assumptions:
            out.bullet(f"{assumption.field_path} = {assumption.value!r} "
                       f"(source {assumption.source})")
    else:
        out.note("none: nothing in this task has been filled in, so nobody has "
                 "had to justify anything yet")

    out.line("")
    if not asked:
        out.line("No gate was asked about, so this is a report only. Pass "
                 "--gate <name> or --all-gates to have an unresolved field "
                 "fail the command.")
        return
    if blocked:
        detail = "; ".join(
            f"{gate} needs {', '.join(task.unresolved_for(gate))}"
            for gate in blocked)
        raise Refusal(
            f"{len(blocked)} gate(s) you asked about cannot be satisfied: {detail}",
            gate=blocked[0],
            next_action=("resolve each field through TaskSpec.resolve() with a "
                         "named authority -- operator:, literature:, database:, "
                         "template: or experiment: -- and validate again"),
            exit_code=EXIT_UNRESOLVED)
    out.line(f"every gate you asked about ({', '.join(sorted(asked))}) has its "
             f"fields resolved. The decision itself is still a human's.")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _argument_map(path: Path | None) -> dict[str, dict[str, Any]]:
    """Per-interface keyword arguments supplied by the operator.

    The harness deliberately has no built-in wiring from one step's output
    into the next: which pose set, which seed list and which template a step
    should consume are scientific choices, and a default invented here would
    be this tool making them silently. So the map is supplied or it is empty,
    and an empty one is reported rather than filled in.
    """
    if path is None:
        return {}
    document = _read_yaml(path, "arguments file") \
        if path.suffix in (".yaml", ".yml") else \
        json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise Refusal(f"{path} must map an interface name to its keyword "
                      f"arguments",
                      next_action='e.g. {"mine_sequences": {"seeds": [...]}}',
                      exit_code=EXIT_USAGE)
    out: dict[str, dict[str, Any]] = {}
    for key, value in document.items():
        if not isinstance(value, Mapping):
            raise Refusal(f"{path}: arguments for '{key}' must be a mapping",
                          exit_code=EXIT_USAGE)
        out[str(key)] = dict(value)
    return out


def _annotation_shape(annotation: Any) -> tuple[set[Any], bool]:
    """The pydantic models a parameter names, and whether it is structured.

    ``structured`` is the gate on coercion. A parameter typed ``float`` or
    ``str`` is left exactly as the operator wrote it -- the CLI has no
    business rewriting a number someone typed. A parameter typed against a
    pydantic model, a dataclass or an enum is a different matter: a YAML file
    can only ever carry a mapping, and the step is going to call attributes
    on it.
    """
    from dataclasses import is_dataclass
    from enum import Enum
    from typing import get_args

    from pydantic import BaseModel

    models: set[Any] = set()
    structured = False
    seen: set[int] = set()

    def walk(node: Any) -> None:
        nonlocal structured
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, type):
            if issubclass(node, BaseModel):
                models.add(node)
                structured = True
            elif is_dataclass(node) or issubclass(node, Enum):
                structured = True
        for arg in get_args(node):
            walk(arg)

    walk(annotation)
    return models, structured


def _accepts_many(annotation: Any) -> bool:
    """Whether the annotation has a container around its model anywhere."""
    from typing import get_args, get_origin

    for member in (get_args(annotation) or (annotation,)):
        if get_origin(member) is not None:
            return True
    return False


def _coerce_arguments(registry: Any, arguments: Mapping[str, Mapping[str, Any]],
                      source: Path | None, out: Out) -> dict[str, dict[str, Any]]:
    """Validate each ``--arguments`` payload against the model its step declares.

    Every interface's ``execute`` is typed against pydantic models, and a
    YAML or JSON file can only ever produce mappings and lists. Handing those
    straight to ``execute(**kwargs)`` puts a dict where a model is expected,
    and the mistake surfaces as an ``AttributeError`` from somewhere deep
    inside the step -- a bare traceback, which is the one thing this CLI must
    never print, and which names neither the file nor the field that was
    wrong.

    So the boundary is here. Candidate-shaped payloads go through
    :mod:`eagent.tools.handoff`, which is the agreed crossing for them, and
    everything else is validated against its own annotation. A payload that
    does not validate stops the run before any step executes, with the
    interface, the parameter and the validator's complaint, because a
    partially-filled model would be scored as though the missing fields were
    genuinely absent.
    """
    import inspect
    import typing

    from pydantic import TypeAdapter, ValidationError

    from .schemas.candidate import Candidate
    from .tools.handoff import HandoffError, as_candidate, as_candidates

    where = str(source) if source is not None else "--arguments"
    coerced: dict[str, dict[str, Any]] = {}
    for interface, kwargs in arguments.items():
        if interface not in registry:
            out.warn(f"{where}: '{interface}' is not one of the protocol's "
                     f"interfaces, so its arguments were never going to reach "
                     f"a step; the ten are "
                     f"{', '.join(sorted(registry.names()))}")
            coerced[interface] = dict(kwargs)
            continue
        execute = type(registry.get(interface)).execute
        try:
            hints = typing.get_type_hints(execute)
            parameters = inspect.signature(execute).parameters
        except Exception:                 # an un-resolvable annotation
            # Better to pass the payload through untouched than to refuse a
            # run because this module could not read a type hint.
            coerced[interface] = dict(kwargs)
            continue

        out_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            annotation = hints.get(key)
            if key not in parameters or annotation is None:
                out_kwargs[key] = value
                continue
            models, structured = _annotation_shape(annotation)
            if not structured:
                out_kwargs[key] = value
                continue
            label = f"{where}: {interface}.{key}"
            try:
                if Candidate in models:
                    out_kwargs[key] = (
                        as_candidates(value, source=label)
                        if _accepts_many(annotation) or not isinstance(
                            value, Mapping)
                        else as_candidate(value, source=label))
                else:
                    out_kwargs[key] = TypeAdapter(annotation).validate_python(
                        value)
            except HandoffError as exc:
                raise Refusal(
                    f"{label} is not a usable candidate payload: {exc}",
                    next_action=("fix the entry in the arguments file; a "
                                 "candidate that lost fields in transit would "
                                 "be scored as though those fields had never "
                                 "been measured"),
                    exit_code=EXIT_USAGE) from exc
            except ValidationError as exc:
                raise Refusal(
                    f"{label} does not validate as "
                    f"{_annotation_name(annotation)}: {exc}",
                    next_action=("correct the payload in the arguments file, "
                                 "or drop the key to let the step use its own "
                                 "default; nothing is filled in here"),
                    exit_code=EXIT_USAGE) from exc
            except Exception:
                # The annotation is real but pydantic cannot build a schema
                # for it -- a Protocol, a runner object. Nothing in a YAML
                # file can satisfy it anyway, so leave it for the step to
                # reject with its own message.
                out_kwargs[key] = value
        coerced[interface] = out_kwargs
    return coerced


def _annotation_name(annotation: Any) -> str:
    """A readable name for an annotation, for a refusal message."""
    return getattr(annotation, "__name__", None) or str(annotation).replace(
        "typing.", "")


def _report_attempts(controller: Any, out: Out, already: int) -> int:
    """Stream the attempts recorded since the last call."""
    from .envelope import Severity

    for attempt in controller.attempts[already:]:
        if attempt.skipped:
            out.line(f"  [{attempt.stage.value}] {attempt.interface}: skipped "
                     f"on resume (input hash unchanged)")
            continue
        out.line(f"  [{attempt.stage.value}] {attempt.interface}: "
                 f"{attempt.status} (attempt {attempt.attempt}"
                 + (f", {attempt.failure.value}"
                    if attempt.failure.value != "none" else "") + ")")
        if attempt.message:
            out.note(f"      {attempt.message}", indent=0)
        result = controller.results.get(attempt.interface)
        if result is None:
            continue
        for flag in result.qc_flags:
            marker = {Severity.BLOCKER: "BLOCKER", Severity.WARN: "warn",
                      Severity.INFO: "info"}[flag.severity]
            subject = f" [{flag.subject}]" if flag.subject else ""
            out.line(f"      qc {marker}: {flag.code}{subject} -- {flag.message}")
        for item in result.uncertainty:
            out.line(f"      uncertainty {item.code}: {item.question}"
                     + (f" (resolvable by {item.resolvable_by})"
                        if item.resolvable_by else ""))
        for action in result.next_actions:
            who = "human" if action.requires_human else "harness"
            out.line(f"      next ({who}): {action.action} -- {action.rationale}")
    return len(controller.attempts)


def _dry_run_plan(ctx: Any, registry: Any, templates: Any,
                  task: Any, policy: Any) -> dict[str, Any]:
    """What a run would do, and what it would cost, without doing any of it."""
    from .harness.approval import APPROVAL_GATES
    from .harness.registry import PROTOCOL_ORDER, registry_report

    steps = []
    for name in PROTOCOL_ORDER:
        iface = registry.get(name)
        steps.append({
            "interface": name,
            "version": iface.version,
            "description": iface.description,
            "required_fields": list(iface.required_fields),
            "unresolved_required_fields": [
                field for field in iface.required_fields
                if _unset(task, field)],
            "required_approvals": list(iface.required_approvals),
            "depends_on": list(iface.depends_on),
        })
    calibration = templates.calibration_report().to_dict() if templates else None
    return {
        "mode": "dry_run",
        "executed": False,
        "statement": ("no interface was executed, no external binary was "
                      "started and no network call was made"),
        "task_id": task.task_id,
        "policy": {"allow_network": policy.allow_network,
                   "allow_external_binaries": policy.allow_external_binaries,
                   "allow_gpu_models": policy.allow_gpu_models,
                   "dry_run": policy.dry_run, "strict": policy.strict,
                   "max_retries": policy.max_retries,
                   "cost_ceiling": dict(policy.cost_ceiling)},
        "steps": steps,
        "gates": {gate: task.unresolved_for(gate) for gate in APPROVAL_GATES},
        "budget": task.budget.model_dump(mode="json"),
        "cost": {
            "currency": None,
            "per_step": None,
            "estimated_total": None,
            "recorded_so_far": dict(ctx.manifest.cost_total) or None,
            "note": ("this harness has no price list: compute time, synthesis "
                     "and plate costs are site-specific. A total is left null "
                     "rather than filled with a plausible figure, because a "
                     "figure in a plan gets approved as though it were checked"),
        },
        "templates": {"calibration": calibration,
                      "integrity_problems": templates.integrity_problems()
                      if templates else []},
        "registry": registry_report(registry),
    }


def _unset(task: Any, dotted: str) -> bool:
    cur: Any = task
    for part in dotted.split("."):
        if cur is None:
            return True
        cur = getattr(cur, part, None)
    if cur is None:
        return True
    return isinstance(cur, (list, tuple, dict, str)) and len(cur) == 0


@main.command("run")
@click.argument("task_file", type=click.Path(exists=True, dir_okay=False,
                                             path_type=Path))
@click.option("--rundir", type=click.Path(path_type=Path), default=None,
              envvar="EAGENT_RUNDIR",
              help="Working directory for this run. Default: "
                   "runs/<task_id>. An existing run is resumed.")
@click.option("--step", "step", default=None,
              help="Execute exactly one stage and stop.")
@click.option("--from", "from_stage", default=None,
              help="Start at this stage instead of the beginning.")
@click.option("--until", "until_stage", default=None,
              help="Stop once this stage has been executed.")
@click.option("--dry-run", is_flag=True,
              help="Plan and cost the run. Executes no interface, starts no "
                   "external binary and makes no network call.")
@click.option("--offline", is_flag=True,
              help="Force ExecutionPolicy.allow_network false.")
@click.option("--allow-network", is_flag=True,
              help="Permit network access for steps that declare it. --offline "
                   "wins if both are given.")
@click.option("--arguments", "arguments_file",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None,
              help="YAML/JSON mapping of interface name to keyword arguments.")
@click.option("--templates", "template_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Template library root.")
@click.option("--max-retries", type=int, default=1, show_default=True,
              help="Upper bound on retries; the per-failure-kind policy still "
                   "applies and the tighter of the two wins.")
@click.option("--seed", type=int, default=0, show_default=True,
              help="Global seed; per-step seeds are derived from it.")
def run_command(task_file: Path, rundir: Path | None, step: str | None,
                from_stage: str | None, until_stage: str | None,
                dry_run: bool, offline: bool, allow_network: bool,
                arguments_file: Path | None, template_dir: Path | None,
                max_retries: int, seed: int) -> None:
    """Drive the controller, streaming each step's status as it arrives."""
    from .context import ExecutionPolicy, RunContext
    from .harness.approval import ApprovalQueue
    from .harness.controller import ResearchController, RunOutcome, TERMINAL_STAGES
    from .harness.registry import build_interface_registry
    from .provenance import RunManifest, sha256_file, utc_now

    out = Out()
    task = _load_task(task_file)

    if step is not None and (from_stage is not None or until_stage is not None):
        raise Refusal("--step cannot be combined with --from or --until",
                      next_action="--step X is the same as --from X --until X",
                      exit_code=EXIT_USAGE)
    start = _parse_stage(step or from_stage)
    stop_after = _parse_stage(step or until_stage)

    run_dir = Path(rundir) if rundir is not None else Path("runs") / task.task_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_run_manifest(run_dir, required=False)
    resumed = manifest is not None
    if manifest is None:
        stamp = utc_now().replace(":", "").replace("-", "").replace("+0000", "Z")
        manifest = RunManifest(run_id=f"{task.task_id}-{stamp}",
                               task_id=task.task_id, global_seed=seed)
    manifest.task_input_sha256 = sha256_file(task_file)

    policy = ExecutionPolicy(
        allow_network=bool(allow_network and not offline and not dry_run),
        dry_run=dry_run, max_retries=max_retries)
    if offline and allow_network:
        out.warn("--offline and --allow-network were both given; --offline "
                 "wins, because the safe reading of a contradictory "
                 "instruction is the restrictive one")

    templates = _load_templates(template_dir)
    try:
        registry = build_interface_registry()
    except LookupError as exc:
        raise Refusal(f"the protocol is incomplete in this tree: {exc}",
                      next_action=("install or restore the missing interface "
                                   "module; a run that executes nine of ten "
                                   "steps produces a batch plan that looks "
                                   "complete"),
                      exit_code=EXIT_FAILED) from exc

    ctx = RunContext(task=task, workdir=run_dir, manifest=manifest,
                     policy=policy, templates=templates,
                     logger=lambda message: out.note(f". {message}", indent=2))

    out.kv("task", task.task_id)
    out.kv("run directory", run_dir)
    out.kv("run id", manifest.run_id)
    out.kv("resumed", resumed)
    out.kv("network", "permitted" if policy.allow_network else "disabled")
    problems = templates.integrity_problems()
    if problems:
        out.kv("template gaps", len(problems))
        for problem in problems:
            out.bullet(problem)
    calibration = templates.calibration_report()
    out.kv("geometry windows", calibration.summary())

    if dry_run:
        plan = _dry_run_plan(ctx, registry, templates, task, policy)
        out.heading("dry run -- nothing was executed")
        out.note(plan["statement"])
        out.heading("steps that would run")
        for item in plan["steps"]:
            out.bullet(f"{item['interface']} v{item['version']}")
            if item["unresolved_required_fields"]:
                out.note(f"      waits on: "
                         f"{', '.join(item['unresolved_required_fields'])}")
            if item["required_approvals"]:
                out.note(f"      needs approval: "
                         f"{', '.join(item['required_approvals'])}")
        out.heading("human decision points")
        for gate, unresolved in plan["gates"].items():
            state = "ready to ask" if not unresolved else \
                f"{len(unresolved)} unresolved field(s): {', '.join(unresolved)}"
            out.bullet(f"{gate}: {state}")
        out.heading("cost")
        out.note(plan["cost"]["note"])
        recorded = plan["cost"]["recorded_so_far"]
        out.kv("  recorded so far",
               recorded if recorded else "nothing has been recorded by any "
                                         "step of this run", width=26)
        out.kv("  ceilings",
               policy.cost_ceiling if policy.cost_ceiling
               else "none configured for this run", width=26)
        out.heading("planned scale (from the task file, not a prediction)")
        for key, value in plan["budget"].items():
            out.kv(f"  {key}", value, width=34)
        path = run_dir / "dry_run_plan.json"
        path.write_text(json.dumps(plan, indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")
        out.line("")
        out.line(f"wrote {path}")
        out.line("No run manifest was written: nothing ran, and a manifest "
                 "for a run that did not happen would be read as one that did.")
        return

    queue = ApprovalQueue.load(run_dir / "approvals.json", manifest)
    # Coerced before the controller exists, so a bad payload stops the run
    # with the file and the field named, rather than from inside a step.
    arguments = _coerce_arguments(
        registry, _argument_map(arguments_file), arguments_file, out)
    controller = ResearchController(ctx, registry, queue, arguments=arguments)
    if arguments_file is None:
        out.note("no --arguments file was given, so each step runs with its own "
                 "defaults; the harness does not invent the data flow between "
                 "steps, because which pose set or seed list a step consumes "
                 "is a scientific choice")

    out.heading("run")
    reported = 0
    reached_requested_stage = False
    try:
        if start is not None:
            controller.stage = start
            controller.path.append(start)
        if stop_after is None:
            controller.run()
            reported = _report_attempts(controller, out, reported)
        else:
            moves = 0
            while controller.stage not in TERMINAL_STAGES:
                before = controller.stage
                controller.step()
                reported = _report_attempts(controller, out, reported)
                moves += 1
                if before is stop_after:
                    # The stage the operator asked for has now executed. This
                    # is the ordinary end of a --step or --until run, not a
                    # failure, and it is recorded as such below.
                    reached_requested_stage = True
                    break
                if moves > controller.max_transitions:
                    break
            if controller.stage in TERMINAL_STAGES:
                controller.run()          # records the outcome and the note
    except Exception as exc:              # a defect here must not be swallowed
        manifest.write(run_dir / "run_manifest.json")
        raise Refusal(
            f"the controller raised {type(exc).__name__}: {exc}",
            next_action=("this is a defect in the harness, not a scientific "
                         f"result; the manifest up to the failure is in "
                         f"{run_dir / 'run_manifest.json'}"),
            exit_code=EXIT_FAILED) from exc
    finally:
        reported = _report_attempts(controller, out, reported)

    report_path = controller.write_report()
    manifest_path = manifest.write(run_dir / "run_manifest.json")
    queue.save()

    out.heading("outcome")
    out.kv("stage", controller.stage.value)
    out.kv("outcome", controller.outcome.value if controller.outcome
           else "stopped before a terminal stage")
    out.kv("reason", controller.stop_reason or "none recorded")
    out.kv("cost recorded", dict(manifest.cost_total)
           if manifest.cost_total else "nothing recorded by any step")
    for breach in controller.budget_breaches():
        out.warn(f"cost ceiling: {breach}")

    pending = queue.pending()
    if pending:
        out.heading("waiting on a human")
        for request in pending:
            question = _gate_question(request.gate)
            out.bullet(f"{request.kind.value} {request.gate} "
                       f"(request {request.request_id})")
            if question:
                out.note(f"      {question}")
            if request.detail:
                out.note(f"      {request.detail}")
            if queue.is_granted(request.gate):
                out.note("      this gate already carries a recorded grant; "
                         "the request is a duplicate raised by a later step")
            out.note(f"      answer it with: eagent approve {request.gate} "
                     f"--rundir {run_dir} --actor <your name>")
    for escalation in controller.escalations:
        out.heading("escalation")
        out.kv("stage", escalation.stage.value)
        out.kv("interface", escalation.interface)
        out.kv("kind", escalation.kind.value)
        out.kv("reason", escalation.reason)
        for blocker in escalation.blockers:
            out.bullet(blocker)

    out.line("")
    out.line(f"manifest: {manifest_path}")
    out.line(f"report:   {report_path}")

    outcome = controller.outcome
    undecided = [r for r in pending if not queue.is_granted(r.gate)]
    if outcome is None and stop_after is not None:
        # --step and --until stop the machine in a stage that is not
        # terminal, so the controller never records an outcome. Reaching the
        # stage the operator asked for is the command doing exactly what it
        # was told; reporting it as a failed run made the documented way to
        # drive one stage at a time unusable. A queued decision still
        # outranks it: the next stage genuinely cannot run.
        if undecided:
            outcome = RunOutcome.AWAITING_HUMAN
        elif reached_requested_stage and not controller.escalations:
            out.line("")
            out.line(f"stopped after {stop_after.value}, as asked. The run is "
                     f"intact: continue it with `eagent run {task_file} "
                     f"--rundir {run_dir} --from "
                     f"{controller.stage.value}`.")
            return
    if outcome is RunOutcome.COMPLETED or outcome is RunOutcome.AWAITING_RESULTS:
        return
    if outcome is RunOutcome.AWAITING_HUMAN:
        # The most recent request that is not already answered elsewhere is
        # the one this stop is about. An older pending duplicate, or one
        # whose gate another request already carries a grant for, would name
        # a decision the operator has in fact already taken.
        blocking = (undecided or pending)[-1] if pending else None
        gate = blocking.gate if blocking is not None else None
        raise Refusal(
            controller.stop_reason or "the run is waiting on a human decision",
            gate=gate,
            next_action=(f"eagent approve {gate} --rundir {run_dir} "
                         f"--actor <your name>" if gate else
                         "answer the queued request, then run this command again"),
            exit_code=EXIT_BLOCKED)
    raise Refusal(controller.stop_reason or f"the run ended in "
                                            f"{controller.stage.value}",
                  next_action=f"eagent status {run_dir}",
                  exit_code=EXIT_FAILED)


# ---------------------------------------------------------------------------
# approve
# ---------------------------------------------------------------------------

@main.command("approve")
@click.argument("gate")
@click.option("--actor", required=True,
              help="Who is deciding. An anonymous approval is refused.")
@click.option("--rundir", type=click.Path(path_type=Path), default=Path("."),
              envvar="EAGENT_RUNDIR", show_default=True,
              help="The run whose queue holds the request.")
@click.option("--deny", is_flag=True, help="Record a refusal instead.")
@click.option("--note", default="", help="The reason, recorded with the decision.")
@click.option("--request-id", default=None,
              help="Decide one specific request rather than the latest pending "
                   "one for the gate.")
def approve_command(gate: str, actor: str, rundir: Path, deny: bool,
                    note: str, request_id: str | None) -> None:
    """Record a human decision at a gate, with the name of the human.

    The decision answers a queued request, which is what states *what* was
    being approved. There is no way here to approve a gate in the abstract:
    an approval of "the pipeline" is not an authorisation of a batch.
    """
    from .harness.approval import APPROVAL_GATES, ApprovalQueue, DECISION_POINTS

    out = Out()
    run_dir = Path(rundir)
    manifest = _load_run_manifest(run_dir)
    queue_path = run_dir / "approvals.json"
    if not queue_path.exists():
        raise Refusal(f"no approval queue at {queue_path}",
                      next_action=f"run `eagent run <task.yaml> --rundir "
                                  f"{run_dir}` until it raises the request",
                      exit_code=EXIT_UNRESOLVED)
    queue = ApprovalQueue.load(queue_path, manifest)

    key = request_id or gate
    target = queue.get(key) if request_id else None
    if target is None:
        pending = queue.pending(gate)
        if not pending:
            decided = [r for r in queue.for_gate(gate) if r.decision.is_final]
            detail = (f"it was already {decided[-1].decision.value}d by "
                      f"{decided[-1].actor} at {decided[-1].decided_at}"
                      if decided else "no request for it has been raised")
            raise Refusal(
                f"there is no pending request at '{gate}': {detail}",
                next_action=("a decision must answer a request that states what "
                             "was being decided; run the pipeline until it asks"),
                exit_code=EXIT_UNRESOLVED)
        target = pending[-1]

    point = DECISION_POINTS.get(target.gate)
    out.heading(f"gate {target.gate}")
    if point:
        out.note(point.question)
        out.note(f"if this is wrong: {point.consequence_if_wrong}")
    out.kv("request", target.request_id)
    out.kv("raised by", target.requested_by)
    out.kv("raised at", target.requested_at)
    out.kv("detail", target.detail or None)
    if target.payload:
        out.heading("what this decision covers")
        for key_name, value in sorted(target.payload.items()):
            out.kv(f"  {key_name}", value, width=28)
    if point:
        unshown = [f for f in point.must_show if f not in target.payload]
        if unshown:
            out.warn("the request does not carry: " + ", ".join(unshown)
                     + " -- decide only if you have those facts from elsewhere")

    try:
        if deny:
            decided = queue.deny(target.request_id, actor=actor, reason=note)
        else:
            decided = queue.grant(target.request_id, actor=actor, reason=note)
    except ValueError as exc:
        raise Refusal(str(exc),
                      next_action="raise a new request rather than overwriting "
                                  "the record of what was agreed",
                      exit_code=EXIT_FAILED) from exc

    manifest.write(run_dir / "run_manifest.json")
    out.heading("recorded")
    out.kv("decision", decided.decision.value)
    out.kv("actor", decided.actor)
    out.kv("at", decided.decided_at)
    out.kv("reason", decided.reason or None)
    if target.gate not in APPROVAL_GATES:
        out.note("this was an operator task, not one of the three gates; it "
                 "does not unblock a gated step")
    out.line("")
    out.line(f"resume with: eagent run <task.yaml> --rundir {run_dir}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@main.command("status")
@click.argument("rundir", type=click.Path(exists=True, file_okay=False,
                                          path_type=Path))
def status_command(rundir: Path) -> None:
    """Where the state machine is, what it cost, and what is still open."""
    out = Out()
    manifest = _load_run_manifest(rundir)

    out.kv("run", manifest.run_id)
    out.kv("task", manifest.task_id)
    out.kv("created", manifest.created_at)
    out.kv("task input sha256", manifest.task_input_sha256)

    report_path = rundir / "controller_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        out.heading("state machine")
        out.kv("stage", report.get("stage"))
        out.kv("outcome", report.get("outcome"))
        out.kv("stop reason", report.get("stop_reason") or None)
        path = report.get("path") or []
        out.kv("path", " -> ".join(path) if path else None)
        for escalation in report.get("escalations", []):
            out.bullet(f"escalation at {escalation.get('stage')} "
                       f"({escalation.get('kind')}): {escalation.get('reason')}")
    else:
        out.heading("state machine")
        out.note("no controller_report.json in this directory, so the stage "
                 "cannot be stated; the steps below are what the manifest "
                 "recorded")

    out.heading(f"steps ({len(manifest.steps)})")
    if not manifest.steps:
        out.note("none recorded")
    for record in manifest.steps:
        out.bullet(f"{record.step_id}: {record.status} "
                   f"({record.started_at} -> {record.finished_at})")
        if record.message:
            out.note(f"      {record.message}")

    out.heading("cost")
    if manifest.cost_total:
        for key, value in sorted(manifest.cost_total.items()):
            out.kv(f"  {key}", value, width=26)
    else:
        out.note("nothing was recorded by any step. This is not a claim that "
                 "the run was free: it means no step reported a cost")

    out.heading("approvals")
    if manifest.approvals:
        for approval in manifest.approvals:
            out.bullet(f"{approval.get('gate')}: {approval.get('decision')} by "
                       f"{approval.get('actor')} at {approval.get('at')}")
    else:
        out.note("none recorded; no gate in this run has been cleared by a "
                 "named person")

    queue_path = rundir / "approvals.json"
    if queue_path.exists():
        from .harness.approval import ApprovalQueue
        queue = ApprovalQueue.load(queue_path, manifest)
        pending = queue.pending()
        out.heading(f"pending decisions ({len(pending)})")
        for request in pending:
            out.bullet(f"{request.gate} (request {request.request_id}) "
                       f"raised by {request.requested_by}")
            if request.detail:
                out.note(f"      {request.detail}")
        if not pending:
            out.note("none")

    open_uncertainties = [
        (record.step_id, item)
        for record in manifest.steps for item in record.uncertainty]
    out.heading(f"open uncertainties ({len(open_uncertainties)})")
    if not open_uncertainties:
        out.note("none recorded")
    for step_id, item in open_uncertainties:
        out.bullet(f"[{step_id}] {item.get('code')}: {item.get('question')}")
        if item.get("resolvable_by"):
            out.note(f"      resolvable by: {item['resolvable_by']}")

    blockers = [(record.step_id, flag)
                for record in manifest.steps for flag in record.qc_flags
                if flag.get("severity") == "blocker"]
    out.heading(f"blocking QC flags ({len(blockers)})")
    if not blockers:
        out.note("none recorded")
    for step_id, flag in blockers:
        out.bullet(f"[{step_id}] {flag.get('code')}: {flag.get('message')}")


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def _task_from_run(run_dir: Path, task_file: Path | None):
    """The task a run was about, from ``--task`` or from its reaction spec.

    Reconstructed from the artifact the run itself wrote rather than from a
    fresh guess, so a verification is performed against the chemistry the run
    actually used.
    """
    from .schemas import TaskSpec

    if task_file is not None:
        return _load_task(task_file)
    spec_path = run_dir / "reaction_spec.yaml"
    if not spec_path.exists():
        raise Refusal(
            f"no reaction_spec.yaml in {run_dir} and no --task given, so the "
            f"verifier has no statement of what the run was supposed to be "
            f"about",
            next_action="pass --task <task.yaml>",
            exit_code=EXIT_UNRESOLVED)
    document = _read_yaml(spec_path, "reaction spec")
    if not isinstance(document, Mapping):
        raise Refusal(f"{spec_path} did not parse into a mapping",
                      exit_code=EXIT_UNRESOLVED)
    fields = {k: document[k] for k in
              ("task_id", "task_mode", "reaction", "conditions", "objectives",
               "assumptions") if k in document}
    fields.setdefault("task_id", "unrecorded")
    try:
        return TaskSpec(**fields)
    except Exception as exc:
        raise Refusal(f"{spec_path} could not be read back as a task "
                      f"({type(exc).__name__}: {exc})",
                      next_action="pass --task <task.yaml> instead",
                      exit_code=EXIT_UNRESOLVED) from exc


def _load_json_list(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        for key in ("candidates", "claims", "items", "records"):
            if key in data:
                data = data[key]
                break
    if not isinstance(data, list):
        raise Refusal(f"{path} must hold a list",
                      exit_code=EXIT_UNRESOLVED)
    return data


@main.command("verify")
@click.argument("rundir", type=click.Path(exists=True, file_okay=False,
                                          path_type=Path))
@click.option("--task", "task_file",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="The task file, if the run did not write a "
                                 "reaction spec.")
@click.option("--candidates", "candidates_file",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Serialised candidates. Default: "
                                 "<rundir>/candidates.json when it exists.")
@click.option("--claims", "claims_file",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              default=None, help="Claims to check against the artifacts. "
                                 "Default: <rundir>/claims.json when it exists.")
@click.option("--templates", "template_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Template library root.")
def verify_command(rundir: Path, task_file: Path | None,
                   candidates_file: Path | None, claims_file: Path | None,
                   template_dir: Path | None) -> None:
    """Run the independent verifier over what this run recorded.

    Independent means it re-derives what it checks from the primary material
    and never reads the producing step's own verdict. A check it cannot run is
    reported as unverifiable, which is not the same as passed.
    """
    from .context import ExecutionPolicy, RunContext
    from .envelope import Artifact, Severity
    from .harness.verifier import Claim, IndependentVerifier
    from .schemas.candidate import Candidate
    from .schemas.record import ExperimentRecord

    out = Out()
    manifest = _load_run_manifest(rundir)
    task = _task_from_run(rundir, task_file)
    templates = _load_templates(template_dir)

    candidates_path = candidates_file or (rundir / "candidates.json")
    claims_path = claims_file or (rundir / "claims.json")
    records_path = rundir / "ingest_results" / "experiment_records.jsonl"

    candidates = [Candidate(**item) for item in _load_json_list(candidates_path)] \
        if candidates_path.exists() else []
    claims = [Claim(**item) for item in _load_json_list(claims_path)] \
        if claims_path.exists() else []
    records = []
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(ExperimentRecord(**json.loads(line)))

    artifacts = []
    for record in manifest.steps:
        for raw in record.artifacts:
            artifacts.append(Artifact(**raw))

    if not candidates and not claims and not records:
        raise Refusal(
            f"there is nothing in {rundir} for the verifier to check: no "
            f"{candidates_path.name}, no {claims_path.name} and no "
            f"ingest_results/experiment_records.jsonl",
            next_action=("export the candidates the run produced, or point at "
                         "them with --candidates; an empty verification is not "
                         "a passed one"),
            exit_code=EXIT_UNRESOLVED)

    ctx = RunContext(task=task, workdir=rundir, manifest=manifest,
                     policy=ExecutionPolicy(allow_network=False),
                     templates=templates)
    result = IndependentVerifier().verify(
        ctx, candidates=candidates, claims=claims, records=records,
        artifacts=artifacts)

    out.kv("candidates checked", len(candidates))
    out.kv("claims checked", len(claims))
    out.kv("records checked", len(records))
    out.kv("status", result.status.value)
    out.kv("message", result.message or None)

    findings = result.data.get("verification", {})
    out.heading(f"checks run ({len(findings.get('checks_run', []))})")
    for check in findings.get("checks_run", []):
        out.bullet(check)

    out.heading(f"findings ({len(result.qc_flags)})")
    if not result.qc_flags:
        out.note("none")
    for flag in result.qc_flags:
        marker = {Severity.BLOCKER: "BLOCKER", Severity.WARN: "warn",
                  Severity.INFO: "info"}[flag.severity]
        subject = f" [{flag.subject}]" if flag.subject else ""
        out.bullet(f"{marker}: {flag.code}{subject} -- {flag.message}")

    unverifiable = findings.get("unverifiable", [])
    out.heading(f"could not be checked ({len(unverifiable)})")
    if not unverifiable:
        out.note("nothing: every check had the material it needed")
    for item in unverifiable:
        out.bullet(str(item))
    if unverifiable:
        out.note("an unverified claim is not a verified one")

    for artifact in result.artifacts:
        out.line("")
        out.line(f"report: {artifact.path}")

    if result.blockers:
        raise Refusal(
            f"verification failed: {len(result.blockers)} blocking problem(s)",
            next_action=("fix the inputs the verifier names and verify again; "
                         "these are defects, not trade-offs"),
            exit_code=EXIT_FAILED)


# ---------------------------------------------------------------------------
# bundle
# ---------------------------------------------------------------------------

@main.command("bundle")
@click.argument("rundir", type=click.Path(exists=True, file_okay=False,
                                          path_type=Path))
@click.option("--out", "-o", "out_dir", type=click.Path(path_type=Path),
              default=None, help="Where to write the package. "
                                 "Default: <rundir>/bundle.")
@click.option("--overwrite", is_flag=True,
              help="Replace the contents of an existing output directory.")
@click.option("--strict", is_flag=True,
              help="Exit non-zero when the package is incomplete.")
def bundle_command(rundir: Path, out_dir: Path | None, overwrite: bool,
                   strict: bool) -> None:
    """Assemble the deliverables, recording every item that is not there."""
    from .deliverables.bundle import assemble_bundle, bundle_summary_lines

    out = Out()
    try:
        result = assemble_bundle(rundir, out_dir, overwrite=overwrite)
    except FileExistsError as exc:
        raise Refusal(str(exc), next_action="pass --overwrite, or choose "
                                            "another --out directory",
                      exit_code=EXIT_USAGE) from exc

    for line in bundle_summary_lines(result):
        out.line(line)
    out.line("")
    out.line(f"manifest: {result.manifest_path}")
    if result.report_path is not None:
        out.line(f"report:   {result.report_path}")
    out.line("")
    out.line("verify it anywhere with: eagent bundle-verify "
             f"{result.bundle_dir}")

    if not result.complete:
        message = (f"the package is incomplete: "
                   f"{len(result.missing_names)} missing, "
                   f"{len(result.partial_names)} partial")
        if strict:
            raise Refusal(message,
                          next_action=("run the steps named in the manifest's "
                                       "missing list, or have a curator supply "
                                       "the files"),
                          exit_code=EXIT_FAILED)
        out.warn(message + " -- it is recorded as such in the manifest, so it "
                           "will not be mistaken for a complete one")


@main.command("bundle-verify")
@click.argument("bundle_dir", type=click.Path(exists=True, path_type=Path))
def bundle_verify_command(bundle_dir: Path) -> None:
    """Re-check a package: every declared file present, every hash matching."""
    from .deliverables.bundle import verify_bundle

    out = Out()
    verification = verify_bundle(bundle_dir)
    out.kv("bundle", verification.bundle_dir)
    out.kv("items checked", verification.n_items_checked)
    out.kv("files checked", verification.n_files_checked)
    out.kv("declared complete", verification.complete)
    out.heading(f"problems ({len(verification.problems)})")
    if not verification.problems:
        out.note("none: every declared file is present and hashes as recorded")
    for problem in verification.problems:
        out.bullet(problem)
    if not verification.ok:
        raise Refusal(f"{len(verification.problems)} problem(s) in this package",
                      next_action=("do not quote from it until each one is "
                                   "explained; a file that does not hash as "
                                   "recorded has changed since assembly"),
                      exit_code=EXIT_FAILED)


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------

@main.group("templates")
def templates_group() -> None:
    """Read the template library: every threshold in a run comes from it."""


@templates_group.command("list")
@click.option("--templates", "template_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None)
def templates_list(template_dir: Path | None) -> None:
    """List the loaded templates by kind."""
    out = Out()
    library = _load_templates(template_dir)
    out.kv("templates loaded", len(library))
    out.kv("families", library.family_names())
    out.kv("reaction classes", library.reaction_classes())
    out.heading("templates")
    for template_id in library.ids():
        kind = library.kind_of(template_id)
        theoretical = " [theoretical model]" \
            if library.is_theoretical(template_id) else ""
        out.bullet(f"{template_id:<46} {kind}{theoretical}")
        out.note(f"      {library.path_of(template_id)}")


@templates_group.command("show")
@click.argument("template_id")
@click.option("--templates", "template_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None)
def templates_show(template_id: str, template_dir: Path | None) -> None:
    """Print one template, with the caveats a report quoting it must carry."""
    import yaml
    from .errors import TemplateError

    out = Out()
    library = _load_templates(template_dir)
    try:
        template = library.require(template_id)
    except TemplateError as exc:
        raise Refusal(str(exc),
                      next_action="list what is loaded with `eagent templates list`",
                      exit_code=EXIT_UNRESOLVED) from exc
    out.kv("template", template_id)
    out.kv("kind", library.kind_of(template_id))
    out.kv("file", library.path_of(template_id))
    out.heading("caveats for any report quoting this template")
    caveats = library.confidence_caveats(template_id)
    if not caveats:
        out.note("none recorded")
    for caveat in caveats:
        out.bullet(caveat)
    out.heading("content")
    out.line(yaml.safe_dump(template.model_dump(mode="json"), sort_keys=False,
                            allow_unicode=True).rstrip())


@templates_group.command("lint")
@click.option("--templates", "template_dir",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None)
@click.option("--strict", is_flag=True,
              help="Exit non-zero when any gap is reported.")
def templates_lint(template_dir: Path | None, strict: bool) -> None:
    """Report the library's gaps as sentences a curator can act on.

    Exits zero by default: an incomplete library that says so is a legitimate
    state, and the gaps belong in a run manifest rather than in a crash. Use
    ``--strict`` where a pipeline must refuse to start on one.
    """
    out = Out()
    library = _load_templates(template_dir)
    calibration = library.calibration_report()

    out.kv("templates loaded", len(library))
    out.heading("geometry windows")
    out.note(calibration.summary())
    if calibration.fully_uncalibrated:
        out.note("no window in this library was fitted to systems of known "
                 "activity: a pass against any of them is not evidence of "
                 "catalytic competence, and a failure may not disqualify a "
                 "candidate")
    for record in calibration.uncalibrated:
        out.bullet(f"{record.template_id}: {record.name} "
                   f"({record.severity}, {record.authority})")
    out.heading("gating windows that are not calibrated")
    if not calibration.gating_uncalibrated:
        out.note("none: no unfitted window is allowed to reject a candidate")
    for record in calibration.gating_uncalibrated:
        out.bullet(f"{record.template_id}: {record.name}")

    problems = library.integrity_problems()
    out.heading(f"integrity ({len(problems)})")
    if not problems:
        out.note("no gap found")
    for problem in problems:
        out.bullet(problem)

    if problems and strict:
        raise Refusal(f"{len(problems)} gap(s) in the template library",
                      next_action="have a curator close them, or drop --strict "
                                  "to treat them as recorded gaps",
                      exit_code=EXIT_FAILED)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

@main.group("sources")
def sources_group() -> None:
    """Read the data-source registry: what each resource may be used to claim."""


def _load_sources(directory: Path | None):
    from .datalayer.registry import RegistryError, SourceRegistry
    try:
        return SourceRegistry.from_directory(directory)
    except RegistryError as exc:
        raise Refusal(f"the data-source registry did not load: {exc}",
                      next_action="fix or restore the named file under "
                                  "configs/datasources/",
                      exit_code=EXIT_UNRESOLVED) from exc


@sources_group.command("list")
@click.option("--dir", "directory",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None)
@click.option("--layer", default=None,
              help="Only sources serving this data layer.")
def sources_list(directory: Path | None, layer: str | None) -> None:
    """List registered sources, per layer, with what nobody has verified."""
    from .datalayer.layers import DataLayer

    out = Out()
    registry = _load_sources(directory)
    if layer is not None:
        try:
            wanted = DataLayer(layer)
        except ValueError as exc:
            raise Refusal(f"'{layer}' is not a data layer",
                          next_action="one of: "
                                      + ", ".join(l.value for l in DataLayer),
                          exit_code=EXIT_USAGE) from exc
        out.heading(f"layer {wanted.value}")
        for source in registry.by_layer(wanted):
            out.bullet(source.summary())
        return
    for line in registry.report_lines():
        out.line(line)
    out.heading("sources")
    for source in registry:
        out.bullet(source.summary())


@sources_group.command("show")
@click.argument("source_id")
@click.option("--dir", "directory",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None)
def sources_show(source_id: str, directory: Path | None) -> None:
    """Print one source: what it is good for, and what it may never claim."""
    from .datalayer.registry import UnknownSourceError

    out = Out()
    registry = _load_sources(directory)
    try:
        source = registry.get(source_id)
    except UnknownSourceError as exc:
        raise Refusal(str(exc),
                      next_action="list them with `eagent sources list`",
                      exit_code=EXIT_UNRESOLVED) from exc
    out.kv("id", source.id)
    out.kv("name", source.display_name)
    out.kv("layers", [l.value for l in source.layers])
    out.kv("access modes", [m.value for m in source.access_modes])
    out.kv("endpoint", source.endpoint)
    out.kv("connectivity verified", source.connectivity_verified)
    out.kv("licence", source.license)
    out.kv("licence source", source.license_source)
    out.kv("needs legal review", source.needs_legal_review)
    out.kv("evidence ceiling", getattr(source.evidence_strength_ceiling, "value",
                                       source.evidence_strength_ceiling))
    out.kv("version", source.version)
    out.kv("record count", source.approximate_record_count)
    out.heading("good for")
    for item in source.good_for:
        out.bullet(item)
    out.heading("not good for")
    for item in source.not_good_for:
        out.bullet(item)
    out.heading("capabilities (documented, not tested here)")
    for name, state in sorted(source.capabilities.as_dict().items()):
        out.kv(f"  {name}", state, width=30)
    out.heading("lineage")
    out.kv("  derived from", source.derived_from or None, width=22)
    out.kv("  lineage complete", source.derived_from_complete, width=22)
    out.heading("curation")
    out.kv("  needs curation", source.needs_curation, width=22)
    for note in source.curation_notes:
        out.bullet(note)


@sources_group.command("independence")
@click.option("--dir", "directory",
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None)
@click.argument("source_ids", nargs=-1)
def sources_independence(directory: Path | None,
                         source_ids: tuple[str, ...]) -> None:
    """Collapse sources that share an upstream, so re-publication is not corroboration."""
    out = Out()
    registry = _load_sources(directory)
    report = registry.independence_report(list(source_ids) or None)
    for line in report.report_lines():
        out.line(line)
    out.line("")
    out.line("The two numbers differ on purpose: a group holding a source "
             "whose lineage is admittedly incomplete has not been shown to be "
             "independent of anything, so it is not counted.")


if __name__ == "__main__":  # pragma: no cover - module entry point
    main()
