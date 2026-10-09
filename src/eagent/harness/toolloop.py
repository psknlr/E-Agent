"""Letting a model drive the project's own readers, within a fence.

WHAT THIS IS FOR
================
The reference set is a few thousand numbers with a rule attached to almost
every one: this ``Km`` is a bound and not a point, that zero is a detection
limit and not a measurement, these two entries are one lineage. Answering
"which HBDH records have a matched complex, and what is the density behind it"
means joining three tables and respecting four of those rules. A model is
useful for that, and only if it reads the tables through code that already
knows the rules.

So the tools here are the project's own loaders. The model does not get the
CSVs; it gets :mod:`eagent.eval.kred_reference`,
:mod:`eagent.eval.kred_activity` and the audit, which refuse a withheld
quantity instead of returning a plausible number for it. A model that asks for
the ``Km`` of ``PaHBDH_H150N`` is told it is a bound and why, in the same words
a person would get.

WHAT THE FENCE IS
=================
* **Read-only, structurally.** A tool is registered through
  :meth:`ToolLoop.register`, which refuses any tool that declares
  :attr:`ReadOnlyTool.writes` or :attr:`ReadOnlyTool.reaches_network`. The
  shipped set (:func:`reference_tools`) are closures over data loaded before
  the loop starts; none of them can write a calibration record, edit a
  template, or open a socket.
* **Bounded.** A maximum number of turns, a maximum number of tool calls per
  turn and overall, and a cap on the bytes a result may add to the
  conversation. A loop that can run forever on somebody's metered key is not a
  tool, it is a bill.
* **The model proposes and reads. It decides nothing.** Nothing in this module
  writes to the repository, and the loop's output is a transcript plus whatever
  the model said, with the numeric guard applied to it exactly as in the
  planner. A quantity in the model's prose still needs a citation that names
  the row it came from, and every tool result carries the citation keys for its
  own rows so that the model *can* comply rather than being asked to invent
  them.
* **Every call and result is recorded**, with the provider and model that asked
  for it, because "the model concluded" is not a finding unless somebody can
  see what it was shown.

WHAT IT IS NOT
==============
It is not part of a run. The controller's planner
(:mod:`eagent.harness.planner`) is the seam where a model influences what the
pipeline does, and it has its own guards and approvals. This is an analysis
console over data that is already final: useful for asking questions, incapable
of changing an answer.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..errors import EAgentError
from .llm import LLMClient, ModelTurn, NumericGuard, ToolCall

__all__ = [
    "ToolLoopError",
    "ToolRefusal",
    "ReadOnlyTool",
    "ToolResult",
    "LoopLimits",
    "ToolLoop",
    "reference_tools",
    "SYSTEM_PROMPT",
]


class ToolLoopError(EAgentError):
    """The loop was configured in a way that would not be safe or bounded."""


class ToolRefusal(EAgentError):
    """A tool declined to answer, with the reason the caller should see."""


@dataclass(frozen=True)
class ReadOnlyTool:
    """One thing a model may ask for, and the code that answers it.

    ``writes`` and ``reaches_network`` exist to be ``False``. They are declared
    per tool rather than assumed so that :meth:`ToolLoop.register` can refuse a
    tool that is neither, instead of the fence being a sentence in a docstring.
    """

    name: str
    description: str
    parameters: Mapping[str, str]
    run: Callable[..., Any]
    writes: bool = False
    reaches_network: bool = False

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": dict(self.parameters)}


@dataclass
class ToolResult:
    """What one tool call produced, or why it did not."""

    call: ToolCall
    ok: bool
    value: Any = None
    refusal: str = ""
    truncated: bool = False
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.call.interface, "arguments": dict(self.call.arguments),
                "rationale": self.call.rationale, "ok": self.ok,
                "value": self.value, "refusal": self.refusal,
                "truncated": self.truncated, "elapsed_ms": self.elapsed_ms}


@dataclass(frozen=True)
class LoopLimits:
    """Every way the loop can be made to stop."""

    max_turns: int = 6
    max_calls_per_turn: int = 4
    max_calls_total: int = 20
    max_result_bytes: int = 8_000
    max_wall_seconds: float = 120.0

    def __post_init__(self) -> None:
        for name, value in (("max_turns", self.max_turns),
                            ("max_calls_per_turn", self.max_calls_per_turn),
                            ("max_calls_total", self.max_calls_total),
                            ("max_result_bytes", self.max_result_bytes)):
            if value < 1:
                raise ToolLoopError(f"{name} must be at least 1, got {value}")
        if self.max_wall_seconds <= 0:
            raise ToolLoopError("max_wall_seconds must be positive")


SYSTEM_PROMPT = """\
You are reading a curated reference set of enzyme structures and kinetic
measurements through the project's own loaders. Answer the question you are
given from what the tools return, and from nothing else.

The tools are read-only. You cannot change any value, calibrate any window, or
run any experiment, and nothing you say will be acted on automatically.

Rules that the data itself enforces, so do not try to work around them:
- A quantity the loader withholds has no value. It will tell you why: a Km
  printed as a bound is not a point estimate, a product concentration of zero
  in a fixed-time assay is a detection limit and not a measured zero, an ND is
  not a zero, a blank enantiomeric excess is undefined. Report the refusal.
  Never substitute a nearby number, and never treat an absence as a zero.
- Different label types do not pool. A kcat, a reported catalytic efficiency, a
  relative depletion slope, a fixed-time product concentration and an
  enantiomeric excess are five quantities. Do not rank across them or average
  them.
- Counts of records are not counts of independent enzymes. Ask the independence
  tools; chains, conformers, PDB entries of one protein and orthologs of one
  parent are not separate samples.
- An insoluble construct was never assayed. It is not a catalytically negative
  enzyme.

When you quote a number, put a citation straight after it naming where it can
be found again:
  [cite artifact=<key> sha256=<digest> row=<id> field=<column> method=read]
Every tool result carries the artifact key, digest and row for its own values,
so use those. One citation backs one number.

Respond with a single JSON object containing any of:
  "reasoning":   what you concluded and why
  "tool_calls":  [{"interface": <tool name>, "arguments": {...}, "rationale": ...}]
  "questions":   [strings, where the data cannot settle something]
Call tools until you can answer, then reply with "reasoning" and no tool_calls.
"""


class ToolLoop:
    """A bounded read-only agent loop over registered tools."""

    def __init__(self, client: LLMClient, *, limits: LoopLimits | None = None,
                 guard: NumericGuard | None = None,
                 system: str = SYSTEM_PROMPT) -> None:
        self.client = client
        self.limits = limits or LoopLimits()
        # The same guard the planner uses. Off-strict by default here because
        # an analysis console has no artifact index to check a value against,
        # and a guard that always raises gets switched off; the grammar and the
        # adjacency binding are still enforced.
        self.guard = guard if guard is not None else NumericGuard(strict=False)
        self.system = system
        self._tools: dict[str, ReadOnlyTool] = {}

    # -- registration ------------------------------------------------------
    def register(self, tool: ReadOnlyTool) -> None:
        """Add a tool, or refuse it. The fence is here."""
        if tool.writes or tool.reaches_network:
            raise ToolLoopError(
                f"tool {tool.name!r} declares writes={tool.writes} "
                f"reaches_network={tool.reaches_network}; this loop registers "
                f"read-only local tools only. A model that can write is a "
                f"different thing with different approvals.")
        if tool.name in self._tools:
            raise ToolLoopError(f"a tool named {tool.name!r} is already registered")
        if not tool.description.strip():
            raise ToolLoopError(f"tool {tool.name!r} has no description, so the "
                                f"model has nothing to choose it by")
        self._tools[tool.name] = tool

    def register_all(self, tools: Sequence[ReadOnlyTool]) -> None:
        for tool in tools:
            self.register(tool)

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[n].schema() for n in self.tool_names]

    # -- execution ---------------------------------------------------------
    def _execute(self, call: ToolCall) -> ToolResult:
        started = time.monotonic()
        tool = self._tools.get(call.interface)
        if tool is None:
            return ToolResult(call, ok=False, refusal=(
                f"there is no tool named {call.interface!r}. Available: "
                f"{', '.join(self.tool_names)}"))
        arguments = dict(call.arguments or {})
        unexpected = sorted(set(arguments) - set(tool.parameters))
        if unexpected:
            return ToolResult(call, ok=False, refusal=(
                f"{tool.name} has no parameter(s) {unexpected}; it takes "
                f"{sorted(tool.parameters)}"))
        try:
            value = tool.run(**arguments)
        except ToolRefusal as exc:
            return ToolResult(call, ok=False, refusal=str(exc),
                              elapsed_ms=int((time.monotonic() - started) * 1000))
        except TypeError as exc:
            return ToolResult(call, ok=False, refusal=(
                f"{tool.name} could not be called with those arguments: {exc}"))
        except EAgentError as exc:
            # A loader's own refusal -- a withheld quantity, an unknown id. This
            # is the useful case: the model is told why there is no number.
            return ToolResult(call, ok=False, refusal=str(exc),
                              elapsed_ms=int((time.monotonic() - started) * 1000))
        except Exception as exc:                                 # noqa: BLE001
            return ToolResult(call, ok=False, refusal=(
                f"{tool.name} failed: {type(exc).__name__}: {exc}"),
                elapsed_ms=int((time.monotonic() - started) * 1000))
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        truncated = False
        if len(rendered.encode("utf-8")) > self.limits.max_result_bytes:
            truncated = True
            value = {"truncated": True,
                     "note": (f"this result exceeded "
                              f"{self.limits.max_result_bytes} bytes and was cut; "
                              f"ask a narrower question"),
                     "head": rendered[:self.limits.max_result_bytes]}
        return ToolResult(call, ok=True, value=value, truncated=truncated,
                          elapsed_ms=int((time.monotonic() - started) * 1000))

    def run(self, question: str) -> dict[str, Any]:
        """Answer one question, recording everything. Never raises on a refusal.

        Returns the transcript: the turns, every tool call with its result or
        refusal, the guard's report on each model message, and why the loop
        stopped. A caller reads the record; it is not asked to trust a summary.
        """
        if not self._tools:
            raise ToolLoopError("no tools are registered, so there is nothing "
                                "for the model to read")
        messages: list[dict[str, str]] = [{"role": "user", "content": question}]
        transcript: list[dict[str, Any]] = []
        calls_made = 0
        stopped = "the model answered without asking for another tool"
        started = time.monotonic()
        answer = ""

        for turn_index in range(self.limits.max_turns):
            if time.monotonic() - started > self.limits.max_wall_seconds:
                stopped = (f"the wall-clock limit of "
                           f"{self.limits.max_wall_seconds:g} s was reached")
                break
            try:
                raw = self.client.complete(self.system, messages,
                                           tools=self.schemas(), temperature=0.0)
            except EAgentError as exc:
                stopped = f"the provider call failed: {exc}"
                transcript.append({"turn": turn_index, "provider_error": str(exc)})
                break
            report = self.guard.inspect(raw)
            turn = _parse(raw)
            entry: dict[str, Any] = {
                "turn": turn_index, "reasoning": turn.reasoning,
                "questions": list(turn.questions),
                "requested": [c.interface for c in turn.tool_calls],
                "guard": {"summary": report.summary(), "clean": report.clean,
                          "uncited_quantities": list(report.uncited_quantities),
                          "broken_citations": list(report.broken_citations)},
            }
            if turn.reasoning:
                answer = turn.reasoning
            messages.append({"role": "assistant", "content": raw})

            if self.guard.strict and not report.clean:
                # The guard raises in strict mode. Here that is recorded and the
                # loop stops rather than propagating: the transcript is the
                # output, and a refused answer is a result worth reading.
                entry["guard"]["refused"] = True
                transcript.append(entry)
                stopped = ("the numeric guard refused the model's output: "
                           + report.summary())
                answer = ""
                break

            if not turn.tool_calls:
                transcript.append(entry)
                break
            if len(turn.tool_calls) > self.limits.max_calls_per_turn:
                entry["clipped"] = (
                    f"{len(turn.tool_calls)} tools requested; "
                    f"{self.limits.max_calls_per_turn} is the per-turn limit")
            results: list[ToolResult] = []
            for call in turn.tool_calls[:self.limits.max_calls_per_turn]:
                if calls_made >= self.limits.max_calls_total:
                    entry["clipped"] = (f"the total limit of "
                                        f"{self.limits.max_calls_total} tool "
                                        f"calls was reached")
                    break
                results.append(self._execute(call))
                calls_made += 1
            entry["results"] = [r.to_dict() for r in results]
            transcript.append(entry)
            if calls_made >= self.limits.max_calls_total:
                stopped = (f"the total limit of {self.limits.max_calls_total} "
                           f"tool calls was reached")
                break
            messages.append({
                "role": "user",
                "content": "Tool results:\n" + json.dumps(
                    [r.to_dict() for r in results], ensure_ascii=False,
                    indent=1, default=str)})
        else:
            stopped = f"the turn limit of {self.limits.max_turns} was reached"

        return {
            "question": question,
            "provider": getattr(self.client, "name", "unknown"),
            "runs_remotely": getattr(self.client, "runs_remotely", True),
            "limits": {"max_turns": self.limits.max_turns,
                       "max_calls_per_turn": self.limits.max_calls_per_turn,
                       "max_calls_total": self.limits.max_calls_total,
                       "max_result_bytes": self.limits.max_result_bytes,
                       "max_wall_seconds": self.limits.max_wall_seconds},
            "tools_offered": self.tool_names,
            "turns": transcript,
            "tool_calls_made": calls_made,
            "stopped_because": stopped,
            "answer": answer,
            "caveat": ("the model read the project's loaders and said this; "
                       "nothing here was measured by the model, nothing was "
                       "written, and a quantity without a citation naming its "
                       "row is not sourced"),
        }


def _parse(raw: str) -> ModelTurn:
    from .llm import parse_turn
    return parse_turn(raw)


# ==========================================================================
# the shipped read-only tools over the reference set
# ==========================================================================

def reference_tools(reference_dir: Any = None) -> list[ReadOnlyTool]:
    """Readers over the KRED reference set, as tools. Everything is loaded once.

    Each result carries ``cite`` with the artifact key, the data digest and the
    row, so a model quoting a number has the citation keys to hand rather than
    being asked for ones it cannot know.
    """
    import json as _json
    from pathlib import Path

    from ..eval.kred_activity import (
        ENDPOINT_SUBSTRATES, SUBSTRATES, load_activity_set, independence as
        activity_independence,
    )
    from ..eval.kred_reference import default_reference_dir, load_reference_set

    base = Path(reference_dir) if reference_dir is not None else default_reference_dir()
    rs = load_reference_set(base, verify=False, strict=False)
    digest = rs.manifest_digest or _json.loads(
        (base / "MANIFEST.json").read_text(encoding="utf-8")).get("data_digest", "")
    activity = load_activity_set(base, strict=False)

    def cite(artifact: str, row: str) -> dict[str, str]:
        return {"artifact": artifact, "sha256": digest, "row": row,
                "method": "read"}

    def reference_summary() -> dict[str, Any]:
        """Counts and lineage independence for the structure and kinetic set."""
        return {"counts": rs.counts(), "independence": rs.independence(),
                "cite": cite("kred_reference", "summary"),
                "note": ("record counts are not counts of independent enzymes; "
                         "the lineage counts are")}

    def list_kinetic_records(tier: str = "") -> dict[str, Any]:
        """Label ids, optionally one tier: core, secondary, sensitivity or nd."""
        records = rs.records(tier) if tier else rs.kinetics
        if tier and not records:
            raise ToolRefusal(
                f"no records in tier {tier!r}; tiers are core, secondary, "
                f"sensitivity, nd")
        return {"tier": tier or "all",
                "label_ids": [k.label_id for k in records],
                "n": len(records), "cite": cite("kred_reference", "kinetics")}

    def list_structure_entries() -> dict[str, Any]:
        """Enumerate available PDB ids before calling structure_entry."""
        return {"entries": [{"pdb_id": entry.pdb_id, "enzyme": entry.enzyme,
                             "variant": entry.variant, "lineage": entry.lineage}
                            for entry in rs.structures],
                "cite": cite("kred_reference", "structures")}

    def list_activity_constructs() -> dict[str, Any]:
        """Enumerate construct and substrate ids before calling activity_endpoint."""
        return {"enzyme_ids": [record.enzyme_id for record in activity.records],
                "substrate_ids": sorted(SUBSTRATES),
                "cite": cite("kred_activity", "constructs"),
                "note": "Listing a construct does not imply it was soluble or assayed."}

    def kinetic_record(label_id: str) -> dict[str, Any]:
        """One kinetic record: its numbers, its label type, and what is withheld."""
        try:
            record = rs.kinetic(label_id)
        except KeyError:
            raise ToolRefusal(
                f"no kinetic record {label_id!r}; call list_kinetic_records") from None
        out: dict[str, Any] = {
            "label_id": record.label_id, "enzyme": record.enzyme,
            "variant": record.variant, "substrate": record.substrate,
            "record_status": record.record_status, "label_type": record.label_type,
            "tier": record.tier, "lineage": record.lineage,
            "assay_group": record.assay_group,
            "pH": record.ph, "temperature_C": record.temperature_c,
            "assay_cofactor": record.assay_cofactor,
            "quality_flags": record.quality_flags,
            "source_id": record.source_id, "source_location": record.source_location,
            "has_matched_complex": record.has_matched_complex,
            "experimental_complex_pdb_ids": list(record.experimental_complex_pdb_ids),
            "available_quantities": list(record.point_quantities()),
            "withheld": dict(record.withheld),
            "cite": cite("kred_reference", record.label_id),
        }
        for quantity in ("kcat", "km", "efficiency_reported", "efficiency_derived"):
            try:
                out[quantity] = record.require(quantity)
            except EAgentError as exc:
                out[quantity] = None
                out.setdefault("refusals", {})[quantity] = str(exc)
        return out

    def structure_entry(pdb_id: str) -> dict[str, Any]:
        """One audited PDB entry: its grade, density, cofactor state and links."""
        try:
            entry = rs.structure(pdb_id)
        except KeyError:
            raise ToolRefusal(
                f"no structure {pdb_id!r} in the set; it holds "
                f"{', '.join(s.pdb_id for s in rs.structures)}") from None
        return {"pdb_id": entry.pdb_id, "enzyme": entry.enzyme,
                "variant": entry.variant, "geometry_use": entry.geometry_use,
                "role": entry.role, "lineage": entry.lineage,
                "resolution_A": entry.resolution_a,
                "bound_reaction_ligand": entry.bound_reaction_ligand,
                "experimental_state": entry.experimental_state,
                "scoring_chain": entry.scoring_chain,
                "altlocs": list(entry.altlocs),
                "reaction_ligand_ccds": list(entry.reaction_ligand_ccds),
                "cofactor_state": entry.cofactor_state,
                "ligand_rscc": list(entry.ligand_rscc),
                "cofactor_rscc": entry.cofactor_rscc,
                "linked_label_ids": list(entry.linked_label_ids),
                "selection_notes": entry.selection_notes,
                "cite": cite("kred_reference", entry.pdb_id)}

    def audit_verdict() -> dict[str, Any]:
        """The committed eligibility and calibration verdict, per scenario."""
        path = base.parent.parent.parent.parent / "docs" / "results" \
            / "kred_reference_audit.json"
        if not path.is_file():
            raise ToolRefusal("the audit result is not in this checkout; run "
                              "`eagent reference audit --out docs/results`")
        report = _json.loads(path.read_text(encoding="utf-8"))
        return {"scenarios": [
                    {"scenario": s["scenario"],
                     "eligible_entries": s["eligible_entries"],
                     "eligible_lineages": s["eligible_lineages"],
                     "n_independent_actives": s["n_independent_actives"],
                     "n_known_inactives": s["n_known_inactives"],
                     "any_constraint_calibrated": any(
                         rec["meets_policy"] for cal in s["calibrations"]
                         for rec in cal["records"].values())}
                    for s in report["scenarios"]],
                "what_a_calibration_needs": report["what_a_calibration_needs"],
                "cite": cite("kred_reference_audit", "scenarios")}

    def activity_summary() -> dict[str, Any]:
        """The 2026 ortholog activity data: counts and what each absence means."""
        return {"counts": activity.counts(),
                "absences": {
                    "not_assayed": ("insoluble constructs; never assayed and not "
                                    "catalytically negative"),
                    "below_detection": ("soluble, no product detected; "
                                        "left-censored, not a measured zero"),
                    "missing_ee": "0/0 and undefined, not zero"},
                "cite": cite("kred_activity", "summary")}

    def activity_endpoint(enzyme_id: str, substrate_id: str) -> dict[str, Any]:
        """One ortholog on one substrate: products, ee, and the censoring."""
        if substrate_id not in SUBSTRATES:
            raise ToolRefusal(f"{substrate_id!r} is not assayed; the substrates "
                              f"are {sorted(SUBSTRATES)}")
        try:
            record = activity.record(enzyme_id)
        except KeyError:
            raise ToolRefusal(f"no construct {enzyme_id!r} in the activity set") \
                from None
        if substrate_id not in ENDPOINT_SUBSTRATES:
            return {"enzyme_id": enzyme_id, "substrate_id": substrate_id,
                    "label_type": SUBSTRATES[substrate_id].label_type,
                    "relative_depletion_slope": record.relative_depletion_slope,
                    "withheld": dict(record.withheld),
                    "note": ("1a carries a relative depletion slope, not a "
                             "product concentration; the two do not pool"),
                    "cite": cite("kred_activity", enzyme_id)}
        endpoint = record.endpoint(substrate_id)
        out: dict[str, Any] = {
            "enzyme_id": enzyme_id, "substrate_id": substrate_id,
            "solubly_expressed": record.solubly_expressed,
            "status": endpoint.status, "label_type": "product_concentration",
            "product_r_mM_as_printed": endpoint.product_r_mM,
            "product_s_mM_as_printed": endpoint.product_s_mM,
            "ee_reported": endpoint.ee_reported,
            "ee_from_products": endpoint.ee_from_products,
            "ee_residual": endpoint.ee_residual,
            "withheld": dict(endpoint.withheld),
            "substrate_smiles_quarantined": SUBSTRATES[substrate_id].quarantined,
            "cite": cite("kred_activity", f"{enzyme_id}/{substrate_id}"),
        }
        try:
            out["total_product_mM"] = endpoint.require("total_product_mM")
        except EAgentError as exc:
            out["total_product_mM"] = None
            out["refusal"] = str(exc)
        return out

    def activity_independence_groups() -> dict[str, Any]:
        """How many independent groups the soluble orthologs form."""
        try:
            return {**activity_independence(activity),
                    "cite": cite("kred_activity", "independence")}
        except EAgentError as exc:
            raise ToolRefusal(str(exc)) from None

    def source_verification() -> dict[str, Any]:
        """Which kinetic numbers matched the printed table they came from."""
        path = base / "verification" / "results.json"
        if not path.is_file():
            raise ToolRefusal("no source-verification record in this checkout")
        report = _json.loads(path.read_text(encoding="utf-8"))
        return {"summary": report["summary"],
                "documents": {k: {"url": v["url"], "primary": v["primary"]}
                              for k, v in report["documents"].items()},
                "cite": cite("kred_source_verification", "summary")}

    return [
        ReadOnlyTool("reference_summary", reference_summary.__doc__ or "", {},
                     reference_summary),
        ReadOnlyTool("list_kinetic_records", list_kinetic_records.__doc__ or "",
                     {"tier": "optional: core, secondary, sensitivity or nd"},
                     list_kinetic_records),
        ReadOnlyTool("list_structure_entries", list_structure_entries.__doc__ or "",
                     {}, list_structure_entries),
        ReadOnlyTool("list_activity_constructs", list_activity_constructs.__doc__ or "",
                     {}, list_activity_constructs),
        ReadOnlyTool("kinetic_record", kinetic_record.__doc__ or "",
                     {"label_id": "the record's label id"}, kinetic_record),
        ReadOnlyTool("structure_entry", structure_entry.__doc__ or "",
                     {"pdb_id": "a four-character PDB id in the set"},
                     structure_entry),
        ReadOnlyTool("audit_verdict", audit_verdict.__doc__ or "", {},
                     audit_verdict),
        ReadOnlyTool("activity_summary", activity_summary.__doc__ or "", {},
                     activity_summary),
        ReadOnlyTool("activity_endpoint", activity_endpoint.__doc__ or "",
                     {"enzyme_id": "a construct id, e.g. Ssal-KRED or Ort-RDM-1",
                      "substrate_id": "1a, 2a, 3a, 4a or 5a"},
                     activity_endpoint),
        ReadOnlyTool("activity_independence_groups",
                     activity_independence_groups.__doc__ or "", {},
                     activity_independence_groups),
        ReadOnlyTool("source_verification", source_verification.__doc__ or "", {},
                     source_verification),
    ]
