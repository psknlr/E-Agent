"""Contract shared by the ten scientific interfaces.

Every interface is a class with a stable ``name``, a declared set of required
task fields and approval gates, and a ``run`` returning a :class:`ToolResult`.
The controller only ever sees this contract, which is what keeps the
orchestration layer free of chemistry.
"""

from __future__ import annotations

import abc
import traceback
from typing import Any, ClassVar

from ..context import RunContext
from ..envelope import Provenance, Severity, Status, ToolResult
from ..errors import (
    ApprovalRequiredError, EAgentError, LicenseError, ToolUnavailableError,
    UnresolvedFieldError,
)
from ..provenance import utc_now


class ScientificInterface(abc.ABC):
    """Base class for an interface in the protocol."""

    #: Stable identifier used in the manifest and by the controller.
    name: ClassVar[str] = "unnamed"
    #: One-line description shown in the CLI.
    description: ClassVar[str] = ""
    #: Task fields that must be resolved before this step may run.
    required_fields: ClassVar[tuple[str, ...]] = ()
    #: Approval gates that must be cleared first.
    required_approvals: ClassVar[tuple[str, ...]] = ()
    #: Interfaces whose artifacts this one consumes.
    depends_on: ClassVar[tuple[str, ...]] = ()
    #: Version of this implementation, recorded in provenance.
    version: ClassVar[str] = "0.1.0"

    # -- subclass hook -----------------------------------------------------
    @abc.abstractmethod
    def execute(self, ctx: RunContext, **kwargs: Any) -> ToolResult:
        """Do the work. Raise or return ``failed``; never invent a result."""

    # -- fixed wrapper -----------------------------------------------------
    def run(self, ctx: RunContext, **kwargs: Any) -> ToolResult:
        """Check preconditions, execute, and stamp provenance.

        Precondition failures come back as structured results rather than
        tracebacks, so the controller can route them to the operator.
        """
        started = utc_now()
        try:
            self._check_preconditions(ctx)
        except UnresolvedFieldError as e:
            r = ToolResult.failure(self.name, str(e), code="unresolved_fields")
            r.add_uncertainty(
                "unresolved_fields",
                f"Which values should fill: {', '.join(e.paths)}?",
                affects=[self.name], resolvable_by="operator input",
            )
            r.add_next("resolve_fields", "Operator must supply these values",
                       {"paths": e.paths}, requires_human=True)
            return self._stamp(r, ctx, started)
        except ApprovalRequiredError as e:
            r = ToolResult.failure(self.name, str(e), code="approval_required")
            r.add_next("request_approval", "Human decision point not yet cleared",
                       {"gate": e.gate}, requires_human=True)
            return self._stamp(r, ctx, started)

        try:
            result = self.execute(ctx, **kwargs)
        except ToolUnavailableError as e:
            result = ToolResult.failure(self.name, str(e), code="tool_unavailable")
            result.add_uncertainty(
                "tool_unavailable",
                f"{e.tool} is not installed; this step produced no result rather "
                f"than a substitute heuristic.",
                affects=[self.name], resolvable_by="install the tool",
            )
        except LicenseError as e:
            result = ToolResult.failure(self.name, str(e), code="license_blocked")
        except EAgentError as e:
            result = ToolResult.failure(self.name, str(e),
                                        code=type(e).__name__.lower())
        except Exception as e:  # unexpected: record the traceback, do not mask
            result = ToolResult.failure(
                self.name, f"unhandled {type(e).__name__}: {e}", code="internal_error"
            )
            result.data["traceback"] = traceback.format_exc()

        return self._stamp(result, ctx, started)

    # -- helpers -----------------------------------------------------------
    def _check_preconditions(self, ctx: RunContext) -> None:
        missing = [p for p in self.required_fields
                   if _unset(ctx.task, p)]
        if missing and ctx.policy.strict:
            raise UnresolvedFieldError(missing, self.name)
        for gate in self.required_approvals:
            ctx.require_approval(gate, f"needed by interface '{self.name}'")

    def _stamp(self, result: ToolResult, ctx: RunContext, started: str) -> ToolResult:
        prov = result.provenance or Provenance(tool=self.name)
        prov.tool = self.name
        if prov.tool_version in ("unknown", ""):
            prov.tool_version = self.version
        prov.started_at = prov.started_at or started
        prov.finished_at = prov.finished_at or utc_now()
        prov.random_seed = prov.random_seed if prov.random_seed is not None \
            else ctx.seed_for(self.name)
        result.provenance = prov
        return result

    # -- convenience for subclasses ---------------------------------------
    def ok(self, ctx: RunContext, **data: Any) -> ToolResult:
        return ToolResult(status=Status.SUCCESS,
                          provenance=Provenance(tool=self.name,
                                                tool_version=self.version),
                          data=dict(data))

    def partial(self, ctx: RunContext, message: str, **data: Any) -> ToolResult:
        r = ToolResult(status=Status.PARTIAL,
                       provenance=Provenance(tool=self.name,
                                             tool_version=self.version),
                       data=dict(data), message=message)
        r.add_flag("partial_result", Severity.WARN, message)
        return r


def _unset(task: Any, dotted: str) -> bool:
    cur = task
    for part in dotted.split("."):
        if cur is None:
            return True
        cur = getattr(cur, part, None)
    if cur is None:
        return True
    if isinstance(cur, (list, tuple, dict, str)) and len(cur) == 0:
        return True
    return False


class InterfaceRegistry:
    """Name -> interface instance, with dependency-order resolution."""

    def __init__(self) -> None:
        self._items: dict[str, ScientificInterface] = {}

    def register(self, iface: ScientificInterface) -> ScientificInterface:
        if iface.name in self._items:
            raise ValueError(f"duplicate interface name: {iface.name}")
        self._items[iface.name] = iface
        return iface

    def get(self, name: str) -> ScientificInterface:
        if name not in self._items:
            raise KeyError(f"unknown interface: {name}; "
                           f"known: {sorted(self._items)}")
        return self._items[name]

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self):
        return iter(self._items.values())

    def names(self) -> list[str]:
        return sorted(self._items)

    def topological_order(self) -> list[str]:
        """Dependency order; raises on a cycle."""
        order: list[str] = []
        state: dict[str, int] = {}

        def visit(n: str) -> None:
            if state.get(n) == 2:
                return
            if state.get(n) == 1:
                raise ValueError(f"dependency cycle at {n}")
            state[n] = 1
            for dep in self._items[n].depends_on:
                if dep in self._items:
                    visit(dep)
            state[n] = 2
            order.append(n)

        for n in sorted(self._items):
            visit(n)
        return order
