"""Enzyme function mining and substrate-directed engineering agent.

Importing this package must stay cheap. ``import eagent`` is executed by the
console script on every invocation -- including ``eagent --help`` -- and by
any tool that only wants to read :data:`__version__`. An eager
``from .schemas import *`` here would pull pydantic, the six-layer data
model, the structure parser and the geometry tables into a process that was
about to print four lines of help text, and the cost would be paid again by
every test module that touches any corner of the tree.

So nothing is imported at module scope. Every public name is resolved on
first use by :func:`__getattr__`, which means a module that is broken or
absent raises an ``AttributeError`` naming it, instead of making the whole
package unimportable. The same discipline is already applied in
``eagent.tools``; this module extends it to the top level.

``__version__`` is resolved the same way: from the installed distribution
metadata when there is one, and otherwise from :data:`_DECLARED_VERSION`,
which mirrors ``pyproject.toml``. It is never guessed -- an unknown version
in a run manifest is worse than no manifest, because it looks like a fact.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

#: The version declared in ``pyproject.toml``. Used only when this tree is not
#: installed as a distribution (a source checkout run with ``PYTHONPATH=src``),
#: in which case there is no metadata to read and this is the single place the
#: number is written down.
_DECLARED_VERSION = "0.1.0"

#: Public name -> (submodule, attribute). Written out rather than discovered by
#: scanning, so a rename inside a submodule shows up as a failing import here
#: instead of quietly changing what ``from eagent import ...`` means.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # typed failures
    "ApprovalRequiredError": ("errors", "ApprovalRequiredError"),
    "CircularEvidenceError": ("errors", "CircularEvidenceError"),
    "EAgentError": ("errors", "EAgentError"),
    "FabricationGuardError": ("errors", "FabricationGuardError"),
    "LicenseError": ("errors", "LicenseError"),
    "ProvenanceError": ("errors", "ProvenanceError"),
    "TemplateError": ("errors", "TemplateError"),
    "ToolUnavailableError": ("errors", "ToolUnavailableError"),
    "UnresolvedFieldError": ("errors", "UnresolvedFieldError"),
    # the uniform envelope
    "Artifact": ("envelope", "Artifact"),
    "NextAction": ("envelope", "NextAction"),
    "Provenance": ("envelope", "Provenance"),
    "QCFlag": ("envelope", "QCFlag"),
    "Severity": ("envelope", "Severity"),
    "Status": ("envelope", "Status"),
    "ToolResult": ("envelope", "ToolResult"),
    "Uncertainty": ("envelope", "Uncertainty"),
    # run context and provenance
    "ExecutionPolicy": ("context", "ExecutionPolicy"),
    "RunContext": ("context", "RunContext"),
    "RunManifest": ("provenance", "RunManifest"),
    "StepRecord": ("provenance", "StepRecord"),
    "sequence_hash": ("provenance", "sequence_hash"),
    "sha256_file": ("provenance", "sha256_file"),
    "sha256_obj": ("provenance", "sha256_obj"),
    "utc_now": ("provenance", "utc_now"),
    # the task model most callers start from
    "TaskSpec": ("schemas", "TaskSpec"),
    # deliverables
    "assemble_bundle": ("deliverables.bundle", "assemble_bundle"),
    "verify_bundle": ("deliverables.bundle", "verify_bundle"),
    # command line
    "main": ("cli", "main"),
}

#: Submodules reachable as attributes without importing them up front, so that
#: ``import eagent; eagent.harness`` works while ``import eagent`` alone still
#: costs nothing.
_SUBMODULES: tuple[str, ...] = (
    "cli", "connectors", "context", "datalayer", "deliverables", "envelope",
    "errors", "eval", "harness", "provenance", "schemas", "science", "tools",
)

__all__ = ["__version__", *sorted(_LAZY_EXPORTS), *sorted(_SUBMODULES)]

if TYPE_CHECKING:  # pragma: no cover - type checkers only, never at runtime
    from .context import ExecutionPolicy, RunContext
    from .envelope import (
        Artifact, NextAction, Provenance, QCFlag, Severity, Status, ToolResult,
        Uncertainty,
    )
    from .errors import (
        ApprovalRequiredError, CircularEvidenceError, EAgentError,
        FabricationGuardError, LicenseError, ProvenanceError, TemplateError,
        ToolUnavailableError, UnresolvedFieldError,
    )
    from .provenance import (
        RunManifest, StepRecord, sequence_hash, sha256_file, sha256_obj, utc_now,
    )
    from .schemas import TaskSpec


def _resolve_version() -> str:
    """Installed version, or the declared one, never an invented one.

    ``importlib.metadata`` is imported here rather than at module scope
    because reading the distribution metadata walks ``sys.path`` and a caller
    that never asks for the version should not pay for it.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - importlib.metadata is stdlib on 3.11
        return _DECLARED_VERSION
    try:
        return version("eagent")
    except PackageNotFoundError:
        # A source checkout on PYTHONPATH has no metadata. That is normal, and
        # the declared number is a fact about this tree rather than a guess.
        return _DECLARED_VERSION
    except Exception:  # pragma: no cover - a broken metadata directory
        return _DECLARED_VERSION


def __getattr__(name: str) -> Any:
    """Resolve a public name on first use, or say exactly what is missing."""
    if name == "__version__":
        resolved = _resolve_version()
        globals()["__version__"] = resolved        # resolved once per process
        return resolved
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        if name in _SUBMODULES:
            try:
                return importlib.import_module(f".{name}", __name__)
            except ModuleNotFoundError as exc:
                # A submodule listed here but absent from this tree is an
                # AttributeError, not an ImportError three frames deep: the
                # caller asked this package for a name it does not have.
                raise AttributeError(
                    f"eagent/{name}/ is listed as a submodule of this package "
                    f"but could not be imported ({exc})") from exc
        raise AttributeError(
            f"module 'eagent' has no attribute '{name}'; exported names are "
            f"{', '.join(sorted(_LAZY_EXPORTS))} and the submodules "
            f"{', '.join(_SUBMODULES)}"
        )
    module_name, attr = target
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ModuleNotFoundError as exc:
        raise AttributeError(
            f"'{name}' lives in eagent/{module_name.replace('.', '/')}.py, "
            f"which could not be imported ({exc})"
        ) from exc
    try:
        value = getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(
            f"eagent/{module_name.replace('.', '/')}.py exists but defines no "
            f"'{attr}'"
        ) from exc
    globals()[name] = value                        # subsequent lookups are free
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__))
