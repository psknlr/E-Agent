"""The scientific interfaces, imported lazily so a partial tree still works.

Every interface in this package subclasses
:class:`~eagent.tools.base.ScientificInterface` and returns the uniform
:class:`~eagent.envelope.ToolResult`. The controller routes on that envelope
and never on the chemistry, which is what keeps orchestration free of science.

Why the imports are lazy
------------------------
The ten interfaces are written independently and land at different times.
A package ``__init__`` that imported all of them eagerly would mean that one
unwritten module breaks ``import eagent.tools`` for everybody, and that
importing a single interface drags in every other interface's dependencies --
including the heavy structure and geometry code a reaction-normalisation step
has no use for. So this module imports nothing at import time:
``from eagent.tools import NormalizeReaction`` resolves through
:func:`__getattr__`, and a missing sibling raises a plain ``AttributeError``
naming the module that is absent rather than a confusing ``ImportError`` from
three frames deep.

:func:`build_registry` is the companion for a run: it loads the interfaces that
exist, reports the ones that do not, and refuses to guess at what a missing
step would have produced.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # pragma: no cover - for type checkers only, never at runtime
    from .base import InterfaceRegistry, ScientificInterface

#: Interface name -> module basename inside this package: the ten steps of the
#: protocol, declared so a planner can see the full set even while some modules
#: are still unwritten. It is a *hint*, not the source of truth. Module discovery
#: (:func:`discover_interfaces`) is authoritative, because an interface whose
#: module was named something else would otherwise vanish from a run while this
#: table went on implying the step had been planned for.
#:
#: The keys are the names of
#: :data:`eagent.harness.registry.PROTOCOL_ORDER`, in that order, and the test
#: suite asserts the two tables hold the same set. They drifted once: this table
#: kept six planning-era names (``map_catalytic_site``, ``build_complex``,
#: ``screen_geometry``, ``rank_candidates``, ``design_variants``, ``plan_batch``)
#: that no module ever implemented, so :func:`available_interfaces` reported the
#: roster as 4 of 10 and :func:`build_registry` attached six phantom gaps to
#: ``missing_interfaces`` -- a planner reading either one would have believed
#: six implemented steps were absent and six absent steps were planned.
INTERFACE_MODULES: dict[str, str] = {
    "normalize_reaction": "normalize_reaction",
    "retrieve_evidence": "retrieve_evidence",
    "mine_sequences": "mine_sequences",
    "annotate_family": "annotate_family",
    "prepare_structures": "prepare_structures",
    "model_complexes": "model_complexes",
    "evaluate_catalysis": "evaluate_catalysis",
    "select_batch": "select_batch",
    "ingest_results": "ingest_results",
    "propose_mutations": "propose_mutations",
}

#: Exported symbol -> (module basename, attribute). Kept explicit rather than
#: discovered by scanning, so a typo in an interface module cannot quietly
#: change this package's public surface.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ScientificInterface": ("base", "ScientificInterface"),
    "InterfaceRegistry": ("base", "InterfaceRegistry"),
    "NormalizeReaction": ("normalize_reaction", "NormalizeReaction"),
    "SubstrateChemotype": ("normalize_reaction", "SubstrateChemotype"),
    "SUBSTRATE_CLASS_SCAFFOLD": ("normalize_reaction", "SUBSTRATE_CLASS_SCAFFOLD"),
    "RetrieveEvidence": ("retrieve_evidence", "RetrieveEvidence"),
}

__all__ = ["INTERFACE_MODULES", "available_interfaces", "discover_interfaces",
           "load_interface", "build_registry", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str) -> Any:
    """Resolve a public name on first use, or explain precisely what is missing."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        # Allow `from eagent.tools import <module>` for any sibling module too.
        try:
            return importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as exc:
            raise AttributeError(
                f"module 'eagent.tools' has no attribute '{name}'; known "
                f"exports: {', '.join(sorted(_LAZY_EXPORTS))}"
            ) from exc
    module_name, attr = target
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ModuleNotFoundError as exc:
        raise AttributeError(
            f"'{name}' lives in eagent/tools/{module_name}.py, which does not "
            f"exist yet ({exc})"
        ) from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(
            f"eagent/tools/{module_name}.py exists but defines no '{attr}'"
        ) from exc


def __dir__() -> list[str]:
    return sorted(set(__all__))


def available_interfaces() -> dict[str, bool]:
    """Which of the ten interface modules can be imported in this tree.

    Uses :func:`importlib.util.find_spec` rather than importing, so asking the
    question costs nothing and cannot fail because of a sibling's heavy or
    broken dependency.
    """
    from importlib.util import find_spec

    out: dict[str, bool] = {}
    for iface, module_name in INTERFACE_MODULES.items():
        try:
            out[iface] = find_spec(f"{__name__}.{module_name}") is not None
        except (ImportError, ValueError):
            out[iface] = False
    return out


def _interface_classes(module: Any) -> list[type]:
    """Interface classes a module defines, excluding ones it merely imported.

    The ``__module__`` check matters: without it, a module that imports a
    sibling's interface would re-register it, and the registry would refuse the
    duplicate or, worse, record the wrong provenance for a step.
    """
    from .base import ScientificInterface

    return [
        obj for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, ScientificInterface)
        and obj is not ScientificInterface
        and getattr(obj, "__module__", "") == module.__name__
        and getattr(obj, "name", "unnamed") != "unnamed"
    ]


def discover_interfaces() -> tuple[dict[str, type], list[str]]:
    """Import every module in this package and collect the interfaces found.

    Returns ``(by_name, problems)``. Discovery is authoritative over
    :data:`INTERFACE_MODULES`, because a module whose filename does not match
    its interface name would otherwise be invisible. A module that fails to
    import is reported in ``problems`` rather than aborting discovery: one
    unwritten or broken step must not make the other nine unreachable, and the
    failure has to stay visible so it can be recorded in the run manifest.
    """
    import pkgutil

    found: dict[str, type] = {}
    problems: list[str] = []
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("_") or info.name == "base":
            continue
        try:
            module = importlib.import_module(f".{info.name}", __name__)
        except Exception as exc:          # a broken sibling, reported not hidden
            problems.append(f"eagent/tools/{info.name}.py failed to import: "
                            f"{type(exc).__name__}: {exc}")
            continue
        for cls in _interface_classes(module):
            previous = found.get(cls.name)
            if previous is not None and previous is not cls:
                problems.append(
                    f"interface name '{cls.name}' is claimed by both "
                    f"{previous.__module__} and {cls.__module__}; the manifest "
                    f"could not say which one ran")
                continue
            found[cls.name] = cls
    return found, problems


def load_interface(name: str) -> "ScientificInterface":
    """Instantiate one interface by its protocol name.

    Tries the declared module first and falls back to discovery, then raises
    when there is no match or more than one. Guessing which class was meant is
    exactly the kind of convenience that makes a run execute a different step
    than the manifest records.
    """
    module_name = INTERFACE_MODULES.get(name, name)
    try:
        module = importlib.import_module(f".{module_name}", __name__)
    except ModuleNotFoundError:
        module = None

    if module is not None:
        matches = [c for c in _interface_classes(module) if c.name == name]
        if len(matches) > 1:
            raise LookupError(
                f"eagent/tools/{module_name}.py defines {len(matches)} "
                f"interfaces named '{name}'; each step is named once")
        if matches:
            return matches[0]()

    found, problems = discover_interfaces()
    if name in found:
        return found[name]()
    detail = f"; discovery also reported: {'; '.join(problems)}" if problems else ""
    raise LookupError(
        f"no ScientificInterface named '{name}' in eagent.tools "
        f"(looked in eagent/tools/{module_name}.py and in every module of the "
        f"package){detail}")


def build_registry(names: Iterable[str] | None = None, *,
                   strict: bool = False) -> "InterfaceRegistry":
    """Register the interfaces that exist; report, never fake, the ones that do not.

    With ``strict=False`` (the default) a missing step is skipped so a partial
    tree is still runnable up to the first gap, and the gaps are attached to the
    registry as ``missing_interfaces``. With ``strict=True`` it raises, which is
    what a production run wants: silently running nine of ten steps produces a
    batch plan that looks complete and is not.
    """
    from .base import InterfaceRegistry

    registry = InterfaceRegistry()
    missing: list[str] = []

    if names is None:
        found, problems = discover_interfaces()
        missing.extend(problems)
        for iface_name in sorted(found):
            registry.register(found[iface_name]())
        for declared in INTERFACE_MODULES:
            if declared not in found:
                missing.append(f"{declared}: not implemented in eagent.tools")
    else:
        for iface_name in names:
            try:
                registry.register(load_interface(iface_name))
            except (ModuleNotFoundError, LookupError) as exc:
                if strict:
                    raise
                missing.append(f"{iface_name}: {exc}")

    if strict and missing:
        raise LookupError("interfaces missing from eagent.tools: "
                          + "; ".join(missing))
    #: Attached rather than logged so a caller can put the gaps in the manifest.
    registry.missing_interfaces = missing  # type: ignore[attr-defined]
    return registry
