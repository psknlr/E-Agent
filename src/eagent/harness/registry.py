"""Assembling the ten scientific interfaces into one registry.

``eagent.tools.build_registry`` can already discover whatever interface
modules happen to exist. That is the right behaviour for a partially written
tree and the wrong behaviour for a run: discovery answers "what is here",
and a run needs "is everything the protocol requires here". A run that
silently executes nine of ten steps produces a batch plan that looks complete,
and nothing downstream can tell that the geometry step never ran.

So this module names the ten steps explicitly in :data:`PROTOCOL_INTERFACES`
and refuses, by default, to build a registry that is missing one.
:func:`build_interface_registry` is what the controller calls; the pipeline
order it should walk is :data:`PROTOCOL_ORDER`, while
:func:`topological_order` answers the different question of what *must*
precede what according to the interfaces' own ``depends_on``.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..tools import build_registry
from ..tools.base import InterfaceRegistry, ScientificInterface

__all__ = [
    "PROTOCOL_INTERFACES",
    "PROTOCOL_ORDER",
    "build_interface_registry",
    "dependency_problems",
    "registry_report",
    "topological_order",
]

#: The ten interfaces of the protocol, in the order the pilot walks them.
#: This is the protocol's own order, which is a scientific statement (evidence
#: before mining, mechanism before engineering) and is therefore written down
#: rather than derived from ``depends_on``, which only records data flow.
PROTOCOL_ORDER: tuple[str, ...] = (
    "normalize_reaction",
    "retrieve_evidence",
    "mine_sequences",
    "annotate_family",
    "prepare_structures",
    "model_complexes",
    "evaluate_catalysis",
    "select_batch",
    "ingest_results",
    "propose_mutations",
)

#: Same set, as a frozen set for membership tests.
PROTOCOL_INTERFACES: frozenset[str] = frozenset(PROTOCOL_ORDER)


def build_interface_registry(
    names: Iterable[str] | None = None, *, strict: bool = True,
) -> InterfaceRegistry:
    """Register the protocol's interfaces.

    ``strict=True`` (the default) raises when any of the ten is missing, which
    is what a real run wants. ``strict=False`` is for a tree under
    construction: the gaps are then readable on the registry as
    ``missing_interfaces`` and belong in the manifest, never swallowed.
    """
    wanted = tuple(names) if names is not None else PROTOCOL_ORDER
    registry = build_registry(wanted, strict=strict)
    if not hasattr(registry, "missing_interfaces"):
        registry.missing_interfaces = []     # type: ignore[attr-defined]
    return registry


def topological_order(registry: InterfaceRegistry | None = None) -> list[str]:
    """Dependency order from the interfaces' declared ``depends_on``.

    Not the same thing as :data:`PROTOCOL_ORDER`: ``depends_on`` says which
    artifacts a step consumes, so it constrains ordering without fixing it.
    A controller that walked this order instead of the protocol order would be
    free to mine sequences before the reaction spec was normalised, because no
    artifact links those two steps even though the whole search is defined by
    the spec.
    """
    reg = registry if registry is not None else build_interface_registry()
    return reg.topological_order()


def dependency_problems(registry: InterfaceRegistry) -> list[str]:
    """Declared dependencies that the registry cannot satisfy.

    A step whose ``depends_on`` names an interface that is not registered will
    run anyway -- nothing in the base class checks -- and will simply find no
    input, which it reports as "no candidates supplied". That reads like an
    empty search result rather than like a missing step, so the mismatch is
    surfaced here before the run starts.
    """
    known = set(registry.names())
    problems: list[str] = []
    for iface in registry:
        for dep in iface.depends_on:
            if dep not in known:
                problems.append(
                    f"interface '{iface.name}' declares a dependency on "
                    f"'{dep}', which is not registered; its inputs would "
                    f"silently be empty")
    return problems


def registry_report(registry: InterfaceRegistry) -> dict[str, Any]:
    """A manifest-ready description of what will actually run."""
    interfaces: list[dict[str, Any]] = []
    for name in registry.names():
        iface: ScientificInterface = registry.get(name)
        interfaces.append({
            "name": iface.name,
            "version": iface.version,
            "depends_on": list(iface.depends_on),
            "required_fields": list(iface.required_fields),
            "required_approvals": list(iface.required_approvals),
        })
    return {
        "registered": registry.names(),
        "protocol_order": list(PROTOCOL_ORDER),
        "topological_order": registry.topological_order(),
        "missing": list(getattr(registry, "missing_interfaces", []) or []),
        "unregistered_protocol_steps": sorted(
            PROTOCOL_INTERFACES - set(registry.names())),
        "dependency_problems": dependency_problems(registry),
        "interfaces": interfaces,
    }
