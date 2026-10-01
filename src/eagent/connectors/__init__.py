"""Data-layer connectors: cache-first access to the ten public resources.

Everything a run reads from the outside world passes through
:mod:`eagent.connectors.base`. The package exists so that three properties hold
for every resource rather than for whichever one the author remembered:

* **Offline is the default.** A connector consults a :class:`~.base.FileCache`
  and, when the record is not there and the policy forbids the network, returns
  a structured miss naming exactly what a human would have to place where. No
  code path in this package produces database content.
* **Layers do not substitute for one another.** :data:`~.base.CONNECTOR_REGISTRY`
  records, as data, what each resource answers and what it must never be used to
  answer, and :func:`~.base.substitution` turns that into a verdict a planner can
  act on.
* **Evidence cannot be promoted by accident.** Each entry carries an
  ``evidence_strength_ceiling``; :func:`~.base.resolve_strength` caps an ingest
  at it unless a named human reviewer takes responsibility.

Concrete per-resource connectors (a UniProt client, a PDB client) are separate
modules added as they are needed. Until one exists, the resource is wired as an
:class:`~.base.OfflineConnector`, which is the honest description of a resource
whose records enter through a curated import. :func:`offline_connectors` builds
that default wiring for a whole run.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .base import (
    AMINO_ACID_ALPHABET,
    MIN_SEQUENCE_LIKE_DISTINCT_LETTERS,
    MIN_SEQUENCE_LIKE_LENGTH,
    AccessPolicy,
    CONNECTOR_REGISTRY,
    CacheIntegrityError,
    CachedResponse,
    Connector,
    ConnectorError,
    ConnectorLayer,
    ConnectorSpec,
    FileCache,
    NetworkDisabledError,
    OfflineConnector,
    ResponseStatus,
    StrengthDecision,
    SubmissionAuthorization,
    SubstitutionVerdict,
    UnauthorizedSubmissionError,
    UnknownConnectorError,
    connector_keys,
    connector_spec,
    cross_check_source_registry,
    default_cache_root,
    looks_like_biological_sequence,
    records_in,
    resolve_strength,
    specs_for_layer,
    substitution,
)

__all__ = [
    "AMINO_ACID_ALPHABET",
    "MIN_SEQUENCE_LIKE_DISTINCT_LETTERS",
    "MIN_SEQUENCE_LIKE_LENGTH",
    "AccessPolicy",
    "CONNECTOR_REGISTRY",
    "CacheIntegrityError",
    "CachedResponse",
    "Connector",
    "ConnectorError",
    "ConnectorLayer",
    "ConnectorSpec",
    "FileCache",
    "NetworkDisabledError",
    "OfflineConnector",
    "ResponseStatus",
    "StrengthDecision",
    "SubmissionAuthorization",
    "SubstitutionVerdict",
    "UnauthorizedSubmissionError",
    "UnknownConnectorError",
    "connector_keys",
    "connector_spec",
    "cross_check_source_registry",
    "default_cache_root",
    "looks_like_biological_sequence",
    "offline_connectors",
    "records_in",
    "resolve_strength",
    "specs_for_layer",
    "substitution",
]


def offline_connectors(
    *, cache: FileCache | None = None,
    snapshots: Mapping[str, str] | None = None,
    access: AccessPolicy | None = None,
    keys: Iterable[str] | None = None,
) -> dict[str, OfflineConnector]:
    """Default wiring: every registered resource as a cache-only connector.

    Used when a run supplies no connectors of its own. Building them here rather
    than inside each interface keeps the decision about network access in one
    place, and means an interface cannot acquire a live client by importing a
    different module.

    ``snapshots`` pins the release string per resource. A resource absent from
    it is marked ``"unpinned"`` rather than being given a plausible release
    identifier, so the run manifest shows which parts of the evidence were not
    reproducible against a named version.
    """
    snaps = dict(snapshots or {})
    chosen = list(keys) if keys is not None else connector_keys()
    out: dict[str, OfflineConnector] = {}
    for key in chosen:
        spec = connector_spec(key)       # raises on an unknown key, by design
        out[key] = OfflineConnector(
            spec.key, spec.data_layer, cache=cache,
            snapshot=snaps.get(key), access=access,
            description=spec.display_name,
        )
    return out


def __getattr__(name: str) -> Any:
    """Lazily expose per-resource connector modules once they are written.

    ``from eagent.connectors import uniprot`` works the moment someone adds
    ``uniprot.py``, and raises a plain :class:`AttributeError` naming the
    missing module before that. Importing this package never requires a sibling
    module to exist.
    """
    import importlib

    try:
        return importlib.import_module(f".{name}", __name__)
    except ModuleNotFoundError as exc:
        raise AttributeError(
            f"module 'eagent.connectors' has no attribute '{name}' "
            f"(no eagent/connectors/{name}.py yet): {exc}"
        ) from exc
