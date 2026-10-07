"""Chemistry-layer connectors: PubChem, ChEBI, Rhea, MetaNetX, EnzymeMap.

Why this module exists
----------------------
The reaction-and-chemistry layer is where the pilot task is defined, and three
of its failure modes are unrecoverable once they have happened:

*A name becomes a stereo-defined structure.* "1-phenylethanone" resolved
through a name service yields a connectivity graph. Nothing in that lookup
decided the stereochemistry of the product, the salt form of the material in
the flask, or the protonation state that will be docked.
:class:`PubChemConnector` therefore writes its answer into the rungs of a
:class:`~eagent.datalayer.identity.ChemicalIdentityLadder` and refuses to fill
the stereo rung from a string that carries no stereo descriptor.

*A chemical class becomes a substrate.* A ChEBI class entry ("ketone") is a
set, not a compound. Docked, measured or ordered, it is nonsense.
:class:`ChEBIConnector` keeps ``class`` and ``instance`` apart and treats
"the entry does not say" as a third state, because a class silently accepted
as a substrate produces an entire campaign aimed at a set.

*An oxidation becomes evidence for a reduction.* Rhea stores a reference
direction that is a curatorial convention. :class:`RheaConnector` maps that
convention onto :class:`~eagent.schemas.record.ReactionDirection` relative to
the task's own reaction class, so a record of the alcohol oxidation comes back
marked :attr:`ReactionDirection.REVERSE_OF_TARGET` and cannot be counted as
support by :func:`~eagent.datalayer.intake.direction_check`.

Where the shared connector foundation lives
-------------------------------------------
The connector package is laid out as five topic modules, so the registry-gated
foundation that all of them need (:class:`RegistryBackedConnector`,
:class:`CuratedFileImporter`, the capability and endpoint guards) is defined
here and imported by the other four. It is placed in the chemistry module
rather than duplicated because five copies of a guard are five chances for one
of them to drift into permissiveness, and a guard that is right in four modules
is not a guard.

Three properties hold for every connector built on it:

* **The base URL is read from the registry at call time.** No URL, path
  fragment or query template is written down in this package. Every source in
  ``configs/datasources`` records ``endpoint: null`` because none has been
  connectivity-tested here, so every remote route currently refuses with
  :class:`EndpointNotEstablishedError` naming the source and its curation note.
* **Capabilities gate operations.** ``NOT_SUPPORTED`` is a refusal;
  ``UNKNOWN`` proceeds but stamps :data:`UNVERIFIED_CAPABILITY` on the result,
  so a planner can tell "this worked" from "nobody has checked that this can
  work".
* **A human-import-only source cannot be built as a connector at all.**
  :class:`RegistryBackedConnector` refuses construction for a source whose every
  registered access mode needs a person, and those sources get a
  :class:`CuratedFileImporter` whose ``fetch`` exists only to refuse.

Cache payload conventions
-------------------------
Payloads are written by a curator (or, if a route is ever established, by the
service) and are never generated here. Each connector documents the shape it
reads. The common shape is the one :func:`~eagent.connectors.base.records_in`
understands::

    {"records": [ {...}, {...} ], "database_version": "..."}

A payload of a different shape yields no records rather than an improvised
reading of it.
"""

from __future__ import annotations

import abc
import csv
import enum
import io
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

from ..datalayer.identity import (
    ChemicalIdentityLadder,
    IdentityRung,
    Representation,
    RungValue,
    stereo_descriptors,
)
from ..datalayer.intake import (
    EvidenceTier,
    ExtractionMethod,
    IntakeRecord,
    direction_check,
    ingest,
)
from ..datalayer.probe import USER_AGENT
from ..datalayer.registry import (
    AccessMode,
    CAPABILITY_NAMES,
    CapabilityState,
    DataSource,
    SourceRegistry,
)
from ..provenance import sha256_file, utc_now
from ..schemas.reaction import ReactionClass
from ..schemas.record import (
    EvidenceRef,
    EvidenceStrength,
    ExperimentRecord,
    ReactionDirection,
)
from .base import (
    CachedResponse,
    Connector,
    ConnectorError,
    ConnectorLayer,
    FileCache,
    AccessPolicy,
    NetworkDisabledError,
    RemoteCallFailedError,
    ResponseStatus,
    records_in,
)

__all__ = [
    # shared foundation
    "AccessModeRefusedError",
    "CapabilityNotSupportedError",
    "CapabilityVerdict",
    "ConnectorConfigurationError",
    "CuratedFileImporter",
    "CuratedImport",
    "CuratedImportError",
    "EndpointNotEstablishedError",
    "EndpointStatus",
    "ImportProvenance",
    "LayerSemanticsError",
    "OfflineImportOnlyError",
    "RegistryBackedConnector",
    "RejectedRow",
    "UNVERIFIED_CAPABILITY",
    "datasource_registry",
    "evidence_ref_for",
    "upstream_sources_for",
    # chemistry connectors
    "ChEBIClassStatus",
    "ChEBIConnector",
    "ChEBIEntry",
    "ChemicalClassRefusedError",
    "CrossReferenceEdge",
    "CrossReferenceNotAMergeError",
    "EnzymeMapConnector",
    "EnzymeMapReaction",
    "MetaNetXConnector",
    "MetaNetXMapping",
    "PubChemConnector",
    "PubChemResolution",
    "ReactionLevelOnlyError",
    "RheaConnector",
    "RheaDirection",
    "RheaReaction",
]


# ===========================================================================
# Shared foundation: errors
# ===========================================================================

class ConnectorConfigurationError(ConnectorError):
    """A connector was wired against a source that cannot support it.

    Raised at construction rather than at call time so that building a live
    client for a resource that is registered as human-import-only fails in the
    wiring code, where it is a one-line fix, instead of in the middle of a run
    where it looks like a transient outage.
    """


def _error_detail(exc: "urllib.error.HTTPError") -> str:
    """The service's own explanation of an HTTP error, when it gave one.

    A bare ``HTTP 400`` sends somebody to curl to find out what was wrong;
    most APIs say so in the body. Truncated, and never trusted as anything but
    text for a human.
    """
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:                                    # pragma: no cover
        return ""
    return f": {body[:240]}" if body else ""


class EndpointNotEstablishedError(NetworkDisabledError):
    """The registry records no base URL for this source, so there is no route.

    Subclasses :class:`~eagent.connectors.base.NetworkDisabledError` on purpose:
    "nobody has established a route" and "this run may not use the network" are
    the same fact from the caller's point of view -- there is no permitted way
    to reach the service -- and the base resolver already knows how to turn that
    into a structured miss rather than a stack trace.

    The error names the source and repeats its curation notes, because the fix
    is a curation task ("confirm the base URL and record it with its
    documentation citation"), not a code change. Guessing a URL here is the
    specific failure this prevents: code that calls a plausible-looking address
    either reaches nothing or reaches something nobody checked.
    """

    def __init__(self, source_id: str, curation_notes: Sequence[str] = ()) -> None:
        self.source_id = source_id
        self.curation_notes = tuple(curation_notes)
        detail = ("; ".join(self.curation_notes)
                  or "see the source's curation_notes in configs/datasources")
        super().__init__(
            f"no endpoint is established for '{source_id}': the registry records "
            f"endpoint=null because the route has not been connectivity-tested "
            f"here, and this package records no URL of its own. Curation note: "
            f"{detail}")


class CapabilityNotSupportedError(ConnectorError):
    """An operation was requested that the source is registered as lacking.

    Distinct from an unknown capability, and that distinction is the point: a
    planner told "not supported" routes around the source, while a planner told
    "unknown" schedules a curation check. Collapsing the two either hides a
    usable resource or sends code at an interface that does not exist.
    """

    def __init__(self, source_id: str, capability: str, operation: str) -> None:
        self.source_id = source_id
        self.capability = capability
        self.operation = operation
        super().__init__(
            f"'{source_id}' is registered with {capability}=not_supported, so "
            f"the operation '{operation}' is refused. This is a routing fact "
            f"from the registry, not a transient failure: re-running it will "
            f"refuse again.")


class RequestShapeNotVerifiedError(NetworkDisabledError):
    """A base URL is established, but this connector's request shape is not.

    The two are different facts and the second is the one that decides whether
    a call can be made. A connectivity probe shows that *one* request, spelled
    one way, returned the record it asked for. It says nothing about the URL
    this connector would build: the generic ``<base>/<key>`` shape is not
    Rhea's query-parameter API, and calling it would produce a 404 that the
    resolver reports as a miss -- a route silently not working, which is worse
    than a refusal.

    So a connector declares the probed capability its request builder was
    written against. Until one is declared, having an endpoint changes nothing
    about whether it may call.

    Subclasses :class:`~eagent.connectors.base.NetworkDisabledError` for the
    same reason :class:`EndpointNotEstablishedError` does: from the caller's
    side there is no permitted way to reach the service, and the resolver
    already turns that into a structured miss.
    """

    def __init__(self, source_id: str, endpoint: str,
                 verified: Sequence[str] = ()) -> None:
        self.source_id = source_id
        self.endpoint = endpoint
        self.verified_capabilities = tuple(verified)
        super().__init__(
            f"'{source_id}' records the endpoint {endpoint} and "
            + (f"a verified route for {', '.join(self.verified_capabilities)}"
               if self.verified_capabilities else "no verified route")
            + f", but this connector does not declare which probed request "
              f"shape its client was written against. A verified base URL is "
              f"not a verified request: set "
              f"{type(self).__name__.replace('Error', '')!r}'s "
              f"`verified_route_capability` on the connector once its client "
              f"has been checked against a recorded probe, or leave it "
              f"refusing")


class AccessModeRefusedError(ConnectorError):
    """The source has no access mode permitting the attempted operation."""


class OfflineImportOnlyError(AccessModeRefusedError):
    """A live call was attempted against a source that only a person can fetch.

    Raised by every importer's ``fetch``/``search``. Those methods exist solely
    to refuse: without them, calling code would get :class:`AttributeError` and
    an author would "fix" it by adding a client, which is how a browse-only web
    database acquires an imaginary REST API.
    """

    def __init__(self, source_id: str, access_modes: Sequence[AccessMode],
                 operation: str) -> None:
        self.source_id = source_id
        self.operation = operation
        modes = ", ".join(m.value for m in access_modes)
        super().__init__(
            f"'{source_id}' is registered with access modes [{modes}]: every "
            f"route needs a person. The operation '{operation}' has no "
            f"implementation and will not get one. Use "
            f"import_file(path, imported_by=...) with a curator-supplied export.")


class CuratedImportError(ConnectorError):
    """A curator-supplied file could not be admitted as it stands.

    Refuses rather than skipping the bad rows silently, because a partial
    import that reports success is indistinguishable from a complete one, and
    the missing rows are invisible for the rest of the project's life.
    """


class LayerSemanticsError(ConnectorError):
    """A record was asked to answer a question its layer cannot answer.

    The umbrella for the per-resource refusals in this package: a reaction
    definition asked for sequence-level evidence, a cross-reference asked to
    merge chemical states, a chemical class asked for a substrate structure.
    Each subclass names the specific substitution it blocks.
    """


#: Marker stamped on a result produced through a capability nobody has
#: verified. Present in ``CachedResponse.notes`` as a prefix so a planner can
#: test for it without parsing prose.
UNVERIFIED_CAPABILITY = "unverified-capability"


# ===========================================================================
# Shared foundation: registry access
# ===========================================================================

_REGISTRY_CACHE: dict[str, SourceRegistry] = {}


def datasource_registry(directory: str | Path | None = None, *,
                        reload: bool = False) -> SourceRegistry:
    """Load (and memoise) the datasource registry the connectors read.

    Memoised because every connector consults it on every call and re-parsing
    49 YAML files per query would make the offline path slower than the network
    one. ``reload`` exists so a test can point the connectors at a curated copy
    without leaking state into the next test.
    """
    from ..datalayer.registry import default_datasource_dir

    # Resolve the default here rather than caching under a "<default>" key:
    # the default honours EAGENT_DATASOURCE_DIR, so a run that repoints it
    # would otherwise be served the previous directory's registry and would
    # silently enforce the wrong ceilings and capabilities.
    chosen = Path(directory) if directory is not None else default_datasource_dir()
    key = str(chosen.resolve())
    if reload or key not in _REGISTRY_CACHE:
        _REGISTRY_CACHE[key] = SourceRegistry.from_directory(chosen)
    return _REGISTRY_CACHE[key]


def upstream_sources_for(source_id: str, registry: SourceRegistry) -> tuple[str, ...]:
    """The source plus everything it re-integrates, for ``EvidenceRef``.

    Carried on every record this package emits so that
    :mod:`eagent.datalayer.lineage` can collapse four databases republishing one
    curated measurement into one independent measurement. Without it, breadth of
    coverage reads as corroboration, which is the error that makes a weak
    candidate look confirmed.
    """
    return tuple(sorted(registry.lineage(source_id)))


def evidence_ref_for(source: DataSource, identifier: str, *,
                     registry: SourceRegistry,
                     locator: str | None = None,
                     database_version: str | None = None,
                     retrieved_at: str | None = None,
                     strength: EvidenceStrength | None = None,
                     quote: str | None = None) -> EvidenceRef:
    """Build the evidence pointer for one retrieved record.

    ``strength`` defaults to the weakest rung rather than to the source's
    ceiling. The ceiling is a cap on what a human reviewer could later justify;
    using it as a default would stamp the strongest permitted label on rows that
    assert the least, which is the upward guess
    :mod:`eagent.datalayer.intake` exists to prevent.

    ``license`` is copied from the registry verbatim, including ``None``: an
    unknown licence travels with the record so redistribution terms are decided
    where the record is used, not rediscovered from memory.
    """
    return EvidenceRef(
        source_type="database",
        identifier=identifier,
        locator=locator,
        strength=strength or EvidenceStrength.ANNOTATION_ONLY,
        extracted_by="connector",
        retrieved_at=retrieved_at,
        database_version=database_version,
        source_record_id=identifier,
        source_id=source.id,
        license=source.license,
        upstream_sources=list(upstream_sources_for(source.id, registry)),
    )


# ===========================================================================
# Shared foundation: capability and endpoint verdicts
# ===========================================================================

@dataclass(frozen=True)
class CapabilityVerdict:
    """Whether one registered capability permits one operation.

    Returned instead of a bare bool so the three states survive the call.
    ``allowed and unverified`` is the ``UNKNOWN`` case: the operation proceeds
    because refusing every unchecked route would disable most of the registry,
    but the result is marked so nothing downstream reads it as a verified
    retrieval.
    """

    source_id: str
    capability: str
    state: CapabilityState
    operation: str

    @property
    def allowed(self) -> bool:
        """False only for ``NOT_SUPPORTED``; ``UNKNOWN`` is allowed-but-marked."""
        return self.state is not CapabilityState.NOT_SUPPORTED

    @property
    def unverified(self) -> bool:
        return self.state is CapabilityState.UNKNOWN

    @property
    def reason(self) -> str:
        if self.state is CapabilityState.SUPPORTED:
            return (f"'{self.source_id}' documents {self.capability}=supported "
                    f"for '{self.operation}' (documented, not connectivity-tested)")
        if self.state is CapabilityState.NOT_SUPPORTED:
            return (f"'{self.source_id}' is registered with "
                    f"{self.capability}=not_supported, so '{self.operation}' is "
                    f"refused rather than attempted")
        return (f"{UNVERIFIED_CAPABILITY}: nobody has established whether "
                f"'{self.source_id}' supports {self.capability}; the result of "
                f"'{self.operation}' must not be read as a verified retrieval")

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "capability": self.capability,
                "state": self.state.value, "operation": self.operation,
                "allowed": self.allowed, "unverified": self.unverified,
                "reason": self.reason}


@dataclass(frozen=True)
class EndpointStatus:
    """Whether a network route to this source is established, and what is missing.

    Carries the curation notes so the refusal is actionable where it is read.
    A bare ``None`` endpoint would make every caller re-derive what to do about
    it, and some of them would decide to guess.
    """

    source_id: str
    endpoint: str | None
    access_modes: tuple[AccessMode, ...]
    curation_notes: tuple[str, ...]

    @property
    def established(self) -> bool:
        return self.endpoint is not None

    @property
    def has_network_mode(self) -> bool:
        return any(m.is_network_endpoint for m in self.access_modes)

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "endpoint": self.endpoint,
                "established": self.established,
                "has_network_mode": self.has_network_mode,
                "access_modes": [m.value for m in self.access_modes],
                "curation_notes": list(self.curation_notes)}


# ===========================================================================
# Shared foundation: the registry-gated connector
# ===========================================================================

class RegistryBackedConnector(Connector):
    """A connector whose route, capabilities and limits come from the registry.

    Everything that could be hard-coded is read from ``configs/datasources``
    at call time instead:

    * the base URL (always ``None`` today, hence always a typed refusal);
    * the capability flags that gate each operation;
    * the access modes that decide whether a live call is even conceivable;
    * the evidence-strength ceiling and the licence carried onto every record.

    Reading them at call time rather than at import time matters: a curator who
    establishes an endpoint or confirms a capability edits one YAML file, and
    the next call picks it up. A connector holding a copy taken at import would
    keep refusing, and somebody would "fix" that with a literal.

    The registry consulted is ``configs/datasources``, which covers all 49
    sources, so :attr:`source_id` is a datasource id and not a key of
    :data:`~eagent.connectors.base.CONNECTOR_REGISTRY` -- that smaller registry
    describes the ten resources the *planner* reasons about, under its own
    keys. :attr:`~eagent.connectors.base.Connector.spec` is therefore not used
    by anything here; :attr:`source` is the equivalent, and it exists for every
    registered resource rather than for ten of them.
    """

    #: Registry id in ``configs/datasources``. Also the cache namespace, so a
    #: curated import lands where the connector looks for it.
    source_id: ClassVar[str] = ""
    #: Which capability flag gates each operation name.
    operation_capability: ClassVar[Mapping[str, str]] = {
        "fetch": "exact_record_fetch",
        "search": "keyword_query",
    }
    #: The probed capability this connector's request builder was written
    #: against, or ``None`` while it was written against nothing.
    #:
    #: ``None`` is the default and the honest state for every connector here:
    #: the clients are generic (``<base>/<key>`` for a fetch, query parameters
    #: for a search) and no service was available to check them against when
    #: they were written. A verified base URL does not change that -- see
    #: :class:`RequestShapeNotVerifiedError` -- so declaring this is a separate
    #: act from recording an endpoint, done by whoever checked the client
    #: against a real response.
    verified_route_capability: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Keep ``Connector.name`` and :attr:`source_id` identical.

        The cache is keyed on ``name``. If the two drifted apart, a curator's
        import would be written under one key and looked up under another, and
        the run would report a miss over a file sitting right there.
        """
        super().__init_subclass__(**kwargs)
        if getattr(cls, "source_id", ""):
            cls.name = cls.source_id

    def __init__(self, *, cache: FileCache | None = None,
                 snapshot: str | None = None,
                 access: AccessPolicy | None = None,
                 registry: SourceRegistry | None = None) -> None:
        if not self.source_id:
            raise ConnectorConfigurationError(
                f"{type(self).__name__} declares no source_id, so nothing can "
                f"check it against the registry")
        self.registry = registry if registry is not None else datasource_registry()
        source = self.registry.get(self.source_id)   # raises on an unknown id
        if source.is_human_import_only:
            raise ConnectorConfigurationError(
                f"'{self.source_id}' is registered with access modes "
                f"{[m.value for m in source.access_modes]}: every route needs a "
                f"person, so it must be wired as a CuratedFileImporter. Building "
                f"a connector for it would present a browse-only or "
                f"author-archive resource as a live service.")
        super().__init__(cache=cache,
                         snapshot=snapshot or source.version,
                         access=access)

    # -- registry reads ----------------------------------------------------
    @property
    def source(self) -> DataSource:
        """The registry entry, re-read on every access.

        A property rather than a stored copy: the registry is the single place
        a curator records what has been confirmed, and a connector holding a
        stale snapshot of it would keep refusing a route that now exists.
        """
        return self.registry.get(self.source_id)

    @property
    def evidence_strength_ceiling(self) -> EvidenceStrength:
        """The strongest claim an automated ingest may stamp on these records."""
        return self.source.evidence_strength_ceiling

    def capability(self, name: str, operation: str = "") -> CapabilityVerdict:
        """The tri-state verdict for one capability on one operation."""
        if name not in CAPABILITY_NAMES:
            raise ConnectorConfigurationError(
                f"'{name}' is not a registered capability; known: "
                f"{', '.join(CAPABILITY_NAMES)}")
        return CapabilityVerdict(self.source_id, name,
                                 self.source.capabilities.get(name),
                                 operation or name)

    def require_capability(self, name: str, operation: str = "") -> CapabilityVerdict:
        """Return the verdict, raising on ``NOT_SUPPORTED``.

        The raising form exists for callers that are about to do work on the
        strength of the capability; the response-returning operations use the
        non-raising form so a refusal reaches the run manifest as data.
        """
        verdict = self.capability(name, operation)
        if not verdict.allowed:
            raise CapabilityNotSupportedError(self.source_id, name,
                                              verdict.operation)
        return verdict

    def endpoint_status(self) -> EndpointStatus:
        """Whether a route is established, with the curation notes that fix it."""
        src = self.source
        return EndpointStatus(src.id, src.endpoint, tuple(src.access_modes),
                              tuple(src.curation_notes))

    def require_endpoint(self) -> str:
        """The registry's base URL, or a typed refusal naming the curation note.

        This is the only place in the package that produces a URL, and it never
        produces one the registry did not record. A default here -- even a
        commented-out one -- is a string somebody will uncomment.
        """
        status = self.endpoint_status()
        if not status.established:
            raise EndpointNotEstablishedError(self.source_id, status.curation_notes)
        if not status.has_network_mode:
            raise AccessModeRefusedError(
                f"'{self.source_id}' records an endpoint but no network access "
                f"mode ({[m.value for m in status.access_modes]}); a bulk "
                f"archive is not a service to be called per record")
        if not self.verified_route_capability:
            # An established base URL is not an established request. See
            # RequestShapeNotVerifiedError.
            raise RequestShapeNotVerifiedError(
                self.source_id, str(status.endpoint),
                self.source.verified_capabilities)
        return str(status.endpoint)

    # -- gated operations --------------------------------------------------
    def fetch(self, key: str) -> CachedResponse:
        """Retrieve one record by identifier, gated on ``exact_record_fetch``."""
        return self.guarded("fetch", key,
                            self.operation_capability.get("fetch",
                                                          "exact_record_fetch"))

    def search(self, query: Mapping[str, Any]) -> CachedResponse:
        """Run a structured query, gated on ``keyword_query``."""
        return self.guarded("search", query,
                            self.operation_capability.get("search",
                                                          "keyword_query"))

    def guarded(self, operation: str, query: Mapping[str, Any] | str,
                capability: str, *,
                extra_notes: Sequence[str] = (),
                outbound_payload: Any | None = None) -> CachedResponse:
        """Capability gate, disclosure gate, then the cache-first resolver.

        The order matters. A refusal must not touch the cache, because a cached
        answer served through a route the source does not support would make the
        unsupported route look usable the next time anybody checks. A disclosure
        check must happen before anything else records the query, because the
        cache key and the run manifest both retain it.
        """
        verdict = self.capability(capability, operation)
        normalised = self.normalise_query(operation, query)
        if not verdict.allowed:
            return self.refusal(operation, normalised, verdict.reason, needed=(
                f"route this question to a source that supports {capability}",
                f"or have a curator re-check '{self.source_id}' and update "
                f"{capability} in configs/datasources if the flag is wrong",
            ))
        notes: list[str] = list(extra_notes)
        if outbound_payload is not None:
            notes.extend(self.authorise_outbound(outbound_payload))
        if verdict.unverified:
            notes.append(verdict.reason)
        if not self.source.endpoint:
            notes.append(
                f"no endpoint is established for '{self.source_id}', so this "
                f"call can only be answered from the local cache")
        response = self._resolve(operation, normalised)
        return replace(response, notes=tuple(response.notes) + tuple(notes))

    def refusal(self, operation: str, query: Mapping[str, Any], reason: str, *,
                needed: Sequence[str] = ()) -> CachedResponse:
        """A structured ``REFUSED`` response. There is no payload path here.

        Separate from a miss on purpose (see
        :class:`~eagent.connectors.base.ResponseStatus`): populating a cache
        would turn a miss into a hit, and must never turn a policy refusal into
        one.
        """
        return CachedResponse(
            connector=self.name, data_layer=self.data_layer, operation=operation,
            query=dict(query), status=ResponseStatus.REFUSED,
            cache_key=FileCache.key_for(self.name, self.version, dict(query)),
            cache_path=str(self.cache.path_for(self.name, self.version, dict(query))),
            payload=None, connector_version=self.version,
            miss_reason=reason, needed=tuple(needed),
        )

    # -- disclosure --------------------------------------------------------
    def authorise_outbound(self, payload: Any) -> tuple[str, ...]:
        """Refuse unless every sequence-shaped string in ``payload`` is cleared.

        Delegates to :meth:`~eagent.connectors.base.AccessPolicy.check_outbound`,
        which raises :class:`~eagent.connectors.base.UnauthorizedSubmissionError`
        for a sequence covered by neither a named authorisation nor a
        public-record finding.

        Deliberately called on every sequence-bearing query, not only when the
        network happens to be enabled. A guard that runs only on the online
        branch is exercised exactly never in an offline-by-default project, so
        the first time it matters is the first time it runs -- and that is the
        run where an unpublished construct leaves the building. Clearing it
        costs one :class:`~eagent.connectors.base.SubmissionAuthorization`,
        which is the audit record the project wants anyway.
        """
        return tuple(self.access.check_outbound(self.name, payload))

    # -- remote hooks ------------------------------------------------------
    def _fetch_remote(self, key: str) -> tuple[Any, str | None]:
        """Retrieve one record over the network. Refuses while no route exists.

        Reached only with ``allow_network`` true and after the disclosure guard.
        The first statement is :meth:`require_endpoint`, so the absence of a
        recorded base URL is a typed refusal rather than a request to an
        address nobody verified.
        """
        base = self.require_endpoint()
        return self._http_json(urllib.parse.urljoin(
            base if base.endswith("/") else base + "/",
            urllib.parse.quote(str(key), safe="")))

    def _search_remote(self, query: Mapping[str, Any]) -> tuple[Any, str | None]:
        """Search over the network. Refuses while no route exists."""
        base = self.require_endpoint()
        params = {k: str(v) for k, v in sorted(query.items()) if k != "op"}
        sep = "&" if urllib.parse.urlparse(base).query else "?"
        return self._http_json(f"{base}{sep}{urllib.parse.urlencode(params)}")

    def _http_text(self, url: str, timeout: float = 30.0,
                   accept: str = "*/*") -> str | None:
        """GET one URL and return the body, or ``None`` for "nothing there".

        The shared half of :meth:`_http_json`, for services that answer in
        TSV. Status handling is identical because it is the same question:
        404 and 410 say the service looked and found nothing, which is an
        answer; every other failure is :class:`RemoteCallFailedError`, because
        an unanswered question must not be recorded as a negative one.
        """
        if not self.access.allow_network:
            raise NetworkDisabledError(
                f"{self.source_id}: a remote call was attempted while "
                f"allow_network is false")
        request = urllib.request.Request(
            url, method="GET",
            headers={"Accept": accept, "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as handle:  # noqa: S310
                return handle.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 410):
                return None
            raise RemoteCallFailedError(
                self.source_id, url,
                f"HTTP {exc.code}" + _error_detail(exc), status=exc.code) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RemoteCallFailedError(
                self.source_id, url, f"{type(exc).__name__}: {exc}") from exc

    def _http_json(self, url: str, timeout: float = 30.0) -> tuple[Any, str | None]:
        """GET one URL and parse JSON. One of two outbound calls in this package.

        Kept in one place so the policy check cannot be forgotten in a subclass:
        a connector that wrote its own ``urlopen`` would bypass
        ``allow_network``. The request *shape* is as unverified as the base URL,
        which is why the parsed payload is returned with ``database_version``
        ``None`` unless the service states one -- an invented release string
        would claim a reproducibility the run does not have.
        """
        raw_body = self._http_text(url, timeout, accept="application/json")
        if raw_body is None:
            return None, None
        try:
            payload = json.loads(raw_body)
        except ValueError as exc:
            raise RemoteCallFailedError(
                self.source_id, url,
                "the response was not JSON; a login page or an error page "
                "answered instead of the API") from exc
        version = None
        if isinstance(payload, Mapping):
            raw = payload.get("database_version") or payload.get("version")
            version = str(raw) if raw is not None else None
        return payload, version

    # -- record helpers ----------------------------------------------------
    def records(self, response: CachedResponse) -> list[Mapping[str, Any]]:
        """The record list from a successful response, or an empty list.

        Empty rather than improvised: a payload of an unexpected shape means a
        curator has not finished, and a gap reported is recoverable while a
        guessed reading is not.
        """
        if not response.ok:
            return []
        return records_in(response.payload)

    def evidence_ref(self, identifier: str, response: CachedResponse, *,
                     locator: str | None = None,
                     strength: EvidenceStrength | None = None) -> EvidenceRef:
        """Evidence pointer for one record, carrying lineage and licence."""
        return evidence_ref_for(
            self.source, identifier, registry=self.registry, locator=locator,
            database_version=response.database_version,
            retrieved_at=response.retrieved_at, strength=strength)


# ===========================================================================
# Shared foundation: curated file importers
# ===========================================================================

@dataclass(frozen=True)
class ImportProvenance:
    """Who imported which file, when, and what the records were capped at.

    Recorded per import rather than per record because the question a reviewer
    asks is "where did this table come from and who vouched for it", and the
    answer has to survive the import script being deleted. The file digest is
    part of it: a re-import of an edited file is a different import, and
    nothing downstream can tell unless the digest says so.
    """

    source_id: str
    imported_by: str
    imported_at: str
    file_path: str
    file_sha256: str
    file_format: str
    n_rows_read: int
    n_rows_accepted: int
    n_rows_rejected: int
    evidence_strength_ceiling: EvidenceStrength
    database_version: str | None = None
    access_modes: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "imported_by": self.imported_by,
            "imported_at": self.imported_at,
            "file_path": self.file_path,
            "file_sha256": self.file_sha256,
            "file_format": self.file_format,
            "n_rows_read": self.n_rows_read,
            "n_rows_accepted": self.n_rows_accepted,
            "n_rows_rejected": self.n_rows_rejected,
            "evidence_strength_ceiling": self.evidence_strength_ceiling.value,
            "database_version": self.database_version,
            "access_modes": list(self.access_modes),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class RejectedRow:
    """A row that did not validate, kept with its reason.

    Rejections are returned rather than logged so an import that admitted 40 of
    60 rows cannot be mistaken for one that admitted 60. The raw row travels
    with the reason because the fix is usually visible in it.
    """

    row_number: int
    reason: str
    raw: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"row_number": self.row_number, "reason": self.reason,
                "raw": dict(self.raw)}


@dataclass(frozen=True)
class CuratedImport:
    """The result of one curated import: records, rejections and provenance."""

    provenance: ImportProvenance
    records: tuple[IntakeRecord, ...] = ()
    rejected: tuple[RejectedRow, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether every row in the file was admitted."""
        return not self.rejected

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "n_records": len(self.records),
            "record_ids": [r.intake_id for r in self.records],
            "rejected": [r.to_dict() for r in self.rejected],
            "warnings": list(self.warnings),
        }


class CuratedFileImporter(abc.ABC):
    """A source whose records enter through a person, not through a call.

    Deliberately **not** a :class:`~eagent.connectors.base.Connector`. It has no
    cache-first resolver, no remote hook and no endpoint, because it is not a
    client: ``fetch`` and ``search`` exist only to raise
    :class:`OfflineImportOnlyError`, so that the absence of a live route is a
    refusal with an explanation rather than an :class:`AttributeError` somebody
    patches by writing a client.

    Every record it produces goes through :func:`eagent.datalayer.intake.ingest`
    with no claimed strength, which stamps the weakest rung and lets the
    registry's ceiling cap anything stronger. The ceiling is never used as a
    default: a hand-curated spreadsheet row is not sequence-level experimental
    evidence because the resource it came from could, in principle, carry some.
    """

    #: Registry id in ``configs/datasources``.
    source_id: ClassVar[str] = ""
    #: Columns (or FASTA-derived keys) a row must carry to be admitted.
    required_fields: ClassVar[tuple[str, ...]] = ()
    #: File extensions this importer knows how to read.
    accepted_suffixes: ClassVar[tuple[str, ...]] = (
        ".csv", ".tsv", ".tab", ".json", ".fasta", ".fa", ".faa")
    #: Tier every row enters at. Never ``EXPERT_VERIFIED_PRIMARY``: importing a
    #: file is a clerical act, and expert verification is a per-record reading.
    tier: ClassVar[EvidenceTier] = EvidenceTier.CURATED_DATABASE
    extraction_method: ClassVar[ExtractionMethod] = ExtractionMethod.DATABASE_EXPORT
    description: ClassVar[str] = ""

    def __init__(self, *, registry: SourceRegistry | None = None) -> None:
        if not self.source_id:
            raise ConnectorConfigurationError(
                f"{type(self).__name__} declares no source_id")
        self.registry = registry if registry is not None else datasource_registry()
        source = self.registry.get(self.source_id)
        if not source.is_human_import_only:
            raise ConnectorConfigurationError(
                f"'{self.source_id}' declares a route that does not need a "
                f"person ({[m.value for m in source.access_modes]}); importing "
                f"it by hand would hide an available programmatic route")

    @property
    def source(self) -> DataSource:
        return self.registry.get(self.source_id)

    # -- the refusals that keep this from looking like a client ------------
    def fetch(self, key: str) -> CachedResponse:
        """Always refuses. See :class:`OfflineImportOnlyError`."""
        raise OfflineImportOnlyError(self.source_id,
                                     tuple(self.source.access_modes), "fetch")

    def search(self, query: Mapping[str, Any]) -> CachedResponse:
        """Always refuses. See :class:`OfflineImportOnlyError`."""
        raise OfflineImportOnlyError(self.source_id,
                                     tuple(self.source.access_modes), "search")

    def require_endpoint(self) -> str:
        """Always refuses: an import-only source has no endpoint by definition."""
        raise EndpointNotEstablishedError(self.source_id,
                                          tuple(self.source.curation_notes))

    # -- the one way in ----------------------------------------------------
    def import_file(self, path: str | Path, *, imported_by: str,
                    database_version: str | None = None,
                    at: str | None = None,
                    strict: bool = False) -> CuratedImport:
        """Read, validate and admit a curator-supplied export.

        ``imported_by`` is required and must name somebody who can be asked
        where the file came from. The check is weaker than
        :mod:`eagent.datalayer.intake`'s reviewer rule on purpose: an import is
        a clerical act and a promotion is a scientific judgement, so the
        importer records an identity while promotion demands a person who read
        the record.

        ``strict`` turns any rejected row into a :class:`CuratedImportError`.
        The default returns the rejections instead, because a curator usually
        wants the 58 good rows plus a list of the 2 bad ones; what is never
        offered is dropping the 2 silently.
        """
        who = " ".join(str(imported_by or "").split())
        if len(who) < 3:
            raise CuratedImportError(
                f"import_file requires imported_by to name whoever supplied the "
                f"file (at least 3 characters); got {imported_by!r}. Without it "
                f"nobody can be asked which export this is.")
        p = Path(path)
        if not p.is_file():
            raise CuratedImportError(f"no such curated export: {p}")
        suffix = p.suffix.lower()
        if suffix not in self.accepted_suffixes:
            raise CuratedImportError(
                f"{type(self).__name__} reads {', '.join(self.accepted_suffixes)}; "
                f"'{suffix}' is not one of them. Convert the export rather than "
                f"letting an unknown format be parsed on a guess.")
        rows = self._read_rows(p, suffix)
        timestamp = at or utc_now()
        source = self.source
        # Hashed once: the digest identifies this exact export, and recomputing
        # it per row would make a large import quadratic in the file size for
        # no added guarantee.
        digest = sha256_file(p)

        records: list[IntakeRecord] = []
        rejected: list[RejectedRow] = []
        warnings: list[str] = []
        for number, row in enumerate(rows, start=1):
            missing = [f for f in self.required_fields
                       if not str(row.get(f, "")).strip()]
            if missing:
                rejected.append(RejectedRow(
                    number,
                    f"missing required field(s): {', '.join(missing)}; the row "
                    f"cannot be identified, and a row nobody can identify "
                    f"cannot be de-duplicated against anything",
                    row))
                continue
            try:
                record = self._build_record(row, number)
            except (ValueError, CuratedImportError) as exc:
                rejected.append(RejectedRow(number, str(exc), row))
                continue
            row_ceiling = self._row_ceiling(row)
            notes = list(self._row_uncertainties(row))
            if row_ceiling is not source.evidence_strength_ceiling:
                notes.append(
                    f"this row is capped at {row_ceiling.value} rather than the "
                    f"source ceiling {source.evidence_strength_ceiling.value}: "
                    f"{self._row_ceiling_reason(row)}")
            notes.append(
                f"imported by {who} from {p.name} "
                f"(sha256:{digest[:16]}...) at {timestamp}; the source "
                f"is registered as human-import-only, so no live route was used")
            records.append(ingest(
                record, tier=self.tier, source_id=self.source_id,
                registry=self.registry, extraction_method=self.extraction_method,
                claimed_strength=None, uncertainties=notes,
                intake_id=f"intake:{self.source_id}:{record.record_id}",
                at=timestamp))

        if source.requires_human_review_per_record:
            warnings.append(
                f"'{self.source_id}' is registered requires_human_review_per_record: "
                f"every imported row needs a named reviewer before it may be used "
                f"as anything but triage")
        if not source.derived_from_complete:
            warnings.append(
                f"'{self.source_id}' declares an incomplete lineage, so these rows "
                f"may not be counted as independent corroboration of anything")
        if strict and rejected:
            raise CuratedImportError(
                f"{len(rejected)} of {len(rows)} rows in {p} did not validate: "
                + "; ".join(f"row {r.row_number}: {r.reason}" for r in rejected))

        provenance = ImportProvenance(
            source_id=self.source_id, imported_by=who, imported_at=timestamp,
            file_path=str(p), file_sha256=digest, file_format=suffix,
            n_rows_read=len(rows), n_rows_accepted=len(records),
            n_rows_rejected=len(rejected),
            evidence_strength_ceiling=source.evidence_strength_ceiling,
            database_version=database_version,
            access_modes=tuple(m.value for m in source.access_modes),
            notes=tuple(source.curation_notes),
        )
        return CuratedImport(provenance, tuple(records), tuple(rejected),
                             tuple(warnings))

    # -- parsing -----------------------------------------------------------
    def _read_rows(self, path: Path, suffix: str) -> list[dict[str, Any]]:
        """Parse a curated export into rows, refusing anything ambiguous."""
        text = path.read_text(encoding="utf-8")
        if suffix in (".csv", ".tsv", ".tab"):
            delimiter = "," if suffix == ".csv" else "\t"
            reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
            if reader.fieldnames is None:
                raise CuratedImportError(
                    f"{path}: no header row; a headerless table would have to be "
                    f"read by column position, and a reordered export would then "
                    f"be silently mis-parsed")
            return [{(k or "").strip(): ("" if v is None else str(v).strip())
                     for k, v in row.items()} for row in reader]
        if suffix == ".json":
            raw = json.loads(text)
            if isinstance(raw, Mapping):
                raw = raw.get("records", raw.get("rows"))
            if not isinstance(raw, list):
                raise CuratedImportError(
                    f"{path}: expected a JSON list of rows, or an object with a "
                    f"'records' list; got {type(raw).__name__}")
            out: list[dict[str, Any]] = []
            for item in raw:
                if not isinstance(item, Mapping):
                    raise CuratedImportError(
                        f"{path}: a row is {type(item).__name__}, not an object")
                out.append(dict(item))
            return out
        return _read_fasta(text, str(path))

    # -- subclass hooks ----------------------------------------------------
    @abc.abstractmethod
    def _build_record(self, row: Mapping[str, Any], row_number: int
                      ) -> ExperimentRecord:
        """Turn one validated row into a record, filling only what it states."""

    def _row_ceiling(self, row: Mapping[str, Any]) -> EvidenceStrength:
        """Per-row cap, defaulting to the source's registered ceiling."""
        return self.source.evidence_strength_ceiling

    def _row_ceiling_reason(self, row: Mapping[str, Any]) -> str:
        return "per-row rule in the importer"

    def _row_uncertainties(self, row: Mapping[str, Any]) -> tuple[str, ...]:
        """Row-level caveats carried onto the intake record."""
        return ()


def _read_fasta(text: str, where: str) -> list[dict[str, Any]]:
    """Parse FASTA into rows, keeping the header verbatim.

    The header is kept unparsed as well as split, because every resource
    encodes something different after the identifier and a parser that assumed
    one convention would quietly truncate identifiers from another.
    """
    rows: list[dict[str, Any]] = []
    identifier: str | None = None
    description = ""
    chunks: list[str] = []

    def flush() -> None:
        if identifier is None:
            return
        rows.append({"id": identifier, "description": description,
                     "header": f"{identifier} {description}".strip(),
                     "sequence": "".join(chunks)})

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            flush()
            head = stripped[1:].strip()
            if not head:
                raise CuratedImportError(
                    f"{where}: line {lineno} is a FASTA header with no "
                    f"identifier; an unnamed sequence cannot be joined to "
                    f"anything")
            identifier, _, description = head.partition(" ")
            description = description.strip()
            chunks = []
        else:
            if identifier is None:
                raise CuratedImportError(
                    f"{where}: line {lineno} carries sequence before any header")
            chunks.append(stripped)
    flush()
    if not rows:
        raise CuratedImportError(f"{where}: no FASTA records found")
    return rows


# ===========================================================================
# PubChem
# ===========================================================================

@dataclass(frozen=True)
class PubChemResolution:
    """A name resolved to a structure, with what the resolution did not decide.

    The ladder is the product, not the SMILES. A caller that wanted "the
    substrate" has to say which rung it means, which is what stops a
    connectivity-only structure being docked as though stereochemistry and
    protonation had been settled.

    :attr:`caveats` is never empty for a successful resolution: a name service
    answers "which connectivity does this string denote", and chirality, salt
    form and charge state are three separate questions it did not answer.
    """

    query_name: str
    ladder: ChemicalIdentityLadder
    response: CachedResponse
    cid: str | None = None
    caveats: tuple[str, ...] = ()
    unresolved: bool = False

    @property
    def resolved(self) -> bool:
        return not self.unresolved

    @property
    def stereo_defined(self) -> bool:
        """Whether the resolution reached the stereo rung with real descriptors."""
        return self.ladder.get(IdentityRung.STEREO_DEFINED_STRUCTURE) is not None

    def require_stereo_defined(self) -> RungValue:
        """The stereo rung, or a refusal naming what is missing.

        An asymmetric reduction's entire objective lives on this rung, so the
        failure this prevents is a campaign scored for enantioselectivity
        against a substrate nobody gave a configuration to.
        """
        rung = self.ladder.get(IdentityRung.STEREO_DEFINED_STRUCTURE)
        if rung is None:
            raise LayerSemanticsError(
                f"'{self.query_name}' resolved to a connectivity-only structure: "
                f"the resolved string carries no stereo descriptor, so the "
                f"stereo_defined_structure rung is absent. A name service did "
                f"not decide the configuration, and no other rung may stand in "
                f"for it.")
        return rung

    def to_dict(self) -> dict[str, Any]:
        return {"query_name": self.query_name, "cid": self.cid,
                "resolved": self.resolved, "stereo_defined": self.stereo_defined,
                "rungs": [r.value for r in self.ladder.present()],
                "caveats": list(self.caveats),
                "response": self.response.to_dict()}


class PubChemConnector(RegistryBackedConnector):
    """Name-to-structure resolution, fed into identity rungs rather than a field.

    Reads the payload shape::

        {"records": [{"cid": "...", "iupac_name": "...",
                      "connectivity_smiles": "...",     # or "canonical_smiles"
                      "isomeric_smiles": "...",
                      "inchikey": "...", "charge": 0}]}

    Nothing here invents a structure: a payload without a structural field
    yields an unresolved result carrying the reason, because an improvised
    SMILES for a compound nobody looked up would be ordered, docked and
    measured against.
    """

    source_id = "pubchem"
    data_layer = ConnectorLayer.REACTION
    description = "PubChem: name and identifier resolution to a structure"

    def resolve_name(self, name: str) -> PubChemResolution:
        """Resolve a written name onto identity rungs, flagging what is undecided.

        The rung discipline is the guard. ``as_written`` keeps the author's
        string so a curator can always re-read it; ``normalised_structure``
        takes the connectivity; ``stereo_defined_structure`` is populated *only*
        when the returned string actually carries a stereo descriptor. The
        failure that prevents: a toolkit default or a reaction-class assumption
        silently supplying the configuration that the whole project is trying to
        measure.
        """
        query = {"name": name}
        response = self.guarded("search", query, "keyword_query")
        ladder = ChemicalIdentityLadder(subject=f"pubchem:{name}")
        ladder.add(IdentityRung.AS_WRITTEN, name,
                   representation=Representation.FREE_TEXT,
                   produced_by="caller", source="query",
                   notes="the string the caller asked about, kept verbatim")
        if not response.ok:
            return PubChemResolution(
                name, ladder, response, caveats=(
                    f"not resolved: {response.miss_reason or response.status.value}",
                    "no structure was synthesised for this name; the rung stays "
                    "absent so the gap is visible downstream",
                ), unresolved=True)

        rows = self.records(response)
        if len(rows) != 1:
            return PubChemResolution(
                name, ladder, response, caveats=(
                    f"the cached payload holds {len(rows)} records for this "
                    f"name; a name that matches several compounds has not been "
                    f"resolved, and picking the first is how the wrong salt or "
                    f"the wrong tautomer enters a campaign",
                ), unresolved=True)

        row = rows[0]
        cid = _text(row.get("cid"))
        version = response.database_version
        source_tag = f"database:pubchem@{version or 'unpinned'}"
        caveats: list[str] = []

        connectivity = _text(row.get("connectivity_smiles")) or _text(
            row.get("canonical_smiles"))
        isomeric = _text(row.get("isomeric_smiles"))
        inchikey = _text(row.get("inchikey"))
        if connectivity:
            ladder.add(IdentityRung.NORMALISED_STRUCTURE, connectivity,
                       representation=Representation.SMILES,
                       produced_by="pubchem", source=source_tag,
                       notes=f"inchikey={inchikey or 'not stated'}")
        elif isomeric:
            ladder.add(IdentityRung.NORMALISED_STRUCTURE, isomeric,
                       representation=Representation.SMILES,
                       produced_by="pubchem", source=source_tag,
                       notes="no connectivity-only form was returned; the "
                             "isomeric string is used for connectivity and the "
                             "stereo rung is assessed separately")
        else:
            caveats.append(
                "the record carries no SMILES, so no structural rung was "
                "populated; a name with no structure is not a substrate")
            return PubChemResolution(name, ladder, response, cid=cid,
                                     caveats=tuple(caveats), unresolved=True)

        if isomeric:
            descriptors = stereo_descriptors(isomeric, Representation.SMILES)
            if descriptors is not None and descriptors.any:
                ladder.add(IdentityRung.STEREO_DEFINED_STRUCTURE, isomeric,
                           representation=Representation.SMILES,
                           produced_by="pubchem", source=source_tag,
                           notes="stereo descriptors are present in the returned "
                                 "string; which stereoisomer was assayed is still "
                                 "a question about the material, not the record")
            else:
                caveats.append(
                    "the returned isomeric string carries no stereo descriptor, "
                    "so the stereo_defined_structure rung was left absent rather "
                    "than filled from a connectivity-only form")
        else:
            caveats.append(
                "no isomeric SMILES was returned; chirality is undetermined and "
                "must be supplied by the operator or read from the paper")

        if "." in (isomeric or connectivity or ""):
            caveats.append(
                "the resolved structure has more than one covalent component "
                "(a salt or co-crystal); the free acid or base actually assayed "
                "is a different material and must be stated explicitly")
        charge = row.get("charge")
        if charge is None:
            caveats.append(
                "the record states no formal charge; the charge and protonation "
                "state for modelling is a separate decision and no rung was "
                "populated for it")
        else:
            caveats.append(
                f"the record states charge={charge}; this is the database's "
                f"depiction, not a protonation state chosen at the assay pH, so "
                f"the charge_and_protonation_state rung stays absent")
        if self.source.capabilities.version_information is not CapabilityState.SUPPORTED:
            caveats.append(
                "no release identifier is recordable for this source, so the "
                "resolution cannot be pinned to a version in the run manifest")
        return PubChemResolution(name, ladder, response, cid=cid,
                                 caveats=tuple(caveats))


# ===========================================================================
# ChEBI
# ===========================================================================

class ChEBIClassStatus(str, enum.Enum):
    """Whether a ChEBI entry denotes one compound or a set of them.

    ``UNDETERMINED`` is a third state and not a synonym for ``INSTANCE``. An
    entry whose payload does not say is exactly as unusable as a class, because
    the project cannot tell whether it has a substrate or a category, and the
    cost of being wrong is a campaign aimed at a set.
    """

    CLASS = "class"
    INSTANCE = "instance"
    UNDETERMINED = "undetermined"

    @property
    def usable_as_substrate(self) -> bool:
        return self is ChEBIClassStatus.INSTANCE


class ChemicalClassRefusedError(LayerSemanticsError):
    """A chemical class was asked to act as a specific substrate structure.

    "Ketone" cannot be weighed out, docked or assigned a configuration. This
    refusal is what stops a class identifier reaching task normalisation, where
    it would be carried forward as though it named a compound.
    """


@dataclass(frozen=True)
class ChEBIEntry:
    """One ChEBI entry, with the class-versus-instance question answered first."""

    chebi_id: str
    name: str | None
    class_status: ChEBIClassStatus
    smiles: str | None
    inchikey: str | None
    response: CachedResponse
    stereo_defined: bool | None = None
    notes: tuple[str, ...] = ()

    @property
    def is_class(self) -> bool:
        return self.class_status is ChEBIClassStatus.CLASS

    def as_substrate_structure(self) -> str:
        """The structure, or a refusal if this entry is not one compound.

        Refuses for ``UNDETERMINED`` as well as ``CLASS``. Treating "the entry
        does not say" as "it is a compound" is the same mistake as treating a
        class as a compound, arrived at more quietly.
        """
        if not self.class_status.usable_as_substrate:
            raise ChemicalClassRefusedError(
                f"ChEBI {self.chebi_id} ({self.name or 'unnamed'}) is "
                f"{self.class_status.value}, not a stereochemically defined "
                f"compound: it denotes a set of structures, so it cannot be a "
                f"substrate. Resolve the specific member that was assayed and "
                f"use its entry.")
        if not self.smiles:
            raise ChemicalClassRefusedError(
                f"ChEBI {self.chebi_id} is registered as an instance but the "
                f"cached record carries no structure; an entry with no structure "
                f"is not a substrate and none was invented for it")
        return self.smiles

    def to_dict(self) -> dict[str, Any]:
        return {"chebi_id": self.chebi_id, "name": self.name,
                "class_status": self.class_status.value, "smiles": self.smiles,
                "inchikey": self.inchikey, "stereo_defined": self.stereo_defined,
                "notes": list(self.notes)}


class ChEBIConnector(RegistryBackedConnector):
    """ChEBI entries, with class entries kept out of the substrate path.

    Reads the payload shape::

        {"records": [{"chebi_id": "CHEBI:...", "name": "...",
                      "entity_type": "class" | "instance",
                      "smiles": "...", "inchikey": "..."}]}

    ``entity_type`` (or ``is_class``) must be stated. The registry's own
    curation note for this source says the project rule is that "a class entry
    must fail task normalisation rather than be accepted", and that rule is
    implemented here rather than left to each caller.
    """

    source_id = "chebi"
    data_layer = ConnectorLayer.REACTION
    description = "ChEBI: chemical entities of biological interest"

    def entry(self, chebi_id: str) -> ChEBIEntry | None:
        """Fetch one entry, or ``None`` when the cache does not hold it.

        ``None`` rather than a placeholder entry: a placeholder with
        ``class_status=UNDETERMINED`` would be indistinguishable from a real
        entry whose payload omitted the field, and the two call for different
        actions.
        """
        response = self.fetch(chebi_id)
        if not response.ok:
            return None
        rows = self.records(response)
        if len(rows) != 1:
            return None
        row = rows[0]
        status = _chebi_class_status(row)
        smiles = _text(row.get("smiles"))
        notes: list[str] = []
        if status is ChEBIClassStatus.UNDETERMINED:
            notes.append(
                "the cached record states neither 'entity_type' nor 'is_class', "
                "so whether this entry is a class or a single compound is "
                "unknown; it is refused as a substrate until a curator records it")
        stereo: bool | None = None
        if smiles:
            descriptors = stereo_descriptors(smiles, Representation.SMILES)
            stereo = bool(descriptors.any) if descriptors is not None else None
            if stereo is False:
                notes.append(
                    "the entry's structure carries no stereo descriptor; it fixes "
                    "connectivity only and does not distinguish the enantiomers")
        else:
            notes.append("the entry carries no structure")
        return ChEBIEntry(
            chebi_id=_text(row.get("chebi_id")) or chebi_id,
            name=_text(row.get("name")), class_status=status, smiles=smiles,
            inchikey=_text(row.get("inchikey")), response=response,
            stereo_defined=stereo, notes=tuple(notes))

    def require_substrate_entry(self, chebi_id: str) -> ChEBIEntry:
        """The entry, refusing anything that is not a single defined compound."""
        entry = self.entry(chebi_id)
        if entry is None:
            raise LayerSemanticsError(
                f"ChEBI {chebi_id} is not in the local cache and no record was "
                f"synthesised for it; place a curated import before using it as "
                f"a substrate")
        entry.as_substrate_structure()
        return entry


def _chebi_class_status(row: Mapping[str, Any]) -> ChEBIClassStatus:
    """Read class-versus-instance from the payload, never from the name.

    Only explicit fields are consulted. Inferring "class" from a plural name or
    from the absence of a SMILES would be a heuristic applied to the one
    distinction the whole substrate path depends on.
    """
    raw = row.get("entity_type", row.get("chebi_entity_type"))
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token in ("class", "chemical_class", "role", "ontology_class"):
            return ChEBIClassStatus.CLASS
        if token in ("instance", "compound", "molecular_entity", "chemical_entity"):
            return ChEBIClassStatus.INSTANCE
        return ChEBIClassStatus.UNDETERMINED
    if isinstance(row.get("is_class"), bool):
        return (ChEBIClassStatus.CLASS if row["is_class"]
                else ChEBIClassStatus.INSTANCE)
    return ChEBIClassStatus.UNDETERMINED


# ===========================================================================
# Rhea
# ===========================================================================

class RheaDirection(str, enum.Enum):
    """Rhea's own directional convention, kept separate from measured direction.

    Rhea publishes an undirected master reaction and its directional children.
    None of those is a statement that anybody ran the reaction that way, which
    is why this enum exists instead of mapping Rhea's convention straight onto
    :class:`~eagent.schemas.record.ReactionDirection`: the mapping has to pass
    through the task's reaction class, and a bidirectional entry has to end up
    as ``UNSPECIFIED`` rather than as "both directions shown".
    """

    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    BIDIRECTIONAL = "bidirectional"
    UNDIRECTED = "undirected"
    UNSTATED = "unstated"


@dataclass(frozen=True)
class RheaReaction:
    """A Rhea reaction, carrying the direction it was written in.

    :attr:`reaction_class` is whatever the curated payload states. It is not
    derived from the participants here, because deciding "this is an alcohol
    oxidation" from a list of ChEBI ids is exactly the inference that, done
    wrongly, turns an oxidation record into reduction evidence.
    """

    rhea_id: str
    equation: str | None
    rhea_direction: RheaDirection
    reaction_class: ReactionClass | None
    participants_left: tuple[str, ...]
    participants_right: tuple[str, ...]
    response: CachedResponse
    master_id: str | None = None
    notes: tuple[str, ...] = ()

    def direction_for(self, target: Any) -> ReactionDirection:
        """Map Rhea's convention onto the measured-direction enum for a target.

        Three rules, each blocking a specific misreading:

        * the entry's class is the chemical reverse of the target's ->
          ``REVERSE_OF_TARGET``, so an alcohol-oxidation entry can never be
          counted as support for the ketone reduction;
        * the entry is bidirectional or undirected -> ``UNSPECIFIED``, because a
          Rhea direction group is a curatorial convention and not a
          demonstration that both directions were observed. Mapping it to
          ``REVERSIBLE_BOTH_SHOWN`` would manufacture experimental support out
          of a database's data model;
        * the class is unstated or does not pair with the target ->
          ``UNSPECIFIED``, because an unrecorded direction is not a forward one.
        """
        target_class = _reaction_class_of(target)
        if target_class is not None and self.reaction_class is not None:
            if self.reaction_class is target_class:
                if self.rhea_direction in (RheaDirection.BIDIRECTIONAL,
                                           RheaDirection.UNDIRECTED,
                                           RheaDirection.UNSTATED):
                    return ReactionDirection.UNSPECIFIED
                return ReactionDirection.FORWARD_AS_TARGET
            if self.reaction_class in _reverse_classes(target_class):
                return ReactionDirection.REVERSE_OF_TARGET
        return ReactionDirection.UNSPECIFIED

    def direction_verdict(self, target: Any) -> Any:
        """Run :func:`~eagent.datalayer.intake.direction_check` on this entry.

        Reuses the intake rule rather than restating it, so there is exactly one
        definition of "this record does not support the target direction" in the
        project and a change to it cannot leave the connectors behind.
        """
        return direction_check(
            _DirectionProbe(record_id=f"rhea:{self.rhea_id}",
                            reaction_direction=self.direction_for(target),
                            reaction_class=self.reaction_class),
            target)

    def supports_target_direction(self, target: Any) -> bool:
        """Whether this entry may be read as being about the target direction."""
        return bool(self.direction_verdict(target).supports)

    def as_sequence_evidence(self) -> None:
        """Always refuses: a reaction definition names no protein.

        Prevents the substitution the registry warns about -- a Rhea identifier
        on an entry answering "does this enzyme reduce this ketone?".
        """
        raise ReactionLevelOnlyError(
            f"Rhea {self.rhea_id} defines a transformation and names no protein; "
            f"it is not evidence that any sequence catalyses it. Use the "
            f"enzymology layer for that question.")

    def to_dict(self) -> dict[str, Any]:
        return {"rhea_id": self.rhea_id, "equation": self.equation,
                "rhea_direction": self.rhea_direction.value,
                "reaction_class": (self.reaction_class.value
                                   if self.reaction_class else None),
                "participants_left": list(self.participants_left),
                "participants_right": list(self.participants_right),
                "master_id": self.master_id, "notes": list(self.notes)}


@dataclass(frozen=True)
class _DirectionProbe:
    """Minimal carrier so :func:`direction_check` can read a Rhea entry.

    A full :class:`~eagent.schemas.record.ExperimentRecord` would have to invent
    a sequence, a substrate and an outcome for a reaction definition that has
    none of them, and an invented record is exactly what this package refuses to
    produce.
    """

    record_id: str
    reaction_direction: ReactionDirection
    reaction_class: ReactionClass | None


#: Columns requested from Rhea's TSV endpoint, in order, with the header the
#: service writes for each. The pair is the parser: the format string and the
#: header check below are one decision written twice, so a column the service
#: renames or reorders is refused instead of being read from the wrong field.
RHEA_COLUMNS: tuple[str, ...] = ("rhea-id", "equation", "ec", "chebi-id")
RHEA_HEADER: tuple[str, ...] = (
    "Reaction identifier", "Equation", "EC number", "ChEBI identifier")

#: Rows asked for per query. Pinned so a truncated answer can be recognised:
#: a result that exactly fills the limit may have more behind it, and saying
#: "these are the reactions for this EC" about a clipped list is a claim about
#: the whole.
RHEA_QUERY_LIMIT: int = 100


def rhea_tsv_payload(text: str, *, limit: int = RHEA_QUERY_LIMIT
                     ) -> dict[str, Any]:
    """Translate Rhea's TSV response into this connector's payload.

    WHAT THIS RESPONSE SHAPE CANNOT SAY
    -----------------------------------
    It carries an identifier, the equation as written, the EC numbers and a
    flat list of ChEBI ids. It does **not** carry the reaction's direction, its
    master id, or which ChEBI ids sit on which side of the equation.

    Those are left unstated rather than derived. The registry's own curation
    note says the relationship between Rhea's directional and bidirectional
    entries must be confirmed "before any direction logic is written against
    them", and nothing in this response confirms it -- the ids returned for
    the EC 1.1.1.1 query include numbers that do not follow the id pattern one
    would guess from the documentation, which is exactly why a guess here
    would be a guess. ``direction`` is therefore ``"unstated"``, and
    :meth:`RheaConnector._build` already refuses to read an unrecorded
    direction as a forward one.

    The equation's two sides are split on `` = `` only into display strings;
    ChEBI participants stay unsided.
    """
    lines = text.splitlines()
    if not lines or tuple(c.strip() for c in lines[0].split("\t")) != RHEA_HEADER:
        raise RemoteCallFailedError(
            "rhea", "<tsv response>",
            f"the header {lines[0][:120]!r} is not the expected "
            f"{list(RHEA_HEADER)}; the column list and the parser have "
            f"diverged, and every value would be read from the wrong field"
            if lines else "the response was empty")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != len(RHEA_HEADER):
            raise RemoteCallFailedError(
                "rhea", "<tsv response>",
                f"line {number} has {len(parts)} column(s), expected "
                f"{len(RHEA_HEADER)}: {line[:120]!r}")
        rhea_id, equation, ec_text, chebi_text = (p.strip() for p in parts)
        if not rhea_id.startswith("RHEA:"):
            raise RemoteCallFailedError(
                "rhea", "<tsv response>",
                f"line {number}: {rhea_id!r} is not a Rhea identifier")
        sides = equation.split(" = ", 1)
        records.append({
            "rhea_id": rhea_id,
            "equation": equation,
            "equation_left_text": sides[0].strip() if len(sides) == 2 else None,
            "equation_right_text": sides[1].strip() if len(sides) == 2 else None,
            "ec_numbers": [e.removeprefix("EC:").strip()
                           for e in ec_text.split(";") if e.strip()],
            "participants": [c.strip() for c in chebi_text.split(";")
                             if c.strip()],
            "direction": "unstated",
            "reaction_class": None,
            "participants_left": [],
            "participants_right": [],
            "master_id": None,
        })
    return {"records": records, "truncated": len(records) >= limit,
            "query_limit": limit}


class RheaConnector(RegistryBackedConnector):
    """Rhea reactions, with the written direction preserved end to end.

    Reads the payload shape::

        {"records": [{"rhea_id": "RHEA:...", "equation": "...",
                      "direction": "left_to_right" | "right_to_left"
                                   | "bidirectional" | "undirected",
                      "reaction_class": "<ReactionClass value>",
                      "participants_left": ["CHEBI:..."],
                      "participants_right": ["CHEBI:..."],
                      "master_id": "RHEA:..."}]}

    ``reaction_class`` is read, never derived. The registry's curation note for
    this source says the relationship between the directional and bidirectional
    entries must be confirmed "before any direction logic is written against
    them"; until then this connector reports bidirectional entries as
    ``UNSPECIFIED`` rather than resolving them.
    """

    source_id = "rhea"
    data_layer = ConnectorLayer.REACTION
    description = "Rhea: expert-curated reactions with ChEBI participants"
    #: The one request shape this client makes, and the one the shipped probes
    #: check: ``/rhea?query=<q>&columns=<pinned>&format=tsv&limit=<n>``, where
    #: ``<q>`` is ``RHEA:<n>`` for a fetch and ``ec:<ec>`` for an EC search.
    verified_route_capability = "keyword_query"

    def _query_url(self, query: str) -> str:
        base = self.require_endpoint().rstrip("/")
        return (f"{base}/rhea?query={urllib.parse.quote(query, safe=':.')}"
                f"&columns={','.join(RHEA_COLUMNS)}&format=tsv"
                f"&limit={RHEA_QUERY_LIMIT}")

    def _fetch_remote(self, key: str) -> tuple[Any, str | None]:
        """One reaction by identifier, through the keyword-query route.

        Rhea has no per-record JSON route that answers without a browser
        (``/rhea/<id>.json`` is a 403 from this environment), so a fetch is a
        keyword query for the identifier. It succeeds only if exactly that
        identifier comes back: a query that returned some *other* reaction
        would otherwise be cached under this key.
        """
        wanted = str(key).strip()
        text = self._http_text(self._query_url(wanted), accept="text/plain")
        if text is None:
            return None, None
        payload = rhea_tsv_payload(text)
        rows = [r for r in payload["records"] if r["rhea_id"] == wanted]
        if not rows:
            return None, None
        return {"records": rows, "truncated": False}, None

    def _search_remote(self, query: Mapping[str, Any]) -> tuple[Any, str | None]:
        """An EC search. Any other query shape is refused, not improvised.

        Only the EC query was probed. A free-text or ChEBI query would be a
        guess at a syntax this client has not been checked against, and the
        service answers a malformed query with an empty, successful-looking
        table -- the failure that cannot be told from "no such reaction".
        """
        keys = {k for k in query if k != "op"}
        if keys != {"ec"}:
            raise ConnectorError(
                f"rhea: this client supports only an EC search "
                f"({{'ec': '1.1.1.1'}}); got {sorted(keys)}. Other query "
                f"syntaxes have not been probed, and a malformed one returns "
                f"an empty table that reads as 'no such reaction'")
        text = self._http_text(self._query_url(f"ec:{str(query['ec']).strip()}"),
                               accept="text/plain")
        if text is None:
            return None, None
        return rhea_tsv_payload(text), None

    def reaction(self, rhea_id: str) -> RheaReaction | None:
        """Fetch one reaction, or ``None`` when it is not cached."""
        response = self.fetch(rhea_id)
        if not response.ok:
            return None
        rows = self.records(response)
        if len(rows) != 1:
            return None
        return self._build(rows[0], rhea_id, response)

    def reactions_for_ec(self, ec_number: str) -> tuple[RheaReaction, ...]:
        """Every cached reaction annotated with an EC number.

        An EC number groups reactions by nomenclature, so this routinely returns
        both directions of a reversible pair. They come back separately, each
        carrying its own direction, rather than as one merged "the EC does this".
        """
        response = self.guarded("search", {"ec": ec_number}, "keyword_query")
        if not response.ok:
            return ()
        return tuple(
            self._build(row, _text(row.get("rhea_id")) or ec_number, response)
            for row in self.records(response))

    def _build(self, row: Mapping[str, Any], fallback_id: str,
               response: CachedResponse) -> RheaReaction:
        direction = _rhea_direction(row.get("direction"))
        reaction_class = _reaction_class_of(_text(row.get("reaction_class")))
        notes: list[str] = []
        if direction in (RheaDirection.BIDIRECTIONAL, RheaDirection.UNDIRECTED):
            notes.append(
                "this is a direction-group or undirected entry; it is a "
                "curatorial convention and not a demonstration that either "
                "direction was observed, so it is reported as unspecified")
        if direction is RheaDirection.UNSTATED:
            notes.append(
                "the cached record states no direction; an unrecorded direction "
                "is not a forward one")
        if reaction_class is None:
            notes.append(
                "the cached record states no reaction class, so the "
                "oxidation-versus-reduction pairing could not be checked; a "
                "curator must record it before this entry is used directionally")
        notes.append(
            "Rhea defines the transformation and names no protein: this entry is "
            "an annotation target, never evidence of catalysis by any sequence")
        return RheaReaction(
            rhea_id=_text(row.get("rhea_id")) or fallback_id,
            equation=_text(row.get("equation")),
            rhea_direction=direction,
            reaction_class=reaction_class,
            participants_left=_text_tuple(row.get("participants_left")),
            participants_right=_text_tuple(row.get("participants_right")),
            response=response,
            master_id=_text(row.get("master_id")),
            notes=tuple(notes))


def _rhea_direction(raw: Any) -> RheaDirection:
    """Read Rhea's direction field, defaulting to ``UNSTATED`` not to forward."""
    if not isinstance(raw, str) or not raw.strip():
        return RheaDirection.UNSTATED
    token = raw.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "lr": RheaDirection.LEFT_TO_RIGHT, "l_to_r": RheaDirection.LEFT_TO_RIGHT,
        "left_to_right": RheaDirection.LEFT_TO_RIGHT,
        "rl": RheaDirection.RIGHT_TO_LEFT, "r_to_l": RheaDirection.RIGHT_TO_LEFT,
        "right_to_left": RheaDirection.RIGHT_TO_LEFT,
        "bi": RheaDirection.BIDIRECTIONAL,
        "bidirectional": RheaDirection.BIDIRECTIONAL,
        "undirected": RheaDirection.UNDIRECTED,
        "master": RheaDirection.UNDIRECTED,
    }
    return aliases.get(token, RheaDirection.UNSTATED)


# ===========================================================================
# MetaNetX
# ===========================================================================

class CrossReferenceNotAMergeError(LayerSemanticsError):
    """A cross-reference edge was treated as permission to merge two identities.

    A reconciliation namespace says "these two identifiers were judged to refer
    to the same thing at some level of abstraction". It does not say the
    protonation states match, the stereochemistry matches, or that one may be
    substituted for the other in a geometric criterion. Merging on the edge is
    how a specific (S)-alcohol becomes an unspecified alcohol halfway through a
    pipeline.
    """


@dataclass(frozen=True)
class CrossReferenceEdge:
    """One identifier mapped to another, with the caveat attached.

    The caveat is a field rather than documentation because the edge travels
    into joins and reports where the docstring does not.
    """

    from_id: str
    to_id: str
    from_namespace: str
    to_namespace: str
    mnx_id: str | None = None
    evidence: str | None = None

    @property
    def caveat(self) -> str:
        return ("a reconciliation cross-reference: it maps identifiers and does "
                "not assert that stereochemistry, protonation, charge or salt "
                "form agree between the two entries")

    def to_dict(self) -> dict[str, Any]:
        return {"from_id": self.from_id, "to_id": self.to_id,
                "from_namespace": self.from_namespace,
                "to_namespace": self.to_namespace, "mnx_id": self.mnx_id,
                "evidence": self.evidence, "caveat": self.caveat}


@dataclass(frozen=True)
class MetaNetXMapping:
    """A set of cross-reference edges, with no merge operation offered."""

    query: str
    edges: tuple[CrossReferenceEdge, ...]
    response: CachedResponse
    upstream_sources: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def targets_in(self, namespace: str) -> tuple[str, ...]:
        """Identifiers this query maps to within one namespace."""
        want = namespace.strip().lower()
        return tuple(e.to_id for e in self.edges
                     if e.to_namespace.strip().lower() == want)

    def merge_identities(self) -> None:
        """Always refuses. See :class:`CrossReferenceNotAMergeError`."""
        raise CrossReferenceNotAMergeError(
            f"MetaNetX maps '{self.query}' onto "
            f"{len(self.edges)} identifier(s); that is a mapping, not a licence "
            f"to merge chemical states. Decide each state question "
            f"(stereochemistry, protonation, charge, salt form) against the "
            f"source entries and record who decided it.")

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "edges": [e.to_dict() for e in self.edges],
                "upstream_sources": list(self.upstream_sources),
                "notes": list(self.notes)}


class MetaNetXConnector(RegistryBackedConnector):
    """MNXref cross-references, returned as edges that refuse to collapse.

    Reads the payload shape::

        {"records": [{"mnx_id": "MNXM...", "from_id": "...",
                      "from_namespace": "chebi", "to_id": "...",
                      "to_namespace": "...", "evidence": "..."}]}

    The registry records this source's upstream list as *incomplete*, so every
    mapping comes back carrying that fact: a join made through MetaNetX cannot
    be counted as independent of the resources it reconciles.
    """

    source_id = "metanetx"
    data_layer = ConnectorLayer.REACTION
    description = "MetaNetX / MNXref: cross-resource identifier reconciliation"

    def cross_references(self, identifier: str) -> MetaNetXMapping:
        """Every cached cross-reference edge for one identifier."""
        response = self.guarded("search", {"identifier": identifier},
                                "keyword_query")
        edges = tuple(
            CrossReferenceEdge(
                from_id=_text(row.get("from_id")) or identifier,
                to_id=_text(row.get("to_id")) or "",
                from_namespace=_text(row.get("from_namespace")) or "unstated",
                to_namespace=_text(row.get("to_namespace")) or "unstated",
                mnx_id=_text(row.get("mnx_id")),
                evidence=_text(row.get("evidence")))
            for row in self.records(response)
            if _text(row.get("to_id")))
        notes = [
            "a cross-reference is a mapping between identifiers; merging the "
            "chemical states behind them is a separate decision that this "
            "resource does not make",
        ]
        if not self.source.derived_from_complete:
            notes.append(
                "this source's lineage is registered as incomplete, so a join "
                "made through it may not be counted as independent corroboration")
        return MetaNetXMapping(
            query=identifier, edges=edges, response=response,
            upstream_sources=upstream_sources_for(self.source_id, self.registry),
            notes=tuple(notes))


# ===========================================================================
# EnzymeMap
# ===========================================================================

class ReactionLevelOnlyError(LayerSemanticsError):
    """Reaction-level data was asked to validate a particular sequence.

    An atom-mapped reaction says which bond changes. It names no protein,
    carries no conditions and reports no measurement, so reading a hit as
    "this enzyme was shown to do this" invents an experiment.
    """


@dataclass(frozen=True)
class EnzymeMapReaction:
    """An atom-mapped reaction, explicitly not a per-sequence result."""

    enzymemap_id: str
    ec_number: str | None
    atom_mapped_reaction_smiles: str | None
    rhea_id: str | None
    mapping_confidence: float | None
    upstream_sources: tuple[str, ...]
    response: CachedResponse
    notes: tuple[str, ...] = ()

    def as_sequence_evidence(self) -> None:
        """Always refuses. See :class:`ReactionLevelOnlyError`."""
        raise ReactionLevelOnlyError(
            f"EnzymeMap {self.enzymemap_id} is reaction-level: it gives an atom "
            f"mapping for a transformation, not an experimental validation of "
            f"any sequence. It also derives from "
            f"{', '.join(self.upstream_sources) or 'an upstream corpus'}, so a "
            f"hit here and a hit upstream are one piece of evidence, not two.")

    def require_reviewed_mapping(self, min_confidence: float) -> str:
        """The atom mapping, refusing an unreviewed or low-confidence one.

        An algorithmic mapping is a hypothesis about which atom becomes which.
        A stereocentre claim resting on an unreviewed mapping is a claim about
        the wrong atom, and it looks identical to a correct one.
        """
        if not self.atom_mapped_reaction_smiles:
            raise ReactionLevelOnlyError(
                f"EnzymeMap {self.enzymemap_id} carries no atom-mapped reaction "
                f"SMILES in the cached record; none was constructed for it")
        if self.mapping_confidence is None:
            raise ReactionLevelOnlyError(
                f"EnzymeMap {self.enzymemap_id} states no mapping confidence, so "
                f"a low-confidence mapping cannot be rejected. The registry's "
                f"curation note requires confirming how confidence is expressed "
                f"before mappings are used.")
        if self.mapping_confidence < min_confidence:
            raise ReactionLevelOnlyError(
                f"EnzymeMap {self.enzymemap_id} mapping confidence "
                f"{self.mapping_confidence} is below the required "
                f"{min_confidence}; an algorithmic mapping below threshold is a "
                f"hypothesis a reviewer has not accepted")
        return self.atom_mapped_reaction_smiles

    def to_dict(self) -> dict[str, Any]:
        return {"enzymemap_id": self.enzymemap_id, "ec_number": self.ec_number,
                "atom_mapped_reaction_smiles": self.atom_mapped_reaction_smiles,
                "rhea_id": self.rhea_id,
                "mapping_confidence": self.mapping_confidence,
                "upstream_sources": list(self.upstream_sources),
                "notes": list(self.notes)}


class EnzymeMapConnector(RegistryBackedConnector):
    """EnzymeMap atom mappings, returned as reaction-level data and nothing more.

    Reads the payload shape::

        {"records": [{"enzymemap_id": "...", "ec": "1.1.1.-",
                      "atom_mapped_reaction_smiles": "...",
                      "rhea_id": "RHEA:...", "mapping_confidence": 0.0}]}

    Every record carries the registry's lineage closure in
    ``upstream_sources`` so :mod:`eagent.datalayer.lineage` can see that this
    resource and its upstream corpus are not two independent confirmations.
    """

    source_id = "enzymemap"
    data_layer = ConnectorLayer.REACTION
    description = "EnzymeMap: corrected, atom-mapped enzymatic reactions"

    def reactions_for_ec(self, ec_number: str) -> tuple[EnzymeMapReaction, ...]:
        """Cached atom-mapped reactions for one EC number."""
        response = self.guarded("search", {"ec": ec_number}, "keyword_query")
        upstream = upstream_sources_for(self.source_id, self.registry)
        notes = (
            "reaction-level data: it says which bond changes, not which protein "
            "changes it",
            f"derived from {', '.join(upstream)}; a hit here does not corroborate "
            f"a hit in an upstream resource",
        )
        return tuple(
            EnzymeMapReaction(
                enzymemap_id=_text(row.get("enzymemap_id")) or ec_number,
                ec_number=_text(row.get("ec")) or ec_number,
                atom_mapped_reaction_smiles=_text(
                    row.get("atom_mapped_reaction_smiles")),
                rhea_id=_text(row.get("rhea_id")),
                mapping_confidence=_float(row.get("mapping_confidence")),
                upstream_sources=upstream, response=response, notes=notes)
            for row in self.records(response))


# ===========================================================================
# small shared readers
# ===========================================================================

def _text(value: Any) -> str | None:
    """A trimmed string, or ``None``. Empty strings become ``None``.

    Keeps ``""`` out of identifier fields, where it would compare equal to
    itself and silently join unrelated records.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Sequence):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return ()


def _float(value: Any) -> float | None:
    """A float, or ``None`` for anything unparseable. Never a default of 0.0."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reaction_class_of(value: Any) -> ReactionClass | None:
    """Read a :class:`ReactionClass` from an enum, a string or a spec object."""
    if value is None:
        return None
    if isinstance(value, ReactionClass):
        return value
    if isinstance(value, str):
        try:
            return ReactionClass(value)
        except ValueError:
            return None
    inner = getattr(value, "reaction_class", None)
    if inner is not None:
        return _reaction_class_of(inner)
    nested = getattr(value, "reaction", None)
    if nested is not None and nested is not value:
        return _reaction_class_of(nested)
    return None


def _reverse_classes(cls: ReactionClass) -> tuple[ReactionClass, ...]:
    """Delegate to intake's reversible-pair table rather than restating it."""
    from ..datalayer.intake import reverse_class_of

    return reverse_class_of(cls)
