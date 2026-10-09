"""The connector capability registry: what each public resource can and cannot do.

Why this module exists
----------------------
The agent will call these resources. If the registry says a REST endpoint exists
and it does not, the failure surfaces at call time as a stack trace inside a
run that has already burned its budget. If the registry says a resource is
"supported" when nobody has checked, a planner will prefer it over a resource
that honestly reports uncertainty. So this registry makes three distinctions
that a plain configuration file collapses:

* **UNKNOWN is not NOT_SUPPORTED.** A capability nobody has verified is a
  research task; a capability the resource genuinely lacks is a routing fact.
  :class:`CapabilityState` keeps them apart so a planner can see the difference.
* **A website is not an API.** :class:`AccessMode` has explicit
  ``OFFLINE_IMPORT`` and ``MANUAL_REVIEW_IMPORT`` members, so a resource that
  ships an author CSV or only a web interface is registered as what it is.
  Dressing it up as a live API is how code comes to call a URL that was never
  documented anywhere.
* **A capability flag is documentation, not proof.** The flags describe what a
  resource is *documented* to offer. ``connectivity_verified`` may only be
  ``True`` when the entry carries a :class:`ConnectivityCheck` saying what was
  called, when, what came back and which capability it exercised -- a bare
  boolean asserting "this works" is worth nothing, and that is why the model
  used to refuse ``True`` outright. Most entries still carry no check and are
  still ``False``.

:class:`DataSource` also requires ``not_good_for``. A registry of what things are
good for produces a planner that uses BRENDA as a sequence database and a
benchmark set as an activity set. The limitation field is the point.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .layers import DataLayer, LAYER_ORDER

try:  # harness error base; guarded so the registry imports standalone
    from ..errors import EAgentError
except Exception:  # pragma: no cover
    class EAgentError(Exception):  # type: ignore[no-redef]
        """Fallback base error when :mod:`eagent.errors` is unavailable."""

try:  # the canonical evidence ladder
    from ..schemas.record import EvidenceStrength as _EvidenceStrength
    EVIDENCE_STRENGTH_IS_CANONICAL = True
except Exception:  # pragma: no cover - sibling package may be mid-write
    _EvidenceStrength = None
    EVIDENCE_STRENGTH_IS_CANONICAL = False


if _EvidenceStrength is not None:
    EvidenceStrength = _EvidenceStrength
else:  # pragma: no cover - mirror of eagent.schemas.record.EvidenceStrength
    class EvidenceStrength(str, enum.Enum):  # type: ignore[no-redef]
        """Stand-in used only when :mod:`eagent.schemas` cannot be imported.

        Values mirror the canonical enum. ``EVIDENCE_STRENGTH_IS_CANONICAL`` is
        False in that case so a caller can refuse to persist anything derived
        from the stand-in rather than silently diverging from the data model.
        """

        SEQUENCE_LEVEL_EXPERIMENTAL = "sequence_level_experimental"
        HOMOLOG_EXPERIMENTAL = "homolog_experimental"
        EC_SPECIES_MAPPED = "ec_species_mapped"
        ANNOTATION_ONLY = "annotation_only"
        COMPUTATIONAL_CONSTRUCT = "computational_construct"


__all__ = [
    "CapabilityState",
    "CapabilityFlags",
    "CAPABILITY_NAMES",
    "AccessMode",
    "DataSource",
    "SourceRegistry",
    "IndependenceReport",
    "RegistryError",
    "UnknownSourceError",
    "DuplicateSourceError",
    "RegistryIntegrityError",
    "EvidenceStrength",
    "EVIDENCE_STRENGTH_IS_CANONICAL",
    "default_datasource_dir",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class RegistryError(EAgentError):
    """Base class for registry load and lookup failures."""


class UnknownSourceError(RegistryError):
    """A source id was requested that is not registered.

    Raised rather than returning ``None`` so a typo in a task configuration
    cannot quietly drop a data layer from a run.
    """


class DuplicateSourceError(RegistryError):
    """Two YAML entries claim the same source id.

    A duplicate id would make lineage and independence grouping meaningless,
    because the second definition would silently win.
    """


class RegistryIntegrityError(RegistryError):
    """The registry as a whole is inconsistent (dangling or cyclic lineage).

    Checked at load time: a ``derived_from`` pointing at an unregistered id
    would make :meth:`SourceRegistry.upstream_closure` understate shared
    provenance, which is exactly the error that turns one re-published
    measurement into four "independent" confirmations.
    """


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------

class CapabilityState(str, enum.Enum):
    """Tri-state capability. ``UNKNOWN`` is the default and means nobody checked.

    A two-valued flag forces every unverified capability into one of two lies.
    Recording it as ``NOT_SUPPORTED`` hides a resource the project could use;
    recording it as ``SUPPORTED`` sends code at an interface that may not exist.
    The third state is what lets a planner prefer a verified route and schedule
    the unknown one for curation.
    """

    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    UNKNOWN = "unknown"

    @property
    def is_usable(self) -> bool:
        """Only ``SUPPORTED`` may be relied on without a curation step."""
        return self is CapabilityState.SUPPORTED

    @property
    def needs_check(self) -> bool:
        return self is CapabilityState.UNKNOWN


#: The complete, closed list of capability flags. Adding one here is a schema
#: change, which is deliberate: ad-hoc flags make registry entries incomparable.
CAPABILITY_NAMES: tuple[str, ...] = (
    "keyword_query",
    "exact_record_fetch",
    "sequence_query",
    "chemical_structure_query",
    "bulk_snapshot",
    "version_information",
    "redistribution_allowed",
)


class CapabilityFlags(BaseModel):
    """What a resource is *documented* to offer, per access route.

    These describe documented capability of the registered access modes, not
    proven behaviour: no entry in this registry has been connectivity-tested in
    this environment. Treating documentation as proof is how a run plan assumes
    a bulk snapshot that turns out to be a 2019 archive behind a login.
    """

    model_config = ConfigDict(extra="forbid")

    keyword_query: CapabilityState = Field(
        CapabilityState.UNKNOWN,
        description="Free-text or field-scoped search returning candidate records.")
    exact_record_fetch: CapabilityState = Field(
        CapabilityState.UNKNOWN,
        description="Retrieval of one record by its stable identifier.")
    sequence_query: CapabilityState = Field(
        CapabilityState.UNKNOWN,
        description="Search by protein or nucleotide sequence (not by accession).")
    chemical_structure_query: CapabilityState = Field(
        CapabilityState.UNKNOWN,
        description="Search by SMILES, InChI or substructure.")
    bulk_snapshot: CapabilityState = Field(
        CapabilityState.UNKNOWN,
        description="A downloadable whole-resource dump that can be pinned.")
    version_information: CapabilityState = Field(
        CapabilityState.UNKNOWN,
        description="A release identifier recordable in run provenance.")
    redistribution_allowed: CapabilityState = Field(
        CapabilityState.UNKNOWN,
        description="Whether derived records may be redistributed. UNKNOWN here "
                    "means the licence has not been read, not that it permits it.")

    def get(self, name: str) -> CapabilityState:
        if name not in CAPABILITY_NAMES:
            raise KeyError(f"unknown capability '{name}'; "
                           f"known: {', '.join(CAPABILITY_NAMES)}")
        return getattr(self, name)

    def supported(self, name: str) -> bool:
        """True only for ``SUPPORTED``; ``UNKNOWN`` is never treated as a yes."""
        return self.get(name).is_usable

    def unknowns(self) -> list[str]:
        """Capabilities a curator still has to determine."""
        return [n for n in CAPABILITY_NAMES if self.get(n).needs_check]

    def as_dict(self) -> dict[str, str]:
        return {n: self.get(n).value for n in CAPABILITY_NAMES}


#: Name of the machine-written observation file inside the datasource
#: directory. Excluded from the curated-source scan and merged separately, so
#: a nightly sweep never rewrites a hand-curated file.
OBSERVED_CONNECTIVITY_FILE: str = "connectivity.observed.yaml"


def _load_observed_connectivity(path: Path) -> dict[str, dict[str, Any]]:
    """Read the observation file, or return nothing if there is none.

    No file means nothing has been reached from this environment, which is the
    correct default and the state a fresh checkout is in.
    """
    if not path.is_file():
        return {}
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RegistryError(f"{path.name}: invalid YAML: {exc}") from exc
    if not isinstance(document, Mapping):
        raise RegistryError(
            f"{path.name}: top level must be a mapping. It is machine-written; "
            f"delete it and re-run `eagent sources verify --write` rather than "
            f"editing it by hand")
    out: dict[str, dict[str, Any]] = {}
    for entry in document.get("observed_connectivity") or ():
        if not isinstance(entry, Mapping) or not entry.get("id"):
            raise RegistryError(
                f"{path.name}: an observation with no source id")
        out[str(entry["id"])] = {k: v for k, v in entry.items() if k != "id"}
    return out


class ConnectivityCheck(BaseModel):
    """One call actually made to a source, and what came back.

    This is what makes ``connectivity_verified`` mean anything. A boolean on
    its own says a route works and gives nobody a way to see whether it still
    does; a check says *which* route, called how, when, and with what result,
    so the claim can be re-run and can expire.

    ``markers`` is the load-bearing field. A 200 is not a working API: a
    redirect to a friendly 404 page returns 200 with a body, and a login wall
    returns 200 with a form. The markers are strings that had to appear in the
    response for the check to pass, chosen so that only the real record
    contains them.

    ``response_sha256`` pins what was seen at that moment. It is deliberately
    *not* re-checked on a later run: a live resource changes, and a digest
    mismatch is news about the resource rather than about the route. The
    markers are what a re-run checks.
    """

    model_config = ConfigDict(extra="forbid")

    capability: str = Field(
        ..., description="Which registered capability this call exercised.")
    url: str = Field(..., description="The exact request that was made.")
    checked_at: str = Field(..., description="UTC ISO-8601 timestamp.")
    checked_by: str = Field(
        ..., description="Who or what ran it, so a stale claim has an owner.")
    ok: bool
    status_code: int | None = None
    markers: list[str] = Field(
        default_factory=list,
        description="Strings that had to appear in the response body.")
    response_sha256: str | None = None
    response_bytes: int | None = None
    elapsed_ms: int | None = None
    documentation_url: str | None = Field(
        None,
        description="The provider's own documentation for this route, fetched "
                    "in the same sweep. A documentation URL nobody opened is "
                    "an assertion; one that answered is an observation.")
    documentation_status: int | None = None
    failure: str = Field(
        "", description="Why it did not pass. Required when ok is false.")
    note: str = ""

    @field_validator("capability")
    @classmethod
    def _known_capability(cls, v: str) -> str:
        if v not in CAPABILITY_NAMES:
            raise ValueError(
                f"unknown capability '{v}'; known: {', '.join(CAPABILITY_NAMES)}")
        return v

    @model_validator(mode="after")
    def _failure_is_explained(self) -> "ConnectivityCheck":
        if not self.ok and not self.failure.strip():
            raise ValueError(
                "a failed connectivity check must say why; 'it did not work' "
                "with no reason cannot be acted on or retried")
        if self.ok and not self.markers:
            raise ValueError(
                f"the check of {self.url} passed on the status code alone. A "
                f"200 is not a working API -- a redirect to a 404 page and a "
                f"login wall both return one -- so a passing check must name "
                f"what it found in the body")
        return self


# ---------------------------------------------------------------------------
# Access modes
# ---------------------------------------------------------------------------

class AccessMode(str, enum.Enum):
    """How a resource is actually reached.

    ``OFFLINE_IMPORT`` and ``MANUAL_REVIEW_IMPORT`` exist so that a valuable
    resource with no live API is registered honestly. An author's supplementary
    CSV and a browse-only web database are both usable; what breaks a run is
    registering either of them as a REST API with a guessed base URL.
    """

    REST_API = "rest_api"
    SPARQL = "sparql"
    SOAP = "soap"
    BULK_DOWNLOAD = "bulk_download"
    LOCAL_PACKAGE = "local_package"
    OFFLINE_IMPORT = "offline_import"
    MANUAL_REVIEW_IMPORT = "manual_review_import"
    UNKNOWN = "unknown"

    @property
    def is_programmatic(self) -> bool:
        """Whether the agent can call it unattended inside a run."""
        return self in (AccessMode.REST_API, AccessMode.SPARQL, AccessMode.SOAP,
                        AccessMode.LOCAL_PACKAGE)

    @property
    def is_network_endpoint(self) -> bool:
        """Whether this mode implies a service URL the code will dial."""
        return self in (AccessMode.REST_API, AccessMode.SPARQL, AccessMode.SOAP)

    @property
    def requires_human_step(self) -> bool:
        """Whether a person must fetch or curate records before ingest."""
        return self in (AccessMode.OFFLINE_IMPORT, AccessMode.MANUAL_REVIEW_IMPORT)

    def describe(self) -> str:
        return _ACCESS_MODE_DOC[self]


_ACCESS_MODE_DOC: dict[AccessMode, str] = {
    AccessMode.REST_API: "documented HTTP API callable inside a run",
    AccessMode.SPARQL: "SPARQL endpoint callable inside a run",
    AccessMode.SOAP: "SOAP/WSDL service, usually requiring credentials",
    AccessMode.BULK_DOWNLOAD: "whole-resource archive, pinned as a snapshot",
    AccessMode.LOCAL_PACKAGE: "installed library or local dataset, no network call",
    AccessMode.OFFLINE_IMPORT: ("author CSV or archive imported by hand; valuable, "
                                "but not a live interface"),
    AccessMode.MANUAL_REVIEW_IMPORT: ("web interface only; a human curates each "
                                      "record before it enters the store"),
    AccessMode.UNKNOWN: "access route not established; a curator must determine it",
}


# ---------------------------------------------------------------------------
# DataSource
# ---------------------------------------------------------------------------

class DataSource(BaseModel):
    """One registered public resource, with its limits recorded as data.

    Two fields carry most of the weight.

    ``not_good_for`` is required and must be non-empty. It is what stops a
    planner using a reaction database as evidence of sequence-level activity, or
    a stability dataset as evidence of selectivity.

    ``evidence_strength_ceiling`` is the strongest
    :class:`~eagent.schemas.record.EvidenceStrength` an automated ingest may
    stamp on a record from this resource; anything stronger requires human
    review. It measures how tightly a claim binds to a specific sequence, and
    says nothing about whether the measured endpoint is relevant to the target
    reaction - that is what ``not_good_for`` is for. A source can be
    sequence-level experimental and still be irrelevant evidence.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    id: str = Field(..., min_length=2, pattern=r"^[a-z0-9][a-z0-9_]*[a-z0-9]$")
    display_name: str = Field(..., min_length=2)
    layers: list[DataLayer] = Field(..., min_length=1)
    good_for: list[str] = Field(..., min_length=1)
    not_good_for: list[str] = Field(
        ..., min_length=1,
        description="Required. What this resource must not be used to claim.")
    access_modes: list[AccessMode] = Field(..., min_length=1)
    capabilities: CapabilityFlags = Field(default_factory=CapabilityFlags)

    license: str | None = Field(
        None, description="Null unless known with confidence; never guessed.")
    license_source: str | None = Field(
        None, description="Where the licence statement came from, so a claim can "
                          "be traced to whoever made it.")
    needs_legal_review: bool = True
    evidence_strength_ceiling: EvidenceStrength = EvidenceStrength.ANNOTATION_ONLY

    derived_from: list[str] = Field(
        default_factory=list,
        description="Ids of resources this one re-integrates. Shared upstreams "
                    "mean shared errors and no independent corroboration.")
    derived_from_complete: bool = Field(
        True,
        description="False when the upstream list is known to be incomplete. Such "
                    "a source must not be counted as independent evidence, and "
                    "IndependenceReport.n_independent holds back every group it "
                    "sits in: an unlisted upstream may be exactly what another "
                    "group rests on, so nothing about its separateness is known.")

    connectivity_verified: bool = Field(
        False,
        description="True only alongside a ConnectivityCheck that earned it. "
                    "See the checks themselves for what was actually shown.")
    connectivity_checks: list[ConnectivityCheck] = Field(
        default_factory=list,
        description="Calls actually made from a run, each naming the capability "
                    "it exercised. Empty for a source nobody has reached.")
    endpoint: str | None = Field(
        None, description="Null unless certain. A guessed base URL is worse than "
                          "no URL, because code will call it.")
    citations: list[str] = Field(
        default_factory=list,
        description="DOI, PMID or canonical documentation identifiers.")
    version: str | None = Field(
        None, description="Null unless a release string is known; never invented.")
    approximate_record_count: int | None = Field(
        None, description="Null unless counted; never estimated from memory.")

    priority_stage: int = Field(2, ge=1, le=3)
    requires_human_review_per_record: bool = False
    needs_curation: bool = True
    curation_notes: list[str] = Field(
        default_factory=list,
        description="Exactly what a curator must confirm. Required whenever "
                    "needs_curation is true.")
    notes: str = ""

    # -- validators --------------------------------------------------------
    @field_validator("good_for", "not_good_for", "curation_notes")
    @classmethod
    def _no_blank_entries(cls, v: list[str]) -> list[str]:
        for item in v:
            if not item or not item.strip():
                raise ValueError("list entries must be non-empty statements")
        return v

    @model_validator(mode="after")
    def _verification_is_earned(self) -> "DataSource":
        """``connectivity_verified`` must be backed by a recorded call.

        The flag tells a planner a route is known to work. Asserted as a bare
        boolean it is worth nothing -- which is why this used to refuse
        ``True`` outright -- so it is now admissible exactly when the entry
        carries at least one :class:`ConnectivityCheck`: a URL, a timestamp, a
        status, a digest of what came back, and the capability it exercised.

        A check must also name a capability the entry claims. Verifying a
        keyword search and recording it against a source whose
        ``keyword_query`` flag is ``not_supported`` would make the two halves
        of the entry describe different services.
        """
        if self.connectivity_verified and not self.connectivity_checks:
            raise ValueError(
                f"{self.id}: connectivity_verified is true with no recorded "
                f"check. Record what was called, when, and what came back, or "
                f"leave it false: a bare boolean tells a planner a route works "
                f"and gives nobody a way to see whether it still does")
        if self.connectivity_checks and not self.endpoint:
            raise ValueError(
                f"{self.id}: a connectivity check is recorded but endpoint is "
                f"null. Something was called; record what")
        for check in self.connectivity_checks:
            state = self.capabilities.get(check.capability)
            if state is CapabilityState.NOT_SUPPORTED:
                raise ValueError(
                    f"{self.id}: a connectivity check exercises "
                    f"'{check.capability}', which this entry registers as "
                    f"not_supported. One half of the entry describes a "
                    f"different service from the other")
        return self

    @property
    def verified_capabilities(self) -> tuple[str, ...]:
        """Capabilities a recorded call actually exercised, in order."""
        seen: list[str] = []
        for check in self.connectivity_checks:
            if check.ok and check.capability not in seen:
                seen.append(check.capability)
        return tuple(seen)

    @field_validator("endpoint")
    @classmethod
    def _endpoint_is_a_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        if not s:
            raise ValueError("use null for an unknown endpoint, not an empty string")
        if not (s.startswith("http://") or s.startswith("https://")):
            raise ValueError(f"endpoint must be an absolute http(s) URL, got '{s}'")
        return s

    @model_validator(mode="after")
    def _coherent(self) -> "DataSource":
        if self.id in self.derived_from:
            raise ValueError(f"source '{self.id}' cannot be derived from itself")
        if len(set(self.derived_from)) != len(self.derived_from):
            raise ValueError(f"source '{self.id}' repeats an id in derived_from")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError(f"source '{self.id}' repeats a layer")

        if self.needs_curation and not self.curation_notes:
            raise ValueError(
                f"source '{self.id}' is marked needs_curation but says nothing "
                f"about what a curator must confirm; an unexplained flag cannot "
                f"be actioned")

        if self.endpoint and not self.citations:
            raise ValueError(
                f"source '{self.id}' records an endpoint but no citation; an "
                f"endpoint the code will call must be traceable to documentation")

        if self.license is None and not self.needs_legal_review:
            raise ValueError(
                f"source '{self.id}' has no recorded licence, so "
                f"needs_legal_review cannot be false")
        if self.license is not None and self.license_source is None:
            raise ValueError(
                f"source '{self.id}' states a licence without saying where that "
                f"statement came from")

        if self.endpoint and not any(m.is_network_endpoint for m in self.access_modes):
            raise ValueError(
                f"source '{self.id}' records an endpoint but no network access "
                f"mode; a web-only or author-archive resource must not be dressed "
                f"up as a service")

        if not self.derived_from and not self.derived_from_complete \
                and not self.needs_curation:
            raise ValueError(
                f"source '{self.id}' has an admittedly incomplete lineage and no "
                f"curation flag")
        return self

    # -- queries -----------------------------------------------------------
    def serves(self, layer: DataLayer) -> bool:
        return DataLayer(layer) in self.layers

    def may_claim(self, strength: EvidenceStrength) -> bool:
        """Whether an automated ingest may stamp ``strength`` on a record here."""
        return _strength_rank(strength) <= _strength_rank(self.evidence_strength_ceiling)

    @property
    def is_programmatically_reachable(self) -> bool:
        return any(m.is_programmatic for m in self.access_modes)

    @property
    def is_human_import_only(self) -> bool:
        """True when every registered route needs a person in the loop."""
        return all(m.requires_human_step or m is AccessMode.UNKNOWN
                   for m in self.access_modes)

    def summary(self) -> str:
        return (f"{self.id} ({self.display_name}) "
                f"layers={[l.value for l in self.layers]} "
                f"stage={self.priority_stage} "
                f"access={[m.value for m in self.access_modes]} "
                f"ceiling={_strength_value(self.evidence_strength_ceiling)} "
                f"endpoint={'null' if self.endpoint is None else self.endpoint} "
                f"connectivity_verified={str(self.connectivity_verified).lower()}"
                + (f" verified={list(self.verified_capabilities)}"
                   if self.verified_capabilities else ""))


def _strength_value(s: Any) -> str:
    return getattr(s, "value", str(s))


_FALLBACK_RANK: dict[str, int] = {
    "sequence_level_experimental": 4,
    "homolog_experimental": 3,
    "ec_species_mapped": 2,
    "annotation_only": 1,
    "computational_construct": 0,
}


def _strength_rank(s: Any) -> int:
    rank = getattr(s, "rank", None)
    if isinstance(rank, int):
        return rank
    return _FALLBACK_RANK[_strength_value(s)]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IndependenceReport:
    """Groups of sources that do not corroborate one another, plus caveats.

    Two counts are kept apart on purpose. :attr:`n_groups` is how many groups
    the lineage collapse produced; :attr:`n_independent` counts only the groups
    whose every member declares a *complete* lineage. A source that admits it
    does not know its own upstreams cannot be shown to be separate from any
    other group -- its unlisted upstreams may be exactly the resource another
    group rests on -- so counting it would silently convert "nobody has traced
    this" into "this is independent corroboration", which is the one error this
    report exists to prevent. Such ids stay visible in
    :attr:`incomplete_lineage` and in every rendered line rather than being
    folded into a confident number.

    ``shared_upstreams`` is keyed by the first id of a group and holds the union
    of the pairwise lineage intersections observed inside it, not the
    intersection across all its members; see
    :meth:`SourceRegistry._collapse_with_causes` for why the global
    intersection would be empty exactly when the collapse most needs
    explaining.
    """

    groups: tuple[tuple[str, ...], ...]
    shared_upstreams: Mapping[str, tuple[str, ...]]
    incomplete_lineage: tuple[str, ...]

    @property
    def n_groups(self) -> int:
        """Groups after lineage collapse, traced and untraced alike."""
        return len(self.groups)

    @property
    def counted_groups(self) -> tuple[tuple[str, ...], ...]:
        """The groups whose every member has ``derived_from_complete`` true."""
        unresolved = set(self.incomplete_lineage)
        return tuple(g for g in self.groups if not unresolved.intersection(g))

    @property
    def withheld_groups(self) -> tuple[tuple[str, ...], ...]:
        """Groups kept out of the count because a member's lineage is incomplete."""
        unresolved = set(self.incomplete_lineage)
        return tuple(g for g in self.groups if unresolved.intersection(g))

    @property
    def n_independent(self) -> int:
        """How many groups may be cited as independent corroboration.

        Deliberately not ``len(self.groups)``: a group holding a source with an
        admittedly incomplete lineage is not shown to be independent of
        anything, and the difference between the two numbers is the whole point
        of the report.
        """
        return len(self.counted_groups)

    def unresolved_in(self, group: Sequence[str]) -> tuple[str, ...]:
        """Which members of ``group`` are the reason it is not counted."""
        unresolved = set(self.incomplete_lineage)
        return tuple(s for s in group if s in unresolved)

    def report_lines(self) -> list[str]:
        lines = [f"{self.n_independent} of {self.n_groups} source group(s) "
                 f"countable as independent after lineage collapse"]
        for g in self.groups:
            shared = self.shared_upstreams.get(g[0], ())
            bits: list[str] = []
            if shared:
                bits.append(f"shared upstream: {', '.join(shared)}")
            unresolved = self.unresolved_in(g)
            if unresolved:
                bits.append("NOT counted, lineage incomplete: "
                            + ", ".join(unresolved))
            suffix = f"  ({'; '.join(bits)})" if bits else ""
            lines.append(f"  - {', '.join(g)}{suffix}")
        for sid in self.incomplete_lineage:
            lines.append(f"  ! {sid}: lineage is known to be incomplete; it must "
                         f"not be counted as independent corroboration")
        return lines

    def describe(self) -> str:
        return "\n".join(self.report_lines())


def default_datasource_dir() -> Path:
    """Where the registry YAML lives, overridable with ``EAGENT_DATASOURCE_DIR``.

    Resolved from this file so a source checkout works without installation, and
    overridable so a curated copy can be pinned for a run.
    """
    env = os.environ.get("EAGENT_DATASOURCE_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "configs" / "datasources"


class SourceRegistry:
    """Loaded registry: lookup by id and layer, lineage closure, staged rollout.

    The lineage machinery is the reason this is a class and not a dict. Four
    databases that re-publish one curated measurement will each answer a query,
    and a scoring function that counts hits will read that as four-fold support.
    :meth:`independent_source_groups` collapses them back into one.
    """

    def __init__(self, sources: Iterable[DataSource] = (),
                 files: Mapping[str, list[str]] | None = None) -> None:
        self._by_id: dict[str, DataSource] = {}
        self._files: dict[str, list[str]] = dict(files or {})
        #: Directory this registry was loaded from, when it was loaded from
        #: one. Set by :meth:`from_directory` so a tool that records a
        #: verified route writes it back to the file it came from rather than
        #: to a default path that may not be the one in use.
        self.directory: Path | None = None
        for s in sources:
            self.add(s)

    # -- loading -----------------------------------------------------------
    @classmethod
    def from_directory(cls, directory: str | Path | None = None) -> "SourceRegistry":
        """Load every ``*.yaml`` file in a datasource directory and validate it."""
        d = Path(directory) if directory is not None else default_datasource_dir()
        if not d.is_dir():
            raise RegistryError(f"datasource directory not found: {d}")
        paths = sorted(p for p in d.iterdir()
                       if p.suffix in (".yaml", ".yml") and p.is_file()
                       and p.name != OBSERVED_CONNECTIVITY_FILE)
        if not paths:
            raise RegistryError(f"no datasource YAML files in {d}")
        docs = []
        for p in paths:
            try:
                raw = yaml.safe_load(p.read_text(encoding="utf-8"))
            except yaml.YAMLError as exc:
                raise RegistryError(f"{p.name}: invalid YAML: {exc}") from exc
            docs.append((p.name, raw))
        observed = _load_observed_connectivity(d / OBSERVED_CONNECTIVITY_FILE)
        registry = cls.from_documents(docs, observed=observed)
        # Where it came from, so a tool that records a verified route can
        # write it back beside the files it was read from instead of guessing.
        registry.directory = d
        return registry

    @classmethod
    def from_documents(
        cls, documents: Iterable[tuple[str, Mapping[str, Any]]],
        observed: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> "SourceRegistry":
        """Build a registry from already-parsed YAML documents.

        Each document declares the layer its file covers, a ``sources`` list, and
        optionally ``cross_layer_refs``: ids defined in another file that also
        serve this layer. Cross-references exist so a resource that genuinely
        serves two layers is defined once; two definitions of one id would make
        lineage and independence grouping meaningless.
        """
        reg = cls()
        seen_observed: set[str] = set()
        pending_refs: list[tuple[str, DataLayer, str]] = []
        for name, raw in documents:
            if not isinstance(raw, Mapping):
                raise RegistryError(f"{name}: top level must be a mapping")
            unknown = set(raw) - {"layer", "layer_question", "agent_stages",
                                  "sources", "cross_layer_refs", "notes"}
            if unknown:
                raise RegistryError(f"{name}: unknown top-level keys: "
                                    f"{', '.join(sorted(unknown))}")
            if "layer" not in raw:
                raise RegistryError(f"{name}: missing required key 'layer'")
            try:
                layer = DataLayer(raw["layer"])
            except ValueError as exc:
                raise RegistryError(f"{name}: unknown layer "
                                    f"'{raw['layer']}'") from exc
            entries = raw.get("sources") or []
            if not isinstance(entries, list) or not entries:
                raise RegistryError(f"{name}: 'sources' must be a non-empty list")
            for entry in entries:
                if isinstance(entry, Mapping) and observed:
                    # An observation is merged onto the curated entry rather
                    # than stored in it, so one file holds the judgements and
                    # another the measurements. See
                    # eagent.datalayer.probe.write_checks.
                    overlay = observed.get(str(entry.get("id", "")))
                    if overlay:
                        entry = {**entry, **overlay}
                        seen_observed.add(str(entry.get("id", "")))
                try:
                    src = DataSource.model_validate(entry)
                except Exception as exc:
                    sid = entry.get("id", "<no id>") if isinstance(entry, Mapping) \
                        else "<not a mapping>"
                    raise RegistryError(f"{name}: source '{sid}' is invalid: "
                                        f"{exc}") from exc
                if layer not in src.layers:
                    raise RegistryError(
                        f"{name}: source '{src.id}' is defined in the {layer.value} "
                        f"file but does not declare that layer")
                reg.add(src)
                reg._files.setdefault(name, []).append(src.id)
            for ref in raw.get("cross_layer_refs") or []:
                pending_refs.append((name, layer, str(ref)))

        for name, layer, ref in pending_refs:
            if ref not in reg._by_id:
                raise RegistryIntegrityError(
                    f"{name}: cross_layer_refs names '{ref}', which is not "
                    f"registered in any datasource file")
            if layer not in reg._by_id[ref].layers:
                raise RegistryIntegrityError(
                    f"{name}: cross_layer_refs names '{ref}', but that source "
                    f"does not declare the {layer.value} layer")
        unmatched = sorted(set(observed or {}) - seen_observed)
        if unmatched:
            # A route recorded for a source nobody registers is either a
            # renamed id or a stale observation. Either way the sweep and the
            # registry disagree, and the right time to find out is at load.
            raise RegistryIntegrityError(
                f"{OBSERVED_CONNECTIVITY_FILE} records connectivity for "
                f"{', '.join(unmatched)}, which no datasource file registers. "
                f"Re-run `eagent sources verify --write`, or delete the stale "
                f"observation; a verified route for a source that does not "
                f"exist is a claim about nothing")
        reg.validate()
        return reg

    def add(self, source: DataSource) -> DataSource:
        if source.id in self._by_id:
            raise DuplicateSourceError(
                f"source id '{source.id}' is defined twice; a resource that "
                f"serves two layers is defined once with both layers listed")
        self._by_id[source.id] = source
        return source

    def validate(self) -> None:
        """Check lineage integrity. Raises rather than returning a warning list."""
        for sid, src in self._by_id.items():
            for up in src.derived_from:
                if up not in self._by_id:
                    raise RegistryIntegrityError(
                        f"source '{sid}' is derived_from '{up}', which is not "
                        f"registered; unresolved lineage would understate how "
                        f"much these resources share")
        for sid in self._by_id:
            self.upstream_closure(sid)  # raises on a cycle

    # -- lookup ------------------------------------------------------------
    def get(self, source_id: str) -> DataSource:
        try:
            return self._by_id[source_id]
        except KeyError as exc:
            raise UnknownSourceError(
                f"unknown data source '{source_id}'; registered: "
                f"{', '.join(sorted(self._by_id))}") from exc

    def __contains__(self, source_id: object) -> bool:
        return source_id in self._by_id

    def __iter__(self) -> Iterator[DataSource]:
        return iter(self._by_id[k] for k in sorted(self._by_id))

    def __len__(self) -> int:
        return len(self._by_id)

    def ids(self) -> list[str]:
        return sorted(self._by_id)

    def files(self) -> dict[str, list[str]]:
        """Which file defined which sources, for provenance in a run manifest."""
        return {k: sorted(v) for k, v in sorted(self._files.items())}

    def by_layer(self, layer: DataLayer) -> list[DataSource]:
        """Every source declaring this layer, including cross-layer entries."""
        lay = DataLayer(layer)
        return [self._by_id[k] for k in sorted(self._by_id)
                if lay in self._by_id[k].layers]

    def layer_counts(self) -> dict[DataLayer, int]:
        """Registered sources per layer, always listing all six."""
        return {l: len(self.by_layer(l)) for l in LAYER_ORDER}

    def needing_curation(self) -> list[DataSource]:
        return [s for s in self if s.needs_curation]

    def without_endpoint(self) -> list[DataSource]:
        """Sources whose network route is unresolved; the honest default."""
        return [s for s in self
                if s.endpoint is None
                and any(m.is_network_endpoint for m in s.access_modes)]

    # -- lineage -----------------------------------------------------------
    def upstream_closure(self, source_id: str) -> frozenset[str]:
        """All resources ``source_id`` ultimately re-integrates, transitively.

        Prevents the four-databases-one-measurement error: if OED re-integrates
        BRENDA, and a third resource also re-integrates BRENDA, their records can
        agree without adding any evidence.
        """
        self.get(source_id)
        out: set[str] = set()
        stack = [(source_id, (source_id,))]
        while stack:
            cur, path = stack.pop()
            for up in self.get(cur).derived_from:
                if up in path:
                    raise RegistryIntegrityError(
                        f"cyclic lineage: {' -> '.join(path + (up,))}")
                if up not in out:
                    out.add(up)
                    stack.append((up, path + (up,)))
        return frozenset(out)

    def lineage(self, source_id: str) -> frozenset[str]:
        """The source itself plus its upstream closure."""
        return frozenset({source_id}) | self.upstream_closure(source_id)

    def _collapse_with_causes(
        self, ids: Sequence[str] | None = None
    ) -> list[tuple[list[str], tuple[str, ...]]]:
        """Group by lineage overlap, keeping *which* resources each overlap was.

        Returns ``(members, shared)`` per group, members sorted, groups ordered
        by their first member. ``shared`` is the union of the pairwise lineage
        intersections seen inside the group, never the intersection across all
        of its members: groups form by transitive union, so a group whose
        members have two unrelated roots (BRENDA shares BRENDA with OED, OED
        shares SABIO-RK with CatPred-DB) has an empty global intersection. A
        report that printed that empty set would leave the collapse unexplained
        in exactly the case where a reader most needs to see what tied the
        sources together.
        """
        chosen = [self.get(i).id for i in (ids if ids is not None else self.ids())]
        chosen = sorted(dict.fromkeys(chosen))
        parent: dict[str, str] = {i: i for i in chosen}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)

        lineages = {i: self.lineage(i) for i in chosen}
        overlaps: list[tuple[str, frozenset[str]]] = []
        for n, i in enumerate(chosen):
            for j in chosen[n + 1:]:
                common = lineages[i] & lineages[j]
                if common:
                    overlaps.append((i, common))
                    union(i, j)

        groups: dict[str, list[str]] = {}
        for i in chosen:
            groups.setdefault(find(i), []).append(i)
        causes: dict[str, set[str]] = {root: set() for root in groups}
        for i, common in overlaps:
            causes[find(i)].update(common)
        return [(sorted(members), tuple(sorted(causes[root])))
                for root, members in sorted(groups.items())]

    def independent_source_groups(
        self, ids: Sequence[str] | None = None
    ) -> list[list[str]]:
        """Collapse sources that share an upstream into one group.

        Two sources belong to the same group when their lineages intersect, so
        BRENDA, a resource derived from BRENDA, and a second resource derived
        from BRENDA come back as one group. Counting groups, not hits, is what
        stops a re-published measurement being read as corroboration.

        The number of groups is **not** the number of independent sources:
        grouping uses the declared lineages, and a source that admits its
        ``derived_from`` list is incomplete forms a group of its own that
        nobody has shown to be separate from the rest. Use
        :meth:`independence_report` and its ``n_independent`` for a count that
        holds those back; ``len()`` of this list silently counts them in.
        """
        return [members for members, _ in self._collapse_with_causes(ids)]

    def independence_report(
        self, ids: Sequence[str] | None = None
    ) -> IndependenceReport:
        """Groups, what collapsed each one, and whose lineage is still unknown.

        Incompleteness is part of the result rather than a side channel: the
        report's ``n_independent`` already excludes the groups holding a source
        with an incomplete lineage, so a caller cannot get the confident number
        without the caveat travelling with it.
        """
        collapsed = self._collapse_with_causes(ids)
        shared = {members[0]: causes for members, causes in collapsed if causes}
        incomplete = sorted(
            sid for members, _ in collapsed for sid in members
            if not self.get(sid).derived_from_complete)
        return IndependenceReport(
            groups=tuple(tuple(members) for members, _ in collapsed),
            shared_upstreams=shared,
            incomplete_lineage=tuple(incomplete),
        )

    # -- rollout -----------------------------------------------------------
    def stage_plan(self, stage: int) -> list[DataSource]:
        """Sources assigned to one rollout stage, ordered by layer then id.

        Staging exists because wiring forty resources at once produces a system
        nobody can debug: stage 1 is the minimum that makes the pilot task
        answerable, stage 3 is breadth that must not be reached for before the
        seeds and family models exist.
        """
        if stage not in (1, 2, 3):
            raise ValueError(f"priority_stage must be 1, 2 or 3; got {stage!r}")
        order = {l: n for n, l in enumerate(LAYER_ORDER)}
        return sorted(
            (s for s in self if s.priority_stage == stage),
            key=lambda s: (min(order[l] for l in s.layers), s.id),
        )

    def stage_plans(self) -> dict[int, list[DataSource]]:
        return {n: self.stage_plan(n) for n in (1, 2, 3)}

    # -- reporting ---------------------------------------------------------
    def human_import_only(self) -> list[DataSource]:
        """Sources with no unattended route; a person must fetch or curate first.

        Separated out because a planner that treats these as callable schedules
        a run step that will simply never happen.
        """
        return [s for s in self if s.is_human_import_only]

    def report_lines(self) -> list[str]:
        """The whole registry in the form a reader can check against the files.

        Prints the independence summary too, because the per-layer counts read
        as coverage and coverage is what gets mistaken for corroboration.
        """
        lines = [f"{len(self)} registered sources "
                 f"(connectivity_verified=false for every one of them)"]
        for l in LAYER_ORDER:
            srcs = self.by_layer(l)
            lines.append(f"  {l.value:<26} {len(srcs):>3} sources: "
                         f"{', '.join(s.id for s in srcs) or '-'}")
        lines.append(f"  needing curation: {len(self.needing_curation())}")
        endpoints = [s.id for s in self if s.endpoint is not None]
        lines.append(f"  recording an endpoint: {len(endpoints)}"
                     + (f" ({', '.join(endpoints)})" if endpoints else ""))
        lines.append(f"  network mode but no endpoint: "
                     f"{len(self.without_endpoint())}")
        human = self.human_import_only()
        lines.append(f"  human-import only: {len(human)}"
                     + (f" ({', '.join(s.id for s in human)})" if human else ""))
        report = self.independence_report()
        lines.append(f"  {report.report_lines()[0]}")
        lines.append(f"  lineage admittedly incomplete: "
                     f"{len(report.incomplete_lineage)}")
        return lines

    def describe(self) -> str:
        return "\n".join(self.report_lines())
