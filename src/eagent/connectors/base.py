"""The data-layer contract: how the agent is allowed to touch a public resource.

Why this module exists
----------------------
Three failures recur in enzyme-mining code that talks to public databases, and
each of them is addressed by a specific construct here rather than by a comment.

*Silent substitution across layers.* Rhea defines a reaction. BRENDA records
that somebody measured an activity. M-CSA records which residues do the
chemistry. A planner that treats all three as "enzyme data" will answer "does
this enzyme reduce this ketone?" with a Rhea identifier, which is an annotation
of a transformation and says nothing about any protein. :class:`ConnectorLayer`
and :data:`CONNECTOR_REGISTRY` encode the question each resource answers and the
questions it must never be used to answer, as structured data a planner can
reason over, not as prose a reader has to remember.

*Invented database content.* A cache miss is a fact about this machine, not a
fact about the world. :class:`FileCache` and :class:`OfflineConnector` therefore
make "I do not have this record" a first-class, structured return value
(:class:`CachedResponse` with :attr:`ResponseStatus.MISS` and a populated
``needed`` list) and give no code path that produces a plausible-looking record
from nothing.

*Leaking an unpublished sequence.* An unreleased construct sent to a remote
search service is disclosed, irreversibly, and no later policy decision undoes
it. :func:`looks_like_biological_sequence` and :class:`AccessPolicy` enforce
that in code: a sequence-shaped string in an outbound payload raises
:class:`UnauthorizedSubmissionError` unless the policy carries either a named
human authorisation covering that exact sequence hash, or evidence that the
sequence is already public. The default policy authorises nothing, so the
failure mode is a refusal, never a disclosure.

The cache payload convention
----------------------------
A cached entry is a JSON envelope written by :meth:`FileCache.store`::

    {
      "cache_key":         "<sha256 of connector+version+query>",
      "connector":         "brenda",
      "connector_version": "2024.1",       # the pinned snapshot, or "unpinned"
      "database_version":  "2024.1",       # the resource's own release string,
                                           # or null when nobody recorded it
      "query":             {...},          # the exact query this answers
      "retrieved_at":      "2025-01-02T...",
      "payload_sha256":    "...",
      "payload":           {...}           # whatever the resource returned
    }

Everything except ``payload`` is written by this module and verified on read.
``payload`` is the resource's own content: it is either downloaded (only when
the policy allows network access) or placed there by a curator performing a
manual import. Nothing in this module ever generates it.
"""

from __future__ import annotations

import abc
import enum
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, Iterator, Mapping, Sequence

from ..errors import EAgentError
from ..provenance import canonical_json, sha256_text, sequence_hash, utc_now
from ..schemas.record import EvidenceStrength

__all__ = [
    "ConnectorError",
    "UnknownConnectorError",
    "CacheIntegrityError",
    "NetworkDisabledError",
    "RemoteCallFailedError",
    "UnauthorizedSubmissionError",
    "ConnectorLayer",
    "ResponseStatus",
    "CachedResponse",
    "FileCache",
    "default_cache_root",
    "SubmissionAuthorization",
    "AccessPolicy",
    "AMINO_ACID_ALPHABET",
    "MIN_SEQUENCE_LIKE_LENGTH",
    "MIN_SEQUENCE_LIKE_DISTINCT_LETTERS",
    "looks_like_biological_sequence",
    "Connector",
    "OfflineConnector",
    "ConnectorSpec",
    "CONNECTOR_REGISTRY",
    "connector_spec",
    "connector_keys",
    "specs_for_layer",
    "SubstitutionVerdict",
    "substitution",
    "StrengthDecision",
    "resolve_strength",
    "records_in",
    "cross_check_source_registry",
]


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class ConnectorError(EAgentError):
    """Base class for every failure raised by the data layer."""


class UnknownConnectorError(ConnectorError):
    """A connector key was requested that is not registered.

    Raised rather than returning ``None`` so that a typo in a query plan cannot
    silently drop a whole data layer from a run and leave the gap invisible.
    """


class CacheIntegrityError(ConnectorError):
    """A cache file exists but does not describe the query it was found under.

    Treated as an error rather than as a hit because a mismatched envelope means
    the cache is no longer a faithful replay of a retrieval, and serving its
    contents would attach one resource's answer to another resource's question.
    """


class RemoteCallFailedError(ConnectorError):
    """The route existed and the call failed: not "no such record".

    Distinct from a miss on purpose. A 404 says the service looked and there
    is nothing under that identifier, which is an answer. A timeout, a 503, a
    TLS failure or a body that is not JSON says the question was never
    answered, and reporting it as "the service returned nothing" would let a
    transient outage read as evidence that a record does not exist -- the
    sequence is then dropped from a pool for the weather.

    Carries the HTTP status when there was one so the resolver can say what
    happened instead of only that something did.
    """

    def __init__(self, source_id: str, url: str, reason: str,
                 status: int | None = None, *, transient: bool = False,
                 attempts: int = 1) -> None:
        self.source_id = source_id
        self.url = url
        self.status = status
        #: Whether trying again could plausibly help: a dropped connection, a
        #: timeout, a 429 or a 5xx. A 4xx, a body over its cap and a response
        #: that is not JSON are not transient -- the same request will fail
        #: the same way, and retrying it would only delay saying so.
        self.transient = transient
        self.attempts = attempts
        suffix = f" (after {attempts} attempts)" if attempts > 1 else ""
        super().__init__(
            f"{source_id}: the call to {url} failed: {reason}{suffix}")


class NetworkDisabledError(ConnectorError):
    """A remote call was attempted while ``allow_network`` is false.

    Offline is the default. This error exists so that a connector which forgot
    to check the policy fails loudly instead of quietly dialling out of a run
    that was supposed to be reproducible from its cache.
    """


class UnauthorizedSubmissionError(ConnectorError):
    """An outbound payload contained a sequence nobody authorised disclosing.

    Disclosure is irreversible, so this is a hard refusal rather than a warning.
    Clearing it requires a named human authorisation naming the exact sequence
    hash (see :class:`SubmissionAuthorization`), which is also what leaves an
    auditable record of who agreed to the disclosure.
    """

    def __init__(self, connector: str, sequence_sha256: str, detail: str = "") -> None:
        self.connector = connector
        self.sequence_sha256 = sequence_sha256
        msg = (f"refusing to send a sequence-shaped payload to '{connector}': "
               f"{sequence_sha256} is not covered by any submission authorisation")
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# layers
# ---------------------------------------------------------------------------

class ConnectorLayer(str, enum.Enum):
    """The question a resource answers, and the ones it cannot.

    This is finer-grained than the six-layer architecture in
    :mod:`eagent.datalayer.layers`: that module groups resources by the stage of
    the agent they serve, while this enum distinguishes the resources a
    *connector* dials, because "structure" and "mechanism" come from different
    services even though they belong to one architectural layer.
    :attr:`architecture_layer` carries the mapping back, so a planner reasoning
    at either granularity reaches the same conclusion.

    ``cannot_answer`` is the load-bearing field. A layer that only says what a
    resource is for produces a planner that reads a reaction definition as
    evidence of catalysis.
    """

    def __new__(cls, value: str, question: str, architecture_layer: str,
                cannot_answer: str) -> "ConnectorLayer":
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.question = question                        # type: ignore[attr-defined]
        obj.architecture_layer = architecture_layer    # type: ignore[attr-defined]
        obj.cannot_answer = cannot_answer              # type: ignore[attr-defined]
        return obj

    SEQUENCE = (
        "sequence",
        "What is the amino-acid sequence of this protein, and what does its "
        "entry assert about it?",
        "sequence_family_evolution",
        "An entry annotation is not a measurement; a sequence record cannot say "
        "that this protein turns over this substrate.",
    )
    STRUCTURE = (
        "structure",
        "What three-dimensional coordinates exist for this protein, and how were "
        "they obtained?",
        "structure_and_mechanism",
        "Coordinates are not activity, and a predicted model is not an observed "
        "structure.",
    )
    FAMILY = (
        "family",
        "Which domain family, signature or clan does this sequence belong to?",
        "sequence_family_evolution",
        "Family membership is not substrate scope; two members of one family "
        "routinely differ in every property this project cares about.",
    )
    REACTION = (
        "reaction",
        "What transformation is this, which bonds change, and how do the atoms "
        "map from substrate to product?",
        "reaction_and_chemistry",
        "A reaction definition names no protein and carries no evidence that any "
        "enzyme performs it.",
    )
    KINETICS = (
        "kinetics",
        "Which enzyme was measured on which substrate, under which conditions, "
        "and what was the measured value?",
        "enzymology_evidence",
        "A kinetic table does not define the reaction's atom mapping and does "
        "not identify the catalytic residues.",
    )
    MECHANISM = (
        "mechanism",
        "Which residues perform the chemistry, in which role, by which "
        "mechanistic step?",
        "structure_and_mechanism",
        "A mechanism entry describes catalysis in a reference enzyme; it is not "
        "substrate scope, rate, or stereopreference.",
    )
    LITERATURE = (
        "literature",
        "Which publications describe this, and where exactly is the claim made?",
        "literature_and_feedback",
        "A citation is a pointer to evidence, not the evidence; nothing may be "
        "ingested as a measurement from a title and abstract.",
    )


# ---------------------------------------------------------------------------
# responses
# ---------------------------------------------------------------------------

class ResponseStatus(str, enum.Enum):
    """Outcome of one connector call.

    ``MISS`` and ``REFUSED`` are deliberately separate. A miss says the cache
    does not hold the record; a refusal says policy forbade asking. Collapsing
    them would let an operator "fix" a policy refusal by populating a cache, and
    would hide from the run manifest that a disclosure was attempted.
    """

    HIT = "hit"            # served from the local cache
    FETCHED = "fetched"    # retrieved remotely and written to the cache
    MISS = "miss"          # not cached, and no permitted route to obtain it
    REFUSED = "refused"    # policy forbade the request
    ERROR = "error"        # the route existed and failed

    @property
    def has_payload(self) -> bool:
        return self in (ResponseStatus.HIT, ResponseStatus.FETCHED)


@dataclass(frozen=True)
class CachedResponse:
    """What a connector returns, whether or not it had anything to return.

    A miss carries the same shape as a hit, with ``payload=None`` and a
    populated ``needed``. The caller therefore cannot accidentally treat an
    absent record as an empty record: ``payload`` is ``None``, not ``[]``, and
    :attr:`ok` is false. ``needed`` states exactly what a human would have to
    place where, which is the difference between a run that can be repaired and
    one that merely failed.
    """

    connector: str
    data_layer: ConnectorLayer
    operation: str                      # "fetch" | "search"
    query: Mapping[str, Any]
    status: ResponseStatus
    cache_key: str
    cache_path: str
    payload: Any | None = None
    retrieved_at: str | None = None
    database_version: str | None = None
    connector_version: str = "unpinned"
    miss_reason: str | None = None
    needed: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True only when a real payload is present."""
        return self.status.has_payload and self.payload is not None

    @property
    def is_miss(self) -> bool:
        return self.status is ResponseStatus.MISS

    @property
    def version_for_provenance(self) -> str:
        """The string to record in ``Provenance.databases`` for this call.

        Falls back to the connector's pinned snapshot, and finally to
        ``"unknown"``. It never invents a release identifier: an unknown version
        is recorded as unknown so a later reader can see that the run was not
        reproducible against a pinned release.
        """
        return self.database_version or self.connector_version or "unknown"

    def payload_or_raise(self) -> Any:
        """Return the payload, or raise rather than let a caller default it."""
        if not self.ok:
            raise ConnectorError(
                f"{self.connector}.{self.operation} produced no payload "
                f"({self.status.value}): {self.miss_reason or 'no reason recorded'}"
            )
        return self.payload

    def to_dict(self) -> dict[str, Any]:
        """JSON form for artifacts and the run manifest (payload excluded).

        The payload is left out on purpose: it can be large, it already lives in
        the cache file named by ``cache_path``, and duplicating it into a
        manifest invites the copy and the cache diverging.
        """
        return {
            "connector": self.connector,
            "data_layer": self.data_layer.value,
            "operation": self.operation,
            "query": dict(self.query),
            "status": self.status.value,
            "cache_key": self.cache_key,
            "cache_path": self.cache_path,
            "payload_present": self.payload is not None,
            "retrieved_at": self.retrieved_at,
            "database_version": self.database_version,
            "connector_version": self.connector_version,
            "miss_reason": self.miss_reason,
            "needed": list(self.needed),
            "notes": list(self.notes),
        }


def records_in(payload: Any) -> list[Mapping[str, Any]]:
    """Extract the record list from a cached payload, or return an empty list.

    The convention (documented in the module docstring) is that an
    evidence-bearing payload is a mapping with a ``records`` list. This helper
    exists so that a payload shaped differently yields *nothing* rather than
    something improvised: a caller that gets ``[]`` reports a gap, which is the
    correct behaviour for a cache entry nobody has curated into shape.
    """
    if isinstance(payload, Mapping):
        recs = payload.get("records")
        if isinstance(recs, Sequence) and not isinstance(recs, (str, bytes)):
            return [r for r in recs if isinstance(r, Mapping)]
    return []


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def default_cache_root() -> Path:
    """Where cached retrievals live, overridable with ``EAGENT_CONNECTOR_CACHE``.

    Deliberately outside the per-run working directory: a cache inside the run
    would be empty at the start of every run, which would make offline replay
    impossible and would quietly turn every query into a miss.
    """
    env = os.environ.get("EAGENT_CONNECTOR_CACHE")
    if env:
        return Path(env)
    return Path.cwd() / ".eagent" / "connector_cache"


class FileCache:
    """Content-addressed store of retrievals, keyed by connector+version+query.

    The version is part of the key on purpose. Two runs against two releases of
    the same resource ask the same question and must not share an answer;
    folding them together would make a run "reproducible" against data that had
    changed underneath it.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_cache_root()

    # -- keys and paths ----------------------------------------------------
    @staticmethod
    def key_for(connector: str, version: str, query: Mapping[str, Any]) -> str:
        """sha256 over the canonical (connector, version, query) triple."""
        return sha256_text(canonical_json(
            {"connector": connector, "version": version, "query": dict(query)}
        ))

    def path_for(self, connector: str, version: str,
                 query: Mapping[str, Any]) -> Path:
        """Where this query's answer is, or would be. Safe to show in a miss."""
        key = self.key_for(connector, version, query)
        return self.root / _slug(connector) / _slug(version) / f"{key}.json"

    # -- io ----------------------------------------------------------------
    def read(self, connector: str, version: str,
             query: Mapping[str, Any]) -> dict[str, Any] | None:
        """Return the stored envelope, or ``None`` when nothing is cached.

        Raises :class:`CacheIntegrityError` when a file exists but describes a
        different query or fails its payload checksum. That is louder than a
        miss because a wrong answer served confidently is worse than no answer.
        """
        path = self.path_for(connector, version, query)
        if not path.is_file():
            return None
        try:
            env = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CacheIntegrityError(f"{path}: unreadable cache entry: {exc}") from exc
        if not isinstance(env, Mapping):
            raise CacheIntegrityError(f"{path}: cache entry is not a mapping")
        expected = self.key_for(connector, version, query)
        if env.get("cache_key") != expected:
            raise CacheIntegrityError(
                f"{path}: cache entry key {env.get('cache_key')!r} does not match "
                f"the query it was found under ({expected}); the file was edited "
                f"or copied, and its contents answer a different question"
            )
        if dict(env.get("query") or {}) != dict(query):
            raise CacheIntegrityError(
                f"{path}: cache entry records a different query than the one asked"
            )
        digest = env.get("payload_sha256")
        if digest and digest != sha256_text(canonical_json(env.get("payload"))):
            raise CacheIntegrityError(
                f"{path}: cached payload fails its recorded checksum; it was "
                f"modified after retrieval and is no longer a faithful replay"
            )
        return dict(env)

    def store(self, connector: str, version: str, query: Mapping[str, Any],
              payload: Any, *, database_version: str | None = None,
              retrieved_at: str | None = None,
              source_note: str = "") -> Path:
        """Write a retrieval (or a curated manual import) into the cache.

        ``database_version`` stays ``None`` when the importer does not know the
        resource's release string. An invented release string is worse than a
        null one, because provenance would then claim a reproducibility the run
        does not have.
        """
        path = self.path_for(connector, version, query)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "cache_key": self.key_for(connector, version, query),
            "connector": connector,
            "connector_version": version,
            "database_version": database_version,
            "query": dict(query),
            "retrieved_at": retrieved_at or utc_now(),
            "payload_sha256": sha256_text(canonical_json(payload)),
            "source_note": source_note,
            "payload": payload,
        }
        path.write_text(json.dumps(envelope, indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")
        return path


def _slug(raw: str) -> str:
    """Filesystem-safe fragment; keeps the original readable where possible."""
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in raw.strip())
    return safe or "unnamed"


# ---------------------------------------------------------------------------
# disclosure policy
# ---------------------------------------------------------------------------

#: Letters accepted as part of a biological sequence, including the ambiguity
#: codes. Nucleotide alphabets are a subset, so DNA is caught by the same test.
AMINO_ACID_ALPHABET: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO*-")

#: Length at which a letters-only string starts to be treated as a sequence.
#: A QC localisation default, not a biological constant: it is the point below
#: which a motif or an identifier is more likely than a construct. Lower it for
#: a project that handles peptides, and record the change in the run config.
MIN_SEQUENCE_LIKE_LENGTH: int = 25

#: Distinct letters required before a letters-only string is called a sequence.
#: Keeps a long unbranched SMILES such as ``CCCCCCCCCCCCCCCCCCCCCCCCCC`` from
#: being refused as a protein. Also a QC default, not a measurement.
MIN_SEQUENCE_LIKE_DISTINCT_LETTERS: int = 8

#: Letters that essentially never appear in a SMILES written with the organic
#: subset, so their presence argues for a sequence rather than a structure.
_PROTEIN_ONLY_LETTERS: frozenset[str] = frozenset("EQRKDGALVTMWY")


def looks_like_biological_sequence(value: Any) -> bool:
    """Whether a value is shaped like a protein or nucleotide sequence.

    Deliberately over-inclusive. The cost of a false positive is a refusal the
    operator can clear with an explicit authorisation; the cost of a false
    negative is an unpublished construct leaving the building. The three
    thresholds are named module constants documented as needing per-project
    calibration rather than being buried as literals here.
    """
    if not isinstance(value, str):
        return False
    s = "".join(value.split()).upper()
    if len(s) < MIN_SEQUENCE_LIKE_LENGTH:
        return False
    if not set(s) <= AMINO_ACID_ALPHABET:
        return False
    letters = {c for c in s if c.isalpha()}
    if len(letters) < MIN_SEQUENCE_LIKE_DISTINCT_LETTERS:
        return False
    return bool(letters & _PROTEIN_ONLY_LETTERS)


@dataclass(frozen=True)
class SubmissionAuthorization:
    """A named human's permission to disclose specific sequences to a service.

    ``sequence_sha256`` lists the exact sequences covered. ``any_sequence`` is a
    separate, explicit opt-in rather than the meaning of an empty list, so that
    a half-filled authorisation object cannot become a blanket permission by
    accident -- which is how blanket permissions normally come about.
    """

    authorized_by: str                   # "operator:<name>", a person, not a role
    scope: str                           # connector key, or "*" for every connector
    justification: str
    sequence_sha256: tuple[str, ...] = ()
    any_sequence: bool = False
    at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.authorized_by.strip():
            raise ValueError("an authorisation must name who gave it")
        if not self.justification.strip():
            raise ValueError("an authorisation must record why it was given")
        if not self.sequence_sha256 and not self.any_sequence:
            raise ValueError(
                "an authorisation covering no sequence and not marked "
                "any_sequence authorises nothing; state one or the other")

    def covers(self, connector: str, seq_sha256: str) -> bool:
        if self.scope not in ("*", connector):
            return False
        return self.any_sequence or seq_sha256 in self.sequence_sha256


@dataclass(frozen=True)
class AccessPolicy:
    """What this run may do with the outside world. Defaults deny everything.

    Separate from :class:`~eagent.context.ExecutionPolicy` because a connector
    is usable outside a run (in an import script, say) and must carry its own
    restrictions rather than depending on a context being present.
    :meth:`from_execution_policy` keeps the two in step when a run is driving.
    """

    allow_network: bool = False
    authorizations: tuple[SubmissionAuthorization, ...] = ()
    public_sequence_sha256: frozenset[str] = frozenset()

    @classmethod
    def from_execution_policy(
        cls, policy: Any, *,
        authorizations: Sequence[SubmissionAuthorization] = (),
        public_sequence_sha256: Iterable[str] = (),
    ) -> "AccessPolicy":
        """Mirror a run's :class:`ExecutionPolicy`, defaulting to offline.

        ``getattr`` with a ``False`` default, not an exception: a context object
        that predates the flag must behave as offline, never as online.
        """
        return cls(
            allow_network=bool(getattr(policy, "allow_network", False)),
            authorizations=tuple(authorizations),
            public_sequence_sha256=frozenset(public_sequence_sha256),
        )

    def authorization_for(self, connector: str,
                          seq_sha256: str) -> SubmissionAuthorization | None:
        for a in self.authorizations:
            if a.covers(connector, seq_sha256):
                return a
        return None

    def check_outbound(self, connector: str, payload: Any) -> list[str]:
        """Raise unless every sequence-shaped string in ``payload`` is cleared.

        Returns the notes to attach to the response (which authorisation or
        public-record finding permitted each disclosure), so the fact that a
        sequence left the building is recorded even when it was allowed.
        """
        notes: list[str] = []
        for value in _walk_strings(payload):
            if not looks_like_biological_sequence(value):
                continue
            h = sequence_hash(value)
            if h in self.public_sequence_sha256:
                notes.append(f"disclosed to {connector}: {h} (already public)")
                continue
            auth = self.authorization_for(connector, h)
            if auth is None:
                raise UnauthorizedSubmissionError(
                    connector, h,
                    "no SubmissionAuthorization covers it and it is not recorded "
                    "as already public",
                )
            notes.append(f"disclosed to {connector}: {h} "
                         f"(authorised by {auth.authorized_by} at {auth.at})")
        return notes


def _walk_strings(obj: Any) -> Iterator[str]:
    """Yield every string anywhere in a nested payload, including dict keys.

    Keys are included because a caller can and does put a sequence in a key
    (``{"<SEQ>": {...}}``); a guard that only looked at values would miss it.
    """
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            if isinstance(k, str):
                yield k
            yield from _walk_strings(v)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for item in obj:
            yield from _walk_strings(item)


# ---------------------------------------------------------------------------
# connector
# ---------------------------------------------------------------------------

class Connector(abc.ABC):
    """One public resource, reached through the cache first and the network never
    by default.

    Subclasses implement ``_fetch_remote`` and ``_search_remote``. Both are
    abstract even for a resource with no API, because the alternative -- a base
    class that silently does nothing remote -- makes "this connector cannot
    reach the network" indistinguishable from "the author forgot". A
    web-interface-only resource states that by subclassing
    :class:`OfflineConnector`, whose remote methods refuse by name.
    """

    #: Registry key; must appear in :data:`CONNECTOR_REGISTRY`.
    name: ClassVar[str] = "unnamed"
    #: Which question this resource answers.
    data_layer: ClassVar[ConnectorLayer] = ConnectorLayer.LITERATURE
    #: One line for the CLI and the query plan artifact.
    description: ClassVar[str] = ""

    def __init__(self, *, cache: FileCache | None = None,
                 snapshot: str | None = None,
                 access: AccessPolicy | None = None) -> None:
        self.cache = cache if cache is not None else FileCache()
        #: The pinned release of the underlying resource. ``"unpinned"`` is an
        #: honest statement that nobody recorded one; it is never a guess at the
        #: current release, and it keeps unpinned results in their own cache
        #: namespace so they cannot masquerade as pinned ones.
        self.version: str = snapshot or "unpinned"
        self.access = access if access is not None else AccessPolicy()

    # -- identity ----------------------------------------------------------
    @property
    def is_pinned(self) -> bool:
        """Whether this run is reproducible against a named database release."""
        return self.version != "unpinned"

    @property
    def spec(self) -> "ConnectorSpec":
        """Registry metadata, including what this resource must not be used for."""
        return connector_spec(self.name)

    # -- public api --------------------------------------------------------
    @staticmethod
    def normalise_query(operation: str, query: Mapping[str, Any] | str
                        ) -> dict[str, Any]:
        """The exact mapping the cache is keyed on, for one operation.

        Public because a curator seeding the cache has to reproduce it byte for
        byte. Leaving the normalisation private would mean every manual import
        landed under a key nothing ever looks up, and the run would report a
        miss over a file that is sitting right there.
        """
        if operation == "fetch":
            key = query if isinstance(query, str) else str(query.get("key", ""))
            if not key.strip():
                raise ConnectorError("a fetch needs a non-empty identifier")
            return {"op": "fetch", "key": key.strip()}
        if operation != "search":
            raise ConnectorError(f"unknown connector operation '{operation}'")
        if not isinstance(query, Mapping) or not query:
            raise ConnectorError("a search needs a non-empty query mapping")
        out: dict[str, Any] = {"op": "search"}
        out.update({str(k): v for k, v in sorted(query.items())})
        return out

    def cache_path_for(self, operation: str,
                       query: Mapping[str, Any] | str) -> Path:
        """Where this call's answer would be cached. Quoted in every miss."""
        return self.cache.path_for(self.name, self.version,
                                   self.normalise_query(operation, query))

    def store_import(self, operation: str, query: Mapping[str, Any] | str,
                     payload: Any, *, database_version: str | None = None,
                     retrieved_at: str | None = None,
                     source_note: str = "manual curated import") -> Path:
        """Place a curated payload where this connector will find it.

        The documented way to close a gap reported by a miss. It exists as a
        method rather than as a note in a README because the alternative -- a
        curator hand-computing a sha256 over a canonical JSON triple -- is a
        step that will be got wrong, and getting it wrong looks exactly like
        having no data.
        """
        normalised = self.normalise_query(operation, query)
        return self.cache.store(self.name, self.version, normalised, payload,
                                database_version=database_version,
                                retrieved_at=retrieved_at,
                                source_note=source_note)

    def fetch(self, key: str) -> CachedResponse:
        """Retrieve one record by its stable identifier."""
        return self._resolve("fetch", self.normalise_query("fetch", key))

    def search(self, query: Mapping[str, Any]) -> CachedResponse:
        """Run a structured query; the mapping is part of the cache key."""
        return self._resolve("search", self.normalise_query("search", query))

    # -- remote hooks ------------------------------------------------------
    @abc.abstractmethod
    def _fetch_remote(self, key: str) -> tuple[Any, str | None]:
        """Retrieve from the service. Return ``(payload, database_version)``.

        Only ever called with ``allow_network`` true and after the outbound
        disclosure guard has passed.
        """

    @abc.abstractmethod
    def _search_remote(self, query: Mapping[str, Any]) -> tuple[Any, str | None]:
        """Search the service. Return ``(payload, database_version)``."""

    # -- machinery ---------------------------------------------------------
    def _resolve(self, operation: str, query: Mapping[str, Any]) -> CachedResponse:
        """Cache first, then the network if and only if policy allows it."""
        cache_key = FileCache.key_for(self.name, self.version, query)
        cache_path = str(self.cache.path_for(self.name, self.version, query))

        try:
            envelope = self.cache.read(self.name, self.version, query)
        except CacheIntegrityError as exc:
            return self._miss(
                operation, query, cache_key, cache_path,
                reason=str(exc),
                needed=(f"repair or delete the cache file at {cache_path}",),
                status=ResponseStatus.ERROR,
            )

        if envelope is not None:
            return CachedResponse(
                connector=self.name, data_layer=self.data_layer,
                operation=operation, query=dict(query),
                status=ResponseStatus.HIT,
                cache_key=cache_key, cache_path=cache_path,
                payload=envelope.get("payload"),
                retrieved_at=envelope.get("retrieved_at"),
                database_version=envelope.get("database_version"),
                connector_version=self.version,
                notes=(() if envelope.get("database_version")
                       else ("cached entry records no database release string; "
                             "this result is not pinned to a version",)),
            )

        if not self.access.allow_network:
            return self._miss(
                operation, query, cache_key, cache_path,
                reason="not cached and network access is disabled for this run",
                needed=(
                    f"a curated import of {self.name} "
                    f"({self.data_layer.value} layer) answering "
                    f"{canonical_json(dict(query))}",
                    f"placed with "
                    f"{type(self).__name__}.store_import('{operation}', <query>, "
                    f"payload), which writes {cache_path}",
                    "or a run policy with allow_network=True and an authorised "
                    "retrieval",
                ),
            )

        notes = tuple(self.access.check_outbound(self.name, query))
        try:
            payload, db_version = (
                self._fetch_remote(str(query.get("key")))
                if operation == "fetch" else self._search_remote(query)
            )
        except NetworkDisabledError as exc:
            return self._miss(
                operation, query, cache_key, cache_path, reason=str(exc),
                needed=(f"a curated import of {self.name} written to {cache_path}",),
            )
        except RemoteCallFailedError as exc:
            # The route existed and did not answer. ERROR, not MISS: a miss
            # means the service said there is nothing, and this is the
            # opposite -- it said nothing at all. Nothing is cached, so the
            # next run asks again instead of inheriting an outage.
            return self._miss(
                operation, query, cache_key, cache_path, reason=str(exc),
                needed=("retry once the service answers, or check the "
                        "endpoint and the run's network policy",),
                status=ResponseStatus.ERROR,
            )
        if payload is None:
            return self._miss(
                operation, query, cache_key, cache_path,
                reason="the service returned nothing for this query",
                needed=("confirm the query is well formed for this resource",),
            )
        retrieved_at = utc_now()
        self.cache.store(self.name, self.version, query, payload,
                         database_version=db_version, retrieved_at=retrieved_at,
                         source_note="retrieved over the network")
        return CachedResponse(
            connector=self.name, data_layer=self.data_layer, operation=operation,
            query=dict(query), status=ResponseStatus.FETCHED,
            cache_key=cache_key, cache_path=cache_path, payload=payload,
            retrieved_at=retrieved_at, database_version=db_version,
            connector_version=self.version, notes=notes,
        )

    def _miss(self, operation: str, query: Mapping[str, Any], cache_key: str,
              cache_path: str, *, reason: str, needed: tuple[str, ...],
              status: ResponseStatus = ResponseStatus.MISS) -> CachedResponse:
        """Build the structured miss. There is no code path that fills payload."""
        return CachedResponse(
            connector=self.name, data_layer=self.data_layer, operation=operation,
            query=dict(query), status=status, cache_key=cache_key,
            cache_path=cache_path, payload=None, retrieved_at=None,
            database_version=None, connector_version=self.version,
            miss_reason=reason, needed=needed,
        )


class OfflineConnector(Connector):
    """A connector with no network route at all: cache hit, or structured miss.

    This is the default wiring for every registered resource, and the only
    behaviour available while ``allow_network`` is false. Instantiating it for a
    resource is a statement that records enter through a curated import, which
    is the honest description of how BRENDA, M-CSA and most of the others are
    actually used here.
    """

    def __init__(self, name: str, data_layer: ConnectorLayer | None = None, *,
                 cache: FileCache | None = None, snapshot: str | None = None,
                 access: AccessPolicy | None = None,
                 description: str = "") -> None:
        spec = CONNECTOR_REGISTRY.get(name)
        self.name = name
        self.data_layer = (data_layer if data_layer is not None
                           else (spec.data_layer if spec else ConnectorLayer.LITERATURE))
        self.description = description or (spec.display_name if spec else name)
        super().__init__(cache=cache, snapshot=snapshot, access=access)

    def _fetch_remote(self, key: str) -> tuple[Any, str | None]:
        raise NetworkDisabledError(
            f"{self.name} is registered as an offline connector: records enter "
            f"through a curated cache import, not through a live call")

    def _search_remote(self, query: Mapping[str, Any]) -> tuple[Any, str | None]:
        raise NetworkDisabledError(
            f"{self.name} is registered as an offline connector: records enter "
            f"through a curated cache import, not through a live call")


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConnectorSpec:
    """What one resource is, what it is not, and how strong its records may be.

    ``not_good_for`` and ``cannot_substitute_for`` are required and non-empty.
    A registry of capabilities alone produces a planner that reaches for
    whichever resource answered first; these two fields are what let it refuse.

    ``evidence_strength_ceiling`` is the strongest
    :class:`~eagent.schemas.record.EvidenceStrength` an *automated* ingest may
    stamp on a record from this resource. It measures how tightly a claim binds
    to one specific sequence and says nothing about whether the measured
    endpoint is relevant: a source can be sequence-level and still be the wrong
    evidence, which is what ``not_good_for`` is for.
    """

    key: str
    display_name: str
    data_layer: ConnectorLayer
    datasource_ids: tuple[str, ...]
    answers: str
    good_for: tuple[str, ...]
    not_good_for: tuple[str, ...]
    cannot_substitute_for: tuple[str, ...]
    identifier_kinds: tuple[str, ...]
    query_fields: tuple[str, ...]
    evidence_strength_ceiling: EvidenceStrength
    carries_experimental_context: bool
    defines_atom_mapping: bool = False
    derived_from: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.good_for:
            raise ValueError(f"{self.key}: good_for must not be empty")
        if not self.not_good_for:
            raise ValueError(
                f"{self.key}: not_good_for must not be empty; a registry that "
                f"only records strengths produces a planner that cannot refuse")
        if not self.datasource_ids:
            raise ValueError(
                f"{self.key}: must name the configs/datasources id(s) it maps to, "
                f"so the two registries can be checked against each other")

    def may_claim(self, strength: EvidenceStrength) -> bool:
        """Whether automated ingest may stamp ``strength`` on a record here."""
        return strength.rank <= self.evidence_strength_ceiling.rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "data_layer": self.data_layer.value,
            "architecture_layer": self.data_layer.architecture_layer,
            "answers": self.answers,
            "datasource_ids": list(self.datasource_ids),
            "good_for": list(self.good_for),
            "not_good_for": list(self.not_good_for),
            "cannot_substitute_for": list(self.cannot_substitute_for),
            "identifier_kinds": list(self.identifier_kinds),
            "query_fields": list(self.query_fields),
            "evidence_strength_ceiling": self.evidence_strength_ceiling.value,
            "carries_experimental_context": self.carries_experimental_context,
            "defines_atom_mapping": self.defines_atom_mapping,
            "derived_from": list(self.derived_from),
            "notes": self.notes,
        }


def _spec(**kw: Any) -> tuple[str, ConnectorSpec]:
    s = ConnectorSpec(**kw)
    return s.key, s


#: The ten resources this project is wired for. Ceilings mirror
#: ``configs/datasources/*.yaml``; :func:`cross_check_source_registry` verifies
#: that they still agree, so the two registries cannot drift apart silently.
CONNECTOR_REGISTRY: dict[str, ConnectorSpec] = dict([
    _spec(
        key="uniprot", display_name="UniProtKB",
        data_layer=ConnectorLayer.SEQUENCE,
        datasource_ids=("uniprotkb",),
        answers="What is this protein's sequence, and what does its entry assert?",
        good_for=(
            "Canonical sequences with accessions, stable enough to be a join key.",
            "Cross-references out to structures, families and reaction identifiers.",
            "Evidence codes that distinguish a curated assertion from a projection.",
        ),
        not_good_for=(
            "Not a measurement store: an annotated EC number or Rhea link is an "
            "assertion about function, not an assay on the target substrate.",
            "Entry names and protein names are not substrate scope.",
            "An accession is not an identity: isoforms and re-annotations move "
            "under one accession, so records join on the sequence hash.",
        ),
        cannot_substitute_for=("brenda", "sabio_rk", "mcsa", "rhea"),
        identifier_kinds=("uniprot_accession", "sequence_sha256"),
        query_fields=("accession", "gene", "organism", "ec", "keyword", "family"),
        evidence_strength_ceiling=EvidenceStrength.ANNOTATION_ONLY,
        carries_experimental_context=False,
        notes="Sequence spine of the run; every candidate resolves to a sequence "
              "hash here before anything else is attached to it.",
    ),
    _spec(
        key="pdb", display_name="RCSB PDB",
        data_layer=ConnectorLayer.STRUCTURE,
        datasource_ids=("rcsb_pdb",),
        answers="Which experimentally determined coordinates exist, with which "
                "ligands actually modelled?",
        good_for=(
            "Experimental coordinates with resolution and method recorded.",
            "The ligands and cofactors genuinely present in the deposited model, "
            "including their oxidation state via the chemical component id.",
            "Author numbering and the construct that was actually crystallised.",
        ),
        not_good_for=(
            "Not activity: a bound cofactor is not turnover on the target substrate.",
            "The deposited construct often differs from the wild-type sequence "
            "(tags, truncations, surface mutations), so it is not a sequence record.",
            "Occupancy and modelling decisions mean a ligand's presence in a file "
            "is not proof it was present in the experiment at full occupancy.",
        ),
        cannot_substitute_for=("alphafold", "brenda", "sabio_rk", "mcsa"),
        identifier_kinds=("pdb_id", "pdb_chain", "chem_comp_id"),
        query_fields=("pdb_id", "uniprot_accession", "ligand", "sequence"),
        evidence_strength_ceiling=EvidenceStrength.ANNOTATION_ONLY,
        carries_experimental_context=False,
        notes="The only source of observed, as opposed to predicted, placement.",
    ),
    _spec(
        key="alphafold", display_name="AlphaFold Protein Structure Database",
        data_layer=ConnectorLayer.STRUCTURE,
        datasource_ids=("alphafold_db",),
        answers="Is there a predicted model for this accession, and how confident "
                "is it per residue?",
        good_for=(
            "Backbone models for sequences with no experimental structure.",
            "Per-residue confidence that localises where a model may not be used.",
        ),
        not_good_for=(
            "A prediction is not an observation; it must never be recorded with "
            "the same standing as a deposited structure.",
            "Models carry no ligands and no cofactor, so an active site taken from "
            "one is an apo guess about a holo system.",
            "Confidence is not accuracy for the side-chain rotamers that decide "
            "whether a pocket accepts a substrate.",
        ),
        cannot_substitute_for=("pdb", "mcsa", "brenda", "sabio_rk"),
        identifier_kinds=("uniprot_accession", "alphafold_model_id"),
        query_fields=("accession",),
        evidence_strength_ceiling=EvidenceStrength.COMPUTATIONAL_CONSTRUCT,
        carries_experimental_context=False,
        notes="Ceiling is computational_construct: nothing from here is evidence.",
    ),
    _spec(
        key="interpro", display_name="InterPro / Pfam",
        data_layer=ConnectorLayer.FAMILY,
        datasource_ids=("interpro", "pfam"),
        answers="Which domain families and signatures does this sequence match?",
        good_for=(
            "Consistent family and domain assignment across a large candidate set.",
            "Domain architecture, which catches fusion and truncation artefacts "
            "before they reach the structure stage.",
            "A family label that can be joined to a sourced FamilyTemplate.",
        ),
        not_good_for=(
            "Family membership is not substrate scope and not stereopreference; "
            "members of one Pfam family differ in exactly the properties ranked here.",
            "A signature match is a model hit, not an experimental assignment.",
            "Not a mechanism source: a family does not name catalytic residues.",
        ),
        cannot_substitute_for=("mcsa", "brenda", "sabio_rk", "uniprot"),
        identifier_kinds=("interpro_id", "pfam_id"),
        query_fields=("accession", "sequence", "entry_id", "family_name"),
        evidence_strength_ceiling=EvidenceStrength.ANNOTATION_ONLY,
        carries_experimental_context=False,
        derived_from=(),
        notes="InterPro integrates Pfam; both map to one connector so that a hit "
              "in each is not counted as two independent family signals.",
    ),
    _spec(
        key="rhea", display_name="Rhea",
        data_layer=ConnectorLayer.REACTION,
        datasource_ids=("rhea",),
        answers="What exactly is this transformation, with participants given as "
                "structures?",
        good_for=(
            "A balanced reaction with ChEBI participants, so substrate and product "
            "are structures rather than names.",
            "A stable reaction id usable as an explicit join key to annotation.",
            "Direction modelled as a property, so a record can state which way it ran.",
        ),
        not_good_for=(
            "Names no protein: a Rhea id on an entry is an annotation, never "
            "evidence that that sequence catalyses the reaction.",
            "No conditions, no kinetics, no enantioselectivity.",
            "The reference direction is a curatorial convention, not the direction "
            "that was assayed.",
        ),
        cannot_substitute_for=("brenda", "sabio_rk", "mcsa", "uniprot"),
        identifier_kinds=("rhea_id", "chebi_id", "ec_number"),
        query_fields=("rhea_id", "ec", "chebi_id", "query"),
        evidence_strength_ceiling=EvidenceStrength.ANNOTATION_ONLY,
        carries_experimental_context=False,
        defines_atom_mapping=False,
        notes="Defines the target transformation during task normalisation.",
    ),
    _spec(
        key="enzymemap", display_name="EnzymeMap",
        data_layer=ConnectorLayer.REACTION,
        datasource_ids=("enzymemap",),
        answers="What is the atom-mapped, balanced form of this enzymatic reaction?",
        good_for=(
            "Atom-mapped reaction SMILES, which is what makes a reactive-atom "
            "specification checkable instead of decorative.",
            "Corrected and balanced forms of reactions that are unbalanced upstream.",
        ),
        not_good_for=(
            "Derived from BRENDA, so a hit here and a hit in BRENDA are one piece "
            "of evidence, not two.",
            "An algorithmic mapping is a hypothesis about which atom becomes which; "
            "it must be reviewed before a stereocentre claim rests on it.",
            "Not evidence that a particular sequence performs the reaction, and no "
            "conditions or kinetics.",
        ),
        cannot_substitute_for=("brenda", "sabio_rk", "mcsa", "uniprot"),
        identifier_kinds=("enzymemap_id", "rhea_id", "ec_number"),
        query_fields=("ec", "reaction_smiles", "rhea_id"),
        evidence_strength_ceiling=EvidenceStrength.ANNOTATION_ONLY,
        carries_experimental_context=False,
        defines_atom_mapping=True,
        derived_from=("brenda",),
        notes="Primary source of the atom mapping normalize_reaction validates.",
    ),
    _spec(
        key="brenda", display_name="BRENDA",
        data_layer=ConnectorLayer.KINETICS,
        datasource_ids=("brenda",),
        answers="Which enzymes have been reported to act on which substrates, "
                "under what conditions, in the literature?",
        good_for=(
            "The broadest curated view of reported substrate ranges and kinetics, "
            "with the citation attached.",
            "Finding which enzyme classes have ever been reported on a substrate "
            "resembling the target, which is how a seed set starts.",
        ),
        not_good_for=(
            "Records are frequently keyed to an EC number and an organism rather "
            "than to one sequence, so they may not be ingested as sequence-level "
            "evidence without a per-record resolution step.",
            "Values come from heterogeneous assays and must not be pooled onto one "
            "numeric scale.",
            "Substrate strings are often prose names without stereochemistry, "
            "which is fatal for an asymmetric reduction task.",
            "Direction is often implicit, so an oxidation measurement can be read "
            "as reduction evidence.",
        ),
        cannot_substitute_for=("rhea", "enzymemap", "mcsa", "pdb"),
        identifier_kinds=("ec_number", "brenda_ligand_id", "pubmed_id"),
        query_fields=("ec", "substrate", "organism", "uniprot_accession"),
        evidence_strength_ceiling=EvidenceStrength.EC_SPECIES_MAPPED,
        carries_experimental_context=True,
        notes="Treated as a pointer into the literature, not as a measurement store.",
    ),
    _spec(
        key="sabio_rk", display_name="SABIO-RK",
        data_layer=ConnectorLayer.KINETICS,
        datasource_ids=("sabio_rk",),
        answers="What kinetic parameters were measured, under which stated "
                "experimental conditions?",
        good_for=(
            "Kinetic constants carrying pH, temperature, buffer and assay context, "
            "which is what makes two numbers comparable or not.",
            "Explicit links from a parameter back to its publication.",
        ),
        not_good_for=(
            "Coverage is narrow; absence of an entry is not absence of activity.",
            "Parameters are tied to the assayed construct and conditions and do not "
            "transfer to the target substrate or solvent system.",
            "Not a sequence resource and not a reaction definition.",
        ),
        cannot_substitute_for=("rhea", "enzymemap", "mcsa", "uniprot"),
        identifier_kinds=("sabiork_entry_id", "ec_number", "pubmed_id"),
        query_fields=("ec", "substrate", "organism", "uniprot_accession", "pathway"),
        evidence_strength_ceiling=EvidenceStrength.EC_SPECIES_MAPPED,
        carries_experimental_context=True,
        notes="Preferred over BRENDA when the question is whether two measurements "
              "were made under comparable conditions.",
    ),
    _spec(
        key="mcsa", display_name="M-CSA (Mechanism and Catalytic Site Atlas)",
        data_layer=ConnectorLayer.MECHANISM,
        datasource_ids=("mcsa",),
        answers="Which residues perform the chemistry, in which role, by which step?",
        good_for=(
            "Literature-sourced catalytic residues with assigned roles, which is "
            "what a CatalyticTemplate needs and what a fold cannot supply.",
            "Stepwise mechanisms that make the cofactor's role explicit.",
        ),
        not_good_for=(
            "Coverage is partial; no entry is not evidence that the residues are "
            "unknown.",
            "An entry is evidenced for its reference enzyme, so transferring roles "
            "to a homologue is an inference that needs an explicit residue mapping.",
            "Describes mechanism, not substrate scope, rate or stereopreference.",
        ),
        cannot_substitute_for=("brenda", "sabio_rk", "rhea", "enzymemap"),
        identifier_kinds=("mcsa_id", "pdb_id", "uniprot_accession", "ec_number"),
        query_fields=("mcsa_id", "ec", "uniprot_accession", "pdb_id"),
        evidence_strength_ceiling=EvidenceStrength.EC_SPECIES_MAPPED,
        carries_experimental_context=False,
        notes="Preferred source for catalytic templates when an entry exists.",
    ),
    _spec(
        key="pubmed", display_name="PubMed",
        data_layer=ConnectorLayer.LITERATURE,
        datasource_ids=("pubmed",),
        answers="Which publications discuss this enzyme, substrate or campaign?",
        good_for=(
            "Finding the primary report behind a database record so a claim can be "
            "read at source.",
            "Catching engineering campaigns whose variants never reach a database.",
        ),
        not_good_for=(
            "A title and abstract are not data; nothing may be ingested as a "
            "measurement without reading the paper or its supplement.",
            "Indexing bias means absence of a hit is not absence of work.",
            "No structures, no sequences, no conditions in machine-readable form.",
        ),
        cannot_substitute_for=("brenda", "sabio_rk", "mcsa", "rhea", "uniprot"),
        identifier_kinds=("pubmed_id", "doi"),
        query_fields=("query", "pubmed_id", "doi", "year_from", "year_to"),
        evidence_strength_ceiling=EvidenceStrength.ANNOTATION_ONLY,
        carries_experimental_context=False,
        notes="Entry point to evidence, never the evidence itself.",
    ),
])


def connector_keys() -> list[str]:
    """Registered connector keys, sorted for stable artifacts."""
    return sorted(CONNECTOR_REGISTRY)


def connector_spec(key: str) -> ConnectorSpec:
    """Look up a connector, raising on an unknown key.

    Raises rather than returning ``None`` so a mistyped key in a query plan
    surfaces as a failure instead of as a silently narrower search.
    """
    try:
        return CONNECTOR_REGISTRY[key]
    except KeyError as exc:
        raise UnknownConnectorError(
            f"unknown connector '{key}'; registered: {', '.join(connector_keys())}"
        ) from exc


def specs_for_layer(layer: ConnectorLayer | str) -> list[ConnectorSpec]:
    """Every registered resource that answers this layer's question."""
    lay = ConnectorLayer(layer)
    return [CONNECTOR_REGISTRY[k] for k in connector_keys()
            if CONNECTOR_REGISTRY[k].data_layer is lay]


@dataclass(frozen=True)
class SubstitutionVerdict:
    """Whether one resource's records may answer another's question.

    Returned instead of a bare bool so the refusal carries its reason into the
    run manifest: "we did not use BRENDA for the atom mapping" is only auditable
    if the reason travels with the decision.
    """

    substitute: str
    target: str
    allowed: bool
    reason: str
    caveats: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.allowed


def substitution(substitute_key: str, target_key: str) -> SubstitutionVerdict:
    """May ``substitute_key``'s records stand in for ``target_key``'s?

    Two rules, both encoded as data rather than as judgement. A resource on a
    different layer answers a different question and can never substitute. A
    resource on the same layer may, unless the registry names the target in its
    ``cannot_substitute_for``, and then only with that resource's limitations
    carried along as caveats.
    """
    sub = connector_spec(substitute_key)
    tgt = connector_spec(target_key)
    if sub.key == tgt.key:
        return SubstitutionVerdict(sub.key, tgt.key, True, "same resource")
    if tgt.key in sub.cannot_substitute_for:
        return SubstitutionVerdict(
            sub.key, tgt.key, False,
            f"{sub.key} is registered as unable to substitute for {tgt.key}: "
            f"{tgt.answers}",
            caveats=sub.not_good_for,
        )
    if sub.data_layer is not tgt.data_layer:
        return SubstitutionVerdict(
            sub.key, tgt.key, False,
            f"different layers: {sub.key} answers '{sub.data_layer.question}' "
            f"while {tgt.key} answers '{tgt.data_layer.question}'; "
            f"{sub.data_layer.cannot_answer}",
            caveats=sub.not_good_for,
        )
    return SubstitutionVerdict(
        sub.key, tgt.key, True,
        f"both answer the {sub.data_layer.value} question "
        f"('{sub.data_layer.question}')",
        caveats=sub.not_good_for,
    )


@dataclass(frozen=True)
class StrengthDecision:
    """The evidence strength a record from a source is actually granted."""

    source: str
    claimed: EvidenceStrength
    granted: EvidenceStrength
    capped: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "claimed": self.claimed.value,
                "granted": self.granted.value, "capped": self.capped,
                "reason": self.reason}


def resolve_strength(key: str, claimed: EvidenceStrength, *,
                     human_reviewer: str | None = None) -> StrengthDecision:
    """Cap a claimed evidence strength at the source's ceiling.

    The ceiling is what stops an EC-number-plus-species mapping being promoted
    to sequence-level experimental evidence by an automated ingest, which is the
    single step that turns "some enzyme of this class was reported active" into
    "this protein is active". A named human reviewer may exceed the ceiling;
    nothing else may, and the reviewer's name is recorded in the reason so the
    promotion is attributable.
    """
    spec = connector_spec(key)
    ceiling = spec.evidence_strength_ceiling
    if claimed.rank <= ceiling.rank:
        return StrengthDecision(key, claimed, claimed, False,
                                f"within {key}'s ceiling ({ceiling.value})")
    if human_reviewer:
        return StrengthDecision(
            key, claimed, claimed, False,
            f"above {key}'s ceiling ({ceiling.value}) but promoted by "
            f"{human_reviewer}, who is accountable for the per-record check")
    return StrengthDecision(
        key, claimed, ceiling, True,
        f"claimed {claimed.value} exceeds {key}'s ceiling {ceiling.value}; "
        f"capped because no human reviewer resolved the record to one sequence")


def cross_check_source_registry(registry: Any = None) -> list[str]:
    """Compare this registry with ``configs/datasources``; return discrepancies.

    Two registries describing the same resources will drift. This returns the
    list of disagreements (empty when consistent) rather than raising, so a test
    can assert on it and a run can record it as a QC flag. Returns a single
    explanatory entry when the datasource registry cannot be loaded at all,
    because "I could not check" must not read the same as "they agree".
    """
    try:
        from ..datalayer.registry import SourceRegistry  # local import: optional
    except Exception as exc:  # pragma: no cover - only when datalayer is absent
        return [f"datasource registry unavailable, so nothing was checked: {exc}"]
    try:
        reg = registry if registry is not None else SourceRegistry.from_directory()
    except Exception as exc:
        return [f"datasource registry could not be loaded, so nothing was "
                f"checked: {exc}"]

    problems: list[str] = []
    for key in connector_keys():
        spec = CONNECTOR_REGISTRY[key]
        for sid in spec.datasource_ids:
            if sid not in reg:
                problems.append(
                    f"connector '{key}' maps to datasource id '{sid}', which is "
                    f"not registered in configs/datasources")
                continue
            src = reg.get(sid)
            if src.evidence_strength_ceiling != spec.evidence_strength_ceiling:
                problems.append(
                    f"connector '{key}' caps evidence at "
                    f"{spec.evidence_strength_ceiling.value} but datasource "
                    f"'{sid}' caps it at "
                    f"{src.evidence_strength_ceiling.value}")
            expected_layer = spec.data_layer.architecture_layer
            if not any(l.value == expected_layer for l in src.layers):
                problems.append(
                    f"connector '{key}' sits on architecture layer "
                    f"'{expected_layer}' but datasource '{sid}' declares "
                    f"{[l.value for l in src.layers]}")
    return problems
