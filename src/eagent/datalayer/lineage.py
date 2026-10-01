"""Counting independent evidence instead of counting database rows.

The failure this module exists to prevent
=========================================
One measurement, in one paper, is curated into BRENDA. OED re-integrates
BRENDA. SKiD re-integrates both. CatPred-DB re-integrates all three. A naive
pipeline now retrieves four rows that agree, and treats the claim as well
established by four supporting records. It is one experiment. The agreement is
an artefact of copying, and the apparent corroboration is manufactured.

The same defect ruins model evaluation. "Train on database A, test on database
B" sounds like an independent test set, and is not one when B re-curated A: the
test rows are the training rows wearing a different accession. Splitting must
therefore happen over the *original* measurement -- publication, experiment
activity, parent sequence lineage and sequence cluster -- never over the source
database. :func:`grouping_key` and :func:`leakage_safe_groups` exist for that.

What this module deliberately does **not** do
---------------------------------------------
It never invents a link and never invents a separation. Two rows are declared
the same measurement only when an identifier or a complete assay fingerprint
says so. Rows that carry no usable identity at all are returned as
*unlinkable*: they are neither merged nor counted as corroboration, and they
are named in the report so a curator can resolve them. Under-counting
independence is the safe direction; over-counting it is the dangerous one, so
every ambiguous case resolves toward "fewer independent measurements".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..provenance import sha256_text
from ..schemas.candidate import ConfidenceLevel
from ..schemas.record import EvidenceRef, EvidenceStrength, OutcomeClass

__all__ = [
    "EvidenceNode",
    "ProvenanceGraph",
    "EvidenceGroup",
    "DiscountedRow",
    "LineageReport",
    "SplitLeakage",
    "normalise_doi",
    "publication_ids",
    "activity_ids",
    "assay_fingerprint",
    "identity_tokens",
    "independent_evidence_groups",
    "count_independent",
    "corroboration_level",
    "grouping_key",
    "leakage_safe_groups",
    "split_leakage",
]


# ---------------------------------------------------------------------------
# identifier normalisation
# ---------------------------------------------------------------------------

_DOI_PREFIXES = (
    "https://doi.org/", "http://doi.org/",
    "https://dx.doi.org/", "http://dx.doi.org/",
    "doi:", "doi.org/",
)


def normalise_doi(raw: str | None) -> str | None:
    """Reduce any DOI spelling to one comparable string, or ``None``.

    Prevents the commonest false *separation*: the same paper recorded as
    ``10.1021/acscatal.0c00001`` in one resource and
    ``https://doi.org/10.1021/ACSCATAL.0c00001`` in another would otherwise be
    counted as two independent publications, inflating corroboration.
    Returns ``None`` for anything that is not DOI-shaped rather than guessing,
    because a wrong normalisation merges two genuinely distinct papers.
    """
    if not raw:
        return None
    s = str(raw).strip().lower()
    changed = True
    while changed:
        changed = False
        for pref in _DOI_PREFIXES:
            if s.startswith(pref):
                s = s[len(pref):]
                changed = True
    s = s.strip().strip("/")
    if not s.startswith("10.") or "/" not in s:
        return None
    return s


def _is_publication_source(source_type: str | None) -> bool:
    st = (source_type or "").strip().lower()
    return st in {"publication", "paper", "literature", "preprint", "patent"}


def _norm_token(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = " ".join(str(raw).split()).strip().lower()
    return s or None


def _evidence_of(obj: Any) -> list[EvidenceRef]:
    """Evidence refs attached to a record or candidate, tolerating absence."""
    refs = getattr(obj, "evidence", None) or []
    return [r for r in refs if r is not None]


def _record_id(obj: Any, fallback: str) -> str:
    for attr in ("record_id", "candidate_id", "id"):
        v = getattr(obj, attr, None)
        if v:
            return str(v)
    return fallback


def publication_ids(obj: Any) -> list[str]:
    """Canonical publication tokens for a record, strongest spelling first.

    A DOI becomes ``doi:<normalised>``; any other publication-typed identifier
    (a PMID, an internal report number) becomes ``pub:<type>:<identifier>``.
    Keeping the scheme in the token is deliberate: a PMID and a DOI for the same
    paper will **not** merge unless one row carries both. That is a visible
    curation gap, reported as such, rather than a silent merge of two papers or
    a silent split of one.
    """
    out: list[str] = []
    for ev in _evidence_of(obj):
        doi = normalise_doi(getattr(ev, "source_doi", None))
        if doi:
            out.append(f"doi:{doi}")
        ident = getattr(ev, "identifier", None)
        st = getattr(ev, "source_type", None)
        doi_from_ident = normalise_doi(ident)
        if doi_from_ident:
            out.append(f"doi:{doi_from_ident}")
        elif _is_publication_source(st):
            tok = _norm_token(ident)
            if tok:
                out.append(f"pub:{_norm_token(st)}:{tok}")
    return sorted(set(out))


def activity_ids(obj: Any) -> list[str]:
    """Canonical experiment-activity tokens for a record.

    The activity id names the measurement campaign. It is the only identifier
    that survives re-curation intact, so it is the first-priority grouping key.
    """
    out: list[str] = []
    for ev in _evidence_of(obj):
        tok = _norm_token(getattr(ev, "experiment_activity_id", None))
        if tok:
            out.append(f"activity:{tok}")
    return sorted(set(out))


def _num(v: Any) -> Any:
    """Round a float so re-curation rounding noise does not look like a new value."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    return v


def _sequence_identity(obj: Any) -> str | None:
    for attr in ("sequence_sha256", "parent_sequence_sha256", "accession"):
        v = getattr(obj, attr, None)
        if v:
            return str(v)
    seq = getattr(obj, "construct_sequence", None) or getattr(obj, "sequence", None)
    if seq:
        return "seq:" + sha256_text("".join(str(seq).split()).upper())
    return None


def _substrate_identity(obj: Any) -> str | None:
    sub = getattr(obj, "substrate", None)
    if sub is None:
        return None
    for attr in ("inchikey", "isomeric_smiles", "name"):
        v = getattr(sub, attr, None)
        if v:
            return str(v)
    return None


def _measured_values(obj: Any) -> tuple:
    det = getattr(obj, "detection", None)
    return (
        _norm_token(getattr(obj, "measurement_type", None)),
        _num(getattr(obj, "measurement_value", None)),
        _norm_token(getattr(obj, "measurement_unit", None)),
        _num(getattr(obj, "conversion_pct", None)),
        _num(getattr(obj, "ee_target_pct", None)),
        _num(getattr(obj, "specific_activity", None)),
        _norm_token(getattr(obj, "specific_activity_unit", None)),
        _num(getattr(obj, "kcat_s", None)),
        _num(getattr(obj, "km_mM", None)),
        _num(getattr(det, "limit_of_detection", None)) if det is not None else None,
    )


def assay_fingerprint(obj: Any) -> str | None:
    """Fingerprint of (sequence, substrate, conditions, outcome, measured value).

    Last-resort identity for rows that carry no publication and no activity id,
    which is the normal state of a bulk database dump. Two such rows agreeing
    on all of these are one assay re-reported, not two assays.

    Returns ``None`` when the row has no experimental outcome and no measured
    number: there is then nothing that could show it to be a re-report, and
    pretending otherwise would merge unrelated untested rows into one group.
    """
    seq = _sequence_identity(obj)
    sub = _substrate_identity(obj)
    if not seq or not sub:
        return None
    outcome = getattr(obj, "outcome", None)
    outcome_v = getattr(outcome, "value", outcome)
    values = _measured_values(obj)
    has_number = any(v is not None for v in values[1:])
    informative_outcome = bool(
        outcome is not None and getattr(outcome, "is_experimental", False)
    )
    if not has_number and not informative_outcome:
        return None
    conditions = getattr(obj, "conditions", None)
    cond_key: Any = None
    if conditions is not None and hasattr(conditions, "key"):
        cond_key = tuple(_num(x) for x in conditions.key())
    cofactor = getattr(obj, "cofactor", None)
    cof = cofactor.describe() if cofactor is not None and hasattr(cofactor, "describe") else None
    direction = getattr(obj, "reaction_direction", None)
    payload = (
        seq, sub, cof, cond_key,
        getattr(direction, "value", direction),
        outcome_v, values,
    )
    return "assay:" + sha256_text(repr(payload))[:32]


def identity_tokens(obj: Any, *, link_across_tiers: bool = True) -> tuple[list[str], str]:
    """Identity tokens for one row, plus the tier that named it.

    Tiers in priority order: ``experiment_activity``, then ``publication``,
    then ``assay_fingerprint``. The reported tier is the highest one the row
    can fill.

    ``link_across_tiers`` (default) lets a row that carries both an activity id
    and a DOI emit both tokens, so it bridges to a re-curated copy that kept
    only the DOI. That is the common real case -- the activity id is the first
    field a re-integrating resource loses -- and without the bridge the copy is
    counted as a second independent measurement, which is the exact failure
    this module exists to prevent. The cost is that two genuinely distinct
    activities reported in one paper collapse into one group. That direction is
    safe: they share the paper's selection and analysis decisions and are not
    independent corroboration anyway. Pass ``link_across_tiers=False`` for
    strict priority, where the activity id alone decides.

    The fingerprint tier never runs alongside a publication tier under either
    setting. Two genuine replications in different papers can easily report the
    same conversion under the same conditions, and merging them on that
    coincidence would erase a real independent confirmation.

    Tier ``unlinkable`` means the row carries nothing that can establish either
    identity or independence. Such rows are reported, never silently merged and
    never counted as corroboration.
    """
    acts = activity_ids(obj)
    pubs = publication_ids(obj)
    if acts:
        tokens = acts + pubs if link_across_tiers else acts
        return sorted(set(tokens)), "experiment_activity"
    if pubs:
        return pubs, "publication"
    fp = assay_fingerprint(obj)
    if fp:
        return [fp], "assay_fingerprint"
    return [], "unlinkable"


# ---------------------------------------------------------------------------
# provenance graph
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceNode:
    """One addressable origin in the provenance graph.

    Exists so "where did this number come from" has a typed answer instead of a
    free-text citation string. The ``kind`` keeps a re-integrating database
    distinguishable from the publication it copied, which is the whole point:
    only the publication is an origin.
    """

    node_id: str
    kind: str          # publication | resource | database_record | experiment_activity | record
    label: str
    version: str | None = None
    license: str | None = None

    @property
    def is_origin_kind(self) -> bool:
        """Publications and experiment activities can be origins; databases cannot."""
        return self.kind in ("publication", "experiment_activity")


class ProvenanceGraph:
    """Directed "derived from" graph over records, databases and publications.

    Built only from fields that actually travel with the data
    (:attr:`EvidenceRef.source_doi`, ``experiment_activity_id``,
    ``upstream_sources`` and the database record id), so it never asserts a
    lineage nobody recorded.

    The traversal is explicitly cycle-safe. Re-integrating resources routinely
    cite each other -- A lists B upstream while B lists A -- and a naive
    recursive walk hangs or recurses forever on exactly the dataset this module
    is meant to handle. Cycles are reported through :meth:`cycles`, not
    followed.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, EvidenceNode] = {}
        self._derives_from: dict[str, set[str]] = {}
        self._record_sources: dict[str, set[str]] = {}
        self._cycles: set[tuple[str, str]] = set()

    # -- construction ------------------------------------------------------
    def add_node(self, node: EvidenceNode) -> EvidenceNode:
        """Insert a node, keeping the first non-null version/licence seen."""
        existing = self._nodes.get(node.node_id)
        if existing is None:
            self._nodes[node.node_id] = node
            self._derives_from.setdefault(node.node_id, set())
            return node
        merged = EvidenceNode(
            node_id=existing.node_id,
            kind=existing.kind,
            label=existing.label or node.label,
            version=existing.version or node.version,
            license=existing.license or node.license,
        )
        self._nodes[existing.node_id] = merged
        return merged

    def add_edge(self, child_id: str, parent_id: str) -> None:
        """Record that ``child_id`` was derived from ``parent_id``."""
        if child_id == parent_id:
            return
        self._derives_from.setdefault(child_id, set()).add(parent_id)
        self._derives_from.setdefault(parent_id, set())

    def add_evidence(self, record_id: str, ev: EvidenceRef) -> list[str]:
        """Attach one evidence ref to a record, returning the source node ids."""
        lic = getattr(ev, "license", None)
        ver = getattr(ev, "database_version", None)
        sources: list[str] = []

        resource_ids: list[str] = []
        for name in getattr(ev, "upstream_sources", None) or []:
            tok = _norm_token(name)
            if not tok:
                continue
            nid = f"resource:{tok}"
            self.add_node(EvidenceNode(nid, "resource", str(name)))
            resource_ids.append(nid)

        pub_ids: list[str] = []
        doi = normalise_doi(getattr(ev, "source_doi", None)) \
            or normalise_doi(getattr(ev, "identifier", None))
        if doi:
            nid = f"doi:{doi}"
            self.add_node(EvidenceNode(nid, "publication", doi, license=lic))
            pub_ids.append(nid)
        elif _is_publication_source(getattr(ev, "source_type", None)):
            tok = _norm_token(getattr(ev, "identifier", None))
            if tok:
                nid = f"pub:{_norm_token(getattr(ev, 'source_type', None))}:{tok}"
                self.add_node(EvidenceNode(nid, "publication", str(ev.identifier),
                                           license=lic))
                pub_ids.append(nid)

        act = _norm_token(getattr(ev, "experiment_activity_id", None))
        act_id = None
        if act:
            act_id = f"activity:{act}"
            self.add_node(EvidenceNode(act_id, "experiment_activity",
                                       str(ev.experiment_activity_id)))

        db_id = None
        ident = _norm_token(getattr(ev, "source_record_id", None)) \
            or _norm_token(getattr(ev, "identifier", None))
        if ident and not pub_ids:
            db_id = f"dbrec:{ident}"
            label = str(getattr(ev, "source_record_id", None) or ev.identifier)
            self.add_node(EvidenceNode(db_id, "database_record", label,
                                       version=ver, license=lic))

        # database record derives from the resources it was re-integrated from,
        # and ultimately from the publication / activity that produced the number.
        for child in [x for x in (db_id,) if x]:
            for parent in resource_ids + pub_ids:
                self.add_edge(child, parent)
            if act_id:
                self.add_edge(child, act_id)
        for res in resource_ids:
            for parent in pub_ids:
                self.add_edge(res, parent)
            if act_id:
                self.add_edge(res, act_id)
        for pub in pub_ids:
            if act_id:
                # The activity produced the number; the paper reports it.
                self.add_edge(pub, act_id)

        sources = [x for x in ([db_id] + resource_ids + pub_ids + ([act_id] if act_id else []))
                   if x]
        rid = f"record:{record_id}"
        self.add_node(EvidenceNode(rid, "record", record_id))
        for s in sources:
            self.add_edge(rid, s)
        self._record_sources.setdefault(rid, set()).update(sources)
        return sources

    def add_record(self, obj: Any, fallback_id: str = "") -> str:
        """Add a record and all of its evidence; returns the record node id."""
        rid_raw = _record_id(obj, fallback_id or f"anon{len(self._record_sources)}")
        rid = f"record:{rid_raw}"
        self.add_node(EvidenceNode(rid, "record", rid_raw))
        self._record_sources.setdefault(rid, set())
        for ev in _evidence_of(obj):
            self.add_evidence(rid_raw, ev)
        return rid

    @classmethod
    def from_records(cls, records: Iterable[Any]) -> "ProvenanceGraph":
        """Build the graph for a set of rows in one pass."""
        g = cls()
        for i, r in enumerate(records):
            g.add_record(r, fallback_id=f"row{i}")
        return g

    # -- queries -----------------------------------------------------------
    @property
    def nodes(self) -> dict[str, EvidenceNode]:
        """Copy of the node table, so a caller cannot mutate the graph by accident."""
        return dict(self._nodes)

    def node(self, node_id: str) -> EvidenceNode | None:
        """Look up one node, or ``None``. Callers must handle the miss; an invented
        node would turn a gap in provenance into a false origin.
        """
        return self._nodes.get(node_id)

    def parents(self, node_id: str) -> list[str]:
        """Direct upstream sources of one node, sorted so reports are reproducible."""
        return sorted(self._derives_from.get(node_id, set()))

    def sources_of(self, record_id: str) -> list[str]:
        """Source nodes attached to one record, accepting the id with or without the
        ``record:`` prefix so callers cannot silently miss by spelling.
        """
        rid = record_id if record_id.startswith("record:") else f"record:{record_id}"
        return sorted(self._record_sources.get(rid, set()))

    def ancestors(self, node_id: str) -> list[str]:
        """Every node reachable upstream, visiting each node at most once."""
        seen: set[str] = set()
        stack = list(self._derives_from.get(node_id, set()))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for p in self._derives_from.get(cur, set()):
                if p == node_id:
                    self._cycles.add(tuple(sorted((node_id, cur))))  # type: ignore[arg-type]
                    continue
                if p not in seen:
                    stack.append(p)
        return sorted(seen)

    def cycles(self) -> list[tuple[str, str]]:
        """Mutually-citing pairs found while walking. Reported, never followed."""
        return sorted(self._cycles)

    def roots_among(self, node_ids: Iterable[str]) -> list[str]:
        """The most upstream members of a set: those no other member derives into.

        A node with no parent inside the set is upstream of the rest. When
        several survive, the caller is told there are several rather than being
        handed an arbitrary pick.
        """
        ids = [n for n in dict.fromkeys(node_ids)]
        out: list[str] = []
        for n in ids:
            anc = set(self.ancestors(n))
            if not (anc & set(ids)):
                out.append(n)
        return sorted(out)

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form, so the lineage ships inside the run manifest instead of
        being recomputed later against a changed database.
        """
        return {
            "nodes": {k: vars(v) for k, v in sorted(self._nodes.items())},
            "derives_from": {k: sorted(v) for k, v in sorted(self._derives_from.items())},
            "cycles": [list(c) for c in self.cycles()],
        }


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------

class _DisjointSet:
    """Union-find over opaque keys; no third-party graph library available."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, k: str) -> None:
        self._parent.setdefault(k, k)

    def find(self, k: str) -> str:
        self.add(k)
        root = k
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[k] != root:
            self._parent[k], k = root, self._parent[k]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic merge direction keeps group ids stable across runs.
            lo, hi = sorted((ra, rb))
            self._parent[hi] = lo

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for k in self._parent:
            out.setdefault(self.find(k), []).append(k)
        return {k: sorted(v) for k, v in out.items()}


@dataclass(frozen=True)
class EvidenceGroup:
    """One original measurement, and every row that is a copy of it.

    ``n_rows`` minus one is the number of rows that must not be counted again.
    """

    group_id: str
    key_kind: str                       # experiment_activity | publication |
                                        # assay_fingerprint | unlinkable
    key_values: tuple[str, ...]
    record_ids: tuple[str, ...]
    strongest_strength: EvidenceStrength
    representative_record_id: str
    most_upstream_source: str | None
    upstream_candidates: tuple[str, ...]
    upstream_resources: tuple[str, ...]
    outcomes: tuple[str, ...]
    notes: tuple[str, ...] = ()

    @property
    def n_rows(self) -> int:
        """Rows in this group. ``n_rows - 1`` is how many must not be counted again."""
        return len(self.record_ids)

    @property
    def is_linkable(self) -> bool:
        """Whether anything at all could establish this group's identity. An
        unlinkable group is a question for a curator, not a piece of evidence.
        """
        return self.key_kind != "unlinkable"

    @property
    def has_internal_conflict(self) -> bool:
        """Rows claimed to be one measurement that disagree about the outcome.

        That is a curation error in the sources, not a disagreement between
        experiments, and it must surface rather than be averaged away.
        """
        informative = {
            o for o in self.outcomes
            if o in (OutcomeClass.CONFIRMED_TARGET_PRODUCT.value,
                     OutcomeClass.NO_TARGET_PRODUCT_DETECTED.value,
                     OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION.value)
        }
        return len(informative) > 1

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form with enum values unwrapped, so a written deliverable never
        carries a repr like ``EvidenceStrength.ANNOTATION_ONLY`` in place of a value.
        """
        d = dict(vars(self))
        d["strongest_strength"] = self.strongest_strength.value
        for k in ("key_values", "record_ids", "upstream_candidates",
                  "upstream_resources", "outcomes", "notes"):
            d[k] = list(d[k])
        d["n_rows"] = self.n_rows
        return d


def _strength_of(obj: Any) -> EvidenceStrength:
    s = getattr(obj, "max_strength", None)
    if isinstance(s, EvidenceStrength):
        return s
    best = EvidenceStrength.COMPUTATIONAL_CONSTRUCT
    for ev in _evidence_of(obj):
        st = getattr(ev, "strength", None)
        if isinstance(st, EvidenceStrength) and st.rank > best.rank:
            best = st
    return best


def _outcome_value(obj: Any) -> str:
    o = getattr(obj, "outcome", None)
    return str(getattr(o, "value", o)) if o is not None else "unknown"


def independent_evidence_groups(
    records: Sequence[Any],
    *,
    graph: ProvenanceGraph | None = None,
    link_across_tiers: bool = True,
) -> list[EvidenceGroup]:
    """Partition rows into groups that trace to the same original measurement.

    Keys are applied in priority order -- ``experiment_activity_id``, then
    publication (DOI or other publication identifier), then the
    (sequence, substrate, conditions, outcome, measured value) assay
    fingerprint -- and grouping is the transitive closure over shared keys, so
    a row keyed by an activity id and a row keyed only by that paper's DOI land
    in the same group whenever any row carries both (see
    :func:`identity_tokens` and ``link_across_tiers``).

    Two consequences are intentional and are the reason this is safe:

    * Two distinct measurements reported in one paper collapse into one group
      under the default ``link_across_tiers=True``. They share the paper's
      selection, calibration and analysis decisions, so counting them as two
      independent confirmations overstates the evidence. Pass
      ``link_across_tiers=False`` to key strictly on the activity id when the
      activities really are separate experiments worth counting separately.
    * Rows with no identifier of any kind become one-row ``unlinkable`` groups.
      They are never merged on suspicion and never counted as corroboration.

    Returns groups sorted by ``group_id`` for reproducible reports.
    """
    recs = list(records)
    ds = _DisjointSet()
    row_keys: list[str] = []
    row_tokens: list[list[str]] = []
    row_tiers: list[str] = []

    for i, r in enumerate(recs):
        rid = _record_id(r, f"row{i}")
        row_key = f"#{i}:{rid}"
        row_keys.append(row_key)
        ds.add(row_key)
        tokens, tier = identity_tokens(r, link_across_tiers=link_across_tiers)
        row_tokens.append(tokens)
        row_tiers.append(tier)
        for t in tokens:
            ds.add(t)
            ds.union(row_key, t)

    graph = graph if graph is not None else ProvenanceGraph.from_records(recs)

    clusters = ds.groups()
    index_of: dict[str, int] = {k: i for i, k in enumerate(row_keys)}
    groups: list[EvidenceGroup] = []

    for members in clusters.values():
        idxs = sorted(index_of[m] for m in members if m in index_of)
        if not idxs:
            continue
        tokens = sorted({t for i in idxs for t in row_tokens[i]})
        tiers = {row_tiers[i] for i in idxs}
        if "experiment_activity" in tiers:
            kind = "experiment_activity"
        elif "publication" in tiers:
            kind = "publication"
        elif "assay_fingerprint" in tiers:
            kind = "assay_fingerprint"
        else:
            kind = "unlinkable"

        member_records = [recs[i] for i in idxs]
        ids = tuple(_record_id(recs[i], f"row{i}") for i in idxs)
        strengths = [_strength_of(r) for r in member_records]
        strongest = max(strengths, key=lambda s: s.rank)
        rep_pos = max(range(len(idxs)), key=lambda p: (strengths[p].rank, -idxs[p]))
        representative = ids[rep_pos]

        source_nodes: list[str] = []
        for i in idxs:
            source_nodes.extend(graph.sources_of(_record_id(recs[i], f"row{i}")))
        source_nodes = list(dict.fromkeys(source_nodes))
        roots = graph.roots_among(source_nodes)
        pub_roots = [n for n in roots
                     if (graph.node(n) or EvidenceNode(n, "?", n)).is_origin_kind]
        candidates = pub_roots or roots
        notes: list[str] = []
        if len(candidates) == 1:
            most_upstream: str | None = candidates[0]
        elif not candidates:
            most_upstream = None
            if source_nodes:
                notes.append(
                    "no origin could be named: the recorded upstream links form a "
                    "cycle (" + ", ".join(source_nodes) + "); a curator must break it"
                )
            else:
                notes.append("no upstream source recorded for any row in this group")
        else:
            most_upstream = None
            notes.append(
                "ambiguous origin: " + ", ".join(candidates)
                + " -- a curator must say which of these is the original report"
            )
        if kind == "unlinkable":
            notes.append(
                "row carries no activity id, no publication identifier and no "
                "measurable assay fingerprint; it can be shown neither "
                "independent of nor duplicated by any other row"
            )
        resources = tuple(sorted(
            (graph.node(n).label if graph.node(n) else n)
            for n in source_nodes
            if (graph.node(n) or EvidenceNode(n, "?", n)).kind in ("resource", "database_record")
        ))

        gid = "grp:" + sha256_text(repr((kind, tokens or sorted(ids))))[:12]
        groups.append(EvidenceGroup(
            group_id=gid,
            key_kind=kind,
            key_values=tuple(tokens),
            record_ids=ids,
            strongest_strength=strongest,
            representative_record_id=representative,
            most_upstream_source=most_upstream,
            upstream_candidates=tuple(candidates),
            upstream_resources=resources,
            outcomes=tuple(sorted({_outcome_value(r) for r in member_records})),
            notes=tuple(notes),
        ))

    return sorted(groups, key=lambda g: g.group_id)


def count_independent(records: Sequence[Any], *, link_across_tiers: bool = True) -> int:
    """How many distinct original measurements a pile of rows actually contains.

    This is the number to quote, never ``len(records)``. Four rows re-curated
    from one paper return 1.
    """
    return len(independent_evidence_groups(records, link_across_tiers=link_across_tiers))


def corroboration_level(
    records: Sequence[Any],
    *,
    require_experimental: bool = True,
    groups: Sequence[EvidenceGroup] | None = None,
) -> ConfidenceLevel:
    """Ordinal confidence that rises only with genuinely independent groups.

    Adding nine more copies of the same measurement cannot move this value,
    which is the entire point. Strength matters as well as count: two
    independent sequence-level experiments are ``STRONG``; one is ``MODERATE``;
    homolog-level or EC-mapped evidence alone cannot reach ``STRONG`` however
    many times it is repeated.

    With ``require_experimental`` (the default), only rows whose outcome says
    something about catalytic ability contribute. ``NOT_TESTED``,
    ``COMPUTATIONAL_FAILURE``, ``COMPUTATIONAL_NEGATIVE`` and
    ``EXPRESSION_OR_SOLUBILITY_FAILURE`` are excluded, because a modelling
    failure and an unexpressed construct are statements about the pipeline, not
    about the enzyme, and counting them as evidence either way is fabrication.

    Groups that disagree about the outcome return ``CONTRADICTORY`` rather than
    a majority vote: a conflict between independent experiments is information
    for a human, not noise to be averaged.
    """
    gs = list(groups) if groups is not None else independent_evidence_groups(records)
    by_id = {}
    for i, r in enumerate(records):
        by_id[_record_id(r, f"row{i}")] = r

    contributing: list[EvidenceGroup] = []
    for g in gs:
        if not g.is_linkable:
            continue
        rows = [by_id[rid] for rid in g.record_ids if rid in by_id]
        if require_experimental:
            ok = any(
                getattr(getattr(r, "outcome", None), "informs_catalytic_ability", False)
                for r in rows
            )
            if not ok:
                continue
        contributing.append(g)

    if not contributing:
        return ConfidenceLevel.INSUFFICIENT

    positive = {OutcomeClass.CONFIRMED_TARGET_PRODUCT.value}
    negative = {OutcomeClass.NO_TARGET_PRODUCT_DETECTED.value,
                OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION.value}
    saw_pos = any(set(g.outcomes) & positive for g in contributing)
    saw_neg = any(set(g.outcomes) & negative for g in contributing)
    if (saw_pos and saw_neg) or any(g.has_internal_conflict for g in contributing):
        return ConfidenceLevel.CONTRADICTORY

    n_seq = sum(1 for g in contributing if g.strongest_strength.is_sequence_level)
    n_exp = sum(1 for g in contributing
                if g.strongest_strength.rank >= EvidenceStrength.HOMOLOG_EXPERIMENTAL.rank)
    n_mapped = sum(1 for g in contributing
                   if g.strongest_strength.rank >= EvidenceStrength.EC_SPECIES_MAPPED.rank)

    if n_seq >= 2:
        return ConfidenceLevel.STRONG
    if n_seq == 1 or n_exp >= 2:
        return ConfidenceLevel.MODERATE
    if n_exp == 1 or n_mapped >= 1:
        return ConfidenceLevel.WEAK
    return ConfidenceLevel.INSUFFICIENT


# ---------------------------------------------------------------------------
# de-duplication keys for train/test splitting
# ---------------------------------------------------------------------------

ClusterLookup = Mapping[str, str] | Callable[[str], str | None] | None


def _cluster_for(seq_id: str | None, lookup: ClusterLookup) -> str | None:
    if not seq_id or lookup is None:
        return None
    try:
        if callable(lookup):
            v = lookup(seq_id)
        else:
            v = lookup.get(seq_id)
    except Exception:
        return None
    return str(v) if v else None


def grouping_key(obj: Any, sequence_cluster_lookup: ClusterLookup = None) -> tuple[str, ...]:
    """De-duplication facets for leakage-controlled train/test splitting.

    Returns the sorted tuple of facets a row belongs to: its publication(s), its
    experiment activity, its parent sequence lineage (so a variant and its
    parent never straddle the split), and its sequence cluster.

    **The source database is deliberately absent.** "Train on BRENDA, test on
    SKiD" is not a clean split when SKiD re-curated BRENDA: the same measurement
    appears on both sides under different accessions, and the resulting test
    score measures how well the model memorised the training rows. Splitting
    must happen over the original measurement and the sequence cluster instead.

    An unresolved sequence cluster becomes ``cluster:unresolved:<sequence id>``
    rather than a shared ``cluster:unknown``; a shared placeholder would silently
    fuse every unclustered sequence into one giant group, which looks like a
    conservative split and is actually a meaningless one.

    Two rows sharing *any* facet must not be split apart. Use
    :func:`leakage_safe_groups` for the transitive closure; comparing these
    tuples for equality alone still leaks when rows overlap on one facet only.
    """
    facets: list[str] = []
    facets.extend(activity_ids(obj))
    facets.extend(publication_ids(obj))

    parent = getattr(obj, "parent_sequence_sha256", None)
    own = getattr(obj, "sequence_sha256", None) or _sequence_identity(obj)
    lineage = str(parent or own) if (parent or own) else None
    if lineage:
        facets.append(f"lineage:{lineage}")
    else:
        facets.append(f"lineage:unresolved:{_record_id(obj, 'anon')}")

    cluster = _cluster_for(str(own) if own else None, sequence_cluster_lookup)
    if cluster is None and parent:
        cluster = _cluster_for(str(parent), sequence_cluster_lookup)
    if cluster:
        facets.append(f"cluster:{cluster}")
    else:
        facets.append(f"cluster:unresolved:{lineage or _record_id(obj, 'anon')}")

    return tuple(sorted(set(facets)))


def leakage_safe_groups(
    records: Sequence[Any],
    sequence_cluster_lookup: ClusterLookup = None,
) -> dict[str, str]:
    """Map each record id to a split group id, closed transitively over facets.

    Rows that share a paper, an activity, a parent lineage or a sequence cluster
    end up in one group, even when they share no single facet directly (row A
    shares a paper with B, B shares a cluster with C, so A, B and C must stay on
    the same side of the split). Equality of :func:`grouping_key` tuples alone
    would put A and C in different folds and leak.
    """
    ds = _DisjointSet()
    rec_keys: list[str] = []
    for i, r in enumerate(records):
        rid = _record_id(r, f"row{i}")
        key = f"#{i}:{rid}"
        rec_keys.append(key)
        ds.add(key)
        for facet in grouping_key(r, sequence_cluster_lookup):
            ds.add(facet)
            ds.union(key, facet)
    out: dict[str, str] = {}
    for i, r in enumerate(records):
        rid = _record_id(r, f"row{i}")
        out[rid] = "split:" + sha256_text(ds.find(rec_keys[i]))[:12]
    return out


@dataclass(frozen=True)
class SplitLeakage:
    """What a proposed train/test split shares across the boundary."""

    shared_group_ids: tuple[str, ...]
    train_record_ids: tuple[str, ...]
    test_record_ids: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True only when no split group spans both sides of the boundary."""
        return not self.shared_group_ids

    def render(self) -> str:
        """One-line verdict for a log, naming the rows that cross the boundary."""
        if self.ok:
            return "split clean: no split group spans train and test"
        return (
            f"SPLIT LEAKS: {len(self.shared_group_ids)} group(s) span the boundary; "
            f"train rows {', '.join(self.train_record_ids)} share an origin or "
            f"sequence cluster with test rows {', '.join(self.test_record_ids)}"
        )


def split_leakage(
    train: Sequence[Any],
    test: Sequence[Any],
    sequence_cluster_lookup: ClusterLookup = None,
) -> SplitLeakage:
    """Report rows that appear on both sides of a split under different ids.

    Prevents the headline failure of enzyme-activity benchmarks: a test score
    inflated because the "unseen" test rows are re-curations of training rows,
    or near-identical sequences from the same cluster.
    """
    combined = list(train) + list(test)
    assign = leakage_safe_groups(combined, sequence_cluster_lookup)
    train_ids = [_record_id(r, f"row{i}") for i, r in enumerate(train)]
    test_ids = [_record_id(r, f"row{i + len(train)}") for i, r in enumerate(test)]
    train_groups = {assign[i] for i in train_ids if i in assign}
    test_groups = {assign[i] for i in test_ids if i in assign}
    shared = sorted(train_groups & test_groups)
    return SplitLeakage(
        shared_group_ids=tuple(shared),
        train_record_ids=tuple(sorted(i for i in train_ids if assign.get(i) in shared)),
        test_record_ids=tuple(sorted(i for i in test_ids if assign.get(i) in shared)),
    )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscountedRow:
    """A row that was not counted, and the reason it was not counted."""

    record_id: str
    group_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form, so the reason a row was not counted survives into the file
        someone reads months later.
        """
        return dict(vars(self))


@dataclass
class LineageReport:
    """For one claim: rows seen, measurements actually behind them, what was discounted.

    Written so that the sentence "this is supported by N records" can never be
    produced without the accompanying "which are M independent measurements,
    tracing to these resources, with these K rows discounted because they are
    copies". A reviewer months later can check the arithmetic.
    """

    claim: str
    n_rows: int
    n_independent: int
    corroboration: ConfidenceLevel
    groups: list[EvidenceGroup] = field(default_factory=list)
    upstream_resources: list[str] = field(default_factory=list)
    discounted: list[DiscountedRow] = field(default_factory=list)
    unlinkable_record_ids: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        claim: str,
        records: Sequence[Any],
        *,
        require_experimental: bool = True,
    ) -> "LineageReport":
        """Assemble the report, deriving every number from the same grouping."""
        recs = list(records)
        graph = ProvenanceGraph.from_records(recs)
        groups = independent_evidence_groups(recs, graph=graph)
        level = corroboration_level(recs, require_experimental=require_experimental,
                                    groups=groups)
        by_id = {_record_id(r, f"row{i}"): r for i, r in enumerate(recs)}

        discounted: list[DiscountedRow] = []
        unlinkable: list[str] = []
        conflicts: list[str] = []
        resources: set[str] = set()

        for g in groups:
            resources.update(g.upstream_resources)
            if g.most_upstream_source:
                resources.add(g.most_upstream_source)
            if not g.is_linkable:
                unlinkable.extend(g.record_ids)
            if g.has_internal_conflict:
                conflicts.append(
                    f"{g.group_id}: rows claimed to be one measurement disagree "
                    f"about the outcome ({', '.join(g.outcomes)})"
                )
            for rid in g.record_ids:
                if rid == g.representative_record_id:
                    row = by_id.get(rid)
                    if require_experimental and row is not None and not getattr(
                        getattr(row, "outcome", None), "informs_catalytic_ability", False
                    ):
                        discounted.append(DiscountedRow(
                            rid, g.group_id,
                            f"outcome '{_outcome_value(row)}' carries no information "
                            f"about catalytic ability",
                        ))
                    continue
                discounted.append(DiscountedRow(
                    rid, g.group_id,
                    f"re-report of the same measurement (matched on {g.key_kind}); "
                    f"counted once through {g.representative_record_id}",
                ))

        return cls(
            claim=claim,
            n_rows=len(recs),
            n_independent=len(groups),
            corroboration=level,
            groups=groups,
            upstream_resources=sorted(resources),
            discounted=discounted,
            unlinkable_record_ids=sorted(set(unlinkable)),
            conflicts=conflicts,
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form keeping the row count and the independent count side by
        side, so neither number can be quoted without the other.
        """
        return {
            "claim": self.claim,
            "n_rows": self.n_rows,
            "n_independent": self.n_independent,
            "corroboration": self.corroboration.value,
            "groups": [g.to_dict() for g in self.groups],
            "upstream_resources": list(self.upstream_resources),
            "discounted": [d.to_dict() for d in self.discounted],
            "unlinkable_record_ids": list(self.unlinkable_record_ids),
            "conflicts": list(self.conflicts),
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """Plain-text rendering for a deliverable or a review meeting."""
        lines = [
            f"claim: {self.claim}",
            f"rows retrieved:           {self.n_rows}",
            f"independent measurements: {self.n_independent}",
            f"corroboration:            {self.corroboration.value}",
        ]
        lines.append("upstream resources:       "
                     + (", ".join(self.upstream_resources) or "none recorded"))
        lines.append("groups:")
        for g in self.groups:
            upstream = g.most_upstream_source or "origin not established"
            lines.append(
                f"  {g.group_id}  [{g.key_kind}]  rows={g.n_rows}  "
                f"strength={g.strongest_strength.value}  upstream={upstream}"
            )
            for n in g.notes:
                lines.append(f"      note: {n}")
        if self.discounted:
            lines.append("discounted rows:")
            for d in self.discounted:
                lines.append(f"  {d.record_id} ({d.group_id}): {d.reason}")
        else:
            lines.append("discounted rows: none")
        if self.unlinkable_record_ids:
            lines.append("unlinkable rows (no identity to check): "
                         + ", ".join(self.unlinkable_record_ids))
        for c in self.conflicts:
            lines.append(f"CONFLICT {c}")
        for n in self.notes:
            lines.append(f"note: {n}")
        return "\n".join(lines)
