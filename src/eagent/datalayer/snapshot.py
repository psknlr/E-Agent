"""Freezing the data a decision was made on.

Every formal screening round runs against a fixed snapshot. Months later,
"why was this enzyme chosen and that one not" must have an answer, and the
answer is only reconstructible if the inputs are pinned: which source files,
at which checksum, cleaned by which rules in which order, with which records
dropped and why, under which model versions, which seed and which selection
policy.

Two failures motivate every class here.

*Unattributable change.* Round 2 has a different hit rate from round 1. Was
that the method or the data? Without a snapshot per round and a
:func:`diff` between them the question cannot be answered, so the change is
attributable to neither and the round teaches nothing.

*Silent filtering.* A cleaning step quietly drops 40% of the rows -- all the
negatives, say, or every entry from one organism -- and the surviving dataset
looks clean and is biased beyond repair. :class:`ExclusionLog` and
:class:`CleaningPipeline` exist so that dropping a record without writing down
which rule dropped it is not an available code path.

Honesty rule: nothing in this module invents a version string, a date or a
count. An unknown version is ``None`` with ``needs_curation`` set and a note
saying exactly what a curator must confirm. Checksums and counts are measured
here and are therefore real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..errors import ProvenanceError
from ..provenance import sha256_file, sha256_obj, utc_now

__all__ = [
    "CleaningRule",
    "ExclusionEntry",
    "ExclusionLog",
    "CleaningPipeline",
    "FileRef",
    "SourceSpec",
    "SourceSnapshot",
    "EvidenceChain",
    "DatasetSnapshot",
    "FileDrift",
    "VerificationReport",
    "SnapshotChange",
    "SnapshotDiff",
    "freeze",
    "verify",
    "diff",
]


# ---------------------------------------------------------------------------
# cleaning rules and the exclusion log
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CleaningRule:
    """One named, versioned transformation applied to a source.

    Versioned because "we removed duplicates" is not a reproducible statement:
    the de-duplication rule changes between rounds, and an unversioned rule name
    makes that change invisible in a snapshot diff. The version is what lets
    :func:`diff` say "the data is the same but rule ``drop_fragments`` went from
    v1 to v2", which is the difference between a method result and an artefact.
    """

    name: str
    version: str
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """``name@version``: the form :func:`diff` compares, so a version bump shows."""
        return f"{self.name}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        """JSON form. The version travels with the name, because a rule name alone
        cannot tell two rounds apart.
        """
        return {"name": self.name, "version": self.version,
                "description": self.description, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CleaningRule":
        """Rebuild from JSON, requiring name and version; a rule without a version
        cannot be compared across rounds and is refused here.
        """
        return cls(name=str(d["name"]), version=str(d["version"]),
                   description=str(d.get("description") or ""),
                   params=dict(d.get("params") or {}))


@dataclass(frozen=True)
class ExclusionEntry:
    """One record dropped, by one rule, for one stated reason.

    Carries the record id so the drop can be undone, audited, or counted by
    category later. A bare count ("1 240 rows removed") cannot answer "were all
    the negatives removed?", which is the question that matters.
    """

    record_id: str
    source_id: str
    rule_name: str
    rule_version: str
    reason: str
    detail: str = ""
    at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """JSON form, keeping the record id so the drop stays auditable and reversible."""
        return dict(vars(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExclusionEntry":
        """Rebuild from JSON for auditing a past round's log."""
        return cls(
            record_id=str(d["record_id"]), source_id=str(d.get("source_id") or ""),
            rule_name=str(d["rule_name"]), rule_version=str(d.get("rule_version") or ""),
            reason=str(d.get("reason") or ""), detail=str(d.get("detail") or ""),
            at=str(d.get("at") or ""),
        )


class ExclusionLog:
    """Append-only record of every dropped row. Silent filtering is forbidden.

    This is the audit side of the contract; :class:`CleaningPipeline` is the
    enforcement side. Nothing in this module removes a record without first
    appending here, so ``n_before - len(entries for source) == n_after`` is an
    identity that :func:`freeze` checks and refuses to paper over.
    """

    def __init__(self, entries: Iterable[ExclusionEntry] | None = None) -> None:
        self._entries: list[ExclusionEntry] = list(entries or [])

    # -- mutation ----------------------------------------------------------
    def record_exclusion(
        self,
        record_id: str,
        rule: CleaningRule,
        reason: str,
        *,
        source_id: str = "",
        detail: str = "",
    ) -> ExclusionEntry:
        """Append one exclusion. Refuses an unexplained drop.

        An empty reason is rejected rather than stored, because an exclusion log
        full of blank reasons is indistinguishable from no log at all.
        """
        if not str(record_id).strip():
            raise ProvenanceError("an exclusion needs the id of the record it dropped")
        if not reason.strip():
            raise ProvenanceError(
                f"record {record_id} may not be dropped without a stated reason"
            )
        entry = ExclusionEntry(
            record_id=str(record_id), source_id=source_id, rule_name=rule.name,
            rule_version=rule.version, reason=reason.strip(), detail=detail,
        )
        self._entries.append(entry)
        return entry

    def extend(self, other: "ExclusionLog") -> None:
        """Absorb another log. Nothing is de-duplicated: two identical drops are two
        drops, and collapsing them would break the record-count arithmetic.
        """
        self._entries.extend(other.entries)

    # -- queries -----------------------------------------------------------
    @property
    def entries(self) -> list[ExclusionEntry]:
        """Copy of the entries. The log is append-only; this cannot be used to edit it."""
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def for_source(self, source_id: str) -> list[ExclusionEntry]:
        """Entries belonging to one source, for per-source accounting."""
        return [e for e in self._entries if e.source_id == source_id]

    def excluded_ids(self, source_id: str | None = None) -> set[str]:
        """Ids of dropped records, so a caller can prove which rows left the dataset."""
        return {e.record_id for e in self._entries
                if source_id is None or e.source_id == source_id}

    def counts_by_reason(self, source_id: str | None = None) -> dict[str, int]:
        """Exclusion reasons with counts; the headline number a snapshot stores."""
        out: dict[str, int] = {}
        for e in self._entries:
            if source_id is not None and e.source_id != source_id:
                continue
            out[e.reason] = out.get(e.reason, 0) + 1
        return dict(sorted(out.items()))

    def counts_by_rule(self, source_id: str | None = None) -> dict[str, int]:
        """Drops attributed to each versioned rule, which is what makes a changed
        cleaning step visible in a snapshot diff.
        """
        out: dict[str, int] = {}
        for e in self._entries:
            if source_id is not None and e.source_id != source_id:
                continue
            key = f"{e.rule_name}@{e.rule_version}"
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    def to_list(self) -> list[dict[str, Any]]:
        """JSON form of the whole log, written beside the snapshot it explains."""
        return [e.to_dict() for e in self._entries]

    @classmethod
    def from_list(cls, raw: Iterable[Mapping[str, Any]]) -> "ExclusionLog":
        """Rebuild a log from JSON so an old round's exclusions can be re-examined."""
        return cls(ExclusionEntry.from_dict(d) for d in raw)


#: A cleaning predicate: return a reason string to drop the record, or ``None``
#: to keep it. Returning a reason is the only way a record leaves the dataset.
Verdict = Callable[[Any], str | None]


class CleaningPipeline:
    """An ordered list of named rules, each of which must explain every drop.

    There is no "filter" entry point that takes a plain boolean predicate. A
    rule reports *why* it rejects a record, and :meth:`run` writes that reason
    to the :class:`ExclusionLog` before the record disappears. Making the
    logging structurally unavoidable is the point: a reviewer can trust the
    counts because there is no other route out of the dataset.
    """

    def __init__(self, source_id: str = "") -> None:
        self.source_id = source_id
        self._rules: list[tuple[CleaningRule, Verdict]] = []

    def add(self, rule: CleaningRule, verdict: Verdict) -> "CleaningPipeline":
        """Append a rule. Order is preserved and recorded in the snapshot."""
        self._rules.append((rule, verdict))
        return self

    @property
    def rules(self) -> list[CleaningRule]:
        """The rules in application order, as they are recorded in the snapshot."""
        return [r for r, _ in self._rules]

    def run(
        self,
        records: Sequence[Any],
        log: ExclusionLog,
        *,
        id_of: Callable[[Any], str] | None = None,
        source_id: str | None = None,
    ) -> list[Any]:
        """Apply every rule in order, logging each drop; returns the survivors.

        A record that no rule rejects is returned untouched. A record rejected
        by a rule is logged with that rule's name and version and does not reach
        later rules, so the log attributes each drop to exactly one rule.
        """
        src = source_id if source_id is not None else self.source_id
        kept: list[Any] = []
        for i, rec in enumerate(records):
            rid = _default_id(rec, i, src) if id_of is None else str(id_of(rec))
            dropped = False
            for rule, verdict in self._rules:
                reason = verdict(rec)
                if reason:
                    log.record_exclusion(rid, rule, reason, source_id=src)
                    dropped = True
                    break
            if not dropped:
                kept.append(rec)
        return kept


def _default_id(rec: Any, index: int, source_id: str) -> str:
    """Identify a record, falling back to a positional id rather than inventing one."""
    for attr in ("record_id", "candidate_id", "id"):
        v = getattr(rec, attr, None)
        if v:
            return str(v)
    if isinstance(rec, Mapping):
        for k in ("record_id", "candidate_id", "id"):
            if rec.get(k):
                return str(rec[k])
    return f"{source_id or 'source'}#row{index}"


# ---------------------------------------------------------------------------
# snapshot pieces
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileRef:
    """A local input file pinned by checksum.

    The path alone is worthless as provenance: files are overwritten in place by
    the next download. The checksum is what makes "the decision was made on this
    data" a checkable statement rather than a claim.
    """

    path: str
    sha256: str
    size_bytes: int
    modified_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON form carrying the checksum, which is the only part that proves identity."""
        return dict(vars(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FileRef":
        """Rebuild from JSON; the checksum is required, since a path alone pins nothing."""
        return cls(path=str(d["path"]), sha256=str(d["sha256"]),
                   size_bytes=int(d.get("size_bytes") or 0),
                   modified_at=d.get("modified_at"))


@dataclass
class SourceSpec:
    """What the caller declares about one source, before it is frozen.

    Separate from :class:`SourceSnapshot` because the snapshot's checksums and
    exclusion counts are *measured* by :func:`freeze`; only the things a machine
    cannot measure (which release this is, when it was downloaded) are declared
    here, and an undeclared one stays ``None``.
    """

    source_id: str
    paths: list[str | Path] = field(default_factory=list)
    version: str | None = None
    retrieved_at: str | None = None
    cleaning_rules: list[CleaningRule] = field(default_factory=list)
    n_records_before: int | None = None
    n_records_after: int | None = None
    exclusion_log: ExclusionLog | None = None
    license: str | None = None
    curation_notes: list[str] = field(default_factory=list)


@dataclass
class SourceSnapshot:
    """One source as it stood when the round was run."""

    source_id: str
    version: str | None
    retrieved_at: str | None
    files: list[FileRef] = field(default_factory=list)
    cleaning_rules: list[CleaningRule] = field(default_factory=list)
    n_records_before: int | None = None
    n_records_after: int | None = None
    exclusions: dict[str, int] = field(default_factory=dict)
    exclusions_by_rule: dict[str, int] = field(default_factory=dict)
    license: str | None = None
    needs_curation: bool = False
    curation_notes: list[str] = field(default_factory=list)

    @property
    def n_excluded(self) -> int:
        """Total logged exclusions: the number :func:`freeze` checks the record
        arithmetic against, so an unlogged drop cannot hide here.
        """
        return sum(self.exclusions.values())

    def to_dict(self) -> dict[str, Any]:
        """JSON form of one frozen source, including its curation gaps."""
        return {
            "source_id": self.source_id,
            "version": self.version,
            "retrieved_at": self.retrieved_at,
            "files": [f.to_dict() for f in self.files],
            "cleaning_rules": [r.to_dict() for r in self.cleaning_rules],
            "n_records_before": self.n_records_before,
            "n_records_after": self.n_records_after,
            "exclusions": dict(self.exclusions),
            "exclusions_by_rule": dict(self.exclusions_by_rule),
            "license": self.license,
            "needs_curation": self.needs_curation,
            "curation_notes": list(self.curation_notes),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SourceSnapshot":
        """Rebuild from JSON, preserving nulls rather than defaulting them to values
        nobody confirmed.
        """
        return cls(
            source_id=str(d["source_id"]),
            version=d.get("version"),
            retrieved_at=d.get("retrieved_at"),
            files=[FileRef.from_dict(f) for f in d.get("files") or []],
            cleaning_rules=[CleaningRule.from_dict(r) for r in d.get("cleaning_rules") or []],
            n_records_before=d.get("n_records_before"),
            n_records_after=d.get("n_records_after"),
            exclusions=dict(d.get("exclusions") or {}),
            exclusions_by_rule=dict(d.get("exclusions_by_rule") or {}),
            license=d.get("license"),
            needs_curation=bool(d.get("needs_curation")),
            curation_notes=list(d.get("curation_notes") or []),
        )


@dataclass
class EvidenceChain:
    """Why one finally selected candidate was selected, in traceable form.

    Stored inside the snapshot so the justification is pinned to the same frozen
    data as the decision. A justification that lives only in a report can be
    re-read against a changed database and appear to say something it never said.
    """

    candidate_id: str
    record_ids: list[str] = field(default_factory=list)
    independent_measurements: int | None = None
    group_ids: list[str] = field(default_factory=list)
    upstream_resources: list[str] = field(default_factory=list)
    corroboration: str | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON form, so the justification is stored with the decision it justifies."""
        return dict(vars(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EvidenceChain":
        """Rebuild from JSON, leaving unknown counts as ``None``."""
        return cls(
            candidate_id=str(d["candidate_id"]),
            record_ids=list(d.get("record_ids") or []),
            independent_measurements=d.get("independent_measurements"),
            group_ids=list(d.get("group_ids") or []),
            upstream_resources=list(d.get("upstream_resources") or []),
            corroboration=d.get("corroboration"),
            note=str(d.get("note") or ""),
        )

    @classmethod
    def from_lineage_report(cls, candidate_id: str, report: Any) -> "EvidenceChain":
        """Build a chain from a :class:`~eagent.datalayer.lineage.LineageReport`.

        Duck-typed on purpose so the snapshot layer does not depend on the
        lineage layer being importable; a missing attribute leaves the field
        ``None`` instead of being filled with a plausible number.
        """
        groups = list(getattr(report, "groups", []) or [])
        corr = getattr(report, "corroboration", None)
        record_ids: list[str] = []
        for g in groups:
            record_ids.extend(list(getattr(g, "record_ids", ()) or ()))
        return cls(
            candidate_id=candidate_id,
            record_ids=sorted(set(record_ids)),
            independent_measurements=getattr(report, "n_independent", None),
            group_ids=sorted(str(getattr(g, "group_id", "")) for g in groups),
            upstream_resources=sorted(getattr(report, "upstream_resources", []) or []),
            corroboration=str(getattr(corr, "value", corr)) if corr is not None else None,
            note=str(getattr(report, "claim", "") or ""),
        )


@dataclass
class DatasetSnapshot:
    """The complete frozen input state of one screening round."""

    snapshot_id: str
    created_at: str = field(default_factory=utc_now)
    workdir: str | None = None
    round_label: str = ""
    sources: list[SourceSnapshot] = field(default_factory=list)
    model_versions: dict[str, str] = field(default_factory=dict)
    random_seed: int | None = None
    selection_policy_id: str | None = None
    evidence_chains: list[EvidenceChain] = field(default_factory=list)
    content_sha256: str = ""
    needs_curation: bool = False
    curation_notes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # -- queries -----------------------------------------------------------
    def source(self, source_id: str) -> SourceSnapshot | None:
        """One source by id, or ``None`` if this round did not use it."""
        for s in self.sources:
            if s.source_id == source_id:
                return s
        return None

    @property
    def total_records_before(self) -> int | None:
        """Rows across all sources before cleaning, or ``None`` if no source reported it."""
        vals = [s.n_records_before for s in self.sources]
        return sum(v for v in vals if v is not None) if any(v is not None for v in vals) else None

    @property
    def total_records_after(self) -> int | None:
        """Rows across all sources after cleaning, or ``None`` if no source reported it."""
        vals = [s.n_records_after for s in self.sources]
        return sum(v for v in vals if v is not None) if any(v is not None for v in vals) else None

    @property
    def total_excluded(self) -> int:
        """Rows dropped across all sources, each one accounted for by a log entry."""
        return sum(s.n_excluded for s in self.sources)

    def all_files(self) -> list[tuple[str, FileRef]]:
        """Every pinned file with its source id, for verification and reporting."""
        return [(s.source_id, f) for s in self.sources for f in s.files]

    # -- io ----------------------------------------------------------------
    def payload(self) -> dict[str, Any]:
        """The content-bearing part, excluding timestamps and the id itself.

        Hashing this and not ``created_at`` means two freezes of genuinely
        identical data produce the same ``content_sha256``, so "the data did not
        change between rounds" is a one-line check.
        """
        return {
            "sources": [s.to_dict() for s in self.sources],
            "model_versions": dict(sorted(self.model_versions.items())),
            "random_seed": self.random_seed,
            "selection_policy_id": self.selection_policy_id,
            "evidence_chains": [c.to_dict() for c in self.evidence_chains],
        }

    def to_dict(self) -> dict[str, Any]:
        """Full JSON form: the content payload plus the identity and curation fields."""
        d = self.payload()
        d.update({
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "workdir": self.workdir,
            "round_label": self.round_label,
            "content_sha256": self.content_sha256,
            "needs_curation": self.needs_curation,
            "curation_notes": list(self.curation_notes),
            "notes": list(self.notes),
        })
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DatasetSnapshot":
        """Rebuild from JSON without re-deriving anything, so a loaded snapshot says
        exactly what was written and not what today's code would compute.
        """
        return cls(
            snapshot_id=str(d["snapshot_id"]),
            created_at=str(d.get("created_at") or ""),
            workdir=d.get("workdir"),
            round_label=str(d.get("round_label") or ""),
            sources=[SourceSnapshot.from_dict(s) for s in d.get("sources") or []],
            model_versions=dict(d.get("model_versions") or {}),
            random_seed=d.get("random_seed"),
            selection_policy_id=d.get("selection_policy_id"),
            evidence_chains=[EvidenceChain.from_dict(c)
                             for c in d.get("evidence_chains") or []],
            content_sha256=str(d.get("content_sha256") or ""),
            needs_curation=bool(d.get("needs_curation")),
            curation_notes=list(d.get("curation_notes") or []),
            notes=list(d.get("notes") or []),
        )

    def write(self, path: str | Path) -> Path:
        """Write the snapshot as JSON next to the data it pins."""
        import json

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False,
                                sort_keys=False, default=str), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "DatasetSnapshot":
        """Read a snapshot back, refusing a file that was edited by hand.

        The stored ``content_sha256`` is recomputed; a mismatch raises rather
        than loading, because a hand-edited snapshot is worse than no snapshot:
        it carries the authority of a record while describing data that was
        never used.
        """
        import json

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        snap = cls.from_dict(raw)
        expected = snap.content_sha256
        actual = sha256_obj(snap.payload())
        if expected and expected != actual:
            raise ProvenanceError(
                f"snapshot {path} has been modified since it was written "
                f"(content sha256 {actual} != recorded {expected})"
            )
        return snap


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------

def _resolve(path: str | Path, workdir: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (workdir / p)


def freeze(
    sources: Sequence[SourceSpec],
    workdir: str | Path,
    *,
    round_label: str = "",
    model_versions: Mapping[str, str] | None = None,
    random_seed: int | None = None,
    selection_policy_id: str | None = None,
    evidence_chains: Sequence[EvidenceChain] | None = None,
    strict: bool = True,
) -> DatasetSnapshot:
    """Pin the current state of every declared source into a snapshot.

    Measures what can be measured (file checksums, sizes, exclusion counts) and
    refuses to fill in what cannot. A missing declared file raises
    :class:`~eagent.errors.ProvenanceError`: a snapshot that silently omits an
    input is a snapshot that cannot answer the question it exists for.

    With ``strict`` (the default), a record-count discrepancy also raises. If
    ``n_before - n_excluded != n_after`` then records left the dataset without
    passing through the exclusion log, which is precisely the silent filtering
    this module forbids. With ``strict=False`` the discrepancy is recorded as a
    curation note instead, for the case where a legacy source is being ingested
    and the gap itself is the finding.

    An unknown ``version`` or ``retrieved_at`` is kept as ``None`` with
    ``needs_curation`` set and a note naming what must be confirmed. It is never
    replaced by "latest", by today's date, or by anything else invented here.
    """
    wd = Path(workdir)
    snaps: list[SourceSnapshot] = []
    global_notes: list[str] = []

    seen_ids: set[str] = set()
    for spec in sources:
        if spec.source_id in seen_ids:
            raise ProvenanceError(
                f"duplicate source_id '{spec.source_id}' in a single snapshot; "
                f"two different datasets cannot share one identity"
            )
        seen_ids.add(spec.source_id)

        files: list[FileRef] = []
        for raw_path in spec.paths:
            p = _resolve(raw_path, wd)
            if not p.exists():
                raise ProvenanceError(
                    f"source '{spec.source_id}' declares file {p} which does not "
                    f"exist; refusing to freeze a snapshot with a missing input"
                )
            if not p.is_file():
                raise ProvenanceError(
                    f"source '{spec.source_id}' declares {p}, which is not a file"
                )
            st = p.stat()
            files.append(FileRef(
                path=str(raw_path),
                sha256=sha256_file(p),
                size_bytes=st.st_size,
                modified_at=_iso_mtime(st.st_mtime),
            ))

        log = spec.exclusion_log
        exclusions = log.counts_by_reason(spec.source_id) if log is not None else {}
        by_rule = log.counts_by_rule(spec.source_id) if log is not None else {}
        n_excluded = sum(exclusions.values())

        notes = list(spec.curation_notes)
        needs = False
        if spec.version is None:
            needs = True
            notes.append(
                "version is null: a curator must confirm the exact release "
                "identifier of this source as downloaded"
            )
        if spec.retrieved_at is None:
            needs = True
            notes.append(
                "retrieval date is null: a curator must confirm when these files "
                "were downloaded"
            )
        if not files:
            needs = True
            notes.append(
                "no local file was pinned for this source, so its content cannot "
                "be re-verified; a curator must attach the retrieved files"
            )

        before, after = spec.n_records_before, spec.n_records_after
        if before is not None and after is not None:
            if before - n_excluded != after:
                msg = (
                    f"source '{spec.source_id}': {before} records before cleaning "
                    f"minus {n_excluded} logged exclusions is "
                    f"{before - n_excluded}, but {after} records survived. "
                    f"Records left the dataset without an exclusion log entry."
                )
                if strict:
                    raise ProvenanceError(msg)
                needs = True
                notes.append(msg)
                global_notes.append(msg)
        elif before is None or after is None:
            needs = True
            notes.append(
                "record counts before and/or after cleaning are null: a curator "
                "must supply them so the exclusion arithmetic can be checked"
            )

        snaps.append(SourceSnapshot(
            source_id=spec.source_id,
            version=spec.version,
            retrieved_at=spec.retrieved_at,
            files=files,
            cleaning_rules=list(spec.cleaning_rules),
            n_records_before=before,
            n_records_after=after,
            exclusions=exclusions,
            exclusions_by_rule=by_rule,
            license=spec.license,
            needs_curation=needs,
            curation_notes=notes,
        ))

    snap = DatasetSnapshot(
        snapshot_id="",
        workdir=str(wd),
        round_label=round_label,
        sources=snaps,
        model_versions=dict(model_versions or {}),
        random_seed=random_seed,
        selection_policy_id=selection_policy_id,
        evidence_chains=list(evidence_chains or []),
        notes=global_notes,
    )
    curation_notes: list[str] = []
    if random_seed is None:
        curation_notes.append(
            "random_seed is null: the round is not reproducible until the seed "
            "actually used is recorded"
        )
    if selection_policy_id is None:
        curation_notes.append(
            "selection_policy_id is null: without it, 'why this candidate' cannot "
            "be answered from the snapshot alone"
        )
    if not model_versions:
        curation_notes.append(
            "no model versions recorded: a changed structure predictor would be "
            "indistinguishable from a changed dataset in the next diff"
        )
    snap.curation_notes = curation_notes
    snap.needs_curation = bool(curation_notes) or any(s.needs_curation for s in snaps)
    snap.content_sha256 = sha256_obj(snap.payload())
    snap.snapshot_id = "snap:" + snap.content_sha256[:16]
    return snap


def _iso_mtime(ts: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileDrift:
    """One pinned file that no longer matches the snapshot."""

    source_id: str
    path: str
    status: str                 # ok | modified | missing | unreadable | size_changed
    expected_sha256: str
    actual_sha256: str | None = None
    detail: str = ""

    @property
    def is_ok(self) -> bool:
        """Whether this file still matches the checksum it was frozen at."""
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        """JSON form, carrying both the expected and the observed checksum."""
        return dict(vars(self))


@dataclass
class VerificationReport:
    """Result of re-hashing a snapshot's files: what drifted, and how."""

    snapshot_id: str
    checked_at: str = field(default_factory=utc_now)
    results: list[FileDrift] = field(default_factory=list)
    content_sha256_ok: bool = True
    messages: list[str] = field(default_factory=list)

    @property
    def drift(self) -> list[FileDrift]:
        """Only the files that no longer match, which is what a reader acts on."""
        return [r for r in self.results if not r.is_ok]

    @property
    def ok(self) -> bool:
        """True only when every pinned file matches and the snapshot body is intact."""
        return not self.drift and self.content_sha256_ok

    def to_dict(self) -> dict[str, Any]:
        """JSON form, so a drift finding can be attached to the run manifest."""
        return {
            "snapshot_id": self.snapshot_id,
            "checked_at": self.checked_at,
            "ok": self.ok,
            "content_sha256_ok": self.content_sha256_ok,
            "results": [r.to_dict() for r in self.results],
            "messages": list(self.messages),
        }

    def render(self) -> str:
        """Human-readable drift report for a log or a review."""
        lines = [f"snapshot {self.snapshot_id}: "
                 f"{'verified' if self.ok else 'DRIFT DETECTED'}"]
        if not self.content_sha256_ok:
            lines.append("  snapshot body does not match its recorded content hash")
        for r in self.results:
            if r.is_ok:
                continue
            lines.append(f"  [{r.status}] {r.source_id}: {r.path}")
            lines.append(f"      expected {r.expected_sha256}")
            lines.append(f"      actual   {r.actual_sha256 or '-'}")
            if r.detail:
                lines.append(f"      {r.detail}")
        for m in self.messages:
            lines.append(f"  note: {m}")
        return "\n".join(lines)


def verify(snapshot: DatasetSnapshot, workdir: str | Path | None = None) -> VerificationReport:
    """Re-hash every pinned file and report drift instead of continuing silently.

    Returns a report rather than raising: the caller decides whether a changed
    input invalidates the round or is an expected re-download. What it must not
    do is proceed as though the data were the one that was frozen, which is what
    happens when nobody checks. ``ok`` is False when any file is missing,
    unreadable or changed, or when the snapshot body itself was edited.
    """
    wd = Path(workdir) if workdir is not None else Path(snapshot.workdir or ".")
    results: list[FileDrift] = []
    messages: list[str] = []

    for src in snapshot.sources:
        if not src.files:
            messages.append(
                f"source '{src.source_id}' pinned no files; its content cannot be "
                f"verified at all"
            )
        for fr in src.files:
            p = _resolve(fr.path, wd)
            if not p.exists():
                results.append(FileDrift(src.source_id, fr.path, "missing",
                                         fr.sha256, None,
                                         f"expected at {p}"))
                continue
            try:
                actual = sha256_file(p)
            except OSError as exc:
                results.append(FileDrift(src.source_id, fr.path, "unreadable",
                                         fr.sha256, None, str(exc)))
                continue
            if actual != fr.sha256:
                size = p.stat().st_size
                status = "size_changed" if size != fr.size_bytes else "modified"
                detail = (f"size {fr.size_bytes} -> {size} bytes"
                          if size != fr.size_bytes
                          else "same size, different content")
                results.append(FileDrift(src.source_id, fr.path, status,
                                         fr.sha256, actual, detail))
            else:
                results.append(FileDrift(src.source_id, fr.path, "ok",
                                         fr.sha256, actual))

    body_ok = True
    if snapshot.content_sha256:
        body_ok = snapshot.content_sha256 == sha256_obj(snapshot.payload())
    return VerificationReport(
        snapshot_id=snapshot.snapshot_id,
        results=results,
        content_sha256_ok=body_ok,
        messages=messages,
    )


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SnapshotChange:
    """One difference between two rounds, with what it threatens."""

    kind: str          # source_added | source_removed | version | file | counts |
                       # cleaning_rules | model_version | seed | selection_policy |
                       # exclusions | license
    subject: str
    before: Any
    after: Any
    detail: str = ""

    @property
    def is_data_change(self) -> bool:
        """Whether this change alters the inputs rather than the method.

        Cleaning rules count as a data change: a different pipeline produces a
        different dataset, whatever the sources were. Only the model versions,
        the seed and the selection policy are counted as the method, so a
        changed cleaning step can never be mistaken for a methodological
        improvement.
        """
        return self.kind in ("source_added", "source_removed", "version", "file",
                             "counts", "exclusions", "license", "cleaning_rules")

    def to_dict(self) -> dict[str, Any]:
        """JSON form including ``is_data_change``, so the attribution is not re-derived
        by whoever reads the file.
        """
        d = dict(vars(self))
        d["is_data_change"] = self.is_data_change
        return d


@dataclass
class SnapshotDiff:
    """What changed between two rounds, split into data changes and method changes.

    A changed hit rate between rounds must be attributable either to the method
    or to the data. This object is what makes the attribution possible: if the
    data side is empty, the method moved the number; if the method side is
    empty, the data did; if both changed, the comparison does not support a
    causal claim and the diff says so.
    """

    snapshot_a: str
    snapshot_b: str
    changes: list[SnapshotChange] = field(default_factory=list)

    @property
    def data_changes(self) -> list[SnapshotChange]:
        """Changes to the inputs: sources, versions, files, counts, exclusions, cleaning."""
        return [c for c in self.changes if c.is_data_change]

    @property
    def method_changes(self) -> list[SnapshotChange]:
        """Changes to the method: model versions, the seed and the selection policy."""
        return [c for c in self.changes if not c.is_data_change]

    @property
    def identical(self) -> bool:
        """True when nothing changed, so a result difference is nondeterminism alone."""
        return not self.changes

    @property
    def attribution(self) -> str:
        """Plain statement of what a difference in results may be attributed to."""
        d, m = bool(self.data_changes), bool(self.method_changes)
        if not d and not m:
            return ("inputs and method are identical; any difference in results "
                    "comes from nondeterminism, not from either")
        if d and not m:
            return "method unchanged: a difference in results is attributable to the data"
        if m and not d:
            return "data unchanged: a difference in results is attributable to the method"
        return ("both data and method changed: a difference in results is "
                "attributable to neither, and the rounds are not comparable")

    def to_dict(self) -> dict[str, Any]:
        """JSON form including the attribution sentence, which is the part a reader
        needs and the part most easily lost in a summary.
        """
        return {
            "snapshot_a": self.snapshot_a,
            "snapshot_b": self.snapshot_b,
            "identical": self.identical,
            "attribution": self.attribution,
            "changes": [c.to_dict() for c in self.changes],
        }

    def render(self) -> str:
        """Human-readable round-to-round diff ending in the attribution statement."""
        lines = [f"diff {self.snapshot_a} -> {self.snapshot_b}"]
        if self.identical:
            lines.append("  no differences")
        for c in self.changes:
            tag = "data" if c.is_data_change else "method"
            lines.append(f"  [{tag}/{c.kind}] {c.subject}: {c.before!r} -> {c.after!r}")
            if c.detail:
                lines.append(f"      {c.detail}")
        lines.append(f"  attribution: {self.attribution}")
        return "\n".join(lines)


def diff(snapshot_a: DatasetSnapshot, snapshot_b: DatasetSnapshot) -> SnapshotDiff:
    """Describe what changed between two rounds, source by source.

    Covers sources added and removed, version strings, per-file checksums,
    record counts, exclusion-reason counts, the ordered cleaning-rule list,
    model versions, the random seed and the selection policy. A reordered
    cleaning pipeline counts as a change even when the rule set is the same,
    because order changes which rule is credited with each drop.
    """
    changes: list[SnapshotChange] = []
    a_by_id = {s.source_id: s for s in snapshot_a.sources}
    b_by_id = {s.source_id: s for s in snapshot_b.sources}

    for sid in sorted(set(b_by_id) - set(a_by_id)):
        changes.append(SnapshotChange("source_added", sid, None,
                                      b_by_id[sid].version,
                                      "a source present in B but not in A"))
    for sid in sorted(set(a_by_id) - set(b_by_id)):
        changes.append(SnapshotChange("source_removed", sid,
                                      a_by_id[sid].version, None,
                                      "a source present in A but not in B"))

    for sid in sorted(set(a_by_id) & set(b_by_id)):
        a, b = a_by_id[sid], b_by_id[sid]
        if a.version != b.version:
            changes.append(SnapshotChange(
                "version", sid, a.version, b.version,
                "the source release changed between rounds"))
        if a.license != b.license:
            changes.append(SnapshotChange(
                "license", sid, a.license, b.license,
                "redistribution terms changed"))

        a_files = {f.path: f for f in a.files}
        b_files = {f.path: f for f in b.files}
        for path in sorted(set(a_files) | set(b_files)):
            fa, fb = a_files.get(path), b_files.get(path)
            if fa is None:
                changes.append(SnapshotChange("file", f"{sid}:{path}", None,
                                              fb.sha256 if fb else None, "file added"))
            elif fb is None:
                changes.append(SnapshotChange("file", f"{sid}:{path}", fa.sha256,
                                              None, "file removed"))
            elif fa.sha256 != fb.sha256:
                changes.append(SnapshotChange(
                    "file", f"{sid}:{path}", fa.sha256, fb.sha256,
                    f"content changed ({fa.size_bytes} -> {fb.size_bytes} bytes)"))

        if (a.n_records_before, a.n_records_after) != (b.n_records_before, b.n_records_after):
            changes.append(SnapshotChange(
                "counts", sid,
                (a.n_records_before, a.n_records_after),
                (b.n_records_before, b.n_records_after),
                "record counts before/after cleaning changed"))

        if a.exclusions != b.exclusions:
            changes.append(SnapshotChange(
                "exclusions", sid, dict(a.exclusions), dict(b.exclusions),
                "different records were dropped, or dropped for different reasons"))

        a_rules = [r.label for r in a.cleaning_rules]
        b_rules = [r.label for r in b.cleaning_rules]
        if a_rules != b_rules:
            changes.append(SnapshotChange(
                "cleaning_rules", sid, a_rules, b_rules,
                "the ordered cleaning pipeline changed; rule order decides which "
                "rule is credited with each exclusion"))

    for name in sorted(set(snapshot_a.model_versions) | set(snapshot_b.model_versions)):
        va = snapshot_a.model_versions.get(name)
        vb = snapshot_b.model_versions.get(name)
        if va != vb:
            changes.append(SnapshotChange("model_version", name, va, vb,
                                          "a model used in the round changed"))

    if snapshot_a.random_seed != snapshot_b.random_seed:
        changes.append(SnapshotChange("seed", "random_seed", snapshot_a.random_seed,
                                      snapshot_b.random_seed,
                                      "sampling differs; some result change is expected"))
    if snapshot_a.selection_policy_id != snapshot_b.selection_policy_id:
        changes.append(SnapshotChange("selection_policy", "selection_policy_id",
                                      snapshot_a.selection_policy_id,
                                      snapshot_b.selection_policy_id,
                                      "candidates were chosen under a different policy"))

    return SnapshotDiff(snapshot_a=snapshot_a.snapshot_id,
                        snapshot_b=snapshot_b.snapshot_id,
                        changes=changes)
