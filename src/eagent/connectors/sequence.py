"""Sequence and family connectors: UniProtKB, UniRef/UniParc, InterPro/Pfam,
NCBI Identical Protein Groups, SDRED, the AKR superfamily resource and MGnify.

Why this module exists
----------------------
The sequence layer is the spine: every candidate resolves to a sequence hash
here before anything else is attached to it. Four things can go wrong, and each
has a guard rather than a comment.

*An annotation read as a result.* A UniProt entry saying the protein is an
alcohol dehydrogenase is a curated assertion, often projected from a homologue.
:class:`UniProtEntry` keeps the evidence code on every statement and
:meth:`UniProtEntry.as_experimental_evidence` refuses, because a functional
annotation is not an experimental result for that sequence and the two are
indistinguishable once the code is dropped.

*An unpublished construct disclosed.* Searching a remote service by sequence
sends the sequence. :meth:`SequenceSearchMixin.sequence_query` runs the
disclosure guard before anything else touches the query, so an unauthorised
construct raises
:class:`~eagent.connectors.base.UnauthorizedSubmissionError` instead of
leaving the process.

*A computational construct counted as data.* A UniRef cluster is a clustering
decision and an MGnify entry is a gene prediction from assembled metagenomic
reads. Both are registered at the ``computational_construct`` ceiling, and both
come back saying so.

*A browse-only database turned into a client.* SDRED and the AKR superfamily
resource have no verified programmatic route. They are
:class:`~eagent.connectors.chemistry.CuratedFileImporter` subclasses whose
``fetch`` exists only to refuse.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..provenance import sequence_hash
from ..schemas.candidate import ConfidenceLevel
from ..schemas.record import EvidenceRef, ExperimentRecord, OutcomeClass
from .base import CachedResponse, ConnectorLayer, looks_like_biological_sequence
from .chemistry import (
    CuratedFileImporter,
    CuratedImportError,
    LayerSemanticsError,
    RegistryBackedConnector,
    UNVERIFIED_CAPABILITY,
    evidence_ref_for,
)

__all__ = [
    "AKRSuperfamilyImporter",
    "AnnotationEvidence",
    "AnnotationNotEvidenceError",
    "ClusterMembership",
    "FamilySignatureHit",
    "IdenticalProteinGroup",
    "InterProConnector",
    "MGnifyProteinsConnector",
    "MetagenomeProteinRecord",
    "NCBIIdenticalProteinGroupsConnector",
    "PfamConnector",
    "SDREDImporter",
    "SequenceSearchMixin",
    "UniParcConnector",
    "UniProtAnnotation",
    "UniProtEntry",
    "UniProtKBConnector",
    "UniRefConnector",
]


# ---------------------------------------------------------------------------
# evidence codes
# ---------------------------------------------------------------------------

class AnnotationEvidence(str, enum.Enum):
    """How a sequence-database statement came to be asserted.

    Kept as an explicit category rather than as the raw code string, because
    every consumer would otherwise write its own substring test and one of them
    would get it wrong. ``UNSTATED`` is the default and is never upgraded by
    inference: a statement with no recorded code is not an experimental one.
    """

    EXPERIMENTAL = "experimental"
    SEQUENCE_SIMILARITY = "sequence_similarity"
    AUTOMATIC_ASSERTION = "automatic_assertion"
    CURATOR_INFERENCE = "curator_inference"
    UNSTATED = "unstated"

    @property
    def is_experimental_in_source(self) -> bool:
        """Whether the *source entry* cites an experiment for this statement.

        Deliberately named for the source. Even here it means "somebody
        published an experiment behind this annotation", not "this is an
        experimental record for this sequence in this project": the experiment
        may have been on another construct, another substrate, in another
        direction.
        """
        return self is AnnotationEvidence.EXPERIMENTAL

    @property
    def is_projected(self) -> bool:
        """Whether the statement was transferred rather than observed."""
        return self in (AnnotationEvidence.SEQUENCE_SIMILARITY,
                        AnnotationEvidence.AUTOMATIC_ASSERTION)


#: Ontology terms this project recognises. Anything unlisted stays
#: ``UNSTATED``: a code nobody mapped is not evidence of anything, and
#: guessing from the numeric part of an identifier is how an automatic
#: assertion becomes an experiment.
_EVIDENCE_CODE_MAP: dict[str, AnnotationEvidence] = {
    "eco:0000269": AnnotationEvidence.EXPERIMENTAL,
    "eco:0000314": AnnotationEvidence.EXPERIMENTAL,
    "eco:0000315": AnnotationEvidence.EXPERIMENTAL,
    "eco:0000250": AnnotationEvidence.SEQUENCE_SIMILARITY,
    "eco:0000255": AnnotationEvidence.SEQUENCE_SIMILARITY,
    "eco:0000256": AnnotationEvidence.AUTOMATIC_ASSERTION,
    "eco:0000213": AnnotationEvidence.AUTOMATIC_ASSERTION,
    "eco:0000305": AnnotationEvidence.CURATOR_INFERENCE,
    "eco:0000303": AnnotationEvidence.CURATOR_INFERENCE,
}


def _evidence_category(row: Mapping[str, Any]) -> AnnotationEvidence:
    """Read the evidence category from a payload, failing closed to ``UNSTATED``."""
    stated = row.get("evidence_category")
    if isinstance(stated, str) and stated.strip():
        try:
            return AnnotationEvidence(stated.strip().lower())
        except ValueError:
            return AnnotationEvidence.UNSTATED
    code = row.get("evidence_code")
    if isinstance(code, str) and code.strip():
        return _EVIDENCE_CODE_MAP.get(code.strip().lower(),
                                      AnnotationEvidence.UNSTATED)
    return AnnotationEvidence.UNSTATED


class AnnotationNotEvidenceError(LayerSemanticsError):
    """A database annotation was asked to be an experimental result.

    The specific failure: an entry annotated with an EC number or a Rhea
    identifier is read as "this protein was shown to catalyse this reaction",
    and a seed set then contains proteins nobody has ever assayed, labelled as
    though they had been.
    """


# ---------------------------------------------------------------------------
# sequence submission
# ---------------------------------------------------------------------------

class SequenceSearchMixin:
    """Adds a sequence-keyed search that cannot run without authorisation.

    Mixed into the connectors whose registered ``sequence_query`` capability is
    not ``not_supported``. The disclosure guard runs before the capability
    verdict is acted on and before the cache is touched, because the question
    "may this sequence be sent to this service" has to be answered before the
    sequence is written anywhere the run keeps -- the cache key and the run
    manifest both retain it.
    """

    def sequence_query(self, sequence: str, *,
                       extra: Mapping[str, Any] | None = None
                       ) -> CachedResponse:
        """Search by sequence, refusing an unauthorised disclosure.

        Raises :class:`~eagent.connectors.base.UnauthorizedSubmissionError`
        unless the policy carries a named
        :class:`~eagent.connectors.base.SubmissionAuthorization` covering this
        exact sequence hash, or records the sequence as already public.
        Disclosure is irreversible and no later policy decision undoes it, so
        this is a hard refusal rather than a warning.

        The guard runs whether or not the network is enabled. A check that only
        fires on the online branch is never exercised in an offline-by-default
        project, so the first time it runs is the first time it matters -- and
        that is the run where an unreleased construct leaves the building.
        """
        text = "".join(str(sequence).split())
        if not text:
            raise LayerSemanticsError(
                f"{self.source_id}: a sequence query needs a sequence")
        notes = list(self.authorise_outbound({"sequence": text}))
        if not looks_like_biological_sequence(text):
            notes.append(
                "the query string is shorter or less varied than the "
                "sequence-shaped test requires, so the disclosure guard did not "
                "classify it as a sequence; it was still sent through the guard")
        query: dict[str, Any] = {"sequence_sha256": sequence_hash(text)}
        query.update({str(k): v for k, v in (extra or {}).items()})
        return self.guarded("search", query, "sequence_query",
                            extra_notes=notes)


# ---------------------------------------------------------------------------
# UniProtKB
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UniProtAnnotation:
    """One statement from an entry, with the code that says how it was made."""

    kind: str
    text: str | None
    evidence: AnnotationEvidence
    evidence_code: str | None = None
    source_identifier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text,
                "evidence": self.evidence.value,
                "evidence_code": self.evidence_code,
                "source_identifier": self.source_identifier}


@dataclass(frozen=True)
class UniProtEntry:
    """A UniProtKB entry: a sequence, plus assertions about it.

    The sequence hash, not the accession, is the join key. Accessions get
    re-annotated and isoforms share names, so two records referring to "the
    same protein" by accession can be two different sequences, and a geometric
    criterion computed on one would be reported against the other.
    """

    accession: str
    entry_name: str | None
    sequence: str | None
    sequence_sha256: str | None
    organism: str | None
    ec_numbers: tuple[str, ...]
    rhea_ids: tuple[str, ...]
    annotations: tuple[UniProtAnnotation, ...]
    evidence: EvidenceRef
    response: CachedResponse
    reviewed: bool | None = None
    notes: tuple[str, ...] = ()
    #: Whether UniProt reports this accession as deleted or merged away. An
    #: inactive entry is a real answer -- the identifier existed and no longer
    #: names a protein -- and ``sequence`` is ``None`` because there is none,
    #: not because the fetch was partial.
    inactive: bool = False
    inactive_reason: str | None = None

    def annotations_of(self, kind: str) -> tuple[UniProtAnnotation, ...]:
        return tuple(a for a in self.annotations if a.kind == kind)

    def experimentally_evidenced(self) -> tuple[UniProtAnnotation, ...]:
        """Statements the entry itself backs with an experimental code.

        Returned so a curator can go and read the cited experiment. Not
        returned as evidence: see :meth:`as_experimental_evidence`.
        """
        return tuple(a for a in self.annotations
                     if a.evidence.is_experimental_in_source)

    def as_experimental_evidence(self) -> None:
        """Always refuses. See :class:`AnnotationNotEvidenceError`.

        Even an annotation carrying an experimental evidence code is a pointer
        to somebody else's experiment on some construct, under some conditions,
        in some direction -- none of which this entry records. Promoting it
        requires reading the cited paper, which is
        :func:`~eagent.datalayer.intake.promote` with a named human reviewer.
        """
        experimental = len(self.experimentally_evidenced())
        raise AnnotationNotEvidenceError(
            f"UniProt {self.accession} carries {len(self.annotations)} "
            f"annotation(s), {experimental} of them with an experimental "
            f"evidence code in the source entry. An annotation is an assertion "
            f"about function, not a measurement on this sequence with the "
            f"target substrate in the target direction. Read the cited "
            f"publication and ingest that instead.")

    def to_sequence_record(self) -> ExperimentRecord:
        """A record of the sequence itself, carrying no outcome.

        The outcome is ``not_tested`` and stays there. The entry says what the
        protein is thought to do; it reports no assay, so any other outcome
        would be a measurement nobody made.
        """
        return ExperimentRecord(
            record_id=f"uniprotkb:{self.accession}",
            sequence=self.sequence,
            sequence_sha256=self.sequence_sha256,
            accession=self.accession,
            database_version=self.response.database_version,
            outcome=OutcomeClass.NOT_TESTED,
            evidence=[self.evidence],
            notes="; ".join(self.notes),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"accession": self.accession, "entry_name": self.entry_name,
                "sequence_sha256": self.sequence_sha256,
                "organism": self.organism, "reviewed": self.reviewed,
                "ec_numbers": list(self.ec_numbers),
                "rhea_ids": list(self.rhea_ids),
                "annotations": [a.to_dict() for a in self.annotations],
                "notes": list(self.notes)}


#: Fields asked for in the live fetch, pinned here because the field list *is*
#: the parser: a response assembled from a different list has different keys,
#: and a translator reading a key that is not there reports an absence rather
#: than a missing request.
UNIPROT_ENTRY_FIELDS: tuple[str, ...] = (
    "accession", "id", "protein_name", "organism_name", "reviewed", "ec",
    "cc_catalytic_activity", "cc_cofactor", "sequence", "xref_pdb",
)


def _reviewed_flag(entry_type: str) -> bool | None:
    """``True`` for Swiss-Prot, ``False`` for TrEMBL, ``None`` if unstated.

    Whole-word, because ``"reviewed" in "unreviewed"`` is true and that one
    character decides whether an entry is presented as curated.
    """
    tokens = {t.strip("()").lower() for t in (entry_type or "").split()}
    if "unreviewed" in tokens:
        return False
    if "reviewed" in tokens:
        return True
    return None


def uniprot_entry_payload(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Translate one live UniProtKB JSON entry into this connector's payload.

    WHY A TRANSLATOR AND NOT A PARSER IN THE CONNECTOR
    --------------------------------------------------
    The connector was written against a normalised shape so a curator's export
    and a live response reach the same code. This is the half that had never
    existed: nothing turned a real UniProt response into that shape, so the
    live route was unreachable however reachable the host was.

    WHAT IS CARRIED AND WHAT IS DROPPED
    -----------------------------------
    Every catalytic-activity statement is carried **with its own evidence
    code**, because that is the one thing about a UniProt annotation that
    decides what it may be used for: ``ECO:0000269`` cites a published
    experiment, ``ECO:0000250`` says the statement was transferred from a
    homologue, and the two read identically in the entry text. A statement
    whose evidences disagree is carried once per code rather than collapsed --
    the strongest code would otherwise speak for the weakest.

    The EC numbers carried are the ones attached to a catalytic-activity
    statement, which keeps each tied to the reaction and the evidence it came
    with. An EC number floating on the entry is an annotation about the
    protein's class, not about a reaction anybody measured.

    The sequence is carried verbatim; its hash is computed by the connector
    from this string, never from the accession that was asked for.
    """
    accession = _text(entry.get("primaryAccession"))
    sequence = _text((entry.get("sequence") or {}).get("value"))
    entry_type = _text(entry.get("entryType")) or ""
    # UniProt answers a deleted or merged accession with HTTP 200 and an entry
    # of type "Inactive" that has an accession and no sequence. It is an
    # answer -- "this identifier no longer names a protein, and here is why"
    # -- and it must not travel on as an entry whose sequence happens to be
    # empty.
    inactive = entry_type.strip().lower() == "inactive"
    inactive_reason = None
    if inactive:
        reason = entry.get("inactiveReason") or {}
        inactive_reason = " ".join(filter(None, (
            _text(reason.get("inactiveReasonType")),
            _text(reason.get("deletedReason")),
            ("merged into " + ", ".join(str(m) for m in reason["mergeDemergeTo"]))
            if reason.get("mergeDemergeTo") else None))) or "no reason given"
    annotations: list[dict[str, Any]] = []
    ec_numbers: list[str] = []
    rhea_ids: list[str] = []

    for comment in entry.get("comments") or ():
        if not isinstance(comment, Mapping):
            continue
        kind = (_text(comment.get("commentType")) or "").strip().lower()
        kind = kind.replace(" ", "_") or "unspecified"
        reaction = comment.get("reaction") or {}
        text = _text(reaction.get("name")) or _text(
            " ".join(_text(t.get("value")) or ""
                     for t in (comment.get("texts") or ())
                     if isinstance(t, Mapping)))
        ec = _text(reaction.get("ecNumber"))
        if ec and ec not in ec_numbers:
            ec_numbers.append(ec)
        for xref in reaction.get("reactionCrossReferences") or ():
            if isinstance(xref, Mapping) and xref.get("database") == "Rhea":
                rid = _text(xref.get("id"))
                if rid and rid.startswith("RHEA:") and rid not in rhea_ids:
                    rhea_ids.append(rid)
        evidences = [e for e in (reaction.get("evidences")
                                 or comment.get("evidences") or ())
                     if isinstance(e, Mapping)]
        if not evidences:
            # No code is not a weak code: it is no statement about evidence,
            # and _evidence_category fails closed to UNSTATED.
            annotations.append({"kind": kind, "text": text,
                                "evidence_code": None,
                                "source_identifier": None})
            continue
        for item in evidences:
            source = _text(item.get("source"))
            ident = _text(item.get("id"))
            annotations.append({
                "kind": kind,
                "text": text,
                "evidence_code": _text(item.get("evidenceCode")),
                "source_identifier": (f"{source}:{ident}"
                                      if source and ident else ident),
            })

    names = entry.get("proteinDescription") or {}
    recommended = ((names.get("recommendedName") or {}).get("fullName")
                   or {}).get("value")
    submitted = next((((n.get("fullName") or {}).get("value"))
                      for n in (names.get("submissionNames") or ())
                      if isinstance(n, Mapping)), None)

    return {
        "accession": accession,
        "entry_name": _text(entry.get("uniProtkbId")),
        "protein_name": _text(recommended) or _text(submitted),
        "sequence": sequence,
        "organism": _text((entry.get("organism") or {}).get("scientificName")),
        # "reviewed" is read from entryType rather than assumed: an unreviewed
        # entry carries the same keys and a default of True would promote
        # every TrEMBL record to Swiss-Prot in this project's own records.
        #
        # Matched as a whole word. The two values are "UniProtKB reviewed
        # (Swiss-Prot)" and "UniProtKB unreviewed (TrEMBL)", and a substring
        # test for "reviewed" is true of both.
        "reviewed": _reviewed_flag(entry_type),
        "ec_numbers": ec_numbers,
        "rhea_ids": rhea_ids,
        "annotations": annotations,
        "inactive": inactive,
        "inactive_reason": inactive_reason,
        "pdb_ids": [
            _text(x.get("id")) for x in (entry.get("uniProtKBCrossReferences") or ())
            if isinstance(x, Mapping) and x.get("database") == "PDB"
            and _text(x.get("id"))
        ],
    }


class UniProtKBConnector(SequenceSearchMixin, RegistryBackedConnector):
    """UniProtKB entries, with every statement's evidence code preserved.

    Reads the payload shape::

        {"records": [{"accession": "...", "entry_name": "...",
                      "sequence": "MK...", "organism": "...",
                      "reviewed": true, "ec_numbers": ["1.1.1.1"],
                      "rhea_ids": ["RHEA:..."],
                      "annotations": [{"kind": "catalytic_activity",
                                       "text": "...",
                                       "evidence_code": "ECO:0000269"}]}]}

    The sequence hash is computed from the payload's own sequence string, never
    carried over from the query: a cached entry whose sequence differs from the
    one that was asked about is a re-annotation, and silently keeping the old
    hash would attach this run's evidence to the wrong protein.
    """

    source_id = "uniprotkb"
    data_layer = ConnectorLayer.SEQUENCE
    description = "UniProtKB: reviewed and unreviewed protein entries"
    #: The one request this connector's client was written against, and the
    #: one the shipped probe exercises:
    #: ``/uniprotkb/<accession>.json?fields=<pinned list>``. A verified base
    #: URL does not license any other shape; see
    #: :class:`~eagent.connectors.chemistry.RequestShapeNotVerifiedError`.
    verified_route_capability = "exact_record_fetch"

    def _fetch_remote(self, key: str) -> tuple[Any, str | None]:
        """Fetch one entry by accession, in the shape the probe checked.

        The field list is pinned in :data:`UNIPROT_ENTRY_FIELDS` rather than
        taken from the caller, because the list *is* the parser: a response
        assembled from a different list has different keys, and a translator
        reading a key that was never requested reports an absence rather than
        a missing request.

        The release is read from the response header the API documents, not
        invented. ``None`` when the header is absent: a run pinned to a
        version string nobody sent is not pinned.
        """
        import urllib.parse

        base = self.require_endpoint().rstrip("/")
        url = (f"{base}/uniprotkb/{urllib.parse.quote(str(key), safe='')}.json"
               f"?fields={','.join(UNIPROT_ENTRY_FIELDS)}")
        payload, version = self._http_json(url)
        if not isinstance(payload, Mapping):
            return payload, version
        if not payload.get("primaryAccession"):
            # A JSON body that is not an entry: an error object, or a search
            # response somebody pointed at this method. Returning it would let
            # the translator produce an entry with every field empty.
            return None, version
        return {"records": [uniprot_entry_payload(payload)]}, version

    def entry(self, accession: str) -> UniProtEntry | None:
        """Fetch one entry, or ``None`` when the cache does not hold it."""
        response = self.fetch(accession)
        if not response.ok:
            return None
        rows = self.records(response)
        if len(rows) != 1:
            return None
        return self._build(rows[0], accession, response)

    def _build(self, row: Mapping[str, Any], fallback: str,
               response: CachedResponse) -> UniProtEntry:
        accession = _text(row.get("accession")) or fallback
        sequence = _text(row.get("sequence"))
        annotations = tuple(
            UniProtAnnotation(
                kind=_text(item.get("kind")) or "unspecified",
                text=_text(item.get("text")),
                evidence=_evidence_category(item),
                evidence_code=_text(item.get("evidence_code")),
                source_identifier=_text(item.get("source_identifier")))
            for item in (row.get("annotations") or [])
            if isinstance(item, Mapping))
        notes = [
            "an entry annotation is an assertion about function; it is not a "
            "measurement on this sequence and must not be ingested as one",
        ]
        if row.get("inactive"):
            notes.insert(0,
                f"this accession is INACTIVE ({row.get('inactive_reason')}); it "
                f"no longer names a current protein, carries no sequence, and "
                f"must not be used as a sequence record or as a seed")
        projected = [a for a in annotations if a.evidence.is_projected]
        if projected:
            notes.append(
                f"{len(projected)} annotation(s) are projected by similarity or "
                f"asserted automatically: they describe a homologue's behaviour "
                f"transferred to this entry")
        unstated = [a for a in annotations
                    if a.evidence is AnnotationEvidence.UNSTATED]
        if unstated:
            notes.append(
                f"{len(unstated)} annotation(s) carry no recognised evidence "
                f"code; they are treated as unstated, never as experimental")
        if not sequence:
            notes.append(
                "the cached record carries no sequence, so this entry cannot "
                "serve as the join key for a candidate")
        return UniProtEntry(
            accession=accession,
            entry_name=_text(row.get("entry_name")),
            sequence=sequence,
            sequence_sha256=sequence_hash(sequence) if sequence else None,
            organism=_text(row.get("organism")),
            ec_numbers=_strings(row.get("ec_numbers")),
            rhea_ids=_strings(row.get("rhea_ids")),
            annotations=annotations,
            evidence=self.evidence_ref(accession, response),
            response=response,
            reviewed=row.get("reviewed") if isinstance(row.get("reviewed"), bool)
            else None,
            notes=tuple(notes),
            inactive=bool(row.get("inactive")),
            inactive_reason=_text(row.get("inactive_reason")))


# ---------------------------------------------------------------------------
# UniRef / UniParc
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClusterMembership:
    """Membership of a sequence in a clustering, with the clustering's authority.

    A UniRef cluster is the output of a clustering run at an identity
    threshold. It groups sequences; it does not assert that they share a
    substrate, a stereopreference or a cofactor. The registry caps this source
    at ``computational_construct`` and this object repeats why.
    """

    cluster_id: str
    identity_threshold: str | None
    representative_accession: str | None
    member_accessions: tuple[str, ...]
    response: CachedResponse
    notes: tuple[str, ...] = ()

    def as_function_evidence(self) -> None:
        """Always refuses: a cluster is a computational construct."""
        raise LayerSemanticsError(
            f"UniRef cluster {self.cluster_id} groups "
            f"{len(self.member_accessions)} sequence(s) by identity. Clustering "
            f"is a computational construct: members of one cluster routinely "
            f"differ in substrate scope and stereopreference, which are the "
            f"properties this project ranks on.")

    def to_dict(self) -> dict[str, Any]:
        return {"cluster_id": self.cluster_id,
                "identity_threshold": self.identity_threshold,
                "representative_accession": self.representative_accession,
                "member_accessions": list(self.member_accessions),
                "notes": list(self.notes)}


class UniRefConnector(SequenceSearchMixin, RegistryBackedConnector):
    """UniRef clusters. Registered at the ``computational_construct`` ceiling.

    ``sequence_query`` is registered ``unknown`` here, so a sequence search
    comes back carrying :data:`~eagent.connectors.chemistry.UNVERIFIED_CAPABILITY`
    rather than looking like a route somebody has used.

    Reads the payload shape::

        {"records": [{"cluster_id": "UniRef90_...", "identity": "90",
                      "representative": "...", "members": ["...", "..."]}]}
    """

    source_id = "uniref"
    data_layer = ConnectorLayer.SEQUENCE
    description = "UniRef: identity-threshold sequence clusters"

    def cluster(self, cluster_id: str) -> ClusterMembership | None:
        response = self.fetch(cluster_id)
        if not response.ok:
            return None
        rows = self.records(response)
        if len(rows) != 1:
            return None
        row = rows[0]
        return ClusterMembership(
            cluster_id=_text(row.get("cluster_id")) or cluster_id,
            identity_threshold=_text(row.get("identity")),
            representative_accession=_text(row.get("representative")),
            member_accessions=_strings(row.get("members")),
            response=response,
            notes=(
                "a clustering result, not a measurement: the registry caps this "
                "source at computational_construct",
                "cluster co-membership is a leakage risk for a train/test split "
                "and is used for grouping, never as corroboration",
            ))


class UniParcConnector(SequenceSearchMixin, RegistryBackedConnector):
    """UniParc: the sequence archive, used to resolve a sequence to its history.

    Useful precisely because it is annotation-free: it answers "has this exact
    sequence been seen before, and under which accessions", which is the
    question an accession cannot answer after a re-annotation.

    Reads the payload shape::

        {"records": [{"uniparc_id": "UPI...", "sequence": "MK...",
                      "cross_references": [{"database": "...", "id": "..."}]}]}
    """

    source_id = "uniparc"
    data_layer = ConnectorLayer.SEQUENCE
    description = "UniParc: the non-redundant sequence archive"

    def archive_entry(self, uniparc_id: str) -> Mapping[str, Any] | None:
        """One archive record as stored, or ``None``.

        Returns the payload row rather than a typed object because UniParc's
        value here is the cross-reference list, whose shape differs per
        database; typing it would mean inventing a schema for records this
        project has not yet needed to interpret.
        """
        response = self.fetch(uniparc_id)
        if not response.ok:
            return None
        rows = self.records(response)
        return rows[0] if len(rows) == 1 else None


# ---------------------------------------------------------------------------
# InterPro / Pfam
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FamilySignatureHit:
    """A signature match on a sequence, with the integration caveat attached.

    InterPro integrates Pfam. A hit in each is one family signal, not two, and
    the ``counts_independently_of`` field says so where a scoring function can
    read it instead of where only a reader can.
    """

    source_id: str
    entry_id: str
    entry_name: str | None
    entry_type: str | None
    accession: str | None
    start: int | None
    end: int | None
    response: CachedResponse
    counts_independently_of: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def as_substrate_scope(self) -> None:
        """Always refuses: family membership is not substrate scope.

        Members of one Pfam family routinely differ in every property this
        project ranks on. A family label that leaked into a scope claim would
        rank candidates on their fold.
        """
        raise LayerSemanticsError(
            f"{self.source_id} {self.entry_id} is a family or domain signature "
            f"match. Family membership is not substrate scope and not "
            f"stereopreference, and it names no catalytic residue.")

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "entry_id": self.entry_id,
                "entry_name": self.entry_name, "entry_type": self.entry_type,
                "accession": self.accession, "start": self.start,
                "end": self.end,
                "counts_independently_of": list(self.counts_independently_of),
                "notes": list(self.notes)}


class _SignatureConnector(SequenceSearchMixin, RegistryBackedConnector):
    """Shared reader for the two signature resources.

    Reads the payload shape::

        {"records": [{"entry_id": "IPR000000", "name": "...", "type": "family",
                      "accession": "P00000", "start": 1, "end": 250}]}
    """

    data_layer = ConnectorLayer.FAMILY
    #: Sibling resources whose hits are the same signal as this one's.
    not_independent_of: tuple[str, ...] = ()

    def signatures_for(self, accession: str) -> tuple[FamilySignatureHit, ...]:
        """Cached signature matches for one accession."""
        response = self.guarded("search", {"accession": accession},
                                "keyword_query")
        notes = (
            "a signature match is a model hit, not an experimental assignment",
            "domain architecture catches fusions and truncations; it does not "
            "predict activity",
        )
        return tuple(
            FamilySignatureHit(
                source_id=self.source_id,
                entry_id=_text(row.get("entry_id")) or "unstated",
                entry_name=_text(row.get("name")),
                entry_type=_text(row.get("type")),
                accession=_text(row.get("accession")) or accession,
                start=_int(row.get("start")), end=_int(row.get("end")),
                response=response,
                counts_independently_of=self.not_independent_of,
                notes=notes)
            for row in self.records(response))


class InterProConnector(_SignatureConnector):
    """InterPro entries. Integrates Pfam, so the two do not corroborate."""

    source_id = "interpro"
    not_independent_of = ("pfam",)
    description = "InterPro: integrated protein signature matches"


class PfamConnector(_SignatureConnector):
    """Pfam entries. Integrated by InterPro, so the two do not corroborate."""

    source_id = "pfam"
    not_independent_of = ("interpro",)
    description = "Pfam: protein family hidden Markov model matches"


# ---------------------------------------------------------------------------
# NCBI Identical Protein Groups
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IdenticalProteinGroup:
    """Accessions sharing one identical sequence.

    This is the one cross-database join this project trusts without a
    per-record check, because the grouping criterion *is* sequence identity.
    The annotations on the members are not part of that guarantee: the same
    sequence is annotated differently in different databases, and merging the
    annotations because the sequences match is how a cautious record acquires
    somebody else's confident one.
    """

    group_id: str
    sequence: str | None
    sequence_sha256: str | None
    member_accessions: tuple[str, ...]
    source_databases: tuple[str, ...]
    response: CachedResponse
    notes: tuple[str, ...] = ()

    def merge_annotations(self) -> None:
        """Always refuses: identical sequence does not mean identical annotation."""
        raise LayerSemanticsError(
            f"identical protein group {self.group_id} holds "
            f"{len(self.member_accessions)} accession(s) with the same sequence. "
            f"That justifies joining them on the sequence hash; it does not "
            f"justify merging their annotations, which were made independently "
            f"and may disagree.")

    def to_dict(self) -> dict[str, Any]:
        return {"group_id": self.group_id,
                "sequence_sha256": self.sequence_sha256,
                "member_accessions": list(self.member_accessions),
                "source_databases": list(self.source_databases),
                "notes": list(self.notes)}


class NCBIIdenticalProteinGroupsConnector(SequenceSearchMixin,
                                          RegistryBackedConnector):
    """NCBI Protein, used for identical-protein-group resolution.

    Reads the payload shape::

        {"records": [{"group_id": "...", "sequence": "MK...",
                      "members": ["WP_...", "NP_..."],
                      "databases": ["RefSeq", "GenBank"]}]}
    """

    source_id = "ncbi_protein"
    data_layer = ConnectorLayer.SEQUENCE
    description = "NCBI Protein: RefSeq, GenBank and identical protein groups"

    def identical_protein_group(self, group_id: str) -> IdenticalProteinGroup | None:
        response = self.fetch(group_id)
        if not response.ok:
            return None
        rows = self.records(response)
        if len(rows) != 1:
            return None
        row = rows[0]
        sequence = _text(row.get("sequence"))
        return IdenticalProteinGroup(
            group_id=_text(row.get("group_id")) or group_id,
            sequence=sequence,
            sequence_sha256=sequence_hash(sequence) if sequence else None,
            member_accessions=_strings(row.get("members")),
            source_databases=_strings(row.get("databases")),
            response=response,
            notes=(
                "the grouping criterion is sequence identity, so the group is a "
                "safe join key; the members' annotations are not pooled by it",
            ))


# ---------------------------------------------------------------------------
# MGnify
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetagenomeProteinRecord:
    """A predicted protein from assembled metagenomic data.

    Registered at the ``computational_construct`` ceiling: the sequence is a
    gene call on an assembly, so the start codon, the termini and sometimes the
    frame are predictions. Ordering one as a gene without a recorded gate is
    how a frameshifted call reaches a synthesis plate.
    """

    mgnify_id: str
    sequence: str | None
    sequence_sha256: str | None
    catalogue: str | None
    completeness_flag: str | None
    quality_flag: str | None
    response: CachedResponse
    confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT
    notes: tuple[str, ...] = ()

    def require_synthesis_gate(self, gate_name: str | None,
                               passed: bool | None) -> str:
        """The recorded gate this candidate passed, or a refusal.

        The registry's curation note for this source requires that the project
        "define and record the gate that a metagenomic candidate must pass
        before it may enter a synthesis batch". This method is where that
        requirement bites: no gate recorded, no synthesis.
        """
        if not gate_name or not str(gate_name).strip():
            raise LayerSemanticsError(
                f"{self.mgnify_id} is a metagenomic gene prediction and no "
                f"synthesis gate has been named for it. The registry requires a "
                f"recorded gate before such a candidate enters a batch, because "
                f"the termini and the reading frame are predicted, not observed.")
        if passed is not True:
            raise LayerSemanticsError(
                f"{self.mgnify_id} has not passed the recorded gate "
                f"'{gate_name}' (passed={passed!r}); an ungated metagenomic "
                f"prediction may not enter a synthesis batch")
        return str(gate_name).strip()

    def to_dict(self) -> dict[str, Any]:
        return {"mgnify_id": self.mgnify_id,
                "sequence_sha256": self.sequence_sha256,
                "catalogue": self.catalogue,
                "completeness_flag": self.completeness_flag,
                "quality_flag": self.quality_flag,
                "confidence": self.confidence.value,
                "notes": list(self.notes)}


class MGnifyProteinsConnector(SequenceSearchMixin, RegistryBackedConnector):
    """MGnify protein catalogues: sequence space nobody has observed expressed.

    Reads the payload shape::

        {"records": [{"mgnify_id": "MGYP...", "sequence": "MK...",
                      "catalogue": "...", "completeness": "...",
                      "quality": "..."}]}
    """

    source_id = "mgnify_proteins"
    data_layer = ConnectorLayer.SEQUENCE
    description = "MGnify Proteins: metagenome-derived protein catalogues"

    def protein(self, mgnify_id: str) -> MetagenomeProteinRecord | None:
        response = self.fetch(mgnify_id)
        if not response.ok:
            return None
        rows = self.records(response)
        if len(rows) != 1:
            return None
        row = rows[0]
        sequence = _text(row.get("sequence"))
        notes = [
            "a gene prediction on an assembly: the termini and the reading "
            "frame are computational constructs, not observations",
            "the registry caps this source at computational_construct, so "
            "nothing from it is evidence",
        ]
        if not _text(row.get("completeness")) or not _text(row.get("quality")):
            notes.append(
                "the record states no completeness or quality flag; the "
                "registry's staging rule depends on keeping those, so this "
                "candidate cannot be staged until a curator supplies them")
        return MetagenomeProteinRecord(
            mgnify_id=_text(row.get("mgnify_id")) or mgnify_id,
            sequence=sequence,
            sequence_sha256=sequence_hash(sequence) if sequence else None,
            catalogue=_text(row.get("catalogue")),
            completeness_flag=_text(row.get("completeness")),
            quality_flag=_text(row.get("quality")),
            response=response,
            notes=tuple(notes))


# ---------------------------------------------------------------------------
# importers: the two family resources with no verified route
# ---------------------------------------------------------------------------

class _FamilyFastaImporter(CuratedFileImporter):
    """Shared importer for the two curated family resources.

    Both are registered with ``manual_review_import`` and ``offline_import``
    and both carry the same curation note: no stable public REST API has been
    verified, and no URL may be recorded until one has. Both therefore enter as
    a curator-supplied FASTA or table, and both are flagged
    ``requires_human_review_per_record``.
    """

    required_fields = ("id", "sequence")

    def _build_record(self, row: Mapping[str, Any], row_number: int
                      ) -> ExperimentRecord:
        sequence = str(row.get("sequence", "")).strip()
        if not looks_like_biological_sequence(sequence) and len(sequence) < 25:
            raise CuratedImportError(
                f"row {row_number}: '{sequence[:20]}...' is too short to be a "
                f"protein sequence; a truncated FASTA record would join to the "
                f"wrong hash")
        identifier = str(row.get("id")).strip()
        return ExperimentRecord(
            record_id=f"{self.source_id}:{identifier}",
            sequence=sequence,
            accession=_text(row.get("accession")) or identifier,
            outcome=OutcomeClass.NOT_TESTED,
            evidence=[evidence_ref_for(self.source, identifier,
                                       registry=self.registry,
                                       locator=_text(row.get("description")))],
            notes=_text(row.get("description")) or "",
        )

    def _row_uncertainties(self, row: Mapping[str, Any]) -> tuple[str, ...]:
        return (
            "classification from a curated family resource: it is a family "
            "assignment, not a measurement, and the registry caps it at "
            "annotation_only",
            "the resource's lineage is registered as incomplete, so its "
            "sequences are drawn from databases registered separately here and "
            "may not be counted as independent of them",
        )


class SDREDImporter(_FamilyFastaImporter):
    """Short-chain dehydrogenase/reductase engineering database, imported by hand.

    Registered because of its standardised position numbering, which is the one
    thing that makes mutations comparable across SDR members. The registry's
    curation note says that numbering table's publication route is unconfirmed
    and that its licence must be checked before it is copied, so this importer
    reads what a curator supplies and records who supplied it.
    """

    source_id = "sdred"
    description = "SDRED: SDR engineering database, imported from a curated export"


class AKRSuperfamilyImporter(_FamilyFastaImporter):
    """Aldo-keto reductase superfamily resource, imported by hand.

    Registered for the superfamily-name-to-accession mapping. Same access
    caveat as SDRED: no stable public REST API has been verified, and this
    class records no URL.
    """

    source_id = "akr_superfamily"
    description = "AKR superfamily: nomenclature and alignments, imported"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Sequence):
        return tuple(str(v).strip() for v in value if str(v).strip())
    return ()


def _int(value: Any) -> int | None:
    """An int, or ``None``. Never a positional default: a residue number that
    defaulted to 0 or 1 would point at the wrong residue silently."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


#: Re-exported so a caller can test a response for the unverified-capability
#: marker without importing from two modules.
UNVERIFIED_CAPABILITY_MARKER = UNVERIFIED_CAPABILITY
