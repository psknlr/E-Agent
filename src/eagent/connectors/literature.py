"""Literature connectors: PubMed, Europe PMC, EnzChemRED and Zenodo.

Why this module exists
----------------------
The literature layer is where claims are *pointed at*, never where they are
measured, and where two legal and two scientific mistakes are easy to make.

*A citation ingested as data.* A title and an abstract are not a measurement.
:class:`CitationRecord` has no outcome, no value and no conditions, and
:meth:`CitationRecord.as_measurement` refuses, because a seed set assembled
from abstracts contains rows nobody has read with numbers nobody reported.

*Full text redistributed outside its licence.* Only an open licence permits
carrying the text into this project's own artifacts.
:class:`FullTextResult` returns the text only for a licence on an explicit
allow-list and marks everything else not redistributable -- including the case
where the record states no licence at all, because an unstated licence is not
an open one.

*An expert corpus merged with text mining.* EnzChemRED ships an
expert-annotated portion and further text-mined data. The registry's curation
note says to "keep them in separate stores from the first import -- separating
them later is not reliably possible". :class:`EnzChemREDAsset` makes them
different cache keys, different tiers and different objects, and
:meth:`EnzChemREDCorpus.merged_with` refuses.

*A dataset cited by the wrong DOI.* A Zenodo concept DOI resolves to whatever
the latest version happens to be. Pinning a run to it means the run's inputs
change under it. :meth:`ZenodoDeposit.require_version_doi` refuses a concept
DOI where a pinned one is needed.
"""

from __future__ import annotations

import enum
import hashlib
import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..datalayer.intake import (
    EvidenceTier,
    ExtractionMethod,
    IntakeRecord,
    ingest,
)
from ..datalayer.registry import CapabilityState, SourceRegistry
from ..provenance import utc_now
from ..schemas.chem import SubstrateSpec
from ..schemas.record import EvidenceRef, ExperimentRecord, OutcomeClass
from .base import CachedResponse, ConnectorLayer
from .chemistry import (
    LayerSemanticsError,
    RegistryBackedConnector,
    evidence_ref_for,
    upstream_sources_for,
)

__all__ = [
    "CitationNotEvidenceError",
    "CitationRecord",
    "CorpusMergeRefusedError",
    "EnzChemREDAnnotation",
    "EnzChemREDAsset",
    "EnzChemREDConnector",
    "EnzChemREDCorpus",
    "EuropePMCConnector",
    "FullTextResult",
    "NotRedistributableError",
    "OPEN_REDISTRIBUTABLE_LICENCES",
    "PubMedConnector",
    "ZenodoConnector",
    "ZenodoDeposit",
    "DownloadedFile",
    "ChecksumMismatchError",
    "DEFAULT_MAX_DOWNLOAD_BYTES",
    "zenodo_record_payload",
    "is_open_redistributable",
]


# ---------------------------------------------------------------------------
# PubMed
# ---------------------------------------------------------------------------

class CitationNotEvidenceError(LayerSemanticsError):
    """A bibliographic record was asked to be a measurement.

    The failure: an abstract saying an enzyme "showed high activity towards
    aromatic ketones" becomes a positive row for the target substrate, with a
    conversion nobody reported, in a direction nobody stated.
    """


@dataclass(frozen=True)
class CitationRecord:
    """A pointer to a publication. Carries no result by construction.

    There is deliberately no ``outcome``, ``value`` or ``conditions`` field. A
    record that could hold a number read from an abstract would eventually hold
    one.
    """

    source_id: str
    identifier: str
    doi: str | None
    title: str | None
    abstract: str | None
    journal: str | None
    year: int | None
    evidence: EvidenceRef
    response: CachedResponse
    notes: tuple[str, ...] = ()

    def as_measurement(self) -> None:
        """Always refuses. See :class:`CitationNotEvidenceError`."""
        raise CitationNotEvidenceError(
            f"{self.source_id}:{self.identifier} is a bibliographic record. A "
            f"title and an abstract are not data: nothing may be ingested as a "
            f"measurement without reading the paper or its supplement, which "
            f"is a human act recorded by intake.promote().")

    def to_experiment_record(self) -> ExperimentRecord:
        """A record that points at the paper and claims nothing about it."""
        return ExperimentRecord(
            record_id=f"{self.source_id}:{self.identifier}",
            outcome=OutcomeClass.NOT_TESTED,
            evidence=[self.evidence],
            notes=f"citation only: {self.title or 'untitled'}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "identifier": self.identifier,
                "doi": self.doi, "title": self.title, "journal": self.journal,
                "year": self.year, "notes": list(self.notes)}


class _CitationConnector(RegistryBackedConnector):
    """Shared reader for the two bibliographic resources.

    Reads the payload shape::

        {"records": [{"identifier": "...", "doi": "...", "title": "...",
                      "abstract": "...", "journal": "...", "year": 2020}]}
    """

    data_layer = ConnectorLayer.LITERATURE

    def citations(self, query: Mapping[str, Any]) -> tuple[CitationRecord, ...]:
        """Cached citations for a structured query."""
        response = self.guarded("search", query, "keyword_query")
        return self._citations_from(response)

    def citation(self, identifier: str) -> CitationRecord | None:
        """One citation by identifier, or ``None``."""
        response = self.fetch(identifier)
        if not response.ok:
            return None
        found = self._citations_from(response)
        return found[0] if len(found) == 1 else None

    def _citations_from(self, response: CachedResponse
                        ) -> tuple[CitationRecord, ...]:
        notes = [
            "a pointer to evidence, not the evidence: nothing may be ingested "
            "as a measurement from a title and an abstract",
            "indexing bias means the absence of a hit is not the absence of work",
        ]
        if self.source.capabilities.version_information \
                is CapabilityState.NOT_SUPPORTED:
            notes.append(
                "this source publishes no release identifier, so a literature "
                "search cannot be pinned in the run manifest and a rerun may "
                "return a different set")
        out: list[CitationRecord] = []
        for index, row in enumerate(self.records(response), start=1):
            identifier = (_text(row.get("identifier")) or _text(row.get("pmid"))
                          or _text(row.get("doi")) or f"row{index}")
            out.append(CitationRecord(
                source_id=self.source_id, identifier=identifier,
                doi=_text(row.get("doi")), title=_text(row.get("title")),
                abstract=_text(row.get("abstract")),
                journal=_text(row.get("journal")), year=_int(row.get("year")),
                evidence=evidence_ref_for(
                    self.source, identifier, registry=self.registry,
                    database_version=response.database_version,
                    retrieved_at=response.retrieved_at),
                response=response, notes=tuple(notes)))
        return tuple(out)


class PubMedConnector(_CitationConnector):
    """PubMed records: the entry point to evidence, never the evidence."""

    source_id = "pubmed"
    description = "PubMed: bibliographic records"


# ---------------------------------------------------------------------------
# Europe PMC
# ---------------------------------------------------------------------------

#: Licence identifiers under which this project may carry an article's full
#: text into its own artifacts. The list is explicit and short on purpose:
#: anything unlisted -- including a blank, a publisher-specific phrase, or an
#: open-access flag with no named licence -- is treated as not redistributable.
#: Failing closed costs a curation step; failing open redistributes somebody
#: else's copyrighted text.
OPEN_REDISTRIBUTABLE_LICENCES: frozenset[str] = frozenset({
    "cc0", "cc0-1.0",
    "cc-by", "cc-by-3.0", "cc-by-4.0",
    "cc-by-sa", "cc-by-sa-3.0", "cc-by-sa-4.0",
})


def is_open_redistributable(licence: str | None) -> bool:
    """Whether a licence string is on the explicit allow-list.

    Normalises spacing and case only. It does not try to parse a licence
    sentence: a near-match such as "CC BY-NC 4.0" differs from an allowed one
    by two characters and by the whole question of commercial use.
    """
    if not licence:
        return False
    token = "-".join(str(licence).strip().lower().replace("_", "-").split())
    return token in OPEN_REDISTRIBUTABLE_LICENCES


class NotRedistributableError(LayerSemanticsError):
    """Full text was requested for an article whose licence does not permit it."""


@dataclass(frozen=True)
class FullTextResult:
    """Full text, or an explanation of why it was withheld.

    ``text`` is ``None`` whenever ``redistributable`` is false, and the two are
    set together in one place. A result that carried the text alongside a
    "do not redistribute" flag would be redistributed by the first caller that
    read only one of the two fields.
    """

    article_id: str
    licence: str | None
    redistributable: bool
    text: str | None
    reason: str
    response: CachedResponse

    def require_text(self) -> str:
        """The text, or a refusal naming the licence that withheld it."""
        if not self.redistributable or self.text is None:
            raise NotRedistributableError(
                f"full text for {self.article_id} was not returned: "
                f"{self.reason}")
        return self.text

    def to_dict(self) -> dict[str, Any]:
        return {"article_id": self.article_id, "licence": self.licence,
                "redistributable": self.redistributable,
                "text_present": self.text is not None, "reason": self.reason}


class EuropePMCConnector(_CitationConnector):
    """Europe PMC citations, and full text only within an open licence.

    Reads the payload shape::

        {"records": [{"identifier": "PMC...", "doi": "...", "title": "...",
                      "license": "cc-by-4.0", "is_open_access": true,
                      "full_text": "..."}]}

    The registry records this source's ``redistribution_allowed`` capability as
    ``unknown`` and its curation note requires implementing "the refusal path
    for non-open full text before any extraction runs". That refusal path is
    :meth:`full_text`.
    """

    source_id = "europe_pmc"
    description = "Europe PMC: literature records and open-licence full text"

    def full_text(self, article_id: str) -> FullTextResult:
        """Return the article's text only if its licence is on the allow-list.

        Four outcomes, each distinct because each needs a different action: the
        record is not cached; the record states no licence; the licence is not
        open; the licence is open and the text is present. Collapsing the
        middle two would let an unlicensed article be treated as merely missing
        and fetched again later by somebody less careful.
        """
        response = self.guarded("fetch", article_id, "exact_record_fetch")
        if not response.ok:
            return FullTextResult(
                article_id, None, False, None,
                f"not available locally: "
                f"{response.miss_reason or response.status.value}; no text was "
                f"synthesised for it",
                response)
        rows = self.records(response)
        if len(rows) != 1:
            return FullTextResult(
                article_id, None, False, None,
                f"the cached payload holds {len(rows)} records for this "
                f"article, so the licence cannot be attributed to one of them",
                response)
        row = rows[0]
        licence = _text(row.get("license")) or _text(row.get("licence"))
        text = _text(row.get("full_text"))
        if licence is None:
            return FullTextResult(
                article_id, None, False, None,
                "the record states no licence; an unstated licence is not an "
                "open one, so the text is marked not redistributable",
                response)
        if not is_open_redistributable(licence):
            return FullTextResult(
                article_id, licence, False, None,
                f"licence '{licence}' is not on this project's open "
                f"redistribution allow-list "
                f"({', '.join(sorted(OPEN_REDISTRIBUTABLE_LICENCES))}); the "
                f"text may be read at source but not carried into this "
                f"project's artifacts",
                response)
        if text is None:
            return FullTextResult(
                article_id, licence, True, None,
                f"licence '{licence}' permits redistribution but the cached "
                f"record carries no text; none was generated for it",
                response)
        return FullTextResult(
            article_id, licence, True, text,
            f"licence '{licence}' is on the open redistribution allow-list",
            response)


# ---------------------------------------------------------------------------
# EnzChemRED
# ---------------------------------------------------------------------------

class EnzChemREDAsset(str, enum.Enum):
    """The two assets EnzChemRED ships, kept apart from the first import.

    The expert-annotated corpus and further text-mined data have different
    authority and different error modes. Once merged they cannot reliably be
    separated again, because the merged rows carry no marker saying which they
    were -- which is why they are different cache keys and different tiers
    here rather than a column.
    """

    EXPERT_ANNOTATED = "expert_annotated"
    TEXT_MINED = "text_mined"

    @property
    def tier(self) -> EvidenceTier:
        """Where rows from this asset enter.

        Text-mined rows enter at ``machine_extracted_pending``: usable for
        triage, never as a label, and promotable only by a named human.
        """
        return (EvidenceTier.CURATED_DATABASE
                if self is EnzChemREDAsset.EXPERT_ANNOTATED
                else EvidenceTier.MACHINE_EXTRACTED_PENDING)

    @property
    def extraction_method(self) -> ExtractionMethod:
        return (ExtractionMethod.HUMAN_READING_PRIMARY
                if self is EnzChemREDAsset.EXPERT_ANNOTATED
                else ExtractionMethod.MACHINE_EXTRACTION_LLM)


class CorpusMergeRefusedError(LayerSemanticsError):
    """Two corpora with different authority were asked to become one table.

    Separating them afterwards is not reliably possible, so the refusal has to
    happen before the merge rather than being corrected after it.
    """


@dataclass(frozen=True)
class EnzChemREDAnnotation:
    """One annotated relation from the corpus, with its asset recorded."""

    asset: EnzChemREDAsset
    annotation_id: str
    publication_id: str | None
    uniprot_accession: str | None
    chebi_ids: tuple[str, ...]
    rhea_ids: tuple[str, ...]
    ec_numbers: tuple[str, ...]
    text_span: str | None
    evidence: EvidenceRef

    def to_experiment_record(self) -> ExperimentRecord:
        """A record pointing at the annotated relation, claiming no outcome.

        The corpus annotates that a paper mentions a protein, a chemical and a
        reaction together. That is a relation, not a result, so the outcome
        stays ``not_tested``.
        """
        return ExperimentRecord(
            record_id=f"enzchemred:{self.asset.value}:{self.annotation_id}",
            accession=self.uniprot_accession,
            substrate=SubstrateSpec(name=None,
                                    notes="; ".join(self.chebi_ids)),
            reaction_id=self.rhea_ids[0] if self.rhea_ids else None,
            outcome=OutcomeClass.NOT_TESTED,
            evidence=[self.evidence],
            notes=(f"{self.asset.value} annotation; the corpus records that a "
                   f"publication relates these entities, not that an assay was "
                   f"performed"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"asset": self.asset.value, "annotation_id": self.annotation_id,
                "publication_id": self.publication_id,
                "uniprot_accession": self.uniprot_accession,
                "chebi_ids": list(self.chebi_ids),
                "rhea_ids": list(self.rhea_ids),
                "ec_numbers": list(self.ec_numbers),
                "text_span": self.text_span}


@dataclass(frozen=True)
class EnzChemREDCorpus:
    """One asset's annotations, with no operation that joins it to the other."""

    asset: EnzChemREDAsset
    annotations: tuple[EnzChemREDAnnotation, ...]
    response: CachedResponse
    upstream_sources: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def storage_partition(self) -> str:
        """Where these rows are stored. The two assets never share a partition."""
        return f"enzchemred/{self.asset.value}"

    def merged_with(self, other: "EnzChemREDCorpus") -> None:
        """Always refuses when the assets differ. See :class:`CorpusMergeRefusedError`."""
        if other.asset is self.asset:
            raise CorpusMergeRefusedError(
                f"both corpora are the '{self.asset.value}' asset; concatenate "
                f"them through the intake store, which keeps the tier, rather "
                f"than here")
        raise CorpusMergeRefusedError(
            f"refusing to merge the '{self.asset.value}' and "
            f"'{other.asset.value}' assets: they carry different authority and "
            f"enter at different tiers ({self.asset.tier.value} versus "
            f"{other.asset.tier.value}). Once merged, which row was which "
            f"cannot be recovered.")

    def to_intake_records(self, registry: SourceRegistry, *,
                          at: str | None = None) -> tuple[IntakeRecord, ...]:
        """Admit this asset's rows at its own tier and the weakest strength.

        Text-mined rows land at ``machine_extracted_pending`` because that is
        what :attr:`EnzChemREDAsset.tier` says, and
        :func:`~eagent.datalayer.intake.ingest` refuses to place a machine
        extraction any higher. Nothing here can promote them.
        """
        return tuple(
            ingest(annotation.to_experiment_record(),
                   tier=self.asset.tier,
                   source_id="enzchemred",
                   registry=registry,
                   extraction_method=self.asset.extraction_method,
                   claimed_strength=None,
                   uncertainties=self.notes,
                   at=at)
            for annotation in self.annotations)

    def to_dict(self) -> dict[str, Any]:
        return {"asset": self.asset.value,
                "storage_partition": self.storage_partition,
                "n_annotations": len(self.annotations),
                "upstream_sources": list(self.upstream_sources),
                "notes": list(self.notes)}


class EnzChemREDConnector(RegistryBackedConnector):
    """EnzChemRED, with its two assets kept in separate cache namespaces.

    Reads the payload shape, keyed per asset::

        {"records": [{"annotation_id": "...", "publication_id": "PMID:...",
                      "uniprot_accession": "...", "chebi_ids": ["CHEBI:..."],
                      "rhea_ids": ["RHEA:..."], "ec_numbers": ["1.1.1.1"],
                      "text_span": "..."}]}

    ``keyword_query`` is registered ``not_supported`` for this source, so
    :meth:`search` is a typed refusal: the corpus is obtained as a bulk asset
    and read from the cache, not queried per record. The asset read is
    therefore gated on ``bulk_snapshot``, which is registered ``unknown`` and
    so marks every result unverified.
    """

    source_id = "enzchemred"
    data_layer = ConnectorLayer.LITERATURE
    description = "EnzChemRED: enzyme chemistry relation corpus"

    def corpus(self, asset: EnzChemREDAsset | str, *,
               slice_id: str = "all") -> EnzChemREDCorpus:
        """One asset's annotations. The asset is part of the cache key.

        Making the asset part of the key is what keeps the two apart in
        storage: a curator importing the text-mined slice cannot overwrite the
        expert-annotated one, and a reader cannot receive a mixture without
        asking for one.
        """
        chosen = EnzChemREDAsset(asset)
        response = self.guarded("search",
                                {"asset": chosen.value, "slice": slice_id},
                                "bulk_snapshot")
        upstream = upstream_sources_for(self.source_id, self.registry)
        notes = [
            f"asset '{chosen.value}': stored in its own partition and entered "
            f"at tier {chosen.tier.value}; the two assets are never merged",
            f"re-integrated lineage: {', '.join(upstream)}",
        ]
        if chosen is EnzChemREDAsset.TEXT_MINED:
            notes.append(
                "machine-extracted: usable for triage, never as a label, and "
                "promotable only by a named human reviewer who read the paper")
        if not self.source.derived_from_complete:
            notes.append(
                "this source's lineage is registered as incomplete, so it may "
                "not be counted as independent corroboration")
        annotations = tuple(
            EnzChemREDAnnotation(
                asset=chosen,
                annotation_id=_text(row.get("annotation_id")) or f"row{index}",
                publication_id=_text(row.get("publication_id")),
                uniprot_accession=_text(row.get("uniprot_accession")),
                chebi_ids=_strings(row.get("chebi_ids")),
                rhea_ids=_strings(row.get("rhea_ids")),
                ec_numbers=_strings(row.get("ec_numbers")),
                text_span=_text(row.get("text_span")),
                evidence=evidence_ref_for(
                    self.source,
                    _text(row.get("annotation_id")) or f"row{index}",
                    registry=self.registry,
                    locator=f"asset={chosen.value}",
                    database_version=response.database_version,
                    retrieved_at=response.retrieved_at))
            for index, row in enumerate(self.records(response), start=1))
        return EnzChemREDCorpus(chosen, annotations, response,
                                upstream_sources=upstream, notes=tuple(notes))


# ---------------------------------------------------------------------------
# Zenodo
# ---------------------------------------------------------------------------

class ChecksumMismatchError(LayerSemanticsError):
    """A downloaded file is not the file the record says it is.

    Raised after the bytes are read and before anything is written under the
    final name. A dataset pinned to a snapshot it does not match is the
    failure the snapshot module exists to prevent; catching it at download
    time means the bad file never has a name anybody could load it by.
    """


def zenodo_record_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a live Zenodo record into this connector's payload.

    The two DOIs are kept apart because they mean different things: ``doi`` is
    this record's own version, ``conceptdoi`` resolves to whichever version is
    latest. The payload's ``version_doi`` is the first and ``concept_doi`` the
    second, never the other way round -- a run pinned to the concept DOI is not
    pinned.

    The licence is read from the deposit's own metadata and is ``None`` when
    the deposit states none, never defaulted to an open licence. Each file
    keeps the checksum Zenodo records for it (``md5:<hex>``), which is what a
    download is later verified against.
    """
    metadata = record.get("metadata") or {}
    licence = metadata.get("license")
    licence_id = licence.get("id") if isinstance(licence, Mapping) else licence
    files = []
    for item in record.get("files") or ():
        if not isinstance(item, Mapping) or not item.get("key"):
            continue
        links = item.get("links") or {}
        files.append({
            "filename": str(item["key"]),
            "size": item.get("size"),
            "checksum": item.get("checksum"),
            "url": links.get("self"),
        })
    return {
        "deposit_id": str(record.get("id") or record.get("recid") or ""),
        "concept_doi": record.get("conceptdoi"),
        "version_doi": record.get("doi") or metadata.get("doi"),
        "version": metadata.get("version"),
        "title": metadata.get("title") or record.get("title"),
        "license": licence_id,
        "access_right": metadata.get("access_right"),
        "created": record.get("created"),
        "modified": record.get("modified"),
        "files": files,
    }


@dataclass(frozen=True)
class DownloadedFile:
    """A file fetched from a deposit and verified against its record."""

    path: Path
    filename: str
    deposit_id: str
    version_doi: str | None
    license: str | None
    size_bytes: int
    md5: str
    sha256: str
    retrieved_at: str
    source_url: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "filename": self.filename,
                "deposit_id": self.deposit_id, "version_doi": self.version_doi,
                "license": self.license, "size_bytes": self.size_bytes,
                "md5": self.md5, "sha256": self.sha256,
                "retrieved_at": self.retrieved_at, "source_url": self.source_url}


#: Largest single file a download will accept unless the caller raises it.
#: Not a scientific number: it is a guard against pulling a multi-gigabyte
#: archive into a run that asked for a table. The record states every file's
#: size, so exceeding this is known before a byte is fetched.
DEFAULT_MAX_DOWNLOAD_BYTES: int = 200 * 1024 * 1024


@dataclass(frozen=True)
class ZenodoDeposit:
    """An author data archive, with the two DOIs kept apart.

    A concept DOI always resolves to the latest version. A run pinned to one is
    not pinned at all: its inputs change whenever the authors upload again, and
    nothing in the manifest records that they did.
    """

    deposit_id: str
    concept_doi: str | None
    version_doi: str | None
    version: str | None
    title: str | None
    license: str | None
    files: tuple[Mapping[str, Any], ...]
    evidence: EvidenceRef
    response: CachedResponse
    notes: tuple[str, ...] = ()

    @property
    def redistributable(self) -> bool:
        """Whether this deposit's own licence allows redistribution.

        Per deposit, never per repository: Zenodo hosts deposits under many
        licences and the registry says "licence is per deposit and must be read
        per deposit; nothing general can be asserted here".
        """
        return is_open_redistributable(self.license)

    def require_version_doi(self) -> str:
        """The version DOI, or a refusal naming the concept DOI problem."""
        if not self.version_doi:
            raise LayerSemanticsError(
                f"Zenodo deposit {self.deposit_id} records "
                f"{'only a concept DOI (' + self.concept_doi + ')' if self.concept_doi else 'no DOI'}. "
                f"A concept DOI resolves to whatever the latest version is, so "
                f"a run pinned to it is not reproducible. Record the version "
                f"DOI of the deposit actually used.")
        return self.version_doi

    def to_dict(self) -> dict[str, Any]:
        return {"deposit_id": self.deposit_id, "concept_doi": self.concept_doi,
                "version_doi": self.version_doi, "version": self.version,
                "title": self.title, "license": self.license,
                "redistributable": self.redistributable,
                "n_files": len(self.files), "notes": list(self.notes)}


class ZenodoConnector(RegistryBackedConnector):
    """Author data archives, with per-deposit licence and DOI discipline.

    Reads the payload shape::

        {"records": [{"deposit_id": "...", "concept_doi": "10.5281/zenodo....",
                      "version_doi": "10.5281/zenodo....", "version": "v2",
                      "title": "...", "license": "cc-by-4.0",
                      "files": [{"filename": "...", "checksum": "..."}]}]}

    Every capability on this source that matters for retrieval is registered
    ``unknown``, so results are marked unverified. ``version_information`` is
    the one ``supported`` flag, which is why the version DOI is treated as
    required rather than optional.
    """

    source_id = "zenodo"
    data_layer = ConnectorLayer.LITERATURE
    description = "Zenodo: author-deposited datasets"
    #: The requests this client makes are the two the shipped probes check:
    #: ``/api/records/<id>`` and ``/api/records/<id>/files/<name>/content``.
    verified_route_capability = "exact_record_fetch"

    def _fetch_remote(self, key: str) -> tuple[Any, str | None]:
        """One record's metadata, in the probed shape."""
        base = self.require_endpoint().rstrip("/")
        body, _ = self._http_json(
            f"{base}/api/records/{urllib.parse.quote(str(key).strip(), safe='')}")
        if not isinstance(body, Mapping) or not (body.get("id") or body.get("recid")):
            return None, None
        payload = zenodo_record_payload(body)
        # The version DOI is Zenodo's own statement of which version this is;
        # it doubles as the release string, so a cached record is pinned to
        # the version it was read from rather than to "latest".
        return {"records": [payload]}, payload.get("version_doi")

    def download_file(self, deposit: "ZenodoDeposit", filename: str,
                      destination: str | Path, *,
                      max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
                      require_checksum: bool = True) -> DownloadedFile:
        """Fetch one file of a deposit and verify it against the record.

        The order is the point: read into memory under a hard cap, hash it,
        compare with the checksum **the record states**, and only then write
        it -- to a temporary name first, renamed on success. A file that fails
        verification never exists under its final name, so nothing can load it
        by accident.

        Refused rather than attempted: a filename the deposit does not list,
        a file larger than ``max_bytes`` (the record states every size, so
        this is known before fetching), and -- unless ``require_checksum`` is
        turned off by a caller who has decided to accept it -- a file the
        record gives no checksum for, because a download that cannot be shown
        to be the same file next time pins nothing.
        """
        entry = next((f for f in deposit.files if f.get("filename") == filename),
                     None)
        if entry is None:
            raise LayerSemanticsError(
                f"deposit {deposit.deposit_id} lists no file named "
                f"{filename!r}; it lists "
                f"{', '.join(sorted(str(f.get('filename')) for f in deposit.files)[:8])}")
        size = _int(entry.get("size"))
        if size is not None and size > max_bytes:
            raise LayerSemanticsError(
                f"{filename} is {size} bytes, over the {max_bytes}-byte cap; "
                f"raise max_bytes deliberately if this is the file you want")
        declared = _text(entry.get("checksum"))
        if declared is None and require_checksum:
            raise LayerSemanticsError(
                f"the record states no checksum for {filename}, so a download "
                f"could not be shown to be the same file next time. Pass "
                f"require_checksum=False to accept that explicitly")
        algorithm, _, expected = (declared or "").partition(":")
        if declared is not None and algorithm.lower() != "md5":
            raise LayerSemanticsError(
                f"{filename}: checksum {declared!r} uses {algorithm!r}; this "
                f"client verifies md5, which is what Zenodo records")

        base = self.require_endpoint().rstrip("/")
        url = (f"{base}/api/records/"
               f"{urllib.parse.quote(deposit.deposit_id, safe='')}/files/"
               f"{urllib.parse.quote(filename, safe='')}/content")
        body = self._http_bytes(url, max_bytes=max_bytes)
        if body is None:
            raise LayerSemanticsError(
                f"{filename} is listed by deposit {deposit.deposit_id} but the "
                f"service returned nothing for it")
        md5 = hashlib.md5(body, usedforsecurity=False).hexdigest()  # noqa: S324
        if declared is not None and md5 != expected.lower():
            raise ChecksumMismatchError(
                f"{filename} downloaded as md5 {md5} but the record states "
                f"{expected}. The file was not written.")
        if size is not None and len(body) != size:
            raise ChecksumMismatchError(
                f"{filename} is {len(body)} bytes but the record states "
                f"{size}. The file was not written.")

        directory = Path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        # Only a bare file name may reach the filesystem; a record that lists
        # "../x" must not be able to write outside the destination.
        target = (directory / Path(filename).name).resolve()
        if directory.resolve() not in target.parents:
            raise LayerSemanticsError(
                f"{filename!r} would be written outside {directory}")
        temporary = target.with_name(f".{target.name}.part")
        temporary.write_bytes(body)
        os.replace(temporary, target)
        return DownloadedFile(
            path=target, filename=filename, deposit_id=deposit.deposit_id,
            version_doi=deposit.version_doi, license=deposit.license,
            size_bytes=len(body), md5=md5,
            sha256=hashlib.sha256(body).hexdigest(),
            retrieved_at=utc_now(), source_url=url)

    def deposit(self, deposit_id: str) -> ZenodoDeposit | None:
        response = self.fetch(deposit_id)
        if not response.ok:
            return None
        rows = self.records(response)
        if len(rows) != 1:
            return None
        row = rows[0]
        licence = _text(row.get("license")) or _text(row.get("licence"))
        files = tuple(dict(f) for f in (row.get("files") or [])
                      if isinstance(f, Mapping))
        notes = [
            "licence is per deposit and was read from this deposit's own "
            "record; nothing general about the repository is asserted",
        ]
        if licence is None:
            notes.append(
                "this deposit states no licence, so its contents are treated as "
                "not redistributable until a curator reads the deposit page")
        if not _text(row.get("version_doi")):
            notes.append(
                "no version DOI recorded; a concept DOI moves with every new "
                "upload and cannot pin a run")
        if any(not _text(f.get("checksum")) for f in files):
            notes.append(
                "at least one file has no recorded checksum, so a re-download "
                "cannot be shown to be the same file")
        return ZenodoDeposit(
            deposit_id=_text(row.get("deposit_id")) or deposit_id,
            concept_doi=_text(row.get("concept_doi")),
            version_doi=_text(row.get("version_doi")),
            version=_text(row.get("version")),
            title=_text(row.get("title")),
            license=licence, files=files,
            evidence=evidence_ref_for(
                self.source, _text(row.get("version_doi"))
                or _text(row.get("deposit_id")) or deposit_id,
                registry=self.registry,
                database_version=_text(row.get("version")),
                retrieved_at=response.retrieved_at),
            response=response, notes=tuple(notes))


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
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
