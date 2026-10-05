"""The map from a registered data source to the code that reads it.

Without this, the per-resource connector modules are unreachable: an interface
would have to import ``eagent.connectors.structure`` and name
``SIFTSConnector`` itself, which is how one step ends up with a live client
and another with a cache-only stub for the same resource.

Three kinds of entry, and the difference is the point:

* a **connector** for a resource documented to offer programmatic access;
* an **importer** for a resource that offers only a website or an author's
  file, which reads what a curator placed on disk and records who placed it;
* **nothing**, for a registered resource with no code yet, which falls back to
  a cache-only connector rather than to a plausible-looking client.

:func:`build_connectors` returns one object per requested source and says in
its report which of the three each one got, so a run can state what it could
actually read rather than implying it reached all 49.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .base import AccessPolicy, Connector, FileCache, OfflineConnector

__all__ = [
    "CONCRETE_CONNECTORS",
    "CURATED_IMPORTERS",
    "AccessKind",
    "BuiltConnector",
    "ConnectorBuildReport",
    "connector_class_for",
    "importer_class_for",
    "build_connectors",
    "coverage_report",
]


#: datasource id -> "module:ClassName" for resources with a real client.
#: Written as strings so importing this module does not drag in all five
#: topic modules, and so a typo surfaces as a clear error at build time
#: rather than as a missing attribute deep in a run.
CONCRETE_CONNECTORS: dict[str, str] = {
    # reaction and chemistry
    "chebi": "chemistry:ChEBIConnector",
    "enzymemap": "chemistry:EnzymeMapConnector",
    "metanetx": "chemistry:MetaNetXConnector",
    "pubchem": "chemistry:PubChemConnector",
    "rhea": "chemistry:RheaConnector",
    # enzymology evidence
    "brenda": "enzymology:BRENDAConnector",
    "enzengdb": "enzymology:EnzEngDBConnector",
    "sabio_rk": "enzymology:SABIORKConnector",
    # sequence, family and evolution
    "interpro": "sequence:InterProConnector",
    "mgnify_proteins": "sequence:MGnifyProteinsConnector",
    "ncbi_protein": "sequence:NCBIIdenticalProteinGroupsConnector",
    "pfam": "sequence:PfamConnector",
    "uniparc": "sequence:UniParcConnector",
    "uniprotkb": "sequence:UniProtKBConnector",
    "uniref": "sequence:UniRefConnector",
    # structure and mechanism
    "alphafill": "structure:AlphaFillConnector",
    "alphafold_db": "structure:AlphaFoldDBConnector",
    "mcsa": "structure:MCSAConnector",
    "rcsb_pdb": "structure:RCSBPDBConnector",
    "sifts": "structure:SIFTSConnector",
    "wwpdb_ccd": "structure:WwPDBChemicalComponentConnector",
    # literature and feedback
    "enzchemred": "literature:EnzChemREDConnector",
    "europe_pmc": "literature:EuropePMCConnector",
    "pubmed": "literature:PubMedConnector",
    "zenodo": "literature:ZenodoConnector",
}

#: datasource id -> "module:ClassName" for resources whose records arrive by a
#: curator's file. These deliberately have no fetch path at all; constructing
#: one as a client raises.
CURATED_IMPORTERS: dict[str, str] = {
    "akr_superfamily": "sequence:AKRSuperfamilyImporter",
    "sdred": "sequence:SDREDImporter",
    "oed": "enzymology:OEDImporter",
    "protabank": "enzymology:ProtaBankImporter",
    "retrobiocat_db": "enzymology:RetroBioCatDBImporter",
    "strenda_db": "enzymology:STRENDADBImporter",
}


class AccessKind(str):
    """How a source is reached, as a plain string with named constants."""

    CONNECTOR = "connector"
    IMPORTER = "importer"
    CACHE_ONLY = "cache_only"


def _resolve(spec: str) -> type:
    module_name, _, class_name = spec.partition(":")
    module = importlib.import_module(f".{module_name}", __package__)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:  # pragma: no cover - catalogue typo
        raise AttributeError(
            f"catalogue points at {spec}, which does not exist. The catalogue "
            f"and the connector modules have drifted apart."
        ) from exc


def connector_class_for(source_id: str) -> type | None:
    """The client class for a source, or None when none has been written."""
    spec = CONCRETE_CONNECTORS.get(source_id)
    return _resolve(spec) if spec else None


def importer_class_for(source_id: str) -> type | None:
    """The curated-file importer for a source, or None."""
    spec = CURATED_IMPORTERS.get(source_id)
    return _resolve(spec) if spec else None


@dataclass(frozen=True)
class BuiltConnector:
    """One wired source, with how it is reached and why."""

    source_id: str
    kind: str
    obj: Any
    reason: str = ""

    @property
    def is_cache_only(self) -> bool:
        return self.kind == AccessKind.CACHE_ONLY


@dataclass
class ConnectorBuildReport:
    """What a run can actually read, stated rather than implied."""

    built: dict[str, BuiltConnector] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)

    def by_kind(self, kind: str) -> list[str]:
        return sorted(k for k, v in self.built.items() if v.kind == kind)

    @property
    def connectors(self) -> dict[str, Any]:
        return {k: v.obj for k, v in self.built.items()}

    def report_lines(self) -> list[str]:
        lines = [
            f"{len(self.built)} source(s) wired: "
            f"{len(self.by_kind(AccessKind.CONNECTOR))} with a client, "
            f"{len(self.by_kind(AccessKind.IMPORTER))} by curated import, "
            f"{len(self.by_kind(AccessKind.CACHE_ONLY))} cache-only"
        ]
        for sid in self.by_kind(AccessKind.CACHE_ONLY):
            lines.append(f"  - {sid}: {self.built[sid].reason}")
        for sid, why in sorted(self.failures.items()):
            lines.append(f"  ! {sid}: could not be wired -- {why}")
        return lines


def build_connectors(
    source_ids: Iterable[str],
    *,
    registry: Any = None,
    cache: FileCache | None = None,
    access: AccessPolicy | None = None,
    snapshots: Mapping[str, str] | None = None,
) -> ConnectorBuildReport:
    """Wire each requested source to the best code available for it.

    A source with no client and no importer gets a cache-only connector and is
    named as such in the report. That is the honest fallback: it can still
    serve a record a curator placed in the cache, and it cannot pretend to
    fetch one that is not there.
    """
    report = ConnectorBuildReport()
    snaps = dict(snapshots or {})
    for sid in source_ids:
        kwargs: dict[str, Any] = {}
        if cache is not None:
            kwargs["cache"] = cache
        if access is not None:
            kwargs["access"] = access
        if registry is not None:
            kwargs["registry"] = registry
        try:
            cls = connector_class_for(sid)
            if cls is not None:
                report.built[sid] = BuiltConnector(
                    sid, AccessKind.CONNECTOR, cls(**kwargs),
                    "documented programmatic access")
                continue
            cls = importer_class_for(sid)
            if cls is not None:
                report.built[sid] = BuiltConnector(
                    sid, AccessKind.IMPORTER, cls(**kwargs),
                    "records arrive by a curator's file, not an API")
                continue
        except Exception as exc:
            report.failures[sid] = f"{type(exc).__name__}: {exc}"
            continue
        report.built[sid] = BuiltConnector(
            sid, AccessKind.CACHE_ONLY,
            OfflineConnector(sid, None, cache=cache,
                             snapshot=snaps.get(sid), access=access),
            "no client written yet; serves only what a curator cached")
    return report


def coverage_report(registry: Any) -> dict[str, list[str]]:
    """Which registered sources have code, grouped by how they are reached.

    Lets a run answer "what can this system actually read" without inferring
    it from which imports happen to succeed.
    """
    ids = sorted(s.id for s in registry)
    return {
        AccessKind.CONNECTOR: [i for i in ids if i in CONCRETE_CONNECTORS],
        AccessKind.IMPORTER: [i for i in ids if i in CURATED_IMPORTERS],
        AccessKind.CACHE_ONLY: [i for i in ids
                                if i not in CONCRETE_CONNECTORS
                                and i not in CURATED_IMPORTERS],
    }
