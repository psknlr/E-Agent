"""Structure and mechanism connectors: RCSB PDB, AlphaFold DB, SIFTS,
the wwPDB Chemical Component Dictionary, M-CSA and AlphaFill.

Why this module exists
----------------------
Every number this project computes about a binding site is a distance between
two atoms in two numbering systems, taken from a file whose provenance decides
what the number means. Five things go wrong here, each with a guard:

*A prediction read as an observation.* An AlphaFold model has no ligands, no
cofactor and no occupancy. :class:`PredictedModelRecord` refuses to be read as
an observed structure, because an apo backbone prediction scored as though it
were a holo crystal structure produces a confident pocket measurement of a
pocket nobody has seen.

*A numbering offset guessed.* UniProt position 155 is not author residue 155
plus a constant: constructs are truncated, tagged and renumbered, and
insertion codes exist. :class:`SiftsResidueMapping` is a per-residue map and
:meth:`SiftsResidueMapping.numbering_offset` refuses to exist. A guessed offset
mutates a neighbour of the intended residue, and the resulting construct looks
perfectly reasonable.

*Ligand atoms matched by file order.* Two depositions of the same chemical
component can list its atoms in different orders. :class:`ChemicalComponent`
matches atoms by component id and atom name only, and
:meth:`ChemicalComponent.match_by_order` refuses, because an order-based match
silently swaps the nicotinamide C4 for some other carbon and every hydride
distance after that is a distance to the wrong atom.

*A transplanted ligand reported as observed.* AlphaFill places ligands by
homology. Every :class:`TransplantedLigand` is tagged
:attr:`~eagent.schemas.chem.LigandSource.HOMOLOGY_TRANSPLANTED` and the tag
cannot be overridden, because a transplant recorded as an observation turns a
hypothesis into a measurement with nobody having decided that it should.

*Catalytic residues without a source.* An unsourced catalytic template is a
guess with a template's authority. :class:`MCSAEntry` carries the reference
enzyme and the literature identifiers into a
:class:`~eagent.schemas.templates.TemplateProvenance`, and refuses to build one
when the record names no source.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..provenance import utc_now
from ..schemas.chem import CofactorSpec, CofactorState, LigandSource, \
    cofactor_state_from_ligand_code
from ..schemas.templates import TemplateProvenance, TemplateSourceType
from ..science.numbering import AuthorPosition
from .base import CachedResponse, ConnectorLayer, NetworkDisabledError
from .chemistry import LayerSemanticsError, RegistryBackedConnector
from .literature import ChecksumMismatchError

__all__ = [
    "AlphaFillConnector",
    "AlphaFoldDBConnector",
    "AtomOrderMatchRefusedError",
    "CCDAtom",
    "CCDAtomMatch",
    "CCDBond",
    "CatalyticResidueRecord",
    "ChemicalComponent",
    "DEFAULT_MAX_STRUCTURE_BYTES",
    "FileRouteNotVerifiedError",
    "MCSAConnector",
    "MCSAEntry",
    "ModelledLigand",
    "NumberingOffsetRefusedError",
    "PDBEntryRecord",
    "PredictedModelRecord",
    "PredictionNotObservationError",
    "RCSBPDBConnector",
    "SIFTSConnector",
    "SiftsResiduePair",
    "SiftsResidueMapping",
    "RCSB_STRUCTURE_FILE_PATH",
    "StructureFile",
    "TransplantedLigand",
    "WwPDBChemicalComponentConnector",
    "validate_mmcif_bytes",
]


# ---------------------------------------------------------------------------
# RCSB PDB
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelledLigand:
    """A ligand actually present in a deposited model.

    ``occupancy`` is ``None`` when the record does not state it, never 1.0. A
    fabricated full occupancy would let a partially modelled ligand pass the
    screen that exists to catch exactly that.
    """

    component_id: str
    chain: str | None
    author_seq_id: int | None
    occupancy: float | None
    source: LigandSource = LigandSource.EXPERIMENTAL_OBSERVED

    @property
    def cofactor_state(self) -> CofactorState:
        """Oxidation state read from the component id, or ``UNKNOWN``.

        Delegates to :func:`~eagent.schemas.chem.cofactor_state_from_ligand_code`
        rather than restating the table, because NAD+ and NADH differ by one
        hydride and by one letter of a three-letter code.
        """
        return cofactor_state_from_ligand_code(self.component_id)

    def to_dict(self) -> dict[str, Any]:
        return {"component_id": self.component_id, "chain": self.chain,
                "author_seq_id": self.author_seq_id,
                "occupancy": self.occupancy, "source": self.source.value,
                "cofactor_state": self.cofactor_state.value}


@dataclass(frozen=True)
class PDBEntryRecord:
    """A deposited entry: what was modelled, how, and how well.

    ``deposited_sequence`` is kept separate from any candidate's sequence. The
    crystallised construct routinely carries tags, truncations and surface
    mutations, so treating the entry as a sequence record attaches this run's
    evidence to a protein that was never ordered.
    """

    pdb_id: str
    method: str | None
    resolution_angstrom: float | None
    chains: tuple[str, ...]
    ligands: tuple[ModelledLigand, ...]
    deposited_sequences: Mapping[str, str]
    response: CachedResponse
    notes: tuple[str, ...] = ()

    def ligand(self, component_id: str) -> ModelledLigand | None:
        """One modelled ligand by component id, matched case-insensitively."""
        want = component_id.strip().upper()
        for lig in self.ligands:
            if lig.component_id.strip().upper() == want:
                return lig
        return None

    def as_activity_evidence(self) -> None:
        """Always refuses: a bound cofactor is not turnover."""
        raise LayerSemanticsError(
            f"PDB {self.pdb_id} records coordinates, not catalysis. A bound "
            f"cofactor shows that the cofactor was present in the crystal, not "
            f"that this protein turns over the target substrate.")

    def to_dict(self) -> dict[str, Any]:
        return {"pdb_id": self.pdb_id, "method": self.method,
                "resolution_angstrom": self.resolution_angstrom,
                "chains": list(self.chains),
                "ligands": [l.to_dict() for l in self.ligands],
                "notes": list(self.notes)}


def rcsb_entry_payload(
    entry: Mapping[str, Any],
    polymer_entities: Mapping[str, Mapping[str, Any]],
    nonpolymer_entities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Translate live RCSB core-API responses into this connector's payload.

    Three response kinds, because that is how the API divides a structure: the
    entry (method, resolution, entity ids), one record per polymer entity (the
    sequence and the chains it occupies) and one per non-polymer entity (the
    component id and the chains it sits on).

    WHAT THE CORE API DOES NOT CARRY
    --------------------------------
    Ligand **occupancy** and **author residue number** live in the coordinate
    file, not in the metadata. They are left ``None`` rather than defaulted:
    a ligand listed in an entry is not proof it was present at full occupancy,
    and the connector already says so in its notes whenever the field is
    absent. Filling in 1.0 here would silence that warning with a number
    nobody read.

    A non-polymer entity occupying several chains yields one ligand per chain.
    ZN and NAD each sit on both chains of 1CDO; collapsing them into one row
    per component would make the second chain's cofactor invisible.

    ``resolution`` is the entry's combined resolution, and ``None`` for a
    method that has none -- an NMR ensemble does not have a resolution of 0.
    """
    ids = entry.get("rcsb_entry_container_identifiers") or {}
    info = entry.get("rcsb_entry_info") or {}
    methods = [m.get("method") for m in (entry.get("exptl") or ())
               if isinstance(m, Mapping) and m.get("method")]
    resolutions = [r for r in (info.get("resolution_combined") or ())
                   if isinstance(r, (int, float))]

    sequences: dict[str, str] = {}
    chains: list[str] = []
    for entity_id, polymer in polymer_entities.items():
        identifiers = polymer.get("rcsb_polymer_entity_container_identifiers") or {}
        sequence = ((polymer.get("entity_poly") or {})
                    .get("pdbx_seq_one_letter_code_can") or "")
        sequence = "".join(str(sequence).split())
        for chain in identifiers.get("auth_asym_ids") or ():
            chain = str(chain)
            if chain not in chains:
                chains.append(chain)
            if sequence:
                sequences[chain] = sequence

    ligands: list[dict[str, Any]] = []
    for entity_id, nonpolymer in nonpolymer_entities.items():
        identifiers = (nonpolymer.get("rcsb_nonpolymer_entity_container_identifiers")
                       or {})
        component = (identifiers.get("nonpolymer_comp_id")
                     or (nonpolymer.get("pdbx_entity_nonpoly") or {}).get("comp_id"))
        if not component:
            continue
        for chain in identifiers.get("auth_asym_ids") or ():
            ligands.append({"component_id": str(component), "chain": str(chain),
                            "author_seq_id": None, "occupancy": None})

    return {
        "pdb_id": _text(ids.get("entry_id") or entry.get("rcsb_id")),
        "method": methods[0] if methods else None,
        "resolution": min(resolutions) if resolutions else None,
        "chains": chains,
        "ligands": ligands,
        "sequences": sequences,
        "entity_ids": {
            "polymer": list(ids.get("polymer_entity_ids") or ()),
            "non_polymer": list(ids.get("non_polymer_entity_ids") or ()),
        },
    }


#: Coordinate files are served from a different host than the data API above,
#: and **no host is written here**: this package holds no URL literal, because a
#: literal is a route nobody verified. The host the client calls is whichever
#: one the registry holds a *passing* probe for, of exactly this shape -- the
#: ``.../download/<ID>.cif`` entry in :data:`eagent.datalayer.probe.PROBES`, which
#: asks for the same path. A verified route for the data API is not a verified
#: route for the file host, and until a probe of this shape has passed the
#: client does not call anything.
#:
#: The asymmetric-unit file, not an assembly: author chain identifiers in the
#: file are the ones the wwPDB validation report is written in.
RCSB_STRUCTURE_FILE_PATH: str = "/download/{pdb_id}.cif"

#: A recorded check URL of the file shape; group 1 is ``scheme://host``. Written
#: without the scheme literal on purpose (see above).
_FILE_CHECK_URL = re.compile(r"^([a-z]+://[^/\s]+)/download/[0-9][A-Za-z0-9]{3}\.cif$")

#: A guard against pulling a ribosome into a run that asked for a kinase. Not a
#: scientific number: the asymmetric unit of an enzyme complex is megabytes.
DEFAULT_MAX_STRUCTURE_BYTES: int = 64 * 1024 * 1024

_PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")


class FileRouteNotVerifiedError(NetworkDisabledError):
    """The file host has not been shown to answer the request this client makes.

    A base URL for the data API is established; the file host is another
    service. Until ``eagent sources verify`` has recorded a passing check for a
    ``.../download/<ID>.cif`` request the client does not call it, because a
    request shape that was never tried is a guess, and a guess that fails with a
    404 looks like "the entry does not exist".
    """

    def __init__(self, source_id: str, verified: Sequence[str] = ()) -> None:
        self.source_id = source_id
        super().__init__(
            f"'{source_id}': no passing connectivity check is recorded for the "
            f"file download shape {RCSB_STRUCTURE_FILE_PATH} (verified so far: "
            f"{', '.join(verified) or 'nothing'}). The data API and the file "
            f"host are different services; run `eagent sources verify "
            f"--allow-network --write` and try again")


@dataclass(frozen=True)
class StructureFile:
    """A coordinate file on disk and the hash it is known by.

    ``retrieved_at`` is ``None`` when the file was already there and was only
    verified: nothing was fetched, and a timestamp would claim otherwise, and
    ``source_url`` is empty for the same reason.
    """

    pdb_id: str
    path: Path
    source_url: str
    size_bytes: int
    sha256: str
    retrieved_at: str | None
    from_cache: bool

    def to_dict(self) -> dict[str, Any]:
        return {"pdb_id": self.pdb_id, "path": str(self.path),
                "source_url": self.source_url, "size_bytes": self.size_bytes,
                "sha256": self.sha256, "retrieved_at": self.retrieved_at,
                "from_cache": self.from_cache}


def validate_mmcif_bytes(body: bytes, pdb_id: str) -> str:
    """The decoded text of an mmCIF file, or a refusal saying what it is instead.

    A 200 is not a coordinate file: a CDN error page, a login wall and a
    maintenance notice all answer 200. The check is cheap and specific -- the
    data block is named for this entry, ``_entry.id`` agrees with it, and an
    ``_atom_site`` loop exists -- so a file that is not the one asked for never
    receives a name a loader would trust.
    """
    pid = pdb_id.strip().upper()
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LayerSemanticsError(
            f"{pid}: the response is not UTF-8 text, so it is not an mmCIF file "
            f"({exc.reason} at byte {exc.start})") from exc
    first = next((ln.strip() for ln in text.splitlines()
                  if ln.strip() and not ln.lstrip().startswith("#")), "")
    if first.lower() != f"data_{pid}".lower():
        raise LayerSemanticsError(
            f"{pid}: the first data line is {first[:60]!r}, not 'data_{pid}'; "
            f"something other than this entry's mmCIF answered")
    entry = re.search(r"^_entry\.id\s+(\S+)", text, re.MULTILINE)
    if entry is None or entry.group(1).strip("'\"").upper() != pid:
        raise LayerSemanticsError(
            f"{pid}: _entry.id is {entry.group(1) if entry else 'absent'}, not "
            f"{pid}")
    if "_atom_site.Cartn_x" not in text:
        raise LayerSemanticsError(f"{pid}: the file has no _atom_site coordinates")
    return text


class RCSBPDBConnector(RegistryBackedConnector):
    """Experimentally determined coordinates and the ligands really in them.

    Reads the payload shape::

        {"records": [{"pdb_id": "0XYZ", "method": "X-RAY DIFFRACTION",
                      "resolution": 1.8, "chains": ["A"],
                      "ligands": [{"component_id": "NAP", "chain": "A",
                                   "author_seq_id": 301, "occupancy": 1.0}],
                      "sequences": {"A": "MK..."}}]}

    Ligands come back tagged
    :attr:`~eagent.schemas.chem.LigandSource.EXPERIMENTAL_OBSERVED` because
    they were modelled into experimental density. That is the only connector in
    this package entitled to that tag, which is why
    :class:`AlphaFillConnector` cannot reach it.
    """

    source_id = "rcsb_pdb"
    data_layer = ConnectorLayer.STRUCTURE
    description = "RCSB PDB: experimentally determined structures"
    #: The requests this client makes are the three the shipped probes check:
    #: ``/rest/v1/core/entry/<id>``, ``/core/polymer_entity/<id>/<n>`` and
    #: ``/core/nonpolymer_entity/<id>/<n>``. A verified base URL does not
    #: license any other shape.
    verified_route_capability = "exact_record_fetch"

    def _fetch_remote(self, key: str) -> tuple[Any, str | None]:
        """Fetch one entry and the entities it names, in the probed shapes.

        An entry is several resources: the metadata names its polymer and
        non-polymer entities and each is a separate request. If any of them
        fails the whole fetch fails -- a structure assembled from the entry and
        only some of its entities would present a protein with its cofactor
        missing, which is a different protein as far as every downstream check
        is concerned.
        """
        import urllib.parse

        base = self.require_endpoint().rstrip("/")
        pdb_id = urllib.parse.quote(str(key).strip().upper(), safe="")
        entry, version = self._http_json(f"{base}/rest/v1/core/entry/{pdb_id}")
        if not isinstance(entry, Mapping) or not entry.get("rcsb_id"):
            return None, version
        ids = entry.get("rcsb_entry_container_identifiers") or {}
        polymers: dict[str, Mapping[str, Any]] = {}
        for entity in ids.get("polymer_entity_ids") or ():
            body, _ = self._http_json(
                f"{base}/rest/v1/core/polymer_entity/{pdb_id}/"
                f"{urllib.parse.quote(str(entity), safe='')}")
            if not isinstance(body, Mapping):
                return None, version
            polymers[str(entity)] = body
        nonpolymers: dict[str, Mapping[str, Any]] = {}
        for entity in ids.get("non_polymer_entity_ids") or ():
            body, _ = self._http_json(
                f"{base}/rest/v1/core/nonpolymer_entity/{pdb_id}/"
                f"{urllib.parse.quote(str(entity), safe='')}")
            if not isinstance(body, Mapping):
                return None, version
            nonpolymers[str(entity)] = body
        return ({"records": [rcsb_entry_payload(entry, polymers, nonpolymers)]},
                version)

    def polymer_entity_annotations(self, pdb_id: str, entity_id: str | int
                                   ) -> dict[str, Any]:
        """The family-level annotations the RCSB attaches to one polymer entity.

        Read from the ``core/polymer_entity/<id>/<n>`` response -- the request
        shape the shipped probe checks -- and reduced to what a family question
        needs: InterPro, Pfam, and GO identifiers with their names, the entity's
        description and its EC number as the depositor gave it. These are the
        RCSB's *annotations*, not a classification this project made, and the
        returned dictionary says so by carrying nothing else.

        Not cached and not routed through :meth:`fetch`: the caller records the
        result where it is used (the reference set's coordinate manifest), which
        is the persistent copy. Needs ``allow_network`` and a verified data-API
        route, like every other call here.
        """
        import urllib.parse

        pid = str(pdb_id).strip().upper()
        if not _PDB_ID.match(pid):
            raise LayerSemanticsError(f"{pdb_id!r} is not a four-character PDB id")
        base = self.require_endpoint().rstrip("/")
        body, _ = self._http_json(
            f"{base}/rest/v1/core/polymer_entity/{pid}/"
            f"{urllib.parse.quote(str(entity_id), safe='')}")
        if not isinstance(body, Mapping):
            raise LayerSemanticsError(
                f"the RCSB has no polymer entity {entity_id} for {pid}")
        by_type: dict[str, list[dict[str, str]]] = {"InterPro": [], "Pfam": [], "GO": []}
        for item in body.get("rcsb_polymer_entity_annotation") or ():
            if not isinstance(item, Mapping):
                continue
            kind = str(item.get("type") or "")
            if kind in by_type and item.get("annotation_id"):
                by_type[kind].append({"id": str(item["annotation_id"]),
                                      "name": str(item.get("name") or "")})
        entity = body.get("rcsb_polymer_entity") or {}
        return {
            "pdb_id": pid, "entity_id": str(entity_id),
            "description": _text(entity.get("pdbx_description")),
            "ec": _text(entity.get("pdbx_ec")),
            "interpro": by_type["InterPro"], "pfam": by_type["Pfam"],
            "go": by_type["GO"],
        }

    def _file_route_base(self) -> str:
        """``scheme://host`` of the file service, from a recorded passing probe.

        Never from a constant: the base is read off the registry's own record of
        a check that passed for exactly the shape this client requests, so the
        host called is the host that was shown to answer it.
        """
        source = self.source
        for check in source.connectivity_checks:
            hit = _FILE_CHECK_URL.match(check.url) if check.ok else None
            if hit:
                return hit.group(1)
        raise FileRouteNotVerifiedError(self.source_id, source.verified_capabilities)

    def download_structure(self, pdb_id: str, destination: str | Path, *,
                           expected_sha256: str | None = None,
                           max_bytes: int = DEFAULT_MAX_STRUCTURE_BYTES,
                           refresh: bool = False) -> StructureFile:
        """Fetch one entry's mmCIF file and verify it, or refuse.

        The order is the point, and it is the order the Zenodo client uses:

        1. a file already at the destination is **verified, not replaced**. If
           its hash is not the expected one the call raises and leaves it
           alone -- overwriting a file that fails verification with whatever
           the network returns now is how a pinned input stops being pinned;
        2. the network is touched only if ``allow_network`` is on **and** the
           registry holds a passing check for this exact URL shape;
        3. the body is read under a hard cap, checked to be this entry's
           mmCIF, hashed, compared with ``expected_sha256`` when one is given,
           and only then written -- to a temporary name, renamed on success.

        The RCSB revises entries (re-refinement, remediation), so the same
        identifier can serve different bytes next year. ``expected_sha256`` is
        what turns "the file for 1IPF" into "the file this analysis used".
        """
        pid = str(pdb_id).strip().upper()
        if not _PDB_ID.match(pid):
            raise LayerSemanticsError(
                f"{pdb_id!r} is not a four-character PDB identifier; refusing "
                f"to build a URL from it")
        expected = expected_sha256.strip().lower().removeprefix("sha256:") \
            if expected_sha256 else None
        directory = Path(destination)
        target = directory / f"{pid}.cif"
        if target.is_file() and not refresh:
            body = target.read_bytes()
            digest = hashlib.sha256(body).hexdigest()
            if expected is not None and digest != expected:
                raise ChecksumMismatchError(
                    f"{target} has sha256 {digest} but {expected} was expected. "
                    f"It was left in place; delete it, or pass refresh=True, to "
                    f"fetch it again")
            validate_mmcif_bytes(body, pid)
            return StructureFile(pid, target, "", len(body), digest, None, True)

        if not self.access.allow_network:
            raise NetworkDisabledError(
                f"{self.source_id}: {pid}.cif is not at {directory} and "
                f"allow_network is false, so it cannot be fetched")
        url = self._file_route_base() + RCSB_STRUCTURE_FILE_PATH.format(pdb_id=pid)
        body = self._http_bytes(url, max_bytes=max_bytes)
        if body is None:
            raise LayerSemanticsError(
                f"the RCSB file service has no coordinate file for {pid}")
        validate_mmcif_bytes(body, pid)
        digest = hashlib.sha256(body).hexdigest()
        if expected is not None and digest != expected:
            raise ChecksumMismatchError(
                f"{pid}.cif downloaded as sha256 {digest} but {expected} was "
                f"expected: the RCSB is serving a different revision of the "
                f"entry than the one this analysis was pinned to. The file was "
                f"not written.")
        directory.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.part")
        temporary.write_bytes(body)
        os.replace(temporary, target)
        return StructureFile(pid, target, url, len(body), digest, utc_now(), False)

    def entry(self, pdb_id: str) -> PDBEntryRecord | None:
        response = self.fetch(pdb_id)
        if not response.ok:
            return None
        rows = self.records(response)
        if len(rows) != 1:
            return None
        row = rows[0]
        ligands = tuple(
            ModelledLigand(
                component_id=str(item.get("component_id", "")).strip().upper(),
                chain=_text(item.get("chain")),
                author_seq_id=_int(item.get("author_seq_id")),
                occupancy=_float(item.get("occupancy")))
            for item in (row.get("ligands") or [])
            if isinstance(item, Mapping)
            and str(item.get("component_id", "")).strip())
        notes = [
            "the deposited construct is not the wild-type sequence: tags, "
            "truncations and surface mutations are routine, so this entry is "
            "not a sequence record",
            "coordinates are not activity",
        ]
        if any(l.occupancy is None for l in ligands):
            notes.append(
                "at least one ligand has no recorded occupancy; its presence in "
                "the file is not proof it was present at full occupancy")
        unknown_state = [l.component_id for l in ligands
                         if l.cofactor_state is CofactorState.UNKNOWN]
        if unknown_state:
            notes.append(
                f"oxidation state is not determined by the component id for "
                f"{', '.join(sorted(set(unknown_state)))}; it must be read from "
                f"the entry rather than assumed")
        sequences = {str(k): str(v) for k, v in (row.get("sequences") or {}).items()}
        return PDBEntryRecord(
            pdb_id=_text(row.get("pdb_id")) or pdb_id,
            method=_text(row.get("method")),
            resolution_angstrom=_float(row.get("resolution")),
            chains=_strings(row.get("chains")),
            ligands=ligands,
            deposited_sequences=sequences,
            response=response,
            notes=tuple(notes))


# ---------------------------------------------------------------------------
# AlphaFold DB
# ---------------------------------------------------------------------------

class PredictionNotObservationError(LayerSemanticsError):
    """A predicted model was asked to stand in for an experimental structure.

    The failure: a predicted apo backbone is scored, reported and compared
    alongside crystal structures, and a reader cannot tell which pocket
    measurements rest on observed density and which on a model's guess at a
    side-chain rotamer.
    """


@dataclass(frozen=True)
class PredictedModelRecord:
    """An AlphaFold model, with per-residue confidence and no ligands.

    ``per_residue_plddt`` is kept per residue rather than averaged. A whole
    chain mean of 90 hides a disordered active-site loop at 45, and the loop is
    exactly where the substrate goes.
    """

    accession: str
    model_id: str
    model_version: str | None
    per_residue_plddt: tuple[float, ...]
    response: CachedResponse
    notes: tuple[str, ...] = ()

    @property
    def ligands(self) -> tuple[ModelledLigand, ...]:
        """Always empty: a predicted model carries no ligand and no cofactor."""
        return ()

    def plddt_at(self, index: int) -> float | None:
        """Confidence at a 0-based residue index, or ``None`` when absent.

        ``None`` rather than a default: a missing confidence is not a low one,
        and a default would make an unmodelled region look merely uncertain.
        """
        if 0 <= index < len(self.per_residue_plddt):
            return self.per_residue_plddt[index]
        return None

    def pocket_plddt(self, indices: Iterable[int]) -> float | None:
        """Mean confidence over named residues, or ``None`` if any is missing.

        Refuses to average over a partial set: a pocket confidence computed
        from the residues that happen to be present describes a different
        pocket from the one that was asked about.
        """
        values = [self.plddt_at(i) for i in indices]
        if not values or any(v is None for v in values):
            return None
        return sum(float(v) for v in values) / len(values)

    def as_observed_structure(self) -> None:
        """Always refuses. See :class:`PredictionNotObservationError`."""
        raise PredictionNotObservationError(
            f"AlphaFold model {self.model_id} for {self.accession} is a "
            f"prediction. It carries no ligand, no cofactor and no occupancy, "
            f"and per-residue confidence is not accuracy for the side-chain "
            f"rotamers that decide whether a pocket accepts a substrate. The "
            f"registry caps this source at computational_construct.")

    def to_dict(self) -> dict[str, Any]:
        return {"accession": self.accession, "model_id": self.model_id,
                "model_version": self.model_version,
                "n_residues_with_plddt": len(self.per_residue_plddt),
                "notes": list(self.notes)}


class AlphaFoldDBConnector(RegistryBackedConnector):
    """Predicted models, returned as predictions.

    Reads the payload shape::

        {"records": [{"accession": "...", "model_id": "AF-...",
                      "model_version": "...", "plddt": [90.1, 88.4, ...]}]}

    Every capability on this source is registered ``unknown`` or
    ``not_supported``, so every call is marked unverified: nobody here has
    established a route to it.
    """

    source_id = "alphafold_db"
    data_layer = ConnectorLayer.STRUCTURE
    description = "AlphaFold DB: predicted protein structures"

    def model(self, accession: str) -> PredictedModelRecord | None:
        response = self.fetch(accession)
        if not response.ok:
            return None
        rows = self.records(response)
        if len(rows) != 1:
            return None
        row = rows[0]
        plddt = tuple(float(v) for v in (row.get("plddt") or [])
                      if isinstance(v, (int, float)) and not isinstance(v, bool))
        notes = [
            "a prediction, never an observation; it must not be recorded with "
            "the same standing as a deposited structure",
            "no ligand and no cofactor: an active site taken from this model is "
            "an apo guess about a holo system",
        ]
        if not plddt:
            notes.append(
                "no per-residue confidence in the cached record, so there is no "
                "way to localise where this model may not be used")
        return PredictedModelRecord(
            accession=_text(row.get("accession")) or accession,
            model_id=_text(row.get("model_id")) or accession,
            model_version=_text(row.get("model_version")),
            per_residue_plddt=plddt, response=response, notes=tuple(notes))


# ---------------------------------------------------------------------------
# SIFTS
# ---------------------------------------------------------------------------

class NumberingOffsetRefusedError(LayerSemanticsError):
    """A residue-level mapping was asked for a single numeric offset.

    There is no such number in general. Constructs are truncated and tagged,
    chains are renumbered by the depositor, and insertion codes make author
    numbering non-contiguous. An offset derived from one aligned pair and
    applied to the rest points at a neighbour of the intended residue, and a
    neighbour mutation is a perfectly plausible-looking construct that tests
    the wrong hypothesis.
    """


@dataclass(frozen=True)
class SiftsResiduePair:
    """One UniProt residue mapped to one author-numbered structure residue."""

    uniprot_accession: str
    uniprot_position: int
    pdb_id: str
    chain_id: str
    author_seq_id: int
    insertion_code: str = ""
    pdb_residue_name: str | None = None
    uniprot_residue: str | None = None

    @property
    def author_position(self) -> AuthorPosition:
        """The structure-side position as the numbering code consumes it.

        Returns an :class:`~eagent.science.numbering.AuthorPosition`, whose
        insertion code is part of its identity: ``100`` and ``100A`` are
        different residues and a map keyed on the integer alone collides them.
        """
        return AuthorPosition(self.chain_id, self.author_seq_id,
                              self.insertion_code)

    def to_dict(self) -> dict[str, Any]:
        return {"uniprot_accession": self.uniprot_accession,
                "uniprot_position": self.uniprot_position,
                "pdb_id": self.pdb_id, "chain_id": self.chain_id,
                "author_seq_id": self.author_seq_id,
                "insertion_code": self.insertion_code,
                "pdb_residue_name": self.pdb_residue_name,
                "uniprot_residue": self.uniprot_residue}


@dataclass(frozen=True)
class SiftsResidueMapping:
    """A residue-level UniProt-to-PDB map, with no offset shortcut.

    This is the object the numbering code consumes: ``uniprot_to_author`` and
    ``author_to_uniprot`` are explicit per-residue dictionaries, and a position
    absent from them is reported absent rather than extrapolated. An
    unmapped UniProt position means "this residue is not resolved in this
    structure", which is a fact worth reporting and not a gap to interpolate
    across.
    """

    uniprot_accession: str
    pdb_id: str
    chain_id: str
    pairs: tuple[SiftsResiduePair, ...]
    response: CachedResponse
    notes: tuple[str, ...] = ()

    @property
    def uniprot_to_author(self) -> dict[int, AuthorPosition]:
        """Per-residue map, UniProt position to author position."""
        return {p.uniprot_position: p.author_position for p in self.pairs}

    @property
    def author_to_uniprot(self) -> dict[AuthorPosition, int]:
        """Per-residue map, author position to UniProt position."""
        return {p.author_position: p.uniprot_position for p in self.pairs}

    def author_position_for(self, uniprot_position: int) -> AuthorPosition | None:
        """The author position for a UniProt position, or ``None``."""
        return self.uniprot_to_author.get(int(uniprot_position))

    def uniprot_position_for(self, author_seq_id: int, *,
                             insertion_code: str = "",
                             chain_id: str | None = None) -> int | None:
        """The UniProt position for an author position, or ``None``."""
        key = AuthorPosition(chain_id or self.chain_id, int(author_seq_id),
                             insertion_code)
        return self.author_to_uniprot.get(key)

    def require_author_position(self, uniprot_position: int) -> AuthorPosition:
        """The author position, or a refusal naming the unresolved residue."""
        got = self.author_position_for(uniprot_position)
        if got is None:
            raise LayerSemanticsError(
                f"UniProt {self.uniprot_accession} position {uniprot_position} "
                f"has no mapped residue in {self.pdb_id} chain {self.chain_id}: "
                f"it is not resolved in this structure. No position was "
                f"interpolated for it.")
        return got

    def numbering_offset(self) -> int:
        """Always refuses. See :class:`NumberingOffsetRefusedError`."""
        raise NumberingOffsetRefusedError(
            f"the SIFTS map for {self.uniprot_accession} / {self.pdb_id} chain "
            f"{self.chain_id} holds {len(self.pairs)} residue pairs and no "
            f"single offset. Use author_position_for() per residue; a constant "
            f"derived from one pair is wrong wherever the construct is "
            f"truncated, renumbered or carries an insertion code.")

    @property
    def is_contiguous(self) -> bool:
        """Whether author numbering happens to advance in step with UniProt.

        Exposed only so a report can say "this one happened to be simple". It
        is never used to take a shortcut: the per-residue map costs nothing and
        is right in the cases where the shortcut is wrong.
        """
        ordered = sorted(self.pairs, key=lambda p: p.uniprot_position)
        if len(ordered) < 2:
            return True
        deltas = {p.author_seq_id - p.uniprot_position for p in ordered}
        return len(deltas) == 1 and not any(p.insertion_code.strip()
                                            for p in ordered)

    def unmapped_uniprot_positions(self, first: int, last: int) -> tuple[int, ...]:
        """UniProt positions in a range that this structure does not resolve."""
        mapped = self.uniprot_to_author
        return tuple(i for i in range(int(first), int(last) + 1)
                     if i not in mapped)

    def to_dict(self) -> dict[str, Any]:
        return {"uniprot_accession": self.uniprot_accession,
                "pdb_id": self.pdb_id, "chain_id": self.chain_id,
                "n_pairs": len(self.pairs),
                "is_contiguous": self.is_contiguous,
                "pairs": [p.to_dict() for p in self.pairs],
                "notes": list(self.notes)}


class SIFTSConnector(RegistryBackedConnector):
    """Residue-level UniProt-to-PDB mappings.

    Reads the payload shape::

        {"records": [{"uniprot_accession": "...", "pdb_id": "0XYZ",
                      "chain_id": "A", "residues": [
                          {"uniprot_position": 1, "author_seq_id": 5,
                           "insertion_code": "", "pdb_residue_name": "MET",
                           "uniprot_residue": "M"}]}]}

    ``keyword_query`` is registered ``not_supported`` for this source, so a
    free-text search here is a typed refusal rather than an attempt: the
    mapping is looked up by the accession-and-entry pair it was built for.
    """

    source_id = "sifts"
    data_layer = ConnectorLayer.STRUCTURE
    description = "SIFTS: residue-level structure-to-sequence mappings"

    def mapping(self, uniprot_accession: str, pdb_id: str,
                chain_id: str) -> SiftsResidueMapping | None:
        """The residue map for one accession/entry/chain triple, or ``None``.

        Keyed on all three because a mapping is only valid for the pair it was
        built from. Reusing a chain A mapping for chain B, or a mapping built
        against an earlier release, is how residue numbers drift by a few
        positions with nothing to show for it.
        """
        key = f"{uniprot_accession}:{pdb_id}:{chain_id}"
        response = self.fetch(key)
        if not response.ok:
            return None
        rows = self.records(response)
        if len(rows) != 1:
            return None
        row = rows[0]
        accession = _text(row.get("uniprot_accession")) or uniprot_accession
        entry = _text(row.get("pdb_id")) or pdb_id
        chain = _text(row.get("chain_id")) or chain_id
        pairs: list[SiftsResiduePair] = []
        for item in (row.get("residues") or []):
            if not isinstance(item, Mapping):
                continue
            up = _int(item.get("uniprot_position"))
            author = _int(item.get("author_seq_id"))
            if up is None or author is None:
                continue
            pairs.append(SiftsResiduePair(
                uniprot_accession=accession, uniprot_position=up,
                pdb_id=entry, chain_id=chain, author_seq_id=author,
                insertion_code=str(item.get("insertion_code") or "").strip(),
                pdb_residue_name=_text(item.get("pdb_residue_name")),
                uniprot_residue=_text(item.get("uniprot_residue"))))
        return SiftsResidueMapping(
            uniprot_accession=accession, pdb_id=entry, chain_id=chain,
            pairs=tuple(pairs), response=response,
            notes=(
                "a mapping is valid only for the sequence release and the "
                "structure release it was built from; record all three together",
                "positions absent from this map are unresolved in the "
                "structure, not shifted by an offset",
            ))


# ---------------------------------------------------------------------------
# wwPDB Chemical Component Dictionary
# ---------------------------------------------------------------------------

class AtomOrderMatchRefusedError(LayerSemanticsError):
    """Ligand atoms were asked to be matched by their order in a file.

    Two depositions of one component can list its atoms in any order, and a
    file can omit atoms entirely. An order-based match therefore pairs the
    nicotinamide C4 with whatever happens to sit at that index, and every
    hydride-transfer distance computed afterwards is a distance to the wrong
    atom -- with no symptom other than a number that looks plausible.
    """


@dataclass(frozen=True)
class CCDAtom:
    """One atom of a chemical component, addressed by its name.

    The name is the identity. ``element`` is carried for checks, but two atoms
    of a component share an element routinely and only the name distinguishes
    them.
    """

    atom_id: str
    element: str
    charge: int | None = None
    aromatic: bool | None = None
    leaving_atom: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"atom_id": self.atom_id, "element": self.element,
                "charge": self.charge, "aromatic": self.aromatic,
                "leaving_atom": self.leaving_atom}


@dataclass(frozen=True)
class CCDBond:
    """One bond, named by its two atom ids rather than by two indices."""

    atom_id_1: str
    atom_id_2: str
    order: str | None = None
    aromatic: bool | None = None

    @property
    def atoms(self) -> frozenset[str]:
        return frozenset({self.atom_id_1, self.atom_id_2})

    def to_dict(self) -> dict[str, Any]:
        return {"atom_id_1": self.atom_id_1, "atom_id_2": self.atom_id_2,
                "order": self.order, "aromatic": self.aromatic}


@dataclass(frozen=True)
class CCDAtomMatch:
    """The result of matching a file's atoms against a dictionary component.

    ``missing`` and ``unexpected`` are separate: a component atom absent from
    the file is usually an unmodelled part of the ligand, while a file atom
    absent from the component is usually the wrong component id. The two call
    for different actions, and a single "did not match" count hides both.
    """

    component_id: str
    matched: Mapping[str, CCDAtom]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing and not self.unexpected

    def require_atom(self, atom_id: str) -> CCDAtom:
        """One matched atom, or a refusal naming it.

        Refuses rather than returning the nearest atom: a functional-atom role
        such as the nicotinamide C4 is defined on one named atom, and
        substituting another produces a geometric criterion about a different
        bond.
        """
        got = self.matched.get(atom_id)
        if got is None:
            raise LayerSemanticsError(
                f"atom '{atom_id}' of component {self.component_id} was not "
                f"found in the file by name (missing: "
                f"{', '.join(self.missing) or 'none'}). No substitute atom was "
                f"chosen for it.")
        return got

    def to_dict(self) -> dict[str, Any]:
        return {"component_id": self.component_id,
                "matched": sorted(self.matched),
                "missing": list(self.missing),
                "unexpected": list(self.unexpected),
                "complete": self.complete}


@dataclass(frozen=True)
class ChemicalComponent:
    """A dictionary entry: the atoms and bonds that define a ligand's identity."""

    component_id: str
    name: str | None
    formula: str | None
    atoms: tuple[CCDAtom, ...]
    bonds: tuple[CCDBond, ...]
    response: CachedResponse
    notes: tuple[str, ...] = ()

    @property
    def atoms_by_name(self) -> dict[str, CCDAtom]:
        """Atoms keyed by name. The only index this class offers."""
        return {a.atom_id: a for a in self.atoms}

    def atom(self, atom_id: str) -> CCDAtom | None:
        return self.atoms_by_name.get(str(atom_id).strip())

    def bonded_to(self, atom_id: str) -> tuple[str, ...]:
        """Names of the atoms bonded to one atom, by name throughout."""
        name = str(atom_id).strip()
        out: list[str] = []
        for bond in self.bonds:
            if bond.atom_id_1 == name:
                out.append(bond.atom_id_2)
            elif bond.atom_id_2 == name:
                out.append(bond.atom_id_1)
        return tuple(out)

    def match_atoms(self, file_atoms: Sequence[Any], *,
                    component_id: str | None = None) -> CCDAtomMatch:
        """Match a file's atoms to this component by id and name.

        ``file_atoms`` may be mappings with ``name``/``atom_id`` keys, plain
        strings, or objects with a ``name`` attribute (such as
        :class:`~eagent.science.structure_io.Atom`). The order they arrive in
        is irrelevant by construction: matching is a dictionary lookup on the
        name, so two files listing the same ligand's atoms in different orders
        produce the same match.

        ``component_id`` guards the other half of the identity: matching the
        atom names of NAP against the component NAI would pair up most names
        happily and give an oxidation state nobody intended.
        """
        if component_id is not None:
            want = str(component_id).strip().upper()
            if want != self.component_id.strip().upper():
                raise LayerSemanticsError(
                    f"refusing to match atoms of component '{want}' against "
                    f"dictionary entry '{self.component_id}': atoms are matched "
                    f"by component id and atom name together, and two "
                    f"nicotinamide components share most of their atom names "
                    f"while differing in oxidation state")
        names = [_atom_name(a) for a in file_atoms]
        present = [n for n in names if n]
        index = self.atoms_by_name
        matched = {n: index[n] for n in present if n in index}
        missing = tuple(sorted(set(index) - set(matched)))
        unexpected = tuple(sorted(set(present) - set(index)))
        return CCDAtomMatch(self.component_id, matched, missing, unexpected)

    def match_by_order(self, file_atoms: Sequence[Any]) -> None:
        """Always refuses. See :class:`AtomOrderMatchRefusedError`."""
        raise AtomOrderMatchRefusedError(
            f"component {self.component_id} has {len(self.atoms)} atoms and the "
            f"file offers {len(file_atoms)}; matching them by position would "
            f"pair atoms whose only relationship is their index. Use "
            f"match_atoms(), which matches by component id and atom name.")

    def to_dict(self) -> dict[str, Any]:
        return {"component_id": self.component_id, "name": self.name,
                "formula": self.formula,
                "atoms": [a.to_dict() for a in self.atoms],
                "bonds": [b.to_dict() for b in self.bonds],
                "notes": list(self.notes)}


def _atom_name(item: Any) -> str | None:
    """Read an atom name from a mapping, a string or a structure_io Atom."""
    if isinstance(item, str):
        return item.strip() or None
    if isinstance(item, Mapping):
        for key in ("atom_id", "name", "atom_name"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
    value = getattr(item, "name", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


class WwPDBChemicalComponentConnector(RegistryBackedConnector):
    """Ligand chemical identity: atom names, elements and bonds.

    Reads the payload shape::

        {"records": [{"component_id": "NAP", "name": "...", "formula": "...",
                      "atoms": [{"atom_id": "C4N", "element": "C"}],
                      "bonds": [{"atom_id_1": "C4N", "atom_id_2": "C3N",
                                 "order": "SING"}]}]}

    The registry's curation note requires recording the dictionary version used,
    "because atom naming changes are silent breakages for every stored
    geometric criterion". A component retrieved without a version comes back
    saying so.
    """

    source_id = "wwpdb_ccd"
    data_layer = ConnectorLayer.STRUCTURE
    description = "wwPDB Chemical Component Dictionary: ligand definitions"

    def component(self, component_id: str) -> ChemicalComponent | None:
        response = self.fetch(component_id.strip().upper())
        if not response.ok:
            return None
        rows = self.records(response)
        if len(rows) != 1:
            return None
        row = rows[0]
        atoms = tuple(
            CCDAtom(atom_id=str(item.get("atom_id", "")).strip(),
                    element=str(item.get("element", "")).strip().upper(),
                    charge=_int(item.get("charge")),
                    aromatic=_bool(item.get("aromatic")),
                    leaving_atom=_bool(item.get("leaving_atom")))
            for item in (row.get("atoms") or [])
            if isinstance(item, Mapping) and str(item.get("atom_id", "")).strip())
        bonds = tuple(
            CCDBond(atom_id_1=str(item.get("atom_id_1", "")).strip(),
                    atom_id_2=str(item.get("atom_id_2", "")).strip(),
                    order=_text(item.get("order")),
                    aromatic=_bool(item.get("aromatic")))
            for item in (row.get("bonds") or [])
            if isinstance(item, Mapping)
            and str(item.get("atom_id_1", "")).strip()
            and str(item.get("atom_id_2", "")).strip())
        notes = [
            "atoms are matched by component id and atom name, never by their "
            "order in a file",
        ]
        if not response.database_version:
            notes.append(
                "no dictionary release is recorded for this component; atom "
                "naming changes between releases, and an unpinned component "
                "silently breaks every geometric criterion stored against it")
        duplicates = sorted({a.atom_id for a in atoms
                             if [x.atom_id for x in atoms].count(a.atom_id) > 1})
        if duplicates:
            notes.append(
                f"the cached component repeats atom name(s) "
                f"{', '.join(duplicates)}; a repeated name cannot identify an "
                f"atom and the match will be ambiguous")
        return ChemicalComponent(
            component_id=_text(row.get("component_id")) or component_id.upper(),
            name=_text(row.get("name")), formula=_text(row.get("formula")),
            atoms=atoms, bonds=bonds, response=response, notes=tuple(notes))


# ---------------------------------------------------------------------------
# M-CSA
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CatalyticResidueRecord:
    """One catalytic residue with its role and where that role came from."""

    residue_name: str
    role: str
    uniprot_accession: str | None = None
    uniprot_position: int | None = None
    pdb_id: str | None = None
    chain_id: str | None = None
    author_seq_id: int | None = None
    mechanism_step: str | None = None
    source_reference: str | None = None

    @property
    def author_position(self) -> AuthorPosition | None:
        """The structure-side position, or ``None`` when the record has none."""
        if self.chain_id is None or self.author_seq_id is None:
            return None
        return AuthorPosition(self.chain_id, self.author_seq_id, "")

    def as_template_entry(self) -> dict[str, Any]:
        """The ``{label, residue_types, role, functional_atoms, evidence}`` shape.

        Built to the shape
        :attr:`~eagent.schemas.templates.CatalyticTemplate.catalytic_residues`
        documents, with ``evidence`` populated from the record's own source.
        An entry whose evidence is empty would make the template unsourced,
        which the template loader rejects -- correctly, since an unsourced
        catalytic residue is a guess with a template's authority.
        """
        label = (f"{self.residue_name}{self.author_seq_id}"
                 if self.author_seq_id is not None
                 else f"{self.residue_name}{self.uniprot_position or ''}")
        return {
            "label": label,
            "residue_types": [self.residue_name],
            "role": self.role,
            "functional_atoms": [],
            "evidence": self.source_reference or "",
        }

    def to_dict(self) -> dict[str, Any]:
        return {"residue_name": self.residue_name, "role": self.role,
                "uniprot_accession": self.uniprot_accession,
                "uniprot_position": self.uniprot_position,
                "pdb_id": self.pdb_id, "chain_id": self.chain_id,
                "author_seq_id": self.author_seq_id,
                "mechanism_step": self.mechanism_step,
                "source_reference": self.source_reference}


@dataclass(frozen=True)
class MCSAEntry:
    """A mechanism entry: which residues do the chemistry in a reference enzyme.

    The reference enzyme is part of the record, not an aside. An M-CSA entry is
    evidenced for the enzyme it was curated on, so transferring its roles to a
    homologue is an inference that needs an explicit residue mapping -- which is
    what :class:`SiftsResidueMapping` and the numbering code are for.
    """

    mcsa_id: str
    reference_uniprot: str | None
    reference_pdb: str | None
    ec_number: str | None
    residues: tuple[CatalyticResidueRecord, ...]
    literature_ids: tuple[str, ...]
    response: CachedResponse
    notes: tuple[str, ...] = ()

    def roles(self) -> dict[str, list[str]]:
        """Role label to the residue labels assigned to it."""
        out: dict[str, list[str]] = {}
        for residue in self.residues:
            out.setdefault(residue.role, []).append(
                residue.as_template_entry()["label"])
        return out

    def to_template_provenance(self) -> TemplateProvenance:
        """Provenance for a catalytic template built from this entry.

        Refuses when the entry cites nothing. An
        :attr:`~eagent.schemas.templates.TemplateSourceType.UNSOURCED` template
        is rejected by the loader, so producing one here would merely move the
        failure; refusing names the cause instead.
        """
        identifiers = [i for i in
                       (self.mcsa_id, self.reference_pdb, self.reference_uniprot)
                       if i] + list(self.literature_ids)
        if not self.literature_ids and not self.reference_pdb:
            raise LayerSemanticsError(
                f"M-CSA {self.mcsa_id} cites neither a reference structure nor a "
                f"publication in the cached record; a catalytic template built "
                f"from it would be unsourced, which is a guess carrying a "
                f"template's authority")
        return TemplateProvenance(
            source_type=TemplateSourceType.MECHANISM_LITERATURE,
            identifiers=identifiers,
            notes=(f"catalytic residues from M-CSA {self.mcsa_id}, evidenced for "
                   f"the reference enzyme "
                   f"{self.reference_uniprot or self.reference_pdb or 'unstated'}; "
                   f"transfer to a homologue requires an explicit residue mapping"),
        )

    def as_substrate_scope(self) -> None:
        """Always refuses: mechanism is not scope, rate or stereopreference."""
        raise LayerSemanticsError(
            f"M-CSA {self.mcsa_id} describes how the chemistry is done, not "
            f"which substrates are accepted, how fast, or with which "
            f"configuration.")

    def to_dict(self) -> dict[str, Any]:
        return {"mcsa_id": self.mcsa_id,
                "reference_uniprot": self.reference_uniprot,
                "reference_pdb": self.reference_pdb,
                "ec_number": self.ec_number,
                "residues": [r.to_dict() for r in self.residues],
                "literature_ids": list(self.literature_ids),
                "notes": list(self.notes)}


class MCSAConnector(RegistryBackedConnector):
    """Catalytic residues with their roles and their sources.

    Reads the payload shape::

        {"records": [{"mcsa_id": "...", "reference_uniprot": "...",
                      "reference_pdb": "0XYZ", "ec": "1.1.1.1",
                      "literature_ids": ["PMID:..."],
                      "residues": [{"residue_name": "TYR", "role": "acid/base",
                                    "uniprot_position": 155,
                                    "chain_id": "A", "author_seq_id": 155,
                                    "mechanism_step": "...",
                                    "source_reference": "PMID:..."}]}]}
    """

    source_id = "mcsa"
    data_layer = ConnectorLayer.MECHANISM
    description = "M-CSA: catalytic residues, roles and mechanisms"

    def entry(self, mcsa_id: str) -> MCSAEntry | None:
        response = self.fetch(mcsa_id)
        if not response.ok:
            return None
        rows = self.records(response)
        if len(rows) != 1:
            return None
        row = rows[0]
        residues = tuple(
            CatalyticResidueRecord(
                residue_name=str(item.get("residue_name", "")).strip().upper(),
                role=_text(item.get("role")) or "unstated",
                uniprot_accession=_text(item.get("uniprot_accession"))
                or _text(row.get("reference_uniprot")),
                uniprot_position=_int(item.get("uniprot_position")),
                pdb_id=_text(item.get("pdb_id")) or _text(row.get("reference_pdb")),
                chain_id=_text(item.get("chain_id")),
                author_seq_id=_int(item.get("author_seq_id")),
                mechanism_step=_text(item.get("mechanism_step")),
                source_reference=_text(item.get("source_reference")))
            for item in (row.get("residues") or [])
            if isinstance(item, Mapping)
            and str(item.get("residue_name", "")).strip())
        notes = [
            "evidenced for the reference enzyme; transferring a role to a "
            "homologue is an inference needing an explicit residue mapping",
            "coverage is partial: no entry is not evidence that the catalytic "
            "residues are unknown",
        ]
        unsourced = [r for r in residues if not r.source_reference]
        if unsourced:
            notes.append(
                f"{len(unsourced)} residue(s) carry no source reference; a "
                f"template built from them would be unsourced in part")
        return MCSAEntry(
            mcsa_id=_text(row.get("mcsa_id")) or mcsa_id,
            reference_uniprot=_text(row.get("reference_uniprot")),
            reference_pdb=_text(row.get("reference_pdb")),
            ec_number=_text(row.get("ec")),
            residues=residues,
            literature_ids=_strings(row.get("literature_ids")),
            response=response, notes=tuple(notes))


# ---------------------------------------------------------------------------
# AlphaFill
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransplantedLigand:
    """A ligand placed by homology transfer. Tagged as such, permanently.

    :attr:`source` is a property with no setter and no constructor argument, so
    there is no code path -- not even a keyword argument typo -- that produces a
    transplanted ligand labelled as observed. The failure that prevents: a
    cofactor copied in from a homologue being reported, scored and eventually
    published as a cofactor seen in this enzyme.
    """

    component_id: str
    donor_pdb_id: str | None
    donor_chain: str | None
    target_accession: str
    global_rmsd: float | None = None
    local_rmsd: float | None = None
    sequence_identity: float | None = None
    transplant_id: str | None = None

    @property
    def source(self) -> LigandSource:
        """Always :attr:`LigandSource.HOMOLOGY_TRANSPLANTED`."""
        return LigandSource.HOMOLOGY_TRANSPLANTED

    @property
    def cofactor_state(self) -> CofactorState:
        """Oxidation state from the component id, or ``UNKNOWN``."""
        return cofactor_state_from_ligand_code(self.component_id)

    def to_cofactor_spec(self, *, name: str | None = None) -> CofactorSpec:
        """A cofactor spec carrying the transplant tag and the donor.

        ``evidence`` names the donor structure because "where did this cofactor
        come from" is the first question anybody asks of a filled model, and the
        answer has to survive being copied into a template.
        """
        return CofactorSpec(
            name=name or self.component_id,
            state=self.cofactor_state,
            ligand_code=self.component_id,
            source=LigandSource.HOMOLOGY_TRANSPLANTED,
            evidence=(f"transplanted by AlphaFill from "
                      f"{self.donor_pdb_id or 'an unstated donor'}"
                      f"{('/' + self.donor_chain) if self.donor_chain else ''} "
                      f"onto {self.target_accession}; a hypothesis about "
                      f"placement, not an observation of this enzyme"),
        )

    def passes(self, *, max_local_rmsd: float | None,
               min_sequence_identity: float | None) -> bool:
        """Whether this transplant clears the thresholds the caller set.

        Returns ``False`` when a threshold is set and the metric is missing.
        The registry requires the rejection threshold to be set explicitly
        "rather than accepting every transplant", and a missing metric cannot
        clear a threshold it was never compared against.
        """
        if max_local_rmsd is not None:
            if self.local_rmsd is None or self.local_rmsd > max_local_rmsd:
                return False
        if min_sequence_identity is not None:
            if (self.sequence_identity is None
                    or self.sequence_identity < min_sequence_identity):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"component_id": self.component_id,
                "donor_pdb_id": self.donor_pdb_id,
                "donor_chain": self.donor_chain,
                "target_accession": self.target_accession,
                "global_rmsd": self.global_rmsd, "local_rmsd": self.local_rmsd,
                "sequence_identity": self.sequence_identity,
                "transplant_id": self.transplant_id,
                "source": self.source.value,
                "cofactor_state": self.cofactor_state.value}


class AlphaFillConnector(RegistryBackedConnector):
    """AlphaFill transplants: ligands moved onto a predicted model by homology.

    Reads the payload shape::

        {"records": [{"transplant_id": "...", "component_id": "NAP",
                      "donor_pdb_id": "0XYZ", "donor_chain": "A",
                      "global_rmsd": 0.0, "local_rmsd": 0.0,
                      "sequence_identity": 0.0}]}

    Derived from AlphaFold DB and the PDB, so a transplant and the donor entry
    are not independent evidence of anything. Every returned ligand is tagged
    :attr:`~eagent.schemas.chem.LigandSource.HOMOLOGY_TRANSPLANTED`.
    """

    source_id = "alphafill"
    data_layer = ConnectorLayer.STRUCTURE
    description = "AlphaFill: homology-transplanted ligands and cofactors"

    def transplants(self, accession: str) -> tuple[TransplantedLigand, ...]:
        """Cached transplants for one accession, every one tagged as such."""
        response = self.fetch(accession)
        if not response.ok:
            return ()
        out: list[TransplantedLigand] = []
        for row in self.records(response):
            component = str(row.get("component_id", "")).strip().upper()
            if not component:
                continue
            out.append(TransplantedLigand(
                component_id=component,
                donor_pdb_id=_text(row.get("donor_pdb_id")),
                donor_chain=_text(row.get("donor_chain")),
                target_accession=_text(row.get("accession")) or accession,
                global_rmsd=_float(row.get("global_rmsd")),
                local_rmsd=_float(row.get("local_rmsd")),
                sequence_identity=_float(row.get("sequence_identity")),
                transplant_id=_text(row.get("transplant_id"))))
        return tuple(out)

    def accepted_transplants(self, accession: str, *,
                             max_local_rmsd: float | None,
                             min_sequence_identity: float | None
                             ) -> tuple[TransplantedLigand, ...]:
        """Transplants clearing explicitly stated thresholds.

        Both thresholds are required keyword arguments with no defaults. A
        default here would be a project-wide acceptance criterion chosen by
        whoever wrote this line, which is precisely what the registry's
        curation note says must be set explicitly instead.
        """
        return tuple(t for t in self.transplants(accession)
                     if t.passes(max_local_rmsd=max_local_rmsd,
                                 min_sequence_identity=min_sequence_identity))


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
    """An int, or ``None``. A residue number that defaulted would point at a
    real residue, just not the intended one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool | None:
    """Tri-state: ``True``, ``False``, or ``None`` when the record is silent."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("y", "yes", "true", "1"):
            return True
        if token in ("n", "no", "false", "0"):
            return False
    return None
