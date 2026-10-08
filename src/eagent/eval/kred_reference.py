"""The KRED calibration reference set v0.1: loaded, cross-checked, and fenced.

WHAT THIS IS
============
A delivered table set (``configs/references/kred_calibration/v0.1``): 19 PDB
entries audited for use as experimental enzyme--substrate complexes, 28
kinetic records that were matched to them, ten enantioselectivity records, the
ligand-level density statistics behind the structure grades, and the source
list with each source's reuse terms.

It was compiled by an AI research assistant from public sources and handed to
this project as a spreadsheet. This module is the part that does not take that
on trust, and it is deliberately more suspicious of the set than the set is of
itself:

* **the files are pinned.** ``MANIFEST.json`` holds the sha256 of every file,
  and loading re-hashes them. The CSVs are a deterministic conversion of the
  stored workbook, and :func:`verify_conversion` re-runs that conversion;
* **every number is recomputed.** The workbook's SI-unit columns are formula
  cells whose cached results are read as typed. Here ``kcat`` in ``min^-1``,
  ``Km`` in ``uM`` and a catalytic efficiency in ``mM^-1 min^-1`` are converted
  again through :mod:`eagent.science.units` and compared with what the
  spreadsheet says. A disagreement is an error, because the alternative is two
  conversion tables that nobody compared;
* **every cross-reference is followed in both directions.** A structure that
  names a kinetic label must be named back by it; a selectivity record that
  claims its variant matches a kinetic record must actually match it; the
  density summary on a structure must equal the validation row it was taken
  from;
* **what a number may be used as is recorded with the number.** ``ND`` is not
  zero; a ``<`` bound is not a point; a reported efficiency and a ``kcat/Km``
  quotient are two fields; ``kcat`` in a steady-state assay, an apparent
  ``kcat`` with an isopropanol regeneration system, a reported efficiency and
  an ``er`` are four label types and are never pooled.

WHAT IT REFUSES TO BE
=====================
It is not a benchmark of the pipeline, not a training set, and not a reason
for a window to gain the authority to reject an enzyme. The set's own audit
says that no entry is an unconditional strict-geometry reference, and
:data:`GEOMETRY_USE` has no class that would let one be. How many *independent*
enzyme lineages the 19 entries and 28 records come from is computed, not
assumed, by :meth:`ReferenceSet.independence`; it is far smaller than 19 or 28,
and it is the number a tolerance interval is allowed to count.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..errors import EAgentError
from ..science.units import canonical_unit
from .kred_workbook import SHEET_TABLES, read_workbook, write_tables

__all__ = [
    "ReferenceSetError",
    "Finding",
    "REFERENCE_NAME",
    "REFERENCE_VERSION",
    "MANIFEST_NAME",
    "STATED_COUNTS",
    "GEOMETRY_USE",
    "TIERS",
    "LINEAGE_GROUPS",
    "LINEAGE_IDENTITY_THRESHOLD",
    "AUXILIARY_LABELS",
    "default_reference_dir",
    "build_manifest",
    "verify_manifest",
    "write_manifest",
    "verify_conversion",
    "LigandValidation",
    "StructureEntry",
    "KineticRecord",
    "SelectivityRecord",
    "SourceRecord",
    "ReferenceSet",
    "load_reference_set",
]

REFERENCE_NAME = "kred_calibration_reference"
REFERENCE_VERSION = "v0.1"
MANIFEST_NAME = "MANIFEST.json"

#: Directories whose bytes are the data product. Documentation beside them is
#: listed in the manifest too, but is not part of ``data_digest``: a typo fix in
#: the README must not look like a change to the numbers.
DATA_DIRECTORIES = ("source", "tables", "bindings", "coordinates")


class ReferenceSetError(EAgentError):
    """The reference set is not the one this module was written for."""

    def __init__(self, message: str, findings: Sequence["Finding"] = ()) -> None:
        super().__init__(message)
        self.findings = tuple(findings)


@dataclass(frozen=True)
class Finding:
    """One thing the checks noticed. ``error`` findings stop a load."""

    severity: str            # "error" | "warning" | "info"
    code: str
    subject: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code} {self.subject}: {self.message}"


# ==========================================================================
# what the delivery says it contains
# ==========================================================================

#: The counts the delivery report and the usage sheet state. They are
#: recomputed from the tables and compared: a table that no longer holds what
#: its cover note says is a different delivery wearing the old note.
STATED_COUNTS: dict[str, int] = {
    "structures": 19,
    "kinetic_records": 28,
    "numeric_records": 26,
    "not_determined_records": 2,
    "records_with_kcat": 18,
    "efficiency_only_records": 8,
    "selectivity_records": 10,
    "core_records": 22,
    "secondary_records": 3,
    "sensitivity_records": 1,
}

#: ``geometry_use`` -> what the project may do with the entry. There is no
#: "strict gold" class, on purpose: the audit found none, and a value that does
#: not appear here is refused so that a later version cannot add one quietly.
GEOMETRY_USE: dict[str, str] = {
    "conditional_pose_reference": "pose_candidate",
    "conditional_product_reference": "product_state",
    "review_clash": "pose_under_review",
    "low_ligand_density": "excluded_density",
    "auxiliary_ensemble_reference": "auxiliary",
    "protein_scaffold_only": "scaffold_only",
    "unmatched_product_excluded": "excluded",
    "inhibitor_excluded": "excluded",
}

#: ``default_use`` of a kinetic row -> the tier the set's usage notes place it
#: in. ``core`` rows are the ones meant to anchor a calibration; the others are
#: kept apart so that nothing is averaged across them by accident.
TIERS: dict[str, str] = {
    "within_assay_kinetic_reference": "core",
    "kcat_only_Km_bound_quarantined": "core",
    "within_assay_efficiency_ranking": "core",
    "within_assay_apparent_kinetic_ranking": "core",
    "secondary_kinetic_reference": "secondary",
    "sensitivity_only_unsaturated": "sensitivity",
    "ND_not_zero": "nd",
}

#: The workbook's ``enzyme`` strings -> the unit of independence. Entries and
#: records of one group are not replicates of each other: chains, alternate
#: conformers, different PDB entries of one enzyme and point mutants of one
#: parent all share a parent.
#:
#: Two groups here are **merges the workbook does not make**, and both are
#: stated so they can be argued with:
#:
#: * ``HBDH`` joins PaHBDH and AbHBDH: homologues (37 % identical,
#:   coverage-adjusted) characterised by one group in one assay system. A
#:   declared merge, not a measured one;
#: * ``LbADH/LkKRED`` joins *Lactobacillus brevis* ADH and *Lactobacillus
#:   kefir* KRED, which the workbook treats as two enzymes. Their pinned
#:   sequences are 88 % identical over the aligned region (82 % after scaling
#:   by coverage; the difference is mostly LkKRED's 20-residue N-terminal
#:   fusion), far above any conventional same-family threshold, so counting
#:   them as two draws of the enzyme population would double-count one lineage.
#:   This one **is** measured, and
#:   :func:`eagent.eval.kred_coordinates.lineage_findings` re-derives it from the
#:   pinned sequences and fails if a pair at or above the threshold is ever left
#:   in different groups.
LINEAGE_GROUPS: dict[str, str] = {
    "PaHBDH": "HBDH",
    "AbHBDH": "HBDH",
    "SmBdh": "SmBdh",
    "LbADH": "LbADH/LkKRED",
    "LkKRED": "LbADH/LkKRED",
    "TR-II": "TR-II",
    "tropinone reductase II": "TR-II",
    "Ssal-KRED": "Ssal-KRED",
    "TeSADH": "TeSADH",
}

#: Coverage-adjusted global identity at or above which two enzymes are one
#: lineage for the purpose of counting independent samples. 40 % is the
#: conventional same-family clustering level; the audit also reports how the
#: count moves at 30 % and 50 %, because the threshold is a choice.
LINEAGE_IDENTITY_THRESHOLD = 0.40

#: A structure may name a label that lives in the auxiliary sheet instead of
#: the kinetics sheet. Exactly one such link exists and it is listed so that a
#: second one has to be added here, deliberately.
AUXILIARY_LABELS: dict[str, str] = {
    "SMBDH_WT_ACETOIN_2020": "SmBdh / 6XEW / WT",
}

#: ``record_status`` values and what each is allowed to carry.
RECORD_STATUS = ("measured_numeric", "measured_apparent_numeric", "not_determined")

#: Tolerance for comparing a recomputed value with the spreadsheet's cached one.
RELATIVE_TOLERANCE = 1e-9

#: A reported efficiency and a ``kcat/Km`` quotient of the same row differ by
#: the rounding of two published numbers. Beyond this the difference is worth a
#: warning rather than an information line.
EFFICIENCY_DISCREPANCY_WARN = 0.20

#: The two RSCC columns of a structure's summary, and the tolerance with which
#: they must match the ligand-validation rows they were read from.
RSCC_TOLERANCE = 5e-4


def default_reference_dir() -> Path:
    """The shipped copy of the set, beside the other configuration."""
    return (Path(__file__).resolve().parents[3] / "configs" / "references"
            / "kred_calibration" / REFERENCE_VERSION)


# ==========================================================================
# the manifest: what makes the files tamper-evident
# ==========================================================================

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(files: Mapping[str, Mapping[str, Any]]) -> str:
    """One hash over (path, sha256) pairs, independent of file order."""
    lines = "".join(f"{path}\0{info['sha256']}\n"
                    for path, info in sorted(files.items()))
    return _sha256_bytes(lines.encode("utf-8"))


def _data_files(files: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {p: i for p, i in files.items() if p.split("/", 1)[0] in DATA_DIRECTORIES}


def build_manifest(directory: str | Path) -> dict[str, Any]:
    """Hash every file under ``directory`` (but the manifest itself)."""
    base = Path(directory)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = path.relative_to(base).as_posix()
        if rel == MANIFEST_NAME or "__pycache__" in rel.split("/"):
            continue
        data = path.read_bytes()
        files[rel] = {"sha256": _sha256_bytes(data), "bytes": len(data)}
    return {
        "product": REFERENCE_NAME,
        "version": REFERENCE_VERSION,
        "files": files,
        "data_digest": _digest(_data_files(files)),
        "files_digest": _digest(files),
        "note": ("written by `eagent reference manifest`; data_digest covers "
                 + ", ".join(DATA_DIRECTORIES) + " and excludes documentation"),
    }


def write_manifest(directory: str | Path) -> Path:
    base = Path(directory)
    path = base / MANIFEST_NAME
    path.write_text(json.dumps(build_manifest(base), indent=2, sort_keys=True,
                               ensure_ascii=False) + "\n", encoding="utf-8")
    return path


#: Files whose bytes carry the claims: the data directories, and the record of
#: what was compared with the sources. A change to one of these is an error.
#: Documentation beside them (README, NOTICE, .gitattributes) is still hashed,
#: and a change is reported, but as a warning: a typo fix in a README must not
#: stop the set from loading, and a README cannot change a number.
def _is_data_path(rel: str) -> bool:
    return rel.split("/", 1)[0] in DATA_DIRECTORIES or rel == "verification/results.json"


def verify_manifest(directory: str | Path) -> list[Finding]:
    """Compare the files on disk with the manifest. Empty list == intact.

    A changed, missing or unlisted **data** file is an error; the same for a
    documentation file is a warning (see :func:`_is_data_path`).
    """
    base = Path(directory)
    path = base / MANIFEST_NAME
    if not path.is_file():
        return [Finding("error", "manifest.missing", str(base),
                        f"{MANIFEST_NAME} is not there, so nothing about these "
                        f"files can be shown to be what was delivered")]
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [Finding("error", "manifest.unreadable", MANIFEST_NAME, str(exc))]
    findings: list[Finding] = []
    current = build_manifest(base)
    listed = stored.get("files") or {}

    def sev(rel: str) -> str:
        return "error" if _is_data_path(rel) else "warning"

    for rel, info in sorted(listed.items()):
        have = current["files"].get(rel)
        if have is None:
            findings.append(Finding(sev(rel), "manifest.file_missing", rel,
                                    "listed in the manifest but not on disk"))
        elif have["sha256"] != info.get("sha256"):
            findings.append(Finding(
                sev(rel), "manifest.file_changed", rel,
                f"sha256 is {have['sha256'][:16]}..., the manifest says "
                f"{str(info.get('sha256'))[:16]}...; the file was edited after "
                f"the manifest was written"
                + ("" if _is_data_path(rel) else " (documentation: regenerate the "
                   "manifest after reading the change)")))
    for rel in sorted(set(current["files"]) - set(listed)):
        findings.append(Finding(sev(rel), "manifest.file_unlisted", rel,
                                "on disk but not in the manifest"))
    if stored.get("data_digest") != _digest(_data_files(listed)):
        findings.append(Finding(
            "error", "manifest.digest_inconsistent", MANIFEST_NAME,
            "data_digest does not follow from the listed hashes"))
    return findings


def verify_conversion(directory: str | Path) -> list[Finding]:
    """Convert the stored workbook again and compare with the stored tables.

    The tables are the reading copy; the workbook is the delivery. If they
    disagree, either the tables were edited by hand or the converter changed,
    and neither is something to find out later.
    """
    import tempfile

    base = Path(directory)
    workbooks = sorted((base / "source").glob("*.xlsx")) if (base / "source").is_dir() else []
    findings: list[Finding] = []
    if len(workbooks) != 1:
        return [Finding("error", "conversion.no_workbook", str(base / "source"),
                        f"expected exactly one delivered workbook beside its tables, "
                        f"found {len(workbooks)}")]
    source = workbooks[0]
    with tempfile.TemporaryDirectory() as tmp:
        written = write_tables(read_workbook(source), tmp)
        for path in written:
            stored = base / "tables" / path.name
            if not stored.is_file():
                findings.append(Finding("error", "conversion.table_missing",
                                        path.name, "the converter writes it; "
                                        "the stored tables do not have it"))
            elif stored.read_bytes() != path.read_bytes():
                findings.append(Finding(
                    "error", "conversion.table_differs", path.name,
                    "converting the stored workbook gives different bytes from "
                    "the stored table"))
        expected = {p.name for p in written}
        for extra in sorted(p.name for p in (base / "tables").iterdir()
                            if p.is_file() and p.name not in expected):
            findings.append(Finding("error", "conversion.table_extra", extra,
                                    "a stored table the converter does not write"))
    return findings


# ==========================================================================
# typed records
# ==========================================================================

def _split(value: str, sep: str = ";") -> tuple[str, ...]:
    return tuple(p.strip() for p in value.split(sep) if p.strip())


def _clean(value: str | None) -> str:
    return (value or "").strip()


@dataclass(frozen=True)
class LigandValidation:
    """One row of per-residue density statistics for a ligand or cofactor."""

    pdb_id: str
    chain: str
    residue_number: int
    ccd: str
    altloc: str
    rscc: float | None
    rsr: float | None
    occupancy: float | None
    b_factor: float | None
    raw_clash_children: int | None
    geometry_children: int | None
    warning: str
    source_url: str


@dataclass(frozen=True)
class StructureEntry:
    """One audited PDB entry, as the workbook grades it."""

    pdb_id: str
    enzyme: str
    variant: str
    geometry_use: str
    resolution_a: float
    bound_reaction_ligand: str
    experimental_state: str
    scoring_chain: str
    altlocs: tuple[str, ...]
    reaction_ligand_ccds: tuple[str, ...]
    cofactor_state: str
    ligand_rscc: tuple[float, ...]
    cofactor_rscc: float | None
    assembly_oligomeric_count: int
    linked_label_ids: tuple[str, ...]
    selection_notes: str
    pdb_url: str
    validation_url: str
    sequence_id: str

    @property
    def role(self) -> str:
        return GEOMETRY_USE[self.geometry_use]

    @property
    def lineage(self) -> str:
        return LINEAGE_GROUPS[self.enzyme]

    @property
    def has_reaction_ligand(self) -> bool:
        return bool(self.reaction_ligand_ccds)


@dataclass(frozen=True)
class KineticRecord:
    """One kinetic label, with every number recomputed from the original.

    Quantities are in ``s^-1``, ``M`` and ``M^-1 s^-1``. ``*_original`` and the
    units are kept next to them, so the conversion can be audited from the
    record alone. A quantity that is not a point estimate is ``None`` here and
    the reason is in :attr:`withheld`: asking for it through
    :meth:`require` raises instead of returning something nearby.
    """

    label_id: str
    enzyme: str
    variant: str
    substrate: str
    record_status: str
    kcat_original: float | None
    kcat_unit: str
    kcat_uncertainty_original: str
    km_original: float | None
    km_unit: str
    km_comparator: str
    km_uncertainty_original: str
    efficiency_original: float | None
    efficiency_unit: str
    efficiency_uncertainty_original: str
    kcat_s: float | None
    km_m: float | None
    efficiency_reported_m_s: float | None
    efficiency_derived_m_s: float | None
    ph: float
    temperature_c: float
    assay_cofactor: str
    label_qualification: str
    default_use: str
    experimental_complex_pdb_ids: tuple[str, ...]
    experimental_protein_pdb_ids: tuple[str, ...]
    template_pdb_ids: tuple[str, ...]
    source_id: str
    source_location: str
    quality_flags: str
    withheld: Mapping[str, str] = field(default_factory=dict)

    @property
    def tier(self) -> str:
        return TIERS[self.default_use]

    @property
    def lineage(self) -> str:
        return LINEAGE_GROUPS[self.enzyme]

    @property
    def label_type(self) -> str:
        """The kind of number this is. Records of different types never pool."""
        if self.record_status == "not_determined":
            return "not_determined"
        if self.record_status == "measured_apparent_numeric":
            return "apparent_kcat_km_regeneration_system"
        if self.kcat_original is None:
            return "reported_efficiency"
        if self.km_comparator and self.km_comparator != "eq":
            return "kcat_only_km_bounded"
        return "kcat_km_steady_state"

    @property
    def assay_group(self) -> str:
        """Records may be ranked against each other only within one of these."""
        return (f"{self.source_id}|pH {self.ph:g}|{self.temperature_c:g} C|"
                f"{self.assay_cofactor}|{self.label_qualification}")

    @property
    def all_pdb_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for pid in (self.experimental_complex_pdb_ids
                    + self.experimental_protein_pdb_ids + self.template_pdb_ids):
            if pid not in seen:
                seen.append(pid)
        return tuple(seen)

    @property
    def has_matched_complex(self) -> bool:
        return bool(self.experimental_complex_pdb_ids)

    @property
    def efficiency_discrepancy(self) -> float | None:
        """Relative difference between the reported and the derived efficiency."""
        a, b = self.efficiency_reported_m_s, self.efficiency_derived_m_s
        if a is None or b is None or a == 0:
            return None
        return abs(a - b) / abs(a)

    def point_quantities(self) -> tuple[str, ...]:
        out = []
        if self.kcat_s is not None:
            out.append("kcat")
        if self.km_m is not None:
            out.append("km")
        if self.efficiency_reported_m_s is not None:
            out.append("efficiency_reported")
        return tuple(out)

    def require(self, quantity: str) -> float:
        """The value of one quantity, or a refusal that says why there is none.

        The refusal is the point. ``ND`` is "not determined", a ``<`` bound is
        a bound, and a row with no ``Km`` has no ``kcat/Km`` of its own: each
        has an empty field, and an empty field read as ``0`` or as the nearest
        number is how a ranking acquires an order nobody measured.
        """
        values = {"kcat": self.kcat_s, "km": self.km_m,
                  "efficiency_reported": self.efficiency_reported_m_s,
                  "efficiency_derived": self.efficiency_derived_m_s}
        if quantity not in values:
            raise ReferenceSetError(f"{quantity!r} is not a quantity of a kinetic "
                                    f"record; known: {sorted(values)}")
        value = values[quantity]
        if value is None:
            why = self.withheld.get(quantity) or "the record carries no value"
            raise ReferenceSetError(f"{self.label_id}: no {quantity}: {why}")
        return value


@dataclass(frozen=True)
class SelectivityRecord:
    selectivity_id: str
    enzyme: str
    variant: str
    substrate: str
    er_s_over_r: float
    kinetic_label_id: str
    variant_identity_match: bool
    conditions_status: str
    note: str
    source_url: str

    @property
    def lineage(self) -> str:
        return LINEAGE_GROUPS[self.enzyme]


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    title: str
    doi: str
    location: str
    verification: str
    access_reuse: str
    url: str


# ==========================================================================
# the loaded set
# ==========================================================================

@dataclass(frozen=True)
class ReferenceSet:
    """Everything in the delivery, typed, with the checks' findings attached."""

    directory: Path
    manifest_digest: str
    usage_notes: tuple[tuple[str, str], ...]
    structures: tuple[StructureEntry, ...]
    kinetics: tuple[KineticRecord, ...]
    assay_conditions: Mapping[str, Mapping[str, str]]
    selectivity: tuple[SelectivityRecord, ...]
    ligand_validation: tuple[LigandValidation, ...]
    sources: tuple[SourceRecord, ...]
    auxiliary: tuple[Mapping[str, str], ...]
    findings: tuple[Finding, ...] = ()

    # -- lookups -----------------------------------------------------------
    def structure(self, pdb_id: str) -> StructureEntry:
        for s in self.structures:
            if s.pdb_id == pdb_id.upper():
                return s
        raise KeyError(pdb_id)

    def kinetic(self, label_id: str) -> KineticRecord:
        for k in self.kinetics:
            if k.label_id == label_id:
                return k
        raise KeyError(label_id)

    def validation_rows(self, pdb_id: str) -> tuple[LigandValidation, ...]:
        return tuple(v for v in self.ligand_validation if v.pdb_id == pdb_id.upper())

    def source(self, source_id: str) -> SourceRecord:
        for s in self.sources:
            if s.source_id == source_id:
                return s
        raise KeyError(source_id)

    def records(self, tier: str) -> tuple[KineticRecord, ...]:
        return tuple(k for k in self.kinetics if k.tier == tier)

    def kinetics_for_structure(self, pdb_id: str) -> tuple[KineticRecord, ...]:
        pid = pdb_id.upper()
        return tuple(k for k in self.kinetics if pid in k.all_pdb_ids)

    # -- counts the cover note makes claims about --------------------------
    def counts(self) -> dict[str, int]:
        numeric = [k for k in self.kinetics if k.record_status != "not_determined"]
        return {
            "structures": len(self.structures),
            "kinetic_records": len(self.kinetics),
            "numeric_records": len(numeric),
            "not_determined_records": sum(
                1 for k in self.kinetics if k.record_status == "not_determined"),
            "records_with_kcat": sum(1 for k in numeric if k.kcat_original is not None),
            "efficiency_only_records": sum(
                1 for k in numeric if k.kcat_original is None),
            "selectivity_records": len(self.selectivity),
            "core_records": sum(1 for k in numeric if k.tier == "core"),
            "secondary_records": sum(1 for k in numeric if k.tier == "secondary"),
            "sensitivity_records": sum(1 for k in numeric if k.tier == "sensitivity"),
        }

    def independence(self) -> dict[str, Any]:
        """How many independent things the many entries and records are.

        A tolerance interval counts samples, and it counts them as if each came
        from a different draw of the population. Nineteen entries and 28
        records come from a handful of enzyme lineages, so the honest sample
        size for a claim about *enzymes* is the lineage count. Returned with
        its working so the number can be checked against the groups.
        """
        def by_lineage(items: Iterable[Any]) -> dict[str, int]:
            out: dict[str, int] = {}
            for item in items:
                out[item.lineage] = out.get(item.lineage, 0) + 1
            return dict(sorted(out.items()))

        numeric = [k for k in self.kinetics if k.record_status != "not_determined"]
        core = [k for k in numeric if k.tier == "core"]
        return {
            "structure_entries": len(self.structures),
            "structure_lineages": by_lineage(self.structures),
            "kinetic_records": len(numeric),
            "kinetic_lineages": by_lineage(numeric),
            "core_records": len(core),
            "core_lineages": by_lineage(core),
            "n_independent_core_lineages": len({k.lineage for k in core}),
            "n_independent_structure_lineages": len({s.lineage for s in self.structures}),
            "n_independent_pose_lineages": len({
                s.lineage for s in self.structures
                if s.geometry_use in ("conditional_pose_reference",
                                      "conditional_product_reference",
                                      "review_clash")}),
        }

    def summary(self) -> dict[str, Any]:
        return {"directory": str(self.directory),
                "manifest_digest": self.manifest_digest,
                "counts": self.counts(),
                "independence": self.independence(),
                "findings": {sev: sum(1 for f in self.findings if f.severity == sev)
                             for sev in ("error", "warning", "info")}}


# ==========================================================================
# reading the tables
# ==========================================================================

def _read_csv(path: Path, expected: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if header != list(expected):
            raise ReferenceSetError(
                f"{path.name}: the header is not the one this loader was "
                f"written against.\n  expected {list(expected)}\n  found    {header}")
        return [dict(row) for row in reader]


class _Reader:
    """Parses cells and collects what it could not, rather than stopping."""

    def __init__(self) -> None:
        self.findings: list[Finding] = []

    def error(self, code: str, subject: str, message: str) -> None:
        self.findings.append(Finding("error", code, subject, message))

    def number(self, raw: str | None, subject: str, field_name: str,
               *, required: bool = False) -> float | None:
        text = _clean(raw)
        if not text:
            if required:
                self.error("cell.required", subject, f"{field_name} is blank")
            return None
        try:
            value = float(text)
        except ValueError:
            self.error("cell.not_a_number", subject,
                       f"{field_name} is {text!r}, not a number")
            return None
        if not math.isfinite(value):
            self.error("cell.not_finite", subject, f"{field_name} is {text!r}")
            return None
        return value

    def integer(self, raw: str | None, subject: str, field_name: str) -> int | None:
        value = self.number(raw, subject, field_name)
        if value is None:
            return None
        if value != int(value):
            self.error("cell.not_an_integer", subject, f"{field_name} is {raw!r}")
            return None
        return int(value)


def _convert(reader: _Reader, subject: str, field_name: str, value: float | None,
             unit: str, *, expect: str) -> float | None:
    """Convert through the shared unit table, requiring the expected canonical unit."""
    if value is None:
        return None
    canon = canonical_unit(unit)
    if canon is None:
        reader.error("unit.unknown", subject,
                     f"{field_name} is in {unit!r}, which the unit table does not "
                     f"know; it will not be guessed")
        return None
    name, factor = canon
    if name != expect:
        reader.error("unit.wrong_quantity", subject,
                     f"{field_name} is in {unit!r} ({name}), expected a unit that "
                     f"canonicalises to {expect}")
        return None
    return value * factor


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=RELATIVE_TOLERANCE, abs_tol=0.0)


def _parse_kinetics(rows: list[dict[str, str]], reader: _Reader) -> list[KineticRecord]:
    out: list[KineticRecord] = []
    for row in rows:
        subject = _clean(row["label_id"])
        status = _clean(row["record_status"])
        if status not in RECORD_STATUS:
            reader.error("kinetics.status", subject,
                         f"record_status {status!r} is not one of {RECORD_STATUS}")
        default_use = _clean(row["default_use"])
        if default_use not in TIERS:
            reader.error("kinetics.default_use", subject,
                         f"default_use {default_use!r} has no tier; a new value "
                         f"needs a person to say what it means")
            continue
        if _clean(row["enzyme"]) not in LINEAGE_GROUPS:
            reader.error("kinetics.lineage", subject,
                         f"enzyme {row['enzyme']!r} has no lineage group, so it "
                         f"could not be kept on one side of a split")
            continue

        kcat0 = reader.number(row["kcat_original"], subject, "kcat_original")
        km0 = reader.number(row["Km_original"], subject, "Km_original")
        eff0 = reader.number(row["efficiency_original"], subject, "efficiency_original")
        comparator = _clean(row["Km_comparator"])
        if comparator not in ("", "eq", "<", ">"):
            reader.error("kinetics.comparator", subject,
                         f"Km_comparator {comparator!r} is not eq, <, > or blank")

        kcat_s = _convert(reader, subject, "kcat", kcat0, row["kcat_unit"], expect="s-1")
        km_mm = _convert(reader, subject, "Km", km0, row["Km_unit"], expect="mM")
        km_m = None if km_mm is None else km_mm / 1000.0
        eff_m_s = _convert(reader, subject, "efficiency", eff0, row["efficiency_unit"],
                           expect="M-1 s-1")

        withheld: dict[str, str] = {}
        if status == "not_determined":
            for q in ("kcat", "km", "efficiency_reported", "efficiency_derived"):
                withheld[q] = ("recorded as ND (not determined): the experiment "
                               "gave no value, which is not a value of zero")
        if comparator in ("<", ">"):
            withheld["km"] = (f"the source gives only a bound ({comparator} "
                              f"{row['Km_original']} {row['Km_unit']}); a bound is "
                              f"not a point estimate")
            withheld["efficiency_derived"] = (
                "kcat/Km would need a Km point; the source gives only a bound")
        if kcat0 is None and status != "not_determined":
            withheld.setdefault("kcat", "the source reports an efficiency, not kcat")
            withheld.setdefault("km", "the source reports an efficiency, not Km")
            withheld.setdefault("efficiency_derived",
                                "no kcat and Km were reported to divide")
        if eff0 is None and status != "not_determined":
            withheld.setdefault("efficiency_reported",
                                "the source reports no efficiency for this row; the "
                                "derived quotient is a different field")

        derived = None
        if kcat_s is not None and km_m not in (None, 0.0) and comparator in ("", "eq"):
            derived = kcat_s / km_m

        # The spreadsheet's own SI columns, as the file stores them.
        cached = {
            "kcat_s_1": reader.number(row["kcat_s_1"], subject, "kcat_s_1"),
            "Km_value_or_bound_M": reader.number(row["Km_value_or_bound_M"], subject,
                                                 "Km_value_or_bound_M"),
            "efficiency_reported_M_1_s_1": reader.number(
                row["efficiency_reported_M_1_s_1"], subject, "efficiency_reported_M_1_s_1"),
            "efficiency_ratio_derived_M_1_s_1": reader.number(
                row["efficiency_ratio_derived_M_1_s_1"], subject,
                "efficiency_ratio_derived_M_1_s_1"),
        }
        # Km_value_or_bound_M holds a bound too: for a '<' row it is the bound
        # in molar, which is not a point and is compared as what it is.
        km_for_compare = km_m
        checks = (("kcat_s_1", kcat_s), ("Km_value_or_bound_M", km_for_compare),
                  ("efficiency_reported_M_1_s_1", eff_m_s),
                  ("efficiency_ratio_derived_M_1_s_1", derived))
        for name, mine in checks:
            if not _close(mine, cached[name]):
                reader.error(
                    "kinetics.recomputation", subject,
                    f"{name}: the workbook caches {cached[name]!r} but converting "
                    f"the original values through the unit table gives {mine!r}")

        ph = reader.number(row["pH"], subject, "pH", required=True)
        temp = reader.number(row["temperature_C_nominal"], subject,
                             "temperature_C_nominal", required=True)
        if ph is None or temp is None:
            continue
        out.append(KineticRecord(
            label_id=subject, enzyme=_clean(row["enzyme"]),
            variant=_clean(row["variant"]), substrate=_clean(row["substrate"]),
            record_status=status,
            kcat_original=kcat0, kcat_unit=_clean(row["kcat_unit"]),
            kcat_uncertainty_original=_clean(row["kcat_uncertainty_original"]),
            km_original=km0, km_unit=_clean(row["Km_unit"]), km_comparator=comparator,
            km_uncertainty_original=_clean(row["Km_uncertainty_original"]),
            efficiency_original=eff0, efficiency_unit=_clean(row["efficiency_unit"]),
            efficiency_uncertainty_original=_clean(row["efficiency_uncertainty_original"]),
            kcat_s=None if status == "not_determined" else kcat_s,
            km_m=None if (status == "not_determined" or comparator in ("<", ">")) else km_m,
            efficiency_reported_m_s=None if status == "not_determined" else eff_m_s,
            efficiency_derived_m_s=derived,
            ph=ph, temperature_c=temp, assay_cofactor=_clean(row["assay_cofactor"]),
            label_qualification=_clean(row["label_qualification"]),
            default_use=default_use,
            experimental_complex_pdb_ids=_split(row["experimental_complex_pdb_ids"]),
            experimental_protein_pdb_ids=_split(row["experimental_protein_pdb_ids"]),
            template_pdb_ids=_split(row["template_pdb_ids"]),
            source_id=_clean(row["source_id"]),
            source_location=_clean(row["source_location"]),
            quality_flags=_clean(row["quality_flags"]),
            withheld=withheld))
    return out


def _parse_structures(rows: list[dict[str, str]], reader: _Reader) -> list[StructureEntry]:
    out: list[StructureEntry] = []
    for row in rows:
        pid = _clean(row["pdb_id"]).upper()
        use = _clean(row["geometry_use"])
        if use not in GEOMETRY_USE:
            reader.error("structures.geometry_use", pid,
                         f"geometry_use {use!r} is not one of {sorted(GEOMETRY_USE)}; "
                         f"a new grade is a decision, not a parse")
            continue
        if _clean(row["enzyme"]) not in LINEAGE_GROUPS:
            reader.error("structures.lineage", pid,
                         f"enzyme {row['enzyme']!r} has no lineage group")
            continue
        summary = _clean(row["selected_ligand_cofactor_RSCC_summary"])
        lig_rscc: tuple[float, ...] = ()
        cof_rscc: float | None = None
        if summary:
            lig_text, _, cof_text = summary.partition(";")
            try:
                lig_rscc = tuple(float(p) for p in lig_text.split("/") if p.strip())
                cof_rscc = float(cof_text) if cof_text.strip() else None
            except ValueError:
                reader.error("structures.rscc", pid, f"unreadable RSCC summary {summary!r}")
        resolution = reader.number(row["resolution_A"], pid, "resolution_A", required=True)
        count = reader.integer(row["assembly_oligomeric_count"], pid,
                               "assembly_oligomeric_count")
        if resolution is None or count is None:
            continue
        out.append(StructureEntry(
            pdb_id=pid, enzyme=_clean(row["enzyme"]), variant=_clean(row["variant"]),
            geometry_use=use, resolution_a=resolution,
            bound_reaction_ligand=_clean(row["bound_reaction_ligand"]),
            experimental_state=_clean(row["experimental_state"]),
            scoring_chain=_clean(row["scoring_auth_chain"]),
            altlocs=_split(_clean(row["selected_altloc"]), "/"),
            reaction_ligand_ccds=_split(_clean(row["reaction_ligand_ccd"]), "/"),
            cofactor_state=_clean(row["cofactor_state"]),
            ligand_rscc=lig_rscc, cofactor_rscc=cof_rscc,
            assembly_oligomeric_count=count,
            linked_label_ids=_split(row["linked_label_ids"]),
            selection_notes=_clean(row["selection_notes"]),
            pdb_url=_clean(row["pdb_url"]), validation_url=_clean(row["validation_url"]),
            sequence_id=_clean(row["pdb_sequence_id"])))
    return out


def _parse_ligand_validation(rows: list[dict[str, str]], reader: _Reader
                             ) -> list[LigandValidation]:
    out: list[LigandValidation] = []
    for row in rows:
        subject = f"{_clean(row['pdb_id'])}:{_clean(row['auth_chain'])}{_clean(row['auth_residue_number'])}"
        number = reader.integer(row["auth_residue_number"], subject, "auth_residue_number")
        if number is None:
            continue
        out.append(LigandValidation(
            pdb_id=_clean(row["pdb_id"]).upper(), chain=_clean(row["auth_chain"]),
            residue_number=number, ccd=_clean(row["ccd"]).upper(),
            altloc=_clean(row["altloc"]),
            rscc=reader.number(row["RSCC"], subject, "RSCC"),
            rsr=reader.number(row["RSR"], subject, "RSR"),
            occupancy=reader.number(row["average_occupancy"], subject, "average_occupancy"),
            b_factor=reader.number(row["average_B_A2"], subject, "average_B_A2"),
            raw_clash_children=reader.integer(row["raw_clash_child_count"], subject,
                                              "raw_clash_child_count"),
            geometry_children=reader.integer(row["geometry_child_count"], subject,
                                             "geometry_child_count"),
            warning=_clean(row["warning"]), source_url=_clean(row["source_url"])))
    return out


def _parse_selectivity(rows: list[dict[str, str]], reader: _Reader
                       ) -> list[SelectivityRecord]:
    out: list[SelectivityRecord] = []
    for row in rows:
        subject = _clean(row["selectivity_id"])
        er = reader.number(row["er_S_over_R"], subject, "er_S_over_R", required=True)
        match = _clean(row["kinetic_variant_identity_match"])
        if match not in ("0", "1"):
            reader.error("selectivity.match_flag", subject,
                         f"kinetic_variant_identity_match is {match!r}, not 0 or 1")
            continue
        if _clean(row["enzyme"]) not in LINEAGE_GROUPS or er is None:
            if er is not None:
                reader.error("selectivity.lineage", subject,
                             f"enzyme {row['enzyme']!r} has no lineage group")
            continue
        out.append(SelectivityRecord(
            selectivity_id=subject, enzyme=_clean(row["enzyme"]),
            variant=_clean(row["variant"]), substrate=_clean(row["substrate"]),
            er_s_over_r=er, kinetic_label_id=_clean(row["kinetic_label_id"]),
            variant_identity_match=match == "1",
            conditions_status=_clean(row["conditions_status"]),
            note=_clean(row["note"]), source_url=_clean(row["source_url"])))
    return out


# ==========================================================================
# the checks
# ==========================================================================

def _check(rs_parts: dict[str, Any], reader: _Reader) -> list[Finding]:
    """Cross-table integrity. Returns every finding, errors first."""
    structures: list[StructureEntry] = rs_parts["structures"]
    kinetics: list[KineticRecord] = rs_parts["kinetics"]
    selectivity: list[SelectivityRecord] = rs_parts["selectivity"]
    validation: list[LigandValidation] = rs_parts["ligand_validation"]
    sources: list[SourceRecord] = rs_parts["sources"]
    assay: Mapping[str, Mapping[str, str]] = rs_parts["assay_conditions"]
    auxiliary: list[Mapping[str, str]] = rs_parts["auxiliary"]
    findings = reader.findings

    def err(code: str, subject: str, message: str) -> None:
        findings.append(Finding("error", code, subject, message))

    def warn(code: str, subject: str, message: str) -> None:
        findings.append(Finding("warning", code, subject, message))

    def info(code: str, subject: str, message: str) -> None:
        findings.append(Finding("info", code, subject, message))

    pdb_ids = {s.pdb_id for s in structures}
    label_ids = {k.label_id for k in kinetics}
    for kind, items in (("structure", [s.pdb_id for s in structures]),
                        ("kinetic label", [k.label_id for k in kinetics]),
                        ("selectivity id", [s.selectivity_id for s in selectivity])):
        dupes = sorted({i for i in items if items.count(i) > 1})
        if dupes:
            err("ids.duplicate", kind, f"duplicated: {dupes}; one complex or one "
                f"label counted twice inflates every count built on it")

    # -- the cover note's counts ------------------------------------------
    counts = ReferenceSet(
        directory=Path("."), manifest_digest="", usage_notes=(),
        structures=tuple(structures), kinetics=tuple(kinetics),
        assay_conditions=assay, selectivity=tuple(selectivity),
        ligand_validation=tuple(validation), sources=tuple(sources),
        auxiliary=tuple(auxiliary)).counts()
    for name, stated in STATED_COUNTS.items():
        if counts.get(name) != stated:
            err("counts.mismatch", name,
                f"the delivery states {stated}, the tables hold {counts.get(name)}")

    # -- structure <-> kinetics, both directions ---------------------------
    by_label = {k.label_id: k for k in kinetics}
    for s in structures:
        for lid in s.linked_label_ids:
            if lid in by_label:
                if s.pdb_id not in by_label[lid].all_pdb_ids:
                    err("link.one_way", f"{s.pdb_id}->{lid}",
                        f"{s.pdb_id} names this label but the label names none of "
                        f"{by_label[lid].all_pdb_ids}")
            elif lid in AUXILIARY_LABELS:
                names = {a.get("item") for a in auxiliary}
                if AUXILIARY_LABELS[lid] not in names:
                    err("link.auxiliary_missing", f"{s.pdb_id}->{lid}",
                        f"expected auxiliary item {AUXILIARY_LABELS[lid]!r}")
            else:
                err("link.dangling", f"{s.pdb_id}->{lid}",
                    "names a label that is in neither the kinetics nor the "
                    "known auxiliary records")
    by_pdb = {s.pdb_id: s for s in structures}
    for k in kinetics:
        for pid in k.all_pdb_ids:
            if pid not in by_pdb:
                err("link.dangling", f"{k.label_id}->{pid}",
                    "names a PDB entry that is not in the structure table")
            elif k.label_id not in by_pdb[pid].linked_label_ids:
                err("link.one_way", f"{k.label_id}->{pid}",
                    f"the label names {pid} but {pid} does not name the label")
    for k in kinetics:
        if k.source_id not in {s.source_id for s in sources}:
            err("source.unknown", k.label_id, f"source_id {k.source_id!r} is not in "
                f"the source table")
    for s in structures:
        if s.pdb_url != f"https://www.rcsb.org/structure/{s.pdb_id}":
            err("structures.url", s.pdb_id, f"pdb_url is {s.pdb_url!r}")
        if s.sequence_id != f"PDB_{s.pdb_id}_entity1":
            warn("structures.sequence_id", s.pdb_id, f"pdb_sequence_id is {s.sequence_id!r}")

    # -- assay conditions agree with the kinetic rows ----------------------
    if set(assay) != label_ids:
        err("assay.labels", "assay_conditions",
            f"label ids differ from the kinetics sheet: only in assay "
            f"{sorted(set(assay) - label_ids)}, only in kinetics "
            f"{sorted(label_ids - set(assay))}")
    for k in kinetics:
        row = assay.get(k.label_id)
        if row is None:
            continue
        try:
            same = (float(row["pH"]) == k.ph
                    and float(row["temperature_C_nominal"]) == k.temperature_c)
        except ValueError:
            same = False
        if not same:
            err("assay.disagrees", k.label_id,
                f"pH/temperature differ between the kinetics and assay sheets")
        if _clean(row.get("cofactor")) != k.assay_cofactor:
            err("assay.cofactor", k.label_id,
                f"cofactor {row.get('cofactor')!r} (assay sheet) vs "
                f"{k.assay_cofactor!r} (kinetics sheet)")

    # -- label-type separation rules --------------------------------------
    for k in kinetics:
        if k.record_status == "not_determined":
            if k.kcat_original is not None or k.km_original is not None \
                    or k.efficiency_original is not None:
                err("nd.has_value", k.label_id,
                    "a record marked not determined carries a number; ND is not a "
                    "value, and a number here would be read as one")
            if k.default_use != "ND_not_zero":
                err("nd.use", k.label_id, f"default_use is {k.default_use!r}")
        if k.km_comparator in ("<", ">"):
            if k.efficiency_derived_m_s is not None:
                err("bound.derived", k.label_id,
                    "an efficiency was derived from a Km that is only a bound")
            if k.default_use != "kcat_only_Km_bound_quarantined":
                err("bound.use", k.label_id,
                    f"a bounded Km with default_use {k.default_use!r}")
        if k.default_use == "kcat_only_Km_bound_quarantined" and k.km_comparator == "eq":
            err("bound.use", k.label_id, "quarantined as a bound but the row is a point")
        disc = k.efficiency_discrepancy
        if disc is not None:
            sev = warn if disc > EFFICIENCY_DISCREPANCY_WARN else info
            sev("efficiency.reported_vs_derived", k.label_id,
                f"reported efficiency differs from kcat/Km by {disc:.1%}; both are "
                f"kept, as two fields")

    # -- selectivity -------------------------------------------------------
    for sel in selectivity:
        if sel.variant_identity_match:
            k = by_label.get(sel.kinetic_label_id)
            if k is None:
                err("selectivity.link", sel.selectivity_id,
                    f"claims a matching kinetic record {sel.kinetic_label_id!r} "
                    f"that does not exist")
            elif (k.enzyme, k.variant, k.substrate) != (sel.enzyme, sel.variant,
                                                         sel.substrate):
                err("selectivity.identity", sel.selectivity_id,
                    f"claims to match {k.label_id} but enzyme/variant/substrate "
                    f"differ: {(sel.enzyme, sel.variant, sel.substrate)} vs "
                    f"{(k.enzyme, k.variant, k.substrate)}")
        elif sel.kinetic_label_id:
            err("selectivity.unmatched_linked", sel.selectivity_id,
                "is marked as not matching its kinetic record yet names one; a "
                "record that does not match must not be joined")

    # -- the density summary against the validation rows it came from ------
    for s in structures:
        if not (s.ligand_rscc or s.cofactor_rscc is not None):
            continue
        rows = [v for v in validation if v.pdb_id == s.pdb_id
                and (not s.scoring_chain or v.chain == s.scoring_chain)]
        have_lig = [v.rscc for v in rows if v.ccd in s.reaction_ligand_ccds
                    and v.rscc is not None]
        have_cof = [v.rscc for v in rows if v.ccd not in s.reaction_ligand_ccds
                    and v.rscc is not None]
        for want in s.ligand_rscc:
            if not any(abs(want - h) <= RSCC_TOLERANCE for h in have_lig):
                err("rscc.unsupported", s.pdb_id,
                    f"the summary gives a ligand RSCC of {want} but the "
                    f"validation rows for chain {s.scoring_chain} "
                    f"{list(s.reaction_ligand_ccds)} are {have_lig}")
        if s.cofactor_rscc is not None and not any(
                abs(s.cofactor_rscc - h) <= RSCC_TOLERANCE for h in have_cof):
            err("rscc.unsupported", s.pdb_id,
                f"the summary gives a cofactor RSCC of {s.cofactor_rscc} but the "
                f"validation rows give {have_cof}")
    for v in validation:
        if v.pdb_id not in pdb_ids:
            err("validation.dangling", v.pdb_id, "validation row for an entry that "
                "is not in the structure table")
        if v.rscc is not None and not 0.0 <= v.rscc <= 1.0:
            err("validation.range", v.pdb_id, f"RSCC {v.rscc} is outside [0, 1]")

    # -- geometry grades: no strict gold, and grades agree with their evidence
    for s in structures:
        if s.geometry_use == "low_ligand_density" and s.ligand_rscc:
            if max(s.ligand_rscc) >= 0.9 and (s.cofactor_rscc or 0) >= 0.9:
                warn("grade.density", s.pdb_id,
                     "graded low_ligand_density but both RSCC values are >= 0.9")
        if s.geometry_use == "conditional_pose_reference":
            if not s.has_reaction_ligand:
                err("grade.no_ligand", s.pdb_id,
                    "a pose reference with no reaction ligand named")
            elif s.ligand_rscc and min(s.ligand_rscc) < 0.8:
                warn("grade.density", s.pdb_id,
                     f"a pose reference with ligand RSCC {min(s.ligand_rscc)}")
        if s.role == "scaffold_only" and s.has_reaction_ligand:
            err("grade.scaffold", s.pdb_id, "a scaffold-only entry that names a "
                "reaction ligand")

    # -- source coverage ---------------------------------------------------
    for src in sources:
        if not src.access_reuse.strip():
            err("source.reuse", src.source_id, "no reuse statement recorded")
    return findings


# ==========================================================================
# loading
# ==========================================================================

_TABLES = {stem: header for _, (stem, header) in SHEET_TABLES.items()}


def load_reference_set(directory: str | Path | None = None, *,
                       verify: bool = True, strict: bool = True) -> ReferenceSet:
    """Load the set, checking the files and the numbers.

    ``verify`` hashes every file against the manifest and re-runs the
    conversion; turning it off is for a caller that is about to *write* the
    manifest. ``strict`` raises :class:`ReferenceSetError` when any ``error``
    finding exists; with it off the findings are returned on the set so that an
    audit can print every one rather than the first.
    """
    base = Path(directory) if directory is not None else default_reference_dir()
    findings: list[Finding] = []
    manifest_digest = ""
    if verify:
        findings += verify_manifest(base)
        findings += verify_conversion(base)
        try:
            manifest_digest = json.loads(
                (base / MANIFEST_NAME).read_text(encoding="utf-8")).get("data_digest", "")
        except (OSError, ValueError):
            manifest_digest = ""
        integrity = [f for f in findings if f.severity == "error"]
        if integrity and strict:
            # Do not read tables that are not the delivered ones: a parse of
            # edited data would report errors about the edit, not about the file.
            shown = "\n".join(f"  {f}" for f in integrity[:12])
            raise ReferenceSetError(
                f"{base}: the files are not the delivered ones:\n{shown}", findings)
    tables_dir = base / "tables"
    raw = {stem: _read_csv(tables_dir / f"{stem}.csv", header)
           for stem, header in _TABLES.items()}

    reader = _Reader()
    structures = _parse_structures(raw["structures"], reader)
    kinetics = _parse_kinetics(raw["kinetics"], reader)
    validation = _parse_ligand_validation(raw["ligand_validation"], reader)
    selectivity = _parse_selectivity(raw["selectivity"], reader)
    sources = [SourceRecord(
        source_id=_clean(r["source_id"]), title=_clean(r["title"]), doi=_clean(r["doi"]),
        location=_clean(r["location"]), verification=_clean(r["verification"]),
        access_reuse=_clean(r["access_reuse"]), url=_clean(r["url"]))
        for r in raw["sources"]]
    parts = {
        "structures": structures, "kinetics": kinetics,
        "ligand_validation": validation, "selectivity": selectivity,
        "sources": sources,
        "assay_conditions": {_clean(r["label_id"]): r for r in raw["assay_conditions"]},
        "auxiliary": raw["auxiliary"],
    }
    findings += _check(parts, reader)
    findings.sort(key=lambda f: ({"error": 0, "warning": 1, "info": 2}[f.severity],
                                 f.code, f.subject))
    errors = [f for f in findings if f.severity == "error"]
    if errors and strict:
        shown = "\n".join(f"  {f}" for f in errors[:12])
        more = f"\n  ... and {len(errors) - 12} more" if len(errors) > 12 else ""
        raise ReferenceSetError(
            f"{base}: {len(errors)} error(s) in the reference set:\n{shown}{more}",
            findings)
    return ReferenceSet(
        directory=base, manifest_digest=manifest_digest,
        usage_notes=tuple((_clean(r["topic"]), _clean(r["guidance"]))
                          for r in raw["usage_notes"]),
        structures=tuple(structures), kinetics=tuple(kinetics),
        assay_conditions=parts["assay_conditions"], selectivity=tuple(selectivity),
        ligand_validation=tuple(validation), sources=tuple(sources),
        auxiliary=tuple(raw["auxiliary"]), findings=tuple(findings))
