"""The KRED reference set's second delivery: a 12 MB archive, ingested by rule.

WHAT ARRIVED
============
The reference set was delivered twice. First a spreadsheet, which
:mod:`eagent.eval.kred_reference` converts and checks. Then an archive of 194
files: the same spreadsheet byte for byte, a JSON/CSV export of the same tables,
the full assay conditions that the spreadsheet only summarised, the compiler's
per-structure audit output, three evidence files, the 19 coordinate files with
their biological assemblies and validation reports, the wwPDB chemical component
definitions, and the 2026 Ssal-KRED ortholog activity data.

WHAT IS COMMITTED AND WHAT IS PINNED
====================================
The archive labels every member ``curated_or_supporting_artifact`` (the
compilation itself) or ``upstream_original`` (somebody else's file, with the URL
it came from). That label decides the disposition, with one stated exception:

* **committed** -- the curated compilation, about 1.8 MB. It is the delivery;
  it is small; and its reuse terms are the ones ``NOTICE.txt`` states.
* **committed_upstream** -- the 2026 activity data (two round CSVs, the ortholog
  FASTA, the substrate SMILES, the licence and the two Zenodo metadata files).
  These are ``upstream_original``, but they are openly licensed (MIT and
  CC BY 4.0), under 100 KB in total, and a loader in this package reads them:
  pinning them alone would make :mod:`eagent.eval.kred_activity` unusable
  offline, which is the opposite of the point.
* **pinned** -- the 46 MB of large upstream originals: coordinates, assemblies,
  validation XML, the RCSB entry/entity JSON and the chemical components. Each
  keeps its sha256 and its URL in ``archive.manifest.json``, and every one is
  re-fetchable from a route this project has probed. The 19 asymmetric-unit
  files are already pinned a second time, independently, in
  ``coordinates/coordinates.manifest.json``.
* **duplicate_of_source** -- the spreadsheet, which is byte-identical to the one
  already under ``source/``.
* **dropped_duplicate** -- ``data/structure_audit_raw.json``, which is exactly
  the 19 per-structure ``audit_metadata.json`` files collected into a list. The
  per-structure files are committed; the 787 KB aggregate is not, and its hash
  is kept so the decision can be checked.

WHAT THE INGEST ESTABLISHED
===========================
Not "the archive says so". Each of these is a check in this module or in
:mod:`eagent.eval.kred_activity`, run against the files:

* the archive's own ``file_manifest.json`` lists 193 of its 194 members and
  every listed hash matches the member (:func:`verify_archive_manifest`);
* the 19 coordinate files are **byte-identical** to the ones this project
  fetched independently from the RCSB and pinned a day earlier. Two
  retrievals by two parties agreeing exactly is the strongest provenance
  statement available here (:func:`coordinate_agreement`);
* the JSON export and the spreadsheet export of the same five tables agree in
  all 1632 shared fields once three representation conventions are normalised
  -- a JSON list against a joined string, ``=`` against ``eq``, a boolean
  against ``1``/``0`` (:func:`cross_check_tables`);
* the spreadsheet in the archive is byte-identical to the committed one.

None of that makes the numbers right. It makes the two deliveries the same
delivery, and the compilation the thing it says it is.
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..errors import EAgentError
from .kred_reference import Finding, default_reference_dir

__all__ = [
    "BundleError",
    "ARCHIVE_NAME",
    "ARCHIVE_SHA256",
    "ARCHIVE_ROOT",
    "BUNDLE_DIRNAME",
    "ARCHIVE_MANIFEST",
    "COMMITTED_UPSTREAM",
    "DERIVED_FILES",
    "DROPPED_DUPLICATES",
    "MAX_MEMBER_BYTES",
    "MAX_TOTAL_BYTES",
    "disposition_for",
    "read_archive",
    "import_archive",
    "verify_archive_manifest",
    "verify_bundle",
    "cross_check_tables",
    "coordinate_agreement",
    "bundle_dir",
]


class BundleError(EAgentError):
    """The archive is not the one this importer was written for."""


#: The delivered archive, and the sha256 of the bytes this importer was run on.
#: Recorded rather than required: a later delivery has another hash, and the
#: importer says so instead of silently ingesting a different archive.
ARCHIVE_NAME = "KRED_Calibration_Reference_v0.1.zip"
ARCHIVE_SHA256 = "0b32db1402150ac9bcde2a8c4aac8c4da5b8eae13428f35b24a25f25c9266025"

#: Every member sits under this one directory; anything else is refused.
ARCHIVE_ROOT = "KRED_Calibration_Reference_v0.1/"

BUNDLE_DIRNAME = "bundle"
ARCHIVE_MANIFEST = "archive.manifest.json"

#: ``upstream_original`` members that are committed anyway: openly licensed,
#: small, and read by a loader in this package. Listed one by one, because
#: "small and open" is a judgement and it should be visible per file.
COMMITTED_UPSTREAM: frozenset[str] = frozenset({
    "activity_expansion_2026/LICENSE",
    "activity_expansion_2026/github_README.md",
    "activity_expansion_2026/ortholog_sequences.fasta",
    "activity_expansion_2026/round_1.csv",
    "activity_expansion_2026/round_2.csv",
    "activity_expansion_2026/zenodo_README.txt",
    "activity_expansion_2026/zenodo_metadata.json",
    "activity_expansion_2026/zenodo_substrate_SMILES.csv",
})

#: Files written into the bundle by this package rather than delivered in the
#: archive, with the command that regenerates each. They are not archive
#: members, so the archive manifest does not list them and
#: :func:`verify_bundle` must not call them unlisted; the reference set's own
#: ``MANIFEST.json`` hashes them like every other data file, so they are still
#: tamper-evident.
DERIVED_FILES: Mapping[str, str] = {
    "activity_expansion_2026/derived_identity_groups.json":
        "eagent reference activity --write-identity",
}

#: Curated members that are exactly reconstructible from other committed
#: members, with what each duplicates. Not committed; the hash stays.
DROPPED_DUPLICATES: Mapping[str, str] = {
    "data/structure_audit_raw.json":
        "a list of the 19 structures/<ID>/audit_metadata.json files, which are "
        "committed individually; checked equal entry by entry at import",
}

#: The spreadsheet, which ``source/`` already holds.
_SOURCE_WORKBOOK = "KRED_Calibration_Reference_v0.1.xlsx"

#: Caps on what the archive may expand to. A research bundle of this kind is
#: tens of megabytes; anything past these is not one.
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024

#: ``(json export, csv table, key column)`` for every table delivered twice.
_CROSS_TABLES: tuple[tuple[str, str, str, str], ...] = (
    ("kinetics", "data/activity_labels.json", "kinetics.csv", "label_id"),
    ("selectivity", "data/selectivity_labels.json", "selectivity.csv", "selectivity_id"),
    ("structures", "data/structure_manifest.json", "structures.csv", "pdb_id"),
    ("sources", "data/sources.json", "sources.csv", "source_id"),
    ("assay_conditions", "data/assay_conditions.json", "assay_conditions.csv", "label_id"),
)


def bundle_dir(reference_dir: str | Path | None = None) -> Path:
    base = Path(reference_dir) if reference_dir is not None else default_reference_dir()
    return base / BUNDLE_DIRNAME


# ==========================================================================
# the disposition rule
# ==========================================================================

def disposition_for(path: str, kind: str) -> str:
    """Where one member goes, and why. The rule is the module docstring's."""
    if path == _SOURCE_WORKBOOK:
        return "duplicate_of_source"
    if path in DROPPED_DUPLICATES:
        return "dropped_duplicate"
    if path in COMMITTED_UPSTREAM:
        return "committed_upstream"
    if kind == "curated_or_supporting_artifact":
        return "committed"
    if kind == "upstream_original":
        return "pinned"
    raise BundleError(
        f"{path}: the archive labels this {kind!r}, which this importer has no "
        f"rule for; a new label is a decision, not a parse")


# ==========================================================================
# reading the archive
# ==========================================================================

@dataclass(frozen=True)
class _Member:
    path: str
    data: bytes
    sha256: str
    kind: str
    source_url: str | None


def read_archive(zip_path: str | Path) -> tuple[dict[str, _Member], dict[str, Any]]:
    """Every member of the archive, hashed, plus the archive's own manifest.

    Refuses an absolute or traversing member name, a member or total over the
    caps, a member outside the single expected root, and an archive whose own
    manifest is missing. Nothing is written.
    """
    try:
        archive = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise BundleError(f"{zip_path} is not a readable zip: {exc}") from exc

    total = 0
    members: dict[str, _Member] = {}
    raw: dict[str, bytes] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if name.startswith("/") or "\\" in name or ".." in name.split("/"):
            raise BundleError(f"{name!r} is not a safe member name")
        if not name.startswith(ARCHIVE_ROOT):
            raise BundleError(
                f"{name!r} is outside {ARCHIVE_ROOT!r}; this importer expects one "
                f"root directory")
        if info.file_size > MAX_MEMBER_BYTES:
            raise BundleError(f"{name} expands to {info.file_size} bytes")
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise BundleError("the archive expands beyond the allowed total")
        raw[name[len(ARCHIVE_ROOT):]] = archive.read(info)

    if "file_manifest.json" not in raw:
        raise BundleError(
            "the archive has no file_manifest.json, so its members carry no "
            "declared kind or upstream URL and nothing can be dispositioned")
    manifest = json.loads(raw["file_manifest.json"].decode("utf-8"))
    kinds = {f["path"]: f for f in manifest.get("files") or ()}
    for path, data in raw.items():
        entry = kinds.get(path, {})
        members[path] = _Member(
            path=path, data=data, sha256=hashlib.sha256(data).hexdigest(),
            kind=str(entry.get("kind") or "curated_or_supporting_artifact"),
            source_url=entry.get("source_url"))
    return members, manifest


def verify_archive_manifest(members: Mapping[str, _Member],
                            manifest: Mapping[str, Any]) -> list[Finding]:
    """Check the archive's own manifest against the members it ships."""
    findings: list[Finding] = []
    listed = {f["path"]: f for f in manifest.get("files") or ()}
    for path, entry in sorted(listed.items()):
        got = members.get(path)
        if got is None:
            findings.append(Finding("error", "archive.listed_absent", path,
                                    "the manifest lists it; the archive has no "
                                    "such member"))
        elif got.sha256 != entry.get("sha256"):
            findings.append(Finding(
                "error", "archive.hash", path,
                f"the member hashes to {got.sha256[:16]}..., the manifest says "
                f"{str(entry.get('sha256'))[:16]}..."))
        elif len(got.data) != entry.get("bytes"):
            findings.append(Finding("error", "archive.size", path, "size differs"))
    for path in sorted(set(members) - set(listed) - {"file_manifest.json"}):
        findings.append(Finding("error", "archive.unlisted", path,
                                "shipped but not in the archive's own manifest"))
    return findings


# ==========================================================================
# the import
# ==========================================================================

def import_archive(zip_path: str | Path, reference_dir: str | Path | None = None,
                   *, write: bool = False) -> dict[str, Any]:
    """Ingest the archive into ``<reference_dir>/bundle``, or report what would be.

    Returns the report: the archive's hash, the per-disposition counts, the
    findings of every check, and the manifest that would be written. With
    ``write`` false nothing is created, which is the default because an ingest
    that cannot first be read is an ingest nobody reviewed.
    """
    base = Path(reference_dir) if reference_dir is not None else default_reference_dir()
    out = base / BUNDLE_DIRNAME
    members, manifest = read_archive(zip_path)
    archive_sha = hashlib.sha256(Path(zip_path).read_bytes()).hexdigest()

    findings = verify_archive_manifest(members, manifest)
    if archive_sha != ARCHIVE_SHA256:
        findings.append(Finding(
            "warning", "archive.another_delivery", Path(zip_path).name,
            f"this archive hashes to {archive_sha[:16]}..., not the "
            f"{ARCHIVE_SHA256[:16]}... this importer was run on; the layout "
            f"checks still apply but the ingest is of different bytes"))

    # the spreadsheet must be the one already committed
    workbook = members.get(_SOURCE_WORKBOOK)
    committed = base / "source" / _SOURCE_WORKBOOK
    if workbook is None:
        findings.append(Finding("error", "archive.no_workbook", _SOURCE_WORKBOOK,
                                "the archive ships no spreadsheet"))
    elif committed.is_file():
        have = hashlib.sha256(committed.read_bytes()).hexdigest()
        if have != workbook.sha256:
            findings.append(Finding(
                "error", "archive.workbook_differs", _SOURCE_WORKBOOK,
                f"the archive's spreadsheet ({workbook.sha256[:16]}...) is not "
                f"the committed one ({have[:16]}...): these are two different "
                f"deliveries and the tables under tables/ describe the other one"))

    findings += _check_dropped(members)
    findings += coordinate_agreement(members, base)

    entries: dict[str, Any] = {}
    for path, member in sorted(members.items()):
        if path == "file_manifest.json":
            continue
        where = disposition_for(path, member.kind)
        entries[path] = {
            "sha256": member.sha256, "bytes": len(member.data),
            "kind": member.kind, "disposition": where,
            **({"source_url": member.source_url} if member.source_url else {}),
            **({"dropped_because": DROPPED_DUPLICATES[path]}
               if where == "dropped_duplicate" else {}),
        }
    counts: dict[str, int] = {}
    sizes: dict[str, int] = {}
    for info in entries.values():
        counts[info["disposition"]] = counts.get(info["disposition"], 0) + 1
        sizes[info["disposition"]] = sizes.get(info["disposition"], 0) + info["bytes"]

    document = {
        "archive": ARCHIVE_NAME,
        "archive_sha256": archive_sha,
        "archive_audit_date": manifest.get("audit_date"),
        "archive_root": ARCHIVE_ROOT,
        "disposition_counts": dict(sorted(counts.items())),
        "disposition_bytes": dict(sorted(sizes.items())),
        "note": ("committed and committed_upstream members are in this "
                 "directory; pinned members are not committed and are "
                 "re-fetchable from their source_url, with the hash here the "
                 "one this ingest read. See ARCHIVE.md."),
        "members": entries,
    }

    if write:
        if [f for f in findings if f.severity == "error"]:
            raise BundleError(
                "the archive did not pass its checks, so nothing was written:\n"
                + "\n".join(f"  {f}" for f in findings if f.severity == "error"))
        out.mkdir(parents=True, exist_ok=True)
        for path, info in entries.items():
            if info["disposition"] not in ("committed", "committed_upstream"):
                continue
            target = out / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(members[path].data)
        (out / ARCHIVE_MANIFEST).write_text(
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8")
        findings += cross_check_tables(out, base / "tables")

    return {"report": document, "findings": findings,
            "written": str(out) if write else None}


def _check_dropped(members: Mapping[str, _Member]) -> list[Finding]:
    """A member dropped as a duplicate must actually be one."""
    findings: list[Finding] = []
    path = "data/structure_audit_raw.json"
    if path not in members:
        return findings
    try:
        aggregate = json.loads(members[path].data.decode("utf-8"))
    except ValueError as exc:
        return [Finding("error", "archive.dropped_unreadable", path, str(exc))]
    if not isinstance(aggregate, list):
        return [Finding("error", "archive.dropped_shape", path,
                        "expected a list of per-structure audits")]
    for item in aggregate:
        pid = (item or {}).get("pdb_id")
        per = members.get(f"structures/{pid}/audit_metadata.json")
        if per is None:
            findings.append(Finding(
                "error", "archive.dropped_not_duplicate", path,
                f"the aggregate holds {pid}, which has no per-structure file, so "
                f"dropping the aggregate would lose it"))
            continue
        if json.loads(per.data.decode("utf-8")) != item:
            findings.append(Finding(
                "error", "archive.dropped_not_duplicate", path,
                f"the aggregate's {pid} differs from "
                f"structures/{pid}/audit_metadata.json, so the aggregate is not "
                f"a duplicate and may not be dropped"))
    return findings


def coordinate_agreement(members: Mapping[str, _Member],
                         reference_dir: str | Path) -> list[Finding]:
    """Compare the archive's coordinate files with this project's own pins.

    The interesting outcome is agreement: the compiler and this project fetched
    the same 19 entries from the RCSB independently, and the bytes are the same.
    A disagreement would mean one of the two retrievals caught a revision the
    other did not, which is a fact about the entry and is reported as such.
    """
    pins_path = Path(reference_dir) / "coordinates" / "coordinates.manifest.json"
    if not pins_path.is_file():
        return [Finding("info", "coordinates.not_pinned_here", str(pins_path),
                        "no coordinate pins to compare the archive against")]
    pins = json.loads(pins_path.read_text(encoding="utf-8")).get("files") or {}
    findings: list[Finding] = []
    agree = 0
    for pid, pin in sorted(pins.items()):
        member = members.get(f"structures/{pid}/{pid}.cif")
        if member is None:
            findings.append(Finding("warning", "coordinates.absent_from_archive", pid,
                                    "this project pins it; the archive ships no "
                                    "such coordinate file"))
        elif member.sha256 != pin["sha256"]:
            findings.append(Finding(
                "error", "coordinates.archive_differs", pid,
                f"the archive's file hashes to {member.sha256[:16]}..., this "
                f"project pinned {pin['sha256'][:16]}.... Two retrievals of one "
                f"entry disagree: the RCSB revised it between them, and which "
                f"revision the audit's grades describe has to be settled before "
                f"either is used"))
        else:
            agree += 1
    if agree:
        findings.append(Finding(
            "info", "coordinates.independently_confirmed", f"{agree} entries",
            f"{agree} coordinate files are byte-identical to the ones this "
            f"project fetched from the RCSB independently; two retrievals by two "
            f"parties agree exactly"))
    return findings


# ==========================================================================
# the two exports of one table
# ==========================================================================

def _normalise(value: Any) -> str:
    """One convention for the three the two exports differ in.

    A JSON list against a joined string, ``=`` against ``eq``, a boolean against
    ``1``/``0``. These are spellings of one value. Anything else that differs is
    a disagreement and is reported.
    """
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return "; ".join(str(v) for v in value)
    text = "" if value is None else str(value).strip()
    return "eq" if text == "=" else text


def _agree(a: Any, b: Any) -> bool:
    sa, sb = _normalise(a), _normalise(b)
    if sa == sb:
        return True
    if not sa or not sb:
        return False
    try:
        fa, fb = float(sa), float(sb)
    except ValueError:
        return False
    return abs(fa - fb) <= 1e-12 * max(1.0, abs(fa), abs(fb))


def _records(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping) and isinstance(payload.get("records"), list):
        return payload["records"]
    raise BundleError(f"{path.name}: expected a list of records")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(r) for r in csv.DictReader(handle)]


def cross_check_tables(bundle: str | Path, tables: str | Path) -> list[Finding]:
    """Compare the archive's JSON export with the spreadsheet's CSV conversion.

    Both describe the same five tables. Every field they share is compared; a
    field only one of them carries is not a disagreement (the JSON export adds
    ``assay_group``, ``reaction_direction``, ``sequence_id`` and
    ``evidence_status``, and the spreadsheet is the thing those were derived
    alongside). The count of comparisons is reported so that "they agree" is a
    statement with a size.
    """
    bundle, tables = Path(bundle), Path(tables)
    findings: list[Finding] = []
    compared = 0
    for name, json_path, csv_name, key in _CROSS_TABLES:
        jp, cp = bundle / json_path, tables / csv_name
        if not jp.is_file() or not cp.is_file():
            findings.append(Finding("error", "cross.table_missing", name,
                                    f"need both {json_path} and {csv_name}"))
            continue
        left = {str(r[key]): r for r in _records(jp)}
        right = {str(r[key]): r for r in _csv_rows(cp)}
        if set(left) != set(right):
            findings.append(Finding(
                "error", "cross.keys", name,
                f"only in the archive {sorted(set(left) - set(right))[:4]}; "
                f"only in the spreadsheet {sorted(set(right) - set(left))[:4]}"))
        for k in sorted(set(left) & set(right)):
            for field, csv_value in right[k].items():
                if field not in left[k]:
                    continue
                compared += 1
                if not _agree(left[k][field], csv_value):
                    findings.append(Finding(
                        "error", "cross.value", f"{name}[{k}].{field}",
                        f"the archive's JSON says {left[k][field]!r}, the "
                        f"spreadsheet's CSV says {csv_value!r}"))
    findings.append(Finding(
        "info", "cross.compared", "five tables",
        f"{compared} shared fields compared between the archive's JSON export "
        f"and the spreadsheet's CSV conversion"))
    return findings


# ==========================================================================
# verifying what was committed
# ==========================================================================

def verify_bundle(reference_dir: str | Path | None = None) -> list[Finding]:
    """Check the committed bundle against ``archive.manifest.json``. Offline.

    A committed member whose bytes have changed is an error. A pinned member is
    *expected* to be absent; one that is present is checked against its hash,
    because a file somebody dropped in beside the manifest should be the file
    the manifest describes.
    """
    out = bundle_dir(reference_dir)
    path = out / ARCHIVE_MANIFEST
    if not path.is_file():
        return [Finding("error", "bundle.no_manifest", str(out),
                        f"{ARCHIVE_MANIFEST} is not there, so nothing about "
                        f"these files can be shown to be what was delivered")]
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [Finding("error", "bundle.unreadable", ARCHIVE_MANIFEST, str(exc))]

    findings: list[Finding] = []
    members = document.get("members") or {}
    for rel, info in sorted(members.items()):
        target = out / rel
        expected = info.get("sha256")
        committed = info["disposition"] in ("committed", "committed_upstream")
        if not target.is_file():
            if committed:
                findings.append(Finding("error", "bundle.missing", rel,
                                        "the manifest commits it; it is not there"))
            continue
        got = hashlib.sha256(target.read_bytes()).hexdigest()
        if got != expected:
            findings.append(Finding(
                "error", "bundle.changed", rel,
                f"sha256 is {got[:16]}..., the manifest says {str(expected)[:16]}..."))
        elif not committed:
            findings.append(Finding(
                "info", "bundle.pinned_present", rel,
                f"a {info['disposition']} member is present and matches its hash"))

    listed = set(members)
    for found in sorted(p for p in out.rglob("*") if p.is_file()):
        rel = found.relative_to(out).as_posix()
        if rel in (ARCHIVE_MANIFEST, "ARCHIVE.md") or rel in listed:
            continue
        if rel in DERIVED_FILES:
            findings.append(Finding(
                "info", "bundle.derived", rel,
                f"written by this package, not delivered in the archive; "
                f"regenerate with `{DERIVED_FILES[rel]}`"))
            continue
        findings.append(Finding(
            "error", "bundle.unlisted", rel,
            "in the bundle, but neither an archive member nor a file this "
            "package writes; it came from somewhere nobody recorded"))
    return findings
