"""Reading the KRED calibration reference workbook with the standard library only.

The reference set arrives as an ``.xlsx``. A spreadsheet is a poor thing to
build a pipeline on -- a cell can hold a formula whose cached value is stale, a
number formatted so that what is displayed is not what is stored, a hidden
sheet -- so the workbook is converted once, deterministically, into plain CSV
that a diff can read, and the original is kept beside it, hashed, so the
conversion can be repeated and compared.

WHAT THE CONVERSION DOES NOT DO
===============================
It does not interpret, round, convert or correct anything. A numeric cell is
written as the text the file stores (``0.00014000000000000001`` stays that),
an empty cell stays empty, and the cached result of a formula is written as
the value while the formula text is saved separately, so a later reader can
see which columns were computed rather than typed. All interpretation happens
in :mod:`eagent.eval.kred_reference`, where it can be tested.

SAFETY
======
The workbook is untrusted input. No formula is evaluated. Any part of the
package that declares a DTD or an entity is refused (the standard library's
XML parser expands entities, and a spreadsheet has no reason to define one),
and the package is bounded in the size it may expand to.
"""

from __future__ import annotations

import csv
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import EAgentError

__all__ = [
    "WorkbookError",
    "SHEET_TABLES",
    "WorkbookTable",
    "read_workbook",
    "write_tables",
]


class WorkbookError(EAgentError):
    """The workbook is not the one this importer was written for."""


#: sheet name -> (file stem, the header the importer was written against).
#: Every sheet must be present and every header must match: a new workbook
#: version that renames a column or adds a sheet needs a person to decide what
#: that means, not an importer that carries on.
SHEET_TABLES: dict[str, tuple[str, tuple[str, ...]]] = {
    "使用说明": ("usage_notes", ("topic", "guidance")),
    "结构清单": ("structures", (
        "pdb_id", "enzyme", "variant", "geometry_use", "resolution_A",
        "bound_reaction_ligand", "experimental_state", "scoring_auth_chain",
        "selected_altloc", "reaction_ligand_ccd", "cofactor_state",
        "selected_ligand_cofactor_RSCC_summary", "assembly_oligomeric_count",
        "linked_label_ids", "selection_notes", "pdb_url", "validation_url",
        "pdb_sequence_id")),
    "活性标签": ("kinetics", (
        "label_id", "enzyme", "variant", "substrate", "record_status",
        "kcat_original", "kcat_unit", "kcat_uncertainty_original",
        "Km_original", "Km_unit", "Km_comparator", "Km_uncertainty_original",
        "efficiency_original", "efficiency_unit",
        "efficiency_uncertainty_original", "kcat_s_1", "Km_value_or_bound_M",
        "efficiency_reported_M_1_s_1", "efficiency_ratio_derived_M_1_s_1",
        "pH", "temperature_C_nominal", "assay_cofactor", "label_qualification",
        "default_use", "experimental_complex_pdb_ids",
        "experimental_protein_pdb_ids", "template_pdb_ids", "source_id",
        "source_url", "source_location", "quality_flags")),
    "测定条件": ("assay_conditions", (
        "label_id", "pH", "temperature_C_nominal", "cofactor", "buffer",
        "readout", "protocol_summary", "source_url")),
    "选择性": ("selectivity", (
        "selectivity_id", "enzyme", "variant", "substrate", "er_S_over_R",
        "kinetic_label_id", "kinetic_variant_identity_match",
        "conditions_status", "note", "source_url")),
    "配体验证": ("ligand_validation", (
        "pdb_id", "auth_chain", "auth_residue_number", "ccd", "altloc", "RSCC",
        "RSR", "average_occupancy", "average_B_A2", "raw_clash_child_count",
        "geometry_child_count", "warning", "source_url")),
    "来源": ("sources", (
        "source_id", "title", "doi", "location", "verification",
        "access_reuse", "url")),
    "辅助与扩展": ("auxiliary", (
        "item", "value", "unit", "label_type", "use", "source")),
}

_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS = {"m": _MAIN, "r": _REL}

#: A real workbook of this kind is under 100 KB. Anything that expands beyond
#: this is not one.
MAX_PART_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024


@dataclass
class WorkbookTable:
    """One sheet's table, as the file stores it."""

    sheet: str
    stem: str
    table_ref: str
    leading_rows: list[list[str]]
    header: list[str]
    rows: list[list[str]]
    formulas: dict[str, str] = field(default_factory=dict)


def _column_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref)
    if not letters:
        raise WorkbookError(f"unreadable cell reference {ref!r}")
    n = 0
    for ch in letters.group(0):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _row_number(ref: str) -> int:
    digits = re.search(r"\d+", ref)
    if not digits:
        raise WorkbookError(f"unreadable cell reference {ref!r}")
    return int(digits.group(0))


def _split_ref(ref: str) -> tuple[int, int, int, int]:
    """``A4:R23`` -> (first_row, first_col, last_row, last_col), 0-based columns."""
    start, _, end = ref.partition(":")
    if not end:
        end = start
    return (_row_number(start), _column_index(start),
            _row_number(end), _column_index(end))


def _parse(data: bytes, what: str) -> ET.Element:
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise WorkbookError(
            f"{what} declares a DTD or an entity; a spreadsheet has no reason "
            f"to, and the parser would expand it")
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise WorkbookError(f"{what} is not well-formed XML: {exc}") from exc


def _read_part(archive: zipfile.ZipFile, name: str, budget: list[int]) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > MAX_PART_BYTES:
        raise WorkbookError(f"{name} expands to {info.file_size} bytes")
    budget[0] += info.file_size
    if budget[0] > MAX_TOTAL_BYTES:
        raise WorkbookError("the package expands beyond the allowed total")
    return archive.read(name)


def read_workbook(path: str | Path) -> list[WorkbookTable]:
    """Read every expected table from the workbook, or refuse.

    Only the cached value of a formula cell is read. A formula cell with no
    cached value is refused: its value is whatever a spreadsheet program would
    have computed, and this reader does not compute.
    """
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise WorkbookError(f"{path} is not a readable .xlsx: {exc}") from exc
    budget = [0]
    names = set(archive.namelist())
    for required in ("xl/workbook.xml", "xl/_rels/workbook.xml.rels"):
        if required not in names:
            raise WorkbookError(f"{path} has no {required}")
    if any(n.startswith("xl/externalLinks") or n.endswith("vbaProject.bin")
           for n in names):
        raise WorkbookError(
            "the workbook carries external links or macros, which this "
            "importer does not follow")

    workbook = _parse(_read_part(archive, "xl/workbook.xml", budget), "workbook.xml")
    rels = _parse(_read_part(archive, "xl/_rels/workbook.xml.rels", budget),
                  "workbook.xml.rels")
    targets = {r.get("Id"): r.get("Target") for r in rels}
    shared: list[str] = []
    if "xl/sharedStrings.xml" in names:
        sst = _parse(_read_part(archive, "xl/sharedStrings.xml", budget),
                     "sharedStrings.xml")
        for si in sst.findall("m:si", _NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{_MAIN}}}t")))

    sheets = workbook.find("m:sheets", _NS)
    if sheets is None:
        raise WorkbookError("workbook.xml lists no sheets")
    found: dict[str, tuple[str, str]] = {}
    for sheet in sheets:
        name = sheet.get("name") or ""
        if sheet.get("state", "visible") != "visible":
            raise WorkbookError(f"sheet {name!r} is hidden; a hidden sheet is "
                                f"content nobody reviewed")
        rid = sheet.get(f"{{{_REL}}}id")
        target = targets.get(rid or "")
        if target is None:
            raise WorkbookError(f"sheet {name!r} has no part")
        found[name] = (f"xl/{target.lstrip('/').removeprefix('xl/')}", rid or "")
    unexpected = sorted(set(found) - set(SHEET_TABLES))
    missing = sorted(set(SHEET_TABLES) - set(found))
    if unexpected or missing:
        raise WorkbookError(
            f"this importer was written for sheets {sorted(SHEET_TABLES)}; "
            f"unexpected {unexpected}, missing {missing}")

    tables: list[WorkbookTable] = []
    for sheet_name, (stem, expected_header) in SHEET_TABLES.items():
        part, _ = found[sheet_name]
        root = _parse(_read_part(archive, part, budget), part)
        table_ref = _table_ref_for(archive, part, budget)
        first_row, first_col, last_row, last_col = _split_ref(table_ref)

        cells: dict[int, dict[int, str]] = {}
        formulas: dict[str, str] = {}
        for row in root.iter(f"{{{_MAIN}}}row"):
            for c in row.findall("m:c", _NS):
                ref = c.get("r") or ""
                kind = c.get("t")
                value = c.find("m:v", _NS)
                formula = c.find("m:f", _NS)
                if kind == "inlineStr":
                    inline = c.find("m:is", _NS)
                    text = ("".join(t.text or "" for t in inline.iter(f"{{{_MAIN}}}t"))
                            if inline is not None else "")
                elif kind == "s":
                    text = shared[int(value.text)] if value is not None else ""
                elif value is not None:
                    text = value.text or ""
                else:
                    text = ""
                if formula is not None:
                    if value is None:
                        raise WorkbookError(
                            f"{sheet_name}!{ref} is a formula with no cached "
                            f"value; this reader does not compute")
                    formulas[ref] = formula.text or ""
                cells.setdefault(_row_number(ref), {})[_column_index(ref)] = text

        def row_values(number: int, width: int | None = None) -> list[str]:
            cols = cells.get(number, {})
            top = (max(cols) + 1) if cols else 0
            n = width if width is not None else top
            return [cols.get(i, "") for i in range(n)]

        leading = [row_values(n) for n in range(1, first_row)]
        leading = [r for r in leading if any(v.strip() for v in r)]
        width = last_col - first_col + 1
        header = row_values(first_row, last_col + 1)[first_col:]
        if tuple(header) != expected_header:
            raise WorkbookError(
                f"sheet {sheet_name!r}: the header is not the one this "
                f"importer was written for.\n  expected {list(expected_header)}"
                f"\n  found    {header}")
        body: list[list[str]] = []
        for number in range(first_row + 1, last_row + 1):
            values = row_values(number, last_col + 1)[first_col:first_col + width]
            if any(v.strip() for v in values):
                body.append(values)
        tables.append(WorkbookTable(
            sheet=sheet_name, stem=stem, table_ref=table_ref,
            leading_rows=leading, header=header, rows=body, formulas=formulas))
    return tables


def _table_ref_for(archive: zipfile.ZipFile, sheet_part: str,
                   budget: list[int]) -> str:
    """The declared range of the (single) Excel table on a sheet."""
    rels_name = sheet_part.replace("worksheets/", "worksheets/_rels/") + ".rels"
    if rels_name not in archive.namelist():
        raise WorkbookError(f"{sheet_part} has no table; refusing to guess "
                            f"where the data is")
    rels = _parse(_read_part(archive, rels_name, budget), rels_name)
    table_targets = [r.get("Target") for r in rels
                     if (r.get("Type") or "").endswith("/table")]
    if len(table_targets) != 1:
        raise WorkbookError(f"{sheet_part} has {len(table_targets)} tables; "
                            f"expected exactly one")
    target = table_targets[0] or ""
    part = ("xl/" + target.replace("../", "")) if target.startswith("..") \
        else target.lstrip("/")
    table = _parse(_read_part(archive, part, budget), part)
    ref = table.get("ref")
    if not ref:
        raise WorkbookError(f"{part} declares no range")
    return ref


def write_tables(tables: list[WorkbookTable], out_dir: str | Path) -> list[Path]:
    """Write one CSV per table plus the notes that sit around them.

    Deterministic: the same workbook always produces the same bytes (UTF-8, ``\\n``
    line ends, no BOM, JSON with sorted keys), which is what lets a test convert
    the stored workbook again and compare it with what is committed.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    notes: dict[str, Any] = {}
    formulas: dict[str, Any] = {}
    for table in tables:
        path = out / f"{table.stem}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(table.header)
            writer.writerows(table.rows)
        written.append(path)
        notes[table.stem] = {"sheet": table.sheet, "table_ref": table.table_ref,
                             "leading_rows": table.leading_rows,
                             "n_rows": len(table.rows)}
        if table.formulas:
            formulas[table.stem] = dict(sorted(table.formulas.items()))
    for name, payload in (("sheet_notes.json", notes), ("formulas.json", formulas)):
        path = out / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                   sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    return written
