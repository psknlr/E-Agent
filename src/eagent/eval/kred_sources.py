"""Checking the workbook's kinetic numbers against the tables they came from.

The workbook says, of its 28 kinetic records, that each was read from a named
table of a named paper or from a curated database. That is a claim made by the
compiler. This module re-reads the tables and compares, so the claim has a
second reader and a record of what each reader saw.

IT DOES NOT FETCH ANYTHING
==========================
The documents are read from a directory the caller filled. None of these
publishers is a registered route in this project, and a scraper that reached out
to one by itself would be the thing the connector layer exists to prevent. The
caller downloads the six documents (the URLs are in :data:`DOCUMENTS`), this
module hashes the bytes it is given, and the record says which bytes were read.
A page that is re-rendered tomorrow has another hash and the record still says
what was seen today.

WHAT "MATCHES" MEANS, AND WHAT IT DOES NOT
==========================================
A record *matches* when each number the workbook gives for a quantity equals the
number printed in the source, compared as numbers (``0.014`` equals
``0.0140``), together with the printed ``±`` term where the workbook keeps one.
Nothing is rounded to make it agree. A quantity the workbook gives and the
source does not print is ``not_found``, not ``matches``. A quantity this module
did not read at all -- the two LbADH efficiencies came from a thesis reprint
nobody here opened -- is ``not_checked`` and says why.

Six parsers, one per document shape. They are deliberately narrow: each knows the
layout of one table in one document and refuses (``not_found``) when the layout
is not there, rather than guessing at a neighbouring cell.
"""

from __future__ import annotations

import hashlib
import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..errors import EAgentError
from .kred_reference import ReferenceSet

__all__ = [
    "SourceVerificationError",
    "DOCUMENTS",
    "Document",
    "Cell",
    "jats_tables",
    "html_tables",
    "si_rows",
    "brenda_entries",
    "parse_value",
    "verify_sources",
    "summarise",
]


class SourceVerificationError(EAgentError):
    """A document is missing or is not the shape a parser was written for."""


@dataclass(frozen=True)
class Document:
    key: str
    filename: str
    kind: str
    url: str
    what: str
    #: Whether the document is the paper's (or its SI's) own table. A curated
    #: database's extraction of a paper is *secondary*: a second reader of the
    #: paper, not the paper.
    primary: bool = True


#: Where each document comes from and what it is read for. The caller fetches
#: them; ``filename`` is what this module looks for in the directory.
DOCUMENTS: tuple[Document, ...] = (
    Document("Ssal2024", "PMC10902378.xml", "jats",
             "https://europepmc.org/api/service/rest/PMC10902378/fullTextXML",
             "Table 2: apparent KM, kcat, kcat/KM of Ssal-KRED WT and M1-M6"),
    Document("HBDH2020", "PMC7773212.xml", "jats",
             "https://europepmc.org/api/service/rest/PMC7773212/fullTextXML",
             "Table 4: steady-state parameters of the four HBDH mutants"),
    Document("PNAS2015", "PMC4697376.html", "html",
             "https://pmc.ncbi.nlm.nih.gov/articles/PMC4697376/",
             "Table 1: er and kcat/KM of the LkKRED variants (page HTML)"),
    Document("HBDH2018_SI", "hbdh2018_si.txt", "si_text",
             "https://doi.org/10.1021/acs.biochem.8b01099.s001",
             "Tables S1-S4, 283 K rows (the SI PDF through `pdftotext -layout`)"),
    Document("TRII_BRENDA", "brenda_654707.html", "brenda",
             "https://www.brenda-enzymes.org/literature.php?e=1.1.1.236&r=654707",
             "tropinone KM and turnover number, pH 7.5, 15 C", primary=False),
    Document("LbADH_BRENDA", "brenda_675348.html", "brenda",
             "https://brenda-enzymes.info/literature.php?e=1.1.1.2&r=675348",
             "acetophenone KM and turnover number, WT and G37D, 30 C, pH 7.0",
             primary=False),
)


# ==========================================================================
# numbers as printed
# ==========================================================================

_MINUS = str.maketrans({"−": "-", "–": "-", "—": "-"})


@dataclass(frozen=True)
class Cell:
    """A printed value: ``30 ± 1``, ``<5400``, ``ND``, ``41``."""

    text: str
    value: float | None
    error: float | None
    comparator: str        # "", "<", ">"
    nd: bool

    def __str__(self) -> str:
        return self.text


_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def parse_value(text: str) -> Cell:
    """Read a printed table cell. ``ND`` is not a number and has no value."""
    raw = " ".join(str(text).translate(_MINUS).split())
    if re.fullmatch(r"(?i)n\.?d\.?", raw.split()[0] if raw else ""):
        return Cell(raw, None, None, "", True)
    comparator = ""
    body = raw
    if body[:1] in "<>":
        comparator, body = body[0], body[1:].strip()
    # "a ± b" (the sign may have been dropped by a text extractor: "a  b")
    m = re.match(rf"^({_NUM})\s*(?:±|\+/-|\+-)?\s*({_NUM})?(?!\d)", body)
    if not m:
        return Cell(raw, None, None, comparator, False)
    value = float(m.group(1))
    error = float(m.group(2)) if m.group(2) is not None and (
        "±" in body or "+/-" in body or "+-" in body or "  " in body) else None
    return Cell(raw, value, error, comparator, False)


# ==========================================================================
# the parsers
# ==========================================================================

def _text(el: ET.Element) -> str:
    return " ".join("".join(el.itertext()).split())


def jats_tables(xml: bytes | str) -> dict[str, list[list[str]]]:
    """``{table label: rows}`` of every ``<table-wrap>`` of a JATS article."""
    if isinstance(xml, bytes):
        if b"<!DOCTYPE" in xml and b"<!ENTITY" in xml:
            raise SourceVerificationError("the document declares an entity; refused")
        xml = xml.decode("utf-8", errors="replace")
    if "<!ENTITY" in xml:
        raise SourceVerificationError("the document declares an entity; refused")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise SourceVerificationError(f"not well-formed XML: {exc}") from exc
    out: dict[str, list[list[str]]] = {}
    for wrap in root.iter("table-wrap"):
        label = wrap.find("label")
        name = _text(label) if label is not None else ""
        rows = [[_text(c) for c in tr if c.tag in ("th", "td")] for tr in wrap.iter("tr")]
        if name:
            out[name] = rows
    return out


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag == "table":
            self._depth += 1
            if self._depth == 1:
                self.tables.append([])
        elif tag == "tr" and self._depth:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self.tables:
                self.tables[-1].append(self._row)
            self._row = None
        elif tag == "table" and self._depth:
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def html_tables(page: bytes | str) -> list[list[list[str]]]:
    """Every top-level ``<table>`` of an HTML page, as rows of cell text."""
    text = page.decode("utf-8", errors="replace") if isinstance(page, bytes) else page
    parser = _TableParser()
    parser.feed(text)
    return parser.tables


_SI_ROW = re.compile(r"^\s*283 K\s+(?P<rest>.*)$")


def si_rows(text: str, table: str) -> dict[str, float | None]:
    """The 283 K row of one SI table, as ``{quantity: number}``.

    ``text`` is the SI through ``pdftotext -layout``. The plus-minus survives
    extraction as the Symbol-font private-use glyph U+F0B1 and the exponent as a
    superscript flattened onto the ten (``105`` is 10^5), so the row is read as
    numbers in order: kcat, its error, KM, its error, KM(NADH), its error, then
    the efficiency as ``mantissa ± error x 10<exponent>``. Returns an empty dict
    when the table or the row is not where it is expected.
    """
    # a page break leaves a form feed at the start of the heading's line
    start = re.search(rf"^[ \t\f]*Table {re.escape(table)}\.", text, re.MULTILINE)
    if not start:
        return {}
    block = text[start.start():start.start() + 3000]
    for line in block.splitlines():
        m = _SI_ROW.match(line)
        if not m:
            continue
        rest = m.group("rest").replace("\uf0b1", "\u00b1")
        nums = re.findall(_NUM, rest.replace("\u00d7", " "))
        eff = re.search(rf"({_NUM})\s*\u00b1?\s*({_NUM})\s*\u00d7\s*10(\d)", rest)
        if len(nums) < 6 or eff is None:
            return {}
        exponent = int(eff.group(3))
        return {"kcat": float(nums[0]), "kcat_error": float(nums[1]),
                "km": float(nums[2]), "km_error": float(nums[3]),
                "efficiency": float(eff.group(1)) * 10 ** exponent,
                "efficiency_error": float(eff.group(2)) * 10 ** exponent}
    return {}


@dataclass(frozen=True)
class BrendaEntry:
    section: str            # "km" | "kcat"
    value: float
    lines: tuple[str, ...]  # what follows the number: substrate, then conditions

    def text(self) -> str:
        return " | ".join(self.lines)


def brenda_entries(page: bytes | str) -> list[BrendaEntry]:
    """The KM and turnover-number entries of a BRENDA literature page.

    The page is flattened to lines; a bare number opens an entry and the lines
    up to the next number or section header are its substrate and conditions.
    Only the two sections this project reads are returned.
    """
    raw = page.decode("utf-8", errors="replace") if isinstance(page, bytes) else page
    raw = re.sub(r"<script.*?</script>|<style.*?</style>", "", raw, flags=re.S)
    raw = re.sub(r"<br\s*/?>|</tr>|</p>|</div>|</li>|</td>|</th>", "\n", raw)
    raw = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    lines = [" ".join(l.split()) for l in raw.splitlines() if l.strip()]
    entries: list[BrendaEntry] = []
    section = ""
    current: tuple[str, float] | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current, buffer
        if current is not None and section in ("km", "kcat"):
            entries.append(BrendaEntry(current[0], current[1], tuple(buffer)))
        current, buffer = None, []

    for line in lines:
        if line.startswith("KM Value [mM]"):
            flush(); section = "km"; continue
        if line.startswith("Turnover Number [1/s]"):
            flush(); section = "kcat"; continue
        if re.match(r"^(KM Value|Turnover Number) (Minimum|Maximum)", line):
            continue                      # column captions of the section just opened
        if re.match(r"^((Temperature|pH) (Optimum|Range|Stability)|Specific Activity|"
                    r"Ki Value|IC50|Application|Cofactor\b|Reference|Use of this|Release)",
                    line):
            flush(); section = "other"; continue
        if re.fullmatch(_NUM, line) and section in ("km", "kcat"):
            flush()
            current = (section, float(line))
            continue
        if current is not None:
            buffer.append(line)
    flush()
    return entries


# ==========================================================================
# the comparison
# ==========================================================================

@dataclass
class Check:
    """One quantity of one record, compared with one source cell."""

    quantity: str
    workbook: str
    source: str
    status: str        # matches | mismatch | not_found | not_checked
    locator: str
    note: str = ""
    document: str = ""
    primary: bool = True

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "quantity": self.quantity, "workbook": self.workbook,
            "source": self.source, "status": self.status, "locator": self.locator,
            "document": self.document, "primary": self.primary}
        if self.note:
            d["note"] = self.note
        return d


def _num(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


def _compare(quantity: str, workbook: str, cell: Cell | None, locator: str,
             *, with_error: bool = False, wb_error: str = "", comparator: str = "") -> Check:
    if cell is None:
        return Check(quantity, workbook, "", "not_found", locator,
                     "the source row or column was not found where expected")
    if cell.nd:
        wb_nd = workbook == ""
        return Check(quantity, workbook or "ND", cell.text,
                     "matches" if wb_nd else "mismatch", locator,
                     "the source prints ND; the workbook must hold no value")
    ok = _same(_num(workbook), cell.value) and cell.comparator == comparator
    if with_error and wb_error != "":
        ok = ok and _same(_num(wb_error), cell.error)
    shown = f"{workbook}" + (f" ± {wb_error}" if with_error and wb_error != "" else "")
    return Check(quantity, ("<" if comparator else "") + shown, cell.text,
                 "matches" if ok else "mismatch", locator)


def _row(table: Sequence[Sequence[str]], first: str) -> list[str] | None:
    for r in table:
        if r and r[0].strip().rstrip("a").strip() == first or (r and r[0].strip() == first):
            return list(r)
    return None


def _cell_at(row: Sequence[str] | None, index: int) -> Cell | None:
    if row is None or index >= len(row):
        return None
    return parse_value(row[index])


def verify_sources(rs: ReferenceSet, docs_dir: str | Path) -> dict[str, Any]:
    """Compare every kinetic record with the document it names, and report.

    Returns the verification record: for each document its hash and size, for
    each kinetic record each quantity's comparison, and a summary. Raises
    :class:`SourceVerificationError` if a document is missing, because a record
    that silently skipped one would read as having checked it.
    """
    base = Path(docs_dir)
    docs: dict[str, dict[str, Any]] = {}
    data: dict[str, bytes] = {}
    for d in DOCUMENTS:
        path = base / d.filename
        if not path.is_file():
            raise SourceVerificationError(
                f"{path} is missing. Download {d.url} and save it there "
                f"({d.what}); this module reads documents, it does not fetch them")
        data[d.key] = path.read_bytes()
        docs[d.key] = {"filename": d.filename, "url": d.url, "kind": d.kind,
                       "primary": d.primary,
                       "bytes": len(data[d.key]),
                       "sha256": hashlib.sha256(data[d.key]).hexdigest(),
                       "read_for": d.what}

    ssal = jats_tables(data["Ssal2024"]).get("Table 2", [])
    hbdh = jats_tables(data["HBDH2020"]).get("Table 4", [])
    pnas = next((t for t in html_tables(data["PNAS2015"])
                 if t and t[0] and t[0][0].startswith("Variant")), [])
    si_text = data["HBDH2018_SI"].decode("utf-8", errors="replace")
    brenda = {"TRII_BRENDA": brenda_entries(data["TRII_BRENDA"]),
              "LbADH_BRENDA": brenda_entries(data["LbADH_BRENDA"])}

    results: dict[str, list[Check]] = {}
    by_key = {d.key: d for d in DOCUMENTS}

    def add(label: str, check: Check) -> None:
        results.setdefault(label, []).append(check)

    for rec in rs.kinetics:
        lid = rec.label_id
        if rec.source_id == "Ssal2024":
            variant = rec.variant
            row = _row(ssal, variant) or _row(ssal, variant + "a")
            where = f"Table 2, row {variant}"
            add(lid, _compare("Km", f"{rec.km_original:g}", _cell_at(row, 1), where + ", app. KM [µM]",
                              with_error=True, wb_error=rec.km_uncertainty_original))
            add(lid, _compare("kcat", f"{rec.kcat_original:g}", _cell_at(row, 2),
                              where + ", app. kcat [min-1]", with_error=True,
                              wb_error=rec.kcat_uncertainty_original))
            add(lid, _compare("efficiency", f"{rec.efficiency_original:g}", _cell_at(row, 3),
                              where + ", app. kcat/KM [min-1 mM-1]"))
        elif rec.source_id == "HBDH2020":
            row = _row(hbdh, f"{rec.variant}-{rec.enzyme}")
            where = f"Table 4, row {rec.variant}-{rec.enzyme}"
            add(lid, _compare("kcat", f"{rec.kcat_original:g}", _cell_at(row, 1),
                              where + ", kcat (s-1)", with_error=True,
                              wb_error=rec.kcat_uncertainty_original))
            add(lid, _compare("Km", f"{rec.km_original:g}", _cell_at(row, 2),
                              where + ", KM AcAc (mM)",
                              with_error=rec.km_comparator in ("", "eq"),
                              wb_error=rec.km_uncertainty_original,
                              comparator="" if rec.km_comparator in ("", "eq")
                              else rec.km_comparator))
        elif rec.source_id == "PNAS2015":
            row = _row(pnas, rec.variant)
            index = 3 if rec.label_id.endswith("_1") else 6
            where = (f"Table 1, row {rec.variant}, kcat/KM "
                     f"({'3-oxacyclopentanol' if index == 3 else '3-thiacyclopentanol'})")
            cell = _cell_at(row, index)
            add(lid, _compare("efficiency",
                              "" if rec.record_status == "not_determined"
                              else f"{rec.efficiency_original:g}", cell, where))
        elif rec.source_id == "HBDH2018_SI":
            table = {"6ZZO_PaHBDH_AAE": "S2", "6ZZP_PaHBDH_QT8": "S4",
                     "6ZZQ_AbHBDH_AAE": "S1", "6ZZS_AbHBDH_QT8": "S3"}[lid]
            vals = si_rows(si_text, table)
            where = f"Table {table}, 283 K row"
            for q, wb, key, err in (("kcat", rec.kcat_original, "kcat", "kcat_error"),
                                    ("Km", rec.km_original, "km", "km_error")):
                src = vals.get(key)
                wb_err = float(rec.kcat_uncertainty_original if q == "kcat"
                               else rec.km_uncertainty_original)
                ok = vals and _same(wb, src) and _same(wb_err, vals.get(err))
                add(lid, Check(q, f"{wb:g} ± {wb_err:g}",
                               f"{src:g} ± {vals.get(err):g}" if vals else "",
                               "matches" if ok else ("mismatch" if vals else "not_found"),
                               where))
            ok = vals and _same(rec.efficiency_original, vals.get("efficiency")) \
                and _same(float(rec.efficiency_uncertainty_original), vals.get("efficiency_error"))
            add(lid, Check("efficiency",
                           f"{rec.efficiency_original:g} ± {rec.efficiency_uncertainty_original}",
                           f"{vals['efficiency']:g} ± {vals['efficiency_error']:g}" if vals else "",
                           "matches" if ok else ("mismatch" if vals else "not_found"),
                           where + ", kcat/KM (M-1 s-1)"))
        elif rec.source_id in ("TRII_BRENDA", "LbADH_BRENDA"):
            entries = brenda[rec.source_id]
            substrate = rec.substrate.lower()
            variant_words = ("G37D",) if rec.variant == "G37D" else ("wild-type", "wild-typ")
            cofactor = rec.assay_cofactor.upper()

            def find(section: str, value: float) -> BrendaEntry | None:
                for e in entries:
                    text = e.text().lower()
                    if e.section == section and _same(e.value, value) and substrate in text \
                            and (rec.source_id == "TRII_BRENDA"
                                 or (any(w.lower() in text for w in variant_words)
                                     and f"cofactor {cofactor.lower()}" in text)):
                        return e
                return None

            for q, section, value in (("kcat", "kcat", rec.kcat_original),
                                      ("Km", "km", rec.km_original)):
                hit = find(section, value)
                add(lid, Check(q, f"{value:g}", f"{hit.value:g}  [{hit.text()[:110]}]" if hit else "",
                               "matches" if hit else "not_found",
                               f"BRENDA literature page, {'turnover number' if q == 'kcat' else 'KM value'}"
                               f", {rec.substrate}, {rec.variant}",
                               "BRENDA's own curated extraction of the paper; the paper's "
                               "table was not read"))
            if rec.efficiency_original is not None:
                add(lid, Check("efficiency", f"{rec.efficiency_original:g}", "", "not_checked",
                               "the 2005 paper / Kulishova 2010 Table 4-11",
                               "from a university thesis reprint of the paper's table; "
                               "neither was opened"))
            for q in ("kcat", "Km"):
                err = rec.kcat_uncertainty_original if q == "kcat" else rec.km_uncertainty_original
                if err:
                    add(lid, Check(f"{q}_uncertainty", err, "", "not_checked",
                                   "the 2005 paper / Kulishova 2010 Table 4-11",
                                   "BRENDA's page prints no error terms"))
        else:
            add(lid, Check("all", "", "", "not_checked", rec.source_id,
                           "no parser exists for this source"))

    # selectivity (PNAS2015 er) -- a different label type, read from the same table
    sel_checks: dict[str, list[Check]] = {}
    for sel in rs.selectivity:
        row = _row(pnas, sel.variant.split("_")[0])
        index = 1 if sel.selectivity_id.endswith("_1_er") else 4
        where = f"Table 1, row {sel.variant}, er S/R"
        cell = _cell_at(row, index)
        check = _compare("er_S_over_R", f"{sel.er_s_over_r:g}", cell, where)
        if cell is not None and re.search(r"[*\u2020]", cell.text):
            check.note = ("printed with the table's footnote marker(s); the number matches, "
                          "and what the footnote says about which variant it belongs to is "
                          "the workbook's to carry (see its selectivity note)")
        sel_checks[sel.selectivity_id] = [check]

    source_of = {rec.label_id: rec.source_id for rec in rs.kinetics}
    for lid, checks in results.items():
        doc = by_key.get(source_of[lid])
        for c in checks:
            c.document = doc.key if doc else source_of[lid]
            c.primary = doc.primary if doc else False
    for checks in sel_checks.values():
        for c in checks:
            c.document, c.primary = "PNAS2015", True
    labels = {lid: [c.to_dict() for c in checks] for lid, checks in sorted(results.items())}
    sel_labels = {sid: [c.to_dict() for c in checks] for sid, checks in sorted(sel_checks.items())}
    return {
        "workbook_sha256": _workbook_sha256(rs),
        "verified_by": ("eagent.eval.kred_sources.verify_sources, run by the build "
                        "session on documents downloaded the same day; the values were "
                        "also read by hand before the parsers were written"),
        "documents": docs,
        "kinetic_records": labels,
        "selectivity_records": sel_labels,
        "summary": summarise(labels, sel_labels),
    }


def _workbook_sha256(rs: ReferenceSet) -> str:
    sources = sorted((rs.directory / "source").glob("*.xlsx"))
    return hashlib.sha256(sources[0].read_bytes()).hexdigest() if sources else ""


def summarise(labels: Mapping[str, Sequence[Mapping[str, Any]]],
              selectivity: Mapping[str, Sequence[Mapping[str, Any]]] | None = None
              ) -> dict[str, Any]:
    """Counts of what was compared and how it came out.

    A record is *fully matched* when every quantity compared matched and none was
    left unchecked; whether the document it matched against is the paper's own
    table or a database's extraction of it decides which list it is on.
    """
    def tally(group: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, int]:
        out = {"matches": 0, "mismatch": 0, "not_found": 0, "not_checked": 0}
        for checks in group.values():
            for c in checks:
                out[c["status"]] += 1
        return out

    def full(lid: str) -> bool:
        cs = labels[lid]
        return bool(cs) and all(c["status"] == "matches" for c in cs)

    fully_primary = sorted(l for l in labels if full(l)
                           and all(c["primary"] for c in labels[l]))
    fully_secondary = sorted(l for l in labels if full(l)
                             and not all(c["primary"] for c in labels[l]))
    partly = sorted(lid for lid, cs in labels.items()
                    if any(c["status"] == "matches" for c in cs)
                    and any(c["status"] == "not_checked" for c in cs))
    return {"kinetic_quantities": tally(labels),
            "selectivity_quantities": tally(selectivity or {}),
            "records_fully_matched_against_the_papers_own_table": fully_primary,
            "records_fully_matched_against_a_curated_database_only": fully_secondary,
            "records_matched_in_part_rest_not_checked": partly,
            "records_with_a_mismatch_or_missing_value": sorted(
                lid for lid, cs in labels.items()
                if any(c["status"] in ("mismatch", "not_found") for c in cs))}
