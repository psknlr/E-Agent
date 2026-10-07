"""Binding a number in model prose to the cell it came from.

WHAT THE OLD CHECK ACTUALLY CHECKED
===================================
That a bracketed string appeared somewhere on the same line. Five sentences
that all passed it:

* ``the distance is 3.6 A and the pLDDT is 42 [artifact:geometry.tsv]`` -- one
  citation licensing two numbers from two different files;
* ``conversion reached 85 % [artifact:does_not_exist.tsv]`` -- a file nobody
  ever wrote;
* ``kcat is 12 s-1 [artifact:kinetics.tsv]`` -- which row? which column?
* ``the pLDDT is 95 [artifact:scorecards.tsv]`` where the file says 42;
* ``[artifact:geometry.tsv] says the distance; we estimate 3.6 A`` -- a
  citation placed before the number it is supposed to back.

A guard that accepts all five is not stopping a fabricated measurement, it is
teaching the model which punctuation to add. The characteristic failure of a
research agent is not refusing to answer: it is answering with a number that
reads like a measurement and was never measured, and a syntax check cannot
tell the two apart.

WHAT A CITATION HAS TO SAY
==========================
Enough to find the number again::

    [cite artifact=candidate_scorecards sha256=9f2c1a7b4e55 row=cand_0a1b
          field=plddt method=read]

Five keys, all required: which artifact, which version of it (the hash the
run manifest recorded, so a later edit of the file invalidates the citation),
which row, which field, and how the number was obtained. With those, the
guard opens the file and compares -- so a value that contradicts its own
source is refused rather than counted as cited.

BINDING IS BY ADJACENCY, NOT BY LINE
====================================
A quantity is bound to the first citation that follows it with no other
quantity in between. ``3.6 A [cite ...] and 42 pLDDT [cite ...]`` binds each
number to its own citation; ``3.6 A and 42 pLDDT [cite ...]`` leaves the 3.6
unbound, which is the correct reading of what the author wrote.

WHAT IS VERIFIED AND WHAT IS ONLY DECLARED
==========================================
``method=read`` and ``method=rounded`` are checked against the cell, with
rounding judged at the precision the author wrote. ``method=derived:...``
cannot be: this module does not re-run arithmetic it was not given. Those are
counted separately as *declared but unverified*, and a guard constructed with
``require_verified=True`` refuses them, so the distinction is a setting rather
than a silence.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..errors import EAgentError

__all__ = [
    "CitationError",
    "CITATION_KEYS",
    "VERIFIABLE_METHODS",
    "Citation",
    "parse_citations",
    "ArtifactEntry",
    "ArtifactIndex",
    "Verification",
]


class CitationError(EAgentError):
    """A citation is malformed, so nothing can be verified against it."""


#: Every key a citation must carry. Each one answers a question that, left
#: open, makes the number unfindable: *which* file, which version of it, which
#: row, which column, and whether the author read it or worked it out.
CITATION_KEYS: tuple[str, ...] = ("artifact", "sha256", "row", "field", "method")

#: Methods whose value this module can check against the cell itself.
VERIFIABLE_METHODS: frozenset[str] = frozenset({"read", "rounded"})

#: Shortest hash prefix a citation may give. Twelve hex characters is 48 bits;
#: a run holds tens of artifacts, so this identifies the version beyond doubt
#: while staying readable in prose.
MIN_SHA256_PREFIX: int = 12

_CITATION_RE = re.compile(r"\[cite\s+([^\]]+)\]", re.IGNORECASE)
_PAIR_RE = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|(\S+?))(?=\s+\w+\s*=|\s*$)')

#: Columns tried, in order, when matching a citation's ``row`` against a table.
ROW_ID_COLUMNS: tuple[str, ...] = (
    "candidate_id", "record_id", "pose_id", "sequence_sha256", "id", "name",
    "template_id", "plan_id", "well",
)


@dataclass(frozen=True)
class Citation:
    """One parsed citation, and where in the text it sat."""

    artifact: str
    sha256: str
    row: str
    field_name: str
    method: str
    start: int
    end: int
    raw: str

    @property
    def is_verifiable(self) -> bool:
        return self.method.split(":", 1)[0].strip().lower() in VERIFIABLE_METHODS

    def describe(self) -> str:
        return (f"{self.artifact}[{self.sha256[:MIN_SHA256_PREFIX]}]"
                f".{self.row}.{self.field_name} ({self.method})")


def parse_citations(text: str) -> tuple[list[Citation], list[str]]:
    """``(citations, problems)`` for one piece of model output.

    A malformed citation is a problem, never a citation: accepting it would
    make a missing ``row=`` the cheapest way to stop the value being checked.
    """
    citations: list[Citation] = []
    problems: list[str] = []
    for match in _CITATION_RE.finditer(text):
        body = match.group(1)
        pairs: dict[str, str] = {}
        for key, quoted, bare in _PAIR_RE.findall(body):
            pairs[key.strip().lower()] = (quoted if quoted else bare).strip()
        missing = [k for k in CITATION_KEYS if not pairs.get(k)]
        if missing:
            problems.append(
                f"{match.group(0)}: missing {', '.join(missing)}. A citation "
                f"without them does not say where the number can be found "
                f"again")
            continue
        digest = pairs["sha256"].strip().lower()
        if not re.fullmatch(r"[0-9a-f]{%d,64}" % MIN_SHA256_PREFIX, digest):
            problems.append(
                f"{match.group(0)}: sha256={pairs['sha256']!r} is not at least "
                f"{MIN_SHA256_PREFIX} hex characters of a digest; without the "
                f"version, a later edit of the file leaves the citation "
                f"looking valid")
            continue
        citations.append(Citation(
            artifact=pairs["artifact"], sha256=digest, row=pairs["row"],
            field_name=pairs["field"], method=pairs["method"],
            start=match.start(), end=match.end(), raw=match.group(0)))
    return citations, problems


@dataclass(frozen=True)
class ArtifactEntry:
    """One artifact a run produced: its key, its file and its recorded hash."""

    key: str
    path: Path | None
    sha256: str | None


class ArtifactIndex:
    """What the run actually wrote, for checking what the model says it wrote.

    Built from a :class:`~eagent.provenance.RunManifest` so that the hashes
    are the ones recorded when the file was produced, not recomputed from
    whatever is on disk now. A citation whose hash does not match the recorded
    one is refused: the file has changed since the number was read out of it,
    and which of the two is right is not this module's to decide.
    """

    def __init__(self, entries: Iterable[ArtifactEntry] = ()) -> None:
        self._entries: dict[str, ArtifactEntry] = {e.key: e for e in entries}
        self._tables: dict[str, list[dict[str, str]]] = {}

    @classmethod
    def from_manifest(cls, manifest: Any) -> "ArtifactIndex":
        entries: list[ArtifactEntry] = []
        for step in getattr(manifest, "steps", ()):
            for artifact in getattr(step, "artifacts", ()) or ():
                data = artifact if isinstance(artifact, Mapping) else {}
                key = str(data.get("key") or "")
                if not key:
                    continue
                raw_path = data.get("path")
                entries.append(ArtifactEntry(
                    key=key,
                    path=Path(str(raw_path)) if raw_path else None,
                    sha256=(str(data["sha256"]).lower()
                            if data.get("sha256") else None)))
        return cls(entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    @property
    def keys(self) -> list[str]:
        return sorted(self._entries)

    def resolve(self, key: str) -> ArtifactEntry | None:
        return self._entries.get(key)

    # -- table access ------------------------------------------------------
    def rows(self, key: str) -> list[dict[str, str]]:
        """Every row of a tabular artifact, as strings. Empty if unreadable."""
        if key in self._tables:
            return self._tables[key]
        entry = self._entries.get(key)
        rows: list[dict[str, str]] = []
        if entry is not None and entry.path is not None and entry.path.is_file():
            rows = _read_table(entry.path)
        self._tables[key] = rows
        return rows

    def row(self, key: str, row_id: str) -> dict[str, str] | None:
        """The row a citation names, matched on any plausible id column."""
        rows = self.rows(key)
        wanted = row_id.strip()
        for record in rows:
            for column in ROW_ID_COLUMNS:
                if column in record and str(record[column]).strip() == wanted:
                    return record
        for record in rows:                      # fall back to the first column
            if record and str(next(iter(record.values()))).strip() == wanted:
                return record
        return None


def _read_table(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    try:
        if suffix in (".tsv", ".tab"):
            with open(path, "r", encoding="utf-8", newline="") as fh:
                return [dict(r) for r in csv.DictReader(fh, delimiter="\t")]
        if suffix == ".csv":
            with open(path, "r", encoding="utf-8", newline="") as fh:
                return [dict(r) for r in csv.DictReader(fh)]
        if suffix in (".jsonl", ".ndjson"):
            out: list[dict[str, str]] = []
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        out.append({k: _as_text(v) for k, v in obj.items()})
            return out
        if suffix == ".json":
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
            if isinstance(obj, list):
                return [{k: _as_text(v) for k, v in o.items()}
                        for o in obj if isinstance(o, dict)]
            if isinstance(obj, dict):
                return [{k: _as_text(v) for k, v in obj.items()}]
    except (OSError, ValueError):
        return []
    return []


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def values_agree(cited: str, stored: str) -> bool:
    """Whether a number written in prose is the one in the cell.

    Rounding is judged at the precision the author wrote: ``3.6`` against a
    stored ``3.5987`` agrees, because that is what writing one decimal place
    means, and so does ``3.60``, which is the same number to two. ``3.59``
    does not: at the precision claimed, the cell reads 3.60. Writing more
    digits therefore narrows the claim rather than widening it, which is the
    right direction. A cell that is not a number is compared as text.
    """
    try:
        cited_value = float(cited)
    except (TypeError, ValueError):
        return str(cited).strip() == str(stored).strip()
    try:
        stored_value = float(stored)
    except (TypeError, ValueError):
        return False
    text = str(cited).strip()
    decimals = len(text.split(".", 1)[1]) if "." in text else 0
    return math.isclose(round(stored_value, decimals), cited_value,
                        rel_tol=1e-9, abs_tol=10 ** -(decimals + 6))


@dataclass
class Verification:
    """What checking one cited quantity against the run's artifacts found."""

    quantity: str
    citation: Citation | None
    ok: bool = False
    verified: bool = False
    problem: str = ""

    def describe(self) -> str:
        where = self.citation.describe() if self.citation else "no citation"
        return f"{self.quantity} <- {where}" + (f": {self.problem}"
                                                if self.problem else "")


def verify(quantity_value: str, citation: Citation,
           index: ArtifactIndex) -> Verification:
    """Check one cited quantity against the artifact it names."""
    entry = index.resolve(citation.artifact)
    if entry is None:
        known = ", ".join(index.keys[:8]) or "none"
        return Verification(
            quantity_value, citation,
            problem=(f"this run produced no artifact called "
                     f"'{citation.artifact}'. It produced: {known}"))
    if entry.sha256 and not entry.sha256.startswith(citation.sha256):
        return Verification(
            quantity_value, citation,
            problem=(f"the citation gives sha256={citation.sha256} but the run "
                     f"recorded {entry.sha256[:16]}... for "
                     f"'{citation.artifact}'; the file is not the one the "
                     f"number was read from"))
    if not citation.is_verifiable:
        return Verification(
            quantity_value, citation, ok=True, verified=False,
            problem=(f"method={citation.method} is declared, not checked: this "
                     f"guard does not re-run arithmetic it was not given"))
    record = index.row(citation.artifact, citation.row)
    if record is None:
        if not index.rows(citation.artifact):
            return Verification(
                quantity_value, citation, ok=True, verified=False,
                problem=(f"'{citation.artifact}' could not be read as a table, "
                         f"so row '{citation.row}' could not be checked"))
        return Verification(
            quantity_value, citation,
            problem=f"'{citation.artifact}' has no row '{citation.row}'")
    if citation.field_name not in record:
        return Verification(
            quantity_value, citation,
            problem=(f"'{citation.artifact}' row '{citation.row}' has no field "
                     f"'{citation.field_name}'; it has "
                     f"{', '.join(sorted(record)[:8])}"))
    stored = record[citation.field_name]
    if not values_agree(quantity_value, stored):
        return Verification(
            quantity_value, citation,
            problem=(f"the text says {quantity_value} but "
                     f"{citation.artifact}.{citation.row}."
                     f"{citation.field_name} is {stored!r}"))
    return Verification(quantity_value, citation, ok=True, verified=True)
