"""Dependency-free reader and writer for PDB and mmCIF coordinate files.

Why this exists at all: biopython is not installed, and the alternative --
regex-scraping coordinates inline wherever they happen to be needed -- is how
cofactors go missing. This module is the single place that turns a coordinate
file into objects, and it is written so that *anything it cannot interpret is
an error, never a skipped line*. A silently dropped HETATM record is the exact
failure that produces a confident wrong answer: the pocket looks empty, the
docking run has nowhere to put the nicotinamide, and the geometry report says
the hydride donor is unresolvable rather than saying the file was misread.

mmCIF is the preferred storage format for this project and
:func:`read_mmcif` is the reader to reach for. The reasons are concrete, not
stylistic:

* PDB format cannot hold an author residue number outside -999..9999, an atom
  serial above 99999, a chain identifier longer than one character, or a
  chemical component id longer than three characters. Every one of those
  limits is reached by real entries, and the usual workaround -- truncation --
  changes chemical identity.
* PDB files routinely omit the element column (77-78), forcing the reader to
  infer the element from the atom name. mmCIF carries ``_atom_site.type_symbol``
  explicitly, which removes the Ca/C-alpha ambiguity described in
  :func:`_element_from_pdb_name`.

:func:`write_pdb` is therefore deliberately limited: it exists to round-trip a
small selection (a pocket, a ligand, a pose fragment) into tools that only
speak PDB, and it raises rather than truncating when a field does not fit.

What this module does *not* do: it does not assign bonds, protonate anything,
infer a missing occupancy, or guess a cofactor oxidation state. A quantity
that is absent from the file stays ``None``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..errors import EAgentError

__all__ = [
    "StructureParseError",
    "StructureWriteError",
    "Atom",
    "Residue",
    "Chain",
    "Structure",
    "read_pdb",
    "read_mmcif",
    "mmcif_categories",
    "read_structure",
    "write_pdb",
    "WATER_RESNAMES",
    "TWO_LETTER_ELEMENTS",
]


class StructureParseError(EAgentError):
    """A coordinate file contains something this reader cannot interpret.

    Raised instead of skipping the offending record. The whole point of the
    reader is that the set of atoms it returns is the set of atoms in the
    file; a reader that drops what it does not understand cannot make that
    promise, and the caller has no way to notice.
    """


class StructureWriteError(EAgentError):
    """A structure cannot be expressed in the requested output format.

    Raised rather than truncating. Writing residue ``NAPX`` as ``NAP`` or
    chain ``AAA`` as ``A`` produces a file that parses cleanly and describes a
    different molecule.
    """


#: Residue names treated as bulk solvent. Waters are excluded from
#: :meth:`Structure.ligands` because a pocket water is not a cofactor, but
#: they are still parsed and still available through :meth:`Structure.select`,
#: since an ordered water can be a genuine part of a mechanism.
WATER_RESNAMES: frozenset[str] = frozenset(
    {"HOH", "DOD", "WAT", "H2O", "SOL", "TIP", "TIP3", "TIP4", "T3P", "TP3"}
)

#: Element symbols that are two characters long and plausible in a protein
#: structure file. Used only to disambiguate an unpadded PDB atom name; see
#: :func:`_element_from_pdb_name` for why the padding matters.
TWO_LETTER_ELEMENTS: frozenset[str] = frozenset(
    {
        "AG", "AL", "AR", "AS", "AU", "BA", "BE", "BI", "BR", "CA", "CD", "CE",
        "CL", "CO", "CR", "CS", "CU", "DY", "ER", "EU", "FE", "GA", "GD", "GE",
        "HG", "HO", "IN", "IR", "KR", "LA", "LI", "LU", "MG", "MN", "MO", "NA",
        "NB", "ND", "NE", "NI", "OS", "PB", "PD", "PR", "PT", "RB", "RE", "RH",
        "RU", "SB", "SC", "SE", "SI", "SM", "SN", "SR", "TA", "TB", "TC", "TE",
        "TI", "TL", "TM", "XE", "YB", "ZN", "ZR",
    }
)

#: Marker values that mean "no value" in mmCIF. They are not data and must not
#: be parsed as a number or a name.
_CIF_NULL: frozenset[str] = frozenset({".", "?"})


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Atom:
    """One coordinate record.

    ``occupancy`` and ``bfactor_or_plddt`` are ``None`` when the source file
    did not carry them. They are deliberately *not* defaulted to 1.0 and 0.0:
    occupancy is a measured refinement quantity and a fabricated 1.0 would let
    a partially occupied -- or entirely unobserved -- atom pass a QC screen
    that exists precisely to catch it.

    ``bfactor_or_plddt`` carries one of two incommensurable quantities
    depending on provenance: a crystallographic B-factor (low is good, units
    A^2) or a predicted-model pLDDT (high is good, 0-100). The field is named
    for both so that no caller can read it without first asking which kind of
    structure it came from; :attr:`Structure.source_format` and the owning
    :class:`~eagent.schemas.candidate.StructureRecord` say which.
    """

    serial: int
    name: str
    element: str
    resname: str
    chain: str
    resseq: int
    icode: str
    altloc: str
    x: float
    y: float
    z: float
    occupancy: float | None = None
    bfactor_or_plddt: float | None = None
    is_hetatm: bool = False
    model: int = 1
    #: True when the element was inferred from the atom name rather than read
    #: from the file. Surfaced in :meth:`Structure.quality_notes`.
    element_inferred: bool = False

    @property
    def coords(self) -> tuple[float, float, float]:
        """Position as a plain tuple, for the geometry helpers."""
        return (self.x, self.y, self.z)

    @property
    def is_heavy(self) -> bool:
        """Non-hydrogen. Shells and clash screens are defined on heavy atoms
        because most deposited structures have no hydrogens at all, so a
        hydrogen-inclusive criterion would silently mean different things for
        a crystal structure and a predicted model."""
        return self.element.upper() not in ("H", "D")

    @property
    def residue_key(self) -> tuple[str, int, str, str]:
        """(chain, resseq, icode, resname) -- the grouping key for residues."""
        return (self.chain, self.resseq, self.icode, self.resname)

    @property
    def is_water(self) -> bool:
        """Bulk solvent, by residue name. Waters are parsed and kept, but
        they are not ligands and must not be counted as a cofactor."""
        return self.resname.strip().upper() in WATER_RESNAMES

    def label(self) -> str:
        """Human-readable identifier used in QC messages and reports."""
        ic = self.icode.strip()
        alt = f".{self.altloc}" if self.altloc.strip() else ""
        return f"{self.chain}/{self.resname}{self.resseq}{ic}/{self.name}{alt}"


@dataclass
class Residue:
    """A group of atoms sharing (chain, resseq, icode, resname).

    Residues are kept as objects rather than as (chain, number) pairs because
    every downstream question -- is this residue in the pocket, is it the
    catalytic tyrosine, is it observed -- needs the atoms, and re-deriving the
    grouping at each call site is how two call sites end up disagreeing about
    what a residue is.
    """

    chain: str
    resname: str
    resseq: int
    icode: str
    atoms: list[Atom] = field(default_factory=list)
    #: True when every atom of the residue is a HETATM record. A residue with
    #: mixed ATOM/HETATM records is malformed and is reported in
    #: :meth:`Structure.quality_notes` rather than silently coerced.
    is_hetatm: bool = False

    @property
    def key(self) -> tuple[str, int, str, str]:
        """Full identity including the component name, so that a ligand and a
        residue that share an author position cannot be confused."""
        return (self.chain, self.resseq, self.icode, self.resname)

    @property
    def author_key(self) -> tuple[str, int, str]:
        """(chain, resseq, icode): the author numbering handle used by
        :mod:`eagent.science.numbering`."""
        return (self.chain, self.resseq, self.icode)

    @property
    def is_water(self) -> bool:
        """Bulk solvent. Excluded from ligands and from mutation shells."""
        return self.resname.strip().upper() in WATER_RESNAMES

    def heavy_atoms(self) -> list[Atom]:
        """Non-hydrogen atoms -- the only ones most deposited structures have,
        so shells and contacts are defined on these to mean the same thing for
        a crystal structure and for a predicted model."""
        return [a for a in self.atoms if a.is_heavy]

    def atom(self, name: str, altloc: str | None = None) -> Atom | None:
        """Fetch one named atom, or ``None``.

        Returns ``None`` rather than raising because a missing side-chain atom
        is ordinary in a real structure; the caller decides whether that is
        fatal for its particular measurement.
        """
        want = name.strip().upper()
        for a in self.atoms:
            if a.name.strip().upper() != want:
                continue
            if altloc is not None and a.altloc.strip() != altloc.strip():
                continue
            return a
        return None

    def has_altloc(self) -> bool:
        """Whether any atom is modelled in more than one place. A geometry
        measurement on such a residue is ambiguous until a conformer is
        chosen, which is why this is surfaced rather than averaged away."""
        return any(a.altloc.strip() for a in self.atoms)

    def __str__(self) -> str:
        return f"{self.chain}/{self.resname}{self.resseq}{self.icode.strip()}"

    def __len__(self) -> int:
        return len(self.atoms)


@dataclass
class Chain:
    """Residues of one author chain, in the order the file listed them.

    File order is preserved rather than sorted by residue number. Deposited
    order is the polymer order, and author numbering is not always monotonic
    (chimeras, engineered insertions, antibody numbering schemes). Sorting
    would quietly reorder such a chain and corrupt the sequence derived from
    it; :mod:`eagent.science.numbering` instead reports non-monotonic
    numbering as a note.
    """

    chain_id: str
    residues: list[Residue] = field(default_factory=list)

    def residue(self, resseq: int, icode: str = "") -> Residue | None:
        """Residue by author number, or ``None``. The insertion code is part
        of the lookup: 100 and 100A are different residues."""
        for r in self.residues:
            if r.resseq == resseq and r.icode.strip() == icode.strip():
                return r
        return None

    def atoms(self) -> list[Atom]:
        """Every atom of this chain, in file order."""
        return [a for r in self.residues for a in r.atoms]

    def polymer_residues(self) -> list[Residue]:
        """Residues recorded as ATOM (not HETATM) -- the candidate polymer."""
        return [r for r in self.residues if not r.is_hetatm]

    def __len__(self) -> int:
        return len(self.residues)


@dataclass
class Structure:
    """A parsed coordinate file: one model, its chains, and its parse history.

    ``parse_notes`` records every decision the reader had to make (a dropped
    NMR model, an inferred element, a duplicated residue key). They are
    surfaced through :meth:`quality_notes` so that a reviewer can see what the
    reader did, instead of having to trust that it did nothing.
    """

    structure_id: str
    chains: list[Chain] = field(default_factory=list)
    source_format: str = "pdb"            # pdb | mmcif
    source_path: str | None = None
    models_present: list[int] = field(default_factory=list)
    model_selected: int | None = None
    parse_notes: list[str] = field(default_factory=list)

    # -- access ------------------------------------------------------------
    def chain(self, chain_id: str) -> Chain | None:
        """Chain by author identifier, or ``None`` if this file has no such
        chain -- which is usually a sign the wrong entry or the wrong assembly
        was picked, so callers should check rather than fall through."""
        for c in self.chains:
            if c.chain_id == chain_id:
                return c
        return None

    def atoms(self) -> list[Atom]:
        """Every atom of the selected model, in file order."""
        return [a for c in self.chains for r in c.residues for a in r.atoms]

    def residues(self) -> list[Residue]:
        """Every residue of the selected model, polymer and HETATM alike."""
        return [r for c in self.chains for r in c.residues]

    def n_atoms(self) -> int:
        """Atom count of the selected model. Compare it against the source
        file when a reader change is being reviewed: a drop means atoms were
        lost, which is the failure this module is built to prevent."""
        return sum(len(r.atoms) for c in self.chains for r in c.residues)

    def waters(self) -> list[Residue]:
        """Solvent residues, kept separate from :meth:`ligands` because an
        ordered active-site water can matter while bulk solvent never does."""
        return [r for r in self.residues() if r.is_water]

    def ligands(self) -> list[Residue]:
        """HETATM residues that are not bulk solvent.

        Metal ions are included. They are HETATM, they are not water, and in a
        zinc-dependent alcohol dehydrogenase the catalytic zinc is the whole
        mechanism -- filtering ions out "because they are not really ligands"
        is one of the ways a cofactor disappears between parsing and docking.
        Modified amino acids that a depositor recorded as HETATM (MSE, SEP)
        will also appear here; :mod:`eagent.science.numbering` recognises them
        as polymer residues when building a sequence, so the two views are
        intentionally different and each says which it is.
        """
        return [r for r in self.residues() if r.is_hetatm and not r.is_water]

    # -- selection ---------------------------------------------------------
    def select(
        self,
        chain: str | Iterable[str] | None = None,
        resname: str | Iterable[str] | None = None,
        resseq: int | Iterable[int] | None = None,
        element: str | Iterable[str] | None = None,
        is_hetatm: bool | None = None,
        name: str | Iterable[str] | None = None,
        altloc: str | Iterable[str] | None = None,
        icode: str | Iterable[str] | None = None,
        heavy_only: bool = False,
    ) -> list[Atom]:
        """Filter atoms by any combination of fields.

        Every criterion accepts a single value or an iterable of values, and
        ``None`` means "do not filter on this". Name-like fields are compared
        case-insensitively after stripping, because PDB pads them and mmCIF
        does not -- comparing raw strings is a reliable source of "the atom is
        definitely there but the selection is empty" bugs.
        """
        out: list[Atom] = []
        for a in self.atoms():
            if heavy_only and not a.is_heavy:
                continue
            if not _matches(a.chain.strip(), chain, ci=True):
                continue
            if not _matches(a.resname.strip(), resname, ci=True):
                continue
            if not _matches(a.resseq, resseq):
                continue
            if not _matches(a.element.strip(), element, ci=True):
                continue
            if not _matches(a.name.strip(), name, ci=True):
                continue
            if not _matches(a.altloc.strip(), altloc, ci=True):
                continue
            if not _matches(a.icode.strip(), icode, ci=True):
                continue
            if is_hetatm is not None and a.is_hetatm is not is_hetatm:
                continue
            out.append(a)
        return out

    # -- quality -----------------------------------------------------------
    def quality_notes(self) -> list[str]:
        """Structural caveats a geometry measurement must not ignore.

        These are *localisation* notes, not a verdict. Each one describes a
        way the coordinates may not mean what a naive distance calculation
        assumes:

        * **Alternate locations.** An atom with altloc A and altloc B is one
          atom modelled in two places. Measuring to both and averaging, or
          taking whichever comes first in the file, both produce a number with
          no physical meaning. The caller must pick a conformer explicitly.
        * **Occupancy below 1.** The atom was not fully present in the
          crystal. A catalytic residue at occupancy 0.4 is weak evidence for
          anything; at occupancy 0 it was placed by the refinement program,
          not observed.
        * **Missing occupancy column.** Reported separately, because "absent"
          and "1.00" are different statements and only one of them is in the
          file.
        * **No non-water HETATM group.** If the catalytic template requires a
          cofactor, this file does not contain it, and any pose built from it
          is an apo model regardless of what the pipeline calls it.
        * Everything the reader itself had to decide (``parse_notes``).
        """
        notes: list[str] = list(self.parse_notes)
        atoms = self.atoms()

        alt = [a for a in atoms if a.altloc.strip()]
        if alt:
            residues = sorted({str(_residue_label(a)) for a in alt})
            shown = ", ".join(residues[:10])
            more = f" (+{len(residues) - 10} more)" if len(residues) > 10 else ""
            notes.append(
                f"altloc: {len(alt)} atom(s) in {len(residues)} residue(s) have "
                f"alternate locations [{shown}{more}]; select one conformer "
                f"explicitly before measuring geometry"
            )

        missing_occ = [a for a in atoms if a.occupancy is None]
        if missing_occ:
            notes.append(
                f"occupancy: {len(missing_occ)} atom(s) carry no occupancy value; "
                f"they are reported as unknown rather than assumed fully occupied"
            )

        partial = [a for a in atoms if a.occupancy is not None and a.occupancy < 1.0]
        if partial:
            zero = [a for a in partial if a.occupancy == 0.0]
            worst = min(a.occupancy for a in partial)  # type: ignore[type-var]
            notes.append(
                f"occupancy: {len(partial)} atom(s) have occupancy < 1.0 "
                f"(lowest {worst:.2f}); partially occupied atoms are weak "
                f"evidence for a geometric claim"
            )
            if zero:
                notes.append(
                    f"occupancy: {len(zero)} atom(s) have occupancy 0.0 "
                    f"({', '.join(sorted({a.label() for a in zero})[:5])}); these "
                    f"positions were modelled, not observed"
                )

        if not self.ligands():
            notes.append(
                "ligands: no non-water HETATM group in this file; if the "
                "catalytic template requires a cofactor or metal, it is absent "
                "and this is an apo structure"
            )

        if len(self.models_present) > 1:
            others = [m for m in self.models_present if m != self.model_selected]
            notes.append(
                f"models: file contains models {self.models_present}; model "
                f"{self.model_selected} was parsed and {others} were not. "
                f"Re-read with model=<n> to measure a different one."
            )

        return notes


def _residue_label(a: Atom) -> str:
    return f"{a.chain}/{a.resname}{a.resseq}{a.icode.strip()}"


def _matches(value: Any, criterion: Any, ci: bool = False) -> bool:
    """Single-value or membership test used by :meth:`Structure.select`."""
    if criterion is None:
        return True
    if ci and isinstance(value, str):
        value = value.upper()
    if isinstance(criterion, (list, tuple, set, frozenset)):
        if ci:
            return value in {c.upper() if isinstance(c, str) else c for c in criterion}
        return value in criterion
    if ci and isinstance(criterion, str):
        criterion = criterion.upper()
    return value == criterion


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------


def _resolve_source(path_or_text: str | Path, what: str) -> tuple[str, str | None]:
    """Return (text, path) from either a path or the file contents.

    Both callers accept either so that tests can use inline fixtures without
    touching the filesystem. The disambiguation is explicit -- a ``Path`` is
    always a path, a string containing a newline is always text -- because an
    ambiguous rule here turns a typo in a filename into an empty structure.
    """
    if isinstance(path_or_text, Path):
        if not path_or_text.exists():
            raise StructureParseError(f"{what}: file not found: {path_or_text}")
        return path_or_text.read_text(encoding="utf-8", errors="strict"), str(path_or_text)
    if not isinstance(path_or_text, str):
        raise StructureParseError(
            f"{what}: expected a path or the file contents, got "
            f"{type(path_or_text).__name__}"
        )
    if "\n" in path_or_text or "\r" in path_or_text:
        return path_or_text, None
    candidate = path_or_text.strip()
    if candidate and os.path.exists(candidate):
        return Path(candidate).read_text(encoding="utf-8", errors="strict"), candidate
    raise StructureParseError(
        f"{what}: '{path_or_text[:80]}' is neither an existing file nor "
        f"multi-line {what.split()[0]} text"
    )


def _element_from_pdb_name(raw_name_field: str) -> str:
    """Infer an element from a PDB atom-name field, using the column padding.

    The PDB atom name occupies columns 13-16. The convention is that a
    two-character element symbol starts in column 13, while a one-character
    element is preceded by a blank. That single space is the only thing
    separating the protein C-alpha (``" CA "``) from a calcium ion
    (``"CA  "``), and getting it wrong changes a carbon into a metal -- which
    changes the van der Waals radius, the clash screen and whether the pocket
    appears to contain a catalytic metal.

    This function is only reached when the file omits the element columns
    (77-78). It raises if it cannot decide, rather than returning a plausible
    ``"C"``. mmCIF avoids the whole problem by carrying ``type_symbol``.
    """
    padded = raw_name_field.ljust(4)[:4]
    head = padded[:2].strip().upper()
    stripped = padded.strip()
    if not stripped:
        raise StructureParseError("atom record has an empty atom name field")
    # Column 13 occupied by a letter and the first two characters form a known
    # two-letter symbol -> that symbol.
    if padded[0] not in " " and head.isalpha() and head in TWO_LETTER_ELEMENTS:
        return head
    # Hydrogens are often named 1HB, 2HG1 ... with a leading count digit.
    for ch in stripped:
        if ch.isalpha():
            return ch.upper()
    raise StructureParseError(
        f"cannot infer an element from atom name {raw_name_field!r}; supply the "
        f"element columns (77-78) or use mmCIF, which carries type_symbol"
    )


def _float_or_raise(token: str, what: str, where: str) -> float:
    try:
        return float(token)
    except (TypeError, ValueError):
        raise StructureParseError(
            f"{where}: {what} is not a number: {token!r}"
        ) from None


def _int_or_raise(token: str, what: str, where: str) -> int:
    try:
        return int(token)
    except (TypeError, ValueError):
        raise StructureParseError(
            f"{where}: {what} is not an integer: {token!r}"
        ) from None


# ---------------------------------------------------------------------------
# PDB
# ---------------------------------------------------------------------------


def read_pdb(
    path_or_text: str | Path,
    structure_id: str | None = None,
    model: int | None = None,
) -> Structure:
    """Parse a PDB-format coordinate file (or its text) into a :class:`Structure`.

    Only ATOM, HETATM, MODEL and ENDMDL records are interpreted; headers and
    annotation records are ignored because nothing downstream reads them and
    pretending to parse them would invite trusting them. Any ATOM/HETATM line
    that cannot be interpreted raises :class:`StructureParseError`: that is the
    whole contract of this reader.

    ``model`` selects an NMR/ensemble model. When omitted the first model is
    used and the others are recorded in :meth:`Structure.quality_notes`, so a
    20-model NMR ensemble cannot be measured as if it were one structure
    without the caller being told.
    """
    text, path = _resolve_source(path_or_text, "PDB")
    sid = structure_id or (Path(path).stem if path else "inline")

    atoms: list[Atom] = []
    notes: list[str] = []
    models_present: list[int] = []
    current_model = 1
    seen_model_record = False
    n_inferred_elements = 0

    for lineno, raw in enumerate(text.splitlines(), start=1):
        record = raw[:6].strip().upper()
        if record == "MODEL":
            seen_model_record = True
            token = raw[10:14].strip() or raw[6:].strip()
            current_model = _int_or_raise(token, "MODEL number", f"line {lineno}")
            if current_model not in models_present:
                models_present.append(current_model)
            continue
        if record == "ENDMDL":
            continue
        if record not in ("ATOM", "HETATM"):
            continue

        if len(raw) < 54:
            raise StructureParseError(
                f"line {lineno}: {record} record is {len(raw)} characters long; "
                f"coordinates require at least 54. Refusing to guess the "
                f"missing fields."
            )
        if not seen_model_record and not models_present:
            models_present.append(current_model)

        where = f"line {lineno}"
        serial_tok = raw[6:11].strip()
        # Serial numbers overflow past 99999 and are then written in hex or as
        # '*****' by various programs. They are bookkeeping, not science, so an
        # unparsable serial becomes a negative placeholder and a note -- but it
        # is a note, not a silent substitution.
        if serial_tok.isdigit():
            serial = int(serial_tok)
        else:
            serial = -(len(atoms) + 1)
            notes.append(
                f"{where}: atom serial {serial_tok!r} is not a decimal integer "
                f"(overflow past 99999?); a negative placeholder was used and "
                f"serial numbers in this structure are not file-faithful"
            )

        name_field = raw[12:16]
        altloc = raw[16:17].strip()
        resname = raw[17:20].strip()
        chain_id = raw[21:22].strip()
        resseq = _int_or_raise(raw[22:26].strip(), "residue sequence number", where)
        icode = raw[26:27].strip()
        x = _float_or_raise(raw[30:38].strip(), "x coordinate", where)
        y = _float_or_raise(raw[38:46].strip(), "y coordinate", where)
        z = _float_or_raise(raw[46:54].strip(), "z coordinate", where)

        occ_tok = raw[54:60].strip() if len(raw) >= 55 else ""
        occupancy = _float_or_raise(occ_tok, "occupancy", where) if occ_tok else None
        b_tok = raw[60:66].strip() if len(raw) >= 61 else ""
        bfactor = _float_or_raise(b_tok, "B-factor/pLDDT", where) if b_tok else None

        el_tok = raw[76:78].strip() if len(raw) >= 77 else ""
        if el_tok:
            element = el_tok.upper()
            inferred = False
        else:
            element = _element_from_pdb_name(name_field)
            inferred = True
            n_inferred_elements += 1

        if not resname:
            raise StructureParseError(f"{where}: {record} record has no residue name")

        atoms.append(
            Atom(
                serial=serial,
                name=name_field.strip(),
                element=element,
                resname=resname,
                chain=chain_id,
                resseq=resseq,
                icode=icode,
                altloc=altloc,
                x=x,
                y=y,
                z=z,
                occupancy=occupancy,
                bfactor_or_plddt=bfactor,
                is_hetatm=(record == "HETATM"),
                model=current_model,
                element_inferred=inferred,
            )
        )

    if not atoms:
        raise StructureParseError(
            "no ATOM or HETATM records found; this is not a usable coordinate file"
        )

    if n_inferred_elements:
        notes.append(
            f"elements: {n_inferred_elements} atom(s) had no element column and "
            f"the element was inferred from the atom-name padding; mmCIF carries "
            f"type_symbol explicitly and should be preferred"
        )

    return _assemble(
        atoms,
        structure_id=sid,
        source_format="pdb",
        source_path=path,
        models_present=models_present or [1],
        requested_model=model,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# mmCIF
# ---------------------------------------------------------------------------


def _split_cif_line(line: str, lineno: int) -> list[tuple[str, bool]]:
    """Tokenise one mmCIF line into (value, was_quoted) pairs.

    Quoting matters for correctness, not just tidiness: an unquoted ``_`` or
    ``loop_`` would terminate the value list of the ``_atom_site`` loop, while
    the same characters inside quotes are data. A tokeniser that ignores
    quoting therefore truncates the atom list -- silently.
    """
    toks: list[tuple[str, bool]] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch in " \t":
            i += 1
            continue
        if ch == "#":
            break
        if ch in "'\"":
            quote = ch
            i += 1
            start = i
            closed = False
            while i < n:
                if line[i] == quote and (i + 1 >= n or line[i + 1] in " \t"):
                    closed = True
                    break
                i += 1
            if not closed:
                raise StructureParseError(
                    f"mmCIF line {lineno}: unterminated {quote} quoted value"
                )
            toks.append((line[start:i], True))
            i += 1
            continue
        start = i
        while i < n and line[i] not in " \t":
            i += 1
        toks.append((line[start:i], False))
    return toks


def _cif_tokens(text: str) -> list[tuple[str, bool]]:
    """Whole-file token stream, handling semicolon-delimited multiline values."""
    toks: list[tuple[str, bool]] = []
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith(";"):
            buf = [line[1:]]
            i += 1
            closed = False
            while i < n:
                if lines[i].startswith(";"):
                    closed = True
                    break
                buf.append(lines[i])
                i += 1
            if not closed:
                raise StructureParseError(
                    f"mmCIF line {i}: unterminated ';' multiline value"
                )
            toks.append(("\n".join(buf).strip(), True))
            # the closing ';' line may carry further tokens after it
            rest = lines[i][1:]
            if rest.strip():
                toks.extend(_split_cif_line(rest, i + 1))
            i += 1
            continue
        toks.extend(_split_cif_line(line, i + 1))
        i += 1
    return toks


def _is_cif_keyword(value: str, quoted: bool) -> bool:
    if quoted:
        return False
    low = value.lower()
    return (
        value.startswith("_")
        or low in ("loop_", "stop_")
        or low.startswith("data_")
        or low.startswith("save_")
        or low.startswith("global_")
    )


def _extract_atom_site_loop(text: str) -> tuple[list[str], list[list[str]]]:
    """Return the ``_atom_site`` loop's tags and rows.

    Full CIF generality is not attempted -- this reader only needs the
    coordinate loop. What it does guarantee is that it never returns a partial
    loop: a value count that is not a whole multiple of the tag count means
    the file was misread, and that raises.
    """
    toks = _cif_tokens(text)
    i = 0
    n = len(toks)
    while i < n:
        value, quoted = toks[i]
        if not quoted and value.lower() == "loop_":
            i += 1
            tags: list[str] = []
            while i < n and not toks[i][1] and toks[i][0].startswith("_"):
                tags.append(toks[i][0])
                i += 1
            values: list[str] = []
            while i < n and not _is_cif_keyword(toks[i][0], toks[i][1]):
                values.append(toks[i][0])
                i += 1
            if tags and tags[0].lower().startswith("_atom_site."):
                if not values:
                    raise StructureParseError(
                        "mmCIF: the _atom_site loop_ declares tags but holds no rows"
                    )
                if len(values) % len(tags) != 0:
                    raise StructureParseError(
                        f"mmCIF: the _atom_site loop_ has {len(values)} values for "
                        f"{len(tags)} tags, which is not a whole number of rows. "
                        f"Refusing to parse a truncated atom list."
                    )
                rows = [
                    values[k : k + len(tags)]
                    for k in range(0, len(values), len(tags))
                ]
                return tags, rows
            continue
        i += 1
    raise StructureParseError(
        "mmCIF: no '_atom_site' loop_ found; this file carries no coordinates "
        "this reader can interpret"
    )


def mmcif_categories(text: str, categories: Iterable[str]) -> dict[str, list[dict[str, str]]]:
    """The named non-coordinate categories of an mmCIF file, as lists of rows.

    ``_struct`` (single values) comes back as one row, ``_entity`` (a loop) as
    one row per entity; the keys of a row are the item names without the
    category prefix (``title``, not ``_struct.title``). Values are the raw
    strings, including the CIF nulls ``.`` and ``?``, which are *not* converted
    to ``None`` here: whether a missing value matters is the caller's decision.

    Deliberately not a general CIF reader. It exists so that facts about an
    entry (title, method, entities, revision history) can be read with the same
    quoting and multi-line rules the coordinate reader uses, instead of by a
    second regex that disagrees about quotes. The coordinate loop is never
    requested through it; use :func:`read_mmcif` for that.
    """
    wanted = {c.strip().lower() for c in categories}
    toks = _cif_tokens(text)
    out: dict[str, list[dict[str, str]]] = {}
    n = len(toks)
    i = 0
    while i < n:
        value, quoted = toks[i]
        if not quoted and value.lower() == "loop_":
            i += 1
            tags: list[str] = []
            while i < n and not toks[i][1] and toks[i][0].startswith("_"):
                tags.append(toks[i][0])
                i += 1
            values: list[str] = []
            while i < n and not _is_cif_keyword(toks[i][0], toks[i][1]):
                values.append(toks[i][0])
                i += 1
            if not tags:
                continue
            category = tags[0].split(".", 1)[0].lower()
            if category not in wanted:
                continue
            if len(values) % len(tags) != 0:
                raise StructureParseError(
                    f"mmCIF: the {category} loop_ has {len(values)} values for "
                    f"{len(tags)} tags, which is not a whole number of rows")
            names = [t.split(".", 1)[1] if "." in t else t for t in tags]
            for k in range(0, len(values), len(tags)):
                out.setdefault(category, []).append(
                    dict(zip(names, values[k:k + len(tags)])))
            continue
        if not quoted and value.startswith("_") and "." in value:
            category, _, item = value.partition(".")
            if category.lower() in wanted and i + 1 < n:
                rows = out.setdefault(category.lower(), [{}])
                rows[0][item] = toks[i + 1][0]
            i += 2
            continue
        i += 1
    return out


def _cif_value(row: Sequence[str], idx: int | None) -> str | None:
    """Row value, or ``None`` for an absent tag or a CIF null ('.' / '?')."""
    if idx is None:
        return None
    v = row[idx]
    if v in _CIF_NULL:
        return None
    return v


def read_mmcif(
    path_or_text: str | Path,
    structure_id: str | None = None,
    model: int | None = None,
) -> Structure:
    """Parse the ``_atom_site`` loop of an mmCIF file into a :class:`Structure`.

    This is the preferred reader (see the module docstring). It reads only the
    coordinate loop, but it reads all of it: a row it cannot interpret raises
    :class:`StructureParseError` naming the row and the tag, instead of being
    dropped. A dropped row is typically the cofactor, because cofactors sit at
    the end of the loop after the polymer.

    Author numbering (``auth_asym_id`` / ``auth_seq_id`` / ``auth_comp_id`` /
    ``auth_atom_id``) is preferred over the label_* columns, because author
    numbering is what the literature, the catalytic template and every
    mutation name refer to. The label_* columns are used only as a fallback
    and the substitution is recorded as a parse note -- label_seq_id is a
    1..n index over the entity, not the published numbering, so quietly
    mixing the two is an off-by-dozens error, not an off-by-one.

    ``type_symbol`` is required. Without it the element would have to be
    inferred from an unpadded atom name, which cannot distinguish calcium from
    C-alpha (see :func:`_element_from_pdb_name`); this reader raises instead.
    """
    text, path = _resolve_source(path_or_text, "mmCIF")
    sid = structure_id or (Path(path).stem if path else "inline")

    tags, rows = _extract_atom_site_loop(text)
    index = {t.lower(): k for k, t in enumerate(tags)}

    def col(*names: str) -> int | None:
        for nm in names:
            k = index.get(f"_atom_site.{nm.lower()}")
            if k is not None:
                return k
        return None

    i_x = col("Cartn_x")
    i_y = col("Cartn_y")
    i_z = col("Cartn_z")
    missing = [n for n, k in (("Cartn_x", i_x), ("Cartn_y", i_y), ("Cartn_z", i_z))
               if k is None]
    if missing:
        raise StructureParseError(
            f"mmCIF: the _atom_site loop_ is missing required tag(s): "
            f"{', '.join('_atom_site.' + m for m in missing)}"
        )

    i_el = col("type_symbol")
    if i_el is None:
        raise StructureParseError(
            "mmCIF: the _atom_site loop_ has no _atom_site.type_symbol. The "
            "element cannot be inferred from an unpadded mmCIF atom name "
            "without confusing e.g. calcium with C-alpha, so this file is "
            "rejected rather than guessed at."
        )

    i_group = col("group_PDB")
    i_serial = col("id")
    i_name_auth = col("auth_atom_id")
    i_name_label = col("label_atom_id")
    i_alt = col("label_alt_id", "auth_alt_id", "pdbx_label_alt_id")
    i_comp_auth = col("auth_comp_id")
    i_comp_label = col("label_comp_id")
    i_asym_auth = col("auth_asym_id")
    i_asym_label = col("label_asym_id")
    i_seq_auth = col("auth_seq_id")
    i_seq_label = col("label_seq_id")
    i_icode = col("pdbx_PDB_ins_code")
    i_occ = col("occupancy")
    i_b = col("B_iso_or_equiv")
    i_model = col("pdbx_PDB_model_num")

    if i_name_auth is None and i_name_label is None:
        raise StructureParseError(
            "mmCIF: the _atom_site loop_ carries neither auth_atom_id nor "
            "label_atom_id; atoms cannot be named"
        )
    if i_comp_auth is None and i_comp_label is None:
        raise StructureParseError(
            "mmCIF: the _atom_site loop_ carries neither auth_comp_id nor "
            "label_comp_id; residues cannot be identified"
        )
    if i_asym_auth is None and i_asym_label is None:
        raise StructureParseError(
            "mmCIF: the _atom_site loop_ carries neither auth_asym_id nor "
            "label_asym_id; chains cannot be identified"
        )
    if i_seq_auth is None and i_seq_label is None:
        raise StructureParseError(
            "mmCIF: the _atom_site loop_ carries neither auth_seq_id nor "
            "label_seq_id; residues cannot be numbered"
        )

    notes: list[str] = []
    if i_seq_auth is None:
        notes.append(
            "numbering: auth_seq_id is absent; label_seq_id was used instead. "
            "label_seq_id is an entity-local 1..n index and is NOT the "
            "published author numbering -- every residue number derived from "
            "this structure must be treated as provisional."
        )
    if i_asym_auth is None:
        notes.append(
            "numbering: auth_asym_id is absent; label_asym_id was used as the "
            "chain identifier and will not match chain names in the literature"
        )

    atoms: list[Atom] = []
    models_present: list[int] = []
    for rown, row in enumerate(rows, start=1):
        where = f"mmCIF _atom_site row {rown}"
        group = (_cif_value(row, i_group) or "ATOM").upper()
        if group not in ("ATOM", "HETATM"):
            raise StructureParseError(
                f"{where}: unexpected group_PDB value {group!r}; expected ATOM "
                f"or HETATM"
            )

        x = _float_or_raise(row[i_x], "Cartn_x", where)
        y = _float_or_raise(row[i_y], "Cartn_y", where)
        z = _float_or_raise(row[i_z], "Cartn_z", where)

        element = _cif_value(row, i_el)
        if not element:
            raise StructureParseError(
                f"{where}: _atom_site.type_symbol is null; the element is "
                f"required and is not inferred"
            )

        name = _cif_value(row, i_name_auth) or _cif_value(row, i_name_label)
        if not name:
            raise StructureParseError(f"{where}: atom has no usable atom id")
        resname = _cif_value(row, i_comp_auth) or _cif_value(row, i_comp_label)
        if not resname:
            raise StructureParseError(f"{where}: atom has no usable component id")
        chain_id = _cif_value(row, i_asym_auth) or _cif_value(row, i_asym_label)
        if not chain_id:
            raise StructureParseError(f"{where}: atom has no usable chain id")
        seq_tok = _cif_value(row, i_seq_auth)
        if seq_tok is None:
            seq_tok = _cif_value(row, i_seq_label)
        if seq_tok is None:
            raise StructureParseError(
                f"{where}: both auth_seq_id and label_seq_id are null, so this "
                f"atom cannot be assigned to a numbered residue"
            )
        resseq = _int_or_raise(seq_tok, "residue sequence number", where)

        serial_tok = _cif_value(row, i_serial)
        serial = _int_or_raise(serial_tok, "atom id", where) if serial_tok else rown

        altloc = _cif_value(row, i_alt) or ""
        icode = _cif_value(row, i_icode) or ""
        occ_tok = _cif_value(row, i_occ)
        occupancy = _float_or_raise(occ_tok, "occupancy", where) if occ_tok else None
        b_tok = _cif_value(row, i_b)
        bfactor = _float_or_raise(b_tok, "B_iso_or_equiv", where) if b_tok else None
        model_tok = _cif_value(row, i_model)
        model_num = _int_or_raise(model_tok, "pdbx_PDB_model_num", where) \
            if model_tok else 1
        if model_num not in models_present:
            models_present.append(model_num)

        atoms.append(
            Atom(
                serial=serial,
                name=name.strip(),
                element=element.strip().upper(),
                resname=resname.strip(),
                chain=chain_id.strip(),
                resseq=resseq,
                icode=icode.strip(),
                altloc=altloc.strip(),
                x=x,
                y=y,
                z=z,
                occupancy=occupancy,
                bfactor_or_plddt=bfactor,
                is_hetatm=(group == "HETATM"),
                model=model_num,
                element_inferred=False,
            )
        )

    if not atoms:
        raise StructureParseError("mmCIF: the _atom_site loop_ produced no atoms")

    return _assemble(
        atoms,
        structure_id=sid,
        source_format="mmcif",
        source_path=path,
        models_present=models_present or [1],
        requested_model=model,
        notes=notes,
    )


def read_structure(
    path: str | Path,
    structure_id: str | None = None,
    model: int | None = None,
) -> Structure:
    """Dispatch to :func:`read_mmcif` or :func:`read_pdb` by file extension.

    Convenience only, and deliberately extension-driven rather than
    content-sniffing: a file named ``.cif`` that is actually PDB text is a
    provenance problem the operator should see, not something the reader
    should paper over.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        return read_mmcif(p, structure_id=structure_id, model=model)
    if suffix in (".pdb", ".ent"):
        return read_pdb(p, structure_id=structure_id, model=model)
    raise StructureParseError(
        f"unrecognised structure file extension {suffix!r} for {p}; call "
        f"read_pdb or read_mmcif explicitly"
    )


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------


def _assemble(
    atoms: list[Atom],
    structure_id: str,
    source_format: str,
    source_path: str | None,
    models_present: list[int],
    requested_model: int | None,
    notes: list[str],
) -> Structure:
    """Group atoms of one model into residues and chains, preserving file order."""
    notes = list(notes)
    if requested_model is None:
        selected = models_present[0]
    else:
        if requested_model not in models_present:
            raise StructureParseError(
                f"model {requested_model} is not present; file has "
                f"{models_present}"
            )
        selected = requested_model

    kept = [a for a in atoms if a.model == selected]
    if not kept:
        raise StructureParseError(f"model {selected} contains no atoms")

    chains: list[Chain] = []
    chain_by_id: dict[str, Chain] = {}
    residue_by_key: dict[tuple[str, int, str, str], Residue] = {}
    # Detect residues whose atoms are not contiguous in the file, and author
    # positions reused by two different component names. Both are signs of a
    # merged or hand-edited file, and both break the assumption that a
    # (chain, resseq, icode) handle names one chemical entity.
    last_key: tuple[str, int, str, str] | None = None
    split_residues: set[str] = set()
    author_to_resnames: dict[tuple[str, int, str], set[str]] = {}

    for a in kept:
        key = a.residue_key
        chain = chain_by_id.get(a.chain)
        if chain is None:
            chain = Chain(chain_id=a.chain)
            chain_by_id[a.chain] = chain
            chains.append(chain)
        res = residue_by_key.get(key)
        if res is None:
            res = Residue(chain=a.chain, resname=a.resname, resseq=a.resseq,
                          icode=a.icode)
            residue_by_key[key] = res
            chain.residues.append(res)
        elif last_key != key:
            split_residues.add(str(res))
        res.atoms.append(a)
        author_to_resnames.setdefault(
            (a.chain, a.resseq, a.icode), set()).add(a.resname)
        last_key = key

    for res in residue_by_key.values():
        het = [a.is_hetatm for a in res.atoms]
        res.is_hetatm = all(het)
        if any(het) and not all(het):
            notes.append(
                f"records: residue {res} mixes ATOM and HETATM records; it is "
                f"treated as polymer, which may be wrong"
            )

    if split_residues:
        notes.append(
            f"ordering: atoms of {len(split_residues)} residue(s) are not "
            f"contiguous in the file ({', '.join(sorted(split_residues)[:5])}); "
            f"they were merged, but a non-contiguous residue usually means the "
            f"file was concatenated or hand-edited"
        )
    reused = {k: v for k, v in author_to_resnames.items() if len(v) > 1}
    if reused:
        shown = "; ".join(
            f"{c}/{n}{i}={sorted(v)}" for (c, n, i), v in list(reused.items())[:5]
        )
        notes.append(
            f"numbering: {len(reused)} author position(s) carry more than one "
            f"component name ({shown}); a (chain, resseq, icode) handle is "
            f"therefore ambiguous in this structure"
        )

    return Structure(
        structure_id=structure_id,
        chains=chains,
        source_format=source_format,
        source_path=source_path,
        models_present=models_present,
        model_selected=selected,
        parse_notes=notes,
    )


# ---------------------------------------------------------------------------
# PDB output
# ---------------------------------------------------------------------------


def _pdb_atom_name_field(name: str, element: str) -> str:
    """Place an atom name in columns 13-16 so it reads back identically.

    This mirrors :func:`_element_from_pdb_name`: a two-character element starts
    in column 13, a one-character element is indented by one. Writing the name
    left-justified regardless would turn every written carbon into a potential
    two-letter element on re-reading.
    """
    nm = name.strip()
    if len(nm) > 4:
        raise StructureWriteError(
            f"atom name {nm!r} does not fit the 4-character PDB field; write mmCIF"
        )
    if len(nm) == 4 or len(element.strip()) == 2:
        return nm.ljust(4)
    return (" " + nm).ljust(4)


def write_pdb(
    subject: Structure | Iterable[Atom],
    renumber_serials: bool = False,
    include_end: bool = True,
) -> str:
    """Serialise a structure, or a bare atom selection, as PDB text.

    Intended for round-tripping a *small* selection into an external tool that
    only speaks PDB -- a pocket, a ligand, a pose fragment. mmCIF remains the
    storage format (see the module docstring).

    Anything that does not fit a PDB column raises :class:`StructureWriteError`
    instead of being truncated. Truncation here is especially dangerous because
    the result is a syntactically perfect file describing a different molecule:
    a four-character chemical component id clipped to three can name a
    completely unrelated compound in the PDB chemical dictionary.

    An unknown occupancy or B-factor is written as blanks, not as ``1.00`` /
    ``0.00``, so that re-reading the file yields ``None`` again. A written
    ``1.00`` would be a fabricated measurement.
    """
    if isinstance(subject, Structure):
        atoms = subject.atoms()
        chain_breaks = True
    else:
        atoms = list(subject)
        chain_breaks = False
    if not atoms:
        raise StructureWriteError("nothing to write: the selection is empty")

    lines: list[str] = []
    serial_counter = 0
    prev_chain: str | None = None
    prev_polymer: Atom | None = None

    for a in atoms:
        if len(a.resname.strip()) > 3:
            raise StructureWriteError(
                f"residue name {a.resname!r} is longer than 3 characters and "
                f"cannot be written as PDB; use mmCIF"
            )
        if len(a.chain.strip()) > 1:
            raise StructureWriteError(
                f"chain id {a.chain!r} is longer than 1 character and cannot be "
                f"written as PDB; use mmCIF"
            )
        if not -999 <= a.resseq <= 9999:
            raise StructureWriteError(
                f"residue number {a.resseq} is outside the PDB field range "
                f"(-999..9999); use mmCIF"
            )
        for axis, value in (("x", a.x), ("y", a.y), ("z", a.z)):
            if not -999.999 <= value <= 9999.999:
                raise StructureWriteError(
                    f"{axis} coordinate {value} does not fit the 8.3 PDB field"
                )

        serial_counter += 1
        serial = serial_counter if renumber_serials else a.serial
        if not 0 <= serial <= 99999:
            if renumber_serials:
                raise StructureWriteError(
                    f"serial {serial} exceeds the PDB field; the selection is "
                    f"too large for PDB output"
                )
            raise StructureWriteError(
                f"atom serial {serial} does not fit the PDB field; pass "
                f"renumber_serials=True if renumbering is acceptable"
            )

        if chain_breaks and prev_chain is not None and a.chain != prev_chain \
                and prev_polymer is not None:
            lines.append(_ter_line(serial_counter, prev_polymer))
            prev_polymer = None

        record = "HETATM" if a.is_hetatm else "ATOM  "
        occ = "      " if a.occupancy is None else f"{a.occupancy:6.2f}"
        bfac = "      " if a.bfactor_or_plddt is None else f"{a.bfactor_or_plddt:6.2f}"
        lines.append(
            f"{record}"
            f"{serial:5d}"
            f" "
            f"{_pdb_atom_name_field(a.name, a.element)}"
            f"{(a.altloc or ' ')[:1]}"
            f"{a.resname.strip():>3}"
            f" "
            f"{(a.chain.strip() or ' ')[:1]}"
            f"{a.resseq:4d}"
            f"{(a.icode or ' ')[:1]}"
            f"   "
            f"{a.x:8.3f}{a.y:8.3f}{a.z:8.3f}"
            f"{occ}{bfac}"
            f"          "
            f"{a.element.strip().upper():>2}"
        )
        prev_chain = a.chain
        if not a.is_hetatm:
            prev_polymer = a

    if chain_breaks and prev_polymer is not None:
        lines.append(_ter_line(serial_counter + 1, prev_polymer))
    if include_end:
        lines.append("END")
    return "\n".join(lines) + "\n"


def _ter_line(serial: int, last: Atom) -> str:
    serial = min(max(serial, 0), 99999)
    return (
        f"TER   {serial:5d}      {last.resname.strip():>3} "
        f"{(last.chain.strip() or ' ')[:1]}{last.resseq:4d}{(last.icode or ' ')[:1]}"
    )
