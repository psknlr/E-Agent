"""Residue numbering: the mapping that stops off-by-one mutation disasters.

Three numbering systems are in play at once in this pipeline and they rarely
agree:

1. **Candidate index** -- 0-based position in the mined protein sequence. This
   is what ``MutationProposal.apply_to`` indexes and what the ordered library
   is actually built from.
2. **Author numbering** -- the ``(chain, resseq, icode)`` triple written in the
   structure file. This is what the literature, the catalytic template and
   every mutation name in a paper refer to. It starts wherever the depositor
   decided, skips the residues that were disordered, and can carry insertion
   codes.
3. **Reference numbering** -- positions in whichever sequence the family's
   template was annotated against, so that "the catalytic Tyr155 of the SDR
   family" means something for a candidate whose own tyrosine is at index 148.

Nothing lines these up for free. A construct with an N-terminal His-tag shifts
the index by 20+. A structure missing a disordered loop shifts author numbers
relative to the sequence by the length of the gap. A reference numbering that
skips an insertion shifts again. Each shift is an off-by-N error that produces
a mutation at the wrong residue -- which does not fail loudly: it produces a
variant that expresses, folds, and simply does not work, after the plate has
already been made.

This module therefore does three things and refuses to do them implicitly:

* aligns the candidate sequence to the sequence actually observed in the
  structure (:func:`needleman_wunsch`, :func:`build_map`),
* records the unobserved regions explicitly rather than pretending the gap is
  not there (:meth:`ResidueMap.unobserved_regions`),
* makes a wrong wild-type letter impossible to push through silently
  (:func:`verify_residue`).

Pure Python: biopython and numpy are both absent, and a Gotoh affine-gap
alignment of two ~600-residue proteins is fast enough here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from ..errors import EAgentError
from .structure_io import Chain, Residue

__all__ = [
    "NumberingError",
    "THREE_TO_ONE",
    "ONE_TO_THREE",
    "MODIFIED_RESIDUE_PARENT",
    "three_to_one",
    "one_to_three",
    "residue_one_letter",
    "letter_of_residue_token",
    "Alignment",
    "needleman_wunsch",
    "alignment_score",
    "AuthorPosition",
    "ResidueMap",
    "build_map",
    "verify_residue",
]


class NumberingError(EAgentError):
    """A residue numbering operation is inconsistent with the sequence.

    Raised rather than returning a sentinel. An out-of-range index or a
    wild-type letter that does not match is always a bug in the caller or in
    the mapping, never ordinary missing data, and the one thing it must not
    do is continue quietly.
    """


# ---------------------------------------------------------------------------
# residue name tables
# ---------------------------------------------------------------------------

#: The twenty standard amino acids, plus the explicit "unknown residue" code.
#: ``UNK`` maps to ``X`` rather than being dropped: a chain with an unknown
#: residue in the middle still occupies that position, and silently deleting
#: it would shift every author number after it by one.
THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "UNK": "X",
    # Ambiguity codes that appear in older entries. They are placeholders, not
    # residues, and keeping them as X preserves the position count.
    "ASX": "X", "GLX": "X", "XAA": "X",
    # Protonation-state variants written by simulation packages. Same residue,
    # different naming convention; dropping them would shift the numbering.
    "HID": "H", "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H",
    "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K",
}

#: Modified residues that are part of the polymer backbone, mapped to the
#: standard residue they substitute for.
#:
#: Why this table exists: these are written as HETATM in many entries. A
#: sequence builder that keeps only ATOM records would drop a
#: selenomethionine, and every residue after it would be misnumbered -- the
#: classic way a SeMet-phased structure produces mutations at the wrong
#: position. Each substitution made is recorded in ``ResidueMap.notes``, so a
#: reviewer can see that the chemistry at that position is not the standard
#: residue even though the position is.
MODIFIED_RESIDUE_PARENT: dict[str, str] = {
    "MSE": "M",   # selenomethionine
    "SEP": "S",   # phosphoserine
    "TPO": "T",   # phosphothreonine
    "PTR": "Y",   # phosphotyrosine
    "CSO": "C",   # S-hydroxycysteine
    "CSD": "C",   # S-cysteinesulfinic acid
    "CME": "C",   # S,S-(2-hydroxyethyl)thiocysteine
    "OCS": "C",   # cysteinesulfonic acid
    "KCX": "K",   # lysine carboxylic acid (carbamylated lysine)
    "LLP": "K",   # lysine-pyridoxal-5'-phosphate
    "MLY": "K",   # N-dimethyl-lysine
    "M3L": "K",   # N-trimethyl-lysine
    "ALY": "K",   # N-acetyl-lysine
    "HIC": "H",   # 4-methyl-histidine
    "MHO": "M",   # methionine sulfoxide
    "HYP": "P",   # 4-hydroxyproline
    "PCA": "E",   # pyroglutamic acid (cyclised glutamate)
    "FME": "M",   # N-formylmethionine
    "SAC": "S",   # N-acetylserine
    "TYS": "Y",   # O-sulfo-tyrosine
    "SMC": "C",   # S-methylcysteine
}

#: One-letter -> canonical three-letter code, for writing mutation names and
#: for building a residue selection string. ``X`` goes to ``UNK``.
ONE_TO_THREE: dict[str, str] = {v: k for k, v in THREE_TO_ONE.items()
                                if k not in ("ASX", "GLX", "XAA", "HID", "HIE",
                                             "HIP", "HSD", "HSE", "HSP", "CYX",
                                             "CYM", "ASH", "GLH", "LYN")}

#: The alphabet a mutation may legally use. Deliberately excludes ``X``:
#: "mutate position 155 to unknown" is not a proposal.
STANDARD_ONE_LETTER: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")


def three_to_one(resname: str, strict: bool = True) -> str:
    """Three-letter residue name -> one-letter code.

    Recognises the standard twenty, the ``UNK``/``ASX``/``GLX`` placeholders,
    common protonation-state aliases, and the modified residues in
    :data:`MODIFIED_RESIDUE_PARENT`.

    With ``strict=True`` (the default) an unrecognised name raises. That is
    the safe default for sequence building: the alternative -- returning
    ``"X"`` for anything unknown -- would turn a bound ligand that happens to
    sit inside the chain into a residue and shift everything after it.
    :func:`residue_one_letter` is the non-strict variant used when the caller
    genuinely needs to ask "is this a polymer residue at all?".
    """
    key = (resname or "").strip().upper()
    if key in THREE_TO_ONE:
        return THREE_TO_ONE[key]
    if key in MODIFIED_RESIDUE_PARENT:
        return MODIFIED_RESIDUE_PARENT[key]
    if strict:
        raise NumberingError(
            f"unknown residue name {resname!r}; it is neither a standard amino "
            f"acid nor a modified residue in MODIFIED_RESIDUE_PARENT. Add it to "
            f"the table if it is part of the polymer -- do not let it be "
            f"dropped, because dropping a backbone residue shifts every "
            f"position after it."
        )
    return "X"


def one_to_three(letter: str) -> str:
    """One-letter code -> canonical three-letter name. Raises on anything else."""
    key = (letter or "").strip().upper()
    if len(key) != 1 or key not in ONE_TO_THREE:
        raise NumberingError(f"not a one-letter residue code: {letter!r}")
    return ONE_TO_THREE[key]


def residue_one_letter(resname: str) -> str | None:
    """One-letter code for a polymer residue, or ``None`` if it is not one.

    ``None`` is the answer for water, a cofactor, a metal and anything else
    that is not part of the chain. Separating this from :func:`three_to_one`
    keeps the two questions distinct: "what letter is this residue?" must be
    strict, while "is this thing a residue?" must be allowed to say no.
    """
    key = (resname or "").strip().upper()
    if key in THREE_TO_ONE:
        return THREE_TO_ONE[key]
    if key in MODIFIED_RESIDUE_PARENT:
        return MODIFIED_RESIDUE_PARENT[key]
    return None


def letter_of_residue_token(token: str | None) -> str | None:
    """The amino acid in a residue token like ``Y155``, ``TYR155`` or ``Y``.

    ``None`` when the token names no recognisable residue. Used wherever a
    recorded token has to be checked against a sequence: the *number* in such
    a token is author numbering and carries no meaning outside the structure
    it came from, while the letter is the chemistry and can be verified
    against the residue that is actually there.
    """
    text = "".join(str(token or "").split()).upper()
    if not text:
        return None
    stem = text.rstrip("0123456789")
    if len(stem) == 1:
        return stem if stem in ONE_TO_THREE else None
    if len(stem) == 3:
        return residue_one_letter(stem)
    return None


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------


class Alignment(NamedTuple):
    """Result of :func:`needleman_wunsch`.

    A three-field NamedTuple so that ``a, b, ident = needleman_wunsch(...)``
    works, with everything else derived from the two aligned strings rather
    than stored. Derived rather than stored on purpose: a cached statistic
    can disagree with the strings it claims to describe once someone edits
    one of them.
    """

    aligned_a: str
    aligned_b: str
    identity: float

    @property
    def length(self) -> int:
        """Number of alignment columns, gaps included."""
        return len(self.aligned_a)

    @property
    def n_aligned_columns(self) -> int:
        """Columns where neither sequence has a gap."""
        return sum(1 for x, y in zip(self.aligned_a, self.aligned_b)
                   if x != "-" and y != "-")

    @property
    def n_identities(self) -> int:
        """Columns where both sequences carry the same residue."""
        return sum(1 for x, y in zip(self.aligned_a, self.aligned_b)
                   if x != "-" and x == y)

    @property
    def n_gaps(self) -> int:
        """Columns holding a gap on either side. Reported alongside
        ``identity``, because a high identity over a handful of aligned
        columns is not a good alignment."""
        return sum(1 for x, y in zip(self.aligned_a, self.aligned_b)
                   if x == "-" or y == "-")

    def coverage_a(self) -> float:
        """Fraction of sequence a that is aligned to a residue of b."""
        la = sum(1 for x in self.aligned_a if x != "-")
        return self.n_aligned_columns / la if la else 0.0

    def coverage_b(self) -> float:
        """Fraction of sequence b that is aligned to a residue of a."""
        lb = sum(1 for y in self.aligned_b if y != "-")
        return self.n_aligned_columns / lb if lb else 0.0


def needleman_wunsch(
    seq_a: str,
    seq_b: str,
    match: float = 2.0,
    mismatch: float = -1.0,
    gap_open: float = 10.0,
    gap_extend: float = 0.5,
) -> Alignment:
    """Global pairwise alignment with affine gaps (Gotoh), in pure Python.

    SIGN CONVENTION, stated because getting it backwards produces an
    alignment that looks fine and is wrong: ``match`` and ``mismatch`` are
    *scores* added to the total (match positive, mismatch usually negative),
    while ``gap_open`` and ``gap_extend`` are *penalty magnitudes* and must be
    non-negative -- they are subtracted. A gap of length L costs
    ``gap_open + L * gap_extend``.

    Affine rather than linear gaps because the thing this alignment is for is
    a structure missing a disordered loop: one eight-residue gap and eight
    scattered one-residue gaps score identically under a linear penalty, and
    only the first is biologically what happened. Scattered gaps are exactly
    the pattern that misnumbers residues either side of them.

    Global (Needleman-Wunsch) rather than local because the candidate sequence
    and the observed sequence are meant to be the same protein; a local
    alignment would happily trim a terminal mismatch that is actually
    diagnostic of the wrong structure having been picked.

    Time is O(n*m). Memory is O(n*m) for the traceback pointers plus O(m) for
    the scores -- the score matrices are kept one row at a time, which is what
    makes a 600 x 600 alignment comfortable without numpy.

    Raises on an empty sequence: an alignment to nothing has no meaningful
    identity, and returning 0.0 would let a downstream identity filter treat
    "nothing to compare" as "poor match".
    """
    a = "".join((seq_a or "").split()).upper()
    b = "".join((seq_b or "").split()).upper()
    if not a or not b:
        raise NumberingError(
            "cannot align an empty sequence; check that the structure chain "
            "actually contains polymer residues"
        )
    if gap_open < 0 or gap_extend < 0:
        raise NumberingError(
            f"gap_open ({gap_open}) and gap_extend ({gap_extend}) are penalty "
            f"magnitudes and must be >= 0; they are subtracted, not added"
        )

    n, m = len(a), len(b)
    neg = float("-inf")
    first = gap_open + gap_extend          # cost of opening a gap of length 1

    # Row-major pointer matrices: which state we came from (0=M, 1=Ix, 2=Iy).
    ptr_m = [[0] * (m + 1) for _ in range(n + 1)]
    ptr_x = [[0] * (m + 1) for _ in range(n + 1)]
    ptr_y = [[0] * (m + 1) for _ in range(n + 1)]

    prev_m = [neg] * (m + 1)
    prev_x = [neg] * (m + 1)
    prev_y = [neg] * (m + 1)
    prev_m[0] = 0.0
    for j in range(1, m + 1):
        prev_y[j] = -(gap_open + j * gap_extend)
        ptr_y[0][j] = 0 if j == 1 else 2

    for i in range(1, n + 1):
        cur_m = [neg] * (m + 1)
        cur_x = [neg] * (m + 1)
        cur_y = [neg] * (m + 1)
        cur_x[0] = -(gap_open + i * gap_extend)
        ptr_x[i][0] = 0 if i == 1 else 1
        ai = a[i - 1]
        row_ptr_m = ptr_m[i]
        row_ptr_x = ptr_x[i]
        row_ptr_y = ptr_y[i]
        for j in range(1, m + 1):
            s = match if ai == b[j - 1] else mismatch

            best = prev_m[j - 1]
            src = 0
            if prev_x[j - 1] > best:
                best, src = prev_x[j - 1], 1
            if prev_y[j - 1] > best:
                best, src = prev_y[j - 1], 2
            cur_m[j] = best + s
            row_ptr_m[j] = src

            # Ix: a[i-1] against a gap in b -> came from row i-1, column j
            best = prev_m[j] - first
            src = 0
            cand = prev_x[j] - gap_extend
            if cand > best:
                best, src = cand, 1
            cand = prev_y[j] - first
            if cand > best:
                best, src = cand, 2
            cur_x[j] = best
            row_ptr_x[j] = src

            # Iy: b[j-1] against a gap in a -> came from this row, column j-1
            best = cur_m[j - 1] - first
            src = 0
            cand = cur_y[j - 1] - gap_extend
            if cand > best:
                best, src = cand, 2
            cand = cur_x[j - 1] - first
            if cand > best:
                best, src = cand, 1
            cur_y[j] = best
            row_ptr_y[j] = src

        prev_m, prev_x, prev_y = cur_m, cur_x, cur_y

    # Traceback from the best terminal state.
    state = 0
    best = prev_m[m]
    if prev_x[m] > best:
        best, state = prev_x[m], 1
    if prev_y[m] > best:
        best, state = prev_y[m], 2

    out_a: list[str] = []
    out_b: list[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        # Forced states at the edges. The initialisation guarantees these, but
        # asserting them here means a pointer bug cannot silently emit an
        # alignment with the wrong number of residues.
        if i == 0:
            state = 2
        elif j == 0:
            state = 1
        if state == 0:
            out_a.append(a[i - 1])
            out_b.append(b[j - 1])
            state = ptr_m[i][j]
            i -= 1
            j -= 1
        elif state == 1:
            out_a.append(a[i - 1])
            out_b.append("-")
            state = ptr_x[i][j]
            i -= 1
        else:
            out_a.append("-")
            out_b.append(b[j - 1])
            state = ptr_y[i][j]
            j -= 1

    aligned_a = "".join(reversed(out_a))
    aligned_b = "".join(reversed(out_b))
    if aligned_a.replace("-", "") != a or aligned_b.replace("-", "") != b:
        raise NumberingError(
            "internal alignment error: the traceback did not reproduce the "
            "input sequences. Refusing to return a numbering built on it."
        )

    paired = [(x, y) for x, y in zip(aligned_a, aligned_b)
              if x != "-" and y != "-"]
    identity = (sum(1 for x, y in paired if x == y) / len(paired)) if paired else 0.0
    return Alignment(aligned_a, aligned_b, identity)


def alignment_score(
    aligned_a: str,
    aligned_b: str,
    match: float = 2.0,
    mismatch: float = -1.0,
    gap_open: float = 10.0,
    gap_extend: float = 0.5,
) -> float:
    """Re-score an existing alignment under the same affine model.

    Kept separate from :func:`needleman_wunsch` so that the result type stays
    a plain three-tuple, and so that an alignment produced elsewhere (by hand,
    or by an external tool) can be scored under exactly the parameters this
    module uses -- comparing scores from two different gap models is a common
    way to "show" that one alignment is better than another when it is not.
    """
    if len(aligned_a) != len(aligned_b):
        raise NumberingError("aligned strings have different lengths")
    total = 0.0
    in_gap_a = False
    in_gap_b = False
    for x, y in zip(aligned_a, aligned_b):
        if x == "-" and y == "-":
            continue
        if x == "-":
            total -= gap_extend + (0.0 if in_gap_a else gap_open)
            in_gap_a, in_gap_b = True, False
        elif y == "-":
            total -= gap_extend + (0.0 if in_gap_b else gap_open)
            in_gap_a, in_gap_b = False, True
        else:
            total += match if x == y else mismatch
            in_gap_a = in_gap_b = False
    return total


# ---------------------------------------------------------------------------
# residue map
# ---------------------------------------------------------------------------


class AuthorPosition(NamedTuple):
    """A position in the structure's author numbering.

    The insertion code is part of the identity, not decoration. ``100`` and
    ``100A`` are different residues, and a map keyed on the integer alone
    would collide them -- which in an antibody or a protease numbering scheme
    means mutating a neighbour of the intended residue.
    """

    chain: str
    resseq: int
    icode: str = ""

    def __str__(self) -> str:
        return f"{self.chain}/{self.resseq}{self.icode.strip()}"

    @property
    def token(self) -> str:
        """Compact form used in mutation names, e.g. ``155`` or ``100A``."""
        return f"{self.resseq}{self.icode.strip()}"


@dataclass
class ResidueMap:
    """Bidirectional map between candidate index, author numbering and reference.

    Built by :func:`build_map`. Three facts it holds that a bare dict would
    not:

    * which candidate indices are **not** observed in the structure, as
      explicit regions rather than as absent dict keys, so that "no structural
      evidence here" can be reported instead of inferred from silence;
    * which observed residues **disagree** with the candidate sequence, i.e.
      the structure is a mutant or a homologue of the candidate;
    * which observed residues are **not in the candidate at all** (a tag, a
      fusion partner, the wrong chain picked).

    All three are the difference between a mapping you can defend and a
    mapping that happens to produce a number.
    """

    candidate_sequence: str
    chain_id: str
    index_to_author: dict[int, AuthorPosition] = field(default_factory=dict)
    author_to_index: dict[AuthorPosition, int] = field(default_factory=dict)
    index_to_reference: dict[int, int] = field(default_factory=dict)
    reference_to_index: dict[int, int] = field(default_factory=dict)
    reference_label: str | None = None
    structure_identity: float | None = None
    reference_identity: float | None = None
    #: (index, candidate letter, structure letter, author position)
    mismatches: list[tuple[int, str, str, AuthorPosition]] = field(default_factory=list)
    #: Observed residues that found no place in the candidate sequence.
    unmapped_structure_positions: list[AuthorPosition] = field(default_factory=list)
    #: Candidate indices whose author assignment is not uniquely determined.
    #: A run of identical residues that is only partly observed admits several
    #: equal-scoring alignments; the aligner must pick one, and the one it
    #: picks is not evidence. See :meth:`is_ambiguous`.
    ambiguous_indices: frozenset[int] = frozenset()
    notes: list[str] = field(default_factory=list)

    # -- queries -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.candidate_sequence)

    def is_ambiguous(self, index: int) -> bool:
        """Whether this index's structural assignment could equally be another.

        True inside a run of identical residues that the structure only
        partly observes. Two adjacent tyrosines with coordinates for one of
        them admit two alignments of identical score, so which tyrosine the
        author number refers to is not determined by the sequence. The
        aligner still has to choose, and a mutation designed against the
        choice is a mutation at a position nobody established.
        """
        return index in self.ambiguous_indices

    def ambiguity_note(self, index: int) -> str | None:
        """Why this index is ambiguous, for a report or a refusal."""
        if index not in self.ambiguous_indices:
            return None
        letter = self.candidate_sequence[index]
        run = sorted(i for i in self.ambiguous_indices
                     if self.candidate_sequence[i] == letter
                     and abs(i - index) < len(self.candidate_sequence))
        return (f"index {index} ({letter}{index + 1}) sits in a run of "
                f"identical residues that the structure only partly observes "
                f"(candidate indices {run}); several alignments score "
                f"identically, so which one the author numbering refers to is "
                f"not determined here")

    def _check_index(self, index: int) -> None:
        if not isinstance(index, int) or isinstance(index, bool):
            raise NumberingError(f"candidate index must be an int, got {index!r}")
        if not 0 <= index < len(self.candidate_sequence):
            raise NumberingError(
                f"candidate index {index} is outside the sequence "
                f"(0..{len(self.candidate_sequence) - 1}). If this came from a "
                f"1-based author number, convert it first -- do not subtract 1 "
                f"and hope."
            )

    def letter_at(self, index: int) -> str:
        """Candidate residue letter at a 0-based index."""
        self._check_index(index)
        return self.candidate_sequence[index]

    def to_author(self, index: int) -> AuthorPosition | None:
        """Author position for a candidate index, or ``None`` if unobserved.

        ``None`` means "this residue exists in the sequence but not in the
        structure". It is not an error -- disordered termini and loops are
        normal -- but it does mean no structural statement can be made about
        the position, and a caller that silently treats ``None`` as 0 will
        mutate the first residue of the chain.
        """
        self._check_index(index)
        return self.index_to_author.get(index)

    def to_index(
        self, author: AuthorPosition | tuple | int, icode: str = "",
        chain: str | None = None,
    ) -> int | None:
        """Candidate index for an author position, or ``None`` if not mapped.

        Accepts an :class:`AuthorPosition`, a ``(chain, resseq, icode)`` or
        ``(resseq, icode)`` tuple, or a bare residue number (which uses this
        map's chain). Returning ``None`` rather than raising is deliberate:
        a template may name a residue that this candidate's structure does not
        resolve, and that is information, not a crash.
        """
        pos = self._coerce_author(author, icode, chain)
        return self.author_to_index.get(pos)

    def _coerce_author(self, author: AuthorPosition | tuple | int,
                       icode: str = "", chain: str | None = None) -> AuthorPosition:
        if isinstance(author, AuthorPosition):
            return author
        ch = chain if chain is not None else self.chain_id
        if isinstance(author, int) and not isinstance(author, bool):
            return AuthorPosition(ch, author, icode.strip())
        if isinstance(author, tuple):
            if len(author) == 3:
                return AuthorPosition(str(author[0]), int(author[1]),
                                      str(author[2]).strip())
            if len(author) == 2:
                return AuthorPosition(ch, int(author[0]), str(author[1]).strip())
        raise NumberingError(
            f"cannot interpret {author!r} as an author position; pass an "
            f"AuthorPosition, (chain, resseq, icode), (resseq, icode) or an int"
        )

    def is_observed(self, index: int) -> bool:
        """Whether this candidate residue has coordinates in the structure."""
        self._check_index(index)
        return index in self.index_to_author

    def observed_indices(self) -> list[int]:
        """Candidate indices that have coordinates, ascending. The complement
        of :meth:`unobserved_regions`, and the only positions about which a
        structural statement may be made."""
        return sorted(self.index_to_author)

    @property
    def coverage(self) -> float:
        """Fraction of the candidate sequence with coordinates."""
        n = len(self.candidate_sequence)
        return len(self.index_to_author) / n if n else 0.0

    def unobserved_regions(self) -> list[tuple[int, int]]:
        """Contiguous unobserved candidate index ranges, inclusive.

        Returned as regions rather than as a set of indices because the shape
        matters: a 40-residue unobserved stretch in the middle of a chain is a
        disordered loop that may form part of the pocket, while the same
        count spread over both termini is a construct artefact. A mutation
        proposal inside one of these regions is a proposal about a residue
        nobody has seen.
        """
        regions: list[tuple[int, int]] = []
        start: int | None = None
        for i in range(len(self.candidate_sequence)):
            if i in self.index_to_author:
                if start is not None:
                    regions.append((start, i - 1))
                    start = None
            elif start is None:
                start = i
        if start is not None:
            regions.append((start, len(self.candidate_sequence) - 1))
        return regions

    # -- reference numbering ----------------------------------------------
    def to_reference(self, index: int) -> int | None:
        """1-based position in the reference sequence, or ``None``."""
        self._check_index(index)
        return self.index_to_reference.get(index)

    def from_reference(self, reference_number: int) -> int | None:
        """Candidate index for a 1-based reference position, or ``None``.

        This is how a template that says "the catalytic tyrosine is Y155"
        reaches the right residue of a candidate whose own numbering differs.
        ``None`` means the candidate has no residue aligned to that reference
        position -- a deletion relative to the reference -- and the catalytic
        role must then be reported as missing, not mapped to a neighbour.
        """
        return self.reference_to_index.get(int(reference_number))

    # -- reporting ---------------------------------------------------------
    def describe(self) -> str:
        """One-line summary for a manifest note or a QC flag."""
        gaps = self.unobserved_regions()
        parts = [
            f"chain {self.chain_id}: {len(self.index_to_author)}/"
            f"{len(self.candidate_sequence)} residues observed "
            f"(coverage {self.coverage:.0%})"
        ]
        if self.structure_identity is not None:
            parts.append(f"identity to structure {self.structure_identity:.1%}")
        parts.append(f"{len(gaps)} unobserved region(s)")
        parts.append(f"{len(self.mismatches)} sequence mismatch(es)")
        if self.reference_label:
            parts.append(f"reference {self.reference_label}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def _observed_sequence(
    chain: Chain, notes: list[str]
) -> tuple[str, list[AuthorPosition]]:
    """Derive the SEQRES-equivalent from the residues actually present.

    "SEQRES-equivalent" rather than SEQRES: the SEQRES record describes the
    construct that went into the tube, including residues that were never
    resolved. What we need here is the sequence of residues that have
    coordinates, because those are the only ones a geometric measurement can
    refer to. The difference between the two is exactly the unobserved
    regions this module exists to make explicit.

    Residues are taken in file order, not sorted by number -- see
    :class:`~eagent.science.structure_io.Chain` for why -- and non-monotonic
    numbering is reported as a note instead.
    """
    letters: list[str] = []
    positions: list[AuthorPosition] = []
    skipped: list[str] = []
    modified: list[str] = []
    seen: set[tuple[int, str]] = set()

    for res in chain.residues:
        letter = residue_one_letter(res.resname)
        if letter is None:
            if not res.is_water:
                skipped.append(str(res))
            continue
        key = (res.resseq, res.icode.strip())
        if key in seen:
            notes.append(
                f"numbering: author position {res.resseq}{res.icode.strip()} of "
                f"chain {chain.chain_id} appears more than once; only the first "
                f"occurrence is mapped and the duplicate is reported here "
                f"rather than overwriting it"
            )
            continue
        seen.add(key)
        if res.resname.strip().upper() in MODIFIED_RESIDUE_PARENT:
            modified.append(f"{res.resname.strip()}{res.resseq}->{letter}")
        letters.append(letter)
        positions.append(
            AuthorPosition(chain.chain_id, res.resseq, res.icode.strip())
        )

    if skipped:
        notes.append(
            f"chain {chain.chain_id}: {len(skipped)} non-polymer group(s) "
            f"excluded from the observed sequence "
            f"({', '.join(sorted(set(skipped))[:8])}); they remain available "
            f"as ligands"
        )
    if modified:
        notes.append(
            f"chain {chain.chain_id}: {len(modified)} modified residue(s) were "
            f"counted as their parent amino acid to preserve the numbering "
            f"({', '.join(modified[:8])}); the chemistry at those positions is "
            f"not the standard residue"
        )

    numbers = [p.resseq for p in positions]
    if any(b < a for a, b in zip(numbers, numbers[1:])):
        notes.append(
            f"numbering: chain {chain.chain_id} author numbering is not "
            f"monotonically increasing in file order; the observed sequence "
            f"follows file order, which is the polymer order, but every "
            f"residue number derived from this chain should be spot-checked"
        )

    return "".join(letters), positions


def build_map(
    candidate_sequence: str,
    structure_chain: Chain,
    reference_sequence: str | None = None,
    reference_label: str | None = None,
    match: float = 2.0,
    mismatch: float = -1.0,
    gap_open: float = 10.0,
    gap_extend: float = 0.5,
) -> ResidueMap:
    """Align a candidate sequence to a structure chain and build the map.

    The structure's observed residues are turned into a sequence (see
    :func:`_observed_sequence`) and globally aligned to the candidate. Every
    column where both sides have a residue becomes a bidirectional
    index <-> author-position entry. Columns where only the candidate has a
    residue become unobserved indices. Columns where only the structure has a
    residue are recorded in ``unmapped_structure_positions``.

    That last case deserves attention rather than a shrug: an observed residue
    absent from the candidate means the structure carries something the
    candidate does not -- a purification tag, a fusion partner, or, most
    often, the wrong chain or the wrong entry. Reporting it is how that gets
    caught before a mutation is ordered.

    Sequence mismatches in aligned columns are recorded, not corrected. A
    structure that is a point mutant of the candidate is usable, but the
    pipeline has to know, because the catalytic residue it is measuring may
    be the one that was mutated away.

    ``reference_sequence`` optionally adds the third axis: the candidate is
    aligned to it as well, giving ``to_reference`` / ``from_reference``, which
    is how a family template's "Tyr155" finds the right residue in a protein
    numbered differently.
    """
    cand = "".join((candidate_sequence or "").split()).upper()
    if not cand:
        raise NumberingError("candidate sequence is empty")
    if structure_chain is None or not getattr(structure_chain, "residues", None):
        raise NumberingError(
            "structure chain has no residues; a map cannot be built and must "
            "not be faked with an identity mapping"
        )

    notes: list[str] = []
    observed, positions = _observed_sequence(structure_chain, notes)
    if not observed:
        raise NumberingError(
            f"chain {structure_chain.chain_id} contains no polymer residues "
            f"(only ligands/water?); there is nothing to map the candidate onto"
        )

    aln = needleman_wunsch(cand, observed, match, mismatch, gap_open, gap_extend)

    rmap = ResidueMap(
        candidate_sequence=cand,
        chain_id=structure_chain.chain_id,
        reference_label=reference_label,
        structure_identity=aln.identity,
        notes=notes,
    )

    ci = 0
    oi = 0
    for ca, cb in zip(aln.aligned_a, aln.aligned_b):
        if ca != "-" and cb != "-":
            pos = positions[oi]
            rmap.index_to_author[ci] = pos
            rmap.author_to_index[pos] = ci
            if ca != cb:
                rmap.mismatches.append((ci, ca, cb, pos))
            ci += 1
            oi += 1
        elif ca != "-":
            ci += 1
        else:
            rmap.unmapped_structure_positions.append(positions[oi])
            oi += 1

    # Which assignments the alignment could not have determined. A maximal
    # run of identical residues that the structure observes only in part
    # admits several alignments of identical score: with two adjacent
    # tyrosines and coordinates for one, placing the observed one first or
    # second scores the same. The aligner must still choose, and the choice
    # it makes carries no information, so every mapped index in such a run is
    # marked rather than silently trusted.
    ambiguous: set[int] = set()
    start = 0
    while start < len(cand):
        end = start
        while end + 1 < len(cand) and cand[end + 1] == cand[start]:
            end += 1
        if end > start:
            run = range(start, end + 1)
            mapped = [i for i in run if i in rmap.index_to_author]
            if mapped and len(mapped) < (end - start + 1):
                ambiguous.update(mapped)
        start = end + 1
    rmap.ambiguous_indices = frozenset(ambiguous)
    if ambiguous:
        shown = ", ".join(str(i) for i in sorted(ambiguous)[:8])
        rmap.notes.append(
            f"numbering: candidate indices {shown} lie in runs of identical "
            f"residues that the structure observes only in part, so their "
            f"author assignment is one of several equally-scoring "
            f"alignments. A position here is not established and must not be "
            f"reported as observed or used to name a mutation without a "
            f"curator resolving it"
        )

    if rmap.mismatches:
        shown = ", ".join(
            f"index {i} candidate {c} vs structure {s} at {p}"
            for i, c, s, p in rmap.mismatches[:5]
        )
        rmap.notes.append(
            f"sequence: {len(rmap.mismatches)} aligned position(s) differ "
            f"between the candidate and the structure ({shown}); the structure "
            f"is a variant or a homologue of the candidate, not the candidate"
        )
    if rmap.unmapped_structure_positions:
        shown = ", ".join(str(p) for p in rmap.unmapped_structure_positions[:8])
        rmap.notes.append(
            f"coverage: {len(rmap.unmapped_structure_positions)} observed "
            f"residue(s) have no counterpart in the candidate sequence "
            f"({shown}); check for a tag, a fusion, or the wrong chain"
        )
    gaps = rmap.unobserved_regions()
    if gaps:
        shown = ", ".join(f"{s}-{e}" for s, e in gaps[:8])
        rmap.notes.append(
            f"coverage: candidate indices {shown} have no coordinates "
            f"({len(rmap.candidate_sequence) - len(rmap.index_to_author)} "
            f"residue(s) in {len(gaps)} region(s)); no structural claim can be "
            f"made about them"
        )

    if reference_sequence is not None:
        ref = "".join(reference_sequence.split()).upper()
        if not ref:
            raise NumberingError(
                "reference_sequence was supplied but is empty; pass None if "
                "there is no reference rather than an empty string"
            )
        ref_aln = needleman_wunsch(cand, ref, match, mismatch, gap_open, gap_extend)
        rmap.reference_identity = ref_aln.identity
        ci = 0
        ri = 0
        for ca, cr in zip(ref_aln.aligned_a, ref_aln.aligned_b):
            if ca != "-" and cr != "-":
                rmap.index_to_reference[ci] = ri + 1      # reference is 1-based
                rmap.reference_to_index[ri + 1] = ci
                ci += 1
                ri += 1
            elif ca != "-":
                ci += 1
            else:
                ri += 1
        rmap.notes.append(
            f"reference: candidate aligned to "
            f"{reference_label or 'the supplied reference'} at "
            f"{ref_aln.identity:.1%} identity over {ref_aln.n_aligned_columns} "
            f"position(s); reference numbering is 1-based"
        )

    return rmap


# ---------------------------------------------------------------------------
# the guard
# ---------------------------------------------------------------------------


def verify_residue(
    residue_map: ResidueMap,
    index: int,
    expected_letter: str,
    require_observed: bool = False,
) -> AuthorPosition | None:
    """Assert that the candidate really has ``expected_letter`` at ``index``.

    Mutation proposals call this before they are allowed to exist. The failure
    it prevents is the quietest and most expensive one in the whole pipeline:
    a proposal written as ``Y155F`` against a numbering that is off by the
    length of a His-tag mutates some other residue, the variant expresses and
    folds, the assay comes back flat, and the conclusion drawn is "that
    hypothesis was wrong" rather than "we built the wrong protein". Nothing
    downstream can detect it, so it has to be impossible upstream.

    Returns the author position (``None`` when the residue is real but
    unobserved in the structure). Raises :class:`NumberingError` when the
    index is out of range, when ``expected_letter`` is not a standard residue
    letter, or when the candidate sequence disagrees.

    ``require_observed=True`` additionally rejects a position with no
    coordinates, and one whose assignment is ambiguous. Use it for any
    proposal justified by structural evidence: a pocket-contact argument
    about a residue nobody has seen is not evidence, and neither is one about
    a residue the alignment could equally have placed elsewhere.

    A disagreement between the candidate and the *structure* at this position
    is reported as part of the error message when it exists, but it does not
    by itself fail the check -- the candidate sequence is what gets
    synthesised, so the candidate sequence is the authority here.
    """
    if not isinstance(expected_letter, str) or len(expected_letter.strip()) != 1:
        raise NumberingError(
            f"expected_letter must be a single residue letter, got "
            f"{expected_letter!r}"
        )
    want = expected_letter.strip().upper()
    if want not in STANDARD_ONE_LETTER:
        raise NumberingError(
            f"expected_letter {want!r} is not one of the twenty standard "
            f"residues; a mutation cannot be defined against it"
        )

    residue_map._check_index(index)
    actual = residue_map.candidate_sequence[index]
    if actual != want:
        pos = residue_map.index_to_author.get(index)
        where = f" (author position {pos})" if pos else " (not observed in the structure)"
        raise NumberingError(
            f"wild-type mismatch at candidate index {index}{where}: the "
            f"sequence has {actual}, the caller expected {want}. The numbering "
            f"is wrong -- most often a 0-based index was passed where a 1-based "
            f"author number was meant, or a tag shifted the construct. Refusing "
            f"to build a mutation on it."
        )

    pos = residue_map.index_to_author.get(index)
    if require_observed and pos is None:
        raise NumberingError(
            f"candidate index {index} ({actual}) has no coordinates in chain "
            f"{residue_map.chain_id}; a structure-based proposal cannot be made "
            f"for an unobserved residue"
        )
    if require_observed and residue_map.is_ambiguous(index):
        # The index has an author position, but which residue of the run that
        # position refers to was decided by an arbitrary tie-break. Saying
        # "observed" here would hand structural evidence from a residue that
        # has coordinates to one that does not, and the proposal would then
        # name one residue and change another.
        raise NumberingError(
            f"candidate index {index} ({actual}) cannot be confirmed as "
            f"observed: {residue_map.ambiguity_note(index)}. A curator must "
            f"resolve which residue of the run the structure shows before a "
            f"structure-based proposal is made here."
        )
    return pos
