"""Standardised family numbering and cross-subfamily position equivalence.

This module exists to answer one question safely:

    Position 155 in my parent enzyme -- is it the same position as the
    selectivity-determining residue reported at 152 in a related enzyme?

The wrong way to answer is to compare the numbers, or to assume that because
two proteins are both "ketoreductases" their residue 155 means the same thing.
Author numbering is set by whoever deposited the structure, shifts with
construct boundaries and tags, and carries no cross-protein meaning at all. A
mutation designed against a mis-mapped position is not a weak hypothesis, it is
a different experiment from the one intended, and it consumes a synthesis slot
either way.

So every answer here goes through an explicit alignment to a family reference
whose own provenance is recorded, is validated against the family's conserved
anchors, and comes back as a result object carrying confidence and a refusal
reason rather than as a bare integer. Three refusals are hard:

* **Across families.** A short-chain dehydrogenase/reductase position has no
  counterpart in an aldo-keto reductase. The folds differ, the catalytic
  residues differ, and the alignment that would produce a number is spurious.
  Asking for one raises.
* **Below the identity floor.** An alignment too weak to trust returns no
  position, not a low-confidence guess.
* **Into a gap.** A position with no aligned counterpart returns none.

The module ships the machinery, not the reference sequences. A scheme needs a
reference sequence supplied by a curator together with where it came from,
because a fabricated reference would silently shift every position derived
from it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..errors import EAgentError
from ..schemas.candidate import ConfidenceLevel
from .numbering import Alignment, needleman_wunsch

__all__ = [
    "FamilyNumberingError",
    "CrossFamilyComparisonError",
    "UnsourcedReferenceError",
    "AnchorMotif",
    "AnchorCheck",
    "StandardPosition",
    "PositionMapping",
    "EquivalenceResult",
    "FamilyNumberingScheme",
    "naive_number_match",
    "DEFAULT_MIN_IDENTITY",
]


#: Identity below which an alignment is not trusted to carry positions across.
#: A starting value for distant homologues, not a validated constant: each
#: family should calibrate it against pairs whose equivalence is independently
#: known, and the scheme records its own floor so this default is only a
#: fallback.
DEFAULT_MIN_IDENTITY = 0.25


class FamilyNumberingError(EAgentError):
    """A numbering operation could not be performed safely."""


class CrossFamilyComparisonError(FamilyNumberingError):
    """Positions were compared between families that do not share a scheme.

    Raised rather than returning a low-confidence answer, because the caller
    almost always wants a number and would use whatever came back.
    """


class UnsourcedReferenceError(FamilyNumberingError):
    """A numbering scheme was built on a reference with no recorded origin."""


# --------------------------------------------------------------------------
# anchors
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorMotif:
    """A conserved motif used to check that an alignment landed correctly.

    An alignment can score well and still be frame-shifted relative to the
    family's functional landmarks. Anchors catch that: if the family's
    catalytic motif does not appear near where the scheme says it should, the
    alignment is not trustworthy for transferring positions, whatever its
    identity.

    A motif supports or refutes an alignment. It is never on its own evidence
    that a sequence acts on a particular substrate.
    """

    name: str
    pattern: str
    expected_standard_start: int | None = None
    tolerance: int = 15
    role: str = ""
    evidence: str = ""

    def search(self, sequence: str) -> list[int]:
        """Zero-based start indices where the motif matches."""
        try:
            rx = re.compile(self.pattern)
        except re.error as exc:  # pragma: no cover - configuration error
            raise FamilyNumberingError(
                f"anchor '{self.name}' has an invalid pattern: {exc}") from exc
        return [m.start() for m in rx.finditer(sequence.upper())]


@dataclass(frozen=True)
class AnchorCheck:
    """Whether one anchor landed where the scheme expects."""

    name: str
    found: bool
    agreed: bool | None          # None when the scheme states no expectation
    observed_standard: int | None = None
    expected_standard: int | None = None
    detail: str = ""


# --------------------------------------------------------------------------
# positions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StandardPosition:
    """A position in a family's standard numbering.

    Carries the family and the scheme so that a position cannot be moved
    between schemes by accident; two StandardPositions from different schemes
    are not comparable and the type keeps that visible.
    """

    family: str
    scheme_id: str
    number: int
    reference_letter: str
    role: str | None = None

    def __str__(self) -> str:
        role = f" [{self.role}]" if self.role else ""
        return f"{self.family}:{self.reference_letter}{self.number}{role}"

    def same_scheme_as(self, other: "StandardPosition") -> bool:
        return (self.family, self.scheme_id) == (other.family, other.scheme_id)


@dataclass(frozen=True)
class PositionMapping:
    """The result of moving one position between a sequence and a scheme.

    ``standard`` is None whenever the mapping could not be made. The reason is
    always populated in that case, so a caller that logs the result can say why
    a residue was skipped instead of silently dropping it.
    """

    query_index: int | None
    standard: StandardPosition | None
    query_letter: str | None = None
    identity_to_reference: float | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT
    anchors: tuple[AnchorCheck, ...] = ()
    reason: str = ""

    @property
    def mapped(self) -> bool:
        return self.standard is not None

    @property
    def anchors_agree(self) -> bool | None:
        judged = [a.agreed for a in self.anchors if a.agreed is not None]
        if not judged:
            return None
        return all(judged)


@dataclass(frozen=True)
class EquivalenceResult:
    """Whether a position in one sequence corresponds to one in another.

    ``equivalent`` is deliberately tri-state. ``None`` means the question could
    not be answered -- the alignment was too weak, the position fell in a gap,
    or the anchors disagreed -- and must not be read as "not equivalent", which
    would licence the opposite mistake to the one this module prevents.
    """

    equivalent: bool | None
    source_index: int
    target_index: int | None
    standard: StandardPosition | None
    source_letter: str | None = None
    target_letter: str | None = None
    identity_source: float | None = None
    identity_target: float | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.INSUFFICIENT
    anchors: tuple[AnchorCheck, ...] = ()
    reason: str = ""
    evidence: tuple[str, ...] = ()

    @property
    def answered(self) -> bool:
        return self.equivalent is not None

    def describe(self) -> str:
        if self.equivalent is None:
            return f"undetermined: {self.reason}"
        verdict = "equivalent" if self.equivalent else "not equivalent"
        tgt = "-" if self.target_index is None else str(self.target_index + 1)
        return (f"{verdict} (source index {self.source_index + 1} -> target "
                f"index {tgt}, {self.confidence.value}): {self.reason}")


# --------------------------------------------------------------------------
# alignment plumbing
# --------------------------------------------------------------------------

def _column_index_maps(aligned: str) -> tuple[dict[int, int], dict[int, int]]:
    """Map sequence index to alignment column and back, skipping gaps."""
    idx_to_col: dict[int, int] = {}
    col_to_idx: dict[int, int] = {}
    idx = 0
    for col, ch in enumerate(aligned):
        if ch != "-":
            idx_to_col[idx] = col
            col_to_idx[col] = idx
            idx += 1
    return idx_to_col, col_to_idx


@dataclass
class _Aligned:
    """A query aligned to a scheme reference, with the index maps."""

    alignment: Alignment
    query_idx_to_col: dict[int, int]
    col_to_query_idx: dict[int, int]
    ref_idx_to_col: dict[int, int]
    col_to_ref_idx: dict[int, int]

    @property
    def identity(self) -> float:
        return float(self.alignment.identity)


# --------------------------------------------------------------------------
# the scheme
# --------------------------------------------------------------------------

@dataclass
class FamilyNumberingScheme:
    """A family's standard numbering, anchored on a sourced reference sequence.

    The reference is what gives a position number meaning, so its provenance is
    mandatory. ``catalytic_roles`` records where the family's catalytic
    machinery sits in this numbering, which is what lets a caller ask for "the
    catalytic tyrosine" rather than for a number.
    """

    family_name: str
    scheme_id: str
    reference_label: str
    reference_sequence: str
    reference_source: str
    catalytic_roles: Mapping[str, int] = field(default_factory=dict)
    anchors: tuple[AnchorMotif, ...] = ()
    min_identity: float = DEFAULT_MIN_IDENTITY
    first_number: int = 1
    notes: str = ""

    def __post_init__(self) -> None:
        self.reference_sequence = "".join(self.reference_sequence.split()).upper()
        if not self.reference_sequence:
            raise FamilyNumberingError(
                f"{self.scheme_id}: reference sequence is empty")
        if not (self.reference_source or "").strip():
            raise UnsourcedReferenceError(
                f"{self.scheme_id}: the reference sequence has no recorded "
                f"origin. Every position number in this scheme is defined by "
                f"it, so an unsourced reference would silently shift them all."
            )
        bad = set(self.reference_sequence) - set("ACDEFGHIKLMNPQRSTVWYX")
        if bad:
            raise FamilyNumberingError(
                f"{self.scheme_id}: reference contains non-residue characters "
                f"{sorted(bad)}")
        for role, number in self.catalytic_roles.items():
            if self._number_to_ref_index(number) is None:
                raise FamilyNumberingError(
                    f"{self.scheme_id}: catalytic role '{role}' is at "
                    f"{number}, outside the reference sequence")

    # -- numbering arithmetic -------------------------------------------
    def _number_to_ref_index(self, number: int) -> int | None:
        idx = number - self.first_number
        if 0 <= idx < len(self.reference_sequence):
            return idx
        return None

    def _ref_index_to_number(self, index: int) -> int:
        return index + self.first_number

    def reference_letter(self, number: int) -> str | None:
        idx = self._number_to_ref_index(number)
        return None if idx is None else self.reference_sequence[idx]

    def role_position(self, role: str) -> StandardPosition | None:
        """The standard position of a named catalytic role, if the scheme has one."""
        number = self.catalytic_roles.get(role)
        if number is None:
            return None
        letter = self.reference_letter(number)
        if letter is None:  # pragma: no cover - guarded in __post_init__
            return None
        return StandardPosition(self.family_name, self.scheme_id, number, letter, role)

    def role_at(self, number: int) -> str | None:
        for role, n in self.catalytic_roles.items():
            if n == number:
                return role
        return None

    # -- alignment --------------------------------------------------------
    def _align(self, sequence: str) -> _Aligned:
        seq = "".join(sequence.split()).upper()
        if not seq:
            raise FamilyNumberingError("empty query sequence")
        aln = needleman_wunsch(seq, self.reference_sequence)
        q_i2c, q_c2i = _column_index_maps(aln.aligned_a)
        r_i2c, r_c2i = _column_index_maps(aln.aligned_b)
        return _Aligned(aln, q_i2c, q_c2i, r_i2c, r_c2i)

    # -- anchors ----------------------------------------------------------
    def check_anchors(self, sequence: str, aligned: _Aligned | None = None
                      ) -> tuple[AnchorCheck, ...]:
        """Validate that the family's landmarks fall where the scheme expects.

        A well-scoring but frame-shifted alignment is the failure this catches;
        identity alone would not reveal it.
        """
        seq = "".join(sequence.split()).upper()
        al = aligned or self._align(seq)
        checks: list[AnchorCheck] = []
        for anchor in self.anchors:
            hits = anchor.search(seq)
            if not hits:
                checks.append(AnchorCheck(
                    anchor.name, found=False, agreed=None,
                    expected_standard=anchor.expected_standard_start,
                    detail="motif not present in the query sequence"))
                continue
            if anchor.expected_standard_start is None:
                checks.append(AnchorCheck(
                    anchor.name, found=True, agreed=None,
                    detail="scheme states no expected position for this anchor"))
                continue
            best: tuple[int, int] | None = None   # (deviation, observed)
            for hit in hits:
                col = al.query_idx_to_col.get(hit)
                if col is None:
                    continue
                ref_idx = al.col_to_ref_idx.get(col)
                if ref_idx is None:
                    continue
                observed = self._ref_index_to_number(ref_idx)
                dev = abs(observed - anchor.expected_standard_start)
                if best is None or dev < best[0]:
                    best = (dev, observed)
            if best is None:
                checks.append(AnchorCheck(
                    anchor.name, found=True, agreed=False,
                    expected_standard=anchor.expected_standard_start,
                    detail="motif present but aligned entirely to gaps"))
                continue
            dev, observed = best
            checks.append(AnchorCheck(
                anchor.name, found=True, agreed=dev <= anchor.tolerance,
                observed_standard=observed,
                expected_standard=anchor.expected_standard_start,
                detail=f"observed at {observed}, expected near "
                       f"{anchor.expected_standard_start} "
                       f"(deviation {dev}, tolerance {anchor.tolerance})"))
        return tuple(checks)

    # -- confidence -------------------------------------------------------
    def _confidence(self, identity: float, anchors: Sequence[AnchorCheck]
                    ) -> ConfidenceLevel:
        judged = [a.agreed for a in anchors if a.agreed is not None]
        if judged and not all(judged):
            return ConfidenceLevel.CONTRADICTORY
        if identity < self.min_identity:
            return ConfidenceLevel.INSUFFICIENT
        if identity >= 0.60 and judged:
            return ConfidenceLevel.STRONG
        if identity >= 0.40 or judged:
            return ConfidenceLevel.MODERATE
        return ConfidenceLevel.WEAK

    # -- public mapping ---------------------------------------------------
    def to_standard(self, sequence: str, index: int) -> PositionMapping:
        """Map a zero-based index in ``sequence`` onto the family numbering."""
        seq = "".join(sequence.split()).upper()
        if not 0 <= index < len(seq):
            return PositionMapping(
                index, None, reason=f"index {index} outside the query sequence "
                                    f"of length {len(seq)}")
        al = self._align(seq)
        anchors = self.check_anchors(seq, al)
        identity = al.identity
        letter = seq[index]

        if identity < self.min_identity:
            return PositionMapping(
                index, None, letter, identity,
                self._confidence(identity, anchors), anchors,
                reason=f"alignment identity {identity:.2f} is below the "
                       f"scheme floor {self.min_identity:.2f}; positions are "
                       f"not transferred across an alignment this weak")

        col = al.query_idx_to_col.get(index)
        ref_idx = al.col_to_ref_idx.get(col) if col is not None else None
        if ref_idx is None:
            return PositionMapping(
                index, None, letter, identity,
                self._confidence(identity, anchors), anchors,
                reason="this residue aligns to a gap in the reference; it has "
                       "no counterpart in the family numbering")

        number = self._ref_index_to_number(ref_idx)
        std = StandardPosition(self.family_name, self.scheme_id, number,
                               self.reference_sequence[ref_idx],
                               self.role_at(number))
        return PositionMapping(index, std, letter, identity,
                               self._confidence(identity, anchors), anchors,
                               reason="mapped through alignment to the family reference")

    def from_standard(self, sequence: str, number: int) -> PositionMapping:
        """Find the index in ``sequence`` corresponding to a standard position."""
        seq = "".join(sequence.split()).upper()
        ref_idx = self._number_to_ref_index(number)
        if ref_idx is None:
            return PositionMapping(
                None, None, reason=f"standard position {number} is outside "
                                   f"the reference sequence")
        al = self._align(seq)
        anchors = self.check_anchors(seq, al)
        identity = al.identity
        std = StandardPosition(self.family_name, self.scheme_id, number,
                               self.reference_sequence[ref_idx],
                               self.role_at(number))

        if identity < self.min_identity:
            return PositionMapping(
                None, None, None, identity,
                self._confidence(identity, anchors), anchors,
                reason=f"alignment identity {identity:.2f} is below the "
                       f"scheme floor {self.min_identity:.2f}")

        col = al.ref_idx_to_col.get(ref_idx)
        q_idx = al.col_to_query_idx.get(col) if col is not None else None
        if q_idx is None:
            return PositionMapping(
                None, std, None, identity,
                self._confidence(identity, anchors), anchors,
                reason="the query has a deletion at this position; there is no "
                       "residue to mutate")
        return PositionMapping(q_idx, std, seq[q_idx], identity,
                               self._confidence(identity, anchors), anchors,
                               reason="mapped through alignment to the family reference")


# --------------------------------------------------------------------------
# cross-sequence comparison
# --------------------------------------------------------------------------

def equivalent_position(
    scheme: FamilyNumberingScheme,
    source_sequence: str,
    source_index: int,
    target_sequence: str,
    *,
    target_scheme: FamilyNumberingScheme | None = None,
    expect_target_letter: str | None = None,
) -> EquivalenceResult:
    """Decide whether a position in one sequence corresponds to one in another.

    Both sequences are mapped onto the same family numbering and compared
    there, rather than being aligned to each other directly, so the answer is
    stated in terms a second enzyme's literature can also be expressed in.

    Supplying ``target_scheme`` from a different family raises, because the
    answer would be meaningless: a short-chain dehydrogenase/reductase position
    has no aldo-keto reductase counterpart, and returning a number anyway is
    exactly how a mutation gets designed at the wrong residue.
    """
    if target_scheme is not None and (
        target_scheme.family_name != scheme.family_name
        or target_scheme.scheme_id != scheme.scheme_id
    ):
        raise CrossFamilyComparisonError(
            f"cannot transfer a position from {scheme.family_name} "
            f"({scheme.scheme_id}) to {target_scheme.family_name} "
            f"({target_scheme.scheme_id}). These families do not share a fold, "
            f"a catalytic arrangement, or a numbering, so no alignment between "
            f"them transfers a position. Compare within a family, or treat the "
            f"two as separate mechanistic hypotheses."
        )

    src = scheme.to_standard(source_sequence, source_index)
    evidence: list[str] = [
        f"source aligned to {scheme.reference_label} at identity "
        f"{src.identity_to_reference:.2f}" if src.identity_to_reference is not None
        else "source alignment failed",
    ]
    if not src.mapped:
        return EquivalenceResult(
            None, source_index, None, None, src.query_letter, None,
            src.identity_to_reference, None, src.confidence, src.anchors,
            reason=f"source position could not be placed in the family "
                   f"numbering: {src.reason}",
            evidence=tuple(evidence))

    assert src.standard is not None
    tgt = scheme.from_standard(target_sequence, src.standard.number)
    if tgt.identity_to_reference is not None:
        evidence.append(f"target aligned to {scheme.reference_label} at "
                        f"identity {tgt.identity_to_reference:.2f}")

    confidence = min(src.confidence, tgt.confidence, key=lambda c: c.rank)
    anchors = tuple(src.anchors) + tuple(tgt.anchors)

    if tgt.query_index is None:
        return EquivalenceResult(
            None, source_index, None, src.standard, src.query_letter, None,
            src.identity_to_reference, tgt.identity_to_reference,
            confidence, anchors,
            reason=f"no counterpart in the target: {tgt.reason}",
            evidence=tuple(evidence))

    if confidence in (ConfidenceLevel.INSUFFICIENT, ConfidenceLevel.CONTRADICTORY):
        return EquivalenceResult(
            None, source_index, tgt.query_index, src.standard,
            src.query_letter, tgt.query_letter,
            src.identity_to_reference, tgt.identity_to_reference,
            confidence, anchors,
            reason=("the alignments do not support transferring this position: "
                    + (tgt.reason if confidence is ConfidenceLevel.INSUFFICIENT
                       else "family anchors disagree with the alignment")),
            evidence=tuple(evidence))

    if expect_target_letter is not None and tgt.query_letter != expect_target_letter.upper():
        return EquivalenceResult(
            False, source_index, tgt.query_index, src.standard,
            src.query_letter, tgt.query_letter,
            src.identity_to_reference, tgt.identity_to_reference,
            confidence, anchors,
            reason=(f"the aligned target residue is {tgt.query_letter}, not the "
                    f"expected {expect_target_letter.upper()}; the reported "
                    f"position and this alignment disagree, so one of them is "
                    f"about a different residue"),
            evidence=tuple(evidence))

    role = src.standard.role
    evidence.append(f"both map to {src.standard}")
    return EquivalenceResult(
        True, source_index, tgt.query_index, src.standard,
        src.query_letter, tgt.query_letter,
        src.identity_to_reference, tgt.identity_to_reference,
        confidence, anchors,
        reason=(f"both sequences map to standard position "
                f"{src.standard.number}"
                + (f", the family's {role}" if role else "")),
        evidence=tuple(evidence))


def group_by_standard_position(
    scheme: FamilyNumberingScheme,
    sequences: Mapping[str, str],
    indices: Mapping[str, Iterable[int]],
) -> dict[int, list[tuple[str, int, str]]]:
    """Group per-sequence positions by the standard position they share.

    Used to compare reported mutation positions across subfamilies: entries
    landing on the same standard number are talking about the same site, and
    entries that could not be mapped are simply absent rather than being
    assigned to a nearby number.
    """
    grouped: dict[int, list[tuple[str, int, str]]] = {}
    for name, idxs in indices.items():
        seq = sequences.get(name)
        if seq is None:
            continue
        for idx in idxs:
            mapping = scheme.to_standard(seq, idx)
            if not mapping.mapped or mapping.standard is None:
                continue
            if mapping.confidence in (ConfidenceLevel.INSUFFICIENT,
                                      ConfidenceLevel.CONTRADICTORY):
                continue
            grouped.setdefault(mapping.standard.number, []).append(
                (name, idx, mapping.query_letter or "?"))
    return grouped


@dataclass(frozen=True)
class NaiveMatchWarning:
    """The answer you get from comparing position numbers, and why not to use it."""

    source_number: int
    target_number: int
    numbers_equal: bool
    warning: str = (
        "Equal author numbers do not mean equivalent positions. Author "
        "numbering is set per deposition and shifts with construct boundaries, "
        "tags and truncations. Use equivalent_position(), which aligns both "
        "sequences to a sourced family reference and validates the result "
        "against the family's conserved anchors."
    )

    def __bool__(self) -> bool:  # pragma: no cover - deliberately unusable
        raise FamilyNumberingError(
            "a naive number match is not a truth value; call "
            "equivalent_position() instead")


def naive_number_match(source_number: int, target_number: int) -> NaiveMatchWarning:
    """Present, and defuse, the comparison a reader is tempted to make.

    Returning an object that refuses to be used as a boolean is deliberate: the
    tempting one-liner ``if a_pos == b_pos`` is the bug this module exists to
    prevent, so the shortcut is made unusable rather than merely discouraged.
    """
    return NaiveMatchWarning(source_number, target_number,
                             source_number == target_number)
