"""Reading a cofactor-recognition determinant off a sequence, motif-anchored.

Why this module exists
---------------------
The family templates already describe the determinant that settles which
nicotinamide an SDR prefers: ``configs/templates/family/sdr.yaml`` records that
a basic residue in or just after the glycine-rich loop pairs with the
2'-phosphate of NADP(H), and that an acidic residue at the C-terminal end of
the second beta-strand is evidence for NAD(H). Nothing could read it. The one
route to a per-sequence cofactor call in
:mod:`eagent.tools.annotate_family` is :class:`CofactorRecognitionRule`, whose
positions are numbers in a *family reference numbering scheme* -- and no scheme
ships, because a reference sequence must be supplied by a curator with its
source (see ``science/family_numbering.py``). So the signal the templates spell
out was unreachable, and ``_cofactor_call`` fell back to the family-level
preference, labelled "not verified on this sequence".

A determinant that is defined *relative to a motif the sequence itself carries*
does not need a global numbering. The glycine-rich loop is locatable by the
template's own motif pattern; the determinant sits a fixed, short distance from
it. That is what this module reads.

What it refuses to do
---------------------
The cost of a wrong cofactor call is an enzyme assayed with the wrong
nicotinamide and recorded as inactive -- a false negative that removes a
candidate before any experiment can see it. So every ambiguity is an outcome,
not a guess:

* the anchor motif must occur **exactly once**. Several occurrences and nothing
  here chooses between them (:attr:`DeterminantOutcome.ANCHOR_NOT_UNIQUE`);
* the window must lie **entirely** inside the sequence, or a truncated window
  would under-report the determinant (:attr:`DeterminantOutcome.WINDOW_TRUNCATED`);
* a rule is read from a template, never from this module. There are no
  patterns, windows or residue sets written here, and
  :class:`AnchoredDeterminantRule` refuses to be built without the evidence
  string its template carries.

And the asymmetry the template states is kept: the presence of the determinant
is evidence for that cofactor, its absence is not evidence for the other one.
:attr:`DeterminantOutcome.ABSENT` means "this determinant was looked for in a
locatable window and is not there", which is a reason to look harder.
"""

from __future__ import annotations

import enum

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..errors import EAgentError

__all__ = [
    "DeterminantError",
    "DeterminantOutcome",
    "AnchoredDeterminantRule",
    "DeterminantCall",
    "read_determinant",
    "STANDARD_AA",
]

#: The twenty standard one-letter codes. A rule naming anything else is a typo
#: or a modified residue, and either way it would silently never match.
STANDARD_AA: str = "ACDEFGHIKLMNPQRSTVWY"


class DeterminantError(EAgentError):
    """A determinant rule is unusable as written."""


class DeterminantOutcome(str, enum.Enum):
    """What reading the determinant established. Five outcomes, not two.

    ``PRESENT`` and ``ABSENT`` are statements about the sequence. The other
    three are statements about this reader's reach, and are kept apart from
    ``ABSENT`` because folding "could not look" into "looked and found
    nothing" is how a modelling limitation ends up read as chemistry.
    """

    PRESENT = "present"
    ABSENT = "absent"
    ANCHOR_ABSENT = "anchor_absent"
    ANCHOR_NOT_UNIQUE = "anchor_not_unique"
    WINDOW_TRUNCATED = "window_truncated"

    @property
    def is_determined(self) -> bool:
        """Whether the sequence itself answered. False for all three refusals."""
        return self in (DeterminantOutcome.PRESENT, DeterminantOutcome.ABSENT)


@dataclass(frozen=True)
class AnchoredDeterminantRule:
    """One determinant: a cofactor, an anchor motif, a window, and residues.

    ``window`` is a pair of inclusive offsets from the **first residue of the
    anchor match**, so ``(0, 11)`` over a seven-residue anchor means the anchor
    itself plus the five residues after it. Offsets may be negative, for a
    determinant that sits upstream of its anchor.

    Every field but ``description`` comes from the template. ``evidence`` and
    ``source_template_id`` are mandatory and non-empty: an unsourced
    recognition rule is the remembered-looking fact this pipeline exists to
    refuse, and ``needs_curation`` travels with the call so a reader is never
    told a textbook-level window was fitted to an alignment.
    """

    cofactor: str
    anchor_motif: str
    anchor_pattern: str
    window: tuple[int, int]
    accepted_residues: frozenset[str]
    evidence: str
    source_template_id: str
    needs_curation: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if not self.cofactor.strip():
            raise DeterminantError("determinant rule names no cofactor")
        if not self.anchor_motif.strip() or not self.anchor_pattern.strip():
            raise DeterminantError(
                f"determinant rule for {self.cofactor} names no anchor motif")
        if not self.evidence.strip():
            raise DeterminantError(
                f"determinant rule for {self.cofactor} carries no evidence "
                f"string and may not be used")
        if not self.source_template_id.strip():
            raise DeterminantError(
                f"determinant rule for {self.cofactor} does not say which "
                f"template it came from")
        first, last = self.window
        if last < first:
            raise DeterminantError(
                f"determinant rule for {self.cofactor} has window "
                f"({first}, {last}), which is empty")
        letters = {str(r).strip().upper() for r in self.accepted_residues}
        bad = sorted(letters - set(STANDARD_AA))
        if not letters or bad:
            raise DeterminantError(
                f"determinant rule for {self.cofactor}: accepted_residues must "
                f"be standard one-letter codes, got {bad or 'nothing'}")
        object.__setattr__(self, "accepted_residues", frozenset(letters))

    @property
    def window_length(self) -> int:
        return self.window[1] - self.window[0] + 1


@dataclass(frozen=True)
class DeterminantCall:
    """The outcome, where it was read, and what the rule it used can support."""

    outcome: DeterminantOutcome
    cofactor: str
    rule_source: str
    evidence: str
    needs_curation: bool
    anchor_start: int | None = None          # 1-based, sequence numbering
    anchor_matched: str | None = None
    window_start: int | None = None          # 1-based, inclusive
    window_end: int | None = None            # 1-based, inclusive
    window_residues: str | None = None
    hits: tuple[tuple[int, str], ...] = ()   # (1-based position, residue)
    note: str = ""

    @property
    def is_determined(self) -> bool:
        return self.outcome.is_determined

    @property
    def supports_cofactor(self) -> str | None:
        """The cofactor this call is evidence for, or ``None``.

        ``ABSENT`` returns ``None`` rather than the other cofactor: the
        templates state the asymmetry explicitly, and the absence of a
        2'-phosphate-binding basic residue is not proof of an NAD(H)
        preference.
        """
        return self.cofactor if self.outcome is DeterminantOutcome.PRESENT else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "cofactor": self.cofactor,
            "supports_cofactor": self.supports_cofactor,
            "determined": self.is_determined,
            "rule_source": self.rule_source,
            "evidence": self.evidence,
            "needs_curation": self.needs_curation,
            "anchor_start": self.anchor_start,
            "anchor_matched": self.anchor_matched,
            "window": None if self.window_start is None
            else [self.window_start, self.window_end],
            "window_residues": self.window_residues,
            "hits": [[p, r] for p, r in self.hits],
            "note": self.note,
        }


def read_determinant(sequence: str, rule: AnchoredDeterminantRule, *,
                     compile_pattern: Any) -> DeterminantCall:
    """Read one determinant off ``sequence``.

    ``compile_pattern`` turns the rule's pattern into a compiled expression.
    It is injected rather than imported so that this module owns no motif
    syntax: the caller passes the same compiler the family templates are
    linted with, and one syntax is in force everywhere.
    """
    seq = (sequence or "").strip().upper()
    base = dict(cofactor=rule.cofactor, rule_source=rule.source_template_id,
                evidence=rule.evidence, needs_curation=rule.needs_curation)
    if not seq:
        return DeterminantCall(outcome=DeterminantOutcome.ANCHOR_ABSENT,
                               note="empty sequence", **base)

    rx = compile_pattern(rule.anchor_pattern)
    matches = list(rx.finditer(seq))
    if not matches:
        return DeterminantCall(
            outcome=DeterminantOutcome.ANCHOR_ABSENT,
            note=(f"anchor motif {rule.anchor_motif} ({rule.anchor_pattern}) "
                  f"does not match this sequence"),
            **base)
    if len(matches) > 1:
        return DeterminantCall(
            outcome=DeterminantOutcome.ANCHOR_NOT_UNIQUE,
            note=(f"anchor motif {rule.anchor_motif} matches "
                  f"{len(matches)} times (at "
                  f"{', '.join(str(m.start() + 1) for m in matches[:5])}"
                  f"{', ...' if len(matches) > 5 else ''}); nothing here "
                  f"chooses between them"),
            **base)

    match = matches[0]
    first = match.start() + rule.window[0]
    last = match.start() + rule.window[1]
    anchor = dict(anchor_start=match.start() + 1, anchor_matched=match.group(0))
    if first < 0 or last >= len(seq):
        return DeterminantCall(
            outcome=DeterminantOutcome.WINDOW_TRUNCATED,
            note=(f"window offsets {rule.window[0]}..{rule.window[1]} from the "
                  f"anchor at {match.start() + 1} fall outside a sequence of "
                  f"{len(seq)} residues"),
            **base, **anchor)

    residues = seq[first:last + 1]
    hits = tuple((first + i + 1, ch) for i, ch in enumerate(residues)
                 if ch in rule.accepted_residues)
    return DeterminantCall(
        outcome=(DeterminantOutcome.PRESENT if hits
                 else DeterminantOutcome.ABSENT),
        window_start=first + 1, window_end=last + 1, window_residues=residues,
        hits=hits,
        note=("" if hits else
              f"no {'/'.join(sorted(rule.accepted_residues))} in the window"),
        **base, **anchor)


def rules_from_template(template: Any) -> list[AnchoredDeterminantRule]:
    """Build the anchored rules a family template declares, or none.

    The anchor is named, not spelled: a rule says which of the template's own
    ``conserved_motifs`` it hangs off, and the pattern is taken from that
    motif. A rule naming a motif the template does not define raises, because
    the alternative is a rule that silently never fires.
    """
    motifs: dict[str, Mapping[str, Any]] = {
        str(m.get("name") or ""): m
        for m in getattr(template, "conserved_motifs", []) or []}
    out: list[AnchoredDeterminantRule] = []
    declared: Sequence[Mapping[str, Any]] = \
        getattr(template, "cofactor_recognition_rules", []) or []
    template_id = str(getattr(template, "template_id", "") or "")
    for raw in declared:
        name = str(raw.get("anchor_motif") or "")
        motif = motifs.get(name)
        if motif is None:
            raise DeterminantError(
                f"{template_id}: cofactor recognition rule anchors on motif "
                f"{name!r}, which this template does not define "
                f"(it defines {sorted(motifs) or 'none'})")
        window = raw.get("window") or []
        if len(window) != 2:
            raise DeterminantError(
                f"{template_id}: cofactor recognition rule for "
                f"{raw.get('cofactor')!r} needs window [first, last] offsets, "
                f"got {window!r}")
        out.append(AnchoredDeterminantRule(
            cofactor=str(raw.get("cofactor") or ""),
            anchor_motif=name,
            anchor_pattern=str(motif.get("pattern") or ""),
            window=(int(window[0]), int(window[1])),
            accepted_residues=frozenset(raw.get("accepted_residues") or ()),
            evidence=str(raw.get("evidence") or ""),
            source_template_id=template_id,
            needs_curation=bool(raw.get("needs_curation", True)),
            description=str(raw.get("description") or ""),
        ))
    return out
