"""Substrate-pocket residues, carried with the frame they are numbered in.

THE FAILURE THIS MODULE EXISTS TO STOP
======================================
Two ketoreductases with the same pocket, one deposited with an extra
N-terminal methionine. Every author number in the second is one higher. Asked
how different their pockets are, the diversity layer compared ``F98`` with
``F99``, found no token in common, and returned a Jaccard distance of **1.0**:
maximally different, for two proteins that would behave identically toward the
substrate.

That number does not sit still. A batch composed to maximise pocket diversity
will spend two slots on the pair, and the round reports pocket coverage it did
not achieve -- so the error consumes synthesis slots and overstates the result
at the same time.

Author numbering is set by whoever deposited the structure. It shifts with
construct boundaries, tags and cleaved signal peptides, and it carries no
cross-protein meaning. Comparing positions therefore requires a **frame**: a
numbering both sides have been mapped into. This module produces pocket
residues with their frame attached, and :mod:`eagent.science.diversity`
refuses to compare two sets that are not in the same one.

WHAT A FRAME COSTS
==================
Getting into a frame means aligning to a family reference, which can fail --
weak identity, a residue aligning to a gap. Those residues come back in
:attr:`PocketResidues.unmapped` rather than being dropped, because a pocket
compared on four of its seven residues is a different comparison from one
compared on all seven, and only the count makes that visible.

With no scheme supplied there is no frame, and the honest fallback is to
compare the *composition* of the two pockets -- which residue types line it,
and how many of each -- and to say that is what was compared. A composition
distance understates differences between pockets that differ only in
arrangement. Reporting it as a positional comparison would overstate
everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..errors import EAgentError
from .family_numbering import FamilyNumberingScheme
from .geometry import GeometryError, pocket_shell
from .numbering import ResidueMap, build_map, residue_one_letter
from .structure_io import Atom, Residue, Structure

__all__ = [
    "PocketError",
    "OWN_NUMBERING",
    "PocketResidues",
    "pocket_residues_from_shell",
    "pocket_residues_from_tokens",
    "pocket_residues_for_pose",
    "best_chain_map",
]


class PocketError(EAgentError):
    """A pocket could not be placed in the frame it was asked for."""


#: Frame label meaning "this candidate's own numbering, shared with nothing".
#: Named rather than left as an empty string so that a signature built on it
#: says so, and so that two candidates in their own numbering are never
#: mistaken for two candidates in one frame.
OWN_NUMBERING: str = "own_numbering"


@dataclass(frozen=True)
class PocketResidues:
    """One candidate's pocket, the frame it is numbered in, and what was lost.

    ``tokens`` are ``<letter><number>`` in :attr:`frame`. Two
    :class:`PocketResidues` are positionally comparable exactly when their
    frames are equal and neither is :data:`OWN_NUMBERING`.
    """

    candidate_id: str
    tokens: tuple[str, ...]
    frame: str = OWN_NUMBERING
    #: Residues that could not be placed in the frame, with why. They are not
    #: in ``tokens``: a residue whose position is unknown cannot be compared
    #: positionally, and guessing it is the bug this module is about.
    unmapped: tuple[str, ...] = ()
    #: One line per unmapped residue, for the record.
    notes: tuple[str, ...] = ()
    #: The alignment identity to the frame's reference, when there was one.
    frame_identity: float | None = None

    @property
    def letters(self) -> tuple[str, ...]:
        """Just the residue types, in sorted order. The composition fallback."""
        return tuple(sorted(t[0] for t in self.tokens if t))

    @property
    def positional(self) -> bool:
        return bool(self.tokens) and self.frame != OWN_NUMBERING

    @property
    def n_lost(self) -> int:
        return len(self.unmapped)

    def comparable_with(self, other: "PocketResidues") -> bool:
        return (self.positional and other.positional
                and self.frame == other.frame)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "frame": self.frame,
            "tokens": list(self.tokens),
            "unmapped": list(self.unmapped),
            "frame_identity": self.frame_identity,
            "notes": list(self.notes),
        }


def _frame_label(scheme: FamilyNumberingScheme) -> str:
    return f"{scheme.family_name}/{scheme.scheme_id}"


def pocket_residues_from_tokens(
    candidate_id: str, tokens: Sequence[str], *, frame: str = OWN_NUMBERING,
) -> PocketResidues:
    """Wrap tokens a caller already has, with the frame stated.

    The default frame is :data:`OWN_NUMBERING`, which is the truth for a bare
    list of ``F98``-style strings: nothing says they share a numbering with
    anything else. A caller that has genuinely put them in a common frame says
    so by naming it.
    """
    return PocketResidues(candidate_id=candidate_id,
                          tokens=tuple(sorted(set(tokens))), frame=frame)


def pocket_residues_from_shell(
    candidate_id: str,
    sequence: str,
    residue_map: ResidueMap,
    shell: Iterable[Residue],
    *,
    scheme: FamilyNumberingScheme | None = None,
    skip_ambiguous: bool = True,
) -> PocketResidues:
    """Turn a geometric shell into pocket residues in a stated frame.

    ``shell`` is what :func:`eagent.science.geometry.pocket_shell` returned:
    residues of the *structure*, in author numbering. Three steps, each of
    which can fail honestly:

    1. **author number -> sequence index**, through ``residue_map``. A residue
       the map cannot place -- a tag, a fusion partner, the wrong chain -- is
       recorded as unmapped. So is one whose index the map marks ambiguous: a
       partly observed run of identical residues admits several equal-scoring
       alignments, and the one the aligner picked is not evidence.
    2. **index -> family position**, through ``scheme``. Below the scheme's
       identity floor, or aligned to a gap in the reference, the residue is
       unmapped rather than given a number.
    3. **token**. With a scheme the token is the family position, so two
       candidates are comparable; without one it is the author position and
       the frame says it belongs to this candidate alone.

    Nothing here is a claim that a residue matters. A shell is a search scope
    (see :func:`eagent.science.geometry.pocket_shell`), and this function only
    moves it into a frame where two scopes can be compared.
    """
    tokens: list[str] = []
    unmapped: list[str] = []
    notes: list[str] = []
    identity: float | None = None
    frame = _frame_label(scheme) if scheme is not None else OWN_NUMBERING

    for residue in shell:
        letter = residue_one_letter(residue.resname)
        author_token = f"{letter or 'X'}{residue.resseq}{residue.icode.strip()}"
        if letter is None:
            unmapped.append(author_token)
            notes.append(f"{residue.resname}{residue.resseq} is not a standard "
                         f"residue and has no place in a sequence frame")
            continue
        index = residue_map.to_index((residue.chain, residue.resseq,
                                      residue.icode))
        if index is None:
            unmapped.append(author_token)
            notes.append(f"{author_token} is in the structure but not in the "
                         f"candidate sequence; it cannot be given a position")
            continue
        if skip_ambiguous and residue_map.is_ambiguous(index):
            unmapped.append(author_token)
            notes.append(f"{author_token}: "
                         + (residue_map.ambiguity_note(index)
                            or "the author assignment is not unique"))
            continue
        if scheme is None:
            tokens.append(author_token)
            continue
        mapping = scheme.to_standard(sequence, index)
        identity = (mapping.identity_to_reference
                    if mapping.identity_to_reference is not None else identity)
        if mapping.standard is None:
            unmapped.append(author_token)
            notes.append(f"{author_token}: {mapping.reason}")
            continue
        tokens.append(f"{letter}{mapping.standard.number}")

    return PocketResidues(
        candidate_id=candidate_id, tokens=tuple(sorted(set(tokens))),
        frame=frame, unmapped=tuple(sorted(set(unmapped))),
        notes=tuple(notes), frame_identity=identity)


def best_chain_map(sequence: str, structure: Structure) -> tuple[ResidueMap | None, str]:
    """The chain of ``structure`` that is this candidate's, and its map.

    A pose file routinely holds more than one chain: a biological dimer, a
    crystallographic partner, a tag. Taking the first one and aligning the
    candidate to it produces a map that is sometimes right and gives no sign
    when it is not, so every protein chain is mapped and the one that matches
    the candidate best is used -- highest sequence identity, and among equals
    the one that covers more of the candidate.

    Returns ``(map, note)``. The map is ``None`` when the structure has no
    protein chain to align to, and the note always says what was chosen or why
    nothing was.
    """
    best: ResidueMap | None = None
    scores: list[str] = []
    for chain in structure.chains:
        if not any(residue_one_letter(r.resname) for r in chain.residues
                   if not r.is_hetatm):
            continue
        candidate_map = build_map(sequence, chain)
        identity = candidate_map.structure_identity or 0.0
        coverage = candidate_map.coverage
        scores.append(f"{chain.chain_id}: identity {identity:.2f}, "
                      f"coverage {coverage:.2f}")
        if best is None:
            best = candidate_map
            continue
        incumbent = ((best.structure_identity or 0.0), best.coverage)
        if (identity, coverage) > incumbent:
            best = candidate_map
    if best is None:
        return None, ("the structure holds no protein chain to align the "
                      "candidate sequence to")
    return best, (f"chain {best.chain_id} chosen from [{'; '.join(scores)}]"
                  if len(scores) > 1 else f"chain {best.chain_id}")


def pocket_residues_for_pose(
    candidate_id: str,
    sequence: str,
    structure: Structure,
    ligand_atoms: Sequence[Atom],
    *,
    max_angstrom: float,
    min_angstrom: float = 0.0,
    scheme: FamilyNumberingScheme | None = None,
) -> tuple[PocketResidues | None, str]:
    """Extract this pose's pocket and put it in a frame, in one call.

    The shell is a search scope, not a claim about which residues matter; see
    :func:`eagent.science.geometry.pocket_shell`. What this adds is that the
    scope comes back in a stated numbering frame, so two candidates' pockets
    can be compared without the comparison depending on who deposited which
    structure.

    Returns ``(residues, note)``, with ``None`` and a reason when the pocket
    could not be measured at all. A measurement problem is reported, never
    turned into an empty pocket -- an empty pocket and an unmeasured one lead
    to opposite conclusions about how different two candidates are.
    """
    if not ligand_atoms:
        return None, ("no ligand atoms were given, so there is no pocket to "
                      "measure around")
    residue_map, chain_note = best_chain_map(sequence, structure)
    if residue_map is None:
        return None, chain_note
    try:
        shell = pocket_shell(structure, ligand_atoms, min_angstrom, max_angstrom)
    except GeometryError as exc:
        return None, f"the pocket shell could not be measured: {exc}"
    residues = pocket_residues_from_shell(
        candidate_id, sequence, residue_map, shell, scheme=scheme)
    return residues, (
        f"{len(residues.tokens)} pocket residue(s) within "
        f"[{min_angstrom:g}, {max_angstrom:g}] A of the ligand, {chain_note}, "
        f"in frame {residues.frame}"
        + (f"; {residues.n_lost} could not be placed in it"
           if residues.n_lost else ""))
