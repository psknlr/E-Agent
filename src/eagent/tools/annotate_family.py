"""Interface ``annotate_family``: family calls from combined, separable signals.

Why this module is shaped the way it is
---------------------------------------

**A family is never called from one motif.** The Rossmann-like glycine-rich
motif ``GxxxGxG`` appears in NAD(P)-binding proteins across several unrelated
superfamilies; the SDR "YxxxK" catalytic pair appears in proteins that are not
SDRs. Any one of these, taken alone, assigns a family that then silently
licenses a mechanism, a cofactor and a set of catalytic residues. This module
therefore gathers four *separable* signal types -- overall identity to a family
reference, domain architecture, several conserved positions, and catalytic
residue correspondence -- and hands the count to
:meth:`FamilyAnnotation.recompute_confidence` with the family template's own
``min_independent_signals``. A single signal yields ``WEAK``, never a family.

**The three ketoreductase families are separate mechanistic hypotheses.** SDR,
MDR/ADH and AKR all reduce ketones with a nicotinamide cofactor and are
routinely conflated, yet they share no catalytic residues, no fold and no
cofactor-recognition logic:

* **SDR** -- Rossmann fold, catalytic Ser-Tyr-Lys(-Asn), no metal, cofactor
  specificity set by the basic residues of the glycine-rich motif region;
* **MDR/ADH** -- two-domain, catalytic zinc coordinated by Cys/His/Cys or Glu,
  a different Rossmann insertion, catalysis through metal-polarised carbonyl;
* **AKR** -- (beta/alpha)8 barrel, Asp-Tyr-Lys-His tetrad, cofactor bound in an
  extended conformation with no Rossmann motif at all.

Applying one family's catalytic residues or cofactor rule to another is not a
small error: it produces a confident, fully-populated, entirely fictional
mechanism. There is deliberately **no shared default** anywhere in this module.
Every catalytic mapping and every cofactor rule is reachable only through a
:class:`FamilyHypothesis`, which binds one family template, one catalytic
template of the *same* ``family_name``, and one reference sequence whose
residues at the template's catalytic positions are verified on construction.
Rules additionally carry the template id they belong to and are re-checked at
use. A wrong-family application is a construction error, not a runtime
possibility.

**The network is not a tree and not a function.** The sequence similarity
network built here is an exploration aid in the EFI-EST sense. A connected
component at a chosen threshold is a set of sequences linked by pairwise
similarity above that threshold -- it is not a clade, it is not monophyletic,
it does not imply a shared substrate, and it changes shape when the threshold
changes. The artifacts say so, and the alignment hand-off says that
non-homologous families must be aligned and tree-built separately rather than
forced into one MAFFT run and one IQ-TREE.

**Numbering.** Positions produced here are 1-based positions in the *candidate
sequence*, not structure author numbering; there is no structure at this stage.
``CatalyticMapping.role_to_index`` is the 0-based index of the same residue.
The structure step re-derives author numbering through
:class:`~eagent.science.numbering.ResidueMap`, and mixing the two axes is the
off-by-N error that produces a variant at the wrong residue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, Iterator, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..context import RunContext
from ..envelope import Artifact, Provenance, Severity, Status, ToolResult
from ..errors import FabricationGuardError, TemplateError
from ..provenance import sha256_file, sha256_obj
from ..schemas import (
    Candidate, CatalyticMapping, CatalyticTemplate, ConfidenceLevel,
    FamilyAnnotation, FamilyTemplate, SequenceRecord,
)
from ..science.diversity import jaccard_distance, kmer_set
from ..science.numbering import Alignment, needleman_wunsch, three_to_one
from .base import ScientificInterface
from .mine_sequences import (
    STANDARD_AA, FastaEntry, read_fasta, write_fasta, _tsv_cell, _write_tsv,
)

__all__ = [
    "CONSERVATIVE_GROUPS",
    "DEFAULT_NETWORK_IDENTITY_PCT",
    "DEFAULT_NETWORK_COVERAGE",
    "DEFAULT_IDENTITY_MARGIN_PCT",
    "DEFAULT_MIN_MOTIFS_FOR_SIGNAL",
    "DEFAULT_LENGTH_TOLERANCE",
    "DEFAULT_MAX_PAIRWISE_ALIGNMENTS",
    "SIGNAL_IDENTITY",
    "SIGNAL_DOMAIN",
    "SIGNAL_MOTIFS",
    "SIGNAL_CATALYTIC",
    "prosite_to_regex",
    "compile_motif",
    "DomainHit",
    "ReferenceSequence",
    "CofactorRecognitionRule",
    "FamilyHypothesis",
    "HypothesisSet",
    "HypothesisAssessment",
    "map_reference_positions",
    "map_catalytic_roles",
    "match_motifs",
    "AnnotationPolicy",
    "SequenceNetwork",
    "build_similarity_network",
    "AnnotateFamily",
]


# --------------------------------------------------------------------------
# Generic chemical-similarity groups.
#
# Used ONLY to label a catalytic position as "conservatively substituted",
# which is a flag asking a human to look, never a claim that activity is
# retained: Ser->Thr at an SDR catalytic serine is chemically conservative and
# functionally fatal often enough that no automatic conclusion is allowed.
# A catalytic residue entry in a CatalyticTemplate may carry its own
# ``conservative_substitutions`` list, which is family-specific evidence and
# always takes precedence over this generic table.
# --------------------------------------------------------------------------
CONSERVATIVE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset("AG"), frozenset("ST"), frozenset("ILVM"), frozenset("FYW"),
    frozenset("DE"), frozenset("NQ"), frozenset("KR"), frozenset("KRH"),
    frozenset("DN"), frozenset("EQ"),
)

# --------------------------------------------------------------------------
# Exploration defaults. None of these decides anything catalytic.
# Each needs per-family calibration: a family of close paralogues will collapse
# into one network component at any of these settings, and a deep family will
# shatter into singletons.
# --------------------------------------------------------------------------

#: Percent identity for a sequence-similarity-network edge.
DEFAULT_NETWORK_IDENTITY_PCT: float = 40.0
#: Minimum alignment coverage of the shorter sequence for an edge to be drawn.
DEFAULT_NETWORK_COVERAGE: float = 0.80
#: How far ahead the best family's identity must be before identity alone counts
#: as a discriminating signal between competing hypotheses, in percentage points.
DEFAULT_IDENTITY_MARGIN_PCT: float = 5.0
#: Conserved positions needed before the motif signal counts. Two, because one
#: motif is a coincidence waiting to happen.
DEFAULT_MIN_MOTIFS_FOR_SIGNAL: int = 2
#: Relative length deviation from the reference before length is a conflict.
DEFAULT_LENGTH_TOLERANCE: float = 0.30
#: Alignment budget for the network; exceeding it degrades loudly.
DEFAULT_MAX_PAIRWISE_ALIGNMENTS: int = 50_000

#: Signal labels. Kept as constants so the strings in ``signals_supporting``
#: cannot drift apart from the strings a reviewer greps for.
SIGNAL_IDENTITY: str = "overall_identity_to_reference"
SIGNAL_DOMAIN: str = "domain_architecture"
SIGNAL_MOTIFS: str = "conserved_motifs"
SIGNAL_CATALYTIC: str = "catalytic_residue_correspondence"


# ==========================================================================
# Motif patterns
# ==========================================================================

def prosite_to_regex(pattern: str) -> str:
    """Translate a PROSITE-style pattern into a Python regular expression.

    Supported: literal residues, ``[ST]`` (any of), ``{P}`` (none of), ``x``
    (any), ``(n)`` and ``(n,m)`` repetition, ``<`` and ``>`` anchors, and ``-``
    separators. A pattern prefixed ``re:`` is taken as a raw regular expression
    instead.

    Unrecognised syntax raises :class:`~eagent.errors.TemplateError` rather
    than being skipped. A motif that silently fails to compile would never
    match, and "this family's motifs are absent" would then be reported as a
    property of the sequence instead of a typo in the template.
    """
    text = pattern.strip()
    if text.startswith("re:"):
        return text[3:]
    text = text.rstrip(".")
    out: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == "-":
            i += 1
            continue
        if c == "<":
            out.append("^")
            i += 1
            continue
        if c == ">":
            out.append("$")
            i += 1
            continue
        if c in ("x", "X"):
            out.append(".")
            i += 1
        elif c == "[":
            j = text.find("]", i)
            if j < 0:
                raise TemplateError(f"motif pattern {pattern!r}: unclosed '['")
            inner = text[i + 1:j].upper().replace("-", "")
            if not inner:
                raise TemplateError(f"motif pattern {pattern!r}: empty '[]' class")
            out.append(f"[{inner}]")
            i = j + 1
        elif c == "{":
            j = text.find("}", i)
            if j < 0:
                raise TemplateError(f"motif pattern {pattern!r}: unclosed '{{'")
            inner = text[i + 1:j].upper().replace("-", "")
            if not inner:
                raise TemplateError(f"motif pattern {pattern!r}: empty '{{}}' class")
            out.append(f"[^{inner}]")
            i = j + 1
        elif c.isalpha():
            if c.upper() not in STANDARD_AA:
                raise TemplateError(
                    f"motif pattern {pattern!r}: {c!r} is not a standard residue "
                    f"letter; a motif over an ambiguity code cannot be matched"
                )
            out.append(c.upper())
            i += 1
        else:
            raise TemplateError(
                f"motif pattern {pattern!r}: unsupported character {c!r} at "
                f"position {i}"
            )
        if i < len(text) and text[i] == "(":
            j = text.find(")", i)
            if j < 0:
                raise TemplateError(f"motif pattern {pattern!r}: unclosed '('")
            spec = text[i + 1:j].strip()
            if "," in spec:
                lo, _, hi = spec.partition(",")
                out.append("{%s,%s}" % (lo.strip(), hi.strip()))
            else:
                out.append("{%s}" % spec)
            i = j + 1
    return "".join(out)


def compile_motif(pattern: str) -> re.Pattern[str]:
    """Compile a template motif, raising :class:`TemplateError` on bad syntax."""
    expr = prosite_to_regex(pattern)
    try:
        return re.compile(expr)
    except re.error as exc:
        raise TemplateError(
            f"motif pattern {pattern!r} compiled to {expr!r}, which is not a "
            f"valid expression: {exc}"
        ) from exc


def match_motifs(sequence: str,
                 motifs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Find every template motif in a sequence.

    Returns one entry per *matching* motif, with its 1-based start, so a family
    call can be audited position by position. Overlapping occurrences of the
    same motif are reported once: a motif found three times is still one
    conserved position type and must not inflate the signal count.
    """
    found: list[dict[str, Any]] = []
    seq = sequence.upper()
    for motif in motifs:
        pattern = str(motif.get("pattern") or "").strip()
        name = str(motif.get("name") or pattern or "unnamed")
        if not pattern:
            raise TemplateError(f"conserved motif {name!r} has no pattern")
        rx = compile_motif(pattern)
        match = rx.search(seq)
        if match is None:
            continue
        found.append({
            "name": name,
            "pattern": pattern,
            "role": motif.get("role"),
            "evidence": motif.get("evidence"),
            "start": match.start() + 1,          # 1-based, candidate numbering
            "end": match.end(),
            "matched": match.group(0),
        })
    return found


# ==========================================================================
# Inputs
# ==========================================================================

class DomainHit(BaseModel):
    """A domain annotation supplied by an upstream HMM scan.

    This module does not run hmmscan; domain evidence is an *input*. When it is
    absent the domain signal is simply not counted and the gap is recorded as
    an uncertainty -- an absent signal is never replaced by an assumed one.
    """

    model_config = ConfigDict(extra="forbid")

    accession: str                   # Pfam PF..., InterPro IPR..., or a local id
    name: str | None = None
    start: int | None = None
    end: int | None = None
    source: str = "unknown"
    evalue: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Shape expected by ``FamilyAnnotation.domains``."""
        return {"id": self.accession, "name": self.name, "start": self.start,
                "end": self.end, "source": self.source}


class ReferenceSequence(BaseModel):
    """The sequence a family's catalytic template was annotated against.

    ``role_to_number`` maps each catalytic role label to its **1-based**
    position in this reference. The hypothesis constructor checks every one of
    those positions against the residue types the catalytic template allows, so
    a reference numbering that is off by one fails at load time rather than
    producing a plausible mapping onto every candidate in the pool.
    """

    model_config = ConfigDict(extra="forbid")

    accession: str
    sequence: str
    role_to_number: dict[str, int] = Field(default_factory=dict)
    source: str = ""
    organism: str | None = None

    @field_validator("sequence")
    @classmethod
    def _clean(cls, v: str) -> str:
        seq = "".join(str(v).split()).upper()
        if not seq:
            raise ValueError("reference sequence is empty")
        return seq

    @model_validator(mode="after")
    def _sourced(self) -> "ReferenceSequence":
        if not self.source.strip():
            raise TemplateError(
                f"reference {self.accession}: a reference used to place catalytic "
                f"residues must name where it came from"
            )
        return self

    def letter_at(self, number: int) -> str:
        """Residue at a 1-based reference position."""
        if not 1 <= number <= len(self.sequence):
            raise TemplateError(
                f"reference {self.accession}: position {number} is outside the "
                f"sequence (1..{len(self.sequence)}); this is almost always a "
                f"0-based index used as a residue number"
            )
        return self.sequence[number - 1]


class CofactorRecognitionRule(BaseModel):
    """A family-specific rule linking reference positions to a cofactor preference.

    Keyed on ``family_template_id`` and re-checked at use, because the
    cofactor-recognition logic of the three ketoreductase families is not
    transferable: the basic residues that select NADPH in an SDR glycine-rich
    region have no counterpart in an AKR, whose cofactor sits in an entirely
    different pocket. Applying one as a default for the other would output a
    confident cofactor preference with no basis at all.

    ``evidence`` is mandatory and non-empty: an unsourced recognition rule is
    exactly the kind of remembered-looking fact this pipeline refuses.
    """

    model_config = ConfigDict(extra="forbid")

    cofactor: str                                   # e.g. "NADPH", "NADH"
    family_template_id: str
    reference_positions: list[int]
    accepted_residues: list[str]
    evidence: str
    description: str = ""

    @model_validator(mode="after")
    def _usable(self) -> "CofactorRecognitionRule":
        if not self.evidence.strip():
            raise TemplateError(
                f"cofactor rule for {self.cofactor} in {self.family_template_id} "
                f"carries no evidence string and may not be used"
            )
        if not self.reference_positions:
            raise TemplateError(
                f"cofactor rule for {self.cofactor} names no reference positions"
            )
        letters = {str(r).strip().upper() for r in self.accepted_residues}
        bad = letters - set(STANDARD_AA)
        if not letters or bad:
            raise TemplateError(
                f"cofactor rule for {self.cofactor}: accepted_residues must be "
                f"standard one-letter codes, got {sorted(self.accepted_residues)}"
            )
        object.__setattr__(self, "accepted_residues", sorted(letters))
        return self


@dataclass(frozen=True)
class FamilyHypothesis:
    """One mechanistic hypothesis: family + catalytic template + reference.

    This object exists to make a wrong-family application structurally
    impossible. It is the only route to a catalytic mapping or a cofactor call
    in this module, and its constructor refuses:

    * a catalytic template whose ``family_name`` differs from the family
      template's (an AKR tetrad under an SDR family call);
    * a catalytic template the family template does not list, when the family
      template lists any;
    * a reference missing a position for any catalytic role;
    * a reference whose residue at a role position is not one the catalytic
      template allows -- the check that catches an off-by-one reference
      numbering before it is applied to a whole pool;
    * a cofactor rule belonging to a different family template.
    """

    family_template: FamilyTemplate
    catalytic_template: CatalyticTemplate
    reference: ReferenceSequence
    cofactor_rules: tuple[CofactorRecognitionRule, ...] = ()

    def __post_init__(self) -> None:
        fam, cat = self.family_template, self.catalytic_template
        if fam.family_name.strip().upper() != cat.family_name.strip().upper():
            raise FabricationGuardError(
                f"catalytic template {cat.template_id} belongs to family "
                f"'{cat.family_name}' but was bound to family template "
                f"{fam.template_id} ('{fam.family_name}'). SDR, MDR/ADH and AKR "
                f"share neither catalytic residues nor cofactor logic; this "
                f"binding would fabricate a mechanism."
            )
        if fam.catalytic_template_ids and cat.template_id not in fam.catalytic_template_ids:
            raise TemplateError(
                f"family template {fam.template_id} does not list catalytic "
                f"template {cat.template_id} (lists {fam.catalytic_template_ids})"
            )
        missing = [label for label in self.role_labels
                   if label not in self.reference.role_to_number]
        if missing:
            raise TemplateError(
                f"reference {self.reference.accession} has no position for "
                f"catalytic role(s) {missing} of {cat.template_id}; a role with "
                f"no anchor cannot be mapped onto a candidate"
            )
        for spec in cat.catalytic_residues:
            label = _role_label(spec)
            number = self.reference.role_to_number[label]
            actual = self.reference.letter_at(number)
            allowed = _accepted_letters(spec)
            if allowed and actual not in allowed:
                raise TemplateError(
                    f"{cat.template_id}/{label}: reference "
                    f"{self.reference.accession} position {number} is {actual}, "
                    f"not one of {sorted(allowed)}. The reference numbering and "
                    f"the catalytic template disagree -- usually an off-by-one -- "
                    f"and every mapping built from it would be wrong."
                )
        for rule in self.cofactor_rules:
            if rule.family_template_id != fam.template_id:
                raise FabricationGuardError(
                    f"cofactor rule for {rule.cofactor} is keyed to "
                    f"{rule.family_template_id} but was attached to "
                    f"{fam.template_id}"
                )
            for number in rule.reference_positions:
                self.reference.letter_at(number)      # range-checks, raises if not

    # -- identity ---------------------------------------------------------
    @property
    def template_id(self) -> str:
        """The family template id. This is the key everything is stored under."""
        return self.family_template.template_id

    @property
    def family_name(self) -> str:
        """Family name, taken from the family template and never from the data."""
        return self.family_template.family_name

    @property
    def role_labels(self) -> tuple[str, ...]:
        """Catalytic role labels this hypothesis must be able to place."""
        return tuple(_role_label(s) for s in self.catalytic_template.catalytic_residues)

    @property
    def min_independent_signals(self) -> int:
        """Signals this family demands, read from its own template.

        Read from the template rather than fixed here, because how many
        independent signals a family call needs is a property of the family's
        diagnosability, not of this code.
        """
        return self.family_template.min_independent_signals


class HypothesisSet:
    """The competing family hypotheses for one run, keyed by family template id.

    Competing, not ranked: SDR, MDR/ADH and AKR are evaluated independently for
    every sequence and the winner has to out-signal the others. Two hypotheses
    sharing a catalytic template id are rejected, because that would be the one
    way a catalytic mapping could be reached from two different families.
    """

    def __init__(self, hypotheses: Iterable[FamilyHypothesis]) -> None:
        self._by_id: dict[str, FamilyHypothesis] = {}
        seen_catalytic: dict[str, str] = {}
        for h in hypotheses:
            if h.template_id in self._by_id:
                raise TemplateError(
                    f"two hypotheses share family template id {h.template_id}"
                )
            cat_id = h.catalytic_template.template_id
            if cat_id in seen_catalytic:
                raise FabricationGuardError(
                    f"catalytic template {cat_id} is bound to both "
                    f"{seen_catalytic[cat_id]} and {h.template_id}; one "
                    f"mechanism cannot belong to two families"
                )
            seen_catalytic[cat_id] = h.template_id
            self._by_id[h.template_id] = h

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[FamilyHypothesis]:
        return iter(self._by_id.values())

    def __contains__(self, template_id: object) -> bool:
        return template_id in self._by_id

    def get(self, template_id: str) -> FamilyHypothesis:
        """Fetch by family template id, raising when it is not loaded."""
        if template_id not in self._by_id:
            raise TemplateError(
                f"family template {template_id} is not among the loaded "
                f"hypotheses ({sorted(self._by_id)})"
            )
        return self._by_id[template_id]

    def ids(self) -> list[str]:
        """Loaded family template ids, sorted so provenance is reproducible."""
        return sorted(self._by_id)


def _role_label(spec: Mapping[str, Any]) -> str:
    """Label of a catalytic residue entry, required for keying the mapping."""
    label = str(spec.get("label") or spec.get("role") or "").strip()
    if not label:
        raise TemplateError(
            "a catalytic residue entry has neither 'label' nor 'role'; roles are "
            "the keys of every mapping and cannot be positional"
        )
    return label


def _accepted_letters(spec: Mapping[str, Any]) -> frozenset[str]:
    """One-letter residue types a catalytic role accepts, from the template.

    Accepts one- or three-letter codes so a template written either way loads.
    An empty set means the template did not constrain the residue type, which
    is recorded rather than silently treated as "anything matches".
    """
    raw = spec.get("residue_types") or spec.get("residues") or []
    if isinstance(raw, str):
        raw = [raw]
    letters: set[str] = set()
    for item in raw:
        token = str(item).strip().upper()
        if len(token) == 1:
            if token not in STANDARD_AA:
                raise TemplateError(
                    f"catalytic residue type {token!r} is not one of the twenty "
                    f"standard residues; an ambiguity code accepts everything"
                )
            letters.add(token)
        elif len(token) == 3:
            # strict=False returns "X" for an unknown name; "X" is not a
            # residue type and must not quietly widen what a role accepts.
            one = three_to_one(token, strict=False)
            if one not in STANDARD_AA:
                raise TemplateError(
                    f"catalytic residue type {token!r} is not a standard residue"
                )
            letters.add(one)
        elif token:
            raise TemplateError(
                f"catalytic residue type {token!r} is neither a one- nor a "
                f"three-letter code"
            )
    return frozenset(letters)


def _conservative(actual: str, allowed: Iterable[str],
                  spec: Mapping[str, Any]) -> bool:
    """Whether ``actual`` is a conservative stand-in for an allowed residue.

    Template-declared substitutions win; the generic chemical groups are the
    fallback and are only ever used to raise a flag for review.
    """
    declared = spec.get("conservative_substitutions")
    if declared:
        return actual in {str(x).strip().upper() for x in declared}
    return any(actual in group and any(a in group for a in allowed)
               for group in CONSERVATIVE_GROUPS)


# ==========================================================================
# Alignment-based mapping
# ==========================================================================

def map_reference_positions(candidate_sequence: str,
                            reference_sequence: str) -> tuple[dict[int, int], Alignment]:
    """Map 1-based reference positions to 0-based candidate indices.

    Built from a global alignment rather than an offset, because the two
    proteins differ by indels and an offset is right until the first one. Only
    columns where both sides carry a residue produce an entry: a reference
    position aligned to a gap has no counterpart in the candidate, and that
    absence is information the catalytic mapping reports as a missing role.
    """
    aln = needleman_wunsch(candidate_sequence, reference_sequence)
    mapping: dict[int, int] = {}
    cand_index = -1
    ref_number = 0
    for a, b in zip(aln.aligned_a, aln.aligned_b):
        if a != "-":
            cand_index += 1
        if b != "-":
            ref_number += 1
        if a != "-" and b != "-":
            mapping[ref_number] = cand_index
    return mapping, aln


def map_catalytic_roles(sequence: str, hypothesis: FamilyHypothesis,
                        ref_to_index: Mapping[int, int],
                        alignment_quality: float | None = None) -> CatalyticMapping:
    """Place one hypothesis's catalytic roles onto one candidate sequence.

    The mapping is stamped with ``catalytic_template_id`` so a downstream reader
    can never apply it under a different family. Three outcomes per role:
    mapped (residue is one the template allows), conservatively substituted
    (flagged, never counted as present), or missing (no aligned position, or a
    residue the template does not allow).

    ``role_to_residue`` uses **1-based candidate-sequence numbering** (``Y148``),
    not structure author numbering. There is no structure at this stage, and
    labelling these as author positions is what later produces a mutation at
    the wrong residue.
    """
    mapping = CatalyticMapping(
        catalytic_template_id=hypothesis.catalytic_template.template_id,
        alignment_quality=alignment_quality,
    )
    for spec in hypothesis.catalytic_template.catalytic_residues:
        label = _role_label(spec)
        number = hypothesis.reference.role_to_number[label]
        index = ref_to_index.get(number)
        if index is None or index >= len(sequence):
            mapping.missing_roles.append(label)
            continue
        actual = sequence[index]
        allowed = _accepted_letters(spec)
        if allowed and actual not in allowed:
            if _conservative(actual, allowed, spec):
                mapping.substituted_roles[label] = f"{actual}{index + 1}"
            mapping.missing_roles.append(label)
            continue
        mapping.role_to_index[label] = index
        mapping.role_to_residue[label] = f"{actual}{index + 1}"
    return mapping


# ==========================================================================
# Per-hypothesis assessment
# ==========================================================================

class AnnotationPolicy(BaseModel):
    """Run settings for family assignment and network construction.

    ``min_identity_for_signal`` has no default on purpose. There is no
    universal identity at which two proteins share a family, and a hard-coded
    "30%" would be a universal catalytic-family threshold wearing a different
    hat. With several hypotheses loaded, identity is used *comparatively*: it
    counts as a signal for the family whose reference the candidate is closest
    to, by at least ``identity_margin_pct``. With a single hypothesis there is
    nothing to compare against, so the signal counts only if the operator
    supplies a sourced floor; otherwise it is withheld and reported.
    """

    model_config = ConfigDict(extra="forbid")

    min_identity_for_signal: float | None = Field(None, ge=0.0, le=100.0)
    identity_margin_pct: float = Field(DEFAULT_IDENTITY_MARGIN_PCT, ge=0.0)
    min_motifs_for_signal: int = Field(DEFAULT_MIN_MOTIFS_FOR_SIGNAL, ge=1)
    length_tolerance: float = Field(DEFAULT_LENGTH_TOLERANCE, gt=0.0)
    network_min_identity_pct: float = Field(DEFAULT_NETWORK_IDENTITY_PCT, ge=0.0, le=100.0)
    network_min_coverage: float = Field(DEFAULT_NETWORK_COVERAGE, ge=0.0, le=1.0)
    max_pairwise_alignments: int = Field(DEFAULT_MAX_PAIRWISE_ALIGNMENTS, ge=0)
    build_network: bool = True
    source: str = "run policy"


@dataclass
class HypothesisAssessment:
    """What one hypothesis says about one sequence, before any winner is picked."""

    template_id: str
    family_name: str
    identity_pct: float | None
    coverage: float | None
    matched_motifs: list[dict[str, Any]]
    domains: list[dict[str, Any]]
    mapping: CatalyticMapping
    #: 1-based reference position -> 0-based candidate index, kept so the
    #: cofactor rules are evaluated against the same alignment the catalytic
    #: mapping came from rather than a second, possibly different, one.
    ref_to_index: dict[int, int] = field(default_factory=dict)
    supporting: list[str] = field(default_factory=list)
    conflicting: list[str] = field(default_factory=list)

    @property
    def n_supporting(self) -> int:
        """Distinct supporting signal types.

        De-duplicated on purpose: three motifs are one signal type, and
        counting them separately is how one line of evidence turns into a
        confident family call.
        """
        return len(set(self.supporting))


def _assess(sequence: str, hypothesis: FamilyHypothesis,
            domain_hits: Sequence[DomainHit],
            policy: AnnotationPolicy) -> HypothesisAssessment:
    """Evaluate every signal of one hypothesis against one sequence.

    The identity signal is *not* decided here: it depends on the other
    hypotheses and is added by :func:`_apply_identity_signal` once all of them
    have been evaluated.
    """
    ref_to_index, aln = map_reference_positions(sequence, hypothesis.reference.sequence)
    identity_pct = aln.identity * 100.0
    coverage = min(aln.coverage_a(), aln.coverage_b())
    motifs = match_motifs(sequence, hypothesis.family_template.conserved_motifs)
    family_ids = {str(x).strip().upper() for x in
                  list(hypothesis.family_template.pfam_ids)
                  + list(hypothesis.family_template.interpro_ids)
                  + list(hypothesis.family_template.domain_architecture)}
    matched_domains = [d.as_dict() for d in domain_hits
                       if str(d.accession).strip().upper() in family_ids]
    mapping = map_catalytic_roles(sequence, hypothesis, ref_to_index,
                                  alignment_quality=coverage)

    supporting: list[str] = []
    conflicting: list[str] = []
    if matched_domains:
        supporting.append(SIGNAL_DOMAIN)
    if len(motifs) >= policy.min_motifs_for_signal:
        supporting.append(SIGNAL_MOTIFS)
    if not mapping.missing_roles and mapping.role_to_index:
        supporting.append(SIGNAL_CATALYTIC)
    elif mapping.missing_roles:
        conflicting.append(
            f"catalytic_roles_missing:{','.join(sorted(mapping.missing_roles))}"
        )

    ref_len = len(hypothesis.reference.sequence)
    if ref_len and abs(len(sequence) - ref_len) / ref_len > policy.length_tolerance:
        conflicting.append(
            f"length_outside_reference_range:{len(sequence)}vs{ref_len}"
        )

    return HypothesisAssessment(
        template_id=hypothesis.template_id,
        family_name=hypothesis.family_name,
        identity_pct=identity_pct,
        coverage=coverage,
        matched_motifs=motifs,
        domains=matched_domains,
        mapping=mapping,
        ref_to_index=ref_to_index,
        supporting=supporting,
        conflicting=conflicting,
    )


def _apply_identity_signal(assessments: Sequence[HypothesisAssessment],
                           policy: AnnotationPolicy) -> str | None:
    """Add the identity signal where it discriminates. Returns a note if withheld.

    Comparative when there is something to compare with, absolute only when the
    operator supplied a sourced floor. Withholding the signal is reported, not
    hidden: a family call resting on three signals is a different claim from
    one resting on three signals plus an unmeasured fourth.
    """
    scored = [a for a in assessments if a.identity_pct is not None]
    if not scored:
        return "no identity could be computed"
    ordered = sorted(scored, key=lambda a: a.identity_pct or 0.0, reverse=True)
    best = ordered[0]
    floor = policy.min_identity_for_signal
    if len(ordered) == 1:
        if floor is None:
            return ("single hypothesis and no min_identity_for_signal configured; "
                    "the identity signal was withheld rather than granted by a "
                    "hard-coded cut-off")
        if (best.identity_pct or 0.0) >= floor:
            best.supporting.append(SIGNAL_IDENTITY)
        return None
    runner_up = ordered[1].identity_pct or 0.0
    if floor is not None and (best.identity_pct or 0.0) < floor:
        return (f"best identity {best.identity_pct:.1f}% is below the configured "
                f"floor of {floor}%")
    if (best.identity_pct or 0.0) - runner_up >= policy.identity_margin_pct:
        best.supporting.append(SIGNAL_IDENTITY)
        return None
    for a in ordered[:2]:
        a.conflicting.append(
            f"identity_does_not_discriminate:{ordered[0].template_id}"
            f"~{ordered[1].template_id}"
        )
    return (f"top two hypotheses are within {policy.identity_margin_pct} "
            f"percentage points of identity")


def _cofactor_call(sequence: str, hypothesis: FamilyHypothesis,
                   ref_to_index: Mapping[int, int]) -> tuple[str | None, str | None]:
    """Cofactor preference for this sequence under this hypothesis, with evidence.

    Returns ``(None, None)`` whenever the evidence does not single one out --
    no rules, no rule satisfied, two rules satisfied, or positions that could
    not be aligned. An unsupported cofactor preference is the error that turns
    into an enzyme assayed with the wrong nicotinamide and recorded as inactive,
    so the only safe default is no answer.

    Family-level preferences from the family template are used only when the
    template declares exactly one, and are labelled as family-level rather than
    as a statement about this sequence.
    """
    satisfied: list[CofactorRecognitionRule] = []
    for rule in hypothesis.cofactor_rules:
        if rule.family_template_id != hypothesis.template_id:
            raise FabricationGuardError(
                f"cofactor rule keyed to {rule.family_template_id} evaluated "
                f"under {hypothesis.template_id}"
            )
        accepted = set(rule.accepted_residues)
        ok = True
        for number in rule.reference_positions:
            index = ref_to_index.get(number)
            if index is None or index >= len(sequence) or sequence[index] not in accepted:
                ok = False
                break
        if ok:
            satisfied.append(rule)
    if len(satisfied) == 1:
        rule = satisfied[0]
        positions = ",".join(str(p) for p in rule.reference_positions)
        return rule.cofactor, (
            f"{rule.evidence} [rule of {rule.family_template_id} at reference "
            f"positions {positions}]"
        )
    if len(satisfied) > 1:
        return None, None
    prefs = hypothesis.family_template.cofactor_preference
    if len(prefs) == 1:
        cofactor, evidence = next(iter(prefs.items()))
        return cofactor, (
            f"{evidence} [family-level preference of "
            f"{hypothesis.family_template.template_id}; not verified on this "
            f"sequence]"
        )
    return None, None


# ==========================================================================
# Sequence similarity network
# ==========================================================================

@dataclass
class SequenceNetwork:
    """Nodes, edges and connected components of a similarity network.

    **A component is not a phylogeny and not a functional group.** It is the
    set of sequences joined by pairwise similarity above the chosen threshold,
    and it changes when the threshold changes. Nothing in this pipeline may
    conclude shared substrate scope, shared mechanism or common ancestry from
    membership; the component id exists to spread a selection across the pool.
    """

    nodes: list[str]
    edges: list[tuple[str, str, float, float]]      # a, b, identity_pct, coverage
    clusters: dict[str, str]                        # node -> cluster id
    identity_threshold_pct: float
    coverage_threshold: float
    n_alignments: int = 0
    n_prefiltered_pairs: int = 0
    budget_exhausted: bool = False

    def component_sizes(self) -> dict[str, int]:
        """Members per component, reported so a giant component is visible.

        One component holding nearly everything means the threshold is too
        permissive, not that the family is homogeneous.
        """
        sizes: dict[str, int] = {}
        for cid in self.clusters.values():
            sizes[cid] = sizes.get(cid, 0) + 1
        return sizes


def build_similarity_network(
    items: Sequence[tuple[str, str]],
    *,
    identity_threshold_pct: float = DEFAULT_NETWORK_IDENTITY_PCT,
    coverage_threshold: float = DEFAULT_NETWORK_COVERAGE,
    max_alignments: int = DEFAULT_MAX_PAIRWISE_ALIGNMENTS,
    kmer_prefilter: float | None = 0.05,
    kmer_size: int = 3,
) -> SequenceNetwork:
    """All-pairs similarity network over ``(node_id, sequence)`` pairs.

    An edge needs identity **and** coverage. Identity alone over a short
    overlap is the classic way a 60-residue fragment acquires high-identity
    edges to half the pool and welds unrelated components together.

    Pure Python and O(n^2) alignments in the worst case, so a k-mer screen skips
    obviously distant pairs and ``max_alignments`` caps the work. Both the
    screened count and an exhausted budget are reported: a network that stopped
    early has missing edges, and missing edges split components, which would
    otherwise read as real structure.
    """
    nodes = [key for key, _ in items]
    sequences = dict(items)
    kmers = {key: kmer_set(seq, kmer_size) for key, seq in items}
    parent: dict[str, str] = {key: key for key in nodes}

    def find(x: str) -> str:
        """Union-find root with path compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        """Merge two components, keeping the lexicographically smaller root.

        Smaller-root rather than by rank so component membership is identical
        across runs; an unstable component id would make two runs of the same
        pool look like different analyses.
        """
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    edges: list[tuple[str, str, float, float]] = []
    n_align = 0
    n_prefiltered = 0
    exhausted = False
    for i in range(len(nodes)):
        if exhausted:
            break
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if kmer_prefilter is not None:
                dist = jaccard_distance(kmers[a], kmers[b])
                if dist is not None and (1.0 - dist) < kmer_prefilter:
                    n_prefiltered += 1
                    continue
            if n_align >= max_alignments:
                exhausted = True
                break
            n_align += 1
            aln = needleman_wunsch(sequences[a], sequences[b])
            coverage = min(aln.coverage_a(), aln.coverage_b())
            identity_pct = aln.identity * 100.0
            if identity_pct >= identity_threshold_pct and coverage >= coverage_threshold:
                edges.append((a, b, identity_pct, coverage))
                union(a, b)

    roots = sorted({find(n) for n in nodes})
    order = {root: f"ssn_{i + 1:04d}" for i, root in enumerate(roots)}
    clusters = {n: order[find(n)] for n in nodes}
    return SequenceNetwork(
        nodes=nodes, edges=edges, clusters=clusters,
        identity_threshold_pct=identity_threshold_pct,
        coverage_threshold=coverage_threshold,
        n_alignments=n_align, n_prefiltered_pairs=n_prefiltered,
        budget_exhausted=exhausted,
    )


# ==========================================================================
# Hand-off text
# ==========================================================================

HANDOFF_README: str = """\
# Alignment and tree hand-off

Per-family FASTA files in this directory are ready for MAFFT and IQ-TREE:

    mafft --maxiterate 1000 --localpair <family>.fasta > <family>.aln.fasta
    iqtree2 -s <family>.aln.fasta -m MFP -B 1000

## Run them PER FAMILY. Do not concatenate these files.

SDR, MDR/ADH and AKR are not homologous to one another: they have different
folds, different catalytic residues and different cofactor-binding
architecture. A multiple alignment across them is an alignment of positions
that do not correspond, and a tree built from it is a picture of alignment
artefacts with bootstrap support attached. Their relationship, if one is
needed, is a structural comparison, not a sequence tree.

`unassigned.fasta` holds sequences no hypothesis claimed with enough
independent signals. It is a work queue, not a family.

## The network files are not a phylogeny

`network_edges.tsv` and `network_clusters.tsv` describe a sequence similarity
network in the EFI-EST sense: nodes joined when pairwise identity and coverage
clear the thresholds recorded in the header of this run's provenance. A
connected component is not a clade, is not monophyletic, and demonstrates no
shared function. Component membership changes with the threshold. Use it to
spread a selection across the pool, never as evidence about an enzyme.
"""


# ==========================================================================
# The interface
# ==========================================================================

class AnnotateFamily(ScientificInterface):
    """Assign families from combined signals and map catalytic roles per family.

    Produces, per sequence: a :class:`FamilyAnnotation` whose confidence comes
    from counting *independent* signals against the family template's own
    ``min_independent_signals``, and a :class:`CatalyticMapping` stamped with
    the catalytic template it came from. Sequences no hypothesis claims are
    left unassigned rather than being filed under the nearest family.
    """

    name: ClassVar[str] = "annotate_family"
    description: ClassVar[str] = (
        "Family assignment from identity, domain architecture, conserved "
        "positions and catalytic residue correspondence, with per-family "
        "catalytic mapping, a similarity network and an alignment hand-off."
    )
    required_fields: ClassVar[tuple[str, ...]] = ()
    required_approvals: ClassVar[tuple[str, ...]] = ()
    depends_on: ClassVar[tuple[str, ...]] = ("mine_sequences",)
    version: ClassVar[str] = "0.1.0"

    def execute(
        self,
        ctx: RunContext,
        *,
        sequences: Sequence[SequenceRecord] | None = None,
        sequences_fasta: str | Path | None = None,
        hypotheses: Sequence[FamilyHypothesis] | HypothesisSet | None = None,
        domain_hits: Mapping[str, Sequence[DomainHit]] | None = None,
        policy: AnnotationPolicy | None = None,
        **_: Any,
    ) -> ToolResult:
        """Annotate every sequence against every loaded family hypothesis.

        ``domain_hits`` is keyed by candidate id or by ``sequence_sha256``. When
        it is absent the domain signal is simply not available; this is recorded
        as an uncertainty and never replaced by an assumption.
        """
        policy = policy or AnnotationPolicy()
        records, load_error = self._load_sequences(sequences, sequences_fasta)
        if load_error is not None:
            return load_error
        hset, hyp_error = self._load_hypotheses(hypotheses)
        if hyp_error is not None:
            return hyp_error
        assert hset is not None

        result = ToolResult(status=Status.SUCCESS)
        domain_hits = dict(domain_hits or {})
        if not domain_hits:
            result.add_uncertainty(
                "domain_evidence_absent",
                "No domain annotations were supplied, so domain architecture "
                "could not contribute a signal. Run an hmmscan against Pfam and "
                "pass the hits to raise family confidence honestly.",
                affects=[self.name], resolvable_by="hmmscan / InterProScan output",
            )

        # -- similarity network (needed before annotations, to carry cluster ids)
        network = self._build_network(records, policy, result)

        annotations: list[dict[str, Any]] = []
        candidates: list[Candidate] = []
        per_family: dict[str, list[SequenceRecord]] = {}
        unassigned: list[SequenceRecord] = []
        identity_notes: dict[str, int] = {}

        for rec in records:
            hits = list(domain_hits.get(rec.candidate_id, ())) or \
                list(domain_hits.get(rec.sequence_sha256 or "", ()))
            assessments = [_assess(rec.sequence, h, hits, policy) for h in hset]
            note = _apply_identity_signal(assessments, policy)
            if note:
                identity_notes[note] = identity_notes.get(note, 0) + 1

            winner = self._pick(assessments, hset, policy)
            annotation, mapping = self._build_annotation(
                rec, winner, assessments, hset, network, policy
            )
            candidate = Candidate(
                candidate_id=rec.candidate_id,
                sequence_record=rec,
                family=annotation,
                catalytic_mapping=mapping,
            )
            candidates.append(candidate)
            annotations.append(self._annotation_row(rec, annotation, mapping, network))
            if annotation.family_template_id:
                per_family.setdefault(annotation.family_template_id, []).append(rec)
            else:
                unassigned.append(rec)

        for note, count in sorted(identity_notes.items()):
            result.add_flag(
                "identity_signal_withheld", Severity.INFO,
                f"{count} sequence(s): {note}",
            )

        # -- artifacts ---------------------------------------------------------
        tsv_path = self._write_annotations(ctx, annotations)
        family_dir = ctx.dir("annotate_family", "family_analysis")
        fasta_paths = self._write_family_fastas(family_dir, per_family, unassigned, hset)
        edges_path, clusters_path = self._write_network(family_dir, network)
        readme_path = family_dir / "ALIGNMENT_HANDOFF.md"
        readme_path.write_text(HANDOFF_README, encoding="utf-8")

        result.artifacts.append(Artifact(
            key="sequence_annotations", path=str(tsv_path), kind="table",
            sha256=sha256_file(tsv_path), n_records=len(annotations),
            summary="one row per sequence: family call, signals, catalytic roles",
        ))
        result.artifacts.append(Artifact(
            key="family_analysis_dir", path=str(family_dir), kind="object",
            n_records=len(fasta_paths),
            summary=("per-family FASTA for MAFFT/IQ-TREE, network edge list, "
                     "cluster assignments and the hand-off note"),
        ))
        for label, path in fasta_paths.items():
            result.artifacts.append(Artifact(
                key=f"family_fasta:{label}", path=str(path), kind="file",
                sha256=sha256_file(path),
                n_records=len(per_family.get(label, unassigned)),
                summary=f"sequences assigned to {label}",
            ))
        result.artifacts.append(Artifact(
            key="network_edges", path=str(edges_path), kind="table",
            sha256=sha256_file(edges_path), n_records=len(network.edges),
            summary=(f"edges at >= {policy.network_min_identity_pct}% identity and "
                     f">= {policy.network_min_coverage} coverage; not a phylogeny"),
        ))
        result.artifacts.append(Artifact(
            key="network_clusters", path=str(clusters_path), kind="table",
            sha256=sha256_file(clusters_path), n_records=len(network.clusters),
            summary="connected components; not a demonstrated functional group",
        ))
        result.artifacts.append(Artifact(
            key="alignment_handoff", path=str(readme_path), kind="file",
            sha256=sha256_file(readme_path),
            summary="MAFFT/IQ-TREE instructions; families must be run separately",
        ))

        # -- provenance ----------------------------------------------------------
        result.provenance = Provenance(
            tool=self.name,
            tool_version=self.version,
            inputs_sha256={
                "sequences": sha256_obj([r.sequence_sha256 for r in records]),
                "hypotheses": sha256_obj([
                    {"family_template": h.family_template.template_id,
                     "catalytic_template": h.catalytic_template.template_id,
                     "reference": h.reference.accession,
                     "cofactor_rules": [r.model_dump(mode="json")
                                        for r in h.cofactor_rules]}
                    for h in hset
                ]),
            },
            databases={},
            models={h.template_id: f"reference:{h.reference.accession}"
                    for h in hset},
            parameters={
                "n_sequences": len(records),
                "hypotheses": hset.ids(),
                "family_names": sorted({h.family_name for h in hset}),
                "min_independent_signals": {h.template_id: h.min_independent_signals
                                            for h in hset},
                "policy": policy.model_dump(mode="json"),
                "network": {
                    "identity_threshold_pct": network.identity_threshold_pct,
                    "coverage_threshold": network.coverage_threshold,
                    "n_edges": len(network.edges),
                    "n_components": len(set(network.clusters.values())),
                    "n_alignments": network.n_alignments,
                    "n_prefiltered_pairs": network.n_prefiltered_pairs,
                    "budget_exhausted": network.budget_exhausted,
                },
                "numbering_basis": "candidate_sequence_1based",
                "domain_evidence_supplied": bool(domain_hits),
            },
            random_seed=ctx.seed_for(self.name),
        )

        self._summarise(result, annotations, per_family, unassigned, hset, network)
        result.data.update({
            "annotations": annotations,
            "candidates": [c.model_dump(mode="json") for c in candidates],
            "per_family_counts": {k: len(v) for k, v in sorted(per_family.items())},
            "unassigned": [r.candidate_id for r in unassigned],
            "network_clusters": network.clusters,
            "network_component_sizes": network.component_sizes(),
            "numbering_basis": ("role_to_residue is 1-based candidate-sequence "
                                "numbering, not structure author numbering"),
            "network_caveat": ("a connected component is not a phylogeny and not "
                               "a demonstrated functional group"),
            "tree_handoff": ("non-homologous families must be aligned and "
                             "tree-built separately; see ALIGNMENT_HANDOFF.md"),
        })
        return result

    # -- loading -----------------------------------------------------------
    def _load_sequences(
        self, sequences: Sequence[SequenceRecord] | None,
        sequences_fasta: str | Path | None,
    ) -> tuple[list[SequenceRecord], ToolResult | None]:
        """Take records directly, or read the pool FASTA this step depends on."""
        if sequences:
            return list(sequences), None
        if sequences_fasta is None:
            return [], ToolResult.failure(
                self.name,
                "no sequences supplied: pass sequences=[SequenceRecord, ...] or "
                "sequences_fasta=<path to mine_sequences/candidate_sequences.fasta>",
                code="no_sequences",
            )
        path = Path(sequences_fasta)
        if not path.is_file():
            return [], ToolResult.failure(
                self.name,
                f"candidate pool not found at {path}; this step needs the FASTA "
                f"written by mine_sequences and will not re-derive it",
                code="missing_input_artifact",
            )
        records: list[SequenceRecord] = []
        for entry in read_fasta(path):
            meta = _parse_header_fields(entry.description)
            records.append(SequenceRecord(
                candidate_id=entry.identifier,
                sequence=entry.sequence,
                accession=meta.get("accession"),
                source_database=(meta.get("db") or "").split("@")[0] or None,
                database_version=(meta.get("db") or "@").split("@")[-1] or None,
                seed_accession=meta.get("seed"),
                search_method=meta.get("method"),
            ))
        if not records:
            return [], ToolResult.failure(
                self.name, f"{path} contains no sequences", code="empty_pool",
            )
        return records, None

    def _load_hypotheses(
        self, hypotheses: Sequence[FamilyHypothesis] | HypothesisSet | None,
    ) -> tuple[HypothesisSet | None, ToolResult | None]:
        """Build the hypothesis set, failing loudly on an inconsistent binding."""
        if hypotheses is None:
            return None, ToolResult.failure(
                self.name,
                "no family hypotheses supplied; a family cannot be assigned "
                "without a sourced family template, its catalytic template and a "
                "reference sequence",
                code="no_hypotheses",
            )
        if isinstance(hypotheses, HypothesisSet):
            hset = hypotheses
        else:
            hset = HypothesisSet(hypotheses)
        if len(hset) == 0:
            return None, ToolResult.failure(
                self.name, "the hypothesis set is empty", code="no_hypotheses",
            )
        return hset, None

    # -- per-sequence decisions ---------------------------------------------
    def _pick(self, assessments: Sequence[HypothesisAssessment],
              hset: HypothesisSet,
              policy: AnnotationPolicy) -> HypothesisAssessment | None:
        """Choose the winning hypothesis, or none.

        A hypothesis wins only by having strictly more independent supporting
        signals than every other *and* reaching its own family template's
        ``min_independent_signals``. A tie leaves the sequence unassigned: with
        SDR, MDR/ADH and AKR all plausible, "probably the first one" is the
        answer that puts a zinc-dependent enzyme into an SDR mechanism.
        """
        scored = [a for a in assessments if a.n_supporting > 0]
        if not scored:
            return None
        ordered = sorted(scored, key=lambda a: (a.n_supporting,
                                                a.identity_pct or 0.0), reverse=True)
        best = ordered[0]
        if len(ordered) > 1 and ordered[1].n_supporting == best.n_supporting:
            return None
        if best.n_supporting < hset.get(best.template_id).min_independent_signals:
            return None
        return best

    def _build_annotation(
        self,
        record: SequenceRecord,
        winner: HypothesisAssessment | None,
        assessments: Sequence[HypothesisAssessment],
        hset: HypothesisSet,
        network: SequenceNetwork,
        policy: AnnotationPolicy,
    ) -> tuple[FamilyAnnotation, CatalyticMapping]:
        """Assemble the annotation and the catalytic mapping for one sequence.

        When nothing wins, the annotation records the signals that were found
        and stays unassigned with an empty mapping. An unassigned sequence with
        its evidence listed is a work item; a sequence filed under the nearest
        family is a fabricated mechanism.
        """
        cluster_id = network.clusters.get(record.candidate_id)
        competing = [a.template_id for a in assessments
                     if winner is not None and a.template_id != winner.template_id
                     and a.n_supporting >= hset.get(a.template_id).min_independent_signals]

        if winner is None:
            best_by_identity = max(assessments, key=lambda a: a.identity_pct or 0.0,
                                   default=None)
            annotation = FamilyAnnotation(
                family_name=None,
                family_template_id=None,
                overall_identity_to_seed=(best_by_identity.identity_pct
                                          if best_by_identity else None),
                domains=[d for a in assessments for d in a.domains],
                matched_motifs=[m for a in assessments for m in a.matched_motifs],
                sequence_cluster_id=cluster_id,
                signals_supporting=sorted({s for a in assessments
                                           for s in a.supporting}),
                signals_conflicting=sorted({c for a in assessments
                                            for c in a.conflicting}),
            )
            annotation.recompute_confidence(
                min_signals=max(h.min_independent_signals for h in hset)
            )
            return annotation, CatalyticMapping()

        hypothesis = hset.get(winner.template_id)
        cofactor, cofactor_evidence = _cofactor_call(record.sequence, hypothesis,
                                                     winner.ref_to_index)
        conflicting = list(winner.conflicting)
        conflicting += [f"competing_family:{cid}" for cid in competing]
        annotation = FamilyAnnotation(
            family_name=hypothesis.family_name,
            family_template_id=hypothesis.template_id,
            overall_identity_to_seed=winner.identity_pct,
            domains=winner.domains,
            matched_motifs=winner.matched_motifs,
            cofactor_preference=cofactor,
            cofactor_preference_evidence=cofactor_evidence,
            sequence_cluster_id=cluster_id,
            signals_supporting=sorted(set(winner.supporting)),
            signals_conflicting=sorted(set(conflicting)),
        )
        annotation.recompute_confidence(min_signals=hypothesis.min_independent_signals)
        return annotation, winner.mapping

    # -- artifacts -----------------------------------------------------------
    ANNOTATION_COLUMNS: ClassVar[tuple[str, ...]] = (
        "candidate_id", "sequence_sha256", "accession", "length",
        "family_name", "family_template_id", "catalytic_template_id",
        "confidence", "n_signals_supporting", "signals_supporting",
        "signals_conflicting", "identity_to_reference_pct", "n_motifs_matched",
        "motifs", "domains", "cofactor_preference", "cofactor_preference_evidence",
        "catalytic_roles_mapped", "missing_roles", "substituted_roles",
        "alignment_quality", "network_cluster_id", "numbering_basis",
    )

    def _annotation_row(self, record: SequenceRecord, annotation: FamilyAnnotation,
                        mapping: CatalyticMapping,
                        network: SequenceNetwork) -> dict[str, Any]:
        return {
            "candidate_id": record.candidate_id,
            "sequence_sha256": record.sequence_sha256,
            "accession": record.accession,
            "length": record.length,
            "family_name": annotation.family_name,
            "family_template_id": annotation.family_template_id,
            "catalytic_template_id": mapping.catalytic_template_id,
            "confidence": annotation.confidence.value,
            "n_signals_supporting": annotation.n_independent_signals,
            "signals_supporting": ",".join(annotation.signals_supporting),
            "signals_conflicting": ",".join(annotation.signals_conflicting),
            "identity_to_reference_pct": annotation.overall_identity_to_seed,
            "n_motifs_matched": len(annotation.matched_motifs),
            "motifs": ",".join(f"{m['name']}@{m['start']}"
                               for m in annotation.matched_motifs),
            "domains": ",".join(str(d.get("id")) for d in annotation.domains),
            "cofactor_preference": annotation.cofactor_preference,
            "cofactor_preference_evidence": annotation.cofactor_preference_evidence,
            "catalytic_roles_mapped": ",".join(
                f"{k}={v}" for k, v in sorted(mapping.role_to_residue.items())),
            "missing_roles": ",".join(sorted(mapping.missing_roles)),
            "substituted_roles": ",".join(
                f"{k}={v}" for k, v in sorted(mapping.substituted_roles.items())),
            "alignment_quality": mapping.alignment_quality,
            "network_cluster_id": annotation.sequence_cluster_id,
            "numbering_basis": "candidate_sequence_1based",
        }

    def _write_annotations(self, ctx: RunContext,
                           rows: Sequence[Mapping[str, Any]]) -> Path:
        return _write_tsv(
            ctx.path("annotate_family", "sequence_annotations.tsv"),
            self.ANNOTATION_COLUMNS,
            [[_tsv_cell(r.get(c)) for c in self.ANNOTATION_COLUMNS] for r in rows],
        )

    def _write_family_fastas(
        self, family_dir: Path,
        per_family: Mapping[str, Sequence[SequenceRecord]],
        unassigned: Sequence[SequenceRecord],
        hset: HypothesisSet,
    ) -> dict[str, Path]:
        """One FASTA per family template id, plus the unassigned work queue.

        Files are per *template id*, not per family name, so two templates for
        the same family (a different reference, a different curation) never land
        in one alignment by accident.
        """
        paths: dict[str, Path] = {}
        for template_id, records in sorted(per_family.items()):
            name = f"{_safe_name(hset.get(template_id).family_name)}__{_safe_name(template_id)}"
            path = family_dir / f"{name}.fasta"
            write_fasta(
                [FastaEntry(r.candidate_id, f"accession={r.accession or ''}",
                            r.sequence) for r in records],
                path,
            )
            paths[template_id] = path
        if unassigned:
            path = family_dir / "unassigned.fasta"
            write_fasta(
                [FastaEntry(r.candidate_id, "family=unassigned", r.sequence)
                 for r in unassigned],
                path,
            )
            paths["unassigned"] = path
        return paths

    def _write_network(self, family_dir: Path,
                       network: SequenceNetwork) -> tuple[Path, Path]:
        edges = _write_tsv(
            family_dir / "network_edges.tsv",
            ("node_a", "node_b", "percent_identity", "alignment_coverage"),
            [[a, b, _tsv_cell(pid), _tsv_cell(cov)]
             for a, b, pid, cov in network.edges],
        )
        sizes = network.component_sizes()
        clusters = _write_tsv(
            family_dir / "network_clusters.tsv",
            ("node", "network_cluster_id", "component_size", "caveat"),
            [[node, cid, str(sizes[cid]),
              "connected component; not a phylogeny, not a functional group"]
             for node, cid in sorted(network.clusters.items())],
        )
        return edges, clusters

    def _build_network(self, records: Sequence[SequenceRecord],
                       policy: AnnotationPolicy,
                       result: ToolResult) -> SequenceNetwork:
        """Build the SSN, or an empty one when the run disabled it."""
        if not policy.build_network or len(records) < 2:
            return SequenceNetwork(
                nodes=[r.candidate_id for r in records], edges=[],
                clusters={r.candidate_id: f"ssn_{i + 1:04d}"
                          for i, r in enumerate(records)},
                identity_threshold_pct=policy.network_min_identity_pct,
                coverage_threshold=policy.network_min_coverage,
            )
        network = build_similarity_network(
            [(r.candidate_id, r.sequence) for r in records],
            identity_threshold_pct=policy.network_min_identity_pct,
            coverage_threshold=policy.network_min_coverage,
            max_alignments=policy.max_pairwise_alignments,
        )
        if network.budget_exhausted:
            result.add_flag(
                "network_budget_exhausted", Severity.WARN,
                f"the pairwise alignment budget ({policy.max_pairwise_alignments}) "
                f"was reached; edges are missing and components are therefore "
                f"split more finely than the data warrant",
            )
        return network

    # -- summary ---------------------------------------------------------------
    def _summarise(
        self,
        result: ToolResult,
        annotations: Sequence[Mapping[str, Any]],
        per_family: Mapping[str, Sequence[SequenceRecord]],
        unassigned: Sequence[SequenceRecord],
        hset: HypothesisSet,
        network: SequenceNetwork,
    ) -> None:
        """Set status, flags and next actions from what was actually established."""
        n = len(annotations)
        strong = sum(1 for a in annotations
                     if a["confidence"] == ConfidenceLevel.STRONG.value)
        contradictory = [a["candidate_id"] for a in annotations
                         if a["confidence"] == ConfidenceLevel.CONTRADICTORY.value]
        incomplete = [a["candidate_id"] for a in annotations
                      if a["family_template_id"] and a["missing_roles"]]

        if unassigned:
            result.add_flag(
                "family_unassigned", Severity.WARN,
                f"{len(unassigned)} of {n} sequence(s) reached no family with "
                f"enough independent signals and were left unassigned rather "
                f"than filed under the closest family",
            )
        if contradictory:
            result.add_flag(
                "family_signals_contradictory", Severity.WARN,
                f"{len(contradictory)} sequence(s) carry conflicting family "
                f"signals (e.g. {', '.join(contradictory[:3])}); these are "
                f"competing mechanistic hypotheses, not a ranking",
            )
        if incomplete:
            result.add_flag(
                "catalytic_roles_incomplete", Severity.WARN,
                f"{len(incomplete)} assigned sequence(s) are missing at least one "
                f"catalytic role of their family's template "
                f"(e.g. {', '.join(incomplete[:3])})",
            )
            result.add_uncertainty(
                "catalytic_machinery_incomplete",
                "Are the missing catalytic roles genuinely absent, or is the "
                "reference alignment wrong in that region?",
                affects=[a for a in incomplete[:20]],
                resolvable_by="structure-based mapping, or a better family reference",
            )
        if unassigned or contradictory:
            result.status = Status.PARTIAL
            result.message = (
                f"{strong}/{n} sequence(s) reached a strong family call; "
                f"{len(unassigned)} unassigned, {len(contradictory)} contradictory"
            )
        else:
            result.message = (
                f"{n} sequence(s) annotated across {len(per_family)} family "
                f"template(s); {strong} strong call(s)"
            )

        result.add_uncertainty(
            "network_is_not_function",
            "The similarity network groups sequences by similarity only. Which "
            "components, if any, have an experimentally characterised member?",
            affects=sorted(set(network.clusters.values()))[:20],
            resolvable_by="literature evidence per component",
        )
        result.add_next(
            "align_and_tree",
            "Run MAFFT and IQ-TREE PER FAMILY using the per-family FASTA files; "
            "non-homologous families must not be forced into one alignment",
            {"handoff": "family_analysis/ALIGNMENT_HANDOFF.md",
             "families": sorted(per_family)}, requires_human=True,
        )
        if unassigned:
            result.add_next(
                "annotate_family",
                "Unassigned sequences need another signal: supply domain hits, a "
                "further family hypothesis, or a sourced identity floor",
                {"n_unassigned": len(unassigned)}, requires_human=True,
            )


def _safe_name(text: str) -> str:
    """Filesystem-safe family or template label."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text))[:60] \
        or "family"


def _parse_header_fields(description: str) -> dict[str, str]:
    """Pull ``key=value`` tokens out of a pool FASTA description.

    Tolerant on purpose: a header written by another tool simply yields fewer
    fields, which leaves the corresponding record attributes ``None`` instead of
    inventing a database version.
    """
    out: dict[str, str] = {}
    for token in description.split():
        if "=" in token:
            key, _, value = token.partition("=")
            if value:
                out[key] = value
    return out
