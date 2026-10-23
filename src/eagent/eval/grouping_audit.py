"""Audit a fold grouping against the identity threshold it promised.

WHY THIS EXISTS
===============
Every number in the retrospective benchmark rests on one claim: a held-out
sequence has no training-fold neighbour above the identity threshold, so the
score is not homology lookup with extra steps. The benchmark establishes that
claim by clustering at the threshold and assigning whole clusters to folds --
and then it has, until now, *checked* the claim with the same k-mer cosine it
was measuring (``similarity_to_training``). A proxy cannot audit itself: that
cosine was measured at 1.07x its own composition background on this pool, so a
reassuring median tells you about amino-acid composition, not about identity.

:func:`audit_grouping` checks the claim with the definition the clustering
actually used -- a full Needleman-Wunsch identity scaled by coverage. It cannot
afford to check every pair: 309 sequences is 38,000 cross-fold pairs and one
alignment costs around 44 ms, so the complete check is about half an hour. It
therefore aligns the pairs admitted by one or more **screens**, and the honest
statement is about the screen, not about the pool:

    at the clustering's own k-mer screen, every cross-fold pair the clustering
    itself would have aligned was aligned here -- so the audit's blind spot is
    *exactly* the clustering's blind spot, which the clustering already admits
    and counts but cannot localise.

That is a far stronger claim than "the nearest few neighbours looked fine", and
it is cheap: on the SDR pool, 143 of 38,192 cross-fold pairs clear the screen.
A looser screen is strictly more thorough and strictly slower, and the audit
reports which screen it ran, how many pairs each admitted, and how many pairs
no screen looked at.

WHAT A BREACH WOULD MEAN
========================
Three things can put two sequences above the threshold in different folds, and
only one of them is a bug:

1. The clustering's k-mer prefilter is a *screen*, not a bound, so a pair it
   skipped can be a true cluster-mate. ``cluster_sequences`` counts the pairs
   it skipped -- 16,185 of them on this pool -- but cannot say which mattered.
2. Greedy clustering assigns a sequence to the first representative it clears
   the threshold against. If A is a representative, B joins A, and C clears the
   threshold against B but not against A, then C starts its own cluster -- so
   the *pairwise* property the folds promise fails even though the clustering
   did exactly what it documents. This is inherent to the method.
3. The alignment budget ran out (``budget_exhausted``), in which case later
   sequences were never compared to anything.

None of the three is visible in the clustering report, and all three move the
benchmark the same way: a breach is optimism, and undeclared optimism is the
thing this repository is built not to ship.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..errors import EAgentError
from ..science.diversity import jaccard_distance, kmer_set

__all__ = [
    "GroupingAuditError",
    "SCREEN_CAVEAT",
    "COMPLETE_GROUPING_NOTE",
    "CandidateScreen",
    "jaccard_screen",
    "top_k_similarity_screen",
    "IdentityBreach",
    "GroupingAudit",
    "audit_grouping",
]


class GroupingAuditError(EAgentError):
    """The audit could not be run as specified."""


#: Printed with every audit. A screened audit that found nothing has not proved
#: the grouping clean, and that distinction is the whole point of the module,
#: so the sentence travels with the result rather than living in a docstring
#: no reader of the report will ever see.
SCREEN_CAVEAT: str = (
    "a screened audit is a LOWER BOUND on breaches: a cross-fold pair that no "
    "screen admitted was never aligned, so finding none does not make the "
    "grouping clean -- it makes it unbreached among the pairs checked."
)

#: Printed instead, when the grouping under audit aligned every pair of the
#: pool. Then no cross-cluster pair can be over the threshold -- the
#: construction already settled it -- so the audit cannot discover a leak and
#: a clean result is not evidence about the grouping's design. What it can
#: still do is catch an implementation error, which is why it is worth running
#: at all, and that is what the line has to say rather than implying a
#: discovery that was never possible.
COMPLETE_GROUPING_NOTE: str = (
    "the grouping under audit aligned every pair, so no cross-cluster pair "
    "can be over the threshold and this audit cannot discover a leak. It is "
    "an independent check of the clustering code, not evidence about the "
    "grouping's design."
)

#: A pair of example ids, lexically ordered so each unordered pair has one key.
Pair = tuple[str, str]


@dataclass(frozen=True)
class CandidateScreen:
    """A cheap rule for which cross-fold pairs are worth aligning.

    ``select`` receives the sequences and the fold map and returns the pairs it
    admits, each as a lexically ordered ``(id, id)`` tuple. It is a function of
    the whole pool rather than of one pair so that a rank-based screen ("each
    sequence's five nearest cross-fold neighbours") and a
    threshold-based one ("every pair over a k-mer Jaccard of 0.1") can both be
    expressed, and so a screen may memoise whatever it needs.

    ``description`` is printed verbatim in the report. It should say what the
    screen's blind spot is, because that blind spot is the audit's.
    """

    name: str
    description: str
    select: Callable[[Mapping[str, str], Mapping[str, int]], set[Pair]]


def _cross_fold_pairs(sequences: Mapping[str, str],
                      fold_of: Mapping[str, int]) -> list[Pair]:
    """Every pair whose two members are in different folds, ordered by id."""
    ids = sorted(sequences)
    return [(a, b) for a, b in itertools.combinations(ids, 2)
            if fold_of[a] != fold_of[b]]


def jaccard_screen(minimum: float, *, kmer_size: int = 3) -> CandidateScreen:
    """The clustering's own prefilter, applied to every cross-fold pair.

    ``minimum`` at the value :func:`~eagent.tools.mine_sequences.cluster_sequences`
    used makes the audit complete *relative to the clustering*: any pair this
    misses, the clustering never aligned either. Below that value the audit is
    strictly more thorough than the clustering was, at a cost that climbs
    steeply -- on the SDR pool, 0.10 admits 143 pairs and 0.05 admits 11,247.
    """
    if not 0.0 <= minimum <= 1.0:
        raise GroupingAuditError(
            f"a Jaccard similarity is in [0, 1], got {minimum}")

    def select(sequences: Mapping[str, str],
               fold_of: Mapping[str, int]) -> set[Pair]:
        kmers = {i: kmer_set(s, kmer_size) for i, s in sequences.items()}
        out: set[Pair] = set()
        for a, b in _cross_fold_pairs(sequences, fold_of):
            d = jaccard_distance(kmers[a], kmers[b])
            # None means a sequence shorter than k: no signature, so it cannot
            # be screened out on the strength of one. Admit it and align.
            if d is None or (1.0 - d) >= minimum:
                out.add((a, b))
        return out

    return CandidateScreen(
        name=f"kmer_jaccard>={minimum:g} (k={kmer_size})",
        description=(
            f"every cross-fold pair whose {kmer_size}-mer Jaccard similarity "
            f"is at least {minimum:g}. The k-mer/identity relationship is "
            f"monotone in practice but is not a bound, so a pair just under "
            f"the screen can still be a true cluster-mate"),
        select=select)


def top_k_similarity_screen(similarity: Callable[[str, str], float], k: int,
                            *, name: str, description: str) -> CandidateScreen:
    """Each sequence's ``k`` nearest cross-fold neighbours by ``similarity``.

    A budget, not a criterion: it admits a fixed number of pairs per sequence
    whatever their scores, so it cannot support a completeness claim the way a
    threshold screen can. It is here because a second screen with a *different*
    blind spot shrinks the union's blind spot, and a profile search ranks
    homologues better than a k-mer count does. Ties break by id, so the
    selection is deterministic for a given provider.
    """
    if k < 1:
        raise GroupingAuditError(f"k must be >= 1, got {k}")

    def select(sequences: Mapping[str, str],
               fold_of: Mapping[str, int]) -> set[Pair]:
        out: set[Pair] = set()
        ids = sorted(sequences)
        for i in ids:
            scored = [(similarity(sequences[i], sequences[j]), j) for j in ids
                      if fold_of[j] != fold_of[i]]
            scored.sort(key=lambda pair: (-pair[0], pair[1]))
            for _, j in scored[:k]:
                out.add((i, j) if i < j else (j, i))
        return out

    return CandidateScreen(name=name, description=description, select=select)


@dataclass(frozen=True)
class IdentityBreach:
    """One cross-fold pair at or above the threshold the folds promised."""

    example_id: str
    fold: int
    neighbour_id: str
    neighbour_fold: int
    identity: float
    screens: tuple[str, ...]             # which screen(s) admitted the pair

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id, "fold": self.fold,
            "neighbour_id": self.neighbour_id,
            "neighbour_fold": self.neighbour_fold,
            "identity": round(self.identity, 4),
            "admitted_by": list(self.screens),
        }


@dataclass
class GroupingAudit:
    """What the exact identity function found among the screened pairs."""

    threshold: float
    identity_name: str
    n_sequences: int
    n_folds: int
    n_cross_fold_pairs: int
    n_pairs_aligned: int
    #: Per screen: its description, how many pairs it admitted, and how many
    #: breaches it alone would have found. A screen that admits thousands of
    #: pairs and finds nothing the others did not is paying for nothing.
    screens: list[dict[str, Any]] = field(default_factory=list)
    identity_quantiles: dict[str, float] = field(default_factory=dict)
    highest_identity: float = 0.0
    breaches: list[IdentityBreach] = field(default_factory=list)
    n_breaching_sequences: int = 0
    #: Aligned pairs within ``near_miss_band`` below the threshold. Headroom:
    #: a pool whose screened pairs stop at 0.22 against a threshold of 0.40 is
    #: a different situation from one that stops at 0.398.
    n_near_misses: int = 0
    near_miss_band: float = 0.05
    #: Clustering's own identity definition, when the caller knows it. When it
    #: differs from ``identity_name`` the audit is a second opinion from a
    #: different definition, and the report says so instead of implying
    #: agreement.
    clustering_identity_name: str | None = None
    #: Whether the grouping under audit aligned every pair of the pool. True
    #: makes a clean result a check on the code rather than on the design, and
    #: :attr:`caveat` says which.
    grouping_was_complete: bool | None = None
    caveat: str = SCREEN_CAVEAT

    @property
    def n_pairs_unscreened(self) -> int:
        """Cross-fold pairs no screen admitted, so no alignment measured."""
        return self.n_cross_fold_pairs - self.n_pairs_aligned

    @property
    def clean_among_checked(self) -> bool:
        """True when no screened cross-fold pair reached the threshold.

        Deliberately not called ``clean``: the audit cannot establish that.
        """
        return not self.breaches

    @property
    def definitions_agree(self) -> bool | None:
        """Whether the audit measured identity the way the clustering did."""
        if self.clustering_identity_name is None:
            return None
        return self.clustering_identity_name == self.identity_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "identity_definition": self.identity_name,
            "grouping_was_complete": self.grouping_was_complete,
            "clustering_identity_definition": self.clustering_identity_name,
            "definitions_agree": self.definitions_agree,
            "n_sequences": self.n_sequences, "n_folds": self.n_folds,
            "n_cross_fold_pairs": self.n_cross_fold_pairs,
            "n_pairs_aligned": self.n_pairs_aligned,
            "n_pairs_unscreened": self.n_pairs_unscreened,
            "fraction_of_pairs_aligned": (
                round(self.n_pairs_aligned / self.n_cross_fold_pairs, 6)
                if self.n_cross_fold_pairs else None),
            "screens": self.screens,
            "identity_quantiles": self.identity_quantiles,
            "highest_identity": self.highest_identity,
            "n_breaching_sequences": self.n_breaching_sequences,
            "n_breaching_pairs": len(self.breaches),
            "breaches": [b.to_dict() for b in self.breaches],
            "n_near_misses": self.n_near_misses,
            "near_miss_band": self.near_miss_band,
            "clean_among_checked": self.clean_among_checked,
            "caveat": self.caveat,
        }

    def render(self) -> list[str]:
        """Report lines: what was checked, what was found, what was not seen."""
        q = self.identity_quantiles
        frac = (self.n_pairs_aligned / self.n_cross_fold_pairs
                if self.n_cross_fold_pairs else 0.0)
        out = ["", "FOLD-GROUPING AUDIT (the clustering's own identity "
                   "function, not the k-mer proxy the report otherwise uses):",
               f"  identity: {self.identity_name}",
               f"  {self.n_pairs_aligned} of {self.n_cross_fold_pairs} "
               f"cross-fold pair(s) aligned ({frac:.2%}); "
               f"{self.n_pairs_unscreened} admitted by no screen"]
        if self.definitions_agree is False:
            out.append(
                f"  WARNING: the clustering measured identity as "
                f"'{self.clustering_identity_name}', which is not what this "
                f"audit measured. The audit is a second opinion from a "
                f"different definition, not a check of the clustering's own "
                f"threshold")
        for s in self.screens:
            out.append(f"  screen {s['name']}: admitted {s['n_admitted']} "
                       f"pair(s), {s['n_breaches']} over the threshold")
            out.append(f"    {s['description']}")
        if self.n_pairs_aligned:
            out.append(
                f"  identity of the aligned pairs: median={q.get('p50', 0.0):.3f} "
                f"p90={q.get('p90', 0.0):.3f} p99={q.get('p99', 0.0):.3f} "
                f"max={self.highest_identity:.3f} against a threshold of "
                f"{self.threshold}")
        if self.breaches:
            out.append(
                f"  BREACH: {self.n_breaching_sequences} of "
                f"{self.n_sequences} sequence(s) have a cross-fold neighbour "
                f"at or above {self.threshold} identity, in "
                f"{len(self.breaches)} pair(s). Every number in this report is "
                f"optimistic by an unmeasured amount.")
            for b in sorted(self.breaches, key=lambda x: -x.identity)[:10]:
                out.append(
                    f"    {b.example_id} (fold {b.fold}) ~ {b.neighbour_id} "
                    f"(fold {b.neighbour_fold}): identity {b.identity:.3f} "
                    f"[{', '.join(b.screens)}]")
            if len(self.breaches) > 10:
                out.append(f"    ... and {len(self.breaches) - 10} more")
        else:
            out.append(
                f"  no aligned cross-fold pair reached {self.threshold} "
                f"identity; {self.n_near_misses} came within "
                f"{self.near_miss_band:g} of it")
        out.append(f"  {self.caveat}")
        return out


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank quantile. No interpolation, so every value is observed."""
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1,
              max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[idx]


def audit_grouping(sequences: Mapping[str, str],
                   fold_of: Mapping[str, int],
                   *,
                   identity: Callable[[str, str], float],
                   threshold: float,
                   screens: Sequence[CandidateScreen],
                   identity_name: str,
                   clustering_identity_name: str | None = None,
                   grouping_was_complete: bool | None = None,
                   near_miss_band: float = 0.05,
                   ) -> GroupingAudit:
    """Align the screened cross-fold pairs and report every breach found.

    ``sequences`` maps example id to sequence and ``fold_of`` maps the same ids
    to a fold index. A *cross-fold* pair is one whose members are in different
    folds -- which is exactly the relation the grouping promises to keep below
    ``threshold``, since each sequence's training set in the fold where it is
    held out is everything in the other folds. Each screen nominates pairs; the
    union is aligned once; a pair at or above ``threshold`` is a breach.

    ``identity`` is an argument rather than an import because it must be the
    same function the clustering compared against its own threshold -- pass
    ``clustering_identity_name`` and the report checks that for you and warns
    when the two disagree. ``grouping_was_complete`` says whether the grouping
    aligned every pair of the pool; it changes nothing about the measurement
    and everything about what a clean result means, so it decides which
    :data:`SCREEN_CAVEAT` or :data:`COMPLETE_GROUPING_NOTE` is printed.
    ``screens`` are arguments for the same reason the
    alignment provider is: this module does not decide whether an optional
    extra is installed, and a caller that can afford more thoroughness should
    be able to buy it without editing the audit.

    Raises :class:`GroupingAuditError` for an id with no fold, an empty pool,
    no screens, a single fold (nothing is cross-fold, so there is no grouping
    claim to audit), or a screen that nominates a pair outside the pool or
    inside one fold -- a screen that reaches past the relation under audit
    would quietly change what the result means.
    """
    if not screens:
        raise GroupingAuditError(
            "no screens: aligning every cross-fold pair is the complete audit "
            "and is not what this function does, so a screen must be named")
    ids = sorted(sequences)
    if not ids:
        raise GroupingAuditError("no sequences to audit")
    missing = [i for i in ids if i not in fold_of]
    if missing:
        raise GroupingAuditError(f"no fold for {missing[:3]}...")
    folds = {fold_of[i] for i in ids}
    if len(folds) < 2:
        raise GroupingAuditError(
            f"{len(folds)} fold(s): nothing is cross-fold, so there is no "
            f"grouping claim to audit")

    cross = set(_cross_fold_pairs(sequences, fold_of))
    admitted: dict[Pair, list[str]] = {}
    per_screen: dict[str, set[Pair]] = {}
    for screen in screens:
        chosen = set(screen.select(sequences, fold_of))
        stray = chosen - cross
        if stray:
            a, b = sorted(stray)[0]
            raise GroupingAuditError(
                f"screen {screen.name!r} nominated {(a, b)}, which is not a "
                f"cross-fold pair of this pool; a screen may only narrow the "
                f"relation under audit, never widen it")
        per_screen[screen.name] = chosen
        for pair in chosen:
            admitted.setdefault(pair, []).append(screen.name)

    breaches: list[IdentityBreach] = []
    identities: list[float] = []
    near = 0
    for pair in sorted(admitted):
        a, b = pair
        ident = identity(sequences[a], sequences[b])
        identities.append(ident)
        if ident >= threshold:
            breaches.append(IdentityBreach(
                example_id=a, fold=fold_of[a], neighbour_id=b,
                neighbour_fold=fold_of[b], identity=ident,
                screens=tuple(admitted[pair])))
        elif ident >= threshold - near_miss_band:
            near += 1

    breached_pairs = {(b.example_id, b.neighbour_id) for b in breaches}
    screen_rows = [
        {"name": s.name, "description": s.description,
         "n_admitted": len(per_screen[s.name]),
         "n_breaches": len(per_screen[s.name] & breached_pairs)}
        for s in screens]
    identities.sort()
    quantiles = {name: round(_quantile(identities, q), 4)
                 for name, q in (("min", 0.0), ("p50", 0.5), ("p90", 0.9),
                                 ("p99", 0.99), ("max", 1.0))}
    return GroupingAudit(
        threshold=threshold, identity_name=identity_name,
        n_sequences=len(ids), n_folds=len(folds),
        n_cross_fold_pairs=len(cross), n_pairs_aligned=len(admitted),
        screens=screen_rows, identity_quantiles=quantiles,
        highest_identity=round(identities[-1], 4) if identities else 0.0,
        breaches=breaches,
        n_breaching_sequences=len({i for b in breaches
                                   for i in (b.example_id, b.neighbour_id)}),
        n_near_misses=near, near_miss_band=near_miss_band,
        clustering_identity_name=clustering_identity_name,
        grouping_was_complete=grouping_was_complete,
        caveat=(COMPLETE_GROUPING_NOTE if grouping_was_complete
                else SCREEN_CAVEAT))
