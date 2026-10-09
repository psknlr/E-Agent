"""Batch composition under a fixed budget: what 96 wells are actually spent on.

WHY A BATCH IS NOT "THE TOP 96"
-------------------------------
Taking the 96 best-ranked candidates maximises the expected number of hits only
if the ranking is well calibrated. It is not: most of its axes are model
outputs with unquantified error (see
:mod:`eagent.science.scorecard`). Worse, the ranking's errors are *correlated*
-- the same structure predictor, the same docking function and the same family
template act on every candidate -- so the top 96 are usually 96 close relatives
of whatever the models already understood, and a systematic error produces 96
simultaneous failures that teach nothing about where the error was.

A batch is therefore composed of three populations with different jobs:

``HIGH_EVIDENCE``
    Most likely to work. Pays for the round.
``DIVERSITY``
    Covers pocket space, so a failure of the whole batch localises to
    something other than "we only tried one clade".
``UNCERTAINTY_PROBE``
    Mechanistically sound candidates the models could not decide about. These
    are the only wells that can tell you the model was wrong, and they are the
    first thing cut by a pure top-k selection.

WHY POCKET DIVERSITY AND NOT SEQUENCE DIVERSITY
-----------------------------------------------
Global sequence identity is dominated by the scaffold. Two enzymes 45% identical
overall can have identical substrate pockets, and two 85% identical can differ
at the three positions that set substrate size. Spreading a batch on global
identity therefore buys much less coverage of the thing being varied than it
appears to. :func:`pocket_signature` builds the signature from the mapped
pocket, and :func:`pocket_distance` returns ``None`` rather than ``1.0`` when a
signature is unavailable -- a candidate nothing is known about must not win
every diversity slot by looking maximally different.

WHAT ``lambda_weight`` IS, AND WHY IT IS NOT THE THING
:func:`~eagent.science.scorecard.refuse_linear_blend` FORBIDS
-------------------------------------------------------------
:func:`greedy_submodular_select` maximises ``sum(utility) + lambda *
coverage``. That is a weighted combination, but of two *experiment-design*
quantities: how many slots go to the best-supported candidates, and how much of
the pocket space the batch covers. Its effect is a visible property of the
resulting batch, the operator sets it and can sweep it, and it never emerges as
a number attached to an enzyme. The forbidden blend is the one that fuses
incommensurable *measurements* of one candidate into a score that then gets
reported as that candidate's quality.

NO PADDING, ANYWHERE
--------------------
If fewer candidates qualify than the budget allows, :func:`compose_batch`
returns a short batch with ``shortfall_reason`` set. There is no code path that
lengthens a batch by relaxing a standard: the only way to add members is to add
qualifying candidates. A short batch with a stated reason is a result; a full
batch of padding is a fabrication with 96 wells of evidence behind it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..schemas import (
    BatchMember,
    BatchPlan,
    BatchRole,
    Budget,
    Candidate,
    ConfidenceLevel,
    ControlItem,
)
from .numbering import letter_of_residue_token
from .pocket import PocketResidues, pocket_residues_from_tokens
from .scorecard import (
    DEFAULT_LEXICOGRAPHIC_ORDER,
    lexicographic_rank,
)

__all__ = [
    "DEFAULT_KMER_SIZE",
    "DEFAULT_ROLE_TARGETS",
    "DEFAULT_LAMBDA_WEIGHT",
    "SignatureBasis",
    "PocketSignature",
    "ControlReservation",
    "kmer_set",
    "jaccard_distance",
    "hamming_distance",
    "sequence_distance",
    "pocket_signature",
    "PocketResidues",
    "pocket_distance",
    "candidate_distance",
    "facility_location_gain",
    "greedy_submodular_select",
    "rank_utility",
    "family_quota_select",
    "reserve_control_slots",
    "is_model_uncertain",
    "compose_batch",
]


# ==========================================================================
# Named defaults. All of these are experiment-design knobs, not thresholds
# on any measured catalytic quantity.
# ==========================================================================

#: k-mer length for :func:`kmer_set`. Three is the usual compromise between
#: alignment-free sensitivity and k-mer collisions in protein alphabets.
#: CALIBRATION: raise it for long, highly similar families where 3-mers
#: saturate; this changes distances and therefore batch composition.
DEFAULT_KMER_SIZE: int = 3

#: Starting split of a 96-construct round across the three candidate roles.
#:
#: 48 / 24 / 24 is **a starting configuration, not a validated optimum.** No
#: experiment in this repository shows it beats 64/16/16 or 32/32/32. It
#: encodes one judgement -- that half the round should pay for itself and the
#: other half should buy information -- and it is passed in as config precisely
#: so a reviewer can argue with it and a second round can revise it against
#: round-1 outcomes.
DEFAULT_ROLE_TARGETS: dict[BatchRole, int] = {
    BatchRole.HIGH_EVIDENCE: 48,
    BatchRole.DIVERSITY: 24,
    BatchRole.UNCERTAINTY_PROBE: 24,
}

#: Default trade-off between per-candidate utility and batch coverage in
#: :func:`greedy_submodular_select`. Utility is normalised to ``(0, 1]`` by
#: :func:`rank_utility` and the coverage gain of one addition is also bounded by
#: the pool size, so 1.0 means "a slot's rank advantage and the coverage it adds
#: count comparably". CALIBRATION: sweep it and look at the resulting batch;
#: there is no correct value, only a stated one.
DEFAULT_LAMBDA_WEIGHT: float = 1.0

#: Levels that mark a candidate as one the models could not decide about, and
#: therefore eligible for an uncertainty probe. ``CONTRADICTORY`` is included
#: because disagreeing independent routes are the clearest possible signal that
#: an experiment would be informative; a confident model negative is *not*
#: included, because a probe samples blind spots, not refutations.
_UNCERTAIN_LEVELS: frozenset[ConfidenceLevel] = frozenset({
    ConfidenceLevel.INSUFFICIENT,
    ConfidenceLevel.WEAK,
    ConfidenceLevel.CONTRADICTORY,
})


def _default_id_of(item: Any) -> str:
    """Stable identity for tie-breaking, so selection is reproducible."""
    for attr in ("candidate_id", "id"):
        value = getattr(item, attr, None)
        if value is not None:
            return str(value)
    return str(item)


# ==========================================================================
# Distances
# ==========================================================================

def kmer_set(sequence: str, k: int = DEFAULT_KMER_SIZE) -> frozenset[str]:
    """Set of k-mers in a sequence, for alignment-free comparison.

    A set, not a multiset: repeat-rich regions would otherwise dominate the
    distance, which would make low-complexity linkers look like biology.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    seq = "".join(str(sequence).split()).upper()
    if len(seq) < k:
        return frozenset()
    return frozenset(seq[i:i + k] for i in range(len(seq) - k + 1))


def jaccard_distance(a: Iterable[Any], b: Iterable[Any]) -> float | None:
    """``1 - |A n B| / |A u B|``, or ``None`` when either set is empty.

    ``None`` rather than ``1.0`` for an empty set. An empty signature means
    "nothing was characterised", and scoring that as maximal distance would let
    the least-characterised candidate win every diversity slot -- the exact
    opposite of what diversity selection is for.
    """
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return None
    union = sa | sb
    return 1.0 - len(sa & sb) / len(union)


def hamming_distance(a: str, b: str, *, normalise: bool = True) -> float:
    """Positionwise mismatch count over two **equal-length** sequences.

    Raises on unequal lengths instead of truncating or padding. A Hamming
    distance between sequences of different length is not a weaker measurement;
    it is a measurement of the wrong thing, because every position after the
    first indel is compared against an unrelated residue. Use
    :func:`sequence_distance` with ``method='kmer_jaccard'`` for unaligned
    sequences, or align them first.
    """
    sa = "".join(str(a).split()).upper()
    sb = "".join(str(b).split()).upper()
    if len(sa) != len(sb):
        raise ValueError(
            f"hamming_distance needs aligned, equal-length sequences; got "
            f"{len(sa)} and {len(sb)}. Align them or use kmer_jaccard."
        )
    if not sa:
        raise ValueError("hamming_distance of empty sequences is undefined")
    mismatches = sum(1 for x, y in zip(sa, sb) if x != y)
    return mismatches / len(sa) if normalise else float(mismatches)


def _sequence_of(obj: Candidate | str) -> str:
    return obj if isinstance(obj, str) else obj.sequence


def sequence_distance(
    a: Candidate | str,
    b: Candidate | str,
    *,
    method: str = "kmer_jaccard",
    k: int = DEFAULT_KMER_SIZE,
) -> float:
    """Global sequence distance in ``[0, 1]``, by k-mer Jaccard or by Hamming.

    Global distance is the *fallback*, never the preferred basis for diversity:
    it is dominated by scaffold variation, which is not the axis a substrate-
    directed campaign is trying to cover. See :func:`pocket_distance`.

    ``kmer_jaccard`` is the default because it needs no alignment and so cannot
    silently compare residue 200 of one protein against residue 200 of another
    that has an extra loop.
    """
    sa, sb = _sequence_of(a), _sequence_of(b)
    if method == "hamming":
        return hamming_distance(sa, sb)
    if method != "kmer_jaccard":
        raise ValueError(
            f"unknown method {method!r}; use 'kmer_jaccard' or 'hamming'"
        )
    dist = jaccard_distance(kmer_set(sa, k), kmer_set(sb, k))
    if dist is None:
        raise ValueError(
            f"sequence shorter than k={k}; cannot form a k-mer set for comparison"
        )
    return dist


class SignatureBasis(str, enum.Enum):
    """What a pocket signature was actually built from.

    Carried alongside the tokens because the three bases have very different
    authority and a selector that cannot tell them apart will report pocket
    coverage it did not achieve.
    """

    POCKET_RESIDUES = "pocket_residues"
    #: Pocket residues placed in a shared numbering frame. The only basis on
    #: which two candidates' positions mean the same thing.
    ALIGNED_POCKET_RESIDUES = "aligned_pocket_residues"
    #: Which residue types line the pocket and how many of each, with no
    #: positions. What is left when there is no shared frame, and weaker:
    #: two pockets with the same residues in a different arrangement look
    #: identical. Reporting it as positional would be the stronger lie.
    POCKET_COMPOSITION = "pocket_composition"
    CATALYTIC_ROLES_ONLY = "catalytic_roles_only"
    EMPTY = "empty"


def _multiset_tokens(prefix: str, items: Iterable[str]) -> set[str]:
    """Encode a multiset as a set, so Jaccard counts multiplicity.

    Three tyrosines and two tyrosines must not look like one shared "Y". The
    k-th copy of an item becomes its own token, which makes the intersection
    of the encoded sets the sum of the per-item minima and the union the sum
    of the maxima -- i.e. the multiset Jaccard, computed by the ordinary set
    one.
    """
    counts: dict[str, int] = {}
    out: set[str] = set()
    for item in items:
        counts[item] = counts.get(item, 0) + 1
        out.add(f"{prefix}{item}#{counts[item]}")
    return out


@dataclass(frozen=True)
class PocketSignature:
    """Sorted residue tokens describing one candidate's substrate pocket.

    ``frame`` is what makes two signatures comparable. Positions are numbered
    by whoever deposited the structure, so ``F98`` in one protein and ``F99``
    in another say nothing about each other until both have been mapped into
    a common numbering. A signature in no frame carries composition tokens
    instead, and :func:`pocket_distance` will not compare it positionally
    with anything.
    """

    candidate_id: str
    tokens: tuple[str, ...]
    basis: SignatureBasis
    frame: str = ""
    #: Pocket residues that could not be placed in the frame. A pocket
    #: compared on four of its seven residues is a different comparison from
    #: one compared on all seven.
    n_unplaced: int = 0

    def __bool__(self) -> bool:
        return bool(self.tokens)

    @property
    def is_positional(self) -> bool:
        return self.basis in (SignatureBasis.ALIGNED_POCKET_RESIDUES,
                              SignatureBasis.POCKET_RESIDUES)

    def comparable_with(self, other: "PocketSignature") -> bool:
        """Whether a distance between these two signatures means anything.

        Same frame, or neither in one. A composition signature and a
        positional signature are answers to different questions and their
        Jaccard distance is a number with no interpretation.
        """
        if not self or not other:
            return False
        if self.basis is not other.basis:
            return False
        if self.is_positional:
            return self.frame == other.frame and bool(self.frame)
        return True


def pocket_signature(
    candidate: Candidate,
    *,
    extra_pocket_residues: Mapping[str, Sequence[str] | PocketResidues] | None = None,
    include_substitutions: bool = True,
) -> PocketSignature:
    """The residue set that defines this candidate's pocket, with its basis.

    WHY A SIGNATURE AND NOT THE WHOLE SEQUENCE
    ------------------------------------------
    Diversity in this campaign means diversity of the *substrate pocket*. Two
    candidates differing at 300 scaffold positions and at none of the pocket
    positions will behave identically toward the substrate, and spending two
    slots on them buys one experiment.

    WHY A ROLE TOKEN CARRIES NO NUMBER
    ----------------------------------
    ``role_to_residue`` records the catalytic tyrosine as ``Y155`` in author
    numbering. Two members of one family whose numbering differs by an
    N-terminal methionine report ``Y155`` and ``Y156`` for the *same*
    conserved residue, so a token built from the number made two identical
    catalytic machineries look completely different -- a Jaccard distance of
    1.0, which a diversity-maximising selector reads as the most valuable
    pair on the list. The role label is already the family-level position, so
    the token is the role and the amino acid, and nothing that depends on who
    deposited the structure.

    WHY POCKET RESIDUES NEED A FRAME
    --------------------------------
    The same problem, without a role label to fall back on. A
    :class:`~eagent.science.pocket.PocketResidues` carries the numbering frame
    its tokens are in; two of them are compared positionally only when the
    frames match. A bare sequence of tokens -- which is all the older callers
    pass -- is in no shared frame, so the signature falls back to pocket
    *composition* and says so in its basis. That understates differences
    between pockets that differ only in arrangement, and it is the weaker of
    the two errors available.

    WHY THE CATALYTIC FALLBACK IS FLAGGED RATHER THAN HIDDEN
    --------------------------------------------------------
    The schema guarantees only the *catalytic* role mapping, and catalytic
    roles are by definition the conserved positions of a family -- so a
    signature built from them alone is nearly constant within a family and
    badly under-reports pocket differences. When that is all there is, the
    basis is ``CATALYTIC_ROLES_ONLY`` and :func:`compose_batch` records it, so
    nobody reads "pocket coverage" off a signature that could not see the
    pocket. Pass ``extra_pocket_residues`` to get a real signature.

    An unmappable candidate returns an empty signature, which
    :func:`pocket_distance` treats as incomparable rather than as maximally
    distant.
    """
    tokens: set[str] = set()
    mapping = candidate.catalytic_mapping
    roles = dict(mapping.role_to_residue)
    if include_substitutions:
        roles.update(mapping.substituted_roles)
    for role, residue in roles.items():
        tokens.add(f"role:{role}={_residue_letter(residue)}")

    supplied = (extra_pocket_residues or {}).get(candidate.candidate_id)
    shell = _as_pocket_residues(candidate.candidate_id, supplied)
    if shell is not None and shell.tokens:
        if shell.positional:
            tokens.update(f"pocket:{t}" for t in shell.tokens)
            basis = SignatureBasis.ALIGNED_POCKET_RESIDUES
        else:
            tokens.update(_multiset_tokens("pocket_aa:", shell.letters))
            basis = SignatureBasis.POCKET_COMPOSITION
        return PocketSignature(candidate.candidate_id, tuple(sorted(tokens)),
                               basis, frame=shell.frame,
                               n_unplaced=shell.n_lost)
    if tokens:
        basis = SignatureBasis.CATALYTIC_ROLES_ONLY
    else:
        basis = SignatureBasis.EMPTY
    return PocketSignature(candidate.candidate_id, tuple(sorted(tokens)), basis)


def _residue_letter(token: str) -> str:
    """The amino acid in a token like ``Y155``, without the number.

    The number is author numbering and means nothing across proteins; the
    letter is the chemistry. A token this does not recognise comes back
    unchanged rather than being reshaped into something that looks canonical:
    a signature built from an unparseable token is still a signature, and
    quietly turning it into a letter would compare two things that are not
    residues as though they were.
    """
    return letter_of_residue_token(token) or str(token or "").strip()


def _as_pocket_residues(
    candidate_id: str, supplied: Sequence[str] | PocketResidues | None,
) -> PocketResidues | None:
    if supplied is None:
        return None
    if isinstance(supplied, PocketResidues):
        return supplied
    # A bare list of tokens states no frame, and assuming one is the bug.
    return pocket_residues_from_tokens(candidate_id, list(supplied))


def pocket_distance(
    a: Candidate,
    b: Candidate,
    *,
    extra_pocket_residues: Mapping[str, Sequence[str] | PocketResidues] | None = None,
) -> float | None:
    """Jaccard distance between two pocket signatures, or ``None`` if unavailable.

    ``None`` propagates the honesty of :func:`jaccard_distance`: a pocket that
    was never mapped is unknown, not different. It is also what comes back
    when the two signatures are not comparable -- one aligned into a family
    frame and one in no frame, or two in different frames. Their Jaccard
    distance is computable and means nothing, and returning it would let a
    selector rank on it.
    """
    sa = pocket_signature(a, extra_pocket_residues=extra_pocket_residues)
    sb = pocket_signature(b, extra_pocket_residues=extra_pocket_residues)
    if not sa.comparable_with(sb):
        return None
    return jaccard_distance(sa.tokens, sb.tokens)


def candidate_distance(
    a: Candidate,
    b: Candidate,
    *,
    extra_pocket_residues: Mapping[str, Sequence[str]] | None = None,
    k: int = DEFAULT_KMER_SIZE,
    require_pocket: bool = False,
) -> float:
    """Pocket distance where it exists, global sequence distance otherwise.

    The fallback **over-reports** diversity for pocket purposes, because global
    sequence spread counts scaffold differences that the substrate never sees.
    It is used anyway so that an unmapped candidate is not silently excluded
    from diversity selection, and :func:`compose_batch` reports how many
    comparisons fell back. Pass ``require_pocket=True`` to turn the fallback
    into an error where a run cannot tolerate it.
    """
    dist = pocket_distance(a, b, extra_pocket_residues=extra_pocket_residues)
    if dist is not None:
        return dist
    if require_pocket:
        raise ValueError(
            f"no pocket signature for {a.candidate_id} and/or {b.candidate_id}; "
            f"map the pocket or allow the documented sequence fallback"
        )
    return sequence_distance(a, b, k=k)


# ==========================================================================
# Submodular selection
# ==========================================================================

def facility_location_gain(
    item: Any,
    pool: Sequence[Any],
    coverage: Mapping[str, float],
    similarity: Callable[[Any, Any], float],
    *,
    id_of: Callable[[Any], str] = _default_id_of,
) -> float:
    """Marginal facility-location gain of adding ``item`` to the selected set.

    Facility location is ``F(S) = sum over the pool of max similarity to S``:
    every candidate in the pool is "represented" by its nearest selected
    neighbour, and the gain of adding one member is how much better it
    represents the pool than the current selection does.

    WHY THIS OBJECTIVE
    ------------------
    It is monotone and submodular, so greedy selection has the standard
    ``1 - 1/e`` guarantee and, more usefully here, it has the right *shape*: it
    rewards covering an unrepresented region and gives almost nothing for a
    second near-duplicate of something already chosen. A plain
    "sum of pairwise distances" objective does not -- it can be maximised by two
    tight clusters at opposite ends of the space, which is the failure mode this
    module exists to avoid.

    ``coverage`` maps a pool member's id to its current best similarity to the
    selected set (0.0 when nothing is selected yet).
    """
    gain = 0.0
    for other in pool:
        current = coverage.get(id_of(other), 0.0)
        sim = similarity(other, item)
        if sim > current:
            gain += sim - current
    return gain


def greedy_submodular_select(
    items: Sequence[Any],
    k: int,
    utility: Callable[[Any], float] | Mapping[str, float],
    distance: Callable[[Any, Any], float],
    lambda_weight: float = DEFAULT_LAMBDA_WEIGHT,
    *,
    id_of: Callable[[Any], str] = _default_id_of,
    accept: Callable[[Any], bool] | None = None,
    already_selected: Sequence[Any] = (),
) -> list[Any]:
    """Greedily maximise ``sum(utility) + lambda_weight * facility_location_coverage``.

    Pure Python and deterministic: candidates are considered in a fixed order
    and every tie -- which is common, because utilities derived from ordinal
    ranks collide constantly -- is broken by the smallest id. Without that
    tie-break, two runs over the same data would produce different batches and
    the manifest would not reproduce the plan.

    ``lambda_weight`` is an experiment-design knob, not a scientific weight; see
    the module docstring and
    :func:`~eagent.science.scorecard.refuse_linear_blend`.

    Returns at most ``k`` items. If the pool is smaller than ``k`` it returns the
    whole pool: there is no mechanism here for inventing members, and the
    shortfall is reported by the caller.

    ``accept`` is an optional admission check applied to each greedy pick. It
    may have side effects (a quota ledger is the intended use); returning
    ``False`` drops that item and the loop continues with the next best. The
    check runs *inside* the loop rather than as a filter afterwards because the
    caps it enforces change as the batch fills -- applying them after selection
    would silently shorten the batch while a qualifying alternative was still
    available.

    ``already_selected`` seeds the coverage state with items chosen outside
    this call. It matters whenever a batch is filled in stages: a diversity
    stage that starts from "nothing covered" can spend its slot on a near
    duplicate of a candidate the previous stage already put in the batch, and
    the slot then buys no coverage at all while the plan records it as having
    been spent on diversity. Coverage is a property of the batch, not of the
    call.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if lambda_weight < 0:
        raise ValueError(
            f"lambda_weight must be >= 0, got {lambda_weight}; a negative "
            f"coverage weight would actively select near-duplicates"
        )
    pool = list(items)
    if k == 0 or not pool:
        return []

    util: Callable[[Any], float]
    if isinstance(utility, Mapping):
        table = dict(utility)
        util = lambda it: float(table.get(id_of(it), 0.0))  # noqa: E731
    else:
        util = lambda it: float(utility(it))  # noqa: E731

    selected: list[Any] = []
    remaining = sorted(pool, key=id_of)
    coverage: dict[str, float] = {id_of(it): 0.0 for it in pool}
    for chosen in already_selected:
        for other in pool:
            oid = id_of(other)
            sim = 1.0 - distance(other, chosen)
            if sim > coverage[oid]:
                coverage[oid] = sim

    while remaining and len(selected) < k:
        best_item: Any = None
        best_score: tuple[float, str] | None = None
        for item in remaining:
            gain = facility_location_gain(
                item, pool, coverage,
                lambda x, y: 1.0 - distance(x, y),
                id_of=id_of,
            ) if lambda_weight > 0 else 0.0
            score = util(item) + lambda_weight * gain
            # Deterministic argmax: higher score wins, then smallest id.
            key = (-score, id_of(item))
            if best_score is None or key < best_score:
                best_score = key
                best_item = item
        assert best_item is not None
        remaining = [it for it in remaining if id_of(it) != id_of(best_item)]
        if accept is not None and not accept(best_item):
            continue
        selected.append(best_item)
        for other in pool:
            oid = id_of(other)
            sim = 1.0 - distance(other, best_item)
            if sim > coverage[oid]:
                coverage[oid] = sim
    return selected


def rank_utility(
    ordered: Sequence[Any], *, id_of: Callable[[Any], str] = _default_id_of
) -> dict[str, float]:
    """Utilities in ``(0, 1]`` derived from a rank position, best first.

    A *rank*, deliberately, not a score. The input ordering already encodes
    everything this module is entitled to say about relative quality (it comes
    from :func:`~eagent.science.scorecard.lexicographic_rank`, which makes no
    commensurability claim). Converting a position to a utility adds no new
    scientific content -- it only gives the selector something monotone to trade
    against coverage -- and it cannot be mistaken for a measured quantity the
    way a blended score can.
    """
    n = len(ordered)
    if n == 0:
        return {}
    return {id_of(item): (n - i) / n for i, item in enumerate(ordered)}


# ==========================================================================
# Quotas
# ==========================================================================

def _family_of(candidate: Candidate) -> str:
    return candidate.family.family_name or "unassigned"


def _cluster_of(candidate: Candidate) -> str:
    return (candidate.family.sequence_cluster_id
            or candidate.family.phylogenetic_clade
            or f"singleton:{candidate.candidate_id}")


class _QuotaLedger:
    """Running per-family and per-cluster counts for one batch.

    A class rather than two dicts because the counts must be shared across the
    three role fills: a clade that already took its allowance in the
    high-evidence pass must not take it again under a diversity label. Enforcing
    quotas per role instead of per batch is the quiet way a batch collapses into
    one clade while every individual step looks compliant.
    """

    def __init__(
        self,
        quotas: Mapping[str, int] | None,
        cluster_cap: int | None,
        family_of: Callable[[Candidate], str],
        cluster_of: Callable[[Candidate], str],
    ) -> None:
        self.quotas = dict(quotas or {})
        self.cluster_cap = cluster_cap
        self.family_of = family_of
        self.cluster_of = cluster_of
        self.family_counts: dict[str, int] = {}
        self.cluster_counts: dict[str, int] = {}
        self.rejected_family: list[str] = []
        self.rejected_cluster: list[str] = []

    def cap_for(self, family: str) -> int | None:
        if family in self.quotas:
            return self.quotas[family]
        if "*" in self.quotas:
            return self.quotas["*"]
        return None

    def admits(self, candidate: Candidate) -> bool:
        family = self.family_of(candidate)
        cap = self.cap_for(family)
        if cap is not None and self.family_counts.get(family, 0) >= cap:
            self.rejected_family.append(candidate.candidate_id)
            return False
        if self.cluster_cap is not None:
            cluster = self.cluster_of(candidate)
            if self.cluster_counts.get(cluster, 0) >= self.cluster_cap:
                self.rejected_cluster.append(candidate.candidate_id)
                return False
        return True

    def take(self, candidate: Candidate) -> None:
        family = self.family_of(candidate)
        self.family_counts[family] = self.family_counts.get(family, 0) + 1
        cluster = self.cluster_of(candidate)
        self.cluster_counts[cluster] = self.cluster_counts.get(cluster, 0) + 1


def family_quota_select(
    candidates: Sequence[Candidate],
    quotas: Mapping[str, int] | None,
    k: int,
    *,
    cluster_cap: int | None = None,
    family_of: Callable[[Candidate], str] = _family_of,
    cluster_of: Callable[[Candidate], str] = _cluster_of,
) -> list[Candidate]:
    """Take up to ``k`` candidates in the given order, honouring family and clade caps.

    WHY CAPS AT ALL
    ---------------
    Ranking and diversity selection both have a bias toward whatever is
    over-represented in the database: a family with 400 deposited homologues
    produces more high-scoring candidates than one with 6, for reasons that are
    about sequencing effort rather than chemistry. Without a cap, a batch
    becomes a replicate experiment on one clade with 96 slightly different
    constructs, and a negative result cannot distinguish "this chemistry does
    not work" from "this clade does not work".

    ``quotas`` maps a family name to its maximum count; the key ``"*"`` sets a
    default for families not named. ``cluster_cap`` caps each sequence cluster
    or clade, which is the finer control -- one family can still be a single
    clade.

    The input order is respected (pass a ranked list), and nothing is
    substituted when a cap blocks a candidate: the batch simply gets fewer
    members, which the caller reports as a shortfall.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    ledger = _QuotaLedger(quotas, cluster_cap, family_of, cluster_of)
    out: list[Candidate] = []
    for cand in candidates:
        if len(out) >= k:
            break
        if ledger.admits(cand):
            ledger.take(cand)
            out.append(cand)
    return out


# ==========================================================================
# Controls
# ==========================================================================

@dataclass(frozen=True)
class ControlReservation:
    """Result of reserving construct slots for controls, before candidates are chosen."""

    budget: Budget
    controls: tuple[ControlItem, ...]
    reserved_slots: int
    candidate_slots: int
    note: str = ""


def reserve_control_slots(
    budget: Budget, controls: Sequence[ControlItem]
) -> ControlReservation:
    """Subtract controls that need new genes from the construct cap, up front.

    WHY UP FRONT
    ------------
    When ``Budget.constructs_include_controls`` is true, the construct count is a
    hard synthesis cap and a positive-control enzyme is a gene like any other.
    Choosing 96 candidates first and *then* discovering that two controls also
    need synthesising leaves two options: order 98 constructs (the cap was not a
    cap) or drop two chosen candidates (the selection was not the selection).
    Reserving first makes the cap real and the batch honest about its size.

    This also normalises every control with ``requires_new_construct`` to
    ``occupies_batch_slot=True``, which
    :class:`~eagent.schemas.batch.BatchPlan` requires -- a control that needs a
    gene but claims no slot is a budget that does not add up.

    When ``constructs_include_controls`` is false, nothing is reserved: the
    control genes are declared to come from a separate budget line. That is a
    claim about the operator's synthesis arrangement, and the note records it so
    a reviewer can check it rather than discover it at ordering time.
    """
    normalised: list[ControlItem] = []
    n_new = 0
    for control in controls:
        if control.requires_new_construct:
            n_new += 1
            if not control.occupies_batch_slot:
                control = control.model_copy(update={"occupies_batch_slot": True})
        normalised.append(control)

    if not budget.constructs_include_controls:
        return ControlReservation(
            budget=budget,
            controls=tuple(normalised),
            reserved_slots=0,
            candidate_slots=budget.candidate_slots,
            note=(f"{n_new} control(s) need new constructs but "
                  f"constructs_include_controls is false, so they are declared to "
                  f"come from a separate synthesis budget line and no candidate "
                  f"slot was reserved"),
        )

    if n_new > budget.new_constructs_round_1:
        raise ValueError(
            f"{n_new} controls need new constructs but the round allows only "
            f"{budget.new_constructs_round_1}; the control plan does not fit the "
            f"synthesis budget and must be cut deliberately, not silently"
        )
    adjusted = Budget(**{**budget.model_dump(), "reserved_control_slots": n_new})
    return ControlReservation(
        budget=adjusted,
        controls=tuple(normalised),
        reserved_slots=n_new,
        candidate_slots=adjusted.candidate_slots,
        note=(f"{n_new} construct slot(s) reserved for controls out of "
              f"{budget.new_constructs_round_1}; "
              f"{adjusted.candidate_slots} remain for candidates"),
    )


# ==========================================================================
# Batch composition
# ==========================================================================

def is_model_uncertain(candidate: Candidate) -> bool:
    """Is this a candidate the models could not decide about?

    True when the ``model_uncertainty`` axis is INSUFFICIENT, WEAK or
    CONTRADICTORY. Deliberately **not** true for a candidate the models
    confidently rejected: an uncertainty probe is for sampling the region where
    the models have nothing to say, so that a surprise is interpretable. Testing
    a confident model negative is a different and also valuable experiment, but
    it is a falsification run and should be labelled as one rather than smuggled
    in under a probe quota.
    """
    dim = candidate.dimension("model_uncertainty")
    if dim is None:
        return False
    return dim.level in _UNCERTAIN_LEVELS


def _scale_role_targets(
    role_targets: Mapping[BatchRole, int], slots: int
) -> dict[BatchRole, int]:
    """Rescale role targets to the slots actually available, by largest remainder.

    Needed because control reservation shrinks the candidate budget after the
    role split was written down. Scaling proportionally keeps the *stated
    intent* of the split (half paying, half informative) instead of silently
    taking every reserved slot out of the probe quota, which is what happens
    when the last role simply absorbs the difference.
    """
    total = sum(max(0, v) for v in role_targets.values())
    if slots <= 0:
        return {role: 0 for role in role_targets}
    if total == 0:
        return {role: 0 for role in role_targets}
    if total == slots:
        return {role: max(0, v) for role, v in role_targets.items()}
    exact = {role: max(0, v) * slots / total for role, v in role_targets.items()}
    floors = {role: int(value) for role, value in exact.items()}
    shortfall = slots - sum(floors.values())
    # Deterministic largest-remainder: biggest fractional part first, then by
    # role name, so two identical inputs always give the same split.
    order = sorted(exact, key=lambda r: (-(exact[r] - floors[r]), r.value))
    for role in order[:max(0, shortfall)]:
        floors[role] += 1
    return floors


def compose_batch(
    candidates: Sequence[Candidate],
    budget: Budget,
    role_targets: Mapping[BatchRole, int] | None = None,
    quotas: Mapping[str, int] | None = None,
    *,
    controls: Sequence[ControlItem] = (),
    plan_id: str = "batch-round-1",
    round_number: int = 1,
    cofactor_conditions: int = 2,
    replicates: int = 3,
    cluster_cap: int | None = None,
    extra_pocket_residues: Mapping[str, Sequence[str]] | None = None,
    order_of_dimensions: Sequence[str] | None = None,
    lambda_weight: float = DEFAULT_LAMBDA_WEIGHT,
    kmer_size: int = DEFAULT_KMER_SIZE,
    reallocate_unfilled_roles: bool = False,
    rank_tiebreak: Callable[[Candidate], float | None] | None = None,
    high_evidence_eligible: Callable[[Candidate], bool] | None = None,
) -> BatchPlan:
    """Fill a round's construct slots with three labelled populations, or report short.

    PIPELINE
    --------
    1. :func:`reserve_control_slots` takes control genes out of the construct cap
       first, so the cap is real.
    2. The role split is rescaled to the remaining slots
       (:func:`_scale_role_targets`).
    3. Only candidates that **pass every feasibility gate** enter any pool.
       Candidates with an undecided gate are excluded and counted separately,
       because "we did not check" is not a reason to spend a construct and is
       also not a rejection -- the shortfall reason names them so the cheapest
       way to lengthen the batch is visible.
    4. ``HIGH_EVIDENCE`` is filled from the lexicographic rank under family and
       clade quotas.
    5. ``UNCERTAINTY_PROBE`` is filled next, from the gate-passing candidates
       the models could not decide about (:func:`is_model_uncertain`), spread by
       :func:`greedy_submodular_select`. It is filled before diversity because
       its pool is strictly the smaller one and would otherwise be eaten.
    6. ``DIVERSITY`` takes the remainder, maximising pocket coverage.

    Quotas are enforced across the **whole** batch, not per role.

    ``high_evidence_eligible`` -- when given, a candidate for which it returns
    ``False`` is skipped for the HIGH_EVIDENCE role only (see
    :func:`eagent.tools.open_branches.is_high_evidence_eligible`: a de novo
    design is a hypothesis, and a slot labelled "high evidence" would say
    otherwise). It stays eligible for the other two roles.

    NO PADDING
    ----------
    If the qualifying pools run out, the plan comes back short with
    ``shortfall_reason`` naming the exact arithmetic. There is no branch in this
    function that admits a candidate failing a gate, exceeding a quota, or
    lacking a scorecard. The only way to make the batch longer is to make more
    candidates qualify.

    Parameters
    ----------
    role_targets:
        Defaults to :data:`DEFAULT_ROLE_TARGETS` (48/24/24 of 96) -- a starting
        configuration, not a validated optimum.
    reallocate_unfilled_roles:
        Default ``False``. When one role's pool runs dry -- typically the
        uncertainty probes, when every gate-passing candidate happened to get a
        confident model verdict -- its slots are left **unfilled** and named in
        ``shortfall_reason``, rather than quietly becoming more high-evidence
        wells. Changing the composition of a round changes what the round can
        conclude, so it is an operator decision, not a default. Setting this
        ``True`` hands the leftover slots to the diversity role; those
        candidates still pass every gate, so no standard is relaxed, but the
        plan records that the split it ran was not the split that was asked for.
    """
    targets = dict(role_targets or DEFAULT_ROLE_TARGETS)
    for role in targets:
        if role is BatchRole.CONTROL:
            raise ValueError(
                "controls are reserved from the construct budget, not allocated as "
                "a candidate role; pass them in 'controls'"
            )

    reservation = reserve_control_slots(budget, controls)
    slots = reservation.candidate_slots
    scaled = _scale_role_targets(targets, slots)

    ungated = [c for c in candidates if not c.gates()]
    if ungated:
        raise ValueError(
            f"{len(ungated)} candidate(s) have no feasibility gates on their "
            f"scorecard (first: {ungated[0].candidate_id}). Run "
            f"scorecard.build_scorecard first; composing a batch from ungated "
            f"candidates would treat 'never checked' as 'eligible'."
        )

    eligible: list[Candidate] = []
    undecided: list[Candidate] = []
    rejected: list[Candidate] = []
    for cand in candidates:
        if cand.passes_gates:
            eligible.append(cand)
        elif (cand.has_unresolved_gate and not cand.input_errors
              and not cand.disqualified
              and not any(g.gate_passed is False for g in cand.gates())):
            undecided.append(cand)
        else:
            rejected.append(cand)

    def distance(a: Candidate, b: Candidate) -> float:
        return candidate_distance(
            a, b, extra_pocket_residues=extra_pocket_residues, k=kmer_size
        )

    ranked = lexicographic_rank(eligible, order_of_dimensions
                                or DEFAULT_LEXICOGRAPHIC_ORDER,
                                tiebreak=rank_tiebreak)
    utilities = rank_utility(ranked)
    ledger = _QuotaLedger(quotas, cluster_cap, _family_of, _cluster_of)

    members: list[BatchMember] = []
    taken: set[str] = set()
    #: The candidates already in the batch, in admission order. Passed to each
    #: later selection stage as its coverage baseline: a diversity slot spent
    #: on a near duplicate of a high-evidence member buys nothing, and the
    #: plan would still record it as having been spent on diversity.
    chosen: list[Candidate] = []

    def admit(cand: Candidate, role: BatchRole, reason: str) -> bool:
        if cand.candidate_id in taken:
            return False
        if not ledger.admits(cand):
            return False
        ledger.take(cand)
        taken.add(cand.candidate_id)
        chosen.append(cand)
        members.append(BatchMember(
            slot=len(members) + 1,
            candidate_id=cand.candidate_id,
            role=role,
            family=_family_of(cand),
            sequence_cluster_id=cand.family.sequence_cluster_id,
            utility=utilities.get(cand.candidate_id),
            selection_reason=reason,
        ))
        return True

    # -- 1. high evidence ------------------------------------------------
    n_high = scaled.get(BatchRole.HIGH_EVIDENCE, 0)
    n_taken_high = 0
    for cand in ranked:
        if n_taken_high >= n_high:
            break
        if high_evidence_eligible is not None and not high_evidence_eligible(cand):
            # Not a gate and not a rejection: the candidate stays eligible for
            # the probe and diversity roles. It is only not *evidence*, so it
            # may not occupy a slot whose label says it is.
            continue
        if admit(cand, BatchRole.HIGH_EVIDENCE,
                 "top of the within-batch lexicographic rank under the stated "
                 "dimension priority; gates all passed"
                 + ("; candidates equal on every dimension were ordered by the "
                    "supplied tie-break (a model value, consulted only between "
                    "ties)" if rank_tiebreak is not None else "")):
            n_taken_high += 1

    # -- 2. uncertainty probes -------------------------------------------
    n_probe = scaled.get(BatchRole.UNCERTAINTY_PROBE, 0)
    probe_pool = [c for c in ranked
                  if c.candidate_id not in taken and is_model_uncertain(c)]

    def accept_probe(cand: Candidate) -> bool:
        level = cand.dimension("model_uncertainty")
        return admit(
            cand, BatchRole.UNCERTAINTY_PROBE,
            "passes every mechanism feasibility gate but the models could not "
            f"decide (model_uncertainty is {level.level.value if level else 'absent'}); "
            "spent to probe a model blind spot, not to test a guess",
        )

    greedy_submodular_select(probe_pool, n_probe, utilities, distance,
                             lambda_weight, accept=accept_probe,
                             already_selected=list(chosen))

    # -- 3. diversity ----------------------------------------------------
    n_div = scaled.get(BatchRole.DIVERSITY, 0)
    div_pool = [c for c in ranked if c.candidate_id not in taken]

    def accept_diversity(cand: Candidate) -> bool:
        return admit(
            cand, BatchRole.DIVERSITY,
            "selected to maximise pocket coverage of the batch (facility-location "
            f"gain at lambda={lambda_weight:g}), so a whole-batch failure is not "
            "confounded with a single clade",
        )

    greedy_submodular_select(div_pool, n_div, utilities, distance,
                             lambda_weight, accept=accept_diversity,
                             already_selected=list(chosen))

    # -- 4. optional reallocation of slots a role could not fill ----------
    unfilled_note = ""
    deficit = slots - len(members)
    if deficit > 0:
        unfilled = {role.value: scaled.get(role, 0)
                    - sum(1 for m in members if m.role is role)
                    for role in scaled}
        unfilled_note = ("roles left unfilled: "
                         + ", ".join(f"{r}={n}" for r, n in sorted(unfilled.items())
                                     if n > 0))
        if reallocate_unfilled_roles:
            spare_pool = [c for c in ranked if c.candidate_id not in taken]

            def accept_spare(cand: Candidate) -> bool:
                return admit(
                    cand, BatchRole.DIVERSITY,
                    "absorbed a slot another role could not fill "
                    f"({unfilled_note}); gates all passed, no standard relaxed, "
                    "but the executed role split differs from the requested one",
                )

            greedy_submodular_select(spare_pool, deficit, utilities, distance,
                                     lambda_weight, accept=accept_spare,
                                     already_selected=list(chosen))

    # -- shortfall accounting --------------------------------------------
    n_selected = len(members)
    shortfall_reason: str | None = None
    if n_selected < slots:
        bases = {pocket_signature(
            c, extra_pocket_residues=extra_pocket_residues).basis for c in eligible}
        reasons = [
            f"{n_selected} of {slots} candidate slots filled.",
            f"pool: {len(candidates)} candidate(s) considered; {len(eligible)} passed "
            f"every feasibility gate; {len(undecided)} had an undecided gate and were "
            f"not spent on (deciding those gates is the cheapest way to lengthen this "
            f"batch); {len(rejected)} failed a gate or carry an input defect.",
            f"quota refusals: {len(ledger.rejected_family)} by family cap, "
            f"{len(ledger.rejected_cluster)} by cluster cap.",
            f"uncertainty-probe pool held {len(probe_pool)} gate-passing candidate(s).",
        ]
        if unfilled_note:
            reasons.append(
                unfilled_note
                + ("; leftover slots were not reallocated, because changing the "
                   "role split changes what the round can conclude"
                   if not reallocate_unfilled_roles else
                   "; reallocation was requested but the eligible pool was exhausted")
                + ".")
        if SignatureBasis.CATALYTIC_ROLES_ONLY in bases:
            reasons.append(
                "note: some pocket signatures were built from catalytic roles only, "
                "which under-reports pocket differences.")
        if SignatureBasis.POCKET_COMPOSITION in bases:
            reasons.append(
                "note: some pocket residues arrived in no shared numbering frame, "
                "so those pockets were compared by composition rather than by "
                "position; two pockets with the same residues arranged differently "
                "look identical under that comparison.")
        reasons.append(
            "The batch is reported short on purpose: no candidate was admitted by "
            "relaxing a gate or a quota.")
        shortfall_reason = " ".join(reasons)

    return BatchPlan(
        plan_id=plan_id,
        round_number=round_number,
        members=members,
        controls=list(reservation.controls),
        cofactor_conditions=cofactor_conditions,
        replicates=replicates,
        requested_slots=slots,
        family_quota=dict(quotas or {}),
        shortfall_reason=shortfall_reason,
    )
