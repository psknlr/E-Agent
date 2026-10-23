"""Profile-to-sequence similarity from a real alignment, as an opt-in extra.

WHY THIS EXISTS
===============
The model in :mod:`eagent.science.enzyme_substrate` compares two proteins by
the cosine between their k-mer count vectors. ``composition_contrast`` on the
shipped SDR pool measured what that buys: the kernel scores a true pair only
**1.07x** its own composition background, so a k-mer cosine on this data is
very nearly a comparison of amino-acid composition, whichever k it uses. That
is recorded in ``docs/results/sdr_label_recovery.txt`` and it is the reason
widening the window from 2-mers/3-mers to 3-mers/4-mers did not help. What a
k-mer cosine never does is *align*: it cannot tell a conserved catalytic
stretch from the same residues scattered elsewhere, and it has no substitution
matrix, so a conservative replacement costs it as much as a disruptive one.

This module supplies the other thing: a bit score from HMMER's profile search.
A profile is built from one query sequence and scored against a target, which
gives a log-odds score in bits against HMMER's null model, with position
specific gap and substitution costs. It is the comparison a practitioner
actually makes, and it is the direct test of the sentence above.

WHAT IS AND IS NOT CLAIMED
==========================
* **Directional, not symmetric.** The profile is built from the *query*, so
  ``similarity(a, b)`` and ``similarity(b, a)`` are different numbers; a short
  query against a long target does not score the same as the reverse. Nothing
  here symmetrises them behind the caller's back. Use the direction a search
  would use: the new sequence is the query, the characterised ones are the
  targets.
* **Normalised by the query's self score.** A raw bit score grows with the
  length and the information content of the query, so raw scores from
  different queries are not comparable, and a metric pooled over rows would be
  reading query length as similarity. Each row is therefore divided by the
  score of the query's own profile against the query itself. That makes the
  diagonal exactly 1.0 and the rest a fraction of it. **This is a choice, not
  a law**: it is one defensible normalisation among several, it is recorded in
  every report this module emits, and an unnormalised score is available with
  ``normalise=False`` for anyone who wants to check what the choice did.
* **A missing hit is 0.0, and is counted.** HMMER reports what clears its
  thresholds. This module turns its reporting threshold off by default
  (``e_value_ceiling=1e9``, acceleration filters off) so that nearly every
  pair gets a score, but a pair HMMER still does not report is scored 0.0 and
  counted in :attr:`AlignmentReport.pairs_without_hit`, so "no detectable
  similarity" is visible in the report instead of looking like a measurement.
* **No fallback.** Without the extra installed this refuses with
  :class:`~eagent.errors.ToolUnavailableError`. It does not quietly return
  k-mer cosines, because a column labelled "alignment" that holds k-mer
  numbers is worse than an absent column.

INSTALLING IT
=============
``pip install 'eagent[search]'``. ``pyhmmer`` ships HMMER3 as a library on
PyPI with no external binary and no network access at search time, so this
runs against sequences already in the repository. :func:`pyhmmer_available`
answers whether it is here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

from ..errors import EAgentError, ToolUnavailableError

__all__ = [
    "AlignmentSimilarityError",
    "ProfileKernel",
    "profile_contrast",
    "PairwiseSimilarity",
    "AlignmentReport",
    "pyhmmer_available",
    "PhmmerSimilarity",
]

#: Reporting threshold wide enough that HMMER's E-value cut stops deciding
#: which pairs get a number. With the acceleration filters also off, a pair
#: that still goes unreported is a pair HMMER's pipeline found nothing in.
WIDE_E_VALUE = 1e9


class AlignmentSimilarityError(EAgentError):
    """A similarity was asked for that this provider cannot honestly give."""


class PairwiseSimilarity(Protocol):
    """What the benchmark needs of a similarity: one directional number.

    :class:`eagent.science.enzyme_substrate.KernelCache` satisfies this with a
    symmetric k-mer cosine; :class:`PhmmerSimilarity` satisfies it with a
    directional profile bit score. The benchmark holds this type so that it
    scores whichever one it was handed and records which.
    """

    def similarity(self, query: str, target: str) -> float:  # pragma: no cover
        ...


@dataclass(frozen=True)
class AlignmentReport:
    """Everything needed to say what produced a column of these numbers."""

    backend: str
    backend_version: str
    normalisation: str
    pool_size: int
    queries_run: int
    pairs_requested: int
    pairs_without_hit: int
    search_settings: Mapping[str, Any] = field(default_factory=dict)
    builder_settings: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "normalisation": self.normalisation,
            "pool_size": self.pool_size,
            "queries_run": self.queries_run,
            "pairs_requested": self.pairs_requested,
            "pairs_without_hit": self.pairs_without_hit,
            "search_settings": dict(self.search_settings),
            "builder_settings": dict(self.builder_settings),
        }


def pyhmmer_available() -> bool:
    """Whether ``pyhmmer`` can be imported here. Guarded: it is an optional extra."""
    try:
        import pyhmmer  # type: ignore  # noqa: F401
    except Exception:  # pragma: no cover - the absent-extra leg
        return False
    return True


def _text(name: Any) -> str:
    """A hit name as text. pyhmmer has returned ``bytes`` and ``str`` by version."""
    return name.decode() if isinstance(name, (bytes, bytearray)) else str(name)


class PhmmerSimilarity:
    """Directional ``phmmer`` bit scores over a fixed pool, memoised per query.

    The pool is every sequence that may be asked about as a *target*. A query
    is searched against the whole pool at once, the first time that query is
    asked about, because one profile build amortised over the pool is what
    makes this affordable: the shipped 358-sequence SDR pool is a few minutes
    for all 358 queries, against hours for pairwise calls.

    A target outside the pool is refused rather than scored, since a provider
    that silently returned 0.0 for an unknown sequence would be reporting "no
    similarity" for "never looked".
    """

    def __init__(self, pool: Iterable[str], *,
                 cpus: int = 0,
                 e_value_ceiling: float = WIDE_E_VALUE,
                 acceleration_filters: bool = False,
                 normalise: bool = True,
                 seed: int = 42) -> None:
        if not pyhmmer_available():
            raise ToolUnavailableError(
                "phmmer (pyhmmer)",
                hint="pip install 'eagent[search]'; pyhmmer ships HMMER3 as a "
                     "library, so no external binary is needed")
        import pyhmmer
        from pyhmmer.easel import Alphabet, DigitalSequenceBlock, TextSequence
        from pyhmmer.plan7 import Builder

        self._alphabet = Alphabet.amino()
        self._pyhmmer_version = str(pyhmmer.__version__)
        self.cpus = int(cpus)
        self.e_value_ceiling = float(e_value_ceiling)
        self.acceleration_filters = bool(acceleration_filters)
        self.normalise = bool(normalise)
        self.seed = int(seed)

        # Identical sequences collapse to one pool entry: the pool is keyed by
        # the sequence itself, so a duplicated row is searched once and two
        # ids that hold the same residues cannot disagree.
        self._index: dict[str, int] = {}
        digital = []
        for seq in pool:
            s = seq.strip().upper()
            if not s:
                raise AlignmentSimilarityError("the pool contains an empty sequence")
            if s in self._index:
                continue
            self._index[s] = len(digital)
            digital.append(
                TextSequence(name=f"p{len(digital)}".encode(), sequence=s)
                .digitize(self._alphabet))
        if not digital:
            raise AlignmentSimilarityError("the pool is empty")
        self._block = DigitalSequenceBlock(self._alphabet, digital)
        # Recorded once, from a throwaway builder, so that report() can say
        # what the search costs a gap even before a search has run. Each
        # search builds its own: pyhmmer's Builder carries state, and sharing
        # one across the search threads would make the profile depend on the
        # order the pool happened to be visited in.
        probe = Builder(self._alphabet, seed=self.seed)
        self._builder_settings: Mapping[str, Any] = {
            "score_matrix": str(probe.score_matrix),
            "gap_open": probe.popen,
            "gap_extend": probe.pextend,
            "seed": self.seed,
        }
        self._rows: dict[str, dict[int, float]] = {}
        self._self_score: dict[str, float] = {}
        self._pairs_requested = 0
        self._pairs_without_hit = 0

    # -- the PairwiseSimilarity surface ------------------------------------
    def similarity(self, query: str, target: str) -> float:
        """Bit score of ``query``'s profile against ``target``, over its self score.

        ``normalise=False`` returns the raw bit score instead. Either way the
        number is directional, and a pair HMMER did not report is 0.0 and
        counted.
        """
        q = query.strip().upper()
        t = target.strip().upper()
        idx = self._index.get(t)
        if idx is None:
            raise AlignmentSimilarityError(
                "that target is not in this provider's pool, so it was never "
                "searched; build the provider with every sequence that may be "
                "a target")
        row = self._row(q)
        self._pairs_requested += 1
        raw = row.get(idx)
        if raw is None:
            self._pairs_without_hit += 1
            return 0.0
        if not self.normalise:
            return raw
        denominator = self._self_score.get(q) or 0.0
        if denominator <= 0.0:
            # A profile that does not score its own sequence above the null
            # model has nothing to normalise by. Refusing beats dividing by a
            # number that would invert the ranking.
            raise AlignmentSimilarityError(
                "the query's profile does not score positively against the "
                "query itself, so there is no scale to normalise by; pass "
                "normalise=False to read raw bit scores")
        return raw / denominator

    def report(self) -> AlignmentReport:
        """What produced these numbers, for the benchmark report to record."""
        return AlignmentReport(
            backend="phmmer",
            backend_version=f"pyhmmer {self._pyhmmer_version} (vendored HMMER3)",
            normalisation=("bit score divided by the query profile's score "
                           "against the query itself"
                           if self.normalise else "none (raw bit score)"),
            pool_size=len(self._index),
            queries_run=len(self._rows),
            pairs_requested=self._pairs_requested,
            pairs_without_hit=self._pairs_without_hit,
            search_settings={
                "E": self.e_value_ceiling,
                "acceleration_filters": self.acceleration_filters,
                "bias_filter": self.acceleration_filters,
                "cpus": self.cpus,
            },
            builder_settings=self._builder_settings,
        )

    # -- internals ---------------------------------------------------------
    def _row(self, query: str) -> dict[int, float]:
        row = self._rows.get(query)
        if row is None:
            row = self._rows[query] = self._search(query)
        return row

    def _search(self, query: str) -> dict[int, float]:
        from pyhmmer.easel import TextSequence
        from pyhmmer.hmmer import phmmer
        from pyhmmer.plan7 import Builder

        if not query:
            raise AlignmentSimilarityError("an empty query has no profile")
        builder = Builder(self._alphabet, seed=self.seed)
        options: dict[str, Any] = {"E": self.e_value_ceiling, "cpus": self.cpus}
        if not self.acceleration_filters:
            # HMMER's MSV/bias/Viterbi filters exist to skip targets cheaply.
            # Skipping is exactly what must not happen here: an unscored pair
            # would enter the benchmark as 0.0 for a reason that is an
            # optimisation, not a measurement.
            options.update(bias_filter=False, F1=1.0, F2=1.0, F3=1.0)
        q = TextSequence(name=b"query", sequence=query).digitize(self._alphabet)
        hits_list = list(phmmer([q], self._block, builder=builder, **options))
        if not hits_list:  # pragma: no cover - phmmer yields one per query
            raise AlignmentSimilarityError("phmmer returned no result for a query")
        row: dict[int, float] = {}
        for hit in hits_list[0]:
            name = _text(hit.name)
            if not name.startswith("p"):  # pragma: no cover - names are ours
                continue
            row[int(name[1:])] = float(hit.score)
        self_index = self._index.get(query)
        if self_index is not None and self_index in row:
            self._self_score[query] = row[self_index]
        else:
            self._self_score[query] = self._self_against_self(query, builder, options)
        return row

    def _self_against_self(self, query: str, builder: Any,
                           options: Mapping[str, Any]) -> float:
        """The query profile's score against the query, when it is not in the pool."""
        from pyhmmer.easel import DigitalSequenceBlock, TextSequence
        from pyhmmer.hmmer import phmmer

        q = TextSequence(name=b"query", sequence=query).digitize(self._alphabet)
        s = TextSequence(name=b"self", sequence=query).digitize(self._alphabet)
        block = DigitalSequenceBlock(self._alphabet, [s])
        for hits in phmmer([q], block, builder=builder, **dict(options)):
            for hit in hits:
                return float(hit.score)
        return 0.0


class ProfileKernel:
    """A **symmetric** kernel built from the directional profile search.

    Kernel ridge regression needs a symmetric matrix, and it needs the matrix
    to be positive definite. A profile bit score is neither symmetric nor an
    inner product, so this class does one of those two things and refuses to
    pretend about the other:

    * **Symmetry** is imposed explicitly, as the mean of the two directions.
      That is a modelling choice with a cost -- it averages away the asymmetry
      that :class:`PhmmerSimilarity` keeps on purpose -- and it is the cheapest
      choice that lets the existing head read this similarity at all.
    * **Positive definiteness is not imposed.** Nothing here clips an
      eigenvalue or inflates the ridge to force a factorisation through. If the
      Gram matrix is indefinite, :func:`eagent.science.enzyme_substrate._cholesky`
      refuses it and the caller reports a refusal, with
      :func:`eagent.science.enzyme_substrate.min_eigenvalue` to say how far off
      it was. A model fitted on a silently repaired kernel is a model whose
      regulariser was chosen by the failure it had to survive.

    Symmetrising doubles the searches: every sequence becomes a query, not only
    the held-out ones. The provider memoises per query, so the cost is one
    profile search per distinct sequence in the pool, however many pairs are
    read.
    """

    #: A sequence against itself: both directions are the normalised self
    #: score, so exactly 1.0. The head reads this for the Gram diagonal.
    self_similarity: float = 1.0

    def __init__(self, provider: PhmmerSimilarity) -> None:
        self._provider = provider
        self._sim: dict[tuple[str, str], float] = {}

    @property
    def kernel_id(self) -> str:
        return "phmmer_symmetrised"

    def similarity(self, a: str, b: str) -> float:
        key = (a, b) if a <= b else (b, a)
        value = self._sim.get(key)
        if value is None:
            value = self._sim[key] = (self._provider.similarity(a, b)
                                      + self._provider.similarity(b, a)) / 2.0
        return value

    def report(self) -> AlignmentReport:
        return self._provider.report()


def profile_contrast(sequences: Sequence[str], *, n_pairs: int = 400,
                     seed: int = 0, **kwargs: Any) -> dict[str, Any]:
    """The label-free order-versus-composition contrast, for the profile search.

    The same diagnostic that measured the k-mer cosine at 1.07x its own
    composition background, applied to this similarity on the same pool and the
    same seeded pairs, so the two are comparable. The provider is built over
    the real *and* the shuffled sequences, because it refuses a target it never
    searched.

    The report carries the provider's own coverage under ``provider``: on a
    composition-matched shuffle the search is expected to report nothing at
    all, and "nothing at all" is the finding rather than a failure.
    """
    from .enzyme_substrate import similarity_contrast

    box: dict[str, PhmmerSimilarity] = {}

    def prepare(real: Sequence[str], shuffled: Sequence[str]) -> None:
        box["provider"] = PhmmerSimilarity(list(real) + list(shuffled),
                                           **kwargs)

    out = similarity_contrast(
        sequences, lambda a, b: box["provider"].similarity(a, b),
        measure="median normalised profile bit score between pool pairs, over "
                "the median between pairs of composition-preserving shuffles "
                "of the same sequences; no label is read",
        n_pairs=n_pairs, seed=seed, prepare=prepare)
    out["provider"] = box["provider"].report().to_dict()
    return out


def pool_from_examples(examples: Sequence[Any]) -> list[str]:
    """Every distinct sequence on a list of objects carrying ``.sequence``.

    A convenience so that a caller building a provider for a benchmark cannot
    leave a target out of the pool and meet :class:`AlignmentSimilarityError`
    halfway through a run.
    """
    seen: dict[str, None] = {}
    for e in examples:
        seen.setdefault(e.sequence.strip().upper(), None)
    return list(seen)
