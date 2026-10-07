"""A small, inspectable enzyme -> target model, with probabilities that are calibrated or absent.

WHAT THIS IS
============
For each *target* -- a substrate class, a cofactor preference, or one substrate
-- a head scores how likely an enzyme sequence is to belong to the positive
class, from the sequence alone. The score comes from kernel ridge regression
over a normalised k-mer **spectrum kernel** (Leslie et al., 2002): similarity
between two proteins is the cosine between their 2-mer and 3-mer count
vectors. It is deliberately the model a careful reader could rebuild in an
afternoon, and the point of it is not that it is good. The point is that it is
*checkable*: every number it emits has a recorded training set, a stated
hyperparameter set, and a calibration that was measured out-of-fold.

pure Python, no numpy -- the repository's constraint. Kernel ridge is chosen
over iterative logistic regression because with a few hundred sequences the
exact dual solve (one Cholesky factorisation per training set, shared by every
target labelled on the same sequences) is faster in pure Python than a few
hundred epochs of gradient descent, and has no learning rate to tune.

WHAT A PROBABILITY IS ALLOWED TO MEAN HERE
==========================================
A raw kernel-ridge score is not a probability, and reading ``(s + 1) / 2`` as
one is the usual way an uncalibrated model gets quoted as a confidence. The
head therefore reports ``probability = None`` until it has a Platt map fitted
on **out-of-fold** scores -- scores for training sequences produced by models
that never saw those sequences, with folds grouped by the caller's sequence
clusters so that a near-duplicate of a held-out sequence is not in the fit.
Without grouping, calibration scores for redundant data are optimistic and the
map learned from them is overconfident on exactly the sequences that are new.

A map is only fitted with at least ``min_calibration_per_class`` of each class
among the out-of-fold scores. Below that the head still ranks (``score``) and
says it is uncalibrated; it does not invent a probability from five examples.

WHAT A LABEL IS ALLOWED TO CLAIM
================================
Every :class:`Example` says where its label came from: ``annotation`` (derived
from database records -- the SDR substrate-class dataset is this) or
``measured`` (an experiment). A prediction carries the strongest claim its
training labels could support, :attr:`Prediction.claim_ceiling`. A head trained
only on annotation-derived classes is a *classifier of annotated class*, and
the ceiling says so; it cannot be quoted as a prediction of activity on a
target substrate. Mixing the two sources in one target is refused rather than
averaged, because the two labels answer different questions.

UPDATING FROM EXPERIMENTS
=========================
:meth:`EnzymeSubstrateModel.update` returns a new model trained on the union,
and an :class:`UpdateRecord` that is checkable: training-set digests before
and after, what was added, and what could not be used and why. Nothing is
mutated. An update that "changes" the model without changing its training-set
digest is a bug this record exists to expose.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from operator import mul
from typing import Any, Iterable, Mapping, Sequence

from ..errors import EAgentError
from ..provenance import sha256_obj, utc_now

__all__ = [
    "EnzymeSubstrateError",
    "STANDARD_AA",
    "Example",
    "KernelCache",
    "SpectrumVector",
    "spectrum_vector",
    "cosine",
    "PlattCalibrator",
    "HeadReport",
    "Prediction",
    "UpdateRecord",
    "EnzymeSubstrateModel",
    "examples_from_records",
    "candidate_scorer",
    "auroc",
    "average_precision",
    "brier",
    "log_loss",
    "reliability_table",
    "expected_calibration_error",
    "cluster_bootstrap_ci",
    "grouped_folds",
]


class EnzymeSubstrateError(EAgentError):
    """The model was asked something its training data cannot support."""


STANDARD_AA: str = "ACDEFGHIKLMNPQRSTVWY"
_INDEX = {c: i for i, c in enumerate(STANDARD_AA)}

LABEL_SOURCES = ("annotation", "measured")


# ==========================================================================
# the spectrum kernel
# ==========================================================================

@dataclass(frozen=True)
class SpectrumVector:
    """Sparse 2-mer and 3-mer counts of one sequence, L2-normalised per k.

    k-mers containing a non-standard residue are skipped and counted, not
    imputed: ``X`` is not an alanine.
    """

    entries: Mapping[int, float]
    n_skipped: int
    length: int


def spectrum_vector(sequence: str, ks: Sequence[int] = (2, 3)) -> SpectrumVector:
    seq = (sequence or "").strip().upper()
    entries: dict[int, float] = {}
    skipped = 0
    offset = 0
    for k in ks:
        counts: dict[int, float] = {}
        for i in range(len(seq) - k + 1):
            code = 0
            ok = True
            for ch in seq[i:i + k]:
                idx = _INDEX.get(ch)
                if idx is None:
                    ok = False
                    break
                code = code * 20 + idx
            if ok:
                counts[code] = counts.get(code, 0.0) + 1.0
            else:
                skipped += 1
        norm = math.sqrt(sum(v * v for v in counts.values()))
        if norm > 0:
            scale = 1.0 / (norm * math.sqrt(len(ks)))
            for code, v in counts.items():
                entries[offset + code] = v * scale
        offset += 20 ** k
    return SpectrumVector(entries, skipped, len(seq))


def cosine(a: SpectrumVector, b: SpectrumVector) -> float:
    """Cosine of two spectrum vectors. Already normalised, so the dot product."""
    small, large = (a.entries, b.entries) if len(a.entries) <= len(b.entries) \
        else (b.entries, a.entries)
    return sum(v * large.get(k, 0.0) for k, v in small.items())


class KernelCache:
    """Memoised spectrum vectors and pairwise similarities.

    Shared by every fold and every refit of a benchmark so that a few hundred
    sequences are compared once rather than once per fit.
    """

    def __init__(self, ks: Sequence[int] = (2, 3)) -> None:
        self.ks = tuple(ks)
        self._vec: dict[str, SpectrumVector] = {}
        self._sim: dict[tuple[str, str], float] = {}

    def vector(self, sequence: str) -> SpectrumVector:
        v = self._vec.get(sequence)
        if v is None:
            v = self._vec[sequence] = spectrum_vector(sequence, self.ks)
        return v

    def similarity(self, a: str, b: str) -> float:
        key = (a, b) if a <= b else (b, a)
        s = self._sim.get(key)
        if s is None:
            s = self._sim[key] = cosine(self.vector(a), self.vector(b))
        return s


# ==========================================================================
# linear algebra, pure Python
# ==========================================================================

def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _cholesky(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row_i = low[i]
        for j in range(i + 1):
            s = sum(map(mul, row_i[:j], low[j][:j]))
            if i == j:
                d = a[i][i] - s
                if d <= 0.0:
                    raise EnzymeSubstrateError(
                        "the kernel matrix is not positive definite; increase "
                        "the ridge regulariser")
                row_i[j] = math.sqrt(d)
            else:
                row_i[j] = (a[i][j] - s) / low[j][j]
    return low


def _solve(low: list[list[float]], b: Sequence[float]) -> list[float]:
    n = len(low)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(map(mul, low[i][:i], y[:i]))) / low[i][i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(low[k][i] * x[k] for k in range(i + 1, n))
        x[i] = (y[i] - s) / low[i][i]
    return x


# ==========================================================================
# calibration
# ==========================================================================

@dataclass(frozen=True)
class PlattCalibrator:
    """p = sigmoid(a * score + b), fitted on out-of-fold scores.

    Uses Platt's smoothed targets, ``(N+ + 1) / (N+ + 2)`` and
    ``1 / (N- + 2)``, so a perfectly separated sample does not push the slope to
    infinity and turn a handful of examples into a probability of 0.999.
    """

    a: float
    b: float
    n_positive: int
    n_negative: int

    def __call__(self, score: float) -> float:
        z = self.a * score + self.b
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    @classmethod
    def fit(cls, scores: Sequence[float], labels: Sequence[bool],
            iterations: int = 100) -> "PlattCalibrator":
        n_pos = sum(1 for y in labels if y)
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            raise EnzymeSubstrateError(
                "a calibrator needs both classes among the out-of-fold scores")
        t_pos = (n_pos + 1.0) / (n_pos + 2.0)
        t_neg = 1.0 / (n_neg + 2.0)
        targets = [t_pos if y else t_neg for y in labels]
        a, b = 0.0, math.log((n_pos + 1.0) / (n_neg + 1.0))
        for _ in range(iterations):
            g_a = g_b = h_aa = h_ab = h_bb = 0.0
            for s, t in zip(scores, targets):
                z = a * s + b
                p = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else \
                    math.exp(z) / (1.0 + math.exp(z))
                d = p - t
                w = max(p * (1.0 - p), 1e-12)
                g_a += d * s
                g_b += d
                h_aa += w * s * s
                h_ab += w * s
                h_bb += w
            h_aa += 1e-9
            h_bb += 1e-9
            det = h_aa * h_bb - h_ab * h_ab
            if abs(det) < 1e-18:
                break
            step_a = (h_bb * g_a - h_ab * g_b) / det
            step_b = (-h_ab * g_a + h_aa * g_b) / det
            a -= step_a
            b -= step_b
            if abs(step_a) < 1e-9 and abs(step_b) < 1e-9:
                break
        return cls(a, b, n_pos, n_neg)

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "n_positive": self.n_positive,
                "n_negative": self.n_negative}


# ==========================================================================
# data
# ==========================================================================

@dataclass(frozen=True)
class Example:
    """One labelled enzyme for one target.

    ``example_id`` is the identity of the sequence (an accession or a
    candidate id); ``group`` is its leakage group (a sequence cluster), used
    only to keep the model's internal calibration folds honest.
    """

    example_id: str
    sequence: str
    target: str
    label: bool
    label_source: str
    evidence: str = ""
    group: str | None = None

    def __post_init__(self) -> None:
        if self.label_source not in LABEL_SOURCES:
            raise EnzymeSubstrateError(
                f"{self.example_id}: label_source must be one of "
                f"{LABEL_SOURCES}, got {self.label_source!r}")
        if not (self.sequence or "").strip():
            raise EnzymeSubstrateError(f"{self.example_id}: empty sequence")

    def identity(self) -> tuple:
        return (self.example_id, self.sequence, self.target, self.label,
                self.label_source)


def training_digest(examples: Iterable[Example]) -> str:
    """A digest over the training set that ignores order and ``group``."""
    return sha256_obj(sorted(e.identity() for e in examples))


# ==========================================================================
# metrics
# ==========================================================================

def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Area under the ROC curve, ties scored as half. ``None`` for one class."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        rank_sum += avg_rank * sum(1 for k in range(i, j + 1) if pairs[k][1])
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    n_pos = sum(1 for y in labels if y)
    if n_pos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    hits = 0
    total = 0.0
    for rank, i in enumerate(order, start=1):
        if labels[i]:
            hits += 1
            total += hits / rank
    return total / n_pos


def brier(probabilities: Sequence[float], labels: Sequence[bool]) -> float | None:
    if not labels:
        return None
    return sum((p - (1.0 if y else 0.0)) ** 2
               for p, y in zip(probabilities, labels)) / len(labels)


def log_loss(probabilities: Sequence[float], labels: Sequence[bool],
             eps: float = 1e-12) -> float | None:
    if not labels:
        return None
    total = 0.0
    for p, y in zip(probabilities, labels):
        p = min(max(p, eps), 1.0 - eps)
        total -= math.log(p if y else 1.0 - p)
    return total / len(labels)


def reliability_table(probabilities: Sequence[float], labels: Sequence[bool],
                      n_bins: int = 10) -> list[dict[str, Any]]:
    """Equal-width reliability bins: how often each stated probability came true."""
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probabilities, labels):
        idx = min(int(p * n_bins), n_bins - 1) if p >= 0 else 0
        bins[idx].append((p, y))
    table = []
    for i, items in enumerate(bins):
        table.append({
            "lo": i / n_bins, "hi": (i + 1) / n_bins, "n": len(items),
            "mean_predicted": (sum(p for p, _ in items) / len(items)
                               if items else None),
            "fraction_positive": (sum(1 for _, y in items if y) / len(items)
                                  if items else None),
        })
    return table


def expected_calibration_error(probabilities: Sequence[float],
                               labels: Sequence[bool],
                               n_bins: int = 10) -> float | None:
    if not labels:
        return None
    total = len(labels)
    ece = 0.0
    for row in reliability_table(probabilities, labels, n_bins):
        if row["n"]:
            ece += (row["n"] / total) * abs(
                row["mean_predicted"] - row["fraction_positive"])
    return ece


def cluster_bootstrap_ci(
    metric, scores: Sequence[float], labels: Sequence[bool],
    clusters: Sequence[str], *, n_boot: int = 500, seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    """Percentile interval from resampling *clusters*, not rows.

    Rows in one sequence cluster are not independent draws, so a row bootstrap
    understates the width. Resamples where the metric is undefined (one class
    drawn) are skipped and the interval is ``None`` when too few survive.
    """
    by_cluster: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        by_cluster.setdefault(c, []).append(i)
    names = sorted(by_cluster)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n_boot):
        idx: list[int] = []
        for _ in names:
            idx.extend(by_cluster[names[rng.randrange(len(names))]])
        v = metric([scores[i] for i in idx], [labels[i] for i in idx])
        if v is not None:
            values.append(v)
    if len(values) < max(20, n_boot // 5):
        return None
    values.sort()
    lo = values[int((alpha / 2) * len(values))]
    hi = values[min(len(values) - 1, int((1 - alpha / 2) * len(values)))]
    return lo, hi


def grouped_folds(groups: Sequence[str], n_folds: int, seed: int = 0) -> list[int]:
    """Assign each item to a fold so that no group is split across folds.

    Groups are shuffled with ``seed`` and placed greedily into the currently
    smallest fold, which balances fold sizes without splitting a group.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    members: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    names = sorted(members)
    random.Random(seed).shuffle(names)
    names.sort(key=lambda g: -len(members[g]))     # big groups first, stable shuffle
    sizes = [0] * n_folds
    fold_of = [0] * len(groups)
    for g in names:
        k = min(range(n_folds), key=lambda f: (sizes[f], f))
        for i in members[g]:
            fold_of[i] = k
        sizes[k] += len(members[g])
    return fold_of


# ==========================================================================
# the model
# ==========================================================================

@dataclass(frozen=True)
class HeadReport:
    """What one target's head was trained on, and whether it is calibrated."""

    target: str
    n_positive: int
    n_negative: int
    label_source: str
    calibrated: bool
    calibration: Mapping[str, Any] | None
    calibration_note: str
    training_digest: str
    oof_auroc: float | None = None


@dataclass(frozen=True)
class Prediction:
    target: str
    score: float | None
    probability: float | None
    calibrated: bool
    nearest_similarity: float | None
    nearest_id: str | None
    claim_ceiling: str
    basis: str
    n_positive: int = 0
    n_negative: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class UpdateRecord:
    """A checkable account of one update."""

    before_digest: str
    after_digest: str
    n_added: int
    n_duplicates_ignored: int
    targets_added_to: tuple[str, ...]
    new_targets: tuple[str, ...]
    skipped: tuple[str, ...]
    at: str = field(default_factory=utc_now)

    @property
    def changed(self) -> bool:
        return self.before_digest != self.after_digest


class _Head:
    def __init__(self, target: str, examples: list[Example]) -> None:
        self.target = target
        self.examples = examples
        self.sequences = [e.sequence for e in examples]
        self.y = [1.0 if e.label else -1.0 for e in examples]
        self.alpha: list[float] = []
        self.calibrator: PlattCalibrator | None = None
        self.calibration_note = ""
        self.oof_auroc: float | None = None
        self.source = examples[0].label_source


class EnzymeSubstrateModel:
    """Kernel-ridge heads over a spectrum kernel, one per target.

    ``ridge`` and ``ks`` are fixed hyperparameters, recorded, and never tuned
    on the data a result is reported on. ``min_calibration_per_class`` is how
    many of each class the out-of-fold scores must contain before a
    probability is reported at all.
    """

    def __init__(self, examples: Sequence[Example] = (), *, ridge: float = 0.1,
                 ks: Sequence[int] = (2, 3), calibration_folds: int = 4,
                 min_calibration_per_class: int = 8, seed: int = 0,
                 kernel: KernelCache | None = None,
                 factor_cache: dict | None = None,
                 require_skill: bool = True) -> None:
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        self.ridge = ridge
        self.ks = tuple(ks)
        self.calibration_folds = calibration_folds
        self.min_calibration_per_class = min_calibration_per_class
        self.seed = seed
        self.require_skill = require_skill
        self.kernel = kernel or KernelCache(self.ks)
        if self.kernel.ks != self.ks:
            raise ValueError("the kernel cache was built for different k values")
        self.examples: tuple[Example, ...] = ()
        self._heads: dict[str, _Head] = {}
        #: Cholesky factors by (training sequences, ridge): safe to share, since
        #: the key is the whole input of the factorisation.
        self._factor_cache: dict[tuple, list[list[float]]] = (
            factor_cache if factor_cache is not None else {})
        self.updates: list[UpdateRecord] = []
        if examples:
            self._fit(list(examples))

    # -- training ----------------------------------------------------------
    def _check(self, examples: Sequence[Example]) -> None:
        by_key: dict[tuple[str, str], bool] = {}
        sources: dict[str, str] = {}
        for e in examples:
            prior = sources.setdefault(e.target, e.label_source)
            if prior != e.label_source:
                raise EnzymeSubstrateError(
                    f"target {e.target!r} mixes {prior!r} and {e.label_source!r} "
                    f"labels; they answer different questions and are not "
                    f"averaged. Use separate target keys")
            key = (e.sequence, e.target)
            if key in by_key and by_key[key] != e.label:
                raise EnzymeSubstrateError(
                    f"{e.example_id}: the same sequence is labelled both ways "
                    f"for target {e.target!r}")
            by_key[key] = e.label

    def _fit(self, examples: list[Example]) -> None:
        self._check(examples)
        seen: set[tuple[str, str]] = set()
        unique: list[Example] = []
        for e in examples:
            key = (e.sequence, e.target)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        self.examples = tuple(unique)
        by_target: dict[str, list[Example]] = {}
        for e in unique:
            by_target.setdefault(e.target, []).append(e)
        self._heads = {}
        for target in sorted(by_target):
            head = _Head(target, by_target[target])
            self._train_head(head)
            self._heads[target] = head

    def _gram(self, seqs: Sequence[str]) -> list[list[float]]:
        n = len(seqs)
        k = [[0.0] * n for _ in range(n)]
        for i in range(n):
            k[i][i] = 1.0 + self.ridge
            for j in range(i):
                s = self.kernel.similarity(seqs[i], seqs[j])
                k[i][j] = k[j][i] = s
        return k

    def _factor(self, seqs: Sequence[str]) -> list[list[float]]:
        key = (tuple(seqs), self.ridge)
        low = self._factor_cache.get(key)
        if low is None:
            low = self._factor_cache[key] = _cholesky(self._gram(seqs))
        return low

    def _fit_scores(self, train: Sequence[str], y: Sequence[float],
                    test: Sequence[str]) -> list[float]:
        alpha = _solve(self._factor(train), y)
        return [sum(a * self.kernel.similarity(t, s) for a, s in zip(alpha, train))
                for t in test]

    def _train_head(self, head: _Head) -> None:
        head.alpha = _solve(self._factor(head.sequences), head.y)
        n_pos = sum(1 for e in head.examples if e.label)
        n_neg = len(head.examples) - n_pos
        need = self.min_calibration_per_class
        if n_pos < need or n_neg < need:
            head.calibration_note = (
                f"{n_pos} positive and {n_neg} negative training example(s); "
                f"a calibration needs at least {need} of each among the "
                f"out-of-fold scores, so this head ranks but reports no "
                f"probability")
            return
        groups = [e.group or e.example_id for e in head.examples]
        folds = grouped_folds(groups, min(self.calibration_folds,
                                          len({*groups})), self.seed)
        oof: list[float | None] = [None] * len(head.examples)
        for k in range(max(folds) + 1):
            test_idx = [i for i, f in enumerate(folds) if f == k]
            train_idx = [i for i, f in enumerate(folds) if f != k]
            if not train_idx or not test_idx:
                continue
            labels_train = {head.examples[i].label for i in train_idx}
            if len(labels_train) < 2:
                continue
            scores = self._fit_scores(
                [head.sequences[i] for i in train_idx],
                [head.y[i] for i in train_idx],
                [head.sequences[i] for i in test_idx])
            for i, s in zip(test_idx, scores):
                oof[i] = s
        triples = [(sc, e.label, g) for sc, e, g in zip(oof, head.examples, groups)
                   if sc is not None]
        pairs = [(sc, y) for sc, y, _ in triples]
        pos = sum(1 for _, y in pairs if y)
        neg = len(pairs) - pos
        if pos < need or neg < need:
            head.calibration_note = (
                f"only {pos} positive and {neg} negative out-of-fold score(s) "
                f"survived the grouped folds; at least {need} of each are "
                f"needed, so this head reports no probability")
            return
        oof_scores = [sc for sc, _, _ in triples]
        oof_labels = [y for _, y, _ in triples]
        oof_groups = [g for _, _, g in triples]
        head.oof_auroc = auroc(oof_scores, oof_labels)
        candidate = PlattCalibrator.fit(oof_scores, oof_labels)
        if candidate.a <= 0.0:
            # Cross-validating a score with no real signal does not give a flat
            # line: held-out items regress toward a training mean that is
            # *lower* when the held-out fold happened to hold more positives,
            # so the out-of-fold relationship comes out NEGATIVE. A Platt map
            # fitted to that would turn a non-skill score into a confident
            # probability pointing the wrong way.
            head.calibration_note = (
                f"the out-of-fold scores are not positively related to the "
                f"labels (Platt slope {candidate.a:.2f}, out-of-fold AUROC "
                f"{_r(head.oof_auroc)}): this head has no demonstrated skill "
                f"on held-out sequences, so it reports no probability")
            return
        if self.require_skill:
            ci = cluster_bootstrap_ci(auroc, oof_scores, oof_labels, oof_groups,
                                      n_boot=200, seed=self.seed)
            if ci is None or ci[0] <= 0.5:
                # A positive slope is not skill: on pure noise it is positive
                # about half the time. The interval has to exclude chance.
                head.calibration_note = (
                    f"out-of-fold AUROC {_r(head.oof_auroc)} with a 95% "
                    f"cluster-bootstrap interval "
                    f"{'undefined' if ci is None else [_r(ci[0]), _r(ci[1])]} "
                    f"that does not exclude 0.5: this head has no demonstrated "
                    f"skill on held-out sequences, so it reports no "
                    f"probability (set require_skill=False to calibrate "
                    f"anyway)")
                return
        head.calibrator = candidate
        head.calibration_note = (
            f"Platt map fitted on {len(pairs)} out-of-fold score(s) "
            f"({pos} positive, {neg} negative) from {max(folds) + 1} grouped folds")

    # -- prediction --------------------------------------------------------
    @property
    def targets(self) -> list[str]:
        return sorted(self._heads)

    def report(self, target: str) -> HeadReport:
        head = self._heads.get(target)
        if head is None:
            raise EnzymeSubstrateError(f"no head for target {target!r}")
        n_pos = sum(1 for e in head.examples if e.label)
        return HeadReport(
            target=target, n_positive=n_pos,
            n_negative=len(head.examples) - n_pos, label_source=head.source,
            calibrated=head.calibrator is not None,
            calibration=None if head.calibrator is None
            else head.calibrator.to_dict(),
            calibration_note=head.calibration_note,
            training_digest=training_digest(head.examples),
            oof_auroc=head.oof_auroc)

    @staticmethod
    def _ceiling(source: str) -> str:
        return ("annotation_class" if source == "annotation"
                else "measured_activity_on_training_conditions")

    def predict(self, sequence: str, target: str) -> Prediction:
        """Score one sequence for one target. Refuses a target it has no head for."""
        head = self._heads.get(target)
        if head is None:
            return Prediction(
                target=target, score=None, probability=None, calibrated=False,
                nearest_similarity=None, nearest_id=None,
                claim_ceiling="none",
                basis=f"no labelled example exists for target {target!r}; the "
                      f"model does not extrapolate to a target it was never "
                      f"given")
        sims = [(self.kernel.similarity(sequence, e.sequence), e.example_id)
                for e in head.examples]
        score = sum(a * s for a, (s, _) in zip(head.alpha, sims))
        near_sim, near_id = max(sims)
        probability = head.calibrator(score) if head.calibrator else None
        n_pos = sum(1 for e in head.examples if e.label)
        return Prediction(
            target=target, score=score, probability=probability,
            calibrated=head.calibrator is not None,
            nearest_similarity=near_sim, nearest_id=near_id,
            claim_ceiling=self._ceiling(head.source),
            basis=head.calibration_note, n_positive=n_pos,
            n_negative=len(head.examples) - n_pos)

    # -- updating ----------------------------------------------------------
    def update(self, new_examples: Sequence[Example], skipped: Sequence[str] = ()
               ) -> tuple["EnzymeSubstrateModel", UpdateRecord]:
        """A new model trained on the union, and the record of what changed."""
        before = training_digest(self.examples)
        known = {(e.sequence, e.target) for e in self.examples}
        fresh: list[Example] = []
        duplicates = 0
        for e in new_examples:
            if (e.sequence, e.target) in known:
                duplicates += 1
                continue
            known.add((e.sequence, e.target))
            fresh.append(e)
        model = EnzymeSubstrateModel(
            [*self.examples, *fresh], ridge=self.ridge, ks=self.ks,
            calibration_folds=self.calibration_folds,
            min_calibration_per_class=self.min_calibration_per_class,
            seed=self.seed, kernel=self.kernel, factor_cache=self._factor_cache,
            require_skill=self.require_skill)
        record = UpdateRecord(
            before_digest=before, after_digest=training_digest(model.examples),
            n_added=len(fresh), n_duplicates_ignored=duplicates,
            targets_added_to=tuple(sorted({e.target for e in fresh}
                                          & set(self._heads))),
            new_targets=tuple(sorted({e.target for e in fresh}
                                     - set(self._heads))),
            skipped=tuple(skipped))
        model.updates = [*self.updates, record]
        return model, record

    def model_card(self) -> dict[str, Any]:
        return {
            "model": "spectrum-kernel ridge, Platt-calibrated out-of-fold",
            "ks": list(self.ks), "ridge": self.ridge,
            "calibration_folds": self.calibration_folds,
            "min_calibration_per_class": self.min_calibration_per_class,
            "require_skill": self.require_skill,
            "training_digest": training_digest(self.examples),
            "n_examples": len(self.examples),
            "heads": {t: self.report(t).__dict__ for t in self.targets},
            "updates": [u.__dict__ for u in self.updates],
        }


# ==========================================================================
# experiments -> examples
# ==========================================================================

def examples_from_records(
    records: Iterable[Any], *, target_of=None,
) -> tuple[list[Example], list[str]]:
    """Turn experiment records into labelled examples, and say what was left out.

    Only records that inform catalytic ability become examples: a confirmed
    target product is a positive; no product detected and a wrong-product /
    wrong-configuration outcome are negatives *for the target reaction*. An
    expression or solubility failure is undetermined, not negative -- a
    protein that never folded says nothing about whether it could have
    catalysed -- and so are not-tested and computational outcomes. Each
    exclusion is returned with its reason rather than counted silently.

    ``target_of(record)`` names the head a record trains. The default is
    ``substrate:<inchikey or isomeric smiles>``; a record with neither has no
    target and is left out.
    """
    def default_target(rec: Any) -> str | None:
        sub = rec.substrate
        ident = getattr(sub, "inchikey", None) or getattr(sub, "isomeric_smiles", None)
        return f"substrate:{ident}" if ident else None

    pick = target_of or default_target
    examples: list[Example] = []
    skipped: list[str] = []
    for rec in records:
        rid = rec.record_id
        outcome = rec.outcome
        if not outcome.is_experimental:
            skipped.append(f"{rid}: outcome {outcome.value} is not an experiment")
            continue
        if not outcome.informs_catalytic_ability:
            skipped.append(
                f"{rid}: {outcome.value} leaves catalytic ability undetermined, "
                f"so it is neither a positive nor a negative")
            continue
        sequence = rec.construct_sequence or rec.sequence
        if not sequence:
            skipped.append(f"{rid}: no sequence to learn from")
            continue
        target = pick(rec)
        if not target:
            skipped.append(f"{rid}: no substrate identity to name a target")
            continue
        examples.append(Example(
            example_id=rec.sequence_sha256 or rid, sequence=sequence,
            target=target, label=bool(outcome.is_positive),
            label_source="measured", evidence=f"record {rid}",
            group=rec.parent_sequence_sha256 or rec.sequence_sha256))
    return examples, skipped


def candidate_scorer(model: EnzymeSubstrateModel, target: str, *,
                     ranking: str = "probability"):
    """A ``Candidate -> float | None`` for the specificity-model comparator seam.

    ``eagent.eval.baselines.make_specificity_model_comparator`` takes exactly
    this, and a ``None`` is recorded there as "the model gave no prediction"
    rather than as a low score. With ``ranking="probability"`` (the default) a
    head that earned no calibration returns ``None`` for every candidate, so a
    comparison cannot quietly rank on numbers that were never probabilities.
    ``ranking="score"`` ranks on the raw kernel-ridge score instead, which is
    legitimate for ordering and is labelled as such in the model id.
    """
    if ranking not in ("probability", "score"):
        raise ValueError("ranking must be 'probability' or 'score'")

    def scorer(candidate: Any) -> float | None:
        pred = model.predict(candidate.sequence, target)
        return pred.probability if ranking == "probability" else pred.score

    scorer.model_id = (  # type: ignore[attr-defined]
        f"spectrum-krr[{ranking}]:{target}:"
        f"{training_digest(model.examples)[:12]}")
    return scorer
