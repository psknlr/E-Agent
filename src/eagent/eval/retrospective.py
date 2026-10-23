"""A retrospective label-recovery benchmark on the SDR substrate-class dataset.

WHAT QUESTION THIS ANSWERS, AND WHICH IT DOES NOT
=================================================
The only enzymology dataset this repository has ingested through a verified
route is the SDR substrate-classification deposit (Jinich and Rappoport,
Zenodo 7141435, CC BY 4.0): 358 UniProt short-chain dehydrogenase/reductase
sequences with a NAD/NADP cofactor label, three manually defined substrate
classes (phenol, sterol, coenzyme A) and membership of nine substrate
clusters. Its README says the labels were derived from the *substrate and
product annotations of the UniProtKB records*.

So the question this benchmark can ask is: **given a sequence, how well can the
annotated class of an SDR be recovered, when the test sequences are held out
by sequence cluster?** It is a test of a sequence model and of its calibration.
It is not a test of ketoreductase activity on any substrate, of the agent's
selection logic, or of the pipeline's gates, because none of those can be
scored against annotation-derived labels. Nothing printed by this module is
allowed to be read as "the agent finds active KREDs"; every report carries that
sentence.

WHAT IS CONTROLLED
==================
* **Same folds, same pool, same budget for every comparator.** One fold
  assignment per run; every comparator sees the identical training and test
  sets for every target.
* **Folds are grouped by sequence cluster**, so a held-out sequence has no
  close homologue in the training folds -- or, where the clustering grain
  cannot guarantee that, the *measured* similarity of every test sequence to
  its training set is reported. A second run with ungrouped folds reports the
  optimism that leakage adds, which is the reason grouping is not optional.
* **No tuning on test folds.** The hyperparameters are fixed in the protocol
  before the run and recorded in the report. A sensitivity sweep over the
  ridge parameter is reported as a sweep, not used to pick one.
* **Intervals are cluster bootstrap intervals**, because rows in a cluster are
  not independent draws.

THE COMPARATORS
===============
``prevalence``
    the training prevalence for every test sequence. AUROC is 0.5 by
    construction; its Brier score is what "calibrated and uninformed" costs.
``nearest_neighbour``
    similarity to the nearest positive minus similarity to the nearest
    negative, by the same k-mer cosine the model uses. The homology-transfer
    baseline. Similarity is a k-mer proxy for alignment identity -- cheaper,
    cruder, and the same for both methods.
``spectrum_krr``
    the model's raw score, read naively as a probability via ``(s + 1) / 2``
    clipped to [0, 1] -- how an uncalibrated score gets quoted.
``spectrum_krr_calibrated``
    the same score through its out-of-fold Platt map; absent for a target
    that lacks enough of both classes.
``cofactor_determinant``
    the motif-anchored cofactor determinant of
    :mod:`eagent.science.cofactor_determinant`, read off the sequence through
    the rule a family template declares. It is **not fitted**: it has no
    parameters, sees no training fold, and would score the same with the folds
    shuffled -- so it is the only comparator here that cannot be flattered by
    leakage, and the leaky second run is a control on the others rather than on
    it. It answers one question (which nicotinamide), so it is scored on the
    cofactor target and on no other, and it abstains rather than guessing when
    the anchor is absent or repeated: an abstention takes the neutral score and
    the coverage line reports how often it happened. Present only when the
    caller supplies the rules, which the report records verbatim.
``profile_nearest_neighbour``
    the same homology-transfer baseline as ``nearest_neighbour``, with the
    k-mer cosine replaced by a HMMER profile bit score
    (:mod:`eagent.science.alignment_similarity`): the only comparator here
    that aligns, with a substitution matrix and position-specific gap costs.
    It exists because ``kernel_character`` measured the k-mer cosine at 1.07x
    its own composition background on this pool, which turns "the baseline is
    very nearly a composition comparison" from an assertion into something
    testable: put a real alignment in the same slot, over the same folds and
    the same rows, and read the paired difference. Present only when the
    caller supplies a similarity provider -- it needs the ``search`` extra --
    and the report records the provider and its settings.
``profile_krr`` / ``profile_krr_calibrated``
    the same head as ``spectrum_krr``, fitted on the **symmetrised profile
    kernel** instead of the k-mer cosine, so that the one question the
    baselines leave open can be asked as a paired comparison: does the head add
    anything over homology transfer once both read the same similarity? The
    kernel is a modelling choice with a cost, recorded in
    :class:`eagent.science.alignment_similarity.ProfileKernel`: a bit score is
    not symmetric and not an inner product, so symmetry is imposed by averaging
    the two directions and positive definiteness is **not** imposed at all. A
    Gram matrix the Cholesky refuses is reported as a refusal, with an estimate
    of how far from definite it was; nothing clips an eigenvalue and the ridge
    is never raised to force a factorisation. Present only when the caller
    supplies the kernel, and the decision rule for this comparator was
    pre-registered in ``docs/results/profile_kernel_preregistration.md``.
"""

from __future__ import annotations

import csv
import hashlib
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..errors import EAgentError
from ..provenance import sha256_obj
from ..science.alignment_similarity import PairwiseSimilarity
from ..science.cofactor_determinant import (
    AnchoredDeterminantRule, DeterminantOutcome, read_determinant,
)
from ..science.enzyme_substrate import (
    EnzymeSubstrateError, EnzymeSubstrateModel, Example, KernelCache, auroc,
    average_precision,
    composition_contrast,
    brier, cluster_bootstrap_ci, expected_calibration_error, grouped_folds,
    log_loss, reliability_table,
)
from ..science.robustness import wilson_interval

__all__ = [
    "RetrospectiveError",
    "SDR_VERSION_DOI",
    "SDR_FILE_MD5",
    "CLAIM",
    "SdrDataset",
    "load_sdr",
    "sdr_examples",
    "BenchmarkProtocol",
    "TargetResult",
    "BenchmarkReport",
    "SdrSuite",
    "run_sdr_suite",
    "run_benchmark",
    "precision_at_k",
    "COMPARATORS",
    "DETERMINANT_TARGETS",
    "SDR_FAMILY_TEMPLATE_ID",
    "sdr_determinant_rules",
    "sdr_alignment_similarity",
]

#: Every comparator, in report order. One tuple, so a comparator cannot be
#: added to the scoring loop and forgotten in the macro average, or the other
#: way round.
COMPARATORS: tuple[str, ...] = (
    "prevalence",
    "nearest_neighbour",
    "profile_nearest_neighbour",
    "spectrum_krr",
    "spectrum_krr_calibrated",
    "profile_krr",
    "profile_krr_calibrated",
    "cofactor_determinant",
)

#: Comparators whose probability exists only where the calibration gate was
#: met. A pooled metric over a mix of calibrated and absent probabilities would
#: describe neither, so these report ranking metrics with a note instead.
_CALIBRATED: frozenset[str] = frozenset(
    {"spectrum_krr_calibrated", "profile_krr_calibrated"})

#: Which target the determinant comparator answers, and the cofactor a
#: positive label on that target means. Scoring a 2'-phosphate determinant
#: against a sterol-substructure label would produce a number with no content,
#: so the mapping is explicit and short rather than inferred from the target
#: name.
DETERMINANT_TARGETS: Mapping[str, str] = {"sdr:cofactor_NADP": "NADPH"}

#: Comparators that produce a ranking score and no probability. Passing a
#: probability list of ``None``s to the metrics would be reported as a Brier
#: score of nothing rather than as no Brier score.
_RANKING_ONLY: frozenset[str] = frozenset(
    {"nearest_neighbour", "profile_nearest_neighbour", "cofactor_determinant"})

#: What an outcome contributes as a ranking score. ``PRESENT`` ranks above
#: ``ABSENT``; every refusal takes the neutral middle, so an abstention earns
#: no credit and is not scored as a wrong call either, and the report says how
#: often each outcome occurred.
_DETERMINANT_SCORE: Mapping[DeterminantOutcome, float] = {
    DeterminantOutcome.PRESENT: 1.0,
    DeterminantOutcome.ABSENT: 0.0,
    DeterminantOutcome.ANCHOR_ABSENT: 0.5,
    DeterminantOutcome.ANCHOR_NOT_UNIQUE: 0.5,
    DeterminantOutcome.WINDOW_TRUNCATED: 0.5,
}

SDR_VERSION_DOI = "10.5281/zenodo.7141435"

#: The family template whose cofactor determinant this dataset can be scored
#: against. The dataset is SDR sequences, so an AKR or MDR rule would be
#: applied to the wrong fold.
SDR_FAMILY_TEMPLATE_ID = "fam.sdr.v1"

#: md5 of each file, as the Zenodo record itself states them (fixture
#: ``tests/fixtures/zenodo/record_7141435.json``). Checked on every load: a
#: benchmark on a silently different file is a different benchmark.
SDR_FILE_MD5: Mapping[str, str] = {
    "SDR_sequences.fasta": "ea34d586f1e0c8f624b568f57c2ff7e5",
    "SDR_cofactor_classifications.csv": "b8946183ac2e0eef4a7a2dffe3acce58",
    "SDR_substructure_classifications.csv": "67a8f60f79060597e0f89d4f917d8019",
    "SDR_cluster_classifications_2DIMUMAP.csv": "2013eb1793322266638f832f7eee0201",
}

CLAIM = (
    "This benchmark recovers ANNOTATION-DERIVED class labels (UniProtKB "
    "substrate/product annotations, via the SDR deposit) from sequence. It is "
    "not evidence about ketoreductase activity on any substrate, about the "
    "agent's selection logic, or about the pipeline's gates.")


class RetrospectiveError(EAgentError):
    """The benchmark's inputs are not what its numbers would claim they are."""


# ==========================================================================
# the dataset
# ==========================================================================

@dataclass(frozen=True)
class SdrDataset:
    sequences: Mapping[str, str]
    cofactor: Mapping[str, float]
    substructure: Mapping[str, Mapping[str, float]]
    clusters: Mapping[str, Mapping[str, float]]
    file_md5: Mapping[str, str]


def _read_fasta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    key: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(">"):
            key = line[1:].split()[0]
            if key in out:
                raise RetrospectiveError(f"duplicate accession {key} in {path.name}")
            out[key] = ""
        elif key is not None and line:
            out[key] += line
    return out


def _read_table(path: Path) -> dict[str, dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, dict[str, float]] = {}
    for row in rows:
        entry = row["Entry"].strip()
        if entry in out:
            raise RetrospectiveError(f"duplicate entry {entry} in {path.name}")
        out[entry] = {k: float(v) for k, v in row.items()
                      if k != "Entry" and v not in ("", None)}
    return out


def load_sdr(data_dir: str | Path, *, verify: bool = True) -> SdrDataset:
    """Read the four SDR files, checking each against the record's md5."""
    root = Path(data_dir)
    md5s: dict[str, str] = {}
    for name, expected in SDR_FILE_MD5.items():
        path = root / name
        if not path.is_file():
            raise RetrospectiveError(f"{path} is missing")
        got = hashlib.md5(path.read_bytes()).hexdigest()
        md5s[name] = got
        if verify and got != expected:
            raise RetrospectiveError(
                f"{name} has md5 {got}, but the Zenodo record states "
                f"{expected}; refusing to benchmark on a different file")
    cofactor = {k: v["cofactor"] for k, v in
                _read_table(root / "SDR_cofactor_classifications.csv").items()}
    return SdrDataset(
        sequences=_read_fasta(root / "SDR_sequences.fasta"), cofactor=cofactor,
        substructure=_read_table(root / "SDR_substructure_classifications.csv"),
        clusters=_read_table(root / "SDR_cluster_classifications_2DIMUMAP.csv"),
        file_md5=md5s)


def sdr_examples(dataset: SdrDataset) -> list[Example]:
    """One annotation-sourced example per (enzyme, target) the dataset labels."""
    evidence = f"SDR deposit {SDR_VERSION_DOI}: derived from UniProtKB annotations"
    out: list[Example] = []

    def add(acc: str, target: str, value: float) -> None:
        seq = dataset.sequences.get(acc)
        if seq is None:
            return          # labelled but no sequence: cannot be an example
        out.append(Example(acc, seq, target, value >= 0.5, "annotation", evidence))

    for acc, v in sorted(dataset.cofactor.items()):
        add(acc, "sdr:cofactor_NADP", v)
    for acc, row in sorted(dataset.substructure.items()):
        for col, v in sorted(row.items()):
            add(acc, f"sdr:{col}", v)
    for acc, row in sorted(dataset.clusters.items()):
        for col, v in sorted(row.items()):
            add(acc, f"sdr:substrate_{col}", v)
    return out


# ==========================================================================
# the protocol and the run
# ==========================================================================

@dataclass(frozen=True)
class BenchmarkProtocol:
    """Everything fixed before the run. Recorded in the report verbatim."""

    n_folds: int = 5
    identity_threshold: float = 0.4
    grouped: bool = True
    seed: int = 0
    ridge: float = 0.1
    ks: tuple[int, ...] = (2, 3)
    n_boot: int = 300
    min_per_class: int = 10
    calibration_folds: int = 4
    min_calibration_per_class: int = 8

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__, ks=list(self.ks))


@dataclass
class TargetResult:
    target: str
    n: int
    n_positive: int
    comparators: dict[str, dict[str, Any]] = field(default_factory=dict)
    delta_auroc_vs_nearest_neighbour: dict[str, Any] | None = None
    delta_auroc_determinant_vs_nearest_neighbour: dict[str, Any] | None = None
    #: alignment baseline against k-mer baseline: does aligning help at the
    #: same job, on the same rows?
    delta_auroc_profile_vs_nearest_neighbour: dict[str, Any] | None = None
    #: the model against whichever baseline is stronger here. A model that
    #: beats the k-mer baseline and loses to the alignment one has not been
    #: shown to add anything, and one number cannot say that.
    delta_auroc_vs_profile: dict[str, Any] | None = None
    #: the head on the profile kernel against homology transfer over the
    #: *same* similarity: the only pairing that isolates what the head adds.
    delta_auroc_profile_krr_vs_profile_nn: dict[str, Any] | None = None
    #: the same head on two kernels, which is the representation change alone.
    delta_auroc_profile_krr_vs_spectrum_krr: dict[str, Any] | None = None


@dataclass
class BenchmarkReport:
    protocol: BenchmarkProtocol
    claim: str
    dataset_md5: Mapping[str, str]
    n_sequences: int
    n_clusters: int
    clustering: Mapping[str, Any]
    targets: list[TargetResult]
    skipped_targets: dict[str, str]
    macro: dict[str, dict[str, float | None]]
    reliability: dict[str, list[dict[str, Any]]]
    similarity_to_training: dict[str, Any]
    uncalibrated_targets: list[str]
    kernel_character: dict[str, Any] = field(default_factory=dict)
    alignment_similarity: dict[str, Any] = field(default_factory=dict)
    alignment_kernel: dict[str, Any] = field(default_factory=dict)
    determinant_rules: list[dict[str, Any]] = field(default_factory=list)
    determinant_outcomes: dict[str, dict[str, int]] = field(default_factory=dict)
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol.to_dict(), "claim": self.claim,
            "dataset_md5": dict(self.dataset_md5),
            "n_sequences": self.n_sequences, "n_clusters": self.n_clusters,
            "clustering": dict(self.clustering),
            "targets": [t.__dict__ for t in self.targets],
            "skipped_targets": self.skipped_targets, "macro": self.macro,
            "reliability": self.reliability,
            "similarity_to_training": self.similarity_to_training,
            "uncalibrated_targets": self.uncalibrated_targets,
            "kernel_character": self.kernel_character,
            "alignment_similarity": self.alignment_similarity,
            "alignment_kernel": self.alignment_kernel,
            "determinant_rules": self.determinant_rules,
            "determinant_outcomes": self.determinant_outcomes,
            "digest": self.digest,
        }

    def render(self) -> str:
        p = self.protocol
        lines = [
            "RETROSPECTIVE LABEL RECOVERY ON THE SDR DATASET",
            self.claim, "",
            f"folds={p.n_folds} grouped={p.grouped} identity<={p.identity_threshold} "
            f"seed={p.seed} ridge={p.ridge} ks={list(p.ks)}",
            f"{self.n_sequences} sequences in {self.n_clusters} cluster(s); "
            f"clustering: {self.clustering.get('method')}",
            f"median max-similarity of a test sequence to its training folds: "
            f"{self.similarity_to_training.get('median')}"]
        character = self.kernel_character
        if character.get("ratio") is not None:
            lines.append(
                f"kernel: median pair similarity {character['median_real_pair']} "
                f"against {character['median_composition_matched_pair']} for "
                f"composition-matched shuffles of the same sequences, so "
                f"{character['ratio']}x the composition background (1.0 would "
                f"be a composition comparison and nothing more)")
        lines.append("")
        names = ["prevalence", "nearest_neighbour", "spectrum_krr_calibrated"]
        if any("profile_nearest_neighbour" in t.comparators
               for t in self.targets):
            names.insert(2, "profile_nearest_neighbour")
        if any("profile_krr_calibrated" in t.comparators
               for t in self.targets):
            names.append("profile_krr_calibrated")
        if any("cofactor_determinant" in t.comparators for t in self.targets):
            names.append("cofactor_determinant")
        lines.append(f"{'target':<30}{'n':>5}{'pos':>5}  "
                     + "".join(f"{c:>26}" for c in names)
                     + "      (AUROC [95% cluster-bootstrap CI])")
        for t in self.targets:
            cells = []
            for name in names:
                c = t.comparators.get(name)
                if not c or c.get("auroc") is None:
                    cells.append(f"{'-':>26}")
                    continue
                ci = c.get("auroc_ci")
                text = f"{c['auroc']:.2f}" + (
                    f" [{ci[0]:.2f},{ci[1]:.2f}]" if ci else "")
                cells.append(f"{text:>26}")
            lines.append(f"{t.target:<30}{t.n:>5}{t.n_positive:>5}  "
                         + "".join(cells))
        lines.append("")
        for name, m in self.macro.items():
            lines.append(
                f"macro {name:<26} AUROC={_f(m.get('auroc'))} "
                f"(over {int(m.get('n_targets_auroc') or 0)} targets)  "
                f"Brier={_f(m.get('brier'))} ECE={_f(m.get('ece'))} "
                f"(over {int(m.get('n_targets_probability') or 0)} targets)")
        if self.uncalibrated_targets:
            lines.append("")
            lines.append(
                "no probability reported for at least one fold (too few of "
                "one class among the out-of-fold scores, or no demonstrated "
                "held-out skill -- the head's own note says which): "
                + ", ".join(self.uncalibrated_targets))
        wins = [t.target for t in self.targets
                if (d := t.delta_auroc_vs_nearest_neighbour)
                and d["interval_excludes_zero"] and d["delta"] > 0]
        losses = [t.target for t in self.targets
                  if (d := t.delta_auroc_vs_nearest_neighbour)
                  and d["interval_excludes_zero"] and d["delta"] < 0]
        lines.append("")
        lines.append(
            f"spectrum_krr vs nearest_neighbour, paired cluster-bootstrap "
            f"interval on the AUROC difference excludes zero: better on "
            f"{len(wins)} target(s) {wins}, worse on {len(losses)} {losses}, "
            f"indistinguishable on "
            f"{len(self.targets) - len(wins) - len(losses)}")
        lines += self._alignment_lines()
        lines += self._determinant_lines()
        for t, why in sorted(self.skipped_targets.items()):
            lines.append(f"skipped {t}: {why}")
        return "\n".join(lines)

    def _alignment_lines(self) -> list[str]:
        """The alignment baseline: what produced it, and the two paired tests.

        Two differences are printed because one would be misread. The first
        asks whether aligning beats the k-mer cosine at the baseline's own job.
        The second asks whether the model still beats the baseline once the
        baseline aligns -- a model that only wins against the weaker of two
        baselines has not been shown to add anything over homology transfer.
        """
        a = self.alignment_similarity
        if not a:
            return []
        out = ["", "ALIGNMENT BASELINE (a real profile search, not a fit: no "
                   "parameters, no training fold consulted):"]
        b = a.get("builder_settings") or {}
        search = a.get("search_settings") or {}
        out.append(
            f"  {a.get('backend')} via {a.get('backend_version')}; "
            f"{b.get('score_matrix')}, gap open {b.get('gap_open')}, gap "
            f"extend {b.get('gap_extend')}; reporting E<={search.get('E')}, "
            f"acceleration filters "
            + ("on" if search.get("acceleration_filters") else "off"))
        out.append(f"  normalisation: {a.get('normalisation')}")
        requested = a.get("pairs_requested") or 0
        missing = a.get("pairs_without_hit") or 0
        out.append(
            f"  {a.get('queries_run')} profile(s) searched against a pool of "
            f"{a.get('pool_size')}; {requested} pair(s) read, {missing} with "
            f"no reported hit"
            + (f" ({missing / requested:.1%}, scored 0.0)" if requested
               else "")
            + ". A pair with no hit is not a measured dissimilarity.")
        out += self._alignment_kernel_lines()
        for t in self.targets:
            for label, d in (
                    ("profile_nearest_neighbour vs nearest_neighbour",
                     t.delta_auroc_profile_vs_nearest_neighbour),
                    ("spectrum_krr vs profile_nearest_neighbour",
                     t.delta_auroc_vs_profile),
                    ("profile_krr vs profile_nearest_neighbour",
                     t.delta_auroc_profile_krr_vs_profile_nn),
                    ("profile_krr vs spectrum_krr",
                     t.delta_auroc_profile_krr_vs_spectrum_krr)):
                if not d:
                    continue
                ci = d.get("ci95")
                out.append(
                    f"  {t.target}: {label} {d['delta']:+.3f}"
                    + (f" [{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else
                       " (no interval)")
                    + " -- "
                    + ("the interval excludes zero"
                       if d["interval_excludes_zero"]
                       else "the interval includes zero, so this pool does "
                            "not separate them"))
        return out

    def _alignment_kernel_lines(self) -> list[str]:
        """The head on the profile kernel: whether it could be fitted at all.

        A refusal is printed as prominently as a result. Kernel ridge needs a
        positive-definite Gram matrix, a symmetrised bit-score matrix need not
        give one, and the pre-registered rule was to report that rather than
        raise the ridge until the factorisation went through.
        """
        k = self.alignment_kernel
        if not k:
            return []
        out = [f"  head refitted on kernel {k.get('kernel_id')} at ridge "
               f"{k.get('ridge')}: "
               + ("fitted on every fold; its column is in the table above"
                  if k.get("scored")
                  else "NOT SCORED -- the Cholesky refused the Gram matrix")]
        if k.get("cholesky_refusal"):
            out.append(f"    refusal: {k['cholesky_refusal']}")
            out.append("    the ridge was NOT raised to force it through; a "
                       "kernel the factorisation refuses is reported refused")
        if k.get("definiteness"):
            d = k["definiteness"]
            spectrum = d.get("spectrum_with_ridge") or {}
            out.append(
                f"    whole-pool check: factorises={d.get('factorises')}, "
                f"smallest eigenvalue with the ridge "
                f"{spectrum.get('smallest_estimate')} (estimate), without it "
                f"{d.get('smallest_eigenvalue_without_ridge_estimate')}")
        return out

    def _determinant_lines(self) -> list[str]:
        """The determinant's rule, what it could not read, and the paired test.

        The coverage line is not decoration. A rule that abstains on a fifth of
        the pool and ranks well on the rest is a different instrument from one
        that answers everywhere, and the AUROC alone does not distinguish them
        because an abstention sits at the neutral score.
        """
        if not self.determinant_rules:
            return []
        out = ["", "COFACTOR DETERMINANT (a rule, not a fit: no parameters, no "
                   "training fold consulted):"]
        for r in self.determinant_rules:
            out.append(
                f"  {r['cofactor']}: {'/'.join(r['accepted_residues'])} at "
                f"offsets {r['window'][0]}..{r['window'][1]} from motif "
                f"{r['anchor_motif']} ({r['anchor_pattern']}), from "
                f"{r['source_template_id']}"
                + ("; THE RULE ITSELF NEEDS CURATION" if r["needs_curation"]
                   else ""))
        for target, counts in sorted(self.determinant_outcomes.items()):
            total = sum(counts.values()) or 1
            determined = counts.get("present", 0) + counts.get("absent", 0)
            out.append(
                f"  {target}: read on {determined}/{total} sequences "
                f"({determined / total:.0%}); "
                + ", ".join(f"{k}={v}" for k, v in counts.items())
                + ". An unread sequence takes the neutral score.")
        for t in self.targets:
            d = t.delta_auroc_determinant_vs_nearest_neighbour
            if not d:
                continue
            ci = d.get("ci95")
            out.append(
                f"  {t.target}: AUROC difference against nearest_neighbour "
                f"{d['delta']:+.3f}"
                + (f" [{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else " (no interval)")
                + " -- "
                + ("the interval excludes zero"
                   if d["interval_excludes_zero"]
                   else "the interval includes zero, so this pool does not "
                        "separate them"))
        return out


def _f(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def precision_at_k(scores: Sequence[float], labels: Sequence[bool], k: int
                   ) -> dict[str, Any]:
    """Hits among the top ``k``, with a Wilson interval, ties broken by index."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    hits = sum(1 for i in order if labels[i])
    lo, hi = wilson_interval(hits, len(order)) if order else (None, None)
    return {"k": len(order), "hits": hits,
            "precision": hits / len(order) if order else None,
            "wilson_95": [lo, hi] if order else None}


def _metrics(scores: list[float], probs: list[float] | None, labels: list[bool],
             clusters: list[str], protocol: BenchmarkProtocol, seed: int
             ) -> dict[str, Any]:
    out: dict[str, Any] = {
        "auroc": auroc(scores, labels),
        "average_precision": average_precision(scores, labels),
        "prevalence": sum(labels) / len(labels) if labels else None,
    }
    out["auroc_ci"] = (cluster_bootstrap_ci(auroc, scores, labels, clusters,
                                            n_boot=protocol.n_boot, seed=seed)
                       if out["auroc"] is not None else None)
    if probs is not None:
        out.update(brier=brier(probs, labels), log_loss=log_loss(probs, labels),
                   ece=expected_calibration_error(probs, labels))
    return out


def _alignment_record(provider: Any) -> dict[str, Any]:
    """What to record about a similarity provider, without requiring a type.

    :class:`eagent.science.alignment_similarity.PhmmerSimilarity` reports its
    backend, normalisation and coverage, and that is what lands in the report.
    A provider that reports nothing is named rather than described: a column
    whose origin the report cannot state is worse than one it admits it cannot
    state.
    """
    report = getattr(provider, "report", None)
    if callable(report):
        value = report()
        as_dict = getattr(value, "to_dict", None)
        if callable(as_dict):
            return dict(as_dict())
        if isinstance(value, Mapping):
            return dict(value)
    return {"backend": type(provider).__name__,
            "note": "this provider records no settings, so the report cannot "
                    "say how the numbers in its column were produced"}


class _DeterminantScorer:
    """Scores the determinant comparator, and counts what it could not read.

    Memoised per sequence: the rule is a pure function of the sequence, so a
    sequence appearing under several targets is read once. There is nothing to
    fit, which is the point -- no training fold is consulted and the score is
    the same whichever fold the row landed in.
    """

    def __init__(self, rules: Sequence[AnchoredDeterminantRule]) -> None:
        from ..tools.annotate_family import compile_motif

        self._compile = compile_motif
        self._by_cofactor: dict[str, AnchoredDeterminantRule] = {}
        for rule in rules:
            if rule.cofactor in self._by_cofactor:
                raise RetrospectiveError(
                    f"two determinant rules for {rule.cofactor}; the benchmark "
                    f"would have to choose between them")
            self._by_cofactor[rule.cofactor] = rule
        self._cache: dict[tuple[str, str], float] = {}
        self.outcomes: dict[str, dict[str, int]] = {}

    @property
    def active(self) -> bool:
        return bool(self._by_cofactor)

    def rule_records(self) -> list[dict[str, Any]]:
        return [
            {"cofactor": r.cofactor, "source_template_id": r.source_template_id,
             "anchor_motif": r.anchor_motif, "anchor_pattern": r.anchor_pattern,
             "window": list(r.window),
             "accepted_residues": sorted(r.accepted_residues),
             "needs_curation": r.needs_curation}
            for r in sorted(self._by_cofactor.values(), key=lambda r: r.cofactor)
        ]

    def score(self, sequence: str, target: str) -> float | None:
        """The ranking score for this target, or ``None`` if it has no rule."""
        cofactor = DETERMINANT_TARGETS.get(target)
        rule = self._by_cofactor.get(cofactor or "")
        if rule is None:
            return None
        key = (sequence, target)
        if key not in self._cache:
            call = read_determinant(sequence, rule, compile_pattern=self._compile)
            counts = self.outcomes.setdefault(target, {})
            counts[call.outcome.value] = counts.get(call.outcome.value, 0) + 1
            self._cache[key] = _DETERMINANT_SCORE[call.outcome]
        return self._cache[key]


def run_benchmark(examples: Sequence[Example], protocol: BenchmarkProtocol | None = None,
                  *, cluster_assignment: Mapping[str, str] | None = None,
                  clustering_note: Mapping[str, Any] | None = None,
                  kernel: KernelCache | None = None,
                  determinant_rules: Sequence[AnchoredDeterminantRule] = (),
                  alignment: PairwiseSimilarity | None = None,
                  alignment_kernel: Any | None = None
                  ) -> BenchmarkReport:
    """Run every comparator over identical folds and report out-of-fold results.

    ``cluster_assignment`` maps ``example_id`` to a cluster id (the caller
    clusters, so the clustering method is the caller's and is recorded). With
    ``protocol.grouped`` False the folds ignore clusters -- the leaky variant,
    run on purpose to measure how much leakage flatters the numbers.

    ``determinant_rules`` are the motif-anchored cofactor rules a family
    template declares. They are passed in rather than loaded here so that the
    report records which rule produced the column, and so that this module
    stays free of template discovery. With none supplied the comparator is
    absent from the report rather than reported as zero.

    ``alignment`` is any provider of a directional pairwise similarity -- in
    practice :class:`eagent.science.alignment_similarity.PhmmerSimilarity`.
    Supplied, it adds the ``profile_nearest_neighbour`` column over the
    identical folds and rows; absent, that column is absent. It is a
    constructor argument rather than an import so that this module does not
    decide whether an optional extra is installed, and so that one provider
    (and therefore one profile search per sequence) is shared by the grouped,
    ungrouped and sweep runs.

    ``alignment_kernel`` is a kernel for the *head* -- in practice
    :class:`eagent.science.alignment_similarity.ProfileKernel`. Supplied, the
    same head is fitted a second time on it over the identical folds, adding
    the ``profile_krr`` columns. A fold whose Gram matrix the Cholesky refuses
    leaves those columns absent for the whole run with the refusal recorded,
    rather than contributing some folds and not others.
    """
    protocol = protocol or BenchmarkProtocol()
    if not examples:
        raise RetrospectiveError("no examples")
    kernel = kernel or KernelCache(protocol.ks)
    sequences: dict[str, str] = {}
    for e in examples:
        if sequences.setdefault(e.example_id, e.sequence) != e.sequence:
            raise RetrospectiveError(
                f"example id {e.example_id} names two different sequences")
    ids = sorted(sequences)
    assignment = dict(cluster_assignment or {i: i for i in ids})
    missing = [i for i in ids if i not in assignment]
    if missing:
        raise RetrospectiveError(f"no cluster for {missing[:3]}...")
    fold_groups = [assignment[i] if protocol.grouped else f"row:{i}" for i in ids]
    fold_list = grouped_folds(fold_groups, protocol.n_folds, protocol.seed)
    fold_of = dict(zip(ids, fold_list))

    by_target: dict[str, list[Example]] = {}
    for e in examples:
        by_target.setdefault(e.target, []).append(e)
    skipped: dict[str, str] = {}
    keep: dict[str, list[Example]] = {}
    for target, rows in sorted(by_target.items()):
        n_pos = sum(1 for e in rows if e.label)
        n_neg = len(rows) - n_pos
        if n_pos < protocol.min_per_class or n_neg < protocol.min_per_class:
            skipped[target] = (f"{n_pos} positive / {n_neg} negative; "
                               f"{protocol.min_per_class} of each are required")
        else:
            keep[target] = rows

    grouped_examples = [
        Example(e.example_id, e.sequence, e.target, e.label, e.label_source,
                e.evidence, assignment[e.example_id])
        for rows in keep.values() for e in rows]

    pooled: dict[str, dict[str, dict[str, list]]] = {
        t: {c: {"score": [], "prob": [], "label": [], "cluster": [], "id": []}
            for c in COMPARATORS}
        for t in keep}
    determinant = _DeterminantScorer(determinant_rules)
    max_sims: list[float] = []
    uncalibrated: set[str] = set()
    kernel_refusal: str | None = None
    factor_cache: dict = {}
    for k in range(protocol.n_folds):
        train = [e for e in grouped_examples if fold_of[e.example_id] != k]
        test = [e for e in grouped_examples if fold_of[e.example_id] == k]
        if not test:
            continue
        model = EnzymeSubstrateModel(
            train, ridge=protocol.ridge, ks=protocol.ks,
            calibration_folds=protocol.calibration_folds,
            min_calibration_per_class=protocol.min_calibration_per_class,
            seed=protocol.seed, kernel=kernel, factor_cache=factor_cache)
        profile_model = None
        if alignment_kernel is not None and kernel_refusal is None:
            try:
                profile_model = EnzymeSubstrateModel(
                    train, ridge=protocol.ridge, ks=protocol.ks,
                    calibration_folds=protocol.calibration_folds,
                    min_calibration_per_class=protocol.min_calibration_per_class,
                    seed=protocol.seed, kernel=alignment_kernel,
                    factor_cache=factor_cache)
            except EnzymeSubstrateError as exc:
                # One fold's refusal disqualifies the column for the run: a
                # macro average over the folds whose matrix happened to
                # factorise is a number about which folds those were, so
                # whatever earlier folds contributed is discarded here.
                kernel_refusal = str(exc)
                profile_model = None
                for slots in pooled.values():
                    for name in ("profile_krr", "profile_krr_calibrated"):
                        for column in slots[name].values():
                            column.clear()
        train_by_target: dict[str, list[Example]] = {}
        for e in train:
            train_by_target.setdefault(e.target, []).append(e)
        train_seqs = sorted({e.sequence for e in train})
        seen_in_fold: set[str] = set()
        for e in test:
            if e.example_id not in seen_in_fold:
                seen_in_fold.add(e.example_id)
                max_sims.append(max((kernel.similarity(e.sequence, s)
                                     for s in train_seqs), default=0.0))
            rows = train_by_target.get(e.target, [])
            pos = [r for r in rows if r.label]
            neg = [r for r in rows if not r.label]
            if not pos or not neg:
                continue
            slot = pooled[e.target]
            prevalence = len(pos) / len(rows)
            sim_pos = max(kernel.similarity(e.sequence, r.sequence) for r in pos)
            sim_neg = max(kernel.similarity(e.sequence, r.sequence) for r in neg)
            pred = model.predict(e.sequence, e.target)
            raw = pred.score if pred.score is not None else 0.0
            # The prevalence comparator ranks nothing: a constant score. Using
            # the fold's prevalence as the score would let the *difference in
            # prevalence between folds* masquerade as ranking skill (or its
            # opposite) once the folds are pooled.
            columns = [
                ("prevalence", 0.0, prevalence),
                ("nearest_neighbour", sim_pos - sim_neg, None),
                ("spectrum_krr", raw, min(max((raw + 1.0) / 2.0, 0.0), 1.0)),
                ("spectrum_krr_calibrated", raw, pred.probability),
            ]
            if alignment is not None:
                # The direction a search would use: the held-out sequence is
                # the query, the training sequences are the targets. Reversing
                # it would score a profile of a known enzyme against the new
                # one, which is a different question and a different number.
                a_pos = max(alignment.similarity(e.sequence, r.sequence)
                            for r in pos)
                a_neg = max(alignment.similarity(e.sequence, r.sequence)
                            for r in neg)
                columns.append(
                    ("profile_nearest_neighbour", a_pos - a_neg, None))
            if profile_model is not None:
                p_pred = profile_model.predict(e.sequence, e.target)
                p_raw = p_pred.score if p_pred.score is not None else 0.0
                columns += [
                    ("profile_krr", p_raw,
                     min(max((p_raw + 1.0) / 2.0, 0.0), 1.0)),
                    ("profile_krr_calibrated", p_raw, p_pred.probability),
                ]
                if not p_pred.calibrated:
                    uncalibrated.add(f"{e.target} (profile kernel)")
            det = determinant.score(e.sequence, e.target)
            if det is not None:
                columns.append(("cofactor_determinant", det, None))
            for name, score, prob in columns:
                s = slot[name]
                s["score"].append(score)
                s["prob"].append(prob)
                s["label"].append(e.label)
                s["cluster"].append(e.group or e.example_id)
                s["id"].append(e.example_id)
            if not pred.calibrated:
                uncalibrated.add(e.target)

    results: list[TargetResult] = []
    for target in sorted(keep):
        slot = pooled[target]
        labels = slot["prevalence"]["label"]
        tr = TargetResult(target=target, n=len(labels),
                          n_positive=sum(labels))
        for name, s in slot.items():
            probs = s["prob"]
            if not s["score"]:
                # A comparator that answers a different question is absent from
                # this target rather than present with a placeholder: an empty
                # cell reads as "not applicable", a zero reads as "measured".
                continue
            if name in _CALIBRATED:
                if any(p is None for p in probs):
                    # a calibrated number only where every fold produced one
                    # -- a pooled metric over a mix of calibrated and absent
                    # probabilities would describe neither
                    tr.comparators[name] = _metrics(
                        s["score"], None, s["label"], s["cluster"], protocol,
                        protocol.seed)
                    tr.comparators[name]["note"] = (
                        "no probability in at least one fold; ranking metrics "
                        "only")
                    continue
            tr.comparators[name] = _metrics(
                s["score"],
                None if name in _RANKING_ONLY else [float(p) for p in probs],
                s["label"], s["cluster"], protocol, protocol.seed)
        tr.delta_auroc_vs_nearest_neighbour = _paired_delta(
            slot["spectrum_krr"], slot["nearest_neighbour"], protocol)
        if slot["profile_nearest_neighbour"]["score"]:
            tr.delta_auroc_profile_vs_nearest_neighbour = _paired_delta(
                slot["profile_nearest_neighbour"], slot["nearest_neighbour"],
                protocol)
            tr.delta_auroc_vs_profile = _paired_delta(
                slot["spectrum_krr"], slot["profile_nearest_neighbour"],
                protocol)
            if slot["profile_krr"]["score"]:
                tr.delta_auroc_profile_krr_vs_profile_nn = _paired_delta(
                    slot["profile_krr"], slot["profile_nearest_neighbour"],
                    protocol)
                tr.delta_auroc_profile_krr_vs_spectrum_krr = _paired_delta(
                    slot["profile_krr"], slot["spectrum_krr"], protocol)
        if slot["cofactor_determinant"]["score"]:
            tr.delta_auroc_determinant_vs_nearest_neighbour = _paired_delta(
                slot["cofactor_determinant"], slot["nearest_neighbour"],
                protocol)
        results.append(tr)

    macro: dict[str, dict[str, float | None]] = {}
    for name in COMPARATORS:
        cells = [t.comparators.get(name, {}) for t in results]
        if not any(cells):
            continue
        macro[name] = {m: _mean([c.get(m) for c in cells])
                       for m in ("auroc", "average_precision", "brier", "ece")}
        # A mean over a different number of targets per metric is a trap when
        # the calibrated comparator only exists for some targets: say how many
        # each average is over.
        macro[name]["n_targets_auroc"] = float(sum(
            1 for c in cells if isinstance(c.get("auroc"), (int, float))))
        macro[name]["n_targets_probability"] = float(sum(
            1 for c in cells if isinstance(c.get("brier"), (int, float))))
    reliability: dict[str, list[dict[str, Any]]] = {}
    for name in ("spectrum_krr", "spectrum_krr_calibrated"):
        probs: list[float] = []
        labels_all: list[bool] = []
        for t in keep:
            s = pooled[t][name]
            if any(p is None for p in s["prob"]):
                continue
            probs += [float(p) for p in s["prob"]]
            labels_all += s["label"]
        if probs:
            reliability[name] = reliability_table(probs, labels_all)
    sims = sorted(max_sims)
    similarity = {
        "n_test_sequences": len(sims),
        "median": sims[len(sims) // 2] if sims else None,
        "p10": sims[int(0.1 * len(sims))] if sims else None,
        "p90": sims[min(len(sims) - 1, int(0.9 * len(sims)))] if sims else None,
        "measure": "max k-mer cosine of a test sequence to any training-fold "
                   "sequence",
    }
    report = BenchmarkReport(
        protocol=protocol, claim=CLAIM, dataset_md5={},
        n_sequences=len(ids), n_clusters=len(set(assignment.values())),
        clustering=dict(clustering_note or {"method": "none (every sequence "
                                                       "its own group)"}),
        targets=results, skipped_targets=skipped, macro=macro,
        reliability=reliability, similarity_to_training=similarity,
        uncalibrated_targets=sorted(uncalibrated),
        kernel_character=composition_contrast(
            [sequences[i] for i in ids], protocol.ks, seed=protocol.seed),
        alignment_similarity=(_alignment_record(alignment)
                              if alignment is not None else {}),
        alignment_kernel=(
            {"kernel_id": getattr(alignment_kernel, "kernel_id",
                                  type(alignment_kernel).__name__),
             "ridge": protocol.ridge,
             "cholesky_refusal": kernel_refusal,
             "scored": kernel_refusal is None}
            if alignment_kernel is not None else {}),
        determinant_rules=determinant.rule_records(),
        determinant_outcomes={t: dict(sorted(c.items()))
                              for t, c in sorted(determinant.outcomes.items())})
    report.digest = sha256_obj({k: v for k, v in report.to_dict().items()
                                if k != "digest"})
    return report


@dataclass
class SdrSuite:
    """The grouped run, the deliberately leaky run, and a ridge sensitivity sweep."""

    grouped: BenchmarkReport
    ungrouped: BenchmarkReport
    ridge_sweep: dict[float, dict[str, float | None]]

    def render(self) -> str:
        lines = [self.grouped.render(), "",
                 "SAME RUN WITH UNGROUPED FOLDS (leakage on purpose):"]
        names = ["nearest_neighbour", "spectrum_krr_calibrated"]
        if "profile_nearest_neighbour" in self.grouped.macro:
            # The stronger a homology baseline is, the more a near-duplicate
            # straddling the split should flatter it. Leaving the alignment
            # column out of the leakage control would hide exactly that, and
            # the same question is worth asking of the head on that kernel.
            names.insert(1, "profile_nearest_neighbour")
        if "profile_krr_calibrated" in self.grouped.macro:
            names.append("profile_krr_calibrated")
        for name in names:
            g = self.grouped.macro[name]["auroc"]
            u = self.ungrouped.macro[name]["auroc"]
            lines.append(f"  macro AUROC {name:<26} grouped={_f(g)} "
                         f"ungrouped={_f(u)} "
                         f"optimism={_f(None if g is None or u is None else u - g)}")
        lines.append(f"  median max-similarity to training: "
                     f"grouped={_f(self.grouped.similarity_to_training['median'])} "
                     f"ungrouped={_f(self.ungrouped.similarity_to_training['median'])}")
        lines += ["", "RIDGE SENSITIVITY (a sweep, not a selection; "
                      "the protocol's ridge was fixed in advance):"]
        for ridge, m in sorted(self.ridge_sweep.items()):
            lines.append(f"  ridge={ridge:<6} macro AUROC={_f(m['auroc'])} "
                         f"Brier={_f(m['brier'])} ECE={_f(m['ece'])}")
        return "\n".join(lines)


def sdr_determinant_rules(template_dir: str | Path | None = None
                          ) -> list[AnchoredDeterminantRule]:
    """The anchored cofactor rules the shipped SDR family template declares.

    Read from the template library rather than written here, so that the rule
    the benchmark scores is the same object the pipeline would apply, and a
    change to one moves the other.
    """
    from ..harness.templates import TemplateLibrary, default_template_dir
    from ..science.cofactor_determinant import rules_from_template

    library = TemplateLibrary.load(Path(template_dir or default_template_dir()))
    template = library.get(SDR_FAMILY_TEMPLATE_ID)
    if template is None:
        return []
    return rules_from_template(template)


def sdr_alignment_similarity(examples: Sequence[Example], **kwargs: Any
                             ) -> PairwiseSimilarity:
    """A profile-search similarity over every sequence in ``examples``.

    The pool is the whole dataset, because a test sequence is a query and
    every training sequence is a target, and a provider asked for a target it
    never searched refuses rather than returning zero. Raises
    :class:`~eagent.errors.ToolUnavailableError` when the ``search`` extra is
    not installed -- the caller decides what to do about that, and the
    benchmark runs without the column rather than with a substitute.
    """
    from ..science.alignment_similarity import PhmmerSimilarity, pool_from_examples

    return PhmmerSimilarity(pool_from_examples(examples), **kwargs)


def run_sdr_suite(data_dir: str | Path, protocol: BenchmarkProtocol | None = None,
                  *, sweep: Sequence[float] = (0.03, 0.1, 0.3, 1.0),
                  determinant_rules: Sequence[AnchoredDeterminantRule] | None = None,
                  alignment: PairwiseSimilarity | None = None,
                  alignment_kernel: Any | None = None
                  ) -> SdrSuite:
    """Load the SDR dataset, cluster it, and run the grouped, ungrouped and sweep runs.

    One ``alignment`` provider serves every run, so each sequence's profile is
    searched once for the whole suite rather than once per run, and
    ``alignment_kernel`` (built over the same provider) adds the head refitted
    on it. The whole-pool definiteness check runs once and is recorded on the
    grouped report.
    """
    from ..tools.mine_sequences import cluster_sequences
    protocol = protocol or BenchmarkProtocol()
    rules = (list(determinant_rules) if determinant_rules is not None
             else sdr_determinant_rules())
    dataset = load_sdr(data_dir)
    examples = sdr_examples(dataset)
    items = sorted({(e.example_id, e.sequence) for e in examples})
    clustering = cluster_sequences(items, protocol.identity_threshold)
    note = {"method": clustering.method,
            "identity_threshold": clustering.identity_threshold,
            "identity_definition": clustering.identity_definition,
            "fallback_reason": clustering.fallback_reason,
            "n_alignments": clustering.n_alignments}
    kernel = KernelCache(protocol.ks)

    def run(proto: BenchmarkProtocol) -> BenchmarkReport:
        report = run_benchmark(examples, proto,
                               cluster_assignment=dict(clustering.assignment),
                               clustering_note=note, kernel=kernel,
                               determinant_rules=rules, alignment=alignment,
                               alignment_kernel=alignment_kernel)
        report.dataset_md5 = dict(dataset.file_md5)
        return report

    grouped = run(protocol)
    if alignment_kernel is not None:
        from ..science.enzyme_substrate import definiteness_report
        grouped.alignment_kernel["definiteness"] = definiteness_report(
            [e.sequence for e in examples], alignment_kernel,
            ridge=protocol.ridge, seed=protocol.seed)
        grouped.digest = sha256_obj({k: v for k, v in grouped.to_dict().items()
                                     if k != "digest"})
    ungrouped = run(BenchmarkProtocol(**{**protocol.__dict__, "grouped": False}))
    sweep_out: dict[float, dict[str, float | None]] = {}
    for ridge in sweep:
        r = run(BenchmarkProtocol(**{**protocol.__dict__, "ridge": ridge,
                                     "n_boot": 20}))
        sweep_out[ridge] = r.macro["spectrum_krr_calibrated"]
    return SdrSuite(grouped, ungrouped, sweep_out)


def _mean(values: Sequence[Any]) -> float | None:
    xs = [v for v in values if isinstance(v, (int, float))]
    return sum(xs) / len(xs) if xs else None


def _paired_delta(a: Mapping[str, list], b: Mapping[str, list],
                  protocol: BenchmarkProtocol) -> dict[str, Any] | None:
    """AUROC(a) - AUROC(b) with a cluster bootstrap over the *same* resamples."""
    labels, clusters = a["label"], a["cluster"]
    base_a, base_b = auroc(a["score"], labels), auroc(b["score"], labels)
    if base_a is None or base_b is None:
        return None
    by_cluster: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        by_cluster.setdefault(c, []).append(i)
    names = sorted(by_cluster)
    rng = random.Random(protocol.seed + 1)
    deltas: list[float] = []
    for _ in range(protocol.n_boot):
        idx: list[int] = []
        for _ in names:
            idx.extend(by_cluster[names[rng.randrange(len(names))]])
        da = auroc([a["score"][i] for i in idx], [labels[i] for i in idx])
        db = auroc([b["score"][i] for i in idx], [labels[i] for i in idx])
        if da is not None and db is not None:
            deltas.append(da - db)
    ci = None
    if len(deltas) >= max(20, protocol.n_boot // 5):
        deltas.sort()
        ci = [deltas[int(0.025 * len(deltas))],
              deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]]
    return {"delta": base_a - base_b, "ci95": ci,
            "interval_excludes_zero": bool(ci and (ci[0] > 0 or ci[1] < 0))}
