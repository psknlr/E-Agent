"""How much a conclusion depends on one lucky pose.

A modelling pipeline can always produce *a* pose in which the catalytic atoms
are where the mechanism wants them. Sample enough conformers, restrain the
right distance, and a satisfying arrangement appears for almost any enzyme and
almost any substrate. The question that actually discriminates is how *rare*
that arrangement was, how much of it was imposed rather than found, and whether
an independent modelling route reproduces it.

This module supplies the three answers:

* :func:`pose_robustness` and :func:`classify_robustness` -- the fraction of
  valid poses that satisfy the mechanism conditions, with an honest statement
  of how little a fraction over five poses means;
* :class:`CircularityGuard` -- the anti-self-justification device: a distance
  that was *enforced* during modelling cannot afterwards corroborate the model
  that enforced it;
* :func:`cross_method_agreement` -- reward for two genuinely independent
  modelling routes concurring, and a CONTRADICTORY flag when they do not.

Nothing here reads coordinates. It consumes
:class:`~eagent.schemas.candidate.GeometryReport` objects that the geometry
layer already produced against a sourced template, so no catalytic threshold
lives in this file.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..errors import CircularEvidenceError
from ..schemas import ComplexPose, ConfidenceLevel, GeometryReport

__all__ = [
    "DEFAULT_MIN_VALID_POSES",
    "DEFAULT_STRONG_LOWER_BOUND",
    "DEFAULT_MODERATE_LOWER_BOUND",
    "DEFAULT_WILSON_Z",
    "pose_robustness",
    "classify_robustness",
    "wilson_interval",
    "bootstrap_fraction_ci",
    "IndependentEvidence",
    "CircularityGuard",
    "cross_method_verdicts",
    "cross_method_agreement",
]


# --------------------------------------------------------------------------
# Named defaults.
#
# None of these is a catalytic threshold: they describe how much *sampling*
# is needed before a fraction means anything, which is a property of the
# modelling protocol, not of the enzyme family. Catalytic windows live in a
# sourced CatalyticTemplate's GeometryConstraint objects and are applied by the
# geometry layer before anything reaches this module. All four are overridable
# per call and all four need per-protocol calibration -- a docking run that
# emits 200 poses and a complex predictor that emits 5 do not deserve the same
# minimum sample.
# --------------------------------------------------------------------------

#: Fewest valid poses from which a robustness fraction is allowed to be
#: classified at all. Below this, :func:`classify_robustness` returns
#: INSUFFICIENT regardless of the fraction: 1/1 and 0/1 are both noise.
#: CALIBRATION: set from the sampler's output size and its pose diversity.
DEFAULT_MIN_VALID_POSES: int = 5

#: Wilson lower bound at or above which agreement is called STRONG.
#: CALIBRATION: per modelling protocol. The lower bound rather than the point
#: estimate is used deliberately -- 5/5 and 50/50 are the same fraction and very
#: different evidence.
DEFAULT_STRONG_LOWER_BOUND: float = 0.50

#: Wilson lower bound at or above which agreement is called MODERATE.
DEFAULT_MODERATE_LOWER_BOUND: float = 0.20

#: Two-sided 95% normal quantile, the default for :func:`wilson_interval`.
DEFAULT_WILSON_Z: float = 1.959963984540054


# --------------------------------------------------------------------------
# The robustness fraction
# --------------------------------------------------------------------------

def pose_robustness(n_satisfying: int, n_valid: int) -> float | None:
    """``G(E, S) = satisfying poses / valid poses``.

    WHAT G IS NOT -- this is the part that gets misread
    ---------------------------------------------------
    G is **not** a probability of catalysis. It is not a yield, a conversion, a
    rate, or a confidence. It is one number describing one thing: of the poses
    this particular protocol generated and accepted as structurally valid, what
    share placed the catalytic atoms inside the template's windows. Change the
    sampler, the seed, the number of output poses or the validity filter and G
    changes without a single atom of the enzyme changing.

    In particular:

    * **G = 0 with n_valid = 0 is a modelling failure, not an experimental
      negative.** That case returns ``None``, never ``0.0``, precisely so it
      cannot be averaged, ranked or plotted alongside a real zero. An enzyme for
      which the pipeline produced no valid pose has not been shown to be
      inactive; it has not been tested.
    * **G = 0 with a healthy n_valid** means the sampler found nothing
      satisfying. That is informative, but it is still a statement about the
      sampler's reach, not about the enzyme's chemistry.
    * **G = 1.0 from two poses** is almost no evidence at all. Use
      :func:`wilson_interval` or :func:`classify_robustness` rather than the bare
      fraction.

    Returns
    -------
    The fraction, or ``None`` when ``n_valid == 0``.

    Raises
    ------
    ValueError
        On negative counts, or ``n_satisfying > n_valid`` -- an impossible
        bookkeeping state that usually means satisfying poses were counted from
        a different pose set than the validity filter was applied to.
    """
    if n_satisfying < 0 or n_valid < 0:
        raise ValueError(
            f"pose counts cannot be negative: n_satisfying={n_satisfying}, "
            f"n_valid={n_valid}"
        )
    if n_satisfying > n_valid:
        raise ValueError(
            f"n_satisfying ({n_satisfying}) exceeds n_valid ({n_valid}); the two "
            f"counts were taken over different pose sets"
        )
    if n_valid == 0:
        return None
    return n_satisfying / n_valid


def wilson_interval(
    successes: int,
    trials: int,
    z: float = DEFAULT_WILSON_Z,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, in pure Python.

    WHY WILSON AND NOT THE TEXTBOOK NORMAL INTERVAL
    -----------------------------------------------
    G is routinely computed from a handful of poses and is routinely 0 or 1. The
    Wald interval ``p +/- z*sqrt(p(1-p)/n)`` collapses to zero width at exactly
    those two values, so 5/5 would report "100%, +/- 0" -- a precise claim from
    five samples. The Wilson interval stays finite there (5/5 at 95% gives about
    [0.57, 1.0]), which is the honest statement: five for five is encouraging and
    is also compatible with a true rate of 60%.

    This is a *sampling* interval over the poses the protocol produced. It does
    not model error in the force field, the template windows, the protonation
    state or the structure itself, all of which are larger than it. Reporting it
    narrows nothing; it only stops a bare fraction being read as exact.

    Returns
    -------
    ``(lower, upper)``, clamped to [0, 1]. The interval always brackets the
    point estimate ``successes / trials``.

    Raises
    ------
    ValueError
        If ``trials <= 0`` (there is no interval around a fraction that does not
        exist -- see :func:`pose_robustness`, which returns ``None`` there), if
        ``successes`` is out of range, or if ``z <= 0``.
    """
    if trials <= 0:
        raise ValueError(
            "wilson_interval needs at least one trial; zero valid poses is a "
            "modelling failure with no interval, not a proportion of 0"
        )
    if successes < 0 or successes > trials:
        raise ValueError(f"successes={successes} out of range for trials={trials}")
    if z <= 0:
        raise ValueError(f"z must be positive, got {z}")

    n = float(trials)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def classify_robustness(
    G: float | None,
    n_valid: int,
    *,
    min_valid_poses: int = DEFAULT_MIN_VALID_POSES,
    strong_lower_bound: float = DEFAULT_STRONG_LOWER_BOUND,
    moderate_lower_bound: float = DEFAULT_MODERATE_LOWER_BOUND,
    z: float = DEFAULT_WILSON_Z,
) -> ConfidenceLevel:
    """Turn a robustness fraction into an ordinal confidence level.

    WHY ORDINAL, AND WHY THE LOWER BOUND
    ------------------------------------
    The scorecard has no weighted total (see
    :mod:`eagent.schemas.candidate`), so this axis contributes a level, not a
    number to be multiplied by 0.3. The level is read from the Wilson *lower*
    bound rather than from G itself, because the point estimate cannot tell 1/1
    from 40/40 and those are not the same evidence.

    ``INSUFFICIENT`` is returned whenever the sample cannot distinguish
    anything: no valid poses at all, or fewer than ``min_valid_poses``. This is
    the case that must not be confused with a negative -- a candidate whose
    docking run produced three poses has not failed a robustness test, it has
    not taken one, and ranking it below a candidate with a genuine low G would
    be reading a modelling budget as chemistry.

    ``CONTRADICTORY`` is never returned here. Contradiction is a property of
    disagreeing *methods*, not of one method's pose fraction; see
    :func:`cross_method_agreement`.

    Parameters
    ----------
    min_valid_poses:
        Named and documented because it is the knob that decides what counts as
        "tested". Needs per-protocol calibration; see
        :data:`DEFAULT_MIN_VALID_POSES`.
    strong_lower_bound, moderate_lower_bound:
        Lower-bound cut points on the Wilson interval. Per-protocol, not
        per-enzyme-family, and both overridable.
    """
    if min_valid_poses < 1:
        raise ValueError(f"min_valid_poses must be >= 1, got {min_valid_poses}")
    if not (0.0 <= moderate_lower_bound <= strong_lower_bound <= 1.0):
        raise ValueError(
            "require 0 <= moderate_lower_bound <= strong_lower_bound <= 1, got "
            f"{moderate_lower_bound} and {strong_lower_bound}"
        )
    if n_valid < 0:
        raise ValueError(f"n_valid cannot be negative, got {n_valid}")
    if G is None or n_valid == 0:
        return ConfidenceLevel.INSUFFICIENT
    if not (0.0 <= G <= 1.0):
        raise ValueError(f"G must lie in [0, 1], got {G}")
    if n_valid < min_valid_poses:
        return ConfidenceLevel.INSUFFICIENT

    successes = int(round(G * n_valid))
    successes = max(0, min(n_valid, successes))
    lower, _ = wilson_interval(successes, n_valid, z=z)
    if lower >= strong_lower_bound:
        return ConfidenceLevel.STRONG
    if lower >= moderate_lower_bound:
        return ConfidenceLevel.MODERATE
    return ConfidenceLevel.WEAK


def bootstrap_fraction_ci(
    flags: Sequence[bool],
    n_resamples: int = 2000,
    seed: int = 0,
    *,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    """Percentile bootstrap interval for the satisfied-pose fraction.

    Complements :func:`wilson_interval` rather than replacing it. Wilson assumes
    independent Bernoulli draws; poses from one docking run are correlated
    (clustered conformers, shared starting structure), so neither interval is
    "the" truth. Reporting the bootstrap alongside makes the resampling
    assumption explicit instead of hiding it inside a closed-form formula.

    Determinism is required, not optional: a candidate's reported interval must
    not move between two runs of the same manifest, so the generator is a local
    :class:`random.Random` seeded from ``seed`` -- never the global
    :mod:`random` state, which another component could advance.

    Returns
    -------
    ``(lower, upper)``, or ``None`` when ``flags`` is empty. Empty means no
    valid pose was produced; as in :func:`pose_robustness`, that is a modelling
    failure and must not be rendered as the interval [0, 0].
    """
    items = list(flags)
    if not items:
        return None
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")

    n = len(items)
    rng = random.Random(seed)
    fractions: list[float] = []
    for _ in range(n_resamples):
        hits = 0
        for _ in range(n):
            if items[rng.randrange(n)]:
                hits += 1
        fractions.append(hits / n)
    fractions.sort()
    lo_idx = int(math.floor((alpha / 2.0) * n_resamples))
    hi_idx = int(math.ceil((1.0 - alpha / 2.0) * n_resamples)) - 1
    lo_idx = max(0, min(n_resamples - 1, lo_idx))
    hi_idx = max(lo_idx, min(n_resamples - 1, hi_idx))
    return (fractions[lo_idx], fractions[hi_idx])


# --------------------------------------------------------------------------
# Circularity
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class IndependentEvidence:
    """Partition of one geometry report into circular and independent evidence."""

    satisfied: tuple[str, ...]
    unsatisfied: tuple[str, ...]
    unmeasured: tuple[str, ...]
    circular_satisfied: tuple[str, ...]
    circular_all: tuple[str, ...]

    @property
    def n_independent_total(self) -> int:
        """Independent constraints that were actually evaluated, measured or not."""
        return len(self.satisfied) + len(self.unsatisfied) + len(self.unmeasured)

    @property
    def n_independent_satisfied(self) -> int:
        return len(self.satisfied)

    @property
    def fraction(self) -> float | None:
        """Satisfied share of the independent constraints, or None if there are none.

        ``None`` rather than ``0.0`` when nothing independent was evaluated: no
        independent evidence is a different state from independent evidence that
        failed, and collapsing them would let a fully circular model rank
        alongside a genuinely tested one.
        """
        measured = len(self.satisfied) + len(self.unsatisfied)
        if measured == 0:
            return None
        return len(self.satisfied) / measured

    @property
    def is_entirely_circular(self) -> bool:
        """True when every satisfied constraint was one that had been imposed."""
        return not self.satisfied and bool(self.circular_satisfied)


class CircularityGuard:
    """Separates constraints that were *imposed* from constraints that were *found*.

    THE TRAP THIS EXISTS TO CLOSE
    -----------------------------
    Template docking and restrained complex prediction both let you enforce
    geometry: hold the nicotinamide C4 within 3.5 A of the carbonyl carbon, keep
    the catalytic tyrosine hydroxyl hydrogen-bonded to the carbonyl oxygen. The
    resulting pose then satisfies exactly those conditions, because it was built
    to. Measuring them afterwards and reporting "4 of 5 catalytic constraints
    satisfied" is not evidence about the enzyme; it is a readback of the input
    file. A distance you enforced cannot afterwards corroborate the model that
    enforced it.

    The failure is seductive because it is invisible in every artefact
    downstream: the numbers are real measurements of real coordinates, the
    report looks identical to an honest one, and the candidate ranks well. The
    only place the distinction survives is
    :attr:`~eagent.schemas.candidate.ComplexPose.restrained_constraints`, which
    is why that field is recorded at modelling time and consulted here.

    An independent constraint is one that was *not* restrained and still came
    out inside its window -- a geometric coincidence the model was free to get
    wrong. Those are the only ones that count toward
    :attr:`~eagent.schemas.candidate.GeometryReport.independent_satisfied`.

    Parameters
    ----------
    restrained:
        Constraint names enforced while building the pose, normally
        ``pose.restrained_constraints``.
    evaluated:
        Constraint names being checked afterwards, normally the names in the
        catalytic template.
    """

    def __init__(self, restrained: Iterable[str], evaluated: Iterable[str]) -> None:
        self.restrained: frozenset[str] = frozenset(str(n) for n in restrained)
        self.evaluated: frozenset[str] = frozenset(str(n) for n in evaluated)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_pose(
        cls, pose: ComplexPose, evaluated: Iterable[str]
    ) -> "CircularityGuard":
        """Build from a pose's recorded restraints.

        Reads ``restrained_constraints`` off the pose rather than taking the
        caller's word for what was enforced, so the guard cannot be defeated by
        a caller that simply forgets to pass the restraint list.
        """
        return cls(pose.restrained_constraints, evaluated)

    @classmethod
    def for_report(
        cls, pose: ComplexPose, report: GeometryReport
    ) -> "CircularityGuard":
        """Build from a pose plus the constraint names that report actually covers."""
        return cls(pose.restrained_constraints, report.satisfied.keys())

    # -- partition ---------------------------------------------------------
    @property
    def circular(self) -> frozenset[str]:
        """Evaluated constraints that were also restrained: self-fulfilling."""
        return self.evaluated & self.restrained

    @property
    def independent(self) -> frozenset[str]:
        """Evaluated constraints that were not restrained: genuinely testable."""
        return self.evaluated - self.restrained

    def partition(self) -> tuple[frozenset[str], frozenset[str]]:
        """``(circular, independent)`` over the evaluated set."""
        return self.circular, self.independent

    @property
    def restrained_but_not_evaluated(self) -> frozenset[str]:
        """Restraints with no matching evaluated constraint.

        Usually a naming mismatch between the modelling run and the template. It
        matters because a restraint whose name does not match is a restraint the
        guard cannot see, so a non-empty set here means the circularity check may
        be under-counting and the caller should raise a QC flag rather than trust
        the partition.
        """
        return self.restrained - self.evaluated

    # -- evidence ----------------------------------------------------------
    def independent_evidence(self, geometry_report: GeometryReport) -> IndependentEvidence:
        """Split one report's outcomes into independent and circular buckets.

        Only constraints present in both the report and the guard's evaluated set
        are considered: a constraint the report never measured cannot be counted
        as satisfied *or* as failed, and is listed under ``unmeasured`` so the gap
        stays visible instead of being silently dropped from the denominator.
        """
        satisfied: list[str] = []
        unsatisfied: list[str] = []
        unmeasured: list[str] = []
        circular_satisfied: list[str] = []
        circular_all: list[str] = []

        for name, outcome in geometry_report.satisfied.items():
            key = str(name)
            if key not in self.evaluated:
                continue
            if key in self.restrained:
                circular_all.append(key)
                if outcome is True:
                    circular_satisfied.append(key)
                continue
            if outcome is True:
                satisfied.append(key)
            elif outcome is False:
                unsatisfied.append(key)
            else:
                unmeasured.append(key)

        return IndependentEvidence(
            satisfied=tuple(sorted(satisfied)),
            unsatisfied=tuple(sorted(unsatisfied)),
            unmeasured=tuple(sorted(unmeasured)),
            circular_satisfied=tuple(sorted(circular_satisfied)),
            circular_all=tuple(sorted(circular_all)),
        )

    def raise_if_only_circular(
        self,
        geometry_report: GeometryReport,
        *,
        allow_no_evidence: bool = True,
        subject: str = "",
    ) -> IndependentEvidence:
        """Raise when every satisfied constraint was one that had been imposed.

        This is the check that stops a restrained pose from being promoted as a
        geometric success. It fires when there is at least one satisfied
        constraint and *all* of them were restrained -- the exact shape of a
        model corroborating itself.

        ``allow_no_evidence=True`` (the default) means a report where nothing at
        all was satisfied does **not** raise. That report is a failed geometry
        check, which the gating layer already handles; it makes no circular claim
        because it makes no claim. Pass ``allow_no_evidence=False`` in contexts
        where the absence of any independent evidence is itself disqualifying,
        for example before writing a candidate into a synthesis batch.

        Returns the :class:`IndependentEvidence` partition so the caller does not
        have to compute it twice.

        Raises
        ------
        ~eagent.errors.CircularEvidenceError
        """
        ev = self.independent_evidence(geometry_report)
        where = f" for {subject}" if subject else ""
        if ev.is_entirely_circular:
            raise CircularEvidenceError(
                f"pose '{geometry_report.pose_id}'{where}: every satisfied "
                f"constraint {list(ev.circular_satisfied)} was restrained during "
                f"modelling. A distance that was enforced cannot corroborate the "
                f"model that enforced it; independent constraints evaluated: "
                f"{sorted(self.independent) or 'none'}."
            )
        if not allow_no_evidence and ev.n_independent_satisfied == 0:
            raise CircularEvidenceError(
                f"pose '{geometry_report.pose_id}'{where}: no independent "
                f"constraint was satisfied, so the pose carries no evidence that "
                f"was not imposed on it."
            )
        return ev

    def annotate(self, geometry_report: GeometryReport) -> GeometryReport:
        """Return a copy of the report with the circularity bookkeeping filled in.

        A copy rather than an in-place edit: the raw report is the measurement
        record, and the independent-evidence counts are an interpretation layered
        on top of it. Keeping them separable means a reviewer can recompute the
        interpretation from the measurements without having to trust that it was
        not overwritten.
        """
        ev = self.independent_evidence(geometry_report)
        return geometry_report.model_copy(update={
            "circular_constraints": list(ev.circular_all),
            "independent_satisfied": ev.n_independent_satisfied,
            "independent_total": len(ev.satisfied) + len(ev.unsatisfied),
        })


# --------------------------------------------------------------------------
# Cross-method agreement
# --------------------------------------------------------------------------

def _verdict_of(value: Any) -> bool | None:
    """Normalise one method's output to a three-valued verdict.

    ``True`` = at least one pose from this method passed the gate, ``False`` =
    poses were produced and none passed, ``None`` = the method did not decide
    (no poses, gate unevaluated). The three-valued form is kept all the way
    through, because folding ``None`` into ``False`` would turn "the predictor
    did not run" into "the predictor disagrees" -- manufacturing a contradiction
    out of a missing result.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, GeometryReport):
        return value.gating_passed
    if isinstance(value, (list, tuple, set, frozenset)):
        outcomes = [_verdict_of(v) for v in value]
        if any(o is True for o in outcomes):
            return True
        if any(o is False for o in outcomes):
            return False
        return None
    raise TypeError(
        f"cannot read a verdict from {type(value).__name__}; supply a bool, None, "
        f"a GeometryReport, or a sequence of them"
    )


def cross_method_verdicts(
    reports_by_method: Mapping[str, Any],
) -> dict[str, bool | None]:
    """Per-method three-valued verdicts, exposed so the *direction* is visible.

    :func:`cross_method_agreement` returns only how strongly the routes concur,
    not what they concur on. Agreement on a negative is as strong as agreement on
    a positive, and a single ``ConfidenceLevel`` cannot say which it was. This
    helper hands the caller the verdicts so the two facts stay separate instead
    of being conflated into one misread number.
    """
    return {str(m): _verdict_of(v) for m, v in reports_by_method.items()}


def cross_method_agreement(
    reports_by_method: Mapping[str, Any],
    *,
    min_methods: int = 2,
    independent_groups: Mapping[str, str] | None = None,
) -> ConfidenceLevel:
    """Confidence from the concurrence of independent modelling routes.

    WHY CROSS-METHOD AND NOT MORE POSES
    -----------------------------------
    Running the same docking program with more seeds reduces sampling noise and
    nothing else: every pose inherits the same scoring function, the same
    protonation assumptions and the same rigid-receptor approximation, so a
    systematic error is reproduced, not detected. Two routes built on different
    assumptions -- template docking against an experimental complex versus joint
    complex prediction from sequence -- can fail independently, so their
    agreement is informative in a way that 500 poses from one of them is not.

    INDEPENDENCE IS ASSERTED, NOT INFERRED
    --------------------------------------
    Two entries in ``reports_by_method`` are treated as independent routes only
    if they map to different groups. By default each method name is its own
    group, which is wrong exactly when the caller passes ``"vina_run1"`` and
    ``"vina_run2"``; pass ``independent_groups`` to collapse reruns of one
    program into one route. Over-counting reruns as independent agreement is the
    easiest way to manufacture STRONG confidence out of nothing.

    WHAT THE RETURNED LEVEL MEANS
    -----------------------------
    The *strength of the cross-method evidence*, not its direction. Routes that
    unanimously say "gate failed" return STRONG just as unanimous passes do:
    they strongly agree. Read the direction from
    :func:`cross_method_verdicts`, which is why that helper exists.

    Ladder:

    * ``CONTRADICTORY`` -- some route passed, another failed. Not resolved by
      majority: a disagreement between independent routes is a finding, and
      overruling it with a vote discards the only signal that something is
      wrong.
    * ``STRONG`` -- at least ``min_methods`` independent routes decided, all
      agreed, and no route was left undecided.
    * ``MODERATE`` -- at least ``min_methods`` agreed, but at least one route
      could not decide. An undecided route is a gap, so the agreement is not
      complete.
    * ``WEAK`` -- exactly one route decided; nothing corroborates it.
    * ``INSUFFICIENT`` -- no route decided.
    """
    if min_methods < 2:
        raise ValueError(
            f"cross-method agreement needs at least two routes to be a "
            f"cross-method claim; got min_methods={min_methods}"
        )
    verdicts = cross_method_verdicts(reports_by_method)
    groups = dict(independent_groups or {})

    decided: dict[str, set[bool]] = {}
    undecided_groups: set[str] = set()
    for method, verdict in verdicts.items():
        group = str(groups.get(method, method))
        if verdict is None:
            undecided_groups.add(group)
        else:
            decided.setdefault(group, set()).add(verdict)

    undecided_groups -= set(decided)

    if not decided:
        return ConfidenceLevel.INSUFFICIENT
    all_outcomes: set[bool] = set()
    for outcomes in decided.values():
        all_outcomes |= outcomes
    if len(all_outcomes) > 1:
        return ConfidenceLevel.CONTRADICTORY
    if len(decided) < min_methods:
        return ConfidenceLevel.WEAK
    if undecided_groups:
        return ConfidenceLevel.MODERATE
    return ConfidenceLevel.STRONG
