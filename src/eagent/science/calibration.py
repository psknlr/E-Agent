"""Turning a reference set into a window a template may cite -- or refusing to.

THE PROBLEM THIS EXISTS FOR
===========================
Every geometric window in the shipped catalytic templates is uncalibrated: 17
of 17, and none gates. That is the honest state, and it is not a state anyone
can leave by editing a YAML file, because ``calibrated_on`` was a free-text
list: ``calibrated_on: ["trust me"]`` makes ``is_calibrated`` true and gives a
window the power to reject enzymes. The authority to reject was one hand-typed
string away from being granted, and nothing checked it.

A calibration is a *claim about a population*, so it needs evidence that can
be checked and a rule for how much evidence is enough. This module is both.

WHAT A CALIBRATION IS HERE
==========================
A window is proposed from reference complexes whose activity is known from
somewhere other than the model that is about to be judged: ``active`` ones,
where the chemistry was shown to occur in that system, and ``inactive`` ones,
where it was shown not to. Each carries the evidence for its label. The
measurement is taken from the coordinates by the same code that later judges a
candidate, never typed in.

The window is the range of the active complexes, and the claim it supports is
the one an order-statistic **tolerance interval** can support, without any
distributional assumption (Wilks, 1941): with *n* actives, the interval
[min, max] covers at least a fraction *p* of the population with confidence

    gamma(n, p) = 1 - n p^(n-1) + (n-1) p^n

So the record says exactly what its sample size buys. Seven actives give a 90%
coverage claim with a confidence of about 0.15 -- which is a number, and it is
low, and the window does not get to reject anything on the strength of it.

THE POLICY IS A POLICY
======================
How much coverage, at what confidence, and how many known inactives may fall
inside the window before it stops being discriminating are choices about how
much risk of wrongly rejecting an enzyme a campaign accepts. They are not
measured quantities, and :class:`CalibrationPolicy` says so in its ``source``
field, which is required. The record carries the policy it was judged under, so
a stricter policy later re-judges the same evidence instead of inheriting a
verdict.

NOT FORGEABLE
=============
A passing calibration has a digest. A template cites it as ``calibration:<hex>``
in ``calibrated_on``; :class:`CalibrationStore` resolves that entry to the
stored record, **recomputes** the digest from the record's contents, **re-runs**
the verdict rather than reading the stored one, and checks that the window in
the template is the window the record proposed. Editing the record, editing the
template's window afterwards, or citing a digest that does not exist all fail
closed to *uncalibrated*.

What this does not and cannot do: decide whether the reference complexes are
representative of the candidates, whether their labels are right, or whether a
heavy-atom distance is the right observable. It makes the sample size, the
coverage claim and the discrimination of the window explicit and tamper-evident.
That is a precondition for trusting a window, and it is not the trust.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..errors import EAgentError
from ..provenance import sha256_obj, utc_now
from .robustness import wilson_interval

__all__ = [
    "CalibrationError",
    "CALIBRATION_PREFIX",
    "CALIBRATION_ALGORITHM",
    "CalibrationPolicy",
    "ReferenceObservation",
    "Exclusion",
    "CalibrationRecord",
    "CalibrationVerdict",
    "wilks_two_sided_confidence",
    "coverage_at_confidence",
    "minimum_actives",
    "calibrate",
    "CalibrationStore",
    "CalibrationStatus",
    "CalibrationContext",
]


class CalibrationError(EAgentError):
    """A calibration was malformed, tampered with, or does not match its window."""


#: Prefix of the ``calibrated_on`` entry that a store can verify. Anything
#: without it keeps the old meaning -- free text naming systems -- and is not
#: promoted or demoted by this module.
CALIBRATION_PREFIX: str = "calibration:"

#: Identifies the recipe, written into every record. A change to how a window
#: is derived or judged is a new algorithm id, so an old record is re-judged
#: by the rule it was made under rather than by the current one.
CALIBRATION_ALGORITHM: str = "wilks-range-v1"


# ==========================================================================
# the statistics, in pure Python
# ==========================================================================

def wilks_two_sided_confidence(n: int, coverage: float) -> float:
    """Confidence that [min, max] of ``n`` samples covers >= ``coverage`` of the population.

    Distribution-free. ``1 - n p^(n-1) + (n-1) p^n``: the probability that the
    population mass between the sample extremes is at least ``p``. Defined for
    ``n >= 2``; fewer samples have no range, so the confidence is ``0``.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must be in (0, 1), got {coverage}")
    if n < 2:
        return 0.0
    p = coverage
    return max(0.0, min(1.0, 1.0 - n * p ** (n - 1) + (n - 1) * p ** n))


def coverage_at_confidence(n: int, confidence: float) -> float:
    """The largest coverage ``p`` that ``n`` samples support at ``confidence``.

    Inverse of :func:`wilks_two_sided_confidence` in ``p``, by bisection: the
    confidence is monotone decreasing in ``p``. ``0.0`` when even a vanishing
    coverage cannot be claimed (``n < 2``).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n < 2:
        return 0.0
    lo, hi = 1e-9, 1.0 - 1e-9
    if wilks_two_sided_confidence(n, lo) < confidence:
        return 0.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if wilks_two_sided_confidence(n, mid) >= confidence:
            lo = mid
        else:
            hi = mid
    return lo


def minimum_actives(coverage: float, confidence: float, *, limit: int = 100000) -> int:
    """Fewest active references that support ``coverage`` at ``confidence``."""
    n = 2
    while n <= limit:
        if wilks_two_sided_confidence(n, coverage) >= confidence:
            return n
        n += 1
    raise CalibrationError(
        f"no sample size up to {limit} supports coverage {coverage} at "
        f"confidence {confidence}")


# ==========================================================================
# inputs
# ==========================================================================

@dataclass(frozen=True)
class CalibrationPolicy:
    """How much evidence a window needs before it may reject anything.

    Every field is a risk the campaign chooses to accept; none is measured.
    ``source`` is required for that reason: a policy with no stated origin is a
    constant somebody typed, and the record would carry it as though it were
    physics.
    """

    coverage: float
    confidence: float
    #: Known-inactive references must be at least this many, or there is
    #: nothing to say the window discriminates at all.
    min_inactives: int
    #: Upper Wilson bound on the fraction of known inactives falling inside the
    #: window that is still accepted as "discriminating".
    max_inactive_inside_upper: float
    source: str

    def __post_init__(self) -> None:
        if not (self.source or "").strip():
            raise CalibrationError(
                "a calibration policy must state where its numbers came from; "
                "an unsourced policy would be recorded as if it were measured")
        for name, value in (("coverage", self.coverage),
                            ("confidence", self.confidence),
                            ("max_inactive_inside_upper",
                             self.max_inactive_inside_upper)):
            if not 0.0 < value < 1.0:
                raise CalibrationError(f"{name} must be in (0, 1), got {value}")
        if self.min_inactives < 1:
            raise CalibrationError("min_inactives must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {"coverage": self.coverage, "confidence": self.confidence,
                "min_inactives": self.min_inactives,
                "max_inactive_inside_upper": self.max_inactive_inside_upper,
                "source": self.source}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CalibrationPolicy":
        return cls(coverage=float(raw["coverage"]),
                   confidence=float(raw["confidence"]),
                   min_inactives=int(raw["min_inactives"]),
                   max_inactive_inside_upper=float(raw["max_inactive_inside_upper"]),
                   source=str(raw["source"]))


@dataclass(frozen=True)
class ReferenceObservation:
    """One reference complex's measured value and the evidence for its label.

    ``evidence`` is required for both labels. "Active" with no citation is an
    opinion; "inactive" with no citation is how a complex that merely was not
    tested ends up defining where the chemistry does not happen.
    """

    reference_id: str
    label: str                      # "active" | "inactive"
    value: float | None
    evidence: str
    source_type: str = "experimental"       # "experimental" | "modelled"
    #: Constraints that were restrained while this complex was built. A
    #: restrained geometry satisfies what it was built to satisfy, and
    #: calibrating a window on it fits the window to the restraint.
    restrained: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.label not in ("active", "inactive"):
            raise CalibrationError(
                f"{self.reference_id}: label must be 'active' or 'inactive', "
                f"got {self.label!r}")
        if not (self.evidence or "").strip():
            raise CalibrationError(
                f"{self.reference_id}: a {self.label} reference with no "
                f"evidence for the label is an opinion, not a reference")
        if self.value is not None and not math.isfinite(self.value):
            raise CalibrationError(
                f"{self.reference_id}: value {self.value} is not finite")

    def to_dict(self) -> dict[str, Any]:
        return {"reference_id": self.reference_id, "label": self.label,
                "value": self.value, "evidence": self.evidence,
                "source_type": self.source_type,
                "restrained": list(self.restrained)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ReferenceObservation":
        return cls(reference_id=str(raw["reference_id"]), label=str(raw["label"]),
                   value=None if raw.get("value") is None else float(raw["value"]),
                   evidence=str(raw["evidence"]),
                   source_type=str(raw.get("source_type", "experimental")),
                   restrained=tuple(raw.get("restrained") or ()))


@dataclass(frozen=True)
class Exclusion:
    """A reference that was kept out of the calibration, and why.

    Listed in the record rather than dropped: a calibration that quietly lost a
    third of its references cannot be audited, and the reason a reference was
    excluded is often the most informative line in the record.
    """

    reference_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"reference_id": self.reference_id, "reason": self.reason}


# ==========================================================================
# the verdict and the record
# ==========================================================================

@dataclass(frozen=True)
class CalibrationVerdict:
    """What the evidence supports under a given policy."""

    meets_policy: bool
    reasons: tuple[str, ...]
    n_active: int
    n_inactive: int
    n_actives_needed: int
    achieved_confidence: float
    coverage_supported: float
    inactive_inside: int | None
    inactive_inside_upper: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "meets_policy": self.meets_policy, "reasons": list(self.reasons),
            "n_active": self.n_active, "n_inactive": self.n_inactive,
            "n_actives_needed": self.n_actives_needed,
            "achieved_confidence": self.achieved_confidence,
            "coverage_supported": self.coverage_supported,
            "inactive_inside": self.inactive_inside,
            "inactive_inside_upper": self.inactive_inside_upper,
        }


def judge(window: tuple[float, float] | None,
          active: Sequence[float], inactive: Sequence[float],
          policy: CalibrationPolicy) -> CalibrationVerdict:
    """Apply a policy to the evidence. Pure, and re-run on every verification."""
    n_a, n_i = len(active), len(inactive)
    needed = minimum_actives(policy.coverage, policy.confidence)
    achieved = wilks_two_sided_confidence(n_a, policy.coverage)
    supported = coverage_at_confidence(n_a, policy.confidence)
    reasons: list[str] = []
    if n_a < needed:
        reasons.append(
            f"{n_a} active reference(s) support a coverage of "
            f"{supported:.3f} at confidence {policy.confidence}, not the "
            f"{policy.coverage} the policy asks for; {policy.coverage} at "
            f"{policy.confidence} needs {needed}. At the policy's coverage the "
            f"{n_a} give a confidence of only {achieved:.3f}")
    inside: int | None = None
    upper: float | None = None
    if window is None:
        reasons.append("no window could be proposed (fewer than two measured "
                       "active references)")
    elif n_i < policy.min_inactives:
        reasons.append(
            f"{n_i} known-inactive reference(s), fewer than the "
            f"{policy.min_inactives} the policy requires: with too few "
            f"negatives there is nothing to show the window discriminates")
    else:
        lo, hi = window
        inside = sum(1 for v in inactive if lo <= v <= hi)
        _, upper = wilson_interval(inside, n_i)
        if upper > policy.max_inactive_inside_upper:
            reasons.append(
                f"{inside} of {n_i} known-inactive references fall inside "
                f"[{lo:g}, {hi:g}]; the upper Wilson bound on that fraction, "
                f"{upper:.2f}, exceeds the policy's "
                f"{policy.max_inactive_inside_upper}. The window covers the "
                f"actives and does not exclude the inactives, so it would "
                f"accept what it exists to reject")
    return CalibrationVerdict(
        meets_policy=not reasons, reasons=tuple(reasons), n_active=n_a,
        n_inactive=n_i, n_actives_needed=needed, achieved_confidence=achieved,
        coverage_supported=supported, inactive_inside=inside,
        inactive_inside_upper=upper)


@dataclass(frozen=True)
class CalibrationRecord:
    """Everything a calibration rests on, with a digest over it.

    The digest covers the constraint, the algorithm, the policy, every
    included observation and the window. It does **not** cover the verdict or
    the creation time: those are derived, and :meth:`CalibrationStore.verify`
    re-derives the verdict from the evidence instead of reading it, so a
    record whose stored ``meets_policy`` was edited to ``true`` is caught by
    being re-judged, not by being trusted.
    """

    constraint: str
    algorithm: str
    policy: CalibrationPolicy
    observations: tuple[ReferenceObservation, ...]
    exclusions: tuple[Exclusion, ...]
    window: tuple[float, float] | None
    verdict: CalibrationVerdict
    created_at: str = field(default_factory=utc_now)

    @property
    def digest(self) -> str:
        return sha256_obj(self.digest_payload())

    def digest_payload(self) -> dict[str, Any]:
        return {
            "constraint": self.constraint, "algorithm": self.algorithm,
            "policy": self.policy.to_dict(),
            "observations": sorted((o.to_dict() for o in self.observations),
                                   key=lambda d: d["reference_id"]),
            "window": None if self.window is None else list(self.window),
        }

    @property
    def calibrated_on_entry(self) -> str | None:
        """The ``calibrated_on`` string a template may cite, or ``None``.

        ``None`` unless the verdict meets the policy: a template must not be
        handed a citation for a calibration that failed.
        """
        if not self.verdict.meets_policy:
            return None
        return f"{CALIBRATION_PREFIX}{self.digest.removeprefix('sha256:')[:16]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.digest_payload(),
            "digest": self.digest,
            "exclusions": [e.to_dict() for e in self.exclusions],
            "verdict": self.verdict.to_dict(),
            "created_at": self.created_at,
            "calibrated_on_entry": self.calibrated_on_entry,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CalibrationRecord":
        policy = CalibrationPolicy.from_dict(raw["policy"])
        observations = tuple(ReferenceObservation.from_dict(o)
                             for o in raw.get("observations") or ())
        window = raw.get("window")
        window_t = None if window is None else (float(window[0]), float(window[1]))
        active = [o.value for o in observations
                  if o.label == "active" and o.value is not None]
        inactive = [o.value for o in observations
                    if o.label == "inactive" and o.value is not None]
        return cls(
            constraint=str(raw["constraint"]), algorithm=str(raw["algorithm"]),
            policy=policy, observations=observations,
            exclusions=tuple(Exclusion(str(e["reference_id"]), str(e["reason"]))
                             for e in raw.get("exclusions") or ()),
            window=window_t,
            # Re-judged, never read back: see the class docstring.
            verdict=judge(window_t, active, inactive, policy),
            created_at=str(raw.get("created_at") or ""))


# ==========================================================================
# the calibration itself
# ==========================================================================

def calibrate(constraint: str, observations: Iterable[ReferenceObservation],
              policy: CalibrationPolicy) -> CalibrationRecord:
    """Propose a window for ``constraint`` and judge it under ``policy``.

    Three kinds of reference never contribute, each recorded as an
    :class:`Exclusion` with its reason:

    * **modelled** -- a predicted or docked complex. Its geometry is a
      property of the model that produced it, so a window fitted to it is a
      window fitted to that model's biases and then used to judge that
      model's output;
    * **restrained on this constraint** -- built to satisfy what is being
      measured, so it satisfies it by construction;
    * **unmeasured** -- the constraint could not be measured on it, which says
      nothing about where the window should be.

    The proposed window is the range of the measured active references. It is
    proposed even when the policy is not met, because the record of *what the
    evidence would suggest and how little it supports it* is the useful
    output; only a passing record yields a citation.
    """
    included: list[ReferenceObservation] = []
    excluded: list[Exclusion] = []
    seen: set[str] = set()
    for obs in observations:
        if obs.reference_id in seen:
            raise CalibrationError(
                f"reference {obs.reference_id!r} appears twice; counting one "
                f"complex twice inflates the sample size that the confidence "
                f"claim is computed from")
        seen.add(obs.reference_id)
        if obs.source_type != "experimental":
            excluded.append(Exclusion(
                obs.reference_id,
                f"source_type is {obs.source_type!r}, not experimental: a "
                f"modelled complex's geometry belongs to the model"))
        elif constraint in obs.restrained:
            excluded.append(Exclusion(
                obs.reference_id,
                f"{constraint} was restrained while this complex was built, so "
                f"it satisfies the constraint by construction"))
        elif obs.value is None:
            excluded.append(Exclusion(
                obs.reference_id,
                f"{constraint} could not be measured on this complex"))
        else:
            included.append(obs)
    active = [o.value for o in included if o.label == "active"]
    inactive = [o.value for o in included if o.label == "inactive"]
    window = (min(active), max(active)) if len(active) >= 2 else None  # type: ignore[type-var]
    return CalibrationRecord(
        constraint=constraint, algorithm=CALIBRATION_ALGORITHM, policy=policy,
        observations=tuple(sorted(included, key=lambda o: o.reference_id)),
        exclusions=tuple(excluded), window=window,
        verdict=judge(window, active, inactive, policy))  # type: ignore[arg-type]


# ==========================================================================
# the store: what makes a citation checkable
# ==========================================================================

class CalibrationStore:
    """Records on disk, and the check that a template's citation is real.

    A record is written under its digest, so two different calibrations of one
    constraint never overwrite each other and a template's citation names
    exactly one body of evidence.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def _path(self, digest16: str) -> Path:
        return self.directory / f"{digest16}.json"

    def write(self, record: CalibrationRecord) -> Path:
        """Store a record. Failed calibrations are stored too.

        A record that did not meet its policy is evidence about the campaign
        (how far short it fell, and on what), and is kept; it simply has no
        ``calibrated_on`` entry for a template to cite.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        digest16 = record.digest.removeprefix("sha256:")[:16]
        path = self._path(digest16)
        path.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True),
                        encoding="utf-8")
        return path

    def load(self, entry: str) -> CalibrationRecord:
        """Resolve a ``calibration:<hex>`` citation to its record, verified.

        Raises :class:`CalibrationError` for every way a citation can be
        wrong, each with its own message: not a calibration entry, no such
        record, a body that does not match its digest (edited), or a stored
        digest that the contents no longer produce.
        """
        if not entry.startswith(CALIBRATION_PREFIX):
            raise CalibrationError(
                f"{entry!r} is not a calibration citation; free-text "
                f"calibrated_on entries are not verified by this store")
        digest16 = entry.removeprefix(CALIBRATION_PREFIX).strip().lower()
        if not digest16 or any(c not in "0123456789abcdef" for c in digest16):
            raise CalibrationError(f"{entry!r}: the digest is not hexadecimal")
        path = self._path(digest16)
        if not path.is_file():
            raise CalibrationError(
                f"{entry} cites a calibration that is not in {self.directory}; "
                f"a template may not claim evidence nobody can open")
        raw = json.loads(path.read_text(encoding="utf-8"))
        record = CalibrationRecord.from_dict(raw)
        recomputed = record.digest.removeprefix("sha256:")[:16]
        if recomputed != digest16:
            raise CalibrationError(
                f"{entry}: the record's contents hash to {recomputed}, not "
                f"{digest16}. It was edited after it was written, so it no "
                f"longer describes the evidence it was calibrated on")
        if raw.get("digest") and raw["digest"].removeprefix("sha256:")[:16] != digest16:
            raise CalibrationError(
                f"{entry}: the digest stored inside the record disagrees with "
                f"its filename")
        return record

    def verify(self, entry: str, constraint_name: str,
               window: tuple[float, float] | None) -> CalibrationRecord:
        """Check that ``entry`` really calibrates this constraint's window.

        Beyond :meth:`load`: the record must be for **this constraint**, the
        window in the template must be **exactly the window the record
        proposed**, and the freshly re-run verdict must meet the policy. Each
        closes a way of keeping the citation while changing what it vouches
        for -- pointing a calibration of one constraint at another, or
        widening the window after it was calibrated.
        """
        record = self.load(entry)
        if record.constraint != constraint_name:
            raise CalibrationError(
                f"{entry} calibrates {record.constraint!r}, not "
                f"{constraint_name!r}")
        if record.algorithm != CALIBRATION_ALGORITHM:
            raise CalibrationError(
                f"{entry} was made by {record.algorithm!r}; this build "
                f"understands {CALIBRATION_ALGORITHM!r} and will not judge a "
                f"record by a rule it does not have")
        if record.window is None or window is None or not (
                math.isclose(record.window[0], window[0], abs_tol=1e-9)
                and math.isclose(record.window[1], window[1], abs_tol=1e-9)):
            raise CalibrationError(
                f"{entry} calibrated the window {record.window}, but the "
                f"template's window is {window}. The calibration vouches for "
                f"exactly the range it was computed from; a window edited "
                f"afterwards is a different, uncalibrated window")
        if not record.verdict.meets_policy:
            raise CalibrationError(
                f"{entry} does not meet its own policy on re-judging: "
                + "; ".join(record.verdict.reasons))
        return record


# ==========================================================================
# what a template's citation is worth, in one place
# ==========================================================================

@dataclass(frozen=True)
class CalibrationStatus:
    """Whether one constraint's window may reject, and on what grounds.

    ``kind`` says which, because "calibrated" meant four different things
    that all looked the same:

    * ``verified`` -- a ``calibration:`` citation resolved to a stored record
      whose digest, constraint, window and re-judged verdict all check out;
    * ``free_text`` -- only prose in ``calibrated_on``. The old behaviour:
      accepted unless the context is strict, and always *named* as unverified;
    * ``unverifiable`` -- the entry claims to be verifiable and is not. This
      fails closed even when the context is not strict, because a citation
      that asserts a record and cannot produce one is a worse sign than no
      citation;
    * ``none`` -- nothing is cited.
    """

    calibrated: bool
    kind: str
    problems: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    @property
    def is_unverified_claim(self) -> bool:
        return self.kind in ("free_text", "unverifiable")


@dataclass(frozen=True)
class CalibrationContext:
    """How a run decides whether a window's calibration counts.

    ``store`` resolves ``calibration:`` citations. ``strict`` refuses to count
    free-text ``calibrated_on`` at all: with it on, the only way a window gains
    the authority to reject is a record somebody can open. The default is off
    because the shipped templates predate this mechanism and none of them
    cites any calibration, so strictness changes nothing for them -- but a
    campaign that intends a window to gate should turn it on, and the report
    names every window that is calibrated on prose alone either way.
    """

    store: CalibrationStore | None = None
    strict: bool = False

    def status(self, constraint: Any) -> CalibrationStatus:
        entries = [str(e) for e in (getattr(constraint, "calibrated_on", None) or ())
                   if str(e).strip()]
        if not entries:
            return CalibrationStatus(False, "none")
        cited = [e for e in entries if e.startswith(CALIBRATION_PREFIX)]
        free = [e for e in entries if not e.startswith(CALIBRATION_PREFIX)]
        if cited:
            window = _window_of(constraint)
            problems: list[str] = []
            for entry in cited:
                if self.store is None:
                    problems.append(
                        f"{entry} cannot be verified: this run was given no "
                        f"calibration store")
                    continue
                try:
                    self.store.verify(entry, constraint.name, window)
                except CalibrationError as exc:
                    problems.append(str(exc))
                else:
                    return CalibrationStatus(True, "verified", evidence=(entry,))
            return CalibrationStatus(False, "unverifiable", tuple(problems),
                                     tuple(cited))
        if self.strict:
            return CalibrationStatus(
                False, "free_text",
                (f"calibrated_on is prose ({', '.join(free)[:80]!r}) and this "
                 f"run requires a verifiable calibration record",),
                tuple(free))
        return CalibrationStatus(True, "free_text", (), tuple(free))


def _window_of(constraint: Any) -> tuple[float, float] | None:
    """The window a constraint applies, as a finite pair, or ``None``.

    A one-sided window (a lower bound only) has no finite pair and so cannot
    match a record, which always proposes both ends. That is correct: a
    calibration vouches for a range, and a half-open one was not calibrated.
    """
    try:
        lo, hi = constraint.window()
    except Exception:
        return None
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return float(lo), float(hi)
