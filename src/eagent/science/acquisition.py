"""Spending a round's slots when a model has something to say, and when it should not.

The batch composer's roles are *evidence* roles: high-evidence, uncertainty
probe, diversity. Once experiments come back there is a second source of
information -- a model fitted to the measured labels -- and the question
becomes how much of the next round to spend on what that model likes
(exploitation) and how much on what it has seen least of (exploration). This
module is that policy, written as plain functions so the same code runs in the
pipeline and in the retrospective simulation that tests it
(:mod:`eagent.eval.feedback_simulation`); a simulated policy that differed from
the shipped one would be a result about something nobody uses.

TWO RULES THAT MATTER MORE THAN THE ARITHMETIC
==============================================
**A model that has not earned a number gets no slots by that number.** The
exploit pool is the candidates the model *scored*. A candidate it returned
``None`` for -- an unknown target, an empty training set -- is never "low
scoring"; it is unscored, and it can only be picked by the exploration rule,
which does not read the model's opinion of it at all.

**Exploration is by novelty, not by confidence.** The tempting rule is "pick
what the model is least sure of", but a weak model's uncertainty is mostly
noise, and spending slots on noise is the opposite of exploration. Novelty --
distance from everything already measured -- is a property of the data, not of
the model, so it stays meaningful when the model is poor. ``uncertainty`` is
available for a model whose probabilities have earned calibration, and says so.

Nothing here knows about gates. It orders a pool the caller has already
restricted to candidates the mechanism checks allow; composing it with
:func:`eagent.science.diversity.compose_batch` is by ``rank_tiebreak``, which
can reorder only candidates that are equal on every evidence dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = [
    "AcquisitionPolicy",
    "Pick",
    "select_by_acquisition",
]


@dataclass(frozen=True)
class AcquisitionPolicy:
    """How many slots go to exploitation and how many to exploration."""

    n_exploit: int
    n_explore: int
    explore_by: str = "novelty"          # "novelty" | "uncertainty"

    def __post_init__(self) -> None:
        if self.n_exploit < 0 or self.n_explore < 0:
            raise ValueError("slot counts cannot be negative")
        if self.explore_by not in ("novelty", "uncertainty"):
            raise ValueError("explore_by must be 'novelty' or 'uncertainty'")

    @property
    def total(self) -> int:
        return self.n_exploit + self.n_explore


@dataclass(frozen=True)
class Pick:
    candidate_id: str
    role: str                            # "exploit" | "explore"
    reason: str


def select_by_acquisition(
    pool: Sequence[str],
    scores: Mapping[str, float | None],
    novelty: Mapping[str, float | None],
    policy: AcquisitionPolicy,
    *,
    probabilities: Mapping[str, float | None] | None = None,
) -> list[Pick]:
    """Choose ``policy.total`` ids from ``pool``, exploit slots first.

    Deterministic: ties break on the id. Returns fewer than requested -- never
    pads -- when a pool runs short, because a padded pick is a pick made for no
    stated reason.
    """
    picks: list[Pick] = []
    taken: set[str] = set()

    exploitable = sorted(
        (c for c in pool if scores.get(c) is not None),
        key=lambda c: (-float(scores[c]), c))            # type: ignore[arg-type]
    for cid in exploitable[:policy.n_exploit]:
        picks.append(Pick(cid, "exploit",
                          "highest model score among the candidates it scored"))
        taken.add(cid)

    if policy.explore_by == "uncertainty":
        if probabilities is None:
            raise ValueError("explore_by='uncertainty' needs calibrated "
                             "probabilities; none were supplied")
        explorable = sorted(
            (c for c in pool if c not in taken
             and probabilities.get(c) is not None),
            key=lambda c: (abs(float(probabilities[c]) - 0.5), c))  # type: ignore[arg-type]
        reason = "calibrated probability closest to 0.5"
    else:
        explorable = sorted(
            (c for c in pool if c not in taken and novelty.get(c) is not None),
            key=lambda c: (-float(novelty[c]), c))        # type: ignore[arg-type]
        reason = ("least similar to anything already measured; chosen without "
                  "reading the model's opinion of it")
    for cid in explorable[:policy.n_explore]:
        picks.append(Pick(cid, "explore", reason))
    return picks
