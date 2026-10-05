"""Interface ``ingest_results`` -- turn a returned plate into records that survive.

WHAT THIS STEP IS GUARDING
==========================
A screening round produces a spreadsheet. Everything that makes that
spreadsheet scientifically usable is destroyed by the obvious way of reading
it, which is to add a ``hit`` column and count.

**The outcome taxonomy is the data.** "the construct never expressed",
"it expressed and no product was detected down to 0.5 uM", "it made the other
enantiomer", "that well was never run" are four different facts, and they all
become ``0`` in a binary label. The third is arguably the most valuable result
a round can produce -- a selective enzyme pointing the wrong way is an
engineering target -- and it is the first to be lost.
:class:`~eagent.schemas.record.OutcomeClass` keeps all of them, and this
module maps into every branch.

**A negative needs a detection limit.** "No product" at an unstated limit is
not a measurement. The schema refuses such a record, and so does this module:
a row that fails the criterion with no limit of detection (and no
pre-registered limit in the :class:`~eagent.schemas.templates.AssayTemplate`
to fall back on) is held as unresolved rather than written down as a negative.

**A positive needs the product.** A rising NADPH absorbance at 340 nm says
something consumed the cofactor. Lysate does that. The schema ties
``CONFIRMED_TARGET_PRODUCT`` to a detection method that identifies the
product, and a row claiming a positive on an indirect signal alone is
surfaced as a ``BLOCKER`` and parked in ``pending_confirmation.jsonl`` -- not
converted into a negative, which would be the opposite error.

**ee is signed.** :func:`~eagent.schemas.record.ee_target` is signed toward
the target enantiomer, so a beautifully selective failure reports as -94%
rather than as "94% ee".

THE PRE-REGISTERED CRITERION IS THE ONLY CRITERION
==================================================
The positive criterion comes from the :class:`AssayTemplate` written into
``experiment_plan.yaml`` before the plate was run. There is no code path in
this module that changes it after seeing data: ``criterion_override`` is
accepted as an argument, **refused**, and recorded as a protocol deviation in
the round summary and in provenance. Writing the override down is the point --
an endpoint that moves silently is unfalsifiable, and an endpoint that moves
on the record is at least arguable.

An unrecognised key in ``positive_criteria`` raises rather than being ignored,
because a criterion with a typo in it silently passes everything.

ACTIVE LEARNING THAT KEEPS THE NEGATIVES
========================================
Model updates that train only on hits learn where the hits were, not where the
boundary is. :class:`ActiveLearningUpdate` partitions every record and then
*checks the arithmetic*: the partitions must sum to the input count, or
:class:`~eagent.errors.FabricationGuardError` is raised. Expression failures
are separated from catalytic negatives rather than discarded, because
``OutcomeClass.informs_catalytic_ability`` is false for them -- they are
training data for the expression-risk model and are silent about chemistry.

A NO-HIT ROUND IS A RESULT
==========================
When nothing turns over, :func:`diagnose_no_hits` produces a structured
differential across five hypotheses -- wrong search scope, cofactor mismatch,
expression failure, unsuitable detection conditions, and the substrate
genuinely lacking a natural catalyst -- each with what supports it, what
contradicts it, and the one experiment that would separate it from the others.
The fifth is never marked ``consistent`` on the strength of one round; a
single plate cannot support it, and saying so is the honest output.

THREE LAYERS, AND WHICH ONE MOVED
=================================
:class:`LayerUpdate` reports the data layer (records, limits, expression,
product identity), the model layer (specificity, expression risk, mutation
effects, uncertainty) and the decision layer (next-round quotas, exploration
scope, selection) separately, each with an explicit ``changed`` flag. A round
that adds records without moving any model is a real and common outcome, and
collapsing the three hides it.
"""

from __future__ import annotations

import csv
import enum
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping, Sequence

import yaml

from ..context import RunContext
from ..envelope import Artifact, Provenance, Severity, Status, ToolResult
from ..errors import FabricationGuardError, TemplateError
from ..provenance import sha256_file, sha256_obj
from ..schemas import (
    AssayTemplate,
    CofactorSpec,
    CofactorState,
    Conditions,
    Detection,
    EvidenceRef,
    EvidenceStrength,
    ExperimentRecord,
    OutcomeClass,
    ProductSpec,
    ReactionDirection,
    ee_target,
)
from ..science.robustness import wilson_interval
from .base import ScientificInterface
from .select_batch import ASSAY_RESULT_COLUMNS

__all__ = [
    "ASSAY_RESULT_COLUMNS",
    "RECOGNISED_CRITERION_KEYS",
    "MIN_FAMILY_QUOTA_AFTER_NEGATIVE",
    "PositiveCriterion",
    "AssayRow",
    "MeasurementGroup",
    "parse_assay_rows",
    "group_rows",
    "Classification",
    "classify_group",
    "UnresolvedRow",
    "ActiveLearningUpdate",
    "LayerUpdate",
    "NoHitHypothesis",
    "HypothesisAssessment",
    "NoHitDiagnosis",
    "diagnose_no_hits",
    "FamilyVerdict",
    "next_round_quotas",
    "IngestResults",
]


#: Keys :class:`PositiveCriterion` understands. Anything else raises: a
#: criterion carrying ``min_converion_pct`` would otherwise be a criterion that
#: silently passes every well, and nobody would notice until the confirmation
#: assay came back empty.
RECOGNISED_CRITERION_KEYS: frozenset[str] = frozenset({
    "min_conversion_pct",
    "min_measurement_value",
    "measurement_type",
    "min_ee_target_pct",
    "requires_authentic_standard",
    "requires_chiral_method_validated",
    "min_fold_over_empty_vector",
})

#: Floor a family's next-round quota is not taken below after negative results.
#:
#: NOT a scientific threshold: it is a sampling-design floor that keeps the
#: decision boundary observed. Cutting a family to zero after one round of
#: negatives means the next round cannot tell whether the family was wrong or
#: the first sample from it was unlucky. CALIBRATION: the right floor depends
#: on family size and on how expensive a slot is.
MIN_FAMILY_QUOTA_AFTER_NEGATIVE: int = 2

_YES = {"yes", "y", "true", "1", "t"}
_NO = {"no", "n", "false", "0", "f"}


# ==========================================================================
# The pre-registered criterion
# ==========================================================================

@dataclass(frozen=True)
class PositiveCriterion:
    """The hit definition, built once from the AssayTemplate and never changed.

    Every field is optional because assays differ, but the *source* is not:
    the object is only ever constructed from
    :attr:`AssayTemplate.positive_criteria`, and :meth:`evaluate` returns
    ``None`` -- undecided -- rather than ``False`` when a quantity the
    criterion needs was not measured. That distinction is the whole reason
    this is a class and not an ``if`` in a loop: "did not meet the bar" and
    "we could not tell" have different consequences, and only the first is a
    negative result.
    """

    min_conversion_pct: float | None = None
    min_measurement_value: float | None = None
    measurement_type: str | None = None
    min_ee_target_pct: float | None = None
    requires_authentic_standard: bool = False
    requires_chiral_method_validated: bool = False
    min_fold_over_empty_vector: float | None = None
    source_template_id: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_template(cls, template: AssayTemplate) -> "PositiveCriterion":
        """Build from the template, rejecting any key this code does not honour."""
        criteria = dict(template.positive_criteria or {})
        if not criteria:
            raise TemplateError(
                f"AssayTemplate {template.template_id} carries no "
                f"positive_criteria; without a pre-registered endpoint the "
                f"criterion would be chosen after the data arrive"
            )
        unknown = sorted(set(criteria) - RECOGNISED_CRITERION_KEYS)
        if unknown:
            raise TemplateError(
                f"AssayTemplate {template.template_id} declares criterion "
                f"key(s) {unknown} that this ingest step does not implement. "
                f"They would be ignored, and an ignored criterion is a "
                f"criterion that passes everything. Recognised keys: "
                f"{sorted(RECOGNISED_CRITERION_KEYS)}"
            )
        return cls(
            min_conversion_pct=_as_float(criteria.get("min_conversion_pct")),
            min_measurement_value=_as_float(criteria.get("min_measurement_value")),
            measurement_type=_as_str(criteria.get("measurement_type")),
            min_ee_target_pct=_as_float(criteria.get("min_ee_target_pct")),
            requires_authentic_standard=bool(
                criteria.get("requires_authentic_standard", False)),
            requires_chiral_method_validated=bool(
                criteria.get("requires_chiral_method_validated", False)),
            min_fold_over_empty_vector=_as_float(
                criteria.get("min_fold_over_empty_vector")),
            source_template_id=template.template_id,
            raw=criteria,
        )

    def digest(self) -> str:
        """Hash of the criterion as pre-registered, carried in provenance."""
        return sha256_obj(dict(self.raw))

    def evaluate(self, group: "MeasurementGroup") -> tuple[bool | None, list[str]]:
        """``(met, reasons)``. ``None`` means undecidable, which is not a no.

        A **definitively missed** bar beats an unmeasurable one: a condition
        that converted 0% cannot be rescued by the fact that nobody could
        compute its ee, so the answer is ``False`` and not ``None``. The
        reverse -- treating an unmeasured quantity as a miss -- is the error
        this guards against, and it only applies when every measurable bar
        passed.

        Product identity is checked by the caller rather than here, because a
        row that meets every numeric bar on an indirect signal is a specific
        failure mode with its own handling, not simply "criterion not met".
        """
        reasons: list[str] = []
        undecidable: list[str] = []
        met = True

        if self.min_conversion_pct is not None:
            value = group.conversion_pct
            if value is None:
                undecidable.append(
                    f"criterion needs conversion_pct >= "
                    f"{self.min_conversion_pct:g} and none was reported")
            elif value < self.min_conversion_pct:
                met = False
                reasons.append(
                    f"conversion {value:g}% < {self.min_conversion_pct:g}%")
            else:
                reasons.append(
                    f"conversion {value:g}% >= {self.min_conversion_pct:g}%")

        if self.min_measurement_value is not None:
            if self.measurement_type and group.measurement_type \
                    and group.measurement_type != self.measurement_type:
                undecidable.append(
                    f"criterion is written for measurement_type "
                    f"'{self.measurement_type}' but the plate reports "
                    f"'{group.measurement_type}'; endpoints are not pooled")
            elif group.measurement_value is None:
                undecidable.append(
                    f"criterion needs measurement_value >= "
                    f"{self.min_measurement_value:g} and none was reported")
            elif group.measurement_value < self.min_measurement_value:
                met = False
                reasons.append(
                    f"{group.measurement_type or 'measurement'} "
                    f"{group.measurement_value:g} < {self.min_measurement_value:g}")
            else:
                reasons.append(
                    f"{group.measurement_type or 'measurement'} "
                    f"{group.measurement_value:g} >= {self.min_measurement_value:g}")

        if self.min_ee_target_pct is not None:
            value = group.ee_target_pct
            if value is None:
                undecidable.append(
                    f"criterion needs signed ee >= {self.min_ee_target_pct:g}% "
                    f"and the enantiomer peak areas were not reported")
            elif value < self.min_ee_target_pct:
                met = False
                reasons.append(
                    f"signed ee {value:+.1f}% < {self.min_ee_target_pct:g}%")
            else:
                reasons.append(
                    f"signed ee {value:+.1f}% >= {self.min_ee_target_pct:g}%")

        if self.min_fold_over_empty_vector is not None:
            fold = group.fold_over_empty_vector
            if fold is None:
                if group.empty_vector_baseline is None:
                    undecidable.append(
                        f"criterion needs >= "
                        f"{self.min_fold_over_empty_vector:g}-fold over the "
                        f"empty-vector control and no such control was run on "
                        f"this plate under this cofactor condition")
                elif group.empty_vector_baseline <= 0:
                    undecidable.append(
                        f"the empty-vector control read "
                        f"{group.empty_vector_baseline:g}, so a fold change is "
                        f"not defined; report the difference or raise the "
                        f"detection limit instead")
                else:
                    undecidable.append(
                        f"criterion needs >= "
                        f"{self.min_fold_over_empty_vector:g}-fold over the "
                        f"empty-vector control and this group reported no "
                        f"measurement value")
            elif fold < self.min_fold_over_empty_vector:
                met = False
                reasons.append(
                    f"{fold:.2f}-fold over empty vector < "
                    f"{self.min_fold_over_empty_vector:g}-fold")
            else:
                reasons.append(
                    f"{fold:.2f}-fold over empty vector >= "
                    f"{self.min_fold_over_empty_vector:g}-fold")

        if self.requires_authentic_standard and group.authentic_standard is not True:
            undecidable.append(
                "criterion requires an authentic standard and the plate does "
                "not record one")
        if self.requires_chiral_method_validated \
                and group.chiral_method_validated is not True:
            undecidable.append(
                "criterion requires a validated chiral method and the plate "
                "does not record one")

        if not met:
            return False, reasons + undecidable
        if undecidable:
            return None, reasons + undecidable
        if not (self.min_conversion_pct is not None
                or self.min_measurement_value is not None
                or self.min_ee_target_pct is not None
                or self.min_fold_over_empty_vector is not None):
            return None, ["the criterion sets no numeric bar, so no well can "
                          "be decided against it"]
        return True, reasons


# ==========================================================================
# Rows and groups
# ==========================================================================

@dataclass(frozen=True)
class AssayRow:
    """One returned well, parsed but not yet interpreted."""

    plan_id: str
    slot: int | None
    plate: str
    well: str
    candidate_id: str
    construct_id: str
    kind: str                      # candidate | variant | control
    role: str
    parent_candidate_id: str
    mutations: tuple[str, ...]
    cofactor: str
    cofactor_state: str
    replicate: int | None
    tested: bool | None
    expressed_soluble: bool | None
    detection_method: str
    confirms_product_identity: bool
    authentic_standard: bool | None
    chiral_method_validated: bool | None
    limit_of_detection: float | None
    limit_unit: str
    measurement_type: str
    measurement_value: float | None
    measurement_unit: str
    conversion_pct: float | None
    product_identity_observed: str      # target | other | none | unknown
    peak_area_target: float | None
    peak_area_opposite: float | None
    notes: str

    @property
    def is_control(self) -> bool:
        return self.kind.strip().lower() == "control"

    @property
    def group_key(self) -> tuple[str, str, str]:
        return (self.candidate_id, self.cofactor, self.cofactor_state)


def parse_assay_rows(
    source: str | Path | Iterable[Mapping[str, Any]]
) -> tuple[list[AssayRow], list[str]]:
    """Parse the returned template into rows, reporting every problem.

    Returns ``(rows, problems)``. A malformed row is reported and skipped
    rather than coerced: a well whose ``conversion_pct`` reads ``"<LOD"``
    carries real information, but turning that string into ``0.0`` invents a
    measurement, and turning it into a hit is worse.

    A missing column is a hard error for the whole file, because a template
    that lost a column has almost certainly been re-ordered too, and a
    silently shifted plate map attributes every result to the wrong enzyme.
    """
    problems: list[str] = []
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8", newline="") as fh:
            raw_rows = list(csv.DictReader(fh))
        if raw_rows:
            missing = sorted(set(ASSAY_RESULT_COLUMNS) - set(raw_rows[0]))
            if missing:
                raise ValueError(
                    f"{source}: the returned file is missing column(s) "
                    f"{missing}. Refusing to parse it: a template with columns "
                    f"removed has usually been re-ordered as well, and a "
                    f"shifted plate map misattributes every result.")
    else:
        raw_rows = [dict(r) for r in source]

    rows: list[AssayRow] = []
    for n, raw in enumerate(raw_rows, start=2):   # line 1 is the header
        try:
            rows.append(AssayRow(
                plan_id=_as_str(raw.get("plan_id")) or "",
                slot=_as_int(raw.get("slot")),
                plate=_as_str(raw.get("plate")) or "",
                well=_as_str(raw.get("well")) or "",
                candidate_id=_as_str(raw.get("candidate_id")) or "",
                construct_id=_as_str(raw.get("construct_id")) or "",
                kind=_as_str(raw.get("kind")) or "candidate",
                role=_as_str(raw.get("role")) or "",
                parent_candidate_id=_as_str(raw.get("parent_candidate_id")) or "",
                mutations=tuple(m for m in
                                (_as_str(raw.get("mutations")) or "").split(";") if m),
                cofactor=_as_str(raw.get("cofactor")) or "",
                cofactor_state=_as_str(raw.get("cofactor_state")) or "unknown",
                replicate=_as_int(raw.get("replicate")),
                tested=_as_bool(raw.get("tested")),
                expressed_soluble=_as_bool(raw.get("expressed_soluble")),
                detection_method=_as_str(raw.get("detection_method")) or "",
                confirms_product_identity=_as_bool(
                    raw.get("confirms_product_identity")) is True,
                authentic_standard=_as_bool(raw.get("authentic_standard")),
                chiral_method_validated=_as_bool(
                    raw.get("chiral_method_validated")),
                limit_of_detection=_as_float(raw.get("limit_of_detection")),
                limit_unit=_as_str(raw.get("limit_unit")) or "",
                measurement_type=_as_str(raw.get("measurement_type")) or "",
                measurement_value=_as_float(raw.get("measurement_value")),
                measurement_unit=_as_str(raw.get("measurement_unit")) or "",
                conversion_pct=_as_float(raw.get("conversion_pct")),
                product_identity_observed=(
                    _as_str(raw.get("product_identity_observed")) or "unknown"
                ).lower(),
                peak_area_target=_as_float(
                    raw.get("peak_area_target_enantiomer")),
                peak_area_opposite=_as_float(
                    raw.get("peak_area_opposite_enantiomer")),
                notes=_as_str(raw.get("notes")) or "",
            ))
        except Exception as exc:                 # one bad row, not a bad file
            problems.append(f"line {n}: {type(exc).__name__}: {exc}")
        else:
            if not rows[-1].candidate_id:
                problems.append(
                    f"line {n}: no candidate_id; the well cannot be attributed "
                    f"and was dropped")
                rows.pop()
    return rows, problems


@dataclass
class MeasurementGroup:
    """Replicates of one (construct, cofactor condition), aggregated.

    Replicates are aggregated into one record because an
    :class:`~eagent.schemas.record.ExperimentRecord` is keyed on
    ``(sequence, substrate, reaction, cofactor, conditions)`` -- three wells
    of the same condition are one measurement with a spread, not three
    records. The spread is kept in :attr:`disagreement` and surfaced, because
    replicates that disagree about whether a well is a hit are the single most
    informative thing on a plate and are exactly what a mean hides.
    """

    candidate_id: str
    cofactor: str
    cofactor_state: str
    rows: list[AssayRow] = field(default_factory=list)
    #: Median empty-vector signal for this group's plate and cofactor
    #: condition, attached by :func:`attach_empty_vector_baselines`. ``None``
    #: means no such control was run, which makes a fold-over-background bar
    #: undecidable rather than satisfied.
    empty_vector_baseline: float | None = None
    #: Where that baseline came from, for the record.
    empty_vector_basis: str = ""

    # -- derived ----------------------------------------------------------
    @property
    def n_replicates(self) -> int:
        return len(self.rows)

    @property
    def tested_rows(self) -> list[AssayRow]:
        return [r for r in self.rows if r.tested is not False]

    @property
    def any_tested(self) -> bool:
        return any(r.tested is True for r in self.rows)

    @property
    def kind(self) -> str:
        return self.rows[0].kind if self.rows else "candidate"

    @property
    def role(self) -> str:
        return self.rows[0].role if self.rows else ""

    @property
    def parent_candidate_id(self) -> str:
        return self.rows[0].parent_candidate_id if self.rows else ""

    @property
    def mutations(self) -> list[str]:
        return list(self.rows[0].mutations) if self.rows else []

    @property
    def expressed_soluble(self) -> bool | None:
        values = [r.expressed_soluble for r in self.rows
                  if r.expressed_soluble is not None]
        if not values:
            return None
        return any(values)         # one soluble prep is enough to have expressed

    @property
    def detection_method(self) -> str:
        for row in self.rows:
            if row.detection_method:
                return row.detection_method
        return ""

    @property
    def confirms_product_identity(self) -> bool:
        return any(r.confirms_product_identity for r in self.rows)

    @property
    def authentic_standard(self) -> bool | None:
        return _consensus([r.authentic_standard for r in self.rows])

    @property
    def chiral_method_validated(self) -> bool | None:
        return _consensus([r.chiral_method_validated for r in self.rows])

    @property
    def limit_of_detection(self) -> float | None:
        values = [r.limit_of_detection for r in self.rows
                  if r.limit_of_detection is not None]
        return max(values) if values else None      # the weakest limit binds

    @property
    def limit_unit(self) -> str:
        for row in self.rows:
            if row.limit_unit:
                return row.limit_unit
        return ""

    @property
    def measurement_type(self) -> str:
        for row in self.rows:
            if row.measurement_type:
                return row.measurement_type
        return ""

    @property
    def measurement_unit(self) -> str:
        for row in self.rows:
            if row.measurement_unit:
                return row.measurement_unit
        return ""

    @property
    def measurement_value(self) -> float | None:
        """Median over the wells that were actually measured.

        Reads :attr:`tested_rows`, never ``rows``. A well marked ``tested=no``
        routinely still carries a number: the commonest way a plate reaches
        this code is a copy of the previous round's sheet with the untested
        wells' old values left in place. Taking the median over every row lets
        a stale value from a well nobody ran outvote the one that was run.
        """
        return _median([r.measurement_value for r in self.tested_rows])

    @property
    def fold_over_empty_vector(self) -> float | None:
        """Signal relative to the empty-vector control on the same plate.

        ``None`` whenever it cannot be computed: no baseline was run, this
        group reported no measurement, or the baseline is zero or negative so
        a ratio would be meaningless. Each of those is a reason the bar cannot
        be decided, never a reason to treat it as met.
        """
        if self.empty_vector_baseline is None:
            return None
        value = self.measurement_value
        if value is None:
            return None
        if self.empty_vector_baseline <= 0:
            return None
        return value / self.empty_vector_baseline

    @property
    def conversion_pct(self) -> float | None:
        """Median conversion over the wells that were actually measured.

        See :attr:`measurement_value`: an untested well's leftover number is
        not a measurement and must not enter an aggregate.
        """
        return _median([r.conversion_pct for r in self.tested_rows])

    @property
    def product_identity_observed(self) -> str:
        """Worst identity observed across the measured wells.

        Restricted to tested wells for the same reason as the numeric
        aggregates: an untested well's leftover "target" is not an
        observation.
        """
        seen = {r.product_identity_observed for r in self.tested_rows}
        for key in ("other", "target", "none"):
            if key in seen:
                return key
        return "unknown"

    @property
    def ee_target_pct(self) -> float | None:
        """Signed ee toward the target enantiomer, from summed peak areas.

        Returns ``None`` unless BOTH enantiomer peaks were quantified in the
        same measured well. A blank opposite-peak cell is missing data, and
        summing it as zero turns "nobody reported the other peak" into
        "the other enantiomer is absent", which is the strongest
        stereochemical claim the assay can make and is manufactured here out
        of an empty cell. An explicit non-detection is a different statement
        and belongs in :meth:`ee_target_lower_bound`, which is bounded by the
        detection limit rather than reported as an exact 100%.

        Summed rather than averaged per replicate so a replicate with almost
        no product cannot swing the ratio, and ``None`` -- never 0.0 -- when
        nothing was quantified, because a 0 would read as "racemic".
        """
        usable = [r for r in self.tested_rows
                  if r.peak_area_target is not None
                  and r.peak_area_opposite is not None]
        if not usable:
            return None
        target = sum(r.peak_area_target for r in usable)
        opposite = sum(r.peak_area_opposite for r in usable)
        if (target + opposite) <= 0:
            return None
        return ee_target(target, opposite)

    @property
    def ee_incomplete_peaks(self) -> bool:
        """Whether a measured well reported one enantiomer peak but not both.

        Surfaced so the caller can say "the chiral analysis is incomplete"
        instead of silently producing no ee and letting it read as "not
        chiral".
        """
        return any(
            (r.peak_area_target is None) != (r.peak_area_opposite is None)
            for r in self.tested_rows)

    def ee_target_lower_bound(self) -> tuple[float, str] | None:
        """Bound on signed ee when the opposite peak was explicitly not detected.

        Only defensible when the well records a quantitation limit for the
        analysis that produced the target peak: the unseen enantiomer could
        sit anywhere up to that limit, so the honest statement is a bound
        computed at the limit, not an exact value computed at zero.

        Returns ``(lower_bound_pct, basis)``, or ``None`` when no well
        supports even a bound.
        """
        best: tuple[float, str] | None = None
        for r in self.tested_rows:
            if r.peak_area_target is None or r.peak_area_opposite is not None:
                continue
            if r.product_identity_observed != "target":
                continue
            limit = r.limit_of_detection
            if limit is None or limit <= 0:
                continue
            bound = ee_target(r.peak_area_target, limit)
            basis = (f"opposite enantiomer not detected; bound computed at the "
                     f"recorded limit {limit:g} {r.limit_unit or ''}".strip())
            if best is None or bound < best[0]:
                best = (bound, basis)
        return best

    @property
    def product_signal(self) -> bool:
        """Whether anything at all was measured above zero in this condition.

        Used to keep "nothing was there" apart from "something was there and it
        missed the bar". The second is turnover and must never be recorded as
        ``NO_TARGET_PRODUCT_DETECTED``, which claims the opposite.
        """
        for value in (self.conversion_pct, self.measurement_value):
            if value is not None and value > 0:
                return True
        return self.ee_target_pct is not None

    @property
    def disagreement(self) -> str:
        """Description of replicate spread worth reporting, or an empty string."""
        notes: list[str] = []
        expressed = {r.expressed_soluble for r in self.rows
                     if r.expressed_soluble is not None}
        if len(expressed) > 1:
            notes.append("replicates disagree about soluble expression")
        values = [r.conversion_pct for r in self.rows if r.conversion_pct is not None]
        if len(values) > 1:
            lo, hi = min(values), max(values)
            if hi > 0 and (hi - lo) > 0.5 * hi:
                notes.append(
                    f"conversion spread {lo:g}-{hi:g}% across "
                    f"{len(values)} replicate(s) exceeds half the maximum")
        identities = {r.product_identity_observed for r in self.rows
                      if r.product_identity_observed != "unknown"}
        if len(identities) > 1:
            notes.append(
                f"replicates report different product identities: "
                f"{', '.join(sorted(identities))}")
        return "; ".join(notes)


def group_rows(rows: Sequence[AssayRow]) -> list[MeasurementGroup]:
    """Group replicate wells by (construct, cofactor, cofactor state).

    The cofactor is part of the key, not metadata: an enzyme that works with
    NADPH and not NADH is two records with two different outcomes, and merging
    them produces one record that is wrong under both conditions.
    """
    groups: dict[tuple[str, str, str], MeasurementGroup] = {}
    for row in rows:
        key = row.group_key
        group = groups.get(key)
        if group is None:
            group = MeasurementGroup(candidate_id=row.candidate_id,
                                     cofactor=row.cofactor,
                                     cofactor_state=row.cofactor_state)
            groups[key] = group
        group.rows.append(row)
    return [groups[k] for k in sorted(groups)]


#: Row ``kind``/``role`` tokens that identify an empty-vector control.
EMPTY_VECTOR_TOKENS: frozenset[str] = frozenset({
    "empty_vector", "empty-vector", "emptyvector", "vector_only", "no_insert",
})


def _is_empty_vector(row: AssayRow) -> bool:
    tokens = {str(row.kind or "").strip().lower(),
              str(row.role or "").strip().lower()}
    return bool(tokens & EMPTY_VECTOR_TOKENS)


def attach_empty_vector_baselines(
    groups: Sequence[MeasurementGroup], rows: Sequence[AssayRow],
) -> list[str]:
    """Attach each group's empty-vector baseline, matched on plate and cofactor.

    A fold-over-background bar compares a well with the control that shared
    its plate and its cofactor condition. Borrowing a baseline from another
    plate would silently compare against a different day's background, so a
    group with no matching control gets no baseline and its bar becomes
    undecidable. That is the honest outcome: the plate did not carry the
    control the criterion needs.

    Returns the notes describing what was and was not matched, for the record.
    """
    by_key: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        if not _is_empty_vector(row):
            continue
        if row.measurement_value is None:
            continue
        key = (str(row.plate or ""), str(row.cofactor or ""),
               str(row.cofactor_state or ""))
        by_key.setdefault(key, []).append(float(row.measurement_value))

    notes: list[str] = []
    if not by_key:
        notes.append(
            "no empty-vector control carried a measurement value, so any "
            "fold-over-background criterion is undecidable for every well")
    unmatched: set[str] = set()
    for group in groups:
        plates = {str(r.plate or "") for r in group.rows}
        matched: list[float] = []
        used: list[str] = []
        for plate in sorted(plates):
            key = (plate, group.cofactor, group.cofactor_state)
            values = by_key.get(key)
            if values:
                matched.extend(values)
                used.append(plate)
        if matched:
            group.empty_vector_baseline = _median(matched)
            group.empty_vector_basis = (
                f"median of {len(matched)} empty-vector well(s) on plate(s) "
                f"{', '.join(used)} under {group.cofactor}"
                f"[{group.cofactor_state}]")
        else:
            group.empty_vector_basis = (
                f"no empty-vector control on plate(s) "
                f"{', '.join(sorted(plates)) or '-'} under {group.cofactor}"
                f"[{group.cofactor_state}]")
            unmatched.update(plates)
    if unmatched:
        notes.append(
            f"plate(s) {', '.join(sorted(p for p in unmatched if p))} carried "
            f"no matching empty-vector control; fold-over-background is "
            f"undecidable there rather than assumed met")
    return notes


# ==========================================================================
# Classification
# ==========================================================================

@dataclass(frozen=True)
class UnresolvedRow:
    """A measurement that cannot become a record without losing information.

    Two cases, both of which a careless pipeline turns into a negative:
    a positive signal resting only on an indirect readout, and a failure to
    meet the criterion with no detection limit to be negative at. Both are
    preserved here in full, with the exact confirmation that would resolve
    them, instead of being written down as something they are not.
    """

    candidate_id: str
    cofactor: str
    reason_code: str
    reason: str
    required_to_resolve: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id, "cofactor": self.cofactor,
            "reason_code": self.reason_code, "reason": self.reason,
            "required_to_resolve": self.required_to_resolve,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class Classification:
    """The outcome a group earns, with the criterion's own words behind it."""

    outcome: OutcomeClass | None
    reasons: tuple[str, ...]
    unresolved: UnresolvedRow | None = None

    @property
    def is_record(self) -> bool:
        return self.outcome is not None


def classify_group(
    group: MeasurementGroup,
    criterion: PositiveCriterion,
    *,
    fallback_limit_of_detection: float | None = None,
    fallback_limit_unit: str = "",
) -> Classification:
    """Assign one outcome class, in the order the failures actually occur.

    The order matters and is not arbitrary:

    1. **not tested** -- a blank well says nothing about anything;
    2. **expression failure** -- an insoluble construct was never given the
       chance to catalyse, so its catalytic ability is undetermined and it
       must not land in a negative set;
    3. **wrong product** -- a different constitutional product means the
       chemoselectivity is wrong, which is a result, not a failure to detect;
    4. the pre-registered criterion, then, and only then.

    Reversing 2 and 4 is how an expression problem becomes "this family does
    not work".
    """
    reasons: list[str] = []

    if not group.any_tested and all(r.tested is not True for r in group.rows):
        return Classification(
            OutcomeClass.NOT_TESTED,
            ("no well of this condition was run; the result is unknown, "
             "which is not the same as negative",))

    if group.expressed_soluble is False:
        return Classification(
            OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE,
            ("no soluble expression; catalytic ability undetermined and this "
             "record must not be counted as a catalytic negative",))

    met, criterion_reasons = criterion.evaluate(group)
    reasons.extend(criterion_reasons)

    if group.product_identity_observed == "other":
        return Classification(
            OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION,
            tuple(reasons + ["a product other than the target was observed; "
                             "turnover occurred but the chemoselectivity is wrong"]))

    ee = group.ee_target_pct
    if ee is not None and ee < 0:
        return Classification(
            OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION,
            tuple(reasons + [
                f"signed ee {ee:+.1f}% toward the target enantiomer: the "
                f"opposite configuration dominated. Selective, and pointing "
                f"the wrong way -- an engineering target, not a negative."]))

    if met is True:
        if not group.confirms_product_identity:
            return Classification(
                None,
                tuple(reasons + [
                    "every numeric bar was met on a detection method that does "
                    "not identify the product"]),
                UnresolvedRow(
                    candidate_id=group.candidate_id, cofactor=group.cofactor,
                    reason_code="indirect_signal_positive",
                    reason=(
                        f"{group.candidate_id} meets the pre-registered "
                        f"criterion on "
                        f"{group.detection_method or 'an unnamed method'}, "
                        f"which does not identify the product. A cofactor "
                        f"absorbance change is consumed NAD(P)H, not the "
                        f"target alcohol; lysate does that."),
                    required_to_resolve=(
                        "re-run on a method that identifies the product "
                        "(GC-MS or chiral HPLC against an authentic standard) "
                        "before this is called a hit"),
                    evidence={
                        "detection_method": group.detection_method,
                        "measurement_type": group.measurement_type,
                        "measurement_value": group.measurement_value,
                        "conversion_pct": group.conversion_pct,
                        "criterion_reasons": criterion_reasons,
                    }))
        if group.product_identity_observed != "target":
            # Three different statements get conflated here if this check is
            # missing: that the substrate was consumed, that the method is
            # CAPABLE of identifying the product, and that this well's product
            # WAS identified as the target. The branch above establishes only
            # the second. A conversion signal is equally consistent with a side
            # reaction, with degradation, or with the substrate decomposing, so
            # a well that reports no identified product is not a hit however
            # well it clears the numeric bars.
            observed = group.product_identity_observed
            return Classification(
                None,
                tuple(reasons + [
                    f"every numeric bar was met and the method can identify "
                    f"the product, but this well reports "
                    f"product_identity_observed={observed!r}"]),
                UnresolvedRow(
                    candidate_id=group.candidate_id, cofactor=group.cofactor,
                    reason_code="bars_met_without_identified_product",
                    reason=(
                        f"{group.candidate_id} meets the pre-registered "
                        f"criterion, but the target product was not identified "
                        f"in this well (reported {observed!r}). Substrate "
                        f"consumption is not product formation: a side "
                        f"reaction or degradation gives the same conversion "
                        f"number."),
                    required_to_resolve=(
                        "record what the identifying method actually saw. If "
                        "the target was observed, set it on the plate; if "
                        "nothing was, this is a conversion without product and "
                        "belongs in the record as such"),
                    evidence={
                        "product_identity_observed": observed,
                        "detection_method": group.detection_method,
                        "measurement_type": group.measurement_type,
                        "measurement_value": group.measurement_value,
                        "conversion_pct": group.conversion_pct,
                        "criterion_reasons": criterion_reasons,
                    }))
        if ee is not None and criterion.min_ee_target_pct is not None \
                and ee < criterion.min_ee_target_pct:
            return Classification(
                OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION,
                tuple(reasons + ["the target product formed but did not reach "
                                 "the pre-registered selectivity bar"]))
        return Classification(OutcomeClass.CONFIRMED_TARGET_PRODUCT, tuple(reasons))

    limit = group.limit_of_detection
    limit_unit = group.limit_unit
    limit_source = "reported on the plate"
    if limit is None and fallback_limit_of_detection is not None:
        limit = fallback_limit_of_detection
        limit_unit = fallback_limit_unit
        limit_source = "the assay template's pre-registered limit"

    if met is None:
        return Classification(
            None, tuple(reasons),
            UnresolvedRow(
                candidate_id=group.candidate_id, cofactor=group.cofactor,
                reason_code="criterion_undecidable",
                reason=("the pre-registered criterion could not be evaluated: "
                        + "; ".join(criterion_reasons)),
                required_to_resolve=(
                    "report the quantities the criterion names, or re-run the "
                    "condition; an undecidable well is not a negative"),
                evidence={"criterion": dict(criterion.raw),
                          "measurement_type": group.measurement_type,
                          "measurement_value": group.measurement_value,
                          "conversion_pct": group.conversion_pct}))

    if group.product_signal:
        if group.confirms_product_identity \
                and group.product_identity_observed == "target":
            return Classification(
                OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION,
                tuple(reasons + [
                    "the target product was formed and identified, but the "
                    "condition did not reach the pre-registered bar; turnover "
                    "occurred, so this is not an absence of product"]))
        return Classification(
            None, tuple(reasons),
            UnresolvedRow(
                candidate_id=group.candidate_id, cofactor=group.cofactor,
                reason_code="signal_without_confirmed_identity",
                reason=("a signal above zero was measured but the product was "
                        "not identified as the target, so the condition can be "
                        "called neither a hit nor an absence of product"),
                required_to_resolve=(
                    "identify what was formed, with a method that resolves the "
                    "product, before this condition is counted either way"),
                evidence={"detection_method": group.detection_method,
                          "product_identity_observed":
                              group.product_identity_observed,
                          "conversion_pct": group.conversion_pct,
                          "measurement_value": group.measurement_value,
                          "criterion_reasons": criterion_reasons}))

    if limit is None:
        return Classification(
            None, tuple(reasons),
            UnresolvedRow(
                candidate_id=group.candidate_id, cofactor=group.cofactor,
                reason_code="negative_without_detection_limit",
                reason=("the condition failed the criterion but no limit of "
                        "detection was reported and the assay template carries "
                        "none; 'no product' at an unstated limit is not a "
                        "measurement"),
                required_to_resolve=(
                    "report the limit of detection for this method, or record "
                    "the template's pre-registered limit"),
                evidence={"criterion_reasons": criterion_reasons,
                          "detection_method": group.detection_method}))

    return Classification(
        OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
        tuple(reasons + [
            f"no target product detected down to {limit:g} "
            f"{limit_unit or 'unit unspecified'} ({limit_source}); this is a "
            f"negative at a stated limit, which is information about the "
            f"decision boundary"]))


# ==========================================================================
# Active learning
# ==========================================================================

@dataclass(frozen=True)
class ActiveLearningUpdate:
    """Every record partitioned, with the arithmetic checked.

    The guarantee is small and load-bearing: the partitions sum to the input.
    A training-set builder that quietly drops the negatives produces a model
    that has only ever seen successes, cannot represent the boundary, and will
    rank the next round by similarity to round 1.
    """

    positives: tuple[str, ...]
    catalytic_negatives: tuple[str, ...]
    wrong_product_or_configuration: tuple[str, ...]
    expression_failures: tuple[str, ...]
    untested: tuple[str, ...]
    note: str

    @property
    def n_total(self) -> int:
        return (len(self.positives) + len(self.catalytic_negatives)
                + len(self.wrong_product_or_configuration)
                + len(self.expression_failures) + len(self.untested))

    @property
    def n_catalytically_informative(self) -> int:
        """Records that say something about whether the chemistry can run."""
        return (len(self.positives) + len(self.catalytic_negatives)
                + len(self.wrong_product_or_configuration))

    def to_dict(self) -> dict[str, Any]:
        return {
            "positives": list(self.positives),
            "catalytic_negatives": list(self.catalytic_negatives),
            "wrong_product_or_configuration":
                list(self.wrong_product_or_configuration),
            "expression_failures": list(self.expression_failures),
            "untested": list(self.untested),
            "n_total": self.n_total,
            "n_catalytically_informative": self.n_catalytically_informative,
            "negatives_retained": len(self.catalytic_negatives),
            "note": self.note,
        }


def build_active_learning_update(
    records: Sequence[ExperimentRecord]
) -> ActiveLearningUpdate:
    """Partition records for the next model fit, keeping the negatives.

    Raises :class:`~eagent.errors.FabricationGuardError` if the partitions do
    not sum to the input count. That check exists because the failure it
    catches is invisible: a dropped negative does not raise, does not log, and
    produces a model that looks better than the data support.
    """
    positives: list[str] = []
    negatives: list[str] = []
    wrong: list[str] = []
    expression: list[str] = []
    untested: list[str] = []
    for record in records:
        if record.outcome is OutcomeClass.CONFIRMED_TARGET_PRODUCT:
            positives.append(record.record_id)
        elif record.outcome is OutcomeClass.NO_TARGET_PRODUCT_DETECTED:
            negatives.append(record.record_id)
        elif record.outcome is OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION:
            wrong.append(record.record_id)
        elif record.outcome is OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE:
            expression.append(record.record_id)
        else:
            untested.append(record.record_id)
    update = ActiveLearningUpdate(
        positives=tuple(positives), catalytic_negatives=tuple(negatives),
        wrong_product_or_configuration=tuple(wrong),
        expression_failures=tuple(expression), untested=tuple(untested),
        note=(
            f"{len(negatives)} catalytic negative(s) retained as training data: "
            f"a negative at a stated detection limit is information about the "
            f"decision boundary and is never dropped. "
            f"{len(expression)} expression failure(s) are held out of the "
            f"catalytic label set -- OutcomeClass.informs_catalytic_ability is "
            f"false for them -- and feed the expression-risk model instead. "
            f"{len(untested)} untested record(s) carry no label at all."),
    )
    if update.n_total != len(records):
        raise FabricationGuardError(
            f"active-learning partition lost {len(records) - update.n_total} "
            f"record(s); every outcome class must land in exactly one partition")
    return update


# ==========================================================================
# Three layers
# ==========================================================================

@dataclass
class LayerUpdate:
    """What moved in the data, model and decision layers, kept apart.

    Three separate ``changed`` flags because the three move at different
    rates and for different reasons. Fifty new records and no model movement
    is the normal result of a well-designed round that confirmed what was
    expected; a model that moves on every round is a model that is fitting
    noise.
    """

    data_changed: bool = False
    model_changed: bool = False
    decision_changed: bool = False
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    data_changes: list[str] = field(default_factory=list)
    model_changes: list[str] = field(default_factory=list)
    decision_changes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_layer": {"changed": self.data_changed,
                           "changes": self.data_changes, "state": self.data},
            "model_layer": {"changed": self.model_changed,
                            "changes": self.model_changes, "state": self.model},
            "decision_layer": {"changed": self.decision_changed,
                               "changes": self.decision_changes,
                               "state": self.decision},
        }

    def summary(self) -> str:
        moved = [name for name, flag in (("data", self.data_changed),
                                         ("model", self.model_changed),
                                         ("decision", self.decision_changed))
                 if flag]
        return (f"layers changed: {', '.join(moved)}" if moved
                else "no layer changed")


# ==========================================================================
# No-hit differential
# ==========================================================================

class NoHitHypothesis(str, enum.Enum):
    """The five reasons a round returns nothing. They need different fixes."""

    SEARCH_SCOPE_WRONG = "search_scope_wrong"
    COFACTOR_MISMATCH = "cofactor_mismatch"
    EXPRESSION_FAILURE = "expression_failure"
    DETECTION_CONDITIONS_UNSUITABLE = "detection_conditions_unsuitable"
    NO_NATURAL_CATALYST = "no_natural_catalyst"

    def question(self) -> str:
        return {
            NoHitHypothesis.SEARCH_SCOPE_WRONG:
                "did we look in the wrong families or too narrow a slice of them",
            NoHitHypothesis.COFACTOR_MISMATCH:
                "were the enzymes offered a cofactor they do not use",
            NoHitHypothesis.EXPRESSION_FAILURE:
                "did the constructs ever become soluble protein",
            NoHitHypothesis.DETECTION_CONDITIONS_UNSUITABLE:
                "would the assay have seen the product if it had been there",
            NoHitHypothesis.NO_NATURAL_CATALYST:
                "does this substrate simply have no natural catalyst",
        }[self]


@dataclass(frozen=True)
class HypothesisAssessment:
    """One hypothesis, what supports it, what rules it out, and how to decide."""

    hypothesis: NoHitHypothesis
    status: str                      # consistent | unlikely | undetermined
    supported_by: tuple[str, ...]
    contradicted_by: tuple[str, ...]
    discriminating_experiment: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis.value,
            "question": self.hypothesis.question(),
            "status": self.status,
            "supported_by": list(self.supported_by),
            "contradicted_by": list(self.contradicted_by),
            "discriminating_experiment": self.discriminating_experiment,
        }


@dataclass(frozen=True)
class NoHitDiagnosis:
    """A differential, never a verdict. A no-hit round is preserved in full."""

    triggered: bool
    assessments: tuple[HypothesisAssessment, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "summary": self.summary,
            "hypotheses": [a.to_dict() for a in self.assessments],
        }


def diagnose_no_hits(
    records: Sequence[ExperimentRecord],
    groups: Sequence[MeasurementGroup],
    families: Mapping[str, str],
    *,
    cofactor_conditions_run: Sequence[str] = (),
    declared_cofactor_preferences: Mapping[str, str] | None = None,
    system_control_worked: bool | None = None,
) -> NoHitDiagnosis:
    """Build the five-way differential for a round that produced no hit.

    Each hypothesis gets ``consistent`` / ``unlikely`` / ``undetermined`` from
    evidence that is actually on the plate, and a named experiment that would
    separate it from the rest. Nothing is ranked: ranking them would imply a
    prior nobody has.

    :attr:`NoHitHypothesis.NO_NATURAL_CATALYST` is never returned as
    ``consistent``. One round, over one sampled slice of sequence space, with
    one assay, cannot support it -- and the moment it is written down as the
    answer the campaign stops. It is reported as ``undetermined`` with the
    scope that would be needed to even start arguing for it.
    """
    positives = [r for r in records if r.outcome.is_positive]
    triggered = not positives
    tested_families = sorted({families.get(g.candidate_id, "unassigned")
                              for g in groups if g.kind != "control"})
    expression_failures = [r for r in records
                           if r.outcome is OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE]
    informative = [r for r in records if r.outcome.informs_catalytic_ability]
    wrong_product = [r for r in records
                     if r.outcome is OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION]
    n_constructs = len({r.sequence_sha256 or r.record_id for r in records})
    expression_rate = (len(expression_failures) / len(records)) if records else 0.0

    assessments: list[HypothesisAssessment] = []

    # -- 1. search scope -------------------------------------------------
    support: list[str] = []
    against: list[str] = []
    if len(tested_families) <= 1:
        support.append(
            f"only {len(tested_families)} family was represented "
            f"({', '.join(tested_families) or 'none'}); a single-family round "
            f"cannot distinguish 'wrong family' from 'wrong chemistry'")
    else:
        against.append(f"{len(tested_families)} families were represented: "
                       f"{', '.join(tested_families)}")
    if wrong_product:
        against.append(
            f"{len(wrong_product)} construct(s) produced a product or a "
            f"configuration: the scaffolds reach the substrate, so the scope "
            f"is not empty of relevant chemistry")
    assessments.append(HypothesisAssessment(
        NoHitHypothesis.SEARCH_SCOPE_WRONG,
        "consistent" if support and not against else
        ("unlikely" if against and not support else "undetermined"),
        tuple(support), tuple(against),
        "widen mining to the families excluded by the round-1 quota, and "
        "include a positive-control enzyme from a family known to reduce a "
        "structurally similar ketone",
    ))

    # -- 2. cofactor mismatch --------------------------------------------
    support, against = [], []
    conditions = sorted({c for c in cofactor_conditions_run if c})
    if len(conditions) <= 1:
        support.append(
            f"only {len(conditions)} cofactor condition was run "
            f"({', '.join(conditions) or 'none recorded'}); an NADPH-preferring "
            f"enzyme screened with NADH alone is a true negative that reads as "
            f"no activity")
    else:
        against.append(f"{len(conditions)} cofactor conditions were run: "
                       f"{', '.join(conditions)}")
    for family, preference in (declared_cofactor_preferences or {}).items():
        if family in tested_families and preference and preference not in conditions:
            support.append(
                f"family {family} is annotated as preferring {preference}, "
                f"which was not among the conditions run")
    assessments.append(HypothesisAssessment(
        NoHitHypothesis.COFACTOR_MISMATCH,
        "consistent" if support else ("unlikely" if against else "undetermined"),
        tuple(support), tuple(against),
        "re-run the same constructs under both NADH and NADPH with a "
        "recycling system for each; this is a plate, not a new synthesis",
    ))

    # -- 3. expression ----------------------------------------------------
    support, against = [], []
    if expression_rate >= 0.5:
        support.append(
            f"{len(expression_failures)} of {len(records)} record(s) "
            f"({expression_rate:.0%}) are expression or solubility failures; "
            f"most of the round never produced protein to test")
    elif expression_failures:
        support.append(
            f"{len(expression_failures)} expression failure(s) remove "
            f"{expression_rate:.0%} of the round from the catalytic question")
    if informative:
        against.append(
            f"{len(informative)} record(s) did reach a catalytic readout, so "
            f"expression cannot explain all of the round")
    assessments.append(HypothesisAssessment(
        NoHitHypothesis.EXPRESSION_FAILURE,
        "consistent" if expression_rate >= 0.5 else
        ("unlikely" if informative and not expression_failures else "undetermined"),
        tuple(support), tuple(against),
        "re-express the non-expressing constructs with a solubility tag, at "
        "lower temperature, or in a chaperone-coexpressing host, and confirm "
        "soluble protein by SDS-PAGE before re-assaying",
    ))

    # -- 4. detection ------------------------------------------------------
    support, against = [], []
    if system_control_worked is False:
        support.append(
            "the positive control that demonstrates the assay system works did "
            "not produce its expected signal; the detection chain itself is "
            "suspect and no well on the plate can be interpreted")
    elif system_control_worked is True:
        against.append(
            "the assay-system positive control produced its expected signal, "
            "so reagents, extraction and detection were alive")
    else:
        support.append(
            "no assay-system positive control result was reported, so a dead "
            "detection chain cannot be excluded")
    limits = [r.detection.limit_of_detection for r in records
              if r.detection.limit_of_detection is not None]
    if not limits:
        support.append(
            "no detection limit is recorded anywhere in the round, so the "
            "sensitivity the negatives are negative at is unknown")
    assessments.append(HypothesisAssessment(
        NoHitHypothesis.DETECTION_CONDITIONS_UNSUITABLE,
        "consistent" if system_control_worked is False else
        ("unlikely" if system_control_worked is True and limits
         else "undetermined"),
        tuple(support), tuple(against),
        "spike the authentic product standard into lysate at the claimed "
        "detection limit and recover it through the full workup; a spike that "
        "is not recovered invalidates every negative in the round",
    ))

    # -- 5. no natural catalyst -------------------------------------------
    assessments.append(HypothesisAssessment(
        NoHitHypothesis.NO_NATURAL_CATALYST,
        "undetermined",
        (f"{len(informative)} construct(s) across {len(tested_families)} "
         f"family/families reached a catalytic readout without a hit",),
        ("one round over one sampled slice of sequence space, with one assay, "
         "cannot establish the absence of a natural catalyst; the other four "
         "hypotheses have to be excluded experimentally first",),
        "this is not decidable by one round. It would need: the detection "
        "chain validated by product spike-in, soluble expression confirmed "
        "for the tested constructs, both cofactors covered, and a scope "
        "spanning the families that perform the analogous chemistry on "
        "related substrates -- and even then the honest conclusion is 'not "
        "found in the space searched', which is what directs engineering on "
        "the closest working scaffold",
    ))

    if triggered:
        summary = (
            f"No construct produced the target product. {n_constructs} "
            f"construct(s), {len(tested_families)} family/families, "
            f"{len(informative)} catalytically informative record(s), "
            f"{len(expression_failures)} expression failure(s). This round is a "
            f"result and is preserved in full; the five hypotheses below are "
            f"not ranked, because nothing here justifies a prior over them.")
    else:
        summary = (f"{len(positives)} confirmed hit(s); the no-hit differential "
                   f"is reported for completeness and is not triggered.")
    return NoHitDiagnosis(triggered, tuple(assessments), summary)


# ==========================================================================
# Next-round decisions
# ==========================================================================

@dataclass(frozen=True)
class FamilyVerdict:
    """What one family earned this round, and what that implies for the next."""

    family: str
    n_records: int
    n_positive: int
    n_negative: int
    n_wrong_product: int
    n_expression_failure: int
    n_untested: int
    hit_rate: float | None
    hit_rate_ci: tuple[float, float] | None
    tier: str                        # expand | hold | reduce | re-express | untested
    rationale: str
    suggested_quota: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family, "n_records": self.n_records,
            "n_positive": self.n_positive, "n_negative": self.n_negative,
            "n_wrong_product": self.n_wrong_product,
            "n_expression_failure": self.n_expression_failure,
            "n_untested": self.n_untested,
            "hit_rate": self.hit_rate,
            "hit_rate_ci": list(self.hit_rate_ci) if self.hit_rate_ci else None,
            "tier": self.tier, "rationale": self.rationale,
            "suggested_quota": self.suggested_quota,
        }


def next_round_quotas(
    records: Sequence[ExperimentRecord],
    families: Mapping[str, str],
    previous_quotas: Mapping[str, int] | None = None,
    *,
    floor: int = MIN_FAMILY_QUOTA_AFTER_NEGATIVE,
) -> list[FamilyVerdict]:
    """Per-family verdicts and quota suggestions, with the negatives kept.

    The rule is ordinal and stated rather than fitted, because one round
    produces nothing like enough data to fit anything:

    ``expand``
        the family produced a confirmed hit, or a wrong-configuration product
        -- which is a working scaffold with a fixable problem and is often the
        better engineering target of the two.
    ``re-express``
        the family's records are mostly expression failures. Its quota is
        **not** reduced: nothing was learnt about its catalysis, and cutting it
        would record a protein-production problem as a biological conclusion.
    ``reduce``
        catalytic negatives at stated limits and no hits. Reduced, never to
        zero: :data:`MIN_FAMILY_QUOTA_AFTER_NEGATIVE` keeps the boundary
        sampled, because a family cut to nothing can never overturn one
        unlucky round.
    ``untested``
        unchanged, because no information arrived.

    The Wilson interval on the hit rate is attached so a 0/4 family is not
    read as a demonstrated zero.
    """
    by_family: dict[str, list[ExperimentRecord]] = {}
    for record in records:
        key = families.get(_record_subject(record), "unassigned")
        by_family.setdefault(key, []).append(record)

    previous = dict(previous_quotas or {})
    out: list[FamilyVerdict] = []
    for family in sorted(by_family):
        group = by_family[family]
        positive = [r for r in group if r.outcome.is_positive]
        negative = [r for r in group
                    if r.outcome is OutcomeClass.NO_TARGET_PRODUCT_DETECTED]
        wrong = [r for r in group
                 if r.outcome is OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION]
        failed = [r for r in group
                  if r.outcome is OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE]
        untested = [r for r in group if r.outcome is OutcomeClass.NOT_TESTED]
        informative = len(positive) + len(negative) + len(wrong)
        base = previous.get(family, max(len(group), floor))

        rate: float | None = None
        ci: tuple[float, float] | None = None
        if informative:
            rate = len(positive) / informative
            ci = wilson_interval(len(positive), informative)

        if positive or wrong:
            tier = "expand"
            quota = base * 2
            rationale = (
                f"{len(positive)} confirmed hit(s) and {len(wrong)} "
                f"wrong-product/wrong-configuration record(s). A selective "
                f"enzyme pointing the wrong way is a working scaffold with a "
                f"fixable problem, so it expands the quota on the same footing "
                f"as a hit.")
        elif failed and len(failed) >= max(1, len(group) // 2):
            tier = "re-express"
            quota = base
            rationale = (
                f"{len(failed)} of {len(group)} record(s) are expression "
                f"failures. The quota is held, not cut: nothing was learnt "
                f"about this family's catalysis, and reducing it would record "
                f"a protein-production problem as a biological conclusion.")
        elif negative:
            tier = "reduce"
            quota = max(floor, base // 2)
            rationale = (
                f"{len(negative)} catalytic negative(s) at stated detection "
                f"limits and no hit"
                + (f"; hit rate {rate:.0%} with a {ci[0]:.0%}-{ci[1]:.0%} "
                   f"Wilson interval, so this is not a demonstrated zero"
                   if ci else "")
                + f". Reduced to {max(floor, base // 2)} rather than dropped: "
                  f"a family cut to nothing can never overturn one unlucky round.")
        else:
            tier = "untested"
            quota = base
            rationale = (
                f"{len(untested)} record(s) and no catalytic readout; no "
                f"information arrived, so the quota is unchanged.")

        out.append(FamilyVerdict(
            family=family, n_records=len(group), n_positive=len(positive),
            n_negative=len(negative), n_wrong_product=len(wrong),
            n_expression_failure=len(failed), n_untested=len(untested),
            hit_rate=rate, hit_rate_ci=ci, tier=tier, rationale=rationale,
            suggested_quota=quota))
    return out


# ==========================================================================
# The interface
# ==========================================================================

class IngestResults(ScientificInterface):
    """Parse a returned plate into records, then update three layers honestly.

    Gated on ``functional_criteria_confirmed``: the hit definition has to be a
    human decision taken before the data, and this is the step that would
    otherwise be tempted to take it afterwards.
    """

    name: ClassVar[str] = "ingest_results"
    description: ClassVar[str] = (
        "Ingest assay results into the full outcome taxonomy under the "
        "pre-registered criterion, keep negatives in the learning update, and "
        "produce a structured no-hit differential."
    )
    required_fields: ClassVar[tuple[str, ...]] = (
        "reaction.product.isomeric_smiles",
    )
    required_approvals: ClassVar[tuple[str, ...]] = ("functional_criteria_confirmed",)
    depends_on: ClassVar[tuple[str, ...]] = ("select_batch",)
    version: ClassVar[str] = "0.1.0"

    def execute(
        self,
        ctx: RunContext,
        *,
        results_csv: str | Path | None = None,
        rows: Sequence[Mapping[str, Any]] | None = None,
        assay_template: AssayTemplate | None = None,
        families: Mapping[str, str] | None = None,
        sequences: Mapping[str, str] | None = None,
        parent_sequence_hashes: Mapping[str, str] | None = None,
        previous_quotas: Mapping[str, int] | None = None,
        declared_cofactor_preferences: Mapping[str, str] | None = None,
        assay_run_id: str | None = None,
        criterion_override: Mapping[str, Any] | None = None,
        round_number: int = 1,
        **_: Any,
    ) -> ToolResult:
        """Ingest one round and write the records, the summary and the proposal."""
        if results_csv is None and rows is None:
            return ToolResult.failure(
                self.name,
                "nothing to ingest: pass results_csv=<path to the filled "
                "assay_results_template.csv> or rows=[...]. This step does not "
                "generate measurements.",
                code="no_results",
            )
        if assay_template is None:
            return ToolResult.failure(
                self.name,
                "no AssayTemplate supplied, so there is no pre-registered "
                "positive criterion. Refusing to classify: whatever the data "
                "look like would become the criterion.",
                code="no_pre_registered_criterion",
            )

        result = ToolResult(status=Status.SUCCESS)
        criterion = PositiveCriterion.from_template(assay_template)
        deviations = self._record_criterion_override(criterion, criterion_override,
                                                     result)

        try:
            parsed, problems = parse_assay_rows(
                results_csv if results_csv is not None else (rows or ()))
        except (OSError, ValueError) as exc:
            return ToolResult.failure(self.name, str(exc), code="unreadable_results")
        for problem in problems:
            result.add_flag("row_dropped", Severity.WARN, problem)
        if not parsed:
            return ToolResult.failure(
                self.name, "the results file contained no usable rows",
                code="no_rows")

        families = dict(families or {})
        sequences = dict(sequences or {})
        parent_sequence_hashes = dict(parent_sequence_hashes or {})
        run_id = assay_run_id or f"{parsed[0].plan_id or ctx.task.task_id}-assay"

        candidate_rows = [r for r in parsed if not r.is_control]
        control_rows = [r for r in parsed if r.is_control]
        groups = group_rows(candidate_rows)
        control_groups = group_rows(control_rows)

        # Attach the plate's own empty-vector background before anything is
        # scored, so a fold-over-background criterion is evaluated against the
        # control that shared the plate rather than quietly ignored.
        baseline_notes = attach_empty_vector_baselines(groups, parsed)
        for note in baseline_notes:
            if criterion.min_fold_over_empty_vector is not None:
                result.add_flag("empty_vector_baseline_missing",
                                Severity.WARN, note, subject="controls")
                result.add_uncertainty(
                    "empty_vector_baseline_missing", note,
                    affects=[g.candidate_id for g in groups
                             if g.empty_vector_baseline is None],
                    resolvable_by="run the empty-vector control on that plate")

        records: list[ExperimentRecord] = []
        unresolved: list[UnresolvedRow] = []
        classifications: list[tuple[MeasurementGroup, Classification]] = []

        for group in groups:
            classification = classify_group(
                group, criterion,
                fallback_limit_of_detection=assay_template.limit_of_detection,
                fallback_limit_unit=assay_template.limit_unit or "")
            classifications.append((group, classification))
            if classification.unresolved is not None:
                unresolved.append(classification.unresolved)
                severity = (Severity.BLOCKER
                            if classification.unresolved.reason_code
                            in ("indirect_signal_positive",
                                "negative_without_detection_limit")
                            else Severity.WARN)
                result.add_flag(
                    classification.unresolved.reason_code, severity,
                    classification.unresolved.reason,
                    subject=group.candidate_id)
                continue
            try:
                records.append(self._build_record(
                    ctx, group, classification, run_id, sequences,
                    parent_sequence_hashes, assay_template))
            except (ValueError, TypeError) as exc:
                result.add_flag(
                    "record_rejected_by_schema", Severity.BLOCKER,
                    f"{group.candidate_id} / {group.cofactor}: {exc}",
                    subject=group.candidate_id)
                unresolved.append(UnresolvedRow(
                    candidate_id=group.candidate_id, cofactor=group.cofactor,
                    reason_code="schema_rejected",
                    reason=str(exc),
                    required_to_resolve=(
                        "supply the field the record model requires; the "
                        "measurement is preserved here rather than reshaped "
                        "into a class it does not support"),
                    evidence={"reasons": list(classification.reasons)}))
            if group.disagreement:
                result.add_flag(
                    "replicate_disagreement", Severity.WARN,
                    f"{group.candidate_id} / {group.cofactor}: "
                    f"{group.disagreement}", subject=group.candidate_id)

        learning = build_active_learning_update(records)
        system_control = self._system_control_verdict(control_groups, criterion)
        diagnosis = diagnose_no_hits(
            records, groups, families,
            cofactor_conditions_run=sorted({g.cofactor for g in groups if g.cofactor}),
            declared_cofactor_preferences=declared_cofactor_preferences,
            system_control_worked=system_control)
        verdicts = next_round_quotas(records, families, previous_quotas)
        layers = self._update_layers(records, learning, verdicts, diagnosis,
                                     unresolved, criterion, deviations)

        if diagnosis.triggered:
            result.add_flag(
                "no_hits_in_round", Severity.WARN, diagnosis.summary)
            result.add_uncertainty(
                "no_hit_cause_unresolved",
                "Which of the five hypotheses explains the empty round? The "
                "differential names a discriminating experiment for each; none "
                "of them is 'the substrate has no natural catalyst', which one "
                "round cannot establish.",
                affects=[ctx.task.task_id],
                resolvable_by="the discriminating experiments in the diagnosis")
        if system_control is False:
            result.add_flag(
                "assay_system_control_failed", Severity.BLOCKER,
                "the control that demonstrates the assay system works did not "
                "produce its expected signal; no negative on this plate can be "
                "interpreted until the detection chain is revalidated")

        records_path = self._write_records(ctx, records)
        pending_path = self._write_unresolved(ctx, unresolved)
        summary_path = self._write_summary(
            ctx, records, unresolved, learning, layers, diagnosis, verdicts,
            criterion, deviations, round_number, system_control)
        proposal_path = self._write_next_round(
            ctx, verdicts, diagnosis, learning, unresolved, criterion,
            round_number)

        result.artifacts.append(Artifact(
            key="experiment_records", path=str(records_path), kind="table",
            sha256=sha256_file(records_path), n_records=len(records),
            summary=("one JSON record per (construct, cofactor condition) with "
                     "the full outcome class, detection limit and signed ee"),
        ))
        result.artifacts.append(Artifact(
            key="pending_confirmation", path=str(pending_path), kind="table",
            sha256=sha256_file(pending_path), n_records=len(unresolved),
            summary=("measurements that cannot become records without losing "
                     "information, with the confirmation each one needs; these "
                     "are not negatives"),
        ))
        result.artifacts.append(Artifact(
            key="round_summary", path=str(summary_path), kind="file",
            sha256=sha256_file(summary_path),
            summary="what changed in the data, model and decision layers",
        ))
        result.artifacts.append(Artifact(
            key="next_round_task", path=str(proposal_path), kind="file",
            sha256=sha256_file(proposal_path),
            summary=("the next round's proposed quotas, exploration scope and "
                     "selection, each traced to a round-1 outcome"),
        ))

        result.data.update({
            "n_rows": len(parsed),
            "n_groups": len(groups),
            "n_records": len(records),
            "outcome_counts": _outcome_counts(records),
            "n_unresolved": len(unresolved),
            "unresolved": [u.to_dict() for u in unresolved],
            "active_learning": learning.to_dict(),
            "layers": layers.to_dict(),
            "no_hit_diagnosis": diagnosis.to_dict(),
            "family_verdicts": [v.to_dict() for v in verdicts],
            "protocol_deviations": deviations,
            "positive_criteria": dict(criterion.raw),
            "positive_criteria_sha256": criterion.digest(),
            "system_control_worked": system_control,
            "records": [r.model_dump(mode="json") for r in records],
        })

        result.add_next(
            "propose_mutations",
            ("Engineer the confirmed parents; the scaffolds that made the wrong "
             "configuration are targets too"),
            {"parents": [r.record_id for r in records if r.outcome.is_positive
                         or r.outcome is
                         OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION]},
            requires_human=False)
        if unresolved:
            result.add_next(
                "confirm_product_identity",
                "Resolve the measurements held in pending_confirmation.jsonl "
                "before they are counted either way",
                {"n": len(unresolved), "file": str(pending_path)},
                requires_human=True)

        result.provenance = Provenance(
            tool=self.name, tool_version=self.version,
            inputs_sha256={
                "results": (sha256_file(results_csv) if results_csv is not None
                            else sha256_obj(list(rows or ()))),
                "assay_template": sha256_obj(
                    assay_template.model_dump(mode="json")),
            },
            databases={},
            models={},
            parameters={
                "round_number": round_number,
                "assay_run_id": run_id,
                "assay_template_id": assay_template.template_id,
                "positive_criteria": dict(criterion.raw),
                "positive_criteria_sha256": criterion.digest(),
                "criterion_override_refused": bool(criterion_override),
                "protocol_deviations": deviations,
                "fallback_limit_of_detection": assay_template.limit_of_detection,
                "replicate_aggregation": "median per condition, peak areas summed",
                "allow_network": ctx.policy.allow_network,
            },
            random_seed=ctx.seed_for(self.name),
        )

        counts = _outcome_counts(records)
        headline = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        if result.blockers:
            result.status = Status.PARTIAL
            result.message = (f"{len(records)} record(s) ingested ({headline}); "
                              f"{len(unresolved)} held unresolved; "
                              f"{len(result.blockers)} blocking problem(s). "
                              f"{layers.summary()}")
        else:
            result.message = (f"{len(records)} record(s) ingested ({headline}). "
                              f"{layers.summary()}")
        return result

    # -- criterion integrity ------------------------------------------------
    def _record_criterion_override(
        self, criterion: PositiveCriterion,
        override: Mapping[str, Any] | None, result: ToolResult,
    ) -> list[dict[str, Any]]:
        """Refuse an override, and write the attempt down.

        There is deliberately no branch that applies ``override``. Moving the
        endpoint after seeing the data makes the primary result unfalsifiable,
        and the usual way it happens is not dishonesty but a reasonable-looking
        argument made in front of a plate. Recording the attempt keeps the
        argument available to a reviewer instead of erasing it.
        """
        if not override:
            return []
        deviation = {
            "kind": "positive_criterion_override_refused",
            "attempted": dict(override),
            "applied": dict(criterion.raw),
            "applied_sha256": criterion.digest(),
            "statement": (
                "A positive criterion was supplied at ingestion time and was "
                "NOT applied. The pre-registered criterion from AssayTemplate "
                f"{criterion.source_template_id} is the only one used. "
                "Changing the endpoint after seeing the data invalidates the "
                "primary result; if the criterion is genuinely wrong, say so, "
                "re-register it, and re-run the round as a new primary "
                "endpoint."),
        }
        result.add_flag(
            "protocol_deviation_refused", Severity.WARN, deviation["statement"])
        result.add_next(
            "review_protocol_deviation",
            "An attempt was made to change the pre-registered endpoint at "
            "ingestion time; a human should decide what that means for the "
            "round's primary result",
            {"attempted": dict(override)}, requires_human=True)
        return [deviation]

    # -- record construction ------------------------------------------------
    def _build_record(
        self, ctx: RunContext, group: MeasurementGroup,
        classification: Classification, run_id: str,
        sequences: Mapping[str, str], parents: Mapping[str, str],
        template: AssayTemplate,
    ) -> ExperimentRecord:
        """Turn one classified group into an :class:`ExperimentRecord`.

        The record carries the condition it was measured under, not the task's
        whole condition set: the cofactor actually offered is part of the
        record key, and copying the full list would make two conditions look
        like one measurement.
        """
        task = ctx.task
        cofactor = self._cofactor_for(group, task.conditions.cofactor_options)
        conditions = Conditions(**{
            **task.conditions.model_dump(),
            "cofactor_options": [cofactor] if cofactor else [],
        })
        limit = group.limit_of_detection
        limit_unit = group.limit_unit
        if limit is None and template.limit_of_detection is not None:
            limit = template.limit_of_detection
            limit_unit = template.limit_unit or ""

        outcome = classification.outcome
        assert outcome is not None
        detection = Detection(
            method=group.detection_method or template.method,
            limit_of_detection=limit, limit_unit=limit_unit or None,
            authentic_standard=group.authentic_standard,
            confirms_product_identity=group.confirms_product_identity,
            chiral_method_validated=group.chiral_method_validated,
        )
        sequence = sequences.get(group.candidate_id)
        parent_hash = parents.get(group.candidate_id)
        is_variant = bool(group.mutations or group.parent_candidate_id
                          or parent_hash)
        product = (ProductSpec(**task.reaction.product.model_dump())
                   if outcome.is_positive else None)
        evidence = EvidenceRef(
            source_type="internal_experiment",
            identifier=f"{run_id}:{group.candidate_id}:{group.cofactor}",
            locator=f"wells {', '.join(sorted({r.well for r in group.rows if r.well}))}",
            strength=EvidenceStrength.SEQUENCE_LEVEL_EXPERIMENTAL
            if sequence else EvidenceStrength.COMPUTATIONAL_CONSTRUCT,
            extracted_by="instrument_export",
            retrieved_at=None,
            experiment_activity_id=run_id,
        )
        return ExperimentRecord(
            record_id=f"{run_id}:{group.candidate_id}:{group.cofactor}",
            sequence=sequence,
            accession=None,
            is_variant=is_variant,
            parent_sequence_sha256=parent_hash if is_variant else None,
            mutations=list(group.mutations),
            construct_description=(group.rows[0].construct_id or None
                                   if group.rows else None),
            substrate=task.reaction.substrate,
            product_observed=product,
            reaction_class=task.reaction.reaction_class,
            reaction_direction=ReactionDirection.FORWARD_AS_TARGET,
            cofactor=cofactor,
            conditions=conditions,
            outcome=outcome,
            detection=detection,
            conversion_pct=group.conversion_pct,
            ee_target_pct=group.ee_target_pct,
            measurement_type=group.measurement_type or None,
            measurement_value=group.measurement_value,
            measurement_unit=group.measurement_unit or None,
            soluble_expression=group.expressed_soluble,
            evidence=[evidence],
            notes="; ".join(
                [f"{group.n_replicates} replicate(s), aggregated by median"]
                + list(classification.reasons)
                + ([group.disagreement] if group.disagreement else [])),
        )

    @staticmethod
    def _cofactor_for(group: MeasurementGroup,
                      declared: Sequence[CofactorSpec]) -> CofactorSpec | None:
        """Match the reported cofactor to a declared spec, or build a minimal one.

        Matching by name first keeps the task's recorded transfer atom and
        recycling system attached to the record. When the plate names a
        cofactor the task never declared, a minimal spec is built from what the
        plate says rather than from the nearest declared option -- substituting
        NADPH's spec for a well that ran NADH would misrecord the experiment.
        """
        name = (group.cofactor or "").strip()
        if not name:
            return None
        for spec in declared:
            if spec.name.strip().lower() == name.lower():
                return spec
        try:
            state = CofactorState(group.cofactor_state.strip().lower())
        except ValueError:
            state = CofactorState.UNKNOWN
        return CofactorSpec(name=name, state=state)

    @staticmethod
    def _system_control_verdict(
        control_groups: Sequence[MeasurementGroup], criterion: PositiveCriterion
    ) -> bool | None:
        """Did the assay-system positive control produce its expected signal?

        ``None`` when no such control was reported, and ``None`` is the
        important value: it is not "the control passed", and the no-hit
        differential treats a missing system control as leaving a dead
        detection chain unexcluded.
        """
        for group in control_groups:
            role = (group.role or "").lower()
            name = (group.candidate_id or "").lower()
            if "positive" not in role and "positive" not in name:
                continue
            met, _ = criterion.evaluate(group)
            if met is True:
                return True
            if met is False:
                return False
            if group.conversion_pct is not None:
                return group.conversion_pct > 0
            if group.measurement_value is not None:
                return group.measurement_value > 0
            return None
        return None

    # -- three layers -------------------------------------------------------
    def _update_layers(
        self, records: Sequence[ExperimentRecord],
        learning: ActiveLearningUpdate, verdicts: Sequence[FamilyVerdict],
        diagnosis: NoHitDiagnosis, unresolved: Sequence[UnresolvedRow],
        criterion: PositiveCriterion, deviations: Sequence[Mapping[str, Any]],
    ) -> LayerUpdate:
        """Report what moved in each layer, and say so even when nothing did."""
        update = LayerUpdate()

        # -- data layer ----------------------------------------------------
        counts = _outcome_counts(records)
        limits = {r.detection.limit_of_detection for r in records
                  if r.detection.limit_of_detection is not None}
        identified = sum(1 for r in records
                         if r.detection.confirms_product_identity)
        update.data = {
            "n_records": len(records),
            "outcome_counts": counts,
            "detection_limits_recorded": sorted(limits),
            "records_with_product_identity": identified,
            "expression_outcomes": {
                "soluble": sum(1 for r in records if r.soluble_expression is True),
                "insoluble": sum(1 for r in records if r.soluble_expression is False),
                "unrecorded": sum(1 for r in records if r.soluble_expression is None),
            },
            "unresolved_measurements": len(unresolved),
        }
        update.data_changed = bool(records or unresolved)
        if records:
            update.data_changes.append(
                f"{len(records)} record(s) added across "
                f"{len(counts)} outcome class(es): "
                + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        if limits:
            update.data_changes.append(
                f"detection limit(s) {sorted(limits)} now attached to the "
                f"negatives, which is what makes them mean anything")
        if unresolved:
            update.data_changes.append(
                f"{len(unresolved)} measurement(s) held unresolved rather than "
                f"forced into an outcome class")

        # -- model layer ---------------------------------------------------
        informative = learning.n_catalytically_informative
        n_pos = len(learning.positives)
        ci = wilson_interval(n_pos, informative) if informative else None
        expression_total = (update.data["expression_outcomes"]["soluble"]
                            + update.data["expression_outcomes"]["insoluble"])
        expression_ci = (
            wilson_interval(update.data["expression_outcomes"]["insoluble"],
                            expression_total) if expression_total else None)
        mutation_effects = [
            {"record_id": r.record_id, "mutations": r.mutations,
             "parent_sequence_sha256": r.parent_sequence_sha256,
             "outcome": r.outcome.value, "conversion_pct": r.conversion_pct,
             "ee_target_pct": r.ee_target_pct}
            for r in records if r.is_variant and r.mutations
        ]
        update.model = {
            "specificity": {
                "n_informative_records": informative,
                "hit_rate": (n_pos / informative) if informative else None,
                "hit_rate_wilson_95": list(ci) if ci else None,
                "note": ("the interval is a sampling interval over the "
                         "constructs actually tested; it does not cover model "
                         "error, pocket-mapping error or assay bias"),
            },
            "expression_risk": {
                "n_constructs_scored": expression_total,
                "insoluble_fraction": (
                    update.data["expression_outcomes"]["insoluble"]
                    / expression_total) if expression_total else None,
                "insoluble_wilson_95": list(expression_ci) if expression_ci else None,
                "note": ("built from expression failures, which are excluded "
                         "from the catalytic label set entirely"),
            },
            "mutation_effects": mutation_effects,
            "uncertainty": {
                "unresolved_measurements": len(unresolved),
                "replicate_disagreements": sum(
                    1 for r in records if "disagree" in (r.notes or "")),
                "criterion_sha256": criterion.digest(),
            },
        }
        if informative:
            update.model_changed = True
            update.model_changes.append(
                f"specificity model updated on {informative} catalytically "
                f"informative record(s), {len(learning.catalytic_negatives)} of "
                f"them negatives that were retained")
        if expression_total:
            update.model_changed = True
            update.model_changes.append(
                f"expression-risk model updated on {expression_total} "
                f"construct(s); expression failures are held out of the "
                f"catalytic labels")
        if mutation_effects:
            update.model_changed = True
            update.model_changes.append(
                f"{len(mutation_effects)} variant record(s) now carry a "
                f"measured effect against their parent")
        if not update.model_changed:
            update.model_changes.append(
                "no model moved: the round produced no catalytically "
                "informative record")

        # -- decision layer -------------------------------------------------
        quotas = {v.family: v.suggested_quota for v in verdicts}
        update.decision = {
            "next_round_family_quotas": quotas,
            "family_verdicts": [v.to_dict() for v in verdicts],
            "exploration_scope": self._exploration_scope(verdicts, diagnosis),
            "candidate_selection": {
                "engineer": sorted({
                    r.record_id for r in records
                    if r.outcome.is_positive
                    or r.outcome is OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION}),
                "re_express": sorted({
                    r.record_id for r in records
                    if r.outcome is OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE}),
                "retest_under_other_cofactor": sorted({
                    r.record_id for r in records
                    if r.outcome is OutcomeClass.NO_TARGET_PRODUCT_DETECTED}),
            },
            "protocol_deviations": list(deviations),
        }
        update.decision_changed = bool(verdicts or diagnosis.triggered
                                       or deviations)
        for verdict in verdicts:
            if verdict.tier != "untested":
                update.decision_changes.append(
                    f"{verdict.family}: {verdict.tier} -> quota "
                    f"{verdict.suggested_quota}")
        if diagnosis.triggered:
            update.decision_changes.append(
                "the round returned no hit, so the next round's scope is set "
                "by the no-hit differential rather than by a ranking")
        if deviations:
            update.decision_changes.append(
                f"{len(deviations)} protocol deviation(s) recorded")
        if not update.decision_changes:
            update.decision_changes.append("no decision changed")
        return update

    @staticmethod
    def _exploration_scope(verdicts: Sequence[FamilyVerdict],
                           diagnosis: NoHitDiagnosis) -> dict[str, Any]:
        """What the next round should widen, hold or narrow, and why."""
        widen = [v.family for v in verdicts if v.tier == "expand"]
        hold = [v.family for v in verdicts if v.tier in ("re-express", "untested")]
        narrow = [v.family for v in verdicts if v.tier == "reduce"]
        actions = [a.discriminating_experiment for a in diagnosis.assessments
                   if a.status == "consistent"]
        return {
            "widen": widen, "hold": hold, "narrow_but_not_drop": narrow,
            "floor_rationale": (
                f"no family goes below {MIN_FAMILY_QUOTA_AFTER_NEGATIVE} "
                f"slot(s): a family cut to zero can never overturn one round "
                f"of negatives"),
            "diagnostic_actions": actions,
            "note": ("scope changes are driven by outcome class, not by a "
                     "score: a wrong-configuration product expands a family "
                     "just as a hit does, because it is a working scaffold"),
        }

    # -- artifacts ----------------------------------------------------------
    def _write_records(self, ctx: RunContext,
                       records: Sequence[ExperimentRecord]) -> Path:
        """Write ``experiment_records.jsonl``, one record per line."""
        path = ctx.path("ingest_results", "experiment_records.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record.model_dump(mode="json"),
                                    ensure_ascii=False, sort_keys=True) + "\n")
        return path

    def _write_unresolved(self, ctx: RunContext,
                          unresolved: Sequence[UnresolvedRow]) -> Path:
        """Write ``pending_confirmation.jsonl``, always -- an empty file is a claim."""
        path = ctx.path("ingest_results", "pending_confirmation.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for item in unresolved:
                fh.write(json.dumps(item.to_dict(), ensure_ascii=False,
                                    sort_keys=True) + "\n")
        return path

    def _write_summary(
        self, ctx: RunContext, records: Sequence[ExperimentRecord],
        unresolved: Sequence[UnresolvedRow], learning: ActiveLearningUpdate,
        layers: LayerUpdate, diagnosis: NoHitDiagnosis,
        verdicts: Sequence[FamilyVerdict], criterion: PositiveCriterion,
        deviations: Sequence[Mapping[str, Any]], round_number: int,
        system_control: bool | None,
    ) -> Path:
        """Write ``round_summary.md``: what happened, and which layer moved."""
        counts = _outcome_counts(records)
        lines: list[str] = [
            f"# Round {round_number} summary",
            "",
            f"Pre-registered criterion (AssayTemplate "
            f"{criterion.source_template_id}, sha256 {criterion.digest()[:16]}): "
            f"`{json.dumps(dict(criterion.raw), sort_keys=True)}`",
            "",
            "This criterion was fixed before the plate was run and is the only "
            "one applied. No code path in `ingest_results` adjusts it after "
            "seeing the data.",
            "",
            "## Outcomes",
            "",
        ]
        for name in (OutcomeClass.CONFIRMED_TARGET_PRODUCT,
                     OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION,
                     OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                     OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE,
                     OutcomeClass.NOT_TESTED):
            lines.append(f"- `{name.value}`: {counts.get(name.value, 0)} "
                         f"-- {name.claim()}")
        lines += [
            "",
            f"- held unresolved (not counted either way): {len(unresolved)}",
            "",
            "## Layers",
            "",
            f"- data layer changed: {layers.data_changed}",
        ]
        lines += [f"  - {c}" for c in layers.data_changes]
        lines += ["", f"- model layer changed: {layers.model_changed}"]
        lines += [f"  - {c}" for c in layers.model_changes]
        lines += ["", f"- decision layer changed: {layers.decision_changed}"]
        lines += [f"  - {c}" for c in layers.decision_changes]
        lines += [
            "",
            "## Active learning",
            "",
            learning.note,
            "",
            f"- positives: {len(learning.positives)}",
            f"- catalytic negatives retained: {len(learning.catalytic_negatives)}",
            f"- wrong product / configuration: "
            f"{len(learning.wrong_product_or_configuration)}",
            f"- expression failures (held out of catalytic labels): "
            f"{len(learning.expression_failures)}",
            f"- untested: {len(learning.untested)}",
            "",
            "## Families",
            "",
        ]
        for verdict in verdicts:
            lines.append(f"- **{verdict.family}** -- {verdict.tier}, next-round "
                         f"quota {verdict.suggested_quota}. {verdict.rationale}")
        lines += ["", "## Controls", "",
                  f"- assay-system positive control: "
                  f"{_system_control_text(system_control)}"]
        lines += ["", "## No-hit differential", "", diagnosis.summary, ""]
        for assessment in diagnosis.assessments:
            lines.append(f"### {assessment.hypothesis.value} -- {assessment.status}")
            lines.append("")
            lines.append(f"_{assessment.hypothesis.question()}?_")
            lines.append("")
            for item in assessment.supported_by:
                lines.append(f"- supports: {item}")
            for item in assessment.contradicted_by:
                lines.append(f"- against: {item}")
            lines.append(f"- discriminating experiment: "
                         f"{assessment.discriminating_experiment}")
            lines.append("")
        if deviations:
            lines += ["## Protocol deviations", ""]
            for deviation in deviations:
                lines.append(f"- {deviation['statement']} Attempted: "
                             f"`{json.dumps(deviation['attempted'], sort_keys=True)}`")
            lines.append("")
        path = ctx.path("ingest_results", "round_summary.md")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _write_next_round(
        self, ctx: RunContext, verdicts: Sequence[FamilyVerdict],
        diagnosis: NoHitDiagnosis, learning: ActiveLearningUpdate,
        unresolved: Sequence[UnresolvedRow], criterion: PositiveCriterion,
        round_number: int,
    ) -> Path:
        """Write ``next_round_task.yaml``: a proposal, with each item's cause."""
        document: dict[str, Any] = {
            "proposed_round": round_number + 1,
            "derived_from": {
                "round": round_number,
                "criterion_sha256": criterion.digest(),
                "positives": list(learning.positives),
                "negatives_retained": list(learning.catalytic_negatives),
                "expression_failures": list(learning.expression_failures),
            },
            "family_quotas": {v.family: v.suggested_quota for v in verdicts},
            "family_rationale": {v.family: v.rationale for v in verdicts},
            "exploration_scope": self._exploration_scope(verdicts, diagnosis),
            "required_before_next_round": [
                u.required_to_resolve for u in unresolved
            ] + [a.discriminating_experiment for a in diagnosis.assessments
                 if a.status == "consistent"],
            "endpoint": {
                "statement": (
                    "the next round reuses this criterion unless it is "
                    "deliberately re-registered before any new data exist; a "
                    "criterion changed between rounds makes the two rounds "
                    "incomparable and must be stated as such"),
                "positive_criteria": dict(criterion.raw),
            },
            "no_hit_diagnosis": diagnosis.to_dict(),
        }
        path = ctx.path("ingest_results", "next_round_task.yaml")
        path.write_text(
            yaml.safe_dump(document, sort_keys=False, allow_unicode=True,
                           default_flow_style=False),
            encoding="utf-8")
        return path


# ==========================================================================
# Small helpers
# ==========================================================================

def _system_control_text(verdict: bool | None) -> str:
    if verdict is True:
        return "produced its expected signal"
    if verdict is False:
        return ("FAILED -- no negative on this plate is interpretable until the "
                "detection chain is revalidated")
    return "not reported, so a dead detection chain is not excluded"


def _outcome_counts(records: Sequence[ExperimentRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for record in records:
        out[record.outcome.value] = out.get(record.outcome.value, 0) + 1
    return out


def _record_subject(record: ExperimentRecord) -> str:
    """The construct id inside a record id of the form ``run:construct:cofactor``.

    The middle segment is rejoined rather than taken as ``parts[1]`` so a
    construct id that itself contains a colon -- a variant label, a tagged
    construct -- is not silently truncated into a different key, which would
    quietly move its results into another family's bucket.
    """
    parts = record.record_id.split(":")
    if len(parts) >= 3:
        return ":".join(parts[1:-1])
    return record.record_id


def _consensus(values: Sequence[bool | None]) -> bool | None:
    """``True``/``False`` only when every reported replicate agrees."""
    seen = {v for v in values if v is not None}
    if len(seen) == 1:
        return seen.pop()
    return None


def _median(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None and not math.isnan(v)]
    if not present:
        return None
    return float(statistics.median(present))


def _as_float(value: Any) -> float | None:
    """Float, or ``None``. Never 0.0 for a blank: a blank is not a measurement."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_bool(value: Any) -> bool | None:
    """Tri-state: ``True``, ``False`` or ``None`` for blank/unknown.

    ``None`` is a distinct answer throughout this module: "we did not record
    whether it expressed" is not "it did not express", and conflating them
    turns missing metadata into a biological result.
    """
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _YES:
        return True
    if text in _NO:
        return False
    return None
