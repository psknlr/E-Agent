"""Endpoints for a screening round, computed against a criterion that cannot move.

THE ONE FAILURE THIS MODULE IS BUILT AROUND
===========================================
A campaign spends 96 wells against a bar written down in advance -- say 20%
conversion with the target configuration dominant. The plate comes back with
nothing over 12%. There is then an entirely reasonable-sounding conversation
about how 10% is really the meaningful bar for a first round, the endpoint
becomes "10%", and the paper reports four hits. Nothing in the results table
records that this happened, and no reader can detect it.

Every primary-endpoint function here therefore takes a
:class:`PreRegistration` and refuses to run against anything else.
:meth:`PreRegistration.assert_matches` raises
:class:`CriterionChangedError` when the criterion in use differs from the
registered one, and :meth:`PreRegistration.assert_unchanged` re-derives the
digest from the registration's own contents on every call, so editing the
mapping a registration was built from is caught too. There is no parameter
anywhere in this module that accepts pre-computed hit labels, which is the
other way the guard would be bypassed: the hit definition is applied *here*,
from the registered criterion, or the number is not produced.

The criterion object and its digest are
:class:`~eagent.tools.ingest_results.PositiveCriterion` and
``sha256_obj(positive_criteria)`` -- the same ones ``select_batch`` writes
into ``experiment_plan.yaml`` before the plate is run and ``ingest_results``
classifies against. Reusing them is what makes the registration checkable
against the plan file on disk rather than against a copy of it in memory.

WHY EVERY RATE CARRIES TWO DENOMINATORS OR AN INTERVAL
======================================================
Three hits out of eight expressed constructs is 37 percent. The same three out
of twenty-four submitted is 12 percent. Both are true, and only one gets
quoted. :class:`HitRateReport` follows
:class:`eagent.datalayer.house_db.HitRate` and returns both, with no attribute
that could be read as "the" hit rate.

Every rate also carries a Wilson interval from
:func:`eagent.science.robustness.wilson_interval`, because 3/8 and 30/80 are
the same fraction and very different evidence, and a 96-well round produces
numbers much closer to the first.

ee IS SIGNED, AND IS NOT AVERAGED AS AN ABSOLUTE
================================================
:func:`eagent.schemas.record.ee_target` is signed toward the target
enantiomer, so a beautifully selective failure reports -94% rather than "94%
ee". :func:`signed_ee_aggregate` pools peak areas where it has them and
otherwise reports a median with its range; it never means the absolute values,
which would turn a batch half of which made the wrong enantiomer into a
success.

ROUND TWO COMPARES UNDER ONE CONDITION SET OR NOT AT ALL
========================================================
A variant assayed at a different pH, cofactor state, substrate loading or
endpoint has not been shown to be better than its parent.
:func:`variant_versus_parent` raises
:class:`~eagent.datalayer.house_db.ConditionMismatchError` -- the same error
the house database raises -- rather than returning a delta with a footnote.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..datalayer.house_db import (
    CONDITION_KEY_FIELDS,
    ConditionMismatchError,
    ExpressionStatus,
    condition_key,
)
from ..errors import EAgentError
from ..provenance import sha256_obj, utc_now
from ..schemas.record import ExperimentRecord, OutcomeClass, ee_target
from ..schemas.templates import AssayTemplate
from ..science.robustness import DEFAULT_WILSON_Z, wilson_interval
from ..tools.ingest_results import RECOGNISED_CRITERION_KEYS, PositiveCriterion

__all__ = [
    "CriterionChangedError",
    "EndpointMismatchError",
    "HitVerdict",
    "PreRegistration",
    "OutcomeRow",
    "Rate",
    "PrecisionAtK",
    "HitRateReport",
    "ConfigurationCompliance",
    "SignedEEAggregate",
    "ReplicateCheck",
    "VariantParentComparison",
    "RefusedPair",
    "RoundTwoReport",
    "criterion_from_mapping",
    "precision_at_k",
    "hit_rates",
    "configuration_compliance",
    "signed_ee_aggregate",
    "variant_versus_parent",
    "round_two_report",
    "replicate_reproducibility",
]


class CriterionChangedError(EAgentError):
    """The positivity criterion in use is not the one that was pre-registered.

    Raised, never worked around. A criterion that moves after the data arrive
    makes the primary endpoint unfalsifiable, and the move is invisible in
    every artefact downstream, so the only place it can be stopped is at the
    moment the number is computed.
    """


class EndpointMismatchError(EAgentError):
    """Two measurements on different endpoints were about to be compared.

    An initial rate and a conversion are not the same quantity, and
    subtracting one from the other produces a number with no unit and no
    meaning. Raised instead of returning a delta.
    """


# ==========================================================================
# Rates
# ==========================================================================

@dataclass(frozen=True)
class Rate:
    """One proportion, its denominator's meaning, and its Wilson interval.

    ``denominator_meaning`` is required because the arguments about screening
    results are almost never about the numerator. "Out of what" is the whole
    question, and a bare float loses it the moment it is copied into a slide.
    """

    name: str
    successes: int
    trials: int
    denominator_meaning: str
    z: float = DEFAULT_WILSON_Z

    def __post_init__(self) -> None:
        if self.successes < 0 or self.trials < 0:
            raise ValueError(
                f"{self.name}: counts cannot be negative "
                f"({self.successes}/{self.trials})")
        if self.successes > self.trials:
            raise ValueError(
                f"{self.name}: {self.successes} successes out of {self.trials} "
                f"trials is impossible; the two counts were taken over "
                f"different row sets")
        if not self.denominator_meaning.strip():
            raise ValueError(
                f"{self.name}: a rate must say what its denominator counts")

    @property
    def point(self) -> float | None:
        """The fraction, or ``None`` when nothing was counted.

        ``None`` rather than ``0.0``: an empty denominator is an undefined
        rate, and rendering it as zero is how a round that measured nothing
        becomes a plotted point at the bottom of a chart.
        """
        if self.trials <= 0:
            return None
        return self.successes / self.trials

    @property
    def interval(self) -> tuple[float, float] | None:
        """Wilson score interval, or ``None`` when there is no proportion."""
        if self.trials <= 0:
            return None
        return wilson_interval(self.successes, self.trials, z=self.z)

    def describe(self) -> str:
        if self.point is None:
            return (f"{self.name}: undefined ({self.successes}/{self.trials}); "
                    f"denominator counts {self.denominator_meaning}")
        low, high = self.interval            # type: ignore[misc]
        return (f"{self.name}: {self.successes}/{self.trials} = "
                f"{self.point * 100:.1f}% [{low * 100:.1f}%, {high * 100:.1f}%] "
                f"95% Wilson; denominator counts {self.denominator_meaning}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "successes": self.successes,
            "trials": self.trials,
            "denominator_meaning": self.denominator_meaning,
            "point": self.point,
            "wilson_interval": list(self.interval) if self.interval else None,
        }


# ==========================================================================
# One scored construct
# ==========================================================================

@dataclass(frozen=True)
class OutcomeRow:
    """One construct's result, in the shape the registered criterion reads.

    The attribute surface is deliberately the one
    :meth:`~eagent.tools.ingest_results.PositiveCriterion.evaluate` consumes --
    ``conversion_pct``, ``measurement_type``, ``measurement_value``,
    ``ee_target_pct``, ``authentic_standard``, ``chiral_method_validated`` --
    so the pre-registered criterion is applied by the same code that
    classified the plate, rather than by a second implementation of the same
    bar that can drift from it.

    Everything else on the row is here because dropping it loses a distinction
    the endpoints depend on: ``outcome`` keeps "expressed and did nothing"
    apart from "never expressed", ``is_control`` keeps a control out of a
    discovery denominator, and ``conditions`` is what makes a round-two
    comparison legal or refused.
    """

    candidate_id: str
    outcome: OutcomeClass = OutcomeClass.NOT_TESTED
    record_id: str | None = None
    is_control: bool = False
    expression_status: ExpressionStatus = ExpressionStatus.NOT_ASSESSED
    soluble_expression: bool | None = None
    conversion_pct: float | None = None
    measurement_type: str | None = None
    measurement_value: float | None = None
    measurement_unit: str | None = None
    ee_target_pct: float | None = None
    target_peak_area: float | None = None
    opposite_peak_area: float | None = None
    authentic_standard: bool | None = None
    chiral_method_validated: bool | None = None
    conditions: Mapping[str, Any] = field(default_factory=dict)
    cofactor_species: str | None = None
    cofactor_state: str | None = None
    sequence_sha256: str | None = None
    parent_sequence_sha256: str | None = None
    mutations: tuple[str, ...] = ()
    replicate_values: tuple[float, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        derived = self._ee_from_areas()
        if derived is not None:
            if self.ee_target_pct is None:
                object.__setattr__(self, "ee_target_pct", derived)
            elif abs(self.ee_target_pct - derived) > 1e-6:
                raise ValueError(
                    f"{self.candidate_id}: ee_target_pct {self.ee_target_pct:+.3f}% "
                    f"disagrees with the {derived:+.3f}% implied by the peak "
                    f"areas. One of the two is wrong and guessing which would "
                    f"decide an enantioselectivity claim by coin flip")

    def _ee_from_areas(self) -> float | None:
        """Signed ee from peak areas, via the schema's own definition.

        Uses :func:`eagent.schemas.record.ee_target` rather than a local
        formula so there is exactly one place in the codebase where the sign
        convention lives.
        """
        if self.target_peak_area is None or self.opposite_peak_area is None:
            return None
        if self.target_peak_area + self.opposite_peak_area <= 0:
            return None               # nothing quantified; ee is undefined
        return ee_target(self.target_peak_area, self.opposite_peak_area)

    @property
    def expressed(self) -> bool | None:
        """Tri-state soluble expression, on the same rule as the house database.

        ``None`` stays ``None``. Inferring expression from "this row informs
        catalysis" would invent expression data for a construct nobody ran a
        gel on and quietly enlarge the expressed-only denominator, which is one
        of the two numbers :class:`HitRateReport` exists to keep honest.
        """
        if self.outcome is OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE:
            return False
        explicit = self.expression_status.expressed_solubly
        if explicit is not None:
            return explicit
        if self.soluble_expression is not None:
            return self.soluble_expression
        if self.outcome is OutcomeClass.CONFIRMED_TARGET_PRODUCT:
            return True               # the product was made, so protein existed
        return None

    @property
    def condition_key(self) -> str:
        """Identity of the assay context, cofactor identity and state included."""
        return condition_key(self.conditions, self.cofactor_species,
                             self.cofactor_state)

    @classmethod
    def from_record(cls, record: ExperimentRecord, *, is_control: bool = False,
                    candidate_id: str | None = None,
                    replicate_values: Sequence[float] = ()) -> "OutcomeRow":
        """Build from the canonical record type, carrying its conditions across."""
        conditions = {
            "pH": record.conditions.pH,
            "temperature_C": record.conditions.temperature_C,
            "buffer": record.conditions.buffer,
            "solvent_system": record.conditions.solvent_system,
            "cosolvent_fraction": record.conditions.cosolvent_fraction,
            "substrate_concentration_mM": record.conditions.substrate_concentration_mM,
            "enzyme_loading": record.conditions.enzyme_loading,
            "reaction_time_h": record.conditions.reaction_time_h,
            "expression_host": record.conditions.expression_host,
        }
        cofactor = record.cofactor
        return cls(
            candidate_id=candidate_id or record.record_id,
            record_id=record.record_id,
            outcome=record.outcome,
            is_control=is_control,
            soluble_expression=record.soluble_expression,
            conversion_pct=record.conversion_pct,
            measurement_type=record.measurement_type,
            measurement_value=record.measurement_value,
            measurement_unit=record.measurement_unit,
            ee_target_pct=record.ee_target_pct,
            authentic_standard=record.detection.authentic_standard,
            chiral_method_validated=record.detection.chiral_method_validated,
            conditions=conditions,
            cofactor_species=cofactor.name if cofactor else None,
            cofactor_state=cofactor.state.value if cofactor else None,
            sequence_sha256=record.sequence_sha256,
            parent_sequence_sha256=record.parent_sequence_sha256,
            mutations=tuple(record.mutations),
            replicate_values=tuple(replicate_values),
        )


# ==========================================================================
# The pre-registration
# ==========================================================================

@dataclass(frozen=True)
class HitVerdict:
    """Whether one row met the registered definition, and why.

    Three-valued on purpose. "Did not meet the bar" and "we could not tell"
    have different consequences: the first is a result, the second is a
    missing measurement, and only the first may be reported as a negative.
    """

    candidate_id: str
    met: bool | None
    reasons: tuple[str, ...] = ()

    @property
    def is_hit(self) -> bool:
        return self.met is True

    @property
    def undecidable(self) -> bool:
        return self.met is None


def criterion_from_mapping(
    criteria: Mapping[str, Any], *, source_template_id: str | None = None,
) -> PositiveCriterion:
    """Build the criterion object from a plan file's ``positive_criteria`` block.

    ``select_batch`` writes that mapping, and its sha256, into
    ``experiment_plan.yaml`` before the plate is run, so the mapping -- not the
    template object -- is what survives to evaluation time. The key whitelist
    is :data:`~eagent.tools.ingest_results.RECOGNISED_CRITERION_KEYS`, enforced
    here exactly as :meth:`PositiveCriterion.from_template` enforces it,
    because a criterion key nobody implements is a criterion that passes
    everything.
    """
    raw = dict(criteria or {})
    if not raw:
        raise CriterionChangedError(
            "an empty positive_criteria block is not a pre-registration: with "
            "no bar recorded in advance, whatever the data look like becomes "
            "the criterion")
    unknown = sorted(set(raw) - RECOGNISED_CRITERION_KEYS)
    if unknown:
        raise CriterionChangedError(
            f"criterion key(s) {unknown} are not implemented by the ingest "
            f"step and would be ignored; an ignored criterion passes "
            f"everything. Recognised keys: {sorted(RECOGNISED_CRITERION_KEYS)}")

    def as_float(value: Any) -> float | None:
        return None if value is None else float(value)

    def as_str(value: Any) -> str | None:
        text = None if value is None else str(value).strip()
        return text or None

    return PositiveCriterion(
        min_conversion_pct=as_float(raw.get("min_conversion_pct")),
        min_measurement_value=as_float(raw.get("min_measurement_value")),
        measurement_type=as_str(raw.get("measurement_type")),
        min_ee_target_pct=as_float(raw.get("min_ee_target_pct")),
        requires_authentic_standard=bool(raw.get("requires_authentic_standard", False)),
        requires_chiral_method_validated=bool(
            raw.get("requires_chiral_method_validated", False)),
        min_fold_over_empty_vector=as_float(raw.get("min_fold_over_empty_vector")),
        source_template_id=source_template_id,
        raw=raw,
    )


def _digest_of(criterion: Any) -> str:
    """Digest for anything that can stand in for a criterion.

    Accepts the criterion object, the raw mapping, an
    :class:`~eagent.schemas.templates.AssayTemplate`, or a digest string, so a
    caller cannot slip past the guard merely by holding the criterion in a
    different shape.
    """
    if isinstance(criterion, PositiveCriterion):
        return criterion.digest()
    if isinstance(criterion, AssayTemplate):
        return sha256_obj(dict(criterion.positive_criteria or {}))
    if isinstance(criterion, Mapping):
        return sha256_obj(dict(criterion))
    if isinstance(criterion, str):
        return criterion
    raise TypeError(
        f"cannot read a positivity criterion from {type(criterion).__name__}; "
        f"pass a PositiveCriterion, an AssayTemplate, the positive_criteria "
        f"mapping, or its sha256 digest")


def _raw_of(criterion: Any) -> Mapping[str, Any] | None:
    if isinstance(criterion, PositiveCriterion):
        return dict(criterion.raw)
    if isinstance(criterion, AssayTemplate):
        return dict(criterion.positive_criteria or {})
    if isinstance(criterion, Mapping):
        return dict(criterion)
    return None


@dataclass(frozen=True)
class PreRegistration:
    """The hit definition and the budget, fixed before the plate was run.

    Holds the criterion *and* its digest as recorded at registration time.
    Two things protect it, because one is not enough: the criterion mapping is
    copied when it is read, so a caller still holding the original dict cannot
    change the registration by editing it, and :meth:`assert_unchanged`
    re-derives the digest from the criterion's own contents on every endpoint
    call, so an in-place edit of ``criterion.raw`` -- the easy, accidental
    version of moving the goalposts -- is stopped rather than silently
    rescored.
    """

    criterion: PositiveCriterion
    k_slots: int
    registered_at: str
    registered_by: str
    registered_digest: str
    requires_target_product: bool = True
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.k_slots < 1:
            raise ValueError(
                f"a pre-registered budget of {self.k_slots} slot(s) is not a "
                f"budget; precision@k needs the k that was planned")
        if not str(self.registered_by).strip():
            raise ValueError(
                "a pre-registration needs the actor who registered it; an "
                "unattributed bar is one nobody can be held to")
        if not str(self.registered_at).strip():
            raise ValueError(
                "a pre-registration needs the time it was registered; without "
                "it, 'before the data' cannot be checked")
        if self.criterion.digest() != self.registered_digest:
            raise CriterionChangedError(
                f"this registration was built with digest "
                f"{self.registered_digest} but its criterion hashes to "
                f"{self.criterion.digest()}; the two were never the same "
                f"criterion")

    # -- construction ------------------------------------------------------
    @classmethod
    def from_template(
        cls, template: AssayTemplate, *, k_slots: int, registered_by: str,
        registered_at: str | None = None, requires_target_product: bool = True,
        notes: Sequence[str] = (),
    ) -> "PreRegistration":
        """Register the criterion carried by a sourced assay template."""
        criterion = PositiveCriterion.from_template(template)
        return cls(
            criterion=criterion,
            k_slots=k_slots,
            # ``None`` means "now"; an explicit empty string is a caller error
            # and falls through to the validator rather than being filled in.
            registered_at=utc_now() if registered_at is None else registered_at,
            registered_by=registered_by,
            registered_digest=criterion.digest(),
            requires_target_product=requires_target_product,
            notes=tuple(notes),
        )

    @classmethod
    def from_plan(
        cls, positive_criteria: Mapping[str, Any], *, k_slots: int,
        registered_by: str, registered_at: str | None = None,
        recorded_digest: str | None = None, source_template_id: str | None = None,
        requires_target_product: bool = True, notes: Sequence[str] = (),
    ) -> "PreRegistration":
        """Register from ``experiment_plan.yaml``'s own criterion block.

        ``recorded_digest`` is the ``positive_criteria_sha256`` the plan file
        carries. When it is supplied and does not match the mapping beside it,
        the plan was edited after it was written, and that is refused here
        rather than reported as a footnote.
        """
        criterion = criterion_from_mapping(
            positive_criteria, source_template_id=source_template_id)
        digest = criterion.digest()
        if recorded_digest is not None and recorded_digest != digest:
            raise CriterionChangedError(
                f"the plan records positive_criteria_sha256 {recorded_digest} "
                f"but the criterion beside it hashes to {digest}: the "
                f"pre-registered block was edited after the plan was written")
        return cls(
            criterion=criterion,
            k_slots=k_slots,
            registered_at=utc_now() if registered_at is None else registered_at,
            registered_by=registered_by,
            registered_digest=digest,
            requires_target_product=requires_target_product,
            notes=tuple(notes),
        )

    # -- the guard ---------------------------------------------------------
    def assert_unchanged(self) -> None:
        """Re-derive the digest and refuse if the criterion has moved since."""
        current = self.criterion.digest()
        if current != self.registered_digest:
            raise CriterionChangedError(
                f"the registered criterion has been mutated since "
                f"{self.registered_at}: it was {self.registered_digest} and is "
                f"now {current}. The primary endpoint is not computable "
                f"against a bar that moved after the plate was run")

    def assert_matches(self, criterion: Any) -> None:
        """Refuse any criterion whose digest is not the registered one.

        Accepts the criterion object, the raw mapping, an assay template or a
        bare digest, so there is no shape of the same question that escapes the
        check.
        """
        self.assert_unchanged()
        other = _digest_of(criterion)
        if other == self.registered_digest:
            return
        raise CriterionChangedError(
            f"the criterion in use ({other}) is not the one registered on "
            f"{self.registered_at} by {self.registered_by} "
            f"({self.registered_digest}). {self._diff(criterion)} "
            f"Re-scoring a finished round against a different bar makes the "
            f"primary endpoint unfalsifiable; register the new bar, state that "
            f"it is exploratory, and report it beside the registered one.")

    def _diff(self, criterion: Any) -> str:
        """Key-by-key difference, so a refusal says what actually moved."""
        other = _raw_of(criterion)
        if other is None:
            return "The two criteria could not be compared key by key."
        mine = dict(self.criterion.raw)
        keys = sorted(set(mine) | set(other))
        changes = [f"{k}: registered {mine.get(k)!r} -> in use {other.get(k)!r}"
                   for k in keys if mine.get(k) != other.get(k)]
        return "Changed: " + ("; ".join(changes) if changes
                              else "nothing in the recognised keys, so the "
                                   "difference is in a key the criterion "
                                   "parser does not implement") + "."

    # -- applying it -------------------------------------------------------
    def verdict(self, row: OutcomeRow) -> HitVerdict:
        """Apply the registered definition to one row. Tri-state, never a guess.

        Order matters and is stated:

        1. A row that was never measured is **undecidable**, not a miss. A slot
           whose well was not run says nothing about the enzyme.
        2. A construct that did not express is **not a hit** -- the slot was
           spent and produced no product -- while
           :class:`HitRateReport` keeps it out of the catalysis denominator,
           which is where that distinction belongs.
        3. The target-product requirement comes next:
           :class:`~eagent.schemas.record.OutcomeClass` already ties
           ``confirmed_target_product`` to a detection method that identifies
           the product, so an indirect signal cannot reach the numerator.
        4. The pre-registered numeric bars come last.
        """
        self.assert_unchanged()
        if row.outcome in (OutcomeClass.NOT_TESTED,
                           OutcomeClass.COMPUTATIONAL_FAILURE,
                           OutcomeClass.COMPUTATIONAL_NEGATIVE):
            return HitVerdict(row.candidate_id, None, (
                f"outcome '{row.outcome.value}' is not an experimental result: "
                f"{row.outcome.claim()}",))
        if row.outcome is OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE:
            return HitVerdict(row.candidate_id, False, (
                "the construct did not express, so no product was made; the "
                "slot was spent and the enzyme's catalytic ability is "
                "undetermined",))
        if self.requires_target_product and not row.outcome.is_positive:
            return HitVerdict(row.candidate_id, False, (
                f"outcome '{row.outcome.value}': {row.outcome.claim()}",))
        met, reasons = self.criterion.evaluate(row)
        return HitVerdict(row.candidate_id, met, tuple(reasons))

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": dict(self.criterion.raw),
            "digest": self.registered_digest,
            "k_slots": self.k_slots,
            "registered_at": self.registered_at,
            "registered_by": self.registered_by,
            "requires_target_product": self.requires_target_product,
            "notes": list(self.notes),
        }


# ==========================================================================
# The primary endpoint
# ==========================================================================

@dataclass(frozen=True)
class PrecisionAtK:
    """New candidates meeting the registered definition, over the slots spent.

    Two denominators, for the same reason :class:`HitRateReport` has two: a
    round that came up short can be reported over the slots it actually spent
    (flattering) or over the budget it registered (unflattering), and both are
    defensible, so both are returned and :meth:`describe` prints both.
    """

    criterion_digest: str
    k_registered: int
    slots_spent: int
    n_hits: int
    n_undecidable: int
    n_expression_failures: int
    n_controls_excluded: int
    n_known_excluded_from_numerator: int
    verdicts: tuple[HitVerdict, ...]
    rate_over_slots_spent: Rate
    rate_over_registered_k: Rate

    @property
    def denominators_differ(self) -> bool:
        return self.slots_spent != self.k_registered

    def describe(self) -> str:
        return (f"primary endpoint (criterion {self.criterion_digest[:12]}): "
                f"{self.n_hits} new candidate(s) met the registered definition. "
                f"{self.rate_over_slots_spent.describe()} | "
                f"{self.rate_over_registered_k.describe()} | "
                f"{self.n_undecidable} slot(s) could not be decided against the "
                f"criterion, {self.n_expression_failures} did not express, "
                f"{self.n_known_excluded_from_numerator} were already-known "
                f"candidates kept in the denominator and out of the numerator.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_digest": self.criterion_digest,
            "k_registered": self.k_registered,
            "slots_spent": self.slots_spent,
            "n_hits": self.n_hits,
            "n_undecidable": self.n_undecidable,
            "n_expression_failures": self.n_expression_failures,
            "n_controls_excluded": self.n_controls_excluded,
            "n_known_excluded_from_numerator": self.n_known_excluded_from_numerator,
            "rate_over_slots_spent": self.rate_over_slots_spent.to_dict(),
            "rate_over_registered_k": self.rate_over_registered_k.to_dict(),
            "denominators_differ": self.denominators_differ,
        }


def _candidate_rows(rows: Iterable[OutcomeRow]) -> tuple[list[OutcomeRow], int]:
    """Discovery rows and the number of controls dropped.

    A control occupies a well, and sometimes a construct slot, but it is never
    a new candidate: counting one in a discovery denominator dilutes the
    endpoint and counting a positive control in the numerator invents a hit.
    """
    rows = list(rows)
    keep = [r for r in rows if not r.is_control]
    return keep, len(rows) - len(keep)


def precision_at_k(
    rows: Sequence[OutcomeRow],
    registration: PreRegistration,
    *,
    criterion_in_use: Any | None = None,
    known_candidate_ids: Iterable[str] = (),
) -> PrecisionAtK:
    """The primary endpoint: registered hits among new candidates, over slots spent.

    ``criterion_in_use`` is optional and exists so a caller that believes it
    knows the criterion has that belief checked: if it differs from the
    registered one, :class:`CriterionChangedError` is raised. Omitting it does
    not relax anything -- the registration is still re-verified and is still
    the only definition applied.

    ``known_candidate_ids`` names candidates that were already known to work
    before this round. They stay in the denominator, because their slots were
    spent, and are kept out of the numerator, because re-confirming a known
    enzyme is not a discovery. Erring this way means the endpoint can never be
    raised by stacking the batch with sure things.

    Raises
    ------
    CriterionChangedError
        If the criterion moved, in any of the shapes it can be presented in.
    ValueError
        If more constructs are scored than the registered budget. That is not a
        generous round; it means these rows are not the round that was
        registered, and silently dividing by the larger number would answer a
        question nobody asked.
    """
    registration.assert_unchanged()
    if criterion_in_use is not None:
        registration.assert_matches(criterion_in_use)

    candidates, n_controls = _candidate_rows(rows)
    slots_spent = len(candidates)
    if slots_spent > registration.k_slots:
        raise ValueError(
            f"{slots_spent} construct(s) scored against a pre-registered budget "
            f"of {registration.k_slots}; these rows are not the round that was "
            f"registered. Register the round that was actually run")

    known = {str(c) for c in known_candidate_ids}
    verdicts = tuple(registration.verdict(row) for row in candidates)
    n_hits = sum(1 for row, v in zip(candidates, verdicts)
                 if v.is_hit and row.candidate_id not in known)
    n_known_excluded = sum(1 for row, v in zip(candidates, verdicts)
                           if v.is_hit and row.candidate_id in known)
    n_undecidable = sum(1 for v in verdicts if v.undecidable)
    n_expression_failures = sum(1 for row in candidates if row.expressed is False)

    return PrecisionAtK(
        criterion_digest=registration.registered_digest,
        k_registered=registration.k_slots,
        slots_spent=slots_spent,
        n_hits=n_hits,
        n_undecidable=n_undecidable,
        n_expression_failures=n_expression_failures,
        n_controls_excluded=n_controls,
        n_known_excluded_from_numerator=n_known_excluded,
        verdicts=verdicts,
        rate_over_slots_spent=Rate(
            name="precision_at_k_over_slots_spent",
            successes=n_hits, trials=slots_spent,
            denominator_meaning=("construct slots actually spent on new "
                                 "candidates in this round, failures included"),
        ),
        rate_over_registered_k=Rate(
            name="precision_at_k_over_registered_k",
            successes=n_hits, trials=registration.k_slots,
            denominator_meaning=("the k slots the round pre-registered, so a "
                                 "batch that came up short is not flattered by "
                                 "its own shortfall"),
        ),
    )


# ==========================================================================
# Hit rate, both denominators
# ==========================================================================

@dataclass(frozen=True)
class HitRateReport:
    """Both hit-rate denominators, with intervals, so neither can be quoted alone.

    There is intentionally no attribute called ``hit_rate`` or ``rate``. The
    pattern, the field names and the reason are
    :class:`eagent.datalayer.house_db.HitRate`'s; what is added here is the
    Wilson interval on each rate and the shared hit definition, so this number
    and the primary endpoint cannot drift apart.
    """

    criterion_digest: str
    n_submitted: int
    n_expressed: int
    n_expression_failed: int
    n_expression_unknown: int
    n_not_tested: int
    n_hits: int
    n_informative: int
    n_undecidable: int

    @property
    def rate_all_submitted(self) -> Rate:
        return Rate(
            name="hit_rate_all_submitted", successes=self.n_hits,
            trials=self.n_submitted,
            denominator_meaning=("every construct put into the batch, including "
                                 "the ones that never expressed"),
        )

    @property
    def rate_expressed_only(self) -> Rate:
        return Rate(
            name="hit_rate_expressed_only", successes=self.n_hits,
            trials=self.n_expressed,
            denominator_meaning=("only the constructs that yielded soluble "
                                 "protein; undefined when none did"),
        )

    @property
    def denominators_differ(self) -> bool:
        return self.n_submitted != self.n_expressed

    def describe(self) -> str:
        return (f"{self.n_hits} hit(s) under criterion "
                f"{self.criterion_digest[:12]}; "
                f"{self.rate_all_submitted.describe()}; "
                f"{self.rate_expressed_only.describe()}; "
                f"{self.n_expression_failed} expression failure(s), "
                f"{self.n_expression_unknown} unassessed, "
                f"{self.n_not_tested} never tested, "
                f"{self.n_undecidable} undecidable against the criterion")

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_digest": self.criterion_digest,
            "n_submitted": self.n_submitted,
            "n_expressed": self.n_expressed,
            "n_expression_failed": self.n_expression_failed,
            "n_expression_unknown": self.n_expression_unknown,
            "n_not_tested": self.n_not_tested,
            "n_hits": self.n_hits,
            "n_informative": self.n_informative,
            "n_undecidable": self.n_undecidable,
            "rate_all_submitted": self.rate_all_submitted.to_dict(),
            "rate_expressed_only": self.rate_expressed_only.to_dict(),
            "denominators_differ": self.denominators_differ,
        }


def hit_rates(rows: Sequence[OutcomeRow],
              registration: PreRegistration) -> HitRateReport:
    """Both denominators for one round, under the registered hit definition.

    The hit definition is the registered one rather than a looser local test,
    so this number and :func:`precision_at_k` can never disagree about what a
    hit was -- which is the quiet way two tables in one report come to state
    different hit counts.
    """
    registration.assert_unchanged()
    candidates, _ = _candidate_rows(rows)
    verdicts = [registration.verdict(row) for row in candidates]
    return HitRateReport(
        criterion_digest=registration.registered_digest,
        n_submitted=len(candidates),
        n_expressed=sum(1 for r in candidates if r.expressed is True),
        n_expression_failed=sum(1 for r in candidates if r.expressed is False),
        n_expression_unknown=sum(1 for r in candidates if r.expressed is None),
        n_not_tested=sum(1 for r in candidates
                         if r.outcome is OutcomeClass.NOT_TESTED),
        n_hits=sum(1 for v in verdicts if v.is_hit),
        n_informative=sum(1 for r in candidates
                          if r.outcome.informs_catalytic_ability),
        n_undecidable=sum(1 for v in verdicts if v.undecidable),
    )


# ==========================================================================
# Configuration and ee
# ==========================================================================

@dataclass(frozen=True)
class ConfigurationCompliance:
    """How many products actually had the configuration the project wants.

    Separate from the hit rate because a turnover that makes the wrong
    enantiomer is a *different result* from no turnover -- often a better
    engineering starting point -- and
    :class:`~eagent.schemas.record.OutcomeClass` keeps the two apart. Rolling
    them together loses the distinction the second round depends on.
    """

    applicable: bool
    bar_pct: float | None
    bar_is_pre_registered: bool
    rate: Rate
    n_unassessable: int
    n_wrong_configuration: int
    reasons_unassessable: tuple[str, ...] = ()
    note: str = ""

    def describe(self) -> str:
        if not self.applicable:
            return f"target configuration not applicable: {self.note}"
        bar = ("the pre-registered signed-ee bar of "
               f"{self.bar_pct:g}%" if self.bar_is_pre_registered
               else "'the target configuration dominated' (no numeric ee bar "
                    "was pre-registered, so this is a direction, not a "
                    "selectivity result)")
        return (f"meeting {bar}: {self.rate.describe()}; "
                f"{self.n_wrong_configuration} product(s) favoured the opposite "
                f"configuration; {self.n_unassessable} could not be assessed")

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "bar_pct": self.bar_pct,
            "bar_is_pre_registered": self.bar_is_pre_registered,
            "rate": self.rate.to_dict(),
            "n_unassessable": self.n_unassessable,
            "n_wrong_configuration": self.n_wrong_configuration,
            "reasons_unassessable": list(self.reasons_unassessable),
            "note": self.note,
        }


def configuration_compliance(
    rows: Sequence[OutcomeRow],
    registration: PreRegistration,
    *,
    stereo_task: bool = True,
) -> ConfigurationCompliance:
    """Fraction of assessable products meeting the target-configuration requirement.

    ``stereo_task`` mirrors :attr:`eagent.schemas.reaction.TaskSpec.stereo_task`:
    reduction of an aldehyde or a symmetric ketone creates no stereocentre, and
    reporting an enantioselectivity for it is reporting a measurement that does
    not exist. The compliance object then comes back inapplicable rather than
    as 0 of 0.

    The denominator is the rows whose configuration could actually be read: a
    row with no signed ee, or one whose chiral method the registered criterion
    requires to be validated and which does not record that validation, is
    counted as unassessable and named. Treating an unmeasured configuration as
    a failure would convert an analytical gap into a chemical conclusion.
    """
    registration.assert_unchanged()
    if not stereo_task:
        return ConfigurationCompliance(
            applicable=False, bar_pct=None, bar_is_pre_registered=False,
            rate=Rate("target_configuration", 0, 0,
                      "products whose configuration could be read"),
            n_unassessable=0, n_wrong_configuration=0,
            note=("this reaction creates no new stereocentre, so there is no "
                  "target configuration to meet and no ee may be reported"),
        )

    bar = registration.criterion.min_ee_target_pct
    pre_registered = bar is not None
    effective_bar = 0.0 if bar is None else bar
    candidates, _ = _candidate_rows(rows)

    meeting = 0
    assessable = 0
    wrong = 0
    unassessable: list[str] = []
    for row in candidates:
        if registration.criterion.requires_chiral_method_validated \
                and row.chiral_method_validated is not True:
            unassessable.append(
                f"{row.candidate_id}: the registered criterion requires a "
                f"validated chiral method and none is recorded")
            continue
        if row.ee_target_pct is None:
            unassessable.append(
                f"{row.candidate_id}: no signed ee and no enantiomer peak "
                f"areas were reported")
            continue
        assessable += 1
        if row.ee_target_pct >= effective_bar:
            meeting += 1
        if row.ee_target_pct < 0:
            wrong += 1

    return ConfigurationCompliance(
        applicable=True,
        bar_pct=effective_bar,
        bar_is_pre_registered=pre_registered,
        rate=Rate("target_configuration", meeting, assessable,
                  "products whose signed ee could actually be read"),
        n_unassessable=len(unassessable),
        n_wrong_configuration=wrong,
        reasons_unassessable=tuple(unassessable),
        note=("" if pre_registered else
              "no min_ee_target_pct was pre-registered, so the bar used is "
              "'signed ee >= 0', which says the target configuration dominated "
              "and nothing about how selective the enzyme is"),
    )


@dataclass(frozen=True)
class SignedEEAggregate:
    """Signed ee across a round, pooled from areas where possible.

    There is no mean of absolute ee anywhere in this object. Averaging
    ``|ee|`` over a batch in which half the enzymes made the wrong enantiomer
    reports excellent selectivity for a round that failed its objective, and
    the sign is exactly the information :func:`~eagent.schemas.record.ee_target`
    exists to preserve.
    """

    n_rows: int
    n_with_signed_ee: int
    n_wrong_configuration: int
    pooled_ee_pct: float | None
    n_pooled_from_areas: int
    median_signed_ee_pct: float | None
    min_signed_ee_pct: float | None
    max_signed_ee_pct: float | None
    note: str = ""

    def describe(self) -> str:
        if self.n_with_signed_ee == 0:
            return "no signed ee was measurable in this round"
        pooled = ("not poolable (peak areas were not reported)"
                  if self.pooled_ee_pct is None
                  else f"{self.pooled_ee_pct:+.1f}% pooled over "
                       f"{self.n_pooled_from_areas} row(s) of peak areas")
        return (f"signed ee over {self.n_with_signed_ee} row(s): median "
                f"{self.median_signed_ee_pct:+.1f}%, range "
                f"{self.min_signed_ee_pct:+.1f}% to {self.max_signed_ee_pct:+.1f}%; "
                f"{self.n_wrong_configuration} favoured the opposite "
                f"enantiomer; {pooled}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_with_signed_ee": self.n_with_signed_ee,
            "n_wrong_configuration": self.n_wrong_configuration,
            "pooled_ee_pct": self.pooled_ee_pct,
            "n_pooled_from_areas": self.n_pooled_from_areas,
            "median_signed_ee_pct": self.median_signed_ee_pct,
            "min_signed_ee_pct": self.min_signed_ee_pct,
            "max_signed_ee_pct": self.max_signed_ee_pct,
            "note": self.note,
        }


def signed_ee_aggregate(rows: Sequence[OutcomeRow]) -> SignedEEAggregate:
    """Aggregate signed ee without inventing a scale it does not have.

    Two aggregates, both honest, neither a mean of the per-row percentages:

    * **pooled**, from summed enantiomer peak areas through
      :func:`~eagent.schemas.record.ee_target`. This is the only aggregate with
      a physical meaning: it is the ee of the combined product.
    * **median with range**, over the rows that reported a signed ee at all.
      A median because ee is a bounded ratio measured at different
      conversions, and averaging ratios across conversions compares
      quantities that are not on one scale.
    """
    values = [r.ee_target_pct for r in rows if r.ee_target_pct is not None]
    areas = [(r.target_peak_area, r.opposite_peak_area) for r in rows
             if r.target_peak_area is not None and r.opposite_peak_area is not None]
    pooled: float | None = None
    total_target = sum(a for a, _ in areas)
    total_opposite = sum(b for _, b in areas)
    if areas and total_target + total_opposite > 0:
        pooled = ee_target(total_target, total_opposite)
    return SignedEEAggregate(
        n_rows=len(rows),
        n_with_signed_ee=len(values),
        n_wrong_configuration=sum(1 for v in values if v < 0),
        pooled_ee_pct=pooled,
        n_pooled_from_areas=len(areas),
        median_signed_ee_pct=statistics.median(values) if values else None,
        min_signed_ee_pct=min(values) if values else None,
        max_signed_ee_pct=max(values) if values else None,
        note=("the pooled value is the ee of the combined product and the "
              "median is a summary of the per-row values; there is "
              "deliberately no mean of absolute ee, which would report a batch "
              "half of which made the wrong enantiomer as highly selective"),
    )


# ==========================================================================
# Round two: variant versus parent
# ==========================================================================

def _plain(value: Any) -> Any:
    return getattr(value, "value", value)


def _condition_differences(a: Mapping[str, Any],
                           b: Mapping[str, Any]) -> tuple[str, ...]:
    """Which condition fields differ, so a refusal can name them."""
    return tuple(f for f in CONDITION_KEY_FIELDS
                 if _plain((a or {}).get(f)) != _plain((b or {}).get(f)))


@dataclass(frozen=True)
class ReplicateCheck:
    """Whether a variant/parent difference is larger than the replicate scatter.

    Not a significance test, and it must not be reported as one. With three
    wells there is no test worth running; what can honestly be said is whether
    the difference is bigger than the spread the replicates themselves showed.
    ``reproducible`` is ``None`` -- not ``False`` -- when either side has fewer
    than two replicates, because "nobody repeated it" and "it did not
    reproduce" are different statements.
    """

    delta: float | None
    parent_spread: float | None
    variant_spread: float | None
    n_parent_replicates: int
    n_variant_replicates: int
    reproducible: bool | None
    basis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta": self.delta,
            "parent_spread": self.parent_spread,
            "variant_spread": self.variant_spread,
            "n_parent_replicates": self.n_parent_replicates,
            "n_variant_replicates": self.n_variant_replicates,
            "reproducible": self.reproducible,
            "basis": self.basis,
        }


def replicate_reproducibility(
    parent_replicates: Sequence[float],
    variant_replicates: Sequence[float],
    delta: float | None,
) -> ReplicateCheck:
    """Compare a delta with the observed replicate spread on both sides.

    The test is deliberately crude and deliberately conservative: the
    difference counts as separated only when its magnitude exceeds the sum of
    the two full ranges (max minus min). A tighter rule would need a variance
    model that three wells cannot support, and a looser one would call noise a
    result.
    """
    p = [float(v) for v in parent_replicates]
    v = [float(x) for x in variant_replicates]
    p_spread = (max(p) - min(p)) if len(p) >= 2 else None
    v_spread = (max(v) - min(v)) if len(v) >= 2 else None
    if delta is None or p_spread is None or v_spread is None:
        return ReplicateCheck(
            delta=delta, parent_spread=p_spread, variant_spread=v_spread,
            n_parent_replicates=len(p), n_variant_replicates=len(v),
            reproducible=None,
            basis=("fewer than two replicates on at least one side, or no "
                   "shared endpoint: not checkable, which is not the same as "
                   "not reproducible"),
        )
    separated = abs(delta) > (p_spread + v_spread)
    return ReplicateCheck(
        delta=delta, parent_spread=p_spread, variant_spread=v_spread,
        n_parent_replicates=len(p), n_variant_replicates=len(v),
        reproducible=separated,
        basis=(f"|delta| {abs(delta):g} compared with the summed replicate "
               f"ranges {p_spread:g} + {v_spread:g}; a range comparison, not a "
               f"significance test, and {len(p)} and {len(v)} replicates cannot "
               f"support one"),
    )


@dataclass(frozen=True)
class VariantParentComparison:
    """One variant against its parent under one condition set. No summary number.

    One delta per endpoint and nothing that adds them: a variant that gains
    conversion and loses ee has done two different things, and the combined
    number would hide the one that matters.
    """

    parent_candidate_id: str
    variant_candidate_id: str
    mutations: tuple[str, ...]
    condition_key: str
    measurement_type: str | None
    measurement_unit: str | None
    delta_measurement: float | None
    delta_conversion_pct: float | None
    delta_ee_pct: float | None
    parent_outcome: OutcomeClass
    variant_outcome: OutcomeClass
    replicate_check: ReplicateCheck

    def describe(self) -> str:
        parts = [f"{self.variant_candidate_id} vs {self.parent_candidate_id} "
                 f"({'/'.join(self.mutations) or 'no mutation recorded'}) under "
                 f"one condition set"]
        for label, value, unit in (
            ("measurement", self.delta_measurement, self.measurement_unit or ""),
            ("conversion", self.delta_conversion_pct, "%"),
            ("signed ee", self.delta_ee_pct, "%"),
        ):
            parts.append(f"delta {label}: "
                         + ("not comparable" if value is None
                            else f"{value:+g}{unit}"))
        parts.append("reproducible beyond replicate spread: "
                     + {True: "yes", False: "no", None: "not checkable"}[
                         self.replicate_check.reproducible])
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_candidate_id": self.parent_candidate_id,
            "variant_candidate_id": self.variant_candidate_id,
            "mutations": list(self.mutations),
            "condition_key": self.condition_key,
            "measurement_type": self.measurement_type,
            "measurement_unit": self.measurement_unit,
            "delta_measurement": self.delta_measurement,
            "delta_conversion_pct": self.delta_conversion_pct,
            "delta_ee_pct": self.delta_ee_pct,
            "parent_outcome": self.parent_outcome.value,
            "variant_outcome": self.variant_outcome.value,
            "replicate_check": self.replicate_check.to_dict(),
        }


def variant_versus_parent(
    variant: OutcomeRow,
    parent: OutcomeRow,
) -> VariantParentComparison:
    """Compare a variant with its parent, or refuse.

    Raises
    ------
    ~eagent.datalayer.house_db.ConditionMismatchError
        When the two were not assayed under identical conditions, cofactor
        identity and state included. A variant run at a different pH has not
        been shown to be better than its parent, and returning a delta with a
        caveat attached is how a condition change gets reported as an
        engineering result.
    EndpointMismatchError
        When the two report different ``measurement_type`` values. Endpoints
        are not pooled onto one numeric scale.
    """
    if variant.condition_key != parent.condition_key:
        differing = _condition_differences(parent.conditions, variant.conditions)
        extra = []
        if _plain(parent.cofactor_species) != _plain(variant.cofactor_species):
            extra.append("cofactor_species")
        if _plain(parent.cofactor_state) != _plain(variant.cofactor_state):
            extra.append("cofactor_state")
        fields = ", ".join(differing + tuple(extra)) or "an unlisted field"
        raise ConditionMismatchError(
            f"{variant.candidate_id} cannot be compared with parent "
            f"{parent.candidate_id}: they were not assayed under identical "
            f"conditions (differing: {fields}). Re-run both under one "
            f"condition set before claiming an improvement")
    if (variant.measurement_type or None) != (parent.measurement_type or None):
        raise EndpointMismatchError(
            f"{variant.candidate_id} reports endpoint "
            f"{variant.measurement_type!r} and parent {parent.candidate_id} "
            f"reports {parent.measurement_type!r}; these are different "
            f"quantities and their difference has no meaning")

    def delta(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else a - b

    delta_measurement = delta(variant.measurement_value, parent.measurement_value)
    return VariantParentComparison(
        parent_candidate_id=parent.candidate_id,
        variant_candidate_id=variant.candidate_id,
        mutations=tuple(variant.mutations),
        condition_key=variant.condition_key,
        measurement_type=variant.measurement_type,
        measurement_unit=variant.measurement_unit,
        delta_measurement=delta_measurement,
        delta_conversion_pct=delta(variant.conversion_pct, parent.conversion_pct),
        delta_ee_pct=delta(variant.ee_target_pct, parent.ee_target_pct),
        parent_outcome=parent.outcome,
        variant_outcome=variant.outcome,
        replicate_check=replicate_reproducibility(
            parent.replicate_values, variant.replicate_values, delta_measurement),
    )


@dataclass(frozen=True)
class RefusedPair:
    """A variant/parent pair that was not compared, and why.

    Refusals are returned rather than dropped, following
    :class:`eagent.datalayer.house_db.RefusedComparison`: a silently absent
    comparison reads as "no variants were made", while a listed refusal naming
    ``pH`` tells a scientist which assay to repeat.
    """

    parent_candidate_id: str
    variant_candidate_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_candidate_id": self.parent_candidate_id,
            "variant_candidate_id": self.variant_candidate_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RoundTwoReport:
    """Round-two comparisons and the pairs that could not be compared."""

    comparisons: tuple[VariantParentComparison, ...]
    refusals: tuple[RefusedPair, ...]

    @property
    def n_pairs(self) -> int:
        return len(self.comparisons) + len(self.refusals)

    @property
    def n_improved_on_measurement(self) -> int:
        """Variants whose endpoint rose *and* cleared the replicate spread.

        A raw count of positive deltas would include every pair whose
        difference is inside the noise, which in a 96-well round is most of
        them.
        """
        return sum(1 for c in self.comparisons
                   if (c.delta_measurement or 0.0) > 0
                   and c.replicate_check.reproducible is True)

    def describe(self) -> str:
        return (f"{len(self.comparisons)} comparable pair(s), "
                f"{len(self.refusals)} refused; "
                f"{self.n_improved_on_measurement} improved beyond replicate "
                f"spread on the shared endpoint")

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparisons": [c.to_dict() for c in self.comparisons],
            "refusals": [r.to_dict() for r in self.refusals],
            "n_pairs": self.n_pairs,
            "n_improved_on_measurement": self.n_improved_on_measurement,
        }


def round_two_report(
    pairs: Sequence[tuple[OutcomeRow, OutcomeRow]],
) -> RoundTwoReport:
    """Compare every ``(variant, parent)`` pair, keeping the refusals visible."""
    comparisons: list[VariantParentComparison] = []
    refusals: list[RefusedPair] = []
    for variant, parent in pairs:
        try:
            comparisons.append(variant_versus_parent(variant, parent))
        except (ConditionMismatchError, EndpointMismatchError) as exc:
            refusals.append(RefusedPair(
                parent_candidate_id=parent.candidate_id,
                variant_candidate_id=variant.candidate_id,
                reason=str(exc),
            ))
    return RoundTwoReport(tuple(comparisons), tuple(refusals))
