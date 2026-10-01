"""The project's own per-substrate database: the whole batch, failures included.

This is deliberately **not** another general enzyme sequence repository. Public
resources already hold millions of sequences and a thin layer of re-curated
activity labels. What no public resource holds, and what this project cannot
buy, is the thing that actually makes substrate-directed engineering work:

* every construct that was *submitted* in a screening round, including the ones
  that never expressed and the ones that were never measured, so a hit rate has
  an honest denominator;
* the prediction that was made **before** the experiment ran, frozen with the
  snapshot it was computed from, so "did the agent improve discovery
  efficiency" is a measurable question rather than a story told afterwards;
* mutation lineage as a first-class relation instead of a string in a notes
  column;
* the three performance axes a substitution can move, recorded apart, with no
  place anywhere in this schema to collapse them into one number.

Everything here is sqlite3 from the standard library, schema-versioned and
migratable, because a lab database that cannot be opened without a server is a
database that stops being written to.

What this module refuses to do, and why
---------------------------------------
* It will not let a prediction row be written for a round whose results have
  already arrived (:class:`FrozenPredictionError`). A prediction edited after
  the fact cannot measure anything.
* It will not delete an experiment record (:class:`DeletionRefusedError`).
  Deprecation is a flag with a reason. Deleting failures is how a screening
  campaign quietly becomes a success story.
* It will not compare a variant with its parent across differing conditions
  (:class:`ConditionMismatchError`). A variant measured at a different pH is
  not a measured improvement.
* It will not store a combined "mutation quality" score
  (:class:`CollapsedScoreError`), in any table, view or helper.
* It never invents a missing value. A missing required input is a typed
  failure; a missing optional value is ``NULL`` plus a curation note that says
  exactly what a human must supply.
"""

from __future__ import annotations

import enum
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

from ..errors import (
    EAgentError,
    FabricationGuardError,
    UnresolvedFieldError,
)
from ..provenance import canonical_json, sequence_hash, sha256_text
from ..schemas.chem import CofactorState, LigandSource, Stereochemistry
from ..schemas.record import EvidenceStrength, OutcomeClass, ReactionDirection
from ..schemas.variant import EffectDirection, PerformanceAxis

# ``lineage`` is a sibling module in this package, but it is imported
# defensively all the same: the independent-evidence count is a reporting
# convenience, and the house database must still open and record experiments on
# a checkout where that module is absent or broken.
try:  # pragma: no cover - exercised only when the sibling module is missing
    from .lineage import independent_evidence_groups as _independent_evidence_groups
except Exception:  # pragma: no cover
    _independent_evidence_groups = None  # type: ignore[assignment]


__all__ = [
    # errors
    "HouseDBError",
    "SchemaVersionError",
    "FrozenPredictionError",
    "DeletionRefusedError",
    "RecordOverwriteError",
    "ConditionMismatchError",
    "UnknownRecordError",
    "CollapsedScoreError",
    # enums / constants
    "ExpressionStatus",
    "CandidateOrigin",
    "AxisRecordKind",
    "SCHEMA_VERSION",
    "COVERAGE_FACETS",
    "CONDITION_KEY_FIELDS",
    "REQUIRED_CONDITION_FIELDS",
    "FORBIDDEN_COMBINED_SCORE_NAMES",
    "THERE_IS_NO_COMBINED_MUTATION_SCORE",
    # value objects
    "PredictionEntry",
    "RecordRow",
    "BatchOutcomes",
    "HitRate",
    "IngestReport",
    "AxisObservation",
    "AxisReadout",
    "ComparablePair",
    "RefusedComparison",
    "VariantComparisonReport",
    "PredictionOutcomeRow",
    "TopKResult",
    "PredictionOutcomeReport",
    "CoverageAudit",
    "LineageGroupSummary",
    # database
    "HouseDB",
    "condition_key",
]


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


class HouseDBError(EAgentError):
    """Base class for every refusal raised by the house database.

    Exists so a caller can distinguish "the database said no on purpose" from
    an sqlite driver fault, and never has to parse an error string to find out.
    """


class SchemaVersionError(HouseDBError):
    """The file on disk was written by a different schema generation.

    Prevents the quietest data-loss mode there is: new code opening an old file
    (or old code opening a newer one), writing through a column that means
    something else now, and leaving no trace that it happened.
    """


class FrozenPredictionError(HouseDBError):
    """A prediction was written, edited or removed after its results arrived.

    This is the error that protects the only honest measurement of whether the
    agent works. Once the experimental outcome for a round is in the database,
    the prediction for that round is history: a prediction tuned to the result
    makes top-k enrichment, rank-of-first-hit and stereochemical accuracy all
    unfalsifiable.
    """


class DeletionRefusedError(HouseDBError):
    """Someone tried to delete an experiment record through the public API.

    Negatives and failures are the expensive, non-reproducible part of a
    screening campaign and the part a model needs most. Deleting them turns a
    30 percent hit rate into a 100 percent one. Use
    :meth:`HouseDB.deprecate_record`, which keeps the row and records why it
    should no longer be trusted.
    """


class ConditionMismatchError(HouseDBError):
    """A variant-versus-parent comparison was demanded across differing conditions.

    A variant assayed at a different pH, cofactor state, substrate loading or
    endpoint has not been shown to be better than its parent. Refusing is the
    only way to stop a condition change being reported as an engineering
    result.
    """


class RecordOverwriteError(HouseDBError):
    """An ingest tried to change the result stored against an existing record id.

    The sibling of :class:`DeletionRefusedError`. Deleting a negative and
    re-ingesting it as a positive would be refused; quietly overwriting the
    same ``record_id`` achieves the same thing without a delete, so it is
    refused too. A corrected measurement is a new record, or an explicit
    deprecation of the old one with a reason.
    """


class UnknownRecordError(HouseDBError):
    """A referenced round, candidate, substrate target or record does not exist.

    Raised instead of silently creating a stub row, because a stub row created
    by a typo becomes an orphan candidate with no sequence and no provenance.
    """


class CollapsedScoreError(HouseDBError):
    """A caller tried to store a single number summarising a mutation's quality.

    Substrate fit, catalytic function and stability/expression risk move
    independently and are measured by different assays in different units. A
    weighted sum of the three hides the only thing the next round needs to
    know, which is *which* axis moved and in which direction.
    """


# --------------------------------------------------------------------------
# enums and constants
# --------------------------------------------------------------------------


#: Schema generation written by this module. Bumped by appending a migration.
SCHEMA_VERSION: int = 1


class ExpressionStatus(str, enum.Enum):
    """Whether the construct was obtained as soluble protein at all.

    Kept separate from :class:`~eagent.schemas.record.OutcomeClass` because
    "did not express" and "expressed and did nothing" are different facts about
    different things -- the first is about the construct, the second about the
    enzyme -- and only the second belongs in a catalysis denominator.
    """

    SOLUBLE = "soluble"
    INSOLUBLE = "insoluble"
    NOT_DETECTED = "not_detected"
    NOT_ASSESSED = "not_assessed"

    @property
    def expressed_solubly(self) -> bool | None:
        """``True``/``False`` when known, ``None`` when nobody looked.

        ``None`` is a real answer here. Treating "not assessed" as expressed
        inflates the catalysis denominator; treating it as failed inflates the
        hit rate. It is counted and reported on its own instead.
        """
        if self is ExpressionStatus.SOLUBLE:
            return True
        if self in (ExpressionStatus.INSOLUBLE, ExpressionStatus.NOT_DETECTED):
            return False
        return None


class CandidateOrigin(str, enum.Enum):
    """Where a construct came from.

    An engineered variant usually has no accession at all, so "has a database
    identifier" cannot be the test of whether a row is real. Recording the
    origin keeps mined and designed sequences countable apart.
    """

    NATURAL = "natural"
    ENGINEERED = "engineered"
    SYNTHETIC_DESIGN = "synthetic_design"
    UNKNOWN = "unknown"


class AxisRecordKind(str, enum.Enum):
    """Whether an axis entry is a pre-stated expectation or an observation.

    Stored on the same row shape so the two can be compared, and never merged,
    because an expectation that was written after the measurement is not an
    expectation.
    """

    EXPECTED = "expected"
    OBSERVED = "observed"


#: Condition fields that participate in the identical-conditions key. Two
#: records are comparable only if every one of these matches exactly.
CONDITION_KEY_FIELDS: tuple[str, ...] = (
    "pH",
    "temperature_C",
    "buffer",
    "solvent_system",
    "cosolvent_fraction",
    "substrate_concentration_mM",
    "enzyme_loading",
    "reaction_time_h",
    "expression_host",
)

#: Fields that must all be present for a record to count as having "full
#: reaction conditions" in :meth:`HouseDB.coverage_audit`. Narrower than
#: :data:`CONDITION_KEY_FIELDS`: ``expression_host`` and ``enzyme_loading`` are
#: frequently unrecorded in literature-derived rows and are audited separately
#: rather than silently failing every imported record.
REQUIRED_CONDITION_FIELDS: tuple[str, ...] = (
    "pH",
    "temperature_C",
    "buffer",
    "substrate_concentration_mM",
    "reaction_time_h",
)

#: The six facets whose *intersection* :meth:`HouseDB.coverage_audit` reports.
COVERAGE_FACETS: tuple[str, ...] = (
    "defined_sequence",
    "defined_substrate_structure",
    "cofactor_identity_and_state",
    "defined_product",
    "full_reaction_conditions",
    "quantitative_result",
)

#: Key names that would smuggle a collapsed quality score into a scorecard or
#: an axis payload. Checked on every write.
FORBIDDEN_COMBINED_SCORE_NAMES: frozenset[str] = frozenset({
    "aggregate_score",
    "combined",
    "combined_score",
    "composite",
    "composite_score",
    "fitness",
    "fitness_score",
    "mutation_quality",
    "mutation_quality_score",
    "overall",
    "overall_score",
    "quality_score",
    "score",
    "total",
    "total_score",
    "weighted_score",
})

#: Stated once, in one place, so a future contributor looking for the combined
#: score finds the reason it is missing instead of adding one.
THERE_IS_NO_COMBINED_MUTATION_SCORE: str = (
    "There is deliberately no column, view, property or helper in this schema "
    "that combines substrate fit, catalytic function and stability/expression "
    "risk into one number. The three are measured by different assays in "
    "different units and routinely move in opposite directions: a variant that "
    "binds better, turns over worse and expresses poorly is the normal result "
    "of round one. A weighted sum of the three erases exactly the information "
    "round two needs, and the weights would be invented rather than measured."
)


_LADDER_STRUCTURAL_KEYS: tuple[str, ...] = (
    "isomeric_smiles", "smiles", "inchi", "molfile", "molblock", "sdf",
)


def _utc_now() -> str:
    """Timestamp used for every ``*_at`` column, in UTC to the second."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or an object, without guessing a value.

    Exists so the ingest path accepts a pydantic ``ExperimentRecord``, a plain
    dict from a CSV loader and a lab LIMS row object without three code paths
    that can drift apart.
    """
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _enum_value(v: Any) -> Any:
    """Unwrap an enum to its value, leaving anything else alone."""
    return getattr(v, "value", v)


def _as_enum(cls: type[enum.Enum], v: Any, *, default: Any = None,
             field_name: str = "") -> Any:
    """Coerce to an enum member, or raise -- never fall back to a happy default.

    A silently defaulted outcome class or cofactor state is a fabricated
    scientific value, so an unrecognised string is a hard error naming the
    field and the permitted set.
    """
    if v is None:
        return default
    if isinstance(v, cls):
        return v
    try:
        return cls(_enum_value(v))
    except ValueError as exc:
        allowed = ", ".join(sorted(str(m.value) for m in cls))  # type: ignore[attr-defined]
        raise ValueError(
            f"{field_name or cls.__name__}: {v!r} is not one of [{allowed}]"
        ) from exc


def _as_bool(v: Any) -> bool | None:
    """Tri-state coercion: ``None`` stays ``None`` rather than becoming ``False``.

    ``False`` means "measured and absent"; ``None`` means "nobody looked".
    Collapsing them is how an unmeasured control becomes a reported negative.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "yes", "1", "y"):
        return True
    if s in ("false", "no", "0", "n"):
        return False
    return None


def _as_float(v: Any, field_name: str = "") -> float | None:
    """Coerce to float, or raise. Never silently drops an unparseable number."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name or 'value'}: {v!r} is not a number") from exc


def _dumps(obj: Any) -> str:
    """Canonical JSON for a stored blob, so two equal payloads hash equal."""
    return canonical_json(obj if obj is not None else None)


def _loads(text: str | None, default: Any = None) -> Any:
    """Read a stored JSON blob, returning ``default`` for NULL or junk."""
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return default


def _reject_combined_score(payload: Any, where: str) -> None:
    """Refuse a mapping that carries a collapsed quality score.

    Called on every scorecard and axis payload written. Prevents the one change
    that would quietly undo the three-axis design: a caller adding
    ``{"overall": 0.82}`` to the scorecard blob and every downstream report
    starting to sort on it.
    """
    if isinstance(payload, Mapping):
        for key in payload:
            if str(key).strip().lower() in FORBIDDEN_COMBINED_SCORE_NAMES:
                raise CollapsedScoreError(
                    f"{where}: key {key!r} is a collapsed quality score. "
                    f"{THERE_IS_NO_COMBINED_MUTATION_SCORE}"
                )
            _reject_combined_score(payload[key], where)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            _reject_combined_score(item, where)


def _ladder_dict(ladder: Any) -> dict[str, Any] | None:
    """Normalise a chemical identity ladder to plain data for storage.

    Accepts a :class:`~eagent.datalayer.identity.ChemicalIdentityLadder`, a
    mapping, or ``None``. Stored as data rather than as a pickled object so the
    file stays readable by ``sqlite3`` on a machine without this package.
    """
    if ladder is None:
        return None
    if isinstance(ladder, Mapping):
        return dict(ladder)
    as_dict = getattr(ladder, "as_dict", None)
    if callable(as_dict):
        out = as_dict()
        return dict(out) if isinstance(out, Mapping) else {"value": out}
    raise TypeError(
        "ladder must be a ChemicalIdentityLadder, a mapping, or None; "
        f"got {type(ladder).__name__}"
    )


def _ladder_has_structure(ladder: Mapping[str, Any] | None) -> bool:
    """Whether a stored ladder carries a rung a modelling tool could read.

    A name and a CAS number do not define a molecule. This is the test used by
    the coverage audit, and it looks for an actual structural representation
    anywhere in the serialised ladder rather than trusting a ``resolved`` flag.
    """
    if not ladder:
        return False
    found = False

    def walk(node: Any) -> None:
        nonlocal found
        if found:
            return
        if isinstance(node, Mapping):
            for k, v in node.items():
                kl = str(k).strip().lower()
                if kl in _LADDER_STRUCTURAL_KEYS and isinstance(v, str) and v.strip():
                    found = True
                    return
                if kl in ("rung", "representation") and isinstance(v, str) \
                        and v.strip().lower() in _LADDER_STRUCTURAL_KEYS:
                    sibling = node.get("value")
                    if isinstance(sibling, str) and sibling.strip():
                        found = True
                        return
                walk(v)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(ladder)
    return found


def condition_key(conditions: Mapping[str, Any] | None,
                  cofactor_species: str | None,
                  cofactor_state: str | None) -> str:
    """Stable hash of everything that must match for two assays to be comparable.

    The cofactor identity and state are part of the key, not metadata beside
    it: the same enzyme with NADH and with NADPH is two experiments, and a
    comparison that pools them is measuring cofactor preference while claiming
    to measure a mutation.
    """
    payload = {
        "conditions": {
            f: _enum_value((conditions or {}).get(f)) for f in CONDITION_KEY_FIELDS
        },
        "cofactor_species": cofactor_species,
        "cofactor_state": cofactor_state,
    }
    return "sha256:" + sha256_text(canonical_json(payload))


def _condition_differences(a: Mapping[str, Any] | None,
                           b: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Which key condition fields differ, so a refusal can name them."""
    a = a or {}
    b = b or {}
    out = [f for f in CONDITION_KEY_FIELDS
           if _enum_value(a.get(f)) != _enum_value(b.get(f))]
    return tuple(out)


# --------------------------------------------------------------------------
# value objects returned by the queries
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionEntry:
    """One pre-experiment prediction, as handed to :meth:`HouseDB.freeze_predictions`.

    A dataclass rather than a loose dict so the fields that make a prediction
    auditable -- the rank, the selection role, the reason the slot was spent
    and the stated uncertainty -- cannot be omitted by accident and backfilled
    later from memory.
    """

    sequence_sha256: str
    predicted_rank: int
    selection_role: str
    selection_reason: str
    uncertainty: str
    scorecard: Mapping[str, Any] = field(default_factory=dict)
    stereo_call: str = "insufficient_evidence"
    stereo_basis: str = ""
    predicted_ee_pct: float | None = None
    calibration_source: str | None = None
    robustness_G: float | None = None
    confidence: str | None = None
    prediction_id: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.predicted_ee_pct is not None and not self.calibration_source:
            raise FabricationGuardError(
                f"{self.sequence_sha256}: a numeric predicted ee requires a named "
                f"calibration source; leave it None and keep the directional "
                f"stereo call instead"
            )
        if not str(self.selection_reason).strip():
            raise UnresolvedFieldError(
                ["selection_reason"], gate="freeze_predictions"
            )
        if not str(self.uncertainty).strip():
            raise UnresolvedFieldError(["uncertainty"], gate="freeze_predictions")
        _reject_combined_score(dict(self.scorecard), "prediction.scorecard")


@dataclass(frozen=True)
class RecordRow:
    """One experiment record as it is stored, including the ones that failed.

    Returned by every batch query so that a caller reading the results sees the
    expression failures and the never-measured constructs in the same list as
    the hits, rather than having to ask a second question to find them.
    """

    record_id: str
    round_id: str
    substrate_target_id: str
    sequence_sha256: str
    outcome: OutcomeClass
    expression_status: ExpressionStatus
    soluble_expression: bool | None
    reaction_direction: ReactionDirection
    cofactor_species: str | None
    cofactor_state: CofactorState
    cofactor_source: LigandSource
    detection_method: str | None
    limit_of_detection: float | None
    limit_unit: str | None
    confirms_product_identity: bool
    chiral_method_validated: bool | None
    authentic_standard: bool | None
    product_smiles: str | None
    product_inchikey: str | None
    ee_target_pct: float | None
    conversion_pct: float | None
    measurement_type: str | None
    measurement_value: float | None
    measurement_unit: str | None
    specific_activity: float | None
    specific_activity_unit: str | None
    kcat_s: float | None
    km_mM: float | None
    replicates: int | None
    reaction_id: str | None
    construct_sequence: str | None
    construct_description: str | None
    conditions: Mapping[str, Any]
    condition_key: str
    deprecated: bool
    deprecation_reason: str | None
    needs_curation: bool
    curation_notes: tuple[str, ...]
    notes: str
    created_at: str

    @property
    def expressed(self) -> bool | None:
        """Tri-state: did this construct yield soluble protein?

        ``None`` is kept rather than resolved, because the only honest answer
        for an unassessed construct is that nobody knows, and the hit-rate
        object reports those separately instead of folding them into either
        denominator.
        """
        if self.outcome is OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE:
            return False
        explicit = self.expression_status.expressed_solubly
        if explicit is not None:
            return explicit
        if self.soluble_expression is not None:
            return self.soluble_expression
        if self.outcome.informs_catalytic_ability:
            # A turnover measurement is only possible on protein that existed.
            return True
        return None

    @property
    def is_hit(self) -> bool:
        """A hit is a confirmed target product, nothing looser."""
        return self.outcome is OutcomeClass.CONFIRMED_TARGET_PRODUCT

    @property
    def informs_catalysis(self) -> bool:
        """Whether this row says anything about the enzyme's catalytic ability."""
        return self.outcome.informs_catalytic_ability

    def quantitative_value(self) -> tuple[str, float] | None:
        """The one quantitative endpoint this row carries, with its name.

        Returns ``None`` rather than zero when nothing was quantified. Zero and
        "not quantified" are different numbers and only one of them can be
        averaged.
        """
        for name in ("measurement_value", "conversion_pct", "specific_activity",
                     "kcat_s", "ee_target_pct"):
            v = getattr(self, name)
            if v is not None:
                return (name, float(v))
        return None


@dataclass(frozen=True)
class BatchOutcomes:
    """Every construct submitted in a round, failures and untested included.

    The point of this type is that there is no filtered variant of it. A caller
    who wants only the hits must drop rows themselves, in code a reviewer can
    see, instead of calling a convenience method that hides them.
    """

    round_id: str
    substrate_target_id: str | None
    rows: tuple[RecordRow, ...]
    predictions_frozen_at: str | None
    results_ingested_at: str | None

    @property
    def n_submitted(self) -> int:
        """All constructs in the batch, including failures and untested rows."""
        return len(self.rows)

    def by_outcome(self) -> dict[str, int]:
        """Count per outcome class, with every class the enum defines present.

        Zero-valued classes are kept so a report shows ``expression failures:
        0`` rather than omitting the line, which reads as "not measured".
        """
        counts = {o.value: 0 for o in OutcomeClass}
        for r in self.rows:
            counts[r.outcome.value] += 1
        return counts

    @property
    def n_expression_failures(self) -> int:
        return sum(1 for r in self.rows if r.expressed is False)

    @property
    def n_not_tested(self) -> int:
        return sum(1 for r in self.rows if r.outcome is OutcomeClass.NOT_TESTED)

    @property
    def n_deprecated(self) -> int:
        """Rows flagged untrustworthy. They are still here; that is the design."""
        return sum(1 for r in self.rows if r.deprecated)

    def failures(self) -> tuple[RecordRow, ...]:
        """The rows a published table would usually drop."""
        return tuple(
            r for r in self.rows
            if r.outcome in (OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE,
                             OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
                             OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION,
                             OutcomeClass.NOT_TESTED,
                             OutcomeClass.COMPUTATIONAL_FAILURE)
        )


@dataclass(frozen=True)
class HitRate:
    """Both hit-rate denominators in one object, so neither can be quoted alone.

    There is intentionally no attribute called ``hit_rate``. Asking for "the"
    hit rate of a screening round is the question that produces the number in
    the abstract: 3 hits out of 8 expressed constructs is 37 percent, and 3 out
    of 24 submitted is 12 percent, and both are true. Returning them together,
    with the unassessed constructs counted on the side, makes the flattering
    one impossible to report without the other.
    """

    round_id: str
    n_submitted: int
    n_expressed: int
    n_expression_failed: int
    n_expression_unknown: int
    n_not_tested: int
    n_hits: int
    n_informative: int

    @property
    def rate_all_submitted(self) -> float | None:
        """Hits divided by every construct put into the batch."""
        if self.n_submitted <= 0:
            return None
        return self.n_hits / self.n_submitted

    @property
    def rate_expressed_only(self) -> float | None:
        """Hits divided by the constructs that actually yielded soluble protein.

        ``None`` when nothing expressed: a rate with a zero denominator is not
        0.0 and not 1.0, it is undefined, and reporting it as a number is how a
        failed round becomes a plotted point.
        """
        if self.n_expressed <= 0:
            return None
        return self.n_hits / self.n_expressed

    def as_dict(self) -> dict[str, Any]:
        """Both rates plus every count needed to recompute them by hand."""
        return {
            "round_id": self.round_id,
            "n_submitted": self.n_submitted,
            "n_expressed": self.n_expressed,
            "n_expression_failed": self.n_expression_failed,
            "n_expression_unknown": self.n_expression_unknown,
            "n_not_tested": self.n_not_tested,
            "n_hits": self.n_hits,
            "n_informative": self.n_informative,
            "rate_all_submitted": self.rate_all_submitted,
            "rate_expressed_only": self.rate_expressed_only,
            "denominators_differ": self.n_submitted != self.n_expressed,
        }

    def describe(self) -> str:
        """One line that always carries both denominators."""
        def pct(v: float | None) -> str:
            return "undefined" if v is None else f"{v * 100:.1f}%"
        return (
            f"{self.round_id}: {self.n_hits} hits; "
            f"{pct(self.rate_all_submitted)} of {self.n_submitted} submitted, "
            f"{pct(self.rate_expressed_only)} of {self.n_expressed} expressed "
            f"({self.n_expression_failed} expression failures, "
            f"{self.n_expression_unknown} expression unassessed, "
            f"{self.n_not_tested} never tested)"
        )


@dataclass(frozen=True)
class IngestReport:
    """What one ingest actually wrote, and what it noticed was missing.

    Returned rather than logged so a harness step can fail loudly on
    ``curation_notes`` instead of discovering six months later that half the
    batch has no detection limit.
    """

    round_id: str
    n_written: int
    by_outcome: Mapping[str, int]
    n_predicted_not_submitted: int
    n_submitted_without_prediction: int
    curation_notes: tuple[str, ...]
    results_ingested_at: str

    @property
    def needs_curation(self) -> bool:
        return bool(self.curation_notes)


@dataclass(frozen=True)
class AxisObservation:
    """One performance axis for one construct, recorded on its own.

    Three of these describe a variant. There is no fourth row type that sums
    them. See :data:`THERE_IS_NO_COMBINED_MUTATION_SCORE`.
    """

    sequence_sha256: str
    round_id: str | None
    axis: PerformanceAxis
    kind: AxisRecordKind
    direction: EffectDirection
    value: float | None
    unit: str | None
    basis: str
    evidence: tuple[str, ...]
    recorded_at: str


@dataclass(frozen=True)
class AxisReadout:
    """The three axes for one construct, side by side and never summed.

    Deliberately exposes only per-axis access. A reviewer looking for a
    ``total`` here should read the class docstring and
    :data:`THERE_IS_NO_COMBINED_MUTATION_SCORE`, which explains why adding one
    would destroy the second round's information.
    """

    sequence_sha256: str
    observations: tuple[AxisObservation, ...]

    def by_axis(self, kind: AxisRecordKind | None = None) \
            -> dict[str, list[AxisObservation]]:
        """Observations grouped by axis, with all three axes always present."""
        out: dict[str, list[AxisObservation]] = {a.value: [] for a in PerformanceAxis}
        for o in self.observations:
            if kind is not None and o.kind is not kind:
                continue
            out[o.axis.value].append(o)
        return out

    def directions(self, kind: AxisRecordKind = AxisRecordKind.OBSERVED) \
            -> dict[str, str]:
        """Three directions, one per axis. Never a total, never a rank."""
        out = {a.value: EffectDirection.UNKNOWN.value for a in PerformanceAxis}
        for o in self.observations:
            if o.kind is kind:
                out[o.axis.value] = o.direction.value
        return out

    @property
    def axes_covered(self) -> int:
        """How many of the three axes have any entry at all."""
        return len({o.axis for o in self.observations})


@dataclass(frozen=True)
class ComparablePair:
    """A variant and its parent measured under genuinely identical conditions.

    Carries one delta per endpoint and no summary number, for the same reason
    the axes are kept apart: a variant that gains conversion and loses ee has
    not simply "improved".
    """

    parent_sha256: str
    variant_sha256: str
    mutations: tuple[str, ...]
    condition_key: str
    substrate_target_id: str
    measurement_type: str | None
    measurement_unit: str | None
    parent_record_id: str
    variant_record_id: str
    parent_value: float | None
    variant_value: float | None
    parent_ee_pct: float | None
    variant_ee_pct: float | None
    parent_conversion_pct: float | None
    variant_conversion_pct: float | None
    parent_outcome: OutcomeClass
    variant_outcome: OutcomeClass

    @property
    def delta_measurement(self) -> float | None:
        """Variant minus parent on the shared endpoint, or ``None``."""
        if self.parent_value is None or self.variant_value is None:
            return None
        return self.variant_value - self.parent_value

    @property
    def delta_ee_pct(self) -> float | None:
        """Change in signed ee toward the target. Negative means worse."""
        if self.parent_ee_pct is None or self.variant_ee_pct is None:
            return None
        return self.variant_ee_pct - self.parent_ee_pct

    @property
    def delta_conversion_pct(self) -> float | None:
        if self.parent_conversion_pct is None or self.variant_conversion_pct is None:
            return None
        return self.variant_conversion_pct - self.parent_conversion_pct

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_sha256": self.parent_sha256,
            "variant_sha256": self.variant_sha256,
            "mutations": list(self.mutations),
            "condition_key": self.condition_key,
            "substrate_target_id": self.substrate_target_id,
            "measurement_type": self.measurement_type,
            "measurement_unit": self.measurement_unit,
            "parent_record_id": self.parent_record_id,
            "variant_record_id": self.variant_record_id,
            "delta_measurement": self.delta_measurement,
            "delta_ee_pct": self.delta_ee_pct,
            "delta_conversion_pct": self.delta_conversion_pct,
            "parent_outcome": self.parent_outcome.value,
            "variant_outcome": self.variant_outcome.value,
        }


@dataclass(frozen=True)
class RefusedComparison:
    """A variant/parent pair the database declined to compare, and why.

    Refusals are returned rather than dropped. A silently absent comparison
    looks like "no variants were made"; a listed refusal naming ``pH`` and
    ``cofactor state`` tells a scientist exactly which assay to repeat.
    """

    parent_sha256: str
    variant_sha256: str
    mutations: tuple[str, ...]
    reason: str
    differing_fields: tuple[str, ...]
    parent_record_ids: tuple[str, ...]
    variant_record_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_sha256": self.parent_sha256,
            "variant_sha256": self.variant_sha256,
            "mutations": list(self.mutations),
            "reason": self.reason,
            "differing_fields": list(self.differing_fields),
            "parent_record_ids": list(self.parent_record_ids),
            "variant_record_ids": list(self.variant_record_ids),
        }


@dataclass(frozen=True)
class VariantComparisonReport:
    """Everything known about one parent's variants: comparisons and refusals.

    The refusals are first-class because the common real outcome of a second
    round is that the variants were run in a different plate format, and a
    report that shows only the comparisons it managed to make would present
    that as a clean result set.
    """

    parent_sha256: str
    comparisons: tuple[ComparablePair, ...]
    refusals: tuple[RefusedComparison, ...]

    @property
    def n_variants(self) -> int:
        return len({c.variant_sha256 for c in self.comparisons}
                   | {r.variant_sha256 for r in self.refusals})

    def require(self, variant_sha256: str) -> ComparablePair:
        """The comparison for one variant, or a typed refusal explaining why not.

        Use this when a decision depends on the comparison: it raises rather
        than returning ``None``, so a missing comparison cannot be read as "no
        difference".
        """
        for c in self.comparisons:
            if c.variant_sha256 == variant_sha256:
                return c
        for r in self.refusals:
            if r.variant_sha256 == variant_sha256:
                fields = ", ".join(r.differing_fields) or "n/a"
                raise ConditionMismatchError(
                    f"{variant_sha256} cannot be compared with parent "
                    f"{self.parent_sha256}: {r.reason} (differing: {fields}). "
                    f"Re-run both under one condition set before claiming an "
                    f"improvement."
                )
        raise UnknownRecordError(
            f"{variant_sha256} is not a recorded variant of {self.parent_sha256}"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_sha256": self.parent_sha256,
            "comparisons": [c.as_dict() for c in self.comparisons],
            "refusals": [r.as_dict() for r in self.refusals],
            "n_variants": self.n_variants,
        }


@dataclass(frozen=True)
class PredictionOutcomeRow:
    """One frozen prediction paired with what actually happened to it."""

    sequence_sha256: str
    predicted_rank: int | None
    selection_role: str | None
    stereo_call: str | None
    robustness_G: float | None
    uncertainty: str | None
    record_id: str | None
    outcome: OutcomeClass | None
    ee_target_pct: float | None
    informs_catalysis: bool
    had_prediction: bool
    was_submitted: bool

    @property
    def is_hit(self) -> bool:
        return self.outcome is OutcomeClass.CONFIRMED_TARGET_PRODUCT

    @property
    def stereo_agreement(self) -> str:
        """``agree`` / ``disagree`` / ``undetermined`` for the stereo call.

        ``undetermined`` is returned whenever the call was non-directional or
        no ee was measured; scoring those as agreement would make the
        stereochemical accuracy rise simply by predicting nothing.
        """
        call = (self.stereo_call or "").strip().lower()
        if self.ee_target_pct is None or call not in ("favors_target", "favors_opposite"):
            return "undetermined"
        if self.ee_target_pct == 0:
            return "undetermined"
        predicted_target = call == "favors_target"
        observed_target = self.ee_target_pct > 0
        return "agree" if predicted_target == observed_target else "disagree"


@dataclass(frozen=True)
class TopKResult:
    """Hits inside the agent's own top-k, with the denominator it really had."""

    k_requested: int
    n_considered: int
    n_hits: int

    @property
    def rate(self) -> float | None:
        """``None`` when nothing informative was in the top k."""
        if self.n_considered <= 0:
            return None
        return self.n_hits / self.n_considered

    @property
    def short_of_k(self) -> bool:
        """True when fewer than ``k`` informative rows existed to score."""
        return self.n_considered < self.k_requested


@dataclass(frozen=True)
class PredictionOutcomeReport:
    """Did the pre-registered ranking find hits faster than the batch average?

    This is the measurement the whole database exists to make possible, and it
    is only meaningful if the predictions were frozen first. When they were
    not, :attr:`interpretable` is ``False`` and carries the reason, instead of
    the report returning a number that looks like evidence of a working agent.

    Rows that cannot inform catalysis -- expression failures, untested
    constructs, computational failures -- are excluded from the rates and
    counted on the side. A top-ranked candidate that never expressed is a
    cloning result, not a wrong prediction about chemistry.
    """

    round_id: str
    rows: tuple[PredictionOutcomeRow, ...]
    predictions_frozen_at: str | None
    results_ingested_at: str | None
    interpretable: bool
    non_interpretable_reason: str | None
    n_predicted: int
    n_submitted: int
    n_informative: int
    n_hits: int
    n_excluded_uninformative: int
    n_submitted_without_prediction: int
    n_predicted_not_submitted: int

    @property
    def frozen_before_results(self) -> bool:
        """Whether the ranking existed before anyone saw a result."""
        if not self.predictions_frozen_at or not self.results_ingested_at:
            return False
        return self.predictions_frozen_at <= self.results_ingested_at

    @property
    def baseline_rate(self) -> float | None:
        """The round's own hit rate over informative rows: the chance level."""
        if self.n_informative <= 0:
            return None
        return self.n_hits / self.n_informative

    def top_k(self, k: int) -> TopKResult:
        """Hits among the k best-ranked informative, submitted constructs."""
        if k <= 0:
            raise ValueError("k must be positive")
        ranked = sorted(
            (r for r in self.rows
             if r.had_prediction and r.was_submitted and r.informs_catalysis
             and r.predicted_rank is not None),
            key=lambda r: (r.predicted_rank, r.sequence_sha256),
        )[:k]
        return TopKResult(k_requested=k, n_considered=len(ranked),
                          n_hits=sum(1 for r in ranked if r.is_hit))

    def enrichment_vs_baseline(self, k: int) -> float | None:
        """Top-k rate divided by the round's own hit rate, or ``None``.

        ``None`` whenever either side is undefined -- notably when the round
        produced no hits at all, where any ratio would be invented.
        """
        top = self.top_k(k).rate
        base = self.baseline_rate
        if top is None or base in (None, 0):
            return None
        return top / float(base)  # type: ignore[arg-type]

    @property
    def rank_of_first_hit(self) -> int | None:
        """Predicted rank of the first construct that actually worked."""
        ranked = sorted(
            (r for r in self.rows
             if r.had_prediction and r.was_submitted and r.predicted_rank is not None),
            key=lambda r: (r.predicted_rank, r.sequence_sha256),
        )
        for r in ranked:
            if r.is_hit:
                return r.predicted_rank
        return None

    def stereo_agreement_counts(self) -> dict[str, int]:
        """Agreement of the frozen stereochemical call with the measured ee."""
        out = {"agree": 0, "disagree": 0, "undetermined": 0}
        for r in self.rows:
            if r.had_prediction and r.was_submitted:
                out[r.stereo_agreement] += 1
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "interpretable": self.interpretable,
            "non_interpretable_reason": self.non_interpretable_reason,
            "frozen_before_results": self.frozen_before_results,
            "predictions_frozen_at": self.predictions_frozen_at,
            "results_ingested_at": self.results_ingested_at,
            "n_predicted": self.n_predicted,
            "n_submitted": self.n_submitted,
            "n_informative": self.n_informative,
            "n_hits": self.n_hits,
            "n_excluded_uninformative": self.n_excluded_uninformative,
            "n_submitted_without_prediction": self.n_submitted_without_prediction,
            "n_predicted_not_submitted": self.n_predicted_not_submitted,
            "baseline_rate": self.baseline_rate,
            "rank_of_first_hit": self.rank_of_first_hit,
            "stereo_agreement": self.stereo_agreement_counts(),
        }


@dataclass(frozen=True)
class CoverageAudit:
    """How many records are complete on *all six* facets at once.

    The headline row count of an enzyme database answers a question nobody
    needs answered. The question that decides whether a model can be trained,
    or a result reproduced, is how many rows simultaneously have a defined
    sequence, a defined substrate structure, a cofactor identity *and* state, a
    defined product, full reaction conditions and a quantitative result.

    That is an intersection, and it is always far smaller than any individual
    facet count. This type reports the intersection first and keeps the union
    beside it only so the gap is visible.
    """

    substrate_target_id: str
    n_records: int
    facet_counts: Mapping[str, int]
    n_complete: int
    n_union: int
    missing_counts: Mapping[str, int]
    complete_record_ids: tuple[str, ...]

    @property
    def completeness_fraction(self) -> float | None:
        """Complete rows over all rows, or ``None`` when there are no rows."""
        if self.n_records <= 0:
            return None
        return self.n_complete / self.n_records

    @property
    def worst_facet(self) -> str | None:
        """The facet that disqualifies the most records: where to spend effort."""
        if not self.missing_counts:
            return None
        return max(self.missing_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "substrate_target_id": self.substrate_target_id,
            "n_records": self.n_records,
            "facet_counts": dict(self.facet_counts),
            "n_complete_intersection": self.n_complete,
            "n_any_facet_union": self.n_union,
            "missing_counts": dict(self.missing_counts),
            "completeness_fraction": self.completeness_fraction,
            "worst_facet": self.worst_facet,
            "note": (
                "n_complete_intersection is the number of records satisfying all "
                "six facets simultaneously. Quoting a single facet count, or the "
                "union, overstates usable coverage."
            ),
        }

    def describe(self) -> str:
        lines = [
            f"coverage audit for {self.substrate_target_id}: "
            f"{self.n_complete}/{self.n_records} records complete on all "
            f"{len(COVERAGE_FACETS)} facets (union touching any facet: {self.n_union})"
        ]
        for facet in COVERAGE_FACETS:
            lines.append(
                f"  {facet}: present {self.facet_counts.get(facet, 0)}, "
                f"missing {self.missing_counts.get(facet, 0)}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class LineageGroupSummary:
    """How many *independent* measurements back a set of records.

    Four rows re-curated from one paper are one piece of evidence. This wraps
    :mod:`eagent.datalayer.lineage` so that count is a query against the house
    database rather than something a reader has to work out by eye.
    """

    n_rows: int
    n_independent: int | None
    groups: tuple[Mapping[str, Any], ...]
    available: bool
    unavailable_reason: str | None = None

    @property
    def n_discounted(self) -> int | None:
        """Rows that are copies and must not be counted a second time."""
        if self.n_independent is None:
            return None
        return self.n_rows - self.n_independent


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

#: Migration 1: the initial schema.
#:
#: Outcome classes, reaction directions and cofactor states are validated in
#: Python against the enums in :mod:`eagent.schemas` rather than frozen into
#: SQL ``CHECK`` constraints, because the enums are the single authority and a
#: duplicated list in DDL would silently diverge on an existing file. The three
#: performance axes are the exception: they are a fixed architectural
#: commitment, so the ``CHECK`` is written out and a fourth "combined" axis
#: cannot be inserted even with raw SQL.
_MIGRATION_1: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version      INTEGER PRIMARY KEY,
        applied_at   TEXT NOT NULL,
        description  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS substrate_target (
        substrate_target_id       TEXT PRIMARY KEY,
        label                     TEXT NOT NULL DEFAULT '',
        substrate_ladder_json     TEXT,
        product_ladder_json       TEXT,
        target_stereochemistry    TEXT NOT NULL DEFAULT 'unspecified',
        creates_new_stereocenter  INTEGER,
        reaction_id               TEXT,
        reaction_class            TEXT,
        required_cofactor_species TEXT,
        required_cofactor_state   TEXT,
        needs_curation            INTEGER NOT NULL DEFAULT 0,
        curation_notes            TEXT NOT NULL DEFAULT '[]',
        created_at                TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS candidate (
        sequence_sha256     TEXT PRIMARY KEY,
        candidate_id        TEXT,
        sequence            TEXT,
        construct_sequence  TEXT,
        construct_sha256    TEXT,
        origin              TEXT NOT NULL DEFAULT 'unknown',
        family              TEXT,
        family_basis        TEXT NOT NULL DEFAULT '',
        cluster_id          TEXT,
        organism            TEXT,
        discovery_method    TEXT,
        discovery_detail    TEXT NOT NULL DEFAULT '',
        provenance_json     TEXT,
        needs_curation      INTEGER NOT NULL DEFAULT 0,
        curation_notes      TEXT NOT NULL DEFAULT '[]',
        deprecated          INTEGER NOT NULL DEFAULT 0,
        deprecation_reason  TEXT,
        created_at          TEXT NOT NULL,
        updated_at          TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS candidate_accession (
        accession_row_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        sequence_sha256   TEXT NOT NULL REFERENCES candidate(sequence_sha256),
        database          TEXT NOT NULL,
        accession         TEXT NOT NULL,
        version           TEXT,
        retrieved_at      TEXT,
        UNIQUE (sequence_sha256, database, accession, version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lineage (
        edge_id              INTEGER PRIMARY KEY AUTOINCREMENT,
        parent_sha256        TEXT NOT NULL REFERENCES candidate(sequence_sha256),
        variant_sha256       TEXT NOT NULL REFERENCES candidate(sequence_sha256),
        mutations_json       TEXT NOT NULL,
        numbering_reference  TEXT NOT NULL,
        generator            TEXT,
        round_id             TEXT,
        notes                TEXT NOT NULL DEFAULT '',
        created_at           TEXT NOT NULL,
        UNIQUE (parent_sha256, variant_sha256, mutations_json)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS experiment_round (
        round_id                 TEXT PRIMARY KEY,
        substrate_target_id      TEXT NOT NULL
                                 REFERENCES substrate_target(substrate_target_id),
        round_number             INTEGER NOT NULL DEFAULT 1,
        plan_id                  TEXT,
        predictions_frozen_at    TEXT,
        predictions_snapshot_id  TEXT,
        results_ingested_at      TEXT,
        notes                    TEXT NOT NULL DEFAULT '',
        created_at               TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prediction (
        prediction_id        TEXT PRIMARY KEY,
        round_id             TEXT NOT NULL REFERENCES experiment_round(round_id),
        substrate_target_id  TEXT NOT NULL,
        sequence_sha256      TEXT NOT NULL REFERENCES candidate(sequence_sha256),
        predicted_rank       INTEGER NOT NULL,
        scorecard_json       TEXT NOT NULL DEFAULT '{}',
        stereo_call          TEXT NOT NULL DEFAULT 'insufficient_evidence',
        stereo_basis         TEXT NOT NULL DEFAULT '',
        predicted_ee_pct     REAL,
        calibration_source   TEXT,
        robustness_G         REAL,
        selection_role       TEXT NOT NULL,
        selection_reason     TEXT NOT NULL,
        uncertainty          TEXT NOT NULL,
        confidence           TEXT,
        snapshot_id          TEXT NOT NULL,
        frozen_at            TEXT NOT NULL,
        content_sha256       TEXT NOT NULL,
        notes                TEXT NOT NULL DEFAULT '',
        UNIQUE (round_id, sequence_sha256),
        UNIQUE (round_id, predicted_rank)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prediction_freeze_log (
        log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        round_id     TEXT NOT NULL REFERENCES experiment_round(round_id),
        snapshot_id  TEXT NOT NULL,
        n_rows       INTEGER NOT NULL,
        n_replaced   INTEGER NOT NULL DEFAULT 0,
        reason       TEXT NOT NULL DEFAULT '',
        frozen_at    TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS experiment_record (
        record_id                  TEXT PRIMARY KEY,
        round_id                   TEXT NOT NULL REFERENCES experiment_round(round_id),
        substrate_target_id        TEXT NOT NULL,
        sequence_sha256            TEXT NOT NULL REFERENCES candidate(sequence_sha256),
        construct_sequence         TEXT,
        construct_description      TEXT,
        outcome                    TEXT NOT NULL,
        expression_status          TEXT NOT NULL DEFAULT 'not_assessed',
        soluble_expression         INTEGER,
        detection_method           TEXT,
        limit_of_detection         REAL,
        limit_unit                 TEXT,
        authentic_standard         INTEGER,
        confirms_product_identity  INTEGER NOT NULL DEFAULT 0,
        chiral_method_validated    INTEGER,
        product_smiles             TEXT,
        product_inchikey           TEXT,
        ee_target_pct              REAL,
        conversion_pct             REAL,
        measurement_type           TEXT,
        measurement_value          REAL,
        measurement_unit           TEXT,
        specific_activity          REAL,
        specific_activity_unit     TEXT,
        kcat_s                     REAL,
        km_mM                      REAL,
        replicates                 INTEGER,
        reaction_direction         TEXT NOT NULL DEFAULT 'unspecified',
        reaction_id                TEXT,
        cofactor_species           TEXT,
        cofactor_state             TEXT NOT NULL DEFAULT 'unknown',
        cofactor_source            TEXT NOT NULL DEFAULT 'unknown',
        conditions_json            TEXT NOT NULL DEFAULT '{}',
        condition_key              TEXT NOT NULL,
        needs_curation             INTEGER NOT NULL DEFAULT 0,
        curation_notes             TEXT NOT NULL DEFAULT '[]',
        deprecated                 INTEGER NOT NULL DEFAULT 0,
        deprecation_reason         TEXT,
        notes                      TEXT NOT NULL DEFAULT '',
        created_at                 TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS performance_axes (
        axis_row_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        sequence_sha256  TEXT NOT NULL REFERENCES candidate(sequence_sha256),
        round_id         TEXT,
        axis             TEXT NOT NULL CHECK (axis IN (
                             'substrate_fit',
                             'catalytic_function',
                             'stability_expression_risk')),
        kind             TEXT NOT NULL CHECK (kind IN ('expected', 'observed')),
        direction        TEXT NOT NULL DEFAULT 'unknown',
        value            REAL,
        unit             TEXT,
        basis            TEXT NOT NULL DEFAULT '',
        evidence_json    TEXT NOT NULL DEFAULT '[]',
        recorded_at      TEXT NOT NULL,
        UNIQUE (sequence_sha256, round_id, axis, kind)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS evidence (
        evidence_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        record_id              TEXT REFERENCES experiment_record(record_id),
        sequence_sha256        TEXT REFERENCES candidate(sequence_sha256),
        source_type            TEXT NOT NULL,
        identifier             TEXT NOT NULL,
        locator                TEXT,
        strength               TEXT NOT NULL DEFAULT 'annotation_only',
        source_doi             TEXT,
        source_record_id       TEXT,
        license                TEXT,
        database_version       TEXT,
        retrieved_at           TEXT,
        experiment_activity_id TEXT,
        upstream_sources_json  TEXT NOT NULL DEFAULT '[]',
        extracted_by           TEXT,
        verified_by            TEXT,
        quote                  TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lineage_group (
        group_id                  TEXT NOT NULL,
        record_id                 TEXT NOT NULL REFERENCES experiment_record(record_id),
        key_kind                  TEXT NOT NULL,
        key_values_json           TEXT NOT NULL DEFAULT '[]',
        representative_record_id  TEXT,
        strongest_strength        TEXT,
        n_rows                    INTEGER NOT NULL DEFAULT 1,
        computed_at               TEXT NOT NULL,
        PRIMARY KEY (group_id, record_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_record_round ON experiment_record(round_id)",
    "CREATE INDEX IF NOT EXISTS ix_record_target "
    "ON experiment_record(substrate_target_id)",
    "CREATE INDEX IF NOT EXISTS ix_record_sequence "
    "ON experiment_record(sequence_sha256)",
    "CREATE INDEX IF NOT EXISTS ix_record_condition "
    "ON experiment_record(condition_key)",
    "CREATE INDEX IF NOT EXISTS ix_prediction_round ON prediction(round_id)",
    "CREATE INDEX IF NOT EXISTS ix_lineage_parent ON lineage(parent_sha256)",
    "CREATE INDEX IF NOT EXISTS ix_lineage_variant ON lineage(variant_sha256)",
    "CREATE INDEX IF NOT EXISTS ix_evidence_record ON evidence(record_id)",
    "CREATE INDEX IF NOT EXISTS ix_axes_sequence ON performance_axes(sequence_sha256)",
    # -- enforcement at the storage layer, not only in Python ---------------
    # A prediction belonging to a round whose results have arrived is history.
    """
    CREATE TRIGGER IF NOT EXISTS trg_prediction_frozen_insert
    BEFORE INSERT ON prediction
    FOR EACH ROW
    WHEN (SELECT results_ingested_at FROM experiment_round
          WHERE round_id = NEW.round_id) IS NOT NULL
    BEGIN
        SELECT RAISE(ABORT, 'prediction is frozen: results for this round have already been ingested; a prediction written now cannot measure discovery efficiency');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_prediction_frozen_update
    BEFORE UPDATE ON prediction
    FOR EACH ROW
    WHEN (SELECT results_ingested_at FROM experiment_round
          WHERE round_id = OLD.round_id) IS NOT NULL
    BEGIN
        SELECT RAISE(ABORT, 'prediction is frozen: results for this round have already been ingested; editing it now destroys the only honest measurement of the agent');
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_prediction_frozen_delete
    BEFORE DELETE ON prediction
    FOR EACH ROW
    WHEN (SELECT results_ingested_at FROM experiment_round
          WHERE round_id = OLD.round_id) IS NOT NULL
    BEGIN
        SELECT RAISE(ABORT, 'prediction is frozen: results for this round have already been ingested; deleting one now is indistinguishable from never having made it');
    END
    """,
    # No experiment record is ever deleted. Deprecation is a flag with a reason.
    """
    CREATE TRIGGER IF NOT EXISTS trg_experiment_record_no_delete
    BEFORE DELETE ON experiment_record
    FOR EACH ROW
    BEGIN
        SELECT RAISE(ABORT, 'experiment records are never deleted: negatives and expression failures are the batch; use deprecate_record(record_id, reason)');
    END
    """,
)

_MIGRATIONS: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (1, "initial house-database schema", _MIGRATION_1),
)


# --------------------------------------------------------------------------
# the database
# --------------------------------------------------------------------------


class HouseDB:
    """The project's per-substrate screening database.

    One file holds one project's substrate targets, candidates, mutation
    lineage, frozen predictions and complete experimental batches. It exists
    because the three things that decide whether substrate-directed
    engineering works -- the failures, the pre-registered prediction, and the
    conditions each number was measured under -- are exactly the three things
    public enzyme resources discard.

    Open it with a path (``":memory:"`` is accepted for tests), and it creates
    or migrates itself to :data:`SCHEMA_VERSION`. Opening a file written by a
    *newer* schema raises :class:`SchemaVersionError` rather than writing
    through columns whose meaning has changed.
    """

    def __init__(self, path: str | Path, *, create: bool = True,
                 migrate: bool = True) -> None:
        self.path = str(path)
        if not create and self.path != ":memory:" and not Path(self.path).exists():
            raise UnknownRecordError(f"no house database at {self.path}")
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            # WAL keeps a long-running ingest from blocking a reader in the lab.
            self._conn.execute("PRAGMA journal_mode = WAL")
        if migrate:
            self.migrate()
        else:
            self._check_version_readable()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying connection. Safe to call twice."""
        try:
            self._conn.close()
        except Exception:  # pragma: no cover - defensive
            pass

    def __enter__(self) -> "HouseDB":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- schema ------------------------------------------------------------

    def schema_version(self) -> int:
        """Highest migration applied to this file, or 0 for an empty file.

        Exists so a harness can refuse to run against a file it does not
        understand instead of discovering the mismatch through a wrong answer.
        """
        try:
            row = self._conn.execute(
                "SELECT MAX(version) AS v FROM schema_version"
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row["v"]) if row and row["v"] is not None else 0

    def _check_version_readable(self) -> None:
        current = self.schema_version()
        if current > SCHEMA_VERSION:
            raise SchemaVersionError(
                f"{self.path} was written by schema version {current}; this build "
                f"understands up to {SCHEMA_VERSION}. Refusing to open it: a "
                f"column added later may mean something this code would "
                f"misinterpret."
            )

    def migrate(self) -> int:
        """Apply every migration this build knows and return the new version.

        Idempotent. Each migration runs inside one transaction with its
        ``schema_version`` row, so a half-applied schema cannot be left behind
        by an interrupted upgrade.
        """
        self._check_version_readable()
        current = self.schema_version()
        for version, description, statements in _MIGRATIONS:
            if version <= current:
                continue
            with self._conn:
                for sql in statements:
                    self._conn.execute(sql)
                self._conn.execute(
                    "INSERT OR REPLACE INTO schema_version "
                    "(version, applied_at, description) VALUES (?, ?, ?)",
                    (version, _utc_now(), description),
                )
            current = version
        return current

    # -- internal ----------------------------------------------------------

    def _one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        return self._conn.execute(sql, tuple(params)).fetchone()

    def _all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self._conn.execute(sql, tuple(params)).fetchall())

    def _require_round(self, round_id: str) -> sqlite3.Row:
        row = self._one(
            "SELECT * FROM experiment_round WHERE round_id = ?", (round_id,))
        if row is None:
            raise UnknownRecordError(
                f"round {round_id!r} does not exist; create it with create_round()"
            )
        return row

    def _require_candidate(self, sequence_sha256: str) -> sqlite3.Row:
        row = self._one(
            "SELECT * FROM candidate WHERE sequence_sha256 = ?", (sequence_sha256,))
        if row is None:
            raise UnknownRecordError(
                f"candidate {sequence_sha256!r} is not registered; call "
                f"upsert_candidate() first. A stub created here would have no "
                f"sequence and no provenance."
            )
        return row

    def _require_target(self, substrate_target_id: str) -> sqlite3.Row:
        row = self._one(
            "SELECT * FROM substrate_target WHERE substrate_target_id = ?",
            (substrate_target_id,))
        if row is None:
            raise UnknownRecordError(
                f"substrate target {substrate_target_id!r} is not registered"
            )
        return row

    def _assert_predictions_open(self, round_id: str) -> sqlite3.Row:
        """Raise unless this round's predictions may still be written.

        The single checkpoint behind :class:`FrozenPredictionError`. Called
        before any prediction write; the SQL triggers repeat the check so a
        direct ``UPDATE`` cannot slip past it either.
        """
        row = self._require_round(round_id)
        if row["results_ingested_at"]:
            raise FrozenPredictionError(
                f"round {round_id!r}: results were ingested at "
                f"{row['results_ingested_at']}; its predictions are frozen. A "
                f"prediction written or edited after the results cannot measure "
                f"whether the agent improved discovery efficiency, because it "
                f"could have been fitted to the answer. Open a new round "
                f"instead."
            )
        return row

    # -- registration ------------------------------------------------------

    def register_substrate_target(
        self,
        substrate_target_id: str,
        *,
        label: str = "",
        substrate_ladder: Any = None,
        product_ladder: Any = None,
        target_stereochemistry: Stereochemistry | str = Stereochemistry.UNSPECIFIED,
        creates_new_stereocenter: bool | None = None,
        reaction_id: str | None = None,
        reaction_class: str | None = None,
        required_cofactor_species: str | None = None,
        required_cofactor_state: CofactorState | str | None = None,
        needs_curation: bool = False,
        curation_notes: Sequence[str] = (),
    ) -> str:
        """Register one precisely defined substrate and target product.

        A substrate target is the unit everything else hangs from: the same
        enzyme against a second ketone is a different project, not a second row
        in the same campaign. The chemical identity ladder from
        :mod:`eagent.datalayer.identity` is stored whole, so a later reader can
        see whether the molecule was defined by a structure or only by a name.

        A target with no structural rung on either ladder is accepted **only**
        with ``needs_curation=True`` and a note saying what is missing;
        otherwise :class:`~eagent.errors.UnresolvedFieldError` is raised. This
        prevents the failure that poisons a whole campaign: searching very
        efficiently for the wrong molecule because "4-chloroacetophenone" was
        never resolved to a structure.
        """
        sub = _ladder_dict(substrate_ladder)
        prod = _ladder_dict(product_ladder)
        notes = [str(n) for n in curation_notes]
        missing: list[str] = []
        if not _ladder_has_structure(sub):
            missing.append("substrate_ladder.structure")
        if not _ladder_has_structure(prod):
            missing.append("product_ladder.structure")
        if missing and not needs_curation:
            raise UnresolvedFieldError(missing, gate="register_substrate_target")
        if missing:
            notes.append(
                "curator must supply a structural representation (isomeric "
                "SMILES, InChI or molfile) for: " + ", ".join(missing)
            )
        for name, ladder in (("substrate", substrate_ladder),
                             ("product", product_ladder)):
            check = getattr(ladder, "is_consistent", None)
            if callable(check):
                report = check()
                if not getattr(report, "consistent", True):
                    notes.append(
                        f"{name} identity ladder is internally inconsistent "
                        f"({getattr(report, 'describe', lambda: '')()}); a "
                        f"curator must resolve which rung is authoritative"
                    )
                    needs_curation = True
        stereo = _as_enum(Stereochemistry, target_stereochemistry,
                          default=Stereochemistry.UNSPECIFIED,
                          field_name="target_stereochemistry")
        state = _as_enum(CofactorState, required_cofactor_state, default=None,
                         field_name="required_cofactor_state")
        if stereo in (Stereochemistry.R, Stereochemistry.S) \
                and creates_new_stereocenter is None:
            notes.append(
                "target_stereochemistry names a configuration but "
                "creates_new_stereocenter is unset; a curator must confirm "
                "whether this substrate actually produces a new stereocentre"
            )
            needs_curation = True
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO substrate_target (
                    substrate_target_id, label, substrate_ladder_json,
                    product_ladder_json, target_stereochemistry,
                    creates_new_stereocenter, reaction_id, reaction_class,
                    required_cofactor_species, required_cofactor_state,
                    needs_curation, curation_notes, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(substrate_target_id) DO UPDATE SET
                    label = excluded.label,
                    substrate_ladder_json =
                        COALESCE(excluded.substrate_ladder_json,
                                 substrate_target.substrate_ladder_json),
                    product_ladder_json =
                        COALESCE(excluded.product_ladder_json,
                                 substrate_target.product_ladder_json),
                    target_stereochemistry = excluded.target_stereochemistry,
                    creates_new_stereocenter =
                        COALESCE(excluded.creates_new_stereocenter,
                                 substrate_target.creates_new_stereocenter),
                    reaction_id = COALESCE(excluded.reaction_id,
                                           substrate_target.reaction_id),
                    reaction_class = COALESCE(excluded.reaction_class,
                                              substrate_target.reaction_class),
                    required_cofactor_species =
                        COALESCE(excluded.required_cofactor_species,
                                 substrate_target.required_cofactor_species),
                    required_cofactor_state =
                        COALESCE(excluded.required_cofactor_state,
                                 substrate_target.required_cofactor_state),
                    needs_curation = excluded.needs_curation,
                    curation_notes = excluded.curation_notes
                """,
                (
                    substrate_target_id, label,
                    _dumps(sub) if sub is not None else None,
                    _dumps(prod) if prod is not None else None,
                    stereo.value,
                    None if creates_new_stereocenter is None
                    else int(bool(creates_new_stereocenter)),
                    reaction_id, reaction_class, required_cofactor_species,
                    state.value if state is not None else None,
                    int(bool(needs_curation or missing)), _dumps(notes), _utc_now(),
                ),
            )
        return substrate_target_id

    def upsert_candidate(
        self,
        *,
        sequence: str | None = None,
        sequence_sha256: str | None = None,
        construct_sequence: str | None = None,
        candidate_id: str | None = None,
        origin: CandidateOrigin | str = CandidateOrigin.UNKNOWN,
        family: str | None = None,
        family_basis: str = "",
        cluster_id: str | None = None,
        organism: str | None = None,
        accessions: Sequence[Any] = (),
        discovery_method: str | None = None,
        discovery_detail: str = "",
        provenance: Mapping[str, Any] | None = None,
        needs_curation: bool = False,
        curation_notes: Sequence[str] = (),
    ) -> str:
        """Insert or update one enzyme, keyed by the hash of its sequence.

        The sequence hash is the identity, and the *construct* sequence -- the
        thing that was actually expressed, tags and truncations included -- is
        stored beside it, because an engineered variant usually has no
        accession at all and a construct with a His-tag is not the same protein
        as the catalytic domain in a database entry.

        Accessions go to :table:`candidate_accession` as versioned secondary
        attributes rather than into a column here. An accession is a pointer
        that gets re-annotated, merged and withdrawn; treating it as identity
        is how two isoforms become one row.

        Returns the sequence hash. Updating never erases a stored value with
        ``None``: a partial record from a second source adds, it does not
        overwrite with silence.
        """
        if not sequence and not sequence_sha256:
            raise UnresolvedFieldError(
                ["sequence", "sequence_sha256"], gate="upsert_candidate")
        digest = sequence_sha256 or sequence_hash(str(sequence))
        if sequence and sequence_sha256:
            recomputed = sequence_hash(str(sequence))
            if recomputed != sequence_sha256:
                raise FabricationGuardError(
                    f"sequence_sha256 {sequence_sha256} does not hash the supplied "
                    f"sequence ({recomputed}); one of the two is from a different "
                    f"protein and guessing which would corrupt every join"
                )
        origin_e = _as_enum(CandidateOrigin, origin,
                            default=CandidateOrigin.UNKNOWN, field_name="origin")
        construct_sha = sequence_hash(construct_sequence) if construct_sequence else None
        notes = [str(n) for n in curation_notes]
        if origin_e is CandidateOrigin.ENGINEERED and not construct_sequence:
            notes.append(
                "engineered candidate has no construct_sequence; a curator must "
                "supply the sequence that was actually expressed, because no "
                "accession identifies this protein"
            )
            needs_curation = True
        now = _utc_now()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO candidate (
                    sequence_sha256, candidate_id, sequence, construct_sequence,
                    construct_sha256, origin, family, family_basis, cluster_id,
                    organism, discovery_method, discovery_detail, provenance_json,
                    needs_curation, curation_notes, created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(sequence_sha256) DO UPDATE SET
                    candidate_id = COALESCE(excluded.candidate_id,
                                            candidate.candidate_id),
                    sequence = COALESCE(excluded.sequence, candidate.sequence),
                    construct_sequence = COALESCE(excluded.construct_sequence,
                                                  candidate.construct_sequence),
                    construct_sha256 = COALESCE(excluded.construct_sha256,
                                                candidate.construct_sha256),
                    origin = CASE WHEN excluded.origin = 'unknown'
                                  THEN candidate.origin ELSE excluded.origin END,
                    family = COALESCE(excluded.family, candidate.family),
                    family_basis = CASE WHEN excluded.family_basis = ''
                                        THEN candidate.family_basis
                                        ELSE excluded.family_basis END,
                    cluster_id = COALESCE(excluded.cluster_id, candidate.cluster_id),
                    organism = COALESCE(excluded.organism, candidate.organism),
                    discovery_method = COALESCE(excluded.discovery_method,
                                                candidate.discovery_method),
                    discovery_detail = CASE WHEN excluded.discovery_detail = ''
                                            THEN candidate.discovery_detail
                                            ELSE excluded.discovery_detail END,
                    provenance_json = COALESCE(excluded.provenance_json,
                                               candidate.provenance_json),
                    needs_curation = MAX(candidate.needs_curation,
                                         excluded.needs_curation),
                    updated_at = excluded.updated_at
                """,
                (
                    digest, candidate_id, sequence, construct_sequence, construct_sha,
                    origin_e.value, family, family_basis, cluster_id, organism,
                    discovery_method, discovery_detail,
                    _dumps(dict(provenance)) if provenance is not None else None,
                    int(bool(needs_curation)), _dumps(notes), now, now,
                ),
            )
            for acc in accessions:
                db_name, accession, version, retrieved = self._split_accession(acc)
                if not accession:
                    continue
                self._conn.execute(
                    "INSERT OR IGNORE INTO candidate_accession "
                    "(sequence_sha256, database, accession, version, retrieved_at) "
                    "VALUES (?,?,?,?,?)",
                    (digest, db_name, accession, version, retrieved),
                )
        return digest

    @staticmethod
    def _split_accession(acc: Any) -> tuple[str, str, str | None, str | None]:
        """Normalise an accession from a string, mapping or ``AccessionRef``.

        An unversioned accession keeps ``version=None`` rather than being
        given a plausible one, so a later reader can see that the version was
        never recorded.
        """
        if isinstance(acc, str):
            return ("unknown", acc.strip(), None, None)
        database = _get(acc, "database") or _get(acc, "source_database") or "unknown"
        accession = _get(acc, "accession") or _get(acc, "identifier") or ""
        version = _get(acc, "version")
        retrieved = _get(acc, "retrieved_at")
        return (str(database), str(accession).strip(),
                str(version) if version is not None else None,
                str(retrieved) if retrieved is not None else None)

    def add_lineage_edge(
        self,
        *,
        parent_sha256: str,
        variant_sha256: str,
        mutations: Sequence[Any],
        numbering_reference: str,
        generator: str | None = None,
        round_id: str | None = None,
        notes: str = "",
    ) -> int:
        """Record that one candidate was derived from another by these mutations.

        Ancestry is an edge, not a sentence in a notes column, so "every
        descendant of the parent that gained the W110A background" is a query
        rather than a grep. The numbering reference travels with the edge
        because ``W110A`` means two different residues in two numberings, and a
        mutation list without its numbering is unusable.

        Refuses a self-edge, an empty mutation set, and any edge that would
        close a cycle in the lineage graph -- all three are data-entry errors
        that would make ancestry queries silently non-terminating or wrong.
        """
        self._require_candidate(parent_sha256)
        self._require_candidate(variant_sha256)
        if parent_sha256 == variant_sha256:
            raise ValueError("a candidate cannot be its own variant")
        muts = [str(m) for m in mutations if str(m).strip()]
        if not muts:
            raise UnresolvedFieldError(["mutations"], gate="add_lineage_edge")
        if not str(numbering_reference).strip():
            raise UnresolvedFieldError(
                ["numbering_reference"], gate="add_lineage_edge")
        if parent_sha256 in self.descendants(variant_sha256):
            raise ValueError(
                f"adding {parent_sha256} -> {variant_sha256} would close a cycle: "
                f"{parent_sha256} is already a descendant of {variant_sha256}"
            )
        with self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO lineage (parent_sha256, variant_sha256, "
                "mutations_json, numbering_reference, generator, round_id, notes, "
                "created_at) VALUES (?,?,?,?,?,?,?,?)",
                (parent_sha256, variant_sha256, _dumps(muts), numbering_reference,
                 generator, round_id, notes, _utc_now()),
            )
            if cur.lastrowid:
                return int(cur.lastrowid)
        existing = self._one(
            "SELECT edge_id FROM lineage WHERE parent_sha256 = ? AND "
            "variant_sha256 = ? AND mutations_json = ?",
            (parent_sha256, variant_sha256, _dumps(muts)))
        return int(existing["edge_id"]) if existing else 0

    def descendants(self, sequence_sha256: str) -> set[str]:
        """Every candidate reachable downstream through lineage edges."""
        seen: set[str] = set()
        stack = [sequence_sha256]
        while stack:
            node = stack.pop()
            for row in self._all(
                    "SELECT variant_sha256 FROM lineage WHERE parent_sha256 = ?",
                    (node,)):
                child = row["variant_sha256"]
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return seen

    def ancestors(self, sequence_sha256: str) -> list[str]:
        """Chain of parents from the immediate one upward, nearest first."""
        out: list[str] = []
        seen = {sequence_sha256}
        node = sequence_sha256
        while True:
            row = self._one(
                "SELECT parent_sha256 FROM lineage WHERE variant_sha256 = ? "
                "ORDER BY edge_id LIMIT 1", (node,))
            if row is None or row["parent_sha256"] in seen:
                return out
            node = row["parent_sha256"]
            seen.add(node)
            out.append(node)

    def create_round(self, round_id: str, substrate_target_id: str, *,
                     round_number: int = 1, plan_id: str | None = None,
                     notes: str = "") -> str:
        """Open a screening round against one substrate target.

        A round is the unit that a prediction is frozen against and a batch is
        ingested into. Keeping it explicit is what makes "predictions came
        first" a checkable fact rather than an assumption.
        """
        self._require_target(substrate_target_id)
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO experiment_round (round_id, "
                "substrate_target_id, round_number, plan_id, notes, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (round_id, substrate_target_id, int(round_number), plan_id, notes,
                 _utc_now()),
            )
        return round_id

    # -- predictions, frozen before the experiment -------------------------

    def freeze_predictions(self, round_id: str, predictions: Sequence[Any],
                           snapshot_id: str, *,
                           replace_reason: str | None = None) -> int:
        """Freeze this round's pre-experiment ranking against a dataset snapshot.

        Everything the agent claimed before seeing a result goes in here: the
        ranking, the scorecard dimensions kept apart, the directional
        stereochemical call, the robustness value, why each slot was spent, the
        uncertainty that was stated, and the id of the frozen input snapshot it
        was all computed from.

        **Nothing in this table may be written once the matching results have
        been ingested.** The check happens here and again in an SQL trigger.
        The reason is the whole point of the database: if a prediction can be
        edited afterwards, then top-k enrichment, rank-of-first-hit and
        stereochemical accuracy stop being measurements of the agent and become
        descriptions of the experiment, and there is no way for a reader to
        tell which one they are looking at.

        Re-freezing *before* results arrive is allowed -- plans change -- but it
        requires ``replace_reason`` and is written to ``prediction_freeze_log``
        so the superseded ranking is not silently lost.

        Returns the number of prediction rows written.
        """
        round_row = self._assert_predictions_open(round_id)
        if not str(snapshot_id or "").strip():
            raise UnresolvedFieldError(["snapshot_id"], gate="freeze_predictions")
        entries = [self._coerce_prediction(p) for p in predictions]
        if not entries:
            raise UnresolvedFieldError(["predictions"], gate="freeze_predictions")
        ranks = [e.predicted_rank for e in entries]
        if len(set(ranks)) != len(ranks):
            raise ValueError(
                "predicted_rank must be unique within a round; duplicate ranks "
                "make top-k membership ambiguous and the enrichment unverifiable"
            )
        hashes = [e.sequence_sha256 for e in entries]
        if len(set(hashes)) != len(hashes):
            raise ValueError("a candidate may hold only one prediction per round")
        for e in entries:
            self._require_candidate(e.sequence_sha256)

        existing = int(self._one(
            "SELECT COUNT(*) AS n FROM prediction WHERE round_id = ?",
            (round_id,))["n"])
        if existing and not replace_reason:
            raise FrozenPredictionError(
                f"round {round_id!r} already holds {existing} frozen predictions. "
                f"Pass replace_reason=... to supersede them before results "
                f"arrive; the superseded set is logged rather than discarded."
            )

        now = _utc_now()
        target_id = round_row["substrate_target_id"]
        with self._conn:
            if existing:
                self._conn.execute(
                    "DELETE FROM prediction WHERE round_id = ?", (round_id,))
            for e in entries:
                payload = {
                    "sequence_sha256": e.sequence_sha256,
                    "predicted_rank": e.predicted_rank,
                    "scorecard": dict(e.scorecard),
                    "stereo_call": e.stereo_call,
                    "predicted_ee_pct": e.predicted_ee_pct,
                    "robustness_G": e.robustness_G,
                    "selection_role": e.selection_role,
                    "selection_reason": e.selection_reason,
                    "uncertainty": e.uncertainty,
                    "snapshot_id": snapshot_id,
                }
                self._conn.execute(
                    """
                    INSERT INTO prediction (
                        prediction_id, round_id, substrate_target_id,
                        sequence_sha256, predicted_rank, scorecard_json,
                        stereo_call, stereo_basis, predicted_ee_pct,
                        calibration_source, robustness_G, selection_role,
                        selection_reason, uncertainty, confidence, snapshot_id,
                        frozen_at, content_sha256, notes
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        e.prediction_id or f"{round_id}:{e.sequence_sha256[-16:]}",
                        round_id, target_id, e.sequence_sha256,
                        int(e.predicted_rank), _dumps(dict(e.scorecard)),
                        e.stereo_call, e.stereo_basis, e.predicted_ee_pct,
                        e.calibration_source, e.robustness_G,
                        str(e.selection_role), e.selection_reason, e.uncertainty,
                        e.confidence, snapshot_id, now,
                        "sha256:" + sha256_text(canonical_json(payload)), e.notes,
                    ),
                )
            self._conn.execute(
                "INSERT INTO prediction_freeze_log (round_id, snapshot_id, n_rows, "
                "n_replaced, reason, frozen_at) VALUES (?,?,?,?,?,?)",
                (round_id, snapshot_id, len(entries), existing,
                 replace_reason or "initial freeze", now),
            )
            self._conn.execute(
                "UPDATE experiment_round SET predictions_frozen_at = ?, "
                "predictions_snapshot_id = ? WHERE round_id = ?",
                (now, snapshot_id, round_id),
            )
        return len(entries)

    @staticmethod
    def _coerce_prediction(obj: Any) -> PredictionEntry:
        """Accept a :class:`PredictionEntry`, a mapping, or a scorecard object.

        Keeps one validation path, so a prediction loaded from YAML is held to
        the same standard as one built in code -- including the refusal of a
        numeric predicted ee without a named calibration source.
        """
        if isinstance(obj, PredictionEntry):
            return obj
        sha = _get(obj, "sequence_sha256") or _get(obj, "sequence_hash")
        if not sha:
            seq = _get(obj, "sequence")
            sha = sequence_hash(str(seq)) if seq else None
        rank = _get(obj, "predicted_rank", _get(obj, "rank"))
        missing = [n for n, v in (("sequence_sha256", sha), ("predicted_rank", rank))
                   if v in (None, "")]
        if missing:
            raise UnresolvedFieldError(missing, gate="freeze_predictions")
        role = _get(obj, "selection_role", _get(obj, "role"))
        if role is None:
            raise UnresolvedFieldError(["selection_role"], gate="freeze_predictions")
        return PredictionEntry(
            sequence_sha256=str(sha),
            predicted_rank=int(rank),
            selection_role=str(_enum_value(role)),
            selection_reason=str(_get(obj, "selection_reason", "") or ""),
            uncertainty=str(_get(obj, "uncertainty", "") or ""),
            scorecard=dict(_get(obj, "scorecard", {}) or {}),
            stereo_call=str(_enum_value(
                _get(obj, "stereo_call", "insufficient_evidence"))),
            stereo_basis=str(_get(obj, "stereo_basis", "") or ""),
            predicted_ee_pct=_as_float(_get(obj, "predicted_ee_pct"),
                                       "predicted_ee_pct"),
            calibration_source=_get(obj, "calibration_source"),
            robustness_G=_as_float(_get(obj, "robustness_G"), "robustness_G"),
            confidence=_get(obj, "confidence") and
            str(_enum_value(_get(obj, "confidence"))),
            prediction_id=_get(obj, "prediction_id"),
            notes=str(_get(obj, "notes", "") or ""),
        )

    def predictions(self, round_id: str) -> list[dict[str, Any]]:
        """The frozen prediction rows for a round, as plain data.

        Read-only by construction: there is no method that edits one. The
        scorecard comes back as the dict it went in as, with its dimensions
        still separate.
        """
        self._require_round(round_id)
        out: list[dict[str, Any]] = []
        for row in self._all(
                "SELECT * FROM prediction WHERE round_id = ? ORDER BY predicted_rank",
                (round_id,)):
            d = dict(row)
            d["scorecard"] = _loads(d.pop("scorecard_json"), {})
            out.append(d)
        return out

    # -- results -----------------------------------------------------------

    @staticmethod
    def _conditions_dict(obj: Any) -> dict[str, Any]:
        """Normalise assay conditions to plain data without inventing fields."""
        if obj is None:
            return {}
        if isinstance(obj, Mapping):
            return {k: _enum_value(v) for k, v in obj.items()}
        dump = getattr(obj, "model_dump", None)
        if callable(dump):
            try:
                return dict(dump(mode="json"))
            except Exception:  # pragma: no cover - defensive
                pass
        return {f: _enum_value(getattr(obj, f, None)) for f in CONDITION_KEY_FIELDS}

    def _coerce_result(self, obj: Any, *, default_round: str | None) -> dict[str, Any]:
        """Flatten one result row into stored columns, refusing impossible tuples.

        Accepts a pydantic :class:`~eagent.schemas.record.ExperimentRecord`, a
        mapping from a CSV loader, or any object with the same attribute names.
        Every refusal below is a combination that cannot be true at once, and
        letting one through would put a fabricated scientific value in the
        table that later trains a model.
        """
        record_id = _get(obj, "record_id")
        if not record_id:
            raise UnresolvedFieldError(["record_id"], gate="ingest_round")
        round_id = _get(obj, "round_id") or default_round
        if not round_id:
            raise UnresolvedFieldError(["round_id"], gate="ingest_round")

        sha = _get(obj, "sequence_sha256")
        if not sha:
            seq = _get(obj, "sequence") or _get(obj, "construct_sequence")
            sha = sequence_hash(str(seq)) if seq else None
        if not sha:
            raise UnresolvedFieldError(
                ["sequence_sha256"], gate="ingest_round")

        outcome = _as_enum(OutcomeClass, _get(obj, "outcome"),
                           default=None, field_name="outcome")
        if outcome is None:
            raise UnresolvedFieldError(["outcome"], gate="ingest_round")

        det = _get(obj, "detection")
        method = _get(det, "method") or _get(obj, "detection_method")
        lod = _as_float(
            _get(det, "limit_of_detection", _get(obj, "limit_of_detection")),
            "limit_of_detection")
        lod_unit = _get(det, "limit_unit") or _get(obj, "limit_unit")
        authentic = _as_bool(_get(det, "authentic_standard",
                                  _get(obj, "authentic_standard")))
        confirms = _as_bool(_get(det, "confirms_product_identity",
                                 _get(obj, "confirms_product_identity")))
        chiral_ok = _as_bool(_get(det, "chiral_method_validated",
                                  _get(obj, "chiral_method_validated")))

        product = _get(obj, "product_observed")
        product_smiles = _get(product, "isomeric_smiles") or _get(obj, "product_smiles")
        product_inchikey = _get(product, "inchikey") or _get(obj, "product_inchikey")

        cof = _get(obj, "cofactor")
        cof_species = (_get(cof, "name") or _get(obj, "cofactor_species")
                       or _get(obj, "cofactor_name"))
        cof_state = _as_enum(
            CofactorState, _get(cof, "state", _get(obj, "cofactor_state")),
            default=CofactorState.UNKNOWN, field_name="cofactor_state")
        cof_source = _as_enum(
            LigandSource, _get(cof, "source", _get(obj, "cofactor_source")),
            default=LigandSource.UNKNOWN, field_name="cofactor_source")

        direction = _as_enum(
            ReactionDirection, _get(obj, "reaction_direction"),
            default=ReactionDirection.UNSPECIFIED, field_name="reaction_direction")

        conditions = self._conditions_dict(_get(obj, "conditions"))
        ee = _as_float(_get(obj, "ee_target_pct"), "ee_target_pct")
        conversion = _as_float(_get(obj, "conversion_pct"), "conversion_pct")
        meas_value = _as_float(_get(obj, "measurement_value"), "measurement_value")
        spec_act = _as_float(_get(obj, "specific_activity"), "specific_activity")
        kcat = _as_float(_get(obj, "kcat_s"), "kcat_s")
        km = _as_float(_get(obj, "km_mM"), "km_mM")
        soluble = _as_bool(_get(obj, "soluble_expression"))
        status = _as_enum(
            ExpressionStatus, _get(obj, "expression_status"),
            default=None, field_name="expression_status")
        if status is None:
            if outcome is OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE:
                status = ExpressionStatus.NOT_DETECTED
            elif soluble is True:
                status = ExpressionStatus.SOLUBLE
            elif soluble is False:
                status = ExpressionStatus.INSOLUBLE
            else:
                status = ExpressionStatus.NOT_ASSESSED

        notes: list[str] = [str(n) for n in (_get(obj, "curation_notes") or [])]
        needs_curation = bool(_get(obj, "needs_curation", False))

        # -- refusals ------------------------------------------------------
        if outcome.is_positive and not confirms:
            raise FabricationGuardError(
                f"{record_id}: outcome 'confirmed_target_product' requires a "
                f"detection method that identifies the product. A cofactor "
                f"absorbance change or a conversion number alone does not "
                f"confirm which molecule was made."
            )
        if outcome is OutcomeClass.NO_TARGET_PRODUCT_DETECTED and lod is None:
            raise UnresolvedFieldError(
                [f"{record_id}.detection.limit_of_detection"], gate="ingest_round")
        if ee is not None and not (-100.0 <= ee <= 100.0):
            raise ValueError(
                f"{record_id}: ee_target_pct {ee} is outside [-100, 100]")
        if ee is not None and chiral_ok is False:
            raise FabricationGuardError(
                f"{record_id}: an ee value was reported from a chiral method that "
                f"is recorded as not validated; an unvalidated separation cannot "
                f"assign a configuration"
            )
        measured_anything = any(v is not None for v in
                                (ee, conversion, meas_value, spec_act, kcat, km))
        if outcome is OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE and measured_anything:
            raise ValueError(
                f"{record_id}: outcome 'expression_or_solubility_failure' cannot "
                f"carry a catalytic measurement; protein that did not express "
                f"cannot have been assayed. Split the construct failure from the "
                f"assay row."
            )
        if outcome is OutcomeClass.NOT_TESTED and measured_anything:
            raise ValueError(
                f"{record_id}: outcome 'not_tested' cannot carry a measurement")
        if meas_value is not None and not _get(obj, "measurement_type"):
            raise UnresolvedFieldError(
                [f"{record_id}.measurement_type"], gate="ingest_round")

        # -- curation notes, never silent ----------------------------------
        if ee is not None and chiral_ok is None:
            notes.append("ee reported but chiral_method_validated is unrecorded; "
                         "a curator must confirm the chiral separation was validated")
            needs_curation = True
        if outcome.informs_catalytic_ability and cof_state is CofactorState.UNKNOWN:
            notes.append("cofactor oxidation state unknown for a catalytic "
                         "measurement; NAD(P)+ and NAD(P)H are different experiments")
            needs_curation = True
        if outcome.informs_catalytic_ability and not cof_species:
            notes.append("cofactor identity unrecorded for a catalytic measurement")
            needs_curation = True
        if direction is ReactionDirection.UNSPECIFIED and outcome.is_experimental:
            notes.append("reaction direction unspecified; an oxidation "
                         "measurement is not reduction evidence")
            needs_curation = True

        return {
            "record_id": str(record_id),
            "round_id": str(round_id),
            "sequence_sha256": str(sha),
            "construct_sequence": _get(obj, "construct_sequence"),
            "construct_description": _get(obj, "construct_description"),
            "outcome": outcome,
            "expression_status": status,
            "soluble_expression": soluble,
            "detection_method": method,
            "limit_of_detection": lod,
            "limit_unit": lod_unit,
            "authentic_standard": authentic,
            "confirms_product_identity": bool(confirms),
            "chiral_method_validated": chiral_ok,
            "product_smiles": product_smiles,
            "product_inchikey": product_inchikey,
            "ee_target_pct": ee,
            "conversion_pct": conversion,
            "measurement_type": _get(obj, "measurement_type"),
            "measurement_value": meas_value,
            "measurement_unit": _get(obj, "measurement_unit"),
            "specific_activity": spec_act,
            "specific_activity_unit": _get(obj, "specific_activity_unit"),
            "kcat_s": kcat,
            "km_mM": km,
            "replicates": _get(obj, "replicates"),
            "reaction_direction": direction,
            "reaction_id": _get(obj, "reaction_id"),
            "cofactor_species": cof_species,
            "cofactor_state": cof_state,
            "cofactor_source": cof_source,
            "conditions": conditions,
            "condition_key": condition_key(conditions, cof_species, cof_state.value),
            "needs_curation": needs_curation,
            "curation_notes": notes,
            "notes": str(_get(obj, "notes", "") or ""),
            "evidence": list(_get(obj, "evidence") or []),
        }

    #: Fields an ingest may not change on an existing, non-deprecated record.
    #: Everything a later reader would quote as the result.
    _IMMUTABLE_RESULT_FIELDS: tuple[str, ...] = (
        "outcome", "ee_target_pct", "conversion_pct", "measurement_value",
        "specific_activity", "kcat_s", "km_mM",
    )

    def _assert_result_not_silently_rewritten(self, row: Mapping[str, Any]) -> None:
        """Refuse an ingest that changes a stored result in place.

        Re-ingesting an unchanged batch is harmless and common, so identical
        values pass. Changing the outcome or any headline number on an
        existing ``record_id`` is refused: it is the delete-and-replace this
        database forbids, performed without a delete. Deprecate the old record
        with a reason and write the correction under a new id, so both the
        original reading and the correction survive.
        """
        existing = self._one(
            "SELECT * FROM experiment_record WHERE record_id = ?",
            (row["record_id"],))
        if existing is None or existing["deprecated"]:
            return
        changed: list[str] = []
        for fieldname in self._IMMUTABLE_RESULT_FIELDS:
            before = existing[fieldname]
            after = row[fieldname]
            after = after.value if isinstance(after, enum.Enum) else after
            if before is None and after is None:
                continue
            if before is None or after is None or before != after:
                changed.append(f"{fieldname}: {before!r} -> {after!r}")
        if changed:
            raise RecordOverwriteError(
                f"{row['record_id']} already holds a result and is not "
                f"deprecated; this ingest would change " + "; ".join(changed)
                + ". Deprecate the existing record with a reason and ingest the "
                  "correction under a new record_id, so the original reading "
                  "and the correction both survive."
            )

    def ingest_round(self, results: Sequence[Any], *,
                     round_id: str | None = None) -> IngestReport:
        """Write a whole screening batch, failures and untested constructs included.

        Every construct that was submitted gets a row: the ones that expressed
        and worked, the ones that expressed and did nothing, the ones that
        never expressed, and the ones that were planned and never measured. A
        batch filtered down to its successes cannot produce an honest hit rate,
        cannot train anything, and cannot be compared with the prediction that
        preceded it.

        The first ingest for a round stamps ``results_ingested_at``, which is
        the moment that round's predictions become permanently read-only.

        Returns an :class:`IngestReport` naming what was written and what a
        curator still has to supply; it does not log quietly.
        """
        rows = [self._coerce_result(r, default_round=round_id) for r in results]
        if not rows:
            raise UnresolvedFieldError(["results"], gate="ingest_round")
        round_ids = {r["round_id"] for r in rows}
        if round_id and round_ids != {round_id}:
            raise ValueError(
                f"rows name rounds {sorted(round_ids)} but round_id={round_id!r} "
                f"was requested; one ingest writes one round"
            )
        if len(round_ids) != 1:
            raise ValueError(
                f"one ingest writes one round, got {sorted(round_ids)}")
        rid = round_ids.pop()
        round_row = self._require_round(rid)
        target_id = round_row["substrate_target_id"]
        for r in rows:
            self._require_candidate(r["sequence_sha256"])
            self._assert_result_not_silently_rewritten(r)

        now = _utc_now()
        with self._conn:
            for r in rows:
                self._conn.execute(
                    """
                    INSERT INTO experiment_record (
                        record_id, round_id, substrate_target_id, sequence_sha256,
                        construct_sequence, construct_description, outcome,
                        expression_status, soluble_expression, detection_method,
                        limit_of_detection, limit_unit, authentic_standard,
                        confirms_product_identity, chiral_method_validated,
                        product_smiles, product_inchikey, ee_target_pct,
                        conversion_pct, measurement_type, measurement_value,
                        measurement_unit, specific_activity,
                        specific_activity_unit, kcat_s, km_mM, replicates,
                        reaction_direction, reaction_id, cofactor_species,
                        cofactor_state, cofactor_source, conditions_json,
                        condition_key, needs_curation, curation_notes,
                        notes, created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                              ?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(record_id) DO UPDATE SET
                        outcome = excluded.outcome,
                        expression_status = excluded.expression_status,
                        soluble_expression = excluded.soluble_expression,
                        detection_method = excluded.detection_method,
                        limit_of_detection = excluded.limit_of_detection,
                        limit_unit = excluded.limit_unit,
                        authentic_standard = excluded.authentic_standard,
                        confirms_product_identity =
                            excluded.confirms_product_identity,
                        chiral_method_validated = excluded.chiral_method_validated,
                        product_smiles = excluded.product_smiles,
                        product_inchikey = excluded.product_inchikey,
                        ee_target_pct = excluded.ee_target_pct,
                        conversion_pct = excluded.conversion_pct,
                        measurement_type = excluded.measurement_type,
                        measurement_value = excluded.measurement_value,
                        measurement_unit = excluded.measurement_unit,
                        specific_activity = excluded.specific_activity,
                        specific_activity_unit = excluded.specific_activity_unit,
                        kcat_s = excluded.kcat_s,
                        km_mM = excluded.km_mM,
                        replicates = excluded.replicates,
                        reaction_direction = excluded.reaction_direction,
                        reaction_id = excluded.reaction_id,
                        cofactor_species = excluded.cofactor_species,
                        cofactor_state = excluded.cofactor_state,
                        cofactor_source = excluded.cofactor_source,
                        conditions_json = excluded.conditions_json,
                        condition_key = excluded.condition_key,
                        needs_curation = excluded.needs_curation,
                        curation_notes = excluded.curation_notes,
                        notes = excluded.notes
                    """,
                    (
                        r["record_id"], rid, target_id, r["sequence_sha256"],
                        r["construct_sequence"], r["construct_description"],
                        r["outcome"].value, r["expression_status"].value,
                        None if r["soluble_expression"] is None
                        else int(r["soluble_expression"]),
                        r["detection_method"], r["limit_of_detection"],
                        r["limit_unit"],
                        None if r["authentic_standard"] is None
                        else int(r["authentic_standard"]),
                        int(r["confirms_product_identity"]),
                        None if r["chiral_method_validated"] is None
                        else int(r["chiral_method_validated"]),
                        r["product_smiles"], r["product_inchikey"],
                        r["ee_target_pct"], r["conversion_pct"],
                        r["measurement_type"], r["measurement_value"],
                        r["measurement_unit"], r["specific_activity"],
                        r["specific_activity_unit"], r["kcat_s"], r["km_mM"],
                        r["replicates"], r["reaction_direction"].value,
                        r["reaction_id"], r["cofactor_species"],
                        r["cofactor_state"].value, r["cofactor_source"].value,
                        _dumps(r["conditions"]), r["condition_key"],
                        int(r["needs_curation"]), _dumps(r["curation_notes"]),
                        r["notes"], now,
                    ),
                )
                self._write_evidence(r["record_id"], r["sequence_sha256"],
                                     r["evidence"])
            if not round_row["results_ingested_at"]:
                self._conn.execute(
                    "UPDATE experiment_round SET results_ingested_at = ? "
                    "WHERE round_id = ?", (now, rid))

        predicted = {row["sequence_sha256"] for row in self._all(
            "SELECT sequence_sha256 FROM prediction WHERE round_id = ?", (rid,))}
        submitted = {r["sequence_sha256"] for r in rows}
        by_outcome: dict[str, int] = {o.value: 0 for o in OutcomeClass}
        curation: list[str] = []
        for r in rows:
            by_outcome[r["outcome"].value] += 1
            for note in r["curation_notes"]:
                curation.append(f"{r['record_id']}: {note}")
        stamp = self._require_round(rid)["results_ingested_at"] or now
        return IngestReport(
            round_id=rid,
            n_written=len(rows),
            by_outcome=by_outcome,
            n_predicted_not_submitted=len(predicted - submitted),
            n_submitted_without_prediction=len(submitted - predicted),
            curation_notes=tuple(curation),
            results_ingested_at=stamp,
        )

    def _write_evidence(self, record_id: str | None, sequence_sha256: str | None,
                        refs: Iterable[Any]) -> None:
        """Store evidence refs so the independent-evidence count stays queryable.

        ``upstream_sources`` is kept verbatim: four databases carrying one
        re-curated measurement are one piece of evidence, and
        :mod:`eagent.datalayer.lineage` can only say so if the re-integration
        trail survives the write.
        """
        for ref in refs or ():
            identifier = _get(ref, "identifier")
            if not identifier:
                continue
            strength = _as_enum(EvidenceStrength, _get(ref, "strength"),
                                default=EvidenceStrength.ANNOTATION_ONLY,
                                field_name="evidence.strength")
            self._conn.execute(
                "INSERT INTO evidence (record_id, sequence_sha256, source_type, "
                "identifier, locator, strength, source_doi, source_record_id, "
                "license, database_version, retrieved_at, experiment_activity_id, "
                "upstream_sources_json, extracted_by, verified_by, quote) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record_id, sequence_sha256,
                    str(_get(ref, "source_type", "unknown")), str(identifier),
                    _get(ref, "locator"), strength.value, _get(ref, "source_doi"),
                    _get(ref, "source_record_id"), _get(ref, "license"),
                    _get(ref, "database_version"), _get(ref, "retrieved_at"),
                    _get(ref, "experiment_activity_id"),
                    _dumps([str(s) for s in (_get(ref, "upstream_sources") or [])]),
                    _get(ref, "extracted_by"), _get(ref, "verified_by"),
                    _get(ref, "quote"),
                ),
            )

    # -- performance axes --------------------------------------------------

    def record_performance_axis(
        self,
        *,
        sequence_sha256: str,
        axis: PerformanceAxis | str,
        direction: EffectDirection | str = EffectDirection.UNKNOWN,
        kind: AxisRecordKind | str = AxisRecordKind.OBSERVED,
        round_id: str | None = None,
        value: float | None = None,
        unit: str | None = None,
        basis: str = "",
        evidence: Sequence[str] = (),
    ) -> int:
        """Record what one mutation did to **one** axis, on its own row.

        The three axes -- substrate fit, catalytic function, stability and
        expression risk -- are written independently and read back
        independently. There is no column, view, property or helper anywhere in
        this module that combines them, and this method will refuse a payload
        that tries to smuggle one in.

        Why: the three are measured by different assays in different units and
        routinely move in opposite directions. A variant that binds the
        substrate better, turns over worse and expresses poorly is the ordinary
        result of a first engineering round, and it is a *useful* result --
        it says which axis to protect next. One combined number would reduce it
        to "slightly worse", and the next round would repeat the mistake.
        """
        self._require_candidate(sequence_sha256)
        axis_e = _as_enum(PerformanceAxis, axis, default=None, field_name="axis")
        if axis_e is None:
            raise UnresolvedFieldError(["axis"], gate="record_performance_axis")
        kind_e = _as_enum(AxisRecordKind, kind, default=AxisRecordKind.OBSERVED,
                          field_name="kind")
        dir_e = _as_enum(EffectDirection, direction,
                         default=EffectDirection.UNKNOWN, field_name="direction")
        ev = [str(e) for e in evidence]
        _reject_combined_score({"basis": basis, "evidence": ev, "unit": unit},
                               "performance_axes")
        if str(basis).strip().lower() in FORBIDDEN_COMBINED_SCORE_NAMES:
            raise CollapsedScoreError(
                f"performance_axes.basis={basis!r}: "
                f"{THERE_IS_NO_COMBINED_MUTATION_SCORE}")
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO performance_axes (sequence_sha256, round_id, axis, "
                "kind, direction, value, unit, basis, evidence_json, recorded_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(sequence_sha256, round_id, axis, kind) DO UPDATE SET "
                "direction = excluded.direction, value = excluded.value, "
                "unit = excluded.unit, basis = excluded.basis, "
                "evidence_json = excluded.evidence_json, "
                "recorded_at = excluded.recorded_at",
                (sequence_sha256, round_id, axis_e.value, kind_e.value,
                 dir_e.value, value, unit, basis, _dumps(ev), _utc_now()),
            )
        return int(cur.lastrowid or 0)

    def performance_axes(self, sequence_sha256: str, *,
                         round_id: str | None = None) -> AxisReadout:
        """The axes recorded for one construct, kept apart.

        Returns an :class:`AxisReadout`, which has per-axis accessors and no
        total. See :data:`THERE_IS_NO_COMBINED_MUTATION_SCORE`.
        """
        sql = "SELECT * FROM performance_axes WHERE sequence_sha256 = ?"
        params: list[Any] = [sequence_sha256]
        if round_id is not None:
            sql += " AND round_id = ?"
            params.append(round_id)
        obs = tuple(
            AxisObservation(
                sequence_sha256=row["sequence_sha256"],
                round_id=row["round_id"],
                axis=PerformanceAxis(row["axis"]),
                kind=AxisRecordKind(row["kind"]),
                direction=EffectDirection(row["direction"]),
                value=row["value"],
                unit=row["unit"],
                basis=row["basis"],
                evidence=tuple(_loads(row["evidence_json"], [])),
                recorded_at=row["recorded_at"],
            )
            for row in self._all(sql + " ORDER BY axis, kind", params)
        )
        return AxisReadout(sequence_sha256=sequence_sha256, observations=obs)

    # -- deprecation, never deletion ---------------------------------------

    def deprecate_record(self, record_id: str, reason: str) -> None:
        """Mark a record untrustworthy, keeping the row and the reason.

        This is the only way to retire an experimental result. The row stays in
        every batch query with ``deprecated=True``, so a reader sees that a
        measurement was made and later distrusted, which is a different fact
        from the measurement never having existed.
        """
        if not str(reason).strip():
            raise UnresolvedFieldError(["reason"], gate="deprecate_record")
        row = self._one("SELECT record_id FROM experiment_record WHERE record_id = ?",
                        (record_id,))
        if row is None:
            raise UnknownRecordError(f"no experiment record {record_id!r}")
        with self._conn:
            self._conn.execute(
                "UPDATE experiment_record SET deprecated = 1, "
                "deprecation_reason = ? WHERE record_id = ?", (reason, record_id))

    def delete_record(self, record_id: str) -> None:
        """Always raises. Experiment records are not deletable.

        Present on purpose: a caller looking for a delete finds this method and
        its reason instead of reaching for raw SQL. Negatives, expression
        failures and untested constructs are the batch -- removing them turns
        a 12 percent hit rate into 100 percent and makes every later
        comparison meaningless. Use :meth:`deprecate_record`.
        """
        raise DeletionRefusedError(
            f"refusing to delete experiment record {record_id!r}. Negatives and "
            f"failures are the most expensive rows in this database and the ones "
            f"a model needs most. Use deprecate_record(record_id, reason), which "
            f"keeps the row and records why it should not be trusted."
        )

    def deprecate_candidate(self, sequence_sha256: str, reason: str) -> None:
        """Flag a candidate as withdrawn, keeping its experimental history."""
        if not str(reason).strip():
            raise UnresolvedFieldError(["reason"], gate="deprecate_candidate")
        self._require_candidate(sequence_sha256)
        with self._conn:
            self._conn.execute(
                "UPDATE candidate SET deprecated = 1, deprecation_reason = ?, "
                "updated_at = ? WHERE sequence_sha256 = ?",
                (reason, _utc_now(), sequence_sha256))

    # -- reading the batch -------------------------------------------------

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> RecordRow:
        """Inflate one stored row, with enums resolved and nothing defaulted."""
        return RecordRow(
            record_id=row["record_id"],
            round_id=row["round_id"],
            substrate_target_id=row["substrate_target_id"],
            sequence_sha256=row["sequence_sha256"],
            outcome=OutcomeClass(row["outcome"]),
            expression_status=ExpressionStatus(row["expression_status"]),
            soluble_expression=None if row["soluble_expression"] is None
            else bool(row["soluble_expression"]),
            reaction_direction=ReactionDirection(row["reaction_direction"]),
            cofactor_species=row["cofactor_species"],
            cofactor_state=CofactorState(row["cofactor_state"]),
            cofactor_source=LigandSource(row["cofactor_source"]),
            detection_method=row["detection_method"],
            limit_of_detection=row["limit_of_detection"],
            limit_unit=row["limit_unit"],
            confirms_product_identity=bool(row["confirms_product_identity"]),
            chiral_method_validated=None if row["chiral_method_validated"] is None
            else bool(row["chiral_method_validated"]),
            authentic_standard=None if row["authentic_standard"] is None
            else bool(row["authentic_standard"]),
            product_smiles=row["product_smiles"],
            product_inchikey=row["product_inchikey"],
            ee_target_pct=row["ee_target_pct"],
            conversion_pct=row["conversion_pct"],
            measurement_type=row["measurement_type"],
            measurement_value=row["measurement_value"],
            measurement_unit=row["measurement_unit"],
            specific_activity=row["specific_activity"],
            specific_activity_unit=row["specific_activity_unit"],
            kcat_s=row["kcat_s"],
            km_mM=row["km_mM"],
            replicates=row["replicates"],
            reaction_id=row["reaction_id"],
            construct_sequence=row["construct_sequence"],
            construct_description=row["construct_description"],
            conditions=_loads(row["conditions_json"], {}),
            condition_key=row["condition_key"],
            deprecated=bool(row["deprecated"]),
            deprecation_reason=row["deprecation_reason"],
            needs_curation=bool(row["needs_curation"]),
            curation_notes=tuple(_loads(row["curation_notes"], [])),
            notes=row["notes"],
            created_at=row["created_at"],
        )

    def batch_outcomes(self, round_id: str, *,
                       include_deprecated: bool = True) -> BatchOutcomes:
        """The complete batch for a round: hits, negatives, failures, untested.

        Nothing is filtered by default, including deprecated rows, because the
        set of constructs that were submitted is a historical fact and the
        denominator of every rate computed from this round. ``include_
        deprecated=False`` exists for a report that must exclude distrusted
        measurements, and even then the deprecated rows remain in the database.
        """
        round_row = self._require_round(round_id)
        sql = "SELECT * FROM experiment_record WHERE round_id = ?"
        if not include_deprecated:
            sql += " AND deprecated = 0"
        rows = tuple(self._row_to_record(r)
                     for r in self._all(sql + " ORDER BY record_id", (round_id,)))
        return BatchOutcomes(
            round_id=round_id,
            substrate_target_id=round_row["substrate_target_id"],
            rows=rows,
            predictions_frozen_at=round_row["predictions_frozen_at"],
            results_ingested_at=round_row["results_ingested_at"],
        )

    def hit_rate(self, round_id: str, *, include_deprecated: bool = True) -> HitRate:
        """Both hit-rate denominators for a round, in one object.

        Returns all-submitted *and* expressed-only together, with the
        expression failures and the unassessed constructs counted beside them,
        so a caller physically cannot quote the flattering denominator without
        the other one being in the same object. The type has no attribute
        called ``hit_rate`` for exactly that reason.
        """
        batch = self.batch_outcomes(round_id, include_deprecated=include_deprecated)
        expressed = sum(1 for r in batch.rows if r.expressed is True)
        failed = sum(1 for r in batch.rows if r.expressed is False)
        unknown = sum(1 for r in batch.rows if r.expressed is None)
        return HitRate(
            round_id=round_id,
            n_submitted=len(batch.rows),
            n_expressed=expressed,
            n_expression_failed=failed,
            n_expression_unknown=unknown,
            n_not_tested=batch.n_not_tested,
            n_hits=sum(1 for r in batch.rows if r.is_hit),
            n_informative=sum(1 for r in batch.rows if r.informs_catalysis),
        )

    # -- variant versus parent, under identical conditions only ------------

    @staticmethod
    def _pair_differences(parent: RecordRow, variant: RecordRow) -> tuple[str, ...]:
        """Every field that must match before a comparison is allowed, and does not.

        Returned rather than reduced to a boolean so a refusal can tell a
        scientist which assay to repeat instead of only that the comparison
        was impossible.
        """
        diffs: list[str] = []
        if parent.substrate_target_id != variant.substrate_target_id:
            diffs.append("substrate_target_id")
        diffs.extend(_condition_differences(parent.conditions, variant.conditions))
        if (parent.cofactor_species or None) != (variant.cofactor_species or None):
            diffs.append("cofactor_species")
        if parent.cofactor_state is not variant.cofactor_state:
            diffs.append("cofactor_state")
        if parent.reaction_direction is not variant.reaction_direction:
            diffs.append("reaction_direction")
        if (parent.measurement_type or None) != (variant.measurement_type or None):
            diffs.append("measurement_type")
        if (parent.measurement_unit or None) != (variant.measurement_unit or None):
            diffs.append("measurement_unit")
        return tuple(dict.fromkeys(diffs))

    def _records_for(self, sequence_sha256: str, *,
                     substrate_target_id: str | None = None,
                     round_id: str | None = None,
                     include_deprecated: bool = False) -> list[RecordRow]:
        sql = "SELECT * FROM experiment_record WHERE sequence_sha256 = ?"
        params: list[Any] = [sequence_sha256]
        if substrate_target_id is not None:
            sql += " AND substrate_target_id = ?"
            params.append(substrate_target_id)
        if round_id is not None:
            sql += " AND round_id = ?"
            params.append(round_id)
        if not include_deprecated:
            sql += " AND deprecated = 0"
        return [self._row_to_record(r)
                for r in self._all(sql + " ORDER BY record_id", params)]

    def variant_vs_parent(self, parent_hash: str, *,
                          substrate_target_id: str | None = None,
                          round_id: str | None = None,
                          include_deprecated: bool = False) -> VariantComparisonReport:
        """Compare each variant with its parent, **only** under identical conditions.

        Identical means every field in :data:`CONDITION_KEY_FIELDS`, plus the
        cofactor identity and state, plus the reaction direction, plus the
        measured endpoint and its unit. Anything else is refused and reported
        as a refusal naming the fields that differ.

        This refusal is the point of the method. The commonest way a variant
        is wrongly declared an improvement is that it was assayed later, in a
        different buffer, at a different substrate loading, or read out as
        conversion rather than initial rate. Such a pair of numbers does not
        measure the mutation, and no amount of care in interpretation can
        recover the comparison afterwards.

        Deltas are reported per endpoint. There is no single "improvement
        score": a variant that gains conversion and loses ee has not simply
        got better.
        """
        self._require_candidate(parent_hash)
        parent_rows = self._records_for(
            parent_hash, substrate_target_id=substrate_target_id,
            round_id=round_id, include_deprecated=include_deprecated)
        edges = self._all(
            "SELECT * FROM lineage WHERE parent_sha256 = ? ORDER BY edge_id",
            (parent_hash,))
        comparisons: list[ComparablePair] = []
        refusals: list[RefusedComparison] = []
        for edge in edges:
            variant = edge["variant_sha256"]
            muts = tuple(_loads(edge["mutations_json"], []))
            variant_rows = self._records_for(
                variant, substrate_target_id=substrate_target_id,
                round_id=round_id, include_deprecated=include_deprecated)
            if not variant_rows:
                refusals.append(RefusedComparison(
                    parent_hash, variant, muts,
                    "variant has no experimental record in scope", (),
                    tuple(r.record_id for r in parent_rows), ()))
                continue
            if not parent_rows:
                refusals.append(RefusedComparison(
                    parent_hash, variant, muts,
                    "parent has no experimental record in scope; a variant "
                    "cannot be shown to be better than an unmeasured parent",
                    (), (), tuple(r.record_id for r in variant_rows)))
                continue
            if all(v.outcome is OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE
                   for v in variant_rows):
                refusals.append(RefusedComparison(
                    parent_hash, variant, muts,
                    "variant did not express; its catalytic performance relative "
                    "to the parent is undetermined, not worse", (),
                    tuple(r.record_id for r in parent_rows),
                    tuple(r.record_id for r in variant_rows)))
                continue
            matched = False
            best: tuple[int, tuple[str, ...], RecordRow, RecordRow] | None = None
            for p in parent_rows:
                for v in variant_rows:
                    diffs = self._pair_differences(p, v)
                    if not diffs and p.condition_key == v.condition_key:
                        matched = True
                        comparisons.append(ComparablePair(
                            parent_sha256=parent_hash,
                            variant_sha256=variant,
                            mutations=muts,
                            condition_key=p.condition_key,
                            substrate_target_id=p.substrate_target_id,
                            measurement_type=p.measurement_type,
                            measurement_unit=p.measurement_unit,
                            parent_record_id=p.record_id,
                            variant_record_id=v.record_id,
                            parent_value=p.measurement_value,
                            variant_value=v.measurement_value,
                            parent_ee_pct=p.ee_target_pct,
                            variant_ee_pct=v.ee_target_pct,
                            parent_conversion_pct=p.conversion_pct,
                            variant_conversion_pct=v.conversion_pct,
                            parent_outcome=p.outcome,
                            variant_outcome=v.outcome,
                        ))
                    elif best is None or len(diffs) < best[0]:
                        best = (len(diffs), diffs, p, v)
            if not matched and best is not None:
                _, diffs, p, v = best
                refusals.append(RefusedComparison(
                    parent_hash, variant, muts,
                    "no parent record shares this variant's assay conditions; "
                    "comparing across them would measure the condition change, "
                    "not the mutation",
                    diffs,
                    tuple(r.record_id for r in parent_rows),
                    tuple(r.record_id for r in variant_rows)))
        return VariantComparisonReport(
            parent_sha256=parent_hash,
            comparisons=tuple(comparisons),
            refusals=tuple(refusals),
        )

    # -- did the agent actually work? --------------------------------------

    #: Strongest-first priority used to pick one representative row per
    #: construct when a round assayed it under several conditions. A construct
    #: that confirmed the target product anywhere in the round is a hit.
    _OUTCOME_PRIORITY: tuple[OutcomeClass, ...] = (
        OutcomeClass.CONFIRMED_TARGET_PRODUCT,
        OutcomeClass.OTHER_PRODUCT_OR_WRONG_CONFIGURATION,
        OutcomeClass.NO_TARGET_PRODUCT_DETECTED,
        OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE,
        OutcomeClass.COMPUTATIONAL_NEGATIVE,
        OutcomeClass.COMPUTATIONAL_FAILURE,
        OutcomeClass.NOT_TESTED,
    )

    def prediction_vs_outcome(self, round_id: str, *,
                              include_deprecated: bool = False
                              ) -> PredictionOutcomeReport:
        """Compare the frozen ranking with what the experiment actually did.

        This is the measurement the database exists for. It answers whether
        the agent found hits earlier than the batch average would have, how
        often its stereochemical call matched the sign of the measured ee, and
        how many constructs it never ranked at all.

        Three guards keep the answer honest:

        * If the predictions were not frozen before the results arrived, the
          report comes back with ``interpretable=False`` and the reason,
          instead of a number that looks like evidence.
        * Rows that cannot inform catalysis -- expression failures, untested
          constructs, computational failures -- are excluded from the rates
          and counted separately. A top-ranked candidate that never expressed
          is a cloning outcome, not a wrong chemical prediction.
        * Constructs submitted without a prediction are counted, because a
          batch that quietly grew after the ranking was frozen would otherwise
          flatter the top-k.
        """
        round_row = self._require_round(round_id)
        preds = {p["sequence_sha256"]: p for p in self.predictions(round_id)}
        batch = self.batch_outcomes(round_id, include_deprecated=include_deprecated)

        by_seq: dict[str, RecordRow] = {}
        order = {o: i for i, o in enumerate(self._OUTCOME_PRIORITY)}
        for r in batch.rows:
            keep = by_seq.get(r.sequence_sha256)
            if keep is None or order[r.outcome] < order[keep.outcome]:
                by_seq[r.sequence_sha256] = r

        rows: list[PredictionOutcomeRow] = []
        for sha in sorted(set(preds) | set(by_seq)):
            p = preds.get(sha)
            rec = by_seq.get(sha)
            rows.append(PredictionOutcomeRow(
                sequence_sha256=sha,
                predicted_rank=int(p["predicted_rank"]) if p else None,
                selection_role=p["selection_role"] if p else None,
                stereo_call=p["stereo_call"] if p else None,
                robustness_G=p["robustness_G"] if p else None,
                uncertainty=p["uncertainty"] if p else None,
                record_id=rec.record_id if rec else None,
                outcome=rec.outcome if rec else None,
                ee_target_pct=rec.ee_target_pct if rec else None,
                informs_catalysis=bool(rec and rec.informs_catalysis),
                had_prediction=p is not None,
                was_submitted=rec is not None,
            ))

        frozen_at = round_row["predictions_frozen_at"]
        ingested_at = round_row["results_ingested_at"]
        reason: str | None = None
        if not preds:
            reason = ("no predictions were frozen for this round, so there is "
                      "nothing to measure the outcome against")
        elif not ingested_at:
            reason = "no results have been ingested for this round yet"
        elif frozen_at and frozen_at > ingested_at:
            reason = (f"predictions were frozen at {frozen_at}, after results "
                      f"arrived at {ingested_at}; this comparison cannot "
                      f"distinguish prediction from hindsight")
        elif not frozen_at:
            reason = ("the round carries no freeze timestamp, so it cannot be "
                      "shown that the ranking preceded the results")

        informative = [r for r in rows if r.was_submitted and r.informs_catalysis]
        return PredictionOutcomeReport(
            round_id=round_id,
            rows=tuple(rows),
            predictions_frozen_at=frozen_at,
            results_ingested_at=ingested_at,
            interpretable=reason is None,
            non_interpretable_reason=reason,
            n_predicted=len(preds),
            n_submitted=len(by_seq),
            n_informative=len(informative),
            n_hits=sum(1 for r in informative if r.is_hit),
            n_excluded_uninformative=sum(
                1 for r in rows if r.was_submitted and not r.informs_catalysis),
            n_submitted_without_prediction=sum(
                1 for r in rows if r.was_submitted and not r.had_prediction),
            n_predicted_not_submitted=sum(
                1 for r in rows if r.had_prediction and not r.was_submitted),
        )

    # -- coverage: the intersection, not the union -------------------------

    def coverage_audit(self, substrate_target_id: str, *,
                       include_deprecated: bool = False) -> CoverageAudit:
        """How many records are usable on all six facets **at the same time**.

        The headline row count of an enzyme database is close to meaningless
        for this project. The number that decides whether a model can be
        trained or a result reproduced is how many records simultaneously have

        1. a defined sequence,
        2. a defined substrate structure,
        3. a cofactor identity *and* oxidation state,
        4. a defined product,
        5. full reaction conditions (:data:`REQUIRED_CONDITION_FIELDS`), and
        6. a quantitative result.

        That is an intersection. Each facet alone is typically satisfied by
        most rows, and the intersection is typically satisfied by very few;
        reporting facet counts, or their union, is how a dataset of fifty
        thousand rows turns out to contain two hundred usable ones. This
        method reports the intersection first and keeps the union beside it
        only so the gap is visible.

        A negative result with a recorded detection limit counts as
        quantitative: it bounds the value, which is exactly what makes a
        negative usable.
        """
        target = self._require_target(substrate_target_id)
        substrate_ladder = _loads(target["substrate_ladder_json"], {})
        product_ladder = _loads(target["product_ladder_json"], {})
        substrate_ok = _ladder_has_structure(substrate_ladder)
        product_defined_at_target = _ladder_has_structure(product_ladder)

        sql = (
            "SELECT r.*, c.sequence AS cand_sequence, "
            "c.construct_sequence AS cand_construct "
            "FROM experiment_record r "
            "JOIN candidate c ON c.sequence_sha256 = r.sequence_sha256 "
            "WHERE r.substrate_target_id = ?"
        )
        if not include_deprecated:
            sql += " AND r.deprecated = 0"
        raw = self._all(sql + " ORDER BY r.record_id", (substrate_target_id,))

        facet_counts = {f: 0 for f in COVERAGE_FACETS}
        missing_counts = {f: 0 for f in COVERAGE_FACETS}
        complete: list[str] = []
        n_union = 0
        for row in raw:
            rec = self._row_to_record(row)
            conditions = rec.conditions or {}
            facets = {
                "defined_sequence": bool(
                    rec.construct_sequence or row["cand_construct"]
                    or row["cand_sequence"]),
                "defined_substrate_structure": substrate_ok,
                "cofactor_identity_and_state": bool(
                    rec.cofactor_species
                    and rec.cofactor_state is not CofactorState.UNKNOWN),
                "defined_product": bool(
                    rec.product_smiles or rec.product_inchikey
                    or product_defined_at_target),
                "full_reaction_conditions": all(
                    conditions.get(f) is not None
                    for f in REQUIRED_CONDITION_FIELDS),
                "quantitative_result": bool(
                    rec.quantitative_value() is not None
                    or (rec.outcome is OutcomeClass.NO_TARGET_PRODUCT_DETECTED
                        and rec.limit_of_detection is not None)),
            }
            for name, ok in facets.items():
                if ok:
                    facet_counts[name] += 1
                else:
                    missing_counts[name] += 1
            if any(facets.values()):
                n_union += 1
            if all(facets.values()):
                complete.append(rec.record_id)

        return CoverageAudit(
            substrate_target_id=substrate_target_id,
            n_records=len(raw),
            facet_counts=facet_counts,
            n_complete=len(complete),
            n_union=n_union,
            missing_counts=missing_counts,
            complete_record_ids=tuple(complete),
        )

    # -- interoperability export -------------------------------------------

    def export_enzymeml_like(self, round_id: str, *,
                             include_deprecated: bool = True) -> dict[str, Any]:
        """Export one round organised along the EnzymeML data model's concepts.

        **This is an interoperability export, not a certified EnzymeML
        document.** No EnzymeML schema was available to validate against in
        the environment this module was written in, so the document declares
        ``is_certified_enzymeml: false`` and leaves ``enzymeml_version`` as
        ``null`` with a curation note. Claiming a version here would be a
        fabricated interoperability guarantee, and a downstream tool would
        trust it.

        What it does give is the round's content arranged the way EnzymeML
        arranges it -- proteins, small molecules, the reaction, and
        measurements carrying their conditions -- so a curator has a short,
        mechanical path to a real EnzymeML document rather than a re-typing
        job.

        Unlike most exports, the failures travel with it: expression failures
        and untested constructs appear as measurements with their outcome
        class, flagged ``excluded_from_kinetics``, because an export that
        silently drops them misrepresents the batch.
        """
        round_row = self._require_round(round_id)
        target = self._require_target(round_row["substrate_target_id"])
        batch = self.batch_outcomes(round_id, include_deprecated=include_deprecated)

        curation = [
            "A curator must confirm which EnzymeML version this maps to and "
            "validate the document against that schema; no EnzymeML schema was "
            "available here, so enzymeml_version is null rather than guessed.",
            "Vessel definitions (type, volume, units) are not recorded by this "
            "database and are null rather than assumed.",
        ]
        if target["needs_curation"]:
            curation.extend(_loads(target["curation_notes"], []))

        proteins: list[dict[str, Any]] = []
        for sha in sorted({r.sequence_sha256 for r in batch.rows}):
            cand = self._one(
                "SELECT * FROM candidate WHERE sequence_sha256 = ?", (sha,))
            if cand is None:  # pragma: no cover - foreign key makes this unreachable
                continue
            accs = self._all(
                "SELECT database, accession, version FROM candidate_accession "
                "WHERE sequence_sha256 = ? ORDER BY accession_row_id", (sha,))
            proteins.append({
                "id": sha,
                "name": cand["candidate_id"] or sha,
                "sequence": cand["construct_sequence"] or cand["sequence"],
                "sequence_is_expressed_construct": bool(cand["construct_sequence"]),
                "organism": cand["organism"],
                "origin": cand["origin"],
                "references": [
                    {"database": a["database"], "accession": a["accession"],
                     "version": a["version"]} for a in accs
                ],
                "ecnumber": None,
                "note": ("ecnumber is null because this database does not assign "
                         "EC numbers; an unverified assignment here would travel "
                         "as a claim"),
            })

        cofactors = sorted({
            (r.cofactor_species, r.cofactor_state.value)
            for r in batch.rows if r.cofactor_species
        })
        small_molecules: list[dict[str, Any]] = [
            {
                "id": f"substrate:{target['substrate_target_id']}",
                "role": "substrate",
                "name": target["label"] or target["substrate_target_id"],
                "identity_ladder": _loads(target["substrate_ladder_json"], None),
            },
            {
                "id": f"product:{target['substrate_target_id']}",
                "role": "product",
                "name": target["label"] or target["substrate_target_id"],
                "target_stereochemistry": target["target_stereochemistry"],
                "identity_ladder": _loads(target["product_ladder_json"], None),
            },
        ]
        small_molecules.extend({
            "id": f"cofactor:{name}:{state}",
            "role": "cofactor",
            "name": name,
            "redox_state": state,
        } for name, state in cofactors)

        measurements: list[dict[str, Any]] = []
        for r in batch.rows:
            measurements.append({
                "id": r.record_id,
                "protein_id": r.sequence_sha256,
                "outcome_class": r.outcome.value,
                "outcome_claim": r.outcome.claim(),
                "excluded_from_kinetics": not r.informs_catalysis,
                "expression_status": r.expression_status.value,
                "deprecated": r.deprecated,
                "deprecation_reason": r.deprecation_reason,
                "reaction_direction": r.reaction_direction.value,
                "conditions": dict(r.conditions),
                "cofactor": {
                    "name": r.cofactor_species,
                    "redox_state": r.cofactor_state.value,
                    "placement_source": r.cofactor_source.value,
                },
                "detection": {
                    "method": r.detection_method,
                    "limit_of_detection": r.limit_of_detection,
                    "limit_unit": r.limit_unit,
                    "confirms_product_identity": r.confirms_product_identity,
                    "chiral_method_validated": r.chiral_method_validated,
                    "authentic_standard": r.authentic_standard,
                },
                "species_data": [
                    {"species_id": f"substrate:{target['substrate_target_id']}",
                     "initial_concentration":
                         (r.conditions or {}).get("substrate_concentration_mM"),
                     "unit": "mM" if (r.conditions or {}).get(
                         "substrate_concentration_mM") is not None else None},
                    {"species_id": f"product:{target['substrate_target_id']}",
                     "conversion_pct": r.conversion_pct,
                     "ee_target_pct_signed": r.ee_target_pct},
                ],
                "measurement": {
                    "type": r.measurement_type,
                    "value": r.measurement_value,
                    "unit": r.measurement_unit,
                    "replicates": r.replicates,
                },
                "kinetics": {"kcat_s": r.kcat_s, "km_mM": r.km_mM},
                "needs_curation": r.needs_curation,
                "curation_notes": list(r.curation_notes),
            })

        return {
            "format": "enzymeml_like_export",
            "format_version": "eagent.datalayer.house_db/1",
            "is_certified_enzymeml": False,
            "enzymeml_version": None,
            "needs_curation": True,
            "curation_notes": curation,
            "disclaimer": (
                "Organised along the EnzymeML data model's concepts for "
                "interoperability. It has NOT been validated against any "
                "EnzymeML schema version and must not be presented as an "
                "EnzymeML document."
            ),
            "round": {
                "round_id": round_id,
                "round_number": round_row["round_number"],
                "substrate_target_id": round_row["substrate_target_id"],
                "predictions_frozen_at": round_row["predictions_frozen_at"],
                "predictions_snapshot_id": round_row["predictions_snapshot_id"],
                "results_ingested_at": round_row["results_ingested_at"],
            },
            "vessels": [],
            "proteins": proteins,
            "small_molecules": small_molecules,
            "reactions": [{
                "id": target["reaction_id"] or f"reaction:{round_id}",
                "reaction_id_external": target["reaction_id"],
                "reaction_class": target["reaction_class"],
                "reversible": None,
                "note": ("reversible is null: this database records the direction "
                         "each assay ran, not a thermodynamic claim about the "
                         "reaction"),
            }],
            "measurements": measurements,
            "batch_completeness": {
                "n_measurements": len(measurements),
                "includes_expression_failures": any(
                    r.outcome is OutcomeClass.EXPRESSION_OR_SOLUBILITY_FAILURE
                    for r in batch.rows),
                "includes_untested": any(
                    r.outcome is OutcomeClass.NOT_TESTED for r in batch.rows),
                "statement": (
                    "Every construct submitted in this round is present, "
                    "including those that failed to express and those never "
                    "measured."
                ),
            },
        }

    # -- independent evidence ----------------------------------------------

    def _evidence_shims(self, record_ids: Sequence[str]) -> list[Any]:
        """Build duck-typed rows for :mod:`eagent.datalayer.lineage`.

        Shims rather than re-imported pydantic records so the independent-
        evidence count can be computed straight from stored columns, without
        the house database depending on being able to reconstruct a valid
        :class:`~eagent.schemas.record.ExperimentRecord` from a partially
        curated row.
        """
        shims: list[Any] = []
        for rid in record_ids:
            row = self._one(
                "SELECT * FROM experiment_record WHERE record_id = ?", (rid,))
            if row is None:
                continue
            target = self._one(
                "SELECT * FROM substrate_target WHERE substrate_target_id = ?",
                (row["substrate_target_id"],))
            ladder = _loads(target["substrate_ladder_json"], {}) if target else {}
            refs = [
                SimpleNamespace(
                    source_type=e["source_type"],
                    identifier=e["identifier"],
                    source_doi=e["source_doi"],
                    experiment_activity_id=e["experiment_activity_id"],
                    strength=EvidenceStrength(e["strength"]),
                    upstream_sources=_loads(e["upstream_sources_json"], []),
                )
                for e in self._all(
                    "SELECT * FROM evidence WHERE record_id = ? ORDER BY evidence_id",
                    (rid,))
            ]
            conditions = _loads(row["conditions_json"], {})
            cond_key = tuple(
                _enum_value(conditions.get(f)) for f in CONDITION_KEY_FIELDS)
            shims.append(SimpleNamespace(
                record_id=row["record_id"],
                sequence_sha256=row["sequence_sha256"],
                parent_sequence_sha256=None,
                accession=None,
                construct_sequence=row["construct_sequence"],
                substrate=SimpleNamespace(
                    inchikey=_ladder_token(ladder, "inchikey"),
                    isomeric_smiles=_ladder_token(ladder, "isomeric_smiles"),
                    name=row["substrate_target_id"],
                ),
                conditions=SimpleNamespace(key=lambda k=cond_key: k),
                cofactor=SimpleNamespace(
                    describe=lambda n=row["cofactor_species"],
                    s=row["cofactor_state"]: f"{n}[{s}]"),
                outcome=OutcomeClass(row["outcome"]),
                reaction_direction=ReactionDirection(row["reaction_direction"]),
                measurement_type=row["measurement_type"],
                measurement_value=row["measurement_value"],
                measurement_unit=row["measurement_unit"],
                conversion_pct=row["conversion_pct"],
                ee_target_pct=row["ee_target_pct"],
                specific_activity=row["specific_activity"],
                specific_activity_unit=row["specific_activity_unit"],
                kcat_s=row["kcat_s"],
                km_mM=row["km_mM"],
                detection=SimpleNamespace(
                    limit_of_detection=row["limit_of_detection"]),
                evidence=refs,
            ))
        return shims

    def independent_evidence(self, *, round_id: str | None = None,
                             record_ids: Sequence[str] | None = None,
                             persist: bool = True) -> LineageGroupSummary:
        """How many *independent* measurements back a set of records.

        Delegates the grouping to :mod:`eagent.datalayer.lineage`, so four rows
        re-curated from one paper count once. The result is written to
        ``lineage_group`` when ``persist`` is set, which is what makes the
        independent-evidence count a query rather than a recomputation.

        If the lineage module is unavailable this returns
        ``available=False`` with the reason, instead of falling back to
        ``len(records)`` -- a row count presented as an evidence count is the
        exact overstatement the lineage module exists to prevent.
        """
        if record_ids is None:
            if round_id is None:
                raise UnresolvedFieldError(
                    ["round_id", "record_ids"], gate="independent_evidence")
            record_ids = [r["record_id"] for r in self._all(
                "SELECT record_id FROM experiment_record WHERE round_id = ? "
                "ORDER BY record_id", (round_id,))]
        ids = list(record_ids)
        if _independent_evidence_groups is None:
            return LineageGroupSummary(
                n_rows=len(ids), n_independent=None, groups=(), available=False,
                unavailable_reason=(
                    "eagent.datalayer.lineage could not be imported; the number "
                    "of independent measurements is unknown and len(records) is "
                    "not a substitute"),
            )
        shims = self._evidence_shims(ids)
        groups = _independent_evidence_groups(shims)
        payload = tuple(
            {
                "group_id": g.group_id,
                "key_kind": g.key_kind,
                "key_values": list(g.key_values),
                "record_ids": list(g.record_ids),
                "representative_record_id": g.representative_record_id,
                "strongest_strength": _enum_value(g.strongest_strength),
                "n_rows": g.n_rows,
            }
            for g in groups
        )
        if persist:
            now = _utc_now()
            with self._conn:
                for g in payload:
                    for rid in g["record_ids"]:
                        self._conn.execute(
                            "INSERT OR REPLACE INTO lineage_group (group_id, "
                            "record_id, key_kind, key_values_json, "
                            "representative_record_id, strongest_strength, "
                            "n_rows, computed_at) VALUES (?,?,?,?,?,?,?,?)",
                            (g["group_id"], rid, g["key_kind"],
                             _dumps(g["key_values"]),
                             g["representative_record_id"],
                             str(g["strongest_strength"]), g["n_rows"], now),
                        )
        return LineageGroupSummary(
            n_rows=len(shims), n_independent=len(payload), groups=payload,
            available=True,
        )

    # -- public read accessors ---------------------------------------------

    def substrate_target(self, substrate_target_id: str) -> dict[str, Any]:
        """One substrate target as plain data, ladders deserialised.

        A read path that does not require the caller to touch sqlite keeps the
        curation flags visible: ``needs_curation`` and its notes come back with
        the row rather than having to be looked up separately and forgotten.
        """
        row = self._require_target(substrate_target_id)
        d = dict(row)
        d["substrate_ladder"] = _loads(d.pop("substrate_ladder_json"), None)
        d["product_ladder"] = _loads(d.pop("product_ladder_json"), None)
        d["curation_notes"] = _loads(d.pop("curation_notes"), [])
        d["needs_curation"] = bool(d["needs_curation"])
        return d

    def candidate(self, sequence_sha256: str) -> dict[str, Any]:
        """One candidate with its accessions attached as secondary attributes.

        The accessions arrive as a list beside the record, never as the
        record's identity, so a caller cannot accidentally join on one.
        """
        row = self._require_candidate(sequence_sha256)
        d = dict(row)
        d["curation_notes"] = _loads(d.pop("curation_notes"), [])
        d["provenance"] = _loads(d.pop("provenance_json"), None)
        d["needs_curation"] = bool(d["needs_curation"])
        d["deprecated"] = bool(d["deprecated"])
        d["accessions"] = [
            {"database": a["database"], "accession": a["accession"],
             "version": a["version"], "retrieved_at": a["retrieved_at"]}
            for a in self._all(
                "SELECT * FROM candidate_accession WHERE sequence_sha256 = ? "
                "ORDER BY accession_row_id", (sequence_sha256,))
        ]
        return d

    def rounds(self, *, substrate_target_id: str | None = None) \
            -> list[dict[str, Any]]:
        """Every round, with its freeze and ingest timestamps side by side.

        Those two timestamps are what makes "the prediction came first" an
        auditable claim, so they are part of the ordinary round listing rather
        than something a reviewer has to go looking for.
        """
        sql = "SELECT * FROM experiment_round"
        params: list[Any] = []
        if substrate_target_id is not None:
            sql += " WHERE substrate_target_id = ?"
            params.append(substrate_target_id)
        return [dict(r) for r in self._all(sql + " ORDER BY round_number, round_id",
                                           params)]

    def freeze_log(self, round_id: str) -> list[dict[str, Any]]:
        """Every freeze of this round's predictions, superseded ones included.

        A ranking replaced before the results arrived is legitimate, but it is
        not invisible: the log is the evidence that the final ranking was not
        the third attempt after a peek at the plate.
        """
        self._require_round(round_id)
        return [dict(r) for r in self._all(
            "SELECT * FROM prediction_freeze_log WHERE round_id = ? "
            "ORDER BY log_id", (round_id,))]

    def evidence_for(self, *, record_id: str | None = None,
                     sequence_sha256: str | None = None) -> list[dict[str, Any]]:
        """Evidence rows for a record or a candidate, with upstream trails intact.

        ``upstream_sources`` comes back as a list, because the question it
        answers -- is this a measurement or a copy of one -- cannot be answered
        from a flattened string.
        """
        if record_id is None and sequence_sha256 is None:
            raise UnresolvedFieldError(
                ["record_id", "sequence_sha256"], gate="evidence_for")
        clauses, params = [], []
        if record_id is not None:
            clauses.append("record_id = ?")
            params.append(record_id)
        if sequence_sha256 is not None:
            clauses.append("sequence_sha256 = ?")
            params.append(sequence_sha256)
        rows = self._all(
            "SELECT * FROM evidence WHERE " + " AND ".join(clauses)
            + " ORDER BY evidence_id", params)
        out = []
        for r in rows:
            d = dict(r)
            d["upstream_sources"] = _loads(d.pop("upstream_sources_json"), [])
            out.append(d)
        return out

    def lineage_edges(self, *, parent_sha256: str | None = None,
                      variant_sha256: str | None = None) -> list[dict[str, Any]]:
        """Lineage edges, with the mutation set and its numbering reference.

        The numbering reference travels with every edge because a mutation
        label without it names two different residues in two numbering
        schemes, which is how a variant gets built at the wrong position.
        """
        clauses, params = [], []
        if parent_sha256 is not None:
            clauses.append("parent_sha256 = ?")
            params.append(parent_sha256)
        if variant_sha256 is not None:
            clauses.append("variant_sha256 = ?")
            params.append(variant_sha256)
        sql = "SELECT * FROM lineage"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        out = []
        for r in self._all(sql + " ORDER BY edge_id", params):
            d = dict(r)
            d["mutations"] = _loads(d.pop("mutations_json"), [])
            out.append(d)
        return out

    # -- misc --------------------------------------------------------------

    def table_names(self) -> list[str]:
        """Tables present in the file, for an integrity check or a schema diff."""
        return [r["name"] for r in self._all(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]

    def column_names(self, table: str) -> list[str]:
        """Columns of one table, used by the test that no combined score exists."""
        if table not in self.table_names():
            raise UnknownRecordError(f"no table {table!r}")
        return [r["name"] for r in self._all(f"PRAGMA table_info({table})")]

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"HouseDB(path={self.path!r}, schema_version={self.schema_version()})"


def _ladder_token(ladder: Mapping[str, Any] | None, key: str) -> str | None:
    """Pull one named representation out of a serialised identity ladder.

    Returns ``None`` when the ladder does not carry it, rather than falling
    back to a different representation: an InChIKey and a SMILES are not
    interchangeable as a join key.
    """
    if not ladder:
        return None
    found: str | None = None

    def walk(node: Any) -> None:
        nonlocal found
        if found is not None:
            return
        if isinstance(node, Mapping):
            for k, v in node.items():
                kl = str(k).strip().lower()
                if kl == key and isinstance(v, str) and v.strip():
                    found = v.strip()
                    return
                if kl in ("rung", "representation") and isinstance(v, str) \
                        and v.strip().lower() == key:
                    sibling = node.get("value")
                    if isinstance(sibling, str) and sibling.strip():
                        found = sibling.strip()
                        return
                walk(v)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(ladder)
    return found
